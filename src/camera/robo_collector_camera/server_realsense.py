from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import msgpack
import numpy as np
import zmq

try:
    from .raw_spool import RawSpool, scan_camera_spools
except ImportError:  # pragma: no cover - direct script execution
    from robo_collector_camera.raw_spool import RawSpool, scan_camera_spools

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - hardware dependency
    rs = None


def encode_jpeg_bgr(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise RuntimeError("Failed to encode RGB image as JPEG")
    return buffer.tobytes()


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode depth image as PNG")
    return buffer.tobytes()


def timestamp_domain_name(frame: Any) -> str:
    """Return the RealSense SDK timestamp-domain enum as stable JSON text."""
    try:
        domain = frame.get_frame_timestamp_domain()
    except (AttributeError, TypeError, ValueError):
        return "unknown"
    if domain is None:
        return "unknown"
    name = getattr(domain, "name", None)
    text = str(name if name is not None else domain).strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text or "unknown"


@dataclass(frozen=True)
class CameraSpec:
    stream: str
    serial: str | None


@dataclass(frozen=True)
class EncodedFrame:
    sequence: int
    timestamp_sec: float
    image_jpeg: bytes
    depth_png: bytes | None = None
    server_monotonic_sec: float = 0.0
    device_timestamp_ms: float | None = None
    timestamp_domain: str | None = None
    timestamp_quality: str = "host_after_capture"
    clock_domain: str | None = None
    producer_gap_count: int = 0


def require_realsense() -> Any:
    if rs is None:
        raise RuntimeError(
            "pyrealsense2 is required for the RealSense server; "
            "run src/camera/scripts/setup_camera_env.sh --server first"
        )
    return rs


def parse_camera_spec(value: str) -> CameraSpec:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "--camera must use '<stream>:<serial>', for example 'head:123456'"
        )
    stream, serial = value.split(":", 1)
    stream = stream.strip()
    serial = serial.strip()
    if not stream:
        raise argparse.ArgumentTypeError("camera stream name cannot be empty")
    if not serial:
        raise argparse.ArgumentTypeError("camera serial cannot be empty")
    return CameraSpec(stream=stream, serial=serial)


def get_device_info(pipeline_profile: Any) -> dict[str, str]:
    rs_module = require_realsense()
    device = pipeline_profile.get_device()
    info = {}
    for key in [
        rs_module.camera_info.name,
        rs_module.camera_info.serial_number,
        rs_module.camera_info.firmware_version,
    ]:
        if device.supports(key):
            info[str(key).split(".")[-1]] = device.get_info(key)
    return info


def list_devices() -> list[dict[str, str]]:
    rs_module = require_realsense()
    context = rs_module.context()
    devices = []
    for device in context.query_devices():
        row = {}
        for key in [
            rs_module.camera_info.name,
            rs_module.camera_info.serial_number,
            rs_module.camera_info.firmware_version,
        ]:
            if device.supports(key):
                row[str(key).split(".")[-1]] = device.get_info(key)
        devices.append(row)
    return devices


class RealSenseReader:
    def __init__(
        self,
        spec: CameraSpec,
        *,
        width: int,
        height: int,
        fps: int,
        jpeg_quality: int,
        depth: bool,
        raw_spool: RawSpool | None = None,
        depth_raw_spool: RawSpool | None = None,
    ) -> None:
        rs_module = require_realsense()
        self.spec = spec
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.jpeg_quality = int(jpeg_quality)
        self.depth = bool(depth)
        self.raw_spool = raw_spool
        self.depth_raw_spool = depth_raw_spool
        self.pipeline = rs_module.pipeline()
        self.config = rs_module.config()
        if spec.serial:
            self.config.enable_device(spec.serial)
        self.config.enable_stream(
            rs_module.stream.color,
            self.width,
            self.height,
            rs_module.format.bgr8,
            self.fps,
        )
        if self.depth:
            self.config.enable_stream(
                rs_module.stream.depth,
                self.width,
                self.height,
                rs_module.format.z16,
                self.fps,
            )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: EncodedFrame | None = None
        self._error = ""
        self.device_info: dict[str, str] = {}
        self._started = False
        self._pipeline_stop_succeeded = False

    def start(self) -> None:
        if self.raw_spool is not None:
            self.raw_spool.mark_start()
        if self.depth_raw_spool is not None:
            self.depth_raw_spool.mark_start()
        profile = self.pipeline.start(self.config)
        self._started = True
        self._pipeline_stop_succeeded = False
        self.device_info = get_device_info(profile)
        self._thread = threading.Thread(
            target=self._run,
            name=f"realsense_reader_{self.spec.stream}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        stop_error: Exception | None = None
        if self._started:
            # Stop the blocking SDK call before joining.  Joining first can
            # leave the reader alive while the server closes its raw spool.
            try:
                self.pipeline.stop()
                self._pipeline_stop_succeeded = True
            except Exception as exc:  # pragma: no cover - hardware/runtime path
                stop_error = exc
                self._pipeline_stop_succeeded = False
            finally:
                self._started = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError(
                    f"RealSense reader did not stop for {self.spec.stream}"
                )
        if stop_error is not None:
            raise RuntimeError(
                f"failed to stop RealSense pipeline for {self.spec.stream}: "
                f"{stop_error}"
            ) from stop_error

    @property
    def capture_thread_alive(self) -> bool:
        """Whether the producer thread still owns access to its raw spool."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def capture_stopped_cleanly(self) -> bool:
        """Whether it is safe to publish the source STOP fence."""
        return self._pipeline_stop_succeeded and not self.capture_thread_alive

    def latest(self) -> EncodedFrame | None:
        with self._lock:
            return self._latest

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def _run(self) -> None:
        sequence = 0
        previous_sequence: int | None = None
        try:
            while not self._stop.is_set():
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                server_wall_timestamp = time.time()
                server_monotonic_timestamp = time.monotonic()
                device_timestamp_ms: float | None = None
                try:
                    candidate = float(color_frame.get_timestamp())
                except (AttributeError, TypeError, ValueError):
                    candidate = float("nan")
                if math.isfinite(candidate):
                    device_timestamp_ms = candidate
                timestamp_domain = (
                    timestamp_domain_name(color_frame)
                    if device_timestamp_ms is not None
                    else "unknown"
                )
                timestamp_quality = (
                    "device"
                    if device_timestamp_ms is not None
                    else "host_after_capture"
                )
                clock_domain = (
                    "realsense:"
                    f"{self.device_info.get('serial_number', self.spec.serial)}"
                    if device_timestamp_ms is not None
                    else "server_wall"
                )
                color_bgr = np.asanyarray(color_frame.get_data())
                depth_png = None
                if self.depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_png = encode_png(np.asanyarray(depth_frame.get_data()))
                    if depth_png is None:
                        raise RuntimeError(
                            f"RealSense depth frame unavailable for {self.spec.stream}"
                        )

                try:
                    device_sequence = int(color_frame.get_frame_number())
                except (AttributeError, TypeError, ValueError):
                    device_sequence = sequence
                if device_sequence < 0:
                    device_sequence = sequence
                producer_gap_count = (
                    max(0, device_sequence - previous_sequence - 1)
                    if previous_sequence is not None
                    and device_sequence > previous_sequence
                    else 0
                )
                encoded = EncodedFrame(
                    sequence=device_sequence,
                    timestamp_sec=server_wall_timestamp,
                    server_monotonic_sec=server_monotonic_timestamp,
                    image_jpeg=encode_jpeg_bgr(color_bgr, self.jpeg_quality),
                    depth_png=depth_png,
                    device_timestamp_ms=device_timestamp_ms,
                    timestamp_domain=timestamp_domain,
                    timestamp_quality=timestamp_quality,
                    clock_domain=clock_domain,
                    producer_gap_count=producer_gap_count,
                )
                if self.raw_spool is not None:
                    if not self.raw_spool.append(
                        {
                            "stream": self.spec.stream,
                            "session_id": self.raw_spool.session_id,
                            "serial": self.spec.serial,
                            "sequence": device_sequence,
                            "payload": encoded.image_jpeg,
                            "payload_encoding": "image/jpeg",
                            "device_timestamp": device_timestamp_ms,
                            "device_unit": "ms"
                            if device_timestamp_ms is not None
                            else None,
                            "timestamp_domain": timestamp_domain,
                            "server_wall_timestamp": server_wall_timestamp,
                            "server_monotonic_timestamp": server_monotonic_timestamp,
                            "clock_domain": clock_domain,
                            "timestamp_quality": timestamp_quality,
                            "producer_gap_count": producer_gap_count,
                        }
                    ):
                        raise RuntimeError(
                            f"camera raw spool full for {self.spec.stream}"
                        )
                if self.depth_raw_spool is not None and depth_png is not None:
                    if not self.depth_raw_spool.append(
                        {
                            "stream": f"{self.spec.stream}_depth",
                            "session_id": self.depth_raw_spool.session_id,
                            "serial": self.spec.serial,
                            "sequence": device_sequence,
                            "payload": depth_png,
                            "payload_encoding": "image/png",
                            "device_timestamp": device_timestamp_ms,
                            "device_unit": "ms"
                            if device_timestamp_ms is not None
                            else None,
                            "timestamp_domain": timestamp_domain,
                            "server_wall_timestamp": server_wall_timestamp,
                            "server_monotonic_timestamp": server_monotonic_timestamp,
                            "clock_domain": clock_domain,
                            "timestamp_quality": timestamp_quality,
                            "producer_gap_count": producer_gap_count,
                        }
                    ):
                        raise RuntimeError(
                            f"camera depth raw spool full for {self.spec.stream}"
                        )
                with self._lock:
                    self._latest = encoded
                    self._error = ""
                previous_sequence = device_sequence
                sequence = max(sequence + 1, device_sequence + 1)
        except Exception as exc:  # pragma: no cover - hardware/runtime path
            with self._lock:
                self._error = str(exc)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish one composed RealSense camera packet over ZMQ."
    )
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--camera",
        action="append",
        type=parse_camera_spec,
        default=[],
        metavar="STREAM:SERIAL",
        help=(
            "RealSense stream mapping. Pass once per camera, for example "
            "--camera head:<D405_SERIAL> --camera ego_view:<D435I_SERIAL>."
        ),
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Legacy single-camera serial; used only when --camera is not set.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List connected RealSense devices and exit.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--depth", dest="depth", action="store_true", default=False)
    parser.add_argument("--no-depth", dest="depth", action="store_false")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument(
        "--packet-schema",
        choices=("v3", "v4"),
        default="v3",
        help="Wire packet schema. v3 remains the deployment default; v4 is opt-in.",
    )
    parser.add_argument(
        "--raw-spool-dir",
        default=None,
        help=(
            "Optional camera-side durable raw spool directory "
            "(source_scope=camera_capture)."
        ),
    )
    parser.add_argument(
        "--raw-spool-manifest-checkpoint-records",
        type=int,
        default=30,
        help="Persist camera spool manifest at least every N records.",
    )
    parser.add_argument(
        "--raw-spool-manifest-checkpoint-sec",
        type=float,
        default=1.0,
        help="Persist camera spool manifest at least every N seconds.",
    )
    parser.add_argument(
        "--raw-spool-durability-interval-sec",
        type=float,
        default=0.25,
        help=(
            "Group camera chunk fsyncs for this interval; close always fsyncs "
            "all pending chunks."
        ),
    )
    return parser


def main():
    args = build_argparser().parse_args()

    if args.list_devices:
        for index, device in enumerate(list_devices()):
            print(f"{index}: {device}")
        return

    if args.raw_spool_dir:
        if args.raw_spool_manifest_checkpoint_records <= 0:
            raise SystemExit("--raw-spool-manifest-checkpoint-records must be positive")
        if (
            not math.isfinite(args.raw_spool_manifest_checkpoint_sec)
            or args.raw_spool_manifest_checkpoint_sec < 0
        ):
            raise SystemExit(
                "--raw-spool-manifest-checkpoint-sec must be finite and non-negative"
            )
        if (
            not math.isfinite(args.raw_spool_durability_interval_sec)
            or args.raw_spool_durability_interval_sec < 0
        ):
            raise SystemExit(
                "--raw-spool-durability-interval-sec must be finite and non-negative"
            )
        recovered = scan_camera_spools(args.raw_spool_dir)
        if recovered:
            print(f"Recovered {len(recovered)} interrupted camera spool session(s)")

    camera_specs: list[CameraSpec] = list(args.camera)
    if not camera_specs:
        camera_specs = [CameraSpec(stream="ego_view", serial=args.serial)]

    stream_names = [spec.stream for spec in camera_specs]
    if len(stream_names) != len(set(stream_names)):
        raise SystemExit(f"duplicate camera stream name in {stream_names}")

    session_id = uuid4().hex
    readers = [
        RealSenseReader(
            spec,
            width=args.width,
            height=args.height,
            fps=args.fps,
            jpeg_quality=args.jpeg_quality,
            depth=args.depth,
            raw_spool=(
                RawSpool(
                    Path(args.raw_spool_dir) / session_id,
                    stream=spec.stream,
                    session_id=session_id,
                    strict_records=True,
                    manifest_checkpoint_records=args.raw_spool_manifest_checkpoint_records,
                    manifest_checkpoint_interval_sec=args.raw_spool_manifest_checkpoint_sec,
                    durability_interval_sec=args.raw_spool_durability_interval_sec,
                )
                if args.raw_spool_dir
                else None
            ),
            depth_raw_spool=(
                RawSpool(
                    Path(args.raw_spool_dir) / session_id,
                    stream=f"{spec.stream}_depth",
                    session_id=session_id,
                    strict_records=True,
                    manifest_checkpoint_records=args.raw_spool_manifest_checkpoint_records,
                    manifest_checkpoint_interval_sec=args.raw_spool_manifest_checkpoint_sec,
                    durability_interval_sec=args.raw_spool_durability_interval_sec,
                )
                if args.raw_spool_dir and args.depth
                else None
            ),
        )
        for spec in camera_specs
    ]
    started_readers = []
    try:
        for reader in readers:
            reader.start()
            started_readers.append(reader)
    except Exception as startup_error:
        for reader in started_readers:
            try:
                reader.stop()
            except Exception as cleanup_error:
                print(
                    f"failed to stop reader {reader.spec.stream} after "
                    f"startup error: {cleanup_error}"
                )
        for reader in readers:
            for spool in (reader.raw_spool, reader.depth_raw_spool):
                if spool is None:
                    continue
                try:
                    spool.mark_close_failed(
                        f"camera server startup failed: {startup_error}"
                    )
                except Exception as status_error:
                    print(
                        f"failed to persist startup failure for "
                        f"{reader.spec.stream}: {status_error}"
                    )
        raise

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 128)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://*:{args.port}")

    cameras_metadata = {
        reader.spec.stream: {
            "device_info": reader.device_info,
            "serial": reader.spec.serial,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "depth": args.depth,
            "jpeg_quality": args.jpeg_quality,
        }
        for reader in readers
    }

    print(f"Composed RealSense camera server listening on tcp://*:{args.port}")
    print(f"Cameras: {cameras_metadata}")
    print(
        f"Streams: {','.join(stream_names)} {args.width}x{args.height}@{args.fps}, "
        f"depth={'on' if args.depth else 'off'}"
    )

    sent = 0
    packet_sequence = 0
    last_report = time.monotonic()
    last_sequences = {reader.spec.stream: -1 for reader in readers}

    try:
        while True:
            reader_errors = {
                reader.spec.stream: reader.error for reader in readers if reader.error
            }
            if reader_errors:
                details = ", ".join(
                    f"{stream}: {error}" for stream, error in reader_errors.items()
                )
                raise RuntimeError(f"RealSense reader stopped: {details}")

            latest = {reader.spec.stream: reader.latest() for reader in readers}
            if any(frame is None for frame in latest.values()):
                time.sleep(0.001)
                continue
            frames_by_stream = {
                stream: frame for stream, frame in latest.items() if frame is not None
            }
            if any(
                frame.sequence <= last_sequences[stream]
                for stream, frame in frames_by_stream.items()
            ):
                time.sleep(0.001)
                continue

            images: dict[str, bytes] = {}
            timestamps: dict[str, float] = {}
            sequences: dict[str, int] = {}
            producer_gaps: dict[str, int] = {}
            publisher_gaps: dict[str, int] = {}
            provenance: dict[str, dict[str, Any]] = {}
            for stream, frame in frames_by_stream.items():
                images[stream] = frame.image_jpeg
                timestamps[stream] = frame.timestamp_sec
                sequences[stream] = frame.sequence
                producer_gaps[stream] = frame.producer_gap_count
                published_missing = (
                    max(0, frame.sequence - last_sequences[stream] - 1)
                    if last_sequences[stream] >= 0
                    else 0
                )
                publisher_gaps[stream] = max(
                    0,
                    published_missing
                    - min(published_missing, frame.producer_gap_count),
                )
                provenance[stream] = {
                    "device": frame.device_timestamp_ms,
                    "device_unit": "ms"
                    if frame.device_timestamp_ms is not None
                    else None,
                    "device_clock_domain": frame.clock_domain,
                    "device_timestamp_domain_type": frame.timestamp_domain,
                    "server_wall": frame.timestamp_sec,
                    "server_monotonic": frame.server_monotonic_sec,
                    "timestamp_quality": frame.timestamp_quality,
                }
                last_sequences[stream] = frame.sequence
                if args.depth and frame.depth_png is not None:
                    depth_stream = f"{stream}_depth"
                    images[depth_stream] = frame.depth_png
                    timestamps[depth_stream] = frame.timestamp_sec
                    sequences[depth_stream] = frame.sequence
                    producer_gaps[depth_stream] = frame.producer_gap_count
                    publisher_gaps[depth_stream] = publisher_gaps[stream]
                    provenance[depth_stream] = dict(provenance[stream])

            metadata = {
                "cameras": cameras_metadata,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "depth": args.depth,
                "jpeg_quality": args.jpeg_quality,
                "payload_encoding": "jpeg",
            }
            if args.packet_schema == "v4":
                streams = {}
                for stream, payload in images.items():
                    source = provenance[stream]
                    streams[stream] = {
                        "sequence": sequences[stream],
                        "payload": payload,
                        "payload_encoding": (
                            "image/png" if stream.endswith("_depth") else "image/jpeg"
                        ),
                        "timestamps": {
                            "device": source["device"],
                            "device_unit": source["device_unit"],
                            "device_clock_domain": source["device_clock_domain"],
                            "device_timestamp_domain_type": source[
                                "device_timestamp_domain_type"
                            ],
                            "server_wall": source["server_wall"],
                            "server_monotonic": source["server_monotonic"],
                        },
                        "timestamp_quality": source["timestamp_quality"],
                    }
                packet = {
                    "schema": "robo_collector_camera.v4",
                    "session_id": session_id,
                    "streams": streams,
                    "metadata": metadata,
                    "producer_gaps": producer_gaps,
                    "publisher_gaps": publisher_gaps,
                    "packet_sequence": packet_sequence,
                }
            else:
                packet = {
                    "schema": "robo_collector_camera.v3",
                    "session_id": session_id,
                    "timestamps": timestamps,
                    "sequences": sequences,
                    "images": images,
                    "metadata": {**metadata, "frame_provenance": provenance},
                    "producer_gaps": producer_gaps,
                    "publisher_gaps": publisher_gaps,
                    "packet_sequence": packet_sequence,
                }

            socket.send(msgpack.packb(packet, use_bin_type=True))
            packet_sequence += 1
            for reader in readers:
                if reader.raw_spool is not None:
                    reader.raw_spool.mark_sent()
                if reader.depth_raw_spool is not None:
                    reader.depth_raw_spool.mark_sent()
            sent += 1

            if args.print_every > 0 and sent % args.print_every == 0:
                now = time.monotonic()
                elapsed = max(now - last_report, 1e-6)
                print(f"Image sending FPS: {args.print_every / elapsed:.2f}")
                last_report = now

    except KeyboardInterrupt:
        print("Stopping camera server...")
    finally:
        shutdown_failures: list[str] = []
        stopped_readers: set[int] = set()
        for reader in readers:
            try:
                reader.stop()
            except Exception as exc:
                message = f"failed to stop reader {reader.spec.stream}: {exc}"
                shutdown_failures.append(message)
                print(message)
            # Publish STOP only after both boundaries are proven: the SDK
            # pipeline stopped successfully and the producer thread joined.
            if reader.capture_stopped_cleanly:
                stopped_readers.add(id(reader))
            else:
                failure = RuntimeError(
                    "camera source STOP fence withheld because pipeline stop "
                    "or reader join was not successful"
                )
                for spool in (reader.raw_spool, reader.depth_raw_spool):
                    if spool is None:
                        continue
                    try:
                        spool.mark_close_failed(failure)
                    except Exception as exc:
                        message = (
                            f"failed to persist incomplete source status for "
                            f"{reader.spec.stream}: {exc}"
                        )
                        shutdown_failures.append(message)
                        print(message)
        for reader in readers:
            if id(reader) not in stopped_readers:
                continue
            if reader.raw_spool is not None:
                try:
                    reader.raw_spool.close(reason="process_stop")
                except Exception as exc:
                    message = (
                        f"failed to close raw spool for {reader.spec.stream}: {exc}"
                    )
                    shutdown_failures.append(message)
                    print(message)
                    try:
                        reader.raw_spool.mark_close_failed(exc)
                    except Exception as status_exc:
                        status_message = (
                            f"failed to persist raw spool close failure for "
                            f"{reader.spec.stream}: {status_exc}"
                        )
                        shutdown_failures.append(status_message)
                        print(status_message)
            if reader.depth_raw_spool is not None:
                try:
                    reader.depth_raw_spool.close(reason="process_stop")
                except Exception as exc:
                    message = (
                        f"failed to close depth raw spool for {reader.spec.stream}: "
                        f"{exc}"
                    )
                    shutdown_failures.append(message)
                    print(message)
                    try:
                        reader.depth_raw_spool.mark_close_failed(exc)
                    except Exception as status_exc:
                        status_message = (
                            f"failed to persist depth spool close failure for "
                            f"{reader.spec.stream}: {status_exc}"
                        )
                        shutdown_failures.append(status_message)
                        print(status_message)
        socket.close(linger=0)
        context.term()
        if shutdown_failures:
            raise RuntimeError(
                "camera shutdown failed: " + "; ".join(shutdown_failures)
            )


if __name__ == "__main__":
    main()
