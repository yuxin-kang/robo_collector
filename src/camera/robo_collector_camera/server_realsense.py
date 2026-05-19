from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import msgpack
import numpy as np
import zmq

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - hardware dependency
    rs = None


def encode_jpeg_bgr(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode RGB image as JPEG")
    return buffer.tobytes()


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode depth image as PNG")
    return buffer.tobytes()


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
    ) -> None:
        rs_module = require_realsense()
        self.spec = spec
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.jpeg_quality = int(jpeg_quality)
        self.depth = bool(depth)
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

    def start(self) -> None:
        profile = self.pipeline.start(self.config)
        self._started = True
        self.device_info = get_device_info(profile)
        self._thread = threading.Thread(
            target=self._run,
            name=f"realsense_reader_{self.spec.stream}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._started:
            self.pipeline.stop()
            self._started = False

    def latest(self) -> EncodedFrame | None:
        with self._lock:
            return self._latest

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def _run(self) -> None:
        sequence = 0
        try:
            while not self._stop.is_set():
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                timestamp_sec = time.time()
                color_bgr = np.asanyarray(color_frame.get_data())
                depth_png = None
                if self.depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_png = encode_png(np.asanyarray(depth_frame.get_data()))

                encoded = EncodedFrame(
                    sequence=sequence,
                    timestamp_sec=timestamp_sec,
                    image_jpeg=encode_jpeg_bgr(color_bgr, self.jpeg_quality),
                    depth_png=depth_png,
                )
                with self._lock:
                    self._latest = encoded
                    self._error = ""
                sequence += 1
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
    return parser


def main():
    args = build_argparser().parse_args()

    if args.list_devices:
        for index, device in enumerate(list_devices()):
            print(f"{index}: {device}")
        return

    camera_specs: list[CameraSpec] = list(args.camera)
    if not camera_specs:
        camera_specs = [CameraSpec(stream="ego_view", serial=args.serial)]

    stream_names = [spec.stream for spec in camera_specs]
    if len(stream_names) != len(set(stream_names)):
        raise SystemExit(f"duplicate camera stream name in {stream_names}")

    readers = [
        RealSenseReader(
            spec,
            width=args.width,
            height=args.height,
            fps=args.fps,
            jpeg_quality=args.jpeg_quality,
            depth=args.depth,
        )
        for spec in camera_specs
    ]
    started_readers = []
    try:
        for reader in readers:
            reader.start()
            started_readers.append(reader)
    except Exception:
        for reader in started_readers:
            reader.stop()
        raise

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 20)
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
            for stream, frame in frames_by_stream.items():
                images[stream] = frame.image_jpeg
                timestamps[stream] = frame.timestamp_sec
                last_sequences[stream] = frame.sequence
                if args.depth and frame.depth_png is not None:
                    depth_stream = f"{stream}_depth"
                    images[depth_stream] = frame.depth_png
                    timestamps[depth_stream] = frame.timestamp_sec

            packet: dict[str, Any] = {
                "schema": "robo_collector_camera.v2",
                "timestamps": timestamps,
                "images": images,
                "metadata": {
                    "cameras": cameras_metadata,
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "depth": args.depth,
                    "jpeg_quality": args.jpeg_quality,
                },
            }

            socket.send(msgpack.packb(packet, use_bin_type=True))
            sent += 1

            if args.print_every > 0 and sent % args.print_every == 0:
                now = time.monotonic()
                elapsed = max(now - last_report, 1e-6)
                print(f"Image sending FPS: {args.print_every / elapsed:.2f}")
                last_report = now

    except KeyboardInterrupt:
        print("Stopping camera server...")
    finally:
        for reader in readers:
            reader.stop()
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
