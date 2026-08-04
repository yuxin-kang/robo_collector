"""Background ZMQ camera packet cache for collector nodes."""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CachedCameraFrame:
    image: Any
    received_monotonic_sec: float
    camera_timestamp_sec: float | None
    sequence: int | None


@dataclass(frozen=True)
class CachedCameraBundle:
    frames: dict[str, CachedCameraFrame]
    received_monotonic_sec: float

    @property
    def images(self) -> dict[str, Any]:
        return {stream: frame.image for stream, frame in self.frames.items()}

    @property
    def identity(self) -> tuple[tuple[str, int | float | None], ...]:
        return tuple(
            (
                stream,
                frame.sequence
                if frame.sequence is not None
                else frame.camera_timestamp_sec,
            )
            for stream, frame in sorted(self.frames.items())
        )


class CameraFrameCache:
    """Background reader for composed ZMQ camera packets."""

    def __init__(
        self,
        host: str,
        port: int,
        streams: Sequence[str],
        logger: Any,
        *,
        max_inter_camera_skew_sec: float = 0.1,
        reconnect_backoff_sec: float = 0.5,
        expected_fps: float | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.streams = tuple(_normalize_streams(streams))
        self._logger = logger
        self.max_inter_camera_skew_sec = float(max_inter_camera_skew_sec)
        self.reconnect_backoff_sec = float(reconnect_backoff_sec)
        self.expected_fps = (
            None if expected_fps is None else float(expected_fps)
        )
        if self.max_inter_camera_skew_sec <= 0:
            raise ValueError("max_inter_camera_skew_sec must be positive")
        if self.reconnect_backoff_sec <= 0:
            raise ValueError("reconnect_backoff_sec must be positive")
        if self.expected_fps is not None and (
            not math.isfinite(self.expected_fps) or self.expected_fps <= 0
        ):
            raise ValueError("expected_fps must be finite and positive")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._bundle: CachedCameraBundle | None = None
        self._last_error = ""

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="robo_collector_camera_reader", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self) -> CachedCameraBundle | None:
        with self._lock:
            return self._bundle

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def update_from_packet(
        self, packet: dict[str, Any], received_monotonic_sec: float | None = None
    ) -> bool:
        received_at = (
            time.monotonic()
            if received_monotonic_sec is None
            else float(received_monotonic_sec)
        )
        if not isinstance(packet, dict):
            self._record_error("camera packet must be a mapping")
            return False
        images = packet.get("images", {})
        timestamps = packet.get("timestamps", {})
        sequences = packet.get("sequences", {})
        if not isinstance(images, dict) or not isinstance(timestamps, dict):
            self._record_error("camera packet images/timestamps must be mappings")
            return False
        if not isinstance(sequences, dict):
            self._record_error("camera packet sequences must be a mapping")
            return False
        if self.expected_fps is not None:
            try:
                metadata_fps = _required_positive_float(
                    packet.get("metadata", {}).get("fps")
                    if isinstance(packet.get("metadata"), dict)
                    else None,
                    "camera packet metadata.fps",
                )
            except ValueError as exc:
                self._record_error(str(exc))
                return False
            if not math.isclose(
                metadata_fps,
                self.expected_fps,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                self._record_error(
                    "camera packet fps "
                    f"{metadata_fps:g} does not match collector fps "
                    f"{self.expected_fps:g}"
                )
                return False
        missing = [stream for stream in self.streams if stream not in images]
        if missing:
            with self._lock:
                self._bundle = None
            self._record_error(
                "camera packet missing required stream(s): " + ",".join(missing)
            )
            return False

        try:
            frames = {
                stream: CachedCameraFrame(
                    image=images[stream],
                    received_monotonic_sec=received_at,
                    camera_timestamp_sec=_required_timestamp_to_sec(
                        timestamps.get(stream), stream
                    ),
                    sequence=_sequence_or_none(sequences.get(stream), stream),
                )
                for stream in self.streams
            }
        except ValueError as exc:
            self._record_error(str(exc))
            return False
        source_timestamps = [
            frame.camera_timestamp_sec
            for frame in frames.values()
            if frame.camera_timestamp_sec is not None
        ]
        if (
            source_timestamps
            and max(source_timestamps) - min(source_timestamps)
            > self.max_inter_camera_skew_sec + 1e-9
        ):
            self._record_error(
                "camera packet inter-stream timestamp skew exceeds "
                f"{self.max_inter_camera_skew_sec:.3f}s"
            )
            return False
        with self._lock:
            if self._bundle is not None and _bundle_is_not_newer(frames, self._bundle.frames):
                self._last_error = "camera packet did not advance stream sequence/timestamp"
                return False
            self._bundle = CachedCameraBundle(
                frames=frames, received_monotonic_sec=received_at
            )
            self._last_error = ""
        return True

    def _run(self) -> None:
        try:
            from robo_collector_camera.client import CameraClient
        except ImportError as exc:
            self._record_error(
                "cannot import CameraClient; run scripts/setup_data_collection_env.sh"
            )
            self._logger.error(str(exc))
            return

        try:
            from robo_collector_camera.client import CameraPacketError
        except ImportError:
            CameraPacketError = ValueError

        while not self._stop.is_set():
            client = None
            try:
                client = CameraClient(self.host, self.port)
                while not self._stop.is_set():
                    try:
                        packet = client.read(timeout_ms=100)
                    except CameraPacketError as exc:
                        self._record_error(f"invalid camera packet skipped: {exc}")
                        continue
                    if packet is None:
                        continue
                    self.update_from_packet(packet)
            except Exception as exc:  # pragma: no cover - hardware/runtime path
                self._record_error(f"camera reader reconnecting after error: {exc}")
                self._logger.error(f"camera reader reconnecting after error: {exc}")
            finally:
                if client is not None:
                    try:
                        client.close()
                    except Exception as exc:  # pragma: no cover - transport cleanup
                        self._record_error(f"camera client close failed: {exc}")
            self._stop.wait(self.reconnect_backoff_sec)

    def _record_error(self, message: str) -> None:
        with self._lock:
            changed = message != self._last_error
            self._last_error = message
        if changed:
            self._logger.warn(message)


def parse_camera_streams(value: Any) -> list[str]:
    if isinstance(value, str):
        return _normalize_streams(value.split(","))
    if isinstance(value, Sequence):
        return _normalize_streams(value)
    return _normalize_streams([str(value)])


def _normalize_streams(values: Sequence[Any]) -> list[str]:
    streams = [str(value).strip() for value in values if str(value).strip()]
    if not streams:
        raise ValueError("at least one camera stream is required")
    duplicates = sorted({stream for stream in streams if streams.count(stream) > 1})
    if duplicates:
        raise ValueError(f"duplicate camera stream(s): {','.join(duplicates)}")
    invalid = [stream for stream in streams if not re.fullmatch(r"[A-Za-z0-9_-]+", stream)]
    if invalid:
        raise ValueError(
            "invalid camera stream name(s): "
            + ",".join(invalid)
            + "; use letters, digits, underscore, or hyphen"
        )
    return streams


def _required_timestamp_to_sec(value: Any, stream: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"camera packet missing valid timestamp for {stream}")
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"camera packet has invalid timestamp for {stream}") from exc
    if not math.isfinite(timestamp):
        raise ValueError(f"camera packet has non-finite timestamp for {stream}")
    return timestamp


def _sequence_or_none(value: Any, stream: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"camera packet has invalid sequence for {stream}")
    if value < 0:
        raise ValueError(f"camera packet has invalid sequence for {stream}")
    return value


def _required_positive_float(value: Any, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return parsed


def _bundle_is_not_newer(
    frames: dict[str, CachedCameraFrame],
    previous: dict[str, CachedCameraFrame],
) -> bool:
    for stream, frame in frames.items():
        old = previous.get(stream)
        if old is None:
            return False
        if frame.sequence is not None and old.sequence is not None:
            if frame.sequence <= old.sequence:
                return True
        elif (
            frame.camera_timestamp_sec is not None
            and old.camera_timestamp_sec is not None
            and frame.camera_timestamp_sec <= old.camera_timestamp_sec
        ):
            return True
    return False
