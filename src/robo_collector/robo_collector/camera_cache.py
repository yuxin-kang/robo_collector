"""Background ZMQ camera packet cache for collector nodes."""

from __future__ import annotations

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


@dataclass(frozen=True)
class CachedCameraBundle:
    frames: dict[str, CachedCameraFrame]
    received_monotonic_sec: float

    @property
    def images(self) -> dict[str, Any]:
        return {stream: frame.image for stream, frame in self.frames.items()}


class CameraFrameCache:
    """Background reader for composed ZMQ camera packets."""

    def __init__(
        self, host: str, port: int, streams: Sequence[str], logger: Any
    ) -> None:
        self.host = host
        self.port = int(port)
        self.streams = tuple(_normalize_streams(streams))
        self._logger = logger
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
        images = packet.get("images", {})
        missing = [stream for stream in self.streams if stream not in images]
        if missing:
            with self._lock:
                self._bundle = None
            self._record_error(
                "camera packet missing required stream(s): " + ",".join(missing)
            )
            return False

        timestamps = packet.get("timestamps", {})
        frames = {
            stream: CachedCameraFrame(
                image=images[stream],
                received_monotonic_sec=received_at,
                camera_timestamp_sec=_timestamp_to_sec(timestamps.get(stream)),
            )
            for stream in self.streams
        }
        with self._lock:
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

        client = CameraClient(self.host, self.port)
        try:
            while not self._stop.is_set():
                packet = client.read(timeout_ms=100)
                if packet is None:
                    continue
                self.update_from_packet(packet)
        except Exception as exc:  # pragma: no cover - hardware/runtime path
            self._record_error(f"camera reader stopped: {exc}")
            self._logger.error(f"camera reader stopped: {exc}")
        finally:
            client.close()

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
    return streams


def _timestamp_to_sec(value: Any) -> float | None:
    if value is None:
        return None
    timestamp = float(value)
    if timestamp > 1e12:
        return timestamp / 1e9
    return timestamp
