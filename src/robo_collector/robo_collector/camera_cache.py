"""Background ZMQ camera packet cache for collector nodes."""

from __future__ import annotations

import math
import re
import threading
import time
import queue
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from robo_collector_camera.client import (
    NormalizedCameraFrame,
    NormalizedPacket,
    normalize_packet,
)


@dataclass(frozen=True)
class CachedCameraFrame:
    image: Any
    received_monotonic_sec: float
    camera_timestamp_sec: float | None
    sequence: int | None
    raw_payload: bytes | None = None
    payload_encoding: str | None = None
    timestamp_quality: str | None = None
    device_timestamp: float | None = None
    device_unit: str | None = None
    timestamp_domain: str | None = None
    server_wall_timestamp: float | None = None
    server_monotonic_timestamp: float | None = None
    receive_wall_timestamp: float | None = None
    receive_monotonic_timestamp: float | None = None
    clock_domain: str | None = None

    @property
    def provenance(self) -> dict[str, Any]:
        """Return source provenance in the raw episode field vocabulary."""
        value: dict[str, Any] = {
            "sequence": self.sequence,
            "clock_domain": self.clock_domain or "server_wall",
            "timestamp_quality": self.timestamp_quality
            or "host_after_capture",
        }
        for field_name, value_name in (
            ("device_timestamp", "device_timestamp"),
            ("device_unit", "device_unit"),
            ("timestamp_domain", "timestamp_domain"),
            ("server_wall_timestamp", "server_wall_timestamp"),
            ("server_monotonic_timestamp", "server_monotonic_timestamp"),
            ("receive_wall_timestamp", "receive_wall_timestamp"),
            ("receive_monotonic_timestamp", "receive_monotonic_timestamp"),
        ):
            field_value = getattr(self, value_name)
            if field_value is not None:
                value[field_name] = field_value
        value.setdefault("receive_monotonic_timestamp", self.received_monotonic_sec)
        return value


@dataclass(frozen=True)
class CachedCameraBundle:
    frames: dict[str, CachedCameraFrame]
    received_monotonic_sec: float
    session_id: str

    @property
    def images(self) -> dict[str, Any]:
        return {stream: frame.image for stream, frame in self.frames.items()}

    @property
    def identity(
        self,
    ) -> tuple[str, tuple[tuple[str, int | float | None], ...]]:
        return (
            self.session_id,
            tuple(
                (
                    stream,
                    frame.sequence
                    if frame.sequence is not None
                    else frame.camera_timestamp_sec,
                )
                for stream, frame in sorted(self.frames.items())
            ),
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
        receive_mode: str = "recording",
        callback_queue_size: int = 128,
        decode_images: bool = True,
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
        if receive_mode not in {"preview", "recording"}:
            raise ValueError("receive_mode must be 'preview' or 'recording'")
        self.receive_mode = receive_mode
        self.callback_queue_size = int(callback_queue_size)
        self.decode_images = bool(decode_images)
        if self.callback_queue_size <= 0:
            raise ValueError("callback_queue_size must be positive")
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
        self._current_session_id = ""
        self._retired_session_ids: set[str] = set()
        self._retired_session_order: list[str] = []
        self._last_packet_sequence: int | None = None
        self._last_error = ""
        self._packet_callback: Any = None
        self._callback_queue: queue.Queue[Any] = queue.Queue(maxsize=self.callback_queue_size)
        self._callback_thread: threading.Thread | None = None
        self._skew_samples: list[float] = []
        self._skew_sample_cursor = 0
        self._stats = {
            "producer_gap": 0,
            "publisher_gap": 0,
            "transport_gap": 0,
            "unattributed_gap": 0,
            "selection_gap": 0,
            "duplicate": 0,
            "reorder": 0,
            "session_restart": 0,
            "stale": 0,
            "expired": 0,
            "queue_depth": 0,
            "queue_capacity": self.callback_queue_size,
            "queue_overflow": 0,
            "callback_errors": 0,
            "recording_queue_overflow": 0,
            "recording_failed": False,
        }
        self._callback_overflow = 0
        self._callback_errors = 0
        self._last_callback_error = ""
        self._recording_failed = False
        self._stream_stats = {
            stream: {"previous_sequence": None, "current_sequence": None,
                     "producer_gap": 0, "publisher_gap": 0,
                     "transport_gap": 0, "unattributed_gap": 0,
                     "selection_gap": 0,
                     "duplicate": 0, "reorder": 0, "stale": 0, "expired": 0,
                     "session_restart": 0, "generation": 0,
                     "session_id": None, "sequence_gap": 0}
            for stream in self.streams
        }
        self._generation = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="robo_collector_camera_reader", daemon=True
        )
        self._thread.start()
        self._callback_thread = threading.Thread(
            target=self._callback_run, name="robo_collector_camera_callback", daemon=True
        )
        self._callback_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._callback_thread is not None:
            self._callback_thread.join(timeout=2.0)

    def flush_callbacks(self, timeout: float | None = None) -> bool:
        """Wait for callback work already accepted by the bounded queue.

        The reader thread never waits for durable raw I/O.  Lifecycle owners
        use this method at STOP/DISCARD/shutdown boundaries to establish a
        deterministic cut-off before changing the episode state.  A timeout
        returns ``False``; queued packets are not silently discarded.
        """
        if timeout is not None:
            timeout = float(timeout)
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError(
                    "callback flush timeout must be finite and non-negative"
                )
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._callback_queue.all_tasks_done:
            while self._callback_queue.unfinished_tasks:
                if deadline is None:
                    self._callback_queue.all_tasks_done.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._callback_queue.all_tasks_done.wait(remaining)
        return True

    def latest(self) -> CachedCameraBundle | None:
        with self._lock:
            return self._bundle

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._stats)
            result["queue_depth"] = self._callback_queue.qsize()
            return result

    @property
    def counters(self) -> dict[str, int]:
        return self.stats

    def reset_episode_window(self) -> None:
        """Start a fresh per-episode observation window.

        Camera packets received while idle must not contribute gaps or skew to
        the next episode.  Clearing the continuity baseline also prevents the
        first packet after START from attributing idle-time sequence jumps to
        the new episode.
        """
        # Packets already accepted before START belong to the previous
        # observation window. Drop those callback work items explicitly;
        # callbacks that race after this point are serialized by the
        # collector lifecycle lock and belong to the new window.
        while True:
            try:
                self._callback_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._callback_queue.task_done()
        with self._lock:
            self._bundle = None
            self._current_session_id = ""
            self._last_packet_sequence = None
            self._retired_session_ids.clear()
            self._retired_session_order.clear()
            self._skew_samples.clear()
            self._skew_sample_cursor = 0
            for key in self._stats:
                self._stats[key] = self.callback_queue_size if key == "queue_capacity" else 0
            self._stats["recording_failed"] = False
            self._callback_overflow = 0
            self._callback_errors = 0
            self._last_callback_error = ""
            self._recording_failed = False
            self._generation += 1
            for stream in self.streams:
                self._stream_stats[stream] = {
                    "previous_sequence": None, "current_sequence": None,
                    "producer_gap": 0, "publisher_gap": 0,
                    "transport_gap": 0, "unattributed_gap": 0,
                    "selection_gap": 0,
                    "duplicate": 0, "reorder": 0, "stale": 0, "expired": 0,
                    "session_restart": 0, "generation": self._generation,
                    "session_id": None, "sequence_gap": 0,
                }

    def reset(self) -> None:
        """Reset the current episode window and advance its generation."""
        self.reset_episode_window()

    def snapshot(self) -> dict[str, Any]:
        """Return a thread-safe, immutable-in-practice episode audit snapshot."""
        with self._lock:
            self._stats["queue_depth"] = self._callback_queue.qsize()
            return {
                "generation": self._generation,
                "session_id": self._current_session_id or None,
                "latest": self._bundle,
                "stats": dict(self._stats),
                "streams": {name: dict(values) for name, values in self._stream_stats.items()},
                "queue": self._queue_metrics_locked(),
                "recording_failed": self._recording_failed,
                "last_callback_error": self._last_callback_error or None,
            }

    @property
    def quality_statistics(self) -> dict[str, Any]:
        """Return bounded camera skew statistics for the current cache session."""
        with self._lock:
            samples = sorted(self._skew_samples)
            queue_metrics = self._queue_metrics_locked()
            callback_error = self._last_callback_error or None
            recording_failed = self._recording_failed
            stale_bundles = self._stats["stale"]
            expired_bundles = self._stats["expired"]
            gap_metrics = {
                "producer_gaps": self._stats["producer_gap"],
                "publisher_gaps": self._stats["publisher_gap"],
                "transport_gaps": self._stats["transport_gap"],
                "unattributed_gaps": self._stats["unattributed_gap"],
            }
            stream_metrics = {
                name: dict(values) for name, values in self._stream_stats.items()
            }
        if not samples:
            return {
                "camera_camera_skew_sec": None,
                "camera_camera_skew": {
                    "p50": None,
                    "p95": None,
                    "p99": None,
                    "max": None,
                },
                "camera_camera_skew_samples": 0,
                "queue": queue_metrics,
                "queue_depth": queue_metrics["depth"],
                "queue_capacity": queue_metrics["capacity"],
                "queue_overflow": queue_metrics["overflow"],
                "recording_queue_overflow": queue_metrics["recording_overflow"],
                "callback_errors": queue_metrics["callback_errors"],
                "last_callback_error": callback_error,
                "recording_failed": recording_failed,
                "stale_bundles": stale_bundles,
                "expired_bundles": expired_bundles,
                **gap_metrics,
                "streams": stream_metrics,
            }

        def percentile(fraction: float) -> float:
            index = min(len(samples) - 1, int(round((len(samples) - 1) * fraction)))
            return samples[index]

        values = {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": samples[-1],
        }
        return {
            "camera_camera_skew_sec": samples[-1],
            "camera_camera_skew": values,
            "camera_camera_skew_samples": len(samples),
            "queue": queue_metrics,
            "queue_depth": queue_metrics["depth"],
            "queue_capacity": queue_metrics["capacity"],
            "queue_overflow": queue_metrics["overflow"],
            "recording_queue_overflow": queue_metrics["recording_overflow"],
            "callback_errors": queue_metrics["callback_errors"],
            "last_callback_error": callback_error,
            "recording_failed": recording_failed,
            "stale_bundles": stale_bundles,
            "expired_bundles": expired_bundles,
            **gap_metrics,
            "streams": stream_metrics,
        }

    def _queue_metrics_locked(self) -> dict[str, Any]:
        overflow = self._callback_overflow
        return {
            "depth": self._callback_queue.qsize(),
            "capacity": self.callback_queue_size,
            "overflow": overflow,
            "recording_overflow": overflow if self.receive_mode == "recording" else 0,
            "callback_errors": self._callback_errors,
            "last_callback_error": self._last_callback_error or None,
        }

    def record_selection_gap(self, count: int = 1, stream: str | None = None) -> None:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("selection gap count must be a non-negative integer")
        with self._lock:
            if stream is not None:
                if stream not in self._stream_stats:
                    raise ValueError(f"unknown camera stream {stream!r}")
            self._stats["selection_gap"] += count
            if stream is not None:
                self._stream_stats[stream]["selection_gap"] += count

    def record_stale(self, count: int = 1, stream: str | None = None) -> None:
        """Record bundles that were too old to be selected for recording."""
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("stale bundle count must be a non-negative integer")
        streams = self.streams if stream is None else (stream,)
        with self._lock:
            for name in streams:
                if name not in self._stream_stats:
                    raise ValueError(f"unknown camera stream {name!r}")
            self._stats["stale"] += count
            self._stats["expired"] += count
            for name in streams:
                self._stream_stats[name]["stale"] += count
                self._stream_stats[name]["expired"] += count

    def set_packet_callback(self, callback: Any | None) -> None:
        """Set a best-effort callback for every received normalized packet."""
        if callback is not None and not callable(callback):
            raise TypeError("camera packet callback must be callable or None")
        with self._lock:
            self._packet_callback = callback

    @property
    def callback_overflow(self) -> int:
        with self._lock:
            return self._callback_overflow

    def update_from_packet(
        self,
        packet: Any,
        received_monotonic_sec: float | None = None,
        *,
        decode_images: bool = True,
    ) -> bool:
        received_at = (
            time.monotonic()
            if received_monotonic_sec is None
            else float(received_monotonic_sec)
        )
        try:
            normalized = normalize_packet(
                packet,
                self.streams,
                decode_images=decode_images,
                receive_wall_timestamp=time.time(),
                receive_monotonic_timestamp=received_at,
            )
        except ValueError as exc:
            self._record_error(str(exc))
            return False
        return self._update_normalized_packet(
            normalized, received_at, decode_images=decode_images
        )

    def _update_normalized_packet(
        self,
        normalized: NormalizedPacket,
        received_at: float,
        *,
        decode_images: bool | None = None,
    ) -> bool:
        """Apply a structurally valid packet to the latest-value cache.

        Packet normalization is intentionally separate from latest-value
        selection.  The reader thread can therefore enqueue every valid raw
        packet before this method rejects it as stale, reordered, skewed, or
        otherwise unsuitable for the latest bundle.
        """
        if decode_images is None:
            decode_images = self.decode_images
        session_id = normalized.session_id
        metadata = normalized.metadata
        producer_gaps = normalized.producer_gaps or {}
        publisher_gaps = normalized.publisher_gaps or {}
        if self.expected_fps is not None:
            try:
                metadata_fps = _required_positive_float(
                    metadata.get("fps")
                    if isinstance(metadata, Mapping)
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
        missing = [stream for stream in self.streams if stream not in normalized.frames]
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
                    image=(
                        normalized.frames[stream].decode()
                        if decode_images and normalized.frames[stream].decoded_image is None
                        else (
                            normalized.frames[stream].decoded_image
                            if normalized.frames[stream].decoded_image is not None
                            else normalized.frames[stream].payload
                        )
                    ),
                    received_monotonic_sec=received_at,
                    camera_timestamp_sec=normalized.frames[stream].timestamp_sec,
                    sequence=normalized.frames[stream].sequence,
                    raw_payload=normalized.frames[stream].payload or None,
                    payload_encoding=normalized.frames[stream].payload_encoding,
                    timestamp_quality=normalized.frames[stream].timestamp_quality,
                    device_timestamp=normalized.frames[stream].device_timestamp,
                    device_unit=normalized.frames[stream].device_unit,
                    timestamp_domain=normalized.frames[stream].timestamp_domain,
                    server_wall_timestamp=normalized.frames[stream].server_wall_timestamp,
                    server_monotonic_timestamp=normalized.frames[stream].server_monotonic_timestamp,
                    receive_wall_timestamp=normalized.frames[stream].receive_wall_timestamp,
                    receive_monotonic_timestamp=normalized.frames[stream].receive_monotonic_timestamp,
                    clock_domain=normalized.frames[stream].clock_domain,
                )
                for stream in self.streams
            }
        except ValueError as exc:
            self._record_error(str(exc))
            return False
        domains = {
            frame.clock_domain
            for frame in frames.values()
            if frame.clock_domain
        }
        if all(
            frame.server_wall_timestamp is not None for frame in frames.values()
        ):
            source_timestamps = [
                frame.server_wall_timestamp for frame in frames.values()
            ]
        elif len(domains) > 1:
            # Device clocks are not comparable without a clock mapping.  The
            # packet receive timestamp remains available for local auditing.
            source_timestamps = []
        else:
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
            if (
                self._current_session_id
                and session_id != self._current_session_id
                and session_id in self._retired_session_ids
            ):
                self._last_error = (
                    f"camera packet belongs to retired session {session_id!r}"
                )
                return False
            if (
                self._bundle is not None
                and session_id == self._current_session_id
                and _bundle_is_not_newer(
                    frames,
                    self._bundle.frames,
                    session_id=session_id,
                    previous_session_id=self._bundle.session_id,
                )
            ):
                kind = "duplicate" if _frames_equal(frames, self._bundle.frames) else "reorder"
                self._stats[kind] += 1
                for stream in frames:
                    self._stream_stats[stream][kind] += 1
                    self._stream_stats[stream]["stale"] += 1
                self._last_error = "camera packet did not advance stream sequence/timestamp"
                return False
            same_session = session_id == self._current_session_id
            if self._current_session_id and not same_session:
                self._stats["session_restart"] += 1
                for stream in frames:
                    self._stream_stats[stream]["session_restart"] += 1
            packet_gap: int | None = None
            if same_session and normalized.packet_sequence is not None and self._last_packet_sequence is not None:
                packet_gap = max(
                    0,
                    normalized.packet_sequence - self._last_packet_sequence - 1,
                )
            if same_session and self._bundle is not None:
                for stream, frame in frames.items():
                    old = self._bundle.frames.get(stream)
                    if old is None or frame.sequence is None or old.sequence is None:
                        continue
                    missing = max(0, frame.sequence - old.sequence - 1)
                    producer = min(missing, _gap_count(producer_gaps.get(stream, 0)))
                    remaining = missing - producer
                    publisher = min(
                        remaining,
                        _gap_count(publisher_gaps.get(stream, 0)),
                    )
                    remaining -= publisher
                    transport = min(remaining, packet_gap or 0)
                    unattributed = remaining - transport
                    self._stats["producer_gap"] += producer
                    self._stats["publisher_gap"] += publisher
                    self._stats["transport_gap"] += transport
                    self._stats["unattributed_gap"] += unattributed
                    stream_stats = self._stream_stats[stream]
                    stream_stats["sequence_gap"] += missing
                    stream_stats["producer_gap"] += producer
                    stream_stats["publisher_gap"] += publisher
                    stream_stats["transport_gap"] += transport
                    stream_stats["unattributed_gap"] += unattributed
                    stream_stats["previous_sequence"] = old.sequence
                    stream_stats["current_sequence"] = frame.sequence
            for stream, frame in frames.items():
                stream_stats = self._stream_stats[stream]
                if stream_stats["previous_sequence"] is None or not same_session:
                    stream_stats["previous_sequence"] = None if not same_session else stream_stats["previous_sequence"]
                stream_stats["current_sequence"] = frame.sequence
                stream_stats["session_id"] = session_id
            if self._current_session_id and not same_session:
                self._retire_session(self._current_session_id)
            self._current_session_id = session_id
            self._last_packet_sequence = normalized.packet_sequence
            self._bundle = CachedCameraBundle(
                frames=frames,
                received_monotonic_sec=received_at,
                session_id=session_id,
            )
            if source_timestamps:
                self._record_skew_sample_locked(
                    max(source_timestamps) - min(source_timestamps)
                )
            self._last_error = ""
        return True

    def _record_skew_sample_locked(self, value: float) -> None:
        if len(self._skew_samples) < 4096:
            self._skew_samples.append(value)
            return
        self._skew_samples[self._skew_sample_cursor] = value
        self._skew_sample_cursor = (self._skew_sample_cursor + 1) % len(
            self._skew_samples
        )

    def _retire_session(self, session_id: str) -> None:
        if session_id in self._retired_session_ids:
            return
        self._retired_session_ids.add(session_id)
        self._retired_session_order.append(session_id)
        if len(self._retired_session_order) > 64:
            expired = self._retired_session_order.pop(0)
            self._retired_session_ids.discard(expired)

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
                try:
                    client = CameraClient(
                        self.host,
                        self.port,
                        receive_mode=self.receive_mode,
                    )
                except TypeError as exc:
                    # Keep lightweight duck-typed clients used by tests and
                    # downstream integrations working during the migration.
                    if "receive_mode" not in str(exc):
                        raise
                    client = CameraClient(self.host, self.port)
                while not self._stop.is_set():
                    try:
                        reader = getattr(client, "read_envelope", None)
                        if not callable(reader):
                            reader = client.read
                        packet = reader(timeout_ms=100)
                    except CameraPacketError as exc:
                        self._record_error(f"invalid camera packet skipped: {exc}")
                        continue
                    if packet is None:
                        continue
                    received_at = time.monotonic()
                    try:
                        normalized = normalize_packet(
                            packet,
                            self.streams,
                            decode_images=self.decode_images,
                            receive_wall_timestamp=time.time(),
                            receive_monotonic_timestamp=received_at,
                        )
                    except ValueError as exc:
                        self._record_error(str(exc))
                        continue
                    with self._lock:
                        callback = self._packet_callback
                    if callback is not None:
                        # Raw recording observes the validated packet before
                        # latest-value selection.  A duplicate/reordered or
                        # skewed packet must remain auditable even when it is
                        # rejected by the cache below.
                        self._queue_callback(normalized)
                    self._update_normalized_packet(
                        normalized,
                        received_at,
                        decode_images=self.decode_images,
                    )
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

    def _queue_callback(self, packet: Any) -> None:
        try:
            self._callback_queue.put_nowait(packet)
        except queue.Full:
            with self._lock:
                self._callback_overflow += 1
                self._stats["queue_overflow"] = self._callback_overflow
                if self.receive_mode == "recording":
                    self._recording_failed = True
                    self._stats["recording_queue_overflow"] = self._callback_overflow
                    self._stats["recording_failed"] = True
                    message = "camera recording queue overflow"
                else:
                    message = "camera callback queue overflow"
                self._last_error = message
            self._logger.warn(message)

    def _callback_run(self) -> None:
        while not self._stop.is_set() or not self._callback_queue.empty():
            try:
                packet = self._callback_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            with self._lock:
                callback = self._packet_callback
            if callback is not None:
                try:
                    callback(packet)
                except Exception as exc:
                    with self._lock:
                        self._callback_errors += 1
                        self._stats["callback_errors"] = self._callback_errors
                        self._last_callback_error = str(exc)
                    self._record_error(f"camera packet callback failed: {exc}")
            self._callback_queue.task_done()


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


def _gap_count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _frames_equal(
    frames: dict[str, CachedCameraFrame], previous: dict[str, CachedCameraFrame]
) -> bool:
    return frames.keys() == previous.keys() and all(
        frame.sequence == previous[stream].sequence
        and frame.camera_timestamp_sec == previous[stream].camera_timestamp_sec
        for stream, frame in frames.items()
    )


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
    *,
    session_id: str,
    previous_session_id: str,
) -> bool:
    if session_id != previous_session_id:
        return False
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
