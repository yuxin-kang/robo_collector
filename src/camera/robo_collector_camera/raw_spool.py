"""Durable camera-side spool for raw payloads.

Records are msgpack values in length/CRC framed files. A damaged or partially
written final frame is discarded on open, making interrupted writes recoverable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import threading
import time
import uuid
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import msgpack

_HEADER = struct.Struct(">4sII")
_MAGIC = b"RSP1"
_MAX_FRAME = 256 * 1024 * 1024
_DEFAULT_MAX_BYTES = 4 * 1024 * 1024 * 1024
_CAMERA_SPOOL_SCHEMA = "robo_collector.camera_spool.v1"
_CAMERA_RECORD_SCHEMA = "robo_collector.camera_spool.record.v1"
_LEGACY_CAMERA_SPOOL_SCHEMA = "robo_collector.raw_episode.v1"
_MANIFEST_LOCKS: dict[Path, threading.Lock] = {}
_MANIFEST_LOCKS_GUARD = threading.Lock()


class _TruncatedSpoolFrame(ValueError):
    """A frame ended mid-header or mid-payload and may be a final tail."""


class RawSpool:
    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        stream: str,
        max_records: int | None = None,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        chunk_records: int = 1024,
        session_id: str | None = None,
        strict_records: bool = False,
        manifest_checkpoint_records: int = 1,
        manifest_checkpoint_interval_sec: float = 0.0,
        durability_interval_sec: float = 0.0,
    ) -> None:
        if not stream or "/" in stream or "\\" in stream or stream in {".", ".."}:
            raise ValueError("invalid spool stream")
        if max_records is not None and max_records <= 0:
            raise ValueError("max_records must be positive")
        if max_bytes <= 0 or chunk_records <= 0:
            raise ValueError("max_bytes and chunk_records must be positive")
        if manifest_checkpoint_records <= 0:
            raise ValueError("manifest_checkpoint_records must be positive")
        for name, value in (
            ("manifest_checkpoint_interval_sec", manifest_checkpoint_interval_sec),
            ("durability_interval_sec", durability_interval_sec),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        self.root = Path(root)
        self.stream = stream
        self.max_records = int(max_records) if max_records is not None else None
        self.max_bytes = int(max_bytes)
        self.chunk_records = int(chunk_records)
        self.manifest_checkpoint_records = int(manifest_checkpoint_records)
        self.manifest_checkpoint_interval_sec = float(manifest_checkpoint_interval_sec)
        self.durability_interval_sec = float(durability_interval_sec)
        self.directory = self.root / "camera" / stream
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.inprogress.json"
        self.final_manifest_path = self.root / "manifest.json"
        self._lifecycle_lock = threading.RLock()
        self._dirty_chunks: set[Path] = set()
        self._durable_chunk_sizes: dict[str, int] = {}
        self._last_durability_sync_monotonic = time.monotonic()
        self._records_since_manifest_checkpoint = 0
        self._last_manifest_checkpoint_monotonic = time.monotonic()
        old = self._read_manifest()
        self.session_id = str(session_id or old.get("session_id") or uuid.uuid4().hex)
        old_streams = old.get("streams", {})
        old_stream = (
            old_streams.get(stream, {}) if isinstance(old_streams, dict) else {}
        )
        events = old_stream.get("events", {}) if isinstance(old_stream, dict) else {}
        if not isinstance(events, dict):
            events = (
                old.get("events", {}) if isinstance(old.get("events", {}), dict) else {}
            )
        self._strict_records = bool(
            strict_records
            or (
                isinstance(old_stream, dict)
                and old_stream.get("record_schema") == _CAMERA_RECORD_SCHEMA
            )
        )
        self._record_errors = _normalise_record_errors(
            old_stream.get("record_errors") if isinstance(old_stream, dict) else None
        )
        self._sent = int(events.get("sent", 0))
        self._restarts = int(events.get("restart", 0))
        self._gaps = int(events.get("gap", 0))
        self._overflow = int(events.get("spool_overflow", 0))
        self._corrupt_tail_bytes = int(events.get("corrupt_tail_bytes", 0))
        self._count = 0
        self._bytes = 0
        self._chunk_index = 0
        self._chunk_count = 0
        self._first_sequence: int | None = None
        self._last_sequence: int | None = None
        self._sequence_high_watermark: int | None = None
        self._sequence_gap_count = 0
        self._producer_gap_count = 0
        self._transport_gap_count = 0
        self._duplicate_count = 0
        self._reorder_count = 0
        self._serial = (
            str(old_stream.get("serial"))
            if isinstance(old_stream, dict) and old_stream.get("serial")
            else None
        )
        self._clock_domain = (
            str(old_stream.get("clock_domain"))
            if isinstance(old_stream, dict) and old_stream.get("clock_domain")
            else None
        )
        self._timestamp_quality = (
            str(old_stream.get("timestamp_quality"))
            if isinstance(old_stream, dict) and old_stream.get("timestamp_quality")
            else None
        )
        self._timestamp_domain = (
            str(old_stream.get("timestamp_domain"))
            if isinstance(old_stream, dict) and old_stream.get("timestamp_domain")
            else None
        )
        self._closed = bool(old_stream.get("closed", False))
        self._close_reason = str(old_stream.get("close_reason") or "")
        self._close_error = str(old_stream.get("close_error") or "")
        self._generation = max(1, int(old_stream.get("generation", 1)))
        self._start_fence = _normalise_fence(old_stream.get("start_fence"))
        self._stop_fence = _normalise_fence(old_stream.get("stop_fence"))
        self._accepted_count = 0
        self._written_count = 0
        self._durable_count = 0
        self._accepted_high_watermark: int | None = None
        self._written_high_watermark: int | None = None
        self._durable_high_watermark: int | None = None
        self._recover_files()
        # Recovered complete frames are the only trustworthy prefix after an
        # interrupted process. Bind counters to that prefix rather than to a
        # possibly stale in-progress manifest.
        self._accepted_count = self._count
        self._written_count = self._count
        self._durable_count = self._count
        self._accepted_high_watermark = self._sequence_high_watermark
        self._written_high_watermark = self._sequence_high_watermark
        self._durable_high_watermark = self._sequence_high_watermark
        self._durable_chunk_sizes = {
            path.name: path.stat().st_size for path in self._chunk_paths()
        }
        if self._start_fence is None:
            self._start_fence = self._new_fence("START")
        self._write_manifest(include_checksums=self._closed, force=True)

    def append(self, record: dict[str, Any]) -> bool:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("raw spool is closed")
            if not isinstance(record, dict) or "payload" not in record:
                if self._strict_records:
                    self.record_rejection("invalid_record")
                raise ValueError("spool record must contain payload")
            if self._strict_records:
                error_type = _camera_record_error(
                    record,
                    self.stream,
                    self.session_id,
                    require_timestamp=True,
                    require_identity=True,
                )
                if error_type is not None:
                    self.record_rejection(error_type)
                    raise ValueError(f"camera spool record is invalid: {error_type}")
            if self.max_records is not None and self._count >= self.max_records:
                self._overflow += 1
                self.record_rejection("spool_overflow")
                return False
            try:
                payload = msgpack.packb(record, use_bin_type=True)
            except Exception as exc:
                if self._strict_records:
                    self.record_rejection("serialization_error")
                raise ValueError("spool record cannot be serialized") from exc
            if len(payload) > _MAX_FRAME:
                if self._strict_records:
                    self.record_rejection("record_too_large")
                raise ValueError("spool record is too large")
            frame = (
                _HEADER.pack(_MAGIC, len(payload), zlib.crc32(payload) & 0xFFFFFFFF)
                + payload
            )
            if self._bytes + len(frame) > self.max_bytes:
                self._overflow += 1
                self.record_rejection("spool_overflow")
                return False
            self._accepted_count += 1
            self._accepted_high_watermark = _advance_high_watermark(
                self._accepted_high_watermark, record.get("sequence")
            )
            if self._chunk_count >= self.chunk_records:
                # Never rotate away from a chunk whose bytes have not been
                # durably flushed.
                self._sync_dirty_chunks(force=True)
                self._chunk_index += 1
                self._chunk_count = 0
            chunk_path = self._chunk_path(self._chunk_index)
            with chunk_path.open("ab") as handle:
                handle.write(frame)
                handle.flush()
            self._dirty_chunks.add(chunk_path)
            self._written_count += 1
            self._written_high_watermark = _advance_high_watermark(
                self._written_high_watermark, record.get("sequence")
            )
            self._count += 1
            self._bytes += len(frame)
            self._chunk_count += 1
            self._observe_record(record)
            self._sync_dirty_chunks()
            self._records_since_manifest_checkpoint += 1
            self._write_manifest(force=False)
            return True

    def mark_sent(self, count: int = 1) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._sent += _positive_count(count)
            self._write_manifest(force=False)

    def mark_restart(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("raw spool is closed")
            self._restarts += 1
            self._generation += 1
            self._start_fence = self._new_fence("START")
            self._stop_fence = None
            self._close_error = ""
            self._write_manifest(force=True)

    def mark_start(self) -> None:
        """Persist the START fence before a producer can append."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("raw spool is closed")
            if self._start_fence is None:
                self._start_fence = self._new_fence("START")
            self._write_manifest(force=True)

    def mark_gap(self, count: int = 1) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._gaps += _positive_count(count)
            self._write_manifest(force=False)

    def record_rejection(self, error_type: str, count: int = 1) -> None:
        """Persist source records rejected before they enter the spool."""
        with self._lifecycle_lock:
            if self._closed:
                return
            amount = _positive_count(count)
            key = str(error_type).strip() or "invalid_record"
            errors = _normalise_record_errors(self._record_errors)
            errors["total"] += amount
            errors["by_stream"][self.stream] = (
                errors["by_stream"].get(self.stream, 0) + amount
            )
            errors["by_error_type"][key] = errors["by_error_type"].get(key, 0) + amount
            self._record_errors = errors
            self._write_manifest(force=True)

    def close(
        self, *, reason: str = "process_stop", source_stop_observed: bool = True
    ) -> None:
        """Durably mark this stream closed without deleting its raw chunks."""
        with self._lifecycle_lock:
            if self._closed:
                return
            # The final checksum/manifest is the authoritative close boundary.
            # Flush every pending chunk before calculating either one.
            self._sync_dirty_chunks(force=True)
            self._stop_fence = self._new_fence("STOP") if source_stop_observed else None
            self._closed = True
            self._close_reason = str(reason)
            self._close_error = (
                "" if source_stop_observed else "source STOP was not observed"
            )
            try:
                self._write_manifest(include_checksums=True, force=True)
            except Exception as exc:
                # A failed STOP publication must remain retryable and must not
                # be represented as a complete camera source.
                self._closed = False
                self._stop_fence = None
                self._close_error = f"{type(exc).__name__}: {exc}"
                raise

    def mark_close_failed(self, error: BaseException | str) -> None:
        """Best-effort persistence of a server-side close failure."""
        with self._lifecycle_lock:
            self._closed = False
            self._stop_fence = None
            self._close_error = (
                str(error)
                if isinstance(error, str)
                else f"{type(error).__name__}: {error}"
            )
            self._close_reason = "close_failed"
            self._write_manifest(force=True)

    def status(self) -> dict[str, Any]:
        """Return a stable, payload-free source fence/status projection."""
        with self._lifecycle_lock:
            return self._status_projection()

    def recover(self) -> Iterator[dict[str, Any]]:
        for path in self._chunk_paths():
            with path.open("rb") as handle:
                while True:
                    item = self._read_frame(handle)
                    if item is None:
                        break
                    value, _ = item
                    yield value

    def _recover_files(self) -> None:
        paths = self._chunk_paths()
        for position, path in enumerate(paths):
            valid = records = 0
            with path.open("r+b") as handle:
                while True:
                    offset = handle.tell()
                    try:
                        item = self._read_frame(handle)
                    except _TruncatedSpoolFrame as exc:
                        if position != len(paths) - 1:
                            raise ValueError(
                                f"corrupt non-final spool chunk: {path}"
                            ) from exc
                        tail_bytes = path.stat().st_size - offset
                        handle.truncate(offset)
                        self._corrupt_tail_bytes += tail_bytes
                        handle.flush()
                        os.fsync(handle.fileno())
                        break
                    except ValueError as exc:
                        if position != len(paths) - 1:
                            raise ValueError(
                                f"corrupt non-final spool chunk: {path}"
                            ) from exc
                        raise ValueError(f"corrupt spool chunk: {path}") from exc
                    if item is None:
                        break
                    _, frame_size = item
                    valid += frame_size
                    records += 1
                    self._observe_record(item[0])
            self._count += records
            self._bytes += valid
            if position == len(paths) - 1:
                self._chunk_index = int(path.stem.split("-")[-1])
                self._chunk_count = records
        if paths and self._chunk_count >= self.chunk_records:
            self._chunk_index += 1
            self._chunk_count = 0

    @staticmethod
    def _read_frame(handle: Any) -> tuple[dict[str, Any], int] | None:
        header = handle.read(_HEADER.size)
        if not header:
            return None
        if len(header) != _HEADER.size:
            raise _TruncatedSpoolFrame("truncated spool frame header")
        magic, length, checksum = _HEADER.unpack(header)
        if magic != _MAGIC or length > _MAX_FRAME:
            raise ValueError("invalid spool frame header")
        payload = handle.read(length)
        if len(payload) != length:
            raise _TruncatedSpoolFrame("truncated spool frame payload")
        if (zlib.crc32(payload) & 0xFFFFFFFF) != checksum:
            raise ValueError("spool frame checksum mismatch")
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except (msgpack.ExtraData, msgpack.UnpackException, ValueError):
            raise ValueError("invalid spool frame msgpack")
        if not isinstance(value, dict):
            raise ValueError("spool frame must be a mapping")
        return value, _HEADER.size + length

    def _chunk_paths(self) -> list[Path]:
        return sorted(self.directory.glob("chunk-*.msgpack"))

    def _chunk_path(self, index: int) -> Path:
        return self.directory / f"chunk-{index:06d}.msgpack"

    def _read_manifest(self) -> dict[str, Any]:
        path = (
            self.final_manifest_path
            if self.final_manifest_path.exists()
            else self.manifest_path
        )
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"corrupt raw spool manifest: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"raw spool manifest must be an object: {path}")
        return value

    def _observe_record(self, record: dict[str, Any]) -> None:
        sequence = record.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            return
        self._sequence_high_watermark = _advance_high_watermark(
            self._sequence_high_watermark, sequence
        )
        if self._first_sequence is None:
            self._first_sequence = sequence
        if self._last_sequence is not None:
            if sequence == self._last_sequence:
                self._duplicate_count += 1
            elif sequence < self._last_sequence:
                self._reorder_count += 1
            else:
                missing = sequence - self._last_sequence - 1
                self._sequence_gap_count += missing
                explicit = record.get("producer_gap_count")
                producer = (
                    int(explicit)
                    if isinstance(explicit, int)
                    and not isinstance(explicit, bool)
                    and explicit >= 0
                    else missing
                )
                producer = min(producer, missing)
                self._producer_gap_count += producer
                self._transport_gap_count += missing - producer
        self._last_sequence = sequence
        serial = record.get("serial")
        if serial:
            if self._serial is None:
                self._serial = str(serial)
            elif self._serial != str(serial):
                self._serial = "mixed"
        clock_domain = record.get("clock_domain")
        if clock_domain:
            if self._clock_domain is None:
                self._clock_domain = str(clock_domain)
            elif self._clock_domain != str(clock_domain):
                self._clock_domain = "mixed"
        timestamp_quality = record.get("timestamp_quality")
        if timestamp_quality:
            if self._timestamp_quality is None:
                self._timestamp_quality = str(timestamp_quality)
            elif self._timestamp_quality != str(timestamp_quality):
                self._timestamp_quality = "mixed"
        timestamp_domain = record.get(
            "timestamp_domain", record.get("device_timestamp_domain_type")
        )
        if timestamp_domain:
            if self._timestamp_domain is None:
                self._timestamp_domain = str(timestamp_domain)
            elif self._timestamp_domain != str(timestamp_domain):
                self._timestamp_domain = "mixed"

    def _write_manifest(
        self, *, include_checksums: bool = False, force: bool = True
    ) -> None:
        # Keep the checkpoint decision, durability barrier, and manifest
        # replacement in the same lifecycle critical section.  This prevents
        # a close from publishing a manifest while an append is still making
        # its chunk visible, and makes a checkpoint describe a synced prefix.
        with self._lifecycle_lock:
            if not force and not self._manifest_checkpoint_due():
                return
            self._sync_dirty_chunks(force=True)
            with _MANIFEST_LOCKS_GUARD:
                lock = _MANIFEST_LOCKS.setdefault(self.manifest_path, threading.Lock())
            with lock:
                self._write_manifest_locked(include_checksums=include_checksums)
            self._records_since_manifest_checkpoint = 0
            self._last_manifest_checkpoint_monotonic = time.monotonic()

    def _manifest_checkpoint_due(self) -> bool:
        if self.manifest_checkpoint_records <= 1:
            return True
        if self._records_since_manifest_checkpoint >= self.manifest_checkpoint_records:
            return True
        interval = self.manifest_checkpoint_interval_sec
        return interval <= 0 or (
            time.monotonic() - self._last_manifest_checkpoint_monotonic >= interval
        )

    def _sync_dirty_chunks(self, *, force: bool = False) -> None:
        if not self._dirty_chunks:
            return
        if (
            not force
            and self.durability_interval_sec > 0
            and time.monotonic() - self._last_durability_sync_monotonic
            < self.durability_interval_sec
        ):
            return
        for path in sorted(self._dirty_chunks):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            self._durable_chunk_sizes[path.name] = path.stat().st_size
        self._dirty_chunks.clear()
        self._durable_count = self._written_count
        self._durable_high_watermark = self._written_high_watermark
        self._last_durability_sync_monotonic = time.monotonic()

    def _new_fence(self, event: str) -> dict[str, Any]:
        return {
            "event": event,
            "session_id": self.session_id,
            "stream_id": self.stream,
            "generation": self._generation,
            "restart_count": self._restarts,
            "timestamp_ns": time.time_ns(),
            "accepted_count": self._accepted_count,
            "written_count": self._written_count,
            "durable_count": self._durable_count,
            "accepted_high_watermark": self._accepted_high_watermark,
            "written_high_watermark": self._written_high_watermark,
            "durable_high_watermark": self._durable_high_watermark,
        }

    def _status_projection(self) -> dict[str, Any]:
        complete = bool(self._closed and self._stop_fence is not None)
        return {
            "schema": "robo_collector.camera_spool_status.v1",
            "session_id": self.session_id,
            "stream_id": self.stream,
            "generation": self._generation,
            "restart_count": self._restarts,
            "accepted_count": self._accepted_count,
            "written_count": self._written_count,
            "durable_count": self._durable_count,
            "accepted_high_watermark": self._accepted_high_watermark,
            "written_high_watermark": self._written_high_watermark,
            "durable_high_watermark": self._durable_high_watermark,
            "durable_chunks": dict(sorted(self._durable_chunk_sizes.items())),
            "start_fence": dict(self._start_fence) if self._start_fence else None,
            "stop_fence": dict(self._stop_fence) if self._stop_fence else None,
            "source_complete": complete,
            "close_reason": self._close_reason or None,
            "close_error": self._close_error or None,
        }

    def _write_manifest_locked(self, *, include_checksums: bool = False) -> None:
        existing = self._read_manifest()
        streams = existing.get("streams", {})
        if not isinstance(streams, dict):
            streams = {}
        stream_manifest = {
            "frame_count": self._count,
            "bytes": self._bytes,
            "first_sequence": self._first_sequence,
            "last_sequence": self._last_sequence,
            "sequence_gap_count": self._sequence_gap_count,
            "producer_gap_count": self._producer_gap_count,
            "transport_gap_count": self._transport_gap_count,
            "duplicate_count": self._duplicate_count,
            "reorder_count": self._reorder_count,
            "serial": self._serial,
            "clock_domain": self._clock_domain,
            "timestamp_quality": self._timestamp_quality,
            "timestamp_domain": self._timestamp_domain,
            "session_id": self.session_id,
            **self._status_projection(),
            "record_errors": self._record_errors,
            "closed": self._closed,
            "close_reason": self._close_reason or None,
            "events": {
                "sent": self._sent,
                "restart": self._restarts,
                "gap": self._gaps,
                "spool_overflow": self._overflow,
                "corrupt_tail_bytes": self._corrupt_tail_bytes,
            },
        }
        if self._strict_records:
            stream_manifest["record_schema"] = _CAMERA_RECORD_SCHEMA
        streams[self.stream] = stream_manifest
        totals = {
            "sent": 0,
            "restart": 0,
            "gap": 0,
            "spool_overflow": 0,
            "corrupt_tail_bytes": 0,
        }
        record_errors = {
            "total": 0,
            "by_stream": {},
            "by_error_type": {},
        }
        records = 0
        for stream in streams.values():
            if not isinstance(stream, dict):
                continue
            records += int(stream.get("frame_count", 0))
            stream_events = stream.get("events", {})
            if isinstance(stream_events, dict):
                for key in totals:
                    totals[key] += int(stream_events.get(key, 0))
            stream_errors = _normalise_record_errors(stream.get("record_errors"))
            record_errors["total"] += stream_errors["total"]
            for key, count in stream_errors["by_stream"].items():
                record_errors["by_stream"][key] = (
                    record_errors["by_stream"].get(key, 0) + count
                )
            for key, count in stream_errors["by_error_type"].items():
                record_errors["by_error_type"][key] = (
                    record_errors["by_error_type"].get(key, 0) + count
                )
        stream_entries = [
            stream for stream in streams.values() if isinstance(stream, dict)
        ]
        session_finalized = bool(stream_entries) and all(
            bool(stream.get("closed")) for stream in stream_entries
        )
        session_complete = session_finalized and all(
            bool(stream.get("source_complete"))
            and isinstance(stream.get("stop_fence"), dict)
            for stream in stream_entries
        )
        close_failures = {
            str(name): str(stream.get("close_error"))
            for name, stream in streams.items()
            if isinstance(stream, dict) and stream.get("close_error")
        }
        manifest = {
            # The source-side framed-msgpack spool has a different on-disk
            # record contract from the host-side RER1 JSON Raw Episode.  Keep
            # the directory layout compatible, but make the schema boundary
            # explicit so a consumer cannot parse one as the other.
            "schema": _CAMERA_SPOOL_SCHEMA,
            "format": "raw_spool.msgpack_crc32.v1",
            "episode_id": self.session_id,
            "status": (
                "RAW_CLOSED"
                if session_complete
                else "RAW_INCOMPLETE"
                if session_finalized
                else "RAW_IN_PROGRESS"
            ),
            "source_scope": "camera_capture",
            "capture_plane": "camera_side",
            "session_id": self.session_id,
            "stream_id": self.stream,
            "streams": streams,
            "source_fences": {
                str(name): {
                    "start": stream.get("start_fence"),
                    "stop": stream.get("stop_fence"),
                    "generation": stream.get("generation"),
                    "restart_count": stream.get("restart_count"),
                    "source_complete": bool(stream.get("source_complete")),
                }
                for name, stream in streams.items()
                if isinstance(stream, dict)
            },
            "source_complete": session_complete,
            "close_failures": close_failures,
            "events": {"spool_records": records, "produced": records, **totals},
            "record_errors": record_errors,
            "termination": (
                {
                    "reason": (
                        "all_streams_closed"
                        if session_complete
                        else "closed_without_source_stop"
                    )
                }
                if session_finalized
                else None
            ),
        }
        if include_checksums:
            manifest["checksums"] = {
                str(path.relative_to(self.root)): _sha256_file(path)
                for path in sorted(self.root.glob("camera/*/chunk-*.msgpack"))
                if path.is_file()
            }
        if session_finalized:
            # The final source manifest is immutable after all camera streams
            # have closed.  Keep an explicit digest so a collector can bind a
            # task episode to this exact source session rather than to a
            # mutable in-memory copy of the manifest.
            source_hash = hashlib.sha256(
                json.dumps(
                    manifest,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            manifest["manifest_hash"] = source_hash
            manifest["raw_manifest_hash"] = source_hash
        temporary = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(json.dumps(manifest, sort_keys=True).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        target = self.final_manifest_path if session_finalized else self.manifest_path
        os.replace(temporary, target)
        if session_finalized and self.manifest_path.exists():
            try:
                self.manifest_path.unlink()
            except OSError:
                # The final manifest is already authoritative. A stale
                # in-progress name is harmless because all readers prefer the
                # final path and startup scanning only considers sessions below
                # the configured spool parent.
                pass
        try:
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass


def _positive_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("count must be a positive integer")
    return value


def _normalise_record_errors(value: Any) -> dict[str, Any]:
    """Return a JSON-safe error counter without trusting malformed metadata."""
    result: dict[str, Any] = {
        "total": 0,
        "by_stream": {},
        "by_error_type": {},
    }
    if not isinstance(value, dict):
        return result
    total = value.get("total")
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        result["total"] = total
    for target, source in (
        (result["by_stream"], value.get("by_stream")),
        (result["by_error_type"], value.get("by_error_type")),
    ):
        if not isinstance(source, dict):
            continue
        for key, count in source.items():
            if (
                isinstance(key, str)
                and key
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count >= 0
            ):
                target[key] = count
    return result


def _camera_record_error(
    record: Any,
    stream: str,
    session_id: str | None,
    *,
    require_timestamp: bool,
    require_identity: bool = False,
) -> str | None:
    """Validate the source-camera record fields needed for safe import."""
    if not isinstance(record, dict):
        return "invalid_record"
    record_stream = record.get("stream")
    if require_identity and (
        not isinstance(record_stream, str) or not record_stream.strip()
    ):
        return "missing_stream"
    if record_stream is not None and record_stream != stream:
        return "invalid_stream"
    sequence = record.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        return "invalid_sequence"
    payload = record.get("payload")
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        return "invalid_payload"
    record_session = record.get("session_id")
    if require_identity and (
        not isinstance(record_session, str) or not record_session.strip()
    ):
        return "missing_session_id"
    if session_id is not None and str(record_session) != str(session_id):
        return "invalid_session"
    timestamp = record.get("server_wall_timestamp")
    if timestamp is None:
        timestamp = record.get("server_wall")
    if timestamp is None:
        return "invalid_server_wall_timestamp" if require_timestamp else None
    try:
        timestamp_value = float(timestamp)
    except (TypeError, ValueError):
        return "invalid_server_wall_timestamp"
    if not math.isfinite(timestamp_value):
        return "invalid_server_wall_timestamp"
    clock_domain = record.get("clock_domain")
    if require_identity and (
        not isinstance(clock_domain, str) or not clock_domain.strip()
    ):
        return "missing_clock_domain"
    encoding = record.get("payload_encoding")
    if encoding is not None and encoding not in {"image/jpeg", "image/png"}:
        return "invalid_payload_encoding"
    return None


def scan_camera_spools(root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Recover interrupted camera sessions below ``root``.

    A camera process receives a fresh session directory on every start.  An
    older session that still has an in-progress manifest is therefore finalized
    as an incomplete crash-recovered source before the next process starts. A
    missing source STOP is retained explicitly and never promoted to complete.
    No chunk is deleted; damaged non-final chunks are quarantined instead.
    """
    root_path = Path(root)
    if not root_path.exists():
        return []
    recovered: list[dict[str, Any]] = []
    for session_path in sorted(
        path.parent
        for path in root_path.glob("*/manifest.inprogress.json")
        if path.is_file()
    ):
        try:
            manifest = json.loads(
                (session_path / "manifest.inprogress.json").read_text(encoding="utf-8")
            )
            streams = manifest.get("streams", {})
            if not isinstance(streams, dict):
                raise ValueError("camera spool streams must be a mapping")
            stream_names = [
                name for name in sorted(streams) if isinstance(streams[name], dict)
            ]
            if not stream_names:
                continue
            spools = [
                RawSpool(
                    session_path,
                    stream=name,
                    session_id=str(manifest.get("session_id") or session_path.name),
                    strict_records=bool(
                        isinstance(streams.get(name), dict)
                        and streams[name].get("record_schema") == _CAMERA_RECORD_SCHEMA
                    ),
                )
                for name in stream_names
            ]
            for spool in spools:
                spool.close(reason="process_crash", source_stop_observed=False)
            final = session_path / "manifest.json"
            recovered.append(json.loads(final.read_text(encoding="utf-8")))
        except Exception as exc:
            _quarantine_spool(session_path, exc)
    return recovered


def read_camera_spool_snapshot(
    session: str | os.PathLike[str],
    *,
    server_wall_window: tuple[float, float] | None = None,
    include_records: bool = True,
    record_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the durable prefix of one camera session without sealing it.

    The camera server can stay online while a host collector closes a task
    Episode.  This reader snapshots each chunk's size first and therefore does
    not consume records appended after the snapshot.  A partial final frame is
    ignored; a checksum failure in an otherwise complete frame is surfaced so
    the caller cannot publish a corrupted complete-source Episode.  Set
    ``include_records=False`` when only the bounded snapshot counters and
    integrity descriptor are needed; the chunk is still scanned, but payloads
    are not retained in memory.  ``record_callback`` receives each selected
    record during that scan and is useful for streaming import into another
    durable store without building a second in-memory copy.
    """
    if not isinstance(include_records, bool):
        raise ValueError("include_records must be a boolean")
    if record_callback is not None and not callable(record_callback):
        raise ValueError("record_callback must be callable")
    session_path = Path(session).resolve()
    manifest_path = (
        session_path / "manifest.json"
        if (session_path / "manifest.json").exists()
        else session_path / "manifest.inprogress.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("camera spool manifest must be an object")
    if manifest.get("schema") not in {
        _CAMERA_SPOOL_SCHEMA,
        _LEGACY_CAMERA_SPOOL_SCHEMA,
    }:
        raise ValueError("unsupported camera spool schema")
    if server_wall_window is not None:
        if not isinstance(server_wall_window, tuple) or len(server_wall_window) != 2:
            raise ValueError("server_wall_window must be a (start, end) tuple")
        try:
            window_start = float(server_wall_window[0])
            window_end = float(server_wall_window[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("server_wall_window must contain finite numbers") from exc
        if (
            not math.isfinite(window_start)
            or not math.isfinite(window_end)
            or window_end < window_start
        ):
            raise ValueError("server_wall_window must contain finite ordered bounds")
    else:
        window_start = window_end = None
    streams = manifest.get("streams", {})
    if not isinstance(streams, dict):
        raise ValueError("camera spool streams must be a mapping")
    declared_manifest_hash = manifest.get(
        "raw_manifest_hash", manifest.get("manifest_hash")
    )
    if declared_manifest_hash is not None:
        if (
            not isinstance(declared_manifest_hash, str)
            or len(declared_manifest_hash) != 64
        ):
            raise ValueError("camera spool manifest hash is invalid")
        if _manifest_hash(manifest) != declared_manifest_hash:
            raise ValueError("camera spool manifest hash mismatch")
    _verify_snapshot_checksums(session_path, manifest)
    snapshot_paths: dict[str, tuple[Path, int]] = {}
    for stream_name in sorted(streams):
        stream_dir = session_path / "camera" / str(stream_name)
        stream_manifest = streams.get(stream_name)
        durable_chunks = (
            stream_manifest.get("durable_chunks")
            if isinstance(stream_manifest, dict)
            else None
        )
        for path in sorted(stream_dir.glob("chunk-*.msgpack")):
            current_size = path.stat().st_size
            if isinstance(durable_chunks, dict):
                declared_size = durable_chunks.get(path.name)
                if (
                    not isinstance(declared_size, int)
                    or isinstance(declared_size, bool)
                    or declared_size <= 0
                ):
                    continue
                snapshot_size = min(current_size, declared_size)
            else:
                # Compatibility spools predate explicit durable chunk bounds.
                snapshot_size = current_size
            snapshot_paths[path.relative_to(session_path).as_posix()] = (
                path,
                snapshot_size,
            )
    snapshot_chunks = {
        relative: {
            "size": size,
            "sha256": _sha256_prefix(path, size),
        }
        for relative, (path, size) in snapshot_paths.items()
    }
    records: list[dict[str, Any]] = []
    snapshot_record_errors = {
        "total": 0,
        "by_stream": {},
        "by_error_type": {},
    }
    total_record_count = 0
    selected_record_count = 0
    selected_record_counts: dict[str, int] = {
        str(stream_name): 0 for stream_name in streams
    }
    stream_positions = {str(stream_name): [None, None, 0] for stream_name in streams}
    for stream_name in sorted(streams):
        paths = [
            path
            for path, _ in (
                snapshot_paths[relative]
                for relative in sorted(snapshot_paths)
                if Path(relative).parts[1] == str(stream_name)
            )
        ]
        for chunk_position, path in enumerate(paths):
            relative = path.relative_to(session_path).as_posix()
            snapshot_size = snapshot_paths[relative][1]
            with path.open("rb") as handle:
                while handle.tell() < snapshot_size:
                    offset = handle.tell()
                    try:
                        item = RawSpool._read_frame(handle)
                    except _TruncatedSpoolFrame as exc:
                        if chunk_position != len(paths) - 1:
                            raise ValueError(
                                f"corrupt non-final spool chunk: {path}"
                            ) from exc
                        # The writer fsyncs each complete frame.  Only the
                        # final snapshot tail can be a concurrent partial
                        # write, so keep the verified prefix and stop here.
                        break
                    if item is None:
                        break
                    value, _ = item
                    if handle.tell() > snapshot_size:
                        # A writer appended after stat(); do not include the
                        # frame that crossed the snapshot boundary.
                        break
                    strict_stream = (
                        isinstance(streams.get(stream_name), dict)
                        and streams[stream_name].get("record_schema")
                        == _CAMERA_RECORD_SCHEMA
                    )
                    record = dict(value)
                    # Legacy spools did not carry these identity fields.  Keep
                    # the compatibility projection for diagnostic reads, but
                    # never synthesize fields in the strict RSP1 contract: a
                    # missing identity must be counted as a rejected record.
                    if not strict_stream:
                        record.setdefault("stream", str(stream_name))
                        record.setdefault(
                            "session_id",
                            manifest.get("session_id") or manifest.get("episode_id"),
                        )
                    error_type = _camera_record_error(
                        record,
                        str(stream_name),
                        str(
                            manifest.get("session_id")
                            or manifest.get("episode_id")
                            or session_path.name
                        ),
                        require_timestamp=strict_stream,
                        require_identity=strict_stream,
                    )
                    if error_type is not None:
                        snapshot_record_errors["total"] += 1
                        by_stream = snapshot_record_errors["by_stream"]
                        by_type = snapshot_record_errors["by_error_type"]
                        by_stream[str(stream_name)] = (
                            by_stream.get(str(stream_name), 0) + 1
                        )
                        by_type[error_type] = by_type.get(error_type, 0) + 1
                    total_record_count += 1
                    timestamp = record.get("server_wall_timestamp")
                    try:
                        timestamp_value = float(timestamp)
                    except (TypeError, ValueError):
                        timestamp_value = float("nan")
                    selected = (
                        window_start is None
                        or not math.isfinite(timestamp_value)
                        or window_start <= timestamp_value <= window_end
                    )
                    if selected:
                        selected_record_count += 1
                        if record_callback is not None:
                            callback_record = dict(record)
                            callback_record["_spool_stream"] = str(stream_name)
                            record_callback(callback_record)
                        elif include_records:
                            records.append(record)
                        selected_record_counts[str(stream_name)] += 1
                    sequence = record.get("sequence")
                    stream_position = stream_positions[str(stream_name)]
                    if isinstance(sequence, int) and not isinstance(sequence, bool):
                        # A high-water mark is a maximum sequence, not merely
                        # the sequence of the last physically read record.  A
                        # reordered source stream must not make a valid prefix
                        # appear to end before an already observed frame.
                        previous_sequence = stream_position[0]
                        stream_position[0] = (
                            sequence
                            if previous_sequence is None
                            else max(previous_sequence, sequence)
                        )
                    stream_position[1] = record.get("server_wall_timestamp")
                    stream_position[2] += 1
                    if handle.tell() <= offset:
                        raise ValueError(
                            f"camera spool reader made no progress: {path}"
                        )
    final_snapshot_paths = {
        path.relative_to(session_path).as_posix(): path.stat().st_size
        for stream_name in sorted(streams)
        for path in sorted(
            (session_path / "camera" / str(stream_name)).glob("chunk-*.msgpack")
        )
        if path.is_file()
    }
    # A live camera server is expected to append after the snapshot boundary.
    # Stability therefore means that every captured file still contains the
    # same verified prefix; newly created chunks and larger tails are outside
    # this snapshot and must not invalidate it.
    stable = True
    for relative, (path, snapshot_size) in snapshot_paths.items():
        final_size = final_snapshot_paths.get(relative)
        if final_size is None or final_size < snapshot_size:
            stable = False
            break
        try:
            if (
                _sha256_prefix(path, snapshot_size)
                != snapshot_chunks[relative]["sha256"]
            ):
                stable = False
                break
        except (OSError, ValueError):
            stable = False
            break
    snapshot = {
        "schema": "robo_collector.camera_spool_snapshot.v1",
        "session_id": str(
            manifest.get("session_id")
            or manifest.get("episode_id")
            or session_path.name
        ),
        "captured_wall_time": time.time(),
        "manifest_hash": manifest.get(
            "raw_manifest_hash", manifest.get("manifest_hash")
        ),
        "chunks": snapshot_chunks,
        "stable": stable,
        "server_wall_window": (
            [window_start, window_end] if window_start is not None else None
        ),
        "stream_high_watermarks": {
            stream: {
                "last_sequence": values[0],
                "last_server_wall_timestamp": values[1],
                "record_count": values[2],
                "accepted_count": streams.get(stream, {}).get("accepted_count"),
                "written_count": streams.get(stream, {}).get("written_count"),
                "durable_count": streams.get(stream, {}).get("durable_count"),
                "accepted_high_watermark": streams.get(stream, {}).get(
                    "accepted_high_watermark"
                ),
                "written_high_watermark": streams.get(stream, {}).get(
                    "written_high_watermark"
                ),
                "durable_high_watermark": streams.get(stream, {}).get(
                    "durable_high_watermark"
                ),
                "generation": streams.get(stream, {}).get("generation"),
                "restart_count": streams.get(stream, {}).get("restart_count"),
            }
            for stream, values in stream_positions.items()
        },
        "source_fences": manifest.get("source_fences", {}),
        "source_complete": bool(manifest.get("source_complete", False)),
        "close_failures": manifest.get("close_failures", {}),
        # ``record_count`` is the complete durable prefix count. The selected
        # count is separate because a task-window snapshot may return only a
        # bounded subset to the collector.
        "record_count": total_record_count,
        "selected_record_count": selected_record_count,
        "selected_record_counts": selected_record_counts,
        "record_errors": snapshot_record_errors,
    }
    manifest["_snapshot"] = snapshot
    return manifest, records


def read_camera_spool_status(
    session: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read the durable payload-free source status published by a camera."""
    session_path = Path(session).resolve()
    manifest_path = (
        session_path / "manifest.json"
        if (session_path / "manifest.json").exists()
        else session_path / "manifest.inprogress.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") not in {
        _CAMERA_SPOOL_SCHEMA,
        _LEGACY_CAMERA_SPOOL_SCHEMA,
    }:
        raise ValueError("unsupported camera spool status")
    declared_hash = manifest.get("raw_manifest_hash", manifest.get("manifest_hash"))
    if declared_hash is not None and _manifest_hash(manifest) != declared_hash:
        raise ValueError("camera spool manifest hash mismatch")
    streams = manifest.get("streams", {})
    if not isinstance(streams, dict):
        raise ValueError("camera spool status streams must be a mapping")
    return {
        "schema": "robo_collector.camera_spool_status.v1",
        "session_id": str(
            manifest.get("session_id")
            or manifest.get("episode_id")
            or session_path.name
        ),
        "status": manifest.get("status"),
        "source_complete": bool(manifest.get("source_complete", False)),
        "source_fences": manifest.get("source_fences", {}),
        "close_failures": manifest.get("close_failures", {}),
        "streams": {
            str(name): {
                key: stream.get(key)
                for key in (
                    "generation",
                    "restart_count",
                    "accepted_count",
                    "written_count",
                    "durable_count",
                    "accepted_high_watermark",
                    "written_high_watermark",
                    "durable_high_watermark",
                    "durable_chunks",
                    "start_fence",
                    "stop_fence",
                    "source_complete",
                    "close_reason",
                    "close_error",
                )
            }
            for name, stream in streams.items()
            if isinstance(stream, dict)
        },
    }


def _sha256_prefix(path: Path, size: int) -> str:
    """Hash exactly the durable prefix observed by a live spool snapshot."""
    digest = hashlib.sha256()
    remaining = max(0, int(size))
    with path.open("rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    if remaining:
        raise ValueError(f"camera spool chunk shrank during snapshot: {path}")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash a complete chunk without loading the chunk into process memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_hash(manifest: dict[str, Any]) -> str:
    """Hash the immutable source manifest fields, excluding its own digest."""
    value = dict(manifest)
    value.pop("manifest_hash", None)
    value.pop("raw_manifest_hash", None)
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _verify_snapshot_checksums(session_path: Path, manifest: dict[str, Any]) -> None:
    """Verify a sealed session's chunk inventory before importing its records."""
    checksums = manifest.get("checksums")
    if checksums is None:
        return
    if not isinstance(checksums, dict):
        raise ValueError("camera spool checksums must be a mapping")

    expected_paths: set[str] = set()
    for raw_relative, expected_hash in checksums.items():
        relative = str(raw_relative)
        relative_path = Path(relative)
        parts = relative_path.parts
        if (
            relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or len(parts) != 3
            or parts[0] != "camera"
            or not parts[1]
            or not parts[2].startswith("chunk-")
            or not parts[2].endswith(".msgpack")
            or ".." in parts
        ):
            raise ValueError(f"invalid camera spool checksum path: {relative!r}")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"invalid camera spool checksum: {relative!r}")
        path = (session_path / relative_path).resolve()
        try:
            path.relative_to(session_path)
        except ValueError as exc:
            raise ValueError(
                f"camera spool checksum escapes session: {relative!r}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"camera spool checksum file is missing: {relative!r}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"camera spool checksum mismatch: {relative!r}")
        expected_paths.add(relative)

    actual_paths = {
        path.relative_to(session_path).as_posix()
        for path in session_path.glob("camera/*/chunk-*.msgpack")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("camera spool checksums do not cover all chunk files")


def _quarantine_spool(path: Path, error: BaseException) -> None:
    """Make a damaged camera session visible without pretending it is closed."""
    source = path / "manifest.json"
    if not source.exists():
        source = path / "manifest.inprogress.json"

    def preserve_corrupt_manifest() -> None:
        if not source.exists():
            return
        preserved = path / "manifest.corrupt.json"
        suffix = 1
        while preserved.exists():
            preserved = path / f"manifest.corrupt.{suffix}.json"
            suffix += 1
        try:
            shutil.copy2(source, preserved)
        except OSError:
            pass

    try:
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            # A syntactically valid JSON list/scalar is still not a manifest;
            # preserve it before replacing it with the quarantine marker.
            preserve_corrupt_manifest()
            value = None
    except Exception:
        # Preserve malformed state before replacing it with the quarantine
        # marker.  Recovery must never silently erase the evidence that the
        # source manifest itself was unreadable.
        preserve_corrupt_manifest()
        value = {
            "schema": _CAMERA_SPOOL_SCHEMA,
            "episode_id": path.name,
            "source_scope": "camera_capture",
        }
    if not isinstance(value, dict):
        value = {"schema": _CAMERA_SPOOL_SCHEMA, "episode_id": path.name}
    value.update(status="QUARANTINED", quarantine_error=str(error))
    temporary = path / ".manifest.quarantine.tmp"
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path / "manifest.json")
    try:
        (path / "manifest.inprogress.json").unlink()
    except FileNotFoundError:
        pass


def _positive_count(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("count must be a positive integer")
    return value


def _advance_high_watermark(current: int | None, candidate: Any) -> int | None:
    if not isinstance(candidate, int) or isinstance(candidate, bool) or candidate < 0:
        return current
    return candidate if current is None else max(current, candidate)


def _normalise_fence(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    event = value.get("event")
    if event not in {"START", "STOP"}:
        return None
    return dict(value)


__all__ = [
    "RawSpool",
    "read_camera_spool_snapshot",
    "read_camera_spool_status",
    "scan_camera_spools",
]
