"""Crash-safe MCAP v1 landing writer and immutable prefix recovery.

The landing file deliberately uses only the public :mod:`mcap` 1.4.0 writer
and stream-reader APIs.  A single worker thread owns every Writer call.  The
camera Raw Spool is outside this module and is never opened or modified here.
"""

from __future__ import annotations

import hashlib
import os
import queue
import stat
import struct
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Self

from .mcap_contract import (
    MAX_MCAP_RECORD_BYTES,
    MCAP_PROFILE,
    CheckpointFrameError,
    decode_inprogress_manifest,
    encode_checkpoint_frame,
    encode_inprogress_manifest,
    iter_checkpoint_frames,
    validate_checkpoint_payload,
    validate_inprogress_manifest,
)

MCAP_LIBRARY = "mcap-python/1.4.0"


class LandingError(RuntimeError):
    """Base error for landing writer failures."""


class LandingStateError(LandingError):
    """Raised when an operation is invalid for the current lifecycle state."""


class LandingQueueFull(LandingError):
    """Raised when bounded admission rejects a record."""


class LandingFaulted(LandingError):
    """Raised after the writer has entered ``FAULTED``."""


class RecoveryError(LandingError):
    """Raised by strict durable-prefix inspection helpers."""


class RecoveryExitCode(IntEnum):
    """Frozen recovery CLI result contract."""

    RECOVERED = 0
    NO_DURABLE_PREFIX = 2
    CORRUPT_PREFIX_OR_JOURNAL = 3
    SOURCE_INCOMPLETE = 4
    IO_OR_INTERNAL_ERROR = 5


@dataclass(frozen=True)
class RequiredSource:
    source_id: str
    session_id: str
    start_sequence_exclusive: int


@dataclass(frozen=True)
class SourceFence:
    source_id: str
    session_id: str
    start_sequence_exclusive: int
    end_sequence_inclusive: int | None = None


@dataclass(frozen=True)
class LandingChannel:
    topic: str
    schema_name: str
    schema_data: bytes
    metadata: Mapping[str, str] = field(default_factory=dict)
    schema_encoding: str = "protobuf"
    message_encoding: str = "protobuf"


@dataclass(frozen=True)
class LandingRecord:
    channel: str
    data: bytes
    log_time: int
    publish_time: int
    sequence: int
    collector_record_id: int
    source_id: str = ""
    session_id: str = ""
    source_sequence: int | None = None
    packet_sequence: int | None = None


@dataclass(frozen=True)
class LandingSeal:
    sealed_path: Path
    checkpoint_path: Path
    manifest_path: Path
    last_checkpoint: Mapping[str, Any]
    source_complete: bool


@dataclass(frozen=True)
class JournalScan:
    checkpoints: tuple[Mapping[str, Any], ...]
    torn_final: bool


@dataclass(frozen=True)
class DurablePrefix:
    source_path: Path
    checkpoint_path: Path
    byte_offset: int
    sha256: str
    checkpoint: Mapping[str, Any]
    torn_final_journal: bool = False
    source_stat: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class RecoveryResult:
    exit_code: RecoveryExitCode
    recovered_path: Path | None = None
    durable_prefix: DurablePrefix | None = None
    source_complete: bool = False
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == RecoveryExitCode.RECOVERED


@dataclass
class _ChannelProgress:
    accepted_count: int = 0
    written_count: int = 0
    durable_count: int = 0
    accepted_high_watermark: int | None = None
    written_high_watermark: int | None = None
    durable_high_watermark: int | None = None
    session_id: str = ""
    last_source_sequence: int | None = None
    last_packet_sequence: int | None = None


@dataclass
class _FenceProgress:
    fence: SourceFence
    accepted_count: int = 0
    written_count: int = 0
    durable_count: int = 0
    last_accepted_sequence: int | None = None
    written_high_watermark: int | None = None
    durable_high_watermark: int | None = None


@dataclass
class _Control:
    action: str
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


def _load_mcap_writer() -> type[Any]:
    try:
        from mcap.writer import Writer
    except ImportError as exc:  # pragma: no cover - environment error
        raise LandingError("mcap==1.4.0 is required for landing writes") from exc
    _require_mcap_version()
    return Writer


def _load_stream_types() -> tuple[
    type[Any], tuple[type[Any], ...], type[BaseException]
]:
    try:
        from mcap.exceptions import EndOfFile
        from mcap.records import Channel, Header, Message, Schema
        from mcap.stream_reader import StreamReader
    except ImportError as exc:  # pragma: no cover - environment error
        raise RecoveryError("mcap==1.4.0 is required for recovery") from exc
    _require_mcap_version()
    return StreamReader, (Header, Schema, Channel, Message), EndOfFile


def _require_mcap_version() -> None:
    try:
        installed = version("mcap")
    except PackageNotFoundError as exc:  # pragma: no cover - environment error
        raise LandingError("mcap==1.4.0 is required") from exc
    if installed != "1.4.0":
        raise LandingError(f"mcap==1.4.0 is required, found {installed}")


def _nonnegative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _fsync_directory(path: Path, fsync: Callable[[int], None]) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(
    path: Path,
    payload: bytes,
    *,
    fsync: Callable[[int], None],
    replace: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None],
) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with open(temporary, "xb") as stream:
        stream.write(payload)
        stream.flush()
        fsync(stream.fileno())
    replace(temporary, path)
    _fsync_directory(path.parent, fsync)


def _exclusive_write(
    path: Path, payload: bytes, *, fsync: Callable[[int], None]
) -> None:
    """Create an ownership artifact without replacing a concurrent winner."""

    with open(path, "xb") as stream:
        stream.write(payload)
        stream.flush()
        fsync(stream.fileno())
    _fsync_directory(path.parent, fsync)


def _install_no_replace(
    source: Path, target: Path, *, fsync: Callable[[int], None]
) -> None:
    """Atomically install ``target`` while preserving evidence on failures."""

    os.link(source, target)
    _fsync_directory(target.parent, fsync)
    source.unlink()
    try:
        _fsync_directory(target.parent, fsync)
    except OSError:
        # The target link was already made durable.  A crash may resurrect the
        # source name, which is harmless immutable duplicate evidence.
        pass


def _sha256_prefix(path: Path, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with open(path, "rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise RecoveryError("landing file ended before durable byte offset")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_prefix(path, path.stat().st_size)


class LandingWriter:
    """Bounded, single-owner landing writer.

    Channels must be registered before :meth:`start`.  Admission is nonblocking;
    a full queue faults the episode rather than silently dropping a record.
    """

    def __init__(
        self,
        episode_dir: str | os.PathLike[str],
        *,
        episode_id: str,
        attempt: int = 1,
        collection_mode: str = "mcap_first",
        required_sources: Iterable[RequiredSource | Mapping[str, Any]] = (),
        queue_capacity: int = 64,
        max_unsynced_records: int = 1,
        durability_interval_sec: float = 0.0,
        generation: int = 0,
        created_time_ns: int | None = None,
        writer_factory: Callable[..., Any] | None = None,
        fsync: Callable[[int], None] = os.fsync,
        replace: Callable[
            [str | os.PathLike[str], str | os.PathLike[str]], None
        ] = os.replace,
        monotonic: Callable[[], float] = time.monotonic,
        shutdown_timeout_sec: float = 2.0,
    ) -> None:
        self.episode_dir = Path(episode_dir)
        self.landing_dir = self.episode_dir / "landing"
        self.partial_path = self.landing_dir / "episode.mcap.partial"
        self.sealed_path = self.landing_dir / "episode.mcap"
        self.checkpoint_path = self.landing_dir / "checkpoints.bin"
        self.manifest_path = self.episode_dir / "manifest.inprogress.json"
        self.episode_id = episode_id
        self.attempt = _nonnegative(attempt, "attempt")
        if self.attempt == 0:
            raise ValueError("attempt must be positive")
        self.collection_mode = collection_mode
        self.queue_capacity = _nonnegative(queue_capacity, "queue_capacity")
        if self.queue_capacity == 0:
            raise ValueError("queue_capacity must be positive")
        self.max_unsynced_records = _nonnegative(
            max_unsynced_records, "max_unsynced_records"
        )
        if self.max_unsynced_records == 0:
            raise ValueError("max_unsynced_records must be positive")
        if durability_interval_sec < 0:
            raise ValueError("durability_interval_sec must be non-negative")
        self.durability_interval_sec = float(durability_interval_sec)
        self.generation = _nonnegative(generation, "generation")
        self.created_time_ns = (
            time.time_ns()
            if created_time_ns is None
            else _nonnegative(created_time_ns, "created_time_ns")
        )
        self._writer_factory = writer_factory or _load_mcap_writer()
        self._fsync = fsync
        self._replace = replace
        self._monotonic = monotonic
        if shutdown_timeout_sec <= 0:
            raise ValueError("shutdown_timeout_sec must be positive")
        self.shutdown_timeout_sec = float(shutdown_timeout_sec)
        self._channels: dict[str, LandingChannel] = {}
        self._channel_ids: dict[str, int] = {}
        self._channel_progress: dict[str, _ChannelProgress] = {}
        self._required_sources = tuple(
            self._normalize_required_sources(required_sources)
        )
        self._fences: dict[tuple[str, str], _FenceProgress] = {
            (source.source_id, source.session_id): _FenceProgress(
                SourceFence(
                    source.source_id,
                    source.session_id,
                    source.start_sequence_exclusive,
                )
            )
            for source in self._required_sources
        }
        self._queue: queue.Queue[LandingRecord | _Control] = queue.Queue(
            self.queue_capacity
        )
        self._admission_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = "NEW"
        self._fault: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stream: BinaryIO | None = None
        self._journal: BinaryIO | None = None
        self._writer: Any = None
        self._accepted_count = 0
        self._accepted_frontier: int | None = None
        self._written_count = 0
        self._written_frontier: int | None = None
        self._durable_count = 0
        self._durable_frontier: int | None = None
        self._checkpoint_sequence = 0
        self._last_checkpoint: Mapping[str, Any] | None = None
        self._last_checkpoint_time = self._monotonic()

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def fault_error(self) -> BaseException | None:
        return self._fault

    def register_channel(self, channel: LandingChannel) -> None:
        with self._state_lock:
            if self._state != "NEW":
                raise LandingStateError("channels must be registered before start")
            if channel.topic in self._channels:
                raise ValueError(f"duplicate landing channel: {channel.topic}")
            if not channel.topic or not channel.schema_name or not channel.schema_data:
                raise ValueError("landing channel topic/schema must be non-empty")
            self._channels[channel.topic] = LandingChannel(
                topic=channel.topic,
                schema_name=channel.schema_name,
                schema_data=bytes(channel.schema_data),
                metadata=dict(channel.metadata),
                schema_encoding=channel.schema_encoding,
                message_encoding=channel.message_encoding,
            )
            self._channel_progress[channel.topic] = _ChannelProgress()

    def start(self) -> LandingWriter:
        with self._state_lock:
            if self._state != "NEW":
                raise LandingStateError(
                    f"cannot start landing writer from {self._state}"
                )
            if not self._channels:
                raise LandingStateError("at least one landing channel is required")
            self._state = "STARTING"
        try:
            self._create_layout()
            self._write_initial_manifest()
        except BaseException as exc:
            # Ownership may not have been established (for example an existing
            # source partial).  Never rewrite pre-existing recovery evidence.
            with self._state_lock:
                self._state = "FAULTED"
                self._fault = exc
            raise
        with self._state_lock:
            self._state = "OPEN"
            self._thread = threading.Thread(
                target=self._run, name=f"mcap-landing-{self.episode_id}", daemon=True
            )
            self._thread.start()
        self._ready.wait()
        self._raise_if_faulted()
        return self

    def submit(self, record: LandingRecord) -> int:
        immutable_data = bytes(record.data)
        # An MCAP Message record adds 22 bytes of fixed fields before ``data``.
        if len(immutable_data) + 22 > MAX_MCAP_RECORD_BYTES:
            raise ValueError("landing record content exceeds 64 MiB")
        if record.channel not in self._channels:
            raise ValueError(f"unregistered landing channel: {record.channel}")
        for value, name in (
            (record.log_time, "log_time"),
            (record.publish_time, "publish_time"),
            (record.sequence, "sequence"),
            (record.collector_record_id, "collector_record_id"),
        ):
            _nonnegative(value, name)
        if record.sequence > (1 << 32) - 1:
            raise ValueError("sequence exceeds uint32")
        if record.log_time > (1 << 63) - 1 or record.publish_time > (1 << 63) - 1:
            raise ValueError("landing timestamps exceed signed 63-bit contract")
        for value, name in (
            (record.source_sequence, "source_sequence"),
            (record.packet_sequence, "packet_sequence"),
        ):
            if value is not None:
                _nonnegative(value, name)
                if value > (1 << 64) - 1:
                    raise ValueError(f"{name} exceeds uint64")
        accepted_record = LandingRecord(
            channel=record.channel,
            data=immutable_data,
            log_time=record.log_time,
            publish_time=record.publish_time,
            sequence=record.sequence,
            collector_record_id=record.collector_record_id,
            source_id=record.source_id,
            session_id=record.session_id,
            source_sequence=record.source_sequence,
            packet_sequence=record.packet_sequence,
        )
        fence = self._fences.get(
            (accepted_record.source_id, accepted_record.session_id)
        )
        if fence is not None and accepted_record.source_sequence is None:
            raise ValueError("required-source record needs source_sequence")
        with self._admission_lock:
            self._require_open()
            if fence is not None:
                assert accepted_record.source_sequence is not None
                if (
                    accepted_record.source_sequence
                    <= fence.fence.start_sequence_exclusive
                ):
                    raise ValueError(
                        "required source_sequence must follow the START fence"
                    )
                if (
                    fence.last_accepted_sequence is not None
                    and accepted_record.source_sequence <= fence.last_accepted_sequence
                ):
                    raise ValueError(
                        "required source_sequence must increase monotonically"
                    )
            expected = (
                record.collector_record_id
                if self._accepted_frontier is None
                else self._accepted_frontier + 1
            )
            if record.collector_record_id != expected:
                self._fault_writer(
                    LandingError("collector_record_id must be globally contiguous")
                )
                raise LandingFaulted("collector_record_id must be globally contiguous")
            try:
                self._queue.put_nowait(accepted_record)
            except queue.Full as exc:
                error = LandingQueueFull("landing writer queue is full")
                self._fault_writer(error)
                raise error from exc
            self._accepted_count += 1
            self._accepted_frontier = record.collector_record_id
            progress = self._channel_progress[accepted_record.channel]
            progress.accepted_count += 1
            progress.accepted_high_watermark = accepted_record.collector_record_id
            progress.session_id = accepted_record.session_id
            if accepted_record.source_sequence is not None:
                progress.last_source_sequence = accepted_record.source_sequence
            if accepted_record.packet_sequence is not None:
                progress.last_packet_sequence = accepted_record.packet_sequence
            if fence is not None:
                fence.accepted_count += 1
                fence.last_accepted_sequence = accepted_record.source_sequence
        return record.collector_record_id

    append = submit

    def checkpoint(self) -> Mapping[str, Any]:
        control = _Control("checkpoint")
        with self._admission_lock:
            self._require_running()
            self._put_control(control)
        return self._wait_control(control)

    def stop(
        self, *, source_fences: Iterable[SourceFence | Mapping[str, Any]] = ()
    ) -> LandingSeal:
        normalized_fences = tuple(
            self._normalize_source_fence(value) for value in source_fences
        )
        identities = {
            (value.source_id, value.session_id) for value in normalized_fences
        }
        if len(identities) != len(normalized_fences):
            raise ValueError("STOP source fences must be unique")
        with self._admission_lock:
            self._require_open()
            for normalized in normalized_fences:
                existing = self._fences.get(
                    (normalized.source_id, normalized.session_id)
                )
                if existing is None:
                    raise ValueError("STOP fence was not bound in required_sources")
                if (
                    normalized.start_sequence_exclusive
                    != existing.fence.start_sequence_exclusive
                ):
                    raise ValueError(
                        "STOP source fence changed start_sequence_exclusive"
                    )
                if normalized.end_sequence_inclusive != existing.last_accepted_sequence:
                    raise ValueError(
                        "STOP end_sequence_inclusive must equal the accepted watermark"
                    )
            with self._state_lock:
                self._state = "STOPPING"
            try:
                self._write_manifest("STOPPING")
            except BaseException as exc:
                with self._state_lock:
                    self._state = "FAULTED"
                    self._fault = exc
                raise LandingFaulted("failed to persist STOPPING") from exc
            for normalized in normalized_fences:
                key = (normalized.source_id, normalized.session_id)
                existing = self._fences.get(key)
                assert existing is not None
                existing.fence = normalized
        control = _Control("stop")
        with self._admission_lock:
            self._put_control(control)
        seal = self._wait_control(control)
        assert isinstance(seal, LandingSeal)
        return seal

    def fault(self, reason: str | BaseException) -> None:
        error = reason if isinstance(reason, BaseException) else LandingFaulted(reason)
        self._fault_writer(error)

    def close(self) -> None:
        state = self.state
        if state == "OPEN":
            self.fault("landing writer closed without STOP")
        if self._thread is not None:
            self._thread.join(timeout=self.shutdown_timeout_sec)
            if self._thread.is_alive():
                raise LandingFaulted(
                    "landing writer did not stop within shutdown timeout"
                )

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc is None and self.state == "OPEN":
            self.stop()
        elif self.state in {"OPEN", "STOPPING"}:
            self.fault(exc or "landing context aborted")
        self.close()

    def _create_layout(self) -> None:
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        if any(
            path.exists()
            for path in (
                self.partial_path,
                self.sealed_path,
                self.checkpoint_path,
                self.manifest_path,
            )
        ):
            raise LandingStateError(
                "landing artifacts already exist; append is forbidden"
            )
        self.landing_dir.mkdir(exist_ok=True)
        _fsync_directory(self.episode_dir, self._fsync)

    def _manifest(self, state: str) -> Mapping[str, Any]:
        return validate_inprogress_manifest(
            {
                "attempt": str(self.attempt),
                "collection_mode": self.collection_mode,
                "created_time_ns": str(self.created_time_ns),
                "episode_id": self.episode_id,
                "format": "robo_collector.mcap_landing",
                "format_version": 1,
                "landing": {
                    "checkpoint_path": "landing/checkpoints.bin",
                    "partial_path": "landing/episode.mcap.partial",
                    "writer_profile": MCAP_PROFILE,
                },
                "required_sources": [
                    {
                        "session_id": item.session_id,
                        "source_id": item.source_id,
                        "start_sequence_exclusive": str(item.start_sequence_exclusive),
                    }
                    for item in self._required_sources
                ],
                "state": state,
                "writer": {
                    "mcap_library": "mcap",
                    "mcap_version": "1.4.0",
                    "profile": MCAP_PROFILE,
                },
            }
        )

    def _write_manifest(self, state: str) -> None:
        _atomic_write(
            self.manifest_path,
            encode_inprogress_manifest(self._manifest(state)),
            fsync=self._fsync,
            replace=self._replace,
        )

    def _write_initial_manifest(self) -> None:
        _exclusive_write(
            self.manifest_path,
            encode_inprogress_manifest(self._manifest("OPEN")),
            fsync=self._fsync,
        )

    def _run(self) -> None:
        try:
            self._stream = open(self.partial_path, "xb")  # noqa: SIM115
            self._journal = open(self.checkpoint_path, "xb")  # noqa: SIM115
            _fsync_directory(self.landing_dir, self._fsync)
            self._writer = self._writer_factory(
                self._stream,
                use_chunking=False,
                enable_crcs=True,
                enable_data_crcs=True,
                use_statistics=True,
                use_summary_offsets=True,
            )
            self._writer.start(profile=MCAP_PROFILE, library=MCAP_LIBRARY)
            for topic in sorted(self._channels, key=lambda value: value.encode()):
                channel = self._channels[topic]
                schema_id = self._writer.register_schema(
                    channel.schema_name,
                    channel.schema_encoding,
                    bytes(channel.schema_data),
                )
                self._channel_ids[topic] = self._writer.register_channel(
                    topic,
                    channel.message_encoding,
                    schema_id,
                    dict(channel.metadata),
                )
            self._commit_checkpoint()
            self._ready.set()
            while True:
                if self.state == "FAULTED":
                    return
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if isinstance(item, LandingRecord):
                        self._write_record(item)
                    elif item.action == "checkpoint":
                        item.result = self._commit_checkpoint()
                        item.event.set()
                    elif item.action == "stop":
                        item.result = self._finish_and_seal()
                        item.event.set()
                        return
                except BaseException as exc:
                    if isinstance(item, _Control):
                        item.error = exc
                        item.event.set()
                    raise
                finally:
                    self._queue.task_done()
        except BaseException as exc:  # noqa: BLE001
            self._fault_writer(exc, from_writer_thread=True)
        finally:
            self._ready.set()
            for stream in (self._journal, self._stream):
                if stream is not None and not stream.closed:
                    stream.close()

    def _write_record(self, record: LandingRecord) -> None:
        self._writer.add_message(
            self._channel_ids[record.channel],
            log_time=record.log_time,
            publish_time=record.publish_time,
            sequence=record.sequence,
            data=bytes(record.data),
        )
        self._written_count += 1
        self._written_frontier = record.collector_record_id
        progress = self._channel_progress[record.channel]
        progress.written_count += 1
        progress.written_high_watermark = record.collector_record_id
        fence = self._fences.get((record.source_id, record.session_id))
        if fence is not None:
            fence.written_count += 1
            fence.written_high_watermark = record.source_sequence
        due_count = (
            self._written_count - self._durable_count >= self.max_unsynced_records
        )
        due_time = (
            self.durability_interval_sec > 0
            and self._monotonic() - self._last_checkpoint_time
            >= self.durability_interval_sec
        )
        if due_count or due_time:
            self._commit_checkpoint()

    def _commit_checkpoint(self) -> Mapping[str, Any]:
        assert self._stream is not None and self._journal is not None
        self._stream.flush()
        self._fsync(self._stream.fileno())
        offset = self._stream.tell()
        prefix_sha256 = _sha256_prefix(self.partial_path, offset)
        with self._admission_lock:
            accepted_count = self._accepted_count
            accepted_frontier = self._accepted_frontier
            durable_count = self._written_count
            durable_frontier = self._written_frontier
            for progress in self._channel_progress.values():
                progress.durable_count = progress.written_count
                progress.durable_high_watermark = progress.written_high_watermark
            for progress in self._fences.values():
                progress.durable_count = progress.written_count
                progress.durable_high_watermark = progress.written_high_watermark
        checkpoint = self._checkpoint_payload(
            accepted_count=accepted_count,
            accepted_frontier=accepted_frontier,
            durable_count=durable_count,
            durable_frontier=durable_frontier,
            offset=offset,
            prefix_sha256=prefix_sha256,
        )
        frame = encode_checkpoint_frame(checkpoint)
        self._journal.write(frame)
        self._journal.flush()
        self._fsync(self._journal.fileno())
        self._durable_count = durable_count
        self._durable_frontier = durable_frontier
        self._last_checkpoint = checkpoint
        self._checkpoint_sequence += 1
        self._last_checkpoint_time = self._monotonic()
        return checkpoint

    def _checkpoint_payload(
        self,
        *,
        accepted_count: int,
        accepted_frontier: int | None,
        durable_count: int,
        durable_frontier: int | None,
        offset: int,
        prefix_sha256: str,
    ) -> Mapping[str, Any]:
        channels = []
        for topic in sorted(self._channel_progress, key=lambda value: value.encode()):
            value = self._channel_progress[topic]
            channels.append(
                {
                    "accepted_count": str(value.accepted_count),
                    "accepted_high_watermark": _decimal_or_none(
                        value.accepted_high_watermark
                    ),
                    "channel": topic,
                    "durable_count": str(value.durable_count),
                    "durable_high_watermark": _decimal_or_none(
                        value.durable_high_watermark
                    ),
                    "last_packet_sequence": _decimal_or_none(
                        value.last_packet_sequence
                    ),
                    "last_source_sequence": _decimal_or_none(
                        value.last_source_sequence
                    ),
                    "session_id": value.session_id,
                    "written_count": str(value.written_count),
                    "written_high_watermark": _decimal_or_none(
                        value.written_high_watermark
                    ),
                }
            )
        fences = []
        for key in sorted(
            self._fences,
            key=lambda item: (item[0].encode(), item[1].encode()),
        ):
            value = self._fences[key]
            fences.append(
                {
                    "accepted_count": str(value.accepted_count),
                    "durable_count": str(value.durable_count),
                    "durable_high_watermark": _decimal_or_none(
                        value.durable_high_watermark
                    ),
                    "end_sequence_inclusive": _decimal_or_none(
                        value.fence.end_sequence_inclusive
                    ),
                    "session_id": value.fence.session_id,
                    "source_id": value.fence.source_id,
                    "start_sequence_exclusive": str(
                        value.fence.start_sequence_exclusive
                    ),
                    "written_count": str(value.written_count),
                    "written_high_watermark": _decimal_or_none(
                        value.written_high_watermark
                    ),
                }
            )
        return validate_checkpoint_payload(
            {
                "accepted_snapshot_count": str(accepted_count),
                "accepted_snapshot_frontier": _decimal_or_none(accepted_frontier),
                "channels": channels,
                "checkpoint_sequence": str(self._checkpoint_sequence),
                "durable_byte_offset": str(offset),
                "durable_count": str(durable_count),
                "durable_frontier": _decimal_or_none(durable_frontier),
                "format": "robo_collector.mcap_checkpoint",
                "format_version": 1,
                "generation": str(self.generation),
                "landing_prefix_sha256": prefix_sha256,
                "max_unsynced_records": str(self.max_unsynced_records),
                "queue_capacity": str(self.queue_capacity),
                "source_fences": fences,
                "written_count": str(self._written_count),
                "written_frontier": _decimal_or_none(self._written_frontier),
            },
            previous=self._last_checkpoint,
        )

    def _finish_and_seal(self) -> LandingSeal:
        final_checkpoint = self._commit_checkpoint()
        if not (
            self._accepted_count == self._written_count == self._durable_count
            and self._accepted_frontier
            == self._written_frontier
            == self._durable_frontier
        ):
            raise LandingError(
                "STOP cannot seal unequal accepted/written/durable state"
            )
        source_complete = _checkpoint_source_complete(final_checkpoint)
        if not source_complete:
            raise LandingError("STOP source fences are incomplete")
        self._writer.finish()
        assert self._stream is not None
        self._stream.flush()
        self._fsync(self._stream.fileno())
        self._stream.close()
        _install_no_replace(self.partial_path, self.sealed_path, fsync=self._fsync)
        with self._state_lock:
            self._state = "SEALED"
        return LandingSeal(
            self.sealed_path,
            self.checkpoint_path,
            self.manifest_path,
            final_checkpoint,
            source_complete,
        )

    def _put_control(self, control: _Control) -> None:
        deadline = self._monotonic() + self.shutdown_timeout_sec
        while True:
            self._raise_if_faulted()
            if control.action == "checkpoint" and self.state != "OPEN":
                raise LandingStateError("checkpoint admission closed by STOP barrier")
            if self._thread is not None and not self._thread.is_alive():
                error = LandingFaulted("landing worker exited")
                self._fault_writer(error)
                raise error
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                error = LandingFaulted(f"timed out admitting {control.action} control")
                self._fault_writer(error)
                raise error
            try:
                self._queue.put(control, timeout=min(0.05, remaining))
                return
            except queue.Full:
                continue

    def _wait_control(self, control: _Control) -> Any:
        deadline = self._monotonic() + self.shutdown_timeout_sec
        while not control.event.wait(min(0.05, max(0.0, deadline - self._monotonic()))):
            self._raise_if_faulted()
            if self.state in {"SEALED", "FAULTED"} or (
                self._thread is not None and not self._thread.is_alive()
            ):
                error = LandingFaulted("control orphaned by terminal worker")
                self._drain_controls(error)
                raise error
            if self._monotonic() >= deadline:
                error = LandingFaulted(f"timed out waiting for {control.action} control")
                self._fault_writer(error)
                self._drain_controls(error)
                raise error
        if control.error is not None:
            raise LandingFaulted(str(control.error)) from control.error
        return control.result

    def _drain_controls(self, error: BaseException) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, _Control):
                item.error = error
                item.event.set()
            self._queue.task_done()

    def _fault_writer(
        self, error: BaseException, *, from_writer_thread: bool = False
    ) -> None:
        first_fault = False
        with self._state_lock:
            if self._state not in {"FAULTED", "SEALED"}:
                self._state = "FAULTED"
                self._fault = error
                first_fault = True
        if first_fault:
            try:
                self._write_manifest("FAULTED")
            except BaseException as manifest_error:  # noqa: BLE001
                if self._fault is error:
                    self._fault = LandingFaulted(
                        f"{error}; additionally failed to persist FAULTED: "
                        f"{manifest_error}"
                    )
        if not from_writer_thread and self._thread is not None:
            self._thread.join(timeout=min(1.0, self.shutdown_timeout_sec))

    def _raise_if_faulted(self) -> None:
        if self.state == "FAULTED":
            raise LandingFaulted(
                str(self._fault or "landing writer faulted")
            ) from self._fault

    def _require_open(self) -> None:
        state = self.state
        if state == "FAULTED":
            self._raise_if_faulted()
        if state != "OPEN":
            raise LandingStateError(f"landing writer is not OPEN: {state}")

    def _require_running(self) -> None:
        state = self.state
        if state == "FAULTED":
            self._raise_if_faulted()
        if state not in {"OPEN", "STOPPING"}:
            raise LandingStateError(f"landing writer is not running: {state}")

    @staticmethod
    def _normalize_required_sources(
        values: Iterable[RequiredSource | Mapping[str, Any]],
    ) -> Iterable[RequiredSource]:
        normalized = []
        for value in values:
            item = (
                value if isinstance(value, RequiredSource) else RequiredSource(**value)
            )
            _nonnegative(item.start_sequence_exclusive, "start_sequence_exclusive")
            normalized.append(item)
        normalized.sort(
            key=lambda item: (item.source_id.encode(), item.session_id.encode())
        )
        if len({(item.source_id, item.session_id) for item in normalized}) != len(
            normalized
        ):
            raise ValueError("required sources must be unique")
        return normalized

    @staticmethod
    def _normalize_source_fence(
        value: SourceFence | Mapping[str, Any],
    ) -> SourceFence:
        item = value if isinstance(value, SourceFence) else SourceFence(**value)
        _nonnegative(item.start_sequence_exclusive, "start_sequence_exclusive")
        if item.end_sequence_inclusive is not None:
            _nonnegative(item.end_sequence_inclusive, "end_sequence_inclusive")
            if item.end_sequence_inclusive <= item.start_sequence_exclusive:
                raise ValueError("source fence end must follow its start")
        return item


LandingRecorder = LandingWriter


def _decimal_or_none(value: int | None) -> str | None:
    return None if value is None else str(value)


def read_checkpoint_journal(path: str | os.PathLike[str]) -> JournalScan:
    """Read consecutive frames, tolerating only a physically torn final frame."""

    raw = Path(path).read_bytes()
    checkpoints: list[Mapping[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for checkpoint in iter_checkpoint_frames(raw, allow_torn_final=True):
        checkpoint = validate_checkpoint_payload(checkpoint, previous=previous)
        checkpoints.append(checkpoint)
        previous = checkpoint
    complete_length = sum(len(encode_checkpoint_frame(item)) for item in checkpoints)
    return JournalScan(tuple(checkpoints), complete_length != len(raw))


def select_durable_prefix(
    partial_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
) -> DurablePrefix:
    """Select the newest verified checkpoint without mutating either source."""

    source = Path(partial_path)
    journal = Path(checkpoint_path)
    scan = read_checkpoint_journal(journal)
    if not scan.checkpoints:
        raise RecoveryError("no durable checkpoint")
    size = source.stat().st_size
    selected: Mapping[str, Any] | None = None
    for checkpoint in scan.checkpoints:
        offset = int(checkpoint["durable_byte_offset"])
        if offset > size:
            raise RecoveryError("checkpoint durable offset is beyond landing EOF")
        _validate_durable_prefix(source, offset)
        actual = _sha256_prefix(source, offset)
        if actual != checkpoint["landing_prefix_sha256"]:
            raise RecoveryError("landing durable prefix digest mismatch")
        selected = checkpoint
    assert selected is not None
    source_stat = source.stat()
    return DurablePrefix(
        source,
        journal,
        int(selected["durable_byte_offset"]),
        str(selected["landing_prefix_sha256"]),
        selected,
        scan.torn_final,
        (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns),
    )


def _validate_durable_prefix(source: Path, length: int) -> None:
    """Reject torn records and opcodes the recovery reader cannot replay."""
    if length < len(b"\x89MCAP0\r\n"):
        raise RecoveryError("durable prefix is shorter than MCAP magic")
    known_opcodes = {0x01, 0x03, 0x04, 0x05}
    with source.open("rb") as stream:
        data = stream.read(length)
    if len(data) != length or not data.startswith(b"\x89MCAP0\r\n"):
        raise RecoveryError("durable prefix has invalid MCAP magic")
    offset = len(b"\x89MCAP0\r\n")
    while offset < length:
        if length - offset < 9:
            raise RecoveryError("durable prefix ends mid-record header")
        opcode = data[offset]
        content_length = struct.unpack_from("<Q", data, offset + 1)[0]
        if opcode not in known_opcodes:
            raise RecoveryError(f"unsupported MCAP opcode: {opcode:#x}")
        end = offset + 9 + content_length
        if end > length:
            raise RecoveryError("durable prefix ends mid-record")
        offset = end


def recover_landing(
    episode_dir: str | os.PathLike[str],
    *,
    attempt: int,
    writer_factory: Callable[..., Any] | None = None,
    fsync: Callable[[int], None] = os.fsync,
    replace: Callable[
        [str | os.PathLike[str], str | os.PathLike[str]], None
    ] = os.replace,
) -> RecoveryResult:
    """Replay a trusted durable prefix into a new immutable sealed MCAP.

    The original partial, journal, in-progress manifest, and camera Raw Spool
    are read-only inputs.  No recovery path appends to a source artifact.
    """

    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise ValueError("attempt must be a positive integer")
    root = Path(os.path.abspath(os.fspath(episode_dir)))
    _validate_recovery_path(root, "episode root")
    del replace  # Retained for source compatibility; installs are always no-replace.
    manifest_path = root / "manifest.inprogress.json"
    partial_path = root / "landing" / "episode.mcap.partial"
    checkpoint_path = root / "landing" / "checkpoints.bin"
    output_partial = root / "landing" / f"recovery-{attempt}.mcap.partial"
    output_sealed = root / "landing" / f"recovery-{attempt}.mcap"
    if output_partial.exists() and not output_sealed.exists():
        return RecoveryResult(
            RecoveryExitCode.IO_OR_INTERNAL_ERROR,
            detail="stranded recovery partial exists; overwrite is forbidden",
        )
    try:
        decode_inprogress_manifest(manifest_path.read_bytes())
        source_path = (
            partial_path if partial_path.exists() else root / "landing" / "episode.mcap"
        )
        prefix = select_durable_prefix(source_path, checkpoint_path)
    except FileNotFoundError as exc:
        return RecoveryResult(RecoveryExitCode.NO_DURABLE_PREFIX, detail=str(exc))
    except CheckpointFrameError as exc:
        return RecoveryResult(
            RecoveryExitCode.CORRUPT_PREFIX_OR_JOURNAL, detail=str(exc)
        )
    except RecoveryError as exc:
        code = (
            RecoveryExitCode.NO_DURABLE_PREFIX
            if str(exc) == "no durable checkpoint"
            else RecoveryExitCode.CORRUPT_PREFIX_OR_JOURNAL
        )
        return RecoveryResult(code, detail=str(exc))
    recovery_writer_factory = writer_factory or _load_mcap_writer()
    try:
        if output_sealed.exists():
            if not _adopt_recovery_output(
                prefix,
                output_partial,
                output_sealed,
                writer_factory=recovery_writer_factory,
                fsync=fsync,
            ):
                return RecoveryResult(
                    RecoveryExitCode.IO_OR_INTERNAL_ERROR,
                    detail="existing recovery output does not match durable prefix",
                )
            complete = _checkpoint_source_complete(prefix.checkpoint)
            code = (
                RecoveryExitCode.RECOVERED
                if complete
                else RecoveryExitCode.SOURCE_INCOMPLETE
            )
            return RecoveryResult(code, output_sealed, prefix, complete)
        _replay_prefix(
            prefix,
            output_partial,
            writer_factory=recovery_writer_factory,
            fsync=fsync,
        )
        _install_no_replace(output_partial, output_sealed, fsync=fsync)
        complete = _checkpoint_source_complete(prefix.checkpoint)
        code = (
            RecoveryExitCode.RECOVERED
            if complete
            else RecoveryExitCode.SOURCE_INCOMPLETE
        )
        return RecoveryResult(code, output_sealed, prefix, complete)
    except BaseException as exc:  # noqa: BLE001
        try:
            if output_partial.exists() and not output_sealed.exists():
                output_partial.unlink()
                _fsync_directory(output_partial.parent, fsync)
        except OSError:
            pass
        return RecoveryResult(RecoveryExitCode.IO_OR_INTERNAL_ERROR, detail=str(exc))


def _validate_recovery_path(path: Path, label: str) -> None:
    """Reject symlink ancestors and non-directory ancestors before any output."""
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RecoveryError(f"cannot inspect {label}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise RecoveryError(f"{label} contains a symlink component: {current}")
        if current != path and not stat.S_ISDIR(mode):
            raise RecoveryError(f"{label} has a non-directory ancestor: {current}")


def _adopt_recovery_output(
    prefix: DurablePrefix,
    output_partial: Path,
    output_sealed: Path,
    *,
    writer_factory: Callable[..., Any],
    fsync: Callable[[int], None],
) -> bool:
    """Validate and adopt an immutable same-attempt recovery output."""

    if (
        not output_sealed.is_file()
        or output_sealed.is_symlink()
        or (output_partial.exists() and output_partial.is_symlink())
    ):
        return False
    created_scratch = not output_partial.exists()
    if created_scratch:
        _replay_prefix(
            prefix,
            output_partial,
            writer_factory=writer_factory,
            fsync=fsync,
        )
    if not output_partial.is_file():
        return False
    matches = (
        output_partial.stat().st_size == output_sealed.stat().st_size
        and _sha256_file(output_partial) == _sha256_file(output_sealed)
    )
    if not matches:
        if created_scratch:
            output_partial.unlink()
            _fsync_directory(output_partial.parent, fsync)
        return False
    _fsync_directory(output_sealed.parent, fsync)
    output_partial.unlink()
    try:
        _fsync_directory(output_sealed.parent, fsync)
    except OSError:
        pass
    return True


def _replay_prefix(
    prefix: DurablePrefix,
    destination: Path,
    *,
    writer_factory: Callable[..., Any],
    fsync: Callable[[int], None],
) -> None:
    StreamReader, record_types, EndOfFile = _load_stream_types()
    Header, Schema, Channel, Message = record_types
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(prefix.source_path, flags)
    with os.fdopen(descriptor, "rb") as source:
        initial_stat = os.fstat(source.fileno())
        expected_stat = prefix.source_stat
        if expected_stat is not None and (
            initial_stat.st_dev, initial_stat.st_ino, initial_stat.st_size, initial_stat.st_mtime_ns
        ) != expected_stat:
            raise RecoveryError("landing source changed before replay")
        bounded = _BoundedReader(source, prefix.byte_offset)
        reader = StreamReader(
            bounded,
            validate_crcs=True,
            record_size_limit=MAX_MCAP_RECORD_BYTES,
        )
        with open(destination, "xb") as output:
            writer = writer_factory(
                output,
                use_chunking=False,
                enable_crcs=True,
                enable_data_crcs=True,
                use_statistics=True,
                use_summary_offsets=True,
            )
            schema_ids: dict[int, int] = {}
            channel_ids: dict[int, int] = {}
            schemas: dict[int, tuple[str, str, bytes]] = {}
            channels: dict[int, tuple[str, str, int, tuple[tuple[str, str], ...]]] = {}
            started = False
            try:
                for record in reader.records:
                    if isinstance(record, Header):
                        if started:
                            raise RecoveryError(
                                "duplicate MCAP header in durable prefix"
                            )
                        if record.profile != MCAP_PROFILE:
                            raise RecoveryError("unexpected MCAP profile")
                        if record.library != MCAP_LIBRARY:
                            raise RecoveryError("unexpected MCAP library")
                        writer.start(profile=record.profile, library=record.library)
                        started = True
                    elif isinstance(record, Schema):
                        if not started:
                            raise RecoveryError("schema precedes MCAP header")
                        signature = (record.name, record.encoding, record.data)
                        if record.id in schemas:
                            if schemas[record.id] != signature:
                                raise RecoveryError("divergent duplicate schema id")
                            continue
                        schemas[record.id] = signature
                        schema_ids[record.id] = writer.register_schema(
                            record.name, record.encoding, record.data
                        )
                    elif isinstance(record, Channel):
                        if not started:
                            raise RecoveryError("channel precedes MCAP header")
                        if record.schema_id not in schema_ids:
                            raise RecoveryError("channel references unknown schema")
                        signature = (
                            record.topic,
                            record.message_encoding,
                            record.schema_id,
                            tuple(sorted(record.metadata.items())),
                        )
                        if record.id in channels:
                            if channels[record.id] != signature:
                                raise RecoveryError("divergent duplicate channel id")
                            continue
                        channels[record.id] = signature
                        channel_ids[record.id] = writer.register_channel(
                            record.topic,
                            record.message_encoding,
                            schema_ids[record.schema_id],
                            dict(record.metadata),
                        )
                    elif isinstance(record, Message):
                        if len(record.data) > MAX_MCAP_RECORD_BYTES:
                            raise RecoveryError("recovery message exceeds 64 MiB")
                        if record.channel_id not in channel_ids:
                            raise RecoveryError("message references unknown channel")
                        writer.add_message(
                            channel_ids[record.channel_id],
                            log_time=record.log_time,
                            publish_time=record.publish_time,
                            sequence=record.sequence,
                            data=record.data,
                        )
            except EndOfFile:
                pass
            if not started:
                raise RecoveryError("durable prefix has no MCAP header")
            writer.finish()
            output.flush()
            fsync(output.fileno())
        final_stat = os.fstat(source.fileno())
        if (
            initial_stat.st_dev, initial_stat.st_ino, initial_stat.st_size, initial_stat.st_mtime_ns
        ) != (
            final_stat.st_dev, final_stat.st_ino, final_stat.st_size, final_stat.st_mtime_ns
        ):
            raise RecoveryError("landing source changed during replay")


class _BoundedReader:
    """Read-only view that exposes exactly one trusted prefix."""

    def __init__(self, stream: BinaryIO, length: int) -> None:
        self._stream = stream
        self._remaining = length

    def read(self, size: int = -1) -> bytes:
        if self._remaining == 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        data = self._stream.read(size)
        self._remaining -= len(data)
        return data

    def seekable(self) -> bool:
        return False


def _checkpoint_source_complete(checkpoint: Mapping[str, Any]) -> bool:
    if not (
        checkpoint["accepted_snapshot_count"]
        == checkpoint["written_count"]
        == checkpoint["durable_count"]
    ):
        return False
    if not (
        checkpoint["accepted_snapshot_frontier"]
        == checkpoint["written_frontier"]
        == checkpoint["durable_frontier"]
    ):
        return False
    for fence in checkpoint["source_fences"]:
        end = fence["end_sequence_inclusive"]
        if end is None:
            return False
        if not (
            fence["accepted_count"] == fence["written_count"] == fence["durable_count"]
        ):
            return False
        durable = fence["durable_high_watermark"]
        written = fence["written_high_watermark"]
        if durable is None or written is None:
            return False
        if int(durable) != int(written) or int(durable) != int(end):
            return False
    return True


__all__ = [
    "DurablePrefix",
    "JournalScan",
    "LandingChannel",
    "LandingError",
    "LandingFaulted",
    "LandingQueueFull",
    "LandingRecord",
    "LandingRecorder",
    "LandingSeal",
    "LandingStateError",
    "LandingWriter",
    "RecoveryError",
    "RecoveryExitCode",
    "RecoveryResult",
    "RequiredSource",
    "SourceFence",
    "read_checkpoint_journal",
    "recover_landing",
    "select_durable_prefix",
]
