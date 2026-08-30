"""Format-neutral, fail-closed capture fanout coordination.

The coordinator deliberately knows nothing about Raw v1 or MCAP serialization.
Sink adapters own those details; this module owns admission, ordering, failure
containment, and the terminal evidence needed to compare compatibility captures.
"""

from __future__ import annotations

import queue
import threading
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

_UINT64_MAX = (1 << 64) - 1


class CaptureError(RuntimeError):
    """Base error raised by the capture coordinator."""


class CaptureAdmissionClosed(CaptureError):
    """Raised when a producer tries to submit after admission closed."""


class CaptureQueueFull(CaptureError):
    """Raised when bounded admission cannot accept another record."""


class CaptureMode(str, Enum):
    RAW_V1 = "raw_v1"
    DUAL_WRITE = "dual_write"
    MCAP_FIRST = "mcap_first"

    @classmethod
    def parse(cls, value: str | CaptureMode) -> CaptureMode:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        if normalized == "raw_first":
            warnings.warn(
                "recording mode 'raw_first' is deprecated; use 'raw_v1'",
                DeprecationWarning,
                stacklevel=2,
            )
            normalized = cls.RAW_V1.value
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"capture mode must be one of: {choices}") from exc


class SinkDisposition(str, Enum):
    """Per-record sink state, including all allowed terminal dispositions."""

    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    ACCEPTED = "ACCEPTED"
    WRITTEN = "WRITTEN"
    DURABLE = "DURABLE"
    FAILED = "FAILED"
    NOT_ATTEMPTED_AFTER_SINK_FAILURE = "NOT_ATTEMPTED_AFTER_SINK_FAILURE"
    LOST_ON_CLOSE = "LOST_ON_CLOSE"


_RUNTIME_DISPOSITIONS = {
    SinkDisposition.NOT_ATTEMPTED,
    SinkDisposition.ACCEPTED,
    SinkDisposition.WRITTEN,
}
_SUBMIT_DISPOSITIONS = {
    SinkDisposition.ACCEPTED,
    SinkDisposition.WRITTEN,
    SinkDisposition.DURABLE,
}


class CaptureStatus(str, Enum):
    READY = "READY"
    FAILED = "FAILED"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True, order=True)
class CaptureSourceFence:
    """The START/STOP source boundary handed unchanged to every sink adapter."""

    source_id: str
    session_id: str
    start_sequence_exclusive: int
    end_sequence_inclusive: int | None

    def __post_init__(self) -> None:
        if not self.source_id or not self.session_id:
            raise ValueError("source fence identity must be non-empty")
        _validate_uint64(self.start_sequence_exclusive, "start_sequence_exclusive")
        if self.end_sequence_inclusive is not None:
            _validate_uint64(self.end_sequence_inclusive, "end_sequence_inclusive")
            if self.end_sequence_inclusive <= self.start_sequence_exclusive:
                raise ValueError("source fence end must follow its start")


@dataclass(frozen=True)
class CaptureEnvelope:
    """Immutable coordinator envelope shared by all enabled sinks."""

    collector_record_id: int
    payload: Any
    source_id: str = ""
    session_id: str = ""
    source_sequence: int | None = None
    packet_sequence: int | None = None

    def __post_init__(self) -> None:
        _validate_uint64(self.collector_record_id, "collector_record_id")
        for value, name in (
            (self.source_sequence, "source_sequence"),
            (self.packet_sequence, "packet_sequence"),
        ):
            if value is not None:
                _validate_uint64(value, name)


@runtime_checkable
class CaptureSink(Protocol):
    """Adapter contract for a Raw v1 or MCAP landing sink.

    ``submit`` returns the strongest state synchronously proven for the record.
    ``seal`` must not return until every successfully submitted record is
    durable, otherwise it raises. ``close`` releases resources and is attempted
    even after submission or seal failure.
    """

    def submit(self, envelope: CaptureEnvelope) -> SinkDisposition | str | None: ...

    def seal(self, source_fences: tuple[CaptureSourceFence, ...]) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CaptureResult:
    mode: CaptureMode
    status: CaptureStatus
    terminal_accepted_frontier: int | None
    dispositions: Mapping[int, Mapping[str, SinkDisposition]]
    errors: tuple[str, ...]

    @property
    def success(self) -> bool:
        return self.status is CaptureStatus.READY


class CaptureCoordinator:
    """Single-owner bounded fanout with deterministic Raw-then-MCAP dispatch."""

    _SINK_ORDER = ("raw", "mcap")

    def __init__(
        self,
        mode: str | CaptureMode,
        *,
        raw_sink: CaptureSink | None = None,
        mcap_sink: CaptureSink | None = None,
        queue_capacity: int = 128,
        stop_timeout_sec: float = 5.0,
    ) -> None:
        self.mode = CaptureMode.parse(mode)
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self.queue_capacity = int(queue_capacity)
        if isinstance(stop_timeout_sec, bool) or stop_timeout_sec <= 0:
            raise ValueError("stop_timeout_sec must be positive")
        self.stop_timeout_sec = float(stop_timeout_sec)
        self._sinks = {"raw": raw_sink, "mcap": mcap_sink}
        required = self._required_sink_names(self.mode)
        missing = [name for name in required if self._sinks[name] is None]
        if missing:
            raise ValueError(
                f"capture mode {self.mode.value} requires {missing[0]}_sink"
            )
        self._enabled = required
        self._queue: queue.Queue[CaptureEnvelope | object] = queue.Queue(
            maxsize=self.queue_capacity
        )
        self._admission_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._started = False
        self._stopped = False
        self._admission_closed = False
        self._next_record_id = 0
        self._terminal_accepted_frontier: int | None = None
        self._dispositions: dict[int, dict[str, SinkDisposition]] = {}
        self._sink_failed = {name: False for name in self._enabled}
        self._sink_locks = {name: threading.RLock() for name in self._enabled}
        self._errors: list[str] = []
        self._stop_requested = threading.Event()
        self._owner = threading.Thread(
            target=self._run,
            name="capture-fanout",
            daemon=True,
        )

    @staticmethod
    def _required_sink_names(mode: CaptureMode) -> tuple[str, ...]:
        if mode is CaptureMode.RAW_V1:
            return ("raw",)
        if mode is CaptureMode.MCAP_FIRST:
            return ("mcap",)
        return ("raw", "mcap")

    @property
    def admission_closed(self) -> bool:
        with self._admission_lock:
            return self._admission_closed

    @property
    def terminal_accepted_frontier(self) -> int | None:
        with self._admission_lock:
            return self._terminal_accepted_frontier

    def start(self) -> CaptureCoordinator:
        with self._state_lock:
            if self._started:
                raise CaptureError("capture coordinator is already started")
            self._started = True
            self._owner.start()
        return self

    def submit(
        self,
        payload: Any,
        *,
        source_id: str = "",
        session_id: str = "",
        source_sequence: int | None = None,
        packet_sequence: int | None = None,
    ) -> int:
        """Atomically assign an ID and admit one immutable envelope.

        A full queue faults the capture but does not consume an ID. Later calls
        are rejected without assigning IDs.
        """

        with self._admission_lock:
            if not self._started:
                raise CaptureAdmissionClosed("capture coordinator is not started")
            if self._admission_closed:
                raise CaptureAdmissionClosed("capture admission is closed")
            record_id = self._next_record_id
            if record_id > _UINT64_MAX:
                self._close_admission_locked("collector_record_id exhausted uint64")
                raise CaptureAdmissionClosed("collector_record_id exhausted uint64")
            envelope = CaptureEnvelope(
                collector_record_id=record_id,
                payload=payload,
                source_id=source_id,
                session_id=session_id,
                source_sequence=source_sequence,
                packet_sequence=packet_sequence,
            )
            # Publish evidence before the queue makes the envelope visible to
            # the fanout owner. Roll it back if bounded admission fails.
            with self._state_lock:
                self._dispositions[record_id] = {
                    name: SinkDisposition.NOT_ATTEMPTED for name in self._enabled
                }
            try:
                self._queue.put_nowait(envelope)
            except queue.Full as exc:
                with self._state_lock:
                    del self._dispositions[record_id]
                self._close_admission_locked("fanout queue is full")
                raise CaptureQueueFull("fanout queue is full") from exc
            self._next_record_id += 1
            return record_id

    def report_sink_failure(
        self,
        sink_name: str,
        error: BaseException | str,
        *,
        collector_record_id: int | None = None,
    ) -> None:
        """Close admission from an asynchronous sink failure callback."""

        if sink_name not in self._enabled:
            raise ValueError(f"sink is not enabled: {sink_name}")
        if collector_record_id is not None:
            _validate_uint64(collector_record_id, "collector_record_id")
        self._mark_sink_failed(sink_name, collector_record_id, error)

    def stop(
        self,
        *,
        source_fences: Iterable[CaptureSourceFence] = (),
    ) -> CaptureResult:
        """Close admission, drain the terminal frontier, seal and close sinks."""

        with self._state_lock:
            if not self._started:
                raise CaptureError("capture coordinator is not started")
            if self._stopped:
                raise CaptureError("capture coordinator is already stopped")
            self._stopped = True
        with self._admission_lock:
            if not self._admission_closed:
                self._close_admission_locked(None)
        fence_error: Exception | None = None
        try:
            fences = self._normalize_fences(source_fences)
        except Exception as exc:  # noqa: BLE001 - malformed boundary must fail closed
            fence_error = exc
            fences = ()

        self._stop_requested.set()
        self._owner.join(timeout=self.stop_timeout_sec)
        owner_stopped = not self._owner.is_alive()
        if not owner_stopped:
            for name in self._enabled:
                self._mark_sink_failed(
                    name,
                    None,
                    f"fanout owner did not stop within {self.stop_timeout_sec:g}s",
                )
            self._terminalize_unresolved()
            return self._result()

        for name in self._SINK_ORDER:
            if name not in self._enabled:
                continue
            sink = self._sinks[name]
            assert sink is not None
            seal_succeeded = False
            if fence_error is None:
                try:
                    sink.seal(fences)
                    seal_succeeded = True
                except Exception as exc:  # noqa: BLE001 - adapter failure boundary
                    self._mark_sink_failed(name, None, f"seal failed: {exc}")
            else:
                self._mark_sink_failed(
                    name, None, f"invalid source fences: {fence_error}"
                )
            if seal_succeeded:
                self._terminalize_sink(name, SinkDisposition.DURABLE)
            else:
                self._terminalize_sink(name, SinkDisposition.LOST_ON_CLOSE)
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001 - adapters define failure types
                self._mark_sink_failed(name, None, f"close failed: {exc}")
                self._terminalize_sink(name, SinkDisposition.LOST_ON_CLOSE)

        self._terminalize_unresolved()
        result = self._result()
        if fence_error is not None:
            raise fence_error
        return result

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                if self._stop_requested.is_set():
                    return
                continue
            try:
                assert isinstance(item, CaptureEnvelope)
                self._dispatch(item)
            finally:
                self._queue.task_done()
            if self._stop_requested.is_set() and self._queue.empty():
                return

    def _dispatch(self, envelope: CaptureEnvelope) -> None:
        for name in self._SINK_ORDER:
            if name not in self._enabled:
                continue
            with self._sink_locks[name]:
                with self._admission_lock:
                    sink_failed = self._sink_failed[name]
                if sink_failed:
                    self._set_disposition(
                        envelope.collector_record_id,
                        name,
                        SinkDisposition.NOT_ATTEMPTED_AFTER_SINK_FAILURE,
                    )
                    continue
                sink = self._sinks[name]
                assert sink is not None
                try:
                    returned = sink.submit(envelope)
                    disposition = (
                        SinkDisposition.ACCEPTED
                        if returned is None
                        else SinkDisposition(returned)
                    )
                    if disposition not in _SUBMIT_DISPOSITIONS:
                        raise ValueError(
                            "sink returned invalid submit disposition "
                            f"{disposition.value}"
                        )
                    self._set_disposition(
                        envelope.collector_record_id, name, disposition
                    )
                except Exception as exc:  # noqa: BLE001 - adapter failure boundary
                    self._mark_sink_failed(name, envelope.collector_record_id, exc)

    def _mark_sink_failed(
        self,
        name: str,
        record_id: int | None,
        error: BaseException | str,
    ) -> None:
        message = str(error) or type(error).__name__
        # Do not wait for an in-flight adapter call before publishing failure:
        # trailing dispatch must observe the closed sink immediately after that
        # call returns. The per-sink lock only serializes actual submissions.
        with self._admission_lock:
            first_failure = not self._sink_failed[name]
            self._sink_failed[name] = True
            if first_failure:
                self._errors.append(f"{name}: {message}")
            if record_id is not None:
                with self._state_lock:
                    if record_id in self._dispositions:
                        self._dispositions[record_id][name] = SinkDisposition.FAILED
            if not self._admission_closed:
                self._close_admission_locked(None)

    def _close_admission_locked(self, error: str | None) -> None:
        self._admission_closed = True
        self._terminal_accepted_frontier = (
            None if self._next_record_id == 0 else self._next_record_id - 1
        )
        if error:
            self._errors.append(error)

    def _set_disposition(
        self, record_id: int, name: str, disposition: SinkDisposition
    ) -> None:
        with self._state_lock:
            current = self._dispositions[record_id][name]
            if current in {
                SinkDisposition.FAILED,
                SinkDisposition.NOT_ATTEMPTED_AFTER_SINK_FAILURE,
                SinkDisposition.LOST_ON_CLOSE,
            }:
                return
            self._dispositions[record_id][name] = disposition

    def _terminalize_sink(self, name: str, terminal: SinkDisposition) -> None:
        eligible = {
            SinkDisposition.ACCEPTED,
            SinkDisposition.WRITTEN,
        }
        if terminal is SinkDisposition.LOST_ON_CLOSE:
            eligible.add(SinkDisposition.DURABLE)
        with self._state_lock:
            for row in self._dispositions.values():
                if row[name] in eligible:
                    row[name] = terminal

    def _terminalize_unresolved(self) -> None:
        with self._state_lock:
            for row in self._dispositions.values():
                for name, disposition in tuple(row.items()):
                    if disposition in _RUNTIME_DISPOSITIONS:
                        row[name] = SinkDisposition.LOST_ON_CLOSE

    def _result(self) -> CaptureResult:
        with self._admission_lock:
            failed = bool(self._errors) or any(self._sink_failed.values())
            terminal_accepted_frontier = self._terminal_accepted_frontier
            errors = tuple(self._errors)
            with self._state_lock:
                frozen_rows = {
                    record_id: MappingProxyType(dict(row))
                    for record_id, row in sorted(self._dispositions.items())
                }
        status = (
            CaptureStatus.QUARANTINED
            if failed and self.mode is CaptureMode.DUAL_WRITE
            else CaptureStatus.FAILED
            if failed
            else CaptureStatus.READY
        )
        return CaptureResult(
            mode=self.mode,
            status=status,
            terminal_accepted_frontier=terminal_accepted_frontier,
            dispositions=MappingProxyType(frozen_rows),
            errors=errors,
        )

    @staticmethod
    def _normalize_fences(
        values: Iterable[CaptureSourceFence],
    ) -> tuple[CaptureSourceFence, ...]:
        fences = tuple(sorted(values))
        identities = [(item.source_id, item.session_id) for item in fences]
        if len(identities) != len(set(identities)):
            raise ValueError("source fences must have unique source/session identities")
        return fences


def _validate_uint64(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value > _UINT64_MAX:
        raise ValueError(f"{name} must fit uint64")
