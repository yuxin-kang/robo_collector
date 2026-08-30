from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest
from robo_collector.capture_coordinator import (
    CaptureAdmissionClosed,
    CaptureCoordinator,
    CaptureEnvelope,
    CaptureError,
    CaptureMode,
    CaptureQueueFull,
    CaptureSourceFence,
    CaptureStatus,
    SinkDisposition,
)


@dataclass
class FakeSink:
    name: str
    events: list[tuple[str, str, int | None]]
    fail_submit_id: int | None = None
    fail_seal: bool = False
    fail_close: bool = False
    submit_state: SinkDisposition | str = SinkDisposition.WRITTEN
    block_id: int | None = None
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    envelopes: list[CaptureEnvelope] = field(default_factory=list)
    fences: tuple[CaptureSourceFence, ...] = ()

    def submit(self, envelope: CaptureEnvelope) -> SinkDisposition:
        self.events.append(("submit", self.name, envelope.collector_record_id))
        self.entered.set()
        if envelope.collector_record_id == self.block_id:
            assert self.release.wait(2.0)
        if envelope.collector_record_id == self.fail_submit_id:
            raise RuntimeError(f"{self.name} submit fault")
        self.envelopes.append(envelope)
        return self.submit_state

    def seal(self, source_fences: tuple[CaptureSourceFence, ...]) -> None:
        self.events.append(("seal", self.name, None))
        self.fences = source_fences
        if self.fail_seal:
            raise RuntimeError(f"{self.name} seal fault")

    def close(self) -> None:
        self.events.append(("close", self.name, None))
        if self.fail_close:
            raise RuntimeError(f"{self.name} close fault")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("raw_v1", CaptureMode.RAW_V1),
        (" DUAL_WRITE ", CaptureMode.DUAL_WRITE),
        (CaptureMode.MCAP_FIRST, CaptureMode.MCAP_FIRST),
    ],
)
def test_capture_mode_parse(value: str | CaptureMode, expected: CaptureMode) -> None:
    assert CaptureMode.parse(value) is expected


def test_raw_first_is_deprecated_raw_v1_alias() -> None:
    with pytest.warns(DeprecationWarning, match="raw_v1"):
        assert CaptureMode.parse("raw_first") is CaptureMode.RAW_V1


def test_invalid_mode_and_missing_required_sink_are_rejected() -> None:
    with pytest.raises(ValueError, match="capture mode"):
        CaptureMode.parse("legacy")
    with pytest.raises(ValueError, match="raw_sink"):
        CaptureCoordinator("raw_v1")
    with pytest.raises(ValueError, match="mcap_sink"):
        CaptureCoordinator("mcap_first")


def test_dual_write_assigns_contiguous_ids_and_dispatches_raw_then_mcap() -> None:
    events: list[tuple[str, str, int | None]] = []
    raw = FakeSink("raw", events)
    mcap = FakeSink("mcap", events)
    coordinator = CaptureCoordinator(
        "dual_write", raw_sink=raw, mcap_sink=mcap, queue_capacity=4
    ).start()

    assert coordinator.submit(b"a", source_sequence=10) == 0
    assert coordinator.submit(b"b", source_sequence=11) == 1
    result = coordinator.stop()

    assert [event for event in events if event[0] == "submit"] == [
        ("submit", "raw", 0),
        ("submit", "mcap", 0),
        ("submit", "raw", 1),
        ("submit", "mcap", 1),
    ]
    assert [item.collector_record_id for item in raw.envelopes] == [0, 1]
    assert raw.envelopes == mcap.envelopes
    assert result.status is CaptureStatus.READY
    assert result.success
    assert result.terminal_accepted_frontier == 1
    assert all(
        disposition is SinkDisposition.DURABLE
        for row in result.dispositions.values()
        for disposition in row.values()
    )


def test_fast_owner_never_observes_record_before_its_evidence_row() -> None:
    events: list[tuple[str, str, int | None]] = []
    raw = FakeSink("raw", events)
    coordinator = CaptureCoordinator("raw_v1", raw_sink=raw, queue_capacity=256).start()

    assert [coordinator.submit(value) for value in range(200)] == list(range(200))
    result = coordinator.stop()

    assert result.status is CaptureStatus.READY
    assert tuple(result.dispositions) == tuple(range(200))


@pytest.mark.parametrize(
    ("mode", "sink_name"),
    [("raw_v1", "raw"), ("mcap_first", "mcap")],
)
def test_single_required_sink_failure_fails_closed(mode: str, sink_name: str) -> None:
    events: list[tuple[str, str, int | None]] = []
    sink = FakeSink(sink_name, events, fail_submit_id=0)
    kwargs = {f"{sink_name}_sink": sink}
    coordinator = CaptureCoordinator(mode, **kwargs).start()
    coordinator.submit("payload")
    assert sink.entered.wait(1.0)

    result = coordinator.stop()

    assert result.status is CaptureStatus.FAILED
    assert not result.success
    assert coordinator.admission_closed
    assert result.dispositions[0][sink_name] is SinkDisposition.FAILED
    assert ("seal", sink_name, None) in events
    assert ("close", sink_name, None) in events


@pytest.mark.parametrize("failing_sink", ["raw", "mcap"])
def test_dual_write_failure_drains_trailing_ids_to_healthy_sink(
    failing_sink: str,
) -> None:
    events: list[tuple[str, str, int | None]] = []
    raw = FakeSink("raw", events, fail_submit_id=0 if failing_sink == "raw" else None)
    mcap = FakeSink(
        "mcap", events, fail_submit_id=0 if failing_sink == "mcap" else None
    )
    coordinator = CaptureCoordinator(
        "dual_write", raw_sink=raw, mcap_sink=mcap, queue_capacity=8
    ).start()
    for value in range(4):
        coordinator.submit(value)

    result = coordinator.stop()

    healthy = "mcap" if failing_sink == "raw" else "raw"
    assert result.status is CaptureStatus.QUARANTINED
    assert result.terminal_accepted_frontier == 3
    assert result.dispositions[0][failing_sink] is SinkDisposition.FAILED
    assert result.dispositions[0][healthy] is SinkDisposition.DURABLE
    for record_id in range(1, 4):
        assert (
            result.dispositions[record_id][failing_sink]
            is SinkDisposition.NOT_ATTEMPTED_AFTER_SINK_FAILURE
        )
        assert result.dispositions[record_id][healthy] is SinkDisposition.DURABLE
    assert all(
        disposition
        not in {
            SinkDisposition.NOT_ATTEMPTED,
            SinkDisposition.ACCEPTED,
            SinkDisposition.WRITTEN,
        }
        for row in result.dispositions.values()
        for disposition in row.values()
    )


def test_failure_between_raw_and_mcap_still_attempts_mcap_for_failing_id() -> None:
    events: list[tuple[str, str, int | None]] = []
    raw = FakeSink("raw", events, fail_submit_id=0)
    mcap = FakeSink("mcap", events)
    coordinator = CaptureCoordinator("dual_write", raw_sink=raw, mcap_sink=mcap).start()
    coordinator.submit("record")

    result = coordinator.stop()

    assert events[:2] == [("submit", "raw", 0), ("submit", "mcap", 0)]
    assert result.dispositions[0] == {
        "raw": SinkDisposition.FAILED,
        "mcap": SinkDisposition.DURABLE,
    }


def test_failure_after_both_accept_marks_lost_on_failed_seal() -> None:
    events: list[tuple[str, str, int | None]] = []
    raw = FakeSink("raw", events)
    mcap = FakeSink("mcap", events, fail_seal=True)
    coordinator = CaptureCoordinator("dual_write", raw_sink=raw, mcap_sink=mcap).start()
    coordinator.submit("record")

    result = coordinator.stop()

    assert result.status is CaptureStatus.QUARANTINED
    assert result.dispositions[0]["raw"] is SinkDisposition.DURABLE
    assert result.dispositions[0]["mcap"] is SinkDisposition.LOST_ON_CLOSE


def test_temporary_accepted_lag_becomes_durable_at_clean_stop() -> None:
    events: list[tuple[str, str, int | None]] = []
    raw = FakeSink("raw", events, submit_state=SinkDisposition.ACCEPTED)
    mcap = FakeSink("mcap", events, submit_state=SinkDisposition.ACCEPTED)
    coordinator = CaptureCoordinator("dual_write", raw_sink=raw, mcap_sink=mcap).start()
    coordinator.submit("record")

    result = coordinator.stop()

    assert result.status is CaptureStatus.READY
    assert result.dispositions[0] == {
        "raw": SinkDisposition.DURABLE,
        "mcap": SinkDisposition.DURABLE,
    }


def test_failure_before_first_dispatch_has_empty_terminal_frontier() -> None:
    events: list[tuple[str, str, int | None]] = []
    raw = FakeSink("raw", events)
    mcap = FakeSink("mcap", events)
    coordinator = CaptureCoordinator("dual_write", raw_sink=raw, mcap_sink=mcap).start()

    coordinator.report_sink_failure("raw", "startup sink fault")
    with pytest.raises(CaptureAdmissionClosed):
        coordinator.submit("never admitted")
    result = coordinator.stop()

    assert result.status is CaptureStatus.QUARANTINED
    assert result.terminal_accepted_frontier is None
    assert not result.dispositions


def test_async_failure_callback_closes_admission_at_existing_frontier() -> None:
    events: list[tuple[str, str, int | None]] = []
    raw = FakeSink("raw", events, block_id=0)
    mcap = FakeSink("mcap", events)
    coordinator = CaptureCoordinator(
        "dual_write", raw_sink=raw, mcap_sink=mcap, queue_capacity=4
    ).start()
    assert coordinator.submit("zero") == 0
    assert raw.entered.wait(1.0)
    assert coordinator.submit("one") == 1

    reported = threading.Event()

    def report_failure() -> None:
        coordinator.report_sink_failure(
            "raw", "async write fault", collector_record_id=0
        )
        reported.set()

    reporter = threading.Thread(target=report_failure)
    reporter.start()
    raw.release.set()
    assert reported.wait(1.0)
    reporter.join()
    with pytest.raises(CaptureAdmissionClosed):
        coordinator.submit("rejected")
    result = coordinator.stop()

    assert result.terminal_accepted_frontier == 1
    assert result.dispositions[0]["raw"] is SinkDisposition.FAILED
    assert (
        result.dispositions[1]["raw"]
        is SinkDisposition.NOT_ATTEMPTED_AFTER_SINK_FAILURE
    )
    assert result.dispositions[1]["mcap"] is SinkDisposition.DURABLE


def test_bounded_queue_overflow_closes_admission_without_consuming_id() -> None:
    events: list[tuple[str, str, int | None]] = []
    raw = FakeSink("raw", events, block_id=0)
    coordinator = CaptureCoordinator("raw_v1", raw_sink=raw, queue_capacity=1).start()
    assert coordinator.submit("dispatching") == 0
    assert raw.entered.wait(1.0)
    assert coordinator.submit("queued") == 1
    with pytest.raises(CaptureQueueFull):
        coordinator.submit("overflow")
    with pytest.raises(CaptureAdmissionClosed):
        coordinator.submit("still closed")
    raw.release.set()

    result = coordinator.stop()

    assert result.status is CaptureStatus.FAILED
    assert result.terminal_accepted_frontier == 1
    assert tuple(result.dispositions) == (0, 1)


def test_sorted_camera_source_fences_are_handed_to_both_sinks() -> None:
    events: list[tuple[str, str, int | None]] = []
    raw = FakeSink("raw", events)
    mcap = FakeSink("mcap", events)
    coordinator = CaptureCoordinator("dual_write", raw_sink=raw, mcap_sink=mcap).start()
    coordinator.submit("record")
    fences = (
        CaptureSourceFence("z-camera", "session-2", 5, 8),
        CaptureSourceFence("a-camera", "session-1", 2, 4),
    )

    coordinator.stop(source_fences=fences)

    expected = tuple(sorted(fences))
    assert raw.fences == expected
    assert mcap.fences == expected


def test_duplicate_camera_source_fences_fail_closed_and_close_sink() -> None:
    events: list[tuple[str, str, int | None]] = []
    raw = FakeSink("raw", events)
    coordinator = CaptureCoordinator("raw_v1", raw_sink=raw).start()
    duplicate = CaptureSourceFence("camera", "session", 0, 1)
    with pytest.raises(ValueError, match="unique"):
        coordinator.stop(source_fences=(duplicate, duplicate))
    assert coordinator.admission_closed
    assert ("seal", "raw", None) not in events
    assert ("close", "raw", None) in events
    with pytest.raises(CaptureAdmissionClosed):
        coordinator.submit("rejected")
    with pytest.raises(CaptureError, match="already stopped"):
        coordinator.stop(source_fences=(duplicate,))


def test_source_fence_end_must_strictly_follow_start() -> None:
    with pytest.raises(ValueError, match="must follow"):
        CaptureSourceFence("camera", "session", 5, 5)


def test_hung_sink_stop_is_bounded_and_fails_closed() -> None:
    events: list[tuple[str, str, int | None]] = []
    sink = FakeSink("raw", events, block_id=0)
    coordinator = CaptureCoordinator(
        "raw_v1", raw_sink=sink, stop_timeout_sec=0.05
    ).start()
    coordinator.submit("record")
    assert sink.entered.wait(1.0)

    started = time.monotonic()
    result = coordinator.stop()
    elapsed = time.monotonic() - started
    sink.release.set()

    assert elapsed < 0.5
    assert result.status is CaptureStatus.FAILED
    assert result.dispositions[0]["raw"] is SinkDisposition.LOST_ON_CLOSE
    assert any("did not stop" in error for error in result.errors)


def test_malformed_sink_return_fails_capture() -> None:
    events: list[tuple[str, str, int | None]] = []
    sink = FakeSink("raw", events, submit_state="malformed")
    coordinator = CaptureCoordinator("raw_v1", raw_sink=sink).start()
    coordinator.submit("record")

    result = coordinator.stop()

    assert result.status is CaptureStatus.FAILED
    assert result.dispositions[0]["raw"] is SinkDisposition.FAILED
    assert any("malformed" in error for error in result.errors)


def test_close_failure_invalidates_previously_durable_evidence() -> None:
    events: list[tuple[str, str, int | None]] = []
    sink = FakeSink("raw", events, fail_close=True)
    coordinator = CaptureCoordinator("raw_v1", raw_sink=sink).start()
    coordinator.submit("record")

    result = coordinator.stop()

    assert result.status is CaptureStatus.FAILED
    assert result.dispositions[0]["raw"] is SinkDisposition.LOST_ON_CLOSE
    assert any("close failed" in error for error in result.errors)


def test_result_evidence_is_immutable() -> None:
    events: list[tuple[str, str, int | None]] = []
    raw = FakeSink("raw", events)
    result = CaptureCoordinator("raw_v1", raw_sink=raw).start().stop()
    with pytest.raises(TypeError):
        result.dispositions[0] = {}  # type: ignore[index]
