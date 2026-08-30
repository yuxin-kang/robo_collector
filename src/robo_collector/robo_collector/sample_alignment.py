"""Pure sample-selection rules shared by the ROS collector and tests."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from typing import Any, Literal

from .field_config import FieldSelection

_ROBOT_STATE_FIELDS = {
    "joint_position",
    "joint_velocity",
    "joint_torque",
}
_IMU_FIELDS = {
    "imu_angular_velocity",
    "imu_linear_acceleration",
    "projected_gravity_or_quat",
}

_INT64_MAX = (1 << 63) - 1


class AlignmentError(ValueError):
    """Raised when clock or sample evidence cannot produce canonical rows."""


@dataclass(frozen=True)
class ClockRecord:
    """One source record before deterministic clock normalization."""

    stream_id: str
    source_session_id: str
    source_sequence: int
    collector_record_id: int
    receive_time_ns: int
    source_time_ns: int | None
    packet_sequence: int = 0
    payload: Any = None
    clock_session_id: str = ""
    normalized_time_ns: int | None = None
    normalization_mode: str = ""
    normalization_uncertainty_ns: int = 0
    fallback_reason: str = "NONE"


@dataclass(frozen=True)
class ClockSegment:
    stream_id: str
    clock_session_id: str
    source_session_id: str
    first_source_sequence: int
    last_source_sequence: int
    record_count: int
    normalization_mode: Literal["AFFINE_V2", "RECEIVE_FALLBACK"]
    uncertainty_ns: int
    fallback_reason: str
    slope_numerator: int | None = None
    slope_denominator: int | None = None
    offset_numerator: int | None = None
    offset_denominator: int | None = None
    ready_eligible: bool = True


@dataclass(frozen=True)
class ClockNormalizationConfig:
    affine_min_edges: int = 30
    affine_ppm_limit: int = 2000
    even_sample_limit: int = 512
    fallback_max_interval_ns: int = 1_000_000_000
    max_uncertainty_ns: int = 20_000_000
    source_rollback_split_ns: int = 1_000_000


@dataclass(frozen=True)
class ClockNormalizationResult:
    records: tuple[ClockRecord, ...]
    segments: tuple[ClockSegment, ...]

    @property
    def ready_eligible(self) -> bool:
        return all(segment.ready_eligible for segment in self.segments)


@dataclass(frozen=True)
class AlignmentConfig:
    reference_camera_stream: str
    max_camera_residual_ns: int = 20_000_000
    max_state_residual_ns: int = 20_000_000
    action_max_age_ns: int = 20_000_000
    policy: Literal["rgb_affine_v2", "legacy_rgb_v1"] = "rgb_affine_v2"
    configured_rates_hz: tuple[tuple[str, float], ...] = ()
    require_state: bool = True
    require_action: bool = True

    def __post_init__(self) -> None:
        if self.policy == "rgb_affine_v2" and not self.reference_camera_stream:
            raise AlignmentError("rgb_affine_v2 requires reference_camera_stream")
        for name in (
            "max_camera_residual_ns",
            "max_state_residual_ns",
            "action_max_age_ns",
        ):
            if getattr(self, name) < 0:
                raise AlignmentError(f"{name} must be nonnegative")
        seen: set[str] = set()
        for stream_id, rate in self.configured_rates_hz:
            if not stream_id or stream_id in seen:
                raise AlignmentError("configured rates require unique stream IDs")
            if not math.isfinite(rate) or rate <= 0:
                raise AlignmentError("configured rates must be finite and positive")
            seen.add(stream_id)


@dataclass(frozen=True)
class SelectionGap:
    reference_stream_id: str
    reference_session_id: str
    reference_source_sequence: int
    target_time_ns: int
    missing: tuple[str, ...]


@dataclass(frozen=True)
class AlignedRow:
    row_index: int
    reference: ClockRecord
    cameras: tuple[tuple[str, ClockRecord], ...]
    state: ClockRecord | None
    action: ClockRecord | None
    camera_residuals_ns: tuple[tuple[str, int], ...]
    state_residual_ns: int | None
    action_age_ns: int | None


@dataclass(frozen=True)
class AlignmentResult:
    rows: tuple[AlignedRow, ...]
    gaps: tuple[SelectionGap, ...]
    dense_camera_records: tuple[ClockRecord, ...]
    dense_state_records: tuple[ClockRecord, ...]
    dense_action_records: tuple[ClockRecord, ...]
    duplicate_count: int
    observed_rates_hz: tuple[tuple[str, float | None], ...]
    configured_rates_hz: tuple[tuple[str, float], ...]


def _validate_ns(value: int | None, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise AlignmentError(f"{name} must be an integer nanosecond value")
    if value < 0 or value > _INT64_MAX:
        raise AlignmentError(f"{name} is outside signed int64 range")


def _even_sample(values: list[Any], limit: int) -> list[Any]:
    if limit < 2:
        raise AlignmentError("even sample limit must be at least two")
    if len(values) <= limit:
        return values[:]
    return [values[(index * (len(values) - 1)) // (limit - 1)] for index in range(limit)]


def _median(values: Iterable[Fraction]) -> Fraction:
    ordered = sorted(values)
    if not ordered:
        raise AlignmentError("median requires at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _nearest_rank(values: Iterable[Fraction], numerator: int, denominator: int) -> Fraction:
    ordered = sorted(values)
    if not ordered:
        raise AlignmentError("quantile requires at least one value")
    rank = max(1, (numerator * len(ordered) + denominator - 1) // denominator)
    return ordered[rank - 1]


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _round_ties_even(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled = remainder * 2
    if doubled < value.denominator:
        return quotient
    if doubled > value.denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def _split_clock_records(
    records: Iterable[ClockRecord], config: ClockNormalizationConfig
) -> list[list[ClockRecord]]:
    segments: list[list[ClockRecord]] = []
    current: list[ClockRecord] = []
    previous: ClockRecord | None = None
    for record in records:
        _validate_ns(record.receive_time_ns, "receive_time_ns")
        _validate_ns(record.source_time_ns, "source_time_ns", optional=True)
        if record.source_sequence < 0 or record.collector_record_id < 0:
            raise AlignmentError("sequences and collector record IDs must be nonnegative")
        split = previous is not None and (
            record.stream_id != previous.stream_id
            or record.source_session_id != previous.source_session_id
            or record.source_sequence <= previous.source_sequence
            or (
                record.source_time_ns is not None
                and previous.source_time_ns is not None
                and previous.source_time_ns - record.source_time_ns
                > config.source_rollback_split_ns
            )
        )
        if split:
            segments.append(current)
            current = []
        current.append(record)
        previous = record
    if current:
        segments.append(current)
    return segments


def _fallback_segment(
    records: list[ClockRecord],
    *,
    clock_session_id: str,
    reason: str,
    config: ClockNormalizationConfig,
) -> tuple[list[ClockRecord], ClockSegment]:
    intervals = [
        current.receive_time_ns - previous.receive_time_ns
        for previous, current in pairwise(records)
        if current.source_sequence == previous.source_sequence + 1
        and 0 < current.receive_time_ns - previous.receive_time_ns
        <= config.fallback_max_interval_ns
    ]
    sampled = _even_sample(intervals, config.even_sample_limit)
    if sampled:
        median = _median(Fraction(value) for value in sampled)
        uncertainty = _ceil_fraction(
            _nearest_rank(
                (abs(Fraction(value) - median) for value in sampled), 95, 100
            )
        )
    else:
        uncertainty = 0
    ready = len(intervals) >= config.affine_min_edges and uncertainty <= config.max_uncertainty_ns
    normalized = [
        replace(
            record,
            clock_session_id=clock_session_id,
            normalized_time_ns=record.receive_time_ns,
            normalization_mode="RECEIVE_FALLBACK",
            normalization_uncertainty_ns=uncertainty,
            fallback_reason=reason,
        )
        for record in records
    ]
    return normalized, ClockSegment(
        stream_id=records[0].stream_id,
        clock_session_id=clock_session_id,
        source_session_id=records[0].source_session_id,
        first_source_sequence=records[0].source_sequence,
        last_source_sequence=records[-1].source_sequence,
        record_count=len(records),
        normalization_mode="RECEIVE_FALLBACK",
        uncertainty_ns=uncertainty,
        fallback_reason=reason,
        ready_eligible=ready,
    )


def normalize_clock_records(
    records: Iterable[ClockRecord],
    config: ClockNormalizationConfig | None = None,
) -> ClockNormalizationResult:
    """Normalize ordered source records with the frozen exact affine-v2 rules."""

    config = config or ClockNormalizationConfig()
    normalized: list[ClockRecord] = []
    evidence: list[ClockSegment] = []
    stream_generation: dict[tuple[str, str], int] = {}
    input_records = list(records)
    collector_ids = [record.collector_record_id for record in input_records]
    if len(collector_ids) != len(set(collector_ids)):
        raise AlignmentError("collector_record_id must be globally unique")
    streams: dict[str, list[ClockRecord]] = {}
    for record in input_records:
        streams.setdefault(record.stream_id, []).append(record)
    clock_segments = [
        segment
        for stream_id in sorted(streams)
        for segment in _split_clock_records(streams[stream_id], config)
    ]
    for records_in_segment in clock_segments:
        first = records_in_segment[0]
        generation_key = (first.stream_id, first.source_session_id)
        generation = stream_generation.get(generation_key, 0)
        stream_generation[generation_key] = generation + 1
        clock_session_id = f"{first.source_session_id}.{generation}"
        if any(record.source_time_ns is None for record in records_in_segment):
            segment_records, segment = _fallback_segment(
                records_in_segment,
                clock_session_id=clock_session_id,
                reason="SOURCE_TIME_MISSING",
                config=config,
            )
        else:
            edges: list[tuple[ClockRecord, ClockRecord, Fraction]] = []
            low = Fraction(1000_000 - config.affine_ppm_limit, 1_000_000)
            high = Fraction(1000_000 + config.affine_ppm_limit, 1_000_000)
            for previous, current in pairwise(records_in_segment):
                delta_source = current.source_time_ns - previous.source_time_ns  # type: ignore[operator]
                delta_receive = current.receive_time_ns - previous.receive_time_ns
                if (
                    current.source_sequence == previous.source_sequence + 1
                    and 0 < delta_source <= config.fallback_max_interval_ns
                    and 0 < delta_receive <= config.fallback_max_interval_ns
                ):
                    ratio = Fraction(delta_receive, delta_source)
                    if low <= ratio <= high:
                        edges.append((previous, current, ratio))
            if len(edges) < config.affine_min_edges:
                segment_records, segment = _fallback_segment(
                    records_in_segment,
                    clock_session_id=clock_session_id,
                    reason="INSUFFICIENT_VALID_EDGES",
                    config=config,
                )
            else:
                fit_edges = _even_sample(edges, config.even_sample_limit)
                slope = _median(edge[2] for edge in fit_edges)
                endpoints: dict[tuple[str, int], ClockRecord] = {}
                for previous, current, _ratio in fit_edges:
                    endpoints[(previous.source_session_id, previous.source_sequence)] = previous
                    endpoints[(current.source_session_id, current.source_sequence)] = current
                fit_points = sorted(endpoints.values(), key=lambda item: item.source_sequence)
                offsets = [
                    Fraction(point.receive_time_ns) - slope * point.source_time_ns  # type: ignore[arg-type]
                    for point in fit_points
                ]
                offset = _nearest_rank(offsets, 5, 100)
                residuals = [
                    Fraction(point.receive_time_ns)
                    - (slope * point.source_time_ns + offset)  # type: ignore[operator]
                    for point in fit_points
                ]
                residual_median = _median(residuals)
                uncertainty = _ceil_fraction(
                    _nearest_rank(
                        (abs(residual - residual_median) for residual in residuals),
                        95,
                        100,
                    )
                )
                if uncertainty > config.max_uncertainty_ns:
                    segment_records, segment = _fallback_segment(
                        records_in_segment,
                        clock_session_id=clock_session_id,
                        reason="AFFINE_UNCERTAINTY",
                        config=config,
                    )
                else:
                    segment_records = []
                    for record in records_in_segment:
                        mapped = _round_ties_even(
                            slope * record.source_time_ns + offset  # type: ignore[operator]
                        )
                        _validate_ns(mapped, "normalized_time_ns")
                        segment_records.append(
                            replace(
                                record,
                                clock_session_id=clock_session_id,
                                normalized_time_ns=mapped,
                                normalization_mode="AFFINE_V2",
                                normalization_uncertainty_ns=uncertainty,
                                fallback_reason="NONE",
                            )
                        )
                    segment = ClockSegment(
                        stream_id=first.stream_id,
                        clock_session_id=clock_session_id,
                        source_session_id=first.source_session_id,
                        first_source_sequence=first.source_sequence,
                        last_source_sequence=records_in_segment[-1].source_sequence,
                        record_count=len(records_in_segment),
                        normalization_mode="AFFINE_V2",
                        uncertainty_ns=uncertainty,
                        fallback_reason="NONE",
                        slope_numerator=slope.numerator,
                        slope_denominator=slope.denominator,
                        offset_numerator=offset.numerator,
                        offset_denominator=offset.denominator,
                    )
        normalized.extend(segment_records)
        evidence.append(segment)
    normalized.sort(key=lambda record: record.collector_record_id)
    return ClockNormalizationResult(tuple(normalized), tuple(evidence))


def _deduplicate(records: Iterable[ClockRecord]) -> tuple[list[ClockRecord], int]:
    unique: dict[tuple[str, str, int, int], ClockRecord] = {}
    duplicates = 0
    for record in records:
        if record.normalized_time_ns is None:
            raise AlignmentError("alignment requires normalized records")
        _validate_ns(record.normalized_time_ns, "normalized_time_ns")
        identity = (
            record.stream_id,
            record.source_session_id,
            record.source_sequence,
            record.packet_sequence,
        )
        previous = unique.get(identity)
        if previous is None:
            unique[identity] = record
        elif replace(previous, collector_record_id=0) == replace(
            record, collector_record_id=0
        ):
            duplicates += 1
            if record.collector_record_id > previous.collector_record_id:
                unique[identity] = record
        else:
            raise AlignmentError(f"divergent duplicate source identity: {identity!r}")
    return list(unique.values()), duplicates


def _nearest(records: Iterable[ClockRecord], target: int, maximum: int, *, legacy: bool) -> ClockRecord | None:
    candidates = [record for record in records if abs(record.normalized_time_ns - target) <= maximum]  # type: ignore[operator]
    if not candidates:
        return None
    if legacy:
        return min(candidates, key=lambda record: (abs(record.normalized_time_ns - target), -record.normalized_time_ns))  # type: ignore[operator]
    return min(
        candidates,
        key=lambda record: (
            abs(record.normalized_time_ns - target),  # type: ignore[operator]
            int(record.normalized_time_ns > target),
            -record.source_sequence,
            -record.collector_record_id,
            record.source_session_id.encode(),
        ),
    )


def _latest_action(records: Iterable[ClockRecord], target: int, maximum_age: int) -> ClockRecord | None:
    candidates = [record for record in records if record.normalized_time_ns <= target and target - record.normalized_time_ns <= maximum_age]  # type: ignore[operator]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda record: (
            -record.normalized_time_ns,  # type: ignore[operator]
            -record.source_sequence,
            -record.collector_record_id,
            record.source_session_id.encode(),
        ),
    )


def _observed_rates(records: Iterable[ClockRecord]) -> tuple[tuple[str, float | None], ...]:
    streams: dict[str, list[int]] = {}
    for record in records:
        streams.setdefault(record.stream_id, []).append(record.normalized_time_ns)  # type: ignore[arg-type]
    result = []
    for stream_id, values in sorted(streams.items()):
        values.sort()
        duration = values[-1] - values[0] if len(values) > 1 else 0
        rate = (len(values) - 1) * 1_000_000_000 / duration if duration > 0 else None
        result.append((stream_id, rate))
    return tuple(result)


def align_rgb_records(
    camera_records: Iterable[ClockRecord],
    state_records: Iterable[ClockRecord],
    action_records: Iterable[ClockRecord],
    config: AlignmentConfig,
) -> AlignmentResult:
    """Build retained RGB-anchored rows without mutating dense raw inputs."""

    dense_cameras = tuple(camera_records)
    dense_states = tuple(state_records)
    dense_actions = tuple(action_records)
    cameras, camera_duplicates = _deduplicate(dense_cameras)
    states, state_duplicates = _deduplicate(dense_states)
    actions, action_duplicates = _deduplicate(dense_actions)
    camera_streams = sorted({record.stream_id for record in cameras})
    reference_stream = config.reference_camera_stream
    if not reference_stream and config.policy == "legacy_rgb_v1" and dense_cameras:
        reference_stream = dense_cameras[0].stream_id
    if reference_stream not in camera_streams:
        raise AlignmentError("reference camera stream does not exist")
    reference = sorted(
        (record for record in cameras if record.stream_id == reference_stream),
        key=lambda record: (
            record.normalized_time_ns,
            record.source_session_id.encode(),
            record.source_sequence,
            record.packet_sequence,
            record.collector_record_id,
        ),
    )
    other_streams = [stream for stream in camera_streams if stream != reference_stream]
    rows: list[AlignedRow] = []
    gaps: list[SelectionGap] = []
    legacy = config.policy == "legacy_rgb_v1"
    for target_record in reference:
        target = target_record.normalized_time_ns
        selected_cameras: list[tuple[str, ClockRecord]] = []
        camera_residuals: list[tuple[str, int]] = []
        missing: list[str] = []
        for stream_id in other_streams:
            selected = _nearest(
                (record for record in cameras if record.stream_id == stream_id),
                target,
                config.max_camera_residual_ns,
                legacy=legacy,
            )
            if selected is None:
                missing.append(f"camera:{stream_id}")
            else:
                selected_cameras.append((stream_id, selected))
                camera_residuals.append((stream_id, selected.normalized_time_ns - target))
        state = _nearest(states, target, config.max_state_residual_ns, legacy=legacy)
        if state is None and config.require_state:
            missing.append("state")
        action = _latest_action(actions, target, config.action_max_age_ns)
        if action is None and config.require_action:
            missing.append("action")
        if missing:
            gaps.append(
                SelectionGap(
                    reference_stream_id=reference_stream,
                    reference_session_id=target_record.source_session_id,
                    reference_source_sequence=target_record.source_sequence,
                    target_time_ns=target,
                    missing=tuple(sorted(missing)),
                )
            )
            continue
        rows.append(
            AlignedRow(
                row_index=len(rows),
                reference=target_record,
                cameras=tuple(selected_cameras),
                state=state,
                action=action,
                camera_residuals_ns=tuple(camera_residuals),
                state_residual_ns=None if state is None else state.normalized_time_ns - target,
                action_age_ns=None if action is None else target - action.normalized_time_ns,
            )
        )
    return AlignmentResult(
        rows=tuple(rows),
        gaps=tuple(gaps),
        dense_camera_records=dense_cameras,
        dense_state_records=dense_states,
        dense_action_records=dense_actions,
        duplicate_count=camera_duplicates + state_duplicates + action_duplicates,
        observed_rates_hz=_observed_rates((*dense_cameras, *dense_states, *dense_actions)),
        configured_rates_hz=tuple(sorted(config.configured_rates_hz)),
    )


def required_sample_inputs(selection: FieldSelection) -> set[str]:
    required: set[str] = set()
    for field_name in selection.state:
        if field_name in _ROBOT_STATE_FIELDS:
            required.add("joint_states")
        elif field_name in _IMU_FIELDS:
            required.add("imu")
        else:
            required.add(field_name)
    for field_name in selection.target:
        if field_name == "joint_position":
            required.add("target_joint_pos")
        else:
            required.add(field_name)
    if selection.include_policy_action:
        required.add("action")
    return required


def selected_missing_inputs(
    reported_missing: Iterable[str], selection: FieldSelection
) -> list[str]:
    required = required_sample_inputs(selection)
    return sorted(required.intersection(str(name) for name in reported_missing))


def selected_source_timestamps_sec(
    msg: Any, selection: FieldSelection
) -> dict[str, float] | None:
    names = getattr(msg, "source_timestamp_names", None)
    values = getattr(msg, "source_timestamps_sec", None)
    if names is None or values is None:
        return None
    try:
        if len(names) != len(values):
            return None
    except TypeError:
        return None

    timestamps: dict[str, float] = {}
    for raw_name, raw_value in zip(names, values):
        name = str(raw_name).strip()
        if not name or name in timestamps or isinstance(raw_value, bool):
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        timestamps[name] = value

    required = required_sample_inputs(selection)
    if not required.issubset(timestamps):
        return None
    return {name: timestamps[name] for name in sorted(required)}


def message_stamp_sec(msg: Any) -> float | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    try:
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def source_timestamp_skew_sec(
    state_timestamp_sec: float | None,
    camera_timestamps_sec: Iterable[float | None],
) -> float | None:
    values = [state_timestamp_sec, *camera_timestamps_sec]
    if any(value is None for value in values):
        return None
    numeric = [float(value) for value in values if value is not None]
    if not numeric or any(not math.isfinite(value) for value in numeric):
        return None
    return max(numeric) - min(numeric)
