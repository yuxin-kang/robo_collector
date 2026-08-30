"""Deterministic content-quality rules for canonical MCAP candidates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .mcap_contract import build_qc_evidence, reduce_quality_rules


@dataclass(frozen=True)
class VideoObservation:
    stream_id: str
    frame_index: int
    mean_luma: int
    clipped_dark_fraction: Fraction = Fraction(0)
    clipped_bright_fraction: Fraction = Fraction(0)
    sharpness: int = 1
    content_sha256: str = ""
    corrupt: bool = False


@dataclass(frozen=True)
class RobotObservation:
    source_id: str
    sequence: int
    time_ns: int
    values: Sequence[float]
    lower_limits: Sequence[float] = ()
    upper_limits: Sequence[float] = ()
    action: bool = False


@dataclass(frozen=True)
class ContentQualityPolicy:
    black_luma_max: int = 8
    exposure_fraction_max: Fraction = Fraction(95, 100)
    blur_sharpness_min: int = 1
    frozen_run_max: int = 30
    stuck_run_max: int = 100
    discontinuity_max: Fraction = Fraction(1_000_000)
    jerk_max: Fraction = Fraction(1_000_000)
    action_saturation_fraction_max: Fraction = Fraction(1, 2)

    def as_config(self) -> dict[str, Any]:
        def ratio(value: Fraction) -> str:
            return f"{value.numerator}/{value.denominator}"

        return {
            "format": "robo_collector.content_quality_policy",
            "format_version": 1,
            "black_luma_max": str(self.black_luma_max),
            "exposure_fraction_max": ratio(self.exposure_fraction_max),
            "blur_sharpness_min": str(self.blur_sharpness_min),
            "frozen_run_max": str(self.frozen_run_max),
            "stuck_run_max": str(self.stuck_run_max),
            "discontinuity_max": ratio(self.discontinuity_max),
            "jerk_max": ratio(self.jerk_max),
            "action_saturation_fraction_max": ratio(
                self.action_saturation_fraction_max
            ),
        }


DEFAULT_CONTENT_QUALITY_POLICY = ContentQualityPolicy()


def _ratio(value: Fraction) -> str:
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


def _rule(
    rule_id: str,
    severity: str,
    result: str,
    observations: Iterable[Mapping[str, str]],
    metrics: Iterable[Mapping[str, str]],
) -> dict[str, Any]:
    _, evidence_hash = build_qc_evidence(
        rule_id=rule_id, rule_version="1", observations=observations
    )
    return {
        "rule_id": rule_id,
        "severity": severity,
        "result": result,
        "evidence_sha256": evidence_hash,
        "metrics": list(metrics),
    }


def evaluate_video_quality(
    observations: Iterable[VideoObservation],
    *,
    policy: ContentQualityPolicy = DEFAULT_CONTENT_QUALITY_POLICY,
) -> tuple[dict[str, Any], ...]:
    values = sorted(
        observations, key=lambda item: (item.stream_id.encode(), item.frame_index)
    )
    count = len(values)
    corrupt = sum(item.corrupt for item in values)
    black = sum(item.mean_luma <= policy.black_luma_max for item in values)
    exposed = sum(
        item.clipped_dark_fraction >= policy.exposure_fraction_max
        or item.clipped_bright_fraction >= policy.exposure_fraction_max
        for item in values
    )
    blurred = sum(item.sharpness < policy.blur_sharpness_min for item in values)
    longest_frozen = 0
    current = 0
    previous: tuple[str, str] | None = None
    for item in values:
        key = (item.stream_id, item.content_sha256)
        current = current + 1 if item.content_sha256 and key == previous else 1
        longest_frozen = max(longest_frozen, current)
        previous = key
    specs = (
        ("video.corrupt", "CRITICAL", corrupt == 0, corrupt, "corrupt_frames"),
        ("video.black", "WARNING", black == 0, black, "black_frames"),
        ("video.exposure", "WARNING", exposed == 0, exposed, "exposure_frames"),
        ("video.blur", "WARNING", blurred == 0, blurred, "blurred_frames"),
        (
            "video.frozen",
            "WARNING",
            longest_frozen <= policy.frozen_run_max,
            longest_frozen,
            "longest_frozen_run",
        ),
    )
    return tuple(
        _rule(
            rule_id,
            severity,
            "PASS" if passed else "FAIL",
            [{"code": metric, "subject": "all", "value": str(number)}],
            [
                {"name": metric, "unit": "1", "value": str(number)},
                {"name": "frames", "unit": "1", "value": str(count)},
            ],
        )
        for rule_id, severity, passed, number, metric in specs
    )


def evaluate_robot_quality(
    observations: Iterable[RobotObservation],
    *,
    policy: ContentQualityPolicy = DEFAULT_CONTENT_QUALITY_POLICY,
) -> tuple[dict[str, Any], ...]:
    values = sorted(
        observations, key=lambda item: (item.source_id.encode(), item.sequence)
    )
    nonfinite = range_errors = discontinuities = jerk_errors = saturated = 0
    longest_stuck = current_stuck = 0
    previous_by_source: dict[str, RobotObservation] = {}
    delta_by_source: dict[str, tuple[float, ...]] = {}
    for item in values:
        if item.time_ns < 0 or any(not math.isfinite(value) for value in item.values):
            nonfinite += 1
        if item.lower_limits or item.upper_limits:
            if (
                len(item.lower_limits) != len(item.values)
                or len(item.upper_limits) != len(item.values)
                or any(
                    value < low or value > high
                    for value, low, high in zip(
                        item.values, item.lower_limits, item.upper_limits
                    )
                )
            ):
                range_errors += 1
            if item.action and item.values:
                hits = sum(
                    value == low or value == high
                    for value, low, high in zip(
                        item.values, item.lower_limits, item.upper_limits
                    )
                )
                if (
                    Fraction(hits, len(item.values))
                    > policy.action_saturation_fraction_max
                ):
                    saturated += 1
        previous = previous_by_source.get(item.source_id)
        if previous is not None and len(previous.values) == len(item.values):
            delta = tuple(
                current - old for current, old in zip(item.values, previous.values)
            )
            current_stuck = current_stuck + 1 if not any(delta) else 0
            longest_stuck = max(longest_stuck, current_stuck)
            if any(
                abs(Fraction(value)) > policy.discontinuity_max
                for value in delta
                if math.isfinite(value)
            ):
                discontinuities += 1
            prior_delta = delta_by_source.get(item.source_id)
            if prior_delta is not None and any(
                abs(Fraction(current - old)) > policy.jerk_max
                for current, old in zip(delta, prior_delta)
                if math.isfinite(current) and math.isfinite(old)
            ):
                jerk_errors += 1
            delta_by_source[item.source_id] = delta
        else:
            current_stuck = 0
        previous_by_source[item.source_id] = item
    specs = (
        ("robot.nonfinite", "CRITICAL", nonfinite == 0, nonfinite),
        ("robot.range", "CRITICAL", range_errors == 0, range_errors),
        (
            "robot.stuck",
            "WARNING",
            longest_stuck <= policy.stuck_run_max,
            longest_stuck,
        ),
        ("robot.discontinuity", "WARNING", discontinuities == 0, discontinuities),
        ("robot.jerk", "WARNING", jerk_errors == 0, jerk_errors),
        ("robot.action_saturation", "WARNING", saturated == 0, saturated),
    )
    return tuple(
        _rule(
            rule_id,
            severity,
            "PASS" if passed else "FAIL",
            [{"code": "count", "subject": "all", "value": str(number)}],
            [{"name": "count", "unit": "1", "value": str(number)}],
        )
        for rule_id, severity, passed, number in specs
    )


def build_content_quality(
    video: Iterable[VideoObservation] = (),
    robot: Iterable[RobotObservation] = (),
    *,
    structural_valid: bool = True,
    structural_ambiguous: bool = False,
    policy: ContentQualityPolicy = DEFAULT_CONTENT_QUALITY_POLICY,
) -> dict[str, Any]:
    """Evaluate all rules and apply QUARANTINED > REJECT > REVIEW > READY."""

    rules = (
        *evaluate_video_quality(video, policy=policy),
        *evaluate_robot_quality(robot, policy=policy),
    )
    return reduce_quality_rules(
        rules,
        policy_name="canonical_content_v1",
        policy_version="1",
        policy_config=policy.as_config(),
        quarantined=not structural_valid or structural_ambiguous,
    )
