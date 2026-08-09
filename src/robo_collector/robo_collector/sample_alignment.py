"""Pure sample-selection rules shared by the ROS collector and tests."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

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
