"""Pure parsing and validation logic for the robo_state ROS node."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence


DOF = 29
ACTION_DIM = DOF
TARGET_JOINT_POS_DIM = DOF
ALIGNED_TARGET_POS_DIM = 45
POLICY_STATE_DIM = 1110
STEPIT_OBSERVATION_DIM = 1545
OBSERVATION_DIM = POLICY_STATE_DIM

JOINT_SUFFIX = "_joint"
CMD_SUFFIX = "_cmd"
GAIN_SUFFIX = "_gain"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dim: int


POLICY_FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("relative_ori_6d", 90),
    FieldSpec("motion_anchor_lin_vel_b", 45),
    FieldSpec("motion_anchor_ang_vel_b", 45),
    FieldSpec("ang_vel_history", 30),
    FieldSpec("gravity_history", 30),
    FieldSpec("joint_pos_rel_history", 290),
    FieldSpec("joint_vel_history", 290),
    FieldSpec("action_history", 290),
)

POLICY_FIELD_DIMS = {spec.name: spec.dim for spec in POLICY_FIELD_SPECS}
FIELD_DIMS = {
    "target_joint_pos": TARGET_JOINT_POS_DIM,
    "aligned_target_pos": ALIGNED_TARGET_POS_DIM,
    "action": ACTION_DIM,
    "observation": STEPIT_OBSERVATION_DIM,
    **POLICY_FIELD_DIMS,
}

REQUIRED_FIELD_NAMES: tuple[str, ...] = (
    "aligned_target_pos",
    *(spec.name for spec in POLICY_FIELD_SPECS),
)


class ValidationError(ValueError):
    """Raised when incoming StepIt data cannot be represented safely."""


@dataclass(frozen=True)
class TimedValue:
    value: Any
    stamp_sec: float


@dataclass(frozen=True)
class RobotLowStateData:
    joint_names: list[str]
    joint_pos: list[float]
    joint_vel: list[float]
    joint_torque: list[float]
    cmd_joint_pos: list[float]
    cmd_joint_vel: list[float]
    cmd_joint_torque: list[float]
    kp: list[float]
    kd: list[float]
    desired_torque: list[float]
    foot_names: list[str]
    foot_force: list[float]

    @classmethod
    def zero(cls, dof: int = DOF) -> "RobotLowStateData":
        zeros = [0.0] * dof
        return cls(
            joint_names=[""] * dof,
            joint_pos=zeros.copy(),
            joint_vel=zeros.copy(),
            joint_torque=zeros.copy(),
            cmd_joint_pos=zeros.copy(),
            cmd_joint_vel=zeros.copy(),
            cmd_joint_torque=zeros.copy(),
            kp=zeros.copy(),
            kd=zeros.copy(),
            desired_torque=zeros.copy(),
            foot_names=[],
            foot_force=[],
        )


@dataclass(frozen=True)
class SampleData:
    sample_stamp_sec: float
    policy_fields: dict[str, list[float]]
    policy_flattened: list[float]
    robot_state: RobotLowStateData
    imu: Any
    target_joint_pos: list[float]
    aligned_target_pos: list[float]
    action: list[float]
    stepit_observation: list[float]
    observation_l2_error: float | None
    missing_optional_fields: list[str]
    source_timestamps_sec: dict[str, float]


@dataclass(frozen=True)
class BuildResult:
    sample: SampleData | None
    level: str
    message: str
    issues: list[str]


def validate_vector(name: str, values: Sequence[float], expected_dim: int) -> list[float]:
    actual_dim = len(values)
    if actual_dim != expected_dim:
        raise ValidationError(
            f"{name} has dimension {actual_dim}; expected {expected_dim}"
        )
    return [
        _finite_float(f"{name}[{index}]", value)
        for index, value in enumerate(values)
    ]


def flatten_policy_fields(fields: Mapping[str, Sequence[float]]) -> list[float]:
    flattened: list[float] = []
    for spec in POLICY_FIELD_SPECS:
        if spec.name not in fields:
            raise ValidationError(f"missing policy field {spec.name}")
        flattened.extend(validate_vector(spec.name, fields[spec.name], spec.dim))
    if len(flattened) != OBSERVATION_DIM:
        raise ValidationError(
            f"policy fields flatten to {len(flattened)}; expected {OBSERVATION_DIM}"
        )
    return flattened


def observation_l2_error(
    flattened_policy: Sequence[float], stepit_observation: Sequence[float]
) -> float:
    if len(flattened_policy) != len(stepit_observation):
        raise ValidationError(
            "cannot compare policy fields and observation with dimensions "
            f"{len(flattened_policy)} and {len(stepit_observation)}"
        )
    left_values = validate_vector(
        "flattened_policy", flattened_policy, len(flattened_policy)
    )
    right_values = validate_vector(
        "stepit_observation", stepit_observation, len(stepit_observation)
    )
    return sqrt(
        sum(
            (left - right) ** 2
            for left, right in zip(left_values, right_values)
        )
    )


def parse_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    velocities: Sequence[float],
    efforts: Sequence[float],
    expected_dof: int = DOF,
) -> RobotLowStateData:
    _validate_joint_state_lengths(names, positions, velocities, efforts)

    joint_rows: dict[str, tuple[float, float, float]] = {}
    cmd_rows: dict[str, tuple[float, float, float]] = {}
    gain_rows: dict[str, tuple[float, float, float]] = {}
    foot_names: list[str] = []
    foot_force: list[float] = []

    for index, raw_name in enumerate(names):
        name = str(raw_name)
        row = (
            _finite_float(f"joint_states.position[{index}]", positions[index]),
            _finite_float(f"joint_states.velocity[{index}]", velocities[index]),
            _finite_float(f"joint_states.effort[{index}]", efforts[index]),
        )
        if name.endswith(JOINT_SUFFIX):
            _insert_unique(joint_rows, _strip_suffix(name, JOINT_SUFFIX), row, name)
        elif name.endswith(CMD_SUFFIX):
            _insert_unique(cmd_rows, _strip_suffix(name, CMD_SUFFIX), row, name)
        elif name.endswith(GAIN_SUFFIX):
            _insert_unique(gain_rows, _strip_suffix(name, GAIN_SUFFIX), row, name)
        else:
            foot_names.append(name)
            foot_force.append(row[2])

    joint_names = list(joint_rows.keys())
    if len(joint_names) != expected_dof:
        raise ValidationError(
            f"joint_states contains {len(joint_names)} joints; expected {expected_dof}"
        )

    _validate_matching_keys("cmd", joint_names, cmd_rows)
    _validate_matching_keys("gain", joint_names, gain_rows)

    joint_pos, joint_vel, joint_torque = _columns(joint_rows, joint_names)
    cmd_joint_pos, cmd_joint_vel, cmd_joint_torque = _columns(cmd_rows, joint_names)
    kp, kd, desired_torque = _columns(gain_rows, joint_names)

    return RobotLowStateData(
        joint_names=joint_names,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        joint_torque=joint_torque,
        cmd_joint_pos=cmd_joint_pos,
        cmd_joint_vel=cmd_joint_vel,
        cmd_joint_torque=cmd_joint_torque,
        kp=kp,
        kd=kd,
        desired_torque=desired_torque,
        foot_names=foot_names,
        foot_force=foot_force,
    )


class RoboStateAssembler:
    """Caches latest StepIt values and builds complete normalized samples."""

    def __init__(
        self,
        *,
        max_cache_age_sec: float = 0.2,
        max_required_skew_sec: float = 0.2,
        publish_only_when_complete: bool = True,
        validate_observation: bool = True,
    ) -> None:
        self.max_cache_age_sec = _finite_float(
            "max_cache_age_sec", max_cache_age_sec
        )
        self.max_required_skew_sec = _finite_float(
            "max_required_skew_sec", max_required_skew_sec
        )
        self.publish_only_when_complete = bool(publish_only_when_complete)
        self.validate_observation = bool(validate_observation)
        self.fields: dict[str, TimedValue] = {}
        self.robot_state: TimedValue | None = None
        self.imu: TimedValue | None = None

    def update_field(
        self, name: str, values: Sequence[float], stamp_sec: float
    ) -> list[float]:
        if name not in FIELD_DIMS:
            raise ValidationError(f"unknown StepIt field {name}")
        vector = validate_vector(name, values, FIELD_DIMS[name])
        self.fields[name] = TimedValue(vector, _timestamp("stamp_sec", stamp_sec))
        return vector

    def update_robot_state(
        self, robot_state: RobotLowStateData, stamp_sec: float
    ) -> None:
        _validate_robot_state(robot_state)
        self.robot_state = TimedValue(
            robot_state, _timestamp("joint_states stamp_sec", stamp_sec)
        )

    def update_imu(self, imu: Any, stamp_sec: float) -> None:
        _validate_imu(imu)
        self.imu = TimedValue(imu, _timestamp("imu stamp_sec", stamp_sec))

    def build_sample(self, now_sec: float) -> BuildResult:
        now_sec = _timestamp("now_sec", now_sec)
        missing = self._missing_inputs(now_sec)
        if missing and self.publish_only_when_complete:
            return BuildResult(
                sample=None,
                level="WARN",
                message="missing or stale required inputs: " + ", ".join(missing),
                issues=missing,
            )

        skew_violation = self._required_skew_violation(now_sec)
        if skew_violation is not None:
            return BuildResult(
                sample=None,
                level="WARN",
                message=skew_violation,
                issues=["required_input_skew"],
            )

        missing_optional_fields = missing.copy()
        policy_fields = self._policy_fields_with_defaults(
            now_sec, missing_optional_fields
        )
        flattened = flatten_policy_fields(policy_fields)
        observation = self._field_or_default(
            "observation",
            STEPIT_OBSERVATION_DIM,
            now_sec,
            missing_optional_fields,
        )
        l2_error: float | None = None
        if (
            self.validate_observation
            and "observation" not in missing_optional_fields
            and len(flattened) == len(observation)
        ):
            l2_error = observation_l2_error(flattened, observation)
        else:
            missing_optional_fields.append("observation_l2_error")

        sample = SampleData(
            # aligned_target_pos triggers publication and has no source Header, so
            # its local receive timestamp is the best available sample-time anchor.
            sample_stamp_sec=self._aligned_sample_stamp(now_sec),
            policy_fields=policy_fields,
            policy_flattened=flattened,
            robot_state=self._robot_state_or_default(
                now_sec, missing_optional_fields
            ),
            imu=self._imu_or_default(now_sec, missing_optional_fields),
            target_joint_pos=self._field_or_default(
                "target_joint_pos",
                TARGET_JOINT_POS_DIM,
                now_sec,
                missing_optional_fields,
            ),
            aligned_target_pos=self._field_or_default(
                "aligned_target_pos",
                ALIGNED_TARGET_POS_DIM,
                now_sec,
                missing_optional_fields,
            ),
            action=self._field_or_default(
                "action", ACTION_DIM, now_sec, missing_optional_fields
            ),
            stepit_observation=observation,
            observation_l2_error=l2_error,
            missing_optional_fields=sorted(set(missing_optional_fields)),
            source_timestamps_sec=self._fresh_source_timestamps(now_sec),
        )
        return BuildResult(sample=sample, level="OK", message="publishing", issues=[])

    def _missing_inputs(self, now_sec: float) -> list[str]:
        missing: list[str] = []
        for name in REQUIRED_FIELD_NAMES:
            if self._is_missing_or_stale(self.fields.get(name), now_sec):
                missing.append(name)
        if self._is_missing_or_stale(self.robot_state, now_sec):
            missing.append("joint_states")
        if self._is_missing_or_stale(self.imu, now_sec):
            missing.append("imu")
        return missing

    def _is_missing_or_stale(self, value: TimedValue | None, now_sec: float) -> bool:
        if value is None:
            return True
        if self.max_cache_age_sec <= 0:
            return False
        return now_sec - value.stamp_sec > self.max_cache_age_sec

    def _policy_fields_with_defaults(
        self, now_sec: float, missing: list[str]
    ) -> dict[str, list[float]]:
        fields: dict[str, list[float]] = {}
        for spec in POLICY_FIELD_SPECS:
            fields[spec.name] = self._field_or_default(
                spec.name, spec.dim, now_sec, missing
            )
        return fields

    def _field_or_default(
        self, name: str, dim: int, now_sec: float, missing: list[str]
    ) -> list[float]:
        value = self.fields.get(name)
        if self._is_missing_or_stale(value, now_sec):
            if name not in missing:
                missing.append(name)
            return [0.0] * dim
        assert value is not None
        return list(value.value)

    def _robot_state_or_default(
        self, now_sec: float, missing: list[str]
    ) -> RobotLowStateData:
        if self._is_missing_or_stale(self.robot_state, now_sec):
            if "joint_states" not in missing:
                missing.append("joint_states")
            return RobotLowStateData.zero()
        assert self.robot_state is not None
        return self.robot_state.value

    def _imu_or_default(self, now_sec: float, missing: list[str]) -> Any:
        if self._is_missing_or_stale(self.imu, now_sec):
            if "imu" not in missing:
                missing.append("imu")
            return None
        assert self.imu is not None
        return self.imu.value

    def _required_skew_violation(self, now_sec: float) -> str | None:
        if self.max_required_skew_sec <= 0:
            return None

        values = self._required_timed_values(now_sec)
        if len(values) != len(REQUIRED_FIELD_NAMES) + 2:
            return None

        oldest_name, oldest = min(values, key=lambda item: item[1].stamp_sec)
        newest_name, newest = max(values, key=lambda item: item[1].stamp_sec)
        skew_sec = newest.stamp_sec - oldest.stamp_sec
        if skew_sec <= self.max_required_skew_sec:
            return None
        return (
            "required inputs exceed max cross-field skew: "
            f"{skew_sec:.6f}s > {self.max_required_skew_sec:.6f}s "
            f"({oldest_name} -> {newest_name})"
        )

    def _required_timed_values(
        self, now_sec: float
    ) -> list[tuple[str, TimedValue]]:
        values = [
            (name, value)
            for name in REQUIRED_FIELD_NAMES
            if not self._is_missing_or_stale(
                value := self.fields.get(name), now_sec
            )
            and value is not None
        ]
        if not self._is_missing_or_stale(self.robot_state, now_sec):
            assert self.robot_state is not None
            values.append(("joint_states", self.robot_state))
        if not self._is_missing_or_stale(self.imu, now_sec):
            assert self.imu is not None
            values.append(("imu", self.imu))
        return values

    def _aligned_sample_stamp(self, now_sec: float) -> float:
        aligned_target = self.fields.get("aligned_target_pos")
        if self._is_missing_or_stale(aligned_target, now_sec):
            return now_sec
        assert aligned_target is not None
        return aligned_target.stamp_sec

    def _fresh_source_timestamps(self, now_sec: float) -> dict[str, float]:
        timestamps = {
            name: value.stamp_sec
            for name, value in self.fields.items()
            if not self._is_missing_or_stale(value, now_sec)
        }
        if not self._is_missing_or_stale(self.robot_state, now_sec):
            assert self.robot_state is not None
            timestamps["joint_states"] = self.robot_state.stamp_sec
        if not self._is_missing_or_stale(self.imu, now_sec):
            assert self.imu is not None
            timestamps["imu"] = self.imu.stamp_sec
        return dict(sorted(timestamps.items()))


def _validate_joint_state_lengths(
    names: Sequence[str],
    positions: Sequence[float],
    velocities: Sequence[float],
    efforts: Sequence[float],
) -> None:
    name_count = len(names)
    lengths = {
        "position": len(positions),
        "velocity": len(velocities),
        "effort": len(efforts),
    }
    bad_lengths = {
        field_name: length
        for field_name, length in lengths.items()
        if length != name_count
    }
    if bad_lengths:
        details = ", ".join(
            f"{field_name}={length}" for field_name, length in bad_lengths.items()
        )
        raise ValidationError(f"joint_states length mismatch: name={name_count}, {details}")


def _insert_unique(
    rows: dict[str, tuple[float, float, float]],
    base_name: str,
    row: tuple[float, float, float],
    raw_name: str,
) -> None:
    if not base_name:
        raise ValidationError(f"empty base joint name from {raw_name}")
    if base_name in rows:
        raise ValidationError(f"duplicate joint_state entry for {base_name}")
    rows[base_name] = row


def _strip_suffix(name: str, suffix: str) -> str:
    return name[: -len(suffix)]


def _validate_matching_keys(
    label: str, joint_names: Sequence[str], rows: Mapping[str, object]
) -> None:
    joint_name_set = set(joint_names)
    missing = [name for name in joint_names if name not in rows]
    extra = [name for name in rows if name not in joint_name_set]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {label} entries for {missing}")
        if extra:
            parts.append(f"extra {label} entries for {extra}")
        raise ValidationError("; ".join(parts))


def _columns(
    rows: Mapping[str, tuple[float, float, float]], joint_names: Sequence[str]
) -> tuple[list[float], list[float], list[float]]:
    return (
        [rows[name][0] for name in joint_names],
        [rows[name][1] for name in joint_names],
        [rows[name][2] for name in joint_names],
    )


def _finite_float(name: str, value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} is not a valid number: {value!r}") from exc
    if not isfinite(number):
        raise ValidationError(f"{name} must be finite; got {value!r}")
    return number


def _timestamp(name: str, value: object) -> float:
    timestamp = _finite_float(name, value)
    if timestamp < 0:
        raise ValidationError(f"{name} must be non-negative; got {timestamp}")
    return timestamp


def _validate_robot_state(state: RobotLowStateData) -> None:
    if len(state.joint_names) != DOF:
        raise ValidationError(
            f"joint_states contains {len(state.joint_names)} joints; expected {DOF}"
        )
    for field_name in (
        "joint_pos",
        "joint_vel",
        "joint_torque",
        "cmd_joint_pos",
        "cmd_joint_vel",
        "cmd_joint_torque",
        "kp",
        "kd",
        "desired_torque",
    ):
        validate_vector(
            f"joint_states.{field_name}", getattr(state, field_name), DOF
        )
    if len(state.foot_names) != len(state.foot_force):
        raise ValidationError(
            "joint_states foot name/force length mismatch: "
            f"{len(state.foot_names)} != {len(state.foot_force)}"
        )
    validate_vector(
        "joint_states.foot_force", state.foot_force, len(state.foot_names)
    )


def _validate_imu(imu: Any) -> None:
    for group_name, component_names in (
        ("orientation", ("x", "y", "z", "w")),
        ("angular_velocity", ("x", "y", "z")),
        ("linear_acceleration", ("x", "y", "z")),
    ):
        group = getattr(imu, group_name, None)
        if group is None:
            continue
        for component_name in component_names:
            if hasattr(group, component_name):
                _finite_float(
                    f"imu.{group_name}.{component_name}",
                    getattr(group, component_name),
                )

    for covariance_name in (
        "orientation_covariance",
        "angular_velocity_covariance",
        "linear_acceleration_covariance",
    ):
        covariance = getattr(imu, covariance_name, None)
        if covariance is None:
            continue
        validate_vector(f"imu.{covariance_name}", covariance, 9)
