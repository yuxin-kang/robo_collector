"""Pure parsing and validation logic for the robo_state ROS node."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping, Sequence


DOF = 29
ACTION_DIM = DOF
TARGET_JOINT_POS_DIM = DOF
OBSERVATION_DIM = 1545

JOINT_SUFFIX = "_joint"
CMD_SUFFIX = "_cmd"
GAIN_SUFFIX = "_gain"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dim: int


POLICY_FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("motion_joint_pos", 435),
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
    "action": ACTION_DIM,
    "observation": OBSERVATION_DIM,
    **POLICY_FIELD_DIMS,
}

REQUIRED_FIELD_NAMES: tuple[str, ...] = (
    "target_joint_pos",
    "action",
    "observation",
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
    policy_fields: dict[str, list[float]]
    policy_flattened: list[float]
    robot_state: RobotLowStateData
    imu: Any
    target_joint_pos: list[float]
    action: list[float]
    stepit_observation: list[float]
    observation_l2_error: float
    missing_optional_fields: list[str]


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
    return [float(value) for value in values]


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
    return sqrt(
        sum(
            (float(left) - float(right)) ** 2
            for left, right in zip(flattened_policy, stepit_observation)
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
            float(positions[index]),
            float(velocities[index]),
            float(efforts[index]),
        )
        if name.endswith(JOINT_SUFFIX):
            _insert_unique(joint_rows, _strip_suffix(name, JOINT_SUFFIX), row, name)
        elif name.endswith(CMD_SUFFIX):
            _insert_unique(cmd_rows, _strip_suffix(name, CMD_SUFFIX), row, name)
        elif name.endswith(GAIN_SUFFIX):
            _insert_unique(gain_rows, _strip_suffix(name, GAIN_SUFFIX), row, name)
        else:
            foot_names.append(name)
            foot_force.append(float(efforts[index]))

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
        publish_only_when_complete: bool = True,
        validate_observation: bool = True,
    ) -> None:
        self.max_cache_age_sec = float(max_cache_age_sec)
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
        self.fields[name] = TimedValue(vector, float(stamp_sec))
        return vector

    def update_robot_state(
        self, robot_state: RobotLowStateData, stamp_sec: float
    ) -> None:
        self.robot_state = TimedValue(robot_state, float(stamp_sec))

    def update_imu(self, imu: Any, stamp_sec: float) -> None:
        self.imu = TimedValue(imu, float(stamp_sec))

    def build_sample(self, now_sec: float) -> BuildResult:
        missing = self._missing_inputs(float(now_sec))
        if missing and self.publish_only_when_complete:
            return BuildResult(
                sample=None,
                level="WARN",
                message="missing or stale required inputs: " + ", ".join(missing),
                issues=missing,
            )

        missing_optional_fields = missing.copy()
        policy_fields = self._policy_fields_with_defaults(missing_optional_fields)
        flattened = flatten_policy_fields(policy_fields)
        observation = self._field_or_default(
            "observation", OBSERVATION_DIM, missing_optional_fields
        )
        l2_error = (
            observation_l2_error(flattened, observation)
            if self.validate_observation
            else 0.0
        )

        sample = SampleData(
            policy_fields=policy_fields,
            policy_flattened=flattened,
            robot_state=self._robot_state_or_default(missing_optional_fields),
            imu=self._imu_or_default(missing_optional_fields),
            target_joint_pos=self._field_or_default(
                "target_joint_pos", TARGET_JOINT_POS_DIM, missing_optional_fields
            ),
            action=self._field_or_default("action", ACTION_DIM, missing_optional_fields),
            stepit_observation=observation,
            observation_l2_error=l2_error,
            missing_optional_fields=sorted(set(missing_optional_fields)),
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

    def _policy_fields_with_defaults(self, missing: list[str]) -> dict[str, list[float]]:
        fields: dict[str, list[float]] = {}
        for spec in POLICY_FIELD_SPECS:
            fields[spec.name] = self._field_or_default(spec.name, spec.dim, missing)
        return fields

    def _field_or_default(
        self, name: str, dim: int, missing: list[str]
    ) -> list[float]:
        value = self.fields.get(name)
        if value is None:
            if name not in missing:
                missing.append(name)
            return [0.0] * dim
        return list(value.value)

    def _robot_state_or_default(self, missing: list[str]) -> RobotLowStateData:
        if self.robot_state is None:
            if "joint_states" not in missing:
                missing.append("joint_states")
            return RobotLowStateData.zero()
        return self.robot_state.value

    def _imu_or_default(self, missing: list[str]) -> Any:
        if self.imu is None:
            if "imu" not in missing:
                missing.append("imu")
            return None
        return self.imu.value


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
