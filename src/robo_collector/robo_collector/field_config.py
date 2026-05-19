"""YAML field selection for Robo Collector LeRobot datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGET_FIELD_TO_PARQUET = {
    "joint_position": "action.joint_position",
    "aligned_target_pos": "action.aligned_target_pos",
}

STATE_FIELD_TO_PARQUET = {
    "joint_position": "observation.state.joint_position",
    "joint_velocity": "observation.state.joint_velocity",
    "joint_torque": "observation.state.joint_torque",
    "imu_angular_velocity": "observation.state.imu_angular_velocity",
    "imu_linear_acceleration": "observation.state.imu_linear_acceleration",
    "projected_gravity_or_quat": "observation.state.projected_gravity_or_quat",
    "relative_ori_6d": "observation.state.relative_ori_6d",
    "motion_anchor_lin_vel_b": "observation.state.motion_anchor_lin_vel_b",
    "motion_anchor_ang_vel_b": "observation.state.motion_anchor_ang_vel_b",
    "ang_vel_history": "observation.state.ang_vel_history",
    "gravity_history": "observation.state.gravity_history",
    "joint_pos_rel_history": "observation.state.joint_pos_rel_history",
    "joint_vel_history": "observation.state.joint_vel_history",
    "action_history": "observation.state.action_history",
}

POLICY_ACTION_PARQUET_KEY = "action.policy_action"
LEGACY_TARGET_FIELDS = ("joint_position",)
LEGACY_STATE_FIELDS = (
    "joint_position",
    "joint_velocity",
    "joint_torque",
    "imu_angular_velocity",
    "imu_linear_acceleration",
    "projected_gravity_or_quat",
)


class FieldConfigError(ValueError):
    """Raised when a collection field YAML file is invalid."""


@dataclass(frozen=True)
class FieldSelection:
    """Selected robot fields, expressed in user-facing YAML field names."""

    target: tuple[str, ...]
    state: tuple[str, ...]
    include_policy_action: bool = False

    def __post_init__(self) -> None:
        target = tuple(self.target)
        state = tuple(self.state)
        _validate_field_names("target", target, TARGET_FIELD_TO_PARQUET)
        _validate_field_names("state", state, STATE_FIELD_TO_PARQUET)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "state", state)

    @property
    def target_parquet_keys(self) -> tuple[str, ...]:
        return tuple(TARGET_FIELD_TO_PARQUET[field] for field in self.target)

    @property
    def state_parquet_keys(self) -> tuple[str, ...]:
        return tuple(STATE_FIELD_TO_PARQUET[field] for field in self.state)

    @property
    def robot_parquet_keys(self) -> tuple[str, ...]:
        keys = [*self.state_parquet_keys, *self.target_parquet_keys]
        if self.include_policy_action:
            keys.append(POLICY_ACTION_PARQUET_KEY)
        return tuple(keys)

    @property
    def action_fields(self) -> tuple[str, ...]:
        fields = list(self.target)
        if self.include_policy_action:
            fields.append("policy_action")
        return tuple(fields)


def default_field_selection() -> FieldSelection:
    """Return the legacy writer field set."""

    return FieldSelection(
        target=LEGACY_TARGET_FIELDS,
        state=LEGACY_STATE_FIELDS,
        include_policy_action=True,
    )


def load_optional_field_selection(path: str | Path | None) -> FieldSelection | None:
    if path is None:
        return None
    normalized = str(path).strip()
    if not normalized:
        return None
    return load_field_selection(normalized)


def load_field_selection(path: str | Path) -> FieldSelection:
    config_path = Path(path)
    try:
        import yaml
    except ImportError as exc:
        raise FieldConfigError(
            "PyYAML is required to load field_config_path; install python3-yaml "
            "or run scripts/setup_data_collection_env.sh"
        ) from exc

    if not config_path.exists():
        raise FieldConfigError(f"field config file not found: {config_path}")
    if not config_path.is_file():
        raise FieldConfigError(f"field config path is not a file: {config_path}")

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FieldConfigError(
            f"invalid YAML in field config {config_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise FieldConfigError(
            f"failed to read field config {config_path}: {exc}"
        ) from exc

    return field_selection_from_payload(payload, source=str(config_path))


def field_selection_from_payload(
    payload: Any, *, source: str = "field config"
) -> FieldSelection:
    if not isinstance(payload, dict):
        raise FieldConfigError(f"{source}: top level must be a mapping")

    allowed_top_level = {"target", "state"}
    actual_top_level = set(payload)
    if actual_top_level != allowed_top_level:
        got = (
            ",".join(str(key) for key in sorted(actual_top_level, key=str))
            or "<none>"
        )
        raise FieldConfigError(
            f"{source}: top-level keys must be exactly target,state; got {got}"
        )

    target = _parse_group(
        payload["target"],
        group="target",
        supported=TARGET_FIELD_TO_PARQUET,
        source=source,
    )
    state = _parse_group(
        payload["state"],
        group="state",
        supported=STATE_FIELD_TO_PARQUET,
        source=source,
    )
    return FieldSelection(target=target, state=state)


def _parse_group(
    value: Any,
    *,
    group: str,
    supported: dict[str, str],
    source: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise FieldConfigError(f"{source}: {group} must be a non-empty list")
    if not value:
        raise FieldConfigError(f"{source}: {group} must be a non-empty list")

    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise FieldConfigError(
                f"{source}: {group}[{index}] must be a string field name"
            )
        if not item.strip():
            raise FieldConfigError(f"{source}: {group}[{index}] must not be empty")

    fields = tuple(item.strip() for item in value)
    _validate_field_names(group, fields, supported, source=source)
    return fields


def _validate_field_names(
    group: str,
    fields: tuple[str, ...],
    supported: dict[str, str],
    *,
    source: str = "field selection",
) -> None:
    if not fields:
        raise FieldConfigError(f"{source}: {group} must be a non-empty list")

    duplicates = sorted({field for field in fields if fields.count(field) > 1})
    if duplicates:
        raise FieldConfigError(
            f"{source}: duplicate {group} field(s): {','.join(duplicates)}"
        )

    unknown = [field for field in fields if field not in supported]
    if unknown:
        raise FieldConfigError(
            f"{source}: unsupported {group} field(s): {','.join(unknown)}; "
            f"supported fields: {','.join(supported)}"
        )
