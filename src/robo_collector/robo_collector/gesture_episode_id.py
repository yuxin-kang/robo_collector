"""Canonical episode-id helpers for gesture-triggered recording."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote, unquote


_PART_SEPARATOR = "__"


@dataclass(frozen=True)
class GestureEpisodeId:
    plan_id: str
    task_slug: str
    trial_index: int
    attempt_index: int


def build_gesture_episode_id(
    *, plan_id: str, task_slug: str, trial_index: int, attempt_index: int
) -> str:
    normalized_plan = validate_episode_component_text(plan_id, field_name="plan_id")
    normalized_task = validate_episode_component_text(task_slug, field_name="task_slug")
    normalized_trial_index = _non_negative_int(
        trial_index,
        field_name="trial_index",
    )
    normalized_attempt_index = _positive_int(
        attempt_index,
        field_name="attempt_index",
    )
    return _PART_SEPARATOR.join(
        (
            f"plan={quote(normalized_plan, safe='')}",
            f"task={quote(normalized_task, safe='')}",
            f"trial={normalized_trial_index:04d}",
            f"attempt={normalized_attempt_index:02d}",
        )
    )


def parse_gesture_episode_id(value: str) -> GestureEpisodeId:
    text = str(value).strip()
    if not text:
        raise ValueError("gesture episode_id is required")

    items = text.split(_PART_SEPARATOR)
    if len(items) != 4:
        raise ValueError(
            "gesture episode_id must contain exactly plan,task,trial,attempt components"
        )

    parts: dict[str, str] = {}
    for item in items:
        key, separator, raw_value = item.partition("=")
        if not separator:
            raise ValueError(f"invalid gesture episode_id component: {item!r}")
        if key in parts:
            raise ValueError(f"duplicate gesture episode_id component: {key!r}")
        parts[key] = raw_value

    required = {"plan", "task", "trial", "attempt"}
    if set(parts) != required:
        raise ValueError(
            "gesture episode_id must contain exactly plan,task,trial,attempt components"
        )

    try:
        trial_index = _non_negative_int(parts["trial"], field_name="trial_index")
        attempt_index = _positive_int(parts["attempt"], field_name="attempt_index")
    except ValueError as exc:
        raise ValueError(f"invalid numeric component in gesture episode_id: {text!r}") from exc

    plan_id = unquote(parts["plan"]).strip()
    task_slug = unquote(parts["task"]).strip()
    try:
        plan_id = validate_episode_component_text(plan_id, field_name="plan_id")
        task_slug = validate_episode_component_text(task_slug, field_name="task_slug")
    except ValueError as exc:
        raise ValueError("plan and task components must decode to valid strings") from exc

    return GestureEpisodeId(
        plan_id=plan_id,
        task_slug=task_slug,
        trial_index=trial_index,
        attempt_index=attempt_index,
    )


def validate_episode_component_text(value: object, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    if _PART_SEPARATOR in normalized:
        raise ValueError(f"{field_name} must not contain {_PART_SEPARATOR!r}")
    return normalized


def _non_negative_int(value: object, *, field_name: str) -> int:
    parsed = _strict_int(value, field_name=field_name)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")
    return parsed


def _positive_int(value: object, *, field_name: str) -> int:
    parsed = _strict_int(value, field_name=field_name)
    if parsed < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")
    return parsed


def _strict_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_name} must be an integer, got {value!r}")
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    raise ValueError(f"{field_name} must be an integer, got {value!r}")
