"""Plan parsing and validation for gesture-triggered recording."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from .gesture_episode_id import validate_episode_component_text


SUPPORTED_GESTURE_FIELDS = {
    "robot_state.joint_pos",
    "aligned_target_pos",
}


class GesturePlanError(ValueError):
    """Raised when a gesture trigger plan is invalid."""


@dataclass(frozen=True)
class CollectorPlanConfig:
    command_topic: str = "/robo_collector/record_command"
    status_topic: str = "/robo_collector/status"
    fps: float = 30.0
    start_confirm_timeout_sec: float = 5.0
    command_retry_interval_sec: float = 0.5
    command_max_retries: int = 5
    save_confirm_timeout_sec: float = 20.0
    status_timeout_sec: float = 5.0
    auto_discard: bool = False
    ready_policy: str = "wait_for_idle"


@dataclass(frozen=True)
class GestureSourceConfig:
    topic: str
    field: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class GestureReference:
    name: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class GestureCondition:
    reference_name: str
    threshold_l2: float
    stable_samples: int
    release_threshold_l2: float | None = None
    cooldown_sec: float = 0.0


@dataclass(frozen=True)
class TailBounds:
    max_detection_latency_sec: float = 0.2
    max_tail_frames: int = 10


@dataclass(frozen=True)
class PlannedTask:
    task_slug: str
    task_prompt: str
    target_trials: int


@dataclass(frozen=True)
class PlannedTrial:
    task_slug: str
    task_prompt: str
    trial_index: int


@dataclass(frozen=True)
class GestureTriggerPlan:
    plan_id: str
    dataset_root: str
    progress_path: str
    collector: CollectorPlanConfig
    gesture_source: GestureSourceConfig
    references: dict[str, GestureReference]
    task_start_condition: GestureCondition
    task_end_condition: GestureCondition
    return_to_ready_condition: GestureCondition
    tail_bounds: TailBounds
    tasks: tuple[PlannedTask, ...]
    planned_trials: tuple[PlannedTrial, ...]

    @property
    def total_trials(self) -> int:
        return len(self.planned_trials)

    def task_prompt_for(self, task_slug: str) -> str:
        for task in self.tasks:
            if task.task_slug == task_slug:
                return task.task_prompt
        raise KeyError(f"unknown task_slug: {task_slug}")


def load_gesture_trigger_plan(path: str | Path) -> GestureTriggerPlan:
    plan_path = Path(path)
    try:
        import yaml
    except ImportError as exc:
        raise GesturePlanError(
            "PyYAML is required to load a gesture trigger plan; run scripts/setup_data_collection_env.sh"
        ) from exc

    if not plan_path.exists():
        raise GesturePlanError(f"gesture plan file not found: {plan_path}")
    if not plan_path.is_file():
        raise GesturePlanError(f"gesture plan path is not a file: {plan_path}")

    try:
        payload = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GesturePlanError(f"invalid YAML in gesture plan {plan_path}: {exc}") from exc
    except OSError as exc:
        raise GesturePlanError(f"failed to read gesture plan {plan_path}: {exc}") from exc

    return gesture_plan_from_payload(payload, source=str(plan_path))


def gesture_plan_from_payload(payload: Any, *, source: str = "gesture plan") -> GestureTriggerPlan:
    if not isinstance(payload, dict):
        raise GesturePlanError(f"{source}: top level must be a mapping")

    version = _positive_int(payload.get("version", 1), source=f"{source}: version")
    if version != 1:
        raise GesturePlanError(f"{source}: unsupported version {version}; expected 1")

    plan_id = _episode_component_string(
        payload.get("plan_id"),
        source=f"{source}: plan_id",
    )
    dataset_root = _optional_string(payload.get("dataset_root"))
    progress_path = _optional_string(payload.get("progress_path"))
    collector = _parse_collector(payload.get("collector", {}), source=source)
    gesture_source = _parse_gesture_source(payload.get("gesture_source"), source=source)
    references = _parse_references(payload.get("references"), source=source)
    (
        task_start_condition,
        task_end_condition,
        return_to_ready_condition,
    ) = _parse_conditions(payload.get("conditions"), references=references, source=source)
    tail_bounds = _parse_tail_bounds(payload.get("tail_bounds", {}), source=source)
    tasks = _parse_tasks(payload.get("tasks"), source=source)
    planned_trials = tuple(
        PlannedTrial(task.task_slug, task.task_prompt, trial_index)
        for task in tasks
        for trial_index in range(task.target_trials)
    )
    return GestureTriggerPlan(
        plan_id=plan_id,
        dataset_root=dataset_root,
        progress_path=progress_path,
        collector=collector,
        gesture_source=gesture_source,
        references=references,
        task_start_condition=task_start_condition,
        task_end_condition=task_end_condition,
        return_to_ready_condition=return_to_ready_condition,
        tail_bounds=tail_bounds,
        tasks=tasks,
        planned_trials=planned_trials,
    )


def _parse_collector(value: Any, *, source: str) -> CollectorPlanConfig:
    if value is None:
        return CollectorPlanConfig()
    if not isinstance(value, dict):
        raise GesturePlanError(f"{source}: collector must be a mapping")
    ready_policy = str(value.get("ready_policy", "wait_for_idle")).strip() or "wait_for_idle"
    if ready_policy != "wait_for_idle":
        raise GesturePlanError(f"{source}: collector.ready_policy must be wait_for_idle")
    auto_discard = False
    if "auto_discard" in value:
        auto_discard_value = value.get("auto_discard")
        if not isinstance(auto_discard_value, bool):
            raise GesturePlanError(f"{source}: collector.auto_discard must be a boolean")
        auto_discard = auto_discard_value
    if auto_discard:
        raise GesturePlanError(
            f"{source}: collector.auto_discard=true is not implemented in v1"
        )
    return CollectorPlanConfig(
        command_topic=str(value.get("command_topic", "/robo_collector/record_command")).strip(),
        status_topic=str(value.get("status_topic", "/robo_collector/status")).strip(),
        fps=_positive_float(
            value.get("fps", 30.0),
            source=f"{source}: collector.fps",
        ),
        start_confirm_timeout_sec=_positive_float(
            value.get("start_confirm_timeout_sec", 5.0),
            source=f"{source}: collector.start_confirm_timeout_sec",
        ),
        command_retry_interval_sec=_positive_float(
            value.get("command_retry_interval_sec", 0.5),
            source=f"{source}: collector.command_retry_interval_sec",
        ),
        command_max_retries=_positive_int(
            value.get("command_max_retries", 5),
            source=f"{source}: collector.command_max_retries",
        ),
        save_confirm_timeout_sec=_positive_float(
            value.get("save_confirm_timeout_sec", 20.0),
            source=f"{source}: collector.save_confirm_timeout_sec",
        ),
        status_timeout_sec=_positive_float(
            value.get("status_timeout_sec", 5.0),
            source=f"{source}: collector.status_timeout_sec",
        ),
        auto_discard=auto_discard,
        ready_policy=ready_policy,
    )


def _parse_gesture_source(value: Any, *, source: str) -> GestureSourceConfig:
    if not isinstance(value, dict):
        raise GesturePlanError(f"{source}: gesture_source must be a mapping")
    field = _required_non_empty_string(
        value.get("field"), source=f"{source}: gesture_source.field"
    )
    if field not in SUPPORTED_GESTURE_FIELDS:
        raise GesturePlanError(
            f"{source}: unsupported gesture_source.field {field!r}; supported fields: "
            + ",".join(sorted(SUPPORTED_GESTURE_FIELDS))
        )
    indices_value = value.get("indices", [])
    if not isinstance(indices_value, list):
        raise GesturePlanError(f"{source}: gesture_source.indices must be a list")
    indices = tuple(
        _non_negative_int(
            item,
            source=f"{source}: gesture_source.indices[{index}]",
        )
        for index, item in enumerate(indices_value)
    )
    return GestureSourceConfig(
        topic=str(value.get("topic", "/robo_state/sample")).strip() or "/robo_state/sample",
        field=field,
        indices=indices,
    )


def _parse_references(value: Any, *, source: str) -> dict[str, GestureReference]:
    if not isinstance(value, dict) or not value:
        raise GesturePlanError(f"{source}: references must be a non-empty mapping")
    references: dict[str, GestureReference] = {}
    for name, reference_value in value.items():
        normalized_name = _required_non_empty_string(name, source=f"{source}: references key")
        if not isinstance(reference_value, dict):
            raise GesturePlanError(f"{source}: references.{normalized_name} must be a mapping")
        vector_value = reference_value.get("vector")
        if not isinstance(vector_value, list) or not vector_value:
            raise GesturePlanError(
                f"{source}: references.{normalized_name}.vector must be a non-empty list"
            )
        vector = tuple(
            _float_value(
                item,
                source=f"{source}: references.{normalized_name}.vector[{index}]",
            )
            for index, item in enumerate(vector_value)
        )
        references[normalized_name] = GestureReference(normalized_name, vector)
    return references


def _parse_conditions(
    value: Any, *, references: dict[str, GestureReference], source: str
) -> tuple[GestureCondition, GestureCondition, GestureCondition]:
    if not isinstance(value, dict):
        raise GesturePlanError(f"{source}: conditions must be a mapping")
    required = (
        "task_start_condition",
        "task_end_condition",
        "return_to_ready_condition",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise GesturePlanError(f"{source}: missing condition(s): {','.join(missing)}")
    parsed = tuple(
        _parse_condition(
            value[name],
            references=references,
            source=f"{source}: conditions.{name}",
        )
        for name in required
    )
    return parsed  # type: ignore[return-value]


def _parse_condition(
    value: Any, *, references: dict[str, GestureReference], source: str
) -> GestureCondition:
    if not isinstance(value, dict):
        raise GesturePlanError(f"{source}: condition must be a mapping")
    reference_name = _required_non_empty_string(
        value.get("reference_name"), source=f"{source}: reference_name"
    )
    if reference_name not in references:
        raise GesturePlanError(
            f"{source}: unknown reference_name {reference_name!r}; "
            f"known references: {','.join(sorted(references))}"
        )
    return GestureCondition(
        reference_name=reference_name,
        threshold_l2=_positive_float(
            value.get("threshold_l2"),
            source=f"{source}: threshold_l2",
        ),
        stable_samples=_positive_int(
            value.get("stable_samples"),
            source=f"{source}: stable_samples",
        ),
        release_threshold_l2=_optional_positive_float(
            value.get("release_threshold_l2"),
            source=f"{source}: release_threshold_l2",
        ),
        cooldown_sec=_non_negative_float(
            value.get("cooldown_sec", 0.0),
            source=f"{source}: cooldown_sec",
        ),
    )


def _parse_tail_bounds(value: Any, *, source: str) -> TailBounds:
    if value is None:
        return TailBounds()
    if not isinstance(value, dict):
        raise GesturePlanError(f"{source}: tail_bounds must be a mapping")
    return TailBounds(
        max_detection_latency_sec=_positive_float(
            value.get("max_detection_latency_sec", 0.2),
            source=f"{source}: tail_bounds.max_detection_latency_sec",
        ),
        max_tail_frames=_positive_int(
            value.get("max_tail_frames", 10),
            source=f"{source}: tail_bounds.max_tail_frames",
        ),
    )


def _parse_tasks(value: Any, *, source: str) -> tuple[PlannedTask, ...]:
    if not isinstance(value, list) or not value:
        raise GesturePlanError(f"{source}: tasks must be a non-empty list")
    tasks: list[PlannedTask] = []
    seen_slugs: set[str] = set()
    for index, task_value in enumerate(value):
        if not isinstance(task_value, dict):
            raise GesturePlanError(f"{source}: tasks[{index}] must be a mapping")
        task_slug = _episode_component_string(
            task_value.get("task_slug"),
            source=f"{source}: tasks[{index}].task_slug",
        )
        if task_slug in seen_slugs:
            raise GesturePlanError(f"{source}: duplicate task_slug {task_slug!r}")
        seen_slugs.add(task_slug)
        task_prompt = _required_non_empty_string(
            task_value.get("task_prompt"),
            source=f"{source}: tasks[{index}].task_prompt",
        )
        target_trials = _positive_int(
            task_value.get("target_trials"),
            source=f"{source}: tasks[{index}].target_trials",
        )
        tasks.append(PlannedTask(task_slug, task_prompt, target_trials))
    return tuple(tasks)


def _required_non_empty_string(value: Any, *, source: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise GesturePlanError(f"{source} is required")
    return text


def _optional_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _episode_component_string(value: Any, *, source: str) -> str:
    try:
        return validate_episode_component_text(value, field_name=source)
    except ValueError as exc:
        raise GesturePlanError(str(exc)) from exc


def _positive_int(value: Any, *, source: str) -> int:
    if isinstance(value, bool):
        raise GesturePlanError(f"{source} must be a positive integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise GesturePlanError(f"{source} must be a positive integer")
        parsed = int(value)
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise GesturePlanError(f"{source} must be a positive integer") from exc
    if parsed <= 0:
        raise GesturePlanError(f"{source} must be a positive integer")
    return parsed


def _non_negative_int(value: Any, *, source: str) -> int:
    if isinstance(value, bool):
        raise GesturePlanError(f"{source} must be a non-negative integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise GesturePlanError(f"{source} must be a non-negative integer")
        parsed = int(value)
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise GesturePlanError(f"{source} must be a non-negative integer") from exc
    if parsed < 0:
        raise GesturePlanError(f"{source} must be a non-negative integer")
    return parsed


def _float_value(value: Any, *, source: str) -> float:
    if isinstance(value, bool):
        raise GesturePlanError(f"{source} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GesturePlanError(f"{source} must be a number") from exc
    if not math.isfinite(parsed):
        raise GesturePlanError(f"{source} must be a finite number")
    return parsed


def _positive_float(value: Any, *, source: str) -> float:
    if isinstance(value, bool):
        raise GesturePlanError(f"{source} must be a positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GesturePlanError(f"{source} must be a positive number") from exc
    if not math.isfinite(parsed):
        raise GesturePlanError(f"{source} must be a finite positive number")
    if parsed <= 0.0:
        raise GesturePlanError(f"{source} must be a positive number")
    return parsed


def _optional_positive_float(value: Any, *, source: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, source=source)


def _non_negative_float(value: Any, *, source: str) -> float:
    if isinstance(value, bool):
        raise GesturePlanError(f"{source} must be a non-negative number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GesturePlanError(f"{source} must be a non-negative number") from exc
    if not math.isfinite(parsed):
        raise GesturePlanError(f"{source} must be a finite non-negative number")
    if parsed < 0.0:
        raise GesturePlanError(f"{source} must be a non-negative number")
    return parsed
