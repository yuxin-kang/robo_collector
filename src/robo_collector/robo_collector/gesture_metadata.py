"""Metadata reconciliation and advisory progress-log helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gesture_episode_id import parse_gesture_episode_id
from .gesture_plan import GestureTriggerPlan, PlannedTrial


PROGRESS_SCHEMA_VERSION = 1
DEFAULT_PROGRESS_FILENAME = "gesture_trigger_progress.json"


@dataclass(frozen=True)
class TrialMetadataStatus:
    task_slug: str
    trial_index: int
    complete: bool
    latest_state: str
    latest_attempt_index: int
    next_attempt_index: int
    message: str
    episode_id: str = ""

    @property
    def ambiguous(self) -> bool:
        return self.latest_state == "DUPLICATE"


@dataclass(frozen=True)
class MetadataSnapshot:
    dataset_root: Path
    plan_id: str
    statuses: dict[tuple[str, int], TrialMetadataStatus]
    parse_error: str = ""

    @property
    def completed_count(self) -> int:
        return sum(1 for status in self.statuses.values() if status.complete)

    def status_for(self, trial: PlannedTrial) -> TrialMetadataStatus:
        return self.statuses[(trial.task_slug, trial.trial_index)]


@dataclass(frozen=True)
class ProgressCurrent:
    task_slug: str
    trial_index: int
    attempt_index: int
    episode_id: str


@dataclass(frozen=True)
class ProgressEvent:
    ts: str
    event: str
    episode_id: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressLog:
    schema_version: int
    plan_id: str
    dataset_root: str
    updated_at: str
    last_state: str
    current: ProgressCurrent | None
    events: tuple[ProgressEvent, ...]


def scan_plan_metadata(dataset_root: str | Path, plan: GestureTriggerPlan) -> MetadataSnapshot:
    root = Path(dataset_root)
    episodes_path = root / "meta/episodes.jsonl"
    rows, parse_error = _read_jsonl_rows(episodes_path)

    rows_by_attempt: dict[tuple[str, int], dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        episode_id = str(row.get("episode_id", "")).strip()
        if not episode_id:
            continue
        try:
            parsed = parse_gesture_episode_id(episode_id)
        except ValueError:
            continue
        if parsed.plan_id != plan.plan_id:
            continue
        key = (parsed.task_slug, parsed.trial_index)
        rows_by_attempt.setdefault(key, {}).setdefault(parsed.attempt_index, []).append(row)

    statuses: dict[tuple[str, int], TrialMetadataStatus] = {}
    for trial in plan.planned_trials:
        key = (trial.task_slug, trial.trial_index)
        attempts = rows_by_attempt.get(key, {})
        statuses[key] = _classify_trial_attempts(
            trial=trial,
            attempts=attempts,
            dataset_root=root,
        )

    return MetadataSnapshot(
        dataset_root=root,
        plan_id=plan.plan_id,
        statuses=statuses,
        parse_error=parse_error,
    )


def default_progress_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "meta" / DEFAULT_PROGRESS_FILENAME


def load_progress_log(path: str | Path) -> ProgressLog | None:
    progress_path = Path(path)
    if not progress_path.exists():
        return None
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        current = _parse_progress_current(payload.get("current"))
        events = _parse_progress_events(payload.get("events", []))
        return ProgressLog(
            schema_version=_progress_int(
                payload.get("schema_version", PROGRESS_SCHEMA_VERSION)
            ),
            plan_id=str(payload.get("plan_id", "")).strip(),
            dataset_root=str(payload.get("dataset_root", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip(),
            last_state=str(payload.get("last_state", "")).strip(),
            current=current,
            events=events,
        )
    except (TypeError, ValueError):
        return None


def write_progress_log(path: str | Path, progress: ProgressLog) -> None:
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    payload = {
        "schema_version": progress.schema_version,
        "plan_id": progress.plan_id,
        "dataset_root": progress.dataset_root,
        "updated_at": progress.updated_at,
        "last_state": progress.last_state,
        "current": asdict(progress.current) if progress.current is not None else None,
        "events": [asdict(event) for event in progress.events],
    }
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(progress_path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_jsonl_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.exists():
        return [], ""
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return rows, f"failed to read metadata: {exc}"
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            return rows, f"invalid JSONL at line {line_number}: {exc}"
        if not isinstance(row, dict):
            return rows, f"invalid JSONL row at line {line_number}: expected object"
        rows.append(row)
    return rows, ""


def _parse_progress_current(value: Any) -> ProgressCurrent | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("progress current must be an object")
    return ProgressCurrent(
        task_slug=str(value.get("task_slug", "")).strip(),
        trial_index=_progress_int(value.get("trial_index", 0)),
        attempt_index=_progress_int(value.get("attempt_index", 0)),
        episode_id=str(value.get("episode_id", "")).strip(),
    )


def _parse_progress_events(value: Any) -> tuple[ProgressEvent, ...]:
    if not isinstance(value, list):
        raise ValueError("progress events must be a list")
    events: list[ProgressEvent] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("progress event must be an object")
        details = item.get("details", {})
        if not isinstance(details, dict):
            raise ValueError("progress event details must be an object")
        events.append(
            ProgressEvent(
                ts=str(item.get("ts", "")).strip(),
                event=str(item.get("event", "")).strip(),
                episode_id=str(item.get("episode_id", "")).strip(),
                details=details,
            )
        )
    return tuple(events)


def _progress_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("invalid integer")
    if isinstance(value, int):
        return value
    raise ValueError("invalid integer")


def _classify_trial_attempts(
    *,
    trial: PlannedTrial,
    attempts: dict[int, list[dict[str, Any]]],
    dataset_root: Path,
) -> TrialMetadataStatus:
    if not attempts:
        return TrialMetadataStatus(
            task_slug=trial.task_slug,
            trial_index=trial.trial_index,
            complete=False,
            latest_state="MISSING",
            latest_attempt_index=0,
            next_attempt_index=1,
            message="no metadata row for trial yet",
        )

    for attempt_index in sorted(attempts):
        rows = attempts[attempt_index]
        if len(rows) > 1:
            episode_id = str(rows[0].get("episode_id", "")).strip()
            return TrialMetadataStatus(
                task_slug=trial.task_slug,
                trial_index=trial.trial_index,
                complete=False,
                latest_state="DUPLICATE",
                latest_attempt_index=attempt_index,
                next_attempt_index=attempt_index,
                message=f"duplicate metadata rows for episode_id {episode_id}",
                episode_id=episode_id,
            )

    successful_attempts: list[tuple[int, dict[str, Any]]] = []
    for attempt_index, rows in sorted(attempts.items()):
        row = rows[0]
        if _row_is_success(row, trial.task_prompt, dataset_root):
            successful_attempts.append((attempt_index, row))

    if successful_attempts:
        attempt_index, row = successful_attempts[0]
        episode_id = str(row.get("episode_id", "")).strip()
        return TrialMetadataStatus(
            task_slug=trial.task_slug,
            trial_index=trial.trial_index,
            complete=True,
            latest_state="SUCCESS",
            latest_attempt_index=attempt_index,
            next_attempt_index=attempt_index + 1,
            message="metadata row saved successfully",
            episode_id=episode_id,
        )

    latest_attempt_index = max(attempts)
    latest_row = attempts[latest_attempt_index][0]
    latest_state, message = _row_failure_state(
        latest_row, trial.task_prompt, dataset_root
    )
    return TrialMetadataStatus(
        task_slug=trial.task_slug,
        trial_index=trial.trial_index,
        complete=False,
        latest_state=latest_state,
        latest_attempt_index=latest_attempt_index,
        next_attempt_index=latest_attempt_index + 1,
        message=message,
        episode_id=str(latest_row.get("episode_id", "")).strip(),
    )


def _row_is_success(
    row: dict[str, Any], task_prompt: str, dataset_root: Path
) -> bool:
    return _row_failure_state(row, task_prompt, dataset_root)[0] == "SUCCESS"


def _row_failure_state(
    row: dict[str, Any], task_prompt: str, dataset_root: Path
) -> tuple[str, str]:
    tasks = row.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return "TASK_MISMATCH", "metadata row is missing tasks[0]"
    if str(tasks[0]).strip() != task_prompt:
        return (
            "TASK_MISMATCH",
            f"metadata task mismatch: expected {task_prompt!r}, got {tasks[0]!r}",
        )
    length = row.get("length", 0)
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        return "EMPTY", "metadata row length must be a positive integer"

    data_path = _existing_dataset_file(
        dataset_root, row.get("data_path"), label="data_path"
    )
    if data_path is None:
        return "INCOMPLETE_MEDIA", "metadata data_path is missing, unsafe, or empty"

    video_paths = row.get("video_paths")
    if not isinstance(video_paths, dict) or not video_paths:
        legacy_video_path = row.get("video_path")
        video_paths = {"default": legacy_video_path} if legacy_video_path else {}
    if not video_paths:
        return "INCOMPLETE_MEDIA", "metadata row has no video paths"
    for camera_key, relative_path in video_paths.items():
        if (
            _existing_dataset_file(
                dataset_root,
                relative_path,
                label=f"video_paths[{camera_key}]",
            )
            is None
        ):
            return (
                "INCOMPLETE_MEDIA",
                f"video for {camera_key} is missing, unsafe, or empty",
            )
    return "SUCCESS", "metadata and media files are present"


def _existing_dataset_file(
    dataset_root: Path, value: Any, *, label: str
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    root = dataset_root.resolve()
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    try:
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            return None
    except OSError:
        return None
    return candidate
