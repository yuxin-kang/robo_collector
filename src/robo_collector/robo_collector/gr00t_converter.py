"""Convert Robo Collector outputs into an Isaac-GR00T compatible dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from typing import Any

from .lerobot_dataset import ALIGNED_TARGET_POS_DIM, DOF, STATE_FIELD_SHAPES


POLICY_STATE_ORDER: tuple[str, ...] = (
    "relative_ori_6d",
    "motion_anchor_lin_vel_b",
    "motion_anchor_ang_vel_b",
    "ang_vel_history",
    "gravity_history",
    "joint_pos_rel_history",
    "joint_vel_history",
    "action_history",
)
POLICY_STATE_DIM = sum(int(STATE_FIELD_SHAPES[field][0]) for field in POLICY_STATE_ORDER)
ACTION_SOURCE_TO_KEY = {
    "aligned_target_pos": "action.aligned_target_pos",
    "policy_action": "action.policy_action",
    "joint_position": "action.joint_position",
}
ACTION_SOURCE_DIMS = {
    "aligned_target_pos": ALIGNED_TARGET_POS_DIM,
    "policy_action": DOF,
    "joint_position": DOF,
}


class ConversionError(ValueError):
    """Raised when a source dataset cannot be converted safely."""


@dataclass(frozen=True)
class ConversionResult:
    source_dataset: Path
    output_dataset: Path
    episode_count: int
    frame_count: int
    action_source: str


@dataclass(frozen=True)
class _EpisodeConversionPlan:
    episode_index: int
    converted_rows: list[dict[str, Any]]
    dest_data_rel_path: Path
    dest_video_rel_paths: dict[str, Path]
    source_video_paths: dict[str, Path]
    task_strings: list[str]
    fps: int


def convert_dataset(
    source_root: str | Path,
    dataset_name: str,
    dest_root: str | Path,
    *,
    output_name: str | None = None,
    action_source: str,
) -> ConversionResult:
    if action_source not in ACTION_SOURCE_TO_KEY:
        supported = ",".join(sorted(ACTION_SOURCE_TO_KEY))
        raise ConversionError(
            f"unsupported action_source {action_source!r}; supported: {supported}"
        )

    source_root_path = Path(source_root).resolve()
    dest_root_path = Path(dest_root).resolve()
    source_dataset_name = _validate_dataset_dir_name(dataset_name, label="dataset_name")
    output_dataset_name = _validate_dataset_dir_name(
        output_name or f"{source_dataset_name}_gr00t",
        label="output_name",
    )

    source_dataset = _resolve_dataset_dir(source_root_path, source_dataset_name)
    if not source_dataset.exists():
        raise ConversionError(f"source dataset not found: {source_dataset}")
    if not source_dataset.is_dir():
        raise ConversionError(f"source dataset is not a directory: {source_dataset}")

    output_dataset = dest_root_path / output_dataset_name
    if output_dataset.exists():
        raise ConversionError(
            f"output dataset already exists: {output_dataset}; choose a new --output-name"
        )

    source_info = _load_json(source_dataset / "meta/info.json")
    source_episodes = _load_jsonl(source_dataset / "meta/episodes.jsonl")
    source_tasks = _load_jsonl(source_dataset / "meta/tasks.jsonl")
    camera_keys = _camera_keys_from_info(source_info)
    fps = int(source_info.get("fps", 0))
    if fps <= 0:
        raise ConversionError(f"source info.json has invalid fps: {fps}")

    task_lookup = {
        str(row["task"]): int(row["task_index"])
        for row in source_tasks
        if "task" in row and "task_index" in row
    }
    next_task_index = max(task_lookup.values(), default=-1) + 1

    episode_plans: list[_EpisodeConversionPlan] = []
    total_frames = 0
    for source_episode in sorted(source_episodes, key=lambda row: int(row["episode_index"])):
        episode_plan, next_task_index = _preflight_episode(
            source_dataset=source_dataset,
            source_episode=source_episode,
            camera_keys=camera_keys,
            action_source=action_source,
            task_lookup=task_lookup,
            next_task_index=next_task_index,
            total_frames=total_frames,
            fps=fps,
        )
        episode_plans.append(episode_plan)
        total_frames += len(episode_plan.converted_rows)

    tasks_rows = [
        {"task_index": task_index, "task": task}
        for task, task_index in sorted(task_lookup.items(), key=lambda item: item[1])
    ]
    info = _build_info(
        source_info=source_info,
        camera_keys=camera_keys,
        total_frames=total_frames,
        total_episodes=len(episode_plans),
        total_tasks=len(tasks_rows),
        action_source=action_source,
    )
    modality = _build_modality(camera_keys=camera_keys, action_source=action_source)

    dest_root_path.mkdir(parents=True, exist_ok=True)
    staging_output = dest_root_path / f".{output_dataset_name}.{uuid4().hex}.tmp"
    if staging_output.exists():
        raise ConversionError(f"staging output unexpectedly exists: {staging_output}")

    try:
        _materialize_dataset(
            staging_output=staging_output,
            episode_plans=episode_plans,
            tasks_rows=tasks_rows,
            info=info,
            modality=modality,
        )
        staging_output.replace(output_dataset)
    except Exception:
        shutil.rmtree(staging_output, ignore_errors=True)
        raise

    return ConversionResult(
        source_dataset=source_dataset,
        output_dataset=output_dataset,
        episode_count=len(episode_plans),
        frame_count=total_frames,
        action_source=action_source,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an existing Robo Collector dataset to Isaac-GR00T format."
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Parent directory that contains the source dataset folder.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Source dataset folder name under --source-root.",
    )
    parser.add_argument(
        "--dest-root",
        required=True,
        help="Parent directory where the converted dataset should be written.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Converted dataset folder name. Defaults to <dataset-name>_gr00t.",
    )
    parser.add_argument(
        "--action-source",
        required=True,
        choices=sorted(ACTION_SOURCE_TO_KEY),
        help="Source action column to project into GR00T's single action vector.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = convert_dataset(
        args.source_root,
        args.dataset_name,
        args.dest_root,
        output_name=args.output_name,
        action_source=args.action_source,
    )
    print(f"source_dataset: {result.source_dataset}")
    print(f"output_dataset: {result.output_dataset}")
    print(f"episodes: {result.episode_count}")
    print(f"frames: {result.frame_count}")
    print(f"action_source: {result.action_source}")
    return 0


def _preflight_episode(
    *,
    source_dataset: Path,
    source_episode: dict[str, Any],
    camera_keys: list[str],
    action_source: str,
    task_lookup: dict[str, int],
    next_task_index: int,
    total_frames: int,
    fps: int,
) -> tuple[_EpisodeConversionPlan, int]:
    episode_index = int(source_episode["episode_index"])
    parquet_path = _source_parquet_path(source_dataset, source_episode, episode_index)
    rows = _read_parquet_rows(parquet_path)
    if not rows:
        raise ConversionError(f"source parquet has no rows: {parquet_path}")

    dest_video_rel_paths, source_video_paths = _resolve_video_paths(
        source_dataset=source_dataset,
        source_episode=source_episode,
        first_row=rows[0],
        camera_keys=camera_keys,
        episode_index=episode_index,
    )

    converted_rows: list[dict[str, Any]] = []
    for frame_index, row in enumerate(rows):
        task_index, next_task_index = _task_index_from_row(
            row,
            task_lookup=task_lookup,
            next_task_index=next_task_index,
        )
        converted_rows.append(
            _convert_row(
                row,
                frame_index=frame_index,
                global_index=total_frames + frame_index,
                episode_index=episode_index,
                action_source=action_source,
                task_index=task_index,
                camera_keys=camera_keys,
                dest_video_rel_paths=dest_video_rel_paths,
                episode_length=len(rows),
            )
        )

    task_strings = [
        _task_string_from_index(task_lookup, int(task_index))
        for task_index in sorted({int(row["task_index"]) for row in converted_rows})
    ]
    return (
        _EpisodeConversionPlan(
            episode_index=episode_index,
            converted_rows=converted_rows,
            dest_data_rel_path=Path(f"data/chunk-000/episode_{episode_index:06d}.parquet"),
            dest_video_rel_paths=dest_video_rel_paths,
            source_video_paths=source_video_paths,
            task_strings=task_strings,
            fps=fps,
        ),
        next_task_index,
    )


def _materialize_dataset(
    *,
    staging_output: Path,
    episode_plans: list[_EpisodeConversionPlan],
    tasks_rows: list[dict[str, Any]],
    info: dict[str, Any],
    modality: dict[str, Any],
) -> None:
    meta_dir = staging_output / "meta"
    meta_dir.mkdir(parents=True, exist_ok=False)

    for episode_plan in episode_plans:
        for camera_key, source_video_path in episode_plan.source_video_paths.items():
            dest_video_path = _resolve_path_within_root(
                staging_output,
                str(episode_plan.dest_video_rel_paths[camera_key]),
                label=f"dest video path for {camera_key}",
            )
            dest_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video_path, dest_video_path)

        dest_data_path = _resolve_path_within_root(
            staging_output,
            str(episode_plan.dest_data_rel_path),
            label=f"dest parquet path for episode {episode_plan.episode_index}",
        )
        write_parquet(dest_data_path, episode_plan.converted_rows)

    episodes_rows = [
        {
            "episode_index": episode_plan.episode_index,
            "tasks": episode_plan.task_strings,
            "length": len(episode_plan.converted_rows),
            "fps": episode_plan.fps,
            "data_path": str(episode_plan.dest_data_rel_path),
            "video_paths": {
                camera_key: str(video_rel_path)
                for camera_key, video_rel_path in episode_plan.dest_video_rel_paths.items()
            },
        }
        for episode_plan in episode_plans
    ]
    _write_text(meta_dir / "tasks.jsonl", _jsonl_content(tasks_rows))
    _write_text(meta_dir / "episodes.jsonl", _jsonl_content(episodes_rows))
    _write_text(meta_dir / "info.json", _json_content(info))
    _write_text(meta_dir / "modality.json", _json_content(modality))


def _build_info(
    *,
    source_info: dict[str, Any],
    camera_keys: list[str],
    total_frames: int,
    total_episodes: int,
    total_tasks: int,
    action_source: str,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [POLICY_STATE_DIM],
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": [ACTION_SOURCE_DIMS[action_source]],
            "names": None,
        },
        "annotation.human.action.task_description": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.reward": {"dtype": "float32", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }

    source_features = source_info.get("features", {})
    if not isinstance(source_features, dict):
        raise ConversionError("source info.json features must be a mapping")
    for camera_key in camera_keys:
        feature = source_features.get(camera_key)
        if not isinstance(feature, dict):
            raise ConversionError(f"missing video feature metadata for {camera_key}")
        features[camera_key] = feature

    return {
        "codebase_version": str(source_info.get("codebase_version", "v2.1")),
        "robot_type": str(source_info.get("robot_type", "unitree_g1")),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(camera_keys),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": int(source_info.get("fps", 0)),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-000/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def _build_modality(*, camera_keys: list[str], action_source: str) -> dict[str, Any]:
    state = {}
    start = 0
    for field in POLICY_STATE_ORDER:
        width = int(STATE_FIELD_SHAPES[field][0])
        state[field] = {"start": start, "end": start + width}
        start += width

    action_width = ACTION_SOURCE_DIMS[action_source]
    return {
        "state": state,
        "action": {action_source: {"start": 0, "end": action_width}},
        "video": {
            _camera_stream_from_key(camera_key): {"original_key": camera_key}
            for camera_key in camera_keys
        },
        "annotation": {"annotation.human.action.task_description": {}},
    }


def _convert_row(
    row: dict[str, Any],
    *,
    frame_index: int,
    global_index: int,
    episode_index: int,
    action_source: str,
    task_index: int,
    camera_keys: list[str],
    dest_video_rel_paths: dict[str, Path],
    episode_length: int,
) -> dict[str, Any]:
    state_values: list[float] = []
    for field in POLICY_STATE_ORDER:
        source_key = f"observation.state.{field}"
        if source_key not in row:
            raise ConversionError(f"missing required source state field: {source_key}")
        values = _to_float_list(
            row[source_key], expected_dim=int(STATE_FIELD_SHAPES[field][0])
        )
        state_values.extend(values)
    if len(state_values) != POLICY_STATE_DIM:
        raise ConversionError(
            f"observation.state has dimension {len(state_values)}; expected {POLICY_STATE_DIM}"
        )

    action_key = ACTION_SOURCE_TO_KEY[action_source]
    if action_key not in row:
        raise ConversionError(f"missing required source action field: {action_key}")
    action_values = _to_float_list(
        row[action_key], expected_dim=ACTION_SOURCE_DIMS[action_source]
    )

    converted = {
        "observation.state": state_values,
        "action": action_values,
        "timestamp": float(row["timestamp"]),
        "frame_index": int(row.get("frame_index", frame_index)),
        "episode_index": episode_index,
        "index": global_index,
        "task_index": task_index,
        "annotation.human.action.task_description": task_index,
        "next.reward": 0.0,
        "next.done": frame_index == episode_length - 1,
    }
    for camera_key in camera_keys:
        image_ref = row.get(camera_key)
        if not isinstance(image_ref, dict):
            raise ConversionError(f"missing video reference for {camera_key}")
        converted[camera_key] = {
            "path": str(dest_video_rel_paths[camera_key]),
            "timestamp": float(image_ref["timestamp"]),
        }
    return converted


def _resolve_video_paths(
    *,
    source_dataset: Path,
    source_episode: dict[str, Any],
    first_row: dict[str, Any],
    camera_keys: list[str],
    episode_index: int,
) -> tuple[dict[str, Path], dict[str, Path]]:
    video_paths = source_episode.get("video_paths")
    if video_paths is None:
        video_paths = {
            camera_key: first_row.get(camera_key, {}).get("path") for camera_key in camera_keys
        }
    if not isinstance(video_paths, dict):
        raise ConversionError("episode video_paths must be a mapping")

    dest_rel_paths: dict[str, Path] = {}
    source_paths: dict[str, Path] = {}
    for camera_key in camera_keys:
        source_rel_path = video_paths.get(camera_key)
        if not source_rel_path:
            raise ConversionError(f"missing source video path for {camera_key}")
        source_video_path = _resolve_path_within_root(
            source_dataset,
            str(source_rel_path),
            label=f"video_paths[{camera_key}]",
        )
        if not source_video_path.exists():
            raise ConversionError(f"source video not found: {source_video_path}")
        dest_rel_path = Path(
            f"videos/chunk-000/{camera_key}/episode_{episode_index:06d}.mp4"
        )
        dest_rel_paths[camera_key] = dest_rel_path
        source_paths[camera_key] = source_video_path
    return dest_rel_paths, source_paths


def _task_index_from_row(
    row: dict[str, Any],
    *,
    task_lookup: dict[str, int],
    next_task_index: int,
) -> tuple[int, int]:
    task_value = row.get("annotation.human.action.task_description")
    if isinstance(task_value, int):
        return int(task_value), next_task_index
    if not isinstance(task_value, str) or not task_value.strip():
        raise ConversionError(
            "annotation.human.action.task_description must be a non-empty string or int"
        )
    task = task_value.strip()
    if task not in task_lookup:
        task_lookup[task] = next_task_index
        next_task_index += 1
    return task_lookup[task], next_task_index


def _task_string_from_index(task_lookup: dict[str, int], task_index: int) -> str:
    for task, value in task_lookup.items():
        if value == task_index:
            return task
    raise ConversionError(f"task_index {task_index} is not present in tasks.jsonl")


def _source_parquet_path(
    source_dataset: Path, source_episode: dict[str, Any], episode_index: int
) -> Path:
    relative = source_episode.get("data_path", f"data/train-{episode_index:06d}.parquet")
    parquet_path = _resolve_path_within_root(
        source_dataset,
        str(relative),
        label=f"episodes[{episode_index}].data_path",
    )
    if not parquet_path.exists():
        raise ConversionError(f"source parquet not found: {parquet_path}")
    return parquet_path


def _camera_keys_from_info(source_info: dict[str, Any]) -> list[str]:
    features = source_info.get("features")
    if not isinstance(features, dict):
        raise ConversionError("source info.json features must be a mapping")
    camera_keys = []
    for key, value in features.items():
        if not key.startswith("observation.images.") or not isinstance(value, dict):
            continue
        camera_keys.append(_validate_camera_key(key))
    camera_keys = sorted(camera_keys)
    if not camera_keys:
        raise ConversionError("source info.json does not define any video features")
    return camera_keys


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to convert datasets; run scripts/setup_data_collection_env.sh first"
        ) from exc

    return pq.read_table(path).to_pylist()


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to convert datasets; run scripts/setup_data_collection_env.sh first"
        ) from exc

    if not rows:
        raise ConversionError("cannot write empty parquet row set")

    path.parent.mkdir(parents=True, exist_ok=True)
    first_row = rows[0]
    camera_keys = sorted(key for key in first_row if key.startswith("observation.images."))
    schema_fields = [
        pa.field("observation.state", pa.list_(pa.float32())),
        pa.field("action", pa.list_(pa.float32())),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("annotation.human.action.task_description", pa.int64()),
        pa.field("next.reward", pa.float32()),
        pa.field("next.done", pa.bool_()),
    ]
    video_ref_type = pa.struct(
        [
            pa.field("path", pa.string()),
            pa.field("timestamp", pa.float32()),
        ]
    )
    schema_fields.extend(pa.field(camera_key, video_ref_type) for camera_key in camera_keys)
    schema = pa.schema(schema_fields)

    columns: dict[str, Any] = {
        "observation.state": pa.array(
            [row["observation.state"] for row in rows],
            type=pa.list_(pa.float32()),
        ),
        "action": pa.array(
            [row["action"] for row in rows],
            type=pa.list_(pa.float32()),
        ),
        "timestamp": pa.array([row["timestamp"] for row in rows], type=pa.float32()),
        "frame_index": pa.array([row["frame_index"] for row in rows], type=pa.int64()),
        "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
        "index": pa.array([row["index"] for row in rows], type=pa.int64()),
        "task_index": pa.array([row["task_index"] for row in rows], type=pa.int64()),
        "annotation.human.action.task_description": pa.array(
            [row["annotation.human.action.task_description"] for row in rows],
            type=pa.int64(),
        ),
        "next.reward": pa.array([row["next.reward"] for row in rows], type=pa.float32()),
        "next.done": pa.array([row["next.done"] for row in rows], type=pa.bool_()),
    }
    for camera_key in camera_keys:
        columns[camera_key] = pa.array(
            [row[camera_key] for row in rows],
            type=video_ref_type,
        )
    pq.write_table(pa.table(columns, schema=schema), path, compression="snappy")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConversionError(f"required metadata file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ConversionError(f"required metadata file not found: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _to_float_list(values: Any, *, expected_dim: int) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != expected_dim:
        raise ConversionError(
            f"vector has dimension {len(result)}; expected {expected_dim}"
        )
    return result


def _resolve_dataset_dir(root: Path, dataset_name: str) -> Path:
    return _resolve_path_within_root(root, dataset_name, label="dataset_name")


def _resolve_path_within_root(root: Path, candidate: str, *, label: str) -> Path:
    raw_path = Path(candidate)
    if raw_path.is_absolute():
        raise ConversionError(f"{label} must be relative, got absolute path: {candidate}")
    candidate_path = (root / raw_path).resolve()
    if not candidate_path.is_relative_to(root):
        raise ConversionError(f"{label} escapes root via parent traversal: {candidate}")
    return candidate_path


def _validate_camera_key(camera_key: str) -> str:
    prefix = "observation.images."
    if not camera_key.startswith(prefix):
        raise ConversionError(f"camera key must start with {prefix}: {camera_key}")
    stream_name = camera_key[len(prefix) :].strip()
    if not stream_name:
        raise ConversionError(f"camera key has empty stream name: {camera_key}")
    if "/" in stream_name or "\\" in stream_name:
        raise ConversionError(
            f"camera key stream name must not contain path separators: {camera_key}"
        )
    if stream_name in {".", ".."}:
        raise ConversionError(f"camera key stream name is invalid: {camera_key}")
    return camera_key


def _validate_dataset_dir_name(value: str, *, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ConversionError(f"{label} must be a non-empty folder name")
    candidate = Path(normalized)
    if candidate.is_absolute():
        raise ConversionError(f"{label} must be a folder name, got absolute path: {value}")
    if len(candidate.parts) != 1 or candidate.parts[0] in {".", ".."}:
        raise ConversionError(
            f"{label} must be a single folder name without path traversal: {value}"
        )
    return normalized


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _camera_stream_from_key(camera_key: str) -> str:
    prefix = "observation.images."
    if camera_key.startswith(prefix):
        return camera_key[len(prefix) :]
    return camera_key.rsplit(".", 1)[-1]


def _json_content(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _jsonl_content(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
