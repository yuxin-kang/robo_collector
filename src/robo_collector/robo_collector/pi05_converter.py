"""Convert Robo Collector outputs into an OpenPI pi0.5 LeRobot dataset."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from .conversion_provenance import (
    is_raw_episode,
    is_mcap_episode,
    materialize_raw_source,
    reusable_conversion,
    validate_raw_source_ready,
    write_conversion_provenance,
    episode_provenance_from_row,
)
from .lerobot_dataset import DOF


HEAD_CAMERA_KEY = "observation.images.head"
EGO_CAMERA_KEY = "observation.images.ego_view"
HEAD_IMAGE_KEY = "head_image"
EGO_IMAGE_KEY = "ego_image"
TASK_KEY = "annotation.human.action.task_description"
DEFAULT_STATE_KEY = "observation.state.joint_position"
DEFAULT_ACTION_KEY = "action.policy_action"
DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNKS_SIZE = 1000


class FrameReader(Protocol):
    frame_count: int
    shape: tuple[int, int, int]

    def read_rgb(self, index: int) -> Any: ...

    def close(self) -> None: ...


FrameReaderFactory = Callable[[Path], FrameReader]
ImageEncoder = Callable[[Any], bytes]


class ConversionError(ValueError):
    """Raised when a source dataset cannot be converted safely."""


@dataclass(frozen=True)
class ConversionResult:
    source_dataset: Path
    output_dataset: Path
    episode_count: int
    frame_count: int
    state_key: str
    action_key: str


@dataclass(frozen=True)
class _EpisodePlan:
    episode_index: int
    global_start_index: int
    source_parquet_path: Path
    frame_count: int
    task_strings: list[str]
    source_video_paths: dict[str, Path]
    image_shapes: dict[str, tuple[int, int, int]]
    dest_data_rel_path: Path
    source_provenance: dict[str, Any] = field(default_factory=dict)


def convert_dataset(
    source_root: str | Path,
    dataset_name: str,
    dest_root: str | Path,
    *,
    output_name: str | None = None,
    state_key: str = DEFAULT_STATE_KEY,
    action_key: str = DEFAULT_ACTION_KEY,
    history_window_index: int = -1,
    frame_reader_factory: FrameReaderFactory | None = None,
    image_encoder: ImageEncoder | None = None,
    _provenance_source_dataset: Path | None = None,
) -> ConversionResult:
    source_root_path = Path(source_root).resolve()
    dest_root_path = Path(dest_root).resolve()
    source_dataset_name = _validate_dataset_dir_name(dataset_name, label="dataset_name")
    output_dataset_name = _validate_dataset_dir_name(
        output_name or f"{source_dataset_name}_pi05",
        label="output_name",
    )
    state_key = _validate_column_key(state_key, label="state_key")
    action_key = _validate_column_key(action_key, label="action_key")
    frame_reader_factory = frame_reader_factory or OpenCvFrameReader
    image_encoder = image_encoder or encode_png

    source_dataset = _resolve_dataset_dir(source_root_path, source_dataset_name)
    if not source_dataset.exists():
        raise ConversionError(f"source dataset not found: {source_dataset}")
    if not source_dataset.is_dir():
        raise ConversionError(f"source dataset is not a directory: {source_dataset}")

    if is_raw_episode(source_dataset) or is_mcap_episode(source_dataset):
        # Validate the current source before checking an existing output.  A
        # READY conversion must not be reused after raw QC moves to REVIEW or
        # REJECT.
        try:
            validate_raw_source_ready(source_dataset)
        except ValueError as exc:
            raise ConversionError(str(exc)) from exc

    output_dataset = dest_root_path / output_dataset_name
    conversion_config = {
        "action_key": action_key,
        "history_window_index": history_window_index,
        "output_name": output_dataset_name,
        "state_key": state_key,
    }
    if output_dataset.exists():
        if not reusable_conversion(output_dataset, source_dataset,
                                   converter_version="robo_collector.pi05_converter.v1",
                                   output_schema_version="openpi.pi05.v1",
                                   conversion_config=conversion_config):
            raise ConversionError(f"output dataset already exists and is not a complete matching conversion: {output_dataset}")
        info = _load_json(output_dataset / "meta/info.json")
        return ConversionResult(source_dataset, output_dataset, int(info["total_episodes"]), int(info["total_frames"]), state_key, action_key)

    if is_raw_episode(source_dataset) or is_mcap_episode(source_dataset):
        temporary_root, materialized_dataset = materialize_raw_source(
            source_dataset, dest_root_path
        )
        try:
            converted = convert_dataset(
                materialized_dataset.parent,
                materialized_dataset.name,
                dest_root_path,
                output_name=output_dataset_name,
                state_key=state_key,
                action_key=action_key,
                history_window_index=history_window_index,
                frame_reader_factory=frame_reader_factory,
                image_encoder=image_encoder,
                _provenance_source_dataset=source_dataset,
            )
            return ConversionResult(
                source_dataset=source_dataset,
                output_dataset=converted.output_dataset,
                episode_count=converted.episode_count,
                frame_count=converted.frame_count,
                state_key=converted.state_key,
                action_key=converted.action_key,
            )
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    source_info = _load_json(source_dataset / "meta/info.json")
    source_episodes = _load_jsonl(source_dataset / "meta/episodes.jsonl")
    source_tasks = _load_jsonl(source_dataset / "meta/tasks.jsonl")
    fps = int(source_info.get("fps", 0))
    if fps <= 0:
        raise ConversionError(f"source info.json has invalid fps: {fps}")
    _validate_required_camera_features(source_info)
    _validate_dense_episode_indices(source_episodes)

    task_lookup = {
        str(row["task"]): int(row["task_index"])
        for row in source_tasks
        if "task" in row and "task_index" in row
    }
    next_task_index = max(task_lookup.values(), default=-1) + 1

    episode_plans: list[_EpisodePlan] = []
    total_frames = 0
    expected_image_shapes: dict[str, tuple[int, int, int]] | None = None
    for source_episode in sorted(source_episodes, key=lambda row: int(row["episode_index"])):
        episode_plan, next_task_index = _preflight_episode(
            source_dataset=source_dataset,
            source_episode=source_episode,
            global_start_index=total_frames,
            task_lookup=task_lookup,
            next_task_index=next_task_index,
            state_key=state_key,
            action_key=action_key,
            history_window_index=history_window_index,
            frame_reader_factory=frame_reader_factory,
        )
        if expected_image_shapes is None:
            expected_image_shapes = dict(episode_plan.image_shapes)
        elif episode_plan.image_shapes != expected_image_shapes:
            raise ConversionError(
                "source camera shapes changed across episodes: "
                f"{expected_image_shapes} -> {episode_plan.image_shapes}"
            )
        episode_plans.append(episode_plan)
        total_frames += episode_plan.frame_count

    if not episode_plans or expected_image_shapes is None:
        raise ConversionError("source dataset has no episodes to convert")

    tasks_rows = [
        {"task_index": task_index, "task": task}
        for task, task_index in sorted(task_lookup.items(), key=lambda item: item[1])
    ]
    info = _build_info(
        source_info=source_info,
        total_frames=total_frames,
        total_episodes=len(episode_plans),
        total_tasks=len(tasks_rows),
        image_shapes=expected_image_shapes,
        fps=fps,
    )

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
            state_key=state_key,
            action_key=action_key,
            history_window_index=history_window_index,
            frame_reader_factory=frame_reader_factory,
            image_encoder=image_encoder,
            task_lookup=task_lookup,
        )
        write_conversion_provenance(
            staging_output,
            _provenance_source_dataset or source_dataset,
            converter_version="robo_collector.pi05_converter.v1",
            output_schema_version="openpi.pi05.v1",
            conversion_config=conversion_config,
            action_source=action_key,
            state_source=state_key,
            selection_policy="source_frame_index_order",
        )
        _fsync_tree(staging_output)
        staging_output.replace(output_dataset)
        _fsync_dir(dest_root_path)
    except Exception:
        shutil.rmtree(staging_output, ignore_errors=True)
        raise

    return ConversionResult(
        source_dataset=source_dataset,
        output_dataset=output_dataset,
        episode_count=len(episode_plans),
        frame_count=total_frames,
        state_key=state_key,
        action_key=action_key,
    )


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        _fsync_dir(path)
    _fsync_dir(root)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an existing Robo Collector dataset to OpenPI pi0.5 format."
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
        help="Converted dataset folder name. Defaults to <dataset-name>_pi05.",
    )
    parser.add_argument(
        "--state-key",
        default=DEFAULT_STATE_KEY,
        help=f"Source parquet column for the 29-dim G1 state. Default: {DEFAULT_STATE_KEY}.",
    )
    parser.add_argument(
        "--action-key",
        default=DEFAULT_ACTION_KEY,
        help=f"Source parquet column for the 29-dim G1 action. Default: {DEFAULT_ACTION_KEY}.",
    )
    parser.add_argument(
        "--history-window-index",
        type=int,
        default=-1,
        help=(
            "Window index to extract when --state-key or --action-key is a flat "
            "history vector with a length that is a multiple of 29."
        ),
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
        state_key=args.state_key,
        action_key=args.action_key,
        history_window_index=args.history_window_index,
    )
    print(f"source_dataset: {result.source_dataset}")
    print(f"output_dataset: {result.output_dataset}")
    print(f"episodes: {result.episode_count}")
    print(f"frames: {result.frame_count}")
    print(f"state_key: {result.state_key}")
    print(f"action_key: {result.action_key}")
    return 0


class OpenCvFrameReader:
    def __init__(self, path: Path) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required to read source videos for pi0.5 conversion; "
                "install python3-opencv or run scripts/setup_data_collection_env.sh"
            ) from exc

        self.path = path
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(str(path))
        try:
            if not self._capture.isOpened():
                raise ConversionError(f"could not open source video: {path}")
            self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.frame_count <= 0 or width <= 0 or height <= 0:
                raise ConversionError(f"source video has invalid metadata: {path}")
            self.shape = (height, width, 3)
        except Exception:
            try:
                self._capture.release()
            except Exception:
                pass
            raise

    def read_rgb(self, index: int) -> Any:
        self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = self._capture.read()
        if not ok or bgr is None:
            raise ConversionError(f"could not read frame {index} from {self.path}")
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._capture.release()


def encode_png(rgb_frame: Any) -> bytes:
    rgb = _normalize_rgb(rgb_frame)
    try:
        import cv2

        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if ok:
            return bytes(encoded)
    except ImportError:
        pass

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV or Pillow is required to encode PNG image columns"
        ) from exc

    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _preflight_episode(
    *,
    source_dataset: Path,
    source_episode: dict[str, Any],
    global_start_index: int,
    task_lookup: dict[str, int],
    next_task_index: int,
    state_key: str,
    action_key: str,
    history_window_index: int,
    frame_reader_factory: FrameReaderFactory,
) -> tuple[_EpisodePlan, int]:
    episode_index = int(source_episode["episode_index"])
    parquet_path = _source_parquet_path(source_dataset, source_episode, episode_index)
    frame_count = _parquet_row_count(parquet_path)
    if frame_count <= 0:
        raise ConversionError(f"source parquet has no rows: {parquet_path}")
    row_iterator = _iter_parquet_rows(parquet_path)
    first_row = next(row_iterator)

    source_video_paths = _resolve_video_paths(
        source_dataset=source_dataset,
        source_episode=source_episode,
        first_row=first_row,
        episode_index=episode_index,
    )
    image_shapes = _read_image_shapes(
        source_video_paths=source_video_paths,
        frame_reader_factory=frame_reader_factory,
        required_frame_count=frame_count,
    )

    task_indices: set[int] = set()
    rows_seen = 0
    for row_position, row in enumerate(_prepend_row(first_row, row_iterator)):
        actual_frame_index = _frame_index(row)
        if actual_frame_index != row_position:
            raise ConversionError(
                f"episode {episode_index} frame_index values must be dense "
                f"0..N-1 before video sampling; expected {row_position}, "
                f"got {actual_frame_index}"
            )
        task_index, next_task_index = _task_index_from_row(
            row,
            task_lookup=task_lookup,
            next_task_index=next_task_index,
        )
        task_indices.add(task_index)
        _vector(row, state_key, history_window_index=history_window_index, name="state")
        _vector(row, action_key, history_window_index=history_window_index, name="action")
        rows_seen += 1
    if rows_seen != frame_count:
        raise ConversionError(
            f"source parquet row count changed while reading: {parquet_path}"
        )

    task_strings = [
        _task_string_from_index(task_lookup, int(task_index))
        for task_index in sorted(task_indices)
    ]
    source_provenance = episode_provenance_from_row(source_episode)
    return (
        _EpisodePlan(
            episode_index=episode_index,
            global_start_index=global_start_index,
            source_parquet_path=parquet_path,
            frame_count=frame_count,
            task_strings=task_strings,
            source_video_paths=source_video_paths,
            image_shapes=image_shapes,
            dest_data_rel_path=_dest_data_rel_path(episode_index),
            source_provenance=source_provenance,
        ),
        next_task_index,
    )


def _materialize_dataset(
    *,
    staging_output: Path,
    episode_plans: list[_EpisodePlan],
    tasks_rows: list[dict[str, Any]],
    info: dict[str, Any],
    state_key: str,
    action_key: str,
    history_window_index: int,
    frame_reader_factory: FrameReaderFactory,
    image_encoder: ImageEncoder,
    task_lookup: dict[str, int],
) -> None:
    meta_dir = staging_output / "meta"
    meta_dir.mkdir(parents=True, exist_ok=False)
    episode_stats_rows: list[dict[str, Any]] = []

    for episode_plan in episode_plans:
        dest_data_path = _resolve_path_within_root(
            staging_output,
            str(episode_plan.dest_data_rel_path),
            label=f"dest parquet path for episode {episode_plan.episode_index}",
        )
        stats = _convert_episode_to_parquet(
            episode_plan=episode_plan,
            dest_data_path=dest_data_path,
            state_key=state_key,
            action_key=action_key,
            history_window_index=history_window_index,
            frame_reader_factory=frame_reader_factory,
            image_encoder=image_encoder,
            task_lookup=task_lookup,
        )
        episode_stats_rows.append(
            {"episode_index": episode_plan.episode_index, "stats": stats}
        )

    episodes_rows = [
        {
            "episode_index": episode_plan.episode_index,
            "tasks": episode_plan.task_strings,
            "length": episode_plan.frame_count,
            **episode_plan.source_provenance,
            **(
                {"source_provenance": dict(episode_plan.source_provenance)}
                if episode_plan.source_provenance
                else {}
            ),
        }
        for episode_plan in episode_plans
    ]
    _write_text(meta_dir / "tasks.jsonl", _jsonl_content(tasks_rows))
    _write_text(meta_dir / "episodes.jsonl", _jsonl_content(episodes_rows))
    _write_text(meta_dir / "episodes_stats.jsonl", _jsonl_content(episode_stats_rows))
    _write_text(meta_dir / "info.json", _json_content(info))


def _convert_episode_to_parquet(
    *,
    episode_plan: _EpisodePlan,
    dest_data_path: Path,
    state_key: str,
    action_key: str,
    history_window_index: int,
    frame_reader_factory: FrameReaderFactory,
    image_encoder: ImageEncoder,
    task_lookup: dict[str, int],
) -> dict[str, Any]:
    readers = _open_frame_readers(episode_plan.source_video_paths, frame_reader_factory)
    image_stats = {
        HEAD_IMAGE_KEY: _ImageStatsAccumulator(),
        EGO_IMAGE_KEY: _ImageStatsAccumulator(),
    }
    numeric_stats = {
        key: _NumericStatsAccumulator()
        for key in (
            "state",
            "actions",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        )
    }

    def converted_rows():
        rows_seen = 0
        for row_position, row in enumerate(
            _iter_parquet_rows(episode_plan.source_parquet_path)
        ):
            frame_index = _frame_index(row)
            if frame_index != row_position:
                raise ConversionError(
                    f"episode {episode_plan.episode_index} frame_index changed "
                    f"after preflight: expected {row_position}, got {frame_index}"
                )
            head_rgb = _normalize_rgb(readers[HEAD_IMAGE_KEY].read_rgb(frame_index))
            ego_rgb = _normalize_rgb(readers[EGO_IMAGE_KEY].read_rgb(frame_index))
            _validate_image_shape(
                HEAD_IMAGE_KEY,
                head_rgb,
                episode_plan.image_shapes[HEAD_IMAGE_KEY],
            )
            _validate_image_shape(
                EGO_IMAGE_KEY,
                ego_rgb,
                episode_plan.image_shapes[EGO_IMAGE_KEY],
            )

            state = _vector(
                row,
                state_key,
                history_window_index=history_window_index,
                name="state",
            )
            actions = _vector(
                row,
                action_key,
                history_window_index=history_window_index,
                name="action",
            )
            timestamp = float(row["timestamp"])
            output_index = _global_index_for_row(episode_plan, row_position)
            task_index = _known_task_index_from_row(row, task_lookup)
            output_frame_index = row_position

            image_stats[HEAD_IMAGE_KEY].update(head_rgb)
            image_stats[EGO_IMAGE_KEY].update(ego_rgb)
            values = {
                "state": state,
                "actions": actions,
                "timestamp": timestamp,
                "frame_index": output_frame_index,
                "episode_index": episode_plan.episode_index,
                "index": output_index,
                "task_index": task_index,
            }
            for key, value in values.items():
                numeric_stats[key].update(value)

            rows_seen += 1
            yield {
                HEAD_IMAGE_KEY: {
                    "bytes": image_encoder(head_rgb),
                    "path": f"frame_{output_frame_index:06d}.png",
                },
                EGO_IMAGE_KEY: {
                    "bytes": image_encoder(ego_rgb),
                    "path": f"frame_{output_frame_index:06d}.png",
                },
                **values,
            }
        if rows_seen != episode_plan.frame_count:
            raise ConversionError(
                "source parquet row count changed between preflight and conversion: "
                f"{episode_plan.source_parquet_path}"
            )

    try:
        write_parquet_stream(dest_data_path, converted_rows(), batch_size=16)
    finally:
        for reader in readers.values():
            reader.close()

    stats: dict[str, Any] = {
        HEAD_IMAGE_KEY: image_stats[HEAD_IMAGE_KEY].to_json(),
        EGO_IMAGE_KEY: image_stats[EGO_IMAGE_KEY].to_json(),
    }
    for key, accumulator in numeric_stats.items():
        stats[key] = accumulator.to_json()
    return stats


def _dest_data_rel_path(episode_index: int) -> Path:
    episode_chunk = int(episode_index) // CHUNKS_SIZE
    return Path(
        DEFAULT_DATA_PATH.format(
            episode_chunk=episode_chunk,
            episode_index=int(episode_index),
        )
    )


def _open_frame_readers(
    source_video_paths: dict[str, Path],
    frame_reader_factory: FrameReaderFactory,
) -> dict[str, FrameReader]:
    readers: dict[str, FrameReader] = {}
    try:
        for image_key, path in source_video_paths.items():
            readers[image_key] = frame_reader_factory(path)
    except Exception:
        for reader in readers.values():
            reader.close()
        raise
    return readers


def _global_index_for_row(episode_plan: _EpisodePlan, row_position: int) -> int:
    return episode_plan.global_start_index + row_position


def _build_info(
    *,
    source_info: dict[str, Any],
    total_frames: int,
    total_episodes: int,
    total_tasks: int,
    image_shapes: dict[str, tuple[int, int, int]],
    fps: int,
) -> dict[str, Any]:
    return {
        "codebase_version": "v2.1",
        "robot_type": str(source_info.get("robot_type", "unitree_g1")),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": 0,
        "total_chunks": max(1, (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE),
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "features": {
            HEAD_IMAGE_KEY: _image_feature(image_shapes[HEAD_IMAGE_KEY]),
            EGO_IMAGE_KEY: _image_feature(image_shapes[EGO_IMAGE_KEY]),
            "state": {
                "dtype": "float32",
                "shape": [DOF],
                "names": ["joint"],
            },
            "actions": {
                "dtype": "float32",
                "shape": [DOF],
                "names": ["joint"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def _image_feature(shape: tuple[int, int, int]) -> dict[str, Any]:
    return {
        "dtype": "image",
        "shape": [int(shape[0]), int(shape[1]), int(shape[2])],
        "names": ["height", "width", "channel"],
    }


def _validate_required_camera_features(source_info: dict[str, Any]) -> None:
    features = source_info.get("features")
    if not isinstance(features, dict):
        raise ConversionError("source info.json features must be a mapping")
    for camera_key in (HEAD_CAMERA_KEY, EGO_CAMERA_KEY):
        if camera_key not in features:
            raise ConversionError(f"missing required source camera feature: {camera_key}")
        _validate_camera_key(camera_key)


def _resolve_video_paths(
    *,
    source_dataset: Path,
    source_episode: dict[str, Any],
    first_row: dict[str, Any],
    episode_index: int,
) -> dict[str, Path]:
    video_paths = source_episode.get("video_paths")
    if video_paths is None:
        video_paths = {
            camera_key: first_row.get(camera_key, {}).get("path")
            for camera_key in (HEAD_CAMERA_KEY, EGO_CAMERA_KEY)
        }
    if not isinstance(video_paths, dict):
        raise ConversionError("episode video_paths must be a mapping")

    resolved = {}
    for camera_key, image_key in (
        (HEAD_CAMERA_KEY, HEAD_IMAGE_KEY),
        (EGO_CAMERA_KEY, EGO_IMAGE_KEY),
    ):
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
        _validate_camera_key(camera_key)
        resolved[image_key] = source_video_path
    return resolved


def _read_image_shapes(
    *,
    source_video_paths: dict[str, Path],
    frame_reader_factory: FrameReaderFactory,
    required_frame_count: int,
) -> dict[str, tuple[int, int, int]]:
    shapes = {}
    for image_key, path in source_video_paths.items():
        reader = frame_reader_factory(path)
        try:
            if reader.frame_count < required_frame_count:
                raise ConversionError(
                    f"source video {path} has {reader.frame_count} frame(s); "
                    f"expected at least {required_frame_count}"
                )
            shape = tuple(int(value) for value in reader.shape)
            if len(shape) != 3 or shape[2] != 3 or shape[0] <= 0 or shape[1] <= 0:
                raise ConversionError(f"source video has invalid RGB shape for {image_key}: {shape}")
            shapes[image_key] = shape
        finally:
            reader.close()
    return shapes


def _task_index_from_row(
    row: dict[str, Any],
    *,
    task_lookup: dict[str, int],
    next_task_index: int,
) -> tuple[int, int]:
    task_value = row.get(TASK_KEY)
    if isinstance(task_value, int):
        return int(task_value), next_task_index
    if isinstance(task_value, bytes):
        task_value = task_value.decode("utf-8")
    if not isinstance(task_value, str) or not task_value.strip():
        raise ConversionError(f"{TASK_KEY} must be a non-empty string or int")
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


def _known_task_index_from_row(
    row: dict[str, Any], task_lookup: dict[str, int]
) -> int:
    task_value = row.get(TASK_KEY)
    if isinstance(task_value, int):
        task_index = int(task_value)
        if task_index not in task_lookup.values():
            raise ConversionError(
                f"task_index changed after preflight: {task_index}"
            )
        return task_index
    if isinstance(task_value, bytes):
        task_value = task_value.decode("utf-8")
    if not isinstance(task_value, str) or not task_value.strip():
        raise ConversionError(f"{TASK_KEY} must be a non-empty string or int")
    task = task_value.strip()
    if task not in task_lookup:
        raise ConversionError(f"task changed after preflight: {task!r}")
    return task_lookup[task]


def _vector(
    row: dict[str, Any],
    key: str,
    *,
    history_window_index: int,
    name: str,
) -> list[float]:
    if key not in row:
        raise ConversionError(f"missing required source {name} field: {key}")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required to convert pi0.5 vector fields; run "
            "scripts/setup_data_collection_env.sh first"
        ) from exc

    value = np.asarray(row[key], dtype=np.float32)
    if value.shape == (DOF,):
        return [float(item) for item in value.tolist()]
    if value.ndim == 1 and value.size % DOF == 0:
        try:
            selected = value.reshape(-1, DOF)[history_window_index]
        except IndexError as exc:
            raise ConversionError(
                f"{key} has {value.size // DOF} history window(s); "
                f"history_window_index={history_window_index} is out of range"
            ) from exc
        return [float(item) for item in selected.tolist()]
    if value.ndim == 2 and value.shape[-1] == DOF:
        try:
            selected = value[history_window_index]
        except IndexError as exc:
            raise ConversionError(
                f"{key} has {value.shape[0]} history window(s); "
                f"history_window_index={history_window_index} is out of range"
            ) from exc
        return [float(item) for item in selected.tolist()]
    raise ConversionError(f"{key} has shape {value.shape}; expected ({DOF},) or a history multiple")


def _validate_dense_episode_indices(source_episodes: list[dict[str, Any]]) -> None:
    episode_indices = sorted(int(row["episode_index"]) for row in source_episodes)
    expected = list(range(len(source_episodes)))
    if episode_indices != expected:
        raise ConversionError(
            "source episode_index values must be dense 0..N-1 for LeRobot loading; "
            f"got {episode_indices}"
        )


def _validate_dense_frame_indices(rows: list[dict[str, Any]], *, episode_index: int) -> None:
    frame_indices = [_frame_index(row) for row in rows]
    expected = list(range(len(rows)))
    if frame_indices != expected:
        preview = ", ".join(str(value) for value in frame_indices[:10])
        raise ConversionError(
            f"episode {episode_index} frame_index values must be dense 0..N-1 before video sampling; "
            f"got [{preview}] for {len(rows)} row(s)"
        )


def _frame_index(row: dict[str, Any]) -> int:
    if "frame_index" in row:
        return int(row["frame_index"])
    if "index" in row:
        return int(row["index"])
    raise ConversionError("missing required parquet column: frame_index")


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


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to convert datasets; run scripts/setup_data_collection_env.sh first"
        ) from exc

    return int(pq.ParquetFile(path).metadata.num_rows)


def _iter_parquet_rows(path: Path, *, batch_size: int = 256):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to convert datasets; run scripts/setup_data_collection_env.sh first"
        ) from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _prepend_row(first_row: dict[str, Any], rows):
    yield first_row
    yield from rows


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    write_parquet_stream(path, iter(rows))


def write_parquet_stream(path: Path, rows, *, batch_size: int = 16) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to write pi0.5 parquet files; run "
            "scripts/setup_data_collection_env.sh first"
        ) from exc

    image_type = pa.struct(
        [
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string()),
        ]
    )
    vector_type = pa.list_(pa.float32(), DOF)
    schema = pa.schema(
        [
            pa.field(HEAD_IMAGE_KEY, image_type),
            pa.field(EGO_IMAGE_KEY, image_type),
            pa.field("state", vector_type),
            pa.field("actions", vector_type),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ]
    ).with_metadata({b"huggingface": _huggingface_schema_metadata()})

    row_iterator = iter(rows)
    try:
        first_row = next(row_iterator)
    except StopIteration as exc:
        raise ConversionError("cannot write empty parquet row set") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet_writer = pq.ParquetWriter(path, schema, compression="snappy")
    batch = [first_row]
    try:
        for row in row_iterator:
            batch.append(row)
            if len(batch) < batch_size:
                continue
            parquet_writer.write_table(
                pa.Table.from_pylist(batch, schema=schema),
                row_group_size=len(batch),
            )
            batch.clear()
        if batch:
            parquet_writer.write_table(
                pa.Table.from_pylist(batch, schema=schema),
                row_group_size=len(batch),
            )
    finally:
        parquet_writer.close()


def _huggingface_schema_metadata() -> bytes:
    features = {
        HEAD_IMAGE_KEY: {"_type": "Image"},
        EGO_IMAGE_KEY: {"_type": "Image"},
        "state": {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": DOF,
            "_type": "Sequence",
        },
        "actions": {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": DOF,
            "_type": "Sequence",
        },
        "timestamp": {"dtype": "float32", "_type": "Value"},
        "frame_index": {"dtype": "int64", "_type": "Value"},
        "episode_index": {"dtype": "int64", "_type": "Value"},
        "index": {"dtype": "int64", "_type": "Value"},
        "task_index": {"dtype": "int64", "_type": "Value"},
    }
    return json.dumps({"info": {"features": features}}, sort_keys=True).encode()


class _ImageStatsAccumulator:
    def __init__(self) -> None:
        self._frame_count = 0
        self._pixel_count = 0
        self._min: Any = None
        self._max: Any = None
        self._sum: Any = None
        self._sum_sq: Any = None

    def update(self, rgb_frame: Any) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "numpy is required to compute pi0.5 image stats; run "
                "scripts/setup_data_collection_env.sh first"
            ) from exc

        pixels = rgb_frame.astype(np.float64).reshape(-1, 3) / 255.0
        frame_min = pixels.min(axis=0)
        frame_max = pixels.max(axis=0)
        frame_sum = pixels.sum(axis=0)
        frame_sum_sq = np.square(pixels).sum(axis=0)
        if self._min is None:
            self._min = frame_min
            self._max = frame_max
            self._sum = frame_sum
            self._sum_sq = frame_sum_sq
        else:
            self._min = np.minimum(self._min, frame_min)
            self._max = np.maximum(self._max, frame_max)
            self._sum += frame_sum
            self._sum_sq += frame_sum_sq
        self._frame_count += 1
        self._pixel_count += pixels.shape[0]

    def to_json(self) -> dict[str, Any]:
        if self._frame_count <= 0 or self._pixel_count <= 0:
            raise ConversionError("cannot compute image stats for an empty episode")
        mean = self._sum / self._pixel_count
        variance = (self._sum_sq / self._pixel_count) - (mean * mean)
        variance = variance.clip(min=0.0)
        std = variance**0.5
        return {
            "min": _channel_stats_list(self._min),
            "max": _channel_stats_list(self._max),
            "mean": _channel_stats_list(mean),
            "std": _channel_stats_list(std),
            "count": [self._frame_count],
        }


class _NumericStatsAccumulator:
    def __init__(self) -> None:
        self._count = 0
        self._shape: tuple[int, ...] | None = None
        self._min: Any = None
        self._max: Any = None
        self._sum: Any = None
        self._sum_sq: Any = None

    def update(self, value: Any) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "numpy is required to compute pi0.5 stats; run "
                "scripts/setup_data_collection_env.sh first"
            ) from exc

        array = np.asarray(value, dtype=np.float64)
        shape = tuple(int(size) for size in array.shape)
        if self._shape is None:
            self._shape = shape
            self._min = array.copy()
            self._max = array.copy()
            self._sum = array.copy()
            self._sum_sq = np.square(array)
        else:
            if shape != self._shape:
                raise ConversionError(
                    f"numeric stat shape changed: {self._shape} -> {shape}"
                )
            self._min = np.minimum(self._min, array)
            self._max = np.maximum(self._max, array)
            self._sum += array
            self._sum_sq += np.square(array)
        self._count += 1

    def to_json(self) -> dict[str, Any]:
        if self._count <= 0:
            raise ConversionError("cannot compute stats for empty values")
        mean = self._sum / self._count
        variance = (self._sum_sq / self._count) - (mean * mean)
        variance = variance.clip(min=0.0)
        return {
            "min": _numeric_stat_value(self._min),
            "max": _numeric_stat_value(self._max),
            "mean": _numeric_stat_value(mean),
            "std": _numeric_stat_value(variance**0.5),
            "count": [self._count],
        }


def _numeric_stat_value(value: Any) -> list[Any]:
    converted = value.tolist()
    return converted if isinstance(converted, list) else [converted]


def _channel_stats_list(values: Any) -> list[list[list[float]]]:
    return [[[float(value)]] for value in values.tolist()]


def _normalize_rgb(rgb_frame: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required to convert pi0.5 image frames; run "
            "scripts/setup_data_collection_env.sh first"
        ) from exc

    array = np.asarray(rgb_frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ConversionError(f"expected RGB frame with shape HxWx3, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _validate_image_shape(
    image_key: str, rgb_frame: Any, expected_shape: tuple[int, int, int]
) -> None:
    actual_shape = tuple(int(value) for value in rgb_frame.shape)
    if actual_shape != expected_shape:
        raise ConversionError(
            f"RGB frame shape changed for {image_key}: {expected_shape} -> {actual_shape}"
        )


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


def _validate_column_key(value: str, *, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ConversionError(f"{label} must be a non-empty parquet column name")
    return normalized


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


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _json_content(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _jsonl_content(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
