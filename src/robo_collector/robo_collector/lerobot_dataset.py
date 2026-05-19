"""Minimal LeRobot v2.1-style writer for Robo Collector episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from .field_config import FieldSelection, default_field_selection


DOF = 29
CAMERA_KEY = "observation.images.ego_view"
ALIGNED_TARGET_POS_DIM = 45
STATE_FIELD_SHAPES = {
    "joint_position": [DOF],
    "joint_velocity": [DOF],
    "joint_torque": [DOF],
    "imu_angular_velocity": [3],
    "imu_linear_acceleration": [3],
    "projected_gravity_or_quat": [4],
    "relative_ori_6d": [90],
    "motion_anchor_lin_vel_b": [45],
    "motion_anchor_ang_vel_b": [45],
    "ang_vel_history": [30],
    "gravity_history": [30],
    "joint_pos_rel_history": [290],
    "joint_vel_history": [290],
    "action_history": [290],
}
ACTION_FIELD_SHAPES = {
    "joint_position": [DOF],
    "aligned_target_pos": [ALIGNED_TARGET_POS_DIM],
    "policy_action": [DOF],
}


class VideoSink(Protocol):
    def write(self, rgb_frame: Any) -> None: ...

    def close(self) -> None: ...

    def discard(self) -> None: ...


ParquetWriter = Callable[[Path, list[dict[str, Any]]], None]
VideoSinkFactory = Callable[[Path, int, tuple[int, int]], VideoSink]


@dataclass(frozen=True)
class RobotFrame:
    joint_position: list[float]
    joint_velocity: list[float]
    joint_torque: list[float]
    imu_angular_velocity: list[float]
    imu_linear_acceleration: list[float]
    projected_gravity_or_quat: list[float]
    target_joint_pos: list[float]
    policy_action: list[float]
    aligned_target_pos: list[float] = field(default_factory=list)
    policy_state: dict[str, list[float]] = field(default_factory=dict)
    joint_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SaveResult:
    saved: bool
    episode_index: int
    frame_count: int
    data_path: Path | None
    video_path: Path | None
    message: str
    video_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class _ActiveEpisode:
    episode_index: int
    episode_id: str
    task_prompt: str
    task_index: int
    task_is_new: bool
    global_start_index: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    image_shapes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    joint_names: list[str] = field(default_factory=list)
    video_sinks: dict[str, VideoSink] = field(default_factory=dict)
    video_rel_paths: dict[str, Path] = field(default_factory=dict)
    data_rel_path: Path | None = None
    videos_closed: bool = False
    failed_reason: str | None = None


class LeRobotV21Writer:
    """Writes one parquet and one RGB MP4 per saved episode.

    The writer intentionally does not create the dataset directory until an
    episode receives frames, so an idle collector can run without side effects.
    """

    def __init__(
        self,
        root_output_dir: str | Path,
        *,
        dataset_name: str | None = None,
        fps: int = 50,
        camera_key: str = CAMERA_KEY,
        camera_keys: list[str] | tuple[str, ...] | None = None,
        robot_type: str = "unitree_g1",
        field_selection: FieldSelection | None = None,
        parquet_writer: ParquetWriter | None = None,
        video_sink_factory: VideoSinkFactory | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.root_output_dir = Path(root_output_dir)
        self.dataset_name = dataset_name or _default_dataset_name()
        self.root = self.root_output_dir / self.dataset_name
        self.fps = int(fps)
        configured_camera_keys = (
            camera_keys if camera_keys is not None else (camera_key,)
        )
        self.camera_keys = _normalize_camera_keys(configured_camera_keys)
        self.camera_key = self.camera_keys[0]
        self.camera_streams = [
            _camera_stream_from_key(camera_key) for camera_key in self.camera_keys
        ]
        self.camera_stream = self.camera_streams[0]
        self.robot_type = robot_type
        self._field_selection = field_selection or default_field_selection()
        self._parquet_writer = parquet_writer or write_parquet_pyarrow
        self._video_sink_factory = video_sink_factory or OpenCvVideoSink
        self._active: _ActiveEpisode | None = None
        self._tasks_by_text: dict[str, int] = {}
        self._episodes: list[dict[str, Any]] = []
        self._total_frames = 0
        self._image_shapes: dict[str, tuple[int, int, int]] = {}
        self._load_existing_metadata()

    @property
    def active_frame_count(self) -> int:
        return len(self._active.rows) if self._active is not None else 0

    @property
    def active_episode_index(self) -> int | None:
        return self._active.episode_index if self._active is not None else None

    @property
    def active_failed_reason(self) -> str:
        if self._active is None or self._active.failed_reason is None:
            return ""
        return self._active.failed_reason

    def start_episode(self, task_prompt: str, episode_id: str = "") -> int:
        if self._active is not None:
            raise RuntimeError("cannot start a new episode while another is active")
        normalized_prompt = task_prompt.strip()
        if not normalized_prompt:
            raise ValueError("task_prompt is required")

        episode_index = self._next_episode_index()
        task_is_new = normalized_prompt not in self._tasks_by_text
        task_index = self._get_or_allocate_task_index(normalized_prompt)
        self._active = _ActiveEpisode(
            episode_index=episode_index,
            episode_id=episode_id.strip(),
            task_prompt=normalized_prompt,
            task_index=task_index,
            task_is_new=task_is_new,
            global_start_index=self._total_frames,
        )
        return episode_index

    def add_frame(self, frame: RobotFrame, rgb_frame: Any) -> None:
        if self._active is None:
            raise RuntimeError("cannot add a frame before start_episode")
        active = self._active
        if active.failed_reason is not None:
            raise RuntimeError(
                "cannot add frame to failed episode; discard required: "
                f"{active.failed_reason}"
            )
        _validate_robot_frame(frame)

        rgb_frames = self._normalize_frame_bundle(rgb_frame)
        shapes = {
            camera_key: _rgb_shape(image) for camera_key, image in rgb_frames.items()
        }
        for camera_key, shape in shapes.items():
            if shape[2] != 3:
                raise ValueError(
                    f"expected RGB frame with 3 channels for {camera_key}, got {shape[2]}"
                )

        selected_robot_values = self._selected_robot_values(frame)

        if not active.image_shapes:
            active.image_shapes = dict(shapes)
            self._image_shapes.update(active.image_shapes)
            active.data_rel_path = Path(f"data/train-{active.episode_index:06d}.parquet")
            for camera_key in self.camera_keys:
                height, width, _channels = shapes[camera_key]
                video_rel_path = Path(
                    f"videos/{camera_key}/episode_{active.episode_index:06d}.mp4"
                )
                video_path = self.root / video_rel_path
                video_path.parent.mkdir(parents=True, exist_ok=True)
                active.video_rel_paths[camera_key] = video_rel_path
                active.video_sinks[camera_key] = self._video_sink_factory(
                    video_path, self.fps, (width, height)
                )
        else:
            for camera_key, shape in shapes.items():
                if active.image_shapes[camera_key] != shape:
                    raise ValueError(
                        "RGB frame shape changed within episode for "
                        f"{camera_key}: {active.image_shapes[camera_key]} -> {shape}"
                    )

        if not active.joint_names and frame.joint_names:
            active.joint_names = list(frame.joint_names)

        try:
            for camera_key in self.camera_keys:
                active.video_sinks[camera_key].write(rgb_frames[camera_key])
        except Exception as exc:
            active.failed_reason = f"video write failed for {camera_key}: {exc}"
            raise RuntimeError(active.failed_reason) from exc

        frame_index = len(active.rows)
        timestamp = frame_index / self.fps
        row = dict(selected_robot_values)
        row.update(
            {
                "annotation.human.action.task_description": active.task_prompt,
                "timestamp": timestamp,
                "frame_index": frame_index,
                "episode_index": active.episode_index,
                "index": active.global_start_index + frame_index,
                "task_index": active.task_index,
            }
        )
        for camera_key in self.camera_keys:
            row[camera_key] = {
                "path": str(active.video_rel_paths[camera_key]),
                "timestamp": timestamp,
            }
        active.rows.append(row)

    def _selected_robot_values(self, frame: RobotFrame) -> dict[str, list[float]]:
        robot_values = {
            "observation.state.joint_position": frame.joint_position,
            "observation.state.joint_velocity": frame.joint_velocity,
            "observation.state.joint_torque": frame.joint_torque,
            "observation.state.imu_angular_velocity": frame.imu_angular_velocity,
            "observation.state.imu_linear_acceleration": (
                frame.imu_linear_acceleration
            ),
            "observation.state.projected_gravity_or_quat": (
                frame.projected_gravity_or_quat
            ),
            "action.joint_position": frame.target_joint_pos,
            "action.aligned_target_pos": frame.aligned_target_pos,
            "action.policy_action": frame.policy_action,
        }
        for field_name, values in frame.policy_state.items():
            robot_values[f"observation.state.{field_name}"] = values

        selected_values = {}
        for key in self._field_selection.robot_parquet_keys:
            if key not in robot_values:
                raise ValueError(f"selected field {key} is missing from RobotFrame")
            value = robot_values[key]
            _validate_selected_robot_value(key, value)
            selected_values[key] = value
        return selected_values

    def _normalize_frame_bundle(self, rgb_frame: Any) -> dict[str, Any]:
        if len(self.camera_keys) == 1 and not isinstance(rgb_frame, dict):
            return {self.camera_key: rgb_frame}
        if not isinstance(rgb_frame, dict):
            raise ValueError(
                "multi-camera writer expects a dict of camera stream/key to RGB frame"
            )

        frames: dict[str, Any] = {}
        missing = []
        for camera_key, camera_stream in zip(self.camera_keys, self.camera_streams):
            if camera_key in rgb_frame:
                frames[camera_key] = rgb_frame[camera_key]
            elif camera_stream in rgb_frame:
                frames[camera_key] = rgb_frame[camera_stream]
            else:
                missing.append(camera_stream)
        if missing:
            raise ValueError("missing RGB frame(s): " + ",".join(missing))
        return frames

    def save_episode(self) -> SaveResult:
        active = self._require_active()
        if active.failed_reason is not None:
            raise RuntimeError(
                "cannot save failed episode; discard required: "
                f"{active.failed_reason}"
            )
        if not active.rows:
            self._discard_active()
            return SaveResult(
                saved=False,
                episode_index=active.episode_index,
                frame_count=0,
                data_path=None,
                video_path=None,
                message="discarded empty episode",
            )

        if not active.videos_closed:
            for video_sink in active.video_sinks.values():
                video_sink.close()
            active.videos_closed = True

        assert active.data_rel_path is not None
        assert active.video_rel_paths
        data_path = self.root / active.data_rel_path
        data_path.parent.mkdir(parents=True, exist_ok=True)
        self._parquet_writer(data_path, active.rows)

        episode_record = self._episode_record(active)
        pending_episodes = [*self._episodes, episode_record]
        pending_total_frames = self._total_frames + len(active.rows)
        self._write_metadata(
            active,
            episodes=pending_episodes,
            total_frames=pending_total_frames,
            tasks_by_text=self._tasks_by_text,
        )

        self._episodes = pending_episodes
        self._total_frames = pending_total_frames
        result = SaveResult(
            saved=True,
            episode_index=active.episode_index,
            frame_count=len(active.rows),
            data_path=data_path,
            video_path=self.root / active.video_rel_paths[self.camera_key],
            message="episode saved",
            video_paths={
                camera_key: self.root / video_rel_path
                for camera_key, video_rel_path in active.video_rel_paths.items()
            },
        )
        self._active = None
        return result

    def _episode_record(self, active: _ActiveEpisode) -> dict[str, Any]:
        return {
            "episode_index": active.episode_index,
            "episode_id": active.episode_id,
            "task_index": active.task_index,
            "tasks": [active.task_prompt],
            "length": len(active.rows),
            "fps": self.fps,
            "data_path": str(active.data_rel_path),
            "video_path": str(active.video_rel_paths.get(self.camera_key, "")),
            "video_paths": {
                camera_key: str(video_rel_path)
                for camera_key, video_rel_path in active.video_rel_paths.items()
            },
            "dataset_from_index": active.global_start_index,
            "dataset_to_index": active.global_start_index + len(active.rows),
        }

    def discard_episode(self) -> None:
        self._require_active()
        self._discard_active()

    def _discard_active(self) -> None:
        active = self._active
        if active is None:
            return
        for video_sink in active.video_sinks.values():
            video_sink.discard()
        if active.data_rel_path is not None:
            (self.root / active.data_rel_path).unlink(missing_ok=True)
        if (
            active.task_is_new
            and self._tasks_by_text.get(active.task_prompt) == active.task_index
        ):
            del self._tasks_by_text[active.task_prompt]
        self._active = None

    def _require_active(self) -> _ActiveEpisode:
        if self._active is None:
            raise RuntimeError("no active episode")
        return self._active

    def _next_episode_index(self) -> int:
        if not self._episodes:
            return 0
        return max(int(episode["episode_index"]) for episode in self._episodes) + 1

    def _get_or_allocate_task_index(self, task_prompt: str) -> int:
        if task_prompt in self._tasks_by_text:
            return self._tasks_by_text[task_prompt]
        task_index = len(self._tasks_by_text)
        self._tasks_by_text[task_prompt] = task_index
        return task_index

    def _load_existing_metadata(self) -> None:
        tasks_path = self.root / "meta/tasks.jsonl"
        if tasks_path.exists():
            for line in tasks_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self._tasks_by_text[str(row["task"])] = int(row["task_index"])

        episodes_path = self.root / "meta/episodes.jsonl"
        if episodes_path.exists():
            for line in episodes_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                episode = json.loads(line)
                self._episodes.append(episode)
                self._total_frames = max(
                    self._total_frames, int(episode.get("dataset_to_index", 0))
                )

        info_path = self.root / "meta/info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            features = info.get("features", {})
            if not isinstance(features, dict):
                raise ValueError(
                    "existing dataset meta/info.json features must be a mapping"
                )
            self._validate_existing_robot_features(features)
            for camera_key in self.camera_keys:
                image_feature = features.get(camera_key, {})
                shape = image_feature.get("shape")
                if isinstance(shape, list) and len(shape) == 3:
                    self._image_shapes[camera_key] = (
                        int(shape[0]),
                        int(shape[1]),
                        int(shape[2]),
                    )

    def _validate_existing_robot_features(self, features: dict[str, Any]) -> None:
        existing_robot_keys = _robot_feature_keys(features)
        selected_robot_keys = set(self._field_selection.robot_parquet_keys)
        if existing_robot_keys == selected_robot_keys:
            return

        missing = sorted(selected_robot_keys - existing_robot_keys)
        extra = sorted(existing_robot_keys - selected_robot_keys)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError(
            "field selection does not match existing dataset robot features; "
            + "; ".join(details)
            + "; use a matching field_config_path or a new dataset_name/root_output_dir"
        )

    def _write_metadata(
        self,
        active: _ActiveEpisode,
        *,
        episodes: list[dict[str, Any]],
        total_frames: int,
        tasks_by_text: dict[str, int],
    ) -> None:
        meta_dir = self.root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        metadata_files = {
            meta_dir / "tasks.jsonl": _jsonl_content(
                [
                    {"task_index": task_index, "task": task}
                    for task, task_index in sorted(
                        tasks_by_text.items(), key=lambda item: item[1]
                    )
                ]
            ),
            meta_dir / "episodes.jsonl": _jsonl_content(episodes),
            meta_dir / "info.json": _json_content(
                self._info(
                    active,
                    episodes=episodes,
                    total_frames=total_frames,
                    tasks_by_text=tasks_by_text,
                )
            ),
            meta_dir / "modality.json": _json_content(self._modality(active)),
        }
        _write_files_transactional(metadata_files)

    def _info(
        self,
        active: _ActiveEpisode,
        *,
        episodes: list[dict[str, Any]],
        total_frames: int,
        tasks_by_text: dict[str, int],
    ) -> dict[str, Any]:
        episode_count = len(episodes)
        return {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "total_episodes": episode_count,
            "total_frames": total_frames,
            "total_tasks": len(tasks_by_text),
            "total_videos": episode_count * len(self.camera_keys),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{episode_count}"},
            "data_path": "data/train-{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
            "features": self._features(active.image_shapes, active.joint_names),
        }

    def _features(
        self,
        active_image_shapes: dict[str, tuple[int, int, int]],
        joint_names: list[str],
    ) -> dict[str, Any]:
        joint_feature = {
            "dtype": "float32",
            "shape": [DOF],
            "names": joint_names if len(joint_names) == DOF else None,
        }
        video_features = {}
        for camera_key in self.camera_keys:
            height, width, channels = (
                active_image_shapes.get(camera_key)
                or self._image_shapes.get(camera_key)
                or (0, 0, 3)
            )
            video_features[camera_key] = {
                "dtype": "video",
                "shape": [height, width, channels],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self.fps,
                    "video.channels": channels,
                    "has_audio": False,
                },
            }

        robot_feature_by_key = {
            "observation.state.joint_position": joint_feature,
            "observation.state.joint_velocity": joint_feature,
            "observation.state.joint_torque": joint_feature,
            "observation.state.imu_angular_velocity": {
                "dtype": "float32",
                "shape": [3],
                "names": ["x", "y", "z"],
            },
            "observation.state.imu_linear_acceleration": {
                "dtype": "float32",
                "shape": [3],
                "names": ["x", "y", "z"],
            },
            "observation.state.projected_gravity_or_quat": {
                "dtype": "float32",
                "shape": [4],
                "names": ["x", "y", "z", "w"],
            },
            "action.joint_position": joint_feature,
            "action.aligned_target_pos": {
                "dtype": "float32",
                "shape": [ALIGNED_TARGET_POS_DIM],
                "names": None,
            },
            "action.policy_action": joint_feature,
        }
        for field_name, shape in STATE_FIELD_SHAPES.items():
            robot_feature_by_key.setdefault(
                f"observation.state.{field_name}",
                {"dtype": "float32", "shape": shape, "names": None},
            )
        selected_robot_features = {
            key: robot_feature_by_key[key]
            for key in self._field_selection.robot_parquet_keys
        }

        return {
            **video_features,
            **selected_robot_features,
            "annotation.human.action.task_description": {
                "dtype": "string",
                "shape": [1],
                "names": None,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }

    def _modality(self, active: _ActiveEpisode) -> dict[str, Any]:
        image_modalities = {}
        for camera_key, camera_stream in zip(self.camera_keys, self.camera_streams):
            shape = (
                active.image_shapes.get(camera_key)
                or self._image_shapes.get(camera_key)
                or (0, 0, 3)
            )
            image_modalities[camera_stream] = {
                "key": camera_key,
                "dtype": "rgb",
                "shape": list(shape),
                "fps": self.fps,
            }
        return {
            "observation": {
                "images": image_modalities,
                "state": {
                    field: {"shape": STATE_FIELD_SHAPES[field]}
                    for field in self._field_selection.state
                },
            },
            "action": {
                field: {"shape": ACTION_FIELD_SHAPES[field]}
                for field in self._field_selection.action_fields
            },
            "annotation": {
                "human": {
                    "action": {
                        "task_description": {
                            "key": "annotation.human.action.task_description",
                            "dtype": "string",
                        }
                    }
                }
            },
        }


class OpenCvVideoSink:
    def __init__(self, path: Path, fps: int, frame_size: tuple[int, int]) -> None:
        import cv2

        self.path = path
        self._cv2 = cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, float(fps), frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"failed to open video writer for {path}")

    def write(self, rgb_frame: Any) -> None:
        import numpy as np

        bgr = self._cv2.cvtColor(np.asarray(rgb_frame), self._cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)

    def close(self) -> None:
        self._writer.release()

    def discard(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)


def write_parquet_pyarrow(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to write LeRobot parquet files; run "
            "scripts/setup_data_collection_env.sh first"
        ) from exc

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="snappy")


def _validate_robot_frame(frame: RobotFrame) -> None:
    _validate_len("joint_position", frame.joint_position, DOF)
    _validate_len("joint_velocity", frame.joint_velocity, DOF)
    _validate_len("joint_torque", frame.joint_torque, DOF)
    _validate_len("imu_angular_velocity", frame.imu_angular_velocity, 3)
    _validate_len("imu_linear_acceleration", frame.imu_linear_acceleration, 3)
    _validate_len("projected_gravity_or_quat", frame.projected_gravity_or_quat, 4)
    _validate_len("target_joint_pos", frame.target_joint_pos, DOF)
    _validate_len("policy_action", frame.policy_action, DOF)
    if frame.aligned_target_pos:
        _validate_len(
            "aligned_target_pos", frame.aligned_target_pos, ALIGNED_TARGET_POS_DIM
        )
    for field_name, values in frame.policy_state.items():
        shape = STATE_FIELD_SHAPES.get(field_name)
        if shape is not None:
            _validate_len(field_name, values, int(shape[0]))


def _validate_selected_robot_value(key: str, values: list[float]) -> None:
    if key == "action.aligned_target_pos":
        _validate_len(key, values, ALIGNED_TARGET_POS_DIM)
        return
    if key in {"action.joint_position", "action.policy_action"}:
        _validate_len(key, values, DOF)
        return
    state_prefix = "observation.state."
    if key.startswith(state_prefix):
        field_name = key[len(state_prefix) :]
        shape = STATE_FIELD_SHAPES.get(field_name)
        if shape is not None:
            _validate_len(key, values, int(shape[0]))


def _validate_len(name: str, values: list[float], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} has dimension {len(values)}; expected {expected}")


def _rgb_shape(rgb_frame: Any) -> tuple[int, int, int]:
    shape = getattr(rgb_frame, "shape", None)
    if shape is not None and len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    height = len(rgb_frame)
    width = len(rgb_frame[0]) if height else 0
    channels = len(rgb_frame[0][0]) if height and width else 0
    return height, width, channels


def _json_content(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _jsonl_content(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _default_dataset_name() -> str:
    return "robo_collector_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _camera_stream_from_key(camera_key: str) -> str:
    prefix = "observation.images."
    if camera_key.startswith(prefix):
        return camera_key[len(prefix) :]
    return camera_key.rsplit(".", 1)[-1]


def _normalize_camera_keys(camera_keys: list[str] | tuple[str, ...]) -> list[str]:
    normalized = [str(camera_key).strip() for camera_key in camera_keys]
    normalized = [camera_key for camera_key in normalized if camera_key]
    if not normalized:
        raise ValueError("at least one camera key is required")
    duplicates = sorted(
        {camera_key for camera_key in normalized if normalized.count(camera_key) > 1}
    )
    if duplicates:
        raise ValueError(f"duplicate camera key(s): {','.join(duplicates)}")
    return normalized


def _robot_feature_keys(features: dict[str, Any]) -> set[str]:
    return {
        key
        for key in features
        if key.startswith("action.") or key.startswith("observation.state.")
    }


def _write_files_transactional(files: dict[Path, str]) -> None:
    token = uuid4().hex
    staging_paths: dict[Path, Path] = {}
    backup_paths: dict[Path, Path | None] = {}

    try:
        for path, content in files.items():
            staging_path = path.with_name(f".{path.name}.{token}.tmp")
            staging_path.write_text(content, encoding="utf-8")
            staging_paths[path] = staging_path

        for path, staging_path in staging_paths.items():
            backup_path = path.with_name(f".{path.name}.{token}.bak")
            if path.exists():
                _replace_path(path, backup_path)
                backup_paths[path] = backup_path
            else:
                backup_paths[path] = None
            _replace_path(staging_path, path)

        for backup_path in backup_paths.values():
            if backup_path is not None:
                backup_path.unlink(missing_ok=True)
    except Exception:
        for path, backup_path in reversed(list(backup_paths.items())):
            path.unlink(missing_ok=True)
            if backup_path is not None and backup_path.exists():
                _replace_path(backup_path, path)
        for staging_path in staging_paths.values():
            staging_path.unlink(missing_ok=True)
        for backup_path in backup_paths.values():
            if backup_path is not None:
                backup_path.unlink(missing_ok=True)
        raise


def _replace_path(source: Path, target: Path) -> None:
    source.replace(target)
