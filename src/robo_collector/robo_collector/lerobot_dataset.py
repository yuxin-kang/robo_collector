"""Minimal LeRobot v2.1-style writer for Robo Collector episodes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, TextIO
from uuid import uuid4

from .field_config import FieldSelection, default_field_selection


DOF = 29
CAMERA_KEY = "observation.images.ego_view"
ALIGNED_TARGET_POS_DIM = 45
PARQUET_ROW_GROUP_SIZE = 256
IN_PROGRESS_DIR = ".inprogress"
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
    state_timestamp_sec: float | None = None


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
    frame_count: int = 0
    image_shapes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    joint_names: list[str] = field(default_factory=list)
    video_sinks: dict[str, VideoSink] = field(default_factory=dict)
    video_rel_paths: dict[str, Path] = field(default_factory=dict)
    staged_video_rel_paths: dict[str, Path] = field(default_factory=dict)
    data_rel_path: Path | None = None
    staged_data_rel_path: Path | None = None
    row_spool_rel_path: Path | None = None
    row_spool_handle: TextIO | None = None
    manifest_rel_path: Path | None = None
    storage_token: str = field(default_factory=lambda: uuid4().hex)
    timestamp_origin_sec: float | None = None
    last_timestamp_sec: float | None = None
    last_camera_timestamps_sec: dict[str, float] = field(default_factory=dict)
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
        if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
            raise ValueError(f"fps must be a positive integer, got {fps!r}")
        self.root_output_dir = Path(root_output_dir).resolve()
        self.dataset_name = dataset_name or _default_dataset_name()
        self.root = _validated_dataset_root(
            self.root_output_dir, self.dataset_name
        )
        self.fps = fps
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
        self._uses_default_parquet_writer = parquet_writer is None
        self._parquet_writer = parquet_writer or write_parquet_pyarrow
        self._video_sink_factory = video_sink_factory or OpenCvVideoSink
        self._active: _ActiveEpisode | None = None
        self._tasks_by_text: dict[str, int] = {}
        self._episodes: list[dict[str, Any]] = []
        self._total_frames = 0
        self._image_shapes: dict[str, tuple[int, int, int]] = {}
        self._joint_names: list[str] = []
        self._load_existing_metadata()
        self._recover_orphaned_active_episodes()

    @property
    def active_frame_count(self) -> int:
        return self._active.frame_count if self._active is not None else 0

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

    def add_frame(
        self,
        frame: RobotFrame,
        rgb_frame: Any,
        *,
        camera_timestamps_sec: dict[str, float | None] | None = None,
    ) -> None:
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
        self._validate_joint_names(active, frame.joint_names)
        frame_index = active.frame_count
        timestamp, video_timestamps = self._episode_timestamps(
            active,
            frame_index=frame_index,
            state_timestamp_sec=frame.state_timestamp_sec,
            camera_timestamps_sec=camera_timestamps_sec,
        )

        if not active.image_shapes:
            self._validate_image_shapes_against_existing(shapes)
            active.image_shapes = dict(shapes)
            active.data_rel_path = Path(f"data/train-{active.episode_index:06d}.parquet")
            active.staged_data_rel_path = Path(
                IN_PROGRESS_DIR,
                active.storage_token,
                str(active.data_rel_path),
            )
            for camera_key in self.camera_keys:
                video_rel_path = Path(
                    f"videos/{camera_key}/episode_{active.episode_index:06d}.mp4"
                )
                staged_video_rel_path = Path(
                    IN_PROGRESS_DIR,
                    active.storage_token,
                    str(video_rel_path),
                )
                active.video_rel_paths[camera_key] = video_rel_path
                active.staged_video_rel_paths[camera_key] = staged_video_rel_path
            self._prepare_active_storage(active)
            for camera_key in self.camera_keys:
                height, width, _channels = shapes[camera_key]
                video_path = self._root_path(
                    active.staged_video_rel_paths[camera_key]
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)
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

        try:
            for camera_key in self.camera_keys:
                active.video_sinks[camera_key].write(rgb_frames[camera_key])
        except Exception as exc:
            active.failed_reason = f"video write failed for {camera_key}: {exc}"
            raise RuntimeError(active.failed_reason) from exc

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
                "timestamp": video_timestamps[camera_key],
            }
        self._append_spooled_row(active, row)
        active.frame_count += 1

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
            selected_values[key] = _validate_selected_robot_value(key, value)
        return selected_values

    def _validate_joint_names(
        self, active: _ActiveEpisode, joint_names: list[str]
    ) -> None:
        names = [str(name).strip() for name in joint_names]
        if names:
            if len(names) != DOF:
                raise ValueError(
                    f"joint_names has dimension {len(names)}; expected {DOF}"
                )
            if any(not name for name in names):
                raise ValueError("joint_names must not contain empty names")
            if len(set(names)) != DOF:
                raise ValueError("joint_names must be unique")

        expected = active.joint_names or self._joint_names
        if expected:
            if names != expected:
                raise ValueError(
                    "joint_names changed from the dataset canonical ordering"
                )
            active.joint_names = list(expected)
            return

        if active.frame_count > 0 and names:
            raise ValueError(
                "joint_names appeared after unlabeled frames; refusing to relabel "
                "earlier rows"
            )
        if self._episodes and names:
            raise ValueError(
                "existing dataset has no canonical joint_names; use a new dataset "
                "instead of relabeling existing rows"
            )
        if names:
            active.joint_names = names

    def _validate_image_shapes_against_existing(
        self, shapes: dict[str, tuple[int, int, int]]
    ) -> None:
        if not self._image_shapes:
            return
        for camera_key in self.camera_keys:
            expected = self._image_shapes[camera_key]
            actual = shapes[camera_key]
            if actual != expected:
                raise ValueError(
                    "camera shape does not match existing dataset for "
                    f"{camera_key}: expected {expected}, got {actual}; "
                    "use a new dataset_name/root_output_dir"
                )

    def _episode_timestamps(
        self,
        active: _ActiveEpisode,
        *,
        frame_index: int,
        state_timestamp_sec: float | None,
        camera_timestamps_sec: dict[str, float | None] | None,
    ) -> tuple[float, dict[str, float]]:
        expected_timestamp = frame_index / self.fps
        state_source_timestamp = _optional_finite_timestamp(
            state_timestamp_sec, label="state_timestamp_sec"
        )
        camera_source_timestamps = self._normalize_camera_timestamps(
            camera_timestamps_sec
        )

        if active.timestamp_origin_sec is None:
            source_timestamps = [
                timestamp
                for timestamp in [
                    state_source_timestamp,
                    *camera_source_timestamps.values(),
                ]
                if timestamp is not None
            ]
            if source_timestamps:
                active.timestamp_origin_sec = (
                    min(source_timestamps) - expected_timestamp
                )

        if (
            state_source_timestamp is not None
            and active.timestamp_origin_sec is not None
        ):
            timestamp = state_source_timestamp - active.timestamp_origin_sec
        else:
            timestamp = expected_timestamp
        _validate_monotonic_timestamp(
            timestamp,
            active.last_timestamp_sec,
            label="state timestamp",
        )

        video_timestamps: dict[str, float] = {}
        for camera_key in self.camera_keys:
            source_timestamp = camera_source_timestamps[camera_key]
            if (
                source_timestamp is not None
                and active.timestamp_origin_sec is not None
            ):
                video_timestamp = source_timestamp - active.timestamp_origin_sec
            else:
                video_timestamp = timestamp
            _validate_monotonic_timestamp(
                video_timestamp,
                active.last_camera_timestamps_sec.get(camera_key),
                label=f"{camera_key} timestamp",
            )
            video_timestamps[camera_key] = video_timestamp

        active.last_timestamp_sec = timestamp
        active.last_camera_timestamps_sec.update(video_timestamps)
        return timestamp, video_timestamps

    def _normalize_camera_timestamps(
        self, camera_timestamps_sec: dict[str, float | None] | None
    ) -> dict[str, float | None]:
        if camera_timestamps_sec is None:
            return {camera_key: None for camera_key in self.camera_keys}
        if not isinstance(camera_timestamps_sec, dict):
            raise ValueError("camera_timestamps_sec must be a mapping")

        normalized: dict[str, float | None] = {}
        for camera_key, camera_stream in zip(
            self.camera_keys, self.camera_streams
        ):
            by_key = camera_timestamps_sec.get(camera_key)
            by_stream = camera_timestamps_sec.get(camera_stream)
            if (
                by_key is not None
                and by_stream is not None
                and float(by_key) != float(by_stream)
            ):
                raise ValueError(
                    "conflicting camera timestamp values for "
                    f"{camera_key} and {camera_stream}"
                )
            value = by_key if by_key is not None else by_stream
            normalized[camera_key] = _optional_finite_timestamp(
                value, label=f"{camera_key} timestamp"
            )
        return normalized

    def _prepare_active_storage(self, active: _ActiveEpisode) -> None:
        work_rel_path = Path(IN_PROGRESS_DIR, active.storage_token)
        work_path = self._root_path(work_rel_path)
        work_path.mkdir(parents=True, exist_ok=False)
        active.row_spool_rel_path = work_rel_path / "rows.jsonl"
        active.manifest_rel_path = work_rel_path / "manifest.json"
        manifest = {
            "episode_index": active.episode_index,
            "data_path": str(active.data_rel_path),
            "staged_data_path": str(active.staged_data_rel_path),
            "video_paths": {
                key: str(path) for key, path in active.video_rel_paths.items()
            },
            "staged_video_paths": {
                key: str(path)
                for key, path in active.staged_video_rel_paths.items()
            },
            "row_spool_path": str(active.row_spool_rel_path),
        }
        manifest_path = self._root_path(active.manifest_rel_path)
        manifest_path.write_text(_json_content(manifest), encoding="utf-8")
        active.row_spool_handle = self._root_path(
            active.row_spool_rel_path
        ).open("a", encoding="utf-8", buffering=1)

    def _append_spooled_row(
        self, active: _ActiveEpisode, row: dict[str, Any]
    ) -> None:
        if active.row_spool_handle is None:
            raise RuntimeError("row spool is not open")
        try:
            active.row_spool_handle.write(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            )
        except Exception as exc:
            active.failed_reason = f"row spool write failed: {exc}"
            raise RuntimeError(active.failed_reason) from exc

    def _close_row_spool(self, active: _ActiveEpisode) -> None:
        if active.row_spool_handle is None:
            return
        active.row_spool_handle.close()
        active.row_spool_handle = None

    def _commit_active_artifacts(self, active: _ActiveEpisode) -> None:
        assert active.staged_data_rel_path is not None
        assert active.data_rel_path is not None
        staged_data_path = self._root_path(active.staged_data_rel_path)
        data_path = self._root_path(active.data_rel_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        _replace_path(staged_data_path, data_path)

        for camera_key in self.camera_keys:
            staged_video_path = self._root_path(
                active.staged_video_rel_paths[camera_key]
            )
            video_path = self._root_path(active.video_rel_paths[camera_key])
            if staged_video_path.exists():
                video_path.parent.mkdir(parents=True, exist_ok=True)
                _replace_path(staged_video_path, video_path)
            elif not video_path.exists():
                raise RuntimeError(
                    f"missing staged video for {camera_key}: {staged_video_path}"
                )

    def _cleanup_active_staging(
        self, active: _ActiveEpisode, *, remove_committed: bool
    ) -> None:
        self._close_row_spool(active)
        relative_paths = [
            active.row_spool_rel_path,
            active.staged_data_rel_path,
            active.manifest_rel_path,
            *active.staged_video_rel_paths.values(),
        ]
        if remove_committed:
            relative_paths.extend(
                [active.data_rel_path, *active.video_rel_paths.values()]
            )
        for relative_path in relative_paths:
            if relative_path is not None:
                self._root_path(relative_path).unlink(missing_ok=True)
        self._remove_empty_work_dir(active.storage_token)

    def _remove_empty_work_dir(self, storage_token: str) -> None:
        work_path = self._root_path(Path(IN_PROGRESS_DIR, storage_token))
        paths = [path for path in work_path.rglob("*") if path.is_dir()]
        for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass
        try:
            work_path.rmdir()
        except OSError:
            pass
        try:
            self._root_path(Path(IN_PROGRESS_DIR)).rmdir()
        except OSError:
            pass

    def _root_path(self, relative_path: str | Path) -> Path:
        return _safe_path_below(self.root, relative_path)

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
        if active.frame_count == 0:
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
        self._close_row_spool(active)

        assert active.data_rel_path is not None
        assert active.staged_data_rel_path is not None
        assert active.row_spool_rel_path is not None
        assert active.video_rel_paths
        staged_data_path = self._root_path(active.staged_data_rel_path)
        staged_data_path.parent.mkdir(parents=True, exist_ok=True)
        row_spool_path = self._root_path(active.row_spool_rel_path)
        if self._uses_default_parquet_writer:
            write_parquet_pyarrow_stream(
                staged_data_path,
                row_spool_path,
                batch_size=PARQUET_ROW_GROUP_SIZE,
            )
        else:
            self._parquet_writer(
                staged_data_path, _read_spooled_rows(row_spool_path)
            )
        self._commit_active_artifacts(active)
        data_path = self._root_path(active.data_rel_path)

        episode_record = self._episode_record(active)
        pending_episodes = [*self._episodes, episode_record]
        pending_total_frames = self._total_frames + active.frame_count
        self._write_metadata(
            active,
            episodes=pending_episodes,
            total_frames=pending_total_frames,
            tasks_by_text=self._tasks_by_text,
        )

        self._episodes = pending_episodes
        self._total_frames = pending_total_frames
        if active.joint_names and not self._joint_names:
            self._joint_names = list(active.joint_names)
        self._image_shapes.update(active.image_shapes)
        result = SaveResult(
            saved=True,
            episode_index=active.episode_index,
            frame_count=active.frame_count,
            data_path=data_path,
            video_path=self._root_path(active.video_rel_paths[self.camera_key]),
            message="episode saved",
            video_paths={
                camera_key: self._root_path(video_rel_path)
                for camera_key, video_rel_path in active.video_rel_paths.items()
            },
        )
        self._cleanup_active_staging(active, remove_committed=False)
        self._active = None
        return result

    def _episode_record(self, active: _ActiveEpisode) -> dict[str, Any]:
        return {
            "episode_index": active.episode_index,
            "episode_id": active.episode_id,
            "task_index": active.task_index,
            "tasks": [active.task_prompt],
            "length": active.frame_count,
            "fps": self.fps,
            "data_path": str(active.data_rel_path),
            "video_path": str(active.video_rel_paths.get(self.camera_key, "")),
            "video_paths": {
                camera_key: str(video_rel_path)
                for camera_key, video_rel_path in active.video_rel_paths.items()
            },
            "dataset_from_index": active.global_start_index,
            "dataset_to_index": active.global_start_index + active.frame_count,
        }

    def discard_episode(self) -> None:
        self._require_active()
        self._discard_active()

    def _discard_active(self) -> None:
        active = self._active
        if active is None:
            return
        self._close_row_spool(active)
        for video_sink in active.video_sinks.values():
            try:
                video_sink.discard()
            except Exception:
                pass
        self._cleanup_active_staging(active, remove_committed=True)
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
            if info.get("robot_type") != self.robot_type:
                raise ValueError(
                    "robot_type does not match existing dataset: "
                    f"expected {info.get('robot_type')!r}, got {self.robot_type!r}"
                )
            if info.get("fps") != self.fps:
                raise ValueError(
                    "fps does not match existing dataset: "
                    f"expected {info.get('fps')!r}, got {self.fps!r}"
                )
            features = info.get("features", {})
            if not isinstance(features, dict):
                raise ValueError(
                    "existing dataset meta/info.json features must be a mapping"
                )
            self._validate_existing_robot_features(features)
            existing_camera_keys = {
                key
                for key, feature in features.items()
                if isinstance(feature, dict) and feature.get("dtype") == "video"
            }
            configured_camera_keys = set(self.camera_keys)
            if existing_camera_keys != configured_camera_keys:
                raise ValueError(
                    "camera keys do not match existing dataset; "
                    f"expected {sorted(existing_camera_keys)}, "
                    f"got {sorted(configured_camera_keys)}"
                )
            for camera_key in self.camera_keys:
                image_feature = features.get(camera_key, {})
                shape = image_feature.get("shape")
                if not (
                    isinstance(shape, list)
                    and len(shape) == 3
                    and all(isinstance(value, int) and value > 0 for value in shape)
                ):
                    raise ValueError(
                        f"existing camera feature {camera_key} has invalid shape "
                        f"{shape!r}"
                    )
                video_fps = image_feature.get("info", {}).get("video.fps")
                if video_fps != self.fps:
                    raise ValueError(
                        f"camera fps does not match existing dataset for {camera_key}: "
                        f"expected {video_fps!r}, got {self.fps!r}"
                    )
                self._image_shapes[camera_key] = (
                    int(shape[0]),
                    int(shape[1]),
                    int(shape[2]),
                )

    def _validate_existing_robot_features(self, features: dict[str, Any]) -> None:
        existing_robot_keys = _robot_feature_keys(features)
        selected_robot_keys = set(self._field_selection.robot_parquet_keys)
        if existing_robot_keys != selected_robot_keys:
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
                + "; use a matching field_config_path or a new "
                "dataset_name/root_output_dir"
            )

        canonical_names: list[str] | None = None
        for key in sorted(selected_robot_keys):
            feature = features[key]
            if not isinstance(feature, dict):
                raise ValueError(f"existing robot feature {key} must be a mapping")
            expected_shape = _robot_feature_shape(key)
            if feature.get("dtype") != "float32":
                raise ValueError(
                    f"existing robot feature {key} dtype must be float32"
                )
            if feature.get("shape") != expected_shape:
                raise ValueError(
                    f"existing robot feature {key} shape mismatch: "
                    f"expected {expected_shape}, got {feature.get('shape')!r}"
                )
            if key not in _joint_order_feature_keys():
                continue
            names = feature.get("names")
            if names is None:
                continue
            validated_names = _validated_joint_names(names, label=f"{key} names")
            if canonical_names is None:
                canonical_names = validated_names
            elif validated_names != canonical_names:
                raise ValueError(
                    "existing robot features disagree on canonical joint_names"
                )
        if canonical_names is not None:
            self._joint_names = canonical_names

    def _recover_orphaned_active_episodes(self) -> None:
        in_progress_path = self._root_path(Path(IN_PROGRESS_DIR))
        if not in_progress_path.exists():
            return
        committed_episode_indexes = {
            int(episode["episode_index"]) for episode in self._episodes
        }
        for manifest_path in sorted(in_progress_path.glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                episode_index = int(manifest["episode_index"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue

            relative_paths: list[str] = [
                str(manifest.get("row_spool_path", "")),
                str(manifest.get("staged_data_path", "")),
                *[
                    str(path)
                    for path in manifest.get("staged_video_paths", {}).values()
                ],
            ]
            if episode_index not in committed_episode_indexes:
                relative_paths.extend(
                    [
                        str(manifest.get("data_path", "")),
                        *[
                            str(path)
                            for path in manifest.get("video_paths", {}).values()
                        ],
                    ]
                )
            for relative_path in relative_paths:
                if not relative_path:
                    continue
                try:
                    self._root_path(relative_path).unlink(missing_ok=True)
                except ValueError:
                    continue
            manifest_path.unlink(missing_ok=True)
            self._remove_empty_work_dir(manifest_path.parent.name)

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


def write_parquet_pyarrow_stream(
    path: Path, row_spool_path: Path, *, batch_size: int
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to write LeRobot parquet files; run "
            "scripts/setup_data_collection_env.sh first"
        ) from exc

    parquet_writer = None
    schema = None
    try:
        with row_spool_path.open("r", encoding="utf-8") as spool:
            batch: list[dict[str, Any]] = []
            for line in spool:
                if not line.strip():
                    continue
                batch.append(json.loads(line))
                if len(batch) < batch_size:
                    continue
                table = pa.Table.from_pylist(batch, schema=schema)
                if parquet_writer is None:
                    schema = table.schema
                    parquet_writer = pq.ParquetWriter(
                        path, schema, compression="snappy"
                    )
                parquet_writer.write_table(table, row_group_size=len(batch))
                batch.clear()
            if batch:
                table = pa.Table.from_pylist(batch, schema=schema)
                if parquet_writer is None:
                    schema = table.schema
                    parquet_writer = pq.ParquetWriter(
                        path, schema, compression="snappy"
                    )
                parquet_writer.write_table(table, row_group_size=len(batch))
        if parquet_writer is None:
            raise ValueError("cannot write parquet from an empty row spool")
    finally:
        if parquet_writer is not None:
            parquet_writer.close()


def _read_spooled_rows(row_spool_path: Path) -> list[dict[str, Any]]:
    with row_spool_path.open("r", encoding="utf-8") as spool:
        return [json.loads(line) for line in spool if line.strip()]


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


def _validate_selected_robot_value(
    key: str, values: list[float]
) -> list[float]:
    if key == "action.aligned_target_pos":
        _validate_len(key, values, ALIGNED_TARGET_POS_DIM)
    elif key in {"action.joint_position", "action.policy_action"}:
        _validate_len(key, values, DOF)
    elif key.startswith("observation.state."):
        state_prefix = "observation.state."
        field_name = key[len(state_prefix) :]
        shape = STATE_FIELD_SHAPES.get(field_name)
        if shape is not None:
            _validate_len(key, values, int(shape[0]))
    normalized_values: list[float] = []
    for index, value in enumerate(values):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{key}[{index}] is not numeric: {value!r}"
            ) from exc
        if not math.isfinite(numeric_value):
            raise ValueError(
                f"{key}[{index}] must be finite, got {numeric_value!r}"
            )
        normalized_values.append(numeric_value)
    return normalized_values


def _validate_len(name: str, values: list[float], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} has dimension {len(values)}; expected {expected}")


def _optional_finite_timestamp(
    value: float | None, *, label: str
) -> float | None:
    if value is None:
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(timestamp):
        raise ValueError(f"{label} must be finite, got {timestamp!r}")
    return timestamp


def _validate_monotonic_timestamp(
    value: float, previous: float | None, *, label: str
) -> None:
    if previous is not None and value < previous:
        raise ValueError(
            f"{label} moved backwards: previous={previous!r}, current={value!r}"
        )
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value!r}")


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
    for camera_key in normalized:
        if (
            camera_key in {".", ".."}
            or "/" in camera_key
            or "\\" in camera_key
            or "\x00" in camera_key
        ):
            raise ValueError(
                f"camera key is not safe for dataset paths: {camera_key!r}"
            )
        camera_stream = _camera_stream_from_key(camera_key)
        if camera_stream in {"", ".", ".."}:
            raise ValueError(f"camera key has an invalid stream: {camera_key!r}")
    return normalized


def _validated_dataset_root(root_output_dir: Path, dataset_name: str) -> Path:
    name_path = Path(dataset_name)
    if (
        not dataset_name.strip()
        or name_path.is_absolute()
        or any(part in {"", ".", ".."} for part in name_path.parts)
    ):
        raise ValueError(
            f"dataset_name must be a safe relative path, got {dataset_name!r}"
        )
    root = _safe_path_below(root_output_dir, name_path)
    if root == root_output_dir.resolve():
        raise ValueError("dataset_name must identify a directory below root_output_dir")
    return root


def _safe_path_below(root: Path, relative_path: str | Path) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"path must be relative to dataset root: {relative}")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"path escapes dataset root {resolved_root}: {relative}"
        ) from exc
    return candidate


def _robot_feature_shape(key: str) -> list[int]:
    if key in {"action.joint_position", "action.policy_action"}:
        return [DOF]
    if key == "action.aligned_target_pos":
        return [ALIGNED_TARGET_POS_DIM]
    state_prefix = "observation.state."
    if key.startswith(state_prefix):
        field_name = key[len(state_prefix) :]
        if field_name in STATE_FIELD_SHAPES:
            return list(STATE_FIELD_SHAPES[field_name])
    raise ValueError(f"unsupported robot feature in existing dataset: {key}")


def _joint_order_feature_keys() -> set[str]:
    return {
        "observation.state.joint_position",
        "observation.state.joint_velocity",
        "observation.state.joint_torque",
        "action.joint_position",
        "action.policy_action",
    }


def _validated_joint_names(values: Any, *, label: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    names = [str(value).strip() for value in values]
    if len(names) != DOF:
        raise ValueError(f"{label} has dimension {len(names)}; expected {DOF}")
    if any(not name for name in names):
        raise ValueError(f"{label} must not contain empty names")
    if len(set(names)) != DOF:
        raise ValueError(f"{label} must be unique")
    return names


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
