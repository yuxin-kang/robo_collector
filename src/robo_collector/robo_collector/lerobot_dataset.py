"""Minimal LeRobot v2.1-style writer for Robo Collector episodes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Protocol, TextIO
from uuid import uuid4

from .field_config import FieldSelection, default_field_selection


DOF = 29
CAMERA_KEY = "observation.images.ego_view"
ALIGNED_TARGET_POS_DIM = 45
PARQUET_ROW_GROUP_SIZE = 256
SAVE_PROGRESS_BYTES_INTERVAL = 8 * 1024 * 1024
IN_PROGRESS_DIR = ".inprogress"
METADATA_TRANSACTION_FILENAME = ".metadata-transaction.json"
METADATA_LOCK_FILENAME = ".metadata.lock"
ROBO_COLLECTOR_SCHEMA_VERSION = 1
TIMELINE_SEMANTICS = "fixed_rate_v1"
SOURCE_STATE_TIMESTAMP_KEY = "source_timestamp.state"
SOURCE_CAMERA_TIMESTAMP_PREFIX = "source_timestamp.camera."
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


def _report_save_progress(
    callback: Callable[[str], None] | None,
    phase: str,
) -> None:
    if callback is None:
        return
    try:
        callback(phase)
    except Exception:
        # Progress reporting is advisory and must never invalidate a durable save.
        pass


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
    last_state_source_timestamp_sec: float | None = None
    last_camera_source_timestamps_sec: dict[str, float] = field(default_factory=dict)
    videos_closed: bool = False
    artifacts_committed: bool = False
    committed_artifact_rel_paths: set[Path] = field(default_factory=set)
    artifact_integrity: dict[str, Any] = field(default_factory=dict)
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
        self._uses_default_video_sink = video_sink_factory is None
        self._video_sink_factory = video_sink_factory or OpenCvVideoSink
        self._active: _ActiveEpisode | None = None
        self._tasks_by_text: dict[str, int] = {}
        self._episodes: list[dict[str, Any]] = []
        self._total_frames = 0
        self._image_shapes: dict[str, tuple[int, int, int]] = {}
        self._joint_names: list[str] = []
        self._validated_artifact_fingerprints: dict[
            Path, tuple[int, int, int, str]
        ] = {}
        self._dataset_lock_fd: int | None = None
        self._operation_lock = Lock()
        if not self._metadata_transaction_path().exists():
            self._load_existing_metadata()

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
        with self._writer_operation("start an episode"):
            return self._start_episode(task_prompt, episode_id)

    def _start_episode(self, task_prompt: str, episode_id: str = "") -> int:
        if self._active is not None:
            raise RuntimeError("cannot start a new episode while another is active")
        normalized_prompt = task_prompt.strip()
        if not normalized_prompt:
            raise ValueError("task_prompt is required")

        self._acquire_dataset_lock()
        try:
            # Metadata may have changed since this writer was constructed. Reload it
            # while holding the dataset lock before allocating indexes or recovering
            # abandoned staging files.
            _recover_files_transaction(self._metadata_transaction_path())
            self._reload_existing_metadata()
            self._recover_orphaned_active_episodes()
            self._validate_committed_artifacts()
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
        except Exception:
            self._release_dataset_lock()
            raise

    def add_frame(
        self,
        frame: RobotFrame,
        rgb_frame: Any,
        *,
        camera_timestamps_sec: dict[str, float | None] | None = None,
    ) -> None:
        with self._writer_operation("add a frame"):
            self._add_frame(
                frame,
                rgb_frame,
                camera_timestamps_sec=camera_timestamps_sec,
            )

    def _add_frame(
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
        (
            timestamp,
            video_timestamps,
            state_source_timestamp,
            camera_source_timestamps,
        ) = self._episode_timestamps(
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
                SOURCE_STATE_TIMESTAMP_KEY: state_source_timestamp,
            }
        )
        for camera_key, camera_stream in zip(self.camera_keys, self.camera_streams):
            row[camera_key] = {
                "path": str(active.video_rel_paths[camera_key]),
                "timestamp": video_timestamps[camera_key],
            }
            row[_source_camera_timestamp_key(camera_stream)] = (
                camera_source_timestamps[camera_key]
            )
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
    ) -> tuple[
        float,
        dict[str, float],
        float | None,
        dict[str, float | None],
    ]:
        dataset_timestamp = frame_index / self.fps
        state_source_timestamp = _optional_finite_timestamp(
            state_timestamp_sec, label="state_timestamp_sec"
        )
        camera_source_timestamps = self._normalize_camera_timestamps(
            camera_timestamps_sec
        )

        if state_source_timestamp is not None:
            _validate_monotonic_timestamp(
                state_source_timestamp,
                active.last_state_source_timestamp_sec,
                label="state timestamp",
            )

        video_timestamps = {
            camera_key: dataset_timestamp for camera_key in self.camera_keys
        }
        for camera_key in self.camera_keys:
            source_timestamp = camera_source_timestamps[camera_key]
            if source_timestamp is not None:
                _validate_monotonic_timestamp(
                    source_timestamp,
                    active.last_camera_source_timestamps_sec.get(camera_key),
                    label=f"{camera_key} timestamp",
                )
                active.last_camera_source_timestamps_sec[camera_key] = source_timestamp

        if state_source_timestamp is not None:
            active.last_state_source_timestamp_sec = state_source_timestamp
        return (
            dataset_timestamp,
            video_timestamps,
            state_source_timestamp,
            camera_source_timestamps,
        )

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
        _ensure_directory_durable(work_path, exist_ok=False)
        active.row_spool_rel_path = work_rel_path / "rows.jsonl"
        active.manifest_rel_path = work_rel_path / "manifest.json"
        manifest = {
            "version": 1,
            "phase": "recording",
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
        _write_text_atomic_durable(manifest_path, _json_content(manifest))
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
        if active.artifacts_committed:
            return
        assert active.staged_data_rel_path is not None
        assert active.data_rel_path is not None
        staged_data_path = self._root_path(active.staged_data_rel_path)
        data_path = self._root_path(active.data_rel_path)
        targets = [
            data_path,
            *[
                self._root_path(active.video_rel_paths[camera_key])
                for camera_key in self.camera_keys
            ],
        ]
        collisions = [str(path) for path in targets if path.exists()]
        if collisions:
            active.failed_reason = (
                "refusing to overwrite existing episode artifact(s): "
                + ", ".join(collisions)
            )
            raise RuntimeError(active.failed_reason)

        self._update_active_manifest_phase(active, "committing")

        _ensure_directory_durable(data_path.parent)
        # From the durable "committing" phase onward these collision-checked
        # final paths belong to this attempt, even if rename succeeds and the
        # following directory fsync raises before control returns.
        active.committed_artifact_rel_paths.add(active.data_rel_path)
        _replace_path_durable(staged_data_path, data_path)

        for camera_key in self.camera_keys:
            staged_video_path = self._root_path(
                active.staged_video_rel_paths[camera_key]
            )
            video_path = self._root_path(active.video_rel_paths[camera_key])
            if staged_video_path.exists():
                _ensure_directory_durable(video_path.parent)
                active.committed_artifact_rel_paths.add(
                    active.video_rel_paths[camera_key]
                )
                _replace_path_durable(staged_video_path, video_path)
            else:
                raise RuntimeError(
                    f"missing staged video for {camera_key}: {staged_video_path}"
                )
        self._update_active_manifest_phase(active, "artifacts_committed")
        active.artifacts_committed = True

    def _update_active_manifest_phase(
        self, active: _ActiveEpisode, phase: str
    ) -> None:
        if active.manifest_rel_path is None:
            raise RuntimeError("active episode has no recovery manifest")
        manifest_path = self._root_path(active.manifest_rel_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot update episode recovery manifest: {manifest_path}"
            ) from exc
        if manifest.get("version") != 1:
            raise RuntimeError(f"unsupported episode recovery manifest: {manifest_path}")
        manifest["phase"] = phase
        _write_text_atomic_durable(manifest_path, _json_content(manifest))

    def _cleanup_active_staging(
        self, active: _ActiveEpisode, *, remove_committed: bool
    ) -> None:
        self._close_row_spool(active)
        relative_paths = [
            active.row_spool_rel_path,
            active.staged_data_rel_path,
            *active.staged_video_rel_paths.values(),
        ]
        if remove_committed:
            relative_paths.extend(active.committed_artifact_rel_paths)
        for relative_path in relative_paths:
            if relative_path is not None:
                _unlink_path_durable(self._root_path(relative_path))
        # The manifest is the recovery authority. Remove it only after every
        # staged/owned final artifact deletion is durable in its own directory.
        if active.manifest_rel_path is not None:
            _unlink_path_durable(self._root_path(active.manifest_rel_path))
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

    def _metadata_transaction_path(self) -> Path:
        return self.root / "meta" / METADATA_TRANSACTION_FILENAME

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

    def save_episode(
        self,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> SaveResult:
        with self._writer_operation("save an episode"):
            return self._save_episode(progress_callback=progress_callback)

    def _save_episode(
        self,
        *,
        progress_callback: Callable[[str], None] | None,
    ) -> SaveResult:
        active = self._require_active()
        if active.failed_reason is not None:
            raise RuntimeError(
                "cannot save failed episode; discard required: "
                f"{active.failed_reason}"
            )
        if active.frame_count == 0:
            _report_save_progress(progress_callback, "discarding_empty")
            self._discard_active()
            _report_save_progress(progress_callback, "complete")
            return SaveResult(
                saved=False,
                episode_index=active.episode_index,
                frame_count=0,
                data_path=None,
                video_path=None,
                message="discarded empty episode",
            )

        _report_save_progress(progress_callback, "closing_video")
        if not active.videos_closed:
            for video_sink in active.video_sinks.values():
                video_sink.close()
            active.videos_closed = True
        self._close_row_spool(active)

        assert active.data_rel_path is not None
        assert active.staged_data_rel_path is not None
        assert active.row_spool_rel_path is not None
        assert active.video_rel_paths
        if not active.artifacts_committed:
            staged_data_path = self._root_path(active.staged_data_rel_path)
            staged_data_path.parent.mkdir(parents=True, exist_ok=True)
            row_spool_path = self._root_path(active.row_spool_rel_path)
            try:
                _report_save_progress(progress_callback, "writing_parquet")
                if self._uses_default_parquet_writer:
                    write_parquet_pyarrow_stream(
                        staged_data_path,
                        row_spool_path,
                        batch_size=PARQUET_ROW_GROUP_SIZE,
                        progress_callback=progress_callback,
                    )
                    _validate_parquet_row_count(
                        staged_data_path, active.frame_count
                    )
                else:
                    self._parquet_writer(
                        staged_data_path, _read_spooled_rows(row_spool_path)
                    )
                _report_save_progress(progress_callback, "validating_artifacts")
                _require_nonempty_file(staged_data_path, label="episode parquet")
                for camera_key in self.camera_keys:
                    staged_video_path = self._root_path(
                        active.staged_video_rel_paths[camera_key]
                    )
                    _require_nonempty_file(
                        staged_video_path,
                        label=f"video for {camera_key}",
                    )
                    if self._uses_default_video_sink:
                        _validate_video_frame_count(
                            staged_video_path, active.frame_count
                        )
                _report_save_progress(progress_callback, "syncing_artifacts")
                _fsync_file(staged_data_path)
                for camera_key in self.camera_keys:
                    _fsync_file(
                        self._root_path(active.staged_video_rel_paths[camera_key])
                    )
                _report_save_progress(progress_callback, "committing_artifacts")
                self._commit_active_artifacts(active)
            except Exception as exc:
                if active.failed_reason is None:
                    active.failed_reason = f"episode artifact validation failed: {exc}"
                raise
        data_path = self._root_path(active.data_rel_path)
        if not active.artifact_integrity:
            _report_save_progress(progress_callback, "hashing_artifacts")
            active.artifact_integrity = self._artifact_integrity(
                active,
                progress_callback=progress_callback,
            )

        episode_record = self._episode_record(active)
        pending_episodes = [*self._episodes, episode_record]
        pending_total_frames = self._total_frames + active.frame_count
        _report_save_progress(progress_callback, "committing_metadata")
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
        try:
            _report_save_progress(progress_callback, "cleaning_up")
            self._cleanup_active_staging(active, remove_committed=False)
        except OSError:
            # Metadata and artifacts have crossed the durable commit point. Keep
            # reporting success and leave any manifest/staging residue for the
            # next locked startup to clean up safely.
            pass
        finally:
            self._active = None
            self._release_dataset_lock()
        _report_save_progress(progress_callback, "complete")
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
            "integrity": active.artifact_integrity,
        }

    def discard_episode(self) -> None:
        with self._writer_operation("discard an episode"):
            self._require_active()
            self._discard_active()

    @contextmanager
    def _writer_operation(self, operation: str):
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError(
                f"cannot {operation} while another writer operation is running"
            )
        try:
            yield
        finally:
            self._operation_lock.release()

    def _discard_active(self) -> None:
        active = self._active
        if active is None:
            self._release_dataset_lock()
            return
        discarded = False
        try:
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
            discarded = True
        finally:
            if discarded:
                self._release_dataset_lock()

    def _artifact_integrity(
        self,
        active: _ActiveEpisode,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        assert active.data_rel_path is not None
        return {
            "algorithm": "sha256",
            "data": {
                **_file_integrity(
                    self._root_path(active.data_rel_path),
                    progress_callback=progress_callback,
                ),
                "rows": active.frame_count,
            },
            "videos": {
                camera_key: {
                    **_file_integrity(
                        self._root_path(active.video_rel_paths[camera_key]),
                        progress_callback=progress_callback,
                    ),
                    "frames": active.frame_count,
                }
                for camera_key in self.camera_keys
            },
        }

    def _acquire_dataset_lock(self) -> None:
        if self._dataset_lock_fd is not None:
            raise RuntimeError("dataset writer lock is already held")
        digest = hashlib.sha256(str(self.root).encode("utf-8")).hexdigest()[:20]
        lock_dir = self.root_output_dir / ".robo_collector_locks" / digest
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(lock_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(lock_fd)
            raise RuntimeError(
                f"dataset is locked by another active writer: {self.root}"
            ) from exc
        self._dataset_lock_fd = lock_fd

    def _release_dataset_lock(self) -> None:
        lock_fd = self._dataset_lock_fd
        if lock_fd is None:
            return
        self._dataset_lock_fd = None
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    def __del__(self) -> None:
        try:
            self._release_dataset_lock()
        except Exception:
            pass

    def _reload_existing_metadata(self) -> None:
        self._tasks_by_text = {}
        self._episodes = []
        self._total_frames = 0
        self._image_shapes = {}
        self._joint_names = []
        self._load_existing_metadata()

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
        meta_dir = self.root / "meta"
        required_paths = {
            "tasks": meta_dir / "tasks.jsonl",
            "episodes": meta_dir / "episodes.jsonl",
            "info": meta_dir / "info.json",
            "modality": meta_dir / "modality.json",
        }
        existing_names = {
            name for name, path in required_paths.items() if path.exists()
        }
        if existing_names and existing_names != set(required_paths):
            missing_names = sorted(set(required_paths) - existing_names)
            raise RuntimeError(
                "existing dataset metadata is incomplete; missing "
                + ", ".join(missing_names)
            )
        if not existing_names:
            return

        tasks_path = required_paths["tasks"]
        task_indexes: set[int] = set()
        if tasks_path.exists():
            for line in tasks_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                task = str(row["task"])
                task_index = int(row["task_index"])
                if task in self._tasks_by_text or task_index in task_indexes:
                    raise ValueError("existing tasks metadata contains duplicates")
                self._tasks_by_text[task] = task_index
                task_indexes.add(task_index)
        if task_indexes != set(range(len(task_indexes))):
            raise ValueError("existing task indexes must be contiguous from zero")

        episodes_path = required_paths["episodes"]
        if episodes_path.exists():
            for line in episodes_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                episode = json.loads(line)
                self._episodes.append(episode)
                self._total_frames = max(
                    self._total_frames, int(episode.get("dataset_to_index", 0))
                )

        info_path = required_paths["info"]
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
            if (
                info.get("robo_collector_schema_version")
                != ROBO_COLLECTOR_SCHEMA_VERSION
            ):
                raise ValueError(
                    "existing dataset has no compatible robo_collector schema "
                    "marker; use a new dataset_name or migrate it explicitly"
                )
            if info.get("timeline_semantics") != TIMELINE_SEMANTICS:
                raise ValueError(
                    "existing dataset timeline semantics are incompatible; use a "
                    "new dataset_name or migrate it explicitly"
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
            self._validate_existing_source_timestamp_features(features)
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
            self._validate_existing_metadata_consistency(info)

        try:
            modality = json.loads(
                required_paths["modality"].read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("existing modality metadata is invalid") from exc
        if not isinstance(modality, dict):
            raise ValueError("existing modality metadata must be a mapping")

    def _validate_existing_metadata_consistency(
        self, info: dict[str, Any]
    ) -> None:
        sorted_episodes = sorted(
            self._episodes, key=lambda item: int(item["episode_index"])
        )
        expected_indexes = list(range(len(sorted_episodes)))
        actual_indexes = [int(item["episode_index"]) for item in sorted_episodes]
        if actual_indexes != expected_indexes:
            raise ValueError("existing episode indexes must be contiguous from zero")

        expected_from_index = 0
        task_indexes = set(self._tasks_by_text.values())
        for episode in sorted_episodes:
            episode_index = int(episode["episode_index"])
            length = int(episode.get("length", 0))
            from_index = int(episode.get("dataset_from_index", -1))
            to_index = int(episode.get("dataset_to_index", -1))
            if length <= 0:
                raise ValueError(
                    f"existing episode {episode_index} length must be positive"
                )
            if from_index != expected_from_index or to_index != from_index + length:
                raise ValueError(
                    f"existing episode {episode_index} has inconsistent frame range"
                )
            if int(episode.get("fps", -1)) != self.fps:
                raise ValueError(
                    f"existing episode {episode_index} fps does not match dataset"
                )
            if int(episode.get("task_index", -1)) not in task_indexes:
                raise ValueError(
                    f"existing episode {episode_index} references an unknown task"
                )
            expected_from_index = to_index

        expected_counts = {
            "total_episodes": len(sorted_episodes),
            "total_frames": expected_from_index,
            "total_tasks": len(self._tasks_by_text),
            "total_videos": len(sorted_episodes) * len(self.camera_keys),
        }
        for key, expected in expected_counts.items():
            if info.get(key) != expected:
                raise ValueError(
                    f"existing info.json {key} mismatch: expected {expected}, "
                    f"got {info.get(key)!r}"
                )
        self._episodes = sorted_episodes
        self._total_frames = expected_from_index

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

    def _validate_existing_source_timestamp_features(
        self, features: dict[str, Any]
    ) -> None:
        expected_keys = {
            SOURCE_STATE_TIMESTAMP_KEY,
            *(
                _source_camera_timestamp_key(camera_stream)
                for camera_stream in self.camera_streams
            ),
        }
        existing_keys = {
            key for key in features if key.startswith("source_timestamp.")
        }
        if not existing_keys:
            raise ValueError(
                "existing dataset predates the fixed-rate timeline/source "
                "timestamp schema; use a new dataset_name or migrate it explicitly"
            )
        if existing_keys != expected_keys:
            raise ValueError(
                "source timestamp features do not match existing dataset: "
                f"expected {sorted(expected_keys)}, got {sorted(existing_keys)}"
            )
        for key in sorted(expected_keys):
            feature = features.get(key)
            if not isinstance(feature, dict) or feature.get("dtype") != "float64":
                raise ValueError(
                    f"existing source timestamp feature {key} dtype must be float64"
                )
            if feature.get("shape") != [1]:
                raise ValueError(
                    f"existing source timestamp feature {key} shape must be [1]"
                )

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
                phase = manifest.get("phase", "recording")
                if manifest.get("version", 1) != 1 or phase not in {
                    "recording",
                    "committing",
                    "artifacts_committed",
                }:
                    raise ValueError("unsupported episode manifest phase")
                artifact_pairs = _manifest_artifact_pairs(manifest)
                row_spool_path = _required_manifest_path(
                    manifest, "row_spool_path"
                )
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot safely recover invalid episode manifest: {manifest_path}"
                ) from exc

            committed = episode_index in committed_episode_indexes
            if committed:
                if phase == "recording":
                    raise RuntimeError(
                        "committed episode has a pre-commit recovery manifest: "
                        f"{manifest_path}"
                    )
                for staged_relative, final_relative in artifact_pairs:
                    staged_path = self._root_path(staged_relative)
                    final_path = self._root_path(final_relative)
                    if final_path.exists():
                        _unlink_path_durable(staged_path)
                        continue
                    if not staged_path.exists():
                        raise RuntimeError(
                            "committed episode artifact is missing from both final "
                            f"and staging paths: {final_path}"
                        )
                    _ensure_directory_durable(final_path.parent)
                    _replace_path_durable(staged_path, final_path)
            else:
                for staged_relative, final_relative in artifact_pairs:
                    _unlink_path_durable(self._root_path(staged_relative))
                    if phase in {"committing", "artifacts_committed"}:
                        _unlink_path_durable(self._root_path(final_relative))

            _unlink_path_durable(self._root_path(row_spool_path))
            _unlink_path_durable(manifest_path)
            self._remove_empty_work_dir(manifest_path.parent.name)

    def _validate_committed_artifacts(self) -> None:
        for episode in self._episodes:
            episode_index = int(episode["episode_index"])
            episode_length = int(episode["length"])
            integrity = episode.get("integrity")
            if not isinstance(integrity, dict) or integrity.get("algorithm") != "sha256":
                raise RuntimeError(
                    f"episode {episode_index} is missing a supported integrity manifest"
                )
            data_integrity = integrity.get("data")
            if not isinstance(data_integrity, dict):
                raise RuntimeError(
                    f"episode {episode_index} has invalid data integrity metadata"
                )
            if data_integrity.get("rows") != episode_length:
                raise RuntimeError(
                    f"episode {episode_index} parquet integrity row count mismatch"
                )
            self._validate_committed_artifact(
                episode_index=episode_index,
                relative_path=episode.get("data_path"),
                integrity=data_integrity,
                label="parquet",
            )

            video_paths = episode.get("video_paths")
            video_integrity = integrity.get("videos")
            if not isinstance(video_paths, dict) or not isinstance(video_integrity, dict):
                raise RuntimeError(
                    f"episode {episode_index} has invalid video artifact metadata"
                )
            if set(video_integrity) != set(self.camera_keys):
                raise RuntimeError(
                    f"episode {episode_index} video integrity does not match configured cameras"
                )
            if set(video_paths) != set(self.camera_keys):
                raise RuntimeError(
                    f"episode {episode_index} video paths do not match configured cameras"
                )
            for camera_key in self.camera_keys:
                item_integrity = video_integrity.get(camera_key)
                if not isinstance(item_integrity, dict):
                    raise RuntimeError(
                        f"episode {episode_index} is missing integrity for {camera_key}"
                    )
                if item_integrity.get("frames") != episode_length:
                    raise RuntimeError(
                        f"episode {episode_index} video integrity frame count mismatch "
                        f"for {camera_key}"
                    )
                self._validate_committed_artifact(
                    episode_index=episode_index,
                    relative_path=video_paths[camera_key],
                    integrity=item_integrity,
                    label=f"video {camera_key}",
                )

    def _validate_committed_artifact(
        self,
        *,
        episode_index: int,
        relative_path: Any,
        integrity: dict[str, Any],
        label: str,
    ) -> None:
        if not isinstance(relative_path, str) or not relative_path:
            raise RuntimeError(
                f"episode {episode_index} has no path for committed {label}"
            )
        path = self._root_path(relative_path)
        expected_sha256 = integrity.get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise RuntimeError(
                f"episode {episode_index} has invalid SHA-256 metadata for {label}"
            )
        try:
            int(expected_sha256, 16)
        except ValueError as exc:
            raise RuntimeError(
                f"episode {episode_index} has invalid SHA-256 metadata for {label}"
            ) from exc
        try:
            stat = path.stat()
        except OSError as exc:
            raise RuntimeError(
                f"episode {episode_index} committed {label} is missing: {path}"
            ) from exc
        expected_size = integrity.get("size_bytes")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            raise RuntimeError(
                f"episode {episode_index} has invalid size metadata for {label}"
            )
        actual_size = stat.st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"episode {episode_index} committed {label} size mismatch: "
                f"expected {expected_size}, got {actual_size}"
            )
        fingerprint = (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ino,
            expected_sha256.lower(),
        )
        if self._validated_artifact_fingerprints.get(path) == fingerprint:
            return
        actual_integrity = _file_integrity(path)
        actual_sha256 = actual_integrity["sha256"]
        if actual_sha256 != expected_sha256.lower():
            raise RuntimeError(
                f"episode {episode_index} committed {label} SHA-256 mismatch"
            )
        self._validated_artifact_fingerprints[path] = fingerprint

    def _write_metadata(
        self,
        active: _ActiveEpisode,
        *,
        episodes: list[dict[str, Any]],
        total_frames: int,
        tasks_by_text: dict[str, int],
    ) -> None:
        meta_dir = self.root / "meta"
        _ensure_directory_durable(meta_dir)
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
            "robo_collector_schema_version": ROBO_COLLECTOR_SCHEMA_VERSION,
            "timeline_semantics": TIMELINE_SEMANTICS,
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
        source_timestamp_features = {
            SOURCE_STATE_TIMESTAMP_KEY: {
                "dtype": "float64",
                "shape": [1],
                "names": None,
            },
            **{
                _source_camera_timestamp_key(camera_stream): {
                    "dtype": "float64",
                    "shape": [1],
                    "names": None,
                }
                for camera_stream in self.camera_streams
            },
        }

        return {
            **video_features,
            **selected_robot_features,
            **source_timestamp_features,
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

    if not rows:
        raise ValueError("cannot write parquet from an empty row list")
    schema = _arrow_schema_for_row(pa, rows[0])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="snappy")


def write_parquet_pyarrow_stream(
    path: Path,
    row_spool_path: Path,
    *,
    batch_size: int,
    progress_callback: Callable[[str], None] | None = None,
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
                if schema is None:
                    schema = _arrow_schema_for_row(pa, batch[0])
                table = pa.Table.from_pylist(batch, schema=schema)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(
                        path, schema, compression="snappy"
                    )
                parquet_writer.write_table(table, row_group_size=len(batch))
                batch.clear()
                _report_save_progress(progress_callback, "writing_parquet")
            if batch:
                if schema is None:
                    schema = _arrow_schema_for_row(pa, batch[0])
                table = pa.Table.from_pylist(batch, schema=schema)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(
                        path, schema, compression="snappy"
                    )
                parquet_writer.write_table(table, row_group_size=len(batch))
                _report_save_progress(progress_callback, "writing_parquet")
        if parquet_writer is None:
            raise ValueError("cannot write parquet from an empty row spool")
    finally:
        if parquet_writer is not None:
            parquet_writer.close()


def _read_spooled_rows(row_spool_path: Path) -> list[dict[str, Any]]:
    with row_spool_path.open("r", encoding="utf-8") as spool:
        return [json.loads(line) for line in spool if line.strip()]


def _arrow_schema_for_row(pa: Any, row: dict[str, Any]) -> Any:
    video_reference_type = pa.struct(
        [
            pa.field("path", pa.string()),
            pa.field("timestamp", pa.float32()),
        ]
    )
    fields = []
    for key, value in row.items():
        if key.startswith("action.") or key.startswith("observation.state."):
            value_type = pa.list_(pa.float32())
        elif key == "annotation.human.action.task_description":
            value_type = pa.string()
        elif key == "timestamp":
            value_type = pa.float32()
        elif key.startswith("source_timestamp."):
            value_type = pa.float64()
        elif key in {"frame_index", "episode_index", "index", "task_index"}:
            value_type = pa.int64()
        elif isinstance(value, dict) and set(value) == {"path", "timestamp"}:
            value_type = video_reference_type
        else:
            raise ValueError(f"unsupported parquet field {key!r}")
        fields.append(pa.field(key, value_type))
    return pa.schema(fields)


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


def _source_camera_timestamp_key(camera_stream: str) -> str:
    return SOURCE_CAMERA_TIMESTAMP_PREFIX + camera_stream


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


def _require_nonempty_file(path: Path, *, label: str) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"{label} is missing: {path}") from exc
    if size <= 0:
        raise RuntimeError(f"{label} is empty: {path}")


def _validate_parquet_row_count(path: Path, expected_rows: int) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to validate episode parquet") from exc
    actual_rows = int(pq.ParquetFile(path).metadata.num_rows)
    if actual_rows != expected_rows:
        raise RuntimeError(
            "parquet row count mismatch: "
            f"expected {expected_rows}, got {actual_rows}"
        )


def _validate_video_frame_count(path: Path, expected_frames: int) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to validate episode video") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open encoded episode video: {path}")
        actual_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if actual_frames != expected_frames:
        raise RuntimeError(
            "video frame count mismatch for "
            f"{path}: expected {expected_frames}, got {actual_frames}"
        )


def _file_integrity(
    path: Path,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    bytes_since_progress = 0
    with path.open("rb") as artifact:
        while True:
            chunk = artifact.read(1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            bytes_since_progress += len(chunk)
            digest.update(chunk)
            if bytes_since_progress >= SAVE_PROGRESS_BYTES_INTERVAL:
                _report_save_progress(progress_callback, "hashing_artifacts")
                bytes_since_progress = 0
    if size_bytes <= 0:
        raise RuntimeError(f"cannot record integrity for empty artifact: {path}")
    return {"size_bytes": size_bytes, "sha256": digest.hexdigest()}


def _required_manifest_path(manifest: dict[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"episode manifest {key} must be a non-empty path")
    return value


def _manifest_artifact_pairs(
    manifest: dict[str, Any],
) -> list[tuple[str, str]]:
    data_pair = (
        _required_manifest_path(manifest, "staged_data_path"),
        _required_manifest_path(manifest, "data_path"),
    )
    staged_video_paths = manifest.get("staged_video_paths")
    video_paths = manifest.get("video_paths")
    if not isinstance(staged_video_paths, dict) or not isinstance(video_paths, dict):
        raise ValueError("episode manifest video paths must be mappings")
    if not staged_video_paths or set(staged_video_paths) != set(video_paths):
        raise ValueError("episode manifest staged/final video keys do not match")
    video_pairs = []
    for camera_key in sorted(video_paths):
        staged_path = staged_video_paths[camera_key]
        final_path = video_paths[camera_key]
        if not isinstance(staged_path, str) or not staged_path:
            raise ValueError(
                f"episode manifest staged path for {camera_key} is invalid"
            )
        if not isinstance(final_path, str) or not final_path:
            raise ValueError(
                f"episode manifest final path for {camera_key} is invalid"
            )
        video_pairs.append((staged_path, final_path))
    return [data_pair, *video_pairs]


def _write_files_transactional(files: dict[Path, str]) -> None:
    if not files:
        return
    parent_dirs = {path.parent.resolve() for path in files}
    if len(parent_dirs) != 1:
        raise ValueError("transactional files must share one parent directory")
    parent_dir = next(iter(parent_dirs))
    with _metadata_files_lock(parent_dir, exclusive=True):
        _write_files_transactional_locked(files)


def _write_files_transactional_locked(files: dict[Path, str]) -> None:
    if not files:
        return
    parent_dirs = {path.parent.resolve() for path in files}
    if len(parent_dirs) != 1:
        raise ValueError("transactional files must share one parent directory")
    parent_dir = next(iter(parent_dirs))
    _ensure_directory_durable(parent_dir)
    token = uuid4().hex
    journal_path = parent_dir / METADATA_TRANSACTION_FILENAME
    if journal_path.exists():
        raise RuntimeError(
            f"unfinished metadata transaction must be recovered first: {journal_path}"
        )
    entries: list[dict[str, Any]] = []
    committed = False

    try:
        for path, content in files.items():
            staging_path = path.with_name(f".{path.name}.{token}.tmp")
            backup_path = path.with_name(f".{path.name}.{token}.bak")
            _write_text_durable(staging_path, content)
            entries.append(
                {
                    "target": path.name,
                    "staging": staging_path.name,
                    "backup": backup_path.name,
                    "had_original": path.exists(),
                }
            )

        journal = {"version": 1, "phase": "prepared", "entries": entries}
        _write_text_atomic_durable(journal_path, _json_content(journal))

        for entry in entries:
            path = parent_dir / entry["target"]
            staging_path = parent_dir / entry["staging"]
            backup_path = parent_dir / entry["backup"]
            if entry["had_original"]:
                _replace_path(path, backup_path)
            _replace_path(staging_path, path)
            _fsync_directory(parent_dir)

        journal["phase"] = "committed"
        _write_text_atomic_durable(journal_path, _json_content(journal))
        committed = True

        _cleanup_transaction_files(parent_dir, entries)
        journal_path.unlink(missing_ok=True)
        _fsync_directory(parent_dir)
    except Exception:
        if committed:
            # All targets and the committed journal are already durable. Leave the
            # journal in place when cleanup fails so startup can finish roll-forward
            # without ever restoring a partially deleted backup generation.
            return
        _rollback_transaction(parent_dir, entries)
        journal_path.unlink(missing_ok=True)
        _fsync_directory(parent_dir)
        raise


def _recover_files_transaction(journal_path: Path) -> None:
    if not journal_path.parent.exists():
        return
    with _metadata_files_lock(journal_path.parent, exclusive=True):
        _recover_files_transaction_locked(journal_path)


def _recover_files_transaction_locked(journal_path: Path) -> None:
    if not journal_path.exists():
        _cleanup_atomic_text_staging(journal_path)
        return
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot recover invalid metadata transaction journal: {journal_path}"
        ) from exc
    if journal.get("version") != 1 or journal.get("phase") not in {
        "prepared",
        "committed",
    }:
        raise RuntimeError(f"unsupported metadata transaction journal: {journal_path}")
    entries = journal.get("entries")
    if not isinstance(entries, list) or not all(
        _valid_transaction_entry(entry) for entry in entries
    ):
        raise RuntimeError(f"invalid metadata transaction entries: {journal_path}")

    parent_dir = journal_path.parent
    if journal["phase"] == "prepared":
        _rollback_transaction(parent_dir, entries)
    else:
        for entry in entries:
            target = parent_dir / entry["target"]
            staging = parent_dir / entry["staging"]
            if not target.exists() and staging.exists():
                _replace_path_durable(staging, target)
            if not target.exists():
                raise RuntimeError(
                    f"committed metadata transaction is missing {target}"
                )
        _cleanup_transaction_files(parent_dir, entries)
    journal_path.unlink(missing_ok=True)
    _cleanup_atomic_text_staging(journal_path)
    _fsync_directory(parent_dir)


@contextmanager
def _metadata_files_lock(parent_dir: Path, *, exclusive: bool):
    parent_dir.mkdir(parents=True, exist_ok=True)
    lock_path = parent_dir / METADATA_LOCK_FILENAME
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _valid_transaction_entry(entry: Any) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("had_original"), bool):
        return False
    for key in ("target", "staging", "backup"):
        value = entry.get(key)
        if (
            not isinstance(value, str)
            or not value
            or Path(value).name != value
            or value in {".", ".."}
        ):
            return False
    return True


def _rollback_transaction(parent_dir: Path, entries: list[dict[str, Any]]) -> None:
    for entry in reversed(entries):
        target = parent_dir / entry["target"]
        staging = parent_dir / entry["staging"]
        backup = parent_dir / entry["backup"]
        if entry["had_original"]:
            if backup.exists():
                target.unlink(missing_ok=True)
                backup.replace(target)
        else:
            target.unlink(missing_ok=True)
        staging.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)


def _cleanup_transaction_files(
    parent_dir: Path, entries: list[dict[str, Any]]
) -> None:
    for entry in entries:
        (parent_dir / entry["staging"]).unlink(missing_ok=True)
        (parent_dir / entry["backup"]).unlink(missing_ok=True)


def _write_text_durable(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _write_text_atomic_durable(path: Path, content: str) -> None:
    staging_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        _write_text_durable(staging_path, content)
        _replace_path(staging_path, path)
        _fsync_directory(path.parent)
    finally:
        staging_path.unlink(missing_ok=True)


def _cleanup_atomic_text_staging(path: Path) -> None:
    removed = False
    for staging_path in path.parent.glob(f".{path.name}.*.tmp"):
        staging_path.unlink(missing_ok=True)
        removed = True
    if removed:
        _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _unlink_path_durable(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        if path.parent.exists():
            # The file may have been removed by a sink-specific discard method;
            # persist that directory state before the recovery manifest is removed.
            _fsync_directory(path.parent)
        return
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _ensure_directory_durable(path: Path, *, exist_ok: bool = True) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(path)
        if not exist_ok:
            raise FileExistsError(path)
        return

    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    path.mkdir(parents=True, exist_ok=False)
    for directory in reversed(missing):
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _replace_path(source: Path, target: Path) -> None:
    source.replace(target)


def _replace_path_durable(source: Path, target: Path) -> None:
    source_parent = source.parent.resolve()
    target_parent = target.parent.resolve()
    _replace_path(source, target)
    _fsync_directory(target_parent)
    if source_parent != target_parent:
        _fsync_directory(source_parent)
