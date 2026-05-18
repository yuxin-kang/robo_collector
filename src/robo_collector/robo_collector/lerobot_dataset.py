"""Minimal LeRobot v2.1-style writer for Robo Collector episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol


DOF = 29
CAMERA_KEY = "observation.images.ego_view"


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
    joint_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SaveResult:
    saved: bool
    episode_index: int
    frame_count: int
    data_path: Path | None
    video_path: Path | None
    message: str


@dataclass
class _ActiveEpisode:
    episode_index: int
    episode_id: str
    task_prompt: str
    task_index: int
    task_is_new: bool
    global_start_index: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    image_shape: tuple[int, int, int] | None = None
    joint_names: list[str] = field(default_factory=list)
    video_sink: VideoSink | None = None
    video_rel_path: Path | None = None
    data_rel_path: Path | None = None
    video_closed: bool = False


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
        robot_type: str = "unitree_g1",
        parquet_writer: ParquetWriter | None = None,
        video_sink_factory: VideoSinkFactory | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.root_output_dir = Path(root_output_dir)
        self.dataset_name = dataset_name or _default_dataset_name()
        self.root = self.root_output_dir / self.dataset_name
        self.fps = int(fps)
        self.camera_key = camera_key
        self.camera_stream = _camera_stream_from_key(camera_key)
        self.robot_type = robot_type
        self._parquet_writer = parquet_writer or write_parquet_pyarrow
        self._video_sink_factory = video_sink_factory or OpenCvVideoSink
        self._active: _ActiveEpisode | None = None
        self._tasks_by_text: dict[str, int] = {}
        self._episodes: list[dict[str, Any]] = []
        self._total_frames = 0
        self._image_shape: tuple[int, int, int] | None = None
        self._load_existing_metadata()

    @property
    def active_frame_count(self) -> int:
        return len(self._active.rows) if self._active is not None else 0

    @property
    def active_episode_index(self) -> int | None:
        return self._active.episode_index if self._active is not None else None

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
        _validate_robot_frame(frame)

        height, width, channels = _rgb_shape(rgb_frame)
        if channels != 3:
            raise ValueError(f"expected RGB frame with 3 channels, got {channels}")

        active = self._active
        if active.image_shape is None:
            active.image_shape = (height, width, channels)
            self._image_shape = active.image_shape
            active.video_rel_path = Path(
                f"videos/{self.camera_key}/episode_{active.episode_index:06d}.mp4"
            )
            active.data_rel_path = Path(f"data/train-{active.episode_index:06d}.parquet")
            video_path = self.root / active.video_rel_path
            video_path.parent.mkdir(parents=True, exist_ok=True)
            active.video_sink = self._video_sink_factory(
                video_path, self.fps, (width, height)
            )
        elif active.image_shape != (height, width, channels):
            raise ValueError(
                "RGB frame shape changed within episode: "
                f"{active.image_shape} -> {(height, width, channels)}"
            )

        if not active.joint_names and frame.joint_names:
            active.joint_names = list(frame.joint_names)

        assert active.video_sink is not None
        active.video_sink.write(rgb_frame)
        frame_index = len(active.rows)
        timestamp = frame_index / self.fps
        active.rows.append(
            {
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
                "action.policy_action": frame.policy_action,
                "annotation.human.action.task_description": active.task_prompt,
                self.camera_key: {
                    "path": str(active.video_rel_path),
                    "timestamp": timestamp,
                },
                "timestamp": timestamp,
                "frame_index": frame_index,
                "episode_index": active.episode_index,
                "index": active.global_start_index + frame_index,
                "task_index": active.task_index,
            }
        )

    def save_episode(self) -> SaveResult:
        active = self._require_active()
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

        if active.video_sink is not None and not active.video_closed:
            active.video_sink.close()
            active.video_closed = True

        assert active.data_rel_path is not None
        assert active.video_rel_path is not None
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
            video_path=self.root / active.video_rel_path,
            message="episode saved",
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
            "video_path": str(active.video_rel_path),
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
        if active.video_sink is not None:
            active.video_sink.discard()
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
            image_feature = info.get("features", {}).get(self.camera_key, {})
            shape = image_feature.get("shape")
            if isinstance(shape, list) and len(shape) == 3:
                self._image_shape = (int(shape[0]), int(shape[1]), int(shape[2]))

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
        _write_jsonl(
            meta_dir / "tasks.jsonl",
            [
                {"task_index": task_index, "task": task}
                for task, task_index in sorted(
                    tasks_by_text.items(), key=lambda item: item[1]
                )
            ],
        )
        _write_jsonl(meta_dir / "episodes.jsonl", episodes)
        _write_json(
            meta_dir / "info.json",
            self._info(
                active,
                episodes=episodes,
                total_frames=total_frames,
                tasks_by_text=tasks_by_text,
            ),
        )
        _write_json(meta_dir / "modality.json", self._modality(active))

    def _info(
        self,
        active: _ActiveEpisode,
        *,
        episodes: list[dict[str, Any]],
        total_frames: int,
        tasks_by_text: dict[str, int],
    ) -> dict[str, Any]:
        shape = active.image_shape or self._image_shape or (0, 0, 3)
        height, width, channels = shape
        episode_count = len(episodes)
        return {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "total_episodes": episode_count,
            "total_frames": total_frames,
            "total_tasks": len(tasks_by_text),
            "total_videos": episode_count,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{episode_count}"},
            "data_path": "data/train-{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
            "features": self._features(height, width, channels, active.joint_names),
        }

    def _features(
        self, height: int, width: int, channels: int, joint_names: list[str]
    ) -> dict[str, Any]:
        joint_feature = {
            "dtype": "float32",
            "shape": [DOF],
            "names": joint_names if len(joint_names) == DOF else None,
        }
        return {
            self.camera_key: {
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
            },
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
            "action.policy_action": joint_feature,
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
        shape = active.image_shape or self._image_shape or (0, 0, 3)
        return {
            "observation": {
                "images": {
                    self.camera_stream: {
                        "key": self.camera_key,
                        "dtype": "rgb",
                        "shape": list(shape),
                        "fps": self.fps,
                    }
                },
                "state": {
                    "joint_position": {"shape": [DOF]},
                    "joint_velocity": {"shape": [DOF]},
                    "joint_torque": {"shape": [DOF]},
                    "imu_angular_velocity": {"shape": [3]},
                    "imu_linear_acceleration": {"shape": [3]},
                    "projected_gravity_or_quat": {"shape": [4]},
                },
            },
            "action": {
                "joint_position": {"shape": [DOF]},
                "policy_action": {"shape": [DOF]},
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_text_atomic(
        path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _default_dataset_name() -> str:
    return "robo_collector_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _camera_stream_from_key(camera_key: str) -> str:
    prefix = "observation.images."
    if camera_key.startswith(prefix):
        return camera_key[len(prefix) :]
    return camera_key.rsplit(".", 1)[-1]


def _write_text_atomic(path: Path, content: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)
