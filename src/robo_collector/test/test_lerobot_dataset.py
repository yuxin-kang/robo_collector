import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from robo_collector import lerobot_dataset
from robo_collector.field_config import FieldSelection
from robo_collector.lerobot_dataset import DOF, LeRobotV21Writer, RobotFrame


class FakeVideoSink:
    def __init__(self, path: Path, fps: int, frame_size: tuple[int, int]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"fake-mp4")
        self.frames = 0

    def write(self, rgb_frame):
        self.frames += 1

    def close(self):
        self.path.write_bytes(self.path.read_bytes() + f":{self.frames}".encode())

    def discard(self):
        self.path.unlink(missing_ok=True)


class FailingVideoSink(FakeVideoSink):
    def write(self, rgb_frame):
        raise RuntimeError("video write failed")


class FakeFrame:
    shape = (4, 6, 3)


class LeRobotV21WriterTest(unittest.TestCase):
    def test_idle_writer_does_not_create_dataset(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)

            self.assertFalse(writer.root.exists())

    def test_save_episode_writes_structure_and_task_annotation(self):
        parquet_rows = {}

        def write_fake_parquet(path, rows):
            path.parent.mkdir(parents=True, exist_ok=True)
            parquet_rows[path.name] = rows
            path.write_text(json.dumps(rows), encoding="utf-8")

        with TemporaryDirectory() as tmp:
            writer = _writer(tmp, parquet_writer=write_fake_parquet)
            episode_index = writer.start_episode("pick the red cup", "manual-1")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.add_frame(_robot_frame(), FakeFrame())

            result = writer.save_episode()

            self.assertTrue(result.saved)
            self.assertEqual(episode_index, 0)
            self.assertEqual(result.frame_count, 2)
            root = Path(tmp) / "dataset"
            self.assertTrue((root / "data/train-000000.parquet").exists())
            self.assertTrue(
                (
                    root
                    / "videos/observation.images.ego_view/episode_000000.mp4"
                ).exists()
            )
            self.assertTrue((root / "meta/info.json").exists())
            self.assertTrue((root / "meta/modality.json").exists())
            self.assertTrue((root / "meta/episodes.jsonl").exists())
            self.assertTrue((root / "meta/tasks.jsonl").exists())

            rows = parquet_rows["train-000000.parquet"]
            self.assertEqual(
                rows[0]["annotation.human.action.task_description"],
                "pick the red cup",
            )
            self.assertEqual(rows[0]["task_index"], 0)
            self.assertEqual(rows[1]["timestamp"], 1 / 50)
            self.assertEqual(rows[0]["action.policy_action"], [5.0] * DOF)
            self.assertNotIn("action.aligned_target_pos", rows[0])
            self.assertNotIn("observation.state.relative_ori_6d", rows[0])
            self.assertEqual(
                rows[0]["observation.images.ego_view"],
                {
                    "path": "videos/observation.images.ego_view/episode_000000.mp4",
                    "timestamp": 0.0,
                },
            )

            info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(info["total_frames"], 2)
            self.assertEqual(info["features"]["action.joint_position"]["shape"], [29])
            self.assertEqual(info["features"]["action.policy_action"]["shape"], [29])
            self.assertEqual(
                info["features"]["observation.images.ego_view"]["shape"], [4, 6, 3]
            )

            modality = json.loads(
                (root / "meta/modality.json").read_text(encoding="utf-8")
            )
            self.assertIn("policy_action", modality["action"])

            task = json.loads(
                (root / "meta/tasks.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(task, {"task_index": 0, "task": "pick the red cup"})

    def test_save_episode_writes_two_camera_video_features(self):
        parquet_rows = {}

        def write_fake_parquet(path, rows):
            path.parent.mkdir(parents=True, exist_ok=True)
            parquet_rows[path.name] = rows
            path.write_text(json.dumps(rows), encoding="utf-8")

        camera_keys = [
            "observation.images.head",
            "observation.images.ego_view",
        ]
        with TemporaryDirectory() as tmp:
            writer = _writer(
                tmp, parquet_writer=write_fake_parquet, camera_keys=camera_keys
            )
            writer.start_episode("pick the red cup", "manual-1")
            writer.add_frame(
                _robot_frame(),
                {"head": FakeFrame(), "ego_view": FakeFrame()},
            )

            result = writer.save_episode()

            root = Path(tmp) / "dataset"
            self.assertTrue(result.saved)
            self.assertEqual(
                set(result.video_paths),
                {"observation.images.head", "observation.images.ego_view"},
            )
            self.assertTrue(
                (
                    root
                    / "videos/observation.images.head/episode_000000.mp4"
                ).exists()
            )
            self.assertTrue(
                (
                    root
                    / "videos/observation.images.ego_view/episode_000000.mp4"
                ).exists()
            )

            row = parquet_rows["train-000000.parquet"][0]
            self.assertEqual(
                row["observation.images.head"],
                {
                    "path": "videos/observation.images.head/episode_000000.mp4",
                    "timestamp": 0.0,
                },
            )
            self.assertEqual(
                row["observation.images.ego_view"],
                {
                    "path": "videos/observation.images.ego_view/episode_000000.mp4",
                    "timestamp": 0.0,
                },
            )

            info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["total_videos"], 2)
            self.assertEqual(
                info["features"]["observation.images.head"]["shape"], [4, 6, 3]
            )
            self.assertEqual(
                info["features"]["observation.images.ego_view"]["shape"], [4, 6, 3]
            )

            modality = json.loads(
                (root / "meta/modality.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(modality["observation"]["images"]), {"head", "ego_view"}
            )

    def test_field_selection_writes_only_selected_robot_fields(self):
        parquet_rows = {}

        def write_fake_parquet(path, rows):
            path.parent.mkdir(parents=True, exist_ok=True)
            parquet_rows[path.name] = rows
            path.write_text(json.dumps(rows), encoding="utf-8")

        selection = FieldSelection(
            target=("joint_position",),
            state=("joint_position",),
        )
        with TemporaryDirectory() as tmp:
            writer = _writer(
                tmp,
                parquet_writer=write_fake_parquet,
                field_selection=selection,
            )
            writer.start_episode("field subset")
            writer.add_frame(_robot_frame(), FakeFrame())

            writer.save_episode()

            root = Path(tmp) / "dataset"
            row = parquet_rows["train-000000.parquet"][0]
            robot_columns = {
                key
                for key in row
                if key.startswith("action.") or key.startswith("observation.state.")
            }
            self.assertEqual(
                robot_columns,
                {
                    "action.joint_position",
                    "observation.state.joint_position",
                },
            )
            self.assertEqual(row["action.joint_position"], [4.0] * DOF)
            self.assertEqual(
                row["observation.state.joint_position"], [1.0] * DOF
            )
            self.assertIn("timestamp", row)
            self.assertIn("frame_index", row)
            self.assertIn("episode_index", row)
            self.assertIn("index", row)
            self.assertIn("task_index", row)
            self.assertIn("annotation.human.action.task_description", row)
            self.assertIn("observation.images.ego_view", row)

            info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
            feature_robot_columns = {
                key
                for key in info["features"]
                if key.startswith("action.") or key.startswith("observation.state.")
            }
            self.assertEqual(
                feature_robot_columns,
                {
                    "action.joint_position",
                    "observation.state.joint_position",
                },
            )
            self.assertIn("timestamp", info["features"])
            self.assertIn("frame_index", info["features"])
            self.assertIn("episode_index", info["features"])
            self.assertIn("index", info["features"])
            self.assertIn("task_index", info["features"])
            self.assertIn(
                "annotation.human.action.task_description", info["features"]
            )
            self.assertIn("observation.images.ego_view", info["features"])

            modality = json.loads(
                (root / "meta/modality.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(modality["observation"]["state"]), {"joint_position"}
            )
            self.assertEqual(set(modality["action"]), {"joint_position"})

    def test_policy_field_selection_writes_network_target_and_state_inputs(self):
        parquet_rows = {}

        def write_fake_parquet(path, rows):
            path.parent.mkdir(parents=True, exist_ok=True)
            parquet_rows[path.name] = rows
            path.write_text(json.dumps(rows), encoding="utf-8")

        selection = FieldSelection(
            target=("aligned_target_pos",),
            state=(
                "relative_ori_6d",
                "motion_anchor_lin_vel_b",
                "motion_anchor_ang_vel_b",
                "ang_vel_history",
                "gravity_history",
                "joint_pos_rel_history",
                "joint_vel_history",
                "action_history",
            ),
        )
        with TemporaryDirectory() as tmp:
            writer = _writer(
                tmp,
                parquet_writer=write_fake_parquet,
                field_selection=selection,
            )
            writer.start_episode("policy fields")
            writer.add_frame(_robot_frame(), FakeFrame())

            writer.save_episode()

            root = Path(tmp) / "dataset"
            row = parquet_rows["train-000000.parquet"][0]
            robot_columns = {
                key
                for key in row
                if key.startswith("action.") or key.startswith("observation.state.")
            }
            self.assertEqual(
                robot_columns,
                {
                    "action.aligned_target_pos",
                    "observation.state.relative_ori_6d",
                    "observation.state.motion_anchor_lin_vel_b",
                    "observation.state.motion_anchor_ang_vel_b",
                    "observation.state.ang_vel_history",
                    "observation.state.gravity_history",
                    "observation.state.joint_pos_rel_history",
                    "observation.state.joint_vel_history",
                    "observation.state.action_history",
                },
            )
            self.assertEqual(len(row["action.aligned_target_pos"]), 45)
            self.assertEqual(len(row["observation.state.relative_ori_6d"]), 90)
            self.assertEqual(len(row["observation.state.action_history"]), 290)

            info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
            self.assertEqual(
                info["features"]["action.aligned_target_pos"]["shape"], [45]
            )
            self.assertEqual(
                info["features"]["observation.state.relative_ori_6d"]["shape"], [90]
            )
            self.assertEqual(
                info["features"]["observation.state.action_history"]["shape"], [290]
            )

            modality = json.loads(
                (root / "meta/modality.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(modality["action"]), {"aligned_target_pos"})
            self.assertEqual(set(modality["observation"]["state"]), set(selection.state))

    def test_selected_aligned_target_pos_is_required_before_video_write(self):
        selection = FieldSelection(
            target=("aligned_target_pos",),
            state=("relative_ori_6d",),
        )
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp, field_selection=selection)
            writer.start_episode("missing aligned target")

            with self.assertRaisesRegex(
                ValueError, "action.aligned_target_pos has dimension 0"
            ):
                writer.add_frame(
                    _robot_frame(
                        aligned_target_pos=[],
                        policy_state=_policy_state_fields(),
                    ),
                    FakeFrame(),
                )

            root = Path(tmp) / "dataset"
            self.assertFalse(
                (
                    root
                    / "videos/observation.images.ego_view/episode_000000.mp4"
                ).exists()
            )
            self.assertEqual(writer.active_frame_count, 0)

    def test_selected_policy_state_field_is_required_before_video_write(self):
        selection = FieldSelection(
            target=("aligned_target_pos",),
            state=("relative_ori_6d", "action_history"),
        )
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp, field_selection=selection)
            writer.start_episode("missing policy state")

            with self.assertRaisesRegex(
                ValueError,
                "selected field observation.state.action_history is missing",
            ):
                writer.add_frame(
                    _robot_frame(policy_state={"relative_ori_6d": [0.1] * 90}),
                    FakeFrame(),
                )

            root = Path(tmp) / "dataset"
            self.assertFalse(
                (
                    root
                    / "videos/observation.images.ego_view/episode_000000.mp4"
                ).exists()
            )
            self.assertEqual(writer.active_frame_count, 0)

    def test_existing_dataset_rejects_subset_field_selection_after_default_schema(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("full fields")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.save_episode()

            with self.assertRaisesRegex(
                ValueError, "field selection does not match existing dataset"
            ):
                _writer(tmp, field_selection=_joint_position_only_selection())

    def test_existing_dataset_rejects_default_field_selection_after_subset_schema(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(
                tmp, field_selection=_joint_position_only_selection()
            )
            writer.start_episode("subset fields")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.save_episode()

            with self.assertRaisesRegex(
                ValueError, "field selection does not match existing dataset"
            ):
                _writer(tmp)

    def test_discard_does_not_keep_episode_files_or_metadata(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("discard this")
            writer.add_frame(_robot_frame(), FakeFrame())

            writer.discard_episode()

            root = Path(tmp) / "dataset"
            self.assertFalse((root / "meta/episodes.jsonl").exists())
            self.assertFalse(
                (
                    root
                    / "videos/observation.images.ego_view/episode_000000.mp4"
                ).exists()
            )

    def test_metadata_failure_does_not_commit_and_discard_removes_partials(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(
                tmp,
                camera_keys=[
                    "observation.images.head",
                    "observation.images.ego_view",
                ],
            )
            writer.start_episode("partial failure")
            writer.add_frame(
                _robot_frame(),
                {"head": FakeFrame(), "ego_view": FakeFrame()},
            )

            def fail_metadata(active, **kwargs):
                raise RuntimeError("metadata write failed")

            writer._write_metadata = fail_metadata

            with self.assertRaisesRegex(RuntimeError, "metadata write failed"):
                writer.save_episode()

            root = Path(tmp) / "dataset"
            self.assertEqual(writer._episodes, [])
            self.assertEqual(writer._total_frames, 0)
            self.assertTrue((root / "data/train-000000.parquet").exists())
            self.assertTrue(
                (
                    root
                    / "videos/observation.images.head/episode_000000.mp4"
                ).exists()
            )
            self.assertTrue(
                (
                    root
                    / "videos/observation.images.ego_view/episode_000000.mp4"
                ).exists()
            )

            writer.discard_episode()

            self.assertFalse((root / "data/train-000000.parquet").exists())
            self.assertFalse(
                (
                    root
                    / "videos/observation.images.head/episode_000000.mp4"
                ).exists()
            )
            self.assertFalse(
                (
                    root
                    / "videos/observation.images.ego_view/episode_000000.mp4"
                ).exists()
            )
            self.assertEqual(writer._episodes, [])
            self.assertEqual(writer._total_frames, 0)

    def test_half_written_metadata_is_rolled_back(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(
                tmp,
                camera_keys=[
                    "observation.images.head",
                    "observation.images.ego_view",
                ],
            )
            writer.start_episode("partial metadata")
            writer.add_frame(
                _robot_frame(),
                {"head": FakeFrame(), "ego_view": FakeFrame()},
            )
            original_replace_path = lerobot_dataset._replace_path

            def fail_on_info_replace(source, target):
                if target.name == "info.json":
                    raise RuntimeError("info replace failed")
                original_replace_path(source, target)

            lerobot_dataset._replace_path = fail_on_info_replace
            try:
                with self.assertRaisesRegex(RuntimeError, "info replace failed"):
                    writer.save_episode()
            finally:
                lerobot_dataset._replace_path = original_replace_path

            root = Path(tmp) / "dataset"
            meta_dir = root / "meta"
            self.assertFalse((meta_dir / "tasks.jsonl").exists())
            self.assertFalse((meta_dir / "episodes.jsonl").exists())
            self.assertFalse((meta_dir / "info.json").exists())
            self.assertFalse((meta_dir / "modality.json").exists())
            self.assertEqual(list(meta_dir.glob(".*.tmp")), [])
            self.assertEqual(list(meta_dir.glob(".*.bak")), [])

            writer.discard_episode()

            self.assertFalse((root / "data/train-000000.parquet").exists())
            self.assertFalse(
                (
                    root
                    / "videos/observation.images.head/episode_000000.mp4"
                ).exists()
            )
            self.assertFalse(
                (
                    root
                    / "videos/observation.images.ego_view/episode_000000.mp4"
                ).exists()
            )

    def test_video_write_failure_marks_episode_failed_until_discard(self):
        def sink_factory(path, fps, frame_size):
            if "observation.images.ego_view" in str(path):
                return FailingVideoSink(path, fps, frame_size)
            return FakeVideoSink(path, fps, frame_size)

        with TemporaryDirectory() as tmp:
            writer = LeRobotV21Writer(
                tmp,
                dataset_name="dataset",
                fps=50,
                camera_keys=[
                    "observation.images.head",
                    "observation.images.ego_view",
                ],
                parquet_writer=_write_fake_parquet,
                video_sink_factory=sink_factory,
            )
            writer.start_episode("write failure")

            with self.assertRaisesRegex(
                RuntimeError, "video write failed for observation.images.ego_view"
            ):
                writer.add_frame(
                    _robot_frame(),
                    {"head": FakeFrame(), "ego_view": FakeFrame()},
                )

            self.assertEqual(writer.active_frame_count, 0)
            with self.assertRaisesRegex(RuntimeError, "cannot save failed episode"):
                writer.save_episode()

            root = Path(tmp) / "dataset"
            writer.discard_episode()

            self.assertFalse((root / "data/train-000000.parquet").exists())
            self.assertFalse(
                (
                    root
                    / "videos/observation.images.head/episode_000000.mp4"
                ).exists()
            )
            self.assertFalse(
                (
                    root
                    / "videos/observation.images.ego_view/episode_000000.mp4"
                ).exists()
            )

    def test_metadata_failure_can_retry_without_duplicate_episode(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("retry save")
            writer.add_frame(_robot_frame(), FakeFrame())

            original_write_metadata = writer._write_metadata
            attempts = {"count": 0}

            def fail_once(active, **kwargs):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("metadata write failed")
                return original_write_metadata(active, **kwargs)

            writer._write_metadata = fail_once

            with self.assertRaisesRegex(RuntimeError, "metadata write failed"):
                writer.save_episode()

            result = writer.save_episode()

            self.assertTrue(result.saved)
            self.assertEqual(result.frame_count, 1)
            root = Path(tmp) / "dataset"
            episodes = [
                json.loads(line)
                for line in (root / "meta/episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(episodes), 1)
            info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(info["total_frames"], 1)

    def test_modality_uses_configured_camera_stream_name(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp, camera_key="observation.images.front")
            writer.start_episode("front camera")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.save_episode()

            root = Path(tmp) / "dataset"
            modality = json.loads(
                (root / "meta/modality.json").read_text(encoding="utf-8")
            )

            self.assertIn("front", modality["observation"]["images"])
            self.assertNotIn("ego_view", modality["observation"]["images"])
            self.assertEqual(
                modality["observation"]["images"]["front"]["key"],
                "observation.images.front",
            )


def _writer(
    tmp,
    parquet_writer=None,
    camera_key="observation.images.ego_view",
    camera_keys=None,
    field_selection=None,
):
    return LeRobotV21Writer(
        tmp,
        dataset_name="dataset",
        fps=50,
        camera_key=camera_key,
        camera_keys=camera_keys,
        field_selection=field_selection,
        parquet_writer=parquet_writer or _write_fake_parquet,
        video_sink_factory=FakeVideoSink,
    )


def _write_fake_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _joint_position_only_selection():
    return FieldSelection(
        target=("joint_position",),
        state=("joint_position",),
    )


def _policy_state_fields():
    return {
        "relative_ori_6d": [0.1] * 90,
        "motion_anchor_lin_vel_b": [0.2] * 45,
        "motion_anchor_ang_vel_b": [0.3] * 45,
        "ang_vel_history": [0.4] * 30,
        "gravity_history": [0.5] * 30,
        "joint_pos_rel_history": [0.6] * 290,
        "joint_vel_history": [0.7] * 290,
        "action_history": [0.8] * 290,
    }


def _robot_frame(aligned_target_pos=None, policy_state=None):
    return RobotFrame(
        joint_position=[1.0] * DOF,
        joint_velocity=[2.0] * DOF,
        joint_torque=[3.0] * DOF,
        imu_angular_velocity=[0.1, 0.2, 0.3],
        imu_linear_acceleration=[0.0, 0.0, 9.8],
        projected_gravity_or_quat=[0.0, 0.0, 0.0, 1.0],
        target_joint_pos=[4.0] * DOF,
        policy_action=[5.0] * DOF,
        aligned_target_pos=(
            [6.0] * 45 if aligned_target_pos is None else aligned_target_pos
        ),
        policy_state=_policy_state_fields() if policy_state is None else policy_state,
        joint_names=[f"j{i}" for i in range(DOF)],
    )


if __name__ == "__main__":
    unittest.main()
