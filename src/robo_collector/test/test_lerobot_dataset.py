import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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
            self.assertEqual(
                info["features"]["observation.images.ego_view"]["shape"], [4, 6, 3]
            )

            task = json.loads(
                (root / "meta/tasks.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(task, {"task_index": 0, "task": "pick the red cup"})

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
            writer = _writer(tmp)
            writer.start_episode("partial failure")
            writer.add_frame(_robot_frame(), FakeFrame())

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
                    / "videos/observation.images.ego_view/episode_000000.mp4"
                ).exists()
            )

            writer.discard_episode()

            self.assertFalse((root / "data/train-000000.parquet").exists())
            self.assertFalse(
                (
                    root
                    / "videos/observation.images.ego_view/episode_000000.mp4"
                ).exists()
            )
            self.assertEqual(writer._episodes, [])
            self.assertEqual(writer._total_frames, 0)

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


def _writer(tmp, parquet_writer=None, camera_key="observation.images.ego_view"):
    return LeRobotV21Writer(
        tmp,
        dataset_name="dataset",
        fps=50,
        camera_key=camera_key,
        parquet_writer=parquet_writer or _write_fake_parquet,
        video_sink_factory=FakeVideoSink,
    )


def _write_fake_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _robot_frame():
    return RobotFrame(
        joint_position=[1.0] * DOF,
        joint_velocity=[2.0] * DOF,
        joint_torque=[3.0] * DOF,
        imu_angular_velocity=[0.1, 0.2, 0.3],
        imu_linear_acceleration=[0.0, 0.0, 9.8],
        projected_gravity_or_quat=[0.0, 0.0, 0.0, 1.0],
        target_joint_pos=[4.0] * DOF,
        policy_action=[5.0] * DOF,
        joint_names=[f"j{i}" for i in range(DOF)],
    )


if __name__ == "__main__":
    unittest.main()
