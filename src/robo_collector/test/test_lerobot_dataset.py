import json
import unittest
from dataclasses import replace
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


class DifferentShapeFrame:
    shape = (5, 7, 3)


class LeRobotV21WriterTest(unittest.TestCase):
    def test_idle_writer_does_not_create_dataset(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)

            self.assertFalse(writer.root.exists())

    def test_writer_requires_integer_fps(self):
        with TemporaryDirectory() as tmp:
            for fps in (True, 0, 30.5):
                with self.subTest(fps=fps):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        LeRobotV21Writer(tmp, dataset_name="dataset", fps=fps)

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
            self.assertEqual(info["robo_collector_schema_version"], 1)
            self.assertEqual(info["timeline_semantics"], "fixed_rate_v1")
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

    def test_fixed_fps_timeline_keeps_source_timestamps_for_audit(self):
        parquet_rows = {}

        def write_fake_parquet(path, rows):
            path.parent.mkdir(parents=True, exist_ok=True)
            parquet_rows[path.name] = rows
            path.write_text(json.dumps(rows), encoding="utf-8")

        with TemporaryDirectory() as tmp:
            writer = _writer(tmp, parquet_writer=write_fake_parquet)
            writer.start_episode("timed")
            writer.add_frame(
                replace(_robot_frame(), state_timestamp_sec=100.00),
                FakeFrame(),
                camera_timestamps_sec={"ego_view": 99.98},
            )
            writer.add_frame(
                replace(_robot_frame(), state_timestamp_sec=100.12),
                FakeFrame(),
                camera_timestamps_sec={
                    "observation.images.ego_view": 100.10
                },
            )

            writer.save_episode()

            rows = parquet_rows["train-000000.parquet"]
            self.assertAlmostEqual(rows[0]["timestamp"], 0.0)
            self.assertAlmostEqual(
                rows[0]["observation.images.ego_view"]["timestamp"], 0.0
            )
            self.assertAlmostEqual(rows[1]["timestamp"], 1 / 50)
            self.assertAlmostEqual(
                rows[1]["observation.images.ego_view"]["timestamp"], 1 / 50
            )
            self.assertEqual(rows[0]["source_timestamp.state"], 100.0)
            self.assertEqual(rows[1]["source_timestamp.state"], 100.12)
            self.assertEqual(
                rows[0]["source_timestamp.camera.ego_view"], 99.98
            )
            self.assertEqual(
                rows[1]["source_timestamp.camera.ego_view"], 100.10
            )

    def test_source_timestamps_must_be_finite_and_monotonic(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("timed")

            with self.assertRaisesRegex(
                ValueError, "state_timestamp_sec must be finite"
            ):
                writer.add_frame(
                    replace(_robot_frame(), state_timestamp_sec=float("nan")),
                    FakeFrame(),
                )

            writer.add_frame(
                replace(_robot_frame(), state_timestamp_sec=100.0),
                FakeFrame(),
                camera_timestamps_sec={"ego_view": 100.0},
            )
            with self.assertRaisesRegex(ValueError, "state timestamp moved backwards"):
                writer.add_frame(
                    replace(_robot_frame(), state_timestamp_sec=99.0),
                    FakeFrame(),
                    camera_timestamps_sec={"ego_view": 100.1},
                )
            writer.discard_episode()

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

    def test_selected_numeric_values_must_be_finite_before_video_write(self):
        with TemporaryDirectory() as tmp:
            for invalid_value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(invalid_value=invalid_value):
                    writer = _writer(
                        tmp, dataset_name=f"dataset-{str(invalid_value)}"
                    )
                    writer.start_episode("invalid numeric value")
                    frame = _robot_frame()
                    joint_position = list(frame.joint_position)
                    joint_position[7] = invalid_value

                    with self.assertRaisesRegex(
                        ValueError,
                        r"observation\.state\.joint_position\[7\] must be finite",
                    ):
                        writer.add_frame(
                            replace(frame, joint_position=joint_position),
                            FakeFrame(),
                        )

                    self.assertEqual(writer.active_frame_count, 0)
                    self.assertFalse(writer.root.exists())

    def test_joint_names_cannot_change_within_or_across_episodes(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("canonical order")
            writer.add_frame(_robot_frame(), FakeFrame())

            with self.assertRaisesRegex(
                ValueError, "joint_names changed from the dataset canonical ordering"
            ):
                writer.add_frame(
                    replace(
                        _robot_frame(),
                        joint_names=list(reversed(_robot_frame().joint_names)),
                    ),
                    FakeFrame(),
                )

            self.assertEqual(writer.active_frame_count, 1)
            writer.save_episode()

            appended_writer = _writer(tmp)
            appended_writer.start_episode("changed order")
            with self.assertRaisesRegex(
                ValueError, "joint_names changed from the dataset canonical ordering"
            ):
                appended_writer.add_frame(
                    replace(
                        _robot_frame(),
                        joint_names=list(reversed(_robot_frame().joint_names)),
                    ),
                    FakeFrame(),
                )

    def test_existing_dataset_rejects_camera_shape_change(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("original camera")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.save_episode()

            appended_writer = _writer(tmp)
            appended_writer.start_episode("different camera shape")
            with self.assertRaisesRegex(
                ValueError, "camera shape does not match existing dataset"
            ):
                appended_writer.add_frame(
                    _robot_frame(), DifferentShapeFrame()
                )

            self.assertEqual(appended_writer.active_frame_count, 0)
            self.assertFalse(
                (
                    appended_writer.root
                    / "videos/observation.images.ego_view/episode_000001.mp4"
                ).exists()
            )

    def test_existing_dataset_rejects_camera_keys_fps_and_robot_type_changes(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("schema")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.save_episode()

            with self.assertRaisesRegex(ValueError, "camera keys do not match"):
                _writer(tmp, camera_key="observation.images.front")
            with self.assertRaisesRegex(ValueError, "fps does not match"):
                LeRobotV21Writer(
                    tmp,
                    dataset_name="dataset",
                    fps=30,
                    parquet_writer=_write_fake_parquet,
                    video_sink_factory=FakeVideoSink,
                )
            with self.assertRaisesRegex(ValueError, "robot_type does not match"):
                LeRobotV21Writer(
                    tmp,
                    dataset_name="dataset",
                    fps=50,
                    robot_type="different_robot",
                    parquet_writer=_write_fake_parquet,
                    video_sink_factory=FakeVideoSink,
                )

    def test_existing_dataset_rejects_robot_feature_shape_change(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("schema")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.save_episode()

            info_path = writer.root / "meta/info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["features"]["action.joint_position"]["shape"] = [DOF - 1]
            info_path.write_text(json.dumps(info), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "action.joint_position shape mismatch"
            ):
                _writer(tmp)

    def test_legacy_dataset_requires_explicit_timeline_migration(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("legacy episode")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.save_episode()

            info_path = writer.root / "meta/info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            for key in list(info["features"]):
                if key.startswith("source_timestamp."):
                    del info["features"][key]
            info_path.write_text(json.dumps(info), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "predates the fixed-rate timeline"
            ):
                _writer(tmp)

    def test_existing_dataset_rejects_partial_source_timestamp_schema(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("timestamp schema")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.save_episode()

            info_path = writer.root / "meta/info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            del info["features"]["source_timestamp.camera.ego_view"]
            info_path.write_text(json.dumps(info), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "source timestamp features do not match"
            ):
                _writer(tmp)

    def test_default_writer_spools_rows_and_streams_parquet(self):
        calls = []
        original_stream_writer = lerobot_dataset.write_parquet_pyarrow_stream
        original_row_count_validator = lerobot_dataset._validate_parquet_row_count

        def fake_stream_writer(path, row_spool_path, *, batch_size):
            rows = [
                json.loads(line)
                for line in row_spool_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            calls.append((len(rows), batch_size))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows), encoding="utf-8")

        lerobot_dataset.write_parquet_pyarrow_stream = fake_stream_writer
        lerobot_dataset._validate_parquet_row_count = lambda path, count: None
        try:
            with TemporaryDirectory() as tmp:
                writer = LeRobotV21Writer(
                    tmp,
                    dataset_name="dataset",
                    fps=50,
                    video_sink_factory=FakeVideoSink,
                )
                writer.start_episode("spooled")
                for _ in range(300):
                    writer.add_frame(_robot_frame(), FakeFrame())

                active = writer._active
                self.assertIsNotNone(active)
                self.assertFalse(hasattr(active, "rows"))
                self.assertEqual(writer.active_frame_count, 300)
                self.assertTrue(
                    writer._root_path(active.row_spool_rel_path).exists()
                )

                result = writer.save_episode()

                self.assertTrue(result.saved)
                self.assertEqual(calls, [(300, 256)])
                self.assertFalse((writer.root / ".inprogress").exists())
        finally:
            lerobot_dataset.write_parquet_pyarrow_stream = original_stream_writer
            lerobot_dataset._validate_parquet_row_count = original_row_count_validator

    def test_default_parquet_schema_matches_declared_feature_dtypes(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")

        with TemporaryDirectory() as tmp:
            writer = LeRobotV21Writer(
                tmp,
                dataset_name="dataset",
                fps=50,
                video_sink_factory=FakeVideoSink,
            )
            writer.start_episode("typed")
            writer.add_frame(
                replace(_robot_frame(), state_timestamp_sec=10.0),
                FakeFrame(),
                camera_timestamps_sec={"ego_view": 9.99},
            )
            result = writer.save_episode()

            table = pq.read_table(result.data_path)
            schema = table.schema
            self.assertEqual(schema.field("timestamp").type, pa.float32())
            self.assertEqual(
                schema.field("source_timestamp.state").type, pa.float64()
            )
            self.assertEqual(
                schema.field("source_timestamp.camera.ego_view").type,
                pa.float64(),
            )
            robot_type = schema.field("action.joint_position").type
            self.assertTrue(pa.types.is_list(robot_type))
            self.assertEqual(robot_type.value_type, pa.float32())
            video_type = schema.field("observation.images.ego_view").type
            self.assertEqual(video_type.field("timestamp").type, pa.float32())

    def test_episode_metadata_records_artifact_integrity(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("integrity")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.save_episode()

            episode = json.loads(
                (writer.root / "meta/episodes.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            integrity = episode["integrity"]
            self.assertEqual(integrity["algorithm"], "sha256")
            self.assertEqual(integrity["data"]["rows"], 1)
            self.assertEqual(
                integrity["videos"]["observation.images.ego_view"]["frames"],
                1,
            )
            self.assertEqual(len(integrity["data"]["sha256"]), 64)

    def test_committed_artifacts_are_validated_before_append(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("integrity")
            writer.add_frame(_robot_frame(), FakeFrame())
            result = writer.save_episode()
            result.video_path.unlink()

            restarted = _writer(tmp)
            with self.assertRaisesRegex(RuntimeError, "committed video.*missing"):
                restarted.start_episode("must not append")

    def test_same_size_artifact_corruption_is_rejected_before_append(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("integrity")
            writer.add_frame(_robot_frame(), FakeFrame())
            result = writer.save_episode()
            corrupted = bytearray(result.video_path.read_bytes())
            corrupted[0] ^= 0x01
            result.video_path.write_bytes(corrupted)

            restarted = _writer(tmp)
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                restarted.start_episode("must not append")

    def test_incomplete_metadata_bundle_is_rejected(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("metadata")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.save_episode()
            (writer.root / "meta/info.json").unlink()

            with self.assertRaisesRegex(RuntimeError, "metadata is incomplete"):
                _writer(tmp)

    def test_committed_manifest_rolls_forward_missing_final_artifact(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("roll forward")
            writer.add_frame(_robot_frame(), FakeFrame())
            original_cleanup = writer._cleanup_active_staging

            def leave_manifest(active, *, remove_committed):
                raise OSError("simulated cleanup interruption")

            writer._cleanup_active_staging = leave_manifest
            result = writer.save_episode()
            self.assertTrue(result.saved)
            manifest_path = next((writer.root / ".inprogress").glob("*/manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            staged_data = writer._root_path(manifest["staged_data_path"])
            staged_data.parent.mkdir(parents=True, exist_ok=True)
            result.data_path.replace(staged_data)
            writer._cleanup_active_staging = original_cleanup

            restarted = _writer(tmp)
            self.assertEqual(restarted.start_episode("next"), 1)
            self.assertTrue(result.data_path.exists())
            self.assertFalse((writer.root / ".inprogress").exists())
            restarted.discard_episode()

    def test_post_commit_staging_cleanup_failure_still_reports_saved(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("cleanup")
            writer.add_frame(_robot_frame(), FakeFrame())

            def fail_cleanup(active, *, remove_committed):
                raise OSError("simulated staging cleanup failure")

            writer._cleanup_active_staging = fail_cleanup
            result = writer.save_episode()

            self.assertTrue(result.saved)
            self.assertTrue(result.data_path.exists())
            restarted = _writer(tmp)
            self.assertEqual(restarted.start_episode("next"), 1)
            self.assertFalse((writer.root / ".inprogress").exists())
            restarted.discard_episode()

    def test_collision_discard_preserves_preexisting_final_artifacts(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("collision")
            writer.add_frame(_robot_frame(), FakeFrame())
            data_path = writer.root / "data/train-000000.parquet"
            video_path = (
                writer.root
                / "videos/observation.images.ego_view/episode_000000.mp4"
            )
            data_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_bytes(b"existing-data")
            video_path.write_bytes(b"existing-video")

            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                writer.save_episode()
            writer.discard_episode()

            self.assertEqual(data_path.read_bytes(), b"existing-data")
            self.assertEqual(video_path.read_bytes(), b"existing-video")

    def test_post_rename_failure_discard_removes_owned_final_artifact(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("rename interruption")
            writer.add_frame(_robot_frame(), FakeFrame())
            original_replace = lerobot_dataset._replace_path_durable
            calls = 0

            def replace_then_fail(source, target):
                nonlocal calls
                calls += 1
                if calls == 1:
                    lerobot_dataset._replace_path(source, target)
                    raise OSError("directory fsync failed after rename")
                original_replace(source, target)

            lerobot_dataset._replace_path_durable = replace_then_fail
            try:
                with self.assertRaisesRegex(OSError, "after rename"):
                    writer.save_episode()
            finally:
                lerobot_dataset._replace_path_durable = original_replace

            final_data = writer.root / "data/train-000000.parquet"
            self.assertTrue(final_data.exists())
            writer.discard_episode()
            self.assertFalse(final_data.exists())
            self.assertFalse((writer.root / ".inprogress").exists())

            restarted = _writer(tmp)
            self.assertEqual(restarted.start_episode("retry"), 0)
            restarted.discard_episode()

    def test_integrity_cache_only_hashes_new_episode_on_next_start(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("first")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.save_episode()
            writer.start_episode("second")
            writer.add_frame(_robot_frame(), FakeFrame())
            writer.save_episode()

            original_integrity = lerobot_dataset._file_integrity
            hashed_paths = []

            def count_integrity(path):
                hashed_paths.append(path)
                return original_integrity(path)

            lerobot_dataset._file_integrity = count_integrity
            try:
                self.assertEqual(writer.start_episode("third"), 2)
            finally:
                lerobot_dataset._file_integrity = original_integrity

            self.assertEqual(len(hashed_paths), 2)
            writer.discard_episode()

    def test_orphan_recovery_preserves_unowned_collision_artifacts(self):
        with TemporaryDirectory() as tmp:
            abandoned = _writer(tmp)
            abandoned.start_episode("collision")
            abandoned.add_frame(_robot_frame(), FakeFrame())
            data_path = abandoned.root / "data/train-000000.parquet"
            video_path = (
                abandoned.root
                / "videos/observation.images.ego_view/episode_000000.mp4"
            )
            data_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_bytes(b"existing-data")
            video_path.write_bytes(b"existing-video")

            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                abandoned.save_episode()
            abandoned._release_dataset_lock()

            restarted = _writer(tmp)
            self.assertEqual(restarted.start_episode("recovered"), 0)
            self.assertEqual(data_path.read_bytes(), b"existing-data")
            self.assertEqual(video_path.read_bytes(), b"existing-video")
            restarted.discard_episode()

    def test_discard_cleanup_failure_retains_active_episode_for_retry(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("retry discard")
            writer.add_frame(_robot_frame(), FakeFrame())
            original_cleanup = writer._cleanup_active_staging
            calls = 0

            def fail_once(active, *, remove_committed):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError("temporary unlink failure")
                return original_cleanup(active, remove_committed=remove_committed)

            writer._cleanup_active_staging = fail_once
            with self.assertRaisesRegex(OSError, "temporary unlink failure"):
                writer.discard_episode()
            self.assertIsNotNone(writer.active_episode_index)

            writer.discard_episode()
            self.assertIsNone(writer.active_episode_index)
            self.assertFalse((writer.root / ".inprogress").exists())

    def test_restart_removes_uncommitted_episode_artifacts(self):
        with TemporaryDirectory() as tmp:
            abandoned_writer = _writer(tmp)
            abandoned_writer.start_episode("abandoned")
            abandoned_writer.add_frame(_robot_frame(), FakeFrame())
            abandoned_writer._close_row_spool(abandoned_writer._active)
            for sink in abandoned_writer._active.video_sinks.values():
                sink.close()

            self.assertTrue(
                (abandoned_writer.root / ".inprogress").exists()
            )

            abandoned_writer._release_dataset_lock()
            recovered_writer = _writer(tmp)

            self.assertEqual(recovered_writer.start_episode("recovered"), 0)
            self.assertFalse((recovered_writer.root / ".inprogress").exists())
            recovered_writer.discard_episode()

    def test_active_episode_excludes_concurrent_writer_and_reload_prevents_collision(self):
        with TemporaryDirectory() as tmp:
            first = _writer(tmp)
            second = _writer(tmp)
            self.assertEqual(first.start_episode("first"), 0)
            first.add_frame(_robot_frame(), FakeFrame())

            with self.assertRaisesRegex(RuntimeError, "locked by another active writer"):
                second.start_episode("second")

            first.save_episode()
            self.assertEqual(second.start_episode("second"), 1)
            second.add_frame(_robot_frame(), FakeFrame())
            second.save_episode()

            episodes = [
                json.loads(line)
                for line in (second.root / "meta/episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [episode["episode_index"] for episode in episodes], [0, 1]
            )

    def test_dataset_and_camera_paths_cannot_escape_root(self):
        with TemporaryDirectory() as tmp:
            for dataset_name in ("../outside", "/tmp/outside"):
                with self.subTest(dataset_name=dataset_name):
                    with self.assertRaisesRegex(ValueError, "dataset_name"):
                        LeRobotV21Writer(
                            tmp,
                            dataset_name=dataset_name,
                            parquet_writer=_write_fake_parquet,
                            video_sink_factory=FakeVideoSink,
                        )

            with self.assertRaisesRegex(ValueError, "camera key is not safe"):
                LeRobotV21Writer(
                    tmp,
                    dataset_name="dataset",
                    camera_key="../outside",
                    parquet_writer=_write_fake_parquet,
                    video_sink_factory=FakeVideoSink,
                )

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

    def test_restart_recovers_interrupted_metadata_transaction(self):
        with TemporaryDirectory() as tmp:
            abandoned_writer = _writer(tmp)
            abandoned_writer.start_episode("interrupted metadata")
            abandoned_writer.add_frame(_robot_frame(), FakeFrame())
            original_replace_path = lerobot_dataset._replace_path

            def interrupt_on_info_replace(source, target):
                if target.name == "info.json":
                    raise KeyboardInterrupt("simulated power loss")
                original_replace_path(source, target)

            lerobot_dataset._replace_path = interrupt_on_info_replace
            try:
                with self.assertRaisesRegex(KeyboardInterrupt, "power loss"):
                    abandoned_writer.save_episode()
            finally:
                lerobot_dataset._replace_path = original_replace_path
                abandoned_writer._release_dataset_lock()

            meta_dir = abandoned_writer.root / "meta"
            self.assertTrue(
                (meta_dir / lerobot_dataset.METADATA_TRANSACTION_FILENAME).exists()
            )

            recovered_writer = _writer(tmp)
            self.assertEqual(recovered_writer.start_episode("recovered"), 0)
            self.assertFalse(
                (meta_dir / lerobot_dataset.METADATA_TRANSACTION_FILENAME).exists()
            )
            self.assertEqual(list(meta_dir.glob(".*.tmp")), [])
            self.assertEqual(list(meta_dir.glob(".*.bak")), [])
            recovered_writer.discard_episode()

    def test_post_commit_metadata_cleanup_failure_rolls_forward(self):
        with TemporaryDirectory() as tmp:
            writer = _writer(tmp)
            writer.start_episode("committed metadata")
            writer.add_frame(_robot_frame(), FakeFrame())
            original_cleanup = lerobot_dataset._cleanup_transaction_files

            def fail_cleanup(parent_dir, entries):
                raise OSError("simulated backup cleanup failure")

            lerobot_dataset._cleanup_transaction_files = fail_cleanup
            try:
                result = writer.save_episode()
            finally:
                lerobot_dataset._cleanup_transaction_files = original_cleanup

            self.assertTrue(result.saved)
            journal = (
                writer.root
                / "meta"
                / lerobot_dataset.METADATA_TRANSACTION_FILENAME
            )
            self.assertTrue(journal.exists())
            self.assertEqual(
                json.loads(journal.read_text(encoding="utf-8"))["phase"],
                "committed",
            )

            restarted = _writer(tmp)
            self.assertEqual(restarted.start_episode("next"), 1)
            self.assertFalse(journal.exists())
            restarted.discard_episode()

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
    dataset_name="dataset",
):
    return LeRobotV21Writer(
        tmp,
        dataset_name=dataset_name,
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
