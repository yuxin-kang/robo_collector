import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from robo_collector.field_config import FieldSelection
from robo_collector.gr00t_converter import POLICY_STATE_DIM, convert_dataset, main
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


class Gr00tConverterTest(unittest.TestCase):
    def test_convert_dataset_writes_gr00t_layout_and_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(source_root, dataset_name="source_dataset")

            result = convert_dataset(
                source_root,
                "source_dataset",
                dest_root,
                output_name="converted_dataset",
                action_source="aligned_target_pos",
            )

            output_root = dest_root / "converted_dataset"
            self.assertEqual(result.output_dataset, output_root)
            self.assertTrue(
                (output_root / "data/chunk-000/episode_000000.parquet").exists()
            )
            self.assertTrue(
                (
                    output_root
                    / "videos/chunk-000/observation.images.head/episode_000000.mp4"
                ).exists()
            )
            self.assertTrue(
                (
                    output_root
                    / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
                ).exists()
            )

            table = pq.read_table(output_root / "data/chunk-000/episode_000000.parquet")
            row = table.slice(0, 1).to_pylist()[0]
            self.assertEqual(set(row), {
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
                "annotation.human.action.task_description",
                "next.reward",
                "next.done",
                "observation.images.head",
                "observation.images.ego_view",
            })
            self.assertEqual(len(row["observation.state"]), POLICY_STATE_DIM)
            self.assertEqual(len(row["action"]), 45)
            self.assertEqual(row["annotation.human.action.task_description"], 0)
            self.assertEqual(row["task_index"], 0)
            self.assertFalse(row["next.done"])

            last_row = table.slice(table.num_rows - 1, 1).to_pylist()[0]
            self.assertTrue(last_row["next.done"])
            self.assertEqual(
                row["observation.images.head"]["path"],
                "videos/chunk-000/observation.images.head/episode_000000.mp4",
            )

            modality = json.loads(
                (output_root / "meta/modality.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                modality["state"]["relative_ori_6d"], {"start": 0, "end": 90}
            )
            self.assertEqual(
                modality["state"]["action_history"],
                {"start": 820, "end": 1110},
            )
            self.assertEqual(
                modality["action"], {"aligned_target_pos": {"start": 0, "end": 45}}
            )
            self.assertEqual(
                modality["video"]["head"], {"original_key": "observation.images.head"}
            )

            info = json.loads(
                (output_root / "meta/info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                info["data_path"], "data/chunk-000/episode_{episode_index:06d}.parquet"
            )
            self.assertEqual(info["features"]["observation.state"]["shape"], [1110])
            self.assertEqual(info["features"]["action"]["shape"], [45])
            self.assertEqual(
                info["features"]["annotation.human.action.task_description"]["dtype"],
                "int64",
            )
            schema = table.schema
            self.assertEqual(str(schema.field("observation.state").type), "list<element: float>")
            self.assertEqual(str(schema.field("action").type), "list<element: float>")
            self.assertEqual(str(schema.field("timestamp").type), "float")
            self.assertEqual(str(schema.field("next.reward").type), "float")

            tasks = [
                json.loads(line)
                for line in (output_root / "meta/tasks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(tasks, [{"task": "pick the red cup", "task_index": 0}])

    def test_convert_dataset_supports_policy_action_when_present(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(
                source_root,
                dataset_name="source_dataset",
                field_selection=_policy_selection(include_policy_action=True),
            )

            convert_dataset(
                source_root,
                "source_dataset",
                dest_root,
                action_source="policy_action",
            )

            output_root = dest_root / "source_dataset_gr00t"
            row = pq.read_table(
                output_root / "data/chunk-000/episode_000000.parquet"
            ).slice(0, 1).to_pylist()[0]
            self.assertEqual(len(row["action"]), DOF)
            self.assertEqual(row["action"], [5.0] * DOF)

            modality = json.loads(
                (output_root / "meta/modality.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                modality["action"], {"policy_action": {"start": 0, "end": 29}}
            )

    def test_convert_dataset_supports_joint_position_action(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(
                source_root,
                dataset_name="source_dataset",
                field_selection=FieldSelection(
                    target=("joint_position",),
                    state=tuple(_policy_state_fields()),
                ),
            )

            convert_dataset(
                source_root,
                "source_dataset",
                dest_root,
                action_source="joint_position",
            )

            output_root = dest_root / "source_dataset_gr00t"
            row = pq.read_table(
                output_root / "data/chunk-000/episode_000000.parquet"
            ).slice(0, 1).to_pylist()[0]
            self.assertEqual(len(row["action"]), DOF)
            self.assertEqual(row["action"], [4.0] * DOF)

    def test_convert_dataset_requires_full_policy_state(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(
                source_root,
                dataset_name="source_dataset",
                field_selection=FieldSelection(
                    target=("aligned_target_pos",),
                    state=(
                        "relative_ori_6d",
                        "motion_anchor_lin_vel_b",
                        "motion_anchor_ang_vel_b",
                    ),
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "missing required source state field: observation.state.ang_vel_history",
            ):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    action_source="aligned_target_pos",
                )
            self.assertFalse((dest_root / "source_dataset_gr00t").exists())

    def test_failed_conversion_does_not_leave_partial_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(
                source_root,
                dataset_name="source_dataset",
                field_selection=FieldSelection(
                    target=("aligned_target_pos",),
                    state=("relative_ori_6d",),
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "missing required source state field: observation.state.motion_anchor_lin_vel_b",
            ):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    output_name="converted_dataset",
                    action_source="aligned_target_pos",
                )
            self.assertFalse((dest_root / "converted_dataset").exists())
            self.assertEqual(list(dest_root.iterdir()), [])

    def test_convert_dataset_requires_selected_action_field(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(source_root, dataset_name="source_dataset")

            with self.assertRaisesRegex(
                ValueError,
                "missing required source action field: action.policy_action",
            ):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    action_source="policy_action",
                )
            self.assertFalse((dest_root / "source_dataset_gr00t").exists())

    def test_convert_dataset_rejects_path_traversal_names(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(source_root, dataset_name="source_dataset")

            with self.assertRaisesRegex(
                ValueError,
                "dataset_name must be a single folder name without path traversal",
            ):
                convert_dataset(
                    source_root,
                    "../source_dataset",
                    dest_root,
                    action_source="aligned_target_pos",
                )

            with self.assertRaisesRegex(
                ValueError,
                "output_name must be a single folder name without path traversal",
            ):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    output_name="../converted_dataset",
                    action_source="aligned_target_pos",
                )

    def test_convert_dataset_rejects_escaped_source_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            dataset_root = _create_source_dataset(source_root, dataset_name="source_dataset")

            episodes_path = dataset_root / "meta/episodes.jsonl"
            episodes = [
                json.loads(line)
                for line in episodes_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            episodes[0]["data_path"] = "../escape.parquet"
            episodes_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in episodes),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "episodes\\[0\\]\\.data_path escapes root via parent traversal",
            ):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    action_source="aligned_target_pos",
                )

    def test_convert_dataset_rejects_malicious_camera_key(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            dataset_root = _create_source_dataset(source_root, dataset_name="source_dataset")

            info_path = dataset_root / "meta/info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            feature = info["features"].pop("observation.images.head")
            malicious_key = "observation.images./../../../../escaped_camera"
            info["features"][malicious_key] = feature
            info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "camera key stream name must not contain path separators",
            ):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    action_source="aligned_target_pos",
                )
            self.assertFalse((dest_root / "source_dataset_gr00t").exists())
            self.assertFalse((root / "escaped_camera").exists())

    def test_main_runs_cli_arguments(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(source_root, dataset_name="source_dataset")

            exit_code = main(
                [
                    "--source-root",
                    str(source_root),
                    "--dataset-name",
                    "source_dataset",
                    "--dest-root",
                    str(dest_root),
                    "--output-name",
                    "converted_dataset",
                    "--action-source",
                    "aligned_target_pos",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(
                (
                    dest_root
                    / "converted_dataset/data/chunk-000/episode_000000.parquet"
                ).exists()
            )


def _create_source_dataset(
    source_root: Path,
    *,
    dataset_name: str,
    field_selection: FieldSelection | None = None,
) -> Path:
    writer = LeRobotV21Writer(
        source_root,
        dataset_name=dataset_name,
        fps=50,
        camera_keys=["observation.images.head", "observation.images.ego_view"],
        field_selection=field_selection or _policy_selection(),
        video_sink_factory=FakeVideoSink,
    )
    writer.start_episode("pick the red cup")
    writer.add_frame(
        _robot_frame(),
        {"head": FakeFrame(), "ego_view": FakeFrame()},
    )
    writer.add_frame(
        _robot_frame(),
        {"head": FakeFrame(), "ego_view": FakeFrame()},
    )
    writer.save_episode()
    return source_root / dataset_name


def _policy_selection(*, include_policy_action: bool = False) -> FieldSelection:
    return FieldSelection(
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
        include_policy_action=include_policy_action,
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
        aligned_target_pos=[6.0] * 45,
        policy_state=_policy_state_fields(),
        joint_names=[f"j{i}" for i in range(DOF)],
    )


if __name__ == "__main__":
    unittest.main()
