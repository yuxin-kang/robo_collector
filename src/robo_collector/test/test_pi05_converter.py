import io
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pyarrow.parquet as pq
from PIL import Image

from robo_collector.field_config import FieldSelection
from robo_collector.lerobot_dataset import DOF, LeRobotV21Writer, RobotFrame
from robo_collector.pi05_converter import (
    EGO_IMAGE_KEY,
    HEAD_IMAGE_KEY,
    convert_dataset,
    main,
    OpenCvFrameReader,
    _dest_data_rel_path,
    _open_frame_readers,
)


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


class FakeFrameReader:
    frame_count = 2
    shape = (4, 6, 3)

    def __init__(self, path: Path) -> None:
        self.path = path

    def read_rgb(self, index: int):
        import numpy as np

        if index < 0 or index >= self.frame_count:
            raise RuntimeError(f"bad frame index: {index}")
        base = 32 if "head" in str(self.path) else 96
        frame = np.zeros(self.shape, dtype=np.uint8)
        frame[..., 0] = base + index
        frame[..., 1] = base + index + 1
        frame[..., 2] = base + index + 2
        return frame

    def close(self):
        pass


class Pi05ConverterTest(unittest.TestCase):
    def test_convert_dataset_writes_openpi_compact_schema_and_metadata(self):
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
                frame_reader_factory=FakeFrameReader,
            )

            output_root = dest_root / "converted_dataset"
            self.assertEqual(result.output_dataset, output_root)
            data_path = output_root / "data/chunk-000/episode_000000.parquet"
            self.assertTrue(data_path.exists())
            self.assertTrue((output_root / "meta/info.json").exists())
            self.assertTrue((output_root / "meta/episodes_stats.jsonl").exists())

            table = pq.read_table(data_path)
            row = table.slice(0, 1).to_pylist()[0]
            self.assertEqual(
                set(row),
                {
                    "head_image",
                    "ego_image",
                    "state",
                    "actions",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                },
            )
            self.assertEqual(len(row["state"]), DOF)
            self.assertEqual(len(row["actions"]), DOF)
            self.assertEqual(row["state"], [1.0] * DOF)
            self.assertEqual(row["actions"], [5.0] * DOF)
            self.assertEqual(row["task_index"], 0)
            self.assertEqual(row[HEAD_IMAGE_KEY]["path"], "frame_000000.png")
            self.assertEqual(row[EGO_IMAGE_KEY]["path"], "frame_000000.png")
            self.assertEqual(_png_size(row[HEAD_IMAGE_KEY]["bytes"]), (6, 4))
            self.assertEqual(_png_size(row[EGO_IMAGE_KEY]["bytes"]), (6, 4))

            schema = table.schema
            self.assertEqual(
                str(schema.field(HEAD_IMAGE_KEY).type),
                "struct<bytes: binary, path: string>",
            )
            self.assertEqual(schema.field("state").type.list_size, DOF)
            self.assertEqual(schema.field("actions").type.list_size, DOF)
            hf_metadata = json.loads(schema.metadata[b"huggingface"].decode("utf-8"))
            self.assertEqual(
                hf_metadata["info"]["features"][HEAD_IMAGE_KEY], {"_type": "Image"}
            )
            self.assertEqual(
                hf_metadata["info"]["features"]["state"]["length"], DOF
            )

            info = json.loads(
                (output_root / "meta/info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(info["robot_type"], "unitree_g1")
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(info["total_frames"], 2)
            self.assertEqual(info["total_tasks"], 1)
            self.assertEqual(info["total_videos"], 0)
            self.assertEqual(
                info["data_path"],
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            )
            self.assertEqual(info["features"][HEAD_IMAGE_KEY]["dtype"], "image")
            self.assertEqual(info["features"][HEAD_IMAGE_KEY]["shape"], [4, 6, 3])
            self.assertEqual(info["features"]["state"]["shape"], [DOF])
            self.assertNotIn("observation.images.head", info["features"])

            tasks = _read_jsonl(output_root / "meta/tasks.jsonl")
            self.assertEqual(tasks, [{"task": "pick the red cup", "task_index": 0}])
            episodes = _read_jsonl(output_root / "meta/episodes.jsonl")
            self.assertEqual(
                episodes,
                [{"episode_index": 0, "length": 2, "tasks": ["pick the red cup"]}],
            )
            episodes_stats = _read_jsonl(output_root / "meta/episodes_stats.jsonl")
            self.assertEqual(episodes_stats[0]["episode_index"], 0)
            self.assertIn("head_image", episodes_stats[0]["stats"])
            self.assertEqual(
                len(episodes_stats[0]["stats"]["head_image"]["mean"]), 3
            )

    def test_convert_dataset_supports_joint_position_action_and_history_state(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            history = [float(value) for value in range(DOF * 10)]
            _create_source_dataset(
                source_root,
                dataset_name="source_dataset",
                field_selection=FieldSelection(
                    target=("joint_position",),
                    state=("joint_pos_rel_history",),
                    include_policy_action=False,
                ),
                policy_state={"joint_pos_rel_history": history},
            )

            convert_dataset(
                source_root,
                "source_dataset",
                dest_root,
                state_key="observation.state.joint_pos_rel_history",
                action_key="action.joint_position",
                history_window_index=-1,
                frame_reader_factory=FakeFrameReader,
            )

            row = pq.read_table(
                dest_root / "source_dataset_pi05/data/chunk-000/episode_000000.parquet"
            ).slice(0, 1).to_pylist()[0]
            self.assertEqual(row["state"], [float(value) for value in range(261, 290)])
            self.assertEqual(row["actions"], [4.0] * DOF)

    def test_output_data_path_uses_lerobot_episode_chunk(self):
        self.assertEqual(
            _dest_data_rel_path(0),
            Path("data/chunk-000/episode_000000.parquet"),
        )
        self.assertEqual(
            _dest_data_rel_path(999),
            Path("data/chunk-000/episode_000999.parquet"),
        )
        self.assertEqual(
            _dest_data_rel_path(1000),
            Path("data/chunk-001/episode_001000.parquet"),
        )

    def test_open_frame_readers_closes_partial_readers_on_open_failure(self):
        closed = []

        class TrackingReader:
            frame_count = 2
            shape = (4, 6, 3)

            def __init__(self, path: Path) -> None:
                self.path = path

            def read_rgb(self, index: int):
                raise AssertionError("read_rgb should not be called")

            def close(self):
                closed.append(self.path)

        def factory(path: Path):
            if path.name == "ego.mp4":
                raise RuntimeError("open failed")
            return TrackingReader(path)

        with self.assertRaisesRegex(RuntimeError, "open failed"):
            _open_frame_readers(
                {
                    HEAD_IMAGE_KEY: Path("head.mp4"),
                    EGO_IMAGE_KEY: Path("ego.mp4"),
                },
                factory,
            )

        self.assertEqual(closed, [Path("head.mp4")])

    def test_open_cv_frame_reader_releases_capture_on_invalid_metadata(self):
        released = []

        class Capture:
            def isOpened(self):
                return True

            def get(self, prop):
                values = {
                    1: 0,
                    2: 640,
                    3: 480,
                }
                return values[prop]

            def release(self):
                released.append(True)

        capture = Capture()
        fake_cv2 = types.SimpleNamespace(
            VideoCapture=lambda path: capture,
            CAP_PROP_FRAME_COUNT=1,
            CAP_PROP_FRAME_WIDTH=2,
            CAP_PROP_FRAME_HEIGHT=3,
        )

        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
            with self.assertRaisesRegex(ValueError, "source video has invalid metadata"):
                OpenCvFrameReader(Path("bad.mp4"))

        self.assertEqual(released, [True])

    def test_convert_dataset_rejects_missing_default_action_and_cleans_output(self):
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
                    state=("joint_position",),
                    include_policy_action=False,
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "missing required source action field: action.policy_action",
            ):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    frame_reader_factory=FakeFrameReader,
                )
            self.assertFalse((dest_root / "source_dataset_pi05").exists())
            self.assertEqual(list(dest_root.iterdir()), [])

    def test_convert_dataset_rejects_wrong_vector_dimension(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(source_root, dataset_name="source_dataset")

            with self.assertRaisesRegex(
                ValueError,
                r"action\.aligned_target_pos has shape \(45,\); expected \(29,\) or a history multiple",
            ):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    action_key="action.aligned_target_pos",
                    frame_reader_factory=FakeFrameReader,
                )
            self.assertFalse((dest_root / "source_dataset_pi05").exists())

    def test_failed_materialization_does_not_leave_partial_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(source_root, dataset_name="source_dataset")

            def fail_encoder(rgb_frame):
                raise RuntimeError("png encode failed")

            with self.assertRaisesRegex(RuntimeError, "png encode failed"):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    output_name="converted_dataset",
                    frame_reader_factory=FakeFrameReader,
                    image_encoder=fail_encoder,
                )
            self.assertFalse((dest_root / "converted_dataset").exists())
            self.assertEqual(list(dest_root.iterdir()), [])

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
                    frame_reader_factory=FakeFrameReader,
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
                    frame_reader_factory=FakeFrameReader,
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
            episodes = _read_jsonl(episodes_path)
            episodes[0]["data_path"] = "../escape.parquet"
            episodes_path.write_text(_jsonl_content(episodes), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"episodes\[0\]\.data_path escapes root via parent traversal",
            ):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    frame_reader_factory=FakeFrameReader,
                )
            self.assertFalse((dest_root / "source_dataset_pi05").exists())

    def test_convert_dataset_rejects_escaped_video_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            dataset_root = _create_source_dataset(source_root, dataset_name="source_dataset")

            episodes_path = dataset_root / "meta/episodes.jsonl"
            episodes = _read_jsonl(episodes_path)
            episodes[0]["video_paths"]["observation.images.head"] = "../escape.mp4"
            episodes_path.write_text(_jsonl_content(episodes), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"video_paths\[observation\.images\.head\] escapes root via parent traversal",
            ):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    frame_reader_factory=FakeFrameReader,
                )
            self.assertFalse((dest_root / "source_dataset_pi05").exists())

    def test_convert_dataset_requires_head_and_ego_camera_features(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(
                source_root,
                dataset_name="source_dataset",
                camera_keys=["observation.images.head"],
            )

            with self.assertRaisesRegex(
                ValueError,
                "missing required source camera feature: observation.images.ego_view",
            ):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    frame_reader_factory=FakeFrameReader,
                )
            self.assertFalse((dest_root / "source_dataset_pi05").exists())

    def test_convert_dataset_rejects_existing_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(source_root, dataset_name="source_dataset")
            (dest_root / "source_dataset_pi05").mkdir()

            with self.assertRaisesRegex(ValueError, "output dataset already exists"):
                convert_dataset(
                    source_root,
                    "source_dataset",
                    dest_root,
                    frame_reader_factory=FakeFrameReader,
                )

    def test_main_runs_cli_arguments(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "outputs"
            dest_root = root / "exports"
            source_root.mkdir()
            dest_root.mkdir()
            _create_source_dataset(source_root, dataset_name="source_dataset")

            with mock.patch(
                "robo_collector.pi05_converter.OpenCvFrameReader",
                FakeFrameReader,
            ):
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
    camera_keys: list[str] | None = None,
    policy_state: dict[str, list[float]] | None = None,
) -> Path:
    writer = LeRobotV21Writer(
        source_root,
        dataset_name=dataset_name,
        fps=50,
        camera_keys=camera_keys
        or ["observation.images.head", "observation.images.ego_view"],
        field_selection=field_selection or _default_selection(),
        video_sink_factory=FakeVideoSink,
    )
    writer.start_episode("pick the red cup")
    frame = _robot_frame(policy_state=policy_state)
    frame_bundle = _frame_bundle(camera_keys or writer.camera_keys)
    writer.add_frame(frame, frame_bundle)
    writer.add_frame(frame, frame_bundle)
    writer.save_episode()
    return source_root / dataset_name


def _default_selection() -> FieldSelection:
    return FieldSelection(
        target=("aligned_target_pos",),
        state=("joint_position",),
        include_policy_action=True,
    )


def _frame_bundle(camera_keys: list[str]) -> dict[str, FakeFrame]:
    frames = {}
    for camera_key in camera_keys:
        if camera_key.startswith("observation.images."):
            frames[camera_key.removeprefix("observation.images.")] = FakeFrame()
    return frames


def _robot_frame(policy_state: dict[str, list[float]] | None = None) -> RobotFrame:
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
        policy_state=policy_state or {},
        joint_names=[f"j{i}" for i in range(DOF)],
    )


def _png_size(payload: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(payload)) as image:
        return image.size


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _jsonl_content(rows: list[dict]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


if __name__ == "__main__":
    unittest.main()
