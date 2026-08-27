import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from robo_collector import lerobot_dataset
from robo_collector import raw_materializer
from robo_collector.lerobot_dataset import SaveResult
from robo_collector.lerobot_dataset import LeRobotV21Writer
from robo_collector.episode_quality import EpisodeQualityGate
from robo_collector.raw_episode import RawEpisodeReader, RawEpisodeRecorder
from robo_collector.raw_materializer import (
    MaterializationConfig,
    RawEpisodeMaterializer,
    _write_provenance_file,
    _record_timestamp,
)

try:
    import cv2
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as parquet
except ImportError:  # pragma: no cover - optional test dependencies
    cv2 = None
    np = None
    pa = None
    parquet = None


def _write_valid_parquet(path: Path, row_count: int) -> None:
    if parquet is None or pa is None:
        raise RuntimeError("pyarrow is required for artifact validation tests")
    parquet.write_table(
        pa.table({"frame_index": list(range(row_count))}),
        path,
    )


def _write_valid_mp4(path: Path, frame_count: int) -> None:
    if cv2 is None or np is None:
        raise RuntimeError("opencv and numpy are required for artifact validation tests")
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (8, 6),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("opencv mp4v encoder is unavailable")
    try:
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        for _ in range(frame_count):
            writer.write(frame)
    finally:
        writer.release()


def _state(value: float) -> dict:
    return {
        "joint_position": [value] * 29,
        "joint_velocity": [value] * 29,
        "joint_torque": [value] * 29,
        "imu_angular_velocity": [value] * 3,
        "imu_linear_acceleration": [value] * 3,
        "projected_gravity_or_quat": [value] * 4,
        "target_joint_pos": [value] * 29,
        "policy_action": [value] * 29,
        "aligned_target_pos": [value] * 45,
        "policy_state": {"relative_ori_6d": [value] * 90},
        "joint_names": [f"joint_{index}" for index in range(29)],
        "state_timestamp_sec": value,
    }


class _FakeWriter:
    def __init__(self, root: Path, name: str, fps: int, streams: tuple[str, ...]):
        self.root = root / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.active_episode_index = 0

    @property
    def active_frame_count(self):
        return len(self.frames)

    def start_episode(self, task_prompt, episode_id):
        self.task_prompt = task_prompt
        self.episode_id = episode_id
        return 0

    def add_frame(
        self,
        frame,
        images,
        *,
        camera_timestamps_sec,
        alignment_metadata,
    ):
        self.frames.append(
            (frame, images, camera_timestamps_sec, alignment_metadata)
        )

    def save_episode(self, progress_callback=None):
        data = self.root / "data" / "train-000000.parquet"
        data.parent.mkdir(parents=True, exist_ok=True)
        _write_valid_parquet(data, len(self.frames))
        videos = {}
        for stream in ("head", "ego_view"):
            path = self.root / "videos" / stream / "episode_000000.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_valid_mp4(path, len(self.frames))
            videos[f"observation.images.{stream}"] = path
        return SaveResult(
            saved=True,
            episode_index=0,
            frame_count=len(self.frames),
            data_path=data,
            video_path=next(iter(videos.values())),
            video_paths=videos,
            message="saved",
        )

    def discard_episode(self):
        self.active_episode_index = None


class _FailingWriter(_FakeWriter):
    def save_episode(self, progress_callback=None):
        raise RuntimeError("encoder failed")


class RawMaterializerTest(unittest.TestCase):
    def test_materialization_failure_leaves_a_quality_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = RawEpisodeRecorder(root / "raw", "episode-failed")
            recorder.append_camera(
                "head",
                b"image",
                {
                    "sequence": 0,
                    "clock_domain": "camera:head",
                    "device_timestamp": 0.0,
                    "device_unit": "s",
                    "record_monotonic_timestamp": 0.0,
                    "session_id": "camera-session",
                },
                payload_encoding="image/jpeg",
            )
            recorder.append_robot_state(
                _state(0.0),
                {
                    "sequence": 0,
                    "clock_domain": "robot-state",
                    "record_monotonic_timestamp": 0.0,
                    "session_id": "robot-session",
                },
            )
            recorder.close()

            def writer_factory(output_root, name, fps, streams):
                return _FailingWriter(Path(output_root), name, fps, tuple(streams))

            materializer = RawEpisodeMaterializer(
                MaterializationConfig(
                    output_root=root / "derived",
                    dataset_name="dataset",
                    fps=1,
                    camera_streams=("head",),
                ),
                image_decoder=lambda payload, encoding: payload,
                writer_factory=writer_factory,
            )

            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                materializer.materialize(root / "raw" / "episode-failed")

            episode = root / "raw" / "episode-failed"
            manifest = json.loads((episode / "manifest.json").read_text())
            report = json.loads((episode / "quality.json").read_text())
            self.assertEqual(manifest["status"], "MATERIALIZATION_FAILED")
            self.assertEqual(report["status"], "REJECT")
            self.assertTrue(
                any("materialization_failed" in reason for reason in report["reason"])
            )

    def test_per_episode_provenance_sidecars_are_immutable_and_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "derived" / "dataset" / "data" / "train-000000.parquet"
            data_path.parent.mkdir(parents=True)
            data_path.write_bytes(b"PAR1dataPAR1")
            result = SaveResult(
                saved=True,
                episode_index=0,
                frame_count=1,
                data_path=data_path,
                video_path=None,
                message="staged",
            )
            config = MaterializationConfig(
                output_root=root / "derived",
                dataset_name="dataset",
                fps=1,
                camera_streams=("head",),
            )
            first = _write_provenance_file(
                result,
                source_episode_id="episode-1",
                source_manifest_hash="a" * 64,
                config=config,
                dropped_selection_count=0,
                residuals=[0.01],
            )
            first_bytes = first.read_bytes()
            second = _write_provenance_file(
                result,
                source_episode_id="episode-2",
                source_manifest_hash="b" * 64,
                config=config,
                dropped_selection_count=1,
                residuals=[0.02],
            )

            self.assertEqual(first, data_path.parent.parent / "meta/raw_provenance/episode-1.json")
            self.assertEqual(second, data_path.parent.parent / "meta/raw_provenance/episode-2.json")
            self.assertEqual(first.read_bytes(), first_bytes)
            aggregate = json.loads(
                (data_path.parent.parent / "meta/raw_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                list(aggregate["episode_provenance"]), ["episode-1", "episode-2"]
            )
            self.assertEqual(
                aggregate["provenance_files"]["episode-1"],
                "meta/raw_provenance/episode-1.json",
            )
            self.assertEqual(aggregate["source_episode_id"], "episode-1")

    def test_record_timestamp_normalizes_device_units(self):
        self.assertEqual(
            _record_timestamp({"device_timestamp": 2_000, "device_unit": "ms"}),
            2.0,
        )
        self.assertEqual(
            _record_timestamp({"device_timestamp": 3_000_000, "device_unit": "us"}),
            3.0,
        )

    @unittest.skipUnless(
        cv2 is not None and np is not None and parquet is not None and pa is not None,
        "pyarrow/opencv are required for real artifact validation",
    )
    def test_materializes_and_records_provenance_and_qc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = RawEpisodeRecorder(root / "raw", "episode-1", source_scope="camera_capture", task_prompt="pick")
            for index in range(2):
                timestamp = index * 0.5
                provenance = {
                    "sequence": index,
                    "clock_domain": "camera:head",
                    "timestamp_quality": "device",
                    "device_timestamp": timestamp,
                    "device_unit": "s",
                    "record_monotonic_timestamp": timestamp,
                    "session_id": "session-a",
                }
                recorder.append_camera("head", b"head", provenance, payload_encoding="image/jpeg")
                recorder.append_camera(
                    "ego_view",
                    b"ego",
                    {**provenance, "clock_domain": "camera:ego_view"},
                    payload_encoding="image/jpeg",
                )
                recorder.append_robot_state(
                    _state(timestamp),
                    {
                        "sequence": index,
                        "clock_domain": "robot_state",
                        "timestamp_quality": "ros_message",
                        "record_monotonic_timestamp": timestamp,
                        "session_id": "session-robot",
                    },
                )
            recorder.close()

            writers = []

            def writer_factory(output_root, name, fps, streams):
                writer = _FakeWriter(Path(output_root), name, fps, tuple(streams))
                writers.append(writer)
                return writer

            materializer = RawEpisodeMaterializer(
                MaterializationConfig(
                    output_root=root / "derived",
                    dataset_name="dataset",
                    fps=2,
                    camera_streams=("head", "ego_view"),
                    max_alignment_residual_sec=0.01,
                ),
                image_decoder=lambda payload, encoding: payload,
                writer_factory=writer_factory,
            )
            result = materializer.materialize(root / "raw" / "episode-1")

            self.assertEqual(result.frame_count, 2)
            self.assertIn(result.quality_status, {"READY", "REVIEW"})
            self.assertEqual(len(writers[0].frames), 2)
            self.assertEqual(
                writers[0].frames[0][3]["selection_policy"],
                "fixed_rate_nearest_strict",
            )
            self.assertEqual(writers[0].frames[0][3]["state_sequence"], 0)
            self.assertAlmostEqual(
                writers[0].frames[0][3]["selected_dataset_timestamp"], 0.0
            )
            self.assertAlmostEqual(
                writers[0].frames[1][3]["selected_dataset_timestamp"], 0.5
            )
            self.assertAlmostEqual(
                writers[0].frames[0][3]["alignment_target_source_timestamp"],
                0.0,
            )
            manifest = json.loads(
                (root / "raw" / "episode-1" / "manifest.json").read_text()
            )
            self.assertIn(manifest["status"], {"READY", "REVIEW"})
            self.assertEqual(manifest["materialization_jobs"][0]["status"], "MATERIALIZED")
            quality = json.loads((root / "raw" / "episode-1" / "quality.json").read_text())
            self.assertIn(quality["status"], {"READY", "REVIEW"})
            provenance = result.output_dataset / "meta" / "raw_provenance.json"
            self.assertEqual(
                json.loads(provenance.read_text())["source_manifest_hash"],
                manifest["raw_manifest_hash"],
            )
            replay = materializer.materialize(root / "raw" / "episode-1")
            self.assertEqual(replay.job_id, result.job_id)
            self.assertEqual(replay.frame_count, result.frame_count)
            self.assertEqual(len(writers), 1)

    @unittest.skipUnless(
        cv2 is not None and np is not None and parquet is not None and pa is not None,
        "pyarrow/opencv are required for real artifact validation",
    )
    def test_selected_dataset_timestamps_are_dense_after_dropped_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = RawEpisodeRecorder(
                root / "raw",
                "episode-dense-timestamps",
                source_scope="camera_capture",
                task_prompt="pick",
            )
            for timestamp in (0.0, 2.0):
                for stream in ("head", "ego_view"):
                    recorder.append_camera(
                        stream,
                        b"image",
                        {
                            "sequence": int(timestamp),
                            "clock_domain": f"camera:{stream}",
                            "timestamp_quality": "device",
                            "device_timestamp": timestamp,
                            "device_unit": "s",
                            "record_monotonic_timestamp": timestamp,
                            "session_id": f"session-{stream}",
                        },
                        payload_encoding="image/jpeg",
                    )
                recorder.append_robot_state(
                    _state(timestamp),
                    {
                        "sequence": int(timestamp),
                        "clock_domain": "robot_state",
                        "timestamp_quality": "ros_message",
                        "record_monotonic_timestamp": timestamp,
                        "session_id": "session-robot",
                    },
                )
            recorder.close()

            writers = []

            def writer_factory(output_root, name, fps, streams):
                writer = _FakeWriter(Path(output_root), name, fps, tuple(streams))
                writers.append(writer)
                return writer

            result = RawEpisodeMaterializer(
                MaterializationConfig(
                    output_root=root / "derived",
                    dataset_name="dataset",
                    fps=1,
                    camera_streams=("head", "ego_view"),
                    max_alignment_residual_sec=0.1,
                ),
                image_decoder=lambda payload, encoding: payload,
                writer_factory=writer_factory,
            ).materialize(root / "raw" / "episode-dense-timestamps")

            self.assertEqual(result.frame_count, 2)
            self.assertEqual(result.dropped_selection_count, 1)
            rows = [item[3] for item in writers[0].frames]
            self.assertEqual(
                [row["selected_dataset_timestamp"] for row in rows], [0.0, 1.0]
            )
            self.assertEqual(
                [row["alignment_target_source_timestamp"] for row in rows], [0.0, 2.0]
            )

    @unittest.skipUnless(
        cv2 is not None and np is not None and parquet is not None and pa is not None,
        "pyarrow/opencv are required for the default writer integration test",
    )
    def test_default_writer_persists_alignment_provenance_and_valid_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = RawEpisodeRecorder(
                root / "raw",
                "episode-default-writer",
                source_scope="camera_capture",
                task_prompt="pick",
            )
            ok, buffer = cv2.imencode(
                ".jpg", np.zeros((6, 8, 3), dtype=np.uint8)
            )
            self.assertTrue(ok)
            payload = buffer.tobytes()
            for index in range(2):
                timestamp = index * 0.5
                provenance = {
                    "sequence": index,
                    "clock_domain": "camera:head",
                    "timestamp_quality": "device",
                    "device_timestamp": timestamp,
                    "device_unit": "s",
                    "record_monotonic_timestamp": timestamp,
                    "session_id": "session-camera",
                }
                recorder.append_camera(
                    "head", payload, provenance, payload_encoding="image/jpeg"
                )
                recorder.append_robot_state(
                    _state(timestamp),
                    {
                        "sequence": index,
                        "clock_domain": "robot_state",
                        "timestamp_quality": "ros_message",
                        "record_monotonic_timestamp": timestamp,
                        "session_id": "session-robot",
                    },
                )
            recorder.close()

            result = RawEpisodeMaterializer(
                MaterializationConfig(
                    output_root=root / "derived",
                    dataset_name="dataset",
                    fps=2,
                    camera_streams=("head",),
                    max_alignment_residual_sec=0.01,
                )
            ).materialize(root / "raw" / "episode-default-writer")

            self.assertEqual(result.frame_count, 2)
            self.assertEqual(result.quality_status, "READY")
            data = result.output_dataset / "data" / "train-000000.parquet"
            self.assertEqual(parquet.ParquetFile(data).metadata.num_rows, 2)
            info = json.loads(
                (result.output_dataset / "meta" / "info.json").read_text()
            )
            self.assertIn("alignment.selection_policy", info["features"])
            episode_row = json.loads(
                (result.output_dataset / "meta" / "episodes.jsonl")
                .read_text()
                .splitlines()[0]
            )
            manifest = json.loads(
                (root / "raw" / "episode-default-writer" / "manifest.json")
                .read_text()
            )
            self.assertEqual(episode_row["source_episode_id"], "episode-default-writer")
            self.assertEqual(episode_row["source_manifest_hash"], manifest["raw_manifest_hash"])
            metadata = {
                "present": True,
                "tasks": [
                    json.loads(line)
                    for line in (
                        result.output_dataset / "meta" / "tasks.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ],
                "episodes": [episode_row],
                "info": info,
                "modality": json.loads(
                    (result.output_dataset / "meta" / "modality.json").read_text(
                        encoding="utf-8"
                    )
                ),
            }
            self.assertTrue(
                raw_materializer._metadata_generation_is_consistent(
                    metadata, fps=2, camera_streams=("head",)
                )
            )
            broken_metadata = dict(metadata)
            broken_metadata["modality"] = {}
            self.assertFalse(
                raw_materializer._metadata_generation_is_consistent(
                    broken_metadata, fps=2, camera_streams=("head",)
                )
            )
            artifacts = manifest["materialization_jobs"][0]["artifacts"]
            self.assertTrue(
                raw_materializer._pending_publication_artifacts_are_valid(
                    artifacts,
                    job=manifest["materialization_jobs"][0],
                    episode_id="episode-default-writer",
                    allow_non_ready=False,
                )
            )
            data_evidence = artifacts["evidence"]["data"]
            with mock.patch.object(
                raw_materializer,
                "_file_evidence",
                return_value={
                    "sha256": data_evidence["sha256"],
                    "size": data_evidence["size"],
                    "decodable": False,
                    "validation": "decoder_unavailable:pyarrow",
                    "row_count": artifacts["frame_count"],
                    "frame_count": artifacts["frame_count"],
                },
            ):
                self.assertFalse(
                    raw_materializer._pending_publication_artifacts_are_valid(
                        artifacts,
                        job=manifest["materialization_jobs"][0],
                        episode_id="episode-default-writer",
                        allow_non_ready=False,
                    )
                )
            RawEpisodeReader(
                root / "raw" / "episode-default-writer"
            ).validate()

    @unittest.skipUnless(
        cv2 is not None and np is not None and parquet is not None and pa is not None,
        "pyarrow/opencv are required for publication recovery tests",
    )
    def test_publication_recovery_does_not_trust_partial_episode_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = RawEpisodeRecorder(
                root / "raw",
                "episode-interrupted-publication",
                source_scope="camera_capture",
                task_prompt="pick",
            )
            ok, buffer = cv2.imencode(
                ".jpg", np.zeros((6, 8, 3), dtype=np.uint8)
            )
            self.assertTrue(ok)
            payload = buffer.tobytes()
            for index in range(2):
                timestamp = index * 0.5
                recorder.append_camera(
                    "head",
                    payload,
                    {
                        "sequence": index,
                        "clock_domain": "camera:head",
                        "timestamp_quality": "device",
                        "device_timestamp": timestamp,
                        "device_unit": "s",
                        "record_monotonic_timestamp": timestamp,
                        "session_id": "session-camera",
                    },
                    payload_encoding="image/jpeg",
                )
                recorder.append_robot_state(
                    _state(timestamp),
                    {
                        "sequence": index,
                        "clock_domain": "robot_state",
                        "timestamp_quality": "ros_message",
                        "record_monotonic_timestamp": timestamp,
                        "session_id": "session-robot",
                    },
                )
            recorder.close()

            writers = []

            def writer_factory(output_root, name, fps, streams):
                writer = LeRobotV21Writer(
                    output_root,
                    dataset_name=name,
                    fps=fps,
                    camera_keys=[f"observation.images.{stream}" for stream in streams],
                )
                writers.append(writer)
                return writer

            materializer = RawEpisodeMaterializer(
                MaterializationConfig(
                    output_root=root / "derived",
                    dataset_name="dataset",
                    fps=2,
                    camera_streams=("head",),
                    max_alignment_residual_sec=0.01,
                ),
                image_decoder=lambda payload, encoding: np.zeros(
                    (6, 8, 3), dtype=np.uint8
                ),
                writer_factory=writer_factory,
            )
            original_replace_path = lerobot_dataset._replace_path

            def interrupt_after_partial_metadata(source, target):
                if target.name == "info.json":
                    raise KeyboardInterrupt("simulated power loss during publication")
                original_replace_path(source, target)

            lerobot_dataset._replace_path = interrupt_after_partial_metadata
            try:
                with self.assertRaisesRegex(KeyboardInterrupt, "power loss"):
                    materializer.materialize(
                        root / "raw" / "episode-interrupted-publication"
                    )
            finally:
                lerobot_dataset._replace_path = original_replace_path
                writers[0]._release_dataset_lock()

            output = root / "derived" / "dataset"
            self.assertTrue(
                (output / "meta" / lerobot_dataset.METADATA_TRANSACTION_FILENAME).exists()
            )
            interrupted_manifest = json.loads(
                (
                    root
                    / "raw"
                    / "episode-interrupted-publication"
                    / "manifest.json"
                ).read_text()
            )
            self.assertEqual(
                interrupted_manifest["materialization_jobs"][0]["artifacts"][
                    "publication"
                ],
                "PENDING",
            )

            result = materializer.materialize(
                root / "raw" / "episode-interrupted-publication"
            )

            self.assertEqual(result.frame_count, 2)
            self.assertEqual(len(writers), 2)
            episodes = (
                output / "meta" / "episodes.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(episodes), 1)
            self.assertEqual(json.loads(episodes[0])["episode_id"], "episode-interrupted-publication")

    @unittest.skipUnless(
        cv2 is not None and np is not None and parquet is not None and pa is not None,
        "pyarrow/opencv are required for review publication tests",
    )
    def test_transport_observed_non_ready_output_stays_out_of_shared_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = RawEpisodeRecorder(
                root / "raw",
                "episode-transport-review",
                source_scope="transport_observed",
                task_prompt="review me",
                metadata={"fps": 1},
            )
            ok, buffer = cv2.imencode(
                ".jpg", np.zeros((6, 8, 3), dtype=np.uint8)
            )
            self.assertTrue(ok)
            timestamp = 0.0
            recorder.append_camera(
                "head",
                buffer.tobytes(),
                {
                    "sequence": 0,
                    "clock_domain": "camera:head",
                    "timestamp_quality": "device",
                    "device_timestamp": timestamp,
                    "device_unit": "s",
                    "record_monotonic_timestamp": timestamp,
                    "session_id": "camera-session",
                },
                payload_encoding="image/jpeg",
            )
            recorder.append_robot_state(
                _state(timestamp),
                {
                    "sequence": 0,
                    "clock_domain": "robot-state",
                    "timestamp_quality": "ros_message",
                    "record_monotonic_timestamp": timestamp,
                    "session_id": "robot-session",
                },
            )
            recorder.close()

            output_root = root / "derived"
            materializer = RawEpisodeMaterializer(
                MaterializationConfig(
                    output_root=output_root,
                    dataset_name="dataset",
                    fps=1,
                    camera_streams=("head",),
                ),
                quality_gate=EpisodeQualityGate(require_complete_capture=True),
            )

            result = materializer.materialize(
                root / "raw" / "episode-transport-review"
            )

            self.assertEqual(result.quality_status, "REVIEW")
            self.assertIn(".review", result.output_dataset.parts)
            self.assertTrue((result.output_dataset / "data").exists())
            self.assertTrue(
                (result.output_dataset / "meta/raw_provenance.json").exists()
            )
            self.assertFalse((output_root / "dataset/meta/episodes.jsonl").exists())

            replay = materializer.materialize(
                root / "raw" / "episode-transport-review"
            )
            self.assertEqual(replay.output_dataset, result.output_dataset)
            self.assertFalse((output_root / "dataset/meta/episodes.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
