import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from robo_collector.episode_quality import EpisodeQualityGate
from robo_collector.raw_episode import (
    RawEpisodeReader,
    RawEpisodeRecorder,
    create_materialization_job,
    scan_startup,
)
from robo_collector.raw_materializer import (
    MaterializationConfig,
    MaterializationError,
    RawEpisodeMaterializer,
    _align_records,
)
from robo_collector.lerobot_dataset import SaveResult

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
    parquet.write_table(pa.table({"frame_index": list(range(row_count))}), path)


def _write_valid_mp4(path: Path, frame_count: int) -> None:
    if cv2 is None or np is None:
        raise RuntimeError("opencv and numpy are required for artifact validation tests")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (8, 6)
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


def _provenance(sequence, timestamp, clock_domain="camera:test"):
    return {
        "sequence": sequence,
        "clock_domain": clock_domain,
        "record_monotonic_timestamp": timestamp,
        "session_id": "session-test",
    }


def _sealed_episode(root, episode_id="episode-1"):
    recorder = RawEpisodeRecorder(root, episode_id, source_scope="camera_capture")
    recorder.append_camera("head", b"immutable-camera-bytes", _provenance(0, 1.0))
    recorder.append_robot_state({}, _provenance(0, 1.0, "robot:test"))
    recorder.close()
    return Path(root) / episode_id


class _IdempotenceWriter:
    def __init__(self, root, name, fps, streams):
        self.root = Path(root) / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_episode_index = 0
        self.frames = 0

    @property
    def active_frame_count(self):
        return self.frames

    def start_episode(self, task_prompt, episode_id):
        return 0

    def add_frame(self, frame, images, *, camera_timestamps_sec, alignment_metadata):
        self.frames += 1

    def save_episode(self, progress_callback=None):
        data = self.root / "data" / "train.parquet"
        video = self.root / "videos" / "head" / "episode.mp4"
        data.parent.mkdir(parents=True)
        video.parent.mkdir(parents=True)
        _write_valid_parquet(data, self.frames)
        _write_valid_mp4(video, self.frames)
        return SaveResult(
            saved=True, episode_index=0, frame_count=self.frames,
            data_path=data, video_path=video,
            video_paths={"observation.images.head": video}, message="saved",
        )

    def discard_episode(self):
        self.active_episode_index = None


class RawCaptureSpecAcceptanceTest(unittest.TestCase):
    def test_running_materialization_job_is_recoverable_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = _sealed_episode(directory)
            job = create_materialization_job(episode, {"fps": 30}, "schema-1")
            from robo_collector.raw_episode import claim_materialization_job

            claim_materialization_job(episode, job["job_id"])
            recovered = scan_startup(directory)

            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["status"], "PENDING")
            manifest = json.loads((episode / "manifest.json").read_text())
            self.assertEqual(manifest["materialization_jobs"][0]["attempt_count"], 1)

    def test_conversion_failure_keeps_raw_payload_and_marks_job_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = _sealed_episode(directory)
            raw_before = {
                path: path.read_bytes()
                for path in episode.rglob("*.raw")
            }
            manifest_before = json.loads((episode / "manifest.json").read_text())

            def failing_writer(*args):
                raise RuntimeError("encoder unavailable")

            materializer = RawEpisodeMaterializer(
                MaterializationConfig(
                    output_root=Path(directory) / "derived",
                    dataset_name="dataset",
                    fps=1,
                    camera_streams=("head",),
                ),
                image_decoder=lambda payload, encoding: payload,
                writer_factory=failing_writer,
            )
            with self.assertRaises(RuntimeError):
                materializer.materialize(episode)

            self.assertEqual(
                {path: path.read_bytes() for path in episode.rglob("*.raw")},
                raw_before,
            )
            manifest = json.loads((episode / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "MATERIALIZATION_FAILED")
            self.assertEqual(manifest["raw_manifest_hash"], manifest_before["raw_manifest_hash"])
            self.assertEqual(manifest["materialization_jobs"][0]["status"], "FAILED")
            RawEpisodeReader(episode).validate()

    def test_quality_gate_never_returns_ready_for_invalid_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "episode.parquet"
            video = root / "episode.mp4"
            provenance = root / "raw_provenance.json"
            data.write_bytes(b"not-parquet")
            video.write_bytes(b"not-mp4")
            provenance.write_text("{}")
            evidence = lambda path: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "row_count": 1,
                "frame_count": 1,
                "decodable": True,
            }
            manifest = {
                "schema": "robo_collector.raw_episode.v1",
                "episode_id": "episode-1",
                "status": "READY",
                "source_scope": "camera_capture",
                "streams": {"head": {"frame_count": 1}},
                "quality": {"camera_camera_skew_sec": 0, "state_camera_skew_sec": 0},
                "artifacts": {
                    "data": str(data),
                    "videos": {"head": str(video)},
                    "provenance": str(provenance),
                    "evidence": {
                        "data": evidence(data),
                        "videos": {"head": evidence(video)},
                        "provenance": evidence(provenance),
                    },
                },
            }

            report = EpisodeQualityGate().evaluate(manifest)

            self.assertNotEqual(report["status"], "READY")
            self.assertIn("artifact_not_valid_parquet: data", report["reason"])
            self.assertIn("artifact_not_valid_mp4: head", report["reason"])

    def test_strict_alignment_uses_latest_action_at_or_before_target(self):
        config = MaterializationConfig(
            output_root=Path("/tmp/unused"),
            dataset_name="dataset",
            fps=2,
            camera_streams=("head",),
            max_alignment_residual_sec=0.51,
        )
        state = [
            {"sequence": 0, "record_monotonic_timestamp": 1.0, "clock_domain": "robot"},
            {"sequence": 1, "record_monotonic_timestamp": 2.0, "clock_domain": "robot"},
        ]
        camera = {"head": [
            {"sequence": 0, "record_monotonic_timestamp": 1.0, "clock_domain": "camera"},
            {"sequence": 1, "record_monotonic_timestamp": 2.0, "clock_domain": "camera"},
        ]}

        selections = _align_records(camera, state, config)

        self.assertEqual(len(selections), 2)
        self.assertIsNotNone(selections[0])
        self.assertIsNotNone(selections[1])
        self.assertEqual(selections[0].action_record["sequence"], 0)
        self.assertEqual(selections[1].action_record["sequence"], 1)

    def test_cross_clock_alignment_requires_host_time_provenance(self):
        config = MaterializationConfig(
            output_root=Path("/tmp/unused"), dataset_name="dataset", fps=1,
            camera_streams=("head",),
        )
        with self.assertRaisesRegex(MaterializationError, "cross clock-domain"):
            _align_records(
                {"head": [{"device_timestamp": 1, "clock_domain": "camera"}]},
                [{"device_timestamp": 1, "clock_domain": "robot"}],
                config,
            )

    @unittest.skipUnless(
        cv2 is not None and np is not None and parquet is not None and pa is not None,
        "pyarrow/opencv are required for real artifact validation",
    )
    def test_successful_conversion_is_idempotent_and_does_not_append_output(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = _sealed_episode(directory)
            writers = []

            def writer_factory(*args):
                writer = _IdempotenceWriter(*args)
                writers.append(writer)
                return writer

            materializer = RawEpisodeMaterializer(
                MaterializationConfig(
                    output_root=Path(directory) / "derived", dataset_name="dataset",
                    fps=1, camera_streams=("head",),
                ),
                image_decoder=lambda payload, encoding: payload,
                writer_factory=writer_factory,
            )
            first = materializer.materialize(episode)
            second = materializer.materialize(episode)

            self.assertEqual(first.job_id, second.job_id)
            self.assertEqual(first.frame_count, second.frame_count)
            self.assertEqual(len(writers), 1)
            self.assertEqual(writers[0].frames, 1)


if __name__ == "__main__":
    unittest.main()
