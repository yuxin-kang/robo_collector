import json
import tempfile
import unittest
from pathlib import Path

from robo_collector.gr00t_converter import convert_dataset as convert_gr00t
from robo_collector.pi05_converter import convert_dataset as convert_pi05
from robo_collector.raw_episode import RawEpisodeRecorder

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional integration dependencies
    cv2 = None
    np = None


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
        "policy_state": {
            "relative_ori_6d": [value] * 90,
            "motion_anchor_lin_vel_b": [value] * 45,
            "motion_anchor_ang_vel_b": [value] * 45,
            "ang_vel_history": [value] * 30,
            "gravity_history": [value] * 30,
            "joint_pos_rel_history": [value] * 290,
            "joint_vel_history": [value] * 290,
            "action_history": [value] * 290,
        },
        "joint_names": [f"joint_{index}" for index in range(29)],
        "state_timestamp_sec": value,
    }


def _raw_episode(root: Path) -> Path:
    recorder = RawEpisodeRecorder(
        root / "raw",
        "raw-episode-1",
        source_scope="camera_capture",
        task_prompt="pick the red cup",
        metadata={
            "fps": 2,
            "camera_streams": ["head", "ego_view"],
            "capture_config": {
                "alignment": {
                    "max_camera_clock_mapping_uncertainty_sec": 0.05,
                }
            },
            "camera_capture_attached": True,
            "camera_capture_binding": {
                "schema": "robo_collector.camera_capture_binding.v1",
                "status": "ATTACHED",
                "observed_session_ids": ["camera-session"],
                "unbound_observed_session_ids": [],
            },
            "camera_capture_sources": [
                {
                    "session_id": "camera-session",
                    "source_snapshot": {
                        "schema": "robo_collector.camera_spool_snapshot.v1",
                        "session_id": "camera-session",
                        "chunks": {
                            "camera/head/chunk-000000.msgpack": {
                                "size": 1,
                                "sha256": "b" * 64,
                            },
                            "camera/ego_view/chunk-000000.msgpack": {
                                "size": 1,
                                "sha256": "c" * 64,
                            },
                        },
                        "stream_high_watermarks": {
                            "head": {"last_sequence": 1, "record_count": 2},
                            "ego_view": {"last_sequence": 1, "record_count": 2},
                        },
                        "record_count": 4,
                        "selected_record_counts": {
                            "head": 2,
                            "ego_view": 2,
                        },
                        "stable": True,
                    },
                    "source_snapshot_hash": "a" * 64,
                    "source_snapshot_consistent": True,
                    "binding_status": "BOUND",
                    "clock_mapping_samples": 2,
                    "clock_mapping_uncertainty_sec": 0.0,
                    "stream_high_watermarks": {
                        "head": {"last_sequence": 1, "record_count": 2},
                        "ego_view": {"last_sequence": 1, "record_count": 2},
                    },
                    "observed_stream_high_watermarks": {
                        "head": 1,
                        "ego_view": 1,
                    },
                    "selected_sequence_ranges": {
                        "head": {"first_sequence": 0, "last_sequence": 1, "count": 2},
                        "ego_view": {"first_sequence": 0, "last_sequence": 1, "count": 2},
                    },
                    "record_counts": {"head": 2, "ego_view": 2},
                }
            ],
        },
    )
    ok, buffer = cv2.imencode(
        ".jpg", np.zeros((6, 8, 3), dtype=np.uint8)
    )
    if not ok:
        raise RuntimeError("failed to encode test image")
    payload = buffer.tobytes()
    for sequence in range(2):
        timestamp = sequence * 0.5
        for stream in ("head", "ego_view"):
            recorder.append_camera(
                stream,
                payload,
                {
                    "sequence": sequence,
                    "clock_domain": f"camera:{stream}",
                    "timestamp_quality": "device",
                    "device_timestamp": timestamp,
                    "device_unit": "s",
                    "record_monotonic_timestamp": timestamp,
                    "session_id": f"camera-{stream}",
                },
                payload_encoding="image/jpeg",
            )
        recorder.append_robot_state(
            _state(timestamp),
            {
                "sequence": sequence,
                "clock_domain": "robot-state",
                "timestamp_quality": "ros_message",
                "record_monotonic_timestamp": timestamp,
                "session_id": "robot-session",
            },
        )
    recorder.close()
    manifest_path = root / "raw" / "raw-episode-1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Complete-capture conversion requires durable recording health evidence;
    # quality is mutable and intentionally excluded from raw_manifest_hash.
    manifest["quality"] = {
        "statistics": {
            "recording": {
                "recording_failed": False,
                "timer_ticks": 2,
                "timer_deadline_misses": 0,
                "state_age_max_sec": 0.0,
                "camera_age_max_sec": 0.0,
            }
        }
    }
    manifest["status"] = "READY"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (manifest_path.parent / "quality.json").write_text(
        json.dumps({"report_schema": "robo_collector.episode_quality.v1", "status": "READY"}),
        encoding="utf-8",
    )
    return root / "raw" / "raw-episode-1"


@unittest.skipUnless(
    cv2 is not None and np is not None,
    "opencv and numpy are required for raw converter integration tests",
)
class RawConverterIntegrationTest(unittest.TestCase):
    def test_raw_closed_status_is_not_convertible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _raw_episode(root)
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "RAW_CLOSED"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "status 'RAW_CLOSED'"):
                convert_gr00t(
                    source.parent,
                    source.name,
                    root / "exports",
                    output_name="gr00t-raw-closed",
                    action_source="aligned_target_pos",
                )

    def test_raw_review_status_is_not_convertible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _raw_episode(root)
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "REVIEW"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "status 'REVIEW'"):
                convert_gr00t(
                    source.parent,
                    source.name,
                    root / "exports",
                    output_name="gr00t-review",
                    action_source="aligned_target_pos",
                )

    def test_gr00t_accepts_raw_episode_and_preserves_raw_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _raw_episode(root)
            result = convert_gr00t(
                source.parent,
                source.name,
                root / "exports",
                output_name="gr00t-output",
                action_source="aligned_target_pos",
            )

            self.assertEqual(result.source_dataset, source)
            provenance = json.loads(
                (result.output_dataset / "meta/raw_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(provenance["source_episode_id"], "raw-episode-1")
            self.assertEqual(provenance["source_episode_ids"], ["raw-episode-1"])
            self.assertEqual(provenance["source_manifest_type"], "raw_episode")
            self.assertEqual(provenance["output_schema_version"], "gr00t.v1")
            episode_row = json.loads(
                (result.output_dataset / "meta/episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(episode_row["source_episode_id"], "raw-episode-1")
            self.assertEqual(
                episode_row["source_manifest_hash"], provenance["source_manifest_hash"]
            )

            reused = convert_gr00t(
                source.parent,
                source.name,
                root / "exports",
                output_name="gr00t-output",
                action_source="aligned_target_pos",
            )
            self.assertEqual(reused.output_dataset, result.output_dataset)

    def test_gr00t_does_not_reuse_output_after_source_quality_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _raw_episode(root)
            result = convert_gr00t(
                source.parent,
                source.name,
                root / "exports",
                output_name="gr00t-output-quality-revoked",
                action_source="aligned_target_pos",
            )

            quality_path = source / "quality.json"
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            quality["status"] = "REVIEW"
            quality_path.write_text(json.dumps(quality), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "quality status 'REVIEW'"):
                convert_gr00t(
                    source.parent,
                    source.name,
                    root / "exports",
                    output_name=result.output_dataset.name,
                    action_source="aligned_target_pos",
                )

    def test_pi05_accepts_raw_episode_and_preserves_raw_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _raw_episode(root)
            result = convert_pi05(
                source.parent,
                source.name,
                root / "exports",
                output_name="pi05-output",
            )

            self.assertEqual(result.source_dataset, source)
            provenance = json.loads(
                (result.output_dataset / "meta/raw_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(provenance["source_episode_id"], "raw-episode-1")
            self.assertEqual(provenance["source_episode_ids"], ["raw-episode-1"])
            self.assertEqual(provenance["source_manifest_type"], "raw_episode")
            self.assertEqual(provenance["output_schema_version"], "openpi.pi05.v1")
            episode_row = json.loads(
                (result.output_dataset / "meta/episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(episode_row["source_episode_id"], "raw-episode-1")
            self.assertEqual(
                episode_row["source_manifest_hash"], provenance["source_manifest_hash"]
            )

            reused = convert_pi05(
                source.parent,
                source.name,
                root / "exports",
                output_name="pi05-output",
            )
            self.assertEqual(reused.output_dataset, result.output_dataset)


if __name__ == "__main__":
    unittest.main()
