import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from robo_collector.episode_quality import EpisodeQualityGate

try:
    import cv2
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as parquet
except ImportError:  # pragma: no cover - optional validation dependencies
    cv2 = None
    np = None
    pa = None
    parquet = None


def manifest(**overrides):
    value = {
        "schema": "robo_collector.raw_episode.v1",
        "episode_id": "episode-1",
        "status": "RAW_CLOSED",
        "source_scope": "camera_capture",
        "streams": {"head": {"frame_count": 3}},
        "quality": {
            "producer_gap_count": 0,
            "transport_gap_count": 0,
            "selection_gap_count": 0,
            "camera_camera_skew_sec": 0.02,
            "state_camera_skew_sec": 0.03,
        },
    }
    value.update(overrides)
    return value


def _write_valid_parquet(path: Path, row_count: int) -> None:
    parquet.write_table(pa.table({"frame_index": list(range(row_count))}), path)


def _write_valid_mp4(path: Path, frame_count: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (8, 6)
    )
    if not writer.isOpened():
        writer.release()
        raise unittest.SkipTest("opencv mp4v encoder is unavailable")
    try:
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        for _ in range(frame_count):
            writer.write(frame)
    finally:
        writer.release()


def _complete_capture_metadata(*, attached: bool = True) -> dict:
    source = {
        "source_snapshot": {
            "schema": "robo_collector.camera_spool_snapshot.v1",
            "session_id": "camera-session",
            "chunks": {
                "camera/head/chunk-000000.msgpack": {
                    "size": 1,
                    "sha256": "a" * 64,
                }
            },
            "stream_high_watermarks": {
                "head": {"last_sequence": 2, "record_count": 3}
            },
            "record_count": 3,
            "selected_record_counts": {"head": 3},
            "stable": True,
        },
        "session_id": "camera-session",
        "source_snapshot_hash": "b" * 64,
        "source_snapshot_consistent": True,
        "binding_status": "BOUND",
        "clock_mapping_samples": 2,
        "clock_mapping_uncertainty_sec": 0.0,
        "stream_high_watermarks": {
            "head": {"last_sequence": 2, "record_count": 3}
        },
        "observed_stream_high_watermarks": {"head": 2},
        "selected_sequence_ranges": {
            "head": {"first_sequence": 0, "last_sequence": 2, "count": 3}
        },
        "record_counts": {"head": 3},
    }
    return {
        "camera_streams": ["head"],
        "capture_config": {
            "alignment": {
                "max_camera_clock_mapping_uncertainty_sec": 0.05,
            }
        },
        "camera_capture_attached": attached,
        "camera_capture_binding": {
            "schema": "robo_collector.camera_capture_binding.v1",
            "status": "ATTACHED" if attached else "OPEN",
            "observed_session_ids": ["camera-session"],
            "unbound_observed_session_ids": [],
        },
        "camera_capture_sources": [source],
    }
class EpisodeQualityGateTest(unittest.TestCase):
    @unittest.skipUnless(
        cv2 is not None and np is not None and parquet is not None and pa is not None,
        "pyarrow/opencv are required for real artifact validation",
    )
    def test_strict_closed_loop_ready_and_json_serializable(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = root / "train.parquet"
            video = root / "head.mp4"
            provenance = root / "raw_provenance.json"
            _write_valid_parquet(data, 3)
            _write_valid_mp4(video, 3)
            provenance.write_text(json.dumps({
                "source_episode_id": "episode-1",
                "source_manifest_hash": "a" * 64,
                "converter_version": "test.converter.v1",
                "conversion_config_hash": "b" * 64,
                "output_schema_version": "test.v1",
            }), encoding="utf-8")
            evidence = lambda path: {
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "row_count": 3, "frame_count": 3, "decodable": True,
            }
            value = manifest(
                status="MATERIALIZED",
                materialization={"parquet_row_count": 3, "video_frame_counts": {"head": 3}},
                artifacts={"provenance": str(provenance), "encoder_identity": {
                    "library": "opencv", "library_version": "test",
                    "backend": "test", "codec": "mp4v",
                }, "evidence": {
                    "data": evidence(data), "videos": {"head": evidence(video)},
                    "provenance": {"path": str(provenance), "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest()},
                }},
            )
            report = EpisodeQualityGate().evaluate(value)
        self.assertEqual(report["status"], "READY")
        self.assertIn("thresholds", report["rules"])
        json.dumps(report)

    @unittest.skipUnless(
        cv2 is not None and np is not None and parquet is not None and pa is not None,
        "pyarrow/opencv are required for real artifact validation",
    )
    def test_required_provenance_fields_are_checked(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = root / "train.parquet"
            video = root / "head.mp4"
            provenance = root / "raw_provenance.json"
            _write_valid_parquet(data, 1)
            _write_valid_mp4(video, 1)
            provenance.write_text(json.dumps({"source_episode_id": "episode-1"}), encoding="utf-8")
            evidence = lambda path: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "row_count": 1,
                "frame_count": 1,
                "decodable": True,
            }
            report = EpisodeQualityGate().evaluate(manifest(
                status="MATERIALIZED",
                materialization={"parquet_row_count": 1, "video_frame_counts": {"head": 1}},
                artifacts={"provenance": str(provenance), "evidence": {
                    "data": evidence(data),
                    "videos": {"head": evidence(video)},
                    "provenance": {
                        "path": str(provenance),
                        "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
                    },
                }},
            ))
        self.assertEqual(report["status"], "REJECT")
        self.assertIn("missing_encoder_evidence", report["reason"])
        self.assertTrue(any("provenance_field" in reason for reason in report["reason"]))

    def test_manifest_path_and_transport_observed_complete_capture_reject(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "manifest.json"
            path.write_text(json.dumps(manifest(source_scope="transport_observed")), encoding="utf-8")
            report = EpisodeQualityGate(require_complete_capture=True).check(path)
        # The source scope alone would require REVIEW, but this fixture also
        # has no verifiable raw chunks, which is a hard rejection.
        self.assertEqual(report["status"], "REJECT")
        self.assertTrue(any("transport_observed" in reason for reason in report["reason"]))

    def test_gaps_and_skew_over_configured_threshold_are_not_ready(self):
        report = EpisodeQualityGate(
            max_transport_gaps=1, max_state_camera_skew_sec=0.05
        ).evaluate(manifest(quality={
            "producer_gap_count": 0, "transport_gap_count": 2,
            "selection_gap_count": 0, "camera_camera_skew_sec": 0.01,
            "state_camera_skew_sec": 0.06,
        }))
        self.assertEqual(report["status"], "REVIEW")
        self.assertEqual(report["statistics"]["transport_gaps"], 2)

    def test_strict_count_mismatch_rejects(self):
        report = EpisodeQualityGate().evaluate(manifest(materialization={
            "parquet_row_count": 2, "video_frame_counts": {"head": 3}
        }))
        self.assertEqual(report["status"], "REJECT")

    def test_missing_record_provenance_rejects_and_discard_is_preserved(self):
        bad = manifest(streams={"head": {"records": [{"sequence": 1}]}})
        self.assertEqual(EpisodeQualityGate().evaluate(bad)["status"], "REJECT")
        self.assertEqual(EpisodeQualityGate().evaluate(manifest(status="DISCARDED"))["status"], "DISCARDED")

    def test_stream_integrity_counters_are_exposed(self):
        report = EpisodeQualityGate().evaluate(manifest(streams={
            "head": {
                "frame_count": 3,
                "duplicate_count": 1,
                "reorder_count": 2,
                "session_restart_count": 1,
            }
        }))
        self.assertEqual(report["statistics"]["duplicates"], 1)
        self.assertEqual(report["statistics"]["reorders"], 2)
        self.assertEqual(report["statistics"]["session_restarts"], 1)
        self.assertEqual(report["statistics"]["streams"]["head"]["reorder_count"], 2)

    def test_rejected_record_counter_blocks_ready(self):
        report = EpisodeQualityGate().evaluate(
            manifest(
                record_errors={
                    "total": 1,
                    "by_stream": {"head": 1},
                    "by_error_type": {"ValueError": 1},
                }
            )
        )
        self.assertEqual(report["statistics"]["rejected_record_count"], 1)
        self.assertEqual(report["status"], "REJECT")
        self.assertIn("invalid_records_rejected: 1", report["reason"])

    def test_malformed_rejected_record_counters_fail_closed(self):
        report = EpisodeQualityGate().evaluate(
            manifest(
                record_errors={
                    "total": 1,
                    "by_stream": {"head": 1, "malformed": "1"},
                    "by_error_type": {"ValueError": 1},
                }
            )
        )

        self.assertEqual(report["status"], "REJECT")
        self.assertIn("invalid_record_error_statistics", report["reason"])

    def test_stream_anomalies_are_enforced_by_configured_thresholds(self):
        report = EpisodeQualityGate(max_duplicate_count=0).evaluate(manifest(
            streams={"head": {"frame_count": 3, "duplicate_count": 1}}
        ))
        self.assertTrue(
            any("duplicates_exceed_threshold" in reason for reason in report["reason"])
        )

    def test_complete_capture_health_is_fail_closed(self):
        value = manifest(
            status="MATERIALIZED",
            metadata=_complete_capture_metadata(),
            quality={
                "producer_gap_count": 0,
                "transport_gap_count": 0,
                "selection_gap_count": 0,
                "camera_camera_skew_sec": 0.01,
                "state_camera_skew_sec": 0.01,
                "recording": {
                    "recording_failed": True,
                    "timer_deadline_misses": 2,
                    "state_age_max_sec": 1.0,
                    "camera_age_max_sec": 1.0,
                },
            },
        )
        report = EpisodeQualityGate(
            require_complete_capture=True,
            max_state_age_sec=0.2,
            max_camera_age_sec=0.2,
            max_timer_deadline_misses=0,
        ).evaluate(value)

        self.assertEqual(report["status"], "REJECT")
        self.assertIn("recording_failed", report["reason"])
        self.assertTrue(
            any("timer_deadline_misses_exceed_threshold" in reason for reason in report["reason"])
        )
        self.assertTrue(any("state_age_exceed_threshold" in reason for reason in report["reason"]))
        self.assertTrue(any("camera_age_exceed_threshold" in reason for reason in report["reason"]))

    def test_complete_capture_requires_binding_and_snapshot_evidence(self):
        value = manifest(
            status="MATERIALIZED",
            metadata=_complete_capture_metadata(attached=False),
            quality={
                "producer_gap_count": 0,
                "transport_gap_count": 0,
                "selection_gap_count": 0,
                "camera_camera_skew_sec": 0.01,
                "state_camera_skew_sec": 0.01,
                "recording": {
                    "recording_failed": False,
                    "timer_deadline_misses": 0,
                    "state_age_max_sec": 0.0,
                    "camera_age_max_sec": 0.0,
                },
            },
        )
        value["metadata"]["camera_capture_sources"][0]["source_snapshot_consistent"] = False
        report = EpisodeQualityGate(require_complete_capture=True).evaluate(value)

        self.assertEqual(report["status"], "REJECT")
        self.assertIn("camera_capture_not_attached", report["reason"])
        self.assertIn("camera_capture_binding_incomplete", report["reason"])
        self.assertIn("camera_capture_snapshot_inconsistent: 0", report["reason"])

    def test_complete_capture_reconciles_all_observed_sessions(self):
        value = manifest(
            status="MATERIALIZED",
            metadata=_complete_capture_metadata(),
            quality={
                "camera_camera_skew_sec": 0.01,
                "state_camera_skew_sec": 0.01,
                "recording": {
                    "recording_failed": False,
                    "timer_deadline_misses": 0,
                    "state_age_max_sec": 0.0,
                    "camera_age_max_sec": 0.0,
                },
            },
        )
        value["metadata"]["camera_capture_binding"]["observed_session_ids"] = [
            "camera-session",
            "camera-session-restarted",
        ]
        value["metadata"]["camera_capture_binding"][
            "unbound_observed_session_ids"
        ] = ["camera-session-restarted"]

        report = EpisodeQualityGate(require_complete_capture=True).evaluate(value)

        self.assertEqual(report["status"], "REJECT")
        self.assertIn(
            "camera_capture_observed_session_unbound: camera-session-restarted",
            report["reason"],
        )

    def test_complete_capture_requires_multiple_clock_mapping_samples(self):
        value = manifest(
            status="RAW_CLOSED",
            metadata=_complete_capture_metadata(),
            quality={
                "camera_camera_skew_sec": 0.01,
                "state_camera_skew_sec": 0.01,
                "recording": {
                    "recording_failed": False,
                    "timer_deadline_misses": 0,
                    "state_age_max_sec": 0.0,
                    "camera_age_max_sec": 0.0,
                },
            },
        )
        value["metadata"]["camera_capture_sources"][0][
            "clock_mapping_samples"
        ] = 1

        report = EpisodeQualityGate(require_complete_capture=True).evaluate(value)

        self.assertEqual(report["status"], "REVIEW")
        self.assertIn("clock_mapping_samples_insufficient: 0", report["reason"])

    def test_complete_capture_rejects_snapshot_behind_observed_watermark(self):
        value = manifest(
            status="MATERIALIZED",
            metadata=_complete_capture_metadata(),
            quality={
                "producer_gap_count": 0,
                "transport_gap_count": 0,
                "selection_gap_count": 0,
                "camera_camera_skew_sec": 0.01,
                "state_camera_skew_sec": 0.01,
                "recording": {
                    "recording_failed": False,
                    "timer_deadline_misses": 0,
                    "state_age_max_sec": 0.0,
                    "camera_age_max_sec": 0.0,
                },
            },
        )
        value["metadata"]["camera_capture_sources"][0][
            "observed_stream_high_watermarks"
        ]["head"] = 3
        report = EpisodeQualityGate(require_complete_capture=True).evaluate(value)

        self.assertEqual(report["status"], "REJECT")
        self.assertIn(
            "camera_capture_observed_high_watermark_ahead: 0:head",
            report["reason"],
        )

    def test_clock_mapping_uncertainty_is_reviewed_when_unreported_or_too_large(self):
        missing = manifest(
            status="RAW_CLOSED",
            metadata=_complete_capture_metadata(),
            quality={
                "camera_camera_skew_sec": 0.01,
                "state_camera_skew_sec": 0.01,
                "recording": {
                    "recording_failed": False,
                    "timer_deadline_misses": 0,
                    "state_age_max_sec": 0.0,
                    "camera_age_max_sec": 0.0,
                }
            },
        )
        missing["metadata"].pop("capture_config")
        missing["metadata"]["camera_capture_sources"][0].pop(
            "clock_mapping_uncertainty_sec"
        )
        missing_report = EpisodeQualityGate(require_complete_capture=True).evaluate(
            missing
        )
        self.assertEqual(missing_report["status"], "REVIEW")
        self.assertIn(
            "clock_mapping_uncertainty_threshold_unreported",
            missing_report["reason"],
        )
        self.assertIn(
            "clock_mapping_uncertainty_unreported: 0",
            missing_report["reason"],
        )

        exceeded = manifest(
            status="RAW_CLOSED",
            metadata=_complete_capture_metadata(),
            quality={
                "camera_camera_skew_sec": 0.01,
                "state_camera_skew_sec": 0.01,
                "recording": {
                    "recording_failed": False,
                    "timer_deadline_misses": 0,
                    "state_age_max_sec": 0.0,
                    "camera_age_max_sec": 0.0,
                }
            },
        )
        exceeded["metadata"]["camera_capture_sources"][0][
            "clock_mapping_uncertainty_sec"
        ] = 0.2
        exceeded_report = EpisodeQualityGate(require_complete_capture=True).evaluate(
            exceeded
        )
        self.assertEqual(exceeded_report["status"], "REVIEW")
        self.assertTrue(
            any(
                "clock_mapping_uncertainty_exceed_threshold" in reason
                for reason in exceeded_report["reason"]
            )
        )

    def test_stream_gap_cannot_be_masked_by_smaller_aggregate(self):
        report = EpisodeQualityGate(max_producer_gaps=0).evaluate(
            manifest(
                streams={
                    "head": {
                        "frame_count": 3,
                        "producer_gap_count": 2,
                    }
                },
                quality={
                    "producer_gaps": 0,
                    "transport_gaps": 0,
                    "selection_gaps": 0,
                    "camera_camera_skew_sec": 0.01,
                    "state_camera_skew_sec": 0.01,
                },
            )
        )

        self.assertEqual(report["statistics"]["producer_gaps"], 2)
        self.assertTrue(
            any("producer_gaps_exceed_threshold" in reason for reason in report["reason"])
        )


if __name__ == "__main__":
    unittest.main()
