import json
import tempfile
import unittest
from pathlib import Path

from robo_collector.raw_episode import (
    CorruptChunk,
    RawEpisodeReader,
    RawEpisodeRecorder,
    create_materialization_job,
    discard_sealed_episode,
    scan_startup,
)


class RawEpisodeTest(unittest.TestCase):
    def provenance(self, sequence=1):
        return {
            "sequence": sequence,
            "clock_domain": "camera:test",
            "device_timestamp": float(sequence),
            "session_id": "session-test",
        }

    def test_round_trip_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep", source_scope="camera_capture", max_records_per_chunk=1)
            recorder.append_camera("head", b"jpeg", self.provenance(), payload_encoding="image/jpeg")
            recorder.append_robot_state({"q": [1]}, self.provenance())
            recorder.append_event({"kind": "stop"}, self.provenance())
            self.assertEqual(recorder.close()["status"], "RAW_CLOSED")
            self.assertTrue((Path(directory) / "ep" / "checksums.json").exists())
            reader = RawEpisodeReader(Path(directory) / "ep")
            reader.validate()
            self.assertEqual(next(reader.records("camera", "head"))["payload"], "anBlZw==")

    def test_rejected_records_are_counted_without_corrupting_stream_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            with self.assertRaises(ValueError):
                recorder.append_camera("head", b"x", {"sequence": 0})
            recorder.append_camera("head", b"ok", self.provenance(0))
            manifest = recorder.close()

            self.assertEqual(manifest["record_errors"]["total"], 1)
            self.assertEqual(manifest["record_errors"]["by_stream"]["head"], 1)
            self.assertEqual(
                manifest["record_errors"]["by_error_type"]["ValueError"], 1
            )
            RawEpisodeReader(Path(directory) / "ep").validate()

    def test_extra_metadata_cannot_override_core_record_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            with self.assertRaisesRegex(
                ValueError,
                r"extra cannot override raw record fields: episode_id,sequence,server_wall_timestamp",
            ):
                recorder.append_camera(
                    "head",
                    b"bad",
                    self.provenance(1),
                    episode_id="other-episode",
                    sequence=99,
                    server_wall_timestamp=999.0,
                )

            recorder.append_camera(
                "head",
                b"ok",
                self.provenance(1),
                payload_encoding="image/jpeg",
            )
            manifest = recorder.close()
            record = next(
                RawEpisodeReader(Path(directory) / "ep").records("camera", "head")
            )

            self.assertEqual(record["episode_id"], "ep")
            self.assertEqual(record["sequence"], 1)
            self.assertEqual(record["device_timestamp"], 1.0)
            self.assertEqual(manifest["record_errors"]["total"], 1)

    def test_truncated_tail_recovers_open_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            chunk = Path(directory) / "ep" / "camera" / "head" / "chunk-000000.raw"
            with chunk.open("ab") as stream:
                stream.write(b"bad")
            recorder._files["camera/head"][0].close()
            recorder._files.clear()
            scan_startup(directory)
            manifest = json.loads((Path(directory) / "ep" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "RAW_CLOSED")
            self.assertEqual(manifest["termination"]["reason"], "process_crash")
            self.assertEqual(len(list(RawEpisodeReader.read_chunk(chunk))), 1)

    def test_bad_complete_prefix_is_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            chunk = Path(directory) / "ep" / "camera" / "head" / "chunk-000000.raw"
            recorder._files["camera/head"][0].close()
            recorder._files.clear()
            data = bytearray(chunk.read_bytes())
            data[-1] ^= 1
            chunk.write_bytes(data)
            scan_startup(directory)
            self.assertEqual(
                json.loads((Path(directory) / "ep" / "manifest.json").read_text())["status"],
                "QUARANTINED",
            )

    def test_recovery_and_job_idempotency(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            recorder._files["camera/head"][0].close()
            recorder._files.clear()
            scan_startup(directory)
            path = Path(directory) / "ep"
            self.assertEqual(json.loads((path / "manifest.json").read_text())["status"], "RAW_CLOSED")
            raw_hash = json.loads((path / "manifest.json").read_text())["raw_manifest_hash"]
            first = create_materialization_job(path, {"fps": 30}, "v1")
            second = create_materialization_job(path, {"fps": 30}, "v1")
            self.assertEqual(first["job_id"], second["job_id"])
            self.assertEqual(first["output_target"], {"episode_id": "ep"})
            self.assertEqual(
                json.loads((path / "manifest.json").read_text())["raw_manifest_hash"],
                raw_hash,
            )
            self.assertEqual(first["conversion_config"], {"fps": 30})
            RawEpisodeReader(path).validate()

    def test_closed_episode_without_job_gets_recovered_materialization_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "output_root": str(root / "derived"),
                "dataset_name": "dataset",
                "fps": 30,
                "camera_streams": ["head"],
                "alignment_policy": "strict",
                "require_complete_capture": False,
            }
            recorder = RawEpisodeRecorder(
                root / "raw",
                "ep",
                metadata={"conversion_config": config, "output_schema_version": "schema-1"},
            )
            recorder.append_camera("head", b"x", self.provenance())
            recorder.append_robot_state({"q": [1]}, self.provenance())
            recorder.close()

            jobs = scan_startup(root / "raw")

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["status"], "PENDING")
            self.assertEqual(jobs[0]["conversion_config"], config)
            manifest = json.loads((root / "raw" / "ep" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "RAW_CLOSED")
            self.assertEqual(len(manifest["materialization_jobs"]), 1)

    def test_startup_scan_repairs_missing_quality_report(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = Path(directory) / "ep"

            self.assertFalse((path / "quality.json").exists())
            scan_startup(directory)

            report = json.loads((path / "quality.json").read_text())
            self.assertEqual(report["report_schema"], "robo_collector.episode_quality.v1")
            self.assertTrue(report["recovery"]["generated_at_startup"])
            self.assertEqual(report["status"], "REJECT")

    def test_startup_scan_repairs_invalid_quality_report(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = Path(directory) / "ep"
            (path / "quality.json").write_text("not-json", encoding="utf-8")

            scan_startup(directory)

            report = json.loads((path / "quality.json").read_text())
            self.assertEqual(report["report_schema"], "robo_collector.episode_quality.v1")

    def test_startup_scan_repairs_structurally_invalid_quality_report(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = Path(directory) / "ep"
            (path / "quality.json").write_text(
                json.dumps({"status": "READY"}), encoding="utf-8"
            )

            scan_startup(directory)

            report = json.loads((path / "quality.json").read_text())
            self.assertEqual(report["report_schema"], "robo_collector.episode_quality.v1")
            self.assertTrue(report["recovery"]["generated_at_startup"])

    def test_pending_job_does_not_claim_materialization_is_running(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = Path(directory) / "ep"

            job = create_materialization_job(path, {"fps": 30}, "v1")

            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(job["status"], "PENDING")
            self.assertEqual(manifest["status"], "RAW_CLOSED")

    def test_materialized_and_qc_jobs_are_returned_for_revalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(
                directory,
                "ep",
                metadata={
                    "conversion_config": {"fps": 30},
                    "output_schema_version": "v1",
                },
            )
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = Path(directory) / "ep"
            job = create_materialization_job(path, {"fps": 30}, "v1")
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "MATERIALIZED"
            manifest["materialization_jobs"][0]["status"] = "MATERIALIZED"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            recovered = scan_startup(directory)

            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["job_id"], job["job_id"])
            self.assertEqual(recovered[0]["recovery_action"], "revalidate_materialized")

            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "QC"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            recovered = scan_startup(directory)
            self.assertEqual(recovered[0]["recovery_action"], "revalidate_materialized")

    def test_qc_without_job_gets_a_pending_recovery_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "output_root": str(root / "derived"),
                "dataset_name": "dataset",
                "fps": 30,
                "camera_streams": ["head"],
                "alignment_policy": "strict",
            }
            recorder = RawEpisodeRecorder(
                root / "raw",
                "ep",
                metadata={
                    "conversion_config": config,
                    "output_schema_version": "v1",
                },
            )
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = root / "raw" / "ep"
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "QC"
            manifest["materialization_jobs"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            recovered = scan_startup(root / "raw")

            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["status"], "PENDING")
            self.assertEqual(
                json.loads(manifest_path.read_text())["status"], "QC"
            )

    def test_materializing_without_job_is_repaired_to_raw_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"output_root": str(root / "derived"), "fps": 30}
            recorder = RawEpisodeRecorder(
                root / "raw",
                "ep",
                metadata={
                    "conversion_config": config,
                    "output_schema_version": "v1",
                },
            )
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = root / "raw" / "ep"
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "MATERIALIZING"
            manifest["materialization_jobs"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            recovered = scan_startup(root / "raw")

            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["status"], "PENDING")
            self.assertEqual(
                json.loads(manifest_path.read_text())["status"], "RAW_CLOSED"
            )

    def test_pending_job_repairs_torn_materializing_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = Path(directory) / "ep"
            job = create_materialization_job(path, {"fps": 30}, "v1")
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "MATERIALIZING"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            recovered = scan_startup(directory)

            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["job_id"], job["job_id"])
            self.assertEqual(
                json.loads(manifest_path.read_text())["status"], "RAW_CLOSED"
            )

    def test_running_job_is_reset_without_incrementing_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = Path(directory) / "ep"
            job = create_materialization_job(path, {"fps": 30}, "v1", output_target="exports/ep")
            from robo_collector.raw_episode import claim_materialization_job
            claimed = claim_materialization_job(path, job["job_id"])
            self.assertEqual(claimed["attempt_count"], 1)
            scan_startup(directory)
            recovered = json.loads((path / "manifest.json").read_text())["materialization_jobs"][0]
            self.assertEqual(recovered["status"], "PENDING")
            self.assertEqual(recovered["attempt_count"], 1)
            self.assertEqual(recovered["output_target"], "exports/ep")

    def test_final_manifest_wins_over_stale_open_progress_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = Path(directory) / "ep"
            final_before = (path / "manifest.json").read_bytes()
            final_manifest = json.loads(final_before)
            final_manifest["final_marker"] = "authoritative"
            # Keep the final copy valid after adding the marker.
            from robo_collector.raw_episode import _hash_manifest
            final_manifest["raw_manifest_hash"] = _hash_manifest(final_manifest)
            final_manifest["manifest_hash"] = final_manifest["raw_manifest_hash"]
            (path / "manifest.json").write_text(
                json.dumps(final_manifest), encoding="utf-8"
            )
            old_progress = dict(final_manifest)
            old_progress["status"] = "OPEN"
            old_progress["final_marker"] = "stale-progress"
            (path / "manifest.inprogress.json").write_text(
                json.dumps(old_progress), encoding="utf-8"
            )

            scan_startup(directory)

            after = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(after["status"], "RAW_CLOSED")
            self.assertEqual(after["final_marker"], "authoritative")
            self.assertFalse((path / "manifest.inprogress.json").exists())

    def test_invalid_final_manifest_does_not_fallback_to_old_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = Path(directory) / "ep"
            final_manifest = json.loads((path / "manifest.json").read_text())
            final_manifest["final_marker"] = "invalid-final"
            final_manifest["raw_manifest_hash"] = "not-the-final-hash"
            (path / "manifest.json").write_text(
                json.dumps(final_manifest), encoding="utf-8"
            )
            old_progress = dict(final_manifest)
            old_progress["status"] = "OPEN"
            old_progress["final_marker"] = "old-progress"
            (path / "manifest.inprogress.json").write_text(
                json.dumps(old_progress), encoding="utf-8"
            )

            scan_startup(directory)

            after = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(after["status"], "QUARANTINED")
            self.assertEqual(after["final_marker"], "invalid-final")
            self.assertNotEqual(after["final_marker"], "old-progress")

    def test_discard_sealed_episode_preserves_raw_and_rejects_unsafe_target(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = Path(directory) / "ep"
            result = discard_sealed_episode(path, reason="bad sample")
            self.assertEqual(result["status"], "DISCARDED")
            self.assertTrue((path / "camera/head/chunk-000000.raw").exists())
            with self.assertRaises(ValueError):
                create_materialization_job(path, {"fps": 30}, "v1", output_target="../escape")

    def test_discarded_episode_survives_startup_scan_without_quarantine(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance())
            recorder.close()
            path = Path(directory) / "ep"
            before = json.loads((path / "manifest.json").read_text())

            discard_sealed_episode(path, reason="operator rejected sample")
            recovered_jobs = scan_startup(directory)

            after = json.loads((path / "manifest.json").read_text())
            self.assertEqual(recovered_jobs, [])
            self.assertEqual(after["status"], "DISCARDED")
            self.assertEqual(after["discard_reason"], "operator rejected sample")
            self.assertEqual(after["raw_manifest_hash"], before["raw_manifest_hash"])
            self.assertTrue((path / "camera/head/chunk-000000.raw").exists())
            RawEpisodeReader(path).validate()

    def test_scope_and_record_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                RawEpisodeRecorder(directory, "ep", source_scope="latest")
            recorder = RawEpisodeRecorder(directory, "ep", max_record_bytes=2)
            with self.assertRaises(ValueError):
                recorder.append_camera("head", b"large", self.provenance())
            missing_session = self.provenance()
            missing_session.pop("session_id")
            with self.assertRaisesRegex(ValueError, "session_id"):
                recorder.append_camera("head", b"x", missing_session)
            recorder.discard(reason="record rejected")

    def test_gap_classes_are_persisted_and_balanced_per_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawEpisodeRecorder(directory, "ep")
            recorder.append_camera("head", b"x", self.provenance(sequence=0))
            recorder.append_camera(
                "head",
                b"y",
                self.provenance(sequence=5),
                producer_gap_count=1,
                publisher_gap_count=2,
                transport_gap_count=1,
                unattributed_gap_count=0,
            )
            manifest = recorder.close()
            stats = manifest["streams"]["head"]
            self.assertEqual(stats["sequence_gap_count"], 4)
            self.assertEqual(stats["producer_gap_count"], 1)
            self.assertEqual(stats["publisher_gap_count"], 2)
            self.assertEqual(stats["transport_gap_count"], 1)
            self.assertEqual(stats["unattributed_gap_count"], 0)
