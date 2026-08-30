import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from robo_collector_camera.raw_spool import (
    RawSpool,
    read_camera_spool_snapshot,
    scan_camera_spools,
)
from robo_collector_camera.server_realsense import (
    RealSenseReader,
    timestamp_domain_name,
)


class CameraRawCaptureSpecAcceptanceTest(unittest.TestCase):
    def test_realsense_timestamp_domain_is_preserved_as_explicit_metadata(self):
        class Domain:
            name = "HARDWARE_CLOCK"

        class Frame:
            def get_frame_timestamp_domain(self):
                return Domain()

        self.assertEqual(timestamp_domain_name(Frame()), "hardware_clock")
        self.assertEqual(timestamp_domain_name(object()), "unknown")

    def test_recording_queue_is_bounded_and_overflow_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = RawSpool(directory, stream="head", max_records=2)

            self.assertTrue(spool.append({"payload": b"one", "sequence": 0}))
            self.assertTrue(spool.append({"payload": b"two", "sequence": 1}))
            self.assertFalse(spool.append({"payload": b"three", "sequence": 2}))

            manifest = json.loads(
                (Path(directory) / "manifest.inprogress.json").read_text()
            )
            self.assertEqual(manifest["streams"]["head"]["frame_count"], 2)
            self.assertEqual(manifest["events"]["spool_records"], 2)
            self.assertEqual(manifest["record_errors"]["total"], 1)
            self.assertEqual(
                manifest["record_errors"]["by_error_type"]["spool_overflow"],
                1,
            )
            self.assertEqual([item["sequence"] for item in spool.recover()], [0, 1])

    def test_checkpointed_spool_flushes_pending_records_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spool = RawSpool(
                root,
                stream="head",
                manifest_checkpoint_records=100,
                manifest_checkpoint_interval_sec=3600.0,
                durability_interval_sec=3600.0,
            )
            self.assertTrue(spool.append({"payload": b"raw", "sequence": 0}))

            progress = json.loads(
                (root / "manifest.inprogress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["streams"]["head"]["frame_count"], 0)

            spool.close(reason="operator_stop")

            final = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(final["streams"]["head"]["frame_count"], 1)
            self.assertIn("camera/head/chunk-000000.msgpack", final["checksums"])
            self.assertEqual(
                list(spool.recover()), [{"payload": b"raw", "sequence": 0}]
            )

    def test_corrupt_manifest_is_fail_closed_and_quarantine_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            RawSpool(root / "direct", stream="head")
            corrupt = root / "direct" / "manifest.inprogress.json"
            corrupt_bytes = b"{not-json"
            corrupt.write_bytes(corrupt_bytes)

            with self.assertRaisesRegex(ValueError, "corrupt raw spool manifest"):
                RawSpool(root / "direct", stream="head")
            self.assertEqual(corrupt.read_bytes(), corrupt_bytes)

            broken = root / "scan"
            broken.mkdir()
            broken_manifest = broken / "manifest.inprogress.json"
            broken_bytes = b"also-not-json"
            broken_manifest.write_bytes(broken_bytes)

            self.assertEqual(scan_camera_spools(root), [])
            self.assertEqual(
                (broken / "manifest.corrupt.json").read_bytes(), broken_bytes
            )
            quarantine = json.loads(
                (broken / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(quarantine["status"], "QUARANTINED")
            self.assertFalse(broken_manifest.exists())

    def test_reader_stops_pipeline_before_joining_capture_thread(self):
        events = []
        reader = object.__new__(RealSenseReader)
        reader.spec = SimpleNamespace(stream="head")
        reader._stop = threading.Event()
        reader._started = True

        class Pipeline:
            def stop(self):
                events.append("pipeline.stop")

        class Thread:
            def join(self, timeout):
                events.append(("thread.join", timeout))

            def is_alive(self):
                return False

        reader.pipeline = Pipeline()
        reader._thread = Thread()

        reader.stop()

        self.assertEqual(events, ["pipeline.stop", ("thread.join", 2.0)])

    def test_strict_source_spool_counts_rejected_records_durably(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = RawSpool(directory, stream="head", strict_records=True)
            with self.assertRaisesRegex(ValueError, "invalid_server_wall_timestamp"):
                spool.append(
                    {
                        "stream": "head",
                        "session_id": spool.session_id,
                        "sequence": 0,
                        "payload": b"jpeg",
                    }
                )

            manifest = json.loads(
                (Path(directory) / "manifest.inprogress.json").read_text()
            )
            self.assertEqual(manifest["record_errors"]["total"], 1)
            self.assertEqual(
                manifest["record_errors"]["by_error_type"][
                    "invalid_server_wall_timestamp"
                ],
                1,
            )

    def test_strict_source_spool_requires_record_identity_and_clock_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = RawSpool(directory, stream="head", strict_records=True)
            base = {
                "stream": "head",
                "session_id": spool.session_id,
                "sequence": 0,
                "payload": b"jpeg",
                "server_wall_timestamp": 1.0,
                "clock_domain": "realsense:head",
            }
            for field, message in (
                ("stream", "missing_stream"),
                ("session_id", "missing_session_id"),
                ("clock_domain", "missing_clock_domain"),
            ):
                record = dict(base)
                record.pop(field)
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    spool.append(record)

            manifest = json.loads(
                (Path(directory) / "manifest.inprogress.json").read_text()
            )
            self.assertEqual(manifest["record_errors"]["total"], 3)

    def test_strict_source_spool_counts_serialization_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = RawSpool(directory, stream="head", strict_records=True)
            with self.assertRaisesRegex(ValueError, "cannot be serialized"):
                spool.append(
                    {
                        "stream": "head",
                        "session_id": spool.session_id,
                        "sequence": 0,
                        "payload": b"jpeg",
                        "server_wall_timestamp": 1.0,
                        "clock_domain": "realsense:head",
                        "unsupported": object(),
                    }
                )

            manifest = json.loads(
                (Path(directory) / "manifest.inprogress.json").read_text()
            )
            self.assertEqual(
                manifest["record_errors"]["by_error_type"]["serialization_error"],
                1,
            )

    def test_spool_records_recover_after_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            first = RawSpool(directory, stream="head", max_records=4)
            first.append({"payload": b"raw", "sequence": 7})

            reopened = RawSpool(directory, stream="head", max_records=4)

            self.assertEqual(
                list(reopened.recover()), [{"payload": b"raw", "sequence": 7}]
            )
            self.assertTrue(reopened.append({"payload": b"next", "sequence": 8}))

    def test_gap_classes_and_restart_counters_are_persisted_per_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = RawSpool(directory, stream="head", chunk_records=1)
            spool.mark_restart()
            self.assertTrue(spool.append({"payload": b"zero", "sequence": 0}))
            self.assertTrue(
                spool.append(
                    {"payload": b"two", "sequence": 2, "producer_gap_count": 1}
                )
            )
            self.assertTrue(
                spool.append(
                    {"payload": b"four", "sequence": 4, "producer_gap_count": 0}
                )
            )
            spool.mark_sent(3)
            spool.close(reason="operator_stop")

            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            stream = manifest["streams"]["head"]
            self.assertEqual(manifest["status"], "RAW_CLOSED")
            self.assertEqual(stream["frame_count"], 3)
            self.assertEqual(stream["sequence_gap_count"], 2)
            self.assertEqual(stream["producer_gap_count"], 1)
            self.assertEqual(stream["transport_gap_count"], 1)
            self.assertEqual(stream["events"]["restart"], 1)
            self.assertEqual(stream["events"]["sent"], 3)
            self.assertIn(
                "camera/head/chunk-000000.msgpack",
                manifest["checksums"],
            )
            self.assertFalse((Path(directory) / "manifest.inprogress.json").exists())
            with self.assertRaisesRegex(RuntimeError, "closed"):
                spool.append({"payload": b"late", "sequence": 5})

    def test_interrupted_session_is_sealed_and_nonfinal_corruption_is_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interrupted = root / "session-ok"
            spool = RawSpool(interrupted, stream="head")
            spool.append({"payload": b"raw", "sequence": 0})

            recovered = scan_camera_spools(root)

            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["status"], "RAW_INCOMPLETE")
            self.assertFalse(recovered[0]["source_complete"])
            self.assertIsNone(recovered[0]["streams"]["head"]["stop_fence"])
            self.assertEqual(
                recovered[0]["termination"],
                {"reason": "closed_without_source_stop"},
            )

            damaged = root / "session-damaged"
            damaged_spool = RawSpool(damaged, stream="head", chunk_records=1)
            damaged_spool.append({"payload": b"first", "sequence": 0})
            damaged_spool.append({"payload": b"second", "sequence": 1})
            first_chunk = damaged / "camera" / "head" / "chunk-000000.msgpack"
            contents = bytearray(first_chunk.read_bytes())
            contents[-1] ^= 1
            first_chunk.write_bytes(contents)

            scan_camera_spools(root)

            damaged_manifest = json.loads(
                (damaged / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(damaged_manifest["status"], "QUARANTINED")
            self.assertIn(
                "corrupt non-final spool chunk", damaged_manifest["quarantine_error"]
            )
            self.assertFalse((damaged / "manifest.inprogress.json").exists())

    def test_corrupt_final_frame_is_quarantined_but_truncated_tail_is_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            damaged = root / "session-damaged-final"
            spool = RawSpool(damaged, stream="head")
            spool.append({"payload": b"first", "sequence": 0})
            chunk = damaged / "camera" / "head" / "chunk-000000.msgpack"
            contents = bytearray(chunk.read_bytes())
            contents[-1] ^= 1
            chunk.write_bytes(contents)

            scan_camera_spools(root)

            damaged_manifest = json.loads(
                (damaged / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(damaged_manifest["status"], "QUARANTINED")
            self.assertIn("corrupt spool chunk", damaged_manifest["quarantine_error"])

            repaired = root / "session-truncated-final"
            repaired_spool = RawSpool(repaired, stream="head")
            repaired_spool.append({"payload": b"first", "sequence": 0})
            repaired_session_id = repaired_spool.session_id
            repaired_chunk = repaired / "camera" / "head" / "chunk-000000.msgpack"
            with repaired_chunk.open("ab") as handle:
                handle.write(b"tail")

            recovered = scan_camera_spools(root)

            self.assertTrue(
                any(item["episode_id"] == repaired_session_id for item in recovered)
            )
            repaired_manifest = next(
                item for item in recovered if item["episode_id"] == repaired_session_id
            )
            self.assertEqual(
                repaired_manifest["streams"]["head"]["events"]["corrupt_tail_bytes"],
                4,
            )
            self.assertEqual(
                list(RawSpool(repaired, stream="head").recover()),
                [{"payload": b"first", "sequence": 0}],
            )

    def test_live_snapshot_reads_verified_prefix_without_sealing_session(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session-live"
            spool = RawSpool(session, stream="head")
            spool.append(
                {
                    "stream": "head",
                    "sequence": 3,
                    "payload": b"jpeg",
                    "server_wall_timestamp": 10.0,
                    "payload_encoding": "image/jpeg",
                }
            )
            chunk = session / "camera" / "head" / "chunk-000000.msgpack"
            with chunk.open("ab") as handle:
                handle.write(b"tail")

            manifest, records = read_camera_spool_snapshot(session)

            self.assertEqual(manifest["status"], "RAW_IN_PROGRESS")
            self.assertEqual(manifest["schema"], "robo_collector.camera_spool.v1")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["session_id"], manifest["session_id"])
            self.assertEqual(records[0]["payload"], b"jpeg")
            self.assertTrue((session / "manifest.inprogress.json").exists())
            snapshot = manifest["_snapshot"]
            self.assertEqual(snapshot["record_count"], 1)
            self.assertEqual(snapshot["selected_record_count"], 1)
            self.assertTrue(snapshot["stable"])
            self.assertEqual(
                snapshot["stream_high_watermarks"]["head"]["last_sequence"], 3
            )
            self.assertEqual(len(snapshot["chunks"]), 1)

    def test_snapshot_window_bounds_returned_records_but_keeps_prefix_watermark(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session-window"
            spool = RawSpool(session, stream="head")
            for sequence, timestamp in enumerate((1.0, 2.0, 3.0)):
                spool.append(
                    {
                        "stream": "head",
                        "sequence": sequence,
                        "payload": f"frame-{sequence}".encode(),
                        "server_wall_timestamp": timestamp,
                    }
                )

            manifest, records = read_camera_spool_snapshot(
                session, server_wall_window=(1.5, 2.5)
            )

            self.assertEqual([item["sequence"] for item in records], [1])
            snapshot = manifest["_snapshot"]
            self.assertEqual(snapshot["record_count"], 3)
            self.assertEqual(snapshot["selected_record_count"], 1)
            self.assertEqual(snapshot["selected_record_counts"]["head"], 1)
            self.assertEqual(
                snapshot["stream_high_watermarks"]["head"]["last_sequence"],
                2,
            )
            self.assertEqual(snapshot["server_wall_window"], [1.5, 2.5])
            self.assertTrue(snapshot["stable"])

    def test_snapshot_can_validate_without_retaining_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session-counters"
            spool = RawSpool(session, stream="head")
            for sequence in range(3):
                spool.append(
                    {
                        "stream": "head",
                        "sequence": sequence,
                        "payload": b"large-enough-payload",
                        "server_wall_timestamp": float(sequence),
                    }
                )

            manifest, records = read_camera_spool_snapshot(
                session, include_records=False
            )

            self.assertEqual(records, [])
            self.assertEqual(manifest["_snapshot"]["record_count"], 3)
            self.assertEqual(manifest["_snapshot"]["selected_record_count"], 3)

    def test_snapshot_high_watermark_is_maximum_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session-reordered"
            spool = RawSpool(session, stream="head")
            spool.append({"sequence": 7, "payload": b"later"})
            spool.append({"sequence": 3, "payload": b"reordered"})

            manifest, _ = read_camera_spool_snapshot(session)

            self.assertEqual(
                manifest["_snapshot"]["stream_high_watermarks"]["head"][
                    "last_sequence"
                ],
                7,
            )

    def test_closed_snapshot_rejects_tampered_chunk_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session-closed"
            spool = RawSpool(session, stream="head")
            spool.append({"payload": b"jpeg", "sequence": 0})
            spool.close(reason="operator_stop")

            closed_manifest = json.loads(
                (session / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertRegex(closed_manifest["raw_manifest_hash"], r"^[0-9a-f]{64}$")

            chunk = session / "camera" / "head" / "chunk-000000.msgpack"
            chunk.write_bytes(chunk.read_bytes() + b"tampered")

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                read_camera_spool_snapshot(session)

    def test_closed_snapshot_rejects_tampered_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session-manifest-tampered"
            spool = RawSpool(session, stream="head")
            spool.append({"payload": b"jpeg", "sequence": 0})
            spool.close(reason="operator_stop")

            manifest_path = session / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["termination"]["reason"] = "tampered"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                read_camera_spool_snapshot(session)


if __name__ == "__main__":
    unittest.main()
