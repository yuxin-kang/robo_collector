import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from robo_collector_camera.raw_spool import (
    RawSpool,
    read_camera_spool_snapshot,
    read_camera_spool_status,
)
from robo_collector_camera.server_realsense import RealSenseReader


class RawSpoolFenceTest(unittest.TestCase):
    def test_start_and_stop_fences_bind_durable_watermarks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spool = RawSpool(root, stream="head", session_id="camera-session")

            initial = read_camera_spool_status(root)
            self.assertEqual(
                initial["streams"]["head"]["start_fence"]["event"], "START"
            )
            self.assertIsNone(initial["streams"]["head"]["stop_fence"])
            self.assertFalse(initial["source_complete"])

            spool.append({"payload": b"zero", "sequence": 7})
            live_manifest, _ = read_camera_spool_snapshot(root)
            live = live_manifest["_snapshot"]["stream_high_watermarks"]["head"]
            self.assertEqual(live["accepted_count"], 1)
            self.assertEqual(live["written_count"], 1)
            self.assertEqual(live["durable_count"], 1)
            self.assertEqual(live["durable_high_watermark"], 7)
            self.assertFalse(live_manifest["_snapshot"]["source_complete"])

            spool.close(reason="process_stop")

            final = read_camera_spool_status(root)
            stream = final["streams"]["head"]
            self.assertEqual(stream["stop_fence"]["event"], "STOP")
            self.assertEqual(stream["stop_fence"]["generation"], stream["generation"])
            self.assertEqual(stream["stop_fence"]["durable_count"], 1)
            self.assertEqual(stream["stop_fence"]["durable_high_watermark"], 7)
            self.assertTrue(stream["source_complete"])
            self.assertTrue(final["source_complete"])

    def test_deferred_durability_is_visible_in_live_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spool = RawSpool(
                root,
                stream="head",
                manifest_checkpoint_records=100,
                manifest_checkpoint_interval_sec=3600.0,
                durability_interval_sec=3600.0,
            )

            spool.append({"payload": b"pending", "sequence": 4})

            live = spool.status()
            self.assertEqual(live["accepted_count"], 1)
            self.assertEqual(live["written_count"], 1)
            self.assertEqual(live["durable_count"], 0)
            persisted = read_camera_spool_status(root)["streams"]["head"]
            self.assertEqual(persisted["durable_count"], 0)
            snapshot, records = read_camera_spool_snapshot(root)
            self.assertEqual(records, [])
            self.assertEqual(snapshot["_snapshot"]["record_count"], 0)

            spool.close()
            final = read_camera_spool_status(root)["streams"]["head"]
            self.assertEqual(final["durable_count"], 1)
            self.assertEqual(final["durable_high_watermark"], 4)

    def test_restart_generation_rebinds_start_fence(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = RawSpool(directory, stream="head")
            first = spool.status()

            spool.mark_restart()

            restarted = spool.status()
            self.assertEqual(restarted["generation"], first["generation"] + 1)
            self.assertEqual(restarted["restart_count"], 1)
            self.assertEqual(
                restarted["start_fence"]["generation"], restarted["generation"]
            )
            self.assertEqual(restarted["start_fence"]["restart_count"], 1)

    def test_failed_close_never_implies_stop_or_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = RawSpool(directory, stream="head")
            spool.append({"payload": b"frame", "sequence": 0})
            original = spool._write_manifest

            def fail_final_manifest(*, include_checksums=False, force=True):
                if include_checksums:
                    raise OSError("disk unavailable")
                return original(include_checksums=include_checksums, force=force)

            with (
                mock.patch.object(spool, "_write_manifest", fail_final_manifest),
                self.assertRaisesRegex(OSError, "disk unavailable"),
            ):
                spool.close()

            status = spool.status()
            self.assertFalse(status["source_complete"])
            self.assertIsNone(status["stop_fence"])
            self.assertIn("disk unavailable", status["close_error"])


class RealSenseReaderFenceTest(unittest.TestCase):
    def test_start_fence_is_persisted_before_pipeline_start(self):
        events = []
        spool = SimpleNamespace(mark_start=lambda: events.append("spool.START"))
        reader = object.__new__(RealSenseReader)
        reader.raw_spool = spool
        reader.depth_raw_spool = None
        reader.pipeline = SimpleNamespace(
            start=lambda config: events.append("pipeline.start") or object()
        )
        reader.config = object()
        reader.spec = SimpleNamespace(stream="head")
        reader._run = lambda: None
        reader._started = False
        reader._pipeline_stop_succeeded = False

        thread = SimpleNamespace(start=lambda: events.append("thread.start"))
        with (
            mock.patch(
                "robo_collector_camera.server_realsense.get_device_info",
                return_value={},
            ),
            mock.patch(
                "robo_collector_camera.server_realsense.threading.Thread",
                return_value=thread,
            ),
        ):
            reader.start()

        self.assertEqual(events, ["spool.START", "pipeline.start", "thread.start"])

    def test_pipeline_stop_failure_withholds_clean_stop_boundary(self):
        reader = object.__new__(RealSenseReader)
        reader.spec = SimpleNamespace(stream="head")
        reader._stop = threading.Event()
        reader._started = True
        reader._pipeline_stop_succeeded = False
        reader.pipeline = SimpleNamespace(
            stop=mock.Mock(side_effect=RuntimeError("pipeline failed"))
        )
        reader._thread = SimpleNamespace(
            join=lambda timeout: None, is_alive=lambda: False
        )

        with self.assertRaisesRegex(RuntimeError, "failed to stop RealSense pipeline"):
            reader.stop()

        self.assertFalse(reader.capture_stopped_cleanly)


if __name__ == "__main__":
    unittest.main()
