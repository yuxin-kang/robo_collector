import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

from robo_collector.camera_cache import (
    CameraFrameCache,
    parse_camera_streams,
)
from robo_collector_camera.client import NormalizedCameraFrame, NormalizedPacket


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warn(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)


class CameraFrameCacheTest(unittest.TestCase):
    def test_accepts_duck_typed_envelope_and_prefers_device_timestamp(self):
        class Frame:
            sequence = 10
            timestamp_quality = "device"
            device_timestamp = 2_000
            device_unit = "ms"
            server_wall_timestamp = 999.0
            payload = "encoded"

            def decode(self):
                return "decoded-latest"

        packet = SimpleNamespace(
            session_id="session-a",
            frames={"head": Frame()},
            metadata={},
        )
        cache = CameraFrameCache("127.0.0.1", 5555, ["head"], FakeLogger())
        self.assertTrue(cache.update_from_packet(packet, received_monotonic_sec=4.0))
        self.assertEqual(cache.latest().images["head"], "decoded-latest")
        self.assertEqual(cache.latest().frames["head"].camera_timestamp_sec, 2.0)

    def test_duck_frame_falls_back_to_server_wall_timestamp_and_payload(self):
        frame = SimpleNamespace(
            sequence=1,
            timestamp_quality="host_after_capture",
            server_wall_timestamp=123.5,
            payload=b"encoded",
        )
        packet = SimpleNamespace(session_id="session-a", frames={"head": frame})
        cache = CameraFrameCache("127.0.0.1", 5555, ["head"], FakeLogger())
        self.assertTrue(cache.update_from_packet(packet))
        self.assertEqual(cache.latest().images["head"], b"encoded")
        self.assertEqual(cache.latest().frames["head"].camera_timestamp_sec, 123.5)

    def test_run_prefers_read_envelope_for_duck_transport(self):
        frame = SimpleNamespace(
            sequence=1,
            timestamp_quality="device",
            device_timestamp=3_000_000,
            device_unit="us",
            payload=b"latest",
        )
        packet = SimpleNamespace(session_id="session-a", frames={"head": frame})

        cache = CameraFrameCache("127.0.0.1", 5555, ["head"], FakeLogger(), reconnect_backoff_sec=0.01)

        class FakeClient:
            calls = []

            def __init__(self, host, port):
                self.host, self.port = host, port

            def read_envelope(self, timeout_ms):
                self.calls.append(("read_envelope", timeout_ms))
                cache._stop.set()
                return packet

            def read(self, timeout_ms):
                self.calls.append(("read", timeout_ms))
                return None

            def close(self):
                pass

        fake_module = SimpleNamespace(CameraClient=FakeClient, CameraPacketError=ValueError)
        with patch.dict(sys.modules, {"robo_collector_camera.client": fake_module}):
            cache._run()
        self.assertEqual(FakeClient.calls, [("read_envelope", 100)])
        self.assertEqual(cache.latest().frames["head"].camera_timestamp_sec, 3.0)

    def test_callback_observes_valid_packet_rejected_by_latest_selection(self):
        frame = NormalizedCameraFrame(
            stream="head",
            sequence=4,
            payload=b"raw",
            payload_encoding="image/jpeg",
            timestamp_quality="host_after_capture",
            server_wall_timestamp=4.0,
        )
        packet = NormalizedPacket(
            schema="robo_collector_camera.v4",
            session_id="session-a",
            frames={"head": frame},
        )
        cache = CameraFrameCache(
            "127.0.0.1",
            5555,
            ["head"],
            FakeLogger(),
            receive_mode="recording",
            decode_images=False,
        )
        observed = []
        cache.set_packet_callback(observed.append)

        class FakeClient:
            def __init__(self, host, port, **kwargs):
                self.packets = [packet, packet]

            def read_envelope(self, timeout_ms):
                if self.packets:
                    value = self.packets.pop(0)
                    if not self.packets:
                        cache._stop.set()
                    return value
                cache._stop.set()
                return None

            def close(self):
                pass

        fake_module = SimpleNamespace(CameraClient=FakeClient, CameraPacketError=ValueError)
        with patch.dict(sys.modules, {"robo_collector_camera.client": fake_module}):
            cache._run()
        cache._callback_run()

        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0].frames["head"].sequence, 4)
        self.assertEqual(observed[1].frames["head"].sequence, 4)
        self.assertEqual(cache.stats["duplicate"], 1)

    def test_requires_all_configured_streams_before_updating_latest(self):
        logger = FakeLogger()
        cache = CameraFrameCache("127.0.0.1", 5555, ["head", "ego_view"], logger)
        cache.update_from_packet(
            {
                "session_id": "session-a",
                "timestamps": {"head": 0.9, "ego_view": 1.0},
                "sequences": {"head": 1, "ego_view": 1},
                "images": {"head": "head-image", "ego_view": "ego-image"},
            },
            received_monotonic_sec=9.0,
        )

        updated = cache.update_from_packet(
            {
                "session_id": "session-a",
                "timestamps": {"ego_view": 1.0},
                "sequences": {"ego_view": 2},
                "images": {"ego_view": "ego-image"},
            },
            received_monotonic_sec=10.0,
        )

        self.assertFalse(updated)
        self.assertIsNone(cache.latest())
        self.assertEqual(
            logger.warnings,
            ["camera packet missing required stream(s): head"],
        )

    def test_updates_complete_multi_stream_bundle(self):
        cache = CameraFrameCache(
            "127.0.0.1", 5555, ["head", "ego_view"], FakeLogger()
        )

        updated = cache.update_from_packet(
            {
                "session_id": "session-a",
                "timestamps": {"head": 2.0, "ego_view": 2.1},
                "sequences": {"head": 20, "ego_view": 21},
                "images": {"head": "head-image", "ego_view": "ego-image"},
            },
            received_monotonic_sec=12.0,
        )

        self.assertTrue(updated)
        bundle = cache.latest()
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.images, {"head": "head-image", "ego_view": "ego-image"})
        self.assertEqual(bundle.received_monotonic_sec, 12.0)
        self.assertEqual(bundle.frames["head"].camera_timestamp_sec, 2.0)
        self.assertEqual(bundle.frames["head"].sequence, 20)

    def test_accepts_normalized_packet_and_counts_gaps(self):
        cache = CameraFrameCache("127.0.0.1", 5555, ["head"], FakeLogger())
        self.assertTrue(
            cache.update_from_packet(
                NormalizedPacket(
                    schema="robo_collector_camera.v4",
                    session_id="session-a",
                    frames={"head": NormalizedCameraFrame(
                        stream="head", sequence=10, payload=b"first",
                        payload_encoding="image/jpeg", timestamp_quality="host",
                        server_wall_timestamp=1.0,
                    )},
                )
            )
        )
        self.assertTrue(
            cache.update_from_packet(
                NormalizedPacket(
                    schema="robo_collector_camera.v4",
                    session_id="session-a",
                    frames={"head": NormalizedCameraFrame(
                        stream="head", sequence=15, payload=b"second",
                        payload_encoding="image/jpeg", timestamp_quality="host",
                        server_wall_timestamp=2.0,
                    )},
                    producer_gaps={"head": 2},
                )
            )
        )
        self.assertEqual(
            cache.stats,
            {
                "producer_gap": 2,
                "publisher_gap": 0,
                "transport_gap": 0,
                "unattributed_gap": 2,
                "selection_gap": 0,
                "duplicate": 0,
                "reorder": 0,
                "session_restart": 0,
                "stale": 0,
                "expired": 0,
                "queue_depth": 0,
                "queue_capacity": 128,
                "queue_overflow": 0,
                "callback_errors": 0,
                "recording_queue_overflow": 0,
                "recording_failed": False,
            },
        )
        self.assertEqual(cache.snapshot()["streams"]["head"]["sequence_gap"], 4)
        self.assertEqual(cache.snapshot()["streams"]["head"]["session_id"], "session-a")

    def test_gap_attribution_uses_producer_publisher_and_packet_sequence(self):
        cache = CameraFrameCache("127.0.0.1", 5555, ["head"], FakeLogger())
        self.assertTrue(cache.update_from_packet(
            NormalizedPacket(
                schema="robo_collector_camera.v4",
                session_id="session-a",
                frames={"head": NormalizedCameraFrame(
                    stream="head", sequence=10, payload=b"first",
                    payload_encoding="image/jpeg", timestamp_quality="host",
                    server_wall_timestamp=1.0,
                )},
                packet_sequence=8,
            )
        ))
        self.assertTrue(cache.update_from_packet(
            NormalizedPacket(
                schema="robo_collector_camera.v4",
                session_id="session-a",
                frames={"head": NormalizedCameraFrame(
                    stream="head", sequence=15, payload=b"second",
                    payload_encoding="image/jpeg", timestamp_quality="host",
                    server_wall_timestamp=2.0,
                )},
                producer_gaps={"head": 2},
                publisher_gaps={"head": 1},
                packet_sequence=11,
            )
        ))
        stats = cache.stats
        self.assertEqual(stats["producer_gap"], 2)
        self.assertEqual(stats["publisher_gap"], 1)
        self.assertEqual(stats["transport_gap"], 1)
        self.assertEqual(stats["unattributed_gap"], 0)

    def test_counts_duplicate_reorder_restart_and_selection_gap(self):
        cache = CameraFrameCache("127.0.0.1", 5555, ["head"], FakeLogger())
        packet = {
            "session_id": "session-a",
            "timestamps": {"head": 1.0},
            "sequences": {"head": 1},
            "images": {"head": "image"},
        }
        self.assertTrue(cache.update_from_packet(packet))
        self.assertFalse(cache.update_from_packet(packet))
        older = dict(packet)
        older["timestamps"] = {"head": 0.5}
        older["sequences"] = {"head": 0}
        self.assertFalse(cache.update_from_packet(older))
        self.assertTrue(
            cache.update_from_packet(
                {**packet, "session_id": "session-b", "sequences": {"head": 0}}
            )
        )
        cache.record_selection_gap(3, stream="head")
        self.assertEqual(cache.stats["duplicate"], 1)
        self.assertEqual(cache.stats["reorder"], 1)
        self.assertEqual(cache.stats["session_restart"], 1)
        self.assertEqual(cache.stats["selection_gap"], 3)
        stream = cache.snapshot()["streams"]["head"]
        self.assertEqual(stream["duplicate"], 1)
        self.assertEqual(stream["reorder"], 1)
        self.assertEqual(stream["stale"], 2)
        self.assertEqual(stream["session_restart"], 1)
        self.assertEqual(stream["session_id"], "session-b")

    def test_records_expired_bundle_once_for_each_stream(self):
        cache = CameraFrameCache(
            "127.0.0.1", 5555, ["head", "ego_view"], FakeLogger()
        )
        cache.record_stale()
        cache.record_stale(2, stream="head")

        self.assertEqual(cache.stats["stale"], 3)
        self.assertEqual(cache.stats["expired"], 3)
        streams = cache.snapshot()["streams"]
        self.assertEqual(streams["head"]["stale"], 3)
        self.assertEqual(streams["head"]["expired"], 3)
        self.assertEqual(streams["ego_view"]["stale"], 1)
        self.assertEqual(streams["ego_view"]["expired"], 1)
        self.assertEqual(cache.quality_statistics["expired_bundles"], 3)

    def test_recording_queue_overflow_is_a_failure_and_is_observable(self):
        cache = CameraFrameCache(
            "127.0.0.1", 5555, ["head"], FakeLogger(),
            receive_mode="recording", callback_queue_size=1,
        )
        cache.set_packet_callback(lambda packet: None)
        cache._callback_queue.put_nowait("occupied")
        cache._queue_callback("dropped")

        self.assertEqual(cache.stats["queue_depth"], 1)
        self.assertEqual(cache.stats["queue_capacity"], 1)
        self.assertEqual(cache.stats["queue_overflow"], 1)
        self.assertEqual(cache.stats["recording_queue_overflow"], 1)
        self.assertTrue(cache.stats["recording_failed"])
        self.assertEqual(cache.snapshot()["queue"]["depth"], 1)
        self.assertTrue(cache.quality_statistics["recording_failed"])

    def test_preview_queue_overflow_is_best_effort_but_visible(self):
        cache = CameraFrameCache(
            "127.0.0.1", 5555, ["head"], FakeLogger(),
            receive_mode="preview", callback_queue_size=1,
        )
        cache._callback_queue.put_nowait("occupied")
        cache._queue_callback("dropped")

        self.assertEqual(cache.quality_statistics["queue_overflow"], 1)
        self.assertEqual(cache.quality_statistics["recording_queue_overflow"], 0)
        self.assertFalse(cache.quality_statistics["recording_failed"])

    def test_callback_errors_are_counted_and_retain_last_error(self):
        cache = CameraFrameCache("127.0.0.1", 5555, ["head"], FakeLogger())
        cache.set_packet_callback(lambda packet: (_ for _ in ()).throw(RuntimeError("boom")))
        cache._callback_queue.put_nowait("packet")
        cache._stop.set()
        cache._callback_run()

        self.assertEqual(cache.stats["callback_errors"], 1)
        self.assertEqual(cache.snapshot()["last_callback_error"], "boom")
        self.assertEqual(cache.quality_statistics["callback_errors"], 1)

    def test_rejects_duplicate_bundle_and_excessive_inter_camera_skew(self):
        logger = FakeLogger()
        cache = CameraFrameCache(
            "127.0.0.1",
            5555,
            ["head", "ego_view"],
            logger,
            max_inter_camera_skew_sec=0.05,
        )
        packet = {
            "session_id": "session-a",
            "timestamps": {"head": 2.0, "ego_view": 2.01},
            "sequences": {"head": 20, "ego_view": 21},
            "images": {"head": "head-image", "ego_view": "ego-image"},
        }
        self.assertTrue(cache.update_from_packet(packet, received_monotonic_sec=12.0))
        self.assertFalse(cache.update_from_packet(packet, received_monotonic_sec=12.1))
        self.assertIn("did not advance", cache.last_error)

        skewed = {
            "session_id": "session-a",
            "timestamps": {"head": 3.0, "ego_view": 3.2},
            "sequences": {"head": 21, "ego_view": 22},
            "images": {"head": "head-image", "ego_view": "ego-image"},
        }
        self.assertFalse(cache.update_from_packet(skewed, received_monotonic_sec=13.0))
        self.assertIn("skew", cache.last_error)

    def test_quality_skew_prefers_common_server_wall_clock(self):
        cache = CameraFrameCache(
            "127.0.0.1",
            5555,
            ["head", "ego_view"],
            FakeLogger(),
            max_inter_camera_skew_sec=0.05,
        )
        packet = SimpleNamespace(
            session_id="session-a",
            frames={
                "head": SimpleNamespace(
                    sequence=1,
                    device_timestamp=100.0,
                    device_unit="ms",
                    server_wall_timestamp=10.00,
                    clock_domain="realsense:head",
                    timestamp_quality="device",
                    payload=b"head",
                ),
                "ego_view": SimpleNamespace(
                    sequence=1,
                    device_timestamp=900.0,
                    device_unit="ms",
                    server_wall_timestamp=10.01,
                    clock_domain="realsense:ego",
                    timestamp_quality="device",
                    payload=b"ego",
                ),
            },
        )
        self.assertTrue(cache.update_from_packet(packet))
        self.assertAlmostEqual(
            cache.quality_statistics["camera_camera_skew_sec"], 0.01
        )

    def test_rejects_nonfinite_timestamp_and_unsafe_stream_name(self):
        with self.assertRaisesRegex(ValueError, "invalid camera stream"):
            CameraFrameCache("127.0.0.1", 5555, ["../../escape"], FakeLogger())

        cache = CameraFrameCache("127.0.0.1", 5555, ["head"], FakeLogger())
        self.assertFalse(
            cache.update_from_packet(
                {
                    "session_id": "session-a",
                    "timestamps": {"head": float("inf")},
                    "images": {"head": "image"},
                }
            )
        )
        self.assertIn("non-finite", cache.last_error)

    def test_validates_camera_fps_metadata_when_expected(self):
        cache = CameraFrameCache(
            "127.0.0.1",
            5555,
            ["head"],
            FakeLogger(),
            expected_fps=30,
        )
        packet = {
            "session_id": "session-a",
            "timestamps": {"head": 2.0},
            "sequences": {"head": 20},
            "images": {"head": "head-image"},
            "metadata": {"fps": 50},
        }
        self.assertFalse(cache.update_from_packet(packet))
        self.assertIn("does not match", cache.last_error)

        packet["metadata"]["fps"] = 30
        self.assertTrue(cache.update_from_packet(packet))

    def test_rejects_fractional_sequence(self):
        cache = CameraFrameCache("127.0.0.1", 5555, ["head"], FakeLogger())
        self.assertFalse(
            cache.update_from_packet(
                {
                    "session_id": "session-a",
                    "timestamps": {"head": 1.0},
                    "sequences": {"head": 1.0},
                    "images": {"head": "image"},
                }
            )
        )
        self.assertIn("invalid sequence", cache.last_error)

    def test_accepts_sequence_reset_after_camera_server_restart(self):
        cache = CameraFrameCache("127.0.0.1", 5555, ["head"], FakeLogger())
        first = {
            "session_id": "session-a",
            "timestamps": {"head": 2.0},
            "sequences": {"head": 100},
            "images": {"head": "first"},
        }
        restarted = {
            "session_id": "session-b",
            "timestamps": {"head": 3.0},
            "sequences": {"head": 0},
            "images": {"head": "restarted"},
        }

        self.assertTrue(cache.update_from_packet(first))
        self.assertTrue(cache.update_from_packet(restarted))
        self.assertEqual(cache.latest().session_id, "session-b")
        self.assertEqual(cache.latest().images["head"], "restarted")

    def test_reset_episode_window_discards_idle_gap_and_quality_history(self):
        cache = CameraFrameCache("127.0.0.1", 5555, ["head"], FakeLogger())
        self.assertTrue(
            cache.update_from_packet(
                {
                    "session_id": "session-a",
                    "timestamps": {"head": 1.0},
                    "sequences": {"head": 1},
                    "images": {"head": "first"},
                }
            )
        )
        self.assertTrue(
            cache.update_from_packet(
                {
                    "session_id": "session-a",
                    "timestamps": {"head": 2.0},
                    "sequences": {"head": 5},
                    "images": {"head": "second"},
                }
            )
        )
        self.assertEqual(cache.stats["unattributed_gap"], 3)
        self.assertEqual(cache.quality_statistics["camera_camera_skew_samples"], 2)

        cache.reset_episode_window()

        self.assertEqual(
            cache.stats,
            {
                "producer_gap": 0,
                "publisher_gap": 0,
                "transport_gap": 0,
                "unattributed_gap": 0,
                "selection_gap": 0,
                "duplicate": 0,
                "reorder": 0,
                "session_restart": 0,
                "stale": 0,
                "expired": 0,
                "queue_depth": 0,
                "queue_capacity": 128,
                "queue_overflow": 0,
                "callback_errors": 0,
                "recording_queue_overflow": 0,
                "recording_failed": False,
            },
        )
        self.assertEqual(cache.quality_statistics["camera_camera_skew_samples"], 0)
        self.assertTrue(
            cache.update_from_packet(
                {
                    "session_id": "session-a",
                    "timestamps": {"head": 3.0},
                    "sequences": {"head": 0},
                    "images": {"head": "new-episode"},
                }
            )
        )
        self.assertEqual(cache.stats["transport_gap"], 0)

    def test_rejects_late_packet_from_retired_camera_session(self):
        cache = CameraFrameCache("127.0.0.1", 5555, ["head"], FakeLogger())
        first = {
            "session_id": "session-a",
            "timestamps": {"head": 2.0},
            "sequences": {"head": 100},
            "images": {"head": "first"},
        }
        restarted = {
            "session_id": "session-b",
            "timestamps": {"head": 3.0},
            "sequences": {"head": 0},
            "images": {"head": "restarted"},
        }
        late_old = {
            "session_id": "session-a",
            "timestamps": {"head": 3.1},
            "sequences": {"head": 101},
            "images": {"head": "late-old"},
        }

        self.assertTrue(cache.update_from_packet(first))
        self.assertTrue(cache.update_from_packet(restarted))
        self.assertFalse(cache.update_from_packet(late_old))
        self.assertEqual(cache.latest().session_id, "session-b")
        self.assertEqual(cache.latest().images["head"], "restarted")
        self.assertIn("retired session", cache.last_error)

    def test_parse_camera_streams_accepts_comma_string(self):
        self.assertEqual(parse_camera_streams("head, ego_view"), ["head", "ego_view"])


if __name__ == "__main__":
    unittest.main()
