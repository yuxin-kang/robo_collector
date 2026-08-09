import unittest

from robo_collector.camera_cache import CameraFrameCache, parse_camera_streams


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warn(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)


class CameraFrameCacheTest(unittest.TestCase):
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
