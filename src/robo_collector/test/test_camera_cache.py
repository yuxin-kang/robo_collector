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
                "timestamps": {"head": 0.9, "ego_view": 1.0},
                "images": {"head": "head-image", "ego_view": "ego-image"},
            },
            received_monotonic_sec=9.0,
        )

        updated = cache.update_from_packet(
            {
                "timestamps": {"ego_view": 1.0},
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
                "timestamps": {"head": 2.0, "ego_view": 2.1},
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

    def test_parse_camera_streams_accepts_comma_string(self):
        self.assertEqual(parse_camera_streams("head, ego_view"), ["head", "ego_view"])


if __name__ == "__main__":
    unittest.main()
