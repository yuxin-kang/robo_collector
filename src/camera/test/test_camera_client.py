import unittest

try:
    import cv2
    import numpy as np

    from robo_collector_camera.client import CameraPacketError, decode_packet
except ModuleNotFoundError as exc:
    if exc.name not in {"cv2", "numpy"}:
        raise
    cv2 = None
    np = None
    decode_packet = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, "opencv-python is not installed")
class CameraClientDecodeTest(unittest.TestCase):
    def test_decodes_two_rgb_streams_from_one_payload(self):
        assert cv2 is not None
        assert np is not None
        assert decode_packet is not None
        head = np.full((4, 5, 3), (255, 0, 0), dtype=np.uint8)
        ego_view = np.full((3, 6, 3), (0, 255, 0), dtype=np.uint8)

        packet = {
            "schema": "robo_collector_camera.v3",
            "session_id": "server-session-a",
            "timestamps": {"head": 1.0, "ego_view": 1.1},
            "sequences": {"head": 10, "ego_view": 11},
            "images": {
                "head": _encode_jpeg_rgb(head),
                "ego_view": _encode_jpeg_rgb(ego_view),
            },
            "metadata": {"cameras": {"head": {}, "ego_view": {}}},
        }

        decoded = decode_packet(packet, host="robot", port=5555)

        self.assertEqual(decoded["schema"], "robo_collector_camera.v3")
        self.assertEqual(decoded["session_id"], "server-session-a")
        self.assertEqual(set(decoded["images"]), {"head", "ego_view"})
        self.assertEqual(decoded["images"]["head"].shape, (4, 5, 3))
        self.assertEqual(decoded["images"]["ego_view"].shape, (3, 6, 3))
        self.assertEqual(decoded["sequences"], {"head": 10, "ego_view": 11})
        self.assertEqual(decoded["host"], "robot")
        self.assertEqual(decoded["port"], 5555)

    def test_rejects_wrong_schema_nonfinite_timestamp_and_invalid_blob(self):
        assert decode_packet is not None
        with self.assertRaisesRegex(CameraPacketError, "schema"):
            decode_packet({"schema": "unknown", "images": {}})

        base = {
            "schema": "robo_collector_camera.v3",
            "session_id": "server-session-a",
            "timestamps": {"head": float("nan")},
            "sequences": {"head": 0},
            "images": {"head": b"not-an-image"},
        }
        with self.assertRaisesRegex(CameraPacketError, "finite"):
            decode_packet(base)

        base["timestamps"]["head"] = 1.0
        with self.assertRaisesRegex(CameraPacketError, "decode"):
            decode_packet(base)

    def test_rejects_fractional_sequence(self):
        assert cv2 is not None
        assert np is not None
        assert decode_packet is not None
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaisesRegex(CameraPacketError, "non-negative integer"):
            decode_packet(
                {
                    "schema": "robo_collector_camera.v3",
                    "session_id": "server-session-a",
                    "timestamps": {"head": 1.0},
                    "sequences": {"head": 1.0},
                    "images": {"head": _encode_jpeg_rgb(image)},
                }
            )

    def test_rejects_missing_server_session_id(self):
        assert cv2 is not None
        assert np is not None
        assert decode_packet is not None
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with self.assertRaisesRegex(CameraPacketError, "session_id"):
            decode_packet(
                {
                    "schema": "robo_collector_camera.v3",
                    "timestamps": {"head": 1.0},
                    "sequences": {"head": 1},
                    "images": {"head": _encode_jpeg_rgb(image)},
                }
            )


def _encode_jpeg_rgb(image_rgb):
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("failed to encode test image")
    return buffer.tobytes()


if __name__ == "__main__":
    unittest.main()
