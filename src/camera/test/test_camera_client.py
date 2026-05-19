import unittest

try:
    import cv2
    import numpy as np

    from robo_collector_camera.client import decode_packet
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
            "schema": "robo_collector_camera.v2",
            "timestamps": {"head": 1.0, "ego_view": 1.1},
            "images": {
                "head": _encode_jpeg_rgb(head),
                "ego_view": _encode_jpeg_rgb(ego_view),
            },
            "metadata": {"cameras": {"head": {}, "ego_view": {}}},
        }

        decoded = decode_packet(packet, host="robot", port=5555)

        self.assertEqual(decoded["schema"], "robo_collector_camera.v2")
        self.assertEqual(set(decoded["images"]), {"head", "ego_view"})
        self.assertEqual(decoded["images"]["head"].shape, (4, 5, 3))
        self.assertEqual(decoded["images"]["ego_view"].shape, (3, 6, 3))
        self.assertEqual(decoded["host"], "robot")
        self.assertEqual(decoded["port"], 5555)


def _encode_jpeg_rgb(image_rgb):
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("failed to encode test image")
    return buffer.tobytes()


if __name__ == "__main__":
    unittest.main()
