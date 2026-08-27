import unittest

try:
    import cv2
    import numpy as np

    from robo_collector_camera.client import (
        CameraClient,
        CameraPacketError,
        decode_envelope,
        decode_packet,
        normalize_packet,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"cv2", "numpy"}:
        raise
    cv2 = None
    np = None
    CameraClient = None
    decode_packet = None
    decode_envelope = None
    normalize_packet = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, "opencv-python is not installed")
class CameraClientDecodeTest(unittest.TestCase):
    def test_decode_envelope_retains_v3_payload_until_explicit_decode(self):
        assert decode_envelope is not None
        payload = b"not-decoded-yet"
        envelope = decode_envelope({
            "schema": "robo_collector_camera.v3",
            "session_id": "server-session-a",
            "timestamps": {"head": 1.0},
            "sequences": {"head": 10},
            "images": {"head": payload},
            "metadata": {},
        }, receive_monotonic_timestamp=4.5)
        frame = envelope.frames["head"]
        self.assertIs(frame.payload, payload)
        self.assertEqual(frame.timestamp_quality, "host_after_capture")
        self.assertEqual(frame.server_wall_timestamp, 1.0)
        self.assertEqual(frame.receive_monotonic_timestamp, 4.5)

    def test_decode_envelope_reads_v4_provenance(self):
        assert decode_envelope is not None
        payload = b"jpeg"
        envelope = decode_envelope({
            "schema": "robo_collector_camera.v4",
            "session_id": "server-session-a",
            "streams": {"head": {
                "sequence": 12,
                "payload": payload,
                "payload_encoding": "image/jpeg",
                "timestamps": {
                    "device": 123.4,
                    "device_unit": "ms",
                    "device_clock_domain": "realsense:abc",
                    "server_wall": 1000.5,
                    "server_monotonic": 22.5,
                },
                "timestamp_quality": "device",
            }},
            "packet_sequence": 8,
            "producer_gaps": {"head": 2},
            "publisher_gaps": {"head": 1},
            "metadata": {},
        })
        frame = envelope.frames["head"]
        self.assertEqual(frame.payload, payload)
        self.assertEqual(frame.device_timestamp, 123.4)
        self.assertEqual(frame.clock_domain, "realsense:abc")
        self.assertEqual(frame.timestamp_quality, "device")
        self.assertEqual(envelope.producer_gaps, {"head": 2})
        self.assertEqual(envelope.publisher_gaps, {"head": 1})
        self.assertEqual(envelope.packet_sequence, 8)

    def test_decode_packet_falls_back_to_normalized_device_time(self):
        assert cv2 is not None
        assert np is not None
        assert decode_packet is not None
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        decoded = decode_packet({
            "schema": "robo_collector_camera.v4",
            "session_id": "server-session-a",
            "streams": {"head": {
                "sequence": 12,
                "payload": _encode_jpeg_rgb(image),
                "payload_encoding": "image/jpeg",
                "timestamps": {
                    "device": 123_000,
                    "device_unit": "ms",
                    "device_clock_domain": "realsense:abc",
                },
                "timestamp_quality": "device",
            }},
            "metadata": {},
        })

        self.assertEqual(decoded["timestamps"], {"head": 123.0})

    def test_legacy_decoded_mapping_keeps_encoded_payload_for_recording(self):
        assert cv2 is not None
        assert np is not None
        assert decode_packet is not None
        assert normalize_packet is not None
        payload = _encode_jpeg_rgb(np.zeros((2, 2, 3), dtype=np.uint8))
        decoded = decode_packet({
            "schema": "robo_collector_camera.v4",
            "session_id": "server-session-a",
            "streams": {"head": {
                "sequence": 1,
                "payload": payload,
                "payload_encoding": "image/jpeg",
                "timestamps": {"server_wall": 1.0},
                "timestamp_quality": "host",
            }},
            "metadata": {},
        })

        normalized = normalize_packet(decoded, ("head",), decode_images=False)

        self.assertEqual(normalized.frames["head"].payload, payload)
        self.assertEqual(
            normalized.frames["head"].timestamp_quality,
            "host_after_capture",
        )
        self.assertEqual(normalized.frames["head"].receive_monotonic_timestamp, None)

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

    def test_decodes_v4_to_the_same_mapping_shape(self):
        assert cv2 is not None
        assert np is not None
        assert decode_packet is not None
        image = np.full((4, 5, 3), (255, 0, 0), dtype=np.uint8)
        decoded = decode_packet({
            "schema": "robo_collector_camera.v4",
            "session_id": "server-session-a",
            "streams": {"head": {
                "sequence": 12,
                "payload": _encode_jpeg_rgb(image),
                "payload_encoding": "image/jpeg",
                "timestamps": {"server_wall": 1000.5},
                "timestamp_quality": "host",
            }},
            "metadata": {"cameras": {"head": {}}},
        }, host="robot", port=5555)

        self.assertEqual(decoded["schema"], "robo_collector_camera.v4")
        self.assertEqual(decoded["session_id"], "server-session-a")
        self.assertEqual(decoded["timestamps"], {"head": 1000.5})
        self.assertEqual(decoded["sequences"], {"head": 12})
        self.assertEqual(decoded["images"]["head"].shape, (4, 5, 3))
        self.assertEqual(decoded["metadata"], {"cameras": {"head": {}}})
        self.assertEqual(decoded["host"], "robot")
        self.assertEqual(decoded["port"], 5555)

    def test_read_decodes_v4_packet(self):
        assert CameraClient is not None
        assert np is not None
        import msgpack

        packet = {
            "schema": "robo_collector_camera.v4",
            "session_id": "server-session-a",
            "streams": {"head": {
                "sequence": 12,
                "payload": _encode_jpeg_rgb(np.zeros((2, 2, 3), dtype=np.uint8)),
                "payload_encoding": "image/jpeg",
                "timestamps": {"server_wall": 1000.5},
                "timestamp_quality": "host",
            }},
            "metadata": {},
        }

        class Socket:
            def poll(self, timeout_ms):
                return True

            def recv(self):
                return msgpack.packb(packet, use_bin_type=True)

        client = CameraClient.__new__(CameraClient)
        client.socket = Socket()
        client.host = "robot"
        client.port = 5555
        client.max_packet_bytes = 1024 * 1024

        decoded = client.read()

        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded["schema"], "robo_collector_camera.v4")
        self.assertEqual(decoded["sequences"], {"head": 12})

    def test_rejects_malformed_v3_metadata_as_camera_packet_error(self):
        assert decode_packet is not None
        for metadata in (None, [], "malformed"):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(
                CameraPacketError, "metadata"
            ):
                decode_packet({
                    "schema": "robo_collector_camera.v3",
                    "session_id": "server-session-a",
                    "timestamps": {"head": 1.0},
                    "sequences": {"head": 1},
                    "images": {"head": b"jpeg"},
                    "metadata": metadata,
                })

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
