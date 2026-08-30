import hashlib
import tempfile
import unittest
from pathlib import Path

from mcap.reader import make_reader
from robo_collector.canonical_mcap import (
    CanonicalChannel,
    CanonicalMcapError,
    CanonicalRecord,
    annexb_nal_types,
    validate_canonical_mcap,
    validate_h264_access_units,
    write_canonical_groups,
)
from robo_collector.mcap.v1 import episode_pb2


def _timestamp(value):
    return episode_pb2.TimestampSetV1(
        source_time_ns=value,
        receive_time_ns=value,
        normalized_time_ns=value,
        source_clock_domain=episode_pb2.CLOCK_DOMAIN_COLLECTOR_MONOTONIC,
        normalization_mode=episode_pb2.NORMALIZATION_MODE_COLLECTOR_DIRECT,
        fallback_reason=episode_pb2.CLOCK_FALLBACK_REASON_NONE,
        policy_version="collector_direct_v1",
        clock_session_id="clock.session",
    )


def _video(sequence=1, timestamp=20, *, keyframe=True):
    config = b"\x00\x00\x00\x01\x67\x64\x00\x1f\x00\x00\x00\x01\x68\xee"
    access_unit = (
        config + b"\x00\x00\x00\x01" + (b"\x65" if keyframe else b"\x41") + b"payload"
    )
    return episode_pb2.VideoAccessUnitV1(
        stream_id="wrist",
        source_sequence=sequence,
        collector_record_id=sequence,
        timestamps=_timestamp(timestamp),
        pts=timestamp,
        dts=timestamp,
        timebase_num=1,
        timebase_den=1_000_000_000,
        codec=episode_pb2.PAYLOAD_ENCODING_H264_ANNEX_B,
        profile="high",
        level="4.1",
        codec_config_annexb=config,
        codec_config_sha256=hashlib.sha256(config).digest(),
        keyframe=keyframe,
        config_generation=0,
        source_session_id="camera.session",
        access_unit_annexb=access_unit,
        width=640,
        height=480,
    )


def _robot(sequence=1, timestamp=10):
    return episode_pb2.RobotStateV1(
        source_sequence=sequence,
        collector_record_id=sequence,
        timestamps=_timestamp(timestamp),
        source_session_id="robot.session",
    )


CAMERA_METADATA = {
    "robo.stream_id": "wrist",
    "robo.source_id": "camera",
    "robo.sensor_id": "sensor",
    "robo.frame_id": "wrist_frame",
    "robo.calibration_revision": "1",
    "robo.codec": "h264-annexb",
    "robo.nominal_rate_hz": "30",
    "robo.observed_rate_hz": "30",
    "robo.clock_domain": "collector_monotonic",
    "robo.schema_version": "1",
    "robo.pipeline_version": "phase2",
}
ROBOT_METADATA = {
    "robo.source_id": "robot",
    "robo.robot_id": "robot",
    "robo.frame_id": "base",
    "robo.nominal_rate_hz": "100",
    "robo.observed_rate_hz": "100",
    "robo.clock_domain": "collector_monotonic",
    "robo.schema_version": "1",
    "robo.pipeline_version": "phase2",
}


class CanonicalMcapTest(unittest.TestCase):
    def test_annexb_and_generation_contract(self):
        value = _video()
        self.assertEqual(annexb_nal_types(value.access_unit_annexb), (7, 8, 5))
        self.assertEqual(validate_h264_access_units([value]), (value,))
        value.dts += 1
        with self.assertRaisesRegex(CanonicalMcapError, "PTS/DTS"):
            validate_h264_access_units([value])

    def test_keyframe_requires_sps_pps_idr(self):
        value = _video()
        value.access_unit_annexb = b"\x00\x00\x01\x65payload"
        with self.assertRaisesRegex(CanonicalMcapError, "SPS, PPS, and IDR"):
            validate_h264_access_units([value])

    def test_writer_is_ordered_isolated_and_structurally_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            result = write_canonical_groups(
                directory,
                [
                    CanonicalRecord("/camera/wrist/h264", _video()),
                    CanonicalRecord("/robot/state/raw", _robot()),
                ],
                [
                    CanonicalChannel("/camera/wrist/h264", CAMERA_METADATA),
                    CanonicalChannel("/robot/state/raw", ROBOT_METADATA),
                ],
            )
            camera = validate_canonical_mcap(
                result.camera_path, expected_group="camera"
            )
            robot = validate_canonical_mcap(result.robot_path, expected_group="robot")
            self.assertEqual(camera["message_count"], "1")
            self.assertEqual(robot["message_count"], "1")
            self.assertEqual(len(result.keyframes), 1)
            with Path(result.camera_path).open("rb") as stream:
                topics = [
                    channel.topic
                    for _, channel, _ in make_reader(stream).iter_messages()
                ]
            self.assertEqual(topics, ["/camera/wrist/h264"])

    def test_refuses_to_replace_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "camera.mcap").write_bytes(b"winner")
            with self.assertRaisesRegex(CanonicalMcapError, "refusing to replace"):
                write_canonical_groups(
                    directory,
                    [CanonicalRecord("/camera/wrist/h264", _video())],
                    [CanonicalChannel("/camera/wrist/h264", CAMERA_METADATA)],
                )

    def test_output_bytes_are_independent_of_input_order(self):
        channels = [
            CanonicalChannel("/camera/wrist/h264", CAMERA_METADATA),
            CanonicalChannel("/robot/state/raw", ROBOT_METADATA),
        ]
        records = [
            CanonicalRecord("/camera/wrist/h264", _video()),
            CanonicalRecord("/robot/state/raw", _robot()),
        ]
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            left = write_canonical_groups(first, records, channels)
            right = write_canonical_groups(
                second, reversed(records), reversed(channels)
            )
            self.assertEqual(
                left.camera_path.read_bytes(), right.camera_path.read_bytes()
            )
            self.assertEqual(
                left.robot_path.read_bytes(), right.robot_path.read_bytes()
            )


if __name__ == "__main__":
    unittest.main()
