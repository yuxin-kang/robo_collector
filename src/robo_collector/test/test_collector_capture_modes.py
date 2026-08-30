import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from types import MappingProxyType, SimpleNamespace


def _install_ros_stubs():
    rclpy = types.ModuleType("rclpy")
    node = types.ModuleType("rclpy.node")
    qos = types.ModuleType("rclpy.qos")
    node.Node = object
    qos.QoSProfile = lambda **kwargs: kwargs
    qos.qos_profile_sensor_data = object()
    rclpy.node = node
    rclpy.qos = qos
    sys.modules.setdefault("rclpy", rclpy)
    sys.modules.setdefault("rclpy.node", node)
    sys.modules.setdefault("rclpy.qos", qos)
    for package, names in (
        ("diagnostic_msgs.msg", ("DiagnosticStatus", "KeyValue")),
        ("robo_collector_msgs.msg", ("RecordCommand",)),
        ("robo_state_msgs.msg", ("RoboStateSample",)),
    ):
        parent_name = package.split(".")[0]
        parent = types.ModuleType(parent_name)
        module = types.ModuleType(package)
        for name in names:
            attributes = {"OK": 0, "WARN": 1, "ERROR": 2} if name == "DiagnosticStatus" else {}
            setattr(module, name, type(name, (), attributes))
        parent.msg = module
        sys.modules.setdefault(parent_name, parent)
        sys.modules.setdefault(package, module)


_install_ros_stubs()

from robo_collector.capture_coordinator import (
    CaptureCoordinator,
    CaptureMode,
    CaptureResult,
    CaptureSourceFence,
    CaptureStatus,
)
from robo_collector.collector_node import (
    CollectorCaptureOperation,
    LeRobotCollectorNode,
    McapLandingSinkAdapter,
    camera_source_fences_from_status,
    mcap_camera_record,
    mcap_event_record,
    mcap_robot_state_record,
    normalize_recording_mode,
    observe_callback_rate_hz,
    parse_camera_stream_rates,
    required_camera_sources_from_status,
)
from robo_collector.mcap.v1 import episode_pb2
from robo_collector.mcap_landing import LandingChannel, LandingWriter

from robo_collector import mcap_contract


class CaptureModeConfigTest(unittest.TestCase):
    def test_alias_and_per_stream_rates(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertEqual(normalize_recording_mode("raw_first"), CaptureMode.RAW_V1)
        self.assertEqual(
            parse_camera_stream_rates("head=30,hand=15", ("head", "hand"), fps_alias=5),
            {"head": 30.0, "hand": 15.0},
        )
        self.assertEqual(
            parse_camera_stream_rates("", ("head", "hand"), fps_alias=20),
            {"head": 20.0, "hand": 20.0},
        )

    def test_robot_rate_is_measured_only_from_callback_intervals(self):
        previous, rate = observe_callback_rate_hz(None, None, 10.0)
        previous, rate = observe_callback_rate_hz(previous, rate, 10.02)
        camera_30 = parse_camera_stream_rates("", ("head",), fps_alias=30)
        camera_5 = parse_camera_stream_rates("", ("head",), fps_alias=5)
        self.assertAlmostEqual(rate, 50.0)
        self.assertNotEqual(camera_30, camera_5)
        self.assertAlmostEqual(
            observe_callback_rate_hz(previous, rate, 10.04)[1], 50.0
        )


class FrozenPayloadRecordTest(unittest.TestCase):
    def test_robot_state_round_trip_preserves_values_and_identity(self):
        record = mcap_robot_state_record(
            17,
            state={"arm": {"position": [1.25, -2.5]}, "gripper": 0.75},
            provenance={"receive_monotonic_timestamp": 12.5},
            source_sequence=9,
            session_id="robot-session",
        )
        message = episode_pb2.RobotStateV1.FromString(record.data)
        values = {field.field_name: list(field.values) for field in message.fields}
        self.assertEqual(record.channel, "/robot/state/raw")
        self.assertEqual(message.collector_record_id, 17)
        self.assertEqual(message.source_sequence, 9)
        self.assertEqual(message.source_session_id, "robot-session")
        self.assertEqual(values["arm.position"], [1.25, -2.5])
        self.assertEqual(values["gripper"], [0.75])

    def test_camera_payload_round_trip_preserves_jpeg_and_h264_bytes(self):
        jpeg = b"\xff\xd8payload\xff\xd9"
        jpeg_record = mcap_camera_record(
            3,
            stream="head",
            payload=jpeg,
            payload_encoding="image/jpeg",
            provenance={"width": 640, "height": 480},
            source_sequence=4,
            packet_sequence=2,
            session_id="camera-session",
        )
        jpeg_message = episode_pb2.CameraSampleV1.FromString(jpeg_record.data)
        self.assertEqual(jpeg_record.channel, "/camera/head/sample")
        self.assertEqual(jpeg_message.image_bytes, jpeg)
        self.assertEqual(jpeg_message.collector_record_id, 3)
        self.assertEqual(jpeg_message.source_session_id, "camera-session")

        annex_b = b"\x00\x00\x00\x01\x65\x88"
        h264_record = mcap_camera_record(
            4,
            stream="head",
            payload=annex_b,
            payload_encoding="h264_annex_b",
            provenance={"width": 640, "height": 480},
            source_sequence=5,
            packet_sequence=3,
            session_id="camera-session",
        )
        h264_message = episode_pb2.VideoAccessUnitV1.FromString(h264_record.data)
        self.assertEqual(h264_record.channel, "/camera/head/h264")
        self.assertEqual(h264_message.access_unit_annexb, annex_b)
        self.assertEqual(h264_message.collector_record_id, 4)

    def test_stop_event_round_trips_durable_source_fence(self):
        fence = CaptureSourceFence("camera.head", "camera-session", 4, 7)
        record = mcap_event_record(
            8,
            event_name="STOP",
            details={"reason": "user_stop"},
            source_fences=(fence,),
        )
        message = episode_pb2.EpisodeEventV1.FromString(record.data)
        self.assertEqual(message.collector_record_id, 8)
        self.assertEqual(message.event_type, episode_pb2.EPISODE_EVENT_STOP)
        self.assertEqual(message.source_fences[0].source_id, "camera.head")
        self.assertEqual(message.source_fences[0].durable_high_watermark, 7)

    def test_camera_fences_fail_closed_and_preserve_watermarks(self):
        start = {
            "session_id": "camera-session",
            "streams": {
                "head": {
                    "generation": 2,
                    "durable_high_watermark": 4,
                    "start_fence": {"sequence": 4},
                }
            },
        }
        stop = {
            "session_id": "camera-session",
            "source_complete": True,
            "close_failures": {},
            "streams": {
                "head": {
                    "generation": 2,
                    "durable_high_watermark": 7,
                    "start_fence": {"sequence": 4},
                    "stop_fence": {"sequence": 7},
                    "source_complete": True,
                    "close_error": None,
                }
            },
        }
        self.assertEqual(
            camera_source_fences_from_status(start, stop, ("head",)),
            (CaptureSourceFence("camera.head", "camera-session", 5, 8),),
        )
        stop["source_complete"] = False
        with self.assertRaises(ValueError):
            camera_source_fences_from_status(start, stop, ("head",))

    def test_camera_fences_reject_missing_mismatched_and_incomplete_stop(self):
        start = {
            "session_id": "camera-session",
            "streams": {
                "head": {
                    "generation": 2,
                    "durable_high_watermark": 4,
                    "start_fence": {"sequence": 4},
                }
            },
        }
        valid_stop = {
            "session_id": "camera-session",
            "source_complete": True,
            "close_failures": {},
            "streams": {
                "head": {
                    "generation": 2,
                    "durable_high_watermark": 7,
                    "stop_fence": {"sequence": 7},
                    "source_complete": True,
                    "close_error": None,
                }
            },
        }
        invalid_stops = (
            {},
            {**valid_stop, "session_id": "different-session"},
            {**valid_stop, "streams": {}},
            {
                **valid_stop,
                "streams": {
                    "head": {
                        **valid_stop["streams"]["head"],
                        "stop_fence": None,
                    }
                },
            },
        )

        for stop in invalid_stops:
            with self.subTest(stop=stop), self.assertRaises(ValueError):
                camera_source_fences_from_status(start, stop, ("head",))


class McapOnlyLifecycleTest(unittest.TestCase):
    def test_discarded_capture_result_is_never_ready(self):
        ready = CaptureResult(
            mode=CaptureMode.MCAP_FIRST,
            status=CaptureStatus.READY,
            terminal_accepted_frontier=0,
            dispositions=MappingProxyType({}),
            errors=(),
        )

        class Coordinator:
            def stop(self):
                return ready

        node = object.__new__(LeRobotCollectorNode)
        node._capture_coordinator = Coordinator()
        node._raw_sink_adapter = None
        node._mcap_sink_adapter = None
        node._raw_recorder = None
        node._raw_materialization_job = {"pending": True}
        node._capture_result = None
        node._append_raw_event = lambda *_args, **_kwargs: None

        node._discard_raw_episode(reason="user_discard")

        self.assertIsNotNone(node._capture_result)
        self.assertEqual(node._capture_result.status, CaptureStatus.QUARANTINED)
        self.assertFalse(node._capture_result.success)
        self.assertIn("capture discarded", node._capture_result.errors[-1])

    def test_pre_first_frame_start_sequence_zero_seals_with_landing_offset(self):
        start = {
            "session_id": "camera-session",
            "streams": {
                "head": {
                    "generation": 1,
                    "durable_high_watermark": None,
                    "start_fence": {"sequence": None},
                }
            },
        }
        stop = {
            "session_id": "camera-session",
            "source_complete": True,
            "close_failures": {},
            "streams": {
                "head": {
                    "generation": 1,
                    "durable_high_watermark": 0,
                    "start_fence": {"sequence": None},
                    "stop_fence": {"sequence": 0},
                    "source_complete": True,
                    "close_error": None,
                }
            },
        }
        required = required_camera_sources_from_status(start, ("head",))
        fences = camera_source_fences_from_status(start, stop, ("head",))
        self.assertEqual(required[0].start_sequence_exclusive, 0)
        self.assertEqual(fences[0], CaptureSourceFence("camera.head", "camera-session", 0, 1))

        episode_dir = Path(tempfile.mkdtemp()) / "episode-zero"
        descriptor = mcap_contract.descriptor_set_bytes()
        writer = LandingWriter(
            episode_dir,
            episode_id="episode-zero",
            collection_mode="mcap_first",
            required_sources=required,
        )
        writer.register_channel(
            LandingChannel(
                "/episode/event",
                "EpisodeEventV1",
                descriptor,
                metadata={
                    "robo.robot_id": "collector",
                    "robo.schema_version": "1",
                    "robo.pipeline_version": "test",
                },
            )
        )
        writer.register_channel(
            LandingChannel(
                "/camera/head/sample",
                "CameraSampleV1",
                descriptor,
                metadata={
                    "robo.stream_id": "head",
                    "robo.source_id": "camera.head",
                    "robo.sensor_id": "head",
                    "robo.frame_id": "head",
                    "robo.calibration_revision": "unknown",
                    "robo.pixel_format": "rgb8",
                    "robo.nominal_rate_hz": "30",
                    "robo.observed_rate_hz": "30",
                    "robo.clock_domain": "device",
                    "robo.schema_version": "1",
                    "robo.pipeline_version": "test",
                },
            )
        )
        writer.start()
        sink = McapLandingSinkAdapter(writer)
        coordinator = CaptureCoordinator(
            CaptureMode.MCAP_FIRST, mcap_sink=sink
        ).start()
        coordinator.submit(
            CollectorCaptureOperation(
                None,
                lambda record_id: mcap_event_record(
                    record_id, event_name="START", details={}
                ),
            )
        )
        payload = b"\xff\xd8first-frame\xff\xd9"
        coordinator.submit(
            CollectorCaptureOperation(
                None,
                lambda record_id: mcap_camera_record(
                    record_id,
                    stream="head",
                    payload=payload,
                    payload_encoding="image/jpeg",
                    provenance={},
                    source_sequence=0,
                    packet_sequence=0,
                    session_id="camera-session",
                ),
            ),
            source_id="camera.head",
            session_id="camera-session",
            source_sequence=0,
        )
        coordinator.submit(
            CollectorCaptureOperation(
                None,
                lambda record_id: mcap_event_record(
                    record_id,
                    event_name="STOP",
                    details={},
                    source_fences=fences,
                ),
            )
        )

        result = coordinator.stop(source_fences=fences)
        camera_record = mcap_camera_record(
            1,
            stream="head",
            payload=payload,
            payload_encoding="image/jpeg",
            provenance={},
            source_sequence=0,
            packet_sequence=0,
            session_id="camera-session",
        )
        protobuf = episode_pb2.CameraSampleV1.FromString(camera_record.data)

        self.assertTrue(result.success)
        self.assertEqual(camera_record.source_sequence, 1)
        self.assertEqual(protobuf.source_sequence, 0)
        self.assertEqual(protobuf.image_bytes, payload)
        self.assertTrue(sink.manifest_path.is_file())

    def test_mcap_sink_seals_payload_records_end_to_end(self):
        episode_dir = Path(tempfile.mkdtemp()) / "episode"
        descriptor = mcap_contract.descriptor_set_bytes()
        writer = LandingWriter(
            episode_dir,
            episode_id="episode",
            collection_mode="mcap_first",
        )
        writer.register_channel(
            LandingChannel(
                "/episode/event",
                "EpisodeEventV1",
                descriptor,
                metadata={
                    "robo.robot_id": "collector",
                    "robo.schema_version": "1",
                    "robo.pipeline_version": "test",
                },
            )
        )
        writer.register_channel(
            LandingChannel(
                "/robot/state/raw",
                "RobotStateV1",
                descriptor,
                metadata={
                    "robo.source_id": "robot_state",
                    "robo.robot_id": "collector",
                    "robo.frame_id": "base",
                    "robo.nominal_rate_hz": "50",
                    "robo.observed_rate_hz": "50",
                    "robo.clock_domain": "callback",
                    "robo.schema_version": "1",
                    "robo.pipeline_version": "test",
                },
            )
        )
        writer.start()
        sink = McapLandingSinkAdapter(writer)
        coordinator = CaptureCoordinator(
            CaptureMode.MCAP_FIRST, mcap_sink=sink
        ).start()
        coordinator.submit(
            CollectorCaptureOperation(
                None,
                lambda record_id: mcap_event_record(
                    record_id, event_name="START", details={}
                ),
            )
        )
        coordinator.submit(
            CollectorCaptureOperation(
                None,
                lambda record_id: mcap_robot_state_record(
                    record_id,
                    state={"joint": [1.0, 2.0]},
                    provenance={},
                    source_sequence=1,
                    session_id="robot-session",
                ),
            )
        )
        coordinator.submit(
            CollectorCaptureOperation(
                None,
                lambda record_id: mcap_event_record(
                    record_id, event_name="STOP", details={}
                ),
            )
        )

        result = coordinator.stop()

        self.assertTrue(result.success)
        self.assertTrue(sink.seal_result.sealed_path.is_file())
        self.assertTrue(sink.manifest_path.is_file())

    def test_mcap_first_save_does_not_require_raw_episode_path(self):
        events = []

        class StateMachine:
            session = SimpleNamespace(episode_id="episode-mcap")

            def mark_saving(self):
                events.append("saving")

            def mark_saved(self):
                events.append("saved")

            def mark_failed(self, reason):
                self.failure = reason

        node = object.__new__(LeRobotCollectorNode)
        node._capture_mode = CaptureMode.MCAP_FIRST
        node._capture_result = CaptureResult(
            mode=CaptureMode.MCAP_FIRST,
            status=CaptureStatus.READY,
            terminal_accepted_frontier=2,
            dispositions=MappingProxyType({}),
            errors=(),
        )
        node._raw_episode_path = None
        node._state_machine = StateMachine()
        node._last_episode_id = ""
        node._last_episode_outcome = ""
        node._publish_status = lambda level, message: events.append(message)

        node._begin_save_episode()

        self.assertEqual(events[:2], ["saving", "saved"])
        self.assertEqual(node._last_episode_outcome, "SAVED")
        self.assertIsNone(node._raw_episode_path)


if __name__ == "__main__":
    unittest.main()
