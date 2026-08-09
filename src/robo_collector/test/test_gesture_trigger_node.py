import unittest
from dataclasses import dataclass
from pathlib import Path

from robo_collector.gesture_trigger_node import (
    active_gesture_l2,
    attempt_state_is_armed,
    collector_status_is_fresh,
    detection_tail_frames,
    diagnostic_level_number,
    diagnostic_level_value,
    parse_collector_status,
    resolve_dataset_root,
    resolve_progress_path,
    update_detector_safely,
)
from robo_collector.gesture_logic import AttemptState, DetectionResult, GestureConditionDetector
from robo_collector.gesture_plan import GestureCondition


@dataclass(frozen=True)
class FakeKeyValue:
    key: str
    value: str


@dataclass(frozen=True)
class FakeDiagnosticStatus:
    level: object
    message: str
    values: list[FakeKeyValue]


class GestureTriggerNodeHelpersTest(unittest.TestCase):
    def test_explicit_dataset_root_wins_over_status(self):
        self.assertEqual(
            resolve_dataset_root(
                "outputs/manual_dataset",
                "",
                "outputs/from_status",
            ),
            "outputs/manual_dataset",
        )

    def test_status_dataset_root_is_used_when_plan_is_empty(self):
        self.assertEqual(
            resolve_dataset_root(
                "",
                "",
                "outputs/from_status",
            ),
            "outputs/from_status",
        )

    def test_empty_dataset_root_fails_closed(self):
        self.assertEqual(resolve_dataset_root("", "", ""), "")

    def test_progress_path_priority_is_param_then_plan_then_default(self):
        self.assertEqual(
            resolve_progress_path(
                "outputs/override/progress.json",
                "outputs/from_plan/progress.json",
                "outputs/from_dataset",
            ),
            Path("outputs/override/progress.json"),
        )
        self.assertEqual(
            resolve_progress_path(
                "",
                "outputs/from_plan/progress.json",
                "outputs/from_dataset",
            ),
            Path("outputs/from_plan/progress.json"),
        )
        self.assertEqual(
            resolve_progress_path(
                "",
                "",
                "outputs/from_dataset",
            ),
            Path("outputs/from_dataset/meta/gesture_trigger_progress.json"),
        )

    def test_structured_mode_is_parsed_from_keyvalues_not_message(self):
        snapshot = parse_collector_status(
            FakeDiagnosticStatus(
                level=0,
                message="human readable text that changes",
                values=[
                    FakeKeyValue(key="mode", value="IDLE"),
                    FakeKeyValue(key="dataset_root", value="outputs/demo"),
                ],
            ),
            12.0,
        )

        self.assertEqual(snapshot.mode, "IDLE")
        self.assertEqual(snapshot.dataset_root, "outputs/demo")
        self.assertEqual(snapshot.message, "human readable text that changes")

    def test_status_freshness_uses_timestamp(self):
        snapshot = parse_collector_status(
            FakeDiagnosticStatus(
                level=0,
                message="ready",
                values=[FakeKeyValue(key="mode", value="IDLE")],
            ),
            10.0,
        )

        self.assertTrue(collector_status_is_fresh(snapshot, 12.0, 3.0))
        self.assertFalse(collector_status_is_fresh(snapshot, 14.5, 3.0))

    def test_diagnostic_byte_level_round_trips(self):
        snapshot = parse_collector_status(
            FakeDiagnosticStatus(
                level=b"\x01",
                message="warning",
                values=[FakeKeyValue(key="mode", value="IDLE")],
            ),
            10.0,
        )

        self.assertEqual(snapshot.level, 1)
        self.assertEqual(diagnostic_level_number(bytearray([2])), 2)
        self.assertEqual(diagnostic_level_value(3), b"\x03")
        with self.assertRaisesRegex(ValueError, "exactly one byte"):
            diagnostic_level_number(b"")

    def test_armed_flag_only_tracks_armed_state(self):
        self.assertTrue(attempt_state_is_armed(AttemptState.ARMED))
        self.assertFalse(attempt_state_is_armed(AttemptState.RECORDING))

    def test_active_gesture_l2_tracks_current_phase(self):
        detections = {
            "ready": DetectionResult(False, 0.11),
            "start": DetectionResult(False, 0.22),
            "end": DetectionResult(False, 0.33),
        }

        self.assertEqual(active_gesture_l2(AttemptState.WAITING_READY, detections), 0.11)
        self.assertEqual(active_gesture_l2(AttemptState.ARMED, detections), 0.22)
        self.assertEqual(active_gesture_l2(AttemptState.RECORDING, detections), 0.33)

    def test_tail_frame_budget_uses_collector_fps(self):
        self.assertEqual(detection_tail_frames(0.2, 50.0), 10)
        with self.assertRaisesRegex(ValueError, "collector_fps must be > 0"):
            detection_tail_frames(0.2, 0.0)

    def test_update_detector_safely_returns_error_on_vector_length_mismatch(self):
        detector = GestureConditionDetector(
            [0.0, 0.0],
            GestureCondition(
                reference_name="start",
                threshold_l2=0.1,
                stable_samples=1,
            ),
        )

        result, error = update_detector_safely(detector, [0.0, 0.0, 0.0], 0.0)

        self.assertFalse(result.triggered)
        self.assertIn("gesture vector length mismatch", error)


if __name__ == "__main__":
    unittest.main()
