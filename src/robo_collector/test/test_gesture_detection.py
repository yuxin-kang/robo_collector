import unittest
from dataclasses import dataclass

from robo_collector.gesture_logic import (
    GestureConditionDetector,
    extract_gesture_vector,
    validate_reference_lengths,
)
from robo_collector.gesture_plan import GestureCondition, gesture_plan_from_payload


@dataclass
class FakeRobotState:
    joint_pos: list[float]


@dataclass
class FakeSample:
    robot_state: FakeRobotState
    aligned_target_pos: list[float]


class GestureDetectionTest(unittest.TestCase):
    def test_extract_vector_from_supported_fields(self):
        sample = FakeSample(
            robot_state=FakeRobotState([0.1, 0.2, 0.3]),
            aligned_target_pos=[1.0, 2.0, 3.0, 4.0],
        )

        self.assertEqual(
            extract_gesture_vector(sample, "robot_state.joint_pos", [0, 2]),
            [0.1, 0.3],
        )
        self.assertEqual(
            extract_gesture_vector(sample, "aligned_target_pos", [1, 3]),
            [2.0, 4.0],
        )

    def test_extract_vector_rejects_bad_index(self):
        sample = FakeSample(
            robot_state=FakeRobotState([0.1, 0.2, 0.3]),
            aligned_target_pos=[1.0, 2.0, 3.0],
        )
        with self.assertRaisesRegex(ValueError, "out of range"):
            extract_gesture_vector(sample, "robot_state.joint_pos", [3])

    def test_detector_requires_stable_samples(self):
        detector = GestureConditionDetector(
            [0.0, 0.0],
            GestureCondition(
                reference_name="start",
                threshold_l2=0.1,
                stable_samples=3,
            ),
        )
        self.assertFalse(detector.update([0.01, 0.01], 0.0).triggered)
        self.assertFalse(detector.update([0.01, 0.01], 0.1).triggered)
        self.assertTrue(detector.update([0.01, 0.01], 0.2).triggered)

    def test_detector_resets_after_above_threshold_sample(self):
        detector = GestureConditionDetector(
            [0.0, 0.0],
            GestureCondition(
                reference_name="start",
                threshold_l2=0.1,
                stable_samples=2,
            ),
        )
        self.assertFalse(detector.update([0.01, 0.01], 0.0).triggered)
        self.assertFalse(detector.update([0.5, 0.5], 0.1).triggered)
        self.assertFalse(detector.update([0.01, 0.01], 0.2).triggered)
        self.assertTrue(detector.update([0.01, 0.01], 0.3).triggered)

    def test_release_threshold_and_cooldown_prevent_double_trigger(self):
        detector = GestureConditionDetector(
            [0.0, 0.0],
            GestureCondition(
                reference_name="start",
                threshold_l2=0.1,
                stable_samples=1,
                release_threshold_l2=0.2,
                cooldown_sec=0.5,
            ),
        )
        self.assertTrue(detector.update([0.01, 0.01], 0.0).triggered)
        self.assertFalse(detector.update([0.01, 0.01], 0.1).triggered)
        self.assertFalse(detector.update([0.3, 0.3], 0.2).triggered)
        self.assertFalse(detector.update([0.01, 0.01], 0.3).triggered)
        self.assertTrue(detector.update([0.01, 0.01], 0.6).triggered)

    def test_missing_sample_does_not_trigger(self):
        detector = GestureConditionDetector(
            [0.0, 0.0],
            GestureCondition(
                reference_name="start",
                threshold_l2=0.1,
                stable_samples=1,
            ),
        )
        result = detector.update(None, 0.0)
        self.assertFalse(result.triggered)

    def test_reference_lengths_must_match_gesture_source_indices_projection(self):
        plan = gesture_plan_from_payload(
            {
                "version": 1,
                "plan_id": "handshake_set_a_20260621",
                "gesture_source": {
                    "topic": "/robo_state/sample",
                    "field": "robot_state.joint_pos",
                    "indices": [0, 1, 2],
                },
                "references": {
                    "ready": {"vector": [0.0, 0.0]},
                    "start": {"vector": [0.1, 0.1]},
                    "end": {"vector": [0.2, 0.2]},
                },
                "conditions": {
                    "task_start_condition": {
                        "reference_name": "start",
                        "threshold_l2": 0.05,
                        "stable_samples": 1,
                    },
                    "task_end_condition": {
                        "reference_name": "end",
                        "threshold_l2": 0.05,
                        "stable_samples": 1,
                    },
                    "return_to_ready_condition": {
                        "reference_name": "ready",
                        "threshold_l2": 0.05,
                        "stable_samples": 1,
                    },
                },
                "tasks": [
                    {
                        "task_slug": "shake_hand",
                        "task_prompt": "Shake hand with somebody",
                        "target_trials": 1,
                    }
                ],
            }
        )

        with self.assertRaisesRegex(ValueError, "expected 3, got 2"):
            validate_reference_lengths(plan)


if __name__ == "__main__":
    unittest.main()
