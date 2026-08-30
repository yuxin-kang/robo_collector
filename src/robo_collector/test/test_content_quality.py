import unittest
from fractions import Fraction

from robo_collector.content_quality import (
    ContentQualityPolicy,
    RobotObservation,
    VideoObservation,
    build_content_quality,
)


class ContentQualityTest(unittest.TestCase):
    def test_ready_evidence_is_deterministic(self):
        inputs = [
            VideoObservation("wrist", 1, 80, sharpness=5, content_sha256="b" * 64),
            VideoObservation("wrist", 0, 90, sharpness=6, content_sha256="a" * 64),
        ]
        robot = [RobotObservation("robot", 0, 1, (0.0, 1.0))]
        first = build_content_quality(inputs, robot)
        second = build_content_quality(reversed(inputs), robot)
        self.assertEqual(first, second)
        self.assertEqual(first["canonical_status"], "READY")
        self.assertEqual(len(first["rules"]), 11)

    def test_corrupt_video_and_nonfinite_robot_reject(self):
        quality = build_content_quality(
            [VideoObservation("wrist", 0, 80, corrupt=True)],
            [RobotObservation("robot", 0, 1, (float("nan"),))],
        )
        self.assertEqual(quality["canonical_status"], "REJECT")
        self.assertEqual(quality["summary"]["failed"], "2")

    def test_warning_fail_is_review(self):
        policy = ContentQualityPolicy(frozen_run_max=1)
        quality = build_content_quality(
            [
                VideoObservation("wrist", 0, 80, content_sha256="a" * 64),
                VideoObservation("wrist", 1, 80, content_sha256="a" * 64),
            ],
            policy=policy,
        )
        self.assertEqual(quality["canonical_status"], "REVIEW")

    def test_structural_failure_has_highest_precedence(self):
        quality = build_content_quality(
            [VideoObservation("wrist", 0, 80, corrupt=True)],
            structural_valid=False,
        )
        self.assertEqual(quality["canonical_status"], "QUARANTINED")

    def test_robot_range_stuck_discontinuity_jerk_and_saturation(self):
        policy = ContentQualityPolicy(
            stuck_run_max=0,
            discontinuity_max=Fraction(1),
            jerk_max=Fraction(1),
            action_saturation_fraction_max=Fraction(0),
        )
        quality = build_content_quality(
            robot=[
                RobotObservation("robot", 0, 0, (0.0,), (-1.0,), (1.0,), True),
                RobotObservation("robot", 1, 1, (0.0,), (-1.0,), (1.0,), True),
                RobotObservation("robot", 2, 2, (3.0,), (-1.0,), (1.0,), True),
            ],
            policy=policy,
        )
        failed = {
            rule["rule_id"] for rule in quality["rules"] if rule["result"] == "FAIL"
        }
        self.assertTrue(
            {"robot.range", "robot.stuck", "robot.discontinuity", "robot.jerk"}
            <= failed
        )


if __name__ == "__main__":
    unittest.main()
