import unittest
from types import SimpleNamespace

from robo_collector.field_config import FieldSelection, default_field_selection
from robo_collector.sample_alignment import (
    message_stamp_sec,
    required_sample_inputs,
    selected_missing_inputs,
    source_timestamp_skew_sec,
)


class SampleAlignmentTest(unittest.TestCase):
    def test_default_selection_requires_action_robot_state_and_imu(self):
        required = required_sample_inputs(default_field_selection())

        self.assertIn("action", required)
        self.assertIn("target_joint_pos", required)
        self.assertIn("joint_states", required)
        self.assertIn("imu", required)

    def test_selected_missing_inputs_only_returns_persisted_fields(self):
        selection = FieldSelection(
            target=("aligned_target_pos",),
            state=("relative_ori_6d",),
        )

        self.assertEqual(
            selected_missing_inputs(
                ["action", "target_joint_pos", "aligned_target_pos"],
                selection,
            ),
            ["aligned_target_pos"],
        )

    def test_source_timestamp_skew_requires_all_finite_timestamps(self):
        self.assertAlmostEqual(
            source_timestamp_skew_sec(10.0, [10.02, 9.98]),
            0.04,
        )
        self.assertIsNone(source_timestamp_skew_sec(None, [10.0]))
        self.assertIsNone(source_timestamp_skew_sec(10.0, [float("nan")]))

    def test_message_stamp_rejects_missing_or_zero_stamp(self):
        valid = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=20))
        )
        zero = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0))
        )

        self.assertAlmostEqual(message_stamp_sec(valid), 10.00000002)
        self.assertIsNone(message_stamp_sec(zero))
        self.assertIsNone(message_stamp_sec(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
