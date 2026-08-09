import unittest
from types import SimpleNamespace

from robo_collector.field_config import FieldSelection, default_field_selection
from robo_collector.sample_alignment import (
    message_stamp_sec,
    required_sample_inputs,
    selected_missing_inputs,
    selected_source_timestamps_sec,
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

    def test_selected_source_timestamps_include_selected_action(self):
        selection = default_field_selection()
        msg = SimpleNamespace(
            source_timestamp_names=[
                "joint_states",
                "imu",
                "target_joint_pos",
                "action",
            ],
            source_timestamps_sec=[10.0, 10.01, 10.02, 9.8],
        )

        timestamps = selected_source_timestamps_sec(msg, selection)

        self.assertIsNotNone(timestamps)
        self.assertIn("action", timestamps)
        self.assertAlmostEqual(
            source_timestamp_skew_sec(
                next(iter(timestamps.values())),
                [*list(timestamps.values())[1:], 10.0],
            ),
            0.22,
        )

    def test_selected_source_timestamps_reject_missing_selected_input(self):
        selection = FieldSelection(
            target=("aligned_target_pos",),
            state=("relative_ori_6d",),
        )
        msg = SimpleNamespace(
            source_timestamp_names=["aligned_target_pos"],
            source_timestamps_sec=[10.0],
        )

        self.assertIsNone(selected_source_timestamps_sec(msg, selection))

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
