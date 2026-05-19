import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from robo_collector.field_config import (
    FieldConfigError,
    field_selection_from_payload,
    load_field_selection,
)


class FieldConfigTest(unittest.TestCase):
    def test_valid_yaml_loads_and_maps_parquet_keys(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "fields.yml"
            path.write_text(
                "\n".join(
                    [
                        "target:",
                        "  - joint_position",
                        "state:",
                        "  - joint_position",
                        "  - joint_velocity",
                    ]
                ),
                encoding="utf-8",
            )

            selection = load_field_selection(path)

            self.assertEqual(selection.target, ("joint_position",))
            self.assertEqual(
                selection.target_parquet_keys, ("action.joint_position",)
            )
            self.assertEqual(
                selection.state_parquet_keys,
                (
                    "observation.state.joint_position",
                    "observation.state.joint_velocity",
                ),
            )
            self.assertFalse(selection.include_policy_action)

    def test_policy_yaml_loads_network_target_and_state_fields(self):
        selection = field_selection_from_payload(
            {
                "target": ["aligned_target_pos"],
                "state": [
                    "relative_ori_6d",
                    "motion_anchor_lin_vel_b",
                    "motion_anchor_ang_vel_b",
                    "ang_vel_history",
                    "gravity_history",
                    "joint_pos_rel_history",
                    "joint_vel_history",
                    "action_history",
                ],
            }
        )

        self.assertEqual(
            selection.target_parquet_keys, ("action.aligned_target_pos",)
        )
        self.assertEqual(
            selection.state_parquet_keys,
            (
                "observation.state.relative_ori_6d",
                "observation.state.motion_anchor_lin_vel_b",
                "observation.state.motion_anchor_ang_vel_b",
                "observation.state.ang_vel_history",
                "observation.state.gravity_history",
                "observation.state.joint_pos_rel_history",
                "observation.state.joint_vel_history",
                "observation.state.action_history",
            ),
        )

    def test_unknown_field_is_rejected(self):
        with self.assertRaisesRegex(FieldConfigError, "unsupported state field"):
            field_selection_from_payload(
                {
                    "target": ["joint_position"],
                    "state": ["joint_position", "unknown"],
                }
            )

    def test_duplicate_field_is_rejected(self):
        with self.assertRaisesRegex(FieldConfigError, "duplicate state field"):
            field_selection_from_payload(
                {
                    "target": ["joint_position"],
                    "state": ["joint_position", "joint_position"],
                }
            )

    def test_missing_target_or_state_is_rejected(self):
        cases = [
            {"state": ["joint_position"]},
            {"target": ["joint_position"]},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(
                    FieldConfigError, "top-level keys must be exactly"
                ):
                    field_selection_from_payload(payload)

    def test_empty_list_is_rejected(self):
        with self.assertRaisesRegex(
            FieldConfigError, "target must be a non-empty list"
        ):
            field_selection_from_payload({"target": [], "state": ["joint_position"]})

    def test_non_list_group_is_rejected(self):
        with self.assertRaisesRegex(
            FieldConfigError, "target must be a non-empty list"
        ):
            field_selection_from_payload(
                {"target": "joint_position", "state": ["joint_position"]}
            )

    def test_non_string_item_is_rejected(self):
        with self.assertRaisesRegex(FieldConfigError, r"state\[1\] must be a string"):
            field_selection_from_payload(
                {"target": ["joint_position"], "state": ["joint_position", 12]}
            )


if __name__ == "__main__":
    unittest.main()
