import unittest

from robo_state.state_builder import (
    ALIGNED_TARGET_POS_DIM,
    DOF,
    OBSERVATION_DIM,
    POLICY_FIELD_SPECS,
    RoboStateAssembler,
    RobotLowStateData,
    STEPIT_OBSERVATION_DIM,
    ValidationError,
    flatten_policy_fields,
    parse_joint_state,
)


class StateBuilderTest(unittest.TestCase):
    def test_policy_field_flattening_order_and_dimension(self):
        fields = {}
        expected = []
        for field_index, spec in enumerate(POLICY_FIELD_SPECS):
            values = [float(field_index)] * spec.dim
            fields[spec.name] = values
            expected.extend(values)

        flattened = flatten_policy_fields(fields)

        self.assertEqual(len(flattened), OBSERVATION_DIM)
        self.assertEqual(flattened, expected)

    def test_joint_state_parses_suffix_sections_and_foot_forces(self):
        names, position, velocity, effort = _joint_state_parts()

        parsed = parse_joint_state(names, position, velocity, effort)

        self.assertEqual(parsed.joint_names, [f"j{i}" for i in range(DOF)])
        self.assertEqual(parsed.joint_pos, [100.0 + i for i in range(DOF)])
        self.assertEqual(parsed.joint_vel, [200.0 + i for i in range(DOF)])
        self.assertEqual(parsed.joint_torque, [300.0 + i for i in range(DOF)])
        self.assertEqual(parsed.cmd_joint_pos, [400.0 + i for i in range(DOF)])
        self.assertEqual(parsed.cmd_joint_vel, [500.0 + i for i in range(DOF)])
        self.assertEqual(parsed.cmd_joint_torque, [600.0 + i for i in range(DOF)])
        self.assertEqual(parsed.kp, [700.0 + i for i in range(DOF)])
        self.assertEqual(parsed.kd, [800.0 + i for i in range(DOF)])
        self.assertEqual(parsed.desired_torque, [900.0 + i for i in range(DOF)])
        self.assertEqual(parsed.foot_names, ["LL_FOOT", "LR_FOOT"])
        self.assertEqual(parsed.foot_force, [12.5, 13.5])

    def test_missing_required_fields_do_not_build_sample(self):
        assembler = RoboStateAssembler()
        assembler.update_field("target_joint_pos", [0.0] * DOF, 1.0)

        result = assembler.build_sample(1.0)

        self.assertIsNone(result.sample)
        self.assertEqual(result.level, "WARN")
        self.assertIn("aligned_target_pos", result.issues)
        self.assertIn("relative_ori_6d", result.issues)
        self.assertIn("joint_states", result.issues)

    def test_dimension_error_rejects_bad_field(self):
        assembler = RoboStateAssembler()

        with self.assertRaisesRegex(ValidationError, "action has dimension 28"):
            assembler.update_field("action", [0.0] * (DOF - 1), 1.0)

    def test_sample_contains_aligned_target_and_selected_policy_fields(self):
        assembler = RoboStateAssembler(max_cache_age_sec=1.0)
        now_sec = 10.0

        policy_fields = {}
        for spec in POLICY_FIELD_SPECS:
            values = [0.0] * spec.dim
            policy_fields[spec.name] = values
            assembler.update_field(spec.name, values, now_sec)

        flattened = flatten_policy_fields(policy_fields)

        assembler.update_field("observation", [0.0] * STEPIT_OBSERVATION_DIM, now_sec)
        assembler.update_field("action", [0.0] * DOF, now_sec)
        assembler.update_field("target_joint_pos", [0.0] * DOF, now_sec)
        assembler.update_field(
            "aligned_target_pos", [1.0] * ALIGNED_TARGET_POS_DIM, now_sec
        )
        assembler.update_robot_state(RobotLowStateData.zero(), now_sec)
        assembler.update_imu(object(), now_sec)

        result = assembler.build_sample(now_sec)

        self.assertIsNotNone(result.sample)
        self.assertEqual(result.sample.aligned_target_pos, [1.0] * 45)
        self.assertAlmostEqual(result.sample.observation_l2_error, 0.0)
        self.assertEqual(len(result.sample.policy_flattened), OBSERVATION_DIM)
        self.assertEqual(result.sample.policy_flattened, flattened)


def _joint_state_parts():
    names = []
    position = []
    velocity = []
    effort = []

    for i in range(DOF):
        names.append(f"j{i}_joint")
        position.append(100.0 + i)
        velocity.append(200.0 + i)
        effort.append(300.0 + i)

    for foot_index, foot_name in enumerate(("LL_FOOT", "LR_FOOT")):
        names.append(foot_name)
        position.append(0.0)
        velocity.append(0.0)
        effort.append(12.5 + foot_index)

    for i in range(DOF):
        names.append(f"j{i}_cmd")
        position.append(400.0 + i)
        velocity.append(500.0 + i)
        effort.append(600.0 + i)

    for i in range(DOF):
        names.append(f"j{i}_gain")
        position.append(700.0 + i)
        velocity.append(800.0 + i)
        effort.append(900.0 + i)

    return names, position, velocity, effort
