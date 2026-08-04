import unittest
from types import SimpleNamespace

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
    observation_l2_error,
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

    def test_non_finite_field_value_is_rejected(self):
        assembler = RoboStateAssembler()

        for non_finite in (float("nan"), float("inf"), float("-inf")):
            values = [0.0] * DOF
            values[4] = non_finite
            with self.subTest(non_finite=non_finite):
                with self.assertRaisesRegex(ValidationError, "must be finite"):
                    assembler.update_field("action", values, 1.0)

    def test_non_finite_joint_state_value_is_rejected(self):
        names, position, velocity, effort = _joint_state_parts()
        velocity[3] = float("nan")

        with self.assertRaisesRegex(
            ValidationError, r"joint_states\.velocity\[3\] must be finite"
        ):
            parse_joint_state(names, position, velocity, effort)

    def test_non_finite_imu_value_is_rejected(self):
        assembler = RoboStateAssembler()
        imu = _imu(linear_acceleration_x=float("inf"))

        with self.assertRaisesRegex(
            ValidationError, r"imu\.linear_acceleration\.x must be finite"
        ):
            assembler.update_imu(imu, 1.0)

    def test_stale_optional_fields_are_defaulted_and_reported(self):
        assembler = RoboStateAssembler(max_cache_age_sec=0.5)
        now_sec = 10.0
        _populate_required_inputs(assembler, now_sec)
        assembler.update_field("action", [7.0] * DOF, now_sec - 1.0)
        assembler.update_field("target_joint_pos", [8.0] * DOF, now_sec)

        result = assembler.build_sample(now_sec)

        self.assertIsNotNone(result.sample)
        self.assertEqual(result.sample.action, [0.0] * DOF)
        self.assertEqual(result.sample.target_joint_pos, [8.0] * DOF)
        self.assertIn("action", result.sample.missing_optional_fields)

    def test_required_input_cross_field_skew_rejects_sample(self):
        assembler = RoboStateAssembler(
            max_cache_age_sec=1.0, max_required_skew_sec=0.1
        )
        _populate_required_inputs(assembler, 10.0)
        assembler.update_field(
            POLICY_FIELD_SPECS[0].name,
            [0.0] * POLICY_FIELD_SPECS[0].dim,
            9.8,
        )

        result = assembler.build_sample(10.0)

        self.assertIsNone(result.sample)
        self.assertEqual(result.issues, ["required_input_skew"])
        self.assertIn("0.200000s > 0.100000s", result.message)

    def test_aligned_target_receive_time_is_sample_time_anchor(self):
        assembler = RoboStateAssembler(
            max_cache_age_sec=1.0, max_required_skew_sec=0.2
        )
        _populate_required_inputs(assembler, 10.0, aligned_stamp_sec=9.95)

        result = assembler.build_sample(10.0)

        self.assertIsNotNone(result.sample)
        self.assertEqual(result.sample.sample_stamp_sec, 9.95)

    def test_mismatched_observation_dimensions_are_not_reported_as_zero_error(self):
        assembler = RoboStateAssembler(max_cache_age_sec=1.0)
        now_sec = 10.0
        _populate_required_inputs(assembler, now_sec)
        assembler.update_field(
            "observation", [0.0] * STEPIT_OBSERVATION_DIM, now_sec
        )

        result = assembler.build_sample(now_sec)

        self.assertIsNotNone(result.sample)
        self.assertIsNone(result.sample.observation_l2_error)
        self.assertIn(
            "observation_l2_error", result.sample.missing_optional_fields
        )
        with self.assertRaisesRegex(ValidationError, "cannot compare"):
            observation_l2_error(
                [0.0] * OBSERVATION_DIM,
                [0.0] * STEPIT_OBSERVATION_DIM,
            )

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
        self.assertIsNone(result.sample.observation_l2_error)
        self.assertIn(
            "observation_l2_error", result.sample.missing_optional_fields
        )
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


def _populate_required_inputs(
    assembler: RoboStateAssembler,
    stamp_sec: float,
    *,
    aligned_stamp_sec: float | None = None,
) -> None:
    for spec in POLICY_FIELD_SPECS:
        assembler.update_field(spec.name, [0.0] * spec.dim, stamp_sec)
    assembler.update_field(
        "aligned_target_pos",
        [0.0] * ALIGNED_TARGET_POS_DIM,
        stamp_sec if aligned_stamp_sec is None else aligned_stamp_sec,
    )
    assembler.update_robot_state(RobotLowStateData.zero(), stamp_sec)
    assembler.update_imu(_imu(), stamp_sec)


def _imu(*, linear_acceleration_x: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        angular_velocity=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        linear_acceleration=SimpleNamespace(
            x=linear_acceleration_x, y=0.0, z=0.0
        ),
        orientation_covariance=[0.0] * 9,
        angular_velocity_covariance=[0.0] * 9,
        linear_acceleration_covariance=[0.0] * 9,
    )
