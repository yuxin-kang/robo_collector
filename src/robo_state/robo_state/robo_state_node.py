"""ROS2 node that publishes normalized StepIt robot state samples."""

from __future__ import annotations

from typing import Callable

import rclpy
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray

from robo_state_msgs.msg import PolicyState, RoboStateSample, RobotLowState

from .state_builder import (
    POLICY_FIELD_SPECS,
    RoboStateAssembler,
    RobotLowStateData,
    SampleData,
    ValidationError,
    parse_joint_state,
)


class RoboStateNode(Node):
    """Subscribe to StepIt topics and publish one typed sample per target update."""

    def __init__(self) -> None:
        super().__init__("robo_state_node")

        self.declare_parameter("stepit_ns", "/stepit")
        self.declare_parameter("output_topic", "/robo_state/sample")
        self.declare_parameter("status_topic", "/robo_state/status")
        self.declare_parameter("publish_only_when_complete", True)
        self.declare_parameter("validate_observation", True)
        self.declare_parameter("max_cache_age_sec", 0.2)

        self._stepit_ns = self._normalize_namespace(
            str(self.get_parameter("stepit_ns").value)
        )
        output_topic = str(self.get_parameter("output_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)

        self._assembler = RoboStateAssembler(
            max_cache_age_sec=float(self.get_parameter("max_cache_age_sec").value),
            publish_only_when_complete=bool(
                self.get_parameter("publish_only_when_complete").value
            ),
            validate_observation=bool(
                self.get_parameter("validate_observation").value
            ),
        )
        self._latest_stepit_status: DiagnosticStatus | None = None

        publisher_qos = QoSProfile(depth=10)
        self._sample_pub = self.create_publisher(
            RoboStateSample, output_topic, publisher_qos
        )
        self._status_pub = self.create_publisher(
            DiagnosticStatus, status_topic, publisher_qos
        )
        self._status_timer = self.create_timer(1.0, self._publish_periodic_status)

        self._subscribe_inputs()
        self.get_logger().info(
            "publishing normalized StepIt state from "
            f"{self._stepit_ns} to {output_topic}"
        )

    def _subscribe_inputs(self) -> None:
        self.create_subscription(
            JointState,
            self._topic("joint_states"),
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu, self._topic("imu"), self._on_imu, qos_profile_sensor_data
        )
        self.create_subscription(
            DiagnosticStatus,
            self._topic("status"),
            self._on_stepit_status,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Float32MultiArray,
            self._topic("field/last_target_joint_pos"),
            self._field_callback("target_joint_pos", publish_on_update=True),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Float32MultiArray,
            self._topic("field/action"),
            self._field_callback("action"),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Float32MultiArray,
            self._topic("field/observation"),
            self._field_callback("observation"),
            qos_profile_sensor_data,
        )
        for spec in POLICY_FIELD_SPECS:
            self.create_subscription(
                Float32MultiArray,
                self._topic(f"field/{spec.name}"),
                self._field_callback(spec.name),
                qos_profile_sensor_data,
            )

    def _field_callback(
        self, field_name: str, *, publish_on_update: bool = False
    ) -> Callable[[Float32MultiArray], None]:
        def callback(msg: Float32MultiArray) -> None:
            now_sec = self._now_sec()
            try:
                self._assembler.update_field(field_name, msg.data, now_sec)
            except ValidationError as exc:
                self._publish_status(DiagnosticStatus.ERROR, str(exc), [field_name])
                return

            if publish_on_update:
                self._publish_sample_if_ready(now_sec)

        return callback

    def _on_joint_state(self, msg: JointState) -> None:
        now_sec = self._now_sec()
        try:
            robot_state = parse_joint_state(
                msg.name, msg.position, msg.velocity, msg.effort
            )
        except ValidationError as exc:
            self._publish_status(DiagnosticStatus.ERROR, str(exc), ["joint_states"])
            return
        self._assembler.update_robot_state(robot_state, now_sec)

    def _on_imu(self, msg: Imu) -> None:
        self._assembler.update_imu(msg, self._now_sec())

    def _on_stepit_status(self, msg: DiagnosticStatus) -> None:
        self._latest_stepit_status = msg

    def _publish_sample_if_ready(self, now_sec: float) -> None:
        result = self._assembler.build_sample(now_sec)
        if result.sample is None:
            level = (
                DiagnosticStatus.ERROR
                if result.level == "ERROR"
                else DiagnosticStatus.WARN
            )
            self._publish_status(level, result.message, result.issues)
            return

        msg = self._to_sample_msg(result.sample)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "stepit"
        self._sample_pub.publish(msg)
        self._publish_status(DiagnosticStatus.OK, result.message, [])

    def _publish_periodic_status(self) -> None:
        result = self._assembler.build_sample(self._now_sec())
        if result.sample is None:
            level = (
                DiagnosticStatus.ERROR
                if result.level == "ERROR"
                else DiagnosticStatus.WARN
            )
            self._publish_status(level, result.message, result.issues)
            return
        self._publish_status(DiagnosticStatus.OK, "ready", [])

    def _to_sample_msg(self, sample: SampleData) -> RoboStateSample:
        msg = RoboStateSample()
        msg.policy_state = self._to_policy_msg(sample)
        msg.robot_state = self._to_robot_state_msg(sample.robot_state)
        msg.imu = sample.imu if sample.imu is not None else Imu()
        msg.target_joint_pos = sample.target_joint_pos
        msg.action = sample.action
        msg.stepit_observation = sample.stepit_observation
        msg.observation_l2_error = float(sample.observation_l2_error)
        msg.missing_optional_fields = sample.missing_optional_fields
        return msg

    def _to_policy_msg(self, sample: SampleData) -> PolicyState:
        msg = PolicyState()
        for spec in POLICY_FIELD_SPECS:
            setattr(msg, spec.name, sample.policy_fields[spec.name])
        msg.flattened = sample.policy_flattened
        return msg

    def _to_robot_state_msg(self, state: RobotLowStateData) -> RobotLowState:
        msg = RobotLowState()
        msg.joint_names = state.joint_names
        msg.joint_pos = state.joint_pos
        msg.joint_vel = state.joint_vel
        msg.joint_torque = state.joint_torque
        msg.cmd_joint_pos = state.cmd_joint_pos
        msg.cmd_joint_vel = state.cmd_joint_vel
        msg.cmd_joint_torque = state.cmd_joint_torque
        msg.kp = state.kp
        msg.kd = state.kd
        msg.desired_torque = state.desired_torque
        msg.foot_names = state.foot_names
        msg.foot_force = state.foot_force
        return msg

    def _publish_status(self, level: int, message: str, issues: list[str]) -> None:
        status = DiagnosticStatus()
        stepit_level = (
            self._latest_stepit_status.level
            if self._latest_stepit_status is not None
            else DiagnosticStatus.OK
        )
        status.level = max(level, stepit_level)
        status.name = self.get_name()
        status.hardware_id = "stepit"
        status.message = message
        status.values = [
            KeyValue(
                key="publish_only_when_complete",
                value=str(self._assembler.publish_only_when_complete),
            ),
            KeyValue(
                key="validate_observation",
                value=str(self._assembler.validate_observation),
            ),
            KeyValue(
                key="max_cache_age_sec",
                value=str(self._assembler.max_cache_age_sec),
            ),
        ]
        if self._latest_stepit_status is not None:
            status.values.extend(
                [
                    KeyValue(
                        key="stepit_status_level",
                        value=str(self._latest_stepit_status.level),
                    ),
                    KeyValue(
                        key="stepit_status_message",
                        value=self._latest_stepit_status.message,
                    ),
                ]
            )
        for index, issue in enumerate(issues):
            status.values.append(KeyValue(key=f"issue_{index}", value=issue))
        self._status_pub.publish(status)

    def _topic(self, suffix: str) -> str:
        return f"{self._stepit_ns}/{suffix.lstrip('/')}"

    @staticmethod
    def _normalize_namespace(namespace: str) -> str:
        stripped = namespace.strip().strip("/")
        return f"/{stripped}" if stripped else ""

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RoboStateNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
