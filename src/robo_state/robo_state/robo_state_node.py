"""ROS2 node that publishes normalized StepIt robot state samples."""

from __future__ import annotations

import time
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
        self._output_topic = output_topic

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
        self._last_stepit_input_monotonic_by_source: dict[str, float] = {}
        self._stepit_input_relog_after_sec = max(
            2.0, self._assembler.max_cache_age_sec * 5.0
        )
        self._published_sample_count = 0
        self._last_sample_log_monotonic_sec = 0.0
        self._last_status_level_value: int | None = None
        self._last_status_log_message = ""
        self._last_status_log_monotonic_sec = 0.0

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
            self._field_callback("target_joint_pos"),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Float32MultiArray,
            self._topic("field/aligned_target_pos"),
            self._field_callback("aligned_target_pos", publish_on_update=True),
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
            self._log_stepit_input_received(field_name)
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
        self._log_stepit_input_received("joint_states")
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
        self._log_stepit_input_received("imu")
        self._assembler.update_imu(msg, self._now_sec())

    def _on_stepit_status(self, msg: DiagnosticStatus) -> None:
        self._log_stepit_input_received("status")
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
        self._published_sample_count += 1
        self._log_sample_published_throttled()
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
        msg.aligned_target_pos = sample.aligned_target_pos
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
        stepit_level = DiagnosticStatus.OK
        if self._latest_stepit_status is not None:
            stepit_level = self._latest_stepit_status.level
        status_level = max(
            _diagnostic_level_number(level),
            _diagnostic_level_number(stepit_level),
        )
        status.level = _diagnostic_level_value(status_level)
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
        self._log_status_recovery(status.level)
        self._log_status_issue_throttled(status.level, message)
        self._last_status_level_value = _diagnostic_level_number(status.level)

    def _log_stepit_input_received(self, source: str) -> None:
        now = time.monotonic()
        previous = self._last_stepit_input_monotonic_by_source.get(source)
        self._last_stepit_input_monotonic_by_source[source] = now
        if previous is not None and now - previous <= self._stepit_input_relog_after_sec:
            return

        suffix = ""
        if previous is not None:
            suffix = f" after {now - previous:.1f}s gap"
        self.get_logger().info(
            f"received StepIt input from {self._source_topic(source)}{suffix}"
        )

    def _log_sample_published_throttled(self) -> None:
        now = time.monotonic()
        if (
            self._published_sample_count != 1
            and now - self._last_sample_log_monotonic_sec <= 5.0
        ):
            return
        self._last_sample_log_monotonic_sec = now
        self.get_logger().info(
            "publishing robo_state samples from StepIt to "
            f"{self._output_topic}: count={self._published_sample_count}"
        )

    def _log_status_recovery(self, level: int) -> None:
        current_level = _diagnostic_level_number(level)
        if current_level != _diagnostic_level_number(DiagnosticStatus.OK):
            return
        if self._last_status_level_value is None:
            return
        if self._last_status_level_value == _diagnostic_level_number(
            DiagnosticStatus.OK
        ):
            return
        self.get_logger().info(
            "robo_state recovered; publishing complete samples to "
            f"{self._output_topic}"
        )

    def _log_status_issue_throttled(self, level: int, message: str) -> None:
        if _diagnostic_level_number(level) == _diagnostic_level_number(
            DiagnosticStatus.OK
        ):
            return

        now = time.monotonic()
        if (
            message == self._last_status_log_message
            and now - self._last_status_log_monotonic_sec <= 5.0
        ):
            return

        self._last_status_log_message = message
        self._last_status_log_monotonic_sec = now
        if _diagnostic_level_number(level) == _diagnostic_level_number(
            DiagnosticStatus.ERROR
        ):
            self.get_logger().error(message)
        else:
            self.get_logger().warn(message)

    def _source_topic(self, source: str) -> str:
        if source in {"joint_states", "imu", "status"}:
            return self._topic(source)
        if source == "target_joint_pos":
            return self._topic("field/last_target_joint_pos")
        if source == "aligned_target_pos":
            return self._topic("field/aligned_target_pos")
        return self._topic(f"field/{source}")

    def _topic(self, suffix: str) -> str:
        return f"{self._stepit_ns}/{suffix.lstrip('/')}"

    @staticmethod
    def _normalize_namespace(namespace: str) -> str:
        stripped = namespace.strip().strip("/")
        return f"/{stripped}" if stripped else ""

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def _diagnostic_level_number(level: object) -> int:
    if isinstance(level, bytes):
        if len(level) != 1:
            raise ValueError(f"invalid DiagnosticStatus level bytes: {level!r}")
        return int(level[0])
    if isinstance(level, bytearray):
        if len(level) != 1:
            raise ValueError(f"invalid DiagnosticStatus level bytes: {level!r}")
        return int(level[0])
    return int(level)


def _diagnostic_level_value(level: object) -> bytes:
    return bytes([_diagnostic_level_number(level)])


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: RoboStateNode | None = None
    try:
        node = RoboStateNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
