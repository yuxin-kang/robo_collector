"""ROS2 node that records validated RoboState samples into LeRobot episodes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import rclpy
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from robo_collector_msgs.msg import RecordCommand
from robo_state_msgs.msg import RoboStateSample

from .camera_cache import CameraFrameCache, parse_camera_streams
from .collector_state import CollectorMode, RecordStateMachine
from .field_config import FieldConfigError, load_optional_field_selection
from .lerobot_dataset import LeRobotV21Writer, RobotFrame


@dataclass(frozen=True)
class CachedStateSample:
    msg: RoboStateSample
    received_monotonic_sec: float


class LeRobotCollectorNode(Node):
    """Waits for START/STOP commands and records aligned state + RGB frames."""

    def __init__(self) -> None:
        super().__init__("lerobot_collector_node")

        self.declare_parameter("robo_state_topic", "/robo_state/sample")
        self.declare_parameter("record_command_topic", "/robo_collector/record_command")
        self.declare_parameter("status_topic", "/robo_collector/status")
        self.declare_parameter("camera_host", "192.168.123.164")
        self.declare_parameter("camera_port", 5555)
        self.declare_parameter("camera_stream", "")
        self.declare_parameter("camera_streams", "head,ego_view")
        self.declare_parameter("dataset_name", "")
        self.declare_parameter("root_output_dir", "outputs")
        self.declare_parameter("field_config_path", "")
        self.declare_parameter("fps", 50)
        self.declare_parameter("max_state_age_sec", 0.2)
        self.declare_parameter("max_camera_age_sec", 0.2)

        self._fps = int(self.get_parameter("fps").value)
        self._robo_state_topic = str(self.get_parameter("robo_state_topic").value)
        self._max_state_age_sec = float(
            self.get_parameter("max_state_age_sec").value
        )
        self._max_camera_age_sec = float(
            self.get_parameter("max_camera_age_sec").value
        )
        legacy_camera_stream = str(self.get_parameter("camera_stream").value).strip()
        if legacy_camera_stream:
            camera_streams = [legacy_camera_stream]
        else:
            camera_streams = parse_camera_streams(
                self.get_parameter("camera_streams").value
            )
        dataset_name = str(self.get_parameter("dataset_name").value).strip() or None
        field_config_path = str(self.get_parameter("field_config_path").value).strip()
        try:
            field_selection = load_optional_field_selection(field_config_path)
        except FieldConfigError as exc:
            message = f"invalid field_config_path: {exc}"
            self.get_logger().error(message)
            raise RuntimeError(message) from exc

        self._state_machine = RecordStateMachine()
        self._writer = LeRobotV21Writer(
            str(self.get_parameter("root_output_dir").value),
            dataset_name=dataset_name,
            fps=self._fps,
            camera_keys=[
                f"observation.images.{stream}" for stream in camera_streams
            ],
            field_selection=field_selection,
        )
        self._latest_state: CachedStateSample | None = None
        self._last_warn_message = ""
        self._last_warn_monotonic_sec = 0.0
        self._last_status_log_message = ""
        self._last_status_log_monotonic_sec = 0.0
        self._state_sample_count = 0
        self._last_state_sample_log_monotonic_sec = 0.0

        qos = QoSProfile(depth=10)
        self._status_pub = self.create_publisher(
            DiagnosticStatus, str(self.get_parameter("status_topic").value), qos
        )
        self.create_subscription(
            RoboStateSample,
            self._robo_state_topic,
            self._on_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            RecordCommand,
            str(self.get_parameter("record_command_topic").value),
            self._on_record_command,
            qos,
        )

        self._camera_cache = CameraFrameCache(
            str(self.get_parameter("camera_host").value),
            int(self.get_parameter("camera_port").value),
            camera_streams,
            self.get_logger(),
        )
        self._camera_cache.start()

        self._record_timer = self.create_timer(1.0 / self._fps, self._record_tick)
        self._status_timer = self.create_timer(1.0, self._publish_periodic_status)
        self.get_logger().info(
            "collector ready; waiting for START on "
            f"{self.get_parameter('record_command_topic').value}; "
            f"dataset root will be {self._writer.root}; "
            f"camera streams={','.join(camera_streams)}; "
            f"field config={field_config_path or '<legacy all fields>'}"
        )
        self._publish_status(DiagnosticStatus.OK, "IDLE: waiting for START")

    def destroy_node(self) -> bool:
        self._camera_cache.stop()
        return super().destroy_node()

    def _on_state(self, msg: RoboStateSample) -> None:
        now = time.monotonic()
        was_unavailable_or_stale = self._latest_state is None or (
            now - self._latest_state.received_monotonic_sec > self._max_state_age_sec
        )
        self._latest_state = CachedStateSample(msg=msg, received_monotonic_sec=now)
        self._state_sample_count += 1
        self._log_state_sample_received_throttled(force=was_unavailable_or_stale)

    def _on_record_command(self, msg: RecordCommand) -> None:
        result = self._state_machine.handle_command(
            int(msg.command),
            task_prompt=msg.task_prompt,
            episode_id=msg.episode_id,
            now_sec=self._now_sec(),
        )
        if result.should_start and result.session is not None:
            try:
                episode_index = self._writer.start_episode(
                    result.session.task_prompt, result.session.episode_id
                )
            except Exception as exc:
                self._state_machine = RecordStateMachine()
                self._publish_status(
                    DiagnosticStatus.ERROR, f"failed to start episode: {exc}"
                )
                return
            self._publish_status(
                DiagnosticStatus.OK,
                f"RECORDING episode {episode_index}: {result.session.task_prompt}",
            )
            return

        if result.should_discard:
            try:
                self._writer.discard_episode()
                self._state_machine.mark_discarded()
                self._publish_status(DiagnosticStatus.OK, "DISCARD complete; IDLE")
            except Exception as exc:
                self._publish_status(DiagnosticStatus.ERROR, f"discard failed: {exc}")
            return

        self._publish_status(_diagnostic_level(result.level), result.message)

    def _record_tick(self) -> None:
        if self._state_machine.mode == CollectorMode.NEED_TO_SAVE:
            self._save_episode()
            return
        if self._state_machine.mode == CollectorMode.FAILED:
            self._publish_failed_status_throttled()
            return
        if self._state_machine.mode != CollectorMode.RECORDING:
            return

        now = time.monotonic()
        state = self._latest_state
        camera_bundle = self._camera_cache.latest()
        if state is None:
            self._publish_warn_throttled("missing robo_state sample")
            return
        if camera_bundle is None:
            self._publish_warn_throttled("missing complete camera frame bundle")
            return

        state_age = now - state.received_monotonic_sec
        camera_age = now - camera_bundle.received_monotonic_sec
        if state_age > self._max_state_age_sec:
            self._publish_warn_throttled(
                f"stale robo_state sample: {state_age:.3f}s old"
            )
            return
        if camera_age > self._max_camera_age_sec:
            self._publish_warn_throttled(
                f"stale camera frame: {camera_age:.3f}s old"
            )
            return

        try:
            self._writer.add_frame(
                _robot_frame_from_msg(state.msg), camera_bundle.images
            )
        except Exception as exc:
            reason = self._writer.active_failed_reason or str(exc)
            self._state_machine.mark_failed(reason)
            self._publish_status(
                DiagnosticStatus.ERROR,
                f"recording failed; DISCARD required: {reason}",
            )

    def _save_episode(self) -> None:
        try:
            result = self._writer.save_episode()
        except Exception as exc:
            reason = self._writer.active_failed_reason or str(exc)
            self._state_machine.mark_failed(reason)
            self._publish_status(
                DiagnosticStatus.ERROR,
                f"save failed; DISCARD required: {reason}",
            )
            return

        self._state_machine.mark_saved()
        level = DiagnosticStatus.OK if result.saved else DiagnosticStatus.WARN
        self._publish_status(
            level,
            (
                f"{result.message}: episode={result.episode_index}, "
                f"frames={result.frame_count}"
            ),
        )

    def _publish_periodic_status(self) -> None:
        message = self._state_machine.mode.value
        level = DiagnosticStatus.OK
        if self._state_machine.mode == CollectorMode.RECORDING:
            message = (
                f"RECORDING episode={self._writer.active_episode_index} "
                f"frames={self._writer.active_frame_count}"
            )
        elif self._state_machine.mode == CollectorMode.FAILED:
            message = (
                "FAILED: DISCARD required: "
                f"{self._state_machine.failure_reason}"
            )
            level = DiagnosticStatus.ERROR
        warnings = [
            warning
            for warning in (self._state_warning(), self._camera_warning())
            if warning is not None
        ]
        if warnings:
            if level != DiagnosticStatus.ERROR:
                level = DiagnosticStatus.WARN
            message = f"{message}; {'; '.join(warnings)}"
        self._publish_status(level, message)

    def _publish_failed_status_throttled(self) -> None:
        self._publish_status_throttled(
            DiagnosticStatus.ERROR,
            "FAILED: DISCARD required: "
            f"{self._state_machine.failure_reason}",
        )

    def _publish_warn_throttled(self, message: str) -> None:
        self._publish_status_throttled(DiagnosticStatus.WARN, message)

    def _publish_status_throttled(self, level: Any, message: str) -> None:
        now = time.monotonic()
        if message != self._last_warn_message or now - self._last_warn_monotonic_sec > 1.0:
            self._last_warn_message = message
            self._last_warn_monotonic_sec = now
            self._publish_status(level, message)

    def _log_state_sample_received_throttled(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and self._state_sample_count != 1
            and now - self._last_state_sample_log_monotonic_sec <= 5.0
        ):
            return
        self._last_state_sample_log_monotonic_sec = now
        self.get_logger().info(
            "receiving robo_state samples on "
            f"{self._robo_state_topic}: count={self._state_sample_count}"
        )

    def _publish_status(self, level: Any, message: str) -> None:
        state_age = ""
        if self._latest_state is not None:
            state_age = (
                f"{time.monotonic() - self._latest_state.received_monotonic_sec:.3f}"
            )
        status = DiagnosticStatus()
        status.level = _diagnostic_level_value(level)
        status.name = self.get_name()
        status.hardware_id = "robo_collector"
        status.message = message
        status.values = [
            KeyValue(key="mode", value=self._state_machine.mode.value),
            KeyValue(key="dataset_root", value=str(self._writer.root)),
            KeyValue(key="fps", value=str(self._fps)),
            KeyValue(key="active_episode", value=str(self._writer.active_episode_index)),
            KeyValue(key="active_frames", value=str(self._writer.active_frame_count)),
            KeyValue(key="max_state_age_sec", value=str(self._max_state_age_sec)),
            KeyValue(key="max_camera_age_sec", value=str(self._max_camera_age_sec)),
            KeyValue(
                key="robo_state_available",
                value=str(self._latest_state is not None),
            ),
            KeyValue(key="robo_state_age_sec", value=state_age),
            KeyValue(key="camera_streams", value=",".join(self._camera_cache.streams)),
            KeyValue(key="camera_error", value=self._camera_cache.last_error),
        ]
        self._status_pub.publish(status)
        self._log_status_issue_throttled(status.level, message)

    def _log_status_issue_throttled(self, level: Any, message: str) -> None:
        level_value = _diagnostic_level_value(level)
        if level_value == _diagnostic_level_value(DiagnosticStatus.OK):
            return

        now = time.monotonic()
        if (
            message == self._last_status_log_message
            and now - self._last_status_log_monotonic_sec <= 5.0
        ):
            return

        self._last_status_log_message = message
        self._last_status_log_monotonic_sec = now
        if level_value == _diagnostic_level_value(DiagnosticStatus.ERROR):
            self.get_logger().error(message)
        else:
            self.get_logger().warn(message)

    def _state_warning(self) -> str | None:
        state = self._latest_state
        if state is None:
            return "robo_state sample unavailable; check robo_state_node status"
        state_age = time.monotonic() - state.received_monotonic_sec
        if state_age > self._max_state_age_sec:
            return f"stale robo_state sample: {state_age:.3f}s old; check robo_state_node"
        return None

    def _camera_warning(self) -> str | None:
        camera_error = self._camera_cache.last_error
        if camera_error:
            return camera_error
        camera_bundle = self._camera_cache.latest()
        if camera_bundle is None:
            return "complete camera frame bundle unavailable"
        camera_age = time.monotonic() - camera_bundle.received_monotonic_sec
        if camera_age > self._max_camera_age_sec:
            return f"stale camera frame bundle: {camera_age:.3f}s old"
        return None

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def _robot_frame_from_msg(msg: RoboStateSample) -> RobotFrame:
    imu = msg.imu
    policy_state = msg.policy_state
    return RobotFrame(
        joint_position=[float(value) for value in msg.robot_state.joint_pos],
        joint_velocity=[float(value) for value in msg.robot_state.joint_vel],
        joint_torque=[float(value) for value in msg.robot_state.joint_torque],
        imu_angular_velocity=[
            float(imu.angular_velocity.x),
            float(imu.angular_velocity.y),
            float(imu.angular_velocity.z),
        ],
        imu_linear_acceleration=[
            float(imu.linear_acceleration.x),
            float(imu.linear_acceleration.y),
            float(imu.linear_acceleration.z),
        ],
        projected_gravity_or_quat=[
            float(imu.orientation.x),
            float(imu.orientation.y),
            float(imu.orientation.z),
            float(imu.orientation.w),
        ],
        target_joint_pos=[float(value) for value in msg.target_joint_pos],
        policy_action=[float(value) for value in msg.action],
        aligned_target_pos=[float(value) for value in msg.aligned_target_pos],
        policy_state={
            "relative_ori_6d": [
                float(value) for value in policy_state.relative_ori_6d
            ],
            "motion_anchor_lin_vel_b": [
                float(value) for value in policy_state.motion_anchor_lin_vel_b
            ],
            "motion_anchor_ang_vel_b": [
                float(value) for value in policy_state.motion_anchor_ang_vel_b
            ],
            "ang_vel_history": [
                float(value) for value in policy_state.ang_vel_history
            ],
            "gravity_history": [
                float(value) for value in policy_state.gravity_history
            ],
            "joint_pos_rel_history": [
                float(value) for value in policy_state.joint_pos_rel_history
            ],
            "joint_vel_history": [
                float(value) for value in policy_state.joint_vel_history
            ],
            "action_history": [
                float(value) for value in policy_state.action_history
            ],
        },
        joint_names=list(msg.robot_state.joint_names),
    )


def _diagnostic_level(level: str) -> bytes:
    if level == "ERROR":
        return _diagnostic_level_value(DiagnosticStatus.ERROR)
    if level == "WARN":
        return _diagnostic_level_value(DiagnosticStatus.WARN)
    return _diagnostic_level_value(DiagnosticStatus.OK)


def _diagnostic_level_value(level: Any) -> bytes:
    if isinstance(level, bytes):
        if len(level) != 1:
            raise ValueError(f"invalid DiagnosticStatus level bytes: {level!r}")
        return level
    if isinstance(level, bytearray):
        if len(level) != 1:
            raise ValueError(f"invalid DiagnosticStatus level bytes: {level!r}")
        return bytes(level)
    if isinstance(level, int):
        return bytes([level])
    return bytes([int(level)])


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: LeRobotCollectorNode | None = None
    try:
        node = LeRobotCollectorNode()
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
