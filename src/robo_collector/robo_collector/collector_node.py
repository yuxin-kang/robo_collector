"""ROS2 node that records validated RoboState samples into LeRobot episodes."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import rclpy
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from robo_collector_msgs.msg import RecordCommand
from robo_state_msgs.msg import RoboStateSample

from .collector_state import CollectorMode, RecordStateMachine
from .lerobot_dataset import LeRobotV21Writer, RobotFrame


@dataclass(frozen=True)
class CachedCameraFrame:
    image: Any
    received_monotonic_sec: float
    camera_timestamp_ns: int | None


@dataclass(frozen=True)
class CachedStateSample:
    msg: RoboStateSample
    received_monotonic_sec: float


class CameraFrameCache:
    """Background reader for the ZMQ camera client."""

    def __init__(self, host: str, port: int, stream: str, logger: Any) -> None:
        self.host = host
        self.port = int(port)
        self.stream = stream
        self._logger = logger
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: CachedCameraFrame | None = None
        self._last_error = ""

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="robo_collector_camera_reader", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self) -> CachedCameraFrame | None:
        with self._lock:
            return self._frame

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def _run(self) -> None:
        try:
            from robo_collector_camera.client import CameraClient
        except ImportError as exc:
            self._record_error(
                "cannot import CameraClient; run scripts/setup_data_collection_env.sh"
            )
            self._logger.error(str(exc))
            return

        client = CameraClient(self.host, self.port)
        try:
            while not self._stop.is_set():
                packet = client.read(timeout_ms=100)
                if packet is None:
                    continue
                images = packet.get("images", {})
                image = images.get(self.stream)
                if image is None:
                    self._record_error(
                        f"camera packet missing stream '{self.stream}'"
                    )
                    continue
                timestamps = packet.get("timestamps", {})
                camera_timestamp = timestamps.get(self.stream)
                with self._lock:
                    self._frame = CachedCameraFrame(
                        image=image,
                        received_monotonic_sec=time.monotonic(),
                        camera_timestamp_ns=(
                            int(camera_timestamp)
                            if camera_timestamp is not None
                            else None
                        ),
                    )
                    self._last_error = ""
        except Exception as exc:  # pragma: no cover - hardware/runtime path
            self._record_error(f"camera reader stopped: {exc}")
            self._logger.error(f"camera reader stopped: {exc}")
        finally:
            client.close()

    def _record_error(self, message: str) -> None:
        with self._lock:
            changed = message != self._last_error
            self._last_error = message
        if changed:
            self._logger.warn(message)


class LeRobotCollectorNode(Node):
    """Waits for START/STOP commands and records aligned state + RGB frames."""

    def __init__(self) -> None:
        super().__init__("lerobot_collector_node")

        self.declare_parameter("robo_state_topic", "/robo_state/sample")
        self.declare_parameter("record_command_topic", "/robo_collector/record_command")
        self.declare_parameter("status_topic", "/robo_collector/status")
        self.declare_parameter("camera_host", "192.168.123.164")
        self.declare_parameter("camera_port", 5555)
        self.declare_parameter("camera_stream", "ego_view")
        self.declare_parameter("dataset_name", "")
        self.declare_parameter("root_output_dir", "outputs")
        self.declare_parameter("fps", 50)
        self.declare_parameter("max_state_age_sec", 0.2)
        self.declare_parameter("max_camera_age_sec", 0.2)

        self._fps = int(self.get_parameter("fps").value)
        self._max_state_age_sec = float(
            self.get_parameter("max_state_age_sec").value
        )
        self._max_camera_age_sec = float(
            self.get_parameter("max_camera_age_sec").value
        )
        camera_stream = str(self.get_parameter("camera_stream").value)
        dataset_name = str(self.get_parameter("dataset_name").value).strip() or None

        self._state_machine = RecordStateMachine()
        self._writer = LeRobotV21Writer(
            str(self.get_parameter("root_output_dir").value),
            dataset_name=dataset_name,
            fps=self._fps,
            camera_key=f"observation.images.{camera_stream}",
        )
        self._latest_state: CachedStateSample | None = None
        self._last_warn_message = ""
        self._last_warn_monotonic_sec = 0.0

        qos = QoSProfile(depth=10)
        self._status_pub = self.create_publisher(
            DiagnosticStatus, str(self.get_parameter("status_topic").value), qos
        )
        self.create_subscription(
            RoboStateSample,
            str(self.get_parameter("robo_state_topic").value),
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
            camera_stream,
            self.get_logger(),
        )
        self._camera_cache.start()

        self._record_timer = self.create_timer(1.0 / self._fps, self._record_tick)
        self._status_timer = self.create_timer(1.0, self._publish_periodic_status)
        self.get_logger().info(
            "collector ready; waiting for START on "
            f"{self.get_parameter('record_command_topic').value}; "
            f"dataset root will be {self._writer.root}"
        )
        self._publish_status(DiagnosticStatus.OK, "IDLE: waiting for START")

    def destroy_node(self) -> bool:
        self._camera_cache.stop()
        return super().destroy_node()

    def _on_state(self, msg: RoboStateSample) -> None:
        self._latest_state = CachedStateSample(
            msg=msg, received_monotonic_sec=time.monotonic()
        )

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
        if self._state_machine.mode != CollectorMode.RECORDING:
            return

        now = time.monotonic()
        state = self._latest_state
        camera = self._camera_cache.latest()
        if state is None:
            self._publish_warn_throttled("missing robo_state sample")
            return
        if camera is None:
            self._publish_warn_throttled("missing camera frame")
            return

        state_age = now - state.received_monotonic_sec
        camera_age = now - camera.received_monotonic_sec
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
            self._writer.add_frame(_robot_frame_from_msg(state.msg), camera.image)
        except Exception as exc:
            self._publish_warn_throttled(f"failed to write frame: {exc}")

    def _save_episode(self) -> None:
        try:
            result = self._writer.save_episode()
        except Exception as exc:
            self._state_machine.mark_save_failed()
            self._publish_status(DiagnosticStatus.ERROR, f"save failed: {exc}")
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
        if self._state_machine.mode == CollectorMode.RECORDING:
            message = (
                f"RECORDING episode={self._writer.active_episode_index} "
                f"frames={self._writer.active_frame_count}"
            )
        level = DiagnosticStatus.OK
        camera_warning = self._camera_warning()
        if camera_warning is not None:
            level = DiagnosticStatus.WARN
            message = f"{message}; {camera_warning}"
        self._publish_status(level, message)

    def _publish_warn_throttled(self, message: str) -> None:
        now = time.monotonic()
        if message != self._last_warn_message or now - self._last_warn_monotonic_sec > 1.0:
            self._last_warn_message = message
            self._last_warn_monotonic_sec = now
            self._publish_status(DiagnosticStatus.WARN, message)

    def _publish_status(self, level: Any, message: str) -> None:
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
            KeyValue(key="camera_error", value=self._camera_cache.last_error),
        ]
        self._status_pub.publish(status)

    def _camera_warning(self) -> str | None:
        camera_error = self._camera_cache.last_error
        if camera_error:
            return camera_error
        camera = self._camera_cache.latest()
        if camera is None:
            return "camera frame unavailable"
        camera_age = time.monotonic() - camera.received_monotonic_sec
        if camera_age > self._max_camera_age_sec:
            return f"stale camera frame: {camera_age:.3f}s old"
        return None

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def _robot_frame_from_msg(msg: RoboStateSample) -> RobotFrame:
    imu = msg.imu
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
    node = LeRobotCollectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
