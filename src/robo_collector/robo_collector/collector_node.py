"""ROS2 node that records validated RoboState samples into LeRobot episodes."""

from __future__ import annotations

import math
import shutil
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
from .collector_state import (
    CollectorMode,
    CommandFingerprint,
    CommandReceiptLedger,
    RecordStateMachine,
    recording_safety_reason,
)
from .field_config import (
    FieldConfigError,
    default_field_selection,
    load_optional_field_selection,
)
from .lerobot_dataset import LeRobotV21Writer, RobotFrame, SaveResult
from .save_worker import EpisodeSaveWorker
from .sample_alignment import (
    message_stamp_sec,
    selected_missing_inputs,
    selected_source_timestamps_sec,
    source_timestamp_skew_sec,
)


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
        self.declare_parameter("fps", 30)
        self.declare_parameter("max_state_age_sec", 0.2)
        self.declare_parameter("max_camera_age_sec", 0.2)
        self.declare_parameter("max_inter_camera_skew_sec", 0.1)
        self.declare_parameter("max_state_camera_skew_sec", 0.1)
        self.declare_parameter("max_episode_duration_sec", 600.0)
        self.declare_parameter("max_episode_frames", 18000)
        self.declare_parameter("min_free_disk_bytes", 2147483648)
        self.declare_parameter("save_shutdown_grace_sec", 10.0)

        self._fps = int(self.get_parameter("fps").value)
        self._robo_state_topic = str(self.get_parameter("robo_state_topic").value)
        self._max_state_age_sec = float(
            self.get_parameter("max_state_age_sec").value
        )
        self._max_camera_age_sec = float(
            self.get_parameter("max_camera_age_sec").value
        )
        self._max_inter_camera_skew_sec = float(
            self.get_parameter("max_inter_camera_skew_sec").value
        )
        self._max_state_camera_skew_sec = float(
            self.get_parameter("max_state_camera_skew_sec").value
        )
        self._max_episode_duration_sec = float(
            self.get_parameter("max_episode_duration_sec").value
        )
        self._max_episode_frames = int(
            self.get_parameter("max_episode_frames").value
        )
        self._min_free_disk_bytes = int(
            self.get_parameter("min_free_disk_bytes").value
        )
        self._save_shutdown_grace_sec = float(
            self.get_parameter("save_shutdown_grace_sec").value
        )
        for name, value in (
            ("fps", self._fps),
            ("max_state_age_sec", self._max_state_age_sec),
            ("max_camera_age_sec", self._max_camera_age_sec),
            ("max_inter_camera_skew_sec", self._max_inter_camera_skew_sec),
            ("max_state_camera_skew_sec", self._max_state_camera_skew_sec),
            ("max_episode_duration_sec", self._max_episode_duration_sec),
            ("save_shutdown_grace_sec", self._save_shutdown_grace_sec),
        ):
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError(f"{name} must be finite and positive")
        if self._max_episode_frames <= 0:
            raise RuntimeError("max_episode_frames must be positive")
        if self._min_free_disk_bytes < 0:
            raise RuntimeError("min_free_disk_bytes must be non-negative")
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

        self._field_selection = field_selection or default_field_selection()
        self._state_machine = RecordStateMachine()
        self._writer = LeRobotV21Writer(
            str(self.get_parameter("root_output_dir").value),
            dataset_name=dataset_name,
            fps=self._fps,
            camera_keys=[
                f"observation.images.{stream}" for stream in camera_streams
            ],
            field_selection=self._field_selection,
        )
        self._save_worker = EpisodeSaveWorker[SaveResult]()
        self._save_started_monotonic_sec: float | None = None
        self._save_finished_monotonic_sec: float | None = None
        self._save_progress_monotonic_sec: float | None = None
        self._save_progress_seq = 0
        self._save_phase = ""
        self._saving_episode_index: int | None = None
        self._saving_frame_count = 0
        self._latest_state: CachedStateSample | None = None
        self._last_warn_message = ""
        self._last_warn_monotonic_sec = 0.0
        self._last_status_log_message = ""
        self._last_status_log_monotonic_sec = 0.0
        self._last_command_id = ""
        self._last_command = ""
        self._last_command_outcome = ""
        self._last_command_episode_id = ""
        self._last_episode_id = ""
        self._last_episode_outcome = ""
        self._command_receipts = CommandReceiptLedger()
        self._state_sample_count = 0
        self._last_state_sample_log_monotonic_sec = 0.0
        self._last_recorded_camera_identity: (
            tuple[str, tuple[tuple[str, int | float | None], ...]] | None
        ) = None

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
            max_inter_camera_skew_sec=self._max_inter_camera_skew_sec,
            expected_fps=self._fps,
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
        save_finished = self._save_worker.shutdown(
            timeout=self._save_shutdown_grace_sec
        )
        if not save_finished:
            self.get_logger().warn(
                "episode save is still running after the "
                f"{self._save_shutdown_grace_sec:.3f}s shutdown grace period; "
                "waiting for the durable transaction to finish"
            )
            self._save_worker.shutdown()
        if self._state_machine.mode == CollectorMode.SAVING:
            self._poll_save_episode()
        if self._writer.active_episode_index is not None:
            try:
                self._writer.discard_episode()
            except Exception as exc:
                self.get_logger().error(
                    f"failed to discard active episode during shutdown: {exc}"
                )
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
        command_value = int(msg.command)
        command_id = str(getattr(msg, "command_id", "")).strip()
        command_episode_id = str(msg.episode_id).strip()
        command_name = _record_command_name(command_value)
        fingerprint = CommandFingerprint(
            command=command_value,
            task_prompt=str(msg.task_prompt),
            episode_id=command_episode_id,
            force=bool(getattr(msg, "force", False)),
        )
        replay = self._command_receipts.lookup(command_id, fingerprint)
        if replay is not None:
            if replay.disposition == "CONFLICT":
                self._publish_status(
                    DiagnosticStatus.ERROR,
                    "command_id reuse rejected because its payload changed: "
                    f"{command_id}",
                )
                return
            self._set_command_receipt(
                command_id=command_id,
                command=command_name,
                episode_id=command_episode_id,
                outcome=replay.outcome,
            )
            level = (
                DiagnosticStatus.ERROR
                if replay.outcome == "FAILED"
                or self._state_machine.mode == CollectorMode.FAILED
                else DiagnosticStatus.WARN
                if replay.outcome == "REJECTED"
                else DiagnosticStatus.OK
            )
            self._publish_status(
                level,
                f"{command_name} duplicate replayed as {replay.outcome}; "
                f"mode={self._state_machine.mode.value}",
            )
            return
        self._set_command_receipt(
            command_id=command_id,
            command=command_name,
            episode_id=command_episode_id,
            outcome="RECEIVED",
        )
        result = self._state_machine.handle_command(
            command_value,
            task_prompt=msg.task_prompt,
            episode_id=command_episode_id,
            force=bool(getattr(msg, "force", False)),
            now_sec=self._now_sec(),
        )
        if result.should_start and result.session is not None:
            try:
                episode_index = self._writer.start_episode(
                    result.session.task_prompt, result.session.episode_id
                )
            except Exception as exc:
                self._state_machine = RecordStateMachine()
                self._last_command_outcome = "FAILED"
                self._command_receipts.remember(
                    command_id, fingerprint, self._last_command_outcome
                )
                self._publish_status(
                    DiagnosticStatus.ERROR, f"failed to start episode: {exc}"
                )
                return
            self._clear_save_tracking()
            self._last_recorded_camera_identity = None
            self._last_command_outcome = "SUCCEEDED"
            self._command_receipts.remember(
                command_id, fingerprint, self._last_command_outcome
            )
            self._publish_status(
                DiagnosticStatus.OK,
                f"RECORDING episode {episode_index}: {result.session.task_prompt}",
            )
            return

        if result.should_discard:
            try:
                self._writer.discard_episode()
                self._state_machine.mark_discarded()
                self._last_command_outcome = "SUCCEEDED"
                self._last_episode_id = (
                    result.session.episode_id if result.session is not None else ""
                )
                self._last_episode_outcome = "DISCARDED"
                self._publish_status(DiagnosticStatus.OK, "DISCARD complete; IDLE")
            except Exception as exc:
                reason = f"discard failed: {exc}"
                self._state_machine.mark_discard_failed(reason)
                self._last_command_outcome = "FAILED"
                self._publish_status(DiagnosticStatus.ERROR, reason)
            self._command_receipts.remember(
                command_id,
                fingerprint,
                self._last_command_outcome,
                replayable=self._last_command_outcome != "FAILED",
            )
            return

        self._last_command_outcome = "SUCCEEDED" if result.accepted else "REJECTED"
        self._command_receipts.remember(
            command_id, fingerprint, self._last_command_outcome
        )
        self._publish_status(_diagnostic_level(result.level), result.message)

    def _record_tick(self) -> None:
        if self._state_machine.mode == CollectorMode.NEED_TO_SAVE:
            self._begin_save_episode()
            return
        if self._state_machine.mode == CollectorMode.SAVING:
            self._poll_save_episode()
            return
        if self._state_machine.mode == CollectorMode.FAILED:
            self._publish_failed_status_throttled()
            return
        if self._state_machine.mode != CollectorMode.RECORDING:
            return

        if self._enforce_recording_safety_limits():
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

        if camera_bundle.identity == self._last_recorded_camera_identity:
            self._publish_warn_throttled("camera frame bundle has already been recorded")
            return

        missing_selected = selected_missing_inputs(
            getattr(state.msg, "missing_optional_fields", ()),
            self._field_selection,
        )
        if missing_selected:
            self._publish_warn_throttled(
                "selected state/action field(s) missing or stale: "
                + ",".join(missing_selected)
            )
            return

        selected_timestamps = selected_source_timestamps_sec(
            state.msg, self._field_selection
        )
        if selected_timestamps is None:
            self._publish_warn_throttled(
                "selected state/action source timestamps unavailable or invalid"
            )
            return
        selected_values = list(selected_timestamps.values())
        selected_skew_sec = source_timestamp_skew_sec(
            selected_values[0], selected_values[1:]
        )
        if selected_skew_sec is None:
            self._publish_warn_throttled(
                "selected state/action source timestamp skew is invalid"
            )
            return
        if selected_skew_sec > self._max_state_camera_skew_sec:
            self._publish_warn_throttled(
                "selected state/action source timestamp skew "
                f"{selected_skew_sec:.3f}s exceeds "
                f"{self._max_state_camera_skew_sec:.3f}s"
            )
            return
        receive_skew_sec = source_timestamp_skew_sec(
            state.received_monotonic_sec,
            [camera_bundle.received_monotonic_sec],
        )
        if receive_skew_sec is None:
            self._publish_warn_throttled(
                "state/camera local receive timestamp unavailable or invalid"
            )
            return
        if receive_skew_sec > self._max_state_camera_skew_sec:
            self._publish_warn_throttled(
                "state/camera local receive-time skew "
                f"{receive_skew_sec:.3f}s exceeds "
                f"{self._max_state_camera_skew_sec:.3f}s"
            )
            return

        try:
            self._writer.add_frame(
                _robot_frame_from_msg(state.msg),
                camera_bundle.images,
                camera_timestamps_sec={
                    stream: frame.camera_timestamp_sec
                    for stream, frame in camera_bundle.frames.items()
                },
            )
            self._last_recorded_camera_identity = camera_bundle.identity
        except Exception as exc:
            reason = self._writer.active_failed_reason or str(exc)
            self._state_machine.mark_failed(reason)
            self._publish_status(
                DiagnosticStatus.ERROR,
                f"recording failed; DISCARD required: {reason}",
            )

    def _enforce_recording_safety_limits(self) -> bool:
        session = self._state_machine.session
        if session is None:
            self._state_machine.mark_failed("RECORDING has no active session")
            self._publish_status(
                DiagnosticStatus.ERROR,
                "recording failed; DISCARD required: RECORDING has no active session",
            )
            return True
        try:
            free_disk_bytes = _free_disk_bytes(self._writer.root_output_dir)
        except OSError as exc:
            reason = f"cannot inspect output disk: {exc}"
            return self._discard_for_safety(reason)
        reason = recording_safety_reason(
            elapsed_sec=max(0.0, self._now_sec() - session.started_at_sec),
            frame_count=self._writer.active_frame_count,
            max_duration_sec=self._max_episode_duration_sec,
            max_frames=self._max_episode_frames,
            free_disk_bytes=free_disk_bytes,
            min_free_disk_bytes=self._min_free_disk_bytes,
        )
        if reason is None:
            return False
        return self._discard_for_safety(reason)

    def _discard_for_safety(self, reason: str) -> bool:
        session = self._state_machine.session
        if session is None:
            self._state_machine.mark_failed("RECORDING has no active session")
            self._publish_status(
                DiagnosticStatus.ERROR,
                "recording failed; DISCARD required: RECORDING has no active session",
            )
            return True
        result = self._state_machine.handle_command(
            RecordCommand.DISCARD,
            episode_id=session.episode_id,
            now_sec=self._now_sec(),
        )
        if not result.should_discard:
            failure = f"safety discard command failed: {result.message}"
            self._state_machine.mark_failed(failure)
            self._publish_status(
                DiagnosticStatus.ERROR,
                f"recording failed; DISCARD required: {failure}",
            )
            return True
        try:
            self._writer.discard_episode()
            self._state_machine.mark_discarded()
        except Exception as exc:
            failure = f"safety discard failed: {exc}"
            self._state_machine.mark_discard_failed(failure)
            self._publish_status(DiagnosticStatus.ERROR, failure)
            return True
        self._last_episode_id = session.episode_id
        self._last_episode_outcome = "DISCARDED"
        self._last_recorded_camera_identity = None
        self._publish_status(DiagnosticStatus.WARN, f"SAFETY DISCARD: {reason}; IDLE")
        return True

    def _begin_save_episode(self) -> None:
        if self._save_worker.has_active:
            reason = "cannot start save because another save task is active"
            self._state_machine.mark_failed(reason)
            self._publish_status(DiagnosticStatus.ERROR, reason)
            return

        self._saving_episode_index = self._writer.active_episode_index
        self._saving_frame_count = self._writer.active_frame_count
        self._save_started_monotonic_sec = time.monotonic()
        self._save_finished_monotonic_sec = None
        self._save_progress_monotonic_sec = self._save_started_monotonic_sec
        self._save_progress_seq = 0
        self._save_phase = "queued"
        self._state_machine.mark_saving()
        try:
            self._save_worker.start(
                lambda report_progress: self._writer.save_episode(
                    progress_callback=report_progress
                )
            )
        except Exception as exc:
            reason = f"failed to start save worker: {exc}"
            self._state_machine.mark_failed(reason)
            self._save_phase = "failed"
            self._save_finished_monotonic_sec = time.monotonic()
            self._save_progress_monotonic_sec = self._save_finished_monotonic_sec
            self._publish_status(
                DiagnosticStatus.ERROR,
                f"save failed; DISCARD required: {reason}",
            )
            return

        self._publish_status(
            DiagnosticStatus.OK,
            self._saving_status_message(),
        )

    def _poll_save_episode(self) -> None:
        self._drain_save_progress()
        if not self._save_worker.done:
            return

        session = self._state_machine.session
        try:
            result = self._save_worker.take_result()
            self._drain_save_progress()
        except Exception as exc:
            reason = self._writer.active_failed_reason or str(exc)
            self._state_machine.mark_failed(reason)
            self._save_phase = "failed"
            self._save_finished_monotonic_sec = time.monotonic()
            self._save_progress_monotonic_sec = self._save_finished_monotonic_sec
            self._publish_status(
                DiagnosticStatus.ERROR,
                f"save failed; DISCARD required: {reason}",
            )
            return

        self._state_machine.mark_saved()
        self._save_phase = "complete"
        self._save_finished_monotonic_sec = time.monotonic()
        self._save_progress_monotonic_sec = self._save_finished_monotonic_sec
        self._last_episode_id = session.episode_id if session is not None else ""
        self._last_episode_outcome = "SAVED" if result.saved else "DISCARDED"
        level = DiagnosticStatus.OK if result.saved else DiagnosticStatus.WARN
        self._publish_status(
            level,
            (
                f"{result.message}: episode={result.episode_index}, "
                f"frames={result.frame_count}"
            ),
        )

    def _publish_periodic_status(self) -> None:
        self._drain_save_progress()
        message = self._state_machine.mode.value
        level = DiagnosticStatus.OK
        if self._state_machine.mode == CollectorMode.RECORDING:
            message = (
                f"RECORDING episode={self._writer.active_episode_index} "
                f"frames={self._writer.active_frame_count}"
            )
        elif self._state_machine.mode == CollectorMode.SAVING:
            message = self._saving_status_message()
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

    def _drain_save_progress(self) -> None:
        for progress in self._save_worker.drain_progress():
            self._save_phase = progress.phase
            self._save_progress_monotonic_sec = progress.monotonic_sec
            self._save_progress_seq += 1

    def _clear_save_tracking(self) -> None:
        self._save_started_monotonic_sec = None
        self._save_finished_monotonic_sec = None
        self._save_progress_monotonic_sec = None
        self._save_progress_seq = 0
        self._save_phase = ""
        self._saving_episode_index = None
        self._saving_frame_count = 0

    def _saving_status_message(self) -> str:
        episode = (
            ""
            if self._saving_episode_index is None
            else str(self._saving_episode_index)
        )
        return (
            f"SAVING episode={episode} frames={self._saving_frame_count} "
            f"phase={self._save_phase or 'queued'} "
            f"elapsed={self._save_elapsed_sec():.3f}s"
        )

    def _save_elapsed_sec(self) -> float:
        started = self._save_started_monotonic_sec
        if started is None:
            return 0.0
        finished = self._save_finished_monotonic_sec
        end = finished if finished is not None else time.monotonic()
        return max(0.0, end - started)

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
        if self._state_machine.mode == CollectorMode.SAVING:
            active_episode = self._saving_episode_index
            active_frames = self._saving_frame_count
        else:
            active_episode = self._writer.active_episode_index
            active_frames = self._writer.active_frame_count
        save_progress_age = ""
        if self._save_progress_monotonic_sec is not None:
            save_progress_age = (
                f"{max(0.0, time.monotonic() - self._save_progress_monotonic_sec):.3f}"
            )
        status = DiagnosticStatus()
        status.level = _diagnostic_level_value(level)
        status.name = self.get_name()
        status.hardware_id = "robo_collector"
        status.message = message
        status.values = [
            KeyValue(key="mode", value=self._state_machine.mode.value),
            KeyValue(key="dataset_root", value=str(self._writer.root)),
            KeyValue(
                key="episode_id",
                value=(
                    self._state_machine.session.episode_id
                    if self._state_machine.session is not None
                    else ""
                ),
            ),
            KeyValue(
                key="task_prompt",
                value=(
                    self._state_machine.session.task_prompt
                    if self._state_machine.session is not None
                    else ""
                ),
            ),
            KeyValue(key="last_command_id", value=self._last_command_id),
            KeyValue(key="last_command", value=self._last_command),
            KeyValue(
                key="last_command_outcome", value=self._last_command_outcome
            ),
            KeyValue(
                key="last_command_episode_id",
                value=self._last_command_episode_id,
            ),
            KeyValue(key="last_episode_id", value=self._last_episode_id),
            KeyValue(
                key="last_episode_outcome", value=self._last_episode_outcome
            ),
            KeyValue(key="fps", value=str(self._fps)),
            KeyValue(key="active_episode", value=str(active_episode)),
            KeyValue(key="active_frames", value=str(active_frames)),
            KeyValue(key="save_phase", value=self._save_phase),
            KeyValue(key="save_progress_seq", value=str(self._save_progress_seq)),
            KeyValue(key="save_elapsed_sec", value=f"{self._save_elapsed_sec():.3f}"),
            KeyValue(key="save_progress_age_sec", value=save_progress_age),
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

    def _set_command_receipt(
        self,
        *,
        command_id: str,
        command: str,
        episode_id: str,
        outcome: str,
    ) -> None:
        self._last_command_id = command_id
        self._last_command = command
        self._last_command_episode_id = episode_id
        self._last_command_outcome = outcome

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
        return time.monotonic()


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
        state_timestamp_sec=message_stamp_sec(msg),
    )


def _free_disk_bytes(path: Any) -> int:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return int(shutil.disk_usage(candidate).free)


def _record_command_name(command: int) -> str:
    if command == int(RecordCommand.START):
        return "START"
    if command == int(RecordCommand.STOP):
        return "STOP"
    if command == int(RecordCommand.DISCARD):
        return "DISCARD"
    return f"UNKNOWN({command})"


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
