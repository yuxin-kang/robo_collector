"""ROS2 node that drives repeated recording attempts from gesture detections."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gesture_logic import (
    AttemptState,
    CurrentAttempt,
    DetectionResult,
    GestureTriggerStateMachine,
    TriggerAction,
    create_detectors,
    extract_gesture_vector,
    validate_reference_lengths,
)
from .gesture_metadata import (
    PROGRESS_SCHEMA_VERSION,
    MetadataSnapshot,
    ProgressCurrent,
    ProgressEvent,
    ProgressLog,
    default_progress_path,
    load_progress_log,
    scan_plan_metadata,
    utc_now_iso,
    write_progress_log,
)
from .gesture_plan import GesturePlanError, GestureTriggerPlan, load_gesture_trigger_plan

try:
    import rclpy
    from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, qos_profile_sensor_data
    from robo_collector_msgs.msg import RecordCommand
    from robo_state_msgs.msg import RoboStateSample
except ImportError as exc:  # pragma: no cover - exercised in ROS runtime
    rclpy = None
    DiagnosticStatus = None
    KeyValue = None
    Node = object
    QoSProfile = None
    RecordCommand = None
    RoboStateSample = None
    qos_profile_sensor_data = None
    _ROS_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised in ROS runtime
    _ROS_IMPORT_ERROR = None


@dataclass(frozen=True)
class CachedSample:
    sample: Any
    received_monotonic_sec: float


@dataclass(frozen=True)
class CollectorStatusSnapshot:
    level: int
    message: str
    mode: str
    dataset_root: str
    values: dict[str, str]
    received_monotonic_sec: float


def resolve_dataset_root(
    explicit_dataset_root: str, plan_dataset_root: str, status_dataset_root: str
) -> str:
    explicit = explicit_dataset_root.strip()
    if explicit:
        return explicit
    planned = plan_dataset_root.strip()
    if planned:
        return planned
    return status_dataset_root.strip()


def resolve_progress_path(
    explicit_progress_path: str, plan_progress_path: str, dataset_root: str
) -> Path:
    explicit = explicit_progress_path.strip()
    if explicit:
        return Path(explicit)
    planned = plan_progress_path.strip()
    if planned:
        return Path(planned)
    return default_progress_path(dataset_root)


def collector_status_is_fresh(
    snapshot: CollectorStatusSnapshot | None, now_sec: float, timeout_sec: float
) -> bool:
    return (
        snapshot is not None
        and timeout_sec > 0.0
        and now_sec - snapshot.received_monotonic_sec <= timeout_sec
    )


def diagnostic_values_map(msg: Any) -> dict[str, str]:
    values = getattr(msg, "values", [])
    mapped: dict[str, str] = {}
    for item in values:
        key = str(getattr(item, "key", "")).strip()
        if not key:
            continue
        mapped[key] = str(getattr(item, "value", "")).strip()
    return mapped


def parse_collector_status(msg: Any, received_monotonic_sec: float) -> CollectorStatusSnapshot:
    values = diagnostic_values_map(msg)
    return CollectorStatusSnapshot(
        level=int(getattr(msg, "level", 0)),
        message=str(getattr(msg, "message", "")).strip(),
        mode=values.get("mode", "").strip(),
        dataset_root=values.get("dataset_root", "").strip(),
        values=values,
        received_monotonic_sec=float(received_monotonic_sec),
    )


def attempt_state_is_armed(attempt_state: AttemptState) -> bool:
    return attempt_state == AttemptState.ARMED


def active_gesture_l2(
    attempt_state: AttemptState, last_detection: dict[str, DetectionResult]
) -> float:
    if attempt_state == AttemptState.WAITING_READY:
        return last_detection["ready"].l2
    if attempt_state == AttemptState.ARMED:
        return last_detection["start"].l2
    if attempt_state == AttemptState.RECORDING:
        return last_detection["end"].l2
    return math.inf


def detection_tail_frames(max_detection_latency_sec: float, collector_fps: float) -> int:
    if collector_fps <= 0.0:
        raise ValueError("collector_fps must be > 0")
    return int(math.ceil(max_detection_latency_sec * collector_fps))


def update_detector_safely(
    detector: GestureConditionDetector,
    vector: list[float] | None,
    now_sec: float,
) -> tuple[DetectionResult, str]:
    try:
        return detector.update(vector, now_sec), ""
    except ValueError as exc:
        return DetectionResult(False, math.inf), str(exc)


if rclpy is not None:  # pragma: no branch

    class GestureTriggerNode(Node):
        """Publishes collector START/STOP commands from configured gesture detections."""

        def __init__(self) -> None:
            super().__init__("gesture_trigger_node")

            self.declare_parameter("plan_path", "")
            self.declare_parameter("dataset_root", "")
            self.declare_parameter("progress_path", "")
            self.declare_parameter("trigger_status_topic", "/gesture_trigger/status")
            self.declare_parameter("tick_hz", 20.0)
            self.declare_parameter("metadata_poll_hz", 2.0)
            self.declare_parameter("max_sample_age_sec", 0.0)
            self.declare_parameter("collector_fps", 0.0)

            plan_path = str(self.get_parameter("plan_path").value).strip()
            if not plan_path:
                raise RuntimeError("plan_path parameter is required")
            try:
                self._plan = load_gesture_trigger_plan(plan_path)
            except GesturePlanError as exc:
                raise RuntimeError(f"invalid gesture trigger plan: {exc}") from exc
            validate_reference_lengths(self._plan)

            self._explicit_dataset_root = str(
                self.get_parameter("dataset_root").value
            ).strip()
            self._explicit_progress_path = str(
                self.get_parameter("progress_path").value
            ).strip()
            collector_fps = float(self.get_parameter("collector_fps").value)
            self._collector_fps = (
                collector_fps if collector_fps > 0.0 else float(self._plan.collector.fps)
            )
            self._max_sample_age_sec = float(
                self.get_parameter("max_sample_age_sec").value
            )
            if self._max_sample_age_sec <= 0.0:
                self._max_sample_age_sec = float(
                    self._plan.tail_bounds.max_detection_latency_sec
                )
            self._tick_period_sec = 1.0 / _positive_rate(
                float(self.get_parameter("tick_hz").value),
                "tick_hz",
            )
            self._metadata_poll_period_sec = 1.0 / _positive_rate(
                float(self.get_parameter("metadata_poll_hz").value),
                "metadata_poll_hz",
            )

            self._status_topic = str(
                self.get_parameter("trigger_status_topic").value
            ).strip()
            self._detectors = create_detectors(self._plan)
            self._last_detection: dict[str, DetectionResult] = {
                "ready": DetectionResult(False, math.inf),
                "start": DetectionResult(False, math.inf),
                "end": DetectionResult(False, math.inf),
            }
            self._machine = GestureTriggerStateMachine(self._plan)
            self._latest_sample: CachedSample | None = None
            self._latest_collector_status: CollectorStatusSnapshot | None = None
            self._dataset_root = ""
            self._progress_path: Path | None = None
            self._metadata_snapshot: MetadataSnapshot | None = None
            self._last_metadata_poll_sec = 0.0
            self._bootstrapped = False
            self._fatal_error = ""
            self._events: tuple[ProgressEvent, ...] = ()
            self._last_progress_signature: tuple[Any, ...] | None = None
            self._last_status_log_message = ""
            self._last_status_log_monotonic_sec = 0.0

            qos = QoSProfile(depth=10)
            self._status_pub = self.create_publisher(
                DiagnosticStatus,
                self._status_topic,
                qos,
            )
            self._command_pub = self.create_publisher(
                RecordCommand,
                self._plan.collector.command_topic,
                qos,
            )
            self.create_subscription(
                RoboStateSample,
                self._plan.gesture_source.topic,
                self._on_sample,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                DiagnosticStatus,
                self._plan.collector.status_topic,
                self._on_collector_status,
                qos,
            )
            self._tick_timer = self.create_timer(self._tick_period_sec, self._tick)
            self._status_timer = self.create_timer(1.0, self._publish_periodic_status)

            self.get_logger().info(
                "gesture trigger ready; plan=%s, gesture_topic=%s, collector_status=%s, "
                "command_topic=%s"
                % (
                    plan_path,
                    self._plan.gesture_source.topic,
                    self._plan.collector.status_topic,
                    self._plan.collector.command_topic,
                )
            )
            self._publish_status(DiagnosticStatus.WARN, "waiting for dataset root")

        def _on_sample(self, msg: Any) -> None:
            self._latest_sample = CachedSample(
                sample=msg, received_monotonic_sec=time.monotonic()
            )

        def _on_collector_status(self, msg: Any) -> None:
            snapshot = parse_collector_status(msg, time.monotonic())
            if (
                self._bootstrapped
                and not self._explicit_dataset_root
                and snapshot.dataset_root
                and self._dataset_root
                and snapshot.dataset_root != self._dataset_root
            ):
                self._set_fatal_error(
                    "collector dataset_root changed from "
                    f"{self._dataset_root} to {snapshot.dataset_root}"
                )
            self._latest_collector_status = snapshot

        def _tick(self) -> None:
            now_sec = time.monotonic()
            self._refresh_dataset_root(now_sec)
            self._validate_tail_bound(now_sec)
            if self._fatal_error:
                self._reset_detectors()
                self._write_progress_log_if_needed()
                return
            if not self._dataset_root:
                self._reset_detectors()
                return

            metadata_snapshot = self._poll_metadata(now_sec, force=not self._bootstrapped)
            if metadata_snapshot is None:
                return
            if not self._bootstrapped:
                self._bootstrap_from_metadata(metadata_snapshot)
                self._write_progress_log_if_needed(force=True)
                return

            previous_state = self._machine.attempt_state
            previous_attempt = self._machine.current_attempt
            actions = self._machine.step(
                now_sec,
                ready_triggered=self._ready_triggered(now_sec),
                start_triggered=self._start_triggered(now_sec),
                end_triggered=self._end_triggered(now_sec),
                metadata_snapshot=metadata_snapshot,
                collector_mode=self._collector_mode(now_sec),
                collector_episode_id=self._collector_episode_id(now_sec),
            )
            for action in actions:
                self._publish_action(action)

            state_changed = previous_state != self._machine.attempt_state
            self._record_state_transition(previous_state, self._machine.attempt_state)
            if previous_attempt != self._machine.current_attempt:
                self._append_event(
                    "ATTEMPT_SELECTED",
                    self._machine.current_attempt.episode_id
                    if self._machine.current_attempt is not None
                    else "",
                    self._current_attempt_details(self._machine.current_attempt),
                )
                self._reset_detectors()

            if previous_state != self._machine.attempt_state:
                self._reset_detectors()

            self._write_progress_log_if_needed(force=bool(actions))
            if state_changed:
                self._publish_periodic_status()

        def _bootstrap_from_metadata(self, metadata_snapshot: MetadataSnapshot) -> None:
            self._load_existing_progress_events()
            self._machine.bootstrap(metadata_snapshot)
            self._bootstrapped = True
            self._append_event(
                "BOOTSTRAP",
                self._machine.current_attempt.episode_id
                if self._machine.current_attempt is not None
                else "",
                {
                    "completed_count": self._machine.completed_count,
                    "attempt_state": self._machine.attempt_state.value,
                },
            )
            self._record_state_transition(None, self._machine.attempt_state)
            self._reset_detectors()

        def _refresh_dataset_root(self, now_sec: float) -> None:
            status_dataset_root = ""
            if collector_status_is_fresh(
                self._latest_collector_status,
                now_sec,
                self._plan.collector.status_timeout_sec,
            ):
                status_dataset_root = self._latest_collector_status.dataset_root
            dataset_root = resolve_dataset_root(
                self._explicit_dataset_root,
                self._plan.dataset_root,
                status_dataset_root,
            )
            if dataset_root and dataset_root != self._dataset_root:
                self._dataset_root = dataset_root
                self._progress_path = resolve_progress_path(
                    self._explicit_progress_path,
                    self._plan.progress_path,
                    dataset_root,
                )
                self._metadata_snapshot = None
                self._bootstrapped = False
                self._last_metadata_poll_sec = 0.0
                self._events = ()
                self._last_progress_signature = None

        def _poll_metadata(
            self, now_sec: float, *, force: bool = False
        ) -> MetadataSnapshot | None:
            if not self._dataset_root:
                return None
            if (
                not force
                and self._metadata_snapshot is not None
                and now_sec - self._last_metadata_poll_sec < self._metadata_poll_period_sec
            ):
                return self._metadata_snapshot
            self._metadata_snapshot = scan_plan_metadata(self._dataset_root, self._plan)
            self._last_metadata_poll_sec = now_sec
            return self._metadata_snapshot

        def _load_existing_progress_events(self) -> None:
            if self._progress_path is None:
                return
            progress = load_progress_log(self._progress_path)
            if progress is None:
                return
            if progress.plan_id != self._plan.plan_id:
                return
            if progress.dataset_root and progress.dataset_root != self._dataset_root:
                return
            self._events = progress.events

        def _ready_triggered(self, now_sec: float) -> bool:
            if self._machine.attempt_state != AttemptState.WAITING_READY:
                self._last_detection["ready"] = self._detectors["ready"].update(None, now_sec)
                return False
            if not self._collector_ready_for_start(now_sec):
                self._last_detection["ready"] = self._detectors["ready"].update(None, now_sec)
                return False
            vector = self._current_gesture_vector(now_sec)
            self._last_detection["ready"] = self._update_detector_or_fail(
                "ready",
                vector,
                now_sec,
            )
            return self._last_detection["ready"].triggered

        def _start_triggered(self, now_sec: float) -> bool:
            if self._machine.attempt_state != AttemptState.ARMED:
                self._last_detection["start"] = self._detectors["start"].update(None, now_sec)
                return False
            vector = self._current_gesture_vector(now_sec)
            self._last_detection["start"] = self._update_detector_or_fail(
                "start",
                vector,
                now_sec,
            )
            return self._last_detection["start"].triggered

        def _end_triggered(self, now_sec: float) -> bool:
            if self._machine.attempt_state != AttemptState.RECORDING:
                self._last_detection["end"] = self._detectors["end"].update(None, now_sec)
                return False
            vector = self._current_gesture_vector(now_sec)
            self._last_detection["end"] = self._update_detector_or_fail(
                "end",
                vector,
                now_sec,
            )
            return self._last_detection["end"].triggered

        def _collector_ready_for_start(self, now_sec: float) -> bool:
            status = self._latest_collector_status
            return (
                collector_status_is_fresh(
                    status,
                    now_sec,
                    self._plan.collector.status_timeout_sec,
                )
                and status is not None
                and status.mode == "IDLE"
            )

        def _collector_mode(self, now_sec: float) -> str:
            if not collector_status_is_fresh(
                self._latest_collector_status,
                now_sec,
                self._plan.collector.status_timeout_sec,
            ):
                return ""
            return self._latest_collector_status.mode

        def _collector_episode_id(self, now_sec: float) -> str:
            if not collector_status_is_fresh(
                self._latest_collector_status,
                now_sec,
                self._plan.collector.status_timeout_sec,
            ):
                return ""
            return self._latest_collector_status.values.get("episode_id", "")

        def _current_gesture_vector(self, now_sec: float) -> list[float] | None:
            sample = self._latest_sample
            if sample is None:
                return None
            if now_sec - sample.received_monotonic_sec > self._max_sample_age_sec:
                return None
            try:
                return extract_gesture_vector(
                    sample.sample,
                    self._plan.gesture_source.field,
                    self._plan.gesture_source.indices,
                )
            except ValueError as exc:
                self._set_fatal_error(str(exc))
                return None

        def _update_detector_or_fail(
            self,
            detector_name: str,
            vector: list[float] | None,
            now_sec: float,
        ) -> DetectionResult:
            result, error = update_detector_safely(
                self._detectors[detector_name],
                vector,
                now_sec,
            )
            if error:
                self._set_fatal_error(error)
            return result

        def _validate_tail_bound(self, now_sec: float) -> None:
            if self._fatal_error:
                return
            if self._collector_fps <= 0.0:
                self._set_fatal_error(
                    "collector_fps must be > 0 to enforce max_tail_frames"
                )
                return
            try:
                tail_frames = detection_tail_frames(
                    self._plan.tail_bounds.max_detection_latency_sec,
                    self._collector_fps,
                )
            except ValueError as exc:
                self._set_fatal_error(str(exc))
                return
            if tail_frames > self._plan.tail_bounds.max_tail_frames:
                self._set_fatal_error(
                    "tail bound impossible: "
                    f"latency {self._plan.tail_bounds.max_detection_latency_sec:.3f}s "
                    f"at collector fps {self._collector_fps:g} implies {tail_frames} tail frames, "
                    f"which exceeds max_tail_frames={self._plan.tail_bounds.max_tail_frames}"
                )

        def _publish_action(self, action: TriggerAction) -> None:
            msg = RecordCommand()
            if action.command == "START":
                msg.command = RecordCommand.START
            elif action.command == "STOP":
                msg.command = RecordCommand.STOP
            else:
                msg.command = RecordCommand.DISCARD
            msg.task_prompt = action.task_prompt
            msg.episode_id = action.episode_id
            self._command_pub.publish(msg)
            self._append_event(
                f"{action.command}_SENT",
                action.episode_id,
                {
                    "task_prompt": action.task_prompt,
                },
            )

        def _record_state_transition(
            self,
            previous_state: AttemptState | None,
            current_state: AttemptState,
        ) -> None:
            if previous_state == current_state:
                return
            self._append_event(
                "STATE_TRANSITION",
                self._machine.current_attempt.episode_id
                if self._machine.current_attempt is not None
                else "",
                {
                    "from": previous_state.value if previous_state is not None else "",
                    "to": current_state.value,
                    "error": self._machine.last_error,
                    "metadata_match_reason": self._machine.metadata_match_reason,
                },
            )

        def _append_event(
            self, event: str, episode_id: str, details: dict[str, Any] | None = None
        ) -> None:
            self._events = self._events + (
                ProgressEvent(
                    ts=utc_now_iso(),
                    event=event,
                    episode_id=episode_id,
                    details=details or {},
                ),
            )

        def _write_progress_log_if_needed(self, *, force: bool = False) -> None:
            if self._progress_path is None or not self._dataset_root:
                return
            current = self._current_progress_attempt()
            signature = (
                self._dataset_root,
                self._machine.attempt_state.value if self._bootstrapped else "WAITING_DATASET_ROOT",
                current.task_slug if current is not None else "",
                current.trial_index if current is not None else -1,
                current.attempt_index if current is not None else -1,
                current.episode_id if current is not None else "",
                len(self._events),
                self._machine.completed_count,
                self._fatal_error,
            )
            if not force and signature == self._last_progress_signature:
                return
            progress = ProgressLog(
                schema_version=PROGRESS_SCHEMA_VERSION,
                plan_id=self._plan.plan_id,
                dataset_root=self._dataset_root,
                updated_at=utc_now_iso(),
                last_state=(
                    "FAILED"
                    if self._fatal_error
                    else self._machine.attempt_state.value
                    if self._bootstrapped
                    else "WAITING_DATASET_ROOT"
                ),
                current=current,
                events=self._events,
            )
            try:
                write_progress_log(self._progress_path, progress)
            except OSError as exc:
                message = f"failed to persist gesture progress: {exc}"
                self._set_fatal_error(message)
                return
            self._last_progress_signature = signature

        def _set_fatal_error(self, message: str) -> None:
            if self._fatal_error:
                return
            self._fatal_error = message
            self.get_logger().error(message)
            stop_action = self._machine.abort_active_attempt(message)
            if stop_action is not None:
                self._publish_action(stop_action)

        def _current_progress_attempt(self) -> ProgressCurrent | None:
            attempt = self._machine.current_attempt
            if attempt is None:
                return None
            return ProgressCurrent(
                task_slug=attempt.task_slug,
                trial_index=attempt.trial_index,
                attempt_index=attempt.attempt_index,
                episode_id=attempt.episode_id,
            )

        def _current_attempt_details(
            self, attempt: CurrentAttempt | None
        ) -> dict[str, Any]:
            if attempt is None:
                return {}
            return {
                "task_slug": attempt.task_slug,
                "task_prompt": attempt.task_prompt,
                "trial_index": attempt.trial_index,
                "attempt_index": attempt.attempt_index,
            }

        def _reset_detectors(self) -> None:
            for detector in self._detectors.values():
                detector.reset()
            self._last_detection = {
                "ready": DetectionResult(False, math.inf),
                "start": DetectionResult(False, math.inf),
                "end": DetectionResult(False, math.inf),
            }

        def _publish_periodic_status(self) -> None:
            if self._fatal_error:
                self._publish_status(DiagnosticStatus.ERROR, self._fatal_error)
                return
            if not self._dataset_root:
                self._publish_status(DiagnosticStatus.WARN, "waiting for dataset root")
                return
            if not self._bootstrapped:
                self._publish_status(DiagnosticStatus.WARN, "bootstrapping from metadata")
                return

            level = DiagnosticStatus.OK
            message = self._machine.attempt_state.value
            if self._machine.attempt_state in (
                AttemptState.PAUSED_FAILED,
                AttemptState.PAUSED_AMBIGUOUS_METADATA,
            ):
                level = DiagnosticStatus.ERROR
                message = self._machine.last_error or self._machine.attempt_state.value
            elif self._machine.attempt_state == AttemptState.WAITING_SAVE_METADATA:
                level = DiagnosticStatus.WARN
                message = self._machine.metadata_match_reason
            elif self._machine.attempt_state == AttemptState.COMPLETE:
                message = (
                    f"complete: {self._machine.completed_count}/"
                    f"{self._machine.target_trials} trials saved"
                )
            self._publish_status(level, message)

        def _publish_status(self, level: int, message: str) -> None:
            status = DiagnosticStatus()
            status.level = int(level)
            status.name = self.get_name()
            status.hardware_id = "gesture_trigger"
            status.message = message
            current = self._machine.current_attempt
            current_gesture_l2 = active_gesture_l2(
                self._machine.attempt_state if self._bootstrapped else AttemptState.WAITING_READY,
                self._last_detection,
            )
            status.values = [
                KeyValue(key="plan_id", value=self._plan.plan_id),
                KeyValue(
                    key="attempt_state",
                    value=self._machine.attempt_state.value if self._bootstrapped else "",
                ),
                KeyValue(key="dataset_root", value=self._dataset_root),
                KeyValue(
                    key="collector_mode",
                    value=self._latest_collector_status.mode
                    if self._latest_collector_status is not None
                    else "",
                ),
                KeyValue(
                    key="collector_status_fresh",
                    value=str(
                        collector_status_is_fresh(
                            self._latest_collector_status,
                            time.monotonic(),
                            self._plan.collector.status_timeout_sec,
                        )
                    ),
                ),
                KeyValue(key="completed_count", value=str(self._machine.completed_count)),
                KeyValue(key="target_trials", value=str(self._machine.target_trials)),
                KeyValue(key="task_slug", value=current.task_slug if current else ""),
                KeyValue(
                    key="task_prompt",
                    value=current.task_prompt if current is not None else "",
                ),
                KeyValue(
                    key="trial_index",
                    value=str(current.trial_index) if current is not None else "",
                ),
                KeyValue(
                    key="attempt_index",
                    value=str(current.attempt_index) if current is not None else "",
                ),
                KeyValue(
                    key="episode_id",
                    value=current.episode_id if current is not None else "",
                ),
                KeyValue(
                    key="metadata_match_reason",
                    value=self._machine.metadata_match_reason,
                ),
                KeyValue(key="last_error", value=self._machine.last_error),
                KeyValue(key="gesture_l2", value=_format_l2(current_gesture_l2)),
                KeyValue(
                    key="armed",
                    value=str(
                        attempt_state_is_armed(self._machine.attempt_state)
                        if self._bootstrapped
                        else False
                    ),
                ),
                KeyValue(
                    key="gesture_l2_ready",
                    value=_format_l2(self._last_detection["ready"].l2),
                ),
                KeyValue(
                    key="gesture_l2_start",
                    value=_format_l2(self._last_detection["start"].l2),
                ),
                KeyValue(
                    key="gesture_l2_end",
                    value=_format_l2(self._last_detection["end"].l2),
                ),
            ]
            self._status_pub.publish(status)
            self._log_status_issue_throttled(level, message)

        def _log_status_issue_throttled(self, level: int, message: str) -> None:
            if int(level) == int(DiagnosticStatus.OK):
                return
            now_sec = time.monotonic()
            if (
                message == self._last_status_log_message
                and now_sec - self._last_status_log_monotonic_sec <= 5.0
            ):
                return
            self._last_status_log_message = message
            self._last_status_log_monotonic_sec = now_sec
            if int(level) == int(DiagnosticStatus.ERROR):
                self.get_logger().error(message)
            else:
                self.get_logger().warn(message)


def _positive_rate(value: float, parameter_name: str) -> float:
    if value <= 0.0:
        raise RuntimeError(f"{parameter_name} must be > 0")
    return value


def _format_l2(value: float) -> str:
    if math.isinf(value):
        return ""
    return f"{value:.6f}"


def main(args: list[str] | None = None) -> None:
    if rclpy is None:
        raise RuntimeError(
            "gesture_trigger_node requires ROS 2 Python packages"
        ) from _ROS_IMPORT_ERROR
    rclpy.init(args=args)
    node: GestureTriggerNode | None = None
    try:
        node = GestureTriggerNode()
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
