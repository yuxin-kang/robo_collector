"""Pure gesture detection and orchestration state machine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence
from uuid import uuid4

from .gesture_episode_id import build_gesture_episode_id
from .gesture_metadata import MetadataSnapshot
from .gesture_plan import GestureCondition, GestureReference, GestureTriggerPlan, PlannedTrial


class AttemptState(str, Enum):
    WAITING_READY = "WAITING_READY"
    ARMED = "ARMED"
    WAITING_START_ACK = "WAITING_START_ACK"
    RECORDING = "RECORDING"
    WAITING_STOP_ACK = "WAITING_STOP_ACK"
    WAITING_DISCARD_ACK = "WAITING_DISCARD_ACK"
    WAITING_SAVE_METADATA = "WAITING_SAVE_METADATA"
    SAVED = "SAVED"
    PAUSED_FAILED = "PAUSED_FAILED"
    PAUSED_AMBIGUOUS_METADATA = "PAUSED_AMBIGUOUS_METADATA"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class DetectionResult:
    triggered: bool
    l2: float


@dataclass(frozen=True)
class TriggerAction:
    command: str
    task_prompt: str
    episode_id: str
    command_id: str


@dataclass(frozen=True)
class CurrentAttempt:
    task_slug: str
    task_prompt: str
    trial_index: int
    attempt_index: int
    episode_id: str


class GestureConditionDetector:
    def __init__(self, reference: Sequence[float], condition: GestureCondition) -> None:
        self._reference = tuple(float(value) for value in reference)
        self._threshold_l2 = float(condition.threshold_l2)
        self._stable_samples = int(condition.stable_samples)
        self._release_threshold_l2 = condition.release_threshold_l2
        self._cooldown_sec = float(condition.cooldown_sec)
        self._stable_count = 0
        self._requires_release = False
        self._cooldown_until = 0.0

    def reset(self) -> None:
        self._stable_count = 0
        self._requires_release = False
        self._cooldown_until = 0.0

    def update(self, vector: Sequence[float] | None, now_sec: float) -> DetectionResult:
        if vector is None:
            self._stable_count = 0
            return DetectionResult(False, math.inf)
        if len(vector) != len(self._reference):
            raise ValueError(
                f"gesture vector length mismatch: expected {len(self._reference)}, got {len(vector)}"
            )
        l2 = _l2_distance(vector, self._reference)

        if self._requires_release:
            release_threshold = self._release_threshold_l2
            if release_threshold is not None and l2 >= release_threshold:
                self._requires_release = False
            self._stable_count = 0
            return DetectionResult(False, l2)

        if now_sec < self._cooldown_until:
            self._stable_count = 0
            return DetectionResult(False, l2)

        if l2 <= self._threshold_l2:
            self._stable_count += 1
        else:
            self._stable_count = 0

        if self._stable_count < self._stable_samples:
            return DetectionResult(False, l2)

        self._stable_count = 0
        if self._release_threshold_l2 is not None:
            self._requires_release = True
        if self._cooldown_sec > 0.0:
            self._cooldown_until = now_sec + self._cooldown_sec
        return DetectionResult(True, l2)


def extract_gesture_vector(sample: Any, field: str, indices: Sequence[int]) -> list[float]:
    if field == "robot_state.joint_pos":
        values = getattr(getattr(sample, "robot_state"), "joint_pos")
    elif field == "aligned_target_pos":
        values = getattr(sample, "aligned_target_pos")
    else:
        raise ValueError(f"unsupported gesture field: {field}")

    vector = [float(value) for value in values]
    if not indices:
        return vector
    extracted = []
    for index in indices:
        if index < 0 or index >= len(vector):
            raise ValueError(
                f"gesture_source index {index} is out of range for {field} with length {len(vector)}"
            )
        extracted.append(vector[index])
    return extracted


class GestureTriggerStateMachine:
    def __init__(self, plan: GestureTriggerPlan) -> None:
        self._plan = plan
        self.attempt_state = AttemptState.WAITING_READY
        self.current_attempt: CurrentAttempt | None = None
        self.completed_count = 0
        self.target_trials = plan.total_trials
        self.last_error = ""
        self.metadata_match_reason = ""
        self._start_deadline_sec = 0.0
        self._next_start_retry_sec = 0.0
        self._start_attempts = 0
        self._recording_started_sec = 0.0
        self._stop_deadline_sec = 0.0
        self._next_stop_retry_sec = 0.0
        self._stop_attempts = 0
        self._discard_deadline_sec = 0.0
        self._next_discard_retry_sec = 0.0
        self._discard_attempts = 0
        self._pending_failure_message = ""
        self._save_deadline_sec = 0.0
        self._save_wait_deadline_sec = 0.0
        self._last_save_progress_seq: int | None = None
        self._command_ids: dict[str, str] = {}

    def bootstrap(self, snapshot: MetadataSnapshot) -> None:
        if snapshot.parse_error:
            self._pause_failed(snapshot.parse_error)
            return
        self._select_next_attempt(snapshot)

    def step(
        self,
        now_sec: float,
        *,
        ready_triggered: bool = False,
        start_triggered: bool = False,
        end_triggered: bool = False,
        metadata_snapshot: MetadataSnapshot | None = None,
        collector_mode: str = "",
        collector_episode_id: str = "",
        collector_save_progress_token: str = "",
        collector_last_command_id: str = "",
        collector_last_command: str = "",
        collector_last_command_outcome: str = "",
        collector_last_command_episode_id: str = "",
        collector_last_episode_id: str = "",
        collector_last_episode_outcome: str = "",
        recording_sample_fresh: bool = True,
    ) -> list[TriggerAction]:
        if self.attempt_state == AttemptState.COMPLETE:
            return []

        if self.attempt_state == AttemptState.SAVED and metadata_snapshot is not None:
            self._select_next_attempt(metadata_snapshot)
            return []

        if self.attempt_state == AttemptState.WAITING_DISCARD_ACK:
            return self._handle_waiting_discard_ack(
                collector_mode=collector_mode,
                collector_episode_id=collector_episode_id,
                collector_last_command_id=collector_last_command_id,
                collector_last_command=collector_last_command,
                collector_last_command_outcome=collector_last_command_outcome,
                collector_last_command_episode_id=(
                    collector_last_command_episode_id
                ),
                collector_last_episode_id=collector_last_episode_id,
                collector_last_episode_outcome=collector_last_episode_outcome,
                metadata_snapshot=metadata_snapshot,
                now_sec=now_sec,
            )

        if self.attempt_state == AttemptState.WAITING_SAVE_METADATA and metadata_snapshot is not None:
            return self._handle_waiting_save(
                metadata_snapshot,
                collector_mode,
                collector_episode_id,
                now_sec,
                collector_save_progress_token,
            )

        if self.attempt_state == AttemptState.WAITING_START_ACK:
            return self._handle_waiting_start_ack(
                collector_mode=collector_mode,
                collector_episode_id=collector_episode_id,
                collector_last_command_id=collector_last_command_id,
                collector_last_command=collector_last_command,
                collector_last_command_outcome=collector_last_command_outcome,
                collector_last_command_episode_id=(
                    collector_last_command_episode_id
                ),
                now_sec=now_sec,
            )

        if self.attempt_state == AttemptState.WAITING_STOP_ACK:
            if self.current_attempt is not None and collector_mode == "IDLE":
                expected_episode_id = self.current_attempt.episode_id
                if metadata_snapshot is not None and not metadata_snapshot.parse_error:
                    status = metadata_snapshot.status_for(
                        PlannedTrial(
                            self.current_attempt.task_slug,
                            self.current_attempt.task_prompt,
                            self.current_attempt.trial_index,
                        )
                    )
                    if status.ambiguous:
                        self.attempt_state = AttemptState.PAUSED_AMBIGUOUS_METADATA
                        self.last_error = status.message
                        self.metadata_match_reason = status.message
                        return []
                    if status.complete and status.episode_id == expected_episode_id:
                        return self._begin_save_reconciliation(
                            now_sec,
                            collector_mode,
                            collector_episode_id,
                            metadata_snapshot,
                            "recovered saved metadata after STOP receipt loss",
                            collector_save_progress_token,
                        )
                if (
                    collector_last_episode_id == expected_episode_id
                    and collector_last_episode_outcome == "SAVED"
                ):
                    return self._begin_save_reconciliation(
                        now_sec,
                        collector_mode,
                        collector_episode_id,
                        metadata_snapshot,
                        "collector reports the episode saved after STOP receipt loss",
                        collector_save_progress_token,
                    )
                if (
                    collector_last_episode_id == expected_episode_id
                    and collector_last_episode_outcome == "DISCARDED"
                ):
                    self._pause_failed(
                        "collector discarded the episode while STOP acknowledgement was pending"
                    )
                    return []
            actions = self._handle_waiting_stop_ack(
                collector_mode=collector_mode,
                collector_episode_id=collector_episode_id,
                collector_last_command_id=collector_last_command_id,
                collector_last_command=collector_last_command,
                collector_last_command_outcome=collector_last_command_outcome,
                collector_last_command_episode_id=(
                    collector_last_command_episode_id
                ),
                collector_save_progress_token=collector_save_progress_token,
                now_sec=now_sec,
            )
            if (
                self.attempt_state == AttemptState.WAITING_SAVE_METADATA
                and metadata_snapshot is not None
            ):
                actions.extend(
                    self._handle_waiting_save(
                        metadata_snapshot,
                        collector_mode,
                        collector_episode_id,
                        now_sec,
                        collector_save_progress_token,
                    )
                )
            return actions

        if self.attempt_state == AttemptState.WAITING_READY and ready_triggered:
            self.attempt_state = AttemptState.ARMED
            self.last_error = ""
            return []

        if self.attempt_state == AttemptState.ARMED and start_triggered:
            if self.current_attempt is None:
                self._pause_failed("no current attempt is available for START")
                return []
            self.attempt_state = AttemptState.WAITING_START_ACK
            self._start_deadline_sec = (
                now_sec + self._plan.collector.start_confirm_timeout_sec
            )
            self._next_start_retry_sec = (
                now_sec + self._plan.collector.command_retry_interval_sec
            )
            self._start_attempts = 1
            self._command_ids["START"] = uuid4().hex
            self.last_error = ""
            return [self._current_action("START")]

        if self.attempt_state == AttemptState.RECORDING:
            if self.current_attempt is None:
                self._pause_failed("recording without a current attempt")
                return []
            expected_episode_id = self.current_attempt.episode_id
            if collector_mode == "RECORDING" and collector_episode_id not in (
                "",
                expected_episode_id,
            ):
                self._pause_failed(
                    "collector switched to a different episode while recording: "
                    f"expected {expected_episode_id!r}, got {collector_episode_id!r}"
                )
                return []
            if collector_mode == "FAILED":
                return self._begin_discard(
                    "collector entered FAILED while recording", now_sec
                )
            if collector_mode == "IDLE":
                if (
                    collector_last_episode_id == expected_episode_id
                    and collector_last_episode_outcome == "SAVED"
                ):
                    return self._begin_save_reconciliation(
                        now_sec,
                        collector_mode,
                        collector_episode_id,
                        metadata_snapshot,
                        "collector saved the episode outside the gesture STOP path",
                        collector_save_progress_token,
                    )
                if (
                    collector_last_episode_id == expected_episode_id
                    and collector_last_episode_outcome == "DISCARDED"
                ):
                    self._pause_failed(
                        "collector discarded the active episode while gesture recording"
                    )
                    return []
                self._pause_failed(
                    "collector returned to IDLE before the gesture STOP command"
                )
                return []
            if collector_mode in ("NEED_TO_SAVE", "SAVING"):
                if collector_episode_id not in ("", expected_episode_id):
                    self._pause_failed(
                        "collector is saving a different episode while gesture recording"
                    )
                    return []
                return self._begin_save_reconciliation(
                    now_sec,
                    collector_mode,
                    collector_episode_id,
                    metadata_snapshot,
                    "collector is saving after an external STOP command",
                    collector_save_progress_token,
                )
            if (
                now_sec - self._recording_started_sec
                >= self._plan.collector.max_recording_duration_sec
            ):
                return self._begin_discard(
                    "maximum recording duration exceeded", now_sec
                )
            if not recording_sample_fresh:
                return self._begin_discard(
                    "gesture sample became missing or stale while recording",
                    now_sec,
                )
            if end_triggered:
                return self._begin_stop(now_sec)

        return []

    def _begin_save_reconciliation(
        self,
        now_sec: float,
        collector_mode: str,
        collector_episode_id: str,
        metadata_snapshot: MetadataSnapshot | None,
        reason: str,
        collector_save_progress_token: str,
    ) -> list[TriggerAction]:
        self._begin_save_wait(now_sec, collector_save_progress_token)
        self.metadata_match_reason = reason
        if metadata_snapshot is None:
            return []
        return self._handle_waiting_save(
            metadata_snapshot,
            collector_mode,
            collector_episode_id,
            now_sec,
            collector_save_progress_token,
        )

    def abort_active_attempt(
        self, message: str, now_sec: float = 0.0
    ) -> TriggerAction | None:
        if self.attempt_state == AttemptState.WAITING_DISCARD_ACK:
            return None
        should_discard = self.attempt_state in (
            AttemptState.WAITING_START_ACK,
            AttemptState.RECORDING,
            AttemptState.WAITING_STOP_ACK,
            AttemptState.WAITING_SAVE_METADATA,
        )
        if not should_discard:
            self._pause_failed(message)
            return None
        actions = self._begin_discard(message, now_sec)
        return actions[0] if actions else None

    def _handle_waiting_start_ack(
        self,
        *,
        collector_mode: str,
        collector_episode_id: str,
        collector_last_command_id: str,
        collector_last_command: str,
        collector_last_command_outcome: str,
        collector_last_command_episode_id: str,
        now_sec: float,
    ) -> list[TriggerAction]:
        if self.current_attempt is None:
            self._pause_failed("waiting for START acknowledgement without a current attempt")
            return []

        expected_episode_id = self.current_attempt.episode_id
        expected_command_id = self._current_action("START").command_id
        matching_receipt = (
            collector_last_command_id == expected_command_id
            and collector_last_command == "START"
            and collector_last_command_episode_id == expected_episode_id
        )
        if matching_receipt and collector_last_command_outcome in {
            "FAILED",
            "REJECTED",
        }:
            self._pause_failed("collector rejected or failed the START command")
            return []
        if collector_mode == "RECORDING":
            if collector_episode_id != expected_episode_id:
                self._pause_failed(
                    "collector acknowledged a different episode while waiting for START: "
                    f"expected {expected_episode_id!r}, got {collector_episode_id!r}"
                )
                return []
            if (
                matching_receipt
                and collector_last_command_outcome == "SUCCEEDED"
            ):
                self.attempt_state = AttemptState.RECORDING
                self._recording_started_sec = now_sec
                self.last_error = ""
                return []

        if collector_mode == "FAILED":
            if (
                matching_receipt
                and collector_last_command_outcome == "SUCCEEDED"
                and collector_episode_id in ("", expected_episode_id)
            ):
                return self._begin_discard(
                    "collector failed after accepting the owned START command",
                    now_sec,
                )
            self._pause_failed(
                "collector entered FAILED before START ownership was confirmed"
            )
            return []

        if now_sec >= self._start_deadline_sec:
            return self._begin_discard("START acknowledgement timed out", now_sec)

        if (
            now_sec >= self._next_start_retry_sec
            and self._start_attempts < self._plan.collector.command_max_retries
        ):
            self._start_attempts += 1
            self._next_start_retry_sec = (
                now_sec + self._plan.collector.command_retry_interval_sec
            )
            return [self._current_action("START")]
        return []

    def _begin_stop(self, now_sec: float) -> list[TriggerAction]:
        if self.current_attempt is None:
            self._pause_failed("no current attempt is available for STOP")
            return []
        self.attempt_state = AttemptState.WAITING_STOP_ACK
        self._stop_deadline_sec = (
            now_sec + self._plan.collector.stop_confirm_timeout_sec
        )
        self._next_stop_retry_sec = (
            now_sec + self._plan.collector.command_retry_interval_sec
        )
        self._stop_attempts = 1
        self._command_ids["STOP"] = uuid4().hex
        self.last_error = ""
        self.metadata_match_reason = "waiting for STOP acknowledgement"
        return [self._current_action("STOP")]

    def _handle_waiting_stop_ack(
        self,
        *,
        collector_mode: str,
        collector_episode_id: str,
        collector_last_command_id: str,
        collector_last_command: str,
        collector_last_command_outcome: str,
        collector_last_command_episode_id: str,
        collector_save_progress_token: str,
        now_sec: float,
    ) -> list[TriggerAction]:
        if self.current_attempt is None:
            self._pause_failed("waiting for STOP acknowledgement without a current attempt")
            return []

        expected_episode_id = self.current_attempt.episode_id
        expected_command_id = self._current_action("STOP").command_id
        matching_receipt = (
            collector_last_command_id == expected_command_id
            and collector_last_command == "STOP"
            and collector_last_command_episode_id == expected_episode_id
        )
        if collector_mode == "RECORDING" and collector_episode_id not in (
            "",
            expected_episode_id,
        ):
            self._pause_failed(
                "collector switched to a different episode while waiting for STOP: "
                f"expected {expected_episode_id!r}, got {collector_episode_id!r}"
            )
            return []
        if collector_mode in ("NEED_TO_SAVE", "SAVING", "IDLE"):
            if (
                collector_mode in ("NEED_TO_SAVE", "SAVING")
                and collector_episode_id
                and collector_episode_id != expected_episode_id
            ):
                self._pause_failed(
                    "collector acknowledged STOP for a different episode: "
                    f"expected {expected_episode_id!r}, got {collector_episode_id!r}"
                )
                return []
            if (
                matching_receipt
                and collector_last_command_outcome == "SUCCEEDED"
            ):
                self._begin_save_wait(now_sec, collector_save_progress_token)
                self.metadata_match_reason = "waiting for metadata save confirmation"
                return []
            if collector_mode in ("NEED_TO_SAVE", "SAVING"):
                self._begin_save_wait(now_sec, collector_save_progress_token)
                self.metadata_match_reason = (
                    "collector state confirms save after STOP receipt loss"
                )
                return []
        if collector_mode == "FAILED":
            return self._begin_discard(
                "collector entered FAILED while waiting for STOP acknowledgement",
                now_sec,
            )
        if matching_receipt and collector_last_command_outcome in {
            "FAILED",
            "REJECTED",
        }:
            return self._begin_discard(
                "collector rejected or failed the owned STOP command",
                now_sec,
            )
        if now_sec >= self._stop_deadline_sec:
            return self._begin_discard("STOP acknowledgement timed out", now_sec)
        if (
            now_sec >= self._next_stop_retry_sec
            and self._stop_attempts < self._plan.collector.command_max_retries
        ):
            self._stop_attempts += 1
            self._next_stop_retry_sec = (
                now_sec + self._plan.collector.command_retry_interval_sec
            )
            return [self._current_action("STOP")]
        return []

    def _begin_discard(
        self, message: str, now_sec: float
    ) -> list[TriggerAction]:
        if self.current_attempt is None:
            self._pause_failed(message)
            return []
        self.attempt_state = AttemptState.WAITING_DISCARD_ACK
        self._discard_deadline_sec = (
            now_sec + self._plan.collector.discard_confirm_timeout_sec
        )
        self._next_discard_retry_sec = (
            now_sec + self._plan.collector.command_retry_interval_sec
        )
        self._discard_attempts = 1
        self._command_ids["DISCARD"] = uuid4().hex
        self._pending_failure_message = message
        self.last_error = message
        self.metadata_match_reason = "waiting for DISCARD acknowledgement"
        return [self._current_action("DISCARD")]

    def _handle_waiting_discard_ack(
        self,
        *,
        collector_mode: str,
        collector_episode_id: str,
        collector_last_command_id: str,
        collector_last_command: str,
        collector_last_command_outcome: str,
        collector_last_command_episode_id: str,
        collector_last_episode_id: str,
        collector_last_episode_outcome: str,
        metadata_snapshot: MetadataSnapshot | None,
        now_sec: float,
    ) -> list[TriggerAction]:
        if self.current_attempt is None:
            self._pause_failed("waiting for DISCARD acknowledgement without an attempt")
            return []

        expected_episode_id = self.current_attempt.episode_id
        expected_command_id = self._current_action("DISCARD").command_id
        if (
            collector_mode
            in ("RECORDING", "NEED_TO_SAVE", "SAVING", "FAILED", "DISCARD")
            and collector_episode_id
            and collector_episode_id != expected_episode_id
        ):
            self._pause_failed(
                "collector switched to a different episode while waiting for DISCARD: "
                f"expected {expected_episode_id!r}, got {collector_episode_id!r}"
            )
            return []
        if (
            collector_last_episode_id == expected_episode_id
            and collector_last_episode_outcome == "SAVED"
        ):
            self.attempt_state = AttemptState.PAUSED_AMBIGUOUS_METADATA
            self.last_error = (
                "collector saved the episode while DISCARD acknowledgement was pending; "
                "manual metadata reconciliation is required"
            )
            self.metadata_match_reason = self.last_error
            return []
        discard_acknowledged = (
            collector_mode == "IDLE"
            and collector_last_command_id == expected_command_id
            and collector_last_command == "DISCARD"
            and collector_last_command_outcome == "SUCCEEDED"
            and collector_last_command_episode_id == expected_episode_id
        ) or (
            collector_mode == "IDLE"
            and collector_last_episode_id == expected_episode_id
            and collector_last_episode_outcome == "DISCARDED"
        )
        if discard_acknowledged:
            if metadata_snapshot is not None:
                status = metadata_snapshot.status_for(
                    PlannedTrial(
                        self.current_attempt.task_slug,
                        self.current_attempt.task_prompt,
                        self.current_attempt.trial_index,
                    )
                )
                if status.complete and status.episode_id == expected_episode_id:
                    self.attempt_state = AttemptState.PAUSED_AMBIGUOUS_METADATA
                    self.last_error = (
                        "DISCARD was acknowledged but matching saved metadata exists; "
                        "manual reconciliation is required"
                    )
                    self.metadata_match_reason = self.last_error
                    return []
            self._pause_failed(self._pending_failure_message)
            return []
        if now_sec >= self._discard_deadline_sec:
            self._pause_failed(
                "DISCARD acknowledgement timed out: "
                + (self._pending_failure_message or "active attempt may still exist")
            )
            return []
        if (
            now_sec >= self._next_discard_retry_sec
            and self._discard_attempts < self._plan.collector.command_max_retries
        ):
            self._discard_attempts += 1
            self._next_discard_retry_sec = (
                now_sec + self._plan.collector.command_retry_interval_sec
            )
            return [self._current_action("DISCARD")]
        return []

    def _current_action(self, command: str) -> TriggerAction:
        if self.current_attempt is None:
            raise RuntimeError(f"cannot create {command} without a current attempt")
        return TriggerAction(
            command=command,
            task_prompt=self.current_attempt.task_prompt,
            episode_id=self.current_attempt.episode_id,
            command_id=self._command_ids[command],
        )

    def _begin_save_wait(
        self,
        now_sec: float,
        collector_save_progress_token: str,
    ) -> None:
        self.attempt_state = AttemptState.WAITING_SAVE_METADATA
        self._save_wait_deadline_sec = (
            now_sec + self._plan.collector.max_save_wait_sec
        )
        self._save_deadline_sec = min(
            self._save_wait_deadline_sec,
            now_sec + self._plan.collector.save_confirm_timeout_sec,
        )
        self._last_save_progress_seq = self._parse_save_progress_seq(
            collector_save_progress_token
        )

    def _observe_save_progress(
        self,
        now_sec: float,
        collector_save_progress_token: str,
    ) -> None:
        if now_sec >= self._save_wait_deadline_sec:
            return
        progress_seq = self._parse_save_progress_seq(
            collector_save_progress_token
        )
        if progress_seq is None:
            return
        if (
            self._last_save_progress_seq is not None
            and progress_seq <= self._last_save_progress_seq
        ):
            return
        self._last_save_progress_seq = progress_seq
        self._save_deadline_sec = min(
            self._save_wait_deadline_sec,
            now_sec + self._plan.collector.save_confirm_timeout_sec,
        )

    def _save_wait_timed_out(self, now_sec: float) -> bool:
        return now_sec >= min(
            self._save_deadline_sec,
            self._save_wait_deadline_sec,
        )

    @staticmethod
    def _parse_save_progress_seq(value: str) -> int | None:
        try:
            progress_seq = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return progress_seq if progress_seq >= 0 else None

    def _handle_waiting_save(
        self,
        snapshot: MetadataSnapshot,
        collector_mode: str,
        collector_episode_id: str,
        now_sec: float,
        collector_save_progress_token: str,
    ) -> list[TriggerAction]:
        if self.current_attempt is None:
            self._pause_failed("waiting for save confirmation without a current attempt")
            return []
        if collector_mode == "SAVING":
            expected_episode_id = self.current_attempt.episode_id
            if collector_episode_id not in ("", expected_episode_id):
                self._pause_failed(
                    "collector is saving a different episode: "
                    f"expected {expected_episode_id!r}, got {collector_episode_id!r}"
                )
                return []
            self._observe_save_progress(now_sec, collector_save_progress_token)
        if now_sec >= self._save_wait_deadline_sec:
            return self._discard_owned_save_timeout(
                "maximum save reconciliation wait exceeded",
                collector_mode,
                collector_episode_id,
                now_sec,
            )
        if snapshot.parse_error:
            self.metadata_match_reason = snapshot.parse_error
            if self._save_wait_timed_out(now_sec):
                return self._discard_owned_save_timeout(
                    snapshot.parse_error,
                    collector_mode,
                    collector_episode_id,
                    now_sec,
                )
            return []
        status = snapshot.status_for(
            PlannedTrial(
                self.current_attempt.task_slug,
                self.current_attempt.task_prompt,
                self.current_attempt.trial_index,
            )
        )
        self.completed_count = snapshot.completed_count
        self.metadata_match_reason = status.message

        if status.ambiguous:
            self.attempt_state = AttemptState.PAUSED_AMBIGUOUS_METADATA
            self.last_error = status.message
            return []

        if status.complete:
            if status.episode_id != self.current_attempt.episode_id:
                self.metadata_match_reason = (
                    "successful metadata belongs to a different attempt: "
                    f"expected {self.current_attempt.episode_id!r}, "
                    f"got {status.episode_id!r}"
                )
                if status.latest_attempt_index > self.current_attempt.attempt_index:
                    self.attempt_state = AttemptState.PAUSED_AMBIGUOUS_METADATA
                    self.last_error = self.metadata_match_reason
                elif self._save_wait_timed_out(now_sec):
                    self._pause_failed(self.metadata_match_reason)
                return []
            if collector_mode == "FAILED":
                self._pause_failed("collector entered FAILED while waiting for metadata save")
                return []
            if collector_mode == "IDLE":
                self.attempt_state = AttemptState.SAVED
                self.last_error = ""
                return []
            if self._save_wait_timed_out(now_sec):
                self._pause_failed(
                    "metadata saved but collector did not return to IDLE before timeout"
                )
            return []

        if status.latest_attempt_index == self.current_attempt.attempt_index and status.latest_state in (
            "EMPTY",
            "TASK_MISMATCH",
        ):
            self._pause_failed(status.message)
            return []

        if status.latest_attempt_index > self.current_attempt.attempt_index:
            self._pause_failed(
                "metadata advanced beyond the current attempt; manual reconciliation required"
            )
            return []

        if collector_mode == "FAILED":
            if collector_episode_id in ("", self.current_attempt.episode_id):
                return self._begin_discard(
                    "collector entered FAILED while waiting for metadata save",
                    now_sec,
                )
            self._pause_failed(
                "collector entered FAILED for a different episode while waiting for metadata save"
            )
            return []

        if self._save_wait_timed_out(now_sec):
            return self._discard_owned_save_timeout(
                "save confirmation timed out without matching metadata",
                collector_mode,
                collector_episode_id,
                now_sec,
            )
        return []

    def _discard_owned_save_timeout(
        self,
        message: str,
        collector_mode: str,
        collector_episode_id: str,
        now_sec: float,
    ) -> list[TriggerAction]:
        if self.current_attempt is None:
            self._pause_failed(message)
            return []
        # SAVING is deliberately excluded: a durable commit cannot be cancelled
        # safely. Only a newer save_progress_seq renews the rolling deadline;
        # timeout pauses instead of issuing a concurrent DISCARD.
        if collector_mode in {"RECORDING", "NEED_TO_SAVE", "FAILED", "DISCARD"}:
            if collector_episode_id in ("", self.current_attempt.episode_id):
                return self._begin_discard(message, now_sec)
            self._pause_failed(
                message
                + "; collector reports a different active episode, so it was not discarded"
            )
            return []
        self._pause_failed(message)
        return []

    def _select_next_attempt(self, snapshot: MetadataSnapshot) -> None:
        self.completed_count = snapshot.completed_count
        if snapshot.parse_error:
            self._pause_failed(snapshot.parse_error)
            return

        for trial in self._plan.planned_trials:
            status = snapshot.status_for(trial)
            if status.ambiguous:
                self.current_attempt = CurrentAttempt(
                    task_slug=trial.task_slug,
                    task_prompt=trial.task_prompt,
                    trial_index=trial.trial_index,
                    attempt_index=max(1, status.latest_attempt_index),
                    episode_id=status.episode_id,
                )
                self.attempt_state = AttemptState.PAUSED_AMBIGUOUS_METADATA
                self.last_error = status.message
                self.metadata_match_reason = status.message
                return
            if status.complete:
                continue
            attempt_index = status.next_attempt_index
            self.current_attempt = CurrentAttempt(
                task_slug=trial.task_slug,
                task_prompt=trial.task_prompt,
                trial_index=trial.trial_index,
                attempt_index=attempt_index,
                episode_id=build_gesture_episode_id(
                    plan_id=self._plan.plan_id,
                    task_slug=trial.task_slug,
                    trial_index=trial.trial_index,
                    attempt_index=attempt_index,
                ),
            )
            self.attempt_state = AttemptState.WAITING_READY
            self._command_ids.clear()
            self.last_error = ""
            self.metadata_match_reason = status.message
            return

        self.current_attempt = None
        self.attempt_state = AttemptState.COMPLETE
        self.last_error = ""
        self.metadata_match_reason = "all planned trials have successful metadata rows"

    def _pause_failed(self, message: str) -> None:
        self.attempt_state = AttemptState.PAUSED_FAILED
        self.last_error = message
        self._pending_failure_message = ""


def validate_reference_lengths(plan: GestureTriggerPlan) -> None:
    if plan.gesture_source.indices:
        expected_length = len(plan.gesture_source.indices)
    else:
        expected_length = len(
            _reference_vector(plan.references, plan.task_start_condition.reference_name)
        )
    for reference in plan.references.values():
        if len(reference.vector) != expected_length:
            raise ValueError(
                f"reference vector length mismatch for {reference.name}: "
                f"expected {expected_length}, got {len(reference.vector)}"
            )


def create_detectors(plan: GestureTriggerPlan) -> dict[str, GestureConditionDetector]:
    validate_reference_lengths(plan)
    ready_reference = _reference_vector(plan.references, plan.return_to_ready_condition.reference_name)
    start_reference = _reference_vector(plan.references, plan.task_start_condition.reference_name)
    end_reference = _reference_vector(plan.references, plan.task_end_condition.reference_name)
    expected_length = len(start_reference)
    for name, reference in plan.references.items():
        if len(reference.vector) != expected_length:
            raise ValueError(
                f"reference vector length mismatch for {name}: expected {expected_length}, got {len(reference.vector)}"
            )
    return {
        "ready": GestureConditionDetector(ready_reference, plan.return_to_ready_condition),
        "start": GestureConditionDetector(start_reference, plan.task_start_condition),
        "end": GestureConditionDetector(end_reference, plan.task_end_condition),
    }


def _reference_vector(
    references: dict[str, GestureReference], reference_name: str
) -> tuple[float, ...]:
    return references[reference_name].vector


def _l2_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))
