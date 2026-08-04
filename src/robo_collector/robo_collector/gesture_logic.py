"""Pure gesture detection and orchestration state machine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from .gesture_episode_id import build_gesture_episode_id
from .gesture_metadata import MetadataSnapshot
from .gesture_plan import GestureCondition, GestureReference, GestureTriggerPlan, PlannedTrial


class AttemptState(str, Enum):
    WAITING_READY = "WAITING_READY"
    ARMED = "ARMED"
    WAITING_START_ACK = "WAITING_START_ACK"
    RECORDING = "RECORDING"
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
        self._save_deadline_sec = 0.0

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
    ) -> list[TriggerAction]:
        if self.attempt_state == AttemptState.COMPLETE:
            return []

        if self.attempt_state == AttemptState.SAVED and metadata_snapshot is not None:
            self._select_next_attempt(metadata_snapshot)
            return []

        if self.attempt_state == AttemptState.WAITING_SAVE_METADATA and metadata_snapshot is not None:
            self._handle_waiting_save(metadata_snapshot, collector_mode, now_sec)
            return []

        if self.attempt_state == AttemptState.WAITING_START_ACK:
            return self._handle_waiting_start_ack(
                collector_mode=collector_mode,
                collector_episode_id=collector_episode_id,
                now_sec=now_sec,
            )

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
            self.last_error = ""
            return [self._current_action("START")]

        if self.attempt_state == AttemptState.RECORDING and end_triggered:
            if self.current_attempt is None:
                self._pause_failed("no current attempt is available for STOP")
                return []
            self.attempt_state = AttemptState.WAITING_SAVE_METADATA
            self._save_deadline_sec = (
                now_sec + self._plan.collector.save_confirm_timeout_sec
            )
            self.last_error = ""
            self.metadata_match_reason = "waiting for metadata save confirmation"
            return [
                TriggerAction(
                    command="STOP",
                    task_prompt=self.current_attempt.task_prompt,
                    episode_id=self.current_attempt.episode_id,
                )
            ]

        return []

    def abort_active_attempt(self, message: str) -> TriggerAction | None:
        should_stop = self.attempt_state in (
            AttemptState.WAITING_START_ACK,
            AttemptState.RECORDING,
        )
        action = self._current_action("STOP") if should_stop else None
        self._pause_failed(message)
        return action

    def _handle_waiting_start_ack(
        self,
        *,
        collector_mode: str,
        collector_episode_id: str,
        now_sec: float,
    ) -> list[TriggerAction]:
        if self.current_attempt is None:
            self._pause_failed("waiting for START acknowledgement without a current attempt")
            return []

        expected_episode_id = self.current_attempt.episode_id
        if collector_mode == "RECORDING":
            if collector_episode_id != expected_episode_id:
                self._pause_failed(
                    "collector acknowledged a different episode while waiting for START: "
                    f"expected {expected_episode_id!r}, got {collector_episode_id!r}"
                )
                return []
            self.attempt_state = AttemptState.RECORDING
            self.last_error = ""
            return []

        if collector_mode == "FAILED":
            self._pause_failed("collector entered FAILED while waiting for START acknowledgement")
            return []

        if now_sec >= self._start_deadline_sec:
            stop_action = self._current_action("STOP")
            self._pause_failed("START acknowledgement timed out")
            return [stop_action]

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

    def _current_action(self, command: str) -> TriggerAction:
        if self.current_attempt is None:
            raise RuntimeError(f"cannot create {command} without a current attempt")
        return TriggerAction(
            command=command,
            task_prompt=self.current_attempt.task_prompt,
            episode_id=self.current_attempt.episode_id,
        )

    def _handle_waiting_save(
        self, snapshot: MetadataSnapshot, collector_mode: str, now_sec: float
    ) -> None:
        if snapshot.parse_error:
            self.metadata_match_reason = snapshot.parse_error
            if now_sec >= self._save_deadline_sec:
                self._pause_failed(snapshot.parse_error)
            return
        if self.current_attempt is None:
            self._pause_failed("waiting for save confirmation without a current attempt")
            return
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
            return

        if status.complete:
            if collector_mode == "FAILED":
                self._pause_failed("collector entered FAILED while waiting for metadata save")
                return
            if collector_mode == "IDLE":
                self.attempt_state = AttemptState.SAVED
                self.last_error = ""
                return
            if now_sec >= self._save_deadline_sec:
                self._pause_failed(
                    "metadata saved but collector did not return to IDLE before timeout"
                )
            return

        if status.latest_attempt_index == self.current_attempt.attempt_index and status.latest_state in (
            "EMPTY",
            "TASK_MISMATCH",
        ):
            self._pause_failed(status.message)
            return

        if status.latest_attempt_index > self.current_attempt.attempt_index:
            self._pause_failed(
                "metadata advanced beyond the current attempt; manual reconciliation required"
            )
            return

        if collector_mode == "FAILED":
            self._pause_failed("collector entered FAILED while waiting for metadata save")
            return

        if now_sec >= self._save_deadline_sec:
            self._pause_failed("save confirmation timed out without matching metadata")

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
