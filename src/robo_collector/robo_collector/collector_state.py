"""Pure recording command state machine for the LeRobot collector."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class CollectorMode(str, Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    NEED_TO_SAVE = "NEED_TO_SAVE"
    DISCARD = "DISCARD"


class RecordCommandType(IntEnum):
    START = 1
    STOP = 2
    DISCARD = 3


@dataclass(frozen=True)
class RecordingSession:
    task_prompt: str
    episode_id: str
    started_at_sec: float


@dataclass(frozen=True)
class CommandResult:
    accepted: bool
    level: str
    message: str
    mode: CollectorMode
    session: RecordingSession | None = None
    should_start: bool = False
    should_save: bool = False
    should_discard: bool = False


class RecordStateMachine:
    """Owns legal transitions for START/STOP/DISCARD commands."""

    def __init__(self) -> None:
        self.mode = CollectorMode.IDLE
        self.session: RecordingSession | None = None

    def handle_command(
        self,
        command: int,
        *,
        task_prompt: str = "",
        episode_id: str = "",
        now_sec: float = 0.0,
    ) -> CommandResult:
        try:
            command_type = RecordCommandType(command)
        except ValueError:
            return CommandResult(
                accepted=False,
                level="ERROR",
                message=f"unknown record command: {command}",
                mode=self.mode,
                session=self.session,
            )

        if command_type == RecordCommandType.START:
            return self._start(task_prompt, episode_id, now_sec)
        if command_type == RecordCommandType.STOP:
            return self._stop()
        return self._discard()

    def mark_saved(self) -> None:
        if self.mode != CollectorMode.NEED_TO_SAVE:
            raise RuntimeError(f"cannot mark saved while mode is {self.mode.value}")
        self.mode = CollectorMode.IDLE
        self.session = None

    def mark_discarded(self) -> None:
        if self.mode != CollectorMode.DISCARD:
            raise RuntimeError(f"cannot mark discarded while mode is {self.mode.value}")
        self.mode = CollectorMode.IDLE
        self.session = None

    def mark_save_failed(self) -> None:
        if self.mode != CollectorMode.NEED_TO_SAVE:
            raise RuntimeError(f"cannot mark save failed while mode is {self.mode.value}")

    def _start(
        self, task_prompt: str, episode_id: str, now_sec: float
    ) -> CommandResult:
        normalized_prompt = task_prompt.strip()
        if not normalized_prompt:
            return CommandResult(
                accepted=False,
                level="WARN",
                message="START rejected: task_prompt is required",
                mode=self.mode,
                session=self.session,
            )
        if self.mode != CollectorMode.IDLE:
            return CommandResult(
                accepted=False,
                level="WARN",
                message=f"START ignored while {self.mode.value}",
                mode=self.mode,
                session=self.session,
            )

        self.session = RecordingSession(
            task_prompt=normalized_prompt,
            episode_id=episode_id.strip(),
            started_at_sec=float(now_sec),
        )
        self.mode = CollectorMode.RECORDING
        return CommandResult(
            accepted=True,
            level="OK",
            message="recording started",
            mode=self.mode,
            session=self.session,
            should_start=True,
        )

    def _stop(self) -> CommandResult:
        if self.mode != CollectorMode.RECORDING:
            return CommandResult(
                accepted=False,
                level="WARN",
                message=f"STOP ignored while {self.mode.value}",
                mode=self.mode,
                session=self.session,
            )

        self.mode = CollectorMode.NEED_TO_SAVE
        return CommandResult(
            accepted=True,
            level="OK",
            message="recording stopped; save requested",
            mode=self.mode,
            session=self.session,
            should_save=True,
        )

    def _discard(self) -> CommandResult:
        if self.mode not in (CollectorMode.RECORDING, CollectorMode.NEED_TO_SAVE):
            return CommandResult(
                accepted=False,
                level="WARN",
                message=f"DISCARD ignored while {self.mode.value}",
                mode=self.mode,
                session=self.session,
            )

        self.mode = CollectorMode.DISCARD
        return CommandResult(
            accepted=True,
            level="OK",
            message="recording discard requested",
            mode=self.mode,
            session=self.session,
            should_discard=True,
        )
