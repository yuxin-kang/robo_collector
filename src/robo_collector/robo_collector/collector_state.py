"""Pure recording command state machine for the LeRobot collector."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Literal


class CollectorMode(str, Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    NEED_TO_SAVE = "NEED_TO_SAVE"
    FAILED = "FAILED"
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


@dataclass(frozen=True)
class CommandFingerprint:
    """Payload identity used to make non-empty command IDs idempotent."""

    command: int
    task_prompt: str
    episode_id: str
    force: bool


@dataclass(frozen=True)
class CommandReplay:
    disposition: Literal["REPLAY", "CONFLICT"]
    outcome: str


class CommandReceiptLedger:
    """Keeps a bounded set of terminal command receipts for retry replay."""

    def __init__(self, capacity: int = 128) -> None:
        if capacity <= 0:
            raise ValueError("command receipt capacity must be positive")
        self._capacity = capacity
        self._entries: dict[str, tuple[CommandFingerprint, str, bool]] = {}

    def lookup(
        self, command_id: str, fingerprint: CommandFingerprint
    ) -> CommandReplay | None:
        normalized_id = command_id.strip()
        if not normalized_id:
            return None
        existing = self._entries.get(normalized_id)
        if existing is None:
            return None
        existing_fingerprint, outcome, replayable = existing
        if existing_fingerprint == fingerprint:
            return CommandReplay("REPLAY", outcome) if replayable else None
        return CommandReplay("CONFLICT", outcome)

    def remember(
        self,
        command_id: str,
        fingerprint: CommandFingerprint,
        outcome: str,
        *,
        replayable: bool = True,
    ) -> None:
        normalized_id = command_id.strip()
        if not normalized_id:
            return
        existing = self._entries.get(normalized_id)
        if existing is not None and existing[0] != fingerprint:
            raise ValueError("cannot reuse command_id with a different payload")
        self._entries[normalized_id] = (fingerprint, outcome, replayable)
        while len(self._entries) > self._capacity:
            oldest_id = next(iter(self._entries))
            del self._entries[oldest_id]


class RecordStateMachine:
    """Owns legal transitions for START/STOP/DISCARD commands."""

    def __init__(self) -> None:
        self.mode = CollectorMode.IDLE
        self.session: RecordingSession | None = None
        self.failure_reason = ""

    def handle_command(
        self,
        command: int,
        *,
        task_prompt: str = "",
        episode_id: str = "",
        force: bool = False,
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
            return self._stop(episode_id, force=force)
        return self._discard(episode_id, force=force)

    def mark_saved(self) -> None:
        if self.mode != CollectorMode.NEED_TO_SAVE:
            raise RuntimeError(f"cannot mark saved while mode is {self.mode.value}")
        self.mode = CollectorMode.IDLE
        self.session = None
        self.failure_reason = ""

    def mark_discarded(self) -> None:
        if self.mode != CollectorMode.DISCARD:
            raise RuntimeError(f"cannot mark discarded while mode is {self.mode.value}")
        self.mode = CollectorMode.IDLE
        self.session = None
        self.failure_reason = ""

    def mark_discard_failed(self, reason: str) -> None:
        if self.mode != CollectorMode.DISCARD:
            raise RuntimeError(
                f"cannot mark discard failed while mode is {self.mode.value}"
            )
        self.mode = CollectorMode.FAILED
        self.failure_reason = reason.strip() or "discard failed"

    def mark_save_failed(self) -> None:
        if self.mode != CollectorMode.NEED_TO_SAVE:
            raise RuntimeError(f"cannot mark save failed while mode is {self.mode.value}")
        self.mark_failed("save failed; discard required")

    def mark_failed(self, reason: str) -> None:
        if self.mode not in (CollectorMode.RECORDING, CollectorMode.NEED_TO_SAVE):
            raise RuntimeError(f"cannot mark failed while mode is {self.mode.value}")
        self.mode = CollectorMode.FAILED
        self.failure_reason = reason.strip() or "discard required"

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

    def _stop(self, episode_id: str, *, force: bool) -> CommandResult:
        if self.mode != CollectorMode.RECORDING:
            return CommandResult(
                accepted=False,
                level="WARN",
                message=f"STOP ignored while {self.mode.value}",
                mode=self.mode,
                session=self.session,
            )
        mismatch = self._episode_mismatch(episode_id, force=force)
        if mismatch is not None:
            return mismatch

        self.mode = CollectorMode.NEED_TO_SAVE
        return CommandResult(
            accepted=True,
            level="OK",
            message="recording stopped; save requested",
            mode=self.mode,
            session=self.session,
            should_save=True,
        )

    def _discard(self, episode_id: str, *, force: bool) -> CommandResult:
        if self.mode not in (
            CollectorMode.RECORDING,
            CollectorMode.NEED_TO_SAVE,
            CollectorMode.FAILED,
        ):
            return CommandResult(
                accepted=False,
                level="WARN",
                message=f"DISCARD ignored while {self.mode.value}",
                mode=self.mode,
                session=self.session,
            )
        mismatch = self._episode_mismatch(episode_id, force=force)
        if mismatch is not None:
            return mismatch

        self.mode = CollectorMode.DISCARD
        return CommandResult(
            accepted=True,
            level="OK",
            message="recording discard requested",
            mode=self.mode,
            session=self.session,
            should_discard=True,
        )

    def _episode_mismatch(
        self, episode_id: str, *, force: bool
    ) -> CommandResult | None:
        expected_episode_id = episode_id.strip()
        active_episode_id = self.session.episode_id if self.session is not None else ""
        if force or not expected_episode_id or expected_episode_id == active_episode_id:
            return None
        return CommandResult(
            accepted=False,
            level="WARN",
            message=(
                "command rejected for a different episode: "
                f"expected {expected_episode_id!r}, active {active_episode_id!r}"
            ),
            mode=self.mode,
            session=self.session,
        )


def recording_safety_reason(
    *,
    elapsed_sec: float,
    frame_count: int,
    max_duration_sec: float,
    max_frames: int,
    free_disk_bytes: int,
    min_free_disk_bytes: int,
) -> str | None:
    if elapsed_sec >= max_duration_sec:
        return (
            "maximum episode duration reached: "
            f"{elapsed_sec:.3f}s >= {max_duration_sec:.3f}s"
        )
    if frame_count >= max_frames:
        return f"maximum episode frame count reached: {frame_count} >= {max_frames}"
    if free_disk_bytes < min_free_disk_bytes:
        return (
            "minimum free disk reserve reached: "
            f"{free_disk_bytes} < {min_free_disk_bytes} bytes"
        )
    return None
