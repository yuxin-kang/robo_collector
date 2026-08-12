"""Single-owner background execution for durable episode finalization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from threading import Event, Lock, Thread
from typing import Callable, Generic, TypeVar, cast


ResultT = TypeVar("ResultT")
ProgressReporter = Callable[[str], None]
_UNSET = object()


@dataclass(frozen=True)
class SaveProgress:
    phase: str
    monotonic_sec: float


class EpisodeSaveWorker(Generic[ResultT]):
    """Runs at most one save at a time and transports progress safely."""

    def __init__(self) -> None:
        self._thread: Thread | None = None
        self._done = Event()
        self._state_lock = Lock()
        self._result: object = _UNSET
        self._error: BaseException | None = None
        self._progress: SimpleQueue[SaveProgress] = SimpleQueue()
        self._shutdown = False

    @property
    def has_active(self) -> bool:
        with self._state_lock:
            return self._thread is not None

    @property
    def done(self) -> bool:
        with self._state_lock:
            return self._thread is not None and self._done.is_set()

    def start(self, save: Callable[[ProgressReporter], ResultT]) -> None:
        with self._state_lock:
            if self._shutdown:
                raise RuntimeError("save worker is shut down")
            if self._thread is not None:
                raise RuntimeError("episode save is already running")
            self.drain_progress()
            self._done.clear()
            self._result = _UNSET
            self._error = None
            thread = Thread(
                target=self._run,
                args=(save,),
                name="robo-collector-save",
                daemon=False,
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                raise

    def drain_progress(self) -> list[SaveProgress]:
        progress: list[SaveProgress] = []
        while True:
            try:
                progress.append(self._progress.get_nowait())
            except Empty:
                return progress

    def take_result(self, timeout: float | None = None) -> ResultT:
        with self._state_lock:
            thread = self._thread
        if thread is None:
            raise RuntimeError("no episode save is running")
        if not self._done.wait(timeout):
            raise TimeoutError("episode save did not finish before timeout")
        thread.join(timeout=0.0)
        with self._state_lock:
            error = self._error
            result = self._result
            self._thread = None
            self._error = None
            self._result = _UNSET
        if error is not None:
            raise error
        if result is _UNSET:
            raise RuntimeError("episode save completed without a result")
        return cast(ResultT, result)

    def shutdown(self, timeout: float | None = None) -> bool:
        with self._state_lock:
            self._shutdown = True
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _run(self, save: Callable[[ProgressReporter], ResultT]) -> None:
        try:
            result = save(self._report_progress)
        except BaseException as exc:
            with self._state_lock:
                self._error = exc
        else:
            with self._state_lock:
                self._result = result
        finally:
            self._done.set()

    def _report_progress(self, phase: str) -> None:
        normalized_phase = phase.strip()
        if not normalized_phase:
            return
        self._progress.put(
            SaveProgress(
                phase=normalized_phase,
                monotonic_sec=time.monotonic(),
            )
        )
