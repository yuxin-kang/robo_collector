"""Robo Collector final data recording package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DurablePrefix",
    "JournalScan",
    "LandingChannel",
    "LandingError",
    "LandingFaulted",
    "LandingQueueFull",
    "LandingRecord",
    "LandingRecorder",
    "LandingSeal",
    "LandingStateError",
    "LandingWriter",
    "RecoveryError",
    "RecoveryExitCode",
    "RecoveryResult",
    "RequiredSource",
    "SourceFence",
    "read_checkpoint_journal",
    "recover_landing",
    "select_durable_prefix",
]
_LANDING_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Load optional MCAP landing dependencies only when requested."""

    if name not in _LANDING_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".mcap_landing", __name__), name)
    globals()[name] = value
    return value
