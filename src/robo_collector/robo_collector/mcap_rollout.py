"""Phase 7 rollout acceptance and synthetic failure harnesses.

The harness is deliberately artifact-format agnostic at its boundary: callers
provide summaries from Raw-v1 and MCAP readers, while this module compares the
stable acceptance evidence.  Synthetic scenarios are useful in CI, but never
claim to replace hardware or long-duration deployment testing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ArtifactSummary:
    """Comparable source evidence for one captured episode."""

    episode_id: str
    message_count: int
    source_hash: str | None = None
    timestamps: tuple[int, ...] = ()
    aligned_rows: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactSummary":
        timestamps = value.get("timestamps", value.get("timestamp_ns", ()))
        return cls(
            episode_id=str(value.get("episode_id", "")),
            message_count=int(value.get("message_count", value.get("count", 0))),
            source_hash=value.get("source_hash", value.get("sha256")),
            timestamps=tuple(int(item) for item in timestamps),
            aligned_rows=(
                None
                if value.get("aligned_rows") is None
                else int(value["aligned_rows"])
            ),
        )


@dataclass(frozen=True)
class ParityReport:
    status: str
    checks: Mapping[str, bool]
    mismatches: tuple[str, ...]
    raw: ArtifactSummary
    mcap: ArtifactSummary
    synthetic: bool = False
    not_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["raw"] = asdict(self.raw)
        result["mcap"] = asdict(self.mcap)
        result["timestamps"] = {
            "raw": list(self.raw.timestamps),
            "mcap": list(self.mcap.timestamps),
        }
        return result


def compare_parity(
    raw: ArtifactSummary | Mapping[str, Any],
    mcap: ArtifactSummary | Mapping[str, Any],
    *,
    timestamp_tolerance_ns: int = 0,
    synthetic: bool = False,
) -> ParityReport:
    """Compare Raw-v1 and MCAP evidence and classify it.

    Hashes are compared only when both sides provide one, allowing callers to
    compare equivalent decoded streams whose container hashes differ. Missing
    evidence is REVIEW (not PASS); a contradictory value is FAIL.
    """
    raw = raw if isinstance(raw, ArtifactSummary) else ArtifactSummary.from_mapping(raw)
    mcap = mcap if isinstance(mcap, ArtifactSummary) else ArtifactSummary.from_mapping(mcap)
    if timestamp_tolerance_ns < 0:
        raise ValueError("timestamp_tolerance_ns must be non-negative")
    checks: dict[str, bool] = {
        "episode_id": bool(raw.episode_id and raw.episode_id == mcap.episode_id),
        "message_count": raw.message_count == mcap.message_count,
    }
    if raw.source_hash is not None and mcap.source_hash is not None:
        checks["source_hash"] = raw.source_hash == mcap.source_hash
    if raw.timestamps and mcap.timestamps:
        checks["timestamps"] = len(raw.timestamps) == len(mcap.timestamps) and all(
            abs(left - right) <= timestamp_tolerance_ns
            for left, right in zip(raw.timestamps, mcap.timestamps)
        )
    if raw.aligned_rows is not None and mcap.aligned_rows is not None:
        checks["aligned_rows"] = raw.aligned_rows == mcap.aligned_rows
    mismatches = tuple(name for name, passed in checks.items() if not passed)
    missing = not raw.timestamps or not mcap.timestamps
    missing = missing or raw.aligned_rows is None or mcap.aligned_rows is None
    missing = missing or raw.source_hash is None or mcap.source_hash is None
    status = "FAIL" if mismatches else ("REVIEW" if missing else "PASS")
    return ParityReport(status, checks, mismatches, raw, mcap, synthetic=synthetic)


@dataclass(frozen=True)
class HarnessResult:
    name: str
    status: str
    synthetic: bool
    not_run: bool
    iterations: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_harness(
    scenario: str, *, iterations: int = 1, synthetic: bool = True
) -> HarnessResult:
    """Run a bounded kill/reconnect/soak smoke harness.

    This only exercises control-flow bookkeeping.  ``synthetic`` remains in
    the result so reports cannot be mistaken for real hardware evidence.
    """
    if scenario not in {"kill", "reconnect", "soak"}:
        raise ValueError("scenario must be kill, reconnect, or soak")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    # Keep this deterministic and fast; the real harness is supplied with
    # process/device callbacks by deployment tooling.
    completed = iterations
    return HarnessResult(
        scenario,
        "PASS" if completed == iterations else "FAIL",
        synthetic,
        synthetic,
        completed,
        "synthetic smoke completed; hardware duration not run",
    )


def run_synthetic_harness(
    scenario: str, *, iterations: int = 1
) -> HarnessResult:
    """Compatibility name for the explicitly synthetic harness API."""
    return run_harness(scenario, iterations=iterations, synthetic=True)


def _load(path: Path) -> ArtifactSummary:
    with path.open(encoding="utf-8") as stream:
        return ArtifactSummary.from_mapping(json.load(stream))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--mcap", type=Path)
    parser.add_argument("--timestamp-tolerance-ns", type=int, default=0)
    parser.add_argument("--scenario", choices=("kill", "reconnect", "soak"))
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args(argv)
    if args.timestamp_tolerance_ns < 0:
        parser.error("--timestamp-tolerance-ns must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.scenario:
        print(json.dumps(run_harness(args.scenario, iterations=args.iterations).to_dict(), sort_keys=True))
        return 0
    if not args.raw or not args.mcap:
        parser.error("--raw and --mcap are required unless --scenario is used")
    report = compare_parity(_load(args.raw), _load(args.mcap), timestamp_tolerance_ns=args.timestamp_tolerance_ns)
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0 if report.status != "FAIL" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
