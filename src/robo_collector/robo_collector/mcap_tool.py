"""Operator-facing MCAP v1 inspection and recovery commands.

The command emits JSON so it is safe to use from launch/health checks.  It is
deliberately a thin adapter around the canonical validator and landing
recovery implementation; it never rewrites an input artifact.
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Any


def _info(path: Path) -> dict[str, Any]:
    from mcap.reader import make_reader

    channels: dict[str, dict[str, Any]] = {}
    messages = 0
    start: int | None = None
    end: int | None = None
    with path.open("rb") as stream:
        reader = make_reader(stream)
        for _, channel, message in reader.iter_messages():
            messages += 1
            item = channels.setdefault(
                channel.topic,
                {"topic": channel.topic, "message_encoding": channel.message_encoding, "count": 0},
            )
            item["count"] += 1
            start = message.log_time if start is None else min(start, message.log_time)
            end = message.log_time if end is None else max(end, message.log_time)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "message_count": messages,
        "channels": [channels[key] for key in sorted(channels)],
        "start_log_time_ns": start,
        "end_log_time_ns": end,
    }


def _doctor(path: Path, group: str | None) -> dict[str, Any]:
    from .canonical_mcap import validate_canonical_mcap

    if group:
        details = validate_canonical_mcap(path, expected_group=group)
    else:
        details = _info(path)
    return {"path": str(path), "ok": True, "details": details}


def _recover(path: Path, attempt: int) -> dict[str, Any]:
    from .mcap_landing import recover_landing

    result = recover_landing(path, attempt=attempt)
    return {
        "path": str(path),
        "ok": result.ok,
        "exit_code": int(result.exit_code),
        "recovered_path": str(result.recovered_path) if result.recovered_path else None,
        "source_complete": result.source_complete,
        "detail": result.detail,
    }


def _ready_manifest(path: Path) -> dict[str, Any]:
    """Load a manifest and enforce the publication boundary for tooling."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "READY":
        raise ValueError("tooling requires a READY manifest")
    return manifest


def _replay(path: Path) -> dict[str, Any]:
    manifest = _ready_manifest(path)
    artifacts = manifest.get("canonical", manifest.get("canonical_artifacts", {}))
    if not isinstance(artifacts, dict):
        raise ValueError("READY manifest has no canonical artifact inventory")
    return {"path": str(path), "ok": True, "episode_id": manifest.get("episode_id"),
            "artifact_count": len(artifacts), "replay": "read_only"}


def _migration(path: Path) -> dict[str, Any]:
    manifest = _ready_manifest(path)
    artifacts = manifest.get("canonical", manifest.get("canonical_artifacts", {}))
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("READY manifest has no canonical artifacts to migrate")
    return {"path": str(path), "ok": True, "episode_id": manifest.get("episode_id"),
            "artifact_count": len(artifacts), "migration": "validated_plan"}


def _benchmark(path: Path, iterations: int) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    started = time.perf_counter_ns()
    size = 0
    for _ in range(iterations):
        size = path.stat().st_size
    elapsed = time.perf_counter_ns() - started
    return {"path": str(path), "ok": True, "iterations": iterations,
            "size_bytes": size, "elapsed_ns": elapsed,
            "mean_ns": elapsed // iterations}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcap-tool", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("info", help="summarize a readable MCAP")
    info.add_argument("path", type=Path)
    doctor = sub.add_parser("doctor", help="validate an MCAP and report JSON")
    doctor.add_argument("path", type=Path)
    doctor.add_argument("--group", choices=("camera", "robot"))
    recover = sub.add_parser("recover", help="replay the trusted durable landing prefix")
    recover.add_argument("episode_dir", type=Path)
    recover.add_argument("--attempt", type=int, required=True)
    replay = sub.add_parser("replay", help="inspect a READY manifest without writing")
    replay.add_argument("manifest", type=Path)
    replay.add_argument("--format", choices=("json",), default="json")
    migration = sub.add_parser("migration", help="validate a READY migration plan")
    migration.add_argument("manifest", type=Path, nargs="?")
    migration.add_argument("--raw-root", type=Path)
    benchmark = sub.add_parser("benchmark", help="benchmark filesystem metadata access")
    benchmark.add_argument("path", type=Path)
    benchmark.add_argument("--iterations", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "info":
            result = _info(args.path)
            code = 0
        elif args.command == "doctor":
            result = _doctor(args.path, args.group)
            code = 0
        elif args.command == "recover":
            result = _recover(args.episode_dir, args.attempt)
            code = int(result["exit_code"])
        elif args.command == "replay":
            result, code = _replay(args.manifest), 0
        elif args.command == "migration":
            manifest = args.manifest or ((args.raw_root / "manifest.json") if args.raw_root else None)
            if manifest is None:
                raise ValueError("migration requires a manifest or --raw-root")
            result, code = _migration(manifest), 0
        else:
            result, code = _benchmark(args.path, args.iterations), 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        result = {"ok": False, "error": str(exc)}
        code = 1
    print(json.dumps(result, sort_keys=True))
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
