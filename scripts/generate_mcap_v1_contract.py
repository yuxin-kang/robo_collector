#!/usr/bin/env python3
"""Regenerate the checked-in MCAP v1 Python and descriptor artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTO_ROOT = ROOT / "proto"
PROTO = PROTO_ROOT / "robo_collector/mcap/v1/episode.proto"
DESCRIPTOR = PROTO.with_name("episode.descriptor.pb")
PYTHON_OUT = ROOT / "src/robo_collector"
DESCRIPTOR_MODULE = PYTHON_OUT / "robo_collector/mcap/v1/episode_descriptor.py"


def main() -> int:
    try:
        import grpc_tools  # noqa: F401
    except ImportError:
        print(
            "grpcio-tools is required; install requirements-mcap-dev.txt",
            file=sys.stderr,
        )
        return 2
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_ROOT}",
            f"--python_out={PYTHON_OUT}",
            f"--descriptor_set_out={DESCRIPTOR}",
            "--include_imports",
            str(PROTO),
        ],
        check=True,
    )
    raw = DESCRIPTOR.read_bytes()
    hex_value = raw.hex()
    chunks = [hex_value[index : index + 96] for index in range(0, len(hex_value), 96)]
    DESCRIPTOR_MODULE.write_text(
        '"""Checked-in deterministic FileDescriptorSet for MCAP v1."""\n\n'
        "DESCRIPTOR_SET_BYTES = bytes.fromhex(\n"
        + "".join(f'    "{chunk}"\n' for chunk in chunks)
        + ")\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
