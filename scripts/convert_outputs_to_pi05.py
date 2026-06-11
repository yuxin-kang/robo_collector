#!/usr/bin/env python3
"""Thin CLI wrapper for the Robo Collector OpenPI pi0.5 converter."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src/robo_collector"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from robo_collector.pi05_converter import main


if __name__ == "__main__":
    raise SystemExit(main())
