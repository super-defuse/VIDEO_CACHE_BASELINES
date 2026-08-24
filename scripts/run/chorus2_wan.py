#!/usr/bin/env python3
"""Run the native Wan2.2 Chorus2 baseline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    command = [sys.executable, "-m", "chorus2.wan_ti2v_5b", *sys.argv[1:]]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "external" / "wan2.2"), environment.get("PYTHONPATH", "")))
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
