from __future__ import annotations

import sys
from pathlib import Path


def load_chai_source() -> None:
    source = Path(__file__).resolve().parents[2] / "CHAI"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
