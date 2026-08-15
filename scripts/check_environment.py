from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    lock = json.loads((ROOT / "external.lock.json").read_text())
    failures = []
    for name, source in lock["sources"].items():
        if source.get("subdir"):
            continue
        actual = commit((ROOT / source["path"]).resolve())
        if actual != source["commit"]:
            failures.append(f"{name}: {actual} != {source['commit']}")
    if failures:
        raise SystemExit("environment validation failed:\n" + "\n".join(failures))
    print("environment validation passed")


if __name__ == "__main__":
    main()
