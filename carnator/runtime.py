"""Repository-local paths and JSON helpers used by T2V runners."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baselines import Profile, profile_path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_profile(name: str) -> Profile:
    payload = read_json(profile_path(ROOT, name))
    defaults = payload.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError(f"profile {name} has no defaults object")
    if payload.get("id") != name:
        raise ValueError(f"profile identifier mismatch in {profile_path(ROOT, name)}")
    if name == "t2v-14b-distilled" and defaults.get("enable_cfg") is not False:
        raise ValueError("t2v-14b-distilled must disable CFG")
    if name == "ti2v-5b" and float(defaults.get("guidance_scale", 0.0)) != 5.0:
        raise ValueError("ti2v-5b must default to guidance_scale=5.0")
    return Profile(
        identifier=name,
        runtime=str(payload["runtime"]),
        checkpoint_env=str(payload["checkpoint_env"]),
        checkpoint_subdir=str(payload["checkpoint_subdir"]),
        defaults=defaults,
    )


def default_checkpoint(profile: Profile) -> Path:
    configured = os.environ.get(profile.checkpoint_env)
    if configured:
        return Path(configured).expanduser()
    return ROOT / "models" / profile.checkpoint_subdir


def dependency_versions() -> dict[str, str | None]:
    external = ROOT / "external"
    return {
        "wan2.2": git_revision(external / "wan2.2"),
        "lightx2v": git_revision(external / "lightx2v"),
        "spargeattn": git_revision(external / "spargeattn"),
        "python": sys.version.split()[0],
    }
