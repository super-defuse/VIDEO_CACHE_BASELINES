"""Stable baseline and model-profile contracts shared by every entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True)
class BaselineSpec:
    """The only dimensions that vary among the C0/B0--B5 baselines."""

    identifier: str
    outer_attention: str
    ffn: str
    region_mode: str
    reuse_mode: str

    @property
    def requires_source(self) -> bool:
        return self.region_mode == "carnator"


BASELINES: Final[dict[str, BaselineSpec]] = {
    "C0": BaselineSpec("C0", "dense", "fp", "full", "none"),
    "B0": BaselineSpec("B0", "dense", "fp", "carnator", "latent"),
    "B1": BaselineSpec("B1", "flashinfer", "fp", "carnator", "latent"),
    "B2": BaselineSpec("B2", "sparge-sage2", "fp", "carnator", "latent"),
    "B3": BaselineSpec("B3", "dense", "w8a8", "carnator", "latent"),
    "B4": BaselineSpec("B4", "flashinfer", "w8a8", "carnator", "latent"),
    "B5": BaselineSpec("B5", "sparge-sage2", "w8a8", "carnator", "latent"),
}


@dataclass(frozen=True)
class Profile:
    """A validated T2V model profile loaded from ``configs/*.json``."""

    identifier: str
    runtime: str
    checkpoint_env: str
    checkpoint_subdir: str
    defaults: dict[str, object]

    @property
    def uses_cfg(self) -> bool:
        return bool(self.defaults["enable_cfg"])

    @property
    def guidance_scale(self) -> float | None:
        value = self.defaults.get("guidance_scale")
        return None if value is None else float(value)


PROFILE_NAMES: Final[tuple[str, ...]] = ("t2v-14b-distilled", "ti2v-5b")


def parse_baseline(value: str) -> BaselineSpec:
    try:
        return BASELINES[value.upper()]
    except KeyError as exc:
        raise ValueError(f"unknown baseline {value!r}; expected one of {', '.join(BASELINES)}") from exc


def parse_layers(value: str) -> list[int]:
    """Parse a comma-separated list of inclusive integer ranges."""

    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lower, upper = (int(item) for item in part.split("-", 1))
            if upper < lower:
                raise ValueError(f"invalid descending layer range: {part}")
            layers.extend(range(lower, upper + 1))
        else:
            layers.append(int(part))
    if not layers:
        raise ValueError("at least one layer is required")
    return sorted(set(layers))


def profile_path(root: Path, name: str) -> Path:
    filenames = {
        "t2v-14b-distilled": "t2v_14b_distilled.json",
        "ti2v-5b": "ti2v_5b.json",
    }
    try:
        return root / "configs" / filenames[name]
    except KeyError as exc:
        raise ValueError(f"unknown model profile {name!r}; expected one of {', '.join(PROFILE_NAMES)}") from exc
