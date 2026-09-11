from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "cogvideox15-source-cache/v1"


@dataclass(frozen=True)
class CogVideoCacheManifest:
    model: str
    dtype: str
    width: int
    height: int
    frames: int
    steps: int
    scheduler: str
    reuse_step: int
    latent_shape: tuple[int, ...] | None
    source_uuid: str
    config_fingerprint: str
    schema_version: str = SCHEMA_VERSION
    created_at: float = 0.0

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["latent_shape"] = list(self.latent_shape) if self.latent_shape is not None else None
        return value


def write_manifest(path: Path, manifest: CogVideoCacheManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> CogVideoCacheManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        shape = payload.get("latent_shape")
        return CogVideoCacheManifest(
            model=str(payload["model"]),
            dtype=str(payload["dtype"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            frames=int(payload["frames"]),
            steps=int(payload["steps"]),
            scheduler=str(payload["scheduler"]),
            reuse_step=int(payload["reuse_step"]),
            latent_shape=tuple(int(item) for item in shape) if shape is not None else None,
            source_uuid=str(payload["source_uuid"]),
            config_fingerprint=str(payload["config_fingerprint"]),
            schema_version=str(payload.get("schema_version", "")),
            created_at=float(payload.get("created_at", 0.0)),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid CogVideo source cache manifest: {path}") from exc


def validate_manifest(
    manifest: CogVideoCacheManifest,
    *,
    config: dict[str, Any],
    source_uuid: str | None = None,
    latent: Any | None = None,
) -> None:
    expected = {
        "model": "CogVideoX1.5",
        "dtype": str(config.get("dtype", "bfloat16")),
        "width": int(config.get("width", 1360)),
        "height": int(config.get("height", 768)),
        "frames": int(config.get("frames", 81)),
        "steps": int(config.get("baseline_steps", 50)),
    }
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported CogVideo cache schema: {manifest.schema_version!r}")
    for field, value in expected.items():
        if getattr(manifest, field) != value:
            raise ValueError(f"CogVideo cache {field}={getattr(manifest, field)!r} does not match {value!r}")
    if not 0 < manifest.reuse_step < manifest.steps:
        raise ValueError("CogVideo cache reuse_step must be within the original schedule")
    if source_uuid is not None and manifest.source_uuid != source_uuid:
        raise ValueError("CogVideo cache source UUID does not match replay decision")
    if manifest.config_fingerprint != config_fingerprint(config):
        raise ValueError("CogVideo cache config fingerprint does not match resolved config")
    if latent is not None and manifest.latent_shape is not None:
        payload = latent.get("latents") if isinstance(latent, dict) and "latents" in latent else latent
        shape = tuple(int(item) for item in getattr(payload, "shape", ()))
        if shape and shape != manifest.latent_shape:
            raise ValueError(f"CogVideo cache latent shape {shape} does not match {manifest.latent_shape}")


def config_fingerprint(config: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def create_manifest(config: dict[str, Any], *, source_uuid: str, scheduler: str, reuse_step: int, latent: Any) -> CogVideoCacheManifest:
    payload = latent.get("latents") if isinstance(latent, dict) and "latents" in latent else latent
    shape = tuple(int(item) for item in getattr(payload, "shape", ())) or None
    return CogVideoCacheManifest(
        model="CogVideoX1.5",
        dtype=str(config.get("dtype", "bfloat16")),
        width=int(config.get("width", 1360)),
        height=int(config.get("height", 768)),
        frames=int(config.get("frames", 81)),
        steps=int(config.get("baseline_steps", 50)),
        scheduler=scheduler,
        reuse_step=int(reuse_step),
        latent_shape=shape,
        source_uuid=source_uuid,
        config_fingerprint=config_fingerprint(config),
        created_at=time.time(),
    )


def validate_directory(directory: Path, *, config: dict[str, Any], source_uuid: str | None = None) -> CogVideoCacheManifest:
    manifest = read_manifest(directory / "manifest.json")
    payload = directory / "latent.pt"
    if not payload.is_file():
        raise FileNotFoundError(f"CogVideo source cache payload is missing: {payload}")
    validate_manifest(manifest, config=config, source_uuid=source_uuid)
    return manifest
