#!/usr/bin/env python3
"""Prepare one shared Wan source cache for cross-request baselines.

The script deliberately does not run retrieval or target generation.  A source
request is denoised once, then its post-scheduler latents and canonical outer
attention masks are materialized in a method-neutral directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "shared-cross-request-cache/v3"
ATTENTION_FORMAT = "carnator-source-block-mask-bitpacked-v1"
ALL_WAN_LAYERS = tuple(range(30))
NATIVE_OUTER_SPARSE_LAYERS = "0-9,21-29"
CHORUS2_RAW_MIDDLE_LAYERS = frozenset(range(10, 21))
# Measured from the previous compatible run. These constants are only used by
# --estimate-only, which deliberately performs no GPU work.
ESTIMATED_LATENT_FILE_BYTES = 20_952_680
ESTIMATED_SOURCE_VIDEO_BYTES = 24_788_802
ESTIMATED_METADATA_BYTES_PER_ENTRY = 6 * 1024 * 1024
ESTIMATED_CARNATOR_MASK_BUNDLE_BYTES = 17 * 1024 * 1024
DEFAULT_INPUT = ROOT.parent / "VidProM" / "clusters" / "cluster_02_2170.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "shared_cross_request_cache" / "vidprom_cluster_02_2170_first1000"
DEFAULT_MODEL_DIR = Path(os.environ.get("CARNATOR_TI2V_5B_MODEL_DIR", "/home/tangxuebin/models/Wan2.2-TI2V-5B"))


@dataclass(frozen=True)
class PromptRow:
    request_index: int
    uuid: str
    prompt: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_prompt_rows(path: Path, limit: int) -> list[PromptRow]:
    if limit < 1:
        raise ValueError("--limit must be positive")
    rows: list[PromptRow] = []
    seen: set[str] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = {"uuid", "prompt"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"input CSV is missing columns: {', '.join(sorted(missing))}")
        for request_index, row in enumerate(reader):
            if len(rows) >= limit:
                break
            uuid = str(row.get("uuid", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            if not uuid or not prompt:
                raise ValueError(f"CSV row {request_index + 2} has an empty uuid or prompt")
            if uuid in seen:
                raise ValueError(f"input CSV has duplicate UUID in selected rows: {uuid}")
            seen.add(uuid)
            rows.append(PromptRow(request_index=request_index, uuid=uuid, prompt=prompt))
    if not rows:
        raise ValueError("input CSV has no selected rows")
    return rows


def parse_gpus(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value < 0 for value in values) or len(values) != len(set(values)):
        raise ValueError("--gpus must be a non-empty, duplicate-free list of non-negative GPU IDs")
    return values


def parse_layer_ranges(value: str) -> list[int]:
    layers: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = item.split("-", 1)
            layers.update(range(int(start), int(stop) + 1))
        else:
            layers.add(int(item))
    if not layers or min(layers) < 0 or max(layers) > 29:
        raise ValueError("--attention-layers must contain Wan layers in [0, 29]")
    return sorted(layers)


def cache_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_dir": str(args.model_dir.resolve()),
        "height": int(args.height),
        "width": int(args.width),
        "frames": int(args.frames),
        "fps": int(args.fps),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "scheduler": args.scheduler,
        "solver_order": int(args.solver_order),
        "flow_shift": float(args.flow_shift),
        "guidance_scale": float(args.guidance_scale),
        "model_dtype": args.model_dtype,
        "attention_layers": parse_layer_ranges(args.attention_layers),
        "q_block_size": int(args.q_block_size),
        "k_block_size": int(args.k_block_size),
        "top_p": float(args.top_p),
        "min_top_k": float(args.min_top_k),
        "attention_dtype": args.attention_dtype,
        "reuse_start_step": 10,
        "chorus2_raw_middle_layers": sorted(CHORUS2_RAW_MIDDLE_LAYERS),
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(config, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def model_tree_fingerprint(path: Path) -> str:
    """Cheap reproducibility identity without hashing multi-gigabyte weights."""

    digest = hashlib.sha256()
    if not path.exists():
        digest.update(b"missing")
        return digest.hexdigest()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        stat = item.stat()
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


def pack_bool_mask(mask: Any) -> tuple[bytes, int]:
    import torch

    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 3:
        raise ValueError("canonical attention mask must be bool[heads,q_blocks,k_blocks]")
    flat = mask.detach().cpu().contiguous().reshape(-1).to(torch.uint8)
    element_count = int(flat.numel())
    padding = (-element_count) % 8
    if padding:
        flat = torch.cat((flat, torch.zeros(padding, dtype=torch.uint8)))
    weights = (1 << torch.arange(8, dtype=torch.uint8)).view(1, 8)
    packed = (flat.view(-1, 8) * weights).sum(dim=1, dtype=torch.int16).to(torch.uint8)
    return packed.numpy().tobytes(), element_count


def unpack_bool_mask(payload: bytes, shape: list[int] | tuple[int, int, int]) -> Any:
    import torch

    if len(shape) != 3 or any(int(value) < 1 for value in shape):
        raise ValueError("mask shape must be three positive dimensions")
    element_count = math.prod(int(value) for value in shape)
    if len(payload) != math.ceil(element_count / 8):
        raise ValueError("bit-packed mask byte size does not match metadata")
    raw = torch.frombuffer(memoryview(bytearray(payload)), dtype=torch.uint8)
    shifts = torch.arange(8, dtype=torch.uint8)
    unpacked = ((raw.unsqueeze(1) >> shifts) & 1).reshape(-1)[:element_count]
    return unpacked.to(torch.bool).reshape(tuple(int(value) for value in shape)).contiguous()


def load_canonical_mask(root: Path, entry: dict[str, Any]) -> Any:
    path = root / str(entry["path"])
    payload = path.read_bytes()
    if sha256_bytes(payload) != entry["sha256"]:
        raise ValueError(f"attention mask checksum mismatch: {path}")
    if int(entry.get("packed_elements", 0)) != math.prod(int(value) for value in entry["mask_shape"]):
        raise ValueError(f"attention mask element count does not match metadata: {path}")
    return unpack_bool_mask(payload, entry["mask_shape"])


def flashinfer_mask_view(mask: Any) -> Any:
    """Return the canonical per-head mask consumed to build a FlashInfer plan."""

    return mask


def spargeattn_mask_view(mask: Any) -> Any:
    """Return SpargeAttn's B,H,Qblock,Kblock mask_id view without a copy."""

    return mask.unsqueeze(0)


@dataclass
class _MaskWriteTask:
    path: Path
    payload: bytes
    record: dict[str, Any]


class _AsyncMaskWriter:
    def __init__(self, max_pending: int) -> None:
        self._tasks: queue.Queue[_MaskWriteTask | None] = queue.Queue(maxsize=max_pending)
        self._error: BaseException | None = None
        self.write_seconds = 0.0
        self.bytes_written = 0
        self._thread = threading.Thread(target=self._run, name="shared-mask-writer", daemon=True)
        self._thread.start()

    def submit(self, task: _MaskWriteTask) -> None:
        while True:
            if self._error is not None:
                raise RuntimeError("asynchronous attention write failed") from self._error
            try:
                self._tasks.put(task, timeout=0.2)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        while True:
            if self._error is not None:
                raise RuntimeError("asynchronous attention write failed") from self._error
            try:
                self._tasks.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("asynchronous attention write failed") from self._error

    def _run(self) -> None:
        try:
            while True:
                task = self._tasks.get()
                if task is None:
                    return
                started = time.perf_counter()
                task.path.parent.mkdir(parents=True, exist_ok=True)
                task.path.write_bytes(task.payload)
                self.write_seconds += time.perf_counter() - started
                self.bytes_written += len(task.payload)
                task.record["sha256"] = sha256_bytes(task.payload)
        except BaseException as exc:  # Propagate writer failures to the model worker.
            self._error = exc


class SharedAttentionCapture:
    """Write Carnator's raw source routing in the packed format shared by Chorus2."""

    def __init__(
        self,
        root: Path,
        *,
        source_uuid: str,
        config: dict[str, Any],
        cfg_branch_count: int,
        max_pending_writes: int,
    ) -> None:
        from carnator.outer_sparse import BlockMaskConfig

        self.root = root
        self.source_uuid = source_uuid
        self.config = BlockMaskConfig(
            q_block_size=int(config["q_block_size"]),
            k_block_size=int(config["k_block_size"]),
            top_p=float(config["top_p"]),
            min_top_k=float(config["min_top_k"]),
        )
        self.cfg_branch_count = int(cfg_branch_count)
        self.records: list[dict[str, Any]] = []
        self.capture_seconds = 0.0
        self.writer = _AsyncMaskWriter(max_pending_writes)
        self.closed = False

    def capture(self, completed_step: int, branch: str, layer: int, q: Any, k: Any) -> None:
        from carnator.outer_sparse import block_attention_mass, block_mass_to_mask

        if completed_step < 11 or completed_step > 50:
            raise ValueError(f"unexpected captured destination step: {completed_step}")
        started = time.perf_counter()
        mass = block_attention_mass(q, k, self.config.q_block_size, self.config.k_block_size)
        mask, coverage = block_mass_to_mask(mass, self.config.top_p, self.config.min_top_k)
        if q.is_cuda:
            import torch

            torch.cuda.synchronize(q.device)

        mask_cpu = mask.detach().cpu().contiguous()
        payload, element_count = pack_bool_mask(mask_cpu)
        from_step = completed_step - 1
        relative = Path("maskbits") / f"from_step_{from_step:03d}" / branch / f"layer_{layer:02d}.bin"
        head_densities = mask_cpu.float().mean(dim=(1, 2))
        record = {
            "step": int(completed_step),
            "from_step": from_step,
            "cfg_branch": branch,
            "layer": int(layer),
            "path": str(relative),
            "shape": [int(value) for value in q.shape],
            "dtype": str(q.dtype),
            "mask_shape": [int(value) for value in mask_cpu.shape],
            "packed_elements": element_count,
            "qkv_shape": [int(value) for value in q.shape],
            "qkv_dtype": str(q.dtype),
            "density": float(mask_cpu.float().mean().item()),
            "coverage": float(coverage),
            "head_density_min": float(head_densities.min().item()),
            "head_density_max": float(head_densities.max().item()),
        }
        self.records.append(record)
        self.writer.submit(_MaskWriteTask(self.root / relative, payload, record))
        self.capture_seconds += time.perf_counter() - started

    def close(self) -> dict[str, Any]:
        if not self.closed:
            self.writer.close()
            self.closed = True
            self.root.mkdir(parents=True, exist_ok=True)
            write_json(self.root / "metadata.json", self.metadata())
        return self.metadata()

    def metadata(self) -> dict[str, Any]:
        records = sorted(self.records, key=lambda item: (item["from_step"], item["cfg_branch"], item["layer"]))
        return {
            "format": ATTENTION_FORMAT,
            "source_uuid": self.source_uuid,
            "cfg_branches": self.cfg_branch_count,
            "mask": asdict(self.config),
            "capture_steps": list(range(11, 51)),
            "entries": records,
            "backend_views": {
                "flashinfer": "unpack canonical mask then build packed-BSR or VariableBlock plan",
                "spargeattn": "unpack canonical mask then add batch dimension for mask_id",
            },
            "capture_seconds": self.capture_seconds,
            "write_seconds": self.writer.write_seconds,
            "bytes_written": self.writer.bytes_written,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "format": ATTENTION_FORMAT,
            "entries": len(self.records),
            "capture_seconds": self.capture_seconds,
            "write_seconds": self.writer.write_seconds,
            "cfg_branches": self.cfg_branch_count,
        }


def _relative_artifact(path: Path, entry_root: Path) -> dict[str, Any]:
    relative = path.relative_to(entry_root)
    return {"path": str(relative), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _validate_mask_cache(metadata_path: Path, expected_formats: set[str]) -> bool:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") not in expected_formats:
            return False
        root = metadata_path.parent
        for record in metadata.get("entries", []):
            path = root / str(record["path"])
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def entry_is_complete(entry_root: Path, *, expected_uuid: str, expected_config_fingerprint: str) -> bool:
    manifest_path = entry_root / "manifest.json"
    status_path = entry_root / "status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            manifest.get("format") != SCHEMA_VERSION
            or manifest.get("uuid") != expected_uuid
            or manifest.get("config_fingerprint") != expected_config_fingerprint
            or status.get("status") != "completed"
        ):
            return False
        artifacts = manifest["artifacts"]
        carnator = artifacts["carnator"]
        required = [
            artifacts["source_video"],
            carnator["source_manifest"],
            carnator["initial_latent"],
            *carnator["trajectory"],
            *carnator["fusion_mask_files"],
        ]
        for record in required:
            path = entry_root / str(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                return False
        raw_metadata = entry_root / str(carnator["raw_block_mask_metadata"]["path"])
        if not _validate_mask_cache(raw_metadata, {ATTENTION_FORMAT}):
            return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def entry_bytes(entry_root: Path) -> int:
    return sum(path.stat().st_size for path in entry_root.rglob("*") if path.is_file())


class NativeSharedCacheWorker:
    def __init__(self, worker_config: dict[str, Any]) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_config["physical_gpu"])
        from carnator.adapters import wan_ti2v_5b as native

        self.native = native
        self.root = Path(worker_config["output"])
        self.cache_config = dict(worker_config["cache_config"])
        self.fingerprint = str(worker_config["config_fingerprint"])
        self.max_pending_writes = int(worker_config["max_pending_writes"])
        args = native.build_parser().parse_args([])
        args.no_env_setup = True
        args.model_dir = Path(self.cache_config["model_dir"])
        args.device = "cuda:0"
        args.artifact_root = self.root
        args.results_root = self.root
        args.run_id = "shared-source"
        args.steps = int(self.cache_config["steps"])
        args.reuse_start_step = 10
        args.height = int(self.cache_config["height"])
        args.width = int(self.cache_config["width"])
        args.frames = int(self.cache_config["frames"])
        args.fps = int(self.cache_config["fps"])
        args.seed = int(self.cache_config["seed"])
        args.scheduler = self.cache_config["scheduler"]
        args.solver_order = int(self.cache_config["solver_order"])
        args.flow_shift = float(self.cache_config["flow_shift"])
        args.guidance_scale = float(self.cache_config["guidance_scale"])
        args.model_dtype = self.cache_config["model_dtype"]
        args.convert_model_dtype = True
        args.offload_model = False
        args.t5_cpu = False
        args.disable_progress = True
        args.capture_outer_block_masks = False
        args.outer_attn_backend = "dense"
        # Native validation reserves 10--20 for Carnator's region path. The
        # independent capture hook below still persists all 30 raw layers for
        # Chorus2 and Carnator from the same source forward pass.
        args.outer_sparse_layers = NATIVE_OUTER_SPARSE_LAYERS
        args.outer_mask_q_block = int(self.cache_config["q_block_size"])
        args.outer_mask_k_block = int(self.cache_config["k_block_size"])
        args.outer_mask_top_p = float(self.cache_config["top_p"])
        args.outer_mask_min_top_k = float(self.cache_config["min_top_k"])
        args.outer_sparse_dtype = self.cache_config["attention_dtype"]
        native.ensure_import_paths(args)
        native.validate_experiment_config(args)
        self.args = args
        self.runtime = native.load_runtime(args)

    def close(self) -> None:
        self.runtime.profiler.close()
        self.runtime.pipe.model.cpu()
        self.runtime.torch.cuda.empty_cache()

    def prepare(self, row: PromptRow) -> dict[str, Any]:
        import torch

        from carnator.outer_sparse import OuterSparseCaptureHooks, OuterSparseState, cfg_branches

        final_dir = self.root / "entries" / row.uuid
        if entry_is_complete(final_dir, expected_uuid=row.uuid, expected_config_fingerprint=self.fingerprint):
            return {"status": "skipped", "uuid": row.uuid, "request_index": row.request_index}
        staging = self.root / "entries" / ".staging" / f"{row.uuid}.{os.getpid()}.{time.time_ns()}"
        staging.mkdir(parents=True, exist_ok=False)
        capture: SharedAttentionCapture | None = None
        started = time.perf_counter()
        try:
            self.args.source_prompt = row.prompt
            self.args.source_positive = ""
            cfg_count = cfg_branches(self.args.guidance_scale)
            carnator_root = staging / "carnator" / "cache" / "source_dense"
            raw_mask_root = carnator_root / "outer_block_masks"
            capture = SharedAttentionCapture(
                raw_mask_root,
                source_uuid=row.uuid,
                config=self.cache_config,
                cfg_branch_count=cfg_count,
                max_pending_writes=self.max_pending_writes,
            )
            outer_state = OuterSparseState(cfg_branch_count=cfg_count, start_completed_step=11)
            with OuterSparseCaptureHooks(
                self.runtime.pipe.model,
                self.runtime.wan["model_module"],
                outer_state,
                capture,
                list(self.cache_config["attention_layers"]),
                execution_dtype=(torch.float16 if self.cache_config["attention_dtype"] == "fp16" else torch.bfloat16),
            ):
                stage = self.native.run_native_t2v_stage(
                    self.runtime,
                    self.args,
                    label="shared_source",
                    prompt=row.prompt,
                    out_video=staging / "source.mp4",
                    cache_dir=carnator_root,
                    capture_masks=True,
                    save_trajectory=True,
                    trajectory_start_step=0,
                    outer_sparse_state=outer_state,
                    mask_output_dir=carnator_root / "probe_masks",
                    decode_video=True,
                )
            attention = capture.close()
            trajectories = {
                int(item["step_index"]): Path(item["path"])
                for item in stage["trajectory"]
                if item.get("kind") == "post_scheduler"
            }
            required_trajectory_steps = list(range(self.args.steps))
            if sorted(trajectories) != required_trajectory_steps:
                raise RuntimeError(
                    "native source stage did not produce the complete Carnator/Chorus trajectory: "
                    f"expected {required_trajectory_steps[0]}..{required_trajectory_steps[-1]}"
                )
            self_bundle = self.native.build_native_self_contrast_mask_bundle(
                self.runtime,
                self.args,
                source_trajectory=trajectories,
                coarse_mask_dir=carnator_root / "probe_masks",
                self_mask_dir=carnator_root / "self_contrast_masks",
            )
            self.native.build_native_fusion_mask_bundle(
                self.runtime,
                self.args,
                source_trajectory=trajectories,
                coarse_mask_dir=carnator_root / "probe_masks",
                self_bundle=self_bundle,
                self_mask_dir=carnator_root / "self_contrast_masks",
                fusion_mask_dir=carnator_root / "fusion_masks",
            )
            # Stage paths are absolute while the entry is in .staging. Rewrite
            # them to their final location before atomically publishing it.
            native_manifest_path = carnator_root / "manifest.json"
            native_manifest = json.loads(native_manifest_path.read_text(encoding="utf-8"))
            final_carnator_root = final_dir / "carnator" / "cache" / "source_dense"
            native_manifest["cache_dir"] = str(final_carnator_root)
            native_manifest["output"] = str(final_dir / "source.mp4")
            for record in native_manifest.get("trajectory", []):
                record["path"] = str(final_carnator_root / Path(str(record["path"])).name)
            write_json(native_manifest_path, native_manifest)

            source_video_path = staging / "source.mp4"
            trajectory_records: list[dict[str, Any]] = []
            for step_index in required_trajectory_steps:
                record = _relative_artifact(trajectories[step_index], staging)
                record["step_index"] = step_index
                record["completed_step"] = step_index + 1
                trajectory_records.append(record)
            source_manifest = _relative_artifact(native_manifest_path, staging)
            source_initial = _relative_artifact(carnator_root / "latents_initial.pt", staging)
            raw_mask_metadata = _relative_artifact(raw_mask_root / "metadata.json", staging)
            source_video = _relative_artifact(source_video_path, staging)
            fusion_mask_files = [
                _relative_artifact(carnator_root / "fusion_masks" / filename, staging)
                for filename in ("masks.json", "masks.pt")
            ]
            raw_capture_seconds = float(attention["capture_seconds"])
            source_end_to_end = float(stage["timing"]["end_to_end_seconds"])
            source_end_to_end_without_sync_cache = max(
                0.0,
                source_end_to_end - float(stage["timing"].get("cache_write_seconds", 0.0)) - raw_capture_seconds,
            )
            manifest = {
                "format": SCHEMA_VERSION,
                "uuid": row.uuid,
                "request_index": row.request_index,
                "prompt": row.prompt,
                "created_at": utc_now(),
                "config": self.cache_config,
                "config_fingerprint": self.fingerprint,
                "source": {
                    "prompt": row.prompt,
                    "cache_dir": str(final_carnator_root),
                    "seed": self.args.seed,
                    "steps": self.args.steps,
                    "reuse_start_step": self.args.reuse_start_step,
                },
                "artifacts": {
                    "source_video": source_video,
                    "carnator": {
                        "source_manifest": source_manifest,
                        "initial_latent": source_initial,
                        "trajectory": trajectory_records,
                        "fusion_mask_files": fusion_mask_files,
                        "raw_block_mask_metadata": raw_mask_metadata,
                    },
                },
                "consumers": {
                    "nirvana_vid": {
                        "completed_step": 10,
                        "reuse_start_step": self.args.reuse_start_step,
                        "latent": str(final_carnator_root / "latents_step_009.pt"),
                    },
                    "chorus": {
                        "completed_steps": [10, 40],
                        "source_cache": str(final_carnator_root),
                        "source_video": str(final_dir / source_video["path"]),
                    },
                    "chorus2": {
                        "source_latent_cache": str(final_carnator_root),
                        "source_mask_cache": str(final_carnator_root / "outer_block_masks"),
                        "normal_sparse_raw_layers": sorted(CHORUS2_RAW_MIDDLE_LAYERS),
                    },
                    "carnator": {
                        "source_run_dir": str(final_dir / "carnator"),
                        "source_cache": str(final_carnator_root),
                        "completed_step": 10,
                        "latent": str(final_carnator_root / "latents_step_009.pt"),
                        "fusion_mask_dir": str(final_carnator_root / "fusion_masks"),
                        "raw_block_mask_cache": str(final_carnator_root / "outer_block_masks"),
                        "middle_mask_policy": "target_only_intersect_raw_mask_with_Medit_Msee",
                    },
                },
                "timing": {
                    "latency_policy": "target end-to-end latency excludes all source cache preparation",
                    "source_dense_end_to_end_seconds_excluding_sync_cache": source_end_to_end_without_sync_cache,
                    "cache_preparation": {
                        "end_to_end_seconds": time.perf_counter() - started,
                        "source_dense_raw_end_to_end_seconds": source_end_to_end,
                        "trajectory_write_seconds": stage["timing"].get("cache_write_seconds", 0.0),
                        "raw_mask_capture_seconds": raw_capture_seconds,
                        "raw_mask_async_write_seconds": attention["write_seconds"],
                    },
                },
            }
            write_json(staging / "manifest.json", manifest)
            write_json(staging / "status.json", {"status": "completed", "completed_at": utc_now()})
            if not entry_is_complete(staging, expected_uuid=row.uuid, expected_config_fingerprint=self.fingerprint):
                raise RuntimeError("staged shared cache entry failed checksum validation")
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(final_dir)
            return {
                "status": "completed",
                "uuid": row.uuid,
                "request_index": row.request_index,
                "bytes": entry_bytes(final_dir),
                "cache_preparation_seconds": manifest["timing"]["cache_preparation"]["end_to_end_seconds"],
            }
        except BaseException:
            if capture is not None and not capture.closed:
                try:
                    capture.close()
                except BaseException:
                    pass
            shutil.rmtree(staging, ignore_errors=True)
            raise


def worker_main(tasks: Any, results: Any, worker_config: dict[str, Any]) -> None:
    gpu = int(worker_config["physical_gpu"])
    log_path = Path(worker_config["output"]) / "workers" / f"gpu-{gpu}.jsonl"
    worker: NativeSharedCacheWorker | None = None
    try:
        worker = NativeSharedCacheWorker(worker_config)
        append_jsonl(log_path, {"event": "worker_ready", "gpu": gpu, "at": utc_now()})
        while True:
            payload = tasks.get()
            if payload is None:
                break
            row = PromptRow(**payload)
            try:
                result = worker.prepare(row)
            except BaseException as exc:
                result = {
                    "status": "failed",
                    "uuid": row.uuid,
                    "request_index": row.request_index,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            result["gpu"] = gpu
            result["at"] = utc_now()
            append_jsonl(log_path, result)
            results.put(result)
            if result["status"] == "failed":
                return
    except BaseException as exc:
        failure = {"status": "worker_failed", "gpu": gpu, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        append_jsonl(log_path, failure)
        results.put(failure)
    finally:
        if worker is not None:
            worker.close()


def build_index(root: Path, rows: list[PromptRow], fingerprint: str) -> int:
    output = root / "index.jsonl"
    temporary = output.with_suffix(".tmp")
    completed = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            entry = root / "entries" / row.uuid
            if not entry_is_complete(entry, expected_uuid=row.uuid, expected_config_fingerprint=fingerprint):
                continue
            manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
            stream.write(json.dumps({
                "request_index": row.request_index,
                "uuid": row.uuid,
                "prompt": row.prompt,
                "entry": str(entry.relative_to(root)),
                "config_fingerprint": fingerprint,
                "consumers": manifest["consumers"],
            }, ensure_ascii=False, sort_keys=True) + "\n")
            completed += 1
    temporary.replace(output)
    return completed


def write_run_manifest(root: Path, payload: dict[str, Any]) -> None:
    write_json(root / "run_manifest.json", payload)


def load_space_estimate(root: Path, fingerprint: str, input_sha256: str) -> dict[str, Any]:
    path = root / "space_estimate.json"
    if not path.is_file():
        raise RuntimeError("full run requires a compatible --estimate-only preflight in the same output directory")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_fingerprint") != fingerprint or payload.get("input_sha256") != input_sha256:
        raise RuntimeError("space estimate input/config fingerprint does not match this full run")
    required = int(payload.get("required_free_bytes_with_20_percent_headroom", 0))
    available = shutil.disk_usage(root).free
    if required < 1 or available < required:
        raise RuntimeError(f"insufficient free space: available={available}, required={required}")
    return payload


def estimate_space(args: argparse.Namespace, selected_count: int, available_bytes: int) -> dict[str, Any]:
    """Return a deterministic, no-GPU capacity estimate for the packed layout."""

    temporal_tokens = (int(args.frames) - 1) // 4 + 1
    spatial_tokens = math.ceil(int(args.height) / 32) * math.ceil(int(args.width) / 32)
    token_count = temporal_tokens * spatial_tokens
    q_blocks = math.ceil(token_count / int(args.q_block_size))
    k_blocks = math.ceil(token_count / int(args.k_block_size))
    mask_bytes = math.ceil(24 * q_blocks * k_blocks / 8)
    cfg_branches = 1 if float(args.guidance_scale) == 1.0 else 2
    cached_steps = int(args.steps) - 10
    raw_mask_bytes = mask_bytes * cached_steps * cfg_branches * len(ALL_WAN_LAYERS)
    trajectory_bytes = ESTIMATED_LATENT_FILE_BYTES * (int(args.steps) + 1)
    entry_bytes_estimate = (
        raw_mask_bytes
        + trajectory_bytes
        + ESTIMATED_SOURCE_VIDEO_BYTES
        + ESTIMATED_CARNATOR_MASK_BUNDLE_BYTES
        + ESTIMATED_METADATA_BYTES_PER_ENTRY
    )
    total_bytes = entry_bytes_estimate * selected_count
    required_with_headroom = math.ceil(total_bytes * 1.2)
    return {
        "format": SCHEMA_VERSION,
        "method": "analytical_no_gpu",
        "selected_requests": selected_count,
        "assumptions": {
            "token_count": token_count,
            "attention_heads": 24,
            "q_blocks": q_blocks,
            "k_blocks": k_blocks,
            "cfg_branches": cfg_branches,
            "cached_mask_steps": cached_steps,
            "chorus2_reuses_carnator_raw_masks": True,
            "trajectory_latents": int(args.steps) + 1,
            "latent_file_bytes": ESTIMATED_LATENT_FILE_BYTES,
            "source_video_bytes": ESTIMATED_SOURCE_VIDEO_BYTES,
            "carnator_mask_bundle_bytes": ESTIMATED_CARNATOR_MASK_BUNDLE_BYTES,
            "metadata_bytes_per_entry": ESTIMATED_METADATA_BYTES_PER_ENTRY,
        },
        "per_entry": {
            "carnator_raw_block_masks": raw_mask_bytes,
            "carnator_trajectory": trajectory_bytes,
            "source_video": ESTIMATED_SOURCE_VIDEO_BYTES,
            "carnator_fusion_probe_masks": ESTIMATED_CARNATOR_MASK_BUNDLE_BYTES,
            "metadata": ESTIMATED_METADATA_BYTES_PER_ENTRY,
            "total": entry_bytes_estimate,
        },
        "estimated_total_bytes": total_bytes,
        "required_free_bytes_with_20_percent_headroom": required_with_headroom,
        "available_bytes": available_bytes,
        "fits_without_headroom": available_bytes >= total_bytes,
        "fits_with_20_percent_headroom": available_bytes >= required_with_headroom,
    }


def run_workers(root: Path, rows: list[PromptRow], *, gpus: list[int], worker_config: dict[str, Any]) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    tasks = context.Queue()
    results = context.Queue()
    for row in rows:
        tasks.put(asdict(row))
    for _gpu in gpus:
        tasks.put(None)
    workers = [
        context.Process(
            target=worker_main,
            args=(tasks, results, {**worker_config, "physical_gpu": gpu}),
            name=f"shared-cache-gpu-{gpu}",
        )
        for gpu in gpus
    ]
    for worker in workers:
        worker.start()
    received: list[dict[str, Any]] = []
    try:
        while len(received) < len(rows):
            try:
                result = results.get(timeout=1.0)
            except queue.Empty:
                if not any(worker.is_alive() for worker in workers):
                    raise RuntimeError("all cache workers exited before all requests returned a result")
                continue
            if result.get("status") == "worker_failed":
                raise RuntimeError(f"GPU {result['gpu']} failed to initialize: {result['error']}")
            received.append(result)
            if result.get("status") == "failed":
                return received
    finally:
        for worker in workers:
            worker.join(timeout=10)
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
    return received


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare shared source cache for VidProM cross-request baselines.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--gpus", default="3,4,5,6,7")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--estimate-only", action="store_true", help="Write an analytical no-GPU capacity estimate and exit.")
    parser.add_argument("--pilot", action="store_true", help="Run only one source request and record its realized size.")
    parser.add_argument("--confirm-full-run", action="store_true", help="Required after a compatible no-GPU space estimate for more than one request.")
    parser.add_argument("--max-pending-mask-writes", type=int, default=4)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--scheduler", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--solver-order", type=int, default=2)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--model-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--attention-layers",
        default="0-29",
        help="Must remain all 30 Wan blocks: Carnator consumes raw routing and Chorus2 needs a complete sparse mask.",
    )
    parser.add_argument("--q-block-size", type=int, default=64)
    parser.add_argument("--k-block-size", type=int, default=128)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-top-k", type=float, default=0.1)
    parser.add_argument("--attention-dtype", choices=("fp16", "bf16"), default="fp16")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps != 50:
        raise ValueError("shared cache requires the complete 50-step source scheduler")
    if args.max_pending_mask_writes < 1:
        raise ValueError("--max-pending-mask-writes must be positive")
    if parse_layer_ranges(args.attention_layers) != list(ALL_WAN_LAYERS):
        raise ValueError("--attention-layers must be 0-29 so Chorus2 receives a complete 30-layer normal sparse mask")
    rows = read_prompt_rows(args.input.resolve(), args.limit)
    selected_count = len(rows)
    gpus = parse_gpus(args.gpus)
    config = cache_config(args)
    fingerprint = config_fingerprint(config)
    input_sha = sha256_file(args.input.resolve())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "format": SCHEMA_VERSION,
        "input": str(args.input.resolve()),
        "input_sha256": input_sha,
        "requested_limit": args.limit,
        "selected_requests": len(rows),
        "gpus": gpus,
        "config": config,
        "config_fingerprint": fingerprint,
        "model_tree_fingerprint": model_tree_fingerprint(args.model_dir.resolve()),
        "created_at": utc_now(),
    }
    if args.dry_run:
        write_run_manifest(root, {**run_manifest, "status": "dry_run"})
        print(json.dumps({"status": "dry_run", "selected_requests": len(rows), "output": str(root)}, ensure_ascii=False))
        return 0
    if args.estimate_only:
        estimate = estimate_space(args, selected_count, shutil.disk_usage(root).free)
        estimate["input_sha256"] = input_sha
        estimate["config_fingerprint"] = fingerprint
        estimate["created_at"] = utc_now()
        write_json(root / "space_estimate.json", estimate)
        write_run_manifest(root, {**run_manifest, "status": "estimated", "space_estimate": str(root / "space_estimate.json")})
        print(json.dumps({"status": "estimated", **estimate}, ensure_ascii=False))
        return 0
    if args.pilot:
        rows = rows[:1]
    elif len(rows) > 1:
        if not args.confirm_full_run:
            raise RuntimeError("refusing full run without --confirm-full-run; run --estimate-only first")
        load_space_estimate(root, fingerprint, input_sha)
    todo = [
        row for row in rows
        if not args.resume or not entry_is_complete(root / "entries" / row.uuid, expected_uuid=row.uuid, expected_config_fingerprint=fingerprint)
    ]
    worker_config = {
        "output": str(root),
        "cache_config": config,
        "config_fingerprint": fingerprint,
        "max_pending_writes": args.max_pending_mask_writes,
    }
    results = run_workers(root, todo, gpus=gpus, worker_config=worker_config) if todo else []
    failures = [result for result in results if result["status"] == "failed"]
    indexed = build_index(root, rows, fingerprint)
    if args.pilot and not failures and indexed == 1:
        size = entry_bytes(root / "entries" / rows[0].uuid)
        write_json(root / "pilot.json", {
            "format": SCHEMA_VERSION,
            "input_sha256": input_sha,
            "config_fingerprint": fingerprint,
            "entry_bytes": size,
            "estimated_total_bytes": size * selected_count,
            "required_free_bytes": math.ceil(size * selected_count * 1.2),
            "created_at": utc_now(),
        })
    status = "failed" if failures else ("pilot_completed" if args.pilot else "completed")
    write_run_manifest(root, {
        **run_manifest,
        "status": status,
        "finished_at": utc_now(),
        "indexed_entries": indexed,
        "results": results,
    })
    print(json.dumps({"status": status, "indexed_entries": indexed, "failures": len(failures), "output": str(root)}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
