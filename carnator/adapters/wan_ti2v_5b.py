#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import shutil
import sys
import time
import types
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CARNATOR_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WAN_ROOT = CARNATOR_ROOT / "external" / "wan2.2"
DEFAULT_MODEL_DIR = Path(os.environ.get("CARNATOR_TI2V_5B_MODEL_DIR", str(CARNATOR_ROOT / "models" / "Wan2.2-TI2V-5B")))
DEFAULT_ARTIFACT_ROOT = CARNATOR_ROOT / "artifacts" / "ti2v_5b"
DEFAULT_RESULTS_ROOT = CARNATOR_ROOT / "results" / "ti2v_5b"

SOURCE_PROMPT = "a dog is sleeping on a grassland, focus on the dog"
TARGET_PROMPT = "a cat is sleeping on a grassland, focus on the cat"
EXPECTED_TOKEN_GRID = [31, 22, 40]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CARNATOR latent reuse on the native Wan2.2-TI2V-5B checkpoint."
    )
    parser.add_argument("--runtime-python", default=os.environ.get("CARNATOR_PYTHON"))
    parser.add_argument("--no-env-setup", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Write environment/model status without loading Wan.")
    parser.add_argument("--dry-run", action="store_true", help="Write planned paths without loading Wan.")

    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default=os.environ.get("CARNATOR_DEVICE", "cuda:0"))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", default=os.environ.get("CARNATOR_RUN_ID", "latest"))
    parser.add_argument("--clean", action="store_true")

    parser.add_argument("--source-prompt", default=SOURCE_PROMPT)
    parser.add_argument("--target-prompt", default=TARGET_PROMPT)
    parser.add_argument("--source-positive", default="dog")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--solver-order", type=int, default=2)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--offload-model", action="store_true")
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--model-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--convert-model-dtype",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert DiT weights to --model-dtype. Enabled by default to match Chorus2.",
    )
    parser.add_argument("--disable-progress", action="store_true", default=True)

    parser.add_argument("--reuse-start-step", type=int, default=10)
    parser.add_argument(
        "--latent-injection-mode",
        choices=("hard", "boundary-blend"),
        default="hard",
        help="Latent reuse policy: legacy hard replacement or Mbase--Medit boundary blending.",
    )
    parser.add_argument(
        "--boundary-alpha-function",
        choices=("smoothstep", "cosine"),
        default="smoothstep",
        help="Distance-to-Mbase interpolation used by boundary-blend injection.",
    )
    parser.add_argument("--mask-layers", default="15")
    parser.add_argument("--mask-probe-step", type=int, default=5)
    parser.add_argument("--mask-self-layer", type=int, default=15)
    parser.add_argument("--mask-top-p", type=float, default=0.02)
    parser.add_argument("--mask-background-p", type=float, default=0.02)
    parser.add_argument("--mask-fusion-eps", type=float, default=0.2)
    parser.add_argument("--mask-fusion-gamma", type=float, default=1.0)
    parser.add_argument("--mask-spatial-medit-radius", type=int, default=2)
    parser.add_argument("--mask-spatial-msee-radius", type=int, default=3)
    parser.add_argument("--sparse-layers", default="10-20")
    parser.add_argument(
        "--outer-attn-backend",
        choices=("dense", "flashinfer", "sparge-sage2"),
        default="dense",
        help="Block-sparse self-attention backend for layers outside the Medit/Msee region update.",
    )
    parser.add_argument(
        "--capture-outer-block-masks",
        action="store_true",
        help="Capture source outer-attention block masks without enabling an outer sparse target backend.",
    )
    parser.add_argument("--outer-sparse-layers", default="0-9,21-29")
    parser.add_argument("--outer-sparse-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--outer-mask-q-block", type=int, default=64)
    parser.add_argument("--outer-mask-k-block", type=int, default=128)
    parser.add_argument("--outer-mask-top-p", type=float, default=0.95)
    parser.add_argument("--outer-mask-min-top-k", type=float, default=0.1)
    parser.add_argument("--outer-workspace-mib", type=int, default=256)
    parser.add_argument("--use-fp16-qk-reduction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flashinfer-plan-cache", choices=("layer", "all", "none"), default="layer")
    parser.add_argument(
        "--flashinfer-wrapper", choices=("variable", "block"), default="variable",
        help="5B follows CARNATOR_new: VariableBlock is the supported default.",
    )
    parser.add_argument("--sparge-smooth-k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sparge-pv-threshold", "--sparge-pvthreshd", dest="sparge_pv_threshold", type=float, default=50.0)
    parser.add_argument("--ffn-backend", choices=("fp", "w8a8"), default="fp")
    parser.add_argument(
        "--skip-w8a8-dense-diagnostic",
        action="store_true",
        help="Do not render the optional W8A8 dense diagnostic when C0 is already supplied as the speed reference.",
    )
    parser.add_argument("--w8a8-verify-tokens", type=int, default=0)
    parser.add_argument("--source-run-dir", type=Path, help="Reuse source dense trajectories, masks and block-mask cache from this artifact run.")
    parser.add_argument("--source-only", action="store_true", help="Prepare only the shared source dense trajectory and masks.")
    parser.add_argument(
        "--target-dense-only",
        action="store_true",
        help="Run the unmodified C0 target trajectory only; no source cache or latent reuse is created.",
    )
    parser.add_argument("--target-dense-manifest", type=Path, help="Use this FP target-dense manifest as the timing reference.")
    return parser


def runtime_python(args: argparse.Namespace) -> Path:
    if args.runtime_python:
        return Path(args.runtime_python).expanduser()
    return Path(sys.executable)


def ensure_runtime_environment(args: argparse.Namespace) -> None:
    if args.no_env_setup or os.environ.get("CARNATOR_ENV_READY") == "1":
        return
    target = runtime_python(args)
    if Path(sys.executable).resolve() == target.resolve():
        os.environ["CARNATOR_ENV_READY"] = "1"
        return
    if not target.exists():
        raise SystemExit(f"Missing runtime Python: {target}")
    env = os.environ.copy()
    env["CARNATOR_ENV_READY"] = "1"
    env["PYTHONPATH"] = f"{CARNATOR_ROOT}:{DEFAULT_WAN_ROOT}:{env.get('PYTHONPATH', '')}"
    os.execvpe(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def ensure_import_paths(args: argparse.Namespace) -> None:
    for path in (CARNATOR_ROOT, DEFAULT_WAN_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_ranges(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = item.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(item))
    return sorted(set(out))


def outer_mask_config(args: argparse.Namespace):
    from carnator.outer_sparse import BlockMaskConfig

    return BlockMaskConfig(
        q_block_size=args.outer_mask_q_block,
        k_block_size=args.outer_mask_k_block,
        top_p=args.outer_mask_top_p,
        min_top_k=args.outer_mask_min_top_k,
    )


def outer_sparse_execution_dtype(args: argparse.Namespace):
    import torch

    return torch.float16 if args.outer_sparse_dtype == "fp16" else torch.bfloat16


def model_execution_dtype(args: argparse.Namespace):
    import torch

    return torch.float16 if args.model_dtype == "fp16" else torch.bfloat16


def validate_experiment_config(args: argparse.Namespace) -> None:
    region_layers = parse_int_ranges(args.sparse_layers)
    outer_layers = parse_int_ranges(args.outer_sparse_layers)
    invalid = [layer for layer in region_layers + outer_layers if layer < 0 or layer > 29]
    if invalid:
        raise ValueError(f"Wan2.2-TI2V-5B layer indices must be in [0, 29], got {sorted(set(invalid))}")
    overlap = sorted(set(region_layers) & set(outer_layers))
    if overlap:
        raise ValueError(f"region-update and outer sparse layers must not overlap: {overlap}")
    outer_mask_config(args)
    if args.source_only and args.source_run_dir:
        raise ValueError("--source-only cannot be combined with --source-run-dir")
    if args.source_only and args.target_dense_manifest:
        raise ValueError("--source-only cannot be combined with --target-dense-manifest")
    if args.target_dense_only and (args.source_only or args.source_run_dir or args.target_dense_manifest):
        raise ValueError("--target-dense-only cannot be combined with source/reuse options")
    if args.capture_outer_block_masks and args.source_run_dir:
        raise ValueError("--capture-outer-block-masks cannot be combined with --source-run-dir")


def task_mode(args: argparse.Namespace) -> str:
    return "T2V"


def fmt_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f} s"


def fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def fmt_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100.0:.3f}%"


def guidance_mode(guidance_scale: float) -> str:
    return "no-CFG" if guidance_scale == 1.0 else "CFG"


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def bootstrap_wan(wan_root: Path = DEFAULT_WAN_ROOT):
    wan_root = wan_root.resolve()
    package_dir = wan_root / "wan"
    if not package_dir.exists():
        raise FileNotFoundError(f"native Wan package not found: {package_dir}")
    if str(wan_root) not in sys.path:
        sys.path.insert(0, str(wan_root))
    if "wan" not in sys.modules:
        package = types.ModuleType("wan")
        package.__path__ = [str(package_dir)]
        package.__file__ = str(package_dir / "__init__.py")
        sys.modules["wan"] = package

    import wan.configs as configs
    import wan.modules.model as model_module
    from wan.textimage2video import WanTI2V
    from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from wan.utils.utils import masks_like, save_video

    return {
        "configs": configs,
        "model_module": model_module,
        "WanTI2V": WanTI2V,
        "FlowDPMSolverMultistepScheduler": FlowDPMSolverMultistepScheduler,
        "FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler,
        "get_sampling_sigmas": get_sampling_sigmas,
        "retrieve_timesteps": retrieve_timesteps,
        "masks_like": masks_like,
        "save_video": save_video,
    }


def model_layout_status(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.expanduser()
    return {
        "exists": model_dir.exists(),
        "native_config_json": (model_dir / "config.json").exists(),
        "vae_checkpoint": (model_dir / "Wan2.2_VAE.pth").exists(),
        "t5_checkpoint": (model_dir / "models_t5_umt5-xxl-enc-bf16.pth").exists(),
        "t5_tokenizer": (model_dir / "google" / "umt5-xxl").exists(),
    }


def validate_native_model(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.expanduser()
    layout = model_layout_status(model_dir)
    missing = [
        name
        for name in ("native_config_json", "vae_checkpoint", "t5_checkpoint", "t5_tokenizer")
        if not layout[name]
    ]
    if not layout["exists"]:
        raise FileNotFoundError(f"model dir does not exist: {model_dir}")
    if missing:
        raise FileNotFoundError(f"{model_dir} is not a native Wan2.2-TI2V-5B checkpoint; missing {missing}")


def env_status(args: argparse.Namespace) -> dict[str, Any]:
    status: dict[str, Any] = {
        "python": sys.executable,
        "model_dir": str(args.model_dir.expanduser()),
        "wan_root": str(DEFAULT_WAN_ROOT),
        "model_layout": model_layout_status(args.model_dir),
    }
    try:
        import torch

        status["torch"] = torch.__version__
        status["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            device = torch.device(args.device)
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            props = torch.cuda.get_device_properties(device)
            status["gpu"] = {
                "name": props.name,
                "total_memory_gib": props.total_memory / (1024**3),
            }
    except Exception as exc:
        status["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        wan = bootstrap_wan()
        attention = __import__("wan.modules.attention", fromlist=["FLASH_ATTN_2_AVAILABLE", "FLASH_ATTN_3_AVAILABLE"])
        status["native_wan_available"] = True
        status["flash_attention"] = {
            "fa2": bool(getattr(attention, "FLASH_ATTN_2_AVAILABLE", False)),
            "fa3": bool(getattr(attention, "FLASH_ATTN_3_AVAILABLE", False)),
        }
        cfg = wan["configs"].WAN_CONFIGS["ti2v-5B"]
        status["wan_config"] = {
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "patch_size": list(cfg.patch_size),
            "vae_stride": list(cfg.vae_stride),
        }
    except Exception as exc:
        status["native_wan_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from carnator.outer_sparse import dependency_status

        status["outer_attention"] = dependency_status(args.outer_attn_backend)
    except Exception as exc:
        status["outer_attention"] = {"backend": args.outer_attn_backend, "available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        from carnator.quantized_ffn import w8a8_kernel_available

        status["w8a8"] = {"available": w8a8_kernel_available(), "kernel": "torch._int_mm"}
    except Exception as exc:
        status["w8a8"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return status


def token_grid_for(latent: Any, patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        int(latent.shape[1] // patch_size[0]),
        int(latent.shape[2] // patch_size[1]),
        int(latent.shape[3] // patch_size[2]),
    )


def find_token_indices(text_encoder, prompt: str, phrase: str, text_len: int) -> list[int]:
    import re

    prompt_norm = re.sub(r"\s+", " ", prompt.strip().lower())
    phrase_norm = re.sub(r"\s+", " ", phrase.strip().lower())
    tokenizer = text_encoder.tokenizer.tokenizer
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=text_len,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded["offset_mapping"][0].tolist()
    mask = encoded["attention_mask"][0].tolist()
    if not phrase_norm:
        # The common T2V path has no explicit noun phrase.  In that case use
        # every real prompt token, while excluding special and padding tokens.
        return [
            index
            for index, ((token_start, token_end), valid) in enumerate(zip(offsets, mask))
            if valid and token_end > token_start
        ]

    start = prompt_norm.find(phrase_norm)
    if start < 0:
        raise ValueError(f"could not find attention mask phrase {phrase!r} in prompt {prompt!r}")
    end = start + len(phrase_norm)
    indices = []
    for idx, ((token_start, token_end), valid) in enumerate(zip(offsets, mask)):
        if not valid or token_end <= token_start:
            continue
        if token_start < end and token_end > start:
            indices.append(idx)
    if not indices:
        raise ValueError(f"could not map attention mask phrase {phrase!r} to tokenizer indices")
    return indices


class CudaModuleProfiler:
    def __init__(self):
        self.records: list[tuple[str, str, Any, Any]] = []
        self.handles = []

    def add_module(self, module, kind: str, name: str) -> None:
        import torch

        def pre_hook(mod, _args):
            if not torch.cuda.is_available():
                return
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            mod._carnator_profile_start = start

        def post_hook(mod, _args, _output):
            if not torch.cuda.is_available():
                return
            start = getattr(mod, "_carnator_profile_start", None)
            if start is None:
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.records.append((kind, name, start, end))
            mod._carnator_profile_start = None

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(post_hook))

    def attach(self, model) -> None:
        self.add_module(model, "denoising", "wan.model")
        for idx, block in enumerate(model.blocks):
            self.add_module(block.self_attn, "self_attn", f"blocks.{idx}.self_attn")
            self.add_module(block.cross_attn, "cross_attn", f"blocks.{idx}.cross_attn")
            self.add_module(block.ffn, "ffn", f"blocks.{idx}.ffn")

    def reset(self) -> None:
        self.records.clear()

    def summary(self) -> dict[str, dict[str, float | int]]:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stats: dict[str, dict[str, float | int]] = {
            kind: {"calls": 0, "total_ms": 0.0}
            for kind in ("self_attn", "cross_attn", "ffn", "denoising")
        }
        for kind, _name, start, end in self.records:
            bucket = stats.setdefault(kind, {"calls": 0, "total_ms": 0.0})
            bucket["calls"] = int(bucket["calls"]) + 1
            bucket["total_ms"] = float(bucket["total_ms"]) + float(start.elapsed_time(end))
        for bucket in stats.values():
            calls = int(bucket["calls"])
            bucket["avg_ms"] = float(bucket["total_ms"]) / calls if calls else 0.0
        return stats

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def profile_seconds(profile: dict[str, Any], key: str) -> float | None:
    payload = profile.get(key)
    if not payload:
        return None
    return float(payload["total_ms"]) / 1000.0


class NativeCrossAttentionProbe(AbstractContextManager):
    def __init__(self, model, layers: list[int], positive_indices: list[int], token_grid: tuple[int, int, int]):
        self.model = model
        self.layers = layers
        self.positive_indices = positive_indices
        self.token_grid = token_grid
        self.enabled = False
        self.current_step_index = -1
        self.records: list[dict[str, Any]] = []
        self.handles = []

    def __enter__(self):
        for layer_index in self.layers:
            attn = self.model.blocks[layer_index].cross_attn
            self.handles.append(attn.register_forward_pre_hook(self._make_hook(layer_index)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def _make_hook(self, layer_index: int):
        def hook(attn, inputs):
            if not self.enabled:
                return
            import torch

            x, context = inputs[0], inputs[1]
            b, n, d = x.size(0), attn.num_heads, attn.head_dim
            q = attn.norm_q(attn.q(x)).view(b, -1, n, d)
            k = attn.norm_k(attn.k(context)).view(b, -1, n, d)
            pos_idx = torch.tensor(self.positive_indices, device=k.device, dtype=torch.long)
            pos_key = k.index_select(1, pos_idx)
            score = torch.einsum("bqhd,bkhd->bhqk", q, pos_key).mul(d**-0.5).mean(dim=(0, 1, 3))
            t, h, w = self.token_grid
            self.records.append(
                {
                    "step_index": int(self.current_step_index),
                    "layer_index": int(layer_index),
                    "positive": score[: t * h * w].reshape(t, h, w).detach().float().cpu(),
                }
            )

        return hook


class NativeSelfAttentionContrastProbe(AbstractContextManager):
    def __init__(
        self,
        model,
        model_module,
        layer: int,
        noun_idx,
        background_idx,
        token_grid: tuple[int, int, int],
        chunk_size: int = 512,
    ):
        self.model = model
        self.model_module = model_module
        self.layer = int(layer)
        self.noun_idx = noun_idx
        self.background_idx = background_idx
        self.token_grid = token_grid
        self.token_count = int(math.prod(token_grid))
        self.chunk_size = int(chunk_size)
        self.records: dict[str, Any] = {}
        self.handle = None

    def __enter__(self):
        self.handle = self.model.blocks[self.layer].self_attn.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        return False

    def _hook(self, attn, inputs):
        if self.records:
            return
        import torch

        hidden_states, _seq_lens, grid_sizes, freqs = inputs
        b, s, n, d = hidden_states.shape[0], hidden_states.shape[1], attn.num_heads, attn.head_dim
        query = attn.norm_q(attn.q(hidden_states)).view(b, s, n, d)
        key = attn.norm_k(attn.k(hidden_states)).view(b, s, n, d)
        query = self.model_module.rope_apply(query, grid_sizes, freqs)[:, : self.token_count]
        key = self.model_module.rope_apply(key, grid_sizes, freqs)[:, : self.token_count]

        noun_idx = self.noun_idx.to(device=hidden_states.device)
        background_idx = self.background_idx.to(device=hidden_states.device)
        noun_key = key.index_select(1, noun_idx)
        bg_key = key.index_select(1, background_idx)
        scale = d**-0.5
        noun_max_parts = []
        bg_max_parts = []
        for start in range(0, self.token_count, self.chunk_size):
            end = min(self.token_count, start + self.chunk_size)
            q = query[:, start:end]
            pos_logits = torch.einsum("bqhd,bkhd->bhqk", q, noun_key).mul(scale)
            bg_logits = torch.einsum("bqhd,bkhd->bhqk", q, bg_key).mul(scale)
            noun_max_parts.append(pos_logits.max(dim=3).values.mean(dim=(0, 1)).detach().float().cpu())
            bg_max_parts.append(bg_logits.max(dim=3).values.mean(dim=(0, 1)).detach().float().cpu())
            del pos_logits, bg_logits

        t, h, w = self.token_grid
        noun_max = torch.cat(noun_max_parts).reshape(t, h, w)
        bg_max = torch.cat(bg_max_parts).reshape(t, h, w)
        self.records = {
            "self_noun_max": noun_max,
            "self_bg_max": bg_max,
            "self_contrast_max": noun_max - bg_max,
        }


class SparseTokenIndexCache:
    def __init__(self, mask_bundle: dict[str, Any]):
        self.mask_bundle = mask_bundle
        self.cache: dict[tuple[str, int], tuple[Any, Any]] = {}
        self.records: dict[tuple[str, int], dict[str, Any]] = {}

    def get(self, hidden_states):
        import torch

        key = (str(hidden_states.device), int(hidden_states.shape[1]))
        if key in self.cache:
            return self.cache[key]
        medit = self.mask_bundle["Medit"].to(hidden_states.device).bool().flatten()
        msee = self.mask_bundle["Msee"].to(hidden_states.device).bool().flatten()
        token_count = int(hidden_states.shape[1])
        if medit.numel() != token_count:
            raise ValueError(f"Medit tokens {medit.numel()} != hidden tokens {token_count}")
        if msee.numel() != token_count:
            raise ValueError(f"Msee tokens {msee.numel()} != hidden tokens {token_count}")
        query_idx = torch.nonzero(medit, as_tuple=False).flatten()
        context_idx = torch.nonzero(msee, as_tuple=False).flatten()
        if query_idx.numel() == 0 or context_idx.numel() == 0:
            raise ValueError("empty sparse token set for canonical attention mask")
        self.cache[key] = (query_idx, context_idx)
        self.records[key] = {
            "source_completed_step": int(self.mask_bundle["source_completed_step"]),
            "token_count": token_count,
            "query_count": int(query_idx.numel()),
            "context_count": int(context_idx.numel()),
            "query_ratio": float(query_idx.numel() / token_count),
            "context_ratio": float(context_idx.numel() / token_count),
        }
        return query_idx, context_idx

    def summary(self) -> list[dict[str, Any]]:
        return [self.records[key] for key in sorted(self.records)]


@dataclass
class NativeSparseState:
    model_module: Any
    mask_bundle: dict[str, Any]
    layers: list[int]
    index_cache: SparseTokenIndexCache = field(init=False)
    enabled: bool = False
    originals: list[tuple[Any, Any]] = field(default_factory=list)
    sparse_call_count: int = 0
    dense_fallback_count: int = 0
    sparse_forward_elapsed_seconds: float = 0.0
    calls_by_layer: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.index_cache = SparseTokenIndexCache(self.mask_bundle)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "layers": self.layers,
            "sparse_call_count": self.sparse_call_count,
            "dense_fallback_count": self.dense_fallback_count,
            "sparse_forward_elapsed_seconds": self.sparse_forward_elapsed_seconds,
            "calls_by_layer": dict(sorted(self.calls_by_layer.items(), key=lambda item: int(item[0]))),
            "index_cache": self.index_cache.summary(),
        }


def native_sparse_self_attention(attn, norm_hidden_states, query_idx, context_idx, grid_sizes, freqs, model_module):
    import torch

    q_input = norm_hidden_states.index_select(1, query_idx)
    kv_input = norm_hidden_states.index_select(1, context_idx)
    b, nq, n, d = q_input.shape[0], q_input.shape[1], attn.num_heads, attn.head_dim
    nk = kv_input.shape[1]
    q = attn.norm_q(attn.q(q_input)).view(b, nq, n, d)
    k = attn.norm_k(attn.k(kv_input)).view(b, nk, n, d)
    v = attn.v(kv_input).view(b, nk, n, d)

    # The projected Q/K dtype follows the DiT execution dtype, while hidden
    # states may be promoted to FP32 by the modulation path.
    full_q = q.new_zeros((b, norm_hidden_states.shape[1], n, d))
    full_k = k.new_zeros((b, norm_hidden_states.shape[1], n, d))
    full_q.index_copy_(1, query_idx, q)
    full_k.index_copy_(1, context_idx, k)
    q = model_module.rope_apply(full_q, grid_sizes, freqs).index_select(1, query_idx)
    k = model_module.rope_apply(full_k, grid_sizes, freqs).index_select(1, context_idx)
    x = model_module.flash_attention(q=q, k=k, v=v, window_size=attn.window_size)
    return attn.o(x.flatten(2))


def native_sparse_block_forward(block, x, e, seq_lens, grid_sizes, freqs, context, context_lens, query_idx, context_idx, model_module):
    import torch

    with torch.amp.autocast("cuda", dtype=torch.float32):
        chunks = (block.modulation.unsqueeze(0) + e).chunk(6, dim=2)
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [item.squeeze(2) for item in chunks]

    q_hidden = x.index_select(1, query_idx)
    norm_all = (block.norm1(x).float() * (1 + scale_msa) + shift_msa).type_as(x)
    y = native_sparse_self_attention(block.self_attn, norm_all, query_idx, context_idx, grid_sizes, freqs, model_module)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        q_hidden = q_hidden + y * gate_msa.index_select(1, query_idx)

    q_hidden = q_hidden + block.cross_attn(block.norm3(q_hidden), context, context_lens)
    y = block.ffn(
        block.norm2(q_hidden).float() * (1 + c_scale_msa.index_select(1, query_idx))
        + c_shift_msa.index_select(1, query_idx)
    )
    with torch.amp.autocast("cuda", dtype=torch.float32):
        q_hidden = q_hidden + y * c_gate_msa.index_select(1, query_idx)

    out = x.clone()
    out.index_copy_(1, query_idx, q_hidden)
    return out


class NativeSparseBlockHooks(AbstractContextManager):
    def __init__(self, model, state: NativeSparseState):
        self.model = model
        self.state = state

    def __enter__(self):
        for layer_idx in self.state.layers:
            block = self.model.blocks[layer_idx]
            original_forward = block.forward
            self.state.originals.append((block, original_forward))

            def make_forward(block_ref, original_forward_ref, current_layer_idx: int):
                def sparse_forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens):
                    if not self.state.enabled:
                        self.state.dense_fallback_count += 1
                        return original_forward_ref(x, e, seq_lens, grid_sizes, freqs, context, context_lens)
                    query_idx, context_idx = self.state.index_cache.get(x)
                    start = time.perf_counter()
                    out = native_sparse_block_forward(
                        block_ref,
                        x,
                        e,
                        seq_lens,
                        grid_sizes,
                        freqs,
                        context,
                        context_lens,
                        query_idx,
                        context_idx,
                        self.state.model_module,
                    )
                    self.state.sparse_call_count += 1
                    self.state.sparse_forward_elapsed_seconds += time.perf_counter() - start
                    key = str(current_layer_idx)
                    self.state.calls_by_layer[key] = self.state.calls_by_layer.get(key, 0) + 1
                    return out

                return sparse_forward

            block.forward = make_forward(block, original_forward, layer_idx)
        return self.state

    def __exit__(self, exc_type, exc, tb):
        for block, original_forward in reversed(self.state.originals):
            block.forward = original_forward
        self.state.originals.clear()
        return False


@dataclass
class NativeRuntime:
    torch: Any
    wan: dict[str, Any]
    pipe: Any
    cfg: Any
    device: Any
    profiler: CudaModuleProfiler


def load_runtime(args: argparse.Namespace) -> NativeRuntime:
    import torch

    validate_native_model(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native Wan2.2-TI2V-5B generation")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"native Wan backend requires CUDA, got {args.device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)

    wan = bootstrap_wan()
    cfg = copy.deepcopy(wan["configs"].WAN_CONFIGS["ti2v-5B"])
    cfg.param_dtype = model_execution_dtype(args)
    pipe = wan["WanTI2V"](
        config=cfg,
        checkpoint_dir=str(args.model_dir.expanduser()),
        device_id=int(device.index),
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=args.convert_model_dtype,
    )
    profiler = CudaModuleProfiler()
    profiler.attach(pipe.model)
    return NativeRuntime(torch=torch, wan=wan, pipe=pipe, cfg=cfg, device=device, profiler=profiler)


def build_scheduler(runtime: NativeRuntime, args: argparse.Namespace):
    if args.scheduler == "unipc":
        scheduler = runtime.wan["FlowUniPCMultistepScheduler"](
            num_train_timesteps=runtime.pipe.num_train_timesteps,
            solver_order=args.solver_order,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(args.steps, device=runtime.device, shift=args.flow_shift)
        return scheduler
    scheduler = runtime.wan["FlowDPMSolverMultistepScheduler"](
        num_train_timesteps=runtime.pipe.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    sigmas = runtime.wan["get_sampling_sigmas"](args.steps, args.flow_shift)
    runtime.wan["retrieve_timesteps"](scheduler, device=runtime.device, sigmas=sigmas)
    return scheduler


def encode_prompts(runtime: NativeRuntime, args: argparse.Namespace, prompt: str):
    pipe = runtime.pipe
    if args.negative_prompt == "":
        negative_prompt = pipe.sample_neg_prompt
    else:
        negative_prompt = args.negative_prompt
    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(runtime.device)
        context = pipe.text_encoder([prompt], runtime.device)
        context_null = None
        if args.guidance_scale != 1.0:
            context_null = pipe.text_encoder([negative_prompt], runtime.device)
        if args.offload_model:
            pipe.text_encoder.model.cpu()
    else:
        context = pipe.text_encoder([prompt], runtime.torch.device("cpu"))
        context = [item.to(runtime.device) for item in context]
        context_null = None
        if args.guidance_scale != 1.0:
            context_null = pipe.text_encoder([negative_prompt], runtime.torch.device("cpu"))
            context_null = [item.to(runtime.device) for item in context_null]
    return context, context_null


def prepare_initial_latent(runtime: NativeRuntime, args: argparse.Namespace):
    torch = runtime.torch
    pipe = runtime.pipe
    target_shape = (
        pipe.vae.model.z_dim,
        (args.frames - 1) // pipe.vae_stride[0] + 1,
        args.height // pipe.vae_stride[1],
        args.width // pipe.vae_stride[2],
    )
    generator = torch.Generator(device=runtime.device).manual_seed(args.seed)
    return torch.randn(target_shape, dtype=torch.float32, device=runtime.device, generator=generator), generator


def seq_len_for(runtime: NativeRuntime, latent) -> int:
    pipe = runtime.pipe
    return math.ceil(
        (latent.shape[2] * latent.shape[3])
        / (pipe.patch_size[1] * pipe.patch_size[2])
        * latent.shape[1]
        / pipe.sp_size
    ) * pipe.sp_size


def timestep_for(runtime: NativeRuntime, latent, timestep, seq_len: int):
    torch = runtime.torch
    _, mask2 = runtime.wan["masks_like"]([latent], zero=False)
    temporal_mask = mask2[0]
    timestep = torch.stack([timestep])
    temp_ts = (temporal_mask[0][:, ::2, ::2] * timestep).flatten()
    temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep])
    return temp_ts.unsqueeze(0)


def save_latent(path: Path, latent) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(latent.detach().float().cpu(), path)
    return str(path)


def save_native_video(path: Path, video, fps: int) -> None:
    """Encode a Wan video atomically and preserve the encoder failure context."""
    import imageio
    import torch
    import torchvision

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.partial{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frames = video[None].clamp(-1, 1)
        frames = torch.stack(
            [
                torchvision.utils.make_grid(frame, nrow=1, normalize=True, value_range=(-1, 1))
                for frame in frames.unbind(2)
            ],
            dim=1,
        ).permute(1, 2, 3, 0)
        frames = (frames * 255).to(torch.uint8).cpu()
        with imageio.get_writer(str(temporary), fps=fps, codec="libx264", quality=8) as writer:
            for frame in frames.numpy():
                writer.append_data(frame)
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError("encoder completed without producing a non-empty file")
        temporary.replace(path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"failed to encode video at {path}: {exc}") from exc


def load_latent(path: Path, device, dtype):
    import torch

    return torch.load(path, map_location="cpu").to(device=device, dtype=dtype)


def apply_latent_injection(
    latent,
    source_latent,
    mask_bundle: dict[str, Any],
    *,
    mode: str = "hard",
    alpha_function: str = "smoothstep",
) -> tuple[Any, float, int, dict[str, Any]]:
    import torch

    from carnator.masks import boundary_blend_latent, boundary_blend_weights, get_canonical_latent_masks

    started = time.perf_counter()
    step_masks = get_canonical_latent_masks(mask_bundle, latent.unsqueeze(0))
    source_step = int(step_masks["source_completed_step"])
    source = source_latent.to(device=latent.device, dtype=latent.dtype)
    if mode == "hard":
        mask = step_masks["latent_medit"][0]
        out = torch.where(mask, latent, source)
        stats = {
            "mode": "hard",
            "base_fraction": float(step_masks["latent_mbase"].float().mean().item()),
            "band_fraction": 0.0,
            "edit_fraction": float(step_masks["latent_medit"].float().mean().item()),
        }
    elif mode == "boundary-blend":
        cache = mask_bundle.setdefault("_boundary_alpha_cache", {})
        key = (str(latent.device), tuple(latent.shape), alpha_function)
        if key in cache:
            # Reuse the precomputed alpha/band tensors without recomputing the
            # morphological distance transform at every denoising step.
            alpha, band = cache[key]
            mbase = step_masks["latent_mbase"]
            target = latent.unsqueeze(0)
            alpha = alpha.to(device=latent.device, dtype=latent.dtype)
            band = band.to(device=latent.device, dtype=torch.bool)
            blended = target * (1.0 - alpha) + source.unsqueeze(0) * alpha
            out = torch.where(mbase, target, torch.where(band, blended, source.unsqueeze(0)))[0]
            band_alpha = alpha[band].float()
            stats = {
                "mode": "boundary-blend",
                "base_fraction": float(mbase.float().mean().item()),
                "band_fraction": float(band.float().mean().item()),
                "edit_fraction": float((mbase | band).float().mean().item()),
                "alpha_mean": float(band_alpha.mean().item()) if band_alpha.numel() else 0.0,
                "alpha_min": float(band_alpha.min().item()) if band_alpha.numel() else 0.0,
                "alpha_max": float(band_alpha.max().item()) if band_alpha.numel() else 0.0,
            }
        else:
            alpha, band = boundary_blend_weights(step_masks, function=alpha_function)
            cache[key] = (alpha.detach(), band.detach())
            out, stats = boundary_blend_latent(
                latent,
                source,
                step_masks,
                function=alpha_function,
                weights=(alpha, band),
            )
    else:
        raise ValueError(f"unknown latent injection mode: {mode}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - started, source_step, stats


def stage_paths(args: argparse.Namespace) -> dict[str, Path]:
    artifact_run = args.artifact_root.expanduser().resolve() / args.run_id
    result_run = args.results_root.expanduser().resolve() / args.run_id
    return {
        "artifact_run": artifact_run,
        "result_run": result_run,
        "source_video": artifact_run / "videos" / "source_dense.mp4",
        "target_dense_video": artifact_run / "videos" / "target_dense.mp4",
        "target_dense_w8a8_video": artifact_run / "videos" / "target_dense_w8a8.mp4",
        "target_reuse_video": artifact_run / "videos" / "target_latent_reuse.mp4",
        "source_cache": artifact_run / "cache" / "source_dense",
        "outer_mask_dir": artifact_run / "cache" / "source_dense" / "outer_block_masks",
        "target_dense_cache": artifact_run / "cache" / "target_dense",
        "target_dense_w8a8_cache": artifact_run / "cache" / "target_dense_w8a8",
        "target_reuse_cache": artifact_run / "cache" / "target_latent_reuse",
        "coarse_mask_dir": artifact_run / "cache" / "source_dense" / "probe_masks",
        "self_mask_dir": artifact_run / "cache" / "source_dense" / "self_contrast_masks",
        "mask_dir": artifact_run / "cache" / "source_dense" / "fusion_masks",
        "source_first_frame": result_run / "frames" / "source_dense_first_frame.png",
        "target_first_frame": result_run / "frames" / "target_dense_first_frame.png",
        "summary": result_run / "summary.json",
        "report": result_run / "report.md",
    }


def run_native_t2v_stage(
    runtime: NativeRuntime,
    args: argparse.Namespace,
    *,
    label: str,
    prompt: str,
    out_video: Path,
    cache_dir: Path,
    capture_masks: bool = False,
    save_trajectory: bool = False,
    resume_latent_path: Path | None = None,
    resume_start_step: int = 0,
    source_trajectory: dict[int, Path] | None = None,
    mask_bundle: dict[str, Any] | None = None,
    sparse_state: NativeSparseState | None = None,
    outer_sparse_state: Any | None = None,
    ffn_state: Any | None = None,
    mask_output_dir: Path | None = None,
) -> dict[str, Any]:
    import torch

    from carnator.masks import (
        CoarseMaskConfig,
        add_canonical_latent_masks,
        build_canonical_cross_attention_mask,
        save_mask_bundle,
    )

    stage_started = time.perf_counter()
    pipe = runtime.pipe
    model = pipe.model
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    context, context_null = encode_prompts(runtime, args, prompt)
    if resume_latent_path is None:
        latent, generator = prepare_initial_latent(runtime, args)
    else:
        latent = load_latent(resume_latent_path, runtime.device, torch.float32)
        generator = torch.Generator(device=runtime.device).manual_seed(args.seed)
    seq_len = seq_len_for(runtime, latent)
    token_grid = token_grid_for(latent, pipe.patch_size)
    if list(token_grid) != EXPECTED_TOKEN_GRID and args.height == 704 and args.width == 1280 and args.frames == 121:
        raise ValueError(f"token grid {token_grid} != expected {EXPECTED_TOKEN_GRID}")

    scheduler = build_scheduler(runtime, args)
    if resume_start_step > 0 and hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(resume_start_step)

    probe = None
    if capture_masks:
        positive_indices = find_token_indices(pipe.text_encoder, prompt, args.source_positive, pipe.config.text_len)
        probe = NativeCrossAttentionProbe(model, parse_int_list(args.mask_layers), positive_indices, token_grid)

    trajectory: dict[int, Path] = {}
    if save_trajectory:
        save_latent(cache_dir / "latents_initial.pt", latent)

    if args.offload_model or pipe.init_on_cpu:
        model.to(runtime.device)
        torch.cuda.empty_cache()

    runtime.profiler.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    denoise_start = time.perf_counter()
    injection_seconds = 0.0
    injection_calls = 0
    injection_mask_source_step: int | None = None
    injection_stats: dict[str, Any] = {}

    no_sync = getattr(model, "no_sync", nullcontext)
    probe_context = probe if probe is not None else nullcontext()
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad(), no_sync(), probe_context:
        for step_index, timestep in enumerate(scheduler.timesteps):
            if step_index < resume_start_step:
                continue
            if probe is not None:
                probe.enabled = (step_index + 1) == args.mask_probe_step
                probe.current_step_index = step_index
            if sparse_state is not None:
                sparse_state.enabled = True
            if outer_sparse_state is not None:
                outer_sparse_state.set_step(step_index + 1)

            timestep_input = timestep_for(runtime, latent, timestep, seq_len)
            noise_pred_cond = model([latent], t=timestep_input, context=context, seq_len=seq_len)[0]
            if args.guidance_scale == 1.0:
                noise_pred = noise_pred_cond
            else:
                noise_pred_uncond = model([latent], t=timestep_input, context=context_null, seq_len=seq_len)[0]
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latent = scheduler.step(
                noise_pred.unsqueeze(0),
                timestep,
                latent.unsqueeze(0),
                return_dict=False,
                generator=generator,
            )[0].squeeze(0)
            if source_trajectory is not None and mask_bundle is not None and step_index >= args.reuse_start_step:
                source_path = source_trajectory.get(step_index)
                if source_path is None:
                    raise KeyError(f"source trajectory missing step {step_index}")
                source_latent = load_latent(source_path, runtime.device, latent.dtype)
                latent, elapsed, source_step, current_injection_stats = apply_latent_injection(
                    latent,
                    source_latent,
                    mask_bundle,
                    mode=args.latent_injection_mode,
                    alpha_function=args.boundary_alpha_function,
                )
                injection_seconds += elapsed
                injection_calls += 1
                injection_mask_source_step = source_step
                if not injection_stats:
                    injection_stats = current_injection_stats
            if save_trajectory:
                trajectory[step_index] = Path(save_latent(cache_dir / f"latents_step_{step_index:03d}.pt", latent))

            if not args.disable_progress:
                print(f"[{label}] step={step_index + 1:03d}/{args.steps}", flush=True)

    torch.cuda.synchronize()
    denoising_seconds = time.perf_counter() - denoise_start
    profile = runtime.profiler.summary()

    mask_path = None
    if probe is not None:
        if not probe.records:
            raise RuntimeError("attention mask probe produced no records")
        config = CoarseMaskConfig(
            top_p=args.mask_top_p,
            spatial_medit_radius=args.mask_spatial_medit_radius,
            spatial_msee_radius=args.mask_spatial_msee_radius,
        )
        built_masks = build_canonical_cross_attention_mask(
            probe.records,
            config,
            completed_step=args.mask_probe_step,
        )
        add_canonical_latent_masks(built_masks, latent.unsqueeze(0))
        built_masks["prompt"] = prompt
        built_masks["mask_positive"] = args.source_positive
        selected_mask_output_dir = mask_output_dir or stage_paths(args)["mask_dir"]
        mask_path = save_mask_bundle(selected_mask_output_dir, built_masks, save_cpu=True)

    decode_start = time.perf_counter()
    video = pipe.vae.decode([latent])[0]
    save_native_video(out_video, video, args.fps)
    torch.cuda.synchronize()
    finalize_seconds = time.perf_counter() - decode_start
    if not out_video.exists():
        raise RuntimeError(f"native Wan save_video did not create {out_video}")

    if args.offload_model:
        model.cpu()
        torch.cuda.empty_cache()
    gc.collect()

    manifest = {
        "label": label,
        "prompt": prompt,
        "mode": "T2V",
        "condition_image": None,
        "model_dir": str(args.model_dir.expanduser()),
        "wan_root": str(DEFAULT_WAN_ROOT),
        "dtype": str(pipe.param_dtype),
        "output": str(out_video),
        "height": args.height,
        "width": args.width,
        "frames": args.frames,
        "fps": args.fps,
        "steps": args.steps,
        "scheduler": args.scheduler,
        "solver_order": args.solver_order if args.scheduler == "unipc" else None,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "latent_shape": list(latent.shape),
        "token_grid": list(token_grid),
        "cache_dir": str(cache_dir),
        "trajectory": [
            {"kind": "post_scheduler", "step_index": step, "path": str(path), "shape": list(latent.shape)}
            for step, path in sorted(trajectory.items())
        ],
        "masks": {
            "enabled": capture_masks,
            "path": mask_path,
            "mask_dir": str(mask_output_dir or stage_paths(args)["mask_dir"]) if capture_masks else None,
            "source_completed_step": args.mask_probe_step if capture_masks else None,
        },
        "resume": {
            "resume_latent_path": str(resume_latent_path) if resume_latent_path else None,
            "resume_start_step": resume_start_step,
        },
        "masked_injection": {
            "enabled": source_trajectory is not None and mask_bundle is not None,
            "mode": args.latent_injection_mode,
            "alpha_function": args.boundary_alpha_function if args.latent_injection_mode == "boundary-blend" else None,
            "formula": (
                "L_new = Mbase * L_target + Mband * ((1 - alpha) * L_target + alpha * L_source) + "
                "(1 - Medit) * L_source"
                if args.latent_injection_mode == "boundary-blend"
                else "L_new = Medit * L_target + (1 - Medit) * L_source_cache"
            ),
            "calls": injection_calls,
            "total_seconds": injection_seconds,
            "avg_ms": injection_seconds * 1000.0 / injection_calls if injection_calls else 0.0,
            "mask_source_completed_step": injection_mask_source_step,
            "mask_statistics": injection_stats,
        },
        "sparse_blocks": sparse_state.summary() if sparse_state is not None else {"enabled": False},
        "outer_sparse": outer_sparse_state.summary() if hasattr(outer_sparse_state, "summary") else {"enabled": False},
        "ffn": ffn_state.summary() if ffn_state is not None else {"enabled": False, "backend": "fp"},
        "timing": {
            "loop_denoise_seconds": denoising_seconds,
            "finalize_output_seconds": finalize_seconds,
            "end_to_end_seconds": time.perf_counter() - stage_started,
        },
        "profile": profile,
        "peak_gpu_memory_gib": {
            "allocated": torch.cuda.max_memory_allocated() / (1024**3),
            "reserved": torch.cuda.max_memory_reserved() / (1024**3),
        },
    }
    manifest_path = cache_dir / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def build_native_self_contrast_mask_bundle(
    runtime: NativeRuntime,
    args: argparse.Namespace,
    *,
    source_trajectory: dict[int, Path],
    coarse_mask_dir: Path,
    self_mask_dir: Path,
) -> dict[str, Any]:
    import torch

    from carnator.masks import (
        SelfContrastMaskConfig,
        add_canonical_latent_masks,
        build_canonical_self_contrast_mask,
        load_mask_bundle,
        save_mask_bundle,
        seed_indices_from_cross_score,
    )

    completed_step = int(args.mask_probe_step)
    source_path = source_trajectory.get(completed_step - 1)
    if source_path is None:
        raise KeyError(f"source trajectory missing completed step {completed_step}")

    coarse_bundle = load_mask_bundle(coarse_mask_dir, None)
    cross_score = coarse_bundle["score"].float()
    positive_idx, background_idx = seed_indices_from_cross_score(
        cross_score,
        top_p=args.mask_top_p,
        background_p=args.mask_background_p,
    )

    latent = load_latent(source_path, runtime.device, torch.float32)
    token_grid = token_grid_for(latent, runtime.pipe.patch_size)
    seq_len = seq_len_for(runtime, latent)
    scheduler = build_scheduler(runtime, args)
    timestep = scheduler.timesteps[completed_step - 1]
    timestep_input = timestep_for(runtime, latent, timestep, seq_len)
    context, _context_null = encode_prompts(runtime, args, args.source_prompt)

    model = runtime.pipe.model
    if args.offload_model or runtime.pipe.init_on_cpu:
        model.to(runtime.device)
        torch.cuda.empty_cache()

    no_sync = getattr(model, "no_sync", nullcontext)
    probe_context = NativeSelfAttentionContrastProbe(
        model,
        runtime.wan["model_module"],
        args.mask_self_layer,
        positive_idx,
        background_idx,
        token_grid,
    )
    with torch.amp.autocast("cuda", dtype=runtime.pipe.param_dtype), torch.no_grad(), no_sync(), probe_context as probe:
        _ = model([latent], t=timestep_input, context=context, seq_len=seq_len)[0]
        torch.cuda.synchronize()

    if not probe.records:
        raise RuntimeError("self-attention contrast probe produced no records")

    config = SelfContrastMaskConfig(
        completed_step=completed_step,
        noun=args.source_positive,
        top_p=args.mask_top_p,
        background_p=args.mask_background_p,
        spatial_medit_radius=args.mask_spatial_medit_radius,
        spatial_msee_radius=args.mask_spatial_msee_radius,
        layer=args.mask_self_layer,
    )
    bundle = build_canonical_self_contrast_mask(
        probe.records["self_contrast_max"],
        config,
        noun_seed_count=int(positive_idx.numel()),
        background_seed_count=int(background_idx.numel()),
    )
    add_canonical_latent_masks(bundle, latent.unsqueeze(0).cpu())
    bundle["prompt"] = args.source_prompt
    bundle["mask_positive"] = args.source_positive
    bundle["source_seed_mask_dir"] = str(coarse_mask_dir)
    save_mask_bundle(self_mask_dir, bundle, save_cpu=True)
    return bundle


def build_native_fusion_mask_bundle(
    runtime: NativeRuntime,
    args: argparse.Namespace,
    *,
    source_trajectory: dict[int, Path],
    coarse_mask_dir: Path,
    self_bundle: dict[str, Any],
    self_mask_dir: Path,
    fusion_mask_dir: Path,
) -> dict[str, Any]:
    import torch

    from carnator.masks import (
        FusionMaskConfig,
        add_canonical_latent_masks,
        build_canonical_fusion_mask,
        load_mask_bundle,
        save_mask_bundle,
    )

    completed_step = int(args.mask_probe_step)
    source_path = source_trajectory.get(completed_step - 1)
    if source_path is None:
        raise KeyError(f"source trajectory missing completed step {completed_step}")

    coarse_bundle = load_mask_bundle(coarse_mask_dir, None)
    config = FusionMaskConfig(
        completed_step=completed_step,
        eps=args.mask_fusion_eps,
        gamma=args.mask_fusion_gamma,
        spatial_medit_radius=args.mask_spatial_medit_radius,
        spatial_msee_radius=args.mask_spatial_msee_radius,
    )
    bundle = build_canonical_fusion_mask(
        self_bundle["score"].float(),
        coarse_bundle["score"].float(),
        config,
        original_mbase=self_bundle.get("Mbase"),
    )
    latent = load_latent(source_path, torch.device("cpu"), torch.float32)
    add_canonical_latent_masks(bundle, latent.unsqueeze(0))
    bundle["prompt"] = args.source_prompt
    bundle["mask_positive"] = args.source_positive
    bundle["self_mask_dir"] = str(self_mask_dir)
    bundle["cross_mask_dir"] = str(coarse_mask_dir)
    save_mask_bundle(fusion_mask_dir, bundle, save_cpu=True)
    return bundle


def extract_first_frame(video_path: Path, out_path: Path) -> None:
    import imageio.v3 as iio
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = iio.imread(video_path, index=0)
    Image.fromarray(frame[..., :3]).save(out_path)


def average(values: list[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def mask_area_summary(mask_dir: Path) -> dict[str, Any]:
    import torch

    meta_path = mask_dir / "masks.json"
    tensor_path = mask_dir / "masks.pt"
    if not meta_path.exists() or not tensor_path.exists():
        return {"available": False, "mask_dir": str(mask_dir)}
    meta = read_json(meta_path)
    tensors = torch.load(tensor_path, map_location="cpu")
    mask_variant = meta.get("mask_variant")
    coverage_note = "raw_coverage_area is mask-bundle Mbase area; no external GT mask is used."
    if mask_variant == "offline_cross_attention_probe":
        coverage_note = "raw_coverage_area is attention mask cross-attention seed area; no external GT mask is used."
    raw = tensors.get("raw_coverage")
    token_area = meta.get("token_area", {})
    latent_area = meta.get("latent_area", {})
    return {
        "available": True,
        "mask_dir": str(mask_dir),
        "mask_variant": mask_variant,
        "source_completed_step": meta.get("source_completed_step"),
        "token_area": token_area,
        "latent_area": latent_area,
        "raw_coverage_area": float(raw.float().mean().item()) if raw is not None else None,
        "coverage_note": coverage_note,
    }


def compact_run(manifest: dict[str, Any]) -> dict[str, Any]:
    profile = manifest.get("profile", {})
    return {
        "output": manifest.get("output"),
        "manifest_path": manifest.get("manifest_path"),
        "denoising_seconds": manifest.get("timing", {}).get("loop_denoise_seconds"),
        "end_to_end_seconds": manifest.get("timing", {}).get("end_to_end_seconds"),
        "self_attn_seconds": profile_seconds(profile, "self_attn"),
        "cross_attn_seconds": profile_seconds(profile, "cross_attn"),
        "ffn_seconds": profile_seconds(profile, "ffn"),
        "latent_shape": manifest.get("latent_shape"),
        "token_grid": manifest.get("token_grid"),
        "peak_gpu_memory_gib": manifest.get("peak_gpu_memory_gib"),
        "masked_injection": manifest.get("masked_injection"),
        "sparse_blocks": manifest.get("sparse_blocks"),
        "outer_sparse": manifest.get("outer_sparse"),
        "ffn": manifest.get("ffn"),
    }


def sparse_density_summary(run: dict[str, Any]) -> dict[str, Any]:
    sparse = run.get("sparse_blocks") or {}
    records = sparse.get("index_cache") or []
    query = [item.get("query_ratio") for item in records if item.get("query_ratio") is not None]
    context = [item.get("context_ratio") for item in records if item.get("context_ratio") is not None]
    attn_density = [
        float(item["query_ratio"]) * float(item["context_ratio"])
        for item in records
        if item.get("query_ratio") is not None and item.get("context_ratio") is not None
    ]
    return {
        "sparse_call_count": sparse.get("sparse_call_count"),
        "dense_fallback_count": sparse.get("dense_fallback_count"),
        "sparse_forward_elapsed_seconds": sparse.get("sparse_forward_elapsed_seconds"),
        "avg_medit_query_ratio": average(query),
        "avg_msee_context_ratio": average(context),
        "avg_self_attn_region_density_estimate": average(attn_density),
        "records": records,
    }


def outer_sparse_metrics(run: dict[str, Any]) -> tuple[float | None, float | None]:
    outer_sparse = run.get("outer_sparse") or {}
    if outer_sparse.get("enabled"):
        return outer_sparse.get("avg_runtime_density"), outer_sparse.get("avg_source_gt_coverage")
    return (1.0, 1.0) if run else (None, None)


def build_summary(
    args: argparse.Namespace,
    paths: dict[str, Path],
    source: dict[str, Any] | None,
    target_dense: dict[str, Any] | None,
    target_dense_w8a8: dict[str, Any] | None,
    target_reuse: dict[str, Any] | None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    runs = {}
    if source is not None:
        runs["source_dense"] = compact_run(source)
    if target_dense is not None:
        runs["target_dense"] = compact_run(target_dense)
    if target_dense_w8a8 is not None:
        runs["target_dense_w8a8"] = compact_run(target_dense_w8a8)
    if target_reuse is not None:
        runs["target_latent_reuse"] = compact_run(target_reuse)
        runs["target_latent_reuse"]["sparse_density"] = sparse_density_summary(runs["target_latent_reuse"])

    dense = runs.get("target_dense", {})
    reuse = runs.get("target_latent_reuse", {})
    # C0 is the ordinary full-dense FP target.  A W8A8 dense diagnostic is
    # useful for ablation debugging, but must not redefine the speed baseline.
    speedup_reference_key = "target_dense"
    speedup_reference = runs.get("target_dense", {})
    summary = {
        "status": status,
        "error": error,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config": {
        "model_family": "Wan2.2-TI2V-5B",
            "backend": "native Wan2.2",
            "mode": task_mode(args),
            "model_dtype": args.model_dtype,
            "convert_model_dtype": args.convert_model_dtype,
            "source_prompt": args.source_prompt,
            "target_prompt": args.target_prompt,
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "fps": args.fps,
            "steps": args.steps,
            "seed": args.seed,
            "scheduler": args.scheduler,
            "solver_order": args.solver_order,
            "flow_shift": args.flow_shift,
            "guidance_scale": args.guidance_scale,
            "effective_guidance": guidance_mode(args.guidance_scale),
            "reuse_start_step": args.reuse_start_step,
            "latent_injection_mode": args.latent_injection_mode,
            "boundary_alpha_function": args.boundary_alpha_function,
            "mask_method": "fusion",
            "mask_probe_step": args.mask_probe_step,
            "mask_self_layer": args.mask_self_layer,
            "mask_fusion_eps": args.mask_fusion_eps,
            "mask_fusion_gamma": args.mask_fusion_gamma,
            "sparse_layers": args.sparse_layers,
            "region_update_layers": args.sparse_layers,
            "outer_attn_backend": args.outer_attn_backend,
            "outer_flashinfer_wrapper": args.flashinfer_wrapper if args.outer_attn_backend == "flashinfer" else "n/a",
            "capture_outer_block_masks": args.capture_outer_block_masks,
            "outer_sparse_layers": args.outer_sparse_layers,
            "outer_sparse_dtype": args.outer_sparse_dtype,
            "use_fp16_qk_reduction": args.use_fp16_qk_reduction,
            "flashinfer_plan_cache": args.flashinfer_plan_cache,
            "sparge_smooth_k": args.sparge_smooth_k,
            "sparge_pvthreshd": args.sparge_pv_threshold,
            "outer_block_mask": {
                "q_block_size": args.outer_mask_q_block,
                "k_block_size": args.outer_mask_k_block,
                "top_p": args.outer_mask_top_p,
                "min_top_k": args.outer_mask_min_top_k,
            },
            "ffn_backend": args.ffn_backend,
            "skip_w8a8_dense_diagnostic": args.skip_w8a8_dense_diagnostic,
            "expected_token_grid": EXPECTED_TOKEN_GRID,
        },
        "environment": env_status(args),
        "paths": {key: str(value) for key, value in paths.items()},
        "runs": runs,
        "speedups": {
            "reference_run": speedup_reference_key,
            "denoising_vs_reference": safe_ratio(speedup_reference.get("denoising_seconds"), reuse.get("denoising_seconds")),
            "end_to_end_vs_reference": safe_ratio(speedup_reference.get("end_to_end_seconds"), reuse.get("end_to_end_seconds")),
        },
    }
    if paths["mask_dir"].exists():
        summary["masks"] = mask_area_summary(paths["mask_dir"])
    if paths["source_first_frame"].exists() or paths["target_first_frame"].exists():
        summary["first_frames"] = {
            "source_dense": str(paths["source_first_frame"]) if paths["source_first_frame"].exists() else None,
            "target_dense": str(paths["target_first_frame"]) if paths["target_first_frame"].exists() else None,
        }
    return summary


def write_report(
    path: Path,
    summary: dict[str, Any],
    *,
    reuse_label: str = "Target latent reuse",
    reuse_note: str | None = None,
) -> None:
    runs = summary.get("runs", {})
    source = runs.get("source_dense", {})
    dense = runs.get("target_dense", {})
    dense_w8a8 = runs.get("target_dense_w8a8", {})
    reuse = runs.get("target_latent_reuse", {})
    density = reuse.get("sparse_density", {})
    masks = summary.get("masks", {})
    frames = summary.get("first_frames", {})
    speedups = summary.get("speedups", {})

    lines = [
        "# CARNATOR Latent Reuse Report",
        "",
        f"Updated: `{summary.get('updated_at')}`",
        f"Status: `{summary.get('status')}`",
    ]
    if summary.get("error"):
        lines.append(f"Error: `{summary['error']}`")
    lines.extend(
        [
            "",
            "## Config",
            "",
            f"- Backend: `{summary['config']['backend']}`",
            f"- Model: `{summary['config']['model_family']}`",
            f"- Task: `{summary['config']['mode']}`",
            f"- Prompts: source `{summary['config']['source_prompt']}`, target `{summary['config']['target_prompt']}`",
            f"- Shape: `{summary['config']['width']}x{summary['config']['height']}`, frames `{summary['config']['frames']}`, fps `{summary['config']['fps']}`",
            f"- Sampling: `{summary['config']['steps']}` steps, `{summary['config']['scheduler']}`, solver order `{summary['config']['solver_order']}`, flow shift `{summary['config']['flow_shift']}`",
            f"- Guidance: `{summary['config']['guidance_scale']}` ({summary['config']['effective_guidance']}), region-update layers `{summary['config']['region_update_layers']}`, outer `{summary['config']['outer_attn_backend']}` ({summary['config']['outer_flashinfer_wrapper']}) on `{summary['config']['outer_sparse_layers']}`, FFN `{summary['config']['ffn_backend']}`, reuse start step `{summary['config']['reuse_start_step']}`",
            f"- latent injection: `{summary['config']['latent_injection_mode']}` (alpha `{summary['config']['boundary_alpha_function']}`)",
            f"- attention mask: `{summary['config']['mask_method']}`, completed probe step `{summary['config']['mask_probe_step']}`, self layer `{summary['config']['mask_self_layer']}`",
            "",
            "## Results",
            "",
            "| Run | Self-attn | Cross-attn | FFN | Outer sparse density | Outer mask mass coverage | Latency denoising | Speed Up denoising | Latency End2End | Speed Up End2End | Peak reserved |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    reuse_density, reuse_coverage = outer_sparse_metrics(reuse)
    speedup_reference = speedups.get("reference_run", "target_dense")
    rows = [
        ("Source dense", source, None, None, None, None),
        (
            "C0 full dense target",
            dense,
            1.0 if dense and speedup_reference == "target_dense" else None,
            1.0 if dense and speedup_reference == "target_dense" else None,
            1.0 if dense else None,
            1.0 if dense else None,
        ),
    ]
    if dense_w8a8:
        rows.append(
            (
                "Dense W8A8 diagnostic",
                dense_w8a8,
                safe_ratio(dense.get("denoising_seconds"), dense_w8a8.get("denoising_seconds")),
                safe_ratio(dense.get("end_to_end_seconds"), dense_w8a8.get("end_to_end_seconds")),
                1.0,
                1.0,
            )
        )
    rows.append(
        (
            reuse_label,
            reuse,
            speedups.get("denoising_vs_reference"),
            speedups.get("end_to_end_vs_reference"),
            reuse_density,
            reuse_coverage,
        )
    )
    for label, row, denoise_speedup, e2e_speedup, row_density, coverage in rows:
        peak = (row.get("peak_gpu_memory_gib") or {}).get("reserved")
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    fmt_seconds(row.get("self_attn_seconds")),
                    fmt_seconds(row.get("cross_attn_seconds")),
                    fmt_seconds(row.get("ffn_seconds")),
                    fmt_percent(row_density),
                    fmt_percent(coverage),
                    fmt_seconds(row.get("denoising_seconds")),
                    fmt_ratio(denoise_speedup),
                    fmt_seconds(row.get("end_to_end_seconds")),
                    fmt_ratio(e2e_speedup),
                    "n/a" if peak is None else f"{peak:.2f} GiB",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Source dense video: `{source.get('output')}`",
            f"- Source first frame: `{frames.get('source_dense')}`",
            f"- Target dense video: `{dense.get('output')}`",
            *([f"- Target dense W8A8 reference video: `{dense_w8a8.get('output')}`"] if dense_w8a8 else []),
            f"- Target first frame: `{frames.get('target_dense')}`",
            f"- Target latent reuse video: `{reuse.get('output')}`",
            f"- Summary JSON: `{summary['paths']['summary']}`",
            "",
            "## Reuse",
            "",
            f"- Sparse calls: `{density.get('sparse_call_count')}`",
            f"- Dense fallback calls: `{density.get('dense_fallback_count')}`",
            f"- Avg Medit query ratio: `{fmt_percent(density.get('avg_medit_query_ratio'))}`",
            f"- Avg Msee context ratio: `{fmt_percent(density.get('avg_msee_context_ratio'))}`",
            f"- Latent injection: `{reuse.get('masked_injection')}`",
            *([f"- {reuse_note}"] if reuse_note else []),
            f"- attention mask variant: `{masks.get('mask_variant')}`",
            f"- attention mask dir: `{masks.get('mask_dir')}`",
            f"- attention mask token area: `{masks.get('token_area')}`",
            f"- attention mask latent area: `{masks.get('latent_area')}`",
            "",
            "## Environment",
            "",
            f"- Python: `{summary.get('environment', {}).get('python')}`",
            f"- Model dir: `{summary.get('environment', {}).get('model_dir')}`",
            f"- Wan root: `{summary.get('environment', {}).get('wan_root')}`",
            f"- CUDA available: `{summary.get('environment', {}).get('cuda_available')}`",
            f"- Flash attention: `{summary.get('environment', {}).get('flash_attention')}`",
            f"- Model layout: `{summary.get('environment', {}).get('model_layout')}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_and_reports(paths: dict[str, Path], summary: dict[str, Any]) -> None:
    write_json(paths["summary"], summary)
    write_report(paths["report"], summary)


def source_assets_from_run(args: argparse.Namespace, source_run_dir: Path) -> tuple[dict[str, Any], dict[int, Path], Path, Path]:
    source_run_dir = source_run_dir.expanduser().resolve()
    source_cache = source_run_dir / "cache" / "source_dense"
    manifest_path = source_cache / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"source run has no source_dense manifest: {manifest_path}")
    source = read_json(manifest_path)
    expected_mode = task_mode(args)
    if source.get("mode", "T2V") != expected_mode:
        raise ValueError(f"source run task mismatch: expected {expected_mode}, got {source.get('mode', 'T2V')}")
    trajectory = {
        int(item["step_index"]): Path(item["path"])
        for item in source.get("trajectory", [])
        if item.get("kind") == "post_scheduler"
    }
    if not trajectory:
        raise ValueError("source run does not contain a dense latent trajectory")
    mask_dir = source_cache / "fusion_masks"
    outer_mask_dir = source_cache / "outer_block_masks"
    if not (mask_dir / "masks.pt").exists():
        raise FileNotFoundError(f"source run has no fusion masks: {mask_dir}")
    return source, trajectory, mask_dir, outer_mask_dir


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    from carnator.masks import load_mask_bundle
    from carnator.outer_sparse import (
        OuterSparseAttentionHooks,
        OuterSparseCaptureHooks,
        OuterSparseExecutor,
        OuterSparseState,
        SourceBlockMaskCache,
        cfg_branches,
    )
    from carnator.quantized_ffn import QuantizedFFNHooks

    paths = stage_paths(args)
    if args.dry_run:
        summary = build_summary(args, paths, None, None, None, None, status="dry_run")
        write_summary_and_reports(paths, summary)
        return summary
    if args.clean:
        for key in ("artifact_run", "result_run"):
            if paths[key].exists():
                shutil.rmtree(paths[key])
    paths["artifact_run"].mkdir(parents=True, exist_ok=True)
    paths["result_run"].mkdir(parents=True, exist_ok=True)
    validate_experiment_config(args)
    if args.mask_probe_step < 1 or args.mask_probe_step > args.steps:
        raise ValueError("--mask-probe-step must be in [1, steps]")

    runtime = load_runtime(args)
    try:
        if args.target_dense_only:
            target_dense = run_native_t2v_stage(
                runtime,
                args,
                label="target_dense",
                prompt=args.target_prompt,
                out_video=paths["target_dense_video"],
                cache_dir=paths["target_dense_cache"],
            )
            summary = build_summary(args, paths, None, target_dense, None, None, status="completed")
            write_summary_and_reports(paths, summary)
            return summary
        source_outer_mask_dir = paths["outer_mask_dir"]
        sparse_execution_dtype = outer_sparse_execution_dtype(args)
        if args.source_run_dir:
            source, source_trajectory, source_mask_dir, source_outer_mask_dir = source_assets_from_run(args, args.source_run_dir)
            paths["mask_dir"] = source_mask_dir
            paths["outer_mask_dir"] = source_outer_mask_dir
            source_initial_latent = Path(source["cache_dir"]) / "latents_initial.pt"
        else:
            source_outer_state = None
            if args.capture_outer_block_masks or args.outer_attn_backend != "dense":
                source_cache = SourceBlockMaskCache(
                    source_outer_mask_dir,
                    outer_mask_config(args),
                    source_run_id=args.run_id,
                    cfg_branch_count=cfg_branches(args.guidance_scale),
                    create=True,
                )
                source_outer_state = OuterSparseState(
                    cfg_branch_count=cfg_branches(args.guidance_scale),
                    start_completed_step=args.reuse_start_step + 1,
                )
                with OuterSparseCaptureHooks(
                    runtime.pipe.model,
                    runtime.wan["model_module"],
                    source_outer_state,
                    source_cache,
                    parse_int_ranges(args.outer_sparse_layers),
                    execution_dtype=sparse_execution_dtype,
                ):
                    source = run_native_t2v_stage(
                        runtime,
                        args,
                        label="source_dense",
                        prompt=args.source_prompt,
                        out_video=paths["source_video"],
                        cache_dir=paths["source_cache"],
                        capture_masks=True,
                        save_trajectory=True,
                        outer_sparse_state=source_outer_state,
                        mask_output_dir=paths["coarse_mask_dir"],
                    )
            else:
                source = run_native_t2v_stage(
                    runtime,
                    args,
                    label="source_dense",
                    prompt=args.source_prompt,
                    out_video=paths["source_video"],
                    cache_dir=paths["source_cache"],
                    capture_masks=True,
                    save_trajectory=True,
                    mask_output_dir=paths["coarse_mask_dir"],
                )
            source_trajectory = {
                int(item["step_index"]): Path(item["path"])
                for item in source.get("trajectory", [])
                if item.get("kind") == "post_scheduler"
            }
            self_bundle = build_native_self_contrast_mask_bundle(
                runtime,
                args,
                source_trajectory=source_trajectory,
                coarse_mask_dir=paths["coarse_mask_dir"],
                self_mask_dir=paths["self_mask_dir"],
            )
            build_native_fusion_mask_bundle(
                runtime,
                args,
                source_trajectory=source_trajectory,
                coarse_mask_dir=paths["coarse_mask_dir"],
                self_bundle=self_bundle,
                self_mask_dir=paths["self_mask_dir"],
                fusion_mask_dir=paths["mask_dir"],
            )
            source_mask_dir = paths["mask_dir"]
            source_initial_latent = paths["source_cache"] / "latents_initial.pt"

        if args.source_only:
            summary = build_summary(args, paths, source, None, None, None, status="source_prepared")
            write_summary_and_reports(paths, summary)
            return summary

        if args.reuse_start_step == 0:
            resume_latent_path = source_initial_latent
        else:
            resume_latent_path = source_trajectory[args.reuse_start_step - 1]

        if args.target_dense_manifest:
            target_dense = read_json(args.target_dense_manifest.expanduser())
            if target_dense.get("mode", "T2V") != task_mode(args):
                raise ValueError("target dense manifest task does not match the current run")
        else:
            target_dense = run_native_t2v_stage(
                runtime,
                args,
                label="target_dense",
                prompt=args.target_prompt,
                out_video=paths["target_dense_video"],
                cache_dir=paths["target_dense_cache"],
            )

        target_dense_w8a8 = None
        if args.ffn_backend == "w8a8" and not args.skip_w8a8_dense_diagnostic:
            quantized_hooks = QuantizedFFNHooks(runtime.pipe.model)
            with quantized_hooks as quantized_ffn:
                quantized_hooks.attach_profiler(runtime.profiler)
                if args.w8a8_verify_tokens:
                    quantized_hooks.verify(args.w8a8_verify_tokens)
                target_dense_w8a8 = run_native_t2v_stage(
                    runtime,
                    args,
                    label="target_dense_w8a8",
                    prompt=args.target_prompt,
                    out_video=paths["target_dense_w8a8_video"],
                    cache_dir=paths["target_dense_w8a8_cache"],
                    ffn_state=quantized_ffn,
                )

        mask_bundle = load_mask_bundle(source_mask_dir, None)
        sparse_state = NativeSparseState(
            model_module=runtime.wan["model_module"],
            mask_bundle=mask_bundle,
            layers=parse_int_ranges(args.sparse_layers),
        )
        outer_state = None
        outer_executor = None
        if args.outer_attn_backend != "dense":
            source_cache = SourceBlockMaskCache(source_outer_mask_dir, outer_mask_config(args))
            source_cache.expect_cfg_branches(cfg_branches(args.guidance_scale))
            outer_state = OuterSparseState(
                cfg_branch_count=cfg_branches(args.guidance_scale),
                start_completed_step=args.reuse_start_step + 1,
            )
            outer_executor = OuterSparseExecutor(
                source_cache,
                args.outer_attn_backend,
                flashinfer_wrapper=args.flashinfer_wrapper,
                workspace_mib=args.outer_workspace_mib,
                execution_dtype=sparse_execution_dtype,
                use_fp16_qk_reduction=args.use_fp16_qk_reduction,
                flashinfer_plan_cache=args.flashinfer_plan_cache,
                sparge_smooth_k=args.sparge_smooth_k,
                sparge_pv_threshold=args.sparge_pv_threshold,
            )
        with ExitStack() as stack:
            ffn_state = None
            if args.ffn_backend == "w8a8":
                ffn_hooks = QuantizedFFNHooks(runtime.pipe.model)
                ffn_state = stack.enter_context(ffn_hooks)
                ffn_hooks.attach_profiler(runtime.profiler)
                if args.w8a8_verify_tokens:
                    ffn_hooks.verify(args.w8a8_verify_tokens)
            stack.enter_context(NativeSparseBlockHooks(runtime.pipe.model, sparse_state))
            if outer_state is not None and outer_executor is not None:
                stack.enter_context(
                    OuterSparseAttentionHooks(
                        runtime.pipe.model,
                        runtime.wan["model_module"],
                        outer_state,
                        outer_executor,
                        parse_int_ranges(args.outer_sparse_layers),
                    )
                )
            target_reuse = run_native_t2v_stage(
                runtime,
                args,
                label="target_latent_reuse",
                prompt=args.target_prompt,
                out_video=paths["target_reuse_video"],
                cache_dir=paths["target_reuse_cache"],
                resume_latent_path=resume_latent_path,
                resume_start_step=args.reuse_start_step,
                source_trajectory=source_trajectory,
                mask_bundle=mask_bundle,
                sparse_state=sparse_state,
                outer_sparse_state=outer_state,
                ffn_state=ffn_state,
            )

        if Path(source["output"]).exists():
            extract_first_frame(Path(source["output"]), paths["source_first_frame"])
        if Path(target_dense["output"]).exists():
            extract_first_frame(Path(target_dense["output"]), paths["target_first_frame"])

        summary = build_summary(args, paths, source, target_dense, target_dense_w8a8, target_reuse, status="completed")
        write_summary_and_reports(paths, summary)
        return summary
    finally:
        runtime.profiler.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.model_dir = args.model_dir.expanduser()
    args.artifact_root = args.artifact_root.expanduser()
    args.results_root = args.results_root.expanduser()
    if args.source_run_dir:
        args.source_run_dir = args.source_run_dir.expanduser()
    if args.target_dense_manifest:
        args.target_dense_manifest = args.target_dense_manifest.expanduser()
    ensure_runtime_environment(args)
    ensure_import_paths(args)
    paths = stage_paths(args)

    if args.check_only:
        status = "ready"
        error = None
        try:
            validate_experiment_config(args)
            validate_native_model(args)
            bootstrap_wan()
        except Exception as exc:
            status = "blocked"
            error = f"{type(exc).__name__}: {exc}"
        summary = build_summary(args, paths, None, None, None, None, status=status, error=error)
        write_summary_and_reports(paths, summary)
        print(f"summary={paths['summary']}")
        print(f"report={paths['report']}")
        return 0 if status == "ready" else 2

    try:
        summary = run_experiment(args)
    except Exception as exc:
        summary = build_summary(args, paths, None, None, None, None, status="failed", error=f"{type(exc).__name__}: {exc}")
        write_summary_and_reports(paths, summary)
        print(f"summary={paths['summary']}")
        print(f"report={paths['report']}")
        raise

    print(f"summary={summary['paths']['summary']}")
    print(f"report={summary['paths']['report']}")
    speedups = summary["speedups"]
    print(f"speedup_reference={speedups['reference_run']}")
    print(f"denoising_speedup={speedups['denoising_vs_reference']}")
    print(f"end_to_end_speedup={speedups['end_to_end_vs_reference']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
