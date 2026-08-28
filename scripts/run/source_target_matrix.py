#!/usr/bin/env python3
"""Run a source/target cache experiment for the unified adapter methods.

Each method gets its own cache namespace.  The source request is written first;
the target request then uses ``cache_prompt`` to look up that source entry while
still generating with the target prompt.  Baseline remains dense and does not
write a persistent cache, so its source row is an explicit control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METHOD_RUN_NAMES = {
    "nirvana_vid": "nirvana-vid",
    "chorus_no_tgaa": "chorus-no-tgaa",
    "chorus_mbase_only": "chorus-mbase-only",
    "chorus_tgaa_joint_gamma_1_6": "chorus-tgaa-joint-gamma-1-3",
    "chorus_tgaa_joint_gamma_2_1": "chorus-tgaa-joint-gamma-1-6",
    "chorus_tgaa_joint_gamma_2_0": "chorus-tgaa-joint-gamma-2-0",
}

from adapters import make_adapter
from core import Request, Runner
from core.registry import ensure_supported, make_method


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())


def _resolve_paths(config: dict) -> dict:
    for key in ("model_path", "wan_source"):
        value = config.get(key)
        if value and not Path(value).is_absolute():
            config[key] = str((ROOT / value).resolve())
    chai_retrieval = config.get("retrieval", {}).get("chai", {})
    checkpoint = chai_retrieval.get("longclip_checkpoint")
    if checkpoint and not Path(checkpoint).is_absolute():
        chai_retrieval["longclip_checkpoint"] = str((ROOT / checkpoint).resolve())
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("wan22", "cogvideox15", "synthetic"), required=True)
    parser.add_argument("--source-prompt", required=True)
    parser.add_argument("--target-prompt", required=True)
    parser.add_argument("--methods", default="baseline,chai,teacache,nirvana_vid,carnator")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--target-only", action="store_true",
                        help="Retry only a cross-request target using its persisted source cache.")
    parser.add_argument("--chai-longclip-checkpoint", type=Path)
    parser.add_argument("--chorus-gamma-k", type=float)
    parser.add_argument("--chorus-gamma-o", type=float)
    args = parser.parse_args()

    names = [item.strip() for item in args.methods.split(",") if item.strip()]
    allowed = {"baseline", "chai", "teacache", "nirvana_vid", "carnator", "chorus",
               "chorus_no_tgaa", "chorus_mbase_only", "chorus_tgaa_joint_gamma_1_6",
               "chorus_tgaa_joint_gamma_2_1", "chorus_tgaa_joint_gamma_2_0"}
    if not names or any(name not in allowed for name in names):
        raise SystemExit(f"--methods must be a subset of {sorted(allowed)}")
    gamma_values = (args.chorus_gamma_k, args.chorus_gamma_o)
    if any(value is not None for value in gamma_values):
        if any(value is None for value in gamma_values) or "chorus" not in names:
            raise SystemExit("--chorus-gamma-k and --chorus-gamma-o require --methods chorus")
    cross_request = {"chai", "nirvana_vid", "carnator", "chorus", "chorus_no_tgaa", "chorus_mbase_only",
                     "chorus_tgaa_joint_gamma_1_6", "chorus_tgaa_joint_gamma_2_1",
                     "chorus_tgaa_joint_gamma_2_0"}
    if args.target_only and (args.reset_cache or any(name not in cross_request for name in names)):
        raise SystemExit("--target-only requires cross-request methods and cannot use --reset-cache")
    model_config = ({"model": "synthetic", "dry_run": True, "seed": 0}
                    if args.model == "synthetic" else
                    _resolve_paths(_load_yaml(ROOT / "configs" / "models" / f"{args.model}.yaml")))
    method_config = _load_yaml(ROOT / "configs" / "methods" / "default.yaml")
    config = _resolve_paths({**model_config, **method_config, "dry_run": args.model == "synthetic"})
    if args.chorus_gamma_k is not None:
        config["chorus"] = {
            **config.get("chorus", {}),
            "gamma_k": args.chorus_gamma_k,
            "gamma_o": args.chorus_gamma_o,
        }
    if args.chai_longclip_checkpoint is not None:
        config["retrieval"] = {
            **config.get("retrieval", {}),
            "chai": {
                **config.get("retrieval", {}).get("chai", {}),
                "longclip_checkpoint": str(args.chai_longclip_checkpoint.expanduser().resolve()),
            },
        }
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    run_root = ROOT / "artifacts" / "runs" / args.run_prefix
    run_root.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "source_prompt": args.source_prompt,
               "target_prompt": args.target_prompt, "seed": seed,
               "run_root": str(run_root), "methods": {}}
    summary_path = run_root / "summary.json"
    if args.target_only and summary_path.is_file():
        try:
            previous = json.loads(summary_path.read_text())
            if isinstance(previous, dict) and isinstance(previous.get("methods"), dict):
                summary["methods"] = previous["methods"]
        except (OSError, ValueError):
            pass

    for name in names:
        ensure_supported(args.model, name)
        adapter = make_adapter(args.model, config)
        method = make_method(name, config)
        run_dir = run_root / METHOD_RUN_NAMES.get(name, name)
        run_dir.mkdir(parents=True, exist_ok=True)
        runner = Runner(adapter, method, model=adapter.model_name, output_dir=run_dir,
                        cache_root=ROOT / config["cache_root"])
        if args.reset_cache:
            runner.method.reset_cache()
        if name in cross_request:
            target = Request(1, "target-000", args.target_prompt, seed, cache_prompt=args.source_prompt)
            requests = (target,) if args.target_only else (
                Request(0, "source-000", args.source_prompt, seed), target
            )
        else:
            # Dense and TeaCache are not cross-request methods.  Run only the
            # target prompt; no synthetic source row or cache hit is reported.
            requests = (Request(0, "target-000", args.target_prompt, seed),)
        results_path = run_dir / ("target-retry.jsonl" if args.target_only else "results.jsonl")
        results = runner.run(requests, results_path)
        (run_dir / "config.resolved.json").write_text(json.dumps(
            {**config, "source_prompt": args.source_prompt, "target_prompt": args.target_prompt,
             "seed": seed, "cache_lookup": "target -> source_prompt"}, indent=2
        ) + "\n")
        summary["methods"][name] = {
            "run_dir": str(run_dir),
            "source_status": results[0].status if name in cross_request and not args.target_only else None,
            "target_status": results[-1].status,
            "target_cache_match": results[-1].cache_match.kind,
            "target_latency_s": results[-1].latency_s,
            "target_executed_steps": results[-1].executed_steps,
            "source_timing": results[0].extra.get("timing") if name in cross_request and not args.target_only else None,
            "target_timing": results[-1].extra.get("timing"),
            "target_extra": results[-1].extra,
        }
        adapter.reset_hooks()

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
