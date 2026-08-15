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
    args = parser.parse_args()

    names = [item.strip() for item in args.methods.split(",") if item.strip()]
    allowed = {"baseline", "chai", "teacache", "nirvana_vid", "carnator"}
    if not names or any(name not in allowed for name in names):
        raise SystemExit(f"--methods must be a subset of {sorted(allowed)}")
    model_config = ({"model": "synthetic", "dry_run": True, "seed": 0}
                    if args.model == "synthetic" else
                    _resolve_paths(_load_yaml(ROOT / "configs" / "models" / f"{args.model}.yaml")))
    method_config = _load_yaml(ROOT / "configs" / "methods" / "default.yaml")
    config = {**model_config, **method_config, "dry_run": args.model == "synthetic"}
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    summary = {"model": args.model, "source_prompt": args.source_prompt,
               "target_prompt": args.target_prompt, "seed": seed, "methods": {}}

    cross_request = {"chai", "nirvana_vid", "carnator"}
    for name in names:
        ensure_supported(args.model, name)
        adapter = make_adapter(args.model, config)
        method = make_method(name, config)
        run_dir = ROOT / "artifacts" / "runs" / f"{args.run_prefix}-{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        runner = Runner(adapter, method, model=adapter.model_name, output_dir=run_dir,
                        cache_root=ROOT / config["cache_root"])
        if args.reset_cache:
            runner.method.reset_cache()
        if name in cross_request:
            source = Request(0, "source-000", args.source_prompt, seed)
            target = Request(1, "target-000", args.target_prompt, seed, cache_prompt=args.source_prompt)
            requests = (source, target)
        else:
            # Dense and TeaCache are not cross-request methods.  Run only the
            # target prompt; no synthetic source row or cache hit is reported.
            requests = (Request(0, "target-000", args.target_prompt, seed),)
        results = runner.run(requests, run_dir / "results.jsonl")
        (run_dir / "config.resolved.json").write_text(json.dumps(
            {**config, "source_prompt": args.source_prompt, "target_prompt": args.target_prompt,
             "seed": seed, "cache_lookup": "target -> source_prompt"}, indent=2
        ) + "\n")
        summary["methods"][name] = {
            "run_dir": str(run_dir),
            "source_status": results[0].status if name in cross_request else None,
            "target_status": results[-1].status,
            "target_cache_match": results[-1].cache_match.kind,
            "target_latency_s": results[-1].latency_s,
            "target_executed_steps": results[-1].executed_steps,
        }
        adapter.reset_hooks()

    output = ROOT / "artifacts" / "runs" / f"{args.run_prefix}-summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
