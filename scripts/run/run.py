from __future__ import annotations

import argparse
import csv
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
    chai_retrieval = config.get("retrieval", {}).get("chai", {})
    checkpoint = chai_retrieval.get("longclip_checkpoint")
    if checkpoint and not Path(checkpoint).is_absolute():
        chai_retrieval["longclip_checkpoint"] = str((ROOT / checkpoint).resolve())
    nirvana_retrieval = config.get("retrieval", {}).get("nirvana_vid", {})
    for key in ("cache_selector_profile",):
        value = nirvana_retrieval.get(key)
        if value and not Path(value).is_absolute():
            nirvana_retrieval[key] = str((ROOT / value).resolve())
    predictor = nirvana_retrieval.get("match_predictor", {})
    predictor_path = predictor.get("model_path")
    if predictor_path and not Path(predictor_path).is_absolute():
        predictor["model_path"] = str((ROOT / predictor_path).resolve())
    return config


def _requests(args, config: dict) -> list[Request]:
    prompt = args.prompt or config.get("prompt")
    if prompt:
        return [Request(0, "smoke-000", prompt, int(args.seed if args.seed is not None else config.get("seed", 0)))]
    if not args.workload:
        raise SystemExit("provide --prompt or --workload")
    with Path(args.workload).open(newline="", encoding="utf-8") as handle:
        return [Request(int(row["request_index"]), row["uuid"], row["prompt"],
                        int(args.seed if args.seed is not None else config.get("seed", 0)) + int(row["request_index"]))
                for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one video cache baseline.")
    parser.add_argument("--model", choices=("wan22", "cogvideox15", "synthetic"), required=True)
    parser.add_argument("--method", choices=("baseline", "chai", "teacache", "nirvana_vid", "carnator", "chorus", "chorus_no_tgaa", "chorus_tgaa_joint_gamma_1_6", "chorus_tgaa_joint_gamma_2_1", "chorus_tgaa_joint_gamma_2_0"), required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--workload")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset-cache", action="store_true")
    args = parser.parse_args()
    ensure_supported(args.model, args.method, dry_run=args.dry_run)
    model_config = ({"model": "synthetic", "dry_run": True, "seed": 0} if args.model == "synthetic"
                    else _load_yaml(ROOT / "configs" / "models" / f"{args.model}.yaml"))
    method_config = _load_yaml(ROOT / "configs" / "methods" / "default.yaml")
    config = _resolve_paths({**model_config, **method_config, "dry_run": args.dry_run or args.model == "synthetic"})
    adapter = make_adapter(args.model, config)
    method = make_method(args.method, config)
    output_dir = ROOT / "artifacts" / "runs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(adapter, method, model=adapter.model_name, output_dir=output_dir,
                    cache_root=ROOT / config["cache_root"])
    if args.reset_cache:
        runner.method.reset_cache()
    results = runner.run(_requests(args, config), output_dir / "results.jsonl")
    (output_dir / "config.resolved.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"run": str(output_dir), "requests": len(results),
                      "ok": sum(r.status == "ok" for r in results)}, indent=2))


if __name__ == "__main__":
    main()
