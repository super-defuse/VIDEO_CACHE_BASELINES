from __future__ import annotations

from importlib import import_module


METHOD_MODULES = {
    "baseline": "methods.baseline.method",
    "chai": "methods.chai.method",
    "teacache": "methods.teacache.method",
    "nirvana_vid": "methods.nirvana_vid.method",
    "carnator": "methods.carnator.method",
}

REAL_SUPPORTED = {
    "baseline": {"wan22", "cogvideox15"},
    "chai": {"wan22", "cogvideox15"},
    "teacache": {"wan22", "cogvideox15"},
    "nirvana_vid": {"wan22", "cogvideox15"},
    "carnator": {"cogvideox15"},
}


def make_method(name: str, config: dict):
    try:
        module = import_module(METHOD_MODULES[name])
    except KeyError as exc:
        raise ValueError(f"unsupported method: {name}") from exc
    return module.create(config)


def ensure_supported(model: str, method: str, *, dry_run: bool = False) -> None:
    if dry_run or model == "synthetic":
        return
    if model not in REAL_SUPPORTED.get(method, set()):
        raise SystemExit(f"N/A: {method} is not integrated for {model}; see configs/methods/matrix.yaml")
