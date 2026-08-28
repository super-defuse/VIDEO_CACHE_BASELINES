from __future__ import annotations

from importlib import import_module


METHOD_MODULES = {
    "baseline": "methods.baseline.method",
    "chai": "methods.chai.method",
    "teacache": "methods.teacache.method",
    "nirvana_vid": "methods.nirvana_vid.method",
    "carnator": "methods.carnator.method",
    "chorus": "methods.chorus.method",
    "chorus_no_tgaa": "methods.chorus_no_tgaa.method",
    "chorus_mbase_only": "methods.chorus_mbase_only.method",
    "chorus_tgaa_joint_gamma_1_6": "methods.chorus_tgaa_joint_gamma_1_6.method",
    "chorus_tgaa_joint_gamma_2_1": "methods.chorus_tgaa_joint_gamma_2_1.method",
    "chorus_tgaa_joint_gamma_2_0": "methods.chorus_tgaa_joint_gamma_2_0.method",
}

REAL_SUPPORTED = {
    "baseline": {"wan22", "cogvideox15"},
    "chai": {"wan22", "cogvideox15"},
    "teacache": {"wan22", "cogvideox15"},
    "nirvana_vid": {"wan22", "cogvideox15"},
    "carnator": {"cogvideox15"},
    "chorus": {"wan22"},
    "chorus_no_tgaa": {"wan22"},
    "chorus_mbase_only": {"wan22"},
    "chorus_tgaa_joint_gamma_1_6": {"wan22"},
    "chorus_tgaa_joint_gamma_2_1": {"wan22"},
    "chorus_tgaa_joint_gamma_2_0": {"wan22"},
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
