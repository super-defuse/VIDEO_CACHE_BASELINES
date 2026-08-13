from .cogvideox15 import CogVideoX15Adapter
from .synthetic import SyntheticAdapter
from .wan22 import Wan22Adapter


def make_adapter(model: str, config: dict):
    if config.get("dry_run"):
        return SyntheticAdapter(config)
    if model == "wan22":
        return Wan22Adapter(config)
    if model == "cogvideox15":
        return CogVideoX15Adapter(config)
    raise ValueError(f"unsupported model: {model}")


__all__ = ["CogVideoX15Adapter", "SyntheticAdapter", "Wan22Adapter", "make_adapter"]
