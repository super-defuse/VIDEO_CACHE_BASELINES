"""CARNATOR T2V C0/B0--B5 baseline components."""

__all__ = ["BASELINES", "BaselineSpec", "Profile", "CarnatorRetriever"]


def __getattr__(name):
    if name in {"BASELINES", "BaselineSpec", "Profile"}:
        from .baselines import BASELINES, BaselineSpec, Profile

        return {"BASELINES": BASELINES, "BaselineSpec": BaselineSpec, "Profile": Profile}[name]
    if name == "CarnatorRetriever":
        from .retrieval import CarnatorRetriever

        return CarnatorRetriever
    raise AttributeError(name)
