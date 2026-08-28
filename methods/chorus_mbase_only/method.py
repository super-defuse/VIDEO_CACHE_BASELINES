from __future__ import annotations

from methods.chorus.method import Chorus


class ChorusMbaseOnly(Chorus):
    """Chorus ablation that updates the tight Mbase region in Stage 2."""

    name = "chorus_mbase_only"


def create(config):
    return ChorusMbaseOnly(config)
