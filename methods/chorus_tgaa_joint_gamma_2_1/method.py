from __future__ import annotations

from methods.chorus.method import Chorus


class ChorusTGAAJointGamma21(Chorus):
    """Chorus TGAA probe with gamma_k=gamma_o=1.6."""

    name = "chorus_tgaa_joint_gamma_2_1"


def create(config):
    return ChorusTGAAJointGamma21(config)
