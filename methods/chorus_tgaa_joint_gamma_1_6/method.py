from __future__ import annotations

from methods.chorus.method import Chorus


class ChorusTGAAJointGamma16(Chorus):
    """Chorus TGAA probe with gamma_k=gamma_o=1.30."""

    name = "chorus_tgaa_joint_gamma_1_6"


def create(config):
    return ChorusTGAAJointGamma16(config)
