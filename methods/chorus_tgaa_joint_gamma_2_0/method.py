from __future__ import annotations

from methods.chorus.method import Chorus


class ChorusTGAAJointGamma20(Chorus):
    """Chorus TGAA probe with gamma_k=gamma_o=2.0."""

    name = "chorus_tgaa_joint_gamma_2_0"


def create(config):
    return ChorusTGAAJointGamma20(config)
