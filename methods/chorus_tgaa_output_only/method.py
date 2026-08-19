from __future__ import annotations

from methods.chorus.method import Chorus


class ChorusTGAAOutputOnly(Chorus):
    """Chorus with key amplification disabled and output amplification kept."""

    name = "chorus_tgaa_output_only"


def create(config):
    return ChorusTGAAOutputOnly(config)
