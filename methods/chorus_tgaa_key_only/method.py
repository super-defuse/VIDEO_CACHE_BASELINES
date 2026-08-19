from __future__ import annotations

from methods.chorus.method import Chorus


class ChorusTGAAKeyOnly(Chorus):
    """Chorus with output amplification disabled and key amplification kept."""

    name = "chorus_tgaa_key_only"


def create(config):
    return ChorusTGAAKeyOnly(config)
