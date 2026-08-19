from __future__ import annotations

from methods.chorus.method import Chorus


class ChorusNoTGAA(Chorus):
    """Chorus with only the Stage-2 TGAA hook removed."""

    name = "chorus_no_tgaa"


def create(config):
    return ChorusNoTGAA(config)
