"""Deterministic fake brain. No model, no network.

Unlike witness_brain BrainClient's stub, it never invents an action on
failure; it returns action=None unless the text plainly asks for one.
"""
from __future__ import annotations

from typing import List

from .base import Brain, BrainReply


class FakeBrain(Brain):
    def respond(self, text: str, context: List[str]) -> BrainReply:
        t = text.lower()
        if "look left" in t:
            return BrainReply("Looking left.", "look_left")
        if "look right" in t:
            return BrainReply("Looking right.", "look_right")
        if "look ahead" in t or "look center" in t:
            return BrainReply("Looking ahead.", "look_center")
        return BrainReply("I heard: %s" % text[:80], None)

    def describe_image(self, path: str, prompt: str) -> str:
        return "A stored test image (%s)." % path.rsplit("/", 1)[-1]
