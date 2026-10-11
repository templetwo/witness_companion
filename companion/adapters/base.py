"""Backend interfaces the coordinator talks to, plus a Null backend for each.

A Null backend raises BackendUnavailable on every call; the coordinator turns
that into a clear "unavailable" response. Passing None for a backend has the
same effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..schemas import ActionRequest, ActionResult


class BackendUnavailable(RuntimeError):
    """The backend is not present or not working. `what` names the capability."""

    def __init__(self, message: str, what: str = "") -> None:
        super().__init__(message)
        self.what = what or (message.split(" ")[0] if message else "backend")


@dataclass
class Snapshot:
    path: str
    captured_at: float  # epoch seconds


@dataclass
class BrainReply:
    speech: str
    action: Optional[str] = None  # free model text; the coordinator whitelists it


@dataclass
class Note:
    id: str
    text: str
    tags: List[str]


class SpeechIn:
    def listen(self) -> str:
        """Push-to-talk: capture one deliberate utterance and return its text."""
        raise NotImplementedError


class SpeechOut:
    def say(self, text: str) -> None:
        raise NotImplementedError


class Vision:
    def snapshot(self) -> Snapshot:
        """Return the newest stored, timestamped image. On request only."""
        raise NotImplementedError


class Brain:
    def respond(self, text: str, context: List[str]) -> BrainReply:
        raise NotImplementedError

    def describe_image(self, path: str, prompt: str) -> str:
        raise NotImplementedError


class Notes:
    def get(self, query: str) -> List[Note]:
        """Approved notes only."""
        raise NotImplementedError


class Controller:
    def send(self, action: ActionRequest) -> ActionResult:
        """Deliver to the robot side. Raises ConnectionError on link loss."""
        raise NotImplementedError


class _Null:
    name = "backend"

    def _unavailable(self, *_a, **_k):
        raise BackendUnavailable("%s unavailable (null backend)" % self.name)


class NullSpeechIn(_Null, SpeechIn):
    name = "speech_in"
    listen = _Null._unavailable


class NullSpeechOut(_Null, SpeechOut):
    name = "speech_out"
    say = _Null._unavailable


class NullVision(_Null, Vision):
    name = "vision"
    snapshot = _Null._unavailable


class NullBrain(_Null, Brain):
    name = "brain"
    respond = _Null._unavailable
    describe_image = _Null._unavailable


class NullNotes(_Null, Notes):
    name = "notes"
    get = _Null._unavailable


class NullController(_Null, Controller):
    name = "controller"
    send = _Null._unavailable
