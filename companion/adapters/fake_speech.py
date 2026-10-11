"""Fake push-to-talk input and fake speaker. No audio device is touched."""
from __future__ import annotations

from typing import Iterable, List

from .base import BackendUnavailable, SpeechIn, SpeechOut


class FakeSpeechIn(SpeechIn):
    def __init__(self, utterances: Iterable[str] = ()) -> None:
        self._queue: List[str] = list(utterances)

    def push(self, text: str) -> None:
        self._queue.append(text)

    def listen(self) -> str:
        if not self._queue:
            raise BackendUnavailable("no utterance queued")
        return self._queue.pop(0)


class FakeSpeechOut(SpeechOut):
    def __init__(self) -> None:
        self.spoken: List[str] = []

    def say(self, text: str) -> None:
        self.spoken.append(text)
