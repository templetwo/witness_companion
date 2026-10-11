"""Fake compute backends for routing tests. No network, no model.

Each fake declares capabilities, counts calls, can be told to fail, and
reports provenance the way a real adapter would: only what the endpoint says.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, List, Optional

from .base import BackendUnavailable, BrainReply, Note


class FakeComputeBackend:
    def __init__(self, backend_id: str, caps: Iterable[str] = ("text",),
                 fail: Optional[BaseException] = None, where_computed: str = "unknown",
                 model: str = "", provider: str = "", cold_start: Optional[bool] = None) -> None:
        self.backend_id = backend_id
        self._caps = frozenset(caps)
        self.fail = fail
        self.calls: List[str] = []
        self._report = {"where_computed": where_computed, "model": model,
                        "provider": provider, "cold_start": cold_start}

    def capabilities(self) -> FrozenSet[str]:
        return self._caps

    def report(self) -> Dict[str, Any]:
        return dict(self._report)

    def _hit(self, name: str) -> None:
        self.calls.append(name)
        if self.fail is not None:
            raise self.fail

    def respond(self, text: str, context: List[str]) -> BrainReply:
        self._hit("respond")
        return BrainReply("[%s] %s" % (self.backend_id, text[:60]), None)

    def describe_image(self, path: str, prompt: str) -> str:
        self._hit("describe_image")
        return "[%s] image %s" % (self.backend_id, path.rsplit("/", 1)[-1])

    def transcribe(self, clip: bytes) -> str:
        self._hit("transcribe")
        return "[%s] %d bytes" % (self.backend_id, len(clip))

    def synthesize(self, text: str) -> bytes:
        self._hit("synthesize")
        return text.encode("utf-8")

    def get_notes(self, query: str) -> List[Note]:
        self._hit("get_notes")
        return []


def failing(backend_id: str, caps: Iterable[str] = ("text", "image")) -> FakeComputeBackend:
    return FakeComputeBackend(backend_id, caps, fail=BackendUnavailable("%s down" % backend_id))
