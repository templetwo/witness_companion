"""Shared test fixtures. Keys are random per run; no key is ever written to the repo."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from companion.adapters.fake_brain import FakeBrain  # noqa: E402
from companion.adapters.fake_controller import FakeController  # noqa: E402
from companion.adapters.fake_notes import FakeNotes  # noqa: E402
from companion.adapters.fake_speech import FakeSpeechIn, FakeSpeechOut  # noqa: E402
from companion.adapters.fake_vision import FakeVision  # noqa: E402
from companion.coordinator import Coordinator  # noqa: E402
from companion.guard import CommandGuard  # noqa: E402
from companion.persist import Journal  # noqa: E402

ISSUER = "jetson-coordinator"


class Clock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="wc-test-")
        self.key = os.urandom(32)
        self.clock = Clock()

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def make_guard(self, armed: bool = False) -> CommandGuard:
        g = CommandGuard({ISSUER: self.key}, journal=Journal(self.path("guard.jsonl")),
                         clock=self.clock)
        if armed:
            g.arm()
        return g

    def make_stack(self, armed: bool = False, **over):
        guard = self.make_guard(armed)
        ctrl = FakeController(guard)
        img = self.path("snap.jpg")
        with open(img, "wb") as f:
            f.write(b"not-a-real-jpeg")
        kw = dict(speech_in=FakeSpeechIn(), speech_out=FakeSpeechOut(),
                  vision=FakeVision(img, captured_at=self.clock(), clock=self.clock),
                  brain=FakeBrain(), notes=FakeNotes(self.path("notes.json")),
                  controller=ctrl, clock=self.clock)
        kw.update(over)
        coord = Coordinator(ISSUER, self.key, Journal(self.path("coord.jsonl")), **kw)
        return coord, guard, ctrl
