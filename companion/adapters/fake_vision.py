"""Fake camera: serves a stored, timestamped image path. Opens no device."""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

from .base import BackendUnavailable, Snapshot, Vision


class FakeVision(Vision):
    def __init__(self, image_path: str, captured_at: Optional[float] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.image_path = image_path
        self.captured_at = captured_at
        self.clock = clock

    def snapshot(self) -> Snapshot:
        if not os.path.exists(self.image_path):
            raise BackendUnavailable("no stored image at %s" % self.image_path)
        ts = self.captured_at if self.captured_at is not None else os.path.getmtime(self.image_path)
        return Snapshot(path=self.image_path, captured_at=ts)

    def is_stale(self, snap: Snapshot, max_age_s: float) -> bool:
        return (self.clock() - snap.captured_at) > max_age_s
