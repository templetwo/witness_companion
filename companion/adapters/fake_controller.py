"""Validating fake Pi controller. Records what it WOULD have done. Moves nothing.

It runs every command through a CommandGuard exactly as the robot side will.
An "applied" result here proves only that the guard accepted the command; it
proves nothing about physical motion or physical stopping.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..guard import CommandGuard
from ..schemas import APPLIED, REJECTED, ActionRequest, ActionResult
from .base import Controller


class FakeController(Controller):
    def __init__(self, guard: CommandGuard) -> None:
        self.guard = guard
        self.connected = True
        self.log: List[Dict[str, Any]] = []  # everything it received, with the verdict

    def send(self, action: ActionRequest) -> ActionResult:
        if not self.connected:
            raise ConnectionError("fake link down")
        decision = self.guard.check(action)
        entry = {"command": action.command, "seq": action.seq, "args": dict(action.args),
                 "accepted": decision.accepted, "reason": decision.reason,
                 "would_do": self._describe(action) if decision.accepted else None}
        self.log.append(entry)
        return ActionResult(command=action.command, seq=action.seq, nonce=action.nonce,
                            status=APPLIED if decision.accepted else REJECTED,
                            reason=decision.reason)

    @staticmethod
    def _describe(action: ActionRequest) -> str:
        if action.command in ("say", "display"):
            return "%s %r" % (action.command, action.args.get("text", ""))
        return action.command

    def would_have_done(self) -> List[str]:
        return [e["would_do"] for e in self.log if e["accepted"]]
