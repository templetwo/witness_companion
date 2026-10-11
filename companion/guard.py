"""Robot-side command guard.

Runs on the robot side (the Pi, later). It is the last software check before
anything reaches the vendor robot stack, and it defends itself independently
of the coordinator: it holds its own per-issuer keys and its own
last-applied-seq journal.

Rules, in order:
  1. stop is always accepted: disarmed, expired, duplicate, unauthenticated or
     otherwise malformed, as long as the object says command == "stop".
     Only an authenticated, fresh stop advances last_seq; an unauthenticated
     stop never does (a forged stop with a huge seq must not lock out every
     legitimate command after it).
  2. Strict schema (schemas.ActionRequest.from_dict): closed command set, exact
     keys, typed fields, bounded and clean text, no args on motion commands.
  3. Authentication: known issuer and HMAC-SHA256 over canonical JSON,
     compared with hmac.compare_digest.
  4. Replay: seq must exceed the issuer's last applied seq and the nonce must be
     unseen. Rejected commands never consume a seq.
  5. Time: not expired, ttl (expires_at - issued_at) within max_ttl, issued_at
     not too far in the future.
  6. Arming: motion commands are refused while disarmed. The guard starts
     DISARMED on every construction; armed is never persisted. arm() is a local
     operator call, not a command: nothing in the closed set can arm.

No shell, no eval, no raw motor values. Accepting a command only means the
guard lets it through; it proves nothing about physical motion.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set

from .persist import Journal
from .schemas import MOTION_COMMANDS, ActionRequest, MalformedCommand


def sign(action: ActionRequest, key: bytes) -> ActionRequest:
    """Set action.mac in place and return it."""
    action.mac = hmac.new(key, action.canonical_bytes(), hashlib.sha256).hexdigest()
    return action


@dataclass
class Decision:
    accepted: bool
    reason: str
    command: str = ""
    seq: int = -1
    nonce: str = ""
    authenticated: bool = False


class CommandGuard:
    def __init__(self, keys: Dict[str, bytes], journal: Optional[Journal] = None,
                 clock: Callable[[], float] = time.time, max_ttl: float = 5.0,
                 max_future_skew: float = 1.0) -> None:
        if not keys:
            raise ValueError("guard needs at least one issuer key")
        self._keys = dict(keys)
        self._journal = journal
        self._clock = clock
        self.max_ttl = max_ttl
        self.max_future_skew = max_future_skew
        self.armed = False  # always start disarmed, never restored from disk
        self._last_seq: Dict[str, int] = {}
        self._nonces: Set[str] = set()
        if journal is not None:
            for issuer in self._keys:
                self._last_seq[issuer] = journal.max_seq("applied", issuer)
            self._nonces.update(journal.nonces("applied"))

    # Local operator controls. Not reachable from the command channel.
    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def last_seq(self, issuer: str) -> int:
        return self._last_seq.get(issuer, -1)

    def _authentic(self, action: ActionRequest) -> bool:
        key = self._keys.get(action.issuer)
        if key is None or not action.mac:
            return False
        expected = hmac.new(key, action.canonical_bytes(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, action.mac)

    def _record(self, action: ActionRequest) -> None:
        self._last_seq[action.issuer] = action.seq
        self._nonces.add(action.nonce)
        if self._journal is not None:
            self._journal.append({"type": "applied", "issuer": action.issuer,
                                  "seq": action.seq, "nonce": action.nonce,
                                  "command": action.command})

    def check(self, raw: Any) -> Decision:
        if isinstance(raw, ActionRequest):
            raw = raw.to_dict()
        is_stop = isinstance(raw, dict) and raw.get("command") == "stop"
        try:
            action = ActionRequest.from_dict(raw)
        except MalformedCommand as exc:
            if is_stop:
                # Disarm on any stop; seq is not consumed.
                self.armed = False
                return Decision(True, "stop accepted (malformed: %s); seq not consumed" % exc,
                                command="stop")
            return Decision(False, "malformed: %s" % exc)

        authentic = self._authentic(action)
        now = self._clock()
        replay = (action.seq <= self.last_seq(action.issuer)) or (action.nonce in self._nonces)
        time_problem = self._time_problem(action, now)

        if action.command == "stop":
            self.armed = False
            if authentic and not replay and time_problem is None:
                self._record(action)
                return Decision(True, "stop accepted", "stop", action.seq, action.nonce, True)
            why = ("unauthenticated" if not authentic else
                   "duplicate" if replay else time_problem)
            return Decision(True, "stop accepted (%s); seq not consumed" % why, "stop",
                            action.seq, action.nonce, authentic)

        if not authentic:
            return Decision(False, "unauthorized", action.command, action.seq, action.nonce)
        if replay:
            return Decision(False, "duplicate", action.command, action.seq, action.nonce, True)
        if time_problem is not None:
            return Decision(False, time_problem, action.command, action.seq, action.nonce, True)
        if action.command in MOTION_COMMANDS and not self.armed:
            return Decision(False, "disarmed", action.command, action.seq, action.nonce, True)
        self._record(action)
        return Decision(True, "accepted", action.command, action.seq, action.nonce, True)

    def _time_problem(self, action: ActionRequest, now: float) -> Optional[str]:
        if action.expires_at <= action.issued_at:
            return "expired"
        if action.expires_at - action.issued_at > self.max_ttl:
            return "ttl too long"
        if action.issued_at > now + self.max_future_skew:
            return "issued in the future"
        if now >= action.expires_at:
            return "expired"
        return None
