"""Data shapes for the companion coordinator.

Event mirrors witness_brain.schemas.event_schema.Event (id, timestamp, source,
content) without importing it: that module uses naive datetime.utcnow, and
importing anything under witness_brain/ risks pulling in device code.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Closed command set. Nothing outside this set ever reaches a controller.
# ---------------------------------------------------------------------------
MOTION_COMMANDS = frozenset({"look_left", "look_right", "look_center"})
NON_MOTION_COMMANDS = frozenset({"stop", "say", "display"})
COMMANDS = MOTION_COMMANDS | NON_MOTION_COMMANDS
TEXT_COMMANDS = frozenset({"say", "display"})

REQUEST_KINDS = frozenset({"voice", "image_interpret", "note_retrieve"})

MAX_TEXT_LEN = 200
DISPLAY_LINE_LEN = 64


class MalformedCommand(ValueError):
    """Raised when a raw command does not match the strict schema."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(text: str, limit: int = MAX_TEXT_LEN) -> str:
    """Strip control characters and bound length. Used for say/display text."""
    out = "".join(ch for ch in text if ch == " " or (ch.isprintable() and ch not in "\x7f"))
    return out[:limit]


@dataclass
class Event:
    """Mirror of witness_brain Event, timezone-aware, JSON-line friendly."""
    source: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=utc_now_iso)
    data: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        d = {"id": self.id, "timestamp": self.timestamp, "source": self.source,
             "content": self.content}
        if self.data:
            d["data"] = self.data
        return json.dumps(d, sort_keys=True)


@dataclass
class Request:
    """A deliberate user request. kind is one of REQUEST_KINDS."""
    kind: str
    text: str = ""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.kind not in REQUEST_KINDS:
            raise ValueError("unknown request kind: %r" % (self.kind,))


_ACTION_FIELDS = {
    "command": str,
    "seq": int,
    "issued_at": float,
    "expires_at": float,
    "nonce": str,
    "issuer": str,
    "args": dict,
    "mac": str,
}


@dataclass
class ActionRequest:
    """A bounded action request from the coordinator to the robot side.

    seq is a monotonically increasing integer per issuer. issued_at and
    expires_at are wall-clock epoch seconds (cross-host; see RESIDUALS on clock
    sync). mac is an HMAC-SHA256 over canonical_bytes() with the issuer's key.
    """
    command: str
    seq: int
    issued_at: float
    expires_at: float
    nonce: str
    issuer: str
    args: Dict[str, Any] = field(default_factory=dict)
    mac: str = ""

    def canonical_bytes(self) -> bytes:
        d = self.to_dict()
        d.pop("mac", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> "ActionRequest":
        """Strict parse. Missing, extra or wrongly typed keys raise MalformedCommand."""
        if not isinstance(raw, dict):
            raise MalformedCommand("command is not an object")
        keys = set(raw.keys())
        expected = set(_ACTION_FIELDS)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(str(k) for k in keys - expected)
            raise MalformedCommand("bad keys: missing=%s extra=%s" % (missing, extra))
        for name, typ in _ACTION_FIELDS.items():
            val = raw[name]
            if typ is float:
                ok = isinstance(val, (int, float)) and not isinstance(val, bool)
            elif typ is int:
                ok = isinstance(val, int) and not isinstance(val, bool)
            else:
                ok = isinstance(val, typ)
            if not ok:
                raise MalformedCommand("field %s has wrong type" % name)
        if raw["command"] not in COMMANDS:
            raise MalformedCommand("command not in closed set")
        if raw["seq"] < 0:
            raise MalformedCommand("negative seq")
        args = raw["args"]
        if raw["command"] in TEXT_COMMANDS:
            if set(args.keys()) != {"text"} or not isinstance(args["text"], str):
                raise MalformedCommand("say/display take exactly one string arg 'text'")
            if len(args["text"]) > MAX_TEXT_LEN:
                raise MalformedCommand("text too long")
            if clean_text(args["text"]) != args["text"]:
                raise MalformedCommand("text has control characters")
        elif args:
            # Motion and stop take no arguments: no raw values ever reach a motor.
            raise MalformedCommand("%s takes no args" % raw["command"])
        return cls(command=raw["command"], seq=raw["seq"],
                   issued_at=float(raw["issued_at"]), expires_at=float(raw["expires_at"]),
                   nonce=raw["nonce"], issuer=raw["issuer"], args=dict(args), mac=raw["mac"])


# ActionResult.status values
APPLIED = "applied"      # the (fake) controller accepted and recorded it
REJECTED = "rejected"    # the guard refused it
UNKNOWN = "unknown"      # link lost: outcome not known; never auto-retried


@dataclass
class ActionResult:
    command: str
    seq: int
    nonce: str
    status: str
    reason: str = ""
    at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DISPLAY_MODES = frozenset({"idle", "listening", "thinking", "speaking", "looking",
                           "unavailable", "error"})


@dataclass
class DisplayState:
    """What the status display (OLED, later) should show."""
    mode: str = "idle"
    line: str = ""
    armed: bool = False
    link_ok: bool = True
    last_result: Optional[str] = None
    updated_at: str = field(default_factory=utc_now_iso)

    def set(self, mode: str, line: str = "") -> None:
        if mode not in DISPLAY_MODES:
            raise ValueError("unknown display mode %r" % (mode,))
        self.mode = mode
        self.line = clean_text(line, DISPLAY_LINE_LEN)
        self.updated_at = utc_now_iso()

    def render(self) -> List[str]:
        """Four short lines, OLED-sized."""
        return [
            ("ARMED" if self.armed else "DISARMED") + (" " if self.link_ok else " LINK LOST"),
            self.mode.upper(),
            self.line[:21],
            (self.last_result or "")[:21],
        ]
