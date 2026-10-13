"""Tool-facing interface: the "Witness integration services" layer.

Five plain callables with JSON-serialisable arguments and results, meant to be
wrapped later by an MCP server for an agent harness (OpenClaw is the
candidate). The harness manages the conversation; these only reach the
devices, the notes store and the robot guard.

Every function returns a dict with "ok" (bool) and never raises. Where a
compute backend was involved the dict carries "provenance" (see compute.py);
otherwise "provenance" is null.

- witness_snapshot()                 newest stored camera image, with capture time and age
- witness_speak(text)                speak through the configured speaker
- witness_status()                   what is actually wired: mic, camera, backend, controller, armed
- witness_propose_note(text)         a PROPOSED note (unapproved) in the local notes store; never the Stack
- robot_request_action(name, params) one closed-set action through the guard; returns its verdict

None of these can arm the robot.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .adapters.base import BackendUnavailable
from .coordinator import Coordinator
from .controls import EXTRA_MOTION, validate as validate_control
from .guard import CommandGuard
from .schemas import COMMANDS, MAX_TEXT_LEN, TEXT_COMMANDS, clean_text


@dataclass
class WitnessServices:
    coordinator: Coordinator
    guard: Optional[CommandGuard] = None  # robot-side guard, for status only (fakes/local tests)
    clock: Callable[[], float] = time.time


_services: Optional[WitnessServices] = None


def configure(services: Optional[WitnessServices]) -> None:
    """Wire the services the tool functions use. None unwires them."""
    global _services
    _services = services


def _fail(error: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False, "error": error, "provenance": None}
    out.update(extra)
    return out


def _kind(obj: Any) -> str:
    """Report what an adapter declares itself to be. Never guess "real".

    An adapter may set `adapter_kind` ("fake", "null", "real", "routed").
    Without a declaration, Null*/Fake* class names are recognised and anything
    else is "undeclared", so a wrapper around fakes can never read as real.
    """
    if obj is None:
        return "missing"
    declared = getattr(obj, "adapter_kind", None)
    if isinstance(declared, str) and declared:
        return declared
    name = type(obj).__name__
    if name.startswith("Null"):
        return "null"
    if name.startswith("Fake"):
        return "fake"
    return "undeclared"


def _guarded(fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        if _services is None:
            return _fail("services not configured")
        try:
            return fn(*args, **kwargs)
        except BackendUnavailable as exc:
            return _fail("unavailable: %s" % exc)
        except Exception as exc:  # a tool call must never raise into the harness
            return _fail("%s: %s" % (type(exc).__name__, clean_text(str(exc), 120)))
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@_guarded
def witness_snapshot() -> Dict[str, Any]:
    """Newest camera image: path, capture time, age, and whether it is fresh."""
    c = _services.coordinator
    if c.vision is None:
        return _fail("camera unavailable")
    snap = c.vision.snapshot()
    age = _services.clock() - snap.captured_at
    fresh = age <= c.max_image_age
    return {"ok": fresh, "path": snap.path, "captured_at": snap.captured_at,
            "age_s": round(age, 3), "fresh": fresh, "max_age_s": c.max_image_age,
            "error": None if fresh else "image stale", "provenance": None}


@_guarded
def witness_speak(text: str) -> Dict[str, Any]:
    """Speak text (control characters stripped, length bounded)."""
    if not isinstance(text, str) or not text.strip():
        return _fail("text must be a non-empty string")
    c = _services.coordinator
    if c.speech_out is None:
        return _fail("speaker unavailable")
    said = clean_text(text)
    c.speech_out.say(said)
    prov = getattr(c.speech_out, "last_provenance", None)
    return {"ok": True, "spoken": said, "truncated": len(said) < len(text),
            "provenance": prov.to_dict() if prov is not None else None}


@_guarded
def witness_status() -> Dict[str, Any]:
    """What is actually wired right now. Reports, never probes a device."""
    c = _services.coordinator
    g = _services.guard
    ctrl = c.controller
    backend: Dict[str, Any] = {"brain": _kind(c.brain)}
    router = getattr(c.brain, "router", None)
    if router is not None:
        backend["routes"] = {task: [{"backend_id": b, "status": router.status(b)} for b in order]
                             for task, order in router.config.routes.items()}
    return {
        "ok": True,
        "mic": _kind(c.speech_in),
        "speaker": _kind(c.speech_out),
        "camera": _kind(c.vision),
        "notes": _kind(c.notes),
        "backend": backend,
        "controller": {"kind": _kind(ctrl),
                       "link_ok": bool(getattr(ctrl, "connected", c.display.link_ok))},
        "armed": bool(g.armed) if g is not None else None,
        "armed_source": "local guard object" if g is not None else "unknown (guard is remote)",
        "display": c.display.render(),
        "provenance": None,
    }


@_guarded
def witness_propose_note(text: str) -> Dict[str, Any]:
    """Submit a PROPOSED note (approved=False) to the local notes store.

    It is never returned by retrieval until a human approves it, and it never
    touches the Stack.
    """
    if not isinstance(text, str) or not text.strip():
        return _fail("text must be a non-empty string")
    notes = _services.coordinator.notes
    if notes is None or not hasattr(notes, "add"):
        return _fail("notes store unavailable")
    note_id = "proposed-" + uuid.uuid4().hex[:12]
    notes.add(note_id, clean_text(text, 2000), ["proposed"], approved=False)
    return {"ok": True, "note_id": note_id, "status": "proposed, awaiting human approval",
            "provenance": None}


@_guarded
def robot_request_action(name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One named, bounded action through the guard. Returns the guard's verdict."""
    params = {} if params is None else params
    if not isinstance(name, str) or name not in COMMANDS:
        return _fail("unknown action", verdict="refused",
                     allowed=sorted(COMMANDS))
    if not isinstance(params, dict):
        return _fail("params must be an object", verdict="refused")
    if name in TEXT_COMMANDS:
        if set(params) != {"text"} or not isinstance(params.get("text"), str):
            return _fail("%s takes exactly {'text': str}" % name, verdict="refused")
        if len(params["text"]) > MAX_TEXT_LEN:
            return _fail("text too long", verdict="refused")
        args = {"text": clean_text(params["text"])}
    elif name in EXTRA_MOTION:
        try:
            validate_control(name, params)
        except ValueError as exc:
            return _fail(str(exc), verdict="refused")
        args = params
    else:
        if params:
            return _fail("%s takes no params" % name, verdict="refused")
        args = {}
    resp = _services.coordinator.request_action(name, args)
    if resp.result is None:
        return _fail(resp.text or "not issued", verdict="refused")
    r = resp.result
    return {"ok": r.status == "applied", "verdict": r.status, "reason": r.reason,
            "command": r.command, "seq": r.seq, "provenance": None,
            "note": "applied means the guard accepted it; it does not prove physical motion"}
