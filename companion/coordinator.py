"""Companion coordinator (the Jetson side).

Request -> backends -> spoken response or a bounded, signed ActionRequest ->
controller -> ActionResult; DisplayState updated; one JSON line logged per
event in the witness_brain Event shape.

Rules:
- A missing backend (None or Null) yields an "unavailable" Response, never an
  exception.
- The Brain's action field is free model text. It is mapped through
  BRAIN_ACTIONS; anything else is dropped (logged) and the reply is speech only.
- A spoken "stop" is handled deterministically without the brain. It is a
  convenience, not a safety mechanism.
- Every issued action is journalled BEFORE it is sent. On restart the next seq
  continues above the journal's high-water mark and nothing is resent.
- Link loss: the result is UNKNOWN, the display shows LINK LOST, and the
  command is NOT retried (a retry would be a replay).
- Transcript text is not written to the log unless log_content=True
  (no continuous recording archive by default).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TextIO

from .adapters.base import (BackendUnavailable, Brain, Controller, Notes, SpeechIn,
                            SpeechOut, Vision)
from .persist import Journal
from .schemas import (REJECTED, UNKNOWN, ActionRequest, ActionResult, DisplayState, Event, Request,
                      clean_text)

# Model-facing action names -> closed-set commands. Nothing else passes.
BRAIN_ACTIONS = {
    "look_left": "look_left",
    "look_right": "look_right",
    "look_center": "look_center",
    "stop": "stop",
}

STOP_WORDS = frozenset({"stop", "stop now", "witness stop", "halt"})


@dataclass
class Response:
    kind: str  # "speech" | "action" | "unavailable" | "stale" | "error"
    text: str
    action: Optional[ActionRequest] = None
    result: Optional[ActionResult] = None
    data: Dict[str, Any] = field(default_factory=dict)


class Coordinator:
    def __init__(self, issuer: str, key: bytes, journal: Journal,
                 speech_in: Optional[SpeechIn] = None, speech_out: Optional[SpeechOut] = None,
                 vision: Optional[Vision] = None, brain: Optional[Brain] = None,
                 notes: Optional[Notes] = None, controller: Optional[Controller] = None,
                 clock: Callable[[], float] = time.time, action_ttl: float = 3.0,
                 max_image_age: float = 10.0, log_stream: Optional[TextIO] = None,
                 log_content: bool = False) -> None:
        self.issuer = issuer
        self._key = key
        self.journal = journal
        self.speech_in = speech_in
        self.speech_out = speech_out
        self.vision = vision
        self.brain = brain
        self.notes = notes
        self.controller = controller
        self.clock = clock
        self.action_ttl = action_ttl
        self.max_image_age = max_image_age
        self.log_stream = log_stream
        self.log_content = log_content
        self.display = DisplayState()
        self.context: List[str] = []
        # Never reuse a seq: continue above anything ever issued.
        self._next_seq = journal.max_seq("issued", issuer) + 1

    # ------------------------------------------------------------------ logging
    def _log(self, source: str, content: str, **data: Any) -> Event:
        ev = Event(source=source, content=content, data=data or None)
        if self.log_stream is not None:
            self.log_stream.write(ev.to_json() + "\n")
            self.log_stream.flush()
        return ev

    def _redact(self, text: str) -> str:
        return text if self.log_content else "<%d chars>" % len(text)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _missing(backend: Any) -> bool:
        return backend is None

    def _provenance(self) -> Optional[Dict[str, Any]]:
        prov = getattr(self.brain, "last_provenance", None)
        if prov is None:
            return None
        self.brain.last_provenance = None  # one result, one record
        return prov.to_dict()

    def _unavailable(self, what: str, detail: str = "") -> Response:
        msg = "Sorry, my %s is unavailable right now." % what
        self.display.set("unavailable", what)
        self._log("system:unavailable", what, detail=detail)
        return Response("unavailable", msg, data={"backend": what})

    def _speak(self, text: str) -> None:
        self.display.set("speaking", text)
        if self.speech_out is None:
            self._log("speech_out:missing", "text-only response")
            return
        try:
            self.speech_out.say(text)
        except BackendUnavailable as exc:
            self._log("speech_out:unavailable", str(exc))

    # ------------------------------------------------------------------ entry points
    def listen_once(self) -> Response:
        """Push-to-talk: one utterance from speech_in, handled as a voice request."""
        if self._missing(self.speech_in):
            return self._unavailable("hearing")
        self.display.set("listening")
        try:
            text = self.speech_in.listen()
        except BackendUnavailable as exc:
            return self._unavailable("hearing", str(exc))
        return self.handle(Request(kind="voice", text=text))

    def handle(self, req: Request) -> Response:
        self._log("request:" + req.kind, self._redact(req.text), request_id=req.request_id)
        try:
            if req.kind == "voice":
                resp = self._voice(req)
            elif req.kind == "image_interpret":
                resp = self._image(req)
            else:
                resp = self._note(req)
        except BackendUnavailable as exc:
            resp = self._unavailable(exc.what, str(exc))
        if resp.kind in ("speech", "stale", "unavailable") and resp.text:
            self._speak(resp.text)
        # Leave the display showing the outcome, not the transient "speaking".
        if resp.kind == "unavailable":
            self.display.set("unavailable", str(resp.data.get("backend", "")))
        elif self.display.mode == "speaking":
            self.display.set("idle", resp.text)
        return resp

    # ------------------------------------------------------------------ flows
    def _voice(self, req: Request) -> Response:
        norm = " ".join(req.text.lower().strip(" .!?").split())
        if norm in STOP_WORDS:
            return self._issue("stop", {}, spoken="Stopping.")
        if self._missing(self.brain):
            return self._unavailable("brain")
        self.display.set("thinking")
        reply = self.brain.respond(req.text, list(self.context[-10:]))
        prov = self._provenance()
        speech = clean_text(reply.speech or "")
        self.context.append(req.text)
        self._log("brain:reply", self._redact(speech), provenance=prov)
        if reply.action is not None:
            cmd = BRAIN_ACTIONS.get(str(reply.action).strip().lower())
            if cmd is None:
                self._log("brain:action_dropped", "not in whitelist",
                          action=clean_text(str(reply.action), 40))
            else:
                resp = self._issue(cmd, {}, spoken=speech)
                resp.data["provenance"] = prov
                return resp
        return Response("speech", speech, data={"provenance": prov})

    def _image(self, req: Request) -> Response:
        if self._missing(self.vision):
            return self._unavailable("vision")
        if self._missing(self.brain):
            return self._unavailable("brain")
        caps = getattr(self.brain, "capabilities", None)
        if callable(caps) and "image" not in caps():
            # Refuse before any call: do not send an image to a text-only backend.
            return self._unavailable("vision_interpret", "brain declares no image capability")
        self.display.set("looking")
        snap = self.vision.snapshot()
        age = self.clock() - snap.captured_at
        if age > self.max_image_age:
            self._log("vision:stale", "image too old", age_s=round(age, 3))
            return Response("stale", "My last picture is %d seconds old, so I won't describe it."
                            % int(age), data={"age_s": age})
        desc = clean_text(self.brain.describe_image(snap.path, req.text or "Describe briefly."))
        prov = self._provenance()
        self._log("vision:look", self._redact(desc), age_s=round(age, 3), provenance=prov)
        return Response("speech", desc, data={"age_s": age, "path": snap.path,
                                              "provenance": prov})

    def _note(self, req: Request) -> Response:
        if self._missing(self.notes):
            return self._unavailable("notes")
        found = self.notes.get(req.text)
        if not found:
            return Response("speech", "I have no approved note about that.")
        self._log("notes:hit", found[0].id, count=len(found))
        return Response("speech", clean_text(found[0].text), data={"note_id": found[0].id})

    # ------------------------------------------------------------------ actions
    def _new_action(self, command: str, args: Dict[str, Any]) -> ActionRequest:
        now = self.clock()
        action = ActionRequest(command=command, seq=self._next_seq, issued_at=now,
                               expires_at=now + self.action_ttl, nonce=os.urandom(12).hex(),
                               issuer=self.issuer, args=dict(args))
        self._next_seq += 1
        action.mac = hmac.new(self._key, action.canonical_bytes(), hashlib.sha256).hexdigest()
        return action

    def _issue(self, command: str, args: Dict[str, Any], spoken: str = "") -> Response:
        if self._missing(self.controller):
            return self._unavailable("controller")
        action = self._new_action(command, args)
        # Journal before send: a crash after this line can never cause a resend.
        self.journal.append({"type": "issued", "issuer": self.issuer, "seq": action.seq,
                             "nonce": action.nonce, "command": command})
        try:
            result = self.controller.send(action)
            self.display.link_ok = True
        except (ConnectionError, OSError, TimeoutError) as exc:
            result = ActionResult(command=command, seq=action.seq, nonce=action.nonce,
                                  status=UNKNOWN, reason="link lost: %s" % exc)
            self.display.link_ok = False
            self.display.set("error", "link lost")
        except BackendUnavailable as exc:
            return self._unavailable("controller", str(exc))
        self.journal.append({"type": "result", "issuer": self.issuer, "seq": action.seq,
                             "status": result.status, "reason": result.reason})
        self.display.last_result = "%s %s" % (command, result.status)
        self._log("action:" + result.status, command, seq=action.seq, reason=result.reason)
        if result.status == UNKNOWN:
            text = "I lost the link to the body. I don't know if that happened."
        elif result.status == REJECTED:
            text = "I can't do that right now (%s)." % result.reason
        else:
            text = spoken
        if text:
            self._speak(text)
        if result.status == UNKNOWN:
            self.display.set("error", "link lost")
        return Response("action", text, action=action, result=result)
