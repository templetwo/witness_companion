#!/usr/bin/env python3
"""Stdin-driven demo of the companion coordinator with every backend faked.

Lines on stdin:
    image [prompt]      interpret the stored fake camera image
    note <query>        retrieve an approved note
    stale               age the fake image past the freshness limit
    linkdown / linkup   simulate losing / restoring the Pi link
    anything else       a push-to-talk voice request ("look left", "stop", ...)

--armed only arms the FAKE guard so look_* commands are accepted by the fake
controller. There are no real adapters: nothing moves, records, or speaks
aloud, with or without --armed. State lives in a temp dir unless --state-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

from companion.adapters.fake_brain import FakeBrain
from companion.adapters.fake_controller import FakeController
from companion.adapters.fake_notes import FakeNotes
from companion.adapters.fake_speech import FakeSpeechIn, FakeSpeechOut
from companion.adapters.fake_vision import FakeVision
from companion.coordinator import Coordinator
from companion.guard import CommandGuard
from companion.persist import Journal
from companion.schemas import Request

ISSUER = "jetson-coordinator"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--armed", action="store_true", default=False,
                    help="arm the FAKE guard (default: disarmed)")
    ap.add_argument("--state-dir", default=None, help="journal/notes dir (default: temp)")
    args = ap.parse_args(argv)

    state = args.state_dir or tempfile.mkdtemp(prefix="wc-demo-")
    key = os.urandom(32)  # per-run demo key, never written anywhere
    guard = CommandGuard({ISSUER: key}, journal=Journal(os.path.join(state, "guard.jsonl")))
    if args.armed:
        guard.arm()
        print("ARMED (fake guard only; there are no real adapters, nothing can move)")
    else:
        print("DISARMED: motion commands will be refused; stop is always accepted")
    ctrl = FakeController(guard)
    img = os.path.join(state, "snapshot.jpg")
    with open(img, "wb") as f:
        f.write(b"placeholder image bytes")
    vision = FakeVision(img, captured_at=time.time())
    notes = FakeNotes(os.path.join(state, "notes.json"))
    if not notes.get(""):
        notes.add("welcome", "This is an approved demo note.", ["demo"], approved=True)
    speech_out = FakeSpeechOut()
    coord = Coordinator(ISSUER, key, Journal(os.path.join(state, "coord.jsonl")),
                        speech_in=FakeSpeechIn(), speech_out=speech_out, vision=vision,
                        brain=FakeBrain(), notes=notes, controller=ctrl,
                        log_stream=sys.stderr)
    coord.display.armed = guard.armed
    print("state dir:", state)
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        word, _, rest = line.partition(" ")
        if word == "image":
            resp = coord.handle(Request("image_interpret", rest))
        elif word == "note":
            resp = coord.handle(Request("note_retrieve", rest))
        elif word == "stale":
            vision.captured_at = time.time() - 3600
            print("(fake image aged one hour)")
            continue
        elif word in ("linkdown", "linkup"):
            ctrl.connected = word == "linkup"
            print("(fake link %s)" % ("up" if ctrl.connected else "down"))
            continue
        else:
            coord.speech_in.push(line)
            resp = coord.listen_once()
        coord.display.armed = guard.armed
        print("> %s: %s" % (resp.kind, resp.text))
        if resp.result is not None:
            print("  result:", json.dumps(resp.result.to_dict(), sort_keys=True))
        print("  display:", " | ".join(coord.display.render()))
    print("fake controller log (what it WOULD have done; nothing physical happened):")
    for e in ctrl.log:
        print("  ", json.dumps(e, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
