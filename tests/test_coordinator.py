import io
import json
import unittest

from tests.helpers import ISSUER, TempDirCase
from companion.adapters.base import BrainReply, NullBrain, NullVision
from companion.adapters.fake_speech import FakeSpeechIn
from companion.coordinator import Coordinator
from companion.persist import Journal
from companion.schemas import APPLIED, UNKNOWN, Request


class RoundTrip(TempDirCase):
    def test_voice_request_to_spoken_response(self):
        log = io.StringIO()
        coord, _, ctrl = self.make_stack(log_stream=log)
        coord.speech_in = FakeSpeechIn(["hello there"])
        resp = coord.listen_once()
        self.assertEqual(resp.kind, "speech")
        self.assertIn("hello there", resp.text)
        self.assertEqual(coord.speech_out.spoken, [resp.text])
        self.assertEqual(ctrl.log, [])
        events = [json.loads(line) for line in log.getvalue().splitlines()]
        self.assertTrue(events)
        self.assertTrue(all({"id", "timestamp", "source", "content"} <= set(e) for e in events))
        # transcript content is redacted from the log by default
        self.assertNotIn("hello there", log.getvalue())

    def test_voice_to_action_round_trip_when_armed(self):
        coord, guard, ctrl = self.make_stack(armed=True)
        resp = coord.handle(Request("voice", "please look left"))
        self.assertEqual(resp.kind, "action")
        self.assertEqual(resp.result.status, APPLIED)
        self.assertEqual(ctrl.would_have_done(), ["look_left"])
        self.assertEqual(coord.display.last_result, "look_left applied")

    def test_unknown_brain_action_is_dropped(self):
        class Wild(object):
            def respond(self, text, context):
                return BrainReply("ok", "spin_wheels_max; self_destruct")

        coord, _, ctrl = self.make_stack(armed=True, brain=Wild())
        resp = coord.handle(Request("voice", "anything"))
        self.assertEqual(resp.kind, "speech")
        self.assertEqual(ctrl.log, [])

    def test_spoken_stop_bypasses_brain(self):
        coord, _, ctrl = self.make_stack(brain=None)
        resp = coord.handle(Request("voice", "Stop!"))
        self.assertEqual(resp.result.status, APPLIED)
        self.assertEqual(ctrl.would_have_done(), ["stop"])


class HonestOutcome(TempDirCase):
    def test_rejected_action_is_not_spoken_as_done(self):
        coord, _, ctrl = self.make_stack(armed=False)
        resp = coord.handle(Request("voice", "look left"))
        self.assertEqual(resp.result.status, "rejected")
        self.assertNotIn("Looking left", resp.text)
        self.assertIn("disarmed", resp.text)

    def test_link_lost_stays_on_display(self):
        coord, _, ctrl = self.make_stack(armed=True)
        ctrl.connected = False
        coord.handle(Request("voice", "look left"))
        self.assertEqual(coord.display.mode, "error")
        self.assertIn("LINK LOST", coord.display.render()[0])


class Images(TempDirCase):
    def test_fresh_image_is_interpreted(self):
        coord, _, _ = self.make_stack()
        self.clock.advance(2)
        resp = coord.handle(Request("image_interpret", "what do you see"))
        self.assertEqual(resp.kind, "speech")
        self.assertIn("snap.jpg", resp.text)

    def test_stale_image_is_refused(self):
        coord, _, _ = self.make_stack()
        self.clock.advance(60)
        resp = coord.handle(Request("image_interpret", "what do you see"))
        self.assertEqual(resp.kind, "stale")
        self.assertNotIn("snap.jpg", resp.text)


class MissingBackends(TempDirCase):
    def test_none_backends_give_unavailable(self):
        coord = Coordinator(ISSUER, self.key, Journal(self.path("c.jsonl")), clock=self.clock)
        for req, what in ((Request("voice", "hi"), "brain"),
                          (Request("image_interpret"), "vision"),
                          (Request("note_retrieve", "x"), "notes")):
            resp = coord.handle(req)
            self.assertEqual(resp.kind, "unavailable", what)
            self.assertEqual(resp.data["backend"], what)
        self.assertEqual(coord.listen_once().kind, "unavailable")
        self.assertEqual(coord.handle(Request("voice", "stop")).kind, "unavailable")

    def test_null_backends_give_unavailable(self):
        coord, _, _ = self.make_stack(brain=NullBrain(), vision=NullVision())
        self.assertEqual(coord.handle(Request("voice", "hi")).kind, "unavailable")
        self.assertEqual(coord.handle(Request("image_interpret")).kind, "unavailable")
        self.assertEqual(coord.display.mode, "unavailable")

    def test_missing_speaker_still_returns_text(self):
        coord, _, _ = self.make_stack(speech_out=None)
        resp = coord.handle(Request("voice", "hello"))
        self.assertEqual(resp.kind, "speech")


class ConnectionLoss(TempDirCase):
    def test_link_loss_is_unknown_and_not_retried(self):
        coord, guard, ctrl = self.make_stack(armed=True)
        ctrl.connected = False
        resp = coord.handle(Request("voice", "look right"))
        self.assertEqual(resp.result.status, UNKNOWN)
        self.assertFalse(coord.display.link_ok)
        self.assertEqual(ctrl.log, [])
        ctrl.connected = True
        resp2 = coord.handle(Request("voice", "look right"))
        self.assertEqual(resp2.result.status, APPLIED)
        self.assertGreater(resp2.action.seq, resp.action.seq)
        self.assertEqual(len(ctrl.log), 1)  # the lost one was never resent
        self.assertTrue(coord.display.link_ok)


class Notes(TempDirCase):
    def test_approved_note_survives_restart(self):
        coord, _, _ = self.make_stack()
        coord.notes.add("n1", "The spare key is in the blue tin.", ["key"], approved=True)
        coord.notes.add("n2", "Unreviewed draft.", ["key"], approved=False)
        coord2, _, _ = self.make_stack()  # fresh objects, same files
        resp = coord2.handle(Request("note_retrieve", "key"))
        self.assertEqual(resp.data["note_id"], "n1")
        self.assertNotIn("Unreviewed", resp.text)


if __name__ == "__main__":
    unittest.main()
