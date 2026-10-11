import json
import unittest

from tests.helpers import TempDirCase
from companion import tools


class ToolSurface(TempDirCase):
    def setUp(self):
        super().setUp()
        self.coord, self.guard, self.ctrl = self.make_stack(armed=False)
        tools.configure(tools.WitnessServices(self.coord, self.guard, clock=self.clock))

    def tearDown(self):
        tools.configure(None)
        super().tearDown()

    def call_all(self):
        return {
            "witness_snapshot": tools.witness_snapshot(),
            "witness_speak": tools.witness_speak("hello"),
            "witness_status": tools.witness_status(),
            "witness_propose_note": tools.witness_propose_note("Remember the blue tin."),
            "robot_request_action": tools.robot_request_action("stop", {}),
        }

    def test_all_five_return_json_dicts_and_never_raise(self):
        for name, out in self.call_all().items():
            self.assertIsInstance(out, dict, name)
            json.dumps(out)  # must be JSON-serialisable
            self.assertIn("ok", out, name)
            self.assertIn("provenance", out, name)
            self.assertTrue(out["ok"], (name, out))

    def test_never_raise_on_bad_input_or_missing_backends(self):
        cases = [tools.witness_speak(None), tools.witness_speak(""),
                 tools.witness_propose_note(42),
                 tools.robot_request_action("arm", {}),
                 tools.robot_request_action("look_left", {"angle": 90}),
                 tools.robot_request_action("say", {"text": 5}),
                 tools.robot_request_action(None, None)]
        self.coord.vision = None
        self.coord.speech_out = None
        self.coord.controller = None
        cases += [tools.witness_snapshot(), tools.witness_speak("hi"), tools.witness_status(),
                  tools.robot_request_action("stop", {})]
        for out in cases:
            self.assertIsInstance(out, dict)
            json.dumps(out)
        self.assertFalse(cases[3]["ok"])  # arm is not an action
        tools.configure(None)
        self.assertFalse(tools.witness_status()["ok"])

    def test_disarmed_refuses_motion_accepts_stop(self):
        motion = tools.robot_request_action("look_left", {})
        self.assertFalse(motion["ok"])
        self.assertEqual((motion["verdict"], motion["reason"]), ("rejected", "disarmed"))
        stop = tools.robot_request_action("stop", {})
        self.assertTrue(stop["ok"])
        self.assertEqual(stop["verdict"], "applied")
        self.assertEqual(self.ctrl.would_have_done(), ["stop"])

    def test_status_reports_wiring_and_armed(self):
        st = tools.witness_status()
        self.assertEqual((st["mic"], st["camera"], st["controller"]["kind"]),
                         ("fake", "fake", "fake"))
        self.assertIs(st["armed"], False)

    def test_status_never_guesses_real(self):
        from companion.adapters.fake_compute import FakeComputeBackend
        from companion.compute import ComputeConfig, RoutedBrain, Router

        class Wrapper(object):  # an unlabelled wrapper around a fake
            pass

        cfg = ComputeConfig.from_dict({"backends": {"jet": {"kind": "jetson_local"}},
                                       "tasks": {"conversation": {"backend": "jet"}}})
        self.coord.brain = RoutedBrain(Router(cfg, {"jet": FakeComputeBackend("jet")}, env={}))
        self.coord.vision = Wrapper()
        st = tools.witness_status()
        self.assertEqual(st["backend"]["brain"], "routed")
        self.assertEqual(st["backend"]["routes"]["conversation"][0]["status"], "ready")
        self.assertEqual(st["camera"], "undeclared")

    def test_snapshot_stale_and_proposed_note_not_retrievable(self):
        self.clock.advance(120)
        snap = tools.witness_snapshot()
        self.assertFalse(snap["ok"])
        self.assertFalse(snap["fresh"])
        tools.witness_propose_note("Proposed text about keys")
        self.assertEqual(self.coord.notes.get("keys"), [])


if __name__ == "__main__":
    unittest.main()
