import json
import unittest
from unittest.mock import patch

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

    def test_valid_structured_motion_reaches_coordinator_with_same_args(self):
        cases = [('drive', {'direction': 'forward', 'duty': 20,
                            'duration_seconds': 0.3}),
                 ('joints', {'positions': {'shoulder': 1500},
                             'duration_seconds': 0.5})]
        for name, params in cases:
            with self.subTest(name=name), patch.object(
                    self.coord, 'request_action', wraps=self.coord.request_action) as request:
                out = tools.robot_request_action(name, params)
                request.assert_called_once_with(name, params)
                self.assertIs(request.call_args.args[1], params)
                self.assertEqual((out['verdict'], out['reason']), ('rejected', 'disarmed'))
                self.assertIn('does not prove physical motion', out['note'])
        self.assertEqual(self.ctrl.would_have_done(), [])

    def assert_tool_refusal(self, name, params, error):
        with patch.object(self.coord, 'request_action') as request:
            out = tools.robot_request_action(name, params)
            self.assertFalse(out['ok'])
            self.assertEqual(out['verdict'], 'refused')
            self.assertEqual(out['error'], error)
            request.assert_not_called()

    def test_seven_joints_refused_before_coordinator(self):
        self.assert_tool_refusal('joints', {
            'positions': {'joint_%s' % n: 1500 for n in range(7)},
            'duration_seconds': 0.5}, 'one to six named joints required')

    def test_unknown_drive_direction_refused_before_coordinator(self):
        self.assert_tool_refusal('drive', {'direction': 'up', 'duty': 20,
            'duration_seconds': 0.3}, 'unknown drive direction')

    def test_missing_drive_params_refused_before_coordinator(self):
        for params in (None, {}):
            with self.subTest(params=params):
                self.assert_tool_refusal('drive', params, 'incorrect control arguments')

    def test_look_params_still_refused_before_coordinator(self):
        self.assert_tool_refusal('look_left', {'angle': 90}, 'look_left takes no params')

    def test_existing_shape_and_text_refusals_preserved(self):
        for name, params, error in [
                ('unknown', {}, 'unknown action'),
                ('drive', [], 'params must be an object'),
                ('say', {'text': 5}, "say takes exactly {'text': str}"),
                ('display', {'text': 'x' * 201}, 'text too long')]:
            with self.subTest(name=name):
                self.assert_tool_refusal(name, params, error)

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
