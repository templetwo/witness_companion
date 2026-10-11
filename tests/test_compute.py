import io
import json
import os
import unittest

from tests.helpers import TempDirCase
from companion.adapters.base import BackendUnavailable
from companion.adapters.fake_compute import FakeComputeBackend, failing
from companion.compute import (ComputeConfig, ConfigError, RoutedBrain, Router,
                               load_compute_config)
from companion.schemas import Request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRET = "SECRET-VALUE-MUST-NEVER-BE-LOGGED-" + os.urandom(6).hex()


def cfg(**routes):
    return ComputeConfig.from_dict({
        "backends": {
            "jet": {"kind": "jetson_local", "provider": "ollama", "model": "gemma3:4b"},
            "studio": {"kind": "studio_local", "provider": "ollama", "model": "m",
                       "credential_env": "WC_TEST_STUDIO_TOKEN"},
            "cloud": {"kind": "cloud", "provider": "acme", "model": "c",
                      "credential_env": "WC_TEST_CLOUD_KEY"},
        },
        "tasks": routes or {
            "conversation": {"backend": "jet", "fallbacks": ["studio", "cloud"]},
            "vision_interpret": {"backend": "jet", "fallbacks": ["studio", "cloud"]},
        },
    })


class Routing(unittest.TestCase):
    def setUp(self):
        self.env = {"WC_TEST_STUDIO_TOKEN": SECRET, "WC_TEST_CLOUD_KEY": SECRET}
        self.events = []

    def router(self, adapters, config=None, env=None):
        return Router(config or cfg(), adapters, env=self.env if env is None else env,
                      log=lambda s, d: self.events.append((s, d)))

    def test_per_task_routing_picks_configured_backend(self):
        c = cfg(conversation={"backend": "studio", "fallbacks": ["jet"]},
                vision_interpret={"backend": "cloud", "fallbacks": []})
        jet = FakeComputeBackend("jet", ["text", "image"])
        studio = FakeComputeBackend("studio", ["text"])
        cloud = FakeComputeBackend("cloud", ["text", "image"])
        r = self.router({"jet": jet, "studio": studio, "cloud": cloud}, c)
        _, p1 = r.call("conversation", "respond", "hi", [])
        _, p2 = r.call("vision_interpret", "describe_image", "/x.jpg", "d")
        self.assertEqual((p1.backend_id, p2.backend_id), ("studio", "cloud"))
        self.assertEqual(jet.calls, [])

    def test_fallback_order_honored(self):
        jet = failing("jet")
        studio = FakeComputeBackend("studio", ["text"], fail=ConnectionError("x"))
        cloud = FakeComputeBackend("cloud", ["text"])
        r = self.router({"jet": jet, "studio": studio, "cloud": cloud})
        _, p = r.call("conversation", "respond", "hi", [])
        self.assertEqual(p.backend_id, "cloud")
        self.assertEqual([a["backend_id"] for a in p.attempts], ["jet", "studio", "cloud"])
        self.assertEqual((jet.calls, studio.calls), (["respond"], ["respond"]))

    def test_unavailable_primary_skipped(self):
        cloud = FakeComputeBackend("cloud", ["text"])
        r = self.router({"cloud": cloud})  # jet and studio have no adapter wired
        _, p = r.call("conversation", "respond", "hi", [])
        self.assertEqual(p.backend_id, "cloud")
        self.assertEqual(p.attempts[0]["outcome"], "no adapter")

    def test_all_fail_raises_unavailable_with_attempts(self):
        r = self.router({"jet": failing("jet")}, env={})
        with self.assertRaises(BackendUnavailable) as cm:
            r.call("conversation", "respond", "hi", [])
        self.assertEqual(len(cm.exception.attempts), 3)
        self.assertEqual(cm.exception.what, "conversation")

    def test_provenance_recorded_on_every_result(self):
        jet = FakeComputeBackend("jet", ["text", "image"], where_computed="jetson",
                                 cold_start=True)
        studio = FakeComputeBackend("studio", ["text"])  # reports nothing about location
        r = self.router({"jet": jet, "studio": studio})
        for task, method, args in (("conversation", "respond", ("hi", [])),
                                   ("vision_interpret", "describe_image", ("/x.jpg", "d"))):
            _, p = r.call(task, method, *args)
            d = p.to_dict()
            for k in ("backend_id", "kind", "provider", "model", "where_computed",
                      "latency_ms", "cold_start"):
                self.assertIn(k, d)
            self.assertEqual((p.kind, p.where_computed, p.cold_start),
                             ("jetson_local", "jetson", True))
        r2 = self.router({"studio": studio},
                         cfg(conversation={"backend": "studio", "fallbacks": []}))
        _, p2 = r2.call("conversation", "respond", "hi", [])
        self.assertEqual(p2.kind, "studio_local")
        self.assertEqual(p2.where_computed, "unknown")  # the Studio is not assumed local
        self.assertEqual(len([s for s, _ in self.events if s.startswith("compute:")]), 3)

    def test_capability_mismatch_refused_before_call(self):
        jet = FakeComputeBackend("jet", ["text"])
        studio = FakeComputeBackend("studio", ["text"])
        r = self.router({"jet": jet, "studio": studio},
                        cfg(vision_interpret={"backend": "jet", "fallbacks": ["studio"]}))
        with self.assertRaises(BackendUnavailable) as cm:
            r.call("vision_interpret", "describe_image", "/x.jpg", "d")
        self.assertEqual((jet.calls, studio.calls), ([], []))
        self.assertTrue(all("capability mismatch" in a["outcome"]
                            for a in cm.exception.attempts))

    def test_missing_credential_env_reports_not_configured_without_value(self):
        env = {"WC_TEST_STUDIO_TOKEN": SECRET}  # cloud var unset
        cloud = FakeComputeBackend("cloud", ["text"])
        studio = FakeComputeBackend("studio", ["text"])
        r = self.router({"cloud": cloud, "studio": studio},
                        cfg(conversation={"backend": "cloud", "fallbacks": ["studio"]}), env=env)
        self.assertEqual(r.status("cloud"), "not configured (WC_TEST_CLOUD_KEY unset)")
        _, p = r.call("conversation", "respond", "hi", [])
        self.assertEqual(p.backend_id, "studio")
        self.assertEqual(cloud.calls, [])
        self.assertTrue(p.attempts[0]["outcome"].startswith("not configured"))
        self.assertNotIn(SECRET, json.dumps(self.events) + json.dumps(p.to_dict()))

    def test_config_rejects_bad_shapes(self):
        bad = [
            {"backends": {"x": {"kind": "magic"}}},
            {"backends": {}, "tasks": {"conversation": {"backend": "y"}}},
            {"backends": {"x": {"kind": "cloud", "credential_env": "abc def=1"}}},
            {"backends": {"x": {"kind": "cloud", "api_key": "v"}}},
            {"backends": {}, "tasks": {"telepathy": {"backend": "x"}}},
        ]
        for d in bad:
            with self.assertRaises(ConfigError, msg=str(d)):
                ComputeConfig.from_dict(d)


class TemplateConfig(unittest.TestCase):
    def test_template_has_every_task_and_no_secret_values(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("pyyaml not installed in this interpreter (present on the Jetson)")
        path = os.path.join(ROOT, "config", "compute.yaml")
        c = load_compute_config(path)
        self.assertEqual(set(c.routes), {"conversation", "vision_interpret", "speech_to_text",
                                         "text_to_speech", "note_retrieval"})
        kinds = {s.kind for s in c.backends.values()}
        self.assertTrue({"jetson_local", "studio_local", "cloud"} <= kinds)
        for spec in c.backends.values():
            if spec.credential_env:
                self.assertTrue(spec.credential_env.isupper())  # a NAME, not a value


class CoordinatorWithRouter(TempDirCase):
    def test_provenance_in_log_line_and_image_refused_for_text_only(self):
        log = io.StringIO()
        jet = FakeComputeBackend("jet", ["text"], where_computed="jetson")
        c = cfg(conversation={"backend": "jet", "fallbacks": []},
                vision_interpret={"backend": "jet", "fallbacks": []})
        brain = RoutedBrain(Router(c, {"jet": jet}, env={}))
        coord, _, _ = self.make_stack(brain=brain, log_stream=log)
        resp = coord.handle(Request("voice", "hi"))
        self.assertEqual(resp.data["provenance"]["backend_id"], "jet")
        lines = [json.loads(x) for x in log.getvalue().splitlines()]
        reply = [x for x in lines if x["source"] == "brain:reply"][0]
        self.assertEqual(reply["data"]["provenance"]["where_computed"], "jetson")
        self.clock.advance(1)
        img = coord.handle(Request("image_interpret", "look"))
        self.assertEqual(img.kind, "unavailable")
        self.assertEqual(img.data["backend"], "vision_interpret")
        self.assertEqual(jet.calls, ["respond"])  # the image was never sent


if __name__ == "__main__":
    unittest.main()
