import ast
import os
import unittest

from tests.helpers import ISSUER, TempDirCase
from companion.guard import sign
from companion.schemas import ActionRequest


class GuardCase(TempDirCase):
    def act(self, command="look_left", seq=1, args=None, ttl=2.0, key=None, issuer=ISSUER,
            issued_at=None, nonce=None):
        now = self.clock() if issued_at is None else issued_at
        a = ActionRequest(command=command, seq=seq, issued_at=now, expires_at=now + ttl,
                          nonce=nonce or os.urandom(8).hex(), issuer=issuer,
                          args=args if args is not None else {})
        return sign(a, key if key is not None else self.key)


class Malformed(GuardCase):
    def test_malformed_rejected(self):
        g = self.make_guard(armed=True)
        good = self.act().to_dict()
        cases = {
            "not a dict": "look_left",
            "unknown command": dict(good, command="spin"),
            "extra key": dict(good, speed=255),
            "missing key": {k: v for k, v in good.items() if k != "nonce"},
            "bool seq": dict(good, seq=True),
            "args on motion": dict(good, args={"angle": 90}),
            "control chars": self.act("say", 2, {"text": "hi\x1b[2J"}).to_dict(),
            "long text": self.act("display", 3, {"text": "x" * 500}).to_dict(),
        }
        for name, raw in cases.items():
            d = g.check(raw)
            self.assertFalse(d.accepted, name)
            self.assertTrue(d.reason.startswith("malformed"), (name, d.reason))
        self.assertEqual(g.last_seq(ISSUER), -1)


class Unauthorized(GuardCase):
    def test_wrong_key_unknown_issuer_tampered_unsigned(self):
        g = self.make_guard(armed=True)
        self.assertEqual(g.check(self.act(key=os.urandom(32))).reason, "unauthorized")
        self.assertEqual(g.check(self.act(issuer="stranger")).reason, "unauthorized")
        t = self.act()
        t.command = "look_right"  # tampered after signing
        self.assertEqual(g.check(t).reason, "unauthorized")
        u = self.act()
        u.mac = ""
        self.assertEqual(g.check(u).reason, "unauthorized")
        self.assertEqual(g.last_seq(ISSUER), -1)


class Expiry(GuardCase):
    def test_expired_and_bad_windows(self):
        g = self.make_guard(armed=True)
        a = self.act(ttl=2.0)
        self.clock.advance(3)
        self.assertEqual(g.check(a).reason, "expired")
        self.assertEqual(g.check(self.act(seq=2, ttl=60)).reason, "ttl too long")
        self.assertEqual(g.check(self.act(seq=3, issued_at=self.clock() + 30)).reason,
                         "issued in the future")
        # rejected commands do not consume seq: seq 1 is still usable when fresh
        self.assertTrue(g.check(self.act(seq=1)).accepted)


class Duplicate(GuardCase):
    def test_duplicate_seq_and_nonce(self):
        g = self.make_guard(armed=True)
        a = self.act(seq=5)
        self.assertTrue(g.check(a).accepted)
        self.assertEqual(g.check(a).reason, "duplicate")
        self.assertEqual(g.check(self.act(seq=4)).reason, "duplicate")
        self.assertEqual(g.check(self.act(seq=9, nonce=a.nonce)).reason, "duplicate")
        self.assertTrue(g.check(self.act(seq=6)).accepted)


class Arming(GuardCase):
    def test_disarmed_start_rejects_motion_accepts_stop(self):
        g = self.make_guard()
        self.assertFalse(g.armed)
        self.assertEqual(g.check(self.act("look_left", 1)).reason, "disarmed")
        self.assertTrue(g.check(self.act("say", 2, {"text": "hello"})).accepted)
        self.assertTrue(g.check(self.act("stop", 3)).accepted)

    def test_stop_always_accepted_and_disarms(self):
        g = self.make_guard(armed=True)
        self.assertTrue(g.check({"command": "stop"}).accepted)  # malformed stop
        self.assertFalse(g.armed)
        g.arm()
        forged = self.act("stop", seq=10**9, key=os.urandom(32))
        d = g.check(forged)
        self.assertTrue(d.accepted)
        self.assertFalse(d.authenticated)
        self.assertEqual(g.last_seq(ISSUER), -1)  # forged stop did not poison seq
        g.arm()
        self.assertTrue(g.check(self.act("look_left", 1)).accepted)
        old = self.act("stop", 2, ttl=1)
        self.clock.advance(5)
        self.assertTrue(g.check(old).accepted)  # expired stop still stops

    def test_arm_is_not_a_command_and_never_persisted(self):
        g = self.make_guard()
        a = self.act("look_left", 1).to_dict()
        a["command"] = "arm"
        self.assertFalse(g.check(a).accepted)
        g.arm()
        self.assertTrue(g.check(self.act("look_left", 2)).accepted)
        g2 = self.make_guard()  # restart, same journal
        self.assertFalse(g2.armed)
        self.assertEqual(g2.last_seq(ISSUER), 2)


class GuardIndependentOfCompute(unittest.TestCase):
    def test_guard_imports_no_backend_module(self):
        import companion.guard as gm
        with open(gm.__file__) as f:
            tree = ast.parse(f.read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mods.add(node.module or "")
            elif isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
        self.assertFalse({m for m in mods if "compute" in m or "adapters" in m or
                          "coordinator" in m}, mods)


if __name__ == "__main__":
    unittest.main()
