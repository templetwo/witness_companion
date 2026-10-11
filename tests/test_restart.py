import unittest

from tests.helpers import ISSUER, TempDirCase
from companion.persist import Journal
from companion.schemas import Request


class RestartWithoutReplay(TempDirCase):
    def test_coordinator_restart_does_not_resend(self):
        coord, guard, ctrl = self.make_stack(armed=True)
        r1 = coord.handle(Request("voice", "look left"))
        self.assertEqual(len(ctrl.log), 1)
        # "crash": drop coordinator, guard and controller objects; keep the files
        coord2, guard2, ctrl2 = self.make_stack(armed=True)
        self.assertEqual(ctrl2.log, [])  # nothing replayed on startup
        r2 = coord2.handle(Request("voice", "look right"))
        self.assertGreater(r2.action.seq, r1.action.seq)
        self.assertEqual(ctrl2.would_have_done(), ["look_right"])

    def test_guard_rejects_replayed_command_after_restart(self):
        coord, guard, ctrl = self.make_stack(armed=True)
        r1 = coord.handle(Request("voice", "look left"))
        guard2 = self.make_guard(armed=True)
        self.assertEqual(guard2.check(r1.action).reason, "duplicate")

    def test_seq_continues_above_unacknowledged_issue(self):
        coord, _, ctrl = self.make_stack(armed=True)
        ctrl.connected = False
        lost = coord.handle(Request("voice", "look left")).action
        coord2, _, _ = self.make_stack(armed=True)
        nxt = coord2.handle(Request("voice", "look left")).action
        self.assertGreater(nxt.seq, lost.seq)

    def test_torn_journal_line_is_tolerated(self):
        j = Journal(self.path("g.jsonl"))
        j.append({"type": "applied", "issuer": ISSUER, "seq": 7, "nonce": "a"})
        with open(j.path, "a") as f:
            f.write('{"type": "applied", "issuer": "' + ISSUER + '", "se')  # crash mid-write
        recs, skipped = j.read()
        self.assertEqual(skipped, 1)
        self.assertEqual(j.max_seq("applied", ISSUER), 7)
        j.append({"type": "applied", "issuer": ISSUER, "seq": 8, "nonce": "b"})
        self.assertEqual(j.max_seq("applied", ISSUER), 8)
        self.assertEqual(j.read()[1], 1)


if __name__ == "__main__":
    unittest.main()
