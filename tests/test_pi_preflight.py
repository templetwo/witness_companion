# Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:09:44 EDT
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pi_preflight',Path(__file__).resolve().parents[1]/'scripts/pi_preflight.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class PreflightTests(unittest.TestCase):
    def fixture(self, root):
        (root/'net').mkdir()
        (root/'net/tcp').write_text('header\n0: 00000000:2342 00000000:0000 0A 0 0 0 0 0 123\n')
        (root/'net/tcp6').write_text('header\n')
        process=root/'77'
        (process/'fd').mkdir(parents=True)
        (process/'comm').write_text('python3\n')
        (process/'cgroup').write_text('0::/system.slice/example.service\n')
        (process/'fd/4').symlink_to('/dev/ttyAMA0')
        (process/'fd/5').symlink_to('socket:[123]')
        return process

    def test_maps_serial_and_9026_without_reading_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            process=self.fixture(root)
            (process/'cmdline').write_text('sensitive-placeholder')
            result=probe.inspect_owners(proc=root)
            self.assertEqual(result['serial_holder_pids'],[77])
            self.assertEqual(result['owners'][0]['listening_ports'],[9026])
            self.assertNotIn('sensitive-placeholder',str(result))
            self.assertFalse(result['serial_release_verified'])
            self.assertTrue(result['inspection_complete'])

    def test_permission_denial_never_looks_like_no_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.fixture(root)
            original=Path.iterdir
            def iterdir(path):
                if path.name=='fd':
                    raise PermissionError('test')
                return original(path)
            with patch.object(Path,'iterdir',iterdir):
                result=probe.inspect_owners(proc=root)
            self.assertEqual(result['unreadable_pids'],[77])
            self.assertFalse(result['inspection_complete'])
            self.assertFalse(result['serial_release_verified'])

    def test_missing_network_table_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.fixture(root)
            (root/'net/tcp6').unlink()
            result=probe.inspect_owners(proc=root)
            self.assertFalse(result['inspection_complete'])
            self.assertTrue(result['network_tables_unreadable'])


class SettledScanTests(unittest.TestCase):
    """Bounded rescan: a vanished pid cannot hold the device; every other defect refuses at once."""
    DEVICE = '/dev/null'  # a real character device on macOS and Linux; never opened

    def tree(self, root, extra_listener=False):
        (root/'net').mkdir()
        rows = '0: 00000000:2342 00000000:0000 0A 0 0 0 0 0 123\n'
        if extra_listener:  # a listener on 8080 that no visible process owns
            rows += '1: 00000000:1F90 00000000:0000 0A 0 0 0 0 0 999\n'
        (root/'net/tcp').write_text('header\n' + rows)
        (root/'net/tcp6').write_text('header\n')
        service = root/'50'
        (service/'fd').mkdir(parents=True)
        (service/'comm').write_text('python3\n')
        (service/'cgroup').write_text('0::/system.slice/guard.service\n')
        (service/'fd/5').symlink_to('socket:[123]')
        return service

    def vanishing(self, root, pid=88):
        # Listed in /proc, fd table gone when read: FileNotFoundError on iterdir, as on the Pi.
        (root/str(pid)).mkdir()
        return root/str(pid)

    def holder(self, root, pid=77):
        process = root/str(pid)
        (process/'fd').mkdir(parents=True)
        (process/'comm').write_text('minicom\n')
        (process/'cgroup').write_text('0::/user.slice\n')
        (process/'fd/3').symlink_to(self.DEVICE)
        return process

    def test_vanished_pid_clears_on_rescan_settles_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.tree(root)
            ghost = self.vanishing(root)
            pauses = []
            def sleep(seconds):
                pauses.append(seconds)
                if len(pauses) == 2:   # clean from attempt 3 on
                    ghost.rmdir()
            result = probe.inspect_owners_settled(self.DEVICE, proc=root, sleep=sleep)
            self.assertTrue(result['usable'])
            self.assertEqual(result['settled_on'], 'complete')
            self.assertEqual(result['attempts'], 3)
            self.assertTrue(result['inspection_complete'])
            self.assertEqual(result['serial_holder_pids'], [])
            self.assertEqual(pauses, [0.05, 0.05])

    def test_vanished_only_on_every_attempt_is_usable_with_no_holder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.tree(root)
            self.vanishing(root)
            pauses = []
            result = probe.inspect_owners_settled(self.DEVICE, proc=root, sleep=pauses.append)
            self.assertTrue(result['usable'])
            self.assertEqual(result['settled_on'], 'vanished_only')
            self.assertEqual(result['attempts'], 5)
            self.assertEqual(len(pauses), 4)
            self.assertFalse(result['inspection_complete'])  # inspect_owners' own contract unchanged
            self.assertEqual(result['vanished_pids'], [88])
            self.assertEqual(result['serial_holder_pids'], [])

    def test_denied_pid_refuses_immediately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.tree(root)
            self.vanishing(root)
            original = Path.iterdir
            def iterdir(path):
                if path == root/'50'/'fd':
                    raise PermissionError('test')
                return original(path)
            pauses = []
            with patch.object(Path, 'iterdir', iterdir):
                result = probe.inspect_owners_settled(self.DEVICE, proc=root, sleep=pauses.append)
            self.assertFalse(result['usable'])
            self.assertEqual(result['settled_on'], 'unreadable_pids')
            self.assertEqual(result['attempts'], 1)
            self.assertEqual(pauses, [])

    def test_denied_pid_on_a_later_attempt_still_refuses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.tree(root)
            self.vanishing(root)
            original = Path.iterdir
            state = {'deny': False}
            def iterdir(path):
                if state['deny'] and path == root/'50'/'fd':
                    raise PermissionError('test')
                return original(path)
            def sleep(seconds):
                state['deny'] = True
            with patch.object(Path, 'iterdir', iterdir):
                result = probe.inspect_owners_settled(self.DEVICE, proc=root, sleep=sleep)
            self.assertFalse(result['usable'])
            self.assertEqual(result['settled_on'], 'unreadable_pids')
            self.assertEqual(result['attempts'], 2)

    def test_real_holder_with_vanished_pids_is_reported_and_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.tree(root)
            self.vanishing(root)
            self.holder(root)
            pauses = []
            result = probe.inspect_owners_settled(self.DEVICE, proc=root, sleep=pauses.append)
            self.assertFalse(result['usable'])
            self.assertEqual(result['settled_on'], 'foreign_holder')
            self.assertEqual(result['serial_holder_pids'], [77])
            self.assertEqual(result['attempts'], 1)
            self.assertEqual(pauses, [])

    def test_allowed_holder_with_vanished_pids_is_usable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.tree(root)
            self.vanishing(root)
            self.holder(root)
            result = probe.inspect_owners_settled(self.DEVICE, proc=root, sleep=lambda s: None,
                                                  allowed_holders={77})
            self.assertTrue(result['usable'])
            self.assertEqual(result['settled_on'], 'vanished_only')
            self.assertEqual(result['serial_holder_pids'], [77])

    def test_port_map_mismatch_refuses_immediately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.tree(root, extra_listener=True)
            self.vanishing(root)
            pauses = []
            result = probe.inspect_owners_settled(self.DEVICE, proc=root, sleep=pauses.append)
            self.assertEqual(result['listener_ports'], [8080, 9026])
            self.assertFalse(result['usable'])
            self.assertEqual(result['settled_on'], 'port_map_mismatch')
            self.assertEqual(result['attempts'], 1)
            self.assertEqual(pauses, [])

    def test_unreadable_network_table_refuses_immediately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.tree(root)
            self.vanishing(root)
            (root/'net/tcp6').unlink()
            result = probe.inspect_owners_settled(self.DEVICE, proc=root, sleep=lambda s: None)
            self.assertFalse(result['usable'])
            self.assertEqual(result['settled_on'], 'network_tables_unreadable')
            self.assertEqual(result['attempts'], 1)

    def test_non_character_device_refuses_immediately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.tree(root)
            self.vanishing(root)
            plain = root/'not-a-tty'
            plain.write_text('')
            result = probe.inspect_owners_settled(str(plain), proc=root, sleep=lambda s: None)
            self.assertFalse(result['usable'])
            self.assertEqual(result['settled_on'], 'device_not_character')
            self.assertEqual(result['attempts'], 1)

    def test_malformed_incomplete_result_fails_closed(self):
        # An incomplete result missing the defect fields must never read as vanished-only.
        with patch.object(probe, 'inspect_owners',
                          return_value={'device_is_character': True, 'inspection_complete': False,
                                        'serial_holder_pids': [], 'vanished_pids': [1]}):
            result = probe.inspect_owners_settled(self.DEVICE, sleep=lambda s: None)
        self.assertFalse(result['usable'])
        self.assertEqual(result['attempts'], 1)

    def test_worst_case_is_bounded_by_the_deadline_not_by_sleep_accuracy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.tree(root)
            self.vanishing(root)
            started = time.monotonic()
            result = probe.inspect_owners_settled(self.DEVICE, proc=root)  # real sleeps
            elapsed = time.monotonic() - started
            self.assertTrue(result['usable'])
            self.assertEqual(result['settled_on'], 'vanished_only')
            self.assertLessEqual(result['attempts'], 5)
            # Deadline 0.25 s plus at most one late wake-up; loose on purpose (wall clock under load).
            self.assertLess(elapsed, 0.6)

    def test_slow_sleep_stops_rescanning_at_the_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.tree(root)
            self.vanishing(root)
            # One pause that overruns the whole 0.25 s deadline: no second pause may follow.
            result = probe.inspect_owners_settled(self.DEVICE, proc=root, sleep=lambda s: time.sleep(0.3))
            self.assertTrue(result['usable'])
            self.assertEqual(result['settled_on'], 'vanished_only')
            self.assertEqual(result['attempts'], 2)

    def test_deadline_never_turns_a_later_refusal_into_a_pass(self):
        # The deadline only shortens the loop; a defect on any scan taken still refuses.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.tree(root)
            ghost = self.vanishing(root)
            def sleep(seconds):
                ghost.rmdir()
                self.holder(root)
            result = probe.inspect_owners_settled(self.DEVICE, proc=root, sleep=sleep)
            self.assertFalse(result['usable'])
            self.assertEqual(result['settled_on'], 'foreign_holder')
            self.assertEqual(result['attempts'], 2)
