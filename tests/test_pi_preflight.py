# Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:09:44 EDT
import importlib.util
from pathlib import Path
import tempfile
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
