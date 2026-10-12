# Codex, gpt-6-astra, Jetson seat, 2026-10-12 10:00 EDT
"""Handover safety checks using recording stubs only; never open hardware."""
import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from companion.guard import CommandGuard, sign
from companion.pi.hardware import HardwareBoard, RealBoardFactory, validate_config
from companion.pi.service import GuardService
from companion.schemas import ActionRequest


LOCKED = {'wheel_duty_limit': 0, 'joints': {}, 'outputs': [], 'sonar': False}
DRIVE = {'direction': 'forward', 'duty': 20, 'duration_seconds': 0.05}


class RecordingSDK:
    def __init__(self):
        self.calls = []
        self.port = Mock()
        self.stopped = threading.Event()
        self.fail_stop = False
        self.battery = None

    def set_motor_duty(self, values):
        if all(duty == 0 for _, duty in values):
            if self.fail_stop:
                raise OSError('injected stop failure')
            self.stopped.set()
        self.calls.append(('motor', values))

    def pwm_servo_set_position(self, duration, values):
        self.calls.append(('pwm', duration, values))

    def bus_servo_set_position(self, duration, values):
        self.calls.append(('bus', duration, values))

    def bus_servo_stop(self, ids):
        self.calls.append(('bus_stop', ids))

    def get_battery(self):
        value, self.battery = self.battery, None
        return value

    def get_imu(self): return None
    def get_button(self): return None
    def get_gamepad(self): return None
    def get_sbus(self): return None


class HardwareHandoverTests(unittest.TestCase):
    def board(self, config=None):
        sdk = RecordingSDK()
        board = HardwareBoard(sdk, copy.deepcopy(config or LOCKED))
        self.addCleanup(board.close)
        return board, sdk

    def test_locked_configuration_rejects_every_drive_direction(self):
        from companion.controls import DIRECTIONS
        board, sdk = self.board()
        for direction in DIRECTIONS:
            result = board.execute('drive', dict(DRIVE, direction=direction))
            self.assertTrue(result['unavailable'])
        self.assertEqual(sdk.calls, [])

    def test_locked_configuration_rejects_joints_and_outputs(self):
        board, sdk = self.board()
        for name, args in [('joints', {'positions': {'head': 1500}, 'duration_seconds': .5}),
                           ('look_left', {}), ('display', {'text': 'hello'}),
                           ('rgb', {'index': 1, 'red': 10, 'green': 0, 'blue': 0})]:
            self.assertTrue(board.execute(name, args)['unavailable'])
        self.assertEqual(sdk.calls, [])

    def test_timed_drive_stops_all_wheels_without_another_command(self):
        board, sdk = self.board(dict(LOCKED, wheel_duty_limit=25))
        board.execute('drive', DRIVE)
        self.assertTrue(sdk.stopped.wait(.5))
        self.assertEqual(sdk.calls[-1], ('motor', [[1, 0], [2, 0], [3, 0], [4, 0]]))
        self.assertFalse(board.status()['drive_active'])

    def test_overlapping_drive_cannot_extend_deadline(self):
        board, sdk = self.board(dict(LOCKED, wheel_duty_limit=25))
        board.execute('drive', dict(DRIVE, duration_seconds=.2))
        deadline = board._deadline
        self.assertTrue(board.execute('drive', DRIVE)['unavailable'])
        self.assertEqual(board._deadline, deadline)
        self.assertEqual(len(sdk.calls), 1)

    def test_stop_cancels_drive(self):
        board, sdk = self.board(dict(LOCKED, wheel_duty_limit=25))
        board.execute('drive', DRIVE)
        board.execute('stop', {})
        self.assertIsNone(board._deadline)
        self.assertTrue(sdk.stopped.is_set())

    def test_timer_failure_is_reported_to_watchdog(self):
        board, sdk = self.board(dict(LOCKED, wheel_duty_limit=25))
        sdk.fail_stop = True
        board.execute('drive', DRIVE)
        board._worker.join(.5)
        try:
            with self.assertRaises(OSError): board.health()
            with self.assertRaises(OSError): board.execute('drive', DRIVE)
        finally:
            sdk.fail_stop = False

    def test_invalid_second_joint_prevents_first_joint_write(self):
        config = dict(LOCKED, joints={'head': {'bus': 'pwm', 'id': 1, 'min': 1400, 'max': 1600}})
        board, sdk = self.board(config)
        result = board.execute('joints', {'positions': {'head': 1500, 'unknown': 1500},
                                          'duration_seconds': .5})
        self.assertTrue(result['unavailable'])
        self.assertEqual(sdk.calls, [])

    def test_invalid_bounds_and_duplicate_channels_fail_closed(self):
        for config in [dict(LOCKED, wheel_duty_limit=True), dict(LOCKED, wheel_duty_limit=26),
                       dict(LOCKED, sonar='yes'), dict(LOCKED, outputs=['unknown']),
                       dict(LOCKED, joints={'head': {'bus': 'pwm', 'id': 1, 'min': 1600, 'max': 1400}}),
                       dict(LOCKED, joints={name: {'bus': 'pwm', 'id': 1, 'min': 1400, 'max': 1600}
                                            for name in ('a', 'b')})]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_missing_telemetry_stays_unknown_and_cached_sample_ages(self):
        board, sdk = self.board()
        self.assertEqual(board.status()['telemetry'], {})
        sdk.battery = 8100
        first = board.status()['telemetry']['battery_mv']
        second = board.status()['telemetry']['battery_mv']
        self.assertEqual(second['value'], 8100)
        self.assertEqual(first['read_at'], second['read_at'])
        self.assertFalse(second['sample_time_known'])
        self.assertGreaterEqual(second['age_since_read_seconds'], first['age_since_read_seconds'])

    def test_guard_and_locked_board_provide_independent_motion_barriers(self):
        board, sdk = self.board()
        key = b'test-only-key' * 3
        guard = CommandGuard({'test': key})
        supervisor = Mock()
        supervisor.call.side_effect = lambda op, data: board.execute(data['command'], data['args'])
        service = GuardService(guard, supervisor)
        now = time.time()
        action = sign(ActionRequest('drive', 1, now, now+2, 'nonce', 'test', DRIVE), key).to_dict()
        self.assertEqual(service.action(action)['reason'], 'disarmed')
        supervisor.call.assert_not_called()
        guard.arm()
        self.assertIn('commissioned duty limit', service.action(action)['reason'])
        self.assertEqual(sdk.calls, [])


class FactoryHandoverTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        config = Path(temp.name) / 'hardware.json'
        config.write_text(json.dumps(LOCKED))
        self.factory = RealBoardFactory('/not/a/vendor/module.py', str(config))
        self.empty = {'device_is_character': True, 'inspection_complete': True, 'serial_holder_pids': []}

    def test_existing_owner_or_incomplete_scan_prevents_import(self):
        for check in [dict(self.empty, serial_holder_pids=[123]),
                      dict(self.empty, inspection_complete=False),
                      dict(self.empty, device_is_character=False)]:
            with patch('scripts.pi_preflight.inspect_owners', return_value=check), \
                 patch('companion.pi.hardware.importlib.util.spec_from_file_location') as spec:
                with self.assertRaises(RuntimeError): self.factory()
                spec.assert_not_called()

    def run_factory(self, second, lock_error=False):
        sdk = Mock()
        sdk.port.fileno.return_value = 42
        module = SimpleNamespace(Board=Mock(return_value=sdk))
        with patch('scripts.pi_preflight.inspect_owners', side_effect=[self.empty, second]), \
             patch('companion.pi.hardware.importlib.util.spec_from_file_location', return_value=Mock()), \
             patch('companion.pi.hardware.importlib.util.module_from_spec', return_value=module), \
             patch('companion.pi.hardware.fcntl.flock', side_effect=OSError() if lock_error else None), \
             patch('companion.pi.hardware.fcntl.ioctl'), \
             patch('companion.pi.hardware.HardwareBoard') as board:
            if lock_error or second['serial_holder_pids'] != [os.getpid()]:
                with self.assertRaises((OSError, RuntimeError)): self.factory()
                sdk.port.close.assert_called_once()
                sdk.set_motor_duty.assert_not_called()
                board.assert_not_called()
            else:
                self.factory()
                module.Board.assert_called_once_with(device='/dev/ttyAMA0', timeout=.1)
                sdk.set_motor_duty.assert_called_once_with([[1, 0], [2, 0], [3, 0], [4, 0]])
                board.assert_called_once_with(sdk, LOCKED, None)

    def test_ownership_race_closes_descriptor_without_commands(self):
        self.run_factory(dict(self.empty, serial_holder_pids=[os.getpid(), 123]))

    def test_failed_lock_closes_descriptor_without_commands(self):
        self.run_factory(self.empty, lock_error=True)

    def test_exclusive_factory_initializes_one_board_with_zero_duty_only(self):
        self.run_factory(dict(self.empty, serial_holder_pids=[os.getpid()]))
