# Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:32:40 EDT
"""Single-owner SDK adapter, constructed only by an explicit real service.

No vendor import at module load. Board writes and the timed wheel stop share
one lock. A missed stop or blocked call trips the separate process supervisor.
"""
from dataclasses import dataclass
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import termios
import threading
import time

from ..controls import DIRECTIONS, EXTRA_MOTION, EXTRA_OUTPUT, number, validate


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {'wheel_duty_limit', 'joints', 'outputs', 'sonar'}:
        raise ValueError('invalid hardware configuration fields')
    number(config['wheel_duty_limit'], 0, 25, True)
    if type(config['sonar']) is not bool:
        raise ValueError('sonar must be a boolean')
    if (not isinstance(config['outputs'], list)
            or any(x not in ('led', 'rgb', 'buzzer', 'display', 'sonar_rgb') for x in config['outputs'])):
        raise ValueError('invalid configured outputs')
    if not isinstance(config['joints'], dict) or len(config['joints']) > 6:
        raise ValueError('invalid joint map')
    channels = set()
    for name, joint in config['joints'].items():
        if not name.isidentifier() or len(name) > 32 or not isinstance(joint, dict):
            raise ValueError('invalid joint name')
        if set(joint) != {'bus', 'id', 'min', 'max'} or joint['bus'] not in ('pwm', 'bus'):
            raise ValueError('joint requires bus, id, min, max')
        number(joint['id'], 1, 6, True)
        low, high = (500, 2500) if joint['bus'] == 'pwm' else (0, 1000)
        number(joint['min'], low, high, True)
        number(joint['max'], joint['min'], high, True)
        channel = (joint['bus'], joint['id'])
        if channel in channels:
            raise ValueError('joint channels must be unique')
        channels.add(channel)
    return config


class HardwareBoard:
    adapter_kind = 'exclusive-sdk'

    def __init__(self, sdk, config, sonar=None):
        self.config = validate_config(config)
        self.sdk, self.sonar = sdk, sonar
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._deadline = None
        self._fault = None
        self._telemetry = {}
        self._worker = threading.Thread(target=self._timed_stop, daemon=True)
        self._worker.start()

    def _stop_wheels(self):
        self.sdk.set_motor_duty([[i, 0] for i in range(1, 5)])
        self._deadline = None

    def _timed_stop(self):
        while not self._closed.wait(0.01):
            try:
                with self._lock:
                    if self._deadline is not None and time.monotonic() >= self._deadline:
                        self._stop_wheels()
            except Exception:
                self._fault = 'timed stop failed'
                return

    def execute(self, command, args):
        if command in EXTRA_MOTION | EXTRA_OUTPUT:
            validate(command, args)
        with self._lock:
            if self._fault:
                raise OSError(self._fault)
            if command == 'stop':
                self._stop_wheels()
                ids = [j['id'] for j in self.config['joints'].values() if j['bus'] == 'bus']
                if ids:
                    self.sdk.bus_servo_stop(ids)
            elif command == 'drive':
                if args['duty'] > self.config['wheel_duty_limit']:
                    return self.unavailable('drive exceeds commissioned duty limit')
                if self._deadline is not None:
                    return self.unavailable('drive already active; wait or stop')
                # Deadline is set BEFORE the write so a slow write cannot extend it.
                self._deadline = time.monotonic() + args['duration_seconds']
                self.sdk.set_motor_duty([[i, sign * args['duty']]
                                        for i, sign in enumerate(DIRECTIONS[args['direction']], 1)])
            elif command == 'joints':
                groups = {'pwm': [], 'bus': []}
                for name, pulse in args['positions'].items():
                    joint = self.config['joints'].get(name)
                    if joint is None or not joint['min'] <= pulse <= joint['max']:
                        return self.unavailable('joint uncalibrated or outside commissioned limits')
                    groups[joint['bus']].append([joint['id'], pulse])
                # Validate ALL positions before sending any of them.
                for bus, positions in groups.items():
                    if positions:
                        method = (self.sdk.pwm_servo_set_position if bus == 'pwm'
                                  else self.sdk.bus_servo_set_position)
                        method(args['duration_seconds'], positions)
            elif command in self.config['outputs']:
                if command == 'display':
                    # Vendor packet limits use bytes, not characters.
                    text = args['text'].encode('utf-8')[:32].decode('utf-8', errors='ignore')
                    self.sdk.set_oled_text(0, text)
                elif command == 'led':
                    self.sdk.set_led(args['on_seconds'], args['off_seconds'], args['repeat'])
                elif command == 'buzzer':
                    self.sdk.set_buzzer(args['frequency_hz'], args['on_seconds'], args['off_seconds'], args['repeat'])
                elif command == 'rgb':
                    self.sdk.set_rgb([[args[x] for x in ('index', 'red', 'green', 'blue')]])
                elif command == 'sonar_rgb' and self.sonar is not None:
                    self.sonar.color(args)
                else:
                    return self.unavailable('sonar not configured')
            else:
                return self.unavailable('control not configured; use named calibrated joints for look/arm')
        return {'executed': None, 'sdk_returned': True, 'simulated': False,
                'physical_execution_verified': False,
                'stop_scope': 'wheels and configured bus joints; PWM travel cannot be cancelled'}

    @staticmethod
    def unavailable(reason):
        return {'unavailable': True, 'reason': reason, 'executed': False, 'simulated': False}

    def health(self):
        with self._lock:
            if self._fault:
                raise OSError(self._fault)
            return {"alive": True}

    def status(self):
        with self._lock:
            if self._fault:
                raise OSError(self._fault)
            now = time.monotonic()
            for name, getter in [('battery_mv', 'get_battery'), ('imu', 'get_imu'),
                                 ('button', 'get_button'), ('gamepad', 'get_gamepad'), ('sbus', 'get_sbus')]:
                value = getattr(self.sdk, getter)()
                if value is not None:
                    self._telemetry[name] = (value, now, time.time())
            if self.sonar is not None:
                self._telemetry['sonar_mm'] = (self.sonar.distance(), now, time.time())
            samples = {name: {'value': value, 'read_at': wall, 'age_since_read_seconds': now - mono,
                              'sample_time_known': False}
                       for name, (value, mono, wall) in self._telemetry.items()}
            return {'telemetry': samples, 'drive_active': self._deadline is not None,
                    'capabilities': {'wheel_duty_limit': self.config['wheel_duty_limit'],
                        'joints': self.config['joints'], 'outputs': self.config['outputs'],
                        'sonar': self.sonar is not None,
                        'legacy_modes': 'unavailable: require bounded reimplementation',
                        'arm_ik': 'unavailable: geometry and joint calibration unverified',
                        'stored_actions': 'unavailable: trajectories unaudited'},
                    'telemetry_note': 'Missing packets are unknown, not a dead-link diagnosis.'}

    def snapshot(self):
        return {'ok': False, 'reason': 'use separate camera endpoint', 'simulated': False}

    def close(self):
        self._closed.set()
        try:
            with self._lock:
                self._stop_wheels()
        finally:
            self.sdk.port.close()
        self._worker.join(timeout=0.2)


class SonarIO:
    """I2C sonar protocol; errors propagate instead of becoming valid distances."""
    def __init__(self):
        from smbus2 import SMBus, i2c_msg
        self.bus, self.msg = SMBus, i2c_msg

    def distance(self):
        with self.bus(1) as bus:
            reply = self.msg.read(0x77, 2)
            bus.i2c_rdwr(self.msg.write(0x77, [0]), reply)
            distance = int.from_bytes(bytes(reply), 'little')
            return distance if distance <= 5000 else None

    def color(self, args):
        with self.bus(1) as bus:
            bus.write_byte_data(0x77, 2, 0)
            for offset, channel in enumerate(('red', 'green', 'blue')):
                bus.write_byte_data(0x77, 3 + (args['index'] - 1) * 3 + offset, args[channel])


@dataclass(frozen=True)
class RealBoardFactory:
    sdk_path: str
    config_path: str
    device: str = '/dev/ttyAMA0'

    def __call__(self):
        # Read only the standalone SDK module, never MasterPi or mode imports.
        from scripts.pi_preflight import inspect_owners
        config = validate_config(json.loads(Path(self.config_path).read_text()))
        check = inspect_owners(self.device)
        if (not check['device_is_character'] or not check['inspection_complete']
                or check['serial_holder_pids']):
            raise RuntimeError('serial ownership preflight failed')
        spec = importlib.util.spec_from_file_location('witness_vendor_board', self.sdk_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sdk = module.Board(device=self.device, timeout=0.1)
        try:
            sdk.port.write_timeout = 0.1
            fcntl.flock(sdk.port.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.ioctl(sdk.port.fileno(), termios.TIOCEXCL)
            check = inspect_owners(self.device)
            if not check['inspection_complete'] or set(check['serial_holder_pids']) != {os.getpid()}:
                raise RuntimeError('serial ownership changed')
            sdk.enable_reception()
            sdk.set_motor_duty([[i, 0] for i in range(1, 5)])
            return HardwareBoard(sdk, config, SonarIO() if config['sonar'] else None)
        except BaseException:
            sdk.port.close()
            raise
