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
import queue
from pathlib import Path
import struct
import termios
import threading
import time

from ..controls import DIRECTIONS, EXTRA_MOTION, EXTRA_OUTPUT, number, validate


def validate_config(config):
    required = {'wheel_duty_limit', 'joints', 'outputs', 'sonar'}
    if (not isinstance(config, dict) or not required <= set(config)
            or set(config) - required - {'direction_signs'}):
        raise ValueError('invalid hardware configuration fields')
    if 'direction_signs' in config:
        mapping = config['direction_signs']
        if not isinstance(mapping, dict) or set(mapping) - set(DIRECTIONS):
            raise ValueError('invalid calibrated direction names')
        for signs in mapping.values():
            if (not isinstance(signs, list) or len(signs) != 4
                    or any(type(x) is not int or x not in (-1, 0, 1) for x in signs)
                    or not any(signs)):
                raise ValueError('direction needs four bounded wheel signs')
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
        self._sensor_errors = {}
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
                directions = self.config.get('direction_signs', DIRECTIONS)
                if args['direction'] not in directions:
                    return self.unavailable('direction not commissioned')
                if args['duty'] > self.config['wheel_duty_limit']:
                    return self.unavailable('drive exceeds commissioned duty limit')
                if self._deadline is not None:
                    return self.unavailable('drive already active; wait or stop')
                # Deadline is set BEFORE the write so a slow write cannot extend it.
                self._deadline = time.monotonic() + args['duration_seconds']
                self.sdk.set_motor_duty([[i, sign * args['duty']]
                                        for i, sign in enumerate(directions[args['direction']], 1)])
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
                try:
                    value = self.sonar.distance()
                    if value is None:
                        self._sensor_errors['sonar_mm'] = 'out_of_range_or_invalid'
                    else:
                        self._telemetry['sonar_mm'] = (value, now, time.time())
                        self._sensor_errors.pop('sonar_mm', None)
                except OSError:
                    self._sensor_errors['sonar_mm'] = 'sensor_io_error'
            samples = {name: {'value': value, 'read_at': wall, 'read_at_monotonic': mono,
                              'age_since_read_seconds': now - mono,
                              'sample_time_known': False}
                       for name, (value, mono, wall) in self._telemetry.items()}
            return {'telemetry': samples, 'sensor_errors': dict(self._sensor_errors),
                    'drive_active': self._deadline is not None,
                    'capabilities': {'wheel_duty_limit': self.config['wheel_duty_limit'],
                        'directions': sorted(self.config.get('direction_signs', DIRECTIONS)),
                        'joints': self.config['joints'], 'outputs': self.config['outputs'],
                        'sonar': self.sonar is not None,
                        'legacy_modes': 'unavailable: require bounded reimplementation',
                        'arm_ik': 'unavailable: geometry and joint calibration unverified',
                        'stored_actions': 'unavailable: trajectories unaudited'},
                    'telemetry_note': 'Missing packets are unknown, not a dead-link diagnosis.'}

    def probe(self, args):
        """Bounded read-only board queries; no EEPROM, torque or position writes.

        PWM values report the board's commanded pulse, not measured joint angle.
        Bus replies carry a separate success flag. Never accept another channel's
        reply. Probes are refused while a timed drive is active.
        """
        queries = {
            'pwm_position': (4, 0x05, '<BBH', 'pwm_servo_queue'),
            'pwm_offset': (4, 0x09, '<BBb', 'pwm_servo_queue'),
            'bus_position': (5, 0x05, '<BBbh', 'bus_servo_queue'),
            'bus_voltage': (5, 0x07, '<BBbH', 'bus_servo_queue'),
            'bus_temperature': (5, 0x09, '<BBbB', 'bus_servo_queue'),
            'bus_torque': (5, 0x0D, '<BBbb', 'bus_servo_queue'),
            'bus_angle_limits': (5, 0x32, '<BBbHH', 'bus_servo_queue'),
        }
        if (not isinstance(args, dict) or set(args) != {'kind', 'id'}
                or not isinstance(args['kind'], str) or args['kind'] not in queries):
            raise ValueError('unknown diagnostic query')
        number(args['id'], 1, 6, True)
        function, command, fmt, queue_name = queries[args['kind']]
        with self._lock:
            if self._fault:
                raise OSError(self._fault)
            if self._deadline is not None:
                return {'ok': False, 'reason': 'stop_drive_before_probe'}
            replies = getattr(self.sdk, queue_name)
            # Bounded drain: only the one-slot SDK queues are used here.
            try:
                replies.get_nowait()
            except queue.Empty:
                pass
            self.sdk.buf_write(function, [command, args['id']])
            deadline = time.monotonic() + .15
            while time.monotonic() < deadline:
                try:
                    packet = replies.get(timeout=max(.001, deadline-time.monotonic()))
                except queue.Empty:
                    break
                if len(packet) != struct.calcsize(fmt):
                    continue
                fields = struct.unpack(fmt, packet)
                if fields[:2] != (args['id'], command):
                    continue
                if function == 5 and fields[2] != 0:
                    return {'ok': False, 'reason': 'device_reported_failure', **args}
                values = fields[2:] if function == 4 else fields[3:]
                return {'ok': True, **args, 'value': values[0] if len(values)==1 else list(values),
                        'physical_position_verified': False,
                        'meaning': ('board_commanded_pulse' if args['kind']=='pwm_position'
                                    else 'board_pwm_offset' if function==4 else 'bus_servo_reply')}
            return {'ok': False, 'reason': 'no_matching_reply_within_150ms', **args}

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
            # This device requires a STOP between register selection and read.
            # A combined repeated-start transaction returned misleading values
            # in the live comparison against the installed vendor protocol.
            bus.i2c_rdwr(self.msg.write(0x77, [0]))
            reply = self.msg.read(0x77, 2)
            bus.i2c_rdwr(reply)
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
        from scripts.pi_preflight import inspect_owners_settled
        config = validate_config(json.loads(Path(self.config_path).read_text()))
        # Bounded rescan (<= 5 scans, 50 ms pauses, 0.25 s deadline) tolerates only pids that exited mid-scan.
        check = inspect_owners_settled(self.device)
        if (not check['usable'] or not check['device_is_character']
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
            check = inspect_owners_settled(self.device, allowed_holders={os.getpid()})
            if not check['usable'] or set(check['serial_holder_pids']) != {os.getpid()}:
                raise RuntimeError('serial ownership changed')
            sdk.enable_reception()
            sdk.set_motor_duty([[i, 0] for i in range(1, 5)])
            return HardwareBoard(sdk, config, SonarIO() if config['sonar'] else None)
        except BaseException:
            sdk.port.close()
            raise
