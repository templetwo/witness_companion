# Codex, gpt-6-astra, Jetson seat, 2026-10-12 10:13 EDT
import queue
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, MagicMock, call

from companion.persist import Journal
from companion.pi.client import PiLink
from companion.pi.hardware import HardwareBoard, SonarIO, validate_config
from companion.pi.service import serve_connection
from tests.test_hardware_handover import LOCKED, RecordingSDK, DRIVE


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.sdk = RecordingSDK()
        self.sdk.pwm_servo_queue = queue.Queue()
        self.sdk.bus_servo_queue = queue.Queue()
        self.sdk.buf_write = Mock()
        self.board = HardwareBoard(self.sdk, dict(LOCKED, wheel_duty_limit=20))
        self.addCleanup(self.board.close)

    def test_pwm_reply_is_commanded_pulse_not_physical_feedback(self):
        self.sdk.buf_write.side_effect = lambda *_: self.sdk.pwm_servo_queue.put(struct.pack('<BBH', 3, 5, 1500))
        result = self.board.probe({'kind': 'pwm_position', 'id': 3})
        self.assertTrue(result['ok'])
        self.assertEqual(result['value'], 1500)
        self.assertFalse(result['physical_position_verified'])
        self.assertEqual(result['meaning'], 'board_commanded_pulse')
        self.sdk.buf_write.assert_called_once_with(4, [5, 3])
        self.assertEqual(self.sdk.calls, [])

    def test_missing_reply_times_out_without_hanging(self):
        before = time.monotonic()
        self.assertFalse(self.board.probe({'kind': 'pwm_position', 'id': 1})['ok'])
        self.assertLess(time.monotonic()-before, .4)
        self.assertTrue(self.board.health()['alive'])

    def test_wrong_channel_and_malformed_reply_rejected(self):
        def inject(*_):
            self.sdk.pwm_servo_queue.put(b'bad')
            self.sdk.pwm_servo_queue.put(struct.pack('<BBH', 2, 5, 1500))
        self.sdk.buf_write.side_effect = inject
        self.assertFalse(self.board.probe({'kind': 'pwm_position', 'id': 1})['ok'])

    def test_stale_reply_is_discarded_before_query(self):
        self.sdk.pwm_servo_queue.put(struct.pack('<BBH', 1, 5, 1500))
        self.assertFalse(self.board.probe({'kind': 'pwm_position', 'id': 1})['ok'])

    def test_bus_zero_success_flag_and_failed_reply(self):
        for flag, ok in [(0, True), (1, False), (-1, False)]:
            self.sdk.buf_write.side_effect = lambda *_, f=flag: self.sdk.bus_servo_queue.put(struct.pack('<BBbh', 1, 5, f, 500))
            result=self.board.probe({'kind': 'bus_position', 'id': 1})
            self.assertEqual(result['ok'], ok)
            if ok: self.assertEqual(result['value'], 500)

    def test_no_arbitrary_diagnostic_writes_or_ids(self):
        for args in [{'kind': 'set_id', 'id': 1}, {'kind': 'pwm_position', 'id': 254},
                     {'kind': 'pwm_position', 'id': True}, {'kind': 'pwm_position', 'id': 1, 'pulse': 2500}]:
            with self.assertRaises(ValueError): self.board.probe(args)
        self.sdk.buf_write.assert_not_called()

    def test_probe_refused_during_drive(self):
        self.board.execute('drive', dict(DRIVE, duration_seconds=.5))
        self.assertEqual(self.board.probe({'kind':'pwm_position', 'id':1})['reason'], 'stop_drive_before_probe')
        self.sdk.buf_write.assert_not_called()

    def test_optional_sonar_error_does_not_kill_driver(self):
        self.board.sonar = Mock()
        self.board.sonar.distance.side_effect = [120, OSError('no sensor'), None, 150]
        self.assertEqual(self.board.status()['telemetry']['sonar_mm']['value'], 120)
        self.assertEqual(self.board.status()['sensor_errors']['sonar_mm'], 'sensor_io_error')
        self.assertEqual(self.board.status()['sensor_errors']['sonar_mm'], 'out_of_range_or_invalid')
        result = self.board.status()
        self.assertEqual(result['telemetry']['sonar_mm']['value'], 150)
        self.assertEqual(result['sensor_errors'], {})

    def test_output_mappings_use_bounded_arguments(self):
        self.board.config['outputs'] = ['led', 'buzzer', 'rgb', 'display', 'sonar_rgb']
        for name in ('set_led', 'set_buzzer', 'set_rgb', 'set_oled_text'):
            setattr(self.sdk, name, Mock())
        self.board.sonar = Mock()
        self.board.execute('led', {'on_seconds':.1,'off_seconds':.1,'repeat':2})
        self.board.execute('buzzer', {'frequency_hz':1000,'on_seconds':.1,'off_seconds':.1,'repeat':1})
        self.board.execute('rgb', {'index':1,'red':20,'green':0,'blue':0})
        self.board.execute('display', {'text':'TEMPLE TEST'})
        self.sdk.set_led.assert_called_once_with(.1,.1,2)
        self.sdk.set_buzzer.assert_called_once_with(1000,.1,.1,1)
        self.sdk.set_rgb.assert_called_once_with([[1,20,0,0]])
        self.sdk.set_oled_text.assert_called_once_with(0,'TEMPLE TEST')

    def test_calibrated_directions_override_source_assumptions(self):
        self.board.config['direction_signs']={'forward':[1,-1,1,-1]}
        self.board.execute('drive', DRIVE)
        self.assertEqual(self.sdk.calls[0], ('motor',[[1,20],[2,-20],[3,20],[4,-20]]))
        self.board.execute('stop',{})
        result=self.board.execute('drive',dict(DRIVE,direction='left'))
        self.assertEqual(result['reason'],'direction not commissioned')

    def test_calibration_rejects_unbounded_signs_and_unknown_directions(self):
        for directions in [{'fly':[1,1,1,1]}, {'forward':[20,1,1,1]},
                           {'forward':[True,1,1,1]}, {'forward':[0,0,0,0]},
                           {'forward':[1,1,1]}, []]:
            with self.assertRaises(ValueError):
                validate_config(dict(LOCKED,direction_signs=directions))

    def test_sonar_uses_separate_register_write_and_read_transactions(self):
        sonar=SonarIO.__new__(SonarIO)
        bus=MagicMock()
        bus.__enter__.return_value=bus
        sonar.bus=Mock(return_value=bus)
        sonar.msg=Mock()
        sonar.msg.write.return_value=b'pointer'
        sonar.msg.read.return_value=b'\x00\x02'
        self.assertEqual(sonar.distance(),512)
        self.assertEqual(bus.i2c_rdwr.call_args_list,[call(b'pointer'),call(b'\x00\x02')])


class LocalOperatorTests(unittest.TestCase):
    def test_stale_arm_discarded_and_stop_dominates_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            listener=socket.socket()
            listener.bind(('127.0.0.1',0))
            listener.listen(1)
            arm,stop=threading.Event(),threading.Event()
            arm.set()
            key=b'test-key-only'*3
            errors=[]
            def run():
                try:
                    conn,_=listener.accept()
                    serve_connection(conn,{'test':key},Journal(temp+'/journal'),operator_arm=arm,operator_stop=stop)
                except Exception as exc: errors.append(exc)
            thread=threading.Thread(target=run,daemon=True)
            thread.start()
            link=PiLink(listener.getsockname(),'test',key)
            try:
                self.assertFalse(link.status()['armed'])
                arm.set()
                time.sleep(.15)
                self.assertTrue(link.status()['armed'])
                arm.set(); stop.set()
                time.sleep(.15)
                self.assertFalse(link.status()['armed'])
                self.assertFalse(arm.is_set())
            finally:
                link.close(); listener.close(); thread.join(2)
            self.assertEqual(errors,[])
