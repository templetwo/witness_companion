# Codex, gpt-6-astra, Jetson seat, 2026-10-11 22:55:17 EDT
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from companion.guard import CommandGuard, sign
from companion.persist import Journal
from companion.schemas import ActionRequest, MalformedCommand
from companion.pi.board import FakeBoard, HeadCalibration, SDKBoard
from companion.pi.client import PiLink
from companion.pi.protocol import decode, encode, signed, verify
from companion.pi.service import GuardService, serve_connection
from companion.pi.supervisor import Supervisor


class HungBoard(FakeBoard):
    def execute(self, command, args):
        time.sleep(30)
        return super().execute(command, args)


class BrokenBoard(FakeBoard):
    def execute(self, command, args):
        raise OSError('fake driver crash')


class PiTests(unittest.TestCase):
    def setUp(self):
        self.key = os.urandom(32)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def action(self, command='look_left', seq=1, args=None):
        now = time.time()
        return sign(ActionRequest(command, seq, now, now+3, str(seq), 'jetson', args or {}), self.key)

    def supervisor(self, **kwargs):
        s = Supervisor(**kwargs)
        self.addCleanup(s.close)
        return s

    def service(self, armed=False, **kwargs):
        guard = CommandGuard({'jetson':self.key}, Journal(str(self.path/'guard.jsonl')))
        if armed:
            guard.arm()
        return GuardService(guard, self.supervisor(**kwargs), Journal(str(self.path/'notes.jsonl')))

    def test_disarmed_guard_stops_but_refuses_motion(self):
        service = self.service()
        self.assertEqual(service.action(self.action().to_dict())['reason'], 'disarmed')
        self.assertTrue(service.action({'command':'stop'})['ok'])
        self.assertFalse(service.guard.armed)

    def test_authenticated_action_and_duplicate(self):
        service = self.service(armed=True)
        raw = self.action().to_dict()
        first = service.action(raw)
        self.assertTrue(first['ok'])
        self.assertFalse(first['executed'])
        self.assertTrue(first['simulated'])
        self.assertEqual(service.action(raw)['reason'], 'duplicate')

    def test_invalid_expired_and_unsigned_actions_never_execute(self):
        service = self.service(armed=True)
        raw = self.action().to_dict()
        raw['mac'] = ''
        self.assertEqual(service.action(raw)['reason'], 'unauthorized')
        raw = self.action()
        raw.issued_at -= 20
        raw.expires_at -= 20
        self.assertEqual(service.action(sign(raw,self.key).to_dict())['reason'], 'expired')
        self.assertFalse(service.action({'command':'move'})['ok'])

    def test_signed_stop_disarms_and_replay_still_stops(self):
        service = self.service(armed=True)
        raw = self.action('stop').to_dict()
        self.assertTrue(service.action(raw)['ok'])
        self.assertFalse(service.guard.armed)
        self.assertTrue(service.action(raw)['ok'])

    def test_stop_forgery_does_not_poison_sequence(self):
        service = self.service(armed=True)
        raw = self.action('stop', 999).to_dict()
        raw['mac'] = ''
        self.assertTrue(service.action(raw)['ok'])
        self.assertEqual(service.guard.last_seq('jetson'), -1)

    def test_nan_infinity_huge_int_and_empty_nonce_rejected(self):
        for value in (math.nan, math.inf, -math.inf, 10**400):
            raw = self.action().to_dict()
            raw['expires_at'] = value
            with self.assertRaises(MalformedCommand):
                ActionRequest.from_dict(raw)
        raw = self.action().to_dict()
        raw['nonce'] = ''
        with self.assertRaises(MalformedCommand):
            ActionRequest.from_dict(raw)

    def test_driver_hang_is_killed(self):
        s = self.supervisor(factory=HungBoard, response_timeout=0.1)
        with self.assertRaises(ConnectionError):
            s.call('action', {'command':'stop','args':{}})
        s.process.join(1)
        self.assertEqual(s.process.exitcode, -signal.SIGKILL)

    def test_sigstop_is_killed(self):
        s = self.supervisor(response_timeout=0.1)
        os.kill(s.process.pid, signal.SIGSTOP)
        with self.assertRaises(ConnectionError):
            s.call('ping')
        s.process.join(1)
        self.assertEqual(s.process.exitcode, -signal.SIGKILL)

    def test_missed_heartbeat_kills_even_without_dispatch(self):
        s = self.supervisor(lease_seconds=0.5)
        s.process.join(1.5)
        self.assertEqual(s.process.exitcode, -signal.SIGKILL)
        with self.assertRaises(ConnectionError):
            s.renew()

    def test_valid_heartbeat_keeps_driver_alive(self):
        s = self.supervisor(lease_seconds=0.5)
        for _ in range(7):
            s.renew()
            self.assertTrue(s.call('ping')['alive'])
            time.sleep(0.1)
        self.assertTrue(s.healthy)

    def test_board_crash_is_unknown_and_latched(self):
        s = self.supervisor(factory=BrokenBoard)
        with self.assertRaises(ConnectionError):
            s.call('action', {'command':'stop','args':{}})
        self.assertIsNotNone(s.reason)
        with self.assertRaises(ConnectionError):
            s.call('ping')

    def test_five_tool_contract_and_no_remote_arm(self):
        service = self.service()
        def call(tool, args=None):
            return service.dispatch({'op':'tool','tool':tool,'arguments':args or {}})
        self.assertEqual(call('witness_status')['controller'],'fake')
        self.assertFalse(call('witness_snapshot')['ok'])
        self.assertFalse(call('witness_speak',{'action':self.action('say',1,{'text':'hello'}).to_dict()})['played'])
        note = call('witness_propose_note',{'text':'a proposed note'})
        self.assertFalse(note['approved'])
        self.assertEqual(service.notes.read()[0][0]['type'],'proposed_note')
        self.assertTrue(call('robot_request_action',{'action':{'command':'stop'}})['ok'])
        with self.assertRaises(ValueError):
            service.dispatch({'op':'arm'})
        with self.assertRaises(ValueError):
            call('witness_speak',{'action':self.action('look_left',2).to_dict()})

    def test_guard_restart_retains_replay_high_water(self):
        service = self.service(armed=True)
        action = self.action().to_dict()
        self.assertTrue(service.action(action)['ok'])
        new_guard = CommandGuard({'jetson':self.key},Journal(str(self.path/'guard.jsonl')))
        self.assertFalse(new_guard.armed)
        new_guard.arm()
        self.assertEqual(new_guard.check(action).reason,'duplicate')

    def test_wire_auth_session_tamper_expiry_and_reflection(self):
        request = signed(self.key,'session','client','jetson',0,{'op':'heartbeat'})
        self.assertEqual(verify(request,{'jetson':self.key},'session'),request)
        with self.assertRaises(ValueError):
            verify(request,{'jetson':self.key},'other')
        with self.assertRaises(ValueError):
            verify(request,{'jetson':self.key},'session',kind='response')
        request['payload'] = {'op':'arm'}
        with self.assertRaises(ValueError):
            verify(request,{'jetson':self.key},'session')
        request = signed(self.key,'session','client','jetson',0,{})
        request['expires_at'] -= 20
        with self.assertRaises(ValueError):
            verify(request,{'jetson':self.key},'session')

    def test_json_duplicate_and_nonfinite_rejected(self):
        for data in (b'{"op":"heartbeat","op":"stop"}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                decode(data)

    def test_sdk_mapping_with_recording_stub_only(self):
        class RecordingSDK:
            def __init__(self):
                self.calls = []
            def set_motor_duty(self, values):
                self.calls.append(('motor', values))
            def pwm_servo_set_position(self, seconds, values):
                self.calls.append(('servo', seconds, values))
        stub = RecordingSDK()
        board = SDKBoard(stub, HeadCalibration(6, 1400, 1600, 1500))
        self.assertIsNone(board.execute('stop', {})['executed'])
        board.execute('look_left', {})
        self.assertEqual(stub.calls, [('motor', [[1,0],[2,0],[3,0],[4,0]]),
                                     ('servo', 0.5, [[6,1400]])])
        self.assertIn('error', board.execute('say', {'text':'hello'}))
        with self.assertRaises(ValueError):
            HeadCalibration(6, 100, 1600, 1500)

    def start_server(self, lease=1.0):
        listener = socket.socket()
        listener.bind(('127.0.0.1',0))
        listener.listen(1)
        address = listener.getsockname()
        errors=[]
        def run():
            try:
                conn, _ = listener.accept()
                serve_connection(conn,{'jetson':self.key},Journal(str(self.path/'tcp.jsonl')),
                                 lease_seconds=lease)
            except Exception as exc:
                errors.append(exc)
            finally:
                listener.close()
        thread = threading.Thread(target=run,daemon=True)
        thread.start()
        self.addCleanup(thread.join,3)
        return address, thread, errors

    def test_real_tcp_roundtrip_and_heartbeat(self):
        address, thread, errors = self.start_server()
        link = PiLink(address,'jetson',self.key)
        try:
            time.sleep(1.2)
            status=link.status()
            self.assertTrue(status['simulated'])
            self.assertFalse(status['armed'])
            self.assertEqual(link.send(self.action()).status,'rejected')
            self.assertEqual(link.send(self.action('stop')).status,'applied')
        finally:
            link.close()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors,[])

    def test_partial_tcp_input_expires(self):
        address, thread, errors = self.start_server(0.5)
        with socket.create_connection(address) as conn:
            conn.settimeout(2)
            self.assertIn(b'session',conn.recv(4096))
            conn.sendall(b'{')
            self.assertEqual(conn.recv(4096),b'')
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors,[])

    def test_unauthenticated_heartbeat_cannot_extend_lease(self):
        address, thread, errors = self.start_server(0.5)
        with socket.create_connection(address) as conn:
            conn.settimeout(2)
            conn.recv(4096)
            conn.sendall(encode({'op':'heartbeat'}))
            response=decode(conn.recv(4096))
            self.assertFalse(response['ok'])
            self.assertEqual(conn.recv(4096),b'')
        thread.join(2)
        self.assertEqual(errors,[])

    def test_tcp_replay_and_emergency_stop(self):
        address, thread, errors = self.start_server()
        with socket.create_connection(address) as conn:
            conn.settimeout(2)
            stream = conn.makefile('rb')
            try:
                hello = decode(stream.readline())
                request = signed(self.key, hello['session'], 'test-client', 'jetson',
                                 0, {'op':'heartbeat'})
                conn.sendall(encode(request))
                self.assertTrue(verify(decode(stream.readline()), {'jetson':self.key},
                                       hello['session'], kind='response')['payload']['ok'])
                conn.sendall(encode(request))
                self.assertFalse(decode(stream.readline())['ok'])
                conn.sendall(encode({'op':'stop'}))
                stop = decode(stream.readline())
                self.assertTrue(stop['ok'])
                self.assertFalse(stop['executed'])
            finally:
                stream.close()
        thread.join(2)
        self.assertEqual(errors,[])

    def test_tcp_oversize_frame_closes_link(self):
        address, thread, errors = self.start_server()
        with socket.create_connection(address) as conn:
            conn.settimeout(2)
            conn.recv(4096)
            conn.sendall(b'x'*17000)
            try:
                self.assertEqual(conn.recv(4096),b'')
            except ConnectionResetError:
                pass
        thread.join(2)
        self.assertEqual(errors,[])

    def test_parent_death_kills_driver(self):
        code = ('from companion.pi.supervisor import Supervisor\n'
                'import time\n'
                'if __name__ == "__main__":\n'
                ' s=Supervisor(lease_seconds=5)\n'
                ' print(s.process.pid,flush=True)\n'
                ' time.sleep(20)\n')
        script = self.path/'parent_probe.py'
        script.write_text(code)
        env = dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        proc = subprocess.Popen([sys.executable,str(script)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
        try:
            pid = int(proc.stdout.readline())
            proc.kill()
            proc.wait(2)
            end=time.monotonic()+2
            state=''
            while time.monotonic()<end:
                try:
                    state=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split()[0]
                except FileNotFoundError:
                    state='gone'
                if state in ('Z','gone'):
                    break
                time.sleep(0.02)
            self.assertIn(state,('Z','gone'))
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            proc.stdout.close()
            proc.stderr.close()
