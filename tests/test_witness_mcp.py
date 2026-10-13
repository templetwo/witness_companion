# Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:32:40 EDT
"""Hardware-free contract, guard and real subprocess transport tests."""

from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid

from witness_mcp.protocol import MAX_MESSAGE_BYTES, PROTOCOL_VERSION, Server, serve
from witness_mcp.services import FakeServices, parse_timestamp, timestamp
from witness_mcp.tools import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 11, 21, 25, tzinfo=timezone.utc)
INIT = {'jsonrpc': '2.0', 'id': 0, 'method': 'initialize', 'params': {
    'protocolVersion': PROTOCOL_VERSION, 'capabilities': {},
    'clientInfo': {'name': 'unittest', 'version': '1'}}}


def request(method, params=None, request_id=1):
    result = {'jsonrpc': '2.0', 'id': request_id, 'method': method}
    if params is not None:
        result['params'] = params
    return result


def call(name, arguments=None, request_id=1):
    return request('tools/call', {'name': name, 'arguments': arguments or {}}, request_id)


def payload(response):
    return json.loads(response['result']['content'][0]['text'])


def move_params(**overrides):
    result = {'nonce': 'one', 'expires_at': timestamp(NOW + timedelta(seconds=30)),
              'speed': 0.25, 'duration_seconds': 1}
    result.update(overrides)
    return result


class Fixture(unittest.TestCase):
    def setUp(self):
        # All test writes, including subprocess notes, stay in this worktree.
        self.temp = tempfile.TemporaryDirectory(prefix='.mcp-test-', dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.notes = self.directory / 'pending-notes.json'
        self.sample = self.directory / 'sample.ppm'
        self.sample.write_bytes(b'P6\n1 1\n255\n\x00\x00\x00')
        self.services = FakeServices(self.notes, self.sample, NOW, clock=lambda: NOW)
        self.server = Server(ToolRegistry(self.services))

    def child(self, args, data=b''):
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        return subprocess.run([sys.executable, '-m', 'witness_mcp'] + args,
                              input=data, capture_output=True, cwd=self.directory,
                              env=env, timeout=10)

    def assert_trace(self, result):
        self.assertEqual(str(uuid.UUID(result['trace_id'])), result['trace_id'])
        self.assertIsNotNone(parse_timestamp(result['issued_at']).tzinfo)
        self.assertEqual(result['backend_provenance'], {
            'backend_id': 'fake', 'kind': 'none', 'model': None, 'where_computed': 'unknown'})


class Contract:
    """Identical externally visible checks in memory and over subprocess stdio."""

    def test_initialize_and_list_five_schemas(self):
        responses = self.exchange([request('tools/list')])
        self.assertEqual(responses[0]['result']['protocolVersion'], PROTOCOL_VERSION)
        tools = responses[1]['result']['tools']
        self.assertEqual({t['name'] for t in tools}, {
            'witness_snapshot', 'witness_speak', 'witness_status',
            'witness_propose_note', 'robot_request_action'})
        self.assertEqual(len(tools), 5)
        limits = {t['name']: t['inputSchema']['properties']['text']['maxLength']
                  for t in tools if 'text' in t['inputSchema']['properties']}
        self.assertEqual(limits, {'witness_speak': 200, 'witness_propose_note': 2000})
        for tool in tools:
            schema = tool['inputSchema']
            self.assertEqual(schema['type'], 'object')
            self.assertIsInstance(schema['properties'], dict)
            self.assertIsInstance(schema['required'], list)
            self.assertIs(schema['additionalProperties'], False)

    def test_all_five_results_are_traceable(self):
        responses = self.exchange([
            call('witness_snapshot', request_id='snapshot'),
            call('witness_speak', {'text': 'Hello, Witness.'}, 'speak'),
            call('witness_status', request_id='status'),
            call('witness_propose_note', {'text': 'Pending only.'}, 'note'),
            call('robot_request_action', {'name': 'stop', 'params': {}}, 'robot')])[1:]
        traces = []
        for response in responses:
            result = payload(response)
            self.assert_trace(result)
            traces.append(result['trace_id'])
        self.assertEqual(len(set(traces)), 5)
        self.assertEqual([r['id'] for r in responses], ['snapshot', 'speak', 'status', 'note', 'robot'])
        spoken = payload(responses[1])
        self.assertTrue(spoken['recorded'])
        self.assertFalse(spoken['played'])
        status = payload(responses[2])
        self.assertTrue({'mic', 'camera', 'backend', 'controller', 'armed'} <= status.keys())
        self.assertFalse(status['armed'])

    def test_stale_image_has_explicit_reason_and_original_time(self):
        result = payload(self.exchange([call('witness_snapshot')], stale=True)[1])
        self.assertFalse(result['ok'])
        self.assertEqual(result['reason'], 'stale_image')
        self.assertEqual(parse_timestamp(result['capture_time']), NOW - timedelta(days=365))
        self.assertIsNone(result['image_path'])
        self.assert_trace(result)

    def test_fresh_sample_is_returned(self):
        result = payload(self.exchange([call('witness_snapshot')])[1])
        self.assertTrue(result['ok'])
        self.assertEqual(result['image_path'], str(self.sample))
        self.assertLessEqual(result['age_seconds'], 5)

    def test_disarmed_motion_refused_and_expired_stop_accepted(self):
        params = move_params(expires_at=timestamp(datetime.now(timezone.utc) + timedelta(days=1)))
        responses = self.exchange([
            call('robot_request_action', {'name': 'move', 'params': params}),
            call('robot_request_action', {'name': 'stop', 'params': {
                'nonce': 'one', 'expires_at': '2000-01-01T00:00:00Z'}})])[1:]
        self.assertEqual(payload(responses[0])['reason'], 'disarmed')
        self.assertTrue(responses[0]['result']['isError'])
        self.assertEqual(payload(responses[1])['verdict'], 'accepted')
        self.assertFalse(payload(responses[1])['executed'])

    def test_unknown_tool_error(self):
        response = self.exchange([call('shell_exec')])[1]
        self.assertEqual(response['error']['code'], -32602)
        self.assertEqual(response['error']['message'], 'Unknown tool')

    def test_speech_201_characters_refused_at_schema(self):
        response = self.exchange([call('witness_speak', {'text': 'x' * 201})])[1]
        self.assertEqual(response['error']['code'], -32602)
        self.assertEqual(self.services.speak_log, [])

    def test_speech_200_characters_accepted(self):
        response = self.exchange([call('witness_speak', {'text': 'x' * 200})])[1]
        result = payload(response)
        self.assertTrue(result['ok'])
        self.assertTrue(result['recorded'])
        self.assertFalse(result['played'])

    def test_note_2001_characters_refused_at_schema(self):
        response = self.exchange([call('witness_propose_note', {'text': 'x' * 2001})])[1]
        self.assertEqual(response['error']['code'], -32602)
        self.assertFalse(self.notes.exists())

    def test_note_2000_characters_accepted_without_truncation(self):
        text = 'x' * 2000
        response = self.exchange([call('witness_propose_note', {'text': text})])[1]
        self.assertTrue(payload(response)['ok'])
        self.assertEqual(json.loads(self.notes.read_text())[0]['text'], text)

    def test_malformed_params_errors(self):
        invalid = [request('tools/call', []), request('tools/call', {'name': 9}),
                   request('tools/call', {'name': 'witness_status', 'arguments': None}),
                   call('witness_speak'), call('witness_speak', {'text': 123}),
                   call('witness_speak', {'text': '   '}),
                   call('witness_status', {'arm': True}),
                   call('witness_snapshot', {'max_age_seconds': True}),
                   call('witness_snapshot', {'max_age_seconds': -1}),
                   call('witness_snapshot', {'max_age_seconds': 10 ** 400}),
                   call('robot_request_action', {'name': 'move', 'params': []}),
                   call('witness_propose_note', {'text': 'x', 'path': '/elsewhere'})]
        for response in self.exchange(invalid)[1:]:
            self.assertEqual(response['error']['code'], -32602)
        self.assertFalse(self.notes.exists())

    def test_note_append_is_local_and_preserves_existing_notes(self):
        self.notes.write_text('[{"text": "existing"}]', encoding='utf-8')
        before = {p.relative_to(self.directory): p.read_bytes()
                  for p in self.directory.rglob('*') if p.is_file()}
        responses = self.exchange([call('witness_propose_note', {'text': 'A note: café.'}),
                                   call('witness_propose_note', {'text': 'Second.'})])[1:]
        notes = json.loads(self.notes.read_text(encoding='utf-8'))
        self.assertEqual(notes[0], {'text': 'existing'})
        self.assertEqual([n['text'] for n in notes[1:]], ['A note: café.', 'Second.'])
        for note, response in zip(notes[1:], responses):
            self.assertEqual(note['state'], 'pending')
            self.assertEqual(note['trace_id'], payload(response)['trace_id'])
            self.assertEqual(note['issued_at'], payload(response)['issued_at'])
        after = {p.relative_to(self.directory): p.read_bytes()
                 for p in self.directory.rglob('*') if p.is_file()}
        self.assertEqual(before.keys(), after.keys())
        self.assertEqual({p for p in before if before[p] != after[p]}, {Path('pending-notes.json')})

    def test_unknown_method_error(self):
        self.assertEqual(self.exchange([request('not/a/method')])[1]['error']['code'], -32601)

    def test_notifications_have_no_response_or_side_effect(self):
        notification = call('witness_propose_note', {'text': 'Must not be appended.'})
        del notification['id']
        responses = self.exchange([{'jsonrpc': '2.0', 'method': 'notifications/initialized'},
                                   notification, request('tools/list')])
        self.assertEqual(len(responses), 2)
        self.assertFalse(self.notes.exists())


class InProcessTests(Contract, Fixture):
    def exchange(self, requests, stale=False):
        if stale:
            self.services.capture_time = NOW - timedelta(days=365)
        data = b''.join(json.dumps(r).encode() + b'\n' for r in [INIT] + requests)
        output = io.BytesIO()
        serve(self.server, io.BytesIO(data), output)
        return [json.loads(line) for line in output.getvalue().splitlines()]


class SubprocessTests(Contract, Fixture):
    def exchange(self, requests, stale=False):
        captured = NOW - timedelta(days=365) if stale else datetime.now(timezone.utc)
        data = b''.join(json.dumps(r).encode() + b'\n' for r in [INIT] + requests)
        child = self.child(['--fake', '--notes-file', str(self.notes),
                            '--sample-image', str(self.sample), '--capture-time', timestamp(captured)], data)
        self.assertEqual(child.returncode, 0, child.stderr.decode())
        self.assertEqual(child.stderr, b'')
        return [json.loads(line) for line in child.stdout.splitlines()]


class GuardTests(Fixture):
    def test_start_disarmed_and_speak_is_recorded_only(self):
        self.assertFalse(self.services.armed)
        self.assertFalse(self.services.speak('testing')['played'])
        self.assertEqual(self.services.speak_log, ['testing'])

    def test_stop_always_accepted_and_disarms(self):
        self.services.armed = True
        for _ in range(2):
            self.assertEqual(self.services.request_action('stop', {})['verdict'], 'accepted')
        self.assertFalse(self.services.armed)

    def test_unknown_action_cannot_arm(self):
        self.assertEqual(self.services.request_action('arm', {})['reason'], 'unknown_action')
        self.assertFalse(self.services.armed)

    def test_expiry_rejected_even_when_armed(self):
        self.services.armed = True
        result = self.services.request_action('move', move_params(expires_at=timestamp(NOW)))
        self.assertEqual(result['reason'], 'expired_request')
        self.assertEqual(self.services.action_log, [])

    def test_duplicate_nonce_rejected_even_after_disarmed_refusal(self):
        params = move_params()
        self.assertEqual(self.services.request_action('move', params)['reason'], 'disarmed')
        self.services.armed = True
        self.assertEqual(self.services.request_action('move', params)['reason'], 'duplicate_nonce')
        self.assertEqual(self.services.action_log, [])

    def test_fake_armed_motion_is_bounded_and_never_executed(self):
        self.services.armed = True
        result = self.services.request_action('move', move_params())
        self.assertEqual(result['verdict'], 'accepted')
        self.assertFalse(result['executed'])
        self.assertEqual(len(self.services.action_log), 1)
        self.assertEqual(self.services.request_action('move', move_params())['reason'], 'duplicate_nonce')

    def test_out_of_bounds_motion_refused(self):
        self.services.armed = True
        for index, change in enumerate([{'speed': 2}, {'duration_seconds': 6},
                                        {'duration_seconds': 0}, {'speed': True},
                                        {'speed': float('nan')}, {'speed': float('inf')},
                                        {'speed': 10 ** 400}]):
            with self.subTest(change=change):
                result = self.services.request_action('move', move_params(nonce=str(index), **change))
                self.assertEqual(result['reason'], 'out_of_bounds')
        self.assertEqual(self.services.action_log, [])

    def test_invalid_expiry_nonce_and_missing_params(self):
        for params, reason in [(move_params(expires_at='2026-10-11'), 'invalid_expiry'),
                               (move_params(expires_at=None), 'invalid_expiry'),
                               (move_params(nonce=''), 'invalid_nonce'), ({}, 'invalid_action_params')]:
            self.assertEqual(self.services.request_action('move', params)['reason'], reason)


class FailureTests(Fixture):
    def test_missing_and_future_images_are_explicit_failures(self):
        self.services.capture_time = NOW + timedelta(seconds=1)
        self.assertEqual(self.services.snapshot(5)['reason'], 'capture_time_in_future')
        self.services.capture_time = NOW
        self.sample.unlink()
        self.assertEqual(self.services.snapshot(5)['reason'], 'sample_image_missing')
        self.services.sample_image_path = None
        self.assertEqual(self.services.snapshot(5)['reason'], 'no_sample_image')

    def test_exact_freshness_boundary(self):
        self.services.capture_time = NOW - timedelta(seconds=5)
        self.assertTrue(self.services.snapshot(5)['ok'])
        self.assertEqual(self.services.snapshot(4.9)['reason'], 'stale_image')

    def test_service_exception_becomes_traceable_failure(self):
        def broken(text):
            raise RuntimeError('private adapter details')
        self.services.speak = broken
        result = ToolRegistry(self.services).call('witness_speak', {'text': 'hi'})
        self.assertTrue(result['isError'])
        body = json.loads(result['content'][0]['text'])
        self.assert_trace(body)
        self.assertEqual(body['reason'], 'service_failure')
        self.assertNotIn('private', json.dumps(result))

    def test_invalid_notes_are_preserved(self):
        for content in ('{invalid', '{}'):
            self.notes.write_text(content)
            result = ToolRegistry(self.services).call('witness_propose_note', {'text': 'new'})
            self.assertTrue(result['isError'])
            self.assertEqual(self.notes.read_text(), content)

    def test_symlink_note_target_is_not_written(self):
        target = self.directory / 'other.json'
        target.write_text('[]')
        self.notes.symlink_to(target)
        result = self.services.propose_note('new', 'trace', 'time')
        self.assertFalse(result['ok'])
        self.assertEqual(target.read_text(), '[]')

    def test_note_tool_opens_only_local_file(self):
        # Audit actual open attempts in a fresh process, including integration
        # boundaries. Python startup occurs before the hook is installed.
        script = '''
import json, pathlib, sys
from witness_mcp.services import FakeServices
from witness_mcp.tools import ToolRegistry
root = pathlib.Path.cwd().resolve()
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = pathlib.Path(args[0]).absolute()
        if path != root and root not in path.parents:
            raise AssertionError('file access outside local test directory')
    if event.startswith(('socket.', 'subprocess.')):
        raise AssertionError('external service access')
sys.addaudithook(audit)
result = ToolRegistry(FakeServices(root / 'pending-notes.json')).call(
    'witness_propose_note', {'text': 'local only'})
assert not result['isError'], result
assert sorted(p.name for p in root.iterdir()) == ['pending-notes.json', 'sample.ppm']
'''
        child = subprocess.run([sys.executable, '-c', script], cwd=self.directory,
                               env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1'),
                               capture_output=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr.decode())


class TransportTests(Fixture):
    def run_bytes(self, data):
        output = io.BytesIO()
        serve(self.server, io.BytesIO(data), output)
        return output.getvalue()

    def test_parse_error_then_recovery(self):
        raw = self.run_bytes(b'{bad json\n' + json.dumps(INIT).encode() + b'\n')
        responses = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual(responses[0]['error']['code'], -32700)
        self.assertIn('result', responses[1])

    def test_invalid_json_constants_duplicates_and_encoding(self):
        for message in (b'{"id": NaN}', b'{"id":1,"id":2}', b'\xff'):
            self.assertEqual(json.loads(self.run_bytes(message + b'\n'))['error']['code'], -32700)

    def test_invalid_request_envelopes(self):
        for value in ([], 7, None, {'jsonrpc': '1.0', 'method': 'initialize', 'id': 1},
                      {'jsonrpc': '2.0', 'method': 7, 'id': 1},
                      {'jsonrpc': '2.0', 'method': 'initialize', 'id': True}):
            self.assertEqual(self.server.handle(value)['error']['code'], -32600)

    def test_initialize_requires_valid_params_and_tools_require_initialize(self):
        self.assertEqual(self.server.handle(request('initialize', {}))['error']['code'], -32602)
        self.assertEqual(self.server.handle(request('tools/list'))['error']['code'], -32002)

    def test_initialize_offers_supported_version(self):
        init = dict(INIT, params=dict(INIT['params'], protocolVersion='unknown'))
        self.assertEqual(self.server.handle(init)['result']['protocolVersion'], PROTOCOL_VERSION)

    def test_null_request_id_receives_response(self):
        self.assertIsNone(self.server.handle(dict(INIT, id=None))['id'])

    def test_content_length_framing_and_unicode_byte_count(self):
        self.server.handle(INIT)
        message = json.dumps(call('witness_speak', {'text': 'Héllo 世界'}), ensure_ascii=False).encode()
        framed = ('Content-Length: %d\r\nContent-Type: application/json\r\n\r\n' % len(message)).encode() + message
        raw = self.run_bytes(framed)
        header, body = raw.split(b'\r\n\r\n', 1)
        self.assertEqual(int(header.split(b':')[1]), len(body))
        self.assertTrue(payload(json.loads(body))['ok'])
        self.assertEqual(self.services.speak_log, ['Héllo 世界'])

    def test_framing_in_subprocess(self):
        data = b''
        for req in [INIT, request('tools/list')]:
            body = json.dumps(req).encode()
            data += ('Content-Length: %d\r\n\r\n' % len(body)).encode() + body
        child = self.child(['--fake'], data)
        self.assertEqual(child.returncode, 0, child.stderr.decode())
        remaining = child.stdout
        responses = []
        while remaining:
            head, rest = remaining.split(b'\r\n\r\n', 1)
            size = int(head.split(b':')[1])
            responses.append(json.loads(rest[:size]))
            remaining = rest[size:]
        self.assertEqual(len(responses[1]['result']['tools']), 5)

    def test_bad_framing_is_parse_error(self):
        for raw in (b'Content-Length: x\r\n\r\n', b'Content-Length: 5\r\n\r\n{}',
                    b'Content-Length: 0\r\n\r\n',
                    b'Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}',
                    b'x' * (MAX_MESSAGE_BYTES + 1)):
            self.assertEqual(json.loads(self.run_bytes(raw))['error']['code'], -32700)

    def test_default_fake_has_no_image(self):
        data = b''.join(json.dumps(r).encode() + b'\n' for r in [INIT, call('witness_snapshot')])
        child = self.child(['--fake'], data)
        self.assertEqual(child.returncode, 0)
        result = payload(json.loads(child.stdout.splitlines()[1]))
        self.assertEqual(result['reason'], 'no_sample_image')
        self.assert_trace(result)

    def test_without_fake_exits_nonzero_without_stdout(self):
        child = self.child([])
        self.assertNotEqual(child.returncode, 0)
        self.assertEqual(child.stdout, b'')
        self.assertIn(b'Select --pi', child.stderr)

    def test_sample_requires_explicit_timestamp(self):
        child = self.child(['--fake', '--sample-image', str(self.sample)])
        self.assertNotEqual(child.returncode, 0)
        self.assertEqual(child.stdout, b'')

    def test_imports_do_not_load_device_packages(self):
        script = '''
import sys
import witness_mcp
import witness_mcp.protocol, witness_mcp.tools, witness_mcp.services, witness_mcp.__main__
assert not ({'cv2', 'sounddevice', 'serial', 'pyaudio'} & set(sys.modules))
'''
        child = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                               capture_output=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr.decode())


if __name__ == '__main__':
    unittest.main()
