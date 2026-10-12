# Codex, gpt-6-astra, Jetson seat, 2026-10-11 22:55:17 EDT
"""Five-tool TCP service with an independent robot CommandGuard.

The CLI is fake-only. No remote operation arms the guard. Snapshot/status/notes
are not robot commands and never go through a motion mapping. Speech and robot
actions must carry the existing signed ActionRequest, checked on the Pi.
"""
import argparse
import fcntl
import os
import secrets
import socket
import time

from ..guard import CommandGuard
from ..persist import Journal
from ..schemas import clean_text
from .protocol import MAX_FRAME, decode, encode, signed, verify
from .supervisor import Supervisor

TOOLS = frozenset({'witness_snapshot', 'witness_speak', 'witness_status',
                   'witness_propose_note', 'robot_request_action'})


class GuardService:
    def __init__(self, guard, supervisor, notes=None):
        self.guard, self.supervisor, self.notes = guard, supervisor, notes

    def action(self, raw):
        decision = self.guard.check(raw)
        if not decision.accepted:
            return {'ok': False, 'verdict': 'rejected', 'reason': decision.reason,
                    'executed': False}
        result = self.supervisor.call('action', {'command': decision.command,
                    'args': {} if decision.command == 'stop' else raw['args']})
        if 'error' in result:
            self.guard.disarm()
            self.supervisor.trip('board operation failed')
            return dict(result, ok=False, verdict='unknown')
        return dict(result, ok=True, verdict='applied', reason=decision.reason)

    def dispatch(self, payload):
        if payload == {'op': 'heartbeat'}:
            self.supervisor.renew()
            self.supervisor.call('ping')
            return {'ok': True, 'driver_alive': True}
        if set(payload) != {'op', 'tool', 'arguments'} or payload['op'] != 'tool':
            raise ValueError('invalid operation; arming is local only')
        tool, args = payload['tool'], payload['arguments']
        if not isinstance(tool, str) or tool not in TOOLS or not isinstance(args, dict):
            raise ValueError('unknown tool or invalid arguments')
        if tool in ('robot_request_action', 'witness_speak'):
            if set(args) != {'action'}:
                raise ValueError('signed action required')
            if tool == 'witness_speak' and (not isinstance(args['action'], dict) or args['action'].get('command') != 'say'):
                raise ValueError('witness_speak requires signed say action')
            return self.action(args['action'])
        if tool == 'witness_propose_note':
            if set(args) != {'text'} or not isinstance(args['text'], str) or not args['text'].strip() or len(args['text']) > 2000:
                raise ValueError('bounded nonempty note required')
            if self.notes is None:
                return {'ok': False, 'reason': 'notes store unavailable'}
            note_id = secrets.token_hex(16)
            self.notes.append({'type': 'proposed_note', 'id': note_id,
                               'text': clean_text(args['text'], 2000), 'approved': False})
            return {'ok': True, 'note_id': note_id, 'approved': False}
        if args:
            raise ValueError('tool takes no arguments')
        if tool == 'witness_snapshot':
            return self.supervisor.call('snapshot')
        self.supervisor.call('ping')
        return {'ok': True, 'armed': self.guard.armed, 'driver_alive': True,
                'controller': self.supervisor.kind,
                'simulated': self.supervisor.kind == 'fake',
                'physical_motion_verified': False}


def serve_connection(conn, keys, journal, notes=None, lease_seconds=2.0,
                     response_timeout=0.5, factory=None):
    """One session, disarmed on entry. Caller must exclusively own the journal."""
    session = secrets.token_hex(32)
    guard = CommandGuard(keys, journal)
    options = dict(lease_seconds=lease_seconds, response_timeout=response_timeout)
    if factory is not None:
        options['factory'] = factory
    supervisor = Supervisor(**options)
    service = GuardService(guard, supervisor, notes)
    peer = None
    last_seq = -1
    pending = b''
    conn.settimeout(0.1)
    try:
        conn.sendall(encode({'session': session, 'protocol': 1, 'tools': sorted(TOOLS)}))
        while supervisor.healthy:
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                break
            pending += chunk
            if len(pending) > MAX_FRAME:
                raise ValueError('frame too large')
            while b'\n' in pending:
                frame, pending = pending.split(b'\n', 1)
                message = None
                try:
                    raw = decode(frame)
                    # Safety stop has no authentication/sequence requirement and
                    # never renews the link lease or consumes transport sequence.
                    if raw == {'op': 'stop'}:
                        response = service.action({'command': 'stop'})
                    else:
                        message = verify(raw, keys, session)
                        identity = (message['issuer'], message['client'])
                        if (peer is not None and peer != identity) or message['seq'] <= last_seq:
                            raise ValueError('replayed or different client')
                        peer, last_seq = identity, message['seq']
                        response = service.dispatch(message['payload'])
                    if not supervisor.healthy:
                        guard.disarm()
                    if message is not None:
                        response = signed(keys[message['issuer']], session, message['client'],
                                          message['issuer'], message['seq'], response, kind='response')
                except (ValueError, TypeError) as exc:
                    # Do not echo inputs, keys, MACs or text into diagnostics.
                    response = {'ok': False, 'reason': 'invalid or unauthorized request'}
                except (ConnectionError, OSError):
                    guard.disarm()
                    supervisor.trip('driver failure')
                    response = {'ok': False, 'verdict': 'unknown', 'reason': 'driver unavailable'}
                conn.sendall(encode(response))
                if not supervisor.healthy:
                    break
    except (OSError, ValueError):
        pass  # EOF, malformed flood, or stalled writer closes this one session.
    finally:
        guard.disarm()
        supervisor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fake', action='store_true', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--peer', default='127.0.0.1')
    parser.add_argument('--issuer', default='jetson-coordinator')
    parser.add_argument('--key-env', default='WITNESS_GUARD_KEY_HEX', help='name only; never a value')
    parser.add_argument('--journal', required=True)
    parser.add_argument('--notes-journal', required=True)
    args = parser.parse_args()
    try:
        key = bytes.fromhex(os.environ[args.key_env])
        if len(key) < 32:
            raise ValueError()
    except (KeyError, ValueError):
        parser.error('key environment variable missing or invalid; supply at least 32 bytes')
    # Transport is authenticated but unencrypted. Only the private robot link or
    # loopback is intended. Bind host and peer must be selected explicitly.
    journal, notes = Journal(args.journal), Journal(args.notes_journal)
    with open(args.journal+'.lock', 'a') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as listener:
            listener.bind((args.host, args.port))
            listener.listen(1)
            print('fake Pi service listening on %s:%s' % listener.getsockname(), flush=True)
            while True:
                conn, address = listener.accept()
                if address[0] != args.peer:
                    conn.close()
                    continue
                serve_connection(conn, {args.issuer: key}, journal, notes)


if __name__ == '__main__':
    main()
