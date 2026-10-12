# Codex, gpt-6-astra, Jetson seat, 2026-10-11 22:55:17 EDT
"""Jetson TCP controller adapter. No automatic action retry or remote arming."""
import secrets
import socket
import threading
import time

from ..adapters.base import Controller
from ..schemas import ActionResult
from .protocol import MAX_FRAME, decode, encode, signed, verify


class PiLink(Controller):
    adapter_kind = 'remote-guard'

    def __init__(self, address, issuer, key, timeout=1.0, heartbeat_interval=0.25):
        if not 0.05 <= timeout <= 5 or not 0.05 <= heartbeat_interval <= 1:
            raise ValueError('invalid transport timing')
        self.issuer, self.key = issuer, key
        self.timeout = timeout
        self.client = secrets.token_hex(32)
        self._seq = 0
        self._buffer = b''
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.connected = False
        self.sock = socket.create_connection(address, timeout=timeout)
        self.sock.settimeout(timeout)
        try:
            hello = self._read()
            if not isinstance(hello, dict) or hello.get('protocol') != 1 or not isinstance(hello.get('session'), str):
                raise ValueError('invalid server greeting')
            self.session = hello['session']
            self._exchange({'op': 'heartbeat'})
            self.connected = True
        except BaseException:
            self.close()
            raise
        self._thread = threading.Thread(target=self._heartbeat, args=(heartbeat_interval,), daemon=True)
        self._thread.start()

    def _read(self):
        deadline = time.monotonic()+self.timeout
        while b'\n' not in self._buffer:
            left = deadline-time.monotonic()
            if left <= 0:
                raise TimeoutError('response timeout')
            self.sock.settimeout(left)
            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError('Pi link closed')
            self._buffer += data
            if len(self._buffer) > MAX_FRAME:
                raise ValueError('response too large')
        line, self._buffer = self._buffer.split(b'\n', 1)
        return decode(line)

    def _exchange(self, payload):
        with self._lock:
            try:
                seq = self._seq
                self._seq += 1
                self.sock.settimeout(self.timeout)
                self.sock.sendall(encode(signed(self.key, self.session, self.client,
                                               self.issuer, seq, payload)))
                response = verify(self._read(), {self.issuer: self.key}, self.session, kind='response')
                if response['client'] != self.client or response['seq'] != seq:
                    raise ValueError('response identity mismatch')
                return response['payload']
            except (OSError, ValueError, TypeError) as exc:
                self.close()
                raise ConnectionError('Pi link failed; outcome unknown, not retried') from exc

    def _heartbeat(self, interval):
        while not self._done.wait(interval):
            try:
                if not self._exchange({'op': 'heartbeat'}).get('ok'):
                    self.close()
                    return
            except ConnectionError:
                return

    def request(self, tool, arguments=None):
        return self._exchange({'op': 'tool', 'tool': tool, 'arguments': arguments or {}})

    def status(self):
        return self.request('witness_status')

    def send(self, action):
        result = self.request('robot_request_action', {'action': action.to_dict()})
        return ActionResult(action.command, action.seq, action.nonce,
                            result.get('verdict', 'unknown'), result.get('reason', ''))

    def close(self):
        self.connected = False
        self._done.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
