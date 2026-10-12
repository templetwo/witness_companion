# Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:32:40 EDT
"""Jetson's five tools connected to the Pi guard and local camera/speech adapters."""
import base64
import fcntl
import http.client
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit

from companion.guard import sign
from companion.persist import Journal
from companion.pi.client import PiLink
from companion.schemas import ActionRequest, MalformedCommand, clean_text
from .services import FakeServices, WitnessServices


class CameraSnapshot:
    def __init__(self, url, directory):
        self.url, self.directory = urlsplit(url), Path(directory)
        if self.url.scheme != 'http' or not self.url.hostname or self.url.username:
            raise ValueError('camera must use an operator-configured private HTTP URL')

    def snapshot(self, max_age_seconds):
        conn = http.client.HTTPConnection(self.url.hostname, self.url.port or 80, timeout=2)
        try:
            target = self.url.path or '/'
            if self.url.query:
                target += '?' + self.url.query
            conn.request('GET', target)
            response = conn.getresponse()
            if response.status != 200 or response.getheader('Content-Type', '').split(';')[0] != 'image/jpeg':
                return {'ok': False, 'reason': 'camera_did_not_return_jpeg'}
            captured = response.getheader('X-Witness-Capture-Time')
            deadline = time.monotonic() + 2
            data = bytearray()
            while len(data) <= 1024 * 1024:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError('camera deadline')
                if conn.sock is not None:
                    conn.sock.settimeout(left)
                chunk = response.read1(min(65536, 1024 * 1024 + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) > 1024 * 1024 or not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):
                return {'ok': False, 'reason': 'invalid_or_oversize_jpeg'}
            received = time.time()
            age = received - float(captured) if captured is not None else None
            if age is not None and not 0 <= age <= max_age_seconds:
                return {'ok': False, 'reason': 'stale_or_invalid_camera_timestamp'}
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as output:
                pending = Path(output.name)
                output.write(data)
            destination = self.directory / 'latest.jpg'
            os.replace(pending, destination)
            return {'ok': True, 'image_path': str(destination),
                    'image_base64': base64.b64encode(data).decode('ascii'),
                    'capture_time': float(captured) if captured is not None else None,
                    'received_at': received, 'age_seconds': age,
                    'capture_time_verified': captured is not None,
                    'simulated': False}
        except (OSError, ValueError, http.client.HTTPException):
            return {'ok': False, 'reason': 'camera_unavailable'}
        finally:
            conn.close()


class LocalSpeech:
    def __init__(self, piper, model, player='/usr/bin/aplay'):
        self.piper, self.model, self.player = piper, model, player

    def speak(self, text):
        if not text or clean_text(text) != text:
            return {'ok': False, 'reason': 'speech_requires_clean_text_up_to_200_characters', 'played': False}
        if not all(Path(p).is_file() for p in (self.piper, self.model, self.player)):
            return {'ok': False, 'reason': 'local_speech_not_configured', 'played': False}
        try:
            with tempfile.TemporaryDirectory(prefix='witness-speech-') as temporary:
                wave = str(Path(temporary) / 'speech.wav')
                subprocess.run([self.piper, '--model', self.model, '--output_file', wave],
                               input=text.encode(), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=20, check=True)
                subprocess.run([self.player, '-q', wave], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=30, check=True)
            return {'ok': True, 'played': None, 'playback_completed': True,
                    'audibility_verified': False, 'simulated': False,
                    'speech_backend': 'local_piper'}
        except (OSError, subprocess.SubprocessError):
            return {'ok': False, 'reason': 'speech_failed', 'played': None}


class LiveServices(WitnessServices):
    def __init__(self, address, issuer, key, state_dir, camera=None, speech=None, link_factory=PiLink):
        self.address, self.issuer, self.key = address, issuer, key
        self.camera, self.speech = camera, speech
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lease = open(state_dir / 'tools.lock', 'a')
        fcntl.flock(self._lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.journal = Journal(str(state_dir / 'issued.jsonl'))
        self.notes = FakeServices(state_dir / 'pending-notes.json')
        self._lock = threading.Lock()
        self._link, self._attempted = None, False
        self._link_factory = link_factory

    def backend_provenance(self):
        return {'backend_id': 'witness-direct', 'kind': 'none', 'model': None,
                'where_computed': 'jetson'}

    def _connect(self):
        if self._link is None:
            if self._attempted:
                raise ConnectionError('restart tools to establish a new disarmed session')
            self._attempted = True
            self._link = self._link_factory(self.address, self.issuer, self.key)
        if not self._link.connected:
            raise ConnectionError('link lost; no automatic reconnect')
        return self._link

    def snapshot(self, max_age_seconds):
        if self.camera is None:
            return {'ok': False, 'reason': 'camera_not_configured'}
        return self.camera.snapshot(max_age_seconds)

    def speak(self, text):
        if self.speech is None:
            return {'ok': False, 'reason': 'speech_not_configured', 'played': False}
        return self.speech.speak(text)

    def status(self):
        with self._lock:
            try:
                result = self._connect().status()
            except ConnectionError:
                result = {'ok': False, 'armed': False, 'controller': 'unavailable',
                          'reason': 'Pi guard unavailable; restart tools after restoring link'}
            return dict(result, mic='unconfigured', camera='configured' if self.camera else 'unconfigured',
                        speech='configured' if self.speech else 'unconfigured',
                        backend=self.backend_provenance())

    def propose_note(self, text, trace_id, issued_at):
        with self._lock:
            return self.notes.propose_note(text, trace_id, issued_at)

    def request_action(self, name, params):
        with self._lock:
            now = time.time()
            seq = self.journal.max_seq('issued', self.issuer) + 1
            action = ActionRequest(name, seq, now, now + 2, secrets.token_hex(16),
                                   self.issuer, {} if name == 'stop' else params)
            try:
                action = ActionRequest.from_dict(sign(action, self.key).to_dict())
            except (MalformedCommand, ValueError, TypeError):
                return {'ok': False, 'verdict': 'rejected', 'reason': 'invalid_action', 'executed': False}
            # fsync before transmission, so uncertain outcomes are never replayed.
            self.journal.append({'type': 'issued', 'issuer': self.issuer, 'seq': seq,
                                 'nonce': action.nonce, 'command': name})
            try:
                return self._connect().request('robot_request_action', {'action': action.to_dict()})
            except ConnectionError:
                return {'ok': False, 'verdict': 'unknown', 'reason': 'link_failed_not_retried',
                        'executed': None}

    def close(self):
        if self._link is not None:
            self._link.close()
        self._lease.close()
