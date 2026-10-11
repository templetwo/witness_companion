"""Service contract and deterministic, hardware-free fake implementations."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timestamp must include a timezone')
    return parsed.astimezone(timezone.utc)


class WitnessServices(ABC):
    # Real services are being built on branch prep/companion-coordinator and
    # will implement this interface. Imports here must never load device SDKs.
    @abstractmethod
    def backend_provenance(self) -> dict:
        """Return backend_id, kind, model and where_computed for this adapter."""

    @abstractmethod
    def snapshot(self, max_age_seconds: float) -> dict:
        """Return ok, capture_time and image_path, or ok=False and reason."""

    @abstractmethod
    def speak(self, text: str) -> dict:
        """Request speech and describe whether it was actually played."""

    @abstractmethod
    def status(self) -> dict:
        """Return mic, camera, backend, controller and armed state."""

    @abstractmethod
    def propose_note(self, text: str, trace_id: str, issued_at: str) -> dict:
        """Append only to a configured local pending-notes JSON file."""

    @abstractmethod
    def request_action(self, name: str, params: dict) -> dict:
        """Return a verdict from an independent guard, without bypassing it."""


class FakeServices(WitnessServices):
    """In-memory fake; only note proposals write a file.

    A stored image is never recaptured or retimestamped. Arming is available
    only to test/host code through ``armed``; no tool can arm the guard.
    Nonces survive for this instance only. Notes have one writer per file.
    This guard simulates decisions, never physical execution or stopping.
    """

    def __init__(self, notes_path: Path = Path('pending-notes.json'),
                 sample_image_path: Optional[Path] = None,
                 capture_time: Optional[datetime] = None,
                 clock: Callable[[], datetime] = utc_now):
        self.notes_path = Path(notes_path)
        self.sample_image_path = (Path(sample_image_path)
                                  if sample_image_path is not None else None)
        if capture_time is not None and capture_time.tzinfo is None:
            raise ValueError('capture_time must include a timezone')
        self.capture_time = capture_time
        self.clock = clock
        self.speak_log = []
        self.action_log = []
        self.armed = False
        self._nonces = set()

    def backend_provenance(self) -> dict:
        return {'backend_id': 'fake', 'kind': 'none', 'model': None,
                'where_computed': 'unknown'}

    def snapshot(self, max_age_seconds: float) -> dict:
        captured = timestamp(self.capture_time) if self.capture_time else None
        result = {'ok': False, 'capture_time': captured, 'image_path': None,
                  'simulated': True}
        if self.sample_image_path is None or self.capture_time is None:
            return dict(result, reason='no_sample_image')
        age = (self.clock() - self.capture_time).total_seconds()
        if age < 0:
            return dict(result, reason='capture_time_in_future')
        if age > max_age_seconds:
            return dict(result, reason='stale_image', age_seconds=age)
        if not self.sample_image_path.is_file():
            return dict(result, reason='sample_image_missing')
        return dict(result, ok=True, image_path=str(self.sample_image_path),
                    age_seconds=age)

    def speak(self, text: str) -> dict:
        self.speak_log.append(text)
        return {'ok': True, 'recorded': True, 'played': False, 'simulated': True}

    def status(self) -> dict:
        return {'ok': True, 'mic': 'fake_unavailable',
                'camera': 'fake_stored_sample' if self.sample_image_path else 'fake_unavailable',
                'backend': self.backend_provenance(), 'controller': 'fake_guard',
                'armed': self.armed, 'simulated': True}

    def propose_note(self, text: str, trace_id: str, issued_at: str) -> dict:
        # Atomic replacement preserves existing notes on a failed write. No
        # Stack, model, network, or other persistence integration is invoked.
        path = self.notes_path
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            return {'ok': False, 'reason': 'notes_path_is_symlink'}
        notes = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        if not isinstance(notes, list):
            return {'ok': False, 'reason': 'invalid_pending_notes_file'}
        notes.append({'text': text, 'trace_id': trace_id, 'issued_at': issued_at,
                      'state': 'pending'})
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=path.parent, delete=False) as handle:
                temporary = handle.name
                json.dump(notes, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        return {'ok': True, 'state': 'pending', 'notes_path': str(path),
                'note_count': len(notes)}

    def request_action(self, name: str, params: dict) -> dict:
        def verdict(accepted: bool, reason: str) -> dict:
            return {'ok': accepted, 'verdict': 'accepted' if accepted else 'refused',
                    'reason': reason, 'executed': False, 'simulated': True}

        # Stop takes priority over all freshness, replay and armed checks.
        if name == 'stop':
            self.armed = False
            self.action_log.append({'name': name, 'params': dict(params)})
            return verdict(True, 'stop_accepted')
        if name != 'move':
            return verdict(False, 'unknown_action')
        if set(params) != {'nonce', 'expires_at', 'speed', 'duration_seconds'}:
            return verdict(False, 'invalid_action_params')
        nonce = params['nonce']
        if not isinstance(nonce, str) or not nonce.strip() or len(nonce) > 256:
            return verdict(False, 'invalid_nonce')
        try:
            expires = parse_timestamp(params['expires_at'])
        except (ValueError, TypeError, AttributeError):
            return verdict(False, 'invalid_expiry')
        if expires <= self.clock():
            return verdict(False, 'expired_request')
        if nonce in self._nonces:
            return verdict(False, 'duplicate_nonce')
        self._nonces.add(nonce)
        for key, lower, upper in [('speed', -1.0, 1.0), ('duration_seconds', 0.0, 5.0)]:
            value = params[key]
            if (type(value) not in (int, float) or not lower <= value <= upper
                    or not math.isfinite(value)):
                return verdict(False, 'out_of_bounds')
        if params['duration_seconds'] == 0:
            return verdict(False, 'out_of_bounds')
        if not self.armed:
            return verdict(False, 'disarmed')
        self.action_log.append({'name': name, 'params': dict(params)})
        return verdict(True, 'fake_motion_accepted')
