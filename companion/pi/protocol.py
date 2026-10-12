# Codex, gpt-6-astra, Jetson seat, 2026-10-11 22:55:17 EDT
"""Bounded JSON-line framing and authenticated, session-bound messages."""
import hashlib
import hmac
import json
import math
import time

MAX_FRAME = 16384
FIELDS = {'kind', 'session', 'client', 'issuer', 'seq', 'issued_at', 'expires_at', 'payload', 'mac'}


def canonical(message):
    return json.dumps({k: v for k, v in message.items() if k != 'mac'},
                      allow_nan=False, sort_keys=True, separators=(',', ':')).encode()


def signed(key, session, client, issuer, seq, payload, kind='request'):
    now = time.time()
    result = dict(kind=kind, session=session, client=client, issuer=issuer, seq=seq,
                  issued_at=now, expires_at=now+3, payload=payload)
    result['mac'] = hmac.new(key, canonical(result), hashlib.sha256).hexdigest()
    return result


def verify(raw, keys, session, kind='request'):
    if not isinstance(raw, dict) or set(raw) != FIELDS:
        raise ValueError('invalid envelope')
    if raw['session'] != session or raw['kind'] != kind:
        raise ValueError('wrong session or message kind')
    for field in ('issuer', 'client', 'mac'):
        if not isinstance(raw[field], str) or not 1 <= len(raw[field]) <= 256:
            raise ValueError('invalid identity')
    if not isinstance(raw['payload'], dict) or type(raw['seq']) is not int or raw['seq'] < 0:
        raise ValueError('invalid payload or sequence')
    for field in ('issued_at', 'expires_at'):
        try:
            if type(raw[field]) not in (int, float) or not math.isfinite(raw[field]):
                raise ValueError('invalid time')
        except OverflowError:
            raise ValueError('invalid time')
    now = time.time()
    if not 0 < raw['expires_at']-raw['issued_at'] <= 5 or raw['expires_at'] <= now or raw['issued_at'] > now+1:
        raise ValueError('expired or invalid time window')
    key = keys.get(raw['issuer'])
    if key is None or not hmac.compare_digest(raw['mac'], hmac.new(key, canonical(raw), hashlib.sha256).hexdigest()):
        raise ValueError('unauthorized')
    return raw


def encode(value):
    data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()+b'\n'
    if len(data) > MAX_FRAME:
        raise ValueError('frame too large')
    return data


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def decode(data):
    def bad_constant(value):
        raise ValueError('nonfinite JSON number')
    try:
        return json.loads(data, object_pairs_hook=_object, parse_constant=bad_constant)
    except (RecursionError, UnicodeError) as exc:
        raise ValueError('invalid JSON') from exc
