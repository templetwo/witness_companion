# Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:32:40 EDT
"""Run Witness tools with fake services or the explicit live Pi integration."""

import argparse
import os
from pathlib import Path
import sys

from .protocol import Server, serve
from .services import FakeServices, parse_timestamp
from .tools import ToolRegistry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--fake', action='store_true')
    mode.add_argument('--pi', action='store_true')
    parser.add_argument('--pi-host', default='10.42.0.23')
    parser.add_argument('--pi-port', type=int, default=9040)
    parser.add_argument('--issuer', default='jetson-coordinator')
    parser.add_argument('--key-env', default='WITNESS_GUARD_KEY_HEX')
    parser.add_argument('--state-dir', type=Path, default=Path.home()/'.local/state/witness')
    parser.add_argument('--camera-url')
    parser.add_argument('--piper')
    parser.add_argument('--voice')
    parser.add_argument('--notes-file', type=Path, default=Path('pending-notes.json'))
    parser.add_argument('--sample-image', type=Path)
    parser.add_argument('--capture-time', help='Stored sample capture time, ISO 8601 with timezone')
    args = parser.parse_args(argv)
    if args.pi:
        from .live import CameraSnapshot, LiveServices, LocalSpeech
        try:
            key = bytes.fromhex(os.environ[args.key_env])
            if len(key) < 32:
                raise ValueError()
        except (KeyError, ValueError):
            parser.error('live mode needs an existing key environment variable of at least 32 bytes')
        if bool(args.piper) != bool(args.voice):
            parser.error('--piper and --voice are required together')
        camera = CameraSnapshot(args.camera_url, args.state_dir/'camera') if args.camera_url else None
        speech = LocalSpeech(args.piper, args.voice) if args.piper else None
        services = LiveServices((args.pi_host, args.pi_port), args.issuer, key, args.state_dir, camera, speech)
        try:
            serve(Server(ToolRegistry(services)), sys.stdin.buffer, sys.stdout.buffer)
        finally:
            services.close()
        return 0
    if not args.fake:
        print('Select --pi for live services or --fake for the hardware-free scaffold.', file=sys.stderr)
        return 2
    if bool(args.sample_image) != bool(args.capture_time):
        parser.error('--sample-image and --capture-time must be supplied together')
    captured = None
    if args.capture_time:
        try:
            captured = parse_timestamp(args.capture_time)
        except ValueError:
            parser.error('--capture-time must be ISO 8601 with a timezone')
    services = FakeServices(args.notes_file, args.sample_image, captured)
    serve(Server(ToolRegistry(services)), sys.stdin.buffer, sys.stdout.buffer)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
