"""Run the fake stdio scaffold; real services are deliberately unwired."""

import argparse
from pathlib import Path
import sys

from .protocol import Server, serve
from .services import FakeServices, parse_timestamp
from .tools import ToolRegistry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fake', action='store_true')
    parser.add_argument('--notes-file', type=Path, default=Path('pending-notes.json'))
    parser.add_argument('--sample-image', type=Path)
    parser.add_argument('--capture-time', help='Stored sample capture time, ISO 8601 with timezone')
    args = parser.parse_args(argv)
    if not args.fake:
        print('Real services are not wired. Use --fake for the hardware-free scaffold.', file=sys.stderr)
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
