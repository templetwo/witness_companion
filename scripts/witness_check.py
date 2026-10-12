# Codex, gpt-6-astra, Jetson seat, 2026-10-12 10:05 EDT
"""Read Temple's authenticated status; never arm or send action commands."""
import argparse
import json
from pathlib import Path
import time

from companion.pi.client import PiLink


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='10.42.0.23')
    parser.add_argument('--port', default=9040, type=int)
    parser.add_argument('--key-file', default=str(Path.home()/'.config/temple-witness/guard.key'))
    parser.add_argument('--hold-seconds', default=0, type=float)
    args = parser.parse_args()
    if not 0 <= args.hold_seconds <= 300:
        parser.error('hold must be 0..300 seconds')
    key = bytes.fromhex(Path(args.key_file).read_text().strip())
    link = PiLink((args.host, args.port), 'jetson-coordinator', key, timeout=2.5)
    try:
        print(json.dumps(link.status(), sort_keys=True), flush=True)
        end = time.monotonic() + args.hold_seconds
        while time.monotonic() < end:
            time.sleep(min(1, max(0, end-time.monotonic())))
            print(json.dumps(link.status(), sort_keys=True), flush=True)
    finally:
        link.close()


if __name__ == '__main__':
    main()
