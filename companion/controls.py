# Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:32:40 EDT
"""Public bounded control vocabulary. No hardware dependencies."""
import math

DIRECTIONS = {
    'forward': (-1, 1, -1, 1), 'backward': (1, -1, 1, -1),
    'left': (1, 1, -1, -1), 'right': (-1, -1, 1, 1),
    'forward_left': (0, 1, -1, 0), 'forward_right': (-1, 0, 0, 1),
    'backward_left': (1, 0, 0, -1), 'backward_right': (0, -1, 1, 0),
    'turn_left': (1, 1, 1, 1), 'turn_right': (-1, -1, -1, -1),
}
EXTRA_MOTION = frozenset({'drive', 'joints'})
EXTRA_OUTPUT = frozenset({'led', 'rgb', 'buzzer', 'sonar_rgb'})


def number(value, low, high, integer=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or not low <= value <= high or not math.isfinite(value)):
        raise ValueError('control value out of bounds')


def validate(command, args):
    fields = {
        'drive': {'direction', 'duty', 'duration_seconds'},
        'joints': {'positions', 'duration_seconds'},
        'led': {'on_seconds', 'off_seconds', 'repeat'},
        'buzzer': {'frequency_hz', 'on_seconds', 'off_seconds', 'repeat'},
        'rgb': {'index', 'red', 'green', 'blue'},
        'sonar_rgb': {'index', 'red', 'green', 'blue'},
    }
    if set(args) != fields[command]:
        raise ValueError('incorrect control arguments')
    if command == 'drive':
        if not isinstance(args['direction'], str) or args['direction'] not in DIRECTIONS:
            raise ValueError('unknown drive direction')
        number(args['duty'], 1, 25, True)
        number(args['duration_seconds'], 0.05, 0.5)
    elif command == 'joints':
        number(args['duration_seconds'], 0.1, 1)
        positions = args['positions']
        if not isinstance(positions, dict) or not 1 <= len(positions) <= 6:
            raise ValueError('one to six named joints required')
        for name, pulse in positions.items():
            if not isinstance(name, str) or not name.isidentifier() or len(name) > 32:
                raise ValueError('invalid joint name')
            number(pulse, 0, 2500, True)
    elif command in ('rgb', 'sonar_rgb'):
        number(args['index'], 1, 2, True)
        for channel in ('red', 'green', 'blue'):
            number(args[channel], 0, 255, True)
    else:
        number(args['on_seconds'], 0.01, 1)
        number(args['off_seconds'], 0, 1)
        number(args['repeat'], 1, 3, True)
        if command == 'buzzer':
            number(args['frequency_hz'], 100, 3000, True)
