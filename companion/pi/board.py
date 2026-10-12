# Codex, gpt-6-astra, Jetson seat, 2026-10-11 22:55:17 EDT
"""Board boundary. No vendor imports, device creation, or network on import."""
from dataclasses import dataclass
from typing import Protocol


class Board(Protocol):
    def execute(self, command: str, args: dict) -> dict: ...
    def snapshot(self) -> dict: ...
    def close(self) -> None: ...


class FakeBoard:
    """Records hypothetical operations. Never reports physical execution."""
    adapter_kind = 'fake'
    def __init__(self):
        self.commands = []

    def execute(self, command, args):
        self.commands.append((command, dict(args)))
        return {'executed': False, 'simulated': True,
                'command': command, 'played': False}

    def snapshot(self):
        return {'ok': False, 'reason': 'camera_unavailable', 'simulated': True}

    def close(self):
        pass


@dataclass(frozen=True)
class HeadCalibration:
    servo_id: int
    left: int
    right: int
    center: int

    def __post_init__(self):
        if type(self.servo_id) is not int or not 1 <= self.servo_id <= 6:
            raise ValueError('verified servo id required')
        if any(type(p) is not int or not 500 <= p <= 2500
               for p in (self.left, self.right, self.center)):
            raise ValueError('verified pulse values required')


class SDKBoard:
    """Staged SDK mapping. Requires a caller-owned SDK instance and calibration.

    No CLI constructs this class. A future approved driver factory must create
    the SDK only inside the supervised process, with sole ownership of serial.
    Killing an RPC proxy does not kill MasterPi or close its serial descriptor.
    """
    adapter_kind = 'sdk-unverified'
    def __init__(self, sdk, calibration: HeadCalibration):
        self.sdk, self.calibration = sdk, calibration

    def execute(self, command, args):
        if command == 'stop':
            self.sdk.set_motor_duty([[i, 0] for i in range(1, 5)])
        elif command in ('look_left', 'look_right', 'look_center'):
            pulse = getattr(self.calibration, command.removeprefix('look_'))
            self.sdk.pwm_servo_set_position(0.5, [[self.calibration.servo_id, pulse]])
        else:
            return {'error': 'speaker/display unavailable', 'executed': False,
                    'simulated': False}
        return {'executed': None, 'sdk_returned': True, 'simulated': False,
                'note': 'physical execution unmeasured; stop covers wheels only'}

    def snapshot(self):
        return {'ok': False, 'reason': 'camera_unavailable', 'simulated': False}

    def close(self):
        self.sdk.port.close()
