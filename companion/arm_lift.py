# Codex, gpt-6-astra, Jetson seat, 2026-10-12 EDT
"""Pure planning for attended arm poses; no device access or vendor import.

Geometry and offsets are facts read from the installed Pi, recorded in
receipts/vendor-mapping-facts.json. Coordinates are predictions from commanded
pulses, never encoder feedback. This is not a collision model or general IK API.
"""
from math import acos, atan2, cos, degrees, hypot, isfinite, radians, sin

CHANNELS = (3, 4, 5, 6)
OFFSETS = {3: 59, 4: 72, 5: 63, 6: -95}
LINKS_CM = (9.3, 6.5, 6.2, 10.0)
DEGREES_PER_US = 180.0 / 2000.0


def validate_pulses(pulses):
    if not isinstance(pulses, dict) or set(pulses) != set(CHANNELS):
        raise ValueError('exactly PWM channels 3, 4, 5, 6 required')
    for channel, pulse in pulses.items():
        if (type(pulse) is not int or not 500 <= pulse <= 2500
                or not 500 <= pulse - OFFSETS[channel] <= 2500):
            raise ValueError('nominal or final offset pulse outside limits')


def forward(pulses):
    """Predict x/y/z cm and tool pitch degrees from commanded PWM pulses."""
    validate_pulses(pulses)
    nominal = {c: pulses[c] - OFFSETS[c] for c in CHANNELS}
    wrist = (nominal[3] - 1500) * DEGREES_PER_US
    elbow = (nominal[4] - 1500) * DEGREES_PER_US
    shoulder = 90 - (nominal[5] - 1500) * DEGREES_PER_US
    base = (nominal[6] - 500) * DEGREES_PER_US
    pitch = wrist + shoulder - elbow
    base_height, upper, forearm, tool = LINKS_CM
    radius = (upper * cos(radians(shoulder))
              + forearm * cos(radians(shoulder - elbow))
              + tool * cos(radians(pitch)))
    height = (base_height + upper * sin(radians(shoulder))
              + forearm * sin(radians(shoulder - elbow))
              + tool * sin(radians(pitch)))
    return {'x_cm': radius * cos(radians(base)),
            'y_cm': radius * sin(radians(base)),
            'z_cm': height, 'pitch_degrees': pitch}


def _inverse(pose):
    """One elbow branch, confined to the front half-plane, for this lift only."""
    x, y, z, pitch = (pose[k] for k in
                      ('x_cm', 'y_cm', 'z_cm', 'pitch_degrees'))
    if (any(type(v) not in (int, float) or not isfinite(v) for v in (x, y, z, pitch))
            or y <= 0 or z <= 0):
        raise ValueError('invalid front workspace target')
    base_height, upper, forearm, tool = LINKS_CM
    radial = hypot(x, y) - tool * cos(radians(pitch))
    height = z - base_height - tool * sin(radians(pitch))
    cosine = (radial**2 + height**2 - upper**2 - forearm**2) / (2 * upper * forearm)
    if not -1 <= cosine <= 1:
        raise ValueError('unreachable target')
    elbow_r = acos(cosine)
    shoulder_r = atan2(height, radial) + atan2(forearm * sin(elbow_r),
                                            upper + forearm * cos(elbow_r))
    elbow, shoulder = degrees(elbow_r), degrees(shoulder_r)
    wrist = pitch - shoulder + elbow
    base = degrees(atan2(y, x))
    nominal = {3: round(1500 + wrist / DEGREES_PER_US),
               4: round(1500 + elbow / DEGREES_PER_US),
               5: round(1500 + (90 - shoulder) / DEGREES_PER_US),
               6: round(500 + base / DEGREES_PER_US)}
    final = {c: nominal[c] + OFFSETS[c] for c in CHANNELS}
    validate_pulses(final)  # Includes the offsets; never silently clip a pose.
    return final


def _start_pose(start):
    validate_pulses(start)
    origin = forward(start)
    if (abs(origin['x_cm']) > 1 or not 5 <= origin['y_cm'] <= 7
            or not 17 <= origin['z_cm'] <= 19 or abs(origin['pitch_degrees']) > 5):
        raise ValueError('start outside attended upright commissioning region')
    return origin


def plan_lift(start, lift_cm=1.0):
    """Validate a <=1cm lift near the observed upright pose and its reverse.

    Samples approximate equal-duration linear PWM interpolation. Real servo
    tracking, backlash, obstructions and payload are not measured by this model.
    The caller must independently require attendance and use the hardware guard.
    """
    origin = _start_pose(start)
    if type(lift_cm) not in (int, float) or not isfinite(lift_cm) or not .25 <= lift_cm <= 1:
        raise ValueError('lift must be between 0.25 and 1cm')
    requested = dict(origin, z_cm=origin['z_cm'] + lift_cm)
    target = _inverse(requested)
    if target[6] != start[6] or max(abs(target[c] - start[c]) for c in CHANNELS) > 200:
        raise ValueError('joint change exceeds lift commissioning bound')
    samples = []
    for step in range(41):
        fraction = step / 40
        pulses = {c: round(start[c] + fraction * (target[c] - start[c])) for c in CHANNELS}
        pose = forward(pulses)
        if (hypot(pose['x_cm'] - origin['x_cm'], pose['y_cm'] - origin['y_cm']) > .15
                or not origin['z_cm'] - .05 <= pose['z_cm'] <= requested['z_cm'] + .05
                or abs(pose['pitch_degrees'] - origin['pitch_degrees']) > .2):
            raise ValueError('predicted transition leaves narrow lift corridor')
        samples.append({'fraction': fraction, 'pulses': pulses, 'predicted_pose': pose})
    achieved = forward(target)
    if abs(achieved['z_cm'] - requested['z_cm']) > .05:
        raise ValueError('target quantization exceeds tolerance')
    return {'start': dict(start), 'target': target, 'lift_cm': lift_cm,
            'predicted_start': origin, 'predicted_end': achieved,
            'duration_seconds': .8, 'samples': samples,
            'physical_position_verified': False,
            'scope': 'attended near-upright lift only; no collision or payload model'}


def plan_reach(start):
    """Fixed 5cm-forward/2cm-up commissioning pose, four bounded stages.

    Every stage takes 0.6s; the return traverses the same pulses in reverse.
    The guarded runner checks readback and battery between stages. This does
    not provide collision sensing or unrestricted Cartesian control.
    """
    return _plan_reach(start, 0)


def plan_side_reach(start, side):
    """Add 3cm nominal left/right travel to the fixed forward/up reach.

    Left is negative model x, right positive; physical direction needs an
    attended observation. This introduces at most 20 degrees of base rotation.
    """
    if not isinstance(side, str) or side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    return _plan_reach(start, -3 if side == 'left' else 3)


def _plan_reach(start, lateral_cm):
    if lateral_cm not in (-3, 0, 3):
        raise ValueError('only fixed commissioned reach candidates supported')
    origin = _start_pose(start)
    waypoints = [dict(start)]
    samples = []
    for stage in range(1, 5):
        fraction = stage / 4
        goal = dict(origin, x_cm=origin['x_cm'] + lateral_cm * fraction,
                    y_cm=origin['y_cm'] + 5 * fraction,
                    z_cm=origin['z_cm'] + 2 * fraction)
        target = _inverse(goal)
        previous = waypoints[-1]
        if ((not lateral_cm and target[6] != start[6])
                or abs(target[6] - start[6]) * DEGREES_PER_US > 20
                or max(abs(target[c] - previous[c]) for c in CHANNELS) > 150):
            raise ValueError('reach stage exceeds joint bound or changes base yaw')
        for step in range(21):
            ratio = step / 20
            pulses = {c: round(previous[c] + ratio * (target[c] - previous[c]))
                      for c in CHANNELS}
            pose = forward(pulses)
            along = (pose['y_cm'] - origin['y_cm']) / 5
            if (abs(pose['x_cm'] - (origin['x_cm'] + lateral_cm * along)) > .1
                    or not -.02 <= along <= 1.02
                    or abs(pose['z_cm'] - (origin['z_cm'] + 2 * along)) > .15
                    or abs(pose['pitch_degrees'] - origin['pitch_degrees']) > .2):
                raise ValueError('predicted transition leaves narrow reach corridor')
            samples.append({'stage': stage, 'fraction': ratio, 'pulses': pulses,
                            'predicted_pose': pose})
        waypoints.append(target)
    return {'start': dict(start), 'target': waypoints[-1],
            'outbound_waypoints': waypoints[1:],
            'return_waypoints': list(reversed(waypoints[:-1])),
            'predicted_start': origin, 'predicted_end': forward(waypoints[-1]),
            'stage_duration_seconds': .6, 'samples': samples,
            'lateral_cm': lateral_cm,
            'physical_position_verified': False,
            'scope': 'fixed attended reach only; no collision or payload model'}
