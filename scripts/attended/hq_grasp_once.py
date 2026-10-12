# HQ Mac Studio, claude-fable-5-1, 2026-10-12 EDT. Derived from Codex's grasp_clear_s2_once.py (same harness).
"""One attended bounded batch through the Temple guard. Never auto-replay: refuses if its receipt exists.
Usage (from /home/tony/witness_pi_guard):
  python3 hq_grasp_once.py --name seat --start 1=1800,3=782,4=1803,5=2016,6=1520 --waypoints 5=2116 5=2216 --note '...' [--dry]
Each waypoint is a change applied to the previous pose. Only channels that change are enabled in the override."""
import argparse, ast, json, subprocess, sys, time, urllib.request
from datetime import datetime
from pathlib import Path
import cv2
from companion.arm_lift import OFFSETS
from witness_mcp.live import LiveServices

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True); ap.add_argument('--start', required=True)
ap.add_argument('--waypoints', nargs='+', required=True); ap.add_argument('--note', required=True)
ap.add_argument('--dry', action='store_true'); ap.add_argument('--battery-floor', type=int, default=7000)
ap.add_argument('--duration', type=float, default=0.6)
a = ap.parse_args()
assert 0.1 <= a.duration <= 1.0
def pose(s): return {int(k): int(v) for k, v in (p.split('=') for p in s.split(','))}
start = pose(a.start); assert set(start) == {1, 3, 4, 5, 6}
waypoints, prev = [], dict(start)
for w in a.waypoints:
    nxt = dict(prev); nxt.update(pose(w)); waypoints.append(nxt); prev = nxt
changed = sorted({c for w in waypoints for c in w if w[c] != start[c]} | {c for i, w in enumerate(waypoints[1:], 1) for c in w if w[c] != waypoints[i-1][c]})
assert changed, 'nothing changes'
for i, w in enumerate(waypoints):
    before = start if i == 0 else waypoints[i-1]
    for c in changed: assert abs(w[c] - before[c]) <= 200, 'step over 200us on channel %d' % c
    for c in w: assert 500 <= w[c] <= 2500

folder = Path('/home/tony/witness_scratch/hq-grasp-20261012'); folder.mkdir(exist_ok=True)
tmpl = Path('/home/tony/witness_scratch/capabilities-20261012/one_arm_side.py')
target = folder / ('hq-%s-initialize.json' % a.name)
assert not target.exists(), 'No replay: receipt exists for ' + a.name
receipt = {'identity': 'HQ Mac Studio, claude-fable-5-1', 'at': datetime.now().astimezone().isoformat(),
           'authorization': "Anthony at the terminal: 'ok let me see you grab the block and lift it'; attended, at the switch",
           'physical_angle_measured': False, 'start': start, 'planned_waypoints': waypoints, 'channels': changed,
           'duration_seconds': a.duration, 'note': a.note, 'dry': a.dry}
key = bytes.fromhex(Path('/home/tony/.config/temple-witness/guard.key').read_text().strip())
def connect(): return LiveServices(('10.42.0.23', 9040), 'jetson-coordinator', key, Path('/home/tony/.local/state/witness'))
def remote(script):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', 'witness-pi', 'sudo -n python3 -'], input=script, capture_output=True, text=True, timeout=25)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout) if r.stdout.strip() else None
def positions(s):
    p = {}
    for c in [1, 3, 4, 5, 6]:
        r = s._connect().request('witness_status', {'probe': {'kind': 'pwm_position', 'id': c}}); assert r['ok'], r; p[c] = r['value']
    return p
def snapshot(name):
    with urllib.request.urlopen('http://10.42.0.23:8080/snapshot.jpg', timeout=4) as r: data = r.read(); headers = dict(r.headers)
    assert data.startswith(b'\xff\xd8') and data.endswith(b'\xff\xd9'); assert 0 <= time.time() - float(headers['X-Witness-Capture-Time']) < 2
    (folder / name).write_bytes(data); return {'path': str(folder / name), 'sequence': headers.get('X-Witness-Sequence')}
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2); assert cap.isOpened(), 'External view required'
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
def external(name):
    for _ in range(12): ok, frame = cap.read(); assert ok, 'Lost external view'
    path = folder / ('hq-%s-%s.jpg' % (a.name, name)); assert cv2.imwrite(str(path), frame); return str(path)
def fork(name): return snapshot('hq-%s-fork-%s.jpg' % (a.name, name))
def battery_ok(status, floor):
    b = status['telemetry']['battery_mv']; return b['value'] >= floor and b['age_since_read_seconds'] < 2, b
receipt['external_before'] = external('before')
s = connect()
try:
    status = s.status(); assert status['ok'] and not status['armed'] and not status['drive_active'], status
    got = positions(s); assert got == start, 'Unexpected starting pose %r, no motion' % got
    receipt['before_image'] = fork('before'); receipt['readonly_status'] = status
finally:
    s.close()
s = None
if a.dry:
    cap.release(); receipt['result'] = 'dry run: pose matches start, cameras fresh, no setup, no motion'
    target.write_text(json.dumps(receipt, indent=2) + '\n'); print(json.dumps({'dry': True, 'start_ok': True, 'images': [receipt['external_before'], receipt['before_image']]})); sys.exit(0)
tree = ast.parse(tmpl.read_text())
assign = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
setup = ast.literal_eval(assign['setup'].func.value.func.value); cleanup = ast.literal_eval(assign['cleanup'])
joints = {'pwm_' + str(c): {'bus': 'pwm', 'id': c, 'min': min(q[c] for q in [start] + waypoints), 'max': max(q[c] for q in [start] + waypoints)} for c in changed}
tag = 'temple-hq-grasp-' + a.name
setup = setup.replace('EXPECTED_OFFSETS', repr(OFFSETS)).replace('JOINTS', repr(joints)).replace('temple-arm-side-reach', tag)
cleanup = cleanup.replace('temple-arm-side-reach', tag)
receipt['joints_override'] = joints
try:
    receipt['setup'] = remote(setup); time.sleep(1)
    s = connect(); status = s.status()
    assert status['ok'] and not status['armed'] and not status['drive_active'], status
    assert status['capabilities']['wheel_duty_limit'] == 0 and status['capabilities']['joints'] == joints
    assert positions(s) == start; time.sleep(1)
    receipt['preflight'] = s.status(); ok, b = battery_ok(receipt['preflight'], a.battery_floor); assert ok, 'battery %r' % b
    assert s.request_action('buzzer', {'frequency_hz': 1200, 'on_seconds': .1, 'off_seconds': .1, 'repeat': 1})['ok']; time.sleep(1.5)
    remote('import subprocess\nsubprocess.run(["systemctl","kill","--kill-who=main","--signal=SIGUSR1","temple-witness-guard.service"],check=True)')
    deadline = time.monotonic() + 1
    while not s.status()['armed']: assert time.monotonic() < deadline, 'did not arm'; time.sleep(.03)
    receipt['stages'] = []; expected = dict(start)
    for index, waypoint in enumerate(waypoints, 1):
        assert positions(s) == expected
        status = s.status(); ok, b = battery_ok(status, a.battery_floor); assert status['armed'] and ok, 'battery %r' % b
        action = s.request_action('joints', {'positions': {'pwm_' + str(c): waypoint[c] for c in changed}, 'duration_seconds': a.duration})
        receipt['stages'].append({'index': index, 'target': waypoint, 'action': action, 'battery_mv': b['value']})
        assert action['ok'], 'Rejected; no retry'
        time.sleep(a.duration + 0.7); expected = dict(waypoint)
        receipt['readback'] = positions(s); assert receipt['readback'] == expected
        receipt['stages'][-1]['external_image'] = external('stage-%d' % index); receipt['stages'][-1]['fork_image'] = fork('stage-%d' % index)
    receipt['external_after'] = external('after'); receipt['after_image'] = fork('after')
finally:
    try:
        if s is not None:
            try: receipt['stop'] = s.request_action('stop', {}); receipt['status_after'] = s.status()
            finally: s.close()
    finally:
        cap.release()
        try: receipt['cleanup'] = remote(cleanup)
        finally: target.write_text(json.dumps(receipt, indent=2) + '\n')
time.sleep(1); s = connect()
try: receipt['locked_status'] = s.status(); receipt['locked_readback'] = positions(s)
finally: s.close(); target.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'readback': receipt.get('readback'), 'locked': {'armed': receipt['locked_status']['armed'], 'joints': receipt['locked_status']['capabilities']['joints']}, 'locked_readback': receipt['locked_readback'], 'cleanup': receipt['cleanup'], 'images': [receipt.get('external_after'), (receipt.get('after_image') or {}).get('path')]}))
