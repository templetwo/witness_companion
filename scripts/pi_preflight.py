# Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:09:44 EDT
"""Read-only Linux ownership inventory. Never opens serial or imports an SDK."""
import argparse
import json
import os
from pathlib import Path
import stat
import time

PORTS = {8080, 9026, 9030, 9040}


def socket_ports(proc):
    result = {}
    unreadable = []
    for family in ('tcp', 'tcp6'):
        path = proc/'net'/family
        try:
            lines = path.read_text().splitlines()[1:]
        except OSError:
            unreadable.append(str(path))
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != '0A':
                continue
            try:
                port = int(fields[1].rsplit(':', 1)[1], 16)
            except (ValueError, IndexError):
                continue
            if port in PORTS:
                result[fields[9]] = port
    return result, unreadable


def inspect_owners(device='/dev/ttyAMA0', proc=Path('/proc')):
    proc = Path(proc)
    canonical = os.path.realpath(device)
    ports, network_errors = socket_ports(proc)
    owners, denied, raced = [], [], []
    for path in sorted(proc.iterdir()):
        if not path.name.isdigit():
            continue
        serial_fds, listeners = [], set()
        try:
            fds = list((path/'fd').iterdir())
            for fd in fds:
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if target.startswith('/') and os.path.realpath(target) == canonical:
                    serial_fds.append(int(fd.name))
                if target.startswith('socket:[') and target.endswith(']'):
                    port = ports.get(target[8:-1])
                    if port is not None:
                        listeners.add(port)
            if serial_fds or listeners:
                # Never read cmdline or environ: vendor hotspot argv can contain secrets.
                owners.append({'pid':int(path.name), 'comm':(path/'comm').read_text().strip(),
                               'cgroup':(path/'cgroup').read_text().splitlines(),
                               'serial_fds':sorted(serial_fds), 'listening_ports':sorted(listeners)})
        except FileNotFoundError:
            raced.append(int(path.name))
        except (PermissionError, OSError):
            denied.append(int(path.name))
    try:
        device_is_character = stat.S_ISCHR(os.stat(device).st_mode)
    except OSError:
        device_is_character = False
    holders = [row['pid'] for row in owners if row['serial_fds']]
    observed_ports = sorted({port for row in owners for port in row['listening_ports']})
    expected_ports = sorted(set(ports.values()))
    return {'device':device, 'device_is_character':device_is_character,
            'serial_holder_pids':holders, 'owners':owners,
            'unreadable_pids':denied, 'vanished_pids':raced,
            'network_tables_unreadable':network_errors,
            'listener_ports':expected_ports, 'owner_mapped_ports':observed_ports,
            'inspection_complete':not (denied or raced or network_errors) and observed_ports == expected_ports,
            'serial_release_verified':False,
            'note':'One /proc snapshot cannot prove ongoing exclusivity or stop behavior. No device opened.'}


def _scan_defect(result, allowed_holders):
    """Name the first defect that must refuse at once, 'vanished_only', or None for a clean scan."""
    if result.get('device_is_character') is not True:
        return 'device_not_character'
    holders = result.get('serial_holder_pids')
    if not isinstance(holders, list):
        return 'malformed_result'
    if set(holders) - set(allowed_holders):
        return 'foreign_holder'
    if result.get('inspection_complete') is True:
        return None
    # Fail closed: every field must be present and every defect other than vanished pids empty.
    fields = ('unreadable_pids', 'vanished_pids', 'network_tables_unreadable',
              'listener_ports', 'owner_mapped_ports')
    if any(not isinstance(result.get(name), list) for name in fields):
        return 'malformed_result'
    if result['unreadable_pids']:
        return 'unreadable_pids'
    if result['network_tables_unreadable']:
        return 'network_tables_unreadable'
    if result['owner_mapped_ports'] != result['listener_ports']:
        return 'port_map_mismatch'
    if result['vanished_pids']:
        return 'vanished_only'
    return 'malformed_result'  # incomplete with no named defect


def inspect_owners_settled(device='/dev/ttyAMA0', attempts=5, pause=0.05, proc=Path('/proc'),
                           allowed_holders=(), sleep=None, budget=0.25):
    """Bounded rescan around inspect_owners for process churn only.

    A pid that exits between the /proc listing and its fd read (vanished_pids)
    cannot hold the device, so a scan whose only defect is vanished pids is
    rescanned up to `attempts` times. The first clean scan is accepted. If every
    attempt is vanished-only, the last one is usable for the holder decision.
    A denied pid, an unreadable network table, a port-map mismatch, a
    non-character device, a holder outside `allowed_holders`, or a malformed
    result refuses at once. inspect_owners' own contract is unchanged.
    Rescanning also stops once `budget` seconds of wall clock would be passed
    by another pause, so a slow sleep or a large /proc cannot stretch startup;
    the deadline only shortens the loop and never turns a refusal into a pass.
    """
    sleep = time.sleep if sleep is None else sleep
    started = time.monotonic()
    for attempt in range(1, attempts + 1):
        result = inspect_owners(device, proc)
        defect = _scan_defect(result, allowed_holders)
        if defect is None:
            return {**result, 'usable':True, 'settled_on':'complete', 'attempts':attempt}
        if defect != 'vanished_only':
            return {**result, 'usable':False, 'settled_on':defect, 'attempts':attempt}
        if attempt == attempts or time.monotonic() - started + pause > budget:
            break
        sleep(pause)
    return {**result, 'usable':True, 'settled_on':'vanished_only', 'attempts':attempt}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='/dev/ttyAMA0')
    args = parser.parse_args()
    print(json.dumps(inspect_owners(args.device),indent=2))
