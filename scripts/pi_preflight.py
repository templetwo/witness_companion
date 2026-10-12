# Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:09:44 EDT
"""Read-only Linux ownership inventory. Never opens serial or imports an SDK."""
import argparse
import json
import os
from pathlib import Path
import stat

PORTS = {8080, 9026, 9030}


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='/dev/ttyAMA0')
    args = parser.parse_args()
    print(json.dumps(inspect_owners(args.device),indent=2))
