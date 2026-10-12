Codex, gpt-6-astra, Jetson seat, 2026-10-12T10:05:53.789871-04:00

# Temple Witness Pi controller deployment

The Pi is the subordinate hardware controller under the Jetson. It runs the
serial driver, command guard, watchdog, telemetry and camera. Planning and
reasoning remain on the Jetson; no language model was installed on the Pi.

## Live configuration

- Pi release: `/opt/temple-witness/releases/20261012-handover-01`, selected by
  `/opt/temple-witness/current`. File digests: `receipts/temple-deployed-manifest.json`.
- `temple-witness-guard.service`: enabled at boot, authenticated TCP
  `10.42.0.23:9040`, restricted to Jetson peer `10.42.0.1`.
- `temple-witness-camera.service`: enabled at boot, separate camera-only process,
  `http://10.42.0.23:8080/snapshot.jpg`, restricted to that same peer.
- Guard uses `/etc/temple-witness/hardware.json`: wheel duty limit **0**, no
  joints, outputs or sonar. Always starts disarmed; the deployed command does
  not enable operator arming signals. No remote arming operation exists.
- The guard opens a single SDK Board in a supervised child for each connection.
  Startup writes zero wheel duty. Clean disconnect or heartbeat loss kills the
  worker and releases serial. Between sessions no process owns serial.
- HMAC credential: protected local file
  `/home/tony/.config/temple-witness/guard.key`; protected Pi source
  `/etc/temple-witness/guard.key`, supplied to the service using systemd
  `LoadCredential`. Values are excluded from source, receipts and reports.
- Pi journal paths: `/var/lib/temple-witness/guard.jsonl` and `notes.jsonl`.
  The journal files are populated lazily when relevant accepted actions or
  notes occur. The directory and journal lock already exist.

Read authenticated status from the Jetson:

```sh
cd /home/tony/witness_pi_guard
python3 -m scripts.witness_check
```

This command maintains heartbeat while connected, requests status only, and
closes the session when done. It does not arm or issue action commands.
`--hold-seconds 20` can keep the session alive for a bounded inspection.
No automatic reconnect, action replay, OpenClaw configuration change or live MCP
registration was made. The existing MCP live adapter remains a separate
integration step. Speech and microphone are not deployed as Pi functions.

## Vendor retirement and boot behavior

The following units are inactive, removed from multi-user.target, and
persistently masked by `/etc/systemd/system/<unit> -> /dev/null`:

- `masterpi.service`: former serial/controller and camera owner.
- `hw_remote.service`: vendor phone/network configuration listener on TCP 9026.
- `hw_button_scan.service`: could start a vendor hardware-test program; its
  vendor button shortcuts for Wi-Fi reset and software shutdown are retired too.
- `hw_find.service`: vendor app discovery on UDP 9027.

An actual Pi reboot was performed and verified: the boot ID changed, both
Temple units returned active/enabled with zero service restarts, all four vendor
units remained masked/inactive, and fresh authenticated status/camera worked.
The guard remained disarmed with a zero wheel limit. See `receipts/temple-reboot.json`.

Temple units are enabled through multi-user.target. Neither new service starts
or restores vendor services. No automatic vendor rollback is installed.
NetworkManager, SSH and existing network infrastructure are preserved.
The current Pi address is still DHCP-assigned, not reserved by this work; an
address change requires updating the configured bindings and Jetson endpoint.

## Validation and remaining commissioning

- Jetson Python 3.10: 132 tests passed, including 18 focused handover tests.
- Pi Python 3.11: 132 run, 131 passed, one unrelated PyYAML config test skipped.
  Initial unprivileged tests could not write MCP temporary directories inside
  the root-owned release; rerunning as the intended guard identity resolved it.
- Full live ownership inventory found one serial descriptor in one Temple
  worker; vendor TCP ports 9030/9026 and UDP discovery 9027 were absent.
- Actual heartbeat lapse with wheels disabled released serial; a new explicit
  session started disarmed. This verifies process cleanup, not physical braking.
- Battery telemetry arrived around 7.9 V. Separate camera returned JPEGs with
  advancing frame sequences. Timestamps describe frame-read time, not measured
  sensor exposure. Pi power flags were 0x0 and NTP synchronized.
- Memory snapshot: 4050 MiB total, about 3450 MiB available after disconnect.

No nonzero motor command, servo command, drive test or arm test was sent.
Physical stopping, direction validation, wheel/servo calibration, full-Pi-freeze
cutoff and attended motion commissioning remain outstanding. Outputs and sonar
also remain disabled until their specific validation. The unused operator-arm
signal path and broader adapter behavior still need review before enabling them.

## Rollback, only on a future explicit operator instruction

Original vendor unit files and pre-handover state are on the Pi at
`/var/lib/temple-witness-handover/20261012-handover-01`.
To restore vendor control in a future attended maintenance session: first stop
and disable both Temple services, verify no serial holder, replace each vendor
mask with its backed-up unit, reload systemd, and then explicitly choose which
vendor services to enable/start. Starting vendor code may initialize hardware.
Do not execute rollback automatically on a fault or reboot.
