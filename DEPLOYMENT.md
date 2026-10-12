Codex, gpt-6-astra, Jetson seat, 2026-10-12T12:03:05.050685-04:00

# Temple Witness Pi controller deployment

The Pi is the subordinate hardware controller under the Jetson. It runs the
serial driver, command guard, watchdog, telemetry and camera. Planning and
reasoning remain on the Jetson; no language model was installed on the Pi.

## Live configuration

- Pi release: `/opt/temple-witness/releases/20261012-capabilities-04`, selected by
  `/opt/temple-witness/current`. File digests: `receipts/capability-deployed-manifest.json`.
- `temple-witness-guard.service`: enabled at boot, authenticated TCP
  `10.42.0.23:9040`, restricted to Jetson peer `10.42.0.1`.
- `temple-witness-camera.service`: enabled at boot, separate camera-only process,
  `http://10.42.0.23:8080/snapshot.jpg`, restricted to that same peer.
- Guard uses `/etc/temple-witness/hardware.json`: wheel duty limit **0**, no
  joints, outputs or sonar. Observed direction signs are saved with the zero wheel
  ceiling. Runtime commissioning override has been removed. Always starts disarmed;
  the deployed command does
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

## Initial deployment validation (historical)

See CAPABILITIES.md and receipts/capability-live-summary.json.145 Jetson tests
passed; Pi 144 passed with one unrelated PyYAML skip.52 deployed hashes match.
All eight translations,two turns, four small arm jogs and a small gripper jog were
physically observed with stops/returns. All movements were slight. Camera,
battery, sonar reads, lights/beep, and stationary frozen-driver recovery were
checked. Microphone/playback and several optional peripherals remain unavailable
or unverified. Vendor geometry, channel mapping and actual Pi offsets are recorded
in receipts/vendor-mapping-facts.json; full-range safe trajectories remain work.

Final authenticated status is disarmed, drive inactive and wheel ceiling 0.
No serial owner remains after disconnect. Guard/camera remain active/enabled;
four vendor masks remain intact. Final battery 7672mV, throttled 0x0, fresh JPEG.
Actual reboot testing occurred during handover, before these commissioning tests.
No whole-Pi-freeze stop guarantee, live MCP registration or autonomy is claimed.

## Rollback, only on a future explicit operator instruction

Original vendor unit files and pre-handover state are on the Pi at
`/var/lib/temple-witness-handover/20261012-handover-01`.
To restore vendor control in a future attended maintenance session: first stop
and disable both Temple services, verify no serial holder, replace each vendor
mask with its backed-up unit, reload systemd, and then explicitly choose which
vendor services to enable/start. Starting vendor code may initialize hardware.
Do not execute rollback automatically on a fault or reboot.


## Camera viewing and low-battery stop, latest checkpoint

The wrist was moved 754->604us, then shoulder 845->995us, then elbow
2485->2385us with shoulder 995->1295us in two stages. Fresh images showed the
view progressively lower. These were individually attended scratch commands,
not additions to the reusable pose API. They did not return home.

**Current commanded pulses: 1=1500,3=604,4=2385,5=1295,6=1405.** The model predicts
(0,9.10,12.22)cm and -45 degree pitch; physical coordinates/angles unmeasured.
The green top and red/blue shapes are partly visible at the bottom of the view.
No full stack localization, grasp or payload lift is established.

A further pose3/4/5=572/2283/1476 was planned but aborted before arming/motion at
6945mV. Later battery samples6851/6842/6836mV remain below the7000mV floor.
Normal locked configuration restored. Current state and evidence hashes are in
`receipts/arm-camera-low-battery.json`; resume steps in `PROJECT_STATE.md`.

Sonar125–126mm cannot be assigned to the stack. Installed camera intrinsics and
planar map were read as data; vendor Camera.py rectifies frames while Temple
publishes raw frames. Camera-to-gripper transform and stack-height corrections
are not validated. Do not blindly use the vendor pixel map for this grasp.

Worker startup EOFError caused two service restarts before movement; the cause
is unresolved. Recovery was disarmed. Software test counts remain155 Jetson,
144 Pi plus one skip, from prior runs. No new runtime deployment or test claim.
