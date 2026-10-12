Codex, gpt-6-astra, Jetson seat, 2026-10-12T10:07:30.830175-04:00

# Temple control handover complete; motion commissioning remains

Anthony authorized replacing vendor control with Temple control, specified that
the 4 GB Pi should act as the stacked hardware controller under the Jetson, and
required that reboots not restore vendor control. This supersedes the earlier
selection of HQ for this handover. Full session access was enabled beforehand.

## Verified deployed state

- Temple guard and separate camera are installed, enabled at boot, and running.
  Pi release: `/opt/temple-witness/releases/20261012-handover-01`, selected by
  `/opt/temple-witness/current`. All 51 deployed files matched local SHA256s.
- `masterpi`, `hw_remote`, `hw_button_scan`, and `hw_find` services are inactive
  and persistently masked in `/etc/systemd/system`; vendor app control/discovery
  listeners are gone. The button service could launch vendor hardware tests;
  its vendor Wi-Fi-reset/software-shutdown shortcuts are consequently retired.
- Exactly one Temple supervised worker owns one serial descriptor during each
  connection; none remains after clean disconnect or heartbeat expiry.
- Guard endpoint `10.42.0.23:9040` is authenticated and restricted to the Jetson
  at `10.42.0.1`. Credential values are protected and excluded from the repo.
- Camera: `http://10.42.0.23:8080/snapshot.jpg`, independent of robot controls.
- **Motion stays disabled:** disarmed, wheel duty limit zero, empty joint map,
  no outputs or sonar configured. Operator-arm signals are not enabled.
  Startup sent only zero wheel duty; no nonzero motor or servo action was sent.
- Actual Pi reboot completed and changed the boot ID. Both Temple services
  returned enabled/running with zero restarts; all four vendor units remained
  masked/inactive. Authenticated status was still disarmed, with fresh battery
  telemetry 7921 mV and advancing camera frame sequences. Power flags `0x0`;
  NTP synchronized. Receipt: `receipts/temple-reboot.json`.
- Pi memory before reboot: 4050 MiB total, about 3450 MiB available after the
  control session closed. No language model was installed on the Pi.

## Software validation

Jetson Python 3.10: 132 tests passed. Pi Python 3.11: 132 run, 131 passed, one
unrelated PyYAML config check skipped. Eighteen focused handover tests cover
motion-disabled configuration, timed wheel stops/overlap/failure, all-or-none
joint validation, serial preflight/lock failures, and camera access/freshness.
The first unprivileged Pi test run could not create MCP temporary directories
inside the root-owned release; the intended guard identity completed the suite.

Live stationary checks verified authenticated status, single serial ownership,
clean-disconnect release, heartbeat-loss release, and fresh disarmed sessions.
These are not physical stopping or driving tests. Receipts are in `receipts/`.

## Operation and remaining work

See `DEPLOYMENT.md` for endpoints, protected credential paths, status command,
boot behavior and the attended rollback procedure. Original vendor unit files
are backed up on the Pi at
`/var/lib/temple-witness-handover/20261012-handover-01`. No automatic rollback.

Before any drive: commission wheel directions and physical stop behavior with
Anthony at the switch, resolve whole-Pi-freeze motor cutoff, then set explicit
limits and review/enable local arming. Joint calibration, peripheral validation,
operator-arm signal races, speech/microphone integration and live MCP/OpenClaw
registration remain separate work. The current Pi IP is DHCP-assigned; this task
did not reserve it or change network infrastructure. No nonzero motion is
permitted just because this software handover succeeded.

No OpenClaw configuration changes, Stack write, or git push occurred in this
handover. Earlier push authorization persists but GitHub authentication has not
been repaired. Earlier Stack checkpoint proposal is not claimed committed.
