Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:32:40 EDT

# Witness checkpoint: hardware paused for battery charging

Anthony's latest instruction: "ill just charge the batterys", then "update the project state where we left off and the stack". Hardware work is paused. No new motion, SDK import, serial open, service stop, install or deployment occurred during this integration turn.

## Observed hardware state

- The Pi repeatedly lost Ethernet carrier. SSH returned briefly after USB-C changes and a reported 20 W supply, then failed again. Last observation: carrier=0 and SSH timeout; user reported solid red.
- During reachable periods, vcgencmd returned 0x50005 (active undervoltage and throttling, plus historical flags). One later reading was 0x50000 (historical flags only), before another link loss. Power instability is evidenced; an exhausted battery is Anthony's hypothesis, not a proven root cause of every outage. The USB-C source/cable ratings were not established.
- masterpi.service was still active after boot, PID 728 on one check. Vendor serial handover has NOT happened. Exclusive Witness serial ownership remains Anthony's chosen architecture; the vendor app will not be used.
- A paired clock read showed Pi epoch 1791775573 (~23:26:13 EDT) while Jetson reported 23:29:09 EDT. Check and correct clock synchronization before testing the signed, short-TTL transport; this apparent ~176-second skew would reject valid actions.
- Earlier attended floor pulse at duty 12 produced no observed motion. No subsequent drive attempt. Physical stopping and all new controls remain unverified by this seat.

## Local software checkpoint, NOT deployment ready

Worktree /home/tony/witness_pi_guard, branch prep/pi-guard-service. Prior committed checkpoint: 12fab93. New work is saved as a WIP integration checkpoint on this branch; do not infer completion or authorization to operate from this commit.

Implemented locally, but requiring focused tests and review:

- Expanded strict action schema: bounded named drive directions (duty <=25, duration <=0.5 s), named joint positions, LED, RGB, buzzer and sonar RGB.
- HardwareBoard maps to an injected SDK, serializes writes, schedules a driver-local timed wheel stop, validates configured joint ranges before any joint write, and reports packet telemetry and configured capabilities.
- RealBoardFactory lazily loads the standalone vendor SDK only on explicit --real, checks /proc ownership before/after open, requests advisory and tty-exclusive locks, and sends initial zero wheel duty. This factory has NEVER been run. Kernel tty exclusion does not stop a privileged opener; retiring restart paths is still required.
- Pi service has explicit --fake/--real modes and optional local SIGUSR1 arming / SIGUSR2 stop. Starts disarmed; no remote arm tool. Signal/session race behavior still needs targeted tests and review.
- MCP package imported from the sibling prep/mcp-tool-server worktree, with a live adapter connecting the five tools to PiLink, a persistent issued-action journal and local notes. No automatic reconnect or action retry.
- CameraSnapshot fetches a bounded JPEG and distinguishes receive time from unknown vendor exposure time. Separate camera-only service avoids importing MasterPi control code. Local Piper/aplay adapter uses the existing local voice without downloading.

Not done: deployment/configuration, shared guard credential provisioning, stable power, time sync, exclusive serial handover, camera migration, calibrated joint map, direction validation, physical stop/heartbeat-loss commissioning, focused new-adapter tests, live end-to-end MCP registration. OpenClaw config remains unchanged. No new packages installed. No push.

The new adapter has no commissioned limits file. A zero wheel limit and empty joint map are the intended initial configuration; do not fabricate calibration. Arm IK, stored trajectories and autonomous vendor vision modes report unavailable rather than importing competing serial owners. Microphone input remains unconfigured. PWM servo travel cannot be cancelled by the documented stop; it stops wheels and configured bus joints only. Software cannot guarantee a stop if the entire Pi freezes.

## Verification at pause

`python3 -m unittest discover -s tests -t .`: 114 tests passed on Jetson Python 3.10.12. This is the combined existing guard/coordinator/preflight/MCP regression suite, NOT focused coverage of the new hardware, camera and speech adapters. The MCP no-mode diagnostic assertion was updated for the new explicit --pi option. All 30 companion/MCP Python modules parse under 3.10 and 3.11 grammar; Python 3.11 runtime was not tested. Receipt: receipts/integration-checkpoint.txt.

## Resume in order

1. Anthony finishes charging and reconnects. Inspect carrier, SSH, current/historical power flags and clock synchronization without moving anything.
2. Finish focused tests and review the WIP: bounded drive auto-stop/overlap/failure, all-or-none joint validation, calibration failures, serial exclusivity and restart behavior, local arming races, camera freshness/size/error handling, speech failure, MCP-to-real-guard behavior and credential lifecycle.
3. Coordinate the already-selected HQ Pi handover; retire actual vendor owners and restart paths, preserve camera through the separate service, verify sole serial ownership. Keep initial service disarmed with motion disabled.
4. Register the tools against the reviewed deployed endpoint. Validate sensor/camera/speech/peripheral capabilities honestly, then commission one bounded motor action with Anthony at the physical switch and suitable clearance. No automatic replay of prior pulses.
5. Update project/Stack receipts with what was actually measured. No claim that Witness is fully wired or ready to pilot yet.
