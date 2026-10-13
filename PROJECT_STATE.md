Codex, gpt-6-astra, Jetson seat, 2026-10-12T20:50:34-04:00

## Guard floor release deployed on Anthony's explicit instruction

User: "deploy it. the pi is on usb powqer". This supersedes the previous
source-only/no-deploy task state. No independent HQ review is claimed.

Live Pi guard release is now `/opt/temple-witness/releases/20261013-guard-floor-01`,
selected by the persistent `/opt/temple-witness/current` symlink. Source is
`a9197fb8374c9a3177323569cf6d321543ef6dba`, carrying 8a657b4 startup diagnostics and
all three next-bound changes. All 39 packaged source/configuration files match
the release manifest's SHA256 hashes. Pi source compilation passed. The exact
source already passed 188 tests on Jetson 3.10.12 and Pi scratch 3.11.2.

Guard stopped and started once for deployment. Four authenticated status samples
verified `armed: false`, `battery_floor_mv: 7000`, driver alive, drive inactive,
wheel duty ceiling 0 and empty joint/output maps. First immediate connection
preceded listener readiness and was refused; a later status-only session worked
with the same PID and zero automatic restarts. No action replay occurred.
Controller telemetry was 5295/5305/5302 mV on user-reported USB power; this is not
a battery charge percentage or proof the motor pack is connected/charged.

No arming, override, motion request, arm/gripper command or motion test occurred.
The ordinary status worker opened the SDK and performed its existing zero-wheel
startup/cleanup; serial ownership was absent after disconnect. No vendor image
files were manually inspected or copied. Locked configuration and installed unit
hashes are unchanged. All four vendor services remain masked/inactive; guard is
enabled at boot. No reboot was needed/performed. The camera process was unchanged
(the camera module is byte-identical across the old and new releases).

Receipt: `receipts/guard-floor-deployed.json`; raw evidence and manifest:
`/home/tony/witness_scratch/deploy-guard-floor-01/`.
Rollback target remains `/opt/temple-witness/releases/20261012-capabilities-04`.
No rollback was needed. No network changes, GitHub push, Stack write or message.
Historical arm pose is not remeasured. Motion/calibration/grasp work still needs
fresh operator attendance and a charged motor supply; deployment grants no motion.

Codex, gpt-6-astra, Jetson seat, 2026-10-12T20:40:01-04:00

LATEST: HQ's next software bound is implemented in three separate commits on
prep/pi-guard-service: cached 7000 mV / 5 s battery gate, validated companion
drive/joints params, MCP speech 200 / notes 2000 character schemas. Final suite:
188 passing on Jetson Python 3.10.12 and Pi scratch Python 3.11.2. See PASSDOWN.md
and receipts/next-bound-20261013 for review and transfer details.

NOT DEPLOYED. Live release last reported 20261012-capabilities-04. No live service
restart, motion, SDK/vendor read, Stack write or GitHub push occurred in this task.
Startup diagnostic fix 8a657b4 is in source but not live. Newer HQ physical record
supersedes the old record below: jaws open at 2000; PWM 3=782,4=1803,5=2116,6=1420;
not home; battery 6984 mV; no pickup. Nothing was remeasured during this task.
New robot motion needs fresh charged-pack verification and operator attendance.
HQ review precedes a separately authorized deployment.

Codex, gpt-6-astra, Jetson seat, 2026-10-12T17:44:55.687385-04:00

Current detailed handoff: `PASSDOWN.md`. It supersedes historical state below.
Cleanup correction: runtime arming override is absent, but inactive
`/run/temple-witness/commissioning.json` remains. The active service uses the
locked permanent configuration without operator signals; four vendor units are
masked. This documentation task performed metadata-only SSH inspection, no
hardware action, SDK session or restart. Last pose/battery are historical final
observations from17:35 EDT, not newly measured here. See
`receipts/passdown-state-inspection.json`.

Codex, gpt-6-astra, Jetson seat, 2026-10-12T17:37:27.588607-04:00

## Latest: grasp attempt, no confirmed pickup

ATTENDED GRASP ATTEMPT ENDED WITHOUT CONFIRMED PICKUP. Both cameras now show
blocks and fork; green is above red/blue side-by-side. Pads contacted, tilted
and shifted green, but no inspected frame confirms complete clearance/retention.
Gripper closure1700/1600, then1500us with deeper seating did not secure a pickup.
User reports arm quiet and suspects loose grip. Stronger closure alone was not
sufficient; jaw placement and physical joint response need correction.

FINAL: gripper OPEN and visibly clear above supported stack. Command readbacks
1=1800,3=782,4=1803,5=2016,6=1520us; NOT HOME and not encoder angles. Normal guard
locked/disarmed,driveinactive,wheel0,joints{},outputs[],sonarfalse; runtime
commissioning files removed, normal service has no operator signals. All four
vendor units independently checked masked. Last movement preflight battery
7720mV. No wheels moved. No new runtime deployment or code-test count.

Modelled coordinated lifts and exact pulse reversal diverged from observed tip
movement. Unloaded shoulder100us/0.6s produced clear rise;50us/1s earlier did not.
Cause not established; do not assume deadband, weak servo, calibration error or
full-range/load capability. Camera-to-jaw and joint response remain uncalibrated.
Startup EOFError recurred17:28:31 and17:34:35 before arming/actions. Fresh disarmed
sessions verified unchanged commands before narrowly bounded recovery. Fault
remains unresolved; further grasps held for controller and placement diagnosis.
No scratch runner may auto-replay; historical reports are not fresh permission.

Evidence: witness_pi_guard/receipts/arm-grasp-attempt.json hashes raw receipts
and final images in witness_scratch/capabilities-20261012. Final view:
grasp-clear-s2-after.jpg. No claim of autonomous grasping or successful payload
lift. User's two blue front lights identified provisionally as ultrasonic RGB
indicators, consistent with vendor device documentation and earlier light tests.

## Earlier checkpoints, historical

Codex, gpt-6-astra, Jetson seat, 2026-10-12T17:05:08.748652-04:00

## Latest: charged and upright pose restored

At 17:05 EDT, after explicit operator attendance and clearance,
the arm was restored one joint at a time with external webcam checks. Final
commanded pulses: **1=1500,3=754,4=2485,5=845,6=1405us**. External images show an
upright arm and approximately level, empty fork. These are setpoints and visual
observations, not measured encoder angles. Earlier postboot zero readings no
longer describe current commanded state.

Battery after recharge was about8.08V; before the last move it was7.917V.
Guard is locked/disarmed, wheel0,joints{},outputs[],sonarfalse; runtime override
removed. New external C270 camera is on Jetson /dev/video0 at1280x720. It sees
the whole arm but currently clips floor/blocks. Pi fork camera also works.
No grasp occurred; full stack localization/camera-to-gripper calibration and
prior startup EOF fault remain unresolved. Do not replay one-off initialization
runners. Receipt: `receipts/arm-postboot-recovery.json`.

This supersedes the non-home pose and low-battery state in the earlier records.
Pi runtime unchanged; no new software-test count is claimed.

## Earlier checkpoint, historical

Codex, gpt-6-astra, Jetson seat, 2026-10-12T12:03:05.050685-04:00

# Witness checkpoint: motion stopped for low battery

Temple control and reboot persistence are established. The 4 GB Pi runs the
guard, hardware driver and camera; the Jetson handles planning. Vendor masterpi,
hw_remote, hw_button_scan and hw_find remain persistently masked. The deployed
Pi release is still `/opt/temple-witness/releases/20261012-capabilities-04`.

## Current physical and control state

**The arm is not at home.** Last verified commanded pulses, in microseconds:

| PWM channel | Role | Commanded value |
| --- | --- | --- |
| 1 | Gripper | 1500 |
| 3 | Wrist pitch | 604 |
| 4 | Elbow | 2385 |
| 5 | Shoulder | 1295 |
| 6 | Base yaw | 1405 |

The arm model predicts tip (0,9.10,12.22)cm and pitch -45 degrees. These are
predictions, not measured physical coordinates; PWM reads are board setpoints.
The camera visibly looks down. The green-topped blocks remain partly cropped
at the bottom of the latest successful image. No grasp or payload lift occurred.

Latest battery samples at 11:59 EDT: **6.851, 6.842, 6.836 V**. The planned next
viewing move aborted at 6.945 V, before arming or movement, against the existing
7.0 V floor. Charge the robot before further physical work; do not lower the
threshold or retry for a transient higher sample.

Normal configuration restored and independently checked: disarmed, drive inactive,
wheel limit 0, joints{}, outputs[], sonarfalse, no runtime commissioning override
and no operator-arm signals. Vendor masks persist. No new reboot was performed.

## Completed commissioning

All eight translations and both turns were observed with stops. Early joint and
gripper checks were slight; later vendor gripper release, larger forward/up reach,
and mirrored side reaches were observed with returns. Subsequent wrist and
elbow/shoulder movements produced fresh downward camera views and were left in
the pose above. See ARM_POSES.md and CAPABILITIES.md for evidence and limits.

Buzzer/lights, camera, battery, corrected sonar transactions and stationary driver
freeze recovery were tested. No full speed/range/load calibration, successful
grasp, autonomous vision behavior, audio I/O or whole-Pi-freeze stopping guarantee.

## Unfinished work and resume order

1. Establish charged battery and current attended clearance. Read actual setpoints;
   old home-start runners do not apply. Never replay a scratch action automatically.
2. Diagnose the repeated worker startup failure: EOFError escaped Supervisor
   construction at 11:36 and 11:53; systemd recovered disarmed. Root cause unknown,
   no fix claimed. Both failures preceded the intended movement.
3. Obtain a complete stack view and validate camera-to-gripper localization.
   Vendor images are rectified, Temple snapshots raw; the vendor planar pixel map
   is not established for this viewing pose or the top of a three-block stack.
   Sonar 125–126mm is an unassigned echo, not a verified block/front-edge distance.
4. Plan and check an approach before opening, closing or lifting around the block.
   The latest proposed pose (PWM3/4/5=572/2283/1476) was never executed.
5. Remaining broader work: loaded travel measurement, optional peripherals/audio,
   independent power cutoff, live MCP registration and general bounded arm control.

## Evidence and validation

`receipts/arm-camera-low-battery.json` records current state, completed moves,
preflight abort, battery samples and hashes of local raw evidence. Room images
and one-off runners remain in `/home/tony/witness_scratch/capabilities-20261012/`.
Earlier receipts cover vendor facts, reboot persistence, wheel mapping and poses.

Last software results: **155 Jetson passes**, **144 Pi passes + one skip**.
Those are prior results, not rerun for this documentation-only checkpoint.
Pi runtime code unchanged; previous 52-file deployment manifest remains the last
hash verification. No new installation, motion, OpenClaw edit or deployment in
this checkpoint task. Local git checkpoint follows; GitHub push previously failed
for unavailable authentication. Stack update is tracked separately from local files.
