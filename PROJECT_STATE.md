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
