Codex, gpt-6-astra, Jetson seat, 2026-10-12T17:37:27.588607-04:00

Current checkpoint: attended grasp attempts made contact and moved the green
block, but did not establish secure pickup. Gripper now open and clear, controls
locked. Current pulses1=1800,3=782,4=1803,5=2016,6=1520us, not home. Coordinated
model predictions/reversals diverged from physical motion; require calibration
and fix repeated startup EOF fault before further grasping. See PROJECT_STATE.md
and receipts/arm-grasp-attempt.json. No new runtime/software-test changes.

Codex, gpt-6-astra, Jetson seat, 2026-10-12T12:03:05.050685-04:00

# Witness capability map and commissioning results

**Initial movements were slight**, as Anthony explicitly reported; subsequent
larger reaches were observed. Tests establish
observed direction/response and stopping or return, not useful speed, full travel,
load capacity, positioning accuracy or collision-free reach. The Jetson remained
mounted. Later lateral, turn and diagonal trials were on a user-reported smooth
surface. Do not retroactively assign that surface to the earlier trials.

Vendor data is the nominal baseline; these tests validate the assembled robot.
The installed Pi source, not the downloaded reference, supplies current offsets.
Source presence, an accepted write, a board setpoint and physical observation are
four different kinds of evidence. See `receipts/vendor-mapping-facts.json` and
`receipts/capability-live-summary.json` for values, observations and receipt hashes.

## What the vendor already provides

- Mecanum layout: motors 1/2 front left/right, 3/4 rear left/right in the source
  frame; parameters a=67mm, b=59mm, wheel diameter65mm. The function labels velocity
  mm/s but writes duty values; loaded speed is not measured by that interface.
- PWM 1 gripper; PWM 3 wrist pitch, 4 elbow, 5 shoulder, 6 base yaw from IK equations.
  Arm joint movement was observed without separate anatomical identification.
  PWM 2 returned0 and was excluded from motion tests.
- Installed offsets in microseconds: 1=0,3=59,4=72,5=63,6=-95. The downloaded copy
  differs and must not overwrite the robot's values. Board offset reads of0 are
  separate from software-added offsets in the vendor YAML.
- Effective arm geometry: L1=9.3cm (8.0 + 1.3), L2=6.5cm, L3=6.2cm, L4=10.0cm.
  These are executable constants, not the conflicting stale comments or measured
  dimensions. Nominal PWM 3–6 conversion is 500–2500us over 180degrees.
- Default poses, gripper commands, Cartesian IK and stored sequences already
  exist. Do not rediscover them by blind motion. Validate final pulses after
  offsets and constrain trajectories before allowing them through Temple.

## Live results

| Ability | Result | Limit |
| --- | --- | --- |
| Four wheel drive | Raised duty 20/0.2s: all four wheels turned and stopped | Loaded floor behavior differs |
| Forward / backward | Both directions and stops observed at duty 20/0.5s | User-observed forward reverses vendor source's default forward label |
| Left / right | Both directions and stops eventually confirmed at duty 25/0.5s | Earlier conflicting reports preserved in ledger; no single cause established |
| Diagonals | All four directions and stops confirmed at duty 25/0.5s | Slight movement on smooth surface |
| Turns | Left and right nose turns and stops confirmed at duty 25/0.5s | No measured turn angle |
| PWM 3/4/5/6 | Each 40us jog and return physically observed | About 3.6 nominal degrees per vendor conversion; not measured joint angles or full ranges |
| Gripper PWM 1 | Initial100us jog worked; subsequent vendor release pose1500→1800→1500 at0.6s each way physically opened farther and returned | No measured opening width or grasp/load test |
| Battery | Recharged to about8.08V; 7.917V before last recovery move. Earlier low-battery stop retained in receipts | No capacity/current measurement |
| Sonar | Fixed separate-I2C-transaction reads; ten clear-space samples506–510mm | Not calibrated against a ruler; no autonomous avoidance |
| Camera | 640x480 JPEG; 147 requests/12s, 76 unique sequences, about 6.27fps; median RTT 1.75ms | Frame read time is not measured exposure time |
| Buzzer | Beeps physically confirmed | Short bounded tones |
| Board RGB / sonar RGB / LED | Isolated red→blue→LED sequence observed, including red off and blue blinking | User's sequence report, no photometric measurement |
| OLED text | Write accepted; no text observed | Display hardware/function unverified |
| PWM readback | Channels 1,3,4,5,6 report commanded pulses; jog/return readbacks matched | Not encoders or physical position feedback |
| Bus servos | Position IDs 1–6 and voltage/temp/torque/angle-limit queries on ID 1 returned device failure | Fitted bus hardware not established; no bus motion/EEPROM writes |
| IMU / buttons / gamepad / SBUS | Interfaces inventoried; no packets in passive window | Hardware availability remains unknown |
| Speech / microphone | Piper synthesized a WAV; playback and capture could not open default audio device | No audible speech or microphone capability established |
| Five tool facade | Live status, snapshot, stop and local pending note worked; speech failed at playback | No live MCP/OpenClaw registration; note not committed to Stack |
| Arm IK / stored sequences | Pure bounded lift/reach/side planners tested; downward viewing moves observed in camera frames | General IK and stored playback unavailable; no grasp/full-range validation |
| Vision modes / LAB / avoidance | Vendor source inventory complete | Not re-enabled or claimed operational through Temple |
| Driver stall recovery | Stationary SIGSTOP: status failed in 0.502s, serial released, next session disarmed | No moving fault injection; no guarantee for whole-Pi/kernel freeze |

## Observed wheel map

Signs below are channels 1–4, multiplied by a bounded duty. They preserve the
operator's observed frame; they are not a silent modification of vendor source.

| Direction | Signs |
| --- | --- |
| Forward |1,-1,1,-1 |
| Backward |-1,1,-1,1 |
| Left |1,1,-1,-1 |
| Right |-1,-1,1,1 |
| Forward-left |1,0,0,-1 |
| Forward-right |0,-1,1,0 |
| Backward-left |0,1,-1,0 |
| Backward-right |-1,0,0,1 |
| Turn left |1,1,1,1 |
| Turn right |-1,-1,-1,-1 |

## Initial survey checkpoint (historical; latest state below)

Release `/opt/temple-witness/releases/20261012-capabilities-04` is active; 52 file
hashes match local sources. 145 Jetson tests passed; 144 Pi tests passed and one
unrelated PyYAML check skipped. Tests are fake/stub software validation.

Commissioning override removed. Persistent config saves observed wheel signs
with wheel ceiling 0, no joints, no outputs and sonar disabled. Guard has no
operator-arm signal option; it is disarmed, with no serial owner between sessions.
Temple guard/camera are active/enabled; all four vendor units remain masked and
inactive. Actual reboot persistence was checked during the preceding handover;
this task verified the final unit/config state after restoring normal operation.

Next use installed vendor channel/geometry/offset data to implement bounded
Temple arm poses and IK, checking final offset pulses and transitions. Measure
loaded travel/turn response and usable arm reach instead of treating these slight
checks as full calibration. Autonomous vendor modes remain unavailable. An
independent whole-Pi-freeze cutoff is still outside the software guard.

Raw room images and detailed live receipts stay outside the repo in
`/home/tony/witness_scratch/capabilities-20261012/`. The report retains failed,
unobserved and conflicting trials alongside later successes.


Codex, gpt-6-astra, Jetson seat, 2026-10-12T11:15:22.523552-04:00

Follow-up: vendor gripper release1500→1800→1500,0.6s each way, physically
confirmed opened farther and returned. Receipt: receipts/vendor-gripper-release.json.
This follow-up is distinct from the initial all-slight survey. Runtime override
removed after the test; authenticated status again disarmed, wheel limit0 and
no joints/outputs. Larger arm poses and loaded travel measurements remain pending.


Codex, gpt-6-astra, Jetson seat, 2026-10-12T11:27:08.153274-04:00

Bounded vendor-derived arm poses now tested; see ARM_POSES.md. The modelled1cm
lift rose and returned roughly level but was very slight. A subsequent fixed5cm
forward/2cm up reach in four stages was visibly larger and returned. Exact
Cartesian displacement remains unmeasured. Pure planner is Jetson-only; general
Cartesian control/stored actions remain unavailable.153 Jetson tests now pass;
Pi code unchanged. Final status and boot configuration are again locked.


Codex, gpt-6-astra, Jetson seat, 2026-10-12T11:32:35.886647-04:00

Left and right arm reaches now physically observed with returns. Each combines
a modelled 3cm sideways, 5cm forward and 2cm upward move; exact distances remain
unmeasured. See ARM_POSES.md and receipts/arm-side-live.json.155 Jetson tests pass;
Pi code unchanged. Normal locked state and vendor masks verified afterwards.


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
