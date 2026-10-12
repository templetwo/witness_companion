Codex, gpt-6-astra, Jetson seat, 2026-10-12T11:09:09.146747-04:00

# Witness capability map and commissioning results

**All movements were slight**, as Anthony explicitly reported. Tests establish
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
| Battery | Live telemetry about 7.7–7.9V, final 7.672V | No capacity/current measurement |
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
| Arm IK / stored sequences | Vendor source and eight stored files inspected | Not yet ported to bounded Temple trajectories; no full-range playback |
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

## Final state and next implementation step

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
