Codex, gpt-6-astra, Jetson seat, 2026-10-12T17:44:55.687385-04:00

# Witness detailed passdown to HQ and the next Jetson operator

Requested by Anthony after the attended grasp attempt. This replaces the older
HQ handover packet as the current entry point. Earlier passdowns and results
remain historical evidence. This document is a handoff, not authorization to
resume movement. No hardware motion, arming, restart, deployment, or vendor
activation was performed to prepare it. A read-only SSH service/file inspection
was performed; no SDK connection was opened for this documentation task.

## 1. Read this first: exact state left behind

**No secure block pickup was confirmed. The arm is not home.** The green wooden
block remains supported by the red/blue lower blocks. The final external camera
image shows the empty, open gripper clear above them. Wheels never moved during
the grasp work. These are the last verified commands, not fresh measurements of
physical joint angles and not a guarantee that the robot has stayed unchanged:

| PWM | Vendor role | Last commanded pulse, microseconds |
| --- | --- | --- |
| 1 | Gripper | 1800, open |
| 3 | Wrist pitch | 782 |
| 4 | Elbow | 1803 |
| 5 | Shoulder | 2016 |
| 6 | Base yaw | 1520 |

Final action/image time: about **2026-10-12 17:35:44 EDT**. Last movement preflight
battery: **7720 mV**. This is voltage, not charge percentage or remaining capacity.
The established motion floor is 7000 mV with a fresh reading, age under two seconds.
Do not lower the floor or retry an aborted move hoping for a higher transient.

Final authenticated guard result: disarmed, drive inactive, wheel ceiling 0,
joints {}, outputs [], sonar false. The temporary systemd arming override was
removed. Normal ExecStart uses `/etc/temple-witness/hardware.json` and does not
include `--operator-signals`. The later documentation-only SSH check confirmed
that service active, that ExecStart, and all four vendor units still masked.

**Correction to the previous checkpoint:**
`/run/temple-witness/commissioning.json` still exists as unused runtime data.
Only `/run/systemd/system/temple-witness-guard.service.d/commissioning.conf` was
removed. The old cleanup template removes the override, not that JSON file.
Earlier wording saying all commissioning files were removed was too broad.
The active service does not reference the remaining JSON. It was left untouched
in this documentation task; do not reuse it to enable old limits. See
`receipts/passdown-state-inspection.json`.

Final visual evidence:
`/home/tony/witness_scratch/capabilities-20261012/grasp-clear-s2-after.jpg`.
The corresponding `grasp-clear-s2-fork-after.jpg` is the arm-mounted view.
The final control receipt is `grasp-clear-s2-initialize.json` in that directory.

## 2. Decisions already made and authorization boundaries

Anthony chose Temple control, exclusive Temple serial ownership, and no vendor
app. The Pi is the subordinate controller under the mounted Jetson. Reboots must
restore Temple, not vendor control. These architecture decisions are settled.
Anthony authorized the handover, reboot verification, attended capability tests,
arm tests and grasp retries in the completed session. Do not re-ask those design
choices, but do not treat old attendance or this report as new motion permission.

For any future live trial, establish current operator attendance at motor power,
physical clearance, empty/loaded state, fresh camera views, battery and current
command values. If the board rebooted and PWM readbacks are zero, those are
uninitialized commands, not known angles. Do not send an old home trajectory.
After today's recharge a reboot reset all five relevant channels to zero; recovery
required an external view and isolated joint initialization. That recovery is
historical, not a reusable blind boot procedure.

Do not import/run vendor modes, action groups, RPC control programs, or random
scratch runners. Use Temple's bounded driver path. Vendor source is a read-only
reference; do not commit its code or download links. No changes to OpenClaw,
grants, mirrors or external messaging were made. A passdown request authorizes
writing the handoff, not sending it to someone through messaging tools.

## 3. Host, deployment and access map

| Item | Location / last established configuration |
| --- | --- |
| Jetson | `tony-jetson`, Orin Nano Super, Ubuntu22.04 / JetPack6.2.1, Python3.10 |
| Pi | Raspberry Pi5,4GB, Debian12 / Python3.11; hardware controller and camera, no LLM installed |
| Direct link | Jetson `10.42.0.1`, Pi `10.42.0.23`; Jetson interface `enP8p1s0`, NetworkManager `robot-link` |
| SSH | `ssh witness-pi`, user `pi`, pinned host key, approved Jetson key, passwordless sudo |
| Active worktree | `/home/tony/witness_pi_guard`, branch `prep/pi-guard-service` |
| Checkpoint before this handoff | `2d41206a8fe06b02cf8b537cedc24bcf5fb5a088` |
| Pi release | `/opt/temple-witness/releases/20261012-capabilities-04`, selected through `/opt/temple-witness/current` |
| Guard | `temple-witness-guard.service`, authenticated TCP `10.42.0.23:9040`, permitted peer Jetson |
| Fork camera | `temple-witness-camera.service`, `http://10.42.0.23:8080/snapshot.jpg` |
| Permanent hardware limits | `/etc/temple-witness/hardware.json`, motion/outputs/sonar disabled |
| Pi persistent journals | `/var/lib/temple-witness/guard.jsonl`, `notes.jsonl` |
| Jetson session/sequence state | `/home/tony/.local/state/witness` |

The Pi address is DHCP-assigned; do not assume a reservation exists. Network
infrastructure and SSH were preserved through vendor retirement.

Protected credential locations, never contents: Jetson
`/home/tony/.config/temple-witness/guard.key`; Pi source
`/etc/temple-witness/guard.key`, delivered by systemd LoadCredential at runtime.
The client parses the local text key as hex bytes. Never print, copy into reports,
or commit credential values. Use the canonical Jetson sequence-state directory;
creating fresh sequence journals can cause replay/sequence rejection.

Vendor services `masterpi`, `hw_remote`, `hw_button_scan`, `hw_find` are persistently
masked. The phone listener was TCP9026; discovery used UDP9027. The button scanner
could launch hardware tests, so its old vendor shortcuts were retired too.
An actual reboot previously verified Temple guard/camera startup, disarmed state
and all vendor masks: `receipts/temple-reboot.json`. No automatic vendor rollback
exists. Original units are backed up on Pi under
`/var/lib/temple-witness-handover/20261012-handover-01`.

## 4. What has actually been demonstrated

See `CAPABILITIES.md` and its receipts for the full inventory. The assembled robot
was tested with the Jetson mounted. Later lateral/turn/diagonal tests were on a
smooth surface; do not generalize their traction to the earlier surface.

| Capability | Observed result and limits |
| --- | --- |
| Wheels | All four turned/stopped unloaded; eight translations and two turns eventually observed with stops. Initial floor pulses could be too slight to see. No measured speed, distance, turn angle, braking distance or payload envelope. |
| Arm | Small PWM3–6 movements and returns observed; later larger forward/up and mirrored left/right reaches observed with returns. Initial user report emphasized slight movement. |
| Gripper |1500→1800→1500 visibly opened farther and returned. During grasp,1700/1600 and then1500 closure contacted/moved the block but did not prove retention. No force or jaw-width measurement. |
| Camera / battery | Fresh fork frames and battery telemetry work; second external camera now works. |
| Sonar | Reads available after separate-I2C-transaction fix. Some earlier readings conflicted with clear space; later506–510mm and125–126mm readings cannot be treated as calibrated object localization. |
| Outputs | Beep and isolated light sequences physically observed. OLED writes acknowledged, no text observed. |
| Other peripherals | Bus servo queries failed; fitted bus servos not established. No IMU/button/gamepad/SBUS packets during passive observation. Piper WAV synthesis worked, default playback/capture failed. No working mic/speaker established. |
| Guard recovery | Stationary driver freeze test failed status in about0.502s, released serial, next session disarmed. No moving fault injection or whole-Pi/kernel-freeze stop guarantee. |
| Tool integration | Live status/snapshot/stop/local pending note exercised; speech failed playback. No live MCP/OpenClaw registration completed. Vendor autonomous vision modes remain unavailable. |

Observed wheel signs for channels1–4, multiplied by bounded duty:

| Direction | Signs |
| --- | --- |
| Forward / backward | `1,-1,1,-1` / `-1,1,-1,1` |
| Left / right | `1,1,-1,-1` / `-1,-1,1,1` |
| Forward-left / forward-right | `1,0,0,-1` / `0,-1,1,0` |
| Backward-left / backward-right | `0,1,-1,0` / `-1,0,0,1` |
| Turn left / turn right | `1,1,1,1` / `-1,-1,-1,-1` |

The observed forward convention reverses the vendor default label. Earlier
conflicting lateral reports are retained, not erased. Historical trials used
up to duty25 for0.5s. Persistent wheel ceiling remains0; this table enables no motion.

## 5. Cameras, target geometry and model limitations

The fork camera is640x480, roughly6fps in the earlier measured Temple stream.
Require advancing sequence and a fresh `X-Witness-Capture-Time`; it moves with
the arm/wrist. Temple serves raw images. Vendor Camera.py rectifies images.
Installed calibration and the vendor planar pixel map are saved in
`scratch/grasp-passive-sensors.json` (scratch prefix defined in section8).
Their present optical/extrinsic validity is not established. Do not use the
vendor planar map as a calibrated3D camera-to-gripper transform.

The added external camera is a Logitech C270 on Jetson `/dev/video0`; `/dev/video1`
is its paired interface, not a second independent camera. Open with OpenCV V4L2,
1280x720, small capture buffer, discard/warm frames, and release afterward.
A1280x960 request fell back to1280x720. It has no persistent capture service.
Tony repositioned it to show both tips, the floor and the whole block stack.
Upper arm portions can clip the top, but the contact region is visible.
Do not open concurrent captures or alter this useful view casually.

Tony described three light wooden blocks, each about one inch wide. The camera
shows green above red and blue side-by-side, not three blocks in a vertical column.
The target shifted and rotated during attempts, so earlier target pixels are stale.
Do not assume the final stack is at its starting coordinates. Cables are behind
it and a chair leg is to its left; neither camera supplies collision certification.

Installed software offsets: PWM1=0,3=59,4=72,5=63,6=-95 microseconds.
The downloaded vendor reference has different offsets: never overwrite the Pi's
`/home/pi/MasterPi/Deviation.yaml` from that copy. Board offset readback0 is distinct
from these software-added offsets. Nominal conversion is180degrees/2000us.
Effective vendor lengths are9.3,6.5,6.2,10.0cm. These are constants, not measured
assembled geometry. Pure Jetson planner: `companion/arm_lift.py`.

Historical upright commands were1=1500,3=754,4=2485,5=845,6=1405. They are not
current and must not be sent as a blind return. The reusable lift/reach planners
require their narrow upright start region. They are not general safe IK APIs.
Validate both nominal and offset pulses without clipping; sampled model corridors
still do not establish physical tracking, collision clearance or repeatability.

## 6. Grasp attempt: what happened and what did not

The consolidated receipt `receipts/arm-grasp-attempt.json` indexes24 raw initialize
receipts,52 recorded stages, command/readback results and camera hashes. It also
records the separately documented S preflight abort. This count includes a failed
M receipt; it is not24 successful grasps. Most operations were inspection,
alignment, short approach, release or clearance steps. No wheel motion occurred.

1. Restored the post-reboot arm with explicit attendance and external visual checks.
   Then moved to a downward survey pose, aligned base yaw from1405 to1520 and opened
   the gripper to1800. Camera centering alone did not establish jaw contact geometry.
2. Approached through short forward/down and vertical model steps, inspecting both
   views between batches. The scene remained clear until the tips reached green.
3. Closed partially to1700, tried a small model lift, then1600 and another lift.
   The block tilted/shifted, with at least one corner still close to the support.
   A subsequent modeled vertical lift settled/shifted it back. No complete pickup.
4. Coordinated commands and exact pulse reversals diverged from physical tip motion.
   Isolated shoulder2166→2116 over1s produced no clear lift. Tony reported **Quiet**,
   not buzzing/clicking. That observation is not proof of normal torque or mechanics.
5. Released to1800 with the block supported. After recovering a pre-arm startup
   failure, unloaded shoulder2116→2016 over0.6s visibly raised the tips clear.
   Reversing to2116 did not restore the same observed tip location. Further50us
   downward steps to2166 then2216 seated pads farther down the block.
6. Closed1800→1650→1500 over two0.6s stages. The block rotated. Shoulder2216→2116
   did not establish retention; pads appeared to catch near a corner and slip.
   Tighter closure alone did not solve the problem.
7. Another startup failure occurred before clearing. A fresh locked session verified
   unchanged commands; a new explicitly checked recovery opened the jaws to1800,
   then moved shoulder2116→2016. Final external image confirms open jaws clear of
   the supported block. Further live grasp attempts ended.

Do not turn these observations into a diagnosis of deadband, firmware interpolation,
loose horn, weak servo, incorrect offsets or backlash. Those are possible investigation
topics, not established causes. There is no force sensor or measured current here.
The user's loose-grip hypothesis was plausible; final evidence also implicates
contact placement. Do not solve this by progressively increasing closing force.

Representative raw prefixes: `grasp-survey`, `grasp-approach-a/b/c`,
`grasp-forward-d`, `grasp-seat-e`, `grasp-close-f/h/q`, `grasp-lift-g/i/j/r`,
`grasp-shoulder-k`, `grasp-release-l`, `grasp-unloaded-m/m2`, `grasp-reseat-n`,
`grasp-seat-o/p`, `grasp-clear-s2`. Inspect the consolidated index for exact names
and hashes; suffix `-initialize.json` is inherited harness naming, not proof of a
fresh board initialization. Do not import a runner to reuse its helper functions:
several have executable top-level hardware work. Read as text/AST data instead.

## 7. Controller fault and operating constraints

Repeated real-worker startup failure is **unfixed**. It appeared earlier around
11:36 and11:53, then17:28:31 and17:34:35 during this grasp session. Relevant path:
`companion/pi/service.py:95` constructs `Supervisor` before the session's try/finally;
`companion/pi/supervisor.py:78` calls `connection.recv()` for worker readiness.
EOFError escapes and systemd restarts the service disarmed. The child exit cause
was not established. Do not describe systemd recovery as a bug fix.

M failed after runtime setup but before arming/actions; finally removed its override.
S failed at the initial normal status before setup/arming/actions, so its original
runner emitted no normal receipt. `grasp-clear-s-aborted.json` records that fact.
Independent read-only recovery receipts verified unchanged commands before M2/S2.
No action with an unknown execution outcome was automatically resent.

The guard uses one supervised SDK Board per live connection, released on disconnect.
Even a status session constructs the worker and writes zero wheel duty at startup;
read-only status is not literally serial-free. `python3 -m scripts.witness_check`
from the worktree requests status and closes, but does open that supervised path.
For metadata-only inspection use SSH systemctl/journal/file reads.

Temporary commissioning used narrowly bounded joints, wheel0, buzzer only, no
sonar, a runtime systemd override and local operator signals. Normal startup has
no arm signal option. Authenticated actions and canonical sequence state were
required; stop/close and override cleanup followed each batch. Existing scripts
are evidence, not a production commissioning API. Some early failure paths do
not save a receipt or release capture in a single encompassing finally; improve
that before future reuse. Do not automatically replay any historical runner.

PWM moves cannot be cancelled mid-command by the SDK stop. Keep individual actions
bounded (the schema permits at most1s); zero wheel duty does not release PWM hold.
Disarmed means new movement is blocked, not that servo power is physically off.
Independent hardware cutoff remains the whole-Pi/kernel-freeze limitation.

## 8. Evidence, tests and repository status

Main files in `/home/tony/witness_pi_guard`:

| File | Purpose |
| --- | --- |
| `PASSDOWN.md` | This consolidated handoff |
| `PROJECT_STATE.md` | Dated latest state, older history below |
| `CAPABILITIES.md` / `ARM_POSES.md` | Inventory, observations, model scope |
| `DEPLOYMENT.md` | Services, credentials, persistence, deployment history |
| `receipts/arm-grasp-attempt.json` | Last grasp receipts and image hashes |
| `receipts/arm-postboot-recovery.json` | Attended recovery from zero command state |
| `receipts/arm-camera-low-battery.json` | Earlier camera work/low-battery stop |
| `receipts/vendor-mapping-facts.json` | Actual installed geometry/offset/source facts |
| `receipts/capability-live-summary.json` | Broad live commissioning observations |
| `receipts/temple-reboot.json` | Actual reboot persistence verification |

Raw scratch prefix throughout this handoff:
`/home/tony/witness_scratch/capabilities-20261012/`.
Room images and raw runners remain there, outside the repository. JSON receipt
and final image hashes were checked when the grasp checkpoint was committed.
The original incoming document is `/home/tony/WITNESS-PASSDOWN-CODEX.md`; its
old access restrictions and state must be read with subsequent user decisions.
Local memory: `/home/tony/WITNESS-CODEX-MEMORY.md`. Historical findings:
`/home/tony/WITNESS-CODEX-RESULT.md`. Control inventory:
`/home/tony/WITNESS-CONTROLS.md`. This handoff is also at
`/home/tony/WITNESS-HQ-HANDOVER.md` for HQ retrieval.

Last software test evidence: Jetson155 tests passed in
`receipts/arm-side-tests-jetson.txt`; Pi145 run,144 passed and1 skipped in
`receipts/capability-tests-pi.txt`. These are prior fake/stub software results,
not new grasp tests and not validation of physical IK. No runtime code was changed
or software suite rerun during the grasp or this documentation handoff.
Only documentation, receipt consistency and whitespace checks apply to this change.

Earlier commits: handover780fd19, capability7245de2, gripper6a3b2c5,
lift/reach490338e, side reachesef63e4b, low batterya4970df, recovery886475b,
grasp checkpoint2d41206. This handoff is committed separately. Pi runtime remains
capabilities-04; later pure Jetson planners/documentation do not imply a Pi redeploy.

Git push was authorized but earlier attempts failed for unavailable GitHub HTTPS
credentials; do not imply publication. Use existing per-command seat identity
`Codex Jetson seat <codex@localhost>` if local git lacks identity; no global config
change is needed. No push was retried for this handoff.

Stack proposal `de988f87-89dd-4dde-8d4a-665e30e424c1` from the earlier update request
was created, not committed. Older `e94dcc4b-eef7-40b6-9994-98c0091aacdb` is also not
assumed approved. The connector explicitly required Anthony's approval for commit;
its approve/commit commands were not executed. The latest grasp and this passdown
are durable local records, not a completed Stack chronicle update. No external
message was sent; HQ can retrieve the handoff through its existing route.

## 9. Recommended next work, in order

1. **Keep the robot locked while diagnosing startup.** Inspect journal timestamps,
   child exit status, worker startup duration, watchdog deadline and parent-death
   handling. Distinguish failed initialization from a client disconnect. Add focused
   fake-worker tests for early child exit/EOF, slow initialization and clean resource
   release. Catch failure at the proper session boundary with a bounded disarmed
   error. Preserve fail-closed behavior; do not add automatic action retry or loosen
   the heartbeat merely to hide the symptom. Root cause remains open.
2. **Make the commissioning harness dependable.** Put capture/session/override cleanup
   and an explicit outcome receipt around every failure phase, including initial
   status. Mark unknown action outcomes distinctly. Remove only its own override;
   distinguish inactive JSON from active configuration and remove owned stale data
   during a deliberate maintenance change. Never replay scripts on import/restart.
3. **With fresh attendance and clearance, establish physical joint response unloaded.**
   Use the external camera to observe each channel separately, both directions and
   returns, recording actual landmark displacement and settling. Validate wiring,
   joint roles, installed offsets and mechanics before attributing errors to one
   cause. Investigate the differing50us/1s and100us/0.6s observations without assuming
   those establish a universal timing rule. Check recent images/setpoints first;
   if rebooted, recover pose as unknown rather than reusing this table.
4. **Calibrate jaw placement before another grasp.** Observe the open jaw tips and
   block in the same frame, establish lateral and forward offsets, useful opening
   width and reachable contact region. Align around opposing faces, not a corner.
   Treat raw camera calibration and a constant-height IK request as unverified until
   checked against the actual robot. Do not rely on sonar to identify this block.
5. **Only then retry one light block.** Keep approach steps short, review images before
   contact, close within verified limits, and require visible full separation plus
   retention for pickup success. Set down supported and withdraw open. Preserve
   failed trials and images; neither an SDK acknowledgement nor pulse readback is
   proof that a block was lifted.

The remaining broad commissioning work includes measured loaded wheel travel,
turning response with Jetson mass, calibrated arm workspace and payload behavior,
and optional peripherals. General vendor modes/autonomous manipulation remain
unavailable until replaced with bounded Temple behavior and verified physically.

## 10. Last user-facing clarification

Anthony asked about two blue lights shining forward from the sensor area. They
were identified as the front ultrasonic module's RGB indicators, consistent with
earlier light tests and Hiwonder documentation. The visible blue light is an
indicator; range measurement uses ultrasound. Exact lamp identification from the
room view was not independently measured. References:
https://www.hiwonder.com/products/glowy-rgb-ultrasonic-sensor and
https://docs.hiwonder.com/projects/MasterPi/en/latest/docs/1.getting_ready.html.
No new lighting command was sent in response to that question.
