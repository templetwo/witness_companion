Codex, gpt-6-astra, Jetson seat, 2026-10-13T07:03:00-04:00

Anthony reports "review done" for the preceding guard-floor deployment.
Review completion is now recorded from his report; no review findings or
requested revisions accompanied it. Deployed source last verified a9197fb,
release 20261013-guard-floor-01; deployment evidence commit b4bc895.
This checkpoint changes documentation only. Runtime status was not remeasured;
last verified state was disarmed, floor 7000 mV, wheel ceiling 0, no joints.

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

## Newest bound complete in source; HQ review required before deployment

Implemented all three software changes from
`docs/hq/PASSDOWN_2026-10-12_NEXT_BOUND.md`, in three separate commits:

- 1a, `b53bea6`: cached battery floor for every motion command; 158 -> 174 tests.
- 1b, `0b48825`: existing bounded validation for companion drive/joints; 174 -> 180.
- 1c, this commit: MCP speech capped at 200 characters, proposed notes at 2000;
  180 -> 188. Over-limit requests return standard schema error -32602 before the
  adapter. Boundary values are accepted. No speech chunking was added.

Each stage passed the complete `python3 -m unittest discover -s tests -v` suite
on Jetson Python 3.10.12 and Pi Python 3.11.2. Pi tests ran only in the isolated
`/tmp/hq-eof-check` checkout; changed Python file hashes match the tested Jetson
source. Receipts are `receipts/next-bound-20261013/1a.json`, `1b.json`, `1c.json`.
Each receipt identifies its parent/base commit and records hashes for that stage;
use that stage's commit when checking hashes of documents changed again later.

Pre-fix regression groups failed as expected. The baseline cbdbbca code is identical
to 04e6de4 (cbdbbca adds only the HQ passdown). Before 1b and 1c, their respective
production modules were still identical to 04e6de4. Exact red commands:

- `python3 -m unittest tests.test_pi_service.BatteryFloorTests -v`
- `python3 -m unittest tests.test_tools.ToolSurface -v`
- `python3 -m unittest tests.test_witness_mcp.InProcessTests tests.test_witness_mcp.SubprocessTests -v`

Full and red logs are under `/home/tony/witness_scratch/next-bound-*.log`.
No deployment unit changed. No real SDK/vendor-image read, robot action, arming,
override, signal, guard restart, live release change, Stack write or GitHub push.
The new floor is source-only until reviewed and deployed. The existing startup
failure diagnostic fix 8a657b4 is already in source, also still undeployed.

HQ delivery bundle will be created immediately after this commit at
`/home/tony/witness_scratch/bundles/20261012-next-bound.bundle` using
`git bundle create <path> prep/pi-guard-service ^04e6de4`; it includes the incoming
cbdbbca passdown plus the three implementation commits. HQ can review each with
`git show <commit>` and carry the push. Bundle verification and SHA256 go in
`/home/tony/WITNESS-CODEX-RESULT.md` to avoid changing the bundle's own contents.

Carry forward the newer HQ physical state, not the superseded 17:45 state below:
last reported commands PWM 1=2000, 3=782, 4=1803, 5=2116, 6=1420; jaws open,
arm not home; last battery 6984 mV. No pickup was achieved. These are historical
HQ observations, not fresh measurements. Prior Stack proposals were committed by
HQ (e94dcc4b/de988f87); no new proposal was sent by this seat.

Next: HQ review, then a separately authorized release cut carrying 8a657b4 and the
battery gate. The handoff's proposed deployment procedure, network changes,
mechanical inspection, calibration and grasp attempts remain outside this bound.
No inherited attendance statement authorizes new motion.

Codex, gpt-6-astra, Jetson seat, 2026-10-12T20:38:40-04:00

## Next bound 1b: structured companion motion, ready for HQ review

`robot_request_action` now validates `drive` and `joints` through the existing
`controls.validate` and passes the same params to the coordinator. Unknown names,
non-object params, text shape/length and remaining no-param commands keep their
refusals. The coordinator still enforces its guard; accepted tool input does not
prove physical motion. No new controls, arming path or bounds were introduced.

Six new tests bring the suite from 174 to 180, green on Jetson Python 3.10.12
and the Pi Python 3.11.2 scratch checkout. The pre-fix tool test group failed
six assertions. Receipt: `receipts/next-bound-20261013/1b.json`.
First bound remains separately reviewable at `b53bea6`. This commit is 1b only;
1c (MCP text limits) follows. No deploy, guard restart, robot action, real SDK
read or Stack write. Physical state below is inherited, not remeasured.

Codex, gpt-6-astra, Jetson seat, 2026-10-12T20:35:40-04:00

# Newest bound implementation: 1a ready for HQ review, not deployed

Read `docs/hq/PASSDOWN_2026-10-12_NEXT_BOUND.md` and incoming passdown sections10–13.
The later HQ state supersedes my earlier physical-state table below: last reported
PWM1=2000,3=782,4=1803,5=2116,6=1420; battery floor hit6984mV; no pickup. No hardware
state was remeasured for this software task. HQ subsequently pushed the earlier
commits and committed both older Stack proposals. Their earlier pending status
below is historical. This bound makes no new Stack write and no GitHub push.

1a adds a7000mV default floor to every motion command after guard.check and before
any action IPC. Missing, invalid, future-dated or older-than5s telemetry is unknown
and refused; exactly7000mV and exactly5s pass. Stop and all nonmotion outputs bypass
this battery gate. Refusal keeps arm state and the driver alive; signed sequence
consumption remains the existing guard behavior, so use a new request, never replay.
The floor is configurable by --battery-floor-mv and visible as battery_floor_mv.

Implementation detail needed for review: the old supervisor did not retain status
replies. It now caches a defensive copy; HardwareBoard.status exports the existing
monotonic read stamp. The gate reads only this cache, doing no new worker/sensor
I/O. A caller must obtain status first and keep its battery sample within5s.
Monotonic stamps apply only on the Pi host; packet/sample acquisition age remains
unknown as before. No healthy battery is invented when telemetry is absent.
No deployed unit, live service, SDK, device, override or arm signal was touched.

Baseline158 Jetson tests passed. New battery regressions fail against baseline code;
current174 tests pass on Jetson3.10 and Pi3.11 scratch clone. The old /tmp/hq-eof-check
was absent; a new unprivileged clone was made there from a bundle. Receipt1a.json
under receipts/next-bound-20261013 records counts, interpreters and file hashes.
HQ review remains required before deployment. Remaining bound1b and1c follow in
separate commits, with separate receipts and diffs.

---

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

Claude, claude-opus-5-5, Jetson seat, 2026-10-13T19:3x-04:00

## Section 9 item 1, first part: a failed driver start ends one session, not the service

On Anthony's word in the Jetson Claude session ("Work Codex's next items"; recorded in
~/.claude/PASSDOWN.md 19:09). Code and fake tests only: no deploy, no Pi change, no motion,
no push. `serve_connection` built `Supervisor` before its try/finally, so any startup
failure escaped, skipped cleanup and ended the whole service (systemd restart). Now the
construction sits inside the session boundary: on ConnectionError/TimeoutError the session
sends one fixed unsigned refusal `{'ok': False, 'protocol': 1, 'armed': False,
'executed': False, 'reason': 'driver startup failed'}` (no exception text on the wire; the
Supervisor's stderr line still carries cause, exitcode and watchdog_reason to the journal),
closes disarmed, and the listener keeps accepting. No retry was added. `PiLink` raises
ConnectionError naming the refusal instead of "invalid server greeting".
Tests: 5 new in `SessionStartupFailureTests` (factory exception, silent exit, slow start
bounded with the child released, clean next session, client names the refusal). Red before
the fix (1 failure, 4 errors), green after: 193 OK on Jetson 3.10.12 (188 before). Receipts:
~/witness-opus-notes/session-boundary-{red,green}.log.

**Root cause is still open.** This contains the failure; it does not explain it. Two leads
from reading, unverified on hardware:
1. The fake slow-start run logs `exitcode=-9, watchdog_reason=Jetson heartbeat missed`: the
   lease deadline starts at Supervisor construction and the startup budget is the same
   2.0 s, so a slow start is killed by the watchdog. HQ's lead in section 11 matches.
   The 10-12 failures died under 1.2 s, so this may not be them.
2. `PiLink` waits `timeout=1.0` s for the greeting, which the server sends only after the
   driver is ready. A real start slower than 1 s makes the Jetson give up first.
guard-floor-01 carries 8a657b4, so the next real failure names its cause in the journal.
Read that before changing lease timing.

Claude, claude-opus-5-5, Jetson seat, 2026-10-13T20:4x-04:00

## First block stacked (session opus-01), on Anthony's word, him at the switch

Attended session from the Jetson Claude seat: wake after power cycle, a grasp/stack loop through the
deployed guard (20261013-guard-floor-01) with per-batch systemd commissioning overrides, plus forward
and backward drive pulses Anthony approved. Green block picked up (20:16, again 20:35) and stacked on red
at 20:42; verified by the C270 after release (jaws empty, green on red). Photo by the robot's C270 at
20:44: `receipts/opus-01/first-stack-2026-10-13.jpg`. Receipts: `receipts/opus-01/session.json` (82
batch summaries with sha256 of each raw receipt; raw frames and the runner stay in
~/witness-opus-notes/opus-01/ on the Jetson). Method, measured signs and hazards:
`receipts/opus-01/LEARNING.md` (draft for a skill). Plan and approvals: `receipts/opus-01/SESSION_PLAN.md`.

Key findings for the next bound: the jaws land ~2 in short of the claw camera's aim; one forward pulse
(duty 20, 0.5 s) moves ~2 in on the rug; elbow reversal backlash ~100 us; shoulder ceiling 2500.
Startup refusal root cause now named in the journal 4x: "factory failed, RuntimeError: serial ownership
changed". ModemManager is active on the Pi (serial prober): leading suspect. Fix is a Pi system change
for Anthony/HQ (mask it or udev ID_MM_DEVICE_IGNORE for ttyAMA0), then deploy 4f2468b.
No code changed in this commit. Not pushed. End state: guard disarmed, override removed, arm raised.
