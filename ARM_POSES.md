Codex, gpt-6-astra, Jetson seat, 2026-10-12T17:37:27.588607-04:00

Current checkpoint: attended grasp attempts made contact and moved the green
block, but did not establish secure pickup. Gripper now open and clear, controls
locked. Current pulses1=1800,3=782,4=1803,5=2016,6=1520us, not home. Coordinated
model predictions/reversals diverged from physical motion; require calibration
and fix repeated startup EOF fault before further grasping. See PROJECT_STATE.md
and receipts/arm-grasp-attempt.json. No new runtime/software-test changes.

Codex, gpt-6-astra, Jetson seat, 2026-10-12T12:03:05.050685-04:00

# Bounded Temple arm poses

`companion/arm_lift.py` provides pure Jetson-side planning for bounded attended poses.
It imports no vendor module and accesses no hardware. The Pi still owns the SDK,
authentication, local arming, joint limits and driver supervision. This does not
add an unrestricted Cartesian action or enable autonomous arm control.

Installed vendor data supplies the geometry, channel roles and software offsets;
see `receipts/vendor-mapping-facts.json`. The planner uses ordinary linkage
geometry with those parameters, applies the offsets once, and validates both
nominal and final pulses. A target at (0,7,18)cm with level pitch is rejected
because its offset elbow pulse exceeds 2500us even though the nominal pulse fits.
It never silently clips a target.

| Pose | Planned motion | Operator result |
| --- | --- | --- |
| `plan_lift(start, 1)` | About 1cm up at constant horizontal position/pitch ; 0.8s out and 0.8s return | Rose, stayed roughly level, returned; very slight |
| `plan_reach(start)` | Fixed 5cm forward/2cm up; four 0.6s stages outward and four reverse stages | Clearly larger reach and returned |

The reference start is commanded PWM 3=754,4=2485,5=845,6=1405. Model-predicted
coordinates are approximately (0,6,18)cm, pitch 0 degrees. The reach endpoint is
PWM 3=986,4=2288,5=1273,6=1405, predicted approximately (0,11,20)cm. These are
**predictions from commanded pulses**, not position encoders or measured tip
coordinates. No payload, grasp or complete working-range test was performed.

The planner accepts only a narrow starting region. The lift is limited to
0.25–1cm and 200us per joint; reach stages are limited to 150us per joint, preserve
base yaw, and sample each interpolated transition inside a narrow predicted
corridor. The reverse uses the same waypoints. Checks do not model collisions,
backlash, tracking error, servo failure or the Jetson mounting geometry. Current
attended physical clearance remains necessary. PWM travel cannot be cancelled
mid-command by the SDK stop; each stage has a bounded duration.

The attended runner checked starting setpoints, battery and calibrated limits,
used signed guarded actions, checked readback between stages, then explicitly
stopped/disarmed and removed its runtime systemd override. Both tests returned
the commanded pulses to the original values. Operator reports establish visible
return; readback alone would not establish physical position.

Final state verified: wheel limit 0, no joints/outputs, disarmed, no operator-arm
signals, runtime override absent, all four vendor units still masked. Planner is
new Jetson-only code; deployed Pi release remains capabilities-04. No vendor mode,
stored trajectory, service rollback or OpenClaw registration was used.

Validation: 153 Jetson tests passed, including eight new planning tests. They
cover known upright geometry, round trips, offset-limit rejection, invalid and
out-of-region targets, sampled lift/reach bounds, and exact reverse waypoints.
Pi code was unchanged; its prior 144 passes/one skip remain the deployed result.
Receipts: `receipts/arm-lift-tests-jetson.txt`, `receipts/arm-poses-live.json`.
The first setup attempt failed at remote script parsing before any motion;
that failed receipt is preserved alongside the corrected successful run.

Planning example, no motion:

```python
from companion.arm_lift import plan_reach
plan = plan_reach({3: 754, 4: 2485, 5: 845, 6: 1405})
```

Scratch attended runners are outside the repository at
`/home/tony/witness_scratch/capabilities-20261012/`. The earlier pose runners require an explicit execution flag. Later one-off
camera runners execute when invoked; all refuse reuse of their receipt name. Do not automatically
replay them. A new session requires current readiness and a fresh starting check.

Codex, gpt-6-astra, Jetson seat, 2026-10-12T11:32:35.886647-04:00

## Side reaches, observed and returned

`plan_side_reach(start, "left" | "right")` adds a fixed 3cm sideways component
to the existing 5cm-forward/2cm-up reach. It uses four 0.6s stages outward and
the exact reverse home, with at most 150us per joint per stage and 20degrees
of base rotation. Predicted transitions are sampled against a narrow corridor;
this is not a collision or payload model. Unknown side names and uncommissioned
starting poses are rejected. Limits include every waypoint when configuring
this test's temporary joint allowances.

Anthony confirmed left/forward and return, then right/forward and return.
Predicted endpoints are approximately (-3,11,20)cm and (3,11,20)cm from the
starting (0,6,18)cm. Target PWM3/4/5 are1005/2280/1301; PWM6 is1575 left and
1235 right. Both returned to754/2485/845/1405. These distances remain model
predictions; only qualitative direction and return are physically verified.

155 Jetson tests now pass, including mirrored side poses, exact reverse paths,
per-stage limits and invalid side/start rejection. Pi runtime code is unchanged.
After both attended tests: override absent, normal configuration loaded,
disarmed, wheel ceiling0, joints{}, outputs[], serial released on disconnect,
no operator-arm signals, all four vendor masks intact. Evidence:
`receipts/arm-side-live.json` and `receipts/arm-side-tests-jetson.txt`.


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
