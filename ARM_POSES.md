Codex, gpt-6-astra, Jetson seat, 2026-10-12T11:27:08.153274-04:00

# Bounded Temple arm poses

`companion/arm_lift.py` provides pure Jetson-side planning for two attended poses.
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
`/home/tony/witness_scratch/capabilities-20261012/`. They require an explicit
execution flag and refuse reuse of their receipt name. Do not automatically
replay them. A new session requires current readiness and a fresh starting check.
