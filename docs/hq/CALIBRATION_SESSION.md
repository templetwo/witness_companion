# Calibration session: pulses to what the arm actually does

HQ, 2026-10-12. Drafted on the ARM_SESSION_PLAN_TEMPLATE. Runs only after the
guard worker fails loud (see the startup EOFError fix) so a dead worker cannot
be mistaken for a quiet arm. Anthony approves the filled plan before power-on.

## Why this comes before the next grasp
Three measured facts from 2026-10-12: command readbacks are setpoints, not joint
angles; modelled coordinated lifts diverged from what the external camera showed;
there is no camera-to-gripper mapping. Every grasp attempt so far was flown on the
model. This session replaces the model's guesses with a measured table.

## Goal
A table, per joint, of commanded pulse versus observed angle from the external
camera, a gripper gap-versus-pulse table, and one measured fork-camera pixel
position for a block at a known place. Nothing is lifted.

## Preconditions
- Battery above 7300 mV, arm clear, wheel ceiling 0, Anthony at the switch.
- External C270 on the Jetson framed from the side so the base, shoulder, elbow,
  wrist and fork are all visible with a plain background behind the arm. Tape a
  ruler or a printed protractor on the background at the arm's plane.
- Start from the known upright command pose 1=1500, 3=754, 4=2485, 5=845, 6=1405
  (reached one joint at a time with camera checks, as on 2026-10-12 17:05).

## Batches
Joint order: base yaw (6), shoulder (5), elbow (4), wrist (3), gripper (1).
For each joint, from the upright pose, in steps of 100 us, 0.6 s per step,
out to the vendor limit or the first sign of strain, then back by the same
steps; a side-view frame captured after every step, pulse and frame filename
logged together. Return the joint to its upright value before the next joint.

| # | Joint | Pulses | Capture | Expected | Stop rule |
|---|---|---|---|---|---|
| 1 | 6 base | 1405 -> 1105 -> 1405 -> 1705 -> 1405 | side frame per step | base turns, nothing else moves | any other joint moves, or buzzing |
| 2 | 5 shoulder | 845 -> up to 1445 and back | side frame per step | upper arm rotates about the shoulder | tip nears floor or chassis, buzzing |
| 3 | 4 elbow | 2485 -> down to 1885 and back | side frame per step | forearm rotates about the elbow | tip nears floor or chassis |
| 4 | 3 wrist | 754 -> up to 1354 and back | side frame per step | fork pitch changes only | fork touches anything |
| 5 | 1 gripper | 1800 -> 1500 in 50 us steps, then open | fork-camera and side frame per step, ruler between the jaws | jaws close evenly | servo buzz with no jaw motion (mechanical, see GRIPPER_MECHANICAL_CHECK) |
| 6 | block | arm at upright pose, one block at a marked spot 8 cm in front of the base centre | fork-camera frame | block visible | none (no motion) |

Batch 5 answers the loose-grip question with numbers: jaw gap in mm per pulse.
Batch 6 gives the first camera-to-world point; repeat at 12 cm if time allows.

## What to compute afterwards (Jetson side, no hardware)
- Per joint: angle per 100 us from the frames (two people can read a protractor
  overlay; one script can do it later). Note any dead band at the start of travel.
- Compare each measured angle with `companion/arm_lift.py`'s prediction for the
  same pulse; record the difference per joint. That difference is the fix the
  planner needs, not a new model.
- Gripper: pulse at first contact of the jaws and pulse at full closure on a
  known block width (the blocks are the reference).

## End state
Upright pose restored and read back; guard locked, disarmed, overrides removed;
receipt JSON hashing every frame; two sentences on what the table says.
