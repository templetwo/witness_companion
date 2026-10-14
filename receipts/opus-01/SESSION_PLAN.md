# Attended arm session plan: opus-01 (wake after reboot, then gripper gap vs pulse)

Filled from HQ's ARM_SESSION_PLAN_TEMPLATE.md by claude-opus-5-5, Jetson seat, 2026-10-13 19:4x EDT.

## Header
- Session name / date / seat: opus-01 / 2026-10-13 / claude-opus-5-5, Jetson Claude Code seat
- Approved by Anthony (how and when, verbatim word): "Approve; waive gripper check", structured question in the Jetson Claude session, 2026-10-13 19:4x EDT. Gripper mechanical check waived for this session (no block is gripped).
- Battery at start: 7948 mV at 19:35 (must be above 7300 before any arm move; stop at 7000)
- Guard release deployed: 20261013-guard-floor-01 (4f2468b is NOT deployed; not needed)
- Arm start pose, commanded pulses 1/3/4/5/6: readback 0/0/0/0/0 at 19:35 (post-reboot zero
  state). Last commanded pose before power-off (10-12 19:00): 1=2000, 3=782, 4=1803, 5=2116, 6=1420.
- External camera placed and framed: C270 at floor level forward, sees both jaws and the red and
  blue blocks (Anthony confirmed jaws 19:3x). Claw camera looks straight down. Yes.

## Goal, one sentence
The arm holds a known commanded pose again after the reboot, and we have a measured table of
gripper pulse versus jaw gap from the C270, with no block touched.

## Preconditions
- Gripper mechanical check (GRIPPER_MECHANICAL_CHECK.md, power off, Anthony's hands, ~10 min):
  done and recorded, OR Anthony explicitly waives it for this session because no block is gripped.
- Floor clear around the robot; wheels cannot move (wheel ceiling 0 in every override).
- Anthony at the switch. Stop path verified: signed stop round trip done 19:35, applied.
- Pi NTPSynchronized=yes checked immediately before setup.
- Status read within 2 s before every batch; battery at or above 7300.
- No scratch runner replays. One new runner, written for this plan, refuses if its receipt
  exists, removes its own override in a finally, and records the outcome of every phase.

## Batches
Part A, wake (each batch is one joint, from 0 to the last commanded value; the 200 us rule
cannot apply to a wake from 0, which is why each is a single joint with the cameras watching):

| # | Joint and pulses | Duration | Expected on camera | Stop rule |
|---|---|---|---|---|
| A1 | 5 shoulder 0 -> 2116 | 1.0 s | little or no movement (holds its sagged pose up) | any jump visibly larger than a few cm, or buzzing |
| A2 | 4 elbow 0 -> 1803 | 1.0 s | little or no movement | same |
| A3 | 3 wrist 0 -> 782 | 1.0 s | little or no movement | same |
| A4 | 6 base 0 -> 1420 | 1.0 s | little or no rotation | any rotation larger than a few degrees |
| A5 | 1 gripper 0 -> 2000 | 1.0 s | jaws open (2000 cleared the block on 10-12) | jaws hit anything |

Part B, gripper gap (jaws in the air, no block between them; C270 and claw frame after each step):

| # | Joint and pulses | Duration | Expected on camera | Stop rule |
|---|---|---|---|---|
| B1 | 1 gripper 2000 -> 1900 -> 1800 | 0.6 s per step | jaws close evenly, gap shrinks | uneven jaws, or buzz with no motion |
| B2 | 1 gripper 1800 -> 1700 -> 1600 | 0.6 s per step | gap shrinks further | same, or jaws meet |
| B3 | 1 gripper 1600 -> 1800 -> 2000 | 0.6 s per step | jaws reopen to the A5 gap | jaws do not return |

Rules: one joint at a time. A batch that does not look like its "expected" column ends all motion
for the session. The arm is never moved toward the blocks. 1500 (the 10-12 closing value) is not
commanded; B stops at 1600.

## End state
- Commanded pulses at end: 1=2000, 3=782, 4=1803, 5=2116, 6=1420 (the known pose, jaws open).
- Guard: disarmed, wheel 0, override removed, normal ExecStart (verified by status read).
- Receipts: ~/witness-opus-notes/opus-01/ (json per batch, all frames, hashes).
- What was learned, two sentences, written before power-off.

## Result (written 19:4x EDT, before power-off, by claude-opus-5-5)
- A1 shoulder 0 -> 2116, 19:38:02: matched expected. Claw view 0.0% pixels changed (arm held),
  C270 3.9% (exposure noise). Battery 7897 mV. Applied; readback 5=2116; locked after.
- A2 elbow 0 -> 1803, 19:38:37: DID NOT match "little or no movement". Claw view 42.5% changed:
  the arm rose several cm (before: two blocks filling the frame, blurry; after: green, blue and red
  blocks whole and centred from higher). The jaws left the top corners of the C270 view. No contact.
  Applied; readback 4=1803, 5=2116; guard locked, disarmed, override removed.
- Per the plan's rule, motion ENDED for this session at A2. A3, A4, A5 and B1-B3 not run.
- End state: 1=0, 3=0, 4=1803, 5=2116, 6=0 (wrist, base, gripper uncommanded). Guard disarmed,
  wheel 0, joints {}, override removed.
- What was learned: after a power cycle the elbow sags, so the last commanded pose is not the
  physical pose and waking to it moves the arm; the next wake must pick each target from the camera
  view (Codex's 10-12 method), not from the last readback. The image-diff check works as a stop
  signal: 0.0% on a held joint, 42.5% on a real move.
- Receipts: ~/witness-opus-notes/opus-01/{A1,A2,A1-dry}.json and frames; diff.py.

## Grasp attempt (Anthony 19:4x: "grab a block ant stack it on another"; amended wakes included)
- A4b base 0->1420 (19:41), A5b gripper 0->2000 (19:42), A3b wrist 0->782 (19:42): all small moves.
  Pose 1=2000 3=782 4=1803 5=2116 6=1420 (HQ's 10-12 end pose).
- Measured image signs in the claw view (camera looking down, jaws at bottom corners):
  yaw +50 us: blocks slide about 65 px RIGHT. shoulder +100: blocks move about 45 px UP and grow.
  elbow +100: blocks move about 120 px UP; in the C270 the jaw comes lower and nearer the robot.
  So shoulder+ and elbow+ both bring the gripper down and BACK toward the robot.
- G1 yaw 1420->1470, G2 yaw 1485 + shoulder 2216, G3 shoulder 2316 + elbow 1903, G4 elbow 2003.
  A G3 dry run with a wrong start (my typo) was refused: start-pose check proven to fail when it should.
- Blocks are beyond vertical-gripper reach (about 12 cm from base by vendor link lengths). Paused at
  1=2000 3=782 4=2003 5=2316 6=1485, guard locked, asked Anthony to slide blocks closer.
- 19:49-19:50: G5-G9 (wrist tilt, descent, reach-out probes). G8 stage 2 and G9 (elbow decrease 200 us)
  produced no visible change in either camera. Motion stopped at G9 per stop rule. Pose
  1=2000 3=1022 4=1803 5=2316 6=1485, guard locked. Details: opus-01/LEARNING.md.
- 19:41-20:42: grasp and stack, on Anthony's "grab a block and stack it on another" plus approvals for
  4 forward pulses and his reverse hint. Grab 1 at 20:16 (green held), stack attempt 1 fell beside red,
  grab 2 at 20:35, STACK SUCCESS at 20:42: green on red, verified in the C270 after the jaws lifted away.
  End pose 1=1800 3=1142 4=2253 5=1946 6=1485, guard disarmed, override removed. Full log: opus-01/LEARNING.md.
