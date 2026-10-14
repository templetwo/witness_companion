# Grasp learning log (raw material for a future skill), opus-01, 2026-10-13
Each line: move -> what the cameras showed. Claw cam looks down, jaw tips at its bottom corners.
C270 at floor level on the chassis front, under the arm, same handedness as the claw cam.

## Wake after power cycle
- Joints sag when unpowered. Waking to the last commanded pose moved the elbow several cm (A2).
  Wake order that worked: shoulder, elbow, base, gripper, wrist. Choose targets from the cameras.
- Image diff on the claw cam is a good motion witness: 0.0% changed on a held joint, 40% on a move.

## Signs (from pose 1=2000 3=782 4=1803 5=2116 6=1420)
- base +50 us: claw view slides ~65 px right.
- shoulder +100: claw view blocks ~45 px up and larger (gripper lower).
- elbow +100: claw view blocks ~120 px up; C270 shows the jaw lower and nearer the robot.
- shoulder+ and elbow+ both bring the gripper down AND back toward the robot.
- wrist +100 (782->882): claw view blocks ~110 px DOWN (camera and jaws tilt outward); C270 jaw moves only slightly.

## Reach
- With the gripper vertical, wrist reach at block height is about 12 cm from the base (vendor links).
- Blocks in front of the C270 at hand-span distance were beyond that reach (G4).

## Tooling lessons
- C270 first ~15-30 frames are black (auto-exposure). Discard and check mean brightness.
- Start-pose check in the runner refused a typo'd start: keep it.
- G6 wrist 982: red centred in claw view. G7 shoulder 2416 + wrist 1022: red back near the top; in the
  C270 the right jaw became huge and low, tip at block mid-height BESIDE the blue block. Big = close to
  the C270: the gripper came down BETWEEN the robot and the blocks, not over them.
- LESSON: centring a block in the claw view does not put the jaws over it once the arm is low and the
  wrist tilted. Use the C270 jaw SIZE as the depth cue (bigger = nearer the robot).
- G8: shoulder -100 moved; then elbow 2003->1903 (reversing direction) showed 0.0% change in BOTH cameras with readback 1903: backlash/deadband on reversal under load. Reversals may need >100 us before anything moves.
- G9: elbow 1903->1803 ALSO 0.0% change in both cameras (200 us total decrease, nothing visible).
  Motion stopped here (did not match expected). Candidates: a jaw resting on the blue block (C270 showed
  the tip at block-top height over it), elbow slipped horn/stall on decrease, or large backlash.
  Needs Anthony's eyes on the arm. LESSON for the skill: after every batch, require visible change
  when a move was commanded; 0% after a commanded move is a stop signal, not progress.
- GROUND TRUTH (Anthony, 19:5x, eyes on the arm): at pose 1=2000 3=1022 4=1803 5=2316 6=1485 the gripper is about 4 in (~10 cm) in front of the C270. In the C270 at that distance the jaw appears large and blurred; the blocks are farther.
- GROUND TRUTH (Anthony): blocks about 6 in from the C270, jaws about 4 in: reach gap about 2 in (5 cm). Elbow decrease likely at its limit ('might be the limit').
- G10 shoulder 2216 + wrist 1122: jaws up and out (C270: only tips at top edge); claw: red between jaw tips.
- DRIVE D1 (forward, duty 20, 0.5 s, on the rug, arm raised): robot moved about 2 in (5 cm), judged by the
  red block growing ~85 -> ~130 px wide in the C270 (6 in -> 4 in). NOT "slight". One pulse closed the gap.
- Shoulder pulse ceiling is 2500 (runner range check refused 2516 before any setup). From 2416 only ~84 us of shoulder descent remains.
- ATTEMPT 1 (19:57-19:58): green centred in claw view, shoulder 2496, elbow 1953, wrist 1202, closed 2000->1600.
  Contact: green rotated ~15 deg, blue moved. Lift (shoulder -200) showed jaws empty in the C270; green on floor.
  The jaws had come down on the red/blue edges, which were at the BOTTOM of the claw view.
- LESSON: the grasp point in the claw view depends on wrist tilt. Wrist 782 (vertical, HQ 10-12): image centre.
  Wrist ~1200 (tilted out): near the bottom of the claw view, between the visible jaw tips. Aim there.
- C270 is the best grip witness: closed empty jaws show as two pad tips close together with nothing between.
- ATTEMPT 2 (19:59-20:00): G17 open; G18 yaw 1545 + shoulder 2496: C270 shows right jaw at the red/blue
  junction at block-top height, left jaw beyond red's left side (red between jaws). G19 elbow 1953->2053:
  0.0% change in both cameras (no backlash expected, same direction as G14). Suspected jaw resting on the
  block edge = stall. G20 elbow back to 1953 to unload (also 0.0%). Asked Anthony to look.
- LESSON: when a jaw tip sits exactly at block-top height in the C270, a further descent can land ON the
  block edge. Centre the gap between the jaws on the block before the last descent, using the C270.
- GROUND TRUTH (Anthony 20:0x): the right jaw was blocked by the blue block. Red and blue touch, so there is no gap for a jaw on red's blue side. LESSON: a target block needs a jaw-width gap on both gripping sides; check spacing first.
- 20:0x Anthony: 'im not touching... move blocks if you wish'. Target switched to GREEN (free on its left; behind the red/blue junction). Plan: drive one pulse, yaw back to 1485 (where jaws centred on the junction), descend.
- D2 (20:06, 2nd pulse): overshot. C270 filled by red (chassis at or touching red); claw view shows no blocks (they are under/behind the jaws). Two pulses = about 10 cm; one was enough.
- Yaw gain depends on height: low (shoulder 2496) ~1.2 px/us at 320-px view (about 2.4 px/us full-res), vs ~0.65 px/us full-res high up. Re-measure gain after each height change.
- ATTEMPT 3 (20:06-20:09): D2 overshot; G22 descend brought green back at bottom of claw view; yaw 1445 (overshot
  left), 1490 centred green between jaw tips; elbow 2153 put the camera right over green (fills view); closed
  2000->1600: green barely moved (5.9%); lift: green stayed on floor, rotated ~30 deg, jaw tips together.
  Pads touched and twisted it. Probable: pads at top edge and/or 1600 not closed enough for a 3 cm block.
- LESSON: "block barely moved on close" is NOT proof of a hold; only the lift test is.
- GROUND TRUTH (Anthony 20:1x): camera aim is spot on, but the claw lands ~2 in SHORT (toward the robot) of the camera's aim point. Explains attempts 1-3. FIX: centre the camera on the block at the low pose, then move the gripper ~2 in outward (here: one forward drive pulse) before closing.
- D3 (20:12): chassis pushed red against the C270 lens (C270 view solid red, blind). Claw view: rug only.
- G29: runner stopped on 'stale claw frame' after the move (transient; next snapshot fresh). Finally-block cleanup verified by hand: stop applied, disarmed, override dir empty, normal ExecStart. The guard on exceptions works.
- G31 elbow 2253: 0.0% (pads resting on rug or block top). G32 back to 2153 to unload.
- *** ATTEMPT 4: GREEN HELD (20:16-20:17) ***
  Sequence: centre camera on green at the low pose -> raise -> one forward pulse (~2 in, Anthony's offset) ->
  descend (shoulder 2496) -> yaw so green sits at the bottom centre of the claw view between the jaw tips ->
  elbow +100 stalled (0%) = pads resting low -> unload elbow -100 -> close 2000/1800/1600/1400: the step
  1600->1400 showed 0% change (jaws stopped on the block) -> lift.
  PROOF OF HOLD: second lift (shoulder 2296->2096) changed the claw scene completely (far room visible) while
  green kept its image position and width (x 175-539 -> 181-531, within 2%). A floor block would shrink.
- SKILL SIGNALS: (1) grip closing step that shows 0% change = jaws stopped on something; (2) after two lifts,
  block bbox stable while background changes = held.
- G39 elbow 2353 applied; runner stopped on 'C270 too dark (14.8)' after the move (red at lens). Cleanup verified. C270 floor lowered 20->8 (warm-up black reads ~2).
- Anthony hint 20:2x: 'lift the block back up all the way then reverse about 5 inches then look again'. Taken as approval for 2-3 backward pulses (duty 20, 0.5 s).
- *** ROOT CAUSE CAPTURED (20:24:00, R1 first connect after override restart) ***: guard journal:
  "driver startup failed in child: reason=factory failed, exc_type=RuntimeError, exc=serial ownership changed;
  child exitcode=1, watchdog_reason=None". RealBoardFactory's second inspect_owners found another holder of
  /dev/ttyAMA0 besides its own pid. Deployed release lacks 4f2468b, so the service process exited; cleanup's
  restart recovered it (disarmed, override removed). No motion was sent in R1.
- Other ttyAMA0 holder NOT a getty (serial-getty is on ttyAMA10/serial0). Nothing holds ttyAMA0 at rest. Candidate: a previous session's or old instance's worker still releasing the port when the new worker's second ownership check ran. For HQ: log holder pid/comm/cgroup in the 'serial ownership changed' message.
- R2+R3 (2 backward pulses, ~4 in): C270 clear again: red close front-left (~3-4 in from lens), blue at right edge. Green still held. Stacking guided by the C270 (claw view blocked by the held block).
- STACK ATTEMPT 1 (20:26-20:28): after reversing, C270 + claw guided green over red (claw: red sliver above
  green; C270: block-shaped blob on red). Opened 1400->1800: green fell BESIDE red (left, tilted). Probably
  still above red and slightly toward the robot; the C270 blob was likely the jaw, not green.
- LESSON: before release, (1) lower until a small stall (0% change) shows the held block resting on the
  target, (2) the target must be fully hidden under the held block in the claw view, (3) in the C270, look for
  the held block's own colour, not just a shape, since the sonar LEDs tint things blue.
- REGRAB: G45-G47 yaw onto green then descend with wrist comp. Green ended at TOP of claw view (camera aim), jaws ~2 in short again. Fix: raise, D4 forward pulse, descend.
- 2nd capture 20:31:34: same 'serial ownership changed', 12 s after the guard started (not a restart race).
  ModemManager is ACTIVE on the Pi: a known serial-port prober. Leading suspect for the transient second
  holder of ttyAMA0 (open thread from 10-12). Fix candidate (Anthony/HQ decision, Pi system change):
  mask ModemManager or add a udev rule ID_MM_DEVICE_IGNORE=1 for ttyAMA0. Then the 4f2468b deploy
  so a refused start no longer kills the service.
- *** GRAB 2: GREEN HELD (20:35) ***: raise -> D4 forward -> descend (shoulder 2246 + wrist 1082) -> yaw to
  centre green at bottom of claw view -> elbow +200 (backlash) +100 until green big at bottom -> close
  1800/1600/1400 (last step 0% = stopped on block) -> double lift: green bbox x 121-580 -> 122-578,
  px 72074 -> 71171 while scene changed 35-40%. Repeatable: 2 of 2 using the offset-corrected recipe.
- STACK 2 CONTACT SIGNAL: held block width in claw view constant (457-458 px) while lowering G57-G59 (rigid with camera), then GREW to 508 px at G60: block stopped moving = resting on target. Use this as the release trigger.
- G62 (20:41:39): 4th 'serial ownership changed', on the connect right after the override restart. Added one reconnect there too.
- *** STACK 2: SUCCESS (20:41-20:42) ***: G56-G60 lowered green onto red with C270 + claw guidance; contact
  detected at G60 (green width in claw view jumped 458 -> 508 px); G61 opened 1400->1800; G62b lifted:
  C270 shows green still sitting on red, claw shows empty jaws. Pose at end 1=1800 3=1142 4=2253 5=1946 6=1485.

## RECIPE THAT WORKED (draft for the skill)
1. Wake after power cycle joint by joint, targets chosen from the cameras (joints sag).
2. Reach: if the block is beyond vertical reach, one forward pulse (duty 20, 0.5 s) is ~2 in on this rug.
3. Aim: centre the block in the claw camera at the low pose, then remember the jaws land ~2 in SHORT
   (toward the robot): raise, drive one pulse forward, descend again.
4. Final approach: block at the bottom centre of the claw view between the jaw tips; lower with the elbow
   until a step shows ~0% change (pads down); unload 100 us if it stalled hard.
5. Grip: close 2000 -> 1800 -> 1600 -> 1400; a closing step with 0% change = jaws stopped on the block.
6. Verify hold: two lifts; the block's bbox in the claw view stays fixed while the background changes.
7. Find the target: lift high, reverse ~4-5 in (two pulses) so the C270 can see; yaw toward the target.
8. Place: lower in 50-100 us shoulder steps with small wrist corrections; in the C270 the held block
   should sit over the target at the same apparent width; release only when the held block's width in
   the claw view JUMPS (it stopped moving = resting on the target).
9. Release 1400 -> 1600 -> 1800, lift, confirm in the C270 that the block stayed and the jaws are empty.
Constants: yaw +50 us ~ 65 px right (high, full-res), gain about doubles low; shoulder +100 lowers ~2 cm
and pulls the gripper back; elbow reversal backlash ~100 us; shoulder ceiling 2500; runner gripper floor 1400.
Hazards seen: ModemManager-suspected serial ownership refusals (4x) -> reconnect after ~5 s; claw camera
brief stalls -> retry snapshot; C270 warm-up black frames; red pressed on the C270 lens blinds it.

## TOWER ATTEMPT (Anthony 20:5x: 'im so ready to see this! go fffor it'; +4 fwd, +4 back pulses). Rule: follow the recipe, no hints.
- T: G64 lower, G65 yaw 1375 centred blue, G66 low pose: blue at TOP of claw view, C270 jaws in front straddling blue = 2 in short (recipe step 3 case).
- T: CAUGHT: jaws had stayed at 1800 since the stack release; a 45-deg block's diagonal (~4.2 cm) may not clear 1800. Recipe addition: always open to 2000 before the final approach.
- T: TOWER GRAB 1 MISSED (20:58-20:59): blue rotated ~45 deg to the jaws; closed 2000->1400 (no clean stall,
  last step 7.3%); double lift: blue px 89269 -> 24161 -> 575, jaws empty, blue back on floor.
- RULE (also explains attempt 3 on green at ~30 deg): the block's faces must be roughly parallel to the jaws.
  A corner-first close pivots the block out. Square it first (or yaw so the jaws meet two faces).
- T: STOPPED at 21:0x (battery 7439 mV, ~140 above runner floor). Blue still rotated, now close to the C270.
  TOWER NOT BUILT. Recipe verdict without hints: works for blocks square to the jaws (2/2), fails on rotated
  blocks (0/2: green ~30 deg, blue ~45 deg). Missing step for the skill: square the block first (idea, untested:
  nudge a corner with a closed jaw and yaw). End pose 1=1400 3=1132 4=2353 5=1846 6=1345, guard disarmed.
- T: Anthony 21:0x: 'its ok to stop here if you chose to'. Chose to continue: try squaring blue with a closed-jaw nudge.
- NUDGE TRY (21:05-21:06): closed jaws as a finger; G76 yaw 1375 + elbow 2453 put it on the rug but BEHIND blue
  (elbow+ pulls back); G77 wrist 1232 (out) then yaw sweep 1375->1295: blue's edge angle unchanged (-27 -> -29 deg),
  it only slid in the view = no contact. Next time: the finger must sit ~1 cm farther out (more wrist, or start the
  sweep from the low pose before the elbow step), and confirm contact in the C270 before sweeping.
- END 21:07: battery ~7390 mV, ~80 above the runner floor, too little for square + grab + stack. Parked with
  G78 (pads lifted off the rug). Tower: not built. First stack still standing.
