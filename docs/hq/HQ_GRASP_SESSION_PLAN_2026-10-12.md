# Attended arm session plan: HQ grasp and lift, 2026-10-12 evening

Filled on docs/hq/ARM_SESSION_PLAN_TEMPLATE.md. Runs through Codex's commissioning harness
(runtime override with joint limits, operator arm signal, signed bounded joint actions, readback,
both cameras per stage, override removed and guard locked after every batch). Every pulse below
was already exercised by Codex today; HQ reuses the exercised ranges and adds no new ones.

## Header
- Seat running it: HQ (claude-fable-5-1) from the Studio over ssh to the Jetson.
- Approved by Anthony: "ok let me see you grab the block and lift it" (terminal, 18:4x EDT). Each batch waits for his "go".
- Battery at start: read in batch 1 preflight; must be above 7300 mV, floor 7000 mV.
- Guard release: /opt/temple-witness/releases/20261012-capabilities-04 (loud-worker fix not deployed; a startup EOF restarts the guard disarmed and the batch aborts before motion).
- Arm start pose (readback 18:46): 1=1800 open, 3=782, 4=1803, 5=2016, 6=1520. NOT HOME, jaws about 2 cm above the green block, block between the pads in the fork view.
- External camera: C270 on Jetson /dev/video0, whole arm, fork and stack visible (frame at 18:46).

## Goal
The green block leaves the red/blue supports held in the jaws and stays there for a shoulder lift of 200 us, seen on the side camera. Anything less is reported as not a pickup.

## Preconditions
- Anthony at the switch, within reach, wheels never move (wheel ceiling 0 in every batch).
- Green block square to the jaws (fork view 18:46 shows it square; confirm by eye).
- Stop path: every batch ends with a signed stop and a locked status read; the harness does that.
- No replay: each batch writes a receipt named for it and refuses to run if that receipt exists.

## Batches
| # | Joints and pulses | Duration | Expected on camera | Stop rule |
|---|---|---|---|---|
| 1 seat | 5: 2016 -> 2116 -> 2216 (jaws open 1800) | 2 x 0.6 s | pads come down beside the green block's two faces, block not pushed | block moves, pads land on top of block, any joint other than 5 moves |
| 2 close | 1: 1800 -> 1650 -> 1500 | 2 x 0.6 s | jaws close on opposing faces, block does not rotate | block rotates or pops out: stop, open, reseat |
| 3 lift | 5: 2216 -> 2116 -> 2016 (jaws 1500) | 2 x 0.6 s | block rises with the jaws, daylight under it | jaws rise and block stays: slip, report; nothing rises: shoulder not responding under load, stop |
| 4 set down | 5: 2016 -> 2116 -> 2216, then 1: 1500 -> 1800, then 5: 2216 -> 2016 | 0.6 s each | block back on the supports, jaws open and clear | any strain or buzz |

## End state
Jaws open, clear above the supports, guard locked and disarmed, override removed, receipts and
images under jetson:~/witness_scratch/hq-grasp-20261012/, two sentences on what the frames show.
