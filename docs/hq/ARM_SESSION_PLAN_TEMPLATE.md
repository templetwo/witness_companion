# Attended arm session plan (template)

HQ, 2026-10-12. One page per powered session. Anthony approves the written plan
BEFORE power-on. During the session he is at the switch and is not asked
questions that the plan should have answered. Copy this file, fill it in, keep
it beside the session's receipts.

## Header
- Session name / date / seat running it:
- Approved by Anthony (how and when, verbatim word):
- Battery at start (mV, must be above 7300 before any arm move; stop at 7000):
- Guard release deployed (`/opt/temple-witness/current` target):
- Arm start pose, commanded pulses 1/3/4/5/6 (from readback, never assumed home):
- External camera placed and framed (whole arm, fork and floor visible: yes/no):

## Goal, one sentence
What will be true at the end that is not true now.

## Preconditions (all must hold or the session does not start)
- Robot on the bench or floor with the manoeuvre space clear; wheels will not move (wheel ceiling 0).
- Anthony at the switch, within reach, phone/camera not needed for control.
- Stop path verified first: one signed `stop` round trip before any other action.
- No scratch runner replays; every batch is written here before it runs.

## Batches (each one bounded; observe and record between batches)
| # | Joints and pulses (from -> to) | Duration | Expected on camera | Stop rule |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |

Rules: at most 200 us per joint per batch, one joint at a time unless the batch
says otherwise and why; a batch that does not look like its "expected" column
ends the session's motion, not just the batch.

## End state
- Commanded pulses at end:
- Guard: locked, disarmed, wheel 0, overrides removed (verified by status read):
- Receipt file(s):
- What was learned, in two sentences, written before the robot is powered off.
