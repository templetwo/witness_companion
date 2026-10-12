# Friction points, Witness Companion, as of 2026-10-12 19:1x EDT

HQ, on Anthony's "push all updates to the repo and all friction points". Everything here was hit
by HQ or Codex on 2026-10-11/12 and is measured, not guessed. Ordered by how much it cost.

## Robot and arm
1. **The gripper at 1800 is narrower than a block.** Every grasp before 19:00 pinched two corners
   of a diamond-wise block, which rotates and drops. Vendor code opens to 2000
   (`functions/color_sorting.py:170`). Fix: open to 2000 before any approach; measure gap versus
   pulse in the calibration session (`CALIBRATION_SESSION.md`, batch 5).
2. **No joint encoders.** Every readback is a commanded setpoint. Physical position is only known
   from the two cameras. Modelled lifts diverged from observed motion (Codex) and the elbow sign
   was the opposite of HQ's guess. Fix: the calibration session, then correct `companion/arm_lift.py`
   with measured differences, not a new model.
3. **Battery drains about 15 mV per batch and the floor is 7000 mV.** Each batch restarts the
   guard twice (override in, override out) and holds servos under load; HQ's nine batches took the
   pack from 7327 to 6984. A grasp needs a fresh charge and a plan of at most six batches.
4. **Each batch is slow and restarts the guard.** The commissioning path installs a systemd
   override with per-batch joint limits, restarts the guard, arms by SIGUSR1, acts, removes the
   override, restarts again. About 20 s and two restarts per batch. Each restart is a chance for
   the startup EOFError. Fix candidate: a single attended session with limits set once and
   batches issued inside it; needs design, not a shortcut.
5. **Startup EOFError** (4 times on 10-12): worker dies under 1.2 s with no output. Fix on the
   branch (8a657b4, worker fails loud), NOT deployed to the Pi; Codex deploys at Anthony's go.
6. **Cameras.** The fork camera is wide-angle, blurry at 10 cm, lit blue by the sonar LEDs, and
   its axes are rotated relative to the jaws. The C270 side camera is oblique, so its x axis
   mixes reach and lateral. Fix: tape a ruler in the arm plane, put the C270 square to the arm,
   and run the calibration session.
7. **Sonar cannot guide the jaws.** Chassis-mounted, facing forward; the arm's target is beside
   and below it. Reading it costs battery and tells nothing about the gripper.
8. **Shoulder steps of 100 us move the pads along an arc, not straight down**, and lowering the
   shoulder also swings the pads toward the robot. Reach must be corrected with the elbow, whose
   sign is now known (decreasing pulse swings down and back).
9. **PWM travel cannot be cancelled mid-command.** Stop halts wheels and bus joints; a 0.6 s PWM
   move completes. Keep stages short and bounded (the harness enforces at most 1 s).
10. **Whole-Pi or kernel freeze still has no motor cutoff** (`MOTOR_CUTOFF_DECISION.md`, Anthony's line).

## Seats, repos, access
11. **Codex cannot push to GitHub** (no HTTPS auth in its terminal). HQ carries pushes by git
    bundle; three today. Fix at Anthony's word: a deploy key or `gh auth` on the Jetson.
12. **Ring-2 Stack proposals need a manual drain.** Codex's two checkpoints sat pending until HQ
    drained them on Anthony's word. By design (landing-path law), but the drain is a human step.
13. **Three worktrees of one branch** (Studio, Jetson, origin) diverge when Codex commits while HQ
    works; 684e869 needed a merge. Rule: HQ fetches the Jetson by bundle before every commit.
14. **Pi address is a DHCP lease on the Jetson's shared link.** Reservation staged at
    `jetson:~/robot-link.conf`; needs Anthony's sudo.
15. **Jetson sudo needs a password**, so NetworkManager, apt and shutdown are Anthony's hands.
16. **Compass WITNESS blocks `rm -rf` and multi-path `rm -f` for HQ**; scratch clones are left for
    hands: `jetson:~/witness_scratch/hq-eof-check.mORk`, `pi:/tmp/hq-eof-check`.
17. **zsh expands a word starting with `=` as a command name**, so `echo ===` breaks a remote
    command chain; and `--include=*.py` globs locally. Quote remote commands with double quotes
    or send a script.
18. **Studio has no PIL**; crop frames with `sips`.
19. **Runners import from the guard tree**; run them with `PYTHONPATH=/home/tony/witness_pi_guard`
    from that directory, or the first import fails.

## Open at Anthony's word
Motor cutoff line; sudo install of the address reservation; the gripper power-off check; approval
of the calibration session plan; GitHub auth for the Jetson; deploy of 8a657b4 to the Pi.
