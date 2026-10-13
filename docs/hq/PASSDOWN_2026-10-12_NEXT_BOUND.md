# Passdown: the next bound for the Jetson agent

Written 2026-10-12 20:2x EDT by HQ (Claude Fable 5.1, Mac Studio) on Anthony's word
("write the next passdown containing the next bound for jetson agent"). Baseline is
`prep/pi-guard-service` at 04e6de4 on GitHub, the Studio worktree and `jetson:~/witness_pi_guard`.
Read `PASSDOWN.md` (Codex, 684e869) and `jetson:~/WITNESS-PASSDOWN-CODEX.md` sections 10 to 12
first; this file does not repeat them.

## 0. What is true tonight

- Robot: jaws open at 2000, arm NOT home (1=2000, 3=782, 4=1803, 5=2116, 6=1420), guard
  disarmed, no override, battery tripped the 7000 floor at 6984 mV. **Nothing moves until the
  pack is charged and Anthony is at the switch.** When he powers off, a hand goes under the gripper.
- Fix 8a657b4 (loud worker on startup failure) is on the branch and NOT deployed to the Pi. Live
  release is still `20261012-capabilities-04`.
- The battery floor is enforced only by the attended runners (`scripts/attended/hq_grasp_once.py`,
  `companion/arm_lift.py`). The Pi guard (`companion/pi/service.py`) applies a motion command
  without reading the battery. That is the gap this bound closes first.
- The companion tool path (`companion/tools.py:robot_request_action`) refuses params on every
  command except say and display, so `drive` and `joints` are reachable only through
  `witness_mcp`. `companion/controls.py:validate` already knows the shapes.
- Speech: the MCP schema (`witness_mcp/tools.py` TEXT) allows 10000 characters; the local adapter
  (`witness_mcp/live.py:79`) and `companion/schemas.py:MAX_TEXT_LEN` stop at 200. A long text is
  accepted by the schema and refused by the adapter.
- Suite: 158 tests, green on Studio 3.14 (1 skip off Linux), Jetson 3.10, Pi 3.11. Unittest, not
  pytest. Run from the repo root with `python3 -m unittest discover -s tests -v`.

## 1. The bound (software only, off-robot, no deploy, no motion)

Three changes, each its own commit, each with tests that fail on 04e6de4 and pass after.
Do them in this order. Stop after each and leave the diff for HQ review; do not fold them.

### 1a. Battery floor inside the Pi guard service

Where: `companion/pi/service.py` `Service.action`, between `guard.check` and `supervisor.call('action')`.

- For any command in `MOTION_COMMANDS` (look_*, drive, joints): read the newest `battery_mv`
  the supervisor already holds from `status()` telemetry (`companion/pi/hardware.py:status`).
  Do not open a new SDK read on the action path; the response budget is 0.5 s.
- Refuse with `{'ok': False, 'verdict': 'rejected', 'executed': False,
  'reason': 'battery_below_floor', 'battery_mv': <mv>, 'floor_mv': 7000}` when the reading is
  below the floor. Refuse with `reason: 'battery_unknown'` when there is no reading or the
  reading is older than 5 s (use the monotonic stamp the telemetry tuple already carries).
  Fail closed: unknown is a refusal, never a pass.
- `stop`, `say`, `display` and the output commands are never gated.
- Floor comes from a new `--battery-floor-mv` argument on `companion.pi.service` (default 7000),
  written into the status payload as `battery_floor_mv` so a reader can see which floor is live.
- Tests in `tests/test_pi_service.py` against the fake controller: below floor refused and the
  supervisor never called; at the floor applied; no reading refused; stale reading refused;
  `stop` applied with the battery at 0. Also one test that a refused action does not disarm and
  does not trip the supervisor (it is a refusal, not a board fault).
- Do not change `deploy/temple-witness-guard.service`. The default carries the floor.

### 1b. Structured `drive` and `joints` on the companion tool path

Where: `companion/tools.py:robot_request_action`.

- For `name in EXTRA_MOTION`, run `companion.controls.validate(name, params)`; on `ValueError`
  return `_fail(str(e), verdict='refused')`; on success pass the validated params as `args` to
  `_services.coordinator.request_action(name, args)` (that signature already takes args).
- Keep every existing refusal: unknown name, non-dict params, say/display shape, text length,
  and "takes no params" for the remaining commands.
- Tests in `tests/test_tools.py`: valid drive reaches the fake coordinator with the same args;
  joints with seven joints refused; drive with an unknown direction refused; drive with no
  params refused; look_left with params still refused. The coordinator's guard still requires
  arm, so no test may assert physical motion; the existing note line in the return stays.

### 1c. Speech length: make the MCP schema tell the truth

Where: `witness_mcp/tools.py`.

- Give `witness_speak` its own text schema with `maxLength` 200 instead of the shared TEXT
  (10000 stays for `witness_propose_note`, which the Pi bounds at 2000; align that one to 2000
  too). A long text is then refused at the schema with the standard validation error instead of
  travelling to the adapter.
- No chunking in this bound. If Anthony wants long speech, that is a separate decision (a chunker
  issues several signed say actions and needs its own stop semantics).
- Tests in `tests/test_witness_mcp.py`: 201 characters refused at the schema; 200 accepted;
  a 2001-character note refused.

## 2. Carry-forward rules (unchanged, do not relax)

- No Stack writes from the Jetson seat. Findings go in `PASSDOWN.md` and in receipts; HQ lands them.
- No deploy, no restart of the Pi guard, no override file, no `SIGUSR1`, no motion. This bound
  is complete when the tests are green on the Jetson (3.10) and on a Pi scratch clone (3.11,
  `pi:/tmp/hq-eof-check` exists and may be reused with `git fetch` from the bundle).
- Exceptions raised on the Pi travel to the journal: no secret in any exception message.
- Never read or commit anything from the vendor image; the SDK path stays a runtime argument.
- Push is carried by HQ: leave a bundle at `jetson:~/witness_scratch/bundles/<date>-next-bound.bundle`
  built with `git bundle create <file> prep/pi-guard-service ^04e6de4` and say so in `PASSDOWN.md`.
- Receipts: one JSON per commit under `receipts/next-bound-20261013/` with the test command, the
  interpreter, the count, and the sha256 of the files changed. No estimated clocks; stamp from `date`.

## 3. Acceptance

- Three commits on `prep/pi-guard-service`, each message naming its test count before and after.
- HQ review before any of it reaches the Pi. HQ will read the diff for: a motion command that
  can pass with battery unknown; any path that skips `guard.check`; any new argument shape not
  in `controls.validate`.
- After HQ review and on Anthony's go in the Codex terminal: one release cut
  (`/opt/temple-witness/releases/20261013-guard-floor-01`, manifest hashes, flip `current`,
  restart, status disarmed with `battery_floor_mv: 7000` in the payload) carrying 8a657b4 and
  1a together. Rollback is the symlink back to `20261012-capabilities-04`.

## 4. Not in this bound (waits on Anthony)

- Charge the pack. The next grasp attempt (about four batches from the end pose, plan in
  `WITNESS-PASSDOWN-CODEX.md` section 12) is HQ's or Codex's lane only with him at the switch.
- His one line on `docs/hq/MOTOR_CUTOFF_DECISION.md`.
- `sudo install -m 644 ~/robot-link.conf /etc/NetworkManager/dnsmasq-shared.d/robot-link.conf`
  then `nmcli con down robot-link && nmcli con up robot-link` on the Jetson.
- The power-off gripper check (`docs/hq/GRIPPER_MECHANICAL_CHECK.md`) and approval of
  `docs/hq/CALIBRATION_SESSION.md`.
- GitHub auth for the Jetson, so bundles stop being the push path.
- Long speech (chunking) and whether motion stays MCP-only or the companion path is the front door.

## 5. The outside plan Anthony pasted tonight

A plan from another seat proposed the same three software items plus a discharge test
("run the motors until the Pi refuses below 7.0 V"). Do not run that test: it burns a cycle on
a pack already at the floor and the floor it would exercise lives in the runner, not the Pi,
until 1a lands. Its commands (`pytest companion/tests`, `witness_mcp --live --tool`,
`config/*.service`, `dhcpcd.conf`, `X-Timestamp`, port 5000) do not exist in this repo; HQ's
review is in the session record. Take the three items from it, nothing else.
