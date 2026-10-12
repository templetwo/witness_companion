Codex, gpt-6-astra, Jetson seat, 2026-10-11 22:56:29 EDT

# Pi guard validation

Worktree: /home/tony/witness_pi_guard, branch prep/pi-guard-service, base c12f7dc.
Python runtime: 3.10.12, Linux aarch64, Jetson. No packages installed.

Exact command (from this worktree):

    /usr/bin/python3 -m unittest discover -s tests -t . -v

Result: 64 tests passed, 0 failures/errors/skips. The baseline has 42 tests;
22 additions cover guarded TCP/service/driver behavior and malformed inputs.
Full command and output: pi-guard-unittest.log. Runtime was about 5.5 seconds.
The deliberately crashing fake driver may emit a child-process diagnostic;
the parent detects the crash and latches the fault as tested.

Syntax command is recorded in pi-guard-syntax.log: ast.parse for every Python
file under companion and tests with feature_version=(3,10) and (3,11).
30 files parsed for each grammar. Only Python 3.10 runtime is available here.
Python 3.11 compatibility is a source/stdlib design target, not a runtime result.

New tests use only FakeBoard or a recording SDK stub. Real loopback TCP sockets
and spawned processes test the transport/watchdog. In-memory keys are generated
randomly for each test; no credential is saved. Tests contact no Pi, vendor SDK,
serial port, camera, model, speaker or Stack. The isolated import test still
passes for every companion submodule.

Coverage includes disarmed rejection, signed actions, duplicates, expiry,
unsigned requests, stop priority, forged-stop sequence protection, restart
replay state, nonfinite/oversized numeric timestamps, empty nonce, duplicate JSON
keys, request/response reflection, wrong session, signed payload tampering,
authenticated TCP roundtrip, replayed TCP heartbeat, unauthenticated stop,
partial-input expiry, oversized frames, unauthorized heartbeat expiry,
watchdog operation without request dispatch, driver hang, SIGSTOP, process
crash, supervisor SIGKILL, fake provenance and all five tool handlers.

These tests do not establish physical stopping, servo calibration, installed
vendor equivalence, multi-host clock skew, serial exclusivity or MCP wiring.
The camera read and separately authorized live pulse are documented in
/home/tony/WITNESS-CODEX-RESULT.md. Camera bytes remain outside the repository.
