Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:06:14 EDT

# Pi guard service for review

This branch contains a fake-only TCP entry point, a Jetson client implementing
Controller, and a staged SDK mapping behind a Board interface. It is not
installed on the Pi or registered in OpenClaw. No vendor source is included.
Anthony has selected exclusive Witness serial ownership and will not use the
vendor app; see VENDOR_FINDINGS.md for the decision and pending live handover. The executable CLI only constructs
FakeBoard. There is no --armed or --real flag.

## Wire and tool contract

One client per accepted connection. UTF-8 JSON lines, at most 16 KiB buffered.
The server sends a random session challenge. The client adds its own random
challenge and signs each envelope with HMAC-SHA256. Envelopes carry issuer,
strictly increasing connection sequence, issue time, expiry and request/response
kind. Responses are also signed, matched to the client, session and sequence.
The key is never returned or logged. Authentication does not encrypt content;
this transport is intended for the private robot link or loopback. A future
routed deployment needs transport confidentiality. Clock skew must be measured
before commissioning; 1 second of future skew and a maximum 5-second TTL apply.

The service exposes all five tool names:

| Tool | TCP arguments | Guard / behavior |
| --- | --- | --- |
| witness_snapshot | {} | Authenticated read of Board.snapshot; fake reports camera unavailable |
| witness_speak | {action: signed ActionRequest with command say} | CommandGuard before Board.execute; fake says played=false |
| witness_status | {} | Authenticated read; reports actual driver kind and guard state |
| witness_propose_note | {text: string, max 2000 chars} | Authenticated pending-only local journal; approved=false |
| robot_request_action | {action: signed ActionRequest} | Independent CommandGuard before Board.execute |

These are internal TCP arguments. The existing MCP user-facing schemas have
not changed. A future MCP service adapter must create the signed speech/action
requests with the coordinator's existing issued-sequence journal. This branch
does not pretend that MCP is already wired.

The passdown says "five commands" but the existing action schema has six names:
stop, look_left, look_right, look_center, say and display. The implementation
preserves that set and the five Witness tool names. Read/status/note operations
are not inserted into the robot command allowlist. Every board action, including
say/display, goes through CommandGuard. There is no wheel-drive action, arbitrary
SDK method, RPC passthrough, or remote arming operation.

An exact unauthenticated {"op":"stop"} requests the guard's unconditional stop;
it cannot renew the heartbeat or advance the signed transport/action sequences.
Stop disarms. If a driver is hung, the supervisor kills it and reports an unknown
outcome. Fake success always includes executed=false and simulated=true. An SDK
return only reports sdk_returned=true, executed=null, never proof of motion.

## Driver and link supervision

The Pi uses a separate spawned driver process, so the parent never owns its
serial descriptor. A driver response deadline (default 0.5 seconds) covers a
blocked or suspended worker. A separate watchdog thread kills the driver when
no valid Jetson heartbeat arrives for 2 seconds, including when the reader,
writer or journal work blocks. Partial lines and rejected/replayed messages do
not renew this deadline. Only the authenticated heartbeat operation renews it.
The Jetson client sends one every 0.25 seconds. Driver failure is latched until
a fresh disarmed session; actions are not automatically retried.

On Linux PR_SET_PDEATHSIG kills the child on supervisor death, with a parent-pid
race check. EOF, oversized input, connection loss, failed startup or a stalled
response closes the session and kills its worker. The CLI holds an exclusive
journal lock. Each new guard starts disarmed; persistent action seq/nonce state
survives reconnect/restart. Local test code can arm an in-memory fake guard;
no TCP request can do so.

This is not a hardware watchdog. A frozen kernel, independent serial owner,
wrong SDK mapping or a moving servo is not covered by the prior wheel test.
Killing an RPC client does not close the vendor server's serial descriptor.
SDKBoard.stop requests zero duty on four wheel motors only. Servo stopping and
head calibration remain unverified. SDKBoard does not import or construct the
vendor SDK; a future approved factory must do so only in the isolated worker,
after all other serial owners are dealt with. No factory for that is shipped.

## Local launch and validation

Use Python 3.10 or 3.11 on Linux, standard library only. No install is needed.
The operator must supply an existing private in-memory environment variable
WITNESS_GUARD_KEY_HEX with at least 32 bytes of hex key material. This task does
not provision credentials or persist keys. The CLI reads only the named variable.

    python3 -m companion.pi.service --fake --journal /private/guard.jsonl --notes-journal /private/proposed-notes.jsonl

Default binding is loopback, ephemeral port, loopback peer only. Host/peer are
explicit options for later review; no service was exposed to the Pi tonight.
Use companion.pi.client.PiLink((host, port), issuer, key) in host code, assign
it as coordinator.controller, and close it at shutdown. Keys remain in memory;
action sequencing remains in the coordinator's journal. Do not discard journals
to cure replay errors. No automatic reconnection or action replay is implemented.

    python3 -m unittest discover -s tests -t . -v

Tests use ephemeral loopback sockets, temporary journals, per-test random
in-memory keys and fake/recording devices. They include actual SIGSTOP/SIGKILL
of fake workers and supervisor-death checks. They never import the vendor SDK
or contact the Pi. Syntax is checked with Python 3.10 and 3.11 grammar modes;
only Python 3.10 is installed here, so no Python 3.11 runtime claim is made.

Rollback is removal of this unmerged worktree/branch after review. Nothing in
the live gateway, systemd, NetworkManager, mirror or Pi was installed or changed
by the service build. The separately authorized attended pulse is recorded in
the external result, and is not part of this service or its automated tests.
