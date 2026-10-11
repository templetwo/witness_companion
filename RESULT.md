# IDENTITY

Codex, codex-astra-studio. Model: gpt-6-astra. Date: 2026-10-11.
Assigned lane: Witness Companion MCP tool-server scaffold.
Worktree: `/Users/tony_studio/Projects/witness-companion-prep/wc-mcp`.
Branch: `prep/mcp-tool-server`.

# WHAT WAS BUILT

Implementation HEAD SHA: `49625a916ddaedc93125f2d1d63f133c07a786ac`.
A following receipt-only commit contains this report and the logs. The final
branch HEAD is available with `git rev-parse HEAD`; this file does not embed
its own commit hash.

Files added:

- `witness_mcp/__init__.py`: import-safe package.
- `witness_mcp/protocol.py`: JSON-RPC stdio server and framing.
- `witness_mcp/tools.py`: exactly five schemas, validation and trace envelopes.
- `witness_mcp/services.py`: abstract service interface, fake image and speech services, local pending notes and disarmed robot guard.
- `witness_mcp/__main__.py`: fake-only CLI; real mode exits nonzero.
- `witness_mcp/README.md`: launch examples, service contract and integration limits.
- `tests/test_witness_mcp.py`: unittest coverage in-process and over subprocess stdio.
- `receipts/.gitignore`: retain receipt logs despite the repository-wide log ignore rule.
- `receipts/README.md`: exact commands, versions, results and evidence limits.
- `receipts/unittest.log`: complete final passing test output.
- `receipts/unittest-initial.log`: complete initial test failure output.
- `receipts/py310_syntax.log`: requested grammar check and the additional 3.12-hosted check.
- `RESULT.md`: this handoff.

Final tests: 47 passed, 0 failed, 0 errors, 0 skipped, using
`/opt/homebrew/bin/python3.12` (Python 3.12.12). Python 3.10 grammar checks passed.
No network services or devices were accessed. No packages were installed. No
push was performed. All authored files are in this worktree; Git maintains the
worktree and branch metadata in the shared repository as expected.

# DESIGN NOTES

The supported subset is JSON-RPC 2.0 with initialize, tools/list and tools/call,
using newline-delimited UTF-8 JSON or optional Content-Length byte framing.
The advertised protocol version is 2024-11-05. Initialization is required before
tool calls; the initialized notification is accepted without a strict handshake.
Notifications never execute tools. Batches, resources, prompts, ping,
cancellation, progress, pagination and server requests are absent. The protocol
module documents the full subset and error codes.

Each tool result has a UUID trace_id, UTC issued_at and backend provenance.
The result is JSON inside a text content block, with isError for operational
failures. Invalid params and unknown tools return JSON-RPC errors. Fake results
use backend kind none, model null and unknown compute location. No fake result
claims playback, inference or physical execution.

The fake uses a stored image path and original timestamp. Stale, missing and
future-dated images fail explicitly. Notes append only to a host-selected local
JSON array with atomic replacement. A tool cannot select another note path.
This queue has one writer per file and is not a Stack write.

The robot guard starts disarmed. Stop is accepted even for repeated or expired
requests and disarms the fake. Motion requests require a future expiry, a fresh
nonce and bounded speed and duration. Unknown actions are refused. No tool can
arm. Replay history is in memory for the life of the fake service instance.
The real services being built on prep/companion-coordinator will implement the
abstract interface; no sibling worktree was accessed or changed.

Before wiring OpenClaw, verify its installed version, MCP transport and protocol
negotiation, initialized handshake, schema support, text-result parsing,
error handling, launch environment and tool-policy configuration. Verify whether
it requires any omitted MCP features. Then wire and test the real adapter.
Nothing here proves OpenClaw compatibility or physical behaviour.

# FAILURES

No unresolved test failures. The first full run had 46 passes and one failing
file-access audit assertion. That test treated opening its own temporary root
directory as outside the permitted directory. A focused reproduction identified
that mistake; the test was corrected and all 47 tests passed. The initial full
failure output is preserved in receipts/unittest-initial.log.

No Python 3.10 runtime or real OpenClaw, backend or hardware validation was done.
These remain integration work, not inferred successes.

# MIDDLES

The existing Event schema has a naive timestamp and no backend provenance, so the tool envelope supplies its own UTC timestamp and UUID without modifying that schema.
The default python3 is Python 3.9.6, so the requested syntax command was supplemented with Python 3.12.12 using feature_version=(3,10).
T2Helix tools were not consulted under this file-only lane; a bounded read-only disk query found no insights in a 20261011-named domain, and existing watch logs were dated September.
No current mesh watch delivery or stored session identity was established, and no new mesh writes or watchers were started.
The fake guard has process-local nonce memory and the note queue supports one writer; both limits must be addressed by real integration requirements.
