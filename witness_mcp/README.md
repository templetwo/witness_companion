Codex, gpt-6-astra, Jetson seat, 2026-10-11 23:32:40 EDT

Current integration checkpoint: see ../PROJECT_STATE.md. --pi live adapter is WIP and has not been deployed. The older fake-only description below is retained as scaffold provenance from prep/mcp-tool-server.

# Witness MCP scaffold

This standard-library package exposes five Witness tools to a future harness
adapter. It is a fake, hardware-free scaffold. The installed OpenClaw version
has not been verified against it. Nothing here proves OpenClaw compatibility or
physical behaviour.

Run from the repository root with Python 3.10 or later:

```sh
python3 -m witness_mcp --fake --notes-file ./pending-notes.json
```

Stdout is reserved for protocol responses. Without `--fake`, the command exits
with status 2 and reports that real services are not wired on stderr. Starting
the fake does not create a notes file. Only `witness_propose_note` creates or
updates it. The file's parent directory must already exist. Use one server
process per notes file. This local queue is not an approved durable-memory write.

To supply a stored sample, add both `--sample-image ./sample.ppm` and
`--capture-time 2026-10-11T21:25:00Z`, using the sample's actual capture time.
The fake never captures, decodes or retimestamps the file. Without a sample,
snapshot returns `no_sample_image`. The default freshness limit is five seconds.

Example newline-delimited request sequence:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"local-harness","version":"0.1"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"witness_status","arguments":{}}}
```

## Tools and result contract

| Tool | Arguments | Service method |
| --- | --- | --- |
| `witness_snapshot` | Optional `max_age_seconds`, finite number 0 to 3600 | `snapshot(max_age_seconds)` |
| `witness_speak` | `text`, 1 to 10000 characters, not whitespace only | `speak(text)` |
| `witness_status` | Empty object | `status()` |
| `witness_propose_note` | `text`, same bounds as speak | `propose_note(text, trace_id, issued_at)` |
| `robot_request_action` | `name` string and `params` object | `request_action(name, params)` |

Each successful protocol call returns `content: [{type: "text", text: ...}]`
and `isError`. Parse the text as JSON to get the service result and `trace_id`,
`issued_at` (UTC ISO 8601), and `backend_provenance`. A service failure is a
traceable result with `isError: true`; malformed arguments or an unknown tool
produce JSON-RPC error code -32602. Unknown methods use -32601. Invalid requests,
parse failures and internal failures use -32600, -32700 and -32603 respectively.
Calls before initialization use the scaffold-specific code -32002.

The fake reports `backend_id: "fake"`, `kind: "none"`, `model: null`, and
`where_computed: "unknown"`. These fields never imply model inference. Speech
is recorded but not played. Every robot verdict includes `executed: false`.

Only `stop` and `move` are recognized by the fake guard. Stop is accepted even
when disarmed, expired or repeated, and sets the fake back to disarmed. Move
requires `params` with exactly `nonce`, `expires_at`, `speed`, and
`duration_seconds`. The nonce must be a nonempty string at most 256 characters;
expiry must be in the future with a timezone; speed is normalized to [-1, 1];
duration must be greater than zero and no more than five seconds. A fresh nonce
is consumed before bounds and armed checks, including when motion is refused.
Nonce history is in memory and does not survive restart. No tool arms the guard.
The host/test code can set `FakeServices.armed` to exercise acceptance logic.
This is not the real controller's independent interlock or emergency stop.

## Integration work remaining

Real services are being built on `prep/companion-coordinator` and will implement
`WitnessServices`. Supply that adapter to `ToolRegistry`, then `Server`. The
adapter owns truthful provenance, fresh capture, actual playback, status and
independent controller decisions. Real device imports belong inside those
adapters, not at package import time. The CLI intentionally has no real wiring.

Before connecting the installed OpenClaw, verify its MCP transport, supported
protocol version and initialize handshake, tool schema support, text-result
parsing, error handling, subprocess launch environment and tool permissions.
Verify whether it needs features omitted here, such as ping, resources,
prompts, cancellation, batching or pagination. The optional Content-Length
support is a convenience and is not evidence that OpenClaw expects it.

The protocol module documents framing, size limits and notifications in detail.
Unknown notifications are ignored; notification-shaped tool calls never execute.
Only a complete request with an ID can invoke a tool. No network is used.
