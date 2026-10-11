# receipts: companion coordinator, hardware-free harness

- **Tested sha:** `98012b5512ecc4615548d02d18598ec58dabfac5` on branch `prep/companion-coordinator`
  (worktree `~/Projects/witness-companion-prep/wc-prep`, baseline `origin/master @ 1728b26`).
  The tree was clean (`git status --porcelain` empty) when the tests ran. These receipt files are
  committed in the NEXT commit, so HEAD moves past the tested sha by exactly that commit.
- **Where:** Mac Studio, macOS 27.0, 2026-10-11 ~17:3x EDT (stamps are inside each log).
- **Test command (from the worktree root):**
  `/opt/homebrew/bin/python3.12 -m unittest discover -s tests -t . -v`
  (`-t .` makes the `tests` package importable). Repeated with `/opt/homebrew/bin/python3.11`.
- **Python:** 3.12.12 and 3.11.x (Homebrew). **No 3.10 interpreter exists on the Studio, so 3.10
  runtime is UNPROVEN.** 3.10 coverage is a syntax check only
  (`ast.parse(..., feature_version=(3,10))`, under `-W error`) plus a grep for post-3.10 stdlib APIs
  (`StrEnum`, `datetime.UTC`, `typing.Self`, `tomllib`, `except*`, `TaskGroup`, `kw_only`,
  `slots=True`): no hits in code. Bare `python3` here is 3.14.5 (recorded in `py310_syntax.log`).
- **pyyaml is absent from python3.12 here**, so nothing in the core needs it. 3.11 and 3.14 have it.
- **Files:** `unittest.log` (raw output, both interpreters), `py310_syntax.log`, `demo.log`
  (`run_demo.py` driven from a fixed stdin script, disarmed).

## Results (31 run, 31 pass under 3.12 and under 3.11; 1 group BLOCKED)

| Required check | Test(s) | Result |
|---|---|---|
| request to response round trip | `test_coordinator.RoundTrip.test_voice_request_to_spoken_response`, `test_voice_to_action_round_trip_when_armed` | PASS |
| image interpret, fresh image | `Images.test_fresh_image_is_interpreted` | PASS |
| image interpret, stale image refused | `Images.test_stale_image_is_refused` | PASS |
| missing backend gives "unavailable", never raises | `MissingBackends.test_none_backends_give_unavailable`, `test_null_backends_give_unavailable`, `test_missing_speaker_still_returns_text` | PASS |
| malformed command rejected | `test_guard.Malformed.test_malformed_rejected` (8 shapes) | PASS |
| unauthorized command rejected | `Unauthorized.test_wrong_key_unknown_issuer_tampered_unsigned` | PASS |
| expired command rejected | `Expiry.test_expired_and_bad_windows` (also ttl too long, issued in future, rejected seq not consumed) | PASS |
| duplicate command rejected | `Duplicate.test_duplicate_seq_and_nonce` | PASS |
| connection loss | `ConnectionLoss.test_link_loss_is_unknown_and_not_retried`, `HonestOutcome.test_link_lost_stays_on_display` | PASS |
| restart without replay | `test_restart.*` (4 tests incl. torn journal line) | PASS |
| disarmed start rejects motion, accepts stop | `Arming.test_disarmed_start_rejects_motion_accepts_stop`, `test_stop_always_accepted_and_disarms`, `test_arm_is_not_a_command_and_never_persisted` | PASS |
| import of `companion` touches no device | `test_no_device_import` (fresh `-I` subprocess, imports every submodule; forbids cv2, sounddevice, serial, pyaudio, faster_whisper, whisper, piper, requests, numpy, luma, smbus, spidev, RPi, gpiozero, yaml, witness_brain) | PASS |
| extra: brain action outside whitelist dropped | `RoundTrip.test_unknown_brain_action_is_dropped` | PASS |
| extra: rejected action not spoken as done | `HonestOutcome.test_rejected_action_is_not_spoken_as_done` | PASS |
| extra: guard imports no compute/adapter/coordinator code | `GuardIndependentOfCompute` | PASS |
| extra: approved note survives restart, unapproved never returned | `Notes.test_approved_note_survives_restart` | PASS |
| addition 3: five tool functions return JSON-serialisable dicts and never raise (fakes wired, bad input, missing backends, unconfigured) | `test_tools.ToolSurface.test_all_five_return_json_dicts_and_never_raise`, `test_never_raise_on_bad_input_or_missing_backends` | PASS |
| addition 3: `robot_request_action` disarmed refuses motion, accepts stop | `test_tools.ToolSurface.test_disarmed_refuses_motion_accepts_stop` | PASS |
| addition 3: status reports wiring and armed state; stale snapshot flagged; proposed note not retrievable | `test_status_reports_wiring_and_armed`, `test_snapshot_stale_and_proposed_note_not_retrievable` | PASS |
| **revision 2: per-task routing, fallback order, provenance on every result, capability mismatch refused before any call, missing credential env var reports "not configured" without its value, provenance in the log line** | `tests/test_compute.py` | **BLOCKED**: the command that writes this file was soft-denied by the t2helix compass (rule `credential-paste`, pending confirmation #2, still pending and unapproved at two checks, the second after the 10-minute window had passed). The file was not written by another route. `companion/compute.py` and `config/compute.yaml` exist and are import-checked by the no-device test, but their behavior is **untested**. |

## What these tests do NOT prove

**Fake-controller tests prove nothing about physical motion or physical stopping.** An "applied"
result means only that the software guard accepted a command and the fake recorded what it would
have done. Nothing here exercises the vendor robot stack, the Pi, a motor, a controller timeout, a
physical cutoff, a microphone, a camera, a speaker, or the OLED. Clock behavior across two hosts
(Jetson issuing, Pi checking expiry) is untested; tests use one injected clock.
