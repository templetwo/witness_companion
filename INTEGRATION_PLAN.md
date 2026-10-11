# INTEGRATION_PLAN: Witness Companion coordinator (Phase B)

Written 2026-10-11 by an HQ build lane (Claude Opus) on the Mac Studio. Spec: the ChatGPT
passdown sections 4B, 4C and 5 (revision 1), plus revision 2 "Hybrid Compute" (17:14 EDT).
Host facts come from `../PREP_REPORT.md` (HQ Phase A, measured 17:05 to 17:12 EDT).
Nothing on the Jetson, the Pi, or any live Studio service was touched to produce this.

## 1. Baseline

- **Baseline: `origin/master @ 1728b26`** (2025-11-23, merge of PR #1). Why: it is the newest
  merged state on GitHub, PREP_REPORT found no local-only or uncommitted work on either machine
  (the Studio `~/witness_companion` copy is the initial commit plus exFAT noise), and the only
  unmerged source is one commit (`0802321`, section 3).
- Work lives on branch `prep/companion-coordinator` in the worktree
  `~/Projects/witness-companion-prep/wc-prep`. The branch's upstream was deliberately unset so a
  bare `git push` cannot target `master`. Nothing is pushed.
- No file under `witness_brain/` was modified. The new code is the package `companion/`,
  `config/compute.yaml`, `tests/`, `run_demo.py`, `receipts/`.

## 2. Reuse, wrap, replace (from `witness_brain/`)

| Piece | Decision | Why |
|---|---|---|
| `schemas/event_schema.Event` (id, timestamp, source, content) | **Mirror the shape**, do not import | Import is harmless today but the class uses naive `datetime.utcnow` (deprecated in 3.12). `companion.schemas.Event` keeps the same four keys, timezone-aware, plus an optional `data` dict for provenance. The JSONL log stays compatible with `core.py`'s `witness_events.log` readers. |
| `speech/stt.py` `STT` (faster-whisper) | **Wrap later, lazily**, inside a real adapter | Imports `faster_whisper` and creates `~/.cache/witness_brain_models` at **import** time. faster-whisper is **not installed** on the Jetson. The Jetson has `~/whisper.cpp` built and `openai-whisper` installed; the real STT adapter should target one of those first (staged). |
| `speech/tts.py` `TTS` (piper + sounddevice) | **Wrap later, lazily**, or replace with a piper CLI adapter | Imports `sounddevice` and `piper` at import, calls `logging.basicConfig` at import, and **downloads the voice from HuggingFace on construction**. The Jetson already has piper 1.3.0 and `en_US-lessac-medium` in `~/piper-voices/`, so the adapter should point at the local voice and never download. |
| `models/brain_client.BrainClient` | **Wrap, do not pass through its fallback** | Useful Ollama call and prompt formatting. But on any Ollama error it returns a canned "I heard you" with `action="log_mood_observation"`, i.e. it fails open with an invented action. The wrapper must raise `BackendUnavailable` instead so the router can fall back and the user hears an honest "unavailable". It also tries `import mlx_lm` at import (absent on the Jetson; harmless but noisy). |
| `vision/camera.Camera` | **Replace** with a snapshot adapter | `cv2.VideoCapture(index)` only, opens the device lazily but `import cv2` is at module top. The companion wants on-request, timestamped single snapshots (USB index or the Wyze RTSP source named in the 2026-09-05 Stack entries), with an age check. |
| `vision/analyzer.VisionAnalyzer` | **Wrap** as a vision backend | Ollama `/api/chat` with an image. Its default model `llama3.2-vision` and `audio.yaml`'s `llava:7b` are **not** in the Jetson's measured Ollama list (`gemma3:4b` is). |
| `cns/core.WitnessCNS` | **Do not reuse as the loop** | Continuous mic capture with wake phrases, and it appends every transcript to a JSONL file (an archive in text form). The passdown proposes push-to-talk and no recording archive. The new `Coordinator` is push-to-talk, redacts transcript text from its log by default (`log_content=False`), and is driven one request at a time. |
| `schemas/tools.py` | Ignore | Empty placeholder. |
| `config/*.yaml` | Leave in place | `brain.yaml`'s system prompt is lineage material; any change is Anthony's. |

New, with no predecessor in the repo: the robot-side `CommandGuard`, the restart-safe `Journal`,
`DisplayState`, the compute `Router` with provenance, and every fake adapter.

## 3. Verdict on `0802321` (`witness_companion.py` on `witness_brain_v1_architecture`)

**Leave it. Do not merge, do not cherry-pick.** Read in full (683 lines). Reasons:
- Imports `cv2`, `sounddevice`, `chromadb`, `librosa`, `faster_whisper` at module top. Importing it
  opens nothing yet, but it fails outright on the Jetson (faster-whisper absent; chromadb and
  librosa are **unverified**, not in PREP_REPORT's measured list).
- Its behavior is the opposite of the proposed milestone: continuous listening with no wake word,
  a proactive loop that captures a camera frame and speaks every 2 to 3 minutes, and a chromadb
  store that persists every exchange (`memory.add`). That is continuous capture plus a
  conversation archive, which conflicts with push-to-talk, on-request images, and no recording
  archive.
- Shells out to `piper`/`aplay`/`afplay` via subprocess with temp files (fine in itself, but
  untested and duplicative of the Jetson's existing `~/voice_output.py`).
- Its `COMPANION_SOUL` prompt (identity and embodiment language) is lineage material. Whether any of
  it carries into the companion is Anthony's call, not this lane's.
- Ideas worth keeping as references only: 800 ms pause turn detection, and the
  "acknowledge, then look" pattern for image requests.

## 4. Dependencies

### (i) Already on the Jetson (PREP_REPORT, measured)
- Python 3.10.12 system interpreter, pyyaml 6.0.3. **That is all the core needs**: `companion/`
  is stdlib-only, with pyyaml imported lazily only to read `config/compute.yaml`.
- Tests use `unittest` (stdlib). pytest is not needed and not installed.
- Available for later real adapters with no install: Ollama on 127.0.0.1:11434 with `gemma3:4b`
  and others, piper 1.3.0 plus `en_US-lessac-medium`, `openai-whisper`, `~/whisper.cpp`
  (`whisper-cli`), sounddevice 0.5.3, PyAudio 0.2.14, opencv-python 4.12, numpy 1.26.4,
  `/dev/i2c-*`, `/dev/spidev*`, `/dev/gpiochip*`, user `tony` in `i2c gpio audio video dialout`.

### (ii) Would need a host change: STAGED PROPOSALS, NOT INSTALLED
Each needs Anthony's go (privileged host change) and should go into a project venv
(`python3 -m venv --system-site-packages ~/witness_companion/.venv`), never into system Python.
1. `faster-whisper` (only if whisper.cpp / openai-whisper prove insufficient; pulls ctranslate2,
   whose aarch64 CUDA build is not guaranteed for JetPack 6.2.1: measure first).
2. OLED driver (`luma.oled` plus `smbus2`, or the vendor's library) for the Inland 1.3" OLED.
   Which side drives it (Pi or Jetson) is open; section 5 suggests the Pi.
3. `pyserial` if the Jetson to Pi link is serial; not needed if it is a LAN socket.
4. Any Studio-side authenticated proxy for `studio_local` (section 6): a Studio change, separate gate.
5. Optional: `chromadb`/`librosa` only if a later design brings back vector memory or tone sensing.

## 5. Pi / Jetson split (passdown section 5)

| Side | Holds |
|---|---|
| **Raspberry Pi 5 (on the MasterPi chassis)** | The vendor robot-control stack, unchanged. `companion.guard.CommandGuard` as the last software gate before the vendor API: closed command set, HMAC per issuer, seq/nonce replay defence, expiry, starts DISARMED, stop always accepted. Its own last-applied-seq journal. Later: robot-mounted peripherals (camera on the arm, maybe OLED and speaker). |
| **Jetson Orin Nano Super (off-chassis at first)** | `companion.coordinator.Coordinator`: push-to-talk requests, conversation, on-request image interpretation, approved-note retrieval, display state, structured action requests. The compute router (section 6). |
| **Mac Studio** | Optional compute (`studio_local` / `studio_gateway`), only through an authenticated, restricted endpoint. The Stack: approved reading, human-reviewed write proposals, Ring 1/Ring 2 boundaries unchanged. |

Safety notes that are design, not tests:
- The guard depends on no backend: it imports nothing from `compute`, `adapters` or `coordinator`
  (a test checks the imports). Physical control and stopping stay independent of inference.
- A spoken "stop" is handled without the brain, but it is a convenience. Over a lost link the
  coordinator reports a stop's outcome as UNKNOWN. A software stop is never the only stop: the
  commissioning checklist needs a physical cutoff and a controller-level timeout (passdown 4D).
- `arm` is not a command. Only a local operator call on the Pi can arm, and arming is never
  persisted: every guard start is disarmed.

## 6. Hybrid compute (revision 2)

Compute is hybrid by design. Jetson-local, Studio-hosted (Ollama or other local inference, or the
Studio as a credentialed gateway to a cloud provider), and cloud providers directly are all normal
candidates per capability. Cloud is not a last resort.

- **Routing is explicit per task.** `config/compute.yaml` (template, no secrets) has one slot per
  task (`conversation`, `vision_interpret`, `speech_to_text`, `text_to_speech`, `note_retrieval`),
  each naming a primary backend id and an ordered fallback list. `companion.compute.Router` walks
  that list and nothing else.
- **Backend kinds:** `jetson_local` (Ollama at 127.0.0.1:11434 on the Jetson), `studio_local`,
  `studio_gateway`, `cloud` (provider name only).
- **Studio endpoint rule:** `studio_local` must be an authenticated, restricted endpoint, e.g. a
  token-checking proxy bound to the LAN address and allowlisting the Jetson. **A raw Ollama port is
  not to be exposed on every interface: Ollama's local API has no authentication.** The template
  uses a `.invalid` placeholder address on purpose.
- **Credentials:** config holds only the NAME of an environment variable (`credential_env`). A
  backend whose variable is unset reports `not configured (VAR unset)` and is skipped without
  raising; the value is never read into logs or provenance. No credential file lives in the repo.
- **Provenance on every Brain/Vision/Speech result:** `backend_id`, `kind`, `provider`, `model`,
  `where_computed` (jetson | studio | cloud | unknown), `latency_ms`, `cold_start` (null when not
  observable), plus the attempt list. It is logged in the JSON event line (`data.provenance`).
  A request sent to the Studio is not assumed local: Ollama can serve cloud models through a
  signed-in local server, so adapters record what the endpoint reports and the router writes
  `unknown` when it cannot tell.
- **Capabilities:** every compute adapter declares `capabilities()` from
  {text, image, audio, tools, streaming}. The router refuses a backend that lacks what the task needs
  **before any call**, and the coordinator refuses to send an image to a brain that does not
  declare `image`, returning a clear unavailable result. Ollama's OpenAI compatibility is a subset;
  real adapters must probe or declare capabilities, never infer them from a shared API shape.
- Real compute adapters (Ollama client, Studio proxy client, cloud client) are **not written**; they
  are staged with the other real adapters (`companion/adapters/real/README.md`). No paid API call and
  no external transmission of camera, audio or chronicle data happens from anything in this branch.

### Compute placement comparison (measurement HQ will run later; candidates are tested, not decided)

Per task (conversation, vision_interpret, speech_to_text, text_to_speech), for each candidate route
(Jetson local, Studio local, Studio gateway, cloud), measure:
1. **Quality**: a fixed prompt and image set scored blind against a short rubric (correctness,
   brevity, no invented actions); STT word error rate on a few recorded test phrases; TTS
   intelligibility by ear.
2. **Time to first useful response**: wall clock from request to first spoken word, broken into
   network, queue, load, inference, synthesis.
3. **Cost**: tokens or minutes per request and per day at a stated usage pattern; zero-cost routes
   still report energy/heat on the Jetson.
4. **Behavior when busy or unavailable**: the backend stopped, slow, rate-limited, or the LAN down;
   confirm the fallback order fires and the user hears an honest "unavailable".
5. **Studio-local twice**: model already loaded, and cold (after unload), both **under normal HQ
   activity** on the Studio, recording whether the response actually came from local weights
   (`where_computed`).
Output: a table per task with receipts (commands, timestamps, raw timings), then a recommended
`compute.yaml`. The recommendation is a proposal.

**Anthony's decisions (not this lane's, not HQ's to default):** provider and account; credentials and
where they live; usage budget; and which data each route may receive (camera frames, audio,
transcripts, approved notes, chronicle content), per route.

## 7. Rollback

The only artifacts are the worktree and the branch, both on the Studio:
```
git -C ~/Projects/witness-companion-prep/witness_companion worktree remove ../wc-prep
git -C ~/Projects/witness-companion-prep/witness_companion branch -D prep/companion-coordinator
```
That restores the clean clone at `origin/master @ 1728b26`. Nothing was installed or changed on the
Jetson, the Pi, the Studio's services, `~/witness_companion`, or GitHub. Demo and test state goes to
temp directories.
