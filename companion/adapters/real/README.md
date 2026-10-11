# Real adapters: STAGED, NOT WRITTEN

Nothing in this directory talks to hardware yet. When real adapters are written:

- Each one lazy-imports its device library (`sounddevice`, `cv2`, `piper`,
  `faster_whisper`, `serial`, OLED drivers) **inside** its constructor or first
  call, never at module top level. `import companion` must stay device-free;
  `tests/test_no_device_import.py` enforces this in a subprocess.
- Real adapters are only constructed when the entry point is started with an
  explicit `--armed` flag. Without it, only the fakes in `companion/adapters/`
  are wired.
- Do not import `witness_brain.speech`, `witness_brain.vision` or
  `witness_brain.cns` from `companion/` at module level: those modules import
  device and model libraries at import time, create `~/.cache/witness_brain_models`,
  and `TTS()` downloads a voice from the network on construction. Wrap them by
  lazy import inside a real adapter only.
- A real Brain adapter wrapping `witness_brain.models.brain_client.BrainClient`
  must NOT pass through its stub fallback: on any Ollama error that client
  returns a canned "I heard you" plus `action="log_mood_observation"`. The
  wrapper should raise `BackendUnavailable` instead.
- Planned list: `real/mic_ptt.py` (sounddevice, push-to-talk only),
  `real/stt_whisper.py` (faster-whisper or whisper.cpp), `real/tts_piper.py`
  (the Jetson already has piper 1.3.0 and en_US-lessac-medium),
  `real/camera_snapshot.py` (cv2 or RTSP, on request only, writes one
  timestamped file), `real/oled.py`, `real/pi_link.py` (serial or socket to
  the Pi-side guard).
