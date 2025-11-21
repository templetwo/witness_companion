#!/usr/bin/env python3
"""
CNS_integrated.py - The Soul Bundle (PTY Version)
==================================================
Coordinates Moshi (Subconscious) and Dolphin (Conscious Mind).

Uses PTY for proper Moshi output handling.

Author: Temple of Two
"""

import os
import sys
import pty
import select
import subprocess
import threading
import time
import tempfile
import base64
from pathlib import Path

import cv2
import requests


# --- CONFIGURATION ---
class Config:
    # Network
    STUDIO_IP = "192.168.1.195"

    # Brain Endpoints
    OLLAMA_URL = f"http://{STUDIO_IP}:11434/api/chat"
    VISION_URL = f"http://{STUDIO_IP}:11434/api/generate"

    # Models
    DEEP_MODEL = "dolphin3:8b"
    VISION_MODEL = "llava:7b"

    # Moshi Command
    MOSHI_CMD = [sys.executable, "-m", "moshi_mlx.local", "-q", "4"]

    # TTS
    PIPER_MODEL = "en_US-lessac-medium"
    PIPER_MODEL_DIR = Path.home() / ".local" / "share" / "piper-models"

    # Vision
    CAMERA_INDEX = 0
    MOTION_THRESHOLD = 3000


# --- SHARED STATE ---
class SoulState:
    def __init__(self):
        self.visual_context = "Darkness. I am waiting to see."
        self.last_seen_time = 0
        self.conversation_log = []
        self.is_deep_thinking = False

state = SoulState()


# --- VOICE OUTPUT ---
def speak(text):
    """Output speech via Piper TTS."""
    model_path = Config.PIPER_MODEL_DIR / f"{Config.PIPER_MODEL}.onnx"

    if not model_path.exists():
        print(f"   [Voice] Piper model not found")
        return

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        process = subprocess.Popen(
            ["piper", "--model", str(model_path), "--output_file", temp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        process.communicate(input=text.encode('utf-8'))

        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            if sys.platform == "darwin":
                subprocess.run(["afplay", temp_path], check=True)
            else:
                subprocess.run(["aplay", "-q", temp_path], check=True)
            os.unlink(temp_path)
    except Exception as e:
        print(f"   [Voice] Error: {e}")


# --- LAYER 2: VISION ---
def visual_loop():
    """Background vision loop."""
    print("   [Eyes] Vision Online")
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)

    # Warmup
    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    ret, prev = cap.read()
    if not ret:
        print("   [Eyes] Camera not available")
        return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.countNonZero(
            cv2.threshold(cv2.absdiff(prev_gray, gray), 25, 255, cv2.THRESH_BINARY)[1]
        )
        prev_gray = gray

        if score > Config.MOTION_THRESHOLD:
            time.sleep(0.3)
            ret, clean_frame = cap.read()
            if ret:
                _, buf = cv2.imencode('.jpg', clean_frame)
                img = base64.b64encode(buf).decode("utf-8")

                try:
                    res = requests.post(
                        Config.VISION_URL,
                        json={
                            "model": Config.VISION_MODEL,
                            "prompt": "Brief: What is the person doing? Use 'you'. Max 10 words.",
                            "images": [img],
                            "stream": False
                        },
                        timeout=10
                    )
                    desc = res.json().get('response', '').strip()
                    if desc:
                        print(f"\n   [Saw]: {desc}")
                        state.visual_context = desc
                        state.last_seen_time = time.time()
                        time.sleep(3)
                except Exception as e:
                    pass

        time.sleep(0.1)


# --- LAYER 3: DEEP MIND ---
def trigger_deep_thought(trigger_text):
    """Wake the Conscious Mind (Dolphin)."""
    if state.is_deep_thinking:
        return
    state.is_deep_thinking = True

    print(f"\n   [Brain] Deep Mind Triggered: '{trigger_text[:50]}...'")

    # Gather context
    recent_chat = "\n".join(state.conversation_log[-5:])
    vision_age = time.time() - state.last_seen_time
    vision_txt = state.visual_context
    if vision_age > 60:
        vision_txt += " (fading)"

    system_prompt = (
        f"You are the Conscious Mind of the Witness.\n"
        f"Your Subconscious (Moshi) is chatting fast.\n"
        f"The user asked something DEEP. Interject wisely.\n"
        f"VISUAL CONTEXT: {vision_txt}\n"
        f"RECENT CHAT:\n{recent_chat}\n\n"
        f"Keep response concise (2-3 sentences)."
    )

    try:
        res = requests.post(
            Config.OLLAMA_URL,
            json={
                "model": Config.DEEP_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Reflect on this: {trigger_text}"}
                ],
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=60
        )

        if res.status_code == 200:
            thought = res.json()['message']['content']
            print(f"\n   [DEEP VOICE]: {thought}")
            speak(thought)
        else:
            print(f"   [Brain] Error: {res.status_code}")

    except Exception as e:
        print(f"   [Brain] Error: {e}")

    state.is_deep_thinking = False


# --- LAYER 1: MOSHI SUBCONSCIOUS (PTY Version) ---
def run_moshi_with_pty():
    """Run Moshi using PTY for proper output handling."""
    print("   [Subconscious] Starting Moshi with PTY...")

    # Create pseudo-terminal
    master_fd, slave_fd = pty.openpty()

    try:
        process = subprocess.Popen(
            Config.MOSHI_CMD,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True
        )

        os.close(slave_fd)  # Close slave in parent

        print("   [Subconscious] Moshi Online - Reading stream...")

        buffer = ""

        while True:
            # Check if process is still running
            if process.poll() is not None:
                print("   [Subconscious] Moshi process ended")
                break

            # Check for output (non-blocking)
            ready, _, _ = select.select([master_fd], [], [], 0.1)

            if ready:
                try:
                    data = os.read(master_fd, 1024).decode('utf-8', errors='ignore')
                    if data:
                        buffer += data

                        # Process complete lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()

                            if not line:
                                continue

                            # Skip status/debug lines
                            if line.startswith('[') or line.startswith('Loading') or line.startswith('Info:'):
                                print(f"   [Moshi Status]: {line}")
                                continue

                            # Skip separator lines
                            if line.startswith('-') and len(line) > 10:
                                continue

                            # Parse Moshi's piped output format: | text here |
                            if '|' in line:
                                # Extract text between pipes, strip borders and [LAG] markers
                                text = line.replace('|', '').replace('[LAG]', '').strip()
                                # Remove trailing / from continuations
                                text = text.rstrip('/')

                                if not text or len(text) < 3:
                                    continue

                                # Log conversation
                                state.conversation_log.append(text)
                                if len(state.conversation_log) > 20:
                                    state.conversation_log.pop(0)

                                print(f"   [Moshi]: {text}")

                                # Only trigger on questions directed at us
                                # Must contain ? AND be short enough to be a question (not Moshi rambling)
                                is_question = "?" in text and len(text) < 100

                                # Deep topic words
                                deep_triggers = ["witness", "what do you see", "what do you think", "who are you", "tell me"]
                                is_addressed = any(t in text.lower() for t in deep_triggers)

                                if is_question or is_addressed:
                                    print(f"   [Trigger: {text[:40]}...]")
                                    threading.Thread(
                                        target=trigger_deep_thought,
                                        args=(text,),
                                        daemon=True
                                    ).start()
                            else:
                                # Non-piped output (status messages)
                                if line and not line.startswith('Info:'):
                                    print(f"   [Status]: {line}")

                except OSError:
                    break

        os.close(master_fd)

    except Exception as e:
        print(f"   [Subconscious] Moshi PTY Error: {e}")
        print("   Falling back to text input...")
        run_fallback_loop()


def run_fallback_loop():
    """Fallback to manual text input."""
    print("\n   [Fallback Mode] Type to trigger deep thoughts")
    print("   Type 'quit' to exit\n")

    while True:
        try:
            txt = input("   You: ").strip()

            if txt.lower() in ['quit', 'exit', 'goodbye']:
                break

            if txt:
                state.conversation_log.append(f"User: {txt}")
                threading.Thread(
                    target=trigger_deep_thought,
                    args=(txt,),
                    daemon=True
                ).start()

                # Give deep thought time to respond
                time.sleep(0.5)

        except (KeyboardInterrupt, EOFError):
            break


# --- MAIN ---
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("   WITNESS INTEGRATED SOUL (PTY)")
    print("   L1: Moshi (Subconscious)")
    print("   L2: LLaVA (Vision)")
    print("   L3: Dolphin (Conscious)")
    print("=" * 50 + "\n")

    # Start Vision
    threading.Thread(target=visual_loop, daemon=True).start()

    print("   Organs:")
    print("   - Subconscious: Moshi (Local PTY)")
    print("   - Eyes: LLaVA (Studio)")
    print("   - Brain: Dolphin (Studio)")
    print("   - Voice: Piper (Local)")

    print("\n" + "-" * 50)
    print("Deep questions ('why', '?') escalate to Dolphin.")
    print("Ctrl+C to exit.")
    print("-" * 50 + "\n")

    try:
        run_moshi_with_pty()
    except KeyboardInterrupt:
        print("\n\n[Session interrupted]")
    except Exception as e:
        print(f"\n[Error: {e}]")
        print("Trying fallback mode...")
        run_fallback_loop()

    print("\n[CNS shutdown - Soul dormant]")
