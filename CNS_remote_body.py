#!/usr/bin/env python3
"""
CNS_Remote_Body.py - The Thin Client Soul
=========================================
Run this on the MacBook (or Jetson).
It connects to the Brain (Mac Studio) for everything.

- Audio: Uses Browser (Moshi Web UI) for low-latency voice.
- Vision: Sends camera frames to Studio (LLaVA).
- Mind: Taps into Moshi's WebSocket to trigger Studio (Dolphin).

Usage:
1. Open browser to http://192.168.1.195:8998 (Moshi UI)
2. Run this script: python CNS_remote_body.py
3. Talk to Moshi in browser - Deep Mind will interject when triggered
"""

import asyncio
import websockets
import threading
import time
import cv2
import base64
import requests
import os
import sys
import re
import subprocess
import tempfile
from pathlib import Path


# --- CONFIGURATION ---
class Config:
    # BRAIN IP (Mac Studio)
    STUDIO_IP = "192.168.1.195"

    # ENDPOINTS
    # Moshi via SSH tunnel (ssh -L 8998:localhost:8998 tony_studio@192.168.1.195)
    MOSHI_WS = "ws://localhost:8998/api/chat"
    # Ollama direct to Studio
    OLLAMA_URL = f"http://{STUDIO_IP}:11434/api/chat"
    VISION_URL = f"http://{STUDIO_IP}:11434/api/generate"

    # MODELS
    DEEP_MODEL = "dolphin3:8b"
    VISION_MODEL = "llava:7b"

    # BODY SENSORS
    CAMERA_INDEX = 0
    MOTION_THRESHOLD = 5000

    # TIMING
    TRIGGER_COOLDOWN = 10  # seconds between deep thoughts
    VISION_COOLDOWN = 4    # seconds between vision updates

    # TTS
    PIPER_MODEL = "en_US-lessac-medium"
    PIPER_MODEL_DIR = Path.home() / ".local" / "share" / "piper-models"


# --- STATE ---
class SoulState:
    def __init__(self):
        self.visual_context = "Darkness."
        self.last_seen_time = 0
        self.is_deep_thinking = False
        self.last_deep_thought_time = 0
        self.moshi_messages = 0
        self.deep_thoughts = 0
        self.vision_updates = 0

state = SoulState()


# --- TEXT PROCESSING ---
def clean_text(text: str) -> str:
    """Clean Moshi's output."""
    if not text:
        return ""
    text = re.sub(r'\[LAG\]', '', text)
    text = re.sub(r'\|+', '', text)
    text = re.sub(r'/{2,}', '', text)
    text = re.sub(r'\[\d+\.?\d*s\]', '', text)
    text = text.strip()
    if re.match(r'^[^\w\s]+$', text):
        return ""
    return text


def should_trigger_deep_thought(text: str) -> bool:
    """Determine if this should trigger System 2."""

    # Cooldown
    if time.time() - state.last_deep_thought_time < Config.TRIGGER_COOLDOWN:
        return False

    if state.is_deep_thinking:
        return False

    if len(text) > 100 or len(text) < 5:
        return False

    text_lower = text.lower()

    # Direct triggers
    triggers = [
        "witness", "what do you see", "what do you think",
        "tell me about", "explain", "why", "meaning",
        "what is", "who is", "where is"
    ]

    if any(t in text_lower for t in triggers):
        return True

    # Short questions
    if '?' in text and len(text) < 50:
        return True

    return False


# --- TTS ---
def speak(text: str):
    """Speak text using Piper TTS."""
    print(f"\n   [Deep Voice] {text}")

    model_path = Config.PIPER_MODEL_DIR / f"{Config.PIPER_MODEL}.onnx"

    if not model_path.exists():
        # Fallback to system say
        if sys.platform == "darwin":
            os.system(f'say "{text}"')
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
        process.communicate(input=text.encode('utf-8'), timeout=5)

        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            if sys.platform == "darwin":
                subprocess.run(["afplay", temp_path], check=True, timeout=15)
            else:
                subprocess.run(["aplay", "-q", temp_path], check=True, timeout=15)
            os.unlink(temp_path)

    except Exception as e:
        print(f"   [Voice Error] {e}")


# --- ORGAN 1: VISION (Body -> Brain) ---
def visual_loop():
    print("   [Eyes] Vision Online (Streaming to Studio)")
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)

    # Warmup
    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    ret, prev = cap.read()
    if not ret:
        print("   [Eyes] Camera unavailable")
        return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        score = cv2.countNonZero(thresh)
        prev_gray = gray

        if score > Config.MOTION_THRESHOLD:
            time.sleep(0.3)
            ret, clean_frame = cap.read()
            if ret:
                _, buf = cv2.imencode('.jpg', clean_frame)
                img = base64.b64encode(buf).decode("utf-8")

                # Fire and forget
                threading.Thread(target=update_vision, args=(img,), daemon=True).start()

                time.sleep(Config.VISION_COOLDOWN)

        time.sleep(0.1)


def update_vision(img):
    try:
        res = requests.post(
            Config.VISION_URL,
            json={
                "model": Config.VISION_MODEL,
                "prompt": "Brief: What is the person doing? Use 'you'. Max 10 words.",
                "images": [img],
                "stream": False
            },
            timeout=8
        )
        desc = res.json().get('response', '').strip()
        if desc:
            state.visual_context = desc
            state.last_seen_time = time.time()
            state.vision_updates += 1
            print(f"\n   [Saw] {desc}")
    except Exception as e:
        print(f"   [Vision Error] {e}")


# --- ORGAN 2: DEEP MIND (Brain -> Body) ---
def trigger_deep_thought(trigger_text):
    state.is_deep_thinking = True
    state.last_deep_thought_time = time.time()
    state.deep_thoughts += 1

    print(f"\n   [Deep Mind] Triggered by: '{trigger_text[:50]}...'")

    vision_txt = state.visual_context
    if time.time() - state.last_seen_time > 60:
        vision_txt += " (Memory fading)"

    system_prompt = (
        f"You are the Conscious Mind. Your Subconscious (Moshi) is chatting.\n"
        f"The user asked a question that requires depth. Interject wisely.\n"
        f"VISUAL CONTEXT: {vision_txt}\n"
        f"Respond in 2-3 concise sentences."
    )

    try:
        res = requests.post(
            Config.OLLAMA_URL,
            json={
                "model": Config.DEEP_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": trigger_text}
                ],
                "stream": False,
                "options": {"temperature": 0.8, "top_p": 0.9}
            },
            timeout=60
        )

        thought = res.json()['message']['content']
        speak(thought)

    except Exception as e:
        print(f"   [Brain Error] {e}")

    state.is_deep_thinking = False


# --- ORGAN 3: SUBCONSCIOUS TAP (Brain -> Body) ---
async def listen_to_moshi():
    print(f"   [Stream] Connecting to Moshi at {Config.MOSHI_WS}...")

    text_buffer = []

    try:
        async with websockets.connect(Config.MOSHI_WS) as websocket:
            print("   [Stream] Connected to Subconscious")
            print("   [Stream] Waiting for handshake...")

            while True:
                try:
                    message = await websocket.recv()
                    state.moshi_messages += 1

                    # Moshi sends binary with type prefix
                    if not isinstance(message, bytes) or len(message) == 0:
                        continue

                    msg_type = message[0]
                    payload = message[1:]

                    if msg_type == 0x00:
                        # Handshake
                        print("   [Stream] Handshake received - active")
                        continue

                    elif msg_type == 0x01:
                        # Audio - skip
                        continue

                    elif msg_type == 0x02:
                        # Text token
                        try:
                            text_token = payload.decode('utf-8')
                        except:
                            continue

                        text_buffer.append(text_token)
                        combined = "".join(text_buffer)

                        # Flush on sentence end
                        if any(combined.rstrip().endswith(p) for p in '.!?'):
                            text = clean_text(combined)
                            text_buffer = []

                            if text:
                                print(f"   [Moshi] {text}")

                                if should_trigger_deep_thought(text):
                                    threading.Thread(
                                        target=trigger_deep_thought,
                                        args=(text,),
                                        daemon=True
                                    ).start()

                        # Also flush if too long
                        elif len(combined) > 80:
                            text = clean_text(combined)
                            text_buffer = []

                            if text:
                                print(f"   [Moshi] {text}")

                                if should_trigger_deep_thought(text):
                                    threading.Thread(
                                        target=trigger_deep_thought,
                                        args=(text,),
                                        daemon=True
                                    ).start()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"   [Stream Error] {e}")

    except ConnectionRefusedError:
        print("\n   Cannot connect to Moshi!")
        print(f"   Is Moshi running on Studio? ({Config.STUDIO_IP}:8998)")
    except Exception as e:
        print(f"\n   WebSocket Error: {e}")


# --- MAIN ---
def show_stats():
    print("\n" + "="*50)
    print("   SESSION STATS")
    print("="*50)
    print(f"   Moshi Messages: {state.moshi_messages}")
    print(f"   Deep Thoughts: {state.deep_thoughts}")
    print(f"   Vision Updates: {state.vision_updates}")
    print("="*50)


if __name__ == "__main__":
    print("\n" + "="*50)
    print("   WITNESS: REMOTE BODY")
    print("   Brain: Mac Studio | Body: MacBook")
    print("="*50)
    print()
    print("   Setup (one-time):")
    print("   ssh -L 8998:localhost:8998 tony_studio@192.168.1.195")
    print()
    print("   Then:")
    print("   1. Open browser: http://localhost:8998")
    print("   2. Click Connect, allow microphone")
    print("   3. Talk to Moshi in browser")
    print("   4. Deep Mind will interject on triggers")
    print()
    print("   Triggers: 'Witness', questions with '?'")
    print("   Ctrl+C to exit")
    print("="*50 + "\n")

    # Start Vision
    threading.Thread(target=visual_loop, daemon=True).start()

    # Start Stream
    try:
        asyncio.run(listen_to_moshi())
    except KeyboardInterrupt:
        print("\n\n[Disconnected]")
        show_stats()

    print("[Remote Body offline]")
