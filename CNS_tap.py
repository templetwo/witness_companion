#!/usr/bin/env python3
"""
CNS_tap.py - The Soul Bundle (WebSocket Edition)
================================================
Connects to a running Moshi Web Server and adds the "Deep Soul" layer.

Architecture:
- Moshi (Localhost:8998): Handles audio I/O and fast chat.
- CNS_tap (Localhost): Listens to Moshi's text stream via WebSocket.
- Studio (Remote): Handles Vision (LLaVA) and Deep Thought (Dolphin).

Usage:
1. Run Moshi: python -m moshi_mlx.local_web
2. Run CNS: python CNS_tap.py
"""

import asyncio
import websockets
import json
import threading
import time
import tempfile
import subprocess
import sys
import os
from pathlib import Path

import cv2
import base64
import requests


# --- CONFIGURATION ---
class Config:
    # REMOTE BRAIN (Mac Studio)
    STUDIO_IP = "192.168.1.195"
    OLLAMA_URL = f"http://{STUDIO_IP}:11434/api/chat"
    VISION_URL = f"http://{STUDIO_IP}:11434/api/generate"

    # MODELS
    DEEP_MODEL = "dolphin3:8b"
    VISION_MODEL = "llava:7b"

    # MOSHI CONNECTION
    MOSHI_URI = "ws://localhost:8998/api/chat"

    # TTS
    PIPER_MODEL = "en_US-lessac-medium"
    PIPER_MODEL_DIR = Path.home() / ".local" / "share" / "piper-models"

    # VISION
    CAMERA_INDEX = 0
    MOTION_THRESHOLD = 3000


# --- STATE ---
class SoulState:
    def __init__(self):
        self.visual_context = "Darkness. Waiting for eyes."
        self.last_seen_time = 0
        self.is_deep_thinking = False
        self.conversation_buffer = []

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


# --- LAYER 2: VISION (Background) ---
def visual_loop():
    print("   [Eyes] Vision Online (Studio)")
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
                            "prompt": "Describe ONLY what you actually see. Focus on the main person. Use second person ('you'). Be factual, 1 sentence.",
                            "images": [img],
                            "stream": False
                        },
                        timeout=8
                    )
                    desc = res.json().get('response', '').strip()
                    if desc:
                        state.visual_context = desc
                        state.last_seen_time = time.time()
                        print(f"\n   [Saw]: {desc}")
                        time.sleep(4)
                except:
                    pass

        time.sleep(0.1)


# --- LAYER 3: DEEP MIND (Triggered) ---
def trigger_deep_thought(trigger_text):
    if state.is_deep_thinking:
        return
    state.is_deep_thinking = True

    print(f"\n   [Brain] Deep Mind Triggered: '{trigger_text[:50]}...'")

    vision_txt = state.visual_context
    if time.time() - state.last_seen_time > 60:
        vision_txt += " (Memory fading)"

    # Recent conversation context
    recent = "\n".join(state.conversation_buffer[-5:])

    system_prompt = (
        f"You are the Conscious Mind of the Witness.\n"
        f"Your Subconscious (Moshi) is chatting fast.\n"
        f"The user asked a DEEP question. Interject wisely.\n"
        f"VISUAL CONTEXT: {vision_txt}\n"
        f"RECENT CHAT:\n{recent}\n\n"
        f"Keep response concise (2-3 sentences)."
    )

    try:
        res = requests.post(
            Config.OLLAMA_URL,
            json={
                "model": Config.DEEP_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Reflect on: {trigger_text}"}
                ],
                "stream": False,
                "options": {"temperature": 0.8}
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


# --- LAYER 1: SUBCONSCIOUS (WebSocket Tap) ---
async def listen_to_moshi():
    print(f"   [Subconscious] Connecting to Moshi at {Config.MOSHI_URI}...")

    try:
        async with websockets.connect(Config.MOSHI_URI) as websocket:
            print("   [Subconscious] Connected to Moshi Stream")
            print("   [Debug] Waiting for messages...")

            while True:
                message = await websocket.recv()

                # Debug: show raw message
                print(f"   [Raw]: {message[:200]}..." if len(message) > 200 else f"   [Raw]: {message}")

                try:
                    data = json.loads(message)

                    # Debug: show all keys
                    print(f"   [Keys]: {list(data.keys())}")

                    # Parse Moshi's message format
                    # Try common fields
                    text = None
                    if 'text' in data:
                        text = data['text'].strip()
                    elif 'content' in data:
                        text = data['content'].strip()
                    elif 'message' in data:
                        text = data['message'].strip()
                    elif 'transcript' in data:
                        text = data['transcript'].strip()
                    elif 'response' in data:
                        text = data['response'].strip()

                    if text:
                        # Buffer conversation
                        state.conversation_buffer.append(text)
                        if len(state.conversation_buffer) > 20:
                            state.conversation_buffer.pop(0)

                        print(f"   [Moshi]: {text}")

                        # Check for deep thought triggers
                        user_triggers = ["witness", "what do you see", "what do you think", "tell me about", "can you explain"]
                        if any(t in text.lower() for t in user_triggers):
                            threading.Thread(
                                target=trigger_deep_thought,
                                args=(text,),
                                daemon=True
                            ).start()

                except json.JSONDecodeError:
                    # Plain text or binary message
                    if isinstance(message, bytes):
                        print(f"   [Binary]: {len(message)} bytes")
                    else:
                        text = message.strip()
                        if text:
                            state.conversation_buffer.append(text)
                            print(f"   [Moshi]: {text}")

    except ConnectionRefusedError:
        print("\n   [Error] Cannot connect to Moshi!")
        print("   Make sure Moshi is running: python -m moshi_mlx.local_web")
    except asyncio.CancelledError:
        print("\n   [Disconnected]")
    except Exception as e:
        print(f"\n   [Error] WebSocket: {e}")


# --- MAIN ---
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("   WITNESS: WEBSOCKET SOUL")
    print("   Tapping into Moshi Stream")
    print("=" * 50 + "\n")

    print("   Prerequisites:")
    print("   1. Run Moshi first: python -m moshi_mlx.local_web")
    print("   2. Wait for 'Listening on localhost:8998'")
    print("   3. Then run this script\n")

    # Start Vision
    threading.Thread(target=visual_loop, daemon=True).start()

    print("   Organs:")
    print("   - Subconscious: Moshi WebSocket (localhost:8998)")
    print("   - Eyes: LLaVA (Studio)")
    print("   - Brain: Dolphin (Studio)")
    print("   - Voice: Piper (Local)")

    print("\n" + "-" * 50)
    print("Say 'Witness, what do you see?' to wake Deep Mind")
    print("Ctrl+C to exit")
    print("-" * 50 + "\n")

    try:
        asyncio.run(listen_to_moshi())
    except KeyboardInterrupt:
        print("\n\n[Session interrupted]")

    print("\n[CNS shutdown - Soul dormant]")
