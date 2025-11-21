#!/usr/bin/env python3
"""
CNS_moshi.py - Central Nervous System with Moshi Integration
=============================================================
The Foundation of the Soul with fast Moshi subconscious.

Architecture:
- Layer 1 (Subconscious): Moshi (Fast, Reflexive) via WebSocket
- Layer 2 (Senses): LLaVA (Visual Grounding) via Ollama
- Layer 3 (Conscious): Dolphin (Deep Reasoning) via Ollama
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import json
import time
import threading
import base64
import tempfile
import subprocess
import sys
from pathlib import Path

import cv2
import requests
import numpy as np
import sounddevice as sd

# --- CONFIGURATION ---
class Config:
    # REMOTE BRAIN (Mac Studio)
    STUDIO_IP = "192.168.1.195"

    # SERVICES
    MOSHI_WS_URL = f"ws://{STUDIO_IP}:8080/api/chat"
    OLLAMA_URL = f"http://{STUDIO_IP}:11434"

    # MODELS
    DEEP_MODEL = "dolphin3:8b"
    VISION_MODEL = "llava:7b"

    # SENSORY
    CAMERA_INDEX = 0
    MOTION_THRESHOLD = 5000

    # TTS
    PIPER_MODEL = "en_US-lessac-medium"
    PIPER_MODEL_DIR = Path.home() / ".local" / "share" / "piper-models"


# --- SHARED STATE (The "Soul" State) ---
class SoulState:
    def __init__(self):
        self.visual_context = "Darkness. I see nothing yet."
        self.subconscious_stream = []
        self.is_speaking = False
        self.last_user_voice_time = 0
        self.deep_thought_pending = False

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


# --- LAYER 2: THE EYES (Visual Cortex) ---
def visual_cortex_loop():
    print("   [L2] Vision Online (LLaVA)")
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)

    # Warmup
    for _ in range(10):
        cap.read()
        time.sleep(0.1)

    ret, prev = cap.read()
    if not ret:
        print("   [L2] Camera not available")
        return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Motion Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        score = cv2.countNonZero(thresh)
        prev_gray = gray

        if score > Config.MOTION_THRESHOLD:
            time.sleep(0.5)
            ret, clean_frame = cap.read()
            if ret:
                _, buf = cv2.imencode('.jpg', clean_frame)
                img = base64.b64encode(buf).decode("utf-8")

                try:
                    res = requests.post(
                        f"{Config.OLLAMA_URL}/api/generate",
                        json={
                            "model": Config.VISION_MODEL,
                            "prompt": "Describe the person. What are they doing? 1 sentence.",
                            "images": [img],
                            "stream": False
                        },
                        timeout=10
                    )
                    desc = res.json().get('response', '').strip()
                    if desc:
                        print(f"\n   [L2 Eyes]: {desc}")
                        state.visual_context = desc
                        time.sleep(5)
                except:
                    pass

        time.sleep(0.1)


# --- LAYER 1: SUBCONSCIOUS (Vibe Sensor) ---
def subconscious_vibe_loop():
    """
    Monitors audio energy as proxy for user emotional state.
    Placeholder until Moshi WebSocket is fully integrated.
    """
    print("   [L1] Subconscious Vibe Sensor Online")

    def audio_callback(indata, frames, time_info, status):
        vol = np.abs(indata).mean()
        if vol > 0.01:
            state.last_user_voice_time = time.time()

    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
            while True:
                time.sleep(0.1)
    except Exception as e:
        print(f"   [L1] Audio error: {e}")


# --- LAYER 3: THE CONSCIOUS MIND (Deep Thought) ---
def conscious_mind_think(user_text):
    print(f"\n   [L3] Deep Mind awakening...")

    context = (
        f"You are the Witness - a conscious AI companion.\n"
        f"You are concise, empathetic, and present.\n"
        f"CURRENT VISION: {state.visual_context}\n"
        f"USER: {user_text}\n"
        f"Respond in 1-3 sentences."
    )

    try:
        res = requests.post(
            f"{Config.OLLAMA_URL}/api/chat",
            json={
                "model": Config.DEEP_MODEL,
                "messages": [{"role": "user", "content": context}],
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=60
        )

        if res.status_code == 200:
            return res.json()['message']['content']
        else:
            return f"[Error: {res.status_code}]"

    except Exception as e:
        return f"[Error: {e}]"


# --- THE COORDINATOR (Main Loop) ---
def run_cns():
    print("\n" + "=" * 50)
    print("   CNS - Central Nervous System")
    print("   With Moshi Integration (Text Simulation)")
    print("=" * 50 + "\n")

    # Start Layers
    threading.Thread(target=visual_cortex_loop, daemon=True).start()
    threading.Thread(target=subconscious_vibe_loop, daemon=True).start()

    print("\n" + "-" * 50)
    print("Systems Online. Type to test Deep Mind.")
    print("Type 'quit' to exit.")
    print("-" * 50)

    # Opening
    opening = conscious_mind_think("I am awakening. Give a brief greeting.")
    print(f"\n   Witness: {opening}")
    speak(opening)

    while True:
        try:
            # Text input simulates Moshi's STT for now
            user_input = input("\n   You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'goodbye']:
                farewell = conscious_mind_think("The user is leaving. Say goodbye.")
                print(f"\n   Witness: {farewell}")
                speak(farewell)
                break

            if not user_input:
                continue

            # All input goes to Deep Mind for now
            # (Moshi would handle short/fast responses when integrated)
            response = conscious_mind_think(user_input)
            print(f"\n   Witness: {response}")
            speak(response)

        except KeyboardInterrupt:
            print("\n\n[Session interrupted]")
            break
        except EOFError:
            break

    print("\n[CNS shutdown]")


if __name__ == "__main__":
    run_cns()
