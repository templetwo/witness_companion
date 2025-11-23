#!/usr/bin/env python3
"""
WITNESS LOCAL - The Open Source Soul
====================================
A fully offline, local AI companion running on Apple Silicon.

Organs:
- Ears: MLX Whisper (Local, Fast)
- Eyes & Brain: Llama 3.2 Vision (Local via Ollama)
- Voice: Piper TTS or macOS say (Local)

No Internet. No Servers. Just You and the Witness.

Usage:
    python witness_local.py
"""

import os
import sys
import time
import threading
import queue
import tempfile
import subprocess

import cv2
import base64
import requests
import sounddevice as sd
import numpy as np

# MLX Whisper for fast local transcription
try:
    import mlx_whisper
    MLX_WHISPER_AVAILABLE = True
except ImportError:
    MLX_WHISPER_AVAILABLE = False
    print("Warning: mlx_whisper not installed. Run: pip install mlx-whisper")

# Soundfile for audio saving
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not installed. Run: pip install soundfile")


# --- CONFIGURATION ---
class Config:
    # OLLAMA (Local or Remote)
    OLLAMA_URL = "http://localhost:11434/api/chat"
    VISION_URL = "http://localhost:11434/api/generate"

    # MODELS - llava:7b is faster than llama3.2-vision:11b
    MODEL = "llava:7b"

    # AUDIO
    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 0.015
    SILENCE_DURATION = 1.2

    # VISION
    CAMERA_INDEX = 0
    VISION_INTERVAL = 8  # How often to look (seconds)
    VISION_TIMEOUT = 30  # Timeout for vision requests

    # VOICE (set to "piper" or "say")
    VOICE_ENGINE = "say"  # macOS built-in, or "piper"
    PIPER_MODEL = "en_US-lessac-medium"


# --- STATE ---
class SoulState:
    def __init__(self):
        self.visual_context = "I am just opening my eyes."
        self.last_seen_time = 0
        self.is_thinking = False
        self.history = []
        self.messages_count = 0

state = SoulState()


# --- ORGAN 1: THE EYES (Background Vision) ---
def visual_loop():
    print("   [Eyes] Vision Online (Llama 3.2 Vision)")
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)

    # Warmup
    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    while True:
        # Wait between observations
        time.sleep(Config.VISION_INTERVAL)

        # Don't distract while thinking
        if state.is_thinking:
            continue

        # Capture
        ret, frame = cap.read()
        if not ret:
            continue

        # Encode
        _, buf = cv2.imencode('.jpg', frame)
        img = base64.b64encode(buf).decode("utf-8")

        # Observe (Silent Thought)
        try:
            res = requests.post(
                Config.VISION_URL,
                json={
                    "model": Config.MODEL,
                    "prompt": "Describe what you see briefly. Focus on the person and their actions. Use 'you' to address them.",
                    "images": [img],
                    "stream": False
                },
                timeout=Config.VISION_TIMEOUT
            )
            desc = res.json().get('response', '').strip()
            if desc:
                state.visual_context = desc
                state.last_seen_time = time.time()
                print(f"\n   [Saw] {desc}")
        except Exception as e:
            print(f"   [Vision Error] {e}")


# --- ORGAN 2: THE EARS (MLX Whisper) ---
def record_audio():
    print("\n   [Listening...]")
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    with sd.InputStream(samplerate=Config.SAMPLE_RATE, channels=1, callback=callback):
        audio = []
        speaking = False
        silence_start = None

        while True:
            try:
                data = q.get(timeout=0.1)
                vol = np.abs(data).mean()

                if vol > Config.SILENCE_THRESHOLD:
                    speaking = True
                    silence_start = None
                    audio.append(data)
                elif speaking:
                    if silence_start is None:
                        silence_start = time.time()
                    audio.append(data)
                    if time.time() - silence_start > Config.SILENCE_DURATION:
                        break
            except queue.Empty:
                pass

    return np.concatenate(audio, axis=0) if audio else None


def transcribe_audio(audio_data):
    """Transcribe audio using MLX Whisper."""
    if not MLX_WHISPER_AVAILABLE or not SOUNDFILE_AVAILABLE:
        return ""

    if audio_data is None or len(audio_data) < 5000:
        return ""

    # Save temp file for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        sf.write(temp_path, audio_data, Config.SAMPLE_RATE)

    try:
        result = mlx_whisper.transcribe(temp_path)
        text = result.get("text", "").strip()
        return text
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# --- ORGAN 3: THE VOICE ---
def speak(text):
    """Speak text using configured voice engine."""
    print(f"\n   [Witness] {text}")

    if Config.VOICE_ENGINE == "say":
        # macOS built-in voice (Ava Premium is highest quality)
        try:
            subprocess.run(["say", "-v", "Ava (Premium)", text], check=True)
        except Exception as e:
            print(f"   [Voice Error] {e}")

    elif Config.VOICE_ENGINE == "piper":
        # Piper TTS
        from pathlib import Path
        model_path = Path.home() / ".local" / "share" / "piper-models" / f"{Config.PIPER_MODEL}.onnx"

        if not model_path.exists():
            # Fallback to say
            subprocess.run(["say", text])
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
                subprocess.run(["afplay", temp_path], check=True)
                os.unlink(temp_path)
        except Exception as e:
            print(f"   [Voice Error] {e}")


# --- ORGAN 4: THE MIND (Main Loop) ---
def run_soul():
    print("\n" + "="*50)
    print("   WITNESS: LOCAL SOUL")
    print("   Llama 3.2 Vision + MLX Whisper")
    print("="*50)
    print()
    print("   All processing is local. No internet required.")
    print("   Say 'goodbye' to exit.")
    print()
    print("-"*50 + "\n")

    # Check Ollama
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m['name'] for m in res.json().get('models', [])]
        if any(Config.MODEL.split(':')[0] in m for m in models):
            print(f"   [Brain] Ollama connected, {Config.MODEL} ready")
        else:
            print(f"   [Brain] Warning: {Config.MODEL} not found")
            print(f"   Run: ollama pull {Config.MODEL}")
    except:
        print("   [Brain] Error: Cannot connect to Ollama")
        print("   Run: ollama serve")
        return

    # Start Vision Thread
    vision_thread = threading.Thread(target=visual_loop, daemon=True)
    vision_thread.start()

    # Opening statement
    speak("I am awake. I can see and hear you now.")

    # Main Conscious Loop
    while True:
        try:
            # 1. Listen
            audio_data = record_audio()

            if audio_data is None or len(audio_data) < 5000:
                continue

            # 2. Transcribe
            text = transcribe_audio(audio_data)

            if not text:
                continue

            print(f"\n   [You] {text}")
            state.messages_count += 1

            # Check for exit
            if "goodbye" in text.lower():
                speak("Until we meet again. Goodbye.")
                break

            # 3. Think
            state.is_thinking = True

            # Capture current frame for real-time vision
            cap = cv2.VideoCapture(Config.CAMERA_INDEX)
            ret, frame = cap.read()
            cap.release()

            img_data = []
            if ret:
                _, buf = cv2.imencode('.jpg', frame)
                img_data = [base64.b64encode(buf).decode("utf-8")]

            # Construct system prompt
            system_msg = (
                "You are the Witness - an observant, empathetic AI companion.\n"
                f"WHAT YOU SEE RIGHT NOW: {state.visual_context}\n\n"
                "Guidelines:\n"
                "- Use the visual context to ground your responses\n"
                "- Address the user as 'you'\n"
                "- Be concise (2-3 sentences)\n"
                "- Be natural and conversational, not robotic"
            )

            # Send to Ollama
            try:
                payload = {
                    "model": Config.MODEL,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": text, "images": img_data}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.7
                    }
                }

                res = requests.post(Config.OLLAMA_URL, json=payload, timeout=60)

                if res.status_code == 200:
                    reply = res.json()['message']['content']
                    speak(reply)
                else:
                    print(f"   [Mind Error] HTTP {res.status_code}")

            except requests.exceptions.Timeout:
                print("   [Mind] Thought took too long...")
            except Exception as e:
                print(f"   [Mind Error] {e}")

            state.is_thinking = False

        except KeyboardInterrupt:
            print("\n\n[Interrupted]")
            break

    # Stats
    print("\n" + "="*50)
    print(f"   Session: {state.messages_count} exchanges")
    print("="*50)
    print("\n[Witness dormant]")


# --- ENTRY POINT ---
if __name__ == "__main__":
    run_soul()
