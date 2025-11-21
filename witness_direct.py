#!/usr/bin/env python3
"""
WITNESS - Direct Mode (No Filters)
==================================
Responds to EVERYTHING - no wake words, no filtering.
Best for quiet environments.
"""

# FIX PARALLELISM WARNINGS - MUST be before imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import queue
import time
import threading
import base64
import datetime
import tempfile
import subprocess
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import requests
import chromadb
from faster_whisper import WhisperModel


# --- CONFIGURATION ---
class Config:
    OLLAMA_HOST = "http://localhost:11434"
    CHAT_MODEL = "llama3:8b"
    VISION_MODEL = "moondream"
    MEMORY_PATH = "./witness_memory_db"

    # SENSITIVITY (Calibrated for quiet room)
    SILENCE_THRESHOLD = 0.01
    SILENCE_DURATION = 1.0

    # VISION
    CAMERA_INDEX = 0
    MOTION_THRESHOLD = 5000

    # TTS
    PIPER_MODEL = "en_US-lessac-medium"
    PIPER_MODEL_DIR = Path.home() / ".local" / "share" / "piper-models"


# --- THE SOUL ---
WITNESS_SOUL = """You are the Witness. A mobile AI companion.
You are concise, observant, and friendly.
You possess long-term memory.
If the user mentions visual things, refer to the [VISUAL CONTEXT].
Keep responses brief (1-3 sentences)."""


# --- SYSTEM 1: HIPPOCAMPUS (Memory) ---
class MemorySystem:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.MEMORY_PATH)
        self.collection = self.client.get_or_create_collection(name="witness_logs")
        count = self.collection.count()
        print(f"   Memory loaded ({count} memories)")

    def remember(self, user_text, ai_response):
        timestamp = datetime.datetime.now().isoformat()
        self.collection.add(
            documents=[f"User: {user_text} | Witness: {ai_response}"],
            metadatas=[{"timestamp": timestamp}],
            ids=[f"mem_{int(time.time() * 1000)}"]
        )

    def recall(self, query_text):
        if self.collection.count() == 0:
            return "No memories yet."
        results = self.collection.query(query_texts=[query_text], n_results=2)
        if results['documents'][0]:
            return " | ".join(results['documents'][0])
        return "No relevant memories."


# --- SYSTEM 2: VISUAL CORTEX (Motion Triggered) ---
class VisualCortex(threading.Thread):
    def __init__(self, context_callback):
        super().__init__()
        self.context_callback = context_callback
        self.running = True
        self.daemon = True
        self.last_desc = "Nothing observed yet."

    def get_description(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode("utf-8")
        try:
            res = requests.post(
                f"{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": Config.VISION_MODEL,
                    "prompt": "What objects and people do you see? Be specific. 1 sentence.",
                    "images": [img_str],
                    "stream": False
                },
                timeout=10
            )
            return res.json().get('response', '').strip()
        except:
            return None

    def run(self):
        print("   Vision online (motion-activated)")
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)

        # Warm up camera
        for _ in range(10):
            cap.read()
            time.sleep(0.1)

        ret, prev_frame = cap.read()
        if ret:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = cv2.countNonZero(thresh)
            prev_gray = gray

            if motion_score > Config.MOTION_THRESHOLD:
                time.sleep(0.5)
                ret, clean_frame = cap.read()
                if ret:
                    desc = self.get_description(clean_frame)
                    if desc:
                        print(f"\n   [Motion]: {desc}")
                        self.context_callback(desc)
                        self.last_desc = desc
                        time.sleep(5)

            time.sleep(0.1)

        cap.release()

    def stop(self):
        self.running = False


# --- VOICE SYNTHESIZER ---
class VoiceSynthesizer:
    def __init__(self):
        self.model_path = None
        model_file = Config.PIPER_MODEL_DIR / f"{Config.PIPER_MODEL}.onnx"
        if model_file.exists():
            self.model_path = model_file
            print("   Piper TTS ready")
        else:
            print("   Piper model not found")

    def speak(self, text: str):
        if not self.model_path:
            return

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            process = subprocess.Popen(
                ["piper", "--model", str(self.model_path), "--output_file", temp_path],
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
        except:
            pass


# --- SYSTEM 3: THE WITNESS (Direct - No Filters) ---
class Witness:
    def __init__(self):
        print("\n" + "=" * 50)
        print("  WITNESS - Direct Mode")
        print("  Responds to EVERYTHING (no wake words)")
        print("=" * 50 + "\n")

        self.q = queue.Queue()
        self.visual_memory = "Nothing observed yet."

        self.memory = MemorySystem()
        self.voice = VoiceSynthesizer()

        print("   Loading Whisper...")
        self.ears = WhisperModel("base.en", device="cpu", compute_type="int8")
        print("   Whisper ready")

        self.eyes = VisualCortex(self.update_vision)
        self.eyes.start()

        self.history = []

        print("\n" + "-" * 50)
        print("Ready. I will respond to EVERYTHING you say.")
        print("Say 'goodbye' to exit.")
        print("-" * 50)

    def update_vision(self, desc):
        self.visual_memory = desc

    def listen(self):
        with sd.InputStream(samplerate=16000, channels=1, callback=self.callback):
            audio_data = []
            silence_start = None
            speaking = False
            print("\n   Listening...")

            while True:
                try:
                    data = self.q.get(timeout=0.5)
                    if np.abs(data).mean() > Config.SILENCE_THRESHOLD:
                        speaking = True
                        silence_start = None
                        audio_data.append(data)
                    elif speaking:
                        if silence_start is None:
                            silence_start = time.time()
                        audio_data.append(data)
                        if time.time() - silence_start > Config.SILENCE_DURATION:
                            break
                except queue.Empty:
                    pass

        if audio_data:
            return np.concatenate(audio_data, axis=0)
        return np.array([])

    def callback(self, indata, frames, time_info, status):
        self.q.put(indata.copy())

    def think_and_speak(self, user_text):
        """Direct path - no wake word checks."""
        print(f"\n   You: {user_text}")

        # Build context
        recalled = self.memory.recall(user_text)
        context = f"[VISUAL: {self.visual_memory}]\n[MEMORY: {recalled}]"

        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > 12:
            self.history = self.history[-12:]

        msgs = [
            {"role": "system", "content": f"{WITNESS_SOUL}\n{context}"}
        ] + self.history

        try:
            res = requests.post(
                f"{Config.OLLAMA_HOST}/api/chat",
                json={
                    "model": Config.CHAT_MODEL,
                    "messages": msgs,
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=60
            )

            if res.status_code == 200:
                reply = res.json()['message']['content']

                self.memory.remember(user_text, reply)
                self.history.append({"role": "assistant", "content": reply})

                print(f"\n   Witness: {reply}")
                self.voice.speak(reply)
            else:
                print(f"   [Error: {res.status_code}]")

        except Exception as e:
            print(f"   [Error: {e}]")

    def run(self):
        # Opening
        opening = self.think_direct("I am awakening. Give a brief greeting.")
        print(f"\n   Witness: {opening}")
        self.voice.speak(opening)

        while True:
            try:
                audio = self.listen()
                if len(audio) < 1000:
                    continue

                audio = audio.flatten().astype(np.float32) / 32768.0
                segments, _ = self.ears.transcribe(audio, beam_size=5)
                text = " ".join([s.text for s in segments]).strip()

                if not text:
                    continue

                # Check for exit
                if "goodbye" in text.lower():
                    farewell = self.think_direct("The user is leaving. Say a brief goodbye.")
                    print(f"\n   Witness: {farewell}")
                    self.voice.speak(farewell)
                    break

                # DIRECT PATH - respond to everything
                self.think_and_speak(text)

            except KeyboardInterrupt:
                print("\n\n[Session interrupted]")
                break

        self.eyes.stop()
        print("\n[Witness dormant - memories preserved]")

    def think_direct(self, prompt):
        """Generate response for system prompts."""
        recalled = self.memory.recall(prompt)
        context = f"[VISUAL: {self.visual_memory}]\n[MEMORY: {recalled}]"

        msgs = [
            {"role": "system", "content": f"{WITNESS_SOUL}\n{context}"},
            {"role": "user", "content": prompt}
        ]

        try:
            res = requests.post(
                f"{Config.OLLAMA_HOST}/api/chat",
                json={
                    "model": Config.CHAT_MODEL,
                    "messages": msgs,
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=60
            )

            if res.status_code == 200:
                reply = res.json()['message']['content']
                self.memory.remember(prompt, reply)
                return reply

        except Exception as e:
            return f"[Error: {e}]"

        return "[Error generating response]"


if __name__ == "__main__":
    witness = Witness()
    witness.run()
