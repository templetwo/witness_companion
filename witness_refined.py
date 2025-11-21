#!/usr/bin/env python3
"""
WITNESS - Refined System with Selective Attention
==================================================
Optimized for efficiency with wake word filtering and motion-triggered vision.

Fixes:
- Wake word filter: Ignores TV/background noise
- Motion detection: Only analyzes when movement detected
- Threading optimized for reduced lag

Usage:
    python witness_refined.py
"""

# Fix parallelism warnings - MUST be before any imports
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

import numpy as np
import sounddevice as sd
import requests
import cv2
import chromadb
from faster_whisper import WhisperModel


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    OLLAMA_HOST = "http://localhost:11434"
    CHAT_MODEL = "llama3:8b"
    VISION_MODEL = "moondream"
    MEMORY_PATH = "./witness_memory_db"

    # Audio Settings
    SILENCE_THRESHOLD = 0.01  # Based on mean amplitude (user mean ~0.004)
    SILENCE_DURATION = 1.0
    WAKE_WORDS = ["witness", "hey", "companion", "look", "listen", "what do you see"]

    # Vision Settings
    CAMERA_INDEX = 0
    MOTION_THRESHOLD = 5000  # Sensitivity for motion detection
    VISION_COOLDOWN = 5  # Seconds between vision triggers

    # Piper TTS
    PIPER_MODEL = "en_US-lessac-medium"
    PIPER_MODEL_DIR = Path.home() / ".local" / "share" / "piper-models"


# =============================================================================
# THE SOUL
# =============================================================================

WITNESS_SOUL = """You are the Witness. A mobile AI companion.
You are concise, observant, and friendly.
You possess long-term memory.
If the user mentions visual things, refer to the [VISUAL CONTEXT].
Keep responses brief (1-3 sentences)."""


# =============================================================================
# MEMORY SYSTEM
# =============================================================================

class MemorySystem:
    def __init__(self):
        print("   Loading memory...")
        self.client = chromadb.PersistentClient(path=Config.MEMORY_PATH)
        self.collection = self.client.get_or_create_collection(name="witness_logs")
        count = self.collection.count()
        print(f"   Memory ready ({count} memories)")

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


# =============================================================================
# VISUAL CORTEX - Motion Triggered
# =============================================================================

class VisualCortex(threading.Thread):
    """Motion-triggered vision system - only analyzes when movement detected."""

    def __init__(self, context_callback):
        super().__init__()
        self.context_callback = context_callback
        self.running = True
        self.daemon = True
        self.last_description = "Nothing observed yet."
        self.script_dir = Path(__file__).parent

    def get_description(self, frame) -> str:
        """Send frame to vision model."""
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        try:
            response = requests.post(
                f"{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": Config.VISION_MODEL,
                    "prompt": "What objects and people do you see? Be specific and accurate. 1-2 sentences.",
                    "images": [img_base64],
                    "stream": False
                },
                timeout=15
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip()
        except:
            pass
        return None

    def run(self):
        print("   Vision online (motion-activated)")

        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        if not cap.isOpened():
            print("   Camera not available")
            return

        # Warm up camera
        for _ in range(10):
            cap.read()
            time.sleep(0.1)

        # Get first frame for comparison
        ret, prev_frame = cap.read()
        if not ret:
            return
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Motion detection (very cheap on CPU)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = cv2.countNonZero(thresh)
            prev_gray = gray

            # Only trigger AI vision if significant motion detected
            if motion_score > Config.MOTION_THRESHOLD:
                # Wait for motion to settle, get clear shot
                time.sleep(0.5)
                ret, clean_frame = cap.read()
                if ret:
                    desc = self.get_description(clean_frame)
                    if desc:
                        print(f"\n   [Motion]: {desc}")
                        self.context_callback(desc)
                        self.last_description = desc

                        # Save debug frame
                        debug_path = self.script_dir / "debug_frame.jpg"
                        cv2.imwrite(str(debug_path), clean_frame)

                        # Cooldown to prevent spam
                        time.sleep(Config.VISION_COOLDOWN)

            time.sleep(0.1)

        cap.release()

    def stop(self):
        self.running = False


# =============================================================================
# VOICE SYNTHESIZER
# =============================================================================

class VoiceSynthesizer:
    def __init__(self):
        self.model_path = None
        self._setup()

    def _setup(self):
        model_dir = Config.PIPER_MODEL_DIR
        model_file = model_dir / f"{Config.PIPER_MODEL}.onnx"

        if model_file.exists():
            self.model_path = model_file
            print("   Piper TTS ready")
        else:
            print("   Piper model not found - run witness_complete.py first to download")

    def speak(self, text: str):
        print(f"\n   Witness: {text}")

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


# =============================================================================
# THE WITNESS - Refined
# =============================================================================

class Witness:
    def __init__(self):
        print("\n" + "="*50)
        print("  WITNESS - Refined System")
        print("  'Selective Attention' Mode")
        print("="*50 + "\n")

        self.audio_queue = queue.Queue()
        self.visual_context = "Nothing observed yet."

        # Initialize subsystems
        self.memory = MemorySystem()
        self.voice = VoiceSynthesizer()

        # Whisper - use English-only model for speed
        print("   Loading Whisper...")
        self.ears = WhisperModel("base.en", device="cpu", compute_type="int8")
        print("   Whisper ready")

        # Start vision
        self.eyes = VisualCortex(self.update_visual_context)
        self.eyes.start()

        self.history = []

        print("\n" + "-"*50)
        print(f"Wake words: {', '.join(Config.WAKE_WORDS)}")
        print("Vision triggers on motion only.")
        print("Say 'goodbye' to exit.")
        print("-"*50)

    def update_visual_context(self, description: str):
        self.visual_context = description

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(indata.copy())

    def listen(self) -> np.ndarray:
        """Always-on listening with silence detection."""
        print("\n   Listening...", end="", flush=True)

        # Clear old audio from queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break

        audio_buffer = []
        silence_samples = 0
        speech_detected = False
        samples_per_check = int(16000 * 0.1)
        silence_needed = int(Config.SILENCE_DURATION * 16000 / samples_per_check)

        with sd.InputStream(
            samplerate=16000,
            channels=1,
            callback=self.audio_callback,
            blocksize=samples_per_check
        ):
            while True:
                try:
                    chunk = self.audio_queue.get(timeout=0.5)
                    amplitude = np.abs(chunk).mean()

                    if amplitude > Config.SILENCE_THRESHOLD:
                        if not speech_detected:
                            print(" Speech detected!")
                        speech_detected = True
                        silence_samples = 0
                        audio_buffer.append(chunk)
                    elif speech_detected:
                        silence_samples += 1
                        audio_buffer.append(chunk)
                        if silence_samples >= silence_needed:
                            break
                except queue.Empty:
                    if speech_detected:
                        silence_samples += 1
                        if silence_samples >= silence_needed:
                            break

        if audio_buffer:
            return np.concatenate(audio_buffer, axis=0)
        return np.array([])

    def is_addressed(self, text: str) -> bool:
        """Check if user is talking to us (wake word filter)."""
        text_lower = text.lower()

        # Check wake words
        for word in Config.WAKE_WORDS:
            if word in text_lower:
                return True

        # Also respond to visual questions
        if "what" in text_lower and any(w in text_lower for w in ["see", "holding", "looking"]):
            return True

        return False

    def think_and_respond(self, user_text: str):
        """Process input and generate response."""

        # Check if addressed
        if not self.is_addressed(user_text):
            print(f"   (Ignoring: '{user_text[:50]}...')")
            return

        print(f"\n   You: {user_text}")

        # Build context
        memories = self.memory.recall(user_text)
        context = f"[VISUAL: {self.visual_context}]\n[MEMORY: {memories}]"

        # Add to history
        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > 12:
            self.history = self.history[-12:]

        # Generate response
        messages = [
            {"role": "system", "content": f"{WITNESS_SOUL}\n{context}"}
        ] + self.history

        try:
            response = requests.post(
                f"{Config.OLLAMA_HOST}/api/chat",
                json={
                    "model": Config.CHAT_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=60
            )

            if response.status_code == 200:
                reply = response.json()['message']['content']

                # Save
                self.memory.remember(user_text, reply)
                self.history.append({"role": "assistant", "content": reply})

                # Speak
                self.voice.speak(reply)
            else:
                print(f"   [Error: {response.status_code}]")

        except Exception as e:
            print(f"   [Error: {e}]")

    def run(self):
        # Initial greeting
        print("\n   Witness awakening...")
        opening = self.think_direct("I am awakening. Give a brief greeting.")
        self.voice.speak(opening)

        while True:
            try:
                audio = self.listen()

                if len(audio) == 0:
                    continue

                # Transcribe
                audio_float = audio.flatten().astype(np.float32) / 32768.0
                segments, _ = self.ears.transcribe(audio_float, beam_size=5)
                text = " ".join([s.text for s in segments]).strip()

                if not text:
                    continue

                # Process with wake word filter
                self.think_and_respond(text)

                # Check for exit
                if "goodbye" in text.lower():
                    farewell = self.think_direct("The user is leaving. Say a brief goodbye.")
                    self.voice.speak(farewell)
                    break

            except KeyboardInterrupt:
                print("\n\n[Session interrupted]")
                break

        self.eyes.stop()
        print("\n[Witness dormant - memories preserved]")

    def think_direct(self, user_text: str) -> str:
        """Generate response without wake word check (for system prompts)."""
        # Build context
        memories = self.memory.recall(user_text)
        context = f"[VISUAL: {self.visual_context}]\n[MEMORY: {memories}]"

        # Add to history
        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > 12:
            self.history = self.history[-12:]

        # Generate response
        messages = [
            {"role": "system", "content": f"{WITNESS_SOUL}\n{context}"}
        ] + self.history

        try:
            response = requests.post(
                f"{Config.OLLAMA_HOST}/api/chat",
                json={
                    "model": Config.CHAT_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=60
            )

            if response.status_code == 200:
                reply = response.json()['message']['content']
                self.memory.remember(user_text, reply)
                self.history.append({"role": "assistant", "content": reply})
                return reply
            else:
                return "[Error generating response]"

        except Exception as e:
            return f"[Error: {e}]"


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    witness = Witness()
    witness.run()
