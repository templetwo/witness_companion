#!/usr/bin/env python3
"""
WITNESS - Final Clean Slate
============================
Fresh memory, hallucination filtering, focused vision.
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

    # AUDIO SENSITIVITY
    SILENCE_THRESHOLD = 0.015  # Slight bump to ignore total silence
    SILENCE_DURATION = 1.2     # Wait longer for full sentences

    # HALLUCINATION FILTER (The "You" Fix)
    BAD_PHRASES = ["you", "thank you", "subtitles", "caption", "you.", "thank you.", "thanks.", "bye.", ""]

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


class Witness:
    def __init__(self):
        print("\n" + "=" * 50)
        print("  WITNESS - DEBUG MODE")
        print("  Extra logging for audio pipeline.")
        print("=" * 50 + "\n")

        self.q = queue.Queue()
        self.visual_memory = "I see nothing yet."

        # Initialize Memory
        print("   Initializing fresh memory...")
        self.client = chromadb.PersistentClient(path=Config.MEMORY_PATH)
        self.memory = self.client.get_or_create_collection(name="witness_logs")
        print(f"   Memory ready ({self.memory.count()} memories)")

        # Initialize Ears
        print("   Loading Whisper...")
        self.ears = WhisperModel("base.en", device="cpu", compute_type="int8")
        print("   Whisper ready")

        # Initialize Voice
        self.model_path = None
        model_file = Config.PIPER_MODEL_DIR / f"{Config.PIPER_MODEL}.onnx"
        if model_file.exists():
            self.model_path = model_file
            print("   Piper TTS ready")
        else:
            print("   Piper model not found")

        # Initialize Eyes
        threading.Thread(target=self.visual_loop, daemon=True).start()

        self.history = []

        print("\n" + "-" * 50)
        print("Ready. Filtering 'You' hallucinations.")
        print("Say 'goodbye' to exit.")
        print("-" * 50)

    def visual_loop(self):
        print("   Vision online (motion-activated)")
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)

        # Warm up camera
        for _ in range(10):
            cap.read()
            time.sleep(0.1)

        ret, prev = cap.read()
        if not ret:
            return
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Motion Detect
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            score = cv2.countNonZero(thresh)
            prev_gray = gray

            if score > Config.MOTION_THRESHOLD:
                time.sleep(0.5)
                ret, clean_frame = cap.read()
                if ret:
                    self.describe_view(clean_frame)
                time.sleep(4)  # Cooldown

            time.sleep(0.1)

    def describe_view(self, frame):
        _, buf = cv2.imencode('.jpg', frame)
        img = base64.b64encode(buf).decode("utf-8")
        try:
            # Focus on the person, not background
            prompt = "Describe the person in the foreground. What are they doing? 1 sentence."
            res = requests.post(
                f"{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": Config.VISION_MODEL,
                    "prompt": prompt,
                    "images": [img],
                    "stream": False
                },
                timeout=10
            )
            desc = res.json().get('response', '').strip()
            print(f"\n   [Saw]: {desc}")
            self.visual_memory = desc
        except:
            pass

    def speak(self, text):
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

    def listen_loop(self):
        print("\nDEBUG: Attempting to open audio stream...")
        try:
            with sd.InputStream(callback=self.callback, channels=1, samplerate=16000) as stream:
                print("DEBUG: Audio stream opened successfully.")
                # Opening greeting
                opening = self.think_direct("I am awakening with fresh memory. Give a brief greeting.")
                print(f"\n   Witness: {opening}")
                self.speak(opening)

                print("\n   Listening...")

                while True:
                    try:
                        # 1. Record Audio Chunk
                        print("DEBUG: Waiting for audio chunk...")
                        audio = self.record_chunk()
                        if len(audio) < 5000:
                            print("DEBUG: Chunk too short, ignoring.")
                            continue  # Ignore short blips
                        
                        print(f"DEBUG: Recorded chunk of size {len(audio)}.")

                        # 2. Transcribe
                        # Audio is already float32 from sounddevice, no need to normalize
                        audio = audio.flatten().astype(np.float32)
                        print("DEBUG: Transcribing audio...")
                        segs, _ = self.ears.transcribe(audio, beam_size=5, language="en")
                        text = " ".join([s.text for s in segs]).strip()
                        print(f"DEBUG: Transcription result: '{text}'")


                        if not text:
                            continue

                        # 3. THE HALLUCINATION FILTER
                        if text.lower().strip() in Config.BAD_PHRASES:
                            print(f"   (Filtered: '{text}')")
                            continue

                        # 4. Check for exit
                        if "goodbye" in text.lower():
                            farewell = self.think_direct("The user is leaving. Say a brief goodbye.")
                            print(f"\n   Witness: {farewell}")
                            self.speak(farewell)
                            break

                        # 5. Respond
                        self.respond(text)

                    except KeyboardInterrupt:
                        print("\n\n[Session interrupted]")
                        break
        except Exception as e:
            print(f"\nFATAL ERROR in listen_loop: {e}")
            print("This might be a microphone access issue. Please check your system permissions.")


        print("\n[Witness dormant - memories preserved]")

    def record_chunk(self):
        audio = []
        speaking = False
        silence_start = None
        
        print("DEBUG: record_chunk started.")

        while True:
            try:
                data = self.q.get(timeout=0.3)
                vol = np.abs(data).mean()
                
                # DEBUG: Print volume level
                if not speaking:
                    print(f"DEBUG: Mic volume: {vol:.4f} (Threshold: {Config.SILENCE_THRESHOLD})")

                if vol > Config.SILENCE_THRESHOLD:
                    if not speaking:
                        print("DEBUG: Speaking detected.")
                    speaking = True
                    silence_start = None
                    audio.append(data)
                elif speaking:
                    if silence_start is None:
                        silence_start = time.time()
                    audio.append(data)
                    if time.time() - silence_start > Config.SILENCE_DURATION:
                        print("DEBUG: Silence detected, ending recording.")
                        break
            except queue.Empty:
                if speaking:
                    print("DEBUG: Queue empty, ending recording.")
                    break
                continue

        if not audio:
            return np.array([])
        return np.concatenate(audio, axis=0)

    def callback(self, indata, frames, time_info, status):
        # This is called from a separate thread, keep it lean.
        # print(f"DEBUG: callback received {frames} frames.")
        if status:
            print(f"DEBUG: Audio callback status: {status}")
        self.q.put(indata.copy())

    def respond(self, text):
        print(f"\n   You: {text}")

        # Recall
        mem = "No memories yet."
        if self.memory.count() > 0:
            results = self.memory.query(query_texts=[text], n_results=2)
            if results['documents'][0]:
                mem = " | ".join(results['documents'][0])

        # Build context
        context = f"[VISUAL: {self.visual_memory}]\n[MEMORY: {mem}]"

        # Add to history
        self.history.append({"role": "user", "content": text})
        if len(self.history) > 12:
            self.history = self.history[-12:]

        # Generate
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

                # Save to history and memory
                self.history.append({"role": "assistant", "content": reply})
                self.memory.add(
                    documents=[f"User: {text} | Witness: {reply}"],
                    metadatas=[{"timestamp": datetime.datetime.now().isoformat()}],
                    ids=[f"mem_{int(time.time() * 1000)}"]
                )

                print(f"\n   Witness: {reply}")
                self.speak(reply)
            else:
                print(f"   [Error: {res.status_code}]")

        except Exception as e:
            print(f"   [Error: {e}]")

    def think_direct(self, prompt):
        """Generate response for system prompts."""
        mem = "No memories yet."
        if self.memory.count() > 0:
            results = a = self.memory.query(query_texts=[prompt], n_results=1)
            if results['documents'][0]:
                mem = results['documents'][0][0]

        context = f"[VISUAL: {self.visual_memory}]\n[MEMORY: {mem}]"

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
                self.memory.add(
                    documents=[f"System: {prompt} | Witness: {reply}"],
                    metadatas=[{"timestamp": datetime.datetime.now().isoformat()}],
                    ids=[f"mem_{int(time.time() * 1000)}"]
                )
                return reply

        except Exception as e:
            return f"[Error: {e}]"

        return "[Error generating response]"


if __name__ == "__main__":
    witness = Witness()
    witness.listen_loop()