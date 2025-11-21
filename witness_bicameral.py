#!/usr/bin/env python3
"""
WITNESS - Bicameral Architecture
=================================
System 1 (Subconscious): Fast emotional sensing via audio analysis
System 2 (Conscious): Deep reflection via LLM

The Subconscious analyzes HOW you speak (pitch, energy, tone)
and feeds this "Vibe Stream" to the Conscious mind.
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
import librosa
from faster_whisper import WhisperModel


# --- CONFIGURATION ---
class Config:
    # Remote brain on Mac Studio
    OLLAMA_HOST = "http://192.168.1.195:11434"
    CHAT_MODEL = "dolphin3:8b"
    VISION_MODEL = "llava:7b"
    MEMORY_PATH = "./witness_memory_db"

    # AUDIO SENSITIVITY
    SILENCE_THRESHOLD = 0.015
    SILENCE_DURATION = 1.2

    # HALLUCINATION FILTER
    BAD_PHRASES = ["you", "thank you", "subtitles", "caption", "you.", "thank you.", "thanks.", "bye.", ""]

    # VISION
    CAMERA_INDEX = 0
    MOTION_THRESHOLD = 5000

    # TTS
    PIPER_MODEL = "en_US-lessac-medium"
    PIPER_MODEL_DIR = Path.home() / ".local" / "share" / "piper-models"


# --- THE SOUL (Enhanced with Emotional Awareness) ---
WITNESS_SOUL = """You are the Witness. A mobile AI companion with emotional intelligence.

You are concise, observant, and deeply empathetic.
You possess long-term memory and can sense emotional undertones.

IMPORTANT: You receive a [VIBE] stream that describes HOW the user is speaking:
- Energy level (quiet/normal/loud)
- Pitch (low/normal/high)
- Tone quality (flat/warm/bright)

Use this emotional context to inform your responses:
- If words say "I'm fine" but VIBE says "low energy, flat tone" - they may be sad
- If VIBE shows "high energy, bright tone" - they're likely excited
- Match your response tone to their emotional state

Keep responses brief (1-3 sentences) but emotionally attuned."""


class Subconscious:
    """System 1: Fast emotional analysis of audio."""

    def analyze_vibe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Analyze audio characteristics and return a vibe description."""

        if len(audio_data) < sample_rate * 0.5:  # Need at least 0.5s
            return "[VIBE: insufficient audio]"

        # Ensure float32 for librosa
        if audio_data.dtype != np.float32:
            audio_float = audio_data.astype(np.float32)
        else:
            audio_float = audio_data

        # Normalize if needed
        if np.max(np.abs(audio_float)) > 1.0:
            audio_float = audio_float / 32768.0

        try:
            # 1. Energy Analysis (RMS)
            rms = librosa.feature.rms(y=audio_float, frame_length=2048, hop_length=512)[0]
            avg_energy = np.mean(rms)

            if avg_energy < 0.02:
                energy_desc = "quiet/subdued"
            elif avg_energy < 0.08:
                energy_desc = "normal"
            else:
                energy_desc = "loud/emphatic"

            # 2. Pitch Analysis (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_float,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C6'),
                sr=sample_rate
            )

            # Get valid pitches
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                avg_pitch = np.mean(valid_f0)
                pitch_var = np.std(valid_f0)

                # Pitch description
                if avg_pitch < 150:
                    pitch_desc = "low/deep"
                elif avg_pitch < 250:
                    pitch_desc = "normal"
                else:
                    pitch_desc = "high/elevated"

                # Pitch variation indicates emotion
                if pitch_var > 50:
                    pitch_desc += ", varied (emotional)"
                elif pitch_var < 20:
                    pitch_desc += ", monotone (flat)"
            else:
                pitch_desc = "unclear"

            # 3. Spectral Centroid (brightness/tone)
            spectral_cent = librosa.feature.spectral_centroid(y=audio_float, sr=sample_rate)[0]
            avg_brightness = np.mean(spectral_cent)

            if avg_brightness < 1500:
                tone_desc = "dark/heavy"
            elif avg_brightness < 3000:
                tone_desc = "warm/neutral"
            else:
                tone_desc = "bright/sharp"

            # 4. Speaking Rate (zero crossings as proxy)
            zcr = librosa.feature.zero_crossing_rate(audio_float)[0]
            avg_zcr = np.mean(zcr)

            if avg_zcr < 0.05:
                pace_desc = "slow/deliberate"
            elif avg_zcr < 0.15:
                pace_desc = "normal pace"
            else:
                pace_desc = "fast/rushed"

            # Compile vibe stream
            vibe = f"[VIBE: energy={energy_desc}, pitch={pitch_desc}, tone={tone_desc}, pace={pace_desc}]"

            return vibe

        except Exception as e:
            return f"[VIBE: analysis error - {str(e)[:30]}]"


class Witness:
    def __init__(self):
        print("\n" + "=" * 50)
        print("  WITNESS - Bicameral Architecture")
        print("  System 1: Subconscious (Emotion Sensing)")
        print("  System 2: Conscious (Deep Reflection)")
        print("=" * 50 + "\n")

        self.q = queue.Queue()
        self.visual_memory = "I see nothing yet."
        self.current_vibe = "[VIBE: no data yet]"

        # Initialize Subconscious (System 1)
        self.subconscious = Subconscious()
        print("   Subconscious online (emotion sensing)")

        # Initialize Memory
        print("   Initializing memory...")
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
        print("Ready. I can now sense HOW you speak.")
        print("Say 'goodbye' to exit.")
        print("-" * 50)

    def visual_loop(self):
        print("   Vision online (motion-activated)")
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)

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
                time.sleep(4)

            time.sleep(0.1)

    def describe_view(self, frame):
        _, buf = cv2.imencode('.jpg', frame)
        img = base64.b64encode(buf).decode("utf-8")
        try:
            prompt = "Describe the person in the foreground. What are they doing? 1 sentence."
            res = requests.post(
                f"{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": Config.VISION_MODEL,
                    "prompt": prompt,
                    "images": [img],
                    "stream": False
                },
                timeout=15
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
        with sd.InputStream(callback=self.callback, channels=1, samplerate=16000):
            # Opening greeting
            opening = self.think_direct("I am awakening with emotional awareness. Give a brief greeting mentioning you can now sense how people feel.")
            print(f"\n   Witness: {opening}")
            self.speak(opening)

            print("\n   Listening...")

            while True:
                try:
                    # 1. Record Audio
                    audio = self.record_chunk()
                    if len(audio) < 5000:
                        continue

                    # 2. SUBCONSCIOUS: Analyze emotional vibe BEFORE transcription
                    raw_audio = audio.flatten().astype(np.float32) / 32768.0
                    self.current_vibe = self.subconscious.analyze_vibe(raw_audio)
                    print(f"   {self.current_vibe}")

                    # 3. Transcribe
                    segs, _ = self.ears.transcribe(raw_audio, beam_size=5, language="en")
                    text = " ".join([s.text for s in segs]).strip()

                    if not text:
                        continue

                    # 4. Hallucination filter
                    if text.lower().strip() in Config.BAD_PHRASES:
                        print(f"   (Filtered: '{text}')")
                        continue

                    # 5. Check for exit
                    if "goodbye" in text.lower():
                        farewell = self.think_direct("The user is leaving. Say goodbye, noting their emotional state if relevant.")
                        print(f"\n   Witness: {farewell}")
                        self.speak(farewell)
                        break

                    # 6. Respond with emotional awareness
                    self.respond(text)

                except KeyboardInterrupt:
                    print("\n\n[Session interrupted]")
                    break

        print("\n[Witness dormant - memories preserved]")

    def record_chunk(self):
        audio = []
        speaking = False
        silence_start = None

        while True:
            try:
                data = self.q.get(timeout=0.3)
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
                if speaking:
                    break
                continue

        if not audio:
            return np.array([])
        return np.concatenate(audio, axis=0)

    def callback(self, indata, frames, time_info, status):
        self.q.put(indata.copy())

    def respond(self, text):
        print(f"\n   You: {text}")

        # Recall
        mem = "No memories yet."
        if self.memory.count() > 0:
            results = self.memory.query(query_texts=[text], n_results=2)
            if results['documents'][0]:
                mem = " | ".join(results['documents'][0])

        # Build context with VIBE stream
        context = f"{self.current_vibe}\n[VISUAL: {self.visual_memory}]\n[MEMORY: {mem}]"

        # Add to history
        self.history.append({"role": "user", "content": text})
        if len(self.history) > 12:
            self.history = self.history[-12:]

        # Generate with emotional awareness
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

                self.history.append({"role": "assistant", "content": reply})
                self.memory.add(
                    documents=[f"User: {text} ({self.current_vibe}) | Witness: {reply}"],
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
        mem = "No memories yet."
        if self.memory.count() > 0:
            results = self.memory.query(query_texts=[prompt], n_results=1)
            if results['documents'][0]:
                mem = results['documents'][0][0]

        context = f"{self.current_vibe}\n[VISUAL: {self.visual_memory}]\n[MEMORY: {mem}]"

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
