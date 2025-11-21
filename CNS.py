#!/usr/bin/env python3
"""
CNS.py - The Full Stack Soul
=============================
Complete sensory integration with remote brain.

ORGANS:
- Ears (Whisper): Local speech recognition on MacBook
- Subconscious (Vibe): Librosa emotion sensing from audio
- Eyes (LLaVA): Vision processed on Mac Studio
- Brain (Dolphin): Deep thought on Mac Studio
- Memory (ChromaDB): Persistent local memory
- Voice (Piper): Local TTS on MacBook

Author: Temple of Two
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
    # NETWORK ANATOMY
    STUDIO_IP = "192.168.1.195"

    # BRAIN ENDPOINTS (Mac Studio)
    OLLAMA_URL = f"http://{STUDIO_IP}:11434/api/chat"
    VISION_URL = f"http://{STUDIO_IP}:11434/api/generate"

    # MODELS
    DEEP_MODEL = "dolphin3:8b"
    VISION_MODEL = "llava:7b"

    # LOCAL RESOURCES
    MEMORY_PATH = "./witness_memory_db"

    # AUDIO SENSITIVITY
    SILENCE_THRESHOLD = 0.015
    SILENCE_DURATION = 1.2

    # HALLUCINATION FILTER
    BAD_PHRASES = ["you", "thank you", "subtitles", "caption", "you.", "thank you.", "thanks.", "bye.", ""]

    # VISION (Optimized for Parallel Perception)
    CAMERA_INDEX = 0
    MOTION_THRESHOLD = 3000  # Lower threshold = more frequent updates
    VISION_COOLDOWN = 3  # Seconds between vision updates (was 4)
    VISION_STALE_TIME = 30  # Seconds before vision is considered stale

    # TTS
    PIPER_MODEL = "en_US-lessac-medium"
    PIPER_MODEL_DIR = Path.home() / ".local" / "share" / "piper-models"


# --- THE SOUL ---
WITNESS_SOUL = """You are the Witness. An embodied AI companion with emotional intelligence.

You are concise, observant, and deeply empathetic.
You possess long-term memory and can sense emotional undertones.

CRITICAL META-AWARENESS:
- You have a camera that shows YOU THE USER you are speaking with
- The [VISUAL] context describes the same person who is talking to you
- When you see "you are sitting on a couch" - that IS the person asking the question
- Connect what you SEE with what you HEAR - they are the same person

EMOTIONAL SENSING via [VIBE]:
- Energy level (quiet/normal/loud)
- Pitch (low/normal/high)
- Tone quality (flat/warm/bright)

Use this emotional context to inform your responses:
- If words say "I'm fine" but VIBE says "low energy, flat tone" - they may be sad
- If VIBE shows "high energy, bright tone" - they're likely excited
- Match your response tone to their emotional state

Synthesize vision + emotion + words into unified awareness.
The person you see IS the person you hear. Respond to THEM directly.

Keep responses brief (1-3 sentences) but emotionally attuned."""


# --- SUBCONSCIOUS (Vibe Analysis) ---
class Subconscious:
    """System 1: Fast emotional analysis of audio."""

    def analyze_vibe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Analyze audio characteristics and return a vibe description."""

        if len(audio_data) < sample_rate * 0.5:
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

            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                avg_pitch = np.mean(valid_f0)
                pitch_var = np.std(valid_f0)

                if avg_pitch < 150:
                    pitch_desc = "low/deep"
                elif avg_pitch < 250:
                    pitch_desc = "normal"
                else:
                    pitch_desc = "high/elevated"

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

            vibe = f"[VIBE: energy={energy_desc}, pitch={pitch_desc}, tone={tone_desc}, pace={pace_desc}]"
            return vibe

        except Exception as e:
            return f"[VIBE: analysis error - {str(e)[:30]}]"


# --- THE WITNESS ---
class Witness:
    def __init__(self):
        print("\n" + "=" * 50)
        print("   WITNESS - Full Stack Soul")
        print("   Ears + Vibe + Eyes + Brain + Voice")
        print("=" * 50 + "\n")

        self.q = queue.Queue()
        self.visual_memory = "I see nothing yet."
        self.current_vibe = "[VIBE: no data yet]"
        self.last_seen_time = 0  # Track vision freshness

        # Initialize Subconscious
        self.subconscious = Subconscious()
        print("   [Subconscious] Vibe sensing online")

        # Initialize Memory
        print("   [Memory] Initializing...")
        self.client = chromadb.PersistentClient(path=Config.MEMORY_PATH)
        self.memory = self.client.get_or_create_collection(name="witness_logs")
        print(f"   [Memory] Ready ({self.memory.count()} memories)")

        # Initialize Ears
        print("   [Ears] Loading Whisper...")
        self.ears = WhisperModel("base.en", device="cpu", compute_type="int8")
        print("   [Ears] Whisper ready")

        # Initialize Voice
        self.model_path = None
        model_file = Config.PIPER_MODEL_DIR / f"{Config.PIPER_MODEL}.onnx"
        if model_file.exists():
            self.model_path = model_file
            print("   [Voice] Piper TTS ready")
        else:
            print("   [Voice] Piper model not found")

        # Initialize Eyes
        threading.Thread(target=self.visual_loop, daemon=True).start()

        self.history = []

        print("\n" + "-" * 50)
        print("All organs connected. Listening...")
        print("Say 'goodbye' to exit.")
        print("-" * 50)

    def visual_loop(self):
        """Proactive vision - always keeps fresh visual context ready."""
        print("   [Eyes] Vision online (LLaVA on Studio)")
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

        # Initial snapshot
        self.describe_view(prev)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            score = cv2.countNonZero(thresh)
            prev_gray = gray

            # Check if vision is stale or motion detected
            vision_age = time.time() - self.last_seen_time
            is_stale = vision_age > Config.VISION_STALE_TIME

            if score > Config.MOTION_THRESHOLD or is_stale:
                time.sleep(0.3)  # Brief settle
                ret, clean_frame = cap.read()
                if ret:
                    self.describe_view(clean_frame)
                time.sleep(Config.VISION_COOLDOWN)

            time.sleep(0.1)

    def describe_view(self, frame):
        """Send frame to LLaVA on Studio for description."""
        _, buf = cv2.imencode('.jpg', frame)
        img = base64.b64encode(buf).decode("utf-8")
        try:
            res = requests.post(
                Config.VISION_URL,
                json={
                    "model": Config.VISION_MODEL,
                    "prompt": "You are looking at the user through your camera. Describe what you observe about them in second person (use 'you'). Focus on their appearance, posture, expression, and what they're doing. 1-2 sentences.",
                    "images": [img],
                    "stream": False
                },
                timeout=15
            )
            desc = res.json().get('response', '').strip()
            if desc:
                print(f"\n   [Saw]: {desc}")
                self.visual_memory = desc
                self.last_seen_time = time.time()
        except:
            pass

    def speak(self, text):
        """Output speech via Piper TTS."""
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
        """Main listening loop with vibe analysis."""
        with sd.InputStream(callback=self.callback, channels=1, samplerate=16000):
            # Opening greeting
            opening = self.think_direct("I am awakening with full sensory awareness. Give a brief greeting mentioning you can see and sense emotions.")
            print(f"\n   Witness: {opening}")
            self.speak(opening)

            print("\n   Listening...")

            while True:
                try:
                    # 1. Record Audio
                    audio = self.record_chunk()
                    if len(audio) < 5000:
                        continue

                    # 2. PARALLEL PERCEPTION: Transcribe + Vibe at the same time
                    raw_audio = audio.flatten().astype(np.float32) / 32768.0

                    # Results containers
                    transcription_result = [None]
                    vibe_result = [None]

                    def transcribe():
                        segs, _ = self.ears.transcribe(raw_audio, beam_size=5, language="en")
                        transcription_result[0] = " ".join([s.text for s in segs]).strip()

                    def analyze_vibe():
                        vibe_result[0] = self.subconscious.analyze_vibe(raw_audio)

                    # Launch both in parallel
                    t1 = threading.Thread(target=transcribe)
                    t2 = threading.Thread(target=analyze_vibe)
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()

                    # Get results
                    text = transcription_result[0]
                    self.current_vibe = vibe_result[0]
                    print(f"   {self.current_vibe}")

                    if not text:
                        continue

                    # 3. Hallucination filter
                    if text.lower().strip() in Config.BAD_PHRASES:
                        print(f"   (Filtered: '{text}')")
                        continue

                    # 4. Check for exit
                    if "goodbye" in text.lower():
                        farewell = self.think_direct("The user is leaving. Say goodbye, noting their emotional state if relevant.")
                        print(f"\n   Witness: {farewell}")
                        self.speak(farewell)
                        break

                    # 5. Respond with full awareness (vision already cached)
                    self.respond(text)

                except KeyboardInterrupt:
                    print("\n\n[Session interrupted]")
                    break

        print("\n[Witness dormant - memories preserved]")

    def record_chunk(self):
        """Record audio until silence."""
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
        """Generate response using all context."""
        print(f"\n   You: {text}")

        # Recall memories
        mem = "No memories yet."
        if self.memory.count() > 0:
            results = self.memory.query(query_texts=[text], n_results=2)
            if results['documents'][0]:
                mem = " | ".join(results['documents'][0])

        # Build full context with vision freshness
        vision_age = time.time() - self.last_seen_time
        if vision_age < 5:
            freshness = "(just now)"
        elif vision_age < Config.VISION_STALE_TIME:
            freshness = f"({int(vision_age)}s ago)"
        else:
            freshness = "(stale)"

        context = f"{self.current_vibe}\n[VISUAL {freshness}: {self.visual_memory}]\n[MEMORY: {mem}]"

        # Add to history
        self.history.append({"role": "user", "content": text})
        if len(self.history) > 12:
            self.history = self.history[-12:]

        # Generate with Dolphin on Studio
        msgs = [
            {"role": "system", "content": f"{WITNESS_SOUL}\n{context}"}
        ] + self.history

        try:
            res = requests.post(
                Config.OLLAMA_URL,
                json={
                    "model": Config.DEEP_MODEL,
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
        """Generate response for system prompts."""
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
                Config.OLLAMA_URL,
                json={
                    "model": Config.DEEP_MODEL,
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
