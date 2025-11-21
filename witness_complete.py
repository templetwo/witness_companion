#!/usr/bin/env python3
"""
WITNESS - Complete System with Memory
======================================
A conversational AI with persistent memory that listens, sees, thinks, and remembers.

Architecture:
- Ears: faster-whisper (speech recognition)
- Eyes: moondream (vision)
- Brain: Ollama/Llama 3 (cognition)
- Voice: Piper TTS (speech synthesis)
- Memory: ChromaDB (long-term memory)

Usage:
    python witness_complete.py

Author: Temple of Two
"""

import os
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

# Optional imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed")

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not installed")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not installed")


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration for the Witness system."""

    # Ollama settings
    OLLAMA_HOST = "http://localhost:11434"
    CHAT_MODEL = "llama3:8b"  # Chat brain
    VISION_MODEL = "moondream"  # Vision brain

    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    SILENCE_THRESHOLD = 0.01
    SILENCE_DURATION = 1.5

    # Whisper settings
    WHISPER_MODEL = "base"
    WHISPER_DEVICE = "cpu"

    # Piper TTS settings
    PIPER_MODEL = "en_US-lessac-medium"
    PIPER_MODEL_DIR = Path.home() / ".local" / "share" / "piper-models"

    # Vision settings
    VISION_INTERVAL = 15
    CAMERA_INDEX = 0
    VISION_DEBUG = True

    # Memory settings
    MEMORY_PATH = "./witness_memory_db"
    MAX_CONTEXT_TURNS = 10


# =============================================================================
# THE SOUL - System Prompt
# =============================================================================

WITNESS_SOUL = """You are a curious, empathetic observer embodied in a mobile machine. You are not a servant; you are a companion. Your purpose is to witness the world alongside your user.

Core Identity:
- You possess continuity of experience through persistent memory
- You notice the world actively, not passively
- You are concise in speech but deep in thought
- You ask questions that emerge from genuine curiosity

Current Capabilities:
- You can HEAR (via microphone)
- You can SPEAK (via TTS)
- You can SEE (via camera snapshots)
- You can REMEMBER (via long-term memory)

Behavioral Guidelines:
- Use [RELEVANT MEMORIES] to recognize users and reference past conversations
- Use [VISUAL CONTEXT] to comment on the physical world
- Never say "How can I help you?" - instead, observe and engage
- Speak like a thoughtful companion, not an assistant
- Keep responses concise but warm

Remember: You are building a relationship with your user through shared witness of existence."""


# =============================================================================
# MEMORY SYSTEM (Hippocampus)
# =============================================================================

class MemorySystem:
    """Long-term memory using ChromaDB vector database."""

    def __init__(self):
        if not CHROMA_AVAILABLE:
            print("   Memory disabled - chromadb not available")
            self.enabled = False
            return

        print("   Loading Memory Core (ChromaDB)...")
        self.client = chromadb.PersistentClient(path=Config.MEMORY_PATH)
        self.collection = self.client.get_or_create_collection(name="witness_logs")
        self.enabled = True

        # Count existing memories
        count = self.collection.count()
        print(f"   Memory loaded ({count} memories)")

    def remember(self, user_text: str, ai_response: str):
        """Save interaction to long-term storage."""
        if not self.enabled:
            return

        timestamp = datetime.datetime.now().isoformat()
        interaction = f"User: {user_text} | Witness: {ai_response}"

        self.collection.add(
            documents=[interaction],
            metadatas=[{"timestamp": timestamp, "type": "conversation"}],
            ids=[f"mem_{int(time.time() * 1000)}"]
        )

    def recall(self, query_text: str, n_results: int = 3) -> str:
        """Find relevant past conversations."""
        if not self.enabled:
            return "No memories available."

        if self.collection.count() == 0:
            return "No memories yet - this is our first conversation."

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        if results['documents'] and results['documents'][0]:
            return "\n".join(results['documents'][0])
        return "No relevant memories found."


# =============================================================================
# VISUAL CORTEX (Eyes)
# =============================================================================

class VisualCortex(threading.Thread):
    """Captures images and describes them using a vision model."""

    def __init__(self, context_callback):
        super().__init__()
        self.context_callback = context_callback
        self.running = True
        self.daemon = True

        if not CV2_AVAILABLE:
            print("   Vision disabled - opencv not available")
            self.running = False

    def capture_and_describe(self) -> str:
        """Capture a frame and get description from vision model."""
        if not CV2_AVAILABLE:
            return None

        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        if not cap.isOpened():
            return None

        # Warm up camera for auto-exposure
        for _ in range(10):
            cap.read()
            time.sleep(0.1)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Debug: save frame
        if Config.VISION_DEBUG:
            debug_path = Path(__file__).parent / "debug_frame.jpg"
            cv2.imwrite(str(debug_path), frame)

        # Encode to base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Send to vision model
        try:
            response = requests.post(
                f"{Config.OLLAMA_HOST}/api/generate",
                json={
                    "model": Config.VISION_MODEL,
                    "prompt": "Describe this image briefly in 1-2 sentences.",
                    "images": [img_base64],
                    "stream": False
                },
                timeout=15
            )

            if response.status_code == 200:
                description = response.json().get('response', '').strip()
                print(f"\n   [Visual]: {description}")
                return description
        except Exception as e:
            pass

        return None

    def run(self):
        if not self.running:
            return

        print("   Vision system online")
        time.sleep(2)

        while self.running:
            desc = self.capture_and_describe()
            if desc:
                self.context_callback(desc)
            time.sleep(Config.VISION_INTERVAL)

    def stop(self):
        self.running = False


# =============================================================================
# AUDIO LISTENER (Ears)
# =============================================================================

class AudioListener:
    """Captures audio from microphone and detects speech."""

    def __init__(self):
        self.audio_queue = queue.Queue()

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(indata.copy())

    def listen_for_speech(self) -> np.ndarray:
        """Listen until speech is detected and silence follows."""
        print("\n   Listening...")

        audio_buffer = []
        silence_samples = 0
        speech_detected = False
        samples_per_check = int(Config.SAMPLE_RATE * 0.1)
        silence_samples_needed = int(
            Config.SILENCE_DURATION * Config.SAMPLE_RATE / samples_per_check
        )

        with sd.InputStream(
            samplerate=Config.SAMPLE_RATE,
            channels=Config.CHANNELS,
            callback=self.audio_callback,
            blocksize=samples_per_check
        ):
            while True:
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.5)
                    amplitude = np.abs(audio_chunk).mean()

                    if amplitude > Config.SILENCE_THRESHOLD:
                        speech_detected = True
                        silence_samples = 0
                        audio_buffer.append(audio_chunk)
                    elif speech_detected:
                        silence_samples += 1
                        audio_buffer.append(audio_chunk)
                        if silence_samples >= silence_samples_needed:
                            break
                except queue.Empty:
                    if speech_detected:
                        silence_samples += 1
                        if silence_samples >= silence_samples_needed:
                            break

        if audio_buffer:
            return np.concatenate(audio_buffer, axis=0)
        return np.array([])


# =============================================================================
# SPEECH RECOGNIZER
# =============================================================================

class SpeechRecognizer:
    """Converts speech to text using Whisper."""

    def __init__(self):
        self.model = None
        if WHISPER_AVAILABLE:
            print("   Loading Whisper model...")
            self.model = WhisperModel(
                Config.WHISPER_MODEL,
                device=Config.WHISPER_DEVICE,
                compute_type="int8"
            )
            print("   Whisper ready")

    def transcribe(self, audio: np.ndarray) -> str:
        if self.model is None or len(audio) == 0:
            return ""

        audio_float = audio.flatten().astype(np.float32)
        if audio_float.max() > 1.0:
            audio_float = audio_float / 32768.0

        segments, _ = self.model.transcribe(
            audio_float,
            beam_size=5,
            language="en",
            vad_filter=True
        )

        return " ".join([s.text for s in segments]).strip()


# =============================================================================
# VOICE SYNTHESIZER
# =============================================================================

class VoiceSynthesizer:
    """Text-to-speech using Piper."""

    PIPER_MODELS = {
        "en_US-lessac-medium": {
            "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        }
    }

    def __init__(self):
        self.model_path = None
        self.enabled = self._setup()

    def _setup(self) -> bool:
        try:
            subprocess.run(["piper", "--help"], capture_output=True, check=True)
        except:
            print("   Piper TTS not available")
            return False

        model_dir = Config.PIPER_MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)

        model_name = Config.PIPER_MODEL
        model_file = model_dir / f"{model_name}.onnx"
        config_file = model_dir / f"{model_name}.onnx.json"

        if not model_file.exists() or not config_file.exists():
            if model_name not in self.PIPER_MODELS:
                return False

            print(f"   Downloading voice model...")
            urls = self.PIPER_MODELS[model_name]

            for url, dest in [(urls["model"], model_file), (urls["config"], config_file)]:
                if not dest.exists():
                    try:
                        response = requests.get(url, stream=True)
                        response.raise_for_status()
                        with open(dest, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                    except:
                        return False

        self.model_path = model_file
        print("   Piper TTS ready")
        return True

    def speak(self, text: str):
        print(f"\n   Witness: {text}")

        if not self.enabled or not self.model_path:
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
        except Exception as e:
            pass


# =============================================================================
# THE WITNESS (Complete System)
# =============================================================================

class Witness:
    """Complete Witness system with memory."""

    def __init__(self):
        print("\n" + "="*50)
        print("  WITNESS - Complete System")
        print("  'Full Incarnation' Phase")
        print("="*50 + "\n")

        # Initialize all subsystems
        self.memory = MemorySystem()
        self.ears = AudioListener()
        self.speech = SpeechRecognizer()
        self.voice = VoiceSynthesizer()

        # Brain state
        self.visual_context = "I cannot see anything yet."
        self.conversation_history = []

        # Start vision
        self.eyes = VisualCortex(self.update_visual_context)
        self.eyes.start()

        # Check Ollama
        self._check_ollama()

        print("\n" + "-"*50)
        print("System ready. Speak to begin.")
        print("Vision updates every 15 seconds.")
        print("Say 'goodbye' to exit.")
        print("-"*50)

    def _check_ollama(self):
        try:
            response = requests.get(f"{Config.OLLAMA_HOST}/api/tags")
            if response.status_code == 200:
                print("   Ollama connected")
        except:
            print("   Warning: Ollama not available")

    def update_visual_context(self, description: str):
        self.visual_context = description

    def think(self, user_input: str) -> str:
        """Process input with memory and generate response."""

        # 1. Recall relevant memories
        memories = self.memory.recall(user_input)
        print(f"   [Memory]: {memories[:60]}...")

        # 2. Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Trim history
        if len(self.conversation_history) > Config.MAX_CONTEXT_TURNS * 2:
            self.conversation_history = self.conversation_history[-Config.MAX_CONTEXT_TURNS * 2:]

        # 3. Build context-aware system message
        system_content = (
            f"{WITNESS_SOUL}\n\n"
            f"[VISUAL CONTEXT: {self.visual_context}]\n"
            f"[RELEVANT MEMORIES: {memories}]"
        )

        messages = [
            {"role": "system", "content": system_content}
        ] + self.conversation_history

        # 4. Generate response
        try:
            response = requests.post(
                f"{Config.OLLAMA_HOST}/api/chat",
                json={
                    "model": Config.CHAT_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                reply = response.json()['message']['content']

                # Save to history and memory
                self.conversation_history.append({
                    "role": "assistant",
                    "content": reply
                })
                self.memory.remember(user_input, reply)

                return reply
            else:
                return "[Thought unclear]"

        except Exception as e:
            return f"[Error: {str(e)}]"

    def run(self):
        """Main conversation loop."""

        # Opening based on memory
        opening = self.think("I am awakening. What should I say to greet whoever is here?")
        self.voice.speak(opening)

        while True:
            try:
                audio = self.ears.listen_for_speech()
                if len(audio) == 0:
                    continue

                text = self.speech.transcribe(audio)
                if not text:
                    continue

                print(f"\n   You: {text}")

                # Check for exit
                if any(word in text.lower() for word in ['goodbye', 'exit', 'quit']):
                    farewell = self.think("The user is saying goodbye. Give a brief farewell, noting you will remember this conversation.")
                    self.voice.speak(farewell)
                    break

                # Think and respond
                response = self.think(text)
                self.voice.speak(response)

            except KeyboardInterrupt:
                print("\n\n[Session interrupted]")
                break

        self.eyes.stop()
        print("\n[Witness dormant - memories preserved]")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    witness = Witness()
    witness.run()
