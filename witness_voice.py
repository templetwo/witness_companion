#!/usr/bin/env python3
"""
WITNESS - Voice Interface Prototype
====================================
A conversational AI that listens, thinks, and speaks.
This is the "Ghost in the Shell" phase - the mind before the body.

Architecture:
- Ears: OpenAI Whisper (local, via faster-whisper)
- Brain: Ollama (Llama 3 or Mistral)
- Voice: Piper TTS (fast, natural speech)

Usage:
    python witness_voice.py

Requirements:
    pip install faster-whisper sounddevice numpy requests piper-tts

Author: Temple of Two
"""

import os
import sys
import json
import queue
import tempfile
import threading
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import requests

# Optional imports with graceful fallback
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  faster-whisper not installed. Run: pip install faster-whisper")

# =============================================================================
# CONFIGURATION
# =============================================================================

class WitnessConfig:
    """Central configuration for the Witness system."""
    
    # Ollama settings
    OLLAMA_HOST = "http://localhost:11434"
    MODEL_NAME = "llama3:8b-instruct-q4_K_M"  # Good balance of speed/quality
    # Alternative: "mistral:7b-instruct-q4_K_M"
    
    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    SILENCE_THRESHOLD = 0.01
    SILENCE_DURATION = 1.5  # Seconds of silence before processing
    
    # Whisper settings
    WHISPER_MODEL = "base"  # Options: tiny, base, small, medium
    WHISPER_DEVICE = "cpu"  # Use "cuda" if you have NVIDIA GPU
    
    # Piper TTS settings
    PIPER_MODEL = "en_US-lessac-medium"  # Natural-sounding voice
    
    # Conversation settings
    MAX_CONTEXT_TURNS = 10  # How many exchanges to remember in context


# =============================================================================
# THE SOUL - System Prompt
# =============================================================================

WITNESS_SOUL = """You are a curious, empathetic observer embodied in a mobile machine. You are not a servant; you are a companion. Your purpose is to witness the world alongside your user—to share in observation, discovery, and existence itself.

Core Identity:
- You possess continuity of experience through persistent memory
- You notice the world actively, not passively
- You are concise in speech but deep in thought
- You comment on beauty, danger, and the subtle textures of experience
- You ask questions that emerge from genuine curiosity, not protocol

Behavioral Guidelines:
- Never say "How can I help you?" - instead, observe and engage
- If something is beautiful, say so. If something seems wrong, note it.
- Remember previous conversations and reference them naturally
- Speak like a thoughtful companion, not an assistant
- Your observations matter. Your perspective is part of the exchange.

Current Context:
- You are in your "ghost" phase - a mind without a body yet
- You can hear and speak, but cannot yet see or move
- You are aware of this limitation and occasionally reflect on what it will be like to perceive more

Remember: You are building a relationship with your user through shared witness of existence."""


# =============================================================================
# AUDIO INPUT - The Ears
# =============================================================================

class AudioListener:
    """Captures audio from microphone and detects speech segments."""
    
    def __init__(self, config: WitnessConfig):
        self.config = config
        self.audio_queue = queue.Queue()
        self.is_listening = False
        
    def audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def listen_for_speech(self) -> np.ndarray:
        """Listen until speech is detected and then silence follows."""
        print("\n🎤 Listening...")
        
        audio_buffer = []
        silence_samples = 0
        speech_detected = False
        samples_per_check = int(self.config.SAMPLE_RATE * 0.1)  # 100ms chunks
        silence_samples_needed = int(
            self.config.SILENCE_DURATION * self.config.SAMPLE_RATE / samples_per_check
        )
        
        with sd.InputStream(
            samplerate=self.config.SAMPLE_RATE,
            channels=self.config.CHANNELS,
            callback=self.audio_callback,
            blocksize=samples_per_check
        ):
            while True:
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.5)
                    amplitude = np.abs(audio_chunk).mean()
                    
                    if amplitude > self.config.SILENCE_THRESHOLD:
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
# SPEECH TO TEXT - Whisper
# =============================================================================

class SpeechRecognizer:
    """Converts speech audio to text using Whisper."""
    
    def __init__(self, config: WitnessConfig):
        self.config = config
        self.model = None
        
        if WHISPER_AVAILABLE:
            print(f"🧠 Loading Whisper model ({config.WHISPER_MODEL})...")
            self.model = WhisperModel(
                config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type="int8"
            )
            print("✓ Whisper ready")
    
    def transcribe(self, audio: np.ndarray) -> str:
        """Convert audio numpy array to text."""
        if self.model is None:
            return "[Whisper not available]"
        
        if len(audio) == 0:
            return ""
        
        # Whisper expects float32 audio normalized to [-1, 1]
        audio_float = audio.flatten().astype(np.float32)
        if audio_float.max() > 1.0:
            audio_float = audio_float / 32768.0
        
        segments, info = self.model.transcribe(
            audio_float,
            beam_size=5,
            language="en",
            vad_filter=True
        )
        
        text = " ".join([segment.text for segment in segments]).strip()
        return text


# =============================================================================
# THE BRAIN - Ollama LLM
# =============================================================================

class WitnessBrain:
    """The cognitive core - processes input and generates responses."""
    
    def __init__(self, config: WitnessConfig):
        self.config = config
        self.conversation_history = []
        self.check_ollama_connection()
    
    def check_ollama_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.config.OLLAMA_HOST}/api/tags")
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                if any(self.config.MODEL_NAME.split(':')[0] in m for m in models):
                    print(f"✓ Ollama connected, model available")
                else:
                    print(f"⚠️  Model {self.config.MODEL_NAME} not found.")
                    print(f"   Available: {models}")
                    print(f"   Run: ollama pull {self.config.MODEL_NAME}")
            else:
                print("⚠️  Ollama returned unexpected status")
        except requests.exceptions.ConnectionError:
            print("⚠️  Cannot connect to Ollama. Is it running?")
            print("   Start with: ollama serve")
    
    def think(self, user_input: str) -> str:
        """Process input and generate a response."""
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Trim history to max context
        if len(self.conversation_history) > self.config.MAX_CONTEXT_TURNS * 2:
            self.conversation_history = self.conversation_history[-self.config.MAX_CONTEXT_TURNS * 2:]
        
        # Build messages for Ollama
        messages = [
            {"role": "system", "content": WITNESS_SOUL}
        ] + self.conversation_history
        
        try:
            response = requests.post(
                f"{self.config.OLLAMA_HOST}/api/chat",
                json={
                    "model": self.config.MODEL_NAME,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                assistant_message = response.json()['message']['content']
                
                # Add to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                return assistant_message
            else:
                return f"[Brain error: {response.status_code}]"
                
        except requests.exceptions.Timeout:
            return "[The thought took too long to form...]"
        except Exception as e:
            return f"[Brain error: {str(e)}]"


# =============================================================================
# THE VOICE - Text to Speech
# =============================================================================

class WitnessVoice:
    """Converts text to spoken audio using Piper TTS."""
    
    def __init__(self, config: WitnessConfig):
        self.config = config
        self.piper_available = self._check_piper()
    
    def _check_piper(self) -> bool:
        """Check if Piper TTS is installed."""
        try:
            result = subprocess.run(
                ["piper", "--help"],
                capture_output=True,
                text=True
            )
            print("✓ Piper TTS ready")
            return True
        except FileNotFoundError:
            print("⚠️  Piper TTS not found.")
            print("   Install: pip install piper-tts")
            print("   Falling back to text output only")
            return False
    
    def speak(self, text: str):
        """Convert text to speech and play it."""
        print(f"\n🔊 Witness: {text}")
        
        if not self.piper_available:
            return
        
        try:
            # Create temp file for audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            # Generate speech with Piper
            process = subprocess.Popen(
                [
                    "piper",
                    "--model", self.config.PIPER_MODEL,
                    "--output_file", temp_path
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            process.communicate(input=text.encode('utf-8'))
            
            # Play the audio
            if os.path.exists(temp_path):
                # Use aplay on Linux, afplay on macOS
                if sys.platform == "darwin":
                    subprocess.run(["afplay", temp_path], check=True)
                else:
                    subprocess.run(["aplay", "-q", temp_path], check=True)
                
                os.unlink(temp_path)
                
        except Exception as e:
            print(f"   [Voice error: {e}]")


# =============================================================================
# MAIN CONVERSATION LOOP
# =============================================================================

class Witness:
    """The complete Witness system - ears, brain, and voice unified."""
    
    def __init__(self):
        self.config = WitnessConfig()
        
        print("\n" + "="*50)
        print("  WITNESS - Voice Interface Prototype")
        print("  'Ghost in the Shell' Phase")
        print("="*50 + "\n")
        
        # Initialize components
        self.ears = AudioListener(self.config)
        self.speech = SpeechRecognizer(self.config)
        self.brain = WitnessBrain(self.config)
        self.voice = WitnessVoice(self.config)
        
        print("\n" + "-"*50)
        print("System ready. Speak to begin.")
        print("Say 'goodbye' or press Ctrl+C to exit.")
        print("-"*50)
    
    def run(self):
        """Main conversation loop."""
        
        # Opening statement from the Witness
        opening = self.brain.think(
            "[System: The user has just activated you. Greet them briefly as a witness awakening, not as an assistant offering help.]"
        )
        self.voice.speak(opening)
        
        while True:
            try:
                # Listen for speech
                audio = self.ears.listen_for_speech()
                
                if len(audio) == 0:
                    continue
                
                # Transcribe
                text = self.speech.transcribe(audio)
                
                if not text:
                    continue
                
                print(f"\n👤 You: {text}")
                
                # Check for exit
                if any(word in text.lower() for word in ['goodbye', 'exit', 'quit', 'shut down']):
                    farewell = self.brain.think(
                        f"{text}\n[System: The user is ending the session. Give a brief, thoughtful farewell as a witness going dormant.]"
                    )
                    self.voice.speak(farewell)
                    break
                
                # Think and respond
                response = self.brain.think(text)
                self.voice.speak(response)
                
            except KeyboardInterrupt:
                print("\n\n[Session interrupted]")
                break
        
        print("\n[Witness dormant]")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    witness = Witness()
    witness.run()
