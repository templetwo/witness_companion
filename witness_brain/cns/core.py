import os
import yaml
import json
import logging
import sounddevice as sd
import numpy as np
import queue
import threading
import time
import re
from datetime import datetime
from typing import List, Dict

from witness_brain.speech.stt import STT
from witness_brain.speech.tts import TTS
from witness_brain.schemas.event_schema import Event
from witness_brain.models.brain_client import BrainClient
from witness_brain.vision.camera import Camera
from witness_brain.vision.analyzer import VisionAnalyzer

# Custom JSON Formatter for Event objects
class JSONEventFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'event_data'):
            # Directly format the event_data dictionary
            return json.dumps(record.event_data)
        # Fallback for non-event records, though they shouldn't reach this handler
        return super().format(record)

class WitnessCNS:
    """
    The Central Nervous System (CNS) for the Witness Brain.
    Manages audio input, STT, TTS, event logging, and wake word detection.
    """
    def __init__(self, config_path="witness_brain/config/audio.yaml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # Initialize internal state
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        self.buffer_duration = 0
        self.situation_buffer: List[Event] = []
        self.BUFFER_MAX_SIZE = 20 # Max events to keep in memory

        self.wake_mode = False
        self.wake_until = None  # Timestamp for wake window expiry
        self.wake_window_seconds = 10.0  # How long to listen after wake
        self.wake_phrases = ["witness", "hey witness", "hello witness", "ok witness"]
        self.logger.info(f"Wake phrases: {self.wake_phrases}")

        # Initialize STT and TTS engines based on config
        stt_config = self.config.get("stt", {})
        self.stt_engine = STT(
            model_size=stt_config.get("model_size", "tiny"),
            device=stt_config.get("device", "cpu"),
            compute_type=stt_config.get("compute_type", "int8")
        )
        self.logger.info(f"STT initialized: model={stt_config.get('model_size', 'tiny')}, device={stt_config.get('device', 'cpu')}")

        tts_config = self.config.get("tts", {})
        self.tts_enabled = tts_config.get("enabled", True)
        if self.tts_enabled:
            self.tts_engine = TTS(
                model_name=tts_config.get("voice_model", "en_US-lessac-medium")
            )
        else:
            self.tts_engine = None

        # Initialize Brain client
        self.brain_client = BrainClient()

        # Initialize Vision
        vision_config = self.config.get("vision", {})
        self.vision_enabled = vision_config.get("enabled", False)
        if self.vision_enabled:
            self.camera = Camera(camera_index=vision_config.get("camera_index", 0))
            self.vision_analyzer = VisionAnalyzer(
                model_name=vision_config.get("model_name", "llama3.2-vision:latest")
            )
            self.vision_interval = vision_config.get("interval_seconds", 30)
            self._last_vision_time = 0
            self._vision_in_progress = False
            self.logger.info(f"Vision enabled: interval={self.vision_interval}s")
        else:
            self.camera = None
            self.vision_analyzer = None
            self.vision_interval = 0

        # Conversation settings
        conv_config = self.config.get("conversation", {})
        self.conversation_mode = conv_config.get("mode", "wake")
        self.conversation_until = None  # Timestamp for follow-up window
        self.conversation_window_seconds = 10.0  # Follow-up window after response
        self.logger.info(f"Conversation mode: {self.conversation_mode}")

        # Audio Stream Control
        self._stop_event = threading.Event()
        self._audio_stream = None
        self._processing_thread = None
        self._is_speaking = False  # Flag to ignore input while speaking
        self._last_ai_utterance = None  # Track last speech for echo detection
        self._last_ai_utterance_time = 0  # When AI last spoke

        self.logger.info("Tip: for best results, use headphones for Witness audio to avoid the mic hearing its own voice.")

    def _load_config(self, config_path: str) -> Dict:
        """Loads configuration from a YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config file {config_path}: {e}")

    def _setup_logging(self) -> logging.Logger:
        """Sets up logging for console (human-readable) and file (JSONL)."""
        logger = logging.getLogger("WitnessCNS")
        logger.setLevel(self.config["system"].get("log_level", "INFO").upper())
        logger.propagate = False # Prevent double logging if root logger is also configured

        # Console Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File Handler (JSONL)
        log_file = self.config["system"].get("log_file", "witness_events.log")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(JSONEventFormatter())
        logger.addHandler(file_handler)
        
        return logger

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for the sounddevice input stream."""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())

    def add_event(self, event: Event):
        """
        Appends an Event into internal buffers and logs it to console and JSONL file.
        """
        self.situation_buffer.append(event)
        if len(self.situation_buffer) > self.BUFFER_MAX_SIZE:
            self.situation_buffer.pop(0) # Keep buffer size limited

        # Log to console
        self.logger.info(f"[CONSOLE] {event}")

        # Log to JSONL file using the dedicated handler
        # We create a log record that our JSONEventFormatter can pick up
        log_record = self.logger.makeRecord(
            self.logger.name, 
            self.logger.level, 
            __file__, 
            0, 
            '', # msg
            [], # args
            None, # exc_info
            func='add_event'
        )
        log_record.event_data = {
            "id": event.id,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
            "content": event.content
        }
        # Iterate over handlers and emit the record if it's a file handler configured for JSON
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler) and isinstance(handler.formatter, JSONEventFormatter):
                handler.emit(log_record)
    
    def get_recent_events(self, n: int = 20) -> List[Event]:
        """Return the last n events in memory."""
        return self.situation_buffer[-n:]

    def _normalize_text(self, text: str) -> str:
        """Normalize text for wake word matching."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s']", " ", text)  # remove punctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_wake_phrase(self, text: str) -> bool:
        """Check if text contains a wake phrase."""
        norm = self._normalize_text(text)
        return any(phrase in norm for phrase in self.wake_phrases)

    def _extract_command_from_wake(self, text: str) -> str:
        """Extract command that follows a wake phrase, if any."""
        norm = self._normalize_text(text)
        for phrase in sorted(self.wake_phrases, key=len, reverse=True):
            if phrase in norm:
                # Find where the wake phrase ends
                idx = norm.find(phrase)
                remainder = norm[idx + len(phrase):].strip()
                # Strip common filler words
                for filler in ["can you", "could you", "please", "um", "uh"]:
                    if remainder.startswith(filler):
                        remainder = remainder[len(filler):].strip()
                # Return if there's meaningful content (more than just "yes" or punctuation)
                if remainder and len(remainder) > 3 and remainder not in ["yes", "yeah", "yep"]:
                    return remainder
        return ""

    def _is_probable_echo(self, transcript: str) -> bool:
        """Check if transcript is likely an echo of our own speech."""
        if not self._last_ai_utterance:
            return False

        # Only check for echoes within 5 seconds of AI speaking
        if time.time() - self._last_ai_utterance_time > 5.0:
            return False

        norm_t = self._normalize_text(transcript)
        norm_ai = self._normalize_text(self._last_ai_utterance)

        if not norm_t or not norm_ai:
            return False

        # Token overlap heuristic
        t_tokens = set(norm_t.split())
        ai_tokens = set(norm_ai.split())

        if not t_tokens or not ai_tokens:
            return False

        overlap = len(t_tokens & ai_tokens) / max(len(t_tokens), 1)
        return overlap >= 0.75  # 75% overlap = probable echo

    def _speak_async(self, text: str):
        """Speak text in a background thread so it doesn't block the loop."""
        if not self.tts_enabled or not self.tts_engine:
            return

        def _run():
            self._is_speaking = True
            try:
                self.tts_engine.speak(text)
            except Exception as e:
                self.logger.error(f"TTS error: {e}")
            finally:
                self._is_speaking = False

        threading.Thread(target=_run, daemon=True).start()

    def _check_vision(self):
        """Perform a periodic vision check if enabled and due."""
        if not self.vision_enabled or self.vision_interval <= 0:
            return

        # Don't start new check if one is in progress
        if self._vision_in_progress:
            return

        now = time.time()
        if now - self._last_vision_time < self.vision_interval:
            return

        self._last_vision_time = now
        self._vision_in_progress = True

        # Capture and analyze in background thread
        def _vision_task():
            try:
                image = self.camera.capture_base64()
                if image:
                    description = self.vision_analyzer.describe_scene(image)
                    if description:
                        self.add_event(Event(source="vision:scene", content=description))
                        self.logger.info(f"[VISION] {description}")
            except Exception as e:
                self.logger.error(f"Vision error: {e}")
            finally:
                self._vision_in_progress = False

        threading.Thread(target=_vision_task, daemon=True).start()

    def look(self) -> str:
        """Manually trigger a vision check and return description."""
        if not self.vision_enabled:
            return "Vision not enabled"

        try:
            image = self.camera.capture_base64()
            if image:
                description = self.vision_analyzer.describe_scene(image)
                if description:
                    self.add_event(Event(source="vision:look", content=description))
                    return description
            return "Could not capture image"
        except Exception as e:
            self.logger.error(f"Vision error: {e}")
            return f"Vision error: {e}"

    def _process_audio_loop(self):
        """Main loop for processing audio, transcription, and wake word detection."""
        
        sample_rate = self.config["audio"].get("sample_rate", 16000)
        buffer_duration_seconds = self.config["audio"].get("buffer_duration_seconds", 5.0)

        while not self._stop_event.is_set():
            # Check vision periodically
            self._check_vision()

            # Pull audio from queue
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                self.audio_buffer.append(chunk)
                self.buffer_duration += len(chunk) / sample_rate

            # Process buffer when it reaches a certain size
            if self.buffer_duration >= buffer_duration_seconds:
                # Concatenate all chunks and clear buffer
                full_audio_data = np.concatenate(self.audio_buffer, axis=0)
                self.audio_buffer.clear()
                self.buffer_duration = 0

                # Convert to float32 if not already (faster-whisper expects float32)
                if full_audio_data.dtype != np.float32:
                    full_audio_data = full_audio_data.astype(np.float32)

                # Flatten to 1D (sounddevice gives (N,1) but whisper/VAD expects (N,))
                full_audio_data = full_audio_data.squeeze()

                # Transcribe with error handling
                try:
                    transcribed_text = self.stt_engine.transcribe(full_audio_data).lower().strip()
                except Exception as e:
                    self.logger.error(f"STT transcription error: {e}")
                    continue

                if transcribed_text and len(transcribed_text) > 1:
                    # Skip if we're currently speaking (avoid echo)
                    if self._is_speaking:
                        self.logger.debug(f"Ignoring input while speaking: '{transcribed_text}'")
                        continue

                    # Skip if this is probably an echo of our own speech
                    if self._is_probable_echo(transcribed_text):
                        self.logger.info(f"[ECHO] Suppressing probable self-echo: '{transcribed_text[:50]}...'")
                        continue

                    self.logger.debug(f"Transcription: '{transcribed_text}'")
                    now = time.time()

                    # Check windows
                    in_wake_window = self.wake_until and now <= self.wake_until
                    in_conversation = self.conversation_until and now <= self.conversation_until

                    if self.conversation_mode == "continuous":
                        # Continuous mode: everything is a command
                        self.add_event(Event(source="hearing:command", content=transcribed_text))

                        # Generate brain response
                        recent_events = self.get_recent_events(self.brain_client.max_events)
                        brain_output = self.brain_client.generate(recent_events)

                        # Log thought and action
                        self.logger.info(f"[BRAIN_THOUGHT] {brain_output.thought}")
                        if brain_output.action:
                            self.logger.info(f"[BRAIN_ACTION] {brain_output.action}")

                        # Speak and log
                        if brain_output.speech:
                            self._last_ai_utterance = brain_output.speech
                            self._last_ai_utterance_time = time.time()
                            self.add_event(Event(source="brain:speech", content=brain_output.speech))
                            self._speak_async(brain_output.speech)

                    else:
                        # Wake mode with conversation windows
                        if self._is_wake_phrase(transcribed_text):
                            # Check if there's a command embedded with the wake phrase
                            embedded_command = self._extract_command_from_wake(transcribed_text)

                            if embedded_command:
                                # Process the embedded command directly
                                self.add_event(Event(source="hearing:wake", content=transcribed_text))
                                self.add_event(Event(source="hearing:command", content=embedded_command))

                                # Get recent events and generate brain response
                                recent_events = self.get_recent_events(self.brain_client.max_events)
                                brain_output = self.brain_client.generate(recent_events)

                                # Log the brain's thought and action
                                self.logger.info(f"[BRAIN_THOUGHT] {brain_output.thought}")
                                if brain_output.action:
                                    self.logger.info(f"[BRAIN_ACTION] {brain_output.action}")

                                # Speak the brain's response
                                if brain_output.speech:
                                    self._last_ai_utterance = brain_output.speech
                                    self._last_ai_utterance_time = time.time()
                                    self.add_event(Event(source="brain:speech", content=brain_output.speech))
                                    self._speak_async(brain_output.speech)

                                # Set conversation follow-up window
                                self.conversation_until = now + self.conversation_window_seconds
                                self.logger.info(f"Conversation window open ({self.conversation_window_seconds}s)")
                            else:
                                # Just wake phrase, wait for command
                                self.wake_until = now + self.wake_window_seconds
                                self.add_event(Event(source="hearing:wake", content=transcribed_text))
                                self._last_ai_utterance = "Yes?"
                                self._last_ai_utterance_time = time.time()
                                self._speak_async("Yes?")
                                self.logger.info(f"Witness in wake mode (window: {self.wake_window_seconds}s)")

                        elif in_wake_window or in_conversation:
                            # We're in wake or conversation window - treat as command
                            self.wake_until = None  # Close wake window

                            # Log the user's command
                            self.add_event(Event(source="hearing:command", content=transcribed_text))

                            # Get recent events and generate brain response
                            recent_events = self.get_recent_events(self.brain_client.max_events)
                            brain_output = self.brain_client.generate(recent_events)

                            # Log the brain's thought and action
                            self.logger.info(f"[BRAIN_THOUGHT] {brain_output.thought}")
                            if brain_output.action:
                                self.logger.info(f"[BRAIN_ACTION] {brain_output.action}")

                            # Speak the brain's response (async)
                            if brain_output.speech:
                                self._last_ai_utterance = brain_output.speech
                                self._last_ai_utterance_time = time.time()
                                self.add_event(Event(source="brain:speech", content=brain_output.speech))
                                self._speak_async(brain_output.speech)

                            # Set conversation follow-up window
                            self.conversation_until = now + self.conversation_window_seconds
                            self.logger.info(f"Conversation window open ({self.conversation_window_seconds}s)")

            time.sleep(0.05) # Small delay to prevent busy-waiting

    def start(self):
        """Starts audio input and the main processing loop."""
        self.logger.info("Starting WitnessCNS...")
        audio_config = self.config["audio"]

        self._audio_stream = sd.InputStream(
            samplerate=audio_config.get("sample_rate", 16000),
            blocksize=audio_config.get("block_size", 4096),
            channels=audio_config.get("channels", 1),
            callback=self._audio_callback
        )
        self._audio_stream.start()
        self.logger.info(f"Audio stream started: {self._audio_stream.samplerate} Hz, {self._audio_stream.channels} channel(s)")

        self._processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self._processing_thread.start()
        self.logger.info("Audio processing thread started.")
        
        self.logger.info("WitnessCNS is active. Listening for wake phrase...")

    def stop(self):
        """Signals a clean shutdown of the CNS."""
        self.logger.info("Stopping WitnessCNS...")
        self._stop_event.set() # Signal processing thread to stop

        if self._audio_stream and self._audio_stream.active:
            self._audio_stream.stop()
            self._audio_stream.close()
            self.logger.info("Audio stream stopped.")
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5) # Wait for thread to finish
            if self._processing_thread.is_alive():
                self.logger.warning("Audio processing thread did not terminate cleanly.")
            else:
                self.logger.info("Audio processing thread stopped.")

        # Release camera
        if self.camera:
            self.camera.release()
            self.logger.info("Camera released.")

        self.logger.info("WitnessCNS stopped.")

