import os
import yaml
import json
import logging
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from datetime import datetime
from typing import List, Dict

from witness_brain.speech.stt import STT
from witness_brain.speech.tts import TTS
from witness_brain.schemas.event_schema import Event
from witness_brain.models.brain_client import BrainClient

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
        self.wake_phrase = [phrase.lower() for phrase in ["hey witness", "witness"]]
        self.logger.info(f"Wake phrases: {self.wake_phrase}")

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

        # Audio Stream Control
        self._stop_event = threading.Event()
        self._audio_stream = None
        self._processing_thread = None

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

    def _process_audio_loop(self):
        """Main loop for processing audio, transcription, and wake word detection."""
        
        sample_rate = self.config["audio"].get("sample_rate", 16000)
        buffer_duration_seconds = self.config["audio"].get("buffer_duration_seconds", 5.0)

        while not self._stop_event.is_set():
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
                    self.logger.debug(f"Transcription: '{transcribed_text}'")

                    # Wake word detection logic
                    if not self.wake_mode:
                        is_wake_phrase = any(phrase in transcribed_text for phrase in self.wake_phrase)
                        if is_wake_phrase:
                            self.wake_mode = True
                            self.add_event(Event(source="hearing:wake", content=transcribed_text))
                            if self.tts_enabled:
                                self.tts_engine.speak("Yes?")
                            self.logger.info("Witness in wake mode.")
                        # else:
                            # Optionally log ambient non-wake phrases
                            # self.add_event(Event(source="hearing:ambient", content=transcribed_text))
                    else: # wake_mode is True
                        # Log the user's command
                        self.add_event(Event(source="hearing:command", content=transcribed_text))

                        # Get recent events and generate brain response
                        recent_events = self.get_recent_events(self.brain_client.max_events)
                        brain_output = self.brain_client.generate(recent_events)

                        # Log the brain's thought and action
                        self.logger.info(f"[BRAIN_THOUGHT] {brain_output.thought}")
                        if brain_output.action:
                            self.logger.info(f"[BRAIN_ACTION] {brain_output.action}")

                        # Speak the brain's response
                        if self.tts_enabled and brain_output.speech:
                            self.tts_engine.speak(brain_output.speech)

                        # Log the brain's speech as an event
                        self.add_event(Event(source="brain:speech", content=brain_output.speech))

                        self.wake_mode = False # Reset wake mode after command
                        self.logger.info("Witness returning to sleep mode.")

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
        
        self.logger.info("WitnessCNS stopped.")

