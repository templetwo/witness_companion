# speech/tts.py - Text-to-Speech module
import os
import requests
import sounddevice as sd
from piper.voice import PiperVoice
import numpy as np
import logging

# Use the same cache directory as STT
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "witness_brain_models")
TTS_MODEL_DIR = os.path.join(MODEL_CACHE_DIR, "piper_voices")
os.makedirs(TTS_MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTS:
    """Handles Text-to-Speech using piper-tts."""

    def __init__(self, model_name="en_US-lessac-medium"):
        """
        Initializes the TTS engine.

        Args:
            model_name (str): The name of the Piper voice model to use.
        """
        self.model_name = model_name
        self.voice = self._load_voice()

    def _get_model_files(self):
        """Constructs URLs and local paths for the voice model."""
        base_dir_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
        onnx_file = f"{self.model_name}.onnx"
        json_file = f"{self.model_name}.onnx.json"
        
        return {
            "onnx": {"url": f"{base_dir_url}/{onnx_file}", "local_path": os.path.join(TTS_MODEL_DIR, onnx_file)},
            "json": {"url": f"{base_dir_url}/{json_file}", "local_path": os.path.join(TTS_MODEL_DIR, json_file)}
        }

    def _download_model(self):
        """Downloads the voice model files if they don't exist."""
        files = self._get_model_files()
        
        for key, file_info in files.items():
            local_path = file_info["local_path"]
            if not os.path.exists(local_path):
                logger.info(f"Downloading {key} model file to {local_path}...")
                try:
                    response = requests.get(file_info["url"], stream=True)
                    response.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"Successfully downloaded {os.path.basename(local_path)}.")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to download {file_info['url']}: {e}")
                    # Clean up partial download
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    return False
        return True

    def _load_voice(self):
        """Loads the Piper voice, downloading if necessary."""
        if not self._download_model():
            raise RuntimeError("Could not download TTS model. Please check connection and file paths.")

        files = self._get_model_files()
        onnx_path = files["onnx"]["local_path"]

        logger.info(f"Loading Piper voice from: {onnx_path}")
        voice = PiperVoice.load(onnx_path)
        logger.info("Piper voice loaded.")
        return voice

    def speak(self, text: str, blocking: bool = True):
        """
        Synthesizes text into speech and plays it.

        Args:
            text (str): The text to be spoken.
            blocking (bool): If True, wait for speech to finish. If False, return immediately.
        """
        if not text:
            return

        logger.info(f"Synthesizing speech for: '{text}'")
        try:
            audio_stream = self.voice.synthesize_stream_raw(text)

            # Collect audio data
            audio_data = b''
            for audio_bytes in audio_stream:
                audio_data += audio_bytes

            # Convert to numpy array and play
            if audio_data:
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                sd.play(audio_np, samplerate=self.voice.config.sample_rate)
                if blocking:
                    sd.wait()
                logger.info("Finished speaking.")
        except Exception as e:
            logger.error(f"TTS error: {e}")

# Example usage (for testing)
if __name__ == '__main__':
    print("Running TTS test...")
    try:
        tts_engine = TTS()
        tts_engine.speak("Hello, this is a test of the text to speech system.")
        tts_engine.speak("Witness brain online.")
    except Exception as e:
        logger.error(f"An error occurred during the TTS test: {e}")

