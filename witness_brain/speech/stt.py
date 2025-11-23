# speech/stt.py - Speech-to-Text module
import os
import logging
from faster_whisper import WhisperModel
import numpy as np

# Reduce faster-whisper VAD spam
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

# Recommended: Set a cache directory for models
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "witness_brain_models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class STT:
    """Handles Speech-to-Text using faster-whisper."""
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Initializes the STT engine.

        Args:
            model_size (str): The whisper model size (e.g., "base", "small", "medium").
            device (str): "cuda" or "cpu".
            compute_type (str): "float16", "int8", etc.
        """
        print(f"Loading STT model '{model_size}'...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=MODEL_CACHE_DIR,
            cpu_threads=4,  # Limit threads to prevent resource exhaustion
            num_workers=1   # Single worker to reduce memory
        )
        print("STT model loaded.")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribes a chunk of audio.

        Args:
            audio_data (np.ndarray): A NumPy array containing the audio data (float32).

        Returns:
            str: The transcribed text.
        """
        if audio_data is None or audio_data.size == 0:
            return ""

        # Ensure 1D array
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        segments, _ = self.model.transcribe(
            audio_data,
            beam_size=1,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.4,              # Lower = less aggressive (default ~0.5)
                "min_silence_duration_ms": 300, # Shorter silence detection
                "speech_pad_ms": 200,          # Padding around speech
            }
        )

        # Safely extract text from segments
        texts = []
        for segment in segments:
            if segment.text:
                texts.append(segment.text.strip())

        transcribed_text = " ".join(texts)

        return transcribed_text.strip()

# Example usage (for testing)
if __name__ == '__main__':
    # This is a placeholder for a real audio source
    # In the actual CNS loop, we will get this from the microphone
    print("Running STT test with a dummy audio signal...")
    SAMPLE_RATE = 16000  # 16kHz
    dummy_audio = np.random.randn(SAMPLE_RATE * 5).astype(np.float32) # 5 seconds of noise

    stt_engine = STT(model_size="tiny") # Use tiny for quick testing
    
    transcription = stt_engine.transcribe(dummy_audio)
    
    print(f"Transcription from dummy audio: '{transcription}'")
    # Expected output is likely empty or gibberish from noise, but confirms the pipeline works.

