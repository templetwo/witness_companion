#!/usr/bin/env python3
"""
TEST TRANSCRIPTION
==================
Isolates the faster-whisper model to verify it can transcribe a known audio file.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
MODEL_SIZE = "base.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
AUDIO_FILE = "debug_audio.wav"

def main():
    """
    Loads and transcribes the specified audio file.
    """
    print("="*50)
    print("  Whisper Transcription Test")
    print("="*50 + "\n")

    if not os.path.exists(AUDIO_FILE):
        print(f"ERROR: Audio file not found at '{AUDIO_FILE}'")
        print("Please ensure the audio file exists in the same directory.")
        return

    try:
        print(f"   Loading model: {MODEL_SIZE} ({DEVICE}, {COMPUTE_TYPE})")
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("   Model loaded successfully.")
    except Exception as e:
        print(f"\nFATAL: Failed to load Whisper model: {e}")
        print("This could be an issue with the model files or installation.")
        return

    try:
        print(f"\n   Loading audio file: {AUDIO_FILE}")
        audio, samplerate = sf.read(AUDIO_FILE, dtype='float32')
        print(f"   Audio loaded. Sample rate: {samplerate}, Duration: {len(audio)/samplerate:.2f}s")
        
        if samplerate != 16000:
            print("\nWARNING: Sample rate is not 16kHz. This may affect transcription quality.")
            # In a real scenario, we would resample here, but for a simple test,
            # we'll proceed and see what happens.
        
        # Replicate processing from the main script
        # audio = audio.flatten().astype(np.float32) / 32768.0 -> not needed as soundfile does it

        print("\n   Transcribing... (this may take a moment)")
        segments, info = model.transcribe(audio, beam_size=5, language="en")

        print(f"   Detected language: '{info.language}' with {info.language_probability:.2f} probability")
        
        print("\n--- TRANSCRIPTION RESULT ---")
        full_text = " ".join([seg.text.strip() for seg in segments])
        if full_text:
            print(full_text)
        else:
            print("[No text transcribed]")
        print("----------------------------\n")


    except Exception as e:
        print(f"\nFATAL: An error occurred during transcription: {e}")

if __name__ == "__main__":
    main()
