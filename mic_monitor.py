#!/usr/bin/env python3
"""
RAW MICROPHONE MONITOR
======================
Visual VU meter to see exactly what the mic is receiving.
"""

import numpy as np
import sounddevice as sd

print("\n" + "=" * 50)
print("   RAW MICROPHONE MONITOR")
print("   Talk to see the bars move")
print("   Ctrl+C to stop")
print("=" * 50 + "\n")

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")

    # Calculate volume (RMS)
    volume = np.sqrt(np.mean(indata**2))

    # Also show mean (what Witness uses)
    mean_vol = np.abs(indata).mean()

    # Create visual bar
    bar_len = int(volume * 200)  # Scale for visibility
    bar = '█' * min(bar_len, 50)

    # Show both metrics
    print(f"RMS: {volume:.4f} | Mean: {mean_vol:.4f} | {bar}")

try:
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000, blocksize=1600):
        print("Listening... Speak now!\n")
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n\nStopped.")
