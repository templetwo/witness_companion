#!/usr/bin/env python3
"""
WITNESS - Comprehensive Test Suite
===================================
Tests all components of the Witness system.

Run with: python test_witness.py
"""

import os
import sys
import time
import tempfile
import numpy as np
from pathlib import Path

# Set environment before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_imports():
    """Test that all required packages are installed."""
    print("=" * 50)
    print("  TEST: Imports & Dependencies")
    print("=" * 50)

    errors = []

    # Core packages
    packages = [
        ("numpy", "numpy"),
        ("sounddevice", "sounddevice"),
        ("requests", "requests"),
        ("cv2", "opencv-python"),
        ("faster_whisper", "faster-whisper"),
        ("chromadb", "chromadb"),
    ]

    for module, pip_name in packages:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            errors.append(f"  ✗ {module}: pip install {pip_name}")

    # Piper TTS
    import subprocess
    try:
        subprocess.run(["piper", "--help"], capture_output=True, check=True)
        print("  ✓ piper-tts")
    except:
        errors.append("  ⚠ piper-tts not in PATH")

    return errors


def test_ollama():
    """Test Ollama connectivity and models."""
    print("\n" + "=" * 50)
    print("  TEST: Ollama Connection")
    print("=" * 50)

    import requests
    errors = []

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            print(f"  ✓ Ollama running ({len(models)} models)")

            # Check required models
            required = ["llama3", "moondream"]
            for model in required:
                if any(model in m for m in models):
                    print(f"  ✓ {model} available")
                else:
                    errors.append(f"  ⚠ {model} not found - run: ollama pull {model}")
        else:
            errors.append(f"  ✗ Ollama returned {response.status_code}")
    except:
        errors.append("  ✗ Cannot connect to Ollama - run: ollama serve")

    return errors


def test_audio_devices():
    """Test audio input devices."""
    print("\n" + "=" * 50)
    print("  TEST: Audio Devices")
    print("=" * 50)

    import sounddevice as sd
    errors = []

    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]

    if input_devices:
        default = sd.query_devices(kind='input')
        print(f"  ✓ Found {len(input_devices)} input device(s)")
        print(f"    Default: {default['name']}")
    else:
        errors.append("  ✗ No audio input devices found")

    return errors


def test_camera():
    """Test camera access."""
    print("\n" + "=" * 50)
    print("  TEST: Camera")
    print("=" * 50)

    import cv2
    errors = []

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print(f"  ✓ Camera working ({frame.shape[1]}x{frame.shape[0]})")
        else:
            errors.append("  ✗ Camera opened but cannot read frame")
    else:
        errors.append("  ✗ Cannot open camera (check permissions)")

    return errors


def test_memory_system():
    """Test ChromaDB memory system."""
    print("\n" + "=" * 50)
    print("  TEST: Memory System (ChromaDB)")
    print("=" * 50)

    import chromadb
    errors = []

    # Create temp directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Initialize
            client = chromadb.PersistentClient(path=tmpdir)
            collection = client.get_or_create_collection(name="test_collection")
            print("  ✓ ChromaDB initialized")

            # Add memory
            collection.add(
                documents=["User: Hello | Witness: Hi there!"],
                metadatas=[{"timestamp": "2024-01-01"}],
                ids=["test_1"]
            )
            print("  ✓ Memory storage works")

            # Recall memory
            results = collection.query(
                query_texts=["Hello"],
                n_results=1
            )
            if results['documents'] and results['documents'][0]:
                print("  ✓ Memory recall works")
            else:
                errors.append("  ✗ Memory recall failed")

            # Check persistence
            count = collection.count()
            print(f"  ✓ Memory count: {count}")

        except Exception as e:
            errors.append(f"  ✗ ChromaDB error: {e}")

    return errors


def test_whisper():
    """Test Whisper model loading."""
    print("\n" + "=" * 50)
    print("  TEST: Whisper Speech Recognition")
    print("=" * 50)

    from faster_whisper import WhisperModel
    errors = []

    try:
        print("  Loading Whisper base.en model...")
        model = WhisperModel("base.en", device="cpu", compute_type="int8")
        print("  ✓ Whisper model loaded")

        # Test with silent audio
        silent_audio = np.zeros(16000, dtype=np.float32)  # 1 second silence
        segments, _ = model.transcribe(silent_audio, beam_size=1)
        text = " ".join([s.text for s in segments]).strip()
        print(f"  ✓ Transcription works (silent test: '{text}')")

    except Exception as e:
        errors.append(f"  ✗ Whisper error: {e}")

    return errors


def test_vision_model():
    """Test vision model with a test image."""
    print("\n" + "=" * 50)
    print("  TEST: Vision Model (Moondream)")
    print("=" * 50)

    import cv2
    import base64
    import requests
    errors = []

    # Create a simple test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "TEST", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)

    _, buffer = cv2.imencode('.jpg', test_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    try:
        print("  Sending test image to moondream...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "moondream",
                "prompt": "What text do you see in this image?",
                "images": [img_base64],
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json().get('response', '')
            print(f"  ✓ Vision model responded")
            print(f"    Response: {result[:100]}...")
        else:
            errors.append(f"  ✗ Vision model error: {response.status_code}")

    except Exception as e:
        errors.append(f"  ✗ Vision model error: {e}")

    return errors


def test_piper_tts():
    """Test Piper TTS."""
    print("\n" + "=" * 50)
    print("  TEST: Piper TTS")
    print("=" * 50)

    import subprocess
    errors = []

    model_path = Path.home() / ".local" / "share" / "piper-models" / "en_US-lessac-medium.onnx"

    if not model_path.exists():
        errors.append(f"  ⚠ Piper model not found at {model_path}")
        return errors

    print(f"  ✓ Model found: {model_path.name}")

    # Test synthesis
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        process = subprocess.Popen(
            ["piper", "--model", str(model_path), "--output_file", temp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        process.communicate(input=b"Test synthesis")

        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            print(f"  ✓ TTS synthesis works ({os.path.getsize(temp_path)} bytes)")
            os.unlink(temp_path)
        else:
            errors.append("  ✗ TTS synthesis failed")

    except Exception as e:
        errors.append(f"  ✗ TTS error: {e}")

    return errors


def test_existing_memory():
    """Test existing witness memory database."""
    print("\n" + "=" * 50)
    print("  TEST: Existing Witness Memory")
    print("=" * 50)

    import chromadb
    errors = []

    memory_path = "./witness_memory_db"
    if not os.path.exists(memory_path):
        print("  ⚠ No existing memory database found")
        return errors

    try:
        client = chromadb.PersistentClient(path=memory_path)
        collection = client.get_or_create_collection(name="witness_logs")
        count = collection.count()
        print(f"  ✓ Found {count} memories in database")

        if count > 0:
            # Sample a memory
            results = collection.query(
                query_texts=["hello"],
                n_results=1
            )
            if results['documents'] and results['documents'][0]:
                sample = results['documents'][0][0][:80]
                print(f"  Sample: {sample}...")

    except Exception as e:
        errors.append(f"  ✗ Memory database error: {e}")

    return errors


def test_module_imports():
    """Test that witness modules can be imported."""
    print("\n" + "=" * 50)
    print("  TEST: Witness Module Imports")
    print("=" * 50)

    errors = []
    modules = ["witness", "witness_complete", "witness_refined"]

    for module in modules:
        module_path = Path(__file__).parent / f"{module}.py"
        if module_path.exists():
            try:
                # Just check syntax
                import py_compile
                py_compile.compile(str(module_path), doraise=True)
                print(f"  ✓ {module}.py syntax valid")
            except py_compile.PyCompileError as e:
                errors.append(f"  ✗ {module}.py: {e}")
        else:
            print(f"  - {module}.py not found")

    return errors


def main():
    print("\n" + "#" * 50)
    print("  WITNESS - Comprehensive Test Suite")
    print("#" * 50)

    all_errors = []

    # Run all tests
    all_errors.extend(test_imports())
    all_errors.extend(test_ollama())
    all_errors.extend(test_audio_devices())
    all_errors.extend(test_camera())
    all_errors.extend(test_memory_system())
    all_errors.extend(test_existing_memory())
    all_errors.extend(test_module_imports())

    # Heavy model tests
    print("\n" + "=" * 50)
    print("  Heavy Model Tests")
    print("=" * 50)

    all_errors.extend(test_whisper())
    all_errors.extend(test_vision_model())
    all_errors.extend(test_piper_tts())

    # Summary
    print("\n" + "#" * 50)
    print("  TEST SUMMARY")
    print("#" * 50)

    if all_errors:
        print("\n  Issues found:\n")
        for err in all_errors:
            print(err)
        print(f"\n  Total issues: {len(all_errors)}")
        return 1
    else:
        print("\n  ✓ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
