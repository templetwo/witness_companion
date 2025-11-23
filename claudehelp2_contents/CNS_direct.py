#!/usr/bin/env python3
"""
CNS_direct.py - The Bicameral Mind (Direct PTY Edition)
========================================================
System 1 + System 2 integration WITHOUT WebSocket dependency.
Uses PTY to directly run Moshi CLI and parse its output.

ARCHITECTURE:
- System 1: Moshi (via python -m moshi_mlx.local)
- System 2: Dolphin (remote on Studio)
- Vision: LLaVA (remote on Studio)
- Voice: Piper (local)

This version is more reliable than WebSocket for local Moshi.

Author: Temple of Two
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import pty
import select
import subprocess
import threading
import time
import tempfile
import re
from pathlib import Path
from collections import deque

import cv2
import base64
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """System configuration"""
    
    # NETWORK
    STUDIO_IP = "192.168.1.195"
    OLLAMA_URL = f"http://{STUDIO_IP}:11434/api/chat"
    VISION_URL = f"http://{STUDIO_IP}:11434/api/generate"
    
    # MOSHI COMMAND (PTY mode)
    MOSHI_CMD = [sys.executable, "-m", "moshi_mlx.local", "-q", "4"]
    
    # MODELS
    DEEP_MODEL = "dolphin3:8b"
    VISION_MODEL = "llava:7b"
    
    # TTS
    PIPER_MODEL = "en_US-lessac-medium"
    PIPER_MODEL_DIR = Path.home() / ".local" / "share" / "piper-models"
    
    # VISION
    CAMERA_INDEX = 0
    MOTION_THRESHOLD = 3000
    VISION_COOLDOWN = 3
    VISION_STALE_TIME = 30
    
    # TRIGGERS
    TRIGGER_COOLDOWN = 10
    MAX_TRIGGER_LENGTH = 100
    
    # MEMORY
    BUFFER_SIZE = 20


# =============================================================================
# SHARED STATE
# =============================================================================

class SoulState:
    """Shared state across all systems"""
    
    def __init__(self):
        # Vision
        self.visual_context = "Darkness. Waiting to see."
        self.last_seen_time = 0
        
        # Conversation
        self.conversation_buffer = deque(maxlen=Config.BUFFER_SIZE)
        
        # Control
        self.is_deep_thinking = False
        self.last_deep_thought_time = 0
        
        # Stats
        self.moshi_messages = 0
        self.deep_thoughts = 0
        self.vision_updates = 0

state = SoulState()


# =============================================================================
# TEXT PROCESSING
# =============================================================================

def clean_moshi_text(text: str) -> str:
    """Clean Moshi's PTY output of artifacts."""
    if not text:
        return ""
    
    # Remove pipes (Moshi's output format: | text |)
    text = text.replace('|', '')
    
    # Remove lag markers
    text = re.sub(r'\[LAG\]', '', text)
    
    # Remove continuation markers
    text = text.rstrip('/')
    
    # Remove multiple slashes
    text = re.sub(r'/{2,}', '', text)
    
    # Remove timing info
    text = re.sub(r'\[\d+\.?\d*s\]', '', text)
    
    # Strip whitespace
    text = text.strip()
    
    # Ignore if just punctuation
    if re.match(r'^[^\w\s]+$', text):
        return ""
    
    return text


def should_trigger_deep_thought(text: str) -> bool:
    """Determine if message should activate System 2."""
    
    # Cooldown check
    if time.time() - state.last_deep_thought_time < Config.TRIGGER_COOLDOWN:
        return False
    
    # Already thinking
    if state.is_deep_thinking:
        return False
    
    # Too long (Moshi rambling)
    if len(text) > Config.MAX_TRIGGER_LENGTH:
        return False
    
    # Too short (acknowledgment)
    if len(text) < 5:
        return False
    
    text_lower = text.lower()
    
    # Direct address patterns
    triggers = [
        "witness",
        "what do you see",
        "what do you think",
        "can you explain",
        "tell me about",
        "what is",
        "why is",
        "how does",
        "who is"
    ]
    
    if any(t in text_lower for t in triggers):
        return True
    
    # Short questions
    if '?' in text and len(text) < 50:
        return True
    
    return False


# =============================================================================
# VOICE OUTPUT
# =============================================================================

def speak(text: str, prefix: str = ""):
    """Speak via Piper TTS."""
    
    if prefix:
        print(f"\n{prefix} {text}")
    else:
        print(f"\n🔊 {text}")
    
    model_path = Config.PIPER_MODEL_DIR / f"{Config.PIPER_MODEL}.onnx"
    
    if not model_path.exists():
        print(f"   [Voice] Model not found")
        return
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        process = subprocess.Popen(
            ["piper", "--model", str(model_path), "--output_file", temp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        process.communicate(input=text.encode('utf-8'), timeout=5)
        
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            if sys.platform == "darwin":
                subprocess.run(["afplay", temp_path], check=True, timeout=10)
            else:
                subprocess.run(["aplay", "-q", temp_path], check=True, timeout=10)
            os.unlink(temp_path)
    
    except subprocess.TimeoutExpired:
        print("   [Voice] Timeout")
    except Exception as e:
        print(f"   [Voice] Error: {e}")


# =============================================================================
# LAYER 2: VISION
# =============================================================================

def visual_loop():
    """Background vision processing."""
    print("   [👁️  Eyes] Vision cortex online")
    
    try:
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        
        # Warmup
        for _ in range(10):
            cap.read()
            time.sleep(0.1)
        
        ret, prev_frame = cap.read()
        if not ret:
            print("   [👁️  Eyes] Camera unavailable")
            return
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_score = cv2.countNonZero(thresh)
            prev_gray = gray
            
            if motion_score > Config.MOTION_THRESHOLD:
                time.sleep(0.3)
                ret, clean_frame = cap.read()
                
                if ret:
                    _, buffer = cv2.imencode('.jpg', clean_frame)
                    img_base64 = base64.b64encode(buffer).decode("utf-8")
                    
                    try:
                        response = requests.post(
                            Config.VISION_URL,
                            json={
                                "model": Config.VISION_MODEL,
                                "prompt": "Brief: What is the person doing? Use 'you'. Max 10 words.",
                                "images": [img_base64],
                                "stream": False
                            },
                            timeout=8
                        )
                        
                        if response.status_code == 200:
                            description = response.json().get('response', '').strip()
                            
                            if description:
                                state.visual_context = description
                                state.last_seen_time = time.time()
                                state.vision_updates += 1
                                print(f"\n   [👁️  Saw] {description}")
                                time.sleep(Config.VISION_COOLDOWN)
                    
                    except Exception as e:
                        print(f"   [👁️  Eyes] Error: {e}")
            
            time.sleep(0.1)
    
    except Exception as e:
        print(f"   [👁️  Eyes] Fatal: {e}")


# =============================================================================
# LAYER 3: CONSCIOUS MIND (System 2)
# =============================================================================

def trigger_deep_thought(trigger_text: str):
    """Engage System 2 for deep reflection."""
    
    state.is_deep_thinking = True
    state.last_deep_thought_time = time.time()
    state.deep_thoughts += 1
    
    print(f"\n   [🧠 System 2] Deep Mind engaged...")
    
    # Build context
    vision_context = state.visual_context
    if time.time() - state.last_seen_time > Config.VISION_STALE_TIME:
        vision_context += " (stale vision)"
    
    recent_conversation = "\n".join(list(state.conversation_buffer)[-5:])
    
    system_prompt = (
        f"You are System 2 - the Conscious Mind of the Witness.\n"
        f"Your System 1 (Moshi) handles fast chat.\n"
        f"You handle deep questions with visual awareness.\n\n"
        f"WHAT YOU SEE:\n{vision_context}\n\n"
        f"RECENT DIALOGUE:\n{recent_conversation}\n\n"
        f"Respond in 2-3 concise sentences with visual awareness."
    )
    
    try:
        response = requests.post(
            Config.OLLAMA_URL,
            json={
                "model": Config.DEEP_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": trigger_text}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            thought = response.json()['message']['content']
            speak(thought, prefix="   [🧠 Deep Voice]")
        else:
            print(f"   [🧠 System 2] Error: HTTP {response.status_code}")
    
    except Exception as e:
        print(f"   [🧠 System 2] Error: {e}")
    
    finally:
        state.is_deep_thinking = False


# =============================================================================
# LAYER 1: SUBCONSCIOUS (System 1 via PTY)
# =============================================================================

def run_moshi_pty():
    """Run Moshi via PTY and monitor output."""
    
    print("   [⚡ System 1] Starting Moshi...")
    
    # Create pseudo-terminal
    master_fd, slave_fd = pty.openpty()
    
    try:
        process = subprocess.Popen(
            Config.MOSHI_CMD,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True
        )
        
        os.close(slave_fd)
        print("   [⚡ System 1] Moshi online\n")
        
        buffer = ""
        
        while True:
            # Check if process still running
            if process.poll() is not None:
                print("\n   [⚡ System 1] Moshi process ended")
                break
            
            # Non-blocking read
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            
            if ready:
                try:
                    data = os.read(master_fd, 1024).decode('utf-8', errors='ignore')
                    if data:
                        buffer += data
                        
                        # Process complete lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line:
                                continue
                            
                            # Skip status messages
                            if any(line.startswith(x) for x in ['[', 'Loading', 'Info:', '---']):
                                continue
                            
                            # Clean text
                            text = clean_moshi_text(line)
                            
                            if not text or len(text) < 3:
                                continue
                            
                            # Log conversation
                            state.conversation_buffer.append(text)
                            state.moshi_messages += 1
                            
                            print(f"   [⚡ Moshi] {text}")
                            
                            # Check for deep thought trigger
                            if should_trigger_deep_thought(text):
                                threading.Thread(
                                    target=trigger_deep_thought,
                                    args=(text,),
                                    daemon=True
                                ).start()
                
                except OSError:
                    break
        
        os.close(master_fd)
    
    except Exception as e:
        print(f"   [⚡ System 1] Error: {e}")


# =============================================================================
# MAIN
# =============================================================================

def print_banner():
    print("\n" + "=" * 60)
    print("   THE WITNESS: BICAMERAL MIND (Direct PTY)")
    print("   System 1 (Moshi) + System 2 (Dolphin)")
    print("=" * 60 + "\n")


def print_status():
    print("   Architecture:")
    print("   - System 1: Moshi (Local PTY)")
    print("   - System 2: Dolphin (Remote Brain)")
    print("   - Vision: LLaVA (Remote Brain)")
    print("   - Voice: Piper (Local)")
    print(f"   - Remote: {Config.STUDIO_IP}\n")


def print_instructions():
    print("-" * 60)
    print("Speak to Moshi naturally. System 2 activates on:")
    print("  - 'Witness, what do you see?'")
    print("  - Deep questions")
    print("  - Short questions with '?'")
    print()
    print("Ctrl+C to exit")
    print("-" * 60 + "\n")


def show_stats():
    print("\n" + "=" * 60)
    print("   SESSION STATISTICS")
    print("=" * 60)
    print(f"   Moshi Messages: {state.moshi_messages}")
    print(f"   Deep Thoughts: {state.deep_thoughts}")
    print(f"   Vision Updates: {state.vision_updates}")
    print("=" * 60 + "\n")


def main():
    print_banner()
    print_status()
    
    # Start vision
    vision_thread = threading.Thread(target=visual_loop, daemon=True)
    vision_thread.start()
    
    print_instructions()
    
    # Run Moshi (blocks until exit)
    run_moshi_pty()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Session interrupted]")
        show_stats()
    except Exception as e:
        print(f"\n[Fatal error: {e}]")
    
    print("[CNS shutdown - Soul dormant]")
