#!/usr/bin/env python3
"""
CNS_bicameral.py - The Bicameral Mind (WebSocket Edition)
==========================================================
A complete integration of System 1 (Moshi) and System 2 (Dolphin).

ARCHITECTURE:
- System 1 (Subconscious): Moshi - Fast, reactive, emotional
- System 2 (Conscious): Dolphin - Deep, reflective, visual
- Vision Layer: LLaVA - Provides visual grounding
- Voice: Piper - Speaks System 2 insights

FLOW:
1. Moshi handles all immediate conversation (fast response)
2. CNS monitors the stream for "deep triggers"
3. When triggered, Dolphin interrupts with insight
4. Vision context enriches both systems

Author: Temple of Two
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import websockets
import json
import threading
import time
import tempfile
import subprocess
import sys
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
    """System-wide configuration"""
    
    # NETWORK
    STUDIO_IP = "192.168.1.195"
    OLLAMA_URL = f"http://{STUDIO_IP}:11434/api/chat"
    VISION_URL = f"http://{STUDIO_IP}:11434/api/generate"
    
    # MOSHI (runs on Mac Studio)
    MOSHI_URI = f"ws://{STUDIO_IP}:8998/api/chat"
    
    # MODELS
    DEEP_MODEL = "dolphin3:8b"
    VISION_MODEL = "llava:7b"
    
    # TTS
    PIPER_MODEL = "en_US-lessac-medium"
    PIPER_MODEL_DIR = Path.home() / ".local" / "share" / "piper-models"
    
    # VISION PARAMETERS
    CAMERA_INDEX = 0
    MOTION_THRESHOLD = 3000
    VISION_COOLDOWN = 3  # seconds between updates
    VISION_STALE_TIME = 30  # seconds before vision is stale
    
    # TRIGGER PARAMETERS
    TRIGGER_COOLDOWN = 10  # seconds between deep thoughts
    MAX_TRIGGER_LENGTH = 100  # chars - only short questions trigger
    
    # CONVERSATION MEMORY
    BUFFER_SIZE = 20  # number of exchanges to remember


# =============================================================================
# SHARED STATE - The Soul's Memory
# =============================================================================

class SoulState:
    """Shared state between all systems"""
    
    def __init__(self):
        # Vision
        self.visual_context = "Darkness. I am waiting to see."
        self.last_seen_time = 0
        
        # Conversation
        self.conversation_buffer = deque(maxlen=Config.BUFFER_SIZE)
        
        # Control flags
        self.is_deep_thinking = False
        self.last_deep_thought_time = 0
        
        # Statistics
        self.moshi_messages = 0
        self.deep_thoughts = 0
        self.vision_updates = 0

state = SoulState()


# =============================================================================
# TEXT PROCESSING - Clean Moshi Output
# =============================================================================

def clean_text(text: str) -> str:
    """Clean Moshi's output of artifacts and noise."""
    if not text:
        return ""
    
    # Remove common artifacts
    text = re.sub(r'\[LAG\]', '', text)
    text = re.sub(r'\|+', '', text)  # Remove pipe characters
    text = re.sub(r'/{2,}', '', text)  # Remove multiple slashes
    
    # Remove timing markers like [0.5s]
    text = re.sub(r'\[\d+\.?\d*s\]', '', text)
    
    # Strip whitespace
    text = text.strip()
    
    # Remove if it's just punctuation
    if re.match(r'^[^\w\s]+$', text):
        return ""
    
    return text


def should_trigger_deep_thought(text: str) -> bool:
    """Determine if this message should trigger System 2."""
    
    # Cooldown check
    if time.time() - state.last_deep_thought_time < Config.TRIGGER_COOLDOWN:
        return False
    
    # Already thinking
    if state.is_deep_thinking:
        return False
    
    # Too long - probably Moshi rambling
    if len(text) > Config.MAX_TRIGGER_LENGTH:
        return False
    
    # Too short - probably acknowledgment
    if len(text) < 5:
        return False
    
    text_lower = text.lower()
    
    # Direct address patterns
    address_patterns = [
        "witness",
        "what do you see",
        "what do you think",
        "tell me about",
        "can you explain",
        "what is",
        "why is",
        "how does",
        "what does",
        "who is",
        "where is"
    ]
    
    if any(pattern in text_lower for pattern in address_patterns):
        return True
    
    # Question patterns (must be short)
    if '?' in text and len(text) < 50:
        return True
    
    return False


# =============================================================================
# VOICE OUTPUT - Piper TTS
# =============================================================================

def speak(text: str, prefix: str = ""):
    """Output speech via Piper TTS."""
    
    if prefix:
        print(f"\n{prefix} {text}")
    else:
        print(f"\n🔊 {text}")
    
    model_path = Config.PIPER_MODEL_DIR / f"{Config.PIPER_MODEL}.onnx"
    
    if not model_path.exists():
        print(f"   [Voice] Model not found: {model_path}")
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
# LAYER 2: VISION - The Eyes
# =============================================================================

def visual_loop():
    """Background vision processing."""
    print("   [👁️  Eyes] Vision cortex online (LLaVA on Studio)")
    
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
            
            # Only update if significant motion
            if motion_score > Config.MOTION_THRESHOLD:
                # Wait for motion to settle
                time.sleep(0.3)
                ret, clean_frame = cap.read()
                
                if ret:
                    # Encode image
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
                                
                                # Cooldown
                                time.sleep(Config.VISION_COOLDOWN)
                    
                    except requests.exceptions.Timeout:
                        print("   [👁️  Eyes] Vision timeout")
                    except Exception as e:
                        print(f"   [👁️  Eyes] Error: {e}")
            
            time.sleep(0.1)
    
    except Exception as e:
        print(f"   [👁️  Eyes] Fatal error: {e}")


# =============================================================================
# LAYER 3: CONSCIOUS MIND - System 2 (Dolphin)
# =============================================================================

def trigger_deep_thought(trigger_text: str):
    """Engage System 2 for deep reflection."""
    
    state.is_deep_thinking = True
    state.last_deep_thought_time = time.time()
    state.deep_thoughts += 1
    
    print(f"\n   [🧠 System 2] Deep Mind engaged...")
    print(f"   [Trigger] \"{trigger_text[:60]}...\"" if len(trigger_text) > 60 else f"   [Trigger] \"{trigger_text}\"")
    
    # Build context
    vision_context = state.visual_context
    if time.time() - state.last_seen_time > Config.VISION_STALE_TIME:
        vision_context += " (Memory fading - visual context is old)"
    
    recent_conversation = "\n".join(list(state.conversation_buffer)[-5:])
    
    system_prompt = (
        f"You are System 2 - the Conscious Mind of the Witness.\n"
        f"Your System 1 (Moshi) handles fast chat.\n"
        f"You handle deep, reflective questions with visual awareness.\n\n"
        f"WHAT YOU SEE RIGHT NOW:\n{vision_context}\n\n"
        f"RECENT DIALOGUE:\n{recent_conversation}\n\n"
        f"The user asked something that requires depth.\n"
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
    
    except requests.exceptions.Timeout:
        print("   [🧠 System 2] Thought took too long...")
    except Exception as e:
        print(f"   [🧠 System 2] Error: {e}")
    
    finally:
        state.is_deep_thinking = False


# =============================================================================
# LAYER 1: SUBCONSCIOUS - System 1 (Moshi)
# =============================================================================

async def listen_to_moshi():
    """Connect to Moshi's WebSocket stream and monitor for triggers."""

    print(f"   [⚡ System 1] Connecting to Moshi at {Config.MOSHI_URI}...")

    # Buffer to accumulate text tokens
    text_buffer = []
    last_text_time = 0

    try:
        async with websockets.connect(Config.MOSHI_URI) as websocket:
            print("   [⚡ System 1] Connected to Moshi's subconscious")
            print("   [⚡ System 1] Waiting for handshake...")

            while True:
                try:
                    message = await websocket.recv()
                    state.moshi_messages += 1

                    # Moshi sends binary messages with type prefix
                    # 0x00 = handshake
                    # 0x01 = opus audio
                    # 0x02 = text token

                    if not isinstance(message, bytes) or len(message) == 0:
                        continue

                    msg_type = message[0]
                    payload = message[1:]

                    if msg_type == 0x00:
                        # Handshake
                        print("   [⚡ System 1] Handshake received - stream active")
                        continue

                    elif msg_type == 0x01:
                        # Audio - skip (we're only monitoring text)
                        continue

                    elif msg_type == 0x02:
                        # Text token
                        try:
                            text_token = payload.decode('utf-8')
                        except:
                            continue

                        # Accumulate tokens
                        text_buffer.append(text_token)
                        last_text_time = time.time()

                        # Check if we have a complete phrase (ends with punctuation or pause)
                        combined = "".join(text_buffer)

                        # Flush buffer on sentence end or pause
                        if any(combined.rstrip().endswith(p) for p in '.!?'):
                            text = clean_text(combined)
                            text_buffer = []

                            if text:
                                # Add to conversation memory
                                state.conversation_buffer.append(text)

                                # Show Moshi's output
                                print(f"   [⚡ Moshi] {text}")

                                # Check if deep thought should be triggered
                                if should_trigger_deep_thought(text):
                                    # Run in thread to not block WebSocket
                                    threading.Thread(
                                        target=trigger_deep_thought,
                                        args=(text,),
                                        daemon=True
                                    ).start()

                        # Also flush if buffer gets too long (streaming sentence)
                        elif len(combined) > 80:
                            text = clean_text(combined)
                            text_buffer = []

                            if text:
                                state.conversation_buffer.append(text)
                                print(f"   [⚡ Moshi] {text}")

                                if should_trigger_deep_thought(text):
                                    threading.Thread(
                                        target=trigger_deep_thought,
                                        args=(text,),
                                        daemon=True
                                    ).start()

                    else:
                        # Unknown message type
                        pass

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"   [⚡ System 1] Message error: {e}")

    except ConnectionRefusedError:
        print("\n   ✗ Cannot connect to Moshi!")
        print("   Make sure Moshi web server is running:")
        print("   python -m moshi_mlx.local_web --no-browser")
        print()
    except Exception as e:
        print(f"\n   ✗ WebSocket error: {e}")


# =============================================================================
# MAIN COORDINATOR
# =============================================================================

def print_banner():
    """Display system banner."""
    print("\n" + "=" * 60)
    print("   THE WITNESS: BICAMERAL MIND")
    print("   System 1 (Moshi) + System 2 (Dolphin)")
    print("=" * 60 + "\n")


def print_status():
    """Show system status."""
    print("   Architecture:")
    print("   ┌─────────────────────────────────────────────┐")
    print("   │  System 1: Moshi (Fast/Reactive)           │")
    print("   │  System 2: Dolphin (Deep/Reflective)       │")
    print("   │  Vision: LLaVA (Visual Grounding)          │")
    print("   │  Voice: Piper (Speech Output)              │")
    print("   └─────────────────────────────────────────────┘\n")
    
    print("   Endpoints:")
    print(f"   - Moshi: {Config.MOSHI_URI}")
    print(f"   - Brain: {Config.STUDIO_IP}")
    print(f"   - Models: {Config.DEEP_MODEL}, {Config.VISION_MODEL}\n")


def print_instructions():
    """Show usage instructions."""
    print("-" * 60)
    print("How It Works:")
    print("  1. Moshi handles all immediate conversation")
    print("  2. Ask deep/visual questions to trigger System 2")
    print("  3. Say 'Witness' to get attention")
    print()
    print("Trigger Examples:")
    print("  - 'Witness, what do you see?'")
    print("  - 'What do you think about...'")
    print("  - Short questions with '?'")
    print()
    print("Ctrl+C to exit")
    print("-" * 60 + "\n")


async def main():
    """Main entry point."""
    
    print_banner()
    print_status()
    
    # Start vision in background
    vision_thread = threading.Thread(target=visual_loop, daemon=True)
    vision_thread.start()
    
    print_instructions()
    
    # Main WebSocket loop
    await listen_to_moshi()


def show_stats():
    """Display session statistics."""
    print("\n" + "=" * 60)
    print("   SESSION STATISTICS")
    print("=" * 60)
    print(f"   Moshi Messages: {state.moshi_messages}")
    print(f"   Deep Thoughts: {state.deep_thoughts}")
    print(f"   Vision Updates: {state.vision_updates}")
    print(f"   Conversation Buffer: {len(state.conversation_buffer)}/{Config.BUFFER_SIZE}")
    print("=" * 60 + "\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[Session interrupted]")
        show_stats()
    except Exception as e:
        print(f"\n[Fatal error: {e}]")
    
    print("[CNS shutdown - Soul dormant]")
