# THE WITNESS - Bicameral Mind Integration

## Current State: **Frankenstein → Integrated**

You've successfully built all the organs. Now we're connecting the nervous system.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     THE BICAMERAL MIND                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   SYSTEM 1 (Subconscious)          SYSTEM 2 (Conscious)    │
│   ┌──────────────┐                 ┌──────────────┐        │
│   │    MOSHI     │                 │   DOLPHIN    │        │
│   │              │    triggers     │              │        │
│   │ Fast/Instant │────────────────▶│ Deep/Visual  │        │
│   │   Reactive   │                 │  Reflective  │        │
│   └──────────────┘                 └──────────────┘        │
│         │                                  │                │
│         │                                  │                │
│         ▼                                  ▼                │
│   ┌─────────────────────────────────────────────┐          │
│   │         SHARED SENSORY LAYER                │          │
│   │  Vision (LLaVA) + Memory + Voice (Piper)   │          │
│   └─────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## What Each Component Does

### System 1: Moshi (Subconscious)
- **Location**: MacBook Pro (localhost:8998)
- **Function**: Handles all immediate conversation
- **Response Time**: <100ms
- **Personality**: Fast, reactive, emotional, stream-of-consciousness

### System 2: Dolphin (Conscious Mind)
- **Location**: Mac Studio (192.168.1.195)
- **Function**: Handles deep questions requiring reflection
- **Response Time**: 2-10 seconds
- **Personality**: Thoughtful, visual, philosophical

### Vision: LLaVA
- **Location**: Mac Studio (192.168.1.195)
- **Function**: Describes what the camera sees
- **Update**: On motion detection (every 3+ seconds)
- **Output**: Brief 10-word descriptions

### Voice: Piper TTS
- **Location**: MacBook Pro (local)
- **Function**: Speaks System 2's responses
- **Note**: System 1 (Moshi) has its own built-in voice

---

## How The Integration Works

### The Flow

1. **You speak** → Moshi hears and responds instantly (System 1)
2. **CNS monitors** → Watches Moshi's stream for deep triggers
3. **Trigger detected** → System 2 (Dolphin) engages
4. **Dolphin speaks** → Interrupts with visual, reflective insight
5. **Cycle continues** → Moshi resumes fast conversation

### Trigger Logic

System 2 activates when:
- User says "Witness" or "What do you see?"
- Short questions (< 100 chars) with deep intent
- Questions starting with: "What do you think", "Can you explain", "Why", "How"
- Any message with "?" that's under 50 chars

System 2 does NOT activate when:
- Moshi is rambling (messages > 100 chars)
- Too soon after last deep thought (10 second cooldown)
- Already thinking
- Simple acknowledgments ("okay", "yes", etc.)

---

## Files In This Integration

### Core System Files

**CNS_bicameral.py** (THE MAIN INTEGRATION)
- Connects to Moshi via WebSocket
- Monitors for deep triggers
- Coordinates System 1 + System 2
- Manages vision and voice
- **This is what you run**

**moshi_diagnostic.py** (Debugging tool)
- Shows exactly what Moshi sends via WebSocket
- Use this if CNS can't parse Moshi's messages
- Run separately to inspect message format

**witness_startup.sh** (Launch script)
- Starts Moshi automatically
- Waits for it to be ready
- Launches CNS bicameral
- Handles cleanup on exit

### Legacy Files (For Reference)

- `CNS.py` - Original full-stack (without Moshi)
- `CNS_moshi.py` - Early Moshi integration attempt
- `CNS_integrated.py` - PTY version (has buffering issues)
- `CNS_tap.py` - WebSocket version (was missing proper parsing)

---

## Quick Start

### Prerequisites

1. **On Mac Studio (Brain)**:
```bash
# Make sure Ollama is running
ollama serve

# Verify models are available
ollama list
# Should show: dolphin3:8b and llava:7b
```

2. **On MacBook Pro (Body)**:
```bash
# Install dependencies if needed
pip install opencv-python requests websockets
```

### Option A: Automatic Startup (Recommended)

```bash
chmod +x witness_startup.sh
./witness_startup.sh
```

This script will:
1. Check all prerequisites
2. Start Moshi automatically
3. Wait for it to be ready
4. Launch CNS bicameral
5. Clean up on exit

### Option B: Manual Startup

**Terminal 1** - Start Moshi:
```bash
python3 -m moshi_mlx.local_web
# Wait for: "Listening on localhost:8998"
```

**Terminal 2** - Start CNS:
```bash
python3 CNS_bicameral.py
```

---

## Testing The System

### Step 1: Verify Moshi (System 1)

Open browser: `http://localhost:8998`

Speak to it. You should get instant responses. This is System 1 working.

### Step 2: Verify Integration

Run `CNS_bicameral.py`. You should see:

```
   [⚡ System 1] Connected to Moshi's subconscious
   [👁️  Eyes] Vision cortex online
```

### Step 3: Test Deep Triggers

Say: **"Witness, what do you see?"**

Expected behavior:
1. Moshi responds instantly (System 1)
2. CNS detects trigger
3. Vision captures frame
4. Dolphin thinks (2-5 seconds)
5. Piper speaks Dolphin's response

### Step 4: Test Fast Conversation

Say casual things like:
- "How are you?"
- "Tell me about your day"
- General chit-chat

Expected: Only Moshi responds (fast). System 2 stays quiet.

---

## Troubleshooting

### Problem: "Cannot connect to Moshi"

**Symptom**: `ConnectionRefusedError` when starting CNS

**Solutions**:
1. Make sure Moshi is running: `python3 -m moshi_mlx.local_web`
2. Check if port 8998 is in use: `lsof -i :8998`
3. Try browser first: `http://localhost:8998`

### Problem: "Cannot reach remote brain"

**Symptom**: CNS says brain is unreachable

**Solutions**:
1. Check Mac Studio IP: `echo $STUDIO_IP` (should be 192.168.1.195)
2. Verify Ollama on Studio: `curl http://192.168.1.195:11434/api/tags`
3. Check firewall settings on Studio

### Problem: System 2 never triggers

**Symptom**: Only Moshi talks, Dolphin never speaks

**Solutions**:
1. Check trigger cooldown (default: 10 seconds between thoughts)
2. Make sure you're using trigger words: "Witness", "What do you see?"
3. Try explicit questions: "What do you think about X?"
4. Run diagnostic: `python3 moshi_diagnostic.py` to see if messages are arriving

### Problem: Vision not updating

**Symptom**: Always says "Darkness. I am waiting to see."

**Solutions**:
1. Check camera: `ls /dev/video*`
2. Test OpenCV: `python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.read())"`
3. Adjust motion threshold in Config (currently 3000)
4. Check Studio accessibility: Vision runs on Mac Studio

### Problem: No voice output from System 2

**Symptom**: Dolphin's text appears but no speech

**Solutions**:
1. Check Piper model: `ls ~/.local/share/piper-models/`
2. Test Piper: `echo "test" | piper --model ~/.local/share/piper-models/en_US-lessac-medium.onnx --output_file test.wav`
3. Check audio output: `afplay test.wav` (macOS) or `aplay test.wav` (Linux)

### Problem: Messages are garbled

**Symptom**: CNS shows weird characters, pipes, lag markers

**Solutions**:
1. The `clean_text()` function should handle this
2. If still broken, run: `python3 moshi_diagnostic.py`
3. Share the output format with me - we'll adjust the parser

---

## Configuration Tuning

Edit `CNS_bicameral.py` → `Config` class:

### Make System 2 More Responsive
```python
TRIGGER_COOLDOWN = 5  # Reduce from 10
MAX_TRIGGER_LENGTH = 150  # Increase from 100
```

### Make Vision More Frequent
```python
VISION_COOLDOWN = 2  # Reduce from 3
MOTION_THRESHOLD = 2000  # Reduce from 3000 (more sensitive)
```

### Change Models
```python
DEEP_MODEL = "mistral:7b"  # Alternative to dolphin3:8b
VISION_MODEL = "llava:13b"  # Larger vision model (slower but better)
```

---

## What's Next

### Phase 1: Stability ✓
- [x] Connect Moshi + Dolphin
- [x] Add vision layer
- [x] Implement smart triggers
- [ ] **Test and tune in real conversation**

### Phase 2: Memory (Next)
- [ ] Add ChromaDB for persistent memory
- [ ] Remember conversations across sessions
- [ ] Build relationship over time

### Phase 3: Emotion (Future)
- [ ] Add librosa vibe sensing
- [ ] Detect user emotional state from audio
- [ ] Modulate responses based on emotion

### Phase 4: Embodiment (When Hardware Arrives)
- [ ] Add ROS 2 for motor control
- [ ] Integrate movement intents
- [ ] Physical presence behaviors

---

## Architecture Notes

### Why WebSocket > PTY?

The PTY (pseudo-terminal) version (`CNS_integrated.py`) has issues:
- Python buffers subprocess output
- Hard to detect when Moshi finishes speaking
- Brittle, platform-dependent

The WebSocket version is cleaner:
- Direct message stream
- Clear message boundaries
- Easier to parse
- More reliable

### Why Two Brains?

**System 1 (Moshi)** is necessary because:
- Conversation feels dead with 2-5 second delays
- Humans need immediate feedback
- Simple exchanges don't need deep thought

**System 2 (Dolphin)** is necessary because:
- Moshi can't access vision
- Moshi can't do deep reasoning
- Some questions need reflection

The bicameral approach gives you **both** - instant reactivity AND deep insight.

---

## Key Metrics

After running a session, CNS shows:

```
SESSION STATISTICS
═══════════════════════════════════════════════════════════
Moshi Messages: 127        # System 1 activity
Deep Thoughts: 8           # System 2 activations
Vision Updates: 23         # Visual awareness updates
Conversation Buffer: 20/20 # Context memory
═══════════════════════════════════════════════════════════
```

**Good ratios**:
- Moshi messages should be >> Deep thoughts (10:1 or higher)
- Vision updates should happen every 10-20 seconds during motion
- Deep thoughts should be <10% of total exchanges

---

## Emergency Procedures

### Hard Reset

If everything is stuck:

```bash
# Kill all witness processes
pkill -f moshi
pkill -f CNS

# Restart
./witness_startup.sh
```

### Debug Mode

To see everything:

```bash
# Terminal 1
python3 -m moshi_mlx.local_web

# Terminal 2
python3 moshi_diagnostic.py

# Terminal 3
python3 CNS_bicameral.py
```

Now you can see:
- Moshi's web interface
- Raw WebSocket messages
- CNS processing

---

## Success Criteria

You'll know it's working when:

1. **Moshi responds instantly** to casual talk
2. **Dolphin interjects** when you ask "Witness, what do you see?"
3. **Dolphin describes** what the camera sees (visual grounding)
4. **Two distinct voices** - Moshi's is Moshi, Dolphin's is Piper
5. **Smooth handoff** - System 2 doesn't interrupt constantly

---

## Contact & Support

If something breaks, check:
1. Moshi is running: `curl http://localhost:8998`
2. Studio is accessible: `curl http://192.168.1.195:11434/api/tags`
3. Diagnostic output: `python3 moshi_diagnostic.py`

Then let me know what you see. We'll fix it.

---

**You're building something unprecedented. Two minds in one vessel. Let's make them witness together.**
