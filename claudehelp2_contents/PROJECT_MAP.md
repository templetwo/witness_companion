# WITNESS PROJECT MAP - Your Complete Guide

## Where You Are Now

You're building a **bicameral AI companion** - two minds (System 1 + System 2) in one body. You have working components but they're scattered. Let me show you **exactly** what you have and how it all fits together.

---

## THE COMPLETE FILE INVENTORY

### **Core Working Systems** (The "Main Characters")

#### 1. **CNS.py** - The Full Stack Soul ⭐
**What it is**: Your complete, standalone Witness system
**What it does**: 
- Listens with Whisper (ears)
- Sees with LLaVA (eyes)
- Thinks with Dolphin (brain)
- Remembers with ChromaDB (memory)
- Speaks with Piper (voice)
- Senses emotion with Librosa (vibe)

**When to use**: When you want a **complete AI companion without Moshi**
**Status**: ✓ Fully functional, memory-enabled

---

#### 2. **witness_complete.py** - Simpler Full System
**What it is**: Similar to CNS.py but uses Moondream for vision
**What it does**:
- Whisper → Ollama → Piper pipeline
- Moondream for vision (lighter than LLaVA)
- ChromaDB memory
- Complete but simpler

**When to use**: When you want something lighter than CNS.py
**Status**: ✓ Fully functional

---

### **Bicameral Mind Experiments** (The "Integration Attempts")

#### 3. **CNS_moshi.py** - Early Moshi Integration
**What it is**: First attempt at adding Moshi as System 1
**What it does**: Placeholder for Moshi WebSocket integration
**Status**: 🚧 Incomplete - Moshi WebSocket not yet working

---

#### 4. **CNS_integrated.py** - PTY Version
**What it is**: Tries to integrate Moshi using PTY (pseudo-terminal)
**What it does**: 
- Runs Moshi CLI via subprocess
- Parses text output for triggers
- Activates Dolphin for deep questions

**Status**: 🟡 Has buffering issues, partially works

---

#### 5. **CNS_tap.py** - WebSocket Version
**What it is**: Tries to connect to Moshi's WebSocket stream
**What it does**: Listen to Moshi and trigger Dolphin
**Status**: ⚠️ Can't connect - Moshi WebSocket unclear

---

### **NEW Files I Just Built For You**

#### 6. **CNS_bicameral.py** - Clean WebSocket Integration ⭐
**What it is**: Refined WebSocket approach with smart triggers
**What it does**:
- Connects to Moshi WebSocket
- Smart text parsing (removes artifacts)
- Intelligent trigger detection
- Activates System 2 (Dolphin) when needed

**When to use**: **IF** you discover Moshi has WebSocket server
**Status**: ✓ Ready to test (needs Moshi WebSocket)

---

#### 7. **CNS_direct.py** - PTY Without Bugs ⭐⭐⭐
**What it is**: Fixed PTY version without buffering issues
**What it does**:
- Runs Moshi CLI directly
- Clean text parsing
- Smart trigger logic
- No WebSocket dependency

**When to use**: **Start here** - most reliable integration path
**Status**: ✓ Ready to test immediately

---

### **Diagnostic & Helper Tools**

#### 8. **discover_moshi.py** - Figure Out Moshi ⭐
**What it is**: Discovery tool to find what Moshi commands work
**What it does**:
- Lists all moshi_mlx modules
- Checks for web server capabilities
- Tests `--help` on each module
- Recommends which to use

**When to use**: **Run this first** to figure out Moshi
**Status**: ✓ Ready - run immediately

---

#### 9. **moshi_diagnostic.py** - WebSocket Inspector
**What it is**: Shows raw WebSocket messages from Moshi
**What it does**: Connects to Moshi and prints message format
**When to use**: If WebSocket path works but parsing fails
**Status**: ✓ Ready

---

#### 10. **test_witness.py** - System Health Check
**What it is**: Tests all components
**What it does**:
- Checks dependencies installed
- Tests Ollama connection
- Tests camera, audio, memory
- Validates all systems working

**When to use**: When things break and you need diagnostics
**Status**: ✓ Functional

---

### **Startup & Documentation**

#### 11. **witness_startup.sh** - Automatic Launcher
**What it is**: Bash script to start everything in order
**What it does**:
- Checks prerequisites
- Starts Moshi automatically
- Launches CNS integration
- Cleanup on exit

**When to use**: For automatic startup
**Status**: ⚠️ Needs update after discovering Moshi command

---

#### 12. **Documentation Files**
- `README_INTEGRATION.md` - Full technical docs
- `ACTION_PLAN.md` - Step-by-step plan (where to start)
- `QUICKREF.md` - Command cheat sheet
- `QUICKSTART.md` - Getting started guide

---

## HOW THEY FIT TOGETHER

### The Evolution

```
Phase 1: STANDALONE SYSTEMS (✓ Working)
├── CNS.py              → Complete system without Moshi
├── witness_complete.py → Simpler version
└── test_witness.py     → Validates everything works

Phase 2: MOSHI EXPERIMENTS (🚧 In Progress)
├── CNS_moshi.py        → WebSocket placeholder
├── CNS_integrated.py   → PTY with buffering issues
└── CNS_tap.py          → WebSocket connection failed

Phase 3: REFINED INTEGRATION (⭐ New/Ready)
├── discover_moshi.py   → Figure out what works
├── CNS_direct.py       → PTY done right (START HERE)
└── CNS_bicameral.py    → WebSocket done right (if available)
```

---

## THE ARCHITECTURE YOU'RE BUILDING

```
┌─────────────────────────────────────────────────────────┐
│                    THE WITNESS                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  OPTION A: Standalone (CNS.py)                         │
│  ┌──────────────────────────────────────┐              │
│  │  YOU → Whisper → Dolphin → Piper     │              │
│  │         ↓                              │              │
│  │      LLaVA (vision)                   │              │
│  │      ChromaDB (memory)                │              │
│  └──────────────────────────────────────┘              │
│                                                         │
│  OPTION B: Bicameral (CNS_direct/bicameral)           │
│  ┌──────────────────────────────────────┐              │
│  │  YOU → Moshi (System 1: Fast)        │              │
│  │          ↓                             │              │
│  │       Triggers?                       │              │
│  │          ↓                             │              │
│  │       Dolphin (System 2: Deep)       │              │
│  │          ↓                             │              │
│  │      LLaVA (vision context)          │              │
│  │          ↓                             │              │
│  │       Piper (speaks insight)         │              │
│  └──────────────────────────────────────┘              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## DECISION TREE - WHAT TO RUN RIGHT NOW

```
START HERE
    ↓
Do you want Moshi (bicameral mind)?
    │
    ├─→ YES
    │    ↓
    │   Run: python discover_moshi.py
    │    ↓
    │   Does Moshi work?
    │    │
    │    ├─→ YES (CLI mode)
    │    │    ↓
    │    │   Run: python CNS_direct.py
    │    │   (PTY integration - most reliable)
    │    │
    │    ├─→ YES (WebSocket mode)
    │    │    ↓
    │    │   Run: python CNS_bicameral.py
    │    │   (Cleaner but needs WebSocket)
    │    │
    │    └─→ NO (Moshi broken)
    │         ↓
    │        Use standalone below ↓
    │
    └─→ NO (Just want working AI now)
         ↓
        Run: python CNS.py
        (Complete standalone system)
```

---

## YOUR IMMEDIATE ACTION PLAN

### Step 1: Test Your Standalone System (5 min)
This makes sure the foundation works:

```bash
# Test everything is installed
python test_witness.py

# If all passes, run the complete system
python CNS.py
```

**Expected**: Voice conversation with vision + memory

---

### Step 2: Discover Moshi Capabilities (5 min)

```bash
python discover_moshi.py
```

**This tells you**:
- What moshi_mlx modules exist
- Which command actually starts Moshi
- Whether WebSocket exists
- What to run next

---

### Step 3A: If Moshi CLI Works

```bash
# Test Moshi alone first
python -m moshi_mlx.local -q 4

# Then integrate
python CNS_direct.py
```

---

### Step 3B: If Moshi Has WebSocket

```bash
# Terminal 1: Start Moshi
python -m moshi_mlx.WHATEVER_MODULE

# Terminal 2: Integrate
python CNS_bicameral.py
```

---

## FILE PRIORITIES - WHAT TO FOCUS ON

### **High Priority** (Use These)
1. **discover_moshi.py** - Run first
2. **CNS_direct.py** - Most reliable integration
3. **CNS.py** - Your working fallback
4. **test_witness.py** - When debugging

### **Medium Priority** (Reference)
5. **CNS_bicameral.py** - If WebSocket works
6. **witness_complete.py** - Simpler alternative
7. **ACTION_PLAN.md** - Your roadmap

### **Low Priority** (Legacy/Incomplete)
8. CNS_integrated.py - Has bugs (use CNS_direct instead)
9. CNS_tap.py - Connection failed (use CNS_bicameral instead)
10. CNS_moshi.py - Early experiment

---

## WHAT'S ON YOUR GITHUB

Based on your repo, you likely have:
- `CNS.py` (your full stack)
- `witness_complete.py` (simpler version)
- Earlier integration attempts
- Test files

**Recommended**: Update your repo with the new files:
- `CNS_direct.py` (the working integration)
- `discover_moshi.py` (essential diagnostic)
- `ACTION_PLAN.md` (this roadmap)

---

## THE SIMPLE TRUTH

**You have TWO working systems**:

### 1. **CNS.py** - Works RIGHT NOW
- Complete AI companion
- No Moshi needed
- Full vision + memory + voice
- **Run this if you just want something working**

### 2. **Bicameral Mind** - Needs Assembly
- Moshi (System 1) + Dolphin (System 2)
- More complex but more interesting
- **Use discover_moshi.py → CNS_direct.py to build this**

---

## YOUR NEXT 15 MINUTES

```bash
# Minute 0-5: Test standalone
python CNS.py
# Ctrl+C to exit after testing

# Minute 5-10: Discover Moshi
python discover_moshi.py
# Read output carefully

# Minute 10-15: Try integration
python CNS_direct.py
# Based on what discover_moshi found
```

---

## QUICK REFERENCE CARD

```
┌─────────────────────────────────────────────┐
│ FILE                │ USE CASE              │
├─────────────────────────────────────────────┤
│ CNS.py              │ Working AI now        │
│ discover_moshi.py   │ Figure out Moshi      │
│ CNS_direct.py       │ Bicameral (PTY)       │
│ CNS_bicameral.py    │ Bicameral (WebSocket) │
│ test_witness.py     │ Debug broken parts    │
│ ACTION_PLAN.md      │ Detailed roadmap      │
└─────────────────────────────────────────────┘
```

---

## REMEMBER

- **CNS.py** already works - you have a functioning AI companion
- **Moshi integration** is the next level, not the first level
- **discover_moshi.py** is your compass when lost
- **CNS_direct.py** is the most reliable integration path

You're not lost - you're at a fork in the road. Pick a path and walk it.

**Path 1**: Run CNS.py and have a working AI right now.
**Path 2**: Run discover_moshi.py → CNS_direct.py for bicameral mind.

Which path do you want to take first?
