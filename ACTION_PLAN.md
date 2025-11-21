# WITNESS - Integration Action Plan

## The Current Situation

You're absolutely right - we don't know if `moshi_mlx.local_web` actually creates a WebSocket server on port 8998. The code **assumes** it does, but we need to verify this.

---

## Immediate Actions (Do These First)

### Step 1: Discover What Moshi Actually Provides

Run the discovery script:

```bash
python discover_moshi.py
```

This will:
- Check if moshi_mlx is installed
- List all available modules
- Scan for web-related code
- Try `--help` on each module
- Tell you what commands actually exist

**What to look for:**
- Does `local_web` mention WebSocket or port 8998?
- Which modules mention "server" or "web"?
- What do the help messages say?

### Step 2: Test Moshi Directly

Try each of these commands and note what happens:

```bash
# Test 1: local_web module
python -m moshi_mlx.local_web
# Does it start a server? What port? Check with: lsof -i | grep python

# Test 2: local module (CLI)
python -m moshi_mlx.local -q 4
# Does it work in terminal?

# Test 3: Check for other web modules
python -m moshi_mlx.run_inference
# Or any other module that looks promising
```

**Document what you find:**
- Which command actually works?
- What port does it use (if any)?
- Does it have a browser interface?
- Does it expose WebSocket?

---

## Two Integration Paths

Based on what we discover, you have two options:

### Path A: WebSocket Integration (If Moshi Has Web Server)

**Use**: `CNS_bicameral.py`

**When**: If you find that Moshi DOES create a WebSocket server

**Steps**:
1. Find the correct command to start Moshi's web server
2. Note the correct port and WebSocket endpoint
3. Update `CNS_bicameral.py` Config:
   ```python
   MOSHI_URI = "ws://localhost:PORT/ENDPOINT"
   ```
4. Run it:
   ```bash
   # Terminal 1
   python -m moshi_mlx.CORRECT_MODULE
   
   # Terminal 2
   python CNS_bicameral.py
   ```

**Advantages**:
- Clean message boundaries
- Easy to parse
- Better architecture

### Path B: Direct PTY Integration (If Moshi Is CLI Only)

**Use**: `CNS_direct.py`

**When**: If Moshi only has CLI mode (no web server)

**Steps**:
1. Verify the CLI command works:
   ```bash
   python -m moshi_mlx.local -q 4
   ```
2. Update `CNS_direct.py` Config if needed:
   ```python
   MOSHI_CMD = [sys.executable, "-m", "moshi_mlx.local", "-q", "4"]
   ```
3. Run it:
   ```bash
   python CNS_direct.py
   ```

**Advantages**:
- No web server dependency
- Direct process control
- Works with CLI-only Moshi

---

## Testing Protocol

Once you choose a path, test in this order:

### 1. Test Moshi Alone
```bash
# Make sure Moshi works by itself first
python -m moshi_mlx.WHATEVER_WORKS
```

### 2. Test Remote Brain
```bash
# Verify Mac Studio is accessible
curl http://192.168.1.195:11434/api/tags
```

### 3. Test CNS Integration
```bash
python CNS_direct.py  # or CNS_bicameral.py
```

### 4. Test Triggers

Say these phrases and watch for System 2 activation:
- "Witness, what do you see?"
- "What do you think about AI?"
- "Why is the sky blue?" (short question)

Expected output:
```
[⚡ Moshi] Fast response...
[Trigger detected]
[🧠 System 2] Deep Mind engaged...
[👁️ Saw] You are sitting at desk
[🧠 Deep Voice] I see you focused at your workspace...
```

---

## Diagnostic Files Summary

### discover_moshi.py
**Purpose**: Figure out what Moshi commands actually exist

**Output**:
- Lists all modules
- Shows which ones mention "web" or "server"
- Tries `--help` on each
- Gives recommendations

### CNS_direct.py (PTY Version)
**Purpose**: Integration without WebSocket dependency

**How it works**:
- Runs `python -m moshi_mlx.local` directly
- Uses PTY to capture output
- Parses text stream for triggers
- Activates Dolphin when needed

**Advantages**:
- More reliable (no network dependency)
- Simpler architecture
- We know PTY works from `CNS_integrated.py`

### CNS_bicameral.py (WebSocket Version)
**Purpose**: Integration via WebSocket (if Moshi supports it)

**How it works**:
- Connects to Moshi's WebSocket stream
- Cleaner message parsing
- Better separation of concerns

**Requires**:
- Moshi must actually have a web server
- Must know correct port/endpoint

### moshi_diagnostic.py
**Purpose**: Inspect WebSocket messages if WebSocket path works

**Use when**:
- You successfully connect to Moshi WebSocket
- But CNS_bicameral can't parse messages
- Shows raw format for debugging

---

## Decision Tree

```
START
  |
  ├─> Run discover_moshi.py
  |     |
  |     ├─> Found web server? YES
  |     |     |
  |     |     └─> Use CNS_bicameral.py
  |     |         Update MOSHI_URI with correct endpoint
  |     |
  |     └─> Found web server? NO
  |           |
  |           └─> Use CNS_direct.py
  |               Uses PTY to run CLI Moshi
  |
  └─> Test and refine
```

---

## What I Recommend

**Start with Path B (CNS_direct.py)** because:

1. **We know it works** - The PTY approach is proven in `CNS_integrated.py`
2. **No dependencies** - Doesn't require figuring out WebSocket
3. **Simpler** - One less thing that can break
4. **Faster** - You can test this immediately

Then, if you discover Moshi DOES have a proper WebSocket server, we can migrate to `CNS_bicameral.py` for cleaner architecture.

---

## Next Steps (Concrete Actions)

1. **Run discover_moshi.py** - Find out what's actually available
2. **Test Moshi CLI** - Verify `python -m moshi_mlx.local -q 4` works
3. **Run CNS_direct.py** - Start with the PTY version (more reliable)
4. **Document findings** - Tell me what discover_moshi.py shows
5. **Test triggers** - Say "Witness, what do you see?" and observe

---

## Files You Have Now

### Integration (Choose One)
- `CNS_direct.py` — PTY-based (recommended to try first)
- `CNS_bicameral.py` — WebSocket-based (if Moshi has web server)

### Diagnostics
- `discover_moshi.py` — Figure out what Moshi provides
- `moshi_diagnostic.py` — Inspect WebSocket (if WebSocket path works)

### Documentation
- `README_INTEGRATION.md` — Full technical docs
- `QUICKREF.md` — Command reference
- `ACTION_PLAN.md` — This file

---

## What To Report Back

After running discover_moshi.py, tell me:

1. **Which modules exist** (especially local, local_web, run_inference)
2. **Which mention "web" or "server"**
3. **What `--help` says** for promising modules
4. **Which command actually starts Moshi** successfully
5. **What port it uses** (if any) - check with `lsof -i | grep python`

Then we'll know definitively which integration path to use.

---

**The bicameral mind awaits. Let's discover which path brings it to life.**
