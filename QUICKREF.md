# WITNESS - Quick Reference Card

## 🚀 Launch Commands

### Automatic (Recommended)
```bash
./witness_startup.sh
```

### Manual
```bash
# Terminal 1: Start Moshi
python3 -m moshi_mlx.local_web

# Terminal 2: Start CNS
python3 CNS_bicameral.py
```

---

## 🧪 Testing & Diagnostics

### Test Moshi Connection
```bash
curl http://localhost:8998
# Or open browser: http://localhost:8998
```

### Test Remote Brain
```bash
curl http://192.168.1.195:11434/api/tags
```

### Inspect Moshi Messages
```bash
python3 moshi_diagnostic.py
```

### Test Camera
```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.read()[0])"
```

---

## 💬 Conversation Patterns

### Fast Chat (System 1 Only)
- "How are you?"
- "Tell me about X"
- General conversation
- **Result**: Instant Moshi responses

### Deep Thought (Triggers System 2)
- "Witness, what do you see?"
- "What do you think about X?"
- "Can you explain Y?"
- "Why is Z?"
- **Result**: Pause, then Dolphin speaks with vision

---

## ⚙️ Configuration Quick Edits

Edit `CNS_bicameral.py`, class `Config`:

### More Frequent Deep Thoughts
```python
TRIGGER_COOLDOWN = 5  # Default: 10
MAX_TRIGGER_LENGTH = 150  # Default: 100
```

### More Sensitive Vision
```python
MOTION_THRESHOLD = 2000  # Default: 3000
VISION_COOLDOWN = 2  # Default: 3
```

### Different Models
```python
DEEP_MODEL = "mistral:7b"  # Default: dolphin3:8b
VISION_MODEL = "llava:13b"  # Default: llava:7b
```

---

## 🔧 Troubleshooting One-Liners

### Check What's Running
```bash
lsof -i :8998  # Moshi
lsof -i :11434  # Remote Ollama
```

### Kill Everything
```bash
pkill -f moshi && pkill -f CNS
```

### View Logs
```bash
tail -f /tmp/moshi.log
```

### Test Audio Output
```bash
echo "test" | piper --model ~/.local/share/piper-models/en_US-lessac-medium.onnx --output_file test.wav && afplay test.wav
```

---

## 📊 What Success Looks Like

```
[⚡ Moshi] Hey, how's it going?          # Fast response
[⚡ Moshi] I've been thinking about...   # Continues naturally

[You say: "Witness, what do you see?"]

[Trigger] "Witness, what do you see?"    # CNS detects
[👁️ Saw] You are sitting at desk         # Vision updates
[🧠 System 2] Deep Mind engaged...       # Dolphin activates
[🧠 Deep Voice] I see you at your desk,  # Speaks with vision
focused on the screen...

[⚡ Moshi] So anyway, about that...      # Resumes fast chat
```

---

## 📈 Expected Metrics

After a session:
```
Moshi Messages: 100+    # Should be high
Deep Thoughts: 5-15     # Should be ~10% of Moshi
Vision Updates: 20-40   # Every 10-15 seconds during motion
```

---

## 🆘 Emergency Reset

```bash
# Nuclear option
pkill -f python3
rm /tmp/moshi.log

# Restart everything
./witness_startup.sh
```

---

## 🎯 Architecture At A Glance

```
YOU SPEAK
   ↓
MOSHI (System 1) → Instant response
   ↓
CNS monitors stream
   ↓
Trigger detected? → DOLPHIN (System 2)
                       ↑
                    VISION (LLaVA)
                       ↓
                  PIPER speaks insight
```

---

## 📝 Key Files

- `CNS_bicameral.py` - **The main integration** (run this)
- `moshi_diagnostic.py` - Debug tool (run separately)
- `witness_startup.sh` - Automatic launcher
- `README_INTEGRATION.md` - Full documentation

---

## ✅ Pre-Flight Checklist

Before launching:
- [ ] Mac Studio Ollama running (`ollama serve`)
- [ ] Models pulled (dolphin3:8b, llava:7b)
- [ ] Camera accessible
- [ ] Port 8998 free (no Moshi running yet)
- [ ] Python dependencies installed

---

**The bicameral mind awaits. Speak, and it will witness.**
