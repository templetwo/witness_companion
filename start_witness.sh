#!/bin/bash
# start_witness.sh - One-command Witness launcher
# ================================================
# Handles: tunnel cleanup, Moshi detection, tunnel creation, CNS launch

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

STUDIO_IP="192.168.1.195"
STUDIO_USER="tony_studio"
LOCAL_PORT=8998

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   WITNESS SYSTEM LAUNCHER${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Step 1: Kill existing tunnels
echo -e "${YELLOW}Step 1: Cleaning up old tunnels...${NC}"
pkill -f "ssh -L $LOCAL_PORT" 2>/dev/null || true
sleep 1

if lsof -i :$LOCAL_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}Port $LOCAL_PORT still in use. Killing...${NC}"
    kill -9 $(lsof -i :$LOCAL_PORT -t) 2>/dev/null || true
    sleep 1
fi

if ! lsof -i :$LOCAL_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Port $LOCAL_PORT is clear${NC}"
else
    echo -e "${RED}✗ Could not free port $LOCAL_PORT${NC}"
    exit 1
fi

# Step 2: Check if Moshi is running on Studio
echo ""
echo -e "${YELLOW}Step 2: Checking Moshi on Studio...${NC}"

if curl -s --connect-timeout 3 "http://${STUDIO_IP}:${LOCAL_PORT}" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Moshi is running on Studio${NC}"
else
    echo -e "${RED}✗ Moshi not detected on Studio${NC}"
    echo ""
    echo -e "${YELLOW}Please start Moshi on Studio:${NC}"
    echo "  ssh ${STUDIO_USER}@${STUDIO_IP}"
    echo "  cd ~/witness_companion/witness_companion"
    echo "  source venv/bin/activate"
    echo "  python -m moshi_mlx.local_web --host 0.0.0.0"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Step 3: Create fresh SSH tunnel
echo ""
echo -e "${YELLOW}Step 3: Creating SSH tunnel...${NC}"

ssh -f -N -L ${LOCAL_PORT}:localhost:${LOCAL_PORT} ${STUDIO_USER}@${STUDIO_IP}

sleep 2

if lsof -i :$LOCAL_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Tunnel established (localhost:${LOCAL_PORT} → Studio:${LOCAL_PORT})${NC}"
else
    echo -e "${RED}✗ Tunnel failed to establish${NC}"
    exit 1
fi

# Step 4: Verify tunnel works
echo ""
echo -e "${YELLOW}Step 4: Verifying tunnel...${NC}"

if curl -s --connect-timeout 3 "http://localhost:${LOCAL_PORT}" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Tunnel verified - Moshi accessible via localhost${NC}"
else
    echo -e "${RED}✗ Tunnel not working${NC}"
    exit 1
fi

# Step 5: Launch CNS
echo ""
echo -e "${YELLOW}Step 5: Launching CNS Remote Body...${NC}"
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   Open browser: http://localhost:${LOCAL_PORT}${NC}"
echo -e "${GREEN}   Click Connect, allow microphone${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

cd /Users/vaquez/Desktop/witness_companion
source venv/bin/activate
python CNS_remote_body.py
