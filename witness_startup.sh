#!/bin/bash
# witness_startup.sh - Launch the complete Witness system
# ===========================================================
# This script starts all components in the correct order

set -e

echo ""
echo "======================================================================"
echo "   WITNESS SYSTEM STARTUP"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for a port to be ready
wait_for_port() {
    local port=$1
    local service=$2
    local max_wait=30
    local count=0
    
    echo -e "${YELLOW}Waiting for $service to be ready on port $port...${NC}"
    
    while ! check_port $port; do
        sleep 1
        count=$((count + 1))
        if [ $count -ge $max_wait ]; then
            echo -e "${RED}✗ Timeout waiting for $service${NC}"
            return 1
        fi
    done
    
    echo -e "${GREEN}✓ $service is ready${NC}"
    return 0
}

echo "Step 1: Checking prerequisites..."
echo "─────────────────────────────────────────────────────────────────────"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

# Check if Ollama is accessible (remote)
echo -e "${YELLOW}Checking remote brain (Mac Studio)...${NC}"
STUDIO_IP="192.168.1.195"

if curl -s --connect-timeout 3 "http://${STUDIO_IP}:11434/api/tags" > /dev/null; then
    echo -e "${GREEN}✓ Remote brain accessible at ${STUDIO_IP}${NC}"
else
    echo -e "${RED}✗ Cannot reach remote brain at ${STUDIO_IP}${NC}"
    echo "  Make sure Mac Studio Ollama is running: ollama serve"
    exit 1
fi

# Check camera
if [ -e "/dev/video0" ]; then
    echo -e "${GREEN}✓ Camera found${NC}"
else
    echo -e "${YELLOW}⚠ Camera not found (vision will be disabled)${NC}"
fi

echo ""
echo "Step 2: Starting System 1 (Moshi Subconscious)..."
echo "─────────────────────────────────────────────────────────────────────"

# Check if Moshi is already running
if check_port 8998; then
    echo -e "${GREEN}✓ Moshi already running on port 8998${NC}"
else
    echo -e "${BLUE}Starting Moshi web server...${NC}"
    
    # Start Moshi in background
    python3 -m moshi_mlx.local_web > /tmp/moshi.log 2>&1 &
    MOSHI_PID=$!
    echo "Moshi PID: $MOSHI_PID"
    
    # Wait for Moshi to be ready
    if wait_for_port 8998 "Moshi"; then
        echo -e "${GREEN}✓ Moshi started successfully${NC}"
    else
        echo -e "${RED}✗ Failed to start Moshi${NC}"
        echo "Check log: tail /tmp/moshi.log"
        exit 1
    fi
fi

echo ""
echo "Step 3: Starting System 2 (CNS Bicameral Mind)..."
echo "─────────────────────────────────────────────────────────────────────"

sleep 2  # Brief pause to ensure Moshi is fully ready

echo -e "${BLUE}Launching CNS...${NC}"
echo ""

# Run CNS (this will run in foreground)
python3 CNS_bicameral.py

# Cleanup on exit
echo ""
echo "Shutting down..."

if [ ! -z "$MOSHI_PID" ]; then
    echo "Stopping Moshi (PID: $MOSHI_PID)..."
    kill $MOSHI_PID 2>/dev/null || true
fi

echo -e "${GREEN}Shutdown complete${NC}"
