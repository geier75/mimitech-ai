#!/bin/bash
# VX-CTRL Console Port Conflict Fix
# Löst Port-Konflikte und startet Console neu

echo "🔧 VX-CTRL CONSOLE PORT CONFLICT FIX"
echo "===================================="

# Check what's using port 9000
echo "🔍 Checking port 9000..."
PORT_USER=$(lsof -ti :9000)

if [ ! -z "$PORT_USER" ]; then
    echo "⚠️  Port 9000 is in use by process: $PORT_USER"
    
    # Check if it's our own VX-CTRL process
    PROCESS_NAME=$(ps -p $PORT_USER -o comm= 2>/dev/null)
    echo "📋 Process name: $PROCESS_NAME"
    
    if [[ "$PROCESS_NAME" == *"python"* ]]; then
        echo "🎮 Looks like another VX-CTRL Console instance"
        echo "🛑 Killing existing process..."
        kill -9 $PORT_USER
        sleep 2
        echo "✅ Process terminated"
    else
        echo "⚠️  Port used by different application: $PROCESS_NAME"
        echo "🔄 VX-CTRL will automatically find next available port"
    fi
else
    echo "✅ Port 9000 is available"
fi

# Find available ports
echo ""
echo "🔍 Scanning for available ports..."
for port in {9000..9010}; do
    if ! lsof -i :$port > /dev/null 2>&1; then
        echo "✅ Port $port: Available"
    else
        PROC=$(lsof -ti :$port)
        echo "❌ Port $port: In use (PID: $PROC)"
    fi
done

echo ""
echo "🚀 Starting VX-CTRL Console with automatic port detection..."
echo "   Console will use first available port starting from 9000"
echo ""

# Start VX-CTRL Console
cd "$(dirname "$0")/.." || exit 1
python3 vx_ctrl_console/vx_ctrl_main.py

echo ""
echo "🚪 VX-CTRL Console shutdown complete"
