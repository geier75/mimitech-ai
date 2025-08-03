#!/bin/bash
# VX-CTRL Console Startup Script
# Zentrale Steuerungsschnittstelle für VXOR AGI-System

echo "🎮 VX-CTRL CONSOLE STARTUP"
echo "=========================="

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "vx_ctrl_console/vx_ctrl_main.py" ]; then
    echo "❌ VX-CTRL Console not found. Please run from project root."
    exit 1
fi

# Create necessary directories
mkdir -p vx_ctrl_console/logs
mkdir -p vx_ctrl_console/config

# Set permissions
chmod +x vx_ctrl_console/vx_ctrl_main.py

echo "🔍 System Check:"
echo "  ✅ Python3 available"
echo "  ✅ VX-CTRL Console found"
echo "  ✅ Directories created"

echo ""
echo "🚀 Starting VX-CTRL Console..."
echo "   Console URL: http://localhost:9000"
echo "   Press Ctrl+C to exit"
echo ""

# Start VX-CTRL Console
cd "$(dirname "$0")/.." || exit 1
python3 vx_ctrl_console/vx_ctrl_main.py

echo ""
echo "🚪 VX-CTRL Console shutdown complete"
