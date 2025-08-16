#!/bin/bash
# MISO_Ultimate 15.32.28 - macOS M4 Max Smoke Tests

set -e

# Verzeichnis zum Repository-Root wechseln
cd "$(dirname "$0")/.."

echo "=== MISO_Ultimate 15.32.28-rc1 Smoke Tests (macOS M4 Max) ==="

# Python Virtual Environment erstellen
echo "Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate

# Abhängigkeiten installieren
echo "Installing dependencies from requirements.lock..."
pip install -r requirements.lock

# Training-Demo mit einer Epoche ausführen
echo "Running training demo with 1 epoch..."
python macros/start_training_demo.py --epochs 1

# Schnellen Benchmark ausführen
echo "Running quick benchmark suite..."
python vxor_launcher.py --benchmark quick

# Ergebnisse der Tests
echo "Checking test results..."
if [ $? -eq 0 ]; then
    echo "✅ Smoke tests PASSED on macOS M4 Max"
    exit 0
else
    echo "❌ Smoke tests FAILED on macOS M4 Max"
    exit 1
fi
