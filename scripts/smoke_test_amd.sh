#!/bin/bash
# MISO_Ultimate 15.32.28 - AMD RDNA3 Docker Smoke Tests

set -e

# Verzeichnis zum Repository-Root wechseln
cd "$(dirname "$0")/.."

# Parameter
IMAGE_NAME="vxor/miso_ultimate"
IMAGE_TAG="15.32.28-rc1"

echo "=== MISO_Ultimate ${IMAGE_TAG} Smoke Tests (AMD RDNA3) ==="

# GPU-Verfügbarkeit prüfen
echo "Checking GPU availability..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi
else
    echo "rocm-smi not found, skipping GPU check"
fi

# Docker-Image pullen
echo "Pulling Docker image..."
docker pull ${IMAGE_NAME}:${IMAGE_TAG}

# Smoke-Tests im Container ausführen
echo "Running smoke tests in container..."
docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    ${IMAGE_NAME}:${IMAGE_TAG} \
    pytest -m smoke

# Ergebnisse der Tests
if [ $? -eq 0 ]; then
    echo "✅ Docker smoke tests PASSED on AMD RDNA3"
    exit 0
else
    echo "❌ Docker smoke tests FAILED on AMD RDNA3"
    exit 1
fi
