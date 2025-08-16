#!/bin/bash
# MISO_Ultimate 15.32.28 - Multi-Arch Container Build Script

set -e

# Verzeichnis zum Repository-Root wechseln
cd "$(dirname "$0")/.."

# Parameter
IMAGE_NAME="vxor/miso_ultimate"
IMAGE_TAG="15.32.28-rc1"
PYTHON_VERSION="3.12.3"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

echo "=== MISO_Ultimate ${IMAGE_TAG} Container Build ==="
echo "Building for platforms: linux/amd64, linux/arm64"
echo "Python version: ${PYTHON_VERSION}"

# Docker buildx für Multi-Arch-Support verwenden
echo "Setting up Docker buildx..."
docker buildx create --use --name miso-builder || echo "Builder already exists"

# Container bauen
echo "Building container image..."
docker buildx build --platform linux/amd64,linux/arm64 \
    --build-arg PY_VER=${PYTHON_VERSION} \
    --build-arg BUILD_DATE=${BUILD_DATE} \
    --provenance false \
    --pull \
    --push \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -t ${IMAGE_NAME}:latest .

echo "Container build complete!"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Container lokal für Tests verfügbar machen
echo "Making container available locally..."
docker pull ${IMAGE_NAME}:${IMAGE_TAG}

echo "Done!"
