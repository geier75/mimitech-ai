#!/bin/bash
# MISO_Ultimate 15.32.28 - SBOM Generation and Container Signing Script

set -e

# Verzeichnis zum Repository-Root wechseln
cd "$(dirname "$0")/.."

# Parameter
IMAGE_NAME="vxor/miso_ultimate"
IMAGE_TAG="15.32.28-rc1"
SBOM_FILE="sbom_rc1.json"
COSIGN_KEY="cosign.key"

echo "=== MISO_Ultimate ${IMAGE_TAG} SBOM Generation & Signing ==="

# SBOM mit cyclonedx-python generieren
echo "Generating SBOM from requirements.lock..."
cyclonedx-python -i requirements.lock -o ${SBOM_FILE} --format json

# Validieren des SBOMs
echo "Validating SBOM..."
cyclonedx-cli validate --input-file ${SBOM_FILE} --input-format json || echo "Warning: SBOM validation failed but continuing"

# Container-Image signieren
echo "Signing container image with cosign..."
if [ ! -f "${COSIGN_KEY}" ]; then
    echo "cosign key not found. Generating new key pair..."
    # Beim echten Build würde hier ein Passwort abgefragt, für Automation vereinfacht
    export COSIGN_PASSWORD="MISO_Ultimate_15.32.28_RC1"
    cosign generate-key-pair
fi

# Image signieren
cosign sign --key ${COSIGN_KEY} ${IMAGE_NAME}:${IMAGE_TAG}

# Attestierung hinzufügen
echo "Adding SBOM attestation..."
cosign attest --predicate ${SBOM_FILE} --key ${COSIGN_KEY} ${IMAGE_NAME}:${IMAGE_TAG}

echo "SBOM generation and signing complete!"
echo "SBOM file: ${SBOM_FILE}"
echo "Signed image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Verifizierung der Signatur
echo "Verifying signature..."
cosign verify --key cosign.pub ${IMAGE_NAME}:${IMAGE_TAG}

echo "Done!"
