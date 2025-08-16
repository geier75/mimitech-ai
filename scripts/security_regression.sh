#!/bin/bash
# MISO_Ultimate 15.32.28 - Security Regression Tests

set -e

# Verzeichnis zum Repository-Root wechseln
cd "$(dirname "$0")/.."

echo "=== MISO_Ultimate 15.32.28-rc1 Security Regression Tests ==="

# Prüfen, ob in einer virtuellen Umgebung ausgeführt
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: It's recommended to run security tests in a virtual environment"
    read -p "Continue without virtual environment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 1. Pytest Security Tests ausführen
echo "Running security test suite with pytest..."
pytest -v -m security

if [ $? -ne 0 ]; then
    echo "❌ Security tests FAILED"
    exit 1
fi

# 2. ZTM-Validator auf Ultra-Policy ausführen
echo "Running ZTM-Validator on Ultra Policy..."
python security/ztm_scan.py --target configs/ultra_policy.json --level ULTRA

if [ $? -ne 0 ]; then
    echo "❌ ZTM-Scan with Ultra Policy FAILED"
    exit 1
fi

# 3. VOID-Protokoll-Test (Sandbox + ZTM + VOID)
echo "Testing VOID protocol integration..."
python security/tests/test_void_protocol_main.py

if [ $? -ne 0 ]; then
    echo "❌ VOID Protocol tests FAILED"
    exit 1
fi

echo "✅ All security regression tests PASSED"
exit 0
