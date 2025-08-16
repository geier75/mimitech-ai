#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrationstest für die kryptographischen Komponenten der VXOR AI Blackbox.

Dieses Skript testet das Zusammenspiel der verschiedenen kryptographischen Komponenten:
- AES-256-GCM Basisklasse
- TensorCrypto-Komponente
- Kryptographische Hilfsfunktionen
- Schlüsselhierarchie-System

Der Fokus liegt auf der Integration mit der T-Mathematics Engine und M-LINGUA.
"""

import os
import json
import tempfile
import time
import numpy as np
from pathlib import Path

# Kryptographische Komponenten
from miso.security.vxor_blackbox.crypto import (
    # AES-Komponenten
    get_aes_cipher,
    encrypt, decrypt, generate_key,
    
    # Tensor-Kryptographie
    get_tensor_crypto,
    encrypt_tensor, decrypt_tensor,
    
    # Kryptographische Hilfsfunktionen
    derive_key_from_password,
    derive_tensor_specific_key,
    create_key_hierarchy,
    create_tensor_key_hierarchy,
    compute_tensor_hmac,
    pad_tensor_data, unpad_tensor_data,
    
    # Kodierungsfunktionen
    secure_encode
)

# Optionale Importe für spezifische Backends
try:
    import torch
    from miso.security.vxor_blackbox.crypto import encrypt_torch_tensor, decrypt_torch_tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mlx
    from miso.security.vxor_blackbox.crypto import encrypt_mlx_tensor, decrypt_mlx_tensor
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def test_aes_tensor_integration():
    """Testet die Integration zwischen AES und Tensor-Kryptographie."""
    print("\n=== AES + TensorCrypto Integration ===")
    
    # Erzeuge Testdaten
    tensor = np.random.rand(10, 10).astype(np.float32)
    print(f"Tensor erstellt: Form {tensor.shape}, Typ {tensor.dtype}")
    
    # Erzeuge Schlüsselhierarchie
    password = "VXOR-AI-Secure-Password!123"
    user_id = "test_user_123"
    keys = create_tensor_key_hierarchy(password, user_id, tensor_format="numpy")
    print(f"Schlüsselhierarchie erstellt mit {len(keys)} Schlüsseln")

    # Tensor-Verschlüsselung mit TensorCrypto
    encrypted_tensor = encrypt_tensor(tensor, keys["tensor_key"])
    print(f"Tensor mit TensorCrypto verschlüsselt: {len(encrypted_tensor)} Bytes")
    
    # Zusätzliche Verschlüsselung mit AES (für mehrschichtige Sicherheit)
    encrypted_data = encrypt(encrypted_tensor, keys["dek"])
    print(f"Verschlüsselte Tensor-Daten mit AES verschlüsselt: {len(encrypted_data)} Bytes")
    
    # Entschlüsseln (umgekehrte Reihenfolge)
    decrypted_tensor_data = decrypt(encrypted_data, keys["dek"])
    decrypted_tensor = decrypt_tensor(decrypted_tensor_data, keys["tensor_key"])
    
    # Verifizieren
    is_equal = np.array_equal(tensor, decrypted_tensor)
    print(f"Tensor erfolgreich entschlüsselt: {is_equal}")
    
    return is_equal


def test_key_hierarchy_for_different_backends():
    """Testet die Schlüsselhierarchie für verschiedene Backends der T-Mathematics Engine."""
    print("\n=== Schlüsselhierarchie für verschiedene Backends ===")
    
    # Zugangsdaten
    password = "VXOR-AI-Secure-Password!123"
    user_id = "test_user_123"
    
    # Erzeuge Schlüsselhierarchien für alle unterstützten Backends
    backends = ["numpy"]
    if HAS_TORCH:
        backends.append("torch")
    if HAS_MLX:
        backends.append("mlx")
    
    hierarchies = {}
    for backend in backends:
        hierarchies[backend] = create_tensor_key_hierarchy(password, user_id, tensor_format=backend)
        print(f"Schlüsselhierarchie für {backend} erstellt")
    
    # Vergleiche Master-Keys (sollten identisch sein)
    are_master_keys_identical = True
    for backend1 in backends:
        for backend2 in backends:
            if backend1 != backend2:
                if hierarchies[backend1]["master_key"] != hierarchies[backend2]["master_key"]:
                    are_master_keys_identical = False
    
    print(f"Master-Keys für alle Backends identisch: {are_master_keys_identical}")
    
    # Vergleiche Tensor-Keys (sollten unterschiedlich sein)
    are_tensor_keys_different = True
    for backend1 in backends:
        for backend2 in backends:
            if backend1 != backend2:
                if hierarchies[backend1].get("tensor_key") == hierarchies[backend2].get("tensor_key"):
                    are_tensor_keys_different = False
    
    print(f"Tensor-Keys für verschiedene Backends unterschiedlich: {are_tensor_keys_different}")
    
    return are_master_keys_identical and are_tensor_keys_different


def test_multi_layer_encryption():
    """Testet mehrschichtige Verschlüsselung für maximale Sicherheit."""
    print("\n=== Mehrschichtige Verschlüsselung ===")
    
    # Erzeuge Testdaten
    data = b"VXOR AI T-Mathematics Engine vertrauliche Daten"
    print(f"Originaldaten: {len(data)} Bytes")
    
    # Erzeuge Schlüsselhierarchie
    password = "VXOR-AI-Secure-Password!123"
    user_id = "test_user_123"
    keys = create_key_hierarchy(password, user_id)
    
    # Mehrschichtige Verschlüsselung
    # Schicht 1: Passwortbasierte Verschlüsselung
    aes = get_aes_cipher()
    layer1 = aes.encrypt_with_password(data, password)
    print(f"Schicht 1 (Passwort): {len(layer1)} Bytes")
    
    # Schicht 2: DEK-Verschlüsselung
    layer2 = encrypt(layer1, keys["dek"])
    print(f"Schicht 2 (DEK): {len(layer2)} Bytes")
    
    # Schicht 3: KEK-Verschlüsselung
    layer3 = encrypt(layer2, keys["kek"])
    print(f"Schicht 3 (KEK): {len(layer3)} Bytes")
    
    # Entschlüsseln (umgekehrte Reihenfolge)
    decrypted_layer2 = decrypt(layer3, keys["kek"])
    decrypted_layer1 = decrypt(decrypted_layer2, keys["dek"])
    decrypted_data = aes.decrypt_with_password(decrypted_layer1, password)
    
    # Verifizieren
    is_equal = (decrypted_data == data)
    print(f"Daten nach mehrschichtiger Verschlüsselung korrekt entschlüsselt: {is_equal}")
    
    return is_equal


def test_m_lingua_security():
    """Testet die Sicherheitsfunktionen für M-LINGUA-Operationen."""
    print("\n=== M-LINGUA-Sicherheit ===")
    
    # Beispiel für eine M-LINGUA-Operation
    operation = {
        "operation": "matrix_multiply",
        "language_input": "Multipliziere Matrix A mit der transponierten Matrix B",
        "tensor_mapping": {
            "A": "input_tensor_1",
            "B": "input_tensor_2",
            "result": "output_tensor"
        },
        "parameters": {
            "transpose_a": False,
            "transpose_b": True,
            "dtype": "float32"
        }
    }
    
    print(f"M-LINGUA-Operation erstellt: {operation['operation']}")
    
    # Erzeuge Schlüsselhierarchie
    password = "VXOR-AI-Secure-Password!123"
    user_id = "test_user_123"
    keys = create_key_hierarchy(password, user_id)
    
    # Berechne HMAC für die Operation
    operation_bytes = json.dumps(operation).encode("utf-8")
    hmac_value = compute_tensor_hmac(operation_bytes, keys["auth_key"], 
                                    {"type": "m_lingua", "version": "1.0"})
    
    print(f"HMAC für M-LINGUA-Operation berechnet: {secure_encode(hmac_value)[:16]}...")
    
    # Verschlüssele die Operation
    padded_operation = pad_tensor_data(operation_bytes, "tensor_specific")
    encrypted_operation = encrypt(padded_operation, keys["dek"])
    
    print(f"M-LINGUA-Operation verschlüsselt: {len(encrypted_operation)} Bytes")
    
    # Entschlüssele und verifiziere die Operation
    decrypted_padded = decrypt(encrypted_operation, keys["dek"])
    decrypted_operation_bytes = unpad_tensor_data(decrypted_padded, "tensor_specific")
    
    # Verifiziere HMAC
    verification_hmac = compute_tensor_hmac(decrypted_operation_bytes, keys["auth_key"], 
                                          {"type": "m_lingua", "version": "1.0"})
    
    is_hmac_valid = (hmac_value == verification_hmac)
    print(f"HMAC-Verifikation erfolgreich: {is_hmac_valid}")
    
    # Rekonstruiere die Operation
    decrypted_operation = json.loads(decrypted_operation_bytes.decode("utf-8"))
    is_operation_valid = (decrypted_operation == operation)
    print(f"Operation korrekt entschlüsselt: {is_operation_valid}")
    
    return is_hmac_valid and is_operation_valid


def test_tensor_specific_functions():
    """Testet spezifische Funktionen für verschiedene Tensor-Formate."""
    print("\n=== Tensor-spezifische Funktionen ===")
    results = []
    
    # NumPy-Tensoren
    print("\n== NumPy-Tensoren ==")
    numpy_tensor = np.random.rand(10, 10).astype(np.float32)
    print(f"NumPy-Tensor erstellt: Form {numpy_tensor.shape}, Typ {numpy_tensor.dtype}")
    
    # Erzeuge tensor-spezifischen Schlüssel
    tensor_info = {
        "shape": numpy_tensor.shape,
        "dtype": str(numpy_tensor.dtype),
        "format": "numpy"
    }
    
    password = "VXOR-AI-Secure-Password!123"
    key, _ = derive_key_from_password(password)
    tensor_key = derive_tensor_specific_key(key, tensor_info)
    
    # Verschlüssele und entschlüssele
    encrypted = encrypt_tensor(numpy_tensor, tensor_key)
    decrypted = decrypt_tensor(encrypted, tensor_key)
    
    is_equal = np.array_equal(numpy_tensor, decrypted)
    print(f"NumPy-Tensor korrekt entschlüsselt: {is_equal}")
    results.append(is_equal)
    
    # MLX-Tensoren
    if HAS_MLX:
        print("\n== MLX-Tensoren ==")
        np_data = np.random.rand(10, 10).astype(np.float32)
        mlx_tensor = mlx.array(np_data)
        print(f"MLX-Tensor erstellt: Form {mlx_tensor.shape}, Typ {mlx_tensor.dtype}")
        
        # Erzeuge tensor-spezifischen Schlüssel
        tensor_info = {
            "shape": mlx_tensor.shape,
            "dtype": str(mlx_tensor.dtype),
            "format": "mlx",
            "device": "ane"
        }
        
        tensor_key = derive_tensor_specific_key(key, tensor_info)
        
        # Verschlüssele und entschlüssele
        encrypted = encrypt_mlx_tensor(mlx_tensor, tensor_key)
        decrypted = decrypt_mlx_tensor(encrypted, tensor_key)
        
        is_equal = np.array_equal(
            mlx_tensor.astype(np.array), decrypted.astype(np.array))
        print(f"MLX-Tensor korrekt entschlüsselt: {is_equal}")
        results.append(is_equal)
    
    # PyTorch-Tensoren
    if HAS_TORCH:
        print("\n== PyTorch-Tensoren ==")
        # Prüfe, ob MPS verfügbar ist
        use_mps = torch.backends.mps.is_available()
        device = torch.device("mps" if use_mps else "cpu")
        print(f"PyTorch verwendet Device: {device}")
        
        torch_tensor = torch.rand(10, 10, dtype=torch.float32, device=device)
        torch_tensor.requires_grad_()
        print(f"PyTorch-Tensor erstellt: Form {torch_tensor.shape}, Typ {torch_tensor.dtype}")
        print(f"Device: {torch_tensor.device}, Requires Grad: {torch_tensor.requires_grad}")
        
        # Erzeuge tensor-spezifischen Schlüssel
        tensor_info = {
            "shape": list(torch_tensor.shape),
            "dtype": str(torch_tensor.dtype),
            "format": "torch",
            "device": str(torch_tensor.device),
            "requires_grad": torch_tensor.requires_grad
        }
        
        tensor_key = derive_tensor_specific_key(key, tensor_info)
        
        # Verschlüssele und entschlüssele
        encrypted = encrypt_torch_tensor(torch_tensor, tensor_key)
        decrypted = decrypt_torch_tensor(encrypted, tensor_key, to_device=str(device))
        
        is_equal = torch.all(torch.eq(torch_tensor, decrypted)).item()
        is_device_preserved = (decrypted.device == torch_tensor.device)
        is_grad_preserved = (decrypted.requires_grad == torch_tensor.requires_grad)
        
        print(f"PyTorch-Tensor korrekt entschlüsselt: {is_equal}")
        print(f"Device und Gradienten erhalten: {is_device_preserved}, {is_grad_preserved}")
        
        results.append(is_equal and is_device_preserved and is_grad_preserved)
    
    return all(results)


def run_all_integration_tests():
    """Führt alle Integrationstests aus."""
    print("\nVXOR AI Blackbox - Integrationstests für kryptographische Komponenten")
    print("=" * 80)
    
    results = []
    
    # Test 1: AES + TensorCrypto Integration
    results.append(test_aes_tensor_integration())
    
    # Test 2: Schlüsselhierarchie für verschiedene Backends
    results.append(test_key_hierarchy_for_different_backends())
    
    # Test 3: Mehrschichtige Verschlüsselung
    results.append(test_multi_layer_encryption())
    
    # Test 4: M-LINGUA-Sicherheit
    results.append(test_m_lingua_security())
    
    # Test 5: Tensor-spezifische Funktionen
    results.append(test_tensor_specific_functions())
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print(f"Ergebnis der Integrationstests: {all(results)}")
    
    for i, result in enumerate(results):
        print(f"Test {i+1}: {'Erfolgreich' if result else 'Fehlgeschlagen'}")
    
    print("\nIntegrationstests abgeschlossen!")
    
    return all(results)


if __name__ == "__main__":
    run_all_integration_tests()
