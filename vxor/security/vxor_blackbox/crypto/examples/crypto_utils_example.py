#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Beispiele für die Verwendung der kryptographischen Hilfsfunktionen mit der T-Mathematics Engine.

Dieses Modul demonstriert die erweiterten Funktionen für:
- Schlüsselableitung und Hierarchie
- Tensor-spezifisches Padding
- Hash- und HMAC-Funktionen
- M-LINGUA Sicherheitsfunktionen

Die Beispiele sind für verschiedene Backends der T-Mathematics Engine optimiert.
"""

import os
import json
import numpy as np
import tempfile
import time
from pathlib import Path

# Import der kryptographischen Komponenten
from miso.security.vxor_blackbox.crypto import (
    # Schlüsselableitungsfunktionen
    derive_key_from_password,
    derive_key_from_master,
    derive_tensor_specific_key,
    create_key_hierarchy,
    create_tensor_key_hierarchy,
    
    # Padding-Funktionen
    pad_tensor_data, unpad_tensor_data,
    
    # Hash- und HMAC-Funktionen
    compute_hash, compute_hmac, compute_tensor_hmac,
    compute_m_lingua_hmac,
    
    # Tensor-Verschlüsselung
    encrypt_tensor, decrypt_tensor,
    
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


def example_key_hierarchy():
    """Demonstriert die Erstellung und Verwendung einer Schlüsselhierarchie."""
    print("\n=== Schlüsselhierarchie-Beispiel ===")
    
    # Benutzereinstellungen
    password = "VXOR-AI-Secure-Password!123"
    user_id = "example_user_123"
    
    # Erzeuge eine Schlüsselhierarchie für MLX-Tensoren
    print("Erzeuge Schlüsselhierarchie für MLX...")
    mlx_keys = create_tensor_key_hierarchy(password, user_id, tensor_format="mlx")
    
    print(f"Erzeugte Schlüssel:")
    for key_name, key_value in mlx_keys.items():
        # Zeige nur die ersten 16 Zeichen des kodierten Schlüssels
        key_preview = secure_encode(key_value)[:16] + "..."
        print(f"  - {key_name}: {key_preview}")
    
    # Zeige, dass verschiedene Formate unterschiedliche Schlüssel erzeugen
    torch_keys = create_tensor_key_hierarchy(password, user_id, tensor_format="torch")
    
    print("\nSchlüsselvergleich:")
    print(f"MLX und PyTorch Master-Keys identisch: {mlx_keys['master_key'] == torch_keys['master_key']}")
    print(f"MLX und PyTorch Tensor-Keys unterschiedlich: {mlx_keys.get('tensor_key') != torch_keys.get('tensor_key')}")
    
    return mlx_keys


def example_tensor_specific_crypto(keys):
    """Demonstriert tensor-spezifische kryptographische Operationen."""
    print("\n=== Tensor-spezifische Operationen ===")
    
    # Erstelle NumPy-Testdaten
    numpy_tensor = np.random.rand(10, 10).astype(np.float32)
    print(f"NumPy-Tensor erstellt: Form {numpy_tensor.shape}, Typ {numpy_tensor.dtype}")
    
    # Tensor-spezifisches Padding
    tensor_bytes = numpy_tensor.tobytes()
    padded_bytes = pad_tensor_data(tensor_bytes, "tensor_specific")
    print(f"Tensor-Daten gepaddet: {len(tensor_bytes)} → {len(padded_bytes)} Bytes")
    
    # Entpadding
    unpadded_bytes = unpad_tensor_data(padded_bytes, "tensor_specific")
    print(f"Padding entfernt: {len(padded_bytes)} → {len(unpadded_bytes)} Bytes")
    
    # Verifiziere, dass die ursprünglichen Daten wiederhergestellt wurden
    reconstructed = np.frombuffer(unpadded_bytes, dtype=np.float32).reshape(10, 10)
    is_equal = np.array_equal(numpy_tensor, reconstructed)
    print(f"Daten korrekt wiederhergestellt: {is_equal}")
    
    # Tensor-spezifischer HMAC
    tensor_info = {
        "shape": numpy_tensor.shape,
        "dtype": str(numpy_tensor.dtype),
        "format": "numpy"
    }
    
    hmac_value = compute_tensor_hmac(tensor_bytes, keys["auth_key"], tensor_info)
    print(f"Tensor-HMAC berechnet: {secure_encode(hmac_value)[:16]}...")
    
    return numpy_tensor, tensor_info


def example_secure_tensor_storage(keys, tensor, tensor_info):
    """Demonstriert die sichere Speicherung von Tensoren."""
    print("\n=== Sichere Tensor-Speicherung ===")
    
    # Erzeuge einen tensor-spezifischen Schlüssel
    tensor_key = derive_tensor_specific_key(keys["dek"], tensor_info)
    print(f"Tensor-spezifischer Schlüssel abgeleitet: {secure_encode(tensor_key)[:16]}...")
    
    # Verschlüssele den Tensor
    encrypted = encrypt_tensor(tensor, tensor_key)
    print(f"Tensor verschlüsselt: {len(encrypted)} Bytes")
    
    # Speichere in temporärer Datei
    temp_path = Path("./temp_encrypted_tensor.bin")
    with open(temp_path, "wb") as f:
        f.write(encrypted)
    print(f"Verschlüsselter Tensor gespeichert: {temp_path}")
    
    # Lade und entschlüssele den Tensor
    with open(temp_path, "rb") as f:
        loaded_encrypted = f.read()
    
    decrypted = decrypt_tensor(loaded_encrypted, tensor_key)
    print(f"Tensor entschlüsselt: Form {decrypted.shape}, Typ {decrypted.dtype}")
    
    # Verifiziere, dass die Daten korrekt sind
    is_equal = np.array_equal(tensor, decrypted)
    print(f"Daten korrekt entschlüsselt: {is_equal}")
    
    # Lösche die temporäre Datei
    temp_path.unlink()
    print(f"Temporäre Datei gelöscht")
    
    return temp_path


def example_backend_specific(keys):
    """Demonstriert backend-spezifische Operationen für die T-Mathematics Engine."""
    print("\n=== Backend-spezifische Operationen ===")
    
    # MLX-Backend (Apple Neural Engine)
    if HAS_MLX:
        print("\n== MLX-Backend (Apple Neural Engine) ==")
        
        # Erstelle einen MLX-Tensor
        np_tensor = np.random.rand(10, 10).astype(np.float32)
        mlx_tensor = mlx.array(np_tensor)
        print(f"MLX-Tensor erstellt: Form {mlx_tensor.shape}, Typ {mlx_tensor.dtype}")
        
        # Erzeuge einen MLX-spezifischen Schlüssel
        mlx_keys = create_tensor_key_hierarchy(
            "VXOR-AI-Secure-Password!123", "example_user_123", tensor_format="mlx")
        mlx_tensor_key = mlx_keys.get("tensor_key", mlx_keys["dek"])
        
        # Verschlüssele und speichere den MLX-Tensor
        encrypted = encrypt_mlx_tensor(mlx_tensor, mlx_tensor_key)
        temp_path = Path("./temp_mlx_tensor.bin")
        with open(temp_path, "wb") as f:
            f.write(encrypted)
        
        # Lade und entschlüssele den MLX-Tensor
        with open(temp_path, "rb") as f:
            loaded_encrypted = f.read()
        
        decrypted = decrypt_mlx_tensor(loaded_encrypted, mlx_tensor_key)
        
        # Verifiziere, dass die Daten korrekt sind
        is_equal = np.array_equal(
            mlx_tensor.astype(np.array), decrypted.astype(np.array))
        print(f"MLX-Daten korrekt entschlüsselt: {is_equal}")
        
        # Lösche die temporäre Datei
        temp_path.unlink()
    else:
        print("MLX nicht verfügbar, überspringe MLX-Beispiel")
    
    # PyTorch-Backend (MPS)
    if HAS_TORCH:
        print("\n== PyTorch-Backend (MPS) ==")
        
        # Prüfe, ob MPS verfügbar ist
        use_mps = torch.backends.mps.is_available()
        device = torch.device("mps" if use_mps else "cpu")
        print(f"PyTorch verwendet Device: {device}")
        
        # Erstelle einen PyTorch-Tensor
        torch_tensor = torch.rand(10, 10, dtype=torch.float32, device=device)
        torch_tensor.requires_grad_()  # Aktiviere Gradienten
        print(f"PyTorch-Tensor erstellt: Form {torch_tensor.shape}, Typ {torch_tensor.dtype}")
        print(f"Device: {torch_tensor.device}, Requires Grad: {torch_tensor.requires_grad}")
        
        # Erzeuge einen PyTorch-spezifischen Schlüssel
        torch_keys = create_tensor_key_hierarchy(
            "VXOR-AI-Secure-Password!123", "example_user_123", tensor_format="torch")
        torch_tensor_key = torch_keys.get("tensor_key", torch_keys["dek"])
        
        # Verschlüssele und speichere den PyTorch-Tensor
        encrypted = encrypt_torch_tensor(torch_tensor, torch_tensor_key)
        temp_path = Path("./temp_torch_tensor.bin")
        with open(temp_path, "wb") as f:
            f.write(encrypted)
        
        # Lade und entschlüssele den PyTorch-Tensor
        with open(temp_path, "rb") as f:
            loaded_encrypted = f.read()
        
        decrypted = decrypt_torch_tensor(loaded_encrypted, torch_tensor_key, to_device=str(device))
        
        # Verifiziere, dass die Daten korrekt sind
        is_equal = torch.all(torch.eq(torch_tensor, decrypted)).item()
        print(f"PyTorch-Daten korrekt entschlüsselt: {is_equal}")
        print(f"Device und Gradienten erhalten: {decrypted.device == torch_tensor.device}, {decrypted.requires_grad == torch_tensor.requires_grad}")
        
        # Lösche die temporäre Datei
        temp_path.unlink()
    else:
        print("PyTorch nicht verfügbar, überspringe PyTorch-Beispiel")


def example_m_lingua_integration(keys):
    """Demonstriert die Integration mit M-LINGUA."""
    print("\n=== M-LINGUA-Integration ===")
    
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
            "dtype": "float32",
            "device": "auto"  # Automatische Gerätauswahl
        },
        "metadata": {
            "source": "user_query",
            "timestamp": time.time(),
            "session_id": "example-session-123"
        }
    }
    
    print(f"M-LINGUA-Operation erstellt:")
    print(f"  Operation: {operation['operation']}")
    print(f"  Sprachliche Eingabe: \"{operation['language_input']}\"")
    
    # Berechne HMAC für die Operation
    hmac_value = compute_m_lingua_hmac(operation, keys["auth_key"])
    print(f"M-LINGUA-HMAC berechnet: {secure_encode(hmac_value)[:16]}...")
    
    # Verschlüssele die Operation
    encrypted = json.dumps(operation).encode("utf-8")
    encrypted = pad_tensor_data(encrypted, "tensor_specific")
    encrypted = keys["dek"]  # In der Praxis würde hier encrypt(encrypted, keys["dek"]) stehen
    
    print(f"M-LINGUA-Operation verschlüsselt: {len(encrypted)} Bytes")
    
    # Demonstriere, wie die M-LINGUA-Operation in der T-Mathematics Engine verwendet werden könnte
    print("\nVorgestellte Verwendung in der T-Mathematics Engine:")
    print("1. Operation wird vom M-LINGUA-Parser empfangen")
    print("2. HMAC wird validiert, um die Integrität zu überprüfen")
    print("3. Operation wird entschlüsselt und geparsed")
    print("4. Tensoren werden geladen und Operation wird ausgeführt")
    print("5. Ergebnis wird verschlüsselt zurückgegeben")


def run_all_examples():
    """Führt alle Beispiele aus."""
    print("\nVXOR AI Blackbox - Kryptographische Hilfsfunktionen für die T-Mathematics Engine")
    print("=" * 80)
    
    # Erzeuge Schlüsselhierarchie
    keys = example_key_hierarchy()
    
    # Tensor-spezifische Operationen
    tensor, tensor_info = example_tensor_specific_crypto(keys)
    
    # Sichere Tensor-Speicherung
    example_secure_tensor_storage(keys, tensor, tensor_info)
    
    # Backend-spezifische Operationen
    example_backend_specific(keys)
    
    # M-LINGUA-Integration
    example_m_lingua_integration(keys)
    
    print("\n" + "=" * 80)
    print("Alle Beispiele erfolgreich ausgeführt!")


if __name__ == "__main__":
    run_all_examples()
