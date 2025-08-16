#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Beispiele für die Verwendung der TensorCrypto-Komponente mit der T-Mathematics Engine.

Dieses Modul demonstriert die Verschlüsselung und Entschlüsselung von:
- NumPy-Tensoren (Standardfall)
- MLX-Tensoren (für Apple Neural Engine)
- PyTorch-Tensoren (mit MPS-Unterstützung)
- M-LINGUA-Operationen

Die Beispiele können als Template für die Integration in VXOR AI verwendet werden.
"""

import os
import numpy as np
import json
import io
import time
from pathlib import Path

# Importiere die TensorCrypto-Komponente
from miso.security.vxor_blackbox.crypto import (
    generate_key,
    encrypt_tensor, decrypt_tensor,
    encrypt_m_lingua, decrypt_m_lingua
)

# Für MLX- und PyTorch-spezifische Funktionen
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


def encrypt_decrypt_numpy_tensor():
    """Demonstriert Verschlüsselung und Entschlüsselung eines NumPy-Tensors."""
    print("=== NumPy-Tensor-Verschlüsselung ===")
    
    # Generiere einen AES-256-Schlüssel
    key = generate_key()
    print(f"Schlüssel generiert: {len(key)*8} Bit")
    
    # Erstelle einen NumPy-Tensor
    tensor = np.random.rand(10, 10).astype(np.float32)
    print(f"Originaltensor erstellt: Form {tensor.shape}, Typ {tensor.dtype}")
    print(f"Erste Elemente: {tensor[0, 0:3]}")
    
    # Verschlüssele den Tensor
    start_time = time.time()
    encrypted = encrypt_tensor(tensor, key)
    encrypt_time = time.time() - start_time
    print(f"Tensor verschlüsselt: {len(encrypted)} Bytes, Zeit: {encrypt_time:.4f}s")
    
    # Speichere den verschlüsselten Tensor in einer Datei
    encrypted_path = Path("./encrypted_tensor.bin")
    with open(encrypted_path, "wb") as f:
        f.write(encrypted)
    print(f"Verschlüsselter Tensor gespeichert: {encrypted_path}")
    
    # Lade den verschlüsselten Tensor
    with open(encrypted_path, "rb") as f:
        loaded_encrypted = f.read()
    
    # Entschlüssele den Tensor
    start_time = time.time()
    decrypted = decrypt_tensor(loaded_encrypted, key)
    decrypt_time = time.time() - start_time
    print(f"Tensor entschlüsselt: Form {decrypted.shape}, Typ {decrypted.dtype}, Zeit: {decrypt_time:.4f}s")
    print(f"Erste Elemente: {decrypted[0, 0:3]}")
    
    # Überprüfe, ob die Tensoren gleich sind
    is_equal = np.array_equal(tensor, decrypted)
    print(f"Tensoren sind gleich: {is_equal}")
    
    return is_equal


def encrypt_decrypt_torch_tensor():
    """Demonstriert Verschlüsselung und Entschlüsselung eines PyTorch-Tensors mit MPS-Unterstützung."""
    if not HAS_TORCH:
        print("PyTorch ist nicht installiert. Überspringen des PyTorch-Beispiels.")
        return False
    
    print("\n=== PyTorch-Tensor-Verschlüsselung ===")
    
    # Überprüfe, ob MPS verfügbar ist
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else "cpu")
    print(f"Verwende Device: {device}")
    
    # Generiere einen AES-256-Schlüssel
    key = generate_key()
    
    # Erstelle einen PyTorch-Tensor
    tensor = torch.rand(10, 10, dtype=torch.float32, device=device)
    tensor.requires_grad_()  # Aktiviere Gradienten
    print(f"Originaltensor erstellt: Form {tensor.shape}, Typ {tensor.dtype}, Device {tensor.device}")
    print(f"Erfordert Gradienten: {tensor.requires_grad}")
    print(f"Erste Elemente: {tensor[0, 0:3]}")
    
    # Verschlüssele den Tensor
    start_time = time.time()
    encrypted = encrypt_torch_tensor(tensor, key)
    encrypt_time = time.time() - start_time
    print(f"Tensor verschlüsselt: {len(encrypted)} Bytes, Zeit: {encrypt_time:.4f}s")
    
    # Speichere den verschlüsselten Tensor in einer Datei
    encrypted_path = Path("./encrypted_torch_tensor.bin")
    with open(encrypted_path, "wb") as f:
        f.write(encrypted)
    print(f"Verschlüsselter Tensor gespeichert: {encrypted_path}")
    
    # Lade den verschlüsselten Tensor
    with open(encrypted_path, "rb") as f:
        loaded_encrypted = f.read()
    
    # Entschlüssele den Tensor
    start_time = time.time()
    decrypted = decrypt_torch_tensor(loaded_encrypted, key, to_device=str(device))
    decrypt_time = time.time() - start_time
    print(f"Tensor entschlüsselt: Form {decrypted.shape}, Typ {decrypted.dtype}, Device {decrypted.device}")
    print(f"Erfordert Gradienten: {decrypted.requires_grad}")
    print(f"Erste Elemente: {decrypted[0, 0:3]}")
    
    # Überprüfe, ob die Tensoren gleich sind
    is_equal = torch.all(torch.eq(tensor, decrypted)).item()
    print(f"Tensoren sind gleich: {is_equal}")
    
    return is_equal


def encrypt_decrypt_mlx_tensor():
    """Demonstriert Verschlüsselung und Entschlüsselung eines MLX-Tensors für die Apple Neural Engine."""
    if not HAS_MLX:
        print("MLX ist nicht installiert. Überspringen des MLX-Beispiels.")
        return False
    
    print("\n=== MLX-Tensor-Verschlüsselung ===")
    
    # Generiere einen AES-256-Schlüssel
    key = generate_key()
    
    # Erstelle einen MLX-Tensor
    np_tensor = np.random.rand(10, 10).astype(np.float32)
    tensor = mlx.array(np_tensor)
    print(f"Originaltensor erstellt: Form {tensor.shape}, Typ {tensor.dtype}")
    print(f"Erste Elemente: {tensor[0, 0:3]}")
    
    # Verschlüssele den Tensor
    start_time = time.time()
    encrypted = encrypt_mlx_tensor(tensor, key)
    encrypt_time = time.time() - start_time
    print(f"Tensor verschlüsselt: {len(encrypted)} Bytes, Zeit: {encrypt_time:.4f}s")
    
    # Speichere den verschlüsselten Tensor in einer Datei
    encrypted_path = Path("./encrypted_mlx_tensor.bin")
    with open(encrypted_path, "wb") as f:
        f.write(encrypted)
    print(f"Verschlüsselter Tensor gespeichert: {encrypted_path}")
    
    # Lade den verschlüsselten Tensor
    with open(encrypted_path, "rb") as f:
        loaded_encrypted = f.read()
    
    # Entschlüssele den Tensor
    start_time = time.time()
    decrypted = decrypt_mlx_tensor(loaded_encrypted, key)
    decrypt_time = time.time() - start_time
    print(f"Tensor entschlüsselt: Form {decrypted.shape}, Typ {decrypted.dtype}, Zeit: {decrypt_time:.4f}s")
    print(f"Erste Elemente: {decrypted[0, 0:3]}")
    
    # MLX hat keine direkte Vergleichsoperation wie torch.all/torch.eq
    # Daher konvertieren wir beide zu NumPy für den Vergleich
    tensor_np = tensor.astype(np.array)
    decrypted_np = decrypted.astype(np.array)
    is_equal = np.array_equal(tensor_np, decrypted_np)
    print(f"Tensoren sind gleich: {is_equal}")
    
    return is_equal


def encrypt_decrypt_m_lingua_operation():
    """Demonstriert Verschlüsselung und Entschlüsselung einer M-LINGUA-Operation."""
    print("\n=== M-LINGUA-Operationsverschlüsselung ===")
    
    # Generiere einen AES-256-Schlüssel
    key = generate_key()
    
    # Beispiel für eine M-LINGUA-Operation
    operation = {
        "operation": "matrix_multiply",
        "language_input": "Multipliziere Matrix A mit Matrix B und speichere das Ergebnis in C",
        "tensor_mapping": {
            "A": "input_tensor_1",
            "B": "input_tensor_2",
            "C": "output_tensor"
        },
        "parameters": {
            "transpose_a": False,
            "transpose_b": True,
            "dtype": "float32",
            "device": "mps"
        },
        "metadata": {
            "source": "user_query",
            "timestamp": time.time(),
            "session_id": "example-session-123"
        }
    }
    
    print(f"Original M-LINGUA-Operation erstellt")
    
    # Verschlüssele die Operation
    start_time = time.time()
    encrypted = encrypt_m_lingua(operation, key)
    encrypt_time = time.time() - start_time
    print(f"Operation verschlüsselt: {len(encrypted)} Bytes, Zeit: {encrypt_time:.4f}s")
    
    # Speichere die verschlüsselte Operation in einer Datei
    encrypted_path = Path("./encrypted_m_lingua.bin")
    with open(encrypted_path, "wb") as f:
        f.write(encrypted)
    print(f"Verschlüsselte Operation gespeichert: {encrypted_path}")
    
    # Lade die verschlüsselte Operation
    with open(encrypted_path, "rb") as f:
        loaded_encrypted = f.read()
    
    # Entschlüssele die Operation
    start_time = time.time()
    decrypted = decrypt_m_lingua(loaded_encrypted, key)
    decrypt_time = time.time() - start_time
    print(f"Operation entschlüsselt: {len(json.dumps(decrypted))} Zeichen, Zeit: {decrypt_time:.4f}s")
    
    # Überprüfe, ob die Operationen gleich sind
    is_equal = decrypted == operation
    print(f"Operationen sind gleich: {is_equal}")
    
    return is_equal


def run_all_examples():
    """Führt alle Beispiele aus."""
    results = []
    
    print("VXOR AI TensorCrypto Beispiele\n")
    print("Demonstriert die Verschlüsselung und Entschlüsselung von Tensoren")
    print("für die T-Mathematics Engine und M-LINGUA-Integration.\n")
    
    # NumPy-Tensor (immer verfügbar)
    results.append(encrypt_decrypt_numpy_tensor())
    
    # PyTorch-Tensor (wenn verfügbar)
    if HAS_TORCH:
        results.append(encrypt_decrypt_torch_tensor())
    
    # MLX-Tensor (wenn verfügbar)
    if HAS_MLX:
        results.append(encrypt_decrypt_mlx_tensor())
    
    # M-LINGUA-Operation
    results.append(encrypt_decrypt_m_lingua_operation())
    
    # Zusammenfassung
    print("\n=== Zusammenfassung ===")
    all_successful = all(results)
    print(f"Alle Tests erfolgreich: {all_successful}")
    
    # Lösche temporäre Dateien
    cleanup_files = [
        Path("./encrypted_tensor.bin"),
        Path("./encrypted_torch_tensor.bin"),
        Path("./encrypted_mlx_tensor.bin"),
        Path("./encrypted_m_lingua.bin")
    ]
    
    for file in cleanup_files:
        if file.exists():
            file.unlink()
            print(f"Temporäre Datei gelöscht: {file}")


if __name__ == "__main__":
    run_all_examples()
