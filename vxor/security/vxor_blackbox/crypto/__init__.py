#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Postquantenfeste Kryptographiekomponenten für VXOR AI Blackbox
--------------------------------------------------------------

Bietet eine einheitliche API für postquantenfeste Kryptographiealgorithmen:
- Kyber512 für Schlüsselaustausch
- NTRUEncrypt für asymmetrische Verschlüsselung
- Dilithium für digitale Signaturen
- AES-256-GCM für symmetrische Verschlüsselung

Spezialisierte Komponenten für die T-Mathematics Engine (MLXTensor, TorchTensor)
und M-LINGUA Integration sind ebenfalls enthalten. Umfassende Hilfsfunktionen für
die sichere Schlüsselableitung, Padding und Hash/HMAC-Operationen ergänzen die Funktionalität.

© 2025 VXOR AI - Alle Rechte vorbehalten
"""

# Postquantenfeste Kryptographie
from .quantum_resistant_crypto import QuantumResistantCrypto

# Symmetrische Verschlüsselung (AES)
from .aes import AESCrypto
from .aes_core import AESCipher, get_aes_cipher

# T-Mathematics Engine und M-LINGUA Integration
from .tensor_crypto import TensorCrypto, get_tensor_crypto

# Kryptographische Hilfsfunktionen
from .crypto_utils import (
    # Schlüsselableitungsfunktionen
    derive_key_from_password,
    derive_key_from_master,
    derive_tensor_specific_key,
    create_key_hierarchy,
    
    # Padding-Funktionen
    pad_data, unpad_data,
    pad_tensor_data, unpad_tensor_data,
    
    # Hash- und HMAC-Funktionen
    compute_hash, compute_hmac, compute_tensor_hmac,
    
    # Zufallszahlenfunktionen
    generate_random_bytes, generate_random_iv, generate_random_nonce,
    
    # Hilfsfunktionen
    constant_time_compare,
    secure_encode, secure_decode,
    split_key, combine_key_parts
)

# Einfache Zugriffsfunktionen für häufig verwendete Komponenten
def get_crypto_components():
    """Gibt ein Dictionary mit allen wichtigen Kryptographiekomponenten zurück."""
    return {
        "quantum_resistant": QuantumResistantCrypto(),
        "aes": get_aes_cipher(),
        "tensor": get_tensor_crypto()
    }

# Standardverschlüsselungshilfen
encrypt = get_aes_cipher().encrypt
decrypt = get_aes_cipher().decrypt
generate_key = get_aes_cipher().generate_key

# Erweiterte Schlüsselhilfen für T-Mathematics und M-LINGUA
def create_tensor_key_hierarchy(password: str, user_id: str, tensor_format="mlx"):
    """Erstellt eine spezialisierte Schlüsselhierarchie für Tensor-Operationen.
    
    Args:
        password: Das Master-Passwort
        user_id: Die Benutzer-ID
        tensor_format: Das Tensor-Format ("mlx", "torch", "numpy")
        
    Returns:
        Dictionary mit allen benötigten Schlüsseln für Tensor-Operationen
    """
    # Basis-Tensor-Metadaten
    tensor_info = {
        "format": tensor_format,
        "dtype": "float32"
    }
    
    # Format-spezifische Ergänzungen
    if tensor_format == "mlx":
        tensor_info.update({
            "device": "ane",  # Apple Neural Engine
            "shape": (0, 0)   # Platzhaltergröße
        })
    elif tensor_format == "torch":
        tensor_info.update({
            "device": "mps",  # Metal Performance Shaders
            "shape": (0, 0),
            "requires_grad": True
        })
    else:  # numpy
        tensor_info.update({
            "device": "cpu",
            "shape": (0, 0)
        })
    
    # Erstelle die vollständige Schlüsselhierarchie
    return create_key_hierarchy(password, user_id, tensor_info=tensor_info)

# T-Mathematics Tensor-Verschlüsselungshilfen
encrypt_tensor = get_tensor_crypto().encrypt_numpy_tensor
decrypt_tensor = get_tensor_crypto().decrypt_numpy_tensor

# Spezielle Tensor-Verschlüsselungshilfen für MLX und PyTorch
try:
    import torch
    encrypt_torch_tensor = get_tensor_crypto().encrypt_torch_tensor
    decrypt_torch_tensor = get_tensor_crypto().decrypt_torch_tensor
except ImportError:
    pass

try:
    import mlx
    encrypt_mlx_tensor = get_tensor_crypto().encrypt_mlx_tensor
    decrypt_mlx_tensor = get_tensor_crypto().decrypt_mlx_tensor
except ImportError:
    pass

# M-LINGUA Verschlüsselungshilfen
encrypt_m_lingua = get_tensor_crypto().encrypt_m_lingua_operation
decrypt_m_lingua = get_tensor_crypto().decrypt_m_lingua_operation

# Tensor-spezifisches HMAC für M-LINGUA-Operationen
def compute_m_lingua_hmac(operation, key):
    """Berechnet einen HMAC-Wert für eine M-LINGUA-Operation.
    
    Args:
        operation: Die M-LINGUA-Operation als Dictionary
        key: Der Authentifizierungsschlüssel
        
    Returns:
        HMAC-Wert für die Operation
    """
    operation_str = json.dumps(operation, sort_keys=True).encode('utf-8')
    metadata = {"type": "m_lingua", "version": "1.0"}
    return compute_tensor_hmac(operation_str, key, metadata)

# Füge json-Import für M-LINGUA-Funktionalität hinzu
import json
