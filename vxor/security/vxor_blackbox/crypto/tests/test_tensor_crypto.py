#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testmodul für die TensorCrypto-Komponente.

Stellt Tests für die Verschlüsselung und Entschlüsselung von Tensoren
bereit, fokussiert auf NumPy, MLX und PyTorch Tensoren sowie M-LINGUA-Operationen.
"""

import unittest
import os
import numpy as np
import json
import tempfile
import io
from typing import Dict, Any

# Import der zu testenden Komponente
from ..tensor_crypto import TensorCrypto, get_tensor_crypto
from ..aes_core import get_aes_cipher

# Optionale Importe für spezifische Tests
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class TestTensorCrypto(unittest.TestCase):
    """Testet die TensorCrypto-Komponente für VXOR AI Blackbox."""
    
    def setUp(self):
        """Initialisiert die Testumgebung."""
        self.tensor_crypto = TensorCrypto()
        self.aes = get_aes_cipher()
        self.key = self.aes.generate_key()
    
    def test_metadata_encryption(self):
        """Testet die Verschlüsselung und Entschlüsselung von Tensor-Metadaten."""
        # Testdaten
        metadata = {
            "shape": (10, 10),
            "dtype": "float32",
            "format": "numpy",
            "custom_info": "VXOR-TensorTest"
        }
        
        # Verschlüsseln
        encrypted = self.tensor_crypto.encrypt_tensor_metadata(metadata, self.key)
        
        # Sicherstellen, dass verschlüsselte Daten nicht den Originalwerten entsprechen
        self.assertNotEqual(encrypted, json.dumps(metadata).encode('utf-8'))
        
        # Entschlüsseln
        decrypted = self.tensor_crypto.decrypt_tensor_metadata(encrypted, self.key)
        
        # Verifizieren
        self.assertEqual(decrypted, metadata)
    
    def test_tensor_data_encryption(self):
        """Testet die Verschlüsselung und Entschlüsselung von Tensor-Daten."""
        # Testdaten
        test_data = os.urandom(1024)  # 1KB zufällige Daten
        
        # Verschlüsseln
        encrypted = self.tensor_crypto.encrypt_tensor_data(test_data, self.key)
        
        # Sicherstellen, dass verschlüsselte Daten nicht den Originaldaten entsprechen
        self.assertNotEqual(encrypted, test_data)
        
        # Entschlüsseln
        decrypted = self.tensor_crypto.decrypt_tensor_data(encrypted, self.key)
        
        # Verifizieren
        self.assertEqual(decrypted, test_data)
    
    def test_large_tensor_data_encryption(self):
        """Testet die Verschlüsselung und Entschlüsselung von großen Tensor-Daten (Chunk-Modus)."""
        # Testdaten - 5MB zufällige Daten
        test_data = os.urandom(5 * 1024 * 1024)
        
        # Verschlüsseln mit kleiner Chunk-Größe, um Chunking zu erzwingen
        encrypted = self.tensor_crypto.encrypt_tensor_data(test_data, self.key, chunk_size=1024*1024)
        
        # Sicherstellen, dass verschlüsselte Daten nicht den Originaldaten entsprechen
        self.assertNotEqual(encrypted, test_data)
        
        # Entschlüsseln
        decrypted = self.tensor_crypto.decrypt_tensor_data(encrypted, self.key)
        
        # Verifizieren
        self.assertEqual(decrypted, test_data)
    
    def test_numpy_tensor_encryption(self):
        """Testet die Verschlüsselung und Entschlüsselung von NumPy-Tensoren."""
        # Testdaten
        tensor = np.random.rand(10, 10).astype(np.float32)
        
        # Verschlüsseln
        encrypted = self.tensor_crypto.encrypt_numpy_tensor(tensor, self.key)
        
        # Entschlüsseln
        decrypted = self.tensor_crypto.decrypt_numpy_tensor(encrypted, self.key)
        
        # Verifizieren - NumPy-Arrays auf Gleichheit prüfen
        self.assertTrue(np.array_equal(tensor, decrypted))
    
    @unittest.skipIf(not HAS_TORCH, "PyTorch nicht installiert")
    def test_torch_tensor_encryption(self):
        """Testet die Verschlüsselung und Entschlüsselung von PyTorch-Tensoren."""
        # Testdaten
        tensor = torch.rand(10, 10, dtype=torch.float32)
        
        # Verschlüsseln
        encrypted = self.tensor_crypto.encrypt_torch_tensor(tensor, self.key)
        
        # Entschlüsseln
        decrypted = self.tensor_crypto.decrypt_torch_tensor(encrypted, self.key)
        
        # Verifizieren - PyTorch-Tensoren auf Gleichheit prüfen
        self.assertTrue(torch.all(torch.eq(tensor, decrypted)))
    
    @unittest.skipIf(not HAS_TORCH, "PyTorch nicht installiert")
    def test_torch_tensor_with_grad_encryption(self):
        """Testet die Verschlüsselung und Entschlüsselung von PyTorch-Tensoren mit Gradienten."""
        # Testdaten
        tensor = torch.rand(10, 10, dtype=torch.float32, requires_grad=True)
        
        # Verschlüsseln
        encrypted = self.tensor_crypto.encrypt_torch_tensor(tensor, self.key)
        
        # Entschlüsseln
        decrypted = self.tensor_crypto.decrypt_torch_tensor(encrypted, self.key)
        
        # Verifizieren - PyTorch-Tensoren auf Gleichheit prüfen
        self.assertTrue(torch.all(torch.eq(tensor, decrypted)))
        # Verifizieren - requires_grad wurde beibehalten
        self.assertEqual(tensor.requires_grad, decrypted.requires_grad)
    
    @unittest.skipIf(not HAS_MLX, "MLX nicht installiert")
    def test_mlx_tensor_encryption(self):
        """Testet die Verschlüsselung und Entschlüsselung von MLX-Tensoren."""
        # Testdaten
        np_tensor = np.random.rand(10, 10).astype(np.float32)
        tensor = mx.array(np_tensor)  # Verwende mlx.core.array statt mlx.array
        
        # Verschlüsseln
        encrypted = self.tensor_crypto.encrypt_mlx_tensor(tensor, self.key)
        
        # Entschlüsseln
        decrypted = self.tensor_crypto.decrypt_mlx_tensor(encrypted, self.key)
        
        # Verifizieren - MLX-Tensoren auf Gleichheit prüfen
        # MLX hat keine direkte Vergleichsfunktion wie torch.all/torch.eq
        # Daher Vergleich über NumPy
        self.assertTrue(np.array_equal(np.array(tensor), 
                                       np.array(decrypted)))
    
    def test_m_lingua_operation_encryption(self):
        """Testet die Verschlüsselung und Entschlüsselung von M-LINGUA-Operationen."""
        # Testdaten - Beispiel für eine M-LINGUA-Operation
        operation = {
            "operation": "matrix_multiply",
            "tensor_a": "tensor_1",
            "tensor_b": "tensor_2",
            "output": "result_tensor",
            "parameters": {
                "transpose_a": False,
                "transpose_b": True,
                "dtype": "float32"
            }
        }
        
        # Verschlüsseln
        encrypted = self.tensor_crypto.encrypt_m_lingua_operation(operation, self.key)
        
        # Entschlüsseln
        decrypted = self.tensor_crypto.decrypt_m_lingua_operation(encrypted, self.key)
        
        # Verifizieren
        self.assertEqual(decrypted, operation)
    
    def test_singleton_access(self):
        """Testet den Singleton-Zugriff auf die TensorCrypto-Instanz."""
        # Hole Instanz über Singleton-Zugriffsfunktion
        tensor_crypto1 = get_tensor_crypto()
        tensor_crypto2 = get_tensor_crypto()
        
        # Stelle sicher, dass es sich um dieselbe Instanz handelt
        self.assertIs(tensor_crypto1, tensor_crypto2)


if __name__ == "__main__":
    unittest.main()
