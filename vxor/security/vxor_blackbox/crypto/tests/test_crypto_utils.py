#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testmodul für die kryptographischen Hilfsfunktionen.

Testet die Schlüsselableitung, Padding-Funktionen, Hash- und HMAC-Funktionen
sowie die speziellen Funktionen für die T-Mathematics Engine.
"""

import unittest
import os
import json
import numpy as np
from typing import Dict, Any

# Import der zu testenden Komponenten
from ..crypto_utils import (
    derive_key_from_password,
    derive_key_from_master,
    derive_tensor_specific_key,
    create_key_hierarchy,
    pad_data, unpad_data,
    pad_tensor_data, unpad_tensor_data,
    compute_hash, compute_hmac, compute_tensor_hmac,
    generate_random_bytes, generate_random_iv, generate_random_nonce,
    constant_time_compare,
    secure_encode, secure_decode,
    split_key, combine_key_parts
)

# Optionale Importe für spezifische Tests
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mlx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class TestCryptoUtils(unittest.TestCase):
    """Testet die kryptographischen Hilfsfunktionen für VXOR AI Blackbox."""

    def setUp(self):
        """Initialisiert die Testumgebung."""
        self.test_password = "VXOR-AI-Secure-Passphrase!123"
        self.test_user_id = "test_user_123"
        self.test_data = b"VXOR AI T-Mathematics Test Data"
        
        # Beispiel-Tensor-Metadaten
        self.tensor_info = {
            "shape": (10, 10),
            "dtype": "float32",
            "format": "numpy",
            "device": "cpu"
        }
        
        # MLX-spezifische Tensor-Metadaten
        self.mlx_tensor_info = {
            "shape": (10, 10),
            "dtype": "float32",
            "format": "mlx",
            "device": "ane"  # Apple Neural Engine
        }
        
        # PyTorch-spezifische Tensor-Metadaten
        self.torch_tensor_info = {
            "shape": (10, 10),
            "dtype": "float32",
            "format": "torch",
            "device": "mps",
            "requires_grad": True
        }

    # ===== Tests für Schlüsselableitungsfunktionen =====
    
    def test_derive_key_from_password(self):
        """Testet die Passwort-basierte Schlüsselableitung."""
        # Generiere zwei Schlüssel mit demselben Passwort und Salt
        key1, salt = derive_key_from_password(self.test_password)
        key2, _ = derive_key_from_password(self.test_password, salt=salt)
        
        # Stelle sicher, dass die Schlüssel die richtige Länge haben
        self.assertEqual(len(key1), 32)  # 256 Bit
        
        # Stelle sicher, dass gleiche Passwörter mit gleichem Salt gleiche Schlüssel erzeugen
        self.assertEqual(key1, key2)
        
        # Stelle sicher, dass unterschiedliche Passwörter unterschiedliche Schlüssel ergeben
        key3, _ = derive_key_from_password("AnotherPassword", salt=salt)
        self.assertNotEqual(key1, key3)

    def test_derive_key_from_master(self):
        """Testet die Ableitung von Sekundärschlüsseln aus einem Master-Schlüssel."""
        # Erzeuge einen Master-Schlüssel
        master_key, _ = derive_key_from_password(self.test_password)
        
        # Leite zwei verschiedene Schlüssel aus demselben Master-Schlüssel ab
        key1 = derive_key_from_master(master_key, "context1")
        key2 = derive_key_from_master(master_key, "context2")
        
        # Stelle sicher, dass unterschiedliche Kontexte unterschiedliche Schlüssel ergeben
        self.assertNotEqual(key1, key2)
        
        # Stelle sicher, dass gleiche Kontexte gleiche Schlüssel ergeben
        key3 = derive_key_from_master(master_key, "context1")
        self.assertEqual(key1, key3)

    def test_derive_tensor_specific_key(self):
        """Testet die Ableitung von Tensor-spezifischen Schlüsseln."""
        # Erzeuge einen Master-Schlüssel
        master_key, _ = derive_key_from_password(self.test_password)
        
        # Leite Schlüssel für verschiedene Tensor-Typen ab
        key1 = derive_tensor_specific_key(master_key, self.tensor_info)
        key2 = derive_tensor_specific_key(master_key, self.mlx_tensor_info)
        key3 = derive_tensor_specific_key(master_key, self.torch_tensor_info)
        
        # Stelle sicher, dass unterschiedliche Tensor-Metadaten unterschiedliche Schlüssel ergeben
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertNotEqual(key2, key3)
        
        # Stelle sicher, dass gleiche Tensor-Metadaten gleiche Schlüssel ergeben
        key4 = derive_tensor_specific_key(master_key, self.tensor_info)
        self.assertEqual(key1, key4)

    def test_create_key_hierarchy(self):
        """Testet die Erstellung einer Schlüsselhierarchie."""
        # Erzeuge eine Schlüsselhierarchie
        keys = create_key_hierarchy(self.test_password, self.test_user_id)
        
        # Stelle sicher, dass alle erwarteten Schlüssel vorhanden sind
        self.assertIn("master_key", keys)
        self.assertIn("kek", keys)
        self.assertIn("dek", keys)
        self.assertIn("auth_key", keys)
        
        # Teste mit Tensor-Metadaten
        keys_with_tensor = create_key_hierarchy(
            self.test_password, self.test_user_id, tensor_info=self.tensor_info
        )
        
        # Stelle sicher, dass der Tensor-Schlüssel vorhanden ist
        self.assertIn("tensor_key", keys_with_tensor)
        
        # Stelle sicher, dass verschiedene Benutzer unterschiedliche Schlüssel erhalten
        keys2 = create_key_hierarchy(self.test_password, "another_user")
        self.assertNotEqual(keys["master_key"], keys2["master_key"])

    # ===== Tests für Padding-Funktionen =====
    
    def test_standard_padding(self):
        """Testet Standard-PKCS#7-Padding."""
        # Teste Padding und Unpadding
        padded = pad_data(self.test_data)
        unpadded = unpad_data(padded)
        
        # Stelle sicher, dass die ungepaddetenen Daten den Originaldaten entsprechen
        self.assertEqual(unpadded, self.test_data)
        
        # Teste Padding für verschiedene Datengrößen
        for size in range(1, 32):
            data = os.urandom(size)
            padded = pad_data(data)
            
            # Stelle sicher, dass die Länge ein Vielfaches der Blockgröße ist
            self.assertEqual(len(padded) % 16, 0)
            
            # Stelle sicher, dass Unpadding die Originaldaten wiederherstellt
            self.assertEqual(unpad_data(padded), data)

    def test_tensor_padding(self):
        """Testet Tensor-spezifisches Padding."""
        # Teste verschiedene Padding-Strategien
        strategies = ["pkcs7", "zero", "tensor_specific"]
        
        for strategy in strategies:
            # Teste für verschiedene Datengrößen
            for size in range(1, 32):
                data = os.urandom(size)
                padded = pad_tensor_data(data, strategy)
                
                # Stelle sicher, dass die Länge ein Vielfaches der Blockgröße ist
                self.assertEqual(len(padded) % 16, 0)
                
                # Stelle sicher, dass Unpadding die Originaldaten wiederherstellt
                unpadded = unpad_tensor_data(padded, strategy)
                self.assertEqual(unpadded, data)

    # ===== Tests für Hash- und HMAC-Funktionen =====
    
    def test_hash_functions(self):
        """Testet Hash-Funktionen."""
        # Teste verschiedene Hash-Algorithmen
        algorithms = ["sha256", "sha512", "sha3_256"]
        
        for algorithm in algorithms:
            hash1 = compute_hash(self.test_data, algorithm)
            hash2 = compute_hash(self.test_data, algorithm)
            
            # Stelle sicher, dass gleiche Daten gleiche Hashes ergeben
            self.assertEqual(hash1, hash2)
            
            # Stelle sicher, dass unterschiedliche Daten unterschiedliche Hashes ergeben
            modified_data = self.test_data + b"X"
            hash3 = compute_hash(modified_data, algorithm)
            self.assertNotEqual(hash1, hash3)

    def test_hmac_functions(self):
        """Testet HMAC-Funktionen."""
        # Erzeuge einen Schlüssel
        key, _ = derive_key_from_password(self.test_password)
        
        # Teste verschiedene HMAC-Algorithmen
        algorithms = ["sha256", "sha512"]
        
        for algorithm in algorithms:
            hmac1 = compute_hmac(self.test_data, key, algorithm)
            hmac2 = compute_hmac(self.test_data, key, algorithm)
            
            # Stelle sicher, dass gleiche Daten und Schlüssel gleiche HMACs ergeben
            self.assertEqual(hmac1, hmac2)
            
            # Stelle sicher, dass unterschiedliche Daten unterschiedliche HMACs ergeben
            modified_data = self.test_data + b"X"
            hmac3 = compute_hmac(modified_data, key, algorithm)
            self.assertNotEqual(hmac1, hmac3)
            
            # Stelle sicher, dass unterschiedliche Schlüssel unterschiedliche HMACs ergeben
            key2, _ = derive_key_from_password("DifferentPassword")
            hmac4 = compute_hmac(self.test_data, key2, algorithm)
            self.assertNotEqual(hmac1, hmac4)

    def test_tensor_hmac(self):
        """Testet Tensor-spezifischen HMAC."""
        # Erzeuge einen Schlüssel
        key, _ = derive_key_from_password(self.test_password)
        
        # Berechne HMAC ohne Metadaten
        hmac1 = compute_tensor_hmac(self.test_data, key)
        
        # Berechne HMAC mit Metadaten
        hmac2 = compute_tensor_hmac(self.test_data, key, self.tensor_info)
        
        # Stelle sicher, dass Metadaten den HMAC beeinflussen
        self.assertNotEqual(hmac1, hmac2)
        
        # Stelle sicher, dass unterschiedliche Tensor-Metadaten unterschiedliche HMACs ergeben
        hmac3 = compute_tensor_hmac(self.test_data, key, self.mlx_tensor_info)
        self.assertNotEqual(hmac2, hmac3)

    # ===== Tests für Zufallszahlenfunktionen =====
    
    def test_random_functions(self):
        """Testet Funktionen zur Erzeugung von Zufallszahlen."""
        # Teste Erzeugung von Zufallsbytes
        random1 = generate_random_bytes(32)
        random2 = generate_random_bytes(32)
        
        # Stelle sicher, dass zwei aufeinanderfolgende Aufrufe unterschiedliche Werte ergeben
        self.assertNotEqual(random1, random2)
        
        # Teste Generierung von IVs
        iv1 = generate_random_iv()
        iv2 = generate_random_iv()
        
        # Stelle sicher, dass zwei aufeinanderfolgende Aufrufe unterschiedliche Werte ergeben
        self.assertNotEqual(iv1, iv2)
        
        # Teste Generierung von Nonces
        nonce1 = generate_random_nonce()
        nonce2 = generate_random_nonce()
        
        # Stelle sicher, dass zwei aufeinanderfolgende Aufrufe unterschiedliche Werte ergeben
        self.assertNotEqual(nonce1, nonce2)

    # ===== Tests für sonstige Hilfsfunktionen =====
    
    def test_constant_time_compare(self):
        """Testet den zeitinvarianten Vergleich von Bytes."""
        a = b"VXOR-AI-Test-Data"
        b = b"VXOR-AI-Test-Data"
        c = b"VXOR-AI-Modified"
        
        # Stelle sicher, dass gleiche Daten als gleich erkannt werden
        self.assertTrue(constant_time_compare(a, b))
        
        # Stelle sicher, dass unterschiedliche Daten als unterschiedlich erkannt werden
        self.assertFalse(constant_time_compare(a, c))

    def test_secure_encode_decode(self):
        """Testet sichere Kodierung und Dekodierung."""
        # Teste verschiedene Kodierungen
        encodings = ["base64", "hex"]
        
        for encoding in encodings:
            # Kodiere und dekodiere Daten
            encoded = secure_encode(self.test_data, encoding)
            decoded = secure_decode(encoded, encoding)
            
            # Stelle sicher, dass Dekodierung die Originaldaten wiederherstellt
            self.assertEqual(decoded, self.test_data)

    def test_key_splitting(self):
        """Testet die Aufteilung und Rekombination von Schlüsseln."""
        # Erzeuge einen Schlüssel
        key, _ = derive_key_from_password(self.test_password)
        
        # Teile den Schlüssel in mehrere Teile auf
        num_parts = 3
        parts = split_key(key, num_parts)
        
        # Stelle sicher, dass die richtige Anzahl von Teilen erzeugt wurde
        self.assertEqual(len(parts), num_parts)
        
        # Stelle sicher, dass jeder Teil die gleiche Länge wie der Originalschlüssel hat
        for part in parts:
            self.assertEqual(len(part), len(key))
        
        # Rekombiniere die Teile
        recombined = combine_key_parts(parts)
        
        # Stelle sicher, dass der rekombinierte Schlüssel dem Original entspricht
        self.assertEqual(recombined, key)

    # ===== Tests für T-Mathematics Engine Integration =====
    
    def test_numpy_tensor_integration(self):
        """Testet die Integration mit NumPy-Tensoren."""
        # Erstelle einen NumPy-Tensor
        tensor = np.random.rand(10, 10).astype(np.float32)
        
        # Erzeuge einen Schlüssel für diesen Tensor
        master_key, _ = derive_key_from_password(self.test_password)
        tensor_key = derive_tensor_specific_key(master_key, self.tensor_info)
        
        # Konvertiere Tensor zu Bytes
        tensor_bytes = tensor.tobytes()
        
        # Wende Tensor-spezifisches Padding an
        padded = pad_tensor_data(tensor_bytes, "tensor_specific")
        
        # Stelle sicher, dass Unpadding die ursprünglichen Bytes wiederherstellt
        unpadded = unpad_tensor_data(padded, "tensor_specific")
        self.assertEqual(unpadded, tensor_bytes)
        
        # Rekonstruiere den Tensor aus den Bytes
        reconstructed = np.frombuffer(unpadded, dtype=np.float32).reshape(10, 10)
        
        # Stelle sicher, dass der rekonstruierte Tensor dem Original entspricht
        np.testing.assert_array_equal(tensor, reconstructed)

    @unittest.skipIf(not HAS_TORCH, "PyTorch nicht installiert")
    def test_pytorch_tensor_integration(self):
        """Testet die Integration mit PyTorch-Tensoren."""
        # Erstelle einen PyTorch-Tensor
        tensor = torch.rand(10, 10, dtype=torch.float32)
        
        # Erzeuge einen Schlüssel für diesen Tensor
        master_key, _ = derive_key_from_password(self.test_password)
        tensor_key = derive_tensor_specific_key(master_key, self.torch_tensor_info)
        
        # Konvertiere Tensor zu Bytes
        tensor_bytes = tensor.cpu().numpy().tobytes()
        
        # Wende Tensor-spezifisches Padding an
        padded = pad_tensor_data(tensor_bytes, "tensor_specific")
        
        # Stelle sicher, dass Unpadding die ursprünglichen Bytes wiederherstellt
        unpadded = unpad_tensor_data(padded, "tensor_specific")
        self.assertEqual(unpadded, tensor_bytes)
        
        # Rekonstruiere den Tensor aus den Bytes
        np_tensor = np.frombuffer(unpadded, dtype=np.float32).reshape(10, 10)
        reconstructed = torch.from_numpy(np_tensor)
        
        # Stelle sicher, dass der rekonstruierte Tensor dem Original entspricht
        self.assertTrue(torch.all(torch.eq(tensor, reconstructed)))

    @unittest.skipIf(not HAS_MLX, "MLX nicht installiert")
    def test_mlx_tensor_integration(self):
        """Testet die Integration mit MLX-Tensoren für die Apple Neural Engine."""
        # Erstelle einen MLX-Tensor
        np_tensor = np.random.rand(10, 10).astype(np.float32)
        tensor = mlx.array(np_tensor)
        
        # Erzeuge einen Schlüssel für diesen Tensor
        master_key, _ = derive_key_from_password(self.test_password)
        tensor_key = derive_tensor_specific_key(master_key, self.mlx_tensor_info)
        
        # Konvertiere Tensor zu Bytes (über NumPy)
        tensor_bytes = tensor.astype(np.array).tobytes()
        
        # Wende Tensor-spezifisches Padding an
        padded = pad_tensor_data(tensor_bytes, "tensor_specific")
        
        # Stelle sicher, dass Unpadding die ursprünglichen Bytes wiederherstellt
        unpadded = unpad_tensor_data(padded, "tensor_specific")
        self.assertEqual(unpadded, tensor_bytes)
        
        # Rekonstruiere den Tensor aus den Bytes
        np_reconstructed = np.frombuffer(unpadded, dtype=np.float32).reshape(10, 10)
        reconstructed = mlx.array(np_reconstructed)
        
        # Stelle sicher, dass der rekonstruierte Tensor dem Original entspricht
        # MLX hat keine direkte Vergleichsoperation wie torch.all/torch.eq
        # Daher Vergleich über NumPy
        np.testing.assert_array_equal(
            tensor.astype(np.array), reconstructed.astype(np.array)
        )

    def test_m_lingua_integration(self):
        """Testet die Integration mit M-LINGUA-Operationen."""
        # Beispiel für eine M-LINGUA-Operation
        operation = {
            "operation": "matrix_multiply",
            "language_input": "Multipliziere Matrix A mit Matrix B",
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
        
        # Erzeuge einen Schlüssel
        key, _ = derive_key_from_password(self.test_password)
        
        # Konvertiere Operation zu Bytes
        operation_bytes = json.dumps(operation).encode('utf-8')
        
        # Berechne HMAC für die Operation
        hmac_value = compute_tensor_hmac(operation_bytes, key, 
                                        {"type": "m_lingua", "version": "1.0"})
        
        # Stelle sicher, dass die HMAC-Berechnung konsistent ist
        hmac_value2 = compute_tensor_hmac(operation_bytes, key, 
                                         {"type": "m_lingua", "version": "1.0"})
        self.assertEqual(hmac_value, hmac_value2)


if __name__ == "__main__":
    unittest.main()
