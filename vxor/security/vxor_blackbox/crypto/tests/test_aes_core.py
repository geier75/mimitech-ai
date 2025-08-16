#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Umfassendes Testmodul für die AES-256-GCM Implementierung.

Dieses Modul testet alle Aspekte der AES-Implementierung:
- Basisverschlüsselung und -entschlüsselung
- Schlüsselgenerierung und -verwaltung
- Passwortbasierte Verschlüsselung
- Dateioperationen
- Integration mit dem Logging-System
- Fehlerbehandlung und Edge Cases
- Performance-Tests für große Datenmengen

Alle Tests sind auf die spezifischen Anforderungen der T-Mathematics Engine zugeschnitten.
"""

import unittest
import os
import tempfile
import time
import io
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple

# Import der zu testenden Komponente
from ..aes_core import AESCipher, get_aes_cipher

# Import der Crypto-Utilities für erweiterte Tests
from ..crypto_utils import (
    derive_key_from_password,
    compute_hmac,
    pad_data,
    unpad_data,
    constant_time_compare
)


class TestAESCore(unittest.TestCase):
    """Testet die AES-256-GCM Implementierung für VXOR AI Blackbox."""
    
    def setUp(self):
        """Initialisiert die Testumgebung."""
        self.aes = AESCipher()
        self.test_data = b"VXOR AI T-Mathematics Engine Test Data"
        self.test_password = "VXOR-AI-Secure-Password!123"
        
        # Temporäres Verzeichnis für Dateitests
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Räumt die Testumgebung auf."""
        # Lösche temporäres Verzeichnis
        shutil.rmtree(self.temp_dir)
    
    # ===== Basis-Verschlüsselungstests =====
    
    def test_encrypt_decrypt_basic(self):
        """Testet die grundlegende Verschlüsselung und Entschlüsselung."""
        # Generiere einen Schlüssel
        key = self.aes.generate_key()
        
        # Verschlüssele Daten
        encrypted = self.aes.encrypt(self.test_data, key)
        
        # Stelle sicher, dass die verschlüsselten Daten nicht den Originaldaten entsprechen
        self.assertNotEqual(encrypted, self.test_data)
        
        # Entschlüssele Daten
        decrypted = self.aes.decrypt(encrypted, key)
        
        # Stelle sicher, dass die entschlüsselten Daten den Originaldaten entsprechen
        self.assertEqual(decrypted, self.test_data)
    
    def test_encrypt_decrypt_empty(self):
        """Testet Verschlüsselung und Entschlüsselung von leeren Daten."""
        # Generiere einen Schlüssel
        key = self.aes.generate_key()
        
        # Verschlüssele leere Daten
        empty_data = b""
        encrypted = self.aes.encrypt(empty_data, key)
        
        # Stelle sicher, dass die verschlüsselten Daten nicht leer sind
        self.assertNotEqual(encrypted, empty_data)
        
        # Entschlüssele Daten
        decrypted = self.aes.decrypt(encrypted, key)
        
        # Stelle sicher, dass die entschlüsselten Daten leer sind
        self.assertEqual(decrypted, empty_data)
    
    def test_encrypt_decrypt_large_data(self):
        """Testet Verschlüsselung und Entschlüsselung großer Datenmengen."""
        # Generiere einen Schlüssel
        key = self.aes.generate_key()
        
        # Generiere große Testdaten (1 MB)
        large_data = os.urandom(1024 * 1024)
        
        # Verschlüssele Daten
        start_time = time.time()
        encrypted = self.aes.encrypt(large_data, key)
        encrypt_time = time.time() - start_time
        
        # Entschlüssele Daten
        start_time = time.time()
        decrypted = self.aes.decrypt(encrypted, key)
        decrypt_time = time.time() - start_time
        
        # Stelle sicher, dass die entschlüsselten Daten den Originaldaten entsprechen
        self.assertEqual(decrypted, large_data)
        
        # Protokolliere Performance-Informationen
        print(f"Große Daten (1 MB): Verschlüsselung: {encrypt_time:.4f}s, Entschlüsselung: {decrypt_time:.4f}s")
    
    # ===== Schlüsselgenerierung und -verwaltung =====
    
    def test_key_generation(self):
        """Testet die Schlüsselgenerierung."""
        # Generiere zwei Schlüssel
        key1 = self.aes.generate_key()
        key2 = self.aes.generate_key()
        
        # Stelle sicher, dass die Schlüssel 32 Byte (256 Bit) lang sind
        self.assertEqual(len(key1), 32)
        self.assertEqual(len(key2), 32)
        
        # Stelle sicher, dass die Schlüssel unterschiedlich sind
        self.assertNotEqual(key1, key2)
    
    def test_iv_generation(self):
        """Testet die IV-Generierung."""
        # Generiere zwei IVs
        iv1 = self.aes.generate_iv()
        iv2 = self.aes.generate_iv()
        
        # Stelle sicher, dass die IVs 12 Byte (96 Bit) lang sind
        self.assertEqual(len(iv1), 12)
        self.assertEqual(len(iv2), 12)
        
        # Stelle sicher, dass die IVs unterschiedlich sind
        self.assertNotEqual(iv1, iv2)
    
    # ===== Passwortbasierte Verschlüsselung =====
    
    def test_encrypt_decrypt_with_password(self):
        """Testet die passwortbasierte Verschlüsselung und Entschlüsselung."""
        # Passwort
        password = self.test_password
        
        # Verschlüssele Daten mit Passwort
        encrypted = self.aes.encrypt_with_password(self.test_data, password)
        
        # Stelle sicher, dass die verschlüsselten Daten nicht den Originaldaten entsprechen
        self.assertNotEqual(encrypted, self.test_data)
        
        # Entschlüssele Daten mit Passwort
        decrypted = self.aes.decrypt_with_password(encrypted, password)
        
        # Stelle sicher, dass die entschlüsselten Daten den Originaldaten entsprechen
        self.assertEqual(decrypted, self.test_data)
    
    def test_encrypt_decrypt_with_wrong_password(self):
        """Testet die Entschlüsselung mit falschem Passwort."""
        # Passwörter
        correct_password = self.test_password
        wrong_password = "WrongPassword123!"
        
        # Verschlüssele Daten mit korrektem Passwort
        encrypted = self.aes.encrypt_with_password(self.test_data, correct_password)
        
        # Versuche, mit falschem Passwort zu entschlüsseln
        with self.assertRaises(ValueError):
            self.aes.decrypt_with_password(encrypted, wrong_password)
    
    # ===== Dateioperationen =====
    
    def test_encrypt_decrypt_file(self):
        """Testet die Dateiverschlüsselung und -entschlüsselung."""
        # Erstelle Testdatei
        source_path = os.path.join(self.temp_dir, "test_file.txt")
        encrypted_path = os.path.join(self.temp_dir, "test_file.enc")
        decrypted_path = os.path.join(self.temp_dir, "test_file_decrypted.txt")
        
        # Schreibe Testdaten in Datei
        with open(source_path, "wb") as f:
            f.write(self.test_data)
        
        # Generiere einen Schlüssel
        key = self.aes.generate_key()
        
        # Verschlüssele Datei
        self.aes.encrypt_file(source_path, encrypted_path, key)
        
        # Stelle sicher, dass die verschlüsselte Datei existiert und anders ist
        self.assertTrue(os.path.exists(encrypted_path))
        with open(encrypted_path, "rb") as f:
            encrypted_data = f.read()
        self.assertNotEqual(encrypted_data, self.test_data)
        
        # Entschlüssele Datei
        self.aes.decrypt_file(encrypted_path, decrypted_path, key)
        
        # Stelle sicher, dass die entschlüsselte Datei existiert und gleich der Originaldatei ist
        self.assertTrue(os.path.exists(decrypted_path))
        with open(decrypted_path, "rb") as f:
            decrypted_data = f.read()
        self.assertEqual(decrypted_data, self.test_data)
    
    def test_encrypt_decrypt_large_file(self):
        """Testet die Verschlüsselung und Entschlüsselung großer Dateien."""
        # Erstelle große Testdatei (5 MB)
        source_path = os.path.join(self.temp_dir, "large_file.bin")
        encrypted_path = os.path.join(self.temp_dir, "large_file.enc")
        decrypted_path = os.path.join(self.temp_dir, "large_file_decrypted.bin")
        
        # Generiere zufällige Daten
        large_data = os.urandom(5 * 1024 * 1024)
        
        # Schreibe Daten in Datei
        with open(source_path, "wb") as f:
            f.write(large_data)
        
        # Generiere einen Schlüssel
        key = self.aes.generate_key()
        
        # Verschlüssele Datei
        start_time = time.time()
        self.aes.encrypt_file(source_path, encrypted_path, key)
        encrypt_time = time.time() - start_time
        
        # Entschlüssele Datei
        start_time = time.time()
        self.aes.decrypt_file(encrypted_path, decrypted_path, key)
        decrypt_time = time.time() - start_time
        
        # Vergleiche Originaldatei mit entschlüsselter Datei
        with open(decrypted_path, "rb") as f:
            decrypted_data = f.read()
        self.assertEqual(decrypted_data, large_data)
        
        # Protokolliere Performance-Informationen
        print(f"Große Datei (5 MB): Verschlüsselung: {encrypt_time:.4f}s, Entschlüsselung: {decrypt_time:.4f}s")
    
    # ===== Zusätzliche Funktionalitäten =====
    
    def test_key_wrapping(self):
        """Testet die Schlüsselumhüllung (Key Wrapping)."""
        # Generiere KEK (Key Encryption Key) und DEK (Data Encryption Key)
        kek = self.aes.generate_key()
        dek = self.aes.generate_key()
        
        # Umhülle DEK mit KEK
        wrapped_key = self.aes.wrap_key(dek, kek)
        
        # Stelle sicher, dass der umhüllte Schlüssel nicht dem Original-DEK entspricht
        self.assertNotEqual(wrapped_key, dek)
        
        # Enthülle DEK
        unwrapped_key = self.aes.unwrap_key(wrapped_key, kek)
        
        # Stelle sicher, dass der enthüllte Schlüssel dem Original-DEK entspricht
        self.assertEqual(unwrapped_key, dek)
    
    def test_hmac_verification(self):
        """Testet die HMAC-Verifikation."""
        # Generiere Schlüssel
        key = self.aes.generate_key()
        hmac_key = self.aes.generate_key()
        
        # Verschlüssele Daten
        encrypted = self.aes.encrypt(self.test_data, key)
        
        # Berechne HMAC
        hmac_value = compute_hmac(encrypted, hmac_key)
        
        # Verifiziere HMAC (sollte erfolgreich sein)
        self.assertTrue(constant_time_compare(hmac_value, compute_hmac(encrypted, hmac_key)))
        
        # Verändere verschlüsselte Daten
        tampered_encrypted = encrypted[:-1] + bytes([encrypted[-1] ^ 1])
        
        # Verifiziere HMAC (sollte fehlschlagen)
        self.assertFalse(constant_time_compare(hmac_value, compute_hmac(tampered_encrypted, hmac_key)))
    
    # ===== Fehlerfälle und Robustheitstests =====
    
    def test_decrypt_tampered_data(self):
        """Testet die Entschlüsselung manipulierter Daten."""
        # Generiere einen Schlüssel
        key = self.aes.generate_key()
        
        # Verschlüssele Daten
        encrypted = self.aes.encrypt(self.test_data, key)
        
        # Manipuliere verschlüsselte Daten (verändere letztes Byte)
        tampered_encrypted = encrypted[:-1] + bytes([encrypted[-1] ^ 1])
        
        # Versuche, manipulierte Daten zu entschlüsseln
        with self.assertRaises(ValueError):
            self.aes.decrypt(tampered_encrypted, key)
    
    def test_decrypt_with_wrong_key(self):
        """Testet die Entschlüsselung mit falschem Schlüssel."""
        # Generiere Schlüssel
        correct_key = self.aes.generate_key()
        wrong_key = self.aes.generate_key()
        
        # Verschlüssele Daten mit korrektem Schlüssel
        encrypted = self.aes.encrypt(self.test_data, correct_key)
        
        # Versuche, mit falschem Schlüssel zu entschlüsseln
        with self.assertRaises(ValueError):
            self.aes.decrypt(encrypted, wrong_key)
    
    def test_encrypt_decrypt_special_characters(self):
        """Testet die Verschlüsselung und Entschlüsselung von Daten mit Sonderzeichen."""
        # Testdaten mit Sonderzeichen
        special_data = b"VXOR AI \x00\x01\x02\x03\xff\xfe\xfd\xfc\t\n\r\\"
        
        # Generiere einen Schlüssel
        key = self.aes.generate_key()
        
        # Verschlüssele Daten
        encrypted = self.aes.encrypt(special_data, key)
        
        # Entschlüssele Daten
        decrypted = self.aes.decrypt(encrypted, key)
        
        # Stelle sicher, dass die entschlüsselten Daten den Originaldaten entsprechen
        self.assertEqual(decrypted, special_data)
    
    # ===== Singleton-Zugriff =====
    
    def test_singleton_access(self):
        """Testet den Singleton-Zugriff auf die AESCipher-Instanz."""
        # Hole Instanz über Singleton-Zugriffsfunktion
        cipher1 = get_aes_cipher()
        cipher2 = get_aes_cipher()
        
        # Stelle sicher, dass es sich um dieselbe Instanz handelt
        self.assertIs(cipher1, cipher2)
        
        # Stelle sicher, dass die Instanz funktioniert
        key = cipher1.generate_key()
        encrypted = cipher1.encrypt(self.test_data, key)
        decrypted = cipher2.decrypt(encrypted, key)
        self.assertEqual(decrypted, self.test_data)
    
    # ===== Integrationstests mit T-Mathematics Engine =====
    
    def test_tensor_data_encryption(self):
        """Simuliert die Verschlüsselung und Entschlüsselung von Tensor-Daten."""
        # Simuliere Tensor-Daten
        tensor_shape = (10, 10)
        tensor_data = os.urandom(4 * 10 * 10)  # 4 Bytes pro Float32-Wert
        
        # Metadaten
        metadata = {
            "shape": tensor_shape,
            "dtype": "float32",
            "format": "numpy"
        }
        
        # Konvertiere Metadaten zu JSON
        metadata_json = json.dumps(metadata).encode('utf-8')
        
        # Generiere einen Schlüssel
        key = self.aes.generate_key()
        
        # Verschlüssele Metadaten und Tensor-Daten
        encrypted_metadata = self.aes.encrypt(metadata_json, key)
        encrypted_tensor = self.aes.encrypt(tensor_data, key)
        
        # Entschlüssele Metadaten und Tensor-Daten
        decrypted_metadata_json = self.aes.decrypt(encrypted_metadata, key)
        decrypted_tensor = self.aes.decrypt(encrypted_tensor, key)
        
        # Stelle sicher, dass die entschlüsselten Daten den Originaldaten entsprechen
        self.assertEqual(decrypted_metadata_json, metadata_json)
        self.assertEqual(decrypted_tensor, tensor_data)
        
        # Rekonstruiere Metadaten
        decrypted_metadata = json.loads(decrypted_metadata_json.decode('utf-8'))
        
        # Bei JSON-Serialisierung/Deserialisierung werden Tupel als Listen dargestellt
        # Entweder konvertieren wir die deserialisierte Form zurück zu einem Tupel...
        if 'shape' in decrypted_metadata and isinstance(decrypted_metadata['shape'], list):
            decrypted_metadata['shape'] = tuple(decrypted_metadata['shape'])
            
        # ...oder wir prüfen die Werte einzeln
        self.assertEqual(decrypted_metadata, metadata)
    
    def test_m_lingua_operation_encryption(self):
        """Simuliert die Verschlüsselung und Entschlüsselung von M-LINGUA-Operationen."""
        # Simuliere M-LINGUA-Operation
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
        
        # Konvertiere Operation zu JSON
        operation_json = json.dumps(operation).encode('utf-8')
        
        # Generiere einen Schlüssel
        key = self.aes.generate_key()
        
        # Verschlüssele Operation
        encrypted_operation = self.aes.encrypt(operation_json, key)
        
        # Entschlüssele Operation
        decrypted_operation_json = self.aes.decrypt(encrypted_operation, key)
        
        # Stelle sicher, dass die entschlüsselte Operation der Original-Operation entspricht
        self.assertEqual(decrypted_operation_json, operation_json)
        
        # Rekonstruiere Operation
        decrypted_operation = json.loads(decrypted_operation_json.decode('utf-8'))
        self.assertEqual(decrypted_operation, operation)


if __name__ == "__main__":
    unittest.main()
