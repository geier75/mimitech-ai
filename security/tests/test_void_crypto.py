#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unittests für die kryptografischen Funktionen des VOID-Protokolls 3.0.
"""

import os
import sys
import time
import unittest
import tempfile
import shutil
from pathlib import Path

# Pfad zum VOID-Protokoll hinzufügen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from void.void_crypto import (
    KyberSimulator, DilithiumSimulator, AESGCMCipher, 
    KeyPair, EncryptedData
)


class TestKyberSimulator(unittest.TestCase):
    """Tests für den KyberSimulator (Post-Quanten-Schlüsselaustausch)."""
    
    def test_keygen(self):
        """Test für die Schlüsselgenerierung."""
        keypair = KyberSimulator.keygen()
        
        # Überprüfe, ob die Schlüssel die richtigen Typen und Längen haben
        self.assertIsInstance(keypair, KeyPair)
        self.assertIsInstance(keypair.public_key, bytes)
        self.assertIsInstance(keypair.private_key, bytes)
        self.assertGreater(len(keypair.public_key), 0)
        self.assertGreater(len(keypair.private_key), 0)
    
    def test_encaps_decaps(self):
        """Test für Encapsulation und Decapsulation (Key Exchange)."""
        # Erzeuge Schlüsselpaar
        keypair = KyberSimulator.keygen()
        
        # Führe Encapsulation durch
        shared_secret, ciphertext = KyberSimulator.encaps(keypair.public_key)
        
        # Überprüfe Ergebnisse
        self.assertIsInstance(shared_secret, bytes)
        self.assertIsInstance(ciphertext, bytes)
        self.assertGreater(len(shared_secret), 0)
        self.assertGreater(len(ciphertext), 0)
        
        # Führe Decapsulation durch
        recovered_secret = KyberSimulator.decaps(ciphertext, keypair.private_key)
        
        # Überprüfe, ob der wiederhergestellte Secret mit dem originalen übereinstimmt
        self.assertEqual(shared_secret, recovered_secret)


class TestDilithiumSimulator(unittest.TestCase):
    """Tests für den DilithiumSimulator (Post-Quanten-Signatur)."""
    
    def test_keygen(self):
        """Test für die Schlüsselgenerierung."""
        keypair = DilithiumSimulator.keygen()
        
        # Überprüfe, ob die Schlüssel die richtigen Typen und Längen haben
        self.assertIsInstance(keypair, KeyPair)
        self.assertIsInstance(keypair.public_key, bytes)
        self.assertIsInstance(keypair.private_key, bytes)
        self.assertGreater(len(keypair.public_key), 0)
        self.assertGreater(len(keypair.private_key), 0)
    
    def test_sign_verify(self):
        """Test für Signieren und Verifizieren."""
        # Erzeuge Schlüsselpaar
        keypair = DilithiumSimulator.keygen()
        
        # Signiere eine Nachricht
        message = b"Dies ist eine Testnachricht, die signiert werden soll."
        signature = DilithiumSimulator.sign(message, keypair.private_key)
        
        # Überprüfe Signatur
        self.assertIsInstance(signature, bytes)
        self.assertGreater(len(signature), 0)
        
        # Verifiziere die Signatur
        is_valid = DilithiumSimulator.verify(message, signature, keypair.public_key)
        self.assertTrue(is_valid)
        
        # Verifiziere mit manipulierter Nachricht
        manipulated_message = b"Dies ist eine MANIPULIERTE Testnachricht."
        is_valid = DilithiumSimulator.verify(manipulated_message, signature, keypair.public_key)
        self.assertFalse(is_valid)


class TestAESGCMCipher(unittest.TestCase):
    """Tests für den AES-256-GCM Verschlüsselungsciphertext."""
    
    def test_encrypt_decrypt(self):
        """Test für Verschlüsselung und Entschlüsselung."""
        # Erzeuge Schlüssel (32 Bytes für AES-256)
        key = os.urandom(32)
        
        # Verschlüssele eine Nachricht
        plaintext = b"Dies ist eine geheime Nachricht, die verschlüsselt werden soll."
        encrypted_data = AESGCMCipher.encrypt(plaintext, key)
        
        # Überprüfe verschlüsselte Daten
        self.assertIsInstance(encrypted_data, EncryptedData)
        self.assertIsInstance(encrypted_data.ciphertext, bytes)
        self.assertIsInstance(encrypted_data.nonce, bytes)
        self.assertIsInstance(encrypted_data.tag, bytes)
        self.assertIsInstance(encrypted_data.timestamp, float)
        self.assertIsInstance(encrypted_data.key_id, str)
        
        # Entschlüssele die Nachricht
        decrypted = AESGCMCipher.decrypt(encrypted_data, key)
        
        # Überprüfe, ob die entschlüsselte Nachricht mit dem Original übereinstimmt
        self.assertEqual(plaintext, decrypted)
    
    def test_decrypt_with_wrong_key(self):
        """Test für Entschlüsselung mit falschem Schlüssel."""
        # Erzeuge Schlüssel
        correct_key = os.urandom(32)
        wrong_key = os.urandom(32)
        
        # Verschlüssele eine Nachricht
        plaintext = b"Dies ist eine geheime Nachricht, die verschlüsselt werden soll."
        encrypted_data = AESGCMCipher.encrypt(plaintext, correct_key)
        
        # Versuche, mit falschem Schlüssel zu entschlüsseln
        with self.assertRaises(ValueError):
            AESGCMCipher.decrypt(encrypted_data, wrong_key)


if __name__ == '__main__':
    unittest.main()
