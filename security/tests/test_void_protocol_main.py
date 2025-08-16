#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unittests für das VOID-Protokoll 3.0 Hauptmodul.
"""

import os
import sys
import time
import unittest
import tempfile
import shutil
import json
from pathlib import Path

# Pfad zum VOID-Protokoll hinzufügen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from void.void_protocol import (
    VoidProtocol, VoidMessage, VoidEndpoint, HandshakeResult,
    initialize, get_instance, encrypt, decrypt, handshake, verify_handshake,
    register_endpoint, close
)


class TestVoidProtocol(unittest.TestCase):
    """Tests für das VOID-Protokoll Hauptmodul."""
    
    def setUp(self):
        """Test-Setup: Temporäres Verzeichnis für Schlüssel erstellen."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialisiere das VOID-Protokoll für jeden Test neu
        self.module_a = VoidProtocol("TestModuleA", self.temp_dir)
        self.module_b = VoidProtocol("TestModuleB", self.temp_dir)
        
        # Registriere die Module gegenseitig
        self.module_a.register_trusted_endpoint(
            self.module_b.endpoint_id,
            self.module_b.module_name,
            self.module_b.endpoint.kyber_keypair.public_key,
            self.module_b.endpoint.dilithium_keypair.public_key
        )
        
        self.module_b.register_trusted_endpoint(
            self.module_a.endpoint_id,
            self.module_a.module_name,
            self.module_a.endpoint.kyber_keypair.public_key,
            self.module_a.endpoint.dilithium_keypair.public_key
        )
    
    def tearDown(self):
        """Test-Cleanup: Temporäres Verzeichnis entfernen."""
        shutil.rmtree(self.temp_dir)
    
    def test_protocol_initialization(self):
        """Test für die Protokoll-Initialisierung."""
        # Überprüfe, ob die Endpunkte korrekt initialisiert wurden
        self.assertIsInstance(self.module_a.endpoint, VoidEndpoint)
        self.assertIsInstance(self.module_b.endpoint, VoidEndpoint)
        
        # Überprüfe, ob Schlüsseldateien erstellt wurden
        key_file_a = os.path.join(self.temp_dir, f"{self.module_a.module_name}_keys.json")
        key_file_b = os.path.join(self.temp_dir, f"{self.module_b.module_name}_keys.json")
        
        self.assertTrue(os.path.exists(key_file_a))
        self.assertTrue(os.path.exists(key_file_b))
        
        # Überprüfe Schlüsseldateiinhalt
        with open(key_file_a, 'r') as f:
            keys_data = json.load(f)
            self.assertEqual(keys_data["module_name"], self.module_a.module_name)
            self.assertIn("kyber_public_key", keys_data)
            self.assertIn("kyber_private_key", keys_data)
            self.assertIn("dilithium_public_key", keys_data)
            self.assertIn("dilithium_private_key", keys_data)
    
    def test_handshake_and_encryption(self):
        """Test für Handshake und Verschlüsselung/Entschlüsselung."""
        # Führe Handshake von A zu B durch
        session_id, session_key = self.module_a.perform_handshake(self.module_b.endpoint_id)
        
        # Überprüfe Ergebnisse
        self.assertIsInstance(session_id, str)
        self.assertIsInstance(session_key, bytes)
        self.assertIn(session_id, self.module_a.endpoint.active_sessions)
        
        # Verschlüssele eine Nachricht von A zu B
        plaintext = b"Dies ist eine geheime Nachricht von A zu B."
        void_message = self.module_a.encrypt_message(session_id, plaintext)
        
        # Überprüfe verschlüsselte Nachricht
        self.assertIsInstance(void_message, VoidMessage)
        self.assertEqual(void_message.sender_id, self.module_a.endpoint_id)
        self.assertEqual(void_message.recipient_id, self.module_b.endpoint_id)
        self.assertEqual(void_message.session_id, session_id)
        
        # Füge manuelle Session auf der B-Seite hinzu, da wir den vollständigen Handshake nicht simulieren
        # In einer realen Implementierung würde B die Session aus dem Handshake-Ergebnis erstellen
        self.module_b.endpoint.active_sessions[session_id] = {
            'sender_endpoint_id': self.module_a.endpoint_id,
            'session_key': session_key,
            'created_at': time.time(),
            'expires_at': time.time() + 3600,  # 1 Stunde
            'last_used': time.time(),
            'messages_sent': 0,
            'messages_received': 0
        }
        
        # Entschlüssele die Nachricht bei B
        decrypted = self.module_b.decrypt_message(void_message)
        
        # Überprüfe, ob die entschlüsselte Nachricht mit dem Original übereinstimmt
        self.assertEqual(plaintext, decrypted)
    
    def test_session_rotation(self):
        """Test für die Session-Schlüsselrotation."""
        # Führe Handshake durch
        session_id, session_key = self.module_a.perform_handshake(self.module_b.endpoint_id)
        
        # Speichere den ursprünglichen Schlüssel
        original_key = session_key
        
        # Aufruf der internen Methode für Schlüsselrotation
        self.module_a._rotate_session_keys()
        
        # Überprüfe, ob der Schlüssel rotiert wurde
        # In unserem Testfall wird der Schlüssel nicht rotiert, da er gerade erst erstellt wurde
        # und das Rotationsintervall noch nicht erreicht hat
        self.assertIn(session_id, self.module_a.endpoint.active_sessions)
        
        # Manuelles Rotieren durch Manipulation des Erstellungszeitpunkts
        session = self.module_a.endpoint.active_sessions[session_id]
        session['created_at'] = time.time() - self.module_a.REKEY_INTERVAL_SECONDS - 1
        
        # Jetzt sollte die Rotation durchgeführt werden
        self.module_a._rotate_session_keys()
        
        # Überprüfe, ob der Schlüssel rotiert wurde
        rotated_key = self.module_a.endpoint.active_sessions[session_id]['session_key']
        self.assertNotEqual(original_key, rotated_key)
    
    def test_api_functions(self):
        """Test für die vereinfachten API-Funktionen."""
        # Initialisiere das Protokoll mit der API
        temp_dir2 = tempfile.mkdtemp()
        try:
            # Initialisiere das VOID-Protokoll
            protocol_instance = initialize("APITestModule", temp_dir2)
            
            # Überprüfe, ob die Instanz zurückgegeben wird
            self.assertIsInstance(protocol_instance, VoidProtocol)
            
            # Überprüfe get_instance
            instance = get_instance()
            self.assertIs(instance, protocol_instance)
            
            # Schließe das Protokoll
            close()
        finally:
            shutil.rmtree(temp_dir2)


class TestVoidProtocolIntegration(unittest.TestCase):
    """Integrationstests für das VOID-Protokoll zwischen mehreren Modulen."""
    
    def setUp(self):
        """Test-Setup: Drei Module für eine End-to-End-Testkette erstellen."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Drei Module erstellen, die in einer Kette kommunizieren sollen
        self.t_math = VoidProtocol("T-Math", self.temp_dir)
        self.prism = VoidProtocol("PRISM", self.temp_dir)
        self.echo = VoidProtocol("ECHO", self.temp_dir)
        
        # Vertrauensbeziehungen aufbauen
        # T-Math -> PRISM
        self.t_math.register_trusted_endpoint(
            self.prism.endpoint_id,
            self.prism.module_name,
            self.prism.endpoint.kyber_keypair.public_key,
            self.prism.endpoint.dilithium_keypair.public_key
        )
        
        # PRISM -> T-Math
        self.prism.register_trusted_endpoint(
            self.t_math.endpoint_id,
            self.t_math.module_name,
            self.t_math.endpoint.kyber_keypair.public_key,
            self.t_math.endpoint.dilithium_keypair.public_key
        )
        
        # PRISM -> ECHO
        self.prism.register_trusted_endpoint(
            self.echo.endpoint_id,
            self.echo.module_name,
            self.echo.endpoint.kyber_keypair.public_key,
            self.echo.endpoint.dilithium_keypair.public_key
        )
        
        # ECHO -> PRISM
        self.echo.register_trusted_endpoint(
            self.prism.endpoint_id,
            self.prism.module_name,
            self.prism.endpoint.kyber_keypair.public_key,
            self.prism.endpoint.dilithium_keypair.public_key
        )
    
    def tearDown(self):
        """Test-Cleanup: Temporäres Verzeichnis entfernen."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_message_chain(self):
        """Test für eine End-to-End-Nachrichtenkette: T-Math -> PRISM -> ECHO."""
        # Handshake zwischen T-Math und PRISM
        tmath_prism_session, tmath_prism_key = self.t_math.perform_handshake(self.prism.endpoint_id)
        
        # Handshake zwischen PRISM und ECHO
        prism_echo_session, prism_echo_key = self.prism.perform_handshake(self.echo.endpoint_id)
        
        # Simuliere den vollständigen Handshake, indem wir die Sessions manuell auf der anderen Seite erstellen
        self.prism.endpoint.active_sessions[tmath_prism_session] = {
            'sender_endpoint_id': self.t_math.endpoint_id,
            'session_key': tmath_prism_key,
            'created_at': time.time(),
            'expires_at': time.time() + 3600,
            'last_used': time.time(),
            'messages_sent': 0,
            'messages_received': 0
        }
        
        self.echo.endpoint.active_sessions[prism_echo_session] = {
            'sender_endpoint_id': self.prism.endpoint_id,
            'session_key': prism_echo_key,
            'created_at': time.time(),
            'expires_at': time.time() + 3600,
            'last_used': time.time(),
            'messages_sent': 0,
            'messages_received': 0
        }
        
        # Originalnachricht von T-Math
        original_message = b"TENSOR_CALCULATION_RESULT: [[1.0, 2.0], [3.0, 4.0]]"
        
        # T-Math -> PRISM
        tmath_message = self.t_math.encrypt_message(tmath_prism_session, original_message)
        prism_received = self.prism.decrypt_message(tmath_message)
        
        # Überprüfe erste Übertragung
        self.assertEqual(original_message, prism_received)
        
        # PRISM modifiziert und leitet die Nachricht an ECHO weiter
        prism_processed = prism_received + b" | PRISM_PROCESSED"
        prism_message = self.prism.encrypt_message(prism_echo_session, prism_processed)
        echo_received = self.echo.decrypt_message(prism_message)
        
        # Überprüfe zweite Übertragung
        self.assertEqual(prism_processed, echo_received)
        
        # Überprüfe die komplette Kette
        self.assertIn(b"TENSOR_CALCULATION_RESULT", echo_received)
        self.assertIn(b"PRISM_PROCESSED", echo_received)


if __name__ == '__main__':
    unittest.main()
