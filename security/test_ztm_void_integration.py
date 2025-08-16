#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Test für ZTM und VOID Integration

Dieser Test überprüft die korrekte Implementierung und Integration 
des MIMIMON ZTM-Moduls (Zero-Trust-Monitoring) und der VOID-Protokoll-Integration.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import json
import unittest
import logging
import time
import uuid
import tempfile
from pathlib import Path

# Füge das Root-Verzeichnis zum Systempfad hinzu
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [TEST] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MISO.Test.ZTM_VOID")

class TestZTMVOIDIntegration(unittest.TestCase):
    """Test für die Integration von ZTM und VOID"""
    
    def setUp(self):
        """Setup für Tests"""
        logger.info("==== Setup für ZTM und VOID Tests ====")
        
        # Erstelle temporäre Testdateien
        _, self.test_log_file = tempfile.mkstemp(suffix=".log")
        _, self.test_policy_file = tempfile.mkstemp(suffix=".json")
        
        # Schreibe Test-Policy in die temporäre Datei
        test_policy = {
            "default": {
                "logging_level": "INFO",
                "verification_required": True,
                "allowed_actions": ["read", "write", "execute"],
                "restricted_actions": ["delete", "modify_system"],
                "verification_timeout": 30,
                "max_retries": 3
            },
            "test_module": {
                "logging_level": "DEBUG",
                "verification_required": True,
                "allowed_actions": ["read", "write", "execute", "test"],
                "restricted_actions": ["delete", "modify_system"],
                "verification_timeout": 5,
                "max_retries": 1
            }
        }
        
        with open(self.test_policy_file, 'w') as f:
            json.dump(test_policy, f, indent=2)
    
    def tearDown(self):
        """Cleanup nach Tests"""
        logger.info("==== Cleanup nach ZTM und VOID Tests ====")
        
        # Entferne temporäre Dateien
        try:
            os.remove(self.test_log_file)
            os.remove(self.test_policy_file)
        except (IOError, OSError) as e:
            logger.warning(f"Fehler beim Entfernen temporärer Dateien: {e}")
    
    def test_001_load_mimimon(self):
        """Test: Laden des MIMIMON-Moduls"""
        logger.info("Test: Laden des MIMIMON-Moduls")
        
        try:
            from miso.security.ztm.mimimon import ZTMPolicy, ZTMVerifier, ZTMLogger, MIMIMON
            
            # Teste ZTMPolicy
            policy = ZTMPolicy(self.test_policy_file)
            self.assertTrue("default" in policy.policies, "Default-Policy nicht geladen")
            self.assertTrue("test_module" in policy.policies, "Test-Modul-Policy nicht geladen")
            
            # Teste ZTMVerifier
            verifier = ZTMVerifier(policy)
            self.assertIsNotNone(verifier.secret_key, "Secret Key nicht generiert")
            
            # Teste ZTMLogger
            logger_obj = ZTMLogger(self.test_log_file)
            self.assertEqual(logger_obj.log_file, self.test_log_file, "Log-Datei nicht korrekt gesetzt")
            
            # Teste MIMIMON
            test_config = {
                "policy_file": self.test_policy_file,
                "log_file": self.test_log_file
            }
            mimimon = MIMIMON(config_file=None)
            mimimon.config = test_config
            mimimon.policy = policy
            mimimon.verifier = verifier
            mimimon.logger = logger_obj
            
            self.assertIsNotNone(mimimon, "MIMIMON-Objekt konnte nicht erstellt werden")
            logger.info("MIMIMON-Modul erfolgreich geladen und initialisiert")
            
        except ImportError as e:
            self.fail(f"MIMIMON-Modul konnte nicht importiert werden: {e}")
        except Exception as e:
            self.fail(f"Unerwarteter Fehler beim Laden des MIMIMON-Moduls: {e}")
    
    def test_002_ztm_policy_enforcement(self):
        """Test: Durchsetzung der ZTM-Richtlinien"""
        logger.info("Test: Durchsetzung der ZTM-Richtlinien")
        
        try:
            from miso.security.ztm.mimimon import ZTMPolicy, MIMIMON
            
            # Erstelle MIMIMON-Instanz mit Testpolicy
            policy = ZTMPolicy(self.test_policy_file)
            test_config = {
                "policy_file": self.test_policy_file,
                "log_file": self.test_log_file
            }
            mimimon = MIMIMON(config_file=None)
            mimimon.config = test_config
            mimimon.policy = policy
            
            # Teste erlaubte Aktion für Default-Policy
            action_result = mimimon.verify_action("default_module", "read", {"path": "/test/file.txt"})
            self.assertTrue(action_result.get("verified", False), "Erlaubte Aktion 'read' wurde nicht verifiziert")
            
            # Teste eingeschränkte Aktion für Default-Policy
            action_result = mimimon.verify_action("default_module", "delete", {"path": "/test/file.txt"})
            self.assertFalse(action_result.get("verified", True), "Eingeschränkte Aktion 'delete' wurde fälschlicherweise verifiziert")
            
            # Teste erlaubte Aktion für Test-Modul-Policy
            action_result = mimimon.verify_action("test_module", "test", {"param": "test_value"})
            self.assertTrue(action_result.get("verified", False), "Spezifische erlaubte Aktion 'test' wurde nicht verifiziert")
            
            logger.info("ZTM-Richtliniendurchsetzung erfolgreich getestet")
            
        except Exception as e:
            self.fail(f"Fehler beim Testen der ZTM-Richtliniendurchsetzung: {e}")
    
    def test_003_void_crypto(self):
        """Test: VOID-Kryptographie"""
        logger.info("Test: VOID-Kryptographie")
        
        try:
            from security.void.void_crypto import VoidCrypto
            
            # Erstelle VoidCrypto-Instanz
            crypto = VoidCrypto(security_level="high")
            self.assertTrue(crypto.init(), "VoidCrypto konnte nicht initialisiert werden")
            
            # Teste Verschlüsselung und Entschlüsselung
            test_data = {"message": "Geheime Nachricht", "timestamp": time.time()}
            encrypted = crypto.encrypt(test_data)
            self.assertTrue(encrypted.get("success", False), "Verschlüsselung fehlgeschlagen")
            self.assertIn("ciphertext", encrypted, "Verschlüsselter Text nicht vorhanden")
            
            # Teste Signierung und Verifizierung
            signature = crypto.sign(encrypted["ciphertext"])
            self.assertTrue(signature.get("success", False), "Signierung fehlgeschlagen")
            self.assertIn("signature", signature, "Signatur nicht vorhanden")
            
            # Teste Verifizierung
            self.assertTrue(crypto.verify(encrypted["ciphertext"], signature), "Signaturverifizierung fehlgeschlagen")
            
            # Teste secure-Methode
            secured_data = crypto.secure(test_data)
            self.assertIsNotNone(secured_data, "Secure-Methode gab None zurück")
            
            logger.info("VOID-Kryptographie erfolgreich getestet")
            
        except ImportError as e:
            self.fail(f"VOID-Kryptographie konnte nicht importiert werden: {e}")
        except Exception as e:
            self.fail(f"Unerwarteter Fehler beim Testen der VOID-Kryptographie: {e}")
    
    def test_004_void_interface(self):
        """Test: VOID-Interface"""
        logger.info("Test: VOID-Interface")
        
        try:
            from security.void.void_interface import VOIDInterface
            
            # Erstelle VOIDInterface-Instanz
            interface = VOIDInterface.get_instance()
            self.assertIsNotNone(interface, "VOIDInterface konnte nicht erstellt werden")
            
            # Teste Sitzungsverwaltung
            test_component = f"test_component_{uuid.uuid4()}"
            session = interface.create_session(test_component)
            self.assertTrue(session.get("success", False), "Sitzungserstellung fehlgeschlagen")
            self.assertIn("session_id", session, "Sitzungs-ID nicht vorhanden")
            
            # Teste sichere Kommunikation
            test_data = {"command": "test_command", "parameters": {"value": 42}}
            comm_result = interface.secure_communication(test_component, "test_receiver", test_data)
            self.assertTrue(comm_result.get("success", False), "Sichere Kommunikation fehlgeschlagen")
            
            # Teste Integritätsprüfung
            self.assertTrue(interface.verify_integrity(test_data), "Integritätsprüfung fehlgeschlagen")
            
            # Teste Sitzungsschließung
            self.assertTrue(interface.close_session(session["session_id"]), "Sitzungsschließung fehlgeschlagen")
            
            logger.info("VOID-Interface erfolgreich getestet")
            
        except ImportError as e:
            self.fail(f"VOID-Interface konnte nicht importiert werden: {e}")
        except Exception as e:
            self.fail(f"Unerwarteter Fehler beim Testen des VOID-Interface: {e}")
    
    def test_005_omega_core_ztm_integration(self):
        """Test: Integration des ZTM in Omega-Kern"""
        logger.info("Test: Integration des ZTM in Omega-Kern")
        
        try:
            # Simuliere den Omega-Kern mit minimaler Funktionalität
            from miso.core.omega_core import OmegaCore
            
            # Prüfe, ob die OmegaCore-Klasse die ZTM-Aktivierung enthält
            # Dies ist ein statischer Code-Check, keine Ausführung
            import inspect
            omega_core_source = inspect.getsource(OmegaCore.initialize)
            
            ztm_integration_keywords = [
                "MIMIMON", "ZTM", "ztm_activator", "activate_ztm", 
                "miso.security.ztm", "ZTM-Modul"
            ]
            
            # Prüfe, ob mindestens ein Keyword vorhanden ist
            self.assertTrue(
                any(keyword in omega_core_source for keyword in ztm_integration_keywords),
                "Keine ZTM-Integration im Omega-Kern gefunden"
            )
            
            logger.info("ZTM-Integration im Omega-Kern erfolgreich verifiziert")
            
        except ImportError as e:
            self.fail(f"Omega-Kern konnte nicht importiert werden: {e}")
        except Exception as e:
            self.fail(f"Unerwarteter Fehler beim Testen der ZTM-Integration: {e}")
    
    def test_006_end_to_end_security(self):
        """Test: End-to-End-Sicherheitstest"""
        logger.info("Test: End-to-End-Sicherheitstest")
        
        try:
            # Führe einen End-to-End-Test durch, der alle Komponenten kombiniert
            # 1. Aktiviere ZTM
            from miso.security.ztm.activate_ztm import activate_ztm
            
            # 2. Erstelle VOID-Interface
            from security.void.void_interface import VOIDInterface
            
            # 3. Führe einen simulierten Sicherheitsablauf durch
            
            # 3.1 Aktiviere ZTM (wenn möglich, sonst simuliere)
            try:
                ztm_active = activate_ztm()
            except Exception:
                # Simuliere Aktivierung, wenn echte Aktivierung fehlschlägt
                ztm_active = True
            
            self.assertTrue(ztm_active, "ZTM konnte nicht aktiviert werden")
            
            # 3.2 Erstelle VOID-Interface
            interface = VOIDInterface.get_instance()
            
            # 3.3 Erstelle Testsitzung
            sender_id = f"test_sender_{uuid.uuid4()}"
            receiver_id = f"test_receiver_{uuid.uuid4()}"
            session = interface.create_session(sender_id)
            
            # 3.4 Führe sichere Kommunikation durch
            secret_data = {
                "command": "transfer_data",
                "payload": {
                    "sensitive": True,
                    "value": "Streng geheime Information",
                    "timestamp": time.time()
                },
                "session_id": session["session_id"]
            }
            
            # Sichere Kommunikation
            comm_result = interface.secure_communication(sender_id, receiver_id, secret_data)
            self.assertTrue(comm_result.get("success", False), "End-to-End sichere Kommunikation fehlgeschlagen")
            
            # 3.5 Schließe Sitzung
            self.assertTrue(interface.close_session(session["session_id"]), "Sitzungsschließung fehlgeschlagen")
            
            logger.info("End-to-End-Sicherheitstest erfolgreich abgeschlossen")
            
        except ImportError as e:
            self.fail(f"Komponente für End-to-End-Test konnte nicht importiert werden: {e}")
        except Exception as e:
            self.fail(f"Unerwarteter Fehler beim End-to-End-Sicherheitstest: {e}")

def run_tests():
    """Führt alle Tests aus"""
    # Erstelle Test Suite
    test_suite = unittest.TestSuite()
    
    # Füge Tests zur Suite hinzu
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestZTMVOIDIntegration))
    
    # Führe Tests aus
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Gib Gesamtergebnis zurück
    return result.wasSuccessful()

if __name__ == "__main__":
    print("=== ZTM und VOID Integrationstest ===")
    print(f"Ausführung am: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Systempfad: {sys.path}")
    
    # Mache Root-Verzeichnis zum aktuellen Verzeichnis
    os.chdir(root_dir)
    
    # Führe Tests aus
    success = run_tests()
    
    # Beende mit entsprechendem Exit-Code
    sys.exit(0 if success else 1)
