#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unittests für das VOID-Kontext-Modul des VOID-Protokolls 3.0.
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
from void.void_context import VoidContext, ContextVerifier


class TestVoidContext(unittest.TestCase):
    """Tests für den VOID-Kontext-Verifier."""
    
    def setUp(self):
        """Test-Setup: Temporäres Verzeichnis für Kontexte erstellen."""
        self.temp_dir = tempfile.mkdtemp()
        self.context_verifier = ContextVerifier(self.temp_dir)
    
    def tearDown(self):
        """Test-Cleanup: Temporäres Verzeichnis entfernen."""
        shutil.rmtree(self.temp_dir)
    
    def test_context_creation(self):
        """Test für die Kontexterstellung."""
        # Erstelle einen Kontext
        context = self.context_verifier.create_context_token(
            module_name="TestModule",
            environment="testing",
            security_level="HIGH",
            validity_seconds=3600,
            attributes={"test_attr": "test_value"}
        )
        
        # Überprüfe, ob der Kontext korrekt erstellt wurde
        self.assertIsInstance(context, VoidContext)
        self.assertEqual(context.module_name, "TestModule")
        self.assertEqual(context.environment, "testing")
        self.assertEqual(context.security_level, "HIGH")
        self.assertGreater(context.expires_at, time.time())
        self.assertEqual(context.attributes.get("test_attr"), "test_value")
        
        # Überprüfe, ob die Kontextdatei erstellt wurde
        context_file = os.path.join(self.temp_dir, f"{context.context_id}.json")
        self.assertTrue(os.path.exists(context_file))
    
    def test_context_verification(self):
        """Test für die Kontextverifizierung."""
        # Erstelle einen Kontext
        context = self.context_verifier.create_context_token(
            module_name="TestModule",
            environment="testing",
            security_level="HIGH"
        )
        
        # Verifiziere den Kontext
        is_valid = self.context_verifier.verify(context)
        self.assertTrue(is_valid)
        
        # Manipuliere den Kontext und verifiziere erneut
        manipulated_context = VoidContext(
            context_id=context.context_id,
            module_name=context.module_name,
            host_id=context.host_id,
            boot_time=context.boot_time,
            environment="production",  # Geändert!
            security_level=context.security_level,
            signature=context.signature,
            issued_at=context.issued_at,
            expires_at=context.expires_at,
            attributes=context.attributes
        )
        
        is_valid = self.context_verifier.verify(manipulated_context)
        self.assertFalse(is_valid)
    
    def test_context_loading(self):
        """Test für das Laden eines Kontexts."""
        # Erstelle einen Kontext
        original_context = self.context_verifier.create_context_token(
            module_name="TestModule",
            environment="testing",
            security_level="HIGH"
        )
        
        # Lade den Kontext
        loaded_context = self.context_verifier.load_context(original_context.context_id)
        
        # Überprüfe, ob der geladene Kontext mit dem Original übereinstimmt
        self.assertIsNotNone(loaded_context)
        self.assertEqual(loaded_context.context_id, original_context.context_id)
        self.assertEqual(loaded_context.module_name, original_context.module_name)
        self.assertEqual(loaded_context.environment, original_context.environment)
        self.assertEqual(loaded_context.security_level, original_context.security_level)
    
    def test_context_enforcement(self):
        """Test für die Kontexterzwingung."""
        # Erstelle einen Kontext mit MEDIUM-Sicherheitslevel
        medium_context = self.context_verifier.create_context_token(
            module_name="TestModule",
            environment="testing",
            security_level="MEDIUM"
        )
        
        # Erstelle einen Kontext mit HIGH-Sicherheitslevel
        high_context = self.context_verifier.create_context_token(
            module_name="TestModule",
            environment="testing",
            security_level="HIGH"
        )
        
        # Überprüfe die Kontexterzwingung mit verschiedenen Anforderungen
        
        # MEDIUM-Kontext sollte MEDIUM-Anforderung erfüllen
        self.assertTrue(self.context_verifier.enforce_context(
            medium_context, required_level="MEDIUM", required_env="testing"))
        
        # MEDIUM-Kontext sollte HIGH-Anforderung NICHT erfüllen
        self.assertFalse(self.context_verifier.enforce_context(
            medium_context, required_level="HIGH", required_env="testing"))
        
        # HIGH-Kontext sollte MEDIUM-Anforderung erfüllen
        self.assertTrue(self.context_verifier.enforce_context(
            high_context, required_level="MEDIUM", required_env="testing"))
        
        # HIGH-Kontext sollte HIGH-Anforderung erfüllen
        self.assertTrue(self.context_verifier.enforce_context(
            high_context, required_level="HIGH", required_env="testing"))
        
        # Umgebungsanforderungen überprüfen
        self.assertFalse(self.context_verifier.enforce_context(
            high_context, required_level="HIGH", required_env="production"))


class TestContextIntegration(unittest.TestCase):
    """Integrationstests für das VOID-Kontext-Modul."""
    
    def setUp(self):
        """Test-Setup: Temporäres Verzeichnis für Kontexte erstellen."""
        self.temp_dir = tempfile.mkdtemp()
        self.context_verifier = ContextVerifier(self.temp_dir)
    
    def tearDown(self):
        """Test-Cleanup: Temporäres Verzeichnis entfernen."""
        shutil.rmtree(self.temp_dir)
    
    def test_module_boot_sequence(self):
        """Test für eine simulierte Modul-Boot-Sequenz mit Kontextverifizierung."""
        # Simuliere die Erzeugung eines Boot-Kontextes für ein Modul
        boot_context = self.context_verifier.create_context_token(
            module_name="CriticalModule",
            environment="production",
            security_level="ULTRA",
            attributes={
                "permitted_operations": ["tensor_calc", "neural_inference"],
                "max_resource_allocation": 80,
                "requires_network": False
            }
        )
        
        # Simuliere den Modulstart mit Kontextverifizierung
        def module_boot(context):
            """Simuliert den Boot-Prozess eines Moduls."""
            # Verifiziere den Boot-Kontext
            if not self.context_verifier.verify(context):
                return False, "Kontext konnte nicht verifiziert werden"
            
            # Erzwinge ULTRA-Sicherheitslevel für kritische Module
            if not self.context_verifier.enforce_context(context, required_level="ULTRA"):
                return False, "Sicherheitslevel nicht ausreichend"
            
            # Prüfe, ob das Modul Netzwerkzugriff benötigt
            requires_network = context.attributes.get("requires_network", True)
            if requires_network:
                # Hier würden wir Netzwerkberechtigungen einrichten
                pass
            
            # Gewähre nur erlaubte Operationen
            permitted_ops = context.attributes.get("permitted_operations", [])
            if not permitted_ops:
                return False, "Keine Operationen erlaubt"
            
            # Boot erfolgreich
            return True, f"Modul {context.module_name} erfolgreich gestartet mit Operationen: {permitted_ops}"
        
        # Führe den Boot-Prozess aus
        success, message = module_boot(boot_context)
        self.assertTrue(success)
        self.assertIn("CriticalModule", message)
        self.assertIn("tensor_calc", message)
        
        # Manipuliere den Kontext und versuche erneut zu booten
        boot_context.security_level = "MEDIUM"  # Manipuliere den Sicherheitslevel
        success, message = module_boot(boot_context)
        self.assertFalse(success)  # Boot sollte fehlschlagen


if __name__ == '__main__':
    unittest.main()
