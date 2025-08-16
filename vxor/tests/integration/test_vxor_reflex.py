#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spezifischer Test für die VX-REFLEX-Integration

Dieser Test fokussiert sich auf die Integration des VX-REFLEX-Moduls mit ECHO-PRIME
und testet die Reaktionsfunktionalität und Spontanverhaltensgenerierung.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import logging
from typing import Dict, List, Any

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_vxor_reflex")

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ZTM-Protokoll-Initialisierung (aktiviert für Integrationstests)
os.environ['MISO_ZTM_MODE'] = '1'
os.environ['MISO_ZTM_LOG_LEVEL'] = 'DEBUG'
# Stelle sicher, dass ZTM-Logs in das richtige Verzeichnis geschrieben werden
ztm_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(ztm_log_dir, exist_ok=True)
logger.info(f"Zero-Trust Monitoring (ZTM) aktiviert mit Log-Verzeichnis: {ztm_log_dir}")

# Importiere VXOR-Integrationsmodule
from miso.timeline.vxor_echo_integration import VXOREchoIntegration, get_vxor_echo_integration

# Importiere VXOR-Adapter
from vXor_Modules.vxor_integration import VXORAdapter, check_module_availability


class TestVXORReflex(unittest.TestCase):
    """
    Testklasse speziell für die VX-REFLEX-Integration.
    """
    
    def setUp(self):
        """
        Initialisiert die Testumgebung.
        """
        self.vxor_adapter = VXORAdapter()
        self.echo_integration = get_vxor_echo_integration()
        logger.info("Testumgebung für VX-REFLEX-Integration initialisiert")
    
    def test_reflex_availability(self):
        """
        Testet die Verfügbarkeit des VX-REFLEX-Moduls.
        """
        # Überprüfe, ob das VX-REFLEX-Modul im Manifest als implementiert markiert ist
        manifest = self.vxor_adapter._load_manifest()
        self.assertIn("VX-REFLEX", manifest["modules"]["implemented"])
        
        # Überprüfe die Verfügbarkeit des VX-REFLEX-Moduls
        vx_reflex_availability = check_module_availability("VX-REFLEX")
        self.assertIsInstance(vx_reflex_availability, dict)
        self.assertEqual(vx_reflex_availability["name"], "VX-REFLEX")
        self.assertEqual(vx_reflex_availability["status"], "vollständig implementiert")
        
        logger.info("VX-REFLEX-Verfügbarkeits-Test erfolgreich")
    
    def test_process_reflex_stimulus(self):
        """
        Testet die Verarbeitung von Stimuli mit dem VX-REFLEX-Modul.
        """
        # Teste mit einem externen Stimulus mit hoher Priorität
        high_priority_stimulus = {
            "type": "external", 
            "priority": "high", 
            "content": "Unerwartetes Ereignis"
        }
        high_priority_response = self.echo_integration.process_reflex_stimulus(high_priority_stimulus)
        self.assertIsNotNone(high_priority_response)
        self.assertIn("reaction", high_priority_response)
        self.assertIn("latency", high_priority_response)
        
        # Überprüfe, ob die Latenz bei hoher Priorität niedriger ist
        self.assertLessEqual(high_priority_response["latency"], 50)
        
        # Teste mit einem internen Stimulus mit niedriger Priorität
        low_priority_stimulus = {
            "type": "internal", 
            "priority": "low", 
            "content": "Routineüberprüfung"
        }
        low_priority_response = self.echo_integration.process_reflex_stimulus(low_priority_stimulus)
        self.assertIsNotNone(low_priority_response)
        self.assertIn("reaction", low_priority_response)
        self.assertIn("latency", low_priority_response)
        
        # Überprüfe, ob die Latenz bei niedriger Priorität höher ist
        self.assertGreaterEqual(low_priority_response["latency"], 100)
        
        logger.info("VX-REFLEX-Stimulus-Verarbeitungs-Test erfolgreich")
    
    def test_generate_spontaneous_behavior(self):
        """
        Testet die Generierung von Spontanverhalten mit dem VX-REFLEX-Modul.
        """
        # Generiere mehrere Spontanverhaltensweisen und überprüfe die Struktur
        behaviors = []
        for _ in range(5):
            behavior = self.echo_integration.generate_spontaneous_behavior()
            self.assertIsNotNone(behavior)
            self.assertIn("type", behavior)
            self.assertIn("priority", behavior)
            self.assertIn("content", behavior)
            behaviors.append(behavior)
        
        # Überprüfe, ob verschiedene Verhaltenstypen generiert wurden
        behavior_types = set(behavior["type"] for behavior in behaviors)
        self.assertGreater(len(behavior_types), 1, "Es sollten verschiedene Verhaltenstypen generiert werden")
        
        # Überprüfe, ob verschiedene Prioritäten generiert wurden
        priorities = set(behavior["priority"] for behavior in behaviors)
        self.assertGreater(len(priorities), 1, "Es sollten verschiedene Prioritäten generiert werden")
        
        logger.info("VX-REFLEX-Spontanverhaltens-Generierungs-Test erfolgreich")


if __name__ == "__main__":
    unittest.main()
