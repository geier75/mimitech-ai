#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - VXOR-Integrationstest

Dieses Skript testet die Integration der VXOR-Module mit MISO-Komponenten.
Es überprüft die Kommunikation zwischen den Modulen und die korrekte Funktionsweise
der integrierten Komponenten.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
from typing import Dict, List, Any
import unittest

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [VXOR-TEST] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.VXOR.Test")

# Füge das Stammverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiere die benötigten Module
try:
    # VXOR-Adapter
    from miso.vxor.vxor_adapter import VXORAdapter
    
    # M-LINGUA-Komponenten für Sprachintegration
    from miso.lang.mlingua.mlingua_interface import MLinguaInterface
    
    # PRISM-Engine für Simulationsintegration
    from miso.simulation.prism_engine import PRISMEngine
    
    # T-Mathematics für mathematische Integration
    from miso.tmathematics.t_mathematics_engine import TMathematicsEngine
    
    # ECHO-PRIME für temporale Integration
    from miso.echo.echo_prime import ECHO_PRIME
    
    logger.info("Module erfolgreich importiert")
except ImportError as e:
    logger.error(f"Fehler beim Importieren der Module: {e}")
    sys.exit(1)

class VXORIntegrationTest(unittest.TestCase):
    """Testet die Integration der VXOR-Module mit MISO-Komponenten"""
    
    def setUp(self):
        """Initialisiert die Testumgebung"""
        self.vxor_adapter = VXORAdapter()
        self.mlingua = MLinguaInterface()
        self.prism = PRISMEngine()
        self.tmath = TMathematicsEngine()
        self.echo = ECHO_PRIME()
        
        # Lade das VXOR-Manifest
        with open(os.path.join(os.path.dirname(__file__), '..', 'miso', 'vxor', 'vxor_manifest.json'), 'r') as f:
            self.manifest = json.load(f)
        
        logger.info("Testumgebung initialisiert")
    
    def test_vxor_manifest(self):
        """Testet, ob das VXOR-Manifest korrekt geladen wurde"""
        self.assertIsNotNone(self.manifest)
        self.assertIn("modules", self.manifest)
        logger.info("VXOR-Manifest erfolgreich geladen")
    
    def test_vxor_modules(self):
        """Testet, ob die VXOR-Module verfügbar sind"""
        modules = self.vxor_adapter.get_available_modules()
        self.assertIsNotNone(modules)
        
        # Überprüfe, ob die erforderlichen Module verfügbar sind
        required_modules = ["VX-PSI", "VX-SOMA", "VX-MEMEX"]
        for module in required_modules:
            self.assertIn(module, modules)
            logger.info(f"Modul {module} verfügbar")
    
    def test_mlingua_vxor_integration(self):
        """Testet die Integration von M-LINGUA mit VXOR-Modulen"""
        # Teste die Verarbeitung eines Textes mit M-LINGUA und VXOR
        result = self.mlingua.process_text("Öffne den Browser mit Suchparameter Wetter")
        self.assertTrue(result.success)
        self.assertEqual(result.detected_language, "de")
        self.assertEqual(result.intent, "EXECUTION")
        self.assertEqual(result.action, "open")
        self.assertEqual(result.target, "browser")
        self.assertEqual(result.parameters, {"search": "wetter"})
        
        # Überprüfe, ob VXOR-Befehle generiert wurden
        self.assertGreater(len(result.vxor_commands), 0)
        logger.info("M-LINGUA-VXOR-Integration erfolgreich getestet")
    
    def test_prism_vxor_integration(self):
        """Testet die Integration von PRISM-Engine mit VXOR-Modulen"""
        # Erstelle eine einfache Simulation mit PRISM und VXOR
        simulation_config = {
            "name": "test_simulation",
            "duration": 10,
            "vxor_modules": ["VX-PLANNER", "VX-REASON", "VX-CONTEXT"]
        }
        
        simulation = self.prism.create_simulation(simulation_config)
        self.assertIsNotNone(simulation)
        
        # Führe die Simulation aus
        result = self.prism.run_simulation(simulation)
        self.assertTrue(result.success)
        
        # Überprüfe, ob VXOR-Module verwendet wurden
        self.assertIn("vxor_modules_used", result.metadata)
        self.assertGreater(len(result.metadata["vxor_modules_used"]), 0)
        logger.info("PRISM-VXOR-Integration erfolgreich getestet")
    
    def test_tmath_vxor_integration(self):
        """Testet die Integration von T-Mathematics mit VXOR-Modulen"""
        # Erstelle eine mathematische Operation mit T-Mathematics und VXOR
        operation = {
            "type": "matrix_multiplication",
            "matrix_a": [[1, 2], [3, 4]],
            "matrix_b": [[5, 6], [7, 8]],
            "vxor_modules": ["VX-REASON", "VX-METACODE"]
        }
        
        result = self.tmath.execute_operation(operation)
        self.assertTrue(result.success)
        
        # Überprüfe das Ergebnis
        expected_result = [[19, 22], [43, 50]]
        self.assertEqual(result.result, expected_result)
        
        # Überprüfe, ob VXOR-Module verwendet wurden
        self.assertIn("vxor_modules_used", result.metadata)
        self.assertGreater(len(result.metadata["vxor_modules_used"]), 0)
        logger.info("T-Mathematics-VXOR-Integration erfolgreich getestet")
    
    def test_echo_vxor_integration(self):
        """Testet die Integration von ECHO-PRIME mit VXOR-Modulen"""
        # Erstelle einen temporalen Knoten mit ECHO-PRIME und VXOR
        node_config = {
            "name": "test_node",
            "timestamp": time.time(),
            "data": {"test": "data"},
            "vxor_modules": ["VX-MEMEX", "VX-CONTEXT", "VX-PLANNER"]
        }
        
        node = self.echo.create_time_node(node_config)
        self.assertIsNotNone(node)
        
        # Füge den Knoten zu einer Zeitlinie hinzu
        timeline = self.echo.create_timeline("test_timeline")
        self.echo.add_node_to_timeline(timeline, node)
        
        # Überprüfe, ob der Knoten korrekt hinzugefügt wurde
        nodes = self.echo.get_timeline_nodes(timeline)
        self.assertIn(node.id, [n.id for n in nodes])
        
        # Überprüfe, ob VXOR-Module verwendet wurden
        self.assertIn("vxor_modules_used", node.metadata)
        self.assertGreater(len(node.metadata["vxor_modules_used"]), 0)
        logger.info("ECHO-PRIME-VXOR-Integration erfolgreich getestet")
    
    def test_vxor_communication(self):
        """Testet die Kommunikation zwischen VXOR-Modulen"""
        # Teste die Kommunikation zwischen VX-PSI und VX-MEMEX
        message = {
            "source": "VX-PSI",
            "target": "VX-MEMEX",
            "action": "store",
            "data": {"memory": "test_memory", "content": "Dies ist ein Test"}
        }
        
        result = self.vxor_adapter.send_message(message)
        self.assertTrue(result.success)
        
        # Überprüfe, ob die Nachricht empfangen wurde
        response = self.vxor_adapter.get_module_status("VX-MEMEX")
        self.assertIn("last_message", response)
        self.assertEqual(response["last_message"]["source"], "VX-PSI")
        logger.info("VXOR-Kommunikation erfolgreich getestet")

def run_tests():
    """Führt alle Tests aus"""
    logger.info("=== VXOR-Integrationstests ===")
    logger.info(f"Startzeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(VXORIntegrationTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    success = result.wasSuccessful()
    logger.info(f"Endzeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Gesamtergebnis: {'✓ Alle Tests bestanden' if success else '✗ Einige Tests fehlgeschlagen'}")
    
    return success

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
