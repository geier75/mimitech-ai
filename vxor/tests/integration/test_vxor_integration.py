#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für die VXOR-Integration mit MISO-Komponenten

Dieser Test überprüft die korrekte Integration der VXOR-Module mit den
MISO-Komponenten (PRISM-Engine, T-Mathematics Engine, ECHO-PRIME).

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import logging
from typing import Dict, List, Any

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_vxor_integration")

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
from miso.simulation.vxor_prism_integration import VXORPrismIntegration, get_vxor_prism_integration
from miso.math.t_mathematics.vxor_math_integration import VXORMathIntegration, get_vxor_math_integration
from miso.timeline.vxor_echo_integration import VXOREchoIntegration, get_vxor_echo_integration

# Importiere VXOR-Adapter
from vXor_Modules.vxor_integration import VXORAdapter, check_module_availability


class TestVXORIntegration(unittest.TestCase):
    """
    Testklasse für die VXOR-Integration mit MISO-Komponenten.
    """
    
    def setUp(self):
        """
        Initialisiert die Testumgebung.
        """
        self.vxor_adapter = VXORAdapter()
        self.prism_integration = get_vxor_prism_integration()
        self.math_integration = get_vxor_math_integration()
        self.echo_integration = get_vxor_echo_integration()
        logger.info("Testumgebung für VXOR-Integration initialisiert")
    
    def test_vxor_adapter(self):
        """
        Testet den VXOR-Adapter.
        """
        # Überprüfe, ob der Adapter korrekt initialisiert wurde
        self.assertIsNotNone(self.vxor_adapter)
        
        # Überprüfe, ob das Manifest geladen wurde
        manifest = self.vxor_adapter._load_manifest()
        self.assertIsNotNone(manifest)
        self.assertIn("modules", manifest)
        self.assertIn("implemented", manifest["modules"])
        self.assertIn("planned", manifest["modules"])
        
        # Überprüfe die implementierten Module
        implemented_modules = manifest["modules"]["implemented"]
        self.assertIn("VX-PSI", implemented_modules)
        self.assertIn("VX-SOMA", implemented_modules)
        self.assertIn("VX-MEMEX", implemented_modules)
        self.assertIn("VX-SELFWRITER", implemented_modules)
        self.assertIn("VX-REFLEX", implemented_modules)
        
        # Überprüfe die geplanten Module
        planned_modules = manifest["modules"]["planned"]
        self.assertIn("VX-PLANNER", planned_modules)
        self.assertIn("VX-REASON", planned_modules)
        self.assertIn("VX-CONTEXT", planned_modules)
        
        logger.info("VXOR-Adapter-Test erfolgreich")
    
    def test_prism_integration(self):
        """
        Testet die VXOR-PRISM-Integration.
        """
        # Überprüfe, ob die Integration korrekt initialisiert wurde
        self.assertIsNotNone(self.prism_integration)
        
        # Überprüfe die kompatiblen Module
        compatible_modules = self.prism_integration.compatible_modules
        self.assertIsInstance(compatible_modules, list)
        self.assertIn("VX-PLANNER", compatible_modules)
        self.assertIn("VX-REASON", compatible_modules)
        self.assertIn("VX-CONTEXT", compatible_modules)
        
        # Überprüfe die verfügbaren Module
        available_modules = self.prism_integration.get_available_modules()
        self.assertIsInstance(available_modules, dict)
        
        # Teste die Planner-Funktion (sollte ein leeres Dictionary zurückgeben, da das Modul nicht implementiert ist)
        planner_result = self.prism_integration.use_planner("test_timeline", 5)
        self.assertIsInstance(planner_result, dict)
        
        logger.info("VXOR-PRISM-Integration-Test erfolgreich")
    
    def test_math_integration(self):
        """
        Testet die VXOR-T-Mathematics-Integration.
        """
        # Überprüfe, ob die Integration korrekt initialisiert wurde
        self.assertIsNotNone(self.math_integration)
        
        # Überprüfe die kompatiblen Module
        compatible_modules = self.math_integration.compatible_modules
        self.assertIsInstance(compatible_modules, list)
        self.assertIn("VX-REASON", compatible_modules)
        self.assertIn("VX-METACODE", compatible_modules)
        
        # Überprüfe die verfügbaren Module
        available_modules = self.math_integration.get_available_modules()
        self.assertIsInstance(available_modules, dict)
        
        # Teste die Reason-Funktion (sollte ein leeres Dictionary zurückgeben, da das Modul nicht implementiert ist)
        reason_result = self.math_integration.use_reason({"test": "data"})
        self.assertIsInstance(reason_result, dict)
        
        # Teste die Metacode-Funktion (sollte ein leeres Dictionary zurückgeben, da das Modul nicht implementiert ist)
        metacode_result = self.math_integration.use_metacode("test_code", 2)
        self.assertIsInstance(metacode_result, dict)
        
        logger.info("VXOR-T-Mathematics-Integration-Test erfolgreich")
    
    def test_echo_integration(self):
        """
        Testet die VXOR-ECHO-PRIME-Integration.
        """
        # Überprüfe, ob die Integration korrekt initialisiert wurde
        self.assertIsNotNone(self.echo_integration)
        
        # Überprüfe die kompatiblen Module
        compatible_modules = self.echo_integration.compatible_modules
        self.assertIsInstance(compatible_modules, list)
        self.assertIn("VX-MEMEX", compatible_modules)
        self.assertIn("VX-CONTEXT", compatible_modules)
        self.assertIn("VX-PLANNER", compatible_modules)
        
        # Überprüfe die verfügbaren Module
        available_modules = self.echo_integration.get_available_modules()
        self.assertIsInstance(available_modules, dict)
        
        # Teste die Memex-Funktion (sollte ein leeres Dictionary zurückgeben, da das Modul nicht implementiert ist)
        memex_result = self.echo_integration.use_memex("episodic", {"test": "data"})
        self.assertIsInstance(memex_result, dict)
        
        # Teste die Context-Funktion (sollte ein leeres Dictionary zurückgeben, da das Modul nicht implementiert ist)
        context_result = self.echo_integration.use_context("test_timeline", {"test": "data"})
        self.assertIsInstance(context_result, dict)
        
        # Teste die Planner-Funktion (sollte ein leeres Dictionary zurückgeben, da das Modul nicht implementiert ist)
        planner_result = self.echo_integration.use_planner("test_timeline", 5)
        self.assertIsInstance(planner_result, dict)
        
        logger.info("VXOR-ECHO-PRIME-Integration-Test erfolgreich")
    
    def test_module_availability(self):
        """
        Testet die Verfügbarkeit der VXOR-Module.
        """
        # Überprüfe die Verfügbarkeit der implementierten Module
        vx_psi_availability = check_module_availability("VX-PSI")
        self.assertIsInstance(vx_psi_availability, dict)
        self.assertEqual(vx_psi_availability["name"], "VX-PSI")
        
        vx_soma_availability = check_module_availability("VX-SOMA")
        self.assertIsInstance(vx_soma_availability, dict)
        self.assertEqual(vx_soma_availability["name"], "VX-SOMA")
        
        vx_memex_availability = check_module_availability("VX-MEMEX")
        self.assertIsInstance(vx_memex_availability, dict)
        self.assertEqual(vx_memex_availability["name"], "VX-MEMEX")
        
        vx_reflex_availability = check_module_availability("VX-REFLEX")
        self.assertIsInstance(vx_reflex_availability, dict)
        self.assertEqual(vx_reflex_availability["name"], "VX-REFLEX")
        
        # Überprüfe die Verfügbarkeit der geplanten Module
        vx_planner_availability = check_module_availability("VX-PLANNER")
        self.assertIsInstance(vx_planner_availability, dict)
        self.assertEqual(vx_planner_availability["name"], "VX-PLANNER")
        
        logger.info("VXOR-Modul-Verfügbarkeits-Test erfolgreich")
    
    def test_reflex_integration(self):
        """
        Testet die Integration des VX-REFLEX-Moduls.
        """
        # Überprüfe, ob das VX-REFLEX-Modul verfügbar ist
        self.assertTrue(self.vxor_adapter.is_module_available("VX-REFLEX"))
        
        # Überprüfe die Integration mit ECHO-PRIME
        if "VX-REFLEX" in self.echo_integration.available_modules:
            self.assertTrue(self.echo_integration.available_modules["VX-REFLEX"])
            # Teste die Reaktionsfunktionalität
            test_stimulus = {"type": "external", "priority": "high", "content": "Test-Stimulus"}
            response = self.echo_integration.process_reflex_stimulus(test_stimulus)
            self.assertIsNotNone(response)
            self.assertIn("reaction", response)
            self.assertIn("latency", response)
            
            # Teste die Spontanverhaltensfunktionalität
            behavior = self.echo_integration.generate_spontaneous_behavior()
            self.assertIsNotNone(behavior)
            self.assertIn("type", behavior)
            self.assertIn("priority", behavior)
            
            logger.info("VX-REFLEX-Integration mit ECHO-PRIME erfolgreich getestet")


if __name__ == "__main__":
    unittest.main()
