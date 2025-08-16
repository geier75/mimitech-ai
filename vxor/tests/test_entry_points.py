#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vXor Modul Entry-Point Tests

Testet die standardisierten Entry-Points der vXor-Module gemäß der konsolidierten Konvention.
Führt Tests für init(), boot(), configure() und andere standardisierte Schnittstellen durch.
"""

import sys
import os
import unittest
import logging
import importlib
from typing import Dict, List, Any, Set, Optional

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiere die Testhelfer
from tests.test_helpers import ModuleTestHelper, module_test_context

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("vxor.tests.entry_points")

# Liste der zu testenden Module
TEST_MODULES = [
    'security_layer',
    'simulation_engine',
    # Hier können weitere Module hinzugefügt werden
]

class EntryPointTestCase(unittest.TestCase):
    """Testet standardisierte Entry-Points in vXor-Modulen."""
    
    def test_init_entry_point(self):
        """Testet, ob die Module den init() Entry-Point korrekt implementieren."""
        for module_name in TEST_MODULES:
            with self.subTest(module=module_name):
                with module_test_context(module_name) as (helper, module):
                    self.assertIsNotNone(module, f"Modul {module_name} konnte nicht geladen werden")
                    self.assertTrue(hasattr(module, 'init'), f"init() nicht in {module_name} gefunden")
                    self.assertTrue(callable(getattr(module, 'init')), f"init in {module_name} ist keine Funktion")
                    
                    # Führe init() aus und prüfe das Ergebnis
                    result = module.init()
                    self.assertIsNotNone(result, f"init() in {module_name} gibt None zurück")
                    
                    # Prüfe Idempotenz
                    result2 = module.init()
                    self.assertEqual(id(result), id(result2), 
                                     f"init() in {module_name} ist nicht idempotent")
    
    def test_boot_entry_point(self):
        """Testet, ob die Module den boot() Entry-Point korrekt implementieren."""
        for module_name in TEST_MODULES:
            with self.subTest(module=module_name):
                with module_test_context(module_name) as (helper, module):
                    self.assertIsNotNone(module, f"Modul {module_name} konnte nicht geladen werden")
                    self.assertTrue(hasattr(module, 'boot'), f"boot() nicht in {module_name} gefunden")
                    self.assertTrue(callable(getattr(module, 'boot')), f"boot in {module_name} ist keine Funktion")
                    
                    # Führe boot() aus und prüfe das Ergebnis
                    result = module.boot()
                    self.assertIsNotNone(result, f"boot() in {module_name} gibt None zurück")
                    
                    # Prüfe, ob boot() das gleiche Objekt wie init() zurückgibt
                    init_result = module.init()
                    self.assertEqual(id(result), id(init_result), 
                                     f"boot() in {module_name} gibt nicht das gleiche Objekt wie init() zurück")
    
    def test_configure_entry_point(self):
        """Testet, ob die Module den configure() Entry-Point korrekt implementieren."""
        for module_name in TEST_MODULES:
            with self.subTest(module=module_name):
                with module_test_context(module_name) as (helper, module):
                    if not hasattr(module, 'configure'):
                        logger.warning(f"configure() nicht in {module_name} gefunden - optional")
                        continue
                    
                    self.assertTrue(callable(getattr(module, 'configure')), 
                                    f"configure in {module_name} ist keine Funktion")
                    
                    # Führe configure() mit leerer Konfiguration aus
                    result = module.configure({})
                    self.assertIsNotNone(result, f"configure({{}}) in {module_name} gibt None zurück")
                    
                    # Prüfe, ob configure() das gleiche Objekt wie init() zurückgibt
                    init_result = module.init()
                    self.assertEqual(id(result), id(init_result), 
                                    f"configure() in {module_name} gibt nicht das gleiche Objekt wie init() zurück")
    
    def test_all_entry_points(self):
        """Testet alle standardisierten Entry-Points in den Modulen."""
        # Liste aller standardisierten Entry-Points
        standard_entry_points = ['init', 'boot', 'configure', 'start', 'setup', 'activate']
        
        results = {}
        
        for module_name in TEST_MODULES:
            module_results = {}
            
            with module_test_context(module_name) as (helper, module):
                if not module:
                    logger.error(f"Modul {module_name} konnte nicht geladen werden")
                    for ep in standard_entry_points:
                        module_results[ep] = False
                    results[module_name] = module_results
                    continue
                
                for ep in standard_entry_points:
                    if hasattr(module, ep) and callable(getattr(module, ep)):
                        try:
                            result = getattr(module, ep)()
                            module_results[ep] = result is not None
                        except Exception as e:
                            logger.error(f"Fehler beim Testen von {ep}() in {module_name}: {e}")
                            module_results[ep] = False
                    else:
                        module_results[ep] = False
            
            results[module_name] = module_results
        
        # Ausgabe der Ergebnisse
        logger.info("===== ENTRY-POINT TEST ERGEBNISSE =====")
        for module_name, module_results in results.items():
            logger.info(f"Modul: {module_name}")
            implemented = sum(1 for v in module_results.values() if v)
            total = len(module_results)
            percentage = (implemented / total) * 100 if total > 0 else 0
            
            for ep, result in module_results.items():
                status = "✅" if result else "❌"
                logger.info(f"  {status} {ep}()")
            
            logger.info(f"  Implementiert: {implemented}/{total} ({percentage:.1f}%)")
            logger.info("-" * 40)
        
        # Prüfe Gesamtabdeckung
        all_results = [result for module_results in results.values() for result in module_results.values()]
        implemented = sum(1 for r in all_results if r)
        total = len(all_results)
        percentage = (implemented / total) * 100 if total > 0 else 0
        
        logger.info(f"GESAMTABDECKUNG: {implemented}/{total} ({percentage:.1f}%)")
        
        # Prüfe mindestens 50% Abdeckung
        self.assertGreaterEqual(percentage, 50.0, "Gesamtabdeckung der Entry-Points unter 50%")

if __name__ == "__main__":
    unittest.main()
