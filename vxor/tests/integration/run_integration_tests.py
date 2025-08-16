#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-Runner für die MISO-Integration-Tests.

Dieser Runner führt alle Tests für die Integrationskomponenten zwischen
Q-Logik und ECHO-PRIME aus und generiert einen Testbericht.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import unittest
import sys
import os
import time
import logging
from datetime import datetime

# Pfad zum MISO-Paket hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.tests.integration")

def run_tests():
    """Führt alle Integrationstests aus und erstellt einen Bericht."""
    # Startzeit erfassen
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"Starte Integrationstests um {start_datetime}")
    
    # Erstelle eine TestSuite, die alle Tests im aktuellen Verzeichnis enthält
    loader = unittest.TestLoader()
    test_suite = loader.discover(os.path.dirname(__file__), pattern="test_*.py")
    
    # Testbericht-Verzeichnis erstellen, falls es nicht existiert
    report_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Testbericht-Dateiname mit Datum und Uhrzeit generieren
    report_filename = f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = os.path.join(report_dir, report_filename)
    
    # Tests ausführen und Ergebnisse in Datei und Konsole ausgeben
    with open(report_path, "w") as report_file:
        # Header für den Testbericht
        header = f"""
===============================================================
MISO INTEGRATION TEST REPORT
===============================================================
Datum und Uhrzeit: {start_datetime}
Testumgebung: Python {sys.version}
===============================================================

"""
        report_file.write(header)
        print(header)
        
        # TestRunner konfigurieren und Tests ausführen
        runner = unittest.TextTestRunner(verbosity=2, stream=report_file)
        result = runner.run(test_suite)
        
        # Testdauer berechnen
        end_time = time.time()
        duration = end_time - start_time
        
        # Zusammenfassung für den Testbericht
        summary = f"""
===============================================================
ZUSAMMENFASSUNG
===============================================================
Tests ausgeführt: {result.testsRun}
Fehler: {len(result.errors)}
Fehlschläge: {len(result.failures)}
Skips: {len(getattr(result, 'skipped', []))}
Dauer: {duration:.2f} Sekunden
===============================================================
"""
        report_file.write(summary)
        print(summary)
        
        # Details zu Fehlern und Fehlschlägen
        if result.errors:
            errors_section = "\nFEHLER\n===============================================================\n"
            report_file.write(errors_section)
            print(errors_section)
            for test, error in result.errors:
                error_detail = f"{test}: {error}\n"
                report_file.write(error_detail)
                print(error_detail)
        
        if result.failures:
            failures_section = "\nFEHLSCHLÄGE\n===============================================================\n"
            report_file.write(failures_section)
            print(failures_section)
            for test, failure in result.failures:
                failure_detail = f"{test}: {failure}\n"
                report_file.write(failure_detail)
                print(failure_detail)
    
    logger.info(f"Integrationstests abgeschlossen. Dauer: {duration:.2f} Sekunden")
    logger.info(f"Testbericht gespeichert in: {report_path}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
