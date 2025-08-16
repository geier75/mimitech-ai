#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Haupttestskript

Führt alle Tests für Module, Submodule und Agentenstrukturen aus.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import importlib
import logging
from pathlib import Path

# Konfiguriere Logging
log_dir = Path("/Volumes/My Book/VXOR_AI 15.32.28/VXOR_Logs/SystemTest")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "main_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MISO.MainTest")

# Importiere die Testmodule
try:
    import run_comprehensive_tests
    import run_comprehensive_tests_part2
    import run_comprehensive_tests_part3
    logger.info("Testmodule erfolgreich importiert")
except ImportError as e:
    logger.error(f"Fehler beim Importieren der Testmodule: {e}")
    sys.exit(1)

if __name__ == "__main__":
    logger.info("Starte umfassenden Systemtest")
    
    # Führe die Hauptfunktion aus run_comprehensive_tests_part3 aus
    run_comprehensive_tests_part3.main()
    
    logger.info("Umfassender Systemtest abgeschlossen")
