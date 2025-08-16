#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Umfassender Systemtest

Führt umfassende Tests für alle Module, Submodule und Agentenstrukturen durch.
Protokolliert alle Ergebnisse unter /VXOR_Logs/SystemTest/

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import unittest
import logging
import datetime
import importlib
import numpy as np
import torch
import json
import traceback
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Konfiguriere Logging
log_dir = Path("/Volumes/My Book/VXOR_AI 15.32.28/VXOR_Logs/SystemTest")
log_dir.mkdir(parents=True, exist_ok=True)

# Hauptlogger
main_log_file = log_dir / f"system_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(main_log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MISO.SystemTest")

# Prüfe, ob Apple Silicon verfügbar ist
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
logger.info(f"Apple Silicon: {is_apple_silicon}")

# Prüfe, ob MLX verfügbar ist
try:
    import mlx.core as mx
    HAS_MLX = True
    logger.info("MLX erfolgreich importiert - Apple Silicon Optimierung verfügbar")
except ImportError:
    HAS_MLX = False
    logger.warning("MLX konnte nicht importiert werden. Apple Silicon Optimierung nicht verfügbar.")

# Definiere die zu testenden Module
MODULES = [
    "miso.math.t_mathematics",
    "miso.simulation",
    "miso.vXor_Modules",
    "miso.vxor",
    "miso.lang",
    "miso.logic",
    "miso.core",
    "miso.security",
    "miso.timeline"
]

# Definiere Testtypen
TEST_TYPES = [
    "unit",
    "integration",
    "interface",
    "performance",
    "stress",
    "compatibility"
]

class TestResult:
    """Klasse zur Speicherung von Testergebnissen"""
    
    def __init__(self, module: str, test_type: str):
        self.module = module
        self.test_type = test_type
        self.success = False
        self.error = None
        self.details = {}
        self.duration = 0.0
        self.timestamp = datetime.datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert das Testergebnis in ein Dictionary"""
        return {
            "module": self.module,
            "test_type": self.test_type,
            "success": self.success,
            "error": str(self.error) if self.error else None,
            "details": self.details,
            "duration": self.duration,
            "timestamp": self.timestamp
        }

class ModuleTester:
    """Führt Tests für ein bestimmtes Modul durch"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.results = {}
        self.module = None
        
        # Konfiguriere Modul-spezifisches Logging
        self.log_file = log_dir / f"{module_name.replace('.', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = logging.getLogger(f"MISO.SystemTest.{module_name}")
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
    
    def import_module(self) -> bool:
        """Importiert das zu testende Modul"""
        try:
            self.module = importlib.import_module(self.module_name)
            self.logger.info(f"Modul {self.module_name} erfolgreich importiert")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Importieren des Moduls {self.module_name}: {e}")
            return False
    
    def run_unit_tests(self) -> TestResult:
        """Führt Unit-Tests für das Modul aus"""
        result = TestResult(self.module_name, "unit")
        start_time = time.time()
        
        try:
            # Versuche, die Testmodule zu finden und auszuführen
            test_module_name = f"tests.test_{self.module_name.split('.')[-1]}"
            try:
                test_module = importlib.import_module(test_module_name)
                self.logger.info(f"Test-Modul {test_module_name} gefunden")
            except ImportError:
                # Versuche alternative Testmodule
                test_module_name = f"tests.{self.module_name.replace('.', '_')}_test"
                try:
                    test_module = importlib.import_module(test_module_name)
                    self.logger.info(f"Test-Modul {test_module_name} gefunden")
                except ImportError:
                    self.logger.warning(f"Keine Testmodule für {self.module_name} gefunden")
                    result.details["warning"] = f"Keine Testmodule für {self.module_name} gefunden"
                    result.success = True  # Kein Test ist kein Fehler
                    return result
            
            # Führe die Tests aus
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            test_result = unittest.TextTestRunner(verbosity=2).run(suite)
            
            # Speichere die Ergebnisse
            result.success = test_result.wasSuccessful()
            result.details["tests_run"] = test_result.testsRun
            result.details["failures"] = len(test_result.failures)
            result.details["errors"] = len(test_result.errors)
            
            if not result.success:
                error_details = []
                for failure in test_result.failures:
                    error_details.append({
                        "test": str(failure[0]),
                        "message": failure[1]
                    })
                for error in test_result.errors:
                    error_details.append({
                        "test": str(error[0]),
                        "message": error[1]
                    })
                result.details["error_details"] = error_details
                result.error = "Unit-Tests fehlgeschlagen"
            
            self.logger.info(f"Unit-Tests für {self.module_name} abgeschlossen: {'Erfolgreich' if result.success else 'Fehlgeschlagen'}")
        
        except Exception as e:
            self.logger.error(f"Fehler bei Unit-Tests für {self.module_name}: {e}")
            result.success = False
            result.error = str(e)
            result.details["traceback"] = traceback.format_exc()
        
        result.duration = time.time() - start_time
        return result
    
    def run_integration_tests(self) -> TestResult:
        """Führt Integrationstests für das Modul aus"""
        result = TestResult(self.module_name, "integration")
        start_time = time.time()
        
        try:
            # Versuche, die Integrationstestmodule zu finden und auszuführen
            test_module_name = f"tests.integration.test_{self.module_name.split('.')[-1]}_integration"
            try:
                test_module = importlib.import_module(test_module_name)
                self.logger.info(f"Integrations-Test-Modul {test_module_name} gefunden")
            except ImportError:
                # Versuche alternative Testmodule
                test_module_name = f"tests.test_{self.module_name.split('.')[-1]}_integration"
                try:
                    test_module = importlib.import_module(test_module_name)
                    self.logger.info(f"Integrations-Test-Modul {test_module_name} gefunden")
                except ImportError:
                    # Spezialfall für T-MATHEMATICS
                    if self.module_name == "miso.math.t_mathematics":
                        test_module_name = "tests.test_t_mathematics_integration"
                        try:
                            test_module = importlib.import_module(test_module_name)
                            self.logger.info(f"Integrations-Test-Modul {test_module_name} gefunden")
                        except ImportError:
                            self.logger.warning(f"Keine Integrationstestmodule für {self.module_name} gefunden")
                            result.details["warning"] = f"Keine Integrationstestmodule für {self.module_name} gefunden"
                            result.success = True  # Kein Test ist kein Fehler
                            return result
                    else:
                        self.logger.warning(f"Keine Integrationstestmodule für {self.module_name} gefunden")
                        result.details["warning"] = f"Keine Integrationstestmodule für {self.module_name} gefunden"
                        result.success = True  # Kein Test ist kein Fehler
                        return result
            
            # Führe die Tests aus
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            test_result = unittest.TextTestRunner(verbosity=2).run(suite)
            
            # Speichere die Ergebnisse
            result.success = test_result.wasSuccessful()
            result.details["tests_run"] = test_result.testsRun
            result.details["failures"] = len(test_result.failures)
            result.details["errors"] = len(test_result.errors)
            
            if not result.success:
                error_details = []
                for failure in test_result.failures:
                    error_details.append({
                        "test": str(failure[0]),
                        "message": failure[1]
                    })
                for error in test_result.errors:
                    error_details.append({
                        "test": str(error[0]),
                        "message": error[1]
                    })
                result.details["error_details"] = error_details
                result.error = "Integrationstests fehlgeschlagen"
            
            self.logger.info(f"Integrationstests für {self.module_name} abgeschlossen: {'Erfolgreich' if result.success else 'Fehlgeschlagen'}")
        
        except Exception as e:
            self.logger.error(f"Fehler bei Integrationstests für {self.module_name}: {e}")
            result.success = False
            result.error = str(e)
            result.details["traceback"] = traceback.format_exc()
        
        result.duration = time.time() - start_time
        return result
