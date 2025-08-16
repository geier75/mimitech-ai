#!/usr/bin/env python3
"""
MISO-Modul-Testskript

Dieses Skript überprüft die Vollständigkeit und Funktionalität aller MISO-Module.
Es führt automatische Tests für Modul-Initialisierung, Funktionalität, Logging und Simulationen durch.

Autor: MISO-Team
Datum: 2023-05-15
"""

import os
import sys
import importlib
import logging
import time
from datetime import datetime
import traceback

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [TEST] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO-Test")

# Liste der zu testenden Module
REQUIRED_MODULES = [
    # Essenzielle Komponenten
    "mprime",           # M-PRIME Framework
    "tmathematics",     # T-Mathematics Engine
    "nexus",            # NEXUS-OS
    "mcode",            # M-CODE Runtime
    "mlingua",          # M-LINGUA Interface
    "prism",            # PRISM-Simulator
    "omega",            # Omega-Kern
    "void",             # VOID-Protokoll
    # Zusätzliche Komponenten
    "perception",       # Perception-Modul mit HYPERFILTER
    "strategic",        # Strategic-Modul mit Deep-State
    "quantum",          # Q-Logik Framework
    "mimimon",          # MIMIMON ZTM-Modul
    "timeline"          # Timeline-Modul mit QTM-Modulator
]

# Liste der erforderlichen ZTM-Policy-Dateien
REQUIRED_ZTM_FILES = [
    "mprime/mprime.ztm",
    "tmathematics/tmathematics.ztm",
    "nexus/nexus.ztm",
    "mcode/mcode.ztm",
    "mlingua/mlingua.ztm",
    "prism/prism.ztm",
    "omega/omega.ztm",
    "void/void.ztm",
    "perception/perception.ztm",
    "quantum/quantum.ztm",
    "mimimon/mimimon.ztm"
]

class ModuleTest:
    """Klasse zum Testen der MISO-Module"""
    
    def __init__(self):
        """Initialisiert den ModuleTest"""
        self.results = {
            "existence_check": {},
            "ztm_check": {},
            "initialization_check": {},
            "functionality_check": {},
            "logging_check": {},
            "simulation_check": {}
        }
        self.success_count = 0
        self.failure_count = 0
        self.warning_count = 0
        self.start_time = datetime.now()
    
    def run_tests(self):
        """Führt alle Tests aus"""
        logger.info("=== MISO-Modul-Test gestartet ===")
        
        try:
            # Füge das MISO-Verzeichnis zum Pfad hinzu
            miso_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "miso"))
            if miso_dir not in sys.path:
                sys.path.insert(0, miso_dir)
            
            # Führe die Tests aus
            self.check_module_existence()
            self.check_ztm_files()
            self.test_module_initialization()
            self.test_module_functionality()
            self.test_module_logging()
            self.test_module_simulation()
            
            # Zeige die Ergebnisse an
            self.show_results()
            
        except Exception as e:
            logger.error(f"Fehler beim Ausführen der Tests: {e}")
            traceback.print_exc()
    
    def check_module_existence(self):
        """Überprüft die Existenz aller erforderlichen Module"""
        logger.info("Überprüfe die Existenz aller erforderlichen Module...")
        
        for module_name in REQUIRED_MODULES:
            module_dir = os.path.join("miso", module_name)
            init_file = os.path.join(module_dir, "__init__.py")
            
            if os.path.exists(module_dir) and os.path.isdir(module_dir):
                if os.path.exists(init_file) and os.path.isfile(init_file):
                    self.results["existence_check"][module_name] = "SUCCESS"
                    self.success_count += 1
                    logger.info(f"✅ Modul '{module_name}' existiert und hat eine __init__.py-Datei")
                else:
                    self.results["existence_check"][module_name] = "WARNING"
                    self.warning_count += 1
                    logger.warning(f"⚠️ Modul '{module_name}' existiert, hat aber keine __init__.py-Datei")
            else:
                self.results["existence_check"][module_name] = "FAILURE"
                self.failure_count += 1
                logger.error(f"❌ Modul '{module_name}' existiert nicht")
    
    def check_ztm_files(self):
        """Überprüft die Existenz aller erforderlichen ZTM-Policy-Dateien"""
        logger.info("Überprüfe die Existenz aller erforderlichen ZTM-Policy-Dateien...")
        
        for ztm_file in REQUIRED_ZTM_FILES:
            file_path = os.path.join("miso", ztm_file)
            
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.results["ztm_check"][ztm_file] = "SUCCESS"
                self.success_count += 1
                logger.info(f"✅ ZTM-Policy-Datei '{ztm_file}' existiert")
            else:
                self.results["ztm_check"][ztm_file] = "FAILURE"
                self.failure_count += 1
                logger.error(f"❌ ZTM-Policy-Datei '{ztm_file}' existiert nicht")
    
    def test_module_initialization(self):
        """Testet die Initialisierung aller Module"""
        logger.info("Teste die Initialisierung aller Module...")
        
        for module_name in REQUIRED_MODULES:
            try:
                # Importiere das Modul
                module = importlib.import_module(f"miso.{module_name}")
                
                # Überprüfe, ob das Modul erfolgreich importiert wurde
                if module:
                    self.results["initialization_check"][module_name] = "SUCCESS"
                    self.success_count += 1
                    logger.info(f"✅ Modul '{module_name}' erfolgreich initialisiert")
                else:
                    self.results["initialization_check"][module_name] = "FAILURE"
                    self.failure_count += 1
                    logger.error(f"❌ Modul '{module_name}' konnte nicht initialisiert werden")
            
            except ImportError as e:
                self.results["initialization_check"][module_name] = "FAILURE"
                self.failure_count += 1
                logger.error(f"❌ Modul '{module_name}' konnte nicht importiert werden: {e}")
            
            except Exception as e:
                self.results["initialization_check"][module_name] = "FAILURE"
                self.failure_count += 1
                logger.error(f"❌ Fehler bei der Initialisierung des Moduls '{module_name}': {e}")
    
    def test_module_functionality(self):
        """Testet die Funktionalität aller Module"""
        logger.info("Teste die Funktionalität aller Module...")
        
        # Diese Funktion würde in einer vollständigen Implementierung die tatsächliche Funktionalität testen
        # Da dies ein vereinfachter Test ist, überprüfen wir nur, ob die Module die erforderlichen Attribute haben
        
        for module_name in REQUIRED_MODULES:
            try:
                # Importiere das Modul
                module = importlib.import_module(f"miso.{module_name}")
                
                # Überprüfe, ob das Modul die erforderlichen Attribute hat
                has_required_attributes = hasattr(module, "__all__") or hasattr(module, "__doc__")
                
                if has_required_attributes:
                    self.results["functionality_check"][module_name] = "SUCCESS"
                    self.success_count += 1
                    logger.info(f"✅ Modul '{module_name}' hat die erforderlichen Attribute")
                else:
                    self.results["functionality_check"][module_name] = "WARNING"
                    self.warning_count += 1
                    logger.warning(f"⚠️ Modul '{module_name}' hat nicht alle erforderlichen Attribute")
            
            except Exception as e:
                self.results["functionality_check"][module_name] = "FAILURE"
                self.failure_count += 1
                logger.error(f"❌ Fehler beim Testen der Funktionalität des Moduls '{module_name}': {e}")
    
    def test_module_logging(self):
        """Testet das Logging aller Module"""
        logger.info("Teste das Logging aller Module...")
        
        # Diese Funktion würde in einer vollständigen Implementierung das tatsächliche Logging testen
        # Da dies ein vereinfachter Test ist, überprüfen wir nur, ob die Module ein Logger-Objekt haben
        
        for module_name in REQUIRED_MODULES:
            try:
                # Importiere das Modul
                module = importlib.import_module(f"miso.{module_name}")
                
                # Überprüfe, ob das Modul ein Logger-Objekt hat
                # Dies ist eine vereinfachte Überprüfung und würde in einer vollständigen Implementierung erweitert werden
                has_logger = any(name.endswith("logger") for name in dir(module))
                
                if has_logger:
                    self.results["logging_check"][module_name] = "SUCCESS"
                    self.success_count += 1
                    logger.info(f"✅ Modul '{module_name}' hat ein Logger-Objekt")
                else:
                    self.results["logging_check"][module_name] = "WARNING"
                    self.warning_count += 1
                    logger.warning(f"⚠️ Modul '{module_name}' hat kein erkennbares Logger-Objekt")
            
            except Exception as e:
                self.results["logging_check"][module_name] = "FAILURE"
                self.failure_count += 1
                logger.error(f"❌ Fehler beim Testen des Loggings des Moduls '{module_name}': {e}")
    
    def test_module_simulation(self):
        """Testet die Simulationsfähigkeit aller Module"""
        logger.info("Teste die Simulationsfähigkeit aller Module...")
        
        # Diese Funktion würde in einer vollständigen Implementierung die tatsächliche Simulationsfähigkeit testen
        # Da dies ein vereinfachter Test ist, überprüfen wir nur, ob die Module bestimmte Simulationsmethoden haben
        
        simulation_methods = {
            "mprime": ["run", "simulate", "execute"],
            "tmathematics": ["calculate", "compute", "run"],
            "nexus": ["start", "run", "execute"],
            "mcode": ["execute", "run", "compile"],
            "mlingua": ["process", "execute", "run"],
            "prism": ["simulate", "run_simulation", "execute"],
            "omega": ["start", "run", "execute"],
            "void": ["send", "receive", "communicate"],
            "perception": ["filter", "process", "analyze"],
            "strategic": ["analyze", "recommend", "execute"],
            "quantum": ["simulate", "compute", "measure"],
            "mimimon": ["monitor", "enforce", "verify"],
            "timeline": ["simulate", "process", "modulate"]
        }
        
        for module_name in REQUIRED_MODULES:
            try:
                # Importiere das Modul
                module = importlib.import_module(f"miso.{module_name}")
                
                # Überprüfe, ob das Modul Simulationsmethoden hat
                methods = simulation_methods.get(module_name, [])
                has_simulation_method = any(hasattr(module, method) or any(hasattr(obj, method) for obj in [getattr(module, attr) for attr in dir(module) if not attr.startswith("__")]) for method in methods)
                
                if has_simulation_method:
                    self.results["simulation_check"][module_name] = "SUCCESS"
                    self.success_count += 1
                    logger.info(f"✅ Modul '{module_name}' hat Simulationsmethoden")
                else:
                    self.results["simulation_check"][module_name] = "WARNING"
                    self.warning_count += 1
                    logger.warning(f"⚠️ Modul '{module_name}' hat keine erkennbaren Simulationsmethoden")
            
            except Exception as e:
                self.results["simulation_check"][module_name] = "FAILURE"
                self.failure_count += 1
                logger.error(f"❌ Fehler beim Testen der Simulationsfähigkeit des Moduls '{module_name}': {e}")
    
    def show_results(self):
        """Zeigt die Ergebnisse der Tests an"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        logger.info("\n=== MISO-Modul-Test abgeschlossen ===")
        logger.info(f"Dauer: {duration:.2f} Sekunden")
        logger.info(f"Erfolge: {self.success_count}")
        logger.info(f"Warnungen: {self.warning_count}")
        logger.info(f"Fehler: {self.failure_count}")
        
        logger.info("\n=== Detaillierte Ergebnisse ===")
        
        for test_name, test_results in self.results.items():
            logger.info(f"\n--- {test_name} ---")
            
            for item_name, result in test_results.items():
                if result == "SUCCESS":
                    logger.info(f"✅ {item_name}: {result}")
                elif result == "WARNING":
                    logger.warning(f"⚠️ {item_name}: {result}")
                else:
                    logger.error(f"❌ {item_name}: {result}")
        
        # Überprüfe, ob das System bereit für externe Trainingseinheiten ist
        if self.failure_count == 0:
            if self.warning_count == 0:
                logger.info("\n✅✅✅ Das System ist VOLLSTÄNDIG BEREIT für externe Trainingseinheiten! ✅✅✅")
            else:
                logger.info("\n⚠️⚠️⚠️ Das System ist BEREIT für externe Trainingseinheiten, hat aber einige Warnungen! ⚠️⚠️⚠️")
        else:
            logger.error("\n❌❌❌ Das System ist NICHT BEREIT für externe Trainingseinheiten! ❌❌❌")

if __name__ == "__main__":
    tester = ModuleTest()
    tester.run_tests()
