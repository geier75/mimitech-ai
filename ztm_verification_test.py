#!/usr/bin/env python3
"""
ZTM-Verifikationstest für MISO-Module
Datum: 23.03.2025
Status: [ZTM VERIFICATION]

Dieses Skript testet alle MISO-Module auf Funktionalität und ZTM-Konformität.
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
    format='[%(asctime)s] [ZTM] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("ztm_verification.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ZTM-Verification")

# Liste aller zu testenden Module
MODULES = [
    "omega",
    "mcode",
    "mlingua",
    "mprime",
    "quantum",
    "prism",
    "void",
    "nexus",
    "timeline",
    "perception",
    "strategic",
    "tmathematics",
    "mimimon"
]

# Mapping von Modulnamen zu tatsächlichen Import-Pfaden
MODULE_IMPORT_PATHS = {
    "omega": "miso.core.omega_core",
    "mcode": "miso.lang.mcode.code_compiler",
    "mlingua": "miso.lang.mlingua.syntax_analyzer",
    "mprime": "miso.math.mprime.optimization_engine",
    "quantum": "miso.logic.quantum",
    "prism": "miso.simulation.prism_engine",
    "void": "miso.protect.void_protocol",
    "nexus": "miso.core.nexus_os.resource_manager",
    "timeline": "miso.timeline",
    "perception": "miso.perception",
    "strategic": "miso.strategic",
    "tmathematics": "miso.math.t_mathematics",
    "mimimon": "miso.security.ztm.mimimon"
}

# Testfunktionen
def test_module_existence(module_name):
    """Testet, ob das Modul existiert"""
    try:
        # Verwende das Mapping für den korrekten Import-Pfad
        module_path = MODULE_IMPORT_PATHS.get(module_name, f"miso.{module_name}")
        module = importlib.import_module(module_path)
        logger.info(f"✅ [ZTM VERIFIED] Modul {module_name} existiert")
        return module
    except ImportError as e:
        logger.error(f"❌ Modul {module_name} konnte nicht importiert werden: {e}")
        return None

def test_ztm_policy(module_name):
    """Testet, ob die ZTM-Policy-Datei existiert"""
    # Bestimme den Basispfad basierend auf dem Import-Pfad
    import_path = MODULE_IMPORT_PATHS.get(module_name, f"miso.{module_name}")
    base_path = import_path.replace(".", "/").split("/")[0:2]
    base_dir = "/".join(base_path)
    
    # Überprüfe verschiedene mögliche Speicherorte für die ZTM-Policy
    policy_paths = [
        f"{base_dir}/{module_name}.ztm",
        f"{base_dir}/{module_name}_ztm_policy.json",
        f"{base_dir}/ztm/{module_name}_ztm_policy.json"
    ]
    
    for policy_path in policy_paths:
        if os.path.exists(policy_path):
            logger.info(f"✅ [ZTM VERIFIED] ZTM-Policy für {module_name} existiert: {policy_path}")
            return True
    
    logger.warning(f"⚠️ ZTM-Policy für {module_name} fehlt")
    return False

def test_module_functionality(module, module_name):
    """Testet die Funktionalität des Moduls"""
    if module is None:
        logger.error(f"❌ Funktionalitätstest für {module_name} übersprungen (Modul nicht vorhanden)")
        return False
    
    success = True
    
    # Boot-Ping Test
    logger.info(f"[ZTM] Boot-Ping Test für {module_name}")
    try:
        # Versuche, die Hauptklasse zu instanziieren
        main_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr_name.lower() == module_name.lower() or \
               attr_name.endswith("Core") or attr_name.endswith("Engine") or attr_name.endswith("Processor"):
                main_class = attr
                break
        
        if main_class:
            instance = main_class()
            logger.info(f"✅ [ZTM VERIFIED] Boot-Ping für {module_name} erfolgreich")
        else:
            # Fallback: Prüfe, ob das Modul selbst initialisiert werden kann
            if hasattr(module, "initialize"):
                module.initialize()
                logger.info(f"✅ [ZTM VERIFIED] Boot-Ping für {module_name} erfolgreich (Modul-Initialisierung)")
            else:
                logger.warning(f"⚠️ Keine Hauptklasse oder Initialisierungsmethode für {module_name} gefunden")
                success = False
    except Exception as e:
        logger.error(f"❌ Boot-Ping für {module_name} fehlgeschlagen: {e}")
        traceback.print_exc()
        success = False
    
    # Methoden-Test
    logger.info(f"[ZTM] Methoden-Test für {module_name}")
    try:
        # Prüfe, ob das Modul die erwarteten Methoden hat
        expected_methods = []
        if module_name == "omega":
            expected_methods = ["initialize", "register_component", "start"]
        elif module_name == "mcode":
            expected_methods = ["execute_code", "validate_code"]
        elif module_name == "mlingua":
            expected_methods = ["process_text", "parse_command"]
        elif module_name == "mprime":
            expected_methods = ["run_simulation", "calculate"]
        elif module_name == "quantum":
            expected_methods = ["create_superposition", "measure"]
        elif module_name == "prism":
            expected_methods = ["simulate_timeline", "resolve_paradox"]
        elif module_name == "void":
            expected_methods = ["initialize_protocol", "secure_channel"]
        elif module_name == "nexus":
            expected_methods = ["schedule_task", "optimize_workflow"]
        elif module_name == "timeline":
            expected_methods = ["create_node", "connect_nodes"]
        elif module_name == "perception":
            expected_methods = ["process_input", "filter_data"]
        elif module_name == "strategic":
            expected_methods = ["analyze_threat", "profile_entity"]
        elif module_name == "tmathematics":
            expected_methods = ["calculate_vector", "optimize_matrix"]
        elif module_name == "mimimon":
            expected_methods = ["verify_ztm", "log_action"]
        
        if main_class:
            instance = main_class()
            for method in expected_methods:
                if hasattr(instance, method):
                    logger.info(f"✅ [ZTM VERIFIED] Methode {method} für {module_name} vorhanden")
                else:
                    logger.warning(f"⚠️ Methode {method} für {module_name} fehlt")
                    success = False
        else:
            # Fallback: Prüfe, ob das Modul selbst die Methoden hat
            for method in expected_methods:
                if hasattr(module, method):
                    logger.info(f"✅ [ZTM VERIFIED] Methode {method} für {module_name} vorhanden (Modul-Ebene)")
                else:
                    logger.warning(f"⚠️ Methode {method} für {module_name} fehlt")
                    success = False
    except Exception as e:
        logger.error(f"❌ Methoden-Test für {module_name} fehlgeschlagen: {e}")
        traceback.print_exc()
        success = False
    
    # Logging-Test
    logger.info(f"[ZTM] Logging-Test für {module_name}")
    try:
        # Prüfe, ob das Modul Logging verwendet
        if hasattr(module, "logger"):
            logger.info(f"✅ [ZTM VERIFIED] Logging für {module_name} implementiert")
        else:
            # Suche nach Logging-Instanzen in Untermodulen
            logging_found = False
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, logging.Logger):
                    logging_found = True
                    break
            
            if logging_found:
                logger.info(f"✅ [ZTM VERIFIED] Logging für {module_name} implementiert")
            else:
                logger.warning(f"⚠️ Kein Logging für {module_name} gefunden")
                success = False
    except Exception as e:
        logger.error(f"❌ Logging-Test für {module_name} fehlgeschlagen: {e}")
        traceback.print_exc()
        success = False
    
    # Simulationstest
    logger.info(f"[ZTM] Simulationstest für {module_name}")
    try:
        # Führe eine einfache Simulation durch
        if main_class:
            instance = main_class()
            if hasattr(instance, "run_simulation"):
                instance.run_simulation()
                logger.info(f"✅ [ZTM VERIFIED] Simulationstest für {module_name} erfolgreich")
            elif hasattr(instance, "initialize") and hasattr(instance, "process"):
                instance.initialize()
                instance.process({"test": "data"})
                logger.info(f"✅ [ZTM VERIFIED] Simulationstest für {module_name} erfolgreich")
            else:
                logger.warning(f"⚠️ Keine Simulationsmethode für {module_name} gefunden")
                success = False
        else:
            logger.warning(f"⚠️ Keine Hauptklasse für Simulationstest von {module_name} gefunden")
            success = False
    except Exception as e:
        logger.error(f"❌ Simulationstest für {module_name} fehlgeschlagen: {e}")
        traceback.print_exc()
        success = False
    
    return success

def run_tests():
    """Führt alle Tests für alle Module durch"""
    results = {}
    
    logger.info("=== [ZTM VERIFICATION] Start der Modultests ===")
    logger.info(f"Zeitstempel: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for module_name in MODULES:
        logger.info(f"\n=== [ZTM] Teste Modul: {module_name} ===")
        
        # Teste Modulexistenz
        module = test_module_existence(module_name)
        
        # Teste ZTM-Policy
        policy_exists = test_ztm_policy(module_name)
        
        # Teste Funktionalität
        functionality_ok = test_module_functionality(module, module_name)
        
        # Speichere Ergebnisse
        results[module_name] = {
            "exists": module is not None,
            "policy_exists": policy_exists,
            "functionality_ok": functionality_ok,
            "status": "[ZTM VERIFIED]" if module is not None and functionality_ok else "Unvollständig"
        }
    
    # Ausgabe der Zusammenfassung
    logger.info("\n=== [ZTM VERIFICATION] Zusammenfassung ===")
    all_verified = True
    for module_name, result in results.items():
        status = result["status"]
        if status != "[ZTM VERIFIED]":
            all_verified = False
        logger.info(f"{module_name}: {status}")
    
    if all_verified:
        logger.info("\n✅ [ZTM VERIFIED] Alle Module erfolgreich verifiziert!")
        # Akustisches Signal
        print('\a')  # Beep
    else:
        logger.warning("\n⚠️ Nicht alle Module konnten vollständig verifiziert werden.")
    
    logger.info(f"Zeitstempel: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=== [ZTM VERIFICATION] Ende der Modultests ===")
    
    return results

if __name__ == "__main__":
    # Stelle sicher, dass das MISO-Paket im Python-Pfad ist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir)
    
    # Gib den Suchpfad aus
    logger.info(f"Python-Suchpfad: {sys.path}")
    
    # Führe Tests durch
    results = run_tests()
    
    # Ausgabe für Benutzer
    print("\nZTM-Verifikation abgeschlossen. Siehe ztm_verification.log für Details.")
    
    # Akustisches Signal
    print('\a')  # Beep
