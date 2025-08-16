#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-REFLEX: Hauptmodul
---------------------
Einstiegspunkt für das VX-REFLEX Modul.

Version: 0.1.0
Author: VXOR Build Core / Omega One
"""

import os
import sys
import logging
import argparse
import json
import time

# Konfiguriere Logging
os.makedirs("/home/ubuntu/vXor_Modules/VX-REFLEX/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/ubuntu/vXor_Modules/VX-REFLEX/logs/reflex.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VX-REFLEX.main")

# Füge Modul-Pfad zum Suchpfad hinzu
sys.path.append("/home/ubuntu/vXor_Modules/VX-REFLEX/src")

# Importiere Module
try:
    from vxor_integration import get_vxor_integration
    from reflex_core import get_reflex_core
    from stimulus_analyzer import get_stimulus_analyzer
    from reflex_responder import get_reflex_responder
    from reaction_profile_manager import get_profile_manager
except ImportError as e:
    logger.error(f"Fehler beim Importieren der Module: {e}")
    sys.exit(1)


def create_manifest():
    """Erstellt die vxor_manifest.json-Datei"""
    manifest = {
        "name": "VX-REFLEX",
        "version": "0.1.0",
        "author": "VXOR Build Core / Omega One",
        "description": "Autonomes Reaktions- und Spontanverhaltensmodul für das VXOR-System",
        "dependencies": [
            "VX-SOMA",
            "VX-PSI",
            "VX-MEMEX",
            "Q-LOGIK"
        ],
        "components": [
            "reflex_core.py",
            "stimulus_analyzer.py",
            "reflex_responder.py",
            "reaction_profile_manager.py",
            "vxor_integration.py"
        ],
        "entry_point": "main.py",
        "config_file": "config/reflex_config.json",
        "logs_dir": "logs",
        "tests_dir": "tests"
    }
    
    manifest_path = "/home/ubuntu/vXor_Modules/VX-REFLEX/vxor_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest erstellt: {manifest_path}")


def create_default_config():
    """Erstellt die Standardkonfiguration, falls nicht vorhanden"""
    config_path = "/home/ubuntu/vXor_Modules/VX-REFLEX/config/reflex_config.json"
    
    if os.path.exists(config_path):
        logger.info(f"Konfigurationsdatei existiert bereits: {config_path}")
        return
    
    # Erstelle Verzeichnis
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Standardkonfiguration
    config = {
        "thresholds": {
            "cpu_load": {
                "high": 90,
                "medium": 75,
                "low": 50
            },
            "audio_level": {
                "high": 75,
                "medium": 60,
                "low": 40
            },
            "object_proximity": {
                "high": 0.8,
                "medium": 1.5,
                "low": 3.0
            }
        },
        "pattern_recognition": {
            "visual_patterns": [
                "schlag",
                "sturz",
                "explosion",
                "feuer",
                "waffe"
            ],
            "audio_patterns": [
                "schrei",
                "explosion",
                "alarm",
                "glasbruch"
            ],
            "danger_objects": [
                "messer",
                "feuer",
                "waffe",
                "fahrzeug"
            ]
        },
        "reaction_profiles": {
            "default": {
                "response_delay": 0.05,
                "priority_weights": {
                    "HIGH": 1.0,
                    "MEDIUM": 0.6,
                    "LOW": 0.3
                },
                "thresholds": {
                    "visual": 0.7,
                    "audio": 0.6,
                    "system": 0.8,
                    "emotional": 0.5
                },
                "response_strength": {
                    "soma": 0.8,
                    "psi": 0.7,
                    "system": 1.0
                }
            },
            "emergency": {
                "response_delay": 0.01,
                "priority_weights": {
                    "HIGH": 1.0,
                    "MEDIUM": 0.8,
                    "LOW": 0.5
                },
                "thresholds": {
                    "visual": 0.5,
                    "audio": 0.4,
                    "system": 0.6,
                    "emotional": 0.3
                },
                "response_strength": {
                    "soma": 1.0,
                    "psi": 0.9,
                    "system": 1.0
                }
            }
        },
        "active_profile": "default",
        "learning": {
            "enabled": False,
            "rate": 0.1,
            "max_adjustments": {
                "thresholds": 0.3,
                "response_strength": 0.3,
                "priority_weights": 0.2
            }
        },
        "vxor_bridge": {
            "connection_retry_interval": 5,
            "max_connection_attempts": 3,
            "signal_timeout": 2.0,
            "queue_processing_interval": 0.05,
            "max_queue_size": 100,
            "modules": {
                "vx-soma": {
                    "enabled": True,
                    "path": "/vXor_Modules/VX-SOMA",
                    "interface": "api"
                },
                "vx-psi": {
                    "enabled": True,
                    "path": "/vXor_Modules/VX-PSI",
                    "interface": "api"
                },
                "vx-memex": {
                    "enabled": True,
                    "path": "/vXor_Modules/VX-MEMEX",
                    "interface": "api"
                },
                "q-logik": {
                    "enabled": True,
                    "path": "/vXor_Modules/Q-LOGIK",
                    "interface": "api"
                }
            }
        },
        "max_queue_size": 100,
        "processing_interval": 0.01
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Standardkonfiguration erstellt: {config_path}")


def run_tests():
    """Führt die Testsuite aus"""
    import unittest
    
    logger.info("Führe Tests aus...")
    
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("/home/ubuntu/vXor_Modules/VX-REFLEX/tests", pattern="test_*.py")
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    if result.wasSuccessful():
        logger.info("Alle Tests erfolgreich")
        return True
    else:
        logger.error("Tests fehlgeschlagen")
        return False


def run_demo():
    """Führt eine Demo-Simulation aus"""
    logger.info("Starte Demo-Simulation...")
    
    # Initialisiere VX-REFLEX
    integration = get_vxor_integration()
    
    # Starte VX-REFLEX
    integration.start()
    logger.info("VX-REFLEX gestartet")
    
    try:
        # Simuliere verschiedene Reize
        logger.info("Simuliere visuellen Reiz (Waffe)")
        visual_result = integration.process_external_stimulus("visual", {
            "objects": [
                {"name": "waffe", "distance": 0.5, "confidence": 0.92}
            ],
            "motion_data": {"velocity": 7.5}
        })
        logger.info(f"Ergebnis: {visual_result}")
        
        time.sleep(1)
        
        logger.info("Simuliere auditiven Reiz (Schrei)")
        audio_result = integration.process_external_stimulus("audio", {
            "level": 80,
            "frequency_data": [100, 500, 2000, 5000],
            "audio_pattern": "schrei"
        })
        logger.info(f"Ergebnis: {audio_result}")
        
        time.sleep(1)
        
        logger.info("Simuliere Systemreiz (CPU-Überlast)")
        system_result = integration.process_external_stimulus("system", {
            "cpu_load": 95,
            "memory": 82,
            "temperature": 78
        })
        logger.info(f"Ergebnis: {system_result}")
        
        time.sleep(1)
        
        # Zeige Status
        status = integration.get_status()
        logger.info(f"Status: {json.dumps(status, indent=2)}")
        
    finally:
        # Stoppe VX-REFLEX
        integration.stop()
        logger.info("VX-REFLEX gestoppt")


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="VX-REFLEX: Autonomes Reaktions- und Spontanverhaltensmodul")
    parser.add_argument("--test", action="store_true", help="Führt die Testsuite aus")
    parser.add_argument("--demo", action="store_true", help="Führt eine Demo-Simulation aus")
    parser.add_argument("--create-manifest", action="store_true", help="Erstellt die vxor_manifest.json-Datei")
    parser.add_argument("--create-config", action="store_true", help="Erstellt die Standardkonfiguration")
    
    args = parser.parse_args()
    
    if args.create_manifest:
        create_manifest()
    
    if args.create_config:
        create_default_config()
    
    if args.test:
        success = run_tests()
        if not success:
            sys.exit(1)
    
    if args.demo:
        run_demo()
    
    # Wenn keine Argumente angegeben wurden, zeige Hilfe
    if not (args.test or args.demo or args.create_manifest or args.create_config):
        parser.print_help()


if __name__ == "__main__":
    main()
