#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - ZTM-Aktivierung

Dieses Skript aktiviert das MIMIMON ZTM-Modul und initialisiert die VOID-Protokoll-Integration.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import json
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Füge den MISO-Hauptpfad zum Systempfad hinzu
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [ZTM-ACTIVATOR] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.Security.ZTM.Activator")

# Konstanten
MIMIMON_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "mimimon_config.json")
VOID_MODULES = [
    "security.void.void_protocol",
    "security.void.void_crypto",
    "security.void.void_context"
]

def ensure_log_directory_exists():
    """Stellt sicher, dass das Log-Verzeichnis existiert"""
    log_dir = os.path.join(root_dir, "logs")
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
            logger.info(f"Log-Verzeichnis erstellt: {log_dir}")
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Log-Verzeichnisses: {e}")
            return False
    return True

def activate_ztm():
    """Aktiviert das MIMIMON ZTM-Modul"""
    try:
        # Stelle sicher, dass das Log-Verzeichnis existiert
        if not ensure_log_directory_exists():
            return False
        
        # Importiere das MIMIMON-Modul
        from miso.security.ztm.mimimon import MIMIMON
        
        # Erstelle eine Instanz mit der Konfigurationsdatei
        mimimon = MIMIMON(config_file=MIMIMON_CONFIG_PATH)
        
        # Führe eine Selbstverifizierung durch
        if not mimimon.verify_ztm():
            logger.error("MIMIMON ZTM-Selbstverifizierung fehlgeschlagen")
            return False
        
        # Verifiziere kritische Module
        config = None
        try:
            with open(MIMIMON_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden der MIMIMON-Konfiguration: {e}")
            return False
        
        critical_modules = config.get("critical_modules", [])
        for module_name in critical_modules:
            try:
                # Ermittle den Modulpfad
                module_path = None
                try:
                    # Versuche, das Modul zu importieren, um den Pfad zu ermitteln
                    module = importlib.import_module(module_name)
                    module_path = getattr(module, "__file__", None)
                except ImportError:
                    # Wenn Import fehlschlägt, setze einen Dummy-Pfad
                    module_path = f"{module_name.replace('.', '/')}.py"
                
                if module_path:
                    logger.info(f"Verifiziere Modul: {module_name} ({module_path})")
                    result = mimimon.verify_module(module_name, module_path)
                    if not result.get("verified", False):
                        logger.warning(f"Modul-Verifikation fehlgeschlagen: {module_name}")
                else:
                    logger.warning(f"Konnte keinen Pfad für Modul ermitteln: {module_name}")
            except Exception as e:
                logger.error(f"Fehler bei der Verifikation von Modul {module_name}: {e}")
        
        # Aktiviere VOID-Integration
        if config.get("void_integration", {}).get("enabled", False):
            if not activate_void_integration(config.get("void_integration", {})):
                logger.warning("VOID-Integration konnte nicht vollständig aktiviert werden")
        
        logger.info("MIMIMON ZTM-Modul erfolgreich aktiviert")
        return True
    
    except Exception as e:
        logger.error(f"Fehler bei der Aktivierung des MIMIMON ZTM-Moduls: {e}")
        return False

def activate_void_integration(void_config: Dict[str, Any]):
    """Aktiviert die VOID-Protokoll-Integration"""
    try:
        logger.info("Aktiviere VOID-Protokoll-Integration")
        
        success_count = 0
        for module_name in VOID_MODULES:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "init"):
                    init_func = getattr(module, "init")
                    if init_func():
                        logger.info(f"VOID-Modul erfolgreich initialisiert: {module_name}")
                        success_count += 1
                    else:
                        logger.warning(f"VOID-Modul konnte nicht initialisiert werden: {module_name}")
                else:
                    logger.warning(f"VOID-Modul hat keine init()-Funktion: {module_name}")
            except Exception as e:
                logger.error(f"Fehler beim Importieren/Initialisieren von VOID-Modul {module_name}: {e}")
        
        # VOID-Integration war erfolgreich, wenn mindestens 2 von 3 Modulen initialisiert wurden
        if success_count >= 2:
            logger.info(f"VOID-Protokoll-Integration aktiviert ({success_count}/{len(VOID_MODULES)} Module)")
            
            # Speichere die VOID-Konfiguration in einer separaten Datei
            void_config_path = os.path.join(os.path.dirname(__file__), "..", "..", "void_config.json")
            try:
                with open(void_config_path, 'w') as f:
                    json.dump(void_config, f, indent=2)
                logger.info(f"VOID-Konfiguration gespeichert: {void_config_path}")
            except Exception as e:
                logger.warning(f"Konnte VOID-Konfiguration nicht speichern: {e}")
            
            return True
        else:
            logger.warning(f"VOID-Protokoll-Integration teilweise fehlgeschlagen ({success_count}/{len(VOID_MODULES)} Module)")
            return False
    
    except Exception as e:
        logger.error(f"Fehler bei der Aktivierung der VOID-Protokoll-Integration: {e}")
        return False

if __name__ == "__main__":
    # Aktiviere das ZTM-System
    if activate_ztm():
        logger.info("ZTM-System erfolgreich aktiviert")
        sys.exit(0)
    else:
        logger.error("ZTM-System konnte nicht aktiviert werden")
        sys.exit(1)
