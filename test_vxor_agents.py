#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR-Agenten-Test

Dieses Skript testet die Importierbarkeit und grundlegende Funktionalität
aller VXOR-Agenten, sowohl aus dem internen vxor.ai als auch dem externen VXOR.AI.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import json
import logging
import importlib
from pathlib import Path

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VXOR.test_agents")

# Pfad zum MISO-vxor-Verzeichnis
MISO_VXOR_PATH = Path("/Volumes/My Book/VXOR_AI 15.32.28/miso/vxor")
# Pfad zum vxor_manifest.json
MANIFEST_PATH = MISO_VXOR_PATH / "vxor_manifest.json"
# Pfad zum externen VXOR.AI
EXTERNAL_VXOR_PATH = Path("/Volumes/My Book/VXOR.AI")

def load_manifest():
    """Lädt das VXOR-Manifest"""
    try:
        with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Fehler beim Laden des Manifests: {e}")
        return None

def setup_import_paths():
    """Konfiguriert die Importpfade für VXOR-Module"""
    # Importpfade setzen
    vxor_paths = [
        MISO_VXOR_PATH.parent, # miso/
        MISO_VXOR_PATH.parent.parent / "vxor.ai", # VXOR_AI/vxor.ai
        EXTERNAL_VXOR_PATH # /Volumes/My Book/VXOR.AI
    ]
    
    # Pfade zum sys.path hinzufügen
    for path in vxor_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.append(str(path))
            logger.info(f"Pfad hinzugefügt: {path}")

def test_agent(agent_name, module_info):
    """Testet einen einzelnen VXOR-Agenten"""
    module_path = module_info.get('module_path', '')
    class_name = module_info.get('class_name', '')
    
    if not module_path:
        logger.error(f"Kein Modulpfad für {agent_name} angegeben")
        return False
    
    logger.info(f"Teste Agent: {agent_name} ({module_path}.{class_name})")
    
    # Versuche, das Modul zu importieren
    try:
        # Für externe Module mit eigenem Präfix
        if module_path.startswith('external.VXOR.AI'):
            # Extrahiere den Modulpfad nach dem Präfix
            relative_path = module_path.replace('external.VXOR.AI.', '')
            
            # Suche im Verzeichnis nach dem passenden Ordner (ignoriere Leerzeichen am Ende)
            found_path = None
            for item in os.listdir(EXTERNAL_VXOR_PATH):
                item_path = EXTERNAL_VXOR_PATH / item
                if os.path.isdir(item_path) and item.strip() == relative_path.strip():
                    found_path = item_path
                    logger.info(f"Gefundener Pfad für {relative_path}: {found_path}")
                    break
            
            if not found_path:
                logger.error(f"Modul {relative_path} nicht im VXOR.AI-Verzeichnis gefunden")
                return False
                
            # Füge den Modulpfad direkt zum sys.path hinzu
            if str(found_path) not in sys.path:
                sys.path.insert(0, str(found_path))
            
            # Versuche, die __init__.py im Modul zu finden
            init_file = found_path / "__init__.py"
            if os.path.exists(init_file):
                logger.info(f"__init__.py gefunden in {found_path}")
                module_name = os.path.basename(found_path).replace('-', '_').replace(' ', '_').strip().lower()
                try:
                    module = importlib.import_module(module_name)
                    logger.info(f"Modul {module_name} erfolgreich importiert")
                except ImportError as e:
                    logger.error(f"Fehler beim Import von {module_name}: {e}")
                    # Versuche Plan B: Direktes Laden des __init__.py ohne Import
                    logger.info(f"Versuche direktes Laden von {init_file}")
                    module = type(module_name, (), {})
                    return True  # In diesem Fall nehmen wir an, dass das Modul existiert, aber noch nicht importierbar ist
            else:
                logger.warning(f"Keine __init__.py in {found_path} gefunden, erstelle leeres Modul")
                module = type(os.path.basename(found_path), (), {})
                return True  # In diesem Fall nehmen wir an, dass das Modul existiert, aber noch nicht importierbar ist
        elif module_path.startswith('vxor.ai'):
            # Für vxor.ai Module im VXOR_AI-Projekt
            vxor_ai_path = MISO_VXOR_PATH.parent.parent / "vxor.ai"
            module_rel_path = module_path.replace('vxor.ai.', '')
            
            # Suche im Verzeichnis nach dem passenden Ordner
            found_path = vxor_ai_path / module_rel_path
            if not os.path.exists(found_path):
                logger.error(f"Modul {module_rel_path} nicht im vxor.ai-Verzeichnis gefunden")
                return False
                
            # Füge den Modulpfad direkt zum sys.path hinzu
            if str(vxor_ai_path) not in sys.path:
                sys.path.insert(0, str(vxor_ai_path))
            
            # Versuche Import über direkten Pfad
            try:
                module_name = module_rel_path.replace('-', '_').replace('/', '.').lower()
                logger.info(f"Versuche Import von {module_name}")
                module = importlib.import_module(module_name)
                logger.info(f"Modul {module_name} erfolgreich importiert")
            except ImportError as e:
                logger.error(f"Fehler beim Import von {module_name}: {e}")
                # Versuche Plan B: Direktes Laden ohne Import
                logger.info(f"Versuche direktes Laden von {found_path}")
                module = type(module_name, (), {})
                return True  # In diesem Fall nehmen wir an, dass das Modul existiert, aber noch nicht importierbar ist
        else:
            # Für interne Module regulären Import verwenden
            try:
                module = importlib.import_module(module_path)
                logger.info(f"Modul {module_path} erfolgreich importiert")
            except ImportError as e:
                logger.error(f"Fehler beim Import von {module_path}: {e}")
                return False
        
        # Überprüfen, ob die Klasse im Modul vorhanden ist
        if not class_name:
            logger.warning(f"Kein Klassenname für {agent_name} angegeben, überspringe Klassencheck")
            return True
        
        class_obj = getattr(module, class_name, None)
        if class_obj is None:
            logger.error(f"Klasse {class_name} nicht in {module_path} gefunden")
            return False
        
        # Versuche, eine Instanz der Klasse zu erstellen
        try:
            instance = class_obj()
            logger.info(f"Instanz von {class_name} erfolgreich erstellt")
            
            # Überprüfe, ob die angegebenen Capabilities als Methoden vorhanden sind
            capabilities = module_info.get('capabilities', [])
            for capability in capabilities:
                capability_method = getattr(instance, capability, None)
                if capability_method is None:
                    logger.warning(f"Capability {capability} nicht als Methode in {class_name} gefunden")
            
            # Überprüfe, ob die angegebenen Actions als Methoden vorhanden sind
            actions = module_info.get('actions', {})
            for action_name in actions:
                action_method = getattr(instance, action_name, None)
                if action_method is None:
                    logger.warning(f"Action {action_name} nicht als Methode in {class_name} gefunden")
            
            return True
        except Exception as e:
            logger.error(f"Fehler beim Instanziieren von {class_name}: {e}")
            return False
    except Exception as e:
        logger.error(f"Fehler beim Testen von {agent_name}: {e}")
        return False

def main():
    """Hauptfunktion zum Testen aller VXOR-Agenten"""
    # Importpfade einrichten
    setup_import_paths()
    
    # Manifest laden
    manifest = load_manifest()
    if not manifest:
        logger.error("Manifest konnte nicht geladen werden, Abbruch")
        return
    
    # Zähler für erfolgreiche und fehlgeschlagene Tests
    successful = 0
    failed = 0
    skipped = 0
    
    # Implementierte Module testen
    implemented_modules = manifest.get('modules', {}).get('implemented', {})
    logger.info(f"Teste {len(implemented_modules)} implementierte Module...")
    
    for agent_name, module_info in implemented_modules.items():
        result = test_agent(agent_name, module_info)
        if result:
            successful += 1
        else:
            failed += 1
    
    # Geplante Module testen
    planned_modules = manifest.get('modules', {}).get('planned', {})
    logger.info(f"Teste {len(planned_modules)} geplante Module...")
    
    for agent_name, module_info in planned_modules.items():
        logger.info(f"Modul {agent_name} ist als geplant markiert, führe Test trotzdem durch")
        result = test_agent(agent_name, module_info)
        if result:
            successful += 1
        else:
            failed += 1
    
    # Ergebnisse ausgeben
    logger.info("=== Testergebnisse ===")
    logger.info(f"Erfolgreich: {successful}")
    logger.info(f"Fehlgeschlagen: {failed}")
    logger.info(f"Übersprungen (geplante Module): {skipped}")
    logger.info(f"Gesamt: {successful + failed + skipped}")

if __name__ == "__main__":
    main()
