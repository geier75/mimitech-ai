#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Komponentenprüfung
"""

import os
import sys
import importlib
import logging

# Konfiguration
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("MISO.Validator")

# ZTM aktivieren
os.environ["MISO_ZTM_MODE"] = "1"
os.environ["MISO_ZTM_LOG_LEVEL"] = "DEBUG"

# Liste der zu prüfenden Module
modules_to_check = [
    # T-Mathematics Engine
    {"path": "miso.math.t_mathematics.engine", "name": "T-Mathematics Engine", "mlx_check": True},
    # VX-QUANTUM
    {"path": "miso.quantum.vx_quantum", "name": "VX-QUANTUM (Bell-Zustände)", "class_check": "BellStateGenerator"},
    # VX-CONTROL
    {"path": "miso.vxor.adapter_core", "name": "VX-CONTROL, vx_adapter_core", "function_check": "get_module_status"},
    # Benchmark-Reporter
    {"path": "miso.benchmarks.reporter", "name": "Benchmark-Reporter mit HTML", "function_check": "generate_html_report"},
    # VX-VISION
    {"path": "miso.vision.vx_vision", "name": "VX-VISION-Kernels", "function_check": "get_available_kernels"}
]

active_modules = 0
inactive_modules = []

# Überprüfe die Module
for module_info in modules_to_check:
    module_path = module_info["path"]
    module_name = module_info["name"]
    
    logger.info(f"Prüfe {module_name}...")
    
    try:
        # Versuche, das Modul zu importieren
        module = importlib.import_module(module_path)
        
        # Prüfe auf MLX-Unterstützung
        if module_info.get("mlx_check"):
            has_mlx = False
            try:
                import mlx
                has_mlx = True
                logger.info(f"✅ {module_name} mit MLX-Unterstützung aktiv")
            except ImportError:
                logger.warning(f"⚠️ {module_name} aktiv, aber MLX nicht verfügbar")
                inactive_modules.append(f"{module_name} (MLX fehlt)")
        
        # Prüfe auf bestimmte Klasse
        elif module_info.get("class_check"):
            class_name = module_info["class_check"]
            if hasattr(module, class_name):
                logger.info(f"✅ {module_name} mit {class_name} aktiv")
                active_modules += 1
            else:
                logger.warning(f"⚠️ {module_name} aktiv, aber {class_name} nicht gefunden")
                inactive_modules.append(f"{module_name} ({class_name} fehlt)")
        
        # Prüfe auf bestimmte Funktion
        elif module_info.get("function_check"):
            function_name = module_info["function_check"]
            if hasattr(module, function_name):
                logger.info(f"✅ {module_name} mit {function_name}() aktiv")
                active_modules += 1
            else:
                logger.warning(f"⚠️ {module_name} aktiv, aber {function_name}() nicht gefunden")
                inactive_modules.append(f"{module_name} ({function_name}() fehlt)")
        
        # Standard-Check
        else:
            logger.info(f"✅ {module_name} aktiv")
            active_modules += 1
            
    except ImportError as e:
        logger.error(f"❌ {module_name} konnte nicht importiert werden: {e}")
        inactive_modules.append(module_name)
    except Exception as e:
        logger.error(f"❌ Fehler bei Prüfung von {module_name}: {e}")
        inactive_modules.append(f"{module_name} (Fehler: {str(e)})")

# Ergebnis ausgeben
print("\n" + "="*50)
if not inactive_modules:
    print("✅ Systemkomponenten vollständig geladen")
else:
    print(f"⚠️ {len(inactive_modules)} von {len(modules_to_check)} Komponenten nicht aktiv:")
    for module in inactive_modules:
        print(f"  - {module}")

print("="*50)
