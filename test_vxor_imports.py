#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR-Module Importtest
----------------------
Testet, ob die VXOR-Module nach der Pfadkorrektur korrekt importiert werden können.
"""

import sys
import os
import importlib
import traceback

# Füge den Pfad zu MISO_Ultimate hinzu, falls er nicht bereits im Pythonpfad ist
sys.path.append("/Volumes/My Book/MISO_Ultimate 15.32.28")

print("=" * 50)
print("VXOR-Module Importtest")
print("=" * 50)

# Liste der zu testenden Module
modules_to_test = [
    "vXor_Modules",
    "vXor_Modules.vx_memex",
    "vXor_Modules.vx_reflex"
]

# Zusätzliche Module, die möglicherweise noch nicht hinzugefügt wurden
missing_modules = [
    "vXor_Modules.vx_psi",
    "vXor_Modules.vx_soma"
]

print("\nTeste verfügbare Module:")
print("-" * 50)
for module_name in modules_to_test:
    try:
        module = importlib.import_module(module_name)
        print(f"✅ {module_name} erfolgreich importiert")
        
        # Zeige verfügbare Untermodule/Klassen für das erste Niveau
        if "." not in module_name:
            print(f"   Verfügbare Untermodule/Dateien:")
            for item in dir(module):
                if not item.startswith("__"):
                    print(f"   - {item}")
        
    except ImportError as e:
        print(f"❌ {module_name} konnte nicht importiert werden: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"❌ {module_name} verursachte einen Fehler: {e}")
        traceback.print_exc()

print("\nBekannte fehlende Module (werden später hinzugefügt):")
print("-" * 50)
for module_name in missing_modules:
    try:
        module = importlib.import_module(module_name)
        print(f"✅ {module_name} erfolgreich importiert (überraschenderweise vorhanden)")
    except ImportError as e:
        print(f"❌ {module_name} fehlt wie erwartet: {e}")
    except Exception as e:
        print(f"❌ {module_name} verursachte einen Fehler: {e}")

print("\nTest abgeschlossen.")
