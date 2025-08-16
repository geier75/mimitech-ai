#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Importtest
--------------------
Testet, ob das VX-CONTEXT-Modul nach der Integration korrekt importiert werden kann.
"""

import sys
import importlib
import traceback

# Füge den Pfad zu MISO_Ultimate hinzu, falls er nicht bereits im Pythonpfad ist
sys.path.append("/Volumes/My Book/MISO_Ultimate 15.32.28")

print("=" * 50)
print("VX-CONTEXT Importtest")
print("=" * 50)

# Liste der zu testenden Module
modules_to_test = [
    "vXor_Modules.vx_context",
    "vXor_Modules.vx_context.context_core",
    "vXor_Modules.vx_context.context_bridge",
    "vXor_Modules.vx_context.context_analyzer"
]

print("\nTeste VX-CONTEXT Module:")
print("-" * 50)
for module_name in modules_to_test:
    try:
        module = importlib.import_module(module_name)
        print(f"✅ {module_name} erfolgreich importiert")
        
        # Für Hauptmodul: Zeige verfügbare Klassen/Funktionen
        if module_name == "vXor_Modules.vx_context.context_core":
            print(f"   Verfügbare Klassen/Funktionen:")
            for item in dir(module):
                if not item.startswith("__"):
                    print(f"   - {item}")
        
    except ImportError as e:
        print(f"❌ {module_name} konnte nicht importiert werden: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"❌ {module_name} verursachte einen Fehler: {e}")
        traceback.print_exc()

print("\nTest abgeschlossen.")
