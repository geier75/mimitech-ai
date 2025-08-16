#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug-Skript für Import-Probleme
"""

import os
import sys
import importlib
import traceback

# Konfiguriere Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_imports")

# Verzeichnispfade
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
print(f"Aktuelles Verzeichnis: {current_dir}")
print(f"Elternverzeichnis: {parent_dir}")

# Prüfe sys.path
print("\nSystem-Pfade:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

# Prüfe Abhängigkeiten
dependencies = [
    "psutil",
    "mlx",
    "torch",
    "numpy",
    "matplotlib"
]

print("\nPrüfe Abhängigkeiten:")
for dep in dependencies:
    try:
        module = importlib.import_module(dep)
        version = getattr(module, "__version__", "Unbekannte Version")
        print(f"  ✓ {dep} ({version})")
    except ImportError as e:
        print(f"  ✗ {dep}: {e}")

# Prüfe lokale Module
print("\nPrüfe lokale Module:")

modules_to_check = [
    "miso.math.t_mathematics.mlx_support",
    "miso.math.t_mathematics.optimizations.integration",
    "miso.math.t_mathematics.optimizations.advanced_svd_benchmark",
    "miso.math.t_mathematics.optimizations.advanced_svd_benchmark_functions"
]

for module_name in modules_to_check:
    try:
        module = importlib.import_module(module_name)
        print(f"  ✓ {module_name}")
    except Exception as e:
        print(f"  ✗ {module_name}: {e}")
        print(f"    {traceback.format_exc().splitlines()[-3]}")

print("\nImport-Test abgeschlossen.")
