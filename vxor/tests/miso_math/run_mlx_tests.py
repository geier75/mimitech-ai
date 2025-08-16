#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testrunner für die MLX-Optimierung der T-Mathematics Engine.

Dieses Skript führt die Tests für die MLX-Optimierung der T-Mathematics Engine aus
und gibt einen Bericht über die Ergebnisse aus.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import platform
import subprocess
import importlib.util

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Prüfe auf Apple Silicon
is_apple_silicon = False
try:
    is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
    if is_apple_silicon:
        print("Apple Silicon erkannt")
    else:
        print(f"Kein Apple Silicon erkannt: {platform.processor()}, {platform.system()}")
except Exception as e:
    print(f"Fehler bei der Erkennung der Prozessorarchitektur: {e}")

# Prüfe auf MLX
has_mlx = False
try:
    import mlx.core
    has_mlx = True
    print("MLX-Bibliothek erfolgreich importiert")
except ImportError:
    print("MLX-Bibliothek nicht gefunden. Versuche, sie zu installieren...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mlx"])
        import mlx.core
        has_mlx = True
        print("MLX-Bibliothek erfolgreich installiert und importiert")
    except Exception as e:
        print(f"Fehler beim Installieren von MLX: {e}")
        print("Führe Tests ohne MLX-Optimierung aus")

# Prüfe auf PyTorch
has_pytorch = False
try:
    import torch
    has_pytorch = True
    print(f"PyTorch Version {torch.__version__} erfolgreich importiert")
except ImportError:
    print("PyTorch nicht gefunden. Versuche, es zu installieren...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
        import torch
        has_pytorch = True
        print(f"PyTorch Version {torch.__version__} erfolgreich installiert und importiert")
    except Exception as e:
        print(f"Fehler beim Installieren von PyTorch: {e}")
        print("Tests können nicht ausgeführt werden, da PyTorch benötigt wird")
        sys.exit(1)

# Prüfe auf NumPy
has_numpy = False
try:
    import numpy
    has_numpy = True
    print(f"NumPy Version {numpy.__version__} erfolgreich importiert")
except ImportError:
    print("NumPy nicht gefunden. Versuche, es zu installieren...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
        import numpy
        has_numpy = True
        print(f"NumPy Version {numpy.__version__} erfolgreich installiert und importiert")
    except Exception as e:
        print(f"Fehler beim Installieren von NumPy: {e}")
        print("Tests können nicht ausgeführt werden, da NumPy benötigt wird")
        sys.exit(1)

# Prüfe, ob die Testdatei existiert
test_file = os.path.join(os.path.dirname(__file__), 'test_t_mathematics_mlx.py')
if not os.path.exists(test_file):
    print(f"Testdatei {test_file} nicht gefunden")
    sys.exit(1)

# Prüfe, ob die T-Mathematics Engine existiert
try:
    from miso.math.t_mathematics.engine import TMathEngine
    print("T-Mathematics Engine erfolgreich importiert")
except ImportError as e:
    print(f"Fehler beim Importieren der T-Mathematics Engine: {e}")
    print("Tests können nicht ausgeführt werden")
    sys.exit(1)

# Führe die Tests aus
print("\n" + "="*80)
print("Starte Tests für die MLX-Optimierung der T-Mathematics Engine")
print("="*80 + "\n")

if is_apple_silicon and has_mlx:
    print("Führe Tests mit MLX-Optimierung aus")
    os.environ["T_MATH_USE_MLX"] = "1"
else:
    print("Führe Tests ohne MLX-Optimierung aus (Fallback-Modus)")
    os.environ["T_MATH_USE_MLX"] = "0"

# Lade die Testdatei
spec = importlib.util.spec_from_file_location("test_t_mathematics_mlx", test_file)
test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)

# Führe die Tests aus
test_suite = unittest.TestLoader().loadTestsFromModule(test_module)
test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)

# Gib das Ergebnis aus
print("\n" + "="*80)
print(f"Testergebnisse: {test_result.testsRun} Tests ausgeführt")
print(f"Erfolgreich: {test_result.testsRun - len(test_result.errors) - len(test_result.failures)}")
print(f"Fehler: {len(test_result.errors)}")
print(f"Fehlschläge: {len(test_result.failures)}")
print("="*80 + "\n")

# Gib Details zu Fehlern aus
if test_result.errors:
    print("Fehler:")
    for test, error in test_result.errors:
        print(f"\n{test}")
        print("-"*80)
        print(error)
        print("-"*80)

# Gib Details zu Fehlschlägen aus
if test_result.failures:
    print("Fehlschläge:")
    for test, failure in test_result.failures:
        print(f"\n{test}")
        print("-"*80)
        print(failure)
        print("-"*80)

# Beende mit entsprechendem Exit-Code
sys.exit(len(test_result.errors) + len(test_result.failures))
