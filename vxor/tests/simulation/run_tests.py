#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Testlauf-Skript

Dieses Skript führt die Tests für die PRISM-Engine und ihre Integration mit ECHO-PRIME aus.
Es prüft zuerst, ob alle erforderlichen Abhängigkeiten installiert sind, und führt dann die Tests aus.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import subprocess
import importlib
import unittest

def check_dependencies():
    """Überprüft, ob alle erforderlichen Abhängigkeiten installiert sind"""
    required_packages = [
        "numpy",
        "torch",
        "mock"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} ist installiert")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} fehlt")
    
    return missing_packages

def install_dependencies(packages):
    """Installiert fehlende Abhängigkeiten"""
    for package in packages:
        print(f"Installiere {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} wurde installiert")
        except subprocess.CalledProcessError:
            print(f"❌ Fehler bei der Installation von {package}")
            return False
    
    return True

def run_tests():
    """Führt die Tests aus"""
    print("\nFühre Tests aus...\n")
    
    # Füge Projektverzeichnis zum Pfad hinzu
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    # Entdecke und führe Tests aus
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests/simulation", pattern="test_*.py")
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("MISO - Testlauf für PRISM-Engine und PRISM-ECHO-PRIME Integration\n")
    
    # Überprüfe Abhängigkeiten
    print("Überprüfe Abhängigkeiten...")
    missing_packages = check_dependencies()
    
    # Installiere fehlende Abhängigkeiten
    if missing_packages:
        print("\nFehlende Abhängigkeiten gefunden. Möchten Sie diese installieren? (j/n)")
        choice = input().lower()
        
        if choice == "j" or choice == "ja":
            if not install_dependencies(missing_packages):
                print("\n❌ Fehler bei der Installation der Abhängigkeiten. Bitte installieren Sie diese manuell.")
                sys.exit(1)
        else:
            print("\n❌ Tests können nicht ausgeführt werden, da Abhängigkeiten fehlen.")
            sys.exit(1)
    
    # Führe Tests aus
    success = run_tests()
    
    if success:
        print("\n✅ Alle Tests wurden erfolgreich ausgeführt.")
        sys.exit(0)
    else:
        print("\n❌ Einige Tests sind fehlgeschlagen.")
        sys.exit(1)
