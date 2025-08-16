#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test-Skript für die Verschlüsselung eines einzelnen VXOR AI-Moduls
"""

import os
import sys
import glob
import subprocess
from datetime import datetime

def encrypt_test_module():
    """Verschlüsselt ein einzelnes Modul zum Testen der PyArmor-Konfiguration."""
    print(f"=== Test-Verschlüsselung gestartet: {datetime.now()} ===")
    
    # Verzeichnispfad für das Test-Modul
    test_module = "miso/security/vxor_blackbox/crypto"
    vxor_root = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(vxor_root, test_module)
    
    if not os.path.exists(module_path):
        print(f"FEHLER: Test-Modul {test_module} existiert nicht.")
        return False
    
    # Ausgabeverzeichnis für verschlüsselte Dateien
    output_dir = os.path.join(vxor_root, "test_dist", test_module)
    os.makedirs(output_dir, exist_ok=True)
    
    # Finde alle Python-Dateien im Modul
    py_files = glob.glob(os.path.join(module_path, "*.py"))
    if not py_files:
        print(f"FEHLER: Keine Python-Dateien in {module_path} gefunden.")
        return False
    
    print(f"Gefundene Python-Dateien: {len(py_files)}")
    
    # PyArmor-Befehl für die Verschlüsselung (PyArmor 8+ Syntax)
    # Wir verwenden das Verzeichnis als Input und nicht einzelne Dateien
    cmd = [
        'pyarmor', 'gen',
        '--restrict',        # Aktiviere Restrict-Modus
        '--obf-code', '2',  # Höchste Code-Obfuskationsstufe
        '--mix-str',        # String-Konstanten schützen
        '-r',               # Rekursiver Modus
        '-O', output_dir,   # Ausgabeverzeichnis
        module_path         # Eingabeverzeichnis (keine Wildcards mehr)
    ]
    
    print(f"Ausführung des Befehls: {' '.join(cmd)}")
    
    try:
        # Befehl ausführen
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Ergebnis überprüfen
        if result.returncode == 0:
            print("Test-Verschlüsselung erfolgreich!")
            print(f"Verschlüsselte Dateien befinden sich in: {output_dir}")
            if result.stdout:
                print(f"Ausgabe: {result.stdout}")
            return True
        else:
            print(f"FEHLER bei der Test-Verschlüsselung. Exit-Code: {result.returncode}")
            if result.stderr:
                print(f"Fehlermeldung: {result.stderr}")
            return False
    except Exception as e:
        print(f"AUSNAHME bei der Test-Verschlüsselung: {str(e)}")
        return False
    finally:
        print(f"=== Test-Verschlüsselung beendet: {datetime.now()} ===")

if __name__ == "__main__":
    success = encrypt_test_module()
    sys.exit(0 if success else 1)
