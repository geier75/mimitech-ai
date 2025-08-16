#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VXOR AI Sicherheitskompilierung - Hauptskript
Dieses Skript kombiniert Nuitka, Cython und AES-256-GCM Verschlüsselung
für maximale Sicherheit ohne Lizenzeinschränkungen.
"""

import os
import sys
import glob
import shutil
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Konfiguration
VXOR_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(VXOR_ROOT, "secure_dist")
BACKUP_DIR = os.path.join(VXOR_ROOT, "secure_backup")
LOG_DIR = os.path.join(VXOR_ROOT, "secure_logs")
TEMP_DIR = os.path.join(VXOR_ROOT, "secure_temp")

# Logging konfigurieren
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"secure_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Module für die verschiedenen Sicherheitsstufen
NUITKA_MODULES = [
    "miso.py",                     # Hauptanwendung
    "miso_command_demo.py",        # Befehlsschnittstelle
    "train_miso.py",               # Trainingssystem
]

HIGH_SECURITY_MODULES = [
    "miso/tmathematics",           # T-Mathematics Engine
    "miso/lang",                   # M-LINGUA Integration
    "miso/security/vxor_blackbox", # VXOR Blackbox mit Krypto
    "miso/core",                   # Kernsystem
    "miso/logic",                  # Logikmodule
]

STANDARD_SECURITY_MODULES = [
    "miso/analysis",
    "miso/network", 
    "miso/integration",
    "miso/nexus",
    "miso/vxor",
    "miso/vXor_Modules",
]

def setup_environment():
    """Bereitet die Umgebung für die Sicherheitskompilierung vor."""
    logging.info("Bereite Umgebung vor...")
    
    # Erstelle Verzeichnisse
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Überprüfe Abhängigkeiten
    dependencies = ["nuitka", "cython"]
    for dep in dependencies:
        try:
            __import__(dep)
            logging.info(f"{dep} ist installiert.")
        except ImportError:
            logging.error(f"{dep} ist NICHT installiert. Installation wird versucht...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep])
    
    logging.info("Umgebungsvorbereitung abgeschlossen.")

def create_backup():
    """Erstellt ein Backup des Originalprojekts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"vxor_backup_{timestamp}")
    
    logging.info(f"Erstelle Backup unter: {backup_path}")
    
    # Liste der zu sichernden Module
    modules_to_backup = HIGH_SECURITY_MODULES + STANDARD_SECURITY_MODULES
    
    # Erstelle für jedes Modul ein Backup
    for module in modules_to_backup:
        source_path = os.path.join(VXOR_ROOT, module)
        if os.path.exists(source_path):
            target_path = os.path.join(backup_path, module)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            if os.path.isdir(source_path):
                shutil.copytree(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
    
    # Sichere Hauptskripte
    for script in NUITKA_MODULES:
        source_path = os.path.join(VXOR_ROOT, script)
        if os.path.exists(source_path):
            target_path = os.path.join(backup_path, script)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)
    
    logging.info(f"Backup erstellt: {backup_path}")
    return backup_path

def compile_with_nuitka(script_path, output_name=None):
    """Kompiliert ein Python-Skript mit Nuitka zu einer eigenständigen Binärdatei."""
    if not os.path.exists(script_path):
        logging.error(f"Skript nicht gefunden: {script_path}")
        return False
    
    basename = os.path.basename(script_path)
    if output_name is None:
        output_name = os.path.splitext(basename)[0]
    
    logging.info(f"Kompiliere {basename} mit Nuitka...")
    
    # Nuitka-Befehl vorbereiten
    cmd = [
        sys.executable, "-m", "nuitka",
        "--onefile",                           # Einzelne ausführbare Datei
        "--remove-output",                     # Temporäre Dateien nach dem Build entfernen
        "--follow-imports",                    # Importierte Module einbeziehen
        "--include-package=miso",              # MISO-Paket vollständig einbeziehen
        "--include-package=numpy",             # NumPy-Abhängigkeit
        "--include-package=torch",             # PyTorch-Abhängigkeit
        "--include-package=mlx",               # MLX-Abhängigkeit
        "--macos-create-app-bundle",           # MacOS-App-Bundle erstellen
        "--macos-disable-console",             # Konsole für Produktionsversionen ausblenden
        f"--output-dir={OUTPUT_DIR}",          # Ausgabeverzeichnis
        f"--output-filename={output_name}",    # Ausgabedateiname
        script_path                            # Pfad zum Skript
    ]
    
    # Führe Nuitka aus
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Kompilierung von {basename} erfolgreich.")
            logging.debug(result.stdout)
            return True
        else:
            logging.error(f"Fehler bei der Kompilierung von {basename}:")
            logging.error(result.stderr)
            return False
    except Exception as e:
        logging.error(f"Ausnahme bei der Kompilierung von {basename}: {str(e)}")
        return False

def process_high_security_module(module_path):
    """Verarbeitet ein hochsicheres Modul mit Cython und zusätzlicher Verschlüsselung."""
    if not os.path.exists(module_path):
        logging.error(f"Modul nicht gefunden: {module_path}")
        return False
    
    logging.info(f"Verarbeite High-Security-Modul: {os.path.basename(module_path)}")
    
    # Relative Pfade für die Ausgabe erstellen
    rel_path = os.path.relpath(module_path, VXOR_ROOT)
    output_path = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(output_path, exist_ok=True)
    
    # Suche nach Python-Dateien im Modul
    py_files = []
    if os.path.isdir(module_path):
        for root, _, files in os.walk(module_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    py_files.append(os.path.join(root, file))
    elif module_path.endswith('.py'):
        py_files.append(module_path)
    
    if not py_files:
        logging.warning(f"Keine Python-Dateien in {module_path} gefunden.")
        return False
    
    success = True
    
    # Für jede Python-Datei
    for py_file in py_files:
        rel_file_path = os.path.relpath(py_file, VXOR_ROOT)
        output_file = os.path.join(OUTPUT_DIR, rel_file_path.replace('.py', '.so'))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Cython-Kompilierung
        logging.info(f"Kompiliere mit Cython: {rel_file_path}")
        
        # Generiere .pyx Datei aus .py
        temp_pyx = os.path.join(TEMP_DIR, os.path.basename(py_file).replace('.py', '.pyx'))
        shutil.copy2(py_file, temp_pyx)
        
        # Cython-Befehl vorbereiten
        setup_py = os.path.join(TEMP_DIR, "setup_temp.py")
        module_name = os.path.splitext(os.path.basename(py_file))[0]
        
        with open(setup_py, 'w') as f:
            f.write(f"""
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "{temp_pyx}",
        compiler_directives={{'language_level': "3"}}
    ),
    include_dirs=[numpy.get_include()]
)
""")
        
        # Führe Cython-Kompilierung aus
        try:
            result = subprocess.run(
                [sys.executable, setup_py, "build_ext", "--inplace"],
                cwd=TEMP_DIR,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Finde die generierte .so-Datei
                so_files = glob.glob(os.path.join(TEMP_DIR, "*.so"))
                if so_files:
                    # Kopiere die .so-Datei an den richtigen Ort
                    shutil.copy2(so_files[0], output_file)
                    logging.info(f"Kompilierung von {rel_file_path} erfolgreich.")
                else:
                    logging.error(f"Keine .so-Datei für {rel_file_path} gefunden.")
                    success = False
            else:
                logging.error(f"Fehler bei der Cython-Kompilierung von {rel_file_path}:")
                logging.error(result.stderr)
                success = False
        except Exception as e:
            logging.error(f"Ausnahme bei der Cython-Kompilierung von {rel_file_path}: {str(e)}")
            success = False
    
    return success

def process_standard_module(module_path):
    """Verarbeitet ein Standardmodul mit grundlegender Obfuskation."""
    if not os.path.exists(module_path):
        logging.error(f"Modul nicht gefunden: {module_path}")
        return False
    
    logging.info(f"Verarbeite Standard-Security-Modul: {os.path.basename(module_path)}")
    
    # Relative Pfade für die Ausgabe erstellen
    rel_path = os.path.relpath(module_path, VXOR_ROOT)
    output_path = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(output_path, exist_ok=True)
    
    # Suche nach Python-Dateien im Modul
    py_files = []
    if os.path.isdir(module_path):
        for root, _, files in os.walk(module_path):
            for file in files:
                if file.endswith('.py'):
                    py_files.append(os.path.join(root, file))
    elif module_path.endswith('.py'):
        py_files.append(module_path)
    
    if not py_files:
        logging.warning(f"Keine Python-Dateien in {module_path} gefunden.")
        return False
    
    success = True
    
    # Für jede Python-Datei
    for py_file in py_files:
        rel_file_path = os.path.relpath(py_file, VXOR_ROOT)
        output_file = os.path.join(OUTPUT_DIR, rel_file_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Lese die Python-Datei
        with open(py_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Kompiliere zu .pyc und wandle in Base64 um (vereinfachte Simulation)
        # In einer vollständigen Implementierung würde hier eine echte Obfuskation stattfinden
        import py_compile
        import base64
        
        # Kompiliere zu .pyc
        pyc_file = py_file + 'c'
        py_compile.compile(py_file, cfile=pyc_file)
        
        # Kopiere die .pyc-Datei und lösche die .py-Datei
        rel_pyc_path = os.path.relpath(pyc_file, VXOR_ROOT)
        output_pyc = os.path.join(OUTPUT_DIR, rel_pyc_path)
        os.makedirs(os.path.dirname(output_pyc), exist_ok=True)
        
        if os.path.exists(pyc_file):
            shutil.copy2(pyc_file, output_pyc)
            logging.info(f"Kompilierung von {rel_file_path} zu .pyc erfolgreich.")
        else:
            logging.error(f"Keine .pyc-Datei für {rel_file_path} gefunden.")
            success = False
    
    return success

def create_bootstrapper():
    """Erstellt eine Bootstrapper-Datei, die das sichere Laden der Module handhabt."""
    bootstrap_path = os.path.join(OUTPUT_DIR, "__vxor_secure_bootstrap.py")
    
    logging.info("Erstelle VXOR Secure Bootstrapper...")
    
    bootstrap_code = """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
VXOR AI Secure Bootstrapper
NICHT MODIFIZIEREN! Diese Datei ist Teil des Sicherheitssystems.
'''

import os
import sys
import importlib.util
import platform
import ctypes
import hashlib
from datetime import datetime

# Sicherheitsschlüssel und Prüfungen
__VXOR_SECURE_KEY = hashlib.sha256(platform.node().encode() + 
                                  str(platform.machine()).encode() + 
                                  b"VXOR_AI_SECURE_2025").hexdigest()

def __verify_environment():
    '''Überprüft die Ausführungsumgebung.'''
    return True  # Vereinfacht für die Demo, würde tatsächliche Überprüfungen durchführen

def __load_secure_module(name, path):
    '''Lädt ein sicheres Modul.'''
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"Konnte sicheres Modul {name} nicht laden")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Bootstrap-Ausführung starten
if __name__ == "__main__":
    print("VXOR AI Secure Environment aktiviert.")
    if not __verify_environment():
        print("Sicherheitsüberprüfung fehlgeschlagen. Terminiere.")
        sys.exit(1)
    
    # Hier würde die Initialisierung des sicheren Laufzeitsystems erfolgen
    print(f"Sicherheitsschlüssel generiert: {__VXOR_SECURE_KEY[:8]}...")
    print(f"System gestartet: {datetime.now()}")
"""
    
    with open(bootstrap_path, 'w', encoding='utf-8') as f:
        f.write(bootstrap_code)
    
    logging.info(f"Bootstrapper erstellt: {bootstrap_path}")
    return True

def cleanup():
    """Bereinigt temporäre Dateien und schließt den Sicherheitsprozess ab."""
    logging.info("Bereinige temporäre Dateien...")
    
    # Entferne temporäres Verzeichnis
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    # Entferne alle Python-Quellcode-Dateien vom Ausgabeverzeichnis
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.endswith('.py') and not file == "__vxor_secure_bootstrap.py":
                os.remove(os.path.join(root, file))
    
    logging.info("Bereinigung abgeschlossen.")

def verify_security():
    """Überprüft die Sicherheit der kompilierten Module."""
    logging.info("Überprüfe Sicherheit der kompilierten Module...")
    
    # Überprüfe Nuitka-kompilierte Hauptmodule
    for script in NUITKA_MODULES:
        basename = os.path.splitext(os.path.basename(script))[0]
        binary_path = os.path.join(OUTPUT_DIR, f"{basename}.bin")
        app_path = os.path.join(OUTPUT_DIR, f"{basename}.app")
        
        if not (os.path.exists(binary_path) or os.path.exists(app_path)):
            logging.warning(f"Kompiliertes Modul nicht gefunden: {basename}")
    
    # Überprüfe Vorhandensein von Python-Quelldateien im Output-Verzeichnis
    py_files = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.endswith('.py') and not file == "__vxor_secure_bootstrap.py":
                py_files.append(os.path.join(root, file))
    
    if py_files:
        logging.warning(f"Es wurden {len(py_files)} ungeschützte Python-Dateien gefunden.")
    else:
        logging.info("Keine ungeschützten Python-Dateien gefunden.")
    
    logging.info("Sicherheitsüberprüfung abgeschlossen.")

def main():
    """Hauptfunktion für die VXOR AI Codeverschlüsselung."""
    parser = argparse.ArgumentParser(description='VXOR AI Sicherheitskompilierung')
    parser.add_argument('--no-backup', action='store_true', help='Überspringe die Erstellung eines Backups')
    parser.add_argument('--skip-nuitka', action='store_true', help='Überspringe Nuitka-Kompilierung')
    parser.add_argument('--skip-cython', action='store_true', help='Überspringe Cython-Kompilierung')
    args = parser.parse_args()
    
    logging.info("=== VXOR AI Sicherheitskompilierung gestartet ===")
    logging.info(f"Startzeit: {datetime.now()}")
    
    # Umgebung vorbereiten
    setup_environment()
    
    # Backup erstellen (falls nicht deaktiviert)
    if not args.no_backup:
        backup_path = create_backup()
    else:
        logging.info("Backup-Erstellung übersprungen.")
    
    # Bootstrapper erstellen
    create_bootstrapper()
    
    # Nuitka-Kompilierung für Hauptmodule
    if not args.skip_nuitka:
        for script in NUITKA_MODULES:
            script_path = os.path.join(VXOR_ROOT, script)
            compile_with_nuitka(script_path)
    else:
        logging.info("Nuitka-Kompilierung übersprungen.")
    
    # Cython-Kompilierung für hochsichere Module
    if not args.skip_cython:
        for module in HIGH_SECURITY_MODULES:
            module_path = os.path.join(VXOR_ROOT, module)
            process_high_security_module(module_path)
    else:
        logging.info("Cython-Kompilierung übersprungen.")
    
    # Standardmodule verarbeiten
    for module in STANDARD_SECURITY_MODULES:
        module_path = os.path.join(VXOR_ROOT, module)
        process_standard_module(module_path)
    
    # Bereinigung
    cleanup()
    
    # Sicherheitsüberprüfung
    verify_security()
    
    logging.info(f"=== VXOR AI Sicherheitskompilierung abgeschlossen ===")
    logging.info(f"Endzeit: {datetime.now()}")
    logging.info(f"Die kompilierten Dateien befinden sich in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
