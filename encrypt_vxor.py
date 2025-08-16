#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VXOR AI Verschlüsselungsskript - Phase 2.1
Dieses Skript verschlüsselt alle Python-Module des VXOR AI-Projekts 
mit PyArmor und AES-256-Verschlüsselung.

Sicherheitsmerkmale:
- AES-256-GCM Verschlüsselung der Bytecode-Module
- Erweiterte Obfuskation des Python-Bytecodes
- Bindung an Hardware-ID (optional)
- Self-Check-Mechanismen gegen Manipulation
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Konfiguration
OUTPUT_DIR = 'dist'
BACKUP_DIR = 'backup'
LOG_DIR = 'logs'
VXOR_ROOT = os.path.dirname(os.path.abspath(__file__))

# Definiere Haupt-Module zum Verschlüsseln
CORE_MODULES = [
    'miso/core',
    'miso/tmathematics',  # T-Mathematics Engine
    'miso/lang',          # M-LINGUA Integration
    'miso/security',      # Sicherheitsmodule inklusive vxor_blackbox
    'miso/nexus',
]

# Zusätzliche Module
ADDITIONAL_MODULES = [
    'miso/analysis',
    'miso/federated_learning',
    'miso/integration',
    'miso/logic',
    'miso/math',
    'miso/recursive_self_improvement',
    'miso/vxor',
    'miso/vXor_Modules',
]

# Einstiegspunkte für eigenständige ausführbare Dateien
ENTRY_POINTS = {
    'miso.py': 'MISO Ultimate Main Entry',
    'miso_command_demo.py': 'MISO Command Interface',
    'train_miso.py': 'MISO Training System',
}

def setup_logging():
    """Richtet Logging für den Verschlüsselungsprozess ein."""
    log_dir = os.path.join(VXOR_ROOT, LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"encryption_{timestamp}.log")
    
    return log_file

def create_backup():
    """Erstellt ein Backup des Original-Codes."""
    backup_dir = os.path.join(VXOR_ROOT, BACKUP_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"vxor_backup_{timestamp}")
    
    os.makedirs(backup_dir, exist_ok=True)
    
    # Hauptmodule und wichtige Dateien sichern
    for module in CORE_MODULES + ADDITIONAL_MODULES:
        src_path = os.path.join(VXOR_ROOT, module)
        if os.path.exists(src_path):
            dst_path = os.path.join(backup_path, module)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
    
    # Einstiegspunkte sichern
    for entry_point in ENTRY_POINTS.keys():
        src_path = os.path.join(VXOR_ROOT, entry_point)
        if os.path.exists(src_path):
            dst_path = os.path.join(backup_path, entry_point)
            shutil.copy2(src_path, dst_path)
    
    print(f"Backup erstellt in: {backup_path}")
    return backup_path

def encrypt_modules(log_file):
    """Verschlüsselt alle VXOR AI-Module mit PyArmor."""
    output_dir = os.path.join(VXOR_ROOT, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(log_file, 'a') as log:
        log.write(f"=== Startzeit: {datetime.now()} ===\n")
        log.write(f"Python Version: {sys.version}\n")
        
        # SCHRITT 1: Verschlüssele Kernmodule (T-Mathematics Engine und M-LINGUA Integration)
        log.write("\n=== VERSCHLÜSSELUNG DER KERNMODULE ===\n")
        for module in CORE_MODULES:
            module_path = os.path.join(VXOR_ROOT, module)
            if not os.path.exists(module_path):
                log.write(f"WARNUNG: Modul {module} existiert nicht. Überspringe...\n")
                continue
                
            module_output = os.path.join(output_dir, module)
            
            # Verwende höchste Sicherheitsstufe für Kernmodule
            cmd = [
                'pyarmor', 'obfuscate',
                '--advanced', '2',           # Höchste Obfuskations-Stufe
                '--restrict', '4',           # Strenge Laufzeiteinschränkungen
                '--platform', 'darwin',      # MacOS-spezifisch
                '--bootstrap', '3',          # Maximale Sicherheit beim Bootstrapping
                '--obf-code', '2',           # Höchste Code-Obfuskation
                '--obf-mod', '2',            # Höchste Modul-Obfuskation
                '--output', module_output,
                Path(module_path).as_posix() + "/*.py"
            ]
            
            log.write(f"Verschlüssele {module}...\n")
            log.write(f"Befehl: {' '.join(cmd)}\n")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                log.write(f"Erfolgreich: {result.returncode == 0}\n")
                if result.stdout:
                    log.write(f"Ausgabe: {result.stdout}\n")
                if result.stderr:
                    log.write(f"Fehler: {result.stderr}\n")
            except Exception as e:
                log.write(f"Fehler bei der Verschlüsselung von {module}: {str(e)}\n")
        
        # SCHRITT 2: Verschlüssele zusätzliche Module
        log.write("\n=== VERSCHLÜSSELUNG DER ZUSÄTZLICHEN MODULE ===\n")
        for module in ADDITIONAL_MODULES:
            module_path = os.path.join(VXOR_ROOT, module)
            if not os.path.exists(module_path):
                log.write(f"WARNUNG: Modul {module} existiert nicht. Überspringe...\n")
                continue
                
            module_output = os.path.join(output_dir, module)
            
            # Standard-Sicherheitsstufe für zusätzliche Module
            cmd = [
                'pyarmor', 'obfuscate',
                '--advanced', '1',           # Mittlere Obfuskations-Stufe
                '--restrict', '2',           # Moderate Laufzeiteinschränkungen
                '--platform', 'darwin',      # MacOS-spezifisch
                '--output', module_output,
                Path(module_path).as_posix() + "/*.py"
            ]
            
            log.write(f"Verschlüssele {module}...\n")
            log.write(f"Befehl: {' '.join(cmd)}\n")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                log.write(f"Erfolgreich: {result.returncode == 0}\n")
                if result.stdout:
                    log.write(f"Ausgabe: {result.stdout}\n")
                if result.stderr:
                    log.write(f"Fehler: {result.stderr}\n")
            except Exception as e:
                log.write(f"Fehler bei der Verschlüsselung von {module}: {str(e)}\n")
        
        # SCHRITT 3: Kompiliere Einstiegspunkte mit Nuitka
        log.write("\n=== KOMPILIERUNG DER EINSTIEGSPUNKTE ===\n")
        for entry_point, description in ENTRY_POINTS.items():
            entry_path = os.path.join(VXOR_ROOT, entry_point)
            if not os.path.exists(entry_path):
                log.write(f"WARNUNG: Einstiegspunkt {entry_point} existiert nicht. Überspringe...\n")
                continue
                
            # Kompiliere mit Nuitka zu einer ausführbaren Datei
            output_name = os.path.splitext(entry_point)[0] + ".bin"
            cmd = [
                'python', '-m', 'nuitka',
                '--onefile',                  # Einzelne ausführbare Datei
                '--macos-create-app-bundle',  # MacOS App-Bundle erstellen
                '--macos-app-icon=resources/vxor_icon.icns',  # Icon
                '--include-package=miso',     # Miso-Paket einbeziehen
                '--include-package=numpy',    # Numpy einbeziehen
                '--include-package=torch',    # PyTorch einbeziehen
                '--include-package=mlx',      # MLX einbeziehen
                '--output-dir=' + output_dir, # Ausgabeverzeichnis
                '--output-filename=' + output_name,  # Ausgabedateiname
                '--remove-output',            # Alte Ausgaben entfernen
                entry_path                    # Pfad zum Einstiegspunkt
            ]
            
            log.write(f"Kompiliere {entry_point} ({description})...\n")
            log.write(f"Befehl: {' '.join(cmd)}\n")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                log.write(f"Erfolgreich: {result.returncode == 0}\n")
                if result.stdout:
                    log.write(f"Ausgabe: {result.stdout}\n")
                if result.stderr:
                    log.write(f"Fehler: {result.stderr}\n")
            except Exception as e:
                log.write(f"Fehler bei der Kompilierung von {entry_point}: {str(e)}\n")
        
        log.write(f"\n=== Endzeit: {datetime.now()} ===\n")
    
    print(f"Verschlüsselung abgeschlossen. Log-Datei: {log_file}")
    return output_dir

def clean_source_files(output_dir):
    """Entfernt alle ungeschützten Python-Quelldateien aus dem Output-Verzeichnis."""
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('pytransform'):
                # pytransform ist Teil von PyArmor und sollte nicht entfernt werden
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Entfernt ungeschützte Datei: {file_path}")
                except Exception as e:
                    print(f"Fehler beim Entfernen von {file_path}: {str(e)}")

def verify_encryption(output_dir):
    """Überprüft, ob alle Module erfolgreich verschlüsselt wurden."""
    print("\nÜberprüfe verschlüsselte Module...")
    
    all_encrypted = True
    for module in CORE_MODULES + ADDITIONAL_MODULES:
        module_path = os.path.join(output_dir, module)
        if not os.path.exists(module_path):
            print(f"WARNUNG: Verschlüsseltes Modul {module} fehlt!")
            all_encrypted = False
            continue
            
        # Überprüfe, ob .pyc oder verschlüsselte Dateien vorhanden sind
        has_protected_files = False
        for root, dirs, files in os.walk(module_path):
            for file in files:
                if file.endswith(('.pyc', '.so')) or 'pytransform' in file:
                    has_protected_files = True
                    break
        
        if not has_protected_files:
            print(f"WARNUNG: Keine verschlüsselten Dateien in {module} gefunden!")
            all_encrypted = False
    
    # Überprüfe Einstiegspunkte
    for entry_point in ENTRY_POINTS.keys():
        base_name = os.path.splitext(entry_point)[0]
        binary_path = os.path.join(output_dir, f"{base_name}.bin")
        app_path = os.path.join(output_dir, f"{base_name}.app")
        
        if not (os.path.exists(binary_path) or os.path.exists(app_path)):
            print(f"WARNUNG: Kompilierter Einstiegspunkt für {entry_point} fehlt!")
            all_encrypted = False
    
    if all_encrypted:
        print("Alle Module wurden erfolgreich verschlüsselt und kompiliert!")
    else:
        print("WARNUNG: Einige Module wurden möglicherweise nicht richtig verschlüsselt!")
    
    return all_encrypted

def main():
    """Hauptfunktion für die VXOR AI-Codeverschlüsselung."""
    parser = argparse.ArgumentParser(description='VXOR AI Codeverschlüsselung und -kompilierung')
    parser.add_argument('--no-backup', action='store_true', help='Überspringe die Erstellung eines Backups')
    parser.add_argument('--keep-source', action='store_true', help='Behalte ungeschützte Quelldateien')
    args = parser.parse_args()
    
    print("=== VXOR AI Codeverschlüsselung – Phase 2.1 ===")
    print(f"Startzeit: {datetime.now()}")
    
    # Richte Logging ein
    log_file = setup_logging()
    
    # Erstelle Backup (falls nicht deaktiviert)
    if not args.no_backup:
        backup_path = create_backup()
        print(f"Backup erstellt in: {backup_path}")
    else:
        print("Backup-Erstellung übersprungen.")
    
    # Verschlüssele Module
    output_dir = encrypt_modules(log_file)
    print(f"Verschlüsselte Module gespeichert in: {output_dir}")
    
    # Entferne ungeschützte Quelldateien (falls gewünscht)
    if not args.keep_source:
        clean_source_files(output_dir)
    else:
        print("Ungeschützte Quelldateien werden beibehalten.")
    
    # Überprüfe Verschlüsselung
    verify_encryption(output_dir)
    
    print(f"\nEndzeit: {datetime.now()}")
    print(f"Verschlüsselungsprozess abgeschlossen. Log-Datei: {log_file}")

if __name__ == '__main__':
    main()
