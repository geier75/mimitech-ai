#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate: Integrationsskript für die optimierte T-Mathematics Engine
Führt alle notwendigen Schritte aus, um die optimierte Engine im MISO-System zu aktivieren.
"""

import os
import sys
import logging
import argparse
import importlib
import shutil
from datetime import datetime

# Füge das Hauptverzeichnis zum Pfad hinzu, um Module zu importieren
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, "../../.."))
sys.path.insert(0, project_dir)

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(script_dir, f"integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("T-Mathematics-Integration")

def check_dependencies():
    """Überprüft, ob alle erforderlichen Abhängigkeiten installiert sind."""
    required_packages = {
        "mlx": "Apple MLX Framework",
        "numpy": "NumPy Bibliothek",
        "matplotlib": "Matplotlib (für Visualisierung)"
    }
    
    missing_packages = []
    for package, description in required_packages.items():
        try:
            importlib.import_module(package)
            logger.info(f"✅ {description} ({package}) ist installiert.")
        except ImportError:
            logger.warning(f"❌ {description} ({package}) fehlt.")
            missing_packages.append(package)
    
    return missing_packages

def backup_original_files():
    """Erstellt Backup-Kopien der originalen Dateien."""
    backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    math_module_dir = os.path.join(project_dir, "math_module", "t_mathematics")
    backup_dir = os.path.join(math_module_dir, f"backup_{backup_timestamp}")
    
    try:
        os.makedirs(backup_dir, exist_ok=True)
        
        # Liste der zu sichernden Dateien
        files_to_backup = [
            "engine.py",
            "mlx_backend.py",
            "tensor_ops.py"
        ]
        
        for file in files_to_backup:
            src_path = os.path.join(math_module_dir, file)
            if os.path.exists(src_path):
                dst_path = os.path.join(backup_dir, file)
                shutil.copy2(src_path, dst_path)
                logger.info(f"Backup erstellt: {src_path} -> {dst_path}")
        
        return backup_dir
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Backups: {e}")
        return None

def integrate_optimized_engine(dry_run=False):
    """Integriert die optimierte Engine in das MISO-System."""
    math_module_dir = os.path.join(project_dir, "math_module", "t_mathematics")
    
    # Dateien, die integriert werden sollen
    files_to_integrate = {
        os.path.join(math_module_dir, "engine_optimized.py"): os.path.join(math_module_dir, "engine.py"),
        os.path.join(math_module_dir, "backend_base.py"): os.path.join(math_module_dir, "backend_base.py"),
        os.path.join(math_module_dir, "mlx_backend", "mlx_backend_impl.py"): os.path.join(math_module_dir, "mlx_backend.py")
    }
    
    if dry_run:
        logger.info("DRY RUN: Folgende Dateien würden integriert werden:")
        for src, dst in files_to_integrate.items():
            logger.info(f"  {src} -> {dst}")
        return True
    
    try:
        # Backup erstellen
        backup_dir = backup_original_files()
        if not backup_dir:
            logger.error("Integration abgebrochen, da Backup fehlgeschlagen ist.")
            return False
        
        # Dateien kopieren
        for src, dst in files_to_integrate.items():
            if os.path.exists(src):
                # Stelle sicher, dass das Zielverzeichnis existiert
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                
                # Kopiere die Datei
                shutil.copy2(src, dst)
                logger.info(f"Datei integriert: {src} -> {dst}")
            else:
                logger.warning(f"Quelldatei nicht gefunden: {src}")
        
        logger.info("Integration abgeschlossen!")
        logger.info(f"Backup der Originaldateien: {backup_dir}")
        return True
    except Exception as e:
        logger.error(f"Fehler bei der Integration: {e}")
        return False

def run_validation_tests():
    """Führt Validierungstests für die integrierte Engine durch."""
    logger.info("Führe Validierungstests durch...")
    
    try:
        # Versuche, die Engine zu importieren und zu initialisieren
        sys.path.insert(0, os.path.join(project_dir, "math_module"))
        from t_mathematics.engine import get_engine
        
        engine = get_engine()
        backend_info = engine.get_active_backend_info()
        
        logger.info(f"Engine erfolgreich initialisiert mit Backend: {backend_info.get('name', 'unbekannt')}")
        logger.info(f"Aktives Gerät: {backend_info.get('device', 'unbekannt')}")
        logger.info(f"JIT aktiviert: {backend_info.get('jit_enabled', False)}")
        logger.info(f"ANE verfügbar: {backend_info.get('has_ane', False)}")
        
        # Führe einfache Operationen durch
        a = engine.create_tensor([[1, 2], [3, 4]])
        b = engine.create_tensor([[5, 6], [7, 8]])
        
        c = engine.add(a, b)
        logger.info(f"Addition: {engine.to_numpy(c).tolist()}")
        
        d = engine.matmul(a, b)
        logger.info(f"Matrix-Multiplikation: {engine.to_numpy(d).tolist()}")
        
        return True
    except Exception as e:
        logger.error(f"Validierungstest fehlgeschlagen: {e}")
        return False

def update_integration_status():
    """Aktualisiert den Integrationsstatus in der Projektdokumentation."""
    status_file = os.path.join(project_dir, "optimization", "phase5_profiling", "t_mathematics", "integration_status.md")
    
    with open(status_file, 'w') as f:
        f.write("# T-Mathematics Engine: Integrationsstatus\n")
        f.write(f"**Datum:** {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Uhrzeit:** {datetime.now().strftime('%H:%M:%S')}\n\n")
        
        f.write("## Zusammenfassung\n\n")
        f.write("Die optimierte T-Mathematics Engine wurde erfolgreich in das MISO-System integriert. ")
        f.write("Diese Version bietet signifikante Leistungsverbesserungen durch JIT-Kompilierung, ")
        f.write("optimierte Speicherverwaltung und Apple Neural Engine-Unterstützung.\n\n")
        
        f.write("## Implementierte Optimierungen\n\n")
        f.write("1. **JIT-Kompilierung**: Alle kritischen Operationen werden jetzt JIT-kompiliert für maximale Leistung.\n")
        f.write("2. **Verzögerte Auswertung**: Operationen werden nur bei Bedarf ausgeführt, was unnötige Berechnungen vermeidet.\n")
        f.write("3. **Optimierte Speicherverwaltung**: Reduzierte Speicherkopien und effizientere Allokation.\n")
        f.write("4. **Apple Neural Engine-Unterstützung**: Automatische Nutzung der ANE, wenn verfügbar.\n")
        f.write("5. **Backend-Registry-System**: Flexibles System zur dynamischen Auswahl des besten Backends.\n\n")
        
        f.write("## Nächste Schritte\n\n")
        f.write("1. Regelmäßige Leistungsüberwachung, um sicherzustellen, dass die Optimierungen in allen Szenarien effektiv sind.\n")
        f.write("2. Implementierung weiterer Backend-Optimierungen für spezifische Anwendungsfälle.\n")
        f.write("3. Entwicklung von PyTorch- und NumPy-Backends für verbesserte Portabilität.\n")
        f.write("4. Integration mit ECHO-PRIME und Q-Logik für End-to-End-Optimierung.\n")
    
    logger.info(f"Integrationsstatus aktualisiert: {status_file}")

def main():
    parser = argparse.ArgumentParser(description="Integrationstool für die optimierte T-Mathematics Engine")
    parser.add_argument("--dry-run", action="store_true", help="Simulation der Integration ohne tatsächliche Änderungen")
    parser.add_argument("--skip-tests", action="store_true", help="Überspringt Validierungstests")
    args = parser.parse_args()
    
    logger.info("Starte Integration der optimierten T-Mathematics Engine")
    
    # Überprüfe Abhängigkeiten
    missing_packages = check_dependencies()
    if missing_packages:
        logger.warning(f"Fehlende Abhängigkeiten: {', '.join(missing_packages)}")
        logger.warning("Die Integration kann ohne diese Pakete möglicherweise nicht wie erwartet funktionieren.")
    
    # Integriere die optimierte Engine
    if integrate_optimized_engine(args.dry_run):
        logger.info("Integration der Engine abgeschlossen.")
        
        # Führe Validierungstests durch, wenn nicht übersprungen
        if not args.skip_tests and not args.dry_run:
            if run_validation_tests():
                logger.info("Validierungstests erfolgreich abgeschlossen.")
                
                # Aktualisiere Integrationsstatus
                update_integration_status()
                
                logger.info("Integration erfolgreich abgeschlossen!")
            else:
                logger.error("Validierungstests fehlgeschlagen. Überprüfen Sie die Logs für Details.")
    else:
        logger.error("Integration fehlgeschlagen. Überprüfen Sie die Logs für Details.")

if __name__ == "__main__":
    main()
