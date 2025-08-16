#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Stabilitätstests Ausführungsskript

Dieses Skript führt alle Stabilitätstests für das MISO Ultimate AGI-System aus.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import datetime
import subprocess
import argparse
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stability_tests_runner.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("miso_stability_tests_runner")

def run_test(test_script, output_dir):
    """
    Führt einen Test aus
    
    Args:
        test_script: Pfad zum Testskript
        output_dir: Ausgabeverzeichnis
        
    Returns:
        Exitcode des Tests
    """
    logger.info(f"Führe Test aus: {test_script}")
    
    # Erstelle Ausgabeverzeichnis
    os.makedirs(output_dir, exist_ok=True)
    
    # Führe Test aus
    start_time = time.time()
    
    try:
        # Führe Test aus
        result = subprocess.run(
            [sys.executable, test_script],
            cwd=output_dir,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Speichere Ausgabe
        with open(os.path.join(output_dir, "stdout.log"), "w", encoding="utf-8") as f:
            f.write(result.stdout)
        
        with open(os.path.join(output_dir, "stderr.log"), "w", encoding="utf-8") as f:
            f.write(result.stderr)
        
        exitcode = result.returncode
    except subprocess.CalledProcessError as e:
        # Speichere Ausgabe bei Fehler
        with open(os.path.join(output_dir, "stdout.log"), "w", encoding="utf-8") as f:
            f.write(e.stdout)
        
        with open(os.path.join(output_dir, "stderr.log"), "w", encoding="utf-8") as f:
            f.write(e.stderr)
        
        exitcode = e.returncode
    except Exception as e:
        # Speichere Fehler
        with open(os.path.join(output_dir, "error.log"), "w", encoding="utf-8") as f:
            f.write(str(e))
        
        exitcode = 1
    
    end_time = time.time()
    
    # Speichere Metadaten
    metadata = {
        "test_script": test_script,
        "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.datetime.fromtimestamp(end_time).isoformat(),
        "duration": end_time - start_time,
        "exitcode": exitcode
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Test abgeschlossen: {test_script} (Exitcode: {exitcode}, Dauer: {end_time - start_time:.2f}s)")
    
    return exitcode

def main():
    """Hauptfunktion"""
    # Parse Argumente
    parser = argparse.ArgumentParser(description="MISO Ultimate Stabilitätstests")
    parser.add_argument("--output-dir", default="test_results", help="Ausgabeverzeichnis")
    parser.add_argument("--skip-base", action="store_true", help="Basistests überspringen")
    parser.add_argument("--skip-integration", action="store_true", help="Integrationstests überspringen")
    parser.add_argument("--skip-load", action="store_true", help="Lasttests überspringen")
    args = parser.parse_args()
    
    # Erstelle Ausgabeverzeichnis
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Aktuelles Verzeichnis
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Liste der Tests
    tests = []
    
    if not args.skip_base:
        tests.append({
            "name": "Basistests",
            "script": os.path.join(current_dir, "stability_tests_base.py"),
            "output_dir": os.path.join(output_dir, "base")
        })
    
    if not args.skip_integration:
        tests.append({
            "name": "Integrationstests",
            "script": os.path.join(current_dir, "stability_tests_integration.py"),
            "output_dir": os.path.join(output_dir, "integration")
        })
    
    if not args.skip_load:
        tests.append({
            "name": "Lasttests",
            "script": os.path.join(current_dir, "stability_tests_load.py"),
            "output_dir": os.path.join(output_dir, "load")
        })
    
    # Führe Tests aus
    results = []
    
    for test in tests:
        logger.info(f"Starte {test['name']}...")
        
        # Führe Test aus
        exitcode = run_test(test["script"], test["output_dir"])
        
        # Speichere Ergebnis
        results.append({
            "name": test["name"],
            "exitcode": exitcode,
            "success": exitcode == 0
        })
    
    # Erstelle Zusammenfassung
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "results": results,
        "overall_success": all(result["success"] for result in results)
    }
    
    # Speichere Zusammenfassung
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Erstelle Markdown-Zusammenfassung
    markdown = f"""
# MISO Ultimate Stabilitätstests - Zusammenfassung

Zeitstempel: {datetime.datetime.now().isoformat()}

## Testergebnisse

| Test | Status | Exitcode |
|------|--------|----------|
"""
    
    for result in results:
        status = "✅ Erfolgreich" if result["success"] else "❌ Fehlgeschlagen"
        markdown += f"| {result['name']} | {status} | {result['exitcode']} |\n"
    
    markdown += f"""
## Gesamtergebnis

Status: {"✅ Erfolgreich" if summary["overall_success"] else "❌ Fehlgeschlagen"}

## Details

Detaillierte Ergebnisse finden Sie in den entsprechenden Unterverzeichnissen:
"""
    
    for test in tests:
        markdown += f"- {test['name']}: `{test['output_dir']}`\n"
    
    # Speichere Markdown-Zusammenfassung
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(markdown)
    
    logger.info(f"Alle Tests abgeschlossen. Gesamtergebnis: {'Erfolgreich' if summary['overall_success'] else 'Fehlgeschlagen'}")
    
    return 0 if summary["overall_success"] else 1

if __name__ == "__main__":
    sys.exit(main())
