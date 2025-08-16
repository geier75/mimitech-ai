#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR Benchmark Dashboard - Starter

Dieses Skript startet das VXOR Benchmark Dashboard und öffnet es in einem neuen Browser-Fenster.
"""

import os
import sys
import time
import logging
import webbrowser
import threading
from pathlib import Path
import subprocess

# Erstelle Logs-Verzeichnis, falls es nicht existiert
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "benchmark_ui_start.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VXOR.Benchmark.Starter")

# Konstanten
DASHBOARD_PORT = 5151
DASHBOARD_URL = f"http://127.0.0.1:{DASHBOARD_PORT}/dashboard/benchmark"


def open_browser():
    """Öffnet das Dashboard in einem neuen Browser-Fenster."""
    # Warte kurz, damit der Server Zeit hat zu starten
    time.sleep(2)
    logger.info(f"Öffne Dashboard im Browser: {DASHBOARD_URL}")
    
    # Öffne in neuem Fenster statt Tab
    try:
        if sys.platform == 'darwin':  # macOS
            chrome_path = 'open -na "Google Chrome" --args --new-window ' + DASHBOARD_URL
            subprocess.Popen(chrome_path, shell=True)
        elif sys.platform == 'win32':  # Windows
            chrome_path = 'start chrome --new-window ' + DASHBOARD_URL
            subprocess.Popen(chrome_path, shell=True)
        else:  # Linux und andere
            webbrowser.open_new(DASHBOARD_URL)
    except Exception as e:
        logger.error(f"Fehler beim Öffnen des Browsers: {e}")
        # Fallback zur Standard-Methode
        webbrowser.open(DASHBOARD_URL)


def start_dashboard():
    """Startet das VXOR Benchmark Dashboard."""
    logger.info("VXOR Benchmark Dashboard wird gestartet...")
    
    # Aktiviere den Testmodus über Umgebungsvariablen
    os.environ["MISO_TEST_MODE"] = "1"
    os.environ["MISO_SECURITY_LEVEL"] = "LOW"
    logger.info("Testmodus aktiviert: MISO_TEST_MODE=1")
    
    # Importiere die Dashboard-Module
    try:
        # Starte Browser in separatem Thread
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Importiere und starte den Dashboard-Server
        from vxor_dashboard import run_server
        
        # Server starten (blockierender Aufruf)
        run_server(DASHBOARD_PORT)
    except ImportError as e:
        logger.error(f"Fehler beim Importieren des Dashboard-Moduls: {e}")
        print(f"FEHLER: Dashboard-Modul konnte nicht geladen werden. "
              f"Überprüfen Sie, ob vxor_dashboard.py vorhanden ist.")
        return 1
    except Exception as e:
        logger.error(f"Fehler beim Starten des Dashboards: {e}")
        print(f"FEHLER: Dashboard konnte nicht gestartet werden: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    print("="*80)
    print("VXOR Benchmark Dashboard wird gestartet...")
    print(f"Das Dashboard wird im Browser unter {DASHBOARD_URL} geöffnet.")
    print("Drücken Sie Strg+C, um das Dashboard zu beenden.")
    print("="*80)
    
    sys.exit(start_dashboard())
