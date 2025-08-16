#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Dashboard Starter

Dieses Skript startet das MISO Training Dashboard.
"""

import os
import sys
import webbrowser
import logging
from pathlib import Path

# Füge das Verzeichnis zum Python-Pfad hinzu
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Importiere die Dashboard-Komponenten
from miso_dashboard import create_directories, load_config, setup_logging
from miso_dashboard import TrainingMonitor, TrainingSimulator
from miso_dashboard import run_server

def create_desktop_shortcut():
    """Erstellt eine Verknüpfung auf dem Desktop."""
    desktop_path = os.path.expanduser("~/Desktop")
    shortcut_path = os.path.join(desktop_path, "MISO Training Dashboard.command")
    
    script_path = os.path.abspath(__file__)
    
    with open(shortcut_path, 'w') as f:
        f.write(f"""#!/bin/bash
cd "{os.path.dirname(script_path)}"
python3 "{script_path}"
""")
    
    # Mache die Datei ausführbar
    os.chmod(shortcut_path, 0o755)
    
    print(f"Desktop-Verknüpfung erstellt: {shortcut_path}")

if __name__ == "__main__":
    # Erstelle die erforderlichen Verzeichnisse
    create_directories()
    
    # Konfiguriere das Logging
    logger = setup_logging()
    
    # Lade die Konfiguration
    config = load_config()
    
    # Erstelle eine Desktop-Verknüpfung
    create_desktop_shortcut()
    
    # Erstelle den TrainingMonitor
    monitor = TrainingMonitor(config)
    
    # Erstelle den TrainingSimulator
    simulator = TrainingSimulator(monitor)
    
    # Starte den Server
    port = config["dashboard"]["port"] if "port" in config["dashboard"] else 8080
    logger.info(f"Starte MISO Training Dashboard auf Port {port}")
    
    print("="*80)
    print(f"MISO Ultimate Training Dashboard wird gestartet auf http://localhost:{port}")
    print("Das Dashboard wird in Ihrem Standardbrowser geöffnet.")
    print("Drücken Sie Strg+C, um das Dashboard zu beenden.")
    print("="*80)
    
    # Starte den Server
    run_server(monitor, simulator, port)
