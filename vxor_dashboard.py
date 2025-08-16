#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vXor AGI-System: vxor_dashboard
Dashboard für Benchmark-Ergebnisse und System-Performance

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import json
import glob
import logging
import threading
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VXOR.Dashboard - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Globale Variablen
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5151
DASHBOARD_PATH = Path(__file__).parent / "benchmark_dashboard.html"
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
TEST_MODE = os.environ.get("MISO_TEST_MODE", "0") == "1"

class BenchmarkResultsHandler(SimpleHTTPRequestHandler):
    """Handler für HTTP-Anfragen an das Dashboard"""
    
    def __init__(self, *args, **kwargs):
        self.directory = Path(__file__).parent.absolute()
        super().__init__(*args, directory=str(self.directory), **kwargs)
    
    def do_GET(self):
        """Behandelt GET-Anfragen"""
        if self.path == "/dashboard/benchmark":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            # Dashboard-HTML laden und senden
            with open(DASHBOARD_PATH, 'rb') as file:
                self.wfile.write(file.read())
                
        elif self.path == "/api/benchmark/results":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            # Benchmark-Ergebnisse sammeln und als JSON senden
            results = collect_benchmark_results()
            self.wfile.write(json.dumps(results).encode())
            
        elif self.path.startswith("/benchmark_results/"):
            # Dateien aus dem Benchmark-Ergebnis-Verzeichnis bereitstellen
            file_path = Path(__file__).parent / self.path.lstrip("/")
            if file_path.exists() and file_path.is_file():
                self.send_response(200)
                if file_path.suffix == ".png":
                    self.send_header("Content-type", "image/png")
                elif file_path.suffix == ".json":
                    self.send_header("Content-type", "application/json")
                else:
                    self.send_header("Content-type", "text/plain")
                self.end_headers()
                
                with open(file_path, 'rb') as file:
                    self.wfile.write(file.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"File not found")
        else:
            # Standardverhalten für andere Pfade
            return super().do_GET()
            
    def log_message(self, format, *args):
        """Überschreibt die Standard-Logging-Funktion"""
        logger.info("%s - %s", self.address_string(), format % args)


def collect_benchmark_results():
    """Sammelt alle Benchmark-Ergebnisse aus dem Ergebnisverzeichnis"""
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "categories": {},
        "plots": [],
        "raw_data": []
    }
    
    # Prüfe, ob das Ergebnisverzeichnis existiert
    if not RESULTS_DIR.exists():
        logger.warning(f"Ergebnisverzeichnis {RESULTS_DIR} nicht gefunden")
        return results
    
    # JSON-Dateien für Rohdaten sammeln
    json_files = list(RESULTS_DIR.glob("*.json"))
    logger.info(f"{len(json_files)} JSON-Ergebnisdateien gefunden")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results["raw_data"].append({
                    "name": json_file.stem,
                    "data": data
                })
                
                # Kategorisiere die Ergebnisse
                if "tensor" in json_file.stem:
                    if "categories" not in results["categories"]:
                        results["categories"]["tensor"] = []
                    results["categories"]["tensor"].append(json_file.stem)
                    
                elif "matmul" in json_file.stem or "svd" in json_file.stem:
                    if "matrix" not in results["categories"]:
                        results["categories"]["matrix"] = []
                    results["categories"]["matrix"].append(json_file.stem)
                    
                elif "ai" in json_file.stem or "model" in json_file.stem:
                    if "ai" not in results["categories"]:
                        results["categories"]["ai"] = []
                    results["categories"]["ai"].append(json_file.stem)
        except Exception as e:
            logger.error(f"Fehler beim Lesen von {json_file}: {e}")
    
    # PNG-Dateien für Plots sammeln
    png_files = list(RESULTS_DIR.glob("*.png")) + list(Path(__file__).parent.glob("tests/miso_math/benchmark_results/*.png"))
    logger.info(f"{len(png_files)} PNG-Visualisierungen gefunden")
    
    for png_file in png_files:
        relative_path = png_file.relative_to(Path(__file__).parent)
        results["plots"].append({
            "name": png_file.stem,
            "path": str(relative_path),
            "category": "matrix" if "matmul" in png_file.stem or "svd" in png_file.stem else "tensor"
        })
    
    # Benchmark-Zusammenfassung hinzufügen
    summary_file = RESULTS_DIR / "benchmark_summary.md"
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                results["summary"] = f.read()
        except Exception as e:
            logger.error(f"Fehler beim Lesen der Zusammenfassung: {e}")
    
    return results


def start_server(host=SERVER_HOST, port=SERVER_PORT):
    """Startet den HTTP-Server für das Dashboard"""
    try:
        server = HTTPServer((host, port), BenchmarkResultsHandler)
        logger.info(f"Server gestartet: http://{host}:{port}/dashboard/benchmark")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Fehler beim Starten des Servers: {e}")
        return False
    return True


def run_server(open_browser=True, host=SERVER_HOST, port=SERVER_PORT):
    """Startet den Server in einem separaten Thread und öffnet (optional) den Browser"""
    if not init():
        logger.error("Initialisierung fehlgeschlagen")
        return False
    
    # Bereite Benchmarking-Ergebnisverzeichnis vor
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Starte Server in einem eigenen Thread
    server_thread = threading.Thread(target=start_server, args=(host, port), daemon=True)
    server_thread.start()
    logger.info("Dashboard-Server läuft im Hintergrund")
    
    if open_browser:
        # Öffne Dashboard im Browser
        webbrowser.open(f"http://{host}:{port}/dashboard/benchmark")
        logger.info("Dashboard im Browser geöffnet")
    
    return True


def init():
    """Initialisiert das Modul vxor_dashboard und gibt True zurück bei Erfolg"""
    logger.info("Modul vxor_dashboard erfolgreich initialisiert")
    return True


def boot():
    """Startet das Modul vxor_dashboard (optional)"""
    logger.info("Modul vxor_dashboard gestartet")
    return init()


# Hauptausführung wenn direkt gestartet
if __name__ == "__main__":
    run_server(open_browser=True)
