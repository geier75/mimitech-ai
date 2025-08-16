#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Simple Web Dashboard

Dieses Skript implementiert ein einfaches webbasiertes Dashboard zur Überwachung
des Trainingsfortschritts für das MISO Ultimate AGI-System.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import threading
import numpy as np
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MISO.Dashboard")

# Pfade konfigurieren
EXTERNAL_DRIVE_PATH = "/Volumes/My Book/MISO_Ultimate 15.32.28"
DESKTOP_PATH = os.path.expanduser("~/Desktop")
TRAINING_DATA_PATH = os.path.join(EXTERNAL_DRIVE_PATH, "training_data")
CHECKPOINTS_PATH = os.path.join(EXTERNAL_DRIVE_PATH, "checkpoints")
LOGS_PATH = os.path.join(EXTERNAL_DRIVE_PATH, "logs")

# Stelle sicher, dass alle erforderlichen Verzeichnisse existieren
os.makedirs(TRAINING_DATA_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

# Erstelle Unterverzeichnisse für verschiedene Trainingstypen
COMPONENT_TRAINING_PATH = os.path.join(TRAINING_DATA_PATH, "component")
INTEGRATED_TRAINING_PATH = os.path.join(TRAINING_DATA_PATH, "integrated")
END_TO_END_TRAINING_PATH = os.path.join(TRAINING_DATA_PATH, "end_to_end")
FINE_TUNING_PATH = os.path.join(TRAINING_DATA_PATH, "fine_tuning")

os.makedirs(COMPONENT_TRAINING_PATH, exist_ok=True)
os.makedirs(INTEGRATED_TRAINING_PATH, exist_ok=True)
os.makedirs(END_TO_END_TRAINING_PATH, exist_ok=True)
os.makedirs(FINE_TUNING_PATH, exist_ok=True)

class TrainingMonitor:
    """Überwacht den Trainingsfortschritt und sammelt Metriken."""
    
    def __init__(self, config_path=None):
        """Initialisiert den TrainingMonitor."""
        self.training_active = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_component = ""
        self.metrics = {
            "loss": [],
            "accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "learning_rate": []
        }
        self.components_status = {}
        self.start_time = None
        self.end_time = None
        
        # Lade Konfiguration, falls vorhanden
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._create_default_config()
            
        # Speichere Konfiguration
        self._save_config()
    
    def _create_default_config(self):
        """Erstellt eine Standardkonfiguration."""
        return {
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 100,
                "optimizer": "adam",
                "loss_function": "categorical_crossentropy",
                "validation_split": 0.2,
                "early_stopping": True,
                "patience": 10,
                "components": [
                    "t_mathematics",
                    "prism_engine",
                    "echo_prime",
                    "nexus_os",
                    "vx_hyperfilter",
                    "vx_memex",
                    "vx_selfwriter"
                ]
            },
            "hardware": {
                "use_gpu": True,
                "use_mlx": True,
                "precision": "float16",
                "memory_optimization": True
            },
            "paths": {
                "training_data": TRAINING_DATA_PATH,
                "checkpoints": CHECKPOINTS_PATH,
                "logs": LOGS_PATH
            },
            "dashboard": {
                "update_interval": 5,  # Sekunden
                "metrics_to_display": ["loss", "accuracy", "learning_rate"],
                "show_component_status": True,
                "show_progress_bars": True,
                "show_time_estimates": True
            }
        }
    
    def _save_config(self):
        """Speichert die Konfiguration."""
        config_path = os.path.join(EXTERNAL_DRIVE_PATH, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        logger.info(f"Konfiguration gespeichert in {config_path}")
    
    def start_training(self, component=None):
        """Startet das Training für eine Komponente."""
        self.training_active = True
        self.start_time = datetime.now()
        self.current_epoch = 0
        self.total_epochs = self.config["training"]["epochs"]
        
        if component:
            self.current_component = component
        else:
            self.current_component = self.config["training"]["components"][0]
        
        self.components_status[self.current_component] = {
            "status": "training",
            "progress": 0,
            "metrics": {k: [] for k in self.metrics}
        }
        
        logger.info(f"Training gestartet für {self.current_component}")
    
    def stop_training(self):
        """Stoppt das Training."""
        self.training_active = False
        self.end_time = datetime.now()
        
        if self.current_component:
            self.components_status[self.current_component]["status"] = "completed"
            self.components_status[self.current_component]["progress"] = 100
        
        logger.info(f"Training gestoppt für {self.current_component}")
    
    def update_metrics(self, metrics):
        """Aktualisiert die Trainingsmetriken."""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                
                if self.current_component in self.components_status:
                    self.components_status[self.current_component]["metrics"][key].append(value)
        
        self.current_epoch += 1
        
        if self.current_component in self.components_status:
            progress = min(100, int((self.current_epoch / self.total_epochs) * 100))
            self.components_status[self.current_component]["progress"] = progress
        
        logger.info(f"Metriken aktualisiert für Epoche {self.current_epoch}/{self.total_epochs}")
    
    def get_training_status(self):
        """Gibt den aktuellen Trainingsstatus zurück."""
        return {
            "active": self.training_active,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_component": self.current_component,
            "progress": min(100, int((self.current_epoch / self.total_epochs) * 100)) if self.total_epochs > 0 else 0,
            "metrics": self.metrics,
            "components_status": self.components_status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_time": str(datetime.now() - self.start_time) if self.start_time else "00:00:00",
            "estimated_time_remaining": self._estimate_remaining_time()
        }
    
    def _estimate_remaining_time(self):
        """Schätzt die verbleibende Trainingszeit."""
        if not self.start_time or self.current_epoch == 0:
            return "Unbekannt"
        
        elapsed_time = datetime.now() - self.start_time
        elapsed_epochs = self.current_epoch
        
        if elapsed_epochs == 0:
            return "Berechne..."
        
        time_per_epoch = elapsed_time / elapsed_epochs
        remaining_epochs = self.total_epochs - elapsed_epochs
        
        estimated_time = time_per_epoch * remaining_epochs
        
        # Format as HH:MM:SS
        hours, remainder = divmod(estimated_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

class TrainingSimulator(threading.Thread):
    """Simuliert den Trainingsfortschritt für Testzwecke."""
    
    def __init__(self, monitor, interval=1.0):
        """Initialisiert den TrainingSimulator."""
        threading.Thread.__init__(self)
        self.monitor = monitor
        self.interval = interval
        self.daemon = True
        self.running = False
    
    def run(self):
        """Führt die Simulation aus."""
        self.running = True
        
        # Starte Training für die erste Komponente
        components = self.monitor.config["training"]["components"]
        total_epochs = self.monitor.config["training"]["epochs"]
        
        for component in components:
            if not self.running:
                break
                
            self.monitor.start_training(component)
            
            for epoch in range(total_epochs):
                if not self.running:
                    break
                    
                # Simuliere Metriken
                loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.05)
                accuracy = 1.0 - loss + np.random.normal(0, 0.05)
                val_loss = loss + np.random.normal(0, 0.1)
                val_accuracy = accuracy - np.random.normal(0, 0.1)
                learning_rate = 0.001 * (0.95 ** (epoch // 10))
                
                metrics = {
                    "loss": max(0.01, min(1.0, loss)),
                    "accuracy": max(0, min(1.0, accuracy)),
                    "validation_loss": max(0.01, min(1.0, val_loss)),
                    "validation_accuracy": max(0, min(1.0, val_accuracy)),
                    "learning_rate": learning_rate
                }
                
                self.monitor.update_metrics(metrics)
                
                # Simuliere Checkpoint-Speicherung
                if epoch % 10 == 0:
                    checkpoint_path = os.path.join(CHECKPOINTS_PATH, f"{component}_epoch_{epoch}.ckpt")
                    with open(checkpoint_path, 'w') as f:
                        f.write(f"Simulated checkpoint for {component} at epoch {epoch}")
                    logger.info(f"Checkpoint gespeichert: {checkpoint_path}")
                
                time.sleep(self.interval)
            
            self.monitor.stop_training()
        
        logger.info("Trainingssimulation abgeschlossen")
    
    def stop(self):
        """Stoppt die Simulation."""
        self.running = False

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP-Handler für das Dashboard."""
    
    def __init__(self, *args, monitor=None, **kwargs):
        self.monitor = monitor
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Behandelt GET-Anfragen."""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_dashboard_html().encode())
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.monitor.get_training_status()).encode())
        elif self.path == '/start':
            if not self.monitor.training_active:
                simulator = TrainingSimulator(self.monitor, interval=0.5)
                simulator.start()
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
        elif self.path == '/stop':
            if self.monitor.training_active:
                self.monitor.stop_training()
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
        elif self.path == '/start_benchmark_dashboard':
            # Starte das Benchmark-Dashboard in einem neuen Prozess
            import subprocess
            import sys
            
            try:
                # Starte das Dashboard in einem separaten Prozess
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'start_benchmark_dashboard.py')
                subprocess.Popen([sys.executable, script_path])
                logger.info("Benchmark-Dashboard wurde gestartet")
            except Exception as e:
                logger.error(f"Fehler beim Starten des Benchmark-Dashboards: {e}")
            
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
        elif self.path == '/save_checkpoint':
            if self.monitor.training_active:
                checkpoint_path = os.path.join(
                    self.monitor.config["paths"]["checkpoints"],
                    f"{self.monitor.current_component}_manual_checkpoint_epoch_{self.monitor.current_epoch}.ckpt"
                )
                
                with open(checkpoint_path, 'w') as f:
                    f.write(f"Manual checkpoint for {self.monitor.current_component} at epoch {self.monitor.current_epoch}")
                
                logger.info(f"Checkpoint gespeichert: {checkpoint_path}")
            
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def get_dashboard_html(self):
        """Generiert das HTML für das Dashboard."""
        status = self.monitor.get_training_status()
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MISO Ultimate Training Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
                h1, h2 {
                    color: #333;
                }
                .status-box {
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .progress-container {
                    margin: 10px 0;
                }
                .progress-bar {
                    background-color: #e0e0e0;
                    border-radius: 5px;
                    height: 20px;
                    width: 100%;
                }
                .progress-bar-fill {
                    background-color: #4CAF50;
                    height: 100%;
                    border-radius: 5px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 12px;
                }
                .button-container {
                    margin: 20px 0;
                }
                .button {
                    padding: 10px 15px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-weight: bold;
                    margin-right: 10px;
                }
                .start-button {
                    background-color: #4CAF50;
                    color: white;
                }
                .stop-button {
                    background-color: #f44336;
                    color: white;
                }
                .save-button {
                    background-color: #2196F3;
                    color: white;
                }
                .benchmark-button {
                    background-color: #7B1FA2;
                    color: white;
                }
                .status-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 10px;
                }
                .status-item {
                    padding: 5px;
                }
                .component-list {
                    margin-top: 20px;
                }
                .component-item {
                    margin-bottom: 10px;
                }
                .metrics-container {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-top: 20px;
                }
                .metric-box {
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    padding: 15px;
                    border-radius: 5px;
                }
                .refresh-text {
                    font-size: 12px;
                    color: #666;
                    margin-top: 20px;
                    text-align: center;
                }
            </style>
            <script>
                // Automatische Aktualisierung alle 5 Sekunden
                setInterval(function() {
                    location.reload();
                }, 5000);
            </script>
        </head>
        <body>
            <div class="container">
                <h1>MISO Ultimate Training Dashboard</h1>
                
                <div class="button-container">
        """
        
        if not status["active"]:
            html += '<a href="/start"><button class="button start-button">Training starten</button></a>'
        else:
            html += '<a href="/stop"><button class="button stop-button">Training stoppen</button></a>'
        
        html += '<a href="/save_checkpoint"><button class="button save-button">Checkpoint speichern</button></a>'
        html += '<a href="/start_benchmark_dashboard"><button class="button benchmark-button">Benchmark-Dashboard starten</button></a>'
        
        html += """
                </div>
                
                <div class="status-box">
                    <h2>Training Status</h2>
                    <div class="status-grid">
        """
        
        html += f'<div class="status-item"><strong>Status:</strong> {"Training läuft" if status["active"] else "Bereit"}</div>'
        html += f'<div class="status-item"><strong>Komponente:</strong> {status["current_component"] if status["current_component"] else "-"}</div>'
        html += f'<div class="status-item"><strong>Epoche:</strong> {status["current_epoch"]}/{status["total_epochs"]}</div>'
        html += f'<div class="status-item"><strong>Verstrichene Zeit:</strong> {status["elapsed_time"]}</div>'
        html += f'<div class="status-item"><strong>Verbleibende Zeit:</strong> {status["estimated_time_remaining"]}</div>'
        
        html += """
                    </div>
                    
                    <div class="progress-container">
                        <strong>Gesamtfortschritt:</strong>
                        <div class="progress-bar">
        """
        
        html += f'<div class="progress-bar-fill" style="width: {status["progress"]}%;">{status["progress"]}%</div>'
        
        html += """
                        </div>
                    </div>
                </div>
                
                <div class="component-list">
                    <h2>Komponenten Status</h2>
        """
        
        for component, component_status in status["components_status"].items():
            html += f'<div class="component-item"><strong>{component}:</strong> {component_status["status"]}'
            html += '<div class="progress-bar">'
            html += f'<div class="progress-bar-fill" style="width: {component_status["progress"]}%;">{component_status["progress"]}%</div>'
            html += '</div></div>'
        
        html += """
                </div>
                
                <div class="metrics-container">
                    <div class="metric-box">
                        <h2>Loss</h2>
                        <p>Training Loss: 
        """
        
        if status["metrics"]["loss"]:
            html += f'{status["metrics"]["loss"][-1]:.4f}'
        else:
            html += '-'
        
        html += """
                        </p>
                        <p>Validation Loss: 
        """
        
        if status["metrics"]["validation_loss"]:
            html += f'{status["metrics"]["validation_loss"][-1]:.4f}'
        else:
            html += '-'
        
        html += """
                        </p>
                    </div>
                    
                    <div class="metric-box">
                        <h2>Accuracy</h2>
                        <p>Training Accuracy: 
        """
        
        if status["metrics"]["accuracy"]:
            html += f'{status["metrics"]["accuracy"][-1]:.4f}'
        else:
            html += '-'
        
        html += """
                        </p>
                        <p>Validation Accuracy: 
        """
        
        if status["metrics"]["validation_accuracy"]:
            html += f'{status["metrics"]["validation_accuracy"][-1]:.4f}'
        else:
            html += '-'
        
        html += """
                        </p>
                    </div>
                </div>
                
                <p class="refresh-text">Diese Seite wird automatisch alle 5 Sekunden aktualisiert.</p>
            </div>
        </body>
        </html>
        """
        
        return html

def run_server(monitor, port=8080):
    """Startet den Webserver für das Dashboard."""
    # Erstelle einen benutzerdefinierten Handler mit Zugriff auf den Monitor
    handler = lambda *args, **kwargs: DashboardHandler(*args, monitor=monitor, **kwargs)
    
    # Starte den Server
    server = HTTPServer(('localhost', port), handler)
    logger.info(f"Server gestartet auf http://localhost:{port}")
    
    # Öffne den Browser
    webbrowser.open(f"http://localhost:{port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    
    server.server_close()
    logger.info("Server gestoppt")

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
    
    logger.info(f"Desktop-Verknüpfung erstellt: {shortcut_path}")

if __name__ == "__main__":
    # Erstelle Desktop-Verknüpfung
    create_desktop_shortcut()
    
    # Erstelle den TrainingMonitor
    monitor = TrainingMonitor()
    
    # Starte den Server
    run_server(monitor)
