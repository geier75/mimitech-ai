#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Einfaches Dashboard
=======================================

Ein einfaches, aber funktionales Dashboard für das MISO Ultimate AGI-System.
"""

import os
import sys
import json
import time
import random
import logging
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.Dashboard")

class TrainingState:
    """Repräsentiert den aktuellen Zustand des Trainings."""
    
    def __init__(self):
        self.active = False
        self.start_time = None
        self.current_epoch = 0
        self.total_epochs = 100
        self.current_component = None
        self.components = ["MISO_CORE", "VX_MEMEX", "VX_REASON", "VX_INTENT"]
        self.metrics = {}
        self.hardware_stats = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "gpu_usage": 0.0,
            "temperature": 0.0,
            "is_apple_silicon": True,
            "mlx_available": True,
            "device_name": "Apple M4 Max"
        }
        self.checkpoints = []
        self.errors = []
        self.warnings = []
    
    def to_dict(self):
        """Konvertiert den Trainingszustand in ein Dictionary."""
        return {
            "active": self.active,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_component": self.current_component,
            "components": self.components,
            "metrics": self.metrics,
            "hardware_stats": self.hardware_stats,
            "checkpoints": self.checkpoints,
            "errors": self.errors,
            "warnings": self.warnings,
            "last_update": datetime.now().isoformat()
        }

class TrainingManager:
    """Verwaltet den Trainingsprozess."""
    
    def __init__(self):
        self.state = TrainingState()
        self._training_thread = None
        self._stop_event = threading.Event()
    
    def start_training(self):
        """Startet das Training."""
        if self.state.active:
            logger.warning("Training läuft bereits")
            return False
        
        self.state.active = True
        self.state.start_time = datetime.now()
        self.state.current_epoch = 0
        
        self._stop_event.clear()
        self._training_thread = threading.Thread(target=self._training_loop)
        self._training_thread.daemon = True
        self._training_thread.start()
        
        logger.info("Training gestartet")
        return True
    
    def stop_training(self):
        """Stoppt das laufende Training."""
        if not self.state.active:
            logger.warning("Kein aktives Training zum Stoppen")
            return False
        
        self._stop_event.set()
        if self._training_thread:
            self._training_thread.join(timeout=5.0)
        
        self.state.active = False
        logger.info("Training gestoppt")
        return True
    
    def _training_loop(self):
        """Hauptschleife für das Training."""
        try:
            total_epochs = self.state.total_epochs
            components = self.state.components
            
            for epoch in range(total_epochs):
                if self._stop_event.is_set():
                    break
                
                self.state.current_epoch = epoch + 1
                
                for component in components:
                    if self._stop_event.is_set():
                        break
                    
                    self.state.current_component = component
                    
                    # Simuliere Metriken
                    loss = 1.0 - (epoch / total_epochs) * 0.9
                    accuracy = 0.5 + (epoch / total_epochs) * 0.45
                    
                    # Füge zufällige Schwankungen hinzu
                    loss += random.uniform(-0.05, 0.05)
                    accuracy += random.uniform(-0.02, 0.02)
                    
                    # Begrenze die Werte
                    loss = max(0.01, min(1.0, loss))
                    accuracy = max(0.0, min(0.99, accuracy))
                    
                    # Aktualisiere Metriken
                    if component not in self.state.metrics:
                        self.state.metrics[component] = {}
                    
                    self.state.metrics[component].update({
                        "loss": loss,
                        "accuracy": accuracy,
                        "f1_score": accuracy * 0.95,
                        "precision": accuracy * 0.98,
                        "recall": accuracy * 0.92,
                        "explained_variance": 0.7 + (epoch / total_epochs) * 0.25
                    })
                    
                    # Aktualisiere Hardware-Statistiken
                    self.state.hardware_stats["cpu_usage"] = 30 + random.uniform(0, 40)
                    self.state.hardware_stats["memory_usage"] = 40 + random.uniform(0, 30)
                    self.state.hardware_stats["gpu_usage"] = 60 + random.uniform(0, 30)
                    
                    # Simuliere Verarbeitungszeit
                    time.sleep(0.5)
                
                # Erstelle Checkpoint nach bestimmten Epochen
                if (epoch + 1) % 10 == 0:
                    checkpoint_info = {
                        "epoch": epoch + 1,
                        "timestamp": datetime.now().isoformat(),
                        "metrics": {k: v.copy() for k, v in self.state.metrics.items()}
                    }
                    self.state.checkpoints.append(checkpoint_info)
                    logger.info(f"Checkpoint erstellt für Epoche {epoch + 1}")
            
            logger.info("Training abgeschlossen")
            self.state.active = False
            
        except Exception as e:
            logger.error(f"Fehler im Trainingsprozess: {e}")
            self.state.errors.append(str(e))
            self.state.active = False

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP-Handler für das Dashboard."""
    
    def __init__(self, *args, training_manager=None, **kwargs):
        self.training_manager = training_manager
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Überschreibe die Standard-Logging-Methode."""
        return
    
    def _set_headers(self, content_type="text/html"):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.end_headers()
    
    def do_GET(self):
        """Behandelt GET-Anfragen."""
        if self.path == '/':
            self._serve_dashboard()
        elif self.path == '/api/status':
            self._serve_status()
        else:
            self._serve_not_found()
    
    def do_POST(self):
        """Behandelt POST-Anfragen."""
        if self.path == '/api/start':
            self._handle_start_training()
        elif self.path == '/api/stop':
            self._handle_stop_training()
        else:
            self._serve_not_found()
    
    def _serve_dashboard(self):
        """Liefert die Dashboard-HTML-Seite."""
        self._set_headers()
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MISO Ultimate AGI Training Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    color: #333;
                }
                
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                
                .header {
                    background: linear-gradient(135deg, #2c3e50, #34495e);
                    color: white;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                
                .header h1 {
                    margin: 0;
                    font-size: 24px;
                }
                
                .header p {
                    margin: 5px 0 0 0;
                    opacity: 0.8;
                }
                
                .dashboard-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                
                .card {
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 20px;
                    transition: transform 0.2s;
                }
                
                .card:hover {
                    transform: translateY(-5px);
                }
                
                .card h2 {
                    margin-top: 0;
                    font-size: 18px;
                    color: #2c3e50;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }
                
                .metric {
                    display: flex;
                    justify-content: space-between;
                    margin: 10px 0;
                }
                
                .metric-name {
                    font-weight: 500;
                }
                
                .metric-value {
                    font-weight: 600;
                    color: #3498db;
                }
                
                .controls {
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 20px;
                    margin-top: 20px;
                }
                
                .btn {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 10px 15px;
                    margin-right: 10px;
                    cursor: pointer;
                    transition: background-color 0.2s;
                }
                
                .btn:hover {
                    background-color: #2980b9;
                }
                
                .btn-danger {
                    background-color: #e74c3c;
                }
                
                .btn-danger:hover {
                    background-color: #c0392b;
                }
                
                .btn-success {
                    background-color: #2ecc71;
                }
                
                .btn-success:hover {
                    background-color: #27ae60;
                }
                
                .status-bar {
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 15px 20px;
                    margin-top: 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                
                .status-indicator {
                    display: flex;
                    align-items: center;
                }
                
                .status-dot {
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                
                .status-active {
                    background-color: #2ecc71;
                }
                
                .status-inactive {
                    background-color: #e74c3c;
                }
                
                .progress-container {
                    width: 100%;
                    background-color: #f1f1f1;
                    border-radius: 4px;
                    margin: 10px 0;
                }
                
                .progress-bar {
                    height: 8px;
                    border-radius: 4px;
                    background-color: #3498db;
                    width: 0%;
                    transition: width 0.5s;
                }
                
                .hardware-stats {
                    margin-top: 10px;
                }
                
                .hardware-stat {
                    display: flex;
                    justify-content: space-between;
                    margin: 5px 0;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>MISO Ultimate AGI Training Dashboard</h1>
                    <p>Echtzeit-Überwachung und Steuerung des Trainingsprozesses</p>
                </div>
            </div>
            
            <div class="container">
                <div class="status-bar">
                    <div class="status-indicator">
                        <div id="status-dot" class="status-dot status-inactive"></div>
                        <span id="status-text">Inaktiv</span>
                    </div>
                    <div>
                        <span id="current-epoch">Epoche: 0/0</span> | 
                        <span id="current-component">Komponente: -</span>
                    </div>
                </div>
                
                <div class="controls">
                    <h2>Training steuern</h2>
                    <button id="start-btn" class="btn btn-success">Training starten</button>
                    <button id="stop-btn" class="btn btn-danger">Training stoppen</button>
                    <button id="config-btn" class="btn">Konfiguration</button>
                    
                    <div class="progress-container">
                        <div id="progress-bar" class="progress-bar"></div>
                    </div>
                    <div id="progress-text">Fortschritt: 0%</div>
                </div>
                
                <div class="dashboard-grid">
                    <div class="card">
                        <h2>Hardware-Status</h2>
                        <div class="hardware-stats" id="hardware-stats">
                            <div class="hardware-stat">
                                <span>CPU-Auslastung:</span>
                                <span>0%</span>
                            </div>
                            <div class="hardware-stat">
                                <span>Speichernutzung:</span>
                                <span>0%</span>
                            </div>
                            <div class="hardware-stat">
                                <span>GPU-Auslastung:</span>
                                <span>0%</span>
                            </div>
                            <div class="hardware-stat">
                                <span>Gerät:</span>
                                <span>-</span>
                            </div>
                            <div class="hardware-stat">
                                <span>MLX verfügbar:</span>
                                <span>-</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Training-Übersicht</h2>
                        <div id="training-overview">
                            <div class="metric">
                                <span class="metric-name">Startzeit:</span>
                                <span class="metric-value" id="start-time">-</span>
                            </div>
                            <div class="metric">
                                <span class="metric-name">Laufzeit:</span>
                                <span class="metric-value" id="runtime">-</span>
                            </div>
                            <div class="metric">
                                <span class="metric-name">Geschätzte Restzeit:</span>
                                <span class="metric-value" id="eta">-</span>
                            </div>
                            <div class="metric">
                                <span class="metric-name">Checkpoints:</span>
                                <span class="metric-value" id="checkpoints">0</span>
                            </div>
                        </div>
                    </div>
                    
                    <div id="component-metrics"></div>
                </div>
            </div>
            
            <script>
                // Dashboard-Funktionalität
                let startTime = null;
                
                // Aktualisiere das Dashboard alle 2 Sekunden
                function updateDashboard() {
                    fetch('/api/status')
                        .then(function(response) { return response.json(); })
                        .then(function(data) {
                            // Aktualisiere Status
                            const statusDot = document.getElementById('status-dot');
                            const statusText = document.getElementById('status-text');
                            
                            if (data.active) {
                                statusDot.className = 'status-dot status-active';
                                statusText.textContent = 'Training aktiv';
                                
                                if (!startTime && data.start_time) {
                                    startTime = new Date(data.start_time);
                                }
                            } else {
                                statusDot.className = 'status-dot status-inactive';
                                statusText.textContent = 'Inaktiv';
                            }
                            
                            // Aktualisiere Epochen und Komponenten
                            document.getElementById('current-epoch').textContent = 'Epoche: ' + data.current_epoch + '/' + data.total_epochs;
                            document.getElementById('current-component').textContent = 'Komponente: ' + (data.current_component || '-');
                            
                            // Aktualisiere Fortschrittsbalken
                            const progress = data.total_epochs > 0 ? (data.current_epoch / data.total_epochs) * 100 : 0;
                            document.getElementById('progress-bar').style.width = progress + '%';
                            document.getElementById('progress-text').textContent = 'Fortschritt: ' + progress.toFixed(1) + '%';
                            
                            // Aktualisiere Hardware-Statistiken
                            const hardwareStats = document.getElementById('hardware-stats');
                            if (data.hardware_stats) {
                                hardwareStats.innerHTML = 
                                    '<div class="hardware-stat">' +
                                        '<span>CPU-Auslastung:</span>' +
                                        '<span>' + data.hardware_stats.cpu_usage.toFixed(1) + '%</span>' +
                                    '</div>' +
                                    '<div class="hardware-stat">' +
                                        '<span>Speichernutzung:</span>' +
                                        '<span>' + data.hardware_stats.memory_usage.toFixed(1) + '%</span>' +
                                    '</div>' +
                                    '<div class="hardware-stat">' +
                                        '<span>GPU-Auslastung:</span>' +
                                        '<span>' + data.hardware_stats.gpu_usage.toFixed(1) + '%</span>' +
                                    '</div>' +
                                    '<div class="hardware-stat">' +
                                        '<span>Gerät:</span>' +
                                        '<span>' + (data.hardware_stats.device_name || '-') + '</span>' +
                                    '</div>' +
                                    '<div class="hardware-stat">' +
                                        '<span>MLX verfügbar:</span>' +
                                        '<span>' + (data.hardware_stats.mlx_available ? 'Ja' : 'Nein') + '</span>' +
                                    '</div>';
                            }
                            
                            // Aktualisiere Training-Übersicht
                            if (data.start_time) {
                                document.getElementById('start-time').textContent = new Date(data.start_time).toLocaleTimeString();
                                
                                if (startTime) {
                                    const runtime = Math.floor((new Date() - startTime) / 1000);
                                    const hours = Math.floor(runtime / 3600);
                                    const minutes = Math.floor((runtime % 3600) / 60);
                                    const seconds = runtime % 60;
                                    document.getElementById('runtime').textContent = 
                                        hours.toString().padStart(2, '0') + ':' + 
                                        minutes.toString().padStart(2, '0') + ':' + 
                                        seconds.toString().padStart(2, '0');
                                    
                                    // Geschätzte Restzeit
                                    if (data.current_epoch > 0 && data.total_epochs > 0) {
                                        const secondsPerEpoch = runtime / data.current_epoch;
                                        const remainingEpochs = data.total_epochs - data.current_epoch;
                                        const eta = Math.floor(secondsPerEpoch * remainingEpochs);
                                        
                                        const etaHours = Math.floor(eta / 3600);
                                        const etaMinutes = Math.floor((eta % 3600) / 60);
                                        const etaSeconds = eta % 60;
                                        document.getElementById('eta').textContent = 
                                            etaHours.toString().padStart(2, '0') + ':' + 
                                            etaMinutes.toString().padStart(2, '0') + ':' + 
                                            etaSeconds.toString().padStart(2, '0');
                                    }
                                }
                            }
                            
                            // Aktualisiere Checkpoints
                            document.getElementById('checkpoints').textContent = data.checkpoints.length;
                            
                            // Aktualisiere Komponenten-Metriken
                            const componentMetricsContainer = document.getElementById('component-metrics');
                            componentMetricsContainer.innerHTML = '';
                            
                            for (const component in data.metrics) {
                                const metrics = data.metrics[component];
                                const card = document.createElement('div');
                                card.className = 'card';
                                
                                let metricsHtml = '<h2>' + component + '</h2>';
                                
                                for (const metric in metrics) {
                                    const formattedMetric = metric
                                        .replace(/_/g, ' ')
                                        .replace(/\\b\\w/g, function(l) { return l.toUpperCase(); });
                                    
                                    metricsHtml += 
                                        '<div class="metric">' +
                                            '<span class="metric-name">' + formattedMetric + ':</span>' +
                                            '<span class="metric-value">' + metrics[metric].toFixed(4) + '</span>' +
                                        '</div>';
                                }
                                
                                card.innerHTML = metricsHtml;
                                componentMetricsContainer.appendChild(card);
                            }
                        })
                        .catch(function(error) {
                            console.error('Fehler beim Aktualisieren des Dashboards:', error);
                        });
                }
                
                // Aktualisiere das Dashboard initial und dann alle 2 Sekunden
                updateDashboard();
                setInterval(updateDashboard, 2000);
                
                // Event-Handler für Buttons
                document.getElementById('start-btn').addEventListener('click', function() {
                    fetch('/api/start', {
                        method: 'POST'
                    })
                    .then(function(response) { return response.json(); })
                    .then(function(data) {
                        if (data.success) {
                            startTime = new Date();
                        } else {
                            alert('Fehler beim Starten des Trainings');
                        }
                    });
                });
                
                document.getElementById('stop-btn').addEventListener('click', function() {
                    fetch('/api/stop', {
                        method: 'POST'
                    })
                    .then(function(response) { return response.json(); })
                    .then(function(data) {
                        if (!data.success) {
                            alert('Fehler beim Stoppen des Trainings');
                        }
                    });
                });
                
                document.getElementById('config-btn').addEventListener('click', function() {
                    alert('Konfiguration wird in einem separaten Dashboard geöffnet');
                    window.open('/config', '_blank');
                });
            </script>
        </body>
        </html>
        """
        
        self.wfile.write(html.encode())
    
    def _serve_status(self):
        """Liefert den aktuellen Trainingsstatus als JSON."""
        self._set_headers("application/json")
        status = self.training_manager.state.to_dict()
        self.wfile.write(json.dumps(status).encode())
    
    def _serve_not_found(self):
        """Liefert eine 404-Fehlerseite."""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<html><body><h1>404 Not Found</h1></body></html>")
    
    def _handle_start_training(self):
        """Behandelt die Anfrage zum Starten des Trainings."""
        self._set_headers("application/json")
        success = self.training_manager.start_training()
        self.wfile.write(json.dumps({"success": success}).encode())
    
    def _handle_stop_training(self):
        """Behandelt die Anfrage zum Stoppen des Trainings."""
        self._set_headers("application/json")
        success = self.training_manager.stop_training()
        self.wfile.write(json.dumps({"success": success}).encode())

def run_server(port=62304):
    """Startet den Dashboard-Server."""
    training_manager = TrainingManager()
    
    # Handler-Klasse mit Training Manager
    handler = lambda *args, **kwargs: DashboardHandler(*args, training_manager=training_manager, **kwargs)
    
    server = HTTPServer(('localhost', port), handler)
    
    logger.info(f"MISO Ultimate Training Dashboard wird gestartet auf http://localhost:{port}")
    print("="*80)
    print(f"MISO Ultimate AGI Training Dashboard wird gestartet auf http://localhost:{port}")
    print("Das Dashboard wird in Ihrem Standardbrowser geöffnet.")
    print("Drücken Sie Strg+C, um das Dashboard zu beenden.")
    print("="*80)
    
    # Öffne das Dashboard im Browser
    webbrowser.open(f"http://localhost:{port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server beendet")
        print("\nServer beendet")

if __name__ == "__main__":
    run_server(62304)
