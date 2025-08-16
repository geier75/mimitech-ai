#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Dashboard Starter
====================================

Startet das MISO Ultimate AGI Training Dashboard.
"""

import os
import sys
import logging
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import time
from datetime import datetime
from pathlib import Path

# Füge das miso_dashboard-Verzeichnis zum Pfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from miso_dashboard.dashboard_core import TrainingManager, DataManager

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.Dashboard")

# Initialisiere Training Manager
training_manager = TrainingManager()
data_manager = DataManager()

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP-Handler für das Dashboard."""
    
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
        elif self.path.startswith('/static/'):
            self._serve_static()
        else:
            self._serve_not_found()
    
    def do_POST(self):
        """Behandelt POST-Anfragen."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        if self.path == '/api/start':
            self._handle_start_training(post_data)
        elif self.path == '/api/stop':
            self._handle_stop_training()
        else:
            self._serve_not_found()
    
    def _serve_dashboard(self):
        """Liefert die Dashboard-HTML-Seite."""
        self._set_headers()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MISO Ultimate AGI Training Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #2c3e50, #34495e);
                    color: white;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                
                .header h1 {{
                    margin: 0;
                    font-size: 24px;
                }}
                
                .header p {{
                    margin: 5px 0 0 0;
                    opacity: 0.8;
                }}
                
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }}
                
                .card {{
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 20px;
                    transition: transform 0.2s;
                }}
                
                .card:hover {{
                    transform: translateY(-5px);
                }}
                
                .card h2 {{
                    margin-top: 0;
                    font-size: 18px;
                    color: #2c3e50;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                
                .metric {{
                    display: flex;
                    justify-content: space-between;
                    margin: 10px 0;
                }}
                
                .metric-name {{
                    font-weight: 500;
                }}
                
                .metric-value {{
                    font-weight: 600;
                    color: #3498db;
                }}
                
                .controls {{
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 20px;
                    margin-top: 20px;
                }}
                
                .btn {{
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 10px 15px;
                    margin-right: 10px;
                    cursor: pointer;
                    transition: background-color 0.2s;
                }}
                
                .btn:hover {{
                    background-color: #2980b9;
                }}
                
                .btn-danger {{
                    background-color: #e74c3c;
                }}
                
                .btn-danger:hover {{
                    background-color: #c0392b;
                }}
                
                .btn-success {{
                    background-color: #2ecc71;
                }}
                
                .btn-success:hover {{
                    background-color: #27ae60;
                }}
                
                .status-bar {{
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 15px 20px;
                    margin-top: 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                
                .status-indicator {{
                    display: flex;
                    align-items: center;
                }}
                
                .status-dot {{
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }}
                
                .status-active {{
                    background-color: #2ecc71;
                }}
                
                .status-inactive {{
                    background-color: #e74c3c;
                }}
                
                .progress-container {{
                    width: 100%;
                    background-color: #f1f1f1;
                    border-radius: 4px;
                    margin: 10px 0;
                }}
                
                .progress-bar {{
                    height: 8px;
                    border-radius: 4px;
                    background-color: #3498db;
                    width: 0%;
                    transition: width 0.5s;
                }}
                
                .hardware-stats {{
                    margin-top: 10px;
                }}
                
                .hardware-stat {{
                    display: flex;
                    justify-content: space-between;
                    margin: 5px 0;
                    font-size: 14px;
                }}
                
                .chart-container {{
                    height: 200px;
                    margin-top: 15px;
                }}
                
                @media (max-width: 768px) {{
                    .dashboard-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
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
                    
                    <!-- Dynamische Komponenten-Karten werden hier eingefügt -->
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
                            document.getElementById('current-epoch').textContent = `Epoche: ${data.current_epoch}/${data.total_epochs}`;
                            document.getElementById('current-component').textContent = `Komponente: ${data.current_component || '-'}`;
                            
                            // Aktualisiere Fortschrittsbalken
                            const progress = data.total_epochs > 0 ? (data.current_epoch / data.total_epochs) * 100 : 0;
                            document.getElementById('progress-bar').style.width = `${progress}%`;
                            document.getElementById('progress-text').textContent = `Fortschritt: ${progress.toFixed(1)}%`;
                            
                            // Aktualisiere Hardware-Statistiken
                            const hardwareStats = document.getElementById('hardware-stats');
                            if (data.hardware_stats) {
                                hardwareStats.innerHTML = `
                                    <div class="hardware-stat">
                                        <span>CPU-Auslastung:</span>
                                        <span>${data.hardware_stats.cpu_usage.toFixed(1)}%</span>
                                    </div>
                                    <div class="hardware-stat">
                                        <span>Speichernutzung:</span>
                                        <span>${data.hardware_stats.memory_usage.toFixed(1)}%</span>
                                    </div>
                                    <div class="hardware-stat">
                                        <span>GPU-Auslastung:</span>
                                        <span>${data.hardware_stats.gpu_usage.toFixed(1)}%</span>
                                    </div>
                                    <div class="hardware-stat">
                                        <span>Gerät:</span>
                                        <span>${data.hardware_stats.device_name || '-'}</span>
                                    </div>
                                    <div class="hardware-stat">
                                        <span>MLX verfügbar:</span>
                                        <span>${data.hardware_stats.mlx_available ? 'Ja' : 'Nein'}</span>
                                    </div>
                                `;
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
                                        hours.toString().padStart(2, '0') + ':' + minutes.toString().padStart(2, '0') + ':' + seconds.toString().padStart(2, '0');
                                    
                                    // Geschätzte Restzeit
                                    if (data.current_epoch > 0 && data.total_epochs > 0) {
                                        const secondsPerEpoch = runtime / data.current_epoch;
                                        const remainingEpochs = data.total_epochs - data.current_epoch;
                                        const eta = Math.floor(secondsPerEpoch * remainingEpochs);
                                        
                                        const etaHours = Math.floor(eta / 3600);
                                        const etaMinutes = Math.floor((eta % 3600) / 60);
                                        const etaSeconds = eta % 60;
                                        document.getElementById('eta').textContent = 
                                            etaHours.toString().padStart(2, '0') + ':' + etaMinutes.toString().padStart(2, '0') + ':' + etaSeconds.toString().padStart(2, '0');
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
                                        .replace(/\b\w/g, function(l) { return l.toUpperCase(); });
                                    
                                    metricsHtml += '
                                        <div class="metric">
                                            <span class="metric-name">' + formattedMetric + ':</span>
                                            <span class="metric-value">' + metrics[metric].toFixed(4) + '</span>
                                        </div>
                                    ';
                                }
                                
                                card.innerHTML = metricsHtml;
                                componentMetricsContainer.appendChild(card);
                            }
                        })
                        .catch(error => {
                            console.error('Fehler beim Aktualisieren des Dashboards:', error);
                        });
                }
                
                // Aktualisiere das Dashboard initial und dann alle 2 Sekunden
                updateDashboard();
                setInterval(updateDashboard, 2000);
                
                // Event-Handler für Buttons
                document.getElementById('start-btn').addEventListener('click', function() {
                    fetch('/api/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ config: 'default' })
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
                    // Hier würde normalerweise das Konfigurations-Dashboard geöffnet werden
                });
            </script>
        </body>
        </html>
        """
        
        self.wfile.write(html.encode())
    
    def _serve_status(self):
        """Liefert den aktuellen Trainingsstatus als JSON."""
        self._set_headers("application/json")
        status = training_manager.state.to_dict()
        self.wfile.write(json.dumps(status).encode())
    
    def _serve_static(self):
        """Liefert statische Dateien."""
        # Implementierung für statische Dateien (CSS, JS, etc.)
        self._serve_not_found()
    
    def _serve_not_found(self):
        """Liefert eine 404-Fehlerseite."""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<html><body><h1>404 Not Found</h1></body></html>")
    
    def _handle_start_training(self, post_data):
        """Behandelt die Anfrage zum Starten des Trainings."""
        self._set_headers("application/json")
        try:
            data = json.loads(post_data)
            config_name = data.get('config')
            component = data.get('component')
            success = training_manager.start_training(config_name, component)
            self.wfile.write(json.dumps({"success": success}).encode())
        except Exception as e:
            logger.error(f"Fehler beim Starten des Trainings: {e}")
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())
    
    def _handle_stop_training(self):
        """Behandelt die Anfrage zum Stoppen des Trainings."""
        self._set_headers("application/json")
        try:
            success = training_manager.stop_training()
            self.wfile.write(json.dumps({"success": success}).encode())
        except Exception as e:
            logger.error(f"Fehler beim Stoppen des Trainings: {e}")
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())

def run_server(port=8080):
    """Startet den Dashboard-Server."""
    server_address = ('localhost', port)
    httpd = HTTPServer(server_address, DashboardHandler)
    
    logger.info(f"MISO Ultimate Training Dashboard wird gestartet auf http://localhost:{port}")
    print("="*80)
    print(f"MISO Ultimate Training Dashboard wird gestartet auf http://localhost:{port}")
    print("Das Dashboard wird in Ihrem Standardbrowser geöffnet.")
    print("Drücken Sie Strg+C, um das Dashboard zu beenden.")
    print("="*80)
    
    # Öffne das Dashboard im Browser
    webbrowser.open(f"http://localhost:{port}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server beendet")
        print("\nServer beendet")

if __name__ == "__main__":
    # Erstelle eine Beispielkonfiguration, falls keine existiert
    config_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "config"
    config_dir.mkdir(exist_ok=True)
    
    default_config_path = config_dir / "default.json"
    if not default_config_path.exists():
        default_config = {
            "name": "Standard-Konfiguration",
            "description": "Standardkonfiguration für MISO Ultimate AGI Training",
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "components": ["MISO_CORE", "VX_MEMEX", "VX_REASON", "VX_INTENT"],
            "advanced_options": {
                "use_focal_loss": True,
                "label_smoothing": 0.1,
                "mixed_precision": True,
                "use_mlx": True,
                "residual_blocks": True,
                "attention_layers": True
            }
        }
        
        with open(default_config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    # Starte den Server
    run_server(8080)
