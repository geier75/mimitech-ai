#!/usr/bin/env python3
"""
AGI Training Web Dashboard
Web-basiertes Dashboard für localhost-Monitoring
"""

import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
from agi_training_mission import AGITrainingMission

class AGIWebDashboard:
    """Web-Dashboard für AGI-Training"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.trainer = AGITrainingMission()
        self.training_results_dir = Path("agi_missions/training_results")
        self.training_results_dir.mkdir(parents=True, exist_ok=True)
        self.server = None
        self.server_thread = None
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Sammelt alle Dashboard-Daten"""
        # Lade Training-Ergebnisse
        training_files = list(self.training_results_dir.glob("training_result_*.json"))
        
        training_history = []
        for file_path in training_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    training_history.append(data)
            except Exception as e:
                print(f"Fehler beim Lesen von {file_path}: {e}")
        
        # Sortiere nach Zeit
        training_history.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        # System Status
        system_status = self.trainer.get_training_status()
        
        # Berechne Statistiken
        total_sessions = len(training_history)
        avg_success_rate = sum([h.get('success_rate', 0) for h in training_history]) / max(1, total_sessions)
        total_new_capabilities = sum([len(h.get('new_capabilities', [])) for h in training_history])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": system_status,
            "training_history": training_history[:10],  # Letzte 10 Sessions
            "statistics": {
                "total_sessions": total_sessions,
                "avg_success_rate": avg_success_rate,
                "total_new_capabilities": total_new_capabilities
            },
            "capabilities": system_status["current_capabilities"],
            "recommendations": self._generate_recommendations(system_status)
        }
    
    def _generate_recommendations(self, status: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generiert Empfehlungen"""
        recommendations = []
        capabilities = status["current_capabilities"]
        
        # Schwache Bereiche
        weak_areas = [(name, score) for name, score in capabilities.items() if score < 0.80]
        weak_areas.sort(key=lambda x: x[1])
        
        for name, score in weak_areas[:3]:
            recommendations.append({
                "type": "improvement",
                "title": f"Verbessere {name.replace('_', ' ').title()}",
                "description": f"Aktuelle Performance: {score:.1%}. Empfohlenes Training: Spezialisierte Sessions.",
                "priority": "high" if score < 0.75 else "medium"
            })
        
        # Allgemeine Empfehlungen
        avg_score = sum(capabilities.values()) / len(capabilities)
        if avg_score > 0.90:
            recommendations.append({
                "type": "optimization",
                "title": "System-Performance Exzellent",
                "description": "Fokus auf spezialisierte Trainings für Expertenniveau.",
                "priority": "low"
            })
        elif avg_score < 0.80:
            recommendations.append({
                "type": "urgent",
                "title": "Intensives Training Erforderlich",
                "description": "System-Performance unter Zielwert. Comprehensive Training empfohlen.",
                "priority": "high"
            })
        
        return recommendations

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler für Dashboard"""

    dashboard_instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_dashboard()
        elif self.path == '/api/data':
            self.serve_api_data()
        elif self.path == '/api/trigger-training':
            self.trigger_training()
        elif self.path.startswith('/static/'):
            self.serve_static()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        """Serve main dashboard HTML"""
        html_content = self.generate_dashboard_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_api_data(self):
        """Serve dashboard data as JSON"""
        data = self.dashboard_instance.get_dashboard_data()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def trigger_training(self):
        """Trigger immediate training"""
        try:
            # Parse query parameters
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            training_type = params.get('type', ['comprehensive'])[0]
            
            # Trigger training in background
            def run_training():
                from continuous_agi_training_daemon import ContinuousAGITrainingDaemon
                daemon = ContinuousAGITrainingDaemon()
                daemon.run_immediate_training(training_type)
            
            threading.Thread(target=run_training, daemon=True).start()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": f"{training_type} training started"}).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode())
    
    def generate_dashboard_html(self):
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 VXOR AGI Training Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header .subtitle { opacity: 0.8; font-size: 1.1em; }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { 
            background: rgba(255,255,255,0.1); 
            border-radius: 15px; 
            padding: 20px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .card h3 { margin-bottom: 15px; font-size: 1.3em; }
        
        .capability-bar {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .capability-name {
            width: 150px;
            font-size: 0.9em;
        }
        .progress-bar {
            flex: 1;
            height: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            overflow: hidden;
            margin: 0 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s ease;
        }
        .capability-score {
            width: 50px;
            text-align: right;
            font-weight: bold;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }
        .stat-item {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .training-item {
            padding: 10px;
            margin-bottom: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }
        
        .recommendation {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 8px;
        }
        .rec-high { background: rgba(255,87,34,0.3); border-left: 4px solid #FF5722; }
        .rec-medium { background: rgba(255,193,7,0.3); border-left: 4px solid #FFC107; }
        .rec-low { background: rgba(76,175,80,0.3); border-left: 4px solid #4CAF50; }
        
        .controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            background: #4CAF50;
            color: white;
            cursor: pointer;
            font-size: 0.9em;
            transition: background 0.3s;
        }
        .btn:hover { background: #45a049; }
        .btn-secondary { background: #2196F3; }
        .btn-secondary:hover { background: #1976D2; }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-running { background: #4CAF50; animation: pulse 2s infinite; }
        .status-idle { background: #FFC107; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .last-updated {
            text-align: center;
            margin-top: 20px;
            opacity: 0.7;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 VXOR AGI Training Dashboard</h1>
            <div class="subtitle">
                <span class="status-indicator status-running"></span>
                Live-Monitoring der AGI-Entwicklung
            </div>
        </div>
        
        <div class="grid">
            <!-- System Status -->
            <div class="card">
                <h3>📊 System Status</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="total-sessions">-</div>
                        <div class="stat-label">Sessions</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="success-rate">-</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="new-capabilities">-</div>
                        <div class="stat-label">Neue Fähigkeiten</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="avg-performance">-</div>
                        <div class="stat-label">Ø Performance</div>
                    </div>
                </div>
            </div>
            
            <!-- Capabilities -->
            <div class="card">
                <h3>🧠 AGI-Fähigkeiten</h3>
                <div id="capabilities-list">
                    <!-- Wird dynamisch gefüllt -->
                </div>
            </div>
            
            <!-- Training History -->
            <div class="card">
                <h3>📅 Training-Verlauf</h3>
                <div id="training-history">
                    <!-- Wird dynamisch gefüllt -->
                </div>
            </div>
            
            <!-- Recommendations -->
            <div class="card">
                <h3>💡 Empfehlungen</h3>
                <div id="recommendations">
                    <!-- Wird dynamisch gefüllt -->
                </div>
                
                <div class="controls">
                    <button class="btn" onclick="triggerTraining('comprehensive')">🚀 Comprehensive Training</button>
                    <button class="btn btn-secondary" onclick="triggerTraining('quantum')">⚛️ Quantum Training</button>
                    <button class="btn btn-secondary" onclick="triggerTraining('creative')">💡 Creative Training</button>
                </div>
            </div>
        </div>
        
        <div class="last-updated">
            Letzte Aktualisierung: <span id="last-updated">-</span>
        </div>
    </div>
    
    <script>
        let updateInterval;
        
        async function loadDashboardData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Fehler beim Laden der Daten:', error);
            }
        }
        
        function updateDashboard(data) {
            // System Status
            document.getElementById('total-sessions').textContent = data.statistics.total_sessions;
            document.getElementById('success-rate').textContent = (data.statistics.avg_success_rate * 100).toFixed(1) + '%';
            document.getElementById('new-capabilities').textContent = data.statistics.total_new_capabilities;
            
            const avgPerf = Object.values(data.capabilities).reduce((a, b) => a + b, 0) / Object.keys(data.capabilities).length;
            document.getElementById('avg-performance').textContent = (avgPerf * 100).toFixed(1) + '%';
            
            // Capabilities
            const capabilitiesList = document.getElementById('capabilities-list');
            capabilitiesList.innerHTML = '';
            
            Object.entries(data.capabilities)
                .sort((a, b) => b[1] - a[1])
                .forEach(([name, score]) => {
                    const div = document.createElement('div');
                    div.className = 'capability-bar';
                    div.innerHTML = `
                        <div class="capability-name">${name.replace(/_/g, ' ')}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${score * 100}%"></div>
                        </div>
                        <div class="capability-score">${(score * 100).toFixed(1)}%</div>
                    `;
                    capabilitiesList.appendChild(div);
                });
            
            // Training History
            const historyDiv = document.getElementById('training-history');
            historyDiv.innerHTML = '';
            
            data.training_history.slice(0, 5).forEach(session => {
                const div = document.createElement('div');
                div.className = 'training-item';
                const startTime = new Date(session.start_time).toLocaleString();
                div.innerHTML = `
                    <strong>${session.training_type}</strong><br>
                    <small>${startTime} | Success: ${(session.success_rate * 100).toFixed(1)}% | +${session.new_capabilities.length} Fähigkeiten</small>
                `;
                historyDiv.appendChild(div);
            });
            
            // Recommendations
            const recDiv = document.getElementById('recommendations');
            recDiv.innerHTML = '';
            
            data.recommendations.forEach(rec => {
                const div = document.createElement('div');
                div.className = `recommendation rec-${rec.priority}`;
                div.innerHTML = `
                    <strong>${rec.title}</strong><br>
                    <small>${rec.description}</small>
                `;
                recDiv.appendChild(div);
            });
            
            // Last Updated
            document.getElementById('last-updated').textContent = new Date(data.timestamp).toLocaleString();
        }
        
        async function triggerTraining(type) {
            try {
                const response = await fetch(`/api/trigger-training?type=${type}`);
                const result = await response.json();
                
                if (result.status === 'success') {
                    alert(`${type} Training gestartet!`);
                    // Aktualisiere Dashboard nach kurzer Verzögerung
                    setTimeout(loadDashboardData, 2000);
                } else {
                    alert(`Fehler: ${result.message}`);
                }
            } catch (error) {
                alert(`Fehler beim Starten des Trainings: ${error.message}`);
            }
        }
        
        // Initial load
        loadDashboardData();
        
        // Auto-refresh every 30 seconds
        updateInterval = setInterval(loadDashboardData, 30000);
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (updateInterval) clearInterval(updateInterval);
        });
    </script>
</body>
</html>
        """
    
    def start_server(self):
        """Startet den Web-Server"""
        DashboardHandler.dashboard_instance = self

        self.server = HTTPServer(('localhost', self.port), DashboardHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        print(f"🌐 AGI Training Web-Dashboard gestartet!")
        print(f"📊 URL: http://localhost:{self.port}")
        print(f"🔄 Auto-Refresh: alle 30 Sekunden")
        print(f"⚡ Live-Training Controls verfügbar")
    
    def stop_server(self):
        """Stoppt den Web-Server"""
        if self.server:
            self.server.shutdown()
            print("🛑 Web-Dashboard gestoppt")

def main():
    """Hauptfunktion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AGI Training Web Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port für Web-Server")
    
    args = parser.parse_args()
    
    dashboard = AGIWebDashboard(port=args.port)
    
    try:
        dashboard.start_server()
        
        print("\n🎯 DASHBOARD BEREIT!")
        print("Drücken Sie Ctrl+C zum Beenden")
        
        # Halte Server am Leben
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Beende Web-Dashboard...")
        dashboard.stop_server()

if __name__ == "__main__":
    main()
