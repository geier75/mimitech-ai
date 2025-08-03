#!/usr/bin/env python3
"""
Simple AGI Training Web Dashboard
Einfaches Web-Dashboard für localhost-Monitoring
"""

import json
import time
import threading
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from agi_training_mission import AGITrainingMission

# Global dashboard instance
dashboard_data = {}

class SimpleHandler(BaseHTTPRequestHandler):
    """Einfacher HTTP Handler"""
    
    def do_GET(self):
        if self.path == '/':
            self.serve_html()
        elif self.path == '/data':
            self.serve_data()
        else:
            self.send_error(404)
    
    def serve_html(self):
        """Serve HTML Dashboard"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>🧠 AGI Training Dashboard</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: white; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #4CAF50; font-size: 2.5em; margin-bottom: 10px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #16213e; padding: 20px; border-radius: 10px; border: 1px solid #4CAF50; }
        .card h3 { color: #4CAF50; margin-bottom: 15px; }
        .stat { display: flex; justify-content: space-between; margin: 10px 0; }
        .capability { display: flex; align-items: center; margin: 8px 0; }
        .capability-name { width: 150px; }
        .progress { flex: 1; height: 20px; background: #333; border-radius: 10px; margin: 0 10px; overflow: hidden; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); }
        .score { width: 50px; text-align: right; font-weight: bold; }
        .training-item { background: #0f1419; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #4CAF50; }
        .btn { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #45a049; }
        .status { text-align: center; margin: 20px 0; }
        .live-indicator { display: inline-block; width: 10px; height: 10px; background: #4CAF50; border-radius: 50%; animation: blink 2s infinite; }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0.3; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 VXOR AGI Training Dashboard</h1>
            <div class="status">
                <span class="live-indicator"></span> Live-Monitoring aktiv
                <br><small>Letzte Aktualisierung: <span id="last-update">-</span></small>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 System Status</h3>
                <div class="stat"><span>Training Sessions:</span><span id="sessions">-</span></div>
                <div class="stat"><span>Success Rate:</span><span id="success-rate">-</span></div>
                <div class="stat"><span>Neue Fähigkeiten:</span><span id="capabilities">-</span></div>
                <div class="stat"><span>Ø Performance:</span><span id="avg-perf">-</span></div>
            </div>
            
            <div class="card">
                <h3>🧠 AGI-Fähigkeiten</h3>
                <div id="capabilities-list"></div>
            </div>
            
            <div class="card">
                <h3>📅 Training-Verlauf</h3>
                <div id="training-history"></div>
            </div>
            
            <div class="card">
                <h3>⚡ Training Controls</h3>
                <button class="btn" onclick="triggerTraining('comprehensive')">🚀 Comprehensive</button>
                <button class="btn" onclick="triggerTraining('quantum')">⚛️ Quantum</button>
                <button class="btn" onclick="triggerTraining('creative')">💡 Creative</button>
                <button class="btn" onclick="triggerTraining('meta_learning')">🧠 Meta-Learning</button>
                <div style="margin-top: 15px;">
                    <button class="btn" onclick="location.reload()">🔄 Refresh</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        async function loadData() {
            try {
                const response = await fetch('/data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Fehler:', error);
            }
        }
        
        function updateDashboard(data) {
            document.getElementById('sessions').textContent = data.total_sessions || 0;
            document.getElementById('success-rate').textContent = ((data.avg_success_rate || 0) * 100).toFixed(1) + '%';
            document.getElementById('capabilities').textContent = data.total_new_capabilities || 0;
            
            const avgPerf = data.capabilities ? 
                Object.values(data.capabilities).reduce((a, b) => a + b, 0) / Object.keys(data.capabilities).length : 0;
            document.getElementById('avg-perf').textContent = (avgPerf * 100).toFixed(1) + '%';
            
            // Capabilities
            const capList = document.getElementById('capabilities-list');
            capList.innerHTML = '';
            if (data.capabilities) {
                Object.entries(data.capabilities)
                    .sort((a, b) => b[1] - a[1])
                    .forEach(([name, score]) => {
                        const div = document.createElement('div');
                        div.className = 'capability';
                        div.innerHTML = `
                            <div class="capability-name">${name.replace(/_/g, ' ')}</div>
                            <div class="progress">
                                <div class="progress-bar" style="width: ${score * 100}%"></div>
                            </div>
                            <div class="score">${(score * 100).toFixed(1)}%</div>
                        `;
                        capList.appendChild(div);
                    });
            }
            
            // Training History
            const historyDiv = document.getElementById('training-history');
            historyDiv.innerHTML = '';
            if (data.training_history) {
                data.training_history.slice(0, 5).forEach(session => {
                    const div = document.createElement('div');
                    div.className = 'training-item';
                    const time = new Date(session.start_time).toLocaleString();
                    div.innerHTML = `
                        <strong>${session.training_type || 'Unknown'}</strong><br>
                        <small>${time} | Success: ${((session.success_rate || 0) * 100).toFixed(1)}%</small>
                    `;
                    historyDiv.appendChild(div);
                });
            }
            
            document.getElementById('last-update').textContent = new Date().toLocaleString();
        }
        
        function triggerTraining(type) {
            alert(`${type} Training wird gestartet...`);
            // Hier könnte ein API-Call zum Starten des Trainings stehen
        }
        
        // Initial load
        loadData();
        
        // Auto-refresh every 30 seconds
        setInterval(loadData, 30000);
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_data(self):
        """Serve JSON data"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(dashboard_data, indent=2).encode())

def update_dashboard_data():
    """Update dashboard data"""
    global dashboard_data
    
    try:
        trainer = AGITrainingMission()
        
        # Load training results
        results_dir = Path("agi_missions/training_results")
        training_files = list(results_dir.glob("training_result_*.json")) if results_dir.exists() else []
        
        training_history = []
        for file_path in training_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    training_history.append(data)
            except:
                continue
        
        training_history.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        # Get system status
        status = trainer.get_training_status()
        
        # Calculate stats
        total_sessions = len(training_history)
        avg_success_rate = sum([h.get('success_rate', 0) for h in training_history]) / max(1, total_sessions)
        total_new_capabilities = sum([len(h.get('new_capabilities', [])) for h in training_history])
        
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "total_sessions": total_sessions,
            "avg_success_rate": avg_success_rate,
            "total_new_capabilities": total_new_capabilities,
            "capabilities": status["current_capabilities"],
            "training_history": training_history[:10]
        }
        
    except Exception as e:
        print(f"Fehler beim Update der Dashboard-Daten: {e}")
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "total_sessions": 0,
            "avg_success_rate": 0,
            "total_new_capabilities": 0,
            "capabilities": {},
            "training_history": []
        }

def start_dashboard(port=8080):
    """Start the web dashboard"""
    print(f"🌐 Starte AGI Training Web-Dashboard...")
    
    # Initial data update
    update_dashboard_data()
    
    # Start data update thread
    def update_loop():
        while True:
            time.sleep(30)  # Update every 30 seconds
            update_dashboard_data()
    
    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()
    
    # Start web server
    server = HTTPServer(('localhost', port), SimpleHandler)
    
    print(f"🎯 AGI Training Dashboard läuft!")
    print(f"📊 URL: http://localhost:{port}")
    print(f"🔄 Auto-Update: alle 30 Sekunden")
    print(f"⚡ Drücken Sie Ctrl+C zum Beenden")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Dashboard wird gestoppt...")
        server.shutdown()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple AGI Training Web Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port für Web-Server")
    
    args = parser.parse_args()
    start_dashboard(args.port)
