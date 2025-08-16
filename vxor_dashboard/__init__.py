"""
vXor AGI-System: vxor_dashboard
Dashboard für Benchmark-Ergebnisse und System-Performance

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
from pathlib import Path

# Füge das übergeordnete Verzeichnis zum Pfad hinzu, um vxor_dashboard.py zu importieren
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importiere die Funktionen aus der vxor_dashboard.py Datei
try:
    from vxor_dashboard import run_server, init, boot, collect_benchmark_results
    
    # Setze Logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - VXOR.Dashboard.Package - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info("vXor Dashboard Paket erfolgreich initialisiert")
    
except ImportError as e:
    # Fallback-Implementierung, falls die Hauptimplementierung nicht verfügbar ist
    import threading
    import webbrowser
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    
    # Setze Logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - VXOR.Dashboard.Package - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.error(f"Fehler beim Importieren von vxor_dashboard.py: {e}")
    
    # Fallback-Implementierung
    def serve_benchmark_dashboard(port):
        class DashboardHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                self.directory = Path(__file__).parent.parent.absolute()
                super().__init__(*args, directory=str(self.directory), **kwargs)
        
        server = HTTPServer(("127.0.0.1", port), DashboardHandler)
        logger.info(f"Fallback-Server gestartet auf Port {port}")
        server.serve_forever()
    
    def run_server(port=5151):
        logger.info("Starte Fallback-Dashboard-Server")
        server_thread = threading.Thread(target=serve_benchmark_dashboard, args=(port,), daemon=True)
        server_thread.start()
        
        # Öffne Browser
        dashboard_url = f"http://127.0.0.1:{port}/benchmark_dashboard.html"
        webbrowser.open(dashboard_url)
        logger.info(f"Dashboard geöffnet im Browser: {dashboard_url}")
        
        try:
            # Halte den Hauptthread am Leben
            while True:
                server_thread.join(1)
        except KeyboardInterrupt:
            logger.info("Server wird beendet (Strg+C)")
        
        return True
    
    def init():
        logger.info("Modul vxor_dashboard initialisiert (Fallback)")
        return True
    
    def boot():
        logger.info("Modul vxor_dashboard gestartet (Fallback)")
        return init()
    
    def collect_benchmark_results():
        return {
            "status": "error",
            "message": "Hauptimplementierung nicht verfügbar"
        }
