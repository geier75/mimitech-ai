#!/usr/bin/env python3
"""
🚀 MISO ULTIMATE - BACKEND STARTUP SCRIPT
========================================

Startet das Backend und öffnet automatisch das Dashboard.
Prüft Dependencies und installiert fehlende Pakete.

Usage: python start_backend.py
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def check_and_install_dependencies():
    """Prüft und installiert benötigte Dependencies"""
    print("🔍 Prüfe Dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn[standard]",
        "psutil",
        "pydantic",
        "websockets"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "uvicorn[standard]":
                __import__("uvicorn")
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - FEHLT")
    
    if missing_packages:
        print(f"\n📦 Installiere fehlende Pakete: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("✅ Installation abgeschlossen")
        except subprocess.CalledProcessError:
            print("❌ Installation fehlgeschlagen")
            print("💡 Versuche manuell: pip install fastapi uvicorn[standard] psutil pydantic websockets")
            return False
    
    return True

def start_backend_server():
    """Startet den Backend-Server"""
    print("\n🚀 Starte MISO Ultimate Backend Server...")
    
    backend_file = Path(__file__).parent / "benchmark_backend_server.py"
    
    if not backend_file.exists():
        print(f"❌ Backend-Datei nicht gefunden: {backend_file}")
        return None
    
    try:
        # Backend-Server starten
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "benchmark_backend_server:app",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--reload"
        ], cwd=backend_file.parent)
        
        print("✅ Backend-Server gestartet")
        print("🔗 API: http://127.0.0.1:8000")
        print("📚 Docs: http://127.0.0.1:8000/docs")
        
        return process
        
    except Exception as e:
        print(f"❌ Fehler beim Starten des Servers: {e}")
        return None

def open_dashboard():
    """Öffnet das Dashboard im Browser"""
    print("\n🌐 Öffne Dashboard...")
    
    dashboard_url = "http://127.0.0.1:5151/benchmark_dashboard.html"
    
    try:
        # Warte kurz, damit der Server startet
        time.sleep(3)
        webbrowser.open(dashboard_url)
        print(f"✅ Dashboard geöffnet: {dashboard_url}")
    except Exception as e:
        print(f"❌ Fehler beim Öffnen des Dashboards: {e}")
        print(f"💡 Öffne manuell: {dashboard_url}")

def main():
    """Hauptfunktion"""
    print("=" * 50)
    print("🎯 MISO ULTIMATE - BACKEND STARTUP")
    print("=" * 50)
    
    # 1. Dependencies prüfen
    if not check_and_install_dependencies():
        print("\n❌ Startup abgebrochen - Dependencies fehlen")
        return
    
    # 2. Backend starten
    server_process = start_backend_server()
    if not server_process:
        print("\n❌ Startup abgebrochen - Server konnte nicht gestartet werden")
        return
    
    # 3. Dashboard öffnen
    open_dashboard()
    
    print("\n" + "=" * 50)
    print("✅ MISO ULTIMATE BACKEND LÄUFT")
    print("=" * 50)
    print("🔗 Backend API: http://127.0.0.1:8000")
    print("📊 Dashboard: http://127.0.0.1:5151/benchmark_dashboard.html")
    print("📚 API Docs: http://127.0.0.1:8000/docs")
    print("\n💡 Drücke Ctrl+C zum Beenden")
    
    try:
        # Warte auf Benutzer-Unterbrechung
        server_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown-Signal empfangen...")
        server_process.terminate()
        server_process.wait()
        print("✅ Backend-Server beendet")

if __name__ == "__main__":
    main()
