#!/usr/bin/env python3
"""
MISO Training Local Starter
Startet das lokale MISO Training Dashboard und integriert es mit VXOR
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def find_miso_dashboard():
    """Findet das MISO Dashboard"""
    possible_paths = [
        "../miso_dashboard_simple.py",
        "../../miso_dashboard_simple.py", 
        "../MISO_Training/dashboard.py",
        "miso_dashboard_simple.py"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return Path(path).resolve()
    
    return None

def start_miso_training_local():
    """Startet MISO Training Local"""
    print("🔵 MISO TRAINING LOCAL STARTER")
    print("=" * 50)
    
    # Finde MISO Dashboard
    miso_dashboard = find_miso_dashboard()
    
    if miso_dashboard:
        print(f"✅ MISO Dashboard gefunden: {miso_dashboard}")
        
        try:
            print("🚀 Starte MISO Dashboard...")
            
            # Starte MISO Dashboard auf Port 8082
            miso_process = subprocess.Popen([
                sys.executable, str(miso_dashboard), "--port", "8082"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            time.sleep(3)  # Warte auf Start
            
            if miso_process.poll() is None:  # Prozess läuft noch
                print("✅ MISO Dashboard gestartet auf http://localhost:8082")
                
                # Starte auch integriertes Dashboard
                print("🚀 Starte integriertes MISO + VXOR Dashboard...")
                
                integrated_process = subprocess.Popen([
                    sys.executable, "agi_missions/integrated_miso_dashboard.py", "--port", "8081"
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                time.sleep(3)
                
                if integrated_process.poll() is None:
                    print("✅ Integriertes Dashboard gestartet auf http://localhost:8081")
                    
                    print("\n🎯 DASHBOARDS VERFÜGBAR:")
                    print("🔵 MISO Training:     http://localhost:8082")
                    print("🟢 VXOR AGI:          http://localhost:8080") 
                    print("🌐 Integriert:        http://localhost:8081")
                    
                    # Öffne Browser
                    choice = input("\n🌐 Browser öffnen? (1=MISO, 2=VXOR, 3=Integriert, n=Nein): ").strip()
                    
                    if choice == "1":
                        webbrowser.open("http://localhost:8082")
                    elif choice == "2":
                        webbrowser.open("http://localhost:8080")
                    elif choice == "3":
                        webbrowser.open("http://localhost:8081")
                    
                    print("\n⚡ Drücken Sie Ctrl+C zum Beenden aller Dashboards")
                    
                    try:
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\n🛑 Beende alle Dashboards...")
                        miso_process.terminate()
                        integrated_process.terminate()
                        print("✅ Alle Dashboards gestoppt")
                
                else:
                    print("❌ Integriertes Dashboard konnte nicht gestartet werden")
                    miso_process.terminate()
            
            else:
                print("❌ MISO Dashboard konnte nicht gestartet werden")
                stdout, stderr = miso_process.communicate()
                print(f"Error: {stderr.decode()}")
                
        except Exception as e:
            print(f"❌ Fehler beim Starten: {e}")
    
    else:
        print("❌ MISO Dashboard nicht gefunden")
        print("Verfügbare Alternativen:")
        print("1. Nur VXOR AGI Dashboard: python3 agi_missions/simple_web_dashboard.py")
        print("2. Integriertes Dashboard: python3 agi_missions/integrated_miso_dashboard.py")

def create_miso_training_symlink():
    """Erstellt Symlink zum MISO Training Verzeichnis"""
    miso_training_path = Path("../MISO_Training")
    local_link = Path("agi_missions/MISO_Training_Link")
    
    if miso_training_path.exists() and not local_link.exists():
        try:
            local_link.symlink_to(miso_training_path.resolve())
            print(f"✅ MISO Training Link erstellt: {local_link}")
        except Exception as e:
            print(f"⚠️ Konnte Link nicht erstellen: {e}")

def show_dashboard_status():
    """Zeigt Status aller Dashboards"""
    import requests
    
    dashboards = {
        "VXOR AGI": "http://localhost:8080",
        "Integriert": "http://localhost:8081", 
        "MISO Training": "http://localhost:8082"
    }
    
    print("\n📊 DASHBOARD STATUS:")
    print("-" * 40)
    
    for name, url in dashboards.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ {name:15} | {url}")
            else:
                print(f"⚠️ {name:15} | {url} (Status: {response.status_code})")
        except:
            print(f"❌ {name:15} | {url} (Nicht erreichbar)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MISO Training Local Starter")
    parser.add_argument("--status", action="store_true", help="Zeige Dashboard Status")
    parser.add_argument("--link", action="store_true", help="Erstelle MISO Training Link")
    
    args = parser.parse_args()
    
    if args.status:
        show_dashboard_status()
    elif args.link:
        create_miso_training_symlink()
    else:
        start_miso_training_local()
