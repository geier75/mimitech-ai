#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Direktes Befehlssystem Demo

Diese Demo zeigt, wie MISO Befehle direkt befolgt und die erweiterten
Funktionen wie Trading, Videoschnitt, Bildgenerierung, Posting und
Computer Vision nutzt.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("MISO.CommandDemo")

# Stelle sicher, dass das MISO-Paket im Pfad ist
parent_dir = str(Path(__file__).parent)
sys.path.insert(0, parent_dir)

# Importiere die ComputerControl-Klasse direkt
try:
    # Versuche direkten Import der ComputerControl-Klasse
    import importlib.util
    computer_control_path = os.path.join(parent_dir, "miso", "control", "computer_control.py")
    
    if os.path.exists(computer_control_path):
        spec = importlib.util.spec_from_file_location("computer_control", computer_control_path)
        computer_control_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(computer_control_module)
        ComputerControl = computer_control_module.ComputerControl
        logger.info("ComputerControl-Modul erfolgreich importiert")
    else:
        logger.error(f"Computer Control Modul nicht gefunden unter: {computer_control_path}")
        sys.exit(1)
except Exception as e:
    logger.error(f"Fehler beim Import des ComputerControl-Moduls: {e}")
    sys.exit(1)

# Da wir Probleme mit den relativen Importen haben, implementieren wir eine vereinfachte Version
# des CommandExecutor und MISODirectCommandSystem direkt in dieser Datei
import json
import threading
import queue
from enum import Enum, auto

class CommandPriority(Enum):
    """Prioritätsstufen für Befehle"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class CommandStatus(Enum):
    """Status eines Befehls"""
    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELED = auto()

class Command:
    """Repräsentiert einen Befehl, der von MISO ausgeführt werden soll"""
    
    def __init__(self, command_text, priority=CommandPriority.MEDIUM):
        """Initialisiert einen neuen Befehl"""
        self.command_text = command_text
        self.priority = priority
        self.status = CommandStatus.PENDING
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.id = self._generate_id()
    
    def _generate_id(self):
        """Generiert eine eindeutige ID für den Befehl"""
        import uuid
        return str(uuid.uuid4())
    
    def __str__(self):
        return f"Command({self.id}, '{self.command_text}', {self.status.name})"

class SimplifiedCommandExecutor:
    """Vereinfachte Version des CommandExecutor für die Demo"""
    
    def __init__(self, config=None):
        """Initialisiert den CommandExecutor"""
        self.config = config or {}
        self.command_queue = queue.PriorityQueue()
        self.command_history = []
        self.running = False
        self.executor_thread = None
        
        # Lade ComputerControl
        self.computer_control = ComputerControl(
            self.config.get('computer_control_config', {'security_level': 'low'})
        )
        
        logger.info("SimplifiedCommandExecutor initialisiert")
    
    def start(self):
        """Startet den CommandExecutor"""
        if self.running:
            logger.warning("CommandExecutor läuft bereits")
            return False
        
        self.running = True
        self.executor_thread = threading.Thread(target=self._executor_loop)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        
        logger.info("CommandExecutor gestartet")
        return True
    
    def stop(self):
        """Stoppt den CommandExecutor"""
        if not self.running:
            logger.warning("CommandExecutor läuft nicht")
            return False
        
        self.running = False
        if self.executor_thread:
            self.executor_thread.join(timeout=5.0)
        
        logger.info("CommandExecutor gestoppt")
        return True
    
    def _executor_loop(self):
        """Hauptschleife für die Befehlsausführung"""
        logger.info("Executor-Schleife gestartet")
        
        while self.running:
            try:
                # Hole den nächsten Befehl aus der Warteschlange
                priority, command = self.command_queue.get(block=True, timeout=1.0)
                
                # Führe den Befehl aus
                logger.info(f"Führe Befehl aus: {command}")
                self._execute_command(command)
                
                # Füge den Befehl zur Historie hinzu
                self.command_history.append(command)
                
                # Markiere den Befehl als erledigt
                self.command_queue.task_done()
                
            except queue.Empty:
                # Keine Befehle in der Warteschlange
                pass
            except Exception as e:
                logger.error(f"Fehler in der Executor-Schleife: {e}")
        
        logger.info("Executor-Schleife beendet")
    
    def _execute_command(self, command):
        """Führt einen Befehl aus"""
        command.status = CommandStatus.EXECUTING
        command.start_time = time.time()
        
        try:
            command.result = self.process_command(command.command_text)
            command.status = CommandStatus.COMPLETED
        except Exception as e:
            command.error = str(e)
            command.status = CommandStatus.FAILED
            logger.error(f"Fehler bei der Ausführung des Befehls '{command.command_text}': {e}")
        
        command.end_time = time.time()
    
    def add_command(self, command_text, priority=CommandPriority.MEDIUM):
        """Fügt einen Befehl zur Ausführungswarteschlange hinzu"""
        command = Command(command_text, priority)
        self.command_queue.put((priority.value, command))
        logger.info(f"Befehl hinzugefügt: {command}")
        return command
    
    def process_command(self, command_text):
        """Verarbeitet einen Befehl"""
        logger.info(f"Verarbeite Befehl: '{command_text}'")
        
        # Einfache Befehlsverarbeitung für die Demo
        command_lower = command_text.lower()
        
        # Computersteuerungsbefehle
        if "maus" in command_lower and ("bewege" in command_lower or "position" in command_lower):
            # Extrahiere Koordinaten
            import re
            coords_match = re.search(r'\(?(\d+)\s*,\s*(\d+)\)?', command_text)
            if coords_match:
                x, y = int(coords_match.group(1)), int(coords_match.group(2))
                self.computer_control.move_mouse(x, y, duration=1.0)
                return {"status": "success", "message": f"Maus zu Position ({x}, {y}) bewegt"}
        
        elif "klick" in command_lower:
            # Führe Klick aus
            self.computer_control.click()
            return {"status": "success", "message": "Mausklick ausgeführt"}
        
        elif "tippe" in command_lower or "schreibe" in command_lower:
            # Extrahiere Text
            import re
            text_match = re.search(r'["\']([^"\']+)["\']', command_text)
            if text_match:
                text = text_match.group(1)
                self.computer_control.type_text(text)
                return {"status": "success", "message": f"Text '{text}' eingegeben"}
        
        # Trading-Befehle
        elif any(word in command_lower for word in ["trade", "handel", "kauf", "verkauf", "aktie"]):
            return {"status": "success", "message": f"Trading-Befehl simuliert: {command_text}", "type": "trading"}
        
        # Videoschnitt-Befehle
        elif any(word in command_lower for word in ["video", "schnitt", "schneide", "film"]):
            return {"status": "success", "message": f"Videoschnitt-Befehl simuliert: {command_text}", "type": "video"}
        
        # Bildgenerierungs-Befehle
        elif any(word in command_lower for word in ["bild", "generier", "erstell", "zeichne"]):
            return {"status": "success", "message": f"Bildgenerierungs-Befehl simuliert: {command_text}", "type": "image"}
        
        # Posting-Befehle
        elif any(word in command_lower for word in ["post", "veröffentlich", "teile", "social"]):
            return {"status": "success", "message": f"Posting-Befehl simuliert: {command_text}", "type": "posting"}
        
        # Computer Vision-Befehle
        elif any(word in command_lower for word in ["erkennen", "vision", "analysier", "scan"]):
            return {"status": "success", "message": f"Computer Vision-Befehl simuliert: {command_text}", "type": "vision"}
        
        # Fallback
        return {"status": "success", "message": f"Befehl verstanden: {command_text}", "type": "generic"}

class SimplifiedDirectCommandSystem:
    """Vereinfachte Version des MISODirectCommandSystem für die Demo"""
    
    def __init__(self, config=None):
        """Initialisiert das SimplifiedDirectCommandSystem"""
        self.config = config or {}
        self.command_executor = SimplifiedCommandExecutor(self.config)
        self.command_executor.start()
        
        logger.info("SimplifiedDirectCommandSystem initialisiert")
    
    def execute_command(self, command_text):
        """Führt einen Befehl direkt aus"""
        logger.info(f"DIREKTER BEFEHL: {command_text}")
        
        try:
            result = self.command_executor.process_command(command_text)
            logger.info(f"Befehl direkt ausgeführt: {result}")
            return {"status": "success", "message": "Befehl direkt ausgeführt", "result": result}
        except Exception as e:
            logger.error(f"Fehler bei direkter Befehlsausführung: {e}")
            return {"status": "error", "message": f"Fehler bei direkter Befehlsausführung: {e}"}
    
    def run_interactive(self):
        """Startet eine interaktive Befehlsschnittstelle"""
        print("\nInteraktiver Modus gestartet. Geben Sie 'exit' ein, um zu beenden.")
        
        while True:
            try:
                command = input("\nMISO> ")
                
                if command.lower() in ['exit', 'quit', 'ende', 'beenden']:
                    break
                
                result = self.execute_command(command)
                
                if result.get("status") == "success":
                    print("\n✅ Befehl erfolgreich ausgeführt:")
                    print(f"   {result.get('result', {}).get('message', '')}")
                else:
                    print("\n❌ Fehler bei der Befehlsausführung:")
                    print(f"   {result.get('message', 'Unbekannter Fehler')}")
                
            except KeyboardInterrupt:
                print("\nInteraktiver Modus wird beendet...")
                break
            except Exception as e:
                print(f"\n❌ Fehler: {e}")
        
        print("\nInteraktiver Modus beendet.")
    
    def stop(self):
        """Stoppt das SimplifiedDirectCommandSystem"""
        self.command_executor.stop()
        logger.info("SimplifiedDirectCommandSystem gestoppt")

def main():
    """Hauptfunktion für die Demo"""
    print("\n" + "=" * 80)
    print(" MISO DIREKTES BEFEHLSSYSTEM DEMO ".center(80, "="))
    print("=" * 80 + "\n")
    
    print("Diese Demo zeigt, wie MISO Befehle direkt befolgt und")
    print("die erweiterten Funktionen wie Trading, Videoschnitt,")
    print("Bildgenerierung, Posting und Computer Vision nutzt.")
    print("\nPRIORITÄT 1.1: BEFOLGE BEFEHLE ist aktiviert.")
    print("\nMISO wird Ihre Befehle direkt ausführen.")
    
    # Erstelle und starte SimplifiedDirectCommandSystem
    command_system = SimplifiedDirectCommandSystem()
    
    # Demo-Befehle
    demo_commands = [
        "Bewege die Maus zur Position (500, 500)",
        "Generiere ein Bild von einem Sonnenuntergang am Meer",
        "Schneide das Video example.mp4 und entferne die ersten 10 Sekunden",
        "Analysiere das Bild screenshot.png und erkenne Objekte",
        "Kaufe 10 Aktien von Apple",
        "Poste auf Twitter: 'MISO kann jetzt den Computer steuern!'"
    ]
    
    print("\nBeispiel-Befehle, die MISO ausführen kann:")
    for i, cmd in enumerate(demo_commands, 1):
        print(f"{i}. {cmd}")
    
    print("\nSie können diese Befehle direkt eingeben oder eigene Befehle formulieren.")
    print("Geben Sie 'exit' ein, um die Demo zu beenden.\n")
    
    try:
        # Starte interaktiven Modus
        command_system.run_interactive()
    except KeyboardInterrupt:
        print("\nDemo wird beendet...")
    finally:
        command_system.stop()
        print("\nDemo beendet.")

if __name__ == "__main__":
    main()
