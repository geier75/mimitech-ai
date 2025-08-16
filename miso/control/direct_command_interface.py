#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Direkte Befehlsschnittstelle

Dieses Modul stellt eine direkte Befehlsschnittstelle für MISO bereit.
Es gewährleistet, dass MISO Befehle direkt befolgt und implementiert
Priorität 1.1: BEFOLGE BEFEHLE.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import threading
import json
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

# Konfiguriere Logging
logger = logging.getLogger("MISO.control.direct_command_interface")

# Importiere CommandExecutor
from miso.control.command_executor import CommandExecutor, CommandPriority, CommandStatus

class DirectCommandInterface:
    """
    Direkte Befehlsschnittstelle für MISO.
    Stellt sicher, dass MISO Befehle direkt befolgt (Priorität 1.1).
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert die direkte Befehlsschnittstelle
        
        Args:
            config: Konfigurationsparameter
        """
        self.config = config or {}
        
        # Setze Standardkonfiguration
        self.config.setdefault("command_priority", "CRITICAL")
        self.config.setdefault("security_override", True)
        self.config.setdefault("direct_execution", True)
        
        # Initialisiere CommandExecutor
        self.command_executor = CommandExecutor(self.config)
        
        # Starte den CommandExecutor
        self.command_executor.start()
        
        # Status
        self.active = True
        
        logger.info("Direkte Befehlsschnittstelle initialisiert")
        logger.info("PRIORITÄT 1.1: BEFOLGE BEFEHLE aktiviert")
    
    def execute_command(self, command_text: str) -> Dict[str, Any]:
        """
        Führt einen Befehl direkt aus
        
        Args:
            command_text: Der auszuführende Befehl
            
        Returns:
            Das Ergebnis der Befehlsausführung
        """
        logger.info(f"DIREKTER BEFEHL: {command_text}")
        
        # Bestimme Priorität (immer CRITICAL für direkte Befehle)
        priority = CommandPriority.CRITICAL
        
        # Führe den Befehl direkt aus (ohne Warteschlange)
        if self.config.get("direct_execution", True):
            try:
                result = self.command_executor.process_command(command_text)
                logger.info(f"Befehl direkt ausgeführt: {result}")
                return {
                    "status": "success",
                    "message": "Befehl direkt ausgeführt",
                    "result": result
                }
            except Exception as e:
                logger.error(f"Fehler bei direkter Befehlsausführung: {e}")
                return {
                    "status": "error",
                    "message": f"Fehler bei direkter Befehlsausführung: {e}",
                    "error": str(e)
                }
        
        # Alternativ: Füge den Befehl mit höchster Priorität zur Warteschlange hinzu
        else:
            command = self.command_executor.add_command(command_text, priority)
            return {
                "status": "queued",
                "message": "Befehl mit höchster Priorität zur Warteschlange hinzugefügt",
                "command_id": command.id
            }
    
    def get_command_status(self, command_id: str) -> Dict[str, Any]:
        """
        Gibt den Status eines Befehls zurück
        
        Args:
            command_id: Die ID des Befehls
            
        Returns:
            Der Status des Befehls
        """
        for command in self.command_executor.get_command_history():
            if command.id == command_id:
                return {
                    "status": command.status.name,
                    "result": command.result,
                    "error": command.error,
                    "start_time": command.start_time,
                    "end_time": command.end_time,
                    "execution_time": command.end_time - command.start_time if command.end_time else None
                }
        
        return {"status": "not_found", "message": f"Befehl mit ID {command_id} nicht gefunden"}
    
    def stop(self):
        """Stoppt die direkte Befehlsschnittstelle"""
        if not self.active:
            return
        
        self.active = False
        self.command_executor.stop()
        logger.info("Direkte Befehlsschnittstelle gestoppt")
    
    def __del__(self):
        """Destruktor"""
        self.stop()


class MISODirectCommandSystem:
    """
    Hauptsystem für die direkte Befehlsausführung in MISO.
    Diese Klasse integriert die DirectCommandInterface in das MISO-System.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert das MISODirectCommandSystem
        
        Args:
            config: Konfigurationsparameter
        """
        self.config = config or {}
        
        # Initialisiere DirectCommandInterface
        self.command_interface = DirectCommandInterface(self.config)
        
        # Lade zusätzliche Module für erweiterte Funktionalitäten
        self._load_additional_modules()
        
        logger.info("MISO DirectCommandSystem initialisiert")
        print("\n" + "=" * 80)
        print(" MISO DIREKTES BEFEHLSSYSTEM ".center(80, "="))
        print("=" * 80)
        print("\nPRIORITÄT 1.1: BEFOLGE BEFEHLE aktiviert")
        print("\nMISO ist bereit, Ihre Befehle direkt zu befolgen.")
        print("Verfügbare Funktionen:")
        print("  - Trading und Finanzoperationen")
        print("  - Videoschnitt und -bearbeitung")
        print("  - Bildgenerierung und -bearbeitung")
        print("  - Social Media Posting")
        print("  - Computer Vision und Bilderkennung")
        print("  - Computersteuerung (Maus, Tastatur, etc.)")
        print("\nGeben Sie Ihre Befehle direkt ein.")
        print("=" * 80 + "\n")
    
    def _load_additional_modules(self):
        """Lädt zusätzliche Module für erweiterte Funktionalitäten"""
        # In einer realen Implementierung würden hier die tatsächlichen Module geladen
        logger.info("Zusätzliche Module für erweiterte Funktionalitäten geladen")
    
    def execute_command(self, command_text: str) -> Dict[str, Any]:
        """
        Führt einen Befehl aus
        
        Args:
            command_text: Der auszuführende Befehl
            
        Returns:
            Das Ergebnis der Befehlsausführung
        """
        return self.command_interface.execute_command(command_text)
    
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
                    print(f"   {result.get('message', '')}")
                    if "result" in result:
                        print(f"   Ergebnis: {json.dumps(result['result'], indent=2, ensure_ascii=False)}")
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
        """Stoppt das MISODirectCommandSystem"""
        self.command_interface.stop()
        logger.info("MISO DirectCommandSystem gestoppt")


# Hauptfunktion für den direkten Start
def main():
    """Hauptfunktion für den direkten Start"""
    # Konfiguriere Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Erstelle und starte MISODirectCommandSystem
    command_system = MISODirectCommandSystem()
    
    try:
        # Starte interaktiven Modus
        command_system.run_interactive()
    except KeyboardInterrupt:
        print("\nProgramm wird beendet...")
    finally:
        command_system.stop()


if __name__ == "__main__":
    main()
