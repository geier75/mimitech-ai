#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO ZTM Monitor - Zero-Trust Monitoring System

Diese Datei implementiert den ZTM-Monitor, der Systemaktivitäten überwacht
und Protokolle gemäß den Zero-Trust-Prinzipien erstellt.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import json
import time
import uuid
import logging
import hashlib
import threading
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.monitoring.ztm_monitor")

class ZTMMonitor:
    """
    Zero-Trust Monitoring-System für MISO.
    
    Diese Klasse implementiert das ZTM-System, das alle Systemaktivitäten
    überwacht und protokolliert, um die Sicherheit und Integrität des MISO-Systems
    zu gewährleisten.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'ZTMMonitor':
        """
        Gibt die Singleton-Instanz des ZTMMonitors zurück oder erstellt eine neue.
        
        Returns:
            ZTMMonitor: Die Singleton-Instanz
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ZTMMonitor()
        return cls._instance
    
    def __init__(self):
        """Initialisiert den ZTMMonitor mit grundlegenden Einstellungen."""
        # Prüfe ZTM-Modus aus Umgebungsvariable
        self.ztm_mode = os.environ.get('MISO_ZTM_MODE', '0')
        self.enabled = self.ztm_mode == '1'
        
        # Konfiguriere Log-Level
        log_level_str = os.environ.get('MISO_ZTM_LOG_LEVEL', 'INFO')
        self.log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(self.log_level)
        
        # Konfiguriere Log-Pfad
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(self.base_dir, 'logs', 'ztm')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, 'ztm_log.json')
        
        # Initialisiere Log-Datei, falls sie noch nicht existiert
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
        
        logger.info(f"ZTMMonitor initialisiert. Aktiviert: {self.enabled}, Log-Pfad: {self.log_file}")
    
    def track(self, module_name: str, action: str, data: Dict[str, Any], 
              severity: str = "INFO", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Protokolliert eine Aktivität im ZTM-System.
        
        Args:
            module_name: Name des Moduls oder der Komponente
            action: Ausgeführte Aktion oder Ereignis
            data: Relevante Daten des Ereignisses
            severity: Schweregrad (INFO, WARNING, ERROR, CRITICAL)
            context: Zusätzlicher Kontext für das Ereignis
            
        Returns:
            Dict[str, Any]: Das protokollierte Ereignis
        """
        if not self.enabled:
            return {}
        
        # Erstelle Event-Eintrag
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "module": module_name,
            "action": action,
            "severity": severity,
            "data": data
        }
        
        # Füge Kontext hinzu, falls vorhanden
        if context:
            event["context"] = context
            
        # Berechne Integritäts-Hash
        event_data_str = json.dumps(event, sort_keys=True)
        event["integrity_hash"] = hashlib.sha256(event_data_str.encode()).hexdigest()
        
        # Protokolliere Ereignis
        self._write_to_log(event)
        
        # Logge entsprechend dem Schweregrad
        log_method = getattr(logger, severity.lower(), logger.info)
        log_method(f"ZTM Event: {module_name} - {action} ({severity})")
        
        return event
    
    def _write_to_log(self, event: Dict[str, Any]) -> None:
        """
        Schreibt ein Ereignis in die Log-Datei.
        
        Args:
            event: Das zu protokollierende Ereignis
        """
        try:
            # Lese bestehende Logs
            events = []
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0:
                with open(self.log_file, 'r') as f:
                    try:
                        events = json.load(f)
                    except json.JSONDecodeError:
                        logger.error(f"Fehler beim Lesen der ZTM-Log-Datei: {self.log_file}")
                        events = []
            
            # Füge neues Ereignis hinzu
            events.append(event)
            
            # Schreibe Logs zurück
            with open(self.log_file, 'w') as f:
                json.dump(events, f, indent=2)
                
        except Exception as e:
            logger.error(f"Fehler beim Schreiben in die ZTM-Log-Datei: {e}")

    def get_logs(self, module_filter: Optional[str] = None, 
                 severity_filter: Optional[str] = None, 
                 time_range: Optional[Tuple[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Ruft gefilterte Logs aus der ZTM-Log-Datei ab.
        
        Args:
            module_filter: Optional, filtert nach Modulnamen
            severity_filter: Optional, filtert nach Schweregrad
            time_range: Optional, filtert nach Zeitbereich (start, end) als ISO-Strings
            
        Returns:
            List[Dict[str, Any]]: Die gefilterten Logs
        """
        if not self.enabled or not os.path.exists(self.log_file):
            return []
            
        try:
            with open(self.log_file, 'r') as f:
                events = json.load(f)
                
            # Filtere Ereignisse
            if module_filter:
                events = [e for e in events if e.get("module") == module_filter]
                
            if severity_filter:
                events = [e for e in events if e.get("severity") == severity_filter]
                
            if time_range:
                start, end = time_range
                events = [e for e in events if start <= e.get("timestamp", "") <= end]
                
            return events
                
        except Exception as e:
            logger.error(f"Fehler beim Lesen der ZTM-Logs: {e}")
            return []

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """
        Überprüft die Integrität aller ZTM-Logs.
        
        Returns:
            Tuple[bool, List[str]]: (Integrität OK, Liste von fehlerhaften Event-IDs)
        """
        if not self.enabled or not os.path.exists(self.log_file):
            return (True, [])
            
        try:
            with open(self.log_file, 'r') as f:
                events = json.load(f)
                
            invalid_events = []
            
            for event in events:
                # Kopiere Event ohne Integritäts-Hash
                event_copy = event.copy()
                stored_hash = event_copy.pop("integrity_hash", None)
                
                if stored_hash:
                    # Berechne Hash neu und vergleiche
                    event_data_str = json.dumps(event_copy, sort_keys=True)
                    calculated_hash = hashlib.sha256(event_data_str.encode()).hexdigest()
                    
                    if calculated_hash != stored_hash:
                        invalid_events.append(event.get("id", "unknown"))
                else:
                    invalid_events.append(event.get("id", "unknown"))
                
            return (len(invalid_events) == 0, invalid_events)
                
        except Exception as e:
            logger.error(f"Fehler bei der Integritätsprüfung der ZTM-Logs: {e}")
            return (False, ["error"])


# Globale Instanz für einfachen Zugriff
ztm_monitor = ZTMMonitor.get_instance()


def track(module_name: str, action: str, data: Dict[str, Any], 
          severity: str = "INFO", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Globale Funktion zum Protokollieren einer Aktivität im ZTM-System.
    
    Args:
        module_name: Name des Moduls oder der Komponente
        action: Ausgeführte Aktion oder Ereignis
        data: Relevante Daten des Ereignisses
        severity: Schweregrad (INFO, WARNING, ERROR, CRITICAL)
        context: Zusätzlicher Kontext für das Ereignis
        
    Returns:
        Dict[str, Any]: Das protokollierte Ereignis
    """
    return ztm_monitor.track(module_name, action, data, severity, context)


# Wenn diese Datei direkt ausgeführt wird, führe einen einfachen Test durch
if __name__ == "__main__":
    # Aktiviere ZTM für den Test
    os.environ['MISO_ZTM_MODE'] = '1'
    
    # Erstelle eine Testinstanz
    monitor = ZTMMonitor()
    
    # Protokolliere ein Testereignis
    monitor.track(
        module_name="test_module",
        action="test_action",
        data={"test_key": "test_value"},
        severity="INFO",
        context={"test_context": True}
    )
    
    print(f"ZTM-Test abgeschlossen. Logs wurden in {monitor.log_file} geschrieben.")
