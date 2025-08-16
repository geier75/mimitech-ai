#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Zero Trust Monitor (ZTM)

Implementiert das Zero-Trust-Monitoring-System für MISO Ultimate.
Überwacht alle kritischen Operationen und protokolliert Sicherheitsereignisse.

Copyright (c) 2025 MIMI Tech AI. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import json
import logging
import threading
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

# Konfiguriere Logging
logger = logging.getLogger("MISO.Security.ZTM.Monitor")

class SecurityEvent:
    """Repräsentiert ein Sicherheitsereignis"""
    
    def __init__(self, event_type: str, source: str, details: Dict[str, Any], severity: str = "INFO"):
        self.id = str(uuid.uuid4())
        self.timestamp = time.time()
        self.event_type = event_type
        self.source = source
        self.details = details
        self.severity = severity
        self.processed = False

class ZTMMonitor:
    """Zero Trust Monitor - Überwacht alle kritischen Operationen"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Gibt die Singleton-Instanz zurück oder erstellt eine neue."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ZTMMonitor()
        return cls._instance
    
    def __init__(self):
        """Initialisiert den ZTM-Monitor"""
        if ZTMMonitor._instance is not None:
            logger.warning("ZTMMonitor ist ein Singleton und wurde bereits initialisiert!")
            return
        
        self.initialized = False
        self.active = False
        self.events = []  # Liste der Sicherheitsereignisse
        self.tracked_modules = set()  # Überwachte Module
        self.hooks = {}  # Registrierte Hooks
        self.log_file = None
        self.session_id = str(uuid.uuid4())
        
        # Konfiguration aus Umgebungsvariablen
        self.ztm_mode = os.environ.get("MISO_ZTM_MODE", "0") == "1"
        self.log_level = os.environ.get("MISO_ZTM_LOG_LEVEL", "INFO")
        
        logger.info(f"ZTMMonitor Objekt erstellt (Session: {self.session_id})")
    
    def init(self) -> bool:
        """Initialisiert den ZTM-Monitor"""
        try:
            # Erstelle Log-Verzeichnis
            base_dir = Path(__file__).parent.parent
            log_dir = base_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            
            # Erstelle Session-Log-Datei
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"ztm_session_{timestamp}_{self.session_id[:8]}.log"
            self.log_file = log_dir / log_filename
            
            # Konfiguriere File-Handler für ZTM-Logs
            ztm_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            ztm_handler.setLevel(getattr(logging, self.log_level))
            ztm_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            ztm_handler.setFormatter(ztm_formatter)
            logger.addHandler(ztm_handler)
            
            # Registriere Standard-Hooks für kritische Module
            self._register_default_hooks()
            
            self.initialized = True
            self.active = self.ztm_mode
            
            # Erstes Sicherheitsereignis protokollieren
            self.track("ZTM_INIT", "ztm_monitor", {
                "session_id": self.session_id,
                "log_file": str(self.log_file),
                "ztm_mode": self.ztm_mode,
                "log_level": self.log_level
            }, "INFO")
            
            logger.info(f"ZTMMonitor initialisiert (Aktiv: {self.active}, Log: {self.log_file})")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei ZTM-Initialisierung: {e}")
            return False
    
    def _register_default_hooks(self):
        """Registriert Standard-Hooks für kritische Module"""
        critical_modules = [
            "vx_gestalt", "vx_reason", "vx_matrix", "vx_memex", 
            "vx_intent", "vx_hyperfilter", "prism_engine", 
            "echo_prime", "t_mathematics", "qlogik_core"
        ]
        
        for module in critical_modules:
            self.register_hook(module, self._default_hook)
            self.tracked_modules.add(module)
    
    def _default_hook(self, module: str, operation: str, data: Dict[str, Any]):
        """Standard-Hook für kritische Operationen"""
        self.track("MODULE_OPERATION", module, {
            "operation": operation,
            "data_keys": list(data.keys()) if isinstance(data, dict) else str(type(data)),
            "timestamp": time.time()
        }, "DEBUG")
    
    def register_hook(self, module: str, hook_func: Callable):
        """Registriert einen Hook für ein Modul
        
        Args:
            module: Name des Moduls
            hook_func: Hook-Funktion
        """
        if module not in self.hooks:
            self.hooks[module] = []
        self.hooks[module].append(hook_func)
        logger.debug(f"Hook für Modul '{module}' registriert")
    
    def track(self, event_type: str, source: str, details: Dict[str, Any], severity: str = "INFO"):
        """Verfolgt ein Sicherheitsereignis
        
        Args:
            event_type: Typ des Ereignisses
            source: Quelle des Ereignisses
            details: Details des Ereignisses
            severity: Schweregrad (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if not self.active:
            return
        
        # Erstelle Sicherheitsereignis
        event = SecurityEvent(event_type, source, details, severity)
        self.events.append(event)
        
        # Protokolliere das Ereignis
        log_message = f"[{event_type}] {source}: {json.dumps(details, default=str)}"
        log_level = getattr(logging, severity)
        logger.log(log_level, log_message)
        
        # Führe registrierte Hooks aus
        if source in self.hooks:
            for hook in self.hooks[source]:
                try:
                    hook(source, event_type, details)
                except Exception as e:
                    logger.error(f"Fehler beim Ausführen des Hooks für {source}: {e}")
        
        # Prüfe auf kritische Ereignisse
        if severity in ["ERROR", "CRITICAL"]:
            self._handle_critical_event(event)
    
    def _handle_critical_event(self, event: SecurityEvent):
        """Behandelt kritische Sicherheitsereignisse"""
        logger.critical(f"KRITISCHES EREIGNIS: {event.event_type} von {event.source}")
        
        # Zusätzliche Sicherheitsmaßnahmen bei kritischen Ereignissen
        if event.severity == "CRITICAL":
            # Erstelle Backup der aktuellen Events
            self._backup_events()
            
            # Benachrichtige andere Sicherheitskomponenten
            try:
                from .void.void_protocol import get_instance as get_void
                void_protocol = get_void()
                void_protocol.verify(event.details, None)
            except Exception as e:
                logger.error(f"Konnte VOID-Protokoll nicht benachrichtigen: {e}")
    
    def _backup_events(self):
        """Erstellt ein Backup der aktuellen Events"""
        try:
            base_dir = Path(__file__).parent.parent
            backup_dir = base_dir / "secure_logs"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"ztm_events_backup_{timestamp}.json"
            
            events_data = []
            for event in self.events:
                events_data.append({
                    "id": event.id,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "source": event.source,
                    "details": event.details,
                    "severity": event.severity,
                    "processed": event.processed
                })
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(events_data, f, indent=2, default=str)
            
            logger.info(f"Events-Backup erstellt: {backup_file}")
            
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Events-Backup: {e}")
    
    def get_events(self, event_type: str = None, source: str = None, 
                   severity: str = None, limit: int = None) -> List[SecurityEvent]:
        """Ruft Sicherheitsereignisse ab
        
        Args:
            event_type: Filtert nach Ereignistyp
            source: Filtert nach Quelle
            severity: Filtert nach Schweregrad
            limit: Maximale Anzahl der Ereignisse
            
        Returns:
            Liste der gefilterten Ereignisse
        """
        filtered_events = self.events
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        if source:
            filtered_events = [e for e in filtered_events if e.source == source]
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]
        
        # Sortiere nach Zeitstempel (neueste zuerst)
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        if limit:
            filtered_events = filtered_events[:limit]
        
        return filtered_events
    
    def get_status(self) -> Dict[str, Any]:
        """Gibt den aktuellen Status des ZTM-Monitors zurück"""
        return {
            "initialized": self.initialized,
            "active": self.active,
            "session_id": self.session_id,
            "total_events": len(self.events),
            "tracked_modules": list(self.tracked_modules),
            "log_file": str(self.log_file) if self.log_file else None,
            "ztm_mode": self.ztm_mode,
            "log_level": self.log_level
        }
    
    def verify(self, data: Any, signature: str = None) -> bool:
        """Verifiziert Daten (kompatibel mit ZTM-Validator Interface)
        
        Args:
            data: Zu verifizierende Daten
            signature: Optionale Signatur
            
        Returns:
            True, wenn die Daten verifiziert wurden
        """
        # Protokolliere Verifikationsversuch
        self.track("DATA_VERIFICATION", "ztm_monitor", {
            "data_type": str(type(data)),
            "has_signature": signature is not None,
            "data_hash": hashlib.sha256(str(data).encode()).hexdigest()[:16]
        }, "DEBUG")
        
        # Einfache Verifikation - in einer echten Implementierung würde hier
        # eine komplexere Verifikation stattfinden
        return True
    
    def shutdown(self):
        """Fährt den ZTM-Monitor herunter"""
        if self.active:
            self.track("ZTM_SHUTDOWN", "ztm_monitor", {
                "session_id": self.session_id,
                "total_events": len(self.events),
                "shutdown_time": time.time()
            }, "INFO")
            
            # Erstelle finales Backup
            self._backup_events()
            
            self.active = False
            logger.info("ZTMMonitor heruntergefahren")

# Globale Funktionen für einfachen Zugriff
def get_instance() -> ZTMMonitor:
    """Gibt die Singleton-Instanz des ZTM-Monitors zurück"""
    return ZTMMonitor.get_instance()

def track(event_type: str, source: str, details: Dict[str, Any], severity: str = "INFO"):
    """Verfolgt ein Sicherheitsereignis über die globale Instanz"""
    return get_instance().track(event_type, source, details, severity)

def register_hook(module: str, hook_func: Callable):
    """Registriert einen Hook über die globale Instanz"""
    return get_instance().register_hook(module, hook_func)

def verify(data: Any, signature: str = None) -> bool:
    """Verifiziert Daten über die globale Instanz"""
    return get_instance().verify(data, signature)

def initialize() -> bool:
    """Initialisiert den ZTM-Monitor"""
    return get_instance().init()

# Automatische Initialisierung beim Import
if os.environ.get("MISO_ZTM_MODE", "0") == "1":
    initialize()
