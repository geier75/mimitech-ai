#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AuditLogger für VXOR AI Blackbox.

Implementiert eine spezialisierte Logging-Lösung für Sicherheits-Audit-Trails,
mit Funktionen zur Aufzeichnung, Analyse und Verifizierung sicherheitsrelevanter
Ereignisse im VXOR AI Blackbox-System.
"""

import os
import json
import time
import hashlib
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

from .secure_logger import setup_logger

# Globales Dictionary für alle erstellten Audit-Logger, um Duplizierung zu vermeiden
_audit_loggers = {}

# Audit-Event-Typen
AUDIT_EVENT_TYPES = [
    "authentication",      # Authentifizierungsereignisse
    "authorization",       # Autorisierungsereignisse
    "key_management",      # Schlüsselverwaltungsereignisse
    "configuration",       # Konfigurationsänderungen
    "module_loading",      # Laden/Entladen von Modulen
    "file_access",         # Dateisystemzugriffe
    "cryptographic",       # Kryptographische Operationen
    "secure_boot",         # Sichere Boot-Ereignisse
    "update",              # Update-Ereignisse
    "security_violation",  # Sicherheitsverletzungen
    "system"               # Allgemeine Systemereignisse
]


class AuditLogger:
    """
    Spezialisierter Logger für Sicherheits-Audit-Trails in VXOR AI Blackbox.
    
    Diese Klasse implementiert:
    - Fälschungssichere Audit-Logs
    - Strukturierte Ereignisaufzeichnung
    - Digitale Signaturen für Audit-Einträge
    - Sicheres Speichern in Dateien und Datenbank
    - Forensische Analysefunktionen
    """
    
    def __init__(self, component_name: str, user_id: Optional[str] = None, 
                config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert den AuditLogger.
        
        Args:
            component_name: Name der Komponente, die den Logger verwendet
            user_id: Optional, ID des aktuellen Benutzers
            config: Optionale Konfiguration
        """
        self.component_name = component_name
        self.user_id = user_id or "system"
        self.config = config or self._load_config()
        
        # Initialisiere den Basis-Logger
        self.logger = setup_logger(
            f"audit.{component_name}",
            level=self.config.get("log_level", "INFO"),
            log_file=self._get_audit_log_path(),
            rotation=True,
            encryption=self.config.get("encryption", {}).get("enabled", False)
        )
        
        # Initialisiere die Audit-Datenbank, falls aktiviert
        if self.config.get("database", {}).get("enabled", False):
            self.db_connection = self._setup_database()
        else:
            self.db_connection = None
        
        # Initialisiere den Hash der letzten Einträge für die Verkettung
        self.last_entry_hash = None
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Lädt die Audit-Logger-Konfiguration.
        
        Returns:
            Die Audit-Logger-Konfiguration
        """
        # Standardkonfiguration
        return {
            "log_dir": os.path.join(os.path.expanduser("~"), ".vxor", "logs", "audit"),
            "log_level": "INFO",
            "encryption": {"enabled": False},
            "database": {
                "enabled": True,
                "path": os.path.join(os.path.expanduser("~"), ".vxor", "logs", "audit.db")
            },
            "chain_entries": True,  # Verkettung von Einträgen für Manipulationsschutz
            "sign_entries": False,   # Digitale Signaturen für Einträge (in produktiven Umgebungen aktivieren)
            "include_metadata": True  # Zusätzliche Metadaten in Einträgen
        }
    
    def _get_audit_log_path(self) -> str:
        """
        Bestimmt den Pfad zur Audit-Log-Datei.
        
        Returns:
            Pfad zur Audit-Log-Datei
        """
        log_dir = self.config["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        
        # Setze sichere Berechtigungen für das Verzeichnis
        os.chmod(log_dir, 0o700)  # Nur Besitzer hat vollen Zugriff
        
        # Erzeuge Dateinamen basierend auf Komponente
        filename = f"audit_{self.component_name}.log"
        return os.path.join(log_dir, filename)
    
    def _setup_database(self) -> sqlite3.Connection:
        """
        Richtet die Audit-Datenbank ein.
        
        Returns:
            SQLite-Verbindung zur Datenbank
        """
        db_path = self.config["database"]["path"]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Erstelle die Tabelle für Audit-Einträge
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            component TEXT NOT NULL,
            event_type TEXT NOT NULL,
            user_id TEXT NOT NULL,
            action TEXT NOT NULL,
            status TEXT NOT NULL,
            details TEXT,
            metadata TEXT,
            previous_hash TEXT,
            entry_hash TEXT,
            signature TEXT
        )
        ''')
        
        # Erstelle Indizes für effiziente Suche
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_component ON audit_events (component)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events (event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events (user_id)')
        
        conn.commit()
        
        # Setze sichere Berechtigungen für die Datenbank
        os.chmod(db_path, 0o600)  # Nur Besitzer kann lesen/schreiben
        
        return conn
    
    def _get_last_entry_hash(self) -> Optional[str]:
        """
        Holt den Hash des letzten Audit-Eintrags aus der Datenbank.
        
        Returns:
            Hash des letzten Eintrags oder None, wenn kein Eintrag vorhanden ist
        """
        if not self.db_connection:
            return None
        
        cursor = self.db_connection.cursor()
        cursor.execute('''
        SELECT entry_hash FROM audit_events
        WHERE component = ?
        ORDER BY id DESC LIMIT 1
        ''', (self.component_name,))
        
        result = cursor.fetchone()
        return result[0] if result else None
    
    def _compute_entry_hash(self, entry: Dict[str, Any], previous_hash: Optional[str] = None) -> str:
        """
        Berechnet einen Hash für einen Audit-Eintrag.
        
        Args:
            entry: Der Audit-Eintrag
            previous_hash: Hash des vorherigen Eintrags (für Verkettung)
            
        Returns:
            Der berechnete Hash
        """
        # Erstelle eine Kopie und entferne Felder, die nicht in den Hash einfließen sollen
        entry_copy = dict(entry)
        entry_copy.pop('entry_hash', None)
        entry_copy.pop('signature', None)
        
        # Füge den Hash des vorherigen Eintrags hinzu, falls vorhanden
        if previous_hash:
            entry_copy['previous_hash'] = previous_hash
        
        # Berechne den Hash über den serialisierten Eintrag
        entry_json = json.dumps(entry_copy, sort_keys=True)
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()
        
        return entry_hash
    
    def _sign_entry(self, entry_hash: str) -> Optional[str]:
        """
        Signiert einen Audit-Eintrag.
        
        Dies ist ein Platzhalter - in einer produktiven Umgebung würde hier
        eine echte digitale Signatur mit einem privaten Schlüssel erzeugt werden.
        
        Args:
            entry_hash: Hash des Audit-Eintrags
            
        Returns:
            Die digitale Signatur oder None, wenn Signaturen deaktiviert sind
        """
        if not self.config.get("sign_entries", False):
            return None
        
        # Platzhalter für eine echte Signatur
        # In einer produktiven Umgebung würde hier ein privater Schlüssel verwendet werden
        return f"signature_{entry_hash[:8]}"
    
    def log_event(self, event_type: str, action: str, status: str = "success",
                details: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Protokolliert ein Audit-Ereignis.
        
        Args:
            event_type: Typ des Ereignisses (einer der AUDIT_EVENT_TYPES)
            action: Die durchgeführte Aktion
            status: Status der Aktion ('success', 'failure', 'warning', 'info')
            details: Optionale Details zur Aktion
            metadata: Optionale Metadaten zum Ereignis
            
        Returns:
            Der protokollierte Audit-Eintrag
        """
        # Validiere den Event-Typ
        if event_type not in AUDIT_EVENT_TYPES:
            self.logger.warning(f"Ungültiger Audit-Event-Typ: {event_type}")
            event_type = "system"
        
        # Erstelle den Audit-Eintrag
        timestamp = int(time.time())
        formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Sammle Systeminformationen, falls Metadaten aktiviert sind
        if self.config.get("include_metadata", True) and metadata is None:
            metadata = {
                "hostname": os.uname().nodename,
                "pid": os.getpid()
            }
        
        entry = {
            "timestamp": timestamp,
            "formatted_time": formatted_time,
            "component": self.component_name,
            "event_type": event_type,
            "user_id": self.user_id,
            "action": action,
            "status": status,
            "details": details,
            "metadata": metadata
        }
        
        # Hole den Hash des letzten Eintrags, falls Verkettung aktiviert ist
        previous_hash = None
        if self.config.get("chain_entries", True):
            previous_hash = self.last_entry_hash or self._get_last_entry_hash()
            entry["previous_hash"] = previous_hash
        
        # Berechne den Hash des aktuellen Eintrags
        entry_hash = self._compute_entry_hash(entry, previous_hash)
        entry["entry_hash"] = entry_hash
        self.last_entry_hash = entry_hash
        
        # Signiere den Eintrag, falls aktiviert
        signature = self._sign_entry(entry_hash)
        if signature:
            entry["signature"] = signature
        
        # Protokolliere den Eintrag im Text-Log
        log_message = (f"{formatted_time} - {event_type} - {status} - "
                      f"{action}" + (f" - {details}" if details else ""))
        
        if status == "failure":
            self.logger.error(log_message)
        elif status == "warning":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Speichere in der Datenbank, falls aktiviert
        if self.db_connection:
            cursor = self.db_connection.cursor()
            cursor.execute('''
            INSERT INTO audit_events
            (timestamp, component, event_type, user_id, action, status, details, metadata, previous_hash, entry_hash, signature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                self.component_name,
                event_type,
                self.user_id,
                action,
                status,
                details,
                json.dumps(metadata) if metadata else None,
                previous_hash,
                entry_hash,
                signature
            ))
            self.db_connection.commit()
        
        return entry
    
    def query_events(self, filters: Dict[str, Any] = None, 
                    start_time: Optional[int] = None, end_time: Optional[int] = None, 
                    limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fragt Audit-Ereignisse aus der Datenbank ab.
        
        Args:
            filters: Filterkriterien (component, event_type, user_id, status)
            start_time: Optionaler Startzeitstempel (Unix-Zeit)
            end_time: Optionaler Endzeitstempel (Unix-Zeit)
            limit: Maximale Anzahl von Ergebnissen
            
        Returns:
            Liste der übereinstimmenden Audit-Einträge
        """
        if not self.db_connection:
            return []
        
        cursor = self.db_connection.cursor()
        
        # Basis-Query
        query = '''
        SELECT timestamp, component, event_type, user_id, action, status, details, metadata, 
               previous_hash, entry_hash, signature
        FROM audit_events
        WHERE 1=1
        '''
        params = []
        
        # Füge Filter hinzu
        if filters:
            for key, value in filters.items():
                if key in ['component', 'event_type', 'user_id', 'status', 'action']:
                    query += f" AND {key} = ?"
                    params.append(value)
        
        # Zeitfilter
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        # Sortierung und Limit
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        # Führe die Abfrage aus
        cursor.execute(query, params)
        
        # Konvertiere die Ergebnisse in Dictionaries
        result = []
        for row in cursor.fetchall():
            timestamp, component, event_type, user_id, action, status, details, metadata, \
            previous_hash, entry_hash, signature = row
            
            entry = {
                "timestamp": timestamp,
                "formatted_time": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                "component": component,
                "event_type": event_type,
                "user_id": user_id,
                "action": action,
                "status": status,
                "details": details,
                "metadata": json.loads(metadata) if metadata else None,
                "previous_hash": previous_hash,
                "entry_hash": entry_hash,
                "signature": signature
            }
            result.append(entry)
        
        return result
    
    def verify_chain_integrity(self, start_timestamp: Optional[int] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Überprüft die Integrität der Audit-Log-Kette.
        
        Args:
            start_timestamp: Optionaler Startzeitpunkt für die Überprüfung
            
        Returns:
            Tuple aus (Integrität intakt, Liste der problematischen Einträge)
        """
        if not self.db_connection or not self.config.get("chain_entries", True):
            return False, [{"error": "Verkettung nicht aktiviert oder keine Datenbank"}]
        
        cursor = self.db_connection.cursor()
        
        # Hole alle verketteten Einträge für diese Komponente
        query = '''
        SELECT id, timestamp, component, event_type, user_id, action, status, details, metadata,
               previous_hash, entry_hash
        FROM audit_events
        WHERE component = ?
        '''
        params = [self.component_name]
        
        if start_timestamp:
            query += " AND timestamp >= ?"
            params.append(start_timestamp)
        
        query += " ORDER BY timestamp ASC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            return True, []  # Keine Einträge zu überprüfen
        
        problematic_entries = []
        previous_hash = None
        
        for row in cursor.fetchall():
            id, timestamp, component, event_type, user_id, action, status, details, metadata, \
            stored_previous_hash, stored_entry_hash = row
            
            # Erstelle ein Dictionary für die Hash-Berechnung
            entry = {
                "timestamp": timestamp,
                "component": component,
                "event_type": event_type,
                "user_id": user_id,
                "action": action,
                "status": status,
                "details": details,
                "metadata": json.loads(metadata) if metadata else None
            }
            
            # Überprüfe den Hash des vorherigen Eintrags
            if previous_hash and stored_previous_hash != previous_hash:
                problematic_entries.append({
                    "id": id,
                    "timestamp": timestamp,
                    "formatted_time": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    "error": "Inkonsistenter Previous-Hash",
                    "stored_previous_hash": stored_previous_hash,
                    "expected_previous_hash": previous_hash
                })
            
            # Berechne den Hash für diesen Eintrag neu
            calculated_hash = self._compute_entry_hash(entry, stored_previous_hash)
            
            # Überprüfe, ob der berechnete Hash mit dem gespeicherten übereinstimmt
            if calculated_hash != stored_entry_hash:
                problematic_entries.append({
                    "id": id,
                    "timestamp": timestamp,
                    "formatted_time": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    "error": "Manipulierter Eintrag",
                    "stored_hash": stored_entry_hash,
                    "calculated_hash": calculated_hash
                })
            
            # Aktualisiere den vorherigen Hash für den nächsten Eintrag
            previous_hash = stored_entry_hash
        
        integrity_intact = len(problematic_entries) == 0
        
        if not integrity_intact:
            self.logger.warning(f"Audit-Log-Integritätsprüfung fehlgeschlagen: {len(problematic_entries)} problematische Einträge gefunden")
        
        return integrity_intact, problematic_entries
    
    def close(self):
        """Schließt alle geöffneten Ressourcen."""
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None


# Hilfsfunktionen für einfachen Zugriff

def get_audit_logger(component_name: str, user_id: Optional[str] = None) -> AuditLogger:
    """
    Holt oder erstellt einen AuditLogger für eine Komponente.
    
    Args:
        component_name: Name der Komponente
        user_id: Optionale Benutzer-ID
        
    Returns:
        AuditLogger-Instanz
    """
    # Erstelle einen eindeutigen Schlüssel für den Logger
    key = f"{component_name}:{user_id or 'system'}"
    
    if key in _audit_loggers:
        return _audit_loggers[key]
    
    # Erstelle einen neuen Logger
    logger = AuditLogger(component_name, user_id)
    _audit_loggers[key] = logger
    
    return logger


def audit_event(component_name: str, event_type: str, action: str, status: str = "success",
              details: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
              user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Protokolliert ein Audit-Ereignis mit dem passenden AuditLogger.
    
    Args:
        component_name: Name der Komponente
        event_type: Typ des Ereignisses
        action: Die durchgeführte Aktion
        status: Status der Aktion
        details: Optionale Details
        metadata: Optionale Metadaten
        user_id: Optionale Benutzer-ID
        
    Returns:
        Der protokollierte Audit-Eintrag
    """
    logger = get_audit_logger(component_name, user_id)
    return logger.log_event(event_type, action, status, details, metadata)
