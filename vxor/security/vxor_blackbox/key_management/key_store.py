#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KeyStore für VXOR AI Blackbox.

Verantwortlich für die persistente und sichere Speicherung aller kryptographischen
Schlüssel des VXOR AI Blackbox-Systems mit SQLite-Backend und verschlüsselter
Speicherung sensibler Schlüsselinformationen.
"""

import os
import json
import time
import sqlite3
import hashlib
import logging
from typing import Dict, Any, Tuple, List, Optional, Union

from ..crypto.aes import AESCrypto

class KeyStore:
    """
    Speichert kryptographische Schlüssel sicher und persistent.
    
    Funktionen:
    - SQLite-basierte Schlüsselspeicherung
    - Verschlüsselte Speicherung privater Schlüssel
    - Indexierung und effiziente Suche
    - Audit-Trail für Schlüsseloperationen
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert den KeyStore.
        
        Args:
            config_path: Optionaler Pfad zur Konfigurationsdatei
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.db_path = self._setup_database()
        self.aes = AESCrypto()
        
        self.logger.info("KeyStore initialisiert")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Lädt die Konfiguration aus einer Datei oder verwendet Standardwerte."""
        default_config = {
            "database": {
                "path": os.path.join(os.path.expanduser("~"), ".vxor", "keystore", "vxor_keys.db"),
                "encryption_key_file": os.path.join(os.path.expanduser("~"), ".vxor", "keystore", "db_encryption.key"),
                "auto_backup": True,
                "backup_interval": 86400  # 1 Tag in Sekunden
            },
            "audit": {
                "enabled": True,
                "log_file": os.path.join(os.path.expanduser("~"), ".vxor", "logs", "keystore_audit.log"),
                "log_level": "INFO"
            },
            "retention": {
                "keep_old_keys": True,  # Behalte alte, rotierte Schlüssel
                "archive_after_days": 30  # Archiviere Schlüssel nach X Tagen Inaktivität
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Konfigurationen zusammenführen
                    for section in default_config:
                        if section in loaded_config:
                            default_config[section].update(loaded_config[section])
            except Exception as e:
                # Bei Fehler Standardkonfiguration verwenden
                pass
        
        return default_config
    
    def _setup_logging(self):
        """Konfiguriert das Logging für den KeyStore."""
        self.logger = logging.getLogger("vxor.security.keystore")
        if not self.logger.handlers:
            log_dir = os.path.join(os.path.expanduser("~"), ".vxor", "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            handler = logging.FileHandler(os.path.join(log_dir, "vxor_keystore.log"))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Einrichten des Audit-Logs, falls aktiviert
        if self.config["audit"]["enabled"]:
            self.audit_logger = logging.getLogger("vxor.security.keystore.audit")
            if not self.audit_logger.handlers:
                audit_file = self.config["audit"]["log_file"]
                os.makedirs(os.path.dirname(audit_file), exist_ok=True)
                
                handler = logging.FileHandler(audit_file)
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                handler.setFormatter(formatter)
                self.audit_logger.addHandler(handler)
                self.audit_logger.setLevel(getattr(logging, self.config["audit"]["log_level"]))
        else:
            self.audit_logger = None
    
    def _setup_database(self) -> str:
        """
        Richtet die SQLite-Datenbank für die Schlüsselspeicherung ein.
        
        Returns:
            Pfad zur Datenbank
        """
        db_path = self.config["database"]["path"]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Tabelle für Schlüsselinformationen erstellen
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS keys (
            id TEXT PRIMARY KEY,
            key_type TEXT NOT NULL,
            purpose TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            public_key_path TEXT,
            private_key_path TEXT,
            encrypted_private_key BLOB,
            metadata TEXT,
            status TEXT DEFAULT 'active',
            last_used_at INTEGER
        )
        ''')
        
        # Tabelle für aktive Schlüssel erstellen
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_keys (
            purpose TEXT NOT NULL,
            key_type TEXT NOT NULL,
            key_id TEXT NOT NULL,
            activated_at INTEGER NOT NULL,
            PRIMARY KEY (purpose, key_type),
            FOREIGN KEY (key_id) REFERENCES keys(id)
        )
        ''')
        
        # Tabelle für den Audit-Trail erstellen
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            action TEXT NOT NULL,
            key_id TEXT,
            user_id TEXT,
            details TEXT
        )
        ''')
        
        # Indizes für effiziente Suche erstellen
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_keys_type_purpose ON keys (key_type, purpose)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_keys_status ON keys (status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_keys_expires ON keys (expires_at)')
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Datenbank in {db_path} eingerichtet")
        return db_path
    
    def _get_db_encryption_key(self) -> bytes:
        """
        Holt oder generiert den Schlüssel für die Datenbankverschlüsselung.
        
        Returns:
            32-Byte AES-Schlüssel
        """
        key_file = self.config["database"]["encryption_key_file"]
        
        # Erstelle Verzeichnis, falls nötig
        os.makedirs(os.path.dirname(key_file), exist_ok=True)
        
        if os.path.exists(key_file):
            # Lade vorhandenen Schlüssel
            with open(key_file, 'rb') as f:
                key = f.read()
            
            # Überprüfe Schlüssellänge
            if len(key) != 32:
                self.logger.warning("Ungültiger Datenbankschlüssel, generiere neuen Schlüssel")
                key = os.urandom(32)
                with open(key_file, 'wb') as f:
                    f.write(key)
        else:
            # Generiere neuen Schlüssel
            key = os.urandom(32)
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Setze sichere Berechtigungen für die Schlüsseldatei
            os.chmod(key_file, 0o600)  # Nur Besitzer kann lesen/schreiben
            
            self.logger.info(f"Neuer Datenbankschlüssel in {key_file} generiert")
        
        return key
    
    def store_key(self, key_id: str, key_type: str, purpose: str,
                 public_key: Optional[bytes] = None, private_key: Optional[bytes] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Speichert einen Schlüssel in der Datenbank.
        
        Args:
            key_id: ID des Schlüssels
            key_type: Typ des Schlüssels ('kyber', 'ntru', 'dilithium', 'aes')
            purpose: Verwendungszweck
            public_key: Optionaler öffentlicher Schlüssel
            private_key: Optionaler privater Schlüssel
            metadata: Optionale Metadaten
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = int(time.time())
        
        # Verschlüssele den privaten Schlüssel, falls vorhanden
        encrypted_private_key = None
        if private_key:
            db_key = self._get_db_encryption_key()
            encrypted_private_key = self.aes.encrypt(private_key, db_key)
        
        # Füge den Schlüssel in die Datenbank ein
        cursor.execute('''
        INSERT INTO keys 
        (id, key_type, purpose, created_at, expires_at, 
         public_key_path, private_key_path, encrypted_private_key, metadata, last_used_at) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            key_id,
            key_type,
            purpose,
            current_time,
            current_time + (30 * 86400),  # Standard: 30 Tage gültig
            None,  # In einer vollständigen Implementierung würden hier Pfade gespeichert
            None,
            encrypted_private_key,
            json.dumps(metadata) if metadata else None,
            current_time
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Schlüssel {key_id} ({key_type}) für {purpose} in der Datenbank gespeichert")
        
        # Audit-Log
        self._audit_log("store", key_id, f"Schlüssel {key_type} für {purpose} gespeichert")
    
    def get_key(self, key_id: str) -> Dict[str, Any]:
        """
        Holt einen Schlüssel anhand seiner ID.
        
        Args:
            key_id: ID des Schlüssels
            
        Returns:
            Dictionary mit den Schlüsselinformationen und dem Schlüsselmaterial
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT key_type, purpose, created_at, expires_at, 
               encrypted_private_key, metadata, status
        FROM keys
        WHERE id = ?
        ''', (key_id,))
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            raise KeyError(f"Schlüssel mit ID {key_id} nicht gefunden")
        
        key_type, purpose, created_at, expires_at, encrypted_private_key, metadata_json, status = row
        
        # Aktualisiere last_used_at
        cursor.execute('UPDATE keys SET last_used_at = ? WHERE id = ?', (int(time.time()), key_id))
        conn.commit()
        
        # Entschlüssele den privaten Schlüssel, falls vorhanden
        private_key = None
        if encrypted_private_key:
            db_key = self._get_db_encryption_key()
            private_key = self.aes.decrypt(encrypted_private_key, db_key)
        
        # Erstelle Ergebnisdictionary
        result = {
            "id": key_id,
            "type": key_type,
            "purpose": purpose,
            "created_at": created_at,
            "expires_at": expires_at,
            "status": status,
            "metadata": json.loads(metadata_json) if metadata_json else {}
        }
        
        if private_key:
            result["private_key"] = private_key
        
        conn.close()
        
        self.logger.info(f"Schlüssel {key_id} aus der Datenbank geladen")
        
        # Audit-Log
        self._audit_log("retrieve", key_id, f"Schlüssel {key_type} für {purpose} abgerufen")
        
        return result
    
    def delete_key(self, key_id: str, permanent: bool = False) -> bool:
        """
        Löscht einen Schlüssel aus der Datenbank.
        
        Args:
            key_id: ID des Schlüssels
            permanent: Wenn True, wird der Schlüssel permanent gelöscht, 
                      ansonsten nur als inaktiv markiert
            
        Returns:
            True, wenn der Schlüssel erfolgreich gelöscht wurde
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Prüfe, ob der Schlüssel existiert
        cursor.execute('SELECT key_type, purpose FROM keys WHERE id = ?', (key_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False
        
        key_type, purpose = row
        
        if permanent:
            # Lösche den Schlüssel permanent
            cursor.execute('DELETE FROM keys WHERE id = ?', (key_id,))
            cursor.execute('DELETE FROM active_keys WHERE key_id = ?', (key_id,))
        else:
            # Markiere den Schlüssel als inaktiv
            cursor.execute('UPDATE keys SET status = ? WHERE id = ?', ('inactive', key_id))
        
        conn.commit()
        conn.close()
        
        action = "permanent_delete" if permanent else "deactivate"
        self.logger.info(f"Schlüssel {key_id} {action}")
        
        # Audit-Log
        self._audit_log(
            action, 
            key_id, 
            f"Schlüssel {key_type} für {purpose} {'permanent gelöscht' if permanent else 'deaktiviert'}"
        )
        
        return True
    
    def set_active_key(self, key_id: str) -> bool:
        """
        Setzt einen Schlüssel als aktiv für seinen Typ und Zweck.
        
        Args:
            key_id: ID des Schlüssels
            
        Returns:
            True, wenn der Schlüssel erfolgreich aktiviert wurde
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Hole die Informationen des zu aktivierenden Schlüssels
        cursor.execute('SELECT key_type, purpose FROM keys WHERE id = ?', (key_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False
        
        key_type, purpose = row
        
        # Setze den Schlüssel als aktiv
        cursor.execute('''
        INSERT OR REPLACE INTO active_keys (purpose, key_type, key_id, activated_at)
        VALUES (?, ?, ?, ?)
        ''', (purpose, key_type, key_id, int(time.time())))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Schlüssel {key_id} als aktiv für {purpose}/{key_type} gesetzt")
        
        # Audit-Log
        self._audit_log(
            "activate", 
            key_id, 
            f"Schlüssel {key_type} für {purpose} als aktiv gesetzt"
        )
        
        return True
    
    def get_active_key(self, purpose: str, key_type: str) -> Optional[Dict[str, Any]]:
        """
        Holt den aktiven Schlüssel für einen bestimmten Typ und Zweck.
        
        Args:
            purpose: Verwendungszweck
            key_type: Schlüsseltyp
            
        Returns:
            Schlüsselinformationen oder None, wenn kein aktiver Schlüssel gefunden wurde
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT key_id 
        FROM active_keys 
        WHERE purpose = ? AND key_type = ?
        ''', (purpose, key_type))
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        key_id = row[0]
        conn.close()
        
        # Lade den vollständigen Schlüssel
        return self.get_key(key_id)
    
    def list_keys(self, key_type: Optional[str] = None, purpose: Optional[str] = None,
                status: str = 'active') -> List[Dict[str, Any]]:
        """
        Listet Schlüssel auf, optional gefiltert nach Typ, Zweck und Status.
        
        Args:
            key_type: Optional, filtert nach Schlüsseltyp
            purpose: Optional, filtert nach Verwendungszweck
            status: Status der Schlüssel ('active', 'inactive', 'all')
            
        Returns:
            Liste von Schlüsselinformationen (ohne private Schlüssel)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT id, key_type, purpose, created_at, expires_at, metadata, status FROM keys WHERE 1=1'
        params = []
        
        if key_type:
            query += ' AND key_type = ?'
            params.append(key_type)
        
        if purpose:
            query += ' AND purpose = ?'
            params.append(purpose)
        
        if status != 'all':
            query += ' AND status = ?'
            params.append(status)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            key_id, key_type, purpose, created_at, expires_at, metadata_json, status = row
            
            key_info = {
                "id": key_id,
                "type": key_type,
                "purpose": purpose,
                "created_at": created_at,
                "expires_at": expires_at,
                "status": status,
                "metadata": json.loads(metadata_json) if metadata_json else {}
            }
            
            result.append(key_info)
        
        conn.close()
        return result
    
    def check_expired_keys(self) -> List[str]:
        """
        Überprüft, welche Schlüssel abgelaufen sind.
        
        Returns:
            Liste der IDs abgelaufener Schlüssel
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = int(time.time())
        
        cursor.execute('''
        SELECT id FROM keys 
        WHERE expires_at < ? AND status = 'active'
        ''', (current_time,))
        
        expired_ids = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        if expired_ids:
            self.logger.info(f"{len(expired_ids)} abgelaufene Schlüssel gefunden")
        
        return expired_ids
    
    def backup_database(self) -> str:
        """
        Erstellt ein Backup der Schlüsseldatenbank.
        
        Returns:
            Pfad zur Backup-Datei
        """
        backup_dir = os.path.join(os.path.dirname(self.db_path), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = os.path.join(backup_dir, f"vxor_keys_{timestamp}.db")
        
        conn = sqlite3.connect(self.db_path)
        backup = sqlite3.connect(backup_path)
        
        conn.backup(backup)
        
        backup.close()
        conn.close()
        
        self.logger.info(f"Datenbank-Backup erstellt: {backup_path}")
        
        # Audit-Log
        self._audit_log("backup", None, f"Datenbank-Backup erstellt: {backup_path}")
        
        return backup_path
    
    def _audit_log(self, action: str, key_id: Optional[str], details: str) -> None:
        """
        Fügt einen Eintrag zum Audit-Log hinzu.
        
        Args:
            action: Die durchgeführte Aktion
            key_id: ID des betroffenen Schlüssels (falls vorhanden)
            details: Details zur Aktion
        """
        if not self.audit_logger:
            return
        
        # Logge in die Audit-Datei
        self.audit_logger.info(f"ACTION={action}, KEY_ID={key_id}, DETAILS={details}")
        
        # Speichere auch in der Datenbank
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO audit_trail (timestamp, action, key_id, user_id, details)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            int(time.time()),
            action,
            key_id,
            "system",  # In einer vollständigen Implementierung würde hier der Benutzer gespeichert
            details
        ))
        
        conn.commit()
        conn.close()
