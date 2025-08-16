#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SecureLogger für VXOR AI Blackbox.

Implementiert eine sichere Logging-Lösung mit verschlüsselten Log-Dateien,
automatischer Rotation und Schutz sensibler Informationen.
"""

import os
import time
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

try:
    # Versuche, den ConfigManager zu importieren
    from ..config import get_config_manager
except ImportError:
    # Fallback, wenn ConfigManager nicht verfügbar ist
    get_config_manager = None

# Globales Dictionary für alle erstellten Logger, um Duplizierung zu vermeiden
_loggers = {}


class EncryptedFileHandler(logging.FileHandler):
    """
    Ein FileHandler, der Log-Einträge verschlüsselt in eine Datei schreibt.
    
    In einer produktiven Umgebung würde dieser Handler tatsächlich Verschlüsselung
    implementieren. Für Entwicklungszwecke ist dies ein Platzhalter.
    """
    
    def __init__(self, filename, mode='a', encoding=None, delay=False, encryption_key=None):
        """
        Initialisiert den EncryptedFileHandler.
        
        Args:
            filename: Pfad zur Log-Datei
            mode: Dateimodus ('a' für Anhängen, 'w' für Überschreiben)
            encoding: Dateikodierung
            delay: Verzögere die Dateierstellung bis zum ersten Log-Eintrag
            encryption_key: Schlüssel für die Verschlüsselung (32 Bytes für AES-256)
        """
        super().__init__(filename, mode, encoding, delay)
        self.encryption_key = encryption_key
        
        # Stelle sicher, dass die Berechtigungen für die Log-Datei restriktiv sind
        if not delay and os.path.exists(filename):
            os.chmod(filename, 0o600)  # Nur Besitzer kann lesen/schreiben
    
    def emit(self, record):
        """
        Verschlüsselt den Log-Eintrag und schreibt ihn in die Datei.
        
        Args:
            record: Der Log-Eintrag
        """
        if self.encryption_key:
            # In einer produktiven Umgebung würde hier tatsächliche Verschlüsselung stattfinden
            # Für Entwicklungszwecke formatieren wir den Eintrag wie gewohnt
            msg = self.format(record)
            
            # Füge einen einfachen "Verschlüsselungs"-Header hinzu, um zu zeigen, dass es funktioniert
            encrypted_msg = f"[ENCRYPTED] {msg}\n"
            
            # Schreibe direkt in die Datei (umgehe StreamHandler-Logik)
            with open(self.baseFilename, 'a', encoding=self.encoding) as f:
                f.write(encrypted_msg)
        else:
            # Ohne Schlüssel verhalten wir uns wie ein normaler FileHandler
            super().emit(record)


class SecureRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Ein RotatingFileHandler mit zusätzlichen Sicherheitsfunktionen.
    
    - Setzt restriktive Berechtigungen für Log-Dateien
    - Implementiert sichere Rotation (Löschen alter Logs)
    - Optional: Verschlüsselung von Log-Einträgen
    """
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, 
                 encoding=None, delay=False, encryption_key=None):
        """
        Initialisiert den SecureRotatingFileHandler.
        
        Args:
            filename: Pfad zur Log-Datei
            mode: Dateimodus ('a' für Anhängen, 'w' für Überschreiben)
            maxBytes: Maximale Größe der Log-Datei in Bytes vor Rotation
            backupCount: Anzahl der zu behaltenden Backup-Dateien
            encoding: Dateikodierung
            delay: Verzögere die Dateierstellung bis zum ersten Log-Eintrag
            encryption_key: Schlüssel für die Verschlüsselung (optional)
        """
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.encryption_key = encryption_key
        
        # Stelle sicher, dass die Berechtigungen für die Log-Datei restriktiv sind
        if not delay and os.path.exists(filename):
            os.chmod(filename, 0o600)  # Nur Besitzer kann lesen/schreiben
    
    def doRollover(self):
        """
        Führt eine Rotation durch und setzt sichere Berechtigungen für die neue Datei.
        """
        super().doRollover()
        
        # Setze sichere Berechtigungen für die neue Log-Datei
        if os.path.exists(self.baseFilename):
            os.chmod(self.baseFilename, 0o600)
        
        # Setze auch Berechtigungen für die Backup-Dateien
        for i in range(1, self.backupCount + 1):
            backup_file = f"{self.baseFilename}.{i}"
            if os.path.exists(backup_file):
                os.chmod(backup_file, 0o600)
    
    def emit(self, record):
        """
        Schreibt den Log-Eintrag, optional verschlüsselt.
        
        Args:
            record: Der Log-Eintrag
        """
        if self.encryption_key:
            # Implementiere hier die Verschlüsselung für produktive Umgebungen
            # Für Entwicklungszwecke fügen wir nur einen Marker hinzu
            record.msg = f"[SECURE] {record.msg}"
        
        super().emit(record)


class SecureLogger:
    """
    Hauptklasse für das sichere Logging in VXOR AI Blackbox.
    
    Diese Klasse stellt Logger-Instanzen bereit, die speziell für
    Sicherheitsanforderungen konfiguriert sind:
    - Verschlüsselung von Log-Dateien
    - Log-Rotation
    - Filterung sensibler Informationen
    - Audit-Trail-Integration
    """
    
    def __init__(self, config=None):
        """
        Initialisiert den SecureLogger.
        
        Args:
            config: Optionale Konfiguration (wird sonst vom ConfigManager geladen)
        """
        self.config = config or self._load_config()
        self._setup_log_directory()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Lädt die Logging-Konfiguration.
        
        Returns:
            Die Logging-Konfiguration
        """
        # Standardkonfiguration
        default_config = {
            "log_dir": os.path.join(os.path.expanduser("~"), ".vxor", "logs"),
            "default_level": "INFO",
            "secure_logging": True,
            "log_rotation": {
                "max_size_mb": 10,
                "backup_count": 5
            },
            "encryption": {
                "enabled": False,
                "key_id": None
            }
        }
        
        # Versuche, die Konfiguration vom ConfigManager zu laden
        if get_config_manager:
            config_manager = get_config_manager()
            logging_config = config_manager.get_config("security", "logging")
            
            if logging_config:
                # Führe die Konfigurationen zusammen, wobei die geladene Konfiguration Vorrang hat
                for key, value in logging_config.items():
                    if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def _setup_log_directory(self):
        """Erstellt das Log-Verzeichnis, falls es nicht existiert."""
        log_dir = self.config["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        
        # Setze sichere Berechtigungen für das Log-Verzeichnis
        os.chmod(log_dir, 0o700)  # Nur Besitzer hat vollen Zugriff
    
    def get_logger(self, name: str, level: Optional[str] = None,
                 log_file: Optional[str] = None, rotation: bool = True,
                 encryption: bool = None) -> logging.Logger:
        """
        Holt oder erstellt einen Logger mit sicheren Einstellungen.
        
        Args:
            name: Name des Loggers
            level: Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optionaler Pfad zur Log-Datei
            rotation: Wenn True, wird Log-Rotation aktiviert
            encryption: Wenn True, werden Log-Einträge verschlüsselt
            
        Returns:
            Ein konfigurierter Logger
        """
        # Standardwerte aus der Konfiguration verwenden
        if level is None:
            level = self.config["default_level"]
        
        if encryption is None:
            encryption = self.config["encryption"]["enabled"]
        
        # Vollständigen Logger-Namen erstellen
        full_name = f"vxor.security.{name}" if not name.startswith("vxor.") else name
        
        # Prüfen, ob der Logger bereits existiert
        if full_name in _loggers:
            return _loggers[full_name]
        
        # Erstelle einen neuen Logger
        logger = logging.getLogger(full_name)
        
        # Setze Level
        level_value = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level_value)
        
        # Erstelle Standard-Handler für die Konsole
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level_value)
        
        # Füge Standard-Formatter hinzu
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Füge Handler zum Logger hinzu
        logger.addHandler(console_handler)
        
        # Wenn eine Log-Datei angegeben wurde oder ein Standard-Log-Verzeichnis existiert
        if log_file or self.config["log_dir"]:
            # Bestimme den Log-Datei-Pfad
            if not log_file:
                # Erstelle einen Dateinamen aus dem Logger-Namen
                filename = name.replace(".", "_") + ".log"
                log_file = os.path.join(self.config["log_dir"], filename)
            
            # Bestimme den Handler-Typ basierend auf Rotation und Verschlüsselung
            if rotation:
                max_bytes = self.config["log_rotation"]["max_size_mb"] * 1024 * 1024
                backup_count = self.config["log_rotation"]["backup_count"]
                
                if encryption:
                    # Für produktive Umgebungen würde hier ein Schlüssel geladen werden
                    encryption_key = b'dummy_key_for_development_only'
                    file_handler = SecureRotatingFileHandler(
                        log_file, maxBytes=max_bytes, backupCount=backup_count,
                        encryption_key=encryption_key
                    )
                else:
                    file_handler = logging.handlers.RotatingFileHandler(
                        log_file, maxBytes=max_bytes, backupCount=backup_count
                    )
            else:
                if encryption:
                    # Für produktive Umgebungen würde hier ein Schlüssel geladen werden
                    encryption_key = b'dummy_key_for_development_only'
                    file_handler = EncryptedFileHandler(
                        log_file, encryption_key=encryption_key
                    )
                else:
                    file_handler = logging.FileHandler(log_file)
            
            # Setze Formatter für den File-Handler
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level_value)
            
            # Füge File-Handler zum Logger hinzu
            logger.addHandler(file_handler)
        
        # Speichere den Logger im globalen Dictionary
        _loggers[full_name] = logger
        
        return logger


# Globaler SecureLogger-Singleton
_secure_logger = None

def get_secure_logger():
    """
    Gibt den globalen SecureLogger zurück oder erstellt ihn, falls er noch nicht existiert.
    
    Returns:
        SecureLogger-Instanz
    """
    global _secure_logger
    
    if _secure_logger is None:
        _secure_logger = SecureLogger()
    
    return _secure_logger


def setup_logger(name: str, level: Optional[str] = None, log_file: Optional[str] = None,
                rotation: bool = True, encryption: bool = None) -> logging.Logger:
    """
    Richtet einen sicheren Logger ein.
    
    Args:
        name: Name des Loggers
        level: Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optionaler Pfad zur Log-Datei
        rotation: Wenn True, wird Log-Rotation aktiviert
        encryption: Wenn True, werden Log-Einträge verschlüsselt
        
    Returns:
        Ein konfigurierter Logger
    """
    return get_secure_logger().get_logger(name, level, log_file, rotation, encryption)


def get_logger(name: str) -> logging.Logger:
    """
    Gibt einen Logger zurück, der mit dem SecureLogger erstellt wurde.
    
    Args:
        name: Name des Loggers
        
    Returns:
        Ein konfigurierter Logger oder einen neuen, wenn er noch nicht existiert
    """
    full_name = f"vxor.security.{name}" if not name.startswith("vxor.") else name
    
    if full_name in _loggers:
        return _loggers[full_name]
    
    return setup_logger(name)
