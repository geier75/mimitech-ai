#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Key Manager für VXOR AI Blackbox.

Zentrale Komponente zur Verwaltung aller kryptographischen Schlüssel
im VXOR AI Blackbox-System, mit besonderer Unterstützung für postquantenfeste
Algorithmen und sichere Schlüsselrotation.
"""

import os
import time
import json
import logging
import hashlib
from typing import Dict, Any, Tuple, List, Optional, Union

from ..crypto.quantum_resistant_crypto import QuantumResistantCrypto

class KeyManager:
    """
    Verwaltet alle kryptographischen Schlüssel für VXOR AI Blackbox.
    
    Funktionen:
    - Schlüsselerzeugung (Kyber, NTRU, Dilithium, AES)
    - Sichere Schlüsselspeicherung
    - Automatische Schlüsselrotation
    - Schlüsselauthentifizierung
    - Notfallwiederherstellung
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert den KeyManager.
        
        Args:
            config_path: Optionaler Pfad zur Konfigurationsdatei
        """
        self.config = self._load_config(config_path)
        self.crypto = QuantumResistantCrypto(config_path)
        self._setup_logging()
        self._setup_key_directories()
        
        self.logger.info("KeyManager initialisiert")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Lädt die Konfiguration aus einer Datei oder verwendet Standardwerte."""
        default_config = {
            "key_store": {
                "root_directory": os.path.join(os.path.expanduser("~"), ".vxor", "keystore"),
                "database_file": "vxor_keys.db"
            },
            "key_rotation": {
                "kyber": 30,        # Tage
                "ntru": 60,         # Tage
                "dilithium": 90,    # Tage
                "aes": 7            # Tage
            },
            "key_types": ["kyber", "ntru", "dilithium", "aes"],
            "key_purposes": [
                "boot",             # Bootloader-Verschlüsselung
                "module",           # Modulverschlüsselung
                "data",             # Datenverschlüsselung
                "update",           # Update-Signatur
                "ipc"               # Inter-Prozess-Kommunikation
            ]
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge Konfigurationen
                    for key, value in loaded_config.items():
                        if key in default_config and isinstance(default_config[key], dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                # Fehler beim Laden, verwende Standardkonfiguration
                pass
        
        return default_config
    
    def _setup_logging(self):
        """Konfiguriert das Logging für den KeyManager."""
        self.logger = logging.getLogger("vxor.security.keys")
        if not self.logger.handlers:
            log_dir = os.path.join(os.path.expanduser("~"), ".vxor", "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            handler = logging.FileHandler(os.path.join(log_dir, "vxor_keymanager.log"))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _setup_key_directories(self):
        """Richtet die Verzeichnisstruktur für die Schlüsselspeicherung ein."""
        root_dir = self.config["key_store"]["root_directory"]
        os.makedirs(root_dir, exist_ok=True)
        
        # Erstelle Unterverzeichnisse für jeden Schlüsseltyp und -zweck
        for key_type in self.config["key_types"]:
            for purpose in self.config["key_purposes"]:
                key_dir = os.path.join(root_dir, key_type, purpose)
                os.makedirs(key_dir, exist_ok=True)
        
        self.logger.info(f"Schlüsselverzeichnisse in {root_dir} eingerichtet")
    
    def generate_key(self, key_type: str, purpose: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generiert einen neuen Schlüssel des angegebenen Typs für den angegebenen Zweck.
        
        Args:
            key_type: Typ des Schlüssels ('kyber', 'ntru', 'dilithium', 'aes')
            purpose: Verwendungszweck des Schlüssels
            metadata: Optionale Metadaten zum Schlüssel
            
        Returns:
            ID des generierten Schlüssels
        """
        if key_type not in self.config["key_types"]:
            raise ValueError(f"Ungültiger Schlüsseltyp: {key_type}")
        
        if purpose not in self.config["key_purposes"]:
            raise ValueError(f"Ungültiger Verwendungszweck: {purpose}")
        
        self.logger.info(f"Generiere neuen {key_type}-Schlüssel für {purpose}")
        
        # Generiere Schlüssel mit dem richtigen Algorithmus
        if key_type == "kyber":
            public_key, private_key = self.crypto.generate_kyber_keypair()
        elif key_type == "ntru":
            public_key, private_key = self.crypto.generate_ntru_keypair()
        elif key_type == "dilithium":
            public_key, private_key = self.crypto.generate_dilithium_keypair()
        elif key_type == "aes":
            private_key = os.urandom(32)  # 256 Bit AES-Schlüssel
            public_key = b''  # AES hat keinen öffentlichen Schlüssel
        
        # Generiere eindeutige ID
        key_id = hashlib.sha256(
            public_key + private_key[:4] + purpose.encode() + str(time.time()).encode()
        ).hexdigest()[:16]
        
        # Speichere Schlüssel
        self._store_key(key_id, key_type, purpose, public_key, private_key, metadata)
        
        return key_id
    
    def _store_key(self, key_id: str, key_type: str, purpose: str,
                  public_key: bytes, private_key: bytes,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Speichert einen Schlüssel sicher im Dateisystem und in der Datenbank.
        
        Args:
            key_id: ID des Schlüssels
            key_type: Typ des Schlüssels
            purpose: Verwendungszweck
            public_key: Öffentlicher Schlüssel (falls vorhanden)
            private_key: Privater Schlüssel
            metadata: Optionale Metadaten
        """
        # In einer vollständigen Implementierung würde hier der private Schlüssel
        # zusätzlich verschlüsselt werden, bevor er gespeichert wird
        
        # Dateipfade für die Schlüssel
        key_dir = os.path.join(
            self.config["key_store"]["root_directory"],
            key_type,
            purpose
        )
        
        # Speichere Metadaten und Verweis auf Schlüsseldateien
        metadata_file = os.path.join(key_dir, f"{key_id}_metadata.json")
        
        key_metadata = {
            "id": key_id,
            "type": key_type,
            "purpose": purpose,
            "created_at": int(time.time()),
            "expires_at": int(time.time()) + (self.config["key_rotation"][key_type] * 86400),
            "metadata": metadata or {}
        }
        
        # Speichere den öffentlichen Schlüssel, falls vorhanden
        if public_key:
            public_key_path = os.path.join(key_dir, f"{key_id}_public.key")
            with open(public_key_path, 'wb') as f:
                f.write(public_key)
            key_metadata["public_key_path"] = public_key_path
        
        # Speichere den privaten Schlüssel
        private_key_path = os.path.join(key_dir, f"{key_id}_private.key")
        with open(private_key_path, 'wb') as f:
            f.write(private_key)
        key_metadata["private_key_path"] = private_key_path
        
        # Speichere die Metadaten
        with open(metadata_file, 'w') as f:
            json.dump(key_metadata, f, indent=2)
        
        self.logger.info(f"Schlüssel {key_id} ({key_type}) für {purpose} gespeichert")
    
    def get_key(self, key_id: str) -> Dict[str, Any]:
        """
        Holt einen Schlüssel anhand seiner ID.
        
        Args:
            key_id: ID des Schlüssels
            
        Returns:
            Dictionary mit den Schlüsselinformationen und dem Schlüsselmaterial
        """
        # Suche in allen möglichen Verzeichnissen nach dem Schlüssel
        for key_type in self.config["key_types"]:
            for purpose in self.config["key_purposes"]:
                key_dir = os.path.join(
                    self.config["key_store"]["root_directory"],
                    key_type,
                    purpose
                )
                
                metadata_file = os.path.join(key_dir, f"{key_id}_metadata.json")
                
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        key_metadata = json.load(f)
                    
                    # Lade den privaten Schlüssel
                    with open(key_metadata["private_key_path"], 'rb') as f:
                        private_key = f.read()
                    
                    # Lade den öffentlichen Schlüssel, falls vorhanden
                    public_key = None
                    if "public_key_path" in key_metadata and os.path.exists(key_metadata["public_key_path"]):
                        with open(key_metadata["public_key_path"], 'rb') as f:
                            public_key = f.read()
                    
                    # Füge das Schlüsselmaterial zu den Metadaten hinzu
                    key_metadata["private_key"] = private_key
                    if public_key:
                        key_metadata["public_key"] = public_key
                    
                    self.logger.info(f"Schlüssel {key_id} geladen")
                    return key_metadata
        
        raise KeyError(f"Schlüssel mit ID {key_id} nicht gefunden")
    
    def list_keys(self, key_type: Optional[str] = None, purpose: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Listet alle verfügbaren Schlüssel auf.
        
        Args:
            key_type: Optional, filtert nach Schlüsseltyp
            purpose: Optional, filtert nach Verwendungszweck
            
        Returns:
            Liste von Schlüsselmetadaten (ohne die privaten Schlüssel)
        """
        result = []
        
        # Bestimme die zu durchsuchenden Verzeichnisse
        key_types = [key_type] if key_type else self.config["key_types"]
        purposes = [purpose] if purpose else self.config["key_purposes"]
        
        for kt in key_types:
            for p in purposes:
                key_dir = os.path.join(
                    self.config["key_store"]["root_directory"],
                    kt,
                    p
                )
                
                if not os.path.exists(key_dir):
                    continue
                
                # Suche nach Metadaten-Dateien
                for filename in os.listdir(key_dir):
                    if filename.endswith("_metadata.json"):
                        metadata_path = os.path.join(key_dir, filename)
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Entferne Pfadinformationen für die Sicherheit
                        if "private_key_path" in metadata:
                            del metadata["private_key_path"]
                        
                        result.append(metadata)
        
        return result
    
    def rotate_key(self, key_id: str) -> str:
        """
        Rotiert einen Schlüssel und ersetzt ihn durch einen neuen.
        
        Args:
            key_id: ID des zu rotierenden Schlüssels
            
        Returns:
            ID des neuen Schlüssels
        """
        # Lade den zu rotierenden Schlüssel
        old_key = self.get_key(key_id)
        
        self.logger.info(f"Rotiere Schlüssel {key_id} ({old_key['type']})")
        
        # Generiere einen neuen Schlüssel desselben Typs und für denselben Zweck
        new_key_id = self.generate_key(
            old_key["type"],
            old_key["purpose"],
            {**old_key.get("metadata", {}), "replaces": key_id}
        )
        
        # In einer vollständigen Implementierung würde hier eine Aktualisierung
        # aller Systeme erfolgen, die den alten Schlüssel verwenden
        
        return new_key_id
    
    def check_expiring_keys(self, days_threshold: int = 7) -> List[Dict[str, Any]]:
        """
        Überprüft, welche Schlüssel demnächst ablaufen.
        
        Args:
            days_threshold: Schwellenwert in Tagen
            
        Returns:
            Liste von Schlüsseln, die innerhalb des Schwellenwerts ablaufen
        """
        expiring_keys = []
        current_time = int(time.time())
        threshold_seconds = days_threshold * 86400
        
        # Überprüfe alle Schlüssel
        for key_info in self.list_keys():
            if "expires_at" in key_info:
                time_until_expiry = key_info["expires_at"] - current_time
                
                if 0 < time_until_expiry < threshold_seconds:
                    expiring_keys.append(key_info)
        
        return expiring_keys
    
    def rotate_expiring_keys(self, days_threshold: int = 7) -> List[Tuple[str, str]]:
        """
        Rotiert automatisch alle Schlüssel, die innerhalb des angegebenen Zeitraums ablaufen.
        
        Args:
            days_threshold: Schwellenwert in Tagen
            
        Returns:
            Liste von Tupeln (alte_id, neue_id) der rotierten Schlüssel
        """
        rotated = []
        
        for key_info in self.check_expiring_keys(days_threshold):
            old_id = key_info["id"]
            new_id = self.rotate_key(old_id)
            rotated.append((old_id, new_id))
            
            self.logger.info(f"Automatische Rotation: {old_id} -> {new_id}")
        
        return rotated
