#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - VOID Kryptographie-Modul

Dieses Modul implementiert kryptographische Funktionen für die VOID-Protokoll-Integration.
Es stellt Verschlüsselungs-, Signatur- und Verifizierungsfunktionen bereit.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import hashlib
import hmac
import base64
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple

# Für fortgeschrittene Kryptographie
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    ADVANCED_CRYPTO_AVAILABLE = True
except ImportError:
    ADVANCED_CRYPTO_AVAILABLE = False

# Konfiguriere Logging
logger = logging.getLogger(__name__)

class CryptoStrength(object):
    """Konstanten für Kryptographie-Stärke"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class VoidCrypto:
    """Implementierung für VOID-Kryptographiefunktionen"""
    
    def __init__(self, security_level="high"):
        """Initialisiert die VOID-Kryptographie-Komponente
        
        Args:
            security_level: Sicherheitsstufe (low, medium, high, ultra)
        """
        self.security_level = security_level
        self.initialized = False
        self.keys = {}
        self.key_rotation_timestamps = {}
        self.key_rotation_interval = 86400  # 24 Stunden in Sekunden
        
        # Sekundärer Fallback-Schlüssel (wird nur verwendet, wenn key_derivation fehlschlägt)
        self.fallback_key = None
        
        logger.info(f"VoidCrypto Objekt erstellt (Stufe: {security_level})")
    
    def init(self):
        """Initialisiert die Sicherheitskomponente"""
        try:
            # Generiere initiale Schlüssel
            self._generate_keys()
            
            # Setze Initialisierungsstatus
            self.initialized = True
            self.key_rotation_timestamps["master"] = time.time()
            
            logger.info(f"VoidCrypto initialisiert (Level: {self.security_level}, Advanced: {ADVANCED_CRYPTO_AVAILABLE})")
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung von VoidCrypto: {e}")
            return False
    
    def _generate_keys(self):
        """Generiert kryptographische Schlüssel"""
        # Generiere Master-Schlüssel
        master_key = os.urandom(32)  # 256-Bit-Schlüssel
        self.keys["master"] = master_key
        
        # Generiere abgeleitete Schlüssel für verschiedene Zwecke
        self.keys["encryption"] = self._derive_key(master_key, b"encryption", 32)
        self.keys["signing"] = self._derive_key(master_key, b"signing", 32)
        self.keys["verification"] = self._derive_key(master_key, b"verification", 32)
        
        # Fallback-Schlüssel für Notfälle
        self.fallback_key = hashlib.sha256(os.urandom(32)).digest()
    
    def _derive_key(self, base_key: bytes, purpose: bytes, length: int) -> bytes:
        """Leitet einen Schlüssel für einen bestimmten Zweck ab
        
        Args:
            base_key: Basis-Schlüssel
            purpose: Verwendungszweck
            length: Schlüssellänge in Bytes
            
        Returns:
            Abgeleiteter Schlüssel
        """
        try:
            if ADVANCED_CRYPTO_AVAILABLE:
                # Verwende PBKDF2 mit fortschrittlichen Kryptographie-Bibliotheken
                salt = hashlib.sha256(purpose).digest()
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=length,
                    salt=salt,
                    iterations=100000,
                    backend=default_backend()
                )
                return kdf.derive(base_key)
            else:
                # Fallback: Verwende HMAC mit hashlib
                return hmac.new(base_key, purpose, hashlib.sha256).digest()[:length]
        except Exception as e:
            logger.error(f"Fehler bei der Schlüsselableitung: {e}")
            return hashlib.sha256(base_key + purpose).digest()[:length]  # Einfacher Fallback
    
    def _check_rotation(self, key_name: str):
        """Prüft, ob ein Schlüssel rotiert werden muss und führt die Rotation durch
        
        Args:
            key_name: Name des Schlüssels
        """
        last_rotation = self.key_rotation_timestamps.get(key_name, 0)
        if time.time() - last_rotation > self.key_rotation_interval:
            # Rotiere den Schlüssel
            if key_name == "master":
                # Generiere alle Schlüssel neu
                self._generate_keys()
            else:
                # Rotiere nur den spezifischen Schlüssel
                self.keys[key_name] = self._derive_key(self.keys["master"], key_name.encode(), len(self.keys[key_name]))
            
            # Aktualisiere den Zeitstempel
            self.key_rotation_timestamps[key_name] = time.time()
            logger.info(f"Schlüssel rotiert: {key_name}")
    
    def encrypt(self, data: Union[bytes, str, dict]) -> Dict[str, Any]:
        """Verschlüsselt Daten
        
        Args:
            data: Zu verschlüsselnde Daten (Bytes, String oder Dictionary)
            
        Returns:
            Dict mit verschlüsselten Daten und Metadaten
        """
        if not self.initialized:
            logger.error("VoidCrypto nicht initialisiert")
            return {"success": False, "error": "Not initialized"}
        
        try:
            # Prüfe Schlüsselrotation
            self._check_rotation("encryption")
            
            # Konvertiere Daten zu Bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode()
            elif isinstance(data, str):
                data_bytes = data.encode()
            else:
                data_bytes = data
            
            # Generiere Initialisierungsvektor
            iv = os.urandom(16)  # 128 Bit
            
            if ADVANCED_CRYPTO_AVAILABLE:
                # Verwende AES-GCM für authentifizierte Verschlüsselung
                cipher = Cipher(
                    algorithms.AES(self.keys["encryption"]),
                    modes.GCM(iv),
                    backend=default_backend()
                )
                encryptor = cipher.encryptor()
                ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
                tag = encryptor.tag
                
                return {
                    "success": True,
                    "ciphertext": base64.b64encode(ciphertext).decode(),
                    "iv": base64.b64encode(iv).decode(),
                    "tag": base64.b64encode(tag).decode(),
                    "timestamp": time.time(),
                    "method": "aes-gcm"
                }
            else:
                # Fallback: Verwende einfache XOR-Verschlüsselung mit einem abgeleiteten Schlüssel
                # HINWEIS: Dies ist nicht sicher und sollte nur als Fallback verwendet werden
                key = self.keys["encryption"]
                key_stream = []
                
                while len(key_stream) < len(data_bytes):
                    key_stream.extend(hashlib.sha256(key + iv + str(len(key_stream)).encode()).digest())
                
                key_stream = bytes(key_stream[:len(data_bytes)])
                ciphertext = bytes(a ^ b for a, b in zip(data_bytes, key_stream))
                
                return {
                    "success": True,
                    "ciphertext": base64.b64encode(ciphertext).decode(),
                    "iv": base64.b64encode(iv).decode(),
                    "timestamp": time.time(),
                    "method": "xor-stream"
                }
                
        except Exception as e:
            logger.error(f"Fehler bei der Verschlüsselung: {e}")
            return {"success": False, "error": str(e)}
    
    def decrypt(self, encrypted_data: Dict[str, Any]) -> Union[bytes, None]:
        """Entschlüsselt Daten
        
        Args:
            encrypted_data: Verschlüsselte Daten und Metadaten
            
        Returns:
            Entschlüsselte Daten oder None bei Fehler
        """
        if not self.initialized:
            logger.error("VoidCrypto nicht initialisiert")
            return None
        
        try:
            ciphertext = base64.b64decode(encrypted_data["ciphertext"])
            iv = base64.b64decode(encrypted_data["iv"])
            method = encrypted_data.get("method", "unknown")
            
            if method == "aes-gcm" and ADVANCED_CRYPTO_AVAILABLE:
                # AES-GCM Entschlüsselung
                tag = base64.b64decode(encrypted_data["tag"])
                cipher = Cipher(
                    algorithms.AES(self.keys["encryption"]),
                    modes.GCM(iv, tag),
                    backend=default_backend()
                )
                decryptor = cipher.decryptor()
                return decryptor.update(ciphertext) + decryptor.finalize()
                
            elif method == "xor-stream":
                # XOR-Stream Entschlüsselung
                key = self.keys["encryption"]
                key_stream = []
                
                while len(key_stream) < len(ciphertext):
                    key_stream.extend(hashlib.sha256(key + iv + str(len(key_stream)).encode()).digest())
                
                key_stream = bytes(key_stream[:len(ciphertext)])
                return bytes(a ^ b for a, b in zip(ciphertext, key_stream))
                
            else:
                logger.error(f"Unbekannte Verschlüsselungsmethode: {method}")
                return None
                
        except Exception as e:
            logger.error(f"Fehler bei der Entschlüsselung: {e}")
            return None
    
    def sign(self, data: Union[bytes, str, dict]) -> Dict[str, Any]:
        """Signiert Daten
        
        Args:
            data: Zu signierende Daten
            
        Returns:
            Signatur und Metadaten
        """
        if not self.initialized:
            logger.error("VoidCrypto nicht initialisiert")
            return {"success": False, "error": "Not initialized"}
        
        try:
            # Prüfe Schlüsselrotation
            self._check_rotation("signing")
            
            # Konvertiere Daten zu Bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode()
            elif isinstance(data, str):
                data_bytes = data.encode()
            else:
                data_bytes = data
            
            # Erstelle Signatur
            if ADVANCED_CRYPTO_AVAILABLE:
                h = crypto_hmac.HMAC(self.keys["signing"], hashes.SHA256(), backend=default_backend())
                h.update(data_bytes)
                signature = h.finalize()
            else:
                signature = hmac.new(self.keys["signing"], data_bytes, hashlib.sha256).digest()
            
            return {
                "success": True,
                "signature": base64.b64encode(signature).decode(),
                "timestamp": time.time(),
                "id": str(uuid.uuid4())
            }
            
        except Exception as e:
            logger.error(f"Fehler bei der Signierung: {e}")
            return {"success": False, "error": str(e)}
    
    def verify(self, data: Union[bytes, str, dict], signature_info: Optional[Dict[str, Any]] = None) -> bool:
        """Überprüft die Signatur von Daten
        
        Args:
            data: Zu überprüfende Daten
            signature_info: Signatur und Metadaten (optional)
            
        Returns:
            True, wenn die Signatur gültig ist, sonst False
        """
        if not self.initialized:
            logger.error("VoidCrypto nicht initialisiert")
            return False
        
        # Wenn keine Signatur angegeben ist, vertraue den Daten (für Abwärtskompatibilität)
        if signature_info is None:
            return True
        
        try:
            # Konvertiere Daten zu Bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode()
            elif isinstance(data, str):
                data_bytes = data.encode()
            else:
                data_bytes = data
            
            # Hole Signatur
            signature = base64.b64decode(signature_info["signature"])
            
            # Überprüfe Signatur
            if ADVANCED_CRYPTO_AVAILABLE:
                h = crypto_hmac.HMAC(self.keys["signing"], hashes.SHA256(), backend=default_backend())
                h.update(data_bytes)
                try:
                    h.verify(signature)
                    return True
                except Exception:
                    return False
            else:
                expected_signature = hmac.new(self.keys["signing"], data_bytes, hashlib.sha256).digest()
                return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Fehler bei der Signaturüberprüfung: {e}")
            return False
    
    def secure(self, data: Any) -> Any:
        """Schützt Daten gemäß Sicherheitsrichtlinien
        
        Args:
            data: Zu schützende Daten
            
        Returns:
            Geschützte Daten
        """
        if not self.initialized:
            return data
        
        try:
            # Für primitive Datentypen, gib sie unverändert zurück
            if isinstance(data, (int, float, bool, type(None))):
                return data
            
            # Für komplexere Datentypen, verschlüssele sie
            encrypted = self.encrypt(data)
            if encrypted.get("success", False):
                # Signiere die verschlüsselten Daten
                signature = self.sign(encrypted["ciphertext"])
                if signature.get("success", False):
                    encrypted["signature"] = signature["signature"]
                
                return encrypted
            
            # Bei Fehler gib die Originaldaten zurück
            return data
            
        except Exception as e:
            logger.error(f"Fehler beim Sichern von Daten: {e}")
            return data

# Modul-Initialisierung
def init():
    """Initialisiert das VOID-Kryptographie-Modul"""
    # Lade Konfiguration, um die zu verwendende Sicherheitsstufe zu ermitteln
    security_level = "high"  # Standardwert
    config_path = os.path.join(os.path.dirname(__file__), 'void_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"VOID-Konfiguration geladen: {config_path}")
                # Verwende die konfigurierte Sicherheitsstufe, falls vorhanden
                if 'security_levels' in config:
                    # Hier könnte eine Bestimmung der optimalen Sicherheitsstufe erfolgen
                    security_level = list(config['security_levels'].keys())[-1]  # Verwende höchste Stufe
                    logger.debug(f"Verwende Sicherheitsstufe: {security_level}")
        except Exception as e:
            logger.error(f"Fehler beim Laden der VOID-Konfiguration: {e}")
    else:
        logger.warning(f"VOID-Konfigurationsdatei nicht gefunden: {config_path}, verwende Standardkonfiguration")
    
    # Initialisiere mit der ermittelten Sicherheitsstufe
    component = VoidCrypto(security_level)
    return component.init()
