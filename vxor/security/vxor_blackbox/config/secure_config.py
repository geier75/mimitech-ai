#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sichere Konfigurationskomponente für VXOR AI Blackbox.

Ermöglicht die Verschlüsselung sensibler Konfigurationswerte, um zu verhindern,
dass vertrauliche Daten wie API-Schlüssel oder Zugangsdaten im Klartext in
Konfigurationsdateien gespeichert werden.
"""

import os
import base64
import hashlib
import logging
from typing import Any, Optional, Union, Dict

# In einer vollständigen Implementierung würde hier die Kryptographiekomponente importiert
from ..crypto.aes import AESCrypto

class SecureConfigValue:
    """
    Repräsentiert einen verschlüsselten Konfigurationswert.
    
    Diese Klasse kapselt einen verschlüsselten Wert zusammen mit den
    Metadaten, die für seine Entschlüsselung benötigt werden, und bietet
    eine sichere Möglichkeit, vertrauliche Konfigurationswerte zu speichern.
    """
    
    def __init__(self, encrypted_data: bytes, salt: bytes, iv: bytes, tag: bytes):
        """
        Initialisiert einen verschlüsselten Konfigurationswert.
        
        Args:
            encrypted_data: Die verschlüsselten Daten
            salt: Das Salt für die Schlüsselableitung
            iv: Der Initialisierungsvektor für die Verschlüsselung
            tag: Der Authentifizierungstag (für AES-GCM)
        """
        self.encrypted_data = encrypted_data
        self.salt = salt
        self.iv = iv
        self.tag = tag
    
    def to_dict(self) -> Dict[str, str]:
        """
        Konvertiert den verschlüsselten Wert in ein serialisierbares Dictionary.
        
        Returns:
            Ein Dictionary mit Base64-codierten Werten
        """
        return {
            "data": base64.b64encode(self.encrypted_data).decode('utf-8'),
            "salt": base64.b64encode(self.salt).decode('utf-8'),
            "iv": base64.b64encode(self.iv).decode('utf-8'),
            "tag": base64.b64encode(self.tag).decode('utf-8'),
            "__secure": True
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'SecureConfigValue':
        """
        Erstellt einen SecureConfigValue aus einem serialisierten Dictionary.
        
        Args:
            data: Das Dictionary mit den verschlüsselten Daten
            
        Returns:
            Eine neue SecureConfigValue-Instanz
        """
        if not data.get("__secure", False):
            raise ValueError("Das Dictionary enthält keinen sicheren Konfigurationswert")
        
        return cls(
            encrypted_data=base64.b64decode(data["data"]),
            salt=base64.b64decode(data["salt"]),
            iv=base64.b64decode(data["iv"]),
            tag=base64.b64decode(data["tag"])
        )
    
    def __repr__(self) -> str:
        """String-Repräsentation des sicheren Konfigurationswerts."""
        return f"<SecureConfigValue: {len(self.encrypted_data)} bytes>"


def encrypt_config_value(value: str, passphrase: str) -> SecureConfigValue:
    """
    Verschlüsselt einen Konfigurationswert mit einer Passphrase.
    
    Args:
        value: Der zu verschlüsselnde Wert
        passphrase: Die Passphrase für die Verschlüsselung
        
    Returns:
        Ein verschlüsselter SecureConfigValue
    """
    logger = logging.getLogger("vxor.security.config.secure")
    
    try:
        # Generiere Salt für die Schlüsselableitung
        salt = os.urandom(16)
        
        # Leite Schlüssel aus Passphrase ab
        key = _derive_key_from_passphrase(passphrase, salt)
        
        # Verschlüssele den Wert
        aes = AESCrypto()
        encrypted_data = aes.encrypt(value.encode('utf-8'), key)
        
        # Extrahiere IV und Tag aus den verschlüsselten Daten
        # In einer realen Implementierung würden diese von der AES-Implementierung kommen
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        actual_encrypted_data = encrypted_data[28:]
        
        return SecureConfigValue(actual_encrypted_data, salt, iv, tag)
    
    except Exception as e:
        logger.error(f"Fehler bei der Verschlüsselung des Konfigurationswerts: {str(e)}")
        raise


def decrypt_config_value(secure_value: SecureConfigValue, passphrase: str) -> str:
    """
    Entschlüsselt einen verschlüsselten Konfigurationswert.
    
    Args:
        secure_value: Der verschlüsselte Wert
        passphrase: Die Passphrase für die Entschlüsselung
        
    Returns:
        Der entschlüsselte Wert als String
    """
    logger = logging.getLogger("vxor.security.config.secure")
    
    try:
        # Leite Schlüssel aus Passphrase ab
        key = _derive_key_from_passphrase(passphrase, secure_value.salt)
        
        # Rekonstruiere die vollständigen verschlüsselten Daten
        full_encrypted_data = secure_value.iv + secure_value.tag + secure_value.encrypted_data
        
        # Entschlüssele den Wert
        aes = AESCrypto()
        decrypted_data = aes.decrypt(full_encrypted_data, key)
        
        return decrypted_data.decode('utf-8')
    
    except Exception as e:
        logger.error(f"Fehler bei der Entschlüsselung des Konfigurationswerts: {str(e)}")
        raise


def _derive_key_from_passphrase(passphrase: str, salt: bytes) -> bytes:
    """
    Leitet einen Schlüssel aus einer Passphrase ab.
    
    Args:
        passphrase: Die Passphrase
        salt: Das Salt für die Schlüsselableitung
        
    Returns:
        Der abgeleitete Schlüssel
    """
    # In einer vollständigen Implementierung würde hier PBKDF2, Argon2 oder eine
    # andere sichere Schlüsselableitungsfunktion verwendet werden
    
    # Einfache Demonstration mit einer Hash-Funktion (NICHT für Produktion verwenden!)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        passphrase.encode('utf-8'),
        salt,
        iterations=600000,
        dklen=32  # 256 Bit Schlüssel
    )
    
    return key
