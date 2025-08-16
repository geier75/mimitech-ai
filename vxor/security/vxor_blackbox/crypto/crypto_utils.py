#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kryptographische Hilfsfunktionen für VXOR AI Blackbox.

Dieses Modul bietet spezialisierte Hilfsfunktionen für die kryptographischen
Operationen in VXOR AI, insbesondere für:
- Schlüsselableitung und -verwaltung
- Padding und Entpadding von Daten
- Hashfunktionen und HMAC-Operationen
- Zufallszahlengenerierung

Alle Funktionen sind speziell für die Anforderungen der T-Mathematics Engine
und M-LINGUA optimiert.
"""

import os
import hashlib
import hmac
import base64
import struct
import math
import secrets
import binascii
from typing import Dict, Any, Optional, Union, List, Tuple, ByteString, Callable, BinaryIO

# Für Passwort-basierte Schlüsselableitung
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.x963kdf import X963KDF
from cryptography.hazmat.primitives import hashes

# Für zufällige Byte-Generierung
from cryptography.hazmat.primitives.padding import PKCS7
from cryptography.hazmat.primitives.constant_time import bytes_eq

# Für den Logger
try:
    from ..logging import get_logger
except ImportError:
    # Fallback, falls die Logging-Komponente nicht verfügbar ist
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

# Global logger
logger = get_logger("crypto.utils")


# ===== Schlüsselableitungsfunktionen =====

def derive_key_from_password(password: Union[str, bytes], salt: Optional[bytes] = None, 
                           iterations: int = 600000, key_length: int = 32) -> Tuple[bytes, bytes]:
    """
    Leitet einen kryptographischen Schlüssel aus einem Passwort ab.
    
    Diese Funktion verwendet PBKDF2 mit HMAC-SHA256 für eine sichere,
    brute-force-resistente Schlüsselableitung.
    
    Args:
        password: Das Ausgangspasswort (als String oder Bytes)
        salt: Das zu verwendende Salt (falls None, wird ein zufälliges Salt generiert)
        iterations: Die Anzahl der Iterationen für PBKDF2
        key_length: Die Länge des abgeleiteten Schlüssels in Bytes (32 = 256 Bit)
        
    Returns:
        Tuple aus (abgeleiteter Schlüssel, verwendetes Salt)
    """
    # Konvertiere String-Passwort zu Bytes, falls nötig
    if isinstance(password, str):
        password = password.encode('utf-8')
    
    # Generiere Salt, falls keines angegeben
    if salt is None:
        salt = os.urandom(32)  # 256 Bit Salt
    
    # Erstelle PBKDF2HMAC-Instanz
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=key_length,
        salt=salt,
        iterations=iterations,
    )
    
    # Leite Schlüssel ab
    key = kdf.derive(password)
    
    logger.debug(f"Schlüssel aus Passwort abgeleitet: {len(key)*8} Bit")
    return key, salt


def derive_key_from_master(master_key: bytes, context: Union[str, bytes], 
                         key_length: int = 32) -> bytes:
    """
    Leitet einen sekundären Schlüssel aus einem Master-Schlüssel ab.
    
    Diese Funktion verwendet HKDF (HMAC-based Key Derivation Function)
    für die sichere Ableitung mehrerer Schlüssel aus einem Master-Schlüssel.
    
    Args:
        master_key: Der Master-Schlüssel, aus dem abgeleitet wird
        context: Kontextinformation (z.B. "encryption", "authentication")
        key_length: Die Länge des abgeleiteten Schlüssels in Bytes
        
    Returns:
        Der abgeleitete Schlüssel
    """
    # Konvertiere String-Kontext zu Bytes, falls nötig
    if isinstance(context, str):
        context = context.encode('utf-8')
    
    # Erstelle HKDF-Instanz
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=key_length,
        salt=None,
        info=context,
    )
    
    # Leite Schlüssel ab
    derived_key = hkdf.derive(master_key)
    
    logger.debug(f"Sekundärer Schlüssel abgeleitet für Kontext: {context}")
    return derived_key


def derive_tensor_specific_key(master_key: bytes, tensor_info: Dict[str, Any], 
                            key_length: int = 32) -> bytes:
    """
    Leitet einen tensorspezifischen Schlüssel aus einem Master-Schlüssel ab.
    
    Diese Funktion ist speziell für die T-Mathematics Engine optimiert und
    berücksichtigt Tensorform, Typ und andere Metadaten bei der Schlüsselableitung.
    
    Args:
        master_key: Der Master-Schlüssel, aus dem abgeleitet wird
        tensor_info: Dictionary mit Tensor-Metadaten (Form, Typ, etc.)
        key_length: Die Länge des abgeleiteten Schlüssels in Bytes
        
    Returns:
        Der tensorspezifische Schlüssel
    """
    # Erstelle eindeutigen Kontext aus Tensor-Metadaten
    context_parts = []
    
    # Füge Tensor-Form hinzu
    if 'shape' in tensor_info:
        shape_str = 'x'.join(str(dim) for dim in tensor_info['shape'])
        context_parts.append(f"shape:{shape_str}")
    
    # Füge Datentyp hinzu
    if 'dtype' in tensor_info:
        context_parts.append(f"dtype:{tensor_info['dtype']}")
    
    # Füge Format hinzu
    if 'format' in tensor_info:
        context_parts.append(f"format:{tensor_info['format']}")
    
    # Füge Device hinzu (falls vorhanden)
    if 'device' in tensor_info:
        context_parts.append(f"device:{tensor_info['device']}")
    
    # Erstelle Kontext-String
    context = "|".join(context_parts).encode('utf-8')
    
    # Verwende X963KDF für tensorspezifische Schlüsselableitung
    # Diese KDF ist besonders gut für Hardware-Acceleration geeignet
    kdf = X963KDF(
        algorithm=hashes.SHA256(),
        length=key_length,
        sharedinfo=context,
    )
    
    # Leite Schlüssel ab
    derived_key = kdf.derive(master_key)
    
    logger.debug(f"Tensorspezifischer Schlüssel abgeleitet für: {context}")
    return derived_key


def create_key_hierarchy(master_password: str, user_id: str, 
                       tensor_info: Optional[Dict[str, Any]] = None) -> Dict[str, bytes]:
    """
    Erstellt eine vollständige Schlüsselhierarchie für VXOR AI Blackbox.
    
    Diese Funktion generiert eine Hierarchie von Schlüsseln aus einem Master-Passwort:
    - Master Key (MK): Direkt aus dem Passwort abgeleitet
    - Key Encryption Key (KEK): Verschlüsselt andere Schlüssel
    - Data Encryption Key (DEK): Verschlüsselt Daten
    - Authentication Key (AK): Für HMAC und Authentifizierung
    - Optional: Tensor-spezifischer Schlüssel
    
    Args:
        master_password: Das Master-Passwort des Benutzers
        user_id: Die eindeutige Benutzer-ID
        tensor_info: Optional, Tensor-Metadaten für tensorspezifischen Schlüssel
        
    Returns:
        Dictionary mit allen generierten Schlüsseln
    """
    # Generiere ein benutzerabhängiges Salt
    salt_base = f"VXOR_AI_{user_id}_SALT".encode('utf-8')
    salt = hashlib.sha256(salt_base).digest()
    
    # Derive Master Key (MK)
    master_key, _ = derive_key_from_password(master_password, salt=salt)
    
    # Derive Key Encryption Key (KEK)
    kek = derive_key_from_master(master_key, "VXOR_AI_KEK")
    
    # Derive Data Encryption Key (DEK)
    dek = derive_key_from_master(master_key, "VXOR_AI_DEK")
    
    # Derive Authentication Key (AK)
    auth_key = derive_key_from_master(master_key, "VXOR_AI_AUTH")
    
    # Erstelle das Rückgabe-Dictionary
    key_hierarchy = {
        "master_key": master_key,
        "kek": kek,
        "dek": dek,
        "auth_key": auth_key
    }
    
    # Optional: Tensor-spezifischer Schlüssel
    if tensor_info:
        tensor_key = derive_tensor_specific_key(dek, tensor_info)
        key_hierarchy["tensor_key"] = tensor_key
    
    logger.info(f"Schlüsselhierarchie erstellt für Benutzer: {user_id}")
    return key_hierarchy


# ===== Padding-Funktionen =====

def pad_data(data: bytes, block_size: int = 16) -> bytes:
    """
    Wendet PKCS#7-Padding auf die Daten an.
    
    Args:
        data: Die zu paddenden Daten
        block_size: Die Blockgröße in Bytes
        
    Returns:
        Die gepaddeten Daten
    """
    padder = PKCS7(block_size * 8).padder()
    padded_data = padder.update(data) + padder.finalize()
    return padded_data


def unpad_data(padded_data: bytes, block_size: int = 16) -> bytes:
    """
    Entfernt PKCS#7-Padding von den Daten.
    
    Args:
        padded_data: Die gepaddeten Daten
        block_size: Die Blockgröße in Bytes
        
    Returns:
        Die ungepaddeten Daten
    """
    unpadder = PKCS7(block_size * 8).unpadder()
    try:
        data = unpadder.update(padded_data) + unpadder.finalize()
        return data
    except ValueError as e:
        logger.error(f"Fehler beim Entpadden: {str(e)}")
        raise ValueError("Padding-Fehler: Die Daten wurden möglicherweise manipuliert.")


def pad_tensor_data(tensor_data: bytes, pad_strategy: str = "pkcs7") -> bytes:
    """
    Wendet spezielles Padding für Tensor-Daten an.
    
    Diese Funktion ist für die speziellen Anforderungen der Tensor-Verschlüsselung
    optimiert und unterstützt verschiedene Padding-Strategien.
    
    Args:
        tensor_data: Die zu paddenden Tensor-Daten
        pad_strategy: Die Padding-Strategie ("pkcs7", "zero", "tensor_specific")
        
    Returns:
        Die gepaddeten Tensor-Daten
    """
    if pad_strategy == "pkcs7":
        # Standard-PKCS#7-Padding
        return pad_data(tensor_data)
    
    elif pad_strategy == "zero":
        # Zero-Padding (für bestimmte Tensor-Operationen besser geeignet)
        remainder = len(tensor_data) % 16
        if remainder == 0:
            return tensor_data
        
        padding_size = 16 - remainder
        return tensor_data + (b'\x00' * padding_size)
    
    elif pad_strategy == "tensor_specific":
        # Tensor-spezifisches Padding, das Form und Struktur respektiert
        # Fügt Informationen über die Originalgröße hinzu
        padding_size = (16 - (len(tensor_data) % 16)) % 16
        
        # Speichere die Originalgröße in den ersten 8 Bytes des Paddings
        size_bytes = struct.pack(">Q", len(tensor_data))
        
        # Füge zufällige Bytes für das restliche Padding hinzu
        if padding_size > 8:
            random_padding = os.urandom(padding_size - 8)
            return tensor_data + size_bytes + random_padding
        elif padding_size == 0:
            # Ein zusätzlicher Block mit Größeninformation
            random_padding = os.urandom(8)
            return tensor_data + size_bytes + random_padding
        else:
            # Seltener Fall: Füge einen ganzen Block hinzu
            padding_size = 16
            random_padding = os.urandom(padding_size - 8)
            return tensor_data + size_bytes + random_padding
    
    else:
        raise ValueError(f"Unbekannte Padding-Strategie: {pad_strategy}")


def unpad_tensor_data(padded_tensor_data: bytes, pad_strategy: str = "pkcs7") -> bytes:
    """
    Entfernt spezielles Padding von Tensor-Daten.
    
    Args:
        padded_tensor_data: Die gepaddeten Tensor-Daten
        pad_strategy: Die verwendete Padding-Strategie
        
    Returns:
        Die ungepaddeten Tensor-Daten
    """
    if pad_strategy == "pkcs7":
        # Standard-PKCS#7-Unpadding
        return unpad_data(padded_tensor_data)
    
    elif pad_strategy == "zero":
        # Zero-Padding entfernen
        # Wir müssen die tatsächliche Größe rekonstruieren
        # Dies wird normalerweise aus Metadaten abgeleitet
        # In diesem Fall entfernen wir einfach alle Nullen am Ende
        return padded_tensor_data.rstrip(b'\x00')
    
    elif pad_strategy == "tensor_specific":
        # Tensor-spezifisches Padding entfernen
        # Extrahiere die ursprüngliche Größe aus den ersten 8 Bytes des Paddings
        padding_start = len(padded_tensor_data) - (len(padded_tensor_data) % 16 or 16)
        size_bytes = padded_tensor_data[padding_start:padding_start+8]
        original_size = struct.unpack(">Q", size_bytes)[0]
        
        if original_size > len(padded_tensor_data):
            logger.error("Ungültige Größenangabe im Padding")
            raise ValueError("Padding-Fehler: Die Daten wurden möglicherweise manipuliert.")
        
        return padded_tensor_data[:original_size]
    
    else:
        raise ValueError(f"Unbekannte Padding-Strategie: {pad_strategy}")


# ===== Hash- und HMAC-Funktionen =====

def compute_hash(data: Union[str, bytes], algorithm: str = "sha256") -> bytes:
    """
    Berechnet einen Hashwert für die gegebenen Daten.
    
    Args:
        data: Die zu hashenden Daten
        algorithm: Der zu verwendende Hash-Algorithmus
        
    Returns:
        Der berechnete Hashwert
    """
    # Konvertiere String zu Bytes, falls nötig
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    if algorithm == "sha256":
        return hashlib.sha256(data).digest()
    elif algorithm == "sha512":
        return hashlib.sha512(data).digest()
    elif algorithm == "sha3_256":
        return hashlib.sha3_256(data).digest()
    elif algorithm == "sha3_512":
        return hashlib.sha3_512(data).digest()
    else:
        raise ValueError(f"Nicht unterstützter Hash-Algorithmus: {algorithm}")


def compute_hmac(data: Union[str, bytes], key: bytes, algorithm: str = "sha256") -> bytes:
    """
    Berechnet einen HMAC-Wert für die gegebenen Daten.
    
    Args:
        data: Die zu authentifizierenden Daten
        key: Der Schlüssel für den HMAC
        algorithm: Der zu verwendende Hash-Algorithmus
        
    Returns:
        Der berechnete HMAC-Wert
    """
    # Konvertiere String zu Bytes, falls nötig
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    if algorithm == "sha256":
        return hmac.new(key, data, hashlib.sha256).digest()
    elif algorithm == "sha512":
        return hmac.new(key, data, hashlib.sha512).digest()
    elif algorithm == "sha3_256":
        # HMAC mit SHA3 benötigt spezielle Behandlung
        blocksize = 136  # SHA3-256 Blockgröße
        if len(key) > blocksize:
            key = hashlib.sha3_256(key).digest()
        
        # Folge dem HMAC-Konstruktionsverfahren
        o_key_pad = bytes(b ^ 0x5c for b in key.ljust(blocksize, b'\x00'))
        i_key_pad = bytes(b ^ 0x36 for b in key.ljust(blocksize, b'\x00'))
        
        inner = hashlib.sha3_256(i_key_pad + data).digest()
        return hashlib.sha3_256(o_key_pad + inner).digest()
    else:
        raise ValueError(f"Nicht unterstützter HMAC-Algorithmus: {algorithm}")


def compute_tensor_hmac(tensor_data: bytes, key: bytes, metadata: Dict[str, Any] = None) -> bytes:
    """
    Berechnet einen HMAC-Wert speziell für Tensor-Daten.
    
    Diese Funktion berücksichtigt die Tensor-Metadaten (Form, Typ, etc.)
    bei der HMAC-Berechnung, um die Integrität und Authentizität zu garantieren.
    
    Args:
        tensor_data: Die Tensor-Daten
        key: Der Schlüssel für den HMAC
        metadata: Optional, Dictionary mit Tensor-Metadaten
        
    Returns:
        Der berechnete HMAC-Wert
    """
    if metadata:
        # Füge Metadaten zur HMAC-Berechnung hinzu
        metadata_str = json.dumps(metadata, sort_keys=True).encode('utf-8')
        combined_data = metadata_str + b"||" + tensor_data
        return compute_hmac(combined_data, key, "sha256")
    else:
        # Nur Daten-HMAC
        return compute_hmac(tensor_data, key, "sha256")


# ===== Zufallszahlenfunktionen =====

def generate_random_bytes(length: int) -> bytes:
    """
    Generiert kryptographisch sichere Zufallsbytes.
    
    Args:
        length: Die Anzahl der zu generierenden Bytes
        
    Returns:
        Die generierten Zufallsbytes
    """
    return secrets.token_bytes(length)


def generate_random_iv(length: int = 12) -> bytes:
    """
    Generiert einen zufälligen Initialisierungsvektor (IV).
    
    Args:
        length: Die Länge des IV in Bytes
        
    Returns:
        Der generierte IV
    """
    return generate_random_bytes(length)


def generate_random_nonce(length: int = 16) -> bytes:
    """
    Generiert einen zufälligen Nonce.
    
    Args:
        length: Die Länge des Nonce in Bytes
        
    Returns:
        Der generierte Nonce
    """
    return generate_random_bytes(length)


# ===== Sonstige Hilfsfunktionen =====

def constant_time_compare(a: bytes, b: bytes) -> bool:
    """
    Vergleicht zwei Byte-Strings in konstanter Zeit.
    
    Diese Funktion ist wichtig für die Vermeidung von Timing-Angriffen.
    
    Args:
        a: Der erste Byte-String
        b: Der zweite Byte-String
        
    Returns:
        True, wenn die Strings gleich sind, sonst False
    """
    return bytes_eq(a, b)


def secure_encode(data: bytes, encoding: str = "base64") -> str:
    """
    Kodiert Binärdaten in ein sicheres Stringformat.
    
    Args:
        data: Die zu kodierenden Daten
        encoding: Das zu verwendende Encoding
        
    Returns:
        Die kodierten Daten als String
    """
    if encoding == "base64":
        return base64.b64encode(data).decode('utf-8')
    elif encoding == "hex":
        return binascii.hexlify(data).decode('utf-8')
    else:
        raise ValueError(f"Nicht unterstütztes Encoding: {encoding}")


def secure_decode(encoded_data: str, encoding: str = "base64") -> bytes:
    """
    Dekodiert einen sicheren String zurück zu Binärdaten.
    
    Args:
        encoded_data: Die zu dekodierenden Daten
        encoding: Das verwendete Encoding
        
    Returns:
        Die dekodierten Binärdaten
    """
    try:
        if encoding == "base64":
            return base64.b64decode(encoded_data)
        elif encoding == "hex":
            return binascii.unhexlify(encoded_data)
        else:
            raise ValueError(f"Nicht unterstütztes Encoding: {encoding}")
    except (binascii.Error, ValueError) as e:
        logger.error(f"Fehler beim Dekodieren: {str(e)}")
        raise ValueError("Dekodierungsfehler: Ungültiges Format.")


# ===== Erweiterungspunkte für zukünftige Implementierungen =====

def split_key(key: bytes, num_parts: int = 2) -> List[bytes]:
    """
    Teilt einen Schlüssel in mehrere Teile auf (Shamir Secret Sharing).
    
    Args:
        key: Der aufzuteilende Schlüssel
        num_parts: Die Anzahl der Teile
        
    Returns:
        Liste mit den Schlüsselteilen
    """
    # Einfache Implementierung (für Produktionscode wäre eine richtige
    # Secret-Sharing-Implementierung wie Shamir's Secret Sharing nötig)
    parts = []
    
    for i in range(num_parts - 1):
        part = generate_random_bytes(len(key))
        parts.append(part)
    
    # Letzten Teil so berechnen, dass XOR aller Teile = Originalschlüssel
    last_part = key
    for part in parts:
        last_part = bytes(a ^ b for a, b in zip(last_part, part))
    
    parts.append(last_part)
    
    return parts


def combine_key_parts(parts: List[bytes]) -> bytes:
    """
    Kombiniert Schlüsselteile zurück zum Originalschlüssel.
    
    Args:
        parts: Die Schlüsselteile
        
    Returns:
        Der rekonstruierte Schlüssel
    """
    if not parts:
        raise ValueError("Keine Schlüsselteile angegeben")
    
    # XOR aller Teile
    result = parts[0]
    for part in parts[1:]:
        result = bytes(a ^ b for a, b in zip(result, part))
    
    return result


# Hauptfunktion zum Testen der Utilities
def test_crypto_utils():
    """Führt einen einfachen Test der Crypto-Utilities durch."""
    logger.info("Teste VXOR AI Crypto-Utilities...")
    
    # Test: Schlüsselableitung
    password = "VXORSecretPassword123!"
    key, salt = derive_key_from_password(password)
    logger.info(f"Schlüssel abgeleitet: {secure_encode(key)[:16]}...")
    
    # Test: Schlüsselhierarchie
    keys = create_key_hierarchy(password, "test_user_123")
    logger.info(f"Schlüsselhierarchie erstellt mit {len(keys)} Schlüsseln")
    
    # Test: HMAC
    test_data = b"VXOR AI Test Data"
    hmac_value = compute_hmac(test_data, keys["auth_key"])
    logger.info(f"HMAC berechnet: {secure_encode(hmac_value)[:16]}...")
    
    # Test: Padding
    padded = pad_tensor_data(test_data, "tensor_specific")
    unpadded = unpad_tensor_data(padded, "tensor_specific")
    assert test_data == unpadded
    logger.info(f"Padding/Unpadding erfolgreich: Daten wiederhergestellt")
    
    # Test: Zufallszahlen
    random_bytes = generate_random_bytes(32)
    logger.info(f"Zufallsbytes generiert: {secure_encode(random_bytes)[:16]}...")
    
    logger.info("Alle Tests erfolgreich abgeschlossen!")


# Wenn direkt ausgeführt, führe Tests durch
if __name__ == "__main__":
    test_crypto_utils()
