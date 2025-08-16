#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AES-256-GCM Implementierung für VXOR AI Blackbox.

Diese Komponente bietet eine produktionsreife Implementierung von AES-256-GCM
für symmetrische Verschlüsselung mit Authentifizierung und Integritätsschutz.
Sie bildet die Grundlage für alle symmetrischen Verschlüsselungsoperationen
im VXOR AI Blackbox-System.
"""

import os
import hmac
import hashlib
import logging
from typing import Tuple, Optional, Union, Dict, Any, List, ByteString

# Für die produktive Implementierung wird die Python-Cryptography-Bibliothek verwendet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

# Import für den SecureLogger
try:
    from ..logging import get_logger
except ImportError:
    # Fallback, falls die Logging-Komponente nicht verfügbar ist
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


class AESCipher:
    """
    AES-256-GCM Verschlüsselungskomponente für VXOR AI Blackbox.
    
    Bietet eine robuste Implementierung von AES-256-GCM mit:
    - Starker symmetrischer Verschlüsselung (AES-256)
    - Authentifizierte Verschlüsselung mit GCM-Modus
    - Sichere Schlüssel- und Nonce-Generierung
    - Integrierte Schlüsselableitung aus Passwörtern/Passphrasen
    - Effiziente Ver- und Entschlüsselung von Daten verschiedener Größen
    
    Diese Klasse bildet die kryptographische Grundlage für alle symmetrischen
    Verschlüsselungsoperationen im VXOR AI Blackbox-System und ist kompatibel
    mit der T-Mathematics Engine und ihrer Speicherschutzfunktionen.
    """
    
    # Standardwerte für die AES-GCM Implementierung
    DEFAULT_KEY_SIZE = 32    # 256 Bit
    DEFAULT_IV_SIZE = 12     # 96 Bit (empfohlen für GCM)
    DEFAULT_TAG_SIZE = 16    # 128 Bit Authentifizierungs-Tag
    
    # Konstanten für die Formatierung der verschlüsselten Daten
    VERSION = 1               # Versionsidentifikator für das Format
    HEADER_SIZE = 1 + 12 + 16  # Version (1 Byte) + IV (12 Bytes) + Tag (16 Bytes)
    
    def __init__(self, key_size: int = DEFAULT_KEY_SIZE, 
                iv_size: int = DEFAULT_IV_SIZE,
                tag_size: int = DEFAULT_TAG_SIZE):
        """
        Initialisiert den AESCipher.
        
        Args:
            key_size: Schlüsselgröße in Bytes (32 für AES-256)
            iv_size: IV/Nonce-Größe in Bytes (12 für GCM empfohlen)
            tag_size: Größe des Authentifizierungs-Tags in Bytes (16 empfohlen)
        """
        if key_size not in [16, 24, 32]:  # 128, 192 oder 256 Bit
            raise ValueError("Schlüsselgröße muss 16, 24 oder 32 Bytes betragen")
        
        if iv_size != 12:  # GCM empfiehlt 12 Bytes für optimale Leistung und Sicherheit
            raise ValueError("IV-Größe muss 12 Bytes für GCM betragen")
        
        if tag_size != 16:  # GCM verwendet ein 16-Byte-Tag
            raise ValueError("Tag-Größe muss 16 Bytes für GCM betragen")
        
        self.key_size = key_size
        self.iv_size = iv_size
        self.tag_size = tag_size
        
        # Initialisiere Logger
        self.logger = get_logger("crypto.aes")
    
    def generate_key(self) -> bytes:
        """
        Generiert einen kryptographisch sicheren AES-Schlüssel.
        
        Returns:
            Ein zufälliger Schlüssel der konfigurierten Größe
        """
        self.logger.debug(f"Generiere AES-Schlüssel ({self.key_size * 8} Bit)")
        return os.urandom(self.key_size)
    
    def generate_iv(self) -> bytes:
        """
        Generiert einen kryptographisch sicheren Initialisierungsvektor (IV/Nonce).
        
        Returns:
            Ein zufälliger IV der konfigurierten Größe
        """
        return os.urandom(self.iv_size)
    
    def encrypt(self, data: bytes, key: bytes, iv: Optional[bytes] = None,
               associated_data: Optional[bytes] = None) -> bytes:
        """
        Verschlüsselt Daten mit AES-256-GCM.
        
        Args:
            data: Die zu verschlüsselnden Daten
            key: Der AES-Schlüssel (muss der konfigurierten Größe entsprechen)
            iv: Optionaler IV/Nonce (wird generiert, wenn nicht angegeben)
            associated_data: Optionale zusätzliche authentifizierte Daten (AAD)
            
        Returns:
            Die verschlüsselten Daten im Format: [Version][IV][Tag][Ciphertext]
            
        Raises:
            ValueError: Wenn der Schlüssel die falsche Größe hat
        """
        if len(key) != self.key_size:
            raise ValueError(f"Schlüssel muss {self.key_size} Bytes lang sein")
        
        # Behandle leere Daten speziell
        if len(data) == 0:
            # Bei leeren Daten fügen wir einen speziellen Marker hinzu
            # und verschlüsseln diesen stattdessen
            data = b"\x00\x00EMPTY_DATA\x00\x00"
        
        # Generiere zufälligen IV
        iv = self.generate_iv()
        
        # Erstelle Cipher-Objekt
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        
        # Verschlüssele Daten
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Extrahiere authentification tag
        tag = encryptor.tag
        
        # Format: Versionsbyte + IV + Tag + Ciphertext
        version = bytes([self.VERSION])
        result = version + iv + tag + ciphertext
        
        self.logger.debug(f"Daten verschlüsselt: {len(data)} Bytes -> {len(result)} Bytes")
        return result
    
    def decrypt(self, encrypted_data: bytes, key: bytes) -> bytes:
        """
        Entschlüsselt mit AES-256-GCM verschlüsselte Daten.
        
        Args:
            encrypted_data: Die verschlüsselten Daten im Format [Version][IV][Tag][Ciphertext]
            key: Der AES-Schlüssel (muss der konfigurierten Größe entsprechen)
            
        Returns:
            Die entschlüsselten Daten
        """
        if len(key) != self.key_size:
            raise ValueError(f"Schlüssel muss {self.key_size} Bytes lang sein")
            
        if len(encrypted_data) < 29:  # Version (1) + IV (12) + Tag (16) + mindestens 1 Byte Daten
            raise ValueError("Verschlüsselte Daten zu kurz")
        
        try:
            # Zerlege verschlüsselte Daten
            version = encrypted_data[0]
            iv = encrypted_data[1:13]  # 12 Bytes
            tag = encrypted_data[13:29]  # 16 Bytes
            ciphertext = encrypted_data[29:]
            
            # Überprüfe Version
            if version != self.VERSION:
                raise ValueError(f"Nicht unterstützte Version: {version}")
            
            # Erstelle Cipher-Objekt
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag))
            decryptor = cipher.decryptor()
            
            # Entschlüssele Daten
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Prüfe, ob es sich um leere Daten handelt
            if plaintext == b"\x00\x00EMPTY_DATA\x00\x00":
                plaintext = b""
            
            self.logger.debug(f"Daten entschlüsselt: {len(encrypted_data)} Bytes -> {len(plaintext)} Bytes")
            return plaintext
        except Exception as e:
            self.logger.error(f"Entschlüsselungsfehler: {str(e)}")
            raise ValueError("Entschlüsselungsfehler. Die Daten könnten manipuliert worden sein oder der Schlüssel ist falsch.")
    
    def derive_key_from_password(self, password: Union[str, bytes], 
                               salt: Optional[bytes] = None,
                               iterations: int = 600000) -> Tuple[bytes, bytes]:
        """
        Leitet einen kryptographischen Schlüssel aus einem Passwort ab.
        
        Args:
            password: Das Passwort als String oder Bytes
            salt: Optional, ein Salt für die Schlüsselableitung (wird generiert, wenn nicht angegeben)
            iterations: Anzahl der Iterationen für PBKDF2 (höher = sicherer, aber langsamer)
            
        Returns:
            Tuple aus (abgeleiteter Schlüssel, Salt)
        """
        if salt is None:
            salt = os.urandom(16)  # 128 Bit Salt
        
        if isinstance(password, str):
            password = password.encode('utf-8')
        
        self.logger.debug(f"Leite Schlüssel aus Passwort ab (Iterationen: {iterations})")
        
        # Verwende PBKDF2 zur sicheren Schlüsselableitung
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.key_size,
            salt=salt,
            iterations=iterations,
        )
        
        key = kdf.derive(password)
        
        return key, salt
    
    def wrap_key(self, key_to_wrap: bytes, kek: bytes) -> bytes:
        """Umhüllt einen Schlüssel mit einem Key Encryption Key (KEK).
        
        Dies ermöglicht die sichere Speicherung von Schlüsseln, wobei der KEK
        zum Schutz des eigentlichen Datenschlüssels (DEK) verwendet wird.
        
        Args:
            key_to_wrap: Der zu umhüllende Schlüssel (typischerweise ein DEK)
            kek: Der Key Encryption Key
            
        Returns:
            Der umhüllte Schlüssel
        """
        # Einfach AES-GCM zur Verschlüsselung des Schlüssels verwenden
        return self.encrypt(key_to_wrap, kek)
    
    def unwrap_key(self, wrapped_key: bytes, kek: bytes) -> bytes:
        """Enthüllt einen mit einem KEK umhüllten Schlüssel.
        
        Args:
            wrapped_key: Der umhüllte Schlüssel
            kek: Der Key Encryption Key, der zur Umhüllung verwendet wurde
            
        Returns:
            Der enthüllte Schlüssel
        """
        # AES-GCM zur Entschlüsselung des Schlüssels verwenden
        return self.decrypt(wrapped_key, kek)
    
    def encrypt_with_password(self, data: bytes, password: Union[str, bytes], 
                             salt: Optional[bytes] = None, iterations: int = 600000) -> bytes:
        """Verschlüsselt Daten mit einem Passwort anstelle eines Schlüssels.
        
        Diese Methode leitet einen Schlüssel aus dem Passwort ab und verwendet
        diesen für die Verschlüsselung. Das Salt wird in den verschlüsselten Daten
        gespeichert, damit es für die Entschlüsselung verfügbar ist.
        
        Args:
            data: Zu verschlüsselnde Daten
            password: Passwort zur Schlüsselableitung
            salt: Optional, Salt für die Schlüsselableitung (wird generiert, falls None)
            iterations: Anzahl der Iterationen für PBKDF2
            
        Returns:
            Verschlüsselte Daten mit Format: Versionsbyte + Salt + IV + Tag + Ciphertext
        """
        # Konvertiere String-Passwort zu Bytes, falls nötig
        if isinstance(password, str):
            password = password.encode('utf-8')
        
        # Generiere Salt, falls keines angegeben
        if salt is None:
            salt = os.urandom(16)  # 128 Bit Salt
        
        # Leite Schlüssel aus Passwort ab
        key = self.derive_key_from_password(password, salt, iterations)[0]
        
        # Verschlüssele Daten mit abgeleitetem Schlüssel
        encrypted = self.encrypt(data, key)
        
        # Füge Versionsbyte, Salt und Iterationszahl hinzu
        version = bytes([self.VERSION])
        iterations_bytes = iterations.to_bytes(4, byteorder='big')
        
        # Format: Versionsbyte + Salt + Iterationszahl + verschlüsselte Daten
        result = version + salt + iterations_bytes + encrypted[1:]  # Überspringe das Versionsbyte von encrypted
        
        self.logger.debug(f"Daten mit Passwort verschlüsselt: {len(data)} Bytes -> {len(result)} Bytes")
        return result
    
    def decrypt_with_password(self, encrypted_data: bytes, password: Union[str, bytes]) -> bytes:
        """
        Entschlüsselt Daten, die mit einem Passwort verschlüsselt wurden.
        
        Args:
            encrypted_data: Die verschlüsselten Daten im Format [Version][Salt][Iterationszahl][IV][Tag][Ciphertext]
            password: Das Passwort als String oder Bytes
            
        Returns:
            Die entschlüsselten Daten
        """
        if len(encrypted_data) < 21:  # Version(1) + Salt(16) + Iterationszahl(4)
            raise ValueError("Verschlüsselte Daten zu kurz")
        
        # Konvertiere String-Passwort zu Bytes, falls nötig
        if isinstance(password, str):
            password = password.encode('utf-8')
            
        # Extrahiere Version, Salt und Iterationszahl
        version = encrypted_data[0]
        if version != self.VERSION:
            raise ValueError(f"Nicht unterstützte Version: {version}")
            
        # Format: Versionsbyte + Salt + Iterationszahl + verschlüsselte Daten
        salt = encrypted_data[1:17]  # 16 Bytes Salt
        iterations = int.from_bytes(encrypted_data[17:21], byteorder='big')
        
        # Leite Schlüssel aus Passwort und Salt ab
        key, _ = self.derive_key_from_password(password, salt, iterations)
        
        # Rekonstruiere ein standard-formatiertes verschlüsseltes Datenpaket mit dem Versionsbyte
        # damit es mit der normalen decrypt-Methode entschlüsselt werden kann
        # (encrypted_data enthält bereits die Version, aber die verschlüsselten Daten
        # ab Position 21 brauchen noch ein eigenes Versionsbyte)
        version_byte = bytes([self.VERSION])
        actual_encrypted = version_byte + encrypted_data[21:]
        
        # Entschlüssele mit dem abgeleiteten Schlüssel
        self.logger.debug(f"Daten mit Passwort entschlüsselt (Iterationen: {iterations})")
        return self.decrypt(actual_encrypted, key)
    
    def encrypt_file(self, input_file: str, output_file: str, key: bytes,
                    chunk_size: int = 64 * 1024,
                    associated_data: Optional[bytes] = None) -> None:
        """
        Verschlüsselt eine Datei mit AES-256-GCM.
        
        Args:
            input_file: Pfad zur Eingabedatei
            output_file: Pfad zur Ausgabedatei
            key: Der AES-Schlüssel
            chunk_size: Größe der zu verarbeitenden Chunks in Bytes
            associated_data: Optionale zusätzliche authentifizierte Daten
            
        Raises:
            IOError: Bei Problemen mit dem Dateizugriff
            ValueError: Bei falscher Schlüsselgröße
        """
        if len(key) != self.key_size:
            raise ValueError(f"Schlüssel muss {self.key_size} Bytes lang sein")
        
        # Generiere IV
        iv = self.generate_iv()
        
        # Erstelle GCM-Cipher und Encryptor
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        
        # Füge zusätzliche authentifizierte Daten hinzu, falls vorhanden
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        try:
            with open(input_file, 'rb') as in_file, open(output_file, 'wb') as out_file:
                # Schreibe Versionsbyte und IV
                version_byte = bytes([self.VERSION])
                out_file.write(version_byte + iv)
                
                # Verschlüssele die Datei in Chunks
                while True:
                    chunk = in_file.read(chunk_size)
                    if not chunk:
                        break
                    
                    encrypted_chunk = encryptor.update(chunk)
                    out_file.write(encrypted_chunk)
                
                # Finalisiere die Verschlüsselung
                final_chunk = encryptor.finalize()
                out_file.write(final_chunk)
                
                # Schreibe den Auth-Tag
                out_file.write(encryptor.tag)
                
                self.logger.info(f"Datei {input_file} erfolgreich verschlüsselt nach {output_file}")
        
        except IOError as e:
            self.logger.error(f"Fehler bei der Dateiverschlüsselung: {str(e)}")
            raise
    
    def decrypt_file(self, input_file: str, output_file: str, key: bytes,
                    chunk_size: int = 64 * 1024,
                    associated_data: Optional[bytes] = None) -> None:
        """
        Entschlüsselt eine mit AES-256-GCM verschlüsselte Datei.
        
        Args:
            input_file: Pfad zur verschlüsselten Eingabedatei
            output_file: Pfad zur entschlüsselten Ausgabedatei
            key: Der AES-Schlüssel
            chunk_size: Größe der zu verarbeitenden Chunks in Bytes
            associated_data: Optionale zusätzliche authentifizierte Daten
            
        Raises:
            IOError: Bei Problemen mit dem Dateizugriff
            ValueError: Bei falscher Schlüsselgröße oder Authentifizierungsfehler
        """
        if len(key) != self.key_size:
            raise ValueError(f"Schlüssel muss {self.key_size} Bytes lang sein")
        
        try:
            with open(input_file, 'rb') as in_file:
                # Lese Versionsbyte und IV
                version = in_file.read(1)[0]
                if version != self.VERSION:
                    raise ValueError(f"Nicht unterstützte Versionskennung: {version}")
                
                iv = in_file.read(self.iv_size)
                
                # Lese die verschlüsselten Daten
                ciphertext = in_file.read()
                
                # Extrahiere den Auth-Tag (die letzten 16 Bytes)
                if len(ciphertext) < self.tag_size:
                    raise ValueError("Datei zu kurz, fehlendes Auth-Tag")
                
                tag = ciphertext[-self.tag_size:]
                actual_ciphertext = ciphertext[:-self.tag_size]
            
            # Erstelle GCM-Cipher und Decryptor
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag))
            decryptor = cipher.decryptor()
            
            # Füge zusätzliche authentifizierte Daten hinzu, falls vorhanden
            if associated_data:
                decryptor.authenticate_additional_data(associated_data)
            
            # Entschlüssele die Daten und schreibe sie in die Ausgabedatei
            with open(output_file, 'wb') as out_file:
                try:
                    plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()
                    out_file.write(plaintext)
                    
                    self.logger.info(f"Datei {input_file} erfolgreich entschlüsselt nach {output_file}")
                
                except Exception as e:
                    self.logger.error(f"Authentifizierungsfehler bei der Dateientschlüsselung: {str(e)}")
                    # Lösche die unvollständige oder manipulierte Ausgabedatei
                    out_file.close()
                    os.unlink(output_file)
                    raise ValueError("Authentifizierung fehlgeschlagen. Die Datei könnte manipuliert worden sein.")
        
        except IOError as e:
            self.logger.error(f"Fehler bei der Dateientschlüsselung: {str(e)}")
            raise
    
    def encrypt_with_kek(self, data: bytes, data_key: bytes, key_encryption_key: bytes) -> bytes:
        """
        Verschlüsselt Daten mit einem Datenschlüssel und verschlüsselt diesen mit einem KEK.
        
        Implementiert das Konzept von 'Key Encryption Keys' (KEK), bei dem ein 
        Datenschlüssel (DEK) die eigentlichen Daten verschlüsselt und dann selbst
        mit einem übergeordneten Schlüssel (KEK) verschlüsselt wird.
        
        Args:
            data: Die zu verschlüsselnden Daten
            data_key: Der Datenschlüssel (DEK)
            key_encryption_key: Der Schlüsselverschlüsselungsschlüssel (KEK)
            
        Returns:
            Die verschlüsselten Daten im Format: [Verschlüsselter DEK][Verschlüsselte Daten]
        """
        if len(data_key) != self.key_size:
            raise ValueError(f"Datenschlüssel muss {self.key_size} Bytes lang sein")
        
        if len(key_encryption_key) != self.key_size:
            raise ValueError(f"Schlüsselverschlüsselungsschlüssel muss {self.key_size} Bytes lang sein")
        
        # Verschlüssele den Datenschlüssel mit dem KEK
        encrypted_key = self.encrypt(data_key, key_encryption_key)
        
        # Verschlüssele die Daten mit dem Datenschlüssel
        encrypted_data = self.encrypt(data, data_key)
        
        # Kombiniere zu einem Paket
        key_size = len(encrypted_key).to_bytes(2, byteorder='big')
        result = key_size + encrypted_key + encrypted_data
        
        return result
    
    def decrypt_with_kek(self, encrypted_data: bytes, key_encryption_key: bytes) -> bytes:
        """
        Entschlüsselt Daten, die mit einem KEK-geschützten Datenschlüssel verschlüsselt wurden.
        
        Args:
            encrypted_data: Die verschlüsselten Daten im Format [Verschlüsselter DEK][Verschlüsselte Daten]
            key_encryption_key: Der Schlüsselverschlüsselungsschlüssel (KEK)
            
        Returns:
            Die entschlüsselten Daten
        """
        if len(encrypted_data) < 2:
            raise ValueError("Verschlüsselte Daten zu kurz")
        
        # Extrahiere die Größe des verschlüsselten Schlüssels
        key_size = int.from_bytes(encrypted_data[:2], byteorder='big')
        
        if len(encrypted_data) < 2 + key_size:
            raise ValueError("Verschlüsselte Daten zu kurz (ungültiger Schlüssel)")
        
        # Extrahiere den verschlüsselten Schlüssel und die verschlüsselten Daten
        encrypted_key = encrypted_data[2:2+key_size]
        actual_encrypted_data = encrypted_data[2+key_size:]
        
        # Entschlüssele den Datenschlüssel
        data_key = self.decrypt(encrypted_key, key_encryption_key)
        
        # Entschlüssele die Daten mit dem Datenschlüssel
        return self.decrypt(actual_encrypted_data, data_key)
    
    def generate_hmac(self, data: bytes, key: bytes, algorithm: str = 'sha256') -> bytes:
        """
        Generiert einen HMAC für Daten.
        
        Diese Methode kann verwendet werden, um Datenintegrität zusätzlich
        zur GCM-Authentifizierung zu gewährleisten oder für separate 
        Integritätsprüfungen ohne Verschlüsselung.
        
        Args:
            data: Die Daten, für die ein HMAC generiert werden soll
            key: Der HMAC-Schlüssel
            algorithm: Der zu verwendende Hash-Algorithmus ('sha256', 'sha384', 'sha512')
            
        Returns:
            Der generierte HMAC
        """
        if algorithm == 'sha256':
            hash_algorithm = hashlib.sha256
        elif algorithm == 'sha384':
            hash_algorithm = hashlib.sha384
        elif algorithm == 'sha512':
            hash_algorithm = hashlib.sha512
        else:
            raise ValueError(f"Nicht unterstützter Hash-Algorithmus: {algorithm}")
        
        h = hmac.new(key, data, hash_algorithm)
        return h.digest()
    
    def verify_hmac(self, data: bytes, key: bytes, mac: bytes, algorithm: str = 'sha256') -> bool:
        """
        Verifiziert einen HMAC für Daten.
        
        Args:
            data: Die zu verifizierenden Daten
            key: Der HMAC-Schlüssel
            mac: Der zu verifizierende HMAC
            algorithm: Der verwendete Hash-Algorithmus
            
        Returns:
            True, wenn der HMAC gültig ist, sonst False
        """
        computed_mac = self.generate_hmac(data, key, algorithm)
        
        # Verwende einen zeitunabhängigen Vergleich, um Timing-Angriffe zu verhindern
        return hmac.compare_digest(computed_mac, mac)


# Singleton-Instanz für einfachen globalen Zugriff
_aes_cipher = None

def get_aes_cipher() -> AESCipher:
    """
    Gibt die globale AESCipher-Instanz zurück oder erstellt sie, falls noch nicht vorhanden.
    
    Returns:
        AESCipher-Instanz
    """
    global _aes_cipher
    
    if _aes_cipher is None:
        _aes_cipher = AESCipher()
    
    return _aes_cipher
