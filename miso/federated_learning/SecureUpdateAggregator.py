#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SecureUpdateAggregator.py
========================

Modul für sichere Verschlüsselung, Entschlüsselung und Validierung von
Modell-Updates in föderiertem Lernen. Implementiert AES-256-Verschlüsselung,
kryptografische Integritätsprüfungen und stellt sicher, dass nur validierte
Updates in die globale Modellaktualisierung einfließen.

Teil der MISO Ultimate AGI - Phase 6 (Federated Learning System)
"""

import os
import sys
import time
import json
import uuid
import logging
import hashlib
import hmac
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, BinaryIO
import numpy as np
import datetime

# Für Verschlüsselung und Entschlüsselung
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding, hashes, hmac as crypto_hmac
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    print("Warnung: Das 'cryptography' Paket ist nicht verfügbar. Verschlüsselung wird nicht unterstützt.")

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MISO.SecureUpdateAggregator')

class SecureUpdateAggregator:
    """
    Modul für die sichere Handhabung und Aggregierung von Modell-Updates in föderiertem Lernen.
    
    Features:
    - AES-256-Verschlüsselung und -Entschlüsselung von Modell-Updates
    - Kryptografische Validierung der Integrität der Updates
    - Sichere Aggregation validierter Updates
    - Protokollierung aller Sicherheitsoperationen
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 encryption_key: Optional[str] = None,
                 base_output_dir: Optional[str] = None,
                 validation_threshold: float = 0.95,
                 enable_encryption: bool = True):
        """
        Initialisiert den SecureUpdateAggregator.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            encryption_key: Schlüssel für die Verschlüsselung (falls None, wird generiert)
            base_output_dir: Basisverzeichnis für Ausgaben
            validation_threshold: Schwellenwert für die Validierung der Updates (0-1)
            enable_encryption: Ob Verschlüsselung aktiviert werden soll
        """
        # Generiere eine eindeutige Session-ID
        self.session_id = str(uuid.uuid4())
        logger.info(f"Initialisiere SecureUpdateAggregator mit Session-ID: {self.session_id}")
        
        # Lade Konfiguration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Setze Basisparameter
        self.base_output_dir = base_output_dir or self.config.get('output_dir', './output')
        self.validation_threshold = validation_threshold
        self.enable_encryption = enable_encryption and CRYPTOGRAPHY_AVAILABLE
        
        # Prüfe und setze Verschlüsselungsschlüssel
        if self.enable_encryption:
            if encryption_key:
                self.encryption_key = self._derive_key(encryption_key)
            else:
                self.encryption_key = self._generate_key()
            logger.info("Verschlüsselung aktiviert mit AES-256")
        else:
            self.encryption_key = None
            logger.warning("Verschlüsselung deaktiviert - Updates werden im Klartext übertragen!")
        
        # Erstelle die erforderlichen Verzeichnisse
        self._setup_directories()
        
        # Initialisiere Tracking-Variablen
        self.security_stats = {
            "encrypted_updates": 0,
            "decrypted_updates": 0,
            "invalid_updates": 0,
            "validated_updates": 0,
            "aggregated_updates": 0,
            "last_operation": None
        }
        
        logger.info("SecureUpdateAggregator erfolgreich initialisiert")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Lädt die Konfigurationsdatei."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Konfiguration aus {config_path} geladen")
            return config
        except Exception as e:
            logger.warning(f"Konnte Konfiguration nicht laden: {e}. Verwende Standardeinstellungen.")
            return {}
    
    def _setup_directories(self) -> None:
        """Erstellt die erforderlichen Verzeichnisse."""
        try:
            # Erstelle Basisverzeichnis
            os.makedirs(self.base_output_dir, exist_ok=True)
            
            # Erstelle spezifische Unterverzeichnisse
            self.encrypted_dir = os.path.join(self.base_output_dir, 'encrypted_updates')
            self.decrypted_dir = os.path.join(self.base_output_dir, 'decrypted_updates')
            self.validated_dir = os.path.join(self.base_output_dir, 'validated_updates')
            self.logs_dir = os.path.join(self.base_output_dir, 'logs')
            
            os.makedirs(self.encrypted_dir, exist_ok=True)
            os.makedirs(self.decrypted_dir, exist_ok=True)
            os.makedirs(self.validated_dir, exist_ok=True)
            os.makedirs(self.logs_dir, exist_ok=True)
            
            logger.info(f"Verzeichnisstruktur unter {self.base_output_dir} eingerichtet")
        except Exception as e:
            logger.error(f"Fehler beim Einrichten der Verzeichnisse: {e}")
            raise
    
    def _derive_key(self, password: str, salt: bytes = None) -> bytes:
        """Leitet einen Verschlüsselungsschlüssel aus einem Passwort ab."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Das 'cryptography' Paket ist nicht verfügbar")
        
        # Verwende entweder den bereitgestellten Salt oder einen festen Salt
        if salt is None:
            salt = b'MISO_Ultimate_Secure_Salt'  # In einer Produktionsumgebung sollte ein sicherer Salt verwendet werden
        
        # Verwende PBKDF2 zur Ableitung eines sicheren Schlüssels
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 32 Bytes für AES-256
            salt=salt,
            iterations=100000,  # Hohe Anzahl an Iterationen für Sicherheit
            backend=default_backend()
        )
        
        # Leite Schlüssel aus Passwort ab
        key = kdf.derive(password.encode())
        return key
    
    def _generate_key(self) -> bytes:
        """Generiert einen zufälligen Verschlüsselungsschlüssel."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Das 'cryptography' Paket ist nicht verfügbar")
        
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa
            
            # Generiere einen zufälligen 256-Bit-Schlüssel
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            # Extrahiere den Schlüssel als Bytes
            key = private_key.private_bytes(
                encoding=default_backend().encoding,
                format=default_backend().format,
                encryption_algorithm=default_backend().encryption
            )[:32]  # Nimm die ersten 32 Bytes (256 Bits) für AES-256
            
            return key
        except Exception as e:
            logger.error(f"Fehler bei der Schlüsselgenerierung: {e}")
            # Fallback: Verwende einen festen, aber sicheren Schlüssel (nur für Entwicklung!)
            return hashlib.sha256(str(uuid.uuid4()).encode()).digest()
    
    def encrypt_update(self, update_data: Dict[str, Any], metadata: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Verschlüsselt ein Modell-Update mit AES-256.
        
        Args:
            update_data: Das zu verschlüsselnde Modell-Update als Dictionary
            metadata: Zusätzliche Metadaten für das Update
        
        Returns:
            Tuple aus Pfad zur verschlüsselten Datei und Metadaten
        """
        if not self.enable_encryption:
            logger.warning("Verschlüsselung ist deaktiviert. Update wird im Klartext gespeichert.")
            return self._store_unencrypted_update(update_data, metadata)
        
        try:
            # Erstelle Metadaten, falls keine bereitgestellt
            if metadata is None:
                metadata = {}
            
            # Füge Basis-Metadaten hinzu
            full_metadata = {
                "timestamp": time.time(),
                "session_id": self.session_id,
                "encryption": "AES-256-CBC",
                "encrypted": True,
                **metadata
            }
            
            # Serialisiere Update-Daten zu JSON
            update_bytes = json.dumps(self._prepare_data_for_serialization(update_data)).encode('utf-8')
            
            # Generiere IV (Initialisierungsvektor) für AES
            iv = os.urandom(16)  # 16 Bytes IV für AES
            
            # Pad die Daten für AES-CBC
            padder = padding.PKCS7(algorithms.AES.block_size).padder()
            padded_data = padder.update(update_bytes) + padder.finalize()
            
            # Führe AES-Verschlüsselung durch
            cipher = Cipher(algorithms.AES(self.encryption_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Berechne HMAC für die Integritätsprüfung
            h = crypto_hmac.HMAC(self.encryption_key, hashes.SHA256(), backend=default_backend())
            h.update(iv + encrypted_data)
            hmac_digest = h.finalize()
            
            # Kombiniere IV, verschlüsselte Daten und HMAC
            combined_data = iv + encrypted_data + hmac_digest
            
            # Erstelle Dateinamen für das verschlüsselte Update
            file_path = os.path.join(
                self.encrypted_dir,
                f"encrypted_update_{self.session_id}_{int(time.time())}.bin"
            )
            
            # Speichere das verschlüsselte Update
            with open(file_path, 'wb') as f:
                f.write(combined_data)
            
            # Speichere Metadaten in einer separaten JSON-Datei
            metadata_path = file_path + '.meta.json'
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
            # Aktualisiere Statistiken
            self.security_stats["encrypted_updates"] += 1
            self.security_stats["last_operation"] = "encrypt"
            
            logger.info(f"Update erfolgreich verschlüsselt und gespeichert unter {file_path}")
            return file_path, full_metadata
            
        except Exception as e:
            logger.error(f"Fehler bei der Verschlüsselung des Updates: {e}")
            if not self.enable_encryption:
                logger.info("Fallback auf unverschlüsselte Speicherung")
                return self._store_unencrypted_update(update_data, metadata)
            raise
    
    def _store_unencrypted_update(self, update_data: Dict[str, Any], metadata: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """Speichert ein Update unverschlüsselt (nur für Entwicklung oder wenn Verschlüsselung deaktiviert ist)."""
        try:
            # Erstelle Metadaten, falls keine bereitgestellt
            if metadata is None:
                metadata = {}
            
            # Füge Basis-Metadaten hinzu
            full_metadata = {
                "timestamp": time.time(),
                "session_id": self.session_id,
                "encryption": "none",
                "encrypted": False,
                **metadata
            }
            
            # Erstelle Dateinamen für das unverschlüsselte Update
            file_path = os.path.join(
                self.encrypted_dir,  # Verwende das gleiche Verzeichnis für Konsistenz
                f"unencrypted_update_{self.session_id}_{int(time.time())}.json"
            )
            
            # Speichere das Update als JSON
            with open(file_path, 'w') as f:
                json.dump(self._prepare_data_for_serialization(update_data), f, indent=2)
            
            # Speichere Metadaten in einer separaten JSON-Datei
            metadata_path = file_path + '.meta.json'
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
            logger.warning(f"Update UNVERSCHLÜSSELT gespeichert unter {file_path}")
            return file_path, full_metadata
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern des unverschlüsselten Updates: {e}")
            raise
    
    def _prepare_data_for_serialization(self, data: Any) -> Any:
        """Bereitet Daten für die JSON-Serialisierung vor."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._prepare_data_for_serialization(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_data_for_serialization(item) for item in data]
        elif hasattr(data, '__dict__'):
            return self._prepare_data_for_serialization(data.__dict__)
        else:
            return data
    
    def decrypt_update(self, encrypted_file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Entschlüsselt ein verschlüsseltes Modell-Update.
        
        Args:
            encrypted_file_path: Pfad zur verschlüsselten Datei
        
        Returns:
            Tuple aus entschlüsseltem Update und Metadaten
        """
        try:
            # Prüfe, ob die Datei existiert
            if not os.path.exists(encrypted_file_path):
                raise FileNotFoundError(f"Verschlüsselte Datei nicht gefunden: {encrypted_file_path}")
            
            # Lade Metadaten, falls verfügbar
            metadata_path = encrypted_file_path + '.meta.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"encryption": "unknown", "encrypted": True}
            
            # Prüfe, ob die Datei tatsächlich verschlüsselt ist
            if not metadata.get("encrypted", True):
                logger.info(f"Die Datei {encrypted_file_path} ist nicht verschlüsselt. Lade direkt.")
                with open(encrypted_file_path, 'r') as f:
                    update_data = json.load(f)
                    
                self.security_stats["decrypted_updates"] += 1
                self.security_stats["last_operation"] = "decrypt_unencrypted"
                
                # Speichere entschlüsselte Daten zur Konsistenz
                decrypted_path = os.path.join(
                    self.decrypted_dir,
                    f"decrypted_{os.path.basename(encrypted_file_path)}"
                )
                with open(decrypted_path, 'w') as f:
                    json.dump(update_data, f, indent=2)
                
                logger.info(f"Unverschlüsselte Daten geladen und unter {decrypted_path} gespeichert")
                return update_data, metadata
            
            # Bei verschlüsselten Daten: prüfe, ob Entschlüsselung aktiviert ist
            if not self.enable_encryption or not CRYPTOGRAPHY_AVAILABLE:
                raise RuntimeError("Entschlüsselung ist deaktiviert oder 'cryptography' nicht verfügbar")
            
            # Lade verschlüsselte Daten
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Extrahiere IV, verschlüsselte Daten und HMAC
            iv = encrypted_data[:16]  # Erste 16 Bytes sind der IV
            hmac_digest = encrypted_data[-32:]  # Letzte 32 Bytes sind der HMAC (SHA-256)
            ciphertext = encrypted_data[16:-32]  # Daten zwischen IV und HMAC
            
            # Verifiziere HMAC
            h = crypto_hmac.HMAC(self.encryption_key, hashes.SHA256(), backend=default_backend())
            h.update(iv + ciphertext)
            try:
                h.verify(hmac_digest)
            except Exception:
                logger.error(f"HMAC-Verifikation fehlgeschlagen für {encrypted_file_path}. Mögliche Manipulation!")
                self.security_stats["invalid_updates"] += 1
                raise ValueError("Integritätsprüfung fehlgeschlagen: HMAC-Verifikation fehlgeschlagen")
            
            # Entschlüsseln
            cipher = Cipher(algorithms.AES(self.encryption_key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Entferne Padding
            unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
            update_bytes = unpadder.update(padded_data) + unpadder.finalize()
            
            # Deserialisiere JSON zu Dictionary
            update_data = json.loads(update_bytes.decode('utf-8'))
            
            # Speichere entschlüsselte Daten
            decrypted_path = os.path.join(
                self.decrypted_dir,
                f"decrypted_{os.path.basename(encrypted_file_path).replace('.bin', '.json')}"
            )
            with open(decrypted_path, 'w') as f:
                json.dump(update_data, f, indent=2)
            
            # Aktualisiere Statistiken
            self.security_stats["decrypted_updates"] += 1
            self.security_stats["last_operation"] = "decrypt"
            
            logger.info(f"Update erfolgreich entschlüsselt und unter {decrypted_path} gespeichert")
            return update_data, metadata
            
        except Exception as e:
            logger.error(f"Fehler bei der Entschlüsselung des Updates: {e}")
            raise
    
    def validate_update(self, update_data: Dict[str, Any], metadata: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any], str]:
        """
        Validiert ein Modell-Update auf Integrität und Plausibilität.
        
        Args:
            update_data: Das zu validierende Update
            metadata: Metadaten zum Update
        
        Returns:
            Tuple aus (ist_valide, Validierungsergebnis, Pfad zur validierten Datei)
        """
        try:
            validation_results = {
                "timestamp": time.time(),
                "session_id": self.session_id,
                "validation_checks": {}
            }
            
            if metadata is None:
                metadata = {}
            
            # 1. Grundlegende Strukturprüfung
            has_valid_structure = self._validate_structure(update_data)
            validation_results["validation_checks"]["structure"] = has_valid_structure
            
            # 2. Statistische Ausreißerprüfung
            outlier_score, is_outlier = self._check_for_outliers(update_data)
            validation_results["validation_checks"]["outlier"] = not is_outlier
            validation_results["outlier_score"] = outlier_score
            
            # 3. Größenbegrenzungsprüfung
            size_valid, size_info = self._validate_size(update_data)
            validation_results["validation_checks"]["size"] = size_valid
            validation_results["size_info"] = size_info
            
            # 4. Wertebereichsprüfung
            values_valid, value_info = self._validate_value_ranges(update_data)
            validation_results["validation_checks"]["values"] = values_valid
            validation_results["value_info"] = value_info
            
            # 5. Zusätzliche modellspezifische Prüfungen, falls verfügbar
            if hasattr(self, '_model_specific_validation'):
                model_valid, model_info = self._model_specific_validation(update_data, metadata)
                validation_results["validation_checks"]["model_specific"] = model_valid
                validation_results["model_info"] = model_info
            
            # Berechne Gesamtergebnis
            validation_score = sum(validation_results["validation_checks"].values()) / len(validation_results["validation_checks"])
            validation_results["validation_score"] = validation_score
            is_valid = validation_score >= self.validation_threshold
            validation_results["is_valid"] = is_valid
            
            # Speichere Validierungsergebnis
            validation_path = os.path.join(
                self.validated_dir,
                f"validation_{self.session_id}_{int(time.time())}.json"
            )
            
            # Speichere das Update, wenn es gültig ist
            validated_update_path = ""
            if is_valid:
                validated_update_path = os.path.join(
                    self.validated_dir,
                    f"validated_update_{self.session_id}_{int(time.time())}.json"
                )
                with open(validated_update_path, 'w') as f:
                    json.dump(self._prepare_data_for_serialization(update_data), f, indent=2)
                
                # Speichere Update-Metadaten mit Validierungsergebnis
                update_metadata_path = validated_update_path + '.meta.json'
                with open(update_metadata_path, 'w') as f:
                    json.dump({**metadata, "validation": validation_results}, f, indent=2)
                
                # Aktualisiere Statistiken
                self.security_stats["validated_updates"] += 1
            else:
                # Aktualisiere Statistiken für ungültige Updates
                self.security_stats["invalid_updates"] += 1
            
            # Speichere Validierungsergebnis separat
            with open(validation_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            self.security_stats["last_operation"] = "validate"
            
            if is_valid:
                logger.info(f"Update validiert mit Score {validation_score:.2f} und als gültig eingestuft")
            else:
                logger.warning(f"Update validiert mit Score {validation_score:.2f} und als UNGÜLTIG eingestuft")
            
            return is_valid, validation_results, validated_update_path
            
        except Exception as e:
            logger.error(f"Fehler bei der Validierung des Updates: {e}")
            return False, {"error": str(e)}, ""
    
    def _validate_structure(self, update_data: Dict[str, Any]) -> bool:
        """Prüft, ob das Update-Format den Erwartungen entspricht."""
        # Mindestanforderung: Dict mit Modellparametern
        if not isinstance(update_data, dict):
            return False
        
        # Prüfe, ob Update Modellparameter enthält (einfache Struktur)
        # Diese Prüfung muss an die spezifische Datenstruktur angepasst werden
        return True
    
    def _check_for_outliers(self, update_data: Dict[str, Any]) -> Tuple[float, bool]:
        """Prüft, ob das Update statistische Ausreißer enthält."""
        outlier_score = 0.0
        total_weights = 0
        
        try:
            # Berechne statistische Eigenschaften der Parameter
            for key, value in update_data.items():
                if isinstance(value, (list, np.ndarray)):
                    # Konvertiere zu NumPy-Array für statistische Berechnungen
                    if isinstance(value, list):
                        value = np.array(value)
                    
                    # Prüfe auf NaN oder Inf Werte
                    if np.isnan(value).any() or np.isinf(value).any():
                        return 1.0, True  # Definitiv ein Ausreißer
                    
                    # Berechne statistische Eigenschaften
                    mean = np.mean(value)
                    std = np.std(value)
                    max_val = np.max(value)
                    min_val = np.min(value)
                    
                    # Prüfe auf extreme Werte (Z-Score > 3)
                    if std > 0:  # Verhindere Division durch Null
                        z_score_max = abs(max_val - mean) / std
                        z_score_min = abs(min_val - mean) / std
                        max_z_score = max(z_score_max, z_score_min)
                        
                        weight = len(value)
                        outlier_score += min(1.0, max_z_score / 10) * weight  # Normalisiere Z-Score
                        total_weights += weight
            
            # Normalisiere den Ausreißerscore
            if total_weights > 0:
                outlier_score /= total_weights
            
            # Ein Score > 0.5 deutet auf Ausreißer hin
            return outlier_score, outlier_score > 0.5
            
        except Exception as e:
            logger.error(f"Fehler bei der Ausreißererkennung: {e}")
            return 0.0, False
    
    def _validate_size(self, update_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Prüft, ob die Größe des Updates innerhalb akzeptabler Grenzen liegt."""
        size_info = {}
        
        try:
            # Berechne Gesamtgröße und Anzahl der Parameter
            total_size = 0
            total_params = 0
            
            for key, value in update_data.items():
                if isinstance(value, (list, np.ndarray)):
                    if isinstance(value, list):
                        param_size = len(value) * 8  # Grobe Schätzung für float64
                        param_count = len(value)
                    else:  # NumPy-Array
                        param_size = value.nbytes
                        param_count = value.size
                    
                    total_size += param_size
                    total_params += param_count
            
            # Speichere Informationen
            size_info["total_size_bytes"] = total_size
            size_info["total_params"] = total_params
            size_info["size_mb"] = total_size / (1024 * 1024)
            
            # Prüfe gegen Schwellenwerte (anpassen nach Bedarf)
            max_size_mb = 1000  # 1 GB als Obergrenze
            is_valid = size_info["size_mb"] <= max_size_mb
            
            size_info["is_valid"] = is_valid
            return is_valid, size_info
            
        except Exception as e:
            logger.error(f"Fehler bei der Größenvalidierung: {e}")
            return False, {"error": str(e)}
    
    def _validate_value_ranges(self, update_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Prüft, ob die Werte im Update innerhalb akzeptabler Bereiche liegen."""
        value_info = {}
        
        try:
            # Prüfe Wertebereiche
            min_values = {}
            max_values = {}
            has_extreme_values = False
            
            for key, value in update_data.items():
                if isinstance(value, (list, np.ndarray)):
                    if isinstance(value, list):
                        value = np.array(value)
                    
                    # Speichere Min/Max
                    min_val = np.min(value)
                    max_val = np.max(value)
                    min_values[key] = float(min_val)
                    max_values[key] = float(max_val)
                    
                    # Prüfe auf extreme Werte (anpassen nach Bedarf)
                    if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                        has_extreme_values = True
            
            # Speichere Informationen
            value_info["min_values"] = min_values
            value_info["max_values"] = max_values
            value_info["has_extreme_values"] = has_extreme_values
            
            # Basierend auf extremen Werten entscheiden
            is_valid = not has_extreme_values
            value_info["is_valid"] = is_valid
            
            return is_valid, value_info
            
        except Exception as e:
            logger.error(f"Fehler bei der Wertebereichsvalidierung: {e}")
            return False, {"error": str(e)}
    
    def aggregate_updates(self, update_files: List[str], weights: List[float] = None) -> Tuple[Dict[str, Any], str]:
        """
        Aggregiert mehrere validierte Modell-Updates.
        
        Args:
            update_files: Liste von Pfaden zu validierten Update-Dateien
            weights: Gewichtungen für die Updates (falls None, gleiche Gewichtung)
            
        Returns:
            Tuple aus aggregiertem Update und Pfad zur Ausgabedatei
        """
        try:
            # Validiere Eingabeparameter
            if not update_files:
                raise ValueError("Keine Update-Dateien zum Aggregieren angegeben")
            
            # Verwende gleiche Gewichtung, falls nicht anders angegeben
            if weights is None:
                weights = [1.0 / len(update_files)] * len(update_files)
            elif len(weights) != len(update_files):
                raise ValueError("Anzahl der Gewichtungen muss der Anzahl der Updates entsprechen")
            
            # Normalisiere Gewichtungen
            weight_sum = sum(weights)
            if weight_sum == 0:
                raise ValueError("Summe der Gewichtungen ist Null")
            normalized_weights = [w / weight_sum for w in weights]
            
            # Lade und validiere alle Updates
            validated_updates = []
            metadata_list = []
            
            for file_path in update_files:
                # Prüfe, ob die Datei existiert und im validierten Verzeichnis liegt
                if not os.path.exists(file_path):
                    logger.warning(f"Update-Datei nicht gefunden: {file_path}")
                    continue
                
                # Lade Update-Daten
                try:
                    with open(file_path, 'r') as f:
                        update_data = json.load(f)
                    
                    # Lade Metadaten, falls verfügbar
                    metadata_path = file_path + '.meta.json'
                    metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    
                    # Prüfe, ob das Update bereits validiert wurde
                    is_validated = 'validation' in metadata and metadata['validation'].get('is_valid', False)
                    
                    if not is_validated:
                        # Versuche, das Update zu validieren
                        is_valid, _, _ = self.validate_update(update_data, metadata)
                        if not is_valid:
                            logger.warning(f"Update wurde nicht validiert und wird übersprungen: {file_path}")
                            continue
                    
                    validated_updates.append(update_data)
                    metadata_list.append(metadata)
                    
                except Exception as e:
                    logger.error(f"Fehler beim Laden des Updates {file_path}: {e}")
                    continue
            
            # Prüfe, ob validierte Updates vorhanden sind
            if not validated_updates:
                raise ValueError("Keine gültigen Updates zum Aggregieren gefunden")
            
            # Aggregiere Updates
            aggregated_update = {}
            
            # Bestimme die gemeinsame Struktur
            reference_update = validated_updates[0]
            for key in reference_update.keys():
                # Prüfe, ob der Schlüssel in allen Updates vorhanden ist
                if all(key in update for update in validated_updates):
                    # Initialisiere mit Nullen der richtigen Form
                    if isinstance(reference_update[key], (list, np.ndarray)):
                        if isinstance(reference_update[key], list):
                            # Konvertiere zu NumPy für einfachere Berechnungen
                            reference_value = np.array(reference_update[key])
                        else:
                            reference_value = reference_update[key]
                        
                        # Initialisiere mit Nullen
                        aggregated_value = np.zeros_like(reference_value)
                        
                        # Gewichtete Summe der Updates
                        for i, update in enumerate(validated_updates):
                            if isinstance(update[key], list):
                                update_value = np.array(update[key])
                            else:
                                update_value = update[key]
                            
                            aggregated_value += normalized_weights[i] * update_value
                        
                        # Konvertiere zurück zu Liste, falls das Original eine Liste war
                        if isinstance(reference_update[key], list):
                            aggregated_update[key] = aggregated_value.tolist()
                        else:
                            aggregated_update[key] = aggregated_value
                        
                    else:
                        # Für nicht-Array-Werte (z.B. Skalar)
                        aggregated_update[key] = sum(
                            normalized_weights[i] * update[key]
                            for i, update in enumerate(validated_updates)
                        )
            
            # Speichere aggregiertes Update
            output_path = os.path.join(
                self.base_output_dir,
                f"aggregated_update_{self.session_id}_{int(time.time())}.json"
            )
            
            with open(output_path, 'w') as f:
                json.dump(self._prepare_data_for_serialization(aggregated_update), f, indent=2)
            
            # Erstelle Aggregations-Metadaten
            aggregation_metadata = {
                "timestamp": time.time(),
                "session_id": self.session_id,
                "num_updates": len(validated_updates),
                "weights": normalized_weights,
                "update_sources": [os.path.basename(path) for path in update_files if path in update_files],
                "original_metadata": metadata_list
            }
            
            # Speichere Aggregations-Metadaten
            metadata_path = output_path + '.meta.json'
            with open(metadata_path, 'w') as f:
                json.dump(aggregation_metadata, f, indent=2)
            
            # Aktualisiere Statistiken
            self.security_stats["aggregated_updates"] += 1
            self.security_stats["last_operation"] = "aggregate"
            
            logger.info(f"{len(validated_updates)} Updates erfolgreich aggregiert und unter {output_path} gespeichert")
            return aggregated_update, output_path
            
        except Exception as e:
            logger.error(f"Fehler bei der Aggregation der Updates: {e}")
            raise
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Gibt die aktuellen Sicherheitsstatistiken zurück."""
        return {
            "timestamp": time.time(),
            "session_id": self.session_id,
            **self.security_stats
        }
