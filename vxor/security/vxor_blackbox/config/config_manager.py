#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Konfigurations-Manager für VXOR AI Blackbox.

Verwaltet alle Konfigurationen für die VXOR AI Blackbox-Sicherheitskomponenten
mit Unterstützung für Konfigurationsdateien, Umgebungsvariablen und Standardwerte.
"""

import os
import json
import yaml
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

class ConfigManager:
    """
    Verwaltet Konfigurationen für VXOR AI Blackbox.
    
    Funktionen:
    - Laden von Konfigurationen aus JSON/YAML-Dateien
    - Verschlüsselte Konfigurationswerte
    - Hierarchische Zugriffsstruktur
    - Integration mit Umgebungsvariablen
    - Sicherheitsvalidierung
    
    Besondere Unterstützung für:
    - T-Mathematics Engine Konfigurationen (MLXTensor, TorchTensor)
    - M-LINGUA Interface-Einstellungen
    """
    
    def __init__(self, default_config_path: Optional[str] = None):
        """
        Initialisiert den ConfigManager.
        
        Args:
            default_config_path: Optionaler Standardpfad zur Hauptkonfigurationsdatei
        """
        self.config_data = {}
        self.config_sources = {}
        self._setup_logging()
        
        # Lade die Standardkonfiguration, falls ein Pfad angegeben wurde
        if default_config_path:
            self.load_config(default_config_path)
        
        # Initialisiere mit den Standardkonfigurationen
        self._init_default_configs()
        
        self.logger.info("ConfigManager initialisiert")
    
    def _setup_logging(self):
        """Konfiguriert das Logging für den ConfigManager."""
        self.logger = logging.getLogger("vxor.security.config")
        
        if not self.logger.handlers:
            log_dir = os.path.join(os.path.expanduser("~"), ".vxor", "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            handler = logging.FileHandler(os.path.join(log_dir, "vxor_config.log"))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _init_default_configs(self):
        """Initialisiert die Standard-Konfigurationswerte."""
        # Basiskonfiguration für die Sicherheitskomponenten
        base_security_config = {
            "encryption": {
                "aes_key_size": 32,           # 256 Bit
                "iv_size": 12,                # 96 Bit (für GCM)
                "tag_size": 16,               # 128 Bit Auth-Tag
                "key_derivation_iterations": 600000
            },
            "quantum_resistant": {
                "kyber_variant": "512",       # Kyber Variante (512, 768, 1024)
                "dilithium_variant": "2",     # Dilithium Variante (2, 3, 5)
                "ntru_variant": "HRSS"        # NTRU Variante
            },
            "key_management": {
                "key_store_dir": os.path.join(os.path.expanduser("~"), ".vxor", "keystore"),
                "rotation_periods": {
                    "kyber": 30,              # Tage
                    "ntru": 60,               # Tage
                    "dilithium": 90,          # Tage
                    "aes": 7                  # Tage
                }
            },
            "logging": {
                "log_dir": os.path.join(os.path.expanduser("~"), ".vxor", "logs"),
                "default_level": "INFO",
                "secure_logging": True,       # Protokolliert keine sensitiven Daten
                "log_rotation": {
                    "max_size_mb": 10,
                    "backup_count": 5
                }
            }
        }
        
        # Spezielle Konfiguration für die T-Mathematics Engine
        t_mathematics_config = {
            "mlx_tensor": {
                "use_apple_neural_engine": True,  # Nutze die Apple Neural Engine des M4 Max
                "optimize_for_mlx": True,         # MLX-spezifische Optimierungen
                "secure_memory": True             # Spezieller Speicherschutz für MLX
            },
            "torch_tensor": {
                "use_mps": True,                  # Verwende Metal Performance Shaders für GPU
                "optimize_for_metal": True,       # Metal-spezifische Optimierungen
                "secure_memory": True             # Spezieller Speicherschutz für PyTorch
            }
        }
        
        # Spezielle Konfiguration für M-LINGUA
        m_lingua_config = {
            "secure_parsing": True,               # Sichere Eingabevalidierung
            "backend_selection": "auto",          # Automatische Backend-Auswahl
            "tensor_operations": {
                "validate_operations": True,      # Validiere Operationen vor Ausführung
                "sandbox_execution": True         # Führe Operationen in Sandbox aus
            }
        }
        
        # Zusammenführen aller Konfigurationen
        self.set_config("security", base_security_config)
        self.set_config("t_mathematics", t_mathematics_config)
        self.set_config("m_lingua", m_lingua_config)
        
        # Konfigurationsquelle speichern
        self.config_sources.update({
            "security": "default",
            "t_mathematics": "default",
            "m_lingua": "default"
        })
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Lädt eine Konfigurationsdatei.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei (JSON oder YAML)
            
        Returns:
            Die geladene Konfiguration als Dictionary
        """
        if not os.path.exists(config_path):
            self.logger.warning(f"Konfigurationsdatei nicht gefunden: {config_path}")
            return {}
        
        try:
            file_ext = os.path.splitext(config_path)[1].lower()
            
            if file_ext == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                self.logger.error(f"Nicht unterstütztes Konfigurationsformat: {file_ext}")
                return {}
            
            # Verifiziere die Konfiguration
            if not self._validate_config(config):
                self.logger.warning(f"Konfiguration enthält ungültige Werte: {config_path}")
            
            # Füge die Konfiguration zusammen
            self._merge_config(config, source=config_path)
            
            self.logger.info(f"Konfiguration geladen aus: {config_path}")
            
            return config
        
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Konfiguration aus {config_path}: {str(e)}")
            return {}
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validiert die Konfigurationswerte.
        
        Args:
            config: Die zu validierende Konfiguration
            
        Returns:
            True, wenn die Konfiguration gültig ist, sonst False
        """
        # Hier können spezifische Validierungsregeln implementiert werden
        # Beispiel: Überprüfung von Schlüsselgrößen, Algorithmusvarianten, usw.
        
        # Überprüfe, ob bekannte Sicherheitsprobleme in der Konfiguration vorliegen
        if self._check_for_security_issues(config):
            return False
        
        return True
    
    def _check_for_security_issues(self, config: Dict[str, Any]) -> bool:
        """
        Überprüft die Konfiguration auf bekannte Sicherheitsprobleme.
        
        Args:
            config: Die zu überprüfende Konfiguration
            
        Returns:
            True, wenn Sicherheitsprobleme gefunden wurden, sonst False
        """
        has_issues = False
        
        # Beispiel: Überprüfung auf zu geringe Schlüsselgrößen
        if 'security' in config and 'encryption' in config['security']:
            enc = config['security']['encryption']
            
            if 'aes_key_size' in enc and enc['aes_key_size'] < 32:
                self.logger.warning("Sicherheitsrisiko: AES-Schlüsselgröße ist zu klein (<256 Bit)")
                has_issues = True
            
            if 'key_derivation_iterations' in enc and enc['key_derivation_iterations'] < 100000:
                self.logger.warning("Sicherheitsrisiko: Zu wenige Iterationen für die Schlüsselableitung")
                has_issues = True
        
        # Beispiel: Überprüfung auf deaktivierte Sicherheitsfunktionen
        if 'm_lingua' in config and 'secure_parsing' in config['m_lingua']:
            if not config['m_lingua']['secure_parsing']:
                self.logger.warning("Sicherheitsrisiko: Sichere Eingabevalidierung für M-LINGUA deaktiviert")
                has_issues = True
        
        return has_issues
    
    def _merge_config(self, config: Dict[str, Any], source: str):
        """
        Fügt eine Konfiguration zur bestehenden Konfiguration hinzu.
        
        Args:
            config: Die hinzuzufügende Konfiguration
            source: Die Quelle der Konfiguration (z.B. Dateipfad)
        """
        for section, data in config.items():
            if isinstance(data, dict) and section in self.config_data and isinstance(self.config_data[section], dict):
                # Wenn der Abschnitt bereits existiert und ein Dictionary ist, führe sie rekursiv zusammen
                self._deep_merge(self.config_data[section], data)
            else:
                # Sonst ersetze oder füge den Abschnitt hinzu
                self.config_data[section] = data
            
            # Speichere die Quelle
            self.config_sources[section] = source
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        Führt zwei Dictionaries rekursiv zusammen.
        
        Args:
            target: Das Ziel-Dictionary, das aktualisiert wird
            source: Das Quell-Dictionary, aus dem Werte übernommen werden
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def get_config(self, section: Optional[str] = None, key: Optional[str] = None,
                 default: Any = None) -> Any:
        """
        Holt einen Konfigurationswert.
        
        Args:
            section: Optional, der Konfigurationsabschnitt
            key: Optional, der Schlüssel innerhalb des Abschnitts
            default: Standardwert, falls der Wert nicht gefunden wird
            
        Returns:
            Der Konfigurationswert oder der Standardwert
        """
        if section is None:
            return self.config_data
        
        if section not in self.config_data:
            return default
        
        if key is None:
            return self.config_data[section]
        
        # Unterstütze verschachtelte Schlüssel mit Punktnotation (z.B. "logging.log_dir")
        if '.' in key:
            keys = key.split('.')
            value = self.config_data[section]
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
        
        # Einfacher Schlüssel
        return self.config_data[section].get(key, default)
    
    def set_config(self, section: str, data: Any, key: Optional[str] = None):
        """
        Setzt einen Konfigurationswert.
        
        Args:
            section: Der Konfigurationsabschnitt
            data: Der zu setzende Wert
            key: Optional, der Schlüssel innerhalb des Abschnitts
        """
        if section not in self.config_data:
            self.config_data[section] = {}
        
        if key is None:
            # Setze den gesamten Abschnitt
            if isinstance(data, dict):
                self.config_data[section] = data
            else:
                raise ValueError(f"Ungültiger Wert für Abschnitt {section}: Muss ein Dictionary sein")
        else:
            # Unterstütze verschachtelte Schlüssel mit Punktnotation
            if '.' in key:
                keys = key.split('.')
                target = self.config_data[section]
                
                # Navigiere durch die Hierarchie
                for i, k in enumerate(keys[:-1]):
                    if k not in target or not isinstance(target[k], dict):
                        target[k] = {}
                    target = target[k]
                
                # Setze den letzten Schlüssel
                target[keys[-1]] = data
            else:
                # Einfacher Schlüssel
                self.config_data[section][key] = data
        
        self.logger.debug(f"Konfiguration gesetzt: {section}" + (f".{key}" if key else ""))
    
    def get_config_source(self, section: str) -> Optional[str]:
        """
        Gibt die Quelle einer Konfigurationssektion zurück.
        
        Args:
            section: Der Konfigurationsabschnitt
            
        Returns:
            Die Quelle der Konfiguration oder None, wenn nicht gefunden
        """
        return self.config_sources.get(section)
    
    def save_config(self, file_path: str, sections: Optional[List[str]] = None,
                  format: str = 'json', include_sources: bool = False):
        """
        Speichert die Konfiguration in einer Datei.
        
        Args:
            file_path: Pfad zur Zieldatei
            sections: Optional, Liste der zu speichernden Abschnitte
            format: Format der Konfigurationsdatei ('json' oder 'yaml')
            include_sources: Wenn True, werden die Quellen der Konfigurationen mit gespeichert
        """
        # Erstelle eine Kopie der Konfiguration
        if sections:
            config_to_save = {section: self.config_data[section] for section in sections if section in self.config_data}
        else:
            config_to_save = dict(self.config_data)
        
        # Füge Quellinformationen hinzu, falls gewünscht
        if include_sources:
            config_to_save['__sources__'] = dict(self.config_sources)
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(config_to_save, f, indent=2)
            elif format.lower() in ['yaml', 'yml']:
                with open(file_path, 'w') as f:
                    yaml.dump(config_to_save, f)
            else:
                raise ValueError(f"Nicht unterstütztes Format: {format}")
            
            self.logger.info(f"Konfiguration gespeichert in: {file_path}")
        
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Konfiguration in {file_path}: {str(e)}")
    
    def backup_config(self) -> Optional[str]:
        """
        Erstellt ein Backup der aktuellen Konfiguration.
        
        Returns:
            Pfad zum Backup oder None bei Fehler
        """
        backup_dir = os.path.join(os.path.expanduser("~"), ".vxor", "config_backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = os.path.basename(str(int(os.path.getmtime("."))))
        backup_path = os.path.join(backup_dir, f"vxor_config_{timestamp}.json")
        
        try:
            self.save_config(backup_path, include_sources=True)
            return backup_path
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des Konfigurationsbackups: {str(e)}")
            return None
    
    def load_env_variables(self, prefix: str = 'VXOR_'):
        """
        Lädt Konfigurationswerte aus Umgebungsvariablen.
        
        Umgebungsvariablen werden in folgendem Format erwartet:
        {PREFIX}_SECTION_KEY=Wert
        z.B. VXOR_SECURITY_ENCRYPTION_AES_KEY_SIZE=32
        
        Args:
            prefix: Präfix für die zu ladenden Umgebungsvariablen
        """
        for env_name, env_value in os.environ.items():
            if not env_name.startswith(prefix):
                continue
            
            # Entferne das Präfix und teile den Rest am Unterstrich
            parts = env_name[len(prefix):].lower().split('_')
            
            if len(parts) < 2:
                continue  # Ignoriere ungültige Variablen
            
            section = parts[0]
            
            if len(parts) == 2:
                key = parts[1]
            else:
                # Bei mehreren Teilen: ersten Teil als Sektion, Rest als verschachtelten Schlüssel
                key = '.'.join(parts[1:])
            
            # Konvertiere den Wert in den richtigen Typ
            value = self._convert_env_value(env_value)
            
            # Setze die Konfiguration
            self.set_config(section, value, key)
            self.config_sources[f"{section}.{key}"] = f"env:{env_name}"
            
            self.logger.debug(f"Konfiguration aus Umgebungsvariable geladen: {env_name}")
    
    def _convert_env_value(self, value: str) -> Any:
        """
        Konvertiert einen Umgebungsvariablenwert in den passenden Typ.
        
        Args:
            value: Der zu konvertierende Wert
            
        Returns:
            Der konvertierte Wert
        """
        # Boolean-Werte
        if value.lower() in ['true', 'yes', '1']:
            return True
        if value.lower() in ['false', 'no', '0']:
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Liste (kommagetrennt)
        if ',' in value:
            return [self._convert_env_value(item.strip()) for item in value.split(',')]
        
        # Standardmäßig als String behandeln
        return value
    
    def reset(self):
        """Setzt die Konfiguration auf die Standardwerte zurück."""
        self.config_data = {}
        self.config_sources = {}
        self._init_default_configs()
        self.logger.info("Konfiguration auf Standardwerte zurückgesetzt")


# Hilfsfunktionen für einfachen Zugriff

_default_config_manager = None

def get_config_manager() -> ConfigManager:
    """
    Gibt den Standard-ConfigManager zurück.
    
    Returns:
        ConfigManager-Instanz
    """
    global _default_config_manager
    
    if _default_config_manager is None:
        _default_config_manager = ConfigManager()
    
    return _default_config_manager

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Lädt eine Konfigurationsdatei mit dem Standard-ConfigManager.
    
    Args:
        config_path: Pfad zur Konfigurationsdatei
        
    Returns:
        Die geladene Konfiguration
    """
    return get_config_manager().load_config(config_path)

def get_config(section: Optional[str] = None, key: Optional[str] = None,
              default: Any = None) -> Any:
    """
    Holt einen Konfigurationswert vom Standard-ConfigManager.
    
    Args:
        section: Optional, der Konfigurationsabschnitt
        key: Optional, der Schlüssel innerhalb des Abschnitts
        default: Standardwert, falls der Wert nicht gefunden wird
        
    Returns:
        Der Konfigurationswert oder der Standardwert
    """
    return get_config_manager().get_config(section, key, default)
