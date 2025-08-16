#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-REFLEX: Reaction Profile Manager Module
------------------------------------------
Definiert & speichert unterschiedliche Reaktionsprofile.
Lernmodul: Reaktionen können sich durch VXOR-Verlauf weiterentwickeln.

Version: 0.1.0
Author: VXOR Build Core / Omega One
"""

import json
import logging
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

# Konfiguration des Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/ubuntu/vXor_Modules/VX-REFLEX/logs/reflex.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VX-REFLEX.profile_manager")

class ProfileType(Enum):
    """Typen von Reaktionsprofilen"""
    DEFAULT = "default"         # Standardprofil
    EMERGENCY = "emergency"     # Notfallprofil
    SENSITIVE = "sensitive"     # Empfindliches Profil
    DAMPENED = "dampened"       # Gedämpftes Profil
    CUSTOM = "custom"           # Benutzerdefiniertes Profil


class ReactionProfileManager:
    """
    Verwaltet verschiedene Reaktionsprofile für das VX-REFLEX Modul.
    """
    
    def __init__(self, config_path: str = "/home/ubuntu/vXor_Modules/VX-REFLEX/config/reflex_config.json",
                 profiles_dir: str = "/home/ubuntu/vXor_Modules/VX-REFLEX/config/profiles"):
        """
        Initialisiert den ReactionProfileManager mit Konfigurationsparametern.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            profiles_dir: Verzeichnis für Profilkonfigurationen
        """
        self.logger = logger
        self.logger.info("Initialisiere VX-REFLEX ReactionProfileManager...")
        
        self.config_path = config_path
        self.profiles_dir = profiles_dir
        
        # Stelle sicher, dass das Profilverzeichnis existiert
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        # Lade Konfiguration
        try:
            with open(config_path, 'r') as config_file:
                self.config = json.load(config_file)
            self.logger.info("Konfiguration erfolgreich geladen")
        except FileNotFoundError:
            self.logger.warning(f"Konfigurationsdatei {config_path} nicht gefunden. Verwende Standardkonfiguration.")
            self.config = self._get_default_config()
            
            # Erstelle Konfigurationsdatei mit Standardwerten
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as config_file:
                json.dump(self.config, config_file, indent=4)
        
        # Lade Profile
        self.profiles = self._load_profiles()
        
        # Setze aktives Profil
        self.active_profile_name = self.config.get("active_profile", "default")
        if self.active_profile_name not in self.profiles:
            self.logger.warning(f"Aktives Profil '{self.active_profile_name}' nicht gefunden. Verwende 'default'.")
            self.active_profile_name = "default"
        
        # Initialisiere Lernparameter
        self.learning_enabled = self.config.get("learning", {}).get("enabled", False)
        self.learning_rate = self.config.get("learning", {}).get("rate", 0.1)
        self.learning_history = []
        self.max_history_size = 1000
        
        # VXOR-Bridge (wird später gesetzt)
        self.vxor_bridge = None
        
        self.logger.info(f"VX-REFLEX ReactionProfileManager initialisiert mit aktivem Profil '{self.active_profile_name}'")
    
    def _get_default_config(self) -> Dict:
        """Liefert die Standardkonfiguration zurück"""
        return {
            "reaction_profiles": {
                "default": {
                    "response_delay": 0.05,  # in Sekunden
                    "priority_weights": {
                        "HIGH": 1.0,
                        "MEDIUM": 0.6,
                        "LOW": 0.3
                    },
                    "thresholds": {
                        "visual": 0.7,
                        "audio": 0.6,
                        "system": 0.8,
                        "emotional": 0.5
                    },
                    "response_strength": {
                        "soma": 0.8,
                        "psi": 0.7,
                        "system": 1.0
                    }
                },
                "emergency": {
                    "response_delay": 0.01,
                    "priority_weights": {
                        "HIGH": 1.0,
                        "MEDIUM": 0.8,
                        "LOW": 0.5
                    },
                    "thresholds": {
                        "visual": 0.5,
                        "audio": 0.4,
                        "system": 0.6,
                        "emotional": 0.3
                    },
                    "response_strength": {
                        "soma": 1.0,
                        "psi": 0.9,
                        "system": 1.0
                    }
                }
            },
            "active_profile": "default",
            "learning": {
                "enabled": False,
                "rate": 0.1,
                "max_adjustments": {
                    "thresholds": 0.3,
                    "response_strength": 0.3,
                    "priority_weights": 0.2
                }
            }
        }
    
    def _load_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Lädt alle verfügbaren Reaktionsprofile.
        
        Returns:
            Dictionary mit Profilnamen und -daten
        """
        profiles = {}
        
        # Lade eingebettete Profile aus der Konfiguration
        if "reaction_profiles" in self.config:
            profiles.update(self.config["reaction_profiles"])
        
        # Lade externe Profildateien
        if os.path.exists(self.profiles_dir):
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith(".json"):
                    profile_name = os.path.splitext(filename)[0]
                    profile_path = os.path.join(self.profiles_dir, filename)
                    
                    try:
                        with open(profile_path, 'r') as profile_file:
                            profile_data = json.load(profile_file)
                            profiles[profile_name] = profile_data
                            self.logger.info(f"Profil '{profile_name}' aus {profile_path} geladen")
                    except Exception as e:
                        self.logger.error(f"Fehler beim Laden von Profil '{profile_name}': {e}")
        
        # Stelle sicher, dass mindestens das Standardprofil existiert
        if "default" not in profiles:
            profiles["default"] = self.config.get("reaction_profiles", {}).get("default", {
                "response_delay": 0.05,
                "priority_weights": {
                    "HIGH": 1.0,
                    "MEDIUM": 0.6,
                    "LOW": 0.3
                },
                "thresholds": {
                    "visual": 0.7,
                    "audio": 0.6,
                    "system": 0.8,
                    "emotional": 0.5
                },
                "response_strength": {
                    "soma": 0.8,
                    "psi": 0.7,
                    "system": 1.0
                }
            })
        
        return profiles
    
    def set_vxor_bridge(self, bridge):
        """
        Setzt die VXOR-Bridge für die Kommunikation mit anderen Modulen.
        
        Args:
            bridge: VXOR-Bridge-Instanz
        """
        self.vxor_bridge = bridge
        self.logger.info("VXOR-Bridge gesetzt")
    
    def get_active_profile(self) -> Dict[str, Any]:
        """
        Liefert das aktive Reaktionsprofil.
        
        Returns:
            Aktives Reaktionsprofil
        """
        return self.profiles[self.active_profile_name]
    
    def get_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """
        Liefert ein bestimmtes Reaktionsprofil.
        
        Args:
            profile_name: Name des Profils
            
        Returns:
            Reaktionsprofil oder None, wenn nicht gefunden
        """
        return self.profiles.get(profile_name)
    
    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Liefert alle verfügbaren Reaktionsprofile.
        
        Returns:
            Dictionary mit allen Profilen
        """
        return self.profiles
    
    def set_active_profile(self, profile_name: str) -> bool:
        """
        Setzt das aktive Reaktionsprofil.
        
        Args:
            profile_name: Name des Profils
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        if profile_name in self.profiles:
            self.active_profile_name = profile_name
            
            # Aktualisiere Konfiguration
            self.config["active_profile"] = profile_name
            self._save_config()
            
            self.logger.info(f"Aktives Profil auf '{profile_name}' gesetzt")
            return True
        else:
            self.logger.warning(f"Profil '{profile_name}' nicht gefunden")
            return False
    
    def create_profile(self, profile_name: str, profile_data: Dict[str, Any], 
                       profile_type: ProfileType = ProfileType.CUSTOM) -> bool:
        """
        Erstellt ein neues Reaktionsprofil.
        
        Args:
            profile_name: Name des Profils
            profile_data: Profildaten
            profile_type: Typ des Profils
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        # Prüfe, ob Profil bereits existiert
        if profile_name in self.profiles:
            self.logger.warning(f"Profil '{profile_name}' existiert bereits")
            return False
        
        # Validiere Profildaten
        if not self._validate_profile(profile_data):
            self.logger.error(f"Ungültige Profildaten für '{profile_name}'")
            return False
        
        # Füge Profiltyp hinzu
        profile_data["type"] = profile_type.value
        
        # Speichere Profil
        self.profiles[profile_name] = profile_data
        
        # Speichere Profil in Datei, wenn es sich um ein benutzerdefiniertes Profil handelt
        if profile_type == ProfileType.CUSTOM:
            profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
            try:
                with open(profile_path, 'w') as profile_file:
                    json.dump(profile_data, profile_file, indent=4)
                self.logger.info(f"Profil '{profile_name}' in {profile_path} gespeichert")
            except Exception as e:
                self.logger.error(f"Fehler beim Speichern von Profil '{profile_name}': {e}")
                return False
        
        self.logger.info(f"Profil '{profile_name}' erstellt")
        return True
    
    def update_profile(self, profile_name: str, profile_data: Dict[str, Any]) -> bool:
        """
        Aktualisiert ein bestehendes Reaktionsprofil.
        
        Args:
            profile_name: Name des Profils
            profile_data: Neue Profildaten
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        # Prüfe, ob Profil existiert
        if profile_name not in self.profiles:
            self.logger.warning(f"Profil '{profile_name}' existiert nicht")
            return False
        
        # Validiere Profildaten
        if not self._validate_profile(profile_data):
            self.logger.error(f"Ungültige Profildaten für '{profile_name}'")
            return False
        
        # Behalte Profiltyp bei
        profile_data["type"] = self.profiles[profile_name].get("type", ProfileType.CUSTOM.value)
        
        # Aktualisiere Profil
        self.profiles[profile_name] = profile_data
        
        # Speichere Profil in Datei, wenn es sich um ein benutzerdefiniertes Profil handelt
        if profile_data["type"] == ProfileType.CUSTOM.value:
            profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
            try:
                with open(profile_path, 'w') as profile_file:
                    json.dump(profile_data, profile_file, indent=4)
                self.logger.info(f"Profil '{profile_name}' in {profile_path} aktualisiert")
            except Exception as e:
                self.logger.error(f"Fehler beim Speichern von Profil '{profile_name}': {e}")
                return False
        
        self.logger.info(f"Profil '{profile_name}' aktualisiert")
        return True
    
    def delete_profile(self, profile_name: str) -> bool:
        """
        Löscht ein Reaktionsprofil.
        
        Args:
            profile_name: Name des Profils
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        # Prüfe, ob Profil existiert
        if profile_name not in self.profiles:
            self.logger.warning(f"Profil '{profile_name}' existiert nicht")
            return False
        
        # Verhindere Löschen des aktiven Profils
        if profile_name == self.active_profile_name:
            self.logger.warning(f"Aktives Profil '{profile_name}' kann nicht gelöscht werden")
            return False
        
        # Verhindere Löschen des Standardprofils
        if profile_name == "default":
            self.logger.warning("Standardprofil 'default' kann nicht gelöscht werden")
            return False
        
        # Lösche Profil
        del self.profiles[profile_name]
        
        # Lösche Profildatei, wenn vorhanden
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
        if os.path.exists(profile_path):
            try:
                os.remove(profile_path)
                self.logger.info(f"Profildatei {profile_path} gelöscht")
            except Exception as e:
                self.logger.error(f"Fehler beim Löschen von Profildatei '{profile_path}': {e}")
                return False
        
        self.logger.info(f"Profil '{profile_name}' gelöscht")
        return True
    
    def _validate_profile(self, profile_data: Dict[str, Any]) -> bool:
        """
        Validiert Profildaten.
        
        Args:
            profile_data: Zu validierende Profildaten
            
        Returns:
            True, wenn gültig, sonst False
        """
        # Prüfe erforderliche Felder
        required_fields = ["response_delay", "priority_weights", "thresholds", "response_strength"]
        for field in required_fields:
            if field not in profile_data:
                self.logger.error(f"Erforderliches Feld '{field}' fehlt in Profildaten")
                return False
        
        # Prüfe Prioritätsgewichte
        if not isinstance(profile_data["priority_weights"], dict):
            self.logger.error("Prioritätsgewichte müssen ein Dictionary sein")
            return False
        
        for priority in ["HIGH", "MEDIUM", "LOW"]:
            if priority not in profile_data["priority_weights"]:
                self.logger.error(f"Prioritätsgewicht '{priority}' fehlt in Profildaten")
                return False
        
        # Prüfe Schwellenwerte
        if not isinstance(profile_data["thresholds"], dict):
            self.logger.error("Schwellenwerte müssen ein Dictionary sein")
            return False
        
        for stimulus_type in ["visual", "audio", "system", "emotional"]:
            if stimulus_type not in profile_data["thresholds"]:
                self.logger.error(f"Schwellenwert für '{stimulus_type}' fehlt in Profildaten")
                return False
        
        # Prüfe Reaktionsstärken
        if not isinstance(profile_data["response_strength"], dict):
            self.logger.error("Re
(Content truncated due to size limit. Use line ranges to read in chunks)