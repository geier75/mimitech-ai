#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-REFLEX: Reflex Responder Module
----------------------------------
Reflexausführung je nach Regelwerk & Reaktionsprofil.
Zuweisung von Körperaktionen (VX-SOMA) oder Bewusstseinsimpulsen (VX-PSI).

Version: 0.1.0
Author: VXOR Build Core / Omega One
"""

import json
import logging
import time
import threading
from enum import Enum
from typing import Dict, List, Any, Callable, Optional, Tuple

# Konfiguration des Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/ubuntu/vXor_Modules/VX-REFLEX/logs/reflex.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VX-REFLEX.responder")

class ResponseType(Enum):
    """Typen von Reflexantworten"""
    SOMA = "soma"       # Körperliche Reaktion
    PSI = "psi"         # Bewusstseinsimpuls
    SYSTEM = "system"   # Systemreaktion
    COMBINED = "combined"  # Kombinierte Reaktion


class ReflexResponder:
    """
    Führt Reflexreaktionen basierend auf analysierten Reizen aus.
    """
    
    def __init__(self, config_path: str = "/home/ubuntu/vXor_Modules/VX-REFLEX/config/reflex_config.json"):
        """
        Initialisiert den ReflexResponder mit Konfigurationsparametern.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.logger = logger
        self.logger.info("Initialisiere VX-REFLEX ReflexResponder...")
        
        # Lade Konfiguration
        try:
            with open(config_path, 'r') as config_file:
                self.config = json.load(config_file)
            self.logger.info("Konfiguration erfolgreich geladen")
        except FileNotFoundError:
            self.logger.warning(f"Konfigurationsdatei {config_path} nicht gefunden. Verwende Standardkonfiguration.")
            self.config = self._get_default_config()
        
        # Initialisiere Reaktionsregeln
        self.response_rules = self._load_response_rules()
        
        # Initialisiere VXOR-Bridge (wird später importiert)
        self.vxor_bridge = None
        
        # Initialisiere Reaktionsprofile
        self.active_profile = self.config.get("active_profile", "default")
        
        # Performance-Tracking
        self.performance_metrics = {
            "total_responses": 0,
            "response_times": [],
            "avg_response_time": 0,
            "max_response_time": 0,
            "response_types": {
                "soma": 0,
                "psi": 0,
                "system": 0,
                "combined": 0
            }
        }
        
        # Reaktionshistorie
        self.response_history = []
        self.max_history_size = 100
        
        self.logger.info("VX-REFLEX ReflexResponder initialisiert")
    
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
                    }
                },
                "emergency": {
                    "response_delay": 0.01,
                    "priority_weights": {
                        "HIGH": 1.0,
                        "MEDIUM": 0.8,
                        "LOW": 0.5
                    }
                }
            },
            "active_profile": "default",
            "response_rules": {
                "visual": [
                    {"pattern": "schlag", "response": {"type": "soma", "action": "dodge"}},
                    {"pattern": "sturz", "response": {"type": "soma", "action": "brace"}},
                    {"pattern": "feuer", "response": {"type": "combined", "actions": [
                        {"type": "soma", "action": "retreat"},
                        {"type": "psi", "alert": "danger-fire"}
                    ]}}
                ],
                "audio": [
                    {"pattern": "schrei", "response": {"type": "soma", "action": "turn_to_source"}},
                    {"pattern": "explosion", "response": {"type": "soma", "action": "duck"}}
                ],
                "system": [
                    {"warning": "cpu_overload", "response": {"type": "system", "action": "reduce_processes"}},
                    {"warning": "temperature_critical", "response": {"type": "system", "action": "emergency_cooldown"}}
                ]
            }
        }
    
    def _load_response_rules(self) -> Dict[str, List[Dict]]:
        """
        Lädt die Reaktionsregeln für verschiedene Reiztypen.
        
        Returns:
            Dictionary mit Reaktionsregeln
        """
        if "response_rules" in self.config:
            return self.config["response_rules"]
        else:
            # Standardregeln
            return {
                "visual": [
                    {"pattern": "schlag", "response": {"type": "soma", "action": "dodge"}},
                    {"pattern": "sturz", "response": {"type": "soma", "action": "brace"}},
                    {"pattern": "feuer", "response": {"type": "combined", "actions": [
                        {"type": "soma", "action": "retreat"},
                        {"type": "psi", "alert": "danger-fire"}
                    ]}}
                ],
                "audio": [
                    {"pattern": "schrei", "response": {"type": "soma", "action": "turn_to_source"}},
                    {"pattern": "explosion", "response": {"type": "soma", "action": "duck"}}
                ],
                "system": [
                    {"warning": "cpu_overload", "response": {"type": "system", "action": "reduce_processes"}},
                    {"warning": "temperature_critical", "response": {"type": "system", "action": "emergency_cooldown"}}
                ],
                "danger": [
                    {"level": "high", "response": {"type": "combined", "actions": [
                        {"type": "soma", "action": "emergency_stop"},
                        {"type": "psi", "alert": "danger-high"}
                    ]}}
                ],
                "emotional": [
                    {"emotion": "angst", "intensity_min": 0.7, "response": {"type": "psi", "alert": "emotion-fear"}}
                ]
            }
    
    def set_vxor_bridge(self, bridge):
        """
        Setzt die VXOR-Bridge für die Kommunikation mit anderen Modulen.
        
        Args:
            bridge: VXOR-Bridge-Instanz
        """
        self.vxor_bridge = bridge
        self.logger.info("VXOR-Bridge gesetzt")
    
    def set_active_profile(self, profile_name: str):
        """
        Setzt das aktive Reaktionsprofil.
        
        Args:
            profile_name: Name des Reaktionsprofils
        """
        if profile_name in self.config["reaction_profiles"]:
            self.active_profile = profile_name
            self.logger.info(f"Aktives Reaktionsprofil auf '{profile_name}' gesetzt")
        else:
            self.logger.warning(f"Reaktionsprofil '{profile_name}' nicht gefunden. Behalte '{self.active_profile}'.")
    
    def respond_to_stimulus(self, analyzed_stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt eine Reflexreaktion basierend auf einem analysierten Reiz aus.
        
        Args:
            analyzed_stimulus: Analysierter Reiz
            
        Returns:
            Ergebnis der Reaktion
        """
        start_time = time.perf_counter()
        
        # Extrahiere Reiztyp
        stimulus_type = analyzed_stimulus.get("type", "unknown")
        
        # Protokolliere eingehenden Reiz
        self.logger.debug(f"Reagiere auf Reiz: Typ={stimulus_type}, Daten={analyzed_stimulus}")
        
        # Bestimme passende Reaktion
        response_data = self._determine_response(stimulus_type, analyzed_stimulus)
        
        # Führe Reaktion aus
        result = self._execute_response(response_data)
        
        # Speichere Reaktion in Historie
        self._add_to_history(stimulus_type, analyzed_stimulus, response_data, result)
        
        # Aktualisiere Performance-Metriken
        end_time = time.perf_counter()
        response_time = (end_time - start_time) * 1000  # in ms
        
        self.performance_metrics["total_responses"] += 1
        self.performance_metrics["response_times"].append(response_time)
        self.performance_metrics["avg_response_time"] = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"])
        self.performance_metrics["max_response_time"] = max(self.performance_metrics["response_times"])
        
        # Zähle Reaktionstypen
        response_type = response_data.get("type", "unknown")
        if response_type in self.performance_metrics["response_types"]:
            self.performance_metrics["response_types"][response_type] += 1
        
        if response_time > 80:  # 80ms Schwellenwert
            self.logger.warning(f"Reaktionszeit überschreitet Schwellenwert: {response_time:.2f}ms für {stimulus_type}")
        
        return result
    
    def _determine_response(self, stimulus_type: str, analyzed_stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bestimmt die passende Reaktion für einen analysierten Reiz.
        
        Args:
            stimulus_type: Typ des Reizes
            analyzed_stimulus: Analysierter Reiz
            
        Returns:
            Reaktionsdaten
        """
        # Standardreaktion
        default_response = {"type": "psi", "alert": "unknown-stimulus"}
        
        # Prüfe, ob Regeln für diesen Reiztyp existieren
        if stimulus_type not in self.response_rules:
            self.logger.warning(f"Keine Reaktionsregeln für Reiztyp '{stimulus_type}' gefunden")
            return default_response
        
        # Durchsuche Regeln nach passender Reaktion
        rules = self.response_rules[stimulus_type]
        
        # Visuelle Reize
        if stimulus_type == "visual":
            # Prüfe auf erkannte Muster
            if "patterns_detected" in analyzed_stimulus and analyzed_stimulus["patterns_detected"]:
                for pattern in analyzed_stimulus["patterns_detected"]:
                    for rule in rules:
                        if "pattern" in rule and rule["pattern"] == pattern:
                            return rule["response"]
            
            # Prüfe auf Gefahrenobjekte
            if "danger_objects" in analyzed_stimulus and analyzed_stimulus["danger_objects"]:
                for obj in analyzed_stimulus["danger_objects"]:
                    for rule in rules:
                        if "danger_object" in rule and rule["danger_object"] == obj:
                            return rule["response"]
        
        # Auditive Reize
        elif stimulus_type == "audio":
            # Prüfe auf erkannte Muster
            if "patterns_detected" in analyzed_stimulus and analyzed_stimulus["patterns_detected"]:
                for pattern in analyzed_stimulus["patterns_detected"]:
                    for rule in rules:
                        if "pattern" in rule and rule["pattern"] == pattern:
                            return rule["response"]
            
            # Prüfe auf Lautstärke
            if "level" in analyzed_stimulus:
                level = analyzed_stimulus["level"]
                for rule in rules:
                    if "min_level" in rule and "max_level" in rule:
                        if rule["min_level"] <= level <= rule["max_level"]:
                            return rule["response"]
        
        # Systemreize
        elif stimulus_type == "system":
            # Prüfe auf Warnungen
            if "warnings" in analyzed_stimulus and analyzed_stimulus["warnings"]:
                for warning in analyzed_stimulus["warnings"]:
                    for rule in rules:
                        if "warning" in rule and rule["warning"] == warning:
                            return rule["response"]
            
            # Prüfe auf Systemstatus
            if "system_status" in analyzed_stimulus:
                status = analyzed_stimulus["system_status"]
                for rule in rules:
                    if "status" in rule and rule["status"] == status:
                        return rule["response"]
        
        # Emotionale Reize
        elif stimulus_type == "emotional":
            # Prüfe auf Emotionstyp und Intensität
            if "emotion_type" in analyzed_stimulus and "intensity" in analyzed_stimulus:
                emotion = analyzed_stimulus["emotion_type"]
                intensity = analyzed_stimulus["intensity"]
                
                for rule in rules:
                    if "emotion" in rule and rule["emotion"] == emotion:
                        # Prüfe Intensitätsschwelle
                        if "intensity_min" in rule and intensity >= rule["intensity_min"]:
                            return rule["response"]
        
        # Gefahrenreize
        elif stimulus_type == "danger":
            # Prüfe auf Gefahrenlevel
            if "danger_level" in analyzed_stimulus:
                level = analyzed_stimulus["danger_level"]
                for rule in rules:
                    if "level" in rule and rule["level"] == level:
                        return rule["response"]
            
            # Prüfe auf unmittelbare Reaktion
            if "immediate_response_required" in analyzed_stimulus and analyzed_stimulus["immediate_response_required"]:
                for rule in rules:
                    if "immediate" in rule and rule["immediate"]:
                        return rule["response"]
        
        # Keine passende Regel gefunden
        self.logger.warning(f"Keine passende Reaktionsregel für Reiz gefunden: {stimulus_type}")
        return default_response
    
    def _execute_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt eine Reaktion aus.
        
        Args:
            response_data: Reaktionsdaten
            
        Returns:
            Ergebnis der Reaktion
        """
        # Extrahiere Reaktionstyp
        response_type = response_data.get("type", "unknown")
        
        # Protokolliere Reaktion
        self.logger.info(f"Führe Reaktion aus: Typ={response_type}, Daten={response_data}")
        
        # Prüfe, ob VXOR-Bridge verfügbar ist
        if self.vxor_bridge is None:
            self.logger.error("VXOR-Bridge nicht verfügbar. Reaktion kann nicht ausgeführt werden.")
            return {"success": False, "error": "VXOR-Bridge nicht verfügbar"}
        
        # Führe Reaktion entsprechend ihrem Typ aus
        if response_type == ResponseType.SOMA.value:
            # Körperliche Reaktion über VX-SOMA
            action = response_data.get("action", "")
            result = self.vxor_bridge.send_signal("vx-soma", {"action": action})
            return {"success": True, "type": "soma", "action": action, "result": result}
        
        elif response_type == ResponseType.PSI.value:
            # Bewusstseinsimpuls über VX-PSI
            alert = response_data.get("alert", "")
            result = self.vxor_bridge.send_signal("vx-psi", {"alert": alert})
            return {"success": True, "type": "psi", "alert": alert, "result": result}
        
        elif response_type == ResponseType.SYSTEM.value:
            # Systemreaktion
            action = response_data.get("action", "")
            result = self._execute_system_action(action)
            return {"success": True, "type": "system", "action": action, "result": result}
        
        elif response_type == ResponseType.COMBINED.value:
            # Kombinierte Reaktion
            actions = response_data.get("actions", [])
   
(Content truncated due to size limit. Use line ranges to read in chunks)