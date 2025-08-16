#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-REFLEX: Stimulus Analyzer Module
-----------------------------------
Analyse eingehender Reize (optisch, auditiv, intern) und
Klassifizierung nach Typ, Relevanz & Dringlichkeit.

Version: 0.1.0
Author: VXOR Build Core / Omega One
"""

import json
import logging
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
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

logger = logging.getLogger("VX-REFLEX.stimulus")

class StimulusType(Enum):
    """Typen von Reizen, die das System verarbeiten kann"""
    VISUAL = "visual"
    AUDIO = "audio"
    SYSTEM = "system"
    EMOTIONAL = "emotional"
    DANGER = "danger"
    UNKNOWN = "unknown"


class StimulusAnalyzer:
    """
    Analysiert eingehende Reize und klassifiziert sie nach Typ, Relevanz und Dringlichkeit.
    """
    
    def __init__(self, config_path: str = "/home/ubuntu/vXor_Modules/VX-REFLEX/config/reflex_config.json"):
        """
        Initialisiert den StimulusAnalyzer mit Konfigurationsparametern.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.logger = logger
        self.logger.info("Initialisiere VX-REFLEX StimulusAnalyzer...")
        
        # Lade Konfiguration
        try:
            with open(config_path, 'r') as config_file:
                self.config = json.load(config_file)
            self.logger.info("Konfiguration erfolgreich geladen")
        except FileNotFoundError:
            self.logger.warning(f"Konfigurationsdatei {config_path} nicht gefunden. Verwende Standardkonfiguration.")
            self.config = self._get_default_config()
        
        # Initialisiere Analysefunktionen für verschiedene Reiztypen
        self.analyzers = {
            StimulusType.VISUAL.value: self._analyze_visual_stimulus,
            StimulusType.AUDIO.value: self._analyze_audio_stimulus,
            StimulusType.SYSTEM.value: self._analyze_system_stimulus,
            StimulusType.EMOTIONAL.value: self._analyze_emotional_stimulus,
            StimulusType.DANGER.value: self._analyze_danger_stimulus
        }
        
        # Initialisiere Muster-Erkennung
        self.pattern_database = self._load_pattern_database()
        
        # Performance-Tracking
        self.performance_metrics = {
            "total_analyses": 0,
            "analysis_times": [],
            "avg_analysis_time": 0,
            "max_analysis_time": 0,
        }
        
        self.logger.info("VX-REFLEX StimulusAnalyzer initialisiert")
    
    def _get_default_config(self) -> Dict:
        """Liefert die Standardkonfiguration zurück"""
        return {
            "thresholds": {
                "cpu_load": {
                    "high": 90,
                    "medium": 75,
                    "low": 50
                },
                "audio_level": {
                    "high": 75,  # in dB
                    "medium": 60,
                    "low": 40
                },
                "object_proximity": {
                    "high": 0.8,  # in Metern
                    "medium": 1.5,
                    "low": 3.0
                }
            },
            "pattern_recognition": {
                "visual_patterns": [
                    "schlag",
                    "sturz",
                    "explosion",
                    "feuer",
                    "waffe"
                ],
                "audio_patterns": [
                    "schrei",
                    "explosion",
                    "alarm",
                    "glasbruch"
                ],
                "danger_objects": [
                    "messer",
                    "feuer",
                    "waffe",
                    "fahrzeug"
                ]
            },
            "analysis": {
                "visual_sensitivity": 0.8,
                "audio_sensitivity": 0.7,
                "system_sensitivity": 0.9,
                "emotional_sensitivity": 0.6
            }
        }
    
    def _load_pattern_database(self) -> Dict[str, List[Dict]]:
        """
        Lädt die Muster-Datenbank für die Reizerkennung.
        In einer realen Implementierung würde dies komplexere Muster und ML-Modelle laden.
        
        Returns:
            Dictionary mit Mustern für verschiedene Reiztypen
        """
        # Einfache Implementierung für Demonstrationszwecke
        # In einer realen Implementierung würden hier ML-Modelle oder komplexere Muster geladen
        return {
            "visual": [
                {"name": "schlag", "features": ["schnelle Bewegung", "Richtungsänderung"]},
                {"name": "sturz", "features": ["vertikale Bewegung", "Beschleunigung"]},
                {"name": "explosion", "features": ["plötzliche Expansion", "Helligkeitsänderung"]},
                {"name": "feuer", "features": ["rote/orange Farbe", "flackernde Bewegung"]},
                {"name": "waffe", "features": ["längliche Form", "metallische Oberfläche"]}
            ],
            "audio": [
                {"name": "schrei", "features": ["hohe Frequenz", "plötzlicher Anstieg"]},
                {"name": "explosion", "features": ["lauter Knall", "tiefe Frequenz"]},
                {"name": "alarm", "features": ["wiederholtes Muster", "konstante Frequenz"]},
                {"name": "glasbruch", "features": ["hohe Frequenz", "kurze Dauer"]}
            ],
            "danger": [
                {"name": "messer", "features": ["scharfe Kante", "reflektierende Oberfläche"]},
                {"name": "feuer", "features": ["hohe Temperatur", "Rauch"]},
                {"name": "waffe", "features": ["metallisch", "Lauf"]},
                {"name": "fahrzeug", "features": ["Bewegung", "Motor"]}
            ]
        }
    
    def analyze_stimulus(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analysiert einen eingehenden Reiz und klassifiziert ihn.
        
        Args:
            raw_data: Rohdaten des Reizes
            
        Returns:
            Analysierter und klassifizierter Reiz
        """
        start_time = time.perf_counter()
        
        # Bestimme Reiztyp
        stimulus_type = self._determine_stimulus_type(raw_data)
        
        # Protokolliere eingehenden Reiz
        self.logger.debug(f"Analysiere Reiz: Typ={stimulus_type}, Rohdaten={raw_data}")
        
        # Analysiere Reiz entsprechend seinem Typ
        if stimulus_type.value in self.analyzers:
            analyzed_data = self.analyzers[stimulus_type.value](raw_data)
        else:
            self.logger.warning(f"Kein Analyzer für Reiztyp '{stimulus_type}' verfügbar")
            analyzed_data = {"type": stimulus_type.value, "raw_data": raw_data, "analyzed": False}
        
        # Füge Metadaten hinzu
        analyzed_data["timestamp"] = time.time()
        analyzed_data["type"] = stimulus_type.value
        
        # Aktualisiere Performance-Metriken
        end_time = time.perf_counter()
        analysis_time = (end_time - start_time) * 1000  # in ms
        
        self.performance_metrics["total_analyses"] += 1
        self.performance_metrics["analysis_times"].append(analysis_time)
        self.performance_metrics["avg_analysis_time"] = sum(self.performance_metrics["analysis_times"]) / len(self.performance_metrics["analysis_times"])
        self.performance_metrics["max_analysis_time"] = max(self.performance_metrics["analysis_times"])
        
        if analysis_time > 30:  # 30ms Schwellenwert für Analyse
            self.logger.warning(f"Analysezeit überschreitet Schwellenwert: {analysis_time:.2f}ms für {stimulus_type}")
        
        return analyzed_data
    
    def _determine_stimulus_type(self, raw_data: Dict[str, Any]) -> StimulusType:
        """
        Bestimmt den Typ eines Reizes basierend auf seinen Rohdaten.
        
        Args:
            raw_data: Rohdaten des Reizes
            
        Returns:
            Typ des Reizes
        """
        # Prüfe auf expliziten Typ in den Daten
        if "stimulus_type" in raw_data:
            try:
                return StimulusType(raw_data["stimulus_type"])
            except ValueError:
                pass
        
        # Bestimme Typ anhand der Datenstruktur
        if "image" in raw_data or "frame" in raw_data or "visual_pattern" in raw_data:
            return StimulusType.VISUAL
        elif "audio" in raw_data or "sound" in raw_data or "frequency" in raw_data or "level" in raw_data:
            return StimulusType.AUDIO
        elif "cpu_load" in raw_data or "memory" in raw_data or "temperature" in raw_data:
            return StimulusType.SYSTEM
        elif "emotion" in raw_data or "feeling" in raw_data or "mood" in raw_data:
            return StimulusType.EMOTIONAL
        elif "danger" in raw_data or "threat" in raw_data or "warning" in raw_data:
            return StimulusType.DANGER
        
        # Standardtyp für unbekannte Reize
        return StimulusType.UNKNOWN
    
    def _analyze_visual_stimulus(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analysiert einen visuellen Reiz.
        
        Args:
            raw_data: Rohdaten des visuellen Reizes
            
        Returns:
            Analysierte Daten
        """
        result = {
            "analyzed": True,
            "patterns_detected": [],
            "danger_objects": [],
            "motion_detected": False,
            "proximity": None
        }
        
        # Prüfe auf Bewegungsmuster
        if "visual_pattern" in raw_data:
            pattern = raw_data["visual_pattern"]
            if pattern in self.config["pattern_recognition"]["visual_patterns"]:
                result["patterns_detected"].append(pattern)
                result["motion_detected"] = True
        
        # Prüfe auf Objekterkennung
        if "objects" in raw_data:
            objects = raw_data["objects"]
            for obj in objects:
                if obj["name"] in self.config["pattern_recognition"]["danger_objects"]:
                    result["danger_objects"].append(obj["name"])
                    
                    # Prüfe auf Nähe zu gefährlichem Objekt
                    if "distance" in obj:
                        result["proximity"] = obj["distance"]
                        result["danger_object"] = True
        
        # Simulierte Mustererkennung für Demonstrationszwecke
        # In einer realen Implementierung würde hier Computer Vision eingesetzt
        if "frame" in raw_data:
            # Simuliere Bildanalyse
            result["frame_analyzed"] = True
            
            # Simuliere Bewegungserkennung
            if "motion_data" in raw_data:
                motion_data = raw_data["motion_data"]
                if isinstance(motion_data, dict) and "velocity" in motion_data:
                    velocity = motion_data["velocity"]
                    if velocity > 5.0:  # Schwellenwert für schnelle Bewegung
                        result["motion_detected"] = True
                        result["patterns_detected"].append("schnelle_bewegung")
        
        return result
    
    def _analyze_audio_stimulus(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analysiert einen auditiven Reiz.
        
        Args:
            raw_data: Rohdaten des auditiven Reizes
            
        Returns:
            Analysierte Daten
        """
        result = {
            "analyzed": True,
            "patterns_detected": [],
            "level": None,
            "frequency_analysis": {}
        }
        
        # Extrahiere Lautstärkepegel
        if "level" in raw_data:
            result["level"] = raw_data["level"]
        
        # Prüfe auf Audiomuster
        if "audio_pattern" in raw_data:
            pattern = raw_data["audio_pattern"]
            if pattern in self.config["pattern_recognition"]["audio_patterns"]:
                result["patterns_detected"].append(pattern)
        
        # Simulierte Frequenzanalyse für Demonstrationszwecke
        # In einer realen Implementierung würde hier Signalverarbeitung eingesetzt
        if "frequency_data" in raw_data:
            freq_data = raw_data["frequency_data"]
            
            # Einfache Frequenzanalyse
            if isinstance(freq_data, list) and len(freq_data) > 0:
                # Berechne Durchschnitt und Maximum
                avg_freq = sum(freq_data) / len(freq_data)
                max_freq = max(freq_data)
                
                result["frequency_analysis"] = {
                    "avg_frequency": avg_freq,
                    "max_frequency": max_freq,
                    "is_high_pitched": max_freq > 2000,  # Schwellenwert für hohe Töne
                    "is_low_pitched": avg_freq < 300     # Schwellenwert für tiefe Töne
                }
                
                # Musteranalyse basierend auf Frequenzdaten
                if max_freq > 3000 and "level" in raw_data and raw_data["level"] > 70:
                    result["patterns_detected"].append("schrei")
                elif avg_freq < 300 and "level" in raw_data and raw_data["level"] > 80:
                    result["patterns_detected"].append("explosion")
        
        return result
    
    def _analyze_system_stimulus(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analysiert einen Systemreiz.
        
        Args:
            raw_data: Rohdaten des Systemreizes
            
        Returns:
            Analysierte Daten
        """
        result = {
            "analyzed": True,
            "system_status": "normal",
            "warnings": []
        }
        
        # Prüfe CPU-Auslastung
        if "cpu_load" in raw_data:
            cpu_load = raw_data["cpu_load"]
            result["cpu_load"] = cpu_load
            
            if cpu_load >= self.config["thresholds"]["cpu_load"]["high"]:
                result["system_status"] = "critical"
                result["warnings"].append("cpu_overload")
            elif cpu_load >= self.config["thresholds"]["cpu_load"]["medium"]:
                result["system_status"] = "warning"
                result["warnings"].append("cpu_high")
        
        # Prüfe Speicherauslastung
        if "memory" in raw_data:
            memory = raw_data["memory"]
            result["memory"] = memory
            
            if memory >= 90:  # 90% Schwellenwert
                result["system_status"] = "critical"
                result["warnings"].append("memory_overload")
            elif memory >= 75:  # 75% Schwellenwert
                if result["system_status"] != "critical":
                    result["system_status"] = "warning"
                result["warnings"].append("memory_high")
        
        # Prüfe Systemtemperatur
        if "temperature" in raw_data:
            temperature = raw_data["temperature"]
            result["temperature"] = temperature
            
            if temperature >= 85:  # 85°C Schwellenwert
                result["system_status"] = "critical"
                result["warnings"].append("temperature_critical")
            elif temperature >= 75:  # 75°C Schwellenwert
                if result["system_status"] != "critical":
                    result["system_status"] = "warning"
                result["warnings"].append("temperature_high")
        
        return result
    
    def _analyze_emotional_stimulus(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analysiert einen emotionalen Reiz.
        
        Args:
            raw_data: Rohdaten des emotionalen Reizes
            
        Returns:
            Analysierte Daten
        """
        result = {
      
(Content truncated due to size limit. Use line ranges to read in chunks)