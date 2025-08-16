#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - PRISM-Engine Event Generator

Der EventGenerator erzeugt Ereignisse für die PRISM-Engine zur Simulation und Analyse.
Er ist ein wesentlicher Bestandteil des MISO-Systems und arbeitet eng mit der PRISM-Engine zusammen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import datetime
import logging
import random
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import uuid
from pathlib import Path

# Konfiguriere Logging
logger = logging.getLogger("MISO.prism.event_generator")

class EventType(Enum):
    """Typen von Ereignissen für die PRISM-Engine"""
    DATA_POINT = auto()
    PATTERN_CHANGE = auto()
    THRESHOLD_CROSSED = auto()
    ANOMALY_DETECTED = auto()
    PREDICTION_TRIGGER = auto()
    SIMULATION_TRIGGER = auto()
    FEEDBACK_LOOP = auto()
    SYSTEM_STATUS = auto()
    USER_INPUT = auto()
    EXTERNAL_DATA = auto()

@dataclass
class Event:
    """Ereignis für die PRISM-Engine"""
    type: EventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "event_generator"
    priority: int = 0

class EventGenerator:
    """Generator für Ereignisse in der PRISM-Engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den EventGenerator
        
        Args:
            config: Konfigurationsparameter
        """
        self.config = config or {}
        self.patterns = {}
        self.thresholds = {}
        self.anomaly_detectors = {}
        self.prediction_triggers = {}
        self.simulation_triggers = {}
        self.running = False
        self.event_callback = None
        logger.info("EventGenerator initialisiert")
    
    def register_event_callback(self, callback: Callable[[Event], None]):
        """
        Registriert einen Callback für Ereignisse
        
        Args:
            callback: Callback-Funktion
        """
        self.event_callback = callback
        logger.info("Event-Callback registriert")
    
    def start(self):
        """Startet den EventGenerator"""
        self.running = True
        logger.info("EventGenerator gestartet")
    
    def stop(self):
        """Stoppt den EventGenerator"""
        self.running = False
        logger.info("EventGenerator gestoppt")
    
    def add_data_point(self, stream_id: str, value: Any):
        """
        Fügt einen Datenpunkt hinzu und erzeugt ein Ereignis
        
        Args:
            stream_id: ID des Datenstroms
            value: Wert des Datenpunkts
        """
        event = Event(
            type=EventType.DATA_POINT,
            data={
                "stream_id": stream_id,
                "value": value
            }
        )
        self._emit_event(event)
        
        # Prüfe Schwellenwerte
        self._check_thresholds(stream_id, value)
        
        # Prüfe auf Anomalien
        self._check_anomalies(stream_id, value)
        
        # Prüfe auf Musteränderungen
        self._check_pattern_changes(stream_id, value)
        
        # Prüfe auf Vorhersage-Trigger
        self._check_prediction_triggers(stream_id, value)
        
        # Prüfe auf Simulations-Trigger
        self._check_simulation_triggers(stream_id, value)
    
    def register_pattern(self, pattern_id: str, pattern_data: Dict[str, Any]):
        """
        Registriert ein Muster für die Erkennung von Musteränderungen
        
        Args:
            pattern_id: ID des Musters
            pattern_data: Musterdaten
        """
        self.patterns[pattern_id] = pattern_data
        logger.info(f"Muster {pattern_id} registriert")
    
    def register_threshold(self, threshold_id: str, stream_id: str, value: float, direction: str = "above"):
        """
        Registriert einen Schwellenwert für einen Datenstrom
        
        Args:
            threshold_id: ID des Schwellenwerts
            stream_id: ID des Datenstroms
            value: Schwellenwert
            direction: Richtung (above, below)
        """
        self.thresholds[threshold_id] = {
            "stream_id": stream_id,
            "value": value,
            "direction": direction,
            "crossed": False
        }
        logger.info(f"Schwellenwert {threshold_id} für Datenstrom {stream_id} registriert")
    
    def register_anomaly_detector(self, detector_id: str, stream_id: str, detector_config: Dict[str, Any]):
        """
        Registriert einen Anomaliedetektor für einen Datenstrom
        
        Args:
            detector_id: ID des Detektors
            stream_id: ID des Datenstroms
            detector_config: Konfiguration des Detektors
        """
        self.anomaly_detectors[detector_id] = {
            "stream_id": stream_id,
            "config": detector_config,
            "state": {}
        }
        logger.info(f"Anomaliedetektor {detector_id} für Datenstrom {stream_id} registriert")
    
    def register_prediction_trigger(self, trigger_id: str, stream_id: str, condition: Dict[str, Any]):
        """
        Registriert einen Vorhersage-Trigger für einen Datenstrom
        
        Args:
            trigger_id: ID des Triggers
            stream_id: ID des Datenstroms
            condition: Bedingung für den Trigger
        """
        self.prediction_triggers[trigger_id] = {
            "stream_id": stream_id,
            "condition": condition,
            "triggered": False
        }
        logger.info(f"Vorhersage-Trigger {trigger_id} für Datenstrom {stream_id} registriert")
    
    def register_simulation_trigger(self, trigger_id: str, stream_id: str, condition: Dict[str, Any]):
        """
        Registriert einen Simulations-Trigger für einen Datenstrom
        
        Args:
            trigger_id: ID des Triggers
            stream_id: ID des Datenstroms
            condition: Bedingung für den Trigger
        """
        self.simulation_triggers[trigger_id] = {
            "stream_id": stream_id,
            "condition": condition,
            "triggered": False
        }
        logger.info(f"Simulations-Trigger {trigger_id} für Datenstrom {stream_id} registriert")
    
    def _emit_event(self, event: Event):
        """
        Sendet ein Ereignis an den registrierten Callback
        
        Args:
            event: Zu sendendes Ereignis
        """
        if self.event_callback and self.running:
            self.event_callback(event)
    
    def _check_thresholds(self, stream_id: str, value: Any):
        """
        Prüft, ob ein Datenpunkt einen Schwellenwert überschreitet
        
        Args:
            stream_id: ID des Datenstroms
            value: Wert des Datenpunkts
        """
        for threshold_id, threshold in self.thresholds.items():
            if threshold["stream_id"] != stream_id:
                continue
            
            crossed = False
            if threshold["direction"] == "above" and value > threshold["value"]:
                crossed = True
            elif threshold["direction"] == "below" and value < threshold["value"]:
                crossed = True
            
            if crossed != threshold["crossed"]:
                threshold["crossed"] = crossed
                event = Event(
                    type=EventType.THRESHOLD_CROSSED,
                    data={
                        "threshold_id": threshold_id,
                        "stream_id": stream_id,
                        "value": value,
                        "threshold_value": threshold["value"],
                        "direction": threshold["direction"],
                        "crossed": crossed
                    }
                )
                self._emit_event(event)
    
    def _check_anomalies(self, stream_id: str, value: Any):
        """
        Prüft, ob ein Datenpunkt eine Anomalie darstellt
        
        Args:
            stream_id: ID des Datenstroms
            value: Wert des Datenpunkts
        """
        for detector_id, detector in self.anomaly_detectors.items():
            if detector["stream_id"] != stream_id:
                continue
            
            # Einfache Anomalieerkennung basierend auf Z-Score
            if "history" not in detector["state"]:
                detector["state"]["history"] = []
            
            history = detector["state"]["history"]
            history.append(value)
            
            # Begrenze Historienlänge
            max_history = detector["config"].get("max_history", 100)
            if len(history) > max_history:
                history = history[-max_history:]
                detector["state"]["history"] = history
            
            # Berechne Z-Score
            if len(history) > 10:
                mean = np.mean(history[:-1])
                std = np.std(history[:-1])
                if std > 0:
                    z_score = abs((value - mean) / std)
                    threshold = detector["config"].get("z_threshold", 3.0)
                    
                    if z_score > threshold:
                        event = Event(
                            type=EventType.ANOMALY_DETECTED,
                            data={
                                "detector_id": detector_id,
                                "stream_id": stream_id,
                                "value": value,
                                "z_score": z_score,
                                "threshold": threshold,
                                "mean": mean,
                                "std": std
                            }
                        )
                        self._emit_event(event)
    
    def _check_pattern_changes(self, stream_id: str, value: Any):
        """
        Prüft, ob ein Datenpunkt eine Musteränderung darstellt
        
        Args:
            stream_id: ID des Datenstroms
            value: Wert des Datenpunkts
        """
        for pattern_id, pattern in self.patterns.items():
            if pattern.get("stream_id") != stream_id:
                continue
            
            # Implementiere Mustererkennungslogik hier
            # ...
            
            # Beispiel: Einfache Trendänderungserkennung
            if "history" not in pattern:
                pattern["history"] = []
            
            history = pattern["history"]
            history.append(value)
            
            # Begrenze Historienlänge
            max_history = pattern.get("max_history", 10)
            if len(history) > max_history:
                history = history[-max_history:]
                pattern["history"] = history
            
            # Erkenne Trendänderung
            if len(history) >= 3:
                prev_trend = history[-3] < history[-2]
                curr_trend = history[-2] < history[-1]
                
                if prev_trend != curr_trend:
                    event = Event(
                        type=EventType.PATTERN_CHANGE,
                        data={
                            "pattern_id": pattern_id,
                            "stream_id": stream_id,
                            "value": value,
                            "prev_trend": "up" if prev_trend else "down",
                            "curr_trend": "up" if curr_trend else "down"
                        }
                    )
                    self._emit_event(event)
    
    def _check_prediction_triggers(self, stream_id: str, value: Any):
        """
        Prüft, ob ein Datenpunkt einen Vorhersage-Trigger auslöst
        
        Args:
            stream_id: ID des Datenstroms
            value: Wert des Datenpunkts
        """
        for trigger_id, trigger in self.prediction_triggers.items():
            if trigger["stream_id"] != stream_id:
                continue
            
            # Implementiere Triggerlogik hier
            # ...
            
            # Beispiel: Einfacher Schwellenwert-Trigger
            threshold = trigger["condition"].get("threshold")
            if threshold is not None:
                triggered = value > threshold
                
                if triggered != trigger["triggered"]:
                    trigger["triggered"] = triggered
                    
                    if triggered:
                        event = Event(
                            type=EventType.PREDICTION_TRIGGER,
                            data={
                                "trigger_id": trigger_id,
                                "stream_id": stream_id,
                                "value": value,
                                "threshold": threshold
                            }
                        )
                        self._emit_event(event)
    
    def _check_simulation_triggers(self, stream_id: str, value: Any):
        """
        Prüft, ob ein Datenpunkt einen Simulations-Trigger auslöst
        
        Args:
            stream_id: ID des Datenstroms
            value: Wert des Datenpunkts
        """
        for trigger_id, trigger in self.simulation_triggers.items():
            if trigger["stream_id"] != stream_id:
                continue
            
            # Implementiere Triggerlogik hier
            # ...
            
            # Beispiel: Einfacher Schwellenwert-Trigger
            threshold = trigger["condition"].get("threshold")
            if threshold is not None:
                triggered = value > threshold
                
                if triggered != trigger["triggered"]:
                    trigger["triggered"] = triggered
                    
                    if triggered:
                        event = Event(
                            type=EventType.SIMULATION_TRIGGER,
                            data={
                                "trigger_id": trigger_id,
                                "stream_id": stream_id,
                                "value": value,
                                "threshold": threshold
                            }
                        )
                        self._emit_event(event)
