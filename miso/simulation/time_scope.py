#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Time Scope Unit

Zeitliche Analyseeinheit für die PRISM-Engine.
Ermöglicht die Analyse und Manipulation von Zeitfenstern und temporalen Daten.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import datetime
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum, auto
import uuid

# Importiere Basisklassen und -typen aus prism_base
from miso.simulation.prism_base import TimeNode, Timeline, TimelineType, calculate_probability

# Konfiguriere Logging
logger = logging.getLogger("MISO.simulation.time_scope")


class TimeScope(Enum):
    """Zeitliche Bereiche für die Analyse"""
    IMMEDIATE = auto()    # Sofortige Zeitfenster (Sekunden bis Minuten)
    SHORT = auto()        # Kurze Zeitfenster (Minuten bis Stunden)
    MEDIUM = auto()       # Mittlere Zeitfenster (Stunden bis Tage)
    LONG = auto()         # Lange Zeitfenster (Tage bis Wochen)
    EXTENDED = auto()     # Erweiterte Zeitfenster (Wochen bis Monate)
    STRATEGIC = auto()    # Strategische Zeitfenster (Monate bis Jahre)


class TimeScopeUnit:
    """
    Zeitliche Analyseeinheit für die PRISM-Engine
    
    Ermöglicht die Analyse und Manipulation von Zeitfenstern und temporalen Daten.
    Unterstützt verschiedene zeitliche Bereiche und Granularitäten.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert die TimeScopeUnit
        
        Args:
            config: Konfigurationsparameter
        """
        self.config = config or {}
        self.time_windows = {}
        self.temporal_data = {}
        self.active_scopes = set()
        
        # Standardwerte für Zeitfenster in Sekunden
        self.scope_durations = {
            TimeScope.IMMEDIATE: self.config.get("immediate_duration", 300),        # 5 Minuten
            TimeScope.SHORT: self.config.get("short_duration", 3600),               # 1 Stunde
            TimeScope.MEDIUM: self.config.get("medium_duration", 86400),            # 1 Tag
            TimeScope.LONG: self.config.get("long_duration", 604800),               # 1 Woche
            TimeScope.EXTENDED: self.config.get("extended_duration", 2592000),      # 30 Tage
            TimeScope.STRATEGIC: self.config.get("strategic_duration", 31536000)    # 1 Jahr
        }
        
        # Initialisiere aktive Zeitbereiche
        for scope in TimeScope:
            if self.config.get(f"enable_{scope.name.lower()}", True):
                self.active_scopes.add(scope)
                self.time_windows[scope] = {
                    "start_time": time.time(),
                    "end_time": time.time() + self.scope_durations[scope],
                    "data_points": []
                }
        
        logger.info(f"TimeScopeUnit initialisiert mit {len(self.active_scopes)} aktiven Zeitbereichen")
    
    def register_temporal_data(self, data_id: str, data: Any, scope: TimeScope = TimeScope.MEDIUM):
        """
        Registriert temporale Daten für einen bestimmten Zeitbereich
        
        Args:
            data_id: Eindeutige ID für die Daten
            data: Die zu registrierenden Daten
            scope: Zeitbereich für die Daten
            
        Returns:
            True, wenn die Daten erfolgreich registriert wurden, sonst False
        """
        if scope not in self.active_scopes:
            logger.warning(f"Zeitbereich {scope.name} ist nicht aktiv")
            return False
        
        timestamp = time.time()
        
        if data_id not in self.temporal_data:
            self.temporal_data[data_id] = {}
        
        self.temporal_data[data_id][timestamp] = {
            "data": data,
            "scope": scope,
            "expiry": timestamp + self.scope_durations[scope]
        }
        
        # Füge Datenpunkt zum Zeitfenster hinzu
        self.time_windows[scope]["data_points"].append({
            "timestamp": timestamp,
            "data_id": data_id,
            "data": data
        })
        
        logger.debug(f"Temporale Daten mit ID {data_id} für Zeitbereich {scope.name} registriert")
        return True
    
    def get_temporal_data(self, data_id: str, scope: Optional[TimeScope] = None) -> List[Dict[str, Any]]:
        """
        Gibt temporale Daten für eine bestimmte ID zurück
        
        Args:
            data_id: ID der Daten
            scope: Optional, filtert nach Zeitbereich
            
        Returns:
            Liste von temporalen Datenpunkten
        """
        if data_id not in self.temporal_data:
            return []
        
        current_time = time.time()
        result = []
        
        for timestamp, entry in self.temporal_data[data_id].items():
            # Überprüfe, ob der Eintrag abgelaufen ist
            if entry["expiry"] < current_time:
                continue
            
            # Filtere nach Zeitbereich, falls angegeben
            if scope is not None and entry["scope"] != scope:
                continue
            
            result.append({
                "timestamp": timestamp,
                "data": entry["data"],
                "scope": entry["scope"],
                "expiry": entry["expiry"]
            })
        
        return result
    
    def update_time_windows(self):
        """
        Aktualisiert die Zeitfenster und entfernt abgelaufene Daten
        
        Returns:
            Anzahl der entfernten Datenpunkte
        """
        current_time = time.time()
        removed_count = 0
        
        # Aktualisiere Zeitfenster
        for scope in self.active_scopes:
            window = self.time_windows[scope]
            
            # Überprüfe, ob das Fenster aktualisiert werden muss
            if current_time > window["end_time"]:
                # Berechne neues Zeitfenster
                duration = self.scope_durations[scope]
                window["start_time"] = current_time
                window["end_time"] = current_time + duration
                
                # Entferne abgelaufene Datenpunkte
                new_data_points = []
                for point in window["data_points"]:
                    if point["timestamp"] + duration > current_time:
                        new_data_points.append(point)
                    else:
                        removed_count += 1
                
                window["data_points"] = new_data_points
        
        # Entferne abgelaufene Einträge aus temporal_data
        for data_id in list(self.temporal_data.keys()):
            for timestamp in list(self.temporal_data[data_id].keys()):
                entry = self.temporal_data[data_id][timestamp]
                if entry["expiry"] < current_time:
                    del self.temporal_data[data_id][timestamp]
                    removed_count += 1
            
            # Entferne leere Einträge
            if not self.temporal_data[data_id]:
                del self.temporal_data[data_id]
        
        logger.debug(f"{removed_count} abgelaufene Datenpunkte entfernt")
        return removed_count
    
    def get_data_in_time_window(self, scope: TimeScope, start_time: Optional[float] = None, 
                              end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Gibt alle Datenpunkte in einem bestimmten Zeitfenster zurück
        
        Args:
            scope: Zeitbereich
            start_time: Optional, Startzeit des Fensters (Standard: aktuelles Fenster)
            end_time: Optional, Endzeit des Fensters (Standard: aktuelles Fenster)
            
        Returns:
            Liste von Datenpunkten im Zeitfenster
        """
        if scope not in self.active_scopes:
            logger.warning(f"Zeitbereich {scope.name} ist nicht aktiv")
            return []
        
        window = self.time_windows[scope]
        
        # Verwende aktuelle Fenstergrenzen, falls nicht anders angegeben
        if start_time is None:
            start_time = window["start_time"]
        if end_time is None:
            end_time = window["end_time"]
        
        # Filtere Datenpunkte nach Zeitfenster
        result = []
        for point in window["data_points"]:
            if start_time <= point["timestamp"] <= end_time:
                result.append(point)
        
        return result
    
    def get_timeline_data(self, timeline_id: str, start_time: Optional[float] = None, 
                       end_time: Optional[float] = None, scope: TimeScope = TimeScope.MEDIUM) -> Dict[str, Any]:
        """
        Gibt Daten für eine bestimmte Zeitlinie zurück, gefiltert nach Zeitfenster
        
        Diese Methode wird von der EventGenerator-Klasse verwendet, um Zeitliniendaten
        für die Ereignisgenerierung abzurufen.
        
        Args:
            timeline_id: ID der Zeitlinie
            start_time: Optional, Startzeit des Zeitfensters
            end_time: Optional, Endzeit des Zeitfensters
            scope: Zeitbereich für die Daten
            
        Returns:
            Dictionary mit Zeitliniendaten und Metadaten
        """
        # Standardwerte für Start- und Endzeit
        if start_time is None:
            start_time = time.time() - self.scope_durations[scope]
        if end_time is None:
            end_time = time.time()
        
        # Suche nach Daten mit dem timeline_id-Präfix
        timeline_data = []
        for data_id in self.temporal_data.keys():
            if data_id.startswith(f"timeline_{timeline_id}_") or data_id == timeline_id:
                # Hole alle temporalen Daten für diese ID
                temporal_points = self.get_temporal_data(data_id, scope)
                
                # Filtere nach Zeitfenster
                for point in temporal_points:
                    if start_time <= point["timestamp"] <= end_time:
                        timeline_data.append({
                            "data_id": data_id,
                            "timestamp": point["timestamp"],
                            "data": point["data"]
                        })
        
        # Sortiere Daten nach Zeitstempel
        timeline_data.sort(key=lambda x: x["timestamp"])
        
        # Erstelle Ergebnisdictionary
        result = {
            "timeline_id": timeline_id,
            "start_time": start_time,
            "end_time": end_time,
            "scope": scope.name,
            "data_points": timeline_data,
            "point_count": len(timeline_data)
        }
        
        logger.debug(f"Zeitliniendaten für {timeline_id} abgerufen: {len(timeline_data)} Datenpunkte")
        return result
    
    def analyze_temporal_trends(self, data_id: str, scope: TimeScope = TimeScope.MEDIUM) -> Dict[str, Any]:
        """
        Analysiert temporale Trends für bestimmte Daten
        
        Args:
            data_id: ID der Daten
            scope: Zeitbereich für die Analyse
            
        Returns:
            Analyseergebnis mit Trends und Statistiken
        """
        data_points = self.get_temporal_data(data_id, scope)
        
        if not data_points:
            return {
                "data_id": data_id,
                "scope": scope.name,
                "data_points": 0,
                "trend": "insufficient_data",
                "statistics": {}
            }
        
        # Extrahiere Zeitstempel und Werte
        timestamps = []
        values = []
        
        for point in data_points:
            timestamps.append(point["timestamp"])
            
            # Versuche, einen numerischen Wert zu extrahieren
            data = point["data"]
            if isinstance(data, (int, float)):
                values.append(float(data))
            elif isinstance(data, dict) and "value" in data and isinstance(data["value"], (int, float)):
                values.append(float(data["value"]))
            elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (int, float)):
                values.append(float(data[0]))
            else:
                # Nicht-numerische Daten überspringen
                continue
        
        if not values:
            return {
                "data_id": data_id,
                "scope": scope.name,
                "data_points": len(data_points),
                "trend": "non_numeric_data",
                "statistics": {}
            }
        
        # Berechne Statistiken
        mean_value = np.mean(values)
        median_value = np.median(values)
        std_dev = np.std(values)
        min_value = np.min(values)
        max_value = np.max(values)
        
        # Bestimme Trend
        if len(values) >= 2:
            # Einfache lineare Regression für den Trend
            x = np.array(timestamps)
            y = np.array(values)
            
            # Normalisiere x für numerische Stabilität
            x = x - x[0]
            
            # Berechne Steigung
            if np.sum(x**2) > 0:  # Vermeide Division durch Null
                slope = np.sum(x * y) / np.sum(x**2)
                
                # Bestimme Trendrichtung
                if slope > 0.01:
                    trend = "increasing"
                elif slope < -0.01:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "data_id": data_id,
            "scope": scope.name,
            "data_points": len(values),
            "trend": trend,
            "statistics": {
                "mean": mean_value,
                "median": median_value,
                "std_dev": std_dev,
                "min": min_value,
                "max": max_value,
                "range": max_value - min_value
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Gibt den Status der TimeScopeUnit zurück
        
        Returns:
            Statusbericht
        """
        return {
            "active_scopes": [scope.name for scope in self.active_scopes],
            "time_windows": {scope.name: {
                "start_time": window["start_time"],
                "end_time": window["end_time"],
                "data_points": len(window["data_points"])
            } for scope, window in self.time_windows.items()},
            "temporal_data_count": len(self.temporal_data),
            "total_data_points": sum(len(window["data_points"]) for window in self.time_windows.values())
        }


# Beispiel für die Verwendung der TimeScopeUnit
if __name__ == "__main__":
    # Konfiguriere Logging für Standalone-Ausführung
    logging.basicConfig(level=logging.INFO)
    
    # Erstelle TimeScopeUnit
    time_scope_unit = TimeScopeUnit()
    
    # Registriere einige Testdaten
    time_scope_unit.register_temporal_data("test1", 42, TimeScope.IMMEDIATE)
    time_scope_unit.register_temporal_data("test1", 43, TimeScope.IMMEDIATE)
    time_scope_unit.register_temporal_data("test1", 44, TimeScope.IMMEDIATE)
    
    time_scope_unit.register_temporal_data("test2", {"value": 10}, TimeScope.SHORT)
    time_scope_unit.register_temporal_data("test2", {"value": 15}, TimeScope.SHORT)
    
    # Analysiere Trends
    trend_analysis = time_scope_unit.analyze_temporal_trends("test1", TimeScope.IMMEDIATE)
    print(f"Trend-Analyse für test1: {trend_analysis}")
    
    # Zeige Status
    status = time_scope_unit.get_status()
    print(f"Status der TimeScopeUnit: {status}")
