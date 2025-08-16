#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - TimeScopeUnit

Zeitskalierung für Vorhersagekontext in der PRISM-Engine.
Skaliert Simulation zwischen Minuten bis Jahren, ohne Genauigkeit zu verlieren.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import json
import datetime
import numpy as np

# Konfiguriere Logging
logger = logging.getLogger("MISO.simulation.time_scope")


class TimeScopeUnit:
    """
    Zeitskalierung (z. B. 5 Sek. bis 5 Jahre) für Vorhersagekontext
    Skaliert Simulation zwischen Minuten bis Jahren, ohne Genauigkeit zu verlieren
    """
    
    def __init__(self):
        """Initialisiert die TimeScopeUnit"""
        self.time_scales = {
            "seconds": 1,
            "minutes": 60,
            "hours": 3600,
            "days": 86400,
            "weeks": 604800,
            "months": 2592000,  # 30 Tage
            "years": 31536000   # 365 Tage
        }
        self.current_scale = "days"
        self.reference_time = time.time()
        
        # Spezifische Skalierungsparameter für verschiedene Zeitskalen
        self.scale_parameters = {
            "seconds": {"granularity": 0.1, "max_steps": 600},      # 0.1 Sek. Granularität, max. 1 Minute
            "minutes": {"granularity": 1.0, "max_steps": 180},      # 1 Min. Granularität, max. 3 Stunden
            "hours": {"granularity": 0.25, "max_steps": 96},        # 15 Min. Granularität, max. 24 Stunden
            "days": {"granularity": 1.0, "max_steps": 90},          # 1 Tag Granularität, max. 3 Monate
            "weeks": {"granularity": 1.0, "max_steps": 52},         # 1 Woche Granularität, max. 1 Jahr
            "months": {"granularity": 1.0, "max_steps": 60},        # 1 Monat Granularität, max. 5 Jahre
            "years": {"granularity": 1.0, "max_steps": 100}         # 1 Jahr Granularität, max. 100 Jahre
        }
        
        logger.info(f"TimeScopeUnit initialisiert mit Standardskala '{self.current_scale}'")
    
    def set_time_scale(self, scale: str):
        """
        Setzt die Zeitskala
        
        Args:
            scale: Zeitskala (seconds, minutes, hours, days, weeks, months, years)
        """
        if scale in self.time_scales:
            self.current_scale = scale
            logger.info(f"Zeitskala auf '{scale}' gesetzt")
        else:
            valid_scales = ", ".join(self.time_scales.keys())
            raise ValueError(f"Ungültige Zeitskala: {scale}. Gültige Skalen: {valid_scales}")
    
    def get_current_scale_parameters(self) -> Dict[str, float]:
        """
        Gibt die Parameter der aktuellen Zeitskala zurück
        
        Returns:
            Parameter der aktuellen Zeitskala
        """
        return self.scale_parameters[self.current_scale]
    
    def convert_time_units(self, value: float, from_scale: str, to_scale: str) -> float:
        """
        Konvertiert Zeiteinheiten von einer Skala zu einer anderen
        
        Args:
            value: Zeitwert
            from_scale: Ausgangsskala
            to_scale: Zielskala
            
        Returns:
            Konvertierter Zeitwert
        """
        if from_scale not in self.time_scales or to_scale not in self.time_scales:
            valid_scales = ", ".join(self.time_scales.keys())
            raise ValueError(f"Ungültige Zeitskala. Gültige Skalen: {valid_scales}")
        
        # Konvertiere in Sekunden
        seconds = value * self.time_scales[from_scale]
        
        # Konvertiere von Sekunden in die Zielskala
        result = seconds / self.time_scales[to_scale]
        
        logger.debug(f"Konvertiere {value} {from_scale} zu {result} {to_scale}")
        return result
    
    def get_time_points(self, start_time: float, end_time: float, num_points: int = 10) -> List[float]:
        """
        Generiert Zeitpunkte zwischen Start- und Endzeit
        
        Args:
            start_time: Startzeit in Sekunden seit Epoch
            end_time: Endzeit in Sekunden seit Epoch
            num_points: Anzahl der zu generierenden Zeitpunkte
            
        Returns:
            Liste von Zeitpunkten
        """
        if num_points < 2:
            raise ValueError("Anzahl der Zeitpunkte muss mindestens 2 sein")
        
        return np.linspace(start_time, end_time, num_points).tolist()
    
    def format_time_point(self, time_point: float, format_str: str = None) -> str:
        """
        Formatiert einen Zeitpunkt als String
        
        Args:
            time_point: Zeitpunkt in Sekunden seit Epoch
            format_str: Formatierungsstring (Standard: abhängig von der aktuellen Skala)
            
        Returns:
            Formatierter Zeitpunkt
        """
        if format_str is None:
            # Wähle Standardformat basierend auf der aktuellen Skala
            if self.current_scale == "seconds":
                format_str = "%Y-%m-%d %H:%M:%S"
            elif self.current_scale == "minutes":
                format_str = "%Y-%m-%d %H:%M"
            elif self.current_scale == "hours":
                format_str = "%Y-%m-%d %H:%M"
            elif self.current_scale == "days":
                format_str = "%Y-%m-%d"
            elif self.current_scale == "weeks":
                format_str = "%Y-%m-%d (Woche %W)"
            elif self.current_scale == "months":
                format_str = "%Y-%m"
            else:  # years
                format_str = "%Y"
        
        dt = datetime.datetime.fromtimestamp(time_point)
        return dt.strftime(format_str)
    
    def scale_simulation(self, simulation_data: Dict[str, Any], target_scale: str) -> Dict[str, Any]:
        """
        Skaliert Simulationsdaten auf eine bestimmte Zeitskala
        
        Args:
            simulation_data: Simulationsdaten
            target_scale: Zielskala
            
        Returns:
            Skalierte Simulationsdaten
        """
        if target_scale not in self.time_scales:
            valid_scales = ", ".join(self.time_scales.keys())
            raise ValueError(f"Ungültige Zeitskala: {target_scale}. Gültige Skalen: {valid_scales}")
        
        # Kopiere die Simulationsdaten
        scaled_data = simulation_data.copy()
        
        # Extrahiere Zeitreihen
        time_series = scaled_data.get("time_series", {})
        
        # Skaliere jede Zeitreihe
        for series_name, series_data in time_series.items():
            if "timestamps" in series_data and "values" in series_data:
                timestamps = series_data["timestamps"]
                
                # Konvertiere Zeitstempel in die Zielskala
                if "time_scale" in series_data:
                    from_scale = series_data["time_scale"]
                    scaled_timestamps = [self.convert_time_units(ts, from_scale, target_scale) for ts in timestamps]
                    series_data["timestamps"] = scaled_timestamps
                    series_data["time_scale"] = target_scale
        
        # Aktualisiere Metadaten
        if "metadata" in scaled_data:
            scaled_data["metadata"]["time_scale"] = target_scale
        
        logger.info(f"Simulationsdaten skaliert von '{self.current_scale}' auf '{target_scale}'")
        return scaled_data
    
    def generate_time_series(self, start_time: float, duration: float, 
                           num_points: int = 100, scale: str = None) -> Dict[str, Any]:
        """
        Generiert eine Zeitreihe für Simulationen
        
        Args:
            start_time: Startzeit in Sekunden seit Epoch
            duration: Dauer in der angegebenen Skala
            num_points: Anzahl der zu generierenden Zeitpunkte
            scale: Zeitskala (Standard: aktuelle Skala)
            
        Returns:
            Zeitreihe mit Zeitstempeln
        """
        scale = scale or self.current_scale
        
        if scale not in self.time_scales:
            valid_scales = ", ".join(self.time_scales.keys())
            raise ValueError(f"Ungültige Zeitskala: {scale}. Gültige Skalen: {valid_scales}")
        
        # Konvertiere Dauer in Sekunden
        duration_seconds = duration * self.time_scales[scale]
        
        # Berechne Endzeit
        end_time = start_time + duration_seconds
        
        # Generiere Zeitpunkte
        timestamps = self.get_time_points(start_time, end_time, num_points)
        
        # Formatiere Zeitpunkte
        formatted_timestamps = [self.format_time_point(ts) for ts in timestamps]
        
        return {
            "timestamps": timestamps,
            "formatted_timestamps": formatted_timestamps,
            "time_scale": scale,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "num_points": num_points
        }


# Beispiel für die Verwendung der TimeScopeUnit
if __name__ == "__main__":
    # Konfiguriere Logging für Standalone-Ausführung
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Erstelle eine TimeScopeUnit
    time_scope = TimeScopeUnit()
    
    # Setze die Zeitskala
    time_scope.set_time_scale("days")
    
    # Konvertiere Zeiteinheiten
    days = 30
    hours = time_scope.convert_time_units(days, "days", "hours")
    print(f"{days} Tage = {hours} Stunden")
    
    # Generiere eine Zeitreihe
    current_time = time.time()
    time_series = time_scope.generate_time_series(current_time, 30, num_points=10)
    
    # Zeige die Zeitreihe
    print("Zeitreihe:")
    for i, (ts, formatted_ts) in enumerate(zip(time_series["timestamps"], time_series["formatted_timestamps"])):
        print(f"  {i}: {formatted_ts} ({ts})")
    
    # Skaliere Simulationsdaten
    simulation_data = {
        "metadata": {"time_scale": "days"},
        "time_series": {
            "temperature": {
                "timestamps": [1, 2, 3, 4, 5],
                "values": [20, 22, 25, 23, 21],
                "time_scale": "days"
            }
        }
    }
    
    scaled_data = time_scope.scale_simulation(simulation_data, "hours")
    print(f"Skalierte Daten: {json.dumps(scaled_data, indent=2)}")
