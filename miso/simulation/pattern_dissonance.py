#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Pattern Dissonance Scanner

Erkennt Abweichungen von erwarteten Mustern für die PRISM-Engine.
Identifiziert Anomalien und unerwartete Veränderungen in Datenströmen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import json
import time
from collections import deque
import uuid

# Importiere Basisklassen und -typen aus prism_base
from miso.simulation.prism_base import calculate_probability, sigmoid

# Konfiguriere Logging
logger = logging.getLogger("MISO.simulation.pattern_dissonance")


class PatternModel:
    """Modell für ein erwartetes Muster"""
    
    def __init__(self, pattern_id: str, pattern_data: Any, pattern_type: str = "sequence", 
               metadata: Dict[str, Any] = None):
        """
        Initialisiert ein Mustermodell
        
        Args:
            pattern_id: Eindeutige ID des Musters
            pattern_data: Daten des Musters
            pattern_type: Typ des Musters (sequence, value_range, threshold, etc.)
            metadata: Zusätzliche Metadaten zum Muster
        """
        self.pattern_id = pattern_id
        self.pattern_data = pattern_data
        self.pattern_type = pattern_type
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.last_updated = self.created_at
        self.match_history = deque(maxlen=100)  # Speichert die letzten 100 Übereinstimmungswerte
    
    def update(self, pattern_data: Any = None, metadata: Dict[str, Any] = None):
        """
        Aktualisiert das Mustermodell
        
        Args:
            pattern_data: Neue Musterdaten
            metadata: Neue Metadaten
        """
        if pattern_data is not None:
            self.pattern_data = pattern_data
        
        if metadata is not None:
            self.metadata.update(metadata)
        
        self.last_updated = time.time()
    
    def add_match_result(self, match_value: float, timestamp: float = None):
        """
        Fügt ein Übereinstimmungsergebnis zur Historie hinzu
        
        Args:
            match_value: Übereinstimmungswert (0.0 bis 1.0)
            timestamp: Zeitstempel (Standard: aktuelle Zeit)
        """
        timestamp = timestamp or time.time()
        self.match_history.append({"value": match_value, "timestamp": timestamp})
    
    def get_average_match(self, window: int = None) -> float:
        """
        Berechnet den durchschnittlichen Übereinstimmungswert
        
        Args:
            window: Anzahl der letzten Ergebnisse für die Berechnung (Standard: alle)
            
        Returns:
            Durchschnittlicher Übereinstimmungswert
        """
        if not self.match_history:
            return 0.0
        
        if window is not None and window > 0:
            history = list(self.match_history)[-window:]
        else:
            history = self.match_history
        
        if not history:
            return 0.0
        
        return sum(item["value"] for item in history) / len(history)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert das Mustermodell in ein Dictionary
        
        Returns:
            Dictionary-Repräsentation des Mustermodells
        """
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "metadata": self.metadata,
            "average_match": self.get_average_match()
        }


class PatternDissonanceScanner:
    """
    Erkennt Abweichungen von erwarteten Mustern
    """
    
    def __init__(self, dissonance_threshold: float = 0.3):
        """
        Initialisiert den PatternDissonanceScanner
        
        Args:
            dissonance_threshold: Schwellenwert für Dissonanz (0.0 bis 1.0)
        """
        self.expected_patterns = {}  # pattern_id -> PatternModel
        self.dissonance_threshold = dissonance_threshold
        self.dissonance_history = {}  # pattern_id -> Liste von Dissonanzwerten
        
        logger.info(f"PatternDissonanceScanner initialisiert mit Dissonanzschwellenwert {dissonance_threshold}")
    
    def register_pattern(self, pattern_id: str = None, pattern_data: Any = None, pattern_type: str = "sequence",
                       metadata: Dict[str, Any] = None) -> str:
        """
        Registriert ein erwartetes Muster
        
        Args:
            pattern_id: Eindeutige ID des Musters (wird generiert, wenn nicht angegeben)
            pattern_data: Daten des Musters
            pattern_type: Typ des Musters (sequence, value_range, threshold, etc.)
            metadata: Zusätzliche Metadaten zum Muster
            
        Returns:
            ID des registrierten Musters
        """
        # Generiere eine ID, wenn keine angegeben wurde
        if pattern_id is None:
            pattern_id = str(uuid.uuid4())
        
        # Prüfe, ob die ID bereits existiert
        if pattern_id in self.expected_patterns:
            logger.warning(f"Muster mit ID {pattern_id} existiert bereits und wird überschrieben")
        
        # Erstelle ein neues Mustermodell
        pattern = PatternModel(pattern_id, pattern_data, pattern_type, metadata)
        self.expected_patterns[pattern_id] = pattern
        self.dissonance_history[pattern_id] = deque(maxlen=100)
        
        logger.info(f"Muster {pattern_id} vom Typ {pattern_type} registriert")
        return pattern_id
    
    def update_pattern(self, pattern_id: str, pattern_data: Any = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Aktualisiert ein registriertes Muster
        
        Args:
            pattern_id: ID des Musters
            pattern_data: Neue Musterdaten
            metadata: Neue Metadaten
            
        Returns:
            True, wenn das Muster aktualisiert wurde, sonst False
        """
        if pattern_id not in self.expected_patterns:
            logger.error(f"Muster mit ID {pattern_id} existiert nicht")
            return False
        
        self.expected_patterns[pattern_id].update(pattern_data, metadata)
        logger.info(f"Muster {pattern_id} aktualisiert")
        return True
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """
        Entfernt ein registriertes Muster
        
        Args:
            pattern_id: ID des Musters
            
        Returns:
            True, wenn das Muster entfernt wurde, sonst False
        """
        if pattern_id not in self.expected_patterns:
            logger.error(f"Muster mit ID {pattern_id} existiert nicht")
            return False
        
        del self.expected_patterns[pattern_id]
        if pattern_id in self.dissonance_history:
            del self.dissonance_history[pattern_id]
        
        logger.info(f"Muster {pattern_id} entfernt")
        return True
    
    def scan_for_dissonance(self, current_data: Any, pattern_id: str) -> float:
        """
        Scannt nach Abweichungen vom erwarteten Muster
        
        Args:
            current_data: Aktuelle Daten
            pattern_id: ID des erwarteten Musters
            
        Returns:
            Dissonanzwert (0.0 bis 1.0, wobei 0.0 keine Dissonanz und 1.0 maximale Dissonanz bedeutet)
        """
        if pattern_id not in self.expected_patterns:
            logger.error(f"Muster mit ID {pattern_id} existiert nicht")
            return 1.0  # Maximale Dissonanz, wenn das Muster nicht existiert
        
        pattern = self.expected_patterns[pattern_id]
        pattern_type = pattern.pattern_type
        pattern_data = pattern.pattern_data
        
        # Berechne Dissonanz basierend auf dem Mustertyp
        dissonance = 0.0
        match_value = 0.0
        
        if pattern_type == "sequence":
            # Für Sequenzmuster: Berechne die Ähnlichkeit zwischen den Sequenzen
            dissonance = self._calculate_sequence_dissonance(current_data, pattern_data)
            match_value = 1.0 - dissonance
        
        elif pattern_type == "value_range":
            # Für Wertebereichsmuster: Prüfe, ob der Wert im erwarteten Bereich liegt
            dissonance = self._calculate_range_dissonance(current_data, pattern_data)
            match_value = 1.0 - dissonance
        
        elif pattern_type == "threshold":
            # Für Schwellenwertmuster: Prüfe, ob der Wert den Schwellenwert überschreitet
            dissonance = self._calculate_threshold_dissonance(current_data, pattern_data)
            match_value = 1.0 - dissonance
        
        else:
            logger.warning(f"Unbekannter Mustertyp: {pattern_type}")
            dissonance = 0.5  # Mittlere Dissonanz für unbekannte Mustertypen
            match_value = 0.5
        
        # Speichere das Dissonanzergebnis in der Historie
        self.dissonance_history[pattern_id].append({
            "timestamp": time.time(),
            "dissonance": dissonance
        })
        
        # Speichere das Übereinstimmungsergebnis im Mustermodell
        pattern.add_match_result(match_value)
        
        logger.debug(f"Dissonanz für Muster {pattern_id}: {dissonance:.4f}")
        return dissonance
    
    def _calculate_sequence_dissonance(self, current_sequence: List[Any], expected_sequence: List[Any]) -> float:
        """
        Berechnet die Dissonanz zwischen zwei Sequenzen
        
        Args:
            current_sequence: Aktuelle Sequenz
            expected_sequence: Erwartete Sequenz
            
        Returns:
            Dissonanzwert (0.0 bis 1.0)
        """
        # Konvertiere in NumPy-Arrays, falls nötig
        if not isinstance(current_sequence, np.ndarray):
            try:
                current_sequence = np.array(current_sequence, dtype=float)
            except (ValueError, TypeError):
                logger.error("Aktuelle Sequenz konnte nicht in ein NumPy-Array konvertiert werden")
                return 1.0
        
        if not isinstance(expected_sequence, np.ndarray):
            try:
                expected_sequence = np.array(expected_sequence, dtype=float)
            except (ValueError, TypeError):
                logger.error("Erwartete Sequenz konnte nicht in ein NumPy-Array konvertiert werden")
                return 1.0
        
        # Prüfe, ob die Sequenzen die gleiche Länge haben
        if len(current_sequence) != len(expected_sequence):
            # Wenn die Längen unterschiedlich sind, verwende die kürzere Länge
            min_length = min(len(current_sequence), len(expected_sequence))
            current_sequence = current_sequence[:min_length]
            expected_sequence = expected_sequence[:min_length]
            logger.warning(f"Sequenzlängen unterschiedlich, verwende die ersten {min_length} Elemente")
        
        # Berechne die mittlere quadratische Abweichung
        mse = np.mean((current_sequence - expected_sequence) ** 2)
        
        # Normalisiere die Dissonanz auf einen Wert zwischen 0 und 1
        # Verwende die Standardabweichung der erwarteten Sequenz als Normalisierungsfaktor
        std = np.std(expected_sequence)
        if std == 0:
            std = 1.0  # Vermeide Division durch Null
        
        dissonance = min(1.0, mse / (2 * std ** 2))
        return dissonance
    
    def _calculate_range_dissonance(self, current_value: float, expected_range: Tuple[float, float]) -> float:
        """
        Berechnet die Dissonanz für einen Wertebereich
        
        Args:
            current_value: Aktueller Wert
            expected_range: Erwarteter Wertebereich (min, max)
            
        Returns:
            Dissonanzwert (0.0 bis 1.0)
        """
        min_val, max_val = expected_range
        
        # Wenn der Wert im Bereich liegt, ist die Dissonanz 0
        if min_val <= current_value <= max_val:
            return 0.0
        
        # Berechne die Dissonanz basierend auf der Entfernung zum Bereich
        range_width = max_val - min_val
        if range_width == 0:
            range_width = 1.0  # Vermeide Division durch Null
        
        if current_value < min_val:
            distance = min_val - current_value
        else:  # current_value > max_val
            distance = current_value - max_val
        
        # Normalisiere die Dissonanz
        dissonance = min(1.0, distance / range_width)
        return dissonance
    
    def _calculate_threshold_dissonance(self, current_value: float, threshold_data: Dict[str, Any]) -> float:
        """
        Berechnet die Dissonanz für einen Schwellenwert
        
        Args:
            current_value: Aktueller Wert
            threshold_data: Schwellenwertdaten (threshold, direction, tolerance)
            
        Returns:
            Dissonanzwert (0.0 bis 1.0)
        """
        threshold = threshold_data.get("threshold", 0.0)
        direction = threshold_data.get("direction", "above")  # "above" oder "below"
        tolerance = threshold_data.get("tolerance", 0.1)
        
        # Prüfe, ob der Wert den Schwellenwert überschreitet
        if direction == "above":
            if current_value >= threshold:
                return 0.0  # Keine Dissonanz, wenn der Wert über dem Schwellenwert liegt
            else:
                distance = threshold - current_value
        else:  # direction == "below"
            if current_value <= threshold:
                return 0.0  # Keine Dissonanz, wenn der Wert unter dem Schwellenwert liegt
            else:
                distance = current_value - threshold
        
        # Normalisiere die Dissonanz
        dissonance = min(1.0, distance / (tolerance * abs(threshold)) if threshold != 0 else distance)
        return dissonance
    
    def get_dissonance_report(self) -> Dict[str, Any]:
        """
        Gibt einen Bericht über alle Musterabweichungen zurück
        
        Returns:
            Dissonanzbericht
        """
        report = {
            "timestamp": time.time(),
            "patterns": {},
            "overall_dissonance": 0.0,
            "dissonant_patterns": []
        }
        
        total_dissonance = 0.0
        pattern_count = 0
        
        for pattern_id, pattern in self.expected_patterns.items():
            # Berechne die durchschnittliche Dissonanz für dieses Muster
            dissonance_history = self.dissonance_history.get(pattern_id, [])
            if not dissonance_history:
                avg_dissonance = 0.0
            else:
                avg_dissonance = sum(item["dissonance"] for item in dissonance_history) / len(dissonance_history)
            
            # Füge das Muster zum Bericht hinzu
            report["patterns"][pattern_id] = {
                "pattern_type": pattern.pattern_type,
                "dissonance": avg_dissonance,
                "is_dissonant": avg_dissonance > self.dissonance_threshold,
                "metadata": pattern.metadata
            }
            
            # Aktualisiere die Gesamtdissonanz
            total_dissonance += avg_dissonance
            pattern_count += 1
            
            # Füge dissonante Muster zur Liste hinzu
            if avg_dissonance > self.dissonance_threshold:
                report["dissonant_patterns"].append(pattern_id)
        
        # Berechne die Gesamtdissonanz
        if pattern_count > 0:
            report["overall_dissonance"] = total_dissonance / pattern_count
        
        return report
    
    def get_pattern_info(self, pattern_id: str) -> Dict[str, Any]:
        """
        Gibt Informationen zu einem bestimmten Muster zurück
        
        Args:
            pattern_id: ID des Musters
            
        Returns:
            Musterinformationen
        """
        if pattern_id not in self.expected_patterns:
            logger.error(f"Muster mit ID {pattern_id} existiert nicht")
            return {"error": f"Muster mit ID {pattern_id} existiert nicht"}
        
        pattern = self.expected_patterns[pattern_id]
        
        # Berechne die durchschnittliche Dissonanz für dieses Muster
        dissonance_history = self.dissonance_history.get(pattern_id, [])
        if not dissonance_history:
            avg_dissonance = 0.0
            dissonance_trend = "stabil"
        else:
            avg_dissonance = sum(item["dissonance"] for item in dissonance_history) / len(dissonance_history)
            
            # Berechne den Dissonanztrend
            if len(dissonance_history) >= 10:
                recent_dissonance = sum(item["dissonance"] for item in list(dissonance_history)[-5:]) / 5
                older_dissonance = sum(item["dissonance"] for item in list(dissonance_history)[-10:-5]) / 5
                
                if recent_dissonance > older_dissonance * 1.1:
                    dissonance_trend = "steigend"
                elif recent_dissonance < older_dissonance * 0.9:
                    dissonance_trend = "fallend"
                else:
                    dissonance_trend = "stabil"
            else:
                dissonance_trend = "unbekannt"
        
        return {
            "pattern_id": pattern_id,
            "pattern_type": pattern.pattern_type,
            "created_at": pattern.created_at,
            "last_updated": pattern.last_updated,
            "metadata": pattern.metadata,
            "dissonance": {
                "current": avg_dissonance,
                "threshold": self.dissonance_threshold,
                "is_dissonant": avg_dissonance > self.dissonance_threshold,
                "trend": dissonance_trend
            },
            "match_history": {
                "average": pattern.get_average_match(),
                "recent": pattern.get_average_match(10)
            }
        }


# Beispiel für die Verwendung des PatternDissonanceScanners
if __name__ == "__main__":
    # Konfiguriere Logging für Standalone-Ausführung
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Erstelle einen PatternDissonanceScanner
    scanner = PatternDissonanceScanner(dissonance_threshold=0.3)
    
    # Registriere ein Sequenzmuster
    sequence_pattern = [1.0, 2.0, 3.0, 4.0, 5.0]
    sequence_id = scanner.register_pattern(
        pattern_data=sequence_pattern,
        pattern_type="sequence",
        metadata={"name": "Steigende Sequenz", "description": "Eine linear steigende Sequenz"}
    )
    
    # Registriere ein Wertebereichsmuster
    range_id = scanner.register_pattern(
        pattern_data=(10.0, 20.0),
        pattern_type="value_range",
        metadata={"name": "Normalbereich", "description": "Erwarteter Wertebereich für normale Bedingungen"}
    )
    
    # Registriere ein Schwellenwertmuster
    threshold_id = scanner.register_pattern(
        pattern_data={"threshold": 30.0, "direction": "below", "tolerance": 0.2},
        pattern_type="threshold",
        metadata={"name": "Maximalschwelle", "description": "Wert sollte unter diesem Schwellenwert bleiben"}
    )
    
    # Teste die Dissonanzerkennung
    print(f"Sequenzmuster-Dissonanz: {scanner.scan_for_dissonance([1.1, 2.2, 3.3, 4.4, 5.5], sequence_id):.4f}")
    print(f"Wertebereichsmuster-Dissonanz: {scanner.scan_for_dissonance(15.0, range_id):.4f}")
    print(f"Schwellenwertmuster-Dissonanz: {scanner.scan_for_dissonance(25.0, threshold_id):.4f}")
    
    # Zeige den Dissonanzbericht
    report = scanner.get_dissonance_report()
    print(f"Dissonanzbericht: {json.dumps(report, indent=2)}")
    
    # Zeige Informationen zu einem bestimmten Muster
    pattern_info = scanner.get_pattern_info(sequence_id)
    print(f"Musterinformationen: {json.dumps(pattern_info, indent=2)}")
