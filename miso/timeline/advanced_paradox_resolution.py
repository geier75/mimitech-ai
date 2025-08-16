#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Erweiterte Paradoxauflösung

Diese Datei implementiert die erweiterte Paradoxauflösung für MISO,
die in der Bedarfsanalyse als höchste Priorität identifiziert wurde.
Sie ermöglicht die Erkennung, Klassifizierung und Auflösung komplexer Paradoxien
sowie präventive Maßnahmen zur Vermeidung von Paradoxien.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
from enum import Enum, auto

# Füge Hauptverzeichnis zum Pfad hinzu für Module, die dort liegen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importiere ECHO-PRIME Komponenten
from miso.timeline.echo_prime import Timeline, TimeNode, TriggerLevel, TimelineType
from miso.timeline.echo_prime_controller import EchoPrimeController
from miso.timeline.qtm_modulator import QTM_Modulator, QuantumTimeEffect, QuantumState
from miso.timeline.temporal_integrity_guard import TemporalIntegrityGuard, ParadoxDetection, ParadoxType

# Importiere Q-LOGIK Integration
from miso.logic.qlogik_integration import QLOGIKECHOPrimeIntegration

# Importiere Module aus dem Hauptverzeichnis
from timeline_feedback_loop import TimelineFeedbackLoop, StrategicRecommendation

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.Timeline.AdvancedParadoxResolution")

# ZTM-Schnittstelle
ZTM_ENABLED = True
ZTM_LOG_PATH = os.path.join(os.path.dirname(__file__), "ztm_logs")

def ztm_log(message: str, level: str = "INFO", module: str = "AdvancedParadoxResolution"):
    """
    Loggt eine Nachricht in die ZTM-Logs
    
    Args:
        message: Die zu loggende Nachricht
        level: Log-Level (INFO, WARNING, ERROR, CRITICAL)
        module: Modul, das die Nachricht loggt
    """
    if not ZTM_ENABLED:
        return
    
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "module": module,
        "message": message
    }
    
    try:
        os.makedirs(ZTM_LOG_PATH, exist_ok=True)
        log_file = os.path.join(ZTM_LOG_PATH, f"ztm_paradox_resolution_{datetime.now().strftime('%Y%m%d')}.log")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.warning(f"Konnte ZTM-Log nicht schreiben: {e}")


# Datenmodelle für die erweiterte Paradoxauflösung

class EnhancedParadoxType(Enum):
    """
    Erweiterte Paradoxtypen
    """
    # Grundlegende Paradoxtypen
    GRANDFATHER = 1
    BOOTSTRAP = 2
    PREDESTINATION = 3
    ONTOLOGICAL = 4
    
    # Erweiterte Paradoxtypen
    TEMPORAL_LOOP = 5
    CAUSAL_VIOLATION = 6
    INFORMATION_PARADOX = 7
    QUANTUM_PARADOX = 8
    MULTI_TIMELINE_PARADOX = 9
    SELF_CONSISTENCY_VIOLATION = 10


class ParadoxSeverity(Enum):
    """
    Schweregrad eines Paradoxes
    """
    NEGLIGIBLE = 1  # Vernachlässigbare Auswirkungen
    MINOR = 2       # Geringfügige Auswirkungen
    MODERATE = 3    # Mäßige Auswirkungen
    MAJOR = 4       # Erhebliche Auswirkungen
    CRITICAL = 5    # Kritische Auswirkungen


class ResolutionStrategy(Enum):
    """
    Strategien zur Auflösung von Paradoxien
    """
    TIMELINE_ADJUSTMENT = 1      # Anpassung der Zeitlinie
    EVENT_MODIFICATION = 2       # Modifikation des auslösenden Ereignisses
    CAUSAL_REROUTING = 3         # Umleitung der Kausalität
    QUANTUM_SUPERPOSITION = 4    # Anwendung von Quantensuperposition
    TEMPORAL_ISOLATION = 5       # Isolierung des Paradoxes
    PARADOX_ABSORPTION = 6       # Absorption des Paradoxes in die Zeitlinie


class ParadoxInstance:
    """
    Instanz eines Paradoxes
    """
    def __init__(self, 
                 timeline_id: str,
                 affected_node_ids: List[str],
                 paradox_type: EnhancedParadoxType,
                 severity: ParadoxSeverity,
                 description: str,
                 detection_time: float = None):
        """
        Initialisiert eine Paradoxinstanz
        
        Args:
            timeline_id: ID der betroffenen Zeitlinie
            affected_node_ids: IDs der betroffenen Zeitknoten
            paradox_type: Typ des Paradoxes
            severity: Schweregrad des Paradoxes
            description: Beschreibung des Paradoxes
            detection_time: Zeitpunkt der Erkennung (Standard: aktuelle Zeit)
        """
        self.id = str(uuid.uuid4())
        self.timeline_id = timeline_id
        self.affected_node_ids = affected_node_ids
        self.paradox_type = paradox_type
        self.severity = severity
        self.description = description
        self.detection_time = detection_time or time.time()
        self.resolution_attempts = []
        
        ztm_log(f"Paradoxinstanz erstellt: {self.id} (Typ: {paradox_type.name}, Schweregrad: {severity.name})")


class PotentialParadox:
    """
    Potentielles Paradox (Frühwarnung)
    """
    def __init__(self,
                 timeline_id: str,
                 node_ids: List[str],
                 probability: float,
                 potential_type: EnhancedParadoxType,
                 description: str):
        """
        Initialisiert ein potentielles Paradox
        
        Args:
            timeline_id: ID der betroffenen Zeitlinie
            node_ids: IDs der potentiell betroffenen Zeitknoten
            probability: Wahrscheinlichkeit des Auftretens (0.0-1.0)
            potential_type: Potentieller Typ des Paradoxes
            description: Beschreibung des potentiellen Paradoxes
        """
        self.id = str(uuid.uuid4())
        self.timeline_id = timeline_id
        self.node_ids = node_ids
        self.probability = probability
        self.potential_type = potential_type
        self.description = description
        self.detection_time = time.time()
        
        ztm_log(f"Potentielles Paradox erkannt: {self.id} (Typ: {potential_type.name}, Wahrscheinlichkeit: {probability:.2f})")


class ResolutionOption:
    """
    Option zur Auflösung eines Paradoxes
    """
    def __init__(self,
                 strategy: ResolutionStrategy,
                 confidence: float,
                 impact: float,
                 description: str,
                 side_effects: List[str] = None):
        """
        Initialisiert eine Auflösungsoption
        
        Args:
            strategy: Auflösungsstrategie
            confidence: Konfidenz in die Erfolgswahrscheinlichkeit (0.0-1.0)
            impact: Auswirkung auf die Zeitlinie (0.0-1.0)
            description: Beschreibung der Auflösungsoption
            side_effects: Liste möglicher Nebenwirkungen
        """
        self.id = str(uuid.uuid4())
        self.strategy = strategy
        self.confidence = confidence
        self.impact = impact
        self.description = description
        self.side_effects = side_effects or []
        
        ztm_log(f"Auflösungsoption erstellt: {self.id} (Strategie: {strategy.name}, Konfidenz: {confidence:.2f})")


class ResolutionResult:
    """
    Ergebnis einer Paradoxauflösung
    """
    def __init__(self,
                 paradox_id: str,
                 resolution_option_id: str,
                 success: bool,
                 timeline_id: str,
                 modified_node_ids: List[str] = None,
                 new_timeline_id: str = None,
                 description: str = None):
        """
        Initialisiert ein Auflösungsergebnis
        
        Args:
            paradox_id: ID des aufgelösten Paradoxes
            resolution_option_id: ID der verwendeten Auflösungsoption
            success: Ob die Auflösung erfolgreich war
            timeline_id: ID der betroffenen Zeitlinie
            modified_node_ids: IDs der modifizierten Zeitknoten
            new_timeline_id: ID einer neu erstellten Zeitlinie (falls zutreffend)
            description: Beschreibung des Ergebnisses
        """
        self.id = str(uuid.uuid4())
        self.paradox_id = paradox_id
        self.resolution_option_id = resolution_option_id
        self.success = success
        self.timeline_id = timeline_id
        self.modified_node_ids = modified_node_ids or []
        self.new_timeline_id = new_timeline_id
        self.description = description or ("Erfolgreiche Auflösung" if success else "Fehlgeschlagene Auflösung")
        self.timestamp = time.time()
        
        log_level = "INFO" if success else "WARNING"
        ztm_log(f"Paradoxauflösung: {self.description} (Paradox: {paradox_id})", level=log_level)


class ParadoxHierarchy:
    """
    Hierarchische Kategorisierung von Paradoxien
    """
    def __init__(self,
                 primary_type: EnhancedParadoxType,
                 secondary_types: List[EnhancedParadoxType] = None,
                 related_types: List[EnhancedParadoxType] = None):
        """
        Initialisiert eine hierarchische Paradoxkategorisierung
        
        Args:
            primary_type: Primärer Paradoxtyp
            secondary_types: Sekundäre Paradoxtypen
            related_types: Verwandte Paradoxtypen
        """
        self.primary_type = primary_type
        self.secondary_types = secondary_types or []
        self.related_types = related_types or []


class EnhancedParadoxDetector:
    """
    Erweiterte Paradoxerkennung
    """
    
    def __init__(self, echo_prime_controller):
        """
        Initialisiert den erweiterten Paradoxdetektor
        
        Args:
            echo_prime_controller: ECHO-PRIME Controller-Instanz
        """
        self.echo_prime = echo_prime_controller
        self.integrity_guard = echo_prime_controller.integrity_guard
        self.qlogik_integration = None
        
        # Versuche, die Q-LOGIK-Integration zu laden
        try:
            from miso.logic.qlogik_integration import QLOGIKECHOPrimeIntegration
            if hasattr(echo_prime_controller, 'qlogik_integration'):
                self.qlogik_integration = echo_prime_controller.qlogik_integration
        except ImportError:
            logger.warning("Q-LOGIK-Integration nicht verfügbar")
        
        ztm_log("EnhancedParadoxDetector initialisiert")
    
    def detect_complex_paradoxes(self, timeline_id: str) -> List[ParadoxInstance]:
        """
        Erkennt komplexe Paradoxien in einer Zeitlinie
        
        Args:
            timeline_id: ID der zu prüfenden Zeitlinie
            
        Returns:
            Liste erkannter Paradoxinstanzen
        """
        timeline = self.echo_prime.get_timeline(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} existiert nicht")
            return []
        
        # Basisparadoxien von TemporalIntegrityGuard erhalten
        base_paradoxes = self.integrity_guard.detect_paradoxes(timeline_id)
        
        # Komplexe Paradoxien erkennen
        complex_paradoxes = []
        
        # Prüfe auf temporale Schleifen
        if self._has_temporal_loop(timeline):
            affected_nodes = self._find_temporal_loop_nodes(timeline)
            complex_paradoxes.append(ParadoxInstance(
                timeline_id=timeline_id,
                affected_node_ids=[node.id for node in affected_nodes],
                paradox_type=EnhancedParadoxType.TEMPORAL_LOOP,
                severity=ParadoxSeverity.MAJOR,
                description="Temporale Schleife erkannt: Ereignisse bilden einen geschlossenen Kausalitätskreis"
            ))
        
        # Prüfe auf Kausalitätsverletzungen
        if self._has_causal_violation(timeline):
            affected_nodes = self._find_causal_violation_nodes(timeline)
            complex_paradoxes.append(ParadoxInstance(
                timeline_id=timeline_id,
                affected_node_ids=[node.id for node in affected_nodes],
                paradox_type=EnhancedParadoxType.CAUSAL_VIOLATION,
                severity=ParadoxSeverity.CRITICAL,
                description="Kausalitätsverletzung erkannt: Effekt tritt vor Ursache auf"
            ))
        
        # Prüfe auf Informationsparadoxien
        if self._has_information_paradox(timeline):
            affected_nodes = self._find_information_paradox_nodes(timeline)
            complex_paradoxes.append(ParadoxInstance(
                timeline_id=timeline_id,
                affected_node_ids=[node.id for node in affected_nodes],
                paradox_type=EnhancedParadoxType.INFORMATION_PARADOX,
                severity=ParadoxSeverity.MODERATE,
                description="Informationsparadox erkannt: Informationsquelle ist zirkulär"
            ))
        
        # Prüfe auf Quantenparadoxien, wenn QTM-Modulator verfügbar
        if hasattr(self.echo_prime, 'qtm_modulator') and self._has_quantum_paradox(timeline):
            affected_nodes = self._find_quantum_paradox_nodes(timeline)
            complex_paradoxes.append(ParadoxInstance(
                timeline_id=timeline_id,
                affected_node_ids=[node.id for node in affected_nodes],
                paradox_type=EnhancedParadoxType.QUANTUM_PARADOX,
                severity=ParadoxSeverity.MAJOR,
                description="Quantenparadox erkannt: Inkonsistente Quantenzustände"
            ))
        
        return complex_paradoxes
    
    def evaluate_paradox_probability(self, timeline_id: str, node_id: str) -> float:
        """
        Berechnet die Wahrscheinlichkeit eines Paradoxes für einen Zeitknoten
        
        Args:
            timeline_id: ID der Zeitlinie
            node_id: ID des Zeitknotens
            
        Returns:
            Wahrscheinlichkeit eines Paradoxes (0.0-1.0)
        """
        timeline = self.echo_prime.get_timeline(timeline_id)
        if not timeline or node_id not in timeline.nodes:
            return 0.0
        
        node = timeline.nodes[node_id]
        
        # Faktoren für die Paradoxwahrscheinlichkeit
        factors = {
            "node_probability": node.probability,
            "trigger_level": node.trigger_level.value / 3.0,  # Normalisieren auf 0-1
            "timeline_complexity": min(len(timeline.nodes) / 100.0, 1.0),  # Normalisieren
            "causal_distance": self._calculate_causal_distance(timeline, node),
            "quantum_state": self._get_quantum_state_factor(timeline_id)
        }
        
        # Gewichtete Summe der Faktoren
        weights = {
            "node_probability": 0.2,
            "trigger_level": 0.15,
            "timeline_complexity": 0.25,
            "causal_distance": 0.3,
            "quantum_state": 0.1
        }
        
        probability = sum(factors[k] * weights[k] for k in factors)
        
        # Begrenzen auf 0.0-1.0
        return max(0.0, min(1.0, probability))
    
    def detect_potential_paradoxes(self, timeline_id: str) -> List[PotentialParadox]:
        """
        Frühzeitige Erkennung potentieller Paradoxien
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            Liste potentieller Paradoxien
        """
        timeline = self.echo_prime.get_timeline(timeline_id)
        if not timeline:
            return []
        
        potential_paradoxes = []
        
        # Prüfe jeden Knoten auf Paradoxpotential
        for node_id, node in timeline.nodes.items():
            probability = self.evaluate_paradox_probability(timeline_id, node_id)
            
            # Nur Knoten mit signifikanter Wahrscheinlichkeit berücksichtigen
            if probability > 0.6:
                # Bestimme den wahrscheinlichsten Paradoxtyp
                paradox_type = self._predict_paradox_type(timeline, node)
                
                potential_paradoxes.append(PotentialParadox(
                    timeline_id=timeline_id,
                    node_ids=[node_id],
                    probability=probability,
                    potential_type=paradox_type,
                    description=f"Potentielles {paradox_type.name}-Paradox mit Wahrscheinlichkeit {probability:.2f}"
                ))
        
        return potential_paradoxes
    
    # Hilfsmethoden für die Paradoxerkennung
    
    def _has_temporal_loop(self, timeline) -> bool:
        """
        Prüft, ob eine Zeitlinie eine temporale Schleife enthält
        
        Args:
            timeline: Die zu prüfende Zeitlinie
            
        Returns:
            True, wenn eine temporale Schleife erkannt wurde
        """
        # Implementierung mit Zykluserkennung im Kausalitätsgraphen
        nodes = list(timeline.nodes.values())
        visited = set()
        path = set()
        
        def dfs_cycle_check(node):
            if node.id in path:
                return True
            if node.id in visited:
                return False
            
            visited.add(node.id)
            path.add(node.id)
            
            for child_id in node.children:
                if child_id in timeline.nodes and dfs_cycle_check(timeline.nodes[child_id]):
                    return True
            
            path.remove(node.id)
            return False
        
        for node in nodes:
            if dfs_cycle_check(node):
                return True
        
        return False
    
    def _find_temporal_loop_nodes(self, timeline) -> List:
        """
        Findet Knoten, die Teil einer temporalen Schleife sind
        
        Args:
            timeline: Die zu prüfende Zeitlinie
            
        Returns:
            Liste der Knoten in temporalen Schleifen
        """
        # Implementierung mit Tarjan's Algorithmus für stark zusammenhängende Komponenten
        nodes = list(timeline.nodes.values())
        index_counter = [0]
        index = {}
        lowlink = {}
        onstack = set()
        stack = []
        result = []
        
        def strongconnect(node):
            index[node.id] = index_counter[0]
            lowlink[node.id] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            onstack.add(node.id)
            
            for child_id in node.children:
                if child_id in timeline.nodes:
                    child = timeline.nodes[child_id]
                    if child.id not in index:
                        strongconnect(child)
                        lowlink[node.id] = min(lowlink[node.id], lowlink[child.id])
                    elif child.id in onstack:
                        lowlink[node.id] = min(lowlink[node.id], index[child.id])
            
            if lowlink[node.id] == index[node.id]:
                component = []
                while True:
                    w = stack.pop()
                    onstack.remove(w.id)
                    component.append(w)
                    if w.id == node.id:
                        break
                if len(component) > 1:
                    result.extend(component)
        
        for node in nodes:
            if node.id not in index:
                strongconnect(node)
        
        return result
    
    def _has_causal_violation(self, timeline) -> bool:
        """
        Prüft, ob eine Zeitlinie eine Kausalitätsverletzung enthält
        
        Args:
            timeline: Die zu prüfende Zeitlinie
            
        Returns:
            True, wenn eine Kausalitätsverletzung erkannt wurde
        """
        # Implementierung mit topologischer Sortierung und Zeitstempelprüfung
        nodes = list(timeline.nodes.values())
        
        for node in nodes:
            for child_id in node.children:
                if child_id in timeline.nodes:
                    child = timeline.nodes[child_id]
                    # Wenn ein Kind einen früheren Zeitstempel hat als sein Elternteil,
                    # liegt eine Kausalitätsverletzung vor
                    if hasattr(child, 'timestamp') and hasattr(node, 'timestamp'):
                        if child.timestamp < node.timestamp:
                            return True
        
        return False
    
    def _find_causal_violation_nodes(self, timeline) -> List:
        """
        Findet Knoten mit Kausalitätsverletzungen
        
        Args:
            timeline: Die zu prüfende Zeitlinie
            
        Returns:
            Liste der Knoten mit Kausalitätsverletzungen
        """
        nodes = list(timeline.nodes.values())
        violation_nodes = []
        
        for node in nodes:
            for child_id in node.children:
                if child_id in timeline.nodes:
                    child = timeline.nodes[child_id]
                    if hasattr(child, 'timestamp') and hasattr(node, 'timestamp'):
                        if child.timestamp < node.timestamp:
                            violation_nodes.extend([node, child])
        
        return list(set(violation_nodes))  # Entferne Duplikate
    
    def _has_information_paradox(self, timeline) -> bool:
        """
        Prüft, ob eine Zeitlinie ein Informationsparadox enthält
        
        Args:
            timeline: Die zu prüfende Zeitlinie
            
        Returns:
            True, wenn ein Informationsparadox erkannt wurde
        """
        # Vereinfachte Implementierung: Prüfe auf Knoten mit zirkulärer Informationsquelle
        nodes = list(timeline.nodes.values())
        
        for node in nodes:
            if hasattr(node, 'metadata') and isinstance(node.metadata, dict):
                if 'information_source' in node.metadata:
                    source_id = node.metadata['information_source']
                    if source_id in timeline.nodes:
                        source = timeline.nodes[source_id]
                        # Prüfe, ob die Quelle direkt oder indirekt vom aktuellen Knoten abhängt
                        if self._is_dependent_on(timeline, source, node):
                            return True
        
        return False
    
    def _find_information_paradox_nodes(self, timeline) -> List:
        """
        Findet Knoten mit Informationsparadoxien
        
        Args:
            timeline: Die zu prüfende Zeitlinie
            
        Returns:
            Liste der Knoten mit Informationsparadoxien
        """
        nodes = list(timeline.nodes.values())
        paradox_nodes = []
        
        for node in nodes:
            if hasattr(node, 'metadata') and isinstance(node.metadata, dict):
                if 'information_source' in node.metadata:
                    source_id = node.metadata['information_source']
                    if source_id in timeline.nodes:
                        source = timeline.nodes[source_id]
                        if self._is_dependent_on(timeline, source, node):
                            paradox_nodes.extend([node, source])
        
        return list(set(paradox_nodes))  # Entferne Duplikate
    
    def _has_quantum_paradox(self, timeline) -> bool:
        """
        Prüft, ob eine Zeitlinie ein Quantenparadox enthält
        
        Args:
            timeline: Die zu prüfende Zeitlinie
            
        Returns:
            True, wenn ein Quantenparadox erkannt wurde
        """
        # Prüfe auf inkonsistente Quantenzustände zwischen verbundenen Knoten
        if not hasattr(self.echo_prime, 'qtm_modulator'):
            return False
        
        qtm_modulator = self.echo_prime.qtm_modulator
        nodes = list(timeline.nodes.values())
        
        for node in nodes:
            if hasattr(node, 'quantum_state') and node.quantum_state:
                for child_id in node.children:
                    if child_id in timeline.nodes:
                        child = timeline.nodes[child_id]
                        if hasattr(child, 'quantum_state') and child.quantum_state:
                            # Prüfe auf Quanteninkonsistenz
                            if not qtm_modulator.are_states_compatible(node.quantum_state, child.quantum_state):
                                return True
        
        return False
    
    def _find_quantum_paradox_nodes(self, timeline) -> List:
        """
        Findet Knoten mit Quantenparadoxien
        
        Args:
            timeline: Die zu prüfende Zeitlinie
            
        Returns:
            Liste der Knoten mit Quantenparadoxien
        """
        if not hasattr(self.echo_prime, 'qtm_modulator'):
            return []
        
        qtm_modulator = self.echo_prime.qtm_modulator
        nodes = list(timeline.nodes.values())
        paradox_nodes = []
        
        for node in nodes:
            if hasattr(node, 'quantum_state') and node.quantum_state:
                for child_id in node.children:
                    if child_id in timeline.nodes:
                        child = timeline.nodes[child_id]
                        if hasattr(child, 'quantum_state') and child.quantum_state:
                            if not qtm_modulator.are_states_compatible(node.quantum_state, child.quantum_state):
                                paradox_nodes.extend([node, child])
        
        return list(set(paradox_nodes))  # Entferne Duplikate
    
    def _calculate_causal_distance(self, timeline, node) -> float:
        """
        Berechnet die kausale Distanz eines Knotens
        
        Args:
            timeline: Die Zeitlinie
            node: Der zu prüfende Knoten
            
        Returns:
            Kausale Distanz (0.0-1.0)
        """
        # Vereinfachte Implementierung: Berechne die durchschnittliche Pfadlänge zu anderen Knoten
        if not node.children and not node.parents:
            return 1.0  # Isolierte Knoten haben maximale kausale Distanz
        
        total_distance = 0
        visited = set()
        queue = [(node.id, 0)]  # (Knoten-ID, Distanz)
        
        while queue:
            current_id, distance = queue.pop(0)
            if current_id in visited:
                continue
            
            visited.add(current_id)
            total_distance += distance
            
            current = timeline.nodes[current_id]
            for child_id in current.children:
                if child_id in timeline.nodes and child_id not in visited:
                    queue.append((child_id, distance + 1))
            
            for parent_id in current.parents:
                if parent_id in timeline.nodes and parent_id not in visited:
                    queue.append((parent_id, distance + 1))
        
        # Normalisiere auf 0.0-1.0 (höhere Werte bedeuten größere kausale Distanz)
        avg_distance = total_distance / max(1, len(visited) - 1)
        return min(1.0, avg_distance / 10.0)  # Normalisieren mit Annahme einer maximalen Distanz von 10
    
    def _get_quantum_state_factor(self, timeline_id) -> float:
        """
        Berechnet einen Faktor basierend auf dem Quantenzustand der Zeitlinie
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            Quantenzustandsfaktor (0.0-1.0)
        """
        if not hasattr(self.echo_prime, 'qtm_modulator'):
            return 0.5  # Standardwert, wenn kein QTM-Modulator verfügbar ist
        
        qtm_modulator = self.echo_prime.qtm_modulator
        
        try:
            # Versuche, den Quantenzustand der Zeitlinie zu erhalten
            quantum_state = qtm_modulator.get_timeline_quantum_state(timeline_id)
            if quantum_state:
                # Berechne einen Faktor basierend auf der Komplexität des Quantenzustands
                if hasattr(quantum_state, 'complexity'):
                    return min(1.0, quantum_state.complexity / 10.0)
                elif hasattr(quantum_state, 'entanglement_level'):
                    return min(1.0, quantum_state.entanglement_level / 5.0)
                else:
                    return 0.5
        except Exception as e:
            logger.warning(f"Fehler bei der Berechnung des Quantenzustandsfaktors: {e}")
        
        return 0.5
    
    def _predict_paradox_type(self, timeline, node) -> EnhancedParadoxType:
        """
        Sagt den wahrscheinlichsten Paradoxtyp für einen Knoten voraus
        
        Args:
            timeline: Die Zeitlinie
            node: Der zu prüfende Knoten
            
        Returns:
            Wahrscheinlichster Paradoxtyp
        """
        # Einfache heuristische Vorhersage basierend auf Knoteneigenschaften
        
        # Prüfe auf temporale Schleifen
        if self._is_in_cycle(timeline, node):
            return EnhancedParadoxType.TEMPORAL_LOOP
        
        # Prüfe auf Kausalitätsverletzungen
        if self._has_causal_violation_potential(timeline, node):
            return EnhancedParadoxType.CAUSAL_VIOLATION
        
        # Prüfe auf Informationsparadoxien
        if hasattr(node, 'metadata') and isinstance(node.metadata, dict):
            if 'information_source' in node.metadata:
                return EnhancedParadoxType.INFORMATION_PARADOX
        
        # Prüfe auf Quantenparadoxien
        if hasattr(node, 'quantum_state') and node.quantum_state:
            return EnhancedParadoxType.QUANTUM_PARADOX
        
        # Standardtyp, wenn keine spezifischen Indikatoren gefunden wurden
        return EnhancedParadoxType.BOOTSTRAP
    
    def _is_dependent_on(self, timeline, node1, node2) -> bool:
        """
        Prüft, ob node1 direkt oder indirekt von node2 abhängt
        
        Args:
            timeline: Die Zeitlinie
            node1: Erster Knoten
            node2: Zweiter Knoten
            
        Returns:
            True, wenn node1 von node2 abhängt
        """
        visited = set()
        queue = [node2.id]
        
        while queue:
            current_id = queue.pop(0)
            if current_id == node1.id:
                return True
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            current = timeline.nodes[current_id]
            
            for child_id in current.children:
                if child_id in timeline.nodes and child_id not in visited:
                    queue.append(child_id)
        
        return False
    
    def _is_in_cycle(self, timeline, node) -> bool:
        """
        Prüft, ob ein Knoten Teil eines Zyklus ist
        
        Args:
            timeline: Die Zeitlinie
            node: Der zu prüfende Knoten
            
        Returns:
            True, wenn der Knoten Teil eines Zyklus ist
        """
        visited = set()
        path = set()
        
        def dfs_cycle_check(current_id):
            if current_id in path:
                return True
            if current_id in visited:
                return False
            
            visited.add(current_id)
            path.add(current_id)
            
            current = timeline.nodes[current_id]
            for child_id in current.children:
                if child_id in timeline.nodes and dfs_cycle_check(child_id):
                    return True
            
            path.remove(current_id)
            return False
        
        return dfs_cycle_check(node.id)
    
    def _has_causal_violation_potential(self, timeline, node) -> bool:
        """
        Prüft, ob ein Knoten das Potential für eine Kausalitätsverletzung hat
        
        Args:
            timeline: Die Zeitlinie
            node: Der zu prüfende Knoten
            
        Returns:
            True, wenn der Knoten Potential für eine Kausalitätsverletzung hat
        """
        if not hasattr(node, 'timestamp'):
            return False
        
        # Prüfe, ob der Knoten Kinder mit früheren Zeitstempeln hat
        for child_id in node.children:
            if child_id in timeline.nodes:
                child = timeline.nodes[child_id]
                if hasattr(child, 'timestamp') and child.timestamp < node.timestamp:
                    return True
        
        # Prüfe, ob der Knoten Eltern mit späteren Zeitstempeln hat
        for parent_id in node.parents:
            if parent_id in timeline.nodes:
                parent = timeline.nodes[parent_id]
                if hasattr(parent, 'timestamp') and parent.timestamp > node.timestamp:
                    return True
        
        return False


class ParadoxClassifier:
    """
    Klassifizierung von Paradoxien
    """
    
    def __init__(self):
        """
        Initialisiert den Paradoxklassifizierer
        """
        self.type_patterns = {
            EnhancedParadoxType.GRANDFATHER: {
                "pattern": "ancestor_modification",
                "keywords": ["vorfahre", "vergangenheit", "änderung", "existenz"]
            },
            EnhancedParadoxType.BOOTSTRAP: {
                "pattern": "self_creation",
                "keywords": ["selbst", "erschaffung", "ursprung", "quelle"]
            },
            EnhancedParadoxType.PREDESTINATION: {
                "pattern": "predetermined_outcome",
                "keywords": ["vorbestimmt", "schicksal", "unvermeidlich", "prophezeiung"]
            },
            EnhancedParadoxType.ONTOLOGICAL: {
                "pattern": "origin_question",
                "keywords": ["herkunft", "existenz", "ursprung", "sein"]
            },
            EnhancedParadoxType.TEMPORAL_LOOP: {
                "pattern": "time_loop",
                "keywords": ["schleife", "wiederholung", "zyklus", "endlos"]
            },
            EnhancedParadoxType.CAUSAL_VIOLATION: {
                "pattern": "causality_breach",
                "keywords": ["ursache", "wirkung", "verletzung", "kausalität"]
            },
            EnhancedParadoxType.INFORMATION_PARADOX: {
                "pattern": "circular_information",
                "keywords": ["information", "wissen", "zirkulär", "quelle"]
            },
            EnhancedParadoxType.QUANTUM_PARADOX: {
                "pattern": "quantum_inconsistency",
                "keywords": ["quanten", "zustand", "superposition", "messung"]
            },
            EnhancedParadoxType.MULTI_TIMELINE_PARADOX: {
                "pattern": "timeline_conflict",
                "keywords": ["zeitlinie", "konflikt", "parallel", "divergenz"]
            },
            EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION: {
                "pattern": "consistency_breach",
                "keywords": ["konsistenz", "widerspruch", "logik", "integrität"]
            }
        }
        
        ztm_log("ParadoxClassifier initialisiert")
    
    def classify_paradox(self, paradox_instance: ParadoxInstance) -> EnhancedParadoxType:
        """
        Detaillierte Klassifizierung von Paradoxien
        
        Args:
            paradox_instance: Die zu klassifizierende Paradoxinstanz
            
        Returns:
            Klassifizierter Paradoxtyp
        """
        # Wenn bereits ein Typ zugewiesen wurde, diesen verwenden
        if hasattr(paradox_instance, "paradox_type") and paradox_instance.paradox_type:
            return paradox_instance.paradox_type
        
        # Analyse der Beschreibung und betroffenen Knoten
        description = paradox_instance.description.lower()
        
        # Suche nach Mustern in der Beschreibung
        best_match = None
        best_score = 0
        
        for paradox_type, pattern_info in self.type_patterns.items():
            score = 0
            
            # Prüfe auf Schlüsselwörter
            for keyword in pattern_info["keywords"]:
                if keyword in description:
                    score += 1
            
            # Prüfe auf Muster
            if pattern_info["pattern"] in description:
                score += 2
            
            if score > best_score:
                best_score = score
                best_match = paradox_type
        
        # Fallback auf TEMPORAL_LOOP, wenn kein Muster gefunden wurde
        return best_match or EnhancedParadoxType.TEMPORAL_LOOP
    
    def evaluate_severity(self, paradox_instance: ParadoxInstance) -> ParadoxSeverity:
        """
        Bewertung des Schweregrades eines Paradoxes
        
        Args:
            paradox_instance: Die zu bewertende Paradoxinstanz
            
        Returns:
            Schweregrad des Paradoxes
        """
        # Faktoren für den Schweregrad
        factors = {
            "affected_nodes": len(paradox_instance.affected_node_ids),
            "paradox_type": self._get_type_severity(paradox_instance.paradox_type),
            "timeline_impact": self._estimate_timeline_impact(paradox_instance)
        }
        
        # Gewichtete Summe der Faktoren
        weights = {
            "affected_nodes": 0.3,
            "paradox_type": 0.4,
            "timeline_impact": 0.3
        }
        
        severity_score = sum(factors[k] * weights[k] for k in factors)
        
        # Zuordnung zu ParadoxSeverity
        if severity_score < 0.2:
            return ParadoxSeverity.NEGLIGIBLE
        elif severity_score < 0.4:
            return ParadoxSeverity.MINOR
        elif severity_score < 0.6:
            return ParadoxSeverity.MODERATE
        elif severity_score < 0.8:
            return ParadoxSeverity.MAJOR
        else:
            return ParadoxSeverity.CRITICAL
    
    def get_hierarchical_classification(self, paradox_instance: ParadoxInstance) -> ParadoxHierarchy:
        """
        Hierarchische Kategorisierung von Paradoxien
        
        Args:
            paradox_instance: Die zu kategorisierende Paradoxinstanz
            
        Returns:
            Hierarchische Kategorisierung
        """
        primary_type = self.classify_paradox(paradox_instance)
        
        # Bestimme sekundäre und verwandte Typen
        secondary_types = []
        related_types = []
        
        # Logik zur Bestimmung sekundärer und verwandter Typen
        # basierend auf dem primären Typ und der Paradoxbeschreibung
        description = paradox_instance.description.lower()
        
        # Sekundäre Typen basierend auf Ähnlichkeit zum primären Typ
        if primary_type == EnhancedParadoxType.GRANDFATHER:
            secondary_types.append(EnhancedParadoxType.CAUSAL_VIOLATION)
            related_types.append(EnhancedParadoxType.TEMPORAL_LOOP)
        
        elif primary_type == EnhancedParadoxType.BOOTSTRAP:
            secondary_types.append(EnhancedParadoxType.INFORMATION_PARADOX)
            related_types.append(EnhancedParadoxType.ONTOLOGICAL)
        
        elif primary_type == EnhancedParadoxType.PREDESTINATION:
            secondary_types.append(EnhancedParadoxType.TEMPORAL_LOOP)
            related_types.append(EnhancedParadoxType.BOOTSTRAP)
        
        elif primary_type == EnhancedParadoxType.ONTOLOGICAL:
            secondary_types.append(EnhancedParadoxType.BOOTSTRAP)
            related_types.append(EnhancedParadoxType.INFORMATION_PARADOX)
        
        elif primary_type == EnhancedParadoxType.TEMPORAL_LOOP:
            secondary_types.append(EnhancedParadoxType.PREDESTINATION)
            related_types.append(EnhancedParadoxType.CAUSAL_VIOLATION)
        
        elif primary_type == EnhancedParadoxType.CAUSAL_VIOLATION:
            secondary_types.append(EnhancedParadoxType.GRANDFATHER)
            related_types.append(EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION)
        
        elif primary_type == EnhancedParadoxType.INFORMATION_PARADOX:
            secondary_types.append(EnhancedParadoxType.BOOTSTRAP)
            related_types.append(EnhancedParadoxType.ONTOLOGICAL)
        
        elif primary_type == EnhancedParadoxType.QUANTUM_PARADOX:
            secondary_types.append(EnhancedParadoxType.MULTI_TIMELINE_PARADOX)
            related_types.append(EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION)
        
        elif primary_type == EnhancedParadoxType.MULTI_TIMELINE_PARADOX:
            secondary_types.append(EnhancedParadoxType.QUANTUM_PARADOX)
            related_types.append(EnhancedParadoxType.CAUSAL_VIOLATION)
        
        elif primary_type == EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION:
            secondary_types.append(EnhancedParadoxType.CAUSAL_VIOLATION)
            related_types.append(EnhancedParadoxType.QUANTUM_PARADOX)
        
        # Zusätzliche Typen basierend auf Schlüsselwörtern in der Beschreibung
        for paradox_type, pattern_info in self.type_patterns.items():
            if paradox_type != primary_type and paradox_type not in secondary_types:
                for keyword in pattern_info["keywords"]:
                    if keyword in description and paradox_type not in related_types:
                        related_types.append(paradox_type)
                        break
        
        return ParadoxHierarchy(
            primary_type=primary_type,
            secondary_types=secondary_types,
            related_types=related_types
        )
    
    # Hilfsmethoden
    
    def _get_type_severity(self, paradox_type: EnhancedParadoxType) -> float:
        """
        Gibt den Basisschweregradwert für einen Paradoxtyp zurück
        
        Args:
            paradox_type: Der Paradoxtyp
            
        Returns:
            Basisschweregradwert (0.0-1.0)
        """
        # Basisschweregradwerte für verschiedene Paradoxtypen
        severity_map = {
            EnhancedParadoxType.GRANDFATHER: 0.9,
            EnhancedParadoxType.BOOTSTRAP: 0.7,
            EnhancedParadoxType.PREDESTINATION: 0.5,
            EnhancedParadoxType.ONTOLOGICAL: 0.6,
            EnhancedParadoxType.TEMPORAL_LOOP: 0.8,
            EnhancedParadoxType.CAUSAL_VIOLATION: 0.95,
            EnhancedParadoxType.INFORMATION_PARADOX: 0.75,
            EnhancedParadoxType.QUANTUM_PARADOX: 0.85,
            EnhancedParadoxType.MULTI_TIMELINE_PARADOX: 0.9,
            EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION: 0.7
        }
        
        return severity_map.get(paradox_type, 0.5)
    
    def _estimate_timeline_impact(self, paradox_instance: ParadoxInstance) -> float:
        """
        Schätzt die Auswirkungen eines Paradoxes auf die Zeitlinie
        
        Args:
            paradox_instance: Die zu bewertende Paradoxinstanz
            
        Returns:
            Geschätzte Auswirkung (0.0-1.0)
        """
        # Vereinfachte Implementierung: Basierend auf der Anzahl der betroffenen Knoten
        # und dem Paradoxtyp
        affected_ratio = min(1.0, len(paradox_instance.affected_node_ids) / 10.0)
        type_severity = self._get_type_severity(paradox_instance.paradox_type)
        
        return (affected_ratio * 0.6) + (type_severity * 0.4)


class ParadoxResolver:
    """
    Auflösung von Paradoxien
    """
    
    def __init__(self, echo_prime_controller):
        """
        Initialisiert den Paradoxauflöser
        
        Args:
            echo_prime_controller: ECHO-PRIME Controller-Instanz
        """
        self.echo_prime = echo_prime_controller
        
        # Zugriff auf QTM-Modulator, falls verfügbar
        self.qtm_modulator = None
        if hasattr(echo_prime_controller, 'qtm_modulator'):
            self.qtm_modulator = echo_prime_controller.qtm_modulator
        
        # Zugriff auf Q-LOGIK-Integration, falls verfügbar
        self.qlogik_integration = None
        try:
            if hasattr(echo_prime_controller, 'qlogik_integration'):
                self.qlogik_integration = echo_prime_controller.qlogik_integration
        except ImportError:
            logger.warning("Q-LOGIK-Integration nicht verfügbar")
        
        ztm_log("ParadoxResolver initialisiert")
    
    def resolve_paradox(self, paradox_instance: ParadoxInstance) -> ResolutionResult:
        """
        Auflösung eines Paradoxes
        
        Args:
            paradox_instance: Die aufzulösende Paradoxinstanz
            
        Returns:
            Ergebnis der Auflösung
        """
        # Generiere Auflösungsoptionen
        options = self.get_resolution_options(paradox_instance)
        
        if not options:
            return ResolutionResult(
                paradox_id=paradox_instance.id,
                resolution_option_id="none",
                success=False,
                timeline_id=paradox_instance.timeline_id,
                description="Keine Auflösungsoptionen verfügbar"
            )
        
        # Wähle die optimale Option
        optimal_option = self.select_optimal_resolution(paradox_instance, options)
        
        # Wende die Auflösungsstrategie an
        success = False
        modified_node_ids = []
        new_timeline_id = None
        description = ""
        
        if optimal_option.strategy == ResolutionStrategy.TIMELINE_ADJUSTMENT:
            success, modified_node_ids, description = self._apply_timeline_adjustment(paradox_instance)
        
        elif optimal_option.strategy == ResolutionStrategy.EVENT_MODIFICATION:
            success, modified_node_ids, description = self._apply_event_modification(paradox_instance)
        
        elif optimal_option.strategy == ResolutionStrategy.CAUSAL_REROUTING:
            success, modified_node_ids, description = self._apply_causal_rerouting(paradox_instance)
        
        elif optimal_option.strategy == ResolutionStrategy.QUANTUM_SUPERPOSITION:
            success, modified_node_ids, description = self._apply_quantum_superposition(paradox_instance)
        
        elif optimal_option.strategy == ResolutionStrategy.TEMPORAL_ISOLATION:
            success, new_timeline_id, description = self._apply_temporal_isolation(paradox_instance)
        
        elif optimal_option.strategy == ResolutionStrategy.PARADOX_ABSORPTION:
            success, modified_node_ids, description = self._apply_paradox_absorption(paradox_instance)
        
        return ResolutionResult(
            paradox_id=paradox_instance.id,
            resolution_option_id=optimal_option.id,
            success=success,
            timeline_id=paradox_instance.timeline_id,
            modified_node_ids=modified_node_ids,
            new_timeline_id=new_timeline_id,
            description=description
        )
    
    def get_resolution_options(self, paradox_instance: ParadoxInstance) -> List[ResolutionOption]:
        """
        Generierung mehrerer Auflösungsoptionen
        
        Args:
            paradox_instance: Die Paradoxinstanz, für die Optionen generiert werden sollen
            
        Returns:
            Liste von Auflösungsoptionen
        """
        options = []
        
        # Optionen basierend auf dem Paradoxtyp generieren
        if paradox_instance.paradox_type in [EnhancedParadoxType.GRANDFATHER, EnhancedParadoxType.CAUSAL_VIOLATION]:
            # Für Großvater-Paradoxien und Kausalitätsverletzungen
            options.append(ResolutionOption(
                strategy=ResolutionStrategy.TEMPORAL_ISOLATION,
                confidence=0.85,
                impact=0.7,
                description="Isoliere das Paradox in einer separaten Zeitlinie",
                side_effects=["Erstellt eine neue Zeitlinie", "Reduziert die Wahrscheinlichkeit der ursprünglichen Zeitlinie"]
            ))
            
            options.append(ResolutionOption(
                strategy=ResolutionStrategy.QUANTUM_SUPERPOSITION,
                confidence=0.9,
                impact=0.8,
                description="Wende Quantensuperposition an, um multiple Zustände zu ermöglichen",
                side_effects=["Erhöht die Komplexität der Zeitlinie", "Kann zu Entanglement mit anderen Zeitlinien führen"]
            ))
        
        elif paradox_instance.paradox_type in [EnhancedParadoxType.BOOTSTRAP, EnhancedParadoxType.PREDESTINATION]:
            # Für Bootstrap- und Prädestinations-Paradoxien
            options.append(ResolutionOption(
                strategy=ResolutionStrategy.PARADOX_ABSORPTION,
                confidence=0.95,
                impact=0.5,
                description="Absorbiere das Paradox als stabilen Teil der Zeitlinie",
                side_effects=["Stabilisiert die Zeitlinie", "Reduziert die Flexibilität für zukünftige Änderungen"]
            ))
            
            options.append(ResolutionOption(
                strategy=ResolutionStrategy.CAUSAL_REROUTING,
                confidence=0.8,
                impact=0.6,
                description="Leite die Kausalität um, um eine konsistente Schleife zu erzeugen",
                side_effects=["Kann zu weiteren Paradoxien führen", "Erhöht die Komplexität der Zeitlinie"]
            ))
        
        elif paradox_instance.paradox_type in [EnhancedParadoxType.INFORMATION_PARADOX, EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION]:
            # Für Informationsparadoxien und Konsistenzverletzungen
            options.append(ResolutionOption(
                strategy=ResolutionStrategy.EVENT_MODIFICATION,
                confidence=0.85,
                impact=0.7,
                description="Modifiziere die Informationsquelle, um Konsistenz herzustellen",
                side_effects=["Verändert die Informationsstruktur", "Kann zu Informationsverlust führen"]
            ))
            
            options.append(ResolutionOption(
                strategy=ResolutionStrategy.PARADOX_ABSORPTION,
                confidence=0.75,
                impact=0.5,
                description="Integriere die Inkonsistenz als Teil der Zeitlinie",
                side_effects=["Erhöht die Entropie der Zeitlinie", "Kann zu lokalen Instabilitäten führen"]
            ))
        
        elif paradox_instance.paradox_type in [EnhancedParadoxType.QUANTUM_PARADOX, EnhancedParadoxType.MULTI_TIMELINE_PARADOX]:
            # Für Quantenparadoxien und Multi-Zeitlinien-Paradoxien
            options.append(ResolutionOption(
                strategy=ResolutionStrategy.QUANTUM_SUPERPOSITION,
                confidence=0.95,
                impact=0.8,
                description="Erzeuge eine Quantensuperposition der widersprüchlichen Zustände",
                side_effects=["Erhöht die Quantenkomplexität", "Kann zu Entanglement führen"]
            ))
            
            options.append(ResolutionOption(
                strategy=ResolutionStrategy.TEMPORAL_ISOLATION,
                confidence=0.8,
                impact=0.7,
                description="Isoliere die betroffenen Zeitlinien voneinander",
                side_effects=["Reduziert die Interaktion zwischen Zeitlinien", "Kann zu Informationsverlust führen"]
            ))
        
        else:
            # Für alle anderen Paradoxtypen
            options.append(ResolutionOption(
                strategy=ResolutionStrategy.EVENT_MODIFICATION,
                confidence=0.75,
                impact=0.9,
                description="Modifiziere die auslösenden Ereignisse, um das Paradox zu vermeiden",
                side_effects=["Kann zu signifikanten Änderungen in der Zeitlinie führen", "Hohe Auswirkung auf abhängige Knoten"]
            ))
            
            options.append(ResolutionOption(
                strategy=ResolutionStrategy.TIMELINE_ADJUSTMENT,
                confidence=0.7,
                impact=0.85,
                description="Passe die gesamte Zeitlinie an, um das Paradox zu eliminieren",
                side_effects=["Globale Auswirkungen auf die Zeitlinie", "Kann zu Inkonsistenzen mit anderen Zeitlinien führen"]
            ))
        
        # Füge immer Quantum Superposition als Fallback hinzu, wenn nicht bereits vorhanden
        if not any(o.strategy == ResolutionStrategy.QUANTUM_SUPERPOSITION for o in options):
            options.append(ResolutionOption(
                strategy=ResolutionStrategy.QUANTUM_SUPERPOSITION,
                confidence=0.6,
                impact=0.7,
                description="Fallback: Wende Quantensuperposition an",
                side_effects=["Generische Lösung mit unvorhersehbaren Nebenwirkungen"]
            ))
        
        return options
    
    def evaluate_resolution_impact(self, paradox_instance: ParadoxInstance, resolution: ResolutionOption) -> Dict[str, float]:
        """
        Bewertung der Auswirkungen einer Auflösungsstrategie
        
        Args:
            paradox_instance: Die betroffene Paradoxinstanz
            resolution: Die zu bewertende Auflösungsoption
            
        Returns:
            Dictionary mit Auswirkungsfaktoren
        """
        # Faktoren für die Auswirkungsbewertung
        impact_factors = {
            "timeline_stability": 0.0,  # 0.0 = instabil, 1.0 = stabil
            "node_integrity": 0.0,      # 0.0 = kompromittiert, 1.0 = intakt
            "causal_consistency": 0.0,   # 0.0 = inkonsistent, 1.0 = konsistent
            "probability_impact": 0.0,   # 0.0 = keine Änderung, 1.0 = maximale Änderung
            "quantum_state_impact": 0.0  # 0.0 = keine Änderung, 1.0 = maximale Änderung
        }
        
        # Bewertung basierend auf der Auflösungsstrategie
        if resolution.strategy == ResolutionStrategy.TIMELINE_ADJUSTMENT:
            impact_factors["timeline_stability"] = 0.7
            impact_factors["node_integrity"] = 0.5
            impact_factors["causal_consistency"] = 0.8
            impact_factors["probability_impact"] = 0.9
            impact_factors["quantum_state_impact"] = 0.6
        
        elif resolution.strategy == ResolutionStrategy.EVENT_MODIFICATION:
            impact_factors["timeline_stability"] = 0.8
            impact_factors["node_integrity"] = 0.3
            impact_factors["causal_consistency"] = 0.7
            impact_factors["probability_impact"] = 0.6
            impact_factors["quantum_state_impact"] = 0.4
        
        elif resolution.strategy == ResolutionStrategy.CAUSAL_REROUTING:
            impact_factors["timeline_stability"] = 0.6
            impact_factors["node_integrity"] = 0.7
            impact_factors["causal_consistency"] = 0.5
            impact_factors["probability_impact"] = 0.7
            impact_factors["quantum_state_impact"] = 0.5
        
        elif resolution.strategy == ResolutionStrategy.QUANTUM_SUPERPOSITION:
            impact_factors["timeline_stability"] = 0.5
            impact_factors["node_integrity"] = 0.9
            impact_factors["causal_consistency"] = 0.6
            impact_factors["probability_impact"] = 0.4
            impact_factors["quantum_state_impact"] = 0.9
        
        elif resolution.strategy == ResolutionStrategy.TEMPORAL_ISOLATION:
            impact_factors["timeline_stability"] = 0.9
            impact_factors["node_integrity"] = 0.8
            impact_factors["causal_consistency"] = 0.9
            impact_factors["probability_impact"] = 0.5
            impact_factors["quantum_state_impact"] = 0.7
        
        elif resolution.strategy == ResolutionStrategy.PARADOX_ABSORPTION:
            impact_factors["timeline_stability"] = 0.8
            impact_factors["node_integrity"] = 0.6
            impact_factors["causal_consistency"] = 0.7
            impact_factors["probability_impact"] = 0.3
            impact_factors["quantum_state_impact"] = 0.5
        
        return impact_factors
    
    def select_optimal_resolution(self, paradox_instance: ParadoxInstance, options: List[ResolutionOption]) -> ResolutionOption:
        """
        Automatische Auswahl der optimalen Auflösungsstrategie
        
        Args:
            paradox_instance: Die betroffene Paradoxinstanz
            options: Liste verfügbarer Auflösungsoptionen
            
        Returns:
            Optimale Auflösungsoption
        """
        if not options:
            return None
        
        # Bewerte jede Option
        scored_options = []
        for option in options:
            impact_factors = self.evaluate_resolution_impact(paradox_instance, option)
            
            # Gewichtete Bewertung
            weights = {
                "timeline_stability": 0.25,
                "node_integrity": 0.15,
                "causal_consistency": 0.25,
                "probability_impact": 0.15,
                "quantum_state_impact": 0.2
            }
            
            # Berechne Gesamtpunktzahl
            impact_score = sum(impact_factors[k] * weights[k] for k in impact_factors)
            confidence_score = option.confidence
            
            # Kombiniere Auswirkung und Konfidenz
            total_score = (impact_score * 0.6) + (confidence_score * 0.4)
            
            scored_options.append((option, total_score))
        
        # Wähle Option mit der höchsten Punktzahl
        return max(scored_options, key=lambda x: x[1])[0]
    
    # Implementierungen der Auflösungsstrategien
    
    def _apply_timeline_adjustment(self, paradox_instance: ParadoxInstance) -> Tuple[bool, List[str], str]:
        """
        Implementierung der Zeitlinienanpassung
        
        Args:
            paradox_instance: Die aufzulösende Paradoxinstanz
            
        Returns:
            Tupel aus (Erfolg, Liste modifizierter Knoten-IDs, Beschreibung)
        """
        timeline_id = paradox_instance.timeline_id
        timeline = self.echo_prime.get_timeline(timeline_id)
        
        if not timeline:
            return False, [], "Zeitlinie nicht gefunden"
        
        affected_node_ids = paradox_instance.affected_node_ids
        modified_node_ids = []
        
        try:
            # Anpassung der Zeitlinie durch Modifikation der Wahrscheinlichkeiten
            # und Trigger-Level der betroffenen Knoten
            for node_id in affected_node_ids:
                if node_id in timeline.nodes:
                    node = timeline.nodes[node_id]
                    
                    # Reduziere die Wahrscheinlichkeit des Paradox-verursachenden Pfades
                    original_probability = node.probability
                    node.probability = max(0.1, node.probability * 0.5)
                    
                    # Passe das Trigger-Level an, um die Auslösung zu erschweren
                    if hasattr(node, 'trigger_level') and node.trigger_level:
                        if node.trigger_level.value < 3:  # Annahme: TriggerLevel ist ein Enum mit Werten 0-3
                            node.trigger_level = TriggerLevel(node.trigger_level.value + 1)
                    
                    # Aktualisiere die Zeitlinie
                    timeline.nodes[node_id] = node
                    modified_node_ids.append(node_id)
                    
                    ztm_log(f"Knoten {node_id} angepasst: Wahrscheinlichkeit von {original_probability} auf {node.probability} reduziert")
            
            # Aktualisiere die Zeitlinie im ECHO-PRIME Controller
            self.echo_prime.update_timeline(timeline)
            
            return True, modified_node_ids, f"Zeitlinie {timeline_id} erfolgreich angepasst: {len(modified_node_ids)} Knoten modifiziert"
        
        except Exception as e:
            logger.error(f"Fehler bei der Zeitlinienanpassung: {e}")
            return False, [], f"Fehler bei der Zeitlinienanpassung: {e}"
    
    def _apply_event_modification(self, paradox_instance: ParadoxInstance) -> Tuple[bool, List[str], str]:
        """
        Implementierung der Ereignismodifikation
        
        Args:
            paradox_instance: Die aufzulösende Paradoxinstanz
            
        Returns:
            Tupel aus (Erfolg, Liste modifizierter Knoten-IDs, Beschreibung)
        """
        timeline_id = paradox_instance.timeline_id
        timeline = self.echo_prime.get_timeline(timeline_id)
        
        if not timeline:
            return False, [], "Zeitlinie nicht gefunden"
        
        affected_node_ids = paradox_instance.affected_node_ids
        modified_node_ids = []
        
        try:
            # Identifiziere den Hauptknoten, der das Paradox verursacht
            primary_node_id = affected_node_ids[0] if affected_node_ids else None
            if not primary_node_id or primary_node_id not in timeline.nodes:
                return False, [], "Kein gültiger Hauptknoten gefunden"
            
            primary_node = timeline.nodes[primary_node_id]
            
            # Modifiziere das Ereignis, indem Verbindungen zu problematischen Knoten entfernt werden
            problematic_connections = []
            
            # Finde problematische Verbindungen basierend auf dem Paradoxtyp
            if paradox_instance.paradox_type == EnhancedParadoxType.CAUSAL_VIOLATION:
                # Bei Kausalitätsverletzungen: Entferne Verbindungen, die die Kausalität verletzen
                for child_id in list(primary_node.children):
                    if child_id in timeline.nodes:
                        child = timeline.nodes[child_id]
                        if hasattr(child, 'timestamp') and hasattr(primary_node, 'timestamp'):
                            if child.timestamp < primary_node.timestamp:
                                problematic_connections.append(child_id)
            
            elif paradox_instance.paradox_type == EnhancedParadoxType.TEMPORAL_LOOP:
                # Bei temporalen Schleifen: Entferne eine Verbindung, um die Schleife zu brechen
                for node_id in affected_node_ids[1:]:  # Alle außer dem Hauptknoten
                    if node_id in timeline.nodes:
                        node = timeline.nodes[node_id]
                        if primary_node_id in node.children or primary_node_id in node.parents:
                            problematic_connections.append(node_id)
            
            elif paradox_instance.paradox_type == EnhancedParadoxType.INFORMATION_PARADOX:
                # Bei Informationsparadoxien: Entferne die zirkuläre Informationsquelle
                if hasattr(primary_node, 'metadata') and isinstance(primary_node.metadata, dict):
                    if 'information_source' in primary_node.metadata:
                        source_id = primary_node.metadata['information_source']
                        if source_id in timeline.nodes:
                            # Entferne die Informationsquelle aus den Metadaten
                            primary_node.metadata.pop('information_source')
                            timeline.nodes[primary_node_id] = primary_node
                            modified_node_ids.append(primary_node_id)
                            ztm_log(f"Informationsquelle aus Knoten {primary_node_id} entfernt")
            
            # Entferne problematische Verbindungen
            for connection_id in problematic_connections:
                if connection_id in primary_node.children:
                    primary_node.children.remove(connection_id)
                    child = timeline.nodes[connection_id]
                    if primary_node_id in child.parents:
                        child.parents.remove(primary_node_id)
                        timeline.nodes[connection_id] = child
                        modified_node_ids.append(connection_id)
                
                if connection_id in primary_node.parents:
                    primary_node.parents.remove(connection_id)
                    parent = timeline.nodes[connection_id]
                    if primary_node_id in parent.children:
                        parent.children.remove(primary_node_id)
                        timeline.nodes[connection_id] = parent
                        modified_node_ids.append(connection_id)
            
            # Aktualisiere den Hauptknoten
            timeline.nodes[primary_node_id] = primary_node
            modified_node_ids.append(primary_node_id)
            
            # Aktualisiere die Zeitlinie im ECHO-PRIME Controller
            self.echo_prime.update_timeline(timeline)
            
            return True, modified_node_ids, f"Ereignis erfolgreich modifiziert: {len(modified_node_ids)} Knoten angepasst"
        
        except Exception as e:
            logger.error(f"Fehler bei der Ereignismodifikation: {e}")
            return False, [], f"Fehler bei der Ereignismodifikation: {e}"
    
    def _apply_causal_rerouting(self, paradox_instance: ParadoxInstance) -> Tuple[bool, List[str], str]:
        """
        Implementierung der Kausalitätsumleitung
        
        Args:
            paradox_instance: Die aufzulösende Paradoxinstanz
            
        Returns:
            Tupel aus (Erfolg, Liste modifizierter Knoten-IDs, Beschreibung)
        """
        timeline_id = paradox_instance.timeline_id
        timeline = self.echo_prime.get_timeline(timeline_id)
        
        if not timeline:
            return False, [], "Zeitlinie nicht gefunden"
        
        affected_node_ids = paradox_instance.affected_node_ids
        modified_node_ids = []
        
        try:
            # Identifiziere Start- und Endknoten der problematischen kausalen Kette
            if len(affected_node_ids) < 2:
                return False, [], "Nicht genügend Knoten für Kausalitätsumleitung"
            
            start_node_id = affected_node_ids[0]
            end_node_id = affected_node_ids[-1]
            
            if start_node_id not in timeline.nodes or end_node_id not in timeline.nodes:
                return False, [], "Start- oder Endknoten nicht gefunden"
            
            start_node = timeline.nodes[start_node_id]
            end_node = timeline.nodes[end_node_id]
            
            # Erstelle einen neuen Zwischenknoten für die Umleitung
            intermediate_node_id = str(uuid.uuid4())
            intermediate_node = TimeNode(
                id=intermediate_node_id,
                description=f"Kausale Umleitungsknoten für Paradoxauflösung zwischen {start_node_id} und {end_node_id}",
                parents=[start_node_id],
                children=[end_node_id],
                probability=0.8,
                trigger_level=TriggerLevel.MEDIUM
            )
            
            # Füge den Zwischenknoten zur Zeitlinie hinzu
            timeline.nodes[intermediate_node_id] = intermediate_node
            modified_node_ids.append(intermediate_node_id)
            
            # Aktualisiere die Verbindungen
            start_node.children.append(intermediate_node_id)
            timeline.nodes[start_node_id] = start_node
            modified_node_ids.append(start_node_id)
            
            end_node.parents.append(intermediate_node_id)
            timeline.nodes[end_node_id] = end_node
            modified_node_ids.append(end_node_id)
            
            # Aktualisiere die Zeitlinie im ECHO-PRIME Controller
            self.echo_prime.update_timeline(timeline)
            
            return True, modified_node_ids, f"Kausalität erfolgreich umgeleitet über neuen Knoten {intermediate_node_id}"
        
        except Exception as e:
            logger.error(f"Fehler bei der Kausalitätsumleitung: {e}")
            return False, [], f"Fehler bei der Kausalitätsumleitung: {e}"
    
    def _apply_quantum_superposition(self, paradox_instance: ParadoxInstance) -> Tuple[bool, List[str], str]:
        """
        Implementierung der Quantensuperposition
        
        Args:
            paradox_instance: Die aufzulösende Paradoxinstanz
            
        Returns:
            Tupel aus (Erfolg, Liste modifizierter Knoten-IDs, Beschreibung)
        """
        timeline_id = paradox_instance.timeline_id
        affected_node_ids = paradox_instance.affected_node_ids
        modified_node_ids = []
        
        # Prüfe, ob QTM-Modulator verfügbar ist
        if not self.qtm_modulator:
            return False, [], "QTM-Modulator nicht verfügbar für Quantensuperposition"
        
        try:
            # Wende Quantensuperposition auf die betroffenen Knoten an
            for node_id in affected_node_ids:
                # Erzeuge einen Quanteneffekt für den Knoten
                quantum_effect = self.qtm_modulator.create_quantum_effect(
                    effect_type=QuantumTimeEffect.SUPERPOSITION,
                    target_id=node_id,
                    timeline_id=timeline_id,
                    strength=0.8,  # Hohe Stärke für effektive Auflösung
                    description=f"Quantensuperposition zur Paradoxauflösung für Knoten {node_id}"
                )
                
                # Wende den Quanteneffekt an
                success = self.qtm_modulator.apply_quantum_effect(quantum_effect)
                
                if success:
                    modified_node_ids.append(node_id)
                    ztm_log(f"Quantensuperposition auf Knoten {node_id} angewendet")
            
            if modified_node_ids:
                return True, modified_node_ids, f"Quantensuperposition erfolgreich auf {len(modified_node_ids)} Knoten angewendet"
            else:
                return False, [], "Keine Knoten konnten mit Quantensuperposition modifiziert werden"
        
        except Exception as e:
            logger.error(f"Fehler bei der Anwendung von Quantensuperposition: {e}")
            return False, [], f"Fehler bei der Anwendung von Quantensuperposition: {e}"
    
    def _apply_temporal_isolation(self, paradox_instance: ParadoxInstance) -> Tuple[bool, str, str]:
        """
        Implementierung der temporalen Isolation
        
        Args:
            paradox_instance: Die aufzulösende Paradoxinstanz
            
        Returns:
            Tupel aus (Erfolg, neue Zeitlinien-ID, Beschreibung)
        """
        timeline_id = paradox_instance.timeline_id
        timeline = self.echo_prime.get_timeline(timeline_id)
        
        if not timeline:
            return False, "", "Zeitlinie nicht gefunden"
        
        affected_node_ids = paradox_instance.affected_node_ids
        
        try:
            # Erstelle eine neue Zeitlinie für die Isolation
            new_timeline_id = str(uuid.uuid4())
            new_timeline = Timeline(
                id=new_timeline_id,
                description=f"Isolierte Zeitlinie für Paradox {paradox_instance.id}",
                type=TimelineType.ALTERNATE,
                parent_timeline_id=timeline_id,
                nodes={},
                metadata={
                    "origin": "paradox_isolation",
                    "original_timeline": timeline_id,
                    "isolation_time": datetime.now().isoformat(),
                    "paradox_type": paradox_instance.paradox_type.name
                }
            )
            
            # Kopiere die betroffenen Knoten in die neue Zeitlinie
            for node_id in affected_node_ids:
                if node_id in timeline.nodes:
                    # Kopiere den Knoten
                    node = timeline.nodes[node_id]
                    new_node = TimeNode(
                        id=node_id,  # Behalte die gleiche ID für Referenzzwecke
                        description=node.description,
                        parents=node.parents.copy(),
                        children=node.children.copy(),
                        probability=node.probability,
                        trigger_level=node.trigger_level if hasattr(node, 'trigger_level') else None,
                        metadata=node.metadata.copy() if hasattr(node, 'metadata') else {}
                    )
                    
                    # Füge den Knoten zur neuen Zeitlinie hinzu
                    new_timeline.nodes[node_id] = new_node
            
            # Erstelle die neue Zeitlinie im ECHO-PRIME Controller
            success = self.echo_prime.create_timeline(new_timeline)
            
            if success:
                ztm_log(f"Neue isolierte Zeitlinie erstellt: {new_timeline_id} mit {len(new_timeline.nodes)} Knoten")
                return True, new_timeline_id, f"Paradox erfolgreich in neue Zeitlinie {new_timeline_id} isoliert"
            else:
                return False, "", "Fehler beim Erstellen der neuen Zeitlinie"
        
        except Exception as e:
            logger.error(f"Fehler bei der temporalen Isolation: {e}")
            return False, "", f"Fehler bei der temporalen Isolation: {e}"
    
    def _apply_paradox_absorption(self, paradox_instance: ParadoxInstance) -> Tuple[bool, List[str], str]:
        """
        Implementierung der Paradoxabsorption
        
        Args:
            paradox_instance: Die aufzulösende Paradoxinstanz
            
        Returns:
            Tupel aus (Erfolg, Liste modifizierter Knoten-IDs, Beschreibung)
        """
        timeline_id = paradox_instance.timeline_id
        timeline = self.echo_prime.get_timeline(timeline_id)
        
        if not timeline:
            return False, [], "Zeitlinie nicht gefunden"
        
        affected_node_ids = paradox_instance.affected_node_ids
        modified_node_ids = []
        
        try:
            # Markiere die betroffenen Knoten als "paradox-absorbiert"
            for node_id in affected_node_ids:
                if node_id in timeline.nodes:
                    node = timeline.nodes[node_id]
                    
                    # Füge Metadaten zur Paradoxabsorption hinzu
                    if not hasattr(node, 'metadata') or not isinstance(node.metadata, dict):
                        node.metadata = {}
                    
                    node.metadata["paradox_absorbed"] = True
                    node.metadata["paradox_type"] = paradox_instance.paradox_type.name
                    node.metadata["absorption_time"] = datetime.now().isoformat()
                    
                    # Erhöhe die Stabilität des Knotens
                    node.probability = min(1.0, node.probability * 1.2)
                    
                    # Aktualisiere den Knoten in der Zeitlinie
                    timeline.nodes[node_id] = node
                    modified_node_ids.append(node_id)
                    
                    ztm_log(f"Paradox in Knoten {node_id} absorbiert")
            
            # Aktualisiere die Zeitlinie im ECHO-PRIME Controller
            self.echo_prime.update_timeline(timeline)
            
            return True, modified_node_ids, f"Paradox erfolgreich in {len(modified_node_ids)} Knoten absorbiert"
        
        except Exception as e:
            logger.error(f"Fehler bei der Paradoxabsorption: {e}")
            return False, [], f"Fehler bei der Paradoxabsorption: {e}"


class EnhancedParadoxManagementSystem:
    """
    Integriertes System für das gesamte Paradoxmanagement
    """
    
    def __init__(self, echo_prime_controller):
        """
        Initialisiert das erweiterte Paradoxmanagementsystem
        
        Args:
            echo_prime_controller: ECHO-PRIME Controller-Instanz
        """
        self.echo_prime = echo_prime_controller
        
        # Initialisiere Komponenten
        self.detector = EnhancedParadoxDetector(echo_prime_controller)
        self.classifier = ParadoxClassifier()
        self.resolver = ParadoxResolver(echo_prime_controller)
        self.prevention_system = ParadoxPreventionSystem(echo_prime_controller)
        
        # Paradox-Tracking
        self.detected_paradoxes = {}
        self.resolved_paradoxes = {}
        self.potential_paradoxes = {}
        
        ztm_log("EnhancedParadoxManagementSystem initialisiert")
    
    def scan_timeline(self, timeline_id: str) -> List[ParadoxInstance]:
        """
        Scannt eine Zeitlinie nach Paradoxien
        
        Args:
            timeline_id: ID der zu scannenden Zeitlinie
            
        Returns:
            Liste von erkannten Paradoxinstanzen
        """
        # Erkenne Paradoxien
        paradox_instances = self.detector.detect_paradoxes(timeline_id)
        
        # Klassifiziere und speichere erkannte Paradoxien
        for instance in paradox_instances:
            instance.paradox_type = self.classifier.classify_paradox(instance)
            instance.severity = self.classifier.evaluate_severity(instance)
            self.detected_paradoxes[instance.id] = instance
        
        ztm_log(f"{len(paradox_instances)} Paradoxien in Zeitlinie {timeline_id} erkannt")
        return paradox_instances
    
    def scan_all_timelines(self) -> Dict[str, List[ParadoxInstance]]:
        """
        Scannt alle Zeitlinien nach Paradoxien
        
        Returns:
            Dictionary mit Zeitlinien-IDs als Schlüssel und Listen von Paradoxinstanzen als Werte
        """
        results = {}
        
        # Hole alle Zeitlinien
        timelines = self.echo_prime.get_all_timelines()
        
        for timeline_id in timelines:
            paradox_instances = self.scan_timeline(timeline_id)
            if paradox_instances:
                results[timeline_id] = paradox_instances
        
        return results
    
    def resolve_paradox(self, paradox_id: str) -> ResolutionResult:
        """
        Löst eine spezifische Paradoxie auf
        
        Args:
            paradox_id: ID der aufzulösenden Paradoxie
            
        Returns:
            Ergebnis der Auflösung
        """
        if paradox_id not in self.detected_paradoxes:
            return ResolutionResult(
                paradox_id=paradox_id,
                resolution_option_id="none",
                success=False,
                timeline_id="unknown",
                description="Paradox nicht gefunden"
            )
        
        paradox_instance = self.detected_paradoxes[paradox_id]
        result = self.resolver.resolve_paradox(paradox_instance)
        
        if result.success:
            # Verschiebe von erkannten zu aufgelösten Paradoxien
            self.resolved_paradoxes[paradox_id] = paradox_instance
            self.detected_paradoxes.pop(paradox_id)
            
            ztm_log(f"Paradox {paradox_id} erfolgreich aufgelöst: {result.description}")
        else:
            ztm_log(f"Auflösung von Paradox {paradox_id} fehlgeschlagen: {result.description}")
        
        return result
    
    def resolve_all_paradoxes(self, timeline_id: str = None, min_severity: float = 0.0) -> Dict[str, ResolutionResult]:
        """
        Löst alle erkannten Paradoxien auf
        
        Args:
            timeline_id: Optional, ID der Zeitlinie, deren Paradoxien aufgelöst werden sollen
            min_severity: Minimale Schwere der aufzulösenden Paradoxien
            
        Returns:
            Dictionary mit Paradox-IDs als Schlüssel und Auflösungsergebnissen als Werte
        """
        results = {}
        
        # Filtere Paradoxien nach Zeitlinie und Schwere
        paradoxes_to_resolve = {}
        for paradox_id, instance in self.detected_paradoxes.items():
            if (timeline_id is None or instance.timeline_id == timeline_id) and instance.severity >= min_severity:
                paradoxes_to_resolve[paradox_id] = instance
        
        # Sortiere nach Schwere (höchste zuerst)
        sorted_paradoxes = sorted(
            paradoxes_to_resolve.items(),
            key=lambda x: x[1].severity,
            reverse=True
        )
        
        # Löse Paradoxien auf
        for paradox_id, _ in sorted_paradoxes:
            result = self.resolve_paradox(paradox_id)
            results[paradox_id] = result
        
        return results
    
    def check_potential_paradoxes(self, timeline_id: str) -> List[PotentialParadox]:
        """
        Überprüft eine Zeitlinie auf potenzielle Paradoxien
        
        Args:
            timeline_id: ID der zu überprüfenden Zeitlinie
            
        Returns:
            Liste potenzieller Paradoxien
        """
        potential_paradoxes = self.prevention_system.identify_potential_paradoxes(timeline_id)
        
        # Speichere potenzielle Paradoxien
        for paradox in potential_paradoxes:
            self.potential_paradoxes[paradox.id] = paradox
        
        return potential_paradoxes
    
    def prevent_paradox(self, potential_paradox_id: str) -> bool:
        """
        Verhindert eine potenzielle Paradoxie
        
        Args:
            potential_paradox_id: ID der potenziellen Paradoxie
            
        Returns:
            True, wenn die Prävention erfolgreich war, sonst False
        """
        if potential_paradox_id not in self.potential_paradoxes:
            return False
        
        potential_paradox = self.potential_paradoxes[potential_paradox_id]
        success = self.prevention_system.apply_prevention_measures(potential_paradox)
        
        if success:
            # Entferne aus der Liste potenzieller Paradoxien
            self.potential_paradoxes.pop(potential_paradox_id)
            ztm_log(f"Potenzielle Paradoxie {potential_paradox_id} erfolgreich verhindert")
        else:
            ztm_log(f"Prävention der potenziellen Paradoxie {potential_paradox_id} fehlgeschlagen")
        
        return success
    
    def get_paradox_statistics(self) -> Dict[str, Any]:
        """
        Liefert Statistiken über Paradoxien
        
        Returns:
            Dictionary mit Statistiken
        """
        stats = {
            "detected_count": len(self.detected_paradoxes),
            "resolved_count": len(self.resolved_paradoxes),
            "potential_count": len(self.potential_paradoxes),
            "by_type": {},
            "by_timeline": {},
            "by_severity": {
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0
            }
        }
        
        # Zähle nach Typ
        for instance in self.detected_paradoxes.values():
            paradox_type = instance.paradox_type.name
            if paradox_type not in stats["by_type"]:
                stats["by_type"][paradox_type] = 0
            stats["by_type"][paradox_type] += 1
            
            # Zähle nach Zeitlinie
            timeline_id = instance.timeline_id
            if timeline_id not in stats["by_timeline"]:
                stats["by_timeline"][timeline_id] = 0
            stats["by_timeline"][timeline_id] += 1
            
            # Zähle nach Schwere
            if instance.severity < 0.25:
                stats["by_severity"]["low"] += 1
            elif instance.severity < 0.5:
                stats["by_severity"]["medium"] += 1
            elif instance.severity < 0.75:
                stats["by_severity"]["high"] += 1
            else:
                stats["by_severity"]["critical"] += 1
        
        return stats
    
    def generate_paradox_report(self) -> str:
        """
        Generiert einen Bericht über den aktuellen Paradoxstatus
        
        Returns:
            Formatierter Bericht als String
        """
        stats = self.get_paradox_statistics()
        
        report = "=== PARADOX-STATUSBERICHT ===\n\n"
        
        report += f"Erkannte Paradoxien: {stats['detected_count']}\n"
        report += f"Aufgelöste Paradoxien: {stats['resolved_count']}\n"
        report += f"Potenzielle Paradoxien: {stats['potential_count']}\n\n"
        
        if stats["by_type"]:
            report += "Verteilung nach Typ:\n"
            for paradox_type, count in stats["by_type"].items():
                report += f"  - {paradox_type}: {count}\n"
            report += "\n"
        
        if stats["by_timeline"]:
            report += "Verteilung nach Zeitlinie:\n"
            for timeline_id, count in stats["by_timeline"].items():
                report += f"  - {timeline_id}: {count}\n"
            report += "\n"
        
        report += "Verteilung nach Schwere:\n"
        report += f"  - Niedrig: {stats['by_severity']['low']}\n"
        report += f"  - Mittel: {stats['by_severity']['medium']}\n"
        report += f"  - Hoch: {stats['by_severity']['high']}\n"
        report += f"  - Kritisch: {stats['by_severity']['critical']}\n\n"
        
        # Füge Details zu kritischen Paradoxien hinzu
        critical_paradoxes = [p for p in self.detected_paradoxes.values() if p.severity >= 0.75]
        if critical_paradoxes:
            report += "Kritische Paradoxien:\n"
            for paradox in critical_paradoxes:
                report += f"  - ID: {paradox.id}\n"
                report += f"    Typ: {paradox.paradox_type.name}\n"
                report += f"    Zeitlinie: {paradox.timeline_id}\n"
                report += f"    Schwere: {paradox.severity:.2f}\n"
                report += f"    Betroffene Knoten: {len(paradox.affected_node_ids)}\n"
                report += "\n"
        
        report += "=============================="
        
        return report
