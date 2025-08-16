#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Enhanced Paradox Detector

Diese Komponente erweitert die bestehende ParadoxDetection-Klasse und implementiert
fortgeschrittene Algorithmen zur Erkennung komplexer Paradoxien.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import random
import math
from typing import List, Dict, Tuple, Optional, Set, Any
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [PARADOX-DETECTOR] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.Paradox.Detector")

# Importiere die benötigten Module
try:
    # Importiere die benötigten Module
    # Verwende Mock-Klassen, um Abhängigkeiten zu vermeiden
    from miso.paradox.mock_timeline import MockTimeline, MockTimeNode, MockTemporalEvent
    from miso.paradox.mock_echo_prime import MockECHO_PRIME
    from miso.paradox.mock_paradox_types import MockParadoxType, MockParadoxDetection, MockTemporalIntegrityGuard
    
    # Aliase für die Kompatibilität
    Timeline = MockTimeline
    TimeNode = MockTimeNode
    TemporalEvent = MockTemporalEvent
    ECHO_PRIME = MockECHO_PRIME
    ParadoxType = MockParadoxType
    ParadoxDetection = MockParadoxDetection
    TemporalIntegrityGuard = MockTemporalIntegrityGuard
    
    logger.info("Mock-Module erfolgreich importiert")
except ImportError as e:
    logger.error(f"Fehler beim Importieren der Module: {e}")
    sys.exit(1)

@dataclass
class ParadoxInstance:
    """Repräsentiert eine Instanz eines erkannten Paradoxes"""
    id: str
    type: 'EnhancedParadoxType'
    severity: 'ParadoxSeverity'
    timeline_id: str
    affected_nodes: List[str]
    description: str
    detection_time: datetime
    probability: float
    causal_chain: List[Tuple[str, str]]  # Liste von (node_id, event_id) Paaren
    metadata: Dict[str, Any]

@dataclass
class PotentialParadox:
    """Repräsentiert ein potentielles Paradox, das noch nicht eingetreten ist"""
    id: str
    timeline_id: str
    risk_nodes: List[str]
    probability: float
    estimated_severity: 'ParadoxSeverity'
    description: str
    detection_time: datetime
    time_to_occurrence: float  # Geschätzte Zeit bis zum Eintreten in Sekunden
    preventive_actions: List[str]

class EnhancedParadoxType(Enum):
    """Erweiterte Typen von Paradoxien"""
    # Grundlegende Paradoxtypen
    GRANDFATHER = auto()
    BOOTSTRAP = auto()
    PREDESTINATION = auto()
    ONTOLOGICAL = auto()
    
    # Erweiterte Paradoxtypen
    TEMPORAL_LOOP = auto()
    CAUSAL_VIOLATION = auto()
    INFORMATION_PARADOX = auto()
    QUANTUM_PARADOX = auto()
    MULTI_TIMELINE_PARADOX = auto()
    SELF_CONSISTENCY_VIOLATION = auto()
    
    @classmethod
    def from_basic_type(cls, basic_type: ParadoxType) -> 'EnhancedParadoxType':
        """Konvertiert einen grundlegenden ParadoxType in einen EnhancedParadoxType"""
        mapping = {
            ParadoxType.GRANDFATHER: cls.GRANDFATHER,
            ParadoxType.BOOTSTRAP: cls.BOOTSTRAP,
            ParadoxType.PREDESTINATION: cls.PREDESTINATION,
            # Weitere Mappings können hier hinzugefügt werden
        }
        return mapping.get(basic_type, cls.CAUSAL_VIOLATION)

class ParadoxSeverity(Enum):
    """Schweregrad eines Paradoxes"""
    NEGLIGIBLE = auto()  # Vernachlässigbare Auswirkungen
    MINOR = auto()       # Geringfügige Auswirkungen
    MODERATE = auto()    # Mäßige Auswirkungen
    MAJOR = auto()       # Erhebliche Auswirkungen
    CRITICAL = auto()    # Kritische Auswirkungen

class EnhancedParadoxDetector:
    """
    Erweiterte Paradoxerkennung
    
    Diese Klasse erweitert die bestehende ParadoxDetection-Klasse und implementiert
    fortgeschrittene Algorithmen zur Erkennung komplexer Paradoxien.
    """
    
    def __init__(self, temporal_integrity_guard: Optional[TemporalIntegrityGuard] = None):
        """
        Initialisiert den erweiterten Paradoxdetektor
        
        Args:
            temporal_integrity_guard: Optionale Instanz des TemporalIntegrityGuard
        """
        self.temporal_integrity_guard = temporal_integrity_guard or TemporalIntegrityGuard()
        self.basic_detector = ParadoxDetection()
        self.echo_prime = ECHO_PRIME()
        self.paradox_threshold = 0.65  # Schwellenwert für die Paradoxwahrscheinlichkeit
        self.potential_paradox_threshold = 0.35  # Schwellenwert für potentielle Paradoxien
        self.detection_history: Dict[str, List[ParadoxInstance]] = {}  # Timeline-ID -> Liste von Paradoxien
        self.potential_paradoxes: Dict[str, List[PotentialParadox]] = {}  # Timeline-ID -> Liste potentieller Paradoxien
        
        logger.info("EnhancedParadoxDetector initialisiert")
    
    def detect_complex_paradoxes(self, timelines: List[Timeline]) -> List[ParadoxInstance]:
        """
        Erkennt komplexe Paradoxien in mehreren Zeitlinien
        
        Args:
            timelines: Liste von Zeitlinien, die überprüft werden sollen
            
        Returns:
            Liste erkannter Paradoxinstanzen
        """
        detected_paradoxes: List[ParadoxInstance] = []
        
        for timeline in timelines:
            # Überprüfe zunächst mit dem grundlegenden Detektor
            basic_paradoxes = self.basic_detector.detect_paradoxes(timeline)
            
            # Konvertiere grundlegende Paradoxien in erweiterte Paradoxinstanzen
            for paradox in basic_paradoxes:
                enhanced_type = EnhancedParadoxType.from_basic_type(paradox.type)
                severity = self._evaluate_severity(paradox, timeline)
                
                instance = ParadoxInstance(
                    id=f"paradox-{timeline.id}-{len(detected_paradoxes)}",
                    type=enhanced_type,
                    severity=severity,
                    timeline_id=timeline.id,
                    affected_nodes=[node.id for node in paradox.affected_nodes],
                    description=paradox.description,
                    detection_time=datetime.now(),
                    probability=1.0,  # Bereits erkannte Paradoxien haben eine Wahrscheinlichkeit von 1.0
                    causal_chain=self._extract_causal_chain(paradox, timeline),
                    metadata={"source": "basic_detector"}
                )
                
                detected_paradoxes.append(instance)
            
            # Führe erweiterte Erkennung durch
            complex_paradoxes = self._detect_complex_paradoxes_in_timeline(timeline)
            detected_paradoxes.extend(complex_paradoxes)
            
            # Aktualisiere den Erkennungsverlauf
            if timeline.id not in self.detection_history:
                self.detection_history[timeline.id] = []
            
            self.detection_history[timeline.id].extend(detected_paradoxes)
        
        logger.info(f"{len(detected_paradoxes)} komplexe Paradoxien erkannt")
        return detected_paradoxes
    
    def _detect_complex_paradoxes_in_timeline(self, timeline: Timeline) -> List[ParadoxInstance]:
        """
        Erkennt komplexe Paradoxien in einer einzelnen Zeitlinie
        
        Args:
            timeline: Zeitlinie, die überprüft werden soll
            
        Returns:
            Liste erkannter Paradoxinstanzen
        """
        detected_paradoxes: List[ParadoxInstance] = []
        
        # Implementiere hier fortgeschrittene Erkennungsalgorithmen
        # 1. Überprüfe auf temporale Schleifen
        temporal_loops = self._detect_temporal_loops(timeline)
        detected_paradoxes.extend(temporal_loops)
        
        # 2. Überprüfe auf Kausalitätsverletzungen
        causal_violations = self._detect_causal_violations(timeline)
        detected_paradoxes.extend(causal_violations)
        
        # 3. Überprüfe auf Informationsparadoxien
        information_paradoxes = self._detect_information_paradoxes(timeline)
        detected_paradoxes.extend(information_paradoxes)
        
        # 4. Überprüfe auf Quantenparadoxien
        quantum_paradoxes = self._detect_quantum_paradoxes(timeline)
        detected_paradoxes.extend(quantum_paradoxes)
        
        # 5. Überprüfe auf Selbstkonsistenzverletzungen
        consistency_violations = self._detect_consistency_violations(timeline)
        detected_paradoxes.extend(consistency_violations)
        
        return detected_paradoxes
    
    def _detect_temporal_loops(self, timeline: Timeline) -> List[ParadoxInstance]:
        """Erkennt temporale Schleifen in einer Zeitlinie"""
        detected_loops: List[ParadoxInstance] = []
        nodes = timeline.get_nodes()
        
        # Erstelle einen gerichteten Graphen der Zeitknoten
        graph: Dict[str, Set[str]] = {node.id: set() for node in nodes}
        
        for node in nodes:
            for ref in node.references:
                if ref in graph:
                    graph[node.id].add(ref)
        
        # Suche nach Zyklen im Graphen
        visited: Dict[str, bool] = {node_id: False for node_id in graph}
        rec_stack: Dict[str, bool] = {node_id: False for node_id in graph}
        
        def is_cyclic_util(node_id: str, path: List[str]) -> Tuple[bool, List[str]]:
            visited[node_id] = True
            rec_stack[node_id] = True
            path.append(node_id)
            
            for neighbor in graph[node_id]:
                if not visited[neighbor]:
                    cycle_found, cycle_path = is_cyclic_util(neighbor, path.copy())
                    if cycle_found:
                        return True, cycle_path
                elif rec_stack[neighbor]:
                    # Zyklus gefunden
                    cycle_start_idx = path.index(neighbor)
                    return True, path[cycle_start_idx:]
            
            rec_stack[node_id] = False
            return False, []
        
        for node_id in graph:
            if not visited[node_id]:
                cycle_found, cycle_path = is_cyclic_util(node_id, [])
                if cycle_found:
                    # Temporale Schleife gefunden
                    loop_instance = ParadoxInstance(
                        id=f"temporal-loop-{timeline.id}-{len(detected_loops)}",
                        type=EnhancedParadoxType.TEMPORAL_LOOP,
                        severity=self._calculate_loop_severity(cycle_path, timeline),
                        timeline_id=timeline.id,
                        affected_nodes=cycle_path,
                        description=f"Temporale Schleife erkannt: {' -> '.join(cycle_path)}",
                        detection_time=datetime.now(),
                        probability=self._calculate_loop_probability(cycle_path, timeline),
                        causal_chain=[(node_id, "") for node_id in cycle_path],
                        metadata={"loop_length": len(cycle_path)}
                    )
                    
                    detected_loops.append(loop_instance)
        
        return detected_loops
    
    def _calculate_loop_severity(self, loop_path: List[str], timeline: Timeline) -> ParadoxSeverity:
        """Berechnet den Schweregrad einer temporalen Schleife"""
        # Die Schwere hängt von der Länge der Schleife und der Anzahl der betroffenen Ereignisse ab
        loop_length = len(loop_path)
        
        if loop_length <= 2:
            return ParadoxSeverity.CRITICAL  # Sehr kurze Schleifen sind kritisch
        elif loop_length <= 4:
            return ParadoxSeverity.MAJOR
        elif loop_length <= 6:
            return ParadoxSeverity.MODERATE
        elif loop_length <= 8:
            return ParadoxSeverity.MINOR
        else:
            return ParadoxSeverity.NEGLIGIBLE  # Sehr lange Schleifen sind weniger problematisch
    
    def _calculate_loop_probability(self, loop_path: List[str], timeline: Timeline) -> float:
        """Berechnet die Wahrscheinlichkeit, dass eine temporale Schleife ein echtes Paradox ist"""
        # Die Wahrscheinlichkeit hängt von der Konsistenz der Ereignisse in der Schleife ab
        # Implementiere hier eine probabilistische Bewertung
        return 0.85  # Temporale Schleifen haben eine hohe Wahrscheinlichkeit, echte Paradoxien zu sein
    
    def _detect_causal_violations(self, timeline: Timeline) -> List[ParadoxInstance]:
        """Erkennt Kausalitätsverletzungen in einer Zeitlinie"""
        # Implementiere hier die Erkennung von Kausalitätsverletzungen
        return []
    
    def _detect_information_paradoxes(self, timeline: Timeline) -> List[ParadoxInstance]:
        """Erkennt Informationsparadoxien in einer Zeitlinie"""
        # Implementiere hier die Erkennung von Informationsparadoxien
        return []
    
    def _detect_quantum_paradoxes(self, timeline: Timeline) -> List[ParadoxInstance]:
        """Erkennt Quantenparadoxien in einer Zeitlinie"""
        # Implementiere hier die Erkennung von Quantenparadoxien
        return []
    
    def _detect_consistency_violations(self, timeline: Timeline) -> List[ParadoxInstance]:
        """Erkennt Selbstkonsistenzverletzungen in einer Zeitlinie"""
        # Implementiere hier die Erkennung von Selbstkonsistenzverletzungen
        return []
    
    def evaluate_paradox_probability(self, timeline: Timeline, event: TemporalEvent) -> float:
        """
        Berechnet die Wahrscheinlichkeit eines Paradoxes für ein Ereignis
        
        Args:
            timeline: Zeitlinie, die das Ereignis enthält
            event: Temporales Ereignis, das bewertet werden soll
            
        Returns:
            Wahrscheinlichkeit eines Paradoxes (0.0 - 1.0)
        """
        # Implementiere hier eine probabilistische Bewertung
        # Faktoren, die berücksichtigt werden sollten:
        # 1. Anzahl der Referenzen auf andere Zeitknoten
        # 2. Zeitliche Distanz zu referenzierten Knoten
        # 3. Konsistenz mit bestehenden Ereignissen
        # 4. Komplexität des Ereignisses
        
        # Einfache Implementierung als Platzhalter
        reference_count = len(event.references)
        temporal_distance = self._calculate_temporal_distance(event, timeline)
        consistency_score = self._calculate_consistency_score(event, timeline)
        complexity = self._calculate_event_complexity(event)
        
        # Gewichtete Summe der Faktoren
        probability = (
            0.3 * min(1.0, reference_count / 5.0) +
            0.3 * (1.0 - min(1.0, temporal_distance / 100.0)) +
            0.3 * (1.0 - consistency_score) +
            0.1 * min(1.0, complexity / 10.0)
        )
        
        return probability
    
    def _calculate_temporal_distance(self, event: TemporalEvent, timeline: Timeline) -> float:
        """Berechnet die zeitliche Distanz eines Ereignisses zu seinen Referenzen"""
        # Implementiere hier die Berechnung der zeitlichen Distanz
        return 50.0  # Platzhalter
    
    def _calculate_consistency_score(self, event: TemporalEvent, timeline: Timeline) -> float:
        """Berechnet einen Konsistenzwert für ein Ereignis"""
        # Implementiere hier die Berechnung der Konsistenz
        return 0.8  # Platzhalter
    
    def _calculate_event_complexity(self, event: TemporalEvent) -> float:
        """Berechnet die Komplexität eines Ereignisses"""
        # Implementiere hier die Berechnung der Komplexität
        return 5.0  # Platzhalter
    
    def detect_potential_paradoxes(self, timeline: Timeline) -> List[PotentialParadox]:
        """
        Frühzeitige Erkennung potentieller Paradoxien
        
        Args:
            timeline: Zeitlinie, die überprüft werden soll
            
        Returns:
            Liste potentieller Paradoxien
        """
        potential_paradoxes: List[PotentialParadox] = []
        nodes = timeline.get_nodes()
        
        for node in nodes:
            for event in node.events:
                # Berechne die Paradoxwahrscheinlichkeit für jedes Ereignis
                probability = self.evaluate_paradox_probability(timeline, event)
                
                # Wenn die Wahrscheinlichkeit über dem Schwellenwert liegt, aber unter 100%,
                # handelt es sich um ein potentielles Paradox
                if self.potential_paradox_threshold <= probability < self.paradox_threshold:
                    severity = self._estimate_potential_severity(event, probability, timeline)
                    time_to_occurrence = self._estimate_time_to_occurrence(event, timeline)
                    preventive_actions = self._suggest_preventive_actions(event, timeline, probability)
                    
                    potential = PotentialParadox(
                        id=f"potential-{timeline.id}-{node.id}-{event.id}",
                        timeline_id=timeline.id,
                        risk_nodes=[node.id],
                        probability=probability,
                        estimated_severity=severity,
                        description=f"Potentielles Paradox erkannt in Knoten {node.id}, Ereignis {event.id}",
                        detection_time=datetime.now(),
                        time_to_occurrence=time_to_occurrence,
                        preventive_actions=preventive_actions
                    )
                    
                    potential_paradoxes.append(potential)
        
        # Aktualisiere die Liste der potentiellen Paradoxien
        if timeline.id not in self.potential_paradoxes:
            self.potential_paradoxes[timeline.id] = []
        
        self.potential_paradoxes[timeline.id] = potential_paradoxes
        
        logger.info(f"{len(potential_paradoxes)} potentielle Paradoxien erkannt in Zeitlinie {timeline.id}")
        return potential_paradoxes
    
    def _estimate_potential_severity(self, event: TemporalEvent, probability: float, timeline: Timeline) -> ParadoxSeverity:
        """Schätzt den potentiellen Schweregrad eines Paradoxes"""
        # Die Schwere hängt von der Wahrscheinlichkeit und der Anzahl der betroffenen Knoten ab
        if probability >= 0.6:
            return ParadoxSeverity.MAJOR
        elif probability >= 0.5:
            return ParadoxSeverity.MODERATE
        elif probability >= 0.4:
            return ParadoxSeverity.MINOR
        else:
            return ParadoxSeverity.NEGLIGIBLE
    
    def _estimate_time_to_occurrence(self, event: TemporalEvent, timeline: Timeline) -> float:
        """Schätzt die Zeit bis zum Eintreten eines potentiellen Paradoxes"""
        # Implementiere hier eine Schätzung der Zeit bis zum Eintreten
        return 3600.0  # Platzhalter: 1 Stunde
    
    def _suggest_preventive_actions(self, event: TemporalEvent, timeline: Timeline, probability: float) -> List[str]:
        """Schlägt präventive Maßnahmen zur Vermeidung eines Paradoxes vor"""
        actions = []
        
        if probability >= 0.6:
            actions.append("Isoliere den Zeitknoten temporär")
            actions.append("Führe eine Konsistenzprüfung der referenzierten Knoten durch")
        elif probability >= 0.5:
            actions.append("Überwache den Zeitknoten auf Veränderungen")
            actions.append("Bereite alternative Zeitlinienpfade vor")
        elif probability >= 0.4:
            actions.append("Markiere den Zeitknoten zur weiteren Beobachtung")
        
        return actions
    
    def _evaluate_severity(self, paradox: Any, timeline: Timeline) -> ParadoxSeverity:
        """Bewertet den Schweregrad eines Paradoxes"""
        # Implementiere hier eine Bewertung des Schweregrades
        return ParadoxSeverity.MODERATE  # Platzhalter
    
    def _extract_causal_chain(self, paradox: Any, timeline: Timeline) -> List[Tuple[str, str]]:
        """Extrahiert die Kausalkette eines Paradoxes"""
        # Implementiere hier die Extraktion der Kausalkette
        return []  # Platzhalter
