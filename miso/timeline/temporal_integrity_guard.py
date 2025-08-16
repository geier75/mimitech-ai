#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - ECHO-PRIME TemporalIntegrityGuard

Diese Datei implementiert den TemporalIntegrityGuard, der vor logischen Paradoxien
schützt und die Integrität der Zeitlinien sicherstellt.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import numpy as np
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
from enum import Enum, auto
from dataclasses import dataclass, field

# Konfiguriere Logging
logger = logging.getLogger("MISO.timeline.integrity_guard")

# Importiere gemeinsame Datenstrukturen
from miso.timeline.echo_prime import TimeNode, Timeline, Trigger, TemporalEvent, TriggerLevel

class ParadoxType(Enum):
    """Typen von Paradoxien"""
    CAUSAL_LOOP = auto()        # Ursache-Wirkungs-Schleife
    SELF_DESTRUCTION = auto()   # Selbstzerstörung
    TEMPORAL_RECURSION = auto() # Zeitliche Rekursion
    LOGICAL_CONTRADICTION = auto() # Logischer Widerspruch
    PROBABILITY_INVERSION = auto() # Wahrscheinlichkeitsumkehrung
    TIMELINE_COLLISION = auto() # Kollision von Zeitlinien

@dataclass
class ParadoxDetection:
    """Repräsentiert eine erkannte Paradoxie"""
    id: str
    type: ParadoxType
    description: str
    severity: float  # 0.0-1.0
    timeline_ids: List[str]
    node_ids: List[str]
    detection_time: float
    resolution_suggestions: List[str]
    resolved: bool = False
    resolution_time: Optional[float] = None
    resolution_method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert die Paradoxie in ein Dictionary"""
        return {
            "id": self.id,
            "type": self.type.name,
            "description": self.description,
            "severity": self.severity,
            "timeline_ids": self.timeline_ids,
            "node_ids": self.node_ids,
            "detection_time": self.detection_time,
            "resolution_suggestions": self.resolution_suggestions,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time,
            "resolution_method": self.resolution_method,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParadoxDetection':
        """Erstellt eine Paradoxie aus einem Dictionary"""
        paradox_type = ParadoxType[data["type"]]
        return cls(
            id=data["id"],
            type=paradox_type,
            description=data["description"],
            severity=data["severity"],
            timeline_ids=data["timeline_ids"],
            node_ids=data["node_ids"],
            detection_time=data["detection_time"],
            resolution_suggestions=data["resolution_suggestions"],
            resolved=data["resolved"],
            resolution_time=data["resolution_time"],
            resolution_method=data["resolution_method"],
            metadata=data["metadata"]
        )

class TemporalIntegrityGuard:
    """
    Schützt vor logischen Paradoxien und sichert die Integrität der Zeitlinien
    """
    
    def __init__(self):
        """Initialisiert den TemporalIntegrityGuard"""
        self.paradox_detections = {}
        self.timeline_cache = {}
        self.rollback_buffers = {}
        self.integrity_rules = self._initialize_integrity_rules()
        logger.info("TemporalIntegrityGuard initialisiert")
    
    def _initialize_integrity_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialisiert die Integritätsregeln
        
        Returns:
            Dictionary mit Integritätsregeln
        """
        return {
            "causal_loop": {
                "description": "Verhindert Ursache-Wirkungs-Schleifen",
                "severity": 0.9,
                "check_function": self._check_causal_loop
            },
            "self_destruction": {
                "description": "Verhindert Selbstzerstörung von Zeitlinien",
                "severity": 1.0,
                "check_function": self._check_self_destruction
            },
            "temporal_recursion": {
                "description": "Verhindert zeitliche Rekursionen",
                "severity": 0.8,
                "check_function": self._check_temporal_recursion
            },
            "logical_contradiction": {
                "description": "Verhindert logische Widersprüche",
                "severity": 0.7,
                "check_function": self._check_logical_contradiction
            },
            "probability_inversion": {
                "description": "Verhindert Wahrscheinlichkeitsumkehrungen",
                "severity": 0.6,
                "check_function": self._check_probability_inversion
            },
            "timeline_collision": {
                "description": "Verhindert Kollisionen von Zeitlinien",
                "severity": 0.8,
                "check_function": self._check_timeline_collision
            }
        }
    
    def check_timeline_integrity(self, timeline: Timeline) -> List[ParadoxDetection]:
        """
        Prüft die Integrität einer Zeitlinie
        
        Args:
            timeline: Zu prüfende Zeitlinie
            
        Returns:
            Liste von erkannten Paradoxien
        """
        # Cache Zeitlinie
        self.timeline_cache[timeline.id] = timeline
        
        # Erstelle Rollback-Buffer, falls noch nicht vorhanden
        if timeline.id not in self.rollback_buffers:
            self.rollback_buffers[timeline.id] = []
        
        # Speichere aktuellen Zustand im Rollback-Buffer
        self.rollback_buffers[timeline.id].append({
            "timestamp": time.time(),
            "timeline_state": timeline.to_dict()
        })
        
        # Begrenze Größe des Rollback-Buffers
        if len(self.rollback_buffers[timeline.id]) > 10:
            self.rollback_buffers[timeline.id].pop(0)
        
        # Prüfe alle Integritätsregeln
        detected_paradoxes = []
        
        for rule_name, rule in self.integrity_rules.items():
            check_function = rule["check_function"]
            paradoxes = check_function(timeline)
            
            for paradox in paradoxes:
                # Speichere Paradoxie
                self.paradox_detections[paradox.id] = paradox
                detected_paradoxes.append(paradox)
        
        return detected_paradoxes
    
    def check_timeline_set_integrity(self, timelines: List[Timeline]) -> List[ParadoxDetection]:
        """
        Prüft die Integrität eines Sets von Zeitlinien
        
        Args:
            timelines: Liste von Zeitlinien
            
        Returns:
            Liste von erkannten Paradoxien
        """
        # Cache Zeitlinien
        for timeline in timelines:
            self.timeline_cache[timeline.id] = timeline
        
        # Prüfe jede Zeitlinie einzeln
        individual_paradoxes = []
        for timeline in timelines:
            individual_paradoxes.extend(self.check_timeline_integrity(timeline))
        
        # Prüfe Beziehungen zwischen Zeitlinien
        cross_timeline_paradoxes = self._check_cross_timeline_integrity(timelines)
        
        return individual_paradoxes + cross_timeline_paradoxes
    
    def _check_cross_timeline_integrity(self, timelines: List[Timeline]) -> List[ParadoxDetection]:
        """
        Prüft die Integrität zwischen Zeitlinien
        
        Args:
            timelines: Liste von Zeitlinien
            
        Returns:
            Liste von erkannten Paradoxien
        """
        paradoxes = []
        
        # Prüfe auf Kollisionen zwischen Zeitlinien
        for i in range(len(timelines)):
            for j in range(i+1, len(timelines)):
                timeline1 = timelines[i]
                timeline2 = timelines[j]
                
                # Prüfe auf Kollisionen
                collision_paradoxes = self._check_timeline_collision(timeline1, timeline2)
                paradoxes.extend(collision_paradoxes)
        
        # Prüfe auf Wahrscheinlichkeitsinversionen zwischen Zeitlinien
        for i in range(len(timelines)):
            for j in range(i+1, len(timelines)):
                timeline1 = timelines[i]
                timeline2 = timelines[j]
                
                # Prüfe auf Wahrscheinlichkeitsinversionen
                inversion_paradoxes = self._check_cross_timeline_probability_inversion(timeline1, timeline2)
                paradoxes.extend(inversion_paradoxes)
        
        return paradoxes
    
    def _check_causal_loop(self, timeline: Timeline) -> List[ParadoxDetection]:
        """
        Prüft auf Ursache-Wirkungs-Schleifen
        
        Args:
            timeline: Zu prüfende Zeitlinie
            
        Returns:
            Liste von erkannten Paradoxien
        """
        paradoxes = []
        
        # Eine Ursache-Wirkungs-Schleife liegt vor, wenn ein Knoten indirekt sein eigener Vorfahre ist
        # Implementiere einen Algorithmus zur Erkennung von Zyklen im Graphen
        
        # Für jeden Knoten
        for node_id, node in timeline.nodes.items():
            # Prüfe, ob der Knoten Teil eines Zyklus ist
            visited = set()
            if self._is_in_cycle(timeline, node_id, visited):
                # Erstelle Paradoxie
                paradox_id = str(uuid.uuid4())
                
                paradox = ParadoxDetection(
                    id=paradox_id,
                    type=ParadoxType.CAUSAL_LOOP,
                    description=f"Ursache-Wirkungs-Schleife erkannt in Zeitlinie {timeline.id}",
                    severity=self.integrity_rules["causal_loop"]["severity"],
                    timeline_ids=[timeline.id],
                    node_ids=[node_id],
                    detection_time=time.time(),
                    resolution_suggestions=[
                        "Entferne einen der Knoten in der Schleife",
                        "Füge einen zusätzlichen Knoten ein, um die Schleife zu unterbrechen",
                        "Reduziere die Wahrscheinlichkeit der Schleife"
                    ]
                )
                
                paradoxes.append(paradox)
        
        return paradoxes
    
    def _is_in_cycle(self, timeline: Timeline, node_id: str, visited: Set[str]) -> bool:
        """
        Prüft, ob ein Knoten Teil eines Zyklus ist
        
        Args:
            timeline: Zeitlinie
            node_id: ID des zu prüfenden Knotens
            visited: Set von bereits besuchten Knoten
            
        Returns:
            True, wenn der Knoten Teil eines Zyklus ist, sonst False
        """
        if node_id in visited:
            return True
        
        visited.add(node_id)
        
        node = timeline.nodes.get(node_id)
        if not node:
            return False
        
        # Prüfe alle Kindknoten
        for child_id in node.child_node_ids:
            if self._is_in_cycle(timeline, child_id, visited.copy()):
                return True
        
        return False
    
    def _check_self_destruction(self, timeline: Timeline) -> List[ParadoxDetection]:
        """
        Prüft auf Selbstzerstörung von Zeitlinien
        
        Args:
            timeline: Zu prüfende Zeitlinie
            
        Returns:
            Liste von erkannten Paradoxien
        """
        paradoxes = []
        
        # Eine Selbstzerstörung liegt vor, wenn ein Knoten die Existenz seines Vorfahren verhindert
        # In einer realen Implementierung würde hier eine komplexe Analyse stattfinden
        
        # Für dieses Beispiel suchen wir nach Knoten, die bestimmte Schlüsselwörter enthalten
        self_destruction_keywords = ["zerstör", "vernicht", "eliminier", "lösch", "entfern"]
        
        for node_id, node in timeline.nodes.items():
            # Prüfe, ob die Beschreibung des Knotens Schlüsselwörter enthält
            if any(keyword in node.description.lower() for keyword in self_destruction_keywords):
                # Prüfe, ob der Knoten einen Vorfahren hat
                if node.parent_node_id:
                    # Erstelle Paradoxie
                    paradox_id = str(uuid.uuid4())
                    
                    paradox = ParadoxDetection(
                        id=paradox_id,
                        type=ParadoxType.SELF_DESTRUCTION,
                        description=f"Mögliche Selbstzerstörung erkannt in Zeitlinie {timeline.id}",
                        severity=self.integrity_rules["self_destruction"]["severity"],
                        timeline_ids=[timeline.id],
                        node_ids=[node_id, node.parent_node_id],
                        detection_time=time.time(),
                        resolution_suggestions=[
                            "Entferne den selbstzerstörerischen Knoten",
                            "Modifiziere die Beschreibung des Knotens",
                            "Reduziere die Wahrscheinlichkeit des Knotens auf nahe 0"
                        ]
                    )
                    
                    paradoxes.append(paradox)
        
        return paradoxes
    
    def _check_temporal_recursion(self, timeline: Timeline) -> List[ParadoxDetection]:
        """
        Prüft auf zeitliche Rekursionen
        
        Args:
            timeline: Zu prüfende Zeitlinie
            
        Returns:
            Liste von erkannten Paradoxien
        """
        paradoxes = []
        
        # Eine zeitliche Rekursion liegt vor, wenn ein Knoten zeitlich vor seinem Vorfahren liegt
        
        for node_id, node in timeline.nodes.items():
            # Prüfe, ob der Knoten einen Vorfahren hat
            if node.parent_node_id:
                parent_node = timeline.nodes.get(node.parent_node_id)
                if parent_node and node.timestamp < parent_node.timestamp:
                    # Erstelle Paradoxie
                    paradox_id = str(uuid.uuid4())
                    
                    paradox = ParadoxDetection(
                        id=paradox_id,
                        type=ParadoxType.TEMPORAL_RECURSION,
                        description=f"Zeitliche Rekursion erkannt in Zeitlinie {timeline.id}",
                        severity=self.integrity_rules["temporal_recursion"]["severity"],
                        timeline_ids=[timeline.id],
                        node_ids=[node_id, node.parent_node_id],
                        detection_time=time.time(),
                        resolution_suggestions=[
                            "Passe den Zeitstempel des Knotens an",
                            "Passe den Zeitstempel des Elternknotens an",
                            "Füge einen Zwischenknoten ein"
                        ]
                    )
                    
                    paradoxes.append(paradox)
        
        return paradoxes
    
    def _check_logical_contradiction(self, timeline: Timeline) -> List[ParadoxDetection]:
        """
        Prüft auf logische Widersprüche
        
        Args:
            timeline: Zu prüfende Zeitlinie
            
        Returns:
            Liste von erkannten Paradoxien
        """
        paradoxes = []
        
        # Ein logischer Widerspruch liegt vor, wenn zwei Knoten sich gegenseitig ausschließen
        # In einer realen Implementierung würde hier eine komplexe Analyse stattfinden
        
        # Für dieses Beispiel suchen wir nach Knoten mit gegensätzlichen Beschreibungen
        contradiction_pairs = [
            ("erfolg", "misserfolg"),
            ("ja", "nein"),
            ("wahr", "falsch"),
            ("leben", "tod"),
            ("existier", "nicht existier")
        ]
        
        # Erstelle ein Dictionary mit Knoten-IDs und ihren Beschreibungen
        node_descriptions = {
            node_id: node.description.lower()
            for node_id, node in timeline.nodes.items()
        }
        
        # Prüfe alle Knotenpaare auf Widersprüche
        for i, (node_id1, desc1) in enumerate(node_descriptions.items()):
            for node_id2, desc2 in list(node_descriptions.items())[i+1:]:
                # Prüfe auf Widersprüche
                for word1, word2 in contradiction_pairs:
                    if (word1 in desc1 and word2 in desc2) or (word2 in desc1 and word1 in desc2):
                        # Erstelle Paradoxie
                        paradox_id = str(uuid.uuid4())
                        
                        paradox = ParadoxDetection(
                            id=paradox_id,
                            type=ParadoxType.LOGICAL_CONTRADICTION,
                            description=f"Logischer Widerspruch erkannt in Zeitlinie {timeline.id}",
                            severity=self.integrity_rules["logical_contradiction"]["severity"],
                            timeline_ids=[timeline.id],
                            node_ids=[node_id1, node_id2],
                            detection_time=time.time(),
                            resolution_suggestions=[
                                "Entferne einen der widersprüchlichen Knoten",
                                "Modifiziere die Beschreibungen der Knoten",
                                "Reduziere die Wahrscheinlichkeit eines der Knoten"
                            ]
                        )
                        
                        paradoxes.append(paradox)
        
        return paradoxes
    
    def _check_probability_inversion(self, timeline: Timeline) -> List[ParadoxDetection]:
        """
        Prüft auf Wahrscheinlichkeitsumkehrungen
        
        Args:
            timeline: Zu prüfende Zeitlinie
            
        Returns:
            Liste von erkannten Paradoxien
        """
        paradoxes = []
        
        # Eine Wahrscheinlichkeitsumkehrung liegt vor, wenn ein Kind eine höhere Wahrscheinlichkeit
        # hat als sein Elternknoten
        
        for node_id, node in timeline.nodes.items():
            # Prüfe alle Kindknoten
            for child_id in node.child_node_ids:
                child_node = timeline.nodes.get(child_id)
                if child_node and child_node.probability > node.probability:
                    # Erstelle Paradoxie
                    paradox_id = str(uuid.uuid4())
                    
                    paradox = ParadoxDetection(
                        id=paradox_id,
                        type=ParadoxType.PROBABILITY_INVERSION,
                        description=f"Wahrscheinlichkeitsumkehrung erkannt in Zeitlinie {timeline.id}",
                        severity=self.integrity_rules["probability_inversion"]["severity"],
                        timeline_ids=[timeline.id],
                        node_ids=[node_id, child_id],
                        detection_time=time.time(),
                        resolution_suggestions=[
                            "Erhöhe die Wahrscheinlichkeit des Elternknotens",
                            "Reduziere die Wahrscheinlichkeit des Kindknotens",
                            "Füge einen Zwischenknoten ein"
                        ]
                    )
                    
                    paradoxes.append(paradox)
        
        return paradoxes
    
    def _check_timeline_collision(self, timeline1: Timeline, timeline2: Optional[Timeline] = None) -> List[ParadoxDetection]:
        """
        Prüft auf Kollisionen von Zeitlinien
        
        Args:
            timeline1: Erste Zeitlinie
            timeline2: Zweite Zeitlinie (optional)
            
        Returns:
            Liste von erkannten Paradoxien
        """
        paradoxes = []
        
        # Wenn nur eine Zeitlinie angegeben ist, prüfe auf interne Kollisionen
        if timeline2 is None:
            # In einer realen Implementierung würde hier eine komplexe Analyse stattfinden
            # Für dieses Beispiel gehen wir davon aus, dass keine internen Kollisionen auftreten
            return paradoxes
        
        # Eine Kollision liegt vor, wenn zwei Zeitlinien denselben Knoten haben, aber mit
        # unterschiedlichen Eigenschaften
        
        # Finde gemeinsame Knoten
        common_node_ids = set(timeline1.nodes.keys()) & set(timeline2.nodes.keys())
        
        for node_id in common_node_ids:
            node1 = timeline1.nodes[node_id]
            node2 = timeline2.nodes[node_id]
            
            # Prüfe auf Unterschiede in kritischen Eigenschaften
            if (node1.description != node2.description or
                node1.trigger_level != node2.trigger_level or
                abs(node1.probability - node2.probability) > 0.2):
                
                # Erstelle Paradoxie
                paradox_id = str(uuid.uuid4())
                
                paradox = ParadoxDetection(
                    id=paradox_id,
                    type=ParadoxType.TIMELINE_COLLISION,
                    description=f"Kollision zwischen Zeitlinien {timeline1.id} und {timeline2.id}",
                    severity=self.integrity_rules["timeline_collision"]["severity"],
                    timeline_ids=[timeline1.id, timeline2.id],
                    node_ids=[node_id],
                    detection_time=time.time(),
                    resolution_suggestions=[
                        "Synchronisiere die Eigenschaften des Knotens zwischen den Zeitlinien",
                        "Entferne den Knoten aus einer der Zeitlinien",
                        "Erstelle einen neuen Knoten mit einer eindeutigen ID"
                    ]
                )
                
                paradoxes.append(paradox)
        
        return paradoxes
    
    def _check_cross_timeline_probability_inversion(self, timeline1: Timeline, timeline2: Timeline) -> List[ParadoxDetection]:
        """
        Prüft auf Wahrscheinlichkeitsinversionen zwischen Zeitlinien
        
        Args:
            timeline1: Erste Zeitlinie
            timeline2: Zweite Zeitlinie
            
        Returns:
            Liste von erkannten Paradoxien
        """
        paradoxes = []
        
        # Eine Wahrscheinlichkeitsinversion zwischen Zeitlinien liegt vor, wenn eine Zeitlinie
        # eine höhere Gesamtwahrscheinlichkeit hat, aber ihre Knoten im Durchschnitt eine
        # niedrigere Wahrscheinlichkeit haben
        
        # Berechne durchschnittliche Knotenwahrscheinlichkeit
        avg_prob1 = np.mean([node.probability for node in timeline1.nodes.values()]) if timeline1.nodes else 0
        avg_prob2 = np.mean([node.probability for node in timeline2.nodes.values()]) if timeline2.nodes else 0
        
        # Prüfe auf Inversion
        if (timeline1.probability > timeline2.probability and avg_prob1 < avg_prob2) or \
           (timeline1.probability < timeline2.probability and avg_prob1 > avg_prob2):
            
            # Erstelle Paradoxie
            paradox_id = str(uuid.uuid4())
            
            paradox = ParadoxDetection(
                id=paradox_id,
                type=ParadoxType.PROBABILITY_INVERSION,
                description=f"Wahrscheinlichkeitsinversion zwischen Zeitlinien {timeline1.id} und {timeline2.id}",
                severity=self.integrity_rules["probability_inversion"]["severity"] * 0.8,  # Reduziere Schweregrad für Zeitlinien-übergreifende Inversionen
                timeline_ids=[timeline1.id, timeline2.id],
                node_ids=[],  # Keine spezifischen Knoten
                detection_time=time.time(),
                resolution_suggestions=[
                    "Passe die Gesamtwahrscheinlichkeit der Zeitlinien an",
                    "Passe die Wahrscheinlichkeiten der Knoten an",
                    "Überprüfe die Konsistenz der Wahrscheinlichkeitsberechnungen"
                ]
            )
            
            paradoxes.append(paradox)
        
        return paradoxes
    
    def resolve_paradox(self, 
                      paradox_id: str, 
                      resolution_method: str) -> bool:
        """
        Löst eine Paradoxie
        
        Args:
            paradox_id: ID der Paradoxie
            resolution_method: Methode zur Lösung
            
        Returns:
            True, wenn erfolgreich gelöst, sonst False
        """
        paradox = self.paradox_detections.get(paradox_id)
        if not paradox:
            logger.warning(f"Paradoxie {paradox_id} existiert nicht")
            return False
        
        # Markiere Paradoxie als gelöst
        paradox.resolved = True
        paradox.resolution_time = time.time()
        paradox.resolution_method = resolution_method
        
        logger.info(f"Paradoxie {paradox_id} gelöst mit Methode: {resolution_method}")
        
        return True
    
    def get_paradox(self, paradox_id: str) -> Optional[ParadoxDetection]:
        """
        Gibt eine Paradoxie zurück
        
        Args:
            paradox_id: ID der Paradoxie
            
        Returns:
            ParadoxDetection oder None, falls nicht gefunden
        """
        return self.paradox_detections.get(paradox_id)
    
    def get_all_paradoxes(self) -> Dict[str, ParadoxDetection]:
        """
        Gibt alle Paradoxien zurück
        
        Returns:
            Dictionary mit allen Paradoxien
        """
        return self.paradox_detections
    
    def get_active_paradoxes(self) -> List[ParadoxDetection]:
        """
        Gibt alle aktiven (ungelösten) Paradoxien zurück
        
        Returns:
            Liste von aktiven Paradoxien
        """
        return [
            paradox for paradox in self.paradox_detections.values()
            if not paradox.resolved
        ]
    
    def rollback_timeline(self, timeline_id: str, steps: int = 1) -> Optional[Timeline]:
        """
        Setzt eine Zeitlinie auf einen früheren Zustand zurück
        
        Args:
            timeline_id: ID der Zeitlinie
            steps: Anzahl der Schritte zurück
            
        Returns:
            Zurückgesetzte Timeline oder None, falls nicht möglich
        """
        if timeline_id not in self.rollback_buffers:
            logger.warning(f"Kein Rollback-Buffer für Zeitlinie {timeline_id}")
            return None
        
        # Prüfe, ob genügend Schritte im Buffer sind
        if len(self.rollback_buffers[timeline_id]) <= steps:
            logger.warning(f"Nicht genügend Schritte im Rollback-Buffer für Zeitlinie {timeline_id}")
            return None
        
        # Hole den gewünschten Zustand
        rollback_state = self.rollback_buffers[timeline_id][-(steps+1)]
        
        # Erstelle Timeline aus dem gespeicherten Zustand
        timeline_dict = rollback_state["timeline_state"]
        timeline = Timeline.from_dict(timeline_dict)
        
        # Aktualisiere Cache
        self.timeline_cache[timeline_id] = timeline
        
        # Entferne neuere Zustände aus dem Buffer
        self.rollback_buffers[timeline_id] = self.rollback_buffers[timeline_id][:-(steps)]
        
        logger.info(f"Zeitlinie {timeline_id} um {steps} Schritte zurückgesetzt")
        
        return timeline
