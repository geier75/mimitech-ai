#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - ECHO-PRIME Paradox-Implementierung

Dieses Modul implementiert die Paradox-Komponenten für die ECHO-PRIME Engine.
Es definiert Funktionalität zur Erkennung und Auflösung von temporalen Paradoxien.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import datetime
import uuid
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.engines.echo_prime.paradox")

# Importiere Timeline-Komponenten
from .timeline import Timeline, TimeNode, TemporalEvent, Trigger

class ParadoxType(Enum):
    """Typen von temporalen Paradoxien"""
    GRANDFATHER = auto()  # Großvater-Paradoxon (Ereignis verhindert seine eigene Ursache)
    BOOTSTRAP = auto()    # Bootstrap-Paradoxon (Ereignis ist seine eigene Ursache)
    PREDESTINATION = auto()  # Prädestinations-Paradoxon (Versuch, die Zukunft zu ändern, führt zu ihr)
    CONSISTENCY = auto()  # Konsistenz-Paradoxon (Widersprüchliche Ereignisse)
    TEMPORAL_LOOP = auto()  # Zeitschleife (Endlose Wiederholung von Ereignissen)
    BUTTERFLY = auto()    # Schmetterlingseffekt (Kleine Änderung mit großen Auswirkungen)
    QUANTUM = auto()      # Quantenparadoxon (Quanteneffekte in der Zeitlinie)
    UNKNOWN = auto()      # Unbekanntes Paradoxon

@dataclass
class ParadoxDetectionResult:
    """Ergebnis der Paradoxerkennung"""
    paradox_type: ParadoxType
    description: str
    events: List[str]  # Liste von Event-IDs
    triggers: List[str]  # Liste von Trigger-IDs
    severity: float  # 0-1, wobei 1 die höchste Schwere ist
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert das Ergebnis in ein Wörterbuch
        
        Returns:
            Wörterbuch mit den Ergebnisdaten
        """
        return {
            "id": self.id,
            "paradox_type": self.paradox_type.name,
            "description": self.description,
            "events": self.events,
            "triggers": self.triggers,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParadoxDetectionResult':
        """
        Erstellt ein Ergebnis aus einem Wörterbuch
        
        Args:
            data: Wörterbuch mit den Ergebnisdaten
            
        Returns:
            Neues Ergebnis
        """
        return cls(
            paradox_type=ParadoxType[data["paradox_type"]],
            description=data["description"],
            events=data["events"],
            triggers=data["triggers"],
            severity=data["severity"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            id=data.get("id", str(uuid.uuid4()))
        )

@dataclass
class ParadoxResolutionResult:
    """Ergebnis der Paradoxauflösung"""
    success: bool
    message: str
    paradoxes: List[ParadoxDetectionResult]
    resolved_paradoxes: List[ParadoxDetectionResult]
    unresolved_paradoxes: List[ParadoxDetectionResult]
    modifications: List[Dict[str, Any]]  # Liste von Modifikationen an der Zeitlinie
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert das Ergebnis in ein Wörterbuch
        
        Returns:
            Wörterbuch mit den Ergebnisdaten
        """
        return {
            "id": self.id,
            "success": self.success,
            "message": self.message,
            "paradoxes": [p.to_dict() for p in self.paradoxes],
            "resolved_paradoxes": [p.to_dict() for p in self.resolved_paradoxes],
            "unresolved_paradoxes": [p.to_dict() for p in self.unresolved_paradoxes],
            "modifications": self.modifications,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParadoxResolutionResult':
        """
        Erstellt ein Ergebnis aus einem Wörterbuch
        
        Args:
            data: Wörterbuch mit den Ergebnisdaten
            
        Returns:
            Neues Ergebnis
        """
        return cls(
            success=data["success"],
            message=data["message"],
            paradoxes=[ParadoxDetectionResult.from_dict(p) for p in data["paradoxes"]],
            resolved_paradoxes=[ParadoxDetectionResult.from_dict(p) for p in data["resolved_paradoxes"]],
            unresolved_paradoxes=[ParadoxDetectionResult.from_dict(p) for p in data["unresolved_paradoxes"]],
            modifications=data["modifications"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            id=data.get("id", str(uuid.uuid4()))
        )

class ParadoxDetector:
    """
    Paradoxdetektor für Zeitlinien
    
    Diese Klasse implementiert die Erkennung von temporalen Paradoxien in Zeitlinien.
    """
    
    def __init__(self):
        """Initialisiert den ParadoxDetector"""
        logger.info("ParadoxDetector initialisiert")
    
    def detect_paradoxes(self, timeline: Timeline) -> List[ParadoxDetectionResult]:
        """
        Erkennt Paradoxien in einer Zeitlinie
        
        Args:
            timeline: Zu analysierende Zeitlinie
            
        Returns:
            Liste der erkannten Paradoxien
        """
        paradoxes = []
        
        # Erkenne Großvater-Paradoxon
        paradoxes.extend(self._detect_grandfather_paradox(timeline))
        
        # Erkenne Bootstrap-Paradoxon
        paradoxes.extend(self._detect_bootstrap_paradox(timeline))
        
        # Erkenne Konsistenz-Paradoxon
        paradoxes.extend(self._detect_consistency_paradox(timeline))
        
        # Erkenne Zeitschleife
        paradoxes.extend(self._detect_temporal_loop(timeline))
        
        logger.info(f"{len(paradoxes)} Paradoxien in Zeitlinie '{timeline.name}' erkannt")
        
        return paradoxes
    
    def _detect_grandfather_paradox(self, timeline: Timeline) -> List[ParadoxDetectionResult]:
        """
        Erkennt Großvater-Paradoxien in einer Zeitlinie
        
        Args:
            timeline: Zu analysierende Zeitlinie
            
        Returns:
            Liste der erkannten Paradoxien
        """
        paradoxes = []
        
        # Analysiere Trigger
        for trigger in timeline.triggers.values():
            source_event = timeline.get_event(trigger.source_event_id)
            target_event = timeline.get_event(trigger.target_event_id)
            
            if source_event and target_event:
                # Überprüfe, ob das Zielereignis zeitlich vor dem Quellereignis liegt
                if target_event.timestamp < source_event.timestamp:
                    # Überprüfe, ob das Zielereignis das Quellereignis verhindern könnte
                    if self._could_prevent_event(target_event, source_event, timeline):
                        paradoxes.append(ParadoxDetectionResult(
                            paradox_type=ParadoxType.GRANDFATHER,
                            description=f"Großvater-Paradoxon: Ereignis '{target_event.name}' könnte Ereignis '{source_event.name}' verhindern",
                            events=[source_event.id, target_event.id],
                            triggers=[trigger.id],
                            severity=0.8
                        ))
        
        return paradoxes
    
    def _detect_bootstrap_paradox(self, timeline: Timeline) -> List[ParadoxDetectionResult]:
        """
        Erkennt Bootstrap-Paradoxien in einer Zeitlinie
        
        Args:
            timeline: Zu analysierende Zeitlinie
            
        Returns:
            Liste der erkannten Paradoxien
        """
        paradoxes = []
        
        # Erstelle Abhängigkeitsgraph
        dependencies = {}
        
        for trigger in timeline.triggers.values():
            if trigger.source_event_id not in dependencies:
                dependencies[trigger.source_event_id] = set()
            
            dependencies[trigger.source_event_id].add(trigger.target_event_id)
        
        # Suche nach Zyklen im Abhängigkeitsgraph
        visited = set()
        path = set()
        
        for event_id in timeline.events:
            if event_id not in visited:
                cycle = self._find_cycle(event_id, dependencies, visited, path)
                
                if cycle:
                    # Erstelle Paradox
                    events = []
                    triggers = []
                    
                    for i in range(len(cycle)):
                        events.append(cycle[i])
                        
                        # Finde Trigger zwischen den Ereignissen
                        source_id = cycle[i]
                        target_id = cycle[(i + 1) % len(cycle)]
                        
                        for trigger in timeline.triggers.values():
                            if trigger.source_event_id == source_id and trigger.target_event_id == target_id:
                                triggers.append(trigger.id)
                    
                    paradoxes.append(ParadoxDetectionResult(
                        paradox_type=ParadoxType.BOOTSTRAP,
                        description=f"Bootstrap-Paradoxon: Zyklische Abhängigkeit zwischen {len(cycle)} Ereignissen",
                        events=events,
                        triggers=triggers,
                        severity=0.7
                    ))
        
        return paradoxes
    
    def _detect_consistency_paradox(self, timeline: Timeline) -> List[ParadoxDetectionResult]:
        """
        Erkennt Konsistenz-Paradoxien in einer Zeitlinie
        
        Args:
            timeline: Zu analysierende Zeitlinie
            
        Returns:
            Liste der erkannten Paradoxien
        """
        paradoxes = []
        
        # Gruppiere Ereignisse nach Zeitstempel
        events_by_timestamp = {}
        
        for event in timeline.events.values():
            timestamp = event.timestamp.replace(microsecond=0)
            
            if timestamp not in events_by_timestamp:
                events_by_timestamp[timestamp] = []
            
            events_by_timestamp[timestamp].append(event)
        
        # Überprüfe auf widersprüchliche Ereignisse
        for timestamp, events in events_by_timestamp.items():
            if len(events) > 1:
                for i in range(len(events)):
                    for j in range(i + 1, len(events)):
                        if self._are_events_contradictory(events[i], events[j]):
                            # Finde Trigger, die zu den Ereignissen führen
                            triggers = []
                            
                            for trigger in timeline.triggers.values():
                                if trigger.target_event_id in [events[i].id, events[j].id]:
                                    triggers.append(trigger.id)
                            
                            paradoxes.append(ParadoxDetectionResult(
                                paradox_type=ParadoxType.CONSISTENCY,
                                description=f"Konsistenz-Paradoxon: Widersprüchliche Ereignisse '{events[i].name}' und '{events[j].name}'",
                                events=[events[i].id, events[j].id],
                                triggers=triggers,
                                severity=0.9
                            ))
        
        return paradoxes
    
    def _detect_temporal_loop(self, timeline: Timeline) -> List[ParadoxDetectionResult]:
        """
        Erkennt Zeitschleifen in einer Zeitlinie
        
        Args:
            timeline: Zu analysierende Zeitlinie
            
        Returns:
            Liste der erkannten Paradoxien
        """
        paradoxes = []
        
        # Erstelle Abhängigkeitsgraph
        dependencies = {}
        
        for trigger in timeline.triggers.values():
            if trigger.source_event_id not in dependencies:
                dependencies[trigger.source_event_id] = set()
            
            dependencies[trigger.source_event_id].add(trigger.target_event_id)
        
        # Suche nach Zyklen im Abhängigkeitsgraph
        visited = set()
        path = set()
        
        for event_id in timeline.events:
            if event_id not in visited:
                cycle = self._find_cycle(event_id, dependencies, visited, path)
                
                if cycle:
                    # Überprüfe, ob der Zyklus eine Zeitschleife ist
                    is_temporal_loop = True
                    
                    for i in range(len(cycle)):
                        source_id = cycle[i]
                        target_id = cycle[(i + 1) % len(cycle)]
                        
                        source_event = timeline.get_event(source_id)
                        target_event = timeline.get_event(target_id)
                        
                        if source_event and target_event:
                            # Wenn das Zielereignis nicht zeitlich nach dem Quellereignis liegt,
                            # ist es keine Zeitschleife
                            if not target_event.timestamp > source_event.timestamp:
                                is_temporal_loop = False
                                break
                    
                    if is_temporal_loop:
                        # Erstelle Paradox
                        events = []
                        triggers = []
                        
                        for i in range(len(cycle)):
                            events.append(cycle[i])
                            
                            # Finde Trigger zwischen den Ereignissen
                            source_id = cycle[i]
                            target_id = cycle[(i + 1) % len(cycle)]
                            
                            for trigger in timeline.triggers.values():
                                if trigger.source_event_id == source_id and trigger.target_event_id == target_id:
                                    triggers.append(trigger.id)
                        
                        paradoxes.append(ParadoxDetectionResult(
                            paradox_type=ParadoxType.TEMPORAL_LOOP,
                            description=f"Zeitschleife: {len(cycle)} Ereignisse bilden eine Schleife",
                            events=events,
                            triggers=triggers,
                            severity=0.6
                        ))
        
        return paradoxes
    
    def _could_prevent_event(self, event1: TemporalEvent, event2: TemporalEvent, timeline: Timeline) -> bool:
        """
        Überprüft, ob ein Ereignis ein anderes verhindern könnte
        
        Args:
            event1: Erstes Ereignis
            event2: Zweites Ereignis
            timeline: Zeitlinie
            
        Returns:
            True, wenn das erste Ereignis das zweite verhindern könnte, sonst False
        """
        # In einer realen Implementierung würde hier eine komplexe Analyse durchgeführt
        # Für diese Beispielimplementierung verwenden wir eine einfache Heuristik
        
        # Überprüfe, ob die Ereignisse zeitlich weit genug auseinander liegen
        time_difference = (event2.timestamp - event1.timestamp).total_seconds()
        
        # Wenn die Ereignisse weniger als eine Stunde auseinander liegen,
        # ist es unwahrscheinlich, dass eines das andere verhindern könnte
        if time_difference < 3600:
            return False
        
        # Überprüfe, ob es einen Pfad von Triggern zwischen den Ereignissen gibt
        return self._has_path(event1.id, event2.id, timeline)
    
    def _has_path(self, source_id: str, target_id: str, timeline: Timeline) -> bool:
        """
        Überprüft, ob es einen Pfad von Triggern zwischen zwei Ereignissen gibt
        
        Args:
            source_id: ID des Quellereignisses
            target_id: ID des Zielereignisses
            timeline: Zeitlinie
            
        Returns:
            True, wenn es einen Pfad gibt, sonst False
        """
        # Erstelle Abhängigkeitsgraph
        dependencies = {}
        
        for trigger in timeline.triggers.values():
            if trigger.source_event_id not in dependencies:
                dependencies[trigger.source_event_id] = set()
            
            dependencies[trigger.source_event_id].add(trigger.target_event_id)
        
        # Führe Breitensuche durch
        queue = [source_id]
        visited = set(queue)
        
        while queue:
            current_id = queue.pop(0)
            
            if current_id == target_id:
                return True
            
            for next_id in dependencies.get(current_id, set()):
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append(next_id)
        
        return False
    
    def _find_cycle(self, 
                   event_id: str, 
                   dependencies: Dict[str, Set[str]], 
                   visited: Set[str], 
                   path: Set[str]) -> List[str]:
        """
        Sucht nach einem Zyklus im Abhängigkeitsgraph
        
        Args:
            event_id: ID des aktuellen Ereignisses
            dependencies: Abhängigkeitsgraph
            visited: Menge der besuchten Ereignisse
            path: Aktueller Pfad
            
        Returns:
            Liste der Ereignis-IDs im Zyklus oder leere Liste, wenn kein Zyklus gefunden wurde
        """
        visited.add(event_id)
        path.add(event_id)
        
        for next_id in dependencies.get(event_id, set()):
            if next_id not in visited:
                cycle = self._find_cycle(next_id, dependencies, visited, path)
                
                if cycle:
                    return cycle
            elif next_id in path:
                # Zyklus gefunden
                cycle = []
                current_id = event_id
                
                while current_id != next_id:
                    cycle.append(current_id)
                    
                    # Finde Vorgänger
                    for prev_id in dependencies:
                        if current_id in dependencies[prev_id]:
                            current_id = prev_id
                            break
                
                cycle.append(next_id)
                cycle.reverse()
                
                return cycle
        
        path.remove(event_id)
        
        return []
    
    def _are_events_contradictory(self, event1: TemporalEvent, event2: TemporalEvent) -> bool:
        """
        Überprüft, ob zwei Ereignisse widersprüchlich sind
        
        Args:
            event1: Erstes Ereignis
            event2: Zweites Ereignis
            
        Returns:
            True, wenn die Ereignisse widersprüchlich sind, sonst False
        """
        # In einer realen Implementierung würde hier eine komplexe Analyse durchgeführt
        # Für diese Beispielimplementierung verwenden wir eine einfache Heuristik
        
        # Überprüfe, ob die Ereignisse denselben Namen haben, aber unterschiedliche Beschreibungen
        if event1.name == event2.name and event1.description != event2.description:
            return True
        
        # Überprüfe, ob die Ereignisse denselben Zeitstempel haben, aber unterschiedliche Namen
        if event1.timestamp == event2.timestamp and event1.name != event2.name:
            return False
        
        # Überprüfe, ob die Ereignisse in den Daten als widersprüchlich markiert sind
        if "contradicts" in event1.data and event2.id in event1.data["contradicts"]:
            return True
        
        if "contradicts" in event2.data and event1.id in event2.data["contradicts"]:
            return True
        
        return False

class ParadoxResolver:
    """
    Paradoxauflöser für Zeitlinien
    
    Diese Klasse implementiert die Auflösung von temporalen Paradoxien in Zeitlinien.
    """
    
    def __init__(self):
        """Initialisiert den ParadoxResolver"""
        logger.info("ParadoxResolver initialisiert")
    
    def resolve_paradoxes(self, 
                         timeline: Timeline, 
                         paradoxes: List[ParadoxDetectionResult]) -> ParadoxResolutionResult:
        """
        Löst Paradoxien in einer Zeitlinie auf
        
        Args:
            timeline: Zu modifizierende Zeitlinie
            paradoxes: Liste der aufzulösenden Paradoxien
            
        Returns:
            Ergebnis der Auflösung
        """
        if not paradoxes:
            return ParadoxResolutionResult(
                success=True,
                message="Keine Paradoxien zu lösen",
                paradoxes=[],
                resolved_paradoxes=[],
                unresolved_paradoxes=[],
                modifications=[]
            )
        
        # Sortiere Paradoxien nach Schwere
        paradoxes = sorted(paradoxes, key=lambda p: p.severity, reverse=True)
        
        resolved_paradoxes = []
        unresolved_paradoxes = []
        modifications = []
        
        for paradox in paradoxes:
            # Versuche, das Paradox aufzulösen
            if paradox.paradox_type == ParadoxType.GRANDFATHER:
                success, mods = self._resolve_grandfather_paradox(timeline, paradox)
            elif paradox.paradox_type == ParadoxType.BOOTSTRAP:
                success, mods = self._resolve_bootstrap_paradox(timeline, paradox)
            elif paradox.paradox_type == ParadoxType.CONSISTENCY:
                success, mods = self._resolve_consistency_paradox(timeline, paradox)
            elif paradox.paradox_type == ParadoxType.TEMPORAL_LOOP:
                success, mods = self._resolve_temporal_loop(timeline, paradox)
            else:
                success = False
                mods = []
            
            if success:
                resolved_paradoxes.append(paradox)
                modifications.extend(mods)
            else:
                unresolved_paradoxes.append(paradox)
        
        # Bestimme Erfolg der Auflösung
        success = len(unresolved_paradoxes) == 0
        
        if success:
            message = f"Alle {len(paradoxes)} Paradoxien erfolgreich aufgelöst"
        else:
            message = f"{len(resolved_paradoxes)} von {len(paradoxes)} Paradoxien aufgelöst, {len(unresolved_paradoxes)} verbleiben"
        
        logger.info(message)
        
        return ParadoxResolutionResult(
            success=success,
            message=message,
            paradoxes=paradoxes,
            resolved_paradoxes=resolved_paradoxes,
            unresolved_paradoxes=unresolved_paradoxes,
            modifications=modifications
        )
    
    def _resolve_grandfather_paradox(self, 
                                    timeline: Timeline, 
                                    paradox: ParadoxDetectionResult) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Löst ein Großvater-Paradoxon auf
        
        Args:
            timeline: Zu modifizierende Zeitlinie
            paradox: Aufzulösendes Paradoxon
            
        Returns:
            Tuple aus Erfolg und Liste der Modifikationen
        """
        modifications = []
        
        # Strategie: Entferne den Trigger, der das Paradoxon verursacht
        for trigger_id in paradox.triggers:
            trigger = timeline.get_trigger(trigger_id)
            
            if trigger:
                # Entferne Trigger
                timeline.remove_trigger(trigger_id)
                
                modifications.append({
                    "type": "remove_trigger",
                    "trigger_id": trigger_id
                })
                
                logger.info(f"Großvater-Paradoxon aufgelöst: Trigger {trigger_id} entfernt")
        
        return True, modifications
    
    def _resolve_bootstrap_paradox(self, 
                                  timeline: Timeline, 
                                  paradox: ParadoxDetectionResult) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Löst ein Bootstrap-Paradoxon auf
        
        Args:
            timeline: Zu modifizierende Zeitlinie
            paradox: Aufzulösendes Paradoxon
            
        Returns:
            Tuple aus Erfolg und Liste der Modifikationen
        """
        modifications = []
        
        # Strategie: Entferne einen der Trigger im Zyklus
        if paradox.triggers:
            # Wähle den Trigger mit der niedrigsten Wahrscheinlichkeit
            trigger_id = None
            min_probability = 1.0
            
            for tid in paradox.triggers:
                trigger = timeline.get_trigger(tid)
                
                if trigger and trigger.probability < min_probability:
                    trigger_id = tid
                    min_probability = trigger.probability
            
            if trigger_id:
                # Entferne Trigger
                timeline.remove_trigger(trigger_id)
                
                modifications.append({
                    "type": "remove_trigger",
                    "trigger_id": trigger_id
                })
                
                logger.info(f"Bootstrap-Paradoxon aufgelöst: Trigger {trigger_id} entfernt")
                
                return True, modifications
        
        return False, modifications
    
    def _resolve_consistency_paradox(self, 
                                    timeline: Timeline, 
                                    paradox: ParadoxDetectionResult) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Löst ein Konsistenz-Paradoxon auf
        
        Args:
            timeline: Zu modifizierende Zeitlinie
            paradox: Aufzulösendes Paradoxon
            
        Returns:
            Tuple aus Erfolg und Liste der Modifikationen
        """
        modifications = []
        
        # Strategie: Entferne eines der widersprüchlichen Ereignisse
        if len(paradox.events) >= 2:
            # Wähle das Ereignis mit der niedrigsten Priorität
            event_id = None
            min_priority = float('inf')
            
            for eid in paradox.events:
                event = timeline.get_event(eid)
                
                if event:
                    priority = event.data.get("priority", 0)
                    
                    if priority < min_priority:
                        event_id = eid
                        min_priority = priority
            
            if event_id:
                # Entferne Ereignis
                timeline.remove_event(event_id)
                
                modifications.append({
                    "type": "remove_event",
                    "event_id": event_id
                })
                
                logger.info(f"Konsistenz-Paradoxon aufgelöst: Ereignis {event_id} entfernt")
                
                return True, modifications
        
        return False, modifications
    
    def _resolve_temporal_loop(self, 
                              timeline: Timeline, 
                              paradox: ParadoxDetectionResult) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Löst eine Zeitschleife auf
        
        Args:
            timeline: Zu modifizierende Zeitlinie
            paradox: Aufzulösendes Paradoxon
            
        Returns:
            Tuple aus Erfolg und Liste der Modifikationen
        """
        modifications = []
        
        # Strategie: Entferne einen der Trigger im Zyklus
        if paradox.triggers:
            # Wähle einen zufälligen Trigger
            trigger_id = paradox.triggers[0]
            
            # Entferne Trigger
            timeline.remove_trigger(trigger_id)
            
            modifications.append({
                "type": "remove_trigger",
                "trigger_id": trigger_id
            })
            
            logger.info(f"Zeitschleife aufgelöst: Trigger {trigger_id} entfernt")
            
            return True, modifications
        
        return False, modifications
