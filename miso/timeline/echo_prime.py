#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - ECHO-PRIME Modul

Temporales Analyse- und Prognosemodul für Zeitachsenoperationen, Kausalität,
Wahrscheinlichkeit und Handlungsstränge. Ermöglicht die Kontrolle, Analyse und 
Voraussicht über vergangene, gegenwärtige und mögliche zukünftige Zeitlinien,
inklusive Handlungstrigger und strategischer Entscheidungsverschiebung.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import datetime
import logging
import json
import threading
import numpy as np
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Union, TYPE_CHECKING, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

# Importiere gemeinsame Basisklassen
from miso.core.timeline_base import Timeline, TimeNode, TimelineType, TriggerLevel

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.timeline.echo_prime")

# Spezialisierte ECHO-PRIME-Erweiterungen der Basisdatentypen

class EchoTimeNode(TimeNode):
    """ECHO-PRIME-spezifische Erweiterung des TimeNode"""
    
    def apply_quantum_effect(self, effect_type, strength=1.0):
        """Wendet einen Quanteneffekt auf den Zeitknoten an
        
        Args:
            effect_type: Art des Quanteneffekts
            strength: Stärke des Effekts (0.0-1.0)
        """
        if "quantum_effects" not in self.metadata:
            self.metadata["quantum_effects"] = []
            
        self.metadata["quantum_effects"].append({
            "type": effect_type,
            "strength": strength,
            "applied_time": time.time()
        })

class EchoTimeline(Timeline):
    """ECHO-PRIME-spezifische Erweiterung der Timeline"""
    
    def connect_nodes(self, source_id=None, target_id=None, parent_id=None, child_id=None, metadata=None):
        """Verbindet zwei Knoten miteinander
        
        Args:
            source_id: ID des Quellknotens (Alternative zu parent_id)
            target_id: ID des Zielknotens (Alternative zu child_id)
            parent_id: ID des Elternknotens
            child_id: ID des Kindknotens
            metadata: Zusätzliche Metadaten für die Verbindung
        """
        # Normalisiere Parameter
        parent = parent_id or source_id
        child = child_id or target_id
        metadata = metadata or {}
        
        if parent is None or child is None:
            logger.error("Verbindung konnte nicht hergestellt werden: Quell- oder Zielknoten fehlt")
            return False
            
        # Prüfe, ob Knoten existieren
        if parent not in self.nodes or child not in self.nodes:
            logger.error(f"Verbindung konnte nicht hergestellt werden: Knoten existiert nicht")
            return False
            
        # Füge Kind zum Elternknoten hinzu
        if isinstance(self.nodes[parent], TimeNode):
            self.nodes[parent].add_child(child)
            
        # Setze Eltern-ID im Kindknoten
        if isinstance(self.nodes[child], TimeNode):
            self.nodes[child].parent_id = parent
        
        return True

class Trigger:
    """Repräsentiert einen Auslöser für Zeitlinienverschiebungen"""
    id: str
    name: str
    description: str
    level: TriggerLevel
    conditions: Dict[str, Any]
    weight: float
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert den Trigger in ein Dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "level": self.level.name,
            "conditions": self.conditions,
            "weight": self.weight,
            "category": self.category,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trigger':
        """Erstellt einen Trigger aus einem Dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            level=TriggerLevel[data.get("level", "MEDIUM")],
            conditions=data.get("conditions", {}),
            weight=data.get("weight", 1.0),
            category=data.get("category", "default"),
            metadata=data.get("metadata", {})
        )

@dataclass
class TemporalEvent:
    """Repräsentiert ein Ereignis auf einer Zeitlinie"""
    id: str
    timeline_id: str
    node_id: str
    name: str
    description: str
    timestamp: float
    impact_score: float  # Auswirkungsbewertung (0.0-1.0)
    triggers: List[str]  # Liste von Trigger-IDs
    probability: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert das TemporalEvent in ein Dictionary"""
        return {
            "id": self.id,
            "timeline_id": self.timeline_id,
            "node_id": self.node_id,
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "impact_score": self.impact_score,
            "triggers": self.triggers,
            "probability": self.probability,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalEvent':
        """Erstellt ein TemporalEvent aus einem Dictionary"""
        return cls(
            id=data["id"],
            timeline_id=data["timeline_id"],
            node_id=data["node_id"],
            name=data["name"],
            description=data["description"],
            timestamp=data["timestamp"],
            impact_score=data["impact_score"],
            triggers=data["triggers"],
            probability=data["probability"],
            metadata=data["metadata"]
        )

# Hauptklassen für die ECHO-PRIME Funktionalitäten
class TimeNodeScanner:
    """
    Detektiert kritische Ereignispunkte auf der Zeitachse 
    (Vergangenheit, Jetzt, nahe/zukünftige Szenarien)
    """
    
    def __init__(self):
        """Initialisiert den TimeNodeScanner"""
        self.scan_interval = 60  # Sekunden
        self.scanning_active = False
        self.scan_thread = None
        self.detected_nodes = {}
        self.node_callbacks = []
        logger.info("TimeNodeScanner initialisiert")
    
    def start_scanning(self):
        """Startet den kontinuierlichen Scan-Prozess"""
        if self.scanning_active:
            logger.warning("Scanning bereits aktiv")
            return
        
        self.scanning_active = True
        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()
        logger.info("TimeNodeScanner gestartet")
    
    def stop_scanning(self):
        """Stoppt den Scan-Prozess"""
        if not self.scanning_active:
            logger.warning("Scanning nicht aktiv")
            return
        
        self.scanning_active = False
        if self.scan_thread:
            self.scan_thread.join(timeout=2.0)
        logger.info("TimeNodeScanner gestoppt")
    
    def _scan_loop(self):
        """Kontinuierliche Scan-Schleife"""
        while self.scanning_active:
            try:
                self.scan_current_timeline()
                time.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Fehler im Scan-Loop: {e}")
    
    def scan_current_timeline(self) -> List[TimeNode]:
        """
        Scannt die aktuelle Zeitlinie nach kritischen Ereignispunkten
        
        Returns:
            Liste von erkannten TimeNodes
        """
        # Implementierung des Scans
        # In einer realen Implementierung würde hier die Analyse von
        # aktuellen Daten, Ereignissen und Trends stattfinden
        
        # Beispiel-Implementation
        current_time = time.time()
        detected_nodes = []
        
        # Hier würde die eigentliche Erkennung stattfinden
        # Für dieses Beispiel erstellen wir einen simulierten Node
        if current_time % 300 < 10:  # Alle ~5 Minuten einen Node erstellen
            node_id = str(uuid.uuid4())
            node = TimeNode(
                id=node_id,
                timestamp=current_time,
                description="Automatisch erkannter Zeitknoten",
                trigger_level=TriggerLevel.MEDIUM,
                probability=0.75,
                timeline_id="T0"
            )
            self.detected_nodes[node_id] = node
            detected_nodes.append(node)
            
            # Callbacks aufrufen
            for callback in self.node_callbacks:
                try:
                    callback(node)
                except Exception as e:
                    logger.error(f"Fehler im Node-Callback: {e}")
        
        return detected_nodes
    
    def register_node_callback(self, callback: Callable[[TimeNode], None]):
        """
        Registriert einen Callback für neu erkannte Nodes
        
        Args:
            callback: Funktion, die bei neuen Nodes aufgerufen wird
        """
        self.node_callbacks.append(callback)
    
    def get_nodes_in_timeframe(self, start_time: float, end_time: float) -> List[TimeNode]:
        """
        Gibt alle Nodes in einem bestimmten Zeitrahmen zurück
        
        Args:
            start_time: Startzeit (Unix-Timestamp)
            end_time: Endzeit (Unix-Timestamp)
            
        Returns:
            Liste von TimeNodes im angegebenen Zeitrahmen
        """
        return [
            node for node in self.detected_nodes.values()
            if start_time <= node.timestamp <= end_time
        ]
    
    def get_critical_nodes(self) -> List[TimeNode]:
        """
        Gibt alle kritischen Nodes zurück
        
        Returns:
            Liste von kritischen TimeNodes
        """
        return [
            node for node in self.detected_nodes.values()
            if node.trigger_level == TriggerLevel.CRITICAL
        ]

class AlternativeTimelineBuilder:
    """
    Erzeugt hypothetische Zukunftsszenarien und simuliert deren Resultate
    """
    
    def __init__(self):
        """Initialisiert den AlternativeTimelineBuilder"""
        self.timelines = {}
        self.main_timeline_id = "T0"
        self.next_alt_timeline_id = 1
        self.simulation_depth = 10  # Anzahl der zu simulierenden Schritte
        
        # Erstelle Hauptzeitlinie
        self._create_main_timeline()
        
        logger.info("AlternativeTimelineBuilder initialisiert")
    
    def _create_main_timeline(self):
        """Erstellt die Hauptzeitlinie"""
        # Timeline mit Standardparametern erstellen
        main_timeline = Timeline(
            name="Hauptzeitlinie",
            description="Die primäre Zeitlinie, die den aktuellen Realitätsverlauf repräsentiert",
            timeline_type=TimelineType.PRIMARY
        )
        
        # Die ID manuell setzen, damit sie unseren Erwartungen entspricht
        main_timeline.id = self.main_timeline_id
        self.timelines[self.main_timeline_id] = main_timeline
        logger.info(f"Hauptzeitlinie {self.main_timeline_id} erstellt")
    
    def create_alternative_timeline(self, 
                                   name: str, 
                                   description: str, 
                                   parent_timeline_id: str = "T0",
                                   branch_node_id: Optional[str] = None,
                                   probability: float = 0.5) -> Timeline:
        """
        Erstellt eine alternative Zeitlinie
        
        Args:
            name: Name der Zeitlinie
            description: Beschreibung der Zeitlinie
            parent_timeline_id: ID der Eltern-Zeitlinie
            branch_node_id: ID des Verzweigungsknotens (optional)
            probability: Wahrscheinlichkeit der Zeitlinie (0.0-1.0)
            
        Returns:
            Neu erstellte Timeline
        """
        # Prüfe, ob Eltern-Zeitlinie existiert
        if parent_timeline_id not in self.timelines:
            raise ValueError(f"Eltern-Zeitlinie {parent_timeline_id} existiert nicht")
        
        # Erstelle neue Timeline-ID
        timeline_id = f"T{self.next_alt_timeline_id}"
        self.next_alt_timeline_id += 1
        
        # Erstelle neue Timeline
        timeline = Timeline(
            id=timeline_id,
            type=TimelineType.ALTERNATIVE,
            name=name,
            description=description,
            start_time=time.time(),
            parent_timeline_id=parent_timeline_id,
            probability=probability
        )
        
        # Wenn ein Verzweigungsknoten angegeben ist, kopiere alle Knoten bis zu diesem Punkt
        if branch_node_id:
            parent_timeline = self.timelines[parent_timeline_id]
            if branch_node_id in parent_timeline.nodes:
                # Kopiere alle Knoten bis zum Verzweigungsknoten
                self._copy_nodes_until_branch(parent_timeline, timeline, branch_node_id)
        
        # Speichere Timeline
        self.timelines[timeline_id] = timeline
        logger.info(f"Alternative Zeitlinie {timeline_id} erstellt (Eltern: {parent_timeline_id})")
        
        return timeline
    

    def _copy_nodes_until_branch(self, 
                               parent_timeline: Timeline, 
                               new_timeline: Timeline, 
                               branch_node_id: str):
        """
        Kopiert alle Knoten von der Eltern-Zeitlinie bis zum Verzweigungsknoten
        
        Args:
            parent_timeline: Eltern-Zeitlinie
            new_timeline: Neue Zeitlinie
            branch_node_id: ID des Verzweigungsknotens
        """
        # Finde den Pfad vom Verzweigungsknoten zur Wurzel
        node_path = []
        current_node_id = branch_node_id
        
        while current_node_id:
            node = parent_timeline.nodes.get(current_node_id)
            if not node:
                break
            
            node_path.append(node)
            current_node_id = node.parent_node_id
        
        # Kopiere Knoten in umgekehrter Reihenfolge (von der Wurzel zum Verzweigungsknoten)
        for node in reversed(node_path):
            # Erstelle Kopie des Knotens für die neue Zeitlinie
            new_node = TimeNode(
                id=node.id,
                timestamp=node.timestamp,
                description=node.description,
                trigger_level=node.trigger_level,
                probability=node.probability,
                timeline_id=new_timeline.id,
                parent_node_id=node.parent_node_id,
                child_node_ids=[],  # Leere Liste, da wir die Struktur neu aufbauen
                metadata=node.metadata.copy()
            )
            
            # Füge Knoten zur neuen Zeitlinie hinzu
            new_timeline.nodes[new_node.id] = new_node
            
            # Aktualisiere Eltern-Kind-Beziehungen
            if new_node.parent_node_id and new_node.parent_node_id in new_timeline.nodes:
                parent_node = new_timeline.nodes[new_node.parent_node_id]
                parent_node.child_node_ids.append(new_node.id)
    
    def simulate_timeline(self, 
                        timeline_id: str, 
                        steps: int = None,
                        end_time: float = None) -> List[TimeNode]:
        """
        Simuliert eine Zeitlinie für eine bestimmte Anzahl von Schritten oder bis zu einem Endzeitpunkt
        
        Args:
            timeline_id: ID der zu simulierenden Zeitlinie
            steps: Anzahl der Simulationsschritte (optional)
            end_time: Endzeitpunkt der Simulation (optional)
            
        Returns:
            Liste der erzeugten Simulationsknoten
        """
        if timeline_id not in self.timelines:
            raise ValueError(f"Zeitlinie {timeline_id} existiert nicht")
        
        timeline = self.timelines[timeline_id]
        
        # Setze Standardwerte, falls nicht angegeben
        if steps is None:
            steps = self.simulation_depth
        
        if end_time is None:
            end_time = time.time() + (steps * 3600)  # Standard: steps Stunden in die Zukunft
        
        # Finde den letzten Knoten in der Zeitlinie
        last_nodes = self._get_leaf_nodes(timeline)
        
        # Wenn keine Knoten vorhanden sind, erstelle einen Startknoten
        if not last_nodes:
            start_node = TimeNode(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                description=f"Simulationsstart für Zeitlinie {timeline_id}",
                trigger_level=TriggerLevel.LOW,
                probability=1.0,
                timeline_id=timeline_id
            )
            timeline.nodes[start_node.id] = start_node
            last_nodes = [start_node]
        
        # Simuliere Schritte
        simulated_nodes = []
        current_step = 0
        
        while current_step < steps and time.time() < end_time:
            # Für jeden Blattknoten simuliere einen Schritt
            new_last_nodes = []
            
            for last_node in last_nodes:
                # Simuliere einen Schritt
                new_nodes = self._simulate_step(timeline, last_node)
                simulated_nodes.extend(new_nodes)
                new_last_nodes.extend(new_nodes)
            
            # Aktualisiere die Liste der letzten Knoten
            last_nodes = new_last_nodes
            current_step += 1
        
        return simulated_nodes
    
    def _get_leaf_nodes(self, timeline: Timeline) -> List[TimeNode]:
        """
        Findet alle Blattknoten in einer Zeitlinie
        
        Args:
            timeline: Zeitlinie
            
        Returns:
            Liste von Blattknoten
        """
        # Ein Blattknoten hat keine Kinder
        return [
            node for node in timeline.nodes.values()
            if not node.child_node_ids
        ]
    
    def _simulate_step(self, timeline: Timeline, parent_node: TimeNode) -> List[TimeNode]:
        """
        Simuliert einen Schritt in der Zeitlinie ausgehend von einem Elternknoten
        
        Args:
            timeline: Zeitlinie
            parent_node: Elternknoten
            
        Returns:
            Liste der erzeugten Simulationsknoten
        """
        # In einer realen Implementierung würde hier ein komplexes Modell
        # zur Simulation von Ereignissen und deren Wahrscheinlichkeiten stehen
        
        # Für dieses Beispiel erstellen wir 1-3 zufällige Folgeknoten
        num_branches = np.random.randint(1, 4)
        new_nodes = []
        
        for i in range(num_branches):
            # Erstelle neuen Knoten
            node_id = str(uuid.uuid4())
            timestamp = parent_node.timestamp + np.random.randint(3600, 86400)  # 1 Stunde bis 1 Tag später
            
            # Wahrscheinlichkeit nimmt mit jedem Verzweigungsschritt ab
            probability = parent_node.probability * (1.0 / num_branches) * np.random.uniform(0.7, 1.0)
            
            # Zufälliges Trigger-Level
            trigger_levels = list(TriggerLevel)
            trigger_level = np.random.choice(trigger_levels, p=[0.4, 0.3, 0.2, 0.1])
            
            node = TimeNode(
                id=node_id,
                timestamp=timestamp,
                description=f"Simuliertes Ereignis {i+1} nach {parent_node.description}",
                trigger_level=trigger_level,
                probability=probability,
                timeline_id=timeline.id,
                parent_node_id=parent_node.id
            )
            
            # Füge Knoten zur Zeitlinie hinzu
            timeline.nodes[node_id] = node
            parent_node.child_node_ids.append(node_id)
            new_nodes.append(node)
        
        return new_nodes
    
    def get_timeline(self, timeline_id: str) -> Optional[Timeline]:
        """
        Gibt eine Zeitlinie zurück
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            Timeline oder None, falls nicht gefunden
        """
        return self.timelines.get(timeline_id)
    
    def get_all_timelines(self) -> Dict[str, Timeline]:
        """
        Gibt alle Zeitlinien zurück
        
        Returns:
            Dictionary mit allen Zeitlinien
        """
        return self.timelines
    
    def delete_timeline(self, timeline_id: str) -> bool:
        """
        Löscht eine Zeitlinie
        
        Args:
            timeline_id: ID der zu löschenden Zeitlinie
            
        Returns:
            True, wenn erfolgreich gelöscht, sonst False
        """
        if timeline_id == self.main_timeline_id:
            logger.error("Die Hauptzeitlinie kann nicht gelöscht werden")
            return False
        
        if timeline_id in self.timelines:
            del self.timelines[timeline_id]
            logger.info(f"Zeitlinie {timeline_id} gelöscht")
            return True
        
        logger.warning(f"Zeitlinie {timeline_id} existiert nicht")
        return False
