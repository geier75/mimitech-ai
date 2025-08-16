#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Timeline-Basisdatentypen

Dieses Modul definiert die gemeinsamen Basisdatentypen für Zeitlinien und Zeitknoten,
die von ECHO-PRIME und PRISM-Engine gemeinsam verwendet werden.
Dies verhindert zirkuläre Importe zwischen den beiden Systemen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import uuid
import time
import enum
import logging
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime

# Konfiguriere Logging
logger = logging.getLogger("MISO.core.timeline_base")

class TimelineType(enum.Enum):
    """Typ einer Zeitlinie"""
    PRIMARY = "primary"          # Hauptzeitlinie
    ALTERNATIVE = "alternative"  # Alternative Zeitlinie
    SIMULATION = "simulation"    # Simulierte Zeitlinie
    QUANTUM = "quantum"          # Quantenüberlagerung 
    PARADOX = "paradox"          # Zeitlinie mit Paradoxon

class TriggerLevel(enum.Enum):
    """Level für Trigger in Zeitknoten"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CRITICAL = 1.0

class TimeNode:
    """Basis-Zeitknoten
    
    Ein TimeNode repräsentiert einen Punkt in einer Zeitlinie.
    """
    
    def __init__(self, 
                 description: str,
                 timestamp: Optional[float] = None,
                 parent_id: Optional[str] = None,
                 probability: float = 0.8,
                 trigger_level: TriggerLevel = TriggerLevel.MEDIUM,
                 metadata: Dict[str, Any] = None):
        """
        Initialisiert einen Zeitknoten
        
        Args:
            description: Beschreibung des Knotens
            timestamp: Zeitstempel des Knotens (optional, Standard: aktuelle Zeit)
            parent_id: ID des Elternknotens (optional)
            probability: Wahrscheinlichkeit des Knotens (0.0-1.0)
            trigger_level: Trigger-Level des Knotens
            metadata: Zusätzliche Metadaten
        """
        self.id = str(uuid.uuid4())
        self.description = description
        self.timestamp = timestamp or time.time()
        self.creation_time = time.time()
        self.parent_id = parent_id
        self.children = set()  # Set von Kind-IDs
        self.probability = min(max(probability, 0.0), 1.0)  # Begrenzt auf 0.0-1.0
        self.trigger_level = trigger_level
        self.metadata = metadata or {}
        
    def add_child(self, child_id: str):
        """
        Fügt ein Kind hinzu
        
        Args:
            child_id: ID des Kindknotens
        """
        self.children.add(child_id)
        
    def remove_child(self, child_id: str):
        """
        Entfernt ein Kind
        
        Args:
            child_id: ID des Kindknotens
        """
        if child_id in self.children:
            self.children.remove(child_id)
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert den Knoten in ein Dictionary
        
        Returns:
            Dictionary-Repräsentation des Knotens
        """
        return {
            "id": self.id,
            "description": self.description,
            "timestamp": self.timestamp,
            "creation_time": self.creation_time,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "probability": self.probability,
            "trigger_level": self.trigger_level.name,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeNode':
        """
        Erstellt einen Knoten aus einem Dictionary
        
        Args:
            data: Dictionary-Repräsentation des Knotens
            
        Returns:
            Erstellter TimeNode
        """
        node = cls(
            description=data.get("description", ""),
            timestamp=data.get("timestamp"),
            parent_id=data.get("parent_id"),
            probability=data.get("probability", 0.8),
            trigger_level=TriggerLevel[data.get("trigger_level", "MEDIUM")],
            metadata=data.get("metadata", {})
        )
        node.id = data.get("id", str(uuid.uuid4()))
        node.creation_time = data.get("creation_time", time.time())
        node.children = set(data.get("children", []))
        return node

class Timeline:
    """Basis-Zeitlinie
    
    Eine Timeline repräsentiert eine Sequenz von TimeNodes.
    """
    
    def __init__(self, 
                 name: str, 
                 description: str,
                 timeline_type: TimelineType = TimelineType.PRIMARY,
                 metadata: Dict[str, Any] = None):
        """
        Initialisiert eine Zeitlinie
        
        Args:
            name: Name der Zeitlinie
            description: Beschreibung der Zeitlinie
            timeline_type: Typ der Zeitlinie
            metadata: Zusätzliche Metadaten
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.creation_time = time.time()
        self.last_modified = time.time()
        self.timeline_type = timeline_type
        self.metadata = metadata or {}
        self.nodes = {}  # Dictionary von node_id -> TimeNode
        self.root_nodes = set()  # Set von Root-Node-IDs
        
    def add_node(self, node: TimeNode, parent_id: Optional[str] = None) -> str:
        """
        Fügt einen Knoten zur Zeitlinie hinzu
        
        Args:
            node: Hinzuzufügender Knoten
            parent_id: ID des Elternknotens (optional)
            
        Returns:
            ID des hinzugefügten Knotens
        """
        self.nodes[node.id] = node
        
        if parent_id is not None:
            node.parent_id = parent_id
            if parent_id in self.nodes:
                self.nodes[parent_id].add_child(node.id)
        
        if node.parent_id is None:
            self.root_nodes.add(node.id)
            
        self.last_modified = time.time()
        return node.id
    
    def remove_node(self, node_id: str, recursive: bool = False) -> bool:
        """
        Entfernt einen Knoten aus der Zeitlinie
        
        Args:
            node_id: ID des zu entfernenden Knotens
            recursive: Wenn True, werden auch alle Kindknoten entfernt
            
        Returns:
            True, wenn der Knoten erfolgreich entfernt wurde, sonst False
        """
        if node_id not in self.nodes:
            return False
            
        node = self.nodes[node_id]
        
        # Entferne aus Root-Nodes, falls vorhanden
        if node_id in self.root_nodes:
            self.root_nodes.remove(node_id)
            
        # Entferne aus Eltern-Knoten
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].remove_child(node_id)
            
        # Behandle Kindknoten
        if recursive:
            # Kopiere Liste, um Änderungen während der Iteration zu vermeiden
            children = list(node.children)
            for child_id in children:
                self.remove_node(child_id, recursive=True)
        else:
            # Setze Kinder als Root-Nodes
            for child_id in node.children:
                if child_id in self.nodes:
                    self.nodes[child_id].parent_id = None
                    self.root_nodes.add(child_id)
        
        # Entferne den Knoten selbst
        del self.nodes[node_id]
        self.last_modified = time.time()
        return True
        
    def get_node(self, node_id: str) -> Optional[TimeNode]:
        """
        Gibt einen Knoten zurück
        
        Args:
            node_id: ID des Knotens
            
        Returns:
            TimeNode oder None, falls nicht gefunden
        """
        return self.nodes.get(node_id)
        
    def get_all_nodes(self) -> Dict[str, TimeNode]:
        """
        Gibt alle Knoten zurück
        
        Returns:
            Dictionary aller Knoten (node_id -> TimeNode)
        """
        return self.nodes
        
    def get_root_nodes(self) -> List[TimeNode]:
        """
        Gibt alle Root-Knoten zurück
        
        Returns:
            Liste aller Root-Knoten
        """
        return [self.nodes[node_id] for node_id in self.root_nodes if node_id in self.nodes]
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert die Zeitlinie in ein Dictionary
        
        Returns:
            Dictionary-Repräsentation der Zeitlinie
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "creation_time": self.creation_time,
            "last_modified": self.last_modified,
            "timeline_type": self.timeline_type.name,
            "metadata": self.metadata,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_nodes": list(self.root_nodes)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Timeline':
        """
        Erstellt eine Zeitlinie aus einem Dictionary
        
        Args:
            data: Dictionary-Repräsentation der Zeitlinie
            
        Returns:
            Erstellte Timeline
        """
        timeline = cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            timeline_type=TimelineType[data.get("timeline_type", "PRIMARY")],
            metadata=data.get("metadata", {})
        )
        timeline.id = data.get("id", str(uuid.uuid4()))
        timeline.creation_time = data.get("creation_time", time.time())
        timeline.last_modified = data.get("last_modified", time.time())
        timeline.root_nodes = set(data.get("root_nodes", []))
        
        # Erstelle zuerst alle Knoten
        nodes_data = data.get("nodes", {})
        for node_id, node_data in nodes_data.items():
            node = TimeNode.from_dict(node_data)
            timeline.nodes[node_id] = node
            
        return timeline
