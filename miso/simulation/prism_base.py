#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - PRISM-Base

Basisklassen und gemeinsame Typen für die PRISM-Engine und verwandte Module.
Dient zur Vermeidung zirkulärer Importe zwischen den PRISM-Modulen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import uuid
from pathlib import Path

# Konfiguriere Logging
logger = logging.getLogger("MISO.simulation.prism_base")

# Prüfen, ob Apple Silicon verfügbar ist
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine

# Gemeinsame Konfigurationsklassen
@dataclass
class SimulationConfig:
    """Konfiguration für Simulationen"""
    iterations: int = 100
    time_steps: int = 10
    seed: Optional[int] = None
    precision: str = "float32"
    use_mlx: bool = True
    device: str = "auto"
    optimize_for_apple_silicon: bool = is_apple_silicon
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert die Konfiguration in ein Dictionary"""
        return {
            "iterations": self.iterations,
            "time_steps": self.time_steps,
            "seed": self.seed,
            "precision": self.precision,
            "use_mlx": self.use_mlx,
            "device": self.device,
            "optimize_for_apple_silicon": self.optimize_for_apple_silicon
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimulationConfig':
        """Erstellt eine Konfiguration aus einem Dictionary"""
        return cls(
            iterations=config_dict.get("iterations", 100),
            time_steps=config_dict.get("time_steps", 10),
            seed=config_dict.get("seed"),
            precision=config_dict.get("precision", "float32"),
            use_mlx=config_dict.get("use_mlx", True),
            device=config_dict.get("device", "auto"),
            optimize_for_apple_silicon=config_dict.get("optimize_for_apple_silicon", is_apple_silicon)
        )

# Gemeinsame Enums
class SimulationStatus(Enum):
    """Status einer Simulation"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class TimelineType(Enum):
    """Typ einer Zeitlinie"""
    MAIN = auto()
    FORK = auto()
    MERGED = auto()
    HYPOTHETICAL = auto()
    PARADOX = auto()

# Gemeinsame Basisklassen
class TimeNode:
    """Basisklasse für Zeitknoten"""
    
    def __init__(self, node_id: str = None, timestamp: Any = None, data: Dict[str, Any] = None):
        """
        Initialisiert einen Zeitknoten
        
        Args:
            node_id: ID des Knotens (wird automatisch generiert, wenn nicht angegeben)
            timestamp: Zeitstempel des Knotens
            data: Daten des Knotens
        """
        self.id = node_id or str(uuid.uuid4())
        self.timestamp = timestamp
        self.data = data or {}
        self.connections = []
    
    def add_connection(self, target_node_id: str, weight: float = 1.0, metadata: Dict[str, Any] = None):
        """
        Fügt eine Verbindung zu einem anderen Knoten hinzu
        
        Args:
            target_node_id: ID des Zielknotens
            weight: Gewicht der Verbindung
            metadata: Metadaten für die Verbindung
        """
        self.connections.append({
            "target_id": target_node_id,
            "weight": weight,
            "metadata": metadata or {}
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert den Knoten in ein Dictionary"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "data": self.data,
            "connections": self.connections
        }
    
    @classmethod
    def from_dict(cls, node_dict: Dict[str, Any]) -> 'TimeNode':
        """Erstellt einen Knoten aus einem Dictionary"""
        node = cls(
            node_id=node_dict.get("id"),
            timestamp=node_dict.get("timestamp"),
            data=node_dict.get("data", {})
        )
        node.connections = node_dict.get("connections", [])
        return node

class Timeline:
    """Basisklasse für Zeitlinien"""
    
    def __init__(self, timeline_id: str = None, name: str = None, description: str = None, 
                 timeline_type: TimelineType = TimelineType.MAIN):
        """
        Initialisiert eine Zeitlinie
        
        Args:
            timeline_id: ID der Zeitlinie (wird automatisch generiert, wenn nicht angegeben)
            name: Name der Zeitlinie
            description: Beschreibung der Zeitlinie
            timeline_type: Typ der Zeitlinie
        """
        self.id = timeline_id or str(uuid.uuid4())
        self.name = name or f"Timeline-{self.id[:8]}"
        self.description = description or ""
        self.type = timeline_type
        self.nodes = {}
        self.root_node_id = None
        self.metadata = {}
        self.events = []
        self.triggers = []
    
    def add_node(self, node: TimeNode, is_root: bool = False):
        """
        Fügt einen Knoten zur Zeitlinie hinzu
        
        Args:
            node: Hinzuzufügender Knoten
            is_root: Gibt an, ob der Knoten der Wurzelknoten sein soll
        """
        self.nodes[node.id] = node
        if is_root or self.root_node_id is None:
            self.root_node_id = node.id
    
    def get_node(self, node_id: str) -> Optional[TimeNode]:
        """
        Gibt einen Knoten anhand seiner ID zurück
        
        Args:
            node_id: ID des Knotens
            
        Returns:
            Der Knoten oder None, falls nicht gefunden
        """
        return self.nodes.get(node_id)
    
    def add_event(self, event: Dict[str, Any]):
        """
        Fügt ein Ereignis zur Zeitlinie hinzu
        
        Args:
            event: Hinzuzufügendes Ereignis
        """
        self.events.append(event)
    
    def add_trigger(self, trigger: Dict[str, Any]):
        """
        Fügt einen Trigger zur Zeitlinie hinzu
        
        Args:
            trigger: Hinzuzufügender Trigger
        """
        self.triggers.append(trigger)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert die Zeitlinie in ein Dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.name,
            "root_node_id": self.root_node_id,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "metadata": self.metadata,
            "events": self.events,
            "triggers": self.triggers
        }
    
    @classmethod
    def from_dict(cls, timeline_dict: Dict[str, Any]) -> 'Timeline':
        """Erstellt eine Zeitlinie aus einem Dictionary"""
        timeline_type_str = timeline_dict.get("type", "MAIN")
        try:
            timeline_type = TimelineType[timeline_type_str]
        except KeyError:
            timeline_type = TimelineType.MAIN
            logger.warning(f"Unbekannter Zeitlinientyp: {timeline_type_str}, verwende MAIN")
        
        timeline = cls(
            timeline_id=timeline_dict.get("id"),
            name=timeline_dict.get("name"),
            description=timeline_dict.get("description"),
            timeline_type=timeline_type
        )
        
        timeline.root_node_id = timeline_dict.get("root_node_id")
        timeline.metadata = timeline_dict.get("metadata", {})
        timeline.events = timeline_dict.get("events", [])
        timeline.triggers = timeline_dict.get("triggers", [])
        
        # Knoten hinzufügen
        for node_id, node_dict in timeline_dict.get("nodes", {}).items():
            timeline.nodes[node_id] = TimeNode.from_dict(node_dict)
        
        return timeline

# Gemeinsame Hilfsfunktionen
def calculate_probability(values: List[float]) -> float:
    """
    Berechnet eine normalisierte Wahrscheinlichkeit aus einer Liste von Werten
    
    Args:
        values: Liste von Werten
        
    Returns:
        Normalisierte Wahrscheinlichkeit zwischen 0 und 1
    """
    if not values:
        return 0.0
    
    # Negative Werte auf 0 setzen
    non_negative = [max(0, v) for v in values]
    
    # Summe berechnen
    total = sum(non_negative)
    
    # Wenn die Summe 0 ist, gib 0 zurück
    if total == 0:
        return 0.0
    
    # Normalisieren
    return sum(non_negative) / (len(non_negative) * max(non_negative))

def sigmoid(x: float) -> float:
    """
    Sigmoid-Funktion
    
    Args:
        x: Eingabewert
        
    Returns:
        Sigmoid-Wert zwischen 0 und 1
    """
    return 1 / (1 + np.exp(-x))
