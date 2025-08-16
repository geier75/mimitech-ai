#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - ECHO-PRIME Timeline-Implementierung

Dieses Modul implementiert die Timeline-Komponenten für die ECHO-PRIME Engine.
Es definiert die grundlegenden Datenstrukturen für Zeitlinien, Zeitknoten,
temporale Ereignisse und Trigger.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import datetime
import uuid
import copy
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.engines.echo_prime.timeline")

@dataclass
class TemporalEvent:
    """Temporales Ereignis in einer Zeitlinie"""
    name: str
    description: str
    timestamp: datetime.datetime
    data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Initialisiert zusätzliche Felder nach der Erstellung"""
        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at
    
    def update(self, 
              name: Optional[str] = None, 
              description: Optional[str] = None,
              timestamp: Optional[datetime.datetime] = None,
              data: Optional[Dict[str, Any]] = None) -> None:
        """
        Aktualisiert das Ereignis
        
        Args:
            name: Neuer Name (optional)
            description: Neue Beschreibung (optional)
            timestamp: Neuer Zeitstempel (optional)
            data: Neue Daten (optional)
        """
        if name is not None:
            self.name = name
        
        if description is not None:
            self.description = description
        
        if timestamp is not None:
            self.timestamp = timestamp
        
        if data is not None:
            self.data.update(data)
        
        self.updated_at = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert das Ereignis in ein Wörterbuch
        
        Returns:
            Wörterbuch mit den Ereignisdaten
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalEvent':
        """
        Erstellt ein Ereignis aus einem Wörterbuch
        
        Args:
            data: Wörterbuch mit den Ereignisdaten
            
        Returns:
            Neues Ereignis
        """
        event = cls(
            name=data["name"],
            description=data["description"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {}),
            id=data.get("id", str(uuid.uuid4()))
        )
        
        if "created_at" in data:
            event.created_at = datetime.datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data:
            event.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
        
        return event

@dataclass
class Trigger:
    """Trigger zwischen zwei Ereignissen in einer Zeitlinie"""
    source_event_id: str
    target_event_id: str
    trigger_type: str
    probability: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Initialisiert zusätzliche Felder nach der Erstellung"""
        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at
        
        # Validiere Wahrscheinlichkeit
        if self.probability < 0 or self.probability > 1:
            logger.warning(f"Ungültige Wahrscheinlichkeit für Trigger: {self.probability}, setze auf 1.0")
            self.probability = 1.0
    
    def update(self, 
              trigger_type: Optional[str] = None,
              probability: Optional[float] = None,
              data: Optional[Dict[str, Any]] = None) -> None:
        """
        Aktualisiert den Trigger
        
        Args:
            trigger_type: Neuer Triggertyp (optional)
            probability: Neue Wahrscheinlichkeit (optional)
            data: Neue Daten (optional)
        """
        if trigger_type is not None:
            self.trigger_type = trigger_type
        
        if probability is not None:
            # Validiere Wahrscheinlichkeit
            if probability < 0 or probability > 1:
                logger.warning(f"Ungültige Wahrscheinlichkeit für Trigger: {probability}, ignoriere")
            else:
                self.probability = probability
        
        if data is not None:
            self.data.update(data)
        
        self.updated_at = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert den Trigger in ein Wörterbuch
        
        Returns:
            Wörterbuch mit den Triggerdaten
        """
        return {
            "id": self.id,
            "source_event_id": self.source_event_id,
            "target_event_id": self.target_event_id,
            "trigger_type": self.trigger_type,
            "probability": self.probability,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trigger':
        """
        Erstellt einen Trigger aus einem Wörterbuch
        
        Args:
            data: Wörterbuch mit den Triggerdaten
            
        Returns:
            Neuer Trigger
        """
        trigger = cls(
            source_event_id=data["source_event_id"],
            target_event_id=data["target_event_id"],
            trigger_type=data["trigger_type"],
            probability=data.get("probability", 1.0),
            data=data.get("data", {}),
            id=data.get("id", str(uuid.uuid4()))
        )
        
        if "created_at" in data:
            trigger.created_at = datetime.datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data:
            trigger.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
        
        return trigger

class TimeNode:
    """
    Zeitknoten in einer Zeitlinie
    
    Ein Zeitknoten repräsentiert einen Punkt in einer Zeitlinie und kann
    mehrere Ereignisse und Trigger enthalten.
    """
    
    def __init__(self, timestamp: datetime.datetime, node_id: Optional[str] = None):
        """
        Initialisiert den Zeitknoten
        
        Args:
            timestamp: Zeitstempel des Knotens
            node_id: ID des Knotens (optional)
        """
        self.id = node_id or str(uuid.uuid4())
        self.timestamp = timestamp
        self.events = {}  # event_id -> TemporalEvent
        self.incoming_triggers = {}  # trigger_id -> Trigger
        self.outgoing_triggers = {}  # trigger_id -> Trigger
        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at
    
    def add_event(self, event: TemporalEvent) -> None:
        """
        Fügt ein Ereignis zum Zeitknoten hinzu
        
        Args:
            event: Hinzuzufügendes Ereignis
        """
        self.events[event.id] = event
        self.updated_at = datetime.datetime.now()
    
    def remove_event(self, event_id: str) -> bool:
        """
        Entfernt ein Ereignis aus dem Zeitknoten
        
        Args:
            event_id: ID des zu entfernenden Ereignisses
            
        Returns:
            True, wenn das Ereignis entfernt wurde, sonst False
        """
        if event_id in self.events:
            del self.events[event_id]
            self.updated_at = datetime.datetime.now()
            return True
        
        return False
    
    def add_incoming_trigger(self, trigger: Trigger) -> None:
        """
        Fügt einen eingehenden Trigger zum Zeitknoten hinzu
        
        Args:
            trigger: Hinzuzufügender Trigger
        """
        self.incoming_triggers[trigger.id] = trigger
        self.updated_at = datetime.datetime.now()
    
    def remove_incoming_trigger(self, trigger_id: str) -> bool:
        """
        Entfernt einen eingehenden Trigger aus dem Zeitknoten
        
        Args:
            trigger_id: ID des zu entfernenden Triggers
            
        Returns:
            True, wenn der Trigger entfernt wurde, sonst False
        """
        if trigger_id in self.incoming_triggers:
            del self.incoming_triggers[trigger_id]
            self.updated_at = datetime.datetime.now()
            return True
        
        return False
    
    def add_outgoing_trigger(self, trigger: Trigger) -> None:
        """
        Fügt einen ausgehenden Trigger zum Zeitknoten hinzu
        
        Args:
            trigger: Hinzuzufügender Trigger
        """
        self.outgoing_triggers[trigger.id] = trigger
        self.updated_at = datetime.datetime.now()
    
    def remove_outgoing_trigger(self, trigger_id: str) -> bool:
        """
        Entfernt einen ausgehenden Trigger aus dem Zeitknoten
        
        Args:
            trigger_id: ID des zu entfernenden Triggers
            
        Returns:
            True, wenn der Trigger entfernt wurde, sonst False
        """
        if trigger_id in self.outgoing_triggers:
            del self.outgoing_triggers[trigger_id]
            self.updated_at = datetime.datetime.now()
            return True
        
        return False
    
    def get_event(self, event_id: str) -> Optional[TemporalEvent]:
        """
        Gibt ein Ereignis aus dem Zeitknoten zurück
        
        Args:
            event_id: ID des Ereignisses
            
        Returns:
            Ereignis oder None, falls das Ereignis nicht existiert
        """
        return self.events.get(event_id)
    
    def get_all_events(self) -> Dict[str, TemporalEvent]:
        """
        Gibt alle Ereignisse im Zeitknoten zurück
        
        Returns:
            Wörterbuch mit allen Ereignissen
        """
        return self.events.copy()
    
    def get_incoming_trigger(self, trigger_id: str) -> Optional[Trigger]:
        """
        Gibt einen eingehenden Trigger aus dem Zeitknoten zurück
        
        Args:
            trigger_id: ID des Triggers
            
        Returns:
            Trigger oder None, falls der Trigger nicht existiert
        """
        return self.incoming_triggers.get(trigger_id)
    
    def get_all_incoming_triggers(self) -> Dict[str, Trigger]:
        """
        Gibt alle eingehenden Trigger im Zeitknoten zurück
        
        Returns:
            Wörterbuch mit allen eingehenden Triggern
        """
        return self.incoming_triggers.copy()
    
    def get_outgoing_trigger(self, trigger_id: str) -> Optional[Trigger]:
        """
        Gibt einen ausgehenden Trigger aus dem Zeitknoten zurück
        
        Args:
            trigger_id: ID des Triggers
            
        Returns:
            Trigger oder None, falls der Trigger nicht existiert
        """
        return self.outgoing_triggers.get(trigger_id)
    
    def get_all_outgoing_triggers(self) -> Dict[str, Trigger]:
        """
        Gibt alle ausgehenden Trigger im Zeitknoten zurück
        
        Returns:
            Wörterbuch mit allen ausgehenden Triggern
        """
        return self.outgoing_triggers.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert den Zeitknoten in ein Wörterbuch
        
        Returns:
            Wörterbuch mit den Zeitknotendaten
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "events": {event_id: event.to_dict() for event_id, event in self.events.items()},
            "incoming_triggers": {trigger_id: trigger.to_dict() for trigger_id, trigger in self.incoming_triggers.items()},
            "outgoing_triggers": {trigger_id: trigger.to_dict() for trigger_id, trigger in self.outgoing_triggers.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeNode':
        """
        Erstellt einen Zeitknoten aus einem Wörterbuch
        
        Args:
            data: Wörterbuch mit den Zeitknotendaten
            
        Returns:
            Neuer Zeitknoten
        """
        node = cls(
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            node_id=data.get("id", str(uuid.uuid4()))
        )
        
        # Lade Ereignisse
        for event_data in data.get("events", {}).values():
            event = TemporalEvent.from_dict(event_data)
            node.events[event.id] = event
        
        # Lade eingehende Trigger
        for trigger_data in data.get("incoming_triggers", {}).values():
            trigger = Trigger.from_dict(trigger_data)
            node.incoming_triggers[trigger.id] = trigger
        
        # Lade ausgehende Trigger
        for trigger_data in data.get("outgoing_triggers", {}).values():
            trigger = Trigger.from_dict(trigger_data)
            node.outgoing_triggers[trigger.id] = trigger
        
        if "created_at" in data:
            node.created_at = datetime.datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data:
            node.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
        
        return node

class Timeline:
    """
    Zeitlinie
    
    Eine Zeitlinie repräsentiert eine Sequenz von Ereignissen und Triggern,
    die in temporaler Reihenfolge angeordnet sind.
    """
    
    def __init__(self, name: str, description: Optional[str] = None, timeline_id: Optional[str] = None):
        """
        Initialisiert die Zeitlinie
        
        Args:
            name: Name der Zeitlinie
            description: Beschreibung der Zeitlinie (optional)
            timeline_id: ID der Zeitlinie (optional)
        """
        self.id = timeline_id or str(uuid.uuid4())
        self.name = name
        self.description = description or ""
        self.events = {}  # event_id -> TemporalEvent
        self.triggers = {}  # trigger_id -> Trigger
        self.nodes = {}  # node_id -> TimeNode
        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at
    
    def add_event(self, event: TemporalEvent) -> None:
        """
        Fügt ein Ereignis zur Zeitlinie hinzu
        
        Args:
            event: Hinzuzufügendes Ereignis
        """
        # Füge Ereignis zur Ereignisliste hinzu
        self.events[event.id] = event
        
        # Finde oder erstelle Zeitknoten für das Ereignis
        node_timestamp = event.timestamp.replace(microsecond=0)
        node = self._get_or_create_node(node_timestamp)
        
        # Füge Ereignis zum Zeitknoten hinzu
        node.add_event(event)
        
        self.updated_at = datetime.datetime.now()
    
    def remove_event(self, event_id: str) -> bool:
        """
        Entfernt ein Ereignis aus der Zeitlinie
        
        Args:
            event_id: ID des zu entfernenden Ereignisses
            
        Returns:
            True, wenn das Ereignis entfernt wurde, sonst False
        """
        if event_id not in self.events:
            return False
        
        # Hole Ereignis
        event = self.events[event_id]
        
        # Entferne Ereignis aus der Ereignisliste
        del self.events[event_id]
        
        # Finde Zeitknoten für das Ereignis
        node_timestamp = event.timestamp.replace(microsecond=0)
        node = self.nodes.get(self._get_node_id(node_timestamp))
        
        if node:
            # Entferne Ereignis aus dem Zeitknoten
            node.remove_event(event_id)
            
            # Entferne Zeitknoten, wenn er leer ist
            if not node.events and not node.incoming_triggers and not node.outgoing_triggers:
                del self.nodes[node.id]
        
        # Entferne alle Trigger, die mit dem Ereignis verbunden sind
        trigger_ids_to_remove = []
        
        for trigger_id, trigger in self.triggers.items():
            if trigger.source_event_id == event_id or trigger.target_event_id == event_id:
                trigger_ids_to_remove.append(trigger_id)
        
        for trigger_id in trigger_ids_to_remove:
            self.remove_trigger(trigger_id)
        
        self.updated_at = datetime.datetime.now()
        
        return True
    
    def add_trigger(self, trigger: Trigger) -> None:
        """
        Fügt einen Trigger zur Zeitlinie hinzu
        
        Args:
            trigger: Hinzuzufügender Trigger
        """
        # Überprüfe, ob die Ereignisse existieren
        if trigger.source_event_id not in self.events:
            raise ValueError(f"Quellereignis mit ID {trigger.source_event_id} nicht gefunden")
        
        if trigger.target_event_id not in self.events:
            raise ValueError(f"Zielereignis mit ID {trigger.target_event_id} nicht gefunden")
        
        # Füge Trigger zur Triggerliste hinzu
        self.triggers[trigger.id] = trigger
        
        # Hole Ereignisse
        source_event = self.events[trigger.source_event_id]
        target_event = self.events[trigger.target_event_id]
        
        # Finde oder erstelle Zeitknoten für die Ereignisse
        source_node_timestamp = source_event.timestamp.replace(microsecond=0)
        target_node_timestamp = target_event.timestamp.replace(microsecond=0)
        
        source_node = self._get_or_create_node(source_node_timestamp)
        target_node = self._get_or_create_node(target_node_timestamp)
        
        # Füge Trigger zu den Zeitknoten hinzu
        source_node.add_outgoing_trigger(trigger)
        target_node.add_incoming_trigger(trigger)
        
        self.updated_at = datetime.datetime.now()
    
    def remove_trigger(self, trigger_id: str) -> bool:
        """
        Entfernt einen Trigger aus der Zeitlinie
        
        Args:
            trigger_id: ID des zu entfernenden Triggers
            
        Returns:
            True, wenn der Trigger entfernt wurde, sonst False
        """
        if trigger_id not in self.triggers:
            return False
        
        # Hole Trigger
        trigger = self.triggers[trigger_id]
        
        # Entferne Trigger aus der Triggerliste
        del self.triggers[trigger_id]
        
        # Hole Ereignisse
        source_event = self.events.get(trigger.source_event_id)
        target_event = self.events.get(trigger.target_event_id)
        
        # Entferne Trigger aus den Zeitknoten
        if source_event:
            source_node_timestamp = source_event.timestamp.replace(microsecond=0)
            source_node = self.nodes.get(self._get_node_id(source_node_timestamp))
            
            if source_node:
                source_node.remove_outgoing_trigger(trigger_id)
                
                # Entferne Zeitknoten, wenn er leer ist
                if not source_node.events and not source_node.incoming_triggers and not source_node.outgoing_triggers:
                    del self.nodes[source_node.id]
        
        if target_event:
            target_node_timestamp = target_event.timestamp.replace(microsecond=0)
            target_node = self.nodes.get(self._get_node_id(target_node_timestamp))
            
            if target_node:
                target_node.remove_incoming_trigger(trigger_id)
                
                # Entferne Zeitknoten, wenn er leer ist
                if not target_node.events and not target_node.incoming_triggers and not target_node.outgoing_triggers:
                    del self.nodes[target_node.id]
        
        self.updated_at = datetime.datetime.now()
        
        return True
    
    def get_event(self, event_id: str) -> Optional[TemporalEvent]:
        """
        Gibt ein Ereignis aus der Zeitlinie zurück
        
        Args:
            event_id: ID des Ereignisses
            
        Returns:
            Ereignis oder None, falls das Ereignis nicht existiert
        """
        return self.events.get(event_id)
    
    def get_trigger(self, trigger_id: str) -> Optional[Trigger]:
        """
        Gibt einen Trigger aus der Zeitlinie zurück
        
        Args:
            trigger_id: ID des Triggers
            
        Returns:
            Trigger oder None, falls der Trigger nicht existiert
        """
        return self.triggers.get(trigger_id)
    
    def get_node(self, node_id: str) -> Optional[TimeNode]:
        """
        Gibt einen Zeitknoten aus der Zeitlinie zurück
        
        Args:
            node_id: ID des Zeitknotens
            
        Returns:
            Zeitknoten oder None, falls der Zeitknoten nicht existiert
        """
        return self.nodes.get(node_id)
    
    def get_node_by_timestamp(self, timestamp: datetime.datetime) -> Optional[TimeNode]:
        """
        Gibt einen Zeitknoten basierend auf einem Zeitstempel zurück
        
        Args:
            timestamp: Zeitstempel des Zeitknotens
            
        Returns:
            Zeitknoten oder None, falls der Zeitknoten nicht existiert
        """
        node_timestamp = timestamp.replace(microsecond=0)
        node_id = self._get_node_id(node_timestamp)
        return self.nodes.get(node_id)
    
    def get_events_by_timerange(self, 
                               start_time: datetime.datetime, 
                               end_time: datetime.datetime) -> List[TemporalEvent]:
        """
        Gibt alle Ereignisse in einem Zeitbereich zurück
        
        Args:
            start_time: Startzeit des Zeitbereichs
            end_time: Endzeit des Zeitbereichs
            
        Returns:
            Liste der Ereignisse im Zeitbereich
        """
        return [
            event
            for event in self.events.values()
            if start_time <= event.timestamp <= end_time
        ]
    
    def get_triggers_by_type(self, trigger_type: str) -> List[Trigger]:
        """
        Gibt alle Trigger eines bestimmten Typs zurück
        
        Args:
            trigger_type: Typ der Trigger
            
        Returns:
            Liste der Trigger des angegebenen Typs
        """
        return [
            trigger
            for trigger in self.triggers.values()
            if trigger.trigger_type == trigger_type
        ]
    
    def get_all_nodes_sorted(self) -> List[TimeNode]:
        """
        Gibt alle Zeitknoten sortiert nach Zeitstempel zurück
        
        Returns:
            Liste der Zeitknoten
        """
        return sorted(self.nodes.values(), key=lambda node: node.timestamp)
    
    def clone(self, new_name: Optional[str] = None, new_description: Optional[str] = None) -> 'Timeline':
        """
        Erstellt eine Kopie der Zeitlinie
        
        Args:
            new_name: Name der neuen Zeitlinie (optional)
            new_description: Beschreibung der neuen Zeitlinie (optional)
            
        Returns:
            Kopie der Zeitlinie
        """
        # Erstelle neue Zeitlinie
        clone = Timeline(
            name=new_name or f"Kopie von {self.name}",
            description=new_description or self.description
        )
        
        # Kopiere Ereignisse
        event_id_map = {}  # Altes Event-ID -> Neues Event-ID
        
        for event in self.events.values():
            # Erstelle Kopie des Ereignisses
            event_copy = TemporalEvent(
                name=event.name,
                description=event.description,
                timestamp=event.timestamp,
                data=copy.deepcopy(event.data)
            )
            
            # Speichere ID-Zuordnung
            event_id_map[event.id] = event_copy.id
            
            # Füge Ereignis zur neuen Zeitlinie hinzu
            clone.add_event(event_copy)
        
        # Kopiere Trigger
        for trigger in self.triggers.values():
            # Überprüfe, ob die Ereignisse existieren
            if trigger.source_event_id in event_id_map and trigger.target_event_id in event_id_map:
                # Erstelle Kopie des Triggers
                trigger_copy = Trigger(
                    source_event_id=event_id_map[trigger.source_event_id],
                    target_event_id=event_id_map[trigger.target_event_id],
                    trigger_type=trigger.trigger_type,
                    probability=trigger.probability,
                    data=copy.deepcopy(trigger.data)
                )
                
                # Füge Trigger zur neuen Zeitlinie hinzu
                clone.add_trigger(trigger_copy)
        
        return clone
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert die Zeitlinie in ein Wörterbuch
        
        Returns:
            Wörterbuch mit den Zeitliniendaten
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "events": {event_id: event.to_dict() for event_id, event in self.events.items()},
            "triggers": {trigger_id: trigger.to_dict() for trigger_id, trigger in self.triggers.items()},
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Timeline':
        """
        Erstellt eine Zeitlinie aus einem Wörterbuch
        
        Args:
            data: Wörterbuch mit den Zeitliniendaten
            
        Returns:
            Neue Zeitlinie
        """
        timeline = cls(
            name=data["name"],
            description=data.get("description", ""),
            timeline_id=data.get("id", str(uuid.uuid4()))
        )
        
        # Lade Ereignisse
        for event_data in data.get("events", {}).values():
            event = TemporalEvent.from_dict(event_data)
            timeline.events[event.id] = event
        
        # Lade Trigger
        for trigger_data in data.get("triggers", {}).values():
            trigger = Trigger.from_dict(trigger_data)
            timeline.triggers[trigger.id] = trigger
        
        # Lade Zeitknoten
        for node_data in data.get("nodes", {}).values():
            node = TimeNode.from_dict(node_data)
            timeline.nodes[node.id] = node
        
        if "created_at" in data:
            timeline.created_at = datetime.datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data:
            timeline.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
        
        return timeline
    
    def _get_or_create_node(self, timestamp: datetime.datetime) -> TimeNode:
        """
        Gibt einen existierenden Zeitknoten zurück oder erstellt einen neuen
        
        Args:
            timestamp: Zeitstempel des Zeitknotens
            
        Returns:
            Zeitknoten
        """
        node_id = self._get_node_id(timestamp)
        
        if node_id not in self.nodes:
            self.nodes[node_id] = TimeNode(timestamp, node_id)
        
        return self.nodes[node_id]
    
    def _get_node_id(self, timestamp: datetime.datetime) -> str:
        """
        Generiert eine ID für einen Zeitknoten basierend auf dem Zeitstempel
        
        Args:
            timestamp: Zeitstempel des Zeitknotens
            
        Returns:
            ID des Zeitknotens
        """
        return f"node_{timestamp.isoformat()}"
