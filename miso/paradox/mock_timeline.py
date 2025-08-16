#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Mock Timeline Klassen

Diese Datei enthält Mock-Implementierungen der Timeline-Klassen für Tests.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Optional, Set, Any
from datetime import datetime

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [MOCK-TIMELINE] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.Paradox.MockTimeline")

class MockTemporalEvent:
    """Mock-Implementierung eines temporalen Ereignisses für Tests"""
    
    def __init__(self, event_id: str, event_type: str, data: Dict[str, Any] = None):
        """
        Initialisiert ein Mock-Ereignis
        
        Args:
            event_id: ID des Ereignisses
            event_type: Typ des Ereignisses
            data: Daten des Ereignisses
        """
        self.id = event_id
        self.type = event_type
        self.data = data or {}
        self.creation_time = datetime.now()
        self.references = []
        self.metadata = {}
    
    def copy(self) -> 'MockTemporalEvent':
        """
        Erstellt eine Kopie des Ereignisses
        
        Returns:
            Kopie des Ereignisses
        """
        event_copy = MockTemporalEvent(self.id, self.type, self.data.copy())
        event_copy.creation_time = self.creation_time
        event_copy.references = self.references.copy()
        event_copy.metadata = self.metadata.copy()
        
        return event_copy

class MockTimeNode:
    """Mock-Implementierung eines Zeitknotens für Tests"""
    
    def __init__(self, node_id: str, references: List[str] = None):
        """
        Initialisiert einen Mock-Zeitknoten
        
        Args:
            node_id: ID des Zeitknotens
            references: Liste von Referenzen auf andere Zeitknoten
        """
        self.id = node_id
        self.references = references or []
        self.events = []
        self.creation_time = datetime.now()
        self.last_modified = datetime.now()
        self.metadata = {}
    
    def add_event(self, event_id: str, event_type: str, data: Dict[str, Any] = None) -> MockTemporalEvent:
        """
        Fügt ein Ereignis zum Zeitknoten hinzu
        
        Args:
            event_id: ID des Ereignisses
            event_type: Typ des Ereignisses
            data: Daten des Ereignisses
            
        Returns:
            Erstelltes Ereignis
        """
        event = MockTemporalEvent(event_id, event_type, data or {})
        self.events.append(event)
        return event
    
    def copy(self) -> 'MockTimeNode':
        """
        Erstellt eine Kopie des Zeitknotens
        
        Returns:
            Kopie des Zeitknotens
        """
        node_copy = MockTimeNode(self.id, self.references.copy())
        node_copy.creation_time = self.creation_time
        node_copy.last_modified = datetime.now()
        node_copy.metadata = self.metadata.copy()
        
        # Kopiere alle Ereignisse
        for event in self.events:
            node_copy.events.append(event.copy())
        
        return node_copy

class MockTimeline:
    """Mock-Implementierung einer Zeitlinie für Tests"""
    
    def __init__(self, timeline_id: str):
        """
        Initialisiert eine Mock-Zeitlinie
        
        Args:
            timeline_id: ID der Zeitlinie
        """
        self.id = timeline_id
        self.nodes = {}
        self.name = f"Mock Timeline {timeline_id}"
        self.creation_time = datetime.now()
        self.last_modified = datetime.now()
        self.metadata = {}
    
    def add_node(self, node_id: str, references: List[str] = None) -> MockTimeNode:
        """
        Fügt einen Zeitknoten zur Zeitlinie hinzu
        
        Args:
            node_id: ID des Zeitknotens
            references: Liste von Referenzen auf andere Zeitknoten
            
        Returns:
            Erstellter Zeitknoten
        """
        node = MockTimeNode(node_id, references or [])
        self.nodes[node_id] = node
        return node
    
    def get_nodes(self) -> List[MockTimeNode]:
        """
        Gibt alle Zeitknoten der Zeitlinie zurück
        
        Returns:
            Liste aller Zeitknoten
        """
        return list(self.nodes.values())
    
    def get_node(self, node_id: str) -> Optional[MockTimeNode]:
        """
        Gibt einen Zeitknoten anhand seiner ID zurück
        
        Args:
            node_id: ID des Zeitknotens
            
        Returns:
            Zeitknoten oder None, wenn nicht gefunden
        """
        return self.nodes.get(node_id)
    
    def copy(self) -> 'MockTimeline':
        """
        Erstellt eine Kopie der Zeitlinie
        
        Returns:
            Kopie der Zeitlinie
        """
        timeline_copy = MockTimeline(self.id)
        timeline_copy.name = self.name
        timeline_copy.creation_time = self.creation_time
        timeline_copy.last_modified = datetime.now()
        timeline_copy.metadata = self.metadata.copy()
        
        # Kopiere alle Knoten
        for node_id, node in self.nodes.items():
            timeline_copy.nodes[node_id] = node.copy()
        
        return timeline_copy
