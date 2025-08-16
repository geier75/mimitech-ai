#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - ECHO-PRIME Controller

Dieses Hauptmodul integriert alle Komponenten des ECHO-PRIME Systems:
- TimeNodeScanner
- AlternativeTimelineBuilder
- TriggerMatrixAnalyzer
- TimelineFeedbackLoop
- TemporalIntegrityGuard
- QTM_Modulator

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set, TYPE_CHECKING
import numpy as np
from datetime import datetime

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.timeline.echo_prime_controller")

# Importiere gemeinsame Basisklassen
from miso.core.timeline_base import Timeline, TimeNode, TimelineType, TriggerLevel

# Importiere ECHO-PRIME Komponenten
from miso.timeline.echo_prime import (
    TimeNode, Timeline, Trigger, TemporalEvent, TriggerLevel,
    TimeNodeScanner, AlternativeTimelineBuilder
)
from miso.timeline.trigger_matrix_analyzer import TriggerMatrixAnalyzer
# Importiere aus dem Hauptverzeichnis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from timeline_feedback_loop import TimelineFeedbackLoop, StrategicRecommendation
from miso.timeline.temporal_integrity_guard import TemporalIntegrityGuard, ParadoxDetection, ParadoxType
from miso.timeline.qtm_modulator import QTM_Modulator, QuantumTimeEffect, QuantumState

# Optionale Integrationen
PRISM_AVAILABLE = False

def get_prism_simulator():
    """
    Lazy-Loading Funktion für die PrismEngine
    Verhindert zirkuläre Importe zwischen ECHO-PRIME und PRISM
    
    Returns:
        PrismEngine-Klasse oder None, falls nicht verfügbar
    """
    global PRISM_AVAILABLE
    
    try:
        # Verwende importlib für maximale Kontrolle über den Import-Prozess
        import importlib
        prism_engine_module = importlib.import_module("miso.simulation.prism_engine")
        PRISM_AVAILABLE = True
        return getattr(prism_engine_module, "PrismEngine", None)
    except ImportError as e:
        PRISM_AVAILABLE = False
        logger.warning(f"PRISM-Modul nicht verfügbar, einige Funktionen sind eingeschränkt: {e}")
        return None

# Dummy-Klasse als Fallback
class PrismSimulator:
    """Dummy-Implementierung für PrismEngine"""
    def __init__(self, *args, **kwargs):
        logger.info("Verwende Dummy-Implementierung für PrismEngine")
        
    def initialize(self):
        return True
        
    def simulate(self, *args, **kwargs):
        return {"status": "simulation_skipped", "reason": "PRISM-Engine nicht verfügbar"}

# Prüfe Verfügbarkeit der VX-CHRONOS-Bridge
VX_CHRONOS_BRIDGE_AVAILABLE = False

def get_chronos_bridge():
    """
    Lazy-Loading Funktion für die ChronosEchoBridge
    Verhindert zirkuläre Importe zwischen ECHO-PRIME und VX-CHRONOS
    
    Returns:
        get_bridge-Funktion der ChronosEchoBridge oder None, falls nicht verfügbar
    """
    try:
        from miso.vxor.chronos_echo_prime_bridge import get_bridge
        global VX_CHRONOS_BRIDGE_AVAILABLE
        VX_CHRONOS_BRIDGE_AVAILABLE = True
        return get_bridge
    except ImportError as e:
        logger.warning(f"VX-CHRONOS-Bridge konnte nicht importiert werden: {e}")
        return None

# Prüfe Verfügbarkeit des NEXUS-OS
NEXUS_AVAILABLE = False

try:
    import importlib.util
    nexus_spec = importlib.util.find_spec("miso.core.nexus_os")
    NEXUS_AVAILABLE = nexus_spec is not None
except ImportError as e:
    logger.warning(f"NEXUS-OS nicht verfügbar: {e}")
    NEXUS_AVAILABLE = False
    
def get_nexus_core():
    """
    Lazy-Loading Funktion für das NexusCore
    
    Returns:
        NexusCore-Klasse oder None, falls nicht verfügbar
    """
    if not NEXUS_AVAILABLE:
        return None
        
    try:
        import importlib
        nexus_module = importlib.import_module("miso.core.nexus_os")
        return getattr(nexus_module, "NexusCore", None)
    except ImportError:
        logger.error("NexusCore konnte nicht importiert werden.")
        return None
    
# Dummy-Klasse als Fallback
class NexusCore:
    """Dummy-Implementierung für NexusCore"""
    def __init__(self):
        logger.info("Verwende Dummy-Implementierung für NexusCore")
        
    def initialize(self):
        return True
        
    def register_component(self, *args, **kwargs):
        return True

class EchoPrimeController:
    """Hauptcontroller für das ECHO-PRIME System"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert den ECHO-PRIME Controller
        
        Args:
            config_path: Pfad zur Konfigurationsdatei (optional)
        """
        self.config = self._load_config(config_path)
        
        # Initialisiere Komponenten
        self.node_scanner = TimeNodeScanner()
        self.timeline_builder = AlternativeTimelineBuilder()
        self.trigger_analyzer = TriggerMatrixAnalyzer()
        self.feedback_loop = TimelineFeedbackLoop()
        self.integrity_guard = TemporalIntegrityGuard()
        self.qtm_modulator = QTM_Modulator()
        
        # Speicher für Zeitlinien und Trigger
        self.timelines = {}
        self.triggers = {}
        
        # Integrationen
        self.prism_simulator = None
        self.nexus_core = None
        
        if self.config.get("enable_prism", True):
            # Verwende Lazy-Loading, um zirkuläre Importe zu vermeiden
            PrismSimulator = get_prism_simulator()
            if PrismSimulator:
                self.prism_simulator = PrismSimulator()
                logger.info("PRISM-Engine erfolgreich initialisiert")
            else:
                logger.warning("PRISM-Engine konnte nicht geladen werden")
            
        if NEXUS_AVAILABLE and self.config.get("enable_nexus", True):
            # Verwende Lazy-Loading für NexusCore
            NexusCoreClass = get_nexus_core()
            if NexusCoreClass:
                self.nexus_core = NexusCoreClass()
            else:
                self.nexus_core = NexusCore()
        
        logger.info("ECHO-PRIME Controller initialisiert")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Lädt die Konfiguration aus einer Datei oder verwendet Standardwerte
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            
        Returns:
            Dictionary mit Konfigurationseinstellungen
        """
        default_config = {
            "enable_prism": True,
            "enable_nexus": True,
            "auto_integrity_check": True,
            "quantum_effects_enabled": True,
            "max_alternative_timelines": 10,
            "min_node_probability": 0.1,
            "max_paradox_severity": 0.8,
            "data_sources": ["system", "user", "external"],
            "storage_path": os.path.join(os.path.dirname(__file__), "data")
        }
        
        if not config_path:
            logger.info("Keine Konfigurationsdatei angegeben, verwende Standardwerte")
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                
            # Kombiniere Benutzer- und Standardkonfiguration
            config = {**default_config, **user_config}
            logger.info(f"Konfiguration aus {config_path} geladen")
            return config
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            logger.info("Verwende Standardwerte")
            return default_config
    
    # Zeitlinien-Management
    
    def create_timeline(self, 
                      name: str, 
                      description: str, 
                      initial_nodes: List[TimeNode] = None) -> Timeline:
        """
        Erstellt eine neue Zeitlinie
        
        Args:
            name: Name der Zeitlinie
            description: Beschreibung der Zeitlinie
            initial_nodes: Liste von initialen Zeitknoten (optional)
            
        Returns:
            Erstellte Zeitlinie
        """
        timeline_id = str(uuid.uuid4())
        
        # Verwende die Basisklasse aus timeline_base
        timeline = Timeline(
            name=name,
            description=description,
            timeline_type=TimelineType.PRIMARY
        )
        
        # Setze ID manuell nach der Konstruktion
        timeline.id = timeline_id
        
        # Füge initiale Knoten hinzu, falls vorhanden
        if initial_nodes:
            for node in initial_nodes:
                timeline.add_node(node)
                
        # Speichere Zeitlinie
        self.timelines[timeline_id] = timeline
        
        logger.info(f"Zeitlinie '{name}' ({timeline_id}) erstellt")
        return self.timelines[timeline_id]
        
        # Prüfe Integrität, falls aktiviert
        if self.config.get("auto_integrity_check", True) and initial_nodes:
            self.check_temporal_integrity(timeline)
        
        logger.info(f"Neue Zeitlinie erstellt: {name} (ID: {timeline_id})")
    
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
    
    def update_timeline(self, 
                      timeline_id: str, 
                      updates: Dict[str, Any]) -> Optional[Timeline]:
        """
        Aktualisiert eine Zeitlinie
        
        Args:
            timeline_id: ID der Zeitlinie
            updates: Dictionary mit zu aktualisierenden Werten
            
        Returns:
            Aktualisierte Timeline oder None, falls nicht gefunden
        """
        timeline = self.timelines.get(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} existiert nicht")
            return None
        
        # Aktualisiere Werte
        for key, value in updates.items():
            if hasattr(timeline, key):
                setattr(timeline, key, value)
        
        # Aktualisiere Zeitstempel
        timeline.last_modified = time.time()
        
        # Prüfe Integrität, falls aktiviert
        if self.config.get("auto_integrity_check", True):
            self.check_temporal_integrity(timeline)
        
        logger.info(f"Zeitlinie {timeline_id} aktualisiert")
        
        return timeline
    
    def delete_timeline(self, timeline_id: str) -> bool:
        """
        Löscht eine Zeitlinie
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            True, wenn erfolgreich gelöscht, sonst False
        """
        if timeline_id not in self.timelines:
            logger.warning(f"Zeitlinie {timeline_id} existiert nicht")
            return False
        
        # Lösche Zeitlinie
        del self.timelines[timeline_id]
        
        logger.info(f"Zeitlinie {timeline_id} gelöscht")
        
        return True
    
    # Zeitknoten-Management
    
    def add_time_node(self, 
                    timeline_id: str, 
                    description: str,
                    timestamp: Optional[float] = None,
                    parent_node_id: Optional[str] = None,
                    probability: float = 0.8,
                    trigger_level: TriggerLevel = TriggerLevel.MEDIUM,
                    metadata: Dict[str, Any] = None) -> Optional[TimeNode]:
        """
        Fügt einen Zeitknoten zu einer Zeitlinie hinzu
        
        Args:
            timeline_id: ID der Zeitlinie
            description: Beschreibung des Knotens
            timestamp: Zeitstempel des Knotens (optional, Standard: aktuelle Zeit)
            parent_node_id: ID des Elternknotens (optional)
            probability: Wahrscheinlichkeit des Knotens (0.0-1.0)
            trigger_level: Trigger-Level des Knotens
            metadata: Zusätzliche Metadaten
            
        Returns:
            Erstellter TimeNode oder None, falls Fehler
        """
        # Prüfe, ob Zeitlinie existiert
        if timeline_id not in self.timelines:
            logger.error(f"Zeitlinie {timeline_id} nicht gefunden")
            return None
            
        timeline = self.timelines[timeline_id]
        
        # Erstelle neuen Knoten mit der Basisklasse aus timeline_base
        node = TimeNode(
            description=description,
            timestamp=timestamp or time.time(),
            parent_id=parent_node_id,
            probability=probability,
            trigger_level=trigger_level,
            metadata=metadata or {}
        )
        
        # Füge Knoten zur Zeitlinie hinzu
        node_id = timeline.add_node(node)
        
        # Füge Knoten zum Elternknoten hinzu, falls vorhanden
        if parent_node_id and parent_node_id in timeline.nodes:
            parent_node = timeline.get_node(parent_node_id)
            if parent_node:
                parent_node.add_child(node.id)
            
        logger.info(f"Zeitknoten {node.id} zur Zeitlinie {timeline_id} hinzugefügt")
        logger.info(f"Zeitknoten {node_id} zu Zeitlinie {timeline_id} hinzugefügt")
        
        return node
    
    def update_time_node(self, 
                       timeline_id: str, 
                       node_id: str,
                       updates: Dict[str, Any]) -> Optional[TimeNode]:
        """
        Aktualisiert einen Zeitknoten
        
        Args:
            timeline_id: ID der Zeitlinie
            node_id: ID des Knotens
            updates: Dictionary mit zu aktualisierenden Werten
            
        Returns:
            Aktualisierter TimeNode oder None, falls nicht gefunden
        """
        timeline = self.timelines.get(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} existiert nicht")
            return None
        
        node = timeline.nodes.get(node_id)
        if not node:
            logger.warning(f"Knoten {node_id} existiert nicht in Zeitlinie {timeline_id}")
            return None
        
        # Aktualisiere Werte
        for key, value in updates.items():
            if hasattr(node, key) and key != "id":  # ID darf nicht geändert werden
                setattr(node, key, value)
        
        # Aktualisiere Zeitstempel der Zeitlinie
        timeline.last_modified = time.time()
        
        # Prüfe Integrität, falls aktiviert
        if self.config.get("auto_integrity_check", True):
            self.check_temporal_integrity(timeline)
        
        logger.info(f"Zeitknoten {node_id} in Zeitlinie {timeline_id} aktualisiert")
        
        return node
    
    def delete_time_node(self, 
                       timeline_id: str, 
                       node_id: str,
                       recursive: bool = False) -> bool:
        """
        Löscht einen Zeitknoten
        
        Args:
            timeline_id: ID der Zeitlinie
            node_id: ID des Knotens
            recursive: Wenn True, werden auch alle Kindknoten gelöscht
            
        Returns:
            True, wenn erfolgreich gelöscht, sonst False
        """
        timeline = self.timelines.get(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} existiert nicht")
            return False
        
        node = timeline.nodes.get(node_id)
        if not node:
            logger.warning(f"Knoten {node_id} existiert nicht in Zeitlinie {timeline_id}")
            return False
        
        # Wenn rekursiv, lösche alle Kindknoten
        if recursive:
            for child_id in node.child_node_ids[:]:  # Kopie, da wir die Liste ändern
                self.delete_time_node(timeline_id, child_id, recursive=True)
        elif node.child_node_ids:
            logger.warning(f"Knoten {node_id} hat Kindknoten, setze recursive=True, um auch diese zu löschen")
            return False
        
        # Entferne Knoten aus dem Elternknoten
        if node.parent_node_id and node.parent_node_id in timeline.nodes:
            parent_node = timeline.nodes[node.parent_node_id]
            if node_id in parent_node.child_node_ids:
                parent_node.child_node_ids.remove(node_id)
        
        # Lösche Knoten
        del timeline.nodes[node_id]
        
        # Aktualisiere Zeitstempel der Zeitlinie
        timeline.last_modified = time.time()
        
        logger.info(f"Zeitknoten {node_id} aus Zeitlinie {timeline_id} gelöscht")
        
        return True
