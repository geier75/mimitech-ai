#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: context_state.py
Zustandsmodell zur Erfassung des aktuellen Kontexts im VXOR-System

Dieses Modul ist verantwortlich für die Verwaltung und Aktualisierung des aktuellen
Systemkontexts. Es erfasst und aggregiert Kontextinformationen aus verschiedenen Quellen
und stellt eine konsistente Darstellung des aktuellen Zustands bereit.

Optimiert für: Python 3.10+, Apple Silicon (M4 Max)
"""

import os
import time
import logging
import threading
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from collections import defaultdict
import pickle
import copy

# Logging-Konfiguration
LOG_DIR = "/home/ubuntu/VXOR_Logs/CONTEXT/"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}context_state.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VX-CONTEXT.state")

# Import von ContextCore-Definitionen
try:
    from context_core import ContextSource, ContextPriority, ContextData, get_context_core
except ImportError:
    logger.error("Konnte ContextCore-Definitionen nicht importieren")
    # Fallback-Definitionen
    class ContextSource(Enum):
        VISUAL = auto()
        LANGUAGE = auto()
        INTERNAL = auto()
        EXTERNAL = auto()
        MEMORY = auto()
        EMOTION = auto()
        INTENT = auto()
        REFLEX = auto()

    class ContextPriority(Enum):
        CRITICAL = auto()
        HIGH = auto()
        MEDIUM = auto()
        LOW = auto()
        BACKGROUND = auto()

    @dataclass
    class ContextData:
        source: ContextSource
        priority: ContextPriority
        timestamp: float = field(default_factory=time.time)
        data: Dict[str, Any] = field(default_factory=dict)
        metadata: Dict[str, Any] = field(default_factory=dict)
        processing_time_ms: Optional[float] = None
    
    def get_context_core():
        return None

# Import von FocusRouter-Definitionen
try:
    from focus_router import FocusType, FocusTarget, FocusState
except ImportError:
    logger.error("Konnte FocusRouter-Definitionen nicht importieren")
    # Fallback-Definitionen
    class FocusType(Enum):
        COGNITIVE = auto()
        SENSORY = auto()
        MOTOR = auto()
        EMOTIONAL = auto()
        MEMORY = auto()

    class FocusTarget(Enum):
        VISION = auto()
        AUDIO = auto()
        LANGUAGE = auto()
        MEMORY = auto()
        EMOTION = auto()
        PLANNING = auto()
        REASONING = auto()
        MOTOR = auto()
        AGENT = auto()

    @dataclass
    class FocusState:
        target: Union[FocusTarget, str]
        focus_type: FocusType
        intensity: float
        timestamp: float = field(default_factory=time.time)
        metadata: Dict[str, Any] = field(default_factory=dict)
        agent_id: Optional[str] = None


class ContextStateType(Enum):
    """Typen von Kontextzuständen"""
    VISUAL = auto()     # Visueller Kontext
    AUDITORY = auto()   # Auditiver Kontext
    LINGUISTIC = auto() # Sprachlicher Kontext
    EMOTIONAL = auto()  # Emotionaler Kontext
    COGNITIVE = auto()  # Kognitiver Kontext
    PHYSICAL = auto()   # Physischer Kontext
    TEMPORAL = auto()   # Zeitlicher Kontext
    SOCIAL = auto()     # Sozialer Kontext
    SYSTEM = auto()     # Systemkontext
    GLOBAL = auto()     # Globaler Kontext


@dataclass
class ContextStateEntry:
    """Eintrag im Kontextzustand"""
    value: Any
    source: ContextSource
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0  # 0.0 bis 1.0
    expiration: Optional[float] = None  # Zeitstempel, wann der Eintrag abläuft
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextSnapshot:
    """Snapshot des gesamten Kontextzustands zu einem bestimmten Zeitpunkt"""
    timestamp: float
    state: Dict[ContextStateType, Dict[str, ContextStateEntry]]
    focus_state: Dict[Tuple, FocusState]
    global_metadata: Dict[str, Any] = field(default_factory=dict)


class ContextState:
    """
    Verwaltet den aktuellen Kontextzustand des Systems
    
    Diese Klasse ist verantwortlich für:
    1. Erfassung und Aktualisierung des Kontextzustands aus verschiedenen Quellen
    2. Bereitstellung einer konsistenten Darstellung des aktuellen Zustands
    3. Verwaltung der Kontexthistorie und Erstellung von Snapshots
    4. Bereitstellung von Abfragemöglichkeiten für den Kontextzustand
    """
    
    def __init__(self, context_core=None):
        """
        Initialisiert den ContextState
        
        Args:
            context_core: Referenz zum ContextCore-Modul, falls verfügbar
        """
        self.context_core = context_core or get_context_core()
        
        # Aktueller Kontextzustand: {ContextStateType: {key: ContextStateEntry}}
        self.current_state = {state_type: {} for state_type in ContextStateType}
        
        # Globale Metadaten
        self.global_metadata = {
            "system_start_time": time.time(),
            "last_update_time": time.time(),
            "update_count": 0,
            "active_sources": set()
        }
        
        # Kontexthistorie: Liste von Snapshots
        self.state_history = []
        self.max_history_length = 100  # Maximale Anzahl von Snapshots in der Historie
        
        # Mapping von Kontextquellen zu Zustandstypen
        self.source_to_state_type = {
            ContextSource.VISUAL: ContextStateType.VISUAL,
            ContextSource.LANGUAGE: ContextStateType.LINGUISTIC,
            ContextSource.INTERNAL: ContextStateType.COGNITIVE,
            ContextSource.EXTERNAL: ContextStateType.PHYSICAL,
            ContextSource.MEMORY: ContextStateType.COGNITIVE,
            ContextSource.EMOTION: ContextStateType.EMOTIONAL,
            ContextSource.INTENT: ContextStateType.COGNITIVE,
            ContextSource.REFLEX: ContextStateType.PHYSICAL
        }
        
        # Registrierte Aktualisierungshandler
        self.update_handlers = {}
        
        # Zustandsänderungshandler
        self.state_change_handlers = []
        
        # Synchronisierung
        self.state_lock = threading.RLock()
        
        # Snapshot-Intervall in Sekunden
        self.snapshot_interval = 5.0
        self.last_snapshot_time = time.time()
        
        # Automatische Bereinigung abgelaufener Einträge
        self.cleanup_interval = 10.0
        self.last_cleanup_time = time.time()
        
        # Snapshot-Thread
        self.snapshot_thread = None
        self.active = False
        
        logger.info("ContextState initialisiert")
        
        # Registriere Handler beim ContextCore, falls verfügbar
        if self.context_core:
            try:
                for source in ContextSource:
                    self.context_core.context_processor.register_handler(
                        source, self.process_context_data
                    )
                logger.info("ContextState-Handler bei ContextCore registriert")
            except Exception as e:
                logger.error(f"Fehler bei der Registrierung von Handlern: {str(e)}", exc_info=True)
    
    def start(self):
        """Startet den ContextState"""
        if self.active:
            logger.warning("ContextState läuft bereits")
            return
        
        self.active = True
        
        # Starte Snapshot-Thread
        self.snapshot_thread = threading.Thread(
            target=self._snapshot_loop,
            name="ContextStateSnapshotThread",
            daemon=True
        )
        self.snapshot_thread.start()
        
        logger.info("ContextState gestartet")
    
    def stop(self):
        """Stoppt den ContextState"""
        if not self.active:
            logger.warning("ContextState läuft nicht")
            return
        
        self.active = False
        
        # Stoppe Snapshot-Thread
        if self.snapshot_thread:
            self.snapshot_thread.join(timeout=1.0)
        
        logger.info("ContextState gestoppt")
    
    def process_context_data(self, context_data: ContextData):
        """
        Verarbeitet eingehende Kontextdaten und aktualisiert den Zustand
        
        Args:
            context_data: Die zu verarbeitenden Kontextdaten
        """
        source = context_data.source
        data = context_data.data
        
        # Bestimme den Zustandstyp basierend auf der Quelle
        state_type = self.source_to_state_type.get(source, ContextStateType.GLOBAL)
        
        # Aktualisiere den Zustand
        self.update_state(state_type, data, source, context_data.metadata)
    
    def update_state(self, state_type: ContextStateType, data: Dict[str, Any],
                    source: ContextSource, metadata: Optional[Dict[str, Any]] = None):
        """
        Aktualisiert den Kontextzustand
        
        Args:
            state_type: Der zu aktualisierende Zustandstyp
            data: Die neuen Daten
            source: Die Quelle der Daten
            metadata: Optionale Metadaten
        """
        if metadata is None:
            metadata = {}
        
        with self.state_lock:
            state_dict = self.current_state[state_type]
            
            # Flache Daten in den Zustand einfügen
            self._update_state_dict(state_dict, data, source, metadata)
            
            # Aktualisiere globale Metadaten
            self.global_metadata["last_update_time"] = time.time()
            self.global_metadata["update_count"] += 1
            self.global_metadata["active_sources"].add(source.name)
            
            # Rufe Zustandsänderungshandler auf
            self._notify_state_change(state_type, data, source)
    
    def _update_state_dict(self, state_dict: Dict[str, ContextStateEntry], data: Dict[str, Any],
                          source: ContextSource, metadata: Dict[str, Any], prefix: str = ""):
        """
        Aktualisiert ein Zustandswörterbuch rekursiv
        
        Args:
            state_dict: Das zu aktualisierende Zustandswörterbuch
            data: Die neuen Daten
            source: Die Quelle der Daten
            metadata: Metadaten
            prefix: Präfix für verschachtelte Schlüssel
        """
        current_time = time.time()
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Rekursiv verschachtelte Daten aktualisieren
                self._update_state_dict(state_dict, value, source, metadata, full_key)
            else:
                # Bestimme Ablaufzeit basierend auf Metadaten oder Standardwert
                expiration = None
                if "expiration" in metadata:
                    expiration = current_time + metadata["expiration"]
                elif "ttl" in metadata:
                    expiration = current_time + metadata["ttl"]
                
                # Bestimme Konfidenz basierend auf Metadaten oder Standardwert
                confidence = metadata.get("confidence", 1.0)
                
                # Erstelle oder aktualisiere Eintrag
                entry = ContextStateEntry(
                    value=value,
                    source=source,
                    timestamp=current_time,
                    confidence=confidence,
                    expiration=expiration,
                    metadata={
                        "key": full_key,
                        "update_time": current_time,
                        **metadata
                    }
                )
                
                state_dict[full_key] = entry
    
    def get_state(self, state_type: Optional[ContextStateType] = None, 
                 key: Optional[str] = None) -> Union[Dict[str, Any], Any, None]:
        """
        Gibt den aktuellen Kontextzustand zurück
        
        Args:
            state_type: Optional, der abzufragende Zustandstyp
            key: Optional, der abzufragende Schlüssel
            
        Returns:
            Union[Dict[str, Any], Any, None]: Der Kontextzustand, Wert oder None
        """
        with self.state_lock:
            if state_type is None and key is None:
                # Gesamten Zustand zurückgeben
                return {
                    state_type.name: {
                        k: v.value for k, v in state_dict.items()
                    } for state_type, state_dict in self.current_state.items()
                }
            
            if state_type is not None and key is None:
                # Zustand für einen bestimmten Typ zurückgeben
                state_dict = self.current_state.get(state_type, {})
                return {k: v.value for k, v in state_dict.items()}
            
            if state_type is None and key is not None:
                # Suche nach Schlüssel in allen Zustandstypen
                for state_dict in self.current_state.values():
                    if key in state_dict:
                        return state_dict[key].value
                return None
            
            # Spezifischen Wert zurückgeben
            state_dict = self.current_state.get(state_type, {})
            entry = state_dict.get(key)
            
            return entry.value if entry else None
    
    def get_state_entry(self, state_type: ContextStateType, 
                       key: str) -> Optional[ContextStateEntry]:
        """
        Gibt einen spezifischen Zustandseintrag zurück
        
        Args:
            state_type: Der Zustandstyp
            key: Der Schlüssel
            
        Returns:
            Optional[ContextStateEntry]: Der Zustandseintrag oder None
        """
        with self.state_lock:
            state_dict = self.current_state.get(state_type, {})
            return state_dict.get(key)
    
    def get_state_by_source(self, source: ContextSource) -> Dict[str, Any]:
        """
        Gibt alle Zustandseinträge von einer bestimmten Quelle zurück
        
        Args:
            source: Die Quelle
            
        Returns:
            Dict[str, Any]: Die Zustandseinträge
        """
        result = {}
        
        with self.state_lock:
            for state_type, state_dict in self.current_state.items():
                for key, entry in state_dict.items():
                    if entry.source == source:
                        result[key] = entry.value
        
        return result
    
    def create_snapshot(self) -> ContextSnapshot:
        """
        Erstellt einen Snapshot des aktuellen Kontextzustands
        
        Returns:
            ContextSnapshot: Der erstellte Snapshot
        """
        with self.state_lock:
            # Kopiere den aktuellen Zustand
            state_copy = {
                state_type: state_dict.copy()
                for state_type, state_dict in self.current_state.items()
            }
            
            # Hole den aktuellen Fokuszustand, falls verfügbar
            focus_state = {}
            if self.context_core and self.context_core.focus_router:
                focus_state = self.context_core.focus_router.get_all_focus_states()
            
            # Erstelle Snapshot
            snapshot = ContextSnapshot(
                timestamp=time.time(),
                state=state_copy,
                focus_state=focus_state,
                global_metadata=self.global_metadata.copy()
            )
            
            # Füge Snapshot zur Historie hinzu
            self.state_history.append(snapshot)
            
            # Begrenze die Größe der Historie
            if len(self.state_history) > self.max_history_length:
                self.state_history = self.state_history[-self.max_history_length:]
  
(Content truncated due to size limit. Use line ranges to read in chunks)