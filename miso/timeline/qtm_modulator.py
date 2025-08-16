#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - ECHO-PRIME QTM_Modulator

Diese Datei implementiert den QTM_Modulator (Quantum Temporal Modulation),
der Quanteneffekte für die Zeitlinienmanipulation nutzt.

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
logger = logging.getLogger("MISO.timeline.qtm_modulator")

# Importiere gemeinsame Datenstrukturen
from miso.timeline.echo_prime import TimeNode, Timeline, Trigger, TemporalEvent, TriggerLevel

class QuantumState(Enum):
    """Quantenzustände für die temporale Modulation"""
    SUPERPOSITION = auto()  # Überlagerung mehrerer Zustände
    ENTANGLED = auto()      # Verschränkung mit anderen Zeitlinien
    COLLAPSED = auto()      # Kollabierter Zustand (beobachtet)
    TUNNELING = auto()      # Tunneleffekt zwischen Zeitlinien
    COHERENT = auto()       # Kohärenter Zustand
    DECOHERENT = auto()     # Dekohärenter Zustand

@dataclass
class QuantumTimeEffect:
    """Repräsentiert einen Quanteneffekt auf die Zeit"""
    id: str
    state: QuantumState
    description: str
    strength: float  # 0.0-1.0
    duration: float  # Dauer in Sekunden
    timeline_ids: List[str]
    node_ids: List[str]
    creation_time: float
    expiration_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert den Quanteneffekt in ein Dictionary"""
        return {
            "id": self.id,
            "state": self.state.name,
            "description": self.description,
            "strength": self.strength,
            "duration": self.duration,
            "timeline_ids": self.timeline_ids,
            "node_ids": self.node_ids,
            "creation_time": self.creation_time,
            "expiration_time": self.expiration_time,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumTimeEffect':
        """Erstellt einen Quanteneffekt aus einem Dictionary"""
        quantum_state = QuantumState[data["state"]]
        return cls(
            id=data["id"],
            state=quantum_state,
            description=data["description"],
            strength=data["strength"],
            duration=data["duration"],
            timeline_ids=data["timeline_ids"],
            node_ids=data["node_ids"],
            creation_time=data["creation_time"],
            expiration_time=data["expiration_time"],
            metadata=data["metadata"]
        )

class QTM_Modulator:
    """
    Quantum Temporal Modulation für Zeitlinien
    """
    
    def __init__(self):
        """Initialisiert den QTM_Modulator"""
        self.quantum_effects = {}
        self.timeline_cache = {}
        self.quantum_states = {}
        self.entanglement_pairs = []
        logger.info("QTM_Modulator initialisiert")
    
    def apply_superposition(self, 
                          timelines: List[Timeline], 
                          duration: float = 3600.0) -> QuantumTimeEffect:
        """
        Wendet eine Quantensuperposition auf mehrere Zeitlinien an
        
        Args:
            timelines: Liste von Zeitlinien
            duration: Dauer der Superposition in Sekunden
            
        Returns:
            Erstellter Quanteneffekt
        """
        # Cache Zeitlinien
        for timeline in timelines:
            self.timeline_cache[timeline.id] = timeline
        
        # Erstelle Quanteneffekt
        effect_id = str(uuid.uuid4())
        
        timeline_ids = [timeline.id for timeline in timelines]
        node_ids = []
        
        # Sammle alle aktiven Knoten (Knoten mit höchstem Zeitstempel)
        for timeline in timelines:
            active_nodes = self._get_active_nodes(timeline)
            node_ids.extend([node.id for node in active_nodes])
        
        effect = QuantumTimeEffect(
            id=effect_id,
            state=QuantumState.SUPERPOSITION,
            description=f"Superposition von {len(timelines)} Zeitlinien",
            strength=0.8,
            duration=duration,
            timeline_ids=timeline_ids,
            node_ids=node_ids,
            creation_time=time.time(),
            expiration_time=time.time() + duration
        )
        
        # Speichere Quanteneffekt
        self.quantum_effects[effect_id] = effect
        
        # Aktualisiere Quantenzustände der Zeitlinien
        for timeline_id in timeline_ids:
            self.quantum_states[timeline_id] = QuantumState.SUPERPOSITION
        
        logger.info(f"Superposition auf {len(timelines)} Zeitlinien angewendet")
        
        return effect
    
    def entangle_timelines(self, 
                         timeline1: Timeline, 
                         timeline2: Timeline,
                         node_pairs: List[Tuple[str, str]],
                         duration: float = 7200.0) -> QuantumTimeEffect:
        """
        Verschränkt zwei Zeitlinien miteinander
        
        Args:
            timeline1: Erste Zeitlinie
            timeline2: Zweite Zeitlinie
            node_pairs: Liste von Knotenpaaren (node_id1, node_id2) zur Verschränkung
            duration: Dauer der Verschränkung in Sekunden
            
        Returns:
            Erstellter Quanteneffekt
        """
        # Cache Zeitlinien
        self.timeline_cache[timeline1.id] = timeline1
        self.timeline_cache[timeline2.id] = timeline2
        
        # Erstelle Quanteneffekt
        effect_id = str(uuid.uuid4())
        
        # Sammle alle Knoten-IDs
        node_ids = []
        for node_id1, node_id2 in node_pairs:
            node_ids.append(node_id1)
            node_ids.append(node_id2)
        
        effect = QuantumTimeEffect(
            id=effect_id,
            state=QuantumState.ENTANGLED,
            description=f"Verschränkung zwischen Zeitlinien {timeline1.id} und {timeline2.id}",
            strength=0.9,
            duration=duration,
            timeline_ids=[timeline1.id, timeline2.id],
            node_ids=node_ids,
            creation_time=time.time(),
            expiration_time=time.time() + duration
        )
        
        # Speichere Quanteneffekt
        self.quantum_effects[effect_id] = effect
        
        # Aktualisiere Quantenzustände der Zeitlinien
        self.quantum_states[timeline1.id] = QuantumState.ENTANGLED
        self.quantum_states[timeline2.id] = QuantumState.ENTANGLED
        
        # Speichere Verschränkungspaare
        for node_id1, node_id2 in node_pairs:
            self.entanglement_pairs.append({
                "timeline1_id": timeline1.id,
                "timeline2_id": timeline2.id,
                "node1_id": node_id1,
                "node2_id": node_id2,
                "effect_id": effect_id,
                "creation_time": time.time(),
                "expiration_time": time.time() + duration
            })
        
        logger.info(f"Verschränkung zwischen Zeitlinien {timeline1.id} und {timeline2.id} erstellt")
        
        return effect
    
    def collapse_timeline(self, 
                        timeline: Timeline, 
                        target_nodes: List[str] = None) -> QuantumTimeEffect:
        """
        Kollabiert eine Zeitlinie in einen bestimmten Zustand
        
        Args:
            timeline: Zu kollabierende Zeitlinie
            target_nodes: Liste von Zielknoten (optional)
            
        Returns:
            Erstellter Quanteneffekt
        """
        # Cache Zeitlinie
        self.timeline_cache[timeline.id] = timeline
        
        # Wenn keine Zielknoten angegeben sind, verwende alle aktiven Knoten
        if not target_nodes:
            active_nodes = self._get_active_nodes(timeline)
            target_nodes = [node.id for node in active_nodes]
        
        # Erstelle Quanteneffekt
        effect_id = str(uuid.uuid4())
        
        effect = QuantumTimeEffect(
            id=effect_id,
            state=QuantumState.COLLAPSED,
            description=f"Kollabierung der Zeitlinie {timeline.id}",
            strength=1.0,  # Maximale Stärke für Kollabierung
            duration=0.0,  # Kollabierung ist sofort
            timeline_ids=[timeline.id],
            node_ids=target_nodes,
            creation_time=time.time(),
            expiration_time=None  # Kein Ablauf
        )
        
        # Speichere Quanteneffekt
        self.quantum_effects[effect_id] = effect
        
        # Aktualisiere Quantenzustand der Zeitlinie
        self.quantum_states[timeline.id] = QuantumState.COLLAPSED
        
        # Entferne alle anderen Quanteneffekte für diese Zeitlinie
        self._remove_timeline_effects(timeline.id, exclude_effect_id=effect_id)
        
        logger.info(f"Zeitlinie {timeline.id} kollabiert")
        
        return effect
    
    def apply_quantum_tunneling(self, 
                              source_timeline: Timeline, 
                              target_timeline: Timeline,
                              node_id: str,
                              duration: float = 1800.0) -> QuantumTimeEffect:
        """
        Wendet Quantentunneling zwischen zwei Zeitlinien an
        
        Args:
            source_timeline: Quellzeitlinie
            target_timeline: Zielzeitlinie
            node_id: ID des zu tunnelnden Knotens
            duration: Dauer des Tunneleffekts in Sekunden
            
        Returns:
            Erstellter Quanteneffekt
        """
        # Cache Zeitlinien
        self.timeline_cache[source_timeline.id] = source_timeline
        self.timeline_cache[target_timeline.id] = target_timeline
        
        # Prüfe, ob der Knoten in der Quellzeitlinie existiert
        if node_id not in source_timeline.nodes:
            logger.error(f"Knoten {node_id} existiert nicht in Zeitlinie {source_timeline.id}")
            return None
        
        # Erstelle Quanteneffekt
        effect_id = str(uuid.uuid4())
        
        effect = QuantumTimeEffect(
            id=effect_id,
            state=QuantumState.TUNNELING,
            description=f"Quantentunneling von Zeitlinie {source_timeline.id} zu {target_timeline.id}",
            strength=0.7,
            duration=duration,
            timeline_ids=[source_timeline.id, target_timeline.id],
            node_ids=[node_id],
            creation_time=time.time(),
            expiration_time=time.time() + duration
        )
        
        # Speichere Quanteneffekt
        self.quantum_effects[effect_id] = effect
        
        # Aktualisiere Quantenzustände der Zeitlinien
        self.quantum_states[source_timeline.id] = QuantumState.TUNNELING
        self.quantum_states[target_timeline.id] = QuantumState.TUNNELING
        
        logger.info(f"Quantentunneling zwischen Zeitlinien {source_timeline.id} und {target_timeline.id} angewendet")
        
        return effect
    
    def maintain_coherence(self, 
                         timeline: Timeline, 
                         duration: float = 14400.0) -> QuantumTimeEffect:
        """
        Hält die Kohärenz einer Zeitlinie aufrecht
        
        Args:
            timeline: Zeitlinie
            duration: Dauer der Kohärenz in Sekunden
            
        Returns:
            Erstellter Quanteneffekt
        """
        # Cache Zeitlinie
        self.timeline_cache[timeline.id] = timeline
        
        # Erstelle Quanteneffekt
        effect_id = str(uuid.uuid4())
        
        # Sammle alle aktiven Knoten
        active_nodes = self._get_active_nodes(timeline)
        node_ids = [node.id for node in active_nodes]
        
        effect = QuantumTimeEffect(
            id=effect_id,
            state=QuantumState.COHERENT,
            description=f"Kohärenz der Zeitlinie {timeline.id}",
            strength=0.85,
            duration=duration,
            timeline_ids=[timeline.id],
            node_ids=node_ids,
            creation_time=time.time(),
            expiration_time=time.time() + duration
        )
        
        # Speichere Quanteneffekt
        self.quantum_effects[effect_id] = effect
        
        # Aktualisiere Quantenzustand der Zeitlinie
        self.quantum_states[timeline.id] = QuantumState.COHERENT
        
        logger.info(f"Kohärenz für Zeitlinie {timeline.id} aufrechterhalten")
        
        return effect
    
    def induce_decoherence(self, 
                         timeline: Timeline, 
                         strength: float = 0.5,
                         duration: float = 3600.0) -> QuantumTimeEffect:
        """
        Induziert Dekohärenz in einer Zeitlinie
        
        Args:
            timeline: Zeitlinie
            strength: Stärke der Dekohärenz (0.0-1.0)
            duration: Dauer der Dekohärenz in Sekunden
            
        Returns:
            Erstellter Quanteneffekt
        """
        # Cache Zeitlinie
        self.timeline_cache[timeline.id] = timeline
        
        # Erstelle Quanteneffekt
        effect_id = str(uuid.uuid4())
        
        # Sammle alle aktiven Knoten
        active_nodes = self._get_active_nodes(timeline)
        node_ids = [node.id for node in active_nodes]
        
        effect = QuantumTimeEffect(
            id=effect_id,
            state=QuantumState.DECOHERENT,
            description=f"Dekohärenz der Zeitlinie {timeline.id}",
            strength=strength,
            duration=duration,
            timeline_ids=[timeline.id],
            node_ids=node_ids,
            creation_time=time.time(),
            expiration_time=time.time() + duration
        )
        
        # Speichere Quanteneffekt
        self.quantum_effects[effect_id] = effect
        
        # Aktualisiere Quantenzustand der Zeitlinie
        self.quantum_states[timeline.id] = QuantumState.DECOHERENT
        
        logger.info(f"Dekohärenz für Zeitlinie {timeline.id} induziert")
        
        return effect
    
    def _get_active_nodes(self, timeline: Timeline) -> List[TimeNode]:
        """
        Gibt die aktiven Knoten einer Zeitlinie zurück
        
        Args:
            timeline: Zeitlinie
            
        Returns:
            Liste von aktiven Knoten
        """
        # Aktive Knoten sind Knoten ohne Kinder
        active_nodes = []
        
        for node_id, node in timeline.nodes.items():
            if not node.child_node_ids:
                active_nodes.append(node)
        
        return active_nodes
    
    def _remove_timeline_effects(self, timeline_id: str, exclude_effect_id: str = None):
        """
        Entfernt alle Quanteneffekte für eine Zeitlinie
        
        Args:
            timeline_id: ID der Zeitlinie
            exclude_effect_id: ID eines zu ausschließenden Effekts
        """
        effects_to_remove = []
        
        for effect_id, effect in self.quantum_effects.items():
            if timeline_id in effect.timeline_ids and effect_id != exclude_effect_id:
                effects_to_remove.append(effect_id)
        
        for effect_id in effects_to_remove:
            del self.quantum_effects[effect_id]
    
    def get_quantum_effect(self, effect_id: str) -> Optional[QuantumTimeEffect]:
        """
        Gibt einen Quanteneffekt zurück
        
        Args:
            effect_id: ID des Quanteneffekts
            
        Returns:
            QuantumTimeEffect oder None, falls nicht gefunden
        """
        return self.quantum_effects.get(effect_id)
    
    def get_all_quantum_effects(self) -> Dict[str, QuantumTimeEffect]:
        """
        Gibt alle Quanteneffekte zurück
        
        Returns:
            Dictionary mit allen Quanteneffekten
        """
        return self.quantum_effects
    
    def get_active_quantum_effects(self) -> List[QuantumTimeEffect]:
        """
        Gibt alle aktiven Quanteneffekte zurück (nicht abgelaufen)
        
        Returns:
            Liste von aktiven Quanteneffekten
        """
        current_time = time.time()
        
        return [
            effect for effect in self.quantum_effects.values()
            if not effect.expiration_time or effect.expiration_time > current_time
        ]
    
    def get_timeline_quantum_state(self, timeline_id: str) -> Optional[QuantumState]:
        """
        Gibt den Quantenzustand einer Zeitlinie zurück
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            QuantumState oder None, falls nicht gefunden
        """
        return self.quantum_states.get(timeline_id)
    
    def get_entangled_timelines(self, timeline_id: str) -> List[str]:
        """
        Gibt alle mit einer Zeitlinie verschränkten Zeitlinien zurück
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            Liste von verschränkten Zeitlinien-IDs
        """
        entangled_timelines = set()
        current_time = time.time()
        
        for pair in self.entanglement_pairs:
            # Prüfe, ob die Verschränkung noch aktiv ist
            if pair["expiration_time"] and pair["expiration_time"] < current_time:
                continue
            
            if pair["timeline1_id"] == timeline_id:
                entangled_timelines.add(pair["timeline2_id"])
            elif pair["timeline2_id"] == timeline_id:
                entangled_timelines.add(pair["timeline1_id"])
        
        return list(entangled_timelines)
    
    def clear_expired_effects(self) -> int:
        """
        Löscht abgelaufene Quanteneffekte
        
        Returns:
            Anzahl der gelöschten Effekte
        """
        current_time = time.time()
        expired_ids = [
            effect_id for effect_id, effect in self.quantum_effects.items()
            if effect.expiration_time and effect.expiration_time < current_time
        ]
        
        # Lösche abgelaufene Effekte
        for effect_id in expired_ids:
            del self.quantum_effects[effect_id]
        
        # Lösche abgelaufene Verschränkungspaare
        self.entanglement_pairs = [
            pair for pair in self.entanglement_pairs
            if not pair["expiration_time"] or pair["expiration_time"] >= current_time
        ]
        
        logger.info(f"{len(expired_ids)} abgelaufene Quanteneffekte gelöscht")
        
        return len(expired_ids)
