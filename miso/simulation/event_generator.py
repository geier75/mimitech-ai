#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - PRISM-EventGenerator

Generiert Ereignisse für die PRISM-Simulation basierend auf Wahrscheinlichkeitsmodellen
und Zeitlinienanalysen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import random
import uuid
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field

# Importiere Basisklassen und -typen aus prism_base
from miso.simulation.prism_base import TimeNode, Timeline, TimelineType, calculate_probability

# Konfiguriere Logging
logger = logging.getLogger("MISO.simulation.event_generator")

# Prüfen, ob Apple Silicon verfügbar ist und entsprechende Optimierungen aktivieren
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
if is_apple_silicon:
    # Apple Neural Engine Optimierungen
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import von internen Modulen
try:
    from miso.timeline.echo_prime import TimeNode, Timeline, Trigger, TemporalEvent, TriggerLevel
    from miso.math.mprime_engine import MPrimeEngine
    from miso.math.prob_mapper import ProbabilisticMapper
    HAS_DEPENDENCIES = True
except ImportError:
    logger.warning("Einige Abhängigkeiten konnten nicht importiert werden. EventGenerator läuft im eingeschränkten Modus.")
    HAS_DEPENDENCIES = False


class EventType(Enum):
    """Typen von Ereignissen in der Simulation"""
    TEMPORAL = auto()      # Zeitliche Ereignisse
    CAUSAL = auto()        # Ursache-Wirkungs-Ereignisse
    PROBABILISTIC = auto() # Wahrscheinlichkeitsbasierte Ereignisse
    QUANTUM = auto()       # Quantenereignisse
    PARADOX = auto()       # Paradoxe Ereignisse
    SYSTEM = auto()        # Systemereignisse
    USER = auto()          # Benutzerereignisse
    EXTERNAL = auto()      # Externe Ereignisse


@dataclass
class SimulationEvent:
    """Repräsentiert ein Ereignis in der Simulation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.TEMPORAL
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0
    source_timeline_id: Optional[str] = None
    target_timeline_id: Optional[str] = None
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert das Ereignis in ein Dictionary"""
        return {
            "id": self.id,
            "type": self.type.name,
            "timestamp": self.timestamp,
            "data": self.data,
            "probability": self.probability,
            "source_timeline_id": self.source_timeline_id,
            "target_timeline_id": self.target_timeline_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationEvent':
        """Erstellt ein Ereignis aus einem Dictionary"""
        event_type = EventType[data["type"]] if isinstance(data["type"], str) else data["type"]
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=event_type,
            timestamp=data.get("timestamp", time.time()),
            data=data.get("data", {}),
            probability=data.get("probability", 1.0),
            source_timeline_id=data.get("source_timeline_id"),
            target_timeline_id=data.get("target_timeline_id"),
            source_node_id=data.get("source_node_id"),
            target_node_id=data.get("target_node_id"),
            metadata=data.get("metadata", {})
        )


class EventGenerator:
    """
    Generiert Ereignisse für die PRISM-Simulation basierend auf Wahrscheinlichkeitsmodellen
    und Zeitlinienanalysen.
    """
    
    def __init__(self, config: Dict[str, Any] = None, prism_engine = None):
        """
        Initialisiert den EventGenerator
        
        Args:
            config: Konfigurationsparameter
            prism_engine: Instanz der PRISM-Engine für die Integration
        """
        self.config = config or {}
        self.event_templates = {}
        self.probability_models = {}
        self.event_history = []
        self.max_history_size = self.config.get("max_history_size", 1000)
        self.mprime_engine = None
        self.prob_mapper = None
        self.prism_engine = prism_engine
        
        # Initialisiere Abhängigkeiten
        if HAS_DEPENDENCIES:
            self.initialize_dependencies()
        
        # Lade vordefinierte Ereignisvorlagen
        self.load_event_templates()
        
        logger.info("EventGenerator initialisiert" + (" mit PRISM-Engine" if prism_engine else ""))
    
    def initialize_dependencies(self):
        """Initialisiert Abhängigkeiten zu anderen MISO-Systemen"""
        try:
            self.mprime_engine = MPrimeEngine()
            self.prob_mapper = ProbabilisticMapper()
            
            # Prüfe, ob die PRISM-Engine verfügbar ist
            if self.prism_engine is None:
                try:
                    # Importiere PrismEngine nur wenn nötig, um zirkuläre Importe zu vermeiden
                    # Verwende from miso.simulation.prism_base importierte Basisklassen
                    from miso.simulation.prism_engine import PrismEngine
                    self.prism_engine = PrismEngine()
                    logger.info("PRISM-Engine automatisch initialisiert")
                except ImportError:
                    logger.warning("PRISM-Engine konnte nicht automatisch initialisiert werden")
            
            logger.info("Abhängigkeiten erfolgreich initialisiert")
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren der Abhängigkeiten: {e}")
    
    def load_event_templates(self):
        """Lädt vordefinierte Ereignisvorlagen"""
        # Hier könnten Vorlagen aus einer Datei oder Datenbank geladen werden
        # Für jetzt verwenden wir einige hartcodierte Vorlagen
        self.event_templates = {
            "timeline_branch": {
                "type": EventType.TEMPORAL,
                "data": {
                    "action": "branch",
                    "branch_factor": 2,
                    "branch_probability": 0.5
                },
                "metadata": {
                    "description": "Verzweigung einer Zeitlinie in mehrere Alternativen"
                }
            },
            "causal_link": {
                "type": EventType.CAUSAL,
                "data": {
                    "action": "link",
                    "strength": 0.7
                },
                "metadata": {
                    "description": "Kausale Verbindung zwischen zwei Ereignissen"
                }
            },
            "quantum_fluctuation": {
                "type": EventType.QUANTUM,
                "data": {
                    "action": "fluctuate",
                    "amplitude": 0.3
                },
                "metadata": {
                    "description": "Quantenfluktuation mit Auswirkungen auf Wahrscheinlichkeiten"
                }
            },
            "paradox_detection": {
                "type": EventType.PARADOX,
                "data": {
                    "action": "detect",
                    "severity": 0.8
                },
                "metadata": {
                    "description": "Erkennung einer potenziellen Paradoxie"
                }
            }
        }
        logger.info(f"{len(self.event_templates)} Ereignisvorlagen geladen")
    
    def create_event(self, event_type: EventType, data: Dict[str, Any] = None, 
                     source_timeline_id: str = None, target_timeline_id: str = None,
                     source_node_id: str = None, target_node_id: str = None,
                     probability: float = None) -> SimulationEvent:
        """
        Erstellt ein neues Ereignis
        
        Args:
            event_type: Typ des Ereignisses
            data: Ereignisdaten
            source_timeline_id: ID der Quellzeitlinie
            target_timeline_id: ID der Zielzeitlinie
            source_node_id: ID des Quellknotens
            target_node_id: ID des Zielknotens
            probability: Wahrscheinlichkeit des Ereignisses (0.0 bis 1.0)
            
        Returns:
            Neues SimulationEvent
        """
        if probability is None:
            # Berechne Wahrscheinlichkeit basierend auf Ereignistyp und Daten
            probability = self._calculate_event_probability(event_type, data)
        
        event = SimulationEvent(
            type=event_type,
            data=data or {},
            probability=probability,
            source_timeline_id=source_timeline_id,
            target_timeline_id=target_timeline_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id
        )
        
        # Füge das Ereignis zur Historie hinzu
        self._add_to_history(event)
        
        return event
    
    def create_event_from_template(self, template_name: str, 
                                  override_data: Dict[str, Any] = None,
                                  source_timeline_id: str = None, 
                                  target_timeline_id: str = None,
                                  source_node_id: str = None, 
                                  target_node_id: str = None) -> Optional[SimulationEvent]:
        """
        Erstellt ein Ereignis basierend auf einer Vorlage
        
        Args:
            template_name: Name der Vorlage
            override_data: Daten, die die Vorlagendaten überschreiben
            source_timeline_id: ID der Quellzeitlinie
            target_timeline_id: ID der Zielzeitlinie
            source_node_id: ID des Quellknotens
            target_node_id: ID des Zielknotens
            
        Returns:
            Neues SimulationEvent oder None, wenn die Vorlage nicht existiert
        """
        if template_name not in self.event_templates:
            logger.warning(f"Ereignisvorlage '{template_name}' nicht gefunden")
            return None
        
        template = self.event_templates[template_name]
        event_type = template["type"]
        data = template["data"].copy()
        
        # Überschreibe Daten, falls angegeben
        if override_data:
            data.update(override_data)
        
        # Erstelle das Ereignis
        event = self.create_event(
            event_type=event_type,
            data=data,
            source_timeline_id=source_timeline_id,
            target_timeline_id=target_timeline_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id
        )
        
        # Füge Metadaten aus der Vorlage hinzu
        if "metadata" in template:
            event.metadata.update(template["metadata"])
        
        return event
    
    def generate_random_event(self, timeline_ids: List[str] = None, 
                             node_ids: List[str] = None) -> SimulationEvent:
        """
        Generiert ein zufälliges Ereignis
        
        Args:
            timeline_ids: Liste von verfügbaren Zeitlinien-IDs
            node_ids: Liste von verfügbaren Knoten-IDs
            
        Returns:
            Zufälliges SimulationEvent
        """
        # Wähle einen zufälligen Ereignistyp
        event_type = random.choice(list(EventType))
        
        # Erstelle zufällige Daten basierend auf dem Ereignistyp
        data = self._generate_random_data_for_event_type(event_type)
        
        # Wähle zufällige Quell- und Zielzeitlinien und -knoten, falls verfügbar
        source_timeline_id = random.choice(timeline_ids) if timeline_ids else None
        target_timeline_id = random.choice(timeline_ids) if timeline_ids else None
        source_node_id = random.choice(node_ids) if node_ids else None
        target_node_id = random.choice(node_ids) if node_ids else None
        
        # Generiere eine zufällige Wahrscheinlichkeit
        probability = random.random()
        
        return self.create_event(
            event_type=event_type,
            data=data,
            source_timeline_id=source_timeline_id,
            target_timeline_id=target_timeline_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            probability=probability
        )
    
    def generate_events_for_timeline(self, timeline_id: str, 
                                    node_ids: List[str] = None,
                                    count: int = 5) -> List[SimulationEvent]:
        """
        Generiert mehrere Ereignisse für eine Zeitlinie
        
        Args:
            timeline_id: ID der Zeitlinie
            node_ids: Liste von verfügbaren Knoten-IDs
            count: Anzahl der zu generierenden Ereignisse
            
        Returns:
            Liste von SimulationEvent-Objekten
        """
        events = []
        
        # Wenn PRISM-Engine verfügbar ist, nutze ihre Wahrscheinlichkeitsmodelle
        if self.prism_engine is not None:
            try:
                # Hole Zeitliniendaten aus der PRISM-Engine, falls verfügbar
                timeline_data = None
                if hasattr(self.prism_engine, 'time_scope') and timeline_id:
                    timeline_data = self.prism_engine.time_scope.get_timeline_data(timeline_id)
                
                # Nutze die Wahrscheinlichkeitsmodelle der PRISM-Engine für bessere Ereignisgenerierung
                for _ in range(count):
                    # Wähle Ereignistyp basierend auf PRISM-Wahrscheinlichkeiten, falls verfügbar
                    if timeline_data and 'event_probabilities' in timeline_data:
                        weights = [timeline_data['event_probabilities'].get(et.name, 1.0) for et in EventType]
                        event_type = random.choices(list(EventType), weights=weights, k=1)[0]
                    else:
                        event_type = random.choice(list(EventType))
                    
                    # Wähle Knoten basierend auf PRISM-Daten, falls verfügbar
                    source_node = None
                    target_node = None
                    if node_ids:
                        source_node = random.choice(node_ids)
                        if len(node_ids) > 1:
                            target_node = random.choice([n for n in node_ids if n != source_node])
                    
                    # Generiere Daten mit PRISM-Kontext
                    data = self._generate_random_data_for_event_type(event_type)
                    
                    # Erstelle Ereignis
                    event = self.create_event(
                        event_type=event_type,
                        data=data,
                        source_timeline_id=timeline_id,
                        target_timeline_id=timeline_id,
                        source_node_id=source_node,
                        target_node_id=target_node
                    )
                    
                    events.append(event)
            except Exception as e:
                logger.warning(f"Fehler bei der Verwendung der PRISM-Engine für Ereignisgenerierung: {e}")
                # Fallback auf Standard-Implementierung
        
        # Fallback, wenn keine Ereignisse generiert wurden oder PRISM-Engine nicht verfügbar ist
        if not events:
            for _ in range(count):
                # Generiere ein Ereignis mit der angegebenen Zeitlinie als Quelle oder Ziel
                is_source = random.choice([True, False])
                
                source_timeline_id = timeline_id if is_source else None
                target_timeline_id = None if is_source else timeline_id
                
                source_node_id = random.choice(node_ids) if node_ids and is_source else None
                target_node_id = random.choice(node_ids) if node_ids and not is_source else None
                
                event = self.generate_random_event(
                    timeline_ids=[timeline_id],
                    node_ids=node_ids
                )
                
                events.append(event)
        
        return events
    
    def _calculate_event_probability(self, event_type: EventType, 
                                    data: Optional[Dict[str, Any]] = None) -> float:
        """
        Berechnet die Wahrscheinlichkeit eines Ereignisses basierend auf Typ und Daten
        
        Args:
            event_type: Typ des Ereignisses
            data: Ereignisdaten
            
        Returns:
            Wahrscheinlichkeit (0.0 bis 1.0)
        """
        if data is None:
            data = {}
        
        # Verwende die PRISM-Engine für fortgeschrittene Wahrscheinlichkeitsberechnungen, falls verfügbar
        if self.prism_engine is not None:
            try:
                # Konvertiere Ereignistyp und Daten in ein Format für die PRISM-Engine
                event_data = {
                    "type": event_type.name,
                    "data": data
                }
                
                # Nutze die Wahrscheinlichkeitsanalyse der PRISM-Engine
                if hasattr(self.prism_engine, 'evaluate_probability_recommendation'):
                    recommendation = self.prism_engine.evaluate_probability_recommendation(
                        0.5  # Übergebe Standardwahrscheinlichkeit als Ausgangspunkt
                    )
                    if recommendation and 'adjusted_probability' in recommendation:
                        return max(0.0, min(1.0, recommendation['adjusted_probability']))
                
                # Alternativ: Verwende die Matrix für Wahrscheinlichkeitsberechnung
                if hasattr(self.prism_engine, 'matrix'):
                    # Bestimme Koordinaten für das Ereignis in der Matrix
                    if hasattr(self.prism_engine, '_determine_coordinates_for_stream'):
                        stream_id = f"event_{event_type.name.lower()}"
                        coordinates = self.prism_engine._determine_coordinates_for_stream(stream_id, event_data)
                        if coordinates:
                            # Hole Wahrscheinlichkeitswert aus der Matrix
                            probability = self.prism_engine.matrix.get_probability_at_coordinates(coordinates)
                            if probability is not None:
                                return max(0.0, min(1.0, probability))
            except Exception as e:
                logger.warning(f"Fehler bei der Verwendung der PRISM-Engine für Wahrscheinlichkeitsberechnung: {e}")
        
        # Verwende M-PRIME und ProbabilisticMapper als Fallback, falls verfügbar
        if self.mprime_engine and self.prob_mapper:
            try:
                # Konvertiere Ereignistyp und Daten in ein Format für den ProbabilisticMapper
                event_data = {
                    "type": event_type.name,
                    "data": data
                }
                
                # Berechne Wahrscheinlichkeit mit dem ProbabilisticMapper
                return self.prob_mapper.calculate_probability(event_data)
            except Exception as e:
                logger.warning(f"Fehler bei der Wahrscheinlichkeitsberechnung: {e}")
        
        # Fallback: Einfache Berechnung basierend auf Ereignistyp
        base_probabilities = {
            EventType.TEMPORAL: 0.8,
            EventType.CAUSAL: 0.7,
            EventType.PROBABILISTIC: 0.6,
            EventType.QUANTUM: 0.4,
            EventType.PARADOX: 0.2,
            EventType.SYSTEM: 0.9,
            EventType.USER: 0.95,
            EventType.EXTERNAL: 0.5
        }
        
        # Basiswahrscheinlichkeit für den Ereignistyp
        base_prob = base_probabilities.get(event_type, 0.5)
        
        # Modifikator basierend auf Daten (falls vorhanden)
        modifier = 0.0
        if "probability_modifier" in data:
            modifier = data["probability_modifier"]
        elif "severity" in data:
            modifier = data["severity"] * 0.2 - 0.1  # Schweregrad beeinflusst die Wahrscheinlichkeit
        elif "strength" in data:
            modifier = data["strength"] * 0.1        # Stärke beeinflusst die Wahrscheinlichkeit
        
        # Kombiniere Basiswahrscheinlichkeit und Modifikator
        probability = max(0.0, min(1.0, base_prob + modifier))
        
        return probability
    
    def _generate_random_data_for_event_type(self, event_type: EventType) -> Dict[str, Any]:
        """
        Generiert zufällige Daten für einen Ereignistyp
        
        Args:
            event_type: Typ des Ereignisses
            
        Returns:
            Zufällige Ereignisdaten
        """
        data = {}
        
        if event_type == EventType.TEMPORAL:
            data = {
                "action": random.choice(["create", "modify", "delete", "branch", "merge"]),
                "timestamp_offset": random.uniform(-10.0, 10.0),
                "duration": random.uniform(0.1, 5.0)
            }
        elif event_type == EventType.CAUSAL:
            data = {
                "action": random.choice(["link", "unlink", "strengthen", "weaken"]),
                "strength": random.uniform(0.1, 1.0),
                "bidirectional": random.choice([True, False])
            }
        elif event_type == EventType.PROBABILISTIC:
            data = {
                "action": random.choice(["shift", "amplify", "dampen", "invert"]),
                "magnitude": random.uniform(0.1, 0.9),
                "target_probability": random.uniform(0.0, 1.0)
            }
        elif event_type == EventType.QUANTUM:
            data = {
                "action": random.choice(["fluctuate", "entangle", "collapse", "superpose"]),
                "amplitude": random.uniform(0.1, 0.5),
                "coherence": random.uniform(0.3, 0.9)
            }
        elif event_type == EventType.PARADOX:
            data = {
                "action": random.choice(["detect", "create", "resolve", "amplify"]),
                "severity": random.uniform(0.1, 1.0),
                "type": random.choice(["causal_loop", "grandfather", "bootstrap", "predestination"])
            }
        elif event_type == EventType.SYSTEM:
            data = {
                "action": random.choice(["start", "stop", "pause", "resume", "reset"]),
                "component": random.choice(["timeline", "node", "engine", "analyzer", "scanner"]),
                "priority": random.randint(1, 10)
            }
        elif event_type == EventType.USER:
            data = {
                "action": random.choice(["input", "query", "command", "feedback"]),
                "content": f"User action {uuid.uuid4().hex[:8]}",
                "priority": random.randint(1, 10)
            }
        elif event_type == EventType.EXTERNAL:
            data = {
                "action": random.choice(["import", "export", "integrate", "disconnect"]),
                "source": random.choice(["database", "api", "file", "stream", "sensor"]),
                "format": random.choice(["json", "xml", "binary", "text"])
            }
        
        return data
    
    def _add_to_history(self, event: SimulationEvent):
        """
        Fügt ein Ereignis zur Historie hinzu
        
        Args:
            event: Hinzuzufügendes Ereignis
        """
        self.event_history.append(event)
        
        # Begrenze die Größe der Historie
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]
    
    def get_event_history(self) -> List[SimulationEvent]:
        """
        Gibt die Ereignishistorie zurück
        
        Returns:
            Liste von SimulationEvent-Objekten
        """
        return self.event_history
    
    def clear_history(self):
        """Löscht die Ereignishistorie"""
        self.event_history = []
        logger.info("Ereignishistorie gelöscht")
    
    def save_event_templates(self, filepath: str) -> bool:
        """
        Speichert Ereignisvorlagen in eine Datei
        
        Args:
            filepath: Pfad zur Zieldatei
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Konvertiere Vorlagen in ein serialisierbares Format
            serializable_templates = {}
            for name, template in self.event_templates.items():
                serializable_template = template.copy()
                if "type" in serializable_template and isinstance(serializable_template["type"], EventType):
                    serializable_template["type"] = serializable_template["type"].name
                serializable_templates[name] = serializable_template
            
            with open(filepath, 'w') as f:
                json.dump(serializable_templates, f, indent=2)
            
            logger.info(f"Ereignisvorlagen gespeichert in {filepath}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Ereignisvorlagen: {e}")
            return False
    
    def load_event_templates_from_file(self, filepath: str) -> bool:
        """
        Lädt Ereignisvorlagen aus einer Datei
        
        Args:
            filepath: Pfad zur Quelldatei
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            with open(filepath, 'r') as f:
                templates = json.load(f)
            
            # Konvertiere Vorlagen in das richtige Format
            for name, template in templates.items():
                if "type" in template and isinstance(template["type"], str):
                    template["type"] = EventType[template["type"]]
            
            self.event_templates = templates
            logger.info(f"{len(self.event_templates)} Ereignisvorlagen geladen aus {filepath}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Laden der Ereignisvorlagen: {e}")
            return False


# Beispiel für die Verwendung des EventGenerator
if __name__ == "__main__":
    # Konfiguriere Logging für Standalone-Ausführung
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Erstelle einen EventGenerator
    generator = EventGenerator()
    
    # Generiere einige Ereignisse
    events = []
    for _ in range(5):
        event = generator.generate_random_event()
        events.append(event)
        print(f"Generiertes Ereignis: {event.type.name}, Wahrscheinlichkeit: {event.probability:.2f}")
    
    # Erstelle ein Ereignis aus einer Vorlage
    template_event = generator.create_event_from_template("quantum_fluctuation")
    if template_event:
        print(f"Ereignis aus Vorlage: {template_event.type.name}, Daten: {template_event.data}")
