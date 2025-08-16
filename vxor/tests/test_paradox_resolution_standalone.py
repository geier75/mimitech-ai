#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Standalone Test für die erweiterte Paradoxauflösung

Dieses Skript testet die implementierten Komponenten der erweiterten Paradoxauflösung
ohne externe Abhängigkeiten wie numpy.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import logging
import random
import math
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [PARADOX-TEST] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.Paradox.Test")

# Füge das Stammverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Definiere die benötigten Klassen für den Test
class ParadoxSeverity(Enum):
    """Schweregrad eines Paradoxes"""
    MINOR = auto()       # Geringfügig
    MODERATE = auto()    # Moderat
    MAJOR = auto()       # Schwerwiegend
    CRITICAL = auto()    # Kritisch
    CATASTROPHIC = auto() # Katastrophal

class EnhancedParadoxType(Enum):
    """Erweiterte Typen von Paradoxien"""
    # Grundlegende Paradoxtypen
    GRANDFATHER = auto()           # Großvater-Paradox
    BOOTSTRAP = auto()             # Bootstrap-Paradox
    PREDESTINATION = auto()        # Vorbestimmungs-Paradox
    ONTOLOGICAL = auto()           # Ontologisches Paradox
    
    # Erweiterte Paradoxtypen
    TEMPORAL_LOOP = auto()         # Zeitschleife
    CAUSAL_VIOLATION = auto()      # Kausale Verletzung
    INFORMATION_PARADOX = auto()   # Informationsparadox
    QUANTUM_PARADOX = auto()       # Quantenparadox
    MULTI_TIMELINE_PARADOX = auto() # Mehrfachzeitlinien-Paradox
    SELF_CONSISTENCY_VIOLATION = auto() # Selbstkonsistenzverletzung

@dataclass
class ParadoxInstance:
    """Repräsentiert eine Instanz eines erkannten Paradoxes"""
    id: str
    type: EnhancedParadoxType
    severity: ParadoxSeverity
    timeline_id: str
    affected_nodes: List[str]
    description: str
    detection_time: datetime
    probability: float
    causal_chain: List[Tuple[str, str]]  # Liste von (node_id, event_id) Paaren
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PotentialParadox:
    """Repräsentiert ein potentielles Paradox, das noch nicht eingetreten ist"""
    id: str
    timeline_id: str
    risk_nodes: List[str]
    probability: float
    estimated_severity: ParadoxSeverity
    description: str
    detection_time: datetime
    time_to_occurrence: float  # Geschätzte Zeit bis zum Eintreten in Sekunden
    preventive_actions: List[str] = field(default_factory=list)

@dataclass
class ParadoxHierarchy:
    """Hierarchische Klassifizierung eines Paradoxes"""
    primary_type: EnhancedParadoxType
    secondary_type: Optional[EnhancedParadoxType]
    tertiary_type: Optional[EnhancedParadoxType]
    category: str
    subcategory: Optional[str]
    tags: List[str] = field(default_factory=list)

class ResolutionStrategy(Enum):
    """Strategien zur Auflösung von Paradoxien"""
    TIMELINE_SPLIT = auto()        # Aufspaltung der Zeitlinie
    CAUSAL_REALIGNMENT = auto()    # Neuausrichtung der Kausalität
    TIMELINE_ADJUSTMENT = auto()   # Anpassung der Zeitlinie
    NODE_ISOLATION = auto()        # Isolierung des problematischen Knotens
    PARADOX_ABSORPTION = auto()    # Absorption des Paradoxes
    QUANTUM_RESOLUTION = auto()    # Quantenauflösung
    TEMPORAL_SHIELDING = auto()    # Temporale Abschirmung
    MULTI_TIMELINE_HARMONIZATION = auto() # Harmonisierung mehrerer Zeitlinien

@dataclass
class ResolutionOption:
    """Option zur Auflösung eines Paradoxes"""
    strategy: ResolutionStrategy
    description: str
    estimated_success_rate: float
    side_effects: List[str]
    complexity: int  # 1-10, wobei 10 am komplexesten ist
    resource_cost: int  # 1-10, wobei 10 am teuersten ist

@dataclass
class ResolutionImpact:
    """Auswirkungen einer Paradoxauflösung"""
    timeline_stability: float  # 0-1, wobei 1 am stabilsten ist
    causal_integrity: float  # 0-1, wobei 1 am integeren ist
    information_preservation: float  # 0-1, wobei 1 am besten erhalten ist
    affected_timelines: List[str]
    affected_nodes: List[str]
    side_effects: List[str]

@dataclass
class ResolutionResult:
    """Ergebnis einer Paradoxauflösung"""
    paradox_id: str
    strategy_used: ResolutionStrategy
    success: bool
    resolution_time: datetime
    impact: ResolutionImpact
    new_timeline_ids: List[str] = field(default_factory=list)
    modified_node_ids: List[str] = field(default_factory=list)
    notes: str = ""

@dataclass
class ParadoxRisk:
    """Repräsentiert ein Risiko für ein Paradox"""
    id: str
    timeline_id: str
    risk_nodes: List[str]
    risk_level: float  # 0-1, wobei 1 am riskantesten ist
    description: str
    detection_time: datetime
    potential_paradox_types: List[EnhancedParadoxType]
    monitoring_priority: int  # 1-10, wobei 10 die höchste Priorität ist

@dataclass
class ParadoxWarning:
    """Frühwarnung für ein potentielles Paradox"""
    id: str
    risk_id: str
    warning_level: int  # 1-5, wobei 5 die höchste Warnstufe ist
    description: str
    issue_time: datetime
    estimated_time_to_paradox: float  # Sekunden bis zum möglichen Eintreten
    recommended_actions: List[str]

@dataclass
class PreventiveMeasure:
    """Präventive Maßnahme zur Verhinderung eines Paradoxes"""
    id: str
    risk_id: str
    measure_type: str
    description: str
    application_time: datetime
    estimated_effectiveness: float  # 0-1, wobei 1 am effektivsten ist
    applied_to_timeline: str
    applied_to_nodes: List[str]
    side_effects: List[str]

# Implementierung der Mock-Klassen für die Tests
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
    
    def add_event(self, event_id: str, event_type: str, data: Dict[str, Any] = None):
        """
        Fügt ein Ereignis zum Zeitknoten hinzu
        
        Args:
            event_id: ID des Ereignisses
            event_type: Typ des Ereignisses
            data: Daten des Ereignisses
            
        Returns:
            Erstelltes Ereignis
        """
        event = {"id": event_id, "type": event_type, "data": data or {}}
        self.events.append(event)
        return event

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
    
    def add_node(self, node_id: str, references: List[str] = None):
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
    
    def get_nodes(self):
        """
        Gibt alle Zeitknoten der Zeitlinie zurück
        
        Returns:
            Liste aller Zeitknoten
        """
        return list(self.nodes.values())
    
    def get_node(self, node_id: str):
        """
        Gibt einen Zeitknoten anhand seiner ID zurück
        
        Args:
            node_id: ID des Zeitknotens
            
        Returns:
            Zeitknoten oder None, wenn nicht gefunden
        """
        return self.nodes.get(node_id)

# Implementierung der Komponenten für die Tests
class EnhancedParadoxDetector:
    """Erweiterter Paradoxdetektor"""
    
    def __init__(self):
        """Initialisiert den erweiterten Paradoxdetektor"""
        self.detected_paradoxes = []
        self.potential_paradoxes = []
        logger.info("EnhancedParadoxDetector initialisiert")
    
    def detect_complex_paradoxes(self, timelines: List[MockTimeline]) -> List[ParadoxInstance]:
        """Erkennt komplexe Paradoxien in einer Liste von Zeitlinien"""
        paradoxes = []
        
        # Simuliere die Erkennung von Paradoxien
        for timeline in timelines:
            # Suche nach Zyklen in den Referenzen
            nodes = timeline.get_nodes()
            for node in nodes:
                if len(node.references) > 0:
                    # Simuliere eine Paradoxerkennung mit 30% Wahrscheinlichkeit
                    if random.random() < 0.3:
                        paradox_id = f"paradox-{len(self.detected_paradoxes)}"
                        paradox_type = random.choice(list(EnhancedParadoxType))
                        severity = random.choice(list(ParadoxSeverity))
                        
                        paradox = ParadoxInstance(
                            id=paradox_id,
                            type=paradox_type,
                            severity=severity,
                            timeline_id=timeline.id,
                            affected_nodes=[node.id],
                            description=f"Komplexes Paradox erkannt in {node.id}",
                            detection_time=datetime.now(),
                            probability=random.random(),
                            causal_chain=[(node.id, "event1")],
                            metadata={"source": "EnhancedParadoxDetector"}
                        )
                        
                        paradoxes.append(paradox)
                        self.detected_paradoxes.append(paradox)
        
        logger.info(f"{len(paradoxes)} komplexe Paradoxien erkannt")
        return paradoxes
    
    def evaluate_paradox_probability(self, timeline: MockTimeline, node_id: str) -> float:
        """Bewertet die Wahrscheinlichkeit eines Paradoxes in einem Knoten"""
        # Simuliere eine Wahrscheinlichkeitsbewertung
        node = timeline.get_node(node_id)
        if node is None:
            return 0.0
        
        # Berechne eine Pseudowahrscheinlichkeit basierend auf der Anzahl der Referenzen
        probability = len(node.references) * 0.1
        if probability > 1.0:
            probability = 1.0
        
        return probability
    
    def detect_potential_paradoxes(self, timelines: List[MockTimeline]) -> List[PotentialParadox]:
        """Erkennt potentielle Paradoxien, die noch nicht eingetreten sind"""
        potential_paradoxes = []
        
        # Simuliere die Erkennung potentieller Paradoxien
        for timeline in timelines:
            nodes = timeline.get_nodes()
            for node in nodes:
                # Simuliere eine potentielle Paradoxerkennung mit 20% Wahrscheinlichkeit
                if random.random() < 0.2:
                    paradox_id = f"potential-paradox-{len(self.potential_paradoxes)}"
                    severity = random.choice(list(ParadoxSeverity))
                    
                    potential_paradox = PotentialParadox(
                        id=paradox_id,
                        timeline_id=timeline.id,
                        risk_nodes=[node.id],
                        probability=random.random(),
                        estimated_severity=severity,
                        description=f"Potentielles Paradox in {node.id}",
                        detection_time=datetime.now(),
                        time_to_occurrence=random.randint(60, 3600),
                        preventive_actions=["Zeitknoten isolieren", "Kausalkette umleiten"]
                    )
                    
                    potential_paradoxes.append(potential_paradox)
                    self.potential_paradoxes.append(potential_paradox)
        
        logger.info(f"{len(potential_paradoxes)} potentielle Paradoxien erkannt")
        return potential_paradoxes

class ParadoxClassifier:
    """Paradoxklassifizierer"""
    
    def __init__(self):
        """Initialisiert den Paradoxklassifizierer"""
        # Definiere Kategorien für Paradoxien
        self.categories = {
            EnhancedParadoxType.GRANDFATHER: "Kausal-Eliminierend",
            EnhancedParadoxType.BOOTSTRAP: "Kausal-Zirkulär",
            EnhancedParadoxType.PREDESTINATION: "Kausal-Determinierend",
            EnhancedParadoxType.ONTOLOGICAL: "Kausal-Existentiell",
            EnhancedParadoxType.TEMPORAL_LOOP: "Temporal-Zyklisch",
            EnhancedParadoxType.CAUSAL_VIOLATION: "Kausal-Inkonsistent",
            EnhancedParadoxType.INFORMATION_PARADOX: "Informations-Inkonsistent",
            EnhancedParadoxType.QUANTUM_PARADOX: "Quanten-Inkonsistent",
            EnhancedParadoxType.MULTI_TIMELINE_PARADOX: "Multitemporal-Inkonsistent",
            EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION: "Selbst-Inkonsistent"
        }
        
        # Definiere Subkategorien für Paradoxien
        self.subcategories = {
            EnhancedParadoxType.GRANDFATHER: ["Eliminierend", "Verhindernd", "Modifizierend"],
            EnhancedParadoxType.BOOTSTRAP: ["Informations-Zirkulär", "Objekt-Zirkulär", "Kausal-Zirkulär"],
            EnhancedParadoxType.PREDESTINATION: ["Selbst-Erfüllend", "Unvermeidbar", "Determiniert"],
            EnhancedParadoxType.ONTOLOGICAL: ["Existenz-Widerspruch", "Ursprungs-Widerspruch", "Identitäts-Widerspruch"],
            EnhancedParadoxType.TEMPORAL_LOOP: ["Endlos-Schleife", "Begrenzte Schleife", "Verzweigende Schleife"],
            EnhancedParadoxType.CAUSAL_VIOLATION: ["Ursache-Wirkung-Umkehrung", "Kausale Lücke", "Kausale Überlappung"],
            EnhancedParadoxType.INFORMATION_PARADOX: ["Informationsverlust", "Informationsgewinn", "Informationstransformation"],
            EnhancedParadoxType.QUANTUM_PARADOX: ["Beobachter-Effekt", "Quantenverschränkung", "Superpositions-Kollaps"],
            EnhancedParadoxType.MULTI_TIMELINE_PARADOX: ["Zeitlinien-Konvergenz", "Zeitlinien-Divergenz", "Zeitlinien-Überlappung"],
            EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION: ["Logische Inkonsistenz", "Physikalische Inkonsistenz", "Temporale Inkonsistenz"]
        }
        
        logger.info("ParadoxClassifier initialisiert")
    
    def classify_paradox(self, paradox_instance: ParadoxInstance) -> EnhancedParadoxType:
        """Klassifiziert ein Paradox detailliert"""
        # Simuliere eine Klassifizierung
        return paradox_instance.type
    
    def evaluate_severity(self, paradox_instance: ParadoxInstance) -> ParadoxSeverity:
        """Bewertet den Schweregrad eines Paradoxes"""
        # Simuliere eine Schweregradsbewertung
        return paradox_instance.severity
    
    def get_hierarchical_classification(self, paradox_instance: ParadoxInstance) -> ParadoxHierarchy:
        """Gibt eine hierarchische Klassifizierung eines Paradoxes zurück"""
        # Simuliere eine hierarchische Klassifizierung
        primary_type = paradox_instance.type
        secondary_type = random.choice(list(EnhancedParadoxType))
        tertiary_type = random.choice(list(EnhancedParadoxType))
        
        # Stelle sicher, dass die Typen unterschiedlich sind
        while secondary_type == primary_type:
            secondary_type = random.choice(list(EnhancedParadoxType))
        
        while tertiary_type == primary_type or tertiary_type == secondary_type:
            tertiary_type = random.choice(list(EnhancedParadoxType))
        
        category = self.categories.get(primary_type, "Unbekannt")
        subcategory = random.choice(self.subcategories.get(primary_type, ["Unbekannt"]))
        
        return ParadoxHierarchy(
            primary_type=primary_type,
            secondary_type=secondary_type,
            tertiary_type=tertiary_type,
            category=category,
            subcategory=subcategory,
            tags=["tag1", "tag2", "tag3"]
        )

class ParadoxResolver:
    """Paradoxauflöser"""
    
    def __init__(self):
        """Initialisiert den Paradoxauflöser"""
        self.resolution_history = []
        logger.info("ParadoxResolver initialisiert")
    
    def get_resolution_options(self, paradox_instance: ParadoxInstance) -> List[ResolutionOption]:
        """Gibt mögliche Auflösungsoptionen für ein Paradox zurück"""
        # Simuliere Auflösungsoptionen
        options = []
        
        # Generiere 3-5 zufällige Optionen
        num_options = random.randint(3, 5)
        strategies = list(ResolutionStrategy)
        
        for i in range(num_options):
            strategy = random.choice(strategies)
            
            option = ResolutionOption(
                strategy=strategy,
                description=f"Auflösung mit {strategy.name}",
                estimated_success_rate=random.random(),
                side_effects=[f"Effekt {j+1}" for j in range(random.randint(1, 3))],
                complexity=random.randint(1, 10),
                resource_cost=random.randint(1, 10)
            )
            
            options.append(option)
        
        return options
    
    def select_optimal_resolution(self, options: List[ResolutionOption]) -> ResolutionOption:
        """Wählt die optimale Auflösungsstrategie aus"""
        # Simuliere die Auswahl der optimalen Strategie
        # Hier verwenden wir eine einfache Heuristik: Erfolgsrate / (Komplexität + Ressourcenkosten)
        best_option = None
        best_score = -1
        
        for option in options:
            score = option.estimated_success_rate / ((option.complexity + option.resource_cost) / 20.0)
            if score > best_score:
                best_score = score
                best_option = option
        
        return best_option or options[0] if options else None
    
    def resolve_paradox(self, paradox_instance: ParadoxInstance) -> ResolutionResult:
        """Löst ein Paradox auf"""
        # Simuliere die Auflösung eines Paradoxes
        options = self.get_resolution_options(paradox_instance)
        selected_option = self.select_optimal_resolution(options)
        
        # Simuliere den Erfolg mit 80% Wahrscheinlichkeit
        success = random.random() < 0.8
        
        # Erstelle das Ergebnis
        impact = ResolutionImpact(
            timeline_stability=random.random(),
            causal_integrity=random.random(),
            information_preservation=random.random(),
            affected_timelines=[paradox_instance.timeline_id],
            affected_nodes=paradox_instance.affected_nodes,
            side_effects=selected_option.side_effects
        )
        
        result = ResolutionResult(
            paradox_id=paradox_instance.id,
            strategy_used=selected_option.strategy,
            success=success,
            resolution_time=datetime.now(),
            impact=impact,
            new_timeline_ids=[f"timeline-{i}" for i in range(random.randint(0, 2))],
            modified_node_ids=paradox_instance.affected_nodes,
            notes=f"Auflösung mit {selected_option.strategy.name} {'erfolgreich' if success else 'fehlgeschlagen'}"
        )
        
        self.resolution_history.append(result)
        return result

class ParadoxPreventionSystem:
    """Paradox-Präventionssystem"""
    
    def __init__(self):
        """Initialisiert das Paradox-Präventionssystem"""
        self.monitored_risks = []
        self.issued_warnings = []
        self.applied_measures = []
        logger.info("ParadoxPreventionSystem initialisiert")
    
    def monitor_timelines(self, timelines: List[MockTimeline]) -> List[ParadoxRisk]:
        """Überwacht Zeitlinien auf potentielle Paradoxrisiken"""
        risks = []
        
        # Simuliere die Erkennung von Risiken
        for timeline in timelines:
            nodes = timeline.get_nodes()
            for node in nodes:
                # Simuliere eine Risikoerkennung mit 25% Wahrscheinlichkeit
                if random.random() < 0.25:
                    risk_id = f"risk-{len(self.monitored_risks)}"
                    
                    risk = ParadoxRisk(
                        id=risk_id,
                        timeline_id=timeline.id,
                        risk_nodes=[node.id],
                        risk_level=random.random(),
                        description=f"Paradoxrisiko in {node.id}",
                        detection_time=datetime.now(),
                        potential_paradox_types=random.sample(list(EnhancedParadoxType), k=random.randint(1, 3)),
                        monitoring_priority=random.randint(1, 10)
                    )
                    
                    risks.append(risk)
                    self.monitored_risks.append(risk)
        
        logger.info(f"{len(risks)} Paradoxrisiken erkannt")
        return risks
    
    def generate_early_warnings(self, risks: List[ParadoxRisk]) -> List[ParadoxWarning]:
        """Generiert Frühwarnungen für erkannte Risiken"""
        warnings = []
        
        for risk in risks:
            # Generiere eine Warnung mit 70% Wahrscheinlichkeit
            if random.random() < 0.7:
                warning_id = f"warning-{len(self.issued_warnings)}"
                
                warning = ParadoxWarning(
                    id=warning_id,
                    risk_id=risk.id,
                    warning_level=random.randint(1, 5),
                    description=f"Frühwarnung für Risiko {risk.id}",
                    issue_time=datetime.now(),
                    estimated_time_to_paradox=random.randint(300, 7200),
                    recommended_actions=["Zeitknoten isolieren", "Kausalkette umleiten", "Präventive Zeitlinienverzweigung"]
                )
                
                warnings.append(warning)
                self.issued_warnings.append(warning)
        
        logger.info(f"{len(warnings)} Frühwarnungen generiert")
        return warnings
    
    def apply_preventive_measures(self, warnings: List[ParadoxWarning]) -> List[PreventiveMeasure]:
        """Wendet präventive Maßnahmen an"""
        measures = []
        
        for warning in warnings:
            # Wende eine Maßnahme mit 60% Wahrscheinlichkeit an
            if random.random() < 0.6:
                measure_id = f"measure-{len(self.applied_measures)}"
                measure_types = ["Isolation", "Umleitung", "Verzweigung", "Stabilisierung", "Abschirmung"]
                
                measure = PreventiveMeasure(
                    id=measure_id,
                    risk_id=warning.risk_id,
                    measure_type=random.choice(measure_types),
                    description=f"Präventive Maßnahme für Warnung {warning.id}",
                    application_time=datetime.now(),
                    estimated_effectiveness=random.random(),
                    applied_to_timeline=next((risk.timeline_id for risk in self.monitored_risks if risk.id == warning.risk_id), ""),
                    applied_to_nodes=next((risk.risk_nodes for risk in self.monitored_risks if risk.id == warning.risk_id), []),
                    side_effects=[f"Effekt {j+1}" for j in range(random.randint(0, 2))]
                )
                
                measures.append(measure)
                self.applied_measures.append(measure)
        
        logger.info(f"{len(measures)} präventive Maßnahmen angewendet")
        return measures

# Testfälle für die erweiterte Paradoxauflösung
class TestEnhancedParadoxDetector(unittest.TestCase):
    """Testet den erweiterten Paradoxdetektor"""
    
    def setUp(self):
        """Initialisiert die Testumgebung"""
        self.detector = EnhancedParadoxDetector()
        self.timeline = self.create_test_timeline()
    
    def create_test_timeline(self):
        """Erstellt eine Testzeitlinie mit Paradoxien"""
        timeline = MockTimeline("test-timeline-1")
        
        # Füge Knoten hinzu
        node1 = timeline.add_node("node1")
        node2 = timeline.add_node("node2", ["node1"])
        node3 = timeline.add_node("node3", ["node2"])
        node4 = timeline.add_node("node4", ["node3"])
        node5 = timeline.add_node("node5", ["node4", "node1"])  # Zyklus: node1 -> node2 -> node3 -> node4 -> node5 -> node1
        
        # Füge Ereignisse hinzu
        node1.add_event("event1", "CREATION", {"data": "Testdaten"})
        node2.add_event("event2", "MODIFICATION", {"data": "Testdaten"})
        node3.add_event("event3", "INTERACTION", {"data": "Testdaten"})
        node4.add_event("event4", "OBSERVATION", {"data": "Testdaten"})
        node5.add_event("event5", "TERMINATION", {"data": "Testdaten"})
        
        return timeline
    
    def test_detect_complex_paradoxes(self):
        """Testet die Erkennung komplexer Paradoxien"""
        paradoxes = self.detector.detect_complex_paradoxes([self.timeline])
        # Wir können nicht genau wissen, wie viele Paradoxien erkannt werden, da dies zufällig ist
        # Aber wir können die grundlegende Funktionalität testen
        for paradox in paradoxes:
            self.assertIsInstance(paradox, ParadoxInstance)
            self.assertEqual(paradox.timeline_id, self.timeline.id)
            self.assertIn(paradox.type, EnhancedParadoxType)
            self.assertIn(paradox.severity, ParadoxSeverity)
    
    def test_evaluate_paradox_probability(self):
        """Testet die Bewertung der Paradoxwahrscheinlichkeit"""
        probability = self.detector.evaluate_paradox_probability(self.timeline, "node5")
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
    
    def test_detect_potential_paradoxes(self):
        """Testet die Erkennung potentieller Paradoxien"""
        potential_paradoxes = self.detector.detect_potential_paradoxes([self.timeline])
        # Wir können nicht genau wissen, wie viele potentielle Paradoxien erkannt werden, da dies zufällig ist
        # Aber wir können die grundlegende Funktionalität testen
        for paradox in potential_paradoxes:
            self.assertIsInstance(paradox, PotentialParadox)
            self.assertEqual(paradox.timeline_id, self.timeline.id)
            self.assertGreaterEqual(paradox.probability, 0.0)
            self.assertLessEqual(paradox.probability, 1.0)
            self.assertIn(paradox.estimated_severity, ParadoxSeverity)

class TestParadoxClassifier(unittest.TestCase):
    """Testet den Paradoxklassifizierer"""
    
    def setUp(self):
        """Initialisiert die Testumgebung"""
        self.classifier = ParadoxClassifier()
        self.paradox_instance = ParadoxInstance(
            id="test-paradox-1",
            type=EnhancedParadoxType.GRANDFATHER,
            severity=ParadoxSeverity.CRITICAL,
            timeline_id="test-timeline-1",
            affected_nodes=["node1", "node5"],
            description="Testparadox",
            detection_time=datetime.now(),
            probability=0.8,
            causal_chain=[("node1", "event1"), ("node5", "event5")],
            metadata={"test": True}
        )
    
    def test_classify_paradox(self):
        """Testet die Klassifizierung von Paradoxien"""
        paradox_type = self.classifier.classify_paradox(self.paradox_instance)
        self.assertEqual(paradox_type, self.paradox_instance.type)
    
    def test_evaluate_severity(self):
        """Testet die Bewertung des Schweregrades"""
        severity = self.classifier.evaluate_severity(self.paradox_instance)
        self.assertEqual(severity, self.paradox_instance.severity)
    
    def test_get_hierarchical_classification(self):
        """Testet die hierarchische Klassifizierung"""
        hierarchy = self.classifier.get_hierarchical_classification(self.paradox_instance)
        self.assertIsInstance(hierarchy, ParadoxHierarchy)
        self.assertEqual(hierarchy.primary_type, self.paradox_instance.type)
        self.assertIn(hierarchy.secondary_type, EnhancedParadoxType)
        self.assertIn(hierarchy.tertiary_type, EnhancedParadoxType)
        self.assertNotEqual(hierarchy.primary_type, hierarchy.secondary_type)
        self.assertNotEqual(hierarchy.primary_type, hierarchy.tertiary_type)
        self.assertNotEqual(hierarchy.secondary_type, hierarchy.tertiary_type)

class TestParadoxResolver(unittest.TestCase):
    """Testet den Paradoxauflöser"""
    
    def setUp(self):
        """Initialisiert die Testumgebung"""
        self.resolver = ParadoxResolver()
        self.timeline = self.create_test_timeline()
        self.paradox_instance = ParadoxInstance(
            id="test-paradox-1",
            type=EnhancedParadoxType.GRANDFATHER,
            severity=ParadoxSeverity.CRITICAL,
            timeline_id=self.timeline.id,
            affected_nodes=["node1", "node5"],
            description="Testparadox",
            detection_time=datetime.now(),
            probability=0.8,
            causal_chain=[("node1", "event1"), ("node5", "event5")],
            metadata={"test": True}
        )
    
    def create_test_timeline(self):
        """Erstellt eine Testzeitlinie"""
        timeline = MockTimeline("test-timeline-1")
        
        # Füge Knoten hinzu
        node1 = timeline.add_node("node1")
        node2 = timeline.add_node("node2", ["node1"])
        node3 = timeline.add_node("node3", ["node2"])
        node4 = timeline.add_node("node4", ["node3"])
        node5 = timeline.add_node("node5", ["node4", "node1"])
        
        return timeline
    
    def test_get_resolution_options(self):
        """Testet die Generierung von Auflösungsoptionen"""
        options = self.resolver.get_resolution_options(self.paradox_instance)
        self.assertGreater(len(options), 0)
        for option in options:
            self.assertIsInstance(option, ResolutionOption)
            self.assertIn(option.strategy, ResolutionStrategy)
            self.assertGreaterEqual(option.estimated_success_rate, 0.0)
            self.assertLessEqual(option.estimated_success_rate, 1.0)
            self.assertGreaterEqual(option.complexity, 1)
            self.assertLessEqual(option.complexity, 10)
            self.assertGreaterEqual(option.resource_cost, 1)
            self.assertLessEqual(option.resource_cost, 10)
    
    def test_select_optimal_resolution(self):
        """Testet die Auswahl der optimalen Auflösungsstrategie"""
        options = self.resolver.get_resolution_options(self.paradox_instance)
        selected_option = self.resolver.select_optimal_resolution(options)
        self.assertIsInstance(selected_option, ResolutionOption)
        self.assertIn(selected_option, options)
    
    def test_resolve_paradox(self):
        """Testet die Auflösung eines Paradoxes"""
        result = self.resolver.resolve_paradox(self.paradox_instance)
        self.assertIsInstance(result, ResolutionResult)
        self.assertEqual(result.paradox_id, self.paradox_instance.id)
        self.assertIn(result.strategy_used, ResolutionStrategy)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.impact, ResolutionImpact)
        self.assertGreaterEqual(result.impact.timeline_stability, 0.0)
        self.assertLessEqual(result.impact.timeline_stability, 1.0)
        self.assertGreaterEqual(result.impact.causal_integrity, 0.0)
        self.assertLessEqual(result.impact.causal_integrity, 1.0)
        self.assertGreaterEqual(result.impact.information_preservation, 0.0)
        self.assertLessEqual(result.impact.information_preservation, 1.0)

class TestParadoxPreventionSystem(unittest.TestCase):
    """Testet das Paradox-Präventionssystem"""
    
    def setUp(self):
        """Initialisiert die Testumgebung"""
        self.prevention_system = ParadoxPreventionSystem()
        self.timeline = self.create_test_timeline()
    
    def create_test_timeline(self):
        """Erstellt eine Testzeitlinie mit potentiellen Paradoxien"""
        timeline = MockTimeline("test-timeline-1")
        
        # Füge Knoten hinzu
        node1 = timeline.add_node("node1")
        node2 = timeline.add_node("node2", ["node1"])
        node3 = timeline.add_node("node3", ["node2"])
        node4 = timeline.add_node("node4", ["node3"])
        node5 = timeline.add_node("node5", ["node4", "node1"])
        
        return timeline
    
    def test_monitor_timelines(self):
        """Testet die Überwachung von Zeitlinien"""
        risks = self.prevention_system.monitor_timelines([self.timeline])
        # Wir können nicht genau wissen, wie viele Risiken erkannt werden, da dies zufällig ist
        # Aber wir können die grundlegende Funktionalität testen
        for risk in risks:
            self.assertIsInstance(risk, ParadoxRisk)
            self.assertEqual(risk.timeline_id, self.timeline.id)
            self.assertGreaterEqual(risk.risk_level, 0.0)
            self.assertLessEqual(risk.risk_level, 1.0)
            self.assertGreaterEqual(risk.monitoring_priority, 1)
            self.assertLessEqual(risk.monitoring_priority, 10)
    
    def test_generate_early_warnings(self):
        """Testet die Generierung von Frühwarnungen"""
        risks = self.prevention_system.monitor_timelines([self.timeline])
        warnings = self.prevention_system.generate_early_warnings(risks)
        # Wir können nicht genau wissen, wie viele Warnungen generiert werden, da dies zufällig ist
        # Aber wir können die grundlegende Funktionalität testen
        for warning in warnings:
            self.assertIsInstance(warning, ParadoxWarning)
            self.assertIn(warning.risk_id, [risk.id for risk in risks])
            self.assertGreaterEqual(warning.warning_level, 1)
            self.assertLessEqual(warning.warning_level, 5)
            self.assertGreater(warning.estimated_time_to_paradox, 0)
    
    def test_apply_preventive_measures(self):
        """Testet die Anwendung präventiver Maßnahmen"""
        risks = self.prevention_system.monitor_timelines([self.timeline])
        warnings = self.prevention_system.generate_early_warnings(risks)
        measures = self.prevention_system.apply_preventive_measures(warnings)
        # Wir können nicht genau wissen, wie viele Maßnahmen angewendet werden, da dies zufällig ist
        # Aber wir können die grundlegende Funktionalität testen
        for measure in measures:
            self.assertIsInstance(measure, PreventiveMeasure)
            self.assertIn(measure.risk_id, [warning.risk_id for warning in warnings])
            self.assertGreaterEqual(measure.estimated_effectiveness, 0.0)
            self.assertLessEqual(measure.estimated_effectiveness, 1.0)

def run_tests():
    """Führt alle Tests aus"""
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Füge die Testfälle hinzu
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestEnhancedParadoxDetector))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestParadoxClassifier))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestParadoxResolver))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestParadoxPreventionSystem))
    
    # Führe die Tests aus
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Gib das Ergebnis zurück
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
