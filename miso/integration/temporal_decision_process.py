#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Temporal Decision Process

Dieses Modul implementiert Entscheidungsprozesse in temporalen Kontexten
für die Integration zwischen Q-LOGIK und ECHO-PRIME.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
from enum import Enum, auto
from dataclasses import dataclass, field

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.integration.temporal_decision_process")

# Importiere verwandte Module
try:
    from miso.integration.ql_echo_bridge import QLEchoBridge, TemporalDecision, TemporalDecisionType
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    logger.warning("QL-ECHO-Bridge nicht verfügbar, Funktionalität eingeschränkt")

try:
    from miso.integration.bayesian_time_analyzer import BayesianTimeNodeAnalyzer, BayesianTimeNodeResult
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    logger.warning("BayesianTimeNodeAnalyzer nicht verfügbar, Funktionalität eingeschränkt")

try:
    from engines.echo_prime.timeline import Timeline, TimeNode, TemporalEvent, Trigger
    from engines.echo_prime.engine import get_echo_prime_engine
    ECHO_PRIME_AVAILABLE = True
except ImportError:
    ECHO_PRIME_AVAILABLE = False
    logger.warning("ECHO-PRIME nicht verfügbar, Funktionalität eingeschränkt")

# Importiere Q-LOGIK
try:
    from miso.qlogik.qlogik_core import QLogikCore
    QLOGIK_AVAILABLE = True
except ImportError:
    QLOGIK_AVAILABLE = False
    logger.warning("Q-LOGIK nicht verfügbar, Funktionalität eingeschränkt")

class DecisionSequenceType(Enum):
    """Typen von Entscheidungssequenzen"""
    LINEAR = auto()       # Lineare Sequenz von Entscheidungen
    BRANCHING = auto()    # Verzweigende Sequenz
    CONVERGING = auto()   # Konvergierende Sequenz
    MIXED = auto()        # Gemischte Sequenz
    CYCLICAL = auto()     # Zyklische Sequenz

@dataclass
class TemporalDecisionContext:
    """Kontext für temporale Entscheidungsprozesse"""
    timeline_id: str
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    decision_points: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    prior_beliefs: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionSequence:
    """Repräsentiert eine Sequenz von Entscheidungen"""
    id: str
    sequence_type: DecisionSequenceType
    decisions: List[TemporalDecision] = field(default_factory=list)
    probability: float = 1.0
    confidence: float = 1.0
    entropy: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class TemporalDecisionProcess:
    """
    Implementiert Entscheidungsprozesse in temporalen Kontexten
    für die Integration zwischen Q-LOGIK und ECHO-PRIME
    """
    
    def __init__(self, bridge: Optional[QLEchoBridge] = None):
        """
        Initialisiert den temporalen Entscheidungsprozess
        
        Args:
            bridge: QL-ECHO-Bridge-Instanz (optional)
        """
        # Initialisiere Bridge
        self.bridge = bridge
        if self.bridge is None and BRIDGE_AVAILABLE:
            # Importiere dynamisch, um Zirkelbezüge zu vermeiden
            from miso.integration.ql_echo_bridge import get_ql_echo_bridge
            self.bridge = get_ql_echo_bridge()
            logger.info("QL-ECHO-Bridge erfolgreich initialisiert")
            
        # Initialisiere Analyzer
        self.analyzer = None
        if ANALYZER_AVAILABLE:
            from miso.integration.bayesian_time_analyzer import get_bayesian_time_analyzer
            self.analyzer = get_bayesian_time_analyzer(self.bridge)
            logger.info("BayesianTimeNodeAnalyzer erfolgreich initialisiert")
            
        # Initialisiere ECHO-PRIME
        self.echo_prime = None
        if ECHO_PRIME_AVAILABLE:
            self.echo_prime = get_echo_prime_engine()
            logger.info("ECHO-PRIME Engine erfolgreich initialisiert")
            
        # Initialisiere Q-LOGIK
        self.q_logic = None
        if QLOGIK_AVAILABLE:
            self.q_logic = QLogikCore()
            logger.info("Q-LOGIK Core erfolgreich initialisiert")
            
        logger.info("TemporalDecisionProcess initialisiert")
        
    def identify_decision_points(self, timeline: Timeline) -> List[Dict[str, Any]]:
        """
        Identifiziert Entscheidungspunkte in einer Zeitlinie
        
        Args:
            timeline: Die zu analysierende Zeitlinie
            
        Returns:
            Liste von Entscheidungspunkten als Dictionaries
        """
        if not timeline:
            logger.error("Keine Zeitlinie angegeben")
            return []
            
        decision_points = []
        nodes = timeline.get_all_nodes_sorted()
        
        # Analysiere jeden Knoten nach Entscheidungscharakteristiken
        for node in nodes:
            # Ein Knoten ist ein Entscheidungspunkt, wenn er:
            # - mehrere ausgehende Trigger hat (Verzweigung)
            # - oder Ereignisse vom Typ "decision" enthält
            # - oder eine hohe Bayes'sche Unsicherheit in seinem Zeitpunkt hat
            
            is_decision_point = False
            decision_events = []
            
            # Prüfe auf mehrere ausgehende Trigger
            if len(node.outgoing_triggers) > 1:
                is_decision_point = True
                
            # Prüfe auf Entscheidungsereignisse
            for event_id, event in node.events.items():
                # Prüfe zuerst auf das event_type-Attribut
                if hasattr(event, 'event_type') and event.event_type in ["decision", "choice", "selection"]:
                    is_decision_point = True
                    decision_events.append({
                        "event_id": event_id,
                        "name": event.name,
                        "description": event.description if hasattr(event, 'description') else ""
                    })
                # Prüfe auf type-Feld im data-Dictionary
                elif hasattr(event, 'data') and isinstance(event.data, dict) and event.data.get('type') in ["decision", "choice", "selection"]:
                    is_decision_point = True
                    decision_events.append({
                        "event_id": event_id,
                        "name": event.name,
                        "description": event.description if hasattr(event, 'description') else ""
                    })
                    
            # Verwende analyzer für unsichere Zeitknoten, falls verfügbar
            if self.analyzer:
                analysis = self.analyzer.analyze_node(timeline, node.id)
                if analysis and analysis.uncertainty > 0.5:
                    is_decision_point = True
                    
            if is_decision_point:
                # Extrahiere Optionen aus den Ereignisdaten
                options = []
                for event_id, event in node.events.items():
                    if hasattr(event, 'data') and isinstance(event.data, dict):
                        # Extrahiere Optionen direkt aus dem data-Dictionary, falls vorhanden
                        if 'options' in event.data and isinstance(event.data['options'], list):
                            options = event.data['options']  # Direkte Übernahme der Optionsliste
                            break
                
                # Falls keine Optionen in den Ereignisdaten gefunden wurden, verwende Trigger
                if not options:
                    trigger_options = []
                    for trigger in node.outgoing_triggers:
                        trigger_options.append({
                            "target_node_id": trigger.target_node_id,
                            "target_event_id": trigger.target_event_id,
                            "probability": trigger.probability,
                            "description": f"Option über {trigger.trigger_type}"
                        })
                    options = trigger_options
                
                # Index des Knotens in der sortierten Liste ermitteln
                node_index = nodes.index(node)
                
                decision_point = {
                    "node_id": node.id,
                    "node_index": node_index,  # Wichtig für Tests
                    "timestamp": node.timestamp,
                    "events": decision_events,
                    "options": options,
                    "uncertainty": self.analyzer.analyze_node(timeline, node.id).uncertainty if self.analyzer else 0.5
                }
                
                decision_points.append(decision_point)
                
        return decision_points
            
    def calculate_decision_utility(self, decision_point: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Berechnet Nutzwerte für verschiedene Entscheidungsoptionen
        
        Args:
            decision_point: Entscheidungspunkt mit Optionen
            context: Zusätzlicher Kontext (optional)
            
        Returns:
            Dictionary mit Options-IDs und ihren Nutzwerten
        """
        utilities = {}
        
        # Hole Optionen aus dem Entscheidungspunkt
        options = decision_point.get("options", [])
        
        # Nutze Q-LOGIK für die Bewertung, falls verfügbar
        if self.q_logic and options:
            # Bereite Eingabe für Q-LOGIK vor
            q_input = {
                "context": context or {},
                "decision_point": decision_point,
                "options": options
            }
            
            # Verarbeite mit Q-LOGIK
            q_result = self.q_logic.process(q_input)
            
            # Extrahiere Utilities aus dem Ergebnis
            if isinstance(q_result, dict) and "option_utilities" in q_result:
                return q_result["option_utilities"]
            else:
                # Simuliere Utilities falls Q-LOGIK nicht vollständig ist
                for i, option in enumerate(options):
                    # Berechne simulierte Utility mit leicht zufälliger Komponente
                    option_id = option.get("target_node_id") or f"option_{i}"
                    base_utility = option.get("probability", 0.5)
                    
                    # Füge Kontext-Modifikatoren hinzu
                    context_factor = 1.0
                    if context:
                        # Priorität berücksichtigen
                        if "priorities" in context and isinstance(context["priorities"], dict):
                            for key, value in context["priorities"].items():
                                if key in option.get("description", "").lower():
                                    context_factor *= (1.0 + value * 0.2)
                    
                    # Leichter Zufallsfaktor
                    import random
                    random_factor = random.uniform(0.9, 1.1)
                    
                    utilities[option_id] = base_utility * context_factor * random_factor
        else:
            # Erstelle einfache Utilities basierend auf Wahrscheinlichkeiten
            for i, option in enumerate(options):
                option_id = option.get("target_node_id") or f"option_{i}"
                utilities[option_id] = option.get("probability", 0.5)
                
        # Normalisiere Utilities
        total = sum(utilities.values())
        if total > 0:
            for option_id in utilities:
                utilities[option_id] /= total
                
        return utilities
    
    def identify_decision_sequences(self, context: TemporalDecisionContext) -> List[DecisionSequence]:
        """
        Identifiziert Entscheidungssequenzen in einem temporalen Kontext
        
        Args:
            context: Temporaler Entscheidungskontext
            
        Returns:
            Liste von Entscheidungssequenzen
        """
        if not self.echo_prime:
            logger.error("ECHO-PRIME ist nicht verfügbar")
            return []
            
        # Hole Zeitlinie
        timeline = self.echo_prime.get_timeline(context.timeline_id)
        if timeline is None:
            logger.error(f"Zeitlinie {context.timeline_id} nicht gefunden")
            return []
            
        # Hole Zeitknoten im Bereich
        nodes = []
        if context.start_time and context.end_time:
            # Hole alle Knoten und filtere nach Zeitbereich
            all_nodes = timeline.get_all_nodes_sorted()
            nodes = [node for node in all_nodes 
                    if context.start_time <= node.timestamp <= context.end_time]
        elif context.decision_points:
            # Hole spezifizierte Entscheidungspunkte
            nodes = [timeline.get_node(node_id) for node_id in context.decision_points
                    if timeline.get_node(node_id) is not None]
        else:
            # Hole alle Knoten
            nodes = timeline.get_all_nodes_sorted()
            
        if not nodes:
            logger.warning("Keine Zeitknoten im angegebenen Bereich gefunden")
            return []
            
        # Analysiere Knoten mit dem Analyzer, falls verfügbar
        node_analyses = {}
        if self.analyzer:
            for node in nodes:
                analysis = self.analyzer.analyze_node(timeline, node.id)
                node_analyses[node.id] = analysis
                
        # Suche nach zusammenhängenden Entscheidungssequenzen
        sequences = []
        
        # Identifiziere Sequenzstartpunkte (Knoten mit mehreren ausgehenden Triggers)
        for i, node in enumerate(nodes):
            if len(node.outgoing_triggers) > 1:
                # Potentieller Sequenzstartpunkt
                sequence_id = f"seq_{timeline.id}_{node.id}_{i}"
                
                # Klassifiziere Sequenztyp
                sequence_type = self._classify_sequence_type(node, timeline)
                
                # Erstelle Sequenz
                sequence = DecisionSequence(
                    id=sequence_id,
                    sequence_type=sequence_type,
                    decisions=[],
                    probability=1.0,
                    confidence=node_analyses.get(node.id, BayesianTimeNodeResult(node_id=node.id, timestamp=node.timestamp)).confidence if node_analyses else 0.8
                )
                
                # Verfolge Sequenz
                self._trace_decision_sequence(sequence, node, timeline, nodes, node_analyses, context)
                
                if sequence.decisions:
                    sequences.append(sequence)
                    
        # Füge isolierte Entscheidungspunkte hinzu
        for node in nodes:
            if node.id not in [d.id for seq in sequences for d in seq.decisions]:
                if len(node.outgoing_triggers) > 0:
                    sequence_id = f"iso_{timeline.id}_{node.id}"
                    
                    # Erstelle Entscheidung für diesen Knoten
                    decision = self._create_decision_from_node(node, timeline, node_analyses.get(node.id) if node_analyses else None)
                    
                    # Erstelle Sequenz mit nur einer Entscheidung
                    sequence = DecisionSequence(
                        id=sequence_id,
                        sequence_type=DecisionSequenceType.LINEAR,
                        decisions=[decision],
                        probability=decision.probability,
                        confidence=decision.confidence,
                        entropy=0.0 if len(node.outgoing_triggers) <= 1 else 0.5
                    )
                    
                    sequences.append(sequence)
                    
        return sequences
        
    def evaluate_decision_sequence(self, sequence: DecisionSequence, context: TemporalDecisionContext) -> Dict[str, Any]:
        """
        Bewertet eine Entscheidungssequenz im Kontext
        
        Args:
            sequence: Zu bewertende Entscheidungssequenz
            context: Bewertungskontext
            
        Returns:
            Bewertungsergebnis
        """
        if not sequence.decisions:
            logger.warning("Leere Entscheidungssequenz zur Bewertung")
            return {
                "quality": 0.0,
                "confidence": 0.0,
                "stability": 0.0,
                "message": "Leere Entscheidungssequenz"
            }
            
        # Q-LOGIK für Probabilistische Inferenz verwenden
        if self.q_logic:
            # Bereite Input für Q-LOGIK vor
            input_data = {
                "sequence_type": sequence.sequence_type.name,
                "decisions": [
                    {
                        "id": d.id,
                        "type": d.decision_type.name,
                        "probability": d.probability,
                        "confidence": d.confidence,
                        "alternatives": len(d.alternatives)
                    } for d in sequence.decisions
                ],
                "constraints": context.constraints,
                "prior_beliefs": context.prior_beliefs
            }
            
            # Prozessiere mit Q-LOGIK
            self.q_logic.process(input_data)
            
            # Simuliere Q-LOGIK-Bewertung für den Prototyp
            # In der tatsächlichen Implementation würde dies eine echte Bayes'sche Analyse sein
            quality = 0.7  # Basisqualität
            
            # Sequenztyp-Einfluss
            if sequence.sequence_type == DecisionSequenceType.LINEAR:
                quality += 0.1  # Lineare Sequenzen sind einfacher zu bewerten
            elif sequence.sequence_type == DecisionSequenceType.CYCLICAL:
                quality -= 0.2  # Zyklische Sequenzen sind besonders problematisch
                
            # Berechne durchschnittliche Werte aus der Sequenz
            avg_probability = sum(d.probability for d in sequence.decisions) / len(sequence.decisions)
            avg_confidence = sum(d.confidence for d in sequence.decisions) / len(sequence.decisions)
            
            # Gewichtete Bewertung
            quality = 0.4 * quality + 0.3 * avg_probability + 0.3 * avg_confidence
            
            # Begrenze auf [0,1]
            quality = max(0.0, min(1.0, quality))
            
            # Stabilität basierend auf Sequenztyp und Wahrscheinlichkeiten
            stability = 0.5  # Basisstabilität
            
            if sequence.sequence_type == DecisionSequenceType.LINEAR:
                stability += 0.2
            elif sequence.sequence_type == DecisionSequenceType.BRANCHING:
                stability -= 0.1
            elif sequence.sequence_type == DecisionSequenceType.CYCLICAL:
                stability -= 0.3
                
            # Hohe Wahrscheinlichkeiten und Konfidenz erhöhen Stabilität
            stability += 0.3 * avg_probability + 0.2 * avg_confidence
            
            # Begrenze auf [0,1]
            stability = max(0.0, min(1.0, stability))
            
            # Klassifiziere Qualität
            quality_category = "Hoch"
            if quality < 0.4:
                quality_category = "Niedrig"
            elif quality < 0.7:
                quality_category = "Mittel"
                
            # Klassifiziere Stabilität
            stability_category = "Stabil"
            if stability < 0.4:
                stability_category = "Instabil"
            elif stability < 0.7:
                stability_category = "Moderat"
                
            return {
                "quality": quality,
                "quality_category": quality_category,
                "confidence": avg_confidence,
                "stability": stability,
                "stability_category": stability_category,
                "message": f"Sequenz vom Typ {sequence.sequence_type.name} mit {len(sequence.decisions)} Entscheidungen erfolgreich bewertet"
            }
        else:
            # Einfache fallback-Bewertung ohne Q-LOGIK
            avg_probability = sum(d.probability for d in sequence.decisions) / len(sequence.decisions)
            avg_confidence = sum(d.confidence for d in sequence.decisions) / len(sequence.decisions)
            
            return {
                "quality": avg_probability * 0.8,
                "confidence": avg_confidence,
                "stability": avg_confidence * 0.7,
                "message": "Einfache Bewertung ohne Q-LOGIK"
            }
            
    def optimize_decision_sequence(self, sequence: DecisionSequence, context: TemporalDecisionContext) -> DecisionSequence:
        """
        Optimiert eine Entscheidungssequenz basierend auf dem Kontext
        
        Args:
            sequence: Zu optimierende Entscheidungssequenz
            context: Optimierungskontext
            
        Returns:
            Optimierte Entscheidungssequenz
        """
        if not sequence.decisions:
            logger.warning("Leere Entscheidungssequenz zur Optimierung")
            return sequence
            
        # Erstelle Kopie der Sequenz für Optimierung
        optimized = DecisionSequence(
            id=sequence.id,
            sequence_type=sequence.sequence_type,
            decisions=[d for d in sequence.decisions],  # Flache Kopie der Entscheidungen
            probability=sequence.probability,
            confidence=sequence.confidence,
            entropy=sequence.entropy,
            metadata=sequence.metadata.copy()
        )
        
        # Prüfe auf vorhandene Optimierungsziele im Kontext
        optimization_targets = context.constraints.get("optimization_targets", [])
        
        # Optimiere jede Entscheidung in der Sequenz
        for i, decision in enumerate(optimized.decisions):
            # Kopiere Entscheidung für Optimierung
            optimized_decision = TemporalDecision(
                id=decision.id,
                description=decision.description,
                decision_type=decision.decision_type,
                probability=decision.probability,
                confidence=decision.confidence,
                alternatives=decision.alternatives.copy() if decision.alternatives else [],
                metadata=decision.metadata.copy()
            )
            
            # Wende Optimierungsziele an
            for target in optimization_targets:
                if target == "maximize_probability":
                    # Identifiziere Alternative mit höchster Wahrscheinlichkeit
                    if optimized_decision.alternatives:
                        best_alt = max(optimized_decision.alternatives, key=lambda a: a.get("probability", 0))
                        optimized_decision.probability = best_alt.get("probability", optimized_decision.probability)
                        optimized_decision.description = best_alt.get("description", optimized_decision.description)
                        
                elif target == "maximize_confidence":
                    # Identifiziere Alternative mit höchster Konfidenz
                    if optimized_decision.alternatives:
                        best_alt = max(optimized_decision.alternatives, key=lambda a: a.get("confidence", 0))
                        optimized_decision.confidence = best_alt.get("confidence", optimized_decision.confidence)
                        optimized_decision.probability = best_alt.get("probability", optimized_decision.probability)
                        optimized_decision.description = best_alt.get("description", optimized_decision.description)
                        
                elif target == "minimize_entropy":
                    # Wähle Alternativen mit niedrigerer Entropie
                    optimized_decision.alternatives = sorted(
                        optimized_decision.alternatives,
                        key=lambda a: a.get("entropy", 1.0) if "entropy" in a else 1.0
                    )[:2]  # Behalte nur die besten 2
                    
                elif target == "balance":
                    # Balanciere Wahrscheinlichkeit und Konfidenz
                    if optimized_decision.alternatives:
                        best_alt = max(
                            optimized_decision.alternatives,
                            key=lambda a: 0.6 * a.get("probability", 0) + 0.4 * a.get("confidence", 0)
                        )
                        optimized_decision.probability = best_alt.get("probability", optimized_decision.probability)
                        optimized_decision.confidence = best_alt.get("confidence", optimized_decision.confidence)
                        optimized_decision.description = best_alt.get("description", optimized_decision.description)
            
            # Update die Entscheidung in der Sequenz
            optimized.decisions[i] = optimized_decision
            
        # Aktualisiere Gesamtmetriken der Sequenz
        if optimized.decisions:
            optimized.probability = np.mean([d.probability for d in optimized.decisions])
            optimized.confidence = np.mean([d.confidence for d in optimized.decisions])
            
            # Füge Optimierungsinfo zu Metadata hinzu
            optimized.metadata["optimization"] = {
                "targets": optimization_targets,
                "timestamp": datetime.datetime.now().isoformat(),
                "improvement": optimized.probability / sequence.probability if sequence.probability > 0 else 1.0
            }
            
        return optimized
        
    def _classify_sequence_type(self, start_node: TimeNode, timeline: Timeline) -> DecisionSequenceType:
        """
        Klassifiziert den Typ einer Entscheidungssequenz
        
        Args:
            start_node: Startknoten der Sequenz
            timeline: Zeitlinie
            
        Returns:
            Sequenztyp
        """
        # Prüfe ausgehende Trigger
        outgoing_count = len(start_node.outgoing_triggers)
        
        if outgoing_count <= 1:
            return DecisionSequenceType.LINEAR
            
        # Verfolge ausgehende Pfade
        target_node_ids = set()
        for trigger_id, trigger in start_node.outgoing_triggers.items():
            # Suche Zielknoten für diesen Trigger
            for node_id, node in timeline.nodes.items():
                if trigger.target_event_id in [e_id for e_id in node.events.keys()]:
                    target_node_ids.add(node_id)
                    break
        
        # Prüfe auf Konvergenz (mehrere Pfade führen zum gleichen Ziel)
        if len(target_node_ids) < outgoing_count:
            return DecisionSequenceType.CONVERGING
            
        # Prüfe auf Zyklen
        for target_id in target_node_ids:
            target_node = timeline.get_node(target_id)
            if target_node:
                for trigger_id, trigger in target_node.outgoing_triggers.items():
                    # Pfad führt zurück zum Start
                    for node_id, node in timeline.nodes.items():
                        if node_id == start_node.id and trigger.target_event_id in [e_id for e_id in node.events.keys()]:
                            return DecisionSequenceType.CYCLICAL
        
        # Standardfall: Verzweigend
        return DecisionSequenceType.BRANCHING
        
    def _trace_decision_sequence(self, sequence: DecisionSequence, start_node: TimeNode, 
                                 timeline: Timeline, nodes: List[TimeNode], 
                                 node_analyses: Dict[str, Any], context: TemporalDecisionContext):
        """
        Verfolgt eine Entscheidungssequenz ausgehend von einem Startknoten
        
        Args:
            sequence: Zu füllende Entscheidungssequenz
            start_node: Startknoten
            timeline: Zeitlinie
            nodes: Liste der Knoten in Betrachtung
            node_analyses: Vorberechnete Knotenanalysen
            context: Entscheidungskontext
        """
        # Maximale Tiefe für Sequenzverfolgung
        max_depth = context.constraints.get("max_sequence_depth", 5)
        
        # Erstelle Entscheidung für Startknoten
        start_decision = self._create_decision_from_node(start_node, timeline, node_analyses.get(start_node.id))
        sequence.decisions.append(start_decision)
        
        # Verfolge ausgehende Pfade, beginnend mit dem wahrscheinlichsten
        if sequence.sequence_type != DecisionSequenceType.CYCLICAL:
            visited_nodes = {start_node.id}
            self._trace_decision_path(sequence, start_node, timeline, nodes, node_analyses, 
                                     visited_nodes, 1, max_depth)
                                     
        # Berechne Sequenz-Entropie basierend auf Verzweigungen und Wahrscheinlichkeiten
        branch_points = sum(1 for d in sequence.decisions if d.decision_type == TemporalDecisionType.PROBABILISTIC)
        prob_variation = np.std([d.probability for d in sequence.decisions]) if sequence.decisions else 0
        
        # Höhere Entropie bei mehr Verzweigungspunkten und geringerer Wahrscheinlichkeitsvariation
        sequence.entropy = min(1.0, 0.2 * branch_points + 0.5 * (1.0 - prob_variation))
        
    def _trace_decision_path(self, sequence: DecisionSequence, current_node: TimeNode, 
                            timeline: Timeline, nodes: List[TimeNode], 
                            node_analyses: Dict[str, Any], visited_nodes: Set[str], 
                            current_depth: int, max_depth: int):
        """
        Verfolgt einen Entscheidungspfad rekursiv
        
        Args:
            sequence: Zu füllende Entscheidungssequenz
            current_node: Aktueller Knoten
            timeline: Zeitlinie
            nodes: Liste der Knoten in Betrachtung
            node_analyses: Vorberechnete Knotenanalysen
            visited_nodes: Bereits besuchte Knoten
            current_depth: Aktuelle Rekursionstiefe
            max_depth: Maximale Rekursionstiefe
        """
        if current_depth >= max_depth:
            return
            
        # Sortiere ausgehende Trigger nach Wahrscheinlichkeit
        sorted_triggers = sorted(
            current_node.outgoing_triggers.items(),
            key=lambda item: item[1].probability,
            reverse=True
        )
        
        for trigger_id, trigger in sorted_triggers:
            # Finde Zielknoten für diesen Trigger
            target_node = None
            for node in nodes:
                if trigger.target_event_id in [e_id for e_id in node.events.keys()]:
                    target_node = node
                    break
                    
            if target_node and target_node.id not in visited_nodes:
                # Erstelle Entscheidung für diesen Knoten
                decision = self._create_decision_from_node(target_node, timeline, node_analyses.get(target_node.id))
                sequence.decisions.append(decision)
                
                # Markiere als besucht
                visited_nodes.add(target_node.id)
                
                # Rekursiv weitergehen
                self._trace_decision_path(sequence, target_node, timeline, nodes, node_analyses,
                                        visited_nodes, current_depth + 1, max_depth)
                
                # Bei Verzweigungssequenzen nur den ersten Pfad verfolgen
                if sequence.sequence_type == DecisionSequenceType.BRANCHING:
                    break
        
    def _create_decision_from_node(self, node: TimeNode, timeline: Timeline, analysis: Optional[Any] = None) -> TemporalDecision:
        """
        Erstellt eine temporale Entscheidung aus einem Zeitknoten
        
        Args:
            node: Zeitknoten
            timeline: Zeitlinie
            analysis: Vorberechnete Analyse (optional)
            
        Returns:
            Temporale Entscheidung
        """
        # Bestimme Entscheidungstyp basierend auf ausgehenden Triggern
        decision_type = TemporalDecisionType.DETERMINISTIC
        if len(node.outgoing_triggers) > 1:
            decision_type = TemporalDecisionType.PROBABILISTIC
            
        # Wenn Analyse verfügbar ist, Paradoxwahrscheinlichkeit prüfen
        if analysis and hasattr(analysis, 'paradox_likelihood') and analysis.paradox_likelihood > 0.5:
            decision_type = TemporalDecisionType.PARADOXICAL
        elif analysis and hasattr(analysis, 'entropy') and analysis.entropy > 0.7:
            decision_type = TemporalDecisionType.UNCERTAIN
            
        # Erstelle Beschreibung
        events_desc = ", ".join([e.name for e in node.events.values()][:3])
        description = f"Zeitknoten {node.id[:8]} mit Ereignissen: {events_desc}"
        
        # Erstelle Alternativen aus ausgehenden Triggern
        alternatives = []
        for trigger_id, trigger in node.outgoing_triggers.items():
            # Finde Zielereignis
            target_event = None
            for event_id, event in timeline.events.items():
                if event_id == trigger.target_event_id:
                    target_event = event
                    break
                    
            if target_event:
                alt = {
                    "trigger_id": trigger_id,
                    "target_event_id": trigger.target_event_id,
                    "probability": trigger.probability,
                    "confidence": 0.8,  # Standardwert
                    "description": f"{target_event.name}"
                }
                alternatives.append(alt)
                
        # Berechne Entscheidungswahrscheinlichkeit und -konfidenz
        probability = 1.0
        confidence = 1.0
        
        if alternatives:
            # Höchste Wahrscheinlichkeit der Alternativen
            probability = max(alt["probability"] for alt in alternatives)
            
            # Bei mehreren Alternativen mit ähnlichen Wahrscheinlichkeiten, reduziere Konfidenz
            if len(alternatives) > 1:
                prob_stdev = np.std([alt["probability"] for alt in alternatives])
                if prob_stdev < 0.2:  # Geringe Variation = Unsicherheit
                    confidence = 0.6
                    
        # Verwende Analyse, falls verfügbar
        if analysis:
            if hasattr(analysis, 'confidence'):
                confidence = analysis.confidence
                
        return TemporalDecision(
            id=node.id,
            description=description,
            decision_type=decision_type,
            probability=probability,
            confidence=confidence,
            alternatives=alternatives,
            metadata={
                "timestamp": node.timestamp.isoformat(),
                "event_count": len(node.events),
                "outgoing_count": len(node.outgoing_triggers),
                "incoming_count": len(node.incoming_triggers)
            }
        )

# Globale Instanz
_TEMPORAL_DECISION_PROCESS = None

def get_temporal_decision_process(bridge: Optional[QLEchoBridge] = None):
    """
    Gibt die globale TemporalDecisionProcess-Instanz zurück
    
    Args:
        bridge: Optional QLEchoBridge instance
        
    Returns:
        TemporalDecisionProcess-Instanz
    """
    global _TEMPORAL_DECISION_PROCESS
    
    if _TEMPORAL_DECISION_PROCESS is None:
        _TEMPORAL_DECISION_PROCESS = TemporalDecisionProcess(bridge)
        
    return _TEMPORAL_DECISION_PROCESS
