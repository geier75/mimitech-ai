#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK ECHO-PRIME Integration

Integrationsschicht zwischen Q-LOGIK und ECHO-PRIME.
Ermöglicht die Anwendung von Q-LOGIK Entscheidungsfindung auf Zeitlinien
und die Nutzung von Quanteneffekten in ECHO-PRIME.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple, Set
import numpy as np
from datetime import datetime

# Importiere Q-LOGIK Komponenten
from miso.logic.qlogik_engine import (
    BayesianDecisionCore, 
    FuzzyLogicUnit, 
    SymbolMap, 
    ConflictResolver, 
    simple_emotion_weight,
    simple_priority_mapping,
    qlogik_decision,
    advanced_qlogik_decision
)

# Importiere ECHO-PRIME Komponenten
from miso.timeline.echo_prime import Timeline, TimeNode, TriggerLevel, TimelineType
from miso.timeline.echo_prime_controller import EchoPrimeController
from miso.timeline.qtm_modulator import QTM_Modulator, QuantumTimeEffect, QuantumState

# Logger einrichten
logger = logging.getLogger("MISO.Logic.Q-LOGIK.ECHOPrimeIntegration")

class TimelineDecisionContext:
    """
    Kontext für Entscheidungen auf Zeitlinien
    
    Enthält alle relevanten Informationen für eine Entscheidung
    im Kontext einer Zeitlinie.
    """
    
    def __init__(self, 
                 timeline: Timeline, 
                 current_node: TimeNode = None, 
                 quantum_state: QuantumState = None,
                 decision_factors: Dict[str, float] = None):
        """
        Initialisiert den Zeitlinien-Entscheidungskontext
        
        Args:
            timeline: Die betroffene Zeitlinie
            current_node: Der aktuelle Zeitknoten
            quantum_state: Der Quantenzustand der Zeitlinie
            decision_factors: Faktoren für die Entscheidungsfindung
        """
        self.timeline = timeline
        self.current_node = current_node
        self.quantum_state = quantum_state
        self.decision_factors = decision_factors or {}
        
        # Standardfaktoren, falls nicht angegeben
        if not self.decision_factors:
            self.decision_factors = {
                "risk": 0.5,
                "benefit": 0.5,
                "urgency": 0.5,
                "confidence": 0.5,
                "impact": 0.5
            }
    
    def to_qlogik_context(self) -> Dict[str, Any]:
        """
        Konvertiert den Zeitlinienkontext in einen Q-LOGIK-Kontext
        
        Returns:
            Q-LOGIK-Kontext als Dictionary
        """
        context = {
            "timeline_id": self.timeline.id,
            "timeline_name": self.timeline.name,
            "timeline_type": self.timeline.type.value,
            "decision_factors": self.decision_factors,
            "metadata": {}
        }
        
        # Füge Knotenkontext hinzu, falls vorhanden
        if self.current_node:
            context["node_id"] = self.current_node.id
            context["node_description"] = self.current_node.description
            context["node_probability"] = self.current_node.probability
            context["node_trigger_level"] = self.current_node.trigger_level.value
            context["node_metadata"] = self.current_node.metadata
            
            # Füge Eltern- und Kindknoteninfos hinzu
            context["parent_node_id"] = self.current_node.parent_node_id
            context["child_node_count"] = len(self.current_node.child_node_ids)
        
        # Füge Quantenzustand hinzu, falls vorhanden
        if self.quantum_state:
            context["quantum_state"] = self.quantum_state.name
            
            # Passe Entscheidungsfaktoren basierend auf Quantenzustand an
            if self.quantum_state == QuantumState.SUPERPOSITION:
                context["decision_factors"]["uncertainty"] = 0.8
                context["decision_factors"]["branching_factor"] = 0.9
            elif self.quantum_state == QuantumState.ENTANGLED:
                context["decision_factors"]["correlation"] = 0.9
                context["decision_factors"]["synchronization"] = 0.8
            elif self.quantum_state == QuantumState.COLLAPSED:
                context["decision_factors"]["certainty"] = 0.9
                context["decision_factors"]["stability"] = 0.8
        
        return context


class QLOGIKECHOPrimeIntegration:
    """
    Q-LOGIK ECHO-PRIME Integration
    
    Integrationsschicht zwischen Q-LOGIK und ECHO-PRIME.
    Ermöglicht die Anwendung von Q-LOGIK Entscheidungsfindung auf Zeitlinien
    und die Nutzung von Quanteneffekten in ECHO-PRIME.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert die Integrationsschicht
        
        Args:
            config: Konfigurationsobjekt für die Integration
        """
        self.config = config or {}
        
        # Initialisiere Q-LOGIK Komponenten
        self.bayesian = BayesianDecisionCore()
        self.fuzzylogic = FuzzyLogicUnit()
        self.symbolmap = SymbolMap()
        self.conflict_resolver = ConflictResolver()
        
        # Verwende die vereinfachten Funktionen statt der Klassen
        self.emotion_weight_func = simple_emotion_weight
        self.priority_mapping_func = simple_priority_mapping
        
        # Initialisiere ECHO-PRIME Controller
        self.echo_prime = EchoPrimeController()
        
        # Cache für Entscheidungen
        self.decision_cache = {}
        
        logger.info("Q-LOGIK ECHO-PRIME Integration initialisiert")
    
    def evaluate_timeline(self, timeline_id: str) -> Dict[str, Any]:
        """
        Evaluiert eine Zeitlinie mit Q-LOGIK
        
        Args:
            timeline_id: ID der zu evaluierenden Zeitlinie
            
        Returns:
            Evaluierungsergebnisse
        """
        timeline = self.echo_prime.get_timeline(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} existiert nicht")
            return {"error": f"Zeitlinie {timeline_id} existiert nicht"}
        
        # Erstelle Zeitlinienkontext
        quantum_state = None
        if hasattr(self.echo_prime, "qtm_modulator") and hasattr(self.echo_prime.qtm_modulator, "quantum_states"):
            quantum_state = self.echo_prime.qtm_modulator.quantum_states.get(timeline_id)
        
        # Finde den aktuellen Knoten (neuester Knoten)
        current_node = None
        if timeline.nodes:
            current_node = max(timeline.nodes.values(), key=lambda n: n.timestamp)
        
        # Erstelle Kontext
        context = TimelineDecisionContext(
            timeline=timeline,
            current_node=current_node,
            quantum_state=quantum_state
        )
        
        # Konvertiere in Q-LOGIK-Kontext
        qlogik_context = context.to_qlogik_context()
        
        # Wende Q-LOGIK-Entscheidungsfindung an
        decision = advanced_qlogik_decision(qlogik_context)
        
        # Speichere Entscheidung im Cache
        cache_key = f"{timeline_id}_{int(time.time())}"
        self.decision_cache[cache_key] = decision
        
        return decision
    
    def recommend_timeline_action(self, timeline_id: str) -> Dict[str, Any]:
        """
        Empfiehlt eine Aktion für eine Zeitlinie basierend auf Q-LOGIK
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            Empfohlene Aktion
        """
        # Evaluiere Zeitlinie
        evaluation = self.evaluate_timeline(timeline_id)
        if "error" in evaluation:
            return evaluation
        
        # Erstelle Aktionsempfehlung
        recommendation = {
            "timeline_id": timeline_id,
            "timestamp": datetime.now().isoformat(),
            "recommendation_type": "timeline_action",
            "actions": []
        }
        
        # Bestimme empfohlene Aktionen basierend auf Evaluation
        confidence = evaluation.get("confidence", 0.5)
        risk = evaluation.get("risk_assessment", {}).get("overall_risk", 0.5)
        decision = evaluation.get("decision", "WARNUNG")
        
        if decision == "JA":
            if confidence > 0.8:
                recommendation["actions"].append({
                    "action": "expand_timeline",
                    "priority": "high",
                    "description": "Zeitlinie erweitern mit neuen Knoten"
                })
            else:
                recommendation["actions"].append({
                    "action": "analyze_timeline",
                    "priority": "medium",
                    "description": "Weitere Analyse der Zeitlinie durchführen"
                })
        elif decision == "NEIN":
            if risk > 0.7:
                recommendation["actions"].append({
                    "action": "prune_timeline",
                    "priority": "high",
                    "description": "Riskante Zweige der Zeitlinie entfernen"
                })
            else:
                recommendation["actions"].append({
                    "action": "freeze_timeline",
                    "priority": "medium",
                    "description": "Zeitlinie einfrieren und keine weiteren Änderungen vornehmen"
                })
        else:  # WARNUNG
            recommendation["actions"].append({
                "action": "apply_quantum_effect",
                "priority": "medium",
                "description": "Quanteneffekt anwenden, um Unsicherheit zu reduzieren",
                "suggested_effect": "superposition"
            })
            recommendation["actions"].append({
                "action": "consult_integrity_guard",
                "priority": "high",
                "description": "Temporale Integritätsprüfung durchführen"
            })
        
        return recommendation
    
    def apply_quantum_decision(self, 
                              timeline_id: str, 
                              effect_type: str = "auto") -> Dict[str, Any]:
        """
        Wendet einen Quanteneffekt basierend auf Q-LOGIK-Entscheidung an
        
        Args:
            timeline_id: ID der Zeitlinie
            effect_type: Typ des anzuwendenden Effekts ("superposition", "entanglement", "collapse", "auto")
            
        Returns:
            Angewendeter Quanteneffekt
        """
        timeline = self.echo_prime.get_timeline(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} existiert nicht")
            return {"error": f"Zeitlinie {timeline_id} existiert nicht"}
        
        # Evaluiere Zeitlinie, wenn effect_type "auto" ist
        if effect_type == "auto":
            evaluation = self.evaluate_timeline(timeline_id)
            if "error" in evaluation:
                return evaluation
            
            # Bestimme den besten Effekt basierend auf Evaluation
            confidence = evaluation.get("confidence", 0.5)
            risk = evaluation.get("risk_assessment", {}).get("overall_risk", 0.5)
            uncertainty = evaluation.get("uncertainty", 0.5)
            
            if uncertainty > 0.7:
                effect_type = "superposition"
            elif risk > 0.7:
                effect_type = "collapse"
            else:
                # Finde eine andere Zeitlinie für Entanglement
                timelines = self.echo_prime.get_all_timelines()
                compatible_timelines = []
                
                for other_id, other_timeline in timelines.items():
                    if other_id != timeline_id and other_timeline.type == timeline.type:
                        compatible_timelines.append(other_timeline)
                
                if compatible_timelines:
                    effect_type = "entanglement"
                    # Wähle die kompatible Zeitlinie mit der höchsten Wahrscheinlichkeit
                    target_timeline = max(compatible_timelines, key=lambda t: t.probability)
                else:
                    effect_type = "superposition"  # Fallback, wenn keine kompatible Zeitlinie gefunden wurde
        
        # Wende den ausgewählten Quanteneffekt an
        result = {"effect_type": effect_type, "timeline_id": timeline_id}
        
        if effect_type == "superposition":
            # Erstelle Superpositionseffekt
            # Verwende die korrekte Methode des QTM_Modulators
            effect = self.echo_prime.qtm_modulator.apply_superposition(
                timelines=[timeline],
                duration=3600.0  # 1 Stunde
            )
            result["effect_id"] = effect.id
            result["branches"] = branches
            logger.info(f"Superpositionseffekt auf Zeitlinie {timeline_id} mit {branches} Branches angewendet")
            
        elif effect_type == "entanglement":
            # Finde eine kompatible Zeitlinie für Entanglement
            timelines = self.echo_prime.get_all_timelines()
            compatible_timelines = []
            
            for other_id, other_timeline in timelines.items():
                if other_id != timeline_id and other_timeline.type == timeline.type:
                    compatible_timelines.append(other_timeline)
            
            if compatible_timelines:
                # Wähle die kompatible Zeitlinie mit der höchsten Wahrscheinlichkeit
                target_timeline = max(compatible_timelines, key=lambda t: t.probability)
                
                # Erstelle Entanglement-Effekt
                # Verwende die korrekte Methode des QTM_Modulators
                # Erstelle ein Paar von Knoten für die Verschränkung
                node_pairs = []
                if timeline.nodes and target_timeline.nodes:
                    # Wähle einen Knoten aus jeder Zeitlinie für die Verschränkung
                    node_pairs = [(list(timeline.nodes.keys())[0], list(target_timeline.nodes.keys())[0])]
                
                effect = self.echo_prime.qtm_modulator.entangle_timelines(
                    timeline1=timeline,
                    timeline2=target_timeline,
                    node_pairs=node_pairs,
                    duration=7200.0  # 2 Stunden
                )
                result["effect_id"] = effect.id
                result["target_timeline_id"] = target_timeline.id
                logger.info(f"Entanglement-Effekt zwischen Zeitlinien {timeline_id} und {target_timeline.id} angewendet")
            else:
                logger.warning(f"Keine kompatible Zeitlinie für Entanglement mit {timeline_id} gefunden")
                result["error"] = "Keine kompatible Zeitlinie für Entanglement gefunden"
                
        elif effect_type == "collapse":
            # Erstelle Collapse-Effekt
            # Verwende die korrekte Methode des QTM_Modulators
            effect = self.echo_prime.qtm_modulator.collapse_timeline(
                timeline=timeline,
                target_nodes=None  # Alle Knoten kollabieren
            )
            result["effect_id"] = effect.id
            logger.info(f"Collapse-Effekt auf Zeitlinie {timeline_id} angewendet")
            
        else:
            logger.warning(f"Unbekannter Effekttyp: {effect_type}")
            result["error"] = f"Unbekannter Effekttyp: {effect_type}"
        
        return result
    
    def resolve_timeline_conflict(self, 
                                 timeline_ids: List[str], 
                                 conflict_type: str = "probability") -> Dict[str, Any]:
        """
        Löst einen Konflikt zwischen Zeitlinien mit Q-LOGIK
        
        Args:
            timeline_ids: IDs der konfligierenden Zeitlinien
            conflict_type: Typ des Konflikts ("probability", "causality", "integrity")
            
        Returns:
            Konfliktlösung
        """
        if len(timeline_ids) < 2:
            logger.warning("Mindestens zwei Zeitlinien für Konfliktlösung erforderlich")
            return {"error": "Mindestens zwei Zeitlinien für Konfliktlösung erforderlich"}
        
        # Lade Zeitlinien
        timelines = {}
        for timeline_id in timeline_ids:
            timeline = self.echo_prime.get_timeline(timeline_id)
            if timeline:
                timelines[timeline_id] = timeline
            else:
                logger.warning(f"Zeitlinie {timeline_id} existiert nicht")
        
        if len(timelines) < 2:
            logger.warning("Nicht genügend gültige Zeitlinien für Konfliktlösung")
            return {"error": "Nicht genügend gültige Zeitlinien für Konfliktlösung"}
        
        # Erstelle Konfliktkontext für Q-LOGIK
        conflict_context = {
            "conflict_type": conflict_type,
            "timeline_count": len(timelines),
            "timelines": {},
            "decision_factors": {
                "risk": 0.0,
                "benefit": 0.0,
                "urgency": 0.0,
                "impact": 0.0,
                "confidence": 0.0
            }
        }
        
        # Füge Zeitlinieninformationen hinzu
        max_probability = 0.0
        for timeline_id, timeline in timelines.items():
            conflict_context["timelines"][timeline_id] = {
                "name": timeline.name,
                "type": timeline.type.value,
                "probability": timeline.probability,
                "node_count": len(timeline.nodes),
                "creation_time": timeline.creation_time,
                "last_modified": timeline.last_modified
            }
            
            # Aktualisiere maximale Wahrscheinlichkeit
            max_probability = max(max_probability, timeline.probability)
            
            # Aktualisiere Entscheidungsfaktoren
            conflict_context["decision_factors"]["risk"] += timeline.probability * 0.5
            conflict_context["decision_factors"]["benefit"] += timeline.probability * 0.7
            conflict_context["decision_factors"]["impact"] += timeline.probability * 0.6
        
        # Normalisiere Entscheidungsfaktoren
        for factor in conflict_context["decision_factors"]:
            conflict_context["decision_factors"][factor] /= len(timelines)
        
        # Setze Dringlichkeit basierend auf Konflikttyp
        if conflict_type == "integrity":
            conflict_context["decision_factors"]["urgency"] = 0.9
        elif conflict_type == "causality":
            conflict_context["decision_factors"]["urgency"] = 0.7
        else:  # probability
            conflict_context["decision_factors"]["urgency"] = 0.5
        
        # Setze Vertrauen basierend auf maximaler Wahrscheinlichkeit
        conflict_context["decision_factors"]["confidence"] = max_probability
        
        # Wende Q-LOGIK Konfliktlösung an
        conflict_resolution = self.conflict_resolver.resolve(
            conflict={
                "type": conflict_type,
                "parties": [{"id": tid, "weight": t.probability} for tid, t in timelines.items()]
            },
            context=conflict_context
        )
        
        # Erstelle Lösungsvorschlag
        resolution = {
            "conflict_type": conflict_type,
            "timeline_ids": timeline_ids,
            "timestamp": datetime.now().isoformat(),
            "resolution_strategy": conflict_resolution.get("strategy", "unknown"),
            "actions": []
        }
        
        # Bestimme Aktionen basierend auf Lösungsstrategie
        strategy = conflict_resolution.get("strategy", "")
        
        if strategy == "prioritize_safety":
            # Wähle die Zeitlinie mit der höchsten Wahrscheinlichkeit
            best_timeline = max(timelines.values(), key=lambda t: t.probability)
            resolution["actions"].append({
                "action": "preserve_timeline",
                "timeline_id": best_timeline.id,
                "priority": "high",
                "description": "Bewahre die Zeitlinie mit der höchsten Wahrscheinlichkeit"
            })
            
            # Markiere andere Zeitlinien als veraltet
            for tid, timeline in timelines.items():
                if tid != best_timeline.id:
                    resolution["actions"].append({
                        "action": "deprecate_timeline",
                        "timeline_id": tid,
                        "priority": "medium",
                        "description": "Markiere Zeitlinie als veraltet"
                    })
                    
        elif strategy == "compromise":
            # Erstelle eine neue Zeitlinie, die Elemente aller Zeitlinien kombiniert
            resolution["actions"].append({
                "action": "merge_timelines",
                "timeline_ids": list(timelines.keys()),
                "priority": "high",
                "description": "Erstelle eine neue Zeitlinie, die Elemente aller Zeitlinien kombiniert"
            })
            
        elif strategy == "prioritize_utility":
            # Wende Entanglement auf die beiden besten Zeitlinien an
            sorted_timelines = sorted(timelines.values(), key=lambda t: t.probability, reverse=True)
            if len(sorted_timelines) >= 2:
                resolution["actions"].append({
                    "action": "apply_entanglement",
                    "timeline_id1": sorted_timelines[0].id,
                    "timeline_id2": sorted_timelines[1].id,
                    "priority": "high",
                    "description": "Wende Entanglement auf die beiden besten Zeitlinien an"
                })
                
        else:  # Fallback
            resolution["actions"].append({
                "action": "analyze_timelines",
                "timeline_ids": list(timelines.keys()),
                "priority": "medium",
                "description": "Führe weitere Analyse der Zeitlinien durch"
            })
        
        return resolution


# Globale Instanz für einfachen Zugriff
integration = QLOGIKECHOPrimeIntegration()

def evaluate_timeline(timeline_id: str) -> Dict[str, Any]:
    """
    Evaluiert eine Zeitlinie mit Q-LOGIK
    
    Args:
        timeline_id: ID der zu evaluierenden Zeitlinie
        
    Returns:
        Evaluierungsergebnisse
    """
    return integration.evaluate_timeline(timeline_id)

def recommend_timeline_action(timeline_id: str) -> Dict[str, Any]:
    """
    Empfiehlt eine Aktion für eine Zeitlinie basierend auf Q-LOGIK
    
    Args:
        timeline_id: ID der Zeitlinie
        
    Returns:
        Empfohlene Aktion
    """
    return integration.recommend_timeline_action(timeline_id)

def apply_quantum_decision(timeline_id: str, effect_type: str = "auto") -> Dict[str, Any]:
    """
    Wendet einen Quanteneffekt basierend auf Q-LOGIK-Entscheidung an
    
    Args:
        timeline_id: ID der Zeitlinie
        effect_type: Typ des anzuwendenden Effekts ("superposition", "entanglement", "collapse", "auto")
        
    Returns:
        Angewendeter Quanteneffekt
    """
    return integration.apply_quantum_decision(timeline_id, effect_type)

def resolve_timeline_conflict(timeline_ids: List[str], conflict_type: str = "probability") -> Dict[str, Any]:
    """
    Löst einen Konflikt zwischen Zeitlinien mit Q-LOGIK
    
    Args:
        timeline_ids: IDs der konfligierenden Zeitlinien
        conflict_type: Typ des Konflikts ("probability", "causality", "integrity")
        
    Returns:
        Konfliktlösung
    """
    return integration.resolve_timeline_conflict(timeline_ids, conflict_type)


if __name__ == "__main__":
    # Beispiel für die Verwendung der Integration
    logging.basicConfig(level=logging.INFO)
    
    # Erstelle ECHO-PRIME Controller
    echo_prime = EchoPrimeController()
    
    # Erstelle eine Testzeitlinie
    timeline = echo_prime.create_timeline(
        name="Testzeitlinie",
        description="Eine Testzeitlinie für die Q-LOGIK Integration"
    )
    
    # Füge einige Knoten hinzu
    node1 = echo_prime.add_time_node(
        timeline_id=timeline.id,
        description="Startknoten",
        probability=0.9,
        trigger_level=TriggerLevel.LOW
    )
    
    node2 = echo_prime.add_time_node(
        timeline_id=timeline.id,
        description="Zweiter Knoten",
        parent_node_id=node1.id,
        probability=0.8,
        trigger_level=TriggerLevel.MEDIUM
    )
    
    # Evaluiere die Zeitlinie
    evaluation = evaluate_timeline(timeline.id)
    print(f"Evaluation: {evaluation}")
    
    # Empfehle eine Aktion
    recommendation = recommend_timeline_action(timeline.id)
    print(f"Empfehlung: {recommendation}")
    
    # Wende einen Quanteneffekt an
    effect = apply_quantum_decision(timeline.id, "superposition")
    print(f"Quanteneffekt: {effect}")
