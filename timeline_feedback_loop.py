#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - ECHO-PRIME TimelineFeedbackLoop

Diese Datei implementiert die TimelineFeedbackLoop, die Strategie-Vorschläge 
durch Rückvergleiche von Zeitlinien und Ereignissen generiert.

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
logger = logging.getLogger("MISO.timeline.feedback_loop")

# Importiere gemeinsame Datenstrukturen
from miso.timeline.echo_prime import TimeNode, Timeline, Trigger, TemporalEvent, TriggerLevel

@dataclass
class StrategicRecommendation:
    """Repräsentiert eine strategische Empfehlung"""
    id: str
    title: str
    description: str
    confidence: float  # 0.0-1.0
    impact: float  # 0.0-1.0
    timeline_id: str
    node_ids: List[str]
    trigger_ids: List[str]
    action_steps: List[str]
    optimal_execution_time: Optional[float] = None
    expiration_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert die Empfehlung in ein Dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "impact": self.impact,
            "timeline_id": self.timeline_id,
            "node_ids": self.node_ids,
            "trigger_ids": self.trigger_ids,
            "action_steps": self.action_steps,
            "optimal_execution_time": self.optimal_execution_time,
            "expiration_time": self.expiration_time,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategicRecommendation':
        """Erstellt eine Empfehlung aus einem Dictionary"""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            confidence=data["confidence"],
            impact=data["impact"],
            timeline_id=data["timeline_id"],
            node_ids=data["node_ids"],
            trigger_ids=data["trigger_ids"],
            action_steps=data["action_steps"],
            optimal_execution_time=data["optimal_execution_time"],
            expiration_time=data["expiration_time"],
            metadata=data["metadata"]
        )

class TimelineFeedbackLoop:
    """
    Liefert Empfehlungen auf Basis vergangener Muster & zukünftiger Simulationen
    """
    
    def __init__(self):
        """Initialisiert die TimelineFeedbackLoop"""
        self.recommendations = {}
        self.feedback_history = {}
        self.timeline_cache = {}
        self.trigger_cache = {}
        self.confidence_threshold = 0.6  # Mindestwert für Empfehlungen
        logger.info("TimelineFeedbackLoop initialisiert")
    
    def analyze_timeline_pair(self, 
                            timeline1: Timeline, 
                            timeline2: Timeline) -> Dict[str, Any]:
        """
        Analysiert ein Paar von Zeitlinien und identifiziert Unterschiede und Gemeinsamkeiten
        
        Args:
            timeline1: Erste Zeitlinie
            timeline2: Zweite Zeitlinie
            
        Returns:
            Analyseergebnis mit Unterschieden und Gemeinsamkeiten
        """
        result = {
            "timeline1_id": timeline1.id,
            "timeline2_id": timeline2.id,
            "common_nodes": [],
            "divergent_nodes": [],
            "critical_divergence_points": [],
            "probability_delta": timeline1.probability - timeline2.probability,
            "time_shift_points": []
        }
        
        # Finde gemeinsame Knoten (gleiche ID)
        common_node_ids = set(timeline1.nodes.keys()) & set(timeline2.nodes.keys())
        result["common_nodes"] = list(common_node_ids)
        
        # Finde unterschiedliche Knoten
        divergent_nodes1 = set(timeline1.nodes.keys()) - set(timeline2.nodes.keys())
        divergent_nodes2 = set(timeline2.nodes.keys()) - set(timeline1.nodes.keys())
        result["divergent_nodes"] = {
            "timeline1_only": list(divergent_nodes1),
            "timeline2_only": list(divergent_nodes2)
        }
        
        # Identifiziere kritische Verzweigungspunkte
        # Ein kritischer Verzweigungspunkt ist ein gemeinsamer Knoten, der in beiden Zeitlinien
        # unterschiedliche Kinder hat
        for node_id in common_node_ids:
            node1 = timeline1.nodes[node_id]
            node2 = timeline2.nodes[node_id]
            
            # Wenn die Kinder unterschiedlich sind, ist dies ein Verzweigungspunkt
            if set(node1.child_node_ids) != set(node2.child_node_ids):
                result["critical_divergence_points"].append(node_id)
        
        # Identifiziere Zeitverschiebungspunkte
        # Ein Zeitverschiebungspunkt ist ein Knoten, der in beiden Zeitlinien existiert,
        # aber zu unterschiedlichen Zeiten auftritt
        for node_id in common_node_ids:
            node1 = timeline1.nodes[node_id]
            node2 = timeline2.nodes[node_id]
            
            # Wenn der Zeitunterschied signifikant ist (> 1 Stunde)
            time_delta = abs(node1.timestamp - node2.timestamp)
            if time_delta > 3600:
                result["time_shift_points"].append({
                    "node_id": node_id,
                    "time_delta": time_delta,
                    "earlier_timeline": timeline1.id if node1.timestamp < node2.timestamp else timeline2.id
                })
        
        return result
    
    def generate_recommendations(self, 
                               timeline: Timeline, 
                               alternative_timelines: List[Timeline],
                               triggers: Dict[str, Trigger]) -> List[StrategicRecommendation]:
        """
        Generiert strategische Empfehlungen basierend auf einer Zeitlinie und Alternativen
        
        Args:
            timeline: Hauptzeitlinie
            alternative_timelines: Alternative Zeitlinien
            triggers: Verfügbare Trigger
            
        Returns:
            Liste von strategischen Empfehlungen
        """
        recommendations = []
        
        # Cache Zeitlinien und Trigger
        self.timeline_cache[timeline.id] = timeline
        for alt_timeline in alternative_timelines:
            self.timeline_cache[alt_timeline.id] = alt_timeline
        self.trigger_cache.update(triggers)
        
        # Sortiere alternative Zeitlinien nach Wahrscheinlichkeit (absteigend)
        sorted_alternatives = sorted(
            alternative_timelines,
            key=lambda t: t.probability,
            reverse=True
        )
        
        # Analysiere jede alternative Zeitlinie im Vergleich zur Hauptzeitlinie
        for alt_timeline in sorted_alternatives:
            analysis = self.analyze_timeline_pair(timeline, alt_timeline)
            
            # Wenn die alternative Zeitlinie wahrscheinlicher ist als die Hauptzeitlinie,
            # generiere Empfehlungen, um zur alternativen Zeitlinie zu wechseln
            if alt_timeline.probability > timeline.probability:
                recommendations.extend(
                    self._generate_shift_recommendations(timeline, alt_timeline, analysis, triggers)
                )
            
            # Wenn die Hauptzeitlinie wahrscheinlicher ist, aber die alternative Zeitlinie
            # bestimmte Vorteile bietet, generiere Empfehlungen zur Optimierung
            elif timeline.probability > alt_timeline.probability:
                recommendations.extend(
                    self._generate_optimization_recommendations(timeline, alt_timeline, analysis, triggers)
                )
        
        # Filtere Empfehlungen nach Konfidenz
        filtered_recommendations = [
            rec for rec in recommendations
            if rec.confidence >= self.confidence_threshold
        ]
        
        # Sortiere nach Auswirkung und Konfidenz
        sorted_recommendations = sorted(
            filtered_recommendations,
            key=lambda r: (r.impact * r.confidence),
            reverse=True
        )
        
        # Speichere Empfehlungen
        for rec in sorted_recommendations:
            self.recommendations[rec.id] = rec
        
        return sorted_recommendations
    
    def _generate_shift_recommendations(self, 
                                      main_timeline: Timeline, 
                                      alt_timeline: Timeline,
                                      analysis: Dict[str, Any],
                                      triggers: Dict[str, Trigger]) -> List[StrategicRecommendation]:
        """
        Generiert Empfehlungen zum Wechsel zur alternativen Zeitlinie
        
        Args:
            main_timeline: Hauptzeitlinie
            alt_timeline: Alternative Zeitlinie
            analysis: Analyseergebnis
            triggers: Verfügbare Trigger
            
        Returns:
            Liste von strategischen Empfehlungen
        """
        recommendations = []
        
        # Identifiziere kritische Verzweigungspunkte
        for node_id in analysis["critical_divergence_points"]:
            # Finde relevante Trigger für diesen Verzweigungspunkt
            relevant_triggers = self._find_relevant_triggers(node_id, main_timeline, alt_timeline, triggers)
            
            if relevant_triggers:
                # Erstelle Empfehlung für jeden kritischen Verzweigungspunkt
                rec_id = str(uuid.uuid4())
                
                # Bestimme optimalen Ausführungszeitpunkt
                # In der Regel kurz vor dem Verzweigungspunkt
                node = main_timeline.nodes[node_id]
                optimal_time = node.timestamp - 3600  # 1 Stunde vor dem Verzweigungspunkt
                
                # Erstelle Aktionsschritte
                action_steps = []
                for trigger_id in relevant_triggers:
                    trigger = triggers.get(trigger_id)
                    if trigger:
                        action_steps.append(f"Aktiviere Trigger '{trigger.name}' ({trigger.category})")
                
                # Berechne Konfidenz und Auswirkung
                confidence = min(alt_timeline.probability, 0.95)  # Maximal 95% Konfidenz
                impact = abs(alt_timeline.probability - main_timeline.probability) * 10  # Skaliere auf 0-10
                
                recommendation = StrategicRecommendation(
                    id=rec_id,
                    title=f"Wechsel zu optimaler Zeitlinie {alt_timeline.id}",
                    description=f"Durch gezielte Aktivierung von Triggern kann ein Wechsel zur wahrscheinlicheren Zeitlinie {alt_timeline.id} ({alt_timeline.name}) erreicht werden.",
                    confidence=confidence,
                    impact=min(impact, 1.0),  # Begrenze auf 0.0-1.0
                    timeline_id=alt_timeline.id,
                    node_ids=[node_id],
                    trigger_ids=relevant_triggers,
                    action_steps=action_steps,
                    optimal_execution_time=optimal_time,
                    expiration_time=node.timestamp  # Verfällt zum Zeitpunkt des Verzweigungspunkts
                )
                
                recommendations.append(recommendation)
        
        # Identifiziere Zeitverschiebungspunkte
        for time_shift in analysis["time_shift_points"]:
            node_id = time_shift["node_id"]
            
            # Wenn die alternative Zeitlinie früher ist, generiere Empfehlung zur Beschleunigung
            if time_shift["earlier_timeline"] == alt_timeline.id:
                # Finde relevante Trigger
                relevant_triggers = self._find_relevant_triggers(node_id, main_timeline, alt_timeline, triggers)
                
                if relevant_triggers:
                    rec_id = str(uuid.uuid4())
                    
                    # Bestimme optimalen Ausführungszeitpunkt
                    node = main_timeline.nodes[node_id]
                    optimal_time = node.timestamp - time_shift["time_delta"] - 3600  # 1 Stunde vor dem früheren Zeitpunkt
                    
                    # Erstelle Aktionsschritte
                    action_steps = [
                        f"Beschleunige Ereignis um {time_shift['time_delta'] / 3600:.1f} Stunden"
                    ]
                    for trigger_id in relevant_triggers:
                        trigger = triggers.get(trigger_id)
                        if trigger:
                            action_steps.append(f"Aktiviere Trigger '{trigger.name}' frühzeitig")
                    
                    # Berechne Konfidenz und Auswirkung
                    confidence = 0.7  # Zeitverschiebungen sind weniger zuverlässig
                    impact = 0.6  # Moderate Auswirkung
                    
                    recommendation = StrategicRecommendation(
                        id=rec_id,
                        title=f"Beschleunige Ereignis für optimalen Zeitpunkt",
                        description=f"Durch frühzeitige Aktivierung von Triggern kann das Ereignis '{node.description}' beschleunigt werden, was zu einem günstigeren Zeitverlauf führt.",
                        confidence=confidence,
                        impact=impact,
                        timeline_id=alt_timeline.id,
                        node_ids=[node_id],
                        trigger_ids=relevant_triggers,
                        action_steps=action_steps,
                        optimal_execution_time=optimal_time,
                        expiration_time=node.timestamp - time_shift["time_delta"]  # Verfällt zum früheren Zeitpunkt
                    )
                    
                    recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_optimization_recommendations(self, 
                                             main_timeline: Timeline, 
                                             alt_timeline: Timeline,
                                             analysis: Dict[str, Any],
                                             triggers: Dict[str, Trigger]) -> List[StrategicRecommendation]:
        """
        Generiert Empfehlungen zur Optimierung der Hauptzeitlinie
        
        Args:
            main_timeline: Hauptzeitlinie
            alt_timeline: Alternative Zeitlinie
            analysis: Analyseergebnis
            triggers: Verfügbare Trigger
            
        Returns:
            Liste von strategischen Empfehlungen
        """
        recommendations = []
        
        # Identifiziere vorteilhafte Knoten in der alternativen Zeitlinie
        # In einer realen Implementierung würde hier eine komplexe Analyse stattfinden
        # Für dieses Beispiel nehmen wir an, dass Knoten mit hoher Wahrscheinlichkeit vorteilhaft sind
        
        # Finde Knoten, die nur in der alternativen Zeitlinie existieren
        alt_only_nodes = analysis["divergent_nodes"]["timeline2_only"]
        
        for node_id in alt_only_nodes:
            node = alt_timeline.nodes[node_id]
            
            # Wenn der Knoten eine hohe Wahrscheinlichkeit hat, ist er möglicherweise vorteilhaft
            if node.probability > 0.7:
                # Finde den Elternknoten
                parent_id = node.parent_node_id
                if parent_id and parent_id in main_timeline.nodes:
                    # Finde relevante Trigger
                    relevant_triggers = self._find_relevant_triggers(parent_id, main_timeline, alt_timeline, triggers)
                    
                    if relevant_triggers:
                        rec_id = str(uuid.uuid4())
                        
                        # Bestimme optimalen Ausführungszeitpunkt
                        parent_node = main_timeline.nodes[parent_id]
                        optimal_time = parent_node.timestamp
                        
                        # Erstelle Aktionsschritte
                        action_steps = [
                            f"Integriere vorteilhaftes Ereignis '{node.description}' in die Hauptzeitlinie"
                        ]
                        for trigger_id in relevant_triggers:
                            trigger = triggers.get(trigger_id)
                            if trigger:
                                action_steps.append(f"Aktiviere Trigger '{trigger.name}' zum optimalen Zeitpunkt")
                        
                        # Berechne Konfidenz und Auswirkung
                        confidence = node.probability * 0.8  # Reduziere Konfidenz etwas
                        impact = 0.5  # Moderate Auswirkung
                        
                        recommendation = StrategicRecommendation(
                            id=rec_id,
                            title=f"Integriere vorteilhaftes Ereignis in Hauptzeitlinie",
                            description=f"Durch gezielte Aktivierung von Triggern kann das vorteilhafte Ereignis '{node.description}' aus der alternativen Zeitlinie in die Hauptzeitlinie integriert werden.",
                            confidence=confidence,
                            impact=impact,
                            timeline_id=main_timeline.id,
                            node_ids=[parent_id, node_id],
                            trigger_ids=relevant_triggers,
                            action_steps=action_steps,
                            optimal_execution_time=optimal_time,
                            expiration_time=optimal_time + 86400  # Verfällt nach 24 Stunden
                        )
                        
                        recommendations.append(recommendation)
        
        return recommendations
    
    def _find_relevant_triggers(self, 
                              node_id: str, 
                              main_timeline: Timeline, 
                              alt_timeline: Timeline,
                              triggers: Dict[str, Trigger]) -> List[str]:
        """
        Findet relevante Trigger für einen Verzweigungspunkt
        
        Args:
            node_id: ID des Knotens
            main_timeline: Hauptzeitlinie
            alt_timeline: Alternative Zeitlinie
            triggers: Verfügbare Trigger
            
        Returns:
            Liste von relevanten Trigger-IDs
        """
        relevant_triggers = []
        
        # In einer realen Implementierung würde hier eine komplexe Analyse stattfinden
        # Für dieses Beispiel verwenden wir eine einfache Heuristik
        
        # Prüfe alle Trigger
        for trigger_id, trigger in triggers.items():
            # Prüfe, ob der Trigger für den Knoten relevant ist
            # Einfache Heuristik: Wenn der Trigger-Name oder die Beschreibung im Knoten vorkommt
            node = main_timeline.nodes.get(node_id) or alt_timeline.nodes.get(node_id)
            if node:
                if (trigger.name.lower() in node.description.lower() or 
                    any(word.lower() in node.description.lower() 
                        for word in trigger.description.lower().split())):
                    relevant_triggers.append(trigger_id)
        
        return relevant_triggers
    
    def get_recommendation(self, recommendation_id: str) -> Optional[StrategicRecommendation]:
        """
        Gibt eine Empfehlung zurück
        
        Args:
            recommendation_id: ID der Empfehlung
            
        Returns:
            StrategicRecommendation oder None, falls nicht gefunden
        """
        return self.recommendations.get(recommendation_id)
    
    def get_all_recommendations(self) -> Dict[str, StrategicRecommendation]:
        """
        Gibt alle Empfehlungen zurück
        
        Returns:
            Dictionary mit allen Empfehlungen
        """
        return self.recommendations
    
    def get_active_recommendations(self) -> List[StrategicRecommendation]:
        """
        Gibt alle aktiven Empfehlungen zurück (nicht abgelaufen)
        
        Returns:
            Liste von aktiven Empfehlungen
        """
        current_time = time.time()
        
        return [
            rec for rec in self.recommendations.values()
            if not rec.expiration_time or rec.expiration_time > current_time
        ]
    
    def get_optimal_recommendations(self, count: int = 3) -> List[StrategicRecommendation]:
        """
        Gibt die optimalen Empfehlungen zurück
        
        Args:
            count: Anzahl der zurückzugebenden Empfehlungen
            
        Returns:
            Liste von optimalen Empfehlungen
        """
        # Filtere aktive Empfehlungen
        active_recommendations = self.get_active_recommendations()
        
        # Sortiere nach Auswirkung und Konfidenz
        sorted_recommendations = sorted(
            active_recommendations,
            key=lambda r: (r.impact * r.confidence),
            reverse=True
        )
        
        # Gib die Top N zurück
        return sorted_recommendations[:min(count, len(sorted_recommendations))]
    
    def record_feedback(self, 
                      recommendation_id: str, 
                      success: bool, 
                      notes: str = None) -> bool:
        """
        Zeichnet Feedback zu einer Empfehlung auf
        
        Args:
            recommendation_id: ID der Empfehlung
            success: War die Empfehlung erfolgreich?
            notes: Optionale Notizen
            
        Returns:
            True, wenn erfolgreich aufgezeichnet, sonst False
        """
        if recommendation_id not in self.recommendations:
            logger.warning(f"Empfehlung {recommendation_id} existiert nicht")
            return False
        
        # Erstelle Feedback-Eintrag
        feedback_id = str(uuid.uuid4())
        feedback = {
            "id": feedback_id,
            "recommendation_id": recommendation_id,
            "success": success,
            "timestamp": time.time(),
            "notes": notes
        }
        
        # Speichere Feedback
        if recommendation_id not in self.feedback_history:
            self.feedback_history[recommendation_id] = []
        
        self.feedback_history[recommendation_id].append(feedback)
        
        # Aktualisiere Konfidenz der Empfehlung basierend auf Feedback
        self._update_confidence(recommendation_id, success)
        
        logger.info(f"Feedback für Empfehlung {recommendation_id} aufgezeichnet: {'Erfolg' if success else 'Misserfolg'}")
        
        return True
    
    def _update_confidence(self, recommendation_id: str, success: bool):
        """
        Aktualisiert die Konfidenz einer Empfehlung basierend auf Feedback
        
        Args:
            recommendation_id: ID der Empfehlung
            success: War die Empfehlung erfolgreich?
        """
        recommendation = self.recommendations.get(recommendation_id)
        if not recommendation:
            return
        
        # Berechne neue Konfidenz
        if success:
            # Erhöhe Konfidenz bei Erfolg
            new_confidence = recommendation.confidence * 1.1
        else:
            # Verringere Konfidenz bei Misserfolg
            new_confidence = recommendation.confidence * 0.8
        
        # Begrenze auf 0.0-1.0
        recommendation.confidence = min(max(new_confidence, 0.0), 1.0)
    
    def get_feedback_history(self, recommendation_id: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Gibt die Feedback-Historie zurück
        
        Args:
            recommendation_id: ID der Empfehlung (optional)
            
        Returns:
            Dictionary mit Feedback-Historie
        """
        if recommendation_id:
            return {recommendation_id: self.feedback_history.get(recommendation_id, [])}
        
        return self.feedback_history
    
    def clear_expired_recommendations(self) -> int:
        """
        Löscht abgelaufene Empfehlungen
        
        Returns:
            Anzahl der gelöschten Empfehlungen
        """
        current_time = time.time()
        expired_ids = [
            rec_id for rec_id, rec in self.recommendations.items()
            if rec.expiration_time and rec.expiration_time < current_time
        ]
        
        # Lösche abgelaufene Empfehlungen
        for rec_id in expired_ids:
            del self.recommendations[rec_id]
        
        logger.info(f"{len(expired_ids)} abgelaufene Empfehlungen gelöscht")
        
        return len(expired_ids)
