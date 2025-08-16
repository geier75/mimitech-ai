#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - ECHO-PRIME TriggerMatrixAnalyzer

Diese Datei implementiert den TriggerMatrixAnalyzer, der Auslöser für 
Zeitlinienverschiebungen analysiert und deren Gewichtung berechnet.

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
logger = logging.getLogger("MISO.timeline.trigger_analyzer")

# Importiere gemeinsame Datenstrukturen
from miso.timeline.echo_prime import TimeNode, Timeline, Trigger, TemporalEvent, TriggerLevel

class TriggerCategory(Enum):
    """Kategorien für Trigger"""
    EMOTIONAL = "emotional"
    POLITICAL = "political"
    ECONOMIC = "economic"
    STRATEGIC = "strategic"
    SOCIAL = "social"
    TECHNOLOGICAL = "technological"
    ENVIRONMENTAL = "environmental"
    LEGAL = "legal"

class TriggerMatrixAnalyzer:
    """
    Analysiert Auslöser für Zeitlinienverschiebungen und deren Gewichtung
    """
    
    def __init__(self):
        """Initialisiert den TriggerMatrixAnalyzer"""
        self.triggers = {}
        self.trigger_matrix = {}  # Beziehungen zwischen Triggern
        self.category_weights = {
            TriggerCategory.EMOTIONAL.value: 0.7,
            TriggerCategory.POLITICAL.value: 0.8,
            TriggerCategory.ECONOMIC.value: 0.9,
            TriggerCategory.STRATEGIC.value: 1.0,
            TriggerCategory.SOCIAL.value: 0.75,
            TriggerCategory.TECHNOLOGICAL.value: 0.85,
            TriggerCategory.ENVIRONMENTAL.value: 0.6,
            TriggerCategory.LEGAL.value: 0.8
        }
        logger.info("TriggerMatrixAnalyzer initialisiert")
    
    def register_trigger(self, trigger: Trigger) -> str:
        """
        Registriert einen neuen Trigger
        
        Args:
            trigger: Trigger-Objekt
            
        Returns:
            ID des registrierten Triggers
        """
        self.triggers[trigger.id] = trigger
        logger.info(f"Trigger '{trigger.name}' (ID: {trigger.id}) registriert")
        
        # Aktualisiere Trigger-Matrix
        self._update_trigger_matrix(trigger)
        
        return trigger.id
    
    def _update_trigger_matrix(self, trigger: Trigger):
        """
        Aktualisiert die Trigger-Matrix mit einem neuen Trigger
        
        Args:
            trigger: Neuer Trigger
        """
        if trigger.id not in self.trigger_matrix:
            self.trigger_matrix[trigger.id] = {}
        
        # Berechne Beziehungen zu anderen Triggern
        for other_id, other_trigger in self.triggers.items():
            if other_id != trigger.id:
                # Berechne Beziehungsstärke (0.0-1.0)
                relation_strength = self._calculate_relation_strength(trigger, other_trigger)
                
                # Speichere Beziehung in beide Richtungen
                self.trigger_matrix[trigger.id][other_id] = relation_strength
                
                if other_id not in self.trigger_matrix:
                    self.trigger_matrix[other_id] = {}
                self.trigger_matrix[other_id][trigger.id] = relation_strength
    
    def _calculate_relation_strength(self, trigger1: Trigger, trigger2: Trigger) -> float:
        """
        Berechnet die Beziehungsstärke zwischen zwei Triggern
        
        Args:
            trigger1: Erster Trigger
            trigger2: Zweiter Trigger
            
        Returns:
            Beziehungsstärke (0.0-1.0)
        """
        # Basisstärke basierend auf Kategorie
        if trigger1.category == trigger2.category:
            base_strength = 0.7
        else:
            base_strength = 0.3
        
        # Gewichtung basierend auf Trigger-Level
        level_factor = 1.0
        if trigger1.level == TriggerLevel.CRITICAL or trigger2.level == TriggerLevel.CRITICAL:
            level_factor = 1.5
        elif trigger1.level == TriggerLevel.HIGH or trigger2.level == TriggerLevel.HIGH:
            level_factor = 1.3
        elif trigger1.level == TriggerLevel.MEDIUM or trigger2.level == TriggerLevel.MEDIUM:
            level_factor = 1.1
        
        # Gewichtung basierend auf Gewicht
        weight_factor = (trigger1.weight + trigger2.weight) / 2.0
        
        # Berechne Gesamtstärke
        strength = base_strength * level_factor * weight_factor
        
        # Begrenze auf 0.0-1.0
        return min(max(strength, 0.0), 1.0)
    
    def create_trigger(self, 
                     name: str, 
                     description: str, 
                     level: TriggerLevel, 
                     conditions: Dict[str, Any],
                     weight: float,
                     category: str) -> Trigger:
        """
        Erstellt einen neuen Trigger
        
        Args:
            name: Name des Triggers
            description: Beschreibung des Triggers
            level: Trigger-Level
            conditions: Bedingungen für den Trigger
            weight: Gewicht des Triggers (0.0-1.0)
            category: Kategorie des Triggers
            
        Returns:
            Neu erstellter Trigger
        """
        trigger_id = str(uuid.uuid4())
        
        trigger = Trigger(
            id=trigger_id,
            name=name,
            description=description,
            level=level,
            conditions=conditions,
            weight=weight,
            category=category
        )
        
        # Registriere Trigger
        self.register_trigger(trigger)
        
        return trigger
    
    def get_trigger(self, trigger_id: str) -> Optional[Trigger]:
        """
        Gibt einen Trigger zurück
        
        Args:
            trigger_id: ID des Triggers
            
        Returns:
            Trigger oder None, falls nicht gefunden
        """
        return self.triggers.get(trigger_id)
    
    def get_all_triggers(self) -> Dict[str, Trigger]:
        """
        Gibt alle Trigger zurück
        
        Returns:
            Dictionary mit allen Triggern
        """
        return self.triggers
    
    def delete_trigger(self, trigger_id: str) -> bool:
        """
        Löscht einen Trigger
        
        Args:
            trigger_id: ID des zu löschenden Triggers
            
        Returns:
            True, wenn erfolgreich gelöscht, sonst False
        """
        if trigger_id in self.triggers:
            # Lösche Trigger
            del self.triggers[trigger_id]
            
            # Lösche Einträge in der Trigger-Matrix
            if trigger_id in self.trigger_matrix:
                del self.trigger_matrix[trigger_id]
            
            for other_id in self.trigger_matrix:
                if trigger_id in self.trigger_matrix[other_id]:
                    del self.trigger_matrix[other_id][trigger_id]
            
            logger.info(f"Trigger {trigger_id} gelöscht")
            return True
        
        logger.warning(f"Trigger {trigger_id} existiert nicht")
        return False
    
    def analyze_event(self, event: TemporalEvent) -> Dict[str, Any]:
        """
        Analysiert ein temporales Ereignis und identifiziert relevante Trigger
        
        Args:
            event: Temporales Ereignis
            
        Returns:
            Analyseergebnis mit relevanten Triggern und deren Gewichtung
        """
        result = {
            "event_id": event.id,
            "relevant_triggers": [],
            "trigger_weights": {},
            "dominant_triggers": [],
            "total_impact": 0.0
        }
        
        # Wenn das Ereignis bereits Trigger hat, analysiere diese
        if event.triggers:
            for trigger_id in event.triggers:
                trigger = self.get_trigger(trigger_id)
                if trigger:
                    # Berechne gewichteten Einfluss
                    impact = trigger.weight * self.category_weights.get(trigger.category, 0.5)
                    
                    result["relevant_triggers"].append(trigger_id)
                    result["trigger_weights"][trigger_id] = impact
                    result["total_impact"] += impact
        
        # Suche nach weiteren relevanten Triggern basierend auf der Ereignisbeschreibung
        # In einer realen Implementierung würde hier eine komplexe Analyse stattfinden
        # Für dieses Beispiel verwenden wir eine einfache Wortübereinstimmung
        for trigger_id, trigger in self.triggers.items():
            if trigger_id not in result["relevant_triggers"]:
                # Prüfe, ob Trigger-Name oder Beschreibung im Ereignis vorkommt
                if (trigger.name.lower() in event.description.lower() or 
                    any(word.lower() in event.description.lower() 
                        for word in trigger.description.lower().split())):
                    
                    # Berechne gewichteten Einfluss
                    impact = trigger.weight * self.category_weights.get(trigger.category, 0.5) * 0.7  # Reduziere Gewicht für indirekte Übereinstimmungen
                    
                    result["relevant_triggers"].append(trigger_id)
                    result["trigger_weights"][trigger_id] = impact
                    result["total_impact"] += impact
        
        # Identifiziere dominante Trigger (Top 3 nach Gewicht)
        if result["trigger_weights"]:
            sorted_triggers = sorted(
                result["trigger_weights"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Nehme die Top 3 oder alle, falls weniger als 3
            result["dominant_triggers"] = [t[0] for t in sorted_triggers[:min(3, len(sorted_triggers))]]
        
        return result
    
    def find_related_triggers(self, trigger_id: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Findet verwandte Trigger basierend auf der Trigger-Matrix
        
        Args:
            trigger_id: ID des Ausgangs-Triggers
            threshold: Schwellenwert für die Beziehungsstärke (0.0-1.0)
            
        Returns:
            Liste von Tupeln (Trigger-ID, Beziehungsstärke)
        """
        if trigger_id not in self.trigger_matrix:
            logger.warning(f"Trigger {trigger_id} nicht in der Trigger-Matrix")
            return []
        
        # Finde verwandte Trigger über dem Schwellenwert
        related = [
            (other_id, strength)
            for other_id, strength in self.trigger_matrix[trigger_id].items()
            if strength >= threshold
        ]
        
        # Sortiere nach Beziehungsstärke (absteigend)
        related.sort(key=lambda x: x[1], reverse=True)
        
        return related
    
    def get_trigger_impact(self, trigger_id: str) -> float:
        """
        Berechnet den Gesamteinfluss eines Triggers basierend auf seinen Eigenschaften
        
        Args:
            trigger_id: ID des Triggers
            
        Returns:
            Einfluss des Triggers (0.0-1.0)
        """
        trigger = self.get_trigger(trigger_id)
        if not trigger:
            logger.warning(f"Trigger {trigger_id} existiert nicht")
            return 0.0
        
        # Basiseinfluss basierend auf Gewicht
        base_impact = trigger.weight
        
        # Faktor basierend auf Trigger-Level
        level_factor = 1.0
        if trigger.level == TriggerLevel.CRITICAL:
            level_factor = 2.0
        elif trigger.level == TriggerLevel.HIGH:
            level_factor = 1.5
        elif trigger.level == TriggerLevel.MEDIUM:
            level_factor = 1.2
        
        # Faktor basierend auf Kategorie
        category_factor = self.category_weights.get(trigger.category, 0.5)
        
        # Berechne Gesamteinfluss
        impact = base_impact * level_factor * category_factor
        
        # Begrenze auf 0.0-1.0
        return min(max(impact, 0.0), 1.0)
    
    def get_dominant_triggers(self, count: int = 3) -> List[Tuple[str, float]]:
        """
        Gibt die dominanten Trigger basierend auf ihrem Einfluss zurück
        
        Args:
            count: Anzahl der zurückzugebenden Trigger
            
        Returns:
            Liste von Tupeln (Trigger-ID, Einfluss)
        """
        # Berechne Einfluss für alle Trigger
        impacts = [
            (trigger_id, self.get_trigger_impact(trigger_id))
            for trigger_id in self.triggers
        ]
        
        # Sortiere nach Einfluss (absteigend)
        impacts.sort(key=lambda x: x[1], reverse=True)
        
        # Gib die Top N zurück
        return impacts[:min(count, len(impacts))]
    
    def analyze_timeline(self, timeline: Timeline) -> Dict[str, Any]:
        """
        Analysiert eine Zeitlinie und identifiziert dominante Trigger
        
        Args:
            timeline: Zeitlinie
            
        Returns:
            Analyseergebnis mit dominanten Triggern und deren Einfluss
        """
        result = {
            "timeline_id": timeline.id,
            "timeline_name": timeline.name,
            "trigger_impacts": {},
            "dominant_triggers": [],
            "trigger_categories": {},
            "critical_nodes": []
        }
        
        # Erstelle temporale Ereignisse für jeden Knoten
        events = []
        for node_id, node in timeline.nodes.items():
            event = TemporalEvent(
                id=str(uuid.uuid4()),
                timeline_id=timeline.id,
                node_id=node_id,
                name=f"Event at {node.description}",
                description=node.description,
                timestamp=node.timestamp,
                impact_score=0.0,  # Wird später berechnet
                triggers=[],  # Wird später gefüllt
                probability=node.probability
            )
            events.append(event)
        
        # Analysiere jedes Ereignis
        all_triggers = set()
        for event in events:
            analysis = self.analyze_event(event)
            
            # Sammle alle Trigger
            all_triggers.update(analysis["relevant_triggers"])
            
            # Speichere kritische Knoten
            if analysis["total_impact"] > 0.7:
                result["critical_nodes"].append(event.node_id)
        
        # Berechne Gesamteinfluss für jeden Trigger
        for trigger_id in all_triggers:
            impact = self.get_trigger_impact(trigger_id)
            result["trigger_impacts"][trigger_id] = impact
            
            # Kategorisiere Trigger
            trigger = self.get_trigger(trigger_id)
            if trigger:
                if trigger.category not in result["trigger_categories"]:
                    result["trigger_categories"][trigger.category] = []
                result["trigger_categories"][trigger.category].append(trigger_id)
        
        # Identifiziere dominante Trigger
        if result["trigger_impacts"]:
            sorted_triggers = sorted(
                result["trigger_impacts"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Nehme die Top 5 oder alle, falls weniger als 5
            result["dominant_triggers"] = [t[0] for t in sorted_triggers[:min(5, len(sorted_triggers))]]
        
        return result
