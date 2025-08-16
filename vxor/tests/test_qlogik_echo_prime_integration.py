#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für die Q-LOGIK ECHO-PRIME Integration

Dieses Testskript überprüft die korrekte Funktionsweise der Integration
zwischen Q-LOGIK und ECHO-PRIME.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import logging
from typing import Dict, Any, List
import uuid
from datetime import datetime

# Pfad zum Hauptverzeichnis hinzufügen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiere zu testende Module
from miso.logic.qlogik_echo_prime import (
    QLOGIKECHOPrimeIntegration,
    TimelineDecisionContext,
    evaluate_timeline,
    recommend_timeline_action,
    apply_quantum_decision,
    resolve_timeline_conflict
)
from miso.timeline.echo_prime import Timeline, TimeNode, TriggerLevel, TimelineType
from miso.timeline.echo_prime_controller import EchoPrimeController
from miso.timeline.qtm_modulator import QTM_Modulator, QuantumTimeEffect, QuantumState

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.Test.QLOGIKECHOPrimeIntegration")


class TestQLOGIKECHOPrimeIntegration(unittest.TestCase):
    """Testklasse für die Q-LOGIK ECHO-PRIME Integration"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        self.echo_prime = EchoPrimeController()
        self.integration = QLOGIKECHOPrimeIntegration()
        
        # Erstelle Testzeitlinien
        self.timeline1 = self.echo_prime.create_timeline(
            name="Testzeitlinie 1",
            description="Eine Testzeitlinie für die Q-LOGIK Integration"
        )
        
        self.timeline2 = self.echo_prime.create_timeline(
            name="Testzeitlinie 2",
            description="Eine weitere Testzeitlinie für die Q-LOGIK Integration"
        )
        
        # Füge Knoten zu Zeitlinie 1 hinzu
        self.node1_1 = self.echo_prime.add_time_node(
            timeline_id=self.timeline1.id,
            description="Startknoten",
            probability=0.9,
            trigger_level=TriggerLevel.LOW
        )
        
        self.node1_2 = self.echo_prime.add_time_node(
            timeline_id=self.timeline1.id,
            description="Zweiter Knoten",
            parent_node_id=self.node1_1.id,
            probability=0.8,
            trigger_level=TriggerLevel.MEDIUM
        )
        
        # Füge Knoten zu Zeitlinie 2 hinzu
        self.node2_1 = self.echo_prime.add_time_node(
            timeline_id=self.timeline2.id,
            description="Startknoten",
            probability=0.7,
            trigger_level=TriggerLevel.MEDIUM
        )
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        # Lösche Testzeitlinien
        if hasattr(self, 'timeline1') and self.timeline1:
            self.echo_prime.delete_timeline(self.timeline1.id)
        
        if hasattr(self, 'timeline2') and self.timeline2:
            self.echo_prime.delete_timeline(self.timeline2.id)
    
    def test_timeline_decision_context(self):
        """Test für die TimelineDecisionContext-Klasse"""
        # Erstelle Kontext
        context = TimelineDecisionContext(
            timeline=self.timeline1,
            current_node=self.node1_2,
            quantum_state=QuantumState.SUPERPOSITION
        )
        
        # Konvertiere in Q-LOGIK-Kontext
        qlogik_context = context.to_qlogik_context()
        
        # Überprüfe Kontext
        self.assertEqual(qlogik_context["timeline_id"], self.timeline1.id)
        self.assertEqual(qlogik_context["timeline_name"], self.timeline1.name)
        self.assertEqual(qlogik_context["node_id"], self.node1_2.id)
        self.assertEqual(qlogik_context["node_description"], self.node1_2.description)
        self.assertEqual(qlogik_context["quantum_state"], "SUPERPOSITION")
        
        # Überprüfe angepasste Entscheidungsfaktoren
        self.assertGreater(qlogik_context["decision_factors"]["uncertainty"], 0.7)
        self.assertGreater(qlogik_context["decision_factors"]["branching_factor"], 0.8)
    
    def test_evaluate_timeline(self):
        """Test für die evaluate_timeline-Funktion"""
        # Evaluiere Zeitlinie
        evaluation = self.integration.evaluate_timeline(self.timeline1.id)
        
        # Überprüfe Evaluation
        self.assertIsNotNone(evaluation)
        self.assertIn("decision", evaluation)
        self.assertIn("confidence", evaluation)
        self.assertIn("risk_assessment", evaluation)
    
    def test_recommend_timeline_action(self):
        """Test für die recommend_timeline_action-Funktion"""
        # Empfehle Aktion
        recommendation = self.integration.recommend_timeline_action(self.timeline1.id)
        
        # Überprüfe Empfehlung
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation["timeline_id"], self.timeline1.id)
        self.assertIn("actions", recommendation)
        self.assertGreater(len(recommendation["actions"]), 0)
        
        # Überprüfe Aktionsstruktur
        action = recommendation["actions"][0]
        self.assertIn("action", action)
        self.assertIn("priority", action)
        self.assertIn("description", action)
    
    def test_apply_quantum_decision_superposition(self):
        """Test für die apply_quantum_decision-Funktion mit Superposition"""
        # Wende Superposition an
        effect = self.integration.apply_quantum_decision(
            timeline_id=self.timeline1.id,
            effect_type="superposition"
        )
        
        # Überprüfe Effekt
        self.assertIsNotNone(effect)
        self.assertEqual(effect["effect_type"], "superposition")
        self.assertEqual(effect["timeline_id"], self.timeline1.id)
        self.assertIn("effect_id", effect)
        self.assertIn("branches", effect)
        
        # Überprüfe, ob der Effekt in der QTM_Modulator-Instanz gespeichert wurde
        self.assertIn(effect["effect_id"], self.echo_prime.qtm_modulator.quantum_effects)
    
    def test_apply_quantum_decision_auto(self):
        """Test für die apply_quantum_decision-Funktion mit automatischer Auswahl"""
        # Wende automatischen Effekt an
        effect = self.integration.apply_quantum_decision(
            timeline_id=self.timeline1.id,
            effect_type="auto"
        )
        
        # Überprüfe Effekt
        self.assertIsNotNone(effect)
        self.assertIn("effect_type", effect)
        self.assertEqual(effect["timeline_id"], self.timeline1.id)
        
        # Wenn kein Fehler aufgetreten ist, sollte ein effect_id vorhanden sein
        if "error" not in effect:
            self.assertIn("effect_id", effect)
    
    def test_resolve_timeline_conflict(self):
        """Test für die resolve_timeline_conflict-Funktion"""
        # Löse Konflikt
        resolution = self.integration.resolve_timeline_conflict(
            timeline_ids=[self.timeline1.id, self.timeline2.id],
            conflict_type="probability"
        )
        
        # Überprüfe Auflösung
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution["conflict_type"], "probability")
        self.assertIn(self.timeline1.id, resolution["timeline_ids"])
        self.assertIn(self.timeline2.id, resolution["timeline_ids"])
        self.assertIn("resolution_strategy", resolution)
        self.assertIn("actions", resolution)
        self.assertGreater(len(resolution["actions"]), 0)
        
        # Überprüfe Aktionsstruktur
        action = resolution["actions"][0]
        self.assertIn("action", action)
        self.assertIn("priority", action)
        self.assertIn("description", action)


class TestQLOGIKECHOPrimeIntegrationFunctions(unittest.TestCase):
    """Testklasse für die globalen Funktionen der Q-LOGIK ECHO-PRIME Integration"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        self.echo_prime = EchoPrimeController()
        
        # Erstelle Testzeitlinie
        self.timeline = self.echo_prime.create_timeline(
            name="Testzeitlinie",
            description="Eine Testzeitlinie für die Q-LOGIK Integration"
        )
        
        # Füge Knoten hinzu
        self.node1 = self.echo_prime.add_time_node(
            timeline_id=self.timeline.id,
            description="Startknoten",
            probability=0.9,
            trigger_level=TriggerLevel.LOW
        )
        
        self.node2 = self.echo_prime.add_time_node(
            timeline_id=self.timeline.id,
            description="Zweiter Knoten",
            parent_node_id=self.node1.id,
            probability=0.8,
            trigger_level=TriggerLevel.MEDIUM
        )
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        # Lösche Testzeitlinie
        if hasattr(self, 'timeline') and self.timeline:
            self.echo_prime.delete_timeline(self.timeline.id)
    
    def test_evaluate_timeline_function(self):
        """Test für die evaluate_timeline-Funktion"""
        # Evaluiere Zeitlinie
        evaluation = evaluate_timeline(self.timeline.id)
        
        # Überprüfe Evaluation
        self.assertIsNotNone(evaluation)
        self.assertIn("decision", evaluation)
    
    def test_recommend_timeline_action_function(self):
        """Test für die recommend_timeline_action-Funktion"""
        # Empfehle Aktion
        recommendation = recommend_timeline_action(self.timeline.id)
        
        # Überprüfe Empfehlung
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation["timeline_id"], self.timeline.id)
        self.assertIn("actions", recommendation)
    
    def test_apply_quantum_decision_function(self):
        """Test für die apply_quantum_decision-Funktion"""
        # Wende Quanteneffekt an
        effect = apply_quantum_decision(self.timeline.id, "superposition")
        
        # Überprüfe Effekt
        self.assertIsNotNone(effect)
        self.assertEqual(effect["effect_type"], "superposition")
        self.assertEqual(effect["timeline_id"], self.timeline.id)
    
    def test_resolve_timeline_conflict_function(self):
        """Test für die resolve_timeline_conflict-Funktion"""
        # Erstelle zweite Testzeitlinie
        timeline2 = self.echo_prime.create_timeline(
            name="Testzeitlinie 2",
            description="Eine weitere Testzeitlinie für die Q-LOGIK Integration"
        )
        
        try:
            # Löse Konflikt
            resolution = resolve_timeline_conflict(
                timeline_ids=[self.timeline.id, timeline2.id],
                conflict_type="probability"
            )
            
            # Überprüfe Auflösung
            self.assertIsNotNone(resolution)
            self.assertEqual(resolution["conflict_type"], "probability")
            self.assertIn("actions", resolution)
        finally:
            # Lösche zweite Testzeitlinie
            self.echo_prime.delete_timeline(timeline2.id)


if __name__ == "__main__":
    unittest.main()
