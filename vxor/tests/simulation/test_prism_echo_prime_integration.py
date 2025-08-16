#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Tests für PRISM-ECHO-PRIME Integration

Umfassende Testfälle für die Integration zwischen PRISM-Engine und ECHO-PRIME.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import logging
import time
from unittest.mock import MagicMock, patch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.tests.simulation.test_prism_echo_prime_integration")

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importiere zu testende Module
from miso.simulation.prism_echo_prime_integration import PRISMECHOPrimeIntegration, TimelineSimulationContext

# Definiere Mock-Klassen für ECHO-PRIME
class MockTimeNode:
    def __init__(self, node_id, description="Test Node", parent_node_id=None, probability=0.8, trigger_level=None):
        self.id = node_id
        self.description = description
        self.parent_node_id = parent_node_id
        self.probability = probability
        self.trigger_level = trigger_level or MockTriggerLevel.MEDIUM
        self.child_node_ids = []
        self.timestamp = time.time()
        self.metadata = {}

class MockTimeline:
    def __init__(self, timeline_id, name="Test Timeline", description="Test Description", probability=0.8):
        self.id = timeline_id
        self.name = name
        self.description = description
        self.probability = probability
        self.nodes = {}
        self.type = MockTimelineType.STANDARD
        self.metadata = {}

class MockTriggerLevel:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    
    def __init__(self, value):
        self.value = value

class MockTimelineType:
    STANDARD = "STANDARD"
    ALTERNATIVE = "ALTERNATIVE"
    HYPOTHETICAL = "HYPOTHETICAL"
    
    def __init__(self, value):
        self.value = value

class MockEchoPrimeController:
    def __init__(self, config_path=None):
        self.timelines = {}
        self.next_node_id = 1
    
    def get_timeline(self, timeline_id):
        return self.timelines.get(timeline_id)
    
    def create_timeline(self, name, description=None):
        timeline_id = f"timeline_{len(self.timelines) + 1}"
        timeline = MockTimeline(timeline_id, name, description)
        self.timelines[timeline_id] = timeline
        return timeline
    
    def add_time_node(self, timeline_id, description, parent_node_id=None, probability=0.8, trigger_level=None, metadata=None):
        if timeline_id not in self.timelines:
            return None
        
        timeline = self.timelines[timeline_id]
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1
        
        node = MockTimeNode(node_id, description, parent_node_id, probability, trigger_level)
        if metadata:
            node.metadata = metadata
        
        timeline.nodes[node_id] = node
        
        # Aktualisiere Eltern-Kind-Beziehung
        if parent_node_id and parent_node_id in timeline.nodes:
            parent_node = timeline.nodes[parent_node_id]
            parent_node.child_node_ids.append(node_id)
        
        return node
    
    def update_timeline(self, timeline_id, updates):
        if timeline_id not in self.timelines:
            return False
        
        timeline = self.timelines[timeline_id]
        
        if "metadata" in updates:
            timeline.metadata = updates["metadata"]
        
        return True


# Patch für ECHO-PRIME Imports
patch_echo_prime = patch("miso.simulation.prism_echo_prime_integration.HAS_ECHO_PRIME", True)
patch_echo_prime_controller = patch("miso.simulation.prism_echo_prime_integration.EchoPrimeController", MockEchoPrimeController)
patch_trigger_level = patch("miso.simulation.prism_echo_prime_integration.TriggerLevel", MockTriggerLevel)
patch_timeline_type = patch("miso.simulation.prism_echo_prime_integration.TimelineType", MockTimelineType)
patch_timeline = patch("miso.simulation.prism_echo_prime_integration.Timeline", MockTimeline)
patch_time_node = patch("miso.simulation.prism_echo_prime_integration.TimeNode", MockTimeNode)


@patch_echo_prime
@patch_echo_prime_controller
@patch_trigger_level
@patch_timeline_type
@patch_timeline
@patch_time_node
class TestPRISMECHOPrimeIntegration(unittest.TestCase):
    """Tests für die PRISMECHOPrimeIntegration-Klasse"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        self.integration = PRISMECHOPrimeIntegration()
        
        # Erstelle Testzeitlinie
        self.echo_prime = self.integration.echo_prime
        self.timeline = self.echo_prime.create_timeline(
            name="Testzeitlinie",
            description="Eine Testzeitlinie für die PRISM-ECHO-PRIME Integration"
        )
        
        # Füge Knoten hinzu
        self.node1 = self.echo_prime.add_time_node(
            timeline_id=self.timeline.id,
            description="Startknoten",
            probability=0.9,
            trigger_level=MockTriggerLevel.LOW
        )
        
        self.node2 = self.echo_prime.add_time_node(
            timeline_id=self.timeline.id,
            description="Zweiter Knoten",
            parent_node_id=self.node1.id,
            probability=0.8,
            trigger_level=MockTriggerLevel.MEDIUM
        )
    
    def test_initialization(self):
        """Test der Initialisierung"""
        self.assertIsNotNone(self.integration)
        self.assertIsNotNone(self.integration.prism_engine)
        self.assertIsNotNone(self.integration.echo_prime)
        self.assertIsInstance(self.integration.simulation_cache, dict)
    
    def test_timeline_to_prism_state(self):
        """Test der Konvertierung von Zeitlinie zu PRISM-Zustand"""
        state = self.integration._timeline_to_prism_state(self.timeline, self.node2)
        
        # Überprüfe Zustandsstruktur
        self.assertIsInstance(state, dict)
        self.assertEqual(state["timeline_id"], self.timeline.id)
        self.assertEqual(state["timeline_name"], self.timeline.name)
        self.assertEqual(state["current_node_id"], self.node2.id)
        self.assertEqual(state["current_node_description"], self.node2.description)
        self.assertEqual(state["current_node_probability"], self.node2.probability)
        self.assertEqual(state["parent_node_id"], self.node2.parent_node_id)
    
    def test_generate_timeline_variations(self):
        """Test der Generierung von Zeitlinienvariationen"""
        variations = self.integration._generate_timeline_variations(self.timeline, self.node2)
        
        # Überprüfe Variationsstruktur
        self.assertIsInstance(variations, list)
        self.assertGreater(len(variations), 0)
        
        # Überprüfe erste Variation
        first_variation = variations[0]
        self.assertIn("factor", first_variation)
        self.assertIn("impact", first_variation)
    
    def test_simulate_timeline(self):
        """Test der Zeitliniensimulation"""
        from miso.simulation.time_scope import TimeScope
        
        # Patche die simulate_reality_fork-Methode
        with patch.object(self.integration.prism_engine, 'simulate_reality_fork') as mock_simulate:
            # Setze Rückgabewert für simulate_reality_fork
            mock_simulate.return_value = {
                "alternative_realities": [
                    {"timeline_id": self.timeline.id, "timeline_probability": 0.8},
                    {"timeline_id": self.timeline.id, "timeline_probability": 0.9, "event_description": "Alternative 1"}
                ]
            }
            
            # Führe Simulation durch
            result = self.integration.simulate_timeline(self.timeline.id)
            
            # Überprüfe, ob simulate_reality_fork aufgerufen wurde
            mock_simulate.assert_called_once()
            
            # Überprüfe Ergebnisstruktur
            self.assertIsInstance(result, dict)
            self.assertIn("simulation_id", result)
            self.assertEqual(result["timeline_id"], self.timeline.id)
            self.assertEqual(result["timeline_name"], self.timeline.name)
            self.assertIn("alternative_timelines", result)
            self.assertIn("probability_map", result)
            self.assertIn("recommendations", result)
            
            # Überprüfe, ob Simulation im Cache gespeichert wurde
            self.assertIn(result["simulation_id"], self.integration.simulation_cache)
    
    def test_generate_alternative_timelines(self):
        """Test der Generierung alternativer Zeitlinien"""
        # Simulationsergebnis mit alternativen Realitäten
        simulation_result = {
            "alternative_realities": [
                {"timeline_id": self.timeline.id, "timeline_probability": 0.8},
                {"timeline_id": self.timeline.id, "timeline_probability": 0.9, "event_description": "Alternative 1"},
                {"timeline_id": self.timeline.id, "timeline_probability": 0.7, "event_description": "Alternative 2"}
            ]
        }
        
        # Generiere alternative Zeitlinien
        alternative_timelines = self.integration._generate_alternative_timelines(
            self.timeline, self.node2, simulation_result
        )
        
        # Überprüfe Ergebnis
        self.assertIsInstance(alternative_timelines, list)
        self.assertEqual(len(alternative_timelines), 2)  # Zwei Alternativen (ohne Basis)
        
        # Überprüfe erste alternative Zeitlinie
        alt_timeline = alternative_timelines[0]
        self.assertIsInstance(alt_timeline, MockTimeline)
        self.assertIn("simulation_based", alt_timeline.metadata)
        self.assertTrue(alt_timeline.metadata["simulation_based"])
    
    def test_calculate_probability_map(self):
        """Test der Berechnung der Wahrscheinlichkeitskarte"""
        # Simulationsergebnis mit alternativen Realitäten
        simulation_result = {
            "alternative_realities": [
                {"timeline_id": self.timeline.id, "timeline_probability": 0.5},
                {"timeline_id": self.timeline.id, "timeline_probability": 0.3, "event_description": "Alternative 1"},
                {"timeline_id": self.timeline.id, "timeline_probability": 0.2, "event_description": "Alternative 2"}
            ]
        }
        
        # Berechne Wahrscheinlichkeitskarte
        probability_map = self.integration._calculate_probability_map(simulation_result)
        
        # Überprüfe Ergebnis
        self.assertIsInstance(probability_map, dict)
        self.assertEqual(len(probability_map), 3)
        
        # Überprüfe Wahrscheinlichkeiten
        self.assertIn("baseline", probability_map)
        self.assertIn("Alternative 1", probability_map)
        self.assertIn("Alternative 2", probability_map)
        
        # Überprüfe, ob Summe der Wahrscheinlichkeiten ungefähr 1 ist
        total_probability = sum(probability_map.values())
        self.assertAlmostEqual(total_probability, 1.0, places=5)
    
    def test_generate_recommendations(self):
        """Test der Generierung von Handlungsempfehlungen"""
        # Simulationsergebnis mit alternativen Realitäten
        simulation_result = {
            "alternative_realities": [
                {"timeline_id": self.timeline.id, "timeline_probability": 0.5, "benefit": 0.6, "risk": 0.3},
                {"timeline_id": self.timeline.id, "timeline_probability": 0.3, "event_description": "Alternative 1", "benefit": 0.8, "risk": 0.2},
                {"timeline_id": self.timeline.id, "timeline_probability": 0.2, "event_description": "Alternative 2", "benefit": 0.4, "risk": 0.7}
            ]
        }
        
        # Wahrscheinlichkeitskarte
        probability_map = {
            "baseline": 0.5,
            "Alternative 1": 0.3,
            "Alternative 2": 0.2
        }
        
        # Generiere Empfehlungen
        recommendations = self.integration._generate_recommendations(simulation_result, probability_map)
        
        # Überprüfe Ergebnis
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Überprüfe erste Empfehlung
        first_recommendation = recommendations[0]
        self.assertIn("type", first_recommendation)
        self.assertIn("description", first_recommendation)
        self.assertIn("probability", first_recommendation)
        self.assertIn("action", first_recommendation)
    
    def test_analyze_timeline_probabilities(self):
        """Test der Analyse von Zeitlinienwahrscheinlichkeiten"""
        # Patche die evaluate_probability_recommendation-Methode
        with patch.object(self.integration.prism_engine, 'evaluate_probability_recommendation') as mock_evaluate:
            # Setze Rückgabewert für evaluate_probability_recommendation
            mock_evaluate.return_value = {
                "recommendation": "Die Zeitlinie ist stabil",
                "confidence": 0.85
            }
            
            # Führe Analyse durch
            result = self.integration.analyze_timeline_probabilities(self.timeline.id)
            
            # Überprüfe, ob evaluate_probability_recommendation aufgerufen wurde
            mock_evaluate.assert_called_once()
            
            # Überprüfe Ergebnisstruktur
            self.assertIsInstance(result, dict)
            self.assertEqual(result["timeline_id"], self.timeline.id)
            self.assertEqual(result["timeline_name"], self.timeline.name)
            self.assertIn("node_probabilities", result)
            self.assertIn("path_probabilities", result)
            self.assertIn("probability_assessment", result)
    
    def test_get_simulation_result(self):
        """Test des Abrufs von Simulationsergebnissen"""
        # Erstelle Simulationskontext
        from miso.simulation.time_scope import TimeScope
        context = TimelineSimulationContext(self.timeline, self.node2, simulation_scope=TimeScope.MEDIUM)
        context.results = {"test": "result"}
        context.alternative_timelines = [self.timeline]
        context.probability_map = {"baseline": 1.0}
        
        # Speichere im Cache
        self.integration.simulation_cache[context.simulation_id] = context
        
        # Rufe Ergebnis ab
        result = self.integration.get_simulation_result(context.simulation_id)
        
        # Überprüfe Ergebnis
        self.assertIsInstance(result, dict)
        self.assertEqual(result["simulation_id"], context.simulation_id)
        self.assertEqual(result["timeline_id"], self.timeline.id)
        self.assertEqual(result["results"], context.results)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        self.integration = None
        self.echo_prime = None
        self.timeline = None
        self.node1 = None
        self.node2 = None


class TestTimelineSimulationContext(unittest.TestCase):
    """Tests für die TimelineSimulationContext-Klasse"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        from miso.simulation.time_scope import TimeScope
        
        # Erstelle Mock-Objekte
        self.timeline = MockTimeline("test_timeline", "Test Timeline")
        self.node = MockTimeNode("test_node", "Test Node")
        
        # Erstelle Kontext
        self.context = TimelineSimulationContext(
            timeline=self.timeline,
            current_node=self.node,
            simulation_steps=50,
            simulation_scope=TimeScope.MEDIUM
        )
    
    def test_initialization(self):
        """Test der Initialisierung"""
        self.assertIsNotNone(self.context)
        self.assertEqual(self.context.timeline, self.timeline)
        self.assertEqual(self.context.current_node, self.node)
        self.assertEqual(self.context.simulation_steps, 50)
        self.assertIsInstance(self.context.simulation_id, str)
        self.assertIsInstance(self.context.creation_time, float)
        self.assertIsInstance(self.context.results, dict)
        self.assertIsInstance(self.context.alternative_timelines, list)
        self.assertIsInstance(self.context.probability_map, dict)
    
    def test_to_dict(self):
        """Test der Konvertierung in ein Dictionary"""
        result = self.context.to_dict()
        
        # Überprüfe Ergebnisstruktur
        self.assertIsInstance(result, dict)
        self.assertEqual(result["simulation_id"], self.context.simulation_id)
        self.assertEqual(result["timeline_id"], self.timeline.id)
        self.assertEqual(result["timeline_name"], self.timeline.name)
        self.assertEqual(result["current_node_id"], self.node.id)
        self.assertEqual(result["simulation_steps"], 50)
        self.assertIn("simulation_scope", result)
        self.assertIn("creation_time", result)
        self.assertIn("has_results", result)
        self.assertIn("alternative_timeline_count", result)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        self.context = None
        self.timeline = None
        self.node = None


if __name__ == "__main__":
    unittest.main()
