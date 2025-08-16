#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Tests für EventGenerator mit PRISM-Integration

Testfälle für die Integration des EventGenerators mit der PRISM-Engine.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import logging
import time
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.tests.simulation.test_event_generator_prism")

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importiere zu testende Module
from miso.simulation.event_generator import EventGenerator, EventType, SimulationEvent
from miso.simulation.prism_engine import PrismEngine


class TestEventGeneratorPrismIntegration(unittest.TestCase):
    """Tests für die Integration des EventGenerators mit der PRISM-Engine"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        # Mock-PRISM-Engine erstellen
        self.mock_prism_engine = MagicMock(spec=PrismEngine)
        
        # Konfiguriere Mock-Verhalten
        self.mock_prism_engine.time_scope = MagicMock()
        self.mock_prism_engine.time_scope.get_timeline_data.return_value = {
            "event_probabilities": {
                "TEMPORAL": 0.8,
                "CAUSAL": 0.7,
                "PROBABILISTIC": 0.6,
                "QUANTUM": 0.4,
                "PARADOX": 0.2,
                "SYSTEM": 0.9,
                "USER": 0.95,
                "EXTERNAL": 0.5
            }
        }
        
        self.mock_prism_engine.evaluate_probability_recommendation.return_value = {
            "adjusted_probability": 0.75
        }
        
        self.mock_prism_engine.matrix = MagicMock()
        self.mock_prism_engine.matrix.get_probability_at_coordinates.return_value = 0.65
        self.mock_prism_engine._determine_coordinates_for_stream = MagicMock()
        self.mock_prism_engine._determine_coordinates_for_stream.return_value = [1, 2, 3]
        
        # EventGenerator mit Mock-PRISM-Engine erstellen
        self.event_generator = EventGenerator(prism_engine=self.mock_prism_engine)
        
        # EventGenerator ohne PRISM-Engine für Vergleichstests
        self.event_generator_no_prism = EventGenerator()
        
        # Testdaten
        self.test_timeline_id = "test_timeline"
        self.test_node_ids = ["node1", "node2", "node3"]
    
    def test_initialization_with_prism_engine(self):
        """Test der Initialisierung mit PRISM-Engine"""
        self.assertIsNotNone(self.event_generator.prism_engine)
        self.assertEqual(self.event_generator.prism_engine, self.mock_prism_engine)
    
    def test_generate_events_for_timeline_with_prism(self):
        """Test der Ereignisgenerierung für eine Zeitlinie mit PRISM-Engine"""
        events = self.event_generator.generate_events_for_timeline(
            timeline_id=self.test_timeline_id,
            node_ids=self.test_node_ids,
            count=5
        )
        
        # Überprüfe, ob die richtige Anzahl von Ereignissen generiert wurde
        self.assertEqual(len(events), 5)
        
        # Überprüfe, ob die PRISM-Engine-Methoden aufgerufen wurden
        self.mock_prism_engine.time_scope.get_timeline_data.assert_called_with(self.test_timeline_id)
        
        # Überprüfe, ob alle Ereignisse die richtige Zeitlinien-ID haben
        for event in events:
            self.assertEqual(event.source_timeline_id, self.test_timeline_id)
    
    def test_calculate_event_probability_with_prism(self):
        """Test der Wahrscheinlichkeitsberechnung mit PRISM-Engine"""
        # Test mit evaluate_probability_recommendation
        probability = self.event_generator._calculate_event_probability(
            event_type=EventType.TEMPORAL,
            data={"test": "data"}
        )
        
        # Überprüfe, ob die PRISM-Engine-Methode aufgerufen wurde
        self.mock_prism_engine.evaluate_probability_recommendation.assert_called()
        
        # Überprüfe, ob die zurückgegebene Wahrscheinlichkeit korrekt ist
        self.assertEqual(probability, 0.75)
        
        # Setze evaluate_probability_recommendation auf None, um den alternativen Pfad zu testen
        self.mock_prism_engine.evaluate_probability_recommendation.return_value = None
        
        # Test mit matrix.get_probability_at_coordinates
        probability = self.event_generator._calculate_event_probability(
            event_type=EventType.CAUSAL,
            data={"test": "data"}
        )
        
        # Überprüfe, ob die PRISM-Engine-Methoden aufgerufen wurden
        self.mock_prism_engine._determine_coordinates_for_stream.assert_called()
        self.mock_prism_engine.matrix.get_probability_at_coordinates.assert_called_with([1, 2, 3])
        
        # Überprüfe, ob die zurückgegebene Wahrscheinlichkeit korrekt ist
        self.assertEqual(probability, 0.65)
    
    def test_fallback_without_prism_engine(self):
        """Test des Fallback-Verhaltens ohne PRISM-Engine"""
        # Generiere Ereignisse ohne PRISM-Engine
        events_no_prism = self.event_generator_no_prism.generate_events_for_timeline(
            timeline_id=self.test_timeline_id,
            node_ids=self.test_node_ids,
            count=5
        )
        
        # Überprüfe, ob die richtige Anzahl von Ereignissen generiert wurde
        self.assertEqual(len(events_no_prism), 5)
        
        # Überprüfe, ob alle Ereignisse eine Wahrscheinlichkeit haben
        for event in events_no_prism:
            self.assertIsNotNone(event.probability)
            self.assertTrue(0 <= event.probability <= 1)
    
    def test_exception_handling(self):
        """Test der Ausnahmebehandlung bei PRISM-Engine-Fehlern"""
        # Konfiguriere Mock, um eine Ausnahme auszulösen
        self.mock_prism_engine.time_scope.get_timeline_data.side_effect = Exception("Test-Ausnahme")
        
        # Ereignisse sollten trotz Ausnahme generiert werden (Fallback)
        events = self.event_generator.generate_events_for_timeline(
            timeline_id=self.test_timeline_id,
            node_ids=self.test_node_ids,
            count=5
        )
        
        # Überprüfe, ob die richtige Anzahl von Ereignissen generiert wurde
        self.assertEqual(len(events), 5)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        self.event_generator = None
        self.event_generator_no_prism = None
        self.mock_prism_engine = None


if __name__ == "__main__":
    unittest.main()
