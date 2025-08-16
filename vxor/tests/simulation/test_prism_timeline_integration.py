#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Tests für PRISM-Timeline-Integration

Testfälle für die Integration der PRISM-Engine mit der Timeline-Klasse.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.tests.simulation.test_prism_timeline_integration")

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importiere zu testende Module
from miso.simulation.prism_engine import PrismEngine
from miso.timeline.echo_prime import Timeline, TimeNode, TemporalEvent, TriggerLevel, TimelineType


class TestPrismTimelineIntegration(unittest.TestCase):
    """Tests für die Integration der PRISM-Engine mit der Timeline-Klasse"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        # Erstelle PRISM-Engine
        self.prism_engine = PrismEngine()
        
        # Erstelle Timeline
        from miso.timeline import TimelineType
        self.timeline = Timeline(
            id="test_timeline",
            type=TimelineType.MAIN,
            name="Test Timeline",
            description="Eine Testzeitlinie für die PRISM-Integration"
        )
        
        # Erstelle TimeNodes
        self.nodes = []
        from miso.timeline import TriggerLevel
        for i in range(5):
            node = TimeNode(
                id=f"node_{i}",
                timestamp=time.time() + i * 3600,  # Stündliche Intervalle
                description=f"Test Node {i}",
                trigger_level=TriggerLevel.MEDIUM,
                probability=0.8 - (i * 0.1),  # Abnehmende Wahrscheinlichkeit
                metadata={
                    "value": i * 10,
                    "label": f"Test Node {i}"
                }
            )
            self.nodes.append(node)
            self.timeline.add_node(node)
        
        # Verbinde Knoten
        for i in range(4):
            self.timeline.connect_nodes(
                source_id=self.nodes[i].id,
                target_id=self.nodes[i+1].id,
                metadata={"probability": 0.9 - (i * 0.1)}
            )
    
    def test_register_timeline_with_prism(self):
        """Test der Registrierung einer Timeline bei der PRISM-Engine"""
        # Registriere Timeline bei der PRISM-Engine
        self.prism_engine.register_timeline(self.timeline)
        
        # Überprüfe, ob die Timeline in der PRISM-Engine gespeichert wurde
        self.assertIn(self.timeline.id, self.prism_engine.get_registered_timeline_ids())
    
    def test_timeline_probability_calculation(self):
        """Test der Wahrscheinlichkeitsberechnung für eine Timeline"""
        # Registriere Timeline bei der PRISM-Engine
        self.prism_engine.register_timeline(self.timeline)
        
        # Berechne Wahrscheinlichkeit der Timeline
        probability = self.prism_engine.calculate_timeline_probability(self.timeline.id)
        
        # Überprüfe, ob eine gültige Wahrscheinlichkeit zurückgegeben wurde
        self.assertIsNotNone(probability)
        self.assertTrue(0 <= probability <= 1)
    
    def test_node_probability_calculation(self):
        """Test der Wahrscheinlichkeitsberechnung für einen TimeNode"""
        # Registriere Timeline bei der PRISM-Engine
        self.prism_engine.register_timeline(self.timeline)
        
        # Berechne Wahrscheinlichkeit eines Knotens
        node_id = self.nodes[2].id
        probability = self.prism_engine.calculate_node_probability(
            timeline_id=self.timeline.id,
            node_id=node_id
        )
        
        # Überprüfe, ob eine gültige Wahrscheinlichkeit zurückgegeben wurde
        self.assertIsNotNone(probability)
        self.assertTrue(0 <= probability <= 1)
    
    def test_timeline_stability_analysis(self):
        """Test der Stabilitätsanalyse für eine Timeline"""
        # Registriere Timeline bei der PRISM-Engine
        self.prism_engine.register_timeline(self.timeline)
        
        # Führe Stabilitätsanalyse durch
        stability = self.prism_engine.analyze_timeline_stability(self.timeline.id)
        
        # Überprüfe, ob ein gültiges Stabilitätsergebnis zurückgegeben wurde
        self.assertIsNotNone(stability)
        self.assertIsInstance(stability, dict)
        
        # Überprüfe, ob die Stabilitätsanalyse die erwarteten Metriken enthält
        self.assertIn("stability_index", stability)
        self.assertIn("paradox_risk", stability)
        self.assertIn("entropy", stability)
        
        # Überprüfe, ob die Werte im gültigen Bereich liegen
        self.assertTrue(0 <= stability["stability_index"] <= 1)
        self.assertTrue(0 <= stability["paradox_risk"] <= 1)
        self.assertTrue(0 <= stability["entropy"] <= 1)
    
    def test_timeline_fork(self):
        """Test der Zeitlinienverzweigung mit der PRISM-Engine"""
        # Registriere Timeline bei der PRISM-Engine
        self.prism_engine.register_timeline(self.timeline)
        
        # Erstelle eine Verzweigung der Timeline
        fork_node_id = self.nodes[1].id
        fork_result = self.prism_engine.fork_timeline(
            source_timeline_id=self.timeline.id,
            fork_node_id=fork_node_id,
            variation_factor=0.7
        )
        
        # Überprüfe, ob die Verzweigung erfolgreich war
        self.assertIsNotNone(fork_result)
        self.assertIn("forked_timeline_id", fork_result)
        self.assertIn("probability", fork_result)
        
        # Überprüfe, ob die verzweigte Timeline in der PRISM-Engine registriert wurde
        forked_timeline_id = fork_result["forked_timeline_id"]
        self.assertIn(forked_timeline_id, self.prism_engine.get_registered_timeline_ids())
    
    def test_timeline_merge(self):
        """Test der Zeitlinienzusammenführung mit der PRISM-Engine"""
        # Registriere Timeline bei der PRISM-Engine
        self.prism_engine.register_timeline(self.timeline)
        
        # Erstelle eine Verzweigung der Timeline
        fork_result = self.prism_engine.fork_timeline(
            source_timeline_id=self.timeline.id,
            fork_node_id=self.nodes[1].id,
            variation_factor=0.7
        )
        forked_timeline_id = fork_result["forked_timeline_id"]
        
        # Führe die Zeitlinien zusammen
        merge_result = self.prism_engine.merge_timelines(
            source_timeline_id=forked_timeline_id,
            target_timeline_id=self.timeline.id,
            merge_point_source=self.nodes[3].id,  # Knoten in der verzweigten Timeline
            merge_point_target=self.nodes[4].id   # Knoten in der Ursprungszeitlinie
        )
        
        # Überprüfe, ob die Zusammenführung erfolgreich war
        self.assertIsNotNone(merge_result)
        self.assertIn("merged_timeline_id", merge_result)
        self.assertIn("probability", merge_result)
        self.assertIn("stability", merge_result)
        
        # Überprüfe, ob die zusammengeführte Timeline in der PRISM-Engine registriert wurde
        merged_timeline_id = merge_result["merged_timeline_id"]
        self.assertIn(merged_timeline_id, self.prism_engine.get_registered_timeline_ids())
    
    def test_paradox_detection(self):
        """Test der Paradoxerkennung mit der PRISM-Engine"""
        # Registriere Timeline bei der PRISM-Engine
        self.prism_engine.register_timeline(self.timeline)
        
        # Erstelle einen paradoxen Knoten (Verbindung zu einem früheren Knoten)
        paradox_node = TimeNode(
            id="paradox_node",
            timestamp=time.time() + 6 * 3600,  # Nach dem letzten Knoten
            description="Paradox Node",
            trigger_level=TriggerLevel.HIGH,
            probability=0.3
        )
        paradox_node.metadata = {"value": 100, "label": "Paradox Node"}
        self.timeline.add_node(paradox_node)
        
        # Erstelle eine paradoxe Verbindung (von einem späteren zu einem früheren Knoten)
        self.timeline.connect_nodes(
            source_id=paradox_node.id,
            target_id=self.nodes[0].id,  # Verbindung zum ersten Knoten (zeitlich früher)
            metadata={"probability": 0.2, "is_paradox": True}
        )
        
        # Führe Paradoxerkennung durch
        paradox_result = self.prism_engine.detect_paradoxes(self.timeline.id)
        
        # Überprüfe, ob Paradoxe erkannt wurden
        self.assertIsNotNone(paradox_result)
        self.assertIn("paradoxes", paradox_result)
        self.assertGreater(len(paradox_result["paradoxes"]), 0)
        
        # Überprüfe, ob die paradoxe Verbindung erkannt wurde
        paradoxes = paradox_result["paradoxes"]
        found_paradox = False
        for paradox in paradoxes:
            if paradox.get("source_id") == paradox_node.id and paradox.get("target_id") == self.nodes[0].id:
                found_paradox = True
                break
        
        self.assertTrue(found_paradox, "Die paradoxe Verbindung wurde nicht erkannt")
    
    def test_timeline_events_generation(self):
        """Test der Ereignisgenerierung für eine Timeline mit der PRISM-Engine"""
        # Registriere Timeline bei der PRISM-Engine
        self.prism_engine.register_timeline(self.timeline)
        
        # Generiere Ereignisse für die Timeline
        events = self.prism_engine.generate_timeline_events(
            timeline_id=self.timeline.id,
            count=3
        )
        
        # Überprüfe, ob Ereignisse generiert wurden
        self.assertIsNotNone(events)
        self.assertGreater(len(events), 0)
        
        # Überprüfe, ob die Ereignisse gültig sind
        for event in events:
            self.assertIsInstance(event, dict)
            self.assertIn('id', event)
            self.assertIn('source_timeline_id', event)
        
    def test_timeline_simulation(self):
        """Test der Zeitliniensimulation mit der PRISM-Engine"""
        # Registriere Timeline bei der PRISM-Engine
        self.prism_engine.register_timeline(self.timeline)
        
        # Führe Simulation durch
        simulation_result = self.prism_engine.simulate_timeline(
            timeline_id=self.timeline.id,
            steps=5,
            variation_factor=0.2
        )
        
        # Überprüfe, ob die Simulation erfolgreich war
        self.assertIsNotNone(simulation_result)
        self.assertIn("initial_state", simulation_result)
        self.assertIn("steps", simulation_result)
        self.assertIn("final_state", simulation_result)
        
        # Überprüfe, ob Simulationsschritte generiert wurden
        self.assertGreater(len(simulation_result["steps"]), 0)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        # Entferne alle registrierten Zeitlinien
        for timeline_id in self.prism_engine.get_registered_timeline_ids():
            self.prism_engine.unregister_timeline(timeline_id)
        
        self.prism_engine = None
        self.timeline = None
        self.nodes = []


if __name__ == "__main__":
    unittest.main()
