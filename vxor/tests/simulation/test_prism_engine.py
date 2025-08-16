#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Tests für PRISM-Engine

Umfassende Testfälle für die PRISM-Engine und ihre Komponenten.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional
import json

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.tests.simulation.test_prism_engine")

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importiere zu testende Module
from miso.simulation.prism_engine import PrismEngine, RealityForker
from miso.simulation.prism_matrix import PrismMatrix
from miso.simulation.time_scope import TimeScopeUnit, TimeScope
from miso.simulation.predictive_stream import PredictiveStreamAnalyzer
from miso.simulation.pattern_dissonance import PatternDissonanceScanner


class TestPrismEngine(unittest.TestCase):
    """Tests für die PrismEngine-Klasse"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        self.prism_engine = PrismEngine()
        
        # Beispielzustand für Tests
        self.test_state = {
            "timeline_id": "test_timeline",
            "timeline_name": "Test Timeline",
            "timeline_probability": 0.8,
            "node_count": 5,
            "current_timestamp": time.time()
        }
        
        # Beispielvariationen für Tests
        self.test_variations = [
            {"factor": "timeline_probability", "impact": 0.1},
            {"factor": "timeline_probability", "impact": -0.1}
        ]
    
    def test_initialization(self):
        """Test der Initialisierung"""
        self.assertIsNotNone(self.prism_engine)
        self.assertIsNotNone(self.prism_engine.reality_forker)
        self.assertIsNotNone(self.prism_engine.prism_matrix)
        
        # Überprüfe, ob optionale Komponenten korrekt initialisiert wurden
        if hasattr(self.prism_engine, 'time_scope_unit'):
            self.assertIsNotNone(self.prism_engine.time_scope_unit)
        
        if hasattr(self.prism_engine, 'predictive_analyzer'):
            self.assertIsNotNone(self.prism_engine.predictive_analyzer)
        
        if hasattr(self.prism_engine, 'dissonance_scanner'):
            self.assertIsNotNone(self.prism_engine.dissonance_scanner)
    
    def test_simulate_reality_fork(self):
        """Test der Realitätsverzweigung"""
        result = self.prism_engine.simulate_reality_fork(
            current_state=self.test_state,
            variations=self.test_variations,
            steps=10
        )
        
        # Überprüfe Ergebnisstruktur
        self.assertIn("alternative_realities", result)
        self.assertIsInstance(result["alternative_realities"], list)
        self.assertGreaterEqual(len(result["alternative_realities"]), 2)  # Mindestens Basis + 1 Alternative
        
        # Überprüfe erste Alternative (Basis)
        base_reality = result["alternative_realities"][0]
        self.assertEqual(base_reality["timeline_id"], self.test_state["timeline_id"])
        self.assertEqual(base_reality["timeline_name"], self.test_state["timeline_name"])
        
        # Überprüfe zweite Alternative (erste Variation)
        alt_reality = result["alternative_realities"][1]
        self.assertEqual(alt_reality["timeline_id"], self.test_state["timeline_id"])
        self.assertNotEqual(alt_reality["timeline_probability"], self.test_state["timeline_probability"])
    
    def test_evaluate_probability_recommendation(self):
        """Test der Wahrscheinlichkeitsbewertung"""
        # Test mit hoher Wahrscheinlichkeit
        high_prob_result = self.prism_engine.evaluate_probability_recommendation(0.9)
        self.assertIn("recommendation", high_prob_result)
        self.assertIn("confidence", high_prob_result)
        self.assertGreaterEqual(high_prob_result["confidence"], 0.8)
        
        # Test mit mittlerer Wahrscheinlichkeit
        med_prob_result = self.prism_engine.evaluate_probability_recommendation(0.5)
        self.assertIn("recommendation", med_prob_result)
        self.assertIn("confidence", med_prob_result)
        
        # Test mit niedriger Wahrscheinlichkeit
        low_prob_result = self.prism_engine.evaluate_probability_recommendation(0.1)
        self.assertIn("recommendation", low_prob_result)
        self.assertIn("confidence", low_prob_result)
        self.assertLessEqual(low_prob_result["confidence"], 0.3)
    
    def test_analyze_timeline_stability(self):
        """Test der Zeitlinienstabilitätsanalyse"""
        stability_result = self.prism_engine.analyze_timeline_stability(self.test_state)
        
        self.assertIn("stability_score", stability_result)
        self.assertIn("factors", stability_result)
        self.assertIn("recommendations", stability_result)
        
        # Überprüfe Stabilitätswert
        self.assertGreaterEqual(stability_result["stability_score"], 0.0)
        self.assertLessEqual(stability_result["stability_score"], 1.0)
    
    def test_generate_probability_matrix(self):
        """Test der Wahrscheinlichkeitsmatrixgenerierung"""
        matrix_result = self.prism_engine.generate_probability_matrix(
            self.test_state,
            dimensions=["timeline_probability", "node_count"]
        )
        
        self.assertIn("matrix", matrix_result)
        self.assertIn("dimensions", matrix_result)
        self.assertEqual(len(matrix_result["dimensions"]), 2)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        self.prism_engine = None


class TestRealityForker(unittest.TestCase):
    """Tests für die RealityForker-Klasse"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        self.reality_forker = RealityForker()
        
        # Beispielzustand für Tests
        self.current_state = {
            "timeline_id": "test_timeline",
            "timeline_probability": 0.8,
            "event_count": 5
        }
        
        # Beispielvariationen für Tests
        self.variations = [
            {"factor": "timeline_probability", "impact": 0.1},
            {"factor": "event_count", "impact": 2},
            {"factor": "nonexistent_factor", "impact": 0.5}  # Sollte ignoriert werden
        ]
    
    def test_fork_reality(self):
        """Test der Realitätsverzweigung"""
        alternative_realities = self.reality_forker.fork_reality(
            current_state=self.current_state,
            variations=self.variations
        )
        
        # Überprüfe Ergebnisstruktur
        self.assertIsInstance(alternative_realities, list)
        self.assertGreaterEqual(len(alternative_realities), len(self.variations) + 1)  # +1 für Basisrealität
        
        # Überprüfe Basisrealität
        base_reality = alternative_realities[0]
        self.assertEqual(base_reality["timeline_id"], self.current_state["timeline_id"])
        self.assertEqual(base_reality["timeline_probability"], self.current_state["timeline_probability"])
        
        # Überprüfe erste Variation
        first_variation = alternative_realities[1]
        self.assertEqual(first_variation["timeline_id"], self.current_state["timeline_id"])
        self.assertAlmostEqual(
            first_variation["timeline_probability"], 
            self.current_state["timeline_probability"] + self.variations[0]["impact"],
            places=5
        )
        
        # Überprüfe zweite Variation
        second_variation = alternative_realities[2]
        self.assertEqual(second_variation["timeline_id"], self.current_state["timeline_id"])
        self.assertEqual(
            second_variation["event_count"], 
            self.current_state["event_count"] + self.variations[1]["impact"]
        )
    
    def test_apply_variation(self):
        """Test der Variationsanwendung"""
        # Teste positive Variation
        positive_variation = {"factor": "timeline_probability", "impact": 0.1}
        positive_result = self.reality_forker._apply_variation(self.current_state, positive_variation)
        self.assertAlmostEqual(
            positive_result["timeline_probability"], 
            self.current_state["timeline_probability"] + 0.1,
            places=5
        )
        
        # Teste negative Variation
        negative_variation = {"factor": "timeline_probability", "impact": -0.2}
        negative_result = self.reality_forker._apply_variation(self.current_state, negative_variation)
        self.assertAlmostEqual(
            negative_result["timeline_probability"], 
            self.current_state["timeline_probability"] - 0.2,
            places=5
        )
        
        # Teste Variation mit nicht existierendem Faktor
        invalid_variation = {"factor": "nonexistent_factor", "impact": 0.5}
        invalid_result = self.reality_forker._apply_variation(self.current_state, invalid_variation)
        self.assertEqual(invalid_result, self.current_state)  # Sollte unverändert sein
    
    def test_generate_event_description(self):
        """Test der Ereignisbeschreibungsgenerierung"""
        # Teste mit positiver Auswirkung
        positive_variation = {"factor": "timeline_probability", "impact": 0.2}
        positive_description = self.reality_forker._generate_event_description(positive_variation)
        self.assertIsInstance(positive_description, str)
        self.assertGreater(len(positive_description), 0)
        
        # Teste mit negativer Auswirkung
        negative_variation = {"factor": "timeline_probability", "impact": -0.3}
        negative_description = self.reality_forker._generate_event_description(negative_variation)
        self.assertIsInstance(negative_description, str)
        self.assertGreater(len(negative_description), 0)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        self.reality_forker = None


class TestPrismMatrix(unittest.TestCase):
    """Tests für die PrismMatrix-Klasse"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        self.prism_matrix = PrismMatrix()
        
        # Beispieldaten für Tests
        self.test_data = {
            "dimension1": [0.1, 0.2, 0.3],
            "dimension2": [1, 2, 3, 4]
        }
    
    def test_create_matrix(self):
        """Test der Matrixerstellung"""
        matrix = self.prism_matrix.create_matrix(self.test_data)
        
        # Überprüfe Matrixstruktur
        self.assertIsInstance(matrix, dict)
        self.assertIn("matrix", matrix)
        self.assertIn("dimensions", matrix)
        self.assertIn("shape", matrix)
        
        # Überprüfe Dimensionen
        self.assertEqual(len(matrix["dimensions"]), len(self.test_data))
        for dim in self.test_data.keys():
            self.assertIn(dim, matrix["dimensions"])
        
        # Überprüfe Form
        expected_shape = tuple(len(values) for values in self.test_data.values())
        self.assertEqual(matrix["shape"], expected_shape)
        
        # Überprüfe Matrixdaten
        self.assertIsInstance(matrix["matrix"], np.ndarray)
        self.assertEqual(matrix["matrix"].shape, expected_shape)
    
    def test_calculate_probability_distribution(self):
        """Test der Wahrscheinlichkeitsverteilungsberechnung"""
        matrix = self.prism_matrix.create_matrix(self.test_data)
        distribution = self.prism_matrix.calculate_probability_distribution(matrix)
        
        # Überprüfe Verteilungsstruktur
        self.assertIsInstance(distribution, dict)
        self.assertIn("distribution", distribution)
        self.assertIn("dimensions", distribution)
        
        # Überprüfe Dimensionen
        self.assertEqual(distribution["dimensions"], matrix["dimensions"])
        
        # Überprüfe Verteilungsdaten
        self.assertIsInstance(distribution["distribution"], np.ndarray)
        self.assertEqual(distribution["distribution"].shape, matrix["shape"])
        
        # Überprüfe, ob Summe der Wahrscheinlichkeiten ungefähr 1 ist
        self.assertAlmostEqual(np.sum(distribution["distribution"]), 1.0, places=5)
    
    def test_get_most_probable_state(self):
        """Test der Bestimmung des wahrscheinlichsten Zustands"""
        matrix = self.prism_matrix.create_matrix(self.test_data)
        distribution = self.prism_matrix.calculate_probability_distribution(matrix)
        
        # Setze einen Wert in der Verteilung auf ein Maximum
        max_indices = (1, 2)  # Beispielindizes
        distribution["distribution"] = np.zeros_like(distribution["distribution"])
        distribution["distribution"][max_indices] = 1.0
        
        most_probable = self.prism_matrix.get_most_probable_state(distribution)
        
        # Überprüfe Ergebnisstruktur
        self.assertIsInstance(most_probable, dict)
        self.assertIn("state", most_probable)
        self.assertIn("probability", most_probable)
        self.assertIn("indices", most_probable)
        
        # Überprüfe Wahrscheinlichkeit
        self.assertEqual(most_probable["probability"], 1.0)
        
        # Überprüfe Indizes
        self.assertEqual(most_probable["indices"], max_indices)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        self.prism_matrix = None


class TestTimeScopeUnit(unittest.TestCase):
    """Tests für die TimeScopeUnit-Klasse"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        self.time_scope_unit = TimeScopeUnit()
        
        # Beispieldaten für Tests
        self.test_data_id = "test_data"
        self.test_data = 42
    
    def test_initialization(self):
        """Test der Initialisierung"""
        self.assertIsNotNone(self.time_scope_unit)
        self.assertIsInstance(self.time_scope_unit.active_scopes, set)
        self.assertGreater(len(self.time_scope_unit.active_scopes), 0)
        
        # Überprüfe, ob Zeitfenster initialisiert wurden
        for scope in self.time_scope_unit.active_scopes:
            self.assertIn(scope, self.time_scope_unit.time_windows)
            window = self.time_scope_unit.time_windows[scope]
            self.assertIn("start_time", window)
            self.assertIn("end_time", window)
            self.assertIn("data_points", window)
    
    def test_register_temporal_data(self):
        """Test der Datenregistrierung"""
        # Registriere Daten für verschiedene Zeitbereiche
        for scope in TimeScope:
            if scope in self.time_scope_unit.active_scopes:
                result = self.time_scope_unit.register_temporal_data(
                    f"{self.test_data_id}_{scope.name}", 
                    self.test_data, 
                    scope
                )
                self.assertTrue(result)
        
        # Überprüfe, ob Daten registriert wurden
        for scope in TimeScope:
            if scope in self.time_scope_unit.active_scopes:
                data_id = f"{self.test_data_id}_{scope.name}"
                self.assertIn(data_id, self.time_scope_unit.temporal_data)
                
                # Überprüfe, ob Datenpunkt zum Zeitfenster hinzugefügt wurde
                window = self.time_scope_unit.time_windows[scope]
                self.assertGreater(len(window["data_points"]), 0)
    
    def test_get_temporal_data(self):
        """Test des Datenabrufs"""
        # Registriere Testdaten
        data_id = f"{self.test_data_id}_get"
        scope = TimeScope.MEDIUM
        
        if scope in self.time_scope_unit.active_scopes:
            self.time_scope_unit.register_temporal_data(data_id, self.test_data, scope)
            
            # Rufe Daten ab
            data = self.time_scope_unit.get_temporal_data(data_id)
            
            # Überprüfe Ergebnis
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
            
            first_entry = data[0]
            self.assertIn("timestamp", first_entry)
            self.assertIn("data", first_entry)
            self.assertIn("scope", first_entry)
            self.assertEqual(first_entry["data"], self.test_data)
            
            # Teste Filterung nach Zeitbereich
            filtered_data = self.time_scope_unit.get_temporal_data(data_id, scope)
            self.assertGreater(len(filtered_data), 0)
            
            # Teste nicht existierende ID
            nonexistent_data = self.time_scope_unit.get_temporal_data("nonexistent_id")
            self.assertEqual(len(nonexistent_data), 0)
    
    def test_update_time_windows(self):
        """Test der Zeitfensteraktualisierung"""
        # Registriere einige Testdaten
        self.time_scope_unit.register_temporal_data(
            f"{self.test_data_id}_update", 
            self.test_data, 
            TimeScope.IMMEDIATE
        )
        
        # Simuliere Zeitablauf
        original_time = time.time
        try:
            # Mock time.time, um einen Zeitsprung zu simulieren
            future_time = original_time() + self.time_scope_unit.scope_durations[TimeScope.IMMEDIATE] + 10
            time.time = lambda: future_time
            
            # Aktualisiere Zeitfenster
            removed_count = self.time_scope_unit.update_time_windows()
            
            # Überprüfe, ob Datenpunkte entfernt wurden
            self.assertGreaterEqual(removed_count, 0)
            
            # Überprüfe, ob Zeitfenster aktualisiert wurden
            for scope in self.time_scope_unit.active_scopes:
                window = self.time_scope_unit.time_windows[scope]
                self.assertGreaterEqual(window["start_time"], original_time())
                self.assertGreaterEqual(window["end_time"], window["start_time"])
        finally:
            # Stelle original time.time wieder her
            time.time = original_time
    
    def test_analyze_temporal_trends(self):
        """Test der temporalen Trendanalyse"""
        # Registriere Testdaten mit ansteigendem Trend
        data_id = f"{self.test_data_id}_trend"
        scope = TimeScope.MEDIUM
        
        if scope in self.time_scope_unit.active_scopes:
            for i in range(5):
                self.time_scope_unit.register_temporal_data(data_id, i, scope)
            
            # Analysiere Trend
            trend_analysis = self.time_scope_unit.analyze_temporal_trends(data_id, scope)
            
            # Überprüfe Ergebnisstruktur
            self.assertIsInstance(trend_analysis, dict)
            self.assertEqual(trend_analysis["data_id"], data_id)
            self.assertEqual(trend_analysis["scope"], scope.name)
            self.assertIn("trend", trend_analysis)
            self.assertIn("statistics", trend_analysis)
            
            # Überprüfe Statistiken
            stats = trend_analysis["statistics"]
            self.assertIn("mean", stats)
            self.assertIn("median", stats)
            self.assertIn("std_dev", stats)
            self.assertIn("min", stats)
            self.assertIn("max", stats)
            
            # Überprüfe Trend (sollte "increasing" sein)
            self.assertEqual(trend_analysis["trend"], "increasing")
    
    def test_get_status(self):
        """Test des Statusabrufs"""
        status = self.time_scope_unit.get_status()
        
        # Überprüfe Statusstruktur
        self.assertIsInstance(status, dict)
        self.assertIn("active_scopes", status)
        self.assertIn("time_windows", status)
        self.assertIn("temporal_data_count", status)
        self.assertIn("total_data_points", status)
        
        # Überprüfe aktive Zeitbereiche
        self.assertIsInstance(status["active_scopes"], list)
        self.assertEqual(len(status["active_scopes"]), len(self.time_scope_unit.active_scopes))
        
        # Überprüfe Zeitfenster
        self.assertIsInstance(status["time_windows"], dict)
        for scope_name, window in status["time_windows"].items():
            self.assertIn("start_time", window)
            self.assertIn("end_time", window)
            self.assertIn("data_points", window)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        self.time_scope_unit = None


if __name__ == "__main__":
    unittest.main()
