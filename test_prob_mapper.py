#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für das ProbabilisticMapper-Modul der MPRIME Engine.
"""

import unittest
import sys
import os
import numpy as np
from typing import Dict, Any, List

# Füge das Hauptverzeichnis zum Pfad hinzu, um die MISO-Module zu importieren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from miso.math.mprime.prob_mapper import ProbabilisticMapper

class TestProbabilisticMapper(unittest.TestCase):
    """
    Testklasse für das ProbabilisticMapper-Modul
    """
    
    def setUp(self):
        """
        Einrichtung für die Tests
        """
        self.prob_mapper = ProbabilisticMapper()
    
    def test_initialization(self):
        """
        Testet die korrekte Initialisierung
        """
        self.assertIsNotNone(self.prob_mapper)
        self.assertEqual(self.prob_mapper.default_confidence, 0.95)
        self.assertEqual(self.prob_mapper.min_probability, 0.01)
        self.assertEqual(self.prob_mapper.max_iterations, 1000)
    
    def test_create_probability_space(self):
        """
        Testet die Erstellung eines Wahrscheinlichkeitsraums
        """
        dimensions = 2
        size = 5
        
        # Erstelle einen Wahrscheinlichkeitsraum
        result = self.prob_mapper.create_probability_space(dimensions, size)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob der Wahrscheinlichkeitsraum korrekt erstellt wurde
        self.assertEqual(result["dimensions"], dimensions)
        self.assertEqual(result["size"], size)
        self.assertIsInstance(result["probability_matrix"], np.ndarray)
        
        # Überprüfe die Form der Matrix
        expected_shape = tuple([size] * dimensions)
        self.assertEqual(result["probability_matrix"].shape, expected_shape)
        
        # Überprüfe, ob alle Werte in der Matrix zwischen 0 und 1 liegen
        self.assertTrue(np.all(result["probability_matrix"] >= 0))
        self.assertTrue(np.all(result["probability_matrix"] <= 1))
        
        # Überprüfe, ob die Summe aller Wahrscheinlichkeiten ungefähr 1 ist
        self.assertAlmostEqual(np.sum(result["probability_matrix"]), 1.0, places=5)
    
    def test_add_event(self):
        """
        Testet das Hinzufügen eines Ereignisses zum Wahrscheinlichkeitsraum
        """
        # Erstelle einen Wahrscheinlichkeitsraum
        self.prob_mapper.create_probability_space(2, 5)
        
        # Füge ein Ereignis hinzu
        event_id = "event1"
        coordinates = [1, 2]
        probability = 0.7
        
        result = self.prob_mapper.add_event(event_id, coordinates, probability)
        
        # Überprüfe, ob das Ergebnis True ist
        self.assertTrue(result)
        
        # Überprüfe, ob das Ereignis korrekt hinzugefügt wurde
        self.assertIn(event_id, self.prob_mapper.events)
        self.assertEqual(self.prob_mapper.events[event_id]["coordinates"], coordinates)
        self.assertEqual(self.prob_mapper.events[event_id]["probability"], probability)
        
        # Überprüfe, ob der Wert in der Matrix korrekt gesetzt wurde
        self.assertEqual(self.prob_mapper.probability_space["probability_matrix"][tuple(coordinates)], probability)
    
    def test_add_conditional_event(self):
        """
        Testet das Hinzufügen eines bedingten Ereignisses
        """
        # Erstelle einen Wahrscheinlichkeitsraum
        self.prob_mapper.create_probability_space(2, 5)
        
        # Füge zwei Ereignisse hinzu
        self.prob_mapper.add_event("event1", [1, 2], 0.7)
        self.prob_mapper.add_event("event2", [3, 4], 0.6)
        
        # Füge ein bedingtes Ereignis hinzu
        event_id = "conditional1"
        condition_event_id = "event1"
        target_coordinates = [2, 3]
        conditional_probability = 0.8
        
        result = self.prob_mapper.add_conditional_event(event_id, condition_event_id, target_coordinates, conditional_probability)
        
        # Überprüfe, ob das Ergebnis True ist
        self.assertTrue(result)
        
        # Überprüfe, ob das bedingte Ereignis korrekt hinzugefügt wurde
        self.assertIn(event_id, self.prob_mapper.conditional_events)
        self.assertEqual(self.prob_mapper.conditional_events[event_id]["condition_event_id"], condition_event_id)
        self.assertEqual(self.prob_mapper.conditional_events[event_id]["coordinates"], target_coordinates)
        self.assertEqual(self.prob_mapper.conditional_events[event_id]["conditional_probability"], conditional_probability)
    
    def test_calculate_joint_probability(self):
        """
        Testet die Berechnung der gemeinsamen Wahrscheinlichkeit
        """
        # Erstelle einen Wahrscheinlichkeitsraum
        self.prob_mapper.create_probability_space(2, 5)
        
        # Füge zwei Ereignisse hinzu
        self.prob_mapper.add_event("event1", [1, 2], 0.7)
        self.prob_mapper.add_event("event2", [3, 4], 0.6)
        
        # Berechne die gemeinsame Wahrscheinlichkeit
        result = self.prob_mapper.calculate_joint_probability(["event1", "event2"])
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die gemeinsame Wahrscheinlichkeit korrekt berechnet wurde
        # P(event1 ∩ event2) = P(event1) * P(event2) (wenn unabhängig)
        expected_probability = 0.7 * 0.6
        self.assertAlmostEqual(result["joint_probability"], expected_probability, places=5)
    
    def test_calculate_conditional_probability(self):
        """
        Testet die Berechnung der bedingten Wahrscheinlichkeit
        """
        # Erstelle einen Wahrscheinlichkeitsraum
        self.prob_mapper.create_probability_space(2, 5)
        
        # Füge zwei Ereignisse hinzu
        self.prob_mapper.add_event("event1", [1, 2], 0.7)
        self.prob_mapper.add_event("event2", [3, 4], 0.6)
        
        # Berechne die bedingte Wahrscheinlichkeit
        result = self.prob_mapper.calculate_conditional_probability("event2", "event1")
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die bedingte Wahrscheinlichkeit korrekt berechnet wurde
        # P(event2|event1) = P(event2) (wenn unabhängig)
        expected_probability = 0.6
        self.assertAlmostEqual(result["conditional_probability"], expected_probability, places=5)
    
    def test_update_probability(self):
        """
        Testet die Aktualisierung einer Wahrscheinlichkeit
        """
        # Erstelle einen Wahrscheinlichkeitsraum
        self.prob_mapper.create_probability_space(2, 5)
        
        # Füge ein Ereignis hinzu
        self.prob_mapper.add_event("event1", [1, 2], 0.7)
        
        # Aktualisiere die Wahrscheinlichkeit
        new_probability = 0.8
        result = self.prob_mapper.update_probability("event1", new_probability)
        
        # Überprüfe, ob das Ergebnis True ist
        self.assertTrue(result)
        
        # Überprüfe, ob die Wahrscheinlichkeit korrekt aktualisiert wurde
        self.assertEqual(self.prob_mapper.events["event1"]["probability"], new_probability)
        self.assertEqual(self.prob_mapper.probability_space["probability_matrix"][1, 2], new_probability)
    
    def test_normalize_probability_space(self):
        """
        Testet die Normalisierung des Wahrscheinlichkeitsraums
        """
        # Erstelle einen Wahrscheinlichkeitsraum
        self.prob_mapper.create_probability_space(2, 5)
        
        # Füge mehrere Ereignisse hinzu
        self.prob_mapper.add_event("event1", [1, 2], 0.7)
        self.prob_mapper.add_event("event2", [3, 4], 0.6)
        self.prob_mapper.add_event("event3", [0, 1], 0.5)
        
        # Normalisiere den Wahrscheinlichkeitsraum
        result = self.prob_mapper.normalize_probability_space()
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Summe aller Wahrscheinlichkeiten ungefähr 1 ist
        self.assertAlmostEqual(np.sum(result["probability_matrix"]), 1.0, places=5)
    
    def test_calculate_entropy(self):
        """
        Testet die Berechnung der Entropie
        """
        # Erstelle einen Wahrscheinlichkeitsraum
        self.prob_mapper.create_probability_space(2, 5)
        
        # Füge mehrere Ereignisse hinzu
        self.prob_mapper.add_event("event1", [1, 2], 0.7)
        self.prob_mapper.add_event("event2", [3, 4], 0.3)
        
        # Berechne die Entropie
        result = self.prob_mapper.calculate_entropy()
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Entropie korrekt berechnet wurde
        # H = -∑ p_i * log2(p_i)
        expected_entropy = -(0.7 * np.log2(0.7) + 0.3 * np.log2(0.3))
        self.assertAlmostEqual(result["entropy"], expected_entropy, places=5)
    
    def test_monte_carlo_simulation(self):
        """
        Testet die Monte-Carlo-Simulation
        """
        # Erstelle einen Wahrscheinlichkeitsraum
        self.prob_mapper.create_probability_space(2, 5)
        
        # Füge mehrere Ereignisse hinzu
        self.prob_mapper.add_event("event1", [1, 2], 0.7)
        self.prob_mapper.add_event("event2", [3, 4], 0.3)
        
        # Führe eine Monte-Carlo-Simulation durch
        num_simulations = 1000
        result = self.prob_mapper.monte_carlo_simulation(num_simulations)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Simulation die richtige Anzahl an Durchläufen hat
        self.assertEqual(result["num_simulations"], num_simulations)
        self.assertEqual(len(result["simulation_results"]), num_simulations)
        
        # Überprüfe, ob die Häufigkeiten ungefähr den Wahrscheinlichkeiten entsprechen
        event1_count = sum(1 for sim in result["simulation_results"] if sim["event_id"] == "event1")
        event2_count = sum(1 for sim in result["simulation_results"] if sim["event_id"] == "event2")
        
        event1_frequency = event1_count / num_simulations
        event2_frequency = event2_count / num_simulations
        
        # Erlaubt eine Abweichung von 10% aufgrund der zufälligen Natur der Simulation
        self.assertAlmostEqual(event1_frequency, 0.7, delta=0.1)
        self.assertAlmostEqual(event2_frequency, 0.3, delta=0.1)
    
    def test_calculate_expected_value(self):
        """
        Testet die Berechnung des Erwartungswerts
        """
        # Erstelle einen Wahrscheinlichkeitsraum
        self.prob_mapper.create_probability_space(2, 5)
        
        # Füge mehrere Ereignisse mit Werten hinzu
        self.prob_mapper.add_event("event1", [1, 2], 0.7)
        self.prob_mapper.events["event1"]["value"] = 10
        
        self.prob_mapper.add_event("event2", [3, 4], 0.3)
        self.prob_mapper.events["event2"]["value"] = 20
        
        # Berechne den Erwartungswert
        result = self.prob_mapper.calculate_expected_value()
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob der Erwartungswert korrekt berechnet wurde
        # E[X] = ∑ x_i * p_i
        expected_value = 0.7 * 10 + 0.3 * 20
        self.assertAlmostEqual(result["expected_value"], expected_value, places=5)
    
    def test_calculate_variance(self):
        """
        Testet die Berechnung der Varianz
        """
        # Erstelle einen Wahrscheinlichkeitsraum
        self.prob_mapper.create_probability_space(2, 5)
        
        # Füge mehrere Ereignisse mit Werten hinzu
        self.prob_mapper.add_event("event1", [1, 2], 0.7)
        self.prob_mapper.events["event1"]["value"] = 10
        
        self.prob_mapper.add_event("event2", [3, 4], 0.3)
        self.prob_mapper.events["event2"]["value"] = 20
        
        # Berechne die Varianz
        result = self.prob_mapper.calculate_variance()
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Varianz korrekt berechnet wurde
        # Var[X] = E[(X - E[X])^2] = ∑ (x_i - E[X])^2 * p_i
        expected_value = 0.7 * 10 + 0.3 * 20
        variance = 0.7 * (10 - expected_value)**2 + 0.3 * (20 - expected_value)**2
        self.assertAlmostEqual(result["variance"], variance, places=5)

if __name__ == '__main__':
    unittest.main()
