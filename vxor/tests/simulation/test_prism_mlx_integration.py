#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Tests für PRISM-Engine mit MLX-Optimierung

Testfälle für die Integration der PRISM-Engine mit der MLX-optimierten
T-Mathematics Engine.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import logging
import time
import numpy as np
import platform
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.tests.simulation.test_prism_mlx_integration")

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Prüfe auf Apple Silicon
is_apple_silicon = False
try:
    is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
except:
    pass

# Prüfe auf MLX
has_mlx = False
try:
    import mlx.core
    has_mlx = True
except ImportError:
    pass

# Importiere zu testende Module
from miso.simulation.prism_engine import PrismEngine, PrismMatrix
from miso.math.t_mathematics.engine import TMathEngine
from miso.math.t_mathematics.compat import TMathConfig


@unittest.skipIf(not is_apple_silicon or not has_mlx, "Benötigt Apple Silicon und MLX")
class TestPrismMLXIntegration(unittest.TestCase):
    """Tests für die Integration der PRISM-Engine mit der MLX-optimierten T-Mathematics Engine"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        # Setze Umgebungsvariablen für die Tests
        os.environ["T_MATH_USE_MLX"] = "1"  # Aktiviere MLX
        os.environ["T_MATH_PRECISION"] = "float16"  # Verwende float16 Präzision
        
        # Erstelle T-Mathematics Engine mit MLX
        self.math_engine = TMathEngine(
            config=TMathConfig(
                precision="float16",
                device="auto",
                use_mlx=True
            )
        )
        
        # Erstelle PRISM-Engine mit MLX-optimierter T-Mathematics Engine
        self.prism_engine = PrismEngine(math_engine=self.math_engine)
        
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
            {"name": "Variation 1", "probability": 0.7, "factor": 1.2},
            {"name": "Variation 2", "probability": 0.5, "factor": 0.8},
            {"name": "Variation 3", "probability": 0.3, "factor": 1.5}
        ]
        
        # Startzeit für Performance-Messungen
        self.start_time = time.time()
    
    def test_prism_matrix_with_mlx(self):
        """Test der PrismMatrix mit MLX-Optimierung"""
        # Erstelle eine PrismMatrix mit MLX-optimierter T-Mathematics Engine
        matrix = PrismMatrix(dimensions=3, size=[5, 5, 5], math_engine=self.math_engine)
        
        # Fülle die Matrix mit Testdaten
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    matrix.set_value_at_coordinates([x, y, z], (x + y + z) / 15.0)
        
        # Teste die Wahrscheinlichkeitsberechnung
        prob = matrix.get_probability_at_coordinates([2, 2, 2])
        self.assertIsNotNone(prob)
        self.assertAlmostEqual(prob, 0.4, delta=0.01)
        
        # Teste die Wahrscheinlichkeitsverteilungsberechnung
        distribution = matrix.calculate_probability_distribution()
        self.assertIsNotNone(distribution)
        self.assertEqual(len(distribution), 125)  # 5x5x5
        
        # Teste den wahrscheinlichsten Zustand
        most_probable = matrix.get_most_probable_state()
        self.assertIsNotNone(most_probable)
        self.assertEqual(len(most_probable["coordinates"]), 3)
    
    def test_prism_engine_monte_carlo_with_mlx(self):
        """Test der Monte-Carlo-Simulation mit MLX-Optimierung"""
        # Führe eine Monte-Carlo-Simulation durch
        results = self.prism_engine.run_monte_carlo_simulation(
            initial_state=self.test_state,
            variations=self.test_variations,
            num_iterations=1000,
            max_depth=5
        )
        
        # Überprüfe die Ergebnisse
        self.assertIsNotNone(results)
        self.assertIn("iterations", results)
        self.assertIn("states", results)
        self.assertIn("probabilities", results)
        self.assertIn("most_probable_path", results)
        
        # Überprüfe die Anzahl der Iterationen
        self.assertEqual(results["iterations"], 1000)
        
        # Überprüfe, ob die Wahrscheinlichkeiten gültig sind
        for prob in results["probabilities"]:
            self.assertTrue(0 <= prob <= 1)
    
    def test_prism_engine_reality_modulation_with_mlx(self):
        """Test der Realitätsmodulation mit MLX-Optimierung"""
        # Führe eine Realitätsmodulation durch
        modulation = self.prism_engine.modulate_reality(
            base_state=self.test_state,
            modulation_factor=0.8,
            stability_threshold=0.6
        )
        
        # Überprüfe die Ergebnisse
        self.assertIsNotNone(modulation)
        self.assertIn("modulated_state", modulation)
        self.assertIn("stability", modulation)
        self.assertIn("probability", modulation)
        
        # Überprüfe, ob die Wahrscheinlichkeit gültig ist
        self.assertTrue(0 <= modulation["probability"] <= 1)
        
        # Überprüfe, ob die Stabilität gültig ist
        self.assertTrue(0 <= modulation["stability"] <= 1)
    
    def test_prism_engine_probability_analysis_with_mlx(self):
        """Test der Wahrscheinlichkeitsanalyse mit MLX-Optimierung"""
        # Führe eine Wahrscheinlichkeitsanalyse durch
        analysis = self.prism_engine.analyze_probability_space(
            states=[self.test_state] * 3,  # Verwende den gleichen Zustand dreimal für Einfachheit
            variations=self.test_variations,
            depth=3
        )
        
        # Überprüfe die Ergebnisse
        self.assertIsNotNone(analysis)
        self.assertIn("entropy", analysis)
        self.assertIn("divergence", analysis)
        self.assertIn("stability", analysis)
        
        # Überprüfe, ob die Entropie gültig ist
        self.assertTrue(analysis["entropy"] >= 0)
        
        # Überprüfe, ob die Divergenz gültig ist
        self.assertTrue(analysis["divergence"] >= 0)
        
        # Überprüfe, ob die Stabilität gültig ist
        self.assertTrue(0 <= analysis["stability"] <= 1)
    
    def test_performance_comparison(self):
        """Vergleicht die Performance mit und ohne MLX-Optimierung"""
        # Speichere die aktuelle Konfiguration
        original_use_mlx = os.environ.get("T_MATH_USE_MLX", "1")
        
        # Deaktiviere MLX für den Vergleich
        os.environ["T_MATH_USE_MLX"] = "0"
        
        # Erstelle T-Mathematics Engine ohne MLX
        torch_math_engine = TMathEngine(
            config=TMathConfig(
                precision="float16",
                device="auto",
                use_mlx=False
            )
        )
        
        # Erstelle PRISM-Engine ohne MLX-Optimierung
        torch_prism_engine = PrismEngine(math_engine=torch_math_engine)
        
        # Führe Benchmark für Monte-Carlo-Simulation ohne MLX durch
        start_time_torch = time.time()
        torch_prism_engine.run_monte_carlo_simulation(
            initial_state=self.test_state,
            variations=self.test_variations,
            num_iterations=100,
            max_depth=3
        )
        torch_time = time.time() - start_time_torch
        
        # Aktiviere MLX wieder
        os.environ["T_MATH_USE_MLX"] = "1"
        
        # Führe Benchmark für Monte-Carlo-Simulation mit MLX durch
        start_time_mlx = time.time()
        self.prism_engine.run_monte_carlo_simulation(
            initial_state=self.test_state,
            variations=self.test_variations,
            num_iterations=100,
            max_depth=3
        )
        mlx_time = time.time() - start_time_mlx
        
        # Berechne Speedup
        speedup = torch_time / mlx_time if mlx_time > 0 else float('inf')
        
        # Logge die Ergebnisse
        logger.info(f"PyTorch Zeit: {torch_time:.4f}s")
        logger.info(f"MLX Zeit: {mlx_time:.4f}s")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Überprüfe, ob MLX schneller ist (sollte auf Apple Silicon sein)
        if is_apple_silicon:
            self.assertGreater(speedup, 1.0, "MLX sollte auf Apple Silicon schneller sein als PyTorch")
        
        # Stelle die ursprüngliche Konfiguration wieder her
        os.environ["T_MATH_USE_MLX"] = original_use_mlx
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        self.prism_engine = None
        self.math_engine = None
        
        # Logge die Testdauer
        test_time = time.time() - self.start_time
        logger.info(f"Test abgeschlossen in {test_time:.4f}s")


@unittest.skipIf(is_apple_silicon and has_mlx, "Nur für Systeme ohne Apple Silicon oder MLX")
class TestPrismMLXFallback(unittest.TestCase):
    """Tests für den Fallback der PRISM-Engine ohne MLX-Optimierung"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        # Setze Umgebungsvariablen für die Tests
        os.environ["T_MATH_USE_MLX"] = "1"  # Versuche MLX zu aktivieren
        
        # Erstelle T-Mathematics Engine (sollte auf PyTorch zurückfallen)
        self.math_engine = TMathEngine(
            config=TMathConfig(
                precision="float32",
                device="cpu",
                use_mlx=True  # Wird ignoriert, wenn MLX nicht verfügbar ist
            )
        )
        
        # Erstelle PRISM-Engine
        self.prism_engine = PrismEngine(math_engine=self.math_engine)
    
    def test_fallback_to_pytorch(self):
        """Test, ob die Engine korrekt auf PyTorch zurückfällt"""
        # Überprüfe, ob die Engine auf PyTorch zurückgefallen ist
        self.assertFalse(getattr(self.math_engine, "_use_mlx", False))
        
        # Überprüfe, ob die Engine trotzdem funktioniert
        matrix = PrismMatrix(dimensions=2, size=[3, 3], math_engine=self.math_engine)
        
        # Fülle die Matrix mit Testdaten
        for x in range(3):
            for y in range(3):
                matrix.set_value_at_coordinates([x, y], (x + y) / 6.0)
        
        # Teste die Wahrscheinlichkeitsberechnung
        prob = matrix.get_probability_at_coordinates([1, 1])
        self.assertIsNotNone(prob)
        self.assertAlmostEqual(prob, 0.33, delta=0.01)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        self.prism_engine = None
        self.math_engine = None


if __name__ == "__main__":
    unittest.main()
