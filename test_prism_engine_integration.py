#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Integrationstest für die PRISM-Engine

Validiert die optimierte PRISM-Engine mit MLX-Integration und überprüft die Lösung 
der zirkulären Importprobleme mit dem Factory-Pattern.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import unittest
import numpy as np
import torch
import json
from typing import Dict, List, Any

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.test.prism_engine_integration")

# Prüfe, ob Apple Silicon verfügbar ist
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
if is_apple_silicon:
    logger.info("Test wird auf Apple Silicon ausgeführt - MLX-Optimierungen werden getestet")
    try:
        import mlx.core as mx
        HAS_MLX = True
    except ImportError:
        logger.warning("MLX nicht verfügbar, Tests für MLX-Optimierungen werden übersprungen")
        HAS_MLX = False
else:
    logger.info("Test wird nicht auf Apple Silicon ausgeführt - MLX-Tests werden übersprungen")
    HAS_MLX = False

# Importiere PRISM-Module
try:
    from miso.simulation.prism_factory import (
        get_prism_registry, get_prism_engine, get_prism_matrix,
        get_time_scope_unit, get_predictive_stream_analyzer, get_pattern_dissonance_scanner
    )
    
    from miso.simulation.prism_base import (
        SimulationConfig, TimelineType, TimeNode, Timeline
    )
    
    HAS_FACTORY = True
    logger.info("PRISM-Factory erfolgreich importiert")
except ImportError as e:
    logger.error(f"PRISM-Factory konnte nicht importiert werden: {e}")
    HAS_FACTORY = False


class TestPrismEngineIntegration(unittest.TestCase):
    """Testet die Integration der PRISM-Engine mit dem Factory-Pattern"""
    
    def setUp(self):
        """Testaufbau: Initialisierung der PRISM-Komponenten"""
        if not HAS_FACTORY:
            self.skipTest("PRISM-Factory nicht verfügbar")
            
        # Konfiguration für die PRISM-Engine
        self.config = {
            "matrix_dimensions": 4,
            "matrix_initial_size": 10,
            "sequence_length": 5,
            "prediction_horizon": 3,
            "precision": "float16" if HAS_MLX else "float32",
            "use_mlx": HAS_MLX
        }
        
        # Erstelle die PRISM-Engine mit Factory
        self.engine = get_prism_engine(config=self.config)
        
        # Prüfe, ob die Engine erfolgreich erstellt wurde
        self.assertIsNotNone(self.engine, "PRISM-Engine konnte nicht erstellt werden")
        
        # Erzeuge Testdaten
        self.test_timeline = Timeline(
            name="Test-Timeline",
            description="Testzeitlinie für Integration",
            timeline_type=TimelineType.MAIN
        )
        
        # Füge einige Knoten zur Zeitlinie hinzu
        for i in range(5):
            # Stelle sicher, dass die Daten eine Wahrscheinlichkeit enthalten
            node_data = {"value": float(i)}
            node = TimeNode(
                timestamp=time.time() + i * 3600,  # Stündliche Abstände
                data=node_data
            )
            # Setze die Wahrscheinlichkeit explizit als zusätzliches Attribut
            node.data["probability"] = 0.5 + i * 0.1
            self.test_timeline.add_node(node, is_root=(i == 0))
            
        # Registriere die Zeitlinie bei der PRISM-Engine
        self.engine.register_timeline(self.test_timeline)
        
    def tearDown(self):
        """Testabbau: Bereinigung der PRISM-Komponenten"""
        if HAS_FACTORY:
            # Entferne alle registrierten Komponenten
            registry = get_prism_registry()
            registry.clear()
            
    def test_factory_pattern(self):
        """Testet, ob das Factory-Pattern korrekt funktioniert"""
        # Prüfe, ob Registry verschiedene Komponenten enthält
        registry = get_prism_registry()
        component_ids = registry.get_all_component_ids()
        
        logger.info(f"Registrierte Komponenten: {component_ids}")
        self.assertTrue(len(component_ids) > 0, "Keine Komponenten in der Registry")
        
        # Prüfe, ob die Engine in der Registry ist
        self.assertTrue(registry.has_component("prism_engine"), "PRISM-Engine nicht in der Registry")
        
        # Teste wiederholten Abruf der gleichen Komponente
        engine1 = get_prism_engine(config=self.config)
        engine2 = get_prism_engine(config=self.config)
        
        # Beide sollten die gleiche Instanz sein (Singleton-Verhalten)
        self.assertIs(engine1, engine2, "Factory gibt nicht die gleiche Instanz zurück")
        
    def test_matrix_integration(self):
        """Testet die Integration der Matrix-Komponente mit der Engine"""
        # Prüfe, ob die Matrix korrekt initialisiert wurde
        self.assertIsNotNone(self.engine.matrix, "Matrix nicht initialisiert")
        
        # Teste die Dimensionen der Matrix
        self.assertEqual(self.engine.matrix.dimensions, self.config["matrix_dimensions"],
                         "Matrix hat falsche Dimensionen")
        
        # Prüfe den Tensor-Backend-Typ
        backend = self.engine.matrix.tensor_backend
        logger.info(f"Matrix verwendet Backend: {backend}")
        
        if HAS_MLX and is_apple_silicon:
            # Auf Apple Silicon mit MLX sollte ein MLX-Backend verwendet werden
            self.assertTrue("mlx" in backend, 
                           f"Kein MLX-Backend auf Apple Silicon (aktuell: {backend})")
        
    def test_reality_forker(self):
        """Testet die RealityForker-Komponente mit Factory-Integration"""
        # Prüfe, ob RealityForker korrekt initialisiert wurde
        self.assertIsNotNone(self.engine.reality_forker, "RealityForker nicht initialisiert")
        
        # Erstelle Testdaten für die Realitätsforking
        current_state = {
            "temperature": 20.0,
            "pressure": 1013.0,
            "humidity": 65.0,
            "wind_speed": 5.0
        }
        
        variations = [
            {"temperature": 25.0, "pressure": 1010.0},
            {"temperature": 15.0, "pressure": 1020.0},
            {"humidity": 80.0, "wind_speed": 10.0}
        ]
        
        # Forke die Realität
        alternative_realities = self.engine.reality_forker.fork_reality(current_state, variations)
        
        # Prüfe die Ergebnisse
        self.assertEqual(len(alternative_realities), len(variations),
                         "Anzahl der alternativen Realitäten stimmt nicht")
        
        # Prüfe, ob die Variationen angewendet wurden
        for i, reality in enumerate(alternative_realities):
            for key, value in variations[i].items():
                self.assertIn(key, reality, f"Schlüssel {key} nicht in Realität {i}")
        
        # Simuliere das Ergebnis
        for reality in alternative_realities:
            result = self.engine.reality_forker.simulate_outcome(reality, steps=50)
            self.assertIn("probability", result, "Wahrscheinlichkeit fehlt im Simulationsergebnis")
            
        # Führe die Realitäten zusammen
        merged_reality = self.engine.reality_forker.merge_realities(alternative_realities)
        self.assertIn("probability", merged_reality, "Wahrscheinlichkeit fehlt in zusammengeführter Realität")
        
    def test_timeline_operations(self):
        """Testet die Timeline-Operationen der PRISM-Engine"""
        # Prüfe, ob die Zeitlinie korrekt registriert wurde
        timeline_ids = self.engine.get_registered_timeline_ids()
        self.assertIn(self.test_timeline.id, timeline_ids, "Zeitlinie nicht registriert")
        
        # Berechne die Wahrscheinlichkeit der Zeitlinie
        probability = self.engine.calculate_timeline_probability(self.test_timeline.id)
        logger.info(f"Berechnete Zeitlinienwahrscheinlichkeit: {probability}")
        self.assertGreaterEqual(probability, 0.0, "Wahrscheinlichkeit unter 0")
        self.assertLessEqual(probability, 1.0, "Wahrscheinlichkeit über 1")
        
        # Analysiere die Stabilität der Zeitlinie
        stability = self.engine.analyze_timeline_stability(self.test_timeline.id)
        logger.info(f"Stabilitätsanalyse: {stability}")
        self.assertIn("stability_index", stability, "Stabilitätsindex fehlt")
        
        # Führe eine Simulation der Zeitlinie durch
        simulation = self.engine.simulate_timeline(self.test_timeline.id, steps=5)
        logger.info(f"Simulationsergebnisse: {json.dumps(simulation, indent=2)}")
        self.assertIn("steps", simulation, "Simulationsschritte fehlen")
        
    def test_tensor_operations(self):
        """Testet die Tensor-Operationen mit verschiedenen Backends"""
        if not HAS_MLX and is_apple_silicon:
            self.skipTest("MLX nicht verfügbar auf Apple Silicon")
            
        # Erstelle einen einfachen Tensor für Tests
        test_tensor_np = np.random.rand(3, 3).astype(np.float32)
        
        # Teste die Tensor-Operationen mit T-Mathematics
        result = self.engine.integrate_with_t_mathematics(
            tensor_operation="matmul",
            tensor_data=[test_tensor_np, test_tensor_np]
        )
        
        logger.info(f"Tensor-Operation (matmul) mit Backend: {result.get('backend')}")
        self.assertEqual(result.get("status"), "success", "Tensor-Operation fehlgeschlagen")
        
        # Prüfe das verwendete Backend
        if HAS_MLX and is_apple_silicon:
            self.assertIn("mlx", result.get("backend", ""), 
                         "MLX-Backend wird nicht verwendet auf Apple Silicon")


if __name__ == "__main__":
    unittest.main()
