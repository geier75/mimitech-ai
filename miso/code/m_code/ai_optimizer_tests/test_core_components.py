#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit-Tests für die Kernkomponenten des AI-Optimizers.

Diese Tests überprüfen die Funktionalität der Grundkomponenten des AI-Optimizers,
einschließlich:
- OptimizerConfig
- OptimizationStrategy
- CodePattern
- ExecutionContext
- PatternRecognizer
- ReinforcementLearner
"""

import unittest
import os
import tempfile
import shutil
import time
import numpy as np
from typing import Dict, Any, List, Optional

# Import der zu testenden Komponenten
from miso.code.m_code.ai_optimizer import (
    OptimizerConfig,
    OptimizationStrategy,
    CodePattern,
    ExecutionContext,
    PatternRecognizer,
    ReinforcementLearner,
    get_ai_optimizer
)


class TestOptimizerConfig(unittest.TestCase):
    """Tests für die OptimizerConfig-Klasse."""
    
    def test_default_values(self):
        """Testet die Standardwerte der Konfiguration."""
        config = OptimizerConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.exploration_rate, 0.1)
        self.assertEqual(config.discount_factor, 0.9)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.memory_size, 10000)
        self.assertIsNone(config.model_path)
    
    def test_custom_values(self):
        """Testet benutzerdefinierte Konfigurationswerte."""
        config = OptimizerConfig(
            enabled=False,
            learning_rate=0.05,
            exploration_rate=0.2,
            discount_factor=0.8,
            batch_size=32,
            memory_size=5000,
            model_path="/path/to/model"
        )
        self.assertFalse(config.enabled)
        self.assertEqual(config.learning_rate, 0.05)
        self.assertEqual(config.exploration_rate, 0.2)
        self.assertEqual(config.discount_factor, 0.8)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.memory_size, 5000)
        self.assertEqual(config.model_path, "/path/to/model")


class TestOptimizationStrategy(unittest.TestCase):
    """Tests für die OptimizationStrategy-Klasse."""
    
    def test_basic_strategy(self):
        """Testet die Erstellung einer grundlegenden Strategie."""
        strategy = OptimizationStrategy(
            strategy_id="test_strategy",
            name="Test Strategy"
        )
        self.assertEqual(strategy.strategy_id, "test_strategy")
        self.assertEqual(strategy.name, "Test Strategy")
        self.assertEqual(strategy.parallelization_level, 0)
        self.assertEqual(strategy.jit_level, 0)
        self.assertEqual(strategy.device_target, "auto")
        self.assertEqual(strategy.memory_optimization, "balanced")
        self.assertFalse(strategy.batch_processing)
        self.assertFalse(strategy.tensor_fusion)
    
    def test_advanced_strategy(self):
        """Testet die Erstellung einer erweiterten Strategie."""
        strategy = OptimizationStrategy(
            strategy_id="advanced_strategy",
            name="Advanced Strategy",
            parallelization_level=2,
            jit_level=2,
            device_target="gpu",
            memory_optimization="high",
            batch_processing=True,
            tensor_fusion=True,
            operator_fusion=True,
            loop_unrolling=True,
            automatic_differentiation=True,
            confidence=0.8
        )
        self.assertEqual(strategy.strategy_id, "advanced_strategy")
        self.assertEqual(strategy.name, "Advanced Strategy")
        self.assertEqual(strategy.parallelization_level, 2)
        self.assertEqual(strategy.jit_level, 2)
        self.assertEqual(strategy.device_target, "gpu")
        self.assertEqual(strategy.memory_optimization, "high")
        self.assertTrue(strategy.batch_processing)
        self.assertTrue(strategy.tensor_fusion)
        self.assertTrue(strategy.operator_fusion)
        self.assertTrue(strategy.loop_unrolling)
        self.assertTrue(strategy.automatic_differentiation)
        self.assertEqual(strategy.confidence, 0.8)


class TestCodePattern(unittest.TestCase):
    """Tests für die CodePattern-Klasse."""
    
    def test_pattern_creation(self):
        """Testet die Erstellung eines Codemusters."""
        pattern = CodePattern(
            pattern_id="test_pattern",
            name="Test Pattern",
            description="A test pattern",
            features={"loops": 2, "tensor_ops": 5, "complexity": "medium"}
        )
        self.assertEqual(pattern.pattern_id, "test_pattern")
        self.assertEqual(pattern.name, "Test Pattern")
        self.assertEqual(pattern.description, "A test pattern")
        self.assertEqual(pattern.features["loops"], 2)
        self.assertEqual(pattern.features["tensor_ops"], 5)
        self.assertEqual(pattern.features["complexity"], "medium")
        self.assertEqual(pattern.frequency, 0)
        self.assertEqual(pattern.avg_execution_time, 0.0)
        self.assertIsNone(pattern.optimal_strategy)
    
    def test_pattern_with_strategy(self):
        """Testet ein Muster mit zugewiesener Strategie."""
        strategy = OptimizationStrategy(
            strategy_id="test_strategy",
            name="Test Strategy",
            confidence=0.9
        )
        pattern = CodePattern(
            pattern_id="pattern_with_strategy",
            name="Pattern With Strategy",
            description="A pattern with an optimal strategy",
            features={"tensor_ops": 10},
            frequency=5,
            avg_execution_time=100.0,
            optimal_strategy=strategy
        )
        self.assertEqual(pattern.frequency, 5)
        self.assertEqual(pattern.avg_execution_time, 100.0)
        self.assertIsNotNone(pattern.optimal_strategy)
        self.assertEqual(pattern.optimal_strategy.strategy_id, "test_strategy")
        self.assertEqual(pattern.optimal_strategy.confidence, 0.9)


class TestExecutionContext(unittest.TestCase):
    """Tests für die ExecutionContext-Klasse."""
    
    def test_basic_context(self):
        """Testet die Erstellung eines grundlegenden Ausführungskontexts."""
        context = ExecutionContext(
            code_hash="abc123",
            input_shapes=[(10, 10), (10, 5)],
            input_types=["float32", "float32"]
        )
        self.assertEqual(context.code_hash, "abc123")
        self.assertEqual(context.input_shapes, [(10, 10), (10, 5)])
        self.assertEqual(context.input_types, ["float32", "float32"])
        self.assertIsNone(context.output_shape)
        self.assertIsNone(context.output_type)
        self.assertEqual(context.execution_time_ms, 0.0)
        self.assertEqual(context.memory_usage_mb, 0.0)
        self.assertIsNone(context.strategy)
        self.assertTrue(context.success)
        self.assertIsNone(context.error)
    
    def test_complete_context(self):
        """Testet die Erstellung eines vollständigen Ausführungskontexts."""
        strategy = OptimizationStrategy(
            strategy_id="test_strategy",
            name="Test Strategy"
        )
        context = ExecutionContext(
            code_hash="xyz789",
            input_shapes=[(100, 100)],
            input_types=["float64"],
            output_shape=(100, 100),
            output_type="float64",
            execution_time_ms=150.5,
            memory_usage_mb=256.0,
            strategy=strategy,
            success=False,
            error="Division by zero"
        )
        self.assertEqual(context.code_hash, "xyz789")
        self.assertEqual(context.output_shape, (100, 100))
        self.assertEqual(context.output_type, "float64")
        self.assertEqual(context.execution_time_ms, 150.5)
        self.assertEqual(context.memory_usage_mb, 256.0)
        self.assertEqual(context.strategy.strategy_id, "test_strategy")
        self.assertFalse(context.success)
        self.assertEqual(context.error, "Division by zero")


class TestPatternRecognizer(unittest.TestCase):
    """Tests für die PatternRecognizer-Klasse."""
    
    def setUp(self):
        """Test-Setup."""
        self.recognizer = PatternRecognizer(min_frequency=2)
    
    def test_hash_code(self):
        """Testet die Code-Hash-Funktion."""
        code1 = "def add(a, b):\n    return a + b"
        code2 = "def add(a, b):\n    return a + b"
        code3 = "def multiply(a, b):\n    return a * b"
        
        hash1 = self.recognizer._hash_code(code1)
        hash2 = self.recognizer._hash_code(code2)
        hash3 = self.recognizer._hash_code(code3)
        
        self.assertEqual(hash1, hash2, "Identischer Code sollte identische Hashes haben")
        self.assertNotEqual(hash1, hash3, "Unterschiedlicher Code sollte unterschiedliche Hashes haben")
    
    def test_extract_features(self):
        """Testet die Feature-Extraktion aus Code."""
        code = """
def matrix_multiply(a, b):
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            cell = 0
            for k in range(len(b)):
                cell += a[i][k] * b[k][j]
            row.append(cell)
        result.append(row)
    return result
"""
        context = ExecutionContext(
            code_hash="test",
            input_shapes=[(10, 10), (10, 10)],
            input_types=["float32", "float32"]
        )
        
        features = self.recognizer._extract_features(code, context)
        
        self.assertIn("code_length", features)
        self.assertIn("loop_count", features)
        self.assertIn("function_count", features)
        self.assertIn("math_operations", features)
        self.assertIn("input_dimensions", features)
        
        self.assertGreater(features["code_length"], 0)
        self.assertGreaterEqual(features["loop_count"], 3)  # 3 verschachtelte Schleifen
        self.assertEqual(features["function_count"], 1)  # 1 Funktion definiert
        self.assertGreaterEqual(features["math_operations"], 1)  # Mindestens eine mathematische Operation
    
    def test_analyze_code(self):
        """Testet die Code-Analyse und Mustererkennung."""
        code = """
def simple_add(a, b):
    return a + b
"""
        context = ExecutionContext(
            code_hash="test_simple",
            input_shapes=[(1,), (1,)],
            input_types=["float32", "float32"]
        )
        
        # Erste Analyse sollte ein neues Muster erzeugen
        pattern_id1 = self.recognizer.analyze_code(code, context)
        self.assertIsNotNone(pattern_id1)
        self.assertIn(pattern_id1, self.recognizer.patterns)
        self.assertEqual(self.recognizer.patterns[pattern_id1].frequency, 1)
        
        # Zweite Analyse des gleichen Codes sollte das gleiche Muster erkennen
        # und die Häufigkeit erhöhen
        pattern_id2 = self.recognizer.analyze_code(code, context)
        self.assertEqual(pattern_id1, pattern_id2)
        self.assertEqual(self.recognizer.patterns[pattern_id1].frequency, 2)
    
    def test_update_optimal_strategy(self):
        """Testet die Aktualisierung der optimalen Strategie für ein Muster."""
        code = "def test(): return 42"
        context = ExecutionContext(
            code_hash="test_update",
            input_shapes=[],
            input_types=[]
        )
        
        # Muster erzeugen
        pattern_id = self.recognizer.analyze_code(code, context)
        
        # Strategie erstellen
        strategy = OptimizationStrategy(
            strategy_id="optimal_strategy",
            name="Optimal Strategy",
            confidence=0.9
        )
        
        # Strategie aktualisieren
        self.recognizer.update_optimal_strategy(pattern_id, strategy)
        
        # Überprüfen, ob die Strategie korrekt zugewiesen wurde
        pattern = self.recognizer.get_pattern(pattern_id)
        self.assertIsNotNone(pattern.optimal_strategy)
        self.assertEqual(pattern.optimal_strategy.strategy_id, "optimal_strategy")
        self.assertEqual(pattern.optimal_strategy.confidence, 0.9)


class TestReinforcementLearner(unittest.TestCase):
    """Tests für die ReinforcementLearner-Klasse."""
    
    def setUp(self):
        """Test-Setup."""
        config = OptimizerConfig(
            learning_rate=0.1,
            exploration_rate=0.2,
            discount_factor=0.9,
            batch_size=10,
            memory_size=100
        )
        self.learner = ReinforcementLearner(config)
        
        # Strategien registrieren
        strategies = [
            OptimizationStrategy(strategy_id="strategy1", name="Strategy 1"),
            OptimizationStrategy(strategy_id="strategy2", name="Strategy 2"),
            OptimizationStrategy(strategy_id="strategy3", name="Strategy 3")
        ]
        for strategy in strategies:
            self.learner.register_strategy(strategy)
    
    def test_register_strategy(self):
        """Testet die Registrierung von Strategien."""
        self.assertEqual(len(self.learner.strategies), 3)
        self.assertIn("strategy1", self.learner.strategies)
        self.assertIn("strategy2", self.learner.strategies)
        self.assertIn("strategy3", self.learner.strategies)
        
        # Weitere Strategie registrieren
        strategy4 = OptimizationStrategy(strategy_id="strategy4", name="Strategy 4")
        self.learner.register_strategy(strategy4)
        self.assertEqual(len(self.learner.strategies), 4)
        self.assertIn("strategy4", self.learner.strategies)
    
    def test_get_action(self):
        """Testet die Aktionsauswahl."""
        # Neuer Zustand, sollte zufällige Exploration verwenden
        action1 = self.learner.get_action("state1")
        self.assertIn(action1, ["strategy1", "strategy2", "strategy3"])
        
        # Q-Werte für state1 manuell setzen
        self.learner.q_table["state1"] = {
            "strategy1": 0.1,
            "strategy2": 0.5,
            "strategy3": 0.2
        }
        
        # Exploration deaktivieren
        self.learner.exploration_rate = 0.0
        
        # Sollte die beste Strategie wählen (strategy2)
        action2 = self.learner.get_action("state1")
        self.assertEqual(action2, "strategy2")
    
    def test_remember_and_learn(self):
        """Testet das Speichern von Erfahrungen und Lernen."""
        # Erfahrungen speichern
        self.learner.remember("state1", "strategy1", 0.5, "state2")
        self.learner.remember("state2", "strategy2", 0.8, "state3")
        self.learner.remember("state3", "strategy3", 0.3, "state1")
        
        self.assertEqual(len(self.learner.memory), 3)
        
        # Modell aktualisieren
        initial_q_table = self.learner.q_table.copy()
        self.learner._update_model()
        
        # Überprüfen, ob sich das Q-Table geändert hat
        self.assertNotEqual(self.learner.q_table, initial_q_table)
    
    def test_get_best_strategy(self):
        """Testet die Auswahl der besten Strategie."""
        # Q-Werte für state1 setzen
        self.learner.q_table["state1"] = {
            "strategy1": 0.1,
            "strategy2": 0.5,
            "strategy3": 0.2
        }
        
        # Beste Strategie für state1 abrufen
        strategy = self.learner.get_best_strategy("state1")
        self.assertEqual(strategy.strategy_id, "strategy2")
        
        # Beste Strategie für unbekannten Zustand abrufen
        # (sollte die erste registrierte Strategie sein)
        strategy_unknown = self.learner.get_best_strategy("unknown_state")
        self.assertEqual(strategy_unknown.strategy_id, "strategy1")
    
    def test_save_load_model(self):
        """Testet das Speichern und Laden des Modells."""
        # Temporäres Verzeichnis für Tests
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "model.pkl")
        
        try:
            # Q-Werte setzen
            self.learner.q_table["state1"] = {
                "strategy1": 0.1,
                "strategy2": 0.5,
                "strategy3": 0.2
            }
            self.learner.exploration_rate = 0.15
            self.learner.learning_rate = 0.08
            self.learner.update_count = 10
            
            # Modell speichern
            self.learner.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Neuen Learner erstellen
            new_learner = ReinforcementLearner(OptimizerConfig())
            for strategy in self.learner.strategies.values():
                new_learner.register_strategy(strategy)
            
            # Modell laden
            result = new_learner.load_model(model_path)
            self.assertTrue(result)
            
            # Überprüfen, ob die Daten korrekt geladen wurden
            self.assertEqual(new_learner.q_table, self.learner.q_table)
            self.assertEqual(new_learner.exploration_rate, self.learner.exploration_rate)
            self.assertEqual(new_learner.learning_rate, self.learner.learning_rate)
            self.assertEqual(new_learner.update_count, self.learner.update_count)
        
        finally:
            # Aufräumen
            shutil.rmtree(temp_dir)


class TestIntegrationBasic(unittest.TestCase):
    """Grundlegende Integrationstests für den AI-Optimizer."""
    
    def test_get_ai_optimizer(self):
        """Testet die Singleton-Funktion für den AI-Optimizer."""
        # Erste Instanz abrufen
        optimizer1 = get_ai_optimizer()
        self.assertIsNotNone(optimizer1)
        
        # Zweite Instanz abrufen, sollte identisch sein
        optimizer2 = get_ai_optimizer()
        self.assertIs(optimizer2, optimizer1)
        
        # Instanz mit benutzerdefinierter Konfiguration abrufen
        custom_config = OptimizerConfig(enabled=False)
        optimizer3 = get_ai_optimizer(custom_config)
        
        # In unserem aktuellen Design überschreibt eine neue Konfiguration nicht die bestehende Instanz
        # Dies könnte je nach Anforderungen geändert werden
        self.assertIs(optimizer3, optimizer1)


if __name__ == "__main__":
    unittest.main()
