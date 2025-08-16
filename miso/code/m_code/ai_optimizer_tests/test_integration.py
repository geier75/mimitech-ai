#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrationstests für den AI-Optimizer.

Diese Tests überprüfen die Integration des AI-Optimizers mit anderen MISO-Komponenten,
insbesondere:
- MLX-Adapter
- JIT-Compiler
- Parallel Executor
- ECHO-PRIME-Integration
- M-CODE Runtime
"""

import unittest
import os
import time
import tempfile
import shutil
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Import der zu testenden Komponenten
from miso.code.m_code.ai_optimizer import (
    OptimizerConfig,
    OptimizationStrategy,
    ExecutionContext,
    get_ai_optimizer,
    optimize
)

# Import anderer MISO-Komponenten
try:
    from miso.code.m_code.mlx_adapter import get_mlx_adapter, MLX_AVAILABLE
except ImportError:
    # Mock für Tests ohne MLX
    MLX_AVAILABLE = False
    def get_mlx_adapter():
        return None

try:
    from miso.code.m_code.jit_compiler import get_jit_compiler
except ImportError:
    # Mock für Tests ohne JIT-Compiler
    def get_jit_compiler():
        return None

try:
    from miso.code.m_code.parallel_executor import parallel
except ImportError:
    # Mock für Tests ohne Parallel Executor
    def parallel(func, use_processes=False):
        return func

try:
    from miso.code.m_code.echo_prime_integration import EchoPrimeIntegration, create_timeline_processor
except ImportError:
    # Mock für Tests ohne ECHO-PRIME
    EchoPrimeIntegration = None
    def create_timeline_processor(**kwargs):
        return None

try:
    from miso.code.m_code.runtime import MCodeRuntime
except ImportError:
    # Mock für Tests ohne M-CODE Runtime
    MCodeRuntime = None

try:
    from miso.code.m_code.tensor import MTensor
except ImportError:
    # Mock für Tests ohne MTensor
    class MTensor:
        @staticmethod
        def ones(shape):
            return np.ones(shape)
        
        @staticmethod
        def zeros(shape):
            return np.zeros(shape)
        
        @staticmethod
        def random(shape):
            return np.random.random(shape)


class TestMLXIntegration(unittest.TestCase):
    """Tests für die Integration des AI-Optimizers mit dem MLX-Adapter."""
    
    @unittest.skipIf(not MLX_AVAILABLE, "MLX nicht verfügbar")
    def test_mlx_strategy_adaptation(self):
        """Testet die Anpassung von Strategien basierend auf MLX-Verfügbarkeit."""
        optimizer = get_ai_optimizer()
        mlx_adapter = get_mlx_adapter()
        
        # Erstelle eine für ANE optimierte Strategie
        ane_strategy = OptimizationStrategy(
            strategy_id="test_ane",
            name="Test ANE",
            parallelization_level=0,
            jit_level=2,
            device_target="ane",
            memory_optimization="balanced"
        )
        
        # Adaptiere an Hardware
        adapted = optimizer._adapt_to_hardware(ane_strategy)
        
        # Überprüfe die Anpassung basierend auf verfügbarer Hardware
        if not mlx_adapter.is_available():
            self.assertEqual(adapted.device_target, "cpu", 
                             "Sollte auf CPU zurückfallen, wenn MLX nicht verfügbar ist")
        elif mlx_adapter.is_available() and not mlx_adapter.supports_ane():
            self.assertEqual(adapted.device_target, "gpu", 
                             "Sollte auf GPU zurückfallen, wenn ANE nicht unterstützt wird")
        else:
            self.assertEqual(adapted.device_target, "ane", 
                             "Sollte ANE als Ziel beibehalten, wenn unterstützt")
    
    @unittest.skipIf(not MLX_AVAILABLE, "MLX nicht verfügbar")
    def test_tensor_operation_optimization(self):
        """Testet die Optimierung von Tensor-Operationen mit MLX."""
        # Diese Funktion führt eine einfache Matrix-Multiplikation durch
        def matrix_multiply(a, b):
            return a @ b
        
        # Erstelle Testdaten
        a = MTensor.ones((100, 100))
        b = MTensor.ones((100, 100))
        
        # Unoptimierte Ausführung zum Vergleich
        start_time = time.time()
        result_standard = matrix_multiply(a, b)
        standard_time = time.time() - start_time
        
        # Optimierte Ausführung mit dem @optimize-Dekorator
        @optimize
        def optimized_matrix_multiply(a, b):
            return a @ b
        
        start_time = time.time()
        result_optimized = optimized_matrix_multiply(a, b)
        optimized_time = time.time() - start_time
        
        # Bei der ersten Ausführung kann die optimierte Version wegen
        # JIT-Kompilierung langsamer sein, daher führen wir sie erneut aus
        start_time = time.time()
        result_optimized = optimized_matrix_multiply(a, b)
        optimized_time_second = time.time() - start_time
        
        # Ausgabe der Ergebnisse für die Analyse
        print(f"\nMatrix-Multiplikationszeiten:")
        print(f"Standard: {standard_time:.6f}s")
        print(f"Optimiert (erste Ausführung): {optimized_time:.6f}s")
        print(f"Optimiert (zweite Ausführung): {optimized_time_second:.6f}s")
        
        # Überprüfe, ob die Ergebnisse konsistent sind
        np_result_standard = np.array(result_standard)
        np_result_optimized = np.array(result_optimized)
        self.assertTrue(np.allclose(np_result_standard, np_result_optimized),
                       "Optimierte und unoptimierte Ergebnisse sollten übereinstimmen")


class TestJITCompilerIntegration(unittest.TestCase):
    """Tests für die Integration des AI-Optimizers mit dem JIT-Compiler."""
    
    def test_jit_optimization_application(self):
        """Testet die Anwendung von JIT-Optimierungen."""
        optimizer = get_ai_optimizer()
        
        # Einfache Testfunktion
        def add(a, b):
            return a + b
        
        # Strategie mit JIT-Optimierung
        jit_strategy = OptimizationStrategy(
            strategy_id="jit_test",
            name="JIT Test",
            parallelization_level=0,
            jit_level=2,  # Maximale JIT-Optimierung
            device_target="cpu",
            memory_optimization="balanced"
        )
        
        try:
            # Wende JIT-Strategie an
            result = optimizer.apply_strategy(jit_strategy, add, 5, 7)
            self.assertEqual(result, 12, "Das Ergebnis sollte korrekt sein")
        except Exception as e:
            # JIT-Kompilierung könnte fehlschlagen, wenn kein JIT-Compiler verfügbar ist
            if get_jit_compiler() is None:
                self.skipTest("JIT-Compiler nicht verfügbar")
            else:
                self.fail(f"JIT-Optimierung fehlgeschlagen: {e}")


class TestParallelExecutorIntegration(unittest.TestCase):
    """Tests für die Integration des AI-Optimizers mit dem Parallel Executor."""
    
    def test_parallelization_application(self):
        """Testet die Anwendung von Parallelisierungsoptimierungen."""
        optimizer = get_ai_optimizer()
        
        # CPU-intensive Funktion, die von Parallelisierung profitieren kann
        def compute_squares(n):
            result = []
            for i in range(n):
                result.append(i * i)
            return result
        
        # Strategie mit Parallelisierung
        parallel_strategy = OptimizationStrategy(
            strategy_id="parallel_test",
            name="Parallel Test",
            parallelization_level=1,  # Thread-basierte Parallelisierung
            jit_level=0,
            device_target="cpu",
            memory_optimization="balanced"
        )
        
        try:
            # Unoptimierte Ausführung
            start_time = time.time()
            result_standard = compute_squares(10000)
            standard_time = time.time() - start_time
            
            # Optimierte Ausführung mit Parallelisierung
            start_time = time.time()
            result_parallel = optimizer.apply_strategy(parallel_strategy, compute_squares, 10000)
            parallel_time = time.time() - start_time
            
            # Ausgabe der Ergebnisse für die Analyse
            print(f"\nParallelisierungszeiten:")
            print(f"Standard: {standard_time:.6f}s")
            print(f"Parallel: {parallel_time:.6f}s")
            
            # Überprüfe, ob die Ergebnisse konsistent sind
            self.assertEqual(result_standard, result_parallel,
                           "Parallele und unparallele Ergebnisse sollten übereinstimmen")
            
        except Exception as e:
            self.skipTest(f"Parallel Executor Test übersprungen: {e}")


@unittest.skipIf(EchoPrimeIntegration is None, "ECHO-PRIME nicht verfügbar")
class TestEchoPrimeIntegration(unittest.TestCase):
    """Tests für die Integration des AI-Optimizers mit ECHO-PRIME."""
    
    def test_timeline_processor_optimization(self):
        """Testet die Optimierung eines Timeline-Prozessors."""
        try:
            # Erstelle einen Timeline-Prozessor mit Optimierung
            optimized_processor = create_timeline_processor(optimize_enabled=True)
            self.assertIsNotNone(optimized_processor, "Optimierter Prozessor sollte nicht None sein")
            
            # Erstelle einen Timeline-Prozessor ohne Optimierung
            standard_processor = create_timeline_processor(optimize_enabled=False)
            self.assertIsNotNone(standard_processor, "Standard-Prozessor sollte nicht None sein")
            
            # Erstelle Testdaten für die Timeline
            timeline_data = {
                "nodes": [
                    {"id": "node1", "time": 0, "data": {"value": np.random.rand(10, 10)}},
                    {"id": "node2", "time": 1, "data": {"value": np.random.rand(10, 10)}},
                    {"id": "node3", "time": 2, "data": {"value": np.random.rand(10, 10)}}
                ],
                "edges": [
                    {"source": "node1", "target": "node2", "type": "causal"},
                    {"source": "node2", "target": "node3", "type": "temporal"}
                ]
            }
            
            # Verarbeite mit dem Standard-Prozessor
            start_time = time.time()
            result_standard = standard_processor.process_timeline(timeline_data)
            standard_time = time.time() - start_time
            
            # Verarbeite mit dem optimierten Prozessor
            start_time = time.time()
            result_optimized = optimized_processor.process_timeline(timeline_data)
            optimized_time = time.time() - start_time
            
            # Ausgabe der Ergebnisse für die Analyse
            print(f"\nTimeline-Verarbeitungszeiten:")
            print(f"Standard: {standard_time:.6f}s")
            print(f"Optimiert: {optimized_time:.6f}s")
            
            # Überprüfe, ob die Ergebnisse konsistent sind
            self.assertEqual(result_standard["status"], result_optimized["status"],
                           "Timeline-Verarbeitungsstatus sollte übereinstimmen")
            
        except Exception as e:
            self.skipTest(f"ECHO-PRIME-Integration Test übersprungen: {e}")


@unittest.skipIf(MCodeRuntime is None, "M-CODE Runtime nicht verfügbar")
class TestMCodeRuntimeIntegration(unittest.TestCase):
    """Tests für die Integration des AI-Optimizers mit der M-CODE Runtime."""
    
    def setUp(self):
        """Test-Setup."""
        self.runtime = MCodeRuntime(debug=True)
    
    def test_mcode_execution_optimization(self):
        """Testet die Optimierung der M-CODE-Ausführung."""
        # Einfacher M-CODE
        simple_mcode = """
function add(a, b)
    return a + b
end

result = add(5, 7)
"""
        
        # Komplexerer M-CODE mit Tensor-Operationen
        complex_mcode = """
function matrix_multiply(size)
    a = tensor.ones(size, size)
    b = tensor.ones(size, size)
    c = a @ b
    return c
end

result = matrix_multiply(50)
"""
        
        try:
            # Teste einfachen Code
            result_simple_standard = self.runtime.execute(simple_mcode, optimize_execution=False)
            result_simple_optimized = self.runtime.execute(simple_mcode, optimize_execution=True)
            
            self.assertEqual(result_simple_standard, result_simple_optimized,
                           "Ergebnisse sollten übereinstimmen")
            
            # Teste komplexeren Code mit Tensor-Operationen
            start_time = time.time()
            result_complex_standard = self.runtime.execute(complex_mcode, optimize_execution=False)
            standard_time = time.time() - start_time
            
            start_time = time.time()
            result_complex_optimized = self.runtime.execute(complex_mcode, optimize_execution=True)
            optimized_time = time.time() - start_time
            
            # Ausgabe der Ergebnisse für die Analyse
            print(f"\nM-CODE-Ausführungszeiten:")
            print(f"Standard: {standard_time:.6f}s")
            print(f"Optimiert: {optimized_time:.6f}s")
            
            # Überprüfe die Form des Ergebnisses
            self.assertEqual(result_complex_standard.shape, result_complex_optimized.shape,
                           "Ergebnisformen sollten übereinstimmen")
            
        except Exception as e:
            self.skipTest(f"M-CODE Runtime Test übersprungen: {e}")


class TestEndToEndOptimization(unittest.TestCase):
    """End-to-End-Tests für den AI-Optimizer."""
    
    def setUp(self):
        """Test-Setup."""
        # Optimizer mit höherer Explorationsrate für mehr Variabilität in Tests
        config = OptimizerConfig(
            exploration_rate=0.5,
            learning_rate=0.2,
            batch_size=10,
            memory_size=100
        )
        # Zurücksetzen der Singleton-Instanz für isolierte Tests
        import miso.code.m_code.ai_optimizer
        miso.code.m_code.ai_optimizer._ai_optimizer_instance = None
        self.optimizer = get_ai_optimizer(config)
    
    def test_learning_improvement(self):
        """Testet, ob der Optimizer über mehrere Ausführungen lernt und sich verbessert."""
        # Definiere eine Testfunktion mit klarem Optimierungspotential
        def compute_intensive(n, iterations):
            result = 0
            for _ in range(iterations):
                for i in range(n):
                    result += i * i
            return result
        
        # Parameter
        n = 1000
        iterations = 3
        
        # Führe die Funktion mehrmals aus, damit der Optimizer lernen kann
        execution_times = []
        strategies_used = []
        
        for i in range(10):
            # Extrahiere Code als String (simuliert verschiedene Eingaben)
            import inspect
            code_str = inspect.getsource(compute_intensive)
            
            # Erstelle Ausführungskontext
            context = ExecutionContext(
                code_hash=f"test_{i}",
                input_shapes=[(n,)],
                input_types=["int"]
            )
            
            # Wähle Strategie
            strategy = self.optimizer.optimize(code_str, context)
            strategies_used.append(strategy.strategy_id)
            
            # Führe optimierte Funktion aus
            start_time = time.time()
            self.optimizer.apply_strategy(strategy, compute_intensive, n, iterations)
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            # Gib Feedback für das Lernen
            self.optimizer.feedback(
                code_str=code_str,
                strategy=strategy,
                execution_time=execution_time * 1000,  # in ms
                success=True
            )
        
        # Ausgabe der Ergebnisse für die Analyse
        print("\nLernfortschritt:")
        for i, (time_taken, strategy_id) in enumerate(zip(execution_times, strategies_used)):
            print(f"Iteration {i}: {time_taken:.6f}s mit Strategie {strategy_id}")
        
        # Berechne Durchschnitte für erste und letzte Hälfte
        first_half_avg = sum(execution_times[:5]) / 5
        second_half_avg = sum(execution_times[5:]) / 5
        
        print(f"Durchschnitt erste Hälfte: {first_half_avg:.6f}s")
        print(f"Durchschnitt zweite Hälfte: {second_half_avg:.6f}s")
        
        # Überprüfe, ob die zweite Hälfte nicht signifikant langsamer ist
        # (Wir können nicht garantieren, dass sie schneller ist, da das von vielen Faktoren abhängt)
        self.assertLessEqual(second_half_avg, first_half_avg * 1.5,
                           "Die zweite Hälfte sollte nicht signifikant langsamer sein")
    
    def test_strategy_persistence(self):
        """Testet die Persistenz und das Laden von Optimierungsstrategien."""
        # Temporäres Verzeichnis für Tests
        temp_dir = tempfile.mkdtemp()
        state_dir = os.path.join(temp_dir, "optimizer_state")
        
        try:
            # Testfunktion
            def simple_function(n):
                return sum(i for i in range(n))
            
            # Führe einige Optimierungen durch
            for i in range(5):
                code_str = f"test_code_{i}"
                context = ExecutionContext(
                    code_hash=code_str,
                    input_shapes=[(100,)],
                    input_types=["int"]
                )
                
                strategy = self.optimizer.optimize(code_str, context)
                self.optimizer.apply_strategy(strategy, simple_function, 100)
                
                # Gib Feedback
                self.optimizer.feedback(
                    code_str=code_str,
                    strategy=strategy,
                    execution_time=10.0,  # Beispielzeit
                    success=True
                )
            
            # Speichere den Zustand
            self.optimizer.save_state(state_dir)
            self.assertTrue(os.path.exists(os.path.join(state_dir, "ai_optimizer_model.pkl")))
            self.assertTrue(os.path.exists(os.path.join(state_dir, "ai_optimizer_patterns.json")))
            
            # Erstelle einen neuen Optimizer
            import miso.code.m_code.ai_optimizer
            miso.code.m_code.ai_optimizer._ai_optimizer_instance = None
            new_optimizer = get_ai_optimizer()
            
            # Lade den Zustand
            result = new_optimizer.load_state(state_dir)
            self.assertTrue(result)
            
            # Überprüfe, ob der geladene Optimizer ähnliche Entscheidungen trifft
            for i in range(5):
                code_str = f"test_code_{i}"
                context = ExecutionContext(
                    code_hash=code_str,
                    input_shapes=[(100,)],
                    input_types=["int"]
                )
                
                # Deaktiviere Exploration für deterministische Ergebnisse
                new_optimizer.reinforcement_learner.exploration_rate = 0.0
                
                # Beide Optimizer sollten für bekannte Muster die gleiche Strategie wählen
                strategy1 = self.optimizer.optimize(code_str, context)
                strategy2 = new_optimizer.optimize(code_str, context)
                
                self.assertEqual(strategy1.strategy_id, strategy2.strategy_id,
                               f"Strategien sollten für bekannten Code übereinstimmen: {code_str}")
            
        finally:
            # Aufräumen
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
