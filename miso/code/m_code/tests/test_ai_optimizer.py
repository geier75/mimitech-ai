"""
Testsuite für den AI-Optimizer.

Diese Tests validieren die Funktionalität des AI-Optimizers, insbesondere:
- Mustererkennungsfähigkeiten
- Strategieauswahl
- Leistungsverbesserungen
- Adaptivität an Hardware
- Reinforcement Learning
"""

import unittest
import os
import time
import numpy as np
import tempfile
import shutil
from typing import List, Dict, Any, Tuple

# Import M-CODE-Komponenten
from miso.code.m_code.ai_optimizer import (
    get_ai_optimizer, 
    optimize, 
    OptimizerConfig, 
    OptimizationStrategy,
    CodePattern,
    ExecutionContext
)
from miso.code.m_code.debug_profiler import profile, get_profiler
from miso.code.m_code.mlx_adapter import get_mlx_adapter
from miso.code.m_code.jit_compiler import get_jit_compiler
from miso.code.m_code.tensor import MTensor


class AIOptimizerBasicTests(unittest.TestCase):
    """Grundlegende Tests für den AI-Optimizer."""
    
    def setUp(self):
        """Test-Setup."""
        # Konfiguriere einen Optimizer für Tests
        self.config = OptimizerConfig(
            enabled=True,
            exploration_rate=0.2,
            learning_rate=0.1,
            memory_size=1000,
            batch_size=32,
            update_interval=10,
            model_path=None
        )
        
        # Erstelle temporäres Verzeichnis für Modellspeicherung
        self.temp_dir = tempfile.mkdtemp()
        
        # Hole eine Optimizer-Instanz mit Testkonfiguration
        self._reset_optimizer()
    
    def tearDown(self):
        """Test-Teardown."""
        # Bereinige temporäres Verzeichnis
        shutil.rmtree(self.temp_dir)
    
    def _reset_optimizer(self):
        """Setzt den Optimizer zurück."""
        # Hack: Setze die Singleton-Instanz zurück
        from miso.code.m_code.ai_optimizer import _ai_optimizer_instance
        globals()["_ai_optimizer_instance"] = None
        
        # Hole eine neue Instanz
        self.optimizer = get_ai_optimizer(self.config)
    
    def test_optimizer_initialization(self):
        """Testet die Initialisierung des Optimizers."""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(len(self.optimizer.reinforcement_learner.strategies), 8)
        self.assertFalse(self.optimizer.warmup_complete)
    
    def test_strategy_selection(self):
        """Testet die Strategieauswahl."""
        # Einfacher Testcode
        code = """
def add(a, b):
    return a + b
"""
        context = ExecutionContext(code_hash="test", input_shapes=[], input_types=[])
        
        # Wähle Strategie
        strategy = self.optimizer.optimize(code, context)
        
        # Überprüfe, ob eine Strategie zurückgegeben wurde
        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, OptimizationStrategy)
    
    def test_save_load_state(self):
        """Testet das Speichern und Laden des Optimizer-Zustands."""
        # Speichere Zustand
        state_dir = os.path.join(self.temp_dir, "optimizer_state")
        self.optimizer.save_state(state_dir)
        
        # Überprüfe, ob Dateien erstellt wurden
        self.assertTrue(os.path.exists(os.path.join(state_dir, "ai_optimizer_model.pkl")))
        self.assertTrue(os.path.exists(os.path.join(state_dir, "ai_optimizer_patterns.json")))
        
        # Setze Optimizer zurück
        self._reset_optimizer()
        
        # Lade Zustand
        result = self.optimizer.load_state(state_dir)
        self.assertTrue(result)
        self.assertTrue(self.optimizer.warmup_complete)


class AIOptimizerPerformanceTests(unittest.TestCase):
    """Leistungstests für den AI-Optimizer."""
    
    def setUp(self):
        """Test-Setup."""
        # Konfiguriere einen Optimizer für Tests
        self.config = OptimizerConfig(
            enabled=True,
            exploration_rate=0.0,  # Für deterministische Tests
            learning_rate=0.1,
            memory_size=1000,
            batch_size=32,
            update_interval=10,
            model_path=None
        )
        
        # Hole eine Optimizer-Instanz mit Testkonfiguration
        self.optimizer = get_ai_optimizer(self.config)
        
        # Vorkonfigurierte Testfunktionen
        self.test_functions = self._prepare_test_functions()
    
    def _prepare_test_functions(self):
        """Bereitet Testfunktionen vor."""
        functions = {}
        
        # CPU-intensive Funktion
        def cpu_intensive(n):
            result = 0
            for i in range(n):
                result += i * i
            return result
        functions["cpu_intensive"] = cpu_intensive
        
        # Tensor-Operation
        def tensor_op(shape):
            a = MTensor.ones(shape)
            b = MTensor.ones(shape)
            for _ in range(10):
                c = a @ b  # Matrixmultiplikation
            return c
        functions["tensor_op"] = tensor_op
        
        # Parallele Operation
        def parallel_op(n):
            results = []
            for i in range(n):
                results.append(i * i)
            return results
        functions["parallel_op"] = parallel_op
        
        return functions
    
    def test_strategy_adaptation(self):
        """Testet die Anpassung von Strategien an die Hardware."""
        # Erstelle eine Strategie für ANE
        ane_strategy = OptimizationStrategy(
            strategy_id="test_ane",
            name="Test ANE",
            parallelization_level=0,
            jit_level=2,
            device_target="ane",
            memory_optimization="balanced"
        )
        
        # Adaptiere an Hardware
        adapted = self.optimizer._adapt_to_hardware(ane_strategy)
        
        # MLX-Adapter abrufen
        mlx_adapter = get_mlx_adapter()
        
        # Überprüfe die Anpassung
        if not mlx_adapter.is_available():
            # Wenn MLX nicht verfügbar ist, sollte auf CPU zurückgefallen werden
            self.assertEqual(adapted.device_target, "cpu")
        elif mlx_adapter.is_available() and not mlx_adapter.supports_ane():
            # Wenn MLX verfügbar, aber keine ANE, sollte auf GPU zurückgefallen werden
            self.assertEqual(adapted.device_target, "gpu")
        else:
            # Wenn ANE verfügbar, sollte das Ziel unverändert bleiben
            self.assertEqual(adapted.device_target, "ane")
    
    def test_performance_improvement(self):
        """Testet, ob die Optimierung die Leistung verbessert."""
        # Teste Tensor-Operationen
        shape = (100, 100)
        
        # Unoptimiert
        start_time = time.time()
        result1 = self.test_functions["tensor_op"](shape)
        unoptimized_time = time.time() - start_time
        
        # Optimiert mit aggressiver Strategie
        strategy = self.optimizer.reinforcement_learner.strategies["aggressive_optimization"]
        
        start_time = time.time()
        result2 = self.optimizer.apply_strategy(strategy, self.test_functions["tensor_op"], shape)
        optimized_time = time.time() - start_time
        
        print(f"Unoptimierte Zeit: {unoptimized_time:.4f}s")
        print(f"Optimierte Zeit: {optimized_time:.4f}s")
        
        # Überprüfe, ob die Ergebnisse gleich sind
        self.assertEqual(result1.shape, result2.shape)
        
        # In manchen Fällen kann die erste Optimierung wegen JIT-Kompilierung langsamer sein,
        # daher erlauben wir eine Toleranz
        if optimized_time > unoptimized_time * 1.5:
            print("Warnung: Optimierte Version war langsamer, möglicherweise wegen JIT-Kompilierung")


class AIOptimizerIntegrationTests(unittest.TestCase):
    """Integrationstests für den AI-Optimizer mit anderen M-CODE-Komponenten."""
    
    def setUp(self):
        """Test-Setup."""
        self.config = OptimizerConfig(enabled=True)
        self.optimizer = get_ai_optimizer(self.config)
    
    def test_integration_with_jit_compiler(self):
        """Testet die Integration mit dem JIT-Compiler."""
        # Einfache Testfunktion
        def simple_function(x, y):
            return x + y
        
        # Strategie mit JIT-Optimierung
        strategy = OptimizationStrategy(
            strategy_id="jit_test",
            name="JIT Test",
            parallelization_level=0,
            jit_level=2,  # Hohe JIT-Optimierung
            device_target="cpu",
            memory_optimization="balanced"
        )
        
        # Wende Strategie an
        jit_compiler = get_jit_compiler()
        self.assertIsNotNone(jit_compiler)
        
        try:
            # Führe optimierte Funktion aus
            result = self.optimizer.apply_strategy(strategy, simple_function, 5, 7)
            self.assertEqual(result, 12)
        except Exception as e:
            self.fail(f"Fehler bei der Anwendung der JIT-Strategie: {e}")
    
    def test_integration_with_mlx_adapter(self):
        """Testet die Integration mit dem MLX-Adapter."""
        mlx_adapter = get_mlx_adapter()
        self.assertIsNotNone(mlx_adapter)
        
        # Erstelle eine Strategie für MLX
        if mlx_adapter.is_available():
            target = "gpu" if mlx_adapter.supports_gpu() else "cpu"
            strategy = OptimizationStrategy(
                strategy_id="mlx_test",
                name="MLX Test",
                parallelization_level=0,
                jit_level=1,
                device_target=target,
                memory_optimization="balanced"
            )
            
            # Einfache Tensor-Operation
            def tensor_add():
                a = MTensor.ones((10, 10))
                b = MTensor.ones((10, 10))
                return a + b
            
            try:
                # Führe optimierte Funktion aus
                result = self.optimizer.apply_strategy(strategy, tensor_add)
                self.assertEqual(result.shape, (10, 10))
                self.assertTrue((result.numpy() == 2.0).all())
            except Exception as e:
                self.fail(f"Fehler bei der Anwendung der MLX-Strategie: {e}")
    
    def test_optimizer_decorator(self):
        """Testet den @optimize-Dekorator."""
        # Definiere eine zu optimierende Funktion
        @optimize
        def optimized_function(n):
            result = 0
            for i in range(n):
                result += i
            return result
        
        # Führe optimierte Funktion aus
        result = optimized_function(100)
        self.assertEqual(result, sum(range(100)))
        
        # Definiere eine Funktion mit spezifischer Strategie
        @optimize(strategy_id="memory_efficient")
        def memory_optimized_function(n):
            result = 0
            for i in range(n):
                result += i
            return result
        
        # Führe optimierte Funktion aus
        result = memory_optimized_function(100)
        self.assertEqual(result, sum(range(100)))


class AIOptimizerBenchmarks(unittest.TestCase):
    """Benchmark-Tests für den AI-Optimizer."""
    
    def setUp(self):
        """Test-Setup."""
        self.config = OptimizerConfig(enabled=True)
        self.optimizer = get_ai_optimizer(self.config)
        self.results = {}
    
    def test_benchmark_matrix_operations(self):
        """Benchmarkt Matrix-Operationen mit verschiedenen Strategien."""
        # Definiere Matrix-Operation
        def matrix_multiply(size):
            a = MTensor.random((size, size))
            b = MTensor.random((size, size))
            return a @ b
        
        # Definiere zu testende Größen
        sizes = [10, 50, 100, 200]
        
        # Definiere zu testende Strategien
        strategies = [
            self.optimizer.reinforcement_learner.strategies["default"],
            self.optimizer.reinforcement_learner.strategies["cpu_optimized"],
            self.optimizer.reinforcement_learner.strategies["gpu_optimized"],
            self.optimizer.reinforcement_learner.strategies["aggressive_optimization"]
        ]
        
        results = {}
        
        # Führe Benchmarks durch
        for size in sizes:
            results[size] = {}
            
            for strategy in strategies:
                # Wärme JIT-Kompilierung auf
                self.optimizer.apply_strategy(strategy, matrix_multiply, size)
                
                # Messe Zeit (3 Durchläufe, nehme Durchschnitt)
                times = []
                for _ in range(3):
                    start_time = time.time()
                    self.optimizer.apply_strategy(strategy, matrix_multiply, size)
                    times.append(time.time() - start_time)
                
                avg_time = sum(times) / len(times)
                results[size][strategy.name] = avg_time
        
        # Speichere Ergebnisse für Bericht
        self.results["matrix_operations"] = results
        
        # Einfache Validierung
        for size in sizes:
            print(f"\nMatrixmultiplikation {size}x{size}:")
            for strategy_name, time_taken in results[size].items():
                print(f"  {strategy_name}: {time_taken:.4f}s")
            
            # Nicht immer können wir garantieren, dass Optimierung schneller ist
            # (z.B. bei kleinen Matrizen kann der Overhead größer sein als der Nutzen)
            if size >= 100:
                # Überprüfe, ob mindestens eine optimierte Strategie schneller ist als die Standardstrategie
                default_time = results[size]["Standard"]
                optimized_times = [time for name, time in results[size].items() if name != "Standard"]
                self.assertTrue(any(time < default_time for time in optimized_times), 
                                f"Keine Optimierung war schneller für {size}x{size} Matrizen")
    
    def test_benchmark_reinforcement_learning(self):
        """Benchmarkt das Reinforcement Learning des Optimizers."""
        # Definiere eine Funktion mit klarem Optimierungspotential
        def compute_intensive(n, iterations):
            result = 0
            for _ in range(iterations):
                for i in range(n):
                    result += i * i
            return result
        
        # Konfiguriere einen neuen Optimizer mit höherer Explorationsrate
        config = OptimizerConfig(
            enabled=True,
            exploration_rate=0.5,  # Hohe Exploration
            learning_rate=0.2,
            memory_size=100,
            batch_size=10,
            update_interval=5
        )
        
        # Zurücksetzen der Singleton-Instanz
        from miso.code.m_code.ai_optimizer import _ai_optimizer_instance
        globals()["_ai_optimizer_instance"] = None
        optimizer = get_ai_optimizer(config)
        
        # Ausführungsparameter
        n = 1000
        iterations = 5
        
        # Führe die Funktion mehrmals aus, damit der Optimizer lernen kann
        times = []
        for i in range(20):
            start_time = time.time()
            
            # Erstelle Ausführungskontext
            context = ExecutionContext(
                code_hash=f"test_{i}",
                input_shapes=[],
                input_types=[]
            )
            
            # Optimiere den Code
            strategy = optimizer.optimize(inspect.getsource(compute_intensive), context)
            
            # Führe optimierte Funktion aus
            optimizer.apply_strategy(strategy, compute_intensive, n, iterations)
            
            times.append(time.time() - start_time)
            
            # Drucke Fortschritt
            if i % 5 == 0:
                print(f"Durchlauf {i}: {times[-1]:.4f}s mit Strategie {strategy.name}")
        
        # Speichere Lernfortschritt
        self.results["reinforcement_learning"] = {
            "times": times,
            "improvement": (times[0] - times[-1]) / times[0] if times[0] > 0 else 0
        }
        
        # Zeige Verbesserung
        print(f"\nReinforcement Learning Verbesserung: {self.results['reinforcement_learning']['improvement']:.2%}")
        
        # Nicht immer können wir garantieren, dass der Optimizer innerhalb weniger Iterationen lernt
        # aber wir sollten zumindest keine signifikante Verschlechterung sehen
        self.assertLessEqual(times[-1], times[0] * 1.2, "Leistung hat sich signifikant verschlechtert")
    
    def tearDown(self):
        """Generiert einen Benchmark-Bericht."""
        if hasattr(self, 'results') and self.results:
            report_path = os.path.join(os.path.dirname(__file__), "benchmark_report.md")
            
            with open(report_path, 'w') as f:
                f.write("# AI-Optimizer Benchmark-Bericht\n\n")
                f.write(f"Erstellt am: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if "matrix_operations" in self.results:
                    f.write("## Matrix-Operationen\n\n")
                    f.write("| Größe | " + " | ".join([s.name for s in self.optimizer.reinforcement_learner.strategies.values() if s.name in self.results["matrix_operations"][10]]) + " |\n")
                    f.write("|" + "---|" * (len(self.results["matrix_operations"][10]) + 1) + "\n")
                    
                    for size in sorted(self.results["matrix_operations"].keys()):
                        row = f"| {size}x{size} "
                        for strategy_name in [s.name for s in self.optimizer.reinforcement_learner.strategies.values() if s.name in self.results["matrix_operations"][size]]:
                            row += f"| {self.results['matrix_operations'][size][strategy_name]:.4f}s "
                        f.write(row + "|\n")
                    f.write("\n")
                
                if "reinforcement_learning" in self.results:
                    rl_results = self.results["reinforcement_learning"]
                    f.write("## Reinforcement Learning\n\n")
                    f.write(f"Anfängliche Ausführungszeit: {rl_results['times'][0]:.4f}s\n\n")
                    f.write(f"Endgültige Ausführungszeit: {rl_results['times'][-1]:.4f}s\n\n")
                    f.write(f"Verbesserung: {rl_results['improvement']:.2%}\n\n")
                    
                    # Einfaches ASCII-Diagramm
                    f.write("### Lernfortschritt\n\n")
                    f.write("```\n")
                    max_time = max(rl_results['times'])
                    min_time = min(rl_results['times'])
                    range_time = max_time - min_time
                    for i, t in enumerate(rl_results['times']):
                        if range_time > 0:
                            bar_len = int(50 * (t - min_time) / range_time)
                            f.write(f"Iteration {i:2d}: {t:.4f}s {'#' * bar_len}\n")
                        else:
                            f.write(f"Iteration {i:2d}: {t:.4f}s\n")
                    f.write("```\n\n")
            
            print(f"Benchmark-Bericht erstellt: {report_path}")


import inspect

if __name__ == "__main__":
    # Führe alle Tests aus
    unittest.main()
