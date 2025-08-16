#!/usr/bin/env python3
"""
AI-Optimizer Demo für M-CODE Core.

Diese Demo zeigt die fortschrittlichen Optimierungsfähigkeiten des AI-Optimizers für den M-CODE Core.
Sie demonstriert die selbstadaptive Optimierung, MLX-Integration und Lernfähigkeiten.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
import argparse

# Pfad zum MISO-Paket hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Importiere MISO-Komponenten
from miso.code.m_code.runtime import MCodeRuntime
from miso.code.m_code.ai_optimizer import (
    get_ai_optimizer, 
    optimize, 
    OptimizerConfig,
    OptimizationStrategy
)
from miso.code.m_code.mlx_adapter import get_mlx_adapter
from miso.code.m_code.tensor import MTensor
from miso.code.m_code.debug_profiler import profile, get_profiler


class AIOptimizerDemo:
    """Demonstriert die fortschrittlichen Fähigkeiten des AI-Optimizers."""
    
    def __init__(self, optimize_enabled: bool = True):
        """
        Initialisiert die Demo.
        
        Args:
            optimize_enabled: Ob die KI-Optimierung aktiviert werden soll
        """
        # Initialisiere M-CODE-Runtime
        self.runtime = MCodeRuntime(debug=True)
        
        # Konfiguriere AI-Optimizer
        optimizer_config = OptimizerConfig(
            enabled=optimize_enabled,
            exploration_rate=0.3,
            learning_rate=0.1,
            memory_size=1000,
            batch_size=32,
            update_interval=5
        )
        self.ai_optimizer = get_ai_optimizer(optimizer_config)
        
        # Initialisiere Messwerte
        self.benchmark_results = {}
        
        print(f"AI-Optimizer Demo initialisiert (Optimierung: {'aktiviert' if optimize_enabled else 'deaktiviert'})")
        print(f"MLX Status: {'verfügbar' if get_mlx_adapter().is_available() else 'nicht verfügbar'}")
        print(f"Neural Engine: {'unterstützt' if get_mlx_adapter().supports_ane() else 'nicht unterstützt'}")
    
    def run_matrix_benchmark(self, sizes: List[int] = None, repetitions: int = 3) -> Dict[str, Dict[int, float]]:
        """
        Führt einen Benchmark für Matrixoperationen durch.
        
        Args:
            sizes: Liste von Matrixgrößen
            repetitions: Anzahl der Wiederholungen pro Größe
            
        Returns:
            Benchmark-Ergebnisse
        """
        if sizes is None:
            sizes = [10, 50, 100, 200, 500]
        
        results = {"optimized": {}, "standard": {}}
        
        print("\n=== Matrix-Multiplikations-Benchmark ===")
        print("Größe | Standard (ms) | Optimiert (ms) | Verbesserung")
        print("-----|--------------|---------------|------------")
        
        for size in sizes:
            # Generiere Testmatrizen
            a = MTensor.random((size, size))
            b = MTensor.random((size, size))
            
            # Standard-Ausführung (ohne Optimierung)
            times_standard = []
            for _ in range(repetitions):
                start_time = time.time()
                c_standard = a @ b
                times_standard.append((time.time() - start_time) * 1000)  # ms
            
            avg_time_standard = sum(times_standard) / len(times_standard)
            results["standard"][size] = avg_time_standard
            
            # Optimierte Ausführung mit AI-Optimizer
            @optimize
            def optimized_matmul(a, b):
                return a @ b
            
            times_optimized = []
            for _ in range(repetitions):
                start_time = time.time()
                c_optimized = optimized_matmul(a, b)
                times_optimized.append((time.time() - start_time) * 1000)  # ms
            
            avg_time_optimized = sum(times_optimized) / len(times_optimized)
            results["optimized"][size] = avg_time_optimized
            
            # Berechne Verbesserung
            if avg_time_standard > 0:
                improvement = (avg_time_standard - avg_time_optimized) / avg_time_standard * 100
                improvement_str = f"{improvement:.1f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{size:4d} | {avg_time_standard:12.2f} | {avg_time_optimized:13.2f} | {improvement_str}")
        
        self.benchmark_results["matrix"] = results
        return results
    
    def run_mcode_benchmark(self, code_complexity: List[int] = None, repetitions: int = 3) -> Dict[str, Dict[int, float]]:
        """
        Führt einen Benchmark für M-CODE-Ausführung durch.
        
        Args:
            code_complexity: Liste von Komplexitätsstufen
            repetitions: Anzahl der Wiederholungen pro Komplexität
            
        Returns:
            Benchmark-Ergebnisse
        """
        if code_complexity is None:
            code_complexity = [10, 50, 100, 200]
        
        results = {"optimized": {}, "standard": {}}
        
        print("\n=== M-CODE-Ausführungs-Benchmark ===")
        print("Komplexität | Standard (ms) | Optimiert (ms) | Verbesserung")
        print("------------|--------------|---------------|------------")
        
        for complexity in code_complexity:
            # Generiere M-CODE mit entsprechender Komplexität
            mcode = self._generate_test_mcode(complexity)
            
            # Standard-Ausführung (ohne Optimierung)
            times_standard = []
            for _ in range(repetitions):
                start_time = time.time()
                result_standard = self.runtime.execute(mcode, optimize_execution=False)
                times_standard.append((time.time() - start_time) * 1000)  # ms
            
            avg_time_standard = sum(times_standard) / len(times_standard)
            results["standard"][complexity] = avg_time_standard
            
            # Optimierte Ausführung
            times_optimized = []
            for _ in range(repetitions):
                start_time = time.time()
                result_optimized = self.runtime.execute(mcode, optimize_execution=True)
                times_optimized.append((time.time() - start_time) * 1000)  # ms
            
            avg_time_optimized = sum(times_optimized) / len(times_optimized)
            results["optimized"][complexity] = avg_time_optimized
            
            # Berechne Verbesserung
            if avg_time_standard > 0:
                improvement = (avg_time_standard - avg_time_optimized) / avg_time_standard * 100
                improvement_str = f"{improvement:.1f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{complexity:11d} | {avg_time_standard:12.2f} | {avg_time_optimized:13.2f} | {improvement_str}")
        
        self.benchmark_results["mcode"] = results
        return results
    
    def _generate_test_mcode(self, complexity: int) -> str:
        """
        Generiert M-CODE-Testcode mit der angegebenen Komplexität.
        
        Args:
            complexity: Komplexität des Codes (mehr Operationen)
            
        Returns:
            M-CODE als String
        """
        # Einfacher M-CODE mit Tensor-Operationen und Schleifen,
        # wobei die Komplexität durch die Anzahl der Operationen bestimmt wird
        mcode = f"""
function test_function()
    # Erstelle Tensoren
    a = tensor.ones(100, 100)
    b = tensor.ones(100, 100)
    c = tensor.zeros(100, 100)
    
    # Führe Operationen aus (Komplexität = {complexity})
    for i in range(0, {complexity})
        c = c + a * b
        if i % 10 == 0
            c = c / 2.0
        end
    end
    
    return c
end

# Rufe Funktion auf
result = test_function()
"""
        return mcode
    
    def run_echo_prime_integration_demo(self) -> None:
        """Demonstriert die Integration mit ECHO-PRIME."""
        print("\n=== ECHO-PRIME-Integration Demo ===")
        
        # Zuerst importieren wir die ECHO-PRIME-Komponenten
        try:
            from miso.code.m_code.echo_prime_integration import (
                EchoPrimeIntegration, 
                create_timeline_processor
            )
            
            # Erstelle eine ECHO-PRIME-Integration
            echo_prime = EchoPrimeIntegration()
            
            # Erstelle einen Timeline-Processor mit und ohne Optimierung
            print("Erstelle Timeline-Prozessor...")
            
            # Ohne Optimierung
            start_time = time.time()
            processor_standard = create_timeline_processor(optimize_enabled=False)
            time_standard = (time.time() - start_time) * 1000
            
            # Mit Optimierung
            start_time = time.time()
            processor_optimized = create_timeline_processor(optimize_enabled=True)
            time_optimized = (time.time() - start_time) * 1000
            
            print(f"Erstellungszeit ohne Optimierung: {time_standard:.2f} ms")
            print(f"Erstellungszeit mit Optimierung: {time_optimized:.2f} ms")
            
            # Simuliere eine Timeline-Verarbeitung
            print("\nSimuliere Timeline-Verarbeitung...")
            
            # Erstelle eine Testzeitlinie
            timeline_data = {
                "nodes": [
                    {"id": "node1", "time": 0, "data": {"value": np.random.rand(50, 50)}},
                    {"id": "node2", "time": 1, "data": {"value": np.random.rand(50, 50)}},
                    {"id": "node3", "time": 2, "data": {"value": np.random.rand(50, 50)}}
                ],
                "edges": [
                    {"source": "node1", "target": "node2", "type": "causal"},
                    {"source": "node2", "target": "node3", "type": "temporal"}
                ]
            }
            
            # Verarbeite ohne Optimierung
            start_time = time.time()
            result_standard = processor_standard.process_timeline(timeline_data)
            time_standard = (time.time() - start_time) * 1000
            
            # Verarbeite mit Optimierung
            start_time = time.time()
            result_optimized = processor_optimized.process_timeline(timeline_data)
            time_optimized = (time.time() - start_time) * 1000
            
            print(f"Verarbeitungszeit ohne Optimierung: {time_standard:.2f} ms")
            print(f"Verarbeitungszeit mit Optimierung: {time_optimized:.2f} ms")
            
            if time_standard > 0:
                improvement = (time_standard - time_optimized) / time_standard * 100
                print(f"Verbesserung: {improvement:.1f}%")
            
            # Speichere die Ergebnisse
            self.benchmark_results["echo_prime"] = {
                "standard": time_standard,
                "optimized": time_optimized
            }
            
        except ImportError as e:
            print(f"ECHO-PRIME-Integration nicht verfügbar: {e}")
            print("Überspringen der ECHO-PRIME-Demo.")
    
    def run_learning_demo(self, iterations: int = 20) -> Dict[str, List[float]]:
        """
        Demonstriert die Lernfähigkeiten des AI-Optimizers.
        
        Args:
            iterations: Anzahl der Lerniterationen
            
        Returns:
            Ergebnisse des Lernprozesses
        """
        print("\n=== AI-Optimizer Lernprozess-Demo ===")
        print("Iteration | Zeit (ms) | Verwendete Strategie")
        print("----------|-----------|-------------------")
        
        # Definiere eine Funktion mit klarem Optimierungspotential
        @profile(name="learning_demo_function")
        def compute_intensive(n, iterations):
            """Rechenintensive Funktion für die Demo."""
            result = MTensor.zeros((n, n))
            for _ in range(iterations):
                a = MTensor.random((n, n))
                b = MTensor.random((n, n))
                result = result + (a @ b)
            return result
        
        # Parameter
        n = 100
        iter_count = 5
        
        # Führe die Funktion mehrmals aus, damit der Optimizer lernen kann
        times = []
        strategies = []
        
        for i in range(iterations):
            # Code als String extrahieren (simuliert verschiedene Eingaben)
            import inspect
            code_str = inspect.getsource(compute_intensive)
            
            # Erstelle Ausführungskontext
            from miso.code.m_code.ai_optimizer import ExecutionContext
            context = ExecutionContext(
                code_hash=f"learn_{i}",
                input_shapes=[(n, n)],
                input_types=["float32"]
            )
            
            # Optimiere den Code
            strategy = self.ai_optimizer.optimize(code_str, context)
            strategies.append(strategy.name)
            
            # Führe optimierte Funktion aus
            start_time = time.time()
            self.ai_optimizer.apply_strategy(strategy, compute_intensive, n, iter_count)
            execution_time = (time.time() - start_time) * 1000  # ms
            times.append(execution_time)
            
            # Ausgabe der Ergebnisse
            print(f"{i:9d} | {execution_time:8.2f} | {strategy.name}")
            
            # Zwischenergebnis alle 5 Iterationen
            if i > 0 and i % 5 == 0:
                avg_first = sum(times[:5]) / 5
                avg_last = sum(times[-5:]) / 5
                if avg_first > 0:
                    improvement = (avg_first - avg_last) / avg_first * 100
                    print(f"Durchschnittliche Verbesserung nach {i} Iterationen: {improvement:.1f}%")
        
        # Speichere Lernfortschritt
        self.benchmark_results["learning"] = {
            "times": times,
            "strategies": strategies,
            "improvement": (times[0] - times[-1]) / times[0] if times[0] > 0 else 0
        }
        
        # Zeige Gesamtverbesserung
        print(f"\nGesamtverbesserung: {self.benchmark_results['learning']['improvement']:.2%}")
        
        return self.benchmark_results["learning"]
    
    def generate_report(self, output_path: str = "ai_optimizer_report.md") -> None:
        """
        Generiert einen Benchmark-Bericht.
        
        Args:
            output_path: Pfad zur Ausgabedatei
        """
        with open(output_path, 'w') as f:
            f.write("# AI-Optimizer Benchmark-Bericht\n\n")
            f.write(f"Erstellt am: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Systemkonfiguration\n\n")
            
            # Systeminfo
            import platform
            f.write(f"- Betriebssystem: {platform.system()} {platform.release()}\n")
            f.write(f"- Python-Version: {platform.python_version()}\n")
            f.write(f"- MLX verfügbar: {get_mlx_adapter().is_available()}\n")
            f.write(f"- Neural Engine unterstützt: {get_mlx_adapter().supports_ane()}\n\n")
            
            # Matrix-Benchmark
            if "matrix" in self.benchmark_results:
                matrix_results = self.benchmark_results["matrix"]
                f.write("## Matrix-Operationen\n\n")
                f.write("| Größe | Standard (ms) | Optimiert (ms) | Verbesserung |\n")
                f.write("|-------|--------------|---------------|-------------|\n")
                
                for size in sorted(matrix_results["standard"].keys()):
                    std_time = matrix_results["standard"][size]
                    opt_time = matrix_results["optimized"][size]
                    if std_time > 0:
                        improvement = (std_time - opt_time) / std_time * 100
                        improvement_str = f"{improvement:.1f}%"
                    else:
                        improvement_str = "N/A"
                    
                    f.write(f"| {size} | {std_time:.2f} | {opt_time:.2f} | {improvement_str} |\n")
                f.write("\n")
            
            # M-CODE-Benchmark
            if "mcode" in self.benchmark_results:
                mcode_results = self.benchmark_results["mcode"]
                f.write("## M-CODE-Ausführung\n\n")
                f.write("| Komplexität | Standard (ms) | Optimiert (ms) | Verbesserung |\n")
                f.write("|-------------|--------------|---------------|-------------|\n")
                
                for complexity in sorted(mcode_results["standard"].keys()):
                    std_time = mcode_results["standard"][complexity]
                    opt_time = mcode_results["optimized"][complexity]
                    if std_time > 0:
                        improvement = (std_time - opt_time) / std_time * 100
                        improvement_str = f"{improvement:.1f}%"
                    else:
                        improvement_str = "N/A"
                    
                    f.write(f"| {complexity} | {std_time:.2f} | {opt_time:.2f} | {improvement_str} |\n")
                f.write("\n")
            
            # ECHO-PRIME-Integration
            if "echo_prime" in self.benchmark_results:
                echo_results = self.benchmark_results["echo_prime"]
                f.write("## ECHO-PRIME-Integration\n\n")
                f.write("| Metrik | Standard (ms) | Optimiert (ms) | Verbesserung |\n")
                f.write("|--------|--------------|---------------|-------------|\n")
                
                std_time = echo_results["standard"]
                opt_time = echo_results["optimized"]
                if std_time > 0:
                    improvement = (std_time - opt_time) / std_time * 100
                    improvement_str = f"{improvement:.1f}%"
                else:
                    improvement_str = "N/A"
                
                f.write(f"| Timeline-Verarbeitung | {std_time:.2f} | {opt_time:.2f} | {improvement_str} |\n")
                f.write("\n")
            
            # Lernfortschritt
            if "learning" in self.benchmark_results:
                learning_results = self.benchmark_results["learning"]
                f.write("## Lernfortschritt\n\n")
                f.write(f"Anfängliche Ausführungszeit: {learning_results['times'][0]:.2f} ms\n\n")
                f.write(f"Endgültige Ausführungszeit: {learning_results['times'][-1]:.2f} ms\n\n")
                f.write(f"Gesamtverbesserung: {learning_results['improvement']:.2%}\n\n")
                
                # Strategien-Verteilung
                from collections import Counter
                strategy_counts = Counter(learning_results["strategies"])
                f.write("### Verwendete Strategien\n\n")
                f.write("| Strategie | Häufigkeit |\n")
                f.write("|-----------|------------|\n")
                for strategy, count in strategy_counts.most_common():
                    percentage = count / len(learning_results["strategies"]) * 100
                    f.write(f"| {strategy} | {count} ({percentage:.1f}%) |\n")
                f.write("\n")
                
                # Lernkurve
                f.write("### Lernkurve\n\n")
                f.write("```\n")
                times = learning_results["times"]
                max_time = max(times)
                min_time = min(times)
                range_time = max_time - min_time
                for i, t in enumerate(times):
                    if range_time > 0:
                        bar_len = int(50 * (t - min_time) / range_time)
                        bar = "#" * bar_len
                    else:
                        bar = ""
                    f.write(f"Iteration {i:2d}: {t:.2f} ms {bar}\n")
                f.write("```\n\n")
            
            # Profilierungsdaten
            f.write("## Profilierungsdaten\n\n")
            profiler = get_profiler()
            if profiler and profiler.profiles:
                f.write("| Funktion | Aufrufe | Gesamtzeit (ms) | Durchschnitt (ms) |\n")
                f.write("|----------|---------|-----------------|------------------|\n")
                
                for name, profile_data in profiler.profiles.items():
                    calls = profile_data["calls"]
                    total_time = profile_data["total_time"] * 1000  # ms
                    avg_time = total_time / calls if calls > 0 else 0
                    
                    f.write(f"| {name} | {calls} | {total_time:.2f} | {avg_time:.2f} |\n")
            else:
                f.write("Keine Profilierungsdaten verfügbar.\n")
        
        print(f"\nBenchmark-Bericht erstellt: {output_path}")


def main():
    """Haupteinstiegspunkt für die Demo."""
    parser = argparse.ArgumentParser(description="AI-Optimizer Demo für M-CODE Core")
    parser.add_argument("--optimize", action="store_true", help="Aktiviert die KI-Optimierung")
    parser.add_argument("--report", type=str, default="ai_optimizer_report.md", 
                       help="Pfad für den Benchmark-Bericht")
    parser.add_argument("--matrix", action="store_true", help="Führt Matrix-Benchmark durch")
    parser.add_argument("--mcode", action="store_true", help="Führt M-CODE-Benchmark durch")
    parser.add_argument("--echo", action="store_true", help="Führt ECHO-PRIME-Integration-Demo durch")
    parser.add_argument("--learning", action="store_true", help="Führt Lernprozess-Demo durch")
    parser.add_argument("--all", action="store_true", help="Führt alle Benchmarks durch")
    
    args = parser.parse_args()
    
    # Wenn keine spezifischen Tests ausgewählt, alle ausführen
    run_all = args.all or not (args.matrix or args.mcode or args.echo or args.learning)
    
    # Erstelle Demo-Instanz
    demo = AIOptimizerDemo(optimize_enabled=args.optimize)
    
    # Führe ausgewählte Benchmarks durch
    if run_all or args.matrix:
        demo.run_matrix_benchmark()
    
    if run_all or args.mcode:
        demo.run_mcode_benchmark()
    
    if run_all or args.echo:
        demo.run_echo_prime_integration_demo()
    
    if run_all or args.learning:
        demo.run_learning_demo()
    
    # Generiere Bericht
    demo.generate_report(args.report)


if __name__ == "__main__":
    main()
