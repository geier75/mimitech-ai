#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - PRISM-Engine MLX Performance Benchmark

Dieses Skript führt Benchmark-Tests für die MLX-Optimierung der PRISM-Engine durch
und vergleicht die Performance mit der Standard-Implementierung.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import platform
import json
from typing import Dict, List, Any, Optional, Tuple
from tabulate import tabulate

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("prism_mlx_benchmark")

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


class PrismMLXBenchmark:
    """
    Benchmark-Klasse für die MLX-Optimierung der PRISM-Engine.
    """
    
    def __init__(self, use_mlx: bool = True, precision: str = "float16"):
        """
        Initialisiert den Benchmark.
        
        Args:
            use_mlx: Ob MLX verwendet werden soll
            precision: Präzision für die Berechnungen
        """
        # Setze Umgebungsvariablen für die Tests
        os.environ["T_MATH_USE_MLX"] = "1" if use_mlx else "0"
        os.environ["T_MATH_PRECISION"] = precision
        
        # Erstelle T-Mathematics Engine
        self.math_engine = TMathEngine(
            config=TMathConfig(
                precision=precision,
                device="auto",
                optimize_for_apple_silicon=use_mlx
            )
        )
        
        # Erstelle PRISM-Engine
        self.prism_engine = PrismEngine(config={
            "t_math_engine": self.math_engine,
            "use_mlx": use_mlx,
            "precision": precision
        })
        
        # Speichere Konfiguration
        self.use_mlx = use_mlx
        self.precision = precision
        self.backend_name = "MLX" if use_mlx and has_mlx else "PyTorch"
        
        # Ergebnisse
        self.results = {}
        
        logger.info(f"Benchmark initialisiert mit {self.backend_name} Backend und {precision} Präzision")
    
    def benchmark_matrix_operations(self, dimensions: int = 3, size: int = 10, 
                                   num_iterations: int = 100) -> Dict[str, float]:
        """
        Führt Benchmark-Tests für Matrixoperationen durch.
        
        Args:
            dimensions: Anzahl der Dimensionen
            size: Größe jeder Dimension
            num_iterations: Anzahl der Iterationen
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        logger.info(f"Starte Matrix-Operationen Benchmark mit {dimensions}D Matrix der Größe {size}...")
        
        # Erstelle Matrix
        matrix = PrismMatrix(dimensions=dimensions, initial_size=size)
        
        # Fülle Matrix mit Zufallswerten
        for _ in range(num_iterations):
            # Setze Wert an zufälligen Koordinaten
            coords = tuple(np.random.randint(0, size) for _ in range(dimensions))
            matrix.set_data_point(coords, np.random.random())
        
        # Benchmark: Wahrscheinlichkeitsberechnung
        start_time = time.time()
        for _ in range(num_iterations):
            dim = np.random.randint(0, dimensions)
            matrix.get_probability_distribution(dimension=dim)
        prob_time = (time.time() - start_time) / num_iterations
        
        # Benchmark: Wahrscheinlichkeitsverteilung
        start_time = time.time()
        for _ in range(num_iterations):
            for dim in range(dimensions):
                matrix.get_probability_distribution(dimension=dim)
        dist_time = (time.time() - start_time) / num_iterations
        
        # Benchmark: Tensor-Statistiken
        start_time = time.time()
        for _ in range(10):  # Weniger Iterationen, da diese Operation teurer ist
            matrix.get_tensor_statistics()
        state_time = (time.time() - start_time) / 10
        
        # Speichere Ergebnisse
        results = {
            "get_probability": prob_time,
            "calculate_distribution": dist_time,
            "most_probable_state": state_time,
            "total": prob_time + dist_time + state_time
        }
        
        logger.info(f"Matrix-Operationen Benchmark abgeschlossen: {results}")
        return results
    
    def benchmark_monte_carlo(self, num_iterations: int = 1000, 
                             max_depth: int = 5) -> Dict[str, float]:
        """
        Führt Benchmark-Tests für Monte-Carlo-Simulationen durch.
        
        Args:
            num_iterations: Anzahl der Iterationen
            max_depth: Maximale Tiefe der Simulation
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        logger.info(f"Starte Monte-Carlo Benchmark mit {num_iterations} Iterationen...")
        
        # Erstelle eine Beispiel-Timeline für den Benchmark
        from miso.timeline import Timeline, TimelineType
        timeline = Timeline(
            id="benchmark_timeline",
            type=TimelineType.MAIN,
            name="Benchmark Timeline",
            description="Eine Timeline für Benchmark-Tests"
        )
        
        # Registriere die Timeline bei der PRISM-Engine
        self.prism_engine.register_timeline(timeline)
        timeline_id = timeline.id
        
        # Benchmark: Timeline-Simulation (ähnlich zu Monte-Carlo)
        start_time = time.time()
        results = self.prism_engine.simulate_timeline(
            timeline_id=timeline_id,
            steps=num_iterations,
            variation_factor=0.2
        )
        total_time = time.time() - start_time
        
        # Speichere Ergebnisse
        benchmark_results = {
            "total_time": total_time,
            "time_per_iteration": total_time / num_iterations,
            "iterations_per_second": num_iterations / total_time
        }
        
        logger.info(f"Monte-Carlo Benchmark abgeschlossen: {benchmark_results}")
        return benchmark_results
    
    def benchmark_reality_modulation(self, num_iterations: int = 100) -> Dict[str, float]:
        """
        Führt Benchmark-Tests für Realitätsmodulationen durch.
        
        Args:
            num_iterations: Anzahl der Iterationen
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        logger.info(f"Starte Realitätsmodulation Benchmark mit {num_iterations} Iterationen...")
        
        # Erstelle Testzustand
        test_state = {
            "timeline_id": "benchmark_timeline",
            "timeline_name": "Benchmark Timeline",
            "timeline_probability": 0.8,
            "node_count": 5,
            "current_timestamp": time.time()
        }
        
        # Benchmark: Realitätsmodulation (verwendet simulate_reality_fork)
        start_time = time.time()
        for _ in range(num_iterations):
            # Erzeuge zufällige Variationen
            variations = [
                {"probability_shift": np.random.random() * 0.2 - 0.1,
                 "stability_factor": np.random.random() * 0.5}
                for _ in range(3)  # Erzeuge 3 Variationen
            ]
            
            # Verwende simulate_reality_fork statt modulate_reality
            self.prism_engine.simulate_reality_fork(
                current_state=test_state,
                variations=variations,
                steps=10
            )
        total_time = time.time() - start_time
        
        # Speichere Ergebnisse
        results = {
            "total_time": total_time,
            "time_per_modulation": total_time / num_iterations,
            "modulations_per_second": num_iterations / total_time
        }
        
        logger.info(f"Realitätsmodulation Benchmark abgeschlossen: {results}")
        return results
    
    def benchmark_probability_analysis(self, num_states: int = 10, 
                                      num_iterations: int = 10) -> Dict[str, float]:
        """
        Führt Benchmark-Tests für Wahrscheinlichkeitsanalysen durch.
        
        Args:
            num_states: Anzahl der Zustände
            num_iterations: Anzahl der Iterationen
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        logger.info(f"Starte Wahrscheinlichkeitsanalyse Benchmark mit {num_states} Zuständen...")
        
        # Erstelle Testzustände
        test_states = []
        for i in range(num_states):
            test_states.append({
                "timeline_id": f"benchmark_timeline_{i}",
                "timeline_name": f"Benchmark Timeline {i}",
                "timeline_probability": np.random.random(),
                "node_count": np.random.randint(3, 10),
                "current_timestamp": time.time()
            })
        
        # Erstelle Testvariationen
        test_variations = [
            {"name": "Variation 1", "probability": 0.7, "factor": 1.2},
            {"name": "Variation 2", "probability": 0.5, "factor": 0.8},
            {"name": "Variation 3", "probability": 0.3, "factor": 1.5}
        ]
        
        # Benchmark: Wahrscheinlichkeitsanalyse
        start_time = time.time()
        for _ in range(num_iterations):
            # Für jeden Zustand die Wahrscheinlichkeit berechnen
            for state in test_states:
                timeline_id = state["timeline_id"]
                
                # Erstelle und registriere Timeline, falls noch nicht vorhanden
                if timeline_id not in self.prism_engine.get_registered_timeline_ids():
                    from miso.timeline import Timeline, TimelineType
                    timeline = Timeline(
                        id=timeline_id,
                        type=TimelineType.MAIN,
                        name=f"Benchmark Timeline {timeline_id}",
                        description=f"Eine Timeline für Benchmark-Tests: {timeline_id}"
                    )
                    self.prism_engine.register_timeline(timeline)
                
                # Berechne Wahrscheinlichkeit der Timeline
                prob = self.prism_engine.calculate_timeline_probability(timeline_id)
                
                # Analysiere Stabilität
                stability = self.prism_engine.analyze_timeline_stability(timeline_id)
                
                # Erkenne Paradoxa
                paradoxes = self.prism_engine.detect_paradoxes(timeline_id)
        total_time = time.time() - start_time
        
        # Speichere Ergebnisse
        results = {
            "total_time": total_time,
            "time_per_analysis": total_time / num_iterations,
            "analyses_per_second": num_iterations / total_time
        }
        
        logger.info(f"Wahrscheinlichkeitsanalyse Benchmark abgeschlossen: {results}")
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """
        Führt alle Benchmark-Tests durch.
        
        Returns:
            Dictionary mit allen Benchmark-Ergebnissen
        """
        logger.info(f"Starte alle Benchmarks mit {self.backend_name} Backend...")
        
        # Matrix-Operationen
        self.results["matrix_operations"] = self.benchmark_matrix_operations()
        
        # Monte-Carlo-Simulation
        self.results["monte_carlo"] = self.benchmark_monte_carlo()
        
        # Realitätsmodulation
        self.results["reality_modulation"] = self.benchmark_reality_modulation()
        
        # Wahrscheinlichkeitsanalyse
        self.results["probability_analysis"] = self.benchmark_probability_analysis()
        
        logger.info(f"Alle Benchmarks abgeschlossen")
        return self.results
    
    def save_results(self, output_file: str):
        """
        Speichert die Benchmark-Ergebnisse in einer Datei.
        
        Args:
            output_file: Pfad zur Ausgabedatei
        """
        # Füge Metadaten hinzu
        results_with_metadata = {
            "metadata": {
                "backend": self.backend_name,
                "precision": self.precision,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "timestamp": time.time()
            },
            "results": self.results
        }
        
        # Speichere als JSON
        with open(output_file, "w") as f:
            json.dump(results_with_metadata, f, indent=2)
        
        logger.info(f"Ergebnisse gespeichert in {output_file}")
    
    def print_results(self):
        """Gibt die Benchmark-Ergebnisse in tabellarischer Form aus."""
        if not self.results:
            logger.warning("Keine Ergebnisse vorhanden")
            return
        
        # Matrix-Operationen
        if "matrix_operations" in self.results:
            print("\n=== Matrix-Operationen ===")
            matrix_data = []
            for op, time_val in self.results["matrix_operations"].items():
                matrix_data.append([op, f"{time_val*1000:.2f} ms", f"{1/time_val:.2f} ops/s"])
            print(tabulate(matrix_data, headers=["Operation", "Zeit", "Durchsatz"], tablefmt="grid"))
        
        # Monte-Carlo-Simulation
        if "monte_carlo" in self.results:
            print("\n=== Monte-Carlo-Simulation ===")
            mc_data = []
            for metric, value in self.results["monte_carlo"].items():
                if metric == "time_per_iteration":
                    mc_data.append([metric, f"{value*1000:.2f} ms", ""])
                elif metric == "iterations_per_second":
                    mc_data.append([metric, "", f"{value:.2f} iter/s"])
                else:
                    mc_data.append([metric, f"{value:.2f} s", ""])
            print(tabulate(mc_data, headers=["Metrik", "Zeit", "Durchsatz"], tablefmt="grid"))
        
        # Realitätsmodulation
        if "reality_modulation" in self.results:
            print("\n=== Realitätsmodulation ===")
            rm_data = []
            for metric, value in self.results["reality_modulation"].items():
                if metric == "time_per_modulation":
                    rm_data.append([metric, f"{value*1000:.2f} ms", ""])
                elif metric == "modulations_per_second":
                    rm_data.append([metric, "", f"{value:.2f} mod/s"])
                else:
                    rm_data.append([metric, f"{value:.2f} s", ""])
            print(tabulate(rm_data, headers=["Metrik", "Zeit", "Durchsatz"], tablefmt="grid"))
        
        # Wahrscheinlichkeitsanalyse
        if "probability_analysis" in self.results:
            print("\n=== Wahrscheinlichkeitsanalyse ===")
            pa_data = []
            for metric, value in self.results["probability_analysis"].items():
                if metric == "time_per_analysis":
                    pa_data.append([metric, f"{value*1000:.2f} ms", ""])
                elif metric == "analyses_per_second":
                    pa_data.append([metric, "", f"{value:.2f} ana/s"])
                else:
                    pa_data.append([metric, f"{value:.2f} s", ""])
            print(tabulate(pa_data, headers=["Metrik", "Zeit", "Durchsatz"], tablefmt="grid"))


def compare_backends(output_dir: str = None):
    """
    Vergleicht die Performance von MLX und PyTorch.
    
    Args:
        output_dir: Verzeichnis für die Ausgabedateien
    """
    logger.info("Starte Vergleich zwischen MLX und PyTorch...")
    
    # Erstelle Ausgabeverzeichnis, falls es nicht existiert
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Führe Benchmarks mit MLX durch (falls verfügbar)
    if has_mlx and is_apple_silicon:
        mlx_benchmark = PrismMLXBenchmark(use_mlx=True, precision="float16")
        mlx_results = mlx_benchmark.run_all_benchmarks()
        mlx_benchmark.print_results()
        
        if output_dir:
            mlx_benchmark.save_results(os.path.join(output_dir, "prism_mlx_results.json"))
    else:
        logger.warning("MLX ist nicht verfügbar oder kein Apple Silicon gefunden")
        mlx_results = {}
    
    # Führe Benchmarks mit PyTorch durch
    torch_benchmark = PrismMLXBenchmark(use_mlx=False, precision="float32")
    torch_results = torch_benchmark.run_all_benchmarks()
    torch_benchmark.print_results()
    
    if output_dir:
        torch_benchmark.save_results(os.path.join(output_dir, "prism_torch_results.json"))
    
    # Vergleiche die Ergebnisse
    if mlx_results and torch_results:
        print("\n=== Vergleich: MLX vs. PyTorch ===")
        comparison_data = []
        
        # Matrix-Operationen
        if "matrix_operations" in mlx_results and "matrix_operations" in torch_results:
            mlx_total = mlx_results["matrix_operations"]["total"]
            torch_total = torch_results["matrix_operations"]["total"]
            speedup = torch_total / mlx_total if mlx_total > 0 else float('inf')
            comparison_data.append(["Matrix-Operationen", f"{mlx_total*1000:.2f} ms", 
                                  f"{torch_total*1000:.2f} ms", f"{speedup:.2f}x"])
        
        # Monte-Carlo-Simulation
        if "monte_carlo" in mlx_results and "monte_carlo" in torch_results:
            mlx_total = mlx_results["monte_carlo"]["total_time"]
            torch_total = torch_results["monte_carlo"]["total_time"]
            speedup = torch_total / mlx_total if mlx_total > 0 else float('inf')
            comparison_data.append(["Monte-Carlo", f"{mlx_total:.2f} s", 
                                  f"{torch_total:.2f} s", f"{speedup:.2f}x"])
        
        # Realitätsmodulation
        if "reality_modulation" in mlx_results and "reality_modulation" in torch_results:
            mlx_total = mlx_results["reality_modulation"]["total_time"]
            torch_total = torch_results["reality_modulation"]["total_time"]
            speedup = torch_total / mlx_total if mlx_total > 0 else float('inf')
            comparison_data.append(["Realitätsmodulation", f"{mlx_total:.2f} s", 
                                  f"{torch_total:.2f} s", f"{speedup:.2f}x"])
        
        # Wahrscheinlichkeitsanalyse
        if "probability_analysis" in mlx_results and "probability_analysis" in torch_results:
            mlx_total = mlx_results["probability_analysis"]["total_time"]
            torch_total = torch_results["probability_analysis"]["total_time"]
            speedup = torch_total / mlx_total if mlx_total > 0 else float('inf')
            comparison_data.append(["Wahrscheinlichkeitsanalyse", f"{mlx_total:.2f} s", 
                                  f"{torch_total:.2f} s", f"{speedup:.2f}x"])
        
        print(tabulate(comparison_data, 
                      headers=["Benchmark", "MLX Zeit", "PyTorch Zeit", "Speedup"], 
                      tablefmt="grid"))
        
        # Speichere Vergleichsergebnisse
        if output_dir:
            comparison_results = {
                "metadata": {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "python_version": platform.python_version(),
                    "timestamp": time.time()
                },
                "comparison": {category: {"mlx": mlx_results.get(category, {}), 
                                        "torch": torch_results.get(category, {})}
                              for category in set(mlx_results.keys()) | set(torch_results.keys())}
            }
            
            with open(os.path.join(output_dir, "prism_benchmark_comparison.json"), "w") as f:
                json.dump(comparison_results, f, indent=2)
            
            logger.info(f"Vergleichsergebnisse gespeichert in {os.path.join(output_dir, 'prism_benchmark_comparison.json')}")
    
    logger.info("Vergleich abgeschlossen")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM-Engine MLX Performance Benchmark")
    parser.add_argument("--output-dir", type=str, help="Verzeichnis für die Ausgabedateien")
    parser.add_argument("--backend", type=str, choices=["mlx", "torch", "both"], default="both",
                       help="Backend für den Benchmark (mlx, torch oder both)")
    parser.add_argument("--precision", type=str, choices=["float16", "float32"], default="float16",
                       help="Präzision für die Berechnungen")
    
    args = parser.parse_args()
    
    if args.backend == "both":
        compare_backends(args.output_dir)
    else:
        use_mlx = args.backend == "mlx"
        benchmark = PrismMLXBenchmark(use_mlx=use_mlx, precision=args.precision)
        benchmark.run_all_benchmarks()
        benchmark.print_results()
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = os.path.join(args.output_dir, f"prism_{args.backend}_results.json")
            benchmark.save_results(output_file)
