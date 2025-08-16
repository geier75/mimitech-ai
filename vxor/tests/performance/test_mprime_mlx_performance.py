#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leistungsprofilierung der MPRIME-Engine mit MLX-Optimierungen vs. PyTorch.

Dieses Script misst die Performance der MPRIME-Engine bei der Verarbeitung komplexer
mathematischer Ausdrücke unter Verwendung von MLX-optimierten Tensor-Operationen im 
Vergleich zu den Standard-PyTorch-Implementierungen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import argparse
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum, auto

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Konfiguriere das Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.Performance.MLX")

# Importiere MPRIME-Engine Komponenten
from miso.math.mprime.contextual_math import ContextualMathCore
from miso.math.mprime_engine import MPrimeEngine
from miso.math.mprime.symbol_solver import SymbolTree, get_symbol_solver

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
    logger.info("MLX ist verfügbar (Version: %s)", getattr(mx, "__version__", "unbekannt"))
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX ist nicht verfügbar, nur PyTorch-Benchmarks werden ausgeführt")

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch ist verfügbar (Version: %s)", torch.__version__)
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch ist nicht verfügbar, nur MLX-Benchmarks werden ausgeführt (falls verfügbar)")

class BenchmarkType(Enum):
    """Typ der Benchmark-Tests"""
    SIMPLE = auto()       # Einfache arithmetische Ausdrücke
    COMPLEX = auto()      # Komplexe mathematische Ausdrücke
    MATRIX = auto()       # Matrix-Operationen
    DIFFERENTIAL = auto() # Differentialgleichungen
    INTEGRAL = auto()     # Integrale
    SYSTEM = auto()       # Gleichungssysteme

class MathEngineBackend(Enum):
    """Verfügbare Backend-Engines für mathematische Berechnungen"""
    PYTORCH = auto()
    MLX = auto()
    NUMPY = auto()
    CPU = auto()

class BenchmarkResult:
    """Ergebnisse eines Benchmark-Tests"""
    
    def __init__(self, benchmark_type: BenchmarkType, backend: MathEngineBackend):
        self.benchmark_type = benchmark_type
        self.backend = backend
        self.iterations = 0
        self.execution_times = []
        self.average_time = 0.0
        self.min_time = 0.0
        self.max_time = 0.0
        self.expressions = []
        self.success_rate = 0.0
        self.timestamp = time.time()
        
    def add_result(self, execution_time: float):
        """Fügt ein Ergebnis hinzu"""
        self.execution_times.append(execution_time)
        
    def finalize(self):
        """Berechnet die finalen Metriken"""
        if self.execution_times:
            self.average_time = sum(self.execution_times) / len(self.execution_times)
            self.min_time = min(self.execution_times)
            self.max_time = max(self.execution_times)
            
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert das Ergebnis in ein Dictionary"""
        return {
            "benchmark_type": self.benchmark_type.name,
            "backend": self.backend.name,
            "iterations": self.iterations,
            "average_time": self.average_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "success_rate": self.success_rate,
            "timestamp": self.timestamp
        }
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'BenchmarkResult':
        """Erstellt ein BenchmarkResult aus einem Dictionary"""
        result = BenchmarkResult(
            BenchmarkType[data["benchmark_type"]], 
            MathEngineBackend[data["backend"]]
        )
        result.iterations = data["iterations"]
        result.average_time = data["average_time"]
        result.min_time = data["min_time"]
        result.max_time = data["max_time"]
        result.success_rate = data["success_rate"]
        result.timestamp = data["timestamp"]
        return result

class MPrimeEngineBenchmarker:
    """Benchmark-Tool für die MPRIME-Engine"""
    
    def __init__(self, use_mlx: bool = True, use_pytorch: bool = True):
        """
        Initialisiert den Benchmarker
        
        Args:
            use_mlx: Ob MLX verwendet werden soll (falls verfügbar)
            use_pytorch: Ob PyTorch verwendet werden soll (falls verfügbar)
        """
        self.use_mlx = use_mlx and MLX_AVAILABLE
        self.use_pytorch = use_pytorch and TORCH_AVAILABLE
        
        # Initialisiere die Engines
        self.engines = {}
        self.math_cores = {}
        
        if self.use_pytorch:
            # Konfiguriere für PyTorch-Nutzung
            pytorch_config = {
                "use_mlx": False,
                "use_torch": True,
                "precision": "float32",
                "device": "cpu"  # oder "cuda" wenn verfügbar
            }
            self.engines[MathEngineBackend.PYTORCH] = MPrimeEngine(config=pytorch_config)
            self.math_cores[MathEngineBackend.PYTORCH] = ContextualMathCore()
            self.math_cores[MathEngineBackend.PYTORCH].active_context = "scientific"
            
        if self.use_mlx:
            # Konfiguriere für MLX-Nutzung
            mlx_config = {
                "use_mlx": True,
                "use_torch": False,
                "precision": "float32",
                "optimize_for_apple_silicon": True
            }
            self.engines[MathEngineBackend.MLX] = MPrimeEngine(config=mlx_config)
            self.math_cores[MathEngineBackend.MLX] = ContextualMathCore()
            self.math_cores[MathEngineBackend.MLX].active_context = "scientific"
            
        # Testausdrücke
        self._init_test_expressions()
        
        # Ergebnisse
        self.results = []
        
    def _init_test_expressions(self):
        """Initialisiert die Testausdrücke für verschiedene Benchmark-Typen"""
        
        # Einfache arithmetische Ausdrücke
        self.simple_expressions = [
            "2 + 3 * 4",
            "5^2 - 10",
            "(7 + 8) / 3",
            "sqrt(16) + 5",
            "2^10"
        ]
        
        # Komplexe mathematische Ausdrücke
        self.complex_expressions = [
            "sin(x) * cos(y) + tan(z)",
            "log(x) + ln(y) - exp(z/10)",
            "a*sin(b*x) + c*cos(d*x)",
            "√(a^2 + b^2 - 2*a*b*cos(C))"
        ]
        
        # Matrix-Operationen (als String-Repräsentation)
        self.matrix_expressions = [
            "matrix([[1, 2], [3, 4]]) * matrix([[5, 6], [7, 8]])",
            "transpose(matrix([[1, 2, 3], [4, 5, 6]]))",
            "inverse(matrix([[1, 2], [3, 4]]))",
            "eigenvalues(matrix([[4, 2], [1, 3]]))"
        ]
        
        # Differentialgleichungen
        self.differential_expressions = [
            "d/dx(x^2)",
            "d/dx(sin(x))",
            "d/dx(e^x)",
            "d^2/dx^2(x^3 + 2*x^2 + x)"
        ]
        
        # Integrale
        self.integral_expressions = [
            "∫(x)dx",
            "∫(x^2)dx",
            "∫(sin(x))dx",
            "∫(e^x)dx"
        ]
        
        # Gleichungssysteme
        self.equation_systems = [
            "2*x + 3*y = 7; x - y = 1",
            "x^2 + y^2 = 25; x + y = 7"
        ]
        
        # Zuordnung der Ausdrücke zu Benchmark-Typen
        self.expressions_map = {
            BenchmarkType.SIMPLE: self.simple_expressions,
            BenchmarkType.COMPLEX: self.complex_expressions,
            BenchmarkType.MATRIX: self.matrix_expressions,
            BenchmarkType.DIFFERENTIAL: self.differential_expressions,
            BenchmarkType.INTEGRAL: self.integral_expressions,
            BenchmarkType.SYSTEM: self.equation_systems
        }
        
    def run_benchmark(self, benchmark_type: BenchmarkType, backend: MathEngineBackend, 
                    iterations: int = 100, warmup: int = 10) -> BenchmarkResult:
        """
        Führt einen Benchmark-Test durch
        
        Args:
            benchmark_type: Art des Benchmarks
            backend: Zu verwendendes Backend
            iterations: Anzahl der Testdurchläufe
            warmup: Anzahl der Warmup-Iterationen (werden nicht gezählt)
            
        Returns:
            Das Benchmark-Ergebnis
        """
        # Prüfe, ob das Backend verfügbar ist
        if backend not in self.engines:
            logger.error(f"Backend {backend.name} ist nicht verfügbar")
            return None
        
        engine = self.engines[backend]
        math_core = self.math_cores[backend]
        expressions = self.expressions_map[benchmark_type]
        
        # Erstelle das Ergebnisobjekt
        result = BenchmarkResult(benchmark_type, backend)
        result.iterations = iterations
        result.expressions = expressions
        
        logger.info(f"Starte Benchmark für {benchmark_type.name} mit Backend {backend.name}")
        logger.info(f"Ausdrücke: {expressions}")
        
        # Warmup-Phase
        logger.info(f"Führe {warmup} Warmup-Iterationen durch...")
        for _ in range(warmup):
            for expr in expressions:
                try:
                    math_core.process(expr)
                except Exception as e:
                    logger.warning(f"Fehler während Warmup mit {expr}: {e}")
        
        # Benchmark-Phase
        logger.info(f"Führe {iterations} Benchmark-Iterationen durch...")
        successes = 0
        
        for i in range(iterations):
            # Wähle einen zufälligen Ausdruck für mehr Variation
            import random
            expr = random.choice(expressions)
            
            try:
                start_time = time.time()
                result_data = math_core.process(expr)
                end_time = time.time()
                
                execution_time = end_time - start_time
                result.add_result(execution_time)
                
                if result_data["success"]:
                    successes += 1
                    
                if i % 10 == 0:  # Reduziere die Ausgabe
                    logger.info(f"Iteration {i+1}/{iterations} - Zeit: {execution_time:.6f} s")
                    
            except Exception as e:
                logger.error(f"Fehler bei Iteration {i+1} mit {expr}: {e}")
        
        # Berechne die finalen Metriken
        result.finalize()
        result.success_rate = successes / iterations if iterations > 0 else 0
        
        logger.info(f"Benchmark abgeschlossen. Durchschnittliche Zeit: {result.average_time:.6f} s")
        logger.info(f"Min Zeit: {result.min_time:.6f} s, Max Zeit: {result.max_time:.6f} s")
        logger.info(f"Erfolgsrate: {result.success_rate:.2f}")
        
        self.results.append(result)
        return result
    
    def run_all_benchmarks(self, iterations: int = 100, warmup: int = 10) -> List[BenchmarkResult]:
        """
        Führt alle Benchmark-Tests für alle verfügbaren Backends durch
        
        Args:
            iterations: Anzahl der Testdurchläufe
            warmup: Anzahl der Warmup-Iterationen
            
        Returns:
            Liste der Benchmark-Ergebnisse
        """
        all_results = []
        
        for benchmark_type in BenchmarkType:
            for backend in self.engines.keys():
                result = self.run_benchmark(benchmark_type, backend, iterations, warmup)
                if result:
                    all_results.append(result)
        
        self.results = all_results
        return all_results
    
    def save_results(self, filename: str):
        """
        Speichert die Benchmark-Ergebnisse in einer JSON-Datei
        
        Args:
            filename: Pfad zur Ausgabedatei
        """
        data = [result.to_dict() for result in self.results]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Ergebnisse in {filename} gespeichert")
    
    def load_results(self, filename: str):
        """
        Lädt Benchmark-Ergebnisse aus einer JSON-Datei
        
        Args:
            filename: Pfad zur Eingabedatei
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.results = [BenchmarkResult.from_dict(item) for item in data]
        logger.info(f"{len(self.results)} Ergebnisse aus {filename} geladen")
    
    def generate_html_report(self, output_file: str):
        """
        Generiert einen HTML-Bericht mit den Benchmark-Ergebnissen
        
        Args:
            output_file: Pfad zur HTML-Ausgabedatei
        """
        if not self.results:
            logger.warning("Keine Ergebnisse für den Bericht vorhanden")
            return
            
        # HTML-Template
        html_template = """
        <!DOCTYPE html>
        <html lang="de">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MPRIME Engine Performance-Bericht</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .chart-container { width: 100%; height: 400px; margin-bottom: 30px; }
                .success-high { background-color: #dff0d8; }
                .success-medium { background-color: #fcf8e3; }
                .success-low { background-color: #f2dede; }
                .summary { margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>MPRIME Engine Performance-Bericht</h1>
            <div class="summary">
                <h2>Zusammenfassung</h2>
                <p>Dieser Bericht zeigt die Performance-Vergleiche zwischen verschiedenen Backend-Engines (MLX, PyTorch) 
                für die MPRIME Mathematik-Engine. Die Benchmarks wurden mit MLX Version 0.24.1 und PyTorch durchgeführt.</p>
                <p>Systemumgebung: Apple Silicon, MISO Ultimate v1.4.2</p>
                <p>Datum: DATETIME</p>
            </div>
            
            <h2>Performance-Übersicht</h2>
            <div class="chart-container">
                <canvas id="overviewChart"></canvas>
            </div>
            
            RESULT_TABLES
            
            <script>
                // Chart.js Konfiguration
                const ctx = document.getElementById('overviewChart').getContext('2d');
                
                const chart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: CHART_LABELS,
                        datasets: CHART_DATASETS
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Durchschnittliche Zeit (ms)'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Benchmark-Typ'
                                }
                            }
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: 'Durchschnittliche Ausführungszeit nach Backend und Benchmark-Typ'
                            },
                            legend: {
                                position: 'top',
                            }
                        }
                    }
                });
            </script>
        </body>
        </html>
        """
        
        # Organisiere Daten nach Benchmark-Typ
        results_by_type = {}
        for result in self.results:
            benchmark_type = result.benchmark_type.name
            if benchmark_type not in results_by_type:
                results_by_type[benchmark_type] = []
            results_by_type[benchmark_type].append(result)
        
        # Erzeuge HTML-Tabellen für jeden Benchmark-Typ
        tables_html = ""
        chart_labels = []
        chart_datasets = {}
        
        for benchmark_type, results in results_by_type.items():
            # Tabellen-Header
            tables_html += f"""
            <h2>{benchmark_type} Benchmark</h2>
            <table>
                <tr>
                    <th>Backend</th>
                    <th>Durchschnittliche Zeit (ms)</th>
                    <th>Min. Zeit (ms)</th>
                    <th>Max. Zeit (ms)</th>
                    <th>Erfolgsrate</th>
                    <th>Iterationen</th>
                </tr>
            """
            
            # Tabellen-Zeilen
            chart_labels.append(benchmark_type)
            
            for result in results:
                backend = result.backend.name
                avg_time_ms = result.average_time * 1000
                min_time_ms = result.min_time * 1000
                max_time_ms = result.max_time * 1000
                success_class = ""
                
                if result.success_rate >= 0.9:
                    success_class = "success-high"
                elif result.success_rate >= 0.7:
                    success_class = "success-medium"
                else:
                    success_class = "success-low"
                
                tables_html += f"""
                <tr class="{success_class}">
                    <td>{backend}</td>
                    <td>{avg_time_ms:.4f}</td>
                    <td>{min_time_ms:.4f}</td>
                    <td>{max_time_ms:.4f}</td>
                    <td>{result.success_rate * 100:.2f}%</td>
                    <td>{result.iterations}</td>
                </tr>
                """
                
                # Daten für Chart sammeln
                if backend not in chart_datasets:
                    chart_datasets[backend] = {
                        "label": backend,
                        "data": [],
                        "backgroundColor": "#3e95cd" if backend == "MLX" else "#8e5ea2"
                    }
                chart_datasets[backend]["data"].append(avg_time_ms)
            
            tables_html += "</table>"
        
        # Chart-Daten formatieren
        chart_labels_json = json.dumps(chart_labels)
        chart_datasets_json = json.dumps(list(chart_datasets.values()))
        
        # HTML-Template füllen
        html_content = html_template
        html_content = html_content.replace("RESULT_TABLES", tables_html)
        html_content = html_content.replace("CHART_LABELS", chart_labels_json)
        html_content = html_content.replace("CHART_DATASETS", chart_datasets_json)
        html_content = html_content.replace("DATETIME", time.strftime("%d.%m.%Y %H:%M:%S"))
        
        # HTML-Datei speichern
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        logger.info(f"HTML-Bericht in {output_file} gespeichert")
        
        # Öffne optional den Browser
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(output_file)}")
        except:
            pass

    def compare_precision_performance(self, benchmark_type: BenchmarkType, iterations: int = 50):
        """
        Vergleicht die Performance verschiedener Präzisionstypen (float32, float16, bfloat16)
        
        Args:
            benchmark_type: Art des Benchmarks
            iterations: Anzahl der Iterationen
        
        Returns:
            Dictionary mit den Ergebnissen
        """
        if not MLX_AVAILABLE:
            logger.error("MLX ist nicht verfügbar, kann keinen Präzisionsvergleich durchführen")
            return None
            
        # MLX-Engine initialisieren, falls noch nicht geschehen
        if MathEngineBackend.MLX not in self.engines:
            self.engines[MathEngineBackend.MLX] = MPrimeEngine(backend="mlx")
            self.math_cores[MathEngineBackend.MLX] = ContextualMathCore()
            self.math_cores[MathEngineBackend.MLX].active_context = "scientific"
            
        engine = self.engines[MathEngineBackend.MLX]
        math_core = self.math_cores[MathEngineBackend.MLX]
        expressions = self.expressions_map[benchmark_type]
        
        precision_types = ["float32", "float16", "bfloat16"]
        results = {}
        
        for precision in precision_types:
            logger.info(f"Teste MLX mit Präzision {precision}...")
            
            # Setze die Präzision in der Engine
            if hasattr(engine, "set_precision"):
                engine.set_precision(precision)
            
            # Führe den Benchmark durch
            execution_times = []
            
            for i in range(iterations):
                expr = random.choice(expressions)
                
                try:
                    start_time = time.time()
                    result_data = math_core.process(expr)
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    execution_times.append(execution_time)
                    
                except Exception as e:
                    logger.error(f"Fehler bei Iteration {i+1} mit {expr}: {e}")
            
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                min_time = min(execution_times)
                max_time = max(execution_times)
                
                results[precision] = {
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "iterations": len(execution_times)
                }
                
                logger.info(f"Präzision {precision}: Durchschnittliche Zeit = {avg_time:.6f} s")
        
        return results


def main():
    """Hauptfunktion für die Ausführung der Benchmarks"""
    parser = argparse.ArgumentParser(description="MPRIME-Engine MLX Performance Benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Anzahl der Benchmark-Iterationen")
    parser.add_argument("--warmup", type=int, default=10, help="Anzahl der Warmup-Iterationen")
    parser.add_argument("--output", type=str, default="mprime_benchmark_results.json", help="Ausgabedatei für die Ergebnisse")
    parser.add_argument("--report", type=str, default="mprime_benchmark_report.html", help="Ausgabedatei für den HTML-Bericht")
    parser.add_argument("--no-mlx", action="store_true", help="Deaktiviere MLX-Benchmarks")
    parser.add_argument("--no-pytorch", action="store_true", help="Deaktiviere PyTorch-Benchmarks")
    parser.add_argument("--precision", action="store_true", help="Führe Präzisionsvergleich durch")
    
    args = parser.parse_args()
    
    benchmarker = MPrimeEngineBenchmarker(
        use_mlx=not args.no_mlx,
        use_pytorch=not args.no_pytorch
    )
    
    # Führe Benchmarks durch
    results = benchmarker.run_all_benchmarks(
        iterations=args.iterations,
        warmup=args.warmup
    )
    
    # Speichere die Ergebnisse
    benchmarker.save_results(args.output)
    
    # Generiere HTML-Bericht
    benchmarker.generate_html_report(args.report)
    
    # Optional: Führe Präzisionsvergleich durch
    if args.precision and MLX_AVAILABLE:
        precision_results = {}
        for benchmark_type in [BenchmarkType.MATRIX, BenchmarkType.COMPLEX]:
            precision_results[benchmark_type.name] = benchmarker.compare_precision_performance(benchmark_type)
        
        # Speichere Präzisionsergebnisse
        with open("mprime_precision_results.json", 'w') as f:
            precision_json = {}
            for key, value in precision_results.items():
                if value:
                    precision_json[key] = {k: {kk: vv for kk, vv in v.items() if kk != "iterations"} for k, v in value.items()}
            json.dump(precision_json, f, indent=4)
        
        logger.info("Präzisionsvergleich abgeschlossen und in mprime_precision_results.json gespeichert")


if __name__ == "__main__":
    main()
