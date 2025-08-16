#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Engine - Umfassender Benchmark

Dieser Benchmark testet die T-Mathematics Engine umfassend und speichert die Ergebnisse
im richtigen Format für das VXOR Benchmark Dashboard.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import torch
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importiere T-Mathematics Engine
from miso.math.t_mathematics.engine import TMathEngine
from miso.math.t_mathematics.compat import TMathConfig
from miso.math.t_mathematics.mlx_support import HAS_MLX, IS_APPLE_SILICON

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [T-MATH-BENCH] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.Benchmark.TMathematics")

# Benchmark-Konfiguration
MATRIX_SIZES = [128, 256, 512, 1024, 2048]
ITERATIONS = 5
WARM_UP_ITERATIONS = 2

# Ergebnisverzeichnis
RESULTS_DIR = Path("benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)

class TMathBenchmark:
    """Umfassender Benchmark für die T-Mathematics Engine."""
    
    def __init__(self, use_mlx: bool = True, precision: str = "float16"):
        """
        Initialisiert den Benchmark.
        
        Args:
            use_mlx: Ob MLX verwendet werden soll (falls verfügbar)
            precision: Präzisionstyp (float32, float16, bfloat16)
        """
        self.use_mlx = use_mlx and HAS_MLX and IS_APPLE_SILICON
        self.precision = precision
        self.backend_name = "MLX" if self.use_mlx else "PyTorch"
        self.results = {}
        
        # Initialisiere die T-Mathematics Engine
        config = TMathConfig(
            precision=precision,
            device="auto",
            optimize_for_rdna=False,
            optimize_for_apple_silicon=IS_APPLE_SILICON
        )
        
        self.engine = TMathEngine(
            config=config,
            use_mlx=self.use_mlx
        )
        
        logger.info(f"T-Mathematics Engine initialisiert: Backend={self.backend_name}, "
                   f"Präzision={precision}, MLX={self.use_mlx}")
    
    def _run_benchmark(self, operation_name: str, operation_func, sizes: List[int], 
                     iterations: int = 5) -> Dict[str, Any]:
        """
        Führt einen Benchmark für eine bestimmte Operation durch.
        
        Args:
            operation_name: Name der Operation
            operation_func: Funktion, die die Operation durchführt
            sizes: Liste der Matrixgrößen (N für NxN-Matrizen)
            iterations: Anzahl der Wiederholungen pro Größe
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        results = {}
        
        for size in sizes:
            logger.info(f"Benchmark für {operation_name}: Größe {size}x{size}, {self.backend_name}")
            
            try:
                # Erzeuge Eingabedaten für die Operation
                input_data = self._create_input_data(operation_name, size)
                
                # Warmup
                for _ in range(WARM_UP_ITERATIONS):
                    _ = operation_func(**input_data)
                
                # Eigentliche Messung
                times = []
                for i in range(iterations):
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                    result = operation_func(**input_data)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    execution_time = (time.time() - start_time) * 1000  # Zeit in ms
                    times.append(execution_time)
                    logger.info(f"  Iteration {i+1}/{iterations}: {execution_time:.3f} ms")
                
                # Ergebnisse speichern
                results[size] = {
                    "mean_time_ms": sum(times) / len(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "std_dev": np.std(times),
                    "size": size,
                    "backend": self.backend_name,
                    "precision": self.precision,
                    "success": True
                }
                logger.info(f"  Durchschnittliche Zeit: {results[size]['mean_time_ms']:.3f} ms")
                
            except Exception as e:
                logger.error(f"Fehler bei {operation_name} für Größe {size}: {e}")
                results[size] = {
                    "size": size,
                    "backend": self.backend_name,
                    "precision": self.precision,
                    "error": str(e),
                    "success": False
                }
        
        return results
    
    def _create_input_data(self, operation_name: str, size: int) -> Dict[str, Any]:
        """
        Erstellt Eingabedaten für verschiedene Operationen.
        
        Args:
            operation_name: Name der Operation
            size: Größe der Matrix/des Tensors
            
        Returns:
            Dictionary mit Eingabedaten für die Operation
        """
        if operation_name == "matrix_multiplication":
            matrix_a = torch.rand((size, size))
            matrix_b = torch.rand((size, size))
            return {
                "a": self.engine.prepare_tensor(matrix_a),
                "b": self.engine.prepare_tensor(matrix_b)
            }
        
        elif operation_name == "matrix_addition":
            matrix_a = torch.rand((size, size))
            matrix_b = torch.rand((size, size))
            return {
                "a": self.engine.prepare_tensor(matrix_a),
                "b": self.engine.prepare_tensor(matrix_b)
            }
        
        elif operation_name == "svd":
            matrix = torch.rand((size, size))
            return {
                "matrix": self.engine.prepare_tensor(matrix)
            }
        
        elif operation_name == "element_wise_ops":
            tensor = torch.rand((size, size))
            return {
                "tensor": self.engine.prepare_tensor(tensor)
            }
        
        elif operation_name == "reduction_ops":
            tensor = torch.rand((size, size))
            return {
                "tensor": self.engine.prepare_tensor(tensor)
            }
        
        elif operation_name == "activation_funcs":
            tensor = torch.rand((size, size))
            return {
                "tensor": self.engine.prepare_tensor(tensor)
            }
        
        elif operation_name == "normalization":
            input_tensor = torch.rand((size, size))
            weight = torch.rand((size,))
            bias = torch.rand((size,))
            return {
                "input": self.engine.prepare_tensor(input_tensor),
                "weight": self.engine.prepare_tensor(weight),
                "bias": self.engine.prepare_tensor(bias)
            }
        
        # Standardfall
        tensor = torch.rand((size, size))
        return {
            "tensor": self.engine.prepare_tensor(tensor)
        }
    
    def run_all_benchmarks(self, sizes: List[int] = None):
        """
        Führt alle Benchmarks durch.
        
        Args:
            sizes: Liste der zu testenden Matrixgrößen, Standard=MATRIX_SIZES
        """
        if sizes is None:
            sizes = MATRIX_SIZES
        
        # Matrix-Multiplikation
        self.results["matrix_multiplication"] = self._run_benchmark(
            "matrix_multiplication",
            lambda a, b: self.engine.matmul(a, b),
            sizes
        )
        
        # Matrix-Addition
        self.results["matrix_addition"] = self._run_benchmark(
            "matrix_addition",
            lambda a, b: self.engine.add(a, b),
            sizes
        )
        
        # SVD (nur für kleinere Matrizen)
        svd_sizes = [s for s in sizes if s <= 512]
        self.results["svd"] = self._run_benchmark(
            "svd",
            lambda matrix: self.engine.svd(matrix),
            svd_sizes,
            max(1, ITERATIONS // 2)  # Weniger Iterationen für SVD
        )
        
        # Element-wise Operationen
        self.results["element_wise_ops"] = self._run_benchmark(
            "element_wise_ops",
            lambda tensor: self.engine.mul(tensor, tensor),
            sizes
        )
        
        # Reduktionsoperationen
        self.results["reduction_ops"] = self._run_benchmark(
            "reduction_ops",
            lambda tensor: self.engine.sum(tensor),
            sizes
        )
        
        # Aktivierungsfunktionen
        self.results["activation_funcs"] = self._run_benchmark(
            "activation_funcs",
            lambda tensor: self.engine.gelu(tensor),
            sizes
        )
        
        # Normalisierung
        self.results["normalization"] = self._run_benchmark(
            "normalization",
            lambda input, weight, bias: self.engine.layer_norm(input, weight, bias),
            sizes
        )
    
    def save_results(self, output_dir: Path = RESULTS_DIR):
        """
        Speichert die Benchmark-Ergebnisse im Dashboard-kompatiblen Format.
        
        Args:
            output_dir: Ausgabeverzeichnis für die Ergebnisse
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"t_math_benchmark_{self.backend_name.lower()}_{self.precision}_{timestamp}.json"
        
        # Formatiere Ergebnisse für das Dashboard
        dashboard_results = {
            "metadata": {
                "timestamp": timestamp,
                "test_name": f"T-Mathematics Engine Benchmark ({self.backend_name} / {self.precision})",
                "hardware": {
                    "is_apple_silicon": IS_APPLE_SILICON,
                    "has_mlx": HAS_MLX,
                    "device": str(torch.device("cuda" if torch.cuda.is_available() else 
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
                },
                "config": {
                    "backend": self.backend_name,
                    "precision": self.precision,
                    "iterations": ITERATIONS,
                    "warm_up_iterations": WARM_UP_ITERATIONS,
                    "matrix_sizes": MATRIX_SIZES
                }
            },
            "categories": [],
            "datasets": [],
            "raw_data": self.results
        }
        
        # Erstelle Kategorien und Datensätze
        operation_names = {
            "matrix_multiplication": "Matrix-Multiplikation",
            "matrix_addition": "Matrix-Addition",
            "svd": "Singulärwertzerlegung (SVD)",
            "element_wise_ops": "Element-weise Operationen",
            "reduction_ops": "Reduktionsoperationen",
            "activation_funcs": "Aktivierungsfunktionen",
            "normalization": "Normalisierung"
        }
        
        # Erstelle Kategorien
        for op_key, op_name in operation_names.items():
            if op_key in self.results:
                dashboard_results["categories"].append({
                    "id": op_key,
                    "name": op_name,
                    "description": f"Performance-Messungen für {op_name}",
                    "is_time_series": False
                })
        
        # Erstelle Datensätze
        for op_key, op_name in operation_names.items():
            if op_key in self.results:
                op_results = self.results[op_key]
                
                # Prüfe, ob erfolgreiche Ergebnisse vorliegen
                successful_sizes = [
                    size for size, result in op_results.items() 
                    if result.get("success", False)
                ]
                
                if successful_sizes:
                    dataset = {
                        "category_id": op_key,
                        "name": f"{self.backend_name} ({self.precision})",
                        "data": [
                            {
                                "x": str(size),
                                "y": op_results[size]["mean_time_ms"]
                            }
                            for size in sorted(successful_sizes)
                        ]
                    }
                    dashboard_results["datasets"].append(dataset)
        
        # Speichere die Ergebnisse
        with open(output_file, "w") as f:
            json.dump(dashboard_results, f, indent=2)
        
        logger.info(f"Ergebnisse gespeichert in: {output_file}")
        return output_file

def run_comprehensive_benchmark():
    """Führt den umfassenden Benchmark für alle Konfigurationen durch."""
    results_files = []
    
    # Hardware-Informationen ausgeben
    print(f"\n===== T-Mathematics Engine Umfassender Benchmark =====")
    print(f"- Hardware-Erkennung:")
    print(f"  - Apple Silicon: {'Ja' if IS_APPLE_SILICON else 'Nein'}")
    print(f"  - MLX verfügbar: {'Ja' if HAS_MLX else 'Nein'}")
    print(f"  - PyTorch Gerät: {torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')}")
    
    # Führe Benchmarks für verschiedene Backends und Präzisionen durch
    backends = []
    if HAS_MLX and IS_APPLE_SILICON:
        backends.append((True, "MLX"))
    backends.append((False, "PyTorch"))
    
    precisions = ["float32", "float16"]
    if IS_APPLE_SILICON:
        precisions.append("bfloat16")
    
    for use_mlx, backend_name in backends:
        for precision in precisions:
            print(f"\n===== Benchmark: {backend_name} / {precision} =====")
            try:
                benchmark = TMathBenchmark(use_mlx=use_mlx, precision=precision)
                benchmark.run_all_benchmarks()
                result_file = benchmark.save_results()
                results_files.append(result_file)
            except Exception as e:
                logger.error(f"Fehler beim Benchmark für {backend_name}/{precision}: {e}")
    
    print(f"\n===== Benchmark abgeschlossen =====")
    for result_file in results_files:
        print(f"- Ergebnisse: {result_file}")
    print(f"\nStarten Sie das Dashboard mit: python3 start_benchmark_dashboard.py")

if __name__ == "__main__":
    run_comprehensive_benchmark()
