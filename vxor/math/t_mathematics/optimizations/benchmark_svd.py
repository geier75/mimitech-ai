#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVD Benchmark für T-Mathematics Engine Optimierungen

Dieses Skript vergleicht die Leistung der originalen SVD-Implementierung 
mit der optimierten Version für verschiedene Matrix-Größen und SVD-Szenarien.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import numpy as np
import torch
import logging
import copy
from typing import Dict, List, Tuple, Any

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.math.t_mathematics.optimizations.benchmark_svd")

# Prüfe auf MLX
try:
    import mlx.core as mx
    HAS_MLX = True
    logger.info("MLX verfügbar")
except ImportError:
    HAS_MLX = False
    mx = None
    logger.warning("MLX nicht verfügbar - Benchmarks werden mit NumPy durchgeführt")

# T-Mathematics Engine Pfad hinzufügen
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# MLX-Backend importieren
try:
    from miso.math.t_mathematics.mlx_support import MLXBackend
    logger.info("MLXBackend erfolgreich importiert")
except ImportError:
    logger.error("MLXBackend konnte nicht importiert werden")
    sys.exit(1)

# Optimierungen importieren
try:
    from miso.math.t_mathematics.optimizations.integration import optimize_mlx_backend
    logger.info("Optimierungsmodul erfolgreich importiert")
except ImportError:
    logger.error("Optimierungsmodul konnte nicht importiert werden")
    sys.exit(1)

def run_svd_benchmark(matrices, k_values, num_runs=5, optimization_levels=[0, 2, 3]):
    """
    Führt SVD-Benchmarks für verschiedene Matrix-Größen und k-Werte durch
    
    Args:
        matrices: Liste von (name, matrix) Tupeln
        k_values: Liste von k-Werten für partielle SVD
        num_runs: Anzahl der Durchläufe pro Matrix/k-Kombination
        optimization_levels: Zu testende Optimierungsstufen
        
    Returns:
        Dictionary mit Benchmark-Ergebnissen
    """
    results = {}
    
    # Erstelle Backends für verschiedene Optimierungsstufen
    backends = {}
    for level in optimization_levels:
        # Originales Backend erstellen
        backend = MLXBackend(precision="float32")
        
        # Backend optimieren, wenn Level > 0
        if level > 0:
            backends[level] = optimize_mlx_backend(backend, optimization_level=level)
        else:
            backends[level] = backend
        
        logger.info(f"Backend mit Optimierungsstufe {level} erstellt")
    
    # Durchlaufe alle Matrizen
    for matrix_name, matrix in matrices:
        results[matrix_name] = {}
        
        # Teste vollständige SVD
        logger.info(f"Teste vollständige SVD für {matrix_name}...")
        full_results = _benchmark_svd(backends, matrix, k=None, num_runs=num_runs)
        results[matrix_name]["full"] = full_results
        
        # Teste partielle SVD für verschiedene k-Werte
        for k in k_values:
            if k < min(matrix.shape):  # Überprüfe, ob k gültig ist
                logger.info(f"Teste partielle SVD mit k={k} für {matrix_name}...")
                partial_results = _benchmark_svd(backends, matrix, k=k, num_runs=num_runs)
                results[matrix_name][f"k={k}"] = partial_results
    
    return results

def _benchmark_svd(backends, matrix, k=None, num_runs=5):
    """
    Führt SVD-Benchmark für eine bestimmte Matrix und k-Wert durch
    
    Args:
        backends: Dictionary mit Backends nach Optimierungsstufe
        matrix: Matrix für den Benchmark
        k: k-Wert für partielle SVD (None für vollständige SVD)
        num_runs: Anzahl der Durchläufe
        
    Returns:
        Dictionary mit Benchmark-Ergebnissen
    """
    results = {}
    
    # Konvertiere Matrix zu MLX, wenn verfügbar
    if HAS_MLX:
        matrix_mlx = mx.array(matrix)
    else:
        matrix_mlx = matrix
    
    # Durchlaufe alle Backends
    for level, backend in backends.items():
        times = []
        
        # Aufwärmphase
        try:
            _ = backend.svd(matrix_mlx, k)
        except Exception as e:
            logger.error(f"Fehler während der Aufwärmphase für Optimierungsstufe {level}: {e}")
            continue
        
        # Benchmark-Durchläufe
        for run in range(num_runs):
            try:
                start_time = time.time()
                _ = backend.svd(matrix_mlx, k)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                logger.error(f"Fehler während Durchlauf {run} für Optimierungsstufe {level}: {e}")
        
        # Ergebnisse berechnen
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            results[level] = {
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "times": times
            }
        else:
            results[level] = {
                "avg_time": float('inf'),
                "min_time": float('inf'),
                "max_time": float('inf'),
                "times": []
            }
    
    return results

def generate_test_matrices():
    """
    Generiert Testmatrizen verschiedener Größen und Komplexitäten
    
    Returns:
        Liste von (name, matrix) Tupeln
    """
    matrices = []
    
    # Kleine Matrix (32x32)
    small_matrix = np.random.rand(32, 32).astype(np.float32)
    matrices.append(("small_32x32", small_matrix))
    
    # Mittlere Matrix (128x128)
    medium_matrix = np.random.rand(128, 128).astype(np.float32)
    matrices.append(("medium_128x128", medium_matrix))
    
    # Große Matrix (512x512)
    large_matrix = np.random.rand(512, 512).astype(np.float32)
    matrices.append(("large_512x512", large_matrix))
    
    # Rechteckige Matrix (1024x32)
    tall_matrix = np.random.rand(1024, 32).astype(np.float32)
    matrices.append(("tall_1024x32", tall_matrix))
    
    # Rechteckige Matrix (32x1024)
    wide_matrix = np.random.rand(32, 1024).astype(np.float32)
    matrices.append(("wide_32x1024", wide_matrix))
    
    return matrices

def print_benchmark_results(results):
    """
    Gibt Benchmark-Ergebnisse aus
    
    Args:
        results: Dictionary mit Benchmark-Ergebnissen
    """
    optimization_levels = sorted(list(next(iter(next(iter(results.values())).values())).keys()))
    
    print("\n" + "=" * 100)
    print("SVD BENCHMARK ERGEBNISSE".center(100))
    print("=" * 100)
    
    for matrix_name, matrix_results in results.items():
        print(f"\nMatrix: {matrix_name}")
        print("-" * 100)
        
        # Header
        header = "SVD-Typ".ljust(15)
        for level in optimization_levels:
            level_name = f"Opt Level {level}"
            header += level_name.center(20)
        
        if len(optimization_levels) > 1:
            header += "Verbesserung".center(25)
        
        print(header)
        print("-" * 100)
        
        # Ergebnisse für jeden SVD-Typ
        for svd_type, type_results in matrix_results.items():
            row = svd_type.ljust(15)
            
            # Zeiten für jedes Optimierungslevel
            for level in optimization_levels:
                if level in type_results:
                    avg_time = type_results[level]["avg_time"]
                    row += f"{avg_time:.6f} s".center(20)
                else:
                    row += "N/A".center(20)
            
            # Verbesserung berechnen, wenn mehrere Levels verfügbar
            if len(optimization_levels) > 1 and 0 in optimization_levels:
                if 0 in type_results and type_results[0]["avg_time"] > 0:
                    base_time = type_results[0]["avg_time"]
                    best_time = min(type_results[level]["avg_time"] for level in optimization_levels if level in type_results)
                    improvement = (base_time - best_time) / base_time * 100
                    row += f"{improvement:.2f}% schneller".center(25)
                else:
                    row += "N/A".center(25)
            
            print(row)
    
    print("\n" + "=" * 100)
    print("ZUSAMMENFASSUNG".center(100))
    print("=" * 100)
    
    # Durchschnittliche Verbesserung berechnen
    if len(optimization_levels) > 1 and 0 in optimization_levels:
        total_improvement = 0
        count = 0
        
        for matrix_results in results.values():
            for type_results in matrix_results.values():
                if 0 in type_results and type_results[0]["avg_time"] > 0:
                    base_time = type_results[0]["avg_time"]
                    best_time = min(type_results[level]["avg_time"] for level in optimization_levels if level in type_results)
                    improvement = (base_time - best_time) / base_time * 100
                    total_improvement += improvement
                    count += 1
        
        if count > 0:
            avg_improvement = total_improvement / count
            print(f"\nDurchschnittliche Verbesserung: {avg_improvement:.2f}%")
    
    print("\nOptimierungsstufen:")
    print("  0 = Keine Optimierung (Baseline)")
    print("  2 = Standard-Optimierung (SVD-Optimierung)")
    print("  3 = Aggressive Optimierung (Hybride SVD)")

def main():
    """
    Hauptfunktion für den SVD-Benchmark
    """
    print("\nSVD-Benchmark für T-Mathematics Engine Optimierungen")
    print("===================================================\n")
    
    # Generiere Testmatrizen
    matrices = generate_test_matrices()
    logger.info(f"{len(matrices)} Testmatrizen generiert")
    
    # Teste mit verschiedenen k-Werten
    k_values = [10, 50]
    logger.info(f"Teste mit k-Werten: {k_values}")
    
    # Führe Benchmarks durch
    results = run_svd_benchmark(matrices, k_values, num_runs=3)
    
    # Zeige Ergebnisse
    print_benchmark_results(results)
    
    print("\nBenchmark abgeschlossen.")

if __name__ == "__main__":
    main()
