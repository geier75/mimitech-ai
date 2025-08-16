#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vereinfachter Benchmark für die MLX-Optimierung der T-Mathematics Engine

Dieser Benchmark vergleicht die Leistung der T-Mathematics Engine
mit und ohne MLX-Optimierung auf Apple Silicon Hardware.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from miso.math.t_mathematics.engine import TMathEngine
from miso.math.t_mathematics.mlx_support import HAS_MLX, IS_APPLE_SILICON

# Logger konfigurieren
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mlx_benchmark")

def run_matrix_multiplication_benchmark(sizes: List[int], 
                                       iterations: int = 10, 
                                       use_mlx: bool = True) -> Dict[int, float]:
    """
    Führt einen Benchmark für Matrixmultiplikation durch.
    
    Args:
        sizes: Liste der Matrixgrößen (N für NxN-Matrizen)
        iterations: Anzahl der Wiederholungen pro Größe
        use_mlx: Ob MLX verwendet werden soll
        
    Returns:
        Dictionary mit Matrixgrößen als Schlüssel und durchschnittlichen Zeiten als Werte
    """
    results = {}
    
    # Initialisiere die T-Mathematics Engine
    engine = TMathEngine(use_mlx=use_mlx)
    
    for size in sizes:
        logger.info(f"Benchmark für Matrixgröße {size}x{size} mit MLX={use_mlx}")
        
        # Erzeuge zufällige Matrizen
        a = torch.rand(size, size, device=engine.device)
        b = torch.rand(size, size, device=engine.device)
        
        # Warmup
        _ = engine.matmul(a, b)
        
        # Benchmark
        times = []
        for i in range(iterations):
            start_time = time.time()
            _ = engine.matmul(a, b)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        results[size] = avg_time
        logger.info(f"Durchschnittliche Zeit: {avg_time:.6f} Sekunden")
    
    return results

def run_svd_benchmark(sizes: List[int], 
                     iterations: int = 5, 
                     use_mlx: bool = True) -> Dict[int, float]:
    """
    Führt einen Benchmark für Singulärwertzerlegung (SVD) durch.
    
    Args:
        sizes: Liste der Matrixgrößen (N für NxN-Matrizen)
        iterations: Anzahl der Wiederholungen pro Größe
        use_mlx: Ob MLX verwendet werden soll
        
    Returns:
        Dictionary mit Matrixgrößen als Schlüssel und durchschnittlichen Zeiten als Werte
    """
    results = {}
    
    # Initialisiere die T-Mathematics Engine
    engine = TMathEngine(use_mlx=use_mlx)
    
    for size in sizes:
        logger.info(f"Benchmark für SVD mit Matrixgröße {size}x{size} mit MLX={use_mlx}")
        
        # Erzeuge zufällige Matrix
        a = torch.rand(size, size, device=engine.device)
        
        # Warmup
        try:
            _ = engine.svd(a)
            
            # Benchmark
            times = []
            for i in range(iterations):
                start_time = time.time()
                _ = engine.svd(a)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            results[size] = avg_time
            logger.info(f"Durchschnittliche Zeit: {avg_time:.6f} Sekunden")
        except Exception as e:
            logger.error(f"Fehler bei SVD für Größe {size}: {e}")
            results[size] = None
    
    return results

def calculate_speedup(mlx_results, torch_results):
    """Berechnet den Speedup von MLX gegenüber PyTorch"""
    speedups = {}
    for size in mlx_results:
        if size in torch_results and mlx_results[size] is not None and torch_results[size] is not None:
            speedup = torch_results[size] / mlx_results[size]
            speedups[size] = speedup
            logger.info(f"Speedup für Größe {size}: {speedup:.2f}x")
    
    avg_speedup = sum(speedups.values()) / len(speedups) if speedups else 0
    logger.info(f"Durchschnittlicher Speedup: {avg_speedup:.2f}x")
    return speedups, avg_speedup

def main():
    """Hauptfunktion für den Benchmark"""
    if not IS_APPLE_SILICON:
        logger.error("Dieser Benchmark ist nur für Apple Silicon Hardware gedacht.")
        return
    
    if not HAS_MLX:
        logger.error("MLX ist nicht verfügbar. Bitte installieren Sie MLX mit 'pip install mlx'.")
        return
    
    # Konfiguration
    matrix_sizes = [128, 256, 512, 1024]
    svd_sizes = [32, 64, 128, 256]
    iterations = 5
    
    # Matrix-Multiplikation Benchmark
    logger.info("Starte Matrix-Multiplikation Benchmark...")
    logger.info("=== MLX ===")
    mlx_matmul_results = run_matrix_multiplication_benchmark(matrix_sizes, iterations, use_mlx=True)
    logger.info("=== PyTorch ===")
    torch_matmul_results = run_matrix_multiplication_benchmark(matrix_sizes, iterations, use_mlx=False)
    
    # SVD Benchmark
    logger.info("Starte SVD Benchmark...")
    logger.info("=== MLX ===")
    mlx_svd_results = run_svd_benchmark(svd_sizes, iterations, use_mlx=True)
    logger.info("=== PyTorch ===")
    torch_svd_results = run_svd_benchmark(svd_sizes, iterations, use_mlx=False)
    
    # Speedup berechnen
    logger.info("=== Matrix-Multiplikation Speedup ===")
    matmul_speedups, matmul_avg_speedup = calculate_speedup(mlx_matmul_results, torch_matmul_results)
    
    logger.info("=== SVD Speedup ===")
    svd_speedups, svd_avg_speedup = calculate_speedup(mlx_svd_results, torch_svd_results)
    
    logger.info("Benchmark abgeschlossen.")
    logger.info(f"Matrix-Multiplikation durchschnittlicher Speedup: {matmul_avg_speedup:.2f}x")
    logger.info(f"SVD durchschnittlicher Speedup: {svd_avg_speedup:.2f}x")

if __name__ == "__main__":
    main()
