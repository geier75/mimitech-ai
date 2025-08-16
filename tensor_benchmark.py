#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Tensor Backend Benchmark

Dieses Skript führt Leistungsvergleiche zwischen den verschiedenen
Tensor-Backends (MLX, PyTorch, NumPy) durch.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [TENSOR-BENCH] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.Benchmark.Tensor")

# Benchmark-Konfiguration
MATRIX_SIZES = [128, 512, 1024, 2048]
ITERATIONS = 5
WARMUP_ITERATIONS = 2

# Verfügbare Backends prüfen
BACKENDS = {}

# NumPy (immer verfügbar)
BACKENDS["numpy"] = {"available": True, "module": np}
logger.info("NumPy-Backend verfügbar")

# PyTorch
try:
    import torch
    BACKENDS["torch"] = {"available": True, "module": torch}
    
    # Prüfe MPS-Unterstützung
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        BACKENDS["torch"]["device"] = "mps"
        logger.info("PyTorch-Backend verfügbar (MPS)")
    elif torch.cuda.is_available():
        BACKENDS["torch"]["device"] = "cuda"
        logger.info("PyTorch-Backend verfügbar (CUDA)")
    else:
        BACKENDS["torch"]["device"] = "cpu"
        logger.info("PyTorch-Backend verfügbar (CPU)")
except ImportError:
    BACKENDS["torch"] = {"available": False}
    logger.warning("PyTorch-Backend nicht verfügbar")

# MLX
try:
    import mlx.core as mx
    BACKENDS["mlx"] = {"available": True, "module": mx}
    logger.info("MLX-Backend verfügbar")
except ImportError:
    BACKENDS["mlx"] = {"available": False}
    logger.warning("MLX-Backend nicht verfügbar")

def create_random_matrices(size: int) -> Dict[str, Any]:
    """Erstellt zufällige Matrizen für verschiedene Backends"""
    matrices = {}
    
    # NumPy (Basisformat)
    np_matrix = np.random.rand(size, size).astype(np.float32)
    matrices["numpy"] = np_matrix
    
    # PyTorch
    if BACKENDS["torch"]["available"]:
        torch_matrix = torch.tensor(np_matrix)
        if BACKENDS["torch"].get("device") == "mps":
            torch_matrix = torch_matrix.to("mps")
        elif BACKENDS["torch"].get("device") == "cuda":
            torch_matrix = torch_matrix.to("cuda")
        matrices["torch"] = torch_matrix
    
    # MLX
    if BACKENDS["mlx"]["available"]:
        matrices["mlx"] = mx.array(np_matrix)
    
    return matrices

def benchmark_operation(operation: str, matrices: Dict[str, Any], size: int) -> Dict[str, float]:
    """Führt einen Benchmark für eine bestimmte Operation durch"""
    results = {}
    
    for backend, data in matrices.items():
        if not BACKENDS[backend]["available"]:
            continue
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            if operation == "matmul":
                if backend == "numpy":
                    np.matmul(data, data)
                elif backend == "torch":
                    torch.matmul(data, data)
                elif backend == "mlx":
                    mx.matmul(data, data)
            elif operation == "add":
                if backend == "numpy":
                    data + data
                elif backend == "torch":
                    data + data
                elif backend == "mlx":
                    data + data
            elif operation == "exp":
                if backend == "numpy":
                    np.exp(data)
                elif backend == "torch":
                    torch.exp(data)
                elif backend == "mlx":
                    mx.exp(data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(ITERATIONS):
            if operation == "matmul":
                if backend == "numpy":
                    np.matmul(data, data)
                elif backend == "torch":
                    torch.matmul(data, data)
                    if BACKENDS["torch"].get("device") in ["mps", "cuda"]:
                        torch.cuda.synchronize() if BACKENDS["torch"].get("device") == "cuda" else torch.mps.synchronize()
                elif backend == "mlx":
                    mx.matmul(data, data)
                    mx.eval()
            elif operation == "add":
                if backend == "numpy":
                    data + data
                elif backend == "torch":
                    data + data
                    if BACKENDS["torch"].get("device") in ["mps", "cuda"]:
                        torch.cuda.synchronize() if BACKENDS["torch"].get("device") == "cuda" else torch.mps.synchronize()
                elif backend == "mlx":
                    data + data
                    mx.eval()
            elif operation == "exp":
                if backend == "numpy":
                    np.exp(data)
                elif backend == "torch":
                    torch.exp(data)
                    if BACKENDS["torch"].get("device") in ["mps", "cuda"]:
                        torch.cuda.synchronize() if BACKENDS["torch"].get("device") == "cuda" else torch.mps.synchronize()
                elif backend == "mlx":
                    mx.exp(data)
                    mx.eval()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / ITERATIONS
        results[backend] = avg_time
        
        logger.info(f"{backend.upper()} {operation}: {avg_time*1000:.2f} ms für Größe {size}x{size}")
    
    return results

def run_benchmarks() -> Dict[str, Any]:
    """Führt alle Benchmarks durch"""
    all_results = {}
    
    for operation in ["matmul", "add", "exp"]:
        operation_results = {}
        
        for size in MATRIX_SIZES:
            logger.info(f"\n=== Benchmark: {operation.upper()} mit Größe {size}x{size} ===")
            matrices = create_random_matrices(size)
            results = benchmark_operation(operation, matrices, size)
            operation_results[size] = results
        
        all_results[operation] = operation_results
    
    return all_results

def plot_results(results: Dict[str, Any], output_dir: str = "."):
    """Erstellt Plots der Benchmark-Ergebnisse"""
    os.makedirs(output_dir, exist_ok=True)
    
    for operation, operation_results in results.items():
        plt.figure(figsize=(10, 6))
        
        for backend in ["numpy", "torch", "mlx"]:
            if not BACKENDS[backend]["available"]:
                continue
            
            sizes = []
            times = []
            
            for size, size_results in operation_results.items():
                if backend in size_results:
                    sizes.append(size)
                    times.append(size_results[backend] * 1000)  # Convert to ms
            
            plt.plot(sizes, times, marker='o', label=f"{backend.upper()}")
        
        plt.xlabel("Matrix Size")
        plt.ylabel("Time (ms)")
        plt.title(f"{operation.upper()} Performance Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"benchmark_{operation}.png"))
        logger.info(f"Plot für {operation} erstellt: benchmark_{operation}.png")
    
    # Vergleichsplots für jede Matrixgröße
    for size in MATRIX_SIZES:
        plt.figure(figsize=(10, 6))
        
        backends = []
        operations = []
        times = []
        
        for backend in ["numpy", "torch", "mlx"]:
            if not BACKENDS[backend]["available"]:
                continue
            
            for operation in ["matmul", "add", "exp"]:
                if backend in results[operation][size]:
                    backends.append(backend.upper())
                    operations.append(operation.upper())
                    times.append(results[operation][size][backend] * 1000)  # Convert to ms
        
        x = range(len(backends))
        plt.bar(x, times)
        plt.xticks(x, [f"{b}\n{o}" for b, o in zip(backends, operations)], rotation=0)
        plt.ylabel("Time (ms)")
        plt.title(f"Operation Performance for {size}x{size} Matrix")
        plt.savefig(os.path.join(output_dir, f"benchmark_size_{size}.png"))
        logger.info(f"Plot für Größe {size} erstellt: benchmark_size_{size}.png")
    
    plt.close('all')

def main():
    """Hauptfunktion"""
    logger.info("=== MISO Ultimate Tensor Backend Benchmark ===")
    
    # Führe Benchmarks durch
    results = run_benchmarks()
    
    # Speichere Ergebnisse
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "tensor_benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Erstelle Plots
    plot_results(results, output_dir)
    
    logger.info(f"\nBenchmark abgeschlossen. Ergebnisse in: {output_dir}")

if __name__ == "__main__":
    main()
