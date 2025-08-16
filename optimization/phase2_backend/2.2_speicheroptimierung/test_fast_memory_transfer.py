#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark-Test für optimierten Speichertransfer

Dieser Test vergleicht die Leistung des ursprünglichen und optimierten
Speichertransfers für die T-Mathematics Engine.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from collections import defaultdict

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("test_fast_memory_transfer")

# MLX-Verfügbarkeit prüfen
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    logger.warning("MLX konnte nicht importiert werden. MLX-Benchmarks werden übersprungen.")

# Füge Projektverzeichnis zum Pfad hinzu für Importe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Importiere ursprüngliche MLX-Unterstützung
from miso.math.t_mathematics.mlx_support import MLXBackend

# Importiere neue optimierte Speichertransferfunktionen
from fast_memory_transfer import mps_to_mlx, mlx_to_mps

# Überprüfe, ob MPS verfügbar ist
MPS_AVAILABLE = torch.backends.mps.is_available()

def benchmark_tensor_transfer(shape=(1024, 1024), num_transfers=100):
    """
    Benchmarkt die Tensorübertragung zwischen verschiedenen Geräten
    
    Args:
        shape: Form des Testtensors
        num_transfers: Anzahl der Transferoperationen
        
    Returns:
        Dictionary mit Benchmark-Ergebnissen
    """
    results = {}
    
    if not MPS_AVAILABLE or not HAS_MLX:
        logger.warning("MPS oder MLX nicht verfügbar, Benchmark wird übersprungen")
        return {"error": "MPS oder MLX nicht verfügbar"}
    
    logger.info(f"Benchmarking Tensorübertragung mit Shape {shape}, {num_transfers} Transfers")
    
    # Initialisiere ursprüngliches MLX-Backend
    original_mlx = MLXBackend(precision="float16")
    
    # Erstelle PyTorch-Tensor auf MPS
    torch_tensor = torch.randn(shape, device="mps")
    
    # Benchmarkt ursprüngliche MPS → MLX Implementierung
    start_time = time.time()
    for _ in range(num_transfers):
        mlx_array = original_mlx.to_mlx(torch_tensor)
    original_mps_to_mlx_time = time.time() - start_time
    logger.info(f"Ursprüngliche MPS → MLX: {original_mps_to_mlx_time:.6f}s")
    
    # Erstelle MLX-Array für MLX → MPS Test
    mlx_array = mx.random.normal(shape)
    
    # Benchmarkt ursprüngliche MLX → MPS Implementierung
    start_time = time.time()
    for _ in range(num_transfers):
        torch_result = original_mlx.to_torch(mlx_array)
    original_mlx_to_mps_time = time.time() - start_time
    logger.info(f"Ursprüngliche MLX → MPS: {original_mlx_to_mps_time:.6f}s")
    
    # Benchmarkt optimierte MPS → MLX Implementierung
    torch_tensor = torch.randn(shape, device="mps")
    start_time = time.time()
    for _ in range(num_transfers):
        mlx_array = mps_to_mlx(torch_tensor)
    optimized_mps_to_mlx_time = time.time() - start_time
    logger.info(f"Optimierte MPS → MLX: {optimized_mps_to_mlx_time:.6f}s")
    
    # Benchmarkt optimierte MLX → MPS Implementierung
    mlx_array = mx.random.normal(shape)
    start_time = time.time()
    for _ in range(num_transfers):
        torch_result = mlx_to_mps(mlx_array)
    optimized_mlx_to_mps_time = time.time() - start_time
    logger.info(f"Optimierte MLX → MPS: {optimized_mlx_to_mps_time:.6f}s")
    
    # Berechne Speedups
    mps_to_mlx_speedup = original_mps_to_mlx_time / optimized_mps_to_mlx_time if optimized_mps_to_mlx_time > 0 else float('inf')
    mlx_to_mps_speedup = original_mlx_to_mps_time / optimized_mlx_to_mps_time if optimized_mlx_to_mps_time > 0 else float('inf')
    
    logger.info(f"MPS → MLX Speedup: {mps_to_mlx_speedup:.2f}x")
    logger.info(f"MLX → MPS Speedup: {mlx_to_mps_speedup:.2f}x")
    
    # Ergebnisse sammeln
    results = {
        "shape": shape,
        "num_transfers": num_transfers,
        "original_mps_to_mlx_time": original_mps_to_mlx_time,
        "optimized_mps_to_mlx_time": optimized_mps_to_mlx_time,
        "mps_to_mlx_speedup": mps_to_mlx_speedup,
        "original_mlx_to_mps_time": original_mlx_to_mps_time,
        "optimized_mlx_to_mps_time": optimized_mlx_to_mps_time,
        "mlx_to_mps_speedup": mlx_to_mps_speedup
    }
    
    return results

def run_benchmarks():
    """Führt Benchmarks mit verschiedenen Tensorgrößen durch"""
    tensor_shapes = [
        (32, 32),       # Klein
        (128, 128),     # Mittel
        (512, 512),     # Groß
        (1024, 1024)    # Sehr groß
    ]
    
    results = {}
    
    for shape in tensor_shapes:
        logger.info(f"Führe Benchmark mit Tensorgröße {shape} durch...")
        result = benchmark_tensor_transfer(shape=shape, num_transfers=50)
        shape_key = f"{shape[0]}x{shape[1]}"
        results[shape_key] = result
    
    # Ausgabe der zusammenfassenden Ergebnisse
    logger.info("\n===== ZUSAMMENFASSUNG =====")
    logger.info("Tensorform | MPS→MLX Speedup | MLX→MPS Speedup")
    logger.info("----------|---------------|---------------")
    
    for shape_key, result in results.items():
        if "error" in result:
            logger.info(f"{shape_key} | FEHLER: {result['error']}")
        else:
            logger.info(f"{shape_key} | {result['mps_to_mlx_speedup']:.2f}x | {result['mlx_to_mps_speedup']:.2f}x")
    
    return results

if __name__ == "__main__":
    run_benchmarks()
