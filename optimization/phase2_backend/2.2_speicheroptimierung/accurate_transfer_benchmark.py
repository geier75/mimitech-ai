#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Präziser Benchmark für Tensor-Transfers zwischen MPS und MLX

Dieser Test sorgt durch Zugriff auf die Daten für vollständige Evaluation
der Transfers und liefert somit akkurate Benchmarks.

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
logger = logging.getLogger("accurate_transfer_benchmark")

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

def force_evaluation(tensor_or_array):
    """
    Erzwingt die Auswertung eines Tensors oder Arrays durch Datenzugriff
    
    Args:
        tensor_or_array: PyTorch-Tensor oder MLX-Array
        
    Returns:
        Der ursprüngliche Tensor oder Array nach Auswertung
    """
    if isinstance(tensor_or_array, torch.Tensor):
        # Bei PyTorch: sum() erzwingt Evaluation
        _ = tensor_or_array.sum().item()
    elif HAS_MLX and hasattr(tensor_or_array, "__module__") and "mlx" in tensor_or_array.__module__:
        # Bei MLX: Konvertierung und Zugriff auf Daten erzwingt Evaluation
        _ = float(mx.sum(tensor_or_array).item())
    
    return tensor_or_array

def benchmark_tensor_transfer(shape=(1024, 1024), num_transfers=20, verbose=True):
    """
    Benchmarkt die Tensorübertragung zwischen verschiedenen Geräten
    mit erzwungener Evaluation
    
    Args:
        shape: Form des Testtensors
        num_transfers: Anzahl der Transferoperationen
        verbose: Ob detaillierte Ausgaben erfolgen sollen
        
    Returns:
        Dictionary mit Benchmark-Ergebnissen
    """
    results = {}
    
    if not MPS_AVAILABLE or not HAS_MLX:
        logger.warning("MPS oder MLX nicht verfügbar, Benchmark wird übersprungen")
        return {"error": "MPS oder MLX nicht verfügbar"}
    
    if verbose:
        logger.info(f"Benchmarking Tensorübertragung mit Shape {shape}, {num_transfers} Transfers")
    
    # Initialisiere ursprüngliches MLX-Backend
    original_mlx = MLXBackend(precision="float16")
    
    # Erstelle PyTorch-Tensor auf MPS
    torch_tensor = torch.randn(shape, device="mps")
    
    # Benchmarkt ursprüngliche MPS → MLX Implementierung
    start_time = time.time()
    for _ in range(num_transfers):
        mlx_array = original_mlx.to_mlx(torch_tensor)
        force_evaluation(mlx_array)  # Erzwinge Evaluation
    original_mps_to_mlx_time = time.time() - start_time
    if verbose:
        logger.info(f"Ursprüngliche MPS → MLX: {original_mps_to_mlx_time:.6f}s")
    
    # Erstelle MLX-Array für MLX → MPS Test
    mlx_array = mx.random.normal(shape)
    force_evaluation(mlx_array)  # Erzwinge Evaluation
    
    # Benchmarkt ursprüngliche MLX → MPS Implementierung
    start_time = time.time()
    for _ in range(num_transfers):
        torch_result = original_mlx.to_torch(mlx_array)
        force_evaluation(torch_result)  # Erzwinge Evaluation
    original_mlx_to_mps_time = time.time() - start_time
    if verbose:
        logger.info(f"Ursprüngliche MLX → MPS: {original_mlx_to_mps_time:.6f}s")
    
    # Benchmarkt optimierte MPS → MLX Implementierung
    torch_tensor = torch.randn(shape, device="mps")
    force_evaluation(torch_tensor)  # Erzwinge Evaluation
    
    start_time = time.time()
    for _ in range(num_transfers):
        mlx_array = mps_to_mlx(torch_tensor)
        force_evaluation(mlx_array)  # Erzwinge Evaluation
    optimized_mps_to_mlx_time = time.time() - start_time
    if verbose:
        logger.info(f"Optimierte MPS → MLX: {optimized_mps_to_mlx_time:.6f}s")
    
    # Benchmarkt optimierte MLX → MPS Implementierung
    mlx_array = mx.random.normal(shape)
    force_evaluation(mlx_array)  # Erzwinge Evaluation
    
    start_time = time.time()
    for _ in range(num_transfers):
        torch_result = mlx_to_mps(mlx_array)
        force_evaluation(torch_result)  # Erzwinge Evaluation
    optimized_mlx_to_mps_time = time.time() - start_time
    if verbose:
        logger.info(f"Optimierte MLX → MPS: {optimized_mlx_to_mps_time:.6f}s")
    
    # Berechne Speedups
    mps_to_mlx_speedup = original_mps_to_mlx_time / optimized_mps_to_mlx_time if optimized_mps_to_mlx_time > 0 else float('inf')
    mlx_to_mps_speedup = original_mlx_to_mps_time / optimized_mlx_to_mps_time if optimized_mlx_to_mps_time > 0 else float('inf')
    
    if verbose:
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
        result = benchmark_tensor_transfer(shape=shape, num_transfers=20)
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

def inspect_source_code():
    """Inspiziert den Quellcode der ursprünglichen Implementierung"""
    try:
        import inspect
        from miso.math.t_mathematics.mlx_support import MLXBackend
        
        logger.info("\n===== QUELLCODE-ANALYSE =====")
        
        # Analysiere to_mlx
        to_mlx_source = inspect.getsource(MLXBackend.to_mlx)
        logger.info("MLXBackend.to_mlx Quellcode:")
        logger.info(to_mlx_source)
        
        # Analysiere to_torch
        to_torch_source = inspect.getsource(MLXBackend.to_torch)
        logger.info("MLXBackend.to_torch Quellcode:")
        logger.info(to_torch_source)
    except Exception as e:
        logger.error(f"Fehler bei der Quellcode-Analyse: {e}")

if __name__ == "__main__":
    run_benchmarks()
    inspect_source_code()
