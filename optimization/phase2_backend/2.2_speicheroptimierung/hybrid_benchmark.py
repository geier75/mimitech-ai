#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark für die hybride Speichertransferstrategie

Dieser Test vergleicht die ursprüngliche, die optimierte und die hybride
Implementierung der Speichertransfers zwischen MPS und MLX.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("hybrid_benchmark")

# MLX-Verfügbarkeit prüfen
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    logger.warning("MLX konnte nicht importiert werden. MLX-Benchmarks werden übersprungen.")

# Überprüfe, ob MPS verfügbar ist
MPS_AVAILABLE = torch.backends.mps.is_available()

# Füge Projektverzeichnis zum Pfad hinzu für Importe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Importiere die verschiedenen Implementierungen
from miso.math.t_mathematics.mlx_support import MLXBackend
from fast_memory_transfer import mps_to_mlx as fast_mps_to_mlx, mlx_to_mps as fast_mlx_to_mps
from hybrid_memory_transfer import mps_to_mlx as hybrid_mps_to_mlx, mlx_to_mps as hybrid_mlx_to_mps

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

def benchmark_comparison(shape=(1024, 1024), num_transfers=20, verbose=True):
    """
    Vergleicht die Leistung der verschiedenen Implementierungen
    
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
    
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    
    if verbose:
        logger.info(f"Tensor-Vergleichstest mit Shape {shape} ({num_elements} Elemente), {num_transfers} Transfers")
    
    # Initialisiere ursprüngliches MLX-Backend
    original_mlx = MLXBackend(precision="float16")
    
    # ---- MPS → MLX Transfers ----
    
    # Vorbereite Tensoren
    torch_tensor = torch.randn(shape, device="mps")
    force_evaluation(torch_tensor)
    
    # 1. Ursprüngliche Implementierung
    start_time = time.time()
    for _ in range(num_transfers):
        mlx_array = original_mlx.to_mlx(torch_tensor)
        force_evaluation(mlx_array)
    original_mps_to_mlx_time = time.time() - start_time
    if verbose:
        logger.info(f"Ursprüngliche MPS → MLX: {original_mps_to_mlx_time:.6f}s")
    
    # 2. Unsere optimierte Implementierung
    start_time = time.time()
    for _ in range(num_transfers):
        mlx_array = fast_mps_to_mlx(torch_tensor)
        force_evaluation(mlx_array)
    fast_mps_to_mlx_time = time.time() - start_time
    if verbose:
        logger.info(f"Optimierte MPS → MLX: {fast_mps_to_mlx_time:.6f}s")
    
    # 3. Hybride Implementierung
    start_time = time.time()
    for _ in range(num_transfers):
        mlx_array = hybrid_mps_to_mlx(torch_tensor)
        force_evaluation(mlx_array)
    hybrid_mps_to_mlx_time = time.time() - start_time
    if verbose:
        logger.info(f"Hybride MPS → MLX: {hybrid_mps_to_mlx_time:.6f}s")
    
    # ---- MLX → MPS Transfers ----
    
    # Vorbereite Arrays
    mlx_array = mx.random.normal(shape)
    force_evaluation(mlx_array)
    
    # 1. Ursprüngliche Implementierung
    start_time = time.time()
    for _ in range(num_transfers):
        torch_result = original_mlx.to_torch(mlx_array)
        force_evaluation(torch_result)
    original_mlx_to_mps_time = time.time() - start_time
    if verbose:
        logger.info(f"Ursprüngliche MLX → MPS: {original_mlx_to_mps_time:.6f}s")
    
    # 2. Unsere optimierte Implementierung
    start_time = time.time()
    for _ in range(num_transfers):
        torch_result = fast_mlx_to_mps(mlx_array)
        force_evaluation(torch_result)
    fast_mlx_to_mps_time = time.time() - start_time
    if verbose:
        logger.info(f"Optimierte MLX → MPS: {fast_mlx_to_mps_time:.6f}s")
    
    # 3. Hybride Implementierung
    start_time = time.time()
    for _ in range(num_transfers):
        torch_result = hybrid_mlx_to_mps(mlx_array)
        force_evaluation(torch_result)
    hybrid_mlx_to_mps_time = time.time() - start_time
    if verbose:
        logger.info(f"Hybride MLX → MPS: {hybrid_mlx_to_mps_time:.6f}s")
    
    # Berechne Speedups (relativ zur ursprünglichen Implementierung)
    fast_mps_to_mlx_speedup = original_mps_to_mlx_time / fast_mps_to_mlx_time if fast_mps_to_mlx_time > 0 else float('inf')
    hybrid_mps_to_mlx_speedup = original_mps_to_mlx_time / hybrid_mps_to_mlx_time if hybrid_mps_to_mlx_time > 0 else float('inf')
    
    fast_mlx_to_mps_speedup = original_mlx_to_mps_time / fast_mlx_to_mps_time if fast_mlx_to_mps_time > 0 else float('inf')
    hybrid_mlx_to_mps_speedup = original_mlx_to_mps_time / hybrid_mlx_to_mps_time if hybrid_mlx_to_mps_time > 0 else float('inf')
    
    if verbose:
        logger.info(f"Optimierte MPS → MLX Speedup: {fast_mps_to_mlx_speedup:.2f}x")
        logger.info(f"Hybride MPS → MLX Speedup: {hybrid_mps_to_mlx_speedup:.2f}x")
        logger.info(f"Optimierte MLX → MPS Speedup: {fast_mlx_to_mps_speedup:.2f}x")
        logger.info(f"Hybride MLX → MPS Speedup: {hybrid_mlx_to_mps_speedup:.2f}x")
    
    # Ergebnisse sammeln
    results = {
        "shape": shape,
        "num_elements": num_elements,
        "num_transfers": num_transfers,
        "original_mps_to_mlx_time": original_mps_to_mlx_time,
        "fast_mps_to_mlx_time": fast_mps_to_mlx_time,
        "hybrid_mps_to_mlx_time": hybrid_mps_to_mlx_time,
        "fast_mps_to_mlx_speedup": fast_mps_to_mlx_speedup,
        "hybrid_mps_to_mlx_speedup": hybrid_mps_to_mlx_speedup,
        "original_mlx_to_mps_time": original_mlx_to_mps_time,
        "fast_mlx_to_mps_time": fast_mlx_to_mps_time,
        "hybrid_mlx_to_mps_time": hybrid_mlx_to_mps_time,
        "fast_mlx_to_mps_speedup": fast_mlx_to_mps_speedup, 
        "hybrid_mlx_to_mps_speedup": hybrid_mlx_to_mps_speedup
    }
    
    return results

def run_benchmarks():
    """Führt Benchmarks mit verschiedenen Tensorgrößen durch"""
    tensor_shapes = [
        (32, 32),       # Klein
        (64, 64),       # Klein-Mittel
        (128, 128),     # Mittel
        (256, 256),     # Mittel-Groß
        (512, 512),     # Groß
        (1024, 1024)    # Sehr groß
    ]
    
    results = {}
    
    for shape in tensor_shapes:
        shape_str = f"{shape[0]}x{shape[1]}"
        logger.info(f"Führe Benchmark mit Tensorgröße {shape_str} durch...")
        result = benchmark_comparison(shape=shape, num_transfers=10)
        results[shape_str] = result
    
    # Ausgabe der zusammenfassenden Ergebnisse
    logger.info("\n===== ZUSAMMENFASSUNG =====")
    logger.info("Tensorform | Elemente | MPS→MLX Speedup (Opt/Hybrid) | MLX→MPS Speedup (Opt/Hybrid)")
    logger.info("----------|----------|---------------------------|---------------------------")
    
    mps_to_mlx_speedups = {"optimiert": [], "hybrid": []}
    mlx_to_mps_speedups = {"optimiert": [], "hybrid": []}
    tensor_sizes = []
    
    for shape_key, result in sorted(results.items(), key=lambda x: int(x[0].split('x')[0])):
        if "error" in result:
            logger.info(f"{shape_key} | FEHLER: {result['error']}")
        else:
            tensor_sizes.append(result["num_elements"])
            mps_to_mlx_speedups["optimiert"].append(result["fast_mps_to_mlx_speedup"])
            mps_to_mlx_speedups["hybrid"].append(result["hybrid_mps_to_mlx_speedup"])
            mlx_to_mps_speedups["optimiert"].append(result["fast_mlx_to_mps_speedup"])
            mlx_to_mps_speedups["hybrid"].append(result["hybrid_mlx_to_mps_speedup"])
            
            logger.info(f"{shape_key} | {result['num_elements']} | {result['fast_mps_to_mlx_speedup']:.2f}x / {result['hybrid_mps_to_mlx_speedup']:.2f}x | {result['fast_mlx_to_mps_speedup']:.2f}x / {result['hybrid_mlx_to_mps_speedup']:.2f}x")
    
    # Ergebnisse visualisieren
    visualize_results(tensor_sizes, mps_to_mlx_speedups, mlx_to_mps_speedups)
    
    return results

def visualize_results(tensor_sizes, mps_to_mlx_speedups, mlx_to_mps_speedups):
    """Visualisiert die Benchmarkergebnisse in einem Diagramm"""
    try:
        # Erstelle Diagramm
        plt.figure(figsize=(12, 8))
        
        # MPS → MLX Speedups
        plt.subplot(1, 2, 1)
        plt.semilogx(tensor_sizes, mps_to_mlx_speedups["optimiert"], 'b.-', label='Optimiert')
        plt.semilogx(tensor_sizes, mps_to_mlx_speedups["hybrid"], 'r.-', label='Hybrid')
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel('Tensor-Elemente')
        plt.ylabel('Speedup (relativ zum Original)')
        plt.title('MPS → MLX Speedup')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # MLX → MPS Speedups
        plt.subplot(1, 2, 2)
        plt.semilogx(tensor_sizes, mlx_to_mps_speedups["optimiert"], 'b.-', label='Optimiert')
        plt.semilogx(tensor_sizes, mlx_to_mps_speedups["hybrid"], 'r.-', label='Hybrid')
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.7)
        plt.xlabel('Tensor-Elemente')
        plt.ylabel('Speedup (relativ zum Original)')
        plt.title('MLX → MPS Speedup')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'speichertransfer_speedup.png'))
        logger.info(f"Diagramm gespeichert als 'speichertransfer_speedup.png'")
    except Exception as e:
        logger.error(f"Konnte Ergebnisse nicht visualisieren: {e}")

if __name__ == "__main__":
    run_benchmarks()
