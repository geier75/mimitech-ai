#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVD-Benchmark für T-Mathematics Engine

Dieser Benchmark vergleicht die Leistung verschiedener SVD-Implementierungen:
1. Ursprüngliches MLXBackend
2. Optimiertes MLXBackend mit MLXSVDOptimizer
3. PyTorch (MPS) als Referenz

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("svd_benchmark")

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

# Importiere MLXBackend
from miso.math.t_mathematics.mlx_support import MLXBackend

# Importiere die optimierte SVD-Implementierung und den Backend-Enhancer
from optimized_mlx_svd import MLXSVDOptimizer
from mlx_backend_enhancer import optimize_mlx_backend

def get_backends():
    """
    Erstellt und konfiguriert die verschiedenen Backends
    
    Returns:
        Dictionary mit konfigurierten Backends
    """
    backends = {}
    
    # PyTorch für Referenz
    backends["pytorch"] = {
        "name": "PyTorch",
        "available": torch.backends.mps.is_available()
    }
    if backends["pytorch"]["available"]:
        backends["pytorch"]["device"] = torch.device("mps")
    else:
        backends["pytorch"]["device"] = torch.device("cpu")
    
    # Ursprüngliches MLXBackend
    backends["original_mlx"] = {
        "name": "Original MLX",
        "available": HAS_MLX,
        "backend": MLXBackend(precision="float32") if HAS_MLX else None
    }
    
    # Optimiertes MLXBackend
    backends["optimized_mlx"] = {
        "name": "Optimized MLX",
        "available": HAS_MLX,
        "backend": None
    }
    
    if HAS_MLX:
        # Erstelle eine zweite Instanz des MLXBackend
        optimized_backend = MLXBackend(precision="float32")
        # Optimiere dieses Backend
        optimized_backend = optimize_mlx_backend(optimized_backend)
        backends["optimized_mlx"]["backend"] = optimized_backend
    
    return backends

def create_test_matrices(shapes):
    """
    Erzeugt Testmatrizen für den Benchmark
    
    Args:
        shapes: Liste von Tupeln mit Matrixformen
        dtype: NumPy-Datentyp für die Matrizen
        
    Returns:
        Dictionary mit Testmatrizen
    """
    matrices = {}
    
    for shape in shapes:
        # Erstelle eine zufällige Matrix mit NumPy, immer als float32 für MPS-Kompatibilität
        np_matrix = np.random.rand(*shape).astype(np.float32)
        
        # Erstelle eine konditionierte Matrix für eine stabilere SVD
        # Wir verwenden eine Matrix mit bekannten, gut verteilten Singulärwerten
        u, _, v = np.linalg.svd(np_matrix, full_matrices=False)
        s = np.linspace(1.0, 0.01, min(shape), dtype=np.float32)  # Gut verteilte Singulärwerte
        np_matrix = np.dot(u * s, v)
        
        # Speichere NumPy-Matrix
        matrices[f"{shape[0]}x{shape[1]}"] = {
            "numpy": np_matrix
        }
        
        # Erstelle PyTorch-Tensor, wenn verfügbar - explizit als float32 für MPS-Kompatibilität
        if torch.backends.mps.is_available():
            torch_matrix = torch.tensor(np_matrix, device="mps", dtype=torch.float32)
        else:
            torch_matrix = torch.tensor(np_matrix, dtype=torch.float32)
        matrices[f"{shape[0]}x{shape[1]}"]["torch"] = torch_matrix
        
        # Erstelle MLX-Array, wenn verfügbar
        if HAS_MLX:
            mlx_matrix = mx.array(np_matrix)
            matrices[f"{shape[0]}x{shape[1]}"]["mlx"] = mlx_matrix
    
    return matrices

def run_pytorch_svd(matrix, k=None, device=None):
    """
    Führt eine SVD mit PyTorch durch
    
    Args:
        matrix: PyTorch-Tensor
        k: Anzahl der zu berechnenden Singulärwerte
        device: Gerät für die Berechnung
        
    Returns:
        SVD-Ergebnis, Ausführungszeit
    """
    if device is not None and str(matrix.device) != str(device):
        matrix = matrix.to(device)
    
    start_time = time.time()
    
    if k is None:
        # Vollständige SVD
        u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    else:
        # Partielle SVD, verwende svd_lowrank wenn möglich
        try:
            u, s, v = torch.svd_lowrank(matrix, q=k)
        except (AttributeError, RuntimeError):
            # Fallback auf vollständige SVD mit Trunkierung
            u, s, v = torch.linalg.svd(matrix, full_matrices=False)
            u, s, v = u[:, :k], s[:k], v[:k, :]
    
    # Erzwinge Evaluation
    _ = u[0, 0].item()
    
    end_time = time.time()
    return (u, s, v), end_time - start_time

def benchmark_svd(matrix_shapes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)], 
                 k_values=[None, 10], num_runs=3):
    """
    Führt einen SVD-Benchmark für verschiedene Implementierungen durch
    
    Args:
        matrix_shapes: Liste von Tupeln mit Matrixformen
        k_values: Liste der zu testenden k-Werte
        num_runs: Anzahl der Durchläufe pro Konfiguration
        
    Returns:
        Dictionary mit Benchmark-Ergebnissen
    """
    # Hole die zu testenden Backends
    backends = get_backends()
    
    # Erstelle Testmatrizen
    test_matrices = create_test_matrices(matrix_shapes)
    
    # Ergebnisse sammeln
    results = {}
    
    for shape_key, matrices in test_matrices.items():
        shape_results = {}
        
        for k in k_values:
            # Prüfe, ob k gültig ist für diese Matrix
            shape = matrices["numpy"].shape
            if k is not None and k > min(shape):
                continue
            
            k_results = {}
            k_str = str(k) if k is not None else "None"
            logger.info(f"Teste SVD für Matrix {shape_key} mit k={k_str}")
            
            # 1. PyTorch Benchmark
            if backends["pytorch"]["available"]:
                times = []
                for _ in range(num_runs):
                    _, runtime = run_pytorch_svd(matrices["torch"], k, backends["pytorch"]["device"])
                    times.append(runtime)
                avg_time = sum(times) / len(times)
                k_results["pytorch"] = {"avg_time": avg_time, "times": times}
                logger.info(f"  PyTorch (MPS): {avg_time:.6f}s")
            
            # 2. Original MLX Benchmark
            if backends["original_mlx"]["available"]:
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    u, s, v = backends["original_mlx"]["backend"].svd(matrices["numpy"], k)
                    # Erzwinge Evaluation
                    if HAS_MLX and hasattr(u, 'item'):
                        _ = u[0, 0].item()
                    end_time = time.time()
                    times.append(end_time - start_time)
                avg_time = sum(times) / len(times)
                k_results["original_mlx"] = {"avg_time": avg_time, "times": times}
                logger.info(f"  Original MLX: {avg_time:.6f}s")
            
            # 3. Optimiertes MLX Benchmark
            if backends["optimized_mlx"]["available"]:
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    u, s, v = backends["optimized_mlx"]["backend"].svd(matrices["numpy"], k)
                    # Erzwinge Evaluation
                    if HAS_MLX and hasattr(u, 'item'):
                        _ = u[0, 0].item()
                    end_time = time.time()
                    times.append(end_time - start_time)
                avg_time = sum(times) / len(times)
                k_results["optimized_mlx"] = {"avg_time": avg_time, "times": times}
                logger.info(f"  Optimiertes MLX: {avg_time:.6f}s")
            
            # Berechne Speedups
            speedups = {}
            if "original_mlx" in k_results and "optimized_mlx" in k_results:
                speedup = k_results["original_mlx"]["avg_time"] / k_results["optimized_mlx"]["avg_time"]
                speedups["mlx_vs_optimized"] = speedup
                logger.info(f"  Speedup (Optimiert vs. Original): {speedup:.2f}x")
            
            if "pytorch" in k_results and "optimized_mlx" in k_results:
                speedup = k_results["pytorch"]["avg_time"] / k_results["optimized_mlx"]["avg_time"]
                speedups["pytorch_vs_optimized"] = speedup
                logger.info(f"  Speedup (Optimiert vs. PyTorch): {speedup:.2f}x")
            
            k_results["speedups"] = speedups
            shape_results[f"k={k}"] = k_results
        
        results[shape_key] = shape_results
    
    return results

def visualize_results(results):
    """
    Visualisiert die Benchmark-Ergebnisse
    
    Args:
        results: Dictionary mit Benchmark-Ergebnissen
    """
    # Sammle Daten für das Diagramm
    shapes = []
    pytorch_times = []
    original_mlx_times = []
    optimized_mlx_times = []
    
    # Extrahiere Daten für k=None (vollständige SVD)
    for shape_key, shape_results in results.items():
        if "k=None" in shape_results:
            shapes.append(shape_key)
            
            if "pytorch" in shape_results["k=None"]:
                pytorch_times.append(shape_results["k=None"]["pytorch"]["avg_time"])
            else:
                pytorch_times.append(None)
            
            if "original_mlx" in shape_results["k=None"]:
                original_mlx_times.append(shape_results["k=None"]["original_mlx"]["avg_time"])
            else:
                original_mlx_times.append(None)
            
            if "optimized_mlx" in shape_results["k=None"]:
                optimized_mlx_times.append(shape_results["k=None"]["optimized_mlx"]["avg_time"])
            else:
                optimized_mlx_times.append(None)
    
    # Erstelle Diagramm
    try:
        plt.figure(figsize=(12, 8))
        
        x = range(len(shapes))
        
        if any(t is not None for t in pytorch_times):
            plt.plot(x, [t if t is not None else 0 for t in pytorch_times], 'b.-', label='PyTorch (MPS)')
        
        if any(t is not None for t in original_mlx_times):
            plt.plot(x, [t if t is not None else 0 for t in original_mlx_times], 'r.-', label='Original MLX')
        
        if any(t is not None for t in optimized_mlx_times):
            plt.plot(x, [t if t is not None else 0 for t in optimized_mlx_times], 'g.-', label='Optimiertes MLX')
        
        plt.xlabel('Matrix-Größe')
        plt.ylabel('Ausführungszeit (s)')
        plt.title('SVD Leistungsvergleich')
        plt.xticks(x, shapes)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), 'svd_benchmark.png'))
        logger.info(f"Diagramm gespeichert als 'svd_benchmark.png'")
    except Exception as e:
        logger.error(f"Konnte Ergebnisse nicht visualisieren: {e}")
    
    # Erstelle Tabelle in einfachem Format ohne tabulate
    try:
        logger.info("\n=== SVD-BENCHMARK-ERGEBNISSE ===")
        # Tabellenüberschrift
        logger.info(f"{'Matrix':<10} {'k':<6} {'PyTorch':<12} {'Original MLX':<14} {'Optimiert MLX':<14} {'Speedup':<8}")
        logger.info("-" * 70)
        
        for shape_key in shapes:
            for k_key, k_results in results[shape_key].items():
                # Initialisiere Zellenwerte
                pytorch_val = "N/A"
                orig_mlx_val = "N/A"
                opt_mlx_val = "N/A"
                speedup_val = "N/A"
                
                # PyTorch
                if "pytorch" in k_results:
                    pytorch_val = f"{k_results['pytorch']['avg_time']:.6f}s"
                
                # Original MLX
                if "original_mlx" in k_results:
                    orig_mlx_val = f"{k_results['original_mlx']['avg_time']:.6f}s"
                
                # Optimiertes MLX
                if "optimized_mlx" in k_results:
                    opt_mlx_val = f"{k_results['optimized_mlx']['avg_time']:.6f}s"
                
                # Speedup
                if "speedups" in k_results and "mlx_vs_optimized" in k_results["speedups"]:
                    speedup_val = f"{k_results['speedups']['mlx_vs_optimized']:.2f}x"
                
                # Gib formatierte Zeile aus
                logger.info(f"{shape_key:<10} {k_key.replace('k=', ''):<6} {pytorch_val:<12} {orig_mlx_val:<14} {opt_mlx_val:<14} {speedup_val:<8}")
        
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Konnte Tabelle nicht erstellen: {e}")

def verify_svd_accuracy(shape=(64, 32), k=None):
    """
    Überprüft die Genauigkeit der verschiedenen SVD-Implementierungen
    
    Args:
        shape: Form der Testmatrix
        k: Anzahl der zu berechnenden Singulärwerte
        
    Returns:
        Dictionary mit Genauigkeitsergebnissen
    """
    logger.info(f"Überprüfe SVD-Genauigkeit für Matrix {shape} mit k={k}")
    
    # Erstelle eine bekannte Matrix
    np.random.seed(42)  # Für Reproduzierbarkeit
    np_matrix = np.random.rand(*shape).astype(np.float32)  # Immer float32 für MPS-Kompatibilität
    
    # Hole die zu testenden Backends
    backends = get_backends()
    
    # Referenz-Ergebnis berechnen (NumPy gilt als Gold-Standard)
    if k is None:
        u_ref, s_ref, v_ref = np.linalg.svd(np_matrix, full_matrices=False)
    else:
        u_ref, s_ref, v_ref = np.linalg.svd(np_matrix, full_matrices=False)
        u_ref, s_ref, v_ref = u_ref[:, :k], s_ref[:k], v_ref[:k, :]
    
    results = {}
    
    # 1. Original MLX
    if backends["original_mlx"]["available"]:
        try:
            u, s, v = backends["original_mlx"]["backend"].svd(np_matrix, k)
            
            # Konvertiere zu NumPy für Vergleich
            if HAS_MLX:
                if hasattr(u, 'tolist'):
                    u = np.array(u.tolist())
                    s = np.array(s.tolist())
                    v = np.array(v.tolist())
            
            # Berechne Fehler in Singulärwerten (relativer Fehler)
            s_error = np.mean(np.abs(s - s_ref) / (s_ref + 1e-8))
            
            # Berechne Fehler in der Rekonstruktion
            reconstructed = np.dot(u * s, v)
            reconstruction_error = np.mean(np.abs(reconstructed - np_matrix))
            
            results["original_mlx"] = {
                "s_error": s_error,
                "reconstruction_error": reconstruction_error
            }
            
            logger.info(f"  Original MLX - S-Fehler: {s_error:.6f}, Rekonstruktionsfehler: {reconstruction_error:.6f}")
        except Exception as e:
            logger.error(f"  Original MLX SVD fehlgeschlagen: {e}")
    
    # 2. Optimiertes MLX
    if backends["optimized_mlx"]["available"]:
        try:
            u, s, v = backends["optimized_mlx"]["backend"].svd(np_matrix, k)
            
            # Konvertiere zu NumPy für Vergleich
            if HAS_MLX:
                if hasattr(u, 'tolist'):
                    u = np.array(u.tolist())
                    s = np.array(s.tolist())
                    v = np.array(v.tolist())
            
            # Berechne Fehler in Singulärwerten (relativer Fehler)
            s_error = np.mean(np.abs(s - s_ref) / (s_ref + 1e-8))
            
            # Berechne Fehler in der Rekonstruktion
            reconstructed = np.dot(u * s, v)
            reconstruction_error = np.mean(np.abs(reconstructed - np_matrix))
            
            results["optimized_mlx"] = {
                "s_error": s_error,
                "reconstruction_error": reconstruction_error
            }
            
            logger.info(f"  Optimiertes MLX - S-Fehler: {s_error:.6f}, Rekonstruktionsfehler: {reconstruction_error:.6f}")
        except Exception as e:
            logger.error(f"  Optimiertes MLX SVD fehlgeschlagen: {e}")
    
    # Zusammenfassung
    logger.info("SVD-Genauigkeitsüberprüfung abgeschlossen")
    return results

if __name__ == "__main__":
    # Verifiziere die Genauigkeit
    verify_results = verify_svd_accuracy()
    
    # Führe den Benchmark durch
    benchmark_results = benchmark_svd(num_runs=2)  # Reduziere num_runs für schnellere Ausführung
    
    # Visualisiere die Ergebnisse
    visualize_results(benchmark_results)
