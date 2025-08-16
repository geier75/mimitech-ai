#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abschließender Test der optimierten MatrixCore-Implementierung
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import json

# Pfad zur matrix_core.py Datei hinzufügen
matrix_core_path = os.path.join(os.path.dirname(__file__), 'vxor.ai', 'VX-MATRIX', 'core')
sys.path.insert(0, matrix_core_path)

from matrix_core import MatrixCore

def run_comprehensive_test():
    print("=== VX-MATRIX Performance-Benchmarking ===")
    print("\nInitialisiere MatrixCore-Implementierung...")
    
    # MatrixCore mit MLX als bevorzugtem Backend initialisieren
    core_mlx = MatrixCore(preferred_backend="mlx")
    core_mlx.enable_jit = True
    
    # MatrixCore mit NumPy als bevorzugtem Backend initialisieren (als Referenz)
    core_np = MatrixCore(preferred_backend="numpy")
    
    print(f"JIT-Kompilierung: {'aktiviert' if core_mlx.enable_jit else 'deaktiviert'}")
    
    # Testgrößen
    sizes = [10, 50, 100, 200, 500]
    results = {
        'matmul': {'mlx': {}, 'numpy': {}},
        'inverse': {'mlx': {}, 'numpy': {}}
    }
    
    # Matrixmultiplikation
    print("\n1. Matrix-Multiplikations-Tests")
    for size in sizes:
        print(f"  Teste Größe {size}x{size}...")
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        
        # MLX Performance
        start = time.time()
        for _ in range(5):  # 5 Wiederholungen für stabilere Messungen
            core_mlx.matrix_multiply(a, b)
        mlx_time = (time.time() - start) / 5
        results['matmul']['mlx'][size] = mlx_time
        
        # NumPy Performance
        start = time.time()
        for _ in range(5):
            core_np.matrix_multiply(a, b)
        np_time = (time.time() - start) / 5
        results['matmul']['numpy'][size] = np_time
        
        # Speedup berechnen
        speedup = np_time / mlx_time if mlx_time < np_time else -mlx_time / np_time
        print(f"    MLX: {mlx_time:.6f}s, NumPy: {np_time:.6f}s - Speedup: {speedup:.2f}x")
        
        # Genauigkeit überprüfen
        np_result = np.matmul(a, b)
        mlx_result = core_mlx.matrix_multiply(a, b)
        if isinstance(mlx_result, np.ndarray):
            accuracy = np.allclose(mlx_result, np_result, rtol=1e-5, atol=1e-5)
            print(f"    Ergebnis-Genauigkeit: {'✓' if accuracy else '✗'}")
    
    # Matrix-Inversion
    print("\n2. Matrix-Inversions-Tests")
    for size in sizes:
        print(f"  Teste Größe {size}x{size}...")
        a = np.random.rand(size, size)
        
        # MLX Performance
        start = time.time()
        for _ in range(3):  # 3 Wiederholungen für stabilere Messungen
            core_mlx.matrix_inverse(a)
        mlx_time = (time.time() - start) / 3
        results['inverse']['mlx'][size] = mlx_time
        
        # NumPy Performance
        start = time.time()
        for _ in range(3):
            core_np.matrix_inverse(a)
        np_time = (time.time() - start) / 3
        results['inverse']['numpy'][size] = np_time
        
        # Speedup berechnen
        speedup = np_time / mlx_time if mlx_time < np_time else -mlx_time / np_time
        print(f"    MLX: {mlx_time:.6f}s, NumPy: {np_time:.6f}s - Speedup: {speedup:.2f}x")
        
        # Genauigkeit überprüfen
        np_result = np.linalg.inv(a)
        mlx_result = core_mlx.matrix_inverse(a)
        if isinstance(mlx_result, np.ndarray):
            # Für Inversionen verwenden wir eine großzügigere Toleranz
            accuracy = np.allclose(mlx_result, np_result, rtol=1e-3, atol=1e-3)
            print(f"    Ergebnis-Genauigkeit: {'✓' if accuracy else '✗'}")
    
    # Batch-Matrix-Multiplikation
    print("\n3. Batch-Matrix-Multiplikations-Test")
    batch_size = 10
    matrix_size = 50
    print(f"  Teste Batch-Größe {batch_size} mit {matrix_size}x{matrix_size} Matrizen...")
    
    batch_a = [np.random.rand(matrix_size, matrix_size) for _ in range(batch_size)]
    batch_b = [np.random.rand(matrix_size, matrix_size) for _ in range(batch_size)]
    
    # MLX Batch Performance
    start = time.time()
    mlx_results = core_mlx.batch_matrix_multiply(batch_a, batch_b)
    mlx_batch_time = time.time() - start
    
    # Sequentielle Performance als Vergleich
    start = time.time()
    seq_results = [np.matmul(a, b) for a, b in zip(batch_a, batch_b)]
    seq_time = time.time() - start
    
    batch_speedup = seq_time / mlx_batch_time if mlx_batch_time < seq_time else -mlx_batch_time / seq_time
    print(f"    Batch-MLX: {mlx_batch_time:.6f}s, Sequentiell: {seq_time:.6f}s - Speedup: {batch_speedup:.2f}x")
    
    # Exportiere Ergebnisse als JSON
    metrics_file = "matrix_performance_benchmark.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'matrix_multiplication': {
                'sizes': sizes,
                'mlx_times': [results['matmul']['mlx'].get(s, 0) for s in sizes],
                'numpy_times': [results['matmul']['numpy'].get(s, 0) for s in sizes]
            },
            'matrix_inversion': {
                'sizes': sizes,
                'mlx_times': [results['inverse']['mlx'].get(s, 0) for s in sizes],
                'numpy_times': [results['inverse']['numpy'].get(s, 0) for s in sizes]
            },
            'batch_multiplication': {
                'batch_size': batch_size,
                'matrix_size': matrix_size,
                'mlx_time': mlx_batch_time,
                'sequential_time': seq_time,
                'speedup': batch_speedup
            }
        }, f, indent=2)
    
    print(f"\nPerformance-Metriken in {metrics_file} gespeichert.")
    
    # Performance-Metriken der MatrixCore exportieren
    core_metrics_file = "/Volumes/My Book/MISO_Ultimate 15.32.28/vxor.ai/VX-MATRIX/core/matrix_core_metrics.json"
    core_mlx.export_performance_metrics(core_metrics_file)
    print(f"MatrixCore-Metriken in {core_metrics_file} gespeichert.")
    
    # Plot der Ergebnisse
    try:
        plt.figure(figsize=(12, 10))
        
        # Matrix-Multiplikation
        plt.subplot(2, 1, 1)
        plt.title('Matrix-Multiplikation: MLX vs. NumPy')
        plt.plot(sizes, [results['matmul']['mlx'].get(s, 0) for s in sizes], 'bo-', label='MLX')
        plt.plot(sizes, [results['matmul']['numpy'].get(s, 0) for s in sizes], 'ro-', label='NumPy')
        plt.xlabel('Matrix-Größe')
        plt.ylabel('Zeit (s)')
        plt.legend()
        plt.grid(True)
        
        # Matrix-Inversion
        plt.subplot(2, 1, 2)
        plt.title('Matrix-Inversion: MLX vs. NumPy')
        plt.plot(sizes, [results['inverse']['mlx'].get(s, 0) for s in sizes], 'bo-', label='MLX')
        plt.plot(sizes, [results['inverse']['numpy'].get(s, 0) for s in sizes], 'ro-', label='NumPy')
        plt.xlabel('Matrix-Größe')
        plt.ylabel('Zeit (s)')
        plt.legend()
        plt.grid(True)
        
        # Gesamtlayout verbessern
        plt.tight_layout()
        
        # Speichern
        plot_file = "matrix_performance_plot.png"
        plt.savefig(plot_file)
        print(f"Performance-Plot in {plot_file} gespeichert.")
    except Exception as e:
        print(f"Konnte Plot nicht erstellen: {e}")
    
    print("\nBenchmark erfolgreich abgeschlossen!")
    print("\nZusammenfassung der optimierten MatrixCore-Implementierung:")
    print("- Numerische Stabilität: Equilibration, Guard-Clipping, adaptives Epsilon")
    print("- Hybrides Backend: NumPy für kleine/mittlere Matrizen, MLX für große")
    print("- JIT-Kompilierung: Für MatMul, Inverse, Batch und SVD")
    print("- Performance-Profiling: Op-Counter, Timing-Statistiken")
    print("- Adaptive Schwellenwerte: Basierend auf runtime Metriken")
    print("- Vektorisierte Batch-Verarbeitung: Mit Parallelisierung")

if __name__ == "__main__":
    run_comprehensive_test()
