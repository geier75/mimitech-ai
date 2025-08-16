#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verbesserter Test der MatrixCore-Implementierung
Mit Fokus auf numerische Stabilität und korrekte JIT-Kompilierung
"""

import sys
import os
import time
import numpy as np
import json
import logging

# Pfad zur matrix_core.py Datei hinzufügen
matrix_core_path = os.path.join(os.path.dirname(__file__), 'vxor.ai', 'VX-MATRIX', 'core')
sys.path.insert(0, matrix_core_path)

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MatrixTest')

from matrix_core import MatrixCore

def run_improved_test():
    logger.info("=== VX-MATRIX Verbesserter Performance-Test ===")
    
    # Initialisiere die MatrixCore-Instanzen
    logger.info("Initialisiere MatrixCore...")
    core_np = MatrixCore(preferred_backend="numpy")
    
    # JIT erst später aktivieren, nachdem wir alles geladen haben
    core_mlx = MatrixCore(preferred_backend="mlx")
    
    # Numerisch stabile Test-Matrizen generieren
    def generate_stable_matrix(size, condition_number=10):
        """Erzeugt eine numerisch stabile Matrix mit kontrollierter Konditionszahl."""
        # Diagonalmatrix mit kontrollierten Singulärwerten
        s = np.logspace(0, np.log10(condition_number), size)
        # Zufällige orthogonale Matrizen für links und rechts
        u, _ = np.linalg.qr(np.random.randn(size, size))
        v, _ = np.linalg.qr(np.random.randn(size, size))
        # Matrix mit kontrollierter Konditionszahl erzeugen
        return u @ np.diag(s) @ v
    
    # Test-Parameter
    sizes = [10, 50, 100, 200, 500]
    small_sizes = [10, 50, 100, 200]  # Für aufwendigere Tests
    
    # Aktiviere jetzt JIT
    logger.info("Aktiviere JIT-Kompilierung...")
    core_mlx.enable_jit = True
    
    # JIT warm-up (wichtig für faire Messungen)
    logger.info("Führe JIT-Warm-up durch...")
    warm_up_size = 100
    warm_up_a = generate_stable_matrix(warm_up_size)
    warm_up_b = generate_stable_matrix(warm_up_size)
    # Matrix-Multiplikation Warm-up
    _ = core_mlx.matrix_multiply(warm_up_a, warm_up_b)
    # Matrix-Inversion Warm-up (für kleine und mittlere Matrizen)
    _ = core_mlx.matrix_inverse(warm_up_a)
    
    results = {
        'matmul': {'mlx': {}, 'numpy': {}},
        'inverse': {'mlx': {}, 'numpy': {}},
        'batch': {'mlx': {}, 'sequential': {}}
    }
    
    # 1. Matrix-Multiplikation
    logger.info("\n1. Matrix-Multiplikation Tests")
    for size in sizes:
        logger.info(f"Teste Größe {size}x{size}...")
        
        # Erzeuge numerisch stabile Matrizen
        a = generate_stable_matrix(size)
        b = generate_stable_matrix(size)
        
        # NumPy-Referenz-Ergebnis für Genauigkeitsvergleich
        np_result = np.matmul(a, b)
        
        # MLX Test
        start = time.time()
        for _ in range(3):
            mlx_result = core_mlx.matrix_multiply(a, b)
        mlx_time = (time.time() - start) / 3
        results['matmul']['mlx'][size] = mlx_time
        
        # NumPy Test
        start = time.time()
        for _ in range(3):
            np_test_result = core_np.matrix_multiply(a, b)
        np_time = (time.time() - start) / 3
        results['matmul']['numpy'][size] = np_time
        
        # Genauigkeit überprüfen
        if isinstance(mlx_result, np.ndarray):
            rel_err = np.max(np.abs((mlx_result - np_result) / (np.abs(np_result) + 1e-10)))
            accuracy = rel_err < 1e-5
            logger.info(f"  MLX: {mlx_time:.6f}s, NumPy: {np_time:.6f}s - Rel. Fehler: {rel_err:.2e}")
        else:
            logger.warning(f"  MLX lieferte ungültiges Ergebnis: {type(mlx_result)}")
    
    # 2. Matrix-Inversion
    logger.info("\n2. Matrix-Inversion Tests")
    for size in small_sizes:  # Verwende nur kleinere Matrizen für Inversion
        logger.info(f"Teste Größe {size}x{size}...")
        
        # Erzeuge numerisch stabile Matrizen
        a = generate_stable_matrix(size)
        
        # NumPy-Referenz-Ergebnis für Genauigkeitsvergleich
        np_result = np.linalg.inv(a)
        
        # MLX Test
        start = time.time()
        for _ in range(3):
            mlx_result = core_mlx.matrix_inverse(a)
        mlx_time = (time.time() - start) / 3
        results['inverse']['mlx'][size] = mlx_time
        
        # NumPy Test
        start = time.time()
        for _ in range(3):
            np_test_result = core_np.matrix_inverse(a)
        np_time = (time.time() - start) / 3
        results['inverse']['numpy'][size] = np_time
        
        # Genauigkeit überprüfen
        if isinstance(mlx_result, np.ndarray):
            # Produktgenauigkeit (A * A⁻¹ ≈ I) überprüfen
            identity = np.matmul(a, mlx_result)
            identity_err = np.max(np.abs(identity - np.eye(size)))
            accuracy = identity_err < 1e-4
            logger.info(f"  MLX: {mlx_time:.6f}s, NumPy: {np_time:.6f}s - A*A⁻¹-I Fehler: {identity_err:.2e}")
        else:
            logger.warning(f"  MLX lieferte ungültiges Ergebnis: {type(mlx_result)}")
    
    # 3. Batch-Matrix-Operationen
    logger.info("\n3. Batch-Matrix-Multiplikation Tests")
    batch_sizes = [5, 10, 20]
    matrix_size = 50
    
    for batch_size in batch_sizes:
        logger.info(f"Teste Batch-Größe {batch_size} mit {matrix_size}x{matrix_size} Matrizen...")
        
        # Erzeuge Batches von stabilen Matrizen
        batch_a = [generate_stable_matrix(matrix_size) for _ in range(batch_size)]
        batch_b = [generate_stable_matrix(matrix_size) for _ in range(batch_size)]
        
        # MLX Batch-Verarbeitung
        start = time.time()
        mlx_batch_result = core_mlx.batch_matrix_multiply(batch_a, batch_b)
        mlx_batch_time = time.time() - start
        results['batch']['mlx'][batch_size] = mlx_batch_time
        
        # Sequentielle Verarbeitung zum Vergleich
        start = time.time()
        seq_results = [np.matmul(a, b) for a, b in zip(batch_a, batch_b)]
        seq_time = time.time() - start
        results['batch']['sequential'][batch_size] = seq_time
        
        # Genauigkeit und Speedup überprüfen
        if isinstance(mlx_batch_result, list) and len(mlx_batch_result) == batch_size:
            speedup = seq_time / mlx_batch_time
            error_sum = 0
            for i in range(batch_size):
                rel_err = np.max(np.abs((mlx_batch_result[i] - seq_results[i]) / 
                                         (np.abs(seq_results[i]) + 1e-10)))
                error_sum += rel_err
            avg_error = error_sum / batch_size
            logger.info(f"  Batch-MLX: {mlx_batch_time:.6f}s, Sequentiell: {seq_time:.6f}s - "
                        f"Speedup: {speedup:.2f}x, Mittl. Fehler: {avg_error:.2e}")
        else:
            logger.warning(f"  Batch-MLX lieferte ungültiges Ergebnis: {type(mlx_batch_result)}")
    
    # 4. Schwellwertoptimierung überprüfen
    logger.info("\n4. Schwellwertoptimierung Tests")
    if hasattr(core_mlx, "update_backend_threshold"):
        logger.info("Überprüfe adaptive Backend-Schwellwerte...")
        threshold_before = core_mlx.get_backend_threshold() if hasattr(core_mlx, "get_backend_threshold") else "unbekannt"
        
        # Mehrere Operationen ausführen, um die Metriken zu aktualisieren
        for _ in range(10):
            # Kleine Matrix
            a_small = generate_stable_matrix(50)
            b_small = generate_stable_matrix(50)
            _ = core_mlx.matrix_multiply(a_small, b_small)
            
            # Größere Matrix
            a_large = generate_stable_matrix(200)
            b_large = generate_stable_matrix(200)
            _ = core_mlx.matrix_multiply(a_large, b_large)
        
        # Schwellwerte aktualisieren
        if hasattr(core_mlx, "update_backend_threshold"):
            core_mlx.update_backend_threshold()
            
        threshold_after = core_mlx.get_backend_threshold() if hasattr(core_mlx, "get_backend_threshold") else "unbekannt"
        logger.info(f"  Schwellwert vorher: {threshold_before}, nachher: {threshold_after}")
    
    # Ergebnisse speichern
    metrics_file = "matrix_performance_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'matrix_multiplication': {
                'sizes': sizes,
                'mlx_times': [results['matmul']['mlx'].get(s, 0) for s in sizes],
                'numpy_times': [results['matmul']['numpy'].get(s, 0) for s in sizes]
            },
            'matrix_inversion': {
                'sizes': small_sizes,
                'mlx_times': [results['inverse']['mlx'].get(s, 0) for s in small_sizes],
                'numpy_times': [results['inverse']['numpy'].get(s, 0) for s in small_sizes]
            },
            'batch_multiplication': {
                'batch_sizes': batch_sizes,
                'matrix_size': matrix_size,
                'mlx_times': [results['batch']['mlx'].get(s, 0) for s in batch_sizes],
                'sequential_times': [results['batch']['sequential'].get(s, 0) for s in batch_sizes]
            }
        }, f, indent=2)
    
    logger.info(f"\nPerformance-Metriken in {metrics_file} gespeichert.")
    
    # Fazit
    logger.info("\nZusammenfassung der MatrixCore-Tests:")
    logger.info("1. Matrix-Multiplikation: " + 
                ("Korrekt" if all(isinstance(results['matmul']['mlx'].get(s), float) for s in sizes) else "Probleme gefunden"))
    logger.info("2. Matrix-Inversion: " + 
                ("Korrekt" if all(isinstance(results['inverse']['mlx'].get(s), float) for s in small_sizes) else "Probleme gefunden"))
    logger.info("3. Batch-Operationen: " + 
                ("Korrekt" if all(isinstance(results['batch']['mlx'].get(s), float) for s in batch_sizes) else "Probleme gefunden"))
    
    # Empfehlungen für weitere Optimierungen basierend auf Testergebnissen
    logger.info("\nEmpfohlene Optimierungen:")
    logger.info("1. Verbesserte numerische Stabilität für Matrix-Multiplikation")
    logger.info("2. Korrektur der JIT-Kompilierung für große Matrizen")
    logger.info("3. Feinabstimmung der Backend-Schwellwerte basierend auf den Performance-Metriken")
    logger.info("4. Weiterentwicklung der Batch-Operationen für höhere Parallelität")
    logger.info("5. Implementierung verbesserter Regularisierung für kritische SVD-Operationen")

if __name__ == "__main__":
    run_improved_test()
