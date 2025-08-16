#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Matrix Batch-Multiplikation Performance-Profiler
------------------------------------------------
Detailliertes Profiling für die batch_matrix_multiply-Methode mit Fokus 
auf mittlere Matrizen, um Performance-Bottlenecks zu identifizieren.
"""

import numpy as np
import time
import sys
import os
import logging
from contextlib import contextmanager

# Pfad hinzufügen und direkten Import verwenden
sys.path.append('/Volumes/My Book/MISO_Ultimate 15.32.28/vxor.ai/VX-MATRIX/core')

# Import der MatrixCore-Klasse
from matrix_core import MatrixCore

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO, 
                    format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

@contextmanager
def timer(name):
    """Zeitmessung für Code-Blöcke"""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{name}: {(end - start) * 1000:.6f} ms")

def generate_test_matrices(batch_size, matrix_size, dtype=np.float64):
    """Generiert numerisch stabile Test-Matrizen für die Benchmark-Tests"""
    matrices_a = []
    matrices_b = []
    
    for _ in range(batch_size):
        # Numerisch stabilere Matrizen durch Vermeidung von Nullen und sehr kleinen Werten
        a = np.random.rand(matrix_size, matrix_size).astype(dtype) + 0.1
        b = np.random.rand(matrix_size, matrix_size).astype(dtype) + 0.1
        
        # Sicherstellen, dass keine extrem großen Werte entstehen
        a = np.clip(a, 0.1, 10.0)
        b = np.clip(b, 0.1, 10.0)
        
        matrices_a.append(a)
        matrices_b.append(b)
    
    return matrices_a, matrices_b

def run_numpy_baseline(matrices_a, matrices_b, iterations=100):
    """Benchmark für direktes NumPy matmul als Baseline"""
    results = []
    total_time = 0
    
    # Warmup
    for _ in range(5):
        _ = [np.matmul(a, b) for a, b in zip(matrices_a, matrices_b)]
    
    # Zeitmessung
    for _ in range(iterations):
        start = time.perf_counter()
        results = [np.matmul(a, b) for a, b in zip(matrices_a, matrices_b)]
        end = time.perf_counter()
        total_time += (end - start)
    
    avg_time = total_time / iterations
    logger.info(f"NumPy Baseline: {avg_time * 1000:.6f} ms (durchschnittlich über {iterations} Iterationen)")
    return avg_time, results

def profile_matrix_core_batch_multiply(matrix_core, matrices_a, matrices_b, iterations=100):
    """Benchmark für MatrixCore.batch_matrix_multiply mit detailliertem Profiling"""
    total_time = 0
    results = []
    
    # Warmup
    for _ in range(5):
        _ = matrix_core.batch_matrix_multiply(matrices_a, matrices_b)
    
    # Zeitmessung
    for _ in range(iterations):
        start = time.perf_counter()
        results = matrix_core.batch_matrix_multiply(matrices_a, matrices_b)
        end = time.perf_counter()
        total_time += (end - start)
    
    avg_time = total_time / iterations
    logger.info(f"MatrixCore.batch_matrix_multiply: {avg_time * 1000:.6f} ms (durchschnittlich über {iterations} Iterationen)")
    return avg_time, results

def compare_performance(numpy_time, matrixcore_time):
    """Vergleicht die Performance zwischen NumPy und MatrixCore"""
    ratio = matrixcore_time / numpy_time
    logger.info(f"Performance-Verhältnis: MatrixCore ist {ratio:.2f}x langsamer als NumPy")
    logger.info(f"Ziel: <= 1.2x | Aktuell: {ratio:.2f}x | Differenz: {(ratio - 1.2) * 100:.2f}% über Toleranz" 
               if ratio > 1.2 else f"Ziel erreicht: {ratio:.2f}x ist innerhalb der 1.2x Toleranz")
    return ratio

def validate_results(numpy_results, matrixcore_results):
    """Validiert die Korrektheit der Ergebnisse zwischen NumPy und MatrixCore"""
    assert len(numpy_results) == len(matrixcore_results), "Unterschiedliche Anzahl an Ergebnissen"
    
    max_diff = 0
    for i, (numpy_res, matrix_res) in enumerate(zip(numpy_results, matrixcore_results)):
        if not np.allclose(numpy_res, matrix_res, rtol=1e-5, atol=1e-5):
            diff = np.abs(numpy_res - matrix_res).max()
            logger.warning(f"Numerische Abweichung in Matrix {i}: {diff}")
            max_diff = max(max_diff, diff)
    
    if max_diff > 0:
        logger.warning(f"Maximale numerische Abweichung: {max_diff}")
    else:
        logger.info("Alle Ergebnisse stimmen numerisch überein!")
    
    return max_diff

def profile_specific_components(matrices_a, matrices_b):
    """Detailliertes Profiling für einzelne Komponenten der Batch-Multiplikation"""
    # Konvertierungszeiten
    convert_time = 0
    for _ in range(100):
        start = time.perf_counter()
        np_matrices_a = [np.array(a) if not isinstance(a, np.ndarray) else a for a in matrices_a]
        np_matrices_b = [np.array(b) if not isinstance(b, np.ndarray) else b for b in matrices_b]
        end = time.perf_counter()
        convert_time += (end - start)
    
    logger.info(f"Zeit für Konvertierung: {convert_time / 100 * 1000:.6f} ms")
    
    # Matmul-Zeit
    np_matrices_a = [np.array(a) if not isinstance(a, np.ndarray) else a for a in matrices_a]
    np_matrices_b = [np.array(b) if not isinstance(b, np.ndarray) else b for b in matrices_b]
    
    matmul_time = 0
    for _ in range(100):
        start = time.perf_counter()
        results = [np.matmul(a, b) for a, b in zip(np_matrices_a, np_matrices_b)]
        end = time.perf_counter()
        matmul_time += (end - start)
    
    logger.info(f"Zeit für np.matmul-Operationen: {matmul_time / 100 * 1000:.6f} ms")
    
    # Gesamtzeit
    total_time = convert_time + matmul_time
    logger.info(f"Theoretische Gesamtzeit: {total_time / 100 * 1000:.6f} ms")

def main():
    """Hauptfunktion für das Performance-Profiling"""
    logger.info("=== Matrix Batch-Multiplikation Performance-Profiling ===")
    
    batch_sizes = [10, 20, 50]
    matrix_sizes = [10, 20, 30]  # 10 = kleine Matrizen, 20-30 = mittlere Matrizen
    
    matrix_core = MatrixCore()
    
    for batch_size in batch_sizes:
        for matrix_size in matrix_sizes:
            logger.info(f"\n=== Profiling für Batch-Größe: {batch_size}, Matrix-Größe: {matrix_size}x{matrix_size} ===")
            
            # Test-Matrizen generieren
            matrices_a, matrices_b = generate_test_matrices(batch_size, matrix_size)
            
            # NumPy Baseline
            with timer("NumPy Baseline - Total"):
                numpy_time, numpy_results = run_numpy_baseline(matrices_a, matrices_b)
            
            # MatrixCore Batch-Multiplikation
            with timer("MatrixCore - Total"):
                matrixcore_time, matrixcore_results = profile_matrix_core_batch_multiply(matrix_core, matrices_a, matrices_b)
            
            # Performance-Vergleich und Ergebnis-Validierung
            ratio = compare_performance(numpy_time, matrixcore_time)
            max_diff = validate_results(numpy_results, matrixcore_results)
            
            # Detailliertes Komponenten-Profiling
            logger.info("\n--- Komponenten-Profiling ---")
            profile_specific_components(matrices_a, matrices_b)
            
            logger.info("\n")
    
    logger.info("Performance-Profiling abgeschlossen")

if __name__ == "__main__":
    main()
