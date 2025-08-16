#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für die optimierte Matrix-Multiplikation
"""

import sys
import os
import time
import numpy as np

# Direkter Import
sys.path.insert(0, '/Volumes/My Book/MISO_Ultimate 15.32.28/vxor.ai/VX-MATRIX')
try:
    from core.matrix_core import MatrixCore
except ImportError:
    # Direkter Import-Ansatz mit vollständiger Pfadangabe
    import importlib.util
    import os
    
    matrix_core_path = '/Volumes/My Book/MISO_Ultimate 15.32.28/vxor.ai/VX-MATRIX/core/matrix_core.py'
    spec = importlib.util.spec_from_file_location("matrix_core", matrix_core_path)
    matrix_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix_core)
    MatrixCore = matrix_core.MatrixCore

def test_matrix_operations():
    print("Initialisiere MatrixCore...")
    mc = MatrixCore(preferred_backend="numpy")
    
    # Spezialfall: 5 Matrizen mit (10x10) @ (10x15)
    print("\nTeste Spezialfall 5×(10×10 @ 10×15)...")
    matrices_a = [np.random.rand(10, 10) for _ in range(5)]
    matrices_b = [np.random.rand(10, 15) for _ in range(5)]
    
    # Standard-Multiplikation für Vergleich
    start_time = time.time()
    standard_results = [np.matmul(a, b) for a, b in zip(matrices_a, matrices_b)]
    standard_time = time.time() - start_time
    print(f"Standard-Methode: {standard_time*1000:.2f}ms")
    
    # Optimierte Batch-Multiplikation über prism_batch_operation
    start_time = time.time()
    optimized_results = mc.prism_batch_operation("multiply", matrices_a, matrices_b)
    optimized_time = time.time() - start_time
    print(f"Optimierte Methode: {optimized_time*1000:.2f}ms")
    
    # Ergebnisse überprüfen
    if optimized_results is not None:
        all_close = all(np.allclose(a, b) for a, b in zip(standard_results, optimized_results))
        print(f"Ergebnisse sind {'korrekt' if all_close else 'INKORREKT'}!")
        
        # Performance-Verbesserung
        speedup = standard_time / optimized_time if optimized_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("FEHLER: Optimierte Methode hat kein Ergebnis zurückgegeben!")
    
    print("\nTest abgeschlossen.")

if __name__ == "__main__":
    test_matrix_operations()
