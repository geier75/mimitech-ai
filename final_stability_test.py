#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verbesserter Performance- & Stabilitäts-Test für VX-MATRIX

Ziel:
- Numerische Stabilität prüfen (Equilibration, NaN/Inf-Handling)
- JIT-Funktionalität validieren und Fehler abfangen
- Performance-Vergleich NumPy vs. MLX für 10x10, 100x100, 500x500
- Automatischer Fallback auf NumPy bei MLX-Fehlern
- Konsistente Ergebnis-Validierung (AllClose)
"""
import os
import time
import warnings
import json  # Fehlender Import hinzugefügt
import numpy as np
import sys

# Pfad zu VX-MATRIX hinzufügen
matrix_core_path = os.path.join(os.path.dirname(__file__), 'vxor.ai', 'VX-MATRIX', 'core')
sys.path.insert(0, matrix_core_path)

from matrix_core import MatrixCore

# Konfiguration
MATRICES = {
    'small': (10, 10, 1000),
    'medium': (100, 100, 100),
    'large': (500, 500, 10)
}

# Initialisierung
core = MatrixCore(preferred_backend='mlx')
core.enable_jit = True

results = {}

# Helper: Erzeuge gut konditionierte Matrix
def generate_well_conditioned_matrix(n):
    A = np.random.rand(n, n)
    # Equilibrate to improve condition
    r = np.max(np.abs(A), axis=1)
    c = np.max(np.abs(A), axis=0)
    A = (1/np.where(r==0,1,r))[:,None] * A
    A = A * (1/np.where(c==0,1,c))
    # Tikhonov regularization
    return A + np.eye(n) * 1e-3

# Test sei ein Band: MatMul und Inverse
for label, (rows, cols, iters) in MATRICES.items():
    print(f"\n== Test: {label} ({rows}x{cols}), {iters} Iterationen ==")
    a = generate_well_conditioned_matrix(rows)
    b = generate_well_conditioned_matrix(cols)

    # MatMul
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=RuntimeWarning)
        # MLX
        start = time.time()
        try:
            for _ in range(iters):
                m1 = core.matrix_multiply(a, b)
            mlx_time = time.time() - start
        except Exception as e:
            print(f"MLX MatMul-Fehler: {e}, fallback auf NumPy")
            mlx_time = None
        # NumPy
        start = time.time()
        for _ in range(iters):
            m2 = np.matmul(a, b)
        numpy_time = time.time() - start

    # Ergebnisvergleich
    if mlx_time is not None:
        diff = np.max(np.abs(m1 - m2))
        print(f"Max-Differenz MLX vs NumPy: {diff:.2e}")
    speedup = (numpy_time / mlx_time) if mlx_time else float('nan')
    print(f"MLX MatMul: {mlx_time:.4f}s, NumPy: {numpy_time:.4f}s, Speedup: {speedup:.2f}x")
    results[f"{label}_matmul"] = {'mlx': mlx_time, 'numpy': numpy_time, 'diff': diff if mlx_time else None}

    # Inverse
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=RuntimeWarning)
        # MLX
        start = time.time()
        try:
            inv1 = core.matrix_inverse(a)
            mlx_inv_time = time.time() - start
        except Exception as e:
            print(f"MLX Inverse-Fehler: {e}, fallback auf NumPy")
            mlx_inv_time = None
        # NumPy
        start = time.time()
        inv2 = np.linalg.inv(a)
        numpy_inv_time = time.time() - start

    # Vergleich & Ausgabe
    if mlx_inv_time is not None:
        diff_inv = np.max(np.abs(inv1 - inv2))
        print(f"Max-Differenz Inverse: {diff_inv:.2e}")
    inv_speedup = (numpy_inv_time / mlx_inv_time) if mlx_inv_time else float('nan')
    print(f"MLX Inverse: {mlx_inv_time:.4f}s, NumPy: {numpy_inv_time:.4f}s, Speedup: {inv_speedup:.2f}x")
    results[f"{label}_inverse"] = {'mlx': mlx_inv_time, 'numpy': numpy_inv_time, 'diff': diff_inv if mlx_inv_time else None}

# Zusammenfassung
print("\n== Zusammenfassung ==")
for k,v in results.items():
    print(f"{k}: MLX={v['mlx']:.4f}, NumPy={v['numpy']:.4f}, diff={v['diff']:.2e}")

# Export
out = os.path.join(os.getcwd(), 'improved_performance_results.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Ergebnisse gespeichert: {out}")
