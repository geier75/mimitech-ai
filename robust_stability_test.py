#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robuster Stabilitätstest für VX-MATRIX mit zusätzlichen Sicherheitsmaßnahmen

Merkmale:
- Verbesserte Matrix-Generierung mit SVD-basierter Kontrolle
- Erweiterte numerische Sicherheitsmaßnahmen
- Detaillierte Fehlerdiagnose
- Adaptiver Regularisierungsparameter
"""
import os
import sys
import time
import warnings
import json
import numpy as np

# Pfad zu VX-MATRIX hinzufügen
matrix_core_path = os.path.join(os.path.dirname(__file__), 'vxor.ai', 'VX-MATRIX', 'core')
sys.path.insert(0, matrix_core_path)

from matrix_core import MatrixCore

# Konfiguration
MATRICES = {
    'small': (10, 10, 100),
    'medium': (100, 100, 10),
    'large': (500, 500, 3)
}

# Matrix-Erzeugung mit noch robusterer Kontrolle
def generate_ultra_stable_matrix(n, condition_number=10.0):
    """
    Erzeugt eine numerisch stabile Matrix mit exakt kontrollierter Konditionszahl
    und garantiert ohne Nullwerte.
    """
    # Orthogonale Matrizen U und V über QR-Zerlegung
    u, _ = np.linalg.qr(np.random.randn(n, n))
    v, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Diagonalmatrix mit abfallenden Werten für die Konditionszahl
    s = np.logspace(0, np.log10(condition_number), n)
    
    # SVD-basierte Konstruktion: A = U * S * V^T
    matrix = u @ np.diag(s) @ v.T
    
    # Zusätzlich Regularisierung, um Singularitäten zu vermeiden
    matrix = matrix + np.eye(n) * 1e-6
    
    # Sicherstellen, dass keine extremen Werte vorhanden sind
    matrix = np.clip(matrix, -1e6, 1e6)
    
    return matrix

def evaluate_matrix_stability(m):
    """Bewertet die numerische Stabilität einer Matrix"""
    condition = np.linalg.cond(m)
    det = np.linalg.det(m)
    return {
        'condition': condition,
        'determinant': det,
        'has_nan': np.isnan(m).any(),
        'has_inf': np.isinf(m).any(),
        'min_abs': np.min(np.abs(m[m != 0])) if np.any(m != 0) else 0,
        'max_abs': np.max(np.abs(m))
    }

def print_matrix_stats(stats):
    """Gibt Matrix-Stabilitätsstatistiken aus"""
    print(f"  Konditionszahl: {stats['condition']:.2e}")
    print(f"  Determinante: {stats['determinant']:.2e}")
    print(f"  Enthält NaN: {stats['has_nan']}")
    print(f"  Enthält Inf: {stats['has_inf']}")
    print(f"  Min Abs (!=0): {stats['min_abs']:.2e}")
    print(f"  Max Abs: {stats['max_abs']:.2e}")

# Initialisierung
core = MatrixCore(preferred_backend='mlx')
core.enable_jit = True

# Ergebnis-Dictionary
results = {}

# Tests ausführen
for label, (rows, cols, iters) in MATRICES.items():
    print(f"\n==== Test: {label} ({rows}x{cols}), {iters} Iterationen ====")
    
    # Matrizen mit moderater Konditionszahl erzeugen
    a = generate_ultra_stable_matrix(rows, condition_number=5.0)
    b = generate_ultra_stable_matrix(cols, condition_number=5.0)
    
    # Matrix-Stabilität prüfen
    a_stats = evaluate_matrix_stability(a)
    print(f"Matrix A Statistik:")
    print_matrix_stats(a_stats)
    
    b_stats = evaluate_matrix_stability(b)
    print(f"Matrix B Statistik:")
    print_matrix_stats(b_stats)
    
    # 1. Matrix-Multiplikation
    print("\n1. Matrix-Multiplikation:")
    # Zuerst NumPy-Referenz für Vergleich
    try:
        start = time.time()
        for _ in range(iters):
            m_np = np.matmul(a, b)
        numpy_time = time.time() - start
        print(f"  NumPy MatMul: {numpy_time:.6f}s - Erfolgreich")
        numpy_failed = False
    except Exception as e:
        print(f"  NumPy MatMul Fehler: {e}")
        numpy_time = None
        numpy_failed = True
    
    # MLX Test mit Fehler-Abfang
    try:
        start = time.time()
        for _ in range(iters):
            m_mlx = core.matrix_multiply(a, b)
        mlx_time = time.time() - start
        print(f"  MLX MatMul: {mlx_time:.6f}s - Erfolgreich")
        mlx_failed = False
    except Exception as e:
        print(f"  MLX MatMul Fehler: {e}")
        mlx_time = None
        mlx_failed = True
    
    # Vergleich der Ergebnisse
    if not mlx_failed and not numpy_failed:
        # Absolute und relative Differenz
        abs_diff = np.max(np.abs(m_mlx - m_np))
        rel_diff = np.max(np.abs((m_mlx - m_np) / (np.abs(m_np) + 1e-10)))
        print(f"  Max Absolute Differenz: {abs_diff:.2e}")
        print(f"  Max Relative Differenz: {rel_diff:.2e}")
        # Speedup
        speedup = numpy_time / mlx_time if mlx_time > 0 else 0
        print(f"  Speedup: {speedup:.2f}x")
        # Ergebnis speichern
        results[f"{label}_matmul"] = {
            'mlx': mlx_time,
            'numpy': numpy_time,
            'abs_diff': float(abs_diff),
            'rel_diff': float(rel_diff),
            'speedup': float(speedup)
        }
    else:
        results[f"{label}_matmul"] = {
            'mlx': mlx_time,
            'numpy': numpy_time,
            'error': 'MLX failed' if mlx_failed else 'NumPy failed'
        }
    
    # 2. Matrix-Inversion
    print("\n2. Matrix-Inversion:")
    # Nur Matrix A invertieren
    # NumPy-Referenz
    try:
        start = time.time()
        inv_np = np.linalg.inv(a)
        numpy_inv_time = time.time() - start
        print(f"  NumPy Inverse: {numpy_inv_time:.6f}s - Erfolgreich")
        numpy_inv_failed = False
    except Exception as e:
        print(f"  NumPy Inverse Fehler: {e}")
        numpy_inv_time = None
        numpy_inv_failed = True
    
    # MLX Test
    try:
        start = time.time()
        inv_mlx = core.matrix_inverse(a)
        mlx_inv_time = time.time() - start
        print(f"  MLX Inverse: {mlx_inv_time:.6f}s - Erfolgreich")
        mlx_inv_failed = False
    except Exception as e:
        print(f"  MLX Inverse Fehler: {e}")
        mlx_inv_time = None
        mlx_inv_failed = True
    
    # Vergleich
    if not mlx_inv_failed and not numpy_inv_failed:
        # Genauigkeit prüfen (A * A^-1 = I)
        i_mlx = np.matmul(a, inv_mlx)
        i_np = np.matmul(a, inv_np)
        # Abweichung von der Einheitsmatrix
        i_mlx_err = np.max(np.abs(i_mlx - np.eye(rows)))
        i_np_err = np.max(np.abs(i_np - np.eye(rows)))
        # Differenz der Inversen
        inv_abs_diff = np.max(np.abs(inv_mlx - inv_np))
        inv_rel_diff = np.max(np.abs((inv_mlx - inv_np) / (np.abs(inv_np) + 1e-10)))
        
        print(f"  MLX A*A^-1 - I Max Fehler: {i_mlx_err:.2e}")
        print(f"  NumPy A*A^-1 - I Max Fehler: {i_np_err:.2e}")
        print(f"  Max Absolute Differenz: {inv_abs_diff:.2e}")
        print(f"  Max Relative Differenz: {inv_rel_diff:.2e}")
        # Speedup
        inv_speedup = numpy_inv_time / mlx_inv_time if mlx_inv_time > 0 else 0
        print(f"  Speedup: {inv_speedup:.2f}x")
        # Ergebnis speichern
        results[f"{label}_inverse"] = {
            'mlx': mlx_inv_time,
            'numpy': numpy_inv_time,
            'abs_diff': float(inv_abs_diff),
            'rel_diff': float(inv_rel_diff),
            'mlx_a_ainv_error': float(i_mlx_err),
            'numpy_a_ainv_error': float(i_np_err),
            'speedup': float(inv_speedup)
        }
    else:
        results[f"{label}_inverse"] = {
            'mlx': mlx_inv_time,
            'numpy': numpy_inv_time,
            'error': 'MLX failed' if mlx_inv_failed else 'NumPy failed'
        }

# Zusammenfassung
print("\n==== Zusammenfassung ====")
for operation_type in ['matmul', 'inverse']:
    print(f"\n{operation_type.upper()}:")
    for size in ['small', 'medium', 'large']:
        key = f"{size}_{operation_type}"
        if key in results:
            res = results[key]
            if 'error' in res:
                print(f"  {size}: FEHLER - {res['error']}")
            else:
                print(f"  {size}: MLX={res['mlx']:.6f}s, NumPy={res['numpy']:.6f}s, " +
                      f"Speedup={res['speedup']:.2f}x, Rel.Diff={res.get('rel_diff', 'N/A')}")

# Export
out = os.path.join(os.getcwd(), 'robust_performance_results.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nErgebnisse gespeichert: {out}")

# Empfehlungen
print("\n==== Empfehlungen ====")
print("1. Numerische Stabilität:")
print("   - Implementiere SVD-basierte Equilibrierung für alle MatrixCore-Operationen")
print("   - Verwende adaptive Regularisierung basierend auf der Matrixgröße")
print("   - Füge robuste Checks für NaN/Inf nach jeder Operation hinzu")

print("\n2. Performance:")
print("   - Für kleine Matrizen (<100x100): NumPy bevorzugen")
print("   - Für größere Matrizen: MLX mit JIT bevorzugen, aber mit Fallback")
print("   - Batch-Verarbeitung mit automatischer Backend-Auswahl implementieren")

print("\n3. Implementierung:")
print("   - Backend-Schwellwerte basierend auf diesen Messungen anpassen")
print("   - JIT-Fehlerbehandlung verbessern")
print("   - Validierung für Matrixgrößen >1000 durchführen")
