#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MATRIX Validierung der implementierten Optimierungen

ZTM-Level: STRICT
Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.

Dieser Test validiert die Kernoptimierungen in der VX-MATRIX Komponente:
1. SPD-Inversion per Cholesky-Zerlegung
2. Adaptive Tikhonov-Regularisierung
3. Konditionszahl-gesteuerte Backend-Auswahl
4. Hyper-optimierte Matrix-Multiplikation
"""

import os
import sys
import time
import warnings
import numpy as np
import scipy
import scipy.linalg
import pandas as pd
from sklearn.datasets import make_spd_matrix
import json

# Pfad zum Matrix-Core hinzufügen
matrix_core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vxor/ai/vx_matrix/core'))
sys.path.insert(0, matrix_core_path)
try:
    from matrix_core import MatrixCore
except ImportError:
    # Fallback für alternative Pfade
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vxor', 'ai', 'vx_matrix', 'core'))
    from matrix_core import MatrixCore


def create_original_core():
    """Erstellt eine Instanz des Original-MatrixCore ohne fortgeschrittene Optimierungen."""
    from functools import partial
    
    # Instanz erstellen
    core = MatrixCore(preferred_backend="auto")
    
    # Original matrix_inverse ohne die neuen Optimierungen
    def simple_matrix_inverse(self, matrix):
        """Vereinfachte Matrix-Inverse ohne Cholesky und Tikhonov."""
        self.op_counter['matrix_inverse'] += 1
        shape = getattr(matrix, 'shape', None)
        start = time.time()
        
        # Einfache Regularisierung
        eps = np.finfo(float).eps
        reg_matrix = matrix + np.eye(shape[0]) * eps
        
        try:
            result = np.linalg.inv(reg_matrix)
        except Exception:
            # Einfacher Fallback
            try:
                result = np.linalg.pinv(reg_matrix)
            except:
                result = matrix
        
        end = time.time()
        self.timing_stats['inverse'].append((shape, end-start, 'simple'))
        return result
    
    # Original matrix_multiply ohne BLAS/Strassen-Optimierungen
    def simple_matrix_multiply(self, a, b):
        """Vereinfachte Matrix-Multiplikation ohne Optimierungen."""
        self.op_counter['matrix_multiply'] += 1
        shape = (getattr(a, 'shape', (None,))[0], getattr(b, 'shape', (None,))[1])
        start = time.time()
        
        # Einfache NumPy-Multiplikation
        result = np.matmul(a, b)
        
        end = time.time()
        self.timing_stats['matrix_mult'].append((shape, end-start, 'numpy'))
        return result
    
    # Ersetze durch vereinfachte Versionen
    core.matrix_inverse = partial(simple_matrix_inverse, core)
    core.matrix_multiply = partial(simple_matrix_multiply, core)
    
    return core


def test_matrix_inverse_stability():
    """Testet die Stabilität der Matrix-Inversion für verschiedene Matrixtypen und Konditionszahlen."""
    # Original- und optimierter Core
    original_core = create_original_core()
    optimized_core = MatrixCore(preferred_backend="auto")
    
    # Test-Dimensionen und Konditionsbereiche
    test_dims = [10, 50, 100, 200, 500, 1000]
    condition_ranges = [
        ('gut', (1, 1e3)),
        ('mittel', (1e3, 1e6)),
        ('schlecht', (1e6, 1e10))
    ]
    
    results = []
    
    print("\n=== Matrix-Inversions-Stabilitätstest ===\n")
    
    for matrix_type in ["spd", "diag_dominant"]:
        print(f"\nTeste {matrix_type} Matrizen:")
        
        for condition_name, (cond_min, cond_max) in condition_ranges:
            target_condition = np.sqrt(cond_min * cond_max)
            print(f"  - {condition_name} konditioniert (κ ≈ {target_condition:.1e})")
            
            for size in test_dims:
                if condition_name == "schlecht" and size > 500:
                    continue
                
                # Matrix erzeugen
                if matrix_type == "spd":
                    # SPD-Matrix mit kontrollierter Konditionszahl
                    np.random.seed(42)
                    Q, _ = np.linalg.qr(np.random.randn(size, size))
                    max_eig = 1.0
                    min_eig = max_eig / target_condition
                    if target_condition > 1.0:
                        eig_log_space = np.geomspace(min_eig, max_eig, size)
                    else:
                        eig_log_space = np.ones(size) * max_eig
                    matrix = Q @ np.diag(eig_log_space) @ Q.T
                else:
                    # Diagonal-dominante Matrix
                    np.random.seed(42)
                    A = np.random.rand(size, size) * 2 - 1
                    row_sums = np.sum(np.abs(A), axis=1)
                    for i in range(size):
                        # Diagonale auf Basis der Konditionszahl anpassen
                        dominance = max(2.0, np.log10(target_condition))
                        A[i, i] = row_sums[i] * dominance
                    matrix = A
                
                # Tatsächliche Konditionszahl
                try:
                    cond = np.linalg.cond(matrix)
                except:
                    cond = float('inf')
                
                # Original-Inversion
                orig_start = time.time()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        orig_inv = original_core.matrix_inverse(matrix)
                    orig_success = True
                    orig_time = time.time() - orig_start
                    
                    # A*A^-1 = I Prüfung
                    orig_identity = np.matmul(matrix, orig_inv)
                    orig_identity_error = np.max(np.abs(orig_identity - np.eye(size)))
                except Exception as e:
                    orig_success = False
                    orig_time = float('nan')
                    orig_identity_error = float('inf')
                
                # Optimierte Inversion
                opt_start = time.time()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        opt_inv = optimized_core.matrix_inverse(matrix)
                    opt_success = True
                    opt_time = time.time() - opt_start
                    
                    # A*A^-1 = I Prüfung
                    opt_identity = np.matmul(matrix, opt_inv)
                    opt_identity_error = np.max(np.abs(opt_identity - np.eye(size)))
                except Exception as e:
                    opt_success = False
                    opt_time = float('nan')
                    opt_identity_error = float('inf')
                
                # Ergebnisse
                if orig_success and opt_success:
                    speed_ratio = orig_time / opt_time if opt_time > 0 else float('nan')
                    error_ratio = orig_identity_error / opt_identity_error if opt_identity_error > 0 else float('inf')
                    
                    result = {
                        "matrix_type": matrix_type,
                        "condition_range": condition_name,
                        "condition": cond,
                        "size": size,
                        "orig_time": orig_time,
                        "opt_time": opt_time,
                        "speedup": speed_ratio,
                        "orig_error": orig_identity_error,
                        "opt_error": opt_identity_error,
                        "error_improvement": error_ratio
                    }
                    
                    results.append(result)
                    
                    print(f"    {size}x{size}: κ={cond:.1e}, Speedup: {speed_ratio:.2f}x, "
                          f"Fehlerreduktion: {error_ratio:.1e}x")
                else:
                    status = "Nur Optimiert" if opt_success else "Nur Original" if orig_success else "Beide fehlgeschlagen"
                    print(f"    {size}x{size}: κ={cond:.1e}, Status: {status}")
    
    return results


def test_cholesky_advantage():
    """Testet den Vorteil der Cholesky-Zerlegung bei SPD-Matrizen."""
    optimized_core = MatrixCore(preferred_backend="auto")
    results = []
    
    print("\n=== Cholesky-Vorteil für SPD-Matrizen ===\n")
    
    for size in [50, 100, 200, 500]:
        print(f"\nTeste SPD-Matrizen der Größe {size}x{size}:")
        
        for cond_exp in [2, 4, 6, 8]:
            condition = 10**cond_exp
            
            # SPD-Matrix mit kontrollierter Konditionszahl
            np.random.seed(42)
            Q, _ = np.linalg.qr(np.random.randn(size, size))
            max_eig = 1.0
            min_eig = max_eig / condition
            if condition > 1.0:
                eig_log_space = np.geomspace(min_eig, max_eig, size)
            else:
                eig_log_space = np.ones(size) * max_eig
            matrix = Q @ np.diag(eig_log_space) @ Q.T
            
            # NumPy direkt
            start = time.time()
            try:
                np_inv = np.linalg.inv(matrix)
                np_time = time.time() - start
                np_success = True
            except:
                np_time = float('nan')
                np_success = False
            
            # Cholesky-Methode
            start = time.time()
            try:
                chol_inv = optimized_core._cholesky_inverse(matrix)
                chol_time = time.time() - start
                chol_success = True
            except:
                chol_time = float('nan')
                chol_success = False
            
            # Optimierter Core (sollte Cholesky für SPD wählen)
            start = time.time()
            try:
                opt_inv = optimized_core.matrix_inverse(matrix)
                opt_time = time.time() - start
                opt_success = True
            except:
                opt_time = float('nan')
                opt_success = False
            
            if np_success and chol_success and opt_success:
                np_error = np.max(np.abs(np.matmul(matrix, np_inv) - np.eye(size)))
                chol_error = np.max(np.abs(np.matmul(matrix, chol_inv) - np.eye(size)))
                opt_error = np.max(np.abs(np.matmul(matrix, opt_inv) - np.eye(size)))
                
                np_chol_speedup = np_time / chol_time if chol_time > 0 else float('nan')
                
                result = {
                    "size": size,
                    "condition": condition,
                    "numpy_time": np_time,
                    "cholesky_time": chol_time,
                    "optimized_time": opt_time,
                    "numpy_error": np_error,
                    "cholesky_error": chol_error,
                    "optimized_error": opt_error,
                    "speedup": np_chol_speedup
                }
                
                results.append(result)
                
                print(f"  κ={condition:.1e}: NumPy: {np_time:.6f}s (Err: {np_error:.1e}), "
                      f"Cholesky: {chol_time:.6f}s (Err: {chol_error:.1e}), "
                      f"Speedup: {np_chol_speedup:.2f}x")
            else:
                status = []
                if not np_success: status.append("NumPy fehlgeschlagen")
                if not chol_success: status.append("Cholesky fehlgeschlagen")
                if not opt_success: status.append("Optimiert fehlgeschlagen")
                print(f"  κ={condition:.1e}: {', '.join(status)}")
    
    return results


def test_blas_matmul_performance():
    """Testet die Performance-Verbesserung durch optimierte Matrixmultiplikation."""
    original_core = create_original_core()
    optimized_core = MatrixCore(preferred_backend="auto")
    results = []
    
    print("\n=== Matrix-Multiplikations-Performance ===\n")
    
    for size in [50, 100, 200, 500, 1000]:
        if size > 500 and not os.environ.get('TEST_LARGE_MATRICES'):
            continue
            
        print(f"\nTeste Matrix-Multiplikation für Größe {size}x{size}:")
        
        # Zufällige Matrizen
        np.random.seed(42)
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        # NumPy direkt
        start = time.time()
        np_result = np.matmul(A, B)
        np_time = time.time() - start
        
        # Original-Core
        start = time.time()
        orig_result = original_core.matrix_multiply(A, B)
        orig_time = time.time() - start
        
        # Optimierter Core
        start = time.time()
        opt_result = optimized_core.matrix_multiply(A, B)
        opt_time = time.time() - start
        
        # BLAS-Optimierung direkt
        start = time.time()
        blas_result = optimized_core._blas_optimized_matmul(A, B)
        blas_time = time.time() - start
        
        # Speedups
        numpy_opt_speedup = np_time / opt_time if opt_time > 0 else float('nan')
        orig_opt_speedup = orig_time / opt_time if opt_time > 0 else float('nan')
        blas_speedup = np_time / blas_time if blas_time > 0 else float('nan')
        
        # Abweichungen
        np_opt_diff = np.max(np.abs(np_result - opt_result))
        
        result = {
            "size": size,
            "numpy_time": np_time,
            "original_time": orig_time,
            "optimized_time": opt_time,
            "blas_time": blas_time,
            "numpy_opt_speedup": numpy_opt_speedup,
            "orig_opt_speedup": orig_opt_speedup,
            "blas_speedup": blas_speedup,
            "max_difference": np_opt_diff
        }
        
        results.append(result)
        
        print(f"  NumPy: {np_time:.6f}s, Original: {orig_time:.6f}s, Optimiert: {opt_time:.6f}s")
        print(f"  BLAS-Speedup: {blas_speedup:.2f}x, Opt-Speedup: {numpy_opt_speedup:.2f}x")
        print(f"  Max. Abweichung NumPy vs. Optimiert: {np_opt_diff:.2e}")
    
    return results


def run_validation_tests():
    """Führt alle Validierungstests durch und erstellt einen Bericht."""
    print("\n==================================================")
    print("    VX-MATRIX OPTIMIERUNGS-VALIDIERUNG")
    print("==================================================")
    
    # Test 1: Matrix-Inversions-Stabilität
    inverse_results = test_matrix_inverse_stability()
    
    # Test 2: Cholesky-Vorteil
    cholesky_results = test_cholesky_advantage()
    
    # Test 3: BLAS & Matrixmultiplikation
    matmul_results = test_blas_matmul_performance()
    
    # Zusammenfassung
    print("\n==================================================")
    print("    ZUSAMMENFASSUNG")
    print("==================================================")
    
    # Inverse Performance
    df_inv = pd.DataFrame(inverse_results)
    avg_speedup = df_inv['speedup'].mean()
    avg_error_reduction = df_inv['error_improvement'].mean()
    
    # Cholesky Advantage
    df_chol = pd.DataFrame(cholesky_results)
    avg_chol_speedup = df_chol['speedup'].mean()
    
    # Matrix-Multiplikation
    df_mm = pd.DataFrame(matmul_results)
    avg_mm_speedup = df_mm['numpy_opt_speedup'].mean()
    
    # Ausgabe
    print(f"\n1. Matrix-Inversion Optimierungen:")
    print(f"   - Durchschnittlicher Speedup: {avg_speedup:.2f}x")
    print(f"   - Durchschnittliche Fehlerreduktion: {avg_error_reduction:.2e}x")
    
    print(f"\n2. Cholesky-Optimierungen für SPD-Matrizen:")
    print(f"   - Durchschnittlicher Speedup vs. NumPy: {avg_chol_speedup:.2f}x")
    
    print(f"\n3. Matrix-Multiplikation:")
    print(f"   - Durchschnittlicher Speedup vs. NumPy: {avg_mm_speedup:.2f}x")
    
    # Ergebnisse als JSON speichern
    output_dir = os.path.join(os.path.dirname(__file__), 'validation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    result_data = {
        'timestamp': time.time(),
        'inverse_results': inverse_results,
        'cholesky_results': cholesky_results,
        'matmul_results': matmul_results,
        'summary': {
            'avg_inverse_speedup': float(avg_speedup),
            'avg_error_reduction': float(avg_error_reduction),
            'avg_cholesky_speedup': float(avg_chol_speedup),
            'avg_matmul_speedup': float(avg_mm_speedup)
        }
    }
    
    output_file = os.path.join(output_dir, 'validation_results.json')
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2, default=str)
    
    print(f"\nErgebnisse wurden gespeichert in: {os.path.abspath(output_file)}")
    
    return result_data


if __name__ == "__main__":
    run_validation_tests()
