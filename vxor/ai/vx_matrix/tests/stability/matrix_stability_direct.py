#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MATRIX Stabilitätstest: Matrix-Operationen (Direkter Test)

ZTM-Level: STRICT
Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.

Direkter Test der robusten Matrix-Operationen für die T-Mathematics Engine.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
from sklearn.datasets import make_spd_matrix
import scipy.linalg

# Pfad zum Matrix-Core hinzufügen
matrix_core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core'))
sys.path.insert(0, matrix_core_path)
from matrix_core import MatrixCore


# Konfiguration
MATRIX_DIMS = [10, 50, 100, 200]  # 500 für ausgedehnte Tests
TOLERANCE = {
    'spd': {'atol': 1e-10, 'rtol': 1e-8},
    'diag_dominant': {'atol': 1e-8, 'rtol': 1e-6},
    'orthonormal': {'atol': 1e-10, 'rtol': 1e-8},
    'hilbert': {'atol': 1e-3, 'rtol': 1e-1}  # Großzügigere Toleranz für schlecht konditionierte Matrizen
}


class MatrixGenerators:
    """Sammlung von Funktionen zur Erzeugung verschiedener Arten von Testmatrizen"""
    
    @staticmethod
    def generate_spd_matrix(n, random_state=42):
        """
        Erzeugt eine symmetrisch positiv-definite (SPD) Matrix
        
        SPD-Matrizen haben ausschließlich positive Eigenwerte und sind garantiert
        invertierbar mit guter Stabilität.
        """
        return make_spd_matrix(n_dim=n, random_state=random_state)
    
    @staticmethod
    def generate_diag_dominant_matrix(n, random_state=42):
        """
        Erzeugt eine diagonal-dominante Matrix
        
        Eine Matrix ist diagonal-dominant, wenn der Betrag jedes Diagonaleintrags
        größer oder gleich der Summe der Beträge aller anderen Einträge in der gleichen
        Zeile ist. Solche Matrizen sind garantiert invertierbar.
        """
        np.random.seed(random_state)
        A = np.random.rand(n, n) * 2 - 1  # Werte zwischen -1 und 1
        
        # Mache die Matrix streng diagonal-dominant
        row_sums = np.sum(np.abs(A), axis=1)
        for i in range(n):
            # Setze Diagonalelement auf Zeilensumme + kleinen Offset
            diagonal_value = row_sums[i] + 0.1
            A[i, i] = diagonal_value
            
        return A
    
    @staticmethod
    def generate_orthonormal_matrix(n, random_state=42):
        """
        Erzeugt eine orthonormale Matrix mittels QR-Zerlegung
        
        Orthonormale Matrizen haben eine Konditionszahl von exakt 1 und sind
        somit numerisch optimal stabil. Die Inverse einer orthonormalen Matrix
        ist einfach ihre Transponierte.
        """
        np.random.seed(random_state)
        A = np.random.randn(n, n)
        Q, _ = np.linalg.qr(A)  # Q ist orthonormal
        return Q
    
    @staticmethod
    def generate_hilbert_matrix(n):
        """
        Erzeugt eine Hilbert-Matrix - berüchtigt für ihre schlechte Konditionierung
        
        Hilbert-Matrizen haben die Einträge a_ij = 1/(i+j-1) und sind extrem
        schlecht konditioniert, was sie zu einem strengen Test für numerische
        Stabilitätsalgorithmen macht.
        """
        return scipy.linalg.hilbert(n)


def get_matrix_stats(matrix):
    """Berechnet wichtige statistische Eigenschaften einer Matrix"""
    cond = np.linalg.cond(matrix)
    det = np.linalg.det(matrix)
    has_nan = np.isnan(matrix).any()
    has_inf = np.isinf(matrix).any()
    min_abs = np.min(np.abs(matrix[matrix != 0])) if np.any(matrix != 0) else 0
    max_abs = np.max(np.abs(matrix))
    
    return {
        "condition": cond,
        "determinant": det,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min_abs": min_abs,
        "max_abs": max_abs
    }


def get_test_matrix(matrix_type, n):
    """Erzeugt eine Testmatrix des angegebenen Typs und der Größe n"""
    if matrix_type == "spd":
        return MatrixGenerators.generate_spd_matrix(n)
    elif matrix_type == "diag_dominant":
        return MatrixGenerators.generate_diag_dominant_matrix(n)
    elif matrix_type == "orthonormal":
        return MatrixGenerators.generate_orthonormal_matrix(n)
    elif matrix_type == "hilbert":
        return MatrixGenerators.generate_hilbert_matrix(n)
    else:
        raise ValueError(f"Unbekannter Matrix-Typ: {matrix_type}")


def test_matrix_multiply(matrix_cores, matrix_type, size):
    """Test der Matrixmultiplikation mit verschiedenen Matrixtypen und Größen"""
    print(f"\nTest: matrix_multiply, Typ: {matrix_type}, Größe: {size}x{size}")
    
    # Matrizen erzeugen
    a = get_test_matrix(matrix_type, size)
    b = get_test_matrix(matrix_type, size)
    
    # Matrix-Statistiken ausgeben
    a_stats = get_matrix_stats(a)
    print(f"  Matrix A: Kond={a_stats['condition']:.2e}, Det={a_stats['determinant']:.2e}")
    
    # NumPy-Referenz für Vergleich
    numpy_core = matrix_cores['numpy']
    mlx_core = matrix_cores['mlx']
    
    # NumPy-Multiplikation
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=RuntimeWarning)
        try:
            np_result = numpy_core.matrix_multiply(a, b)
            numpy_success = True
            numpy_time = time.time() - start
            print(f"  NumPy MatMul: {numpy_time:.6f}s - Erfolgreich")
        except Exception as e:
            numpy_success = False
            numpy_time = None
            print(f"  NumPy MatMul Fehler: {e}")
    
    # MLX-Multiplikation
    start = time.time()
    try:
        mlx_result = mlx_core.matrix_multiply(a, b)
        mlx_success = True
        mlx_time = time.time() - start
        print(f"  MLX MatMul: {mlx_time:.6f}s - Erfolgreich")
    except Exception as e:
        mlx_success = False
        mlx_time = None
        print(f"  MLX MatMul Fehler: {e}")
    
    # Ergebnis-Analyse
    result = {
        'operation': 'matrix_multiply',
        'matrix_type': matrix_type,
        'size': size,
        'mlx_time': mlx_time,
        'numpy_time': numpy_time,
        'mlx_success': mlx_success,
        'numpy_success': numpy_success,
    }
    
    if mlx_success and numpy_success:
        # Berechne absolute und relative Differenz
        abs_diff = np.max(np.abs(mlx_result - np_result))
        rel_diff = np.max(np.abs((mlx_result - np_result) / (np.abs(np_result) + 1e-10)))
        
        # Prüfe auf numerische Übereinstimmung mit angepasster Toleranz je nach Matrixtyp
        tol = TOLERANCE[matrix_type]
        is_close = np.allclose(mlx_result, np_result, atol=tol['atol'], rtol=tol['rtol'])
        
        result.update({
            'abs_diff': float(abs_diff),
            'rel_diff': float(rel_diff),
            'speedup': float(numpy_time / mlx_time) if mlx_time > 0 else 0,
            'is_close': is_close
        })
        
        print(f"  Max Abs Diff: {abs_diff:.2e}, Max Rel Diff: {rel_diff:.2e}")
        print(f"  Numerische Kompatibilität (atol={tol['atol']:.1e}, rtol={tol['rtol']:.1e}): {'✓' if is_close else '✗'}")
    
    return result


def test_matrix_inverse(matrix_cores, matrix_type, size):
    """Test der Matrixinversion mit verschiedenen Matrixtypen und Größen"""
    print(f"\nTest: matrix_inverse, Typ: {matrix_type}, Größe: {size}x{size}")
    
    # Matrix erzeugen
    a = get_test_matrix(matrix_type, size)
    
    # Matrix-Statistiken ausgeben
    a_stats = get_matrix_stats(a)
    print(f"  Matrix: Kond={a_stats['condition']:.2e}, Det={a_stats['determinant']:.2e}")
    
    # NumPy-Referenz für Vergleich
    numpy_core = matrix_cores['numpy']
    mlx_core = matrix_cores['mlx']
    
    # NumPy-Inversion
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=RuntimeWarning)
        try:
            np_inv = numpy_core.matrix_inverse(a)
            numpy_success = True
            numpy_time = time.time() - start
            print(f"  NumPy Inverse: {numpy_time:.6f}s - Erfolgreich")
        except Exception as e:
            numpy_success = False
            numpy_time = None
            print(f"  NumPy Inverse Fehler: {e}")
    
    # MLX-Inversion
    start = time.time()
    try:
        mlx_inv = mlx_core.matrix_inverse(a)
        mlx_success = True
        mlx_time = time.time() - start
        print(f"  MLX Inverse: {mlx_time:.6f}s - Erfolgreich")
    except Exception as e:
        mlx_success = False
        mlx_time = None
        print(f"  MLX Inverse Fehler: {e}")
    
    # Ergebnis-Analyse
    result = {
        'operation': 'matrix_inverse',
        'matrix_type': matrix_type,
        'size': size,
        'condition': float(a_stats['condition']),
        'mlx_time': mlx_time,
        'numpy_time': numpy_time,
        'mlx_success': mlx_success,
        'numpy_success': numpy_success,
    }
    
    if mlx_success and numpy_success:
        # Berechne absolute und relative Differenz
        abs_diff = np.max(np.abs(mlx_inv - np_inv))
        rel_diff = np.max(np.abs((mlx_inv - np_inv) / (np.abs(np_inv) + 1e-10)))
        
        # Berechne A*A^-1 für NumPy und MLX, um die tatsächliche Inverse-Qualität zu bewerten
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            i_mlx = np.matmul(a, mlx_inv)
            i_np = np.matmul(a, np_inv)
        
        # Abweichung zur Einheitsmatrix berechnen
        i_mlx_err = np.max(np.abs(i_mlx - np.eye(size)))
        i_np_err = np.max(np.abs(i_np - np.eye(size)))
        
        # Prüfe auf numerische Übereinstimmung mit angepasster Toleranz je nach Matrixtyp
        tol = TOLERANCE[matrix_type]
        is_close = np.allclose(mlx_inv, np_inv, atol=tol['atol'], rtol=tol['rtol'])
        
        result.update({
            'abs_diff': float(abs_diff),
            'rel_diff': float(rel_diff),
            'mlx_identity_error': float(i_mlx_err),
            'numpy_identity_error': float(i_np_err),
            'speedup': float(numpy_time / mlx_time) if mlx_time > 0 else 0,
            'is_close': is_close
        })
        
        print(f"  MLX A*A^-1 - I Max Fehler: {i_mlx_err:.2e}")
        print(f"  NumPy A*A^-1 - I Max Fehler: {i_np_err:.2e}")
        print(f"  Max Abs Diff: {abs_diff:.2e}, Max Rel Diff: {rel_diff:.2e}")
        print(f"  Numerische Kompatibilität (atol={tol['atol']:.1e}, rtol={tol['rtol']:.1e}): {'✓' if is_close else '✗'}")
    
    return result


def run_comprehensive_tests():
    """Führt umfassende Tests für verschiedene Matrixtypen und -größen durch"""
    # Ergebnisverzeichnis erstellen (falls es nicht existiert)
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../docs'))
    os.makedirs(docs_dir, exist_ok=True)
    
    # MatrixCore-Instanzen initialisieren
    print("Initialisiere MatrixCore...")
    matrix_cores = {
        'numpy': MatrixCore(preferred_backend="numpy"),
        'mlx': MatrixCore(preferred_backend="mlx")
    }
    matrix_cores['mlx'].enable_jit = True
    
    # JIT Warm-up
    print("JIT Warm-up...")
    warm_up_size = 50
    a = MatrixGenerators.generate_spd_matrix(warm_up_size)
    b = MatrixGenerators.generate_spd_matrix(warm_up_size)
    _ = matrix_cores['mlx'].matrix_multiply(a, b)
    _ = matrix_cores['mlx'].matrix_inverse(a)
    
    # Ergebnisse speichern
    results = []
    matrix_types = ["spd", "diag_dominant", "orthonormal", "hilbert"]
    
    # Matrix-Multiplikationstests
    print("\n=== Matrix-Multiplikationstests ===")
    for matrix_type in matrix_types:
        for size in MATRIX_DIMS:
            try:
                result = test_matrix_multiply(matrix_cores, matrix_type, size)
                results.append(result)
            except Exception as e:
                print(f"  Test fehlgeschlagen: {e}")
                results.append({
                    'operation': 'matrix_multiply',
                    'matrix_type': matrix_type,
                    'size': size,
                    'error': str(e),
                    'mlx_success': False,
                    'numpy_success': False
                })
    
    # Matrix-Inversionstests
    print("\n=== Matrix-Inversionstests ===")
    for matrix_type in matrix_types:
        for size in MATRIX_DIMS:
            try:
                result = test_matrix_inverse(matrix_cores, matrix_type, size)
                results.append(result)
            except Exception as e:
                print(f"  Test fehlgeschlagen: {e}")
                results.append({
                    'operation': 'matrix_inverse',
                    'matrix_type': matrix_type,
                    'size': size,
                    'error': str(e),
                    'mlx_success': False,
                    'numpy_success': False
                })
    
    # Zusammenfassung
    print("\n=== Zusammenfassung ===")
    for operation in ['matrix_multiply', 'matrix_inverse']:
        print(f"\n{operation.upper()}:")
        
        for matrix_type in matrix_types:
            type_results = [r for r in results if r['operation'] == operation and r['matrix_type'] == matrix_type]
            success_rate = sum(1 for r in type_results if r.get('mlx_success', False)) / len(type_results) if type_results else 0
            
            # Berechne durchschnittliche Geschwindigkeit und Genauigkeit
            speedups = [r.get('speedup', 0) for r in type_results if 'speedup' in r]
            avg_speedup = sum(speedups) / len(speedups) if speedups else 0
            
            # Berechne durchschnittliche Differenz
            if operation == 'matrix_inverse':
                avg_identity_error = sum(r.get('mlx_identity_error', 0) for r in type_results if 'mlx_identity_error' in r) / len([r for r in type_results if 'mlx_identity_error' in r]) if [r for r in type_results if 'mlx_identity_error' in r] else float('nan')
                print(f"  {matrix_type}: Erfolgsrate={success_rate*100:.0f}%, Speedup={avg_speedup:.2f}x, Avg Identity Error={avg_identity_error:.2e}")
            else:
                avg_rel_diff = sum(r.get('rel_diff', 0) for r in type_results if 'rel_diff' in r) / len([r for r in type_results if 'rel_diff' in r]) if [r for r in type_results if 'rel_diff' in r] else float('nan')
                print(f"  {matrix_type}: Erfolgsrate={success_rate*100:.0f}%, Speedup={avg_speedup:.2f}x, Avg Rel Diff={avg_rel_diff:.2e}")
    
    # Ergebnisse als JSON speichern
    output_file = os.path.join(docs_dir, 'matrix_stability_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'results': results,
            'summary': {
                'matrix_types': matrix_types,
                'matrix_dims': MATRIX_DIMS,
                'total_tests': len(results),
                'successful_tests': sum(1 for r in results if r.get('mlx_success', False) and r.get('numpy_success', False))
            }
        }, f, indent=2)
    
    print(f"\nDetailierte Ergebnisse gespeichert in: {output_file}")
    
    return results


if __name__ == "__main__":
    run_comprehensive_tests()
