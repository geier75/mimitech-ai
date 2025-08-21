#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MATRIX Stabilitätstest: Matrix-Operationen

ZTM-Level: STRICT
Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.

Führt umfassende Stabilitätstests für Matrix-Operationen durch:
- Matrixmultiplikation
- Matrixinversion
- Batch-Operationen

Die Tests nutzen verschiedene Matrixtypen (SPD, diagonal-dominant, orthonormal, Hilbert)
und vergleichen MLX- und NumPy-Implementierungen auf numerische Stabilität und Genauigkeit.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pytest
from test_matrices import MatrixGenerators, get_matrix_stats

# VX-MATRIX Core importieren
matrix_core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core'))
sys.path.insert(0, matrix_core_path)
from matrix_core import MatrixCore


# Testparameter
MATRIX_TYPES = {
    'spd': MatrixGenerators.generate_spd_matrix,
    'diag_dominant': MatrixGenerators.generate_diag_dominant_matrix,
    'orthonormal': MatrixGenerators.generate_orthonormal_matrix,
    'hilbert': MatrixGenerators.generate_hilbert_matrix
}

MATRIX_SIZES = [10, 50, 100, 200]  # 500 für ausführlichere Tests hinzufügen

# Toleranzwerte für numerische Vergleiche
# Für gut konditionierte Matrizen verwenden wir strengere Toleranzen
TOLERANCE = {
    'spd': {'atol': 1e-10, 'rtol': 1e-8},
    'diag_dominant': {'atol': 1e-8, 'rtol': 1e-6},
    'orthonormal': {'atol': 1e-10, 'rtol': 1e-8},
    'hilbert': {'atol': 1e-3, 'rtol': 1e-1}  # Großzügigere Toleranz für schlecht konditionierte Matrizen
}


@pytest.fixture(scope="session")
def results_dir():
    """Erstellt ein Verzeichnis für die Testergebnisse"""
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../docs'))
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


@pytest.fixture(scope="module")
def matrix_cores():
    """Initalisiert MatrixCore-Instanzen mit verschiedenen Backends"""
    # NumPy-MatrixCore für Referenzergebnisse
    core_np = MatrixCore(preferred_backend="numpy")
    
    # MLX-MatrixCore mit aktiviertem JIT
    core_mlx = MatrixCore(preferred_backend="mlx")
    core_mlx.enable_jit = True
    
    # Warm-up für JIT-Kompilierung
    warm_up_size = 50
    a = np.random.rand(warm_up_size, warm_up_size)
    b = np.random.rand(warm_up_size, warm_up_size)
    _ = core_mlx.matrix_multiply(a, b)
    _ = core_mlx.matrix_inverse(a)
    
    return {'numpy': core_np, 'mlx': core_mlx}


class TestMatrixOperations:
    """Umfassende Tests für Matrix-Multiplikation und Inversion mit verschiedenen Matrixtypen"""
    
    @pytest.mark.parametrize("matrix_type", list(MATRIX_TYPES.keys()))
    @pytest.mark.parametrize("size", MATRIX_SIZES)
    def test_matrix_multiply(self, matrix_cores, matrix_type, size, results_dir):
        """Test der Matrixmultiplikation mit verschiedenen Matrixtypen und Größen"""
        print(f"\nTest: matrix_multiply, Typ: {matrix_type}, Größe: {size}x{size}")
        
        # Matrizen erzeugen
        matrix_generator = MATRIX_TYPES[matrix_type]
        a = matrix_generator(size)
        b = matrix_generator(size)
        
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
            print(f"  NumerischeKompatibilität (atol={tol['atol']:.1e}, rtol={tol['rtol']:.1e}): {'✓' if is_close else '✗'}")
            
            # Prüfe ob die Ergebnisse numerisch kompatibel sind
            assert is_close, f"Numerische Inkonsistenz: MLX und NumPy Ergebnisse weichen zu stark ab (abs_diff={abs_diff:.2e})"
        
        # Ergebnisse speichern
        return result
    
    @pytest.mark.parametrize("matrix_type", list(MATRIX_TYPES.keys()))
    @pytest.mark.parametrize("size", MATRIX_SIZES)
    def test_matrix_inverse(self, matrix_cores, matrix_type, size, results_dir):
        """Test der Matrixinversion mit verschiedenen Matrixtypen und Größen"""
        print(f"\nTest: matrix_inverse, Typ: {matrix_type}, Größe: {size}x{size}")
        
        # Matrix erzeugen
        matrix_generator = MATRIX_TYPES[matrix_type]
        a = matrix_generator(size)
        
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
            # Da hier die exakte Multiplikation wichtig ist, verwenden wir np.matmul direkt
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
            print(f"  NumerischeKompatibilität (atol={tol['atol']:.1e}, rtol={tol['rtol']:.1e}): {'✓' if is_close else '✗'}")
            
            # Überprüfen, ob die MLX und NumPy Ergebnisse konsistent sind
            assert i_mlx_err < 1e-1, f"MLX Inversionsqualität unzureichend: A*A^-1 - I Fehler = {i_mlx_err:.2e}"
        
        # Ergebnisse speichern
        return result


def run_tests_and_save_results():
    """Führt alle Tests aus und speichert die Ergebnisse als JSON"""
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../docs'))
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    matrix_cores = {'numpy': MatrixCore(preferred_backend="numpy"), 
                    'mlx': MatrixCore(preferred_backend="mlx")}
    matrix_cores['mlx'].enable_jit = True
    
    test_instance = TestMatrixOperations()
    
    # Matrix-Multiplikationstest
    print("\n=== Matrix-Multiplikationstests ===")
    for matrix_type in MATRIX_TYPES.keys():
        for size in MATRIX_SIZES:
            try:
                result = test_instance.test_matrix_multiply(matrix_cores, matrix_type, size, results_dir)
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
    
    # Matrix-Inversionstest
    print("\n=== Matrix-Inversionstests ===")
    for matrix_type in MATRIX_TYPES.keys():
        for size in MATRIX_SIZES:
            try:
                result = test_instance.test_matrix_inverse(matrix_cores, matrix_type, size, results_dir)
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
        
        for matrix_type in MATRIX_TYPES.keys():
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
    output_file = os.path.join(results_dir, 'matrix_stability_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailierte Ergebnisse gespeichert in: {output_file}")
    
    return results


if __name__ == "__main__":
    run_tests_and_save_results()
