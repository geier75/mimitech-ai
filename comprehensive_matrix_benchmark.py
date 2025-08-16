#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Umfassender Benchmark für die VX-MATRIX Optimierungen

ZTM-Level: STRICT
Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.

Dieser Test demonstriert die Verbesserungen in der T-Mathematics-Engine,
insbesondere:
1. SPD-Inversion per Cholesky-Zerlegung
2. Adaptive Tikhonov-Regularisierung
3. Konditionszahl-gesteuerte Backend-Auswahl
4. Hyper-optimierte Matrix-Multiplikation

Der Test umfasst:
- Performance-Messungen
- Numerische Stabilitätstests
- Edge-Case-Prüfung
- Backend-Strategien
"""

import os
import sys
import time
import warnings
import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from sklearn.datasets import make_spd_matrix
import json
from functools import partial

# Pfad zum Matrix-Core hinzufügen
matrix_core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vxor.ai/VX-MATRIX/core'))
sys.path.insert(0, matrix_core_path)
from matrix_core import MatrixCore

# Konfiguration
TEST_DIMS = [10, 50, 100, 200, 500, 1000]
CONDITION_RANGES = [
    ('gut', (1, 1e3)),              # Gut konditioniert
    ('mittel', (1e3, 1e6)),         # Mittel konditioniert
    ('schlecht', (1e6, 1e15))       # Schlecht konditioniert
]
OUTPUT_DIR = 'vxor.ai/docs/benchmarks'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class MatrixGenerators:
    """Sammlung von optimierten Funktionen zur Erzeugung von Testmatrizen"""
    
    @staticmethod
    def generate_spd_matrix(n, condition=None, random_state=42):
        """
        Erzeugt eine symmetrisch positiv-definite (SPD) Matrix mit kontrollierter Konditionszahl
        
        Args:
            n: Matrixgröße
            condition: Ziel-Konditionszahl (None für standardmäßige SPD-Matrix)
            random_state: Seed für Reproduzierbarkeit
            
        Returns:
            SPD-Matrix mit der angegebenen Konditionszahl
        """
        if condition is None:
            return make_spd_matrix(n_dim=n, random_state=random_state)
        
        # Für kontrollierte Konditionszahl verwenden wir SVD-basierte Konstruktion
        np.random.seed(random_state)
        
        # Erzeuge Orthogonalmatrix via QR-Zerlegung
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        
        # Erzeuge Eigenwerte mit gewünschter Konditionszahl
        max_eig = 1.0
        min_eig = max_eig / condition
        
        # Logarithmisch verteilte Eigenwerte zwischen min_eig und max_eig
        if condition > 1.0:
            eig_log_space = np.geomspace(min_eig, max_eig, n)
        else:
            eig_log_space = np.ones(n) * max_eig
            
        # Konstruiere SPD-Matrix: Q * diag(Eigenwerte) * Q^T
        return Q @ np.diag(eig_log_space) @ Q.T
    
    @staticmethod
    def generate_diag_dominant_matrix(n, dominance_factor=2.0, random_state=42):
        """
        Erzeugt eine diagonal-dominante Matrix mit kontrollierbarem Dominanzfaktor
        
        Args:
            n: Matrixgröße
            dominance_factor: Wie stark die Diagonale dominiert (>1.0)
            random_state: Seed für Reproduzierbarkeit
            
        Returns:
            Diagonal-dominante Matrix
        """
        np.random.seed(random_state)
        A = np.random.rand(n, n) * 2 - 1  # Werte zwischen -1 und 1
        
        # Mache die Matrix streng diagonal-dominant
        row_sums = np.sum(np.abs(A), axis=1)
        for i in range(n):
            # Setze Diagonalelement auf Zeilensumme * Dominanzfaktor
            A[i, i] = row_sums[i] * dominance_factor
            
        return A
    
    @staticmethod
    def generate_orthonormal_matrix(n, random_state=42):
        """
        Erzeugt eine orthonormale Matrix mittels QR-Zerlegung
        
        Args:
            n: Matrixgröße
            random_state: Seed für Reproduzierbarkeit
            
        Returns:
            Orthonormale Matrix
        """
        np.random.seed(random_state)
        A = np.random.randn(n, n)
        Q, _ = np.linalg.qr(A)  # Q ist orthonormal
        return Q
    
    @staticmethod
    def generate_controlled_condition_matrix(n, condition, random_state=42):
        """
        Erzeugt eine Matrix mit exakter Konditionszahl mittels SVD-basierter Konstruktion
        
        Args:
            n: Matrixgröße
            condition: Exakte Konditionszahl
            random_state: Seed für Reproduzierbarkeit
            
        Returns:
            Matrix mit exakter Konditionszahl
        """
        np.random.seed(random_state)
        
        # Erzeuge zwei Orthogonalmatrizen
        U, _ = np.linalg.qr(np.random.randn(n, n))
        V, _ = np.linalg.qr(np.random.randn(n, n))
        
        # Erzeuge Singulärwerte mit gewünschter Konditionszahl
        max_sv = 1.0
        min_sv = max_sv / condition
        
        # Logarithmisch verteilte Singulärwerte
        if n > 2 and condition > 1.0:
            middle_values = np.geomspace(min_sv, max_sv, n-2)
            s = np.concatenate(([max_sv], middle_values, [min_sv]))
        else:
            s = np.array([max_sv, min_sv])
            if n > 2:
                s = np.pad(s, (0, n-2), 'constant', constant_values=min_sv)
        
        # Konstruiere Matrix: U * diag(s) * V^T
        return U @ np.diag(s) @ V.T
    
    @staticmethod
    def generate_hilbert_matrix(n):
        """
        Erzeugt eine Hilbert-Matrix - berüchtigt für ihre schlechte Konditionierung
        
        Hilbert-Matrizen haben die Einträge a_ij = 1/(i+j-1) und sind extrem
        schlecht konditioniert, mit exponentiell wachsender Konditionszahl.
        
        Args:
            n: Matrixgröße
            
        Returns:
            Hilbert-Matrix
        """
        return scipy.linalg.hilbert(n)


def get_matrix_info(matrix):
    """Berechnet ausführliche Matrix-Informationen"""
    try:
        cond = np.linalg.cond(matrix)
    except:
        cond = float('inf')
        
    try:
        det = np.linalg.det(matrix)
    except:
        det = float('nan')
        
    has_nan = np.isnan(matrix).any()
    has_inf = np.isinf(matrix).any()
    
    if np.any(matrix != 0):
        min_abs = np.min(np.abs(matrix[matrix != 0]))
        max_abs = np.max(np.abs(matrix))
    else:
        min_abs = 0
        max_abs = 0
    
    symmetric = np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-8)
    
    # Überprüfe auf SPD (nur wenn symmetrisch)
    is_spd = False
    if symmetric:
        try:
            eigvals = np.linalg.eigvalsh(matrix)
            is_spd = np.all(eigvals > 1e-10)
        except:
            is_spd = False
    
    return {
        "condition": cond,
        "determinant": det,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min_abs": min_abs,
        "max_abs": max_abs,
        "symmetric": symmetric,
        "is_spd": is_spd
    }


def create_test_matrix(matrix_type, size, **kwargs):
    """Erzeugt eine Testmatrix des angegebenen Typs und der Größe"""
    if matrix_type == "spd":
        return MatrixGenerators.generate_spd_matrix(size, **kwargs)
    elif matrix_type == "diag_dominant":
        return MatrixGenerators.generate_diag_dominant_matrix(size, **kwargs)
    elif matrix_type == "orthonormal":
        return MatrixGenerators.generate_orthonormal_matrix(size, **kwargs)
    elif matrix_type == "controlled":
        condition = kwargs.get('condition', 100)
        return MatrixGenerators.generate_controlled_condition_matrix(size, condition, **kwargs.get('random_state', 42))
    elif matrix_type == "hilbert":
        return MatrixGenerators.generate_hilbert_matrix(size)
    else:
        raise ValueError(f"Unbekannter Matrix-Typ: {matrix_type}")


def create_original_core():
    """Erstellt eine Instanz des Original-MatrixCore"""
    # In einem echten Szenario würden wir eine ältere Version laden
    # Für den Test nutzen wir eine angepasste Version ohne die neuesten Optimierungen
    core = MatrixCore(preferred_backend="auto")
    
    # Deaktiviere fortgeschrittene Funktionen durch Monkey-Patching
    orig_matrix_inverse = core.matrix_inverse
    
    def simple_matrix_inverse(self, matrix):
        """Vereinfachte Matrix-Inverse ohne fortgeschrittene Optimierungen"""
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
                result = matrix  # Ultimate Fallback
        
        end = time.time()
        self.timing_stats['inverse'].append((shape, end-start, 'simple'))
        return result
    
    # Ersetze durch vereinfachte Version
    core.matrix_inverse = partial(simple_matrix_inverse, core)
    
    # Deaktiviere fortgeschrittene Matrixmultiplikation
    orig_matrix_multiply = core.matrix_multiply
    
    def simple_matrix_multiply(self, a, b):
        """Vereinfachte Matrix-Multiplikation ohne Optimierungen"""
        self.op_counter['matrix_multiply'] += 1
        shape = (getattr(a, 'shape', (None,))[0], getattr(b, 'shape', (None,))[1])
        start = time.time()
        
        # Einfache NumPy-Multiplikation
        result = np.matmul(a, b)
        
        end = time.time()
        self.timing_stats['matrix_mult'].append((shape, end-start, 'numpy'))
        return result
    
    # Ersetze durch vereinfachte Version
    core.matrix_multiply = partial(simple_matrix_multiply, core)
    
    return core


def test_matrix_inverse_stability():
    """
    Testet die numerische Stabilität der Matrix-Inversion für verschiedene Matrixtypen
    und Konditionszahlen, mit Schwerpunkt auf den implementierten Optimierungen.
    """
    # Original- und optimierter Core
    original_core = create_original_core()
    optimized_core = MatrixCore(preferred_backend="auto")
    
    results = []
    
    # Teste verschiedene Matrixgrößen und -typen
    matrix_types = [
        ("spd", "Symmetrisch Positiv-Definit"),
        ("diag_dominant", "Diagonal-Dominant"),
        ("controlled", "Kontrollierte Konditionszahl")
    ]
    
    print("\n===== Matrix-Inversions-Stabilitätstest =====\n")
    
    for matrix_type, type_name in matrix_types:
        print(f"\nTesting {type_name} Matrizen:")
        
        for condition_name, (cond_min, cond_max) in CONDITION_RANGES:
            # Verwende mittlere Konditionszahl im log-Bereich
            target_condition = np.sqrt(cond_min * cond_max)
            
            # Mitte des logarithmischen Bereichs
            print(f"  - {condition_name} konditioniert (κ ≈ {target_condition:.1e})")
            
            for size in TEST_DIMS:
                # Überspringe zu große Matrizen bei schlechter Konditionierung
                if condition_name == "schlecht" and size > 500:
                    continue
                    
                # Erzeuge Testmatrix mit Ziel-Konditionszahl
                if matrix_type == "controlled":
                    matrix = create_test_matrix(matrix_type, size, condition=target_condition)
                elif matrix_type == "spd":
                    matrix = create_test_matrix(matrix_type, size, condition=target_condition)
                else:
                    matrix = create_test_matrix(matrix_type, size)
                
                # Berechne tatsächliche Matrix-Eigenschaften
                matrix_info = get_matrix_info(matrix)
                
                # Original-Inversion testen
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
                
                # Optimierte Inversion testen
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
                
                # Ergebnisse speichern
                speed_ratio = orig_time / opt_time if (opt_time > 0 and not np.isnan(orig_time)) else float('nan')
                error_ratio = orig_identity_error / opt_identity_error if opt_identity_error > 0 else float('inf')
                
                result = {
                    "matrix_type": matrix_type,
                    "condition_range": condition_name,
                    "target_condition": target_condition,
                    "actual_condition": matrix_info["condition"],
                    "is_spd": matrix_info["is_spd"],
                    "size": size,
                    "orig_success": orig_success,
                    "opt_success": opt_success,
                    "orig_time": orig_time,
                    "opt_time": opt_time,
                    "speed_ratio": speed_ratio,
                    "orig_identity_error": orig_identity_error,
                    "opt_identity_error": opt_identity_error,
                    "error_ratio": error_ratio
                }
                
                results.append(result)
                
                # Gib Ergebnis aus für diese Matrix
                if orig_success and opt_success:
                    print(f"    Size {size}x{size}: "
                          f"κ={matrix_info['condition']:.1e}, "
                          f"Speedup: {speed_ratio:.2f}x, "
                          f"Fehlerreduktion: {error_ratio:.1e}x")
                else:
                    status = "Nur Optimiert" if opt_success else "Nur Original" if orig_success else "Beide fehlgeschlagen"
                    print(f"    Size {size}x{size}: κ={matrix_info['condition']:.1e}, Status: {status}")
    
    return results


def test_cholesky_advantage():
    """
    Gezielter Test für den Vorteil der Cholesky-Zerlegung bei SPD-Matrizen
    gegenüber regulärer Inversion.
    """
    print("\n===== Cholesky-Optimierungs-Test für SPD-Matrizen =====\n")
    
    # Erzeuge speziell den optimierten Core
    optimized_core = MatrixCore(preferred_backend="auto")
    results = []
    
    for size in [50, 100, 200, 500]:
        print(f"\nTesting SPD-Matrizen der Größe {size}x{size}:")
        
        for cond_exp in [2, 4, 6, 8]:
            condition = 10**cond_exp
            matrix = create_test_matrix("spd", size, condition=condition)
            
            # Zeitmessung für direkte Inversion
            start = time.time()
            try:
                np_inv = np.linalg.inv(matrix)
                np_time = time.time() - start
                np_success = True
            except Exception:
                np_time = float('nan')
                np_success = False
            
            # Zeitmessung für Cholesky-Inversion
            start = time.time()
            try:
                chol_inv = optimized_core._cholesky_inverse(matrix)
                chol_time = time.time() - start
                chol_success = True
            except Exception:
                chol_time = float('nan')
                chol_success = False
            
            # Zeitmessung für optimierten Core (sollte Cholesky für SPD wählen)
            start = time.time()
            try:
                opt_inv = optimized_core.matrix_inverse(matrix)
                opt_time = time.time() - start
                opt_success = True
            except Exception:
                opt_time = float('nan')
                opt_success = False
            
            # Vergleich der A*A^-1 Fehler
            if np_success and chol_success and opt_success:
                np_error = np.max(np.abs(np.matmul(matrix, np_inv) - np.eye(size)))
                chol_error = np.max(np.abs(np.matmul(matrix, chol_inv) - np.eye(size)))
                opt_error = np.max(np.abs(np.matmul(matrix, opt_inv) - np.eye(size)))
                
                np_chol_speedup = np_time / chol_time if chol_time > 0 else float('nan')
                np_opt_speedup = np_time / opt_time if opt_time > 0 else float('nan')
                
                result = {
                    "size": size,
                    "condition": condition,
                    "numpy_time": np_time,
                    "cholesky_time": chol_time,
                    "optimized_time": opt_time,
                    "numpy_error": np_error,
                    "cholesky_error": chol_error,
                    "optimized_error": opt_error,
                    "np_chol_speedup": np_chol_speedup,
                    "np_opt_speedup": np_opt_speedup
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
    """
    Misst die Performance-Verbesserung durch BLAS-optimierte Matrixmultiplikation
    und den Strassen-Algorithmus.
    """
    print("\n===== BLAS & Strassen Multiplikations-Test =====\n")
    
    # Original- und optimierter Core
    original_core = create_original_core()
    optimized_core = MatrixCore(preferred_backend="auto")
    results = []
    
    # Wir testen für verschiedene Matrixgrößen
    for size in [50, 100, 200, 500, 1000, 2000]:  # Größere Matrizen für Strassen-Vorteile
        # Überspringe sehr große Matrizen wenn nicht benötigt
        if size > 1000 and not os.environ.get('TEST_LARGE_MATRICES'):
            continue
            
        print(f"\nTesting Matrix-Multiplikation für Größe {size}x{size}:")
        
        # Erzeuge zufällige Matrizen
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
        
        # Optimierter Core (BLAS + Strassen)
        start = time.time()
        opt_result = optimized_core.matrix_multiply(A, B)
        opt_time = time.time() - start
        
        # Direkte BLAS-Implementierung
        start = time.time()
        blas_result = optimized_core._blas_optimized_matmul(A, B)
        blas_time = time.time() - start
        
        # Bei großen Matrizen auch Strassen testen
        if size >= 1000:
            start = time.time()
            strassen_result = optimized_core._strassen_multiply(A, B, leaf_size=128)
            strassen_time = time.time() - start
        else:
            strassen_time = float('nan')
        
        # Berechne Speedups
        numpy_opt_speedup = np_time / opt_time if opt_time > 0 else float('nan')
        orig_opt_speedup = orig_time / opt_time if opt_time > 0 else float('nan')
        blas_speedup = np_time / blas_time if blas_time > 0 else float('nan')
        strassen_speedup = np_time / strassen_time if not np.isnan(strassen_time) and strassen_time > 0 else float('nan')
        
        # Berechne max. Abweichungen zwischen Implementierungen
        np_opt_diff = np.max(np.abs(np_result - opt_result))
        
        result = {
            "size": size,
            "numpy_time": np_time,
            "original_time": orig_time,
            "optimized_time": opt_time,
            "blas_time": blas_time,
            "strassen_time": strassen_time if size >= 1000 else None,
            "numpy_opt_speedup": numpy_opt_speedup,
            "orig_opt_speedup": orig_opt_speedup,
            "blas_speedup": blas_speedup,
            "strassen_speedup": strassen_speedup if size >= 1000 else None,
            "np_opt_diff": np_opt_diff
        }
        
        results.append(result)
        
        print(f"  NumPy: {np_time:.6f}s, Original: {orig_time:.6f}s, Optimiert: {opt_time:.6f}s")
        print(f"  BLAS-Speedup: {blas_speedup:.2f}x, Opt-Speedup: {numpy_opt_speedup:.2f}x")
        if not np.isnan(strassen_time):
            print(f"  Strassen-Speedup: {strassen_speedup:.2f}x")
        print(f"  Max. Abweichung NumPy vs. Optimiert: {np_opt_diff:.2e}")
    
    return results


def plot_results(inverse_results, cholesky_results, matmul_results):
    """
    Erzeugt detaillierte Plots für die Testergebnisse.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Plot 1: Speedup nach Matrixgröße und -typ
    plt.figure(figsize=(12, 8))
    df = pd.DataFrame(inverse_results)
    
    # Filtere nach erfolgreichen Tests
    df = df[df['orig_success'] & df['opt_success']]
    
    for matrix_type in df['matrix_type'].unique():
        df_type = df[df['matrix_type'] == matrix_type]
        
        # Gruppiere nach Größe und Kondition
        sizes = []
        speedups = []
        conditions = []
        
        for size in sorted(df_type['size'].unique()):
            df_size = df_type[df_type['size'] == size]
            sizes.append(size)
            speedups.append(df_size['speed_ratio'].mean())
            conditions.append(df_size['actual_condition'].mean())
        
        plt.plot(sizes, speedups, 'o-', label=f"{matrix_type} (κ≈{np.mean(conditions):.1e})")
    
    plt.xscale('log')
    plt.xlabel('Matrixgröße (n)') 
    plt.ylabel('Speedup (Original/Optimiert)')
    plt.title('Geschwindigkeitsverbesserung durch Optimierungen bei Matrix-Inversion')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'inverse_speedup.png'), dpi=300)
    
    # Plot 2: Fehlerreduktion durch Optimierungen
    plt.figure(figsize=(12, 8))
    
    for matrix_type in df['matrix_type'].unique():
        df_type = df[df['matrix_type'] == matrix_type]
        
        # Gruppiere nach Größe und Kondition
        sizes = []
        error_ratios = []
        conditions = []
        
        for size in sorted(df_type['size'].unique()):
            df_size = df_type[df_type['size'] == size]
            sizes.append(size)
            error_ratios.append(df_size['error_ratio'].mean())
            conditions.append(df_size['actual_condition'].mean())
        
        plt.plot(sizes, error_ratios, 'o-', label=f"{matrix_type} (κ≈{np.mean(conditions):.1e})")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Matrixgröße (n)') 
    plt.ylabel('Fehlerreduktion (Original/Optimiert)')
    plt.title('Reduktion des numerischen Fehlers durch Optimierungen')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'inverse_error_reduction.png'), dpi=300)
    
    # Plot 3: Cholesky-Vorteile für SPD-Matrizen
    plt.figure(figsize=(12, 8))
    df_chol = pd.DataFrame(cholesky_results)
    
    for size in df_chol['size'].unique():
        df_size = df_chol[df_chol['size'] == size]
        plt.plot(df_size['condition'], df_size['np_chol_speedup'], 'o-', 
                 label=f"{size}x{size}")
    
    plt.xscale('log')
    plt.xlabel('Konditionszahl (κ)') 
    plt.ylabel('Speedup (NumPy/Cholesky)')
    plt.title('Performance-Vorteil der Cholesky-Inversion für SPD-Matrizen')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cholesky_speedup.png'), dpi=300)
    
    # Plot 4: Matrix-Multiplikations-Vorteile
    plt.figure(figsize=(12, 8))
    df_mm = pd.DataFrame(matmul_results)
    
    plt.plot(df_mm['size'], df_mm['numpy_opt_speedup'], 'o-', label='NumPy vs. Optimiert')
    plt.plot(df_mm['size'], df_mm['blas_speedup'], 's-', label='NumPy vs. BLAS')
    
    # Füge Strassen hinzu, nur wo verfügbar
    strassen_sizes = [size for size in df_mm['size'] if not pd.isna(df_mm[df_mm['size']==size]['strassen_speedup'].values[0])]
    strassen_speedups = [df_mm[df_mm['size']==size]['strassen_speedup'].values[0] for size in strassen_sizes]
    
    if strassen_sizes:
        plt.plot(strassen_sizes, strassen_speedups, '^-', label='NumPy vs. Strassen')
    
    plt.xscale('log')
    plt.xlabel('Matrixgröße (n)') 
    plt.ylabel('Speedup Faktor')
    plt.title('Performance-Verbesserung bei Matrix-Multiplikation')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'matmul_speedup.png'), dpi=300)
    
    # Speichere Ergebnisse als CSV
    pd.DataFrame(inverse_results).to_csv(os.path.join(OUTPUT_DIR, 'inverse_results.csv'), index=False)
    pd.DataFrame(cholesky_results).to_csv(os.path.join(OUTPUT_DIR, 'cholesky_results.csv'), index=False)
    pd.DataFrame(matmul_results).to_csv(os.path.join(OUTPUT_DIR, 'matmul_results.csv'), index=False)


def run_complete_benchmark():
    """
    Führt alle Tests aus und generiert einen Bericht.
    """
    print("\n==================================================")
    print("    VX-MATRIX OPTIMIERUNGS-BENCHMARK")
    print("==================================================")
    
    # Test 1: Matrix-Inversions-Stabilität
    inverse_results = test_matrix_inverse_stability()
    
    # Test 2: Cholesky-Vorteil
    cholesky_results = test_cholesky_advantage()
    
    # Test 3: BLAS & Strassen Performance
    matmul_results = test_blas_matmul_performance()
    
    # Generiere Visualisierungen
    plot_results(inverse_results, cholesky_results, matmul_results)
    
    # Zusammenfassung
    print("\n==================================================")
    print("    ZUSAMMENFASSUNG")
    print("==================================================")
    
    # Extrahiere Durchschnittswerte
    df_inv = pd.DataFrame(inverse_results)
    df_inv_success = df_inv[df_inv['orig_success'] & df_inv['opt_success']]
    
    avg_speedup = df_inv_success['speed_ratio'].mean()
    avg_error_reduction = df_inv_success['error_ratio'].mean()
    success_rate_orig = df_inv['orig_success'].mean() * 100
    success_rate_opt = df_inv['opt_success'].mean() * 100
    
    df_chol = pd.DataFrame(cholesky_results)
    avg_chol_speedup = df_chol['np_chol_speedup'].mean()
    
    df_mm = pd.DataFrame(matmul_results)
    avg_mm_speedup = df_mm['numpy_opt_speedup'].mean()
    
    print(f"\n1. Matrix-Inversion Optimierungen:")
    print(f"   - Durchschnittlicher Speedup: {avg_speedup:.2f}x")
    print(f"   - Durchschnittliche Fehlerreduktion: {avg_error_reduction:.2e}x")
    print(f"   - Erfolgsrate Original: {success_rate_orig:.1f}%")
    print(f"   - Erfolgsrate Optimiert: {success_rate_opt:.1f}%")
    
    print(f"\n2. Cholesky-Optimierungen für SPD-Matrizen:")
    print(f"   - Durchschnittlicher Speedup vs. NumPy: {avg_chol_speedup:.2f}x")
    
    print(f"\n3. Matrix-Multiplikation:")
    print(f"   - Durchschnittlicher Speedup vs. NumPy: {avg_mm_speedup:.2f}x")
    
    print("\nVisualisierte Ergebnisse und CSV-Daten wurden gespeichert in:")
    print(f"  {os.path.abspath(OUTPUT_DIR)}")
    
    return {
        "inverse_results": inverse_results,
        "cholesky_results": cholesky_results,
        "matmul_results": matmul_results
    }


if __name__ == "__main__":
    run_complete_benchmark()
