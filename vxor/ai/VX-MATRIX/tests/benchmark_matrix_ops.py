#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MATRIX Benchmark für Matrixoperationen
=========================================

Analysiert die Performance von Matrixoperationen mit dem Profiling-Framework.
Vergleicht die originalen Implementierungen mit den optimierten Versionen.

ZTM-konform für MISO Ultimate.
"""

import numpy as np
import sys
import os
import time
import json
import pandas as pd
from scipy import linalg

# Visualisierung deaktiviert wegen Unicode-Problemen in matplotlib
VISUALIZATION_ENABLED = False

# Pfade für den Import konfigurieren
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

# VX-MATRIX-Module importieren
from core.matrix_core import MatrixCore, ztm_log
from optimizers.optimized_matmul import optimized_matrix_multiply
from optimizers.optimized_inverse import optimized_matrix_inverse, is_spd

# Überprüfen, ob das Profiling-Modul verfügbar ist
try:
    from utils.profiling import profile_function, time_function, performance_metrics
    PROFILING_ENABLED = True
except ImportError:
    # Dummy-Dekoratoren für den Fall, dass Profiling nicht verfügbar ist
    def profile_function(func): return func
    def time_function(func): return func
    PROFILING_ENABLED = False

# Ergebnisverzeichnis
RESULTS_DIR = os.path.join(root_dir, 'profiling', 'benchmark_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

class MatrixBenchmark:
    """
    Benchmark-Suite für VX-MATRIX Optimierungen.
    Vergleicht die Performance der Original-Implementierung mit optimierten Versionen.
    """
    
    def __init__(self):
        """Initialisiert die Benchmark-Suite."""
        self.matrix_core = MatrixCore()
        self.results = {
            'matrix_multiplication': [],
            'matrix_inversion': [],
            'spd_inversion': []
        }
        self.matrix_sizes = [10, 50, 100, 200, 500, 1000]
        self.condition_numbers = [1.0e2, 1.0e4, 1.0e6, 1.0e8]
    
    def generate_random_matrix(self, size, condition_number=None):
        """Erzeugt eine zufällige Matrix mit optionaler Konditionszahl."""
        if condition_number is None:
            # Standard-Zufallsmatrix
            return np.random.randn(size, size)
        else:
            # Matrix mit spezifischer Konditionszahl
            # Erzeuge orthogonale Matrix über QR-Zerlegung
            A = np.random.randn(size, size)
            Q, _ = np.linalg.qr(A)
            
            # Erzeuge Diagonalmatrix mit abnehmenden Singulärwerten
            singular_values = np.linspace(condition_number, 1, size)
            S = np.diag(singular_values)
            
            # Kombiniere zu Matrix mit spezifischer Konditionszahl
            return Q @ S @ Q.T
    
    def generate_spd_matrix(self, size, condition_number=None):
        """Erzeugt eine symmetrisch positiv-definite Matrix mit optionaler Konditionszahl."""
        A = np.random.randn(size, size)
        # SPD-Matrix durch A*A^T erzeugen
        spd = A @ A.T
        
        if condition_number is not None:
            # Konditionszahl anpassen
            eigvals, eigvecs = np.linalg.eigh(spd)
            smallest = eigvals.min()
            eigvals = np.linspace(smallest * condition_number, smallest, size)
            return eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        return spd
    
    @profile_function
    def benchmark_matrix_multiplication(self):
        """Benchmark für Matrixmultiplikation."""
        print("\n=== Matrix-Multiplikations-Benchmark ===")
        
        for size in self.matrix_sizes:
            try:
                # Erzeuge Zufallsmatrizen
                A = self.generate_random_matrix(size)
                B = self.generate_random_matrix(size)
                
                # Original VX-MATRIX Implementierung
                start_time = time.time()
                original_result = self.matrix_core.matrix_multiply(A, B)
                original_time = time.time() - start_time
                
                # NumPy-Referenz
                start_time = time.time()
                numpy_result = np.matmul(A, B)
                numpy_time = time.time() - start_time
                
                # Optimierte Implementierung
                start_time = time.time()
                optimized_result = optimized_matrix_multiply(A, B)
                optimized_time = time.time() - start_time
                
                # Fehlerberechnung
                error_numpy = np.max(np.abs(original_result - numpy_result))
                error_optimized = np.max(np.abs(original_result - optimized_result))
                
                # Speedup-Berechnung
                numpy_speedup = numpy_time / original_time if original_time > 0 else float('inf')
                optimized_speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
                
                # Ergebnis ausgeben
                print(f"Matrix-Größe {size}x{size}:")
                print(f"  Original: {original_time:.6f}s, NumPy: {numpy_time:.6f}s, Optimiert: {optimized_time:.6f}s")
                print(f"  Speedup vs NumPy: {numpy_speedup:.2f}x, Speedup optimiert: {optimized_speedup:.2f}x")
                print(f"  Fehler NumPy: {error_numpy:.2e}, Fehler optimiert: {error_optimized:.2e}")
                
                # Ergebnis speichern
                self.results['matrix_multiplication'].append({
                    'size': size,
                    'original_time': original_time,
                    'numpy_time': numpy_time,
                    'optimized_time': optimized_time,
                    'numpy_speedup': numpy_speedup,
                    'optimized_speedup': optimized_speedup,
                    'error_numpy': float(error_numpy),
                    'error_optimized': float(error_optimized)
                })
            except Exception as e:
                print(f"Fehler bei Matrix-Größe {size}x{size}: {e}")
    
    @profile_function
    def benchmark_matrix_inversion(self):
        """Benchmark für Matrixinversion."""
        print("\n=== Matrix-Inversions-Benchmark ===")
        
        for size in [10, 50, 100, 200, 500]:
            for condition in [1e2, 1e6, 1e8]:
                try:
                    # Erzeuge Matrix mit spezifischer Konditionszahl
                    A = self.generate_random_matrix(size, condition)
                    
                    # Original VX-MATRIX Implementierung
                    start_time = time.time()
                    original_result = self.matrix_core.matrix_inverse(A)
                    original_time = time.time() - start_time
                    
                    # NumPy-Referenz
                    start_time = time.time()
                    numpy_result = np.linalg.inv(A)
                    numpy_time = time.time() - start_time
                    
                    # Optimierte Implementierung
                    start_time = time.time()
                    optimized_result = optimized_matrix_inverse(A)
                    optimized_time = time.time() - start_time
                    
                    # Fehlerberechnung - verwende A·A⁻¹ ≈ I als Maß
                    error_original = np.max(np.abs(A @ original_result - np.eye(size)))
                    error_numpy = np.max(np.abs(A @ numpy_result - np.eye(size)))
                    error_optimized = np.max(np.abs(A @ optimized_result - np.eye(size)))
                    
                    # Speedup-Berechnung
                    numpy_speedup = numpy_time / original_time if original_time > 0 else float('inf')
                    optimized_speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
                    
                    # Ergebnis ausgeben
                    print(f"Matrix-Größe {size}x{size}, κ={condition:.1e}:")
                    print(f"  Original: {original_time:.6f}s, NumPy: {numpy_time:.6f}s, Optimiert: {optimized_time:.6f}s")
                    print(f"  Speedup vs NumPy: {numpy_speedup:.2f}x, Speedup optimiert: {optimized_speedup:.2f}x")
                    print(f"  Fehler Original: {error_original:.2e}, NumPy: {error_numpy:.2e}, Optimiert: {error_optimized:.2e}")
                    
                    # Ergebnis speichern
                    self.results['matrix_inversion'].append({
                        'size': size,
                        'condition': float(condition),
                        'original_time': original_time,
                        'numpy_time': numpy_time,
                        'optimized_time': optimized_time,
                        'numpy_speedup': numpy_speedup,
                        'optimized_speedup': optimized_speedup,
                        'error_original': float(error_original),
                        'error_numpy': float(error_numpy),
                        'error_optimized': float(error_optimized)
                    })
                except Exception as e:
                    print(f"Fehler bei Matrix-Größe {size}x{size}, κ={condition:.1e}: {e}")
    
    @profile_function
    def benchmark_spd_inversion(self):
        """Benchmark für SPD-Matrixinversion."""
        print("\n=== SPD-Matrix-Inversions-Benchmark ===")
        
        for size in [10, 50, 100, 200, 500]:
            for condition in [1e2, 1e6, 1e8]:
                try:
                    # Erzeuge SPD-Matrix mit spezifischer Konditionszahl
                    A = self.generate_spd_matrix(size, condition)
                    
                    # Prüfe, ob Matrix wirklich SPD ist
                    if not is_spd(A):
                        print(f"Generierte Matrix {size}x{size}, κ={condition:.1e} ist nicht SPD!")
                        continue
                    
                    # Original VX-MATRIX Implementierung - _cholesky_inverse direkt aufrufen
                    start_time = time.time()
                    original_result = self.matrix_core._cholesky_inverse(A)
                    original_time = time.time() - start_time
                    
                    # NumPy-Referenz
                    start_time = time.time()
                    numpy_result = np.linalg.inv(A)
                    numpy_time = time.time() - start_time
                    
                    # Optimierte Implementierung - mit Methode 'cholesky' erzwingen
                    start_time = time.time()
                    optimized_result = optimized_matrix_inverse(A, method='cholesky')
                    optimized_time = time.time() - start_time
                    
                    # Fehlerberechnung
                    error_original = np.max(np.abs(A @ original_result - np.eye(size)))
                    error_numpy = np.max(np.abs(A @ numpy_result - np.eye(size)))
                    error_optimized = np.max(np.abs(A @ optimized_result - np.eye(size)))
                    
                    # Speedup-Berechnung
                    numpy_speedup = numpy_time / original_time if original_time > 0 else float('inf')
                    optimized_speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
                    
                    # Ergebnis ausgeben
                    print(f"SPD-Matrix {size}x{size}, κ={condition:.1e}:")
                    print(f"  Original: {original_time:.6f}s, NumPy: {numpy_time:.6f}s, Optimiert: {optimized_time:.6f}s")
                    print(f"  Speedup vs NumPy: {numpy_speedup:.2f}x, Speedup optimiert: {optimized_speedup:.2f}x")
                    print(f"  Fehler Original: {error_original:.2e}, NumPy: {error_numpy:.2e}, Optimiert: {error_optimized:.2e}")
                    
                    # Ergebnis speichern
                    self.results['spd_inversion'].append({
                        'size': size,
                        'condition': float(condition),
                        'original_time': original_time,
                        'numpy_time': numpy_time,
                        'optimized_time': optimized_time,
                        'numpy_speedup': numpy_speedup,
                        'optimized_speedup': optimized_speedup,
                        'error_original': float(error_original),
                        'error_numpy': float(error_numpy),
                        'error_optimized': float(error_optimized)
                    })
                except Exception as e:
                    print(f"Fehler bei SPD-Matrix {size}x{size}, κ={condition:.1e}: {e}")
    
    def run_benchmarks(self):
        """Führt alle Benchmarks aus."""
        print("Starte VX-MATRIX Benchmarks...")
        
        self.benchmark_matrix_multiplication()
        self.benchmark_matrix_inversion()
        self.benchmark_spd_inversion()
        
        # Speichere die Ergebnisse
        self.save_results()
        
        # Visualisiere die Ergebnisse (ohne Plots)
        self.visualize_results()
        
        print("\nBenchmarks abgeschlossen. Ergebnisse gespeichert in:", RESULTS_DIR)
    
    def save_results(self):
        """Speichert die Benchmark-Ergebnisse als JSON."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(RESULTS_DIR, f"benchmark_results_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Exportiere auch die Profiling-Metriken, wenn verfügbar
        if PROFILING_ENABLED:
            try:
                profile_file = performance_metrics.save_metrics()
                print(f"Performance-Metriken gespeichert in: {profile_file}")
            except Exception as e:
                print(f"Fehler beim Speichern der Performance-Metriken: {e}")
    
    def visualize_results(self):
        """Visualisiert die Benchmark-Ergebnisse (ohne Matplotlib)."""
        if not VISUALIZATION_ENABLED:
            print("\nVisualisierung deaktiviert. Nur numerische Ergebnisse werden angezeigt.")
            
            # Matrixmultiplikations-Zusammenfassung
            if self.results['matrix_multiplication']:
                df = pd.DataFrame(self.results['matrix_multiplication'])
                print("\n=== Matrixmultiplikations-Zusammenfassung ===")
                print(f"Durchschnittlicher Speedup (Optimiert vs. Original): {df['optimized_speedup'].mean():.2f}x")
                print(f"Maximaler Speedup: {df['optimized_speedup'].max():.2f}x bei Matrixgröße {df.loc[df['optimized_speedup'].idxmax(), 'size']}")
                print(f"Durchschnittlicher Fehler (Optimiert): {df['error_optimized'].mean():.2e}")
            
            # Matrixinversions-Zusammenfassung
            if self.results['matrix_inversion']:
                df = pd.DataFrame(self.results['matrix_inversion'])
                print("\n=== Matrixinversions-Zusammenfassung ===")
                print(f"Durchschnittlicher Speedup (Optimiert vs. Original): {df['optimized_speedup'].mean():.2f}x")
                print(f"Maximaler Speedup: {df['optimized_speedup'].max():.2f}x")
                print(f"Durchschnittlicher Fehler (Optimiert): {df['error_optimized'].mean():.2e}")
                
                # Nach Konditionszahl gruppieren
                for condition in df['condition'].unique():
                    df_cond = df[df['condition'] == condition]
                    print(f"  - Konditionszahl κ={condition:.1e}:")
                    print(f"    Durchschnittlicher Speedup: {df_cond['optimized_speedup'].mean():.2f}x")
                    print(f"    Durchschnittlicher Fehler: {df_cond['error_optimized'].mean():.2e}")
            
            # SPD-Matrixinversions-Zusammenfassung
            if self.results['spd_inversion']:
                df = pd.DataFrame(self.results['spd_inversion'])
                print("\n=== SPD-Matrixinversions-Zusammenfassung ===")
                print(f"Durchschnittlicher Speedup (Optimiert vs. Original): {df['optimized_speedup'].mean():.2f}x")
                print(f"Maximaler Speedup: {df['optimized_speedup'].max():.2f}x")
                print(f"Durchschnittlicher Fehler (Optimiert): {df['error_optimized'].mean():.2e}")
                
                # Nach Konditionszahl gruppieren
                for condition in df['condition'].unique():
                    df_cond = df[df['condition'] == condition]
                    print(f"  - Konditionszahl κ={condition:.1e}:")
                    print(f"    Durchschnittlicher Speedup: {df_cond['optimized_speedup'].mean():.2f}x")
                    print(f"    Durchschnittlicher Fehler: {df_cond['error_optimized'].mean():.2e}")
            
            return


def main():
    """Hauptfunktion."""
    benchmark = MatrixBenchmark()
    benchmark.run_benchmarks()


if __name__ == "__main__":
    main()
