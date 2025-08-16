#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test der optimierten MatrixCore-Implementierung
"""

import sys
import os
import numpy as np

# Pfad zur matrix_core.py Datei hinzufügen
matrix_core_path = os.path.join(os.path.dirname(__file__), 'vxor.ai', 'VX-MATRIX', 'core')
sys.path.insert(0, matrix_core_path)

from matrix_core import MatrixCore

def test_matrix_core():
    print("Starte MatrixCore-Test...")
    
    # MatrixCore mit MLX als bevorzugtem Backend initialisieren
    core = MatrixCore(preferred_backend="mlx")
    core.enable_jit = True
    
    print(f"MatrixCore initialisiert mit Backend: {core.preferred_backend}")
    print(f"JIT-Kompilierung: {'aktiviert' if core.enable_jit else 'deaktiviert'}")
    
    # Einfachen Multiplikations- und Inverse-Durchlauf starten
    print("\nFühre Matrix-Operationen aus...")
    a = np.random.rand(50, 50)
    print(f"Matrix-Größe: {a.shape}")
    
    print("1. Matrix-Multiplikation")
    result_mult = core.matrix_multiply(a, a)
    print(f"Ergebnis-Shape: {result_mult.shape}")
    
    print("2. Matrix-Inversion")
    result_inv = core.matrix_inverse(a)
    print(f"Ergebnis-Shape: {result_inv.shape}")
    
    # Verifizieren der Korrektheit
    print("\nPrüfe Ergebnisse...")
    np_mult = np.matmul(a, a)
    np_inv = np.linalg.inv(a)
    
    print(f"Matrix-Multiplikation Genauigkeit: {np.allclose(result_mult, np_mult, rtol=1e-5, atol=1e-5)}")
    print(f"Matrix-Inversion Genauigkeit: {np.allclose(result_inv, np_inv, rtol=1e-5, atol=1e-5)}")
    
    # Metriken als JSON exportieren
    metrics_path = "/Volumes/My Book/MISO_Ultimate 15.32.28/vxor.ai/VX-MATRIX/core/test_metrics.json"
    print(f"\nExportiere Performance-Metriken nach {metrics_path}")
    core.export_performance_metrics(metrics_path)
    
    print("\nTest erfolgreich abgeschlossen!")

if __name__ == "__main__":
    test_matrix_core()
