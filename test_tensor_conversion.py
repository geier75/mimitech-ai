#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Engine: Tests für verbesserte Tensor-Konvertierungen

Diese Datei testet die neu implementierten Tensor-Konvertierungsfunktionen
zwischen verschiedenen Backend-Formaten (NumPy, MLX, PyTorch, JAX)
mit besonderem Fokus auf die MLX-Optimierung für Apple Silicon.
"""

import sys
import os
import numpy as np
from optimized_matrix_core import improved_tensor_conversion

def test_tensor_conversion():
    """Test der verbesserten Tensor-Konvertierungsfunktionen"""
    print("Teste verbesserte Tensor-Konvertierungen...")
    
    # Erzeuge Testdaten
    data = np.random.rand(10, 15).astype(np.float32)
    print(f"Originaldaten (NumPy): Shape={data.shape}, Dtype={data.dtype}")
    
    # 1. Test: NumPy -> NumPy (Identitätsfall)
    numpy_result = improved_tensor_conversion(data, target_type='numpy')
    print(f"NumPy -> NumPy: Shape={numpy_result.shape}, Dtype={numpy_result.dtype}")
    print(f"Datengleichheit: {np.array_equal(data, numpy_result)}")
    
    # 2. Test: NumPy -> MLX (wenn verfügbar)
    try:
        import mlx.core as mx
        print("\nMLX Backend verfügbar, teste MLX-Konvertierung...")
        
        # NumPy -> MLX
        mlx_tensor = improved_tensor_conversion(data, target_type='mlx')
        print(f"NumPy -> MLX: Shape={mlx_tensor.shape}, Dtype={mlx_tensor.dtype}")
        
        # MLX -> NumPy 
        numpy_from_mlx = improved_tensor_conversion(mlx_tensor, target_type='numpy')
        print(f"MLX -> NumPy: Shape={numpy_from_mlx.shape}, Dtype={numpy_from_mlx.dtype}")
        
        # Vergleich der Daten
        data_equal = np.allclose(data, numpy_from_mlx, rtol=1e-5, atol=1e-5)
        print(f"Datengleichheit nach Rundreise (NumPy -> MLX -> NumPy): {data_equal}")
        
    except ImportError:
        print("\nMLX nicht verfügbar, überspringe MLX-Tests...")
    
    # 3. Test: NumPy -> PyTorch (wenn verfügbar)
    try:
        import torch
        print("\nPyTorch Backend verfügbar, teste PyTorch-Konvertierung...")
        
        # NumPy -> PyTorch
        torch_tensor = improved_tensor_conversion(data, target_type='torch')
        print(f"NumPy -> PyTorch: Shape={torch_tensor.shape}, Dtype={torch_tensor.dtype}")
        
        # PyTorch -> NumPy
        numpy_from_torch = improved_tensor_conversion(torch_tensor, target_type='numpy')
        print(f"PyTorch -> NumPy: Shape={numpy_from_torch.shape}, Dtype={numpy_from_torch.dtype}")
        
        # Vergleich der Daten
        data_equal = np.allclose(data, numpy_from_torch, rtol=1e-5, atol=1e-5)
        print(f"Datengleichheit nach Rundreise (NumPy -> PyTorch -> NumPy): {data_equal}")
        
    except ImportError:
        print("\nPyTorch nicht verfügbar, überspringe PyTorch-Tests...")
    
    # 4. Test: Fehlerbehandlung
    print("\nTeste Fehlerbehandlung...")
    try:
        # Ungültiger Ziel-Typ
        improved_tensor_conversion(data, target_type='invalid_backend')
        print("FEHLER: Exception für ungültigen Zieltyp wurde nicht ausgelöst!")
    except ValueError as e:
        print(f"Korrekte Exception für ungültigen Zieltyp: {e}")
    
    # 5. Test: Spezialfall für bfloat16 bei MLX
    try:
        import mlx.core as mx
        print("\nTeste Spezialfall bfloat16 mit MLX...")
        
        # Erzeuge bfloat16 Daten in MLX
        mlx_bf16 = mx.array(data, dtype=mx.bfloat16)
        print(f"MLX bfloat16: Shape={mlx_bf16.shape}, Dtype={mlx_bf16.dtype}")
        
        # MLX bfloat16 -> NumPy
        numpy_from_bf16 = improved_tensor_conversion(mlx_bf16, target_type='numpy')
        print(f"MLX bfloat16 -> NumPy: Shape={numpy_from_bf16.shape}, Dtype={numpy_from_bf16.dtype}")
        
        # Vergleich der Daten (mit höherer Toleranz wegen bfloat16 Genauigkeit)
        data_close = np.allclose(data, numpy_from_bf16, rtol=1e-2, atol=1e-2)
        print(f"Datenähnlichkeit nach bfloat16 Konvertierung: {data_close}")
        
    except (ImportError, AttributeError):
        print("\nMLX mit bfloat16 nicht verfügbar, überspringe bfloat16-Tests...")
    
    print("\nTest der Tensor-Konvertierung abgeschlossen.")

if __name__ == "__main__":
    test_tensor_conversion()
