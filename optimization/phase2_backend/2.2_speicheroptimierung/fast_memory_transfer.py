#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimierter Speichertransfer für T-Mathematics Engine

Diese Datei implementiert einen hocheffizienten Speichertransfer zwischen 
MLX und MPS Tensoren mit minimalem Overhead.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import time
import numpy as np
import torch
from typing import Dict, Any, Optional, Union, Tuple

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.fast_memory_transfer")

# MLX-Verfügbarkeit prüfen
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    logger.warning("MLX konnte nicht importiert werden. Optimierte Speichertransfers sind nicht verfügbar.")

# Apple Silicon-Erkennung
IS_APPLE_SILICON = False
try:
    import platform
    IS_APPLE_SILICON = platform.processor() == 'arm' and platform.system() == 'Darwin'
except:
    pass

# Optimale Transfermethode wird beim Import bestimmt
_OPTIMAL_MPS_TO_MLX_METHOD = None
_OPTIMAL_MLX_TO_MPS_METHOD = None

# MLX Version bestimmen für API-Kompatibilität
_MLX_VERSION = None
try:
    import pkg_resources
    _MLX_VERSION = pkg_resources.get_distribution("mlx").version
    logger.info(f"MLX Version: {_MLX_VERSION}")
except:
    pass

def _determine_optimal_transfer_methods():
    """
    Bestimmt die optimalen Transfermethoden basierend auf Benchmarks
    und setzt die globalen Variablen.
    """
    global _OPTIMAL_MPS_TO_MLX_METHOD, _OPTIMAL_MLX_TO_MPS_METHOD
    
    if not HAS_MLX or not torch.backends.mps.is_available():
        logger.warning("MLX oder MPS nicht verfügbar, Transferoptimierung nicht möglich")
        return
    
    logger.info("Bestimme optimale Transfermethoden...")
    
    # Test-Tensoren erstellen
    test_size = (64, 64)
    mps_tensor = torch.randn(test_size, device="mps")
    
    # Für MPS -> MLX
    methods = [
        ("direct", _mps_to_mlx_direct),
        ("via_cpu", _mps_to_mlx_via_cpu),
        ("via_numpy", _mps_to_mlx_via_numpy)
    ]
    
    best_method = None
    best_time = float('inf')
    
    # Benchmark jede Methode
    for name, method in methods:
        try:
            # Warmup
            for _ in range(2):
                mlx_result = method(mps_tensor)
                del mlx_result
            
            # Timing
            start = time.time()
            for _ in range(10):
                mlx_result = method(mps_tensor)
                del mlx_result
            elapsed = time.time() - start
            
            logger.info(f"MPS->MLX Methode '{name}': {elapsed:.6f}s")
            
            if elapsed < best_time:
                best_time = elapsed
                best_method = name
        except Exception as e:
            logger.warning(f"Methode '{name}' für MPS->MLX fehlgeschlagen: {e}")
    
    _OPTIMAL_MPS_TO_MLX_METHOD = best_method
    logger.info(f"Optimale MPS->MLX Methode: {best_method} ({best_time:.6f}s)")
    
    # MLX Array erstellen für MLX -> MPS Tests
    try:
        mlx_array = mx.random.normal(test_size)
        
        # Für MLX -> MPS
        methods = [
            ("direct", _mlx_to_mps_direct),
            ("via_cpu", _mlx_to_mps_via_cpu),
            ("via_numpy", _mlx_to_mps_via_numpy)
        ]
        
        best_method = None
        best_time = float('inf')
        
        # Benchmark jede Methode
        for name, method in methods:
            try:
                # Warmup
                for _ in range(2):
                    torch_result = method(mlx_array)
                    del torch_result
                
                # Timing
                start = time.time()
                for _ in range(10):
                    torch_result = method(mlx_array)
                    del torch_result
                elapsed = time.time() - start
                
                logger.info(f"MLX->MPS Methode '{name}': {elapsed:.6f}s")
                
                if elapsed < best_time:
                    best_time = elapsed
                    best_method = name
            except Exception as e:
                logger.warning(f"Methode '{name}' für MLX->MPS fehlgeschlagen: {e}")
        
        _OPTIMAL_MLX_TO_MPS_METHOD = best_method
        logger.info(f"Optimale MLX->MPS Methode: {best_method} ({best_time:.6f}s)")
    
    except Exception as e:
        logger.warning(f"Konnte MLX->MPS Methoden nicht testen: {e}")

# Implementierung der verschiedenen Transfermethoden

def _mps_to_mlx_direct(mps_tensor):
    """Direkte Konvertierung von MPS nach MLX mit minimal möglichen Schritten"""
    with torch.no_grad():
        # Wenn MLX direkte Metal-Buffer-Kopplung unterstützt (für zukünftige Versionen)
        # würde das hier implementiert werden
        
        # Derzeit effizientester Weg
        cpu_tensor = mps_tensor.detach().cpu()
        # Der Aufruf von numpy() scheint schneller als tolist() in diesem Fall
        try:
            # Direkt zu float32 für konsistente Typen
            numpy_tensor = cpu_tensor.numpy().astype(np.float32)
            return mx.array(numpy_tensor)
        except:
            # Fallback ohne Typkonvertierung
            return mx.array(cpu_tensor.numpy())

def _mps_to_mlx_via_cpu(mps_tensor):
    """Konvertierung von MPS nach MLX über CPU-Tensor"""
    with torch.no_grad():
        cpu_tensor = mps_tensor.detach().cpu()
        return mx.array(cpu_tensor.tolist(), dtype=mx.float32)

def _mps_to_mlx_via_numpy(mps_tensor):
    """Konvertierung von MPS nach MLX über NumPy"""
    with torch.no_grad():
        # Explizite Konversion zu float32 für MPS-Kompatibilität
        numpy_tensor = mps_tensor.detach().cpu().numpy().astype(np.float32)
        return mx.array(numpy_tensor)

def _mlx_to_mps_direct(mlx_array):
    """Direkte Konvertierung von MLX nach MPS mit minimal möglichen Schritten"""
    # Wenn MLX direkte Metal-Buffer-Kopplung unterstützt (für zukünftige Versionen)
    # würde das hier implementiert werden
    
    # Derzeit effizientester Weg
    try:
        # Bei neueren MLX-Versionen könnte es eine numpy() Methode geben
        numpy_array = np.array(mlx_array.tolist(), dtype=np.float32)
    except:
        # Fallback
        numpy_array = np.array(mlx_array.tolist(), dtype=np.float32)
    
    # Explizit dtype=torch.float32 verwenden, da MPS kein float64 unterstützt
    return torch.tensor(numpy_array, dtype=torch.float32, device="mps")

def _mlx_to_mps_via_cpu(mlx_array):
    """Konvertierung von MLX nach MPS über CPU-Tensor"""
    cpu_tensor = torch.tensor(mlx_array.tolist(), dtype=torch.float32)
    return cpu_tensor.to(device="mps")

def _mlx_to_mps_via_numpy(mlx_array):
    """Konvertierung von MLX nach MPS über NumPy"""
    numpy_array = np.array(mlx_array.tolist(), dtype=np.float32)
    return torch.from_numpy(numpy_array).to(device="mps", dtype=torch.float32)

# Öffentliche Funktionen

def mps_to_mlx(mps_tensor):
    """
    Hochoptimierte Konvertierung von MPS-Tensor zu MLX-Array.
    
    Diese Funktion verwendet die schnellste verfügbare Methode basierend
    auf Laufzeitbenchmarks.
    
    Args:
        mps_tensor: PyTorch-Tensor auf MPS-Gerät
        
    Returns:
        MLX-Array
    """
    if not HAS_MLX:
        raise ImportError("MLX ist nicht verfügbar")
    
    if not isinstance(mps_tensor, torch.Tensor):
        raise TypeError(f"Erwartet PyTorch-Tensor, erhalten {type(mps_tensor)}")
    
    if mps_tensor.device.type != 'mps':
        # Wenn es kein MPS-Tensor ist, direkt zu MLX konvertieren
        if mps_tensor.device.type == 'cpu':
            return mx.array(mps_tensor.detach().numpy().astype(np.float32))
        else:
            # Für andere Geräte (z.B. CUDA) erst zu CPU
            return mx.array(mps_tensor.detach().cpu().numpy().astype(np.float32))
    
    # Optimale Methode basierend auf Benchmark wählen
    if _OPTIMAL_MPS_TO_MLX_METHOD == "direct":
        return _mps_to_mlx_direct(mps_tensor)
    elif _OPTIMAL_MPS_TO_MLX_METHOD == "via_cpu":
        return _mps_to_mlx_via_cpu(mps_tensor)
    elif _OPTIMAL_MPS_TO_MLX_METHOD == "via_numpy":
        return _mps_to_mlx_via_numpy(mps_tensor)
    else:
        # Fallback, wenn keine optimale Methode bestimmt wurde
        return _mps_to_mlx_direct(mps_tensor)

def mlx_to_mps(mlx_array):
    """
    Hochoptimierte Konvertierung von MLX-Array zu MPS-Tensor.
    
    Diese Funktion verwendet die schnellste verfügbare Methode basierend
    auf Laufzeitbenchmarks.
    
    Args:
        mlx_array: MLX-Array
        
    Returns:
        PyTorch-Tensor auf MPS-Gerät
    """
    if not HAS_MLX:
        raise ImportError("MLX ist nicht verfügbar")
    
    if not torch.backends.mps.is_available():
        # Wenn MPS nicht verfügbar ist, zu CPU konvertieren
        numpy_array = np.array(mlx_array.tolist(), dtype=np.float32)
        return torch.from_numpy(numpy_array)
    
    # Optimale Methode basierend auf Benchmark wählen
    if _OPTIMAL_MLX_TO_MPS_METHOD == "direct":
        return _mlx_to_mps_direct(mlx_array)
    elif _OPTIMAL_MLX_TO_MPS_METHOD == "via_cpu":
        return _mlx_to_mps_via_cpu(mlx_array)
    elif _OPTIMAL_MLX_TO_MPS_METHOD == "via_numpy":
        return _mlx_to_mps_via_numpy(mlx_array)
    else:
        # Fallback, wenn keine optimale Methode bestimmt wurde
        return _mlx_to_mps_direct(mlx_array)

# Führe die Bestimmung der optimalen Transfermethoden beim Import aus
if HAS_MLX and torch.backends.mps.is_available() and IS_APPLE_SILICON:
    _determine_optimal_transfer_methods()
else:
    logger.warning("Optimale Transfermethoden konnten nicht bestimmt werden: MLX oder MPS nicht verfügbar")
