#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybride Speichertransfer-Strategie für T-Mathematics Engine

Diese Datei implementiert eine größenabhängige Strategie für optimale
Speichertransfers zwischen MLX und MPS Tensoren.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, Union, Tuple

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.hybrid_memory_transfer")

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

# MLX Version bestimmen für API-Kompatibilität
_MLX_VERSION = None
try:
    import pkg_resources
    _MLX_VERSION = pkg_resources.get_distribution("mlx").version
    logger.info(f"MLX Version: {_MLX_VERSION}")
except:
    pass

# Größenschwellwerte für Strategieauswahl
SMALL_TENSOR_THRESHOLD = 128 * 128  # 128x128 und kleiner
MEDIUM_TENSOR_THRESHOLD = 512 * 512  # 512x512 und kleiner

def estimate_tensor_size(shape):
    """
    Schätzt die Anzahl der Elemente in einem Tensor basierend auf seiner Form
    
    Args:
        shape: Tensorform als Tuple oder Liste
        
    Returns:
        Anzahl der Elemente
    """
    if not shape:
        return 0
    
    elements = 1
    for dim in shape:
        elements *= dim
    
    return elements

# Implementierungen für kleine Tensoren (optimierte Version)

def _small_mps_to_mlx(mps_tensor):
    """Optimierte Konvertierung für kleine MPS-Tensoren nach MLX"""
    with torch.no_grad():
        numpy_tensor = mps_tensor.detach().cpu().numpy().astype(np.float32)
        return mx.array(numpy_tensor)

def _small_mlx_to_mps(mlx_array):
    """Optimierte Konvertierung für kleine MLX-Arrays nach MPS"""
    numpy_array = np.array(mlx_array.tolist(), dtype=np.float32)
    return torch.tensor(numpy_array, dtype=torch.float32, device="mps")

# Implementierungen für mittlere Tensoren (vereinfacht und optimiert)

def _medium_mps_to_mlx(mps_tensor):
    """Vereinfachte direkte Konvertierung für mittlere MPS-Tensoren nach MLX"""
    with torch.no_grad():
        # Minimaler Pfad mit nur den notwendigen Schritten
        return mx.array(mps_tensor.detach().cpu().numpy())

def _medium_mlx_to_mps(mlx_array):
    """Vereinfachte direkte Konvertierung für mittlere MLX-Arrays nach MPS"""
    try:
        # Versuche direkte Konvertierung
        numpy_array = np.array(mlx_array.tolist(), dtype=np.float32)
        return torch.from_numpy(numpy_array).to(device="mps")
    except:
        # Fallback
        return torch.tensor(mlx_array.tolist(), dtype=torch.float32, device="mps")

# Implementierungen für große Tensoren (ähnlich zur ursprünglichen Implementierung)

def _large_mps_to_mlx(mps_tensor):
    """Einfachste und schnellste Konvertierung für große MPS-Tensoren nach MLX"""
    try:
        # Direkte Konvertierung wie im Original, minimaler Overhead
        numpy_tensor = mps_tensor.cpu().detach().numpy()
        return mx.array(numpy_tensor)
    except:
        # Minimaler Fallback
        return mx.array(mps_tensor.cpu().detach().tolist())

def _large_mlx_to_mps(mlx_array):
    """Einfachste und schnellste Konvertierung für große MLX-Arrays nach MPS"""
    try:
        # Schnellste Methode für große Arrays
        return torch.tensor(mlx_array.tolist(), device="mps")
    except:
        # Fallback
        return torch.tensor(np.array(mlx_array.tolist()), device="mps")

# Öffentliche Funktionen mit größenabhängiger Strategieauswahl

def mps_to_mlx(mps_tensor):
    """
    Größenoptimierte Konvertierung von MPS-Tensor zu MLX-Array.
    
    Diese Funktion wählt die optimale Konvertierungsstrategie basierend
    auf der Tensorgröße.
    
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
            return mx.array(mps_tensor.detach().numpy())
        else:
            # Für andere Geräte erst zu CPU
            return mx.array(mps_tensor.detach().cpu().numpy())
    
    # Größenabhängige Strategieauswahl
    num_elements = estimate_tensor_size(mps_tensor.shape)
    
    if num_elements <= SMALL_TENSOR_THRESHOLD:
        # Kleine Tensoren: Vollständig optimierte Version
        return _small_mps_to_mlx(mps_tensor)
    elif num_elements <= MEDIUM_TENSOR_THRESHOLD:
        # Mittlere Tensoren: Vereinfachte optimierte Version
        return _medium_mps_to_mlx(mps_tensor)
    else:
        # Große Tensoren: Schnellste Version mit minimalem Overhead
        return _large_mps_to_mlx(mps_tensor)

def mlx_to_mps(mlx_array):
    """
    Größenoptimierte Konvertierung von MLX-Array zu MPS-Tensor.
    
    Diese Funktion wählt die optimale Konvertierungsstrategie basierend
    auf der Arraygröße.
    
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
    
    # Größenabhängige Strategieauswahl
    num_elements = estimate_tensor_size(mlx_array.shape)
    
    if num_elements <= SMALL_TENSOR_THRESHOLD:
        # Kleine Arrays: Vollständig optimierte Version
        return _small_mlx_to_mps(mlx_array)
    elif num_elements <= MEDIUM_TENSOR_THRESHOLD:
        # Mittlere Arrays: Vereinfachte optimierte Version
        return _medium_mlx_to_mps(mlx_array)
    else:
        # Große Arrays: Schnellste Version mit minimalem Overhead
        return _large_mlx_to_mps(mlx_array)

# Initialisierungslog
if HAS_MLX and torch.backends.mps.is_available() and IS_APPLE_SILICON:
    logger.info("Hybride Speichertransfers initialisiert mit Größenschwellwerten:")
    logger.info(f"- Kleine Tensoren: ≤ {SMALL_TENSOR_THRESHOLD} Elemente")
    logger.info(f"- Mittlere Tensoren: ≤ {MEDIUM_TENSOR_THRESHOLD} Elemente")
    logger.info(f"- Große Tensoren: > {MEDIUM_TENSOR_THRESHOLD} Elemente")
else:
    logger.warning("Hybride Speichertransfers nicht verfügbar: MLX oder MPS nicht unterstützt")
