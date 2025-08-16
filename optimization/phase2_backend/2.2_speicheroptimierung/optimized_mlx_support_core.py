#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimiertes MLX-Support-Modul für T-Mathematics Engine

Diese Datei implementiert ein optimiertes MLX-Backend mit verbesserter Speicherverwaltung,
direkten Geräteübertragungen und effizientem Caching für maximale Leistung auf Apple Silicon.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import time
import math
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import lru_cache

# Importiere die optimierte Speicherverwaltung
from tensor_pool import TensorPool
from direct_memory_transfer import (
    DirectMemoryTransfer, mps_to_mlx, mlx_to_mps
)

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.optimized_mlx_support")

# MLX-Verfügbarkeit prüfen
try:
    import mlx.core as mx
    import mlx.nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    logger.warning("MLX konnte nicht importiert werden. Apple Silicon Optimierungen sind nicht verfügbar.")

# Apple Silicon-Erkennung
IS_APPLE_SILICON = False
try:
    import platform
    IS_APPLE_SILICON = platform.processor() == 'arm' and platform.system() == 'Darwin'
    if IS_APPLE_SILICON:
        logger.info("Apple Silicon erkannt, optimiere für Neural Engine")
except Exception as e:
    logger.warning(f"Fehler bei der Apple Silicon-Erkennung: {e}")

# Importiere VXOR-Integration, wenn verfügbar
try:
    from miso.math.t_mathematics.vxor_math_integration import get_vxor_math_integration
    vxor_math = get_vxor_math_integration()
    has_vxor = True
    logger.info("VXOR-Integration für MLXBackend geladen")
except ImportError:
    has_vxor = False
    vxor_math = None
    logger.warning("VXOR-Integration für MLXBackend nicht verfügbar")


class OptimizedMLXBackend:
    """
    Optimiertes Backend für MLX-Operationen, speziell für Apple Silicon.
    
    Diese Klasse bietet optimierte mathematische Operationen mit verbesserter
    Speicherverwaltung, direkten Geräteübertragungen und effizientem Caching.
    """
    
    def __init__(self, precision="float16", enable_tensor_pool=True):
        """
        Initialisiert das optimierte MLX-Backend.
        
        Args:
            precision: Präzision für MLX-Operationen (float16 oder float32)
            enable_tensor_pool: Ob der Tensor-Pool aktiviert werden soll
        """
        self.precision = precision
        self.mlx_available = False
        self.mx = None
        self.jit_cache = {}
        self.operation_cache = {}
        self.has_vxor = has_vxor
        
        # Optimale Geräte-Konfiguration für PyTorch
        if torch.backends.mps.is_available():
            self.torch_device = torch.device("mps")
        else:
            self.torch_device = torch.device("cpu")
        
        # Prüfe, ob MLX verfügbar ist
        try:
            import mlx.core as mx
            self.mx = mx
            self.mlx_available = True
            
            # Prüfe MLX-Version und verfügbare Funktionen
            self.has_jit = hasattr(mx, 'jit')
            
            logger.info(f"MLX-Backend initialisiert mit Präzision {precision} (JIT: {self.has_jit})")
        except ImportError:
            self.mx = None
            self.mlx_available = False
            self.has_jit = False
            logger.warning("MLX konnte nicht importiert werden. Verwende NumPy als Fallback.")
        
        # MLX-Optimierungen aktivieren
        if self.mlx_available:
            # Setze MLX-Datentyp basierend auf der Präzision
            self.dtype = mx.float16 if precision == "float16" else mx.float32
            
            # Aktiviere MLX-Compiler für bessere Performance
            try:
                # Aktiviere Just-In-Time Compilation für MLX-Operationen
                mx.set_default_device(mx.gpu if IS_APPLE_SILICON else mx.cpu)
                logger.info(f"MLX-Backend initialisiert mit Präzision {precision} und JIT-Optimierung")
            except Exception as e:
                logger.warning(f"MLX-JIT-Optimierung konnte nicht aktiviert werden: {e}")
                logger.info(f"MLX-Backend initialisiert mit Präzision {precision}")
        else:
            self.dtype = None
            logger.warning("MLX-Backend nicht verfügbar, verwende Fallback-Implementierung")
        
        # Initialisiere DirectMemoryTransfer
        self.memory_transfer = DirectMemoryTransfer(precision)
        
        # Initialisiere TensorPool, wenn aktiviert
        self.tensor_pool = TensorPool() if enable_tensor_pool else None
        self.enable_tensor_pool = enable_tensor_pool
    
    def to_mlx(self, tensor):
        """
        Konvertiert einen Tensor zu einem MLX-Array mit optimiertem Datenpfad.
        
        Diese optimierte Version vermeidet unnötige CPU-Zwischenschritte und
        nutzt, wo möglich, direkte Speichertransfers.
        
        Args:
            tensor: Eingabetensor (PyTorch-Tensor, NumPy-Array, etc.)
            
        Returns:
            MLX-Array
        """
        if not self.mlx_available:
            return tensor
        
        # Bereits ein MLX-Array?
        if hasattr(tensor, "__module__") and "mlx" in tensor.__module__:
            return tensor
        
        # PyTorch-Tensor
        if isinstance(tensor, torch.Tensor):
            # MPS-Tensor direkt zu MLX konvertieren
            if tensor.device.type == 'mps':
                try:
                    # Optimierter direkter Transfer ohne CPU-Umweg
                    return self.memory_transfer.mps_to_mlx_direct(tensor)
                except Exception as e:
                    logger.warning(f"Direkter MPS→MLX-Transfer fehlgeschlagen: {e}, verwende Fallback")
                    # Fallback zur CPU-Route
                    numpy_tensor = tensor.detach().cpu().numpy()
                    return mx.array(numpy_tensor, dtype=self.dtype)
            # CPU oder anderes Gerät
            else:
                numpy_tensor = tensor.detach().cpu().numpy()
                return mx.array(numpy_tensor, dtype=self.dtype)
        
        # NumPy-Array
        elif isinstance(tensor, np.ndarray):
            return mx.array(tensor, dtype=self.dtype)
        
        # Listen und andere Formate
        else:
            try:
                return mx.array(np.array(tensor), dtype=self.dtype)
            except Exception as e:
                logger.error(f"Konvertierung zu MLX fehlgeschlagen: {e}")
                raise TypeError(f"Unbekannter Tensor-Typ: {type(tensor)}")
    
    def from_mlx(self, array, device=None):
        """
        Konvertiert ein MLX-Array zu einem PyTorch-Tensor mit optimiertem Datenpfad.
        
        Args:
            array: MLX-Array
            device: Zielgerät für den PyTorch-Tensor (None für Standardgerät)
            
        Returns:
            PyTorch-Tensor
        """
        if not self.mlx_available or not isinstance(array, mx.array):
            return array
        
        # Bestimme Zielgerät
        target_device = device or self.torch_device
        
        # Optimierter Pfad für MPS
        if target_device.type == 'mps':
            try:
                # Direkter Transfer zu MPS
                return self.memory_transfer.mlx_to_mps_direct(array)
            except Exception as e:
                logger.warning(f"Direkter MLX→MPS-Transfer fehlgeschlagen: {e}, verwende Fallback")
                # Fallback via NumPy
                numpy_array = array.numpy()
                return torch.from_numpy(numpy_array).to(device=target_device)
        
        # Standard-Pfad für CPU
        try:
            numpy_array = array.numpy()
            return torch.from_numpy(numpy_array).to(device=target_device)
        except Exception as e:
            logger.warning(f"Standard MLX→PyTorch-Konvertierung fehlgeschlagen: {e}")
            # Fallback über Listen
            return torch.tensor(array.tolist(), device=target_device)
    
    def allocate_tensor(self, shape, dtype=None, device=None):
        """
        Alloziert einen Tensor aus dem Pool oder erstellt einen neuen.
        
        Args:
            shape: Tensorform
            dtype: Datentyp
            device: Gerät
            
        Returns:
            Tensor mit den angegebenen Eigenschaften
        """
        if self.enable_tensor_pool and self.tensor_pool is not None:
            return self.tensor_pool.get(shape, dtype, device)
        
        # Kein Pool, erstelle neuen Tensor
        if device is not None and hasattr(device, 'type') and device.type == 'mps':
            return torch.zeros(shape, dtype=dtype, device=torch.device('mps'))
        elif device == 'mlx' and self.mlx_available:
            mlx_dtype = self._convert_dtype_to_mlx(dtype)
            return mx.zeros(shape, dtype=mlx_dtype)
        else:
            return torch.zeros(shape, dtype=dtype, device=torch.device('cpu'))
    
    def release_tensor(self, tensor):
        """
        Gibt einen Tensor zurück in den Pool.
        
        Args:
            tensor: Der zurückzugebende Tensor
        """
        if self.enable_tensor_pool and self.tensor_pool is not None:
            self.tensor_pool.put(tensor)
    
    def _convert_dtype_to_mlx(self, dtype):
        """
        Konvertiert einen Datentyp zu einem MLX-Datentyp.
        
        Args:
            dtype: Eingabedatentyp
            
        Returns:
            MLX-Datentyp
        """
        if not self.mlx_available:
            return None
        
        dtype_str = str(dtype).split('.')[-1] if dtype is not None else 'float32'
        
        return {
            'float16': mx.float16,
            'float32': mx.float32,
            'bfloat16': mx.bfloat16 if hasattr(mx, 'bfloat16') else mx.float16,
            'int32': mx.int32,
            'int64': mx.int32,  # MLX hat kein int64, fallback zu int32
            'bool': mx.bool_,
        }.get(dtype_str, mx.float32)
    
    def get_memory_stats(self):
        """
        Liefert Statistiken über die Speichernutzung.
        
        Returns:
            Dictionary mit Speicherstatistiken
        """
        stats = {}
        
        # TensorPool-Statistik
        if self.enable_tensor_pool and self.tensor_pool is not None:
            stats['tensor_pool'] = self.tensor_pool.get_stats()
        
        # Speichertransfer-Statistik
        stats['memory_transfer'] = self.memory_transfer.get_transfer_stats()
        
        # JIT-Cache-Statistik
        stats['jit_cache_size'] = len(self.jit_cache)
        stats['operation_cache_size'] = len(self.operation_cache)
        
        return stats
