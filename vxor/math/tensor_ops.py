#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Tensor Operations Module

Dieses Modul stellt Wrapper für verschiedene Tensor-Backends (MLX, PyTorch, NumPy)
zur Verfügung und ermöglicht eine optimierte Interoperabilität zwischen ihnen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Type

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.tensor_ops")

# Prüfen, ob Apple Silicon verfügbar ist
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine

# Importiere PyTorch
try:
    import torch
    HAS_TORCH = True
    if is_apple_silicon:
        # MPS-Backend für Apple Silicon
        TORCH_DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        TORCH_DEVICE = torch.device("cuda")
    else:
        TORCH_DEVICE = torch.device("cpu")
except ImportError:
    HAS_TORCH = False
    TORCH_DEVICE = None
    logger.warning("PyTorch konnte nicht importiert werden.")

# Importiere MLX für Apple Silicon
HAS_MLX = False
if is_apple_silicon:
    try:
        import mlx.core as mx
        HAS_MLX = True
        logger.info("MLX für Apple Silicon erfolgreich importiert.")
    except ImportError:
        HAS_MLX = False
        logger.warning("MLX konnte nicht importiert werden, obwohl Apple Silicon erkannt wurde.")

# Basis-Wrapper-Klasse für alle Tensor-Typen
class MISOTensor:
    """Basis-Wrapper-Klasse für verschiedene Tensor-Implementierungen"""
    
    def __init__(self, data: Any, dtype=None):
        """
        Initialisiert den Tensor-Wrapper.
        
        Args:
            data: Eingabedaten (NumPy-Array, PyTorch-Tensor, Liste etc.)
            dtype: Optionaler Datentyp
        """
        self.data = data
        self.dtype = dtype
        self._shape = None
    
    @property
    def shape(self) -> Tuple:
        """Gibt die Form des Tensors zurück"""
        if self._shape is None:
            if hasattr(self.data, 'shape'):
                self._shape = self.data.shape
            else:
                # Fallback für Listen etc.
                self._shape = np.array(self.data).shape
        return self._shape
    
    def numpy(self) -> np.ndarray:
        """Konvertiert zu NumPy-Array"""
        if isinstance(self.data, np.ndarray):
            return self.data
        else:
            return np.array(self.data)
    
    def __str__(self) -> str:
        """String-Darstellung des Tensors"""
        return f"MISOTensor(shape={self.shape}, dtype={self.dtype})"
    
    def __repr__(self) -> str:
        """Detaillierte String-Darstellung des Tensors"""
        return self.__str__()


class TorchTensor(MISOTensor):
    """Wrapper für PyTorch-Tensoren"""
    
    def __init__(self, data: Any, dtype=None, device=None):
        """
        Initialisiert den PyTorch-Tensor-Wrapper.
        
        Args:
            data: Eingabedaten (NumPy-Array, PyTorch-Tensor, Liste etc.)
            dtype: Optionaler PyTorch-Datentyp
            device: Optionales PyTorch-Gerät
        """
        super().__init__(data, dtype)
        
        # Standardwerte für Gerät und Datentyp
        self.device = device or TORCH_DEVICE or torch.device("cpu")
        
        # Konvertiere zu PyTorch, falls nötig
        if not isinstance(data, torch.Tensor):
            if isinstance(data, np.ndarray):
                self.data = torch.from_numpy(data)
            else:
                self.data = torch.tensor(data)
        else:
            self.data = data
        
        # Verschiebe auf richtiges Gerät und konvertiere Datentyp, falls nötig
        if dtype is not None:
            self.data = self.data.to(device=self.device, dtype=dtype)
        else:
            self.data = self.data.to(device=self.device)
    
    def numpy(self) -> np.ndarray:
        """Konvertiert zu NumPy-Array"""
        # Hole von GPU/MPS zurück zur CPU, falls nötig
        if self.data.device.type != "cpu":
            cpu_tensor = self.data.detach().cpu()
        else:
            cpu_tensor = self.data.detach()
        
        return cpu_tensor.numpy()
    
    def __str__(self) -> str:
        """String-Darstellung des PyTorch-Tensors"""
        return f"TorchTensor(shape={tuple(self.data.shape)}, dtype={self.data.dtype}, device={self.data.device})"


class MLXTensor(MISOTensor):
    """Wrapper für MLX-Arrays (optimiert für Apple Silicon)"""
    
    def __init__(self, data: Any, dtype=None):
        """
        Initialisiert den MLX-Tensor-Wrapper.
        
        Args:
            data: Eingabedaten (NumPy-Array, PyTorch-Tensor, MLX-Array, Liste etc.)
            dtype: Optionaler MLX-Datentyp
        """
        super().__init__(data, dtype)
        
        if not HAS_MLX:
            raise ImportError("MLX ist erforderlich für MLXTensor, aber nicht verfügbar.")
        
        # Konvertiere zu MLX, falls nötig
        if not str(type(data)).find('mlx') >= 0:
            if isinstance(data, np.ndarray):
                self.data = mx.array(data, dtype=dtype)
            elif isinstance(data, torch.Tensor):
                # PyTorch zu NumPy zu MLX
                numpy_data = data.detach().cpu().numpy()
                self.data = mx.array(numpy_data, dtype=dtype)
            else:
                # Listen etc. zu MLX
                self.data = mx.array(data, dtype=dtype)
        else:
            # Bereits MLX
            if dtype is not None and hasattr(self.data, 'astype'):
                self.data = self.data.astype(dtype)
            else:
                self.data = data
    
    def numpy(self) -> np.ndarray:
        """Konvertiert zu NumPy-Array"""
        # MLX-Arrays haben keine direkte numpy()-Methode
        try:
            return np.array(self.data.tolist())
        except:
            # Fallback
            return np.array(self.data)
    
    def __str__(self) -> str:
        """String-Darstellung des MLX-Arrays"""
        return f"MLXTensor(shape={self.data.shape}, dtype={self.data.dtype})"


# Hilfsfunktionen

def detect_tensor_type(tensor: Any) -> str:
    """
    Erkennt den Typ eines Tensors.
    
    Args:
        tensor: Tensor oder Array
        
    Returns:
        String-Bezeichnung des Typs: "torch", "mlx", "numpy" oder "other"
    """
    if isinstance(tensor, torch.Tensor):
        return "torch"
    elif str(type(tensor)).find('mlx') >= 0:
        return "mlx"
    elif isinstance(tensor, np.ndarray):
        return "numpy"
    else:
        return "other"


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    """
    Konvertiert einen beliebigen Tensortyp zu NumPy.
    
    Args:
        tensor: Eingangs-Tensor (beliebiger Typ)
        
    Returns:
        NumPy-Array
    """
    tensor_type = detect_tensor_type(tensor)
    
    if tensor_type == "torch":
        if tensor.device.type != "cpu":
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()
    elif tensor_type == "mlx":
        try:
            return np.array(tensor.tolist())
        except:
            return np.array(tensor)
    elif tensor_type == "numpy":
        return tensor
    else:
        return np.array(tensor)


def numpy_to_tensor(array: np.ndarray, target_type: str, **kwargs) -> Any:
    """
    Konvertiert ein NumPy-Array zu einem spezifischen Tensortyp.
    
    Args:
        array: NumPy-Array
        target_type: Zieltyp ("torch", "mlx", "numpy")
        **kwargs: Zusätzliche Parameter für die Tensorerstellung
        
    Returns:
        Tensor des gewünschten Typs
    """
    if target_type == "torch":
        if not HAS_TORCH:
            raise ImportError("PyTorch ist erforderlich, aber nicht verfügbar.")
        return TorchTensor(array, **kwargs).data
    elif target_type == "mlx":
        if not HAS_MLX:
            raise ImportError("MLX ist erforderlich, aber nicht verfügbar.")
        return MLXTensor(array, **kwargs).data
    elif target_type == "numpy":
        return array
    else:
        raise ValueError(f"Unbekannter Zieltyp: {target_type}")


def convert_tensor(tensor: Any, target_type: str, **kwargs) -> Any:
    """
    Konvertiert einen Tensor von einem Typ zu einem anderen.
    
    Args:
        tensor: Eingangs-Tensor (beliebiger Typ)
        target_type: Zieltyp ("torch", "mlx", "numpy")
        **kwargs: Zusätzliche Parameter für die Tensorerstellung
        
    Returns:
        Tensor des gewünschten Typs
    """
    # Wenn der Eingangstyp bereits dem Zieltyp entspricht, gib ihn unverändert zurück
    current_type = detect_tensor_type(tensor)
    if current_type == target_type:
        return tensor
    
    # Konvertiere erst zu NumPy als gemeinsames Zwischenformat
    numpy_tensor = tensor_to_numpy(tensor)
    
    # Dann konvertiere zu Zielformat
    return numpy_to_tensor(numpy_tensor, target_type, **kwargs)


# Exportiere wichtige Klassen und Funktionen
__all__ = [
    'MISOTensor', 'TorchTensor', 'MLXTensor',
    'detect_tensor_type', 'tensor_to_numpy', 'numpy_to_tensor', 'convert_tensor',
    'HAS_TORCH', 'HAS_MLX', 'TORCH_DEVICE', 'is_apple_silicon'
]
