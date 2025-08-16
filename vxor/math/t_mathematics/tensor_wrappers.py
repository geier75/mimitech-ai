#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Engine - Tensor-Wrapper-Klassen

Diese Datei implementiert Wrapper-Klassen für verschiedene Tensor-Backends,
die eine einheitliche Schnittstelle bieten, unabhängig vom tatsächlichen Backend.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import base64
import json
import logging
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from .tensor_interface import MISOTensorInterface

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.t_mathematics.tensor_wrappers")

class MLXTensorWrapper(MISOTensorInterface):
    """
    Ein Wrapper für MLX-Arrays, der die MISOTensorInterface implementiert.
    
    Diese Klasse umhüllt ein MLX-Array und stellt eine standardisierte Schnittstelle
    für die Verwendung in VXOR-Komponenten bereit.
    """
    
    def __init__(self, mlx_array):
        """
        Initialisiert den Wrapper mit einem MLX-Array.
        
        Args:
            mlx_array: Das zu umhüllende MLX-Array
        """
        self._data = mlx_array
        self._backend = "mlx"
    
    @property
    def backend(self) -> str:
        """Gibt den Namen des Backend-Systems zurück."""
        return self._backend
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Gibt die Form des Tensors zurück."""
        return self._data.shape
    
    @property
    def dtype(self) -> str:
        """Gibt den Datentyp des Tensors zurück."""
        # MLX hat keine direkte dtype-Eigenschaft wie NumPy, verwenden wir den Typ aus den Metadaten
        return str(self._data.dtype)
    
    def __add__(self, other) -> 'MLXTensorWrapper':
        """Addition"""
        if isinstance(other, MLXTensorWrapper):
            result = self._data + other._data
        else:
            result = self._data + other
        return MLXTensorWrapper(result)
    
    def __matmul__(self, other) -> 'MLXTensorWrapper':
        """Matrix-Multiplikation"""
        if isinstance(other, MLXTensorWrapper):
            result = self._data @ other._data
        else:
            result = self._data @ other
        return MLXTensorWrapper(result)
    
    def to_numpy(self) -> np.ndarray:
        """Konvertiert zu NumPy-Array"""
        # MLX verwendet nicht die numpy()-Methode, sondern meist eigene Funktionen
        import mlx.core as mx
        
        # Versuche verschiedene Konvertierungsmethoden
        if hasattr(self._data, 'numpy'):
            return self._data.numpy()
        elif hasattr(mx, 'to_numpy'):
            return mx.to_numpy(self._data)
        elif hasattr(mx, 'asnumpy'):
            return mx.asnumpy(self._data)
        else:
            # Fallback: Konvertieren zu CPU und dann manuell zu numpy
            try:
                # Oft ist ein einfacher direkter Aufruf von np.array möglich
                return np.array(self._data)  
            except Exception as e:
                raise RuntimeError(f"Konnte MLX-Array nicht zu NumPy konvertieren: {e}")
    
    def exp(self) -> 'MLXTensorWrapper':
        """Exponentielle Funktion"""
        import mlx.core as mx
        return MLXTensorWrapper(mx.exp(self._data))
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialisiert den Tensor für die Übertragung zwischen Komponenten.
        
        Returns:
            Dictionary mit serialisierten Daten des Tensors
        """
        # Konvertiere zu NumPy und dann zu Base64-String für JSON-Serialisierbarkeit
        numpy_data = self.to_numpy()
        encoded = base64.b64encode(numpy_data.tobytes()).decode('ascii')
        
        return {
            "backend": self.backend,
            "shape": self.shape,
            "dtype": str(numpy_data.dtype),
            "data": encoded
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'MLXTensorWrapper':
        """
        Deserialisiert einen Tensor aus einem Dictionary.
        
        Args:
            data: Dictionary mit serialisierten Daten des Tensors
            
        Returns:
            Neuer MLXTensorWrapper aus den deserialisierten Daten
        """
        if data["backend"] != "mlx":
            logger.warning(f"Tensor-Backend-Konvertierung: {data['backend']} -> mlx")
        
        # Dekodiere Base64-String zu NumPy-Array
        binary_data = base64.b64decode(data["data"])
        dtype = np.dtype(data["dtype"])
        shape = tuple(data["shape"])
        numpy_data = np.frombuffer(binary_data, dtype=dtype).reshape(shape)
        
        # Konvertiere NumPy-Array zu MLX-Array
        import mlx.core as mx
        mlx_array = mx.array(numpy_data)
        
        return cls(mlx_array)
    
    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'MLXTensorWrapper':
        """
        Erstellt einen MLXTensorWrapper aus einem NumPy-Array.
        
        Args:
            array: NumPy-Array, aus dem der Tensor erstellt werden soll
            
        Returns:
            Neuer MLXTensorWrapper mit den Daten des NumPy-Arrays
        """
        import mlx.core as mx
        return cls(mx.array(array))


class TorchTensorWrapper(MISOTensorInterface):
    """
    Ein Wrapper für PyTorch-Tensoren, der die MISOTensorInterface implementiert.
    
    Diese Klasse umhüllt einen PyTorch-Tensor und stellt eine standardisierte Schnittstelle
    für die Verwendung in VXOR-Komponenten bereit.
    """
    
    def __init__(self, torch_tensor):
        """
        Initialisiert den Wrapper mit einem PyTorch-Tensor.
        
        Args:
            torch_tensor: Der zu umhüllende PyTorch-Tensor
        """
        self._data = torch_tensor
        self._backend = "torch"
    
    @property
    def backend(self) -> str:
        """Gibt den Namen des Backend-Systems zurück."""
        return self._backend
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Gibt die Form des Tensors zurück."""
        return self._data.shape
    
    @property
    def dtype(self) -> str:
        """Gibt den Datentyp des Tensors zurück."""
        return str(self._data.dtype)
    
    def __add__(self, other) -> 'TorchTensorWrapper':
        """Addition"""
        if isinstance(other, TorchTensorWrapper):
            result = self._data + other._data
        else:
            result = self._data + other
        return TorchTensorWrapper(result)
    
    def __matmul__(self, other) -> 'TorchTensorWrapper':
        """Matrix-Multiplikation"""
        if isinstance(other, TorchTensorWrapper):
            result = torch.matmul(self._data, other._data)
        else:
            result = torch.matmul(self._data, other)
        return TorchTensorWrapper(result)
    
    def to_numpy(self) -> np.ndarray:
        """Konvertiert zu NumPy-Array"""
        return self._data.cpu().detach().numpy()
    
    def exp(self) -> 'TorchTensorWrapper':
        """Exponentielle Funktion"""
        return TorchTensorWrapper(torch.exp(self._data))
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialisiert den Tensor für die Übertragung zwischen Komponenten.
        
        Returns:
            Dictionary mit serialisierten Daten des Tensors
        """
        # Konvertiere zu NumPy und dann zu Base64-String für JSON-Serialisierbarkeit
        numpy_data = self.to_numpy()
        encoded = base64.b64encode(numpy_data.tobytes()).decode('ascii')
        
        return {
            "backend": self.backend,
            "shape": self.shape,
            "dtype": str(numpy_data.dtype),
            "data": encoded,
            "device": str(self._data.device)
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'TorchTensorWrapper':
        """
        Deserialisiert einen Tensor aus einem Dictionary.
        
        Args:
            data: Dictionary mit serialisierten Daten des Tensors
            
        Returns:
            Neuer TorchTensorWrapper aus den deserialisierten Daten
        """
        if data["backend"] != "torch":
            logger.warning(f"Tensor-Backend-Konvertierung: {data['backend']} -> torch")
        
        # Dekodiere Base64-String zu NumPy-Array
        binary_data = base64.b64decode(data["data"])
        dtype = np.dtype(data["dtype"])
        shape = tuple(data["shape"])
        numpy_data = np.frombuffer(binary_data, dtype=dtype).reshape(shape)
        
        # Konvertiere NumPy-Array zu PyTorch-Tensor
        device = data.get("device", "cpu")
        if "cuda" in device and torch.cuda.is_available():
            torch_device = torch.device(device)
        elif "mps" in device and hasattr(torch, "mps") and torch.mps.is_available():
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")
            
        torch_tensor = torch.tensor(numpy_data, device=torch_device)
        
        return cls(torch_tensor)
    
    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'TorchTensorWrapper':
        """
        Erstellt einen TorchTensorWrapper aus einem NumPy-Array.
        
        Args:
            array: NumPy-Array, aus dem der Tensor erstellt werden soll
            
        Returns:
            Neuer TorchTensorWrapper mit den Daten des NumPy-Arrays
        """
        # Wähle das beste verfügbare Gerät
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "mps") and torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        torch_tensor = torch.tensor(array, device=device)
        return cls(torch_tensor)
