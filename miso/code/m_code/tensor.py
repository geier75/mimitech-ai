#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Tensor

Dieses Modul implementiert die Tensor-Klasse für M-CODE.
Die Tensor-Klasse nutzt den MLX-Adapter, um optimierte Operationen auf Apple Silicon durchzuführen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import sys
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set, Sequence
from pathlib import Path

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.tensor")

# Import von internen Modulen
from .mlx_adapter import get_mlx_adapter, MLXAdapter, MLX_AVAILABLE


class MTensorError(Exception):
    """Fehler in der M-CODE Tensor-Klasse"""
    pass


class MTensor:
    """Optimierte Tensor-Klasse für M-CODE"""
    
    def __init__(self, data, dtype=None, device=None, name=None):
        """
        Initialisiert einen neuen M-CODE Tensor.
        
        Args:
            data: Daten für den Tensor (NumPy-Array, Liste, etc.)
            dtype: Datentyp (optional)
            device: Gerät (cpu, gpu, auto)
            name: Optionaler Name für Debugging und Profiling
        """
        self.name = name
        self.device = device
        self._adapter = get_mlx_adapter()
        
        # Erstelle Tensor mit MLX-Adapter
        try:
            self._tensor = self._adapter.create_tensor(data, dtype)
        except Exception as e:
            logger.error(f"Fehler bei der Erstellung des Tensors: {e}")
            # Fallback zu NumPy
            self._tensor = np.array(data, dtype=dtype)
        
        # Speichere Shape
        self._update_shape()
    
    def _update_shape(self):
        """Aktualisiert die Shape-Information des Tensors"""
        if hasattr(self._tensor, 'shape'):
            self.shape = self._tensor.shape
        else:
            self.shape = np.array(self._tensor).shape
    
    def __repr__(self) -> str:
        """Liefert eine String-Darstellung des Tensors"""
        name_info = f" '{self.name}'" if self.name else ""
        shape_info = f"{self.shape}"
        device_info = f", device={self.device}" if self.device else ""
        
        return f"MTensor{name_info}(shape={shape_info}{device_info})"
    
    def __str__(self) -> str:
        """Liefert eine lesbare String-Darstellung des Tensors"""
        try:
            return f"MTensor(shape={self.shape}, values=\n{self.to_numpy()})"
        except Exception:
            return self.__repr__()
    
    def to_numpy(self) -> np.ndarray:
        """
        Konvertiert den Tensor zu einem NumPy-Array.
        
        Returns:
            NumPy-Array
        """
        return self._adapter.to_numpy(self._tensor)
    
    def _ensure_tensor(self, other) -> Any:
        """Stellt sicher, dass ein Wert ein Tensor ist"""
        if isinstance(other, MTensor):
            return other._tensor
        return other
    
    def __add__(self, other) -> 'MTensor':
        """Addition"""
        other_tensor = self._ensure_tensor(other)
        result = self._adapter.execute_tensor_operation("add", self._tensor, other_tensor)
        return MTensor(result)
    
    def __sub__(self, other) -> 'MTensor':
        """Subtraktion"""
        other_tensor = self._ensure_tensor(other)
        result = self._adapter.execute_tensor_operation("subtract", self._tensor, other_tensor)
        return MTensor(result)
    
    def __mul__(self, other) -> 'MTensor':
        """Multiplikation (elementweise)"""
        other_tensor = self._ensure_tensor(other)
        result = self._adapter.execute_tensor_operation("multiply", self._tensor, other_tensor)
        return MTensor(result)
    
    def __truediv__(self, other) -> 'MTensor':
        """Division"""
        other_tensor = self._ensure_tensor(other)
        result = self._adapter.execute_tensor_operation("divide", self._tensor, other_tensor)
        return MTensor(result)
    
    def __matmul__(self, other) -> 'MTensor':
        """Matrix-Multiplikation"""
        other_tensor = self._ensure_tensor(other)
        result = self._adapter.execute_tensor_operation("matmul", self._tensor, other_tensor)
        return MTensor(result)
    
    def __pow__(self, power) -> 'MTensor':
        """Potenzierung"""
        result = self._adapter.execute_tensor_operation("pow", self._tensor, power)
        return MTensor(result)
    
    def transpose(self, axes=None) -> 'MTensor':
        """
        Transponiert den Tensor.
        
        Args:
            axes: Achsen für die Transposition (optional)
            
        Returns:
            Transponierter Tensor
        """
        result = self._adapter.execute_tensor_operation("transpose", self._tensor, axes)
        return MTensor(result)
    
    def reshape(self, *shape) -> 'MTensor':
        """
        Ändert die Form des Tensors.
        
        Args:
            shape: Neue Form
            
        Returns:
            Tensor mit neuer Form
        """
        # Behandle den Fall, dass shape als Tupel oder Liste übergeben wird
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
            
        result = self._adapter.execute_tensor_operation("reshape", self._tensor, shape)
        return MTensor(result)
    
    def sum(self, axis=None, keepdims=False) -> 'MTensor':
        """
        Berechnet die Summe entlang einer Achse.
        
        Args:
            axis: Achse (optional)
            keepdims: Dimensionen beibehalten (optional)
            
        Returns:
            Summe
        """
        result = self._adapter.execute_tensor_operation("sum", self._tensor, axis, keepdims=keepdims)
        return MTensor(result)
    
    def mean(self, axis=None, keepdims=False) -> 'MTensor':
        """
        Berechnet den Mittelwert entlang einer Achse.
        
        Args:
            axis: Achse (optional)
            keepdims: Dimensionen beibehalten (optional)
            
        Returns:
            Mittelwert
        """
        result = self._adapter.execute_tensor_operation("mean", self._tensor, axis, keepdims=keepdims)
        return MTensor(result)
    
    def std(self, axis=None, keepdims=False) -> 'MTensor':
        """
        Berechnet die Standardabweichung entlang einer Achse.
        
        Args:
            axis: Achse (optional)
            keepdims: Dimensionen beibehalten (optional)
            
        Returns:
            Standardabweichung
        """
        result = self._adapter.execute_tensor_operation("std", self._tensor, axis, keepdims=keepdims)
        return MTensor(result)
    
    def max(self, axis=None, keepdims=False) -> 'MTensor':
        """
        Berechnet das Maximum entlang einer Achse.
        
        Args:
            axis: Achse (optional)
            keepdims: Dimensionen beibehalten (optional)
            
        Returns:
            Maximum
        """
        result = self._adapter.execute_tensor_operation("max", self._tensor, axis, keepdims=keepdims)
        return MTensor(result)
    
    def min(self, axis=None, keepdims=False) -> 'MTensor':
        """
        Berechnet das Minimum entlang einer Achse.
        
        Args:
            axis: Achse (optional)
            keepdims: Dimensionen beibehalten (optional)
            
        Returns:
            Minimum
        """
        result = self._adapter.execute_tensor_operation("min", self._tensor, axis, keepdims=keepdims)
        return MTensor(result)
    
    @classmethod
    def zeros(cls, shape, dtype=None, device=None, name=None) -> 'MTensor':
        """
        Erstellt einen Tensor mit Nullen.
        
        Args:
            shape: Form des Tensors
            dtype: Datentyp (optional)
            device: Gerät (optional)
            name: Name (optional)
            
        Returns:
            Tensor mit Nullen
        """
        adapter = get_mlx_adapter()
        data = adapter.execute_tensor_operation("zeros", shape, dtype=dtype)
        return cls(data, dtype=dtype, device=device, name=name)
    
    @classmethod
    def ones(cls, shape, dtype=None, device=None, name=None) -> 'MTensor':
        """
        Erstellt einen Tensor mit Einsen.
        
        Args:
            shape: Form des Tensors
            dtype: Datentyp (optional)
            device: Gerät (optional)
            name: Name (optional)
            
        Returns:
            Tensor mit Einsen
        """
        adapter = get_mlx_adapter()
        data = adapter.execute_tensor_operation("ones", shape, dtype=dtype)
        return cls(data, dtype=dtype, device=device, name=name)
    
    @classmethod
    def eye(cls, n, m=None, dtype=None, device=None, name=None) -> 'MTensor':
        """
        Erstellt eine Einheitsmatrix.
        
        Args:
            n: Anzahl der Zeilen
            m: Anzahl der Spalten (optional, Standard: n)
            dtype: Datentyp (optional)
            device: Gerät (optional)
            name: Name (optional)
            
        Returns:
            Einheitsmatrix
        """
        m = n if m is None else m
        adapter = get_mlx_adapter()
        data = adapter.execute_tensor_operation("eye", n, m, dtype=dtype)
        return cls(data, dtype=dtype, device=device, name=name)
    
    @classmethod
    def random(cls, shape, dtype=None, device=None, name=None) -> 'MTensor':
        """
        Erstellt einen Tensor mit Zufallswerten.
        
        Args:
            shape: Form des Tensors
            dtype: Datentyp (optional)
            device: Gerät (optional)
            name: Name (optional)
            
        Returns:
            Tensor mit Zufallswerten
        """
        adapter = get_mlx_adapter()
        data = adapter.execute_tensor_operation("random", shape, dtype=dtype)
        return cls(data, dtype=dtype, device=device, name=name)
    
    def jit_compile(self, func: Callable) -> Callable:
        """
        Kompiliert eine Funktion mit JIT für den Tensor.
        
        Args:
            func: Zu kompilierende Funktion
            
        Returns:
            Kompilierte Funktion
        """
        return self._adapter.jit_compile(func)


# Factory-Funktionen
def tensor(data, dtype=None, device=None, name=None) -> MTensor:
    """
    Erstellt einen M-CODE Tensor.
    
    Args:
        data: Daten für den Tensor
        dtype: Datentyp (optional)
        device: Gerät (optional)
        name: Name (optional)
        
    Returns:
        M-CODE Tensor
    """
    return MTensor(data, dtype=dtype, device=device, name=name)


def zeros(shape, dtype=None, device=None, name=None) -> MTensor:
    """
    Erstellt einen Tensor mit Nullen.
    
    Args:
        shape: Form des Tensors
        dtype: Datentyp (optional)
        device: Gerät (optional)
        name: Name (optional)
        
    Returns:
        Tensor mit Nullen
    """
    return MTensor.zeros(shape, dtype=dtype, device=device, name=name)


def ones(shape, dtype=None, device=None, name=None) -> MTensor:
    """
    Erstellt einen Tensor mit Einsen.
    
    Args:
        shape: Form des Tensors
        dtype: Datentyp (optional)
        device: Gerät (optional)
        name: Name (optional)
        
    Returns:
        Tensor mit Einsen
    """
    return MTensor.ones(shape, dtype=dtype, device=device, name=name)


def eye(n, m=None, dtype=None, device=None, name=None) -> MTensor:
    """
    Erstellt eine Einheitsmatrix.
    
    Args:
        n: Anzahl der Zeilen
        m: Anzahl der Spalten (optional, Standard: n)
        dtype: Datentyp (optional)
        device: Gerät (optional)
        name: Name (optional)
        
    Returns:
        Einheitsmatrix
    """
    return MTensor.eye(n, m, dtype=dtype, device=device, name=name)


def random(shape, dtype=None, device=None, name=None) -> MTensor:
    """
    Erstellt einen Tensor mit Zufallswerten.
    
    Args:
        shape: Form des Tensors
        dtype: Datentyp (optional)
        device: Gerät (optional)
        name: Name (optional)
        
    Returns:
        Tensor mit Zufallswerten
    """
    return MTensor.random(shape, dtype=dtype, device=device, name=name)


def jit(func: Callable) -> Callable:
    """
    JIT-Dekorator für Funktionen.
    
    Args:
        func: Zu dekorierende Funktion
        
    Returns:
        Dekorierte Funktion
    """
    adapter = get_mlx_adapter()
    return adapter.jit_compile(func)
