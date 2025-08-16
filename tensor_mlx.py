#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - MLX Tensor-Implementierung

Dieses Modul implementiert die MLX-spezifische Tensor-Klasse für die T-Mathematics Engine,
optimiert für Apple Silicon mit Neural Engine-Unterstützung.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.Ultimate.TMathematics.MLXTensor")

# Importiere MLX
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    logger.warning("Apple MLX nicht verfügbar, verwende Dummy-Implementierung")
    HAS_MLX = False
    
    # Dummy-Klasse für den Fall, dass MLX nicht verfügbar ist
    class mx:
        @staticmethod
        def array(*args, **kwargs):
            return np.array(*args, **kwargs)
        
        @staticmethod
        def asarray(*args, **kwargs):
            return np.asarray(*args, **kwargs)

# Importiere Basisklasse
from .tensor import MISOTensor


class MLXTensor(MISOTensor):
    """
    MLX-spezifische Tensor-Implementierung für die T-Mathematics Engine
    
    Diese Klasse implementiert die Tensor-Operationen mit dem Apple MLX Framework,
    das speziell für Apple Silicon optimiert ist.
    """
    
    def __init__(self, data, dtype=None):
        """
        Initialisiert einen MLX-Tensor
        
        Args:
            data: Daten für den Tensor (kann ein NumPy-Array, eine Liste oder ein MLX-Array sein)
            dtype: Datentyp für den Tensor
        """
        super().__init__(data, dtype)
        self._backend = "mlx"
        
        # Konvertiere Daten zu MLX-Array
        if isinstance(data, mx.array):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = mx.array(data, dtype=dtype)
        elif isinstance(data, (list, tuple)):
            self._data = mx.array(data, dtype=dtype)
        elif isinstance(data, (int, float, bool)):
            self._data = mx.array([data], dtype=dtype)
        elif isinstance(data, MISOTensor):
            if isinstance(data, MLXTensor):
                self._data = data.data
            else:
                self._data = mx.array(data.to_numpy(), dtype=dtype)
        else:
            raise TypeError(f"Datentyp {type(data)} wird nicht unterstützt")
        
        # Setze Datentyp
        if dtype is not None:
            self._data = self._data.astype(dtype)
        self._dtype = str(self._data.dtype)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Gibt die Form des Tensors zurück"""
        return self._data.shape
    
    def to_numpy(self) -> np.ndarray:
        """
        Konvertiert den Tensor zu einem NumPy-Array
        
        Returns:
            NumPy-Array
        """
        return np.array(self._data)
    
    def clone(self) -> 'MLXTensor':
        """
        Erstellt eine Kopie des Tensors
        
        Returns:
            Kopie des Tensors
        """
        return MLXTensor(self._data.copy(), self._dtype)
    
    def reshape(self, *shape) -> 'MLXTensor':
        """
        Ändert die Form des Tensors
        
        Args:
            shape: Neue Form
            
        Returns:
            Tensor mit neuer Form
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return MLXTensor(self._data.reshape(shape), self._dtype)
    
    def transpose(self, *dims) -> 'MLXTensor':
        """
        Transponiert den Tensor
        
        Args:
            dims: Dimensionen, die transponiert werden sollen
            
        Returns:
            Transponierter Tensor
        """
        if not dims:
            # Wenn keine Dimensionen angegeben sind, transponiere alle Dimensionen
            dims = tuple(range(self.ndim))[::-1]
        elif len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]
        
        return MLXTensor(mx.transpose(self._data, dims), self._dtype)
    
    def squeeze(self, dim: Optional[int] = None) -> 'MLXTensor':
        """
        Entfernt Dimensionen mit Größe 1
        
        Args:
            dim: Dimension, die entfernt werden soll (optional)
            
        Returns:
            Tensor mit entfernten Dimensionen
        """
        if dim is None:
            return MLXTensor(mx.squeeze(self._data), self._dtype)
        else:
            return MLXTensor(mx.squeeze(self._data, axis=dim), self._dtype)
    
    def unsqueeze(self, dim: int) -> 'MLXTensor':
        """
        Fügt eine Dimension mit Größe 1 hinzu
        
        Args:
            dim: Position, an der die Dimension hinzugefügt werden soll
            
        Returns:
            Tensor mit hinzugefügter Dimension
        """
        return MLXTensor(mx.expand_dims(self._data, axis=dim), self._dtype)
    
    def __add__(self, other) -> 'MLXTensor':
        """
        Addition
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Addition
        """
        if isinstance(other, MLXTensor):
            return MLXTensor(self._data + other.data, self._dtype)
        else:
            return MLXTensor(self._data + other, self._dtype)
    
    def __sub__(self, other) -> 'MLXTensor':
        """
        Subtraktion
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Subtraktion
        """
        if isinstance(other, MLXTensor):
            return MLXTensor(self._data - other.data, self._dtype)
        else:
            return MLXTensor(self._data - other, self._dtype)
    
    def __rsub__(self, other) -> 'MLXTensor':
        """
        Rechte Subtraktion
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Subtraktion
        """
        if isinstance(other, MLXTensor):
            return MLXTensor(other.data - self._data, self._dtype)
        else:
            return MLXTensor(other - self._data, self._dtype)
    
    def __mul__(self, other) -> 'MLXTensor':
        """
        Multiplikation (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Multiplikation
        """
        if isinstance(other, MLXTensor):
            return MLXTensor(self._data * other.data, self._dtype)
        else:
            return MLXTensor(self._data * other, self._dtype)
    
    def __truediv__(self, other) -> 'MLXTensor':
        """
        Division (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Division
        """
        if isinstance(other, MLXTensor):
            return MLXTensor(self._data / other.data, self._dtype)
        else:
            return MLXTensor(self._data / other, self._dtype)
    
    def __rtruediv__(self, other) -> 'MLXTensor':
        """
        Rechte Division (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Division
        """
        if isinstance(other, MLXTensor):
            return MLXTensor(other.data / self._data, self._dtype)
        else:
            return MLXTensor(other / self._data, self._dtype)
    
    def __pow__(self, other) -> 'MLXTensor':
        """
        Potenzierung (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Potenzierung
        """
        if isinstance(other, MLXTensor):
            return MLXTensor(mx.power(self._data, other.data), self._dtype)
        else:
            return MLXTensor(mx.power(self._data, other), self._dtype)
    
    def __rpow__(self, other) -> 'MLXTensor':
        """
        Rechte Potenzierung (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Potenzierung
        """
        if isinstance(other, MLXTensor):
            return MLXTensor(mx.power(other.data, self._data), self._dtype)
        else:
            return MLXTensor(mx.power(other, self._data), self._dtype)
    
    def __matmul__(self, other) -> 'MLXTensor':
        """
        Matrix-Multiplikation
        
        Args:
            other: Tensor
            
        Returns:
            Ergebnis der Matrix-Multiplikation
        """
        if isinstance(other, MLXTensor):
            return MLXTensor(mx.matmul(self._data, other.data), self._dtype)
        else:
            return MLXTensor(mx.matmul(self._data, mx.array(other)), self._dtype)
    
    def __rmatmul__(self, other) -> 'MLXTensor':
        """
        Rechte Matrix-Multiplikation
        
        Args:
            other: Tensor
            
        Returns:
            Ergebnis der Matrix-Multiplikation
        """
        if isinstance(other, MLXTensor):
            return MLXTensor(mx.matmul(other.data, self._data), self._dtype)
        else:
            return MLXTensor(mx.matmul(mx.array(other), self._data), self._dtype)
    
    def __neg__(self) -> 'MLXTensor':
        """
        Negation
        
        Returns:
            Negierter Tensor
        """
        return MLXTensor(-self._data, self._dtype)
    
    def __abs__(self) -> 'MLXTensor':
        """
        Absolutwert (elementweise)
        
        Returns:
            Tensor mit Absolutwerten
        """
        return MLXTensor(mx.abs(self._data), self._dtype)
    
    def __getitem__(self, idx) -> 'MLXTensor':
        """
        Indexierung
        
        Args:
            idx: Index oder Slice
            
        Returns:
            Teilmenge des Tensors
        """
        return MLXTensor(self._data[idx], self._dtype)
    
    def __setitem__(self, idx, value):
        """
        Indexzuweisung
        
        Args:
            idx: Index oder Slice
            value: Wert, der zugewiesen werden soll
        """
        if isinstance(value, MLXTensor):
            self._data = mx.array_scatter(self._data, idx, value.data)
        else:
            self._data = mx.array_scatter(self._data, idx, mx.array(value))
    
    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> 'MLXTensor':
        """
        Berechnet die Summe entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der summiert wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Summe
        """
        if dim is None:
            return MLXTensor(mx.sum(self._data, keepdims=keepdim), self._dtype)
        else:
            return MLXTensor(mx.sum(self._data, axis=dim, keepdims=keepdim), self._dtype)
    
    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> 'MLXTensor':
        """
        Berechnet den Mittelwert entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der gemittelt wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Mittelwert
        """
        if dim is None:
            return MLXTensor(mx.mean(self._data, keepdims=keepdim), self._dtype)
        else:
            return MLXTensor(mx.mean(self._data, axis=dim, keepdims=keepdim), self._dtype)
    
    def var(self, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True) -> 'MLXTensor':
        """
        Berechnet die Varianz entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Varianz berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            unbiased: Ob die unverzerrte Schätzung verwendet werden soll
            
        Returns:
            Varianz
        """
        # MLX unterstützt unbiased=False nicht direkt, daher müssen wir es selbst implementieren
        if dim is None:
            mean = mx.mean(self._data, keepdims=True)
            var = mx.mean(mx.square(self._data - mean), keepdims=keepdim)
            if unbiased and self.size > 1:
                var = var * self.size / (self.size - 1)
            return MLXTensor(var, self._dtype)
        else:
            mean = mx.mean(self._data, axis=dim, keepdims=True)
            var = mx.mean(mx.square(self._data - mean), axis=dim, keepdims=keepdim)
            if unbiased:
                # Berechne die Größe der Dimension
                dim_size = self.shape[dim]
                if dim_size > 1:
                    var = var * dim_size / (dim_size - 1)
            return MLXTensor(var, self._dtype)
    
    def std(self, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True) -> 'MLXTensor':
        """
        Berechnet die Standardabweichung entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Standardabweichung berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            unbiased: Ob die unverzerrte Schätzung verwendet werden soll
            
        Returns:
            Standardabweichung
        """
        var = self.var(dim=dim, keepdim=keepdim, unbiased=unbiased)
        return MLXTensor(mx.sqrt(var.data), self._dtype)
    
    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Union['MLXTensor', Tuple['MLXTensor', 'MLXTensor']]:
        """
        Berechnet das Maximum entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der das Maximum berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Maximum und Indizes, wenn dim angegeben ist, sonst nur Maximum
        """
        if dim is None:
            return MLXTensor(mx.max(self._data, keepdims=keepdim), self._dtype)
        else:
            values = mx.max(self._data, axis=dim, keepdims=keepdim)
            indices = mx.argmax(self._data, axis=dim, keepdims=keepdim)
            return MLXTensor(values, self._dtype), MLXTensor(indices, 'int32')
    
    def min(self, dim: Optional[int] = None, keepdim: bool = False) -> Union['MLXTensor', Tuple['MLXTensor', 'MLXTensor']]:
        """
        Berechnet das Minimum entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der das Minimum berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Minimum und Indizes, wenn dim angegeben ist, sonst nur Minimum
        """
        if dim is None:
            return MLXTensor(mx.min(self._data, keepdims=keepdim), self._dtype)
        else:
            values = mx.min(self._data, axis=dim, keepdims=keepdim)
            indices = mx.argmin(self._data, axis=dim, keepdims=keepdim)
            return MLXTensor(values, self._dtype), MLXTensor(indices, 'int32')
    
    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> 'MLXTensor':
        """
        Berechnet die Indizes des Maximums entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Indizes berechnet werden
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Indizes des Maximums
        """
        if dim is None:
            # Wenn keine Dimension angegeben ist, finde das globale Maximum
            flat_indices = mx.argmax(mx.reshape(self._data, (-1,)))
            if keepdim:
                # Konvertiere flache Indizes zurück zu mehrdimensionalen Indizes
                indices = []
                shape = self.shape
                for d in range(len(shape)):
                    size = 1
                    for i in range(d + 1, len(shape)):
                        size *= shape[i]
                    indices.append(flat_indices // size)
                    flat_indices = flat_indices % size
                return MLXTensor(mx.array(indices), 'int32')
            else:
                return MLXTensor(flat_indices, 'int32')
        else:
            return MLXTensor(mx.argmax(self._data, axis=dim, keepdims=keepdim), 'int32')
    
    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> 'MLXTensor':
        """
        Berechnet die Indizes des Minimums entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Indizes berechnet werden
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Indizes des Minimums
        """
        if dim is None:
            # Wenn keine Dimension angegeben ist, finde das globale Minimum
            flat_indices = mx.argmin(mx.reshape(self._data, (-1,)))
            if keepdim:
                # Konvertiere flache Indizes zurück zu mehrdimensionalen Indizes
                indices = []
                shape = self.shape
                for d in range(len(shape)):
                    size = 1
                    for i in range(d + 1, len(shape)):
                        size *= shape[i]
                    indices.append(flat_indices // size)
                    flat_indices = flat_indices % size
                return MLXTensor(mx.array(indices), 'int32')
            else:
                return MLXTensor(flat_indices, 'int32')
        else:
            return MLXTensor(mx.argmin(self._data, axis=dim, keepdims=keepdim), 'int32')
    
    def norm(self, p: int = 2, dim: Optional[int] = None, keepdim: bool = False) -> 'MLXTensor':
        """
        Berechnet die Norm entlang einer Dimension
        
        Args:
            p: Ordnung der Norm
            dim: Dimension, entlang der die Norm berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Norm
        """
        if p == 0:
            # L0-Norm (Anzahl der Nicht-Null-Elemente)
            if dim is None:
                return MLXTensor(mx.sum(mx.abs(self._data) > 0, keepdims=keepdim), 'int32')
            else:
                return MLXTensor(mx.sum(mx.abs(self._data) > 0, axis=dim, keepdims=keepdim), 'int32')
        elif p == 1:
            # L1-Norm (Summe der Absolutwerte)
            if dim is None:
                return MLXTensor(mx.sum(mx.abs(self._data), keepdims=keepdim), self._dtype)
            else:
                return MLXTensor(mx.sum(mx.abs(self._data), axis=dim, keepdims=keepdim), self._dtype)
        elif p == 2:
            # L2-Norm (Euklidische Norm)
            if dim is None:
                return MLXTensor(mx.sqrt(mx.sum(mx.square(self._data), keepdims=keepdim)), self._dtype)
            else:
                return MLXTensor(mx.sqrt(mx.sum(mx.square(self._data), axis=dim, keepdims=keepdim)), self._dtype)
        elif p == float('inf'):
            # L∞-Norm (Maximum der Absolutwerte)
            if dim is None:
                return MLXTensor(mx.max(mx.abs(self._data), keepdims=keepdim), self._dtype)
            else:
                return MLXTensor(mx.max(mx.abs(self._data), axis=dim, keepdims=keepdim), self._dtype)
        else:
            # Lp-Norm
            if dim is None:
                return MLXTensor(mx.power(mx.sum(mx.power(mx.abs(self._data), p), keepdims=keepdim), 1/p), self._dtype)
            else:
                return MLXTensor(mx.power(mx.sum(mx.power(mx.abs(self._data), p), axis=dim, keepdims=keepdim), 1/p), self._dtype)
    
    def normalize(self, p: int = 2, dim: int = 0) -> 'MLXTensor':
        """
        Normalisiert den Tensor entlang einer Dimension
        
        Args:
            p: Ordnung der Norm
            dim: Dimension, entlang der normalisiert wird
            
        Returns:
            Normalisierter Tensor
        """
        norm_values = self.norm(p=p, dim=dim, keepdim=True)
        # Vermeide Division durch Null
        norm_values = mx.maximum(norm_values.data, 1e-12)
        return MLXTensor(self._data / norm_values.data, self._dtype)
    
    def exp(self) -> 'MLXTensor':
        """
        Berechnet die Exponentialfunktion (elementweise)
        
        Returns:
            Tensor mit Exponentialwerten
        """
        return MLXTensor(mx.exp(self._data), self._dtype)
    
    def log(self) -> 'MLXTensor':
        """
        Berechnet den natürlichen Logarithmus (elementweise)
        
        Returns:
            Tensor mit Logarithmuswerten
        """
        return MLXTensor(mx.log(self._data), self._dtype)
    
    def sin(self) -> 'MLXTensor':
        """
        Berechnet den Sinus (elementweise)
        
        Returns:
            Tensor mit Sinuswerten
        """
        return MLXTensor(mx.sin(self._data), self._dtype)
    
    def cos(self) -> 'MLXTensor':
        """
        Berechnet den Kosinus (elementweise)
        
        Returns:
            Tensor mit Kosinuswerten
        """
        return MLXTensor(mx.cos(self._data), self._dtype)
    
    def tan(self) -> 'MLXTensor':
        """
        Berechnet den Tangens (elementweise)
        
        Returns:
            Tensor mit Tangenswerten
        """
        return MLXTensor(mx.tan(self._data), self._dtype)
    
    def sinh(self) -> 'MLXTensor':
        """
        Berechnet den Sinus hyperbolicus (elementweise)
        
        Returns:
            Tensor mit Sinus hyperbolicus-Werten
        """
        return MLXTensor(mx.sinh(self._data), self._dtype)
    
    def cosh(self) -> 'MLXTensor':
        """
        Berechnet den Kosinus hyperbolicus (elementweise)
        
        Returns:
            Tensor mit Kosinus hyperbolicus-Werten
        """
        return MLXTensor(mx.cosh(self._data), self._dtype)
    
    def tanh(self) -> 'MLXTensor':
        """
        Berechnet den Tangens hyperbolicus (elementweise)
        
        Returns:
            Tensor mit Tangens hyperbolicus-Werten
        """
        return MLXTensor(mx.tanh(self._data), self._dtype)
    
    def sigmoid(self) -> 'MLXTensor':
        """
        Berechnet die Sigmoid-Funktion (elementweise)
        
        Returns:
            Tensor mit Sigmoid-Werten
        """
        return MLXTensor(mx.sigmoid(self._data), self._dtype)
    
    def relu(self) -> 'MLXTensor':
        """
        Berechnet die ReLU-Funktion (elementweise)
        
        Returns:
            Tensor mit ReLU-Werten
        """
        return MLXTensor(mx.maximum(self._data, 0), self._dtype)
    
    def softmax(self, dim: int = -1) -> 'MLXTensor':
        """
        Berechnet die Softmax-Funktion entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Softmax-Funktion berechnet wird
            
        Returns:
            Tensor mit Softmax-Werten
        """
        return MLXTensor(mx.softmax(self._data, axis=dim), self._dtype)
    
    def to_device(self, device: str) -> 'MLXTensor':
        """
        Verschiebt den Tensor auf ein anderes Gerät
        
        Args:
            device: Zielgerät (z.B. "cpu", "mps", "cuda")
            
        Returns:
            Tensor auf dem Zielgerät
        """
        # MLX verwaltet Geräte automatisch, daher ist diese Methode ein No-op
        logger.info("MLX verwaltet Geräte automatisch, to_device() ist ein No-op")
        return self
    
    def to_dtype(self, dtype: str) -> 'MLXTensor':
        """
        Konvertiert den Tensor zu einem anderen Datentyp
        
        Args:
            dtype: Zieldatentyp
            
        Returns:
            Tensor mit dem Zieldatentyp
        """
        return MLXTensor(self._data.astype(dtype), dtype)
