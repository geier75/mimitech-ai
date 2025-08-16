#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - NumPy Tensor-Implementierung

Dieses Modul implementiert die NumPy-spezifische Tensor-Klasse für die T-Mathematics Engine,
die als Fallback dient, wenn MLX oder PyTorch nicht verfügbar sind.

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
logger = logging.getLogger("MISO.Ultimate.TMathematics.NumPyTensor")

# Importiere Basisklasse
from .tensor import MISOTensor


class NumPyTensor(MISOTensor):
    """
    NumPy-spezifische Tensor-Implementierung für die T-Mathematics Engine
    
    Diese Klasse implementiert die Tensor-Operationen mit NumPy,
    die als Fallback dient, wenn MLX oder PyTorch nicht verfügbar sind.
    """
    
    def __init__(self, data, dtype=None):
        """
        Initialisiert einen NumPy-Tensor
        
        Args:
            data: Daten für den Tensor (kann ein NumPy-Array, eine Liste oder ein Tensor sein)
            dtype: Datentyp für den Tensor
        """
        super().__init__(data, dtype)
        self._backend = "numpy"
        
        # Konvertiere Daten zu NumPy-Array
        if isinstance(data, np.ndarray):
            self._data = data.copy()
        elif isinstance(data, (list, tuple)):
            self._data = np.array(data, dtype=dtype)
        elif isinstance(data, (int, float, bool)):
            self._data = np.array([data], dtype=dtype)
        elif isinstance(data, MISOTensor):
            self._data = data.to_numpy().copy()
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
        return self._data
    
    def clone(self) -> 'NumPyTensor':
        """
        Erstellt eine Kopie des Tensors
        
        Returns:
            Kopie des Tensors
        """
        return NumPyTensor(self._data.copy(), self._dtype)
    
    def reshape(self, *shape) -> 'NumPyTensor':
        """
        Ändert die Form des Tensors
        
        Args:
            shape: Neue Form
            
        Returns:
            Tensor mit neuer Form
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return NumPyTensor(self._data.reshape(shape), self._dtype)
    
    def transpose(self, *dims) -> 'NumPyTensor':
        """
        Transponiert den Tensor
        
        Args:
            dims: Dimensionen, die transponiert werden sollen
            
        Returns:
            Transponierter Tensor
        """
        if not dims:
            # Wenn keine Dimensionen angegeben sind, transponiere alle Dimensionen
            return NumPyTensor(self._data.transpose(), self._dtype)
        elif len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]
        
        return NumPyTensor(np.transpose(self._data, dims), self._dtype)
    
    def squeeze(self, dim: Optional[int] = None) -> 'NumPyTensor':
        """
        Entfernt Dimensionen mit Größe 1
        
        Args:
            dim: Dimension, die entfernt werden soll (optional)
            
        Returns:
            Tensor mit entfernten Dimensionen
        """
        if dim is None:
            return NumPyTensor(np.squeeze(self._data), self._dtype)
        else:
            return NumPyTensor(np.squeeze(self._data, axis=dim), self._dtype)
    
    def unsqueeze(self, dim: int) -> 'NumPyTensor':
        """
        Fügt eine Dimension mit Größe 1 hinzu
        
        Args:
            dim: Position, an der die Dimension hinzugefügt werden soll
            
        Returns:
            Tensor mit hinzugefügter Dimension
        """
        return NumPyTensor(np.expand_dims(self._data, axis=dim), self._dtype)
    
    def __add__(self, other) -> 'NumPyTensor':
        """
        Addition
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Addition
        """
        if isinstance(other, NumPyTensor):
            return NumPyTensor(self._data + other.data, self._dtype)
        else:
            return NumPyTensor(self._data + other, self._dtype)
    
    def __sub__(self, other) -> 'NumPyTensor':
        """
        Subtraktion
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Subtraktion
        """
        if isinstance(other, NumPyTensor):
            return NumPyTensor(self._data - other.data, self._dtype)
        else:
            return NumPyTensor(self._data - other, self._dtype)
    
    def __rsub__(self, other) -> 'NumPyTensor':
        """
        Rechte Subtraktion
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Subtraktion
        """
        if isinstance(other, NumPyTensor):
            return NumPyTensor(other.data - self._data, self._dtype)
        else:
            return NumPyTensor(other - self._data, self._dtype)
    
    def __mul__(self, other) -> 'NumPyTensor':
        """
        Multiplikation (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Multiplikation
        """
        if isinstance(other, NumPyTensor):
            return NumPyTensor(self._data * other.data, self._dtype)
        else:
            return NumPyTensor(self._data * other, self._dtype)
    
    def __truediv__(self, other) -> 'NumPyTensor':
        """
        Division (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Division
        """
        if isinstance(other, NumPyTensor):
            return NumPyTensor(self._data / other.data, self._dtype)
        else:
            return NumPyTensor(self._data / other, self._dtype)
    
    def __rtruediv__(self, other) -> 'NumPyTensor':
        """
        Rechte Division (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Division
        """
        if isinstance(other, NumPyTensor):
            return NumPyTensor(other.data / self._data, self._dtype)
        else:
            return NumPyTensor(other / self._data, self._dtype)
    
    def __pow__(self, other) -> 'NumPyTensor':
        """
        Potenzierung (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Potenzierung
        """
        if isinstance(other, NumPyTensor):
            return NumPyTensor(np.power(self._data, other.data), self._dtype)
        else:
            return NumPyTensor(np.power(self._data, other), self._dtype)
    
    def __rpow__(self, other) -> 'NumPyTensor':
        """
        Rechte Potenzierung (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Potenzierung
        """
        if isinstance(other, NumPyTensor):
            return NumPyTensor(np.power(other.data, self._data), self._dtype)
        else:
            return NumPyTensor(np.power(other, self._data), self._dtype)
    
    def __matmul__(self, other) -> 'NumPyTensor':
        """
        Matrix-Multiplikation
        
        Args:
            other: Tensor
            
        Returns:
            Ergebnis der Matrix-Multiplikation
        """
        if isinstance(other, NumPyTensor):
            return NumPyTensor(self._data @ other.data, self._dtype)
        else:
            return NumPyTensor(self._data @ np.array(other), self._dtype)
    
    def __rmatmul__(self, other) -> 'NumPyTensor':
        """
        Rechte Matrix-Multiplikation
        
        Args:
            other: Tensor
            
        Returns:
            Ergebnis der Matrix-Multiplikation
        """
        if isinstance(other, NumPyTensor):
            return NumPyTensor(other.data @ self._data, self._dtype)
        else:
            return NumPyTensor(np.array(other) @ self._data, self._dtype)
    
    def __neg__(self) -> 'NumPyTensor':
        """
        Negation
        
        Returns:
            Negierter Tensor
        """
        return NumPyTensor(-self._data, self._dtype)
    
    def __abs__(self) -> 'NumPyTensor':
        """
        Absolutwert (elementweise)
        
        Returns:
            Tensor mit Absolutwerten
        """
        return NumPyTensor(np.abs(self._data), self._dtype)
    
    def __getitem__(self, idx) -> 'NumPyTensor':
        """
        Indexierung
        
        Args:
            idx: Index oder Slice
            
        Returns:
            Teilmenge des Tensors
        """
        return NumPyTensor(self._data[idx], self._dtype)
    
    def __setitem__(self, idx, value):
        """
        Indexzuweisung
        
        Args:
            idx: Index oder Slice
            value: Wert, der zugewiesen werden soll
        """
        if isinstance(value, NumPyTensor):
            self._data[idx] = value.data
        else:
            self._data[idx] = value
    
    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> 'NumPyTensor':
        """
        Berechnet die Summe entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der summiert wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Summe
        """
        if dim is None:
            return NumPyTensor(np.sum(self._data, keepdims=keepdim), self._dtype)
        else:
            return NumPyTensor(np.sum(self._data, axis=dim, keepdims=keepdim), self._dtype)
    
    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> 'NumPyTensor':
        """
        Berechnet den Mittelwert entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der gemittelt wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Mittelwert
        """
        if dim is None:
            return NumPyTensor(np.mean(self._data, keepdims=keepdim), self._dtype)
        else:
            return NumPyTensor(np.mean(self._data, axis=dim, keepdims=keepdim), self._dtype)
    
    def var(self, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True) -> 'NumPyTensor':
        """
        Berechnet die Varianz entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Varianz berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            unbiased: Ob die unverzerrte Schätzung verwendet werden soll
            
        Returns:
            Varianz
        """
        # NumPy verwendet ddof=1 für unverzerrte Schätzung
        ddof = 1 if unbiased else 0
        
        if dim is None:
            return NumPyTensor(np.var(self._data, ddof=ddof, keepdims=keepdim), self._dtype)
        else:
            return NumPyTensor(np.var(self._data, axis=dim, ddof=ddof, keepdims=keepdim), self._dtype)
    
    def std(self, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True) -> 'NumPyTensor':
        """
        Berechnet die Standardabweichung entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Standardabweichung berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            unbiased: Ob die unverzerrte Schätzung verwendet werden soll
            
        Returns:
            Standardabweichung
        """
        # NumPy verwendet ddof=1 für unverzerrte Schätzung
        ddof = 1 if unbiased else 0
        
        if dim is None:
            return NumPyTensor(np.std(self._data, ddof=ddof, keepdims=keepdim), self._dtype)
        else:
            return NumPyTensor(np.std(self._data, axis=dim, ddof=ddof, keepdims=keepdim), self._dtype)
    
    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Union['NumPyTensor', Tuple['NumPyTensor', 'NumPyTensor']]:
        """
        Berechnet das Maximum entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der das Maximum berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Maximum und Indizes, wenn dim angegeben ist, sonst nur Maximum
        """
        if dim is None:
            return NumPyTensor(np.max(self._data, keepdims=keepdim), self._dtype)
        else:
            values = np.max(self._data, axis=dim, keepdims=keepdim)
            indices = np.argmax(self._data, axis=dim)
            if keepdim:
                indices = np.expand_dims(indices, axis=dim)
            return NumPyTensor(values, self._dtype), NumPyTensor(indices, "int64")
    
    def min(self, dim: Optional[int] = None, keepdim: bool = False) -> Union['NumPyTensor', Tuple['NumPyTensor', 'NumPyTensor']]:
        """
        Berechnet das Minimum entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der das Minimum berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Minimum und Indizes, wenn dim angegeben ist, sonst nur Minimum
        """
        if dim is None:
            return NumPyTensor(np.min(self._data, keepdims=keepdim), self._dtype)
        else:
            values = np.min(self._data, axis=dim, keepdims=keepdim)
            indices = np.argmin(self._data, axis=dim)
            if keepdim:
                indices = np.expand_dims(indices, axis=dim)
            return NumPyTensor(values, self._dtype), NumPyTensor(indices, "int64")
    
    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> 'NumPyTensor':
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
            flat_indices = np.argmax(self._data.flatten())
            return NumPyTensor(flat_indices, "int64")
        else:
            indices = np.argmax(self._data, axis=dim)
            if keepdim:
                indices = np.expand_dims(indices, axis=dim)
            return NumPyTensor(indices, "int64")
    
    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> 'NumPyTensor':
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
            flat_indices = np.argmin(self._data.flatten())
            return NumPyTensor(flat_indices, "int64")
        else:
            indices = np.argmin(self._data, axis=dim)
            if keepdim:
                indices = np.expand_dims(indices, axis=dim)
            return NumPyTensor(indices, "int64")
    
    def norm(self, p: int = 2, dim: Optional[int] = None, keepdim: bool = False) -> 'NumPyTensor':
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
                return NumPyTensor(np.sum(np.abs(self._data) > 0, keepdims=keepdim), "int64")
            else:
                return NumPyTensor(np.sum(np.abs(self._data) > 0, axis=dim, keepdims=keepdim), "int64")
        elif p == 1:
            # L1-Norm (Summe der Absolutwerte)
            if dim is None:
                return NumPyTensor(np.sum(np.abs(self._data), keepdims=keepdim), self._dtype)
            else:
                return NumPyTensor(np.sum(np.abs(self._data), axis=dim, keepdims=keepdim), self._dtype)
        elif p == 2:
            # L2-Norm (Euklidische Norm)
            if dim is None:
                return NumPyTensor(np.sqrt(np.sum(np.square(self._data), keepdims=keepdim)), self._dtype)
            else:
                return NumPyTensor(np.sqrt(np.sum(np.square(self._data), axis=dim, keepdims=keepdim)), self._dtype)
        elif p == float('inf'):
            # L∞-Norm (Maximum der Absolutwerte)
            if dim is None:
                return NumPyTensor(np.max(np.abs(self._data), keepdims=keepdim), self._dtype)
            else:
                return NumPyTensor(np.max(np.abs(self._data), axis=dim, keepdims=keepdim), self._dtype)
        else:
            # Lp-Norm
            if dim is None:
                return NumPyTensor(np.power(np.sum(np.power(np.abs(self._data), p), keepdims=keepdim), 1/p), self._dtype)
            else:
                return NumPyTensor(np.power(np.sum(np.power(np.abs(self._data), p), axis=dim, keepdims=keepdim), 1/p), self._dtype)
    
    def normalize(self, p: int = 2, dim: int = 0) -> 'NumPyTensor':
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
        norm_values = np.maximum(norm_values.data, 1e-12)
        return NumPyTensor(self._data / norm_values, self._dtype)
    
    def exp(self) -> 'NumPyTensor':
        """
        Berechnet die Exponentialfunktion (elementweise)
        
        Returns:
            Tensor mit Exponentialwerten
        """
        return NumPyTensor(np.exp(self._data), self._dtype)
    
    def log(self) -> 'NumPyTensor':
        """
        Berechnet den natürlichen Logarithmus (elementweise)
        
        Returns:
            Tensor mit Logarithmuswerten
        """
        return NumPyTensor(np.log(self._data), self._dtype)
    
    def sin(self) -> 'NumPyTensor':
        """
        Berechnet den Sinus (elementweise)
        
        Returns:
            Tensor mit Sinuswerten
        """
        return NumPyTensor(np.sin(self._data), self._dtype)
    
    def cos(self) -> 'NumPyTensor':
        """
        Berechnet den Kosinus (elementweise)
        
        Returns:
            Tensor mit Kosinuswerten
        """
        return NumPyTensor(np.cos(self._data), self._dtype)
    
    def tan(self) -> 'NumPyTensor':
        """
        Berechnet den Tangens (elementweise)
        
        Returns:
            Tensor mit Tangenswerten
        """
        return NumPyTensor(np.tan(self._data), self._dtype)
    
    def sinh(self) -> 'NumPyTensor':
        """
        Berechnet den Sinus hyperbolicus (elementweise)
        
        Returns:
            Tensor mit Sinus hyperbolicus-Werten
        """
        return NumPyTensor(np.sinh(self._data), self._dtype)
    
    def cosh(self) -> 'NumPyTensor':
        """
        Berechnet den Kosinus hyperbolicus (elementweise)
        
        Returns:
            Tensor mit Kosinus hyperbolicus-Werten
        """
        return NumPyTensor(np.cosh(self._data), self._dtype)
    
    def tanh(self) -> 'NumPyTensor':
        """
        Berechnet den Tangens hyperbolicus (elementweise)
        
        Returns:
            Tensor mit Tangens hyperbolicus-Werten
        """
        return NumPyTensor(np.tanh(self._data), self._dtype)
    
    def sigmoid(self) -> 'NumPyTensor':
        """
        Berechnet die Sigmoid-Funktion (elementweise)
        
        Returns:
            Tensor mit Sigmoid-Werten
        """
        return NumPyTensor(1 / (1 + np.exp(-self._data)), self._dtype)
    
    def relu(self) -> 'NumPyTensor':
        """
        Berechnet die ReLU-Funktion (elementweise)
        
        Returns:
            Tensor mit ReLU-Werten
        """
        return NumPyTensor(np.maximum(self._data, 0), self._dtype)
    
    def softmax(self, dim: int = -1) -> 'NumPyTensor':
        """
        Berechnet die Softmax-Funktion entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Softmax-Funktion berechnet wird
            
        Returns:
            Tensor mit Softmax-Werten
        """
        # Numerisch stabile Softmax-Implementierung
        x = self._data - np.max(self._data, axis=dim, keepdims=True)
        exp_x = np.exp(x)
        return NumPyTensor(exp_x / np.sum(exp_x, axis=dim, keepdims=True), self._dtype)
    
    def to_device(self, device: str) -> 'NumPyTensor':
        """
        Verschiebt den Tensor auf ein anderes Gerät
        
        Args:
            device: Zielgerät (z.B. "cpu", "mps", "cuda")
            
        Returns:
            Tensor auf dem Zielgerät
        """
        # NumPy unterstützt keine Geräte, daher ist diese Methode ein No-op
        logger.info("NumPy unterstützt keine Geräte, to_device() ist ein No-op")
        return self
    
    def to_dtype(self, dtype: str) -> 'NumPyTensor':
        """
        Konvertiert den Tensor zu einem anderen Datentyp
        
        Args:
            dtype: Zieldatentyp
            
        Returns:
            Tensor mit dem Zieldatentyp
        """
        return NumPyTensor(self._data.astype(dtype), dtype)
