#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - MISOTensor Basisklasse

Dieses Modul implementiert die Basisklasse für Tensoren in der T-Mathematics Engine,
die für verschiedene Backends (MLX, PyTorch, NumPy) verwendet werden kann.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, Type, TypeVar

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.Ultimate.TMathematics.Tensor")

# Typ-Variable für MISOTensor
T = TypeVar('T', bound='MISOTensor')


class MISOTensor:
    """
    Basisklasse für Tensoren in der T-Mathematics Engine
    
    Diese Klasse definiert die gemeinsame Schnittstelle für alle Tensor-Implementierungen,
    unabhängig vom verwendeten Backend (MLX, PyTorch, NumPy).
    """
    
    def __init__(self, data, dtype=None):
        """
        Initialisiert einen Tensor
        
        Args:
            data: Daten für den Tensor
            dtype: Datentyp für den Tensor
        """
        self._data = None  # Wird von Unterklassen überschrieben
        self._dtype = dtype
        self._backend = "base"  # Wird von Unterklassen überschrieben
    
    @property
    def data(self):
        """Gibt die Rohdaten des Tensors zurück"""
        return self._data
    
    @property
    def dtype(self):
        """Gibt den Datentyp des Tensors zurück"""
        return self._dtype
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Gibt die Form des Tensors zurück"""
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    @property
    def ndim(self) -> int:
        """Gibt die Anzahl der Dimensionen des Tensors zurück"""
        return len(self.shape)
    
    @property
    def size(self) -> int:
        """Gibt die Gesamtanzahl der Elemente des Tensors zurück"""
        return np.prod(self.shape)
    
    @property
    def backend(self) -> str:
        """Gibt das Backend des Tensors zurück"""
        return self._backend
    
    def __repr__(self) -> str:
        """Gibt eine String-Repräsentation des Tensors zurück"""
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype}, backend={self.backend})"
    
    def __str__(self) -> str:
        """Gibt eine String-Repräsentation des Tensors zurück"""
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype})"
    
    def to_numpy(self) -> np.ndarray:
        """
        Konvertiert den Tensor zu einem NumPy-Array
        
        Returns:
            NumPy-Array
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def to_list(self) -> List:
        """
        Konvertiert den Tensor zu einer verschachtelten Liste
        
        Returns:
            Verschachtelte Liste
        """
        return self.to_numpy().tolist()
    
    def item(self) -> Union[float, int, bool]:
        """
        Gibt den Wert eines skalaren Tensors zurück
        
        Returns:
            Skalarer Wert
        
        Raises:
            ValueError: Wenn der Tensor nicht skalar ist
        """
        if self.size != 1:
            raise ValueError(f"Tensor mit Größe {self.size} kann nicht in einen skalaren Wert konvertiert werden")
        return self.to_numpy().item()
    
    def clone(self: T) -> T:
        """
        Erstellt eine Kopie des Tensors
        
        Returns:
            Kopie des Tensors
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def reshape(self: T, *shape) -> T:
        """
        Ändert die Form des Tensors
        
        Args:
            shape: Neue Form
            
        Returns:
            Tensor mit neuer Form
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def transpose(self: T, *dims) -> T:
        """
        Transponiert den Tensor
        
        Args:
            dims: Dimensionen, die transponiert werden sollen
            
        Returns:
            Transponierter Tensor
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def flatten(self: T) -> T:
        """
        Flacht den Tensor zu einem 1D-Tensor ab
        
        Returns:
            Abgeflachter Tensor
        """
        return self.reshape(-1)
    
    def squeeze(self: T, dim: Optional[int] = None) -> T:
        """
        Entfernt Dimensionen mit Größe 1
        
        Args:
            dim: Dimension, die entfernt werden soll (optional)
            
        Returns:
            Tensor mit entfernten Dimensionen
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def unsqueeze(self: T, dim: int) -> T:
        """
        Fügt eine Dimension mit Größe 1 hinzu
        
        Args:
            dim: Position, an der die Dimension hinzugefügt werden soll
            
        Returns:
            Tensor mit hinzugefügter Dimension
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __add__(self: T, other) -> T:
        """
        Addition
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Addition
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __radd__(self: T, other) -> T:
        """
        Rechte Addition
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Addition
        """
        return self.__add__(other)
    
    def __sub__(self: T, other) -> T:
        """
        Subtraktion
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Subtraktion
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __rsub__(self: T, other) -> T:
        """
        Rechte Subtraktion
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Subtraktion
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __mul__(self: T, other) -> T:
        """
        Multiplikation (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Multiplikation
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __rmul__(self: T, other) -> T:
        """
        Rechte Multiplikation (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Multiplikation
        """
        return self.__mul__(other)
    
    def __truediv__(self: T, other) -> T:
        """
        Division (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Division
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __rtruediv__(self: T, other) -> T:
        """
        Rechte Division (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Division
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __pow__(self: T, other) -> T:
        """
        Potenzierung (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Potenzierung
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __rpow__(self: T, other) -> T:
        """
        Rechte Potenzierung (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Potenzierung
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __matmul__(self: T, other) -> T:
        """
        Matrix-Multiplikation
        
        Args:
            other: Tensor
            
        Returns:
            Ergebnis der Matrix-Multiplikation
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __rmatmul__(self: T, other) -> T:
        """
        Rechte Matrix-Multiplikation
        
        Args:
            other: Tensor
            
        Returns:
            Ergebnis der Matrix-Multiplikation
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __neg__(self: T) -> T:
        """
        Negation
        
        Returns:
            Negierter Tensor
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __abs__(self: T) -> T:
        """
        Absolutwert (elementweise)
        
        Returns:
            Tensor mit Absolutwerten
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __getitem__(self: T, idx) -> T:
        """
        Indexierung
        
        Args:
            idx: Index oder Slice
            
        Returns:
            Teilmenge des Tensors
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def __setitem__(self, idx, value):
        """
        Indexzuweisung
        
        Args:
            idx: Index oder Slice
            value: Wert, der zugewiesen werden soll
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def sum(self: T, dim: Optional[int] = None, keepdim: bool = False) -> T:
        """
        Berechnet die Summe entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der summiert wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Summe
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def mean(self: T, dim: Optional[int] = None, keepdim: bool = False) -> T:
        """
        Berechnet den Mittelwert entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der gemittelt wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Mittelwert
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def var(self: T, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True) -> T:
        """
        Berechnet die Varianz entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Varianz berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            unbiased: Ob die unverzerrte Schätzung verwendet werden soll
            
        Returns:
            Varianz
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def std(self: T, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True) -> T:
        """
        Berechnet die Standardabweichung entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Standardabweichung berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            unbiased: Ob die unverzerrte Schätzung verwendet werden soll
            
        Returns:
            Standardabweichung
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def max(self: T, dim: Optional[int] = None, keepdim: bool = False) -> Union[T, Tuple[T, T]]:
        """
        Berechnet das Maximum entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der das Maximum berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Maximum und Indizes, wenn dim angegeben ist, sonst nur Maximum
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def min(self: T, dim: Optional[int] = None, keepdim: bool = False) -> Union[T, Tuple[T, T]]:
        """
        Berechnet das Minimum entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der das Minimum berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Minimum und Indizes, wenn dim angegeben ist, sonst nur Minimum
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def argmax(self: T, dim: Optional[int] = None, keepdim: bool = False) -> T:
        """
        Berechnet die Indizes des Maximums entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Indizes berechnet werden
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Indizes des Maximums
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def argmin(self: T, dim: Optional[int] = None, keepdim: bool = False) -> T:
        """
        Berechnet die Indizes des Minimums entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Indizes berechnet werden
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Indizes des Minimums
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def norm(self: T, p: int = 2, dim: Optional[int] = None, keepdim: bool = False) -> T:
        """
        Berechnet die Norm entlang einer Dimension
        
        Args:
            p: Ordnung der Norm
            dim: Dimension, entlang der die Norm berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Norm
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def normalize(self: T, p: int = 2, dim: int = 0) -> T:
        """
        Normalisiert den Tensor entlang einer Dimension
        
        Args:
            p: Ordnung der Norm
            dim: Dimension, entlang der normalisiert wird
            
        Returns:
            Normalisierter Tensor
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def exp(self: T) -> T:
        """
        Berechnet die Exponentialfunktion (elementweise)
        
        Returns:
            Tensor mit Exponentialwerten
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def log(self: T) -> T:
        """
        Berechnet den natürlichen Logarithmus (elementweise)
        
        Returns:
            Tensor mit Logarithmuswerten
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def sin(self: T) -> T:
        """
        Berechnet den Sinus (elementweise)
        
        Returns:
            Tensor mit Sinuswerten
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def cos(self: T) -> T:
        """
        Berechnet den Kosinus (elementweise)
        
        Returns:
            Tensor mit Kosinuswerten
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def tan(self: T) -> T:
        """
        Berechnet den Tangens (elementweise)
        
        Returns:
            Tensor mit Tangenswerten
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def sinh(self: T) -> T:
        """
        Berechnet den Sinus hyperbolicus (elementweise)
        
        Returns:
            Tensor mit Sinus hyperbolicus-Werten
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def cosh(self: T) -> T:
        """
        Berechnet den Kosinus hyperbolicus (elementweise)
        
        Returns:
            Tensor mit Kosinus hyperbolicus-Werten
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def tanh(self: T) -> T:
        """
        Berechnet den Tangens hyperbolicus (elementweise)
        
        Returns:
            Tensor mit Tangens hyperbolicus-Werten
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def sigmoid(self: T) -> T:
        """
        Berechnet die Sigmoid-Funktion (elementweise)
        
        Returns:
            Tensor mit Sigmoid-Werten
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def relu(self: T) -> T:
        """
        Berechnet die ReLU-Funktion (elementweise)
        
        Returns:
            Tensor mit ReLU-Werten
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def softmax(self: T, dim: int = -1) -> T:
        """
        Berechnet die Softmax-Funktion entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Softmax-Funktion berechnet wird
            
        Returns:
            Tensor mit Softmax-Werten
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def to_device(self: T, device: str) -> T:
        """
        Verschiebt den Tensor auf ein anderes Gerät
        
        Args:
            device: Zielgerät (z.B. "cpu", "mps", "cuda")
            
        Returns:
            Tensor auf dem Zielgerät
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
    
    def to_dtype(self: T, dtype: str) -> T:
        """
        Konvertiert den Tensor zu einem anderen Datentyp
        
        Args:
            dtype: Zieldatentyp
            
        Returns:
            Tensor mit dem Zieldatentyp
        """
        raise NotImplementedError("Muss von Unterklassen implementiert werden")
