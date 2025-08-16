#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Engine - Tensor-Interface für VXOR-Komponenten

Diese Datei definiert die abstrakte Basisklasse für Tensor-Operationen,
die von allen VXOR-Komponenten verwendet werden soll, um eine konsistente
Schnittstelle für Tensor-Operationen zu gewährleisten.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class MISOTensorInterface(ABC):
    """
    Abstrakte Basisklasse, die die Schnittstelle für Tensor-Operationen definiert.
    
    Diese Klasse definiert alle Methoden, die von Tensor-Implementierungen bereitgestellt
    werden müssen, um mit dem MISO-System und VXOR-Komponenten kompatibel zu sein.
    """
    
    @property
    @abstractmethod
    def backend(self) -> str:
        """
        Gibt den Namen des Backend-Systems zurück.
        
        Returns:
            Name des Backend-Systems (z.B. 'mlx', 'torch', 'numpy')
        """
        pass
    
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """
        Gibt die Form des Tensors zurück.
        
        Returns:
            Tupel mit den Dimensionen des Tensors
        """
        pass
    
    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """
        Konvertiert den Tensor in ein NumPy-Array.
        
        Returns:
            NumPy-Array mit den Daten des Tensors
        """
        pass
    
    @abstractmethod
    def __add__(self, other: Any) -> 'MISOTensorInterface':
        """
        Addition zweier Tensoren.
        
        Args:
            other: Der zu addierende Tensor oder Skalar
            
        Returns:
            Neuer Tensor mit dem Ergebnis der Addition
        """
        pass
    
    @abstractmethod
    def __matmul__(self, other: Any) -> 'MISOTensorInterface':
        """
        Matrix-Multiplikation zweier Tensoren.
        
        Args:
            other: Der zu multiplizierende Tensor
            
        Returns:
            Neuer Tensor mit dem Ergebnis der Matrix-Multiplikation
        """
        pass
    
    @abstractmethod
    def exp(self) -> 'MISOTensorInterface':
        """
        Anwendung der Exponentialfunktion auf den Tensor.
        
        Returns:
            Neuer Tensor mit dem Ergebnis der Exponentialfunktion
        """
        pass
    
    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """
        Serialisiert den Tensor für die Übertragung zwischen Komponenten.
        
        Returns:
            Dictionary mit serialisierten Daten des Tensors (inkl. Backend-Info)
        """
        pass
    
    @classmethod
    @abstractmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'MISOTensorInterface':
        """
        Deserialisiert einen Tensor aus einem Dictionary.
        
        Args:
            data: Dictionary mit serialisierten Daten des Tensors
            
        Returns:
            Neuer Tensor aus den deserialisierten Daten
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_numpy(cls, array: np.ndarray) -> 'MISOTensorInterface':
        """
        Erstellt einen Tensor aus einem NumPy-Array.
        
        Args:
            array: NumPy-Array, aus dem der Tensor erstellt werden soll
            
        Returns:
            Neuer Tensor mit den Daten des NumPy-Arrays
        """
        pass
