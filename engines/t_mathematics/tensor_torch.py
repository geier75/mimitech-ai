#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - PyTorch Tensor-Implementierung

Dieses Modul implementiert die PyTorch-spezifische Tensor-Klasse für die T-Mathematics Engine,
optimiert für Apple Silicon mit MPS-Unterstützung.

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
logger = logging.getLogger("MISO.Ultimate.TMathematics.TorchTensor")

# Importiere PyTorch
try:
    import torch
    HAS_TORCH = True
    
    # Prüfe, ob MPS verfügbar ist (Apple Silicon)
    if torch.backends.mps.is_available():
        DEFAULT_DEVICE = "mps"
        logger.info("Apple MPS Backend verfügbar, verwende GPU-Beschleunigung")
    elif torch.cuda.is_available():
        DEFAULT_DEVICE = "cuda"
        logger.info("CUDA verfügbar, verwende GPU-Beschleunigung")
    else:
        DEFAULT_DEVICE = "cpu"
        logger.info("Keine GPU-Beschleunigung verfügbar, verwende CPU")
except ImportError:
    logger.warning("PyTorch nicht verfügbar, verwende Dummy-Implementierung")
    HAS_TORCH = False
    DEFAULT_DEVICE = "cpu"
    
    # Dummy-Klasse für den Fall, dass PyTorch nicht verfügbar ist
    class torch:
        class Tensor:
            pass
        
        @staticmethod
        def tensor(*args, **kwargs):
            return np.array(*args, **kwargs)
        
        @staticmethod
        def from_numpy(*args, **kwargs):
            return np.array(*args, **kwargs)
        
        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False
        
        class cuda:
            @staticmethod
            def is_available():
                return False

# Importiere Basisklasse
from .tensor import MISOTensor


class TorchTensor(MISOTensor):
    """
    PyTorch-spezifische Tensor-Implementierung für die T-Mathematics Engine
    
    Diese Klasse implementiert die Tensor-Operationen mit PyTorch,
    optimiert für Apple Silicon mit MPS-Unterstützung.
    """
    
    def __init__(self, data, dtype=None, device=None):
        """
        Initialisiert einen PyTorch-Tensor
        
        Args:
            data: Daten für den Tensor (kann ein NumPy-Array, eine Liste oder ein PyTorch-Tensor sein)
            dtype: Datentyp für den Tensor
            device: Gerät für den Tensor (z.B. "cpu", "mps", "cuda")
        """
        super().__init__(data, dtype)
        self._backend = "torch"
        
        # Setze Standardgerät, wenn keines angegeben ist
        if device is None:
            device = DEFAULT_DEVICE
        self._device = device
        
        # Konvertiere Datentyp zu PyTorch-Datentyp
        if dtype is not None:
            if isinstance(dtype, str):
                if dtype == "float32" or dtype == "float":
                    torch_dtype = torch.float32
                elif dtype == "float64" or dtype == "double":
                    torch_dtype = torch.float64
                elif dtype == "float16" or dtype == "half":
                    torch_dtype = torch.float16
                elif dtype == "int32" or dtype == "int":
                    torch_dtype = torch.int32
                elif dtype == "int64" or dtype == "long":
                    torch_dtype = torch.int64
                elif dtype == "int16" or dtype == "short":
                    torch_dtype = torch.int16
                elif dtype == "int8":
                    torch_dtype = torch.int8
                elif dtype == "uint8":
                    torch_dtype = torch.uint8
                elif dtype == "bool":
                    torch_dtype = torch.bool
                else:
                    torch_dtype = None
            else:
                torch_dtype = dtype
        else:
            torch_dtype = None
        
        # Konvertiere Daten zu PyTorch-Tensor
        if isinstance(data, torch.Tensor):
            self._data = data.to(device=device)
            if torch_dtype is not None:
                self._data = self._data.to(dtype=torch_dtype)
        elif isinstance(data, np.ndarray):
            self._data = torch.from_numpy(data).to(device=device)
            if torch_dtype is not None:
                self._data = self._data.to(dtype=torch_dtype)
        elif isinstance(data, (list, tuple)):
            self._data = torch.tensor(data, dtype=torch_dtype, device=device)
        elif isinstance(data, (int, float, bool)):
            self._data = torch.tensor([data], dtype=torch_dtype, device=device)
        elif isinstance(data, MISOTensor):
            if isinstance(data, TorchTensor):
                self._data = data.data.to(device=device)
                if torch_dtype is not None:
                    self._data = self._data.to(dtype=torch_dtype)
            else:
                self._data = torch.tensor(data.to_numpy(), dtype=torch_dtype, device=device)
        else:
            raise TypeError(f"Datentyp {type(data)} wird nicht unterstützt")
        
        # Setze Datentyp
        self._dtype = str(self._data.dtype).split(".")[-1]
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Gibt die Form des Tensors zurück"""
        return tuple(self._data.shape)
    
    @property
    def device(self) -> str:
        """Gibt das Gerät des Tensors zurück"""
        return self._device
    
    def to_numpy(self) -> np.ndarray:
        """
        Konvertiert den Tensor zu einem NumPy-Array
        
        Returns:
            NumPy-Array
        """
        # Verschiebe Tensor auf CPU, falls er auf einem anderen Gerät ist
        if self._data.device.type != "cpu":
            data = self._data.detach().cpu().numpy()
        else:
            data = self._data.detach().numpy()
        return data
    
    def clone(self) -> 'TorchTensor':
        """
        Erstellt eine Kopie des Tensors
        
        Returns:
            Kopie des Tensors
        """
        return TorchTensor(self._data.clone(), self._dtype, self._device)
    
    def reshape(self, *shape) -> 'TorchTensor':
        """
        Ändert die Form des Tensors
        
        Args:
            shape: Neue Form
            
        Returns:
            Tensor mit neuer Form
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return TorchTensor(self._data.reshape(shape), self._dtype, self._device)
    
    def transpose(self, *dims) -> 'TorchTensor':
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
        
        # Wenn nur zwei Dimensionen angegeben sind, verwende transpose
        if len(dims) == 2:
            return TorchTensor(self._data.transpose(dims[0], dims[1]), self._dtype, self._device)
        else:
            # Ansonsten verwende permute
            return TorchTensor(self._data.permute(dims), self._dtype, self._device)
    
    def squeeze(self, dim: Optional[int] = None) -> 'TorchTensor':
        """
        Entfernt Dimensionen mit Größe 1
        
        Args:
            dim: Dimension, die entfernt werden soll (optional)
            
        Returns:
            Tensor mit entfernten Dimensionen
        """
        if dim is None:
            return TorchTensor(self._data.squeeze(), self._dtype, self._device)
        else:
            return TorchTensor(self._data.squeeze(dim), self._dtype, self._device)
    
    def unsqueeze(self, dim: int) -> 'TorchTensor':
        """
        Fügt eine Dimension mit Größe 1 hinzu
        
        Args:
            dim: Position, an der die Dimension hinzugefügt werden soll
            
        Returns:
            Tensor mit hinzugefügter Dimension
        """
        return TorchTensor(self._data.unsqueeze(dim), self._dtype, self._device)
    
    def __add__(self, other) -> 'TorchTensor':
        """
        Addition
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Addition
        """
        if isinstance(other, TorchTensor):
            return TorchTensor(self._data + other.data, self._dtype, self._device)
        else:
            return TorchTensor(self._data + other, self._dtype, self._device)
    
    def __sub__(self, other) -> 'TorchTensor':
        """
        Subtraktion
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Subtraktion
        """
        if isinstance(other, TorchTensor):
            return TorchTensor(self._data - other.data, self._dtype, self._device)
        else:
            return TorchTensor(self._data - other, self._dtype, self._device)
    
    def __rsub__(self, other) -> 'TorchTensor':
        """
        Rechte Subtraktion
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Subtraktion
        """
        if isinstance(other, TorchTensor):
            return TorchTensor(other.data - self._data, self._dtype, self._device)
        else:
            return TorchTensor(other - self._data, self._dtype, self._device)
    
    def __mul__(self, other) -> 'TorchTensor':
        """
        Multiplikation (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Multiplikation
        """
        if isinstance(other, TorchTensor):
            return TorchTensor(self._data * other.data, self._dtype, self._device)
        else:
            return TorchTensor(self._data * other, self._dtype, self._device)
    
    def __truediv__(self, other) -> 'TorchTensor':
        """
        Division (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Division
        """
        if isinstance(other, TorchTensor):
            return TorchTensor(self._data / other.data, self._dtype, self._device)
        else:
            return TorchTensor(self._data / other, self._dtype, self._device)
    
    def __rtruediv__(self, other) -> 'TorchTensor':
        """
        Rechte Division (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Division
        """
        if isinstance(other, TorchTensor):
            return TorchTensor(other.data / self._data, self._dtype, self._device)
        else:
            return TorchTensor(other / self._data, self._dtype, self._device)
    
    def __pow__(self, other) -> 'TorchTensor':
        """
        Potenzierung (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Potenzierung
        """
        if isinstance(other, TorchTensor):
            return TorchTensor(self._data ** other.data, self._dtype, self._device)
        else:
            return TorchTensor(self._data ** other, self._dtype, self._device)
    
    def __rpow__(self, other) -> 'TorchTensor':
        """
        Rechte Potenzierung (elementweise)
        
        Args:
            other: Tensor oder Skalar
            
        Returns:
            Ergebnis der Potenzierung
        """
        if isinstance(other, TorchTensor):
            return TorchTensor(other.data ** self._data, self._dtype, self._device)
        else:
            return TorchTensor(other ** self._data, self._dtype, self._device)
    
    def __matmul__(self, other) -> 'TorchTensor':
        """
        Matrix-Multiplikation
        
        Args:
            other: Tensor
            
        Returns:
            Ergebnis der Matrix-Multiplikation
        """
        if isinstance(other, TorchTensor):
            return TorchTensor(self._data @ other.data, self._dtype, self._device)
        else:
            if isinstance(other, (list, tuple, np.ndarray)):
                other_tensor = torch.tensor(other, device=self._device)
                return TorchTensor(self._data @ other_tensor, self._dtype, self._device)
            else:
                raise TypeError(f"Matrix-Multiplikation mit Typ {type(other)} wird nicht unterstützt")
    
    def __rmatmul__(self, other) -> 'TorchTensor':
        """
        Rechte Matrix-Multiplikation
        
        Args:
            other: Tensor
            
        Returns:
            Ergebnis der Matrix-Multiplikation
        """
        if isinstance(other, TorchTensor):
            return TorchTensor(other.data @ self._data, self._dtype, self._device)
        else:
            if isinstance(other, (list, tuple, np.ndarray)):
                other_tensor = torch.tensor(other, device=self._device)
                return TorchTensor(other_tensor @ self._data, self._dtype, self._device)
            else:
                raise TypeError(f"Matrix-Multiplikation mit Typ {type(other)} wird nicht unterstützt")
    
    def __neg__(self) -> 'TorchTensor':
        """
        Negation
        
        Returns:
            Negierter Tensor
        """
        return TorchTensor(-self._data, self._dtype, self._device)
    
    def __abs__(self) -> 'TorchTensor':
        """
        Absolutwert (elementweise)
        
        Returns:
            Tensor mit Absolutwerten
        """
        return TorchTensor(torch.abs(self._data), self._dtype, self._device)
    
    def __getitem__(self, idx) -> 'TorchTensor':
        """
        Indexierung
        
        Args:
            idx: Index oder Slice
            
        Returns:
            Teilmenge des Tensors
        """
        return TorchTensor(self._data[idx], self._dtype, self._device)
    
    def __setitem__(self, idx, value):
        """
        Indexzuweisung
        
        Args:
            idx: Index oder Slice
            value: Wert, der zugewiesen werden soll
        """
        if isinstance(value, TorchTensor):
            self._data[idx] = value.data
        else:
            self._data[idx] = value
    
    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> 'TorchTensor':
        """
        Berechnet die Summe entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der summiert wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Summe
        """
        if dim is None:
            return TorchTensor(torch.sum(self._data, keepdim=keepdim), self._dtype, self._device)
        else:
            return TorchTensor(torch.sum(self._data, dim=dim, keepdim=keepdim), self._dtype, self._device)
    
    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> 'TorchTensor':
        """
        Berechnet den Mittelwert entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der gemittelt wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Mittelwert
        """
        if dim is None:
            return TorchTensor(torch.mean(self._data, keepdim=keepdim), self._dtype, self._device)
        else:
            return TorchTensor(torch.mean(self._data, dim=dim, keepdim=keepdim), self._dtype, self._device)
    
    def var(self, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True) -> 'TorchTensor':
        """
        Berechnet die Varianz entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Varianz berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            unbiased: Ob die unverzerrte Schätzung verwendet werden soll
            
        Returns:
            Varianz
        """
        if dim is None:
            return TorchTensor(torch.var(self._data, unbiased=unbiased, keepdim=keepdim), self._dtype, self._device)
        else:
            return TorchTensor(torch.var(self._data, dim=dim, unbiased=unbiased, keepdim=keepdim), self._dtype, self._device)
    
    def std(self, dim: Optional[int] = None, keepdim: bool = False, unbiased: bool = True) -> 'TorchTensor':
        """
        Berechnet die Standardabweichung entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Standardabweichung berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            unbiased: Ob die unverzerrte Schätzung verwendet werden soll
            
        Returns:
            Standardabweichung
        """
        if dim is None:
            return TorchTensor(torch.std(self._data, unbiased=unbiased, keepdim=keepdim), self._dtype, self._device)
        else:
            return TorchTensor(torch.std(self._data, dim=dim, unbiased=unbiased, keepdim=keepdim), self._dtype, self._device)
    
    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Union['TorchTensor', Tuple['TorchTensor', 'TorchTensor']]:
        """
        Berechnet das Maximum entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der das Maximum berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Maximum und Indizes, wenn dim angegeben ist, sonst nur Maximum
        """
        if dim is None:
            return TorchTensor(torch.max(self._data), self._dtype, self._device)
        else:
            values, indices = torch.max(self._data, dim=dim, keepdim=keepdim)
            return TorchTensor(values, self._dtype, self._device), TorchTensor(indices, "int64", self._device)
    
    def min(self, dim: Optional[int] = None, keepdim: bool = False) -> Union['TorchTensor', Tuple['TorchTensor', 'TorchTensor']]:
        """
        Berechnet das Minimum entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der das Minimum berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Minimum und Indizes, wenn dim angegeben ist, sonst nur Minimum
        """
        if dim is None:
            return TorchTensor(torch.min(self._data), self._dtype, self._device)
        else:
            values, indices = torch.min(self._data, dim=dim, keepdim=keepdim)
            return TorchTensor(values, self._dtype, self._device), TorchTensor(indices, "int64", self._device)
    
    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> 'TorchTensor':
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
            return TorchTensor(torch.argmax(self._data.flatten()), "int64", self._device)
        else:
            return TorchTensor(torch.argmax(self._data, dim=dim, keepdim=keepdim), "int64", self._device)
    
    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> 'TorchTensor':
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
            return TorchTensor(torch.argmin(self._data.flatten()), "int64", self._device)
        else:
            return TorchTensor(torch.argmin(self._data, dim=dim, keepdim=keepdim), "int64", self._device)
    
    def norm(self, p: int = 2, dim: Optional[int] = None, keepdim: bool = False) -> 'TorchTensor':
        """
        Berechnet die Norm entlang einer Dimension
        
        Args:
            p: Ordnung der Norm
            dim: Dimension, entlang der die Norm berechnet wird
            keepdim: Ob die Dimension beibehalten werden soll
            
        Returns:
            Norm
        """
        if dim is None:
            return TorchTensor(torch.norm(self._data, p=p, keepdim=keepdim), self._dtype, self._device)
        else:
            return TorchTensor(torch.norm(self._data, p=p, dim=dim, keepdim=keepdim), self._dtype, self._device)
    
    def normalize(self, p: int = 2, dim: int = 0) -> 'TorchTensor':
        """
        Normalisiert den Tensor entlang einer Dimension
        
        Args:
            p: Ordnung der Norm
            dim: Dimension, entlang der normalisiert wird
            
        Returns:
            Normalisierter Tensor
        """
        return TorchTensor(torch.nn.functional.normalize(self._data, p=p, dim=dim), self._dtype, self._device)
    
    def exp(self) -> 'TorchTensor':
        """
        Berechnet die Exponentialfunktion (elementweise)
        
        Returns:
            Tensor mit Exponentialwerten
        """
        return TorchTensor(torch.exp(self._data), self._dtype, self._device)
    
    def log(self) -> 'TorchTensor':
        """
        Berechnet den natürlichen Logarithmus (elementweise)
        
        Returns:
            Tensor mit Logarithmuswerten
        """
        return TorchTensor(torch.log(self._data), self._dtype, self._device)
    
    def sin(self) -> 'TorchTensor':
        """
        Berechnet den Sinus (elementweise)
        
        Returns:
            Tensor mit Sinuswerten
        """
        return TorchTensor(torch.sin(self._data), self._dtype, self._device)
    
    def cos(self) -> 'TorchTensor':
        """
        Berechnet den Kosinus (elementweise)
        
        Returns:
            Tensor mit Kosinuswerten
        """
        return TorchTensor(torch.cos(self._data), self._dtype, self._device)
    
    def tan(self) -> 'TorchTensor':
        """
        Berechnet den Tangens (elementweise)
        
        Returns:
            Tensor mit Tangenswerten
        """
        return TorchTensor(torch.tan(self._data), self._dtype, self._device)
    
    def sinh(self) -> 'TorchTensor':
        """
        Berechnet den Sinus hyperbolicus (elementweise)
        
        Returns:
            Tensor mit Sinus hyperbolicus-Werten
        """
        return TorchTensor(torch.sinh(self._data), self._dtype, self._device)
    
    def cosh(self) -> 'TorchTensor':
        """
        Berechnet den Kosinus hyperbolicus (elementweise)
        
        Returns:
            Tensor mit Kosinus hyperbolicus-Werten
        """
        return TorchTensor(torch.cosh(self._data), self._dtype, self._device)
    
    def tanh(self) -> 'TorchTensor':
        """
        Berechnet den Tangens hyperbolicus (elementweise)
        
        Returns:
            Tensor mit Tangens hyperbolicus-Werten
        """
        return TorchTensor(torch.tanh(self._data), self._dtype, self._device)
    
    def sigmoid(self) -> 'TorchTensor':
        """
        Berechnet die Sigmoid-Funktion (elementweise)
        
        Returns:
            Tensor mit Sigmoid-Werten
        """
        return TorchTensor(torch.sigmoid(self._data), self._dtype, self._device)
    
    def relu(self) -> 'TorchTensor':
        """
        Berechnet die ReLU-Funktion (elementweise)
        
        Returns:
            Tensor mit ReLU-Werten
        """
        return TorchTensor(torch.relu(self._data), self._dtype, self._device)
    
    def softmax(self, dim: int = -1) -> 'TorchTensor':
        """
        Berechnet die Softmax-Funktion entlang einer Dimension
        
        Args:
            dim: Dimension, entlang der die Softmax-Funktion berechnet wird
            
        Returns:
            Tensor mit Softmax-Werten
        """
        return TorchTensor(torch.softmax(self._data, dim=dim), self._dtype, self._device)
    
    def to_device(self, device: str) -> 'TorchTensor':
        """
        Verschiebt den Tensor auf ein anderes Gerät
        
        Args:
            device: Zielgerät (z.B. "cpu", "mps", "cuda")
            
        Returns:
            Tensor auf dem Zielgerät
        """
        return TorchTensor(self._data.to(device=device), self._dtype, device)
    
    def to_dtype(self, dtype: str) -> 'TorchTensor':
        """
        Konvertiert den Tensor zu einem anderen Datentyp
        
        Args:
            dtype: Zieldatentyp
            
        Returns:
            Tensor mit dem Zieldatentyp
        """
        # Konvertiere String-Datentyp zu PyTorch-Datentyp
        if dtype == "float32" or dtype == "float":
            torch_dtype = torch.float32
        elif dtype == "float64" or dtype == "double":
            torch_dtype = torch.float64
        elif dtype == "float16" or dtype == "half":
            torch_dtype = torch.float16
        elif dtype == "int32" or dtype == "int":
            torch_dtype = torch.int32
        elif dtype == "int64" or dtype == "long":
            torch_dtype = torch.int64
        elif dtype == "int16" or dtype == "short":
            torch_dtype = torch.int16
        elif dtype == "int8":
            torch_dtype = torch.int8
        elif dtype == "uint8":
            torch_dtype = torch.uint8
        elif dtype == "bool":
            torch_dtype = torch.bool
        else:
            raise ValueError(f"Unbekannter Datentyp: {dtype}")
        
        return TorchTensor(self._data.to(dtype=torch_dtype), dtype, self._device)
