#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Standard Library

Dieser Modul implementiert die Standardbibliothek für die M-CODE Programmiersprache.
Die Standardbibliothek stellt grundlegende Funktionen und Datentypen für M-CODE bereit.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import math
import time
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Union, Set, Callable

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.stdlib")


class MCodeTensor:
    """Tensor-Klasse für M-CODE"""
    
    def __init__(self, data, device=None):
        """
        Initialisiert einen neuen Tensor.
        
        Args:
            data: Daten für den Tensor
            device: Gerät für den Tensor (CPU, CUDA, MPS)
        """
        # Konvertiere Daten in PyTorch-Tensor
        if isinstance(data, torch.Tensor):
            self.data = data
        elif isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data)
        elif isinstance(data, (list, tuple)):
            self.data = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, (int, float)):
            self.data = torch.tensor([data], dtype=torch.float32)
        else:
            raise TypeError(f"Ungültiger Datentyp für Tensor: {type(data)}")
        
        # Verschiebe Tensor auf Gerät
        if device is not None:
            self.data = self.data.to(device)
    
    def __repr__(self) -> str:
        """String-Repräsentation des Tensors"""
        return f"MCodeTensor({self.data})"
    
    def __add__(self, other):
        """Addition"""
        if isinstance(other, MCodeTensor):
            return MCodeTensor(self.data + other.data)
        else:
            return MCodeTensor(self.data + other)
    
    def __sub__(self, other):
        """Subtraktion"""
        if isinstance(other, MCodeTensor):
            return MCodeTensor(self.data - other.data)
        else:
            return MCodeTensor(self.data - other)
    
    def __mul__(self, other):
        """Multiplikation"""
        if isinstance(other, MCodeTensor):
            return MCodeTensor(self.data * other.data)
        else:
            return MCodeTensor(self.data * other)
    
    def __truediv__(self, other):
        """Division"""
        if isinstance(other, MCodeTensor):
            return MCodeTensor(self.data / other.data)
        else:
            return MCodeTensor(self.data / other)
    
    def __matmul__(self, other):
        """Matrix-Multiplikation"""
        if isinstance(other, MCodeTensor):
            return MCodeTensor(self.data @ other.data)
        else:
            return MCodeTensor(self.data @ other)
    
    def __pow__(self, other):
        """Potenzierung"""
        if isinstance(other, MCodeTensor):
            return MCodeTensor(self.data ** other.data)
        else:
            return MCodeTensor(self.data ** other)
    
    def __getitem__(self, idx):
        """Indexierung"""
        return MCodeTensor(self.data[idx])
    
    def __setitem__(self, idx, value):
        """Indexzuweisung"""
        if isinstance(value, MCodeTensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value
    
    @property
    def shape(self) -> torch.Size:
        """Form des Tensors"""
        return self.data.shape
    
    @property
    def dtype(self) -> torch.dtype:
        """Datentyp des Tensors"""
        return self.data.dtype
    
    @property
    def device(self) -> torch.device:
        """Gerät des Tensors"""
        return self.data.device
    
    def to(self, device) -> 'MCodeTensor':
        """
        Verschiebt den Tensor auf ein anderes Gerät.
        
        Args:
            device: Zielgerät
            
        Returns:
            Verschobener Tensor
        """
        return MCodeTensor(self.data.to(device))
    
    def cpu(self) -> 'MCodeTensor':
        """
        Verschiebt den Tensor auf die CPU.
        
        Returns:
            Tensor auf CPU
        """
        return MCodeTensor(self.data.cpu())
    
    def numpy(self) -> np.ndarray:
        """
        Konvertiert den Tensor in ein NumPy-Array.
        
        Returns:
            NumPy-Array
        """
        return self.data.cpu().numpy()
    
    def item(self) -> Union[int, float]:
        """
        Gibt den Wert eines skalaren Tensors zurück.
        
        Returns:
            Skalarer Wert
            
        Raises:
            ValueError: Wenn der Tensor nicht skalar ist
        """
        return self.data.item()
    
    def reshape(self, *shape) -> 'MCodeTensor':
        """
        Ändert die Form des Tensors.
        
        Args:
            shape: Neue Form
            
        Returns:
            Umgeformter Tensor
        """
        return MCodeTensor(self.data.reshape(*shape))
    
    def transpose(self, dim0, dim1) -> 'MCodeTensor':
        """
        Transponiert den Tensor.
        
        Args:
            dim0: Erste Dimension
            dim1: Zweite Dimension
            
        Returns:
            Transponierter Tensor
        """
        return MCodeTensor(self.data.transpose(dim0, dim1))
    
    @property
    def T(self) -> 'MCodeTensor':
        """Transponierter Tensor"""
        return MCodeTensor(self.data.T)
    
    def dot(self, other) -> 'MCodeTensor':
        """
        Berechnet das Punktprodukt.
        
        Args:
            other: Anderer Tensor
            
        Returns:
            Punktprodukt
        """
        if isinstance(other, MCodeTensor):
            return MCodeTensor(torch.dot(self.data.view(-1), other.data.view(-1)))
        else:
            return MCodeTensor(torch.dot(self.data.view(-1), torch.tensor(other).view(-1)))
    
    def norm(self) -> float:
        """
        Berechnet die Norm des Tensors.
        
        Returns:
            Norm
        """
        return torch.norm(self.data).item()
    
    def normalize(self) -> 'MCodeTensor':
        """
        Normalisiert den Tensor.
        
        Returns:
            Normalisierter Tensor
        """
        norm = self.norm()
        if norm > 0:
            return self / norm
        else:
            return self


class MCodeStdLib:
    """Standardbibliothek für M-CODE"""
    
    def __init__(self, device=None):
        """
        Initialisiert eine neue Standardbibliothek.
        
        Args:
            device: Standardgerät für Tensoren
        """
        self.device = device
        
        # Initialisiere Funktionen
        self.functions = self._initialize_functions()
        
        # Initialisiere Konstanten
        self.constants = self._initialize_constants()
        
        logger.info(f"M-CODE Standardbibliothek initialisiert: Device={device}")
    
    def _initialize_functions(self) -> Dict[str, Callable]:
        """
        Initialisiert Funktionen.
        
        Returns:
            Dictionary mit Funktionen
        """
        functions = {
            # Tensor-Erstellung
            "tensor": self.tensor,
            "zeros": self.zeros,
            "ones": self.ones,
            "eye": self.eye,
            "randn": self.randn,
            "arange": self.arange,
            "linspace": self.linspace,
            
            # Tensor-Operationen
            "reshape": self.reshape,
            "transpose": self.transpose,
            "dot": self.dot,
            "norm": self.norm,
            "normalize": self.normalize,
            
            # Mathematische Funktionen
            "sin": self.sin,
            "cos": self.cos,
            "tan": self.tan,
            "exp": self.exp,
            "log": self.log,
            "sqrt": self.sqrt,
            "pow": self.pow,
            "abs": self.abs,
            "sum": self.sum,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "argmin": self.argmin,
            "argmax": self.argmax,
            
            # Neuronale Netzwerkfunktionen
            "relu": self.relu,
            "sigmoid": self.sigmoid,
            "tanh": self.tanh,
            "softmax": self.softmax,
            
            # Hilfsfunktionen
            "print": self.print,
            "time": self.get_time,
            "sleep": self.sleep
        }
        
        return functions
    
    def _initialize_constants(self) -> Dict[str, Any]:
        """
        Initialisiert Konstanten.
        
        Returns:
            Dictionary mit Konstanten
        """
        constants = {
            "PI": math.pi,
            "E": math.e,
            "INF": float("inf"),
            "NAN": float("nan"),
            "TRUE": True,
            "FALSE": False,
            "NONE": None
        }
        
        return constants
    
    def get_function(self, name: str) -> Optional[Callable]:
        """
        Gibt eine Funktion zurück.
        
        Args:
            name: Name der Funktion
            
        Returns:
            Funktion oder None
        """
        return self.functions.get(name)
    
    def get_constant(self, name: str) -> Optional[Any]:
        """
        Gibt eine Konstante zurück.
        
        Args:
            name: Name der Konstante
            
        Returns:
            Konstante oder None
        """
        return self.constants.get(name)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Gibt alle Funktionen und Konstanten zurück.
        
        Returns:
            Dictionary mit allen Funktionen und Konstanten
        """
        all_items = {}
        all_items.update(self.functions)
        all_items.update(self.constants)
        return all_items
    
    # Tensor-Erstellung
    
    def tensor(self, data, dtype=None) -> MCodeTensor:
        """
        Erstellt einen Tensor.
        
        Args:
            data: Daten für den Tensor
            dtype: Datentyp für den Tensor
            
        Returns:
            Tensor
        """
        if dtype is None:
            return MCodeTensor(data, self.device)
        else:
            tensor_data = torch.tensor(data, dtype=dtype)
            return MCodeTensor(tensor_data, self.device)
    
    def zeros(self, *shape) -> MCodeTensor:
        """
        Erstellt einen Tensor mit Nullen.
        
        Args:
            shape: Form des Tensors
            
        Returns:
            Tensor mit Nullen
        """
        return MCodeTensor(torch.zeros(*shape), self.device)
    
    def ones(self, *shape) -> MCodeTensor:
        """
        Erstellt einen Tensor mit Einsen.
        
        Args:
            shape: Form des Tensors
            
        Returns:
            Tensor mit Einsen
        """
        return MCodeTensor(torch.ones(*shape), self.device)
    
    def eye(self, n, m=None) -> MCodeTensor:
        """
        Erstellt eine Einheitsmatrix.
        
        Args:
            n: Anzahl der Zeilen
            m: Anzahl der Spalten (optional)
            
        Returns:
            Einheitsmatrix
        """
        return MCodeTensor(torch.eye(n, m), self.device)
    
    def randn(self, *shape) -> MCodeTensor:
        """
        Erstellt einen Tensor mit normalverteilten Zufallszahlen.
        
        Args:
            shape: Form des Tensors
            
        Returns:
            Tensor mit Zufallszahlen
        """
        return MCodeTensor(torch.randn(*shape), self.device)
    
    def arange(self, start, end=None, step=1) -> MCodeTensor:
        """
        Erstellt einen Tensor mit einer Zahlenfolge.
        
        Args:
            start: Startwert
            end: Endwert (exklusiv)
            step: Schrittweite
            
        Returns:
            Tensor mit Zahlenfolge
        """
        if end is None:
            end = start
            start = 0
        return MCodeTensor(torch.arange(start, end, step), self.device)
    
    def linspace(self, start, end, steps) -> MCodeTensor:
        """
        Erstellt einen Tensor mit linear interpolierten Werten.
        
        Args:
            start: Startwert
            end: Endwert
            steps: Anzahl der Schritte
            
        Returns:
            Tensor mit linear interpolierten Werten
        """
        return MCodeTensor(torch.linspace(start, end, steps), self.device)
    
    # Tensor-Operationen
    
    def reshape(self, tensor, *shape) -> MCodeTensor:
        """
        Ändert die Form eines Tensors.
        
        Args:
            tensor: Tensor
            shape: Neue Form
            
        Returns:
            Umgeformter Tensor
        """
        if isinstance(tensor, MCodeTensor):
            return tensor.reshape(*shape)
        else:
            return MCodeTensor(tensor, self.device).reshape(*shape)
    
    def transpose(self, tensor, dim0=0, dim1=1) -> MCodeTensor:
        """
        Transponiert einen Tensor.
        
        Args:
            tensor: Tensor
            dim0: Erste Dimension
            dim1: Zweite Dimension
            
        Returns:
            Transponierter Tensor
        """
        if isinstance(tensor, MCodeTensor):
            return tensor.transpose(dim0, dim1)
        else:
            return MCodeTensor(tensor, self.device).transpose(dim0, dim1)
    
    def dot(self, a, b) -> MCodeTensor:
        """
        Berechnet das Punktprodukt.
        
        Args:
            a: Erster Tensor
            b: Zweiter Tensor
            
        Returns:
            Punktprodukt
        """
        if isinstance(a, MCodeTensor):
            return a.dot(b)
        else:
            return MCodeTensor(a, self.device).dot(b)
    
    def norm(self, tensor) -> float:
        """
        Berechnet die Norm eines Tensors.
        
        Args:
            tensor: Tensor
            
        Returns:
            Norm
        """
        if isinstance(tensor, MCodeTensor):
            return tensor.norm()
        else:
            return MCodeTensor(tensor, self.device).norm()
    
    def normalize(self, tensor) -> MCodeTensor:
        """
        Normalisiert einen Tensor.
        
        Args:
            tensor: Tensor
            
        Returns:
            Normalisierter Tensor
        """
        if isinstance(tensor, MCodeTensor):
            return tensor.normalize()
        else:
            return MCodeTensor(tensor, self.device).normalize()
    
    # Mathematische Funktionen
    
    def sin(self, x) -> MCodeTensor:
        """
        Berechnet den Sinus.
        
        Args:
            x: Tensor oder Skalar
            
        Returns:
            Sinus
        """
        if isinstance(x, MCodeTensor):
            return MCodeTensor(torch.sin(x.data), self.device)
        else:
            return MCodeTensor(torch.sin(torch.tensor(x, dtype=torch.float32)), self.device)
    
    def cos(self, x) -> MCodeTensor:
        """
        Berechnet den Kosinus.
        
        Args:
            x: Tensor oder Skalar
            
        Returns:
            Kosinus
        """
        if isinstance(x, MCodeTensor):
            return MCodeTensor(torch.cos(x.data), self.device)
        else:
            return MCodeTensor(torch.cos(torch.tensor(x, dtype=torch.float32)), self.device)
    
    def tan(self, x) -> MCodeTensor:
        """
        Berechnet den Tangens.
        
        Args:
            x: Tensor oder Skalar
            
        Returns:
            Tangens
        """
        if isinstance(x, MCodeTensor):
            return MCodeTensor(torch.tan(x.data), self.device)
        else:
            return MCodeTensor(torch.tan(torch.tensor(x, dtype=torch.float32)), self.device)
    
    def exp(self, x) -> MCodeTensor:
        """
        Berechnet die Exponentialfunktion.
        
        Args:
            x: Tensor oder Skalar
            
        Returns:
            Exponentialfunktion
        """
        if isinstance(x, MCodeTensor):
            return MCodeTensor(torch.exp(x.data), self.device)
        else:
            return MCodeTensor(torch.exp(torch.tensor(x, dtype=torch.float32)), self.device)
    
    def log(self, x) -> MCodeTensor:
        """
        Berechnet den natürlichen Logarithmus.
        
        Args:
            x: Tensor oder Skalar
            
        Returns:
            Natürlicher Logarithmus
        """
        if isinstance(x, MCodeTensor):
            return MCodeTensor(torch.log(x.data), self.device)
        else:
            return MCodeTensor(torch.log(torch.tensor(x, dtype=torch.float32)), self.device)
    
    def sqrt(self, x) -> MCodeTensor:
        """
        Berechnet die Quadratwurzel.
        
        Args:
            x: Tensor oder Skalar
            
        Returns:
            Quadratwurzel
        """
        if isinstance(x, MCodeTensor):
            return MCodeTensor(torch.sqrt(x.data), self.device)
        else:
            return MCodeTensor(torch.sqrt(torch.tensor(x, dtype=torch.float32)), self.device)
    
    def pow(self, x, y) -> MCodeTensor:
        """
        Berechnet die Potenz.
        
        Args:
            x: Basis
            y: Exponent
            
        Returns:
            Potenz
        """
        if isinstance(x, MCodeTensor):
            if isinstance(y, MCodeTensor):
                return MCodeTensor(torch.pow(x.data, y.data), self.device)
            else:
                return MCodeTensor(torch.pow(x.data, y), self.device)
        else:
            if isinstance(y, MCodeTensor):
                return MCodeTensor(torch.pow(torch.tensor(x, dtype=torch.float32), y.data), self.device)
            else:
                return MCodeTensor(torch.pow(torch.tensor(x, dtype=torch.float32), y), self.device)
    
    def abs(self, x) -> MCodeTensor:
        """
        Berechnet den Absolutbetrag.
        
        Args:
            x: Tensor oder Skalar
            
        Returns:
            Absolutbetrag
        """
        if isinstance(x, MCodeTensor):
            return MCodeTensor(torch.abs(x.data), self.device)
        else:
            return MCodeTensor(torch.abs(torch.tensor(x, dtype=torch.float32)), self.device)
    
    def sum(self, x, dim=None) -> MCodeTensor:
        """
        Berechnet die Summe.
        
        Args:
            x: Tensor
            dim: Dimension (optional)
            
        Returns:
            Summe
        """
        if isinstance(x, MCodeTensor):
            if dim is None:
                return MCodeTensor(torch.sum(x.data), self.device)
            else:
                return MCodeTensor(torch.sum(x.data, dim=dim), self.device)
        else:
            if dim is None:
                return MCodeTensor(torch.sum(torch.tensor(x, dtype=torch.float32)), self.device)
            else:
                return MCodeTensor(torch.sum(torch.tensor(x, dtype=torch.float32), dim=dim), self.device)
    
    def mean(self, x, dim=None) -> MCodeTensor:
        """
        Berechnet den Mittelwert.
        
        Args:
            x: Tensor
            dim: Dimension (optional)
            
        Returns:
            Mittelwert
        """
        if isinstance(x, MCodeTensor):
            if dim is None:
                return MCodeTensor(torch.mean(x.data), self.device)
            else:
                return MCodeTensor(torch.mean(x.data, dim=dim), self.device)
        else:
            if dim is None:
                return MCodeTensor(torch.mean(torch.tensor(x, dtype=torch.float32)), self.device)
            else:
                return MCodeTensor(torch.mean(torch.tensor(x, dtype=torch.float32), dim=dim), self.device)
    
    def std(self, x, dim=None) -> MCodeTensor:
        """
        Berechnet die Standardabweichung.
        
        Args:
            x: Tensor
            dim: Dimension (optional)
            
        Returns:
            Standardabweichung
        """
        if isinstance(x, MCodeTensor):
            if dim is None:
                return MCodeTensor(torch.std(x.data), self.device)
            else:
                return MCodeTensor(torch.std(x.data, dim=dim), self.device)
        else:
            if dim is None:
                return MCodeTensor(torch.std(torch.tensor(x, dtype=torch.float32)), self.device)
            else:
                return MCodeTensor(torch.std(torch.tensor(x, dtype=torch.float32), dim=dim), self.device)
    
    def min(self, x, dim=None) -> Union[MCodeTensor, Tuple[MCodeTensor, MCodeTensor]]:
        """
        Berechnet das Minimum.
        
        Args:
            x: Tensor
            dim: Dimension (optional)
            
        Returns:
            Minimum oder (Minimum, Indizes)
        """
        if isinstance(x, MCodeTensor):
            if dim is None:
                return MCodeTensor(torch.min(x.data), self.device)
            else:
                min_values, min_indices = torch.min(x.data, dim=dim)
                return MCodeTensor(min_values, self.device), MCodeTensor(min_indices, self.device)
        else:
            if dim is None:
                return MCodeTensor(torch.min(torch.tensor(x, dtype=torch.float32)), self.device)
            else:
                min_values, min_indices = torch.min(torch.tensor(x, dtype=torch.float32), dim=dim)
                return MCodeTensor(min_values, self.device), MCodeTensor(min_indices, self.device)
    
    def max(self, x, dim=None) -> Union[MCodeTensor, Tuple[MCodeTensor, MCodeTensor]]:
        """
        Berechnet das Maximum.
        
        Args:
            x: Tensor
            dim: Dimension (optional)
            
        Returns:
            Maximum oder (Maximum, Indizes)
        """
        if isinstance(x, MCodeTensor):
            if dim is None:
                return MCodeTensor(torch.max(x.data), self.device)
            else:
                max_values, max_indices = torch.max(x.data, dim=dim)
                return MCodeTensor(max_values, self.device), MCodeTensor(max_indices, self.device)
        else:
            if dim is None:
                return MCodeTensor(torch.max(torch.tensor(x, dtype=torch.float32)), self.device)
            else:
                max_values, max_indices = torch.max(torch.tensor(x, dtype=torch.float32), dim=dim)
                return MCodeTensor(max_values, self.device), MCodeTensor(max_indices, self.device)
    
    def argmin(self, x, dim=None) -> MCodeTensor:
        """
        Berechnet die Indizes des Minimums.
        
        Args:
            x: Tensor
            dim: Dimension (optional)
            
        Returns:
            Indizes des Minimums
        """
        if isinstance(x, MCodeTensor):
            if dim is None:
                return MCodeTensor(torch.argmin(x.data), self.device)
            else:
                return MCodeTensor(torch.argmin(x.data, dim=dim), self.device)
        else:
            if dim is None:
                return MCodeTensor(torch.argmin(torch.tensor(x, dtype=torch.float32)), self.device)
            else:
                return MCodeTensor(torch.argmin(torch.tensor(x, dtype=torch.float32), dim=dim), self.device)
    
    def argmax(self, x, dim=None) -> MCodeTensor:
        """
        Berechnet die Indizes des Maximums.
        
        Args:
            x: Tensor
            dim: Dimension (optional)
            
        Returns:
            Indizes des Maximums
        """
        if isinstance(x, MCodeTensor):
            if dim is None:
                return MCodeTensor(torch.argmax(x.data), self.device)
            else:
                return MCodeTensor(torch.argmax(x.data, dim=dim), self.device)
        else:
            if dim is None:
                return MCodeTensor(torch.argmax(torch.tensor(x, dtype=torch.float32)), self.device)
            else:
                return MCodeTensor(torch.argmax(torch.tensor(x, dtype=torch.float32), dim=dim), self.device)
    
    # Neuronale Netzwerkfunktionen
    
    def relu(self, x) -> MCodeTensor:
        """
        Berechnet die ReLU-Funktion.
        
        Args:
            x: Tensor
            
        Returns:
            ReLU-Funktion
        """
        if isinstance(x, MCodeTensor):
            return MCodeTensor(torch.relu(x.data), self.device)
        else:
            return MCodeTensor(torch.relu(torch.tensor(x, dtype=torch.float32)), self.device)
    
    def sigmoid(self, x) -> MCodeTensor:
        """
        Berechnet die Sigmoid-Funktion.
        
        Args:
            x: Tensor
            
        Returns:
            Sigmoid-Funktion
        """
        if isinstance(x, MCodeTensor):
            return MCodeTensor(torch.sigmoid(x.data), self.device)
        else:
            return MCodeTensor(torch.sigmoid(torch.tensor(x, dtype=torch.float32)), self.device)
    
    def tanh(self, x) -> MCodeTensor:
        """
        Berechnet die Tanh-Funktion.
        
        Args:
            x: Tensor
            
        Returns:
            Tanh-Funktion
        """
        if isinstance(x, MCodeTensor):
            return MCodeTensor(torch.tanh(x.data), self.device)
        else:
            return MCodeTensor(torch.tanh(torch.tensor(x, dtype=torch.float32)), self.device)
    
    def softmax(self, x, dim=None) -> MCodeTensor:
        """
        Berechnet die Softmax-Funktion.
        
        Args:
            x: Tensor
            dim: Dimension (optional)
            
        Returns:
            Softmax-Funktion
        """
        if isinstance(x, MCodeTensor):
            if dim is None:
                return MCodeTensor(torch.softmax(x.data, dim=0), self.device)
            else:
                return MCodeTensor(torch.softmax(x.data, dim=dim), self.device)
        else:
            if dim is None:
                return MCodeTensor(torch.softmax(torch.tensor(x, dtype=torch.float32), dim=0), self.device)
            else:
                return MCodeTensor(torch.softmax(torch.tensor(x, dtype=torch.float32), dim=dim), self.device)
    
    # Hilfsfunktionen
    
    def print(self, *args, **kwargs) -> None:
        """
        Gibt Werte aus.
        
        Args:
            args: Argumente
            kwargs: Schlüsselwortargumente
        """
        print(*args, **kwargs)
    
    def get_time(self) -> float:
        """
        Gibt die aktuelle Zeit zurück.
        
        Returns:
            Aktuelle Zeit in Sekunden
        """
        return time.time()
    
    def sleep(self, seconds: float) -> None:
        """
        Pausiert die Ausführung.
        
        Args:
            seconds: Anzahl der Sekunden
        """
        time.sleep(seconds)
