#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - T-Mathematics Engine Hauptmodul

Dieses Modul implementiert die Hauptklasse der T-Mathematics Engine,
die für Hochleistungs-Tensoralgebra und symbolische Mathematik optimiert ist.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.Ultimate.TMathematics")

# Importiere die Tensor-Factory und Tensor-Implementierungen
from .tensor_factory import TensorFactory, get_available_backends
from .tensor import MISOTensor

# Versuche, PyTorch mit MPS-Unterstützung zu importieren
try:
    import torch
    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
    if HAS_MPS:
        logger.info("PyTorch mit MPS-Unterstützung verfügbar")
        DEFAULT_DEVICE = torch.device("mps")
    else:
        logger.info("PyTorch verfügbar, aber MPS nicht unterstützt")
        DEFAULT_DEVICE = torch.device("cpu")
except ImportError:
    logger.warning("PyTorch nicht verfügbar, verwende NumPy für Tensor-Operationen")
    HAS_TORCH = False
    HAS_MPS = False
    DEFAULT_DEVICE = None

# Versuche, MLX zu importieren (Apple's ML Framework für Apple Silicon)
try:
    import mlx.core as mx
    HAS_MLX = True
    logger.info("Apple MLX Framework verfügbar")
except ImportError:
    logger.warning("Apple MLX nicht verfügbar")
    HAS_MLX = False

# Versuche, SymPy für symbolische Mathematik zu importieren
try:
    import sympy
    HAS_SYMPY = True
    logger.info("SymPy für symbolische Mathematik verfügbar")
except ImportError:
    logger.warning("SymPy nicht verfügbar, symbolische Mathematik deaktiviert")
    HAS_SYMPY = False


class TMathematicsEngine:
    """
    T-Mathematics Engine für MISO Ultimate
    
    Diese Klasse implementiert die Hochleistungs-Tensoralgebra und symbolische
    Mathematik für MISO Ultimate, optimiert für Apple Silicon.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert die T-Mathematics Engine
        
        Args:
            config: Konfigurationsobjekt für die T-Mathematics Engine
        """
        self.config = config
        self.precision = config.get("precision", "float32")
        self.use_symbolic = config.get("use_symbolic", True) and HAS_SYMPY
        self.max_tensor_dimensions = config.get("max_tensor_dimensions", 6)
        self.use_gpu_acceleration = config.get("use_gpu_acceleration", True) and HAS_MPS
        self.optimization_level = config.get("optimization_level", 3)
        
        # Initialisiere die Tensor-Factory
        self.tensor_factory = TensorFactory()
        
        # Wähle das optimale Backend für Tensor-Operationen
        self.backend = self._select_optimal_backend()
        self.tensor_factory.set_default_backend(self.backend)
        
        # Initialisiere Komponenten
        self._initialize_components()
        
        logger.info(f"T-Mathematics Engine initialisiert mit Backend: {self.backend}")
        logger.info(f"Präzision: {self.precision}, Symbolische Mathematik: {self.use_symbolic}")
    
    def _select_optimal_backend(self) -> str:
        """
        Wählt das optimale Backend für Tensor-Operationen basierend auf
        der Systemkonfiguration und verfügbaren Frameworks.
        
        Returns:
            Name des optimalen Backends
        """
        available_backends = get_available_backends()
        preferred_backend = self.config.get("preferred_backend", None)
        
        # Wenn ein bevorzugtes Backend konfiguriert und verfügbar ist, verwende es
        if preferred_backend and preferred_backend in available_backends:
            return preferred_backend
        
        # Ansonsten wähle das beste verfügbare Backend
        if "mlx" in available_backends and HAS_MLX:
            # MLX ist das bevorzugte Backend für Apple Silicon
            return "mlx"
        elif "torch" in available_backends and HAS_TORCH and HAS_MPS and self.use_gpu_acceleration:
            # PyTorch mit MPS ist die zweite Wahl
            return "torch"
        elif "torch" in available_backends and HAS_TORCH:
            # PyTorch ohne MPS ist die dritte Wahl
            return "torch"
        elif "numpy" in available_backends:
            # NumPy ist die letzte Wahl
            return "numpy"
        else:
            # Fallback, wenn keine bekannten Backends verfügbar sind
            if available_backends:
                return available_backends[0]
            else:
                raise RuntimeError("Keine Tensor-Backends verfügbar")
    
    def _initialize_components(self):
        """Initialisiert alle Komponenten der T-Mathematics Engine"""
        # Initialisiere symbolische Mathematik, wenn aktiviert
        if self.use_symbolic:
            from .symbolic import SymbolicMath
            self.symbolic = SymbolicMath()
    
    def create_tensor(self, data, dtype=None, backend=None) -> MISOTensor:
        """
        Erstellt einen Tensor
        
        Args:
            data: Daten für den Tensor
            dtype: Datentyp für den Tensor
            backend: Spezifisches Backend für diesen Tensor (optional)
            
        Returns:
            MISOTensor-Objekt
        """
        return self.tensor_factory.create_tensor(data, dtype=dtype or self.precision, backend=backend)
    
    def zeros(self, *shape, dtype=None, backend=None) -> MISOTensor:
        """
        Erstellt einen Tensor mit Nullen
        
        Args:
            shape: Form des Tensors
            dtype: Datentyp für den Tensor (optional)
            backend: Spezifisches Backend für diesen Tensor (optional)
            
        Returns:
            MISOTensor-Objekt
        """
        return self.tensor_factory.zeros(*shape, dtype=dtype or self.precision, backend=backend)
    
    def ones(self, *shape, dtype=None, backend=None) -> MISOTensor:
        """
        Erstellt einen Tensor mit Einsen
        
        Args:
            shape: Form des Tensors
            dtype: Datentyp für den Tensor (optional)
            backend: Spezifisches Backend für diesen Tensor (optional)
            
        Returns:
            MISOTensor-Objekt
        """
        return self.tensor_factory.ones(*shape, dtype=dtype or self.precision, backend=backend)
    
    def eye(self, n: int, m: Optional[int] = None, dtype=None, backend=None) -> MISOTensor:
        """
        Erstellt eine Einheitsmatrix
        
        Args:
            n: Anzahl der Zeilen
            m: Anzahl der Spalten (optional, standardmäßig gleich n)
            dtype: Datentyp für den Tensor (optional)
            backend: Spezifisches Backend für diesen Tensor (optional)
            
        Returns:
            MISOTensor-Objekt
        """
        return self.tensor_factory.eye(n, m, dtype=dtype or self.precision, backend=backend)
    
    def randn(self, *shape, dtype=None, backend=None) -> MISOTensor:
        """
        Erstellt einen Tensor mit normalverteilten Zufallswerten
        
        Args:
            shape: Form des Tensors
            dtype: Datentyp für den Tensor (optional)
            backend: Spezifisches Backend für diesen Tensor (optional)
            
        Returns:
            MISOTensor-Objekt
        """
        return self.tensor_factory.randn(*shape, dtype=dtype or self.precision, backend=backend)
    
    def rand(self, *shape, dtype=None, backend=None) -> MISOTensor:
        """
        Erstellt einen Tensor mit gleichverteilten Zufallswerten
        
        Args:
            shape: Form des Tensors
            
        Returns:
            MISOTensor-Objekt
        """
        return self.operations.rand(*shape)
    
    def linspace(self, start: float, end: float, steps: int) -> 'MISOTensor':
        """
        Erstellt einen Tensor mit linear verteilten Werten
        
        Args:
            start: Startwert
            end: Endwert
            steps: Anzahl der Schritte
            
        Returns:
            MISOTensor-Objekt
        """
        return self.operations.linspace(start, end, steps)
    
    def arange(self, start: float, end: float, step: float = 1.0) -> 'MISOTensor':
        """
        Erstellt einen Tensor mit Werten in regelmäßigen Abständen
        
        Args:
            start: Startwert
            end: Endwert
            step: Schrittweite
            
        Returns:
            MISOTensor-Objekt
        """
        return self.operations.arange(start, end, step)
    
    def matmul(self, a: 'MISOTensor', b: 'MISOTensor') -> 'MISOTensor':
        """
        Führt eine Matrix-Multiplikation durch
        
        Args:
            a: Erster Tensor
            b: Zweiter Tensor
            
        Returns:
            Ergebnis der Matrix-Multiplikation
        """
        return self.operations.matmul(a, b)
    
    def transpose(self, x: 'MISOTensor', dims: Optional[Tuple[int, int]] = None) -> 'MISOTensor':
        """
        Transponiert einen Tensor
        
        Args:
            x: Tensor
            dims: Dimensionen, die transponiert werden sollen
            
        Returns:
            Transponierter Tensor
        """
        return self.operations.transpose(x, dims)
    
    def norm(self, x: 'MISOTensor', p: int = 2, dim: Optional[int] = None) -> Union['MISOTensor', float]:
        """
        Berechnet die Norm eines Tensors
        
        Args:
            x: Tensor
            p: Ordnung der Norm
            dim: Dimension, entlang der die Norm berechnet wird
            
        Returns:
            Norm des Tensors
        """
        return self.operations.norm(x, p, dim)
    
    def normalize(self, x: 'MISOTensor', p: int = 2, dim: int = 0) -> 'MISOTensor':
        """
        Normalisiert einen Tensor
        
        Args:
            x: Tensor
            p: Ordnung der Norm
            dim: Dimension, entlang der normalisiert wird
            
        Returns:
            Normalisierter Tensor
        """
        return self.operations.normalize(x, p, dim)
    
    def eigen(self, x: 'MISOTensor') -> Tuple['MISOTensor', 'MISOTensor']:
        """
        Berechnet Eigenwerte und Eigenvektoren einer Matrix
        
        Args:
            x: Tensor (Matrix)
            
        Returns:
            Tupel aus Eigenwerten und Eigenvektoren
        """
        return self.operations.eigen(x)
    
    def svd(self, x: 'MISOTensor') -> Tuple['MISOTensor', 'MISOTensor', 'MISOTensor']:
        """
        Berechnet die Singulärwertzerlegung einer Matrix
        
        Args:
            x: Tensor (Matrix)
            
        Returns:
            Tupel aus U, S, V
        """
        return self.operations.svd(x)
    
    def solve(self, a: 'MISOTensor', b: 'MISOTensor') -> 'MISOTensor':
        """
        Löst ein lineares Gleichungssystem Ax = b
        
        Args:
            a: Koeffizientenmatrix A
            b: Vektor b
            
        Returns:
            Lösung x
        """
        return self.operations.solve(a, b)
    
    def inv(self, x: 'MISOTensor') -> 'MISOTensor':
        """
        Berechnet die Inverse einer Matrix
        
        Args:
            x: Tensor (Matrix)
            
        Returns:
            Inverse Matrix
        """
        return self.operations.inv(x)
    
    def det(self, x: 'MISOTensor') -> Union['MISOTensor', float]:
        """
        Berechnet die Determinante einer Matrix
        
        Args:
            x: Tensor (Matrix)
            
        Returns:
            Determinante
        """
        return self.operations.det(x)
    
    def trace(self, x: 'MISOTensor') -> Union['MISOTensor', float]:
        """
        Berechnet die Spur einer Matrix
        
        Args:
            x: Tensor (Matrix)
            
        Returns:
            Spur
        """
        return self.operations.trace(x)
    
    def diag(self, x: 'MISOTensor', k: int = 0) -> 'MISOTensor':
        """
        Extrahiert die Diagonale einer Matrix oder erstellt eine Diagonalmatrix
        
        Args:
            x: Tensor
            k: Offset der Diagonale
            
        Returns:
            Diagonale oder Diagonalmatrix
        """
        return self.operations.diag(x, k)
    
    def outer(self, a: 'MISOTensor', b: 'MISOTensor') -> 'MISOTensor':
        """
        Berechnet das äußere Produkt zweier Vektoren
        
        Args:
            a: Erster Vektor
            b: Zweiter Vektor
            
        Returns:
            Äußeres Produkt
        """
        return self.operations.outer(a, b)
    
    def kron(self, a: 'MISOTensor', b: 'MISOTensor') -> 'MISOTensor':
        """
        Berechnet das Kronecker-Produkt zweier Tensoren
        
        Args:
            a: Erster Tensor
            b: Zweiter Tensor
            
        Returns:
            Kronecker-Produkt
        """
        return self.operations.kron(a, b)
    
    def fft(self, x: 'MISOTensor', dim: int = -1) -> 'MISOTensor':
        """
        Berechnet die schnelle Fourier-Transformation
        
        Args:
            x: Tensor
            dim: Dimension, entlang der die FFT berechnet wird
            
        Returns:
            Fourier-transformierter Tensor
        """
        return self.operations.fft(x, dim)
    
    def ifft(self, x: 'MISOTensor', dim: int = -1) -> 'MISOTensor':
        """
        Berechnet die inverse schnelle Fourier-Transformation
        
        Args:
            x: Tensor
            dim: Dimension, entlang der die IFFT berechnet wird
            
        Returns:
            Invers Fourier-transformierter Tensor
        """
        return self.operations.ifft(x, dim)
    
    def conv(self, x: 'MISOTensor', kernel: 'MISOTensor', stride: int = 1, padding: int = 0) -> 'MISOTensor':
        """
        Berechnet die Faltung eines Tensors mit einem Kernel
        
        Args:
            x: Eingabetensor
            kernel: Faltungskernel
            stride: Schrittweite
            padding: Polsterung
            
        Returns:
            Gefalteter Tensor
        """
        return self.operations.conv(x, kernel, stride, padding)
    
    def benchmark(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Führt einen Benchmark für eine Operation durch
        
        Args:
            operation: Name der Operation
            *args: Argumente für die Operation
            **kwargs: Keyword-Argumente für die Operation
            
        Returns:
            Benchmark-Ergebnisse
        """
        if not hasattr(self, operation):
            raise ValueError(f"Operation {operation} nicht gefunden")
        
        # Führe Operation mehrmals aus und messe die Zeit
        op_func = getattr(self, operation)
        
        num_runs = kwargs.pop("num_runs", 10)
        warmup_runs = kwargs.pop("warmup_runs", 3)
        
        # Warmup
        for _ in range(warmup_runs):
            op_func(*args, **kwargs)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            result = op_func(*args, **kwargs)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        
        return {
            "operation": operation,
            "backend": self.backend,
            "num_runs": num_runs,
            "total_time": end_time - start_time,
            "avg_time": avg_time,
            "args": args,
            "kwargs": kwargs,
            "result_shape": getattr(result, "shape", None)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Gibt den Status der T-Mathematics Engine zurück
        
        Returns:
            Status-Informationen
        """
        return {
            "status": "active",
            "backend": self.backend,
            "precision": self.precision,
            "use_symbolic": self.use_symbolic,
            "use_gpu_acceleration": self.use_gpu_acceleration,
            "optimization_level": self.optimization_level
        }
    
    def shutdown(self):
        """Fährt die T-Mathematics Engine herunter"""
        logger.info("T-Mathematics Engine wird heruntergefahren")
        # Führe Bereinigungsoperationen durch, falls erforderlich
