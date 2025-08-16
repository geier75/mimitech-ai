#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR Math Module - Performance Optimized Implementation
Hochperformante Mathematik-Engine mit Apple Silicon Optimierung

Copyright (c) 2025 VXOR AI. Alle Rechte vorbehalten.
"""

import logging
import numpy as np
from typing import Union, Tuple, Optional, Any, List, Dict
from abc import ABC, abstractmethod
import time

# Setup logging
logger = logging.getLogger("VXOR.Math")

# Try to import MLX for Apple Silicon optimization
try:
    import mlx.core as mx
    HAS_MLX = True
    logger.info("✅ MLX verfügbar - Apple Silicon Optimierung aktiviert")
except ImportError:
    HAS_MLX = False
    logger.info("⚠️ MLX nicht verfügbar - Fallback auf NumPy")

# Try to import PyTorch for additional functionality
try:
    import torch
    HAS_TORCH = True
    logger.info("✅ PyTorch verfügbar")
except ImportError:
    HAS_TORCH = False
    logger.info("⚠️ PyTorch nicht verfügbar")

class MathBackend(ABC):
    """Abstract Math Backend"""
    
    @abstractmethod
    def create_array(self, data: Any) -> Any:
        """Erstellt Array"""
        pass
        
    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix-Multiplikation"""
        pass
        
    @abstractmethod
    def transpose(self, a: Any) -> Any:
        """Matrix-Transposition"""
        pass
        
    @abstractmethod
    def inverse(self, a: Any) -> Any:
        """Matrix-Inverse"""
        pass

class NumPyBackend(MathBackend):
    """NumPy Backend - Fallback Implementation"""
    
    def create_array(self, data: Any) -> np.ndarray:
        """Erstellt NumPy Array"""
        return np.array(data)
        
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix-Multiplikation mit NumPy"""
        return np.matmul(a, b)
        
    def transpose(self, a: np.ndarray) -> np.ndarray:
        """Matrix-Transposition mit NumPy"""
        return np.transpose(a)
        
    def inverse(self, a: np.ndarray) -> np.ndarray:
        """Matrix-Inverse mit NumPy"""
        return np.linalg.inv(a)

class MLXBackend(MathBackend):
    """MLX Backend - Apple Silicon Optimized"""
    
    def __init__(self):
        if not HAS_MLX:
            raise ImportError("MLX nicht verfügbar")
            
    def create_array(self, data: Any) -> mx.array:
        """Erstellt MLX Array"""
        return mx.array(data)
        
    def matmul(self, a: mx.array, b: mx.array) -> mx.array:
        """Matrix-Multiplikation mit MLX"""
        return mx.matmul(a, b)
        
    def transpose(self, a: mx.array) -> mx.array:
        """Matrix-Transposition mit MLX"""
        return mx.transpose(a)
        
    def inverse(self, a: mx.array) -> mx.array:
        """Matrix-Inverse mit MLX"""
        # MLX might not have direct inverse, use solve with identity
        identity = mx.eye(a.shape[0])
        return mx.linalg.solve(a, identity)

class VXORTensor:
    """
    VXOR Tensor - Unified Tensor Interface
    
    Einheitliche Schnittstelle für verschiedene Math-Backends
    mit automatischer Backend-Auswahl basierend auf Performance.
    """
    
    def __init__(self, data: Any, backend: Optional[str] = None):
        """
        Initialisiert VXOR Tensor
        
        Args:
            data: Tensor-Daten
            backend: Backend-Auswahl ('numpy', 'mlx', 'auto')
        """
        self.backend_name = backend or self._select_optimal_backend(data)
        self.backend = self._get_backend(self.backend_name)
        self.data = self.backend.create_array(data)
        self.shape = self._get_shape()
        
    def _select_optimal_backend(self, data: Any) -> str:
        """Wählt optimales Backend basierend auf Daten"""
        if not HAS_MLX:
            return "numpy"
            
        # Für große Matrizen verwende MLX (Apple Silicon optimiert)
        if hasattr(data, 'shape'):
            total_elements = np.prod(data.shape)
            if total_elements > 10000:  # Threshold für MLX
                return "mlx"
                
        return "numpy"
        
    def _get_backend(self, backend_name: str) -> MathBackend:
        """Gibt Backend-Instanz zurück"""
        if backend_name == "mlx" and HAS_MLX:
            return MLXBackend()
        else:
            return NumPyBackend()
            
    def _get_shape(self) -> Tuple[int, ...]:
        """Gibt Tensor-Shape zurück"""
        if hasattr(self.data, 'shape'):
            return tuple(self.data.shape)
        return ()
        
    def matmul(self, other: 'VXORTensor') -> 'VXORTensor':
        """Matrix-Multiplikation"""
        if self.backend_name != other.backend_name:
            # Convert to same backend
            other = other.to_backend(self.backend_name)
            
        result_data = self.backend.matmul(self.data, other.data)
        return VXORTensor(result_data, self.backend_name)
        
    def transpose(self) -> 'VXORTensor':
        """Matrix-Transposition"""
        result_data = self.backend.transpose(self.data)
        return VXORTensor(result_data, self.backend_name)
        
    def inverse(self) -> 'VXORTensor':
        """Matrix-Inverse"""
        result_data = self.backend.inverse(self.data)
        return VXORTensor(result_data, self.backend_name)
        
    def to_backend(self, backend_name: str) -> 'VXORTensor':
        """Konvertiert zu anderem Backend"""
        if backend_name == self.backend_name:
            return self
            
        # Convert to numpy first, then to target backend
        if self.backend_name == "mlx":
            numpy_data = np.array(self.data)
        else:
            numpy_data = self.data
            
        return VXORTensor(numpy_data, backend_name)
        
    def to_numpy(self) -> np.ndarray:
        """Konvertiert zu NumPy Array"""
        if self.backend_name == "numpy":
            return self.data
        else:
            return np.array(self.data)
            
    def __str__(self) -> str:
        return f"VXORTensor(shape={self.shape}, backend={self.backend_name})"
        
    def __repr__(self) -> str:
        return self.__str__()

class MathEngine:
    """
    VXOR Math Engine - High Performance Mathematics
    
    Zentrale Mathematik-Engine mit automatischer Backend-Optimierung
    und Performance-Monitoring.
    """
    
    def __init__(self, preferred_backend: Optional[str] = None):
        """
        Initialisiert Math Engine
        
        Args:
            preferred_backend: Bevorzugtes Backend ('numpy', 'mlx', 'auto')
        """
        self.preferred_backend = preferred_backend or "auto"
        self.performance_stats = {
            "operations_count": 0,
            "total_time": 0.0,
            "backend_usage": {"numpy": 0, "mlx": 0}
        }
        
        logger.info(f"VXOR Math Engine initialisiert (Backend: {self.preferred_backend})")
        
    def create_tensor(self, data: Any, backend: Optional[str] = None) -> VXORTensor:
        """
        Erstellt VXOR Tensor
        
        Args:
            data: Tensor-Daten
            backend: Backend-Auswahl
            
        Returns:
            VXOR Tensor
        """
        backend = backend or self.preferred_backend
        return VXORTensor(data, backend)
        
    def matrix_multiply(self, a: VXORTensor, b: VXORTensor) -> VXORTensor:
        """
        Hochperformante Matrix-Multiplikation
        
        Args:
            a: Erste Matrix
            b: Zweite Matrix
            
        Returns:
            Ergebnis-Matrix
        """
        start_time = time.time()
        
        try:
            result = a.matmul(b)
            
            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(a.backend_name, execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Matrix-Multiplikation fehlgeschlagen: {e}")
            raise
            
    def matrix_inverse(self, matrix: VXORTensor) -> VXORTensor:
        """
        Berechnet Matrix-Inverse
        
        Args:
            matrix: Input-Matrix
            
        Returns:
            Inverse Matrix
        """
        start_time = time.time()
        
        try:
            result = matrix.inverse()
            
            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(matrix.backend_name, execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Matrix-Inverse fehlgeschlagen: {e}")
            raise
            
    def solve_linear_system(self, A: VXORTensor, b: VXORTensor) -> VXORTensor:
        """
        Löst lineares Gleichungssystem Ax = b
        
        Args:
            A: Koeffizienten-Matrix
            b: Ergebnis-Vektor
            
        Returns:
            Lösungs-Vektor x
        """
        # Verwende Matrix-Inverse für einfache Implementierung
        # In Production würde man effizientere Methoden verwenden
        A_inv = self.matrix_inverse(A)
        return self.matrix_multiply(A_inv, b)
        
    def eigenvalues(self, matrix: VXORTensor) -> Tuple[VXORTensor, VXORTensor]:
        """
        Berechnet Eigenwerte und Eigenvektoren
        
        Args:
            matrix: Input-Matrix
            
        Returns:
            Tuple von (Eigenwerte, Eigenvektoren)
        """
        # Convert to numpy for eigenvalue computation
        numpy_matrix = matrix.to_numpy()
        eigenvals, eigenvecs = np.linalg.eig(numpy_matrix)
        
        return (
            self.create_tensor(eigenvals, matrix.backend_name),
            self.create_tensor(eigenvecs, matrix.backend_name)
        )
        
    def svd(self, matrix: VXORTensor) -> Tuple[VXORTensor, VXORTensor, VXORTensor]:
        """
        Singular Value Decomposition
        
        Args:
            matrix: Input-Matrix
            
        Returns:
            Tuple von (U, S, Vt)
        """
        # Convert to numpy for SVD
        numpy_matrix = matrix.to_numpy()
        U, S, Vt = np.linalg.svd(numpy_matrix)
        
        return (
            self.create_tensor(U, matrix.backend_name),
            self.create_tensor(S, matrix.backend_name),
            self.create_tensor(Vt, matrix.backend_name)
        )
        
    def benchmark_backends(self, matrix_size: int = 1000) -> Dict[str, float]:
        """
        Benchmarkt verschiedene Backends
        
        Args:
            matrix_size: Größe der Test-Matrix
            
        Returns:
            Benchmark-Ergebnisse
        """
        results = {}
        
        # Test data
        test_data = np.random.rand(matrix_size, matrix_size)
        
        # Test NumPy
        if True:  # NumPy ist immer verfügbar
            start_time = time.time()
            numpy_tensor = self.create_tensor(test_data, "numpy")
            numpy_result = self.matrix_multiply(numpy_tensor, numpy_tensor)
            numpy_time = time.time() - start_time
            results["numpy"] = numpy_time
            
        # Test MLX
        if HAS_MLX:
            start_time = time.time()
            mlx_tensor = self.create_tensor(test_data, "mlx")
            mlx_result = self.matrix_multiply(mlx_tensor, mlx_tensor)
            mlx_time = time.time() - start_time
            results["mlx"] = mlx_time
            
        return results
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Gibt Performance-Statistiken zurück
        
        Returns:
            Performance-Statistiken
        """
        total_ops = self.performance_stats["operations_count"]
        avg_time = (self.performance_stats["total_time"] / total_ops 
                   if total_ops > 0 else 0.0)
                   
        return {
            "total_operations": total_ops,
            "average_time_per_operation": avg_time,
            "total_time": self.performance_stats["total_time"],
            "backend_usage": self.performance_stats["backend_usage"].copy(),
            "preferred_backend": self.preferred_backend,
            "mlx_available": HAS_MLX,
            "torch_available": HAS_TORCH
        }
        
    def _update_performance_stats(self, backend: str, execution_time: float):
        """Aktualisiert Performance-Statistiken"""
        self.performance_stats["operations_count"] += 1
        self.performance_stats["total_time"] += execution_time

        # Handle 'auto' backend mapping
        actual_backend = "numpy" if backend == "auto" else backend
        if actual_backend not in self.performance_stats["backend_usage"]:
            self.performance_stats["backend_usage"][actual_backend] = 0
        self.performance_stats["backend_usage"][actual_backend] += 1

# Global Math Engine
_math_engine: Optional[MathEngine] = None

def get_math_engine(preferred_backend: Optional[str] = None) -> MathEngine:
    """
    Gibt globale Math Engine zurück
    
    Args:
        preferred_backend: Bevorzugtes Backend
        
    Returns:
        Math Engine Instanz
    """
    global _math_engine
    
    if _math_engine is None:
        _math_engine = MathEngine(preferred_backend)
        
    return _math_engine

# Convenience Functions
def create_tensor(data: Any, backend: Optional[str] = None) -> VXORTensor:
    """Erstellt VXOR Tensor"""
    return get_math_engine().create_tensor(data, backend)

def matmul(a: VXORTensor, b: VXORTensor) -> VXORTensor:
    """Matrix-Multiplikation"""
    return get_math_engine().matrix_multiply(a, b)

def inv(matrix: VXORTensor) -> VXORTensor:
    """Matrix-Inverse"""
    return get_math_engine().matrix_inverse(matrix)

def solve(A: VXORTensor, b: VXORTensor) -> VXORTensor:
    """Löst lineares Gleichungssystem"""
    return get_math_engine().solve_linear_system(A, b)

def benchmark() -> Dict[str, float]:
    """Führt Backend-Benchmark durch"""
    return get_math_engine().benchmark_backends()

def get_stats() -> Dict[str, Any]:
    """Gibt Performance-Statistiken zurück"""
    return get_math_engine().get_performance_stats()

# Initialize logging
logger.info("VXOR Math Module geladen")
if HAS_MLX:
    logger.info("🚀 Apple Silicon Optimierung verfügbar")
if HAS_TORCH:
    logger.info("🔥 PyTorch Integration verfügbar")
