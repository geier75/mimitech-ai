#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MATRIX: Optimierte SVD-Engine

Hochleistungsfähige, numerisch stabile SVD-Implementierung für VX-MATRIX.
Unterstützt verschiedene Backends (MLX, NumPy, PyTorch) mit optimierter 
Performance für verschiedene Matrixgrößen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
ZTM-Level: STRICT
"""

import os
import sys
import time
import enum
import logging
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
from pathlib import Path
from functools import lru_cache

# ZTM-Protokoll-Initialisierung
ZTM_ACTIVE = os.environ.get('MISO_ZTM_MODE', '1') == '1'
ZTM_LOG_LEVEL = os.environ.get('MISO_ZTM_LOG_LEVEL', 'INFO')

# Konfiguriere Logging
logging.basicConfig(level=getattr(logging, ZTM_LOG_LEVEL))
logger = logging.getLogger("vx_svd_engine")

def ztm_log(message: str, level: str = 'INFO', module: str = 'SVD_ENGINE'):
    """ZTM-konforme Logging-Funktion mit Audit-Trail"""
    if not ZTM_ACTIVE:
        return
    
    log_func = getattr(logger, level.lower())
    log_func(f"[ZTM:{module}] {message}")

class SVDAlgorithm(enum.Enum):
    """Algorithmus für SVD-Berechnung"""
    STANDARD = "standard"  # Standard SVD-Algorithmus
    TRUNCATED = "truncated"  # Reduzierte SVD (weniger Speicher)
    RANDOMIZED = "randomized"  # Randomisierte SVD (schneller für große Matrizen)
    QR = "qr"  # QR-basierte SVD (numerisch stabiler)
    JACOBI = "jacobi"  # Jacobi SVD (langsamer, genauer für kleine Matrizen)

class SVDBackend(enum.Enum):
    """Unterstützte Backends für SVD-Berechungen"""
    NUMPY = "numpy"
    TORCH = "torch"
    MLX = "mlx"
    SCIPY = "scipy"
    CUSTOM = "custom"  # Eigene Implementierung

class SVDEngine:
    """
    Hochperformante SVD-Engine für VX-MATRIX
    
    Diese Klasse bietet verschiedene SVD-Algorithmen mit optimierter 
    Performance für unterschiedliche Matrixgrößen und Hardware.
    """
    
    def __init__(self, backend: str = "auto", algorithm: str = "auto"):
        """
        Initialisiert die SVD-Engine
        
        Args:
            backend: Zu verwendendes Backend ('auto', 'numpy', 'torch', 'mlx', 'scipy')
            algorithm: Zu verwendender Algorithmus ('auto', 'standard', 'truncated', 'randomized', 'qr', 'jacobi')
        """
        self.available_backends = self._detect_available_backends()
        
        # Auto-Erkennung des besten Backends
        if backend == "auto":
            self.backend = self._select_best_backend()
        else:
            self.backend = SVDBackend(backend) if backend in [b.value for b in SVDBackend] else SVDBackend.NUMPY
            
        # Auto-Erkennung des besten Algorithmus
        if algorithm == "auto":
            self.algorithm = SVDAlgorithm.STANDARD
        else:
            self.algorithm = SVDAlgorithm(algorithm) if algorithm in [a.value for a in SVDAlgorithm] else SVDAlgorithm.STANDARD
            
        # Performance-Tracking
        self.performance_metrics = {
            "calls": 0,
            "compute_time": 0.0,
            "average_time": 0.0,
            "backend_usage": {b.value: 0 for b in SVDBackend},
            "algorithm_usage": {a.value: 0 for a in SVDAlgorithm}
        }
        
        # Backends laden
        self.backend_modules = {}
        self._load_backend_modules()
        
        # JIT-Kompilierung für MLX
        self.jit_available = False
        if self.backend == SVDBackend.MLX and "mlx" in self.backend_modules:
            self.jit_available = self._check_jit_availability()
            
        # Metal direkte Integration für Apple Silicon
        self.metal_available = self._check_metal_availability()
        
        ztm_log(f"SVD-Engine initialisiert: Backend={self.backend.value}, Algorithmus={self.algorithm.value}, JIT={'aktiv' if self.jit_available else 'inaktiv'}")
    
    def _detect_available_backends(self) -> List[SVDBackend]:
        """Erkennt verfügbare SVD-Backends"""
        available = []
        
        # NumPy ist immer verfügbar (Standardinstallation)
        available.append(SVDBackend.NUMPY)
        
        # PyTorch überprüfen
        try:
            import torch
            available.append(SVDBackend.TORCH)
        except ImportError:
            pass
            
        # MLX überprüfen (Apple Silicon)
        try:
            import mlx.core
            available.append(SVDBackend.MLX)
        except ImportError:
            pass
            
        # SciPy überprüfen
        try:
            import scipy.linalg
            available.append(SVDBackend.SCIPY)
        except ImportError:
            pass
            
        return available
    
    def _select_best_backend(self) -> SVDBackend:
        """Wählt das beste verfügbare Backend basierend auf Hardware und Verfügbarkeit"""
        # Apple Silicon mit MLX ist optimal
        if self._is_apple_silicon() and SVDBackend.MLX in self.available_backends:
            return SVDBackend.MLX
            
        # PyTorch mit CUDA/ROCm ist nächstbeste Option
        if SVDBackend.TORCH in self.available_backends and self._has_gpu_torch():
            return SVDBackend.TORCH
            
        # SciPy für große, dünnbesetzte Matrizen
        if SVDBackend.SCIPY in self.available_backends:
            return SVDBackend.SCIPY
            
        # NumPy als Fallback
        return SVDBackend.NUMPY
    
    def _is_apple_silicon(self) -> bool:
        """Überprüft, ob der Code auf Apple Silicon läuft"""
        import platform
        return platform.processor() == 'arm' and platform.system() == 'Darwin'
    
    def _has_gpu_torch(self) -> bool:
        """Überprüft, ob PyTorch GPU-Unterstützung hat"""
        if SVDBackend.TORCH not in self.available_backends:
            return False
            
        import torch
        return torch.cuda.is_available() or (hasattr(torch, 'mps') and torch.backends.mps.is_available())
    
    def _load_backend_modules(self) -> None:
        """Lädt die benötigten Backend-Module"""
        # NumPy ist immer verfügbar
        self.backend_modules["numpy"] = np
        
        # PyTorch laden, wenn verfügbar
        if SVDBackend.TORCH in self.available_backends:
            import torch
            self.backend_modules["torch"] = torch
            
        # MLX laden, wenn verfügbar
        if SVDBackend.MLX in self.available_backends:
            import mlx.core as mx
            self.backend_modules["mlx"] = mx
            
        # SciPy laden, wenn verfügbar
        if SVDBackend.SCIPY in self.available_backends:
            import scipy.linalg
            self.backend_modules["scipy"] = scipy.linalg
    
    def _check_jit_availability(self) -> bool:
        """Überprüft, ob JIT-Kompilierung für MLX verfügbar ist"""
        if "mlx" not in self.backend_modules:
            return False
            
        try:
            # Versuche, eine JIT-Funktion zu erstellen
            import mlx.core as mx
            
            @mx.compile
            def _test_jit(x):
                return mx.sum(x)
                
            # Test ausführen
            _test_jit(mx.ones((2, 2)))
            return True
        except Exception as e:
            ztm_log(f"MLX JIT-Kompilierung nicht verfügbar: {str(e)}", level="WARNING")
            return False
    
    def _check_metal_availability(self) -> bool:
        """Überprüft, ob Metal direkt verwendet werden kann (nur auf macOS)"""
        import platform
        if platform.system() != 'Darwin':
            return False
            
        try:
            # Versuche, pyobjc zu importieren
            import objc
            return True
        except ImportError:
            return False
    
    def svd(self, matrix: Any, full_matrices: bool = True, k: Optional[int] = None) -> Tuple[Any, Any, Any]:
        """
        Führt eine Singular Value Decomposition (SVD) durch
        
        Berechnet die optimale SVD basierend auf Matrixgröße, verfügbaren Backends
        und gewähltem Algorithmus.
        
        Args:
            matrix: Eingangsmatrix
            full_matrices: Ob vollständige oder reduzierte Matrizen zurückgegeben werden
            k: Anzahl der zu berechnenden Singulärwerte (None für alle)
            
        Returns:
            Tuple aus (U, S, V), wobei U und V die Singulärmatrizen sind und S die Singulärwerte
        """
        # Performance-Tracking starten
        start_time = time.time()
        self.performance_metrics["calls"] += 1
        
        # Matrix-Dimensionen analysieren für algorithmische Entscheidungen
        matrix_shape = getattr(matrix, "shape", None)
        
        # Algorithmus und Backend basierend auf Matrixgröße wählen
        backend, algorithm = self._select_algorithm(matrix_shape, k)
        
        # Backend-Nutzung zählen
        self.performance_metrics["backend_usage"][backend.value] += 1
        self.performance_metrics["algorithm_usage"][algorithm.value] += 1
        
        try:
            # SVD mit optimierten Parametern durchführen
            u, s, vh = self._compute_svd(matrix, backend, algorithm, full_matrices, k)
            
            # Performance-Tracking aktualisieren
            end_time = time.time()
            compute_time = end_time - start_time
            self.performance_metrics["compute_time"] += compute_time
            self.performance_metrics["average_time"] = (
                self.performance_metrics["compute_time"] / self.performance_metrics["calls"]
            )
            
            if ZTM_ACTIVE:
                ztm_log(f"SVD erfolgreich ({backend.value}/{algorithm.value}): Matrix {matrix_shape}, Zeit: {compute_time:.4f}s", level="INFO")
            
            return u, s, vh
            
        except Exception as e:
            ztm_log(f"SVD-Fehler: {str(e)}", level="ERROR")
            raise RuntimeError(f"SVD-Berechnung fehlgeschlagen: {str(e)}")
    
    def _select_algorithm(self, matrix_shape: Tuple[int, int], k: Optional[int]) -> Tuple[SVDBackend, SVDAlgorithm]:
        """
        Wählt den optimalen Algorithmus und Backend basierend auf Matrixgröße
        
        Args:
            matrix_shape: Form der Matrix
            k: Anzahl der Singulärwerte (None für alle)
            
        Returns:
            Tuple aus (backend, algorithmus)
        """
        # Default-Werte
        backend = self.backend
        algorithm = self.algorithm
        
        if not matrix_shape:
            return backend, algorithm
            
        rows, cols = matrix_shape
        size = rows * cols
        smaller_dim = min(rows, cols)
        
        # Kleine Matrizen (<50x50): NumPy ist fast immer schneller
        if smaller_dim < 50:
            return SVDBackend.NUMPY, SVDAlgorithm.STANDARD
            
        # Große dünne Matrizen: Randomisierte SVD
        if smaller_dim > 1000 and k is not None and k < smaller_dim // 10:
            if SVDBackend.SCIPY in self.available_backends:
                return SVDBackend.SCIPY, SVDAlgorithm.RANDOMIZED
                
        # Große Matrizen: Hardwarebeschleunigung nutzen
        if smaller_dim >= 100:
            if SVDBackend.MLX in self.available_backends and self._is_apple_silicon():
                return SVDBackend.MLX, SVDAlgorithm.STANDARD
            if SVDBackend.TORCH in self.available_backends and self._has_gpu_torch():
                return SVDBackend.TORCH, SVDAlgorithm.STANDARD
                
        return backend, algorithm
    
    def _compute_svd(self, 
                     matrix: Any, 
                     backend: SVDBackend, 
                     algorithm: SVDAlgorithm,
                     full_matrices: bool,
                     k: Optional[int]) -> Tuple[Any, Any, Any]:
        """
        Führt die SVD-Berechnung mit dem gewählten Backend und Algorithmus durch
        
        Args:
            matrix: Eingangsmatrix
            backend: Zu verwendendes Backend
            algorithm: Zu verwendender Algorithmus
            full_matrices: Ob vollständige oder reduzierte Matrizen zurückgegeben werden
            k: Anzahl der zu berechnenden Singulärwerte (None für alle)
            
        Returns:
            Tuple aus (U, S, V)
        """
        # NumPy-Backend
        if backend == SVDBackend.NUMPY:
            matrix_np = self._ensure_numpy(matrix)
            
            if algorithm == SVDAlgorithm.RANDOMIZED and k is not None:
                # Randomisierte SVD für große Matrizen mit wenigen Singulärwerten
                return self._randomized_svd(matrix_np, k, full_matrices)
                
            # Standard NumPy SVD
            return np.linalg.svd(matrix_np, full_matrices=full_matrices)
            
        # PyTorch-Backend
        elif backend == SVDBackend.TORCH:
            torch = self.backend_modules["torch"]
            matrix_torch = self._ensure_torch(matrix)
            
            # Hardware-Optimierung aktivieren
            if torch.cuda.is_available():
                matrix_torch = matrix_torch.cuda()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                matrix_torch = matrix_torch.to('mps')
                
            # PyTorch SVD
            u, s, v = torch.linalg.svd(matrix_torch, full_matrices=full_matrices)
            
            # Ergebnis zu NumPy konvertieren für konsistente Rückgabe
            return u.cpu().numpy(), s.cpu().numpy(), v.cpu().numpy()
            
        # MLX-Backend
        elif backend == SVDBackend.MLX:
            mx = self.backend_modules["mlx"]
            matrix_mlx = self._ensure_mlx(matrix)
            
            try:
                # Wenn die optimierte SVD-Implementation vorhanden ist, verwende sie
                sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
                from miso.math.t_mathematics.optimizations.optimized_mlx_svd import MLXSVDOptimizer
                
                # Backend-Wrapper
                class MinimalBackend:
                    def __init__(self):
                        self.precision = "float32"
                
                # Optimizer verwenden
                optimizer = MLXSVDOptimizer(MinimalBackend())
                return optimizer.svd(matrix_mlx, k)
                
            except ImportError:
                # Fallback: Konvertiere zu NumPy, berechne SVD und zurück zu MLX
                matrix_np = self._ensure_numpy(matrix_mlx)
                u, s, v = np.linalg.svd(matrix_np, full_matrices=full_matrices)
                
                # Zurück zu MLX konvertieren
                return mx.array(u), mx.array(s), mx.array(v)
                
        # SciPy-Backend
        elif backend == SVDBackend.SCIPY:
            scipy_linalg = self.backend_modules["scipy"]
            matrix_np = self._ensure_numpy(matrix)
            
            if algorithm == SVDAlgorithm.RANDOMIZED and k is not None:
                # Randomisierte SVD von SciPy
                from scipy.sparse.linalg import svds
                # k muss kleiner als min(matrix_shape) sein
                k = min(k, min(matrix_np.shape) - 1)
                u, s, vt = svds(matrix_np, k=k)
                # svds sortiert nicht, also manuell sortieren
                idx = np.argsort(s)[::-1]
                return u[:, idx], s[idx], vt[idx, :]
                
            # Standard SciPy SVD
            return scipy_linalg.svd(matrix_np, full_matrices=full_matrices)
            
        # Unbekanntes Backend
        else:
            raise ValueError(f"Unbekanntes SVD-Backend: {backend}")
    
    def _randomized_svd(self, matrix_np: np.ndarray, k: int, full_matrices: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Führt eine randomisierte SVD für große Matrizen mit NumPy durch
        
        Args:
            matrix_np: NumPy-Matrix
            k: Anzahl der Singulärwerte
            full_matrices: Ob vollständige oder reduzierte Matrizen zurückgegeben werden
            
        Returns:
            Tuple aus (U, S, V)
        """
        # Implementierung einer einfachen randomisierten SVD
        m, n = matrix_np.shape
        p = min(k + 10, min(m, n))  # Überabtastung für numerische Stabilität
        
        # Zufallsprojektion
        random_matrix = np.random.randn(n, p)
        Y = matrix_np @ random_matrix
        
        # QR-Zerlegung
        Q, _ = np.linalg.qr(Y, mode='reduced')
        
        # Projiziere Matrix auf den Unterraum
        B = Q.T @ matrix_np
        
        # SVD auf der kleineren Matrix
        Uhat, s, v = np.linalg.svd(B, full_matrices=False)
        
        # Projiziere zurück
        u = Q @ Uhat
        
        # Beschränke auf k Singulärwerte
        u = u[:, :k]
        s = s[:k]
        v = v[:k, :]
        
        return u, s, v
    
    def _ensure_numpy(self, matrix: Any) -> np.ndarray:
        """Konvertiert eine Matrix zu NumPy"""
        if isinstance(matrix, np.ndarray):
            return matrix
            
        # MLX zu NumPy
        if "mlx" in self.backend_modules and hasattr(matrix, 'device'):
            return matrix.numpy()
            
        # PyTorch zu NumPy
        if "torch" in self.backend_modules:
            torch = self.backend_modules["torch"]
            if isinstance(matrix, torch.Tensor):
                return matrix.detach().cpu().numpy()
                
        # Liste oder Tuple zu NumPy
        if isinstance(matrix, (list, tuple)):
            return np.array(matrix)
            
        # Unbekannter Typ
        raise TypeError(f"Kann Matrix vom Typ {type(matrix)} nicht zu NumPy konvertieren")
    
    def _ensure_torch(self, matrix: Any):
        """Konvertiert eine Matrix zu PyTorch"""
        torch = self.backend_modules["torch"]
        
        if isinstance(matrix, torch.Tensor):
            return matrix
            
        # NumPy zu PyTorch
        if isinstance(matrix, np.ndarray):
            return torch.from_numpy(matrix)
            
        # MLX zu PyTorch über NumPy
        if "mlx" in self.backend_modules and hasattr(matrix, 'device'):
            return torch.from_numpy(matrix.numpy())
            
        # Liste oder Tuple zu PyTorch
        if isinstance(matrix, (list, tuple)):
            return torch.tensor(matrix)
            
        # Unbekannter Typ
        raise TypeError(f"Kann Matrix vom Typ {type(matrix)} nicht zu PyTorch konvertieren")
    
    def _ensure_mlx(self, matrix: Any):
        """Konvertiert eine Matrix zu MLX"""
        if "mlx" not in self.backend_modules:
            raise ImportError("MLX ist nicht verfügbar")
            
        mx = self.backend_modules["mlx"]
        
        if hasattr(matrix, 'device'):  # MLX-Array hat ein device-Attribut
            return matrix
            
        # NumPy zu MLX
        if isinstance(matrix, np.ndarray):
            return mx.array(matrix)
            
        # PyTorch zu MLX über NumPy
        if "torch" in self.backend_modules:
            torch = self.backend_modules["torch"]
            if isinstance(matrix, torch.Tensor):
                return mx.array(matrix.detach().cpu().numpy())
                
        # Liste oder Tuple zu MLX
        if isinstance(matrix, (list, tuple)):
            return mx.array(matrix)
            
        # Unbekannter Typ
        raise TypeError(f"Kann Matrix vom Typ {type(matrix)} nicht zu MLX konvertieren")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Gibt einen detaillierten Performance-Bericht zurück
        
        Returns:
            Dictionary mit Performance-Metriken
        """
        return {
            "total_calls": self.performance_metrics["calls"],
            "total_compute_time": self.performance_metrics["compute_time"],
            "average_compute_time": self.performance_metrics["average_time"],
            "backend_usage": self.performance_metrics["backend_usage"],
            "algorithm_usage": self.performance_metrics["algorithm_usage"],
            "jit_available": self.jit_available,
            "metal_available": self.metal_available,
            "available_backends": [b.value for b in self.available_backends],
            "preferred_backend": self.backend.value,
            "preferred_algorithm": self.algorithm.value
        }


# Globale Engine-Instanz für Singleton-Pattern
_engine_instance = None

def get_svd_engine(backend: str = "auto", algorithm: str = "auto") -> SVDEngine:
    """
    Gibt eine (gecachte) SVD-Engine-Instanz zurück
    
    Args:
        backend: Zu verwendendes Backend ('auto', 'numpy', 'torch', 'mlx', 'scipy')
        algorithm: Zu verwendender Algorithmus ('auto', 'standard', 'truncated', 'randomized', 'qr', 'jacobi')
        
    Returns:
        SVDEngine-Instanz
    """
    global _engine_instance
    
    if _engine_instance is None:
        _engine_instance = SVDEngine(backend, algorithm)
        
    return _engine_instance


# Hauptfunktion für direkte Nutzung
def compute_svd(matrix: Any, full_matrices: bool = True, k: Optional[int] = None) -> Tuple[Any, Any, Any]:
    """
    Berechnet die SVD einer Matrix mit optimaler Performance
    
    Args:
        matrix: Eingangsmatrix
        full_matrices: Ob vollständige oder reduzierte Matrizen zurückgegeben werden
        k: Anzahl der zu berechnenden Singulärwerte (None für alle)
        
    Returns:
        Tuple aus (U, S, V), wobei U und V die Singulärmatrizen sind und S die Singulärwerte
    """
    engine = get_svd_engine()
    return engine.svd(matrix, full_matrices, k)


if __name__ == "__main__":
    # Test-Code
    print("VX-SVD-Engine Selbsttest")
    engine = SVDEngine()
    
    # Test mit zufälliger Matrix
    test_matrix = np.random.rand(100, 100)
    u, s, v = engine.svd(test_matrix)
    
    # Rekonstruktion testen
    reconstruct = u @ np.diag(s) @ v
    error = np.mean(np.abs(test_matrix - reconstruct))
    
    print(f"SVD-Rekonstruktionsfehler: {error:.8f}")
    print(f"Performance-Bericht: {engine.get_performance_report()}")
