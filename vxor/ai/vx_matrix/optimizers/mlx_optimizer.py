#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MATRIX: MLX-Optimizer für Apple Silicon

Diese Datei implementiert Optimierungen für die MLX-Bibliothek auf Apple Silicon,
inklusive JIT-Kompilierung, Kernel-Fusion und spezialisierte Tensor-Operationen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
ZTM-Level: STRICT
"""

import os
import sys
import enum
import logging
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path

# Füge MISO-Root zum Pythonpfad hinzu
VXOR_AI_DIR = Path(__file__).parent.parent.parent.absolute()
MISO_ROOT = VXOR_AI_DIR.parent / "miso"
sys.path.insert(0, str(MISO_ROOT))

# ZTM-Protokoll-Initialisierung
ZTM_ACTIVE = os.environ.get('MISO_ZTM_MODE', '1') == '1'
ZTM_LOG_LEVEL = os.environ.get('MISO_ZTM_LOG_LEVEL', 'INFO')

# Logger konfigurieren
logger = logging.getLogger("VXOR.VX-MATRIX.mlx_optimizer")
logger.setLevel(getattr(logging, ZTM_LOG_LEVEL))

def ztm_log(message: str, level: str = 'INFO', module: str = 'MLX_OPTIMIZER'):
    """ZTM-konforme Logging-Funktion"""
    if not ZTM_ACTIVE:
        return
    
    log_func = getattr(logger, level.lower())
    log_func(f"[ZTM:{module}] {message}")

# Überprüfe, ob MLX verfügbar ist
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
    ztm_log("MLX-Bibliothek erfolgreich importiert", level="INFO")
except ImportError as e:
    MLX_AVAILABLE = False
    ztm_log(f"MLX-Bibliothek nicht verfügbar: {e}", level="WARNING")

# Überprüfe, ob die Optimierungen aus MISO verfügbar sind
try:
    from miso.math.t_mathematics.optimizations.optimized_mlx_svd import mlx_svd
    from miso.math.t_mathematics.optimizations.optimized_mlx_inverse import mlx_inverse
    MISO_MLX_OPTIMIZATIONS_AVAILABLE = True
    ztm_log("MISO MLX-Optimierungen erfolgreich importiert", level="INFO")
except ImportError:
    MISO_MLX_OPTIMIZATIONS_AVAILABLE = False
    ztm_log("MISO MLX-Optimierungen nicht verfügbar", level="WARNING")

# Klasse für optimierte Funktionen
class OptimizedFunction:
    """Enthält Informationen über eine optimierte Funktion"""
    
    def __init__(self, name: str, function: Callable, jit_compiled: bool = False, is_miso: bool = False):
        """
        Initialisiert eine optimierte Funktion
        
        Args:
            name: Name der Funktion
            function: Die Funktion selbst
            jit_compiled: Ob die Funktion JIT-kompiliert ist
            is_miso: Ob die Funktion aus MISO-Optimierungen stammt
        """
        self.name = name
        self.function = function
        self.jit_compiled = jit_compiled
        self.is_miso = is_miso
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
    
    def __call__(self, *args, **kwargs):
        """Ruft die optimierte Funktion auf und misst die Ausführungszeit"""
        start_time = time.time()
        result = self.function(*args, **kwargs)
        elapsed = time.time() - start_time
        
        # Aktualisiere Statistiken
        self.call_count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        
        return result

class MLXOptimizer:
    """
    Optimierer für MLX-Operationen auf Apple Silicon
    
    Diese Klasse bietet optimierte Implementierungen von häufig verwendeten
    Tensor-Operationen, mit Fokus auf die Leistung auf Apple Silicon-Hardware.
    """
    
    def __init__(self):
        """Initialisiert den MLX-Optimierer"""
        self.initialized = MLX_AVAILABLE
        self.optimized_functions = {}
        self.jit_enabled = True
        self.enable_fusion = True
        self.use_miso_optimizations = True
        
        if self.initialized:
            self._register_optimized_functions()
            ztm_log("MLX-Optimierer erfolgreich initialisiert", level="INFO")
        else:
            ztm_log("MLX-Optimierer konnte nicht initialisiert werden (MLX nicht verfügbar)", level="WARNING")
    
    def _register_optimized_functions(self):
        """Registriert alle optimierten Funktionen"""
        if not self.initialized:
            return
        
        # --- Grundlegende Matrixoperationen ---
        
        # Matrix-Multiplikation
        def optimized_matmul(a, b):
            return mx.matmul(a, b)
        
        if self.jit_enabled:
            optimized_matmul_jit = mx.compile(optimized_matmul)
            self.optimized_functions['matmul'] = OptimizedFunction('matmul', optimized_matmul_jit, jit_compiled=True)
        else:
            self.optimized_functions['matmul'] = OptimizedFunction('matmul', optimized_matmul)
        
        # Transposition
        def optimized_transpose(a, axes=None):
            return mx.transpose(a, axes)
        
        if self.jit_enabled:
            optimized_transpose_jit = mx.compile(optimized_transpose)
            self.optimized_functions['transpose'] = OptimizedFunction('transpose', optimized_transpose_jit, jit_compiled=True)
        else:
            self.optimized_functions['transpose'] = OptimizedFunction('transpose', optimized_transpose)
        
        # Einsum
        def optimized_einsum(equation, *operands):
            return mx.einsum(equation, *operands)
        
        self.optimized_functions['einsum'] = OptimizedFunction('einsum', optimized_einsum)
        
        # --- Lineare Algebra ---
        
        # SVD
        if MISO_MLX_OPTIMIZATIONS_AVAILABLE and self.use_miso_optimizations:
            # Verwende optimierte MISO-Implementierung
            self.optimized_functions['svd'] = OptimizedFunction('svd', mlx_svd, is_miso=True)
        else:
            # Fallback-Implementierung
            def fallback_svd(a, full_matrices=True):
                # Konvertiere zu NumPy, führe SVD durch, konvertiere zurück
                a_np = a.tolist()
                a_np = np.array(a_np)
                u, s, vh = np.linalg.svd(a_np, full_matrices=full_matrices)
                return mx.array(u), mx.array(s), mx.array(vh)
            
            self.optimized_functions['svd'] = OptimizedFunction('svd', fallback_svd)
        
        # Matrixinverse
        if MISO_MLX_OPTIMIZATIONS_AVAILABLE and self.use_miso_optimizations:
            # Verwende optimierte MISO-Implementierung
            self.optimized_functions['inverse'] = OptimizedFunction('inverse', mlx_inverse, is_miso=True)
        else:
            # Fallback-Implementierung
            def fallback_inverse(a):
                # Konvertiere zu NumPy, berechne Inverse, konvertiere zurück
                a_np = a.tolist()
                a_np = np.array(a_np)
                inv = np.linalg.inv(a_np)
                return mx.array(inv)
            
            self.optimized_functions['inverse'] = OptimizedFunction('inverse', fallback_inverse)
        
        # Determinante
        def optimized_det(a):
            # Konvertiere zu NumPy, berechne Determinante, konvertiere zurück
            a_np = a.tolist()
            a_np = np.array(a_np)
            det = np.linalg.det(a_np)
            return mx.array(det)
        
        self.optimized_functions['det'] = OptimizedFunction('det', optimized_det)
        
        # --- Faltungsoperationen ---
        
        # 2D-Faltung (Convolution)
        def optimized_conv2d(x, w, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
            return nn.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
        if self.jit_enabled:
            optimized_conv2d_jit = mx.compile(optimized_conv2d)
            self.optimized_functions['conv2d'] = OptimizedFunction('conv2d', optimized_conv2d_jit, jit_compiled=True)
        else:
            self.optimized_functions['conv2d'] = OptimizedFunction('conv2d', optimized_conv2d)
        
        # --- Aktivierungsfunktionen ---
        
        # ReLU
        def optimized_relu(x):
            return mx.maximum(x, 0)
        
        if self.jit_enabled:
            optimized_relu_jit = mx.compile(optimized_relu)
            self.optimized_functions['relu'] = OptimizedFunction('relu', optimized_relu_jit, jit_compiled=True)
        else:
            self.optimized_functions['relu'] = OptimizedFunction('relu', optimized_relu)
        
        # Sigmoid
        def optimized_sigmoid(x):
            return mx.sigmoid(x)
        
        if self.jit_enabled:
            optimized_sigmoid_jit = mx.compile(optimized_sigmoid)
            self.optimized_functions['sigmoid'] = OptimizedFunction('sigmoid', optimized_sigmoid_jit, jit_compiled=True)
        else:
            self.optimized_functions['sigmoid'] = OptimizedFunction('sigmoid', optimized_sigmoid)
        
        # --- Kombination von Operationen ---
        
        # MatMul + ReLU (häufige Kombination in neuronalen Netzen)
        def optimized_matmul_relu(a, b):
            return mx.maximum(mx.matmul(a, b), 0)
        
        if self.jit_enabled and self.enable_fusion:
            optimized_matmul_relu_jit = mx.compile(optimized_matmul_relu)
            self.optimized_functions['matmul_relu'] = OptimizedFunction('matmul_relu', optimized_matmul_relu_jit, jit_compiled=True)
        else:
            self.optimized_functions['matmul_relu'] = OptimizedFunction('matmul_relu', optimized_matmul_relu)
    
    def invoke(self, function_name: str, *args, **kwargs) -> Any:
        """
        Ruft eine optimierte Funktion auf
        
        Args:
            function_name: Name der optimierten Funktion
            *args, **kwargs: Argumente für die Funktion
            
        Returns:
            Ergebnis der Funktion
        """
        if not self.initialized:
            ztm_log(f"MLX-Optimierer nicht initialisiert, Funktion {function_name} kann nicht aufgerufen werden", level="ERROR")
            return None
        
        if function_name not in self.optimized_functions:
            ztm_log(f"Optimierte Funktion {function_name} nicht gefunden", level="ERROR")
            return None
        
        optimized_func = self.optimized_functions[function_name]
        
        # ZTM-Logging
        if ZTM_ACTIVE:
            args_info = f"{len(args)} Argumenten"
            if args and hasattr(args[0], 'shape'):
                args_info += f", erste Form: {args[0].shape}"
            ztm_log(f"Optimierte Funktion {function_name} aufgerufen mit {args_info}", level="INFO")
        
        try:
            return optimized_func(*args, **kwargs)
        except Exception as e:
            ztm_log(f"Fehler beim Aufruf der optimierten Funktion {function_name}: {e}", level="ERROR")
            return None
    
    def matmul(self, a: Any, b: Any) -> Any:
        """
        Optimierte Matrix-Multiplikation
        
        Args:
            a: Erste Matrix
            b: Zweite Matrix
            
        Returns:
            Ergebnismatrix
        """
        return self.invoke('matmul', a, b)
    
    def svd(self, a: Any, full_matrices: bool = True) -> Tuple[Any, Any, Any]:
        """
        Optimierte Singular Value Decomposition (SVD)
        
        Args:
            a: Eingangsmatrix
            full_matrices: Ob vollständige oder reduzierte Matrizen zurückgegeben werden
            
        Returns:
            Tuple aus (U, S, V), wobei U und V die Singulärmatrizen sind und S die Singulärwerte
        """
        return self.invoke('svd', a, full_matrices=full_matrices)
    
    def inverse(self, a: Any) -> Any:
        """
        Optimierte Matrixinversion
        
        Args:
            a: Eingangsmatrix
            
        Returns:
            Inverse Matrix
        """
        return self.invoke('inverse', a)
    
    def det(self, a: Any) -> Any:
        """
        Optimierte Determinantenberechnung
        
        Args:
            a: Eingangsmatrix
            
        Returns:
            Determinante
        """
        return self.invoke('det', a)
    
    def transpose(self, a: Any, axes: Any = None) -> Any:
        """
        Optimierte Transposition
        
        Args:
            a: Eingangsmatrix
            axes: Achsen für die Transposition
            
        Returns:
            Transponierte Matrix
        """
        return self.invoke('transpose', a, axes)
    
    def einsum(self, equation: str, *operands) -> Any:
        """
        Optimierte Einstein-Summenkonvention
        
        Args:
            equation: Einstein-Summenkonvention-Gleichung
            *operands: Operanden für die Gleichung
            
        Returns:
            Ergebnis der Einstein-Summe
        """
        return self.invoke('einsum', equation, *operands)
    
    def conv2d(self, x: Any, w: Any, stride: Tuple[int, int] = (1, 1), 
               padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), 
               groups: int = 1) -> Any:
        """
        Optimierte 2D-Faltung
        
        Args:
            x: Eingabetensor
            w: Filterkerne
            stride: Schrittweite
            padding: Auffüllung
            dilation: Dilatation
            groups: Gruppen
            
        Returns:
            Gefalteter Tensor
        """
        return self.invoke('conv2d', x, w, stride=stride, padding=padding, 
                          dilation=dilation, groups=groups)
    
    def relu(self, x: Any) -> Any:
        """
        Optimierte ReLU-Aktivierungsfunktion
        
        Args:
            x: Eingabetensor
            
        Returns:
            Aktivierter Tensor
        """
        return self.invoke('relu', x)
    
    def sigmoid(self, x: Any) -> Any:
        """
        Optimierte Sigmoid-Aktivierungsfunktion
        
        Args:
            x: Eingabetensor
            
        Returns:
            Aktivierter Tensor
        """
        return self.invoke('sigmoid', x)
    
    def matmul_relu(self, a: Any, b: Any) -> Any:
        """
        Optimierte Matrix-Multiplikation gefolgt von ReLU
        
        Args:
            a: Erste Matrix
            b: Zweite Matrix
            
        Returns:
            Aktivierte Ergebnismatrix
        """
        return self.invoke('matmul_relu', a, b)
    
    def get_stats(self) -> Dict:
        """
        Gibt Statistiken über die optimierten Funktionen zurück
        
        Returns:
            Dictionary mit Funktionsstatistiken
        """
        stats = {}
        
        for name, func in self.optimized_functions.items():
            # Nur Statistiken für aufgerufene Funktionen erstellen
            if func.call_count > 0:
                stats[name] = {
                    "call_count": func.call_count,
                    "total_time": func.total_time,
                    "avg_time": func.total_time / func.call_count if func.call_count > 0 else 0,
                    "min_time": func.min_time if func.min_time < float('inf') else 0,
                    "max_time": func.max_time,
                    "jit_compiled": func.jit_compiled,
                    "is_miso": func.is_miso
                }
        
        # Allgemeine Statistiken
        stats["general"] = {
            "total_calls": sum(func.call_count for func in self.optimized_functions.values()),
            "total_time": sum(func.total_time for func in self.optimized_functions.values()),
            "jit_enabled": self.jit_enabled,
            "fusion_enabled": self.enable_fusion,
            "miso_optimizations": self.use_miso_optimizations
        }
        
        return stats
    
    def clear_stats(self):
        """Setzt alle Statistiken zurück"""
        for func in self.optimized_functions.values():
            func.call_count = 0
            func.total_time = 0.0
            func.min_time = float('inf')
            func.max_time = 0.0

# Wenn direkt ausgeführt, führe einen kleinen Test durch
if __name__ == "__main__":
    if not MLX_AVAILABLE:
        print("MLX nicht verfügbar, Test wird übersprungen")
        sys.exit(0)
    
    import numpy as np
    import mlx.core as mx
    
    print("MLX-Optimierer Test")
    
    # Erstelle MLX-Optimierer
    optimizer = MLXOptimizer()
    print(f"Optimierer initialisiert: {optimizer.initialized}")
    
    if optimizer.initialized:
        # Erstelle Testmatrizen
        a_np = np.random.rand(3, 3)
        b_np = np.random.rand(3, 3)
        
        # Konvertiere zu MLX-Arrays
        a = mx.array(a_np)
        b = mx.array(b_np)
        
        print("\nMatrix A:")
        print(a)
        print("\nMatrix B:")
        print(b)
        
        # Matrix-Multiplikation
        c = optimizer.matmul(a, b)
        print("\nA * B (optimiert):")
        print(c)
        
        # Vergleiche mit Standard-MLX
        c_std = mx.matmul(a, b)
        print("\nA * B (standard MLX):")
        print(c_std)
        
        # Berechne Inverse
        a_inv = optimizer.inverse(a)
        print("\nInverse von A (optimiert):")
        print(a_inv)
        
        # SVD
        u, s, v = optimizer.svd(a)
        print("\nSVD von A (optimiert):")
        print(f"U: {u}")
        print(f"S: {s}")
        print(f"V: {v}")
        
        # ReLU Test
        x = mx.array([-1.0, 0.0, 1.0])
        y = optimizer.relu(x)
        print("\nReLU([-1, 0, 1]):")
        print(y)
        
        # Matrix-Multiplikation mit ReLU
        c_relu = optimizer.matmul_relu(a, b)
        print("\nMatMul + ReLU (optimiert):")
        print(c_relu)
        
        # Statistiken
        print("\nOptimierungs-Statistiken:")
        stats = optimizer.get_stats()
        print(json.dumps(stats, indent=2))
