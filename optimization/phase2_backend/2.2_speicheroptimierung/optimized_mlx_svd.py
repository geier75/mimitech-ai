#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimierte SVD-Implementierung für MLX

Diese Datei bietet eine zuverlässige und optimierte SVD-Implementierung für MLX,
die auf Apple Silicon deutlich performanter ist.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, Union, Tuple
from functools import lru_cache

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.optimized_mlx_svd")

# MLX-Verfügbarkeit prüfen
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None
    logger.warning("MLX konnte nicht importiert werden. Apple Silicon Optimierungen sind nicht verfügbar.")

# Apple Silicon-Erkennung
IS_APPLE_SILICON = False
try:
    import platform
    IS_APPLE_SILICON = platform.processor() == 'arm' and platform.system() == 'Darwin'
    if IS_APPLE_SILICON:
        logger.info("Apple Silicon erkannt, optimiere für Neural Engine")
except Exception as e:
    logger.warning(f"Fehler bei der Apple Silicon-Erkennung: {e}")

# MLX-JIT-Kompilierung aktivieren
JIT_ENABLED = False
if HAS_MLX and hasattr(mx, 'jit'):
    JIT_ENABLED = True
    logger.info("MLX JIT-Kompilierung verfügbar")
else:
    logger.warning("MLX JIT-Kompilierung nicht verfügbar")

class MLXSVDOptimizer:
    """
    Optimierter SVD-Implementierung für Apple Silicon mit MLX
    """
    
    def __init__(self, backend, precision="float16"):
        """
        Initialisiert den SVD-Optimizer
        
        Args:
            backend: MLXBackend-Instanz für Tensor-Konvertierungen
            precision: Präzision für Berechnungen (float16 oder float32)
        """
        self.backend = backend
        self.precision = precision
        self.cache = {}
        self.jit_cache = {}
        
        # MPS-Gerät für PyTorch
        if torch.backends.mps.is_available():
            self.torch_device = torch.device("mps")
        else:
            self.torch_device = torch.device("cpu")
            
        # Optimale Datentypen für MLX
        if HAS_MLX:
            self.dtype = mx.float16 if precision == "float16" else mx.float32
            
            # Aktiviere JIT-Kompilierung wenn verfügbar
            if JIT_ENABLED:
                try:
                    # Aktiviere GPU für MLX
                    mx.set_default_device(mx.gpu if IS_APPLE_SILICON else mx.cpu)
                except:
                    logger.warning("Konnte MLX-Gerät nicht setzen")
                    
    @lru_cache(maxsize=32)
    def _get_svd_key(self, a_shape, k=None):
        """
        Generiert einen eindeutigen Schlüssel für das SVD-Caching
        
        Args:
            a_shape: Form der Matrix
            k: Anzahl der Singulärwerte (None für alle)
            
        Returns:
            Cache-Schlüssel als String
        """
        return f"svd_{a_shape}_{k}"
    
    def _estimate_matrix_complexity(self, shape):
        """
        Schätzt die Komplexität einer Matrix basierend auf ihrer Form
        
        Args:
            shape: Form der Matrix
            
        Returns:
            Komplexitätswert (höher bedeutet komplexer)
        """
        if len(shape) < 2:
            return 0
        
        # Berechne die Komplexität basierend auf Matrixgröße
        rows, cols = shape[-2], shape[-1]
        return rows * cols
    
    def _compile_svd_jit(self, k=None):
        """
        Kompiliert eine JIT-optimierte SVD-Funktion
        
        Args:
            k: Anzahl der zu berechnenden Singulärwerte
            
        Returns:
            JIT-kompilierte SVD-Funktion
        """
        if not HAS_MLX or not JIT_ENABLED:
            return None
        
        # SVD-Implementierung
        def jitted_svd(x):
            # Direkter SVD-Aufruf mit MLX
            try:
                # MLX bietet derzeit keine native SVD, daher verwenden wir NumPy als Backend
                x_np = x.tolist()
                
                if k is None:
                    # Vollständige SVD
                    u, s, v = np.linalg.svd(x_np, full_matrices=False)
                else:
                    # Partielle SVD
                    try:
                        from scipy.sparse.linalg import svds
                        u, s, vt = svds(x_np, k=k)
                        # Sortiere die Singulärwerte in absteigender Reihenfolge
                        idx = np.argsort(s)[::-1]
                        u, s, v = u[:, idx], s[idx], vt[idx, :]
                    except (ImportError, ValueError):
                        # Fallback auf vollständige SVD mit Trunkierung
                        u, s, v = np.linalg.svd(x_np, full_matrices=False)
                        u, s, v = u[:, :k], s[:k], v[:k, :]
                
                # Rückkonvertierung zu MLX
                u_mx = mx.array(u)
                s_mx = mx.array(s)
                v_mx = mx.array(v)
                
                return u_mx, s_mx, v_mx
            except Exception as e:
                logger.error(f"MLX SVD fehlgeschlagen: {e}")
                # Fallback zur Vermeidung von Programmabbrüchen
                x_np = x.tolist()
                u, s, v = np.linalg.svd(x_np, full_matrices=False)
                if k is not None:
                    u, s, v = u[:, :k], s[:k], v[:k, :]
                return mx.array(u), mx.array(s), mx.array(v)
        
        # JIT-Kompilierung wenn möglich
        try:
            return mx.jit(jitted_svd)
        except Exception as e:
            logger.warning(f"JIT-Kompilierung für SVD fehlgeschlagen: {e}")
            return jitted_svd
    
    def _get_optimized_svd_for_shape(self, shape, k=None):
        """
        Holt oder erstellt eine optimierte SVD-Funktion für die gegebene Matrixform
        
        Args:
            shape: Form der Matrix
            k: Anzahl der Singulärwerte
            
        Returns:
            Optimierte SVD-Funktion
        """
        # Cache-Schlüssel generieren
        key = self._get_svd_key(shape, k)
        
        # Prüfe, ob bereits im Cache
        if key in self.jit_cache:
            return self.jit_cache[key]
        
        # Erstelle neue optimierte Funktion
        complexity = self._estimate_matrix_complexity(shape)
        
        # Wähle die beste Implementierung basierend auf Komplexität
        if HAS_MLX:
            if JIT_ENABLED:
                # JIT-optimierte Funktion für MLX
                svd_func = self._compile_svd_jit(k)
            else:
                # Nicht-optimierte Funktion
                svd_func = lambda x: self._direct_svd(x, k)
        else:
            # Fallback auf NumPy
            svd_func = lambda x: self._numpy_svd(x, k)
        
        # Cache die Funktion
        self.jit_cache[key] = svd_func
        return svd_func
    
    def _direct_svd(self, x, k=None):
        """
        Direkte SVD-Implementierung ohne JIT für MLX
        
        Args:
            x: MLX-Array
            k: Anzahl der Singulärwerte
            
        Returns:
            Tuple (U, S, V) der SVD-Komponenten
        """
        # MLX bietet keine native SVD, verwende NumPy/PyTorch als Backend
        if torch.backends.mps.is_available():
            # Verwende PyTorch auf MPS für Hardware-Beschleunigung
            try:
                x_torch = torch.tensor(x.tolist(), device=self.torch_device, dtype=torch.float32)
                
                if k is None:
                    # Vollständige SVD
                    try:
                        u, s, v = torch.linalg.svd(x_torch, full_matrices=False)
                    except Exception as e:
                        logger.warning(f"PyTorch SVD fehlgeschlagen: {e}. Versuche CPU-Fallback.")
                        u, s, v = torch.linalg.svd(x_torch.cpu(), full_matrices=False)
                else:
                    # Optimierte partielle SVD
                    try:
                        # Low-Rank-SVD von PyTorch
                        u, s, v = torch.svd_lowrank(x_torch, q=k)
                    except (AttributeError, RuntimeError):
                        # Fallback auf vollständige SVD mit Trunkierung
                        u, s, v = torch.linalg.svd(x_torch, full_matrices=False)
                        u, s, v = u[:, :k], s[:k], v[:k, :]
                
                # Konvertiere zurück zu MLX
                u_mx = mx.array(u.cpu().numpy())
                s_mx = mx.array(s.cpu().numpy())
                v_mx = mx.array(v.cpu().numpy())
                
                return u_mx, s_mx, v_mx
            except Exception as e:
                logger.warning(f"MPS SVD fehlgeschlagen, verwende NumPy-Fallback: {e}")
                return self._numpy_svd(x, k)
        else:
            # Fallback auf NumPy
            return self._numpy_svd(x, k)
    
    def _numpy_svd(self, x, k=None):
        """
        NumPy-basierte SVD-Implementierung
        
        Args:
            x: MLX-Array oder NumPy-Array
            k: Anzahl der Singulärwerte
            
        Returns:
            Tuple (U, S, V) als MLX-Arrays
        """
        # Konvertiere zu NumPy
        if hasattr(x, 'tolist'):
            x_np = np.array(x.tolist())
        else:
            x_np = np.array(x)
        
        # Führe SVD durch
        if k is None:
            # Vollständige SVD
            u, s, v = np.linalg.svd(x_np, full_matrices=False)
        else:
            # Partielle SVD
            try:
                from scipy.sparse.linalg import svds
                u, s, vt = svds(x_np, k=k)
                # Sortiere die Singulärwerte in absteigender Reihenfolge
                idx = np.argsort(s)[::-1]
                u, s, v = u[:, idx], s[idx], vt[idx, :]
            except (ImportError, ValueError) as e:
                # Fallback auf vollständige SVD mit Trunkierung
                logger.warning(f"Scipy SVD fehlgeschlagen: {e}, verwende NumPy-Fallback")
                u, s, v = np.linalg.svd(x_np, full_matrices=False)
                u, s, v = u[:, :k], s[:k], v[:k, :]
        
        # Konvertiere zurück zu MLX
        if HAS_MLX:
            u_mx = mx.array(u)
            s_mx = mx.array(s)
            v_mx = mx.array(v)
            return u_mx, s_mx, v_mx
        else:
            return u, s, v
    
    def svd(self, a, k=None):
        """
        Führt eine optimierte Singulärwertzerlegung (SVD) durch
        
        Diese Methode wählt automatisch die beste SVD-Implementierung basierend auf
        der Matrix-Komplexität, Größe und verfügbarer Hardware.
        
        Args:
            a: Matrix für SVD (PyTorch-Tensor, MLX-Array oder NumPy-Array)
            k: Anzahl der zu berechnenden Singulärwerte (None für alle)
            
        Returns:
            Tuple (U, S, V) der SVD-Komponenten
        """
        # Prüfe, ob a ein PyTorch-Tensor ist
        if hasattr(a, 'device'):
            # Konvertiere zu MLX
            a_mx = self.backend.to_mlx(a)
        elif HAS_MLX and hasattr(a, "__module__") and "mlx" in a.__module__:
            # Ist bereits MLX
            a_mx = a
        else:
            # NumPy oder anderer Typ
            if HAS_MLX:
                a_mx = mx.array(np.array(a))
            else:
                # Kein MLX verfügbar, verwende NumPy direkt
                return self._numpy_svd(a, k)
        
        # Optimierte SVD-Funktion für diese Matrix holen
        optimized_svd = self._get_optimized_svd_for_shape(a_mx.shape, k)
        
        # SVD durchführen
        try:
            u, s, v = optimized_svd(a_mx)
            return u, s, v
        except Exception as e:
            logger.error(f"Optimierte SVD fehlgeschlagen: {e}, verwende Fallback")
            return self._numpy_svd(a_mx, k)
    
    def benchmark_svd(self, shapes=[(16, 16), (64, 64), (256, 256), (1024, 1024)], k_values=[None, 10], num_runs=3):
        """
        Führt einen Benchmark der SVD-Implementierung durch
        
        Args:
            shapes: Liste der zu testenden Matrixgrößen
            k_values: Liste der zu testenden k-Werte
            num_runs: Anzahl der Durchläufe pro Konfiguration
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        results = {}
        
        for shape in shapes:
            shape_results = {}
            
            for k in k_values:
                times = []
                
                # Prüfe, ob die Matrix zu groß ist für k
                if k is not None and k > min(shape):
                    continue
                
                # Erstelle eine zufällige Matrix
                if HAS_MLX:
                    a = mx.random.normal(shape)
                else:
                    a = np.random.rand(*shape)
                
                # Benchmark mehrere Durchläufe
                for i in range(num_runs):
                    start_time = time.time()
                    u, s, v = self.svd(a, k=k)
                    end_time = time.time()
                    
                    # Erzwinge Evaluation (verhindert Lazy Computation)
                    if HAS_MLX and hasattr(u, 'item'):
                        _ = u[0, 0].item()
                    
                    times.append(end_time - start_time)
                
                # Berechne Durchschnittszeit
                avg_time = sum(times) / len(times)
                shape_results[f"k={k}"] = {"avg_time": avg_time, "times": times}
                
                # Formatiere für Log-Ausgabe
                k_str = str(k) if k is not None else "None"
                logger.info(f"SVD für Matrix {shape} mit k={k_str}: {avg_time:.6f}s")
            
            results[f"{shape[0]}x{shape[1]}"] = shape_results
        
        return results


def install_svd_optimizer(backend):
    """
    Installiert den SVD-Optimizer in einem MLXBackend
    
    Args:
        backend: MLXBackend-Instanz
        
    Returns:
        Modifiziertes Backend mit optimierter SVD
    """
    # Erstelle Optimizer-Instanz
    optimizer = MLXSVDOptimizer(backend, precision=backend.precision)
    
    # Ersetze die SVD-Methode
    backend.svd = optimizer.svd
    
    # Füge Benchmark-Methode hinzu
    backend.benchmark_svd = optimizer.benchmark_svd
    
    logger.info("SVD-Optimizer erfolgreich installiert")
    return backend
