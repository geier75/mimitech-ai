#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimierte MLX-Operationen für T-Mathematics Engine

Diese Datei implementiert optimierte mathematische Kernoperationen mit MLX,
die von der verbesserten Speicherverwaltung profitieren.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import time
import math
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import lru_cache

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.optimized_mlx_operations")

# MLX-Verfügbarkeit prüfen
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    logger.warning("MLX konnte nicht importiert werden. Optimierte Operationen sind nicht verfügbar.")

# Importiere die optimierte Speicherverwaltung
from tensor_pool import TensorPool
from direct_memory_transfer import (
    DirectMemoryTransfer, mps_to_mlx, mlx_to_mps
)

# Singleton-Instance für TensorPool
_tensor_pool = TensorPool()

# Singleton-Instance für DirectMemoryTransfer
_memory_transfer = DirectMemoryTransfer()

class OptimizedMLXOperations:
    """
    Implementiert optimierte mathematische Operationen mit MLX.
    
    Diese Klasse bietet hochoptimierte Implementierungen der wichtigsten
    mathematischen Operationen, die von der verbesserten Speicherverwaltung
    und JIT-Kompilierung profitieren.
    """
    
    def __init__(self, dtype=None):
        """
        Initialisiert die optimierten MLX-Operationen.
        
        Args:
            dtype: Datentyp für MLX-Operationen (None für automatische Auswahl)
        """
        self.mlx_available = HAS_MLX
        
        # Setze Standard-Datentyp
        if dtype is None and HAS_MLX:
            self.dtype = mx.float16
        elif HAS_MLX:
            self.dtype = dtype
        else:
            self.dtype = None
        
        # JIT-kompilierte Operationen
        self.jit_cache = {}
        
        # Initialisiere JIT-kompilierte Funktionen, wenn MLX verfügbar ist
        if HAS_MLX:
            self._initialize_jit_functions()
    
    def _initialize_jit_functions(self):
        """Initialisiert JIT-kompilierte Funktionen für kritische Operationen"""
        if not hasattr(mx, 'jit'):
            logger.warning("MLX JIT nicht verfügbar, optimierte Operationen werden langsamer sein")
            return
        
        # Matrixmultiplikation
        @mx.jit
        def jit_matmul(a, b):
            return mx.matmul(a, b)
        
        self.jit_cache['matmul'] = jit_matmul
        
        # Transposition
        @mx.jit
        def jit_transpose(a, dims=None):
            if dims is None:
                # Standard-Transposition für 2D-Matrizen
                if len(a.shape) == 2:
                    return mx.transpose(a, (1, 0))
                else:
                    return a
            return mx.transpose(a, dims)
        
        self.jit_cache['transpose'] = jit_transpose
        
        # Singulärwertzerlegung (SVD)
        @mx.jit
        def jit_svd(a, full_matrices=False):
            try:
                return mx.linalg.svd(a, full_matrices=full_matrices)
            except Exception as e:
                logger.error(f"JIT-SVD fehlgeschlagen: {e}")
                # SVD ist komplex und kann in manchen Fällen fehlschlagen
                # Hier können wir keinen Fallback in JIT implementieren
                raise
        
        self.jit_cache['svd'] = jit_svd
        
        # Matrix-Inverse
        @mx.jit
        def jit_inv(a):
            return mx.linalg.inv(a)
        
        self.jit_cache['inv'] = jit_inv
        
        # Matrixaddition
        @mx.jit
        def jit_add(a, b):
            return a + b
        
        self.jit_cache['add'] = jit_add
        
        # Matrix-Subtraktion
        @mx.jit
        def jit_sub(a, b):
            return a - b
        
        self.jit_cache['sub'] = jit_sub
        
        # Element-weise Multiplikation
        @mx.jit
        def jit_multiply(a, b):
            return a * b
        
        self.jit_cache['multiply'] = jit_multiply
        
        # Element-weise Division
        @mx.jit
        def jit_divide(a, b):
            return a / b
        
        self.jit_cache['divide'] = jit_divide
        
        # Aktivierungsfunktionen
        @mx.jit
        def jit_relu(x):
            return mx.maximum(x, 0)
        
        self.jit_cache['relu'] = jit_relu
        
        @mx.jit
        def jit_sigmoid(x):
            return mx.sigmoid(x)
        
        self.jit_cache['sigmoid'] = jit_sigmoid
        
        @mx.jit
        def jit_tanh(x):
            return mx.tanh(x)
        
        self.jit_cache['tanh'] = jit_tanh
        
        # GELU-Aktivierung (wichtig für Transformer)
        @mx.jit
        def jit_gelu(x):
            return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * mx.power(x, 3))))
        
        self.jit_cache['gelu'] = jit_gelu
        
        # Softmax
        @mx.jit
        def jit_softmax(x, axis=-1):
            return mx.softmax(x, axis=axis)
        
        self.jit_cache['softmax'] = jit_softmax
        
        # Layer-Normalisierung
        @mx.jit
        def jit_layer_norm(x, epsilon=1e-5):
            mean = mx.mean(x, axis=-1, keepdims=True)
            var = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
            return (x - mean) / mx.sqrt(var + epsilon)
        
        self.jit_cache['layer_norm'] = jit_layer_norm
        
        # Attention-Mechanismus
        @mx.jit
        def jit_attention(q, k, v, mask=None, scale=None):
            # Berechne Skalierungsfaktor, wenn nicht angegeben
            if scale is None:
                scale = 1.0 / mx.sqrt(mx.array([q.shape[-1]], dtype=q.dtype))
            
            # Berechne Attention-Gewichte
            attn_weights = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
            
            # Wende Maske an, wenn vorhanden
            if mask is not None:
                attn_weights = attn_weights + mask
            
            # Softmax über die Aufmerksamkeitsgewichte
            attn_weights = mx.softmax(attn_weights, axis=-1)
            
            # Berechne Aufmerksamkeitsausgabe
            attn_output = mx.matmul(attn_weights, v)
            
            return attn_output
        
        self.jit_cache['attention'] = jit_attention
        
        logger.info(f"JIT-kompilierte Funktionen initialisiert: {list(self.jit_cache.keys())}")
    
    def matmul(self, a, b):
        """
        Führt eine optimierte Matrixmultiplikation durch.
        
        Args:
            a: Erste Matrix
            b: Zweite Matrix
            
        Returns:
            Ergebnis der Matrixmultiplikation
        """
        if not self.mlx_available:
            # Fallback zu NumPy/PyTorch
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return torch.matmul(a, b)
            else:
                # Konvertiere zu NumPy, wenn nötig
                a_np = a.cpu().numpy() if isinstance(a, torch.Tensor) else np.array(a)
                b_np = b.cpu().numpy() if isinstance(b, torch.Tensor) else np.array(b)
                return np.matmul(a_np, b_np)
        
        # Zeitmessung für Diagnose
        start_time = time.time()
        
        # Konvertiere zu MLX
        try:
            # Optimierte direkte Konvertierung
            if isinstance(a, torch.Tensor) and a.device.type == 'mps':
                a_mlx = mps_to_mlx(a)
            else:
                a_mlx = mx.array(a.cpu().numpy() if isinstance(a, torch.Tensor) else a, dtype=self.dtype)
            
            if isinstance(b, torch.Tensor) and b.device.type == 'mps':
                b_mlx = mps_to_mlx(b)
            else:
                b_mlx = mx.array(b.cpu().numpy() if isinstance(b, torch.Tensor) else b, dtype=self.dtype)
        except Exception as e:
            logger.warning(f"Konvertierung zu MLX für Matrixmultiplikation fehlgeschlagen: {e}")
            # Fallback zu NumPy/PyTorch
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return torch.matmul(a, b)
            else:
                a_np = a.cpu().numpy() if isinstance(a, torch.Tensor) else np.array(a)
                b_np = b.cpu().numpy() if isinstance(b, torch.Tensor) else np.array(b)
                return np.matmul(a_np, b_np)
        
        # Führe Matrixmultiplikation mit JIT durch
        if 'matmul' in self.jit_cache:
            result_mlx = self.jit_cache['matmul'](a_mlx, b_mlx)
        else:
            # Fallback ohne JIT
            result_mlx = mx.matmul(a_mlx, b_mlx)
        
        # Zeitmessung für Diagnose
        matmul_time = time.time() - start_time
        
        # Optimierte Rückkonvertierung
        if isinstance(a, torch.Tensor) and a.device.type == 'mps':
            result = mlx_to_mps(result_mlx)
        else:
            result = torch.from_numpy(result_mlx.numpy())
            if isinstance(a, torch.Tensor) and a.device.type == 'cuda':
                result = result.to(device='cuda')
        
        # Zeitmessung für Diagnose
        total_time = time.time() - start_time
        logger.debug(f"Matrixmultiplikation: {matmul_time:.6f}s, Gesamt: {total_time:.6f}s")
        
        # Rückgabe des Ergebnisses
        return result
    
    def svd(self, a, full_matrices=False, compute_uv=True):
        """
        Führt eine optimierte Singulärwertzerlegung (SVD) durch.
        
        Args:
            a: Eingangsmatrix
            full_matrices: Ob vollständige Matrizen berechnet werden sollen
            compute_uv: Ob U und V berechnet werden sollen
            
        Returns:
            Wenn compute_uv=True: Tuple (U, S, V) der SVD-Komponenten
            Sonst: Vektor S der Singulärwerte
        """
        if not self.mlx_available:
            # Fallback zu NumPy/SciPy
            try:
                if isinstance(a, torch.Tensor):
                    a_np = a.cpu().numpy()
                else:
                    a_np = np.array(a)
                
                if compute_uv:
                    u, s, vh = np.linalg.svd(a_np, full_matrices=full_matrices)
                    if isinstance(a, torch.Tensor):
                        u_torch = torch.from_numpy(u)
                        s_torch = torch.from_numpy(s)
                        vh_torch = torch.from_numpy(vh)
                        if a.device.type != 'cpu':
                            u_torch = u_torch.to(device=a.device)
                            s_torch = s_torch.to(device=a.device)
                            vh_torch = vh_torch.to(device=a.device)
                        return u_torch, s_torch, vh_torch
                    else:
                        return u, s, vh
                else:
                    s = np.linalg.svd(a_np, compute_uv=False)
                    if isinstance(a, torch.Tensor):
                        s_torch = torch.from_numpy(s)
                        if a.device.type != 'cpu':
                            s_torch = s_torch.to(device=a.device)
                        return s_torch
                    else:
                        return s
            except Exception as e:
                logger.error(f"NumPy/SciPy SVD fehlgeschlagen: {e}")
                raise
        
        # Konvertiere zu MLX
        try:
            # Optimierte direkte Konvertierung
            if isinstance(a, torch.Tensor) and a.device.type == 'mps':
                a_mlx = mps_to_mlx(a)
            else:
                a_mlx = mx.array(a.cpu().numpy() if isinstance(a, torch.Tensor) else a, dtype=self.dtype)
        except Exception as e:
            logger.warning(f"Konvertierung zu MLX für SVD fehlgeschlagen: {e}")
            # Fallback zu NumPy/SciPy
            return self.svd(a, full_matrices, compute_uv)
        
        # Führe SVD durch
        try:
            if compute_uv:
                if 'svd' in self.jit_cache:
                    u_mlx, s_mlx, v_mlx = self.jit_cache['svd'](a_mlx, full_matrices)
                else:
                    # Fallback ohne JIT
                    u_mlx, s_mlx, v_mlx = mx.linalg.svd(a_mlx, full_matrices=full_matrices)
                
                # Optimierte Rückkonvertierung
                if isinstance(a, torch.Tensor) and a.device.type == 'mps':
                    u = mlx_to_mps(u_mlx)
                    s = mlx_to_mps(s_mlx)
                    v = mlx_to_mps(v_mlx)
                else:
                    u = torch.from_numpy(u_mlx.numpy())
                    s = torch.from_numpy(s_mlx.numpy())
                    v = torch.from_numpy(v_mlx.numpy())
                    if isinstance(a, torch.Tensor) and a.device.type != 'cpu':
                        u = u.to(device=a.device)
                        s = s.to(device=a.device)
                        v = v.to(device=a.device)
                
                return u, s, v
            else:
                s_mlx = mx.linalg.svd(a_mlx, full_matrices=full_matrices, compute_uv=False)
                
                # Optimierte Rückkonvertierung
                if isinstance(a, torch.Tensor) and a.device.type == 'mps':
                    s = mlx_to_mps(s_mlx)
                else:
                    s = torch.from_numpy(s_mlx.numpy())
                    if isinstance(a, torch.Tensor) and a.device.type != 'cpu':
                        s = s.to(device=a.device)
                
                return s
        except Exception as e:
            logger.error(f"MLX SVD fehlgeschlagen: {e}")
            # Fallback zu NumPy/SciPy
            return self.svd(a, full_matrices, compute_uv)
    
    def attention(self, q, k, v, mask=None, scale=None):
        """
        Berechnet die Attention-Funktion mit optimierter Implementierung.
        
        Args:
            q: Query-Matrix
            k: Key-Matrix
            v: Value-Matrix
            mask: Optionale Attention-Maske
            scale: Skalierungsfaktor (None für 1/sqrt(d_k))
            
        Returns:
            Attention-Ausgabe
        """
        if not self.mlx_available:
            # Fallback zu PyTorch
            if not isinstance(q, torch.Tensor):
                q = torch.tensor(q)
            if not isinstance(k, torch.Tensor):
                k = torch.tensor(k)
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            
            # Berechne Skalierungsfaktor, wenn nicht angegeben
            if scale is None:
                scale = 1.0 / math.sqrt(q.size(-1))
            
            # Berechne Aufmerksamkeitsgewichte
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Wende Maske an, wenn vorhanden
            if mask is not None:
                attn_weights = attn_weights + mask
            
            # Softmax über die Aufmerksamkeitsgewichte
            attn_weights = torch.softmax(attn_weights, dim=-1)
            
            # Berechne Aufmerksamkeitsausgabe
            attn_output = torch.matmul(attn_weights, v)
            
            return attn_output
        
        # Konvertiere zu MLX
        try:
            # Optimierte direkte Konvertierung
            if isinstance(q, torch.Tensor) and q.device.type == 'mps':
                q_mlx = mps_to_mlx(q)
                k_mlx = mps_to_mlx(k)
                v_mlx = mps_to_mlx(v)
                mask_mlx = mps_to_mlx(mask) if mask is not None else None
            else:
                q_mlx = mx.array(q.cpu().numpy() if isinstance(q, torch.Tensor) else q, dtype=self.dtype)
                k_mlx = mx.array(k.cpu().numpy() if isinstance(k, torch.Tensor) else k, dtype=self.dtype)
                v_mlx = mx.array(v.cpu().numpy() if isinstance(v, torch.Tensor) else v, dtype=self.dtype)
                mask_mlx = mx.array(mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask, dtype=self.dtype) if mask is not None else None
        except Exception as e:
            logger.warning(f"Konvertierung zu MLX für Attention fehlgeschlagen: {e}")
            # Fallback zu PyTorch
            return self.attention(q, k, v, mask, scale)
        
        # Führe Attention mit JIT durch
        if 'attention' in self.jit_cache:
            result_mlx = self.jit_cache['attention'](q_mlx, k_mlx, v_mlx, mask_mlx, scale)
        else:
            # Fallback ohne JIT
            # Berechne Skalierungsfaktor, wenn nicht angegeben
            if scale is None:
                scale = 1.0 / math.sqrt(q_mlx.shape[-1])
            
            # Berechne Aufmerksamkeitsgewichte
            attn_weights = mx.matmul(q_mlx, mx.transpose(k_mlx, (0, 1, 3, 2))) * scale
            
            # Wende Maske an, wenn vorhanden
            if mask_mlx is not None:
                attn_weights = attn_weights + mask_mlx
            
            # Softmax über die Aufmerksamkeitsgewichte
            attn_weights = mx.softmax(attn_weights, axis=-1)
            
            # Berechne Aufmerksamkeitsausgabe
            result_mlx = mx.matmul(attn_weights, v_mlx)
        
        # Optimierte Rückkonvertierung
        if isinstance(q, torch.Tensor) and q.device.type == 'mps':
            result = mlx_to_mps(result_mlx)
        else:
            result = torch.from_numpy(result_mlx.numpy())
            if isinstance(q, torch.Tensor) and q.device.type != 'cpu':
                result = result.to(device=q.device)
        
        return result
    
    def layer_norm(self, x, epsilon=1e-5):
        """
        Führt eine optimierte Layer-Normalisierung durch.
        
        Args:
            x: Eingabetensor
            epsilon: Kleiner Wert für numerische Stabilität
            
        Returns:
            Normalisierter Tensor
        """
        if not self.mlx_available:
            # Fallback zu PyTorch
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            
            # Berechne Mittelwert und Varianz
            mean = x.mean(dim=-1, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            
            # Normalisiere
            return (x - mean) / torch.sqrt(var + epsilon)
        
        # Konvertiere zu MLX
        try:
            # Optimierte direkte Konvertierung
            if isinstance(x, torch.Tensor) and x.device.type == 'mps':
                x_mlx = mps_to_mlx(x)
            else:
                x_mlx = mx.array(x.cpu().numpy() if isinstance(x, torch.Tensor) else x, dtype=self.dtype)
        except Exception as e:
            logger.warning(f"Konvertierung zu MLX für Layer-Norm fehlgeschlagen: {e}")
            # Fallback zu PyTorch
            return self.layer_norm(x, epsilon)
        
        # Führe Layer-Norm mit JIT durch
        if 'layer_norm' in self.jit_cache:
            result_mlx = self.jit_cache['layer_norm'](x_mlx, epsilon)
        else:
            # Fallback ohne JIT
            mean = mx.mean(x_mlx, axis=-1, keepdims=True)
            var = mx.mean(mx.square(x_mlx - mean), axis=-1, keepdims=True)
            result_mlx = (x_mlx - mean) / mx.sqrt(var + epsilon)
        
        # Optimierte Rückkonvertierung
        if isinstance(x, torch.Tensor) and x.device.type == 'mps':
            result = mlx_to_mps(result_mlx)
        else:
            result = torch.from_numpy(result_mlx.numpy())
            if isinstance(x, torch.Tensor) and x.device.type != 'cpu':
                result = result.to(device=x.device)
        
        return result


# Singleton-Instance für OptimizedMLXOperations
_optimized_mlx_ops = None

def get_optimized_mlx_operations(dtype=None):
    """
    Liefert eine Singleton-Instance der OptimizedMLXOperations.
    
    Args:
        dtype: Datentyp für MLX-Operationen
        
    Returns:
        OptimizedMLXOperations-Instance
    """
    global _optimized_mlx_ops
    if _optimized_mlx_ops is None:
        _optimized_mlx_ops = OptimizedMLXOperations(dtype)
    return _optimized_mlx_ops

# Praktische Hilfsfunktionen für einfachen Zugriff

def optimized_matmul(a, b):
    """
    Führt eine optimierte Matrixmultiplikation durch.
    
    Args:
        a: Erste Matrix
        b: Zweite Matrix
        
    Returns:
        Ergebnis der Matrixmultiplikation
    """
    return get_optimized_mlx_operations().matmul(a, b)

def optimized_svd(a, full_matrices=False, compute_uv=True):
    """
    Führt eine optimierte Singulärwertzerlegung (SVD) durch.
    
    Args:
        a: Eingangsmatrix
        full_matrices: Ob vollständige Matrizen berechnet werden sollen
        compute_uv: Ob U und V berechnet werden sollen
        
    Returns:
        Wenn compute_uv=True: Tuple (U, S, V) der SVD-Komponenten
        Sonst: Vektor S der Singulärwerte
    """
    return get_optimized_mlx_operations().svd(a, full_matrices, compute_uv)

def optimized_attention(q, k, v, mask=None, scale=None):
    """
    Berechnet die Attention-Funktion mit optimierter Implementierung.
    
    Args:
        q: Query-Matrix
        k: Key-Matrix
        v: Value-Matrix
        mask: Optionale Attention-Maske
        scale: Skalierungsfaktor (None für 1/sqrt(d_k))
        
    Returns:
        Attention-Ausgabe
    """
    return get_optimized_mlx_operations().attention(q, k, v, mask, scale)

def optimized_layer_norm(x, epsilon=1e-5):
    """
    Führt eine optimierte Layer-Normalisierung durch.
    
    Args:
        x: Eingabetensor
        epsilon: Kleiner Wert für numerische Stabilität
        
    Returns:
        Normalisierter Tensor
    """
    return get_optimized_mlx_operations().layer_norm(x, epsilon)
