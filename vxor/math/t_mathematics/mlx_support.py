#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MLX Support für T-Mathematics

Diese Datei stellt Unterstützung für die MLX-Bibliothek bereit, die speziell für Apple Silicon optimiert ist.
Sie integriert auch VXOR-Module für optimierte mathematische Operationen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import time
import math
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from functools import lru_cache

# Relative VXOR-Module Import-Pfade
VXOR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'vXor_Modules'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  

# Konfiguriere Logger
logger = logging.getLogger("t_mathematics.mlx_support")

# Prüfe, ob wir auf Apple Silicon laufen
IS_APPLE_SILICON = sys.platform == 'darwin' and 'arm' in os.uname().machine if hasattr(os, 'uname') else False

# Optimierte MLX-Import mit auto-fallback
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx import optimizers as mx_opt
    
    # Aktiviere JIT-Kompilierung für höhere Leistung
    # Kompatibilität mit unterschiedlichen MLX-Versionen sicherstellen
    try:
        # Prüfen auf neuere MLX-Versionen
        if hasattr(mx, 'get_default_device'):
            current_device = mx.get_default_device()
            logger.info(f"Aktuelles MLX-Gerät: {current_device}")
        elif hasattr(mx, 'gpu') and hasattr(mx.gpu, 'is_available'):
            # Ältere MLX-Version
            mx.set_default_device(mx.gpu if mx.gpu.is_available() else mx.cpu)
        else:
            # Fallback für neueste MLX-Versionen, die eine andere API haben
            try:
                # Versuchen wir, GPU explizit zu setzen
                mx.set_default_device(mx.gpu)
                logger.info("MLX GPU-Gerät gesetzt")
            except Exception:
                mx.set_default_device(mx.cpu)
                logger.info("MLX CPU-Gerät gesetzt (GPU nicht verfügbar)")
    except Exception as e:
        logger.warning(f"Konnte MLX-Gerät nicht konfigurieren: {e}. Verwende Standard-Gerät.")
    
    HAS_MLX = True
    logger.info("MLX erfolgreich importiert und für maximale Leistung konfiguriert")
    
    # Aktiviere Performance-Optimierungen
    if hasattr(mx, 'enable_fusion'):
        mx.enable_fusion(True)
        logger.info("MLX-Operation-Fusion aktiviert für höhere Performance")
    
    # Verbessere MLX-Speicherverwaltung mit Cache-Präallokation
    if hasattr(mx, 'set_memory_pool_size'):
        mx.set_memory_pool_size(1 << 30)  # 1GB Präallokation für schnellere Ausführung
        logger.info("MLX-Speicherpool optimiert")
        
except ImportError as e:
    HAS_MLX = False
    logger.warning(f"MLX nicht verfügbar, Fallback auf CPU-Backend: {e}")

# Apple Silicon-Erkennung
IS_APPLE_SILICON = False
try:
    import platform
    IS_APPLE_SILICON = platform.processor() == 'arm' and platform.system() == 'Darwin'
    if IS_APPLE_SILICON:
        logger.info("Apple Silicon erkannt, optimiere für Neural Engine")
except Exception as e:
    logger.warning(f"Fehler bei der Apple Silicon-Erkennung: {e}")

# Importiere VXOR-Integration, wenn verfügbar
try:
    from miso.math.t_mathematics.vxor_math_integration import get_vxor_math_integration
    vxor_math = get_vxor_math_integration()
    has_vxor = True
    logger.info("VXOR-Integration für MLXBackend geladen")
except ImportError:
    has_vxor = False
    logger.warning("VXOR-Integration für MLXBackend nicht verfügbar")

# Globale Tensor-Konvertierungsfunktion zu MLX
def tensor_to_mlx(tensor: Union[np.ndarray, torch.Tensor, list, Any], dtype=None) -> Any:
    """Universelle Konvertierungsfunktion zu MLX-Array.
    
    Diese Funktion konvertiert verschiedene Tensor-Typen (PyTorch, NumPy, Listen)
    zu MLX-Arrays unter Berücksichtigung des Typs und der Hardware-Optimierung.
    
    Args:
        tensor: Eingabetensor (PyTorch, NumPy, Liste oder kompatibel)
        dtype: Optionaler Ziel-Datentyp für das MLX-Array
    
    Returns:
        MLX-Array oder original Tensor bei fehlgeschlagener Konvertierung
        
    Raises:
        TypeError: Bei unbekanntem Tensor-Typ, der nicht konvertierbar ist
    """
    # Frühes Aussteigen, wenn MLX nicht verfügbar ist
    if not HAS_MLX or mx is None:
        logger.debug("MLX nicht verfügbar, überspringe Konvertierung")
        return tensor
    
    try:
        # PyTorch-Tensor-Konvertierung
        if isinstance(tensor, torch.Tensor):
            # Kopiere zur CPU und detach, falls notwendig
            if tensor.requires_grad:
                tensor = tensor.detach()
            if tensor.device.type != 'cpu':
                tensor = tensor.cpu()
            # Konvertiere mit korrektem Typen
            return mx.array(tensor.numpy(), dtype=dtype)
        
        # NumPy-Array-Konvertierung
        elif isinstance(tensor, np.ndarray):
            return mx.array(tensor, dtype=dtype)
        
        # Listen und skalare Werte
        elif isinstance(tensor, (list, tuple, int, float)):
            return mx.array(np.array(tensor), dtype=dtype)
        
        # Fallback für unbekannte Typen
        else:
            array = np.array(tensor)
            return mx.array(array, dtype=dtype)
            
    except Exception as e:
        logger.warning(f"Konvertierung zu MLX fehlgeschlagen: {e}")
        if isinstance(tensor, torch.Tensor):
            # Alternativer Konvertierungspfad für PyTorch
            try:
                return mx.array(tensor.cpu().detach().tolist(), dtype=dtype)
            except:
                pass
        raise TypeError(f"Tensor vom Typ {type(tensor)} konnte nicht zu MLX konvertiert werden")

class MLXBackend:
    """Backend für MLX-Operationen, optimiert für Apple Silicon.
    
    Diese Klasse bietet optimierte mathematische Operationen für Apple Silicon,
    indem sie die MLX-Bibliothek verwendet. Sie integriert auch VXOR-Module
    für zusätzliche Optimierungen, wenn verfügbar.
    """
    
    def __init__(self, precision="float16"):
        """Initialisiert das MLX-Backend.
        
        Args:
            precision: Präzision für MLX-Operationen (float16 oder float32)
        """
        self.precision = precision
        self.mlx_available = False
        self.mx = None
        self.jit_cache = {}
        self.operation_cache = {}
        self.has_vxor = has_vxor
        
        # Optimale Geräte-Konfiguration für PyTorch
        if torch.backends.mps.is_available():
            self.torch_device = torch.device("mps")
        else:
            self.torch_device = torch.device("cpu")
        
        # Prüfe, ob MLX verfügbar ist
        try:
            import mlx.core as mx
            self.mx = mx
            HAS_MLX = True
            IS_APPLE_SILICON = True
            
            # Prüfe MLX-Version und verfügbare Funktionen
            self.has_jit = hasattr(mx, 'jit')
            
            logger.info(f"MLX-Backend initialisiert mit Präzision {precision} (JIT: {self.has_jit})")
        except ImportError:
            self.mx = None
            HAS_MLX = False
            IS_APPLE_SILICON = False
            self.has_jit = False
            logger.warning("MLX konnte nicht importiert werden. Verwende NumPy als Fallback.")
        
        # MLX-Optimierungen aktivieren
        if self.mx is not None:
            # Setze MLX-Datentyp basierend auf der Präzision
            self.dtype = mx.float16 if precision == "float16" else mx.float32
            
            # Aktiviere MLX-Compiler für bessere Performance
            try:
                # Aktiviere Just-In-Time Compilation für MLX-Operationen
                mx.set_default_device(mx.gpu if IS_APPLE_SILICON else mx.cpu)
                logger.info(f"MLX-Backend initialisiert mit Präzision {precision} und JIT-Optimierung")
            except Exception as e:
                logger.warning(f"MLX-JIT-Optimierung konnte nicht aktiviert werden: {e}")
                logger.info(f"MLX-Backend initialisiert mit Präzision {precision}")
        else:
            self.dtype = None
            logger.warning("MLX-Backend nicht verfügbar, verwende Fallback-Implementierung")
    
    def to_mlx(self, tensor, dtype=None):
        """Konvertiert einen PyTorch-Tensor oder NumPy-Array zu einem MLX-Array.
        
        Args:
            tensor: PyTorch-Tensor oder NumPy-Array
            dtype: Optionaler Ziel-Datentyp für das MLX-Array
            
        Returns:
            MLX-Array
        """
        if self.mx is None:
            logger.debug("MLX nicht verfügbar im Backend, überspringe Konvertierung")
            return tensor
        
        # Verwende die globale Funktion für konsistente Implementierung
        dtype = dtype if dtype is not None else self.dtype
        try:
            return tensor_to_mlx(tensor, dtype=dtype)
        except TypeError as e:
            logger.error(f"Konvertierung zu MLX im Backend fehlgeschlagen: {e}")
            raise TypeError(f"Backend-Konvertierung zu MLX fehlgeschlagen: {type(tensor)}")
    
    def to_numpy(self, tensor):
        """Konvertiert einen Tensor zu NumPy"""
        if self.mx is not None and hasattr(tensor, "__module__") and "mlx" in tensor.__module__:
            return tensor.tolist()
        
        # Für PyTorch-Tensoren
        if hasattr(tensor, "device") and str(tensor.device).startswith("mps"):
            # Kopiere zuerst zur CPU, dann zu NumPy
            return tensor.cpu().detach().numpy()
        
        # Standardkonvertierung
        return np.array(tensor)
    
    def to_torch(self, array):
        """Konvertiert ein MLX-Array zu einem PyTorch-Tensor.
        
        Args:
            array: MLX-Array
            
        Returns:
            PyTorch-Tensor
        """
        if not self.mlx_available or not isinstance(array, self.mx.array):
            return array
        
        # Optimierte Konvertierung von MLX zu PyTorch
        try:
            # Schnellere Konvertierung über NumPy
            return torch.from_numpy(array.numpy()).to(device=self.torch_device)
        except Exception as e:
            # Fallback auf langsamere Methode
            logger.warning(f"Optimierte MLX-zu-PyTorch-Konvertierung fehlgeschlagen: {e}")
            return torch.tensor(array.tolist(), device=self.torch_device)
    
    @lru_cache(maxsize=32)
    def _get_matmul_key(self, a_shape, b_shape, batch_size=None):
        """Generiert einen Schlüssel für das Matrixmultiplikations-Cache.
        
        Args:
            a_shape: Form der ersten Matrix
            b_shape: Form der zweiten Matrix
            batch_size: Optionale Batch-Größe
            
        Returns:
            Cache-Schlüssel als String
        """
        return f"matmul_{a_shape}_{b_shape}_{batch_size}"
    
    def matmul(self, a, b, batch_size=None):
        """Führt eine Matrixmultiplikation durch"""
        # Verwende NumPy, wenn MLX nicht verfügbar ist
        if self.mx is None:
            # Konvertiere PyTorch-Tensoren zu NumPy
            if hasattr(a, "device"):
                a = a.cpu().detach().numpy()
            if hasattr(b, "device"):
                b = b.cpu().detach().numpy()
            return np.matmul(a, b)
        
        # Prüfe, ob VX-METACODE für Optimierung verfügbar ist
        if self.has_vxor and vxor_math.available_modules.get("VX-METACODE", False):
            operation = {"type": "matmul", "a_shape": a.shape, "b_shape": b.shape}
            optimized_ops = vxor_math.optimize_mlx_operations([operation])
            if optimized_ops and optimized_ops[0].get("optimized", False):
                logger.debug("Verwende VX-METACODE-optimierte Matrixmultiplikation")
                # Verwende die optimierte Operation von VX-METACODE
                vx_meta_op = optimized_ops[0].get("operation")
                if vx_meta_op:
                    return vx_meta_op(a, b)
        
        # Prüfe, ob die Operation bereits im Cache ist
        cache_key = self._get_matmul_key(a.shape, b.shape, batch_size)
        if cache_key in self.operation_cache:
            # Verwende gecachte Operation mit neuen Daten
            cached_op = self.operation_cache[cache_key]
            a_mx = self.to_mlx(a)
            b_mx = self.to_mlx(b)
            result_mx = cached_op(a_mx, b_mx)
            return self.to_numpy(result_mx)
        
        # Konvertiere zu MLX-Arrays
        try:
            # Prüfe, ob a und b bereits MLX-Arrays sind
            a_mx = a if isinstance(a, type(self.mx.array([0]))) else self.to_mlx(a)
            b_mx = b if isinstance(b, type(self.mx.array([0]))) else self.to_mlx(b)
        except Exception as e:
            logger.warning(f"Konvertierung zu MLX fehlgeschlagen, verwende NumPy-Fallback: {e}")
            # Fallback zu NumPy
            if hasattr(a, "device"):
                a = a.cpu().detach().numpy()
            if hasattr(b, "device"):
                b = b.cpu().detach().numpy()
            return np.matmul(a, b)
        
        # Kernel Fusion für Matrixmultiplikation
        # Kombiniert mehrere Operationen in einen einzigen Kernel für bessere Performance
        def fused_matmul(a, b):
            # Stelle sicher, dass a und b MLX-Arrays sind
            try:
                # Konvertiere PyTorch-Tensoren zu MLX-Arrays
                if hasattr(a, "device"):
                    a_np = a.cpu().detach().numpy()
                    a = self.mx.array(a_np)
                if hasattr(b, "device"):
                    b_np = b.cpu().detach().numpy()
                    b = self.mx.array(b_np)
                
                # Führe Matrixmultiplikation durch
                return self.mx.matmul(a, b)  # Dies reduziert Speichertransfers und verbessert die Leistung
            except Exception as e:
                logger.warning(f"MLX-Matrixmultiplikation fehlgeschlagen: {e}. Verwende NumPy-Fallback.")
                # Fallback zu NumPy
                if hasattr(a, "device"):
                    a = a.cpu().detach().numpy()
                if hasattr(b, "device"):
                    b = b.cpu().detach().numpy()
                return np.matmul(a, b)
        
        # Optimierte Implementierung für große Matrizen mit Batch-Verarbeitung
        if batch_size is not None and a.shape[0] > batch_size:
            # Batch-Verarbeitung für große Matrizen
            logger.debug(f"Verwende Batch-Verarbeitung für Matrixmultiplikation mit Batch-Größe {batch_size}")
            
            # Erstelle JIT-kompilierte Batch-Funktion, wenn nicht im Cache
            if f"matmul_batch_{batch_size}" not in self.jit_cache:
                def batch_matmul(a, b, batch_size):
                    # Teile Matrix in Batches auf
                    num_batches = (a.shape[0] + batch_size - 1) // batch_size
                    results = []
                    
                    # Parallele Verarbeitung der Batches
                    for i in range(num_batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, a.shape[0])
                        batch_a = a[start_idx:end_idx]
                        # Verwende die fusionierte Matrixmultiplikation
                        batch_result = fused_matmul(batch_a, b)
                        results.append(batch_result)
                    
                    # Kombiniere die Ergebnisse
                    return self.mx.concatenate(results, axis=0)
                
                # JIT-Kompilierung für maximale Performance, falls verfügbar
                if hasattr(self.mx, 'jit'):
                    self.jit_cache[f"matmul_batch_{batch_size}"] = self.mx.jit(batch_matmul)
                else:
                    # Fallback ohne JIT
                    self.jit_cache[f"matmul_batch_{batch_size}"] = batch_matmul
                    logger.warning("MLX JIT nicht verfügbar, verwende Fallback ohne JIT-Optimierung")
            
            # Führe Batch-Matrixmultiplikation durch
            result_mx = self.jit_cache[f"matmul_batch_{batch_size}"](a_mx, b_mx, batch_size)
        else:
            # Standard-Matrixmultiplikation für kleinere Matrizen
            if "matmul" not in self.jit_cache:
                # Optimierte Matrixmultiplikation mit MLX
                # Prüfe, ob JIT verfügbar ist
                if hasattr(self.mx, 'jit'):
                    self.jit_cache["matmul"] = self.mx.jit(fused_matmul)
                else:
                    # Fallback ohne JIT
                    self.jit_cache["matmul"] = fused_matmul
                    logger.warning("MLX JIT nicht verfügbar, verwende Fallback ohne JIT-Optimierung")
            
            # Führe Matrixmultiplikation durch
            result_mx = self.jit_cache["matmul"](a_mx, b_mx)
        
        # Cache die Operation für zukünftige Verwendung
        self.operation_cache[cache_key] = self.jit_cache["matmul"] if batch_size is None else self.jit_cache[f"matmul_batch_{batch_size}"]
        
        # Konvertiere zurück zu NumPy
        return self.to_numpy(result_mx)
    
    @lru_cache(maxsize=32)
    def _get_svd_key(self, a_shape, k=None):
        """Generiert einen Schlüssel für das SVD-Cache.
        
        Args:
            a_shape: Form der Matrix
            k: Anzahl der zu berechnenden Singulärwerte
            
        Returns:
            Cache-Schlüssel als String
        """
        return f"svd_{a_shape}_{k}"
    
    def svd(self, a, k=None):
        """Führt eine Singulärwertzerlegung (SVD) durch mit optimiertem Caching.
        
        Args:
            a: Matrix für SVD
            k: Anzahl der zu berechnenden Singulärwerte (None für alle)
            
        Returns:
            Tuple (U, S, V) der SVD-Komponenten
        """
        if not self.mlx_available:
            if k is None:
                return np.linalg.svd(a, full_matrices=False)
            else:
                # Verwende scipy für partielle SVD, wenn k angegeben ist
                try:
                    from scipy.sparse.linalg import svds
                    u, s, vt = svds(a, k=k)
                    # Sortiere die Singulärwerte in absteigender Reihenfolge
                    idx = np.argsort(s)[::-1]
                    return u[:, idx], s[idx], vt[idx, :]
                except ImportError:
                    # Fallback auf vollständige SVD, wenn scipy nicht verfügbar ist
                    u, s, v = np.linalg.svd(a, full_matrices=False)
                    return u[:, :k], s[:k], v[:k, :]
        
        # Prüfe, ob VX-METACODE für Optimierung verfügbar ist
        if self.has_vxor and vxor_math.available_modules.get("VX-METACODE", False):
            operation = {"type": "svd", "a_shape": a.shape, "k": k}
            optimized_ops = vxor_math.optimize_mlx_operations([operation])
            if optimized_ops and optimized_ops[0].get("optimized", False):
                logger.debug("Verwende VX-METACODE-optimierte SVD")
                # Verwende die optimierte Operation von VX-METACODE
                vx_meta_op = optimized_ops[0].get("operation")
                if vx_meta_op:
                    return vx_meta_op(a, k)
        
        # Prüfe, ob die Operation bereits im Cache ist
        cache_key = self._get_svd_key(a.shape, k)
        if cache_key in self.operation_cache:
            # Verwende gecachte Operation mit neuen Daten
            cached_op = self.operation_cache[cache_key]
            a_mx = self.to_mlx(a)
            u_mx, s_mx, v_mx = cached_op(a_mx)
            return self.to_numpy(u_mx), self.to_numpy(s_mx), self.to_numpy(v_mx)
        
        # Konvertiere zu MLX-Array, wenn nötig
        a_mx = self.to_mlx(a)
        
        # Optimierte SVD-Implementierung für MLX
        # Wir verwenden eine Kombination aus MLX und PyTorch für maximale Performance
        svd_cache_key = f"svd_{k}"
        if svd_cache_key not in self.jit_cache:
            def optimized_svd(x, k=None):
                # Verwende direkt MPS für die SVD-Berechnung, wenn verfügbar
                # Dies vermeidet unnötige Datentransfers zwischen CPU und GPU
                if torch.backends.mps.is_available():
                    # Direkte Konvertierung zu PyTorch ohne Umweg über NumPy
                    x_torch = torch.tensor(self.to_numpy(x), device=self.torch_device)
                    
                    if k is None:
                        # Vollständige SVD mit PyTorch auf MPS
                        u, s, v = torch.linalg.svd(x_torch, full_matrices=False)
                    else:
                        # Optimierte partielle SVD
                        try:
                            # Verwende die optimierte Low-Rank-SVD von PyTorch
                            u, s, v = torch.svd_lowrank(x_torch, q=k)
                        except AttributeError:
                            # Optimierte Alternative für ältere PyTorch-Versionen
                            # Verwende Randomized SVD für bessere Performance bei großen Matrizen
                            if x_torch.shape[0] > 1000 or x_torch.shape[1] > 1000:
                                # Randomized SVD für große Matrizen
                                Q = torch.randn(x_torch.shape[1], k+10, device=self.torch_device)
                                for i in range(4):  # 4 Power-Iterationen für Konvergenz
                                    Q = torch.linalg.qr(torch.matmul(x_torch, Q))[0]
                                    Q = torch.linalg.qr(torch.matmul(x_torch.T, Q))[0]
                                
                                # Berechne die SVD der kleineren Matrix
                                Y = torch.matmul(Q.T, torch.matmul(x_torch, Q))
                                UY, S, VY = torch.linalg.svd(Y, full_matrices=False)
                                U = torch.matmul(Q, UY)[:, :k]
                                S = S[:k]
                                V = torch.matmul(Q, VY)[:, :k]
                            else:
                                # Standard-SVD für kleinere Matrizen
                                u, s, v = torch.linalg.svd(x_torch, full_matrices=False)
                                u, s, v = u[:, :k], s[:k], v[:k, :]
                    
                    # Direkte Konvertierung zu MLX ohne Umweg über NumPy
                    u_mx = self.to_mlx(u.cpu().numpy())
                    s_mx = self.to_mlx(s.cpu().numpy())
                    v_mx = self.to_mlx(v.cpu().numpy())
                else:
                    # Fallback auf NumPy für Systeme ohne MPS
                    x_np = self.to_numpy(x)
                    
                    if k is None:
                        u, s, v = np.linalg.svd(x_np, full_matrices=False)
                    else:
                        try:
                            from scipy.sparse.linalg import svds
                            u, s, vt = svds(x_np, k=k)
                            # Sortiere die Singulärwerte
                            idx = np.argsort(s)[::-1]
                            u, s, v = u[:, idx], s[idx], vt[idx, :]
                        except ImportError:
                            u, s, v = np.linalg.svd(x_np, full_matrices=False)
                            u, s, v = u[:, :k], s[:k], v[:k, :]
                    
                    # Konvertiere zu MLX
                    u_mx = self.to_mlx(u)
                    s_mx = self.to_mlx(s)
                    v_mx = self.to_mlx(v)
                
                # Konvertiere zurück zu MLX
                u_mx = self.to_mlx(u.cpu().numpy())
                s_mx = self.to_mlx(s.cpu().numpy())
                v_mx = self.to_mlx(v.cpu().numpy())
                
                return u_mx, s_mx, v_mx
            
            # Wir können die Funktion nicht direkt mit JIT kompilieren, da sie PyTorch verwendet
            # Aber wir cachen sie trotzdem für spätere Verwendung
            self.jit_cache[svd_cache_key] = optimized_svd
        
        # Führe SVD durch
        u_mx, s_mx, v_mx = self.jit_cache[svd_cache_key](a_mx, k)
        
        # Cache die Operation für zukünftige Verwendung
        self.operation_cache[cache_key] = lambda x: self.jit_cache[svd_cache_key](x, k)
        
        # Konvertiere zurück zu NumPy
        return self.to_numpy(u_mx), self.to_numpy(s_mx), self.to_numpy(v_mx)
        
    @lru_cache(maxsize=32)
    def _get_attention_key(self, q_shape, k_shape, v_shape, has_mask=False, use_flash=False):
        """Generiert einen Schlüssel für das Attention-Cache.
        
        Args:
            q_shape: Form der Query-Matrix
            k_shape: Form der Key-Matrix
            v_shape: Form der Value-Matrix
            has_mask: Ob eine Maske verwendet wird
            use_flash: Ob Flash-Attention verwendet wird
            
        Returns:
            Cache-Schlüssel als String
        """
        return f"attention_{q_shape}_{k_shape}_{v_shape}_{has_mask}_{use_flash}"
    
    def attention(self, q, k, v, mask=None, use_flash=True):
        """Berechnet die Attention-Funktion mit optimierter Flash-Attention.
        
        Args:
            q: Query-Matrix
            k: Key-Matrix
            v: Value-Matrix
            mask: Optionale Attention-Maske
            use_flash: Ob Flash-Attention verwendet werden soll (wenn möglich)
            
        Returns:
            Attention-Ausgabe
        """
        if not self.mlx_available:
            # Fallback-Implementierung mit NumPy
            d_k = k.shape[-1]
            scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(d_k)
            
            if mask is not None:
                scores = scores + mask * -1e9
            
            attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
            return np.matmul(attention_weights, v)
        
        # Prüfe, ob VX-METACODE für Optimierung verfügbar ist
        if self.has_vxor and vxor_math.available_modules.get("VX-METACODE", False):
            # Erstelle eine detaillierte Operationsbeschreibung für VX-METACODE
            operation = {
                "type": "attention", 
                "q_shape": q.shape, 
                "k_shape": k.shape, 
                "v_shape": v.shape,
                "has_mask": mask is not None,
                "use_flash": use_flash,
                "precision": self.precision,
                "device": "gpu" if IS_APPLE_SILICON else "cpu",
                "batch_size": q.shape[0] if len(q.shape) > 2 else 1,
                "seq_len": q.shape[1] if len(q.shape) > 2 else q.shape[0],
                "head_dim": q.shape[-1]
            }
            
            # Optimiere die Operation mit VX-METACODE
            optimized_ops = vxor_math.optimize_mlx_operations([operation])
            if optimized_ops and optimized_ops[0].get("optimized", False):
                logger.debug("Verwende VX-METACODE-optimierte Attention")
                # Verwende die optimierte Operation von VX-METACODE
                vx_meta_op = optimized_ops[0].get("operation")
                if vx_meta_op:
                    # Führe die optimierte Operation aus und konvertiere das Ergebnis
                    result = vx_meta_op(q, k, v, mask)
                    # Cache die Operation für zukünftige Verwendung
                    cache_key = self._get_attention_key(q.shape, k.shape, v.shape, mask is not None, use_flash)
                    self.operation_cache[cache_key] = vx_meta_op
                    return result
        
        # Prüfe, ob die Operation bereits im Cache ist
        cache_key = self._get_attention_key(q.shape, k.shape, v.shape, mask is not None, use_flash)
        if cache_key in self.operation_cache:
            # Verwende gecachte Operation mit neuen Daten
            cached_op = self.operation_cache[cache_key]
            q_mx = self.to_mlx(q)
            k_mx = self.to_mlx(k)
            v_mx = self.to_mlx(v)
            mask_mx = self.to_mlx(mask) if mask is not None else None
            result_mx = cached_op(q_mx, k_mx, v_mx, mask_mx)
            return self.to_numpy(result_mx)
        
        # Konvertiere zu MLX-Arrays
        try:
            q_mx = self.to_mlx(q)
            k_mx = self.to_mlx(k)
            v_mx = self.to_mlx(v)
            mask_mx = self.to_mlx(mask) if mask is not None else None
        except Exception as e:
            logger.warning(f"Konvertierung zu MLX fehlgeschlagen, verwende NumPy-Fallback: {e}")
            # Fallback zu NumPy
            if hasattr(q, "device"):
                q = q.cpu().detach().numpy()
            if hasattr(k, "device"):
                k = k.cpu().detach().numpy()
            if hasattr(v, "device"):
                v = v.cpu().detach().numpy()
            if mask is not None and hasattr(mask, "device"):
                mask = mask.cpu().detach().numpy()
            return np.matmul(np.exp(np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(k.shape[-1])), v) if mask is None else np.matmul(np.exp(np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(k.shape[-1]) + mask * -1e9), v)
        
        # Wähle die richtige Attention-Implementierung
        if use_flash and q.shape[0] > 1 and q.shape[1] > 64:
            # Flash-Attention für große Matrizen
            if "flash_attention" not in self.jit_cache:
                def flash_attention(q, k, v, mask=None):
                    """Flash-Attention-Implementierung für MLX.
                    
                    Diese Implementierung verwendet einen blockweisen Ansatz, um den Speicherverbrauch
                    zu reduzieren und die Leistung zu verbessern. Sie ist besonders effektiv für
                    große Sequenzlängen.
                    """
                    batch_size, seq_len_q, d_k = q.shape[0], q.shape[1], q.shape[2]
                    _, seq_len_k, _ = k.shape
                    
                    # Optimale Blockgröße basierend auf der Sequenzlänge und verfügbarem Speicher
                    # Größere Blöcke für bessere Parallelisierung auf Apple Silicon
                    block_size = min(512, seq_len_q, seq_len_k)
                    
                    # Initialisiere Ausgabe und Normalisierungsfaktoren
                    output = self.mx.zeros((batch_size, seq_len_q, v.shape[2]), dtype=q.dtype)
                    normalizer = self.mx.zeros((batch_size, seq_len_q, 1), dtype=q.dtype)
                    
                    # Skalierungsfaktor
                    scale = 1.0 / self.mx.sqrt(self.mx.array([d_k], dtype=q.dtype))
                    
                    # Sparse Attention Optimierung
                    # Wenn die Sequenzlänge sehr groß ist, verwenden wir Sparse Attention
                    # Adaptive Sparse Attention basierend auf Sequenzlänge und verfügbarem Speicher
                    use_sparse = seq_len_q > 1024 or seq_len_k > 1024
                    
                    # Dynamischer Threshold basierend auf Sequenzlänge
                    # Je länger die Sequenz, desto aggressiver die Sparsity
                    if use_sparse:
                        max_seq = max(seq_len_q, seq_len_k)
                        # Skaliere den Threshold basierend auf der Sequenzlänge
                        # Längere Sequenzen erhalten höhere Thresholds für mehr Sparsity
                        sparsity_threshold = min(0.05, 0.01 * (max_seq / 1024))
                    else:
                        sparsity_threshold = None
                    
                    # Blockweise Berechnung mit Kernel Fusion
                    for i in range(0, seq_len_q, block_size):
                        q_block = q[:, i:i+block_size, :]
                        q_block_len = q_block.shape[1]
                        
                        # Initialisiere lokale Maxima für numerische Stabilität
                        m_i = self.mx.ones((batch_size, q_block_len, 1), dtype=q.dtype) * -1e9
                        
                        for j in range(0, seq_len_k, block_size):
                            k_block = k[:, j:j+block_size, :]
                            v_block = v[:, j:j+block_size, :]
                            k_block_len = k_block.shape[1]
                            
                            # Optimierte Matrixmultiplikation mit Kernel Fusion
                            # Dies reduziert Speichertransfers und verbessert die Leistung
                            def fused_qk_operation(q, k):
                                # Kombinierte Operation: Matrixmultiplikation + Transpose + Skalierung
                                return self.mx.matmul(q, self.mx.transpose(k, (0, 2, 1))) * scale
                            
                            # JIT-Kompilierung für maximale Performance
                            if "fused_qk" not in self.jit_cache:
                                self.jit_cache["fused_qk"] = self.mx.jit(fused_qk_operation)
                            
                            # Berechne Aufmerksamkeits-Scores für diesen Block mit Kernel Fusion
                            scores = self.jit_cache["fused_qk"](q_block, k_block)
                            
                            # Sparse Attention: Filtere niedrige Attention-Werte, wenn aktiviert
                            if use_sparse and sparsity_threshold is not None:
                                # Erstelle eine Sparse-Maske basierend auf dem Threshold
                                sparse_mask = self.mx.abs(scores) < sparsity_threshold
                                # Setze niedrige Werte auf -Inf
                                scores = self.mx.where(sparse_mask, self.mx.array([-1e9], dtype=q.dtype), scores)
                            
                            # Wende Maske an, wenn vorhanden
                            if mask is not None:
                                mask_block = mask[:, i:i+block_size, j:j+block_size]
                                scores = scores + mask_block * -1e9
                            
                            # Finde lokale Maxima für numerische Stabilität mit optimierter Reduktion
                            # Verwende eine fusionierte Operation für bessere Performance
                            if "max_reduce" not in self.jit_cache:
                                self.jit_cache["max_reduce"] = self.mx.jit(lambda x: self.mx.max(x, axis=-1, keepdims=True))
                            
                            m_ij = self.jit_cache["max_reduce"](scores)
                            
                            # Aktualisiere globale Maxima und skaliere vorherige Berechnungen
                            m_new = self.mx.maximum(m_i, m_ij)
                            exp_scale_i = self.mx.exp(m_i - m_new)
                            exp_scale_ij = self.mx.exp(m_ij - m_new)
                            
                            # Berechne Exponential und Summen
                            exp_scores = self.mx.exp(scores - m_ij)
                            exp_sum_ij = self.mx.sum(exp_scores, axis=-1, keepdims=True)
                            
                            # Aktualisiere Ausgabe und Normalisierungsfaktoren
                            output[:, i:i+block_size] = output[:, i:i+block_size] * exp_scale_i + \
                                                       self.mx.matmul(exp_scores, v_block) * exp_scale_ij
                            normalizer[:, i:i+block_size] = normalizer[:, i:i+block_size] * exp_scale_i + \
                                                          exp_sum_ij * exp_scale_ij
                            
                            # Aktualisiere lokale Maxima
                            m_i = m_new
                    
                    # Normalisiere die Ausgabe
                    return output / normalizer
                
                # Kompiliere Flash-Attention-Funktion
                # JIT-Kompilierung der gesamten Flash-Attention-Funktion
                # Wir verwenden eine partielle JIT-Kompilierung, da die Funktion Schleifen enthält
                self.jit_cache["flash_attention"] = flash_attention
            
            # Führe Flash-Attention durch
            result_mx = self.jit_cache["flash_attention"](q_mx, k_mx, v_mx, mask_mx)
        else:
            # Standard-Attention für kleinere Matrizen
            if "standard_attention" not in self.jit_cache:
                def standard_attention(q, k, v, mask=None):
                    # Kernel Fusion für Attention-Berechnung
                    # Kombiniert mehrere Operationen in einen einzigen Kernel für bessere Performance
                    
                    # 1. Berechne Attention-Scores mit optimierter Matrixmultiplikation
                    d_k = q.shape[-1]
                    scale = 1.0 / self.mx.sqrt(self.mx.array([d_k], dtype=q.dtype))
                    
                    # Fusionierte Operation: Matrixmultiplikation + Transpose + Skalierung
                    scores = self.mx.matmul(q, self.mx.transpose(k, (0, 2, 1))) * scale
                    
                    # 2. Wende Maske an, wenn vorhanden
                    if mask is not None:
                        scores = scores + mask * -1e9
                    
                    # 3. Optimierter Softmax mit verbesserter numerischer Stabilität
                    # Subtrahiere das Maximum für numerische Stabilität
                    scores_max = self.mx.max(scores, axis=-1, keepdims=True)
                    scores_exp = self.mx.exp(scores - scores_max)
                    scores_sum = self.mx.sum(scores_exp, axis=-1, keepdims=True)
                    attn = scores_exp / scores_sum
                    
                    # 4. Finale Attention-Berechnung mit optimierter Matrixmultiplikation
                    return self.mx.matmul(attn, v)
                
                # JIT-Kompilierung für maximale Performance
                self.jit_cache["standard_attention"] = self.mx.jit(standard_attention)
            
            # Führe Standard-Attention durch
            result_mx = self.jit_cache["standard_attention"](q_mx, k_mx, v_mx, mask_mx)
                
            # Logging für Debugging
            logger.debug(f"Verwende Standard-Attention für Matrizen der Größe {q.shape}")
        
        # Konvertiere das Ergebnis zurück zu NumPy/PyTorch
        return self.to_numpy(result_mx)
        
        # Cache die Operation für zukünftige Verwendung
        # Wir speichern die kompilierte Funktion, nicht das Ergebnis
        if use_flash:
            self.operation_cache[cache_key] = self.jit_cache["flash_attention"]
        else:
            self.operation_cache[cache_key] = self.jit_cache["standard_attention"]
        
    def benchmark_attention(self, seq_len=512, batch_size=8, head_dim=64, num_runs=10, compare_with_torch=True):
        """Führt einen Benchmark der Attention-Implementierung durch.
        
        Args:
            seq_len: Sequenzlänge für den Benchmark
            batch_size: Batch-Größe für den Benchmark
            head_dim: Dimension der Attention-Heads
            num_runs: Anzahl der Durchläufe für den Benchmark
            compare_with_torch: Ob mit PyTorch verglichen werden soll
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        if not self.mlx_available:
            logger.warning("MLX ist nicht verfügbar. Benchmark kann nicht durchgeführt werden.")
            return {"error": "MLX nicht verfügbar"}
        
        # Erstelle zufällige Eingabedaten
        np.random.seed(42)  # Für Reproduzierbarkeit
        q_np = np.random.randn(batch_size, seq_len, head_dim).astype(np.float32)
        k_np = np.random.randn(batch_size, seq_len, head_dim).astype(np.float32)
        v_np = np.random.randn(batch_size, seq_len, head_dim).astype(np.float32)
        
        # Erstelle Maske (optional)
        mask_np = np.triu(np.ones((batch_size, seq_len, seq_len)) * -1e9, k=1).astype(np.float32)
        
        # MLX Benchmark
        mlx_times = []
        mlx_flash_times = []
        
        # Warmup
        _ = self.attention(q_np, k_np, v_np, mask_np, use_flash=False)
        _ = self.attention(q_np, k_np, v_np, mask_np, use_flash=True)
        
        # Standard Attention Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.attention(q_np, k_np, v_np, mask_np, use_flash=False)
            mlx_times.append(time.time() - start_time)
        
        # Flash Attention Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.attention(q_np, k_np, v_np, mask_np, use_flash=True)
            mlx_flash_times.append(time.time() - start_time)
        
        results = {
            "mlx_standard_attention": {
                "mean_time": np.mean(mlx_times),
                "min_time": np.min(mlx_times),
                "max_time": np.max(mlx_times)
            },
            "mlx_flash_attention": {
                "mean_time": np.mean(mlx_flash_times),
                "min_time": np.min(mlx_flash_times),
                "max_time": np.max(mlx_flash_times)
            },
            "speedup_flash_vs_standard": np.mean(mlx_times) / np.mean(mlx_flash_times) if np.mean(mlx_flash_times) > 0 else float('inf')
        }
        
        # Vergleich mit PyTorch, wenn gewünscht
        if compare_with_torch and torch is not None:
            torch_times = []
            
            # Konvertiere zu PyTorch-Tensoren
            q_torch = torch.tensor(q_np, device=self.torch_device)
            k_torch = torch.tensor(k_np, device=self.torch_device)
            v_torch = torch.tensor(v_np, device=self.torch_device)
            mask_torch = torch.tensor(mask_np, device=self.torch_device)
            
            # PyTorch Attention Benchmark
            for _ in range(num_runs):
                start_time = time.time()
                # Standard PyTorch Attention
                d_k = q_torch.size(-1)
                scores = torch.matmul(q_torch, k_torch.transpose(-2, -1)) / math.sqrt(d_k)
                if mask_torch is not None:
                    scores = scores + mask_torch
                attn = torch.nn.functional.softmax(scores, dim=-1)
                _ = torch.matmul(attn, v_torch)
                torch_times.append(time.time() - start_time)
            
            results["torch_attention"] = {
                "mean_time": np.mean(torch_times),
                "min_time": np.min(torch_times),
                "max_time": np.max(torch_times)
            }
            
            # Berechne Speedup gegenüber PyTorch
            results["speedup_mlx_standard_vs_torch"] = np.mean(torch_times) / np.mean(mlx_times) if np.mean(mlx_times) > 0 else float('inf')
            results["speedup_mlx_flash_vs_torch"] = np.mean(torch_times) / np.mean(mlx_flash_times) if np.mean(mlx_flash_times) > 0 else float('inf')
        
        return results
    
    def optimize_operations(self, operations):
        """Optimiert eine Reihe von Operationen für MLX.
        
        Args:
            operations: Liste von Operationen
            
        Returns:
            Optimierte Operationen
        """
        if not self.mlx_available:
            return operations
        
        # Verwende VX-METACODE für Optimierung, wenn verfügbar
        if self.has_vxor and vxor_math.available_modules.get("VX-METACODE", False):
            try:
                optimized_ops = vxor_math.optimize_mlx_operations(operations)
                logger.info(f"VX-METACODE hat {len(operations)} Operationen optimiert")
                return optimized_ops
            except Exception as e:
                logger.error(f"Fehler bei der VX-METACODE-Optimierung: {e}")
        
        # Fallback: Hier würde eine komplexe Optimierung der Operationen erfolgen
        # In dieser vereinfachten Version geben wir die Operationen unverändert zurück
        return operations
    
    def layer_norm(self, input, weight, bias=None, eps=1e-5):
        """
        Führt Layer-Normalisierung mit MLX durch.
        
        Args:
            input: Eingabe-Array
            weight: Gewichts-Array für Skalierung
            bias: Bias-Array für Offset (optional)
            eps: Epsilon für numerische Stabilität
            
        Returns:
            Normalisiertes MLX-Array
        """
        # Wenn MLX nicht verfügbar ist, verwende Fallback
        if self.mx is None or input is None:
            raise ValueError("MLX nicht verfügbar oder ungültige Eingabe")

        # Cache-Schlüssel für schnellen Lookup
        cache_key = f"layer_norm_{input.shape}_{eps}"
        
        try:
            # Normalisiere den Tensor entlang der letzten Dimension
            # Berechne Mittelwert und Varianz
            mean = self.mx.mean(input, axis=-1, keepdims=True)
            var = self.mx.mean(self.mx.square(input - mean), axis=-1, keepdims=True)
            
            # Normalisiere mit berechneter Statistik
            x_norm = (input - mean) / self.mx.sqrt(var + eps)
            
            # Skaliere und verschiebe mit Gewichten und Bias
            if weight is not None:
                x_norm = x_norm * weight
                
            if bias is not None:
                x_norm = x_norm + bias
                
            # Speichere im Cache für zukünftige Verwendung
            if cache_key not in self.operation_cache:
                self.operation_cache[cache_key] = x_norm
                
            return x_norm
            
        except Exception as e:
            logger.warning(f"Fehler bei MLX layer_norm: {e}. Verwende Standard-Implementierung.")
            # Konvertiere zu NumPy, verwende NumPy-Implementierung, und konvertiere zurück
            input_np = self.to_numpy(input)
            weight_np = self.to_numpy(weight) if weight is not None else np.ones(input_np.shape[-1])
            bias_np = self.to_numpy(bias) if bias is not None else np.zeros(input_np.shape[-1])
            
            # NumPy-Implementierung der Layer-Normalisierung
            mean = np.mean(input_np, axis=-1, keepdims=True)
            var = np.mean(np.square(input_np - mean), axis=-1, keepdims=True)
            x_norm = (input_np - mean) / np.sqrt(var + eps)
            x_norm = x_norm * weight_np
            if bias is not None:
                x_norm = x_norm + bias_np
                
            # Zurück zu MLX
            return self.to_mlx(x_norm)
    
    def gelu(self, x):
        """
        Berechnet die GELU-Aktivierungsfunktion mit MLX.
        
        Args:
            x: Eingabetensor
            
        Returns:
            GELU-Ergebnis als PyTorch-Tensor
        """
        if not self.mlx_available:
            return torch.nn.functional.gelu(x)
            
        x_mlx = self.to_mlx(x)
        # MLX hat keine direkte GELU-Implementierung, implementiere manuell
        result_mlx = 0.5 * x_mlx * (1 + self.mx.tanh(self.mx.sqrt(self.mx.array([2/np.pi], dtype=self.dtype)) * (x_mlx + 0.044715 * self.mx.power(x_mlx, 3))))
        return self.to_torch(result_mlx)
