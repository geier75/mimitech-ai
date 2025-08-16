#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimiertes MLX Support Module für T-Mathematics Engine

Diese Datei enthält die optimierte Implementierung des MLX-Backends
mit korrekter JIT-Kompilierung und effizienter Speicherverwaltung.

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

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.mlx_support")

# MLX-Verfügbarkeit prüfen
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
    logger.info("MLX-Bibliothek erfolgreich importiert")
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
        logger.info("Apple Silicon M-Series erkannt, optimiere für Neural Engine")
except:
    logger.warning("Konnte Prozessortyp nicht ermitteln. Apple Silicon-Optimierungen möglicherweise nicht verfügbar.")

# VXOR-Integration prüfen
has_vxor = False
try:
    from vxor_math_integration import VXORMathIntegration
    has_vxor = True
    logger.info("VXOR-Integration für mathematische Optimierungen verfügbar")
except ImportError:
    has_vxor = False
    logger.info("VXOR-Integration nicht verfügbar, verwende Standard-Optimierungen")


class MLXBackend:
    """
    Optimiertes Backend für MLX-Operationen auf Apple Silicon.
    
    Diese Klasse bietet eine optimierte Implementierung für mathematische
    Operationen mit MLX, einschließlich JIT-Kompilierung und effizientem
    Speichermanagement.
    """
    
    def __init__(self, precision="float16"):
        """
        Initialisiert das MLX-Backend mit optimierter JIT-Unterstützung.
        
        Args:
            precision: Präzisionstyp für Berechnungen ("float16", "float32", "bfloat16")
        """
        self.precision = precision
        self.jit_cache = {}
        self.jit_cache_hits = 0
        self.jit_cache_misses = 0
        self.operation_cache = {}
        self.has_vxor = has_vxor
        
        # Statistiken für Leistungsanalyse
        self.time_in_transfers = 0.0
        self.transfer_count = 0
        self.computation_time = 0.0
        self.computation_count = 0
        
        # Optimale Geräte-Konfiguration für PyTorch
        if torch.backends.mps.is_available():
            self.torch_device = torch.device("mps")
        else:
            self.torch_device = torch.device("cpu")
        
        # Prüfe, ob MLX verfügbar ist
        self.mlx_available = HAS_MLX and mx is not None
        if not self.mlx_available:
            logger.warning("MLX nicht verfügbar. Optimierungen deaktiviert.")
            return
            
        # MLX-Konfiguration
        self.mx = mx
        
        # Prüfe MLX-Version und verfügbare Funktionen
        self.has_jit = hasattr(mx, 'jit')
        
        logger.info(f"MLX-Backend initialisiert mit Präzision {precision} (JIT: {self.has_jit})")
        
        # MLX-Optimierungen aktivieren
        # Setze MLX-Datentyp basierend auf der Präzision
        self.dtype = self._get_mlx_dtype(precision)
        
        # Setze Standardgerät für MLX
        try:
            mx.set_default_device(mx.gpu if IS_APPLE_SILICON else mx.cpu)
            logger.info("MLX-Gerät auf 'gpu' gesetzt für Apple Silicon")
        except Exception as e:
            logger.warning(f"MLX-Gerätekonfiguration fehlgeschlagen: {e}")
        
        # Initialisiere JIT-Funktionen, wenn verfügbar
        if self.has_jit:
            try:
                # Initialisiere JIT-kompilierte Standardfunktionen
                self._init_jit_functions()
                logger.info("MLX-JIT-Funktionen erfolgreich initialisiert")
            except Exception as e:
                logger.warning(f"MLX-JIT-Initialisierung fehlgeschlagen: {e}")
                self.has_jit = False
    
    def _get_mlx_dtype(self, precision):
        """Gibt den entsprechenden MLX-Datentyp für die angegebene Präzision zurück."""
        if not self.mlx_available:
            return None
            
        precision_map = {
            "float16": mx.float16,
            "float32": mx.float32,
            "bfloat16": mx.bfloat16 if hasattr(mx, 'bfloat16') else mx.float16
        }
        
        return precision_map.get(precision, mx.float32)
    
    def _init_jit_functions(self):
        """Initialisiert JIT-kompilierte Funktionen für häufig verwendete Operationen."""
        if not self.has_jit:
            return
        
        # Grundlegende mathematische Operationen
        self.jit_matmul = mx.jit(lambda a, b: mx.matmul(a, b))
        self.jit_add = mx.jit(lambda a, b: mx.add(a, b))
        self.jit_sub = mx.jit(lambda a, b: mx.subtract(a, b))
        self.jit_mul = mx.jit(lambda a, b: mx.multiply(a, b))
        self.jit_div = mx.jit(lambda a, b: mx.divide(a, b))
        
        # Elementweise Operationen
        self.jit_relu = mx.jit(lambda x: mx.maximum(x, 0))
        self.jit_sigmoid = mx.jit(lambda x: 1.0 / (1.0 + mx.exp(-x)))
        self.jit_tanh = mx.jit(lambda x: mx.tanh(x))
        
        # GELU Aktivierungsfunktion
        self.jit_gelu = mx.jit(lambda x: 0.5 * x * (1 + mx.tanh(mx.sqrt(2 / 3.14159) * 
                                                              (x + 0.044715 * mx.power(x, 3)))))
        
        # Layer Normalisierung
        self.jit_layer_norm = mx.jit(lambda x, weight, bias, eps=1e-5: 
                                    self._layer_norm_forward(x, weight, bias, eps))
        
        # Softmax Funktion                            
        self.jit_softmax = mx.jit(lambda x, axis=-1: mx.softmax(x, axis=axis))
        
        logger.info("JIT-kompilierte mathematische Funktionen initialisiert")
        
        # Fortgeschrittene Operationen
        try:
            # SVD mit Fallback und Fehlerbehandlung
            self.jit_svd = mx.jit(lambda x, full_matrices=False: 
                                 mx.linalg.svd(x, full_matrices=full_matrices))
            logger.info("JIT-kompilierte SVD-Funktion initialisiert")
        except Exception as e:
            logger.warning(f"JIT-SVD-Initialisierung fehlgeschlagen: {e}")
            self.jit_svd = None
    
    def _layer_norm_forward(self, x, weight, bias, eps=1e-5):
        """Layer-Normalisierung-Forward-Implementierung für JIT-Kompilierung."""
        input_dtype = x.dtype
        x = x.astype(mx.float32)  # Erhöhte Präzision für Berechnung
        
        # Berechne Mittelwert und Varianz
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
        
        # Normalisierung
        x_norm = (x - mean) / mx.sqrt(var + eps)
        
        # Skalierung und Verschiebung
        if weight is not None and bias is not None:
            x_norm = x_norm * weight + bias
            
        return x_norm.astype(input_dtype)  # Zurück zur Eingabepräzision
    
    def to_mlx(self, tensor):
        """
        Optimierte Konvertierung eines PyTorch-Tensors zu einem MLX-Array.
        
        Diese Implementierung reduziert unnötige Kopieraktionen und
        Speichertransfers zwischen Geräten.
        
        Args:
            tensor: PyTorch-Tensor, NumPy-Array oder ähnliches
            
        Returns:
            MLX-Array
        """
        if not self.mlx_available:
            return tensor
            
        # Messung der Transferzeit beginnen
        start_time = time.time()
        
        try:
            # PyTorch-Tensor
            if isinstance(tensor, torch.Tensor):
                # Extraktion als NumPy-Array - optimiert je nach Gerät
                if tensor.device.type == 'mps':
                    # Direkte Konvertierung von MPS zu MLX (wenn möglich)
                    try:
                        # Versuche direkten Transfer ohne CPU-Umweg
                        numpy_tensor = tensor.detach().numpy()
                    except Exception:
                        # Fallback: Über CPU
                        numpy_tensor = tensor.detach().cpu().numpy()
                else:
                    # CPU oder CUDA Tensor
                    numpy_tensor = tensor.detach().cpu().numpy()
                    
                # Konvertierung zu MLX mit korrektem Datentyp
                mlx_array = mx.array(numpy_tensor, dtype=self.dtype)
                
            # NumPy-Array direkt konvertieren
            elif isinstance(tensor, np.ndarray):
                mlx_array = mx.array(tensor, dtype=self.dtype)
                
            # Liste oder Tupel
            elif isinstance(tensor, (list, tuple)):
                mlx_array = mx.array(np.array(tensor), dtype=self.dtype)
                
            # Andere Typen - versuche direkte Konvertierung
            else:
                try:
                    mlx_array = mx.array(tensor, dtype=self.dtype)
                except Exception as e:
                    raise TypeError(f"Konnte {type(tensor)} nicht zu MLX-Array konvertieren: {e}")
            
            # Transferzeit messen und Statistik aktualisieren
            self.time_in_transfers += time.time() - start_time
            self.transfer_count += 1
            
            return mlx_array
            
        except Exception as e:
            logger.error(f"Fehler bei Konvertierung zu MLX: {e}")
            raise
    
    def to_numpy(self, tensor):
        """
        Konvertiert einen Tensor zu NumPy, unabhängig vom Ursprungstyp.
        
        Args:
            tensor: MLX-Array, PyTorch-Tensor oder NumPy-Array
            
        Returns:
            NumPy-Array
        """
        if tensor is None:
            return None
            
        # Messung der Transferzeit beginnen
        start_time = time.time()
        
        try:
            # MLX-Array
            if self.mlx_available and isinstance(tensor, mx.array):
                numpy_tensor = tensor.tolist()
                
            # PyTorch-Tensor
            elif isinstance(tensor, torch.Tensor):
                numpy_tensor = tensor.detach().cpu().numpy()
                
            # NumPy-Array
            elif isinstance(tensor, np.ndarray):
                numpy_tensor = tensor
                
            # Andere Typen
            else:
                try:
                    numpy_tensor = np.array(tensor)
                except:
                    raise TypeError(f"Konnte {type(tensor)} nicht zu NumPy-Array konvertieren")
            
            # Transferzeit messen und Statistik aktualisieren
            self.time_in_transfers += time.time() - start_time
            self.transfer_count += 1
            
            return numpy_tensor
            
        except Exception as e:
            logger.error(f"Fehler bei Konvertierung zu NumPy: {e}")
            raise
    
    def to_torch(self, array, device=None):
        """
        Optimierte Konvertierung eines MLX-Arrays zu einem PyTorch-Tensor.
        
        Args:
            array: MLX-Array oder anderer Tensor
            device: Zielgerät für PyTorch-Tensor (None für auto)
            
        Returns:
            PyTorch-Tensor
        """
        if array is None:
            return None
            
        # Zielgerät bestimmen
        device = device or self.torch_device
        
        # Messung der Transferzeit beginnen
        start_time = time.time()
        
        try:
            # MLX-Array
            if self.mlx_available and isinstance(array, mx.array):
                # Konvertiere zu NumPy
                numpy_array = array.tolist()
                
                # Dann zu PyTorch mit spezifiziertem Gerät
                torch_tensor = torch.tensor(numpy_array, device=device)
                
            # NumPy-Array
            elif isinstance(array, np.ndarray):
                torch_tensor = torch.tensor(array, device=device)
                
            # PyTorch-Tensor
            elif isinstance(array, torch.Tensor):
                # Wenn bereits PyTorch, stelle sicher, dass es auf dem richtigen Gerät ist
                if array.device != device:
                    torch_tensor = array.to(device)
                else:
                    torch_tensor = array
                    
            # Andere Typen
            else:
                try:
                    # Versuche direkte Konvertierung
                    torch_tensor = torch.tensor(array, device=device)
                except:
                    raise TypeError(f"Konnte {type(array)} nicht zu PyTorch-Tensor konvertieren")
            
            # Transferzeit messen und Statistik aktualisieren
            self.time_in_transfers += time.time() - start_time
            self.transfer_count += 1
            
            return torch_tensor
            
        except Exception as e:
            logger.error(f"Fehler bei Konvertierung zu PyTorch: {e}")
            raise
    
    def _get_matmul_key(self, a_shape, b_shape, batch_size=None):
        """
        Generiert einen Schlüssel für das Matrixmultiplikations-Cache.
        
        Args:
            a_shape: Form der ersten Matrix
            b_shape: Form der zweiten Matrix
            batch_size: Optionale Batch-Größe
            
        Returns:
            Cache-Schlüssel als String
        """
        return f"matmul_{a_shape}_{b_shape}_{batch_size}"
    
    def matmul(self, a, b, batch_size=None):
        """
        Führt eine optimierte Matrixmultiplikation mit JIT-Kompilierung durch.
        
        Args:
            a: Erste Matrix (MLX-Array)
            b: Zweite Matrix (MLX-Array)
            batch_size: Optionale Batch-Größe für Batch-Matrixmultiplikation
            
        Returns:
            Ergebnis der Matrixmultiplikation (MLX-Array)
        """
        if not self.mlx_available:
            raise RuntimeError("MLX ist nicht verfügbar")
            
        # Eingabevalidierung
        if a is None or b is None:
            raise ValueError("Eingabetensoren dürfen nicht None sein")
        
        # Messung der Berechnungszeit beginnen
        start_time = time.time()
        
        # Batch-Matrixmultiplikation
        if batch_size is not None:
            return self.batch_matmul(a, b, batch_size)
        
        # Schlüssel für Cache generieren
        key = self._get_matmul_key(a.shape, b.shape)
        
        try:
            # JIT-kompilierte Funktion aus Cache verwenden oder erstellen
            if self.has_jit:
                if key not in self.jit_cache:
                    # Neue JIT-Funktion erstellen und cachen
                    self.jit_cache[key] = mx.jit(lambda x, y: mx.matmul(x, y))
                    self.jit_cache_misses += 1
                else:
                    self.jit_cache_hits += 1
                
                # JIT-kompilierte Funktion ausführen
                result = self.jit_cache[key](a, b)
            else:
                # Fallback: direkte Matrixmultiplikation ohne JIT
                result = mx.matmul(a, b)
            
            # Berechnungszeit messen und Statistik aktualisieren
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei MLX-Matrixmultiplikation: {e}")
            
            # Berechnungszeit trotzdem in Statistik aufnehmen
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            
            # Fehler weiterleiten
            raise RuntimeError(f"MLX-Matrixmultiplikation fehlgeschlagen: {e}")
    
    def svd(self, matrix, k=None):
        """
        Führt eine optimierte Singulärwertzerlegung (SVD) mit MLX durch.
        
        Args:
            matrix: Eingangsmatrix (MLX-Array)
            k: Anzahl der zu berechnenden Singulärwerte (None für alle)
            
        Returns:
            Tupel (U, S, V) mit den SVD-Komponenten
        """
        if not self.mlx_available:
            raise RuntimeError("MLX ist nicht verfügbar")
            
        # Eingabevalidierung
        if matrix is None:
            raise ValueError("Eingangsmatrix darf nicht None sein")
        
        # Schlüsselgenerierung für Caching
        key = f"svd_{matrix.shape}_{k}"
        
        # Messung der Berechnungszeit beginnen
        start_time = time.time()
        
        try:
            # JIT-kompilierte SVD verwenden, wenn verfügbar
            if self.has_jit and self.jit_svd is not None:
                # Parameter für SVD vorbereiten
                full_matrices = k is None
                
                # SVD ausführen
                result = self.jit_svd(matrix, full_matrices=full_matrices)
                
                # Wenn k spezifiziert ist, beschränke die Ergebnisse
                if k is not None and k < min(matrix.shape):
                    u, s, v = result
                    u = u[:, :k]
                    s = s[:k]
                    v = v[:k, :]
                    result = (u, s, v)
            else:
                # Fallback: Direkte SVD ohne JIT
                result = mx.linalg.svd(matrix, full_matrices=(k is None))
                
                # Bei Bedarf beschränken
                if k is not None and k < min(matrix.shape):
                    u, s, v = result
                    u = u[:, :k]
                    s = s[:k]
                    v = v[:k, :]
                    result = (u, s, v)
            
            # Berechnungszeit messen und Statistik aktualisieren
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei MLX-SVD: {e}")
            
            # Berechnungszeit trotzdem in Statistik aufnehmen
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            
            # Fehler weiterleiten
            raise RuntimeError(f"MLX-SVD fehlgeschlagen: {e}")
    
    def add(self, a, b):
        """Führt eine optimierte Addition durch."""
        if not self.mlx_available:
            raise RuntimeError("MLX ist nicht verfügbar")
            
        # Messung der Berechnungszeit beginnen
        start_time = time.time()
        
        try:
            # JIT-kompilierte Addition verwenden, wenn verfügbar
            if self.has_jit:
                result = self.jit_add(a, b)
            else:
                result = mx.add(a, b)
            
            # Berechnungszeit messen und Statistik aktualisieren
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei MLX-Addition: {e}")
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            raise RuntimeError(f"MLX-Addition fehlgeschlagen: {e}")
    
    def mul(self, a, b):
        """Führt eine optimierte elementweise Multiplikation durch."""
        if not self.mlx_available:
            raise RuntimeError("MLX ist nicht verfügbar")
            
        # Messung der Berechnungszeit beginnen
        start_time = time.time()
        
        try:
            # JIT-kompilierte Multiplikation verwenden, wenn verfügbar
            if self.has_jit:
                result = self.jit_mul(a, b)
            else:
                result = mx.multiply(a, b)
            
            # Berechnungszeit messen und Statistik aktualisieren
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei MLX-Multiplikation: {e}")
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            raise RuntimeError(f"MLX-Multiplikation fehlgeschlagen: {e}")
    
    def gelu(self, x):
        """Führt eine optimierte GELU-Aktivierungsfunktion durch."""
        if not self.mlx_available:
            raise RuntimeError("MLX ist nicht verfügbar")
            
        # Messung der Berechnungszeit beginnen
        start_time = time.time()
        
        try:
            # JIT-kompilierte GELU verwenden, wenn verfügbar
            if self.has_jit:
                result = self.jit_gelu(x)
            else:
                # Direkte GELU-Implementierung
                result = 0.5 * x * (1 + mx.tanh(mx.sqrt(2 / 3.14159) * 
                                             (x + 0.044715 * mx.power(x, 3))))
            
            # Berechnungszeit messen und Statistik aktualisieren
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei MLX-GELU: {e}")
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            raise RuntimeError(f"MLX-GELU fehlgeschlagen: {e}")
    
    def layer_norm(self, input_tensor, weight, bias, eps=1e-5):
        """Führt eine optimierte Layer-Normalisierung durch."""
        if not self.mlx_available:
            raise RuntimeError("MLX ist nicht verfügbar")
            
        # Messung der Berechnungszeit beginnen
        start_time = time.time()
        
        try:
            # JIT-kompilierte Layer-Norm verwenden, wenn verfügbar
            if self.has_jit:
                result = self.jit_layer_norm(input_tensor, weight, bias, eps)
            else:
                # Direkte Layer-Norm-Implementierung
                result = self._layer_norm_forward(input_tensor, weight, bias, eps)
            
            # Berechnungszeit messen und Statistik aktualisieren
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei MLX-Layer-Norm: {e}")
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            raise RuntimeError(f"MLX-Layer-Norm fehlgeschlagen: {e}")
    
    def sum(self, x, axis=None, keepdims=False):
        """Führt eine optimierte Summenbildung durch."""
        if not self.mlx_available:
            raise RuntimeError("MLX ist nicht verfügbar")
            
        # Messung der Berechnungszeit beginnen
        start_time = time.time()
        
        try:
            # JIT-kompilierte Summe, wenn möglich
            if self.has_jit:
                # Für bestimmte axis-Werte können wir vorkompilierte Funktionen verwenden
                if axis is None and not keepdims:
                    key = "sum_all"
                else:
                    key = f"sum_{axis}_{keepdims}"
                
                # Cache prüfen
                if key not in self.jit_cache:
                    # Neue JIT-Funktion erstellen
                    self.jit_cache[key] = mx.jit(lambda t: mx.sum(t, axis=axis, keepdims=keepdims))
                    self.jit_cache_misses += 1
                else:
                    self.jit_cache_hits += 1
                
                # Ausführen
                result = self.jit_cache[key](x)
            else:
                # Direkte Summenberechnung
                result = mx.sum(x, axis=axis, keepdims=keepdims)
            
            # Berechnungszeit messen und Statistik aktualisieren
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei MLX-Summenbildung: {e}")
            self.computation_time += time.time() - start_time
            self.computation_count += 1
            raise RuntimeError(f"MLX-Summenbildung fehlgeschlagen: {e}")
    
    def get_stats(self):
        """Gibt Statistiken über die MLX-Backend-Nutzung zurück."""
        stats = {
            "mlx_verfügbar": self.mlx_available,
            "jit_aktiviert": self.has_jit,
            "cache": {
                "einträge": len(self.jit_cache),
                "hits": self.jit_cache_hits,
                "misses": self.jit_cache_misses,
                "trefferrate": self.jit_cache_hits / (self.jit_cache_hits + self.jit_cache_misses) if (self.jit_cache_hits + self.jit_cache_misses) > 0 else 0
            },
            "transfers": {
                "zeit_ms": self.time_in_transfers * 1000,
                "anzahl": self.transfer_count,
                "durchschnitt_ms": (self.time_in_transfers * 1000) / self.transfer_count if self.transfer_count > 0 else 0
            },
            "berechnungen": {
                "zeit_ms": self.computation_time * 1000,
                "anzahl": self.computation_count,
                "durchschnitt_ms": (self.computation_time * 1000) / self.computation_count if self.computation_count > 0 else 0
            }
        }
        
        return stats
    
    def clear_cache(self):
        """Leert den JIT-Funktionscache."""
        self.jit_cache.clear()
        self.jit_cache_hits = 0
        self.jit_cache_misses = 0
        logger.info("JIT-Funktionscache geleert")
        
        return True
