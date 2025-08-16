#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybride SVD-Implementierung für T-Mathematics Engine

Diese Datei bietet eine adaptive SVD-Implementierung, die automatisch
die schnellste Methode basierend auf der Matrixgröße und dem k-Wert wählt.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import math
from typing import Dict, Any, Optional, Union, Tuple
from functools import lru_cache

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.hybrid_svd")

# MLX-Verfügbarkeit prüfen
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    logger.warning("MLX konnte nicht importiert werden. Apple Silicon Optimierungen sind nicht verfügbar.")

class HybridSVD:
    """
    Hybride SVD-Implementierung, die automatisch die optimale Methode auswählt
    
    Diese Klasse kombiniert die Stärken der ursprünglichen MLX-Implementierung
    und der optimierten SVD-Implementierung, um die beste Leistung zu erzielen.
    """
    
    def __init__(self, original_backend, optimized_backend):
        """
        Initialisiert die hybride SVD-Implementierung
        
        Args:
            original_backend: Ursprüngliches MLXBackend
            optimized_backend: Optimiertes MLXBackend mit SVD-Optimizer
        """
        self.original_backend = original_backend
        self.optimized_backend = optimized_backend
        
        # Leistungsstatistik initialisieren
        self.stats = {
            "total_calls": 0,
            "original_calls": 0,
            "optimized_calls": 0,
            "total_time": 0.0
        }
        
        # Benchmarkergebnisse für Entscheidungslogik
        # Diese basieren auf unserem vorherigen Benchmark
        self.performance_map = {
            "small": {  # < 64x64
                "full": "original",   # Vollständige SVD für kleine Matrizen: Original
                "partial": "optimized"  # Partielle SVD für kleine Matrizen: Optimiert
            },
            "medium": {  # 64x64 - 256x256
                "full": "original",   # Vollständige SVD für mittlere Matrizen: Original
                "partial": "optimized"  # Partielle SVD für mittlere Matrizen: Optimiert
            },
            "large": {  # > 256x256
                "full": "hybrid",     # Vollständige SVD für große Matrizen: Hybridansatz
                "partial": "original"  # Partielle SVD für große Matrizen: Original
            }
        }
        
        logger.info("Hybride SVD-Implementierung initialisiert")
    
    def _get_matrix_size_category(self, shape):
        """
        Bestimmt die Größenkategorie einer Matrix
        
        Args:
            shape: Form der Matrix
            
        Returns:
            Kategorie ("small", "medium", "large")
        """
        if len(shape) < 2:
            return "small"
        
        # Berechne die Anzahl der Elemente
        rows, cols = shape[-2], shape[-1]
        num_elements = rows * cols
        
        if num_elements < 64 * 64:
            return "small"
        elif num_elements < 256 * 256:
            return "medium"
        else:
            return "large"
    
    def _get_svd_type(self, k, shape):
        """
        Bestimmt den SVD-Typ (vollständig oder partiell)
        
        Args:
            k: Anzahl der zu berechnenden Singulärwerte
            shape: Form der Matrix
            
        Returns:
            SVD-Typ ("full" oder "partial")
        """
        if k is None:
            return "full"
        
        # Wenn k nahe an min(shape) ist, betrachten wir es als vollständig
        min_dim = min(shape[-2], shape[-1])
        if k > 0.8 * min_dim:
            return "full"
        
        return "partial"
    
    def _select_optimal_backend(self, a_shape, k=None):
        """
        Wählt das optimale Backend basierend auf Matrix-Form und k-Wert
        
        Args:
            a_shape: Form der Matrix
            k: Anzahl der zu berechnenden Singulärwerte
            
        Returns:
            Backend-Selector ("original", "optimized" oder "hybrid")
        """
        size_category = self._get_matrix_size_category(a_shape)
        svd_type = self._get_svd_type(k, a_shape)
        
        # Hole die optimale Strategie aus der Performance-Map
        strategy = self.performance_map[size_category][svd_type]
        
        return strategy
    
    def svd(self, a, k=None):
        """
        Führt eine adaptive SVD durch und wählt die optimale Methode
        
        Args:
            a: Matrix für SVD
            k: Anzahl der zu berechnenden Singulärwerte (None für alle)
            
        Returns:
            Tuple (U, S, V) der SVD-Komponenten
        """
        self.stats["total_calls"] += 1
        start_time = time.time()
        
        # Bestimme die optimale Strategie
        strategy = self._select_optimal_backend(a.shape, k)
        
        try:
            # Ursprüngliche Implementierung
            if strategy == "original":
                self.stats["original_calls"] += 1
                result = self.original_backend.svd(a, k)
            
            # Optimierte Implementierung
            elif strategy == "optimized":
                self.stats["optimized_calls"] += 1
                result = self.optimized_backend.svd(a, k)
            
            # Hybridansatz - Kombiniert beide Methoden
            elif strategy == "hybrid":
                self.stats["optimized_calls"] += 0.5
                self.stats["original_calls"] += 0.5
                
                # In unserem Fall verwenden wir die optimierte Implementierung für große Matrizen
                result = self.optimized_backend.svd(a, k)
            
            # Fallback zur ursprünglichen Implementierung
            else:
                logger.warning(f"Unbekannte Strategie: {strategy}, verwende ursprüngliche Implementierung")
                self.stats["original_calls"] += 1
                result = self.original_backend.svd(a, k)
        
        except Exception as e:
            logger.error(f"Fehler bei der hybriden SVD: {e}, verwende Fallback")
            # Fallback zur ursprünglichen Implementierung für Robustheit
            result = self.original_backend.svd(a, k)
        
        # Aktualisiere Statistik
        end_time = time.time()
        self.stats["total_time"] += (end_time - start_time)
        
        return result
    
    def get_stats(self):
        """
        Gibt Statistiken zur hybriden SVD-Implementierung zurück
        
        Returns:
            Dictionary mit Statistiken
        """
        if self.stats["total_calls"] > 0:
            stats = dict(self.stats)
            stats["avg_time"] = self.stats["total_time"] / self.stats["total_calls"]
            stats["original_ratio"] = self.stats["original_calls"] / self.stats["total_calls"]
            stats["optimized_ratio"] = self.stats["optimized_calls"] / self.stats["total_calls"]
            return stats
        
        return self.stats


def install_hybrid_svd(original_backend, optimized_backend):
    """
    Installiert die hybride SVD-Implementierung in einem Backend
    
    Args:
        original_backend: Ursprüngliches MLXBackend
        optimized_backend: Optimiertes MLXBackend mit SVD-Optimizer
        
    Returns:
        Backend mit hybrider SVD-Implementierung
    """
    # Erstelle hybride SVD-Instanz
    hybrid_svd = HybridSVD(original_backend, optimized_backend)
    
    # Ersetze die SVD-Methode im optimierten Backend
    optimized_backend.svd = hybrid_svd.svd
    
    # Füge Statistik-Methode hinzu
    optimized_backend.get_svd_stats = hybrid_svd.get_stats
    
    logger.info("Hybride SVD-Implementierung erfolgreich installiert")
    return optimized_backend
