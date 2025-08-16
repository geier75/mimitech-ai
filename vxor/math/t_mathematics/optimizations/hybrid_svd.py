#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybride SVD-Implementierung für T-Mathematics Engine

Diese Datei bietet eine adaptive SVD-Implementierung, die automatisch
die schnellste Methode basierend auf der Matrixgröße und dem k-Wert wählt.
Für MISO Ultimate mit Fokus auf optimale Leistung auf Apple Silicon.

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
logger = logging.getLogger("MISO.math.t_mathematics.optimizations.hybrid_svd")

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
    Basierend auf umfangreichen Benchmark-Ergebnissen.
    """
    
    def __init__(self, original_backend, optimized_backend, enable_telemetry=True):
        """
        Initialisiert die hybride SVD-Implementierung
        
        Args:
            original_backend: Ursprüngliches MLXBackend
            optimized_backend: Optimiertes MLXBackend mit SVD-Optimizer
            enable_telemetry: Aktiviert detaillierte Leistungsstatistiken
        """
        self.original_backend = original_backend
        self.optimized_backend = optimized_backend
        self.enable_telemetry = enable_telemetry
        
        # Wichtig: Sichere die ursprünglichen SVD-Implementierungen, um Rekursion zu vermeiden
        self._original_svd_impl = original_backend.svd
        self._optimized_svd_impl = optimized_backend.svd if hasattr(optimized_backend, 'svd') else original_backend.svd
        
        # Leistungsstatistik initialisieren
        self.stats = {
            "total_calls": 0,
            "original_calls": 0,
            "optimized_calls": 0,
            "hybrid_calls": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "decision_time": 0.0
        }
        
        # Strategieentscheidungs-Cache
        self.strategy_cache = {}
        
        # Benchmarkergebnisse für Entscheidungslogik
        # Diese basieren auf unseren vorherigen SVD-Benchmarks
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
        
        # Größenschwellwerte für die Klassifizierung
        self.size_thresholds = {
            "small_max": 64 * 64,     # Max. Elementzahl für kleine Matrizen
            "medium_max": 256 * 256   # Max. Elementzahl für mittlere Matrizen
        }
        
        # Informationen über installierte Optimierungen sammeln
        self.svd_optimizer_available = hasattr(self.optimized_backend, "get_svd_stats")
        
        logger.info("Hybride SVD-Implementierung initialisiert")
        logger.info(f"SVD-Optimizer verfügbar: {self.svd_optimizer_available}")
    
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
        
        if num_elements < self.size_thresholds["small_max"]:
            return "small"
        elif num_elements < self.size_thresholds["medium_max"]:
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
        # Verwende Cache für schnelle Entscheidungen
        cache_key = f"{a_shape}_{k}"
        if cache_key in self.strategy_cache:
            self.stats["cache_hits"] += 1
            return self.strategy_cache[cache_key]
        
        # Bestimme Matrix-Charakteristika
        start_time = time.time() if self.enable_telemetry else 0
        size_category = self._get_matrix_size_category(a_shape)
        svd_type = self._get_svd_type(k, a_shape)
        
        # Hole die optimale Strategie aus der Performance-Map
        strategy = self.performance_map[size_category][svd_type]
        
        # Cache die Strategie für zukünftige Aufrufe
        self.strategy_cache[cache_key] = strategy
        
        if self.enable_telemetry:
            end_time = time.time()
            self.stats["decision_time"] += (end_time - start_time)
        
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
        start_time = time.time() if self.enable_telemetry else 0
        
        # Bestimme die optimale Strategie
        strategy = self._select_optimal_backend(a.shape, k)
        
        try:
            # Ursprüngliche Implementierung
            if strategy == "original":
                self.stats["original_calls"] += 1
                # Direkte Verwendung der gesicherten Original-Implementierung, um Rekursion zu vermeiden
                result = self._original_svd_impl(a, k)
            
            # Optimierte Implementierung
            elif strategy == "optimized":
                if self.svd_optimizer_available:
                    self.stats["optimized_calls"] += 1
                    # Verwende die gesicherte optimierte Implementierung
                    result = self._optimized_svd_impl(a, k)
                else:
                    # Fallback wenn Optimizer nicht verfügbar
                    logger.warning("SVD-Optimizer nicht verfügbar, verwende ursprüngliche Implementation")
                    self.stats["original_calls"] += 1
                    result = self._original_svd_impl(a, k)
            
            # Hybridansatz - Experimenteller Ansatz für große Matrizen
            elif strategy == "hybrid":
                self.stats["hybrid_calls"] += 1
                
                # Für große Matrizen mit vollständiger SVD:
                # 1. Verwende optimierte Implementierung für die grundlegende Berechnung
                # 2. Validiere mit ursprünglicher Implementierung bei Bedarf
                is_large = self._get_matrix_size_category(a.shape) == "large"
                is_full_svd = self._get_svd_type(k, a.shape) == "full"
                
                if is_large and is_full_svd and self.svd_optimizer_available:
                    # Verwende die optimierte Implementierung direkt
                    result = self._optimized_svd_impl(a, k)
                else:
                    # Für andere Hybrid-Fälle, verwende ursprüngliche Implementierung direkt
                    result = self._original_svd_impl(a, k)
            
            # Fallback zur ursprünglichen Implementierung
            else:
                logger.warning(f"Unbekannte Strategie: {strategy}, verwende ursprüngliche Implementierung")
                self.stats["original_calls"] += 1
                result = self._original_svd_impl(a, k)
        
        except Exception as e:
            logger.error(f"Fehler bei der hybriden SVD: {e}, verwende Fallback")
            # Fallback zur ursprünglichen Implementierung für Robustheit - direkt implementiert
            # um Rekursion zu vermeiden
            try:
                # Versuche direkt das Original zu verwenden
                result = self._original_svd_impl(a, k)
            except Exception as e2:
                # Wenn auch das fehlschlägt, verwende einfache NumPy-Implementation
                logger.error(f"Auch Original-SVD fehlgeschlagen: {e2}, verwende NumPy")
                # Konvertiere zu NumPy
                import numpy as np
                if hasattr(a, 'tolist'):
                    a_np = np.array(a.tolist())
                else:
                    a_np = np.array(a)
                
                # Führe NumPy-SVD durch
                if k is None:
                    u, s, v = np.linalg.svd(a_np, full_matrices=False)
                else:
                    # Nimm die ersten k Singulärwerte
                    u, s, v = np.linalg.svd(a_np, full_matrices=False)
                    u, s, v = u[:, :k], s[:k], v[:k, :]
                
                # Gib als Tuple zurück
                if HAS_MLX and mx is not None:
                    result = (mx.array(u), mx.array(s), mx.array(v))
                else:
                    result = (u, s, v)
        
        # Aktualisiere Statistik
        if self.enable_telemetry:
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
            stats["hybrid_ratio"] = self.stats["hybrid_calls"] / self.stats["total_calls"]
            stats["cache_hit_ratio"] = self.stats["cache_hits"] / self.stats["total_calls"] if self.stats["total_calls"] > 0 else 0
            
            # Füge SVD-Optimizer-Statistiken hinzu, wenn verfügbar
            if self.svd_optimizer_available and hasattr(self.optimized_backend, "get_svd_stats"):
                optimizer_stats = self.optimized_backend.get_svd_stats()
                stats["optimizer"] = optimizer_stats
            
            return stats
        
        return self.stats
    
    def update_performance_map(self, new_map):
        """
        Aktualisiert die Performance-Map für die Strategieauswahl
        
        Args:
            new_map: Neue Performance-Map
            
        Returns:
            Aktualisierte Performance-Map
        """
        # Validiere die neue Map
        required_categories = ["small", "medium", "large"]
        required_types = ["full", "partial"]
        
        for category in required_categories:
            if category not in new_map:
                logger.warning(f"Fehlende Kategorie in Performance-Map: {category}")
                continue
                
            for svd_type in required_types:
                if svd_type not in new_map[category]:
                    logger.warning(f"Fehlender SVD-Typ in Kategorie {category}: {svd_type}")
                    continue
                    
                strategy = new_map[category][svd_type]
                if strategy not in ["original", "optimized", "hybrid"]:
                    logger.warning(f"Ungültige Strategie für {category}/{svd_type}: {strategy}")
                    continue
        
        # Aktualisiere die Map
        self.performance_map.update(new_map)
        
        # Leere den Cache, um neue Entscheidungen zu erzwingen
        self.strategy_cache = {}
        
        logger.info("Performance-Map aktualisiert")
        return self.performance_map


def install_hybrid_svd(original_backend, optimized_backend, enable_telemetry=True):
    """
    Installiert die hybride SVD-Implementierung in einem Backend
    
    Args:
        original_backend: Ursprüngliches MLXBackend
        optimized_backend: Optimiertes MLXBackend mit SVD-Optimizer
        enable_telemetry: Aktiviert detaillierte Leistungsstatistiken
        
    Returns:
        Backend mit hybrider SVD-Implementierung
    """
    # Erstelle hybride SVD-Instanz
    # Wichtig: Wir geben die originalen Backend-Instanzen, bevor
    # wir Änderungen an ihren Methoden vornehmen
    hybrid_svd = HybridSVD(original_backend, optimized_backend, enable_telemetry=enable_telemetry)
    
    # Ersetze die SVD-Methode NUR im optimierten Backend
    # NICHT im original_backend, um Rekursionsschleifen zu vermeiden
    optimized_backend.svd = hybrid_svd.svd
    
    # Füge Statistik-Methode hinzu
    optimized_backend.get_svd_stats = hybrid_svd.get_stats
    
    # Füge Methode zur Aktualisierung der Performance-Map hinzu
    optimized_backend.update_svd_performance_map = hybrid_svd.update_performance_map
    
    logger.info("Hybride SVD-Implementierung erfolgreich installiert")
    return optimized_backend
