#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Engine - Optimierte Integration mit ECHO-PRIME

Diese Datei implementiert eine verbesserte Integration zwischen der T-Mathematics Engine
und ECHO-PRIME mit optimierten Tensor-Konvertierungen und MLX-Unterstützung für Apple Silicon.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch
import numpy as np

# Import für MLX, wenn verfügbar
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from miso.math.t_mathematics.engine import TMathEngine
from miso.math.t_mathematics.compat import TMathConfig
from optimized_matrix_core import improved_tensor_conversion

# Logger konfigurieren
logger = logging.getLogger("t_mathematics.optimized_echo_prime_integration")

class OptimizedTimelineAnalysisEngine:
    """
    Verbesserte Engine für hochperformante Zeitlinienanalysen mit T-Mathematics und MLX.
    
    Diese Klasse bietet optimierte mathematische Operationen für ECHO-PRIME,
    insbesondere für Zeitlinienanalysen und Zeitknotenberechnungen mit verbesserter
    Tensor-Konvertierung zwischen PyTorch, MLX und NumPy.
    """
    
    def __init__(self, 
                use_mlx: bool = True,
                precision: str = "float16",
                device: str = "auto",
                enable_caching: bool = True):
        """
        Initialisiert die optimierte Timeline Analysis Engine.
        
        Args:
            use_mlx: Ob MLX verwendet werden soll (wenn verfügbar)
            precision: Präzisionstyp für Berechnungen
            device: Zielgerät für Berechnungen
            enable_caching: Ob Caching für wiederholte Operationen aktiviert werden soll
        """
        # Überprüfe MLX-Verfügbarkeit
        self.mlx_available = MLX_AVAILABLE
        if use_mlx and not self.mlx_available:
            logger.warning("MLX wurde angefordert, ist aber nicht verfügbar. Fallback auf PyTorch.")
            use_mlx = False
        
        # Erstelle T-Mathematics Engine mit MLX-Optimierung
        self.config = TMathConfig(
            precision=precision,
            device=device,
            optimize_for_rdna=False,
            optimize_for_apple_silicon=True
        )
        self.engine = TMathEngine(
            config=self.config,
            use_mlx=use_mlx
        )
        
        # Cache für wiederholte Operationen
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Bevorzugtes Backend für Hauptoperationen
        self.preferred_backend = "mlx" if use_mlx and self.mlx_available else "torch"
        
        logger.info(f"Optimierte Timeline Analysis Engine initialisiert: MLX={self.engine.use_mlx}, "
                   f"Gerät={self.engine.device}, Präzision={self.engine.precision}, "
                   f"Caching={self.enable_caching}")
        
    def _convert_tensor(self, tensor, target_type):
        """
        Konvertiert einen Tensor in das gewünschte Format mit optimierter Konvertierung.
        
        Args:
            tensor: Eingangs-Tensor (torch.Tensor, np.ndarray, mlx.core.array)
            target_type: Zielformat ('numpy', 'mlx', 'torch')
            
        Returns:
            Konvertierter Tensor im gewünschten Format
        """
        return improved_tensor_conversion(tensor, target_type=target_type)
        
    def _generate_cache_key(self, op_name, *tensors):
        """
        Generiert einen deterministischen Cache-Schlüssel für Tensor-Operationen.
        
        Args:
            op_name: Name der Operation
            *tensors: Tensoren, die in die Operation eingehen
            
        Returns:
            tuple: Cache-Schlüssel oder None bei Problemen
        """
        if not self.enable_caching:
            return None
            
        try:
            # Versuche, einen Hash aus den Tensor-Shapes, Typen und einer Stichprobe zu erzeugen
            keys = []
            for t in tensors:
                if isinstance(t, torch.Tensor):
                    keys.append(hash((t.shape, str(t.dtype), t.detach().cpu().numpy().flatten()[:5].tobytes())))
                elif isinstance(t, np.ndarray):
                    keys.append(hash((t.shape, str(t.dtype), t.flatten()[:5].tobytes())))
                elif MLX_AVAILABLE and isinstance(t, mx.array):
                    np_sample = np.array(t.tolist()).flatten()[:5]
                    keys.append(hash((t.shape, str(t.dtype), np_sample.tobytes())))
                else:
                    # Fallback für andere Typen
                    keys.append(hash(str(t)))
                    
            return (op_name, tuple(keys))
        except Exception as e:
            logger.debug(f"Fehler bei Cache-Key-Generierung: {e}")
            return None
        
    def timeline_similarity(self, timeline1, timeline2):
        """
        Berechnet die Ähnlichkeit zwischen zwei Zeitlinien.
        
        Verwendet optimierte Tensor-Konvertierungen und MLX-Operationen
        auf Apple Silicon, wenn verfügbar.
        
        Args:
            timeline1: Erste Zeitlinie als Tensor [seq_len, features]
            timeline2: Zweite Zeitlinie als Tensor [seq_len, features]
            
        Returns:
            Ähnlichkeitswert zwischen 0 und 1
        """
        # Cache-Schlüssel generieren
        cache_key = self._generate_cache_key("timeline_similarity", timeline1, timeline2)
        if cache_key is not None and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        else:
            if cache_key is not None:
                self.cache_misses += 1
        
        # Wähle optimales Backend für diese Operation
        backend = self.preferred_backend
        
        try:
            # Konvertiere zu bevorzugtem Backend
            t1 = self._convert_tensor(timeline1, backend)
            t2 = self._convert_tensor(timeline2, backend)
            
            if backend == "mlx":
                # MLX-Implementierung
                dot_product = mx.sum(t1 * t2)
                norm1 = mx.sqrt(mx.sum(t1 * t1))
                norm2 = mx.sqrt(mx.sum(t2 * t2))
                similarity = dot_product / (norm1 * norm2 + 1e-8)
                
                # Konvertiere zurück zu PyTorch für Konsistenz
                similarity_torch = torch.tensor(similarity.item())
                
            else:
                # PyTorch-Implementierung
                dot_product = torch.sum(t1 * t2)
                norm1 = torch.sqrt(torch.sum(t1 * t1))
                norm2 = torch.sqrt(torch.sum(t2 * t2))
                similarity_torch = dot_product / (norm1 * norm2 + 1e-8)
            
            # Cache-Ergebnis
            if cache_key is not None:
                self.cache[cache_key] = similarity_torch
                
            return similarity_torch
            
        except Exception as e:
            logger.error(f"Fehler bei timeline_similarity: {e}")
            # Fallback auf robuste Implementierung
            try:
                # Konvertiere zu NumPy für maximale Kompatibilität
                t1_np = self._convert_tensor(timeline1, "numpy")
                t2_np = self._convert_tensor(timeline2, "numpy")
                
                dot_product = np.sum(t1_np * t2_np)
                norm1 = np.sqrt(np.sum(t1_np * t1_np))
                norm2 = np.sqrt(np.sum(t2_np * t2_np))
                similarity_np = dot_product / (norm1 * norm2 + 1e-8)
                
                return torch.tensor(similarity_np.item())
            except:
                # Absoluter Notfall-Fallback
                return torch.tensor(0.0)
    
    def temporal_attention(self, 
                         timenode_queries, 
                         timeline_keys,
                         timeline_values,
                         temporal_mask: Optional = None):
        """
        Führt temporale Attention zwischen Zeitknoten-Queries und Zeitlinien aus.
        
        Verwendet optimierte Tensor-Konvertierungen und MLX-Operationen
        auf Apple Silicon, wenn verfügbar.
        
        Args:
            timenode_queries: Zeitknoten-Queries [batch, num_queries, embed_dim]
            timeline_keys: Zeitlinien-Keys [batch, seq_len, embed_dim]
            timeline_values: Zeitlinien-Values [batch, seq_len, value_dim]
            temporal_mask: Optionale Maske für zeitliche Beschränkungen [batch, num_queries, seq_len]
            
        Returns:
            Tuple mit (attention_output, attention_weights)
        """
        # Cache-Schlüssel generieren
        cache_key = self._generate_cache_key("temporal_attention", 
                                            timenode_queries, 
                                            timeline_keys, 
                                            timeline_values, 
                                            temporal_mask)
        if cache_key is not None and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        else:
            if cache_key is not None:
                self.cache_misses += 1
        
        # Wähle optimales Backend für diese Operation
        backend = self.preferred_backend
        
        try:
            # Konvertiere zu bevorzugtem Backend
            q = self._convert_tensor(timenode_queries, backend)
            k = self._convert_tensor(timeline_keys, backend)
            v = self._convert_tensor(timeline_values, backend)
            
            if temporal_mask is not None:
                mask = self._convert_tensor(temporal_mask, backend)
            else:
                mask = None
            
            if backend == "mlx":
                # MLX-Implementierung mit performance-optimiertem Attention
                # scale factor für numerische Stabilität
                scale = 1.0 / mx.sqrt(mx.array(k.shape[-1], dtype=mx.float32))
                
                # Berechne Attention Scores: [batch, num_queries, seq_len]
                attn = mx.matmul(q, mx.transpose(k, (0, 2, 1))) * scale
                
                # Wende Maske an, wenn vorhanden
                if mask is not None:
                    attn = mx.where(mask, attn, mx.array(-1e9))
                
                # Attention Weights mit Softmax
                attn_weights = mx.softmax(attn, axis=-1)
                
                # Attention Output: [batch, num_queries, value_dim]
                attn_output = mx.matmul(attn_weights, v)
                
                # Konvertiere zurück zu PyTorch für Konsistenz
                output_torch = torch.tensor(attn_output.tolist())
                weights_torch = torch.tensor(attn_weights.tolist())
                
            else:
                # PyTorch-Implementierung
                scale = 1.0 / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32))
                
                # Berechne Attention Scores: [batch, num_queries, seq_len]
                attn = torch.matmul(q, k.transpose(1, 2)) * scale
                
                # Wende Maske an, wenn vorhanden
                if mask is not None:
                    attn = attn.masked_fill(~mask, -1e9)
                
                # Attention Weights mit Softmax
                attn_weights = torch.softmax(attn, dim=-1)
                
                # Attention Output: [batch, num_queries, value_dim]
                attn_output = torch.matmul(attn_weights, v)
                
                output_torch = attn_output
                weights_torch = attn_weights
            
            # Cache-Ergebnis
            if cache_key is not None:
                self.cache[cache_key] = (output_torch, weights_torch)
                
            return output_torch, weights_torch
            
        except Exception as e:
            logger.error(f"Fehler bei temporal_attention: {e}")
            # Fallback auf Engine
            return self.engine.compute_attention(
                timenode_queries, timeline_keys, timeline_values, mask=temporal_mask
            )
            
    def cache_stats(self):
        """Gibt Cache-Statistiken zurück"""
        if not self.enable_caching:
            return {"caching_enabled": False}
            
        return {
            "caching_enabled": True,
            "cache_size": len(self.cache) if self.cache else 0,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }
        
    def clear_cache(self):
        """Leert den Cache"""
        if self.enable_caching and self.cache:
            self.cache.clear()
            logger.info("Cache geleert")
            

# Einfacher Test
if __name__ == "__main__":
    # Konfiguriere Logging
    logging.basicConfig(level=logging.INFO)
    
    # Erstelle eine optimierte Timeline Analysis Engine
    print("Initialisiere Optimierte Timeline Analysis Engine...")
    analysis_engine = OptimizedTimelineAnalysisEngine(use_mlx=True, enable_caching=True)
    
    # Erstelle Test-Zeitlinien
    print("Erzeuge Test-Zeitlinien...")
    timeline1 = torch.randn(5, 10)  # [seq_len, features]
    timeline2 = torch.randn(5, 10)  # [seq_len, features]
    
    # Teste Timeline-Ähnlichkeit
    print("Berechne Zeitlinien-Ähnlichkeit...")
    similarity = analysis_engine.timeline_similarity(timeline1, timeline2)
    print(f"Ähnlichkeit: {similarity.item():.4f}")
    
    # Teste Cache
    print("Teste Cache-Funktionalität...")
    similarity_cached = analysis_engine.timeline_similarity(timeline1, timeline2)
    print(f"Ähnlichkeit (cached): {similarity_cached.item():.4f}")
    print(f"Cache-Stats: {analysis_engine.cache_stats()}")
    
    # Teste Temporal Attention
    print("Teste Temporal Attention...")
    batch_size = 2
    num_queries = 3
    seq_len = 5
    embed_dim = 10
    value_dim = 8
    
    queries = torch.randn(batch_size, num_queries, embed_dim)
    keys = torch.randn(batch_size, seq_len, embed_dim)
    values = torch.randn(batch_size, seq_len, value_dim)
    
    output, weights = analysis_engine.temporal_attention(queries, keys, values)
    print(f"Attention Output Shape: {output.shape}")
    print(f"Attention Weights Shape: {weights.shape}")
