#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Engine - Kernimplementierung

Diese Datei implementiert die Hauptfunktionalität der T-Mathematics Engine,
die für Tensor- und MoE-optimierte mathematische Berechnungen zuständig ist.
Optimiert für Apple Silicon M4 Max und AMD RDNA3 Hardware.
Mit MLX-Optimierung für Apple Silicon.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
import platform
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field

from .compat import TMathConfig, TMathematicsEngine
from .ops import (
    tensor_svd, 
    amd_optimized_matmul, 
    moe_routing, 
    mix_experts_outputs,
    positional_encoding,
    attention_wrapper
)

# Logger konfigurieren
logger = logging.getLogger("t_mathematics")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# MLX-Backend importieren, falls verfügbar
HAS_MLX = False
try:
    from .mlx_support import MLXBackend, HAS_MLX, IS_APPLE_SILICON
    if HAS_MLX:
        logger.info("MLX-Backend für Apple Silicon erfolgreich geladen")
except ImportError:
    logger.warning("MLX-Backend konnte nicht importiert werden. Verwende PyTorch als Fallback.")

# Umgebungsvariablen für Konfiguration
DEFAULT_PRECISION = os.environ.get("T_MATH_PRECISION", "mixed")
DEFAULT_DEVICE = os.environ.get("T_MATH_DEVICE", "auto")
GPU_CORES = int(os.environ.get("T_MATH_GPU_CORES", "0"))
USE_FLASH_ATTENTION = os.environ.get("T_MATH_FLASH_ATTENTION", "1") == "1"
OPTIMIZE_FOR_AMD = os.environ.get("T_MATH_OPTIMIZE_AMD", "1") == "1"
OPTIMIZE_FOR_APPLE = os.environ.get("T_MATH_OPTIMIZE_APPLE", "1") == "1"
USE_MLX = os.environ.get("T_MATH_USE_MLX", "1") == "1" and HAS_MLX

# Prüfe auf Apple Silicon
is_apple_silicon = False
try:
    is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
    if is_apple_silicon:
        logger.info("Apple Silicon erkannt, optimiere für Neural Engine")
except:
    pass


class TMathEngine:
    """
    T-Mathematics Engine - Hauptklasse
    
    Diese Klasse stellt die Hauptschnittstelle für die T-Mathematics Engine dar.
    Sie bietet optimierte mathematische Operationen für Sprachmodelle und
    andere KI-Anwendungen, insbesondere für Apple Silicon und AMD-Hardware.
    """
    
    def __init__(self, 
                config: Optional[TMathConfig] = None,
                device: str = DEFAULT_DEVICE,
                precision: str = DEFAULT_PRECISION,
                optimize_for_amd: bool = OPTIMIZE_FOR_AMD,
                optimize_for_apple: bool = OPTIMIZE_FOR_APPLE,
                use_mlx: bool = USE_MLX):
        """
        Initialisiert die T-Mathematics Engine.
        
        Args:
            config: Optionale Konfiguration für die Engine
            device: Zielgerät ("cpu", "gpu", "auto")
            precision: Präzisionstyp ("single", "double", "mixed", "float16", "bfloat16")
            optimize_for_amd: Ob für AMD-Hardware optimiert werden soll
            optimize_for_apple: Ob für Apple Silicon optimiert werden soll
        """
        # Konfiguration erstellen oder übernehmen
        self.config = config or TMathConfig(
            precision=precision,
            device=device,
            optimize_for_rdna=optimize_for_amd,
            optimize_for_apple_silicon=optimize_for_apple
        )
        
        # MLX-Konfiguration
        self.use_mlx = use_mlx and HAS_MLX and IS_APPLE_SILICON
        self.mlx_backend = MLXBackend(precision=precision) if self.use_mlx else None
        
        # Initialisiere die Kompatibilitätsschicht
        self.compat_engine = TMathematicsEngine(self.config)
        
        # Bestimme das Gerät
        self._setup_device()
        
        # Bestimme den Präzisionstyp
        self._setup_precision()
        
        # Initialisiere die Engine
        self._initialize_engine()
        
        logger.info(f"T-Mathematics Engine initialisiert: Gerät={self.device}, "
                   f"Präzision={self.precision}, "
                   f"AMD-Optimierungen={self.compat_engine.using_amd_optimizations}, "
                   f"Apple-Optimierungen={self.compat_engine.using_apple_silicon}, "
                   f"MLX-Backend={self.use_mlx}")
    
    def _setup_device(self):
        """Konfiguriert das Zielgerät für die Engine"""
        if self.config.device == "auto":
            # Automatische Geräteerkennung
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            # Manuell konfiguriertes Gerät
            self.device = torch.device(self.config.device)
            
        logger.debug(f"Verwende Gerät: {self.device}")
    
    def _setup_precision(self):
        """Konfiguriert den Präzisionstyp für die Engine"""
        precision_map = {
            "single": torch.float32,
            "float32": torch.float32,
            "double": torch.float64,
            "float64": torch.float64,
            "half": torch.float16,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16,
            "mixed": torch.float16 if self.device.type != "cpu" else torch.float32
        }
        
        self.precision = precision_map.get(self.config.precision, torch.float32)
        logger.debug(f"Verwende Präzision: {self.precision}")
    
    def _initialize_engine(self):
        """Initialisiert die Engine mit den konfigurierten Einstellungen"""
        # Aktiviere Autocast für Mixed Precision, falls verfügbar
        self.use_autocast = (
            self.config.precision == "mixed" and 
            self.device.type in ["cuda", "mps"] and
            hasattr(torch.cuda, "amp")
        )
        
        # Konfiguriere Flash Attention
        self.use_flash_attention = (
            USE_FLASH_ATTENTION and 
            self.config.use_flash_attention and
            self.device.type == "cuda"
        )
        
        # Optimierungsfaktor für spezifische Hardware
        self.hardware_scale_factor = 1.0
        if self.compat_engine.using_amd_optimizations:
            self.hardware_scale_factor = 0.9  # Optimierung für AMD RDNA3
        elif self.compat_engine.using_apple_silicon:
            if self.use_mlx:
                self.hardware_scale_factor = 1.3  # Optimierung für Apple Silicon mit MLX
            else:
                self.hardware_scale_factor = 1.1  # Optimierung für Apple Silicon mit PyTorch
            
        logger.debug(f"Engine initialisiert: Autocast={self.use_autocast}, "
                    f"Flash Attention={self.use_flash_attention}, "
                    f"Hardware-Faktor={self.hardware_scale_factor}, "
                    f"MLX-Backend={self.use_mlx}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Gibt detaillierte Geräteinformationen zurück"""
        import platform
        
        device_info = {
            "device": str(self.device),
            "precision": str(self.precision),
            "backend": "mlx" if self.use_mlx else "pytorch",
            "platform": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hardware_scale_factor": self.hardware_scale_factor,
            "use_autocast": self.use_autocast,
            "use_flash_attention": self.use_flash_attention
        }
        
        # GPU-spezifische Informationen
        if self.device.type == "cuda" and torch.cuda.is_available():
            device_info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_count": torch.cuda.device_count()
            })
        elif self.device.type == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info.update({
                "gpu_name": "Apple Silicon GPU",
                "mps_available": True,
                "neural_engine": True
            })
        
        # MLX-spezifische Informationen
        if self.use_mlx and self.mlx_backend:
            try:
                import mlx.core as mx
                device_info.update({
                    "mlx_version": getattr(mx, '__version__', 'unknown'),
                    "mlx_device": "gpu" if hasattr(mx, 'default_device') else "cpu"
                })
            except:
                pass
        
        return device_info
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Verschiebt einen Tensor auf das konfigurierte Gerät.
        
        Args:
            tensor: Eingabetensor
            
        Returns:
            Tensor auf dem konfigurierten Gerät
        """
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor
    
    def to_precision(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Konvertiert einen Tensor in die konfigurierte Präzision.
        
        Args:
            tensor: Eingabetensor
            
        Returns:
            Tensor mit der konfigurierten Präzision
        """
        if tensor.dtype != self.precision and tensor.dtype not in [torch.int32, torch.int64, torch.bool]:
            return tensor.to(dtype=self.precision)
        return tensor
    
    def prepare_tensor(self, tensor: Any) -> torch.Tensor:
        """
        Bereitet einen Tensor für die Verwendung in der Engine vor.
        Unterstützt verschiedene Eingabetypen: PyTorch-Tensor, NumPy-Array, MLX-Array oder Listen.
        
        Args:
            tensor: Eingabedaten (PyTorch-Tensor, NumPy-Array, MLX-Array oder Liste)
            
        Returns:
            Vorbereiteter PyTorch-Tensor
        """
        # Prüfe den Typ des Tensors und konvertiere entsprechend
        if tensor is None:
            raise ValueError("Tensor darf nicht None sein")
        
        # Wenn es bereits ein PyTorch-Tensor ist
        if isinstance(tensor, torch.Tensor):
            return self.to_precision(self.to_device(tensor))
        
        # Wenn es ein NumPy-Array ist
        elif isinstance(tensor, np.ndarray):
            # Konvertiere NumPy-Array zu PyTorch-Tensor
            torch_tensor = torch.from_numpy(tensor)
            return self.to_precision(self.to_device(torch_tensor))
        
        # Wenn es ein MLX-Array ist (falls MLX verfügbar)
        elif self.use_mlx and self.mlx_backend is not None and hasattr(tensor, 'to_numpy'):
            # Konvertiere MLX-Array zu NumPy und dann zu PyTorch
            numpy_array = tensor.to_numpy()
            torch_tensor = torch.from_numpy(numpy_array)
            return self.to_precision(self.to_device(torch_tensor))
        
        # Wenn es eine Liste oder ein Tuple ist
        elif isinstance(tensor, (list, tuple)):
            # Konvertiere Liste zu PyTorch-Tensor
            torch_tensor = torch.tensor(tensor, dtype=torch.float32)
            return self.to_precision(self.to_device(torch_tensor))
        
        # Wenn es ein Skalar ist
        elif isinstance(tensor, (int, float)):
            # Konvertiere Skalar zu PyTorch-Tensor
            torch_tensor = torch.tensor([tensor], dtype=torch.float32)
            return self.to_precision(self.to_device(torch_tensor))
        
        # Fallback: Versuche, es als PyTorch-Tensor zu behandeln
        else:
            try:
                # Versuche, es als PyTorch-Tensor zu behandeln
                return self.to_precision(self.to_device(tensor))
            except Exception as e:
                # Wenn das nicht funktioniert, versuche eine generische Konvertierung
                try:
                    torch_tensor = torch.tensor(tensor, dtype=torch.float32)
                    return self.to_precision(self.to_device(torch_tensor))
                except Exception as inner_e:
                    raise ValueError(f"Konnte den Tensor nicht vorbereiten. Typ: {type(tensor)}. Fehler: {str(e)}, {str(inner_e)}")
    
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Führt eine optimierte Matrixmultiplikation durch.
        
        Verwendet MLX für Apple Silicon, falls verfügbar, oder fällt auf optimierte
        PyTorch-Implementierungen zurück.
        
        Args:
            a: Erster Tensor
            b: Zweiter Tensor
            
        Returns:
            Ergebnis der Matrixmultiplikation
        """
        # Temporär MLX-Backend für matmul umgehen (Blocker-Fix)
        # TODO: MLX matmul-Implementation reparieren (gibt Liste statt Tensor zurück)
        if False:  # self.use_mlx and self.mlx_backend is not None:
            try:
                # Bereite Tensoren vor, um sicherzustellen, dass sie kompatibel sind
                a_prepared = self.prepare_tensor(a)
                b_prepared = self.prepare_tensor(b)
                
                # Konvertiere PyTorch-Tensoren zu CPU vor der MLX-Konvertierung
                if hasattr(a_prepared, "device") and str(a_prepared.device) != "cpu":
                    a_prepared = a_prepared.cpu()
                if hasattr(b_prepared, "device") and str(b_prepared.device) != "cpu":
                    b_prepared = b_prepared.cpu()
                
                # Verwende MLX für Matrixmultiplikation
                return self.mlx_backend.matmul(a_prepared, b_prepared)
            except Exception as e:
                logger.warning(f"MLX-Matrixmultiplikation fehlgeschlagen: {e}. Verwende PyTorch-Fallback.")
                # Fallback zu PyTorch bei Fehlern
        
        # Fallback auf PyTorch-Implementierung
        # Bereite Tensoren vor
        a = self.prepare_tensor(a)
        b = self.prepare_tensor(b)
        
        # Wähle die optimale Implementierung
        if self.compat_engine.using_amd_optimizations:
            optimize_for = "amd"
        elif self.compat_engine.using_apple_silicon:
            optimize_for = "apple"
        else:
            optimize_for = "default"
        
        # Führe die Matrixmultiplikation durch
        if self.use_autocast:
            with torch.cuda.amp.autocast():
                return amd_optimized_matmul(a, b, optimize_for=optimize_for)
        else:
            return amd_optimized_matmul(a, b, optimize_for=optimize_for)
    
    def svd(self, tensor: torch.Tensor, k: Optional[int] = None, 
           implementation: str = "randomized") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Führt eine optimierte SVD-Zerlegung durch.
        
        Args:
            tensor: Eingabetensor
            k: Anzahl der zurückzugebenden Komponenten, oder None für alle
            implementation: Implementierungsmethode ("full", "randomized", "truncated")
            
        Returns:
            Tuple aus (U, S, V) Matrizen
        """
        # Temporär MLX-Backend für SVD umgehen (Blocker-Fix)
        # TODO: MLX SVD-Implementation reparieren
        if False:  # self.use_mlx and self.mlx_backend is not None:
            try:
                # Verwende MLX für SVD
                result = self.mlx_backend.svd(tensor, k)
                # Stelle sicher, dass das Ergebnis ein Tuple aus Tensoren ist
                if isinstance(result, (list, tuple)) and len(result) == 3:
                    return tuple(result)
                else:
                    logger.warning("MLX SVD gab unerwartetes Format zurück, verwende PyTorch Fallback")
            except Exception as e:
                logger.warning(f"MLX SVD fehlgeschlagen: {e}, verwende PyTorch Fallback")
        
        # Fallback auf PyTorch-Implementierung
        # Bereite Tensor vor
        tensor = self.prepare_tensor(tensor)
        
        # Wähle die Implementierung basierend auf der Konfiguration
        impl = implementation or self.config.svd_implementation
        
        # Führe die SVD durch
        result = tensor_svd(tensor, k, implementation=impl)
        
        # Stelle sicher, dass das Ergebnis ein Tuple aus Tensoren ist
        if isinstance(result, (list, tuple)) and len(result) == 3:
            u, s, v = result
            # Prüfe, ob alle Komponenten Tensoren sind
            if hasattr(u, 'shape') and hasattr(s, 'shape') and hasattr(v, 'shape'):
                return u, s, v
            else:
                logger.error(f"SVD Ergebnis enthält Nicht-Tensoren: U={type(u)}, S={type(s)}, V={type(v)}")
                raise ValueError("SVD gab ungültige Tensortypen zurück")
        else:
            logger.error(f"SVD gab unerwartetes Format zurück: {type(result)}")
            raise ValueError("SVD gab unerwartetes Format zurück")
    
    def batch_matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Führt eine Batch-Matrixmultiplikation durch.
        
        Args:
            a: Erster Tensor [batch_size, ..., m, k]
            b: Zweiter Tensor [batch_size, ..., k, n]
            
        Returns:
            Ergebnis der Batch-Matrixmultiplikation [batch_size, ..., m, n]
        """
        # Bereite Tensoren vor
        a = self.prepare_tensor(a)
        b = self.prepare_tensor(b)
        
        # Prüfe auf MLX-Backend für Apple Silicon
        if self.use_mlx and self.mlx_backend is not None:
            try:
                # Verwende MLX für Batch-Matrixmultiplikation
                return self.mlx_backend.batch_matmul(a, b)
            except Exception as e:
                logger.warning(f"MLX Batch-Matmul fehlgeschlagen: {e}, verwende PyTorch Fallback")
        
        # PyTorch Batch-Matrixmultiplikation
        return torch.bmm(a, b)
    
    def route_to_experts(self, inputs: torch.Tensor, 
                        router_weights: torch.Tensor, 
                        top_k: int = 2,
                        noise_epsilon: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Berechnet die Routing-Entscheidung für die MoE-Architektur.
        
        Args:
            inputs: Einbettungen der Eingabe [batch_size, seq_len, hidden_dim]
            router_weights: Routing-Gewichte [hidden_dim, num_experts]
            top_k: Anzahl der Experten pro Token
            noise_epsilon: Rauschfaktor für Load-Balancing
            
        Returns:
            Tuple aus (routing_probabilities, expert_indices)
        """
        # Bereite Tensoren vor
        inputs = self.prepare_tensor(inputs)
        router_weights = self.prepare_tensor(router_weights)
        
        # Berechne Routing
        return moe_routing(inputs, router_weights, top_k, noise_epsilon)
    
    def combine_expert_outputs(self, expert_outputs: List[torch.Tensor], 
                             expert_weights: torch.Tensor,
                             normalize_weights: bool = True) -> torch.Tensor:
        """
        Mischt die Ausgaben mehrerer Experten basierend auf den Gewichten.
        
        Args:
            expert_outputs: Liste der Expertenausgaben
            expert_weights: Gewichte für jeden Experten
            normalize_weights: Ob die Gewichte normalisiert werden sollen
            
        Returns:
            Gewichtete Summe der Expertenausgaben
        """
        # Bereite Tensoren vor
        prepared_outputs = [self.prepare_tensor(output) for output in expert_outputs]
        expert_weights = self.prepare_tensor(expert_weights)
        
        # Mische Expertenausgaben
        return mix_experts_outputs(prepared_outputs, expert_weights, normalize_weights)
    
    def get_positional_encoding(self, seq_len: int, dim: int, 
                              max_len: int = 10000) -> torch.Tensor:
        """
        Erzeugt Positionscodierungen für Transformers.
        
        Args:
            seq_len: Länge der Sequenz
            dim: Dimension der Codierung
            max_len: Maximale Sequenzlänge für die Skalierung
            
        Returns:
            Positionscodierungen mit Form [seq_len, dim]
        """
        return positional_encoding(seq_len, dim, self.device, max_len)
    
    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                dropout_p: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Berechnet die Attention-Mechanismus.
        
        Args:
            query: Query-Tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key-Tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value-Tensor [batch_size, num_heads, seq_len, head_dim]
            mask: Optionale Aufmerksamkeitsmaske [batch_size, 1, seq_len, seq_len]
            dropout_p: Dropout-Wahrscheinlichkeit
            
        Returns:
            Tuple aus (Attention-Output, Attention-Gewichte)
        """
        # Prüfe auf MLX-Backend für Apple Silicon
        if self.use_mlx and self.mlx_backend is not None:
            # Verwende MLX für Attention
            output = self.mlx_backend.attention(query, key, value, mask)
            
            # MLXBackend.attention gibt nur das Ergebnis zurück, nicht die Gewichte
            # Berechne die Gewichte separat
            d_k = query.size(-1)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            weights = torch.nn.functional.softmax(scores, dim=-1)
            
            return output, weights
        
        # Fallback auf PyTorch-Implementierung
        # Bereite Tensoren vor
        query = self.prepare_tensor(query)
        key = self.prepare_tensor(key)
        value = self.prepare_tensor(value)
        if mask is not None:
            mask = self.to_device(mask)
        
        # Wende Attention an
        return attention_wrapper(
            query, key, value, 
            mask=mask, 
            dropout_p=dropout_p,
            use_flash_attention=self.use_flash_attention
        )
    
    def create_attention_layer(self, 
                             embed_dim: int, 
                             num_heads: int,
                             dropout: float = 0.1,
                             bias: bool = True) -> nn.Module:
        """
        Erstellt eine optimierte Multi-Head-Attention-Schicht.
        
        Args:
            embed_dim: Einbettungsdimension
            num_heads: Anzahl der Attention-Köpfe
            dropout: Dropout-Wahrscheinlichkeit
            bias: Ob Bias verwendet werden soll
            
        Returns:
            Multi-Head-Attention-Modul
        """
        return OptimizedMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            engine=self
        )
    
    def create_feedforward_layer(self,
                               input_dim: int,
                               hidden_dim: int,
                               output_dim: int,
                               activation: str = "gelu",
                               dropout: float = 0.1) -> nn.Module:
        """
        Erstellt eine optimierte Feed-Forward-Schicht.
        
        Args:
            input_dim: Eingabedimension
            hidden_dim: Versteckte Dimension
            output_dim: Ausgabedimension
            activation: Aktivierungsfunktion
            dropout: Dropout-Wahrscheinlichkeit
            
        Returns:
            Feed-Forward-Modul
        """
        return OptimizedFeedForward(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
            engine=self
        )
    
    def layer_norm(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                  eps: float = 1e-5) -> torch.Tensor:
        """
        Führt Layer-Normalisierung durch.
        
        Args:
            input: Eingabetensor
            weight: Gewichtstensor für die Skalierung
            bias: Bias-Tensor für den Offset
            eps: Epsilon für numerische Stabilität
            
        Returns:
            Normalisierter Tensor
        """
        # Prüfe auf MLX-Backend für Apple Silicon
        if self.use_mlx and self.mlx_backend is not None:
            # Verwende MLX für Layer-Normalisierung
            input_mlx = tensor_to_mlx(input)
            weight_mlx = tensor_to_mlx(weight)
            bias_mlx = tensor_to_mlx(bias) if bias is not None else None
            
            result_mlx = self.mlx_backend.layer_norm(input_mlx, weight_mlx, bias_mlx, eps)
            
            return mlx_to_tensor(result_mlx)
        
        # Fallback auf PyTorch-Implementierung
        # Bereite Tensoren vor
        input = self.prepare_tensor(input)
        weight = self.prepare_tensor(weight)
        bias = self.prepare_tensor(bias) if bias is not None else None
        
        # Führe Layer-Normalisierung durch
        return F.layer_norm(input, input.shape[-1:], weight, bias, eps)
    
    def gelu(self, input: torch.Tensor) -> torch.Tensor:
        """
        Wendet die GELU-Aktivierungsfunktion an.
        
        Args:
            input: Eingabetensor
            
        Returns:
            Tensor nach Anwendung von GELU
        """
        # Prüfe auf MLX-Backend für Apple Silicon
        if self.use_mlx and self.mlx_backend is not None:
            # Verwende MLX für GELU
            input_mlx = tensor_to_mlx(input)
            result_mlx = self.mlx_backend.gelu(input_mlx)
            
            return mlx_to_tensor(result_mlx)
        
        # Fallback auf PyTorch-Implementierung
        # Bereite Tensor vor
        input = self.prepare_tensor(input)
        
        # Führe GELU durch
        return F.gelu(input)
    
    def relu(self, input: torch.Tensor) -> torch.Tensor:
        """
        Wendet die ReLU-Aktivierungsfunktion an.
        
        Args:
            input: Eingabetensor
            
        Returns:
            Tensor nach Anwendung von ReLU
        """
        # Prüfe auf MLX-Backend für Apple Silicon
        if self.use_mlx and self.mlx_backend is not None:
            # Verwende MLX für ReLU
            input_mlx = tensor_to_mlx(input)
            result_mlx = self.mlx_backend.relu(input_mlx)
            
            return mlx_to_tensor(result_mlx)
        
        # Fallback auf PyTorch-Implementierung
        # Bereite Tensor vor
        input = self.prepare_tensor(input)
        
        # Führe ReLU durch
        return F.relu(input)
    
    def create_moe_layer(self,
                       input_dim: int,
                       hidden_dim: int,
                       output_dim: int,
                       num_experts: int,
                       expert_capacity: int,
                       router_jitter_noise: float = 0.01,
                       activation: str = "gelu") -> nn.Module:
        """
        Erstellt eine Mixture-of-Experts-Schicht.
        
        Args:
            input_dim: Eingabedimension
            hidden_dim: Versteckte Dimension
            output_dim: Ausgabedimension
            num_experts: Anzahl der Experten
            expert_capacity: Kapazität jedes Experten
            router_jitter_noise: Rauschfaktor für den Router
            activation: Aktivierungsfunktion
            
        Returns:
            MoE-Modul
        """
        return MixtureOfExperts(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            expert_capacity=expert_capacity,
            router_jitter_noise=router_jitter_noise,
            activation=activation,
            engine=self
        )
    
    def layer_norm(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                  eps: float = 1e-5) -> torch.Tensor:
        """
        Führt Layer-Normalisierung durch.
        
        Args:
            input: Eingabetensor
            weight: Gewichtstensor für die Skalierung
            bias: Bias-Tensor für den Offset
            eps: Epsilon für numerische Stabilität
            
        Returns:
            Normalisierter Tensor
        """
        # Prüfe auf MLX-Backend für Apple Silicon
        if self.use_mlx and self.mlx_backend is not None:
            # Verwende MLX für Layer-Normalisierung
            input_mlx = self.tensor_to_mlx(input)
            weight_mlx = self.tensor_to_mlx(weight)
            bias_mlx = self.tensor_to_mlx(bias) if bias is not None else None
            
            output_mlx = self.mlx_backend.layer_norm(input_mlx, weight_mlx, bias_mlx, eps)
            return self.mlx_to_tensor(output_mlx)
        
        # Fallback auf PyTorch-Implementierung
        # Bereite Tensoren vor
        input = self.prepare_tensor(input)
        weight = self.prepare_tensor(weight)
        if bias is not None:
            bias = self.prepare_tensor(bias)
        
        # Führe Layer-Normalisierung durch
        return F.layer_norm(input, input.shape[-1:], weight, bias, eps)
    
    def gelu(self, input: torch.Tensor) -> torch.Tensor:
        """
        Wendet die GELU-Aktivierungsfunktion an.
        
        Args:
            input: Eingabetensor
            
        Returns:
            Tensor nach Anwendung von GELU
        """
        # Prüfe auf MLX-Backend für Apple Silicon
        if self.use_mlx and self.mlx_backend is not None:
            # Verwende MLX für GELU
            input_mlx = tensor_to_mlx(input)
            output_mlx = self.mlx_backend.gelu(input_mlx)
            return mlx_to_tensor(output_mlx)
        
        # Fallback auf PyTorch-Implementierung
        # Bereite Tensor vor
        input = self.prepare_tensor(input)
        
        # Führe GELU durch
        return F.gelu(input)
    
    def relu(self, input: torch.Tensor) -> torch.Tensor:
        """
        Wendet die ReLU-Aktivierungsfunktion an.
        
        Args:
            input: Eingabetensor
            
        Returns:
            Tensor nach Anwendung von ReLU
        """
        # Prüfe auf MLX-Backend für Apple Silicon
        if self.use_mlx and self.mlx_backend is not None:
            # Verwende MLX für ReLU
            input_mlx = tensor_to_mlx(input)
            output_mlx = self.mlx_backend.relu(input_mlx)
            return mlx_to_tensor(output_mlx)
        
        # Fallback auf PyTorch-Implementierung
        # Bereite Tensor vor
        input = self.prepare_tensor(input)
        
        # Führe ReLU durch
        return F.relu(input)
        
    def tensor_to_mlx(self, tensor) -> Any:
        """
        Konvertiert einen Tensor (PyTorch, NumPy, etc.) in ein MLX-Array.
        
        Args:
            tensor: Eingabetensor (PyTorch, NumPy, oder anderes Format)
            
        Returns:
            MLX-Array
        """
        import numpy as np
        
        # Wenn bereits MLX-Array, unverändert zurückgeben
        if str(type(tensor)).find('mlx') >= 0:
            return tensor
            
        # Konvertierung zu NumPy als Zwischenschritt
        numpy_tensor = None
        
        # PyTorch zu NumPy
        if str(type(tensor)).find('torch') >= 0:
            try:
                # .detach().cpu().numpy() für PyTorch-Tensoren auf beliebigem Gerät
                numpy_tensor = tensor.detach().cpu().numpy()
            except Exception as e:
                logger.warning(f"Fehler bei PyTorch->NumPy Konvertierung: {e}")
                # Fallback-Methode: Direkte Konvertierung
                try:
                    numpy_tensor = np.array(tensor.tolist())
                except:
                    raise ValueError(f"Konvertierung von PyTorch zu NumPy fehlgeschlagen für {tensor}")
        
        # Bereits NumPy
        elif isinstance(tensor, np.ndarray):
            numpy_tensor = tensor
        
        # Anderer Typ (z.B. Liste, Tuple)
        else:
            try:
                numpy_tensor = np.array(tensor)
            except:
                raise ValueError(f"Konvertierung von {type(tensor)} zu NumPy fehlgeschlagen")
        
        # NumPy zu MLX
        try:
            import mlx.core as mx
            return mx.array(numpy_tensor)
        except ImportError:
            raise ImportError("MLX nicht verfügbar. Bitte installieren Sie MLX für Apple Silicon Support.")
        except Exception as e:
            raise ValueError(f"Konvertierung von NumPy zu MLX fehlgeschlagen: {e}")
    
    def mlx_to_tensor(self, mlx_array) -> torch.Tensor:
        """
        Konvertiert ein MLX-Array in einen PyTorch-Tensor.
        
        Args:
            mlx_array: MLX-Array
            
        Returns:
            PyTorch-Tensor
        """
        import numpy as np
        import torch
        
        # Prüfe, ob es sich um ein MLX-Array handelt
        if not str(type(mlx_array)).find('mlx') >= 0:
            raise ValueError(f"Eingabe ist kein MLX-Array: {type(mlx_array)}")
        
        # MLX zu NumPy
        try:
            # MLX-Arrays haben keine numpy()-Methode, daher tolist() verwenden
            numpy_tensor = np.array(mlx_array.tolist())
        except Exception as e:
            logger.warning(f"Fehler bei MLX->NumPy Konvertierung: {e}")
            # Fallback: Direktes Kopieren
            try:
                numpy_tensor = np.array(mlx_array)
            except:
                raise ValueError(f"Konvertierung von MLX zu NumPy fehlgeschlagen für {mlx_array}")
        
        # NumPy zu PyTorch
        try:
            return torch.from_numpy(numpy_tensor).to(device=self.device, dtype=self.precision)
        except Exception as e:
            raise ValueError(f"Konvertierung von NumPy zu PyTorch fehlgeschlagen: {e}")
    
    def get_active_backend(self) -> str:
        """
        Gibt das aktuell aktive Backend zurück.
        
        Returns:
            Name des aktiven Backends ('mlx', 'torch', 'numpy')
        """
        if self.use_mlx and self.mlx_backend is not None:
            return 'mlx'
        elif self.device in ['cuda', 'mps']:
            return 'torch'
        else:
            # CPU kann entweder PyTorch oder NumPy sein; wir bevorzugen PyTorch
            return 'torch'
    
    def evaluate(self, expression: str) -> Any:
        """
        Evaluiert einen mathematischen Ausdruck.
        
        Args:
            expression: Mathematischer Ausdruck als String
            
        Returns:
            Ergebnis der Auswertung
        """
        logger.info(f"Evaluiere Ausdruck: {expression}")
        try:
            # Für einfache arithmetische Ausdrücke können wir eval verwenden
            # In einer echten Implementierung würde hier eine sichere Parser-Bibliothek verwendet
            # werden (wie z.B. sympy)
            
            # Sicherheitsmaßnahmen: Nur numerische Operationen erlauben
            allowed_names = {
                'abs': abs, 'max': max, 'min': min, 'pow': pow, 'round': round,
                'sum': sum, 'len': len, 'int': int, 'float': float,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
                'pi': math.pi, 'e': math.e
            }
            
            # Verwende NumPy für komplexere mathematische Funktionen
            for name in dir(np):
                if name.startswith('__'):
                    continue
                if callable(getattr(np, name)) or isinstance(getattr(np, name), (int, float)):
                    allowed_names[name] = getattr(np, name)
            
            # Evaluiere den Ausdruck in einem eingeschränkten Kontext
            # Dies ist noch nicht vollständig sicher und sollte in einer produktiven
            # Umgebung durch eine spezialisierte mathematische Parser-Bibliothek ersetzt werden
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return result
        except Exception as e:
            logger.error(f"Fehler bei der Auswertung von '{expression}': {e}")
            return None
    
    def solve_equation(self, equation: str, variable: Optional[str] = None) -> Any:
        """
        Löst eine mathematische Gleichung.
        
        Args:
            equation: Gleichung als String (z.B. "x^2 + 2*x - 3 = 0")
            variable: Zu lösende Variable (z.B. "x")
            
        Returns:
            Lösung der Gleichung
        """
        logger.info(f"Löse Gleichung: {equation} für {variable}")
        try:
            # In einer realen Implementierung würde hier eine symbolische Algebra-Bibliothek wie sympy verwendet
            # Für diesen Prototyp geben wir eine simulierte Lösung zurück
            return f"Lösung für {variable} in {equation}"
        except Exception as e:
            logger.error(f"Fehler beim Lösen der Gleichung '{equation}': {e}")
            return None
    def tensor(self, data, dtype=None, device=None):
        """
        Erstellt einen Tensor aus den gegebenen Daten.
        
        Args:
            data: Eingabedaten (NumPy-Array, Liste, PyTorch-Tensor oder MLX-Array)
            dtype: Optional, Datentyp für den Tensor
            device: Optional, Gerät für den Tensor
            
        Returns:
            Erstellter Tensor (MLXTensorWrapper oder TorchTensorWrapper)
        """
        from .tensor_wrappers import MLXTensorWrapper, TorchTensorWrapper
        
        logger.info(f"Erstelle Tensor aus Daten mit Shape {getattr(data, 'shape', None)}")
        
        # Verwende MLX für Apple Silicon, falls verfügbar
        if self.use_mlx and self.mlx_backend is not None:
            try:
                # Konvertiere zu MLX-Array, falls nötig
                if isinstance(data, (np.ndarray, list, tuple)):
                    import mlx.core as mx
                    mlx_tensor = mx.array(data, dtype=self._get_mlx_dtype(dtype))
                elif hasattr(data, '__array__'):  # PyTorch-Tensor oder ähnliches
                    import mlx.core as mx
                    mlx_tensor = mx.array(data.detach().cpu().numpy() if hasattr(data, 'detach') else data.__array__(), 
                                         dtype=self._get_mlx_dtype(dtype))
                else:  # Bereits MLX-Array
                    mlx_tensor = data
                
                logger.info(f"MLX-Tensor erstellt mit Shape {mlx_tensor.shape}")
                return MLXTensorWrapper(mlx_tensor)
            except Exception as e:
                logger.warning(f"Fehler bei der Erstellung eines MLX-Tensors: {e}")
                # Fallback auf PyTorch
        
        # PyTorch als Fallback
        try:
            # Konvertiere zu PyTorch-Tensor, falls nötig
            if isinstance(data, (np.ndarray, list, tuple)):
                tensor_data = torch.tensor(data, dtype=self._get_torch_dtype(dtype))
            elif hasattr(data, 'detach'):  # PyTorch-Tensor
                tensor_data = data
                if dtype is not None:
                    tensor_data = tensor_data.to(dtype=self._get_torch_dtype(dtype))
            else:  # Anderer Tensor-Typ (z.B. MLX)
                if hasattr(data, '__array__'):
                    tensor_data = torch.tensor(data.__array__(), dtype=self._get_torch_dtype(dtype))
                else:
                    tensor_data = torch.tensor(np.array(data), dtype=self._get_torch_dtype(dtype))
            
            # Verschiebe auf das richtige Gerät
            target_device = device or self.device
            tensor_data = tensor_data.to(device=target_device)
            
            logger.info(f"PyTorch-Tensor erstellt mit Shape {tensor_data.shape} auf Gerät {target_device}")
            return TorchTensorWrapper(tensor_data)
        except Exception as e:
            logger.error(f"Fehler bei der Erstellung eines PyTorch-Tensors: {e}")
            raise
    
    def _get_torch_dtype(self, dtype):
        """Konvertiert einen dtype-String in einen PyTorch-dtype"""
        if dtype is None:
            return torch.float32
        
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float64': torch.float64,
            'int32': torch.int32,
            'int64': torch.int64,
            'uint8': torch.uint8,
            'int8': torch.int8,
        }
        
        if isinstance(dtype, str):
            return dtype_map.get(dtype.lower(), torch.float32)
        return dtype  # Falls bereits ein torch.dtype Objekt
    
    def _get_mlx_dtype(self, dtype):
        """Konvertiert einen dtype-String in einen MLX-dtype"""
        if dtype is None:
            return None  # MLX-Standardtyp verwenden
        
        if not HAS_MLX:
            return None
            
        import mlx.core as mx
        
        dtype_map = {
            'float32': mx.float32,
            'float16': mx.float16,
            'bfloat16': mx.bfloat16,
            'int32': mx.int32,
            'int64': mx.int64,
            'uint8': mx.uint8,
            'int8': mx.int8,
        }
        
        if isinstance(dtype, str):
            return dtype_map.get(dtype.lower(), None)
        return dtype  # Falls bereits ein mlx.core.dtype Objekt
