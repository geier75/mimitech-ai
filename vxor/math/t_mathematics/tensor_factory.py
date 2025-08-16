#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Engine - Tensor-Factory

Diese Datei implementiert eine Factory-Klasse für die Erstellung und Konvertierung
von Tensoren zwischen verschiedenen Backends (MLX, PyTorch).

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import logging
import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Type

from .tensor_interface import MISOTensorInterface
from .tensor_wrappers import MLXTensorWrapper, TorchTensorWrapper

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.t_mathematics.tensor_factory")

class TensorFactory:
    """
    Factory-Klasse für die Erstellung und Konvertierung von Tensoren.
    
    Diese Klasse ermöglicht die einfache Erstellung und Konvertierung von Tensoren
    zwischen verschiedenen Backends (MLX, PyTorch) und bietet damit eine zentrale
    Schnittstelle für die Tensor-Operation-Bridge zwischen VXOR-Komponenten.
    """
    
    # Mapping von Backend-Namen zu Wrapper-Klassen
    _BACKEND_CLASSES = {
        "mlx": MLXTensorWrapper,
        "torch": TorchTensorWrapper
    }
    
    # Cache für häufig verwendete Tensoren
    _tensor_cache = {}
    
    # Maximale Cache-Größe
    _MAX_CACHE_SIZE = 100
    
    @classmethod
    def create_tensor(cls, data: np.ndarray, backend: str = "mlx") -> MISOTensorInterface:
        """
        Erstellt einen Tensor mit dem angegebenen Backend.
        
        Args:
            data: NumPy-Array mit den Daten für den Tensor
            backend: Name des zu verwendenden Backends ('mlx' oder 'torch')
            
        Returns:
            Tensor mit dem angegebenen Backend
            
        Raises:
            ValueError: Wenn das angegebene Backend nicht unterstützt wird
        """
        if backend not in cls._BACKEND_CLASSES:
            supported = ", ".join(cls._BACKEND_CLASSES.keys())
            raise ValueError(f"Nicht unterstütztes Backend: {backend}. Unterstützt werden: {supported}")
        
        # Erstelle den Tensor mit dem angegebenen Backend
        wrapper_class = cls._BACKEND_CLASSES[backend]
        return wrapper_class.from_numpy(data)
    
    @classmethod
    def convert_tensor(cls, tensor: MISOTensorInterface, target_backend: str) -> MISOTensorInterface:
        """
        Konvertiert einen Tensor zu einem anderen Backend.
        
        Args:
            tensor: Zu konvertierender Tensor
            target_backend: Name des Ziel-Backends ('mlx' oder 'torch')
            
        Returns:
            Konvertierter Tensor mit dem Ziel-Backend
            
        Raises:
            ValueError: Wenn das Ziel-Backend nicht unterstützt wird
        """
        # Wenn der Tensor bereits das richtige Backend hat, gib ihn unverändert zurück
        if tensor.backend == target_backend:
            return tensor
        
        # Prüfe, ob das Ziel-Backend unterstützt wird
        if target_backend not in cls._BACKEND_CLASSES:
            supported = ", ".join(cls._BACKEND_CLASSES.keys())
            raise ValueError(f"Nicht unterstütztes Ziel-Backend: {target_backend}. Unterstützt werden: {supported}")
        
        # Cache-Schlüssel generieren
        cache_key = f"{id(tensor)}_{target_backend}"
        
        # Versuche, den konvertierten Tensor aus dem Cache zu holen
        if cache_key in cls._tensor_cache:
            logger.debug(f"Cache-Hit für Tensor-Konvertierung: {tensor.backend} -> {target_backend}")
            return cls._tensor_cache[cache_key]
        
        # Konvertiere den Tensor über NumPy
        numpy_data = tensor.to_numpy()
        
        # Erstelle einen neuen Tensor mit dem Ziel-Backend
        wrapper_class = cls._BACKEND_CLASSES[target_backend]
        converted_tensor = wrapper_class.from_numpy(numpy_data)
        
        # Speichere den konvertierten Tensor im Cache
        if len(cls._tensor_cache) >= cls._MAX_CACHE_SIZE:
            # Entferne ein zufälliges Element aus dem Cache, wenn er voll ist
            cls._tensor_cache.pop(next(iter(cls._tensor_cache)))
        
        cls._tensor_cache[cache_key] = converted_tensor
        logger.debug(f"Tensor konvertiert: {tensor.backend} -> {target_backend}")
        
        return converted_tensor
    
    @classmethod
    def serialize_tensor(cls, tensor: MISOTensorInterface) -> Dict[str, Any]:
        """
        Serialisiert einen Tensor für die Übertragung zwischen Komponenten.
        
        Args:
            tensor: Zu serialisierender Tensor
            
        Returns:
            Dictionary mit serialisierten Daten des Tensors
        """
        return tensor.serialize()
    
    @classmethod
    def deserialize_tensor(cls, data: Dict[str, Any], target_backend: Optional[str] = None) -> MISOTensorInterface:
        """
        Deserialisiert einen Tensor aus einem Dictionary.
        
        Args:
            data: Dictionary mit serialisierten Daten des Tensors
            target_backend: Optionales Ziel-Backend für den deserialisierten Tensor
            
        Returns:
            Deserialisierter Tensor
            
        Raises:
            ValueError: Wenn das Backend des serialisierten Tensors oder das Ziel-Backend nicht unterstützt wird
        """
        # Ermittle das Backend des serialisierten Tensors
        backend = data.get("backend")
        if backend not in cls._BACKEND_CLASSES:
            supported = ", ".join(cls._BACKEND_CLASSES.keys())
            raise ValueError(f"Nicht unterstütztes Backend im serialisierten Tensor: {backend}. Unterstützt werden: {supported}")
        
        # Deserialisiere den Tensor mit dem ursprünglichen Backend
        wrapper_class = cls._BACKEND_CLASSES[backend]
        tensor = wrapper_class.deserialize(data)
        
        # Konvertiere den Tensor zum Ziel-Backend, falls angegeben
        if target_backend is not None and target_backend != backend:
            tensor = cls.convert_tensor(tensor, target_backend)
        
        return tensor
    
    @classmethod
    def get_preferred_backend(cls) -> str:
        """
        Ermittelt das bevorzugte Backend basierend auf der Systemkonfiguration.
        
        Returns:
            Name des bevorzugten Backends ('mlx' oder 'torch')
        """
        # Prüfe, ob MLX verwendet werden soll
        use_mlx = os.environ.get("MISO_USE_MLX", "1").lower() in ("1", "true", "yes")
        
        # Prüfe, ob Apple Silicon vorhanden ist (für MLX)
        is_apple_silicon = False
        try:
            import platform
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                is_apple_silicon = True
        except Exception:
            pass
        
        # Prüfe, ob CUDA verfügbar ist (für PyTorch)
        has_cuda = False
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            pass
        
        # Prüfe, ob MPS verfügbar ist (für PyTorch auf Apple Silicon)
        has_mps = False
        try:
            import torch
            has_mps = hasattr(torch, "mps") and torch.mps.is_available()
        except Exception:
            pass
        
        # Entscheide basierend auf den verfügbaren Backends
        if use_mlx and is_apple_silicon:
            return "mlx"  # MLX auf Apple Silicon
        elif has_cuda:
            return "torch"  # PyTorch mit CUDA
        elif has_mps:
            return "torch"  # PyTorch mit MPS
        else:
            # Fallback: Wähle MLX, falls verfügbar, sonst PyTorch
            try:
                import mlx.core
                return "mlx"
            except ImportError:
                return "torch"

# Globale Instanz für einfachen Zugriff
tensor_factory = TensorFactory()
