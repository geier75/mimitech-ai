#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Tensor-Factory

Dieses Modul implementiert eine Factory-Klasse für die Erstellung von Tensoren
mit verschiedenen Backends (MLX, PyTorch, NumPy).

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, Type

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.Ultimate.TMathematics.TensorFactory")

# Importiere Tensor-Klassen
from .tensor import MISOTensor

# Lazy-Import für Tensor-Implementierungen
_tensor_implementations = {}
_default_backend = None


class TensorFactory:
    """
    Factory-Klasse für die Erstellung von Tensoren mit verschiedenen Backends
    """
    
    @staticmethod
    def get_available_backends() -> List[str]:
        """
        Gibt eine Liste der verfügbaren Backends zurück
        
        Returns:
            Liste der verfügbaren Backends
        """
        # Lazy-Import der Tensor-Implementierungen
        TensorFactory._ensure_implementations_loaded()
        return list(_tensor_implementations.keys())
    
    @staticmethod
    def set_default_backend(backend: str) -> None:
        """
        Setzt das Standard-Backend
        
        Args:
            backend: Name des Backends
        """
        global _default_backend
        
        # Lazy-Import der Tensor-Implementierungen
        TensorFactory._ensure_implementations_loaded()
        
        if backend not in _tensor_implementations:
            available = ", ".join(_tensor_implementations.keys())
            raise ValueError(f"Backend '{backend}' nicht verfügbar. Verfügbare Backends: {available}")
        
        _default_backend = backend
        logger.info(f"Standard-Backend auf '{backend}' gesetzt")
    
    @staticmethod
    def get_default_backend() -> str:
        """
        Gibt das Standard-Backend zurück
        
        Returns:
            Name des Standard-Backends
        """
        # Lazy-Import der Tensor-Implementierungen
        TensorFactory._ensure_implementations_loaded()
        
        global _default_backend
        return _default_backend
    
    @staticmethod
    def create_tensor(data, dtype=None, backend: Optional[str] = None) -> MISOTensor:
        """
        Erstellt einen Tensor mit dem angegebenen Backend
        
        Args:
            data: Daten für den Tensor
            dtype: Datentyp für den Tensor
            backend: Name des Backends (falls None, wird das Standard-Backend verwendet)
            
        Returns:
            Tensor mit dem angegebenen Backend
        """
        # Lazy-Import der Tensor-Implementierungen
        TensorFactory._ensure_implementations_loaded()
        
        global _default_backend
        
        # Verwende Standard-Backend, wenn keines angegeben ist
        if backend is None:
            backend = _default_backend
        
        # Prüfe, ob das Backend verfügbar ist
        if backend not in _tensor_implementations:
            available = ", ".join(_tensor_implementations.keys())
            raise ValueError(f"Backend '{backend}' nicht verfügbar. Verfügbare Backends: {available}")
        
        # Erstelle Tensor mit dem angegebenen Backend
        tensor_class = _tensor_implementations[backend]
        return tensor_class(data, dtype)
    
    @staticmethod
    def _ensure_implementations_loaded() -> None:
        """
        Stellt sicher, dass die Tensor-Implementierungen geladen sind
        """
        global _tensor_implementations, _default_backend
        
        # Wenn die Implementierungen bereits geladen sind, nichts tun
        if _tensor_implementations:
            return
        
        # Lade MLX-Implementierung, wenn verfügbar
        try:
            from .tensor_mlx import MLXTensor
            _tensor_implementations["mlx"] = MLXTensor
            logger.info("MLX-Tensor-Implementierung geladen")
            
            # Setze MLX als Standard-Backend, wenn verfügbar
            if _default_backend is None:
                _default_backend = "mlx"
        except ImportError:
            logger.warning("MLX-Tensor-Implementierung nicht verfügbar")
        
        # Lade PyTorch-Implementierung, wenn verfügbar
        try:
            from .tensor_torch import TorchTensor
            _tensor_implementations["torch"] = TorchTensor
            logger.info("PyTorch-Tensor-Implementierung geladen")
            
            # Setze PyTorch als Standard-Backend, wenn kein anderes verfügbar ist
            if _default_backend is None:
                _default_backend = "torch"
        except ImportError:
            logger.warning("PyTorch-Tensor-Implementierung nicht verfügbar")
        
        # Lade NumPy-Implementierung, wenn verfügbar (Fallback)
        try:
            from .tensor_numpy import NumPyTensor
            _tensor_implementations["numpy"] = NumPyTensor
            logger.info("NumPy-Tensor-Implementierung geladen")
            
            # Setze NumPy als Standard-Backend, wenn kein anderes verfügbar ist
            if _default_backend is None:
                _default_backend = "numpy"
        except ImportError:
            logger.warning("NumPy-Tensor-Implementierung nicht verfügbar")
        
        # Wenn keine Implementierung verfügbar ist, werfe einen Fehler
        if not _tensor_implementations:
            raise ImportError("Keine Tensor-Implementierung verfügbar")
        
        logger.info(f"Standard-Backend: {_default_backend}")


# Funktionen für einfachen Zugriff
def tensor(data, dtype=None, backend: Optional[str] = None) -> MISOTensor:
    """
    Erstellt einen Tensor mit dem angegebenen Backend
    
    Args:
        data: Daten für den Tensor
        dtype: Datentyp für den Tensor
        backend: Name des Backends (falls None, wird das Standard-Backend verwendet)
        
    Returns:
        Tensor mit dem angegebenen Backend
    """
    return TensorFactory.create_tensor(data, dtype, backend)


def set_default_backend(backend: str) -> None:
    """
    Setzt das Standard-Backend
    
    Args:
        backend: Name des Backends
    """
    TensorFactory.set_default_backend(backend)


def get_default_backend() -> str:
    """
    Gibt das Standard-Backend zurück
    
    Returns:
        Name des Standard-Backends
    """
    return TensorFactory.get_default_backend()


def get_available_backends() -> List[str]:
    """
    Gibt eine Liste der verfügbaren Backends zurück
    
    Returns:
        Liste der verfügbaren Backends
    """
    return TensorFactory.get_available_backends()
