#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE MLX Adapter

Dieses Modul implementiert den Adapter zwischen M-CODE und der T-Mathematics Engine mit MLX-Backend.
Es ermöglicht die effiziente Ausführung von Tensor-Operationen auf Apple Silicon mit Neural Engine.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import sys
import time
import importlib.util
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from pathlib import Path

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.mlx_adapter")

# Prüfen, ob MLX verfügbar ist
try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
    logger.info("MLX-Backend erfolgreich geladen. Version: %s", mlx.__version__)
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX-Backend nicht verfügbar. Verwende NumPy-Fallback.")


class MLXAdapterError(Exception):
    """Fehler im MLX-Adapter"""
    pass


class MLXAdapter:
    """Adapter zwischen M-CODE und MLX-Backend"""
    
    def __init__(self, use_ane: bool = True, fallback_on_error: bool = True):
        """
        Initialisiert einen neuen MLX-Adapter.
        
        Args:
            use_ane: Neural Engine aktivieren (falls verfügbar)
            fallback_on_error: Bei Fehlern auf NumPy-Fallback umschalten
        """
        self.use_ane = use_ane and MLX_AVAILABLE
        self.fallback_on_error = fallback_on_error
        self.jit_cache = {}
        self.t_math_engine = None
        
        # Initialisiere MLX-Backend, falls verfügbar
        if MLX_AVAILABLE:
            if self.use_ane:
                # Aktiviere ANE für MLX, wenn verfügbar
                mx.set_default_device(mx.gpu if mx.gpu.is_available() else mx.cpu)
                logger.info(f"MLX-Backend verwendet Gerät: {mx.default_device()}")
            else:
                mx.set_default_device(mx.cpu)
                logger.info("MLX-Backend verwendet CPU-Gerät")
                
            # ANE-Status überprüfen
            self.ane_available = mx.gpu.is_available()
            if self.ane_available:
                logger.info("Apple Neural Engine (ANE) ist verfügbar")
            else:
                logger.info("Apple Neural Engine (ANE) ist nicht verfügbar")
                
            # Optimierungen aktivieren
            self._enable_optimizations()
        else:
            logger.warning("MLX nicht verfügbar. Verwende NumPy-Fallback für alle Operationen.")
            self.ane_available = False
    
    def _enable_optimizations(self):
        """Aktiviert MLX-Optimierungen und JIT-Kompilierung"""
        if not MLX_AVAILABLE:
            return
            
        # Aktiviere MLX-Optimierungen
        mlx.set_global_fused_attn(True)
        
        # Kompilierungsoptionen
        self.compilation_options = {
            "dynamic_batch_size": True,
            "optimization_level": 3,
            "use_experimental_features": False,
        }
        
        logger.info("MLX-Optimierungen aktiviert")
    
    def _load_t_mathematics_engine(self):
        """Lädt die T-Mathematics Engine"""
        if self.t_math_engine is not None:
            return
            
        try:
            # Pfad zur T-Mathematics Engine
            math_module_path = str(Path(__file__).parent.parent.parent.parent / "math_module" / "t_mathematics")
            sys.path.append(math_module_path)
            
            # Importiere die optimierte T-Mathematics Engine
            from t_mathematics.engine_optimized import TMathEngine
            
            # Initialisiere Engine mit MLX-Backend
            self.t_math_engine = TMathEngine(preferred_backend="mlx")
            logger.info("T-Mathematics Engine erfolgreich geladen")
            
            # Konfiguriere JIT und ANE
            self.t_math_engine.configure(
                use_jit=True,
                use_ane=self.use_ane,
                fallback_on_error=self.fallback_on_error
            )
        except Exception as e:
            logger.error(f"Fehler beim Laden der T-Mathematics Engine: {e}")
            if not self.fallback_on_error:
                raise MLXAdapterError(f"Konnte T-Mathematics Engine nicht laden: {e}")
    
    def is_available(self) -> bool:
        """
        Prüft, ob MLX verfügbar ist.
        
        Returns:
            True, wenn MLX verfügbar ist, sonst False
        """
        return MLX_AVAILABLE
    
    def supports_ane(self) -> bool:
        """
        Prüft, ob ANE unterstützt wird.
        
        Returns:
            True, wenn ANE verfügbar ist, sonst False
        """
        return MLX_AVAILABLE and self.ane_available
    
    def create_tensor(self, data, dtype=None) -> Any:
        """
        Erstellt einen MLX-Tensor.
        
        Args:
            data: Daten für den Tensor
            dtype: Datentyp (optional)
            
        Returns:
            MLX-Tensor oder NumPy-Array (Fallback)
        """
        if not MLX_AVAILABLE:
            return np.array(data, dtype=dtype)
            
        try:
            return mx.array(data, dtype=dtype)
        except Exception as e:
            logger.warning(f"Fehler bei der Tensor-Erstellung: {e}")
            if self.fallback_on_error:
                return np.array(data, dtype=dtype)
            raise MLXAdapterError(f"Konnte Tensor nicht erstellen: {e}")
    
    def to_numpy(self, tensor) -> np.ndarray:
        """
        Konvertiert einen Tensor zu einem NumPy-Array.
        
        Args:
            tensor: MLX-Tensor oder NumPy-Array
            
        Returns:
            NumPy-Array
        """
        if not MLX_AVAILABLE or isinstance(tensor, np.ndarray):
            return tensor
            
        try:
            return mx.array(tensor).item() if tensor.size == 1 else mx.array(tensor).numpy()
        except Exception as e:
            logger.warning(f"Fehler bei der Konvertierung zu NumPy: {e}")
            raise MLXAdapterError(f"Konnte Tensor nicht zu NumPy konvertieren: {e}")
    
    def jit_compile(self, func: Callable) -> Callable:
        """
        Wendet Just-In-Time Kompilierung auf eine Funktion an.
        
        Args:
            func: Zu kompilierende Funktion
            
        Returns:
            Kompilierte Funktion
        """
        if not MLX_AVAILABLE:
            return func
            
        try:
            # Prüfe, ob die Funktion bereits kompiliert wurde
            if func in self.jit_cache:
                return self.jit_cache[func]
                
            # Kompiliere die Funktion
            compiled_func = mx.compile(func)
            self.jit_cache[func] = compiled_func
            return compiled_func
        except Exception as e:
            logger.warning(f"Fehler bei der JIT-Kompilierung: {e}")
            if self.fallback_on_error:
                return func
            raise MLXAdapterError(f"Konnte Funktion nicht JIT-kompilieren: {e}")
    
    def execute_tensor_operation(self, operation: str, *args, **kwargs) -> Any:
        """
        Führt eine Tensor-Operation aus.
        
        Args:
            operation: Name der Operation
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Ergebnis der Operation
        """
        # Stelle sicher, dass T-Mathematics Engine geladen ist
        self._load_t_mathematics_engine()
        
        try:
            # Führe Operation mit T-Mathematics Engine aus
            return self.t_math_engine.compute(operation, *args, **kwargs)
        except Exception as e:
            logger.warning(f"Fehler bei der Tensor-Operation '{operation}': {e}")
            if self.fallback_on_error:
                # Fallback mit NumPy
                return self._fallback_tensor_operation(operation, *args, **kwargs)
            raise MLXAdapterError(f"Konnte Operation '{operation}' nicht ausführen: {e}")
    
    def _fallback_tensor_operation(self, operation: str, *args, **kwargs) -> Any:
        """
        Führt eine Tensor-Operation mit NumPy aus (Fallback).
        
        Args:
            operation: Name der Operation
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Ergebnis der Operation
        """
        logger.info(f"Verwende NumPy-Fallback für Operation '{operation}'")
        
        # Konvertiere alle Tensor-Argumente zu NumPy-Arrays
        np_args = [arg.numpy() if hasattr(arg, 'numpy') else arg for arg in args]
        np_kwargs = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in kwargs.items()}
        
        # Grundlegende NumPy-Operationen
        if operation == "add":
            return np.add(*np_args, **np_kwargs)
        elif operation == "subtract":
            return np.subtract(*np_args, **np_kwargs)
        elif operation == "multiply":
            return np.multiply(*np_args, **np_kwargs)
        elif operation == "divide":
            return np.divide(*np_args, **np_kwargs)
        elif operation == "matmul":
            return np.matmul(*np_args, **np_kwargs)
        elif operation == "transpose":
            return np.transpose(*np_args, **np_kwargs)
        elif operation == "sum":
            return np.sum(*np_args, **np_kwargs)
        elif operation == "mean":
            return np.mean(*np_args, **np_kwargs)
        elif operation == "std":
            return np.std(*np_args, **np_kwargs)
        elif operation == "max":
            return np.max(*np_args, **np_kwargs)
        elif operation == "min":
            return np.min(*np_args, **np_kwargs)
        elif operation == "argmax":
            return np.argmax(*np_args, **np_kwargs)
        elif operation == "argmin":
            return np.argmin(*np_args, **np_kwargs)
        elif operation == "concatenate":
            return np.concatenate(*np_args, **np_kwargs)
        elif operation == "reshape":
            return np.reshape(*np_args, **np_kwargs)
        elif operation == "clip":
            return np.clip(*np_args, **np_kwargs)
        elif operation == "abs":
            return np.abs(*np_args, **np_kwargs)
        elif operation == "exp":
            return np.exp(*np_args, **np_kwargs)
        elif operation == "log":
            return np.log(*np_args, **np_kwargs)
        elif operation == "pow":
            return np.power(*np_args, **np_kwargs)
        elif operation == "sqrt":
            return np.sqrt(*np_args, **np_kwargs)
        elif operation == "sin":
            return np.sin(*np_args, **np_kwargs)
        elif operation == "cos":
            return np.cos(*np_args, **np_kwargs)
        elif operation == "tan":
            return np.tan(*np_args, **np_kwargs)
        # Weitere Operationen hier hinzufügen
            
        raise MLXAdapterError(f"Operation '{operation}' wird im NumPy-Fallback nicht unterstützt")
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das aktuelle Gerät zurück.
        
        Returns:
            Wörterbuch mit Geräteinformationen
        """
        info = {
            "mlx_available": MLX_AVAILABLE,
            "ane_available": self.ane_available if MLX_AVAILABLE else False,
            "use_ane": self.use_ane,
            "fallback_on_error": self.fallback_on_error,
        }
        
        if MLX_AVAILABLE:
            info["default_device"] = str(mx.default_device())
            info["mlx_version"] = mlx.__version__
            
        return info


# Singleton-Instanz des Adapters
_mlx_adapter_instance = None

def get_mlx_adapter(use_ane: bool = True, fallback_on_error: bool = True) -> MLXAdapter:
    """
    Gibt eine Singleton-Instanz des MLX-Adapters zurück.
    
    Args:
        use_ane: Neural Engine aktivieren (falls verfügbar)
        fallback_on_error: Bei Fehlern auf NumPy-Fallback umschalten
        
    Returns:
        MLX-Adapter
    """
    global _mlx_adapter_instance
    
    if _mlx_adapter_instance is None:
        _mlx_adapter_instance = MLXAdapter(use_ane=use_ane, fallback_on_error=fallback_on_error)
        
    return _mlx_adapter_instance
