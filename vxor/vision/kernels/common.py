#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION Kernel Common-Modul

Dieses Modul enthält die gemeinsame Infrastruktur für alle Kernel-Implementierungen,
einschließlich Registry, Benchmark-Tools und Hilfsfunktionen.
"""

import time
import enum
import logging
import functools
import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Union, Optional
import hashlib
import json
from dataclasses import dataclass

# Logger konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VX-VISION.kernels")

class KernelOperation(enum.Enum):
    """Unterstützte Bildverarbeitungsoperationen."""
    RESIZE = "resize"
    BLUR = "blur"
    SHARPEN = "sharpen"
    NORMALIZE = "normalize"
    DENOISE = "denoise"
    EDGE_DETECTION = "edge_detection"
    COLOR_CONVERT = "color_convert"
    ROTATE = "rotate"
    FLIP = "flip"
    CONTRAST = "contrast"
    BRIGHTNESS = "brightness"
    SATURATION = "saturation"
    THRESHOLD = "threshold"
    CROP = "crop"
    PAD = "pad"

class KernelType(enum.Enum):
    """Unterstützte Backend-Typen für Kernel-Implementierungen."""
    MLX = "mlx"
    TORCH = "torch"
    NUMPY = "numpy"

@dataclass
class KernelPerformanceData:
    """Speichert Performance-Daten für Kernel-Operationen."""
    operation: str
    backend: str
    avg_execution_time: float
    total_calls: int = 0
    total_execution_time: float = 0.0
    timestamp: float = 0.0
    
    def update(self, execution_time: float):
        """Aktualisiert die Performance-Daten mit einer neuen Messung."""
        self.total_calls += 1
        self.total_execution_time += execution_time
        self.avg_execution_time = self.total_execution_time / self.total_calls
        self.timestamp = time.time()

class KernelRegistry:
    """Registriert und verwaltet verfügbare Kernel-Implementierungen."""
    
    def __init__(self):
        """Initialisiert die Kernel-Registry."""
        self._kernels = {}
        self._performance_data = {}
        
    def register(self, operation: KernelOperation, kernel_type: KernelType, func: Callable):
        """
        Registriert eine Kernelfunktion.
        
        Args:
            operation: Art der Operation (z.B. RESIZE, BLUR)
            kernel_type: Backend-Typ (z.B. MLX, TORCH, NUMPY)
            func: Die zu registrierende Kernelfunktion
        """
        key = (operation.value, kernel_type.value)
        self._kernels[key] = func
        self._performance_data[key] = KernelPerformanceData(
            operation=operation.value,
            backend=kernel_type.value,
            avg_execution_time=float('inf')
        )
        logger.debug(f"Kernel registriert: {operation.value} - {kernel_type.value}")
        
    def get(self, operation: Union[KernelOperation, str], kernel_type: Union[KernelType, str]) -> Optional[Callable]:
        """
        Holt eine registrierte Kernelfunktion.
        
        Args:
            operation: Art der Operation (z.B. RESIZE, BLUR)
            kernel_type: Backend-Typ (z.B. MLX, TORCH, NUMPY)
            
        Returns:
            Die Kernelfunktion oder None, wenn nicht gefunden
        """
        # Konvertiere Enums zu Strings, falls nötig
        op_str = operation.value if isinstance(operation, KernelOperation) else operation
        kt_str = kernel_type.value if isinstance(kernel_type, KernelType) else kernel_type
        
        key = (op_str, kt_str)
        return self._kernels.get(key)
    
    def get_fastest(self, operation: Union[KernelOperation, str]) -> Tuple[str, Callable]:
        """
        Gibt die schnellste verfügbare Implementation für die angegebene Operation zurück.
        
        Args:
            operation: Art der Operation (z.B. RESIZE, BLUR)
            
        Returns:
            Tuple mit Backend-Typ und Kernelfunktion
        """
        op_str = operation.value if isinstance(operation, KernelOperation) else operation
        
        fastest_backend = None
        fastest_time = float('inf')
        
        # Alle verfügbaren Backends für diese Operation durchsuchen
        for (op, backend), perf_data in self._performance_data.items():
            if op == op_str and perf_data.total_calls > 0 and perf_data.avg_execution_time < fastest_time:
                fastest_time = perf_data.avg_execution_time
                fastest_backend = backend
                
        if fastest_backend:
            return fastest_backend, self.get(op_str, fastest_backend)
        else:
            # Wenn keine Performance-Daten vorhanden sind, Prioritätsreihenfolge verwenden
            for backend in [KernelType.MLX.value, KernelType.TORCH.value, KernelType.NUMPY.value]:
                kernel = self.get(op_str, backend)
                if kernel:
                    return backend, kernel
                    
            # Fallback zu NumPy, wenn verfügbar
            kernel = self.get(op_str, KernelType.NUMPY.value)
            if kernel:
                return KernelType.NUMPY.value, kernel
                
        raise KeyError(f"Keine Implementation für Operation {op_str} gefunden")
    
    def list_operations(self) -> List[str]:
        """Listet alle verfügbaren Operationen auf."""
        return list(set(op for op, _ in self._kernels.keys()))
        
    def list_backends(self, operation: Union[KernelOperation, str]) -> List[str]:
        """
        Listet alle verfügbaren Backends für eine Operation auf.
        
        Args:
            operation: Art der Operation
            
        Returns:
            Liste von Backend-Namen
        """
        op_str = operation.value if isinstance(operation, KernelOperation) else operation
        return [backend for (op, backend) in self._kernels.keys() if op == op_str]
        
    def record_performance(self, operation: str, kernel_type: str, execution_time: float):
        """
        Zeichnet Performance-Daten für eine Kernelfunktion auf.
        
        Args:
            operation: Art der Operation
            kernel_type: Backend-Typ
            execution_time: Ausführungszeit in Sekunden
        """
        key = (operation, kernel_type)
        if key in self._performance_data:
            self._performance_data[key].update(execution_time)
        else:
            self._performance_data[key] = KernelPerformanceData(
                operation=operation,
                backend=kernel_type,
                avg_execution_time=execution_time,
                total_calls=1,
                total_execution_time=execution_time,
                timestamp=time.time()
            )
        
    def get_performance_data(self) -> Dict[Tuple[str, str], KernelPerformanceData]:
        """Gibt alle gesammelten Performance-Daten zurück."""
        return self._performance_data

# Globale Registry-Instanz
kernel_registry = KernelRegistry()

def register_kernel(operation: KernelOperation, kernel_type: KernelType):
    """
    Decorator zum Registrieren einer Kernelfunktion.
    
    Args:
        operation: Art der Operation
        kernel_type: Backend-Typ
    """
    def decorator(func):
        kernel_registry.register(operation, kernel_type, func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        return wrapper
    return decorator

def benchmark_kernel(operation: Union[KernelOperation, str], 
                     kernel_type: Union[KernelType, str], 
                     images: List[np.ndarray], 
                     **kwargs) -> Tuple[List[np.ndarray], float]:
    """
    Führt ein Benchmark für die angegebene Kernelfunktion durch.
    
    Args:
        operation: Art der Operation
        kernel_type: Backend-Typ
        images: Liste von Bildern für den Benchmark
        **kwargs: Parameter für die Kernelfunktion
        
    Returns:
        Tuple mit verarbeiteten Bildern und Ausführungszeit
    """
    # Konvertiere Enums zu Strings, falls nötig
    op_str = operation.value if isinstance(operation, KernelOperation) else operation
    kt_str = kernel_type.value if isinstance(kernel_type, KernelType) else kernel_type
    
    kernel_func = kernel_registry.get(op_str, kt_str)
    if kernel_func is None:
        raise ValueError(f"Kein Kernel gefunden für {op_str} und {kt_str}")
    
    start_time = time.time()
    result = kernel_func(images, **kwargs)
    execution_time = time.time() - start_time
    
    # Performance-Daten aufzeichnen
    kernel_registry.record_performance(op_str, kt_str, execution_time)
    
    logger.debug(f"Benchmark für {op_str} mit {kt_str}: {execution_time:.6f}s")
    
    return result, execution_time

def get_best_kernel(operation: Union[KernelOperation, str], 
                   image_shape: Optional[Tuple[int, int, int]] = None) -> Tuple[str, Callable]:
    """
    Bestimmt den besten Kernel für die angegebene Operation basierend auf Performance-Daten.
    
    Args:
        operation: Art der Operation
        image_shape: Optional, Form des Bildes für spezifischere Auswahl
        
    Returns:
        Tuple mit Backend-Typ und Kernelfunktion
    """
    return kernel_registry.get_fastest(operation)

def list_available_operations() -> List[str]:
    """Gibt eine Liste aller verfügbaren Operationen zurück."""
    return kernel_registry.list_operations()
    
def list_available_backends(operation: Union[KernelOperation, str]) -> List[str]:
    """
    Gibt eine Liste aller verfügbaren Backends für die angegebene Operation zurück.
    
    Args:
        operation: Art der Operation
        
    Returns:
        Liste von Backend-Namen
    """
    return kernel_registry.list_backends(operation)

def generate_deterministic_hash(data: Any) -> str:
    """
    Generiert einen deterministischen Hash für beliebige Daten.
    
    Args:
        data: Die zu hashenden Daten
        
    Returns:
        String mit dem deterministischen Hash
    """
    if isinstance(data, np.ndarray):
        # NumPy-Arrays speziell behandeln
        return hashlib.md5(data.tobytes()).hexdigest()
    elif isinstance(data, (list, tuple)) and all(isinstance(x, np.ndarray) for x in data):
        # Listen von NumPy-Arrays
        combined_hash = hashlib.md5()
        for arr in data:
            combined_hash.update(arr.tobytes())
        return combined_hash.hexdigest()
    else:
        # Andere Datentypen
        try:
            serialized = json.dumps(data, sort_keys=True).encode('utf-8')
            return hashlib.md5(serialized).hexdigest()
        except (TypeError, ValueError):
            # Fallback für nicht JSON-serialisierbare Objekte
            return hashlib.md5(str(data).encode('utf-8')).hexdigest()
