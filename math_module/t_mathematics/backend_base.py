#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate: math_module.t_mathematics.backend_base
Definiert die Basis-Schnittstelle für alle Backend-Implementierungen der T-Mathematics Engine.

Diese Klasse stellt sicher, dass alle Backend-Implementierungen eine konsistente API bieten
und ermöglicht die dynamische Registrierung und Auswahl von Backends basierend auf
Hardware-Verfügbarkeit und Leistungsanforderungen.
"""

import abc
import logging
import inspect
import typing
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Konfiguriere Logging
logger = logging.getLogger(__name__)

class PrecisionType(str, Enum):
    """Unterstützte Präzisionstypen für numerische Operationen."""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BFLOAT16 = "bfloat16"
    
    @classmethod
    def from_string(cls, precision_str: str) -> "PrecisionType":
        """Konvertiert einen String in einen PrecisionType-Wert."""
        try:
            return cls(precision_str.lower())
        except ValueError:
            logger.warning(f"Unbekannter Präzisionstyp: {precision_str}. Verwende float32.")
            return cls.FLOAT32

@dataclass
class BackendCapabilities:
    """Definiert die Fähigkeiten eines Backends."""
    name: str
    supports_jit: bool = False
    supports_gpu: bool = False
    supports_ane: bool = False  # Apple Neural Engine
    supports_distributed: bool = False
    supported_precisions: List[PrecisionType] = None
    performance_rank: int = 0  # Höherer Wert = höhere Leistung
    
    def __post_init__(self):
        if self.supported_precisions is None:
            self.supported_precisions = [PrecisionType.FLOAT32]

class BackendRegistry:
    """Registry für verfügbare Backend-Implementierungen."""
    _backends: Dict[str, Tuple[type, BackendCapabilities]] = {}
    
    @classmethod
    def register(cls, name: str, backend_class: type, capabilities: BackendCapabilities) -> None:
        """Registriert ein Backend mit seinen Fähigkeiten."""
        if name in cls._backends:
            logger.warning(f"Backend '{name}' wird überschrieben.")
        cls._backends[name] = (backend_class, capabilities)
        logger.info(f"Backend '{name}' registriert mit Leistungsrang {capabilities.performance_rank}.")
    
    @classmethod
    def get_backend(cls, name: str):
        """Gibt ein registriertes Backend anhand des Namens zurück.
        
        Args:
            name: Name des zu holenden Backends
            
        Returns:
            Tuple aus (BackendClass, Capabilities) oder (None, None) wenn nicht gefunden
        """
        # Expliziter Fallback auf None, None wenn das Backend nicht existiert
        if name not in cls._backends:
            return None, None
            
        return cls._backends.get(name, (None, None))
    
    @classmethod
    def get_all_backends(cls) -> List[Tuple[str, type, BackendCapabilities]]:
        """Gibt alle registrierten Backends zurück."""
        return [(name, cls_cap[0], cls_cap[1]) for name, cls_cap in cls._backends.items()]
    
    @classmethod
    def get_best_backend(cls, 
                         required_capabilities: Optional[List[str]] = None, 
                         preferred_precision: Optional[PrecisionType] = None) -> Optional[str]:
        """
        Wählt das beste verfügbare Backend basierend auf Anforderungen und Leistung aus.
        
        Args:
            required_capabilities: Liste von erforderlichen Fähigkeiten (jit, gpu, ane, distributed)
            preferred_precision: Bevorzugte Präzision
            
        Returns:
            Name des besten Backends oder None, wenn keines verfügbar ist
        """
        if not cls._backends:
            logger.warning("Keine Backends registriert.")
            return None
        
        # Standardwerte
        if required_capabilities is None:
            required_capabilities = []
        if preferred_precision is None:
            preferred_precision = PrecisionType.FLOAT32
        
        # Filtere Backends basierend auf Anforderungen
        candidates = []
        for name, (_, capabilities) in cls._backends.items():
            # Prüfe erforderliche Fähigkeiten
            meets_requirements = True
            for req in required_capabilities:
                has_capability = getattr(capabilities, f"supports_{req.lower()}", False)
                if not has_capability:
                    meets_requirements = False
                    break
            
            # Prüfe Präzisionsunterstützung
            supports_precision = (preferred_precision in capabilities.supported_precisions)
            
            if meets_requirements and supports_precision:
                candidates.append((name, capabilities.performance_rank))
        
        if not candidates:
            logger.warning("Kein Backend erfüllt die Anforderungen.")
            return None
        
        # Wähle das Backend mit dem höchsten Leistungsrang
        return max(candidates, key=lambda x: x[1])[0]

class BackendBase(abc.ABC):
    """Basisklasse für alle Backend-Implementierungen."""
    
    def __init__(self, precision: Union[str, PrecisionType] = PrecisionType.FLOAT32):
        """
        Initialisiert das Backend mit der angegebenen Präzision.
        
        Args:
            precision: Zu verwendende Präzision (float16, float32, float64, bfloat16)
        """
        if isinstance(precision, str):
            self.precision = PrecisionType.from_string(precision)
        else:
            self.precision = precision
        
        self.name = self.__class__.__name__
        logger.info(f"Initialisiere {self.name} mit Präzision {self.precision}")
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Prüft, ob dieses Backend auf dem aktuellen System verfügbar ist."""
        pass
    
    @abc.abstractmethod
    def create_tensor(self, data: Any, dtype: Optional[Any] = None) -> Any:
        """
        Erstellt einen Tensor aus den angegebenen Daten.
        
        Args:
            data: Die Daten, aus denen der Tensor erstellt werden soll (Array, Liste, etc.)
            dtype: Datentyp für den Tensor (verwendet Backend-spezifische Typen)
            
        Returns:
            Ein Backend-spezifischer Tensor
        """
        pass
    
    @abc.abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das aktuelle Gerät zurück.
        
        Returns:
            Ein Dictionary mit Geräteinformationen
        """
        pass
    
    # === Basisoperationen ===
    
    @abc.abstractmethod
    def add(self, a: Any, b: Any) -> Any:
        """Addiert zwei Tensoren."""
        pass
    
    @abc.abstractmethod
    def subtract(self, a: Any, b: Any) -> Any:
        """Subtrahiert einen Tensor von einem anderen."""
        pass
    
    @abc.abstractmethod
    def multiply(self, a: Any, b: Any) -> Any:
        """Multipliziert zwei Tensoren (elementweise)."""
        pass
    
    @abc.abstractmethod
    def divide(self, a: Any, b: Any) -> Any:
        """Dividiert einen Tensor durch einen anderen (elementweise)."""
        pass
    
    @abc.abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Führt eine Matrix-Multiplikation durch."""
        pass
    
    # === Formoperationen ===
    
    @abc.abstractmethod
    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any:
        """Ändert die Form eines Tensors."""
        pass
    
    @abc.abstractmethod
    def transpose(self, x: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        """Transponiert einen Tensor entlang der angegebenen Achsen."""
        pass
    
    # === Mathematische Funktionen ===
    
    @abc.abstractmethod
    def exp(self, x: Any) -> Any:
        """Berechnet die Exponentialfunktion für jeden Wert im Tensor."""
        pass
    
    @abc.abstractmethod
    def log(self, x: Any) -> Any:
        """Berechnet den natürlichen Logarithmus für jeden Wert im Tensor."""
        pass
    
    @abc.abstractmethod
    def sin(self, x: Any) -> Any:
        """Berechnet den Sinus für jeden Wert im Tensor."""
        pass
    
    @abc.abstractmethod
    def cos(self, x: Any) -> Any:
        """Berechnet den Kosinus für jeden Wert im Tensor."""
        pass
    
    # === Reduktionsoperationen ===
    
    @abc.abstractmethod
    def sum(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Any:
        """Berechnet die Summe entlang der angegebenen Achsen."""
        pass
    
    @abc.abstractmethod
    def mean(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Any:
        """Berechnet den Mittelwert entlang der angegebenen Achsen."""
        pass
    
    # === Lineare Algebra ===
    
    @abc.abstractmethod
    def svd(self, x: Any, full_matrices: bool = True) -> Tuple[Any, Any, Any]:
        """
        Berechnet die Singulärwertzerlegung einer Matrix.
        
        Returns:
            Tupel (U, S, V), wobei U und V orthogonale Matrizen und S die Singulärwerte sind
        """
        pass
    
    # === Dienstprogramme ===
    
    @abc.abstractmethod
    def to_numpy(self, x: Any) -> Any:
        """Konvertiert einen Backend-Tensor zu einem NumPy-Array."""
        pass
    
    @abc.abstractmethod
    def from_numpy(self, x: Any) -> Any:
        """Konvertiert ein NumPy-Array in einen Backend-Tensor."""
        pass
    
    # === JIT-Kompilierung ===
    
    def supports_jit(self) -> bool:
        """Gibt an, ob dieses Backend JIT-Kompilierung unterstützt."""
        return False
    
    def jit(self, fn: Callable) -> Callable:
        """
        JIT-kompiliert eine Funktion, wenn unterstützt.
        
        Args:
            fn: Zu kompilierende Funktion
            
        Returns:
            Kompilierte Funktion oder ursprüngliche Funktion, wenn nicht unterstützt
        """
        logger.warning(f"JIT-Kompilierung nicht unterstützt von {self.name}.")
        return fn
