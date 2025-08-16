#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate: math_module.t_mathematics.engine_optimized
Optimierte Hauptengine für T-Mathematics mit Backend-Registry-Integration.

Diese Engine bietet eine einheitliche API für alle mathematischen Operationen
und wählt automatisch das beste verfügbare Backend basierend auf den Anforderungen
und der verfügbaren Hardware.
"""

import os
import sys
import logging
import inspect
import functools
from enum import Enum
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Importiere Backend-Basis
from math_module.t_mathematics.backend_base import BackendBase, BackendRegistry, PrecisionType, BackendCapabilities

# Konfiguriere Logging
logger = logging.getLogger(__name__)

# Singleton-Instanz der Engine
_ENGINE_INSTANCE = None

class TMathEngine:
    """
    Optimierte Hauptengine für T-Mathematics.
    
    Diese Engine bietet:
    - Automatische Backend-Auswahl basierend auf Verfügbarkeit und Leistung
    - Einheitliche API für alle mathematischen Operationen
    - Leistungsoptimierungen für Apple Silicon
    - Transparente Fehlerbehandlung und Fallback-Mechanismen
    """
    
    def __init__(self, 
                 precision: Union[str, PrecisionType] = PrecisionType.FLOAT32,
                 backend: str = "auto",
                 optimize_for_ane: bool = True,
                 enable_jit: bool = True):
        """
        Initialisiert die T-Mathematics Engine.
        
        Args:
            precision: Zu verwendende Präzision (float16, float32, float64, bfloat16)
            backend: Zu verwendendes Backend ('auto', 'mlx', 'torch', 'numpy')
            optimize_for_ane: Optimierung für Apple Neural Engine aktivieren
            enable_jit: JIT-Kompilierung aktivieren
        """
        self.requested_precision = precision if isinstance(precision, PrecisionType) else PrecisionType.from_string(precision)
        self.requested_backend = backend
        self.optimize_for_ane = optimize_for_ane
        self.enable_jit = enable_jit
        
        # Cache für Backend-Instances
        self._backend_cache = {}
        
        # Lade alle verfügbaren Backends
        self._load_backends()
        
        # Initialisiere das Standard-Backend
        self._init_default_backend()
        
        logger.info(f"T-Mathematics Engine initialisiert mit Backend: {self.active_backend_name}")
    
    def _load_backends(self):
        """Lädt alle verfügbaren Backend-Module."""
        # Diese Funktion führt indirekt die Backend-Registrierung durch
        # Die Module registrieren sich selbst beim Import
        
        # Versuche, das MLX-Backend zu laden
        try:
            from mlx_backend.mlx_backend_impl import MLXBackendImpl
            logger.info("MLX-Backend geladen und registriert.")
        except ImportError:
            logger.info("MLX-Backend konnte nicht geladen werden.")
        
        # Versuche, das PyTorch-Backend zu laden
        try:
            # In einer vollständigen Implementierung würden wir hier auch das PyTorch-Backend laden
            pass
        except ImportError:
            logger.info("PyTorch-Backend konnte nicht geladen werden.")
        
        # Versuche, das NumPy-Backend zu laden
        try:
            # In einer vollständigen Implementierung würden wir hier auch das NumPy-Backend laden
            pass
        except ImportError:
            logger.info("NumPy-Backend konnte nicht geladen werden.")
        
        # Protokolliere verfügbare Backends
        available_backends = BackendRegistry.get_all_backends()
        if available_backends:
            logger.info(f"Verfügbare Backends: {', '.join(name for name, _, _ in available_backends)}")
        else:
            logger.warning("Keine Backends verfügbar. Verwende Fallback-Implementierung.")
    
    def _init_default_backend(self):
        """Initialisiert das Standard-Backend basierend auf den Anforderungen."""
        # Bestimme das zu verwendende Backend
        backend_name = self.requested_backend
        
        if backend_name == "auto":
            # Bestimme das beste Backend basierend auf Anforderungen
            required_capabilities = []
            if self.enable_jit:
                required_capabilities.append("jit")
            if self.optimize_for_ane:
                required_capabilities.append("ane")
            
            # Lasse das Registry das beste Backend auswählen
            backend_name = BackendRegistry.get_best_backend(
                required_capabilities=required_capabilities,
                preferred_precision=self.requested_precision
            )
            
            if backend_name is None:
                logger.warning("Kein passendes Backend gefunden. Implementiere Fallback-Strategie.")
                self._initialize_fallback_backend()
                return
        
        # Versuche, das Backend zu initialisieren
        try:
            self.active_backend_name = backend_name
            
            # Holen des Backend-Klassen und seiner Fähigkeiten
            backend_class, capabilities = BackendRegistry.get_backend(backend_name)
            
            if backend_class is None:
                logger.warning(f"Backend '{backend_name}' nicht gefunden. Implementiere Fallback-Strategie.")
                # Explicit Fallback-Strategie implementieren
                self._initialize_fallback_backend()
                return
            
            # Initialisiere das angeforderte Backend
            self.active_backend = backend_class(precision=self.requested_precision)
            logger.info(f"Backend '{backend_name}' erfolgreich initialisiert.")
            
            # Protokolliere Informationen über das aktive Backend
            if self.active_backend is not None:
                device_info = self.active_backend.get_device_info()
                logger.info(f"Aktives Backend: {backend_name}, Gerät: {device_info.get('device', 'unbekannt')}")
                logger.info(f"JIT-Unterstützung: {self.active_backend.supports_jit()}")
            else:
                # Bei Fehlschlag auf Fallback zurückgreifen
                logger.warning(f"Backend '{backend_name}' konnte nicht initialisiert werden. Verwende Fallback.")
                self._initialize_fallback_backend()
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des Backends '{backend_name}': {e}")
            # Bei Ausnahmen auf Fallback zurückgreifen
            self._initialize_fallback_backend()
    
    def _initialize_fallback_backend(self):
        """Initialisiert ein robustes Fallback-Backend basierend auf NumPy.
        
        Diese Methode wird aufgerufen, wenn das angeforderte spezialisierte Backend
        nicht verfügbar ist oder initialisiert werden kann. Sie implementiert ein
        einfaches aber zuverlässiges Backend, das auf NumPy basiert und garantiert
        funktioniert, unabhängig von der Hardwareumgebung.
        """
        # Importiere die Original-Engine als Fallback
        try:
            from math_module.t_mathematics.engine import Engine as FallbackEngine
            self.active_backend = FallbackEngine(precision=self.requested_precision)
            self.active_backend_name = "numpy_fallback"
            logger.info("Fallback-Backend erfolgreich initialisiert (Original-Engine mit NumPy).")
            return
        except ImportError as e:
            # Wenn sogar die Original-Engine nicht verfügbar ist, implementiere ein minimales NumPy-Backend
            logger.warning(f"Original-Engine nicht verfügbar: {e}. Implementiere absolutes Minimal-Fallback.")
        
        # Implementierung des absoluten Minimal-Fallbacks
        import numpy as np
        from math_module.t_mathematics.backend_base import BackendBase
        
        class MinimalFallbackBackend(BackendBase):
            """Absolut minimaler Fallback für kritische Fälle."""
            
            def __init__(self, precision):
                super().__init__(precision)
                self._dtype = np.float32
                if precision == PrecisionType.FLOAT16:
                    self._dtype = np.float16
                elif precision == PrecisionType.FLOAT64:
                    self._dtype = np.float64
            
            def is_available(self):
                return True
                
            def get_device_info(self):
                return {"device": "cpu", "precision": str(self.precision.value)}
                
            def supports_jit(self):
                return False
                
            def create_tensor(self, data, dtype=None):
                return np.array(data, dtype=dtype or self._dtype)
                
            # Minimale Operationen für Fallback
            def add(self, a, b): return np.add(a, b)
            def subtract(self, a, b): return np.subtract(a, b)
            def multiply(self, a, b): return np.multiply(a, b)
            def divide(self, a, b): return np.divide(a, b)
            def matmul(self, a, b): return np.matmul(a, b)
            def svd(self, x, full_matrices=False): return np.linalg.svd(x, full_matrices=full_matrices)
            def relu(self, x): return np.maximum(x, 0)
            def to_numpy(self, x): return np.array(x)
            def from_numpy(self, x): return x  # Bereits NumPy-Array
            def greater(self, a, b): return np.greater(a, b)
            
        self.active_backend = MinimalFallbackBackend(precision=self.requested_precision)
        self.active_backend_name = "minimal_numpy_fallback"
        logger.info("Minimal-Fallback-Backend initialisiert.")
    
    def get_backend(self, name: Optional[str] = None) -> BackendBase:
        """
        Gibt eine Instance des angegebenen Backends zurück.
        
        Args:
            name: Name des gewünschten Backends. Wenn None, wird das aktive Backend zurückgegeben.
            
        Returns:
            Eine Backend-Instance
        """
        if name is None:
            return self.active_backend
        
        # Prüfe, ob wir bereits eine Instance dieses Backends haben
        if name in self._backend_cache:
            return self._backend_cache[name]
        
        # Erstelle eine neue Instance
        backend_class, _ = BackendRegistry.get_backend(name)
        if backend_class is None:
            raise ValueError(f"Backend '{name}' nicht gefunden.")
        
        backend = backend_class(precision=self.requested_precision)
        self._backend_cache[name] = backend
        
        return backend
    
    def get_available_backends(self) -> List[str]:
        """Gibt eine Liste der verfügbaren Backend-Namen zurück."""
        return [name for name, _, _ in BackendRegistry.get_all_backends()]
    
    def get_active_backend_info(self) -> Dict[str, Any]:
        """Gibt Informationen zum aktiven Backend zurück."""
        if self.active_backend is None:
            return {
                "name": "none",
                "device": "cpu",
                "jit_enabled": False,
                "has_ane": False,
                "precision": str(self.requested_precision.value)
            }
        
        backend_info = {
            "name": self.active_backend_name,
            "jit_enabled": self.active_backend.supports_jit() and self.enable_jit,
            "has_ane": hasattr(self.active_backend, "has_ane") and self.active_backend.has_ane
        }
        
        # Aktualisiere mit Geräteinformationen
        backend_info.update(self.active_backend.get_device_info())
        
        return backend_info
    
    # === Tensor-Erstellung und -Manipulation ===
    
    def create_tensor(self, data: Any, dtype: Optional[Any] = None) -> Any:
        """
        Erstellt einen Tensor aus den angegebenen Daten.
        
        Args:
            data: Die Daten, aus denen der Tensor erstellt werden soll
            dtype: Datentyp für den Tensor
            
        Returns:
            Ein Backend-spezifischer Tensor
        """
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.create_tensor(data, dtype)
    
    def to_numpy(self, x: Any) -> Any:
        """Konvertiert einen Tensor zu einem NumPy-Array."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.to_numpy(x)
    
    def from_numpy(self, x: Any) -> Any:
        """Konvertiert ein NumPy-Array in einen Backend-Tensor."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.from_numpy(x)
    
    # === Basisoperationen ===
    # Diese Methoden delegieren an das aktive Backend
    
    def add(self, a: Any, b: Any) -> Any:
        """Addiert zwei Tensoren."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.add(a, b)
    
    def subtract(self, a: Any, b: Any) -> Any:
        """Subtrahiert einen Tensor von einem anderen."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.subtract(a, b)
    
    def multiply(self, a: Any, b: Any) -> Any:
        """Multipliziert zwei Tensoren (elementweise)."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.multiply(a, b)
    
    def divide(self, a: Any, b: Any) -> Any:
        """Dividiert einen Tensor durch einen anderen (elementweise)."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.divide(a, b)
    
    def matmul(self, a: Any, b: Any) -> Any:
        """Führt eine Matrix-Multiplikation durch."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.matmul(a, b)
    
    # === Formoperationen ===
    
    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any:
        """Ändert die Form eines Tensors."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.reshape(x, shape)
    
    def transpose(self, x: Any, axes: Optional[Tuple[int, ...]] = None) -> Any:
        """Transponiert einen Tensor entlang der angegebenen Achsen."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.transpose(x, axes)
    
    # === Mathematische Funktionen ===
    
    def exp(self, x: Any) -> Any:
        """Berechnet die Exponentialfunktion für jeden Wert im Tensor."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.exp(x)
    
    def log(self, x: Any) -> Any:
        """Berechnet den natürlichen Logarithmus für jeden Wert im Tensor."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.log(x)
    
    def sin(self, x: Any) -> Any:
        """Berechnet den Sinus für jeden Wert im Tensor."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.sin(x)
    
    def cos(self, x: Any) -> Any:
        """Berechnet den Kosinus für jeden Wert im Tensor."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.cos(x)
    
    # === Reduktionsoperationen ===
    
    def sum(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Any:
        """Berechnet die Summe entlang der angegebenen Achsen."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.sum(x, axis, keepdims)
    
    def mean(self, x: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Any:
        """Berechnet den Mittelwert entlang der angegebenen Achsen."""
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.mean(x, axis, keepdims)
    
    # === Lineare Algebra ===
    
    def svd(self, x: Any, full_matrices: bool = True) -> Tuple[Any, Any, Any]:
        """
        Berechnet die Singulärwertzerlegung einer Matrix.
        
        Returns:
            Tupel (U, S, V), wobei U und V orthogonale Matrizen und S die Singulärwerte sind
        """
        if self.active_backend is None:
            raise RuntimeError("Kein aktives Backend verfügbar.")
        
        return self.active_backend.svd(x, full_matrices)
    
    # === JIT-Kompilierung ===
    
    def jit(self, fn: Callable) -> Callable:
        """
        JIT-kompiliert eine Funktion, wenn unterstützt.
        
        Args:
            fn: Zu kompilierende Funktion
            
        Returns:
            Kompilierte Funktion oder ursprüngliche Funktion, wenn nicht unterstützt
        """
        if not self.enable_jit:
            return fn
        
        if self.active_backend is None or not self.active_backend.supports_jit():
            return fn
        
        return self.active_backend.jit(fn)

# Singleton-Funktion zur Instanziierung der Engine
def get_engine(
    precision: Union[str, PrecisionType] = PrecisionType.FLOAT32,
    backend: str = "auto",
    optimize_for_ane: bool = True,
    enable_jit: bool = True,
    force_new: bool = False
) -> TMathEngine:
    """
    Gibt eine Instanz der T-Mathematics Engine zurück.
    
    Args:
        precision: Zu verwendende Präzision
        backend: Zu verwendendes Backend
        optimize_for_ane: Optimierung für Apple Neural Engine aktivieren
        enable_jit: JIT-Kompilierung aktivieren
        force_new: Erzwingt die Erstellung einer neuen Instanz
        
    Returns:
        Eine Instanz der T-Mathematics Engine
    """
    global _ENGINE_INSTANCE
    
    if _ENGINE_INSTANCE is None or force_new:
        _ENGINE_INSTANCE = TMathEngine(
            precision=precision,
            backend=backend,
            optimize_for_ane=optimize_for_ane,
            enable_jit=enable_jit
        )
    
    return _ENGINE_INSTANCE

# Modul-Initialisierung
def init():
    """Initialisiert das T-Mathematics-Modul."""
    engine = get_engine()
    logger.info(f"T-Mathematics Engine initialisiert mit {engine.active_backend_name}")
    return True
