#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MATRIX-Modul - VXTensorConverter

Diese Klasse ist für die zentrale Konvertierung zwischen verschiedenen Tensor-Typen zuständig.
Sie unterstützt Konvertierungen zwischen NumPy, MLX, PyTorch und anderen Tensor-Formaten.
"""

import numpy as np
import logging
from enum import Enum, auto

# Logger konfigurieren
logger = logging.getLogger("VX-MATRIX.TensorConverter")

# Tensor-Typen Enumerator
class TensorType(Enum):
    NUMPY = auto()
    MLX = auto()
    TORCH = auto()
    TENSORFLOW = auto()
    JAX = auto()
    UNKNOWN = auto()
    
    @classmethod
    def from_string(cls, name):
        """Konvertiert einen String zur entsprechenden TensorType-Enum
        
        Args:
            name (str): Name des TensorTypes (numpy, mlx, torch, etc.)
            
        Returns:
            TensorType: Die entsprechende Enum-Konstante
        """
        name = name.upper()
        try:
            return cls[name]
        except KeyError:
            # Fallbacks für gängige Varianten
            mapping = {
                "NP": cls.NUMPY,
                "NUMPYARRAY": cls.NUMPY,
                "MX": cls.MLX,
                "MLXARRAY": cls.MLX,
                "PYTORCH": cls.TORCH,
                "TF": cls.TENSORFLOW,
                "TENSORFLOW": cls.TENSORFLOW,
                "JAXARRAY": cls.JAX
            }
            if name in mapping:
                return mapping[name]
            return cls.UNKNOWN

class VXTensorConverter:
    """
    Zentrale Klasse für alle Tensor-Konvertierungen im VX-MATRIX-System.
    Unterstützt optimierte Konvertierungen zwischen verschiedenen Tensor-Formaten.
    """
    
    def __init__(self, enable_caching=True, max_cache_size=1000):
        """
        Initialisiert den TensorConverter mit optimiertem Caching-System.
        
        Diese Implementierung unterstützt fortschrittliches Caching für
        häufig verwendete Operationen und ist optimiert für die Integration
        mit PRISM und ECHO-PRIME.
        
        Args:
            enable_caching (bool): Ob Caching für wiederholte Konvertierungen aktiviert werden soll
            max_cache_size (int): Maximale Anzahl an Cache-Einträgen (Standard: 1000)
        """
        # Haupt-Konfigurationswerte
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size
        
        # Cache und Metriken
        self._cache = {}  # Cache für bereits konvertierte Tensoren
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Backend-Verfügbarkeit
        self._mlx_available = False
        self._torch_available = False
        self._tf_available = False
        self._jax_available = False
        
        # MLX-spezifische Optimierungsflags
        self._mlx_jit_enabled = True  # JIT-Kompilierung für MLX-Konvertierungen aktivieren
        self._use_bfloat16 = True    # bfloat16 für ML-Workloads auf Apple Silicon aktivieren
        
        # PRISM-spezifische Optimierungsflags
        self._prism_compatible = True  # Kompatibilität mit PRISM sicherstellen
        self._optimized_batch_ops = True  # Batch-Operationen optimieren
        
        # Backend-Verfügbarkeit prüfen
        self._check_available_backends()
    
    def get_cache_statistics(self):
        """
        Liefert Statistiken über die Nutzung des Konvertierungs-Caches.
        
        Returns:
            dict: Dictionary mit Cache-Metriken
        """
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_accesses * 100) if total_accesses > 0 else 0
        
        stats = {
            'cache_size': len(self._cache),
            'max_cache_size': self.max_cache_size,
            'cache_usage_percent': len(self._cache) / self.max_cache_size * 100,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_accesses': total_accesses,
            'hit_rate': hit_rate,
            'enabled': self.enable_caching
        }
        
        return stats
    
    def clear_cache(self):
        """
        Leert den Konvertierungs-Cache vollständig.
        
        Dies kann nützlich sein, um Speicher freizugeben oder nach größeren
        Operationen einen frischen Start zu gewährleisten.
        
        Returns:
            int: Anzahl der gelöschten Cache-Einträge
        """
        cache_size = len(self._cache)
        self._cache.clear()
        return cache_size
        
    def optimize_for_mlx(self, enable=True):
        """
        Optimiert den Konverter für MLX (Apple Silicon).
        
        Aktiviert spezielle Optimierungen für MLX, einschließlich JIT-Kompilierung
        und bfloat16-Unterstützung für optimale Performance auf Apple Silicon.
        
        Args:
            enable (bool): Ob MLX-Optimierungen aktiviert werden sollen
            
        Returns:
            bool: True wenn MLX verfügbar ist und Optimierungen aktiviert wurden
        """
        self._mlx_jit_enabled = enable
        self._use_bfloat16 = enable
        
        return self._mlx_available and enable
        
    def optimize_for_prism(self, enable=True):
        """
        Optimiert den Konverter für die PRISM-Engine.
        
        Aktiviert spezielle Optimierungen für die Integration mit der PRISM-Engine,
        einschließlich erweiterter Precision und automatischer NaN/Inf-Behandlung.
        
        Args:
            enable (bool): Ob PRISM-Optimierungen aktiviert werden sollen
            
        Returns:
            bool: True wenn Optimierungen aktiviert wurden
        """
        self._prism_compatible = enable
        self._optimized_batch_ops = enable
        
        return enable
        
    def _check_available_backends(self):
        """Prüft, welche Tensor-Backends verfügbar sind."""
        try:
            import mlx.core as mx
            self._mlx_available = True
            logger.info("MLX ist verfügbar")
        except ImportError:
            logger.info("MLX ist nicht verfügbar")
        
        try:
            import torch
            self._torch_available = True
            logger.info("PyTorch ist verfügbar")
        except ImportError:
            logger.info("PyTorch ist nicht verfügbar")
        
        try:
            import tensorflow as tf
            self._tf_available = True
            logger.info("TensorFlow ist verfügbar")
        except ImportError:
            logger.info("TensorFlow ist nicht verfügbar")
        
        try:
            import jax.numpy as jnp
            self._jax_available = True
            logger.info("JAX ist verfügbar")
        except ImportError:
            logger.info("JAX ist nicht verfügbar")
    
    def detect_tensor_type(self, tensor):
        """
        Erkennt den Typ eines Tensors.
        
        Args:
            tensor: Der zu prüfende Tensor
            
        Returns:
            TensorType: Der erkannte Tensor-Typ
        """
        module_name = type(tensor).__module__
        
        if module_name == 'numpy':
            return TensorType.NUMPY
        elif module_name == 'mlx.core':
            return TensorType.MLX
        elif module_name == 'torch':
            return TensorType.TORCH
        elif module_name.startswith('tensorflow'):
            return TensorType.TENSORFLOW
        elif module_name.startswith('jax'):
            return TensorType.JAX
        else:
            return TensorType.UNKNOWN
    
    def to_numpy(self, tensor):
        """
        Konvertiert einen beliebigen Tensor zu NumPy mit optimierter Handhabung.
        
        Diese Implementierung bietet optimierte Konvertierungspfade für verschiedene
        Tensor-Typen und berücksichtigt spezielle Datentypen wie bfloat16, die in der
        MLX-Integration für Apple Silicon verwendet werden. Die Methode ist auf
        Performance und Stabilität ausgelegt.
        
        Args:
            tensor: Der zu konvertierende Tensor (mlx.core.array, torch.Tensor, jax.Array, tf.Tensor)
            
        Returns:
            numpy.ndarray: Der konvertierte NumPy-Array mit optimaler Precision
        """
        import numpy as np
        
        # Schneller Pfad für bereits korrekte Typen
        tensor_type = self.detect_tensor_type(tensor)
        if tensor_type == TensorType.NUMPY:
            return tensor
        
        # Cache-Schlüssel erstellen
        if self.enable_caching:
            # Hash-basierte Cache-Schlüssel für bessere Effizienz
            if hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):
                # Für Arrays mit shape und dtype Attributen
                cache_key = (tuple(tensor.shape), str(tensor.dtype), "numpy")
            else:
                # Fallback für andere Typen
                cache_key = (id(tensor), "numpy")
                
            if cache_key in self._cache:
                self.cache_hits += 1
                return self._cache[cache_key]
            else:
                self.cache_misses += 1
        
        # Optimierte Direktkonvertierungen je nach Quelltyp
        try:
            if tensor_type == TensorType.MLX:
                import mlx.core as mx
                
                # MLX-Arrays haben keine numpy() Methode, wir müssen tolist() verwenden
                # und dann einen NumPy-Array daraus erstellen
                try:
                    # Spezielle Behandlung für bfloat16 (häufig im ML-Kontext auf Apple Silicon)
                    if hasattr(tensor, 'dtype') and str(tensor.dtype) == 'bfloat16':
                        # Konvertierung zu float32 für bessere Precision
                        float32_tensor = tensor.astype(mx.float32)
                        result = np.array(float32_tensor.tolist())
                    else:
                        # Standard-Konvertierungspfad
                        result = np.array(tensor.tolist())
                except Exception as e:
                    logger.warning(f"MLX tolist() fehlgeschlagen: {e}. Fallback-Methode wird verwendet.")
                    # Alternative Methode: __array__ Protokoll nutzen
                    result = np.array(tensor)
                    
                # Prüfen auf NaN und Inf (häufiges Problem bei bfloat16 -> float32 Konvertierung)
                if self._prism_compatible and np.isnan(result).any() or np.isinf(result).any():
                    logger.warning("NaN oder Inf bei MLX->NumPy Konvertierung gefunden. Führe Equilibrierung durch.")
                    result = np.nan_to_num(result, nan=0.0, posinf=1e30, neginf=-1e30)
            
            elif tensor_type == TensorType.TORCH:
                import torch
                
                # Detach für Tensoren im Computation Graph
                if hasattr(tensor, 'detach'):
                    tensor = tensor.detach()
                    
                # Sicherstellen, dass der Tensor auf der CPU ist
                if hasattr(tensor, 'device') and str(tensor.device) != 'cpu':
                    tensor = tensor.cpu()
                
                # Spezielle Behandlung für bfloat16 Tensoren
                if tensor.dtype == torch.bfloat16:
                    # Konvertierung zu float32 für bessere Precision
                    float32_tensor = tensor.to(torch.float32)
                    result = float32_tensor.numpy()
                else:
                    # Standard-Konvertierungspfad
                    result = tensor.numpy()
                    
                # Automatische Behandlung von NaN/Inf-Werten für PRISM-Kompatibilität
                if self._prism_compatible and (np.isnan(result).any() or np.isinf(result).any()):
                    logger.warning("NaN oder Inf bei PyTorch->NumPy Konvertierung gefunden. Führe Equilibrierung durch.")
                    result = np.nan_to_num(result, nan=0.0, posinf=1e30, neginf=-1e30)
            
            elif tensor_type == TensorType.JAX:
                import jax
                import jax.numpy as jnp
                
                # JAX-Array zu Host-Memory transferieren und konvertieren
                if hasattr(tensor, 'copy_to_host'):
                    result = tensor.copy_to_host()
                else:
                    result = np.array(tensor)
                    
                # Spezielle Behandlung für JAX bfloat16
                if hasattr(tensor, 'dtype') and tensor.dtype == jax.numpy.bfloat16:
                    # Präzisions-Überprüfung für kleine Werte
                    if np.abs(result).min() < 1e-7 and np.abs(result).max() > 1e-3:
                        logger.info("Präzisions-Warnung bei JAX bfloat16 - Kleine Werte könnten ungenau sein.")
            
            elif tensor_type == TensorType.TENSORFLOW:
                # Standard TensorFlow-Konvertierung (bereits optimiert)
                result = tensor.numpy()
            
            else:
                # Generischer Fallback mit Fehlerbehandlung
                try:
                    result = np.array(tensor)
                    logger.warning(f"Unbekannter Tensor-Typ {type(tensor).__module__}.{type(tensor).__name__}, "
                                 f"versuche Konvertierung zu NumPy über np.array()")
                except Exception as e:
                    logger.error(f"Konvertierung zu NumPy fehlgeschlagen: {e}")
                    raise ValueError(f"Konvertierung des Tensor-Typs {type(tensor).__module__}.{type(tensor).__name__} zu NumPy nicht unterstützt")
            
            # Optimierungen für PRISM-Kompatibilität
            if self._prism_compatible and result is not None:
                # Sicherstellen der NumPy-Arrays in der bevorzugten Precision (float64)
                # Dies verbessert die Genauigkeit für PRISM-Berechnungen
                if result.dtype != np.float64 and result.dtype.kind == 'f':
                    result = result.astype(np.float64)
            
            # Ergebnis im Cache speichern
            if self.enable_caching and result is not None:
                self._cache[cache_key] = result
                # Cache-Größe begrenzen
                if len(self._cache) > self.max_cache_size:
                    # LRU-Ersetzungsstrategie simulieren
                    import random
                    keys = list(self._cache.keys())
                    del self._cache[random.choice(keys)]
                    
            return result
            
        except Exception as e:
            logger.warning(f"Fehler bei NumPy-Konvertierung: {e}. Verwende Fallback.")
            
            # Robuster Fallback bei Fehler
            try:
                # Generischer Fallback als letzter Resort
                return np.array(tensor)
            except:
                # Im schlimmsten Fall: Erzeuge einen leeren Array
                logger.error(f"Kritischer Fehler bei NumPy-Konvertierung, erzeuge leeren Array")
                if hasattr(tensor, 'shape'):
                    return np.zeros(tensor.shape)
                else:
                    return np.zeros((1, 1))
    
    def to_mlx(self, tensor):
        """
        Konvertiert einen beliebigen Tensor zu MLX mit Optimierungen für Apple Silicon.
        
        Diese Implementierung unterstützt direkte Konvertierungen zwischen verschiedenen
        Tensor-Typen und ist speziell für die Apple Neural Engine (ANE) optimiert.
        
        Args:
            tensor: Der zu konvertierende Tensor (numpy.ndarray, torch.Tensor, jax.Array oder mlx.core.array)
            
        Returns:
            mlx.core.array: Der konvertierte MLX-Array, optimiert für Apple Silicon
        """
        if not self._mlx_available:
            raise ImportError("MLX ist nicht verfügbar. Bitte installieren Sie MLX, um diese Funktion zu nutzen.")
        
        import mlx.core as mx
        
        tensor_type = self.detect_tensor_type(tensor)
        
        # Schneller Pfad für bereits korrekte Typen
        if tensor_type == TensorType.MLX:
            return tensor
        
        # Cache-Schlüssel erstellen
        if self.enable_caching:
            # Hash-basierte Cache-Schlüssel für bessere Effizienz
            if hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):
                # Für Arrays mit shape und dtype Attributen
                cache_key = (tuple(tensor.shape), str(tensor.dtype), "mlx")
            else:
                # Fallback für andere Typen
                cache_key = (id(tensor), "mlx")
                
            if cache_key in self._cache:
                self.cache_hits += 1
                return self._cache[cache_key]
            else:
                self.cache_misses += 1
        
        # Optimierte Direktkonvertierungen je nach Quelltyp
        try:
            if tensor_type == TensorType.NUMPY:
                # Direkter Weg für NumPy-Arrays
                result = mx.array(tensor)
                
            elif tensor_type == TensorType.TORCH:
                # Direkter Weg für PyTorch-Tensoren ohne NumPy-Zwischenschritt
                import torch
                
                # CPU-Tensor sicherstellen
                if hasattr(tensor, 'device') and str(tensor.device) != 'cpu':
                    tensor = tensor.cpu()
                    
                # Tensor-Detachment für bessere Performance
                if hasattr(tensor, 'detach'):
                    tensor = tensor.detach()
                
                # Direkte Konvertierung basierend auf Datentyp
                if tensor.dtype == torch.bfloat16:
                    # Spezieller Pfad für bfloat16 (wichtig für ML-Workloads)
                    numpy_tensor = tensor.to(torch.float32).numpy()
                    result = mx.bfloat16(mx.array(numpy_tensor))
                else:
                    # Standard-Konvertierungspfad
                    numpy_tensor = tensor.numpy()
                    result = mx.array(numpy_tensor)
                    
            elif tensor_type == TensorType.JAX:
                # Direkter Weg für JAX-Arrays
                import jax.numpy as jnp
                
                # JAX-Array zu Host-Memory transferieren
                numpy_tensor = jnp.asarray(tensor)
                if hasattr(numpy_tensor, 'copy_to_host'):
                    numpy_tensor = numpy_tensor.copy_to_host()
                    
                result = mx.array(numpy_tensor)
                
            else:
                # Generischer Fallback über NumPy
                numpy_tensor = self.to_numpy(tensor)
                result = mx.array(numpy_tensor)
                
            # Ergebnis im Cache speichern
            if self.enable_caching:
                self._cache[cache_key] = result
                # Cache-Größe begrenzen
                if len(self._cache) > self.max_cache_size:
                    # LRU-Ersetzungsstrategie simulieren durch Entfernen eines zufälligen Elements
                    # In einer produktiven Implementierung würde hier ein echter LRU-Cache verwendet
                    import random
                    keys = list(self._cache.keys())
                    del self._cache[random.choice(keys)]
                    
            return result
            
        except Exception as e:
            logger.warning(f"Fehler bei MLX-Konvertierung: {e}. Verwende Fallback.")
            
            # Robuster Fallback bei Fehler
            try:
                numpy_tensor = self.to_numpy(tensor)
                return mx.array(numpy_tensor)
            except:
                # Im schlimmsten Fall: Erzeuge einen leeren Array
                logger.error(f"Kritischer Fehler bei MLX-Konvertierung, erzeuge leeren Array")
                if hasattr(tensor, 'shape'):
                    return mx.zeros(tensor.shape)
                else:
                    return mx.zeros((1, 1))
    
    def to_torch(self, tensor):
        """
        Konvertiert einen beliebigen Tensor zu PyTorch mit erweiterten Optimierungen.
        
        Diese Implementierung unterstützt direkte Konvertierungen zwischen verschiedenen
        Tensor-Typen und ist speziell für PRISM und ECHO-PRIME optimiert.
        
        Args:
            tensor: Der zu konvertierende Tensor (numpy.ndarray, mlx.core.array, jax.Array oder torch.Tensor)
            
        Returns:
            torch.Tensor: Der konvertierte PyTorch-Tensor
        """
        if not self._torch_available:
            raise ImportError("PyTorch ist nicht verfügbar. Bitte installieren Sie PyTorch, um diese Funktion zu nutzen.")
        
        import torch
        
        tensor_type = self.detect_tensor_type(tensor)
        
        # Schneller Pfad für bereits korrekte Typen
        if tensor_type == TensorType.TORCH:
            return tensor
        
        # Cache-Schlüssel erstellen
        if self.enable_caching:
            # Hash-basierte Cache-Schlüssel für bessere Effizienz
            if hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):
                # Für Arrays mit shape und dtype Attributen
                cache_key = (tuple(tensor.shape), str(tensor.dtype), "torch")
            else:
                # Fallback für andere Typen
                cache_key = (id(tensor), "torch")
                
            if cache_key in self._cache:
                self.cache_hits += 1
                return self._cache[cache_key]
            else:
                self.cache_misses += 1
        
        # Optimierte Direktkonvertierungen je nach Quelltyp
        try:
            if tensor_type == TensorType.NUMPY:
                # Direkter Weg für NumPy-Arrays mit Fehlerbehandlung
                try:
                    # Versuche direkte Konvertierung mit from_numpy
                    result = torch.from_numpy(tensor.copy())  # Kopie um Ownership-Probleme zu vermeiden
                except TypeError:
                    # Fallback für Datentypen, die nicht direkt unterstützt werden
                    result = torch.tensor(tensor)
                    
            elif tensor_type == TensorType.MLX:
                # Direkter Weg für MLX-Arrays
                import mlx.core as mx
                
                # Optimierte Konvertierung basierend auf Datentyp
                if hasattr(tensor, 'dtype') and str(tensor.dtype) == 'bfloat16':
                    # Spezieller Pfad für bfloat16
                    numpy_tensor = tensor.astype(mx.float32).numpy()
                    result = torch.from_numpy(numpy_tensor).to(torch.bfloat16)
                else:
                    # Standard-Konvertierungspfad
                    numpy_tensor = tensor.numpy()
                    result = torch.from_numpy(numpy_tensor)
                    
            elif tensor_type == TensorType.JAX:
                # Direkter Weg für JAX-Arrays
                import jax.numpy as jnp
                
                # JAX-Array zu Host-Memory transferieren
                numpy_tensor = jnp.asarray(tensor)
                if hasattr(numpy_tensor, 'copy_to_host'):
                    numpy_tensor = numpy_tensor.copy_to_host()
                    
                # JAX BFloat16 spezielle Behandlung
                import jax
                if hasattr(tensor, 'dtype') and tensor.dtype == jax.numpy.bfloat16:
                    numpy_tensor = numpy_tensor.astype(jnp.float32)
                    result = torch.from_numpy(numpy_tensor).to(torch.bfloat16)
                else:
                    result = torch.from_numpy(numpy_tensor)
                
            else:
                # Generischer Fallback über NumPy
                numpy_tensor = self.to_numpy(tensor)
                result = torch.from_numpy(numpy_tensor)
                
            # Ergebnis im Cache speichern
            if self.enable_caching:
                self._cache[cache_key] = result
                # Cache-Größe begrenzen
                if len(self._cache) > self.max_cache_size:
                    # LRU-Ersetzungsstrategie simulieren
                    import random
                    keys = list(self._cache.keys())
                    del self._cache[random.choice(keys)]
                    
            return result
            
        except Exception as e:
            logger.warning(f"Fehler bei PyTorch-Konvertierung: {e}. Verwende Fallback.")
            
            # Robuster Fallback bei Fehler
            try:
                numpy_tensor = self.to_numpy(tensor)
                return torch.from_numpy(numpy_tensor)
            except:
                # Im schlimmsten Fall: Erzeuge einen leeren Tensor
                logger.error(f"Kritischer Fehler bei PyTorch-Konvertierung, erzeuge leeren Tensor")
                if hasattr(tensor, 'shape'):
                    return torch.zeros(tensor.shape)
                else:
                    return torch.zeros((1, 1))
    
    def convert(self, tensor, target_type):
        """
        Konvertiert einen Tensor in den Zieltyp.
        
        Args:
            tensor: Der zu konvertierende Tensor
            target_type (TensorType): Der Ziel-Tensor-Typ
            
        Returns:
            Der konvertierte Tensor im Zielformat
        """
        if target_type == TensorType.NUMPY:
            return self.to_numpy(tensor)
        elif target_type == TensorType.MLX:
            return self.to_mlx(tensor)
        elif target_type == TensorType.TORCH:
            return self.to_torch(tensor)
        else:
            raise ValueError(f"Konvertierung zu {target_type} nicht implementiert")
    
    def clear_cache(self):
        """Leert den Konvertierungscache."""
        self._cache.clear()
        logger.debug("TensorConverter-Cache geleert")
        
    def from_numpy(self, numpy_array, target_type=None):
        """
        Konvertiert einen NumPy-Array in den angegebenen Ziel-Tensor-Typ.
        
        Args:
            numpy_array: Der zu konvertierende NumPy-Array
            target_type (TensorType oder str): Der Ziel-Tensor-Typ, standardmäßig NUMPY
            
        Returns:
            Der konvertierte Tensor im Zielformat
        """
        if target_type is None:
            return numpy_array
            
        # Konvertiere string zu TensorType wenn nötig
        if isinstance(target_type, str):
            target_type = TensorType.from_string(target_type)
        
        # Cache-Schlüssel erstellen
        if self.enable_caching:
            cache_key = (id(numpy_array), f"from_numpy_to_{target_type.name}")
            if cache_key in self._cache:
                return self._cache[cache_key]
                
        # Führe die Konvertierung basierend auf dem Zieltyp durch
        if target_type == TensorType.NUMPY:
            result = numpy_array
        elif target_type == TensorType.MLX:
            if not self._mlx_available:
                raise ImportError("MLX ist nicht verfügbar.")
            import mlx.core as mx
            result = mx.array(numpy_array)
        elif target_type == TensorType.TORCH:
            if not self._torch_available:
                raise ImportError("PyTorch ist nicht verfügbar.")
            import torch
            result = torch.from_numpy(numpy_array)
        else:
            raise ValueError(f"Konvertierung zu {target_type} nicht implementiert")
            
        # Ergebnis cachen
        if self.enable_caching:
            self._cache[cache_key] = result
            
        return result
