#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MATRIX: TensorBridge für nahtlose Konvertierung zwischen Tensor-Frameworks

Diese Datei implementiert die TensorBridge-Klasse, die spezialisierte und optimierte
Pfade für die Konvertierung zwischen verschiedenen Tensor-Frameworks (MLX, PyTorch, NumPy, JAX)
bereitstellt, mit besonderem Fokus auf die Leistung und Hardwareoptimierung.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
ZTM-Level: STRICT
"""

import os
import sys
import enum
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path

# Berücksichtige den MISO-Root-Pfad
VXOR_AI_DIR = Path(__file__).parent.parent.parent.absolute()
MISO_ROOT = VXOR_AI_DIR.parent / "miso"
sys.path.insert(0, str(MISO_ROOT))

# ZTM-Protokoll-Initialisierung
ZTM_ACTIVE = os.environ.get('MISO_ZTM_MODE', '1') == '1'
ZTM_LOG_LEVEL = os.environ.get('MISO_ZTM_LOG_LEVEL', 'INFO')

# Logger konfigurieren
logger = logging.getLogger("VXOR.VX-MATRIX.tensor_bridge")
logger.setLevel(getattr(logging, ZTM_LOG_LEVEL))

def ztm_log(message: str, level: str = 'INFO', module: str = 'TENSOR_BRIDGE'):
    """ZTM-konforme Logging-Funktion"""
    if not ZTM_ACTIVE:
        return
    
    log_func = getattr(logger, level.lower())
    log_func(f"[ZTM:{module}] {message}")

# Importiere TensorType aus matrix_core
try:
    from .matrix_core import TensorType
except ImportError:
    # Fallback für standalone-Betrieb
    class TensorType(enum.Enum):
        """Definiert die unterstützten Tensor-Typen"""
        NUMPY = "numpy"
        TORCH = "torch"
        MLX = "mlx"
        JAX = "jax"
        UNKNOWN = "unknown"
    
        @classmethod
        def detect(cls, tensor: Any) -> 'TensorType':
            """Erkennt den Typ eines Tensor-Objekts"""
            type_name = type(tensor).__module__ + '.' + type(tensor).__name__
            
            if 'numpy' in type_name:
                return cls.NUMPY
            elif 'torch' in type_name:
                return cls.TORCH
            elif 'mlx' in type_name:
                return cls.MLX
            elif 'jax' in type_name or 'jaxlib' in type_name:
                return cls.JAX
            else:
                return cls.UNKNOWN

class ConversionMode(enum.Enum):
    """Definiert die Konvertierungsmodi für die TensorBridge"""
    EXACT = "exact"        # Exakte Konvertierung, fehlt bei Unverträglichkeit
    RELAXED = "relaxed"    # Versucht Konvertierung mit Toleranz für Präzisionsverlust
    FORCED = "forced"      # Erzwingt Konvertierung, auch mit potenziellen Typverlusten
    
class ConversionStatistics:
    """Speichert Statistiken über Tensor-Konvertierungen"""
    
    def __init__(self):
        """Initialisiert Statistiken"""
        self.successful_conversions = {
            (TensorType.NUMPY.value, TensorType.TORCH.value): 0,
            (TensorType.NUMPY.value, TensorType.MLX.value): 0,
            (TensorType.NUMPY.value, TensorType.JAX.value): 0,
            (TensorType.TORCH.value, TensorType.NUMPY.value): 0,
            (TensorType.TORCH.value, TensorType.MLX.value): 0,
            (TensorType.TORCH.value, TensorType.JAX.value): 0,
            (TensorType.MLX.value, TensorType.NUMPY.value): 0,
            (TensorType.MLX.value, TensorType.TORCH.value): 0,
            (TensorType.MLX.value, TensorType.JAX.value): 0,
            (TensorType.JAX.value, TensorType.NUMPY.value): 0,
            (TensorType.JAX.value, TensorType.TORCH.value): 0,
            (TensorType.JAX.value, TensorType.MLX.value): 0,
        }
        self.failed_conversions = {k: 0 for k in self.successful_conversions.keys()}
        self.total_time = {k: 0.0 for k in self.successful_conversions.keys()}
    
    def record_success(self, from_type: str, to_type: str, time_taken: float):
        """Zeichnet eine erfolgreiche Konvertierung auf"""
        key = (from_type, to_type)
        if key in self.successful_conversions:
            self.successful_conversions[key] += 1
            self.total_time[key] += time_taken
    
    def record_failure(self, from_type: str, to_type: str):
        """Zeichnet eine fehlgeschlagene Konvertierung auf"""
        key = (from_type, to_type)
        if key in self.failed_conversions:
            self.failed_conversions[key] += 1
    
    def get_summary(self) -> Dict:
        """Gibt eine Zusammenfassung der Statistiken zurück"""
        summary = {
            "successful": self.successful_conversions.copy(),
            "failed": self.failed_conversions.copy(),
            "total_success": sum(self.successful_conversions.values()),
            "total_failed": sum(self.failed_conversions.values()),
            "average_times": {}
        }
        
        # Berechne durchschnittliche Konvertierungszeiten
        for key, total_time in self.total_time.items():
            if self.successful_conversions[key] > 0:
                avg_time = total_time / self.successful_conversions[key]
                summary["average_times"][key] = avg_time
            else:
                summary["average_times"][key] = 0.0
                
        return summary

class TensorBridge:
    """
    Brücke für die Konvertierung zwischen verschiedenen Tensor-Frameworks
    
    Diese Klasse bietet optimierte Pfade für die Konvertierung zwischen
    MLX, PyTorch, NumPy und JAX mit hoher Leistung und minimalen Typverlusten.
    """
    
    def __init__(self, preferred_backend: str = "auto"):
        """
        Initialisiert die TensorBridge
        
        Args:
            preferred_backend: 'auto', 'mlx', 'torch', 'numpy', 'jax'
        """
        self.conversion_stats = ConversionStatistics()
        self.backend_modules = {}
        self.available_backends = set()
        self._detect_available_backends()
        
        # Setze bevorzugtes Backend
        if preferred_backend == "auto":
            self.preferred_backend = self._auto_select_backend()
        else:
            if preferred_backend in self.available_backends:
                self.preferred_backend = preferred_backend
            else:
                fallback = self._auto_select_backend()
                ztm_log(f"Backend {preferred_backend} nicht verfügbar. Verwende {fallback}", level="WARNING")
                self.preferred_backend = fallback
                
        ztm_log(f"TensorBridge initialisiert mit Backend: {self.preferred_backend}", level="INFO")
        
        # Optimierungseinstellungen
        self.enable_direct_conversion = True  # Direkte Konvertierung, wo möglich
        self.force_zero_copy = False          # Wenn True, bevorzuge Zero-Copy-Konvertierungen
        self.use_miso_optimizations = True    # Verwende optimierte MISO-Konverter
        
        # Konvertierungscache
        self.enable_caching = True
        self._conversion_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 100
    
    def _detect_available_backends(self):
        """Erkennt die verfügbaren Backend-Module"""
        # NumPy
        try:
            import numpy as np
            self.backend_modules['numpy'] = np
            self.available_backends.add('numpy')
        except ImportError:
            pass
        
        # PyTorch
        try:
            import torch
            self.backend_modules['torch'] = torch
            self.available_backends.add('torch')
            
            # Spezielle Hardware für PyTorch
            if torch.cuda.is_available():
                self.available_backends.add('cuda')
            try:
                if torch.backends.mps.is_available():
                    self.available_backends.add('mps')
            except:
                pass
        except ImportError:
            pass
        
        # MLX
        try:
            import mlx.core as mx
            self.backend_modules['mlx'] = mx
            self.available_backends.add('mlx')
        except ImportError:
            pass
        
        # JAX
        try:
            import jax
            import jax.numpy as jnp
            self.backend_modules['jax'] = jnp
            self.backend_modules['jax_core'] = jax
            self.available_backends.add('jax')
        except ImportError:
            pass
            
        ztm_log(f"Verfügbare Tensor-Backends: {self.available_backends}", level="INFO")
    
    def _auto_select_backend(self) -> str:
        """Wählt automatisch das beste verfügbare Backend aus"""
        # Prioritätsreihenfolge: mlx (Apple Silicon) > cuda > mps > jax > numpy
        if 'mlx' in self.available_backends:
            return 'mlx'
        elif 'cuda' in self.available_backends:
            return 'torch'  # Mit CUDA
        elif 'mps' in self.available_backends:
            return 'torch'  # Mit MPS
        elif 'jax' in self.available_backends:
            return 'jax'
        elif 'numpy' in self.available_backends:
            return 'numpy'
        else:
            ztm_log("Keine Backend-Module verfügbar!", level="ERROR")
            return 'none'
    
    def convert(self, tensor: Any, target_type: str, strict: bool = True) -> Any:
        """
        Konvertiert einen Tensor in den angegebenen Zieltyp
        
        Args:
            tensor: Eingabe-Tensor
            target_type: Zieltyp (z.B. 'numpy', 'torch', 'mlx', 'jax')
            strict: Bei True wird ein Fehler geworfen, wenn die Konvertierung nicht exakt möglich ist
            
        Returns:
            Konvertierter Tensor
        """
        import time
        
        # Bestimme Eingangstyp
        current_type = TensorType.detect(tensor)
        
        # Validiere Zieltyp
        valid_targets = [t.value for t in TensorType if t != TensorType.UNKNOWN]
        if target_type not in valid_targets:
            ztm_log(f"Ungültiger Zieltyp: {target_type}. Erlaubte Werte: {valid_targets}", level="ERROR")
            raise ValueError(f"Ungültiger Zieltyp: {target_type}")
        
        # Wenn bereits der richtige Typ, nichts tun
        if current_type.value == target_type:
            return tensor
        
        # Cache-Check, falls aktiviert
        if self.enable_caching:
            # Wir können nicht das Tensor-Objekt direkt als Schlüssel verwenden, daher hashen wir Metadaten
            shape = getattr(tensor, 'shape', None)
            dtype = getattr(tensor, 'dtype', None)
            device = getattr(tensor, 'device', None) if hasattr(tensor, 'device') else None
            
            if shape is not None and dtype is not None:
                cache_key = (current_type.value, target_type, str(shape), str(dtype), str(device))
                if cache_key in self._conversion_cache:
                    # Wir haben eine ähnliche Konvertierung gesehen, verwenden die optimale Konvertierungsfunktion
                    self._cache_hits += 1
                    conversion_func = self._conversion_cache[cache_key]
                    return conversion_func(tensor)
                self._cache_misses += 1
        
        # ZTM-Logging
        ztm_log(f"Konvertiere Tensor von {current_type.value} zu {target_type}", level="INFO")
        
        # Messung starten
        start_time = time.time()
        
        try:
            # Spezifische direkte Konvertierungspfade
            if self.enable_direct_conversion:
                # Diese Pfade nutzen optimierte direkte Konvertierungen ohne NumPy als Zwischenschritt
                
                # MLX -> PyTorch direkt
                if current_type == TensorType.MLX and target_type == 'torch' and 'mlx' in self.available_backends and 'torch' in self.available_backends:
                    if self.use_miso_optimizations:
                        try:
                            # Versuche, optimierten MISO-Konverter zu verwenden
                            from miso.math.t_mathematics.mlx_support import tensor_to_torch
                            result = tensor_to_torch(tensor)
                            
                            # Caching
                            if self.enable_caching:
                                self._conversion_cache[cache_key] = tensor_to_torch
                                
                            conversion_time = time.time() - start_time
                            self.conversion_stats.record_success(current_type.value, target_type, conversion_time)
                            return result
                        except ImportError:
                            ztm_log("Optimierte MLX->Torch Konvertierung nicht verfügbar", level="WARNING")
                    
                    # Fallback zu eigener Implementierung
                    mx = self.backend_modules['mlx']
                    torch = self.backend_modules['torch']
                    
                    # MLX -> NumPy -> PyTorch
                    result = torch.tensor(tensor.tolist())
                    
                    # Caching
                    if self.enable_caching:
                        # Closure für die Konvertierungsfunktion
                        def convert_mlx_to_torch(t):
                            return torch.tensor(t.tolist())
                        self._conversion_cache[cache_key] = convert_mlx_to_torch
                    
                    conversion_time = time.time() - start_time
                    self.conversion_stats.record_success(current_type.value, target_type, conversion_time)
                    return result
                
                # PyTorch -> MLX direkt
                elif current_type == TensorType.TORCH and target_type == 'mlx' and 'torch' in self.available_backends and 'mlx' in self.available_backends:
                    if self.use_miso_optimizations:
                        try:
                            # Versuche, optimierten MISO-Konverter zu verwenden
                            from miso.math.t_mathematics.mlx_support import tensor_from_torch
                            result = tensor_from_torch(tensor)
                            
                            # Caching
                            if self.enable_caching:
                                self._conversion_cache[cache_key] = tensor_from_torch
                                
                            conversion_time = time.time() - start_time
                            self.conversion_stats.record_success(current_type.value, target_type, conversion_time)
                            return result
                        except ImportError:
                            ztm_log("Optimierte Torch->MLX Konvertierung nicht verfügbar", level="WARNING")
                    
                    # Fallback zu eigener Implementierung
                    mx = self.backend_modules['mlx']
                    
                    # PyTorch -> NumPy -> MLX
                    result = mx.array(tensor.detach().cpu().numpy())
                    
                    # Caching
                    if self.enable_caching:
                        # Closure für die Konvertierungsfunktion
                        def convert_torch_to_mlx(t):
                            return mx.array(t.detach().cpu().numpy())
                        self._conversion_cache[cache_key] = convert_torch_to_mlx
                    
                    conversion_time = time.time() - start_time
                    self.conversion_stats.record_success(current_type.value, target_type, conversion_time)
                    return result
            
            # Standardkonvertierungen über NumPy
            
            # Konvertiere zu NumPy als Zwischenschritt
            tensor_np = None
            if current_type == TensorType.NUMPY:
                tensor_np = tensor
            elif current_type == TensorType.TORCH:
                tensor_np = tensor.detach().cpu().numpy()
            elif current_type == TensorType.MLX:
                # MLX -> NumPy ist nicht trivial, da es keine direkte .numpy() Methode gibt
                # Verschiedene Ansätze abhängig von der Tensor-Größe
                try:
                    # Für kleine Arrays ist tolist() effizient genug
                    tensor_np = np.array(tensor.tolist())
                except Exception as e:
                    ztm_log(f"Fehler bei MLX->NumPy Konvertierung: {e}", level="ERROR")
                    if strict:
                        raise ValueError(f"Konvertierung von MLX zu NumPy fehlgeschlagen: {e}")
                    return None
            elif current_type == TensorType.JAX:
                tensor_np = np.array(tensor)
            else:
                ztm_log(f"Unbekannter Tensor-Typ: {current_type}", level="ERROR")
                if strict:
                    raise ValueError(f"Unbekannter Tensor-Typ: {current_type}")
                return None
            
            # Konvertiere von NumPy zum Zieltyp
            result = None
            if target_type == 'numpy':
                result = tensor_np
            elif target_type == 'torch':
                torch = self.backend_modules['torch']
                result = torch.tensor(tensor_np)
                
                # Spezielle Hardware-Beschleunigung
                if self.preferred_backend == 'torch':
                    if 'cuda' in self.available_backends:
                        result = result.cuda()
                    elif 'mps' in self.available_backends:
                        result = result.to("mps")
            elif target_type == 'mlx':
                mx = self.backend_modules['mlx']
                result = mx.array(tensor_np)
            elif target_type == 'jax':
                jnp = self.backend_modules['jax']
                result = jnp.array(tensor_np)
            else:
                ztm_log(f"Unbekannter Ziel-Tensor-Typ: {target_type}", level="ERROR")
                if strict:
                    raise ValueError(f"Unbekannter Ziel-Tensor-Typ: {target_type}")
                return None
            
            # Caching für diese Art von Konvertierung
            if self.enable_caching:
                # Wir müssen eine Closure erstellen, um die Konvertierung zu kapseln
                if current_type == TensorType.NUMPY and target_type == 'torch':
                    torch = self.backend_modules['torch']
                    def convert_np_to_torch(t):
                        return torch.tensor(t)
                    self._conversion_cache[cache_key] = convert_np_to_torch
                elif current_type == TensorType.NUMPY and target_type == 'mlx':
                    mx = self.backend_modules['mlx']
                    def convert_np_to_mlx(t):
                        return mx.array(t)
                    self._conversion_cache[cache_key] = convert_np_to_mlx
                elif current_type == TensorType.NUMPY and target_type == 'jax':
                    jnp = self.backend_modules['jax']
                    def convert_np_to_jax(t):
                        return jnp.array(t)
                    self._conversion_cache[cache_key] = convert_np_to_jax
                elif current_type == TensorType.TORCH and target_type == 'numpy':
                    def convert_torch_to_np(t):
                        return t.detach().cpu().numpy()
                    self._conversion_cache[cache_key] = convert_torch_to_np
                elif current_type == TensorType.TORCH and target_type == 'jax':
                    jnp = self.backend_modules['jax']
                    def convert_torch_to_jax(t):
                        return jnp.array(t.detach().cpu().numpy())
                    self._conversion_cache[cache_key] = convert_torch_to_jax
                elif current_type == TensorType.MLX and target_type == 'numpy':
                    def convert_mlx_to_np(t):
                        return np.array(t.tolist())
                    self._conversion_cache[cache_key] = convert_mlx_to_np
                elif current_type == TensorType.MLX and target_type == 'jax':
                    jnp = self.backend_modules['jax']
                    def convert_mlx_to_jax(t):
                        return jnp.array(np.array(t.tolist()))
                    self._conversion_cache[cache_key] = convert_mlx_to_jax
                elif current_type == TensorType.JAX and target_type == 'numpy':
                    def convert_jax_to_np(t):
                        return np.array(t)
                    self._conversion_cache[cache_key] = convert_jax_to_np
                elif current_type == TensorType.JAX and target_type == 'torch':
                    torch = self.backend_modules['torch']
                    def convert_jax_to_torch(t):
                        return torch.tensor(np.array(t))
                    self._conversion_cache[cache_key] = convert_jax_to_torch
                elif current_type == TensorType.JAX and target_type == 'mlx':
                    mx = self.backend_modules['mlx']
                    def convert_jax_to_mlx(t):
                        return mx.array(np.array(t))
                    self._conversion_cache[cache_key] = convert_jax_to_mlx
            
            # Cache-Management
            if self.enable_caching and len(self._conversion_cache) > self._max_cache_size:
                # Einfache LRU-Implementierung: Entferne den ältesten Eintrag
                self._conversion_cache.pop(next(iter(self._conversion_cache)))
            
            # Statistiken aktualisieren
            conversion_time = time.time() - start_time
            self.conversion_stats.record_success(current_type.value, target_type, conversion_time)
            
            return result
            
        except Exception as e:
            ztm_log(f"Fehler bei der Tensor-Konvertierung: {e}", level="ERROR")
            self.conversion_stats.record_failure(current_type.value, target_type)
            if strict:
                raise
            return None
    
    def get_stats(self) -> Dict:
        """
        Gibt Statistiken über durchgeführte Konvertierungen zurück
        
        Returns:
            Dictionary mit Konvertierungsstatistiken
        """
        stats = self.conversion_stats.get_summary()
        
        # Füge Cache-Informationen hinzu, falls aktiviert
        if self.enable_caching:
            stats["cache"] = {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "size": len(self._conversion_cache),
                "max_size": self._max_cache_size
            }
            
        return stats
    
    def optimize_for_backend(self, tensor: Any, backend: str = None) -> Any:
        """
        Optimiert einen Tensor für ein bestimmtes Backend
        
        Args:
            tensor: Eingabe-Tensor
            backend: Ziel-Backend (falls None, wird das bevorzugte Backend verwendet)
            
        Returns:
            Optimierter Tensor
        """
        if backend is None:
            backend = self.preferred_backend
        
        current_type = TensorType.detect(tensor)
        
        # MLX-Optimierung
        if backend == 'mlx' and current_type != TensorType.MLX and 'mlx' in self.available_backends:
            return self.convert(tensor, 'mlx')
        
        # PyTorch mit CUDA-Optimierung
        elif backend == 'torch' and 'cuda' in self.available_backends:
            if current_type != TensorType.TORCH:
                tensor = self.convert(tensor, 'torch')
            return tensor.cuda()
        
        # PyTorch mit MPS-Optimierung (Apple Silicon)
        elif backend == 'torch' and 'mps' in self.available_backends:
            if current_type != TensorType.TORCH:
                tensor = self.convert(tensor, 'torch')
            return tensor.to("mps")
        
        # JAX-Optimierung
        elif backend == 'jax' and current_type != TensorType.JAX and 'jax' in self.available_backends:
            return self.convert(tensor, 'jax')
        
        # NumPy als Fallback
        elif backend == 'numpy' and current_type != TensorType.NUMPY and 'numpy' in self.available_backends:
            return self.convert(tensor, 'numpy')
        
        # Wenn keine spezielle Optimierung möglich, gib den ursprünglichen Tensor zurück
        return tensor

# Wenn direkt ausgeführt, führe einen kleinen Test durch
if __name__ == "__main__":
    import numpy as np
    
    print("TensorBridge Test")
    
    # Erstelle TensorBridge
    bridge = TensorBridge()
    print(f"Preferred Backend: {bridge.preferred_backend}")
    print(f"Available Backends: {bridge.available_backends}")
    
    # Testmatrix erstellen
    a_np = np.random.rand(3, 3)
    
    print("\nNumPy Matrix:")
    print(a_np)
    
    # Versuche Konvertierungen, wenn verfügbar
    if 'torch' in bridge.available_backends:
        print("\nKonvertieren zu PyTorch:")
        a_torch = bridge.convert(a_np, 'torch')
        print(a_torch)
        
        print("\nZurück zu NumPy:")
        a_np_back = bridge.convert(a_torch, 'numpy')
        print(a_np_back)
    
    if 'mlx' in bridge.available_backends:
        print("\nKonvertieren zu MLX:")
        a_mlx = bridge.convert(a_np, 'mlx')
        print(a_mlx)
        
        print("\nZurück zu NumPy:")
        a_np_back = bridge.convert(a_mlx, 'numpy')
        print(a_np_back)
    
    if 'jax' in bridge.available_backends:
        print("\nKonvertieren zu JAX:")
        a_jax = bridge.convert(a_np, 'jax')
        print(a_jax)
        
        print("\nZurück zu NumPy:")
        a_np_back = bridge.convert(a_jax, 'numpy')
        print(a_np_back)
    
    # Statistiken
    print("\nKonvertierungsstatistiken:")
    stats = bridge.get_stats()
    print(f"Erfolgreiche Konvertierungen: {stats['total_success']}")
    print(f"Fehlgeschlagene Konvertierungen: {stats['total_failed']}")
