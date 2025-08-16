#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - T-MATHEMATICS VXOR Integration

Dieses Modul implementiert die Integration zwischen der T-MATHEMATICS Engine 
und dem VX-MATRIX Modul für optimierte Tensor-Operationen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple

# Importiere T-MATHEMATICS Komponenten
from miso.math.t_mathematics.engine import TMathematicsEngine
from miso.math.t_mathematics.integration_manager import TMathIntegrationManager
from miso.math.t_mathematics.config import TMathConfig
from miso.math.t_mathematics.tensor_interface import MISOTensorInterface
from miso.math.t_mathematics.tensor_wrappers import MLXTensorWrapper, TorchTensorWrapper
from miso.math.t_mathematics.tensor_factory import TensorFactory, tensor_factory
from miso.math.t_mathematics.tensor_cache import TensorCacheManager, tensor_cache, global_tensor_cache_manager

# Importiere VXOR-Adapter-Core
from miso.vxor.vx_adapter_core import get_module, get_module_status

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.t_mathematics.vxor_integration")

class TMathVXORIntegration:
    """
    Klasse zur Integration der T-MATHEMATICS Engine mit VX-MATRIX
    
    Diese Klasse stellt die Verbindung zwischen der T-MATHEMATICS Engine und
    dem VX-MATRIX Modul her, um optimierte Tensor-Operationen auf verschiedenen
    Hardware-Plattformen auszuführen.
    """
    
    _instance = None  # Singleton-Pattern
    
    def __new__(cls, *args, **kwargs):
        """Implementiert das Singleton-Pattern"""
        if cls._instance is None:
            cls._instance = super(TMathVXORIntegration, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert die T-MATHEMATICS-VXOR-Integration
        
        Args:
            config_path: Pfad zur Konfigurationsdatei (optional)
        """
        # Initialisiere nur einmal (Singleton-Pattern)
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config", "vxor_integration_config.json"
        )
        
        # Initialisiere T-MATHEMATICS-Komponenten
        self.integration_manager = TMathIntegrationManager()
        self.config = TMathConfig()
        
        # Lade oder erstelle Konfiguration
        self.vxor_config = {}
        self.load_config()
        
        # Initialisiere Tensor-Cache für verbesserte Leistung
        max_cache_size = self.vxor_config.get("cache", {}).get("max_size", 1000)
        default_ttl = self.vxor_config.get("cache", {}).get("ttl", 600)  # 10 Minuten
        self.tensor_cache = TensorCacheManager(max_cache_size=max_cache_size, default_ttl=default_ttl)
        
        # Performance-Metriken
        self.performance_metrics = {
            "operations_count": 0,
            "cache_hits": 0,
            "computation_time_saved": 0.0  # in Sekunden
        }
        
        # Dynamischer Import des VX-MATRIX-Moduls
        try:
            self.vx_matrix = get_module("VX-MATRIX")
            self.matrix_available = True
            logger.info("VX-MATRIX erfolgreich initialisiert")
        except Exception as e:
            self.vx_matrix = None
            self.matrix_available = False
            logger.warning(f"VX-MATRIX nicht verfügbar: {e}")
        
        # Registriere diese Integration im T-MATH Integration Manager
        self.register_with_integration_manager()
        
        self.initialized = True
        logger.info("TMathVXORIntegration initialisiert")
    
    def load_config(self):
        """Lädt die Konfiguration aus der Konfigurationsdatei"""
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Wenn die Datei nicht existiert, erstelle Standardkonfiguration
            if not os.path.exists(self.config_path):
                self._create_default_config()
            
            # Lade die Konfiguration
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.vxor_config = json.load(f)
            
            logger.info(f"Konfiguration geladen: {len(self.vxor_config)} Einträge")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            # Erstelle Standardkonfiguration im Fehlerfall
            self._create_default_config()
    
    def _create_default_config(self):
        """Erstellt eine Standardkonfiguration"""
        default_config = {
            "vx_matrix": {
                "enabled": True,
                "preferred_backend": "auto",  # auto, mlx, torch
                "hardware_acceleration": True,
                "optimization_level": 3,  # 1-5
                "cache_operations": True,
                "debug_mode": False
            }
        }
        
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Speichere die Standardkonfiguration
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            
            self.vxor_config = default_config
            logger.info("Standardkonfiguration erstellt")
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Standardkonfiguration: {e}")
            self.vxor_config = default_config
    
    def register_with_integration_manager(self):
        """Registriert diese Integration im T-MATH Integration Manager"""
        try:
            # Erstelle eine optimierte T-MATH Engine für VX-MATRIX
            vx_matrix_config = TMathConfig(
                backend=self.vxor_config.get("vx_matrix", {}).get("preferred_backend", "auto"),
                optimization_level=self.vxor_config.get("vx_matrix", {}).get("optimization_level", 3),
                use_hardware_acceleration=self.vxor_config.get("vx_matrix", {}).get("hardware_acceleration", True),
                debug_mode=self.vxor_config.get("vx_matrix", {}).get("debug_mode", False)
            )
            
            vx_matrix_engine = TMathematicsEngine(config=vx_matrix_config)
            
            # Registriere die Engine im Integration Manager
            self.integration_manager.register_engine("vx_matrix", vx_matrix_engine)
            logger.info("VX-MATRIX-Engine im TMathIntegrationManager registriert")
        except Exception as e:
            logger.error(f"Fehler bei der Registrierung im Integration Manager: {e}")
    
    def compute_tensor_operation(self, operation: str, tensors: List[Any], backend: Optional[str] = None) -> Any:
        """
        Führt eine Tensor-Operation mit VX-MATRIX aus und nutzt Caching für verbesserte Leistung
        
        Args:
            operation: Name der Operation
            tensors: Liste der Tensoren oder Tensorähnlichen Objekte
            backend: Optional, Backend für die Berechnung (mlx, torch)
            
        Returns:
            Ergebnis der Operation als geeigneter TensorWrapper
        """
        # Aktualisiere Performance-Metriken
        self.performance_metrics["operations_count"] += 1
        
        # Parameter für Cache-Key
        cache_params = {
            "backend": backend,
            "optimization_level": self.vxor_config.get("vx_matrix", {}).get("optimization_level", 3)
        }
        
        # Versuche, das Ergebnis aus dem Cache abzurufen
        start_time = time.time()
        cached_result = self.tensor_cache.get(operation, tensors, cache_params)
        if cached_result is not None:
            # Cache-Hit: Verwende das zwischengespeicherte Ergebnis
            self.performance_metrics["cache_hits"] += 1
            
            # Berechne die eingesparte Zeit (geschätzt basierend auf ähnlichen Operationen)
            time_saved = self.vxor_config.get("cache", {}).get("avg_operation_time", 0.05)  # 50ms Standardwert
            self.performance_metrics["computation_time_saved"] += time_saved
            
            logger.debug(f"Cache-Hit für Operation '{operation}' (ca. {time_saved*1000:.1f}ms eingespart)")
            return cached_result
        
        # Matrix-Verfügbarkeit prüfen
        if not self.matrix_available:
            logger.warning("VX-MATRIX nicht verfügbar, verwende Standard-T-MATH-Engine")
            # Fallback zur Standard-Engine
            engine = self.integration_manager.get_engine()
            result = getattr(engine, operation)(*tensors)
            
            # Wenn das Ergebnis bereits ein MISOTensorInterface ist, gib es zurück
            if isinstance(result, MISOTensorInterface):
                # Speichere das Ergebnis im Cache
                self.tensor_cache.set(operation, tensors, result, cache_params)
                return result
                
            # Konvertiere das Ergebnis zu einem geeigneten Tensor
            if hasattr(result, 'numpy') or hasattr(result, '__array__'):
                # Für numpy-ähnliche Objekte
                numpy_result = result.numpy() if hasattr(result, 'numpy') else result.__array__()
                target_backend = backend or tensor_factory.get_preferred_backend()
                final_result = tensor_factory.create_tensor(numpy_result, target_backend)
                
                # Speichere das Ergebnis im Cache
                self.tensor_cache.set(operation, tensors, final_result, cache_params)
                return final_result
            
            return result
        
        # Ermittle das zu verwendende Backend
        if backend is None or backend == "auto":
            target_backend = tensor_factory.get_preferred_backend()
        else:
            target_backend = backend
        
        # Konvertiere alle Tensoren zum einheitlichen Format für die Serialisierung
        serialized_tensors = []
        for t in tensors:
            if isinstance(t, MISOTensorInterface):
                # Wenn ein Tensor bereits MISOTensorInterface implementiert, serialisiere ihn
                serialized_tensors.append(tensor_factory.serialize_tensor(t))
            elif hasattr(t, 'numpy') or hasattr(t, '__array__'):
                # Für numpy-ähnliche Objekte
                numpy_t = t.numpy() if hasattr(t, 'numpy') else t.__array__()
                temp_tensor = tensor_factory.create_tensor(numpy_t, target_backend)
                serialized_tensors.append(tensor_factory.serialize_tensor(temp_tensor))
            else:
                # Für primitive Datentypen
                serialized_tensors.append(t)
        
        # Bereite Parameter für VX-MATRIX vor
        params = {
            "operation": operation,
            "tensors": serialized_tensors,
            "backend": target_backend,
            "optimization_level": self.vxor_config.get("vx_matrix", {}).get("optimization_level", 3)
        }
        
        # Messe die Berechnungszeit für zukünftige Cache-Optimierungen
        operation_start_time = time.time()
        
        try:
            # Führe Operation mit VX-MATRIX aus
            result = self.vx_matrix.compute(params)
            
            # Berechne die Ausführungszeit der Operation
            operation_time = time.time() - operation_start_time
            
            # Aktualisiere die durchschnittliche Operationszeit (exponentieller gleitender Durchschnitt)
            avg_op_time = self.vxor_config.get("cache", {}).get("avg_operation_time", 0.05)
            alpha = 0.1  # Gewichtungsfaktor für neue Messungen
            new_avg_time = alpha * operation_time + (1 - alpha) * avg_op_time
            
            # Aktualisiere die Konfiguration
            if "cache" not in self.vxor_config:
                self.vxor_config["cache"] = {}
            self.vxor_config["cache"]["avg_operation_time"] = new_avg_time
            
            if result.get("success", False):
                # Deserialisiere das Ergebnis mit der TensorFactory
                serialized_result = result.get("result")
                
                if isinstance(serialized_result, dict) and "backend" in serialized_result:
                    # Falls das Ergebnis bereits serialisiert ist, deserialisiere es
                    final_result = tensor_factory.deserialize_tensor(serialized_result, backend)
                    # Speichere das Ergebnis im Cache
                    self.tensor_cache.set(operation, tensors, final_result, cache_params)
                    return final_result
                elif isinstance(serialized_result, (list, tuple)) and all(isinstance(item, dict) and "backend" in item for item in serialized_result):
                    # Liste von Tensoren
                    final_result = [tensor_factory.deserialize_tensor(item, backend) for item in serialized_result]
                    # Speichere das Ergebnis im Cache
                    self.tensor_cache.set(operation, tensors, final_result, cache_params)
                    return final_result
                else:
                    # Primitiver Datentyp oder anderes Format
                    # Speichere auch primitive Ergebnisse im Cache
                    self.tensor_cache.set(operation, tensors, serialized_result, cache_params)
                    return serialized_result
            else:
                logger.error(f"Fehler bei der VX-MATRIX-Operation: {result.get('error')}")
                raise RuntimeError(f"VX-MATRIX-Fehler: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Fehler bei der VX-MATRIX-Berechnung: {e}")
            # Fallback zur Standard-Engine
            logger.info(f"Verwende Fallback zur Standard-T-MATH-Engine für Operation: {operation}")
            engine = self.integration_manager.get_engine()
            fallback_result = getattr(engine, operation)(*tensors)
            
            # Konvertiere das Fallback-Ergebnis, wenn möglich
            if hasattr(fallback_result, 'numpy') or hasattr(fallback_result, '__array__'):
                numpy_result = fallback_result.numpy() if hasattr(fallback_result, 'numpy') else fallback_result.__array__()
                target_backend = backend or tensor_factory.get_preferred_backend()
                final_result = tensor_factory.create_tensor(numpy_result, target_backend)
                
                # Speichere das Fallback-Ergebnis im Cache
                self.tensor_cache.set(operation, tensors, final_result, cache_params)
                return final_result
            
            return fallback_result
    
    def optimize_for_hardware(self, tensor: MISOTensorInterface, target_device: Optional[str] = None) -> MISOTensorInterface:
        """
        Optimiert einen Tensor für die Ausführung auf spezifischer Hardware
        
        Args:
            tensor: Zu optimierender Tensor
            target_device: Zielgerät (cpu, cuda, mps, mlx)
            
        Returns:
            Optimierter Tensor
        """
        if not self.matrix_available:
            logger.warning("VX-MATRIX nicht verfügbar, kann Tensor nicht optimieren")
            return tensor
        
        # Wenn kein Zielgerät angegeben ist, wähle das beste verfügbare Backend
        if target_device is None:
            target_backend = tensor_factory.get_preferred_backend()
            
            # Übersetze Backend in entsprechendes device
            if target_backend == "mlx":
                target_device = "mlx"
            elif target_backend == "torch":
                if torch.cuda.is_available():
                    target_device = "cuda"
                elif hasattr(torch, "mps") and torch.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu"
            else:
                # Unbekanntes Backend, behalte das aktuelle bei
                return tensor
                
            # Wenn der Tensor bereits das richtige Backend verwendet, gib ihn unverändert zurück
            if tensor.backend == target_backend:
                return tensor
        else:
            # Mappe das Zielgerät auf ein Backend
            if target_device == "mlx":
                target_backend = "mlx"
            else:  # cpu, cuda, mps -> torch
                target_backend = "torch"
        
        try:
            # Serialisiere den Tensor für die Übertragung
            serialized_tensor = tensor_factory.serialize_tensor(tensor)
            
            # Bereite Parameter für VX-MATRIX vor
            params = {
                "tensor": serialized_tensor,
                "target_device": target_device,
                "optimization_level": self.vxor_config.get("vx_matrix", {}).get("optimization_level", 3)
            }
            
            # Führe Optimierung mit VX-MATRIX aus
            result = self.vx_matrix.optimize_tensor(params)
            
            if result.get("success", False):
                # Deserialisiere das optimierte Ergebnis
                optimized_tensor_data = result.get("result")
                
                # Überprüfe, ob das Ergebnis ein serialisierter Tensor ist
                if isinstance(optimized_tensor_data, dict) and "backend" in optimized_tensor_data:
                    return tensor_factory.deserialize_tensor(optimized_tensor_data)
                else:
                    logger.error("Ungültiges Ergebnis von VX-MATRIX (kein serialisierter Tensor)")
                    return tensor
            else:
                logger.error(f"Fehler bei der VX-MATRIX-Optimierung: {result.get('error')}")
                return tensor
                
        except Exception as e:
            logger.error(f"Fehler bei der Tensor-Optimierung mit VX-MATRIX: {e}")
            return tensor


# Singleton-Instanz der Integration
_integration_instance = None

def get_tmath_vxor_integration() -> TMathVXORIntegration:
    """
    Gibt die Singleton-Instanz der T-MATHEMATICS-VXOR-Integration zurück
    
    Returns:
        TMathVXORIntegration-Instanz
    """
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = TMathVXORIntegration()
    return _integration_instance


# Initialisiere die Integration, wenn das Modul importiert wird
get_tmath_vxor_integration()

# Hauptfunktion
if __name__ == "__main__":
    integration = get_tmath_vxor_integration()
    print(f"T-MATHEMATICS ↔ VX-MATRIX Integration Status: {integration.matrix_available}")
