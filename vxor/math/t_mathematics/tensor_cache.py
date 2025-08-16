#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Engine - Tensor-Cache-Manager

Diese Datei implementiert einen Cache-Manager für Tensor-Operationen,
der die Leistung durch Zwischenspeicherung häufig verwendeter Ergebnisse verbessert.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import logging
import time
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from functools import lru_cache
from collections import OrderedDict
import numpy as np

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.t_mathematics.tensor_cache")

class TensorCacheManager:
    """
    Cache-Manager für Tensor-Operationen.
    
    Dieser Cache-Manager speichert die Ergebnisse von Tensor-Operationen zwischen,
    um wiederholte Berechnungen zu vermeiden und die Leistung zu verbessern.
    """
    
    def __init__(self, max_cache_size: int = 1000, default_ttl: int = 600):
        """
        Initialisiert den TensorCacheManager.
        
        Args:
            max_cache_size: Maximale Anzahl von Einträgen im Cache
            default_ttl: Standard-Time-to-Live in Sekunden (0 für unbegrenzt)
        """
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        
        # Cache-Struktur: {key: (timestamp, ttl, value)}
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"TensorCacheManager initialisiert mit max_cache_size={max_cache_size}, default_ttl={default_ttl}s")
    
    def _generate_cache_key(self, operation: str, tensors: List[Any], params: Dict[str, Any] = None) -> str:
        """
        Generiert einen eindeutigen Schlüssel für eine Tensor-Operation.
        
        Args:
            operation: Name der Operation
            tensors: Liste der Input-Tensoren (oder deren Metadaten)
            params: Zusätzliche Parameter für die Operation
            
        Returns:
            Eindeutiger Cache-Schlüssel als String
        """
        # Erstelle eine Liste von Tensor-Fingerabdrücken
        tensor_fingerprints = []
        for t in tensors:
            if hasattr(t, 'serialize'):
                # Für MISOTensorInterface-Implementierungen
                t_dict = t.serialize()
                # Wichtige Metadaten für den Fingerabdruck extrahieren
                fingerprint = {
                    "shape": t_dict.get("shape"),
                    "dtype": t_dict.get("dtype"),
                    "backend": t_dict.get("backend")
                }
                tensor_fingerprints.append(fingerprint)
            elif hasattr(t, 'shape') and hasattr(t, 'dtype'):
                # Für numpy-ähnliche Objekte
                fingerprint = {
                    "shape": t.shape if hasattr(t.shape, '__iter__') else (t.shape,),
                    "dtype": str(t.dtype)
                }
                tensor_fingerprints.append(fingerprint)
            elif isinstance(t, (int, float, bool, str)):
                # Für primitive Datentypen
                tensor_fingerprints.append({
                    "value": t,
                    "type": type(t).__name__
                })
            else:
                # Für andere Objekte, verwende den Typ und Hashwert
                tensor_fingerprints.append({
                    "type": type(t).__name__,
                    "hash": str(hash(t)) if hasattr(t, '__hash__') and t.__hash__ is not None else str(id(t))
                })
        
        # Kombiniere Operation, Tensor-Fingerabdrücke und Parameter zu einem JSON-String
        key_data = {
            "operation": operation,
            "tensors": tensor_fingerprints,
            "params": params or {}
        }
        
        # Erzeuge MD5-Hash des JSON-Strings als Cache-Schlüssel
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode('utf-8')).hexdigest()
    
    def get(self, operation: str, tensors: List[Any], params: Dict[str, Any] = None) -> Optional[Any]:
        """
        Ruft ein Ergebnis aus dem Cache ab, falls vorhanden und nicht abgelaufen.
        
        Args:
            operation: Name der Operation
            tensors: Liste der Input-Tensoren
            params: Zusätzliche Parameter für die Operation
            
        Returns:
            Zwischengespeichertes Ergebnis oder None, falls nicht im Cache
        """
        key = self._generate_cache_key(operation, tensors, params)
        
        if key in self.cache:
            timestamp, ttl, value = self.cache[key]
            
            # Prüfe, ob der Cache-Eintrag abgelaufen ist
            if ttl > 0 and time.time() - timestamp > ttl:
                # Eintrag ist abgelaufen, entferne ihn aus dem Cache
                self.cache.pop(key)
                self.evictions += 1
                self.misses += 1
                logger.debug(f"Cache-Eintrag abgelaufen: {operation}")
                return None
            
            # Cache-Hit: Bewege den Eintrag ans Ende (LRU-Strategie)
            self.cache.move_to_end(key)
            self.hits += 1
            logger.debug(f"Cache-Hit: {operation}")
            return value
        
        # Cache-Miss
        self.misses += 1
        logger.debug(f"Cache-Miss: {operation}")
        return None
    
    def set(self, operation: str, tensors: List[Any], result: Any, params: Dict[str, Any] = None, ttl: Optional[int] = None) -> None:
        """
        Speichert ein Ergebnis im Cache.
        
        Args:
            operation: Name der Operation
            tensors: Liste der Input-Tensoren
            result: Zu speicherndes Ergebnis
            params: Zusätzliche Parameter für die Operation
            ttl: Time-to-Live in Sekunden (None für Standardwert)
        """
        key = self._generate_cache_key(operation, tensors, params)
        ttl = self.default_ttl if ttl is None else ttl
        
        # Speichere Zeitstempel, TTL und Wert
        self.cache[key] = (time.time(), ttl, result)
        
        # Bewege den Eintrag ans Ende (LRU-Strategie)
        self.cache.move_to_end(key)
        
        # Wenn der Cache die maximale Größe überschreitet, entferne den ältesten Eintrag
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)  # Entferne den ersten Eintrag (ältester nach LRU)
            self.evictions += 1
        
        logger.debug(f"Cache-Eintrag gespeichert: {operation}")
    
    def invalidate(self, operation: Optional[str] = None) -> None:
        """
        Invalidiert Cache-Einträge basierend auf der Operation.
        
        Args:
            operation: Name der zu invalidierenden Operation (None für alle)
        """
        if operation is None:
            # Invalidiere den gesamten Cache
            self.cache.clear()
            logger.info("Gesamter Cache invalidiert")
            return
        
        # Invalidiere nur Einträge für die angegebene Operation
        keys_to_remove = []
        for key, (_, _, _) in self.cache.items():
            # Der Schlüssel enthält die Operation, aber wir müssen alle Schlüssel durchgehen
            # und nach der Operation im JSON-String suchen
            key_data = json.loads(key)
            if key_data.get("operation") == operation:
                keys_to_remove.append(key)
        
        # Entferne die Schlüssel
        for key in keys_to_remove:
            self.cache.pop(key, None)
        
        logger.info(f"Cache für Operation '{operation}' invalidiert ({len(keys_to_remove)} Einträge)")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über den Cache zurück.
        
        Returns:
            Dictionary mit Cache-Statistiken
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_cache_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions
        }
    
    def cleanup(self) -> int:
        """
        Entfernt abgelaufene Einträge aus dem Cache.
        
        Returns:
            Anzahl der entfernten Einträge
        """
        current_time = time.time()
        keys_to_remove = []
        
        for key, (timestamp, ttl, _) in self.cache.items():
            if ttl > 0 and current_time - timestamp > ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.cache.pop(key)
        
        logger.info(f"Cache-Bereinigung: {len(keys_to_remove)} abgelaufene Einträge entfernt")
        return len(keys_to_remove)


# Dekorator für das Caching von Funktionsergebnissen
def tensor_cache(ttl: Optional[int] = None, cache_manager: Optional[TensorCacheManager] = None):
    """
    Dekorator für das Caching von Tensor-Operationen.
    
    Args:
        ttl: Time-to-Live in Sekunden (None für Standardwert)
        cache_manager: Zu verwendender Cache-Manager (None für globalen Standard)
        
    Returns:
        Dekorierte Funktion
    """
    # Verwende den bereitgestellten Cache-Manager oder erstelle einen neuen
    manager = cache_manager or global_tensor_cache_manager
    
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Erstelle eine Liste der positional und keyword arguments
            all_args = list(args)
            all_kwargs = dict(kwargs)
            
            # Versuche, das Ergebnis aus dem Cache zu holen
            cached_result = manager.get(func.__name__, all_args, all_kwargs)
            if cached_result is not None:
                return cached_result
            
            # Berechne das Ergebnis
            result = func(*args, **kwargs)
            
            # Speichere das Ergebnis im Cache
            manager.set(func.__name__, all_args, result, all_kwargs, ttl)
            
            return result
        
        return wrapper
    
    return decorator


# Globaler Cache-Manager für einfachen Zugriff
global_tensor_cache_manager = TensorCacheManager()
