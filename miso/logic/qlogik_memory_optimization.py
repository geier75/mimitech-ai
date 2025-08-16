#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK Memory Optimization

Speicheroptimierungsmodul für Q-LOGIK.
Implementiert Lazy Loading, Cache-Strategien und Gradient Checkpointing.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
import threading
from datetime import datetime
import weakref
import functools
import pickle
from pathlib import Path

# Logger einrichten
logger = logging.getLogger("MISO.Logic.Q-LOGIK.MemoryOptimization")

class LRUCache:
    """
    LRU (Least Recently Used) Cache-Implementierung
    """
    
    def __init__(self, capacity: int = 100):
        """
        Initialisiert den LRU-Cache
        
        Args:
            capacity: Maximale Kapazität des Caches
        """
        self.capacity = capacity
        self.cache = {}
        self.usage_order = []
    
    def get(self, key: Any) -> Any:
        """
        Holt einen Wert aus dem Cache
        
        Args:
            key: Schlüssel des Werts
            
        Returns:
            Wert oder None, falls nicht im Cache
        """
        if key not in self.cache:
            return None
            
        # Aktualisiere Nutzungsreihenfolge
        self.usage_order.remove(key)
        self.usage_order.append(key)
        
        return self.cache[key]
    
    def put(self, key: Any, value: Any) -> None:
        """
        Fügt einen Wert zum Cache hinzu
        
        Args:
            key: Schlüssel des Werts
            value: Zu speichernder Wert
        """
        if key in self.cache:
            # Aktualisiere Nutzungsreihenfolge
            self.usage_order.remove(key)
        elif len(self.cache) >= self.capacity:
            # Entferne am längsten nicht verwendeten Eintrag
            oldest_key = self.usage_order.pop(0)
            del self.cache[oldest_key]
        
        # Füge neuen Eintrag hinzu
        self.cache[key] = value
        self.usage_order.append(key)
    
    def clear(self) -> None:
        """Leert den Cache"""
        self.cache = {}
        self.usage_order = []
    
    def __len__(self) -> int:
        """Gibt die aktuelle Größe des Caches zurück"""
        return len(self.cache)


class LazyLoader:
    """
    Lazy-Loading-Implementierung für ressourcenintensive Objekte
    """
    
    def __init__(self, loader_func: Callable, *args, **kwargs):
        """
        Initialisiert den LazyLoader
        
        Args:
            loader_func: Funktion zum Laden des Objekts
            *args: Argumente für die Ladefunktion
            **kwargs: Keyword-Argumente für die Ladefunktion
        """
        self.loader_func = loader_func
        self.args = args
        self.kwargs = kwargs
        self._instance = None
        self._loaded = False
    
    def __call__(self):
        """
        Lädt das Objekt bei Bedarf und gibt es zurück
        
        Returns:
            Geladenes Objekt
        """
        if not self._loaded:
            self._instance = self.loader_func(*self.args, **self.kwargs)
            self._loaded = True
        return self._instance
    
    @property
    def is_loaded(self) -> bool:
        """Gibt zurück, ob das Objekt bereits geladen wurde"""
        return self._loaded
    
    def reset(self) -> None:
        """Setzt den Loader zurück, sodass das Objekt bei Bedarf neu geladen wird"""
        self._instance = None
        self._loaded = False


class DiskCache:
    """
    Disk-basierter Cache für große Datenmengen
    """
    
    def __init__(self, cache_dir: str, max_size_mb: int = 1000):
        """
        Initialisiert den DiskCache
        
        Args:
            cache_dir: Verzeichnis für Cache-Dateien
            max_size_mb: Maximale Größe des Caches in MB
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = {"entries": {}, "total_size": 0}
        
        # Erstelle Cache-Verzeichnis, falls es nicht existiert
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Lade Metadaten, falls vorhanden
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Lädt Cache-Metadaten von der Festplatte"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Cache-Metadaten: {e}")
                self.metadata = {"entries": {}, "total_size": 0}
    
    def _save_metadata(self) -> None:
        """Speichert Cache-Metadaten auf der Festplatte"""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Cache-Metadaten: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Gibt den Pfad zur Cache-Datei für einen Schlüssel zurück"""
        # Verwende Hash des Schlüssels als Dateiname
        import hashlib
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Any:
        """
        Holt einen Wert aus dem Cache
        
        Args:
            key: Schlüssel des Werts
            
        Returns:
            Wert oder None, falls nicht im Cache
        """
        if key not in self.metadata["entries"]:
            return None
            
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            # Cache-Datei existiert nicht mehr, entferne Eintrag aus Metadaten
            self._remove_entry(key)
            return None
            
        try:
            with open(cache_path, "rb") as f:
                value = pickle.load(f)
                
            # Aktualisiere Zugriffszeitstempel
            self.metadata["entries"][key]["last_accessed"] = time.time()
            self._save_metadata()
            
            return value
            
        except Exception as e:
            logger.error(f"Fehler beim Laden aus Cache für Schlüssel {key}: {e}")
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """
        Fügt einen Wert zum Cache hinzu
        
        Args:
            key: Schlüssel des Werts
            value: Zu speichernder Wert
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Serialisiere Wert
            cache_path = self._get_cache_path(key)
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
                
            # Ermittle Größe der Cache-Datei
            file_size = os.path.getsize(cache_path)
            
            # Prüfe, ob Cache-Größe überschritten wird
            if key in self.metadata["entries"]:
                # Aktualisiere vorhandenen Eintrag
                old_size = self.metadata["entries"][key]["size"]
                self.metadata["total_size"] -= old_size
            elif self.metadata["total_size"] + file_size > self.max_size_bytes:
                # Entferne älteste Einträge, bis genug Platz frei ist
                self._evict_entries(file_size)
            
            # Aktualisiere Metadaten
            self.metadata["entries"][key] = {
                "size": file_size,
                "created": time.time(),
                "last_accessed": time.time()
            }
            self.metadata["total_size"] += file_size
            
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern im Cache für Schlüssel {key}: {e}")
            return False
    
    def _evict_entries(self, required_space: int) -> None:
        """
        Entfernt älteste Einträge, bis genug Platz frei ist
        
        Args:
            required_space: Benötigter Speicherplatz in Bytes
        """
        # Sortiere Einträge nach letztem Zugriff
        entries = sorted(
            self.metadata["entries"].items(),
            key=lambda x: x[1]["last_accessed"]
        )
        
        # Entferne älteste Einträge, bis genug Platz frei ist
        for key, entry in entries:
            self._remove_entry(key)
            if self.metadata["total_size"] + required_space <= self.max_size_bytes:
                break
    
    def _remove_entry(self, key: str) -> None:
        """
        Entfernt einen Eintrag aus dem Cache
        
        Args:
            key: Schlüssel des zu entfernenden Eintrags
        """
        if key not in self.metadata["entries"]:
            return
            
        # Entferne Cache-Datei
        cache_path = self._get_cache_path(key)
        try:
            if cache_path.exists():
                os.remove(cache_path)
        except Exception as e:
            logger.error(f"Fehler beim Löschen der Cache-Datei für Schlüssel {key}: {e}")
        
        # Aktualisiere Metadaten
        entry_size = self.metadata["entries"][key]["size"]
        self.metadata["total_size"] -= entry_size
        del self.metadata["entries"][key]
        
        self._save_metadata()
    
    def clear(self) -> None:
        """Leert den gesamten Cache"""
        # Entferne alle Cache-Dateien
        for key in list(self.metadata["entries"].keys()):
            self._remove_entry(key)
        
        # Zurücksetzen der Metadaten
        self.metadata = {"entries": {}, "total_size": 0}
        self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über den Cache zurück
        
        Returns:
            Dictionary mit Cache-Statistiken
        """
        return {
            "total_entries": len(self.metadata["entries"]),
            "total_size_mb": self.metadata["total_size"] / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "usage_percent": (self.metadata["total_size"] / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0
        }


class GradientCheckpointer:
    """
    Implementierung von Gradient Checkpointing für speichereffiziente Berechnungen
    """
    
    def __init__(self, checkpoint_interval: int = 5):
        """
        Initialisiert den GradientCheckpointer
        
        Args:
            checkpoint_interval: Intervall zwischen Checkpoints
        """
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints = {}
    
    def checkpoint(self, key: str, value: Any) -> None:
        """
        Speichert einen Checkpoint
        
        Args:
            key: Schlüssel des Checkpoints
            value: Zu speichernder Wert
        """
        self.checkpoints[key] = value
    
    def get_checkpoint(self, key: str) -> Any:
        """
        Gibt einen Checkpoint zurück
        
        Args:
            key: Schlüssel des Checkpoints
            
        Returns:
            Checkpoint-Wert oder None, falls nicht vorhanden
        """
        return self.checkpoints.get(key)
    
    def clear_checkpoints(self) -> None:
        """Löscht alle Checkpoints"""
        self.checkpoints = {}
    
    def checkpoint_function(self, func: Callable) -> Callable:
        """
        Dekorator für Funktionen, die Checkpoints verwenden sollen
        
        Args:
            func: Zu dekorierende Funktion
            
        Returns:
            Dekorierte Funktion
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generiere Checkpoint-Schlüssel
            import hashlib
            key_str = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            key = hashlib.md5(key_str.encode()).hexdigest()
            
            # Prüfe, ob Checkpoint existiert
            checkpoint = self.get_checkpoint(key)
            if checkpoint is not None:
                return checkpoint
            
            # Führe Funktion aus und speichere Checkpoint
            result = func(*args, **kwargs)
            self.checkpoint(key, result)
            
            return result
        
        return wrapper


class MemoryOptimizer:
    """
    Speicheroptimierung für Q-LOGIK
    
    Implementiert Lazy Loading, Cache-Strategien und Gradient Checkpointing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den MemoryOptimizer
        
        Args:
            config: Konfigurationsobjekt für den MemoryOptimizer
        """
        self.config = config or {}
        
        # Konfiguriere Cache-Verzeichnis
        cache_dir = self.config.get("cache_dir", os.path.expanduser("~/.miso/cache"))
        max_cache_size_mb = self.config.get("max_cache_size_mb", 1000)
        
        # Initialisiere Caches
        self.memory_cache = LRUCache(capacity=self.config.get("memory_cache_capacity", 1000))
        self.disk_cache = DiskCache(cache_dir, max_size_mb=max_cache_size_mb)
        
        # Initialisiere Gradient Checkpointer
        self.gradient_checkpointer = GradientCheckpointer(
            checkpoint_interval=self.config.get("checkpoint_interval", 5)
        )
        
        # Lazy-Loading-Registry
        self.lazy_loaders = {}
        
        logger.info(f"MemoryOptimizer initialisiert: Cache-Dir={cache_dir}, Max-Cache-Size={max_cache_size_mb}MB")
    
    def get_from_cache(self, key: str, loader_func: Optional[Callable] = None) -> Any:
        """
        Holt einen Wert aus dem Cache oder lädt ihn bei Bedarf
        
        Args:
            key: Cache-Schlüssel
            loader_func: Funktion zum Laden des Werts, falls nicht im Cache
            
        Returns:
            Wert aus dem Cache oder None
        """
        # Prüfe Memory-Cache
        value = self.memory_cache.get(key)
        if value is not None:
            return value
            
        # Prüfe Disk-Cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Füge Wert auch zum Memory-Cache hinzu
            self.memory_cache.put(key, value)
            return value
            
        # Lade Wert, falls Loader-Funktion angegeben
        if loader_func is not None:
            value = loader_func()
            
            # Speichere Wert im Cache
            self.memory_cache.put(key, value)
            self.disk_cache.put(key, value)
            
            return value
            
        return None
    
    def put_in_cache(self, key: str, value: Any, use_disk_cache: bool = True) -> None:
        """
        Speichert einen Wert im Cache
        
        Args:
            key: Cache-Schlüssel
            value: Zu speichernder Wert
            use_disk_cache: True, um auch im Disk-Cache zu speichern
        """
        self.memory_cache.put(key, value)
        
        if use_disk_cache:
            self.disk_cache.put(key, value)
    
    def clear_cache(self, key: Optional[str] = None) -> None:
        """
        Leert den Cache
        
        Args:
            key: Optionaler Schlüssel, um nur einen bestimmten Eintrag zu löschen
        """
        if key is not None:
            # Entferne nur einen bestimmten Eintrag
            self.memory_cache.put(key, None)  # Überschreibe mit None
            # Entferne aus Disk-Cache
            self.disk_cache._remove_entry(key)
        else:
            # Leere gesamten Cache
            self.memory_cache.clear()
            self.disk_cache.clear()
    
    def register_lazy_loader(self, name: str, loader_func: Callable, *args, **kwargs) -> LazyLoader:
        """
        Registriert einen LazyLoader
        
        Args:
            name: Name des LazyLoaders
            loader_func: Funktion zum Laden des Objekts
            *args: Argumente für die Ladefunktion
            **kwargs: Keyword-Argumente für die Ladefunktion
            
        Returns:
            LazyLoader-Instanz
        """
        loader = LazyLoader(loader_func, *args, **kwargs)
        self.lazy_loaders[name] = loader
        return loader
    
    def get_lazy_loader(self, name: str) -> Optional[LazyLoader]:
        """
        Gibt einen registrierten LazyLoader zurück
        
        Args:
            name: Name des LazyLoaders
            
        Returns:
            LazyLoader-Instanz oder None, falls nicht gefunden
        """
        return self.lazy_loaders.get(name)
    
    def checkpoint(self, key: str, value: Any) -> None:
        """
        Speichert einen Gradient-Checkpoint
        
        Args:
            key: Schlüssel des Checkpoints
            value: Zu speichernder Wert
        """
        self.gradient_checkpointer.checkpoint(key, value)
    
    def get_checkpoint(self, key: str) -> Any:
        """
        Gibt einen Gradient-Checkpoint zurück
        
        Args:
            key: Schlüssel des Checkpoints
            
        Returns:
            Checkpoint-Wert oder None, falls nicht vorhanden
        """
        return self.gradient_checkpointer.get_checkpoint(key)
    
    def checkpoint_function(self, func: Callable) -> Callable:
        """
        Dekorator für Funktionen, die Checkpoints verwenden sollen
        
        Args:
            func: Zu dekorierende Funktion
            
        Returns:
            Dekorierte Funktion
        """
        return self.gradient_checkpointer.checkpoint_function(func)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Gibt Speicherstatistiken zurück
        
        Returns:
            Dictionary mit Speicherstatistiken
        """
        memory_cache_stats = {
            "size": len(self.memory_cache),
            "capacity": self.memory_cache.capacity
        }
        
        disk_cache_stats = self.disk_cache.get_stats()
        
        lazy_loader_stats = {
            "total": len(self.lazy_loaders),
            "loaded": sum(1 for loader in self.lazy_loaders.values() if loader.is_loaded)
        }
        
        checkpoint_stats = {
            "total": len(self.gradient_checkpointer.checkpoints)
        }
        
        return {
            "memory_cache": memory_cache_stats,
            "disk_cache": disk_cache_stats,
            "lazy_loaders": lazy_loader_stats,
            "checkpoints": checkpoint_stats
        }


# Globale Instanz für einfachen Zugriff
memory_optimizer = MemoryOptimizer()

def get_from_cache(key: str, loader_func: Optional[Callable] = None) -> Any:
    """
    Holt einen Wert aus dem Cache oder lädt ihn bei Bedarf
    
    Args:
        key: Cache-Schlüssel
        loader_func: Funktion zum Laden des Werts, falls nicht im Cache
        
    Returns:
        Wert aus dem Cache oder None
    """
    return memory_optimizer.get_from_cache(key, loader_func)

def put_in_cache(key: str, value: Any, use_disk_cache: bool = True, ttl: Optional[int] = None) -> None:
    """
    Speichert einen Wert im Cache
    
    Args:
        key: Cache-Schlüssel
        value: Zu speichernder Wert
        use_disk_cache: True, um auch im Disk-Cache zu speichern
        ttl: Time-to-Live in Sekunden (wird derzeit ignoriert, für API-Kompatibilität)
    """
    # TTL wird derzeit vom MemoryOptimizer nicht unterstützt, aber für API-Kompatibilität akzeptiert
    memory_optimizer.put_in_cache(key, value, use_disk_cache)

def clear_cache(key: Optional[str] = None) -> None:
    """
    Leert den Cache
    
    Args:
        key: Optionaler Schlüssel, um nur einen bestimmten Eintrag zu löschen
    """
    memory_optimizer.clear_cache(key)

def register_lazy_loader(name: str, loader_func: Callable, *args, **kwargs) -> LazyLoader:
    """
    Registriert einen LazyLoader
    
    Args:
        name: Name des LazyLoaders
        loader_func: Funktion zum Laden des Objekts
        *args: Argumente für die Ladefunktion
        **kwargs: Keyword-Argumente für die Ladefunktion
        
    Returns:
        LazyLoader-Instanz
    """
    return memory_optimizer.register_lazy_loader(name, loader_func, *args, **kwargs)

def checkpoint(key: str, value: Any) -> None:
    """
    Speichert einen Gradient-Checkpoint
    
    Args:
        key: Schlüssel des Checkpoints
        value: Zu speichernder Wert
    """
    memory_optimizer.checkpoint(key, value)

def checkpoint_function(func: Callable) -> Callable:
    """
    Dekorator für Funktionen, die Checkpoints verwenden sollen
    
    Args:
        func: Zu dekorierende Funktion
        
    Returns:
        Dekorierte Funktion
    """
    return memory_optimizer.checkpoint_function(func)

def get_memory_stats() -> Dict[str, Any]:
    """
    Gibt Speicherstatistiken zurück
    
    Returns:
        Dictionary mit Speicherstatistiken
    """
    return memory_optimizer.get_memory_stats()


if __name__ == "__main__":
    # Beispiel für die Verwendung des MemoryOptimizers
    logging.basicConfig(level=logging.INFO)
    
    # Beispiel für Cache-Nutzung
    def expensive_computation(x):
        print(f"Führe teure Berechnung für {x} durch...")
        time.sleep(1)  # Simuliere lange Berechnung
        return x * x
    
    # Erste Ausführung (nicht im Cache)
    result1 = get_from_cache("square_10", lambda: expensive_computation(10))
    print(f"Ergebnis 1: {result1}")
    
    # Zweite Ausführung (aus dem Cache)
    result2 = get_from_cache("square_10", lambda: expensive_computation(10))
    print(f"Ergebnis 2: {result2}")
    
    # Beispiel für LazyLoader
    def load_large_model():
        print("Lade großes Modell...")
        time.sleep(2)  # Simuliere langes Laden
        return {"name": "LargeModel", "parameters": 10000000}
    
    # Registriere LazyLoader
    model_loader = register_lazy_loader("large_model", load_large_model)
    print("LazyLoader registriert, Modell noch nicht geladen")
    
    # Lade Modell bei Bedarf
    model = model_loader()
    print(f"Modell geladen: {model}")
    
    # Beispiel für Checkpoint-Funktion
    @checkpoint_function
    def complex_calculation(a, b):
        print(f"Führe komplexe Berechnung für {a}, {b} durch...")
        time.sleep(1)  # Simuliere lange Berechnung
        return a + b
    
    # Erste Ausführung (kein Checkpoint)
    result3 = complex_calculation(5, 7)
    print(f"Komplexe Berechnung 1: {result3}")
    
    # Zweite Ausführung (aus Checkpoint)
    result4 = complex_calculation(5, 7)
    print(f"Komplexe Berechnung 2: {result4}")
    
    # Zeige Speicherstatistiken
    stats = get_memory_stats()
    print("\nSpeicherstatistiken:")
    print(f"Memory-Cache: {stats['memory_cache']['size']}/{stats['memory_cache']['capacity']} Einträge")
    print(f"Disk-Cache: {stats['disk_cache']['total_entries']} Einträge, "
          f"{stats['disk_cache']['total_size_mb']:.2f}/{stats['disk_cache']['max_size_mb']:.2f} MB "
          f"({stats['disk_cache']['usage_percent']:.2f}%)")
    print(f"LazyLoader: {stats['lazy_loaders']['loaded']}/{stats['lazy_loaders']['total']} geladen")
    print(f"Checkpoints: {stats['checkpoints']['total']} gespeichert")
