#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION Memory Manager

Dieser Speichermanager optimiert die Speichernutzung für Batch-Verarbeitungen,
indem er Puffer wiederverwendet, Speicherfragmentierung reduziert und
intelligente Präallokationsstrategien verwendet.
"""

import logging
import sys
import gc
import threading
import weakref
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# Logger konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VX-VISION.memory")


class MemoryManager:
    """
    Speichermanager für effiziente Batch-Verarbeitung.
    
    Diese Klasse verwaltet die Speicherzuweisungen und -freigaben während der
    Batch-Verarbeitung, um Speichereffizienz zu maximieren und Fragmentierung zu minimieren.
    """
    
    def __init__(self, max_memory_usage: float = 0.7, hardware_detector=None):
        """
        Initialisiert den MemoryManager.
        
        Args:
            max_memory_usage: Maximaler Anteil des verfügbaren Speichers, der verwendet werden darf (0-1)
            hardware_detector: Instanz des HardwareDetector
        """
        self.max_memory_usage = max_memory_usage
        self.hardware = hardware_detector
        
        # Speicherpools für verschiedene Tensor-Formen und -Typen
        self.buffer_pools = {}
        
        # Aktive Speicherzuweisungen
        self.active_allocations = {}
        self.allocation_id_counter = 0
        
        # Threading-Sicherheit
        self.lock = threading.RLock()
        
        # Letzter überwachter Speicherstatus
        self.last_memory_check = {
            'free': 0,
            'used': 0,
            'total': 0
        }
        
        # Registerung beim Garbage Collector für proaktive Bereinigung
        self._register_gc_callback()
        
        logger.info(f"MemoryManager initialisiert mit max_memory_usage={max_memory_usage:.2f}")
        
    def prepare_batches(self, images: List[np.ndarray], batch_size: int) -> List[List[np.ndarray]]:
        """
        Bereitet Bilder für die Batch-Verarbeitung vor und organisiert den Speicher effizient.
        
        Args:
            images: Liste von Bildern als NumPy-Arrays
            batch_size: Größe jedes Batches
            
        Returns:
            Liste von Bild-Batches
        """
        # Einfache Aufteilung in Batches der spezifizierten Größe
        batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]
        
        # Memory-Pre-Checks durchführen
        self._update_memory_status()
        
        # Speichernutzung für diese Batches registrieren
        with self.lock:
            for batch_idx, batch in enumerate(batches):
                allocation_id = self._next_allocation_id()
                self.active_allocations[allocation_id] = {
                    'batch_idx': batch_idx,
                    'size': sum(self._estimate_image_size(img) for img in batch),
                    'ref': weakref.ref(batch)
                }
        
        logger.debug(f"Batch-Verarbeitung vorbereitet: {len(batches)} Batches, " +
                    f"Gesamt: {len(images)} Bilder, Batch-Größe: {batch_size}")
        
        return batches
        
    def allocate_tensor(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """
        Allokiert einen Tensor (NumPy-Array) mit der angegebenen Form und dem angegebenen Typ,
        vorzugsweise aus einem Speicherpool.
        
        Args:
            shape: Form des Tensors
            dtype: Datentyp des Tensors
            
        Returns:
            NumPy-Array
        """
        # Schlüssel für den Speicherpool
        pool_key = (shape, dtype)
        
        with self.lock:
            # Prüfen, ob ein passender Puffer im Pool verfügbar ist
            if pool_key in self.buffer_pools and self.buffer_pools[pool_key]:
                buffer = self.buffer_pools[pool_key].pop()
                logger.debug(f"Wiederverwendeter Puffer aus Pool: {shape}, {dtype}")
                return buffer
            
            # Wenn nicht im Pool, neuen Puffer allokieren
            buffer = np.zeros(shape, dtype=dtype)
            
            # Speichernutzung registrieren
            allocation_id = self._next_allocation_id()
            self.active_allocations[allocation_id] = {
                'size': buffer.nbytes,
                'ref': weakref.ref(buffer)
            }
            
            logger.debug(f"Neuer Tensor allokiert: {shape}, {dtype}, Größe: {buffer.nbytes / 1024:.2f} KB")
            
            return buffer
            
    def release_tensor(self, tensor: np.ndarray, return_to_pool: bool = True):
        """
        Gibt einen Tensor frei und führt ihn optional dem Speicherpool zu.
        
        Args:
            tensor: Freizugebender Tensor
            return_to_pool: Wenn True, wird der Tensor in den Pool zurückgegeben
        """
        if tensor is None:
            return
            
        # Pool-Schlüssel
        pool_key = (tensor.shape, tensor.dtype)
        
        with self.lock:
            if return_to_pool:
                # Zum Pool hinzufügen
                if pool_key not in self.buffer_pools:
                    self.buffer_pools[pool_key] = []
                    
                # Pool-Größe begrenzen (maximal 10 Puffer pro Form/Typ)
                if len(self.buffer_pools[pool_key]) < 10:
                    self.buffer_pools[pool_key].append(tensor)
                    logger.debug(f"Tensor zum Pool hinzugefügt: {tensor.shape}, {tensor.dtype}")
                    
            # Speichernutzung aktualisieren
            # Hier könnten wir den aktiven Allokationen-Eintrag entfernen,
            # aber es ist schwierig, den spezifischen Tensor zu identifizieren
            # ohne zusätzliche Metadaten. Die weakref-Callbackd werden sich darum kümmern.
            
    def release_batch_memory(self, batch_idx: Optional[int] = None):
        """
        Gibt den Speicher frei, der mit einem bestimmten Batch verbunden ist.
        
        Args:
            batch_idx: Index des Batches oder None für alle aktiven Batches
        """
        to_release = []
        
        with self.lock:
            for alloc_id, alloc_info in self.active_allocations.items():
                if batch_idx is None or alloc_info.get('batch_idx') == batch_idx:
                    to_release.append(alloc_id)
                    
            for alloc_id in to_release:
                if alloc_id in self.active_allocations:
                    del self.active_allocations[alloc_id]
                    
        if batch_idx is not None:
            logger.debug(f"Speicher für Batch {batch_idx} freigegeben: {len(to_release)} Allokationen")
        else:
            logger.debug(f"Alle Batch-Speicher freigegeben: {len(to_release)} Allokationen")
            
        # Prüfen, ob eine Garbage Collection hilfreich sein könnte
        if len(to_release) > 10:
            gc.collect()
            
    def clean_pools(self, force: bool = False):
        """
        Bereinigt die Speicherpools, um Speicher freizugeben.
        
        Args:
            force: Wenn True, werden alle Pools komplett geleert
        """
        total_freed = 0
        
        with self.lock:
            if force:
                # Alle Puffer aus allen Pools entfernen
                for pool_key, pool in self.buffer_pools.items():
                    shape, dtype = pool_key
                    buffer_size = np.prod(shape) * np.dtype(dtype).itemsize
                    total_freed += len(pool) * buffer_size
                    pool.clear()
                    
                # Dictionary leeren
                self.buffer_pools.clear()
            else:
                # Selektiv bereinigen, ältere/ungenutzte Puffer entfernen
                for pool_key, pool in list(self.buffer_pools.items()):
                    shape, dtype = pool_key
                    buffer_size = np.prod(shape) * np.dtype(dtype).itemsize
                    
                    # Pool halbieren, wenn er zu groß ist
                    if len(pool) > 5:
                        removed_count = len(pool) // 2
                        for _ in range(removed_count):
                            pool.pop(0)  # Älteste zuerst entfernen
                        total_freed += removed_count * buffer_size
                        
                    # Leere Pools entfernen
                    if not pool:
                        del self.buffer_pools[pool_key]
                        
        logger.info(f"Speicherpools bereinigt, {total_freed / (1024*1024):.2f} MB freigegeben")
        
    def check_memory_pressure(self) -> bool:
        """
        Prüft, ob Speicherdruck besteht und bereinigt Pools bei Bedarf.
        
        Returns:
            bool: True, wenn Speicherdruck besteht
        """
        self._update_memory_status()
        
        free_memory = self.last_memory_check['free']
        total_memory = self.last_memory_check['total']
        
        # Speicherdruck berechnen
        memory_pressure = 1.0 - (free_memory / total_memory)
        
        if memory_pressure > self.max_memory_usage:
            # Speicherdruck reduzieren
            logger.warning(f"Hoher Speicherdruck erkannt: {memory_pressure:.2f}, " +
                         f"aktiviere Pool-Bereinigung")
            self.clean_pools(force=memory_pressure > 0.9)
            gc.collect()
            return True
            
        return False
        
    def _estimate_image_size(self, image: np.ndarray) -> int:
        """
        Schätzt die Speichergröße eines Bildes in Bytes.
        
        Args:
            image: Bild als NumPy-Array
            
        Returns:
            int: Geschätzte Speichergröße in Bytes
        """
        if image is None or not hasattr(image, 'nbytes'):
            return 0
            
        # Direktes Auslesen der Speichergröße
        return image.nbytes
        
    def _update_memory_status(self):
        """Aktualisiert den Speicherstatus des Systems."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            self.last_memory_check = {
                'free': memory.available,
                'used': memory.used,
                'total': memory.total
            }
            
        except ImportError:
            # Fallback, wenn psutil nicht verfügbar ist
            self.last_memory_check = {
                'free': 1024 * 1024 * 1024,  # 1 GB als Annahme
                'used': 0,
                'total': 2 * 1024 * 1024 * 1024  # 2 GB als Annahme
            }
            
        logger.debug(f"Speicherstatus: Frei={self.last_memory_check['free'] / (1024*1024*1024):.2f} GB, " +
                   f"Gesamt={self.last_memory_check['total'] / (1024*1024*1024):.2f} GB")
        
    def _next_allocation_id(self) -> int:
        """Generiert eine eindeutige ID für Speicherzuweisungen."""
        self.allocation_id_counter += 1
        return self.allocation_id_counter
        
    def _register_gc_callback(self):
        """Registriert einen Callback für Garbage Collection Events."""
        # Diese Funktion überwacht freigegebene Objekte
        def _finalize_callback(alloc_id):
            with self.lock:
                if alloc_id in self.active_allocations:
                    del self.active_allocations[alloc_id]
                    logger.debug(f"Speicherallokation {alloc_id} durch GC freigegeben")
                    
        # Garbage Collection für alle aktiven Allokationen einrichten
        def _setup_finalizers():
            with self.lock:
                for alloc_id, alloc_info in list(self.active_allocations.items()):
                    obj = alloc_info.get('ref')()
                    if obj is None:
                        # Objekt bereits freigegeben
                        del self.active_allocations[alloc_id]
                    else:
                        weakref.finalize(obj, _finalize_callback, alloc_id)
                        
        # Initial einmal ausführen
        _setup_finalizers()
        
        # Regelmäßige Überprüfung durch Timer
        def _check_memory_periodically():
            _setup_finalizers()
            if self.check_memory_pressure():
                logger.info("Speicherdruck erkannt, Garbage Collection durchgeführt")
                
            # Timer neu starten
            t = threading.Timer(30.0, _check_memory_periodically)
            t.daemon = True
            t.start()
            
        # Timer starten
        t = threading.Timer(30.0, _check_memory_periodically)
        t.daemon = True
        t.start()


if __name__ == "__main__":
    # Einfacher Test des MemoryManagers
    memory_manager = MemoryManager(max_memory_usage=0.8)
    
    # Beispielbilder erstellen
    test_images = [np.random.rand(480, 640, 3) for _ in range(20)]
    
    # Batches erstellen
    batches = memory_manager.prepare_batches(test_images, batch_size=8)
    print(f"Erstellt {len(batches)} Batches aus {len(test_images)} Bildern")
    
    # Tensor-Allokation testen
    tensor = memory_manager.allocate_tensor((256, 256, 3), dtype=np.float32)
    print(f"Tensor allokiert: {tensor.shape}, {tensor.dtype}")
    
    # Tensor freigeben
    memory_manager.release_tensor(tensor)
    print("Tensor freigegeben und in den Pool zurückgeführt")
    
    # Batch-Speicher freigeben
    memory_manager.release_batch_memory()
    print("Batch-Speicher freigegeben")
    
    # Pools bereinigen
    memory_manager.clean_pools(force=True)
    print("Speicherpools vollständig bereinigt")
