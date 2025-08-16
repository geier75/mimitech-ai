#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensor-Pool-Implementierung für T-Mathematics Engine

Diese Datei implementiert einen Speicherpool für die Wiederverwendung von Tensoren,
um unnötige Speicherallokationen zu vermeiden und die Leistung zu verbessern.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import logging
import torch
import numpy as np

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.tensor_pool")

# MLX-Verfügbarkeit prüfen
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    logger.warning("MLX konnte nicht importiert werden. MLX-Tensorpooling ist nicht verfügbar.")

class TensorPool:
    """
    Implementiert einen Speicherpool für die Wiederverwendung von Tensoren.
    Reduziert Speicherallokationen für temporäre Tensoren.
    """
    def __init__(self, max_pooled_tensors=100, max_memory_mb=2048):
        """
        Initialisiert den TensorPool
        
        Args:
            max_pooled_tensors: Maximale Anzahl von gepoolten Tensoren pro Kategorie
            max_memory_mb: Maximale Speichernutzung in MB
        """
        self.pools = {
            'torch_cpu': {},  # PyTorch CPU-Tensoren
            'torch_mps': {},  # PyTorch MPS-Tensoren
            'mlx': {},        # MLX-Arrays
            'numpy': {}       # NumPy-Arrays
        }
        self.max_pooled_tensors = max_pooled_tensors
        self.max_memory_mb = max_memory_mb
        self.current_memory_usage = 0  # In Bytes
        self.stats = {
            'hits': 0,
            'misses': 0,
            'purges': 0,
            'total_allocations': 0,
            'total_deallocations': 0,
            'peak_memory': 0
        }
        logger.info(f"TensorPool initialisiert mit max_pooled_tensors={max_pooled_tensors}, "
                   f"max_memory_mb={max_memory_mb}")
    
    def _get_pool_key(self, shape, dtype, device_type):
        """
        Generiert einen Schlüssel für den Pool basierend auf Tensor-Eigenschaften
        
        Args:
            shape: Tensorform als Tuple
            dtype: Datentyp
            device_type: Gerätetyp ('cpu', 'mps', 'mlx', 'numpy')
            
        Returns:
            Pool-Schlüssel als Tupel
        """
        return (tuple(shape), str(dtype), device_type)
    
    def _estimate_tensor_size(self, shape, dtype):
        """
        Schätzt die Größe eines Tensors in Bytes
        
        Args:
            shape: Tensorform
            dtype: Datentyp
            
        Returns:
            Geschätzte Größe in Bytes
        """
        # Bestimme die Größe eines einzelnen Elements
        element_size = {
            'float16': 2,
            'float32': 4,
            'float64': 8,
            'int8': 1,
            'int16': 2,
            'int32': 4,
            'int64': 8,
            'uint8': 1,
            'uint16': 2,
            'uint32': 4,
            'uint64': 8,
            'bool': 1
        }.get(str(dtype).split('.')[-1], 4)  # Default 4 Bytes
        
        # Berechne Gesamtgröße
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        
        return num_elements * element_size
    
    def get(self, shape, dtype=None, device=None):
        """
        Holt einen Tensor aus dem Pool oder erstellt einen neuen
        
        Args:
            shape: Tensorform
            dtype: Datentyp (Default: None)
            device: Gerät (Default: None)
            
        Returns:
            Tensor mit den angegebenen Eigenschaften
        """
        # Bestimme den Pool und Schlüssel basierend auf dem Gerätetyp
        if device is not None and hasattr(device, 'type') and device.type == 'mps':
            pool_type = 'torch_mps'
            device_type = 'mps'
        elif device is not None and str(device) == 'mlx':
            pool_type = 'mlx'
            device_type = 'mlx'
        elif dtype == 'numpy':
            pool_type = 'numpy'
            device_type = 'numpy'
        else:
            pool_type = 'torch_cpu'
            device_type = 'cpu'
        
        # Generiere Schlüssel
        key = self._get_pool_key(shape, dtype, device_type)
        
        # Versuche, einen gepoolten Tensor zu holen
        if pool_type in self.pools and key in self.pools[pool_type] and self.pools[pool_type][key]:
            # Pool-Hit
            self.stats['hits'] += 1
            tensor = self.pools[pool_type][key].pop()
            
            # Reduziere den aktuellen Speicherverbrauch
            tensor_size = self._estimate_tensor_size(shape, dtype)
            self.current_memory_usage = max(0, self.current_memory_usage - tensor_size)
            
            # Nullsetzen des Tensors für numerische Stabilität
            if pool_type.startswith('torch'):
                tensor.zero_()
            elif pool_type == 'mlx':
                tensor = mx.zeros_like(tensor)
            else:  # numpy
                tensor.fill(0)
            
            return tensor
        
        # Pool-Miss: Erstelle einen neuen Tensor
        self.stats['misses'] += 1
        self.stats['total_allocations'] += 1
        
        if pool_type == 'torch_mps':
            return torch.zeros(shape, dtype=dtype, device=torch.device('mps'))
        elif pool_type == 'torch_cpu':
            return torch.zeros(shape, dtype=dtype, device=torch.device('cpu'))
        elif pool_type == 'mlx' and HAS_MLX:
            mlx_dtype = self._convert_to_mlx_dtype(dtype)
            return mx.zeros(shape, dtype=mlx_dtype)
        else:  # numpy
            numpy_dtype = self._convert_to_numpy_dtype(dtype)
            return np.zeros(shape, dtype=numpy_dtype)
    
    def put(self, tensor):
        """
        Gibt einen Tensor zurück in den Pool
        
        Args:
            tensor: Der zurückzugebende Tensor
        """
        if tensor is None:
            return
        
        # Bestimme Tensortyp und Eigenschaften
        if isinstance(tensor, torch.Tensor):
            if tensor.device.type == 'mps':
                pool_type = 'torch_mps'
                device_type = 'mps'
            else:
                pool_type = 'torch_cpu'
                device_type = 'cpu'
            shape = tensor.shape
            dtype = tensor.dtype
        elif HAS_MLX and isinstance(tensor, mx.array):
            pool_type = 'mlx'
            device_type = 'mlx'
            shape = tensor.shape
            dtype = tensor.dtype
        elif isinstance(tensor, np.ndarray):
            pool_type = 'numpy'
            device_type = 'numpy'
            shape = tensor.shape
            dtype = tensor.dtype
        else:
            logger.warning(f"Unbekannter Tensortyp kann nicht gepooled werden: {type(tensor)}")
            return
        
        # Generiere Schlüssel
        key = self._get_pool_key(shape, dtype, device_type)
        
        # Überprüfe, ob der Pool existiert
        if pool_type not in self.pools:
            self.pools[pool_type] = {}
        
        # Erstelle einen Pool für diesen Tensortyp, wenn er nicht existiert
        if key not in self.pools[pool_type]:
            self.pools[pool_type][key] = []
        
        # Überprüfe, ob der Pool voll ist
        if len(self.pools[pool_type][key]) >= self.max_pooled_tensors:
            self.stats['purges'] += 1
            return  # Pool ist voll, verwerfe den Tensor
        
        # Schätze die Größe des Tensors
        tensor_size = self._estimate_tensor_size(shape, dtype)
        
        # Überprüfe, ob das Speicherlimit erreicht ist
        if self.current_memory_usage + tensor_size > self.max_memory_mb * 1024 * 1024:
            # Speicherlimit erreicht, bereinige den Pool
            self._purge_tensors()
            
            # Überprüfe erneut, ob genug Platz ist
            if self.current_memory_usage + tensor_size > self.max_memory_mb * 1024 * 1024:
                self.stats['purges'] += 1
                return  # Immer noch nicht genug Platz, verwerfe den Tensor
        
        # Füge den Tensor zum Pool hinzu
        self.pools[pool_type][key].append(tensor)
        self.current_memory_usage += tensor_size
        self.stats['total_deallocations'] += 1
        
        # Aktualisiere Peak-Speichernutzung
        if self.current_memory_usage > self.stats['peak_memory']:
            self.stats['peak_memory'] = self.current_memory_usage
    
    def _purge_tensors(self, purge_percentage=0.2):
        """
        Bereinigt den Pool, wenn das Speicherlimit erreicht ist
        
        Args:
            purge_percentage: Prozentsatz der zu bereinigenden Tensoren
        """
        # Berechne, wie viel Speicher freigegeben werden muss
        target_reduction = int(self.current_memory_usage * purge_percentage)
        freed_memory = 0
        
        logger.debug(f"Bereinige TensorPool um {purge_percentage*100}% ({target_reduction} Bytes)")
        
        # Iteriere über alle Pools und entferne Tensoren
        for pool_type in self.pools:
            for key in list(self.pools[pool_type].keys()):
                if not self.pools[pool_type][key]:
                    continue
                
                # Extrahiere Form und Datentyp aus dem Schlüssel
                shape, dtype, _ = key
                
                # Schätze die Größe eines Tensors
                tensor_size = self._estimate_tensor_size(shape, dtype)
                
                # Entferne Tensoren, bis genug Speicher freigegeben ist
                while self.pools[pool_type][key] and freed_memory < target_reduction:
                    self.pools[pool_type][key].pop()
                    freed_memory += tensor_size
                    self.stats['purges'] += 1
                
                if freed_memory >= target_reduction:
                    break
            
            if freed_memory >= target_reduction:
                break
        
        # Aktualisiere den aktuellen Speicherverbrauch
        self.current_memory_usage = max(0, self.current_memory_usage - freed_memory)
        logger.debug(f"TensorPool bereinigt: {freed_memory} Bytes freigegeben")
    
    def clear(self):
        """Leert den gesamten Pool"""
        self.pools = {
            'torch_cpu': {},
            'torch_mps': {},
            'mlx': {},
            'numpy': {}
        }
        self.current_memory_usage = 0
        logger.info("TensorPool geleert")
    
    def get_stats(self):
        """
        Liefert Statistiken über die Pool-Nutzung
        
        Returns:
            Dictionary mit Statistiken
        """
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_ratio': self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0,
            'purges': self.stats['purges'],
            'total_allocations': self.stats['total_allocations'],
            'total_deallocations': self.stats['total_deallocations'],
            'current_memory_mb': self.current_memory_usage / (1024 * 1024),
            'peak_memory_mb': self.stats['peak_memory'] / (1024 * 1024),
            'pool_size': sum(len(tensors) for pool in self.pools.values() for tensors in pool.values())
        }
    
    def _convert_to_mlx_dtype(self, dtype):
        """
        Konvertiert einen Datentyp zu einem MLX-Datentyp
        
        Args:
            dtype: Ursprungsdatentyp
            
        Returns:
            MLX-Datentyp
        """
        if not HAS_MLX:
            return None
        
        dtype_str = str(dtype).split('.')[-1]  # z.B. 'float32'
        
        # Wähle korrekten MLX-Datentyp
        return {
            'float16': mx.float16,
            'float32': mx.float32,
            'bfloat16': mx.bfloat16 if hasattr(mx, 'bfloat16') else mx.float16,
            'int32': mx.int32,
            'int64': mx.int32,  # MLX hat kein int64, fallback zu int32
            'bool': mx.bool_,
        }.get(dtype_str, mx.float32)
    
    def _convert_to_numpy_dtype(self, dtype):
        """
        Konvertiert einen Datentyp zu einem NumPy-Datentyp
        
        Args:
            dtype: Ursprungsdatentyp
            
        Returns:
            NumPy-Datentyp
        """
        dtype_str = str(dtype).split('.')[-1]  # z.B. 'float32'
        
        # Wähle korrekten NumPy-Datentyp
        return {
            'float16': np.float16,
            'float32': np.float32,
            'float64': np.float64,
            'int32': np.int32,
            'int64': np.int64,
            'bool': np.bool_,
        }.get(dtype_str, np.float32)
