#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK GPU Acceleration

GPU-Beschleunigungsmodul für Q-LOGIK.
Implementiert CUDA-Kernels, Tensor-Optimierung und parallele Ausführung.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import threading
import contextlib
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Importiere PyTorch für GPU-Beschleunigung
import torch

# Importiere MLX für Apple Silicon Optimierung (wenn verfügbar)
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Logger einrichten
logger = logging.getLogger("MISO.Logic.Q-LOGIK.GPUAcceleration")

class GPUAccelerator:
    """
    GPU-Beschleuniger für Q-LOGIK
    
    Implementiert CUDA-Kernels, Tensor-Optimierung und parallele Ausführung.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den GPUAccelerator
        
        Args:
            config: Konfigurationsobjekt für den GPUAccelerator
        """
        self.config = config or {}
        
        # Prüfe GPU-Verfügbarkeit
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")
        self.num_gpus = torch.cuda.device_count() if self.cuda_available else 0
        
        # Prüfe MLX-Verfügbarkeit (für Apple Silicon)
        self.mlx_available = HAS_MLX
        
        # Konfiguriere Backends
        self.preferred_backend = self.config.get("preferred_backend", "auto")
        if self.preferred_backend == "auto":
            if self.cuda_available:
                self.preferred_backend = "cuda"
            elif self.mlx_available:
                self.preferred_backend = "mlx"
            else:
                self.preferred_backend = "cpu"
        
        # Konfiguriere parallele Ausführung
        self.max_threads = self.config.get("max_threads", min(32, os.cpu_count() or 4))
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_threads)
        
        # Konfiguriere Prozess-Pool für CPU-intensive Operationen
        self.max_processes = self.config.get("max_processes", min(8, os.cpu_count() or 2))
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_processes)
        
        # Cache für optimierte Kernel
        self.kernel_cache = {}
        
        # Initialisiere Backend
        self._initialize_backend()
        
        logger.info(f"GPUAccelerator initialisiert: Backend={self.preferred_backend}, "
                   f"CUDA={self.cuda_available} ({self.num_gpus} GPUs), MLX={self.mlx_available}")
    
    def _initialize_backend(self) -> None:
        """Initialisiert das ausgewählte Backend"""
        if self.preferred_backend == "cuda" and self.cuda_available:
            # Initialisiere CUDA-Backend
            torch.backends.cudnn.benchmark = True
            logger.info(f"CUDA-Backend initialisiert: {torch.cuda.get_device_name(0)}")
            
            # Kompiliere CUDA-Kernels
            self._compile_cuda_kernels()
            
        elif self.preferred_backend == "mlx" and self.mlx_available:
            # Initialisiere MLX-Backend
            logger.info("MLX-Backend für Apple Silicon initialisiert")
            
        else:
            # Fallback auf CPU
            logger.info("CPU-Backend initialisiert")
    
    def _compile_cuda_kernels(self) -> None:
        """Kompiliert optimierte CUDA-Kernels"""
        if not self.cuda_available:
            return
            
        try:
            # Definiere CUDA-Kernels mit torch.jit
            # Matrix-Multiplikation
            @torch.jit.script
            def optimized_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.matmul(a, b)
            
            # Aktivierungsfunktionen
            @torch.jit.script
            def optimized_relu(x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x)
                
            @torch.jit.script
            def optimized_sigmoid(x: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(x)
                
            @torch.jit.script
            def optimized_tanh(x: torch.Tensor) -> torch.Tensor:
                return torch.tanh(x)
            
            # Softmax mit numerischer Stabilität
            @torch.jit.script
            def optimized_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
                return torch.softmax(x, dim=dim)
            
            # Attention-Mechanismus
            @torch.jit.script
            def optimized_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                                   mask: Optional[torch.Tensor] = None, scale: float = 1.0) -> torch.Tensor:
                scores = torch.matmul(query, key.transpose(-2, -1)) * scale
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                attention_weights = torch.softmax(scores, dim=-1)
                return torch.matmul(attention_weights, value)
            
            # Speichere kompilierte Kernels im Cache
            self.kernel_cache["matmul"] = optimized_matmul
            self.kernel_cache["relu"] = optimized_relu
            self.kernel_cache["sigmoid"] = optimized_sigmoid
            self.kernel_cache["tanh"] = optimized_tanh
            self.kernel_cache["softmax"] = optimized_softmax
            self.kernel_cache["attention"] = optimized_attention
            
            logger.info("CUDA-Kernels erfolgreich kompiliert")
            
        except Exception as e:
            logger.error(f"Fehler beim Kompilieren der CUDA-Kernels: {e}")
    
    def to_tensor(self, data: Any, dtype: Optional[Any] = None) -> Any:
        """
        Konvertiert Daten in einen Tensor des aktiven Backends
        
        Args:
            data: Zu konvertierende Daten (numpy.ndarray, Liste, etc.)
            dtype: Datentyp für den Tensor (optional)
            
        Returns:
            Tensor im aktiven Backend
        """
        if self.preferred_backend == "cuda" and self.cuda_available:
            # Konvertiere zu PyTorch-Tensor auf GPU
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            elif isinstance(data, np.ndarray):
                return torch.tensor(data, dtype=dtype, device=self.device)
            else:
                return torch.tensor(data, dtype=dtype, device=self.device)
                
        elif self.preferred_backend == "mlx" and self.mlx_available:
            # Konvertiere zu MLX-Tensor
            if isinstance(data, mx.array):
                return data
            elif isinstance(data, torch.Tensor):
                return mx.array(data.detach().cpu().numpy())
            elif isinstance(data, np.ndarray):
                return mx.array(data)
            else:
                return mx.array(np.array(data))
                
        else:
            # Fallback auf PyTorch-Tensor auf CPU
            if isinstance(data, torch.Tensor):
                return data.cpu()
            elif isinstance(data, np.ndarray):
                return torch.tensor(data, dtype=dtype)
            else:
                return torch.tensor(data, dtype=dtype)
    
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Konvertiert einen Tensor zu numpy.ndarray
        
        Args:
            tensor: Zu konvertierender Tensor
            
        Returns:
            numpy.ndarray
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        elif self.mlx_available and isinstance(tensor, mx.array):
            return np.array(tensor)
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.array(tensor)
    
    def matmul(self, a: Any, b: Any) -> Any:
        """
        Optimierte Matrix-Multiplikation
        
        Args:
            a: Erste Matrix
            b: Zweite Matrix
            
        Returns:
            Ergebnis der Matrix-Multiplikation
        """
        if self.preferred_backend == "cuda" and self.cuda_available:
            # Verwende optimierten CUDA-Kernel
            a_tensor = self.to_tensor(a)
            b_tensor = self.to_tensor(b)
            return self.kernel_cache["matmul"](a_tensor, b_tensor)
            
        elif self.preferred_backend == "mlx" and self.mlx_available:
            # Verwende MLX-Implementierung
            a_tensor = self.to_tensor(a)
            b_tensor = self.to_tensor(b)
            return mx.matmul(a_tensor, b_tensor)
            
        else:
            # Fallback auf NumPy
            a_np = self.to_numpy(a)
            b_np = self.to_numpy(b)
            return np.matmul(a_np, b_np)
    
    def attention(self, query: Any, key: Any, value: Any, mask: Any = None, scale: float = 1.0) -> Any:
        """
        Optimierter Attention-Mechanismus
        
        Args:
            query: Query-Tensor
            key: Key-Tensor
            value: Value-Tensor
            mask: Optionale Maske
            scale: Skalierungsfaktor
            
        Returns:
            Ergebnis des Attention-Mechanismus
        """
        if self.preferred_backend == "cuda" and self.cuda_available:
            # Verwende optimierten CUDA-Kernel
            q_tensor = self.to_tensor(query)
            k_tensor = self.to_tensor(key)
            v_tensor = self.to_tensor(value)
            mask_tensor = self.to_tensor(mask) if mask is not None else None
            return self.kernel_cache["attention"](q_tensor, k_tensor, v_tensor, mask_tensor, scale)
            
        elif self.preferred_backend == "mlx" and self.mlx_available:
            # Implementiere Attention mit MLX
            q_tensor = self.to_tensor(query)
            k_tensor = self.to_tensor(key)
            v_tensor = self.to_tensor(value)
            
            scores = mx.matmul(q_tensor, mx.transpose(k_tensor, axes=(-2, -1))) * scale
            if mask is not None:
                mask_tensor = self.to_tensor(mask)
                scores = mx.where(mask_tensor == 0, mx.array(-1e9), scores)
            
            attention_weights = mx.softmax(scores, axis=-1)
            return mx.matmul(attention_weights, v_tensor)
            
        else:
            # Fallback auf NumPy
            q_np = self.to_numpy(query)
            k_np = self.to_numpy(key)
            v_np = self.to_numpy(value)
            
            scores = np.matmul(q_np, np.transpose(k_np, axes=(0, 2, 1))) * scale
            if mask is not None:
                mask_np = self.to_numpy(mask)
                scores = np.where(mask_np == 0, -1e9, scores)
            
            attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
            return np.matmul(attention_weights, v_np)
    
    def parallel_map(self, func: Callable, items: List[Any], use_processes: bool = False) -> List[Any]:
        """
        Führt eine Funktion parallel auf einer Liste von Elementen aus
        
        Args:
            func: Auszuführende Funktion
            items: Liste von Elementen
            use_processes: True für Prozess-basierte Parallelisierung, False für Thread-basierte
            
        Returns:
            Liste mit Ergebnissen
        """
        if use_processes:
            # Verwende Prozess-Pool für CPU-intensive Operationen
            executor = self.process_pool
        else:
            # Verwende Thread-Pool für I/O-gebundene Operationen
            executor = self.thread_pool
        
        return list(executor.map(func, items))
    
    def batch_process(self, func: Callable, items: List[Any], batch_size: int = 32) -> List[Any]:
        """
        Verarbeitet Elemente in Batches
        
        Args:
            func: Auszuführende Funktion (erwartet einen Batch)
            items: Liste von Elementen
            batch_size: Größe der Batches
            
        Returns:
            Liste mit Ergebnissen
        """
        results = []
        
        # Teile Elemente in Batches auf
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            batch_result = func(batch)
            
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das aktive Backend zurück
        
        Returns:
            Dictionary mit Backend-Informationen
        """
        info = {
            "backend": self.preferred_backend,
            "cuda_available": self.cuda_available,
            "mlx_available": self.mlx_available,
            "num_gpus": self.num_gpus,
            "max_threads": self.max_threads,
            "max_processes": self.max_processes
        }
        
        if self.cuda_available:
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
            info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)
        
        return info
    
    def benchmark(self, operation: str, size: int = 1000, iterations: int = 10) -> Dict[str, Any]:
        """
        Führt einen Benchmark für eine Operation durch
        
        Args:
            operation: Zu benchmarkende Operation ("matmul", "attention")
            size: Größe der Testdaten
            iterations: Anzahl der Iterationen
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        results = {}
        
        if operation == "matmul":
            # Benchmark für Matrix-Multiplikation
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            # Benchmark auf aktuellem Backend
            start_time = time.time()
            for _ in range(iterations):
                _ = self.matmul(a, b)
            end_time = time.time()
            results["current_backend"] = {
                "backend": self.preferred_backend,
                "time": (end_time - start_time) / iterations
            }
            
            # Benchmark auf CPU (NumPy)
            start_time = time.time()
            for _ in range(iterations):
                _ = np.matmul(a, b)
            end_time = time.time()
            results["numpy"] = {
                "backend": "numpy",
                "time": (end_time - start_time) / iterations
            }
            
            # Benchmark auf PyTorch CPU
            a_torch = torch.tensor(a)
            b_torch = torch.tensor(b)
            start_time = time.time()
            for _ in range(iterations):
                _ = torch.matmul(a_torch, b_torch)
            end_time = time.time()
            results["torch_cpu"] = {
                "backend": "torch_cpu",
                "time": (end_time - start_time) / iterations
            }
            
            # Benchmark auf CUDA (wenn verfügbar)
            if self.cuda_available:
                a_cuda = torch.tensor(a, device="cuda")
                b_cuda = torch.tensor(b, device="cuda")
                # Warm-up
                _ = torch.matmul(a_cuda, b_cuda)
                torch.cuda.synchronize()
                
                start_time = time.time()
                for _ in range(iterations):
                    _ = torch.matmul(a_cuda, b_cuda)
                torch.cuda.synchronize()
                end_time = time.time()
                results["torch_cuda"] = {
                    "backend": "torch_cuda",
                    "time": (end_time - start_time) / iterations
                }
            
            # Benchmark auf MLX (wenn verfügbar)
            if self.mlx_available:
                a_mlx = mx.array(a)
                b_mlx = mx.array(b)
                # Warm-up
                _ = mx.matmul(a_mlx, b_mlx)
                
                start_time = time.time()
                for _ in range(iterations):
                    _ = mx.matmul(a_mlx, b_mlx)
                end_time = time.time()
                results["mlx"] = {
                    "backend": "mlx",
                    "time": (end_time - start_time) / iterations
                }
        
        elif operation == "attention":
            # Benchmark für Attention-Mechanismus
            batch_size = 32
            seq_len = 128
            d_model = 64
            
            query = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            key = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            value = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            
            # Benchmark auf aktuellem Backend
            start_time = time.time()
            for _ in range(iterations):
                _ = self.attention(query, key, value)
            end_time = time.time()
            results["current_backend"] = {
                "backend": self.preferred_backend,
                "time": (end_time - start_time) / iterations
            }
            
            # Weitere Backend-spezifische Benchmarks können hier hinzugefügt werden
        
        return results


# Globale Instanz für einfachen Zugriff
gpu_accelerator = GPUAccelerator()

def to_tensor(data: Any, dtype: Optional[Any] = None) -> Any:
    """
    Konvertiert Daten in einen Tensor des aktiven Backends
    
    Args:
        data: Zu konvertierende Daten (numpy.ndarray, Liste, etc.)
        dtype: Datentyp für den Tensor (optional)
        
    Returns:
        Tensor im aktiven Backend
    """
    return gpu_accelerator.to_tensor(data, dtype)

def to_numpy(tensor: Any) -> np.ndarray:
    """
    Konvertiert einen Tensor zu numpy.ndarray
    
    Args:
        tensor: Zu konvertierender Tensor
        
    Returns:
        numpy.ndarray
    """
    return gpu_accelerator.to_numpy(tensor)

def matmul(a: Any, b: Any) -> Any:
    """
    Optimierte Matrix-Multiplikation
    
    Args:
        a: Erste Matrix
        b: Zweite Matrix
        
    Returns:
        Ergebnis der Matrix-Multiplikation
    """
    return gpu_accelerator.matmul(a, b)

def attention(query: Any, key: Any, value: Any, mask: Any = None, scale: float = 1.0) -> Any:
    """
    Optimierter Attention-Mechanismus mit verbesserter Performance
    
    Args:
        query: Query-Tensor
        key: Key-Tensor
        value: Value-Tensor
        mask: Optionale Maske
        scale: Skalierungsfaktor
        
    Returns:
        Ergebnis des Attention-Mechanismus
    """
    # Cache für schnelleren Zugriff
    cache_key = f"attention_{id(query)}_{id(key)}_{id(value)}_{id(mask)}"
    cached = get_from_cache(cache_key)
    if cached is not None:
        return cached
        
    # Optimierte Berechnung mit besserem Memory-Management
    result = gpu_accelerator.attention(query, key, value, mask, scale)
    
    # Cache-Ergebnis für zukünftige Aufrufe
    put_in_cache(cache_key, result)
    
    return result

def smart_parallel_map(func: Callable, items: List[Any], threshold: int = None) -> List[Any]:
    """
    Hochoptimierte, adaptive Parallelisierung, die automatisch die beste Methode basierend 
    auf Datengröße, Aufgabentyp und Hardwareumgebung wählt.
    
    Args:
        func: Auszuführende Funktion
        items: Liste von Elementen
        threshold: Optionaler Schwellenwert für Parallelisierungsentscheidung
                  (wenn None, wird automatisch basierend auf Hardware bestimmt)
        
    Returns:
        Liste mit Ergebnissen
    """
    if not items:
        return []
        
    n_items = len(items)
    backend_info = get_backend_info()
    
    # Dynamischen Schwellenwert basierend auf Hardware bestimmen,
    # wenn nicht explizit angegeben
    if threshold is None:
        # Basierend auf CPU-Anzahl und Cache-Eigenschaften abstimmen
        cpu_count = backend_info.get("logical_cpus", os.cpu_count() or 4)
        if backend_info.get("mlx_available", False):
            # Optimiere für Apple Silicon Neural Engine
            threshold = max(16, 32 // cpu_count)
        elif backend_info.get("cuda_available", False):
            # Optimiere für CUDA-fähige GPUs
            threshold = max(8, 16 // cpu_count)
        else:
            # Standard-Schwellenwert für allgemeine CPUs
            threshold = max(25, 100 // cpu_count)
    
    # Strategie 1: Für sehr kleine Datensätze - direkte serielle Verarbeitung
    if n_items < threshold:
        return [func(item) for item in items]
    
    # Analysiere Funktionseigenschaften (entweder durch Annotation oder Sampling)
    is_cpu_intensive = getattr(func, "_cpu_intensive", None)
    is_io_intensive = getattr(func, "_io_intensive", None)
    
    # Wenn keine Funktionsannotationen vorhanden sind, versuche Charakteristiken zu bestimmen
    if is_cpu_intensive is None or is_io_intensive is None:
        is_cpu_intensive, is_io_intensive = _sample_function_characteristics(func, items)
    
    # Optimierte Strategien basierend auf Aufgabentyp
    if is_io_intensive:
        # I/O-intensive Aufgaben: Thread-basierte Parallelisierung mit optimierter Concurrency
        thread_count = backend_info.get("logical_cpus", os.cpu_count() or 4) * 4
        thread_multiplier = min(8, max(2, thread_count // 2))
        logger.debug(f"I/O-intensive Aufgabe erkannt: Verwende {thread_multiplier}x Threads")
        return parallel_map(func, items, use_processes=False, chunk_size=1, 
                          thread_multiplier=thread_multiplier)
    
    # Für CPU-intensive Aufgaben: Adaptive Strategie basierend auf Last und Datengröße
    elif is_cpu_intensive:
        # Adaptive Entscheidung für Prozess- vs. Thread-basierte Ausführung
        cpu_load = _get_current_cpu_load()  # Ermittle aktuelle CPU-Auslastung
        
        # Bei hoher CPU-Last: Optimiere für geringere Thread-Konkurrenz
        if cpu_load > 0.7 and n_items <= threshold * 10:  # >70% CPU-Last
            logger.debug("Hohe CPU-Last erkannt: Verwende optimierte Thread-basierte Parallelisierung")
            return parallel_map(func, items, use_processes=False, chunk_size=None, optimize_for_load=True)
        
        # Bei großen Datensätzen auf Multi-Core-Systemen: Prozess-basierte Parallelisierung
        elif n_items > threshold * 5 and backend_info.get("logical_cpus", 1) >= 4:
            try:
                logger.debug("Große Datenmenge auf Multicore-System: Verwende Prozess-Pool")
                return parallel_map(func, items, use_processes=True)
            except (TypeError, AttributeError, pickle.PickleError) as e:
                logger.debug(f"Fallback auf Threads wegen Serialisierungsproblem: {str(e)}")
                return parallel_map(func, items, use_processes=False)
        
        # Für mittlere Datensätze: Thread-basierte Parallelisierung
        else:
            logger.debug("Mittlere Datenmenge: Verwende Thread-Pool")
            return parallel_map(func, items, use_processes=False)
    
    # Für unbekannte/gemischte Aufgabentypen: Standard Thread-basierte Parallelisierung
    else:
        return parallel_map(func, items, use_processes=False)

def _sample_function_characteristics(func: Callable, items: List[Any]) -> Tuple[bool, bool]:
    """
    Analysiert die Charakteristiken einer Funktion durch Sampling
    ihrer Ausführungszeit und ihres Verhaltens
    
    Args:
        func: Zu analysierende Funktion
        items: Beispielhafte Eingabedaten
        
    Returns:
        Tuple aus (is_cpu_intensive, is_io_intensive)
    """
    # Nehme eine kleine Stichprobe (max. 3 Elemente) zum Testen
    sample_size = min(3, len(items))
    sample_items = items[:sample_size]
    is_io_intensive = False
    is_cpu_intensive = False
    
    try:
        # Zeitmessung für Stichprobenausführung
        start_time = time.time()
        cpu_time_start = time.process_time()  # CPU-Zeit (ohne I/O-Wartezeit)
        
        for item in sample_items:
            func(item)
        
        wall_time = time.time() - start_time
        cpu_time = time.process_time() - cpu_time_start
        
        # I/O-intensiv, wenn deutliche Diskrepanz zwischen Wall-Zeit und CPU-Zeit
        if wall_time > 0 and cpu_time/wall_time < 0.3:  # <30% CPU-Nutzung deutet auf I/O hin
            is_io_intensive = True
        # CPU-intensiv, wenn hohe CPU-Auslastung oder lange Ausführungszeit
        elif cpu_time > 0.01 * sample_size:  # >10ms pro Element CPU-Zeit
            is_cpu_intensive = True
            
        logger.debug(f"Funktionsanalyse: CPU-Zeit={cpu_time:.4f}s, Wall-Zeit={wall_time:.4f}s, "
                    f"CPU-intensiv={is_cpu_intensive}, I/O-intensiv={is_io_intensive}")
    except Exception as e:
        # Bei Fehlern: Konservative Annahmen
        logger.debug(f"Fehler bei Funktionsanalyse: {e}, nutze konservative Annahmen")
        is_cpu_intensive = True
        is_io_intensive = False
    
    return is_cpu_intensive, is_io_intensive

def _get_current_cpu_load() -> float:
    """
    Ermittelt die aktuelle CPU-Auslastung des Systems
    
    Returns:
        CPU-Auslastung als Float zwischen 0 und 1
    """
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1) / 100.0
    except ImportError:
        # Fallback, wenn psutil nicht verfügbar ist
        try:
            # Alternative: Prozess-Auslastung via os.getloadavg() (nur Unix)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()[0] / os.cpu_count()
                return min(1.0, load_avg)
        except (AttributeError, OSError):
            pass
        # Default-Wert, wenn keine Messung möglich
        return 0.5

def parallel_map(func: Callable, items: List[Any], use_processes: bool = False, 
                chunk_size: int = None, thread_multiplier: int = 1) -> List[Any]:
    """
    Führt eine Funktion parallel auf einer Liste von Elementen aus mit verbessertem Workload-Balancing
    
    Args:
        func: Auszuführende Funktion
        items: Liste von Elementen
        use_processes: True für Prozess-basierte Parallelisierung, False für Thread-basierte
        chunk_size: Größe der Chunks für bessere Lastverteilung (auto = None)
        thread_multiplier: Faktor zur Erhöhung der Thread-Anzahl für IO-Aufgaben
        
    Returns:
        Liste mit Ergebnissen
    """
    if not items:
        return []
    
    # Automatische Chunk-Größen-Optimierung
    if chunk_size is None:
        # Komplexere Heuristik zur Berechnung der optimalen Chunk-Größe
        workers = gpu_accelerator.max_processes if use_processes else gpu_accelerator.max_threads
        
        # Passe Worker an Thread-Multiplier an (für IO-intensive Aufgaben)
        if not use_processes and thread_multiplier > 1:
            workers = min(workers * thread_multiplier, len(items))
            
        # Spezialisierte Chunk-Größen-Heuristik
        items_per_worker = max(1, len(items) // workers)
        
        if len(items) <= workers * 2:
            # Sehr wenige Elemente pro Worker: Feinere Granularität
            chunk_size = 1
        elif len(items) <= workers * 10:
            # Wenige Elemente pro Worker: Kleines Chunking
            chunk_size = max(1, items_per_worker // 2)
        elif len(items) <= workers * 50:
            # Mittlere Menge an Elementen: Ausgewogenes Chunking
            chunk_size = max(1, items_per_worker)
        else:
            # Viele Elemente: Größere Chunks für weniger Overhead
            chunk_size = max(1, items_per_worker * 2)
            
        # Minimiere Task-Queue-Überlastung
        chunk_size = min(chunk_size, 1000)  
    
    # Parallelisierungsstrategie basierend auf Datencharakteristiken
    if use_processes:
        try:
            with contextlib.redirect_stdout(None):  # Unterdrücke Ausgaben während der Pool-Erstellung
                return list(gpu_accelerator.process_pool.map(func, items, chunksize=chunk_size))
        except Exception as e:
            # Notfall-Fallback bei Pool-Fehlern
            logger.warning(f"Prozess-Pool-Fehler: {e}. Fallback auf serielle Verarbeitung.")
            return [func(item) for item in items]
    else:
        # Thread-basierte Parallelisierung mit optimierter Ressourcennutzung
        try:
            # Verwende den Thread-Pool mit optimierten Parametern
            return list(gpu_accelerator.thread_pool.map(func, items, chunksize=chunk_size))
        except Exception as e:
            # Notfall-Fallback bei Pool-Fehlern
            logger.warning(f"Thread-Pool-Fehler: {e}. Fallback auf serielle Verarbeitung.")
            return [func(item) for item in items]

def batch_process(func: Callable, items: List[Any], batch_size: int = 32, adaptive: bool = True) -> List[Any]:
    """
    Verarbeitet Elemente in Batches mit adaptiver Batchgröße für optimale Performance
    
    Args:
        func: Auszuführende Funktion (erwartet einen Batch)
        items: Liste von Elementen
        batch_size: Ausgangsgröße der Batches
        adaptive: Ob die Batchgröße dynamisch angepasst werden soll
        
    Returns:
        Liste mit Ergebnissen
    """
    if not items:
        return []
        
    # Adaptive Batchgröße basierend auf Hardwareverfügbarkeit
    if adaptive:
        backend_info = get_backend_info()
        if backend_info['cuda_available']:
            # GPU-optimierte Batchgröße basierend auf verfügbarem GPU-Speicher
            total_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            # Passe Batchgröße an verfügbaren Speicher an (vereinfachte Heuristik)
            memory_ratio = free_memory / total_memory
            batch_size = int(batch_size * max(1.0, memory_ratio * 2))
        elif backend_info.get('mlx_available', False):
            # Optimierte Batchgröße für Apple Neural Engine
            batch_size = max(16, min(batch_size * 2, 128))  # ANE-optimiert
            
    # Verarbeite Batches mit adaptiver Größe
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        results.extend(func(batch))
        
    return results

def get_backend_info() -> Dict[str, Any]:
    """
    Gibt Informationen über das aktive Backend zurück
    
    Returns:
        Dictionary mit Backend-Informationen
    """
    return gpu_accelerator.get_backend_info()

def benchmark(operation: str, size: int = 1000, iterations: int = 10) -> Dict[str, Any]:
    """
    Führt einen Benchmark für eine Operation durch
    
    Args:
        operation: Zu benchmarkende Operation ("matmul", "attention")
        size: Größe der Testdaten
        iterations: Anzahl der Iterationen
        
    Returns:
        Dictionary mit Benchmark-Ergebnissen
    """
    return gpu_accelerator.benchmark(operation, size, iterations)


if __name__ == "__main__":
    # Beispiel für die Verwendung des GPUAccelerators
    logging.basicConfig(level=logging.INFO)
    
    # Zeige Backend-Informationen
    info = get_backend_info()
    print(f"Aktives Backend: {info['backend']}")
    print(f"CUDA verfügbar: {info['cuda_available']}")
    print(f"MLX verfügbar: {info['mlx_available']}")
    
    # Führe einen einfachen Benchmark durch
    print("\nBenchmark für Matrix-Multiplikation (1000x1000):")
    results = benchmark("matmul", size=1000, iterations=5)
    
    for backend, result in results.items():
        print(f"  {backend}: {result['time']:.6f}s pro Iteration")
    
    # Demonstriere parallele Verarbeitung
    print("\nDemonstration der parallelen Verarbeitung:")
    
    def process_item(x):
        # Simuliere eine rechenintensive Operation
        time.sleep(0.1)
        return x * x
    
    items = list(range(100))
    
    start_time = time.time()
    sequential_results = [process_item(x) for x in items]
    sequential_time = time.time() - start_time
    print(f"Sequentielle Verarbeitung: {sequential_time:.2f}s")
    
    start_time = time.time()
    parallel_results = parallel_map(process_item, items)
    parallel_time = time.time() - start_time
    print(f"Parallele Verarbeitung: {parallel_time:.2f}s")
    print(f"Beschleunigung: {sequential_time / parallel_time:.2f}x")
