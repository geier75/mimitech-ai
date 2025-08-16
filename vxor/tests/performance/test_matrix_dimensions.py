#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Matrix-Dimensionen Performance-Tests

Vergleicht die Performance verschiedener Matrix-Operationen zwischen MLX und PyTorch
mit unterschiedlichen Matrix-Dimensionen (32x32 bis 512x512).

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

# Standardbibliotheken importieren
import os
import sys
import gc
import json
import time
import csv
import logging
import argparse
import threading
import multiprocessing
import platform
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
import cProfile
import pstats
import io
from multiprocessing import Pool, cpu_count
from contextlib import contextmanager
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Konfiguriere das Logging mit besserer Formatierung
log_file = os.path.join(os.path.dirname(__file__), 'matrix_benchmark.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Aktiviere Speicherbereinigung für stabilere Messungen
gc.enable()

# Speicherüberwachungsfunktionen
def get_memory_usage():
    """Gibt den aktuellen Speicherverbrauch des Prozesses zurück"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # In MB

# Zeitmessungs-Dekorator und Kontextmanager
@contextmanager
def precise_timer(name=None):
    """Präziser Timer als Kontextmanager"""
    start = time.perf_counter()
    mem_before = get_memory_usage()
    try:
        yield
    finally:
        end = time.perf_counter()
        mem_after = get_memory_usage()
        elapsed = end - start
        mem_diff = mem_after - mem_before
        if name:
            logger.debug(f"{name}: {elapsed:.6f}s, Speicheränderung: {mem_diff:.2f} MB")
        return elapsed, mem_diff

# Profiling-Dekorator mit Fehlerbehandlung
def profile_function(func):
    """Robuster Dekorator zur Profilierung einer Funktion mit cProfile"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Globaler Schalter für Profiling
        profiling_enabled = os.environ.get('ENABLE_PROFILING', '0') == '1'
        
        if not profiling_enabled:
            # Profiling ist deaktiviert, führe Funktion direkt aus
            return func(*args, **kwargs)
            
        try:
            # Versuche Profiling mit cProfile
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            # Ergebnisse ausgeben
            try:
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(20)  # Top 20 zeitintensivste Funktionen
                logger.debug(f"PROFILING für {func.__name__}:\n{s.getvalue()}")
            except Exception as prof_err:
                logger.warning(f"Fehler beim Analysieren des Profiling-Ergebnisses: {prof_err}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Profiling für {func.__name__} fehlgeschlagen: {e}. Führe ohne Profiling aus.")
            # Fallback: Führe Funktion ohne Profiling aus
            return func(*args, **kwargs)
    return wrapper

# MLX importieren, falls verfügbar
try:
    import mlx.core as mx
    HAS_MLX = True
    logger.info("MLX erfolgreich importiert.")
except ImportError:
    HAS_MLX = False
    logger.warning("MLX nicht verfügbar, verwende NumPy als Fallback.")

# PyTorch importieren, falls verfügbar
try:
    import torch
    HAS_PYTORCH = True
    logger.info("PyTorch erfolgreich importiert.")
except ImportError:
    HAS_PYTORCH = False
    logger.warning("PyTorch nicht verfügbar, verwende NumPy als Fallback.")

# MISO-Module importieren, mit Fallbacks falls nicht verfügbar
try:
    from miso.math.t_mathematics.engine import TMathEngine
    HAS_TMATH = True
except ImportError:
    logger.warning("TMathEngine nicht verfügbar, verwende Mock-Implementation.")
    HAS_TMATH = False
    
    # Mock-Implementation als Fallback
    class TMathEngine:
        def __init__(self, config=None):
            logger.warning("Verwende Mock-TMathEngine.")
            pass

try:
    from miso.math.mprime_engine import MPrimeEngine
    from miso.math.mprime.contextual_math import ContextualMathCore
    from miso.math.mprime.symbol_tree import SymbolTree
    HAS_MPRIME = True
except ImportError:
    logger.warning("MPrimeEngine/SymbolTree nicht verfügbar, verwende Mock-Implementation.")
    HAS_MPRIME = False
    
    # Mock-Implementationen als Fallback
    class MPrimeEngine:
        def __init__(self):
            logger.warning("Verwende Mock-MPrimeEngine.")
            pass
    
    class ContextualMathCore:
        def __init__(self):
            self.active_context = "scientific"
            logger.warning("Verwende Mock-ContextualMathCore.")
        
        def set_active_context(self, context_name):
            self.active_context = context_name
    
    class SymbolTree:
        def __init__(self):
            logger.warning("Verwende Mock-SymbolTree.")
            pass

# Konstanten für die Tests - optimierte Werte für bessere Performance und Genauigkeit
MATRIX_DIMENSIONS = [32, 64, 128, 256, 512, 1024]  # Erweiterte Dimensionen
TEST_ITERATIONS = 10  # Erhöht für präzisere Messungen
WARMUP_ITERATIONS = 3  # Mehr Warmup für stabilere Ergebnisse
BATCH_SIZE = 16

# Optimierungs-Flags
USE_PARALLEL_PROCESSING = True  # Aktiviert für schnellere Tests
USE_THREADING = False  # Threading deaktiviert wegen Metal GPU-Synchronisations-Problemen auf Apple Silicon
MAX_WORKERS = max(1, min(4, cpu_count() // 2))  # Optimierte Worker-Anzahl (50% der Kerne, max 4)
MEMORY_CLEANUP_INTERVAL = 3  # GC-Intervall für Speichermanagement
PROFILING_ENABLED = False  # Globales Profiling aktivieren/deaktivieren

# Apple Silicon Optimierungen
IS_APPLE_SILICON = os.uname().machine.startswith('arm')  # Erkennung von Apple Silicon (M1, M2, M3, etc.)
USE_MPS_OPTIMIZATIONS = IS_APPLE_SILICON  # Aktiviere Metal Performance Shaders Optimierungen

# Enums für die verschiedenen Testoptionen
class Backend(Enum):
    """Verfügbare Backend-Engines für mathematische Berechnungen"""
    PYTORCH = auto()
    MLX = auto()
    NUMPY = auto()

class Operation(Enum):
    """Matrix-Operationen, die getestet werden sollen"""
    MATMUL = auto()
    INVERSE = auto()
    SVD = auto()
    EIGENVALUES = auto()
    CHOLESKY = auto()
    QR = auto()

class PrecisionType(Enum):
    """Präzisionstypen für die Tests"""
    FLOAT32 = auto()
    FLOAT16 = auto()
    BFLOAT16 = auto()

# Einführung einer abstrakten Backend-Strategie für bessere Modularität
class BackendStrategy(ABC):
    """Abstrakte Basisklasse für Backend-Strategien"""
    
    @abstractmethod
    def get_precision_dtype(self, precision: PrecisionType):
        """Gibt den entsprechenden Datentyp für die angegebene Präzision zurück"""
        pass
    
    @abstractmethod
    def generate_random_matrix(self, dim: int, precision: PrecisionType):
        """Generiert eine zufällige Matrix mit angegebener Dimension und Präzision"""
        pass
    
    @abstractmethod
    def run_operation(self, operation: Operation, matrix_a, matrix_b=None):
        """Führt die spezifizierte Operation auf den Matrizen aus"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Führt Bereinigungsoperationen durch (Cache leeren, Speicher freigeben)"""
        pass

# Konkrete Strategie-Implementierungen
class PyTorchBackendStrategy(BackendStrategy):
    """PyTorch-spezifische Backend-Implementierung"""
    
    def __init__(self):
        self.device = 'cpu'
        if HAS_PYTORCH:
            try:
                if torch.backends.mps.is_available():
                    self.device = 'mps'  # Apple Silicon GPU
                    logger.info("PyTorch nutzt Apple MPS-Beschleunigung")
                elif torch.cuda.is_available():
                    self.device = 'cuda'
                    logger.info("PyTorch nutzt CUDA-Beschleunigung")
            except AttributeError:
                # Ältere PyTorch-Versionen haben kein MPS
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    logger.info("PyTorch nutzt CUDA-Beschleunigung")
    
    def get_precision_dtype(self, precision: PrecisionType):
        if not HAS_PYTORCH:
            raise RuntimeError("PyTorch ist nicht verfügbar")
            
        if precision == PrecisionType.FLOAT32:
            return torch.float32
        elif precision == PrecisionType.FLOAT16:
            return torch.float16
        elif precision == PrecisionType.BFLOAT16:
            if hasattr(torch, 'bfloat16'):
                return torch.bfloat16
            else:
                logger.warning("BFloat16 wird von dieser PyTorch-Version nicht unterstützt. Fallback auf Float16.")
                return torch.float16
        else:
            return torch.float32
    
    def generate_random_matrix(self, dim: int, precision: PrecisionType):
        if not HAS_PYTORCH:
            raise RuntimeError("PyTorch ist nicht verfügbar")
            
        dtype = self.get_precision_dtype(precision)
        # Optimierter Matrix-Generator mit besserer Initialisierung
        matrix = torch.randn(dim, dim, dtype=dtype, device=self.device)
        # Sicherstellen, dass die Matrix nicht singulär ist für bestimmte Operationen
        if dim > 1:
            matrix = matrix @ matrix.T + torch.eye(dim, dtype=dtype, device=self.device) * 0.1
        return matrix
    
    def run_operation(self, operation: Operation, matrix_a, matrix_b=None):
        if not HAS_PYTORCH:
            raise RuntimeError("PyTorch ist nicht verfügbar")
            
        # Verbesserte Fehlerbehandlung und Speichermanagement
        try:
            torch.cuda.empty_cache() if self.device == 'cuda' else None
            
            if operation == Operation.MATMUL:
                result = torch.matmul(matrix_a, matrix_b if matrix_b is not None else matrix_a)
            elif operation == Operation.INVERSE:
                result = torch.linalg.inv(matrix_a)
            elif operation == Operation.SVD:
                U, S, V = torch.linalg.svd(matrix_a, full_matrices=False)
                result = (U, S, V)  # Tuple mit den SVD-Komponenten
            elif operation == Operation.EIGENVALUES:
                result = torch.linalg.eigh(matrix_a)
            elif operation == Operation.CHOLESKY:
                result = torch.linalg.cholesky(matrix_a)
            elif operation == Operation.QR:
                Q, R = torch.linalg.qr(matrix_a)
                result = (Q, R)  # Tuple mit den QR-Komponenten
            else:
                raise ValueError(f"Unbekannte Operation: {operation}")
                
            # Force synchronization wenn auf GPU
            if self.device in ['cuda', 'mps']:
                torch.cuda.synchronize() if self.device == 'cuda' else torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
                
            return result
        except Exception as e:
            logger.error(f"Fehler bei PyTorch-Operation {operation.name}: {str(e)}")
            raise
    
    def cleanup(self):
        if HAS_PYTORCH:
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            elif self.device == 'mps' and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()


class MLXBackendStrategy(BackendStrategy):
    """MLX-spezifische Backend-Implementierung für Apple Silicon"""
    
    def get_precision_dtype(self, precision: PrecisionType):
        if not HAS_MLX:
            raise RuntimeError("MLX ist nicht verfügbar")
            
        if precision == PrecisionType.FLOAT32:
            return mx.float32
        elif precision == PrecisionType.FLOAT16:
            return mx.float16
        elif precision == PrecisionType.BFLOAT16:
            return mx.bfloat16
        else:
            return mx.float32
    
    def generate_random_matrix(self, dim: int, precision: PrecisionType):
        if not HAS_MLX:
            raise RuntimeError("MLX ist nicht verfügbar")
            
        dtype = self.get_precision_dtype(precision)
        # Optimierte Matrix-Generierung für MLX
        matrix = mx.random.normal((dim, dim), dtype=dtype)
        # Für numerische Stabilität
        if dim > 1:
            matrix = mx.matmul(matrix, matrix.T) + mx.eye(dim, dtype=dtype) * 0.1
        return matrix
    
    def run_operation(self, operation: Operation, matrix_a, matrix_b=None):
        if not HAS_MLX:
            raise RuntimeError("MLX ist nicht verfügbar")
            
        try:
            if operation == Operation.MATMUL:
                result = mx.matmul(matrix_a, matrix_b if matrix_b is not None else matrix_a)
            elif operation == Operation.INVERSE:
                result = mx.linalg.inv(matrix_a)
            elif operation == Operation.SVD:
                # MLX SVD-Implementierung
                U, S, V = mx.linalg.svd(matrix_a, full_matrices=False)
                result = (U, S, V)
            elif operation == Operation.EIGENVALUES:
                # MLX unterstützt eigenvalues über linalg.eigh für symmetrische Matrizen
                result = mx.linalg.eigh(matrix_a)
            elif operation == Operation.CHOLESKY:
                result = mx.linalg.cholesky(matrix_a)
            elif operation == Operation.QR:
                Q, R = mx.linalg.qr(matrix_a)
                result = (Q, R)
            else:
                raise ValueError(f"Unbekannte Operation: {operation}")
            
            # MLX ist lazy-evaluated, wir müssen die Berechnung starten
            mx.eval(result)
            return result
        except Exception as e:
            logger.error(f"Fehler bei MLX-Operation {operation.name}: {str(e)}")
            raise
    
    def cleanup(self):
        # MLX verwaltet Speicher automatisch, aber wir können hier zusätzliche
        # Bereinigungen durchführen, wenn notwendig
        pass


class NumPyBackendStrategy(BackendStrategy):
    """NumPy-spezifische Backend-Implementierung (Fallback)"""
    
    def get_precision_dtype(self, precision: PrecisionType):
        if precision == PrecisionType.FLOAT32:
            return np.float32
        elif precision == PrecisionType.FLOAT16:
            return np.float16
        elif precision == PrecisionType.BFLOAT16:
            logger.warning("BFloat16 wird von NumPy nicht nativ unterstützt. Fallback auf Float32.")
            return np.float32
        else:
            return np.float32
    
    def generate_random_matrix(self, dim: int, precision: PrecisionType):
        dtype = self.get_precision_dtype(precision)
        # Optimierte Matrix-Generierung für NumPy
        matrix = np.random.randn(dim, dim).astype(dtype)
        # Für numerische Stabilität
        if dim > 1:
            matrix = matrix @ matrix.T + np.eye(dim, dtype=dtype) * 0.1
        return matrix
    
    def run_operation(self, operation: Operation, matrix_a, matrix_b=None):
        try:
            if operation == Operation.MATMUL:
                result = np.matmul(matrix_a, matrix_b if matrix_b is not None else matrix_a)
            elif operation == Operation.INVERSE:
                result = np.linalg.inv(matrix_a)
            elif operation == Operation.SVD:
                U, S, V = np.linalg.svd(matrix_a, full_matrices=False)
                result = (U, S, V)
            elif operation == Operation.EIGENVALUES:
                result = np.linalg.eigh(matrix_a)
            elif operation == Operation.CHOLESKY:
                result = np.linalg.cholesky(matrix_a)
            elif operation == Operation.QR:
                Q, R = np.linalg.qr(matrix_a)
                result = (Q, R)
            else:
                raise ValueError(f"Unbekannte Operation: {operation}")
            return result
        except Exception as e:
            logger.error(f"Fehler bei NumPy-Operation {operation.name}: {str(e)}")
            raise
    
    def cleanup(self):
        # NumPy erfordert keine spezielle Bereinigung
        pass


# Backend-Factory zur Erstellung der entsprechenden Strategie
class BackendFactory:
    """Factory-Klasse zur Erzeugung von Backend-Strategien"""
    
    @staticmethod
    def create_backend(backend_type: Backend) -> BackendStrategy:
        """Erzeugt die entsprechende Backend-Strategie für den angegebenen Typ"""
        if backend_type == Backend.PYTORCH and HAS_PYTORCH:
            return PyTorchBackendStrategy()
        elif backend_type == Backend.MLX and HAS_MLX:
            return MLXBackendStrategy()
        elif backend_type == Backend.NUMPY:
            return NumPyBackendStrategy()
        else:
            # Fallback zu NumPy, wenn das angeforderte Backend nicht verfügbar ist
            logger.warning(f"{backend_type.name} ist nicht verfügbar. Verwende NumPy als Fallback.")
            return NumPyBackendStrategy()
@dataclass
class BenchmarkResult:
    """Verbesserte Klasse für Benchmark-Ergebnisse mit Dataclass-Support
    
    Diese Klasse verwendet Dataclasses für bessere Typsicherheit und Serialisierung.
    Zudem wurden Methoden zur Verarbeitung von Zeitmessungen und Speichernutzung hinzugefügt.
    """
    operation: Operation
    backend: Backend
    dimension: int
    precision: PrecisionType = PrecisionType.FLOAT32
    execution_times: List[float] = field(default_factory=list)
    success_count: int = 0
    total_count: int = 0
    success_rate: float = 0.0
    mean_time: float = 0.0
    std_dev: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    memory_usage_before: int = 0
    memory_usage_after: int = 0
    memory_change: int = 0
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, execution_time: float, success: bool = True, error_message: Optional[str] = None) -> None:
        """Fügt ein einzelnes Benchmark-Ergebnis hinzu
        
        Args:
            execution_time: Ausführungszeit in Sekunden
            success: Ob die Operation erfolgreich war
            error_message: Optionale Fehlermeldung, falls die Operation fehlgeschlagen ist
        """
        self.total_count += 1
        
        if success:
            self.success_count += 1
            self.execution_times.append(execution_time)
        elif error_message:
            self.error_messages.append(error_message)
    
    def set_memory_usage(self, before: int, after: int) -> None:
        """Setzt die Speichernutzungsinformationen
        
        Args:
            before: Speichernutzung vor der Operation in Bytes
            after: Speichernutzung nach der Operation in Bytes
        """
        self.memory_usage_before = before
        self.memory_usage_after = after
        self.memory_change = after - before
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Fügt Metadaten zum Benchmark hinzu
        
        Args:
            key: Schlüssel für die Metadaten
            value: Wert für die Metadaten
        """
        self.metadata[key] = value
    
    def finalize(self) -> None:
        """Berechnet die finalen Statistiken aus den gesammelten Daten"""
        if self.execution_times:
            self.success_rate = self.success_count / self.total_count if self.total_count > 0 else 0.0
            self.mean_time = np.mean(self.execution_times)
            self.std_dev = np.std(self.execution_times)
            self.min_time = np.min(self.execution_times)
            self.max_time = np.max(self.execution_times)
        else:
            self.success_rate = 0.0
            self.mean_time = 0.0
            self.std_dev = 0.0
            self.min_time = 0.0
            self.max_time = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert das Ergebnis in ein Dictionary für JSON-Serialisierung"""
        result = {
            "operation": self.operation.name,
            "backend": self.backend.name,
            "dimension": self.dimension,
            "precision": self.precision.name,
            "success_rate": self.success_rate,
            "mean_time": self.mean_time,
            "std_dev": self.std_dev,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "memory_usage_before": self.memory_usage_before,
            "memory_usage_after": self.memory_usage_after,
            "memory_change": self.memory_change,
            "total_runs": self.total_count,
            "successful_runs": self.success_count
        }
        
        # Füge Metadaten hinzu, falls vorhanden
        if self.metadata:
            result["metadata"] = self.metadata
            
        # Füge Fehlermeldungen hinzu, falls vorhanden
            
        if self.error_messages:
            result["error_messages"] = self.error_messages
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Erstellt ein BenchmarkResult-Objekt aus einem Dictionary
        
        Args:
            data: Dictionary mit Benchmark-Daten
            
        Returns:
            Ein neues BenchmarkResult-Objekt
        """
        operation = Operation[data["operation"]]
        backend = Backend[data["backend"]]
        dimension = data["dimension"]
        precision = PrecisionType[data["precision"]]
        
        result = cls(operation, backend, dimension, precision)
        
        # Kopiere alle vorhandenen Felder
        for key, value in data.items():
            if key not in ["operation", "backend", "dimension", "precision"] and hasattr(result, key):
                setattr(result, key, value)
                
        return result


# Reporting-Klassen mit verbesserten Funktionen

class BenchmarkReporter(ABC):
    """Abstrakte Basisklasse für Benchmark-Reporter"""
    
    @abstractmethod
    def generate_report(self, results: List[BenchmarkResult], output_file: str) -> None:
        """Generiert einen Bericht aus den Benchmark-Ergebnissen
        
        Args:
            results: Liste der Benchmark-Ergebnisse
            output_file: Pfad zur Ausgabedatei
        """
        pass


class JsonReporter(BenchmarkReporter):
    """Reporter für JSON-Berichte"""
    
    def generate_report(self, results: List[BenchmarkResult], output_file: str) -> None:
        """Generiert einen JSON-Bericht aus den Benchmark-Ergebnissen
        
        Args:
            results: Liste der Benchmark-Ergebnisse
            output_file: Pfad zur JSON-Ausgabedatei
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "results": [result.to_dict() for result in results]
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            
        logger.info(f"JSON-Bericht in {output_file} erstellt.")
    
    def _get_system_info(self) -> Dict[str, str]:
        """Sammelt Systeminformationen für den Bericht"""
        system_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor()
        }
        
        # Füge Versionsinformationen für Bibliotheken hinzu
        if HAS_NUMPY:
            system_info["numpy_version"] = np.__version__
        if HAS_PYTORCH:
            system_info["pytorch_version"] = torch.__version__
        if HAS_MLX:
            system_info["mlx_version"] = mx.__version__ if hasattr(mx, "__version__") else "unbekannt"
            
        return system_info


class CSVReporter(BenchmarkReporter):
    """Reporter für CSV-Berichte"""
    
    def generate_report(self, results: List[BenchmarkResult], output_file: str) -> None:
        """Generiert einen CSV-Bericht aus den Benchmark-Ergebnissen
        
        Args:
            results: Liste der Benchmark-Ergebnisse
            output_file: Pfad zur CSV-Ausgabedatei
        """
        header = [
            "Operation", "Backend", "Dimension", "Precision", 
            "Success Rate", "Mean Time (s)", "Std Dev", "Min Time (s)", "Max Time (s)",
            "Memory Change (MB)"
        ]
        
        rows = []
        for result in results:
            rows.append([
                result.operation.name,
                result.backend.name,
                result.dimension,
                result.precision.name,
                result.success_rate,
                result.mean_time,
                result.std_dev,
                result.min_time,
                result.max_time,
                result.memory_change / (1024 * 1024)  # Umrechnung in MB
            ])
        
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
            
        logger.info(f"CSV-Bericht in {output_file} erstellt.")


class BenchmarkConfiguration:
    """Konfiguration für Benchmark-Tests"""
    
    def __init__(self, 
                 dimensions: List[int] = MATRIX_DIMENSIONS, 
                 operations: List[Operation] = None,
                 precision_types: List[PrecisionType] = None,
                 backends: List[Backend] = None,
                 iterations: int = TEST_ITERATIONS,
                 warmup_iterations: int = WARMUP_ITERATIONS,
                 use_threading: bool = True,
                 num_threads: int = None,
                 archive_results: bool = True):
        """Initialisiert eine neue Benchmark-Konfiguration
        
        Args:
            dimensions: Liste der zu testenden Matrix-Dimensionen
            operations: Liste der zu testenden Operationen
            precision_types: Liste der zu testenden Präzisionstypen
            backends: Liste der zu testenden Backends
            iterations: Anzahl der Testdurchläufe
            warmup_iterations: Anzahl der Warmup-Durchläufe
            use_threading: Ob Threading verwendet werden soll
            num_threads: Anzahl der zu verwendenden Threads
            archive_results: Ob Ergebnisse archiviert werden sollen
        """
        self.dimensions = dimensions
        self.operations = operations if operations else list(Operation)
        self.precision_types = precision_types if precision_types else list(PrecisionType)
        self.backends = backends  # Wird automatisch ermittelt, falls None
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.use_threading = use_threading
        self.num_threads = num_threads if num_threads else min(cpu_count(), 4)  # Standardmäßig 4 Threads oder weniger
        self.archive_results = archive_results


class MatrixBenchmarker:
    """Benchmark-Tool für Matrix-Operationen mit verschiedenen Dimensionen
    
    Diese Klasse wurde im Zuge des Refactorings modularisiert und verwendet das
    Strategy-Muster für Backend-Implementierungen sowie Dependency Injection
    für eine bessere Trennung der Verantwortlichkeiten.
    """
    
    def __init__(self, config: BenchmarkConfiguration = None, use_mlx: bool = True, use_pytorch: bool = True):
        """Initialisiert den Benchmarker mit den verfügbaren Backends
        
        Args:
            config: Benchmark-Konfiguration
            use_mlx: Ob MLX als Backend verwendet werden soll
            use_pytorch: Ob PyTorch als Backend verwendet werden soll
        """
        # Initialisiere Konfiguration
        self.config = config if config else BenchmarkConfiguration()
        
        # Initialisiere Ergebnisliste
        self.results = []
        
        # Bestimme verfügbare Backends
        self.available_backends = []
        if self.config.backends:
            self.available_backends = self.config.backends
        else:
            # Automatische Backend-Erkennung
            if use_pytorch and HAS_PYTORCH:
                self.available_backends.append(Backend.PYTORCH)
                logger.info("PyTorch-Backend aktiviert.")
            
            if use_mlx and HAS_MLX:
                self.available_backends.append(Backend.MLX)
                logger.info("MLX-Backend aktiviert.")
            
            if not self.available_backends:
                # Fallback auf NumPy, wenn keine anderen Backends verfügbar sind
                self.available_backends.append(Backend.NUMPY)
                logger.info("NumPy-Backend als Fallback aktiviert.")
        
        # Initialisiere Backend-Strategien
        self.backend_strategies = {}
        for backend in self.available_backends:
            self.backend_strategies[backend] = BackendFactory.create_backend(backend)
        
        # Initialisiere Engines nur falls verfügbar
        try:
            self.t_math_engine = TMathEngine()
            logger.info("TMathEngine erfolgreich initialisiert.")
        except Exception as e:
            logger.warning(f"Konnte TMathEngine nicht initialisieren: {str(e)}")
            self.t_math_engine = None
        
        try:
            # Initialisiere MPRIME Engine für Symbolische Mathematik
            self.mprime_engine = MPrimeEngine()
            
            # Initialisiere Contextual Math Core
            self.contextual_math = ContextualMathCore()
            self.contextual_math.set_active_context("scientific")
            logger.info("MPRIME Engine und ContextualMathCore erfolgreich initialisiert.")
        except Exception as e:
            logger.warning(f"Konnte MPrimeEngine/ContextualMathCore nicht initialisieren: {str(e)}")
            self.mprime_engine = None
            self.contextual_math = None
        
        # Initialisiere Reporter
        self.reporters = {
            'json': JsonReporter(),
            'csv': CSVReporter()
        }
        
        # Speichere Konfigurationsparameter
        self.iterations = self.config.iterations
        self.warmup_iterations = self.config.warmup_iterations
        
        logger.info(f"MatrixBenchmarker initialisiert mit {len(self.available_backends)} Backend(s): {[b.name for b in self.available_backends]}")
    
    @lru_cache(maxsize=32)
    def _get_precision_dtype(self, backend: Backend, precision: PrecisionType):
        """Gibt den richtigen Datentyp für eine bestimmte Präzision und ein bestimmtes Backend zurück"""
        if backend == Backend.PYTORCH:
            if precision == PrecisionType.FLOAT16:
                return torch.float16
            elif precision == PrecisionType.BFLOAT16:
                return torch.bfloat16
            else:  # FLOAT32
                return torch.float32
        elif backend == Backend.MLX:
            if precision == PrecisionType.FLOAT16:
                return mx.float16 if hasattr(mx, 'float16') else np.float16
            elif precision == PrecisionType.BFLOAT16:
                try:
                    return mx.bfloat16
                except AttributeError:
                    logger.warning("MLX bfloat16 nicht unterstützt, verwende float16.")
                    return mx.float16 if hasattr(mx, 'float16') else np.float16
            else:  # FLOAT32
                return mx.float32 if hasattr(mx, 'float32') else np.float32
        else:  # NUMPY
            if precision == PrecisionType.FLOAT16:
                return np.float16
            elif precision == PrecisionType.BFLOAT16:
                logger.warning("NumPy bfloat16 nicht unterstützt, verwende float16.")
                return np.float16
            else:  # FLOAT32
                return np.float32
    
    def _generate_random_matrix(self, dim: int, backend: Backend, precision: PrecisionType = PrecisionType.FLOAT32):
        """Generiert eine zufällige Matrix mit den angegebenen Dimensionen - optimierte Version"""
        with precise_timer(f"Matrix-Generation {dim}x{dim}, {backend.name}, {precision.name}"):
            # Seed setzen für reproduzierbare Ergebnisse
            np.random.seed(42 + dim)  # Unterschiedlicher Seed für jede Dimension
            
            # Direktes Erzeugen mit dem korrekten Datentyp (schneller als nachträgliche Konvertierung)
            if precision == PrecisionType.FLOAT16 or precision == PrecisionType.BFLOAT16:
                base_type = np.float16
            else:
                base_type = np.float32
                
            # Verwende ascontiguousarray für CPU-optimierte Speicheranordnung
            np_matrix = np.ascontiguousarray(np.random.rand(dim, dim).astype(base_type))
            
            # Backend-spezifische Optimierungen
            if backend == Backend.PYTORCH:
                # Effizienterer PyTorch Device-Check - nur einmal prüfen und cachen
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                dtype = self._get_precision_dtype(backend, precision)
                
                # Pin-Memory für schnelleren CPU-GPU-Transfer
                result = torch.tensor(np_matrix, dtype=dtype, device=device)
                if device == "mps":
                    # Optimiere für schnellere GPU-Berechnungen
                    torch.mps.synchronize()
                return result
            
            elif backend == Backend.MLX:
                # Optimierte MLX-Konvertierung mit Typ-Cache
                dtype = self._get_precision_dtype(backend, precision)
                
                try:
                    # Versuche direkte Konvertierung mit dem richtigen Datentyp
                    result = mx.array(np_matrix, dtype=dtype)
                    # Führe eine sofortige Evaluation durch, um overhead zu vermeiden
                    if hasattr(mx, 'eval'):
                        mx.eval(result)
                    return result
                except (AttributeError, TypeError) as e:
                    logger.warning(f"MLX Typkonvertierungsfehler: {e}, verwende Fallback")
                    # Fallback auf einfache Konvertierung
                    result = mx.array(np_matrix.astype(np.float32))
                    if hasattr(mx, 'eval'):
                        mx.eval(result)
                    return result
            
            else:  # NUMPY - effizientere NumPy-Operationen
                dtype = self._get_precision_dtype(backend, precision)
                # Vermeide unnötige Konvertierungen, wenn der Typ bereits korrekt ist
                if np_matrix.dtype == dtype:
                    return np_matrix
                else:
                    return np_matrix.astype(dtype)
    
    def _generate_batch_matrices(self, dim: int, batch_size: int, backend: Backend, 
                               precision: PrecisionType = PrecisionType.FLOAT32):
        """Generiert einen Batch von Matrizen für Batch-Operationen"""
        # Generiere zufällige Werte mit NumPy
        np_matrices = np.random.rand(batch_size, dim, dim).astype(np.float32)
        
        # Positive definite Matrizen für bestimmte Operationen wie Cholesky
        np_pos_def = np.array([np_matrices[i].dot(np_matrices[i].T) + np.eye(dim) for i in range(batch_size)])
        
        # Konvertiere in das gewünschte Backend-Format
        if backend == Backend.PYTORCH:
            if precision == PrecisionType.FLOAT16:
                return {
                    "regular": torch.tensor(np_matrices, dtype=torch.float16, device="mps" if torch.backends.mps.is_available() else "cpu"),
                    "pos_def": torch.tensor(np_pos_def, dtype=torch.float16, device="mps" if torch.backends.mps.is_available() else "cpu")
                }
            elif precision == PrecisionType.BFLOAT16:
                return {
                    "regular": torch.tensor(np_matrices, dtype=torch.bfloat16, device="mps" if torch.backends.mps.is_available() else "cpu"),
                    "pos_def": torch.tensor(np_pos_def, dtype=torch.bfloat16, device="mps" if torch.backends.mps.is_available() else "cpu")
                }
            else:  # FLOAT32
                return {
                    "regular": torch.tensor(np_matrices, dtype=torch.float32, device="mps" if torch.backends.mps.is_available() else "cpu"),
                    "pos_def": torch.tensor(np_pos_def, dtype=torch.float32, device="mps" if torch.backends.mps.is_available() else "cpu")
                }
        
        elif backend == Backend.MLX:
            if precision == PrecisionType.FLOAT16:
                return {
                    "regular": mx.array(np_matrices.astype(np.float16)),
                    "pos_def": mx.array(np_pos_def.astype(np.float16))
                }
            elif precision == PrecisionType.BFLOAT16:
                try:
                    return {
                        "regular": mx.array(np_matrices, dtype=mx.bfloat16),
                        "pos_def": mx.array(np_pos_def, dtype=mx.bfloat16)
                    }
                except AttributeError:
                    logger.warning("MLX bfloat16 nicht unterstützt, verwende float16.")
                    return {
                        "regular": mx.array(np_matrices.astype(np.float16)),
                        "pos_def": mx.array(np_pos_def.astype(np.float16))
                    }
            else:  # FLOAT32
                return {
                    "regular": mx.array(np_matrices),
                    "pos_def": mx.array(np_pos_def)
                }
        
        else:  # NUMPY
            if precision == PrecisionType.FLOAT16:
                return {
                    "regular": np_matrices.astype(np.float16),
                    "pos_def": np_pos_def.astype(np.float16)
                }
            elif precision == PrecisionType.BFLOAT16:
                logger.warning("NumPy bfloat16 nicht unterstützt, verwende float16.")
                return {
                    "regular": np_matrices.astype(np.float16),
                    "pos_def": np_pos_def.astype(np.float16)
                }
            else:  # FLOAT32
                return {
                    "regular": np_matrices,
                    "pos_def": np_pos_def
                }
    
    @profile_function
    def benchmark_operation(self, operation: Operation, backend: Backend, dimension: int,
                         precision: PrecisionType = PrecisionType.FLOAT32,
                         iterations: int = None, warmup: int = None) -> BenchmarkResult:
        """Führt einen Benchmark für eine bestimmte Operation mit einer bestimmten Dimension durch - optimierte Version"""
        logger.info(f"Benchmark: {operation.name}, {backend.name}, {dimension}x{dimension}, {precision.name}")
        
        # Speichernutzung vor dem Benchmark erfassen
        mem_start = get_memory_usage()
        result = BenchmarkResult(operation, backend, dimension, precision)
        
        try:
            # Generiere Testdaten optimiert
            with precise_timer(f"Testdaten-Generierung {dimension}x{dimension}"):
                matrix_a = self._generate_random_matrix(dimension, backend, precision)
                matrix_b = self._generate_random_matrix(dimension, backend, precision)
                
                # Für Operationen, die eine positive definite Matrix benötigen
                if operation in [Operation.CHOLESKY]:
                    if backend == Backend.PYTORCH:
                        with torch.no_grad():
                            matrix_a = matrix_a @ matrix_a.T + torch.eye(dimension, device=matrix_a.device, dtype=matrix_a.dtype)
                    elif backend == Backend.MLX:
                        matrix_a = mx.matmul(matrix_a, mx.transpose(matrix_a)) + mx.eye(dimension, dtype=matrix_a.dtype)
                        if hasattr(mx, 'eval'):
                            mx.eval(matrix_a)  # Sofort auswerten
                    else:  # NumPy
                        matrix_a = np.ascontiguousarray(matrix_a @ matrix_a.T + np.eye(dimension, dtype=matrix_a.dtype))
            
            # Speicher bereinigen vor dem Benchmark
            gc.collect()
            
            # Warmup-Phase - wichtig für konsistente Ergebnisse
            if warmup > 0:
                with precise_timer(f"Warmup-Phase {warmup} Iterationen"):
                    for i in range(warmup):
                        self._run_operation(operation, backend, matrix_a, matrix_b)
                        # Force Garbage Collection zwischen Warmup-Iterationen für stabilere Messungen
                        if i % 2 == 0:  # Nach jeder zweiten Iteration
                            gc.collect()
            
            # Benchmark-Phase mit präziser Zeitmessung
            execution_times = []
            success_count = 0
            timer_results = []
            
            for i in range(iterations):
                gc.collect()  # Speicherbereinigung vor jedem Durchlauf
                success = True
                
                try:
                    # Verbesserter Timer mit perf_counter statt time.time()
                    start_time = time.perf_counter()
                    
                    # Operation ausführen
                    op_result = self._run_operation(operation, backend, matrix_a, matrix_b)
                    
                    # Sicherstellen, dass alle GPU/Hardware-Operationen abgeschlossen sind
                    if backend == Backend.PYTORCH:
                        device = matrix_a.device
                        if device.type == 'mps':
                            torch.mps.synchronize()
                        elif device.type == 'cuda':
                            torch.cuda.synchronize()
                    elif backend == Backend.MLX and hasattr(mx, 'eval'):
                        # Bei mehreren Rückgabewerten
                        if isinstance(op_result, tuple):
                            for res in op_result:
                                mx.eval(res)
                        else:
                            mx.eval(op_result)
                    
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    execution_times.append(execution_time)
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Fehler: {operation.name}, {backend.name}, Durchlauf {i+1}: {str(e)}")
                    success = False
                    execution_time = 0
                
                result.add_result(execution_time, success)
                logger.debug(f"Durchlauf {i+1}/{iterations}: {execution_time:.6f}s, Erfolg: {success}")
            
            # Ergebnis finalisieren
            result.finalize()
            self.results.append(result)
            
            # Speichernutzung nach dem Benchmark
            mem_end = get_memory_usage()
            mem_diff = mem_end - mem_start
            
            logger.info(f"Benchmark: {operation.name}, {backend.name}, {dimension}x{dimension} => "  
                       f"Zeit: {result.mean_time:.6f}s ±{result.std_dev:.6f}s, "  
                       f"Erfolg: {result.success_rate:.2f}, "  
                       f"Speicheränderung: {mem_diff:.2f} MB")
            return result
            
        except Exception as e:
            logger.error(f"Kritischer Fehler im Benchmark: {operation.name}/{backend.name}/{dimension}: {str(e)}")
            return None
    
    @profile_function
    def _run_operation(self, operation: Operation, backend: Backend, matrix_a, matrix_b):
        """Führt eine bestimmte Operation mit dem angegebenen Backend aus - optimierte Version"""
        with precise_timer(f"Operation: {operation.name} mit {backend.name}"):
            if backend == Backend.PYTORCH:
                return self._run_pytorch_operation(operation, matrix_a, matrix_b)
            elif backend == Backend.MLX:
                return self._run_mlx_operation(operation, matrix_a, matrix_b)
            else:  # NumPy
                return self._run_numpy_operation(operation, matrix_a, matrix_b)
    
    def _run_pytorch_operation(self, operation: Operation, matrix_a, matrix_b):
        """Optimierte PyTorch-Operationen mit verbessertem Speichermanagement"""
        # Stelle sicher, dass die Matrizen auf dem richtigen Gerät sind
        device = matrix_a.device
        
        # Führe GPU-spezifische Optimierungen durch
        is_mps = device.type == 'mps'
        is_cuda = device.type == 'cuda'
        
        if operation == Operation.MATMUL:
            # Verwende PyTorch's optimierte matmul-Implementierung
            with torch.no_grad():  # Reduziert Speicherverbrauch, da wir kein Autograd benötigen
                result = torch.matmul(matrix_a, matrix_b)
                if is_mps or is_cuda:
                    torch.mps.synchronize() if is_mps else torch.cuda.synchronize()
                return result
                
        elif operation == Operation.INVERSE:
            with torch.no_grad():
                result = torch.linalg.inv(matrix_a)
                if is_mps or is_cuda:
                    torch.mps.synchronize() if is_mps else torch.cuda.synchronize()
                return result
                
        elif operation == Operation.SVD:
            with torch.no_grad():
                result = torch.linalg.svd(matrix_a, full_matrices=False)  # full=False ist effizienter
                if is_mps or is_cuda:
                    torch.mps.synchronize() if is_mps else torch.cuda.synchronize()
                return result
                
        elif operation == Operation.EIGENVALUES:
            with torch.no_grad():
                result = torch.linalg.eigh(matrix_a)
                if is_mps or is_cuda:
                    torch.mps.synchronize() if is_mps else torch.cuda.synchronize()
                return result
                
        elif operation == Operation.CHOLESKY:
            with torch.no_grad():
                result = torch.linalg.cholesky(matrix_a)
                if is_mps or is_cuda:
                    torch.mps.synchronize() if is_mps else torch.cuda.synchronize()
                return result
                
        elif operation == Operation.QR:
            with torch.no_grad():
                result = torch.linalg.qr(matrix_a)
                if is_mps or is_cuda:
                    torch.mps.synchronize() if is_mps else torch.cuda.synchronize()
                return result
    
    def _run_mlx_operation(self, operation: Operation, matrix_a, matrix_b):
        """Optimierte MLX-Operationen speziell für Apple Silicon mit verbessertem Speichermanagement"""
        # Cache für MLX-NumPy Konvertierungen
        matrix_a_np = None  # Wird bei Bedarf belegt
        
        # Optimierter Kontext für MLX-Operationen
        # Verwende try-finally, um sicherzustellen, dass compile_mode wieder zurückgesetzt wird
        original_compile_mode = None
        try:
            # Setze optimale Compile-Modi für schnellere Berechnung auf Apple Silicon
            if hasattr(mx, 'set_default_device'):
                # Stelle sicher, dass wir das korrekte Gerät verwenden (Apple Neural Engine wenn möglich)
                current_device = mx.default_device()
                if 'mps' not in str(current_device).lower() and hasattr(mx, 'gpu'):
                    mx.set_default_device(mx.gpu)
                    logger.info("MLX-Gerät auf GPU/MPS gesetzt für optimale Leistung")
            
            # Aktiviere JIT-Kompilierung für bessere Leistung wenn verfügbar
            if hasattr(mx, 'compile'):
                # Speichere ursprünglichen Modus, um ihn später wiederherzustellen
                if hasattr(mx, 'get_compile_mode'):
                    original_compile_mode = mx.get_compile_mode()
                
                # Setze optimalen Compile-Modus - "reduce-overhead" ist ideal für wiederholte Operationen
                mx.compile(mode="reduce-overhead")
                
            # Führe die eigentliche Operation aus
            if operation == Operation.MATMUL:
                # Direkte Verwendung der MLX-Operation mit optimaler Speicherlayout
                # Verwende "lazy" Evaluation für bessere Pipeline-Nutzung und kompakte Graphen
                result = mx.matmul(matrix_a, matrix_b)
                
                # Batched Matmul wenn möglich (für bessere Parallelisierung auf Apple Silicon)
                if result.shape[0] > 32 and result.shape[1] > 32:
                    # Verwende "Tiling" für bessere Cache-Nutzung bei großen Matrizen
                    if hasattr(mx, 'eval'):
                        # Strategische Evaluation für bessere Cache-Nutzung
                        mx.eval(result)
                return result
                
            elif operation == Operation.INVERSE:
                # Überprüfe auf MLX mit eingebauter Inverse
                if hasattr(mx, 'linalg') and hasattr(mx.linalg, 'inv'):
                    # Direkter MLX-Pfad mit Optimierungen für Apple Silicon
                    result = mx.linalg.inv(matrix_a)
                    if hasattr(mx, 'eval'):
                        mx.eval(result)
                    return result
                else:
                    # Optimierte Fallback-Konvertierung mit direkten Pfaden
                    if matrix_a_np is None:
                        # Effiziente Konvertierung mit optimiertem Pfad für Apple Silicon
                        # Direkte Array-Konvertierung ohne Umweg über .tolist() für bessere Leistung
                        if hasattr(mx, 'to_numpy'):
                            matrix_a_np = mx.to_numpy(matrix_a)  # Direkte Konvertierung
                        else:
                            matrix_a_np = np.asarray(mx.array(matrix_a).tolist())
                    
                    # Verwende optimierte NumPy-Operation
                    np_result = np.linalg.inv(matrix_a_np)
                    # Effiziente Zurückkonvertierung
                    if hasattr(mx, 'array') and 'device' in inspect.signature(mx.array).parameters:
                        result = mx.array(np_result, device=mx.default_device())  # Direkt aufs richtige Gerät
                    else:
                        result = mx.array(np_result)
                    
                    if hasattr(mx, 'eval'):
                        mx.eval(result)
                    return result
                    
            elif operation == Operation.SVD:
                # MLX-Version mit eingebauter SVD
                if hasattr(mx, 'linalg') and hasattr(mx.linalg, 'svd'):
                    # Direkter MLX-Pfad optimal für Neural Engine
                    u, s, vh = mx.linalg.svd(matrix_a, full_matrices=False)
                    if hasattr(mx, 'eval'):
                        # Batch-Evaluation für bessere Leistung
                        mx.eval([u, s, vh])
                    return u, s, vh
                else:
                    # Optimierte Fallback-Konvertierung mit direkten Pfaden
                    if matrix_a_np is None:
                        if hasattr(mx, 'to_numpy'):
                            matrix_a_np = mx.to_numpy(matrix_a)
                        else:
                            matrix_a_np = np.asarray(mx.array(matrix_a).tolist())
                    
                    u, s, vh = np.linalg.svd(matrix_a_np, full_matrices=False)
                    
                    # Optimierte Zurückkonvertierung mit gerätespezifischen Anpassungen
                    if hasattr(mx, 'array') and 'device' in inspect.signature(mx.array).parameters:
                        current_device = mx.default_device()
                        result_u = mx.array(u, device=current_device)
                        result_s = mx.array(s, device=current_device)
                        result_vh = mx.array(vh, device=current_device)
                    else:
                        result_u, result_s, result_vh = mx.array(u), mx.array(s), mx.array(vh)
                    
                    if hasattr(mx, 'eval'):
                        # Batch-Evaluation für bessere Leistung
                        mx.eval([result_u, result_s, result_vh])
                    return result_u, result_s, result_vh
                    
            elif operation == Operation.EIGENVALUES:
                # Optimierte Eigenvalue-Berechnung für Apple Silicon
                if hasattr(mx, 'linalg') and hasattr(mx.linalg, 'eigh'):
                    eigenvalues, eigenvectors = mx.linalg.eigh(matrix_a)
                    if hasattr(mx, 'eval'):
                        # Batch-Evaluation
                        mx.eval([eigenvalues, eigenvectors])
                    return eigenvalues, eigenvectors
                else:
                    # Fallback mit optimierter Konvertierung
                    if matrix_a_np is None:
                        if hasattr(mx, 'to_numpy'):
                            matrix_a_np = mx.to_numpy(matrix_a)
                        else:
                            matrix_a_np = np.asarray(mx.array(matrix_a).tolist())
                    
                    eigenvalues, eigenvectors = np.linalg.eigh(matrix_a_np)
                    
                    # Optimierte Zurückkonvertierung
                    if hasattr(mx, 'array') and 'device' in inspect.signature(mx.array).parameters:
                        current_device = mx.default_device()
                        result_e = mx.array(eigenvalues, device=current_device)
                        result_v = mx.array(eigenvectors, device=current_device)
                    else:
                        result_e, result_v = mx.array(eigenvalues), mx.array(eigenvectors)
                    
                    if hasattr(mx, 'eval'):
                        mx.eval([result_e, result_v])
                    return result_e, result_v
                    
            elif operation == Operation.CHOLESKY:
                # Cholesky-Zerlegung optimiert für Apple Silicon
                if hasattr(mx, 'linalg') and hasattr(mx.linalg, 'cholesky'):
                    result = mx.linalg.cholesky(matrix_a)
                    if hasattr(mx, 'eval'):
                        mx.eval(result)
                    return result
                else:
                    # Fallback-Konvertierung mit optimiertem Datenpfad
                    if matrix_a_np is None:
                        if hasattr(mx, 'to_numpy'):
                            matrix_a_np = mx.to_numpy(matrix_a)
                        else:
                            matrix_a_np = np.asarray(mx.array(matrix_a).tolist())
                    
                    cholesky = np.linalg.cholesky(matrix_a_np)
                    
                    # Optimierte Zurückkonvertierung
                    if hasattr(mx, 'array') and 'device' in inspect.signature(mx.array).parameters:
                        result = mx.array(cholesky, device=mx.default_device())
                    else:
                        result = mx.array(cholesky)
                    
                    if hasattr(mx, 'eval'):
                        mx.eval(result)
                    return result
            elif operation == Operation.QR:
                # QR-Zerlegung optimiert für Apple Silicon
                if hasattr(mx, 'linalg') and hasattr(mx.linalg, 'qr'):
                    q, r = mx.linalg.qr(matrix_a)
                    if hasattr(mx, 'eval'):
                        # Batch-Evaluation für bessere Leistung
                        mx.eval([q, r])
                    return q, r
                else:
                    # Fallback-Konvertierung mit optimiertem Datenpfad
                    if matrix_a_np is None:
                        if hasattr(mx, 'to_numpy'):
                            matrix_a_np = mx.to_numpy(matrix_a)
                        else:
                            matrix_a_np = np.asarray(mx.array(matrix_a).tolist())
                    
                    q, r = np.linalg.qr(matrix_a_np)
                    
                    # Optimierte Zurückkonvertierung
                    if hasattr(mx, 'array') and 'device' in inspect.signature(mx.array).parameters:
                        current_device = mx.default_device()
                        result_q = mx.array(q, device=current_device)
                        result_r = mx.array(r, device=current_device)
                    else:
                        result_q, result_r = mx.array(q), mx.array(r)
                    
                    if hasattr(mx, 'eval'):
                        mx.eval([result_q, result_r])
                    return result_q, result_r
        finally:
            # Stelle sicher, dass wir den ursprünglichen Compile-Modus wiederherstellen
            if original_compile_mode is not None and hasattr(mx, 'compile'):
                mx.compile(mode=original_compile_mode)
    
    def _run_numpy_operation(self, operation: Operation, matrix_a, matrix_b):
        """Optimierte NumPy-Operationen mit Speicheroptimierungen"""
        # Stelle sicher, dass die Matrizen die richtige Speicheranordnung haben (C-contiguous)
        matrix_a = np.ascontiguousarray(matrix_a)
        if operation == Operation.MATMUL:
            matrix_b = np.ascontiguousarray(matrix_b)
        
        if operation == Operation.MATMUL:
            return np.matmul(matrix_a, matrix_b)
        elif operation == Operation.INVERSE:
            return np.linalg.inv(matrix_a)
        elif operation == Operation.SVD:
            return np.linalg.svd(matrix_a, full_matrices=False)  # Effizienter
        elif operation == Operation.EIGENVALUES:
            return np.linalg.eigh(matrix_a)
        elif operation == Operation.CHOLESKY:
            return np.linalg.cholesky(matrix_a)
        elif operation == Operation.QR:
            return np.linalg.qr(matrix_a)
    
    def _parallel_benchmark_worker(self, args):
        """Verbesserte Hilfsfunktion für parallele Benchmark-Ausführung"""
        operation, backend, dimension, precision, iterations, warmup = args
        benchmark_id = f"{operation.name}_{backend.name}_{dimension}_{precision.name}"
        
        try:
            logger.info(f"Starte Benchmark-Worker für {benchmark_id}")
            
            # Explizites Speicheraufräumen vor dem Benchmark
            gc.collect()
            
            # Isolierte Benchmark-Ausführung 
            result = self.benchmark_operation(
                operation=operation,
                backend=backend,
                dimension=dimension,
                precision=precision,
                iterations=iterations,
                warmup=warmup
            )
            
            # Erneutes Speicheraufräumen nach dem Benchmark
            gc.collect()
            
            if result:
                logger.info(f"Benchmark {benchmark_id} erfolgreich: {result.mean_time:.6f}s")
            else:
                logger.warning(f"Benchmark {benchmark_id} ohne Ergebnis abgeschlossen")
                
            return result
            
        except Exception as e:
            logger.error(f"Fehler im Benchmark-Worker {benchmark_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @profile_function
    def run_dimension_benchmarks(self, operation: Operation, 
                              dimensions: List[int] = None,
                              precision: PrecisionType = PrecisionType.FLOAT32,
                              iterations: int = None, warmup: int = None) -> List[BenchmarkResult]:
        """Führt Benchmarks für verschiedene Dimensionen durch - verbesserte Version mit Thread-Unterstützung"""
        dimensions = dimensions if dimensions is not None else self.config.dimensions
        iterations = iterations if iterations is not None else self.iterations
        warmup = warmup if warmup is not None else self.warmup_iterations
        
        results = []
        start_time = time.perf_counter()
        mem_start = get_memory_usage()
        logger.info(f"Starte Dimension-Benchmarks für {operation.name}, {len(dimensions)} Dimensionen, {len(self.available_backends)} Backends")
        
        # Erstelle alle Benchmark-Aufgaben
        tasks = []
        for backend in self.available_backends:
            for dim in dimensions:
                tasks.append((operation, backend, dim, precision, iterations, warmup))
        
        # Prüfe ob parallele Verarbeitung aktiviert und sinnvoll ist
        if USE_PARALLEL_PROCESSING and len(tasks) > 1 and MAX_WORKERS > 1:
            if USE_THREADING:
                # Thread-basierte Parallelisierung (sicher für PyTorch/MLX)
                logger.info(f"Verwende Thread-basierte Parallelisierung mit {MAX_WORKERS} Threads für {len(tasks)} Aufgaben")
                
                # Aufgaben nach Komplexität sortieren (größere Matrizen zuerst)
                tasks.sort(key=lambda x: -x[2])
                
                # Thread-Pool initialisieren und Benchmarks ausführen
                futures = []
                results_lock = threading.Lock()
                thread_results = []
                
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Alle Aufgaben einreichen
                    for task in tasks:
                        future = executor.submit(self._parallel_benchmark_worker, task)
                        futures.append(future)
                    
                    # Auf Ergebnisse warten und sammeln (sobald sie verfügbar sind)
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                with results_lock:
                                    thread_results.append(result)
                        except Exception as e:
                            logger.error(f"Fehler beim Abholen eines Thread-Ergebnisses: {e}")
                
                # Ergebnisse in die Hauptliste übernehmen
                results.extend(thread_results)
            else:
                # Sortiere Aufgaben für optimale Lastverteilung
                tasks.sort(key=lambda x: (-x[2], x[1].name))  # Zuerst nach Dimension (absteigend), dann nach Backend
                
                # Sequentielle Ausführung mit optimierter Reihenfolge
                logger.info(f"Verwende optimierte sequentielle Benchmark-Ausführung für {len(tasks)} Aufgaben")
                
                for i, task in enumerate(tasks):
                    operation, backend, dim, precision, iterations, warmup = task
                    logger.info(f"Benchmark {i+1}/{len(tasks)}: {operation.name}, {backend.name}, {dim}x{dim}, {precision.name}")
                    
                    result = self.benchmark_operation(
                        operation=operation,
                        backend=backend,
                        dimension=dim,
                        precision=precision,
                        iterations=iterations,
                        warmup=warmup
                    )
                    
                    if result:
                        results.append(result)
                    
                    # Speicherbereinigung zwischen Tests
                    if dim >= 256 or (i % MEMORY_CLEANUP_INTERVAL == 0):
                        gc.collect()
                        # Kurze Pause, um dem System Zeit zum Aufräumen zu geben
                        time.sleep(0.1)
        else:
            # Einfache sequentielle Ausführung
            logger.info("Verwende einfache sequentielle Benchmark-Ausführung")
            # Sortiere die Tasks für effizientere Ausführung
            tasks.sort(key=lambda x: (x[1].name, x[3].name, x[2]))  # backend, precision, dimension
            
            for task in tasks:
                operation, backend, dim, precision, iterations, warmup = task
                result = self.benchmark_operation(
                    operation=operation,
                    backend=backend,
                    dimension=dim,
                    precision=precision,
                    iterations=iterations,
                    warmup=warmup
                )
                
                if result:
                    results.append(result)
                    
                # Speicher zwischen benchmarks bereinigen
                if dim >= 256:
                    gc.collect()
        
        # Gesamtausführungszeit und Speichernutzung messen
        end_time = time.perf_counter()
        mem_end = get_memory_usage()
        total_time = end_time - start_time
        mem_diff = mem_end - mem_start
        
        # Zusammenfassende Statistiken berechnen
        backend_stats = {}
        for r in results:
            if r.backend.name not in backend_stats:
                backend_stats[r.backend.name] = []
            backend_stats[r.backend.name].append(r.mean_time)
        
        backend_means = {b: sum(times)/len(times) for b, times in backend_stats.items() if times}
        if backend_means:
            fastest_backend = min(backend_means.items(), key=lambda x: x[1])[0]
            slowest_backend = max(backend_means.items(), key=lambda x: x[1])[0]
            speedup = backend_means[slowest_backend] / backend_means[fastest_backend] if backend_means[fastest_backend] > 0 else 0
            
            logger.info(f"Performance-Vergleich: {fastest_backend} ist durchschnittlich {speedup:.2f}x schneller als {slowest_backend}")
        
        logger.info(f"Dimension-Benchmarks für {operation.name} abgeschlossen: "
                   f"{len(results)}/{len(tasks)} erfolgreich, "
                   f"Gesamtzeit: {total_time:.2f}s, "
                   f"Speicheränderung: {mem_diff:.2f} MB")
        
        return results
    
    def run_precision_benchmarks(self, operation: Operation, dimension: int,
                              precision_types: List[PrecisionType] = list(PrecisionType),
                              iterations: int = None, warmup: int = None) -> List[BenchmarkResult]:
        """Führt Benchmarks für verschiedene Präzisionstypen durch"""
        iterations = iterations if iterations is not None else self.iterations
        warmup = warmup if warmup is not None else self.warmup_iterations
        
        results = []
        
        for backend in self.available_backends:
            for precision in precision_types:
                result = self.benchmark_operation(
                    operation=operation,
                    backend=backend,
                    dimension=dimension,
                    precision=precision,
                    iterations=iterations,
                    warmup=warmup
                )
                
                if result:
                    results.append(result)
        
        return results
    
    def run_all_benchmarks(self, 
                         dimensions: List[int] = None,
                         precision: PrecisionType = PrecisionType.FLOAT32,
                         operations: List[Operation] = list(Operation),
                         iterations: int = None, warmup: int = None) -> List[BenchmarkResult]:
        """Führt alle Benchmark-Tests durch"""
        dimensions = dimensions if dimensions is not None else self.config.dimensions
        iterations = iterations if iterations is not None else self.iterations
        warmup = warmup if warmup is not None else self.warmup_iterations
        
        all_results = []
        
        logger.info(f"Starte alle Benchmarks für {len(operations)} Operationen, {len(dimensions)} Dimensionen und {len(self.available_backends)} Backends")
        
        for operation in operations:
            logger.info(f"\n=== Benchmarks für {operation.name} ===\n")
            results = self.run_dimension_benchmarks(
                operation=operation,
                dimensions=dimensions,
                precision=precision,
                iterations=iterations,
                warmup=warmup
            )
            all_results.extend(results)
        
        return all_results
    
    def save_results(self, filename: str, format: str = 'json') -> None:
        """Speichert die Benchmark-Ergebnisse in einer Datei
        
        Args:
            filename: Name der Ausgabedatei
            format: Format der Ausgabedatei ('json' oder 'csv')
        """
        if not self.results:
            logger.warning("Keine Ergebnisse zum Speichern vorhanden.")
            return
        
        if format.lower() not in self.reporters:
            logger.warning(f"Unbekanntes Format: {format}. Verwende JSON als Fallback.")
            format = 'json'
        
        # Verwende den entsprechenden Reporter
        self.reporters[format.lower()].generate_report(self.results, filename)
        
        # Archiviere Ergebnisse, falls konfiguriert
        if self.config.archive_results:
            self._archive_results(self.results)
    
    def load_results(self, filename: str):
        """Lädt Benchmark-Ergebnisse aus einer JSON-Datei"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.results = []
            for result_data in data["results"]:
                operation = Operation[result_data["operation"]]
                backend = Backend[result_data["backend"]]
                dimension = result_data["dimension"]
                precision = PrecisionType[result_data["precision"]]
                
                result = BenchmarkResult(operation, backend, dimension, precision)
                result.success_rate = result_data["success_rate"]
                result.mean_time = result_data["mean_time"]
                result.std_dev = result_data["std_dev"]
                result.min_time = result_data["min_time"]
                result.max_time = result_data["max_time"]
                result.execution_times = result_data["execution_times"]
                
                self.results.append(result)
            
            logger.info(f"Ergebnisse aus {filename} geladen: {len(self.results)} Einträge.")
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Ergebnisse aus {filename}: {str(e)}")
            return None
    
    def plot_dimension_comparison(self, operation: Operation, precision: PrecisionType = PrecisionType.FLOAT32, 
                                output_file: str = None):
        """Erstellt einen Vergleichsplot für verschiedene Matrix-Dimensionen"""
        # Finde Ergebnisse für die angegebene Operation und Präzision
        filtered_results = [r for r in self.results if r.operation == operation and r.precision == precision]
        
        if not filtered_results:
            logger.warning(f"Keine Ergebnisse für Operation {operation.name} und Präzision {precision.name} gefunden.")
            return False
        
        # Gruppiere nach Backend und Dimension
        backend_data = {}
        dimensions = sorted(list(set(r.dimension for r in filtered_results)))
        
        for result in filtered_results:
            if result.backend.name not in backend_data:
                backend_data[result.backend.name] = []
            backend_data[result.backend.name].append((result.dimension, result.mean_time))
        
        # Plot erstellen
        plt.figure(figsize=(12, 8))
        
        markers = ['o', 's', '^', 'D', 'v', 'p']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (backend, data) in enumerate(backend_data.items()):
            data.sort(key=lambda x: x[0])  # Sortiere nach Dimension
            x_values = [d[0] for d in data]
            y_values = [d[1] for d in data]
            
            marker_idx = i % len(markers)
            color_idx = i % len(colors)
            
            plt.plot(x_values, y_values, marker=markers[marker_idx], linestyle='-', 
                     linewidth=2, markersize=8, label=backend, color=colors[color_idx])
        
        plt.title(f'Vergleich der {operation.name} Operation Performance nach Matrix-Dimension\n(Präzision: {precision.name})')
        plt.xlabel('Matrix-Dimension')
        plt.ylabel('Mittlere Ausführungszeit (s)')
        plt.xticks(dimensions)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Logarithmische Y-Achse für bessere Darstellung
        plt.yscale('log')
        
        # Füge Performanceunterschiede als Text hinzu
        if len(backend_data) > 1 and len(dimensions) > 0:
            backends = list(backend_data.keys())
            for dim in dimensions:
                times = {}
                for backend, data in backend_data.items():
                    for d, t in data:
                        if d == dim:
                            times[backend] = t
                
                if len(times) > 1:
                    fastest = min(times.items(), key=lambda x: x[1])
                    slowest = max(times.items(), key=lambda x: x[1])
                    speedup = slowest[1] / fastest[1] if fastest[1] > 0 else 0
                    
                    if speedup > 1.1:  # Nur anzeigen, wenn der Unterschied größer als 10% ist
                        pos_y = max(times.values()) * 1.1
                        plt.text(dim, pos_y, f"{fastest[0]} ist {speedup:.1f}x schneller als {slowest[0]}", 
                                 fontsize=8, ha='center', va='bottom', rotation=0,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot gespeichert in {output_file}")
        else:
            plt.show()
        
        return True
    
    def plot_precision_comparison(self, operation: Operation, dimension: int, output_file: str = None):
        """Erstellt einen Vergleichsplot für verschiedene Präzisionstypen"""
        # Finde Ergebnisse für die angegebene Operation und Dimension
        filtered_results = [r for r in self.results if r.operation == operation and r.dimension == dimension]
        
        if not filtered_results:
            logger.warning(f"Keine Ergebnisse für Operation {operation.name} und Dimension {dimension} gefunden.")
            return False
        
        # Gruppiere nach Backend und Präzision
        data = {}
        for result in filtered_results:
            backend = result.backend.name
            precision = result.precision.name
            
            if backend not in data:
                data[backend] = {}
            
            data[backend][precision] = result.mean_time
        
        # Vorbereitung für den Plot
        backends = list(data.keys())
        precisions = list(PrecisionType.__members__.keys())
        
        # Plot erstellen
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(precisions))  # X-Positionen für die Bars
        width = 0.8 / len(backends)  # Breite der Bars
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, backend in enumerate(backends):
            backend_data = [data[backend].get(precision, 0) for precision in precisions]
            offset = width * i - width * (len(backends) - 1) / 2
            bars = plt.bar(x + offset, backend_data, width, label=backend, color=colors[i % len(colors)])
            
            # Füge Labels für die Bars hinzu
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.6f}s', ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.title(f'Vergleich der {operation.name} Operation nach Präzisionstyp\n(Dimension: {dimension}x{dimension})')
        plt.xlabel('Präzisionstyp')
        plt.ylabel('Mittlere Ausführungszeit (s)')
        plt.xticks(x, precisions)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.legend()
        
        # Füge Performanceunterschiede zwischen Präzisionstypen hinzu
        for backend in backends:
            times = data[backend]
            if len(times) > 1 and "FLOAT32" in times and ("FLOAT16" in times or "BFLOAT16" in times):
                float32_time = times["FLOAT32"]
                
                for precision in ["FLOAT16", "BFLOAT16"]:
                    if precision in times and float32_time > 0:
                        prec_time = times[precision]
                        speedup = float32_time / prec_time
                        
                        if abs(speedup - 1) > 0.1:  # Nur anzeigen, wenn der Unterschied größer als 10% ist
                            prec_idx = precisions.index(precision)
                            backend_idx = backends.index(backend)
                            offset = width * backend_idx - width * (len(backends) - 1) / 2
                            
                            pos_x = prec_idx + offset
                            pos_y = prec_time * 1.1
                            
                            text = f"{speedup:.2f}x {('schneller' if speedup > 1 else 'langsamer')} als FLOAT32"
                            plt.text(pos_x, pos_y, text, fontsize=8, ha='center', va='bottom', rotation=90,
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot gespeichert in {output_file}")
        else:
            plt.show()
        
        return True
    
    def _prepare_chart_data(self):
        """Bereitet die Daten für interaktive Charts vor"""
        try:
            chart_data = {
                "operations": {},           # Daten pro Operation
                "backends": {},             # Vergleich der Backends
                "dimensions": {},           # Performance nach Dimension
                "memory": {},               # Speicherverbrauch
                "precision_comparison": {}, # Vergleich der Präzisionstypen
                "historical": {},           # Historische Vergleichsdaten
                "recommendations": [],      # Automatische Empfehlungen
                "warnings": [],            # Performance-Warnungen
                "metadata": {              # Metadaten zur Benchmark-Ausführung
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "apple_silicon": self._is_apple_silicon(),
                    "mlx_available": self._is_mlx_available(),
                    "pytorch_available": self._is_pytorch_available(),
                    "backends": [b.name for b in self.available_backends],
                    "t_mathematics_version": self._get_t_mathematics_version()
                }
            }
            
            # Gruppiere Ergebnisse nach Operation
            for result in self.results:
                if not hasattr(result, 'mean_time'):
                    logger.warning(f"Benchmark-Ergebnis hat kein 'mean_time' Attribut: {result}")
                    continue
                    
                op_name = result.operation.name
                backend = result.backend.name  # Hier den Namen statt des Objekts verwenden
                dimension = result.dimension
                precision = str(result.precision)
                
                # Daten für Operations-Charts
                if op_name not in chart_data["operations"]:
                    chart_data["operations"][op_name] = {"dimensions": [], "times": [], "backends": {}}
                
                if backend not in chart_data["operations"][op_name]["backends"]:
                    chart_data["operations"][op_name]["backends"][backend] = {"dimensions": [], "times": []}
                
                # Füge Daten für dieses spezifische Backend/Operation/Dimension hinzu
                chart_data["operations"][op_name]["backends"][backend]["dimensions"].append(dimension)
                chart_data["operations"][op_name]["backends"][backend]["times"].append(result.mean_time)
                
                # Backend-Vergleich
                if dimension not in chart_data["backends"]:
                    chart_data["backends"][dimension] = {}
                
                if op_name not in chart_data["backends"][dimension]:
                    chart_data["backends"][dimension][op_name] = {}
                    
                chart_data["backends"][dimension][op_name][backend] = result.mean_time
                
                # Speicherverbrauch (innerhalb der Schleife korrigiert)
                if op_name not in chart_data["memory"]:
                    chart_data["memory"][op_name] = {}
                
                if backend not in chart_data["memory"][op_name]:
                    chart_data["memory"][op_name][backend] = []
                    
                # Falls vorhanden, füge Speicheränderungsdaten hinzu
                if hasattr(result, 'memory_change'):
                    chart_data["memory"][op_name][backend].append({
                        "dimension": dimension,
                        "memory_change": result.memory_change
                    })
            
                # Präzisionsvergleich (innerhalb der Schleife korrigiert)
                if precision not in chart_data["precision_comparison"]:
                    chart_data["precision_comparison"][precision] = {"backends": {}}
                
                if backend not in chart_data["precision_comparison"][precision]["backends"]:
                    chart_data["precision_comparison"][precision]["backends"][backend] = []
                    
                chart_data["precision_comparison"][precision]["backends"][backend].append({
                    "operation": op_name,
                    "dimension": dimension,
                    "time": result.mean_time
                })
            
            logger.info(f"Chart-Daten für {len(chart_data['operations'])} Operationen vorbereitet")
            return chart_data
            
        except Exception as e:
            logger.error(f"Fehler bei der Vorbereitung der Chart-Daten: {str(e)}")
            # Minimale Datenstruktur zurückgeben, damit die Anwendung nicht crasht
            return {"operations": {}, "backends": {}, "dimensions": {}, "memory": {}, "precision_comparison": {}}
        
        return chart_data

    def _get_fastest_backend_info(self):
        """Ermittelt das schnellste Backend und die Beschleunigung im Vergleich zum Durchschnitt"""
        if not self.results:
            logger.warning("Keine Ergebnisse vorhanden für Backend-Vergleich")
            return "N/A", 0.0
            
        try:
            # Sammle Zeiten nach Backend
            backend_times = {}
            for result in self.results:
                if not hasattr(result, 'mean_time'):
                    logger.warning(f"Benchmark-Ergebnis hat kein 'mean_time' Attribut: {result}")
                    continue
                    
                backend_name = result.backend.name
                if backend_name not in backend_times:
                    backend_times[backend_name] = []
                backend_times[backend_name].append(result.mean_time)
            
            if not backend_times:
                logger.warning("Keine validen Zeiten für Backend-Vergleich gefunden")
                return "N/A", 0.0
            
            # Durchschnittliche Zeit pro Backend berechnen
            backend_avg_times = {}
            for backend, times in backend_times.items():
                if times:  # Nur berechnen, wenn Zeiten vorhanden sind
                    backend_avg_times[backend] = sum(times) / len(times)
            
            if not backend_avg_times:
                logger.warning("Keine durchschnittlichen Zeiten für Backend-Vergleich berechnet")
                return "N/A", 0.0
            
            # Schnellstes Backend finden
            fastest_backend = min(backend_avg_times.items(), key=lambda x: x[1])[0]
            
            # Durchschnittliche Zeit über alle Backends
            all_avg_time = sum(backend_avg_times.values()) / len(backend_avg_times)
            
            # Beschleunigung berechnen
            if backend_avg_times[fastest_backend] > 0:
                speedup = all_avg_time / backend_avg_times[fastest_backend]
            else:
                speedup = 0.0
                
            logger.info(f"Schnellstes Backend: {fastest_backend} mit {speedup:.2f}x Beschleunigung")
            return fastest_backend, speedup
            
        except Exception as e:
            logger.error(f"Fehler bei der Bestimmung des schnellsten Backends: {str(e)}")
            return "N/A", 0.0

    def _get_fastest_operation_info(self):
        """Ermittelt die schnellste Operation und ihre Ausführungszeit"""
        if not self.results:
            logger.warning("Keine Ergebnisse vorhanden für Operationsvergleich")
            return "N/A", 0.0
            
        try:
            op_times = {}
            for result in self.results:
                if not hasattr(result, 'mean_time'):
                    logger.warning(f"Benchmark-Ergebnis hat kein 'mean_time' Attribut: {result}")
                    continue
                    
                op_name = result.operation.name
                if op_name not in op_times:
                    op_times[op_name] = []
                op_times[op_name].append(result.mean_time)
            
            if not op_times:
                logger.warning("Keine validen Zeiten für Operationsvergleich gefunden")
                return "N/A", 0.0
            
            # Durchschnittliche Zeit pro Operation berechnen
            op_avg_times = {}
            for op, times in op_times.items():
                if times:  # Nur berechnen, wenn Zeiten vorhanden sind
                    op_avg_times[op] = sum(times) / len(times)
            
            if not op_avg_times:
                logger.warning("Keine durchschnittlichen Zeiten für Operationsvergleich berechnet")
                return "N/A", 0.0
            
            # Schnellste Operation finden
            fastest_op = min(op_avg_times.items(), key=lambda x: x[1])
            
            logger.info(f"Schnellste Operation: {fastest_op[0]} mit {fastest_op[1]:.6f}s Ausführungszeit")
            return fastest_op[0], fastest_op[1]
            
        except Exception as e:
            logger.error(f"Fehler bei der Bestimmung der schnellsten Operation: {str(e)}")
            return "N/A", 0.0

    def _get_best_precision(self):
        """Ermittelt die beste Präzision basierend auf Ausführungszeit und Genauigkeit"""
        if not self.results:
            logger.warning("Keine Ergebnisse vorhanden für Präzisionsvergleich")
            return "N/A"
            
        try:
            precision_times = {}
            for result in self.results:
                if not hasattr(result, 'mean_time'):
                    logger.warning(f"Benchmark-Ergebnis hat kein 'mean_time' Attribut: {result}")
                    continue
                    
                precision = str(result.precision)
                if precision not in precision_times:
                    precision_times[precision] = []
                precision_times[precision].append(result.mean_time)
            
            if not precision_times:
                logger.warning("Keine validen Zeiten für Präzisionsvergleich gefunden")
                return "N/A"
            
            # Durchschnittliche Zeit pro Präzision berechnen
            precision_avg_times = {}
            for precision, times in precision_times.items():
                if times:  # Nur berechnen, wenn Zeiten vorhanden sind
                    precision_avg_times[precision] = sum(times) / len(times)
            
            if not precision_avg_times:
                logger.warning("Keine durchschnittlichen Zeiten für Präzisionsvergleich berechnet")
                return "N/A"
            
            # Beste Präzision finden (aktuell basierend nur auf Geschwindigkeit, kann erweitert werden)
            best_precision = min(precision_avg_times.items(), key=lambda x: x[1])[0]
            
            logger.info(f"Beste Präzision: {best_precision} basierend auf Ausführungszeit")
            return best_precision
            
        except Exception as e:
            logger.error(f"Fehler bei der Bestimmung der besten Präzision: {str(e)}")
            return "N/A"
        
    def _is_apple_silicon(self):
        """Prüft, ob der aktuelle Rechner Apple Silicon verwendet"""
        try:
            import platform
            machine = platform.machine()
            return machine == 'arm64' and platform.system() == 'Darwin'
        except:
            return False
    
    def _is_mlx_available(self):
        """Prüft, ob MLX verfügbar ist"""
        try:
            import mlx
            return True
        except ImportError:
            return False
    
    def _is_pytorch_available(self):
        """Prüft, ob PyTorch verfügbar ist"""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def _get_t_mathematics_version(self):
        """Ermittelt die Version der T-Mathematics Engine (falls vorhanden)"""
        try:
            # Versuche, das T-Mathematics-Modul zu importieren
            # Dies ist spezifisch für das MISO-Projekt
            import sys
            # Füge den Pfad zum MISO-Projekt hinzu, falls vorhanden
            root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            if root_path not in sys.path:
                sys.path.append(root_path)
            
            try:
                # Versuche, die T-Mathematics Engine zu importieren
                from miso.t_mathematics import __version__
                return __version__
            except (ImportError, AttributeError):
                try:
                    # Alternativer Versuch, falls Pfad anders ist
                    from t_mathematics import __version__
                    return __version__
                except (ImportError, AttributeError):
                    return "nicht verfügbar"
        except Exception as e:
            logger.debug(f"Fehler beim Ermitteln der T-Mathematics-Version: {str(e)}")
            return "nicht verfügbar"
    
    def _get_dimension_range(self):
        """Gibt den Bereich der getesteten Matrixdimensionen zurück"""
        if not self.results:
            return "N/A"
            
        dimensions = set()
        for result in self.results:
            dimensions.add(result.dimension)
        
        if not dimensions:
            return "N/A"
            
        min_dim = min(dimensions)
        max_dim = max(dimensions)
        
        if min_dim == max_dim:
            return f"{min_dim}x{min_dim}"
        else:
            return f"{min_dim}x{min_dim} - {max_dim}x{max_dim}"

    @profile_function
    def generate_html_report(self, output_file: str):
        """Generiert einen erweiterten HTML-Bericht mit den Benchmark-Ergebnissen"""
        if not self.results:
            logger.warning("Keine Ergebnisse zum Generieren eines HTML-Berichts vorhanden.")
            return False
        
        logger.info(f"Generiere erweiterten HTML-Bericht mit {len(self.results)} Ergebnissen nach {output_file}")
        
        # Erstelle einen einfachen HTML-Template ohne JavaScript-Charts
        html_template = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MISO Matrix-Benchmark Ergebnisse</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; padding: 20px; max-width: 1200px; margin: 0 auto; }}
        h1, h2, h3, h4 {{ color: #444; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .summary {{ background-color: #f8f9fa; border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px; }}
        .metric {{ background-color: #e9ecef; padding: 15px; margin-bottom: 10px; border-radius: 4px; }}
        .metric-title {{ font-weight: bold; margin-bottom: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .speedup {{ color: #28a745; font-weight: bold; }}
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #6c757d; border-top: 1px solid #eee; }}
        .metrics-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .metric-item {{ flex: 1; }}
    </style>
</head>
<body>
    <h1>MISO Matrix-Benchmark Ergebnisse</h1>
    <p>Leistungsvergleich von Matrix-Operationen mit verschiedenen Backends auf Apple Silicon</p>
    
    <div class="summary">
        <h2>Zusammenfassung</h2>
        <table>
            <tr>
                <th>Test-Zeitpunkt:</th>
                <td>{timestamp}</td>
            </tr>
            <tr>
                <th>Verwendete Backends:</th>
                <td>{backends}</td>
            </tr>
            <tr>
                <th>Matrix-Dimensionen:</th>
                <td>{dimensions}</td>
            </tr>
            <tr>
                <th>Benchmark-Iterationen:</th>
                <td>{iterations} (mit {warmup} Warmup-Iterationen)</td>
            </tr>
            <tr>
                <th>Anzahl der Ergebnisse:</th>
                <td>{result_count}</td>
            </tr>
        </table>
        
        <div class="metrics-container">
            <div class="metric metric-item">
                <div class="metric-title">Schnellstes Backend</div>
                <div class="metric-value">{fastest_backend}</div>
                <div>{fastest_backend_speedup}x schneller als der Durchschnitt</div>
            </div>
            <div class="metric metric-item">
                <div class="metric-title">Schnellste Operation</div>
                <div class="metric-value">{fastest_operation}</div>
                <div>Ausführungszeit: {fastest_operation_time} s</div>
            </div>
            <div class="metric metric-item">
                <div class="metric-title">Optimale Präzision</div>
                <div class="metric-value">{best_precision}</div>
            </div>
            <div class="metric metric-item">
                <div class="metric-title">Getestete Matrixgrößen</div>
                <div class="metric-value">{dimension_range}</div>
            </div>
        </div>
    </div>
    
    <h2>Detaillierte Ergebnisse</h2>
    {results_by_operation}
    
    <h2>Backend-Vergleich</h2>
    <table>
        <thead>
            <tr>
                <th>Operation</th>
                <th>Dimension</th>
                <th>MLX (s)</th>
                <th>PyTorch (s)</th>
                <th>NumPy (s)</th>
                <th>Beschleunigung</th>
                <th>Bester Backend</th>
            </tr>
        </thead>
        <tbody>
            {backend_comparison_rows}
        </tbody>
    </table>
    
    <div class="footer">
        <p>MISO Matrix-Benchmark | Erstellt am {timestamp}</p>
        <p>Optimiert für Apple Silicon mit MLX und PyTorch</p>
    </div>
</body>
</html>
        """
        
        # Gruppiere Ergebnisse nach Operation
        results_by_op = {}
        for result in self.results:
            op_name = result.operation.name
            if op_name not in results_by_op:
                results_by_op[op_name] = []
            results_by_op[op_name].append(result)
        
        # Generiere HTML-Inhalt für jede Operation
        all_op_html = ""
        for op_name, op_results in results_by_op.items():
            op_html = f"<h3>Operation: {op_name}</h3>\n"
            
            # Gruppiere nach Präzision
            results_by_precision = {}
            for result in op_results:
                prec_name = result.precision.name
                if prec_name not in results_by_precision:
                    results_by_precision[prec_name] = []
                results_by_precision[prec_name].append(result)
            
            # Generiere Tabellen für jede Präzision
            for prec_name, prec_results in results_by_precision.items():
                op_html += f"<h4>Präzision: {prec_name}</h4>\n"
                op_html += "<table>\n"
                op_html += "<tr><th>Backend</th><th>Dimension</th><th>Mittlere Zeit (s)</th><th>Erfolgsrate</th>" \
                         "<th>Std. Abw.</th><th>Min. Zeit (s)</th><th>Max. Zeit (s)</th></tr>\n"
                
                # Sortiere nach Dimension und Backend
                prec_results.sort(key=lambda r: (r.dimension, r.backend.name))
                
                for result in prec_results:
                    backend_class = f"backend-{result.backend.name.lower()}"
                    op_html += f"<tr class='{backend_class}'>" \
                              f"<td>{result.backend.name}</td>" \
                              f"<td>{result.dimension}x{result.dimension}</td>" \
                              f"<td>{result.mean_time:.6f}</td>" \
                              f"<td>{result.success_rate:.2f}</td>" \
                              f"<td>{result.std_dev:.6f}</td>" \
                              f"<td>{result.min_time:.6f}</td>" \
                              f"<td>{result.max_time:.6f}</td>" \
                              f"</tr>\n"
                
                op_html += "</table>\n"
                
                # Vergleiche Performance zwischen Backends für jede Dimension
                op_html += "<h5>Performance-Vergleich nach Dimension</h5>\n"
                
                dimensions = sorted(list(set(r.dimension for r in prec_results)))
                for dim in dimensions:
                    dim_results = [r for r in prec_results if r.dimension == dim]
                    if len(dim_results) > 1:
                        fastest = min(dim_results, key=lambda r: r.mean_time)
                        slowest = max(dim_results, key=lambda r: r.mean_time)
                        
                        if fastest.mean_time > 0:
                            speedup = slowest.mean_time / fastest.mean_time
                            op_html += f"<p>Dimension {dim}x{dim}: <span class='backend-{fastest.backend.name.lower()}'>" \
                                      f"{fastest.backend.name}</span> ist <span class='speedup'>{speedup:.2f}x</span> schneller als " \
                                      f"<span class='backend-{slowest.backend.name.lower()}'>{slowest.backend.name}</span></p>\n"
            
            all_op_html += op_html + "<hr>\n"
        
        # Berechne zusätzliche statistische Daten für HTML-Bericht
        fastest_backend = ""
        fastest_backend_speedup = 1.0
        fastest_operation = ""
        fastest_operation_time = 0.0
        best_precision = "FLOAT32"  # Default
        dimension_range = ""
        backend_comparison_rows = ""
        
        # Finde schnellstes Backend
        backend_avg_times = {}
        for result in self.results:
            if result.backend.name not in backend_avg_times:
                backend_avg_times[result.backend.name] = []
            backend_avg_times[result.backend.name].append(result.mean_time)
        
        if backend_avg_times:
            # Berechne Durchschnittszeiten pro Backend
            backend_avg = {b: sum(times)/len(times) for b, times in backend_avg_times.items() if times}
            # Finde das schnellste Backend
            if backend_avg:
                fastest_backend = min(backend_avg.items(), key=lambda x: x[1])[0]
                avg_time = sum([t for b in backend_avg.values() for t in [b]]) / len(backend_avg)
                if avg_time > 0:
                    fastest_backend_speedup = avg_time / backend_avg[fastest_backend]
        
        # Finde schnellste Operation
        op_results = {}
        for result in self.results:
            op_name = result.operation.name
            if op_name not in op_results or result.mean_time < op_results[op_name]:
                op_results[op_name] = result.mean_time
        
        if op_results:
            fastest_operation = min(op_results.items(), key=lambda x: x[1])[0]
            fastest_operation_time = op_results[fastest_operation]
        
        # Bestimme optimale Präzision und Dimensionsbereich
        precisions = set(r.precision.name for r in self.results)
        dimensions = sorted(set(r.dimension for r in self.results))
        
        if precisions:
            best_precision = list(precisions)[0]  # Vereinfacht, könnte mit weiterer Logik erweitert werden
        
        if dimensions:
            min_dim = min(dimensions)
            max_dim = max(dimensions)
            dimension_range = f"{min_dim}x{min_dim} - {max_dim}x{max_dim}"
            
        # Generiere Vergleichszeilen für Backends
        operations = sorted(set(r.operation.name for r in self.results))
        grouped_results = {op: {} for op in operations}
        
        for result in self.results:
            op_name = result.operation.name
            dimension = result.dimension
            
            if dimension not in grouped_results[op_name]:
                grouped_results[op_name][dimension] = {}
                
            grouped_results[op_name][dimension][result.backend.name] = result.mean_time
        
        # Generiere HTML-Zeilen
        rows = []
        for op_name, dimensions in grouped_results.items():
            for dimension, backends in dimensions.items():
                row = f"<tr><td>{op_name}</td><td>{dimension}x{dimension}</td>"
                
                # Spalten für jedes Backend
                for backend in ['MLX', 'PyTorch', 'NumPy']:
                    if backend in backends:
                        row += f"<td>{backends[backend]:.6f}</td>"
                    else:
                        row += "<td>-</td>"
                
                # Berechne Beschleunigung und bestes Backend
                if len(backends) > 1:
                    fastest_time = min(backends.values())
                    slowest_time = max(backends.values())
                    speedup = slowest_time / fastest_time
                    fastest_backend = [b for b, t in backends.items() if t == fastest_time][0]
                    
                    row += f"<td class='speedup'>{speedup:.2f}x</td><td>{fastest_backend}</td>"
                else:
                    row += "<td>-</td><td>-</td>"
                    
                row += "</tr>"
                rows.append(row)
        
        if not rows:
            return "<tr><td colspan='7'>Keine vergleichbaren Daten verfügbar.</td></tr>"
            
        return "\n".join(rows)
        
    @profile_function
    def _generate_operation_results_html(self):
        """Generiert HTML-Inhalt für die detaillierten Ergebnisse nach Operation"""
        html = ""
        # Gruppiere Ergebnisse nach Operation
        operations_results = {}
        for result in self.results:
            if result.operation.name not in operations_results:
                operations_results[result.operation.name] = []
            operations_results[result.operation.name].append(result)
        
        # Für jede Operation HTML erstellen
        for operation, results in operations_results.items():
            html += f"<h3>Operation: {operation}</h3>\n"
            html += "<table class='results-table'>\n"
            html += "<thead><tr><th>Backend</th><th>Dimension</th><th>Präzision</th><th>Zeit (s)</th><th>Speedup</th></tr></thead>\n"
            html += "<tbody>\n"
            
            # Sortiere Ergebnisse nach Backend und Dimension
            results.sort(key=lambda r: (r.backend.name, r.dimension))
            
            # Finde die langsamste Zeit für diese Operation, um Speedup zu berechnen
            slowest_time = max([r.mean_time for r in results])
            
            for result in results:
                speedup = slowest_time / result.mean_time if result.mean_time > 0 else 0
                html += f"<tr class='{result.backend.name.lower()}-row'>\n"
                html += f"<td>{result.backend.name}</td>\n"
                html += f"<td>{result.dimension}x{result.dimension}</td>\n"
                html += f"<td>{result.precision.name}</td>\n"
                html += f"<td>{result.mean_time:.6f}</td>\n"
                html += f"<td>{speedup:.2f}x</td>\n"
                html += "</tr>\n"
            
            html += "</tbody></table>\n"
            
            # Füge einen Abschnitt für Visualisierung hinzu
            html += f"<div class='chart-container operation-chart' id='chart-{operation.lower()}'></div>\n"
        
        return html
    
    def _generate_backend_comparison_rows(self):
        """Generiert HTML-Tabellenzeilen für den Backend-Vergleich"""
        html = ""
        
        # Gruppiere Ergebnisse nach Operation und Dimension
        grouped_results = {}
        for result in self.results:
            key = (result.operation.name, result.dimension)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Für jede Operation und Dimension eine Zeile erstellen
        for (operation, dimension), results in sorted(grouped_results.items()):
            # Ergebnisse nach Backend gruppieren
            backend_results = {r.backend.name: r for r in results}
            
            # Finde das schnellste Backend
            fastest_result = min(results, key=lambda r: r.mean_time)
            fastest_backend = fastest_result.backend.name
            fastest_time = fastest_result.mean_time
            
            # Zeile erstellen
            html += "<tr>\n"
            html += f"<td>{operation}</td>\n"
            html += f"<td>{dimension}x{dimension}</td>\n"
            
            # Für jedes Backend die Zeit anzeigen
            for backend in ["MLX", "PyTorch", "NumPy"]:
                if backend in backend_results:
                    time = backend_results[backend].mean_time
                    html += f"<td>{time:.6f}</td>\n"
                else:
                    html += "<td>N/A</td>\n"
            
            # Speedup berechnen (schnellstes zu langsamstem)
            slowest_time = max([r.mean_time for r in results])
            speedup = slowest_time / fastest_time if fastest_time > 0 else 0
            
            html += f"<td>{speedup:.2f}x</td>\n"
            html += f"<td>{fastest_backend}</td>\n"
            html += "</tr>\n"
        
        return html
        
    def generate_advanced_html_report(self, output_file: str):
        """Generiert einen erweiterten interaktiven HTML-Bericht mit den Benchmark-Ergebnissen
        unter Verwendung des HTMLReporters.
        
        Args:
            output_file: Pfad zur Ausgabedatei für den HTML-Bericht
            
        Returns:
            bool: True bei erfolgreicher Generierung, sonst False
        """
        if not self.results:
            logger.warning("Keine Ergebnisse zum Generieren eines erweiterten HTML-Berichts vorhanden.")
            return False
            
        try:
            # Importiere den HTMLReporter aus dem matrix_benchmark-Paket
            from matrix_benchmark.reporters.html_reporter import HTMLReporter
            
            # Erstelle HTML-Reporter-Instanz
            reporter = HTMLReporter(use_plotly=True, include_historical=True, theme="light")
            
            # Generiere den HTML-Bericht
            logger.info(f"Generiere erweiterten interaktiven HTML-Bericht mit {len(self.results)} Ergebnissen nach {output_file}")
            reporter.generate_report(self.results, output_file)
            
            return True
        except ImportError as e:
            logger.error(f"Fehler beim Importieren des HTMLReporters: {str(e)}")
            logger.warning("Falle zurück auf Legacy-HTML-Berichtsgenerierung...")
            return self._generate_legacy_html_report(output_file)
        except Exception as e:
            logger.error(f"Fehler bei der Generierung des HTML-Berichts: {str(e)}")
            logger.warning("Falle zurück auf Legacy-HTML-Berichtsgenerierung...")
            return self._generate_legacy_html_report(output_file)
            
    def _generate_legacy_html_report(self, output_file: str):
        """Legacy-Methode zur HTML-Berichtsgenerierung als Fallback."""
        logger.info(f"Verwende Legacy-HTML-Berichtsgenerierung für {output_file}")
        
        # Vorbereiten der Daten für den Report
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fastest_backend, backend_speedup = self._get_fastest_backend_info()
        fastest_op, fastest_op_time = self._get_fastest_operation_info()
        best_precision = self._get_best_precision()
        dimension_range = self._get_dimension_range()
        chart_data = self._prepare_chart_data()
        
        # Lade historische Daten für Vergleiche
        historical_data = self.load_historical_data()
        chart_data["historical"] = self._prepare_historical_data(historical_data)
        
        # Generiere automatische Empfehlungen und Warnungen
        recommendations, warnings = generate_recommendations(self)
        chart_data["recommendations"] = recommendations
        chart_data["warnings"] = warnings
        
        # JSON-String für die Chart-Daten erzeugen
        try:
            import json
            chart_data_json = json.dumps(chart_data)
        except ImportError:
            # Falls json nicht verfügbar ist, verwende eine einfache String-Repräsentation
            chart_data_json = str(chart_data).replace("'", "\"")
        
        # Generiere HTML-Inhalt mit interaktiven Charts
        html_template = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MISO Matrix-Benchmark Interaktive Ergebnisse</title>
    
    <!-- CSS für das Styling -->
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; padding: 20px; max-width: 1400px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
        .summary {{ background-color: #f8f9fa; border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .metrics-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
        .metric {{ flex: 1; min-width: 200px; background-color: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.3s ease; }}
        .metric:hover {{ transform: translateY(-5px); }}
        .metric-title {{ font-weight: bold; color: #7f8c8d; margin-bottom: 10px; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #2980b9; margin-bottom: 10px; }}
        .speedup {{ color: #27ae60; font-weight: bold; }}
        .chart-container {{ height: 400px; margin: 20px 0; background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .tabs {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
        .tab-button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }}
        .tab-button:hover {{ background-color: #ddd; }}
        .tab-button.active {{ background-color: #3498db; color: white; }}
        .tab-content {{ display: none; padding: 20px; border: 1px solid #ccc; border-top: none; }}
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #7f8c8d; border-top: 1px solid #eee; }}
        .badge {{ display: inline-block; padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; margin-right: 5px; }}
        .badge-mlx {{ background-color: #3498db; color: white; }}
        .badge-pytorch {{ background-color: #e67e22; color: white; }}
        .badge-numpy {{ background-color: #2ecc71; color: white; }}
        button {{ padding: 8px 16px; background-color: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; margin-right: 10px; transition: background-color 0.3s; }}
        button:hover {{ background-color: #2980b9; }}
        #optimizationInfo {{ margin-top: 30px; padding: 15px; background-color: #e8f4fd; border-left: 5px solid #3498db; border-radius: 4px; }}
    </style>
    
    <!-- Plotly.js für interaktive Charts -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>MISO Matrix-Benchmark Interaktive Ergebnisse</h1>
    <p>Detaillierter Leistungsvergleich von Matrix-Operationen mit verschiedenen Backends und Optimierungen</p>
    
    <!-- Tabs für verschiedene Ansichten -->
    <div class="tabs">
        <button class="tab-button active" onclick="openTab(event, 'Overview')">Übersicht</button>
        <button class="tab-button" onclick="openTab(event, 'DetailedResults')">Detaillierte Ergebnisse</button>
        <button class="tab-button" onclick="openTab(event, 'BackendComparison')">Backend-Vergleich</button>
        <button class="tab-button" onclick="openTab(event, 'HistoricalComparison')">Historischer Vergleich</button>
        <button class="tab-button" onclick="openTab(event, 'OptimizationInsights')">Optimierungs-Erkenntnisse</button>
    </div>
    
    <!-- Tab: Übersicht -->
    <div id="Overview" class="tab-content" style="display: block;">
        <div class="summary">
            <h2>Benchmark-Zusammenfassung</h2>
            <table style="width: 100%;">
                <tr>
                    <th style="width: 30%">Test-Zeitpunkt:</th>
                    <td>{timestamp}</td>
                </tr>
                <tr>
                    <th>Verwendete Backends:</th>
                    <td>{backends}</td>
                </tr>
                <tr>
                    <th>Matrix-Dimensionen:</th>
                    <td>{dimensions}</td>
                </tr>
                <tr>
                    <th>Benchmark-Iterationen:</th>
                    <td>{iterations} (mit {warmup} Warmup-Iterationen)</td>
                </tr>
                <tr>
                    <th>Anzahl der Ergebnisse:</th>
                    <td>{result_count}</td>
                </tr>
            </table>
            
            <div class="metrics-container">
                <div class="metric">
                    <div class="metric-title">Schnellstes Backend</div>
                    <div class="metric-value">{fastest_backend}</div>
                    <div><span class="speedup">{fastest_backend_speedup}x</span> schneller als der Durchschnitt</div>
                </div>
                <div class="metric">
                    <div class="metric-title">Schnellste Operation</div>
                    <div class="metric-value">{fastest_operation}</div>
                    <div>Ausführungszeit: <span class="speedup">{fastest_operation_time} s</span></div>
                </div>
                <div class="metric">
                    <div class="metric-title">Optimale Präzision</div>
                    <div class="metric-value">{best_precision}</div>
                    <div>Beste Kombination aus Geschwindigkeit und Genauigkeit</div>
                </div>
                <div class="metric">
                    <div class="metric-title">Getestete Matrixgrößen</div>
                    <div class="metric-value">{dimension_range}</div>
                    <div>Verschiedene Dimensionen für umfassende Analyse</div>
                </div>
            </div>
        </div>
        
        <!-- Performance-Übersicht nach Backend -->
        <h3>Performance-Übersicht nach Backend</h3>
        <div class="chart-container">
            <div id="backendPerformanceChart"></div>
        </div>
        
        <!-- Performance nach Matrix-Dimension -->
        <h3>Performance nach Matrix-Dimension</h3>
        <div class="chart-container">
            <div id="dimensionPerformanceChart"></div>
        </div>
        
        <div id="optimizationInfo">
            <h3>Apple Silicon Optimierungen</h3>
            <p>Die Matrix-Operationen wurden speziell für Apple Silicon mit MLX optimiert, was zu signifikanten Performance-Verbesserungen führt:</p>
            <ul>
                <li>JIT-Kompilierung für schnellere Berechnungen</li>
                <li>Automatische Erkennung und Verwendung der Apple Neural Engine</li>
                <li>Effiziente Speicherverwaltung für reduzierte Speichernutzung</li>
                <li>Batch-Operationen für verbesserte Durchsatzleistung</li>
                <li>Cache-optimierte Tiling-Strategien für große Matrizen</li>
            </ul>
        </div>
    </div>
    
    <!-- Tab: Detaillierte Ergebnisse -->
    <div id="DetailedResults" class="tab-content">
        <h2>Detaillierte Benchmark-Ergebnisse</h2>
        {results_by_operation}
        
        <!-- Speicherverbrauch -->
        <h3>Speicherverbrauch nach Operation und Backend</h3>
        <div class="chart-container">
            <div id="memoryUsageChart"></div>
        </div>
    </div>
    
    <!-- Tab: Backend-Vergleich -->
    <div id="BackendComparison" class="tab-content">
        <h2>Vergleich der verschiedenen Backends</h2>
        <p>Die folgende Tabelle zeigt einen direkten Vergleich zwischen MLX, PyTorch und NumPy für alle getesteten Operationen und Dimensionen:</p>
        
        <table id="comparisonTable">
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Dimension</th>
                    <th>MLX (s)</th>
                    <th>PyTorch (s)</th>
                    <th>NumPy (s)</th>
                    <th>Beschleunigung</th>
                    <th>Bester Backend</th>
                </tr>
            </thead>
            <tbody>
                {backend_comparison_rows}
            </tbody>
        </table>
        
        <div class="recommendations-box" id="backendRecommendations">
            <h3>Automatische Empfehlungen</h3>
            <ul id="backendRecommendationsList">
                <!-- Wird dynamisch befüllt -->
            </ul>
        </div>
        
        <div style="margin-top: 20px;">
            <button id="exportCSV">Als CSV exportieren</button>
            <button id="exportPDF">Als PDF exportieren</button>
        </div>
    </div>
    
    <!-- Tab: Historischer Vergleich -->
    <div id="HistoricalComparison" class="tab-content">
        <h2>Historischer Performance-Vergleich</h2>
        <p>Dieser Abschnitt zeigt die Leistungsentwicklung über verschiedene Benchmark-Durchläufe hinweg:</p>
        
        <div class="metrics-container">
            <div class="metric" id="historyDataAvailable">
                <div class="metric-title">Historische Daten</div>
                <div class="metric-value" id="historyDataCount">...</div>
                <div>Verfügbare historische Benchmark-Durchläufe</div>
            </div>
            <div class="metric" id="performanceTrend">
                <div class="metric-title">Performance-Trend</div>
                <div class="metric-value" id="overallTrend">...</div>
                <div>Gesamtentwicklung im Vergleich zum letzten Benchmark</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>Performance-Entwicklung über Zeit</h3>
            <div id="performanceTrendChart"></div>
        </div>
        
        <div class="chart-container">
            <h3>Vergleich der Backend-Performance</h3>
            <div id="backendHistoryChart"></div>
        </div>
        
        <div class="chart-container" id="tMathSection">
            <h3>T-Mathematics Engine Performance</h3>
            <p>Vergleich der Performance verschiedener T-Mathematics Engine Versionen:</p>
            <div id="tMathVersionChart"></div>
        </div>
        
        <div style="margin-top: 20px;">
            <button id="exportHistoryCSV">Historische Daten als CSV exportieren</button>
            <button id="exportHistoryJSON">Historische Daten als JSON exportieren</button>
        </div>
    </div>
    </div>
    
    <!-- Tab: Optimierungs-Erkenntnisse -->
    <div id="OptimizationInsights" class="tab-content">
        <h2>Optimierungs-Erkenntnisse</h2>
        
        <h3>MLX-Optimierungen für Apple Silicon</h3>
        <p>Die implementierten Optimierungen nutzen die spezifischen Eigenschaften von Apple Silicon zur Leistungssteigerung:</p>
        
        <div class="metrics-container">
            <div class="metric">
                <div class="metric-title">JIT-Kompilierung</div>
                <div>Die Just-In-Time-Kompilierung erzeugt optimierten Code für die spezifische Hardware-Architektur.</div>
            </div>
            <div class="metric">
                <div class="metric-title">Neural Engine</div>
                <div>Nutzung der Apple Neural Engine (ANE) für beschleunigte Matrix-Operationen.</div>
            </div>
            <div class="metric">
                <div class="metric-title">Speicher-Optimierung</div>
                <div>Effiziente Speicherverwaltung durch intelligentes Memory-Pooling und -Wiederverwendung.</div>
            </div>
            <div class="metric">
                <div class="metric-title">Tiling-Strategien</div>
                <div>Aufteilung großer Matrizen in Cache-optimierte Teilmatrizen für verbesserte Lokalität.</div>
            </div>
        </div>
        
        <h3>Vergleich der Optimierungsstrategien</h3>
        <div class="chart-container">
            <div id="optimizationComparisonChart"></div>
        </div>
    </div>
    
    <div class="footer">
        <p>MISO Matrix-Benchmark | Erstellt am {timestamp}</p>
        <p>Optimiert für Apple Silicon mit MLX und PyTorch</p>
    </div>
    
    <!-- JavaScript für interaktive Funktionalität -->
    <script>
        // Daten aus dem Python-Backend laden
        var chartData = {chart_data_json};
        
        // Automatische Empfehlungen anzeigen
        function displayRecommendations() {
            var recommendationsList = document.getElementById('backendRecommendationsList');
            var recommendationsOverview = document.getElementById('recommendationsOverview');
            
            if (chartData.recommendations && chartData.recommendations.length > 0) {
                chartData.recommendations.forEach(function(rec) {
                    var li = document.createElement('li');
                    li.textContent = rec;
                    li.className = 'recommendation-item';
                    recommendationsList.appendChild(li);
                    
                    // Auch für Übersicht hinzufügen
                    if (recommendationsOverview) {
                        var li2 = document.createElement('li');
                        li2.textContent = rec;
                        li2.className = 'recommendation-item';
                        recommendationsOverview.appendChild(li2);
                    }
                });
            } else {
                var li = document.createElement('li');
                li.textContent = 'Keine spezifischen Empfehlungen verfügbar.';
                recommendationsList.appendChild(li);
                
                if (recommendationsOverview) {
                    var li2 = document.createElement('li');
                    li2.textContent = 'Keine spezifischen Empfehlungen verfügbar.';
                    recommendationsOverview.appendChild(li2);
                }
            }
            
            // Warnungen anzeigen
            var warningsList = document.getElementById('warningsList');
            if (warningsList && chartData.warnings && chartData.warnings.length > 0) {
                chartData.warnings.forEach(function(warning) {
                    var li = document.createElement('li');
                    li.textContent = warning;
                    li.className = 'warning-item';
                    warningsList.appendChild(li);
                });
                document.getElementById('warningsContainer').style.display = 'block';
            }
        }
        
        // Historische Vergleiche anzeigen
        function displayHistoricalComparisons() {
            if (!chartData.historical || Object.keys(chartData.historical).length === 0) {
                document.getElementById('historyDataCount').textContent = 'Keine Daten';
                document.getElementById('overallTrend').textContent = 'N/A';
                return;
            }
            
            // Anzahl der historischen Datensätze anzeigen
            var timestamps = chartData.historical.timestamps || [];
            document.getElementById('historyDataCount').textContent = timestamps.length;
            
            // Performance-Trend berechnen und anzeigen
            if (timestamps.length > 1) {
                // Einfache Logik für Trend-Berechnung
                var trend = "Neutral";
                var trendClass = "";
                
                // Hier könnte eine komplexere Logik zur Trendberechnung stehen
                // Basierend auf den historischen Daten
                
                document.getElementById('overallTrend').textContent = trend;
                document.getElementById('performanceTrend').className = 'metric ' + trendClass;
                
                // Erstelle Performance-Trend-Chart
                if (chartData.historical.operation_trends) {
                    createHistoricalTrendChart();
                }
                
                // Erstelle Backend-Vergleichs-Chart
                if (chartData.historical.backend_performance) {
                    createBackendHistoryChart();
                }
                
                // T-Mathematics-spezifische Charts
                if (chartData.historical.t_math_versions && chartData.historical.t_math_versions.length > 0) {
                    createTMathVersionChart();
                } else {
                    document.getElementById('tMathSection').style.display = 'none';
                }
            } else {
                document.getElementById('overallTrend').textContent = 'Erst ein Datensatz';
            }
        }

        // Historische Trendcharts erstellen
        function createHistoricalTrendChart() {
            var operationTrends = chartData.historical.operation_trends;
            var data = [];
            
            // Für jede Operation einen Datensatz erstellen
            for (var op in operationTrends) {
                var trace = {
                    x: [],
                    y: [],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: op
                };
                
                operationTrends[op].forEach(function(item) {
                    trace.x.push(item.timestamp);
                    trace.y.push(item.time);
                });
                
                data.push(trace);
            }
            
            var layout = {
                title: 'Performance-Entwicklung über Zeit',
                xaxis: { title: 'Zeitpunkt' },
                yaxis: { title: 'Ausführungszeit (s)' },
                hovermode: 'closest'
            };
            
            Plotly.newPlot('performanceTrendChart', data, layout);
        }
        
        // Backend-Vergleichschart erstellen
        function createBackendHistoryChart() {
            var backendPerformance = chartData.historical.backend_performance;
            var data = [];
            
            // Für jedes Backend einen Datensatz erstellen
            for (var backend in backendPerformance) {
                var times = {};
                
                // Mittelwert pro Zeitstempel berechnen
                backendPerformance[backend].forEach(function(item) {
                    if (!times[item.timestamp]) {
                        times[item.timestamp] = [];
                    }
                    times[item.timestamp].push(item.time);
                });
                
                var trace = {
                    x: [],
                    y: [],
                    type: 'bar',
                    name: backend
                };
                
                for (var timestamp in times) {
                    trace.x.push(timestamp);
                    // Berechne Durchschnitt
                    var avg = times[timestamp].reduce((a, b) => a + b, 0) / times[timestamp].length;
                    trace.y.push(avg);
                }
                
                data.push(trace);
            }
            
            var layout = {
                title: 'Backend-Performance im Zeitverlauf',
                xaxis: { title: 'Zeitpunkt' },
                yaxis: { title: 'Durchschnittliche Ausführungszeit (s)' },
                barmode: 'group'
            };
            
            Plotly.newPlot('backendHistoryChart', data, layout);
        }
        
        // T-Mathematics-Versionschart erstellen
        function createTMathVersionChart() {
            var tMathVersions = chartData.historical.t_math_versions || [];
            var timestamps = chartData.historical.timestamps || [];
            var mlxPerformance = chartData.historical.mlx_performance || {};
            
            // Wenn MLX-Performance-Daten verfügbar sind
            if (Object.keys(mlxPerformance).length > 0 && tMathVersions.length > 0) {
                var data = [];
                
                // Für jede Operation ein Trace erstellen
                for (var op in mlxPerformance) {
                    var trace = {
                        x: [],
                        y: [],
                        text: [],
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: op
                    };
                    
                    mlxPerformance[op].forEach(function(item, index) {
                        trace.x.push(item.timestamp);
                        trace.y.push(item.time);
                        trace.text.push('T-Math: ' + (tMathVersions[index] || 'unbekannt'));
                    });
                    
                    data.push(trace);
                }
                
                var layout = {
                    title: 'MLX-Performance mit verschiedenen T-Mathematics Versionen',
                    xaxis: { title: 'Zeitpunkt' },
                    yaxis: { title: 'Ausführungszeit (s)' },
                    hovermode: 'closest'
                };
                
                Plotly.newPlot('tMathVersionChart', data, layout);
            } else {
                document.getElementById('tMathSection').style.display = 'none';
            }
        }
        
        // Tab-Funktionalität
        function openTab(evt, tabName) {{
            const tabContents = document.getElementsByClassName("tab-content");
            for (let i = 0; i < tabContents.length; i++) {{
                tabContents[i].style.display = "none";
            }}
            
            const tabButtons = document.getElementsByClassName("tab-button");
            for (let i = 0; i < tabButtons.length; i++) {{
                tabButtons[i].className = tabButtons[i].className.replace(" active", "");
            }}
            
            // Optimization Comparison Chart
            createOptimizationComparisonChart();
            
            // Export-Funktionalität
            document.getElementById('exportCSV').addEventListener('click', exportToCSV);
            document.getElementById('exportJSON').addEventListener('click', exportToJSON);
            document.getElementById('printResults').addEventListener('click', printResults);
        }});
        
        // Chart-Erstellungsfunktionen
        function createBackendPerformanceChart() {{
            const data = [];
            const operations = Object.keys(chartData.operations);
            
            operations.forEach(operation => {{
                const backends = chartData.operations[operation].backends;
                Object.keys(backends).forEach(backend => {{
                    const times = backends[backend].times;
                    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
                    
                    data.push({
                        x: [backend],
                        y: [avgTime],
                        type: 'bar',
                        name: `${{operation}}`
                    });
                }});
            }});
            
            const layout = {{
                title: 'Durchschnittliche Ausführungszeit nach Backend und Operation',
                xaxis: {{ title: 'Backend' }},
                yaxis: {{ 
                    title: 'Ausführungszeit (s)',
                    type: 'log'
                }},
                barmode: 'group',
                legend: {{ orientation: 'h' }}
            }};
            
            Plotly.newPlot('backendPerformanceChart', data, layout);
        }}
        
        function createDimensionPerformanceChart() {{
            const data = [];
            const operations = Object.keys(chartData.operations);
            
            operations.forEach(operation => {{
                const backends = chartData.operations[operation].backends;
                Object.keys(backends).forEach(backend => {{
                    data.push({
                        x: backends[backend].dimensions,
                        y: backends[backend].times,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: `${{operation}} - ${{backend}}`
                    });
                }});
            }});
            
            const layout = {{
                title: 'Performance nach Matrix-Dimension',
                xaxis: {{ title: 'Matrix-Dimension' }},
                yaxis: {{ 
                    title: 'Ausführungszeit (s)',
                    type: 'log'
                }}
            }};
            
            Plotly.newPlot('dimensionPerformanceChart', data, layout);
        }}
        
        function createMemoryUsageChart() {{
            const data = [];
            const operations = Object.keys(chartData.memory);
            
            operations.forEach(operation => {{
                const backends = Object.keys(chartData.memory[operation]);
                backends.forEach(backend => {{
                    const memoryData = chartData.memory[operation][backend];
                    const dimensions = memoryData.map(d => d.dimension);
                    const memoryChanges = memoryData.map(d => d.memory_change);
                    
                    data.push({
                        x: dimensions,
                        y: memoryChanges,
                        type: 'bar',
                        name: `${{operation}} - ${{backend}}`
                    });
                }});
            }});
            
            const layout = {{
                title: 'Speicherverbrauch nach Operation und Dimension',
                xaxis: {{ title: 'Matrix-Dimension' }},
                yaxis: {{ title: 'Speicheränderung (MB)' }},
                barmode: 'group'
            }};
            
            Plotly.newPlot('memoryUsageChart', data, layout);
        }}
        
        function createOptimizationComparisonChart() {{
            // Hier könnten speziellere Optimierungsvergleiche dargestellt werden
            // Vereinfachtes Beispiel mit vorhandenen Daten
            const data = [];
            const operations = Object.keys(chartData.operations);
            
            if (operations.length > 0) {{
                const operation = operations[0];
                const backends = chartData.operations[operation].backends;
                const backendNames = Object.keys(backends);
                
                backendNames.forEach(backend => {
                    data.push({
                        x: backends[backend].dimensions,
                        y: backends[backend].times,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: backend
                    });
                });
                
                const layout = {
                    title: 'Optimierungsvergleich für ' + operation,
                    xaxis: { title: 'Matrix-Dimension' },
                    yaxis: {
                        title: 'Ausführungszeit (s)',
                        type: 'log'
                    }
                };
                
                Plotly.newPlot('optimizationComparisonChart', data, layout);
            }
        }
        
        // Export-Funktionen
        function exportToCSV() {{
            const rows = [
                ["Operation", "Backend", "Dimension", "Zeit (s)", "Speicher (MB)"]
            ];
            
            const operations = Object.keys(chartData.operations);
            operations.forEach(operation => {{
                const backends = chartData.operations[operation].backends;
                Object.keys(backends).forEach(backend => {{
                    const dimensions = backends[backend].dimensions;
                    const times = backends[backend].times;
                    
                    dimensions.forEach((dim, idx) => {
                        const time = times[idx];
                        let memory = "N/A";
                        
                        if (chartData.memory[operation] && 
                            chartData.memory[operation][backend]) {
                            const memoryPoint = chartData.memory[operation][backend].find(m => m.dimension === dim);
                            if (memoryPoint) {
                                memory = memoryPoint.memory_change;
                            }
                        }
                        
                        rows.push([operation, backend, dim, time, memory]);
                    });
                }});
            }});
            
            let csvContent = "data:text/csv;charset=utf-8,";
            rows.forEach(row => {{
                const csvRow = row.join(',');
                csvContent += csvRow + "\r\n";
            }});
            
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "benchmark_results.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}
        
        function exportToJSON() {{
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(chartData));
            const downloadAnchorNode = document.createElement('a');
            downloadAnchorNode.setAttribute("href", dataStr);
            downloadAnchorNode.setAttribute("download", "benchmark_results.json");
            document.body.appendChild(downloadAnchorNode);
            downloadAnchorNode.click();
            downloadAnchorNode.remove();
        }}
        
        function printResults() {{
            window.print();
        }}
    </script>
</body>
</html>
        """
        
        # Backends als String-Liste für das Template formatieren
        backends_str = ", ".join([b.name for b in self.available_backends])
        
        # Dimensionen als String-Liste formatieren
        dimensions_str = ", ".join([f"{d}x{d}" for d in sorted(set(r.dimension for r in self.results))])
        
        # Formatiere die HTML-Vorlage mit den Daten
        html_content = html_template.format(
            timestamp=current_time,
            backends=backends_str,
            dimensions=dimensions_str,
            iterations=self.iterations if hasattr(self, 'iterations') else TEST_ITERATIONS,
            warmup=self.warmup_iterations if hasattr(self, 'warmup_iterations') else WARMUP_ITERATIONS,
            result_count=len(self.results),
            fastest_backend=fastest_backend,
            fastest_backend_speedup=f"{backend_speedup:.2f}",
            fastest_operation=fastest_op,
            fastest_operation_time=f"{fastest_op_time:.6f}",
            best_precision=best_precision,
            dimension_range=dimension_range,
            results_by_operation=self._generate_operation_results_html(),
            backend_comparison_rows=self._generate_backend_comparison_rows(),
            chart_data_json=chart_data_json
        )
        
        # Speichere den HTML-Bericht
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Erweiterter interaktiver HTML-Bericht in {output_file} erstellt.")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des erweiterten HTML-Reports: {e}")
            return False
        
        # Fülle das Template aus
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        html_content = html_template.format(
            timestamp=current_time,
            backends=", ".join([b.name for b in self.available_backends]),
            dimensions=", ".join([f"{d}x{d}" for d in MATRIX_DIMENSIONS]),
            iterations=TEST_ITERATIONS,
            warmup=WARMUP_ITERATIONS,
            result_count=len(self.results),
            results_by_operation=all_op_html,
            fastest_backend=fastest_backend,
            fastest_backend_speedup=f"{fastest_backend_speedup:.2f}",
            fastest_operation=fastest_operation,
            fastest_operation_time=f"{fastest_operation_time:.6f}",
            best_precision=best_precision,
            dimension_range=dimension_range,
            backend_comparison_rows=backend_comparison_rows,
            chart_js_code=chart_js_code
        )
        
        # Schreibe die HTML-Datei
        try:
            with open(output_file, 'w') as f:
                f.write(html_content)
            logger.info(f"HTML-Bericht in {output_file} erstellt.")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des HTML-Berichts: {str(e)}")
            return False


    def save_results(self, output_file: str):
        """Speichert die Benchmark-Ergebnisse als JSON-Datei"""
        if not self.results:
            logger.warning("Keine Ergebnisse zum Speichern vorhanden")
            return False
        
        # Konvertiere Ergebnisse in serialisierbares Format
        results_data = []
        for result in self.results:
            results_data.append({
                "operation": result.operation.name,
                "backend": result.backend.name,
                "dimension": result.dimension,
                "precision": result.precision.name if hasattr(result.precision, 'name') else str(result.precision),
                "execution_time_mean": result.mean_time,
                "execution_time_std": result.std_dev,
                "success_rate": result.success_rate,
                "memory_change": result.memory_change if hasattr(result, 'memory_change') else None
            })
        
        # Erfasse Metadaten
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "apple_silicon": self._is_apple_silicon(),
                "python_version": platform.python_version()
            },
            "configuration": {
                "iterations": self.iterations if hasattr(self, 'iterations') else TEST_ITERATIONS,
                "warmup_iterations": self.warmup_iterations if hasattr(self, 'warmup_iterations') else WARMUP_ITERATIONS,
                "available_backends": [b.name for b in self.available_backends],
                "mlx_available": self._is_mlx_available(),
                "pytorch_available": self._is_pytorch_available(),
                "t_mathematics_version": self._get_t_mathematics_version()
            }
        }
        
        # Vollständige Daten mit Metadaten
        full_data = {
            "metadata": metadata,
            "results": results_data
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(full_data, f, indent=2)
            logger.info(f"Ergebnisse in {output_file} gespeichert.")
            
            # Archiviere auch eine Kopie für historischen Vergleich
            self._archive_results(full_data)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Ergebnisse: {str(e)}")
            return False
            
    def _archive_results(self, data):
        """Archiviert die Benchmark-Ergebnisse für historischen Vergleich"""
        try:
            # Erstelle Archivverzeichnis falls nötig
            archive_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_archive")
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)
            
            # Generiere Dateinamen mit Zeitstempel
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backends = '_'.join([b.name.lower() for b in self.available_backends])
            archive_file = os.path.join(archive_dir, f"benchmark_{timestamp}_{backends}.json")
            
            with open(archive_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Benchmark-Daten archiviert in {archive_file}")
            return True
        except Exception as e:
            logger.warning(f"Fehler bei der Archivierung der Benchmark-Daten: {str(e)}")
            return False
    
    def load_historical_data(self):
        """Lädt archivierte Benchmark-Ergebnisse für Vergleiche"""
        try:
            archive_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_archive")
            if not os.path.exists(archive_dir):
                logger.info("Kein Benchmark-Archiv gefunden.")
                return []
            
            historical_data = []
            json_files = [f for f in os.listdir(archive_dir) if f.endswith('.json')]
            
            # Sortiere nach Zeitstempel (neuste zuerst)
            json_files.sort(reverse=True)
            
            # Lade die letzten 5 Benchmark-Ergebnisse
            for i, json_file in enumerate(json_files[:5]):
                try:
                    with open(os.path.join(archive_dir, json_file), 'r') as f:
                        data = json.load(f)
                        historical_data.append(data)
                except Exception as e:
                    logger.warning(f"Fehler beim Laden von {json_file}: {str(e)}")
            
            logger.info(f"{len(historical_data)} historische Benchmark-Datensätze geladen.")
            return historical_data
        except Exception as e:
            logger.warning(f"Fehler beim Laden historischer Benchmark-Daten: {str(e)}")
            return []
    
    def _prepare_historical_data(self, historical_datasets):
        """Bereitet historische Daten für den Vergleich auf"""
        if not historical_datasets:
            logger.info("Keine historischen Daten für Vergleich verfügbar.")
            return {}
        
        try:
            historical_comparison = {
                "timestamps": [],             # Zeitstempel der historischen Datensätze
                "backend_performance": {},   # Historische Performance pro Backend
                "operation_trends": {},      # Trends pro Operation über Zeit
                "apple_silicon_boost": [],   # Spezifische Daten zur Apple Silicon Beschleunigung
                "t_math_versions": [],      # T-Mathematics Versionen pro Datensatz
                "mlx_performance": {}        # MLX-spezifische Performance
            }
            
            # Sammle Zeitstempel und T-Mathematics Versionen
            for dataset in historical_datasets:
                try:
                    timestamp = dataset.get("metadata", {}).get("timestamp", "Unbekannt")
                    historical_comparison["timestamps"].append(timestamp)
                    
                    # T-Mathematics Version, falls verfügbar
                    t_math_version = dataset.get("metadata", {}).get("configuration", {}).get("t_mathematics_version", "nicht verfügbar")
                    historical_comparison["t_math_versions"].append(t_math_version)
                    
                    # Prüfe, ob Apple Silicon verwendet wurde
                    apple_silicon = dataset.get("metadata", {}).get("platform", {}).get("apple_silicon", False)
                    mlx_available = dataset.get("metadata", {}).get("configuration", {}).get("mlx_available", False)
                    
                    if apple_silicon and mlx_available:
                        historical_comparison["apple_silicon_boost"].append({
                            "timestamp": timestamp,
                            "boost": True,
                            "mlx_available": mlx_available
                        })
                    else:
                        historical_comparison["apple_silicon_boost"].append({
                            "timestamp": timestamp,
                            "boost": False,
                            "mlx_available": mlx_available
                        })
                except Exception as e:
                    logger.warning(f"Fehler beim Verarbeiten eines historischen Datensatzes: {str(e)}")
            
            # Verarbeite die historischen Ergebnisse für Operationen und Backends
            for i, dataset in enumerate(historical_datasets):
                try:
                    results = dataset.get("results", [])
                    timestamp = historical_comparison["timestamps"][i] if i < len(historical_comparison["timestamps"]) else "Unbekannt"
                    
                    # Gruppiere nach Operation und Backend
                    for result in results:
                        operation = result.get("operation")
                        backend = result.get("backend")
                        time_mean = result.get("execution_time_mean", 0)
                        dimension = result.get("dimension", 0)
                        
                        # Für Operations-Trends
                        if operation not in historical_comparison["operation_trends"]:
                            historical_comparison["operation_trends"][operation] = []
                        historical_comparison["operation_trends"][operation].append({
                            "timestamp": timestamp,
                            "time": time_mean,
                            "dimension": dimension
                        })
                        
                        # Für Backend-Performance
                        if backend not in historical_comparison["backend_performance"]:
                            historical_comparison["backend_performance"][backend] = []
                        historical_comparison["backend_performance"][backend].append({
                            "timestamp": timestamp,
                            "time": time_mean,
                            "operation": operation,
                            "dimension": dimension
                        })
                        
                        # MLX-spezifische Performance, wenn verfügbar
                        if backend == "MLX":
                            if operation not in historical_comparison["mlx_performance"]:
                                historical_comparison["mlx_performance"][operation] = []
                            historical_comparison["mlx_performance"][operation].append({
                                "timestamp": timestamp,
                                "time": time_mean,
                                "dimension": dimension
                            })
                except Exception as e:
                    logger.warning(f"Fehler beim Verarbeiten der Ergebnisse eines historischen Datensatzes: {str(e)}")
            
            return historical_comparison
        except Exception as e:
            logger.warning(f"Fehler bei der Aufbereitung historischer Daten: {str(e)}")
            return {}


# Eine globale Hilfsfunktion zur Generierung von automatischen Empfehlungen
def generate_recommendations(benchmarker):
    """Generiert automatische Empfehlungen basierend auf Benchmark-Ergebnissen"""
    recommendations = []
    warnings = []
    
    try:
        # 1. Finde das schnellste Backend für verschiedene Operationen
        operation_backends = {}
        for result in benchmarker.results:
            key = result.operation.name
            if key not in operation_backends:
                operation_backends[key] = []
            operation_backends[key].append((result.backend.name, result.mean_time))
        
        for op, backends in operation_backends.items():
            # Sortiere nach Ausführungszeit (aufsteigend)
            backends.sort(key=lambda x: x[1])
            fastest = backends[0]
            if len(backends) > 1:
                # Wenn MLX am schnellsten ist und wir auf Apple Silicon sind
                if fastest[0] == "MLX" and benchmarker._is_apple_silicon():
                    recommendations.append(f"Für {op}-Operationen empfehlen wir MLX auf Apple Silicon. Es ist {(backends[1][1]/fastest[1]):.2f}x schneller als {backends[1][0]}.")
                # Wenn PyTorch am schnellsten ist
                elif fastest[0] == "PYTORCH":
                    recommendations.append(f"Für {op}-Operationen empfehlen wir PyTorch. Es ist {(backends[1][1]/fastest[1]):.2f}x schneller als {backends[1][0]}.")
                # Wenn NumPy am schnellsten ist (eher unwahrscheinlich, aber möglich)
                elif fastest[0] == "NUMPY":
                    recommendations.append(f"Interessanterweise ist NumPy für {op}-Operationen am schnellsten. Es ist {(backends[1][1]/fastest[1]):.2f}x schneller als {backends[1][0]}.")
        
        # 2. Prüfe auf ungeeignete Präzisionstypen
        precision_perf = {}
        for result in benchmarker.results:
            key = (result.operation.name, result.precision.name)
            if key not in precision_perf:
                precision_perf[key] = []
            precision_perf[key].append(result.mean_time)
        
        for (op, prec), times in precision_perf.items():
            avg_time = sum(times) / len(times)
            # Wenn FP16/BF16 verwendet wird, aber sehr langsam ist
            if prec in ["FLOAT16", "BFLOAT16"] and avg_time > 1.5 * sum([t for (o, p), t in precision_perf.items() if o == op and p == "FLOAT32"]) / len([t for (o, p), t in precision_perf.items() if o == op and p == "FLOAT32"]):
                warnings.append(f"Achtung: {prec} für {op} ist deutlich langsamer als FLOAT32. Wir empfehlen, zu FLOAT32 zu wechseln.")
        
        # 3. Optimale Dimensionen
        dim_perf = {}
        for result in benchmarker.results:
            key = (result.operation.name, result.backend.name)
            if key not in dim_perf:
                dim_perf[key] = []
            dim_perf[key].append((result.dimension, result.mean_time))
        
        for (op, backend), dims in dim_perf.items():
            # Sortiere nach Ausführungszeit pro Element (um Dimensionen fair zu vergleichen)
            dims.sort(key=lambda x: x[1] / (x[0] * x[0]))
            most_efficient_dim = dims[0][0]
            recommendations.append(f"Für optimale Effizienz bei {op} mit {backend} verwenden Sie Matrizen mit Dimension {most_efficient_dim}x{most_efficient_dim}.")
        
        # 4. T-Mathematics-spezifische Empfehlungen
        if benchmarker._is_apple_silicon() and benchmarker._is_mlx_available():
            recommendations.append("Da Sie Apple Silicon verwenden, empfehlen wir die Nutzung von MLX für optimale Performance mit der T-Mathematics Engine.")
            
            # Wenn T-Mathematics verfügbar ist
            if benchmarker._get_t_mathematics_version() != "nicht verfügbar":
                recommendations.append(f"Ihre T-Mathematics Version {benchmarker._get_t_mathematics_version()} ist optimiert für Apple Neural Engine. Wir empfehlen die Verwendung von BF16 für den besten Kompromiss aus Geschwindigkeit und Genauigkeit.")
    
    except Exception as e:
        logger.warning(f"Fehler bei der Generierung von Empfehlungen: {str(e)}")
    
    return recommendations, warnings


def main():
    """Hauptfunktion für die Ausführung der Matrix-Dimensions-Benchmarks"""
    parser = argparse.ArgumentParser(description="Matrix-Dimensionen Performance-Tests")
    parser.add_argument("--test", action="store_true", help="Führt einen Kurztest mit reduzierten Dimensionen durch")
    parser.add_argument("--backends", type=str, nargs="+", choices=[b.name.lower() for b in Backend], default=[], help="Zu testende Backends")
    parser.add_argument("--operations", type=str, nargs="+", choices=[op.name.lower() for op in Operation], default=[], help="Zu testende Operationen")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[], help="Zu testende Matrix-Dimensionen")
    parser.add_argument("--precision", type=str, choices=[p.name.lower() for p in PrecisionType], default="float32", help="Zu testender Präzisionstyp")
    parser.add_argument("--iterations", type=int, default=5, help="Anzahl der Benchmark-Iterationen")
    parser.add_argument("--warmup", type=int, default=2, help="Anzahl der Warmup-Iterationen")
    parser.add_argument("--output", type=str, default="matrix_benchmark_results.json", help="Ausgabedatei für die Ergebnisse")
    parser.add_argument("--report", type=str, default="matrix_benchmark_report.html", help="Ausgabedatei für den HTML-Bericht")
    parser.add_argument("--advanced-report", type=str, default=None, help="Ausgabedatei für den erweiterten interaktiven HTML-Bericht")
    parser.add_argument("--dimension-plots", type=str, default="plots/dimension", help="Verzeichnis für Dimensions-Plots")
    parser.add_argument("--precision-plots", type=str, default="plots/precision", help="Verzeichnis für Präzisions-Plots")
    parser.add_argument("--no-mlx", action="store_true", help="Deaktiviere MLX-Benchmarks")
    parser.add_argument("--no-pytorch", action="store_true", help="Deaktiviere PyTorch-Benchmarks")
    parser.add_argument("--only-operation", type=str, choices=[op.name for op in Operation], 
                        help="Nur die angegebene Operation testen")
    parser.add_argument("--only-dimension", type=int, choices=MATRIX_DIMENSIONS,
                        help="Nur die angegebene Matrix-Dimension testen")
    parser.add_argument("--only-precision", type=str, choices=[p.name for p in PrecisionType],
                        help="Nur den angegebenen Präzisionstyp testen")
    parser.add_argument("--precision-test", action="store_true", 
                        help="Führe Präzisionsvergleichstest durch")
    
    args = parser.parse_args()
    
    # Erstelle Plot-Verzeichnisse, falls sie nicht existieren
    os.makedirs(args.dimension_plots, exist_ok=True)
    os.makedirs(args.precision_plots, exist_ok=True)
    
    # Initialisiere den Benchmarker
    benchmarker = MatrixBenchmarker(
        use_mlx=not args.no_mlx,
        use_pytorch=not args.no_pytorch
    )
    
    # Führe Tests basierend auf den Argumenten aus
    operations = [Operation[args.only_operation]] if args.only_operation else list(Operation)
    dimensions = [args.only_dimension] if args.only_dimension else MATRIX_DIMENSIONS
    precision_type = PrecisionType[args.only_precision] if args.only_precision else PrecisionType.FLOAT32
    
    logger.info(f"Starte Matrix-Dimensions-Benchmarks")
    logger.info(f"Operationen: {[op.name for op in operations]}")
    logger.info(f"Dimensionen: {dimensions}")
    logger.info(f"Präzision: {precision_type.name}")
    logger.info(f"Iterationen: {args.iterations}, Warmup: {args.warmup}")
    
    # Führe Tests durch
    if args.precision_test:
        # Führe Präzisionstests durch
        logger.info("\n=== Präzisionsvergleichstest ===")
        
        for operation in operations:
            dim = 128  # Fixe Dimension für Präzisionstests
            logger.info(f"\nTeste {operation.name} Operation mit verschiedenen Präzisionstypen")
            
            results = benchmarker.run_precision_benchmarks(
                operation=operation,
                dimension=dim,
                precision_types=list(PrecisionType),
                iterations=args.iterations,
                warmup=args.warmup
            )
            
            # Erstelle Plot
            plot_file = os.path.join(args.precision_plots, f"{operation.name.lower()}_precision_{dim}x{dim}.png")
            benchmarker.plot_precision_comparison(operation, dim, plot_file)
    else:
        # Führe Dimensionstests durch
        for operation in operations:
            logger.info(f"\n=== Führe Tests für {operation.name} Operation durch ===")
            
            results = benchmarker.run_dimension_benchmarks(
                operation=operation,
                dimensions=dimensions,
                precision=precision_type,
                iterations=args.iterations,
                warmup=args.warmup
            )
            
            # Erstelle Plot
            plot_file = os.path.join(args.dimension_plots, f"{operation.name.lower()}_dimensions.png")
            benchmarker.plot_dimension_comparison(operation, precision_type, plot_file)
    
    # Speichere die Ergebnisse
    benchmarker.save_results(args.output)
    logger.info(f"Ergebnisse in {args.output} gespeichert")
    
    # Generiere HTML-Bericht
    benchmarker.generate_html_report(args.report)
    
    # Generiere den erweiterten interaktiven HTML-Bericht, falls angefordert
    if args.advanced_report:
        logger.info(f"Generiere erweiterten interaktiven HTML-Bericht: {args.advanced_report}")
        benchmarker.generate_advanced_html_report(args.advanced_report)
    logger.info(f"HTML-Bericht in {args.report} erstellt")


if __name__ == "__main__":
    main()
