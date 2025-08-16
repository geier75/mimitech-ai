#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Backend-Optimierung

Diese Komponente ist verantwortlich für die automatische Auswahl des optimalen Backends
für Tensor-Operationen basierend auf der verfügbaren Hardware und Laufzeitumgebung.
"""

import os
import platform
import logging
import subprocess
import time
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Any

# Konfiguriere Logger
logger = logging.getLogger("MISO.BackendOptimizer")

class Backend(Enum):
    """Unterstützte Tensor-Backends."""
    MLX = "mlx"         # Apple MLX für Apple Silicon
    PYTORCH = "torch"   # PyTorch mit CUDA/MPS
    TENSORFLOW = "tf"   # TensorFlow mit CUDA
    JAX = "jax"         # JAX mit XLA
    NUMPY = "numpy"     # NumPy als Fallback


class BackendOptimizer:
    """
    Wählt das optimale Backend für Tensor-Operationen basierend auf:
    1. Verfügbare Hardware (CPU, GPU, TPU, Apple Silicon)
    2. Installierte Bibliotheken
    3. Operationstyp
    4. Datengröße
    """
    
    def __init__(self, preferred_backend: Optional[str] = None, enable_benchmarking: bool = False):
        """
        Initialisiert den BackendOptimizer.
        
        Args:
            preferred_backend: Bevorzugtes Backend, falls angegeben
            enable_benchmarking: Ob Benchmarking für die Auswahl verwendet werden soll
        """
        self.preferred_backend = preferred_backend
        self.enable_benchmarking = enable_benchmarking
        
        # Cache für verfügbare Backends und Leistungsmetriken
        self._available_backends = None
        self._performance_metrics = {}
        
        # Hardware-Informationen
        self.hardware_info = self._detect_hardware()
        
        logger.info(f"BackendOptimizer initialisiert mit Hardware: {self.hardware_info}")
        if preferred_backend:
            logger.info(f"Bevorzugtes Backend: {preferred_backend}")
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """
        Erkennt verfügbare Hardware-Ressourcen.
        
        Returns:
            Dictionary mit Hardware-Informationen
        """
        hardware_info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "has_apple_silicon": False,
            "has_nvidia_gpu": False,
            "has_amd_gpu": False,
            "has_intel_gpu": False,
            "cpu_cores": os.cpu_count() or 1,
            "gpu_info": []
        }
        
        # Erkennung von Apple Silicon
        if platform.system() == "Darwin" and platform.machine().startswith(("arm", "aarch")):
            hardware_info["has_apple_silicon"] = True
            
            # Versuche, detaillierte Apple Silicon Informationen zu erhalten
            try:
                import subprocess
                result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    hardware_info["processor_detail"] = result.stdout.strip()
            except:
                pass
        
        # Erkennung von NVIDIA GPUs (Linux/Windows)
        if platform.system() in ["Linux", "Windows"]:
            # Prüfe auf NVIDIA GPU unter Linux
            if platform.system() == "Linux" and os.path.exists("/proc/driver/nvidia/version"):
                hardware_info["has_nvidia_gpu"] = True
                
                # Versuche, nvidia-smi für detaillierte Informationen zu verwenden
                try:
                    nvidia_info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_capability", 
                                                "--format=csv,noheader"], 
                                               capture_output=True, text=True)
                    if nvidia_info.returncode == 0:
                        for line in nvidia_info.stdout.strip().split("\n"):
                            parts = [part.strip() for part in line.split(",")]
                            if len(parts) >= 2:
                                hardware_info["gpu_info"].append({
                                    "name": parts[0],
                                    "memory": parts[1] if len(parts) > 1 else "Unknown",
                                    "compute_capability": parts[2] if len(parts) > 2 else "Unknown"
                                })
                except:
                    pass
            
            # Prüfe auf AMD GPU unter Linux
            if platform.system() == "Linux" and os.path.exists("/sys/class/drm/card0/device/vendor"):
                try:
                    with open("/sys/class/drm/card0/device/vendor", "r") as f:
                        vendor_id = f.read().strip()
                        if vendor_id == "0x1002":  # AMD Vendor ID
                            hardware_info["has_amd_gpu"] = True
                except:
                    pass
        
        return hardware_info
    
    def get_available_backends(self) -> List[Backend]:
        """
        Ermittelt alle verfügbaren Backends auf dem System.
        
        Returns:
            Liste von verfügbaren Backends
        """
        if self._available_backends is not None:
            return self._available_backends
        
        available = []
        
        # Prüfe MLX (nur für Apple Silicon)
        if self.hardware_info["has_apple_silicon"]:
            try:
                import mlx.core
                available.append(Backend.MLX)
                logger.info("MLX Backend verfügbar für Apple Silicon")
            except ImportError:
                logger.warning("MLX nicht verfügbar trotz Apple Silicon")
        
        # Prüfe PyTorch
        try:
            import torch
            available.append(Backend.PYTORCH)
            
            # Prüfe auf CUDA/MPS Verfügbarkeit in PyTorch
            has_cuda = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
            has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
            
            logger.info(f"PyTorch Backend verfügbar (CUDA: {has_cuda}, MPS: {has_mps})")
        except ImportError:
            logger.warning("PyTorch nicht verfügbar")
        
        # Prüfe TensorFlow
        try:
            import tensorflow as tf
            available.append(Backend.TENSORFLOW)
            
            # Prüfe auf GPU-Unterstützung in TensorFlow
            gpus = tf.config.list_physical_devices('GPU') if hasattr(tf.config, 'list_physical_devices') else []
            
            logger.info(f"TensorFlow Backend verfügbar (GPUs: {len(gpus)})")
        except ImportError:
            logger.warning("TensorFlow nicht verfügbar")
        
        # Prüfe JAX
        try:
            import jax
            available.append(Backend.JAX)
            
            # Prüfe auf GPU/TPU-Unterstützung in JAX
            devices = jax.devices() if hasattr(jax, 'devices') else []
            
            logger.info(f"JAX Backend verfügbar (Geräte: {len(devices)})")
        except ImportError:
            logger.warning("JAX nicht verfügbar")
        
        # NumPy sollte immer verfügbar sein
        try:
            import numpy
            available.append(Backend.NUMPY)
            logger.info("NumPy Backend verfügbar")
        except ImportError:
            logger.warning("NumPy nicht verfügbar - dies ist ungewöhnlich!")
        
        self._available_backends = available
        return available
    
    def get_optimal_backend(self, operation_type: Optional[str] = None, 
                          data_size: Optional[Tuple[int, ...]] = None) -> Backend:
        """
        Bestimmt das optimale Backend basierend auf Hardware und Operation.
        
        Args:
            operation_type: Art der Operation (z.B. "matrix_multiply", "convolution")
            data_size: Größe der Eingabedaten
            
        Returns:
            Optimales Backend als Backend-Enum
        """
        # Verwende das bevorzugte Backend, falls angegeben und verfügbar
        if self.preferred_backend:
            try:
                preferred = Backend(self.preferred_backend)
                available_backends = self.get_available_backends()
                if preferred in available_backends:
                    logger.info(f"Verwende bevorzugtes Backend: {preferred.value}")
                    return preferred
                else:
                    logger.warning(f"Bevorzugtes Backend {preferred.value} nicht verfügbar")
            except ValueError:
                logger.warning(f"Ungültiges bevorzugtes Backend: {self.preferred_backend}")
        
        # Überprüfe verfügbare Backends
        available_backends = self.get_available_backends()
        if not available_backends:
            logger.error("Keine Backends verfügbar!")
            raise RuntimeError("Keine Tensor-Backends verfügbar auf diesem System")
        
        # Optimierungslogik basierend auf der Hardware
        
        # 1. Apple Silicon: MLX hat Priorität
        if self.hardware_info["has_apple_silicon"]:
            if Backend.MLX in available_backends:
                logger.info("Verwende MLX für Apple Silicon")
                return Backend.MLX
            elif Backend.PYTORCH in available_backends:
                logger.info("Verwende PyTorch mit MPS für Apple Silicon")
                return Backend.PYTORCH
        
        # 2. NVIDIA GPU: PyTorch oder TensorFlow mit CUDA
        if self.hardware_info["has_nvidia_gpu"]:
            if Backend.PYTORCH in available_backends:
                logger.info("Verwende PyTorch mit CUDA für NVIDIA GPU")
                return Backend.PYTORCH
            elif Backend.TENSORFLOW in available_backends:
                logger.info("Verwende TensorFlow mit CUDA für NVIDIA GPU")
                return Backend.TENSORFLOW
            elif Backend.JAX in available_backends:
                logger.info("Verwende JAX mit CUDA für NVIDIA GPU")
                return Backend.JAX
        
        # 3. Fallbacks für andere Szenarien
        if Backend.PYTORCH in available_backends:
            logger.info("Verwende PyTorch als Fallback")
            return Backend.PYTORCH
        elif Backend.TENSORFLOW in available_backends:
            logger.info("Verwende TensorFlow als Fallback")
            return Backend.TENSORFLOW
        elif Backend.JAX in available_backends:
            logger.info("Verwende JAX als Fallback")
            return Backend.JAX
        
        # 4. NumPy als letzter Ausweg
        logger.info("Verwende NumPy als letzten Ausweg")
        return Backend.NUMPY
    
    def benchmark_operation(self, operation_func, backends: List[Backend], 
                           sample_data, num_runs: int = 5) -> Dict[Backend, float]:
        """
        Führt Benchmarks für eine Operation auf verschiedenen Backends durch.
        
        Args:
            operation_func: Funktion, die die Operation auf einem Backend ausführt
            backends: Liste von zu testenden Backends
            sample_data: Beispieldaten für den Benchmark
            num_runs: Anzahl der Testläufe
            
        Returns:
            Dictionary mit Durchschnittszeiten pro Backend
        """
        if not self.enable_benchmarking:
            logger.warning("Benchmarking ist deaktiviert")
            return {}
        
        results = {}
        
        for backend in backends:
            try:
                logger.info(f"Benchmark für {backend.value} Backend beginnt...")
                
                # Führe Warmup-Durchlauf durch
                operation_func(backend, sample_data)
                
                # Führe Benchmark-Durchläufe durch
                times = []
                for i in range(num_runs):
                    start_time = time.time()
                    operation_func(backend, sample_data)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                # Berechne Durchschnittszeit
                avg_time = sum(times) / len(times)
                results[backend] = avg_time
                
                logger.info(f"Benchmark für {backend.value}: {avg_time:.6f}s im Durchschnitt")
            
            except Exception as e:
                logger.error(f"Fehler beim Benchmark für {backend.value}: {e}")
        
        # Cache-Ergebnisse
        operation_key = str(operation_func.__name__)
        self._performance_metrics[operation_key] = results
        
        return results
    
    def get_backend_for_operation(self, operation_name: str, 
                                data_size: Optional[Tuple[int, ...]] = None) -> Backend:
        """
        Wählt das beste Backend für eine bestimmte Operation basierend auf
        Benchmarks oder Heuristiken aus.
        
        Args:
            operation_name: Name der Operation
            data_size: Größe der Eingabedaten
            
        Returns:
            Bestes Backend für die Operation
        """
        # Prüfe, ob wir Benchmark-Daten für diese Operation haben
        if operation_name in self._performance_metrics:
            # Wähle das schnellste Backend basierend auf Benchmarks
            benchmarks = self._performance_metrics[operation_name]
            if benchmarks:
                fastest_backend = min(benchmarks.items(), key=lambda x: x[1])[0]
                logger.info(f"Verwende {fastest_backend.value} als schnellstes Backend für {operation_name}")
                return fastest_backend
        
        # Andernfalls Fallback auf Hardware-basierte Heuristik
        return self.get_optimal_backend(operation_name, data_size)
    
    def get_backend_instance(self, backend: Backend):
        """
        Gibt eine Instanz des angegebenen Backends zurück.
        
        Args:
            backend: Zu verwendendes Backend
            
        Returns:
            Backend-Instanz oder Modul
        """
        if backend == Backend.MLX:
            try:
                import mlx.core as mx
                return mx
            except ImportError:
                logger.error("MLX Backend wurde angefordert, ist aber nicht verfügbar")
                raise
        
        elif backend == Backend.PYTORCH:
            try:
                import torch
                # Aktiviere CUDA oder MPS, falls verfügbar
                if torch.cuda.is_available():
                    torch.set_default_device('cuda')
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.set_default_device('mps')
                return torch
            except ImportError:
                logger.error("PyTorch Backend wurde angefordert, ist aber nicht verfügbar")
                raise
        
        elif backend == Backend.TENSORFLOW:
            try:
                import tensorflow as tf
                # Aktiviere GPU-Wachstum, falls verfügbar
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                return tf
            except ImportError:
                logger.error("TensorFlow Backend wurde angefordert, ist aber nicht verfügbar")
                raise
        
        elif backend == Backend.JAX:
            try:
                import jax
                import jax.numpy as jnp
                return jnp
            except ImportError:
                logger.error("JAX Backend wurde angefordert, ist aber nicht verfügbar")
                raise
        
        elif backend == Backend.NUMPY:
            try:
                import numpy as np
                return np
            except ImportError:
                logger.error("NumPy Backend wurde angefordert, ist aber nicht verfügbar")
                raise
        
        else:
            raise ValueError(f"Unbekanntes Backend: {backend}")


# Testfunktion
def test_backend_optimizer():
    """Testet die Funktionalität des BackendOptimizers."""
    # Konfiguriere detailliertes Logging für den Test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Testing BackendOptimizer ===")
    
    # Instanz ohne Präferenz
    optimizer = BackendOptimizer(enable_benchmarking=True)
    
    # Hardwareinfo anzeigen
    print("\nHardware Information:")
    for key, value in optimizer.hardware_info.items():
        print(f"  {key}: {value}")
    
    # Verfügbare Backends anzeigen
    available = optimizer.get_available_backends()
    print("\nVerfügbare Backends:")
    for backend in available:
        print(f"  - {backend.value}")
    
    # Optimales Backend ermitteln
    optimal = optimizer.get_optimal_backend()
    print(f"\nOptimales Backend für dieses System: {optimal.value}")
    
    # Test mit bevorzugtem Backend
    if Backend.PYTORCH in available:
        print("\nTest mit PyTorch als bevorzugtem Backend:")
        torch_optimizer = BackendOptimizer(preferred_backend="torch")
        torch_backend = torch_optimizer.get_optimal_backend()
        print(f"  Ermitteltes Backend: {torch_backend.value}")
    
    # Benchmark-Test (falls aktiviert)
    if optimizer.enable_benchmarking and len(available) > 1:
        print("\nBenchmark-Test für Matrix-Multiplikation:")
        
        # Definiere eine einfache Matrixmultiplikation für verschiedene Backends
        def matrix_multiply(backend, data):
            a, b = data
            
            if backend == Backend.MLX:
                import mlx.core as mx
                return mx.matmul(a, b)
            
            elif backend == Backend.PYTORCH:
                import torch
                return torch.matmul(a, b)
            
            elif backend == Backend.TENSORFLOW:
                import tensorflow as tf
                return tf.matmul(a, b)
            
            elif backend == Backend.JAX:
                import jax.numpy as jnp
                return jnp.matmul(a, b)
            
            elif backend == Backend.NUMPY:
                import numpy as np
                return np.matmul(a, b)
        
        # Erstelle Testdaten für jedes Backend
        sample_data = {}
        
        for backend in available:
            try:
                if backend == Backend.MLX:
                    import mlx.core as mx
                    a = mx.random.normal((100, 100))
                    b = mx.random.normal((100, 100))
                
                elif backend == Backend.PYTORCH:
                    import torch
                    a = torch.randn(100, 100)
                    b = torch.randn(100, 100)
                
                elif backend == Backend.TENSORFLOW:
                    import tensorflow as tf
                    a = tf.random.normal((100, 100))
                    b = tf.random.normal((100, 100))
                
                elif backend == Backend.JAX:
                    import jax
                    import jax.numpy as jnp
                    key = jax.random.PRNGKey(0)
                    a = jax.random.normal(key, (100, 100))
                    b = jax.random.normal(key, (100, 100))
                
                elif backend == Backend.NUMPY:
                    import numpy as np
                    a = np.random.randn(100, 100)
                    b = np.random.randn(100, 100)
                
                sample_data[backend] = (a, b)
                print(f"  Testdaten für {backend.value} erstellt")
            
            except Exception as e:
                print(f"  Fehler beim Erstellen von Testdaten für {backend.value}: {e}")
        
        # Führe Benchmarks aus
        for backend in [b for b in available if b in sample_data]:
            try:
                data = sample_data[backend]
                
                # Warmup
                matrix_multiply(backend, data)
                
                # Benchmark
                start_time = time.time()
                for _ in range(10):
                    matrix_multiply(backend, data)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                print(f"  {backend.value}: {avg_time:.6f}s im Durchschnitt für 10 Durchläufe")
            
            except Exception as e:
                print(f"  Fehler beim Benchmark für {backend.value}: {e}")


if __name__ == "__main__":
    # Führe Test aus
    test_backend_optimizer()
