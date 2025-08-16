#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark für die Speicheroptimierung der T-Mathematics Engine

Dieser Benchmark vergleicht die Leistung der optimierten Speicherverwaltung
mit der ursprünglichen Implementierung der T-Mathematics Engine.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("speicheroptimierung_benchmark")

# MLX-Verfügbarkeit prüfen
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    logger.warning("MLX konnte nicht importiert werden. MLX-Benchmarks werden übersprungen.")

# Importiere die ursprüngliche MLX-Unterstützung
from miso.math.t_mathematics.mlx_support import MLXBackend

# Importiere die optimierten Komponenten
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from tensor_pool import TensorPool
from direct_memory_transfer import DirectMemoryTransfer, mps_to_mlx, mlx_to_mps
from optimized_mlx_operations import OptimizedMLXOperations, optimized_matmul, optimized_svd

# Apple Silicon-Erkennung
IS_APPLE_SILICON = False
try:
    import platform
    IS_APPLE_SILICON = platform.processor() == 'arm' and platform.system() == 'Darwin'
    if IS_APPLE_SILICON:
        logger.info("Apple Silicon erkannt")
except Exception as e:
    logger.warning(f"Fehler bei der Apple Silicon-Erkennung: {e}")


class MemoryBenchmark:
    """Benchmark für Speicheroperationen in der T-Mathematics Engine"""
    
    def __init__(self):
        """Initialisiert den Benchmark"""
        self.results = defaultdict(list)
        self.configs = []
        
        # Initialisiere ursprüngliches MLX-Backend
        self.original_mlx = MLXBackend(precision="float16")
        
        # Initialisiere optimierte Komponenten
        self.tensor_pool = TensorPool()
        self.memory_transfer = DirectMemoryTransfer()
        self.optimized_ops = OptimizedMLXOperations()
        
        # PyTorch-Geräte
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        logger.info(f"PyTorch-Gerät: {self.device}")
    
    def benchmark_tensor_transfer(self, shape=(1024, 1024), num_transfers=100):
        """
        Benchmarkt die Tensorübertragung zwischen verschiedenen Geräten
        
        Args:
            shape: Form des Testtensors
            num_transfers: Anzahl der Transferoperationen
        """
        logger.info(f"Benchmarking Tensorübertragung mit Shape {shape}, {num_transfers} Transfers")
        
        # Erstelle PyTorch-Tensor auf MPS
        torch_tensor = torch.randn(shape, device=self.device)
        
        # Benchmarkt ursprüngliche Implementierung
        start_time = time.time()
        for _ in range(num_transfers):
            mlx_array = self.original_mlx.to_mlx(torch_tensor)
            torch_result = self.original_mlx.to_torch(mlx_array)
        original_time = time.time() - start_time
        logger.info(f"Ursprüngliche Implementierung: {original_time:.6f}s")
        
        # Benchmarkt optimierte Implementierung
        start_time = time.time()
        for _ in range(num_transfers):
            mlx_array = mps_to_mlx(torch_tensor)
            torch_result = mlx_to_mps(mlx_array)
        optimized_time = time.time() - start_time
        logger.info(f"Optimierte Implementierung: {optimized_time:.6f}s")
        
        # Berechne Speedup
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Speichere Ergebnisse
        self.results["tensor_transfer"].append({
            "shape": shape,
            "num_transfers": num_transfers,
            "original_time": original_time,
            "optimized_time": optimized_time,
            "speedup": speedup
        })
    
    def benchmark_matmul(self, shape=(1024, 1024), num_ops=10):
        """
        Benchmarkt die Matrixmultiplikation
        
        Args:
            shape: Form der Testmatrizen
            num_ops: Anzahl der Operationen
        """
        logger.info(f"Benchmarking Matrixmultiplikation mit Shape {shape}, {num_ops} Operationen")
        
        # Erstelle PyTorch-Tensoren
        a = torch.randn(shape, device=self.device)
        b = torch.randn(shape, device=self.device)
        
        # Benchmarkt ursprüngliche Implementierung
        start_time = time.time()
        for _ in range(num_ops):
            result = self.original_mlx.matmul(a, b)
        original_time = time.time() - start_time
        logger.info(f"Ursprüngliche Implementierung: {original_time:.6f}s")
        
        # Benchmarkt optimierte Implementierung
        start_time = time.time()
        for _ in range(num_ops):
            result = optimized_matmul(a, b)
        optimized_time = time.time() - start_time
        logger.info(f"Optimierte Implementierung: {optimized_time:.6f}s")
        
        # Berechne Speedup
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Speichere Ergebnisse
        self.results["matmul"].append({
            "shape": shape,
            "num_ops": num_ops,
            "original_time": original_time,
            "optimized_time": optimized_time,
            "speedup": speedup
        })
    
    def benchmark_svd(self, shape=(512, 512), num_ops=5):
        """
        Benchmarkt die Singulärwertzerlegung (SVD)
        
        Args:
            shape: Form der Testmatrix
            num_ops: Anzahl der Operationen
        """
        logger.info(f"Benchmarking SVD mit Shape {shape}, {num_ops} Operationen")
        
        # Erstelle PyTorch-Tensor
        a = torch.randn(shape, device=self.device)
        
        # Benchmarkt ursprüngliche Implementierung
        start_time = time.time()
        try:
            for _ in range(num_ops):
                u, s, v = self.original_mlx.svd(a)
            original_time = time.time() - start_time
            original_error = None
            logger.info(f"Ursprüngliche Implementierung: {original_time:.6f}s")
        except Exception as e:
            original_time = float('inf')
            original_error = str(e)
            logger.error(f"Ursprüngliche SVD fehlgeschlagen: {e}")
        
        # Benchmarkt optimierte Implementierung
        start_time = time.time()
        try:
            for _ in range(num_ops):
                u, s, v = optimized_svd(a)
            optimized_time = time.time() - start_time
            optimized_error = None
            logger.info(f"Optimierte Implementierung: {optimized_time:.6f}s")
        except Exception as e:
            optimized_time = float('inf')
            optimized_error = str(e)
            logger.error(f"Optimierte SVD fehlgeschlagen: {e}")
        
        # Berechne Speedup (nur wenn beide erfolgreich waren)
        if original_error is None and optimized_error is None:
            speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
            logger.info(f"Speedup: {speedup:.2f}x")
        else:
            speedup = float('nan')
            if original_error is not None and optimized_error is None:
                logger.info("Optimierte SVD erfolgreich, ursprüngliche fehlgeschlagen")
            elif original_error is None and optimized_error is not None:
                logger.info("Ursprüngliche SVD erfolgreich, optimierte fehlgeschlagen")
            else:
                logger.info("Beide SVD-Implementierungen fehlgeschlagen")
        
        # Speichere Ergebnisse
        self.results["svd"].append({
            "shape": shape,
            "num_ops": num_ops,
            "original_time": original_time if original_error is None else float('inf'),
            "optimized_time": optimized_time if optimized_error is None else float('inf'),
            "original_error": original_error,
            "optimized_error": optimized_error,
            "speedup": speedup
        })
    
    def benchmark_memory_pool(self, num_tensors=1000, tensor_shape=(256, 256)):
        """
        Benchmarkt die Speicherpooling-Leistung
        
        Args:
            num_tensors: Anzahl der zu erstellenden Tensoren
            tensor_shape: Form der Testtensoren
        """
        logger.info(f"Benchmarking Speicherpooling mit {num_tensors} Tensoren der Form {tensor_shape}")
        
        # Benchmarkt ohne Pooling
        start_time = time.time()
        for _ in range(num_tensors):
            tensor = torch.zeros(tensor_shape, device=self.device)
            # Führe einfache Operation durch
            tensor = tensor + 1.0
            # Verwerfe Tensor (Garbage Collection)
            del tensor
        no_pool_time = time.time() - start_time
        logger.info(f"Ohne Pooling: {no_pool_time:.6f}s")
        
        # Benchmarkt mit Pooling
        self.tensor_pool.clear()  # Leere Pool vor dem Test
        start_time = time.time()
        for _ in range(num_tensors):
            tensor = self.tensor_pool.get(tensor_shape, device=self.device)
            # Führe einfache Operation durch
            tensor = tensor + 1.0
            # Gib Tensor zurück in den Pool
            self.tensor_pool.put(tensor)
        with_pool_time = time.time() - start_time
        logger.info(f"Mit Pooling: {with_pool_time:.6f}s")
        
        # Berechne Speedup
        speedup = no_pool_time / with_pool_time if with_pool_time > 0 else float('inf')
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Speichere Ergebnisse
        self.results["memory_pool"].append({
            "num_tensors": num_tensors,
            "tensor_shape": tensor_shape,
            "no_pool_time": no_pool_time,
            "with_pool_time": with_pool_time,
            "speedup": speedup,
            "pool_stats": self.tensor_pool.get_stats()
        })
    
    def run_all_benchmarks(self):
        """Führt alle Benchmarks aus"""
        logger.info("Starte umfassenden Speicheroptimierungs-Benchmark")
        
        # Konfigurationen für Benchmarks
        transfer_shapes = [(128, 128), (512, 512), (1024, 1024), (2048, 2048)]
        matmul_shapes = [(128, 128), (512, 512), (1024, 1024), (2048, 2048)]
        svd_shapes = [(64, 64), (128, 128), (256, 256), (512, 512)]
        pool_configs = [
            (100, (128, 128)),
            (500, (128, 128)),
            (100, (512, 512)),
            (500, (512, 512))
        ]
        
        # Führe Benchmarks aus
        for shape in transfer_shapes:
            self.benchmark_tensor_transfer(shape)
        
        for shape in matmul_shapes:
            self.benchmark_matmul(shape)
        
        for shape in svd_shapes:
            self.benchmark_svd(shape)
        
        for num_tensors, tensor_shape in pool_configs:
            self.benchmark_memory_pool(num_tensors, tensor_shape)
        
        # Speichere Ergebnisse
        self.save_results()
        
        # Erstelle Visualisierungen
        self.visualize_results()
    
    def save_results(self, filename=None):
        """
        Speichert die Benchmark-Ergebnisse in einer JSON-Datei
        
        Args:
            filename: Dateiname für die Ergebnisse (Default: generierter Name)
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"speicheroptimierung_benchmark_{timestamp}.json"
        
        # Erstelle vollständigen Pfad
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # Bereite Ergebnisse vor
        results_dict = {
            "timestamp": time.time(),
            "timestamp_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": {
                "system": platform.system(),
                "processor": platform.processor(),
                "is_apple_silicon": IS_APPLE_SILICON,
                "has_mlx": HAS_MLX,
                "torch_device": str(self.device)
            },
            "results": dict(self.results)
        }
        
        # Speichere als JSON
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Benchmark-Ergebnisse gespeichert in: {filepath}")
        return filepath
    
    def visualize_results(self, filename=None):
        """
        Erstellt Visualisierungen der Benchmark-Ergebnisse
        
        Args:
            filename: Dateiname für die Grafiken (Default: generierter Name)
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"speicheroptimierung_benchmark_{timestamp}.png"
        
        # Erstelle vollständigen Pfad
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # Erstelle Visualisierung
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("T-Mathematics Speicheroptimierung - Benchmark-Ergebnisse", fontsize=16)
        
        # 1. Tensor-Transfer-Benchmark
        if self.results["tensor_transfer"]:
            ax = axs[0, 0]
            shapes = [f"{r['shape'][0]}x{r['shape'][1]}" for r in self.results["tensor_transfer"]]
            original_times = [r["original_time"] for r in self.results["tensor_transfer"]]
            optimized_times = [r["optimized_time"] for r in self.results["tensor_transfer"]]
            speedups = [r["speedup"] for r in self.results["tensor_transfer"]]
            
            x = np.arange(len(shapes))
            width = 0.35
            
            ax.bar(x - width/2, original_times, width, label='Ursprünglich')
            ax.bar(x + width/2, optimized_times, width, label='Optimiert')
            
            ax.set_title("Tensor-Transfer (MPS ↔ MLX)")
            ax.set_ylabel("Zeit (s)")
            ax.set_xticks(x)
            ax.set_xticklabels(shapes)
            ax.legend()
            
            # Speedup auf sekundärer y-Achse
            ax2 = ax.twinx()
            ax2.plot(x, speedups, 'r-', marker='o', label='Speedup')
            ax2.set_ylabel("Speedup (×)", color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(bottom=0)
        
        # 2. Matrixmultiplikations-Benchmark
        if self.results["matmul"]:
            ax = axs[0, 1]
            shapes = [f"{r['shape'][0]}x{r['shape'][1]}" for r in self.results["matmul"]]
            original_times = [r["original_time"] for r in self.results["matmul"]]
            optimized_times = [r["optimized_time"] for r in self.results["matmul"]]
            speedups = [r["speedup"] for r in self.results["matmul"]]
            
            x = np.arange(len(shapes))
            width = 0.35
            
            ax.bar(x - width/2, original_times, width, label='Ursprünglich')
            ax.bar(x + width/2, optimized_times, width, label='Optimiert')
            
            ax.set_title("Matrixmultiplikation")
            ax.set_ylabel("Zeit (s)")
            ax.set_xticks(x)
            ax.set_xticklabels(shapes)
            ax.legend()
            
            # Speedup auf sekundärer y-Achse
            ax2 = ax.twinx()
            ax2.plot(x, speedups, 'r-', marker='o', label='Speedup')
            ax2.set_ylabel("Speedup (×)", color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(bottom=0)
        
        # 3. SVD-Benchmark
        if self.results["svd"]:
            ax = axs[1, 0]
            shapes = [f"{r['shape'][0]}x{r['shape'][1]}" for r in self.results["svd"]]
            original_times = []
            optimized_times = []
            
            for r in self.results["svd"]:
                # Setze unendliche Zeiten für Fehler auf einen hohen Wert für die Visualisierung
                orig_time = r["original_time"] if r["original_time"] != float('inf') else None
                opt_time = r["optimized_time"] if r["optimized_time"] != float('inf') else None
                original_times.append(orig_time)
                optimized_times.append(opt_time)
            
            x = np.arange(len(shapes))
            width = 0.35
            
            ax.bar(x - width/2, original_times, width, label='Ursprünglich')
            ax.bar(x + width/2, optimized_times, width, label='Optimiert')
            
            ax.set_title("Singulärwertzerlegung (SVD)")
            ax.set_ylabel("Zeit (s)")
            ax.set_xticks(x)
            ax.set_xticklabels(shapes)
            ax.legend()
            
            # Annotiere Fehler
            for i, r in enumerate(self.results["svd"]):
                if r["original_error"] is not None:
                    ax.annotate("Fehler", xy=(i - width/2, 0.1), xytext=(i - width/2, 0.5),
                                arrowprops=dict(arrowstyle="->", color='red'),
                                color='red', ha='center')
                if r["optimized_error"] is not None:
                    ax.annotate("Fehler", xy=(i + width/2, 0.1), xytext=(i + width/2, 0.5),
                                arrowprops=dict(arrowstyle="->", color='red'),
                                color='red', ha='center')
        
        # 4. Speicherpool-Benchmark
        if self.results["memory_pool"]:
            ax = axs[1, 1]
            configs = [f"{r['num_tensors']} x {r['tensor_shape'][0]}²" for r in self.results["memory_pool"]]
            no_pool_times = [r["no_pool_time"] for r in self.results["memory_pool"]]
            with_pool_times = [r["with_pool_time"] for r in self.results["memory_pool"]]
            speedups = [r["speedup"] for r in self.results["memory_pool"]]
            
            x = np.arange(len(configs))
            width = 0.35
            
            ax.bar(x - width/2, no_pool_times, width, label='Ohne Pool')
            ax.bar(x + width/2, with_pool_times, width, label='Mit Pool')
            
            ax.set_title("Speicherpool-Leistung")
            ax.set_ylabel("Zeit (s)")
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=45, ha='right')
            ax.legend()
            
            # Speedup auf sekundärer y-Achse
            ax2 = ax.twinx()
            ax2.plot(x, speedups, 'r-', marker='o', label='Speedup')
            ax2.set_ylabel("Speedup (×)", color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(bottom=0)
        
        # Layoutanpassung und Speichern
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Visualisierung gespeichert in: {filepath}")
        return filepath

    def generate_dashboard_data(self, filename=None):
        """
        Generiert eine JSON-Datei mit den Benchmark-Ergebnissen im VXOR-Dashboard-Format
        
        Args:
            filename: Dateiname für die Dashboard-Daten (Default: generierter Name)
        """
        if filename is None:
            filename = "t_mathematics_memory_optimization_benchmark.json"
        
        # Erstelle vollständigen Pfad
        filepath = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dashboard_data', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Bereite Daten im Dashboard-Format vor
        dashboard_data = {
            "benchmark_title": "T-Mathematics Speicheroptimierung",
            "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmark_metadata": {
                "platform": platform.system(),
                "processor": platform.processor(),
                "is_apple_silicon": IS_APPLE_SILICON,
                "has_mlx": HAS_MLX,
                "torch_device": str(self.device)
            },
            "benchmark_description": "Benchmark der Speicheroptimierungen für die T-Mathematics Engine",
            "categories": []
        }
        
        # Tensor-Transfer-Kategorie
        if self.results["tensor_transfer"]:
            category = {
                "name": "Tensor-Transfer",
                "description": "Speichertransfer zwischen MPS und MLX",
                "tests": []
            }
            
            for r in self.results["tensor_transfer"]:
                shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
                category["tests"].append({
                    "name": f"Transfer {shape_str}",
                    "description": f"Transfer eines {shape_str}-Tensors zwischen MPS und MLX",
                    "implementations": [
                        {
                            "name": "Original",
                            "result": r["original_time"],
                            "unit": "seconds",
                            "good_direction": "lower"
                        },
                        {
                            "name": "Optimiert",
                            "result": r["optimized_time"],
                            "unit": "seconds",
                            "good_direction": "lower"
                        }
                    ]
                })
            
            dashboard_data["categories"].append(category)
        
        # Matrixmultiplikations-Kategorie
        if self.results["matmul"]:
            category = {
                "name": "Matrixmultiplikation",
                "description": "Matrixmultiplikation mit verschiedenen Größen",
                "tests": []
            }
            
            for r in self.results["matmul"]:
                shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
                category["tests"].append({
                    "name": f"MatMul {shape_str}",
                    "description": f"Matrixmultiplikation mit {shape_str}-Matrizen",
                    "implementations": [
                        {
                            "name": "Original",
                            "result": r["original_time"],
                            "unit": "seconds",
                            "good_direction": "lower"
                        },
                        {
                            "name": "Optimiert",
                            "result": r["optimized_time"],
                            "unit": "seconds",
                            "good_direction": "lower"
                        }
                    ]
                })
            
            dashboard_data["categories"].append(category)
        
        # SVD-Kategorie
        if self.results["svd"]:
            category = {
                "name": "Singulärwertzerlegung",
                "description": "SVD mit verschiedenen Matrixgrößen",
                "tests": []
            }
            
            for r in self.results["svd"]:
                shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
                
                # Bereite Implementierungen vor
                implementations = []
                
                if r["original_error"] is None:
                    implementations.append({
                        "name": "Original",
                        "result": r["original_time"],
                        "unit": "seconds",
                        "good_direction": "lower"
                    })
                else:
                    implementations.append({
                        "name": "Original",
                        "result": None,
                        "unit": "seconds",
                        "good_direction": "lower",
                        "error": r["original_error"]
                    })
                
                if r["optimized_error"] is None:
                    implementations.append({
                        "name": "Optimiert",
                        "result": r["optimized_time"],
                        "unit": "seconds",
                        "good_direction": "lower"
                    })
                else:
                    implementations.append({
                        "name": "Optimiert",
                        "result": None,
                        "unit": "seconds",
                        "good_direction": "lower",
                        "error": r["optimized_error"]
                    })
                
                category["tests"].append({
                    "name": f"SVD {shape_str}",
                    "description": f"SVD einer {shape_str}-Matrix",
                    "implementations": implementations
                })
            
            dashboard_data["categories"].append(category)
        
        # Speicherpool-Kategorie
        if self.results["memory_pool"]:
            category = {
                "name": "Speicherpool",
                "description": "Leistung des Tensor-Speicherpools",
                "tests": []
            }
            
            for r in self.results["memory_pool"]:
                shape_str = f"{r['tensor_shape'][0]}x{r['tensor_shape'][1]}"
                test_name = f"{r['num_tensors']} Tensoren ({shape_str})"
                
                category["tests"].append({
                    "name": test_name,
                    "description": f"Erstellung und Verwendung von {r['num_tensors']} Tensoren der Größe {shape_str}",
                    "implementations": [
                        {
                            "name": "Ohne Pool",
                            "result": r["no_pool_time"],
                            "unit": "seconds",
                            "good_direction": "lower"
                        },
                        {
                            "name": "Mit Pool",
                            "result": r["with_pool_time"],
                            "unit": "seconds",
                            "good_direction": "lower"
                        }
                    ]
                })
            
            dashboard_data["categories"].append(category)
        
        # Speichere als JSON
        with open(filepath, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        logger.info(f"Dashboard-Daten gespeichert in: {filepath}")
        return filepath


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description="Benchmark für die Speicheroptimierung der T-Mathematics Engine")
    parser.add_argument("--save", type=str, default=None, help="Dateiname für die Speicherung der Ergebnisse")
    args = parser.parse_args()
    
    # Führe Benchmark aus
    benchmark = MemoryBenchmark()
    benchmark.run_all_benchmarks()
    
    # Generiere Dashboard-Daten
    benchmark.generate_dashboard_data()


if __name__ == "__main__":
    main()
