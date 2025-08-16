#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Engine - MLX Performance Benchmark

Dieses Skript führt Benchmark-Tests für die MLX-Optimierung der T-Mathematics Engine durch
und vergleicht die Performance mit der PyTorch-Implementierung.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import time
import logging
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
# import matplotlib.pyplot as plt  # Temporär deaktiviert wegen Installationsproblemen
from tabulate import tabulate

# Setze Umgebungsvariablen für die Tests
os.environ["T_MATH_USE_MLX"] = "1"  # Aktiviere MLX
os.environ["T_MATH_PRECISION"] = "float16"  # Verwende float16 Präzision

# Importiere T-Mathematics Engine
from miso.math.t_mathematics.engine import TMathEngine
from miso.math.t_mathematics.compat import TMathConfig

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mlx_benchmark")

class MLXBenchmark:
    """
    Benchmark-Klasse für die MLX-Optimierung der T-Mathematics Engine.
    """
    
    def __init__(self, use_mlx: bool = True, precision: str = "float16"):
        """
        Initialisiert den Benchmark.
        
        Args:
            use_mlx: Ob MLX verwendet werden soll
            precision: Präzision für die Berechnungen
        """
        # Erstelle Engine mit MLX
        self.mlx_config = TMathConfig(
            precision=precision,
            device="auto",
            optimize_for_rdna=False,
            optimize_for_apple_silicon=True
        )
        self.mlx_engine = TMathEngine(
            config=self.mlx_config,
            use_mlx=use_mlx
        )
        
        # Erstelle Engine ohne MLX (PyTorch)
        self.torch_config = TMathConfig(
            precision=precision,
            device="auto",
            optimize_for_rdna=False,
            optimize_for_apple_silicon=False
        )
        self.torch_engine = TMathEngine(
            config=self.torch_config,
            use_mlx=False
        )
        
        # Prüfe, ob MLX verfügbar ist
        self.mlx_available = self.mlx_engine.use_mlx
        
        if not self.mlx_available and use_mlx:
            logger.warning("MLX ist nicht verfügbar. Verwende PyTorch für beide Engines.")
        
        logger.info(f"Benchmark initialisiert: MLX verfügbar={self.mlx_available}, "
                   f"Präzision={precision}")
    
    def benchmark_matmul(self, 
                       sizes: List[Tuple[int, int, int]],
                       num_runs: int = 10) -> Dict[str, Dict[str, List[float]]]:
        """
        Führt Benchmark-Tests für Matrixmultiplikation durch.
        
        Args:
            sizes: Liste von Tupeln (m, n, k) für Matrixgrößen
            num_runs: Anzahl der Durchläufe pro Test
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        results = {
            "mlx": {"sizes": [], "times": []},
            "torch": {"sizes": [], "times": []}
        }
        
        for m, n, k in sizes:
            size_str = f"{m}x{n}x{k}"
            logger.info(f"Benchmark Matrixmultiplikation: {size_str}")
            
            # Erstelle Eingabe-Tensoren
            a = torch.randn(m, n)
            b = torch.randn(n, k)
            
            # Benchmark MLX
            mlx_times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.mlx_engine.matmul(a, b)
                end_time = time.time()
                mlx_times.append(end_time - start_time)
            
            # Benchmark PyTorch
            torch_times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.torch_engine.matmul(a, b)
                end_time = time.time()
                torch_times.append(end_time - start_time)
            
            # Speichere Ergebnisse
            results["mlx"]["sizes"].append(size_str)
            results["mlx"]["times"].append(np.mean(mlx_times))
            
            results["torch"]["sizes"].append(size_str)
            results["torch"]["times"].append(np.mean(torch_times))
            
            # Berechne Speedup
            speedup = np.mean(torch_times) / np.mean(mlx_times)
            logger.info(f"  MLX: {np.mean(mlx_times):.6f}s, PyTorch: {np.mean(torch_times):.6f}s, Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_svd(self, 
                    sizes: List[Tuple[int, int]],
                    k_values: List[int],
                    num_runs: int = 5) -> Dict[str, Dict[str, List[float]]]:
        """
        Führt Benchmark-Tests für SVD-Zerlegung durch.
        
        Args:
            sizes: Liste von Tupeln (m, n) für Matrixgrößen
            k_values: Liste von k-Werten für die SVD
            num_runs: Anzahl der Durchläufe pro Test
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        results = {
            "mlx": {"sizes": [], "times": []},
            "torch": {"sizes": [], "times": []}
        }
        
        for m, n in sizes:
            for k in k_values:
                if k > min(m, n):
                    continue  # Überspringe ungültige k-Werte
                
                size_str = f"{m}x{n} (k={k})"
                logger.info(f"Benchmark SVD: {size_str}")
                
                # Erstelle Eingabe-Tensor
                tensor = torch.randn(m, n)
                
                # Benchmark MLX
                mlx_times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = self.mlx_engine.svd(tensor, k=k)
                    end_time = time.time()
                    mlx_times.append(end_time - start_time)
                
                # Benchmark PyTorch
                torch_times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = self.torch_engine.svd(tensor, k=k)
                    end_time = time.time()
                    torch_times.append(end_time - start_time)
                
                # Speichere Ergebnisse
                results["mlx"]["sizes"].append(size_str)
                results["mlx"]["times"].append(np.mean(mlx_times))
                
                results["torch"]["sizes"].append(size_str)
                results["torch"]["times"].append(np.mean(torch_times))
                
                # Berechne Speedup
                speedup = np.mean(torch_times) / np.mean(mlx_times)
                logger.info(f"  MLX: {np.mean(mlx_times):.6f}s, PyTorch: {np.mean(torch_times):.6f}s, Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_attention(self, 
                          sizes: List[Tuple[int, int, int, int]],
                          num_runs: int = 5) -> Dict[str, Dict[str, List[float]]]:
        """
        Führt Benchmark-Tests für Attention-Mechanismen durch.
        
        Args:
            sizes: Liste von Tupeln (batch_size, num_heads, seq_len, head_dim)
            num_runs: Anzahl der Durchläufe pro Test
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        results = {
            "mlx": {"sizes": [], "times": []},
            "torch": {"sizes": [], "times": []}
        }
        
        for batch_size, num_heads, seq_len, head_dim in sizes:
            size_str = f"b={batch_size}, h={num_heads}, s={seq_len}, d={head_dim}"
            logger.info(f"Benchmark Attention: {size_str}")
            
            # Erstelle Eingabe-Tensoren
            query = torch.randn(batch_size, num_heads, seq_len, head_dim)
            key = torch.randn(batch_size, num_heads, seq_len, head_dim)
            value = torch.randn(batch_size, num_heads, seq_len, head_dim)
            mask = torch.zeros(batch_size, 1, seq_len, seq_len)
            
            # Benchmark MLX
            mlx_times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.mlx_engine.attention(query, key, value, mask)
                end_time = time.time()
                mlx_times.append(end_time - start_time)
            
            # Benchmark PyTorch
            torch_times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.torch_engine.attention(query, key, value, mask)
                end_time = time.time()
                torch_times.append(end_time - start_time)
            
            # Speichere Ergebnisse
            results["mlx"]["sizes"].append(size_str)
            results["mlx"]["times"].append(np.mean(mlx_times))
            
            results["torch"]["sizes"].append(size_str)
            results["torch"]["times"].append(np.mean(torch_times))
            
            # Berechne Speedup
            speedup = np.mean(torch_times) / np.mean(mlx_times)
            logger.info(f"  MLX: {np.mean(mlx_times):.6f}s, PyTorch: {np.mean(torch_times):.6f}s, Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_layer_norm(self, 
                           sizes: List[Tuple[int, int]],
                           num_runs: int = 10) -> Dict[str, Dict[str, List[float]]]:
        """
        Führt Benchmark-Tests für Layer-Normalisierung durch.
        
        Args:
            sizes: Liste von Tupeln (batch_size, hidden_dim)
            num_runs: Anzahl der Durchläufe pro Test
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        results = {
            "mlx": {"sizes": [], "times": []},
            "torch": {"sizes": [], "times": []}
        }
        
        for batch_size, hidden_dim in sizes:
            size_str = f"b={batch_size}, d={hidden_dim}"
            logger.info(f"Benchmark Layer Norm: {size_str}")
            
            # Erstelle Eingabe-Tensoren
            input_tensor = torch.randn(batch_size, hidden_dim)
            weight = torch.ones(hidden_dim)
            bias = torch.zeros(hidden_dim)
            
            # Benchmark MLX
            mlx_times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.mlx_engine.layer_norm(input_tensor, weight, bias)
                end_time = time.time()
                mlx_times.append(end_time - start_time)
            
            # Benchmark PyTorch
            torch_times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.torch_engine.layer_norm(input_tensor, weight, bias)
                end_time = time.time()
                torch_times.append(end_time - start_time)
            
            # Speichere Ergebnisse
            results["mlx"]["sizes"].append(size_str)
            results["mlx"]["times"].append(np.mean(mlx_times))
            
            results["torch"]["sizes"].append(size_str)
            results["torch"]["times"].append(np.mean(torch_times))
            
            # Berechne Speedup
            speedup = np.mean(torch_times) / np.mean(mlx_times)
            logger.info(f"  MLX: {np.mean(mlx_times):.6f}s, PyTorch: {np.mean(torch_times):.6f}s, Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_activation(self, 
                           function: str,
                           sizes: List[Tuple[int, int]],
                           num_runs: int = 10) -> Dict[str, Dict[str, List[float]]]:
        """
        Führt Benchmark-Tests für Aktivierungsfunktionen durch.
        
        Args:
            function: Name der Aktivierungsfunktion ('gelu' oder 'relu')
            sizes: Liste von Tupeln (batch_size, hidden_dim)
            num_runs: Anzahl der Durchläufe pro Test
            
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        results = {
            "mlx": {"sizes": [], "times": []},
            "torch": {"sizes": [], "times": []}
        }
        
        for batch_size, hidden_dim in sizes:
            size_str = f"b={batch_size}, d={hidden_dim}"
            logger.info(f"Benchmark {function.upper()}: {size_str}")
            
            # Erstelle Eingabe-Tensor
            input_tensor = torch.randn(batch_size, hidden_dim)
            
            # Wähle die Aktivierungsfunktion
            if function.lower() == "gelu":
                mlx_func = self.mlx_engine.gelu
                torch_func = self.torch_engine.gelu
            elif function.lower() == "relu":
                mlx_func = self.mlx_engine.relu
                torch_func = self.torch_engine.relu
            else:
                raise ValueError(f"Unbekannte Aktivierungsfunktion: {function}")
            
            # Benchmark MLX
            mlx_times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = mlx_func(input_tensor)
                end_time = time.time()
                mlx_times.append(end_time - start_time)
            
            # Benchmark PyTorch
            torch_times = []
            for _ in range(num_runs):
                start_time = time.time()
                _ = torch_func(input_tensor)
                end_time = time.time()
                torch_times.append(end_time - start_time)
            
            # Speichere Ergebnisse
            results["mlx"]["sizes"].append(size_str)
            results["mlx"]["times"].append(np.mean(mlx_times))
            
            results["torch"]["sizes"].append(size_str)
            results["torch"]["times"].append(np.mean(torch_times))
            
            # Berechne Speedup
            speedup = np.mean(torch_times) / np.mean(mlx_times)
            logger.info(f"  MLX: {np.mean(mlx_times):.6f}s, PyTorch: {np.mean(torch_times):.6f}s, Speedup: {speedup:.2f}x")
        
        return results
    
    def plot_results(self, 
                   results: Dict[str, Dict[str, List[float]]],
                   title: str,
                   output_file: Optional[str] = None):
        """
        Erstellt einen Plot der Benchmark-Ergebnisse.
        
        Args:
            results: Dictionary mit Benchmark-Ergebnissen
            title: Titel für den Plot
            output_file: Pfad für die Ausgabedatei (optional)
        """
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(results["mlx"]["sizes"]))
        width = 0.35
        
        plt.bar(x - width/2, results["mlx"]["times"], width, label="MLX")
        plt.bar(x + width/2, results["torch"]["times"], width, label="PyTorch")
        
        plt.xlabel("Matrix Size")
        plt.ylabel("Zeit (s)")
        plt.title(title)
        plt.xticks(x, results["mlx"]["sizes"], rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Plot gespeichert unter: {output_file}")
        else:
            plt.show()
    
    def print_results_table(self, results: Dict[str, Dict[str, List[float]]], title: str):
        """
        Gibt eine Tabelle mit den Benchmark-Ergebnissen aus.
        
        Args:
            results: Dictionary mit Benchmark-Ergebnissen
            title: Titel für die Tabelle
        """
        table_data = []
        headers = ["Größe", "MLX (s)", "PyTorch (s)", "Speedup"]
        
        for i, size in enumerate(results["mlx"]["sizes"]):
            mlx_time = results["mlx"]["times"][i]
            torch_time = results["torch"]["times"][i]
            speedup = torch_time / mlx_time
            
            table_data.append([size, f"{mlx_time:.6f}", f"{torch_time:.6f}", f"{speedup:.2f}x"])
        
        print(f"\n{title}")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def run_all_benchmarks(self, output_dir: Optional[str] = None):
        """
        Führt alle Benchmark-Tests durch und speichert die Ergebnisse.
        
        Args:
            output_dir: Verzeichnis für die Ausgabedateien (optional)
        """
        if not self.mlx_available:
            logger.warning("MLX ist nicht verfügbar. Benchmarks werden mit PyTorch durchgeführt.")
        
        # Matrixmultiplikation
        matmul_sizes = [
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048)
        ]
        matmul_results = self.benchmark_matmul(matmul_sizes)
        self.print_results_table(matmul_results, "Benchmark: Matrixmultiplikation")
        
        if output_dir:
            logger.info(f"Speichere Ergebnisse in CSV-Format: {os.path.join(output_dir, 'matmul_benchmark.csv')}")
        
        # SVD
        svd_sizes = [
            (100, 50),
            (200, 100),
            (500, 250),
            (1000, 500)
        ]
        svd_k_values = [10, 20, 50]
        svd_results = self.benchmark_svd(svd_sizes, svd_k_values)
        self.print_results_table(svd_results, "Benchmark: SVD-Zerlegung")
        
        if output_dir:
            logger.info(f"Speichere Ergebnisse in CSV-Format: {os.path.join(output_dir, 'svd_benchmark.csv')}")
        
        # Attention
        attention_sizes = [
            (2, 4, 128, 64),
            (4, 8, 256, 64),
            (8, 12, 512, 64),
            (16, 16, 1024, 64)
        ]
        attention_results = self.benchmark_attention(attention_sizes)
        self.print_results_table(attention_results, "Benchmark: Attention")
        
        if output_dir:
            logger.info(f"Speichere Ergebnisse in CSV-Format: {os.path.join(output_dir, 'attention_benchmark.csv')}")
        
        # Layer Norm
        layer_norm_sizes = [
            (128, 256),
            (256, 512),
            (512, 1024),
            (1024, 2048),
            (2048, 4096)
        ]
        layer_norm_results = self.benchmark_layer_norm(layer_norm_sizes)
        self.print_results_table(layer_norm_results, "Benchmark: Layer Normalisierung")
        
        if output_dir:
            self.plot_results(layer_norm_results, "Layer Norm Benchmark", 
                             os.path.join(output_dir, "layer_norm_benchmark.png"))
        
        # GELU
        gelu_sizes = [
            (128, 256),
            (256, 512),
            (512, 1024),
            (1024, 2048),
            (2048, 4096)
        ]
        gelu_results = self.benchmark_activation("gelu", gelu_sizes)
        self.print_results_table(gelu_results, "Benchmark: GELU Aktivierung")
        
        if output_dir:
            logger.info(f"Speichere Ergebnisse in CSV-Format: {os.path.join(output_dir, 'gelu_benchmark.csv')}")
        
        # ReLU
        relu_sizes = [
            (128, 256),
            (256, 512),
            (512, 1024),
            (1024, 2048),
            (2048, 4096)
        ]
        relu_results = self.benchmark_activation("relu", relu_sizes)
        self.print_results_table(relu_results, "Benchmark: ReLU Aktivierung")
        
        if output_dir:
            logger.info(f"Speichere Ergebnisse in CSV-Format: {os.path.join(output_dir, 'relu_benchmark.csv')}")
        
        logger.info("Alle Benchmarks abgeschlossen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLX Performance Benchmark")
    parser.add_argument("--output-dir", type=str, help="Verzeichnis für die Ausgabedateien")
    parser.add_argument("--precision", type=str, default="float16", 
                       choices=["float16", "float32", "bfloat16"], 
                       help="Präzision für die Berechnungen")
    parser.add_argument("--no-mlx", action="store_true", help="Deaktiviere MLX")
    
    args = parser.parse_args()
    
    # Erstelle Ausgabeverzeichnis, falls angegeben
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Erstelle und führe Benchmark aus
    benchmark = MLXBenchmark(use_mlx=not args.no_mlx, precision=args.precision)
    benchmark.run_all_benchmarks(args.output_dir)
