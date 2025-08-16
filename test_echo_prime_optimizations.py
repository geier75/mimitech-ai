#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark-Tests für die optimierte ECHO-PRIME-Integration

Dieser Test vergleicht die Leistung der optimierten ECHO-PRIME-Integration
mit der ursprünglichen Implementierung und misst die Performance-Vorteile
durch verbesserte Tensor-Konvertierungen und Matrix-Operationen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import time
import logging
import numpy as np
import torch
from typing import List, Tuple, Dict, Any

# Importiere sowohl die ursprüngliche als auch die optimierte Implementierung
from miso.math.t_mathematics.echo_prime_integration import TimelineAnalysisEngine
from optimized_echo_prime_integration import OptimizedTimelineAnalysisEngine

# Importiere die optimierte MatrixCore-Implementierung für Matrix-Batchoperationen
from test_matrix_core_optimized import OptimizedMatrixCore

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("echo_prime_benchmark")

# MLX-Status überprüfen
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
    logger.info("MLX verfügbar - Apple Silicon Optimierungen werden aktiviert")
except ImportError:
    MLX_AVAILABLE = False
    logger.info("MLX nicht verfügbar - Fallback auf PyTorch/NumPy")

class EchoPrimeBenchmark:
    """Benchmark-Suite für ECHO-PRIME-Optimierungen"""
    
    def __init__(self):
        # Initialisiere beide Engines
        logger.info("Initialisiere Original- und optimierte TimelineAnalysisEngine...")
        self.original_engine = TimelineAnalysisEngine(use_mlx=MLX_AVAILABLE)
        self.optimized_engine = OptimizedTimelineAnalysisEngine(use_mlx=MLX_AVAILABLE, enable_caching=True)
        
        # Initialisiere optimierte MatrixCore für Matrix-Tests
        self.matrix_core = OptimizedMatrixCore()
        
        # Ergebnis-Dictionary
        self.results = {}
        
    def run_timeline_similarity_benchmark(self, 
                                         num_runs: int = 100, 
                                         timeline_sizes: List[Tuple[int, int]] = [(10, 15), (50, 32), (100, 64)]):
        """
        Benchmark für die Zeitlinien-Ähnlichkeitsfunktion
        
        Args:
            num_runs: Anzahl der Durchläufe pro Größe
            timeline_sizes: Liste von (seq_len, features) Tupeln
        """
        logger.info(f"Starte Timeline Similarity Benchmark mit {num_runs} Durchläufen...")
        results = {}
        
        for seq_len, features in timeline_sizes:
            logger.info(f"Teste Größe: {seq_len}x{features}")
            
            # Erstelle zufällige Zeitlinien für diesen Test
            timelines_a = [torch.randn(seq_len, features) for _ in range(num_runs)]
            timelines_b = [torch.randn(seq_len, features) for _ in range(num_runs)]
            
            # Messen: Original Engine
            start_time = time.time()
            for i in range(num_runs):
                _ = self.original_engine.timeline_similarity(timelines_a[i], timelines_b[i])
            original_time = time.time() - start_time
            
            # Messen: Optimierte Engine (erster Durchlauf ohne Cache)
            self.optimized_engine.clear_cache()
            start_time = time.time()
            for i in range(num_runs):
                _ = self.optimized_engine.timeline_similarity(timelines_a[i], timelines_b[i])
            optimized_time_no_cache = time.time() - start_time
            
            # Messen: Optimierte Engine (zweiter Durchlauf mit Cache für identische Operationen)
            start_time = time.time()
            for i in range(num_runs):
                _ = self.optimized_engine.timeline_similarity(timelines_a[i], timelines_b[i])
            optimized_time_with_cache = time.time() - start_time
            
            # Ergebnisse speichern
            speedup_no_cache = original_time / optimized_time_no_cache if optimized_time_no_cache > 0 else float('inf')
            speedup_with_cache = original_time / optimized_time_with_cache if optimized_time_with_cache > 0 else float('inf')
            
            results[f"{seq_len}x{features}"] = {
                "original_time": original_time,
                "optimized_time_no_cache": optimized_time_no_cache,
                "optimized_time_with_cache": optimized_time_with_cache,
                "speedup_no_cache": speedup_no_cache,
                "speedup_with_cache": speedup_with_cache
            }
            
            logger.info(f"Timeline {seq_len}x{features}: Original={original_time:.4f}s, " +
                       f"Optimiert (ohne Cache)={optimized_time_no_cache:.4f}s, " +
                       f"Speedup={speedup_no_cache:.2f}x, " +
                       f"Optimiert (mit Cache)={optimized_time_with_cache:.4f}s, " + 
                       f"Speedup (mit Cache)={speedup_with_cache:.2f}x")
        
        self.results["timeline_similarity"] = results
        return results
    
    def run_temporal_attention_benchmark(self, 
                                       num_runs: int = 50,
                                       attention_configs: List[Dict[str, int]] = [
                                           {"batch": 4, "queries": 5, "seq_len": 10, "embed_dim": 16, "value_dim": 32},
                                           {"batch": 8, "queries": 10, "seq_len": 20, "embed_dim": 32, "value_dim": 64},
                                           {"batch": 16, "queries": 20, "seq_len": 40, "embed_dim": 64, "value_dim": 128}
                                       ]):
        """
        Benchmark für temporale Aufmerksamkeitsmechanismen
        
        Args:
            num_runs: Anzahl der Durchläufe pro Konfiguration
            attention_configs: Liste von Aufmerksamkeitskonfigurationen
        """
        logger.info(f"Starte Temporal Attention Benchmark mit {num_runs} Durchläufen...")
        results = {}
        
        for config in attention_configs:
            batch = config["batch"]
            queries = config["queries"]
            seq_len = config["seq_len"]
            embed_dim = config["embed_dim"]
            value_dim = config["value_dim"]
            
            config_name = f"B{batch}_Q{queries}_S{seq_len}_E{embed_dim}_V{value_dim}"
            logger.info(f"Teste Konfiguration: {config_name}")
            
            # Erstelle zufällige Tensoren für den Test
            all_queries = [torch.randn(batch, queries, embed_dim) for _ in range(num_runs)]
            all_keys = [torch.randn(batch, seq_len, embed_dim) for _ in range(num_runs)]
            all_values = [torch.randn(batch, seq_len, value_dim) for _ in range(num_runs)]
            
            # Optional: Masken für 50% der Tests
            masks = []
            for i in range(num_runs):
                if i % 2 == 0:  # Für jede zweite Iteration
                    mask = torch.ones(batch, queries, seq_len, dtype=torch.bool)
                    # Füge zufällige Masken ein (50% der Zeitlinien-Positions-Kombinationen maskieren)
                    for b in range(batch):
                        for q in range(queries):
                            for s in range(seq_len):
                                if np.random.random() < 0.5:
                                    mask[b, q, s] = False
                    masks.append(mask)
                else:
                    masks.append(None)
            
            # Messen: Original Engine
            start_time = time.time()
            for i in range(num_runs):
                _ = self.original_engine.temporal_attention(
                    all_queries[i], all_keys[i], all_values[i], masks[i]
                )
            original_time = time.time() - start_time
            
            # Messen: Optimierte Engine (erster Durchlauf ohne Cache)
            self.optimized_engine.clear_cache()
            start_time = time.time()
            for i in range(num_runs):
                _ = self.optimized_engine.temporal_attention(
                    all_queries[i], all_keys[i], all_values[i], masks[i]
                )
            optimized_time_no_cache = time.time() - start_time
            
            # Messen: Optimierte Engine (zweiter Durchlauf mit Cache für identische Operationen)
            start_time = time.time()
            for i in range(num_runs):
                _ = self.optimized_engine.temporal_attention(
                    all_queries[i], all_keys[i], all_values[i], masks[i]
                )
            optimized_time_with_cache = time.time() - start_time
            
            # Ergebnisse speichern
            speedup_no_cache = original_time / optimized_time_no_cache if optimized_time_no_cache > 0 else float('inf')
            speedup_with_cache = original_time / optimized_time_with_cache if optimized_time_with_cache > 0 else float('inf')
            
            results[config_name] = {
                "original_time": original_time,
                "optimized_time_no_cache": optimized_time_no_cache,
                "optimized_time_with_cache": optimized_time_with_cache,
                "speedup_no_cache": speedup_no_cache,
                "speedup_with_cache": speedup_with_cache
            }
            
            logger.info(f"Attention {config_name}: Original={original_time:.4f}s, " +
                       f"Optimiert (ohne Cache)={optimized_time_no_cache:.4f}s, " +
                       f"Speedup={speedup_no_cache:.2f}x, " +
                       f"Optimiert (mit Cache)={optimized_time_with_cache:.4f}s, " + 
                       f"Speedup (mit Cache)={speedup_with_cache:.2f}x")
        
        self.results["temporal_attention"] = results
        return results
    
    def run_matrix_batch_operations_benchmark(self,
                                            num_runs: int = 20,
                                            matrix_configs: List[Dict[str, Any]] = [
                                                {"desc": "Spezieller 5x(10x10 @ 10x15) Fall", 
                                                 "n_matrices": 5, 
                                                 "shapes": [(10, 10), (10, 15)]},
                                                {"desc": "Größere Batch-Multiplikation", 
                                                 "n_matrices": 10, 
                                                 "shapes": [(20, 30), (30, 15)]},
                                                {"desc": "Gemischte Größen-Batch", 
                                                 "n_matrices": 8, 
                                                 "shapes": [(32, 64), (64, 48)]}
                                            ]):
        """
        Benchmark für Batch-Matrix-Operationen
        
        Args:
            num_runs: Anzahl der Durchläufe pro Konfiguration
            matrix_configs: Liste von Matrix-Batch-Konfigurationen
        """
        logger.info(f"Starte Matrix Batch Operations Benchmark mit {num_runs} Durchläufen...")
        results = {}
        
        for config in matrix_configs:
            desc = config["desc"]
            n_matrices = config["n_matrices"]
            shape_a, shape_b = config["shapes"]
            
            logger.info(f"Teste Konfiguration: {desc} - {n_matrices}x({shape_a[0]}x{shape_a[1]} @ {shape_b[0]}x{shape_b[1]})")
            
            # Erstelle zufällige Matrizen für diesen Test
            all_tests = []
            for _ in range(num_runs):
                matrices_a = [np.random.randn(*shape_a) for _ in range(n_matrices)]
                matrices_b = [np.random.randn(*shape_b) for _ in range(n_matrices)]
                all_tests.append((matrices_a, matrices_b))
            
            # Messen: Standard NumPy-Implementierung mit for-Schleife
            start_time = time.time()
            for i in range(num_runs):
                matrices_a, matrices_b = all_tests[i]
                results_standard = []
                for j in range(n_matrices):
                    results_standard.append(np.matmul(matrices_a[j], matrices_b[j]))
            standard_time = time.time() - start_time
            
            # Messen: Optimierte prism_batch_operation
            start_time = time.time()
            for i in range(num_runs):
                matrices_a, matrices_b = all_tests[i]
                results_prism = self.matrix_core.prism_batch_operation(
                    matrices_a, matrices_b, operation="matmul"
                )
            optimized_time = time.time() - start_time
            
            # Ergebnisse speichern
            speedup = standard_time / optimized_time if optimized_time > 0 else float('inf')
            
            config_name = f"{n_matrices}x({shape_a[0]}x{shape_a[1]}@{shape_b[0]}x{shape_b[1]})"
            results[config_name] = {
                "description": desc,
                "standard_time": standard_time,
                "optimized_time": optimized_time,
                "speedup": speedup
            }
            
            logger.info(f"Matrix {config_name}: Standard={standard_time:.4f}s, " +
                       f"Optimiert={optimized_time:.4f}s, " +
                       f"Speedup={speedup:.2f}x")
        
        self.results["matrix_batch_operations"] = results
        return results
    
    def run_all_benchmarks(self):
        """Führt alle Benchmarks aus und gibt die Ergebnisse zurück"""
        logger.info("Starte alle Benchmarks...")
        
        self.run_timeline_similarity_benchmark()
        self.run_temporal_attention_benchmark()
        self.run_matrix_batch_operations_benchmark()
        
        return self.results
    
    def print_summary(self):
        """Druckt eine Zusammenfassung der Benchmark-Ergebnisse"""
        if not self.results:
            logger.warning("Keine Benchmark-Ergebnisse verfügbar. Bitte zuerst Benchmarks ausführen.")
            return
            
        print("\n" + "="*80)
        print(" "*30 + "BENCHMARK-ZUSAMMENFASSUNG")
        print("="*80 + "\n")
        
        # Timeline Similarity
        if "timeline_similarity" in self.results:
            print("\n--- ZEITLINIEN-ÄHNLICHKEIT BENCHMARK ---\n")
            for size, result in self.results["timeline_similarity"].items():
                print(f"Zeitlinie {size}:")
                print(f"  Original:                  {result['original_time']:.4f}s")
                print(f"  Optimiert (ohne Cache):    {result['optimized_time_no_cache']:.4f}s")
                print(f"  Optimiert (mit Cache):     {result['optimized_time_with_cache']:.4f}s")
                print(f"  Speedup (ohne Cache):      {result['speedup_no_cache']:.2f}x")
                print(f"  Speedup (mit Cache):       {result['speedup_with_cache']:.2f}x\n")
        
        # Temporal Attention        
        if "temporal_attention" in self.results:
            print("\n--- TEMPORAL ATTENTION BENCHMARK ---\n")
            for config, result in self.results["temporal_attention"].items():
                print(f"Konfiguration {config}:")
                print(f"  Original:                  {result['original_time']:.4f}s")
                print(f"  Optimiert (ohne Cache):    {result['optimized_time_no_cache']:.4f}s")
                print(f"  Optimiert (mit Cache):     {result['optimized_time_with_cache']:.4f}s")
                print(f"  Speedup (ohne Cache):      {result['speedup_no_cache']:.2f}x")
                print(f"  Speedup (mit Cache):       {result['speedup_with_cache']:.2f}x\n")
        
        # Matrix Batch Operations        
        if "matrix_batch_operations" in self.results:
            print("\n--- MATRIX BATCH OPERATIONS BENCHMARK ---\n")
            for config, result in self.results["matrix_batch_operations"].items():
                print(f"{result['description']} ({config}):")
                print(f"  Standard NumPy:            {result['standard_time']:.4f}s")
                print(f"  Optimierte MatrixCore:     {result['optimized_time']:.4f}s")
                print(f"  Speedup:                   {result['speedup']:.2f}x\n")
        
        print("\n" + "="*80)
        print(" "*25 + "ENDE DER BENCHMARK-ZUSAMMENFASSUNG")
        print("="*80 + "\n")
        
        # Cache-Statistiken
        print("\n--- CACHE-STATISTIKEN ---\n")
        print(f"Cache-Stats: {self.optimized_engine.cache_stats()}")
        print("\n")


if __name__ == "__main__":
    # Banner
    print("\n" + "="*80)
    print(" "*30 + "ECHO-PRIME BENCHMARK")
    print("="*80 + "\n")
    
    # Benchmark ausführen
    benchmark = EchoPrimeBenchmark()
    benchmark.run_all_benchmarks()
    
    # Ergebnisse ausgeben
    benchmark.print_summary()
