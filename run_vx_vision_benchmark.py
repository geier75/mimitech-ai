#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION Benchmark Runner

Führt den VX-VISION Benchmark mit optimalen MLX-Einstellungen aus.
Führt zuerst einen MLX-Warmup durch, um "Metal Binding Freeze" zu verhindern.
"""

import os
import sys
import logging
import time
import argparse

# Konfiguration des Loggers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VX-VISION.benchmark_runner")

def main():
    parser = argparse.ArgumentParser(description="VX-VISION Benchmark Runner")
    parser.add_argument("--skip-warmup", action="store_true", help="Überspringt den MLX Warmup")
    parser.add_argument("--cpu-only", action="store_true", help="Erzwingt CPU-Modus (keine GPU/ANE)")
    parser.add_argument("--disable-jit", action="store_true", help="Deaktiviert MLX JIT-Kompilierung")
    parser.add_argument("--test-mode", choices=["full", "quick", "single_op"], default="quick",
                        help="Benchmark-Modus (full, quick, single_op)")
    args = parser.parse_args()
    
    # MLX-Einstellungen
    if args.cpu_only:
        os.environ["MLX_USE_CPU"] = "1"
        logger.info("CPU-Modus erzwungen für den Benchmark")
    
    if args.disable_jit:
        os.environ["MLX_DISABLE_JIT"] = "1"
        logger.info("MLX JIT-Kompilierung deaktiviert")
    
    # Starte den Benchmark-Prozess
    start_time = time.time()
    
    # 1. MLX Warmup (optional)
    if not args.skip_warmup:
        logger.info("Starte MLX Kernel Warmup...")
        from vxor.core.mlx_kernel_warmup import run_warmup
        warmup_results = run_warmup(verbose=True)
        if not warmup_results["success"]:
            logger.warning(f"Warmup nicht erfolgreich: {warmup_results.get('reason')}")
    
    # 2. Starte den eigentlichen Benchmark
    logger.info(f"Starte VX-VISION Benchmark im {args.test_mode}-Modus...")
    
    if args.test_mode == "quick":
        from benchmarks.vision.quick_benchmark import run_quick_benchmark
        run_quick_benchmark()
    elif args.test_mode == "full":
        from benchmarks.vision.kernel_benchmark import KernelBenchmark
        benchmark = KernelBenchmark()
        benchmark.run_benchmark()
    elif args.test_mode == "single_op":
        from benchmarks.vision.quick_benchmark import run_quick_benchmark
        run_quick_benchmark(operations=["resize"])
    
    # Gesamtzeit
    total_time = time.time() - start_time
    logger.info(f"Benchmark abgeschlossen in {total_time:.2f} Sekunden")

if __name__ == "__main__":
    main()
