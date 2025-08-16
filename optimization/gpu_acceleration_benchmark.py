#!/usr/bin/env python3
"""
GPU-Acceleration Benchmark für Apple M4 Max
Massive GFLOPS-Steigerung durch optimierte MLX-Integration
"""

import time
import numpy as np
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    HAS_MLX = True
    logger.info("✅ MLX verfügbar für GPU-Acceleration")
except ImportError:
    HAS_MLX = False
    logger.warning("❌ MLX nicht verfügbar")

class GPUAccelerationBenchmark:
    """Benchmark für GPU-Acceleration auf Apple M4 Max"""
    
    def __init__(self):
        self.results = {}
        self.matrix_sizes = [256, 512, 1024, 2048, 4096]
        self.iterations = 5
        
        if HAS_MLX:
            # Optimiere MLX für M4 Max
            if hasattr(mx, 'set_memory_pool_size'):
                mx.set_memory_pool_size(2 << 30)  # 2GB
            if hasattr(mx, 'enable_fusion'):
                mx.enable_fusion(True)
            logger.info("🚀 MLX für M4 Max optimiert")
    
    def calculate_gflops(self, matrix_size: int, time_seconds: float) -> float:
        """Berechnet GFLOPS für Matrix-Multiplikation"""
        # Matrix-Multiplikation: 2 * n^3 - n^2 FLOPs
        flops = 2 * matrix_size**3 - matrix_size**2
        gflops = flops / (time_seconds * 1e9)
        return gflops
    
    def benchmark_numpy(self, size: int) -> Tuple[float, float]:
        """Benchmark NumPy CPU Performance"""
        logger.info(f"📊 NumPy Benchmark {size}x{size}")
        
        # Erstelle Matrizen
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        # Warmup
        _ = np.matmul(a, b)
        
        # Benchmark
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            result = np.matmul(a, b)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        gflops = self.calculate_gflops(size, avg_time)
        
        logger.info(f"  NumPy: {avg_time:.4f}s, {gflops:.1f} GFLOPS")
        return avg_time, gflops
    
    def benchmark_mlx_gpu(self, size: int) -> Tuple[float, float]:
        """Benchmark MLX GPU Performance auf Apple M4 Max"""
        if not HAS_MLX:
            return 0.0, 0.0
            
        logger.info(f"🚀 MLX GPU Benchmark {size}x{size}")
        
        # Erstelle MLX-Matrizen
        a = mx.random.normal((size, size), dtype=mx.float32)
        b = mx.random.normal((size, size), dtype=mx.float32)
        
        # Warmup mit GPU-Optimierung
        _ = mx.matmul(a, b)
        mx.eval(_)  # Force evaluation
        
        # Benchmark
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            result = mx.matmul(a, b)
            mx.eval(result)  # Force evaluation für korrekte Zeitmessung
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        gflops = self.calculate_gflops(size, avg_time)
        
        logger.info(f"  MLX GPU: {avg_time:.4f}s, {gflops:.1f} GFLOPS")
        return avg_time, gflops
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Führt umfassenden GPU-Acceleration Benchmark durch"""
        logger.info("🎯 Starte GPU-Acceleration Benchmark für Apple M4 Max")
        
        results = {
            'matrix_sizes': self.matrix_sizes,
            'numpy_results': {'times': [], 'gflops': []},
            'mlx_results': {'times': [], 'gflops': []},
            'speedup_factors': [],
            'gflops_improvements': []
        }
        
        for size in self.matrix_sizes:
            logger.info(f"\n📏 Matrix-Größe: {size}x{size}")
            
            # NumPy Benchmark
            numpy_time, numpy_gflops = self.benchmark_numpy(size)
            results['numpy_results']['times'].append(numpy_time)
            results['numpy_results']['gflops'].append(numpy_gflops)
            
            # MLX GPU Benchmark
            if HAS_MLX:
                mlx_time, mlx_gflops = self.benchmark_mlx_gpu(size)
                results['mlx_results']['times'].append(mlx_time)
                results['mlx_results']['gflops'].append(mlx_gflops)
                
                # Berechne Verbesserungen
                if numpy_time > 0:
                    speedup = numpy_time / mlx_time if mlx_time > 0 else 0
                    gflops_improvement = mlx_gflops / numpy_gflops if numpy_gflops > 0 else 0
                    
                    results['speedup_factors'].append(speedup)
                    results['gflops_improvements'].append(gflops_improvement)
                    
                    logger.info(f"  🚀 Speedup: {speedup:.2f}x")
                    logger.info(f"  ⚡ GFLOPS Verbesserung: {gflops_improvement:.2f}x")
            else:
                results['mlx_results']['times'].append(0)
                results['mlx_results']['gflops'].append(0)
                results['speedup_factors'].append(0)
                results['gflops_improvements'].append(0)
        
        return results
    
    def print_summary(self, results: Dict):
        """Druckt Zusammenfassung der Benchmark-Ergebnisse"""
        logger.info("\n" + "="*60)
        logger.info("🏆 GPU-ACCELERATION BENCHMARK ZUSAMMENFASSUNG")
        logger.info("="*60)
        
        if HAS_MLX and results['speedup_factors']:
            max_speedup = max(results['speedup_factors'])
            max_gflops_improvement = max(results['gflops_improvements'])
            avg_speedup = np.mean(results['speedup_factors'])
            avg_gflops_improvement = np.mean(results['gflops_improvements'])
            
            logger.info(f"📈 Maximaler Speedup: {max_speedup:.2f}x")
            logger.info(f"⚡ Maximale GFLOPS-Steigerung: {max_gflops_improvement:.2f}x")
            logger.info(f"📊 Durchschnittlicher Speedup: {avg_speedup:.2f}x")
            logger.info(f"🎯 Durchschnittliche GFLOPS-Steigerung: {avg_gflops_improvement:.2f}x")
            
            # Beste Performance
            best_mlx_gflops = max(results['mlx_results']['gflops'])
            best_numpy_gflops = max(results['numpy_results']['gflops'])
            
            logger.info(f"\n🚀 Beste MLX GPU Performance: {best_mlx_gflops:.1f} GFLOPS")
            logger.info(f"🔧 Beste NumPy CPU Performance: {best_numpy_gflops:.1f} GFLOPS")
            
            if best_mlx_gflops > 1000:
                logger.info("🎉 MASSIVE GFLOPS-STEIGERUNG ERREICHT! (>1 TFLOPS)")
        else:
            logger.warning("❌ MLX nicht verfügbar - keine GPU-Acceleration möglich")
        
        logger.info("="*60)

def main():
    """Hauptfunktion für GPU-Acceleration Benchmark"""
    benchmark = GPUAccelerationBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    benchmark.print_summary(results)
    
    return results

if __name__ == "__main__":
    main()
