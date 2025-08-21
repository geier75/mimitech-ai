#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test-Suite für die optimierte VXTensorConverter Klasse.

Diese Tests validieren die Performance-Optimierungen und die Korrektheit
der Tensor-Konvertierungsfunktionen, insbesondere für die MLX-Integration 
auf Apple Silicon und die PRISM-Engine-Integration.
"""

import os
import sys
import unittest
import time
import logging
import numpy as np

# Pfade für den Import setzen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.vx_tensor_converter import VXTensorConverter, TensorType

# Logger für Testausgaben
logger = logging.getLogger(__name__)

# Globale Variablen für Tests
SKIP_SLOW_TESTS = os.environ.get('SKIP_SLOW_TESTS', 'False').lower() in ('true', '1', 't')


class TestVXTensorConverter(unittest.TestCase):
    """Testet die optimierte VXTensorConverter Klasse."""
    
    def setUp(self):
        """Setup für alle Tests."""
        self.converter = VXTensorConverter(enable_caching=True, max_cache_size=1000)
        self.test_array_small = np.random.random((10, 10)).astype(np.float32)
        self.test_array_medium = np.random.random((100, 100)).astype(np.float32)
        self.test_array_large = np.random.random((1000, 1000)).astype(np.float32)
        
        # MLX und PyTorch wenn verfügbar aktivieren
        self.has_mlx = False
        self.has_torch = False
        
        try:
            import mlx.core as mx
            self.has_mlx = True
            self.mlx_small = mx.array(self.test_array_small)
            self.converter.optimize_for_mlx(True)
        except ImportError:
            logger.warning("MLX nicht verfügbar. MLX-Tests werden übersprungen.")
        
        try:
            import torch
            self.has_torch = True
            self.torch_small = torch.tensor(self.test_array_small)
        except ImportError:
            logger.warning("PyTorch nicht verfügbar. PyTorch-Tests werden übersprungen.")
    
    def test_init_with_parameters(self):
        """Testet die Initialisierung mit Parametern."""
        converter = VXTensorConverter(enable_caching=False, max_cache_size=500)
        self.assertFalse(converter.enable_caching)
        self.assertEqual(converter.max_cache_size, 500)
        
        # Prüfe Standardwerte
        self.assertTrue(converter._mlx_jit_enabled)
        self.assertTrue(converter._use_bfloat16)
        self.assertTrue(converter._prism_compatible)
    
    def test_cache_operations(self):
        """Testet die Cache-Operationen."""
        converter = VXTensorConverter(enable_caching=True, max_cache_size=10)
        
        # Cache sollte initial leer sein
        stats = converter.get_cache_statistics()
        self.assertEqual(stats['cache_size'], 0)
        self.assertEqual(stats['cache_hits'], 0)
        self.assertEqual(stats['cache_misses'], 0)
        
        # Erste Konvertierung sollte Cache Miss sein
        result1 = converter.to_numpy(self.test_array_small)
        stats = converter.get_cache_statistics()
        self.assertEqual(stats['cache_misses'], 1)
        
        # Zweite Konvertierung des selben Arrays sollte Cache Hit sein
        result2 = converter.to_numpy(self.test_array_small)
        stats = converter.get_cache_statistics()
        self.assertEqual(stats['cache_hits'], 1)
        
        # Cache leeren
        cleared = converter.clear_cache()
        self.assertEqual(cleared, 1)  # Ein Eintrag sollte gelöscht worden sein
        stats = converter.get_cache_statistics()
        self.assertEqual(stats['cache_size'], 0)
    
    def test_numpy_conversion_accuracy(self):
        """Testet die Genauigkeit der NumPy-Konvertierung."""
        # NumPy zu NumPy sollte identisch sein
        np_result = self.converter.to_numpy(self.test_array_small)
        np.testing.assert_array_equal(np_result, self.test_array_small)
        
        # Test für PyTorch
        if self.has_torch:
            torch_result = self.converter.to_numpy(self.torch_small)
            np.testing.assert_array_almost_equal(torch_result, self.test_array_small)
        
        # Test für MLX
        if self.has_mlx:
            mlx_result = self.converter.to_numpy(self.mlx_small)
            np.testing.assert_array_almost_equal(mlx_result, self.test_array_small)
    
    def test_optimize_for_mlx(self):
        """Testet die MLX-Optimierungsfunktion."""
        # Deaktivieren und testen
        self.converter.optimize_for_mlx(False)
        self.assertFalse(self.converter._mlx_jit_enabled)
        self.assertFalse(self.converter._use_bfloat16)
        
        # Wieder aktivieren und testen
        success = self.converter.optimize_for_mlx(True)
        self.assertTrue(self.converter._mlx_jit_enabled)
        self.assertTrue(self.converter._use_bfloat16)
        
        # Erfolg sollte von MLX-Verfügbarkeit abhängen
        self.assertEqual(success, self.has_mlx)
    
    def test_optimize_for_prism(self):
        """Testet die PRISM-Optimierungsfunktion."""
        # Deaktivieren und testen
        self.converter.optimize_for_prism(False)
        self.assertFalse(self.converter._prism_compatible)
        self.assertFalse(self.converter._optimized_batch_ops)
        
        # Wieder aktivieren und testen
        success = self.converter.optimize_for_prism(True)
        self.assertTrue(self.converter._prism_compatible)
        self.assertTrue(self.converter._optimized_batch_ops)
        self.assertTrue(success)
    
    @unittest.skipIf(SKIP_SLOW_TESTS, "Langsame Tests übersprungen")
    def test_performance_numpy_large(self):
        """Testet die Performance der NumPy-Konvertierung bei großen Arrays."""
        # Performance-Test für große Arrays
        self.converter.clear_cache()
        
        # Ohne Cache
        start = time.time()
        for _ in range(10):
            self.converter.to_numpy(self.test_array_large)
        no_cache_time = time.time() - start
        
        # Mit Cache
        start = time.time()
        for _ in range(10):
            self.converter.to_numpy(self.test_array_large)
        with_cache_time = time.time() - start
        
        # Cache sollte schneller sein
        self.assertLess(with_cache_time, no_cache_time * 0.5)
        logger.info(f"Ohne Cache: {no_cache_time:.5f}s, Mit Cache: {with_cache_time:.5f}s, Speedup: {no_cache_time/with_cache_time:.2f}x")
    
    @unittest.skipIf(not SKIP_SLOW_TESTS, "MLX-Tests sind nur für Apple Silicon verfügbar")
    def test_mlx_conversion(self):
        """Testet die MLX-Konvertierungsfunktionen."""
        if not self.has_mlx:
            self.skipTest("MLX ist nicht verfügbar.")
        
        import mlx.core as mx
        
        # NumPy zu MLX
        mlx_array = self.converter.to_mlx(self.test_array_small)
        self.assertEqual(mlx_array.shape, self.test_array_small.shape)
        
        # MLX zu NumPy
        np_array = self.converter.to_numpy(mlx_array)
        np.testing.assert_array_almost_equal(np_array, self.test_array_small)
        
        # BFloat16-Test, wenn verfügbar
        if hasattr(mx, 'bfloat16'):
            mlx_bf16 = self.mlx_small.astype(mx.bfloat16)
            np_bf16 = self.converter.to_numpy(mlx_bf16)
            # Toleranz für BFloat16 ist höher
            np.testing.assert_array_almost_equal(np_bf16, self.test_array_small, decimal=2)
    
    @unittest.skipIf(not SKIP_SLOW_TESTS, "PyTorch-Tests sind optional")
    def test_torch_conversion(self):
        """Testet die PyTorch-Konvertierungsfunktionen."""
        if not self.has_torch:
            self.skipTest("PyTorch ist nicht verfügbar.")
        
        import torch
        
        # NumPy zu PyTorch
        torch_tensor = self.converter.to_torch(self.test_array_small)
        self.assertEqual(torch_tensor.shape, self.test_array_small.shape)
        
        # PyTorch zu NumPy
        np_tensor = self.converter.to_numpy(torch_tensor)
        np.testing.assert_array_almost_equal(np_tensor, self.test_array_small)
        
        # BFloat16-Test, wenn verfügbar
        if hasattr(torch, 'bfloat16'):
            torch_bf16 = self.torch_small.to(torch.bfloat16)
            np_bf16 = self.converter.to_numpy(torch_bf16)
            # Toleranz für BFloat16 ist höher
            np.testing.assert_array_almost_equal(np_bf16, self.test_array_small, decimal=2)
    
    def test_prism_compatible_conversion(self):
        """Testet die PRISM-kompatible Konvertierung."""
        # Aktiviere PRISM-Kompatibilität
        self.converter.optimize_for_prism(True)
        
        # Erstelle einen Array mit NaN/Inf-Werten
        test_array_with_nans = np.array([1.0, float('nan'), 3.0, float('inf'), -float('inf')])
        
        # Konvertiere zu NumPy mit PRISM-Kompatibilität
        result = self.converter.to_numpy(test_array_with_nans)
        
        # Prüfe, dass keine NaN/Inf-Werte mehr vorhanden sind
        self.assertFalse(np.isnan(result).any())
        self.assertFalse(np.isinf(result).any())
        
        # Prüfe, dass die richtigen Werte ersetzt wurden
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 0.0)  # NaN wird zu 0
        self.assertEqual(result[2], 3.0)
        self.assertEqual(result[3], 1e30)  # Inf wird zu 1e30
        self.assertEqual(result[4], -1e30)  # -Inf wird zu -1e30
    
    @unittest.skipIf(SKIP_SLOW_TESTS, "Vollständige Benchmark-Tests übersprungen")
    def test_benchmark_all_conversions(self):
        """Führt Benchmark-Tests für alle Konvertierungen durch."""
        results = {}
        
        # NumPy zu NumPy (Baseline)
        start = time.time()
        for _ in range(100):
            self.converter.to_numpy(self.test_array_medium)
        results['numpy_to_numpy'] = time.time() - start
        
        # PyTorch Tests
        if self.has_torch:
            import torch
            torch_medium = torch.tensor(self.test_array_medium)
            
            # NumPy zu PyTorch
            start = time.time()
            for _ in range(100):
                self.converter.to_torch(self.test_array_medium)
            results['numpy_to_torch'] = time.time() - start
            
            # PyTorch zu NumPy
            start = time.time()
            for _ in range(100):
                self.converter.to_numpy(torch_medium)
            results['torch_to_numpy'] = time.time() - start
        
        # MLX Tests
        if self.has_mlx:
            import mlx.core as mx
            mlx_medium = mx.array(self.test_array_medium)
            
            # NumPy zu MLX
            start = time.time()
            for _ in range(100):
                self.converter.to_mlx(self.test_array_medium)
            results['numpy_to_mlx'] = time.time() - start
            
            # MLX zu NumPy
            start = time.time()
            for _ in range(100):
                self.converter.to_numpy(mlx_medium)
            results['mlx_to_numpy'] = time.time() - start
        
        # Ergebnisse ausgeben
        logger.info("Benchmark-Ergebnisse (100 Iterationen mit Arrays der Größe 100x100):")
        for key, value in results.items():
            logger.info(f"{key}: {value:.5f}s")


if __name__ == '__main__':
    # Logger-Konfiguration
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Tests ausführen
    unittest.main()
