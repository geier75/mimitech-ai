#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für die MLX-Optimierung der T-Mathematics Engine.

Diese Tests überprüfen die korrekte Funktionalität der MLX-Optimierung
für Apple Silicon in der T-Mathematics Engine.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import numpy as np
import torch
import platform

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from miso.math.t_mathematics.engine import TMathEngine
from miso.math.t_mathematics.compat import TMathConfig

# Prüfe auf Apple Silicon
is_apple_silicon = False
try:
    is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
except:
    pass

# Prüfe auf MLX
has_mlx = False
try:
    import mlx.core
    has_mlx = True
except ImportError:
    pass

# Umgebungsvariablen für Tests
os.environ["T_MATH_USE_MLX"] = "1"
os.environ["T_MATH_PRECISION"] = "float16"
os.environ["T_MATH_DEVICE"] = "auto"


@unittest.skipIf(not is_apple_silicon or not has_mlx, "Benötigt Apple Silicon und MLX")
class TestTMathematicsMLX(unittest.TestCase):
    """Test-Suite für die MLX-Optimierung der T-Mathematics Engine."""
    
    def setUp(self):
        """Initialisiert die Test-Umgebung."""
        # Erstelle eine Engine-Instanz mit MLX-Optimierung
        self.config = TMathConfig(
            precision="float16",
            device="auto",
            optimize_for_rdna=False,
            optimize_for_apple_silicon=True
        )
        self.engine = TMathEngine(
            config=self.config,
            use_mlx=True
        )
        
        # Stelle sicher, dass MLX verfügbar ist
        self.assertTrue(self.engine.use_mlx, "MLX sollte aktiviert sein")
        self.assertIsNotNone(self.engine.mlx_backend, "MLX-Backend sollte verfügbar sein")
    
    def test_matmul(self):
        """Testet die Matrixmultiplikation mit MLX-Optimierung."""
        # Erstelle Tensoren
        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        
        # Berechne mit MLX
        result_mlx = self.engine.matmul(a, b)
        
        # Berechne mit PyTorch als Referenz
        result_torch = torch.matmul(a, b)
        
        # Vergleiche die Ergebnisse (mit Toleranz für Rundungsfehler)
        np.testing.assert_allclose(
            result_mlx.detach().cpu().numpy(),
            result_torch.detach().cpu().numpy(),
            rtol=1e-3, atol=1e-3
        )
    
    def test_svd(self):
        """Testet die SVD-Zerlegung mit MLX-Optimierung."""
        # Erstelle Tensor
        tensor = torch.randn(50, 30)
        
        # Berechne mit MLX
        U_mlx, S_mlx, V_mlx = self.engine.svd(tensor, k=10)
        
        # Berechne mit PyTorch als Referenz
        U_torch, S_torch, V_torch = torch.svd(tensor)
        U_torch = U_torch[:, :10]
        S_torch = S_torch[:10]
        V_torch = V_torch[:, :10]
        
        # Vergleiche die Ergebnisse (mit Toleranz für Rundungsfehler)
        # Beachte: Die Vorzeichen können unterschiedlich sein, daher vergleichen wir die absoluten Werte
        np.testing.assert_allclose(
            np.abs(U_mlx.detach().cpu().numpy()),
            np.abs(U_torch.detach().cpu().numpy()),
            rtol=1e-2, atol=1e-2
        )
        np.testing.assert_allclose(
            S_mlx.detach().cpu().numpy(),
            S_torch.detach().cpu().numpy(),
            rtol=1e-2, atol=1e-2
        )
    
    def test_attention(self):
        """Testet den Attention-Mechanismus mit MLX-Optimierung."""
        # Erstelle Tensoren
        batch_size = 2
        num_heads = 4
        seq_len = 10
        head_dim = 16
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        
        # Berechne mit MLX
        output_mlx, weights_mlx = self.engine.attention(query, key, value, mask)
        
        # Überprüfe die Formen
        self.assertEqual(output_mlx.shape, (batch_size, num_heads, seq_len, head_dim))
        self.assertEqual(weights_mlx.shape, (batch_size, num_heads, seq_len, seq_len))
    
    def test_layer_norm(self):
        """Testet die Layer-Normalisierung mit MLX-Optimierung."""
        # Erstelle Tensoren
        input_tensor = torch.randn(10, 20)
        weight = torch.ones(20)
        bias = torch.zeros(20)
        
        # Berechne mit MLX
        result_mlx = self.engine.layer_norm(input_tensor, weight, bias)
        
        # Berechne mit PyTorch als Referenz
        result_torch = torch.nn.functional.layer_norm(
            input_tensor, input_tensor.shape[-1:], weight, bias
        )
        
        # Vergleiche die Ergebnisse (mit Toleranz für Rundungsfehler)
        np.testing.assert_allclose(
            result_mlx.detach().cpu().numpy(),
            result_torch.detach().cpu().numpy(),
            rtol=1e-3, atol=1e-3
        )
    
    def test_gelu(self):
        """Testet die GELU-Aktivierungsfunktion mit MLX-Optimierung."""
        # Erstelle Tensor
        input_tensor = torch.randn(100, 100)
        
        # Berechne mit MLX
        result_mlx = self.engine.gelu(input_tensor)
        
        # Berechne mit PyTorch als Referenz
        result_torch = torch.nn.functional.gelu(input_tensor)
        
        # Vergleiche die Ergebnisse (mit Toleranz für Rundungsfehler)
        np.testing.assert_allclose(
            result_mlx.detach().cpu().numpy(),
            result_torch.detach().cpu().numpy(),
            rtol=1e-3, atol=1e-3
        )
    
    def test_relu(self):
        """Testet die ReLU-Aktivierungsfunktion mit MLX-Optimierung."""
        # Erstelle Tensor
        input_tensor = torch.randn(100, 100)
        
        # Berechne mit MLX
        result_mlx = self.engine.relu(input_tensor)
        
        # Berechne mit PyTorch als Referenz
        result_torch = torch.nn.functional.relu(input_tensor)
        
        # Vergleiche die Ergebnisse (mit Toleranz für Rundungsfehler)
        np.testing.assert_allclose(
            result_mlx.detach().cpu().numpy(),
            result_torch.detach().cpu().numpy(),
            rtol=1e-3, atol=1e-3
        )
    
    def test_performance(self):
        """Testet die Performance-Verbesserung durch MLX."""
        import time
        
        # Erstelle große Tensoren für Performance-Test
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        
        # Erstelle eine Engine ohne MLX für Vergleich
        engine_no_mlx = TMathEngine(
            config=self.config,
            use_mlx=False
        )
        
        # Wärme die Engines auf
        _ = self.engine.matmul(a, b)
        _ = engine_no_mlx.matmul(a, b)
        
        # Messe die Zeit für MLX
        start_time = time.time()
        for _ in range(10):
            _ = self.engine.matmul(a, b)
        mlx_time = time.time() - start_time
        
        # Messe die Zeit für PyTorch
        start_time = time.time()
        for _ in range(10):
            _ = engine_no_mlx.matmul(a, b)
        pytorch_time = time.time() - start_time
        
        # Gib die Zeiten aus
        print(f"\nPerformance-Vergleich:")
        print(f"MLX-Zeit: {mlx_time:.4f} Sekunden")
        print(f"PyTorch-Zeit: {pytorch_time:.4f} Sekunden")
        print(f"Beschleunigung: {pytorch_time / mlx_time:.2f}x")
        
        # Die MLX-Version sollte schneller sein
        # Hinweis: Dies ist ein weicher Test, da die tatsächliche Performance
        # von der Hardware und anderen Faktoren abhängt
        self.assertLessEqual(mlx_time, pytorch_time * 1.2, 
                            "MLX sollte nicht wesentlich langsamer sein als PyTorch")


class TestTMathematicsMLXFallback(unittest.TestCase):
    """
    Test-Suite für die T-Mathematics Engine mit MLX-Fallback.
    
    Diese Tests überprüfen, ob die Engine korrekt auf PyTorch zurückfällt,
    wenn MLX nicht verfügbar ist.
    """
    
    def setUp(self):
        """Initialisiert die Test-Umgebung."""
        # Erstelle eine Engine-Instanz mit deaktiviertem MLX
        self.config = TMathConfig(
            precision="float32",
            device="cpu",
            optimize_for_rdna=False,
            optimize_for_apple_silicon=True
        )
        self.engine = TMathEngine(
            config=self.config,
            use_mlx=False
        )
    
    def test_fallback(self):
        """Testet, ob die Engine korrekt auf PyTorch zurückfällt."""
        self.assertFalse(self.engine.use_mlx, "MLX sollte deaktiviert sein")
        
        # Erstelle Tensoren
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        
        # Berechne mit der Engine (sollte PyTorch verwenden)
        result = self.engine.matmul(a, b)
        
        # Berechne mit PyTorch als Referenz
        result_torch = torch.matmul(a, b)
        
        # Vergleiche die Ergebnisse
        np.testing.assert_allclose(
            result.detach().cpu().numpy(),
            result_torch.detach().cpu().numpy(),
            rtol=1e-5, atol=1e-5
        )


if __name__ == '__main__':
    unittest.main()
