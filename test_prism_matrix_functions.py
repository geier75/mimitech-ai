#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für die implementierten PrismMatrix-Funktionen

Testet die neu implementierten Funktionen create_matrix() und _apply_variation()
in der PrismMatrix-Klasse mit Fokus auf MLX-Optimierung für Apple Silicon.
"""

import os
import sys
import logging
import numpy as np
import time
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_prism_matrix")

# Füge das Projektverzeichnis zum Pfad hinzu
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Importiere die PrismMatrix-Klasse
from miso.simulation.prism_matrix import PrismMatrix

# Überprüfe, ob Apple Silicon verfügbar ist
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
logger.info(f"Testumgebung: Apple Silicon verfügbar: {is_apple_silicon}")

def test_create_matrix():
    """Test der create_matrix-Funktion mit verschiedenen Dimensionen"""
    logger.info("=== Teste create_matrix-Funktion ===")
    
    # Teste verschiedene Dimensionen
    for dimensions in [2, 3, 4, 5]:
        logger.info(f"Teste {dimensions}-dimensionale Matrix")
        
        # Erstelle eine PrismMatrix-Instanz
        matrix = PrismMatrix(dimensions=dimensions, initial_size=8)
        
        # Überprüfe die Dimensionen der Matrix
        expected_shape = tuple([8] * dimensions)
        actual_shape = matrix.matrix.shape
        
        if actual_shape == expected_shape:
            logger.info(f"✅ Matrix-Form korrekt: {actual_shape}")
        else:
            logger.error(f"❌ Matrix-Form falsch: {actual_shape}, erwartet: {expected_shape}")
    
    logger.info("Test abgeschlossen.\n")

def test_apply_variation():
    """Test der _apply_variation-Funktion mit verschiedenen Variationsfaktoren"""
    logger.info("=== Teste _apply_variation-Funktion ===")
    
    # Erstelle eine PrismMatrix-Instanz
    matrix = PrismMatrix(dimensions=3, initial_size=5)
    
    # Setze Anfangswerte für bessere Testbarkeit
    if hasattr(matrix, 'use_t_math') and matrix.use_t_math:
        import torch
        matrix.matrix = matrix.t_math_engine.prepare_tensor(torch.ones([5, 5, 5]))
        logger.info("Initialisiere Matrix mit T-Mathematics Engine")
    else:
        matrix.matrix = np.ones([5, 5, 5])
        logger.info("Initialisiere Matrix mit NumPy")
    
    # Teste verschiedene Variationsfaktoren
    for variation_factor in [0.0, 0.1, 0.5, 1.0]:
        logger.info(f"Teste Variationsfaktor {variation_factor}")
        
        # Speichere originale Matrix-Werte für Vergleich
        if hasattr(matrix, 'use_t_math') and matrix.use_t_math:
            original_sum = matrix.matrix.sum().item()
        else:
            original_sum = matrix.matrix.sum()
        
        # Wende Variation an
        start_time = time.time()
        varied_matrix = matrix._apply_variation(matrix.matrix, variation_factor, seed=42)
        duration = time.time() - start_time
        
        # Berechne Summe der variierten Matrix für einfachen Vergleich
        if hasattr(matrix, 'use_t_math') and matrix.use_t_math:
            varied_sum = varied_matrix.sum().item()
        else:
            varied_sum = varied_matrix.sum()
        
        # Berechne relative Änderung
        relative_change = abs((varied_sum - original_sum) / original_sum)
        
        logger.info(f"Variationsfaktor: {variation_factor}, Relative Änderung: {relative_change:.6f}, Dauer: {duration*1000:.2f}ms")
        
        # Überprüfe, dass die Variation dem Faktor entspricht (ungefähr)
        if variation_factor == 0.0 and relative_change < 0.001:
            logger.info("✅ Keine Variation bei Faktor 0.0")
        elif 0.0 < variation_factor <= 1.0 and 0.0 < relative_change:
            logger.info("✅ Variation angemessen für Faktor > 0.0")
        else:
            logger.error(f"❌ Unerwartete Variation: {relative_change} für Faktor {variation_factor}")
    
    # Teste Reproduzierbarkeit mit gleichem Seed
    if hasattr(matrix, 'use_t_math') and matrix.use_t_math:
        import torch
        matrix.matrix = matrix.t_math_engine.prepare_tensor(torch.ones([5, 5, 5]))
    else:
        matrix.matrix = np.ones([5, 5, 5])
    
    var1 = matrix._apply_variation(matrix.matrix, 0.5, seed=123)
    var2 = matrix._apply_variation(matrix.matrix, 0.5, seed=123)
    
    # Vergleiche die beiden Variationen
    if hasattr(matrix, 'use_t_math') and matrix.use_t_math:
        diff = (var1 - var2).abs().sum().item()
    else:
        diff = np.abs(var1 - var2).sum()
    
    if diff < 0.001:
        logger.info("✅ Reproduzierbarkeit mit gleichem Seed bestätigt")
    else:
        logger.error(f"❌ Variation nicht reproduzierbar mit gleichem Seed: Differenz {diff}")
    
    logger.info("Test abgeschlossen.\n")

def test_mlx_optimization():
    """Test der MLX-Optimierung, falls verfügbar"""
    if not is_apple_silicon:
        logger.info("=== MLX-Optimierungstest übersprungen (kein Apple Silicon) ===")
        return
    
    logger.info("=== Teste MLX-Optimierung für Apple Silicon ===")
    
    # Erstelle eine PrismMatrix-Instanz
    matrix = PrismMatrix(dimensions=4, initial_size=16)
    
    # Überprüfe, ob MLX verwendet wird
    has_mlx = False
    try:
        import mlx.core as mx
        has_mlx = True
        logger.info("MLX-Bibliothek ist verfügbar")
    except ImportError:
        logger.warning("MLX-Bibliothek ist nicht verfügbar")
    
    if has_mlx and hasattr(matrix, 'use_t_math') and matrix.use_t_math:
        if hasattr(matrix.t_math_engine, 'use_mlx') and matrix.t_math_engine.use_mlx:
            logger.info("✅ MLX-Optimierung ist aktiv in der T-Mathematics Engine")
            
            # Vergleiche Leistung: NumPy vs. MLX
            logger.info("Leistungsvergleich für _apply_variation:")
            
            # NumPy-Version
            np_matrix = np.ones([32, 32, 32, 32])
            start_time = time.time()
            _ = matrix._apply_variation(np_matrix, 0.5, seed=42)
            np_time = time.time() - start_time
            
            # MLX-Version (wenn verfügbar)
            if hasattr(matrix.t_math_engine, 'tensor_to_mlx'):
                mlx_matrix = matrix.t_math_engine.tensor_to_mlx(np_matrix)
                start_time = time.time()
                _ = matrix._apply_variation(mlx_matrix, 0.5, seed=42)
                mlx_time = time.time() - start_time
                
                logger.info(f"NumPy-Zeit: {np_time*1000:.2f}ms, MLX-Zeit: {mlx_time*1000:.2f}ms")
                speedup = np_time / mlx_time if mlx_time > 0 else 0
                logger.info(f"MLX-Beschleunigung: {speedup:.2f}x")
                
                if speedup > 1.0:
                    logger.info("✅ MLX-Optimierung bietet Leistungsvorteil")
                else:
                    logger.warning("⚠️ MLX-Optimierung bietet keinen Leistungsvorteil")
        else:
            logger.warning("⚠️ MLX-Optimierung ist nicht aktiv in der T-Mathematics Engine")
    else:
        logger.warning("⚠️ T-Mathematics Engine mit MLX-Unterstützung ist nicht verfügbar")
    
    logger.info("Test abgeschlossen.\n")

def run_all_tests():
    """Führt alle Tests aus"""
    logger.info("=== PRISM-Matrix Funktionstests ===")
    logger.info(f"Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"System: {os.uname().sysname} {os.uname().release} auf {os.uname().machine}")
    logger.info("="*50)
    
    # Führe Tests aus
    test_create_matrix()
    test_apply_variation()
    test_mlx_optimization()
    
    logger.info("=== Alle Tests abgeschlossen ===")

if __name__ == "__main__":
    run_all_tests()
