#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Einfacher Test für T-Mathematics Engine Optimierungen
"""

import os
import sys
import time
import logging
import numpy as np

# Konfiguriere Logging mit höchstem Level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.math.t_mathematics.optimizations.simple_test")

logger.info("Teste Importe...")

# Prüfe auf MLX
try:
    import mlx.core as mx
    HAS_MLX = True
    logger.info("MLX verfügbar")
except ImportError as e:
    HAS_MLX = False
    mx = None
    logger.error(f"MLX nicht verfügbar: {e}")

logger.info("MLX-Backend wird importiert...")
# MLX-Backend importieren
try:
    from miso.math.t_mathematics.mlx_support import MLXBackend
    logger.info("MLXBackend erfolgreich importiert")
except ImportError as e:
    logger.error(f"MLXBackend konnte nicht importiert werden: {e}")
    sys.exit(1)

logger.info("Optimierungsmodul wird importiert...")
# Optimierungen importieren
try:
    from miso.math.t_mathematics.optimizations.integration import optimize_mlx_backend, configure_optimizations
    logger.info("Optimierungsmodul erfolgreich importiert")
except ImportError as e:
    logger.error(f"Optimierungsmodul konnte nicht importiert werden: {e}")
    sys.exit(1)

def test_optimization():
    """Einfacher Test der Optimierungen"""
    logger.info("Starte einfachen Test...")
    
    # Konfiguriere Optimierungen
    try:
        logger.info("Konfiguriere Optimierungen...")
        configure_optimizations(optimization_level=2, enable_telemetry=True)
        logger.info("Optimierungen konfiguriert")
    except Exception as e:
        logger.error(f"Fehler bei der Konfiguration der Optimierungen: {e}")
    
    # Erstelle und optimiere Backend
    try:
        logger.info("Erstelle Backend...")
        original_backend = MLXBackend(precision="float32")
        logger.info("Backend erstellt")
        
        logger.info("Optimiere Backend...")
        optimized_backend = optimize_mlx_backend(original_backend, optimization_level=2)
        logger.info("Backend optimiert")
    except Exception as e:
        logger.error(f"Fehler beim Erstellen/Optimieren des Backends: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Einfache Matrix erstellen
    try:
        logger.info("Erstelle Testmatrix...")
        matrix = np.random.rand(10, 10).astype(np.float32)
        logger.info("Testmatrix erstellt")
        
        # Matrix zu MLX konvertieren
        if HAS_MLX:
            logger.info("Konvertiere zu MLX...")
            matrix_mlx = mx.array(matrix)
            logger.info("Zu MLX konvertiert")
        else:
            matrix_mlx = matrix
    except Exception as e:
        logger.error(f"Fehler beim Erstellen der Matrix: {e}")
        return
    
    # SVD testen
    try:
        logger.info("Teste originale SVD...")
        start_time = time.time()
        u1, s1, v1 = original_backend.svd(matrix_mlx)
        original_time = time.time() - start_time
        logger.info(f"Originale SVD: {original_time:.6f} s")
        
        logger.info("Teste optimierte SVD...")
        start_time = time.time()
        u2, s2, v2 = optimized_backend.svd(matrix_mlx)
        optimized_time = time.time() - start_time
        logger.info(f"Optimierte SVD: {optimized_time:.6f} s")
        
        # Ergebnisse vergleichen
        if HAS_MLX:
            u1_np = u1.tolist()
            u2_np = u2.tolist()
            s1_np = s1.tolist()
            s2_np = s2.tolist()
            v1_np = v1.tolist()
            v2_np = v2.tolist()
        else:
            u1_np = u1
            u2_np = u2
            s1_np = s1
            s2_np = s2
            v1_np = v1
            v2_np = v2
        
        logger.info("Vergleiche Ergebnisse...")
        u_diff = np.mean(np.abs(np.array(u1_np) - np.array(u2_np)))
        s_diff = np.mean(np.abs(np.array(s1_np) - np.array(s2_np)))
        v_diff = np.mean(np.abs(np.array(v1_np) - np.array(v2_np)))
        
        logger.info(f"U-Differenz: {u_diff}")
        logger.info(f"S-Differenz: {s_diff}")
        logger.info(f"V-Differenz: {v_diff}")
        
        # Verbesserung berechnen
        if original_time > 0:
            improvement = (original_time - optimized_time) / original_time * 100
            logger.info(f"Verbesserung: {improvement:.2f}%")
        
        # Detailliertere Statistik
        if hasattr(optimized_backend, "get_svd_stats"):
            logger.info("SVD-Statistik:")
            stats = optimized_backend.get_svd_stats()
            for key, value in stats.items():
                logger.info(f"{key}: {value}")
    except Exception as e:
        logger.error(f"Fehler beim SVD-Test: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starte Test-Skript...")
    test_optimization()
    logger.info("Test-Skript beendet")
