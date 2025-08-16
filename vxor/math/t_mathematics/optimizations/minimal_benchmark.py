#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimaler SVD-Benchmark zur Fehlerdiagnose
"""

import os
import sys
import time
import numpy as np
import torch
import logging
import traceback

# Konfiguriere Logging mit detailliertem Format
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("minimal_benchmark")

logger.info("Script gestartet")

# Prüfe auf MLX
try:
    import mlx.core as mx
    HAS_MLX = True
    logger.info("MLX erfolgreich importiert")
except ImportError as e:
    HAS_MLX = False
    mx = None
    logger.error(f"MLX konnte nicht importiert werden: {e}")

# MLX-Backend importieren
try:
    logger.info("Importiere MLXBackend...")
    from miso.math.t_mathematics.mlx_support import MLXBackend
    logger.info("MLXBackend erfolgreich importiert")
except ImportError as e:
    logger.error(f"MLXBackend konnte nicht importiert werden: {e}")
    sys.exit(1)

# Optimierungen importieren
try:
    logger.info("Importiere Optimierungsmodule...")
    from miso.math.t_mathematics.optimizations.integration import optimize_mlx_backend, configure_optimizations
    logger.info("Optimierungsmodule erfolgreich importiert")
except ImportError as e:
    logger.error(f"Optimierungsmodule konnten nicht importiert werden: {e}")
    sys.exit(1)

def test_single_matrix(size=(10, 10), level=0, k=None):
    """
    Testet die SVD für eine einzelne Matrix mit gegebenem Optimierungslevel
    
    Args:
        size: Matrixgröße
        level: Optimierungslevel
        k: k-Wert für partielle SVD (None für vollständige SVD)
    """
    logger.info(f"Teste Matrix {size} mit Level {level}" + (f" und k={k}" if k is not None else ""))
    
    try:
        # Matrix erstellen
        logger.info("Erstelle Matrix...")
        matrix = np.random.rand(*size).astype(np.float32)
        
        # Backend erstellen
        logger.info("Erstelle Backend...")
        backend = MLXBackend(precision="float32")
        
        if level > 0:
            # Optimiere Backend
            logger.info(f"Optimiere Backend mit Level {level}...")
            backend = optimize_mlx_backend(backend, optimization_level=level)
            logger.info("Backend optimiert")
        
        # Konvertiere Matrix zu MLX, wenn verfügbar
        logger.info("Konvertiere Matrix...")
        if HAS_MLX:
            matrix_mlx = mx.array(matrix)
        else:
            matrix_mlx = matrix
        
        # Führe SVD durch
        logger.info("Führe SVD durch...")
        start_time = time.time()
        u, s, v = backend.svd(matrix_mlx, k)
        end_time = time.time()
        
        # Ergebnis validieren
        if u is None or s is None or v is None:
            logger.error("SVD-Ergebnis enthält None-Werte")
            return False
        
        logger.info(f"SVD erfolgreich in {end_time - start_time:.6f} Sekunden")
        
        # Speicher explizit freigeben
        del u, s, v, matrix, matrix_mlx, backend
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            
        return True
    
    except Exception as e:
        logger.error(f"Fehler: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """
    Hauptfunktion für minimalen Benchmark
    """
    logger.info("Starte minimalen SVD-Benchmark...")
    
    # Teste kleine Matrix mit verschiedenen Optimierungsleveln
    success = []
    
    # Level 0 (keine Optimierung)
    logger.info("=== Test mit Level 0 ===")
    success.append(test_single_matrix(size=(10, 10), level=0))
    
    # Level 2 (SVD-Optimierung)
    logger.info("=== Test mit Level 2 ===")
    success.append(test_single_matrix(size=(10, 10), level=2))
    
    # Level 3 (Hybride SVD)
    logger.info("=== Test mit Level 3 ===")
    success.append(test_single_matrix(size=(10, 10), level=3))
    
    # Partielle SVD mit Level 2
    logger.info("=== Test partielle SVD (k=4) mit Level 2 ===")
    success.append(test_single_matrix(size=(10, 10), level=2, k=4))
    
    # Zusammenfassung
    logger.info("=== Zusammenfassung ===")
    logger.info(f"Erfolgreiche Tests: {sum(success)}/{len(success)}")
    
    if all(success):
        logger.info("Alle Tests erfolgreich")
    else:
        logger.warning("Einige Tests sind fehlgeschlagen")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unbehandelte Ausnahme: {e}")
        logger.error(traceback.format_exc())
