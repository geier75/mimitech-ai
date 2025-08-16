#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION MLX Kernel Warmup

Pre-Initialisierung von MLX-Kernels für Apple Silicon.
Führt einen Warmup der Matrix-Operationen durch, um den Metal-Kernel
zu cachen und "Metal Binding Freeze" zu verhindern.
"""

import os
import sys
import logging
import time

# Konfiguration des Loggers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VX-VISION.mlx_warmup")

# Füge das Root-Verzeichnis zum Pythonpfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Importiere den zentralen MLX-Initialisierer
from vxor.core.mlx_core import init_mlx, get_mlx_status

# MLX importieren nach der Initialisierung
try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    logger.warning("MLX konnte nicht importiert werden. Warmup wird übersprungen.")

def run_warmup(verbose=True):
    """
    Führt einen Warmup der MLX-Kernels durch.
    
    Args:
        verbose (bool): Wenn True, werden ausführliche Logs ausgegeben.
    
    Returns:
        dict: Ergebnisse des Warmups
    """
    if not HAS_MLX:
        logger.error("MLX nicht verfügbar, Warmup nicht möglich.")
        return {"success": False, "reason": "MLX nicht verfügbar"}
    
    # Initialisiere MLX
    init_success = init_mlx(force_cpu=False, disable_jit=False)
    if not init_success or not HAS_MLX:
        logger.error("MLX-Initialisierung fehlgeschlagen, Warmup nicht möglich.")
        return {"success": False, "reason": "MLX-Initialisierung fehlgeschlagen"}
    
    results = {}
    
    try:
        if verbose:
            logger.info("⚡ Starte MLX Warmup...")
            logger.info(f"MLX Version: {mlx.__version__ if hasattr(mlx, '__version__') else 'unbekannt'}")
        
        # Timer starten
        start_time = time.time()
        
        # Warmup 1: Matrix-Multiplikation (grundlegend für viele Operationen)
        if verbose:
            logger.info("🔢 Führe Matrix-Multiplikation durch...")
        x = mx.random.normal(shape=(128, 128))
        y = mx.random.normal(shape=(128, 128))
        z = mx.matmul(x, y)
        # Force evaluation
        mx.eval(z)
        logger.debug("✅ Matrix-Multiplikation erfolgreich ausgeführt")
        
        # Warmup 2: Konvolutionsoperationen (wichtig für Bildverarbeitung)
        if verbose:
            logger.info("⚡ Warmup für Konvolutionskernels...")
        
        # Simuliere ein kleines Bild (64x64 mit 3 Kanälen)
        img = mx.random.normal(shape=(1, 3, 64, 64))
        
        # Einfacher Konvolutionsfilter
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        output = conv(img)
        # Force evaluation
        mx.eval(output)
        logger.debug("Konvolutionsoperation erfolgreich ausgeführt")
        
        # Warmup 3: Resize-Operation (wichtig für Bildgrößenänderungen)
        if verbose:
            logger.info("⚡ Warmup für Resize-Kernels...")
        
        # Reshape zu größerem Format
        img_reshaped = mx.reshape(output, (1, 16, 128, 128))
        mx.eval(img_reshaped)
        logger.debug("Resize-Operation erfolgreich ausgeführt")
        
        # Timer beenden
        end_time = time.time()
        duration = end_time - start_time
        
        if verbose:
            logger.info(f"✅ MLX Kernel Warmup abgeschlossen in {duration:.4f}s")
            logger.info(f"MLX Status: {get_mlx_status()}")
        
        results = {
            "success": True,
            "duration": duration,
            "operations": ["matmul", "conv2d", "reshape"],
            "mlx_status": get_mlx_status()
        }
        
    except Exception as e:
        logger.error(f"Fehler während des Warmups: {e}")
        import traceback
        logger.error(traceback.format_exc())
        results = {"success": False, "reason": str(e)}
    
    return results

if __name__ == "__main__":
    # Führe den Warmup aus
    run_warmup(verbose=True)
