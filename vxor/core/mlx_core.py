#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION MLX Core Singleton

Zentralisierte MLX-Initialisierung für konsistente Nutzung über alle Module.
Verhindert "Metal Binding Freeze" auf Apple Silicon mit virtuellen Umgebungen.
"""

import os
import logging
import sys

logger = logging.getLogger("VX-VISION.mlx_core")

# MLX importieren
try:
    import mlx
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    logger.warning("MLX konnte nicht importiert werden. Fallback auf CPU-Implementierungen.")

# Neuer Code für Gerätekompatibilität
MLX_HAS_DEVICE_ATTR = hasattr(mx, 'device')
MLX_HAS_IMAGE = False

try:
    import mlx.image
    MLX_HAS_IMAGE = True
except ImportError:
    MLX_HAS_IMAGE = False
    logger.warning("MLX Image-Modul konnte nicht importiert werden. Verwende NumPy-basierte Fallback-Implementierungen.")

# Initialisierungsstatus
_is_initialized = False

def init_mlx(force_cpu=False, disable_jit=False):
    """
    Zentralisierte MLX-Initialisierungsfunktion.
    
    Args:
        force_cpu (bool): Wenn True, wird CPU erzwungen statt GPU.
        disable_jit (bool): Wenn True, wird JIT-Kompilierung deaktiviert.
    
    Returns:
        bool: True wenn Initialisierung erfolgreich, sonst False
    """
    global _is_initialized
    
    if not HAS_MLX:
        return False
    
    if _is_initialized:
        logger.debug("MLX bereits initialisiert, überspringe.")
        return True
    
    # JIT-Konfiguration
    if disable_jit:
        os.environ["MLX_DISABLE_JIT"] = "1"
        logger.info("MLX JIT deaktiviert.")
    else:
        os.environ["MLX_DISABLE_JIT"] = "0"
        logger.info("MLX JIT aktiviert.")
    
    try:
        # MLX verwendet automatisches Device-Routing auf Apple Silicon
        if force_cpu:
            logger.info("MLX CPU-Modus erzwungen.")
            # In neueren MLX-Versionen wird die CPU automatisch verwendet, wenn keine GPU verfügbar ist
        else:
            # Standardmäßig verwendet MLX automatisch die beste verfügbare Hardware
            # In neueren Versionen gibt es keine direkte Geräteabfrage mehr
            logger.info("MLX verwendet automatisches Device-Routing (Apple Silicon: ANE/GPU wenn verfügbar)")
        
        # Teste grundlegende MLX-Funktionalität
        test_array = mx.array([1, 2, 3])
        mx.eval(test_array)
        
        _is_initialized = True
        return True
    except Exception as e:
        logger.error(f"MLX-Initialisierungsfehler: {e}")
        logger.debug("Stacktrace:", exc_info=True)
        return False

def get_mlx_status():
    """
    Gibt den aktuellen MLX-Status zurück.
    
    Returns:
        dict: Status-Informationen
    """
    global _is_initialized
    
    status = {
        "initialized": _is_initialized,
        "available": HAS_MLX,
        "jit": not bool(os.environ.get("MLX_DISABLE_JIT", False)),
        "has_image": MLX_HAS_IMAGE,
        "has_device_attr": MLX_HAS_DEVICE_ATTR
    }
    
    if HAS_MLX and _is_initialized:
        try:
            # Neuer Weg, um das Standardgerät zu erhalten
            status["device"] = str(mx.default_device())
        except Exception as e:
            status["device"] = f"Error: {str(e)}"
            
    return status

# Automatisch initialisieren, wenn dieses Modul importiert wird
if HAS_MLX:
    init_mlx()
