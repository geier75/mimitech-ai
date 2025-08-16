#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLX-Optimierungspaket für T-Mathematics Engine

Dieses Paket enthält Optimierungen für die T-Mathematics Engine mit Fokus auf
MLX-Integration und Apple Silicon Leistungsverbesserungen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import logging
from typing import Dict, List, Any, Optional, Union

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.t_mathematics.optimizations")

# Importiere Kernkomponenten, wenn verfügbar
try:
    from .integration import optimize_mlx_backend, patch_mlx_support
    __all__ = ['optimize_mlx_backend', 'patch_mlx_support']
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False
    logger.warning("MLX-Optimierungen konnten nicht importiert werden.")
    __all__ = []

# Versionsinformation
__version__ = "0.1.0"
__author__ = "MISO Team"

def get_optimization_status() -> Dict[str, Any]:
    """
    Gibt den Status der verfügbaren Optimierungen zurück
    
    Returns:
        Dictionary mit Statusinformationen
    """
    if not OPTIMIZATIONS_AVAILABLE:
        return {
            "available": False,
            "version": __version__,
            "message": "MLX-Optimierungen sind nicht vollständig verfügbar"
        }
    
    from .integration import get_available_optimizations
    
    return {
        "available": True,
        "version": __version__,
        "optimizations": get_available_optimizations()
    }
