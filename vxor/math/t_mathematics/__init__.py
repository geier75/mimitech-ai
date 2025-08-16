#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - T-Mathematics Engine Modul

Dieses Modul stellt erweiterte Tensor-Mathematik-Operationen für das MISO-System bereit.
Optimiert für Apple Silicon M4 Max und AMD RDNA3 Hardware durch spezialisierte Algorithmen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

# Importiere wichtige Komponenten
from .engine import TMathEngine
from .compat import TMathConfig, TMathematicsEngine
from .ops import (
    tensor_svd, 
    amd_optimized_matmul, 
    moe_routing, 
    mix_experts_outputs, 
    positional_encoding,
    attention_wrapper
)
from .models import (
    OptimizedMultiHeadAttention,
    OptimizedFeedForward,
    MixtureOfExperts,
    TMathTransformerLayer,
    Expert
)

# Exportiere wichtige Schnittstellen
__all__ = [
    # Hauptklassen
    'TMathEngine',
    'TMathematicsEngine',
    'TMathConfig',
    'get_engine',
    'reset_engine',
    'get_default_config',
    
    # Mathematische Operationen
    'tensor_svd',
    'amd_optimized_matmul',
    'moe_routing',
    'mix_experts_outputs',
    'positional_encoding',
    'attention_wrapper',
    
    # Modellklassen
    'OptimizedMultiHeadAttention',
    'OptimizedFeedForward',
    'MixtureOfExperts',
    'TMathTransformerLayer',
    'Expert',
    
    # Hilfsfunktionen
    'initialize_engine'
]

# Versionsinformation
__version__ = "2025.1.0"

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Union

# Konfiguriere Logging
logger = logging.getLogger("MISO.t_mathematics")

def initialize_engine(
    dimensions: int = 8192,
    precision: str = "float16",
    use_matrix_cores: bool = True,
    optimization_level: int = 3
) -> Dict[str, Any]:
    """
    Initialisiert die T-Mathematics Engine für optimale Datenverarbeitung.
    
    Args:
        dimensions: Anzahl der mathematischen Dimensionen für Verarbeitung
        precision: Präzision (float16, bfloat16, float32)
        use_matrix_cores: Aktiviert Matrix-Kerne für beschleunigte Berechnungen
        optimization_level: Optimierungsstufe (1-3)
        
    Returns:
        Engine-Konfiguration und Status
    """
    # Prüfe auf GPU-Verfügbarkeit
    has_gpu = _check_for_gpu()
    device = "gpu" if has_gpu else "cpu"
    
    # Optimale Präzision
    if precision not in ["float16", "bfloat16", "float32", "mixed"]:
        logger.warning(f"Unbekannte Präzision {precision}, verwende 'mixed'")
        precision = "mixed"
    
    # Konfiguration
    config = {
        "dimensions": dimensions,
        "precision": precision,
        "device": device,
        "use_matrix_cores": use_matrix_cores and has_gpu,
        "optimization_level": min(3, max(1, optimization_level)),
        "initialized": True,
        "temp_folding_active": optimization_level >= 3,
        "quantum_estimation": optimization_level >= 2
    }
    
    # Setze Umgebungsvariablen für optimale Performance
    if has_gpu:
        os.environ["T_MATH_PRECISION"] = precision
        os.environ["T_MATH_GPU_CORES"] = "ALL"
        os.environ["T_MATH_CACHE_SIZE"] = str(dimensions * 16)
    
    logger.info(f"T-Mathematics Engine initialisiert auf {device} mit {precision} Präzision")
    print("T-Mathematics Engine geladen")
    
    return config

def _check_for_gpu() -> bool:
    """Prüft, ob eine GPU verfügbar ist."""
    try:
        # Versuche, GPU über direkten Hardware-Zugriff zu erkennen
        import torch
        if torch.cuda.is_available():
            return True
        
        # Prüfe auf Apple Silicon
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            return True
        
        # Prüfe auf AMD ROCm
        if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return True
            
        # Prüfe auf DirectML
        try:
            import onnxruntime as ort
            if 'DirectML' in ort.get_available_providers():
                return True
        except ImportError:
            pass
            
        return False
    except:
        logger.warning("Keine GPU erkannt, verwende CPU")
        return False


# Globale Engine-Instanz
_DEFAULT_ENGINE = None


def get_engine() -> TMathEngine:
    """
    Gibt die aktuelle Engine-Instanz zurück oder erstellt eine neue,
    falls noch keine existiert.
    
    Returns:
        TMathEngine-Instanz
    """
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = TMathEngine()
    return _DEFAULT_ENGINE


def reset_engine() -> None:
    """
    Setzt die globale Engine-Instanz zurück.
    """
    global _DEFAULT_ENGINE
    _DEFAULT_ENGINE = None


def get_default_config() -> TMathConfig:
    """
    Gibt eine Standardkonfiguration für die T-Mathematics Engine zurück.
    
    Returns:
        TMathConfig-Instanz mit Standardwerten
    """
    return TMathConfig(
        precision="mixed",
        device="auto",
        use_fused_ops=True,
        optimize_for_rdna=True,
        optimize_for_apple_silicon=True,
        use_flash_attention=True
    )
