#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MATRIX: Hochoptimierte Tensor-Operation-Bridge für VXOR-MISO Integration

Dieses Modul stellt eine leistungsstarke Brücke zwischen VXOR-Komponenten 
und der T-Mathematics Engine dar, mit spezieller Optimierung für:
- MLX (Apple Silicon)
- PyTorch (mit MPS/Metal Acceleration)
- NumPy (CPU-Fallback)
- JAX (experimentelle Unterstützung)

Kernfunktionen:
- Nahtlose Konvertierung zwischen verschiedenen Tensor-Typen
- Hardware-spezifische Optimierungen (Apple Silicon, CUDA, CPU)
- Multi-dimensionale Matrix-Operationen für VX-Komponenten
- ZTM-konforme Tensor-Manipulation mit Audit-Trail

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
ZTM-Level: STRICT
Version: 1.0.0 (04.05.2025)
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# ZTM-Protokoll-Initialisierung
ZTM_ACTIVE = os.environ.get('MISO_ZTM_MODE', '1') == '1'
ZTM_LOG_LEVEL = os.environ.get('MISO_ZTM_LOG_LEVEL', 'INFO')

# Logger-Konfiguration
logger = logging.getLogger("VXOR.VX-MATRIX")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(getattr(logging, ZTM_LOG_LEVEL))

# Definiere ZTM-Logging-Funktion
def ztm_log(message: str, level: str = 'INFO', module: str = 'VX-MATRIX'):
    """ZTM-konforme Logging-Funktion mit Audit-Trail"""
    if not ZTM_ACTIVE:
        return
    
    log_func = getattr(logger, level.lower())
    log_func(f"[ZTM:{module}] {message}")

# Pfadinformationen
CURRENT_DIR = Path(__file__).parent.absolute()
VXOR_AI_DIR = Path(__file__).parent.parent.absolute()
MISO_ROOT = VXOR_AI_DIR.parent / "miso"

# Füge Pfade zum Pythonpfad hinzu für Importe
sys.path.insert(0, str(VXOR_AI_DIR))
sys.path.insert(0, str(MISO_ROOT))

# Importiere Kern-Module
try:
    from .core.matrix_core import MatrixCore, TensorType
    from .core.tensor_bridge import TensorBridge, ConversionMode
    from .adapters.t_mathematics_adapter import TMathAdapter
    from .optimizers.mlx_optimizer import MLXOptimizer
    
    # Registriere Module als geladen
    ztm_log("VX-MATRIX Kernkomponenten erfolgreich geladen", level="INFO")
    CORE_MODULES_LOADED = True
except ImportError as e:
    ztm_log(f"Fehler beim Laden der VX-MATRIX Kernkomponenten: {e}", level="ERROR")
    CORE_MODULES_LOADED = False

# Versuche, Abhängigkeiten zu laden
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    ztm_log("NumPy nicht verfügbar. CPU-Fallback eingeschränkt.", level="WARNING")

try:
    import torch
    TORCH_AVAILABLE = True
    # Prüfe, ob MPS (Metal Performance Shaders) verfügbar ist
    try:
        MPS_AVAILABLE = torch.backends.mps.is_available()
    except:
        MPS_AVAILABLE = False
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    ztm_log("PyTorch nicht verfügbar. GPU-Beschleunigung eingeschränkt.", level="WARNING")

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    ztm_log("MLX nicht verfügbar. Apple Silicon Optimierung deaktiviert.", level="WARNING")

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = True
    ztm_log("JAX nicht verfügbar. Experimentelle Funktionen deaktiviert.", level="WARNING")

# Konfiguration der Hardware-Beschleunigung
HW_CONFIG = {
    "apple_silicon": MLX_AVAILABLE,
    "cuda": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
    "mps": MPS_AVAILABLE,
    "jax": JAX_AVAILABLE,
    "cpu_only": not any([MLX_AVAILABLE, MPS_AVAILABLE, 
                        (TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False)])
}

# Wähle den besten verfügbaren Backend
if HW_CONFIG["apple_silicon"]:
    PREFERRED_BACKEND = "mlx"
elif HW_CONFIG["cuda"]:
    PREFERRED_BACKEND = "cuda"
elif HW_CONFIG["mps"]:
    PREFERRED_BACKEND = "mps"
elif HW_CONFIG["jax"]:
    PREFERRED_BACKEND = "jax"
else:
    PREFERRED_BACKEND = "cpu"

ztm_log(f"VX-MATRIX initialisiert mit Preferred-Backend: {PREFERRED_BACKEND}", level="INFO")

# Funktionen für Backend-Erkennung
def get_preferred_backend() -> str:
    """Gibt den bevorzugten Backend-Typ zurück"""
    return PREFERRED_BACKEND

def set_preferred_backend(backend: str) -> bool:
    """
    Setzt den bevorzugten Backend-Typ
    
    Args:
        backend: 'mlx', 'cuda', 'mps', 'jax' oder 'cpu'
        
    Returns:
        Erfolgsstatus
    """
    global PREFERRED_BACKEND
    
    valid_backends = ['mlx', 'cuda', 'mps', 'jax', 'cpu']
    if backend not in valid_backends:
        ztm_log(f"Ungültiger Backend-Typ: {backend}. Erlaubte Werte: {valid_backends}", level="ERROR")
        return False
        
    # Prüfe, ob das gewählte Backend verfügbar ist
    if backend == 'mlx' and not HW_CONFIG["apple_silicon"]:
        ztm_log("MLX Backend angefordert, aber Apple Silicon nicht verfügbar. Fallback wird verwendet.", level="WARNING")
    elif backend == 'cuda' and not HW_CONFIG["cuda"]:
        ztm_log("CUDA Backend angefordert, aber CUDA nicht verfügbar. Fallback wird verwendet.", level="WARNING")
    elif backend == 'mps' and not HW_CONFIG["mps"]:
        ztm_log("MPS Backend angefordert, aber Metal nicht verfügbar. Fallback wird verwendet.", level="WARNING")
    elif backend == 'jax' and not HW_CONFIG["jax"]:
        ztm_log("JAX Backend angefordert, aber JAX nicht verfügbar. Fallback wird verwendet.", level="WARNING")
    
    PREFERRED_BACKEND = backend
    ztm_log(f"Preferred-Backend aktualisiert auf: {PREFERRED_BACKEND}", level="INFO")
    return True

# Exportiere Hauptklassen und Funktionen
__all__ = [
    'MatrixCore', 
    'TensorBridge', 
    'TMathAdapter', 
    'MLXOptimizer',
    'TensorType', 
    'ConversionMode',
    'get_preferred_backend',
    'set_preferred_backend',
    'get_matrix_core',
    'get_tensor_bridge',
    'get_t_math_adapter',
    'convert_tensor',
    'optimize_for_hardware'
]

# Singleton-Instanzen der Hauptklassen
_matrix_core = None
_tensor_bridge = None
_t_math_adapter = None
_mlx_optimizer = None

def get_matrix_core() -> 'MatrixCore':
    """
    Gibt die Singleton-Instanz des MatrixCore zurück
    
    Returns:
        MatrixCore-Instanz
    """
    global _matrix_core
    if _matrix_core is None and CORE_MODULES_LOADED:
        _matrix_core = MatrixCore(preferred_backend=PREFERRED_BACKEND)
    return _matrix_core

def get_tensor_bridge() -> 'TensorBridge':
    """
    Gibt die Singleton-Instanz der TensorBridge zurück
    
    Returns:
        TensorBridge-Instanz
    """
    global _tensor_bridge
    if _tensor_bridge is None and CORE_MODULES_LOADED:
        _tensor_bridge = TensorBridge(preferred_backend=PREFERRED_BACKEND)
    return _tensor_bridge

def get_t_math_adapter() -> 'TMathAdapter':
    """
    Gibt die Singleton-Instanz des TMathAdapter zurück
    
    Returns:
        TMathAdapter-Instanz
    """
    global _t_math_adapter
    if _t_math_adapter is None and CORE_MODULES_LOADED:
        _t_math_adapter = TMathAdapter()
    return _t_math_adapter

def get_mlx_optimizer() -> 'MLXOptimizer':
    """
    Gibt die Singleton-Instanz des MLXOptimizer zurück
    
    Returns:
        MLXOptimizer-Instanz
    """
    global _mlx_optimizer
    if _mlx_optimizer is None and CORE_MODULES_LOADED and MLX_AVAILABLE:
        _mlx_optimizer = MLXOptimizer()
    return _mlx_optimizer

def convert_tensor(tensor: Any, target_type: str, strict: bool = True) -> Any:
    """
    Konvertiert einen Tensor in den angegebenen Zieltyp
    
    Args:
        tensor: Eingabe-Tensor (NumPy, PyTorch, MLX, JAX)
        target_type: Zieltyp ('numpy', 'torch', 'mlx', 'jax')
        strict: Bei True wird ein Fehler geworfen, wenn die Konvertierung nicht möglich ist
        
    Returns:
        Konvertierter Tensor
    """
    bridge = get_tensor_bridge()
    if bridge is None:
        ztm_log("TensorBridge nicht verfügbar für Konvertierung", level="ERROR")
        return None
        
    return bridge.convert(tensor, target_type=target_type, strict=strict)

def optimize_for_hardware(tensor: Any, operation: str = None) -> Any:
    """
    Optimiert einen Tensor für die verfügbare Hardware
    
    Args:
        tensor: Eingabe-Tensor (beliebiger Typ)
        operation: Optional, spezifische Operation zur Optimierung
        
    Returns:
        Optimierter Tensor im optimalen Format für die aktuelle Hardware
    """
    # Bestimme den Eingangstyp
    input_type = None
    if NUMPY_AVAILABLE and isinstance(tensor, np.ndarray):
        input_type = "numpy"
    elif TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        input_type = "torch"
    elif MLX_AVAILABLE and isinstance(tensor, mx.array):
        input_type = "mlx"
    elif JAX_AVAILABLE and isinstance(tensor, jnp.ndarray):
        input_type = "jax"
    
    if input_type is None:
        ztm_log(f"Unbekannter Tensor-Typ: {type(tensor)}", level="ERROR")
        return tensor
        
    # Wenn bereits im bevorzugten Format, keine Konvertierung notwendig
    if (PREFERRED_BACKEND == "mlx" and input_type == "mlx") or \
       (PREFERRED_BACKEND == "cuda" and input_type == "torch") or \
       (PREFERRED_BACKEND == "mps" and input_type == "torch") or \
       (PREFERRED_BACKEND == "jax" and input_type == "jax") or \
       (PREFERRED_BACKEND == "cpu" and input_type == "numpy"):
        return tensor
        
    # Konvertiere in das bevorzugte Format
    target_type = {
        "mlx": "mlx",
        "cuda": "torch",
        "mps": "torch",
        "jax": "jax",
        "cpu": "numpy"
    }.get(PREFERRED_BACKEND)
    
    return convert_tensor(tensor, target_type=target_type)

# Selbstregistrierung beim Importieren
ztm_log("VX-MATRIX Modul initialisiert", level="INFO")

# Systeminfo ausgeben
if ZTM_ACTIVE:
    backend_info = {
        "preferred_backend": PREFERRED_BACKEND,
        "available_backends": {
            "mlx": MLX_AVAILABLE,
            "torch": TORCH_AVAILABLE,
            "torch_cuda": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
            "torch_mps": MPS_AVAILABLE,
            "numpy": NUMPY_AVAILABLE,
            "jax": JAX_AVAILABLE
        },
        "hardware_config": HW_CONFIG,
        "ztm_active": ZTM_ACTIVE,
        "ztm_log_level": ZTM_LOG_LEVEL
    }
    ztm_log(f"VX-MATRIX System Info: {json.dumps(backend_info, indent=2)}", level="INFO")
