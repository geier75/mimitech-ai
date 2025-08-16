#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Core Modul

Dieses Modul implementiert die KI-native Programmiersprache M-CODE für das MISO-System.
M-CODE ist eine Hochleistungs-Programmiersprache, die speziell für KI-Operationen optimiert ist.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

# Importiere wichtige Komponenten
from .compiler import MCodeCompiler, compile_m_code
from .interpreter import MCodeInterpreter, execute_m_code
from .syntax import MCodeSyntaxTree, parse_m_code
from .optimizer import MCodeOptimizer, optimize_m_code
from .runtime import MCodeRuntime, MCodeFunction, MCodeObject
from .stdlib import get_stdlib_functions, register_function

# Exportiere wichtige Schnittstellen
__all__ = [
    # Hauptklassen
    'MCodeCompiler',
    'MCodeInterpreter',
    'MCodeSyntaxTree',
    'MCodeOptimizer',
    'MCodeRuntime',
    'MCodeFunction',
    'MCodeObject',
    
    # Hauptfunktionen
    'compile_m_code',
    'execute_m_code',
    'parse_m_code',
    'optimize_m_code',
    'get_stdlib_functions',
    'register_function',
    
    # Hilfsfunktionen
    'initialize_m_code',
    'get_runtime',
    'reset_runtime'
]

# Versionsinformation
__version__ = "2025.1.0"

import os
import logging
from typing import Dict, Any, Optional, List, Callable

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code")

# Globale Runtime-Instanz
_DEFAULT_RUNTIME = None


def initialize_m_code(
    optimization_level: int = 2,
    use_jit: bool = True,
    memory_limit_mb: int = 1024,
    enable_extensions: bool = True
) -> Dict[str, Any]:
    """
    Initialisiert die M-CODE Runtime für optimale Ausführung.
    
    Args:
        optimization_level: Optimierungsstufe (0-3)
        use_jit: Just-In-Time Compilation aktivieren
        memory_limit_mb: Speicherlimit in MB
        enable_extensions: Erweiterungen aktivieren
        
    Returns:
        Runtime-Konfiguration und Status
    """
    from .runtime import MCodeRuntime
    
    # Konfiguration
    config = {
        "optimization_level": min(3, max(0, optimization_level)),
        "use_jit": use_jit,
        "memory_limit_mb": memory_limit_mb,
        "enable_extensions": enable_extensions,
        "initialized": True
    }
    
    # Setze Umgebungsvariablen
    os.environ["M_CODE_OPTIMIZATION"] = str(optimization_level)
    os.environ["M_CODE_USE_JIT"] = "1" if use_jit else "0"
    os.environ["M_CODE_MEMORY_LIMIT"] = str(memory_limit_mb)
    
    # Initialisiere Runtime
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = MCodeRuntime(config)
    
    logger.info(f"M-CODE Runtime initialisiert mit Optimierungsstufe {optimization_level}")
    
    return config


def get_runtime() -> 'MCodeRuntime':
    """
    Gibt die aktuelle Runtime-Instanz zurück oder erstellt eine neue,
    falls noch keine existiert.
    
    Returns:
        MCodeRuntime-Instanz
    """
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        initialize_m_code()
    return _DEFAULT_RUNTIME


def reset_runtime() -> None:
    """
    Setzt die globale Runtime-Instanz zurück.
    """
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is not None:
        _DEFAULT_RUNTIME.shutdown()
    _DEFAULT_RUNTIME = None
