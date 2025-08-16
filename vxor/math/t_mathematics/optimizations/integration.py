#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimierungs-Integration für T-Mathematics Engine

Diese Datei stellt den Haupteinstiegspunkt für MLX-Optimierungen dar und
ermöglicht die Integration in die bestehende T-Mathematics Engine.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import importlib
import inspect
from typing import Dict, List, Any, Optional, Union, Callable
from functools import wraps

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.t_mathematics.optimizations.integration")

# Globale Konfiguration
DEFAULT_OPTIMIZATION_LEVEL = 2  # 0=keine, 1=minimal, 2=standard, 3=aggressiv
CONFIG = {
    "optimization_level": DEFAULT_OPTIMIZATION_LEVEL,
    "auto_jit": True,
    "memory_strategy": "auto",
    "enable_telemetry": True
}

# Verfügbare Optimierungsmodule
OPTIMIZATION_MODULES = {
    "mlx_backend_enhancer": {
        "available": False,
        "module": None,
        "description": "Verbessert das MLXBackend mit optimierten Operationen"
    },
    "optimized_mlx_svd": {
        "available": False,
        "module": None,
        "description": "Bietet eine optimierte SVD-Implementierung"
    },
    "hybrid_svd": {
        "available": False,
        "module": None, 
        "description": "Adaptive SVD-Strategie basierend auf Matrix-Eigenschaften"
    }
}

def initialize_optimization_modules():
    """
    Initialisiert die verfügbaren Optimierungsmodule
    """
    # Definiere mögliche Pfade für die Module
    module_search_paths = [
        # Direkter Pfad
        "{module_name}",
        # Relativer Pfad im Optimierungspaket
        "miso.math.t_mathematics.optimizations.{module_name}",
        # Pfad im Optimierungsverzeichnis
        "optimization.phase2_backend.2_2_speicheroptimierung.{module_name}"
    ]
    
    # Versuche, jedes Modul zu laden
    for module_name in OPTIMIZATION_MODULES:
        for path_template in module_search_paths:
            try:
                path = path_template.format(module_name=module_name)
                module = importlib.import_module(path)
                OPTIMIZATION_MODULES[module_name]["available"] = True
                OPTIMIZATION_MODULES[module_name]["module"] = module
                logger.info(f"Optimierungsmodul '{module_name}' erfolgreich geladen")
                break
            except ImportError:
                continue
        
        if not OPTIMIZATION_MODULES[module_name]["available"]:
            # Versuche, die Datei direkt aus dem Skriptverzeichnis zu laden
            try:
                # Aktuelles Verzeichnis zum Pfad hinzufügen
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                
                # Lade das Modul direkt
                module = importlib.import_module(module_name)
                OPTIMIZATION_MODULES[module_name]["available"] = True
                OPTIMIZATION_MODULES[module_name]["module"] = module
                logger.info(f"Optimierungsmodul '{module_name}' erfolgreich geladen aus Skriptverzeichnis")
            except ImportError:
                logger.warning(f"Optimierungsmodul '{module_name}' konnte nicht geladen werden")

def get_available_optimizations():
    """
    Gibt Informationen zu den verfügbaren Optimierungen zurück
    
    Returns:
        Dictionary mit Informationen zu verfügbaren Optimierungen
    """
    return {
        name: {
            "available": info["available"],
            "description": info["description"]
        } for name, info in OPTIMIZATION_MODULES.items()
    }

def optimize_mlx_backend(mlx_backend, optimization_level=None):
    """
    Optimiert ein MLXBackend mit den verfügbaren Optimierungen
    
    Args:
        mlx_backend: Instanz des MLXBackend
        optimization_level: Optimierungsstufe (0=keine, 1=minimal, 2=standard, 3=aggressiv)
        
    Returns:
        Optimiertes MLXBackend
    """
    # Verwende globale Konfiguration, wenn kein Optimierungslevel angegeben
    if optimization_level is None:
        optimization_level = CONFIG["optimization_level"]
    
    if optimization_level == 0:
        logger.info("Optimierungen deaktiviert (optimization_level=0)")
        return mlx_backend
    
    # Initialisiere Module, falls noch nicht geschehen
    if not any(info["available"] for info in OPTIMIZATION_MODULES.values()):
        initialize_optimization_modules()
    
    # Prüfe, ob Optimierungen verfügbar sind
    if not any(info["available"] for info in OPTIMIZATION_MODULES.values()):
        logger.warning("Keine Optimierungsmodule verfügbar")
        return mlx_backend
    
    logger.info(f"Optimiere MLXBackend mit Optimierungsstufe {optimization_level}")
    
    # Speichere eine Kopie des originalen Backends für hybride Strategien
    original_backend = mlx_backend
    optimized_backend = mlx_backend  # Wird durch Optimierungen modifiziert
    
    # 1. Backend-Enhancer anwenden, wenn verfügbar
    if OPTIMIZATION_MODULES["mlx_backend_enhancer"]["available"]:
        try:
            enhancer_module = OPTIMIZATION_MODULES["mlx_backend_enhancer"]["module"]
            optimize_function = getattr(enhancer_module, "optimize_mlx_backend")
            optimized_backend = optimize_function(optimized_backend)
            logger.info("Backend-Enhancer erfolgreich angewendet")
        except Exception as e:
            logger.error(f"Fehler beim Anwenden des Backend-Enhancers: {e}")
    
    # 2. SVD-Optimierung anwenden, wenn verfügbar und Level ≥ 2
    if optimization_level >= 2 and OPTIMIZATION_MODULES["optimized_mlx_svd"]["available"]:
        try:
            svd_module = OPTIMIZATION_MODULES["optimized_mlx_svd"]["module"]
            install_function = getattr(svd_module, "install_svd_optimizer")
            optimized_backend = install_function(optimized_backend)
            logger.info("SVD-Optimizer erfolgreich angewendet")
        except Exception as e:
            logger.error(f"Fehler beim Anwenden des SVD-Optimizers: {e}")
    
    # 3. Hybride SVD anwenden, wenn verfügbar und Level = 3
    if optimization_level >= 3 and OPTIMIZATION_MODULES["hybrid_svd"]["available"]:
        try:
            hybrid_module = OPTIMIZATION_MODULES["hybrid_svd"]["module"]
            install_function = getattr(hybrid_module, "install_hybrid_svd")
            optimized_backend = install_function(original_backend, optimized_backend)
            logger.info("Hybride SVD erfolgreich angewendet")
        except Exception as e:
            logger.error(f"Fehler beim Anwenden der hybriden SVD: {e}")
    
    return optimized_backend

def patch_mlx_support():
    """
    Patcht die MLXBackend-Klasse für automatische Optimierungen
    
    Returns:
        True, wenn erfolgreich, sonst False
    """
    try:
        # Lade die T-Mathematics Engine
        from miso.math.t_mathematics.mlx_support import MLXBackend
        
        # Speichere die originale __init__ Methode
        original_init = MLXBackend.__init__
        
        # Definiere eine neue __init__ Methode mit Optimierungen
        @wraps(original_init)
        def optimized_init(self, precision="float16"):
            # Rufe die originale __init__ Methode auf
            original_init(self, precision)
            
            # Optimiere das Backend
            optimize_mlx_backend(self)
        
        # Ersetze die __init__ Methode
        MLXBackend.__init__ = optimized_init
        
        logger.info("MLX-Support erfolgreich gepatcht für automatische Optimierungen")
        return True
    
    except Exception as e:
        logger.error(f"Konnte MLX-Support nicht patchen: {e}")
        return False

def configure_optimizations(optimization_level=None, auto_jit=None, 
                           memory_strategy=None, enable_telemetry=None):
    """
    Konfiguriert die Optimierungseinstellungen
    
    Args:
        optimization_level: Optimierungsstufe (0=keine, 1=minimal, 2=standard, 3=aggressiv)
        auto_jit: Automatisch JIT-Kompilierung verwenden
        memory_strategy: Speicherverwaltungsstrategie ('auto', 'conservative', 'aggressive')
        enable_telemetry: Telemetrie aktivieren
        
    Returns:
        Aktualisierte Konfiguration
    """
    global CONFIG
    
    if optimization_level is not None:
        CONFIG["optimization_level"] = max(0, min(int(optimization_level), 3))
    
    if auto_jit is not None:
        CONFIG["auto_jit"] = bool(auto_jit)
    
    if memory_strategy is not None:
        valid_strategies = ['auto', 'conservative', 'aggressive']
        if memory_strategy in valid_strategies:
            CONFIG["memory_strategy"] = memory_strategy
        else:
            logger.warning(f"Ungültige Speicherstrategie: {memory_strategy}. Gültige Werte: {valid_strategies}")
    
    if enable_telemetry is not None:
        CONFIG["enable_telemetry"] = bool(enable_telemetry)
    
    logger.info(f"Optimierungskonfiguration aktualisiert: {CONFIG}")
    return CONFIG

# Initialisiere die Module automatisch
initialize_optimization_modules()
