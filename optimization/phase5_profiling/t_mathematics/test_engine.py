#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate: Einfaches Testskript für die T-Mathematics Engine
Überprüft die grundlegende Funktionalität der Engine.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Konfiguriere Pfade für Imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, "../../.."))
math_dir = os.path.join(project_dir, "math_module", "t_mathematics")

# Füge Pfade zum Suchpfad hinzu
sys.path.insert(0, project_dir)
sys.path.insert(0, math_dir)

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_engine_original():
    """Testet die originale Engine-Implementierung."""
    logger.info("Teste originale Engine...")
    
    try:
        # Versuche, die originale Engine zu importieren
        from engine import Engine
        
        # Initialisiere Engine
        engine = Engine(precision="float32", backend="auto")
        logger.info(f"Engine erfolgreich initialisiert: {engine}")
        
        # Teste Operationen
        result = engine.compute("add", [1, 2], [3, 4])
        logger.info(f"Ergebnis von add: {result}")
        
        tensor = engine.create_tensor((2, 2))
        logger.info(f"Erstellter Tensor: {tensor}")
        
        return True
    except Exception as e:
        logger.error(f"Fehler beim Testen der originalen Engine: {e}")
        return False

def test_engine_optimized():
    """Testet die optimierte Engine-Implementierung."""
    logger.info("Teste optimierte Engine...")
    
    try:
        # Versuche, die optimierte Engine zu importieren
        sys.path.insert(0, math_dir)  # Stelle sicher, dass wir im richtigen Verzeichnis sind
        
        # Lokales Import der backend_base.py
        import backend_base
        logger.info(f"Backend-Basis erfolgreich importiert: {backend_base}")
        
        # Importiere engine_optimized
        import engine_optimized
        logger.info(f"Optimierte Engine erfolgreich importiert: {engine_optimized}")
        
        # Initialisiere Engine
        engine = engine_optimized.get_engine(precision="float32", backend="auto")
        logger.info(f"Optimierte Engine erfolgreich initialisiert: {engine}")
        
        # Zeige Backend-Informationen
        backend_info = engine.get_active_backend_info()
        logger.info(f"Aktives Backend: {backend_info}")
        
        # Teste, ob MLX verfügbar ist
        try:
            import mlx
            logger.info(f"MLX verfügbar: Version {getattr(mlx, '__version__', 'unbekannt')}")
        except ImportError:
            logger.warning("MLX nicht verfügbar. Optimierungen könnten eingeschränkt sein.")
        
        return True
    except Exception as e:
        logger.error(f"Fehler beim Testen der optimierten Engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Hauptfunktion."""
    logger.info("Starte Tests für T-Mathematics Engine...")
    
    # Teste originale Engine
    orig_result = test_engine_original()
    logger.info(f"Test der originalen Engine: {'Erfolgreich' if orig_result else 'Fehlgeschlagen'}")
    
    # Teste optimierte Engine
    opt_result = test_engine_optimized()
    logger.info(f"Test der optimierten Engine: {'Erfolgreich' if opt_result else 'Fehlgeschlagen'}")
    
    return 0 if orig_result or opt_result else 1

if __name__ == "__main__":
    sys.exit(main())
