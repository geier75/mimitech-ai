#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Einfacher T-MATHEMATICS Test

Dieser Test überprüft die grundlegenden Funktionen der T-MATHEMATICS Engine
ohne komplexe Abhängigkeiten.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import numpy as np
import torch
import time

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.tests.simple_t_math_test")

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_t_mathematics_basic():
    """Testet die grundlegenden Funktionen der T-MATHEMATICS Engine"""
    logger.info("Starte einfachen T-MATHEMATICS Test...")
    
    try:
        # Importiere T-MATHEMATICS-Komponenten
        from miso.math.t_mathematics.integration_manager import get_t_math_integration_manager
        
        # Hole T-MATHEMATICS Integration Manager
        t_math_manager = get_t_math_integration_manager()
        t_math_engine = t_math_manager.get_engine("simple_test")
        
        # Prüfe, ob MLX verfügbar ist
        is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
        has_mlx = hasattr(t_math_engine, 'use_mlx') and t_math_engine.use_mlx
        
        logger.info(f"T-MATHEMATICS Engine initialisiert (Apple Silicon: {is_apple_silicon}, MLX: {has_mlx})")
        
        # Erstelle einfache Tensoren
        tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        tensor2 = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32)
        
        # Bereite Tensoren vor
        tensor1_prepared = t_math_engine.prepare_tensor(tensor1)
        tensor2_prepared = t_math_engine.prepare_tensor(tensor2)
        
        # Führe einfache Operationen durch
        # Addition
        start_time = time.time()
        result_add = tensor1_prepared + tensor2_prepared
        logger.info(f"Addition erfolgreich: {result_add}")
        
        # Multiplikation
        result_mul = tensor1_prepared * tensor2_prepared
        logger.info(f"Multiplikation erfolgreich: {result_mul}")
        
        # Skalarprodukt
        result_dot = torch.dot(tensor1_prepared, tensor2_prepared)
        logger.info(f"Skalarprodukt erfolgreich: {result_dot}")
        
        # Zeitmessung
        end_time = time.time()
        logger.info(f"Operationen in {end_time - start_time:.4f} Sekunden durchgeführt")
        
        # Teste PRISM-Integration, falls verfügbar
        try:
            prism_integration = t_math_manager.get_prism_integration()
            if prism_integration:
                logger.info("PRISM-Integration verfügbar")
                # Führe eine einfache Operation mit der PRISM-Integration durch
                result_prism = prism_integration.calculate_probability(tensor1_prepared, tensor2_prepared)
                logger.info(f"PRISM-Operation erfolgreich: {result_prism}")
            else:
                logger.warning("PRISM-Integration nicht verfügbar")
        except Exception as e:
            logger.warning(f"PRISM-Integration konnte nicht getestet werden: {e}")
        
        # Teste VXOR-Integration, falls verfügbar
        try:
            vxor_integration = t_math_manager.get_vxor_integration()
            if vxor_integration:
                logger.info("VXOR-Integration verfügbar")
                # Hole verfügbare Module
                available_modules = vxor_integration.get_available_modules()
                logger.info(f"Verfügbare VXOR-Module: {available_modules}")
            else:
                logger.warning("VXOR-Integration nicht verfügbar")
        except Exception as e:
            logger.warning(f"VXOR-Integration konnte nicht getestet werden: {e}")
        
        logger.info("T-MATHEMATICS Test erfolgreich abgeschlossen")
        return True
    except Exception as e:
        logger.error(f"T-MATHEMATICS Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_numpy_fallback():
    """Testet den NumPy-Fallback, falls T-MATHEMATICS nicht verfügbar ist"""
    logger.info("Teste NumPy-Fallback...")
    
    try:
        # Erstelle NumPy-Arrays
        array1 = np.array([1.0, 2.0, 3.0, 4.0])
        array2 = np.array([5.0, 6.0, 7.0, 8.0])
        
        # Führe einfache Operationen durch
        result_add = array1 + array2
        logger.info(f"NumPy-Addition erfolgreich: {result_add}")
        
        result_mul = array1 * array2
        logger.info(f"NumPy-Multiplikation erfolgreich: {result_mul}")
        
        result_dot = np.dot(array1, array2)
        logger.info(f"NumPy-Skalarprodukt erfolgreich: {result_dot}")
        
        logger.info("NumPy-Fallback-Test erfolgreich abgeschlossen")
        return True
    except Exception as e:
        logger.error(f"NumPy-Fallback-Test fehlgeschlagen: {e}")
        return False

if __name__ == "__main__":
    logger.info("=== Starte einfachen T-MATHEMATICS Test ===")
    
    # Teste T-MATHEMATICS
    t_math_success = test_t_mathematics_basic()
    
    # Teste NumPy-Fallback
    numpy_success = test_numpy_fallback()
    
    # Gib Zusammenfassung aus
    logger.info("=== Testergebnisse ===")
    logger.info(f"T-MATHEMATICS: {'Erfolgreich' if t_math_success else 'Fehlgeschlagen'}")
    logger.info(f"NumPy-Fallback: {'Erfolgreich' if numpy_success else 'Fehlgeschlagen'}")
    
    # Setze Exit-Code
    if t_math_success and numpy_success:
        logger.info("Alle Tests erfolgreich abgeschlossen")
        sys.exit(0)
    else:
        logger.error("Einige Tests sind fehlgeschlagen")
        sys.exit(1)
