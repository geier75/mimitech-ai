#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Einfacher Integrationstest

Testet die Integration zwischen T-MATHEMATICS Engine, PRISM und VXOR-Modulen
ohne externe Abhängigkeiten.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import traceback

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.SimpleIntegrationTest")

# Prüfe, ob Apple Silicon verfügbar ist
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
logger.info(f"Apple Silicon: {is_apple_silicon}")

def test_module_import():
    """Testet, ob alle Module korrekt importiert werden können"""
    modules_to_test = [
        "miso.math.t_mathematics.engine",
        "miso.math.t_mathematics.mlx_support",
        "miso.simulation.prism_engine",
        "miso.simulation.prism_matrix",
        "miso.vxor_modules.hyperfilter_t_mathematics",
        "miso.vxor.t_mathematics_bridge"
    ]
    
    success_count = 0
    failure_count = 0
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            logger.info(f"✅ Modul {module_name} erfolgreich importiert")
            success_count += 1
        except Exception as e:
            logger.error(f"❌ Fehler beim Importieren des Moduls {module_name}: {e}")
            failure_count += 1
    
    return success_count, failure_count

def test_t_mathematics_engine():
    """Testet die grundlegenden Funktionen der T-MATHEMATICS Engine"""
    try:
        from miso.math.t_mathematics.engine import TMathEngine
        from miso.math.t_mathematics.compat import TMathConfig
        
        # Initialisiere die Engine
        config = TMathConfig(optimize_for_apple_silicon=is_apple_silicon)
        engine = TMathEngine(config=config)
        
        logger.info(f"✅ T-MATHEMATICS Engine erfolgreich initialisiert (MLX: {engine.use_mlx})")
        return True
    except Exception as e:
        logger.error(f"❌ Fehler bei der Initialisierung der T-MATHEMATICS Engine: {e}")
        logger.error(traceback.format_exc())
        return False

def test_prism_matrix():
    """Testet die PrismMatrix-Klasse"""
    try:
        from miso.simulation.prism_matrix import PrismMatrix
        
        # Erstelle Matrix
        matrix = PrismMatrix(dimensions=3, initial_size=5)
        
        # Prüfe, ob das tensor_backend-Attribut vorhanden ist
        if hasattr(matrix, "tensor_backend"):
            logger.info(f"✅ PrismMatrix hat das tensor_backend-Attribut: {matrix.tensor_backend}")
            return True
        else:
            logger.error("❌ PrismMatrix hat kein tensor_backend-Attribut")
            return False
    except Exception as e:
        logger.error(f"❌ Fehler bei der Initialisierung der PrismMatrix: {e}")
        logger.error(traceback.format_exc())
        return False

def test_prism_engine():
    """Testet die PrismEngine-Klasse"""
    try:
        from miso.simulation.prism_engine import PrismEngine
        
        # Erstelle Engine
        config = {"use_mlx": is_apple_silicon, "precision": "float16"}
        engine = PrismEngine(config)
        
        # Prüfe, ob die _fallback_tensor_operation-Methode vorhanden ist
        if hasattr(engine, "_fallback_tensor_operation"):
            logger.info("✅ PrismEngine hat die _fallback_tensor_operation-Methode")
            return True
        else:
            logger.error("❌ PrismEngine hat keine _fallback_tensor_operation-Methode")
            return False
    except Exception as e:
        logger.error(f"❌ Fehler bei der Initialisierung der PrismEngine: {e}")
        logger.error(traceback.format_exc())
        return False

def test_hyperfilter_math_engine():
    """Testet die HyperfilterMathEngine-Klasse"""
    try:
        from miso.vxor_modules.hyperfilter_t_mathematics import get_hyperfilter_math_engine
        
        # Hole Engine
        hyperfilter_engine = get_hyperfilter_math_engine()
        
        logger.info(f"✅ HyperfilterMathEngine erfolgreich initialisiert (Apple Silicon: {hyperfilter_engine.is_apple_silicon})")
        return True
    except Exception as e:
        logger.error(f"❌ Fehler bei der Initialisierung der HyperfilterMathEngine: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Hauptfunktion"""
    logger.info("Starte einfachen Integrationstest")
    
    # Teste Modulimporte
    success_count, failure_count = test_module_import()
    logger.info(f"Modulimporte: {success_count} erfolgreich, {failure_count} fehlgeschlagen")
    
    # Teste T-MATHEMATICS Engine
    t_math_success = test_t_mathematics_engine()
    
    # Teste PrismMatrix
    prism_matrix_success = test_prism_matrix()
    
    # Teste PrismEngine
    prism_engine_success = test_prism_engine()
    
    # Teste HyperfilterMathEngine
    hyperfilter_success = test_hyperfilter_math_engine()
    
    # Gesamtergebnis
    all_tests = [
        t_math_success,
        prism_matrix_success,
        prism_engine_success,
        hyperfilter_success
    ]
    
    success_count = sum(1 for test in all_tests if test)
    failure_count = sum(1 for test in all_tests if not test)
    
    logger.info(f"Gesamtergebnis: {success_count} von {len(all_tests)} Tests erfolgreich")
    
    if failure_count == 0:
        logger.info("✅ Alle Tests erfolgreich!")
        return 0
    else:
        logger.error(f"❌ {failure_count} Tests fehlgeschlagen")
        return 1

if __name__ == "__main__":
    sys.exit(main())
