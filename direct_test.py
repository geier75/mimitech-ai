#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Direkter Tensor-Test

Dieses Skript testet die T-Mathematics Engine direkt über den vorgesehenen Import-Pfad.
"""

import os
import sys
import logging
import numpy as np
import time

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [DIRECT-TEST] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.Direct.Test")

# Füge Hauptpfad hinzu
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Versuche Tensor-Imports
logger.info("Versuche Tensor-Module zu importieren...")

try:
    from miso.math.t_mathematics.engine import TMathEngine
    logger.info("TMathEngine erfolgreich importiert")
    
    # Initialisiere T-Math Engine
    try:
        engine = TMathEngine()
        logger.info(f"T-Mathematics Engine initialisiert mit Backend: {engine.get_active_backend()}")
        
        # Versuche einfache Tensor-Operation
        try:
            # Erzeuge Testdaten
            data = np.random.rand(5, 5).astype(np.float32)
            logger.info(f"Test-Daten erzeugt: Shape {data.shape}")
            
            # Erzeuge Tensor
            start_time = time.time()
            tensor = engine.tensor(data)
            logger.info(f"Tensor erzeugt: Backend {tensor.backend}, Shape {tensor.shape}")
            
            # Addition
            result = tensor + tensor
            logger.info(f"Addition durchgeführt: Shape {result.shape}")
            
            # Matrix-Multiplikation
            result = tensor @ tensor
            logger.info(f"Matrix-Multiplikation durchgeführt: Shape {result.shape}")
            
            # Exp-Funktion
            result = tensor.exp()
            logger.info(f"Exponentialfunktion berechnet: Shape {result.shape}")
            
            end_time = time.time()
            logger.info(f"Alle Operationen abgeschlossen in {(end_time - start_time)*1000:.2f} ms")
            
        except Exception as e:
            logger.error(f"Fehler bei Tensor-Operationen: {e}")
    except Exception as e:
        logger.error(f"Fehler bei Engine-Initialisierung: {e}")
except ImportError as e:
    logger.error(f"Fehler beim Import von TMathEngine: {e}")

# Versuche direkten Import der M-LINGUA Integration
logger.info("\nVersuche M-LINGUA Integration zu testen...")

try:
    from miso.lang.mlingua.math_bridge import MathBridge
    logger.info("MathBridge erfolgreich importiert")
    
    # Initialisiere MathBridge
    try:
        bridge = MathBridge()
        logger.info("MathBridge initialisiert")
        
        # Teste mathematische Ausdrücke
        test_expressions = [
            ("Berechne 2 + 3 * 4", "de"),
            ("Calculate the square root of 16", "en")
        ]
        
        for expr, lang in test_expressions:
            try:
                start_time = time.time()
                result = bridge.process_math_expression(expr, lang)
                end_time = time.time()
                
                if result.success:
                    logger.info(f"Expression: '{expr}', Result: {result.result}, Zeit: {(end_time - start_time)*1000:.2f} ms")
                else:
                    logger.warning(f"Expression: '{expr}', Fehler: {result.error_message}")
            except Exception as e:
                logger.error(f"Fehler bei Verarbeitung von '{expr}': {e}")
    except Exception as e:
        logger.error(f"Fehler bei MathBridge-Initialisierung: {e}")
except ImportError as e:
    logger.error(f"Fehler beim Import von MathBridge: {e}")

logger.info("Test abgeschlossen")
