#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - M-LINGUA T-Mathematics Integration Demo

Dieses Skript demonstriert die Integration zwischen dem M-LINGUA Interface
und der T-Mathematics Engine zur Verarbeitung natürlichsprachlicher
Befehle für Tensor-Operationen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import time
from typing import Dict, Any

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("MISO.Ultimate.Examples.MLinguaTMathDemo")

# Importiere benötigte Module
from miso_ultimate.lang.m_lingua.integration.t_mathematics_bridge import TMathematicsBridge

def main():
    """Hauptfunktion für die Demo"""
    logger.info("Starte M-LINGUA T-Mathematics Integration Demo")
    
    # Konfiguration für die Demo
    config = {
        "language": "de",
        "m_lingua": {
            "language": "de",
            "components": {
                "intent_parser": {
                    "model_path": "models/intent_parser"
                },
                "symbol_mapper": {
                    "language": "de"
                },
                "context_resolver": {
                    "language": "de"
                },
                "emotion_tagger": {
                    "language": "de"
                },
                "lingua_rules_engine": {
                    "language": "de"
                }
            }
        },
        "t_mathematics": {
            "default_backend": "mlx",  # Verwende MLX als Standard-Backend
            "backends": ["mlx", "torch", "numpy"],
            "precision": "float32"
        },
        "m_code": {
            "security_level": "high",
            "sandbox_mode": True,
            "max_execution_time": 10.0,
            "allowed_imports": [
                "miso_ultimate", "numpy", "torch", "mlx", "math", "datetime", "time"
            ]
        }
    }
    
    # Initialisiere T-Mathematics Bridge
    bridge = TMathematicsBridge(config)
    
    # Zeige verfügbare Backends
    logger.info(f"Verfügbare Backends: {bridge.get_available_backends()}")
    logger.info(f"Aktives Backend: {bridge.get_default_backend()}")
    
    # Beispiel-Befehle
    commands = [
        "Erstelle einen 3x3 Tensor mit Zufallswerten",
        "Transponiere den Tensor",
        "Berechne die Determinante der Matrix",
        "Wechsle zum PyTorch-Backend",
        "Erstelle eine 4x4 Matrix und berechne die Eigenwerte",
        "Optimiere den Tensor mit MLX"
    ]
    
    # Verarbeite Befehle
    for i, command in enumerate(commands):
        logger.info(f"\n--- Befehl {i+1}: {command} ---")
        
        # Verarbeite Befehl
        result = bridge.process(command)
        
        # Zeige Ergebnis
        if result["success"]:
            logger.info(f"Erfolg: {result['message']}")
            logger.info(f"Ausführungszeit: {result['execution_time']:.4f} Sekunden")
            
            if result["m_code"]:
                logger.info("Generierter M-CODE:")
                for line in result["m_code"].split("\n"):
                    logger.info(f"  {line}")
            
            if result["output"]:
                logger.info("Ausgabe:")
                for line in result["output"].split("\n"):
                    logger.info(f"  {line}")
        else:
            logger.error(f"Fehler: {result['message']}")
            if result["error"]:
                logger.error(f"Details: {result['error']}")
        
        # Kurze Pause zwischen Befehlen
        time.sleep(1)
    
    logger.info("\nM-LINGUA T-Mathematics Integration Demo abgeschlossen")

if __name__ == "__main__":
    main()
