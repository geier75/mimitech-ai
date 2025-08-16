#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - NEXUS-OS Beispielanwendung

Diese Datei demonstriert die Verwendung des NEXUS-OS zur Integration von
M-LINGUA und T-Mathematics für natürlichsprachige Tensor-Operationen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
from typing import Dict, Any

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.nexus_os.example")

# Importiere NEXUS-OS Komponenten
from miso.core.nexus_os.nexus_core import get_nexus_core
from miso.core.nexus_os.lingua_math_bridge import LinguaMathBridge
from miso.core.nexus_os.tensor_language_processor import TensorLanguageProcessor

def demonstrate_nexus_os():
    """Demonstriert die Funktionalität des NEXUS-OS"""
    logger.info("Starte NEXUS-OS Demonstration")
    
    # Initialisiere NEXUS-OS Kern
    nexus = get_nexus_core()
    
    # Hole Komponenten
    lingua_math_bridge = nexus.get_component("lingua_math_bridge")
    tensor_processor = nexus.get_component("tensor_processor")
    
    if not lingua_math_bridge:
        logger.error("LinguaMathBridge nicht gefunden, initialisiere manuell")
        lingua_math_bridge = LinguaMathBridge()
        lingua_math_bridge.initialize()
    
    if not tensor_processor:
        logger.error("TensorLanguageProcessor nicht gefunden, initialisiere manuell")
        tensor_processor = TensorLanguageProcessor()
        tensor_processor.initialize()
    
    # Zeige verfügbare Backends
    backends = tensor_processor.get_available_backends()
    logger.info(f"Verfügbare Tensor-Backends: {', '.join(backends)}")
    
    # Beispiel 1: Tensor erstellen über natürliche Sprache
    example_1 = "Erstelle einen Tensor mit den Werten [1, 2, 3, 4, 5]"
    logger.info(f"Beispiel 1: {example_1}")
    result_1 = tensor_processor.process_language_request(example_1)
    
    if result_1["success"]:
        # Registriere den erstellten Tensor
        tensor_processor.register_tensor("zahlen", result_1["result"])
        logger.info(f"Tensor erstellt mit Backend: {result_1['backend']}")
    else:
        logger.error(f"Fehler: {result_1['error']}")
    
    # Beispiel 2: Matrix erstellen
    example_2 = "Erstelle einen Tensor mit den Werten [[1, 2], [3, 4]]"
    logger.info(f"Beispiel 2: {example_2}")
    result_2 = tensor_processor.process_language_request(example_2)
    
    if result_2["success"]:
        # Registriere den erstellten Tensor
        tensor_processor.register_tensor("matrix", result_2["result"])
        logger.info(f"Matrix erstellt mit Backend: {result_2['backend']}")
    else:
        logger.error(f"Fehler: {result_2['error']}")
    
    # Beispiel 3: Komplexere Anfrage über LinguaMathBridge
    example_3 = "Berechne die Summe von zahlen und normalisiere das Ergebnis"
    logger.info(f"Beispiel 3: {example_3}")
    
    # Parse natürliche Sprache mit LinguaMathBridge
    operations = lingua_math_bridge.parse_natural_language(example_3)
    logger.info(f"Erkannte Operation: {operations['operation']}")
    
    # Führe Operation aus
    result_3 = lingua_math_bridge.execute_math_operation(operations)
    
    if result_3["success"]:
        logger.info(f"Ergebnis: {result_3['result']}")
    else:
        logger.error(f"Fehler: {result_3['error']}")
    
    # Beispiel 4: Automatische Backend-Auswahl
    example_4 = "Erstelle einen Tensor mit den Werten [[1, 2, 3], [4, 5, 6], [7, 8, 9]]"
    logger.info(f"Beispiel 4: {example_4}")
    result_4 = tensor_processor.process_language_request(example_4)
    
    if result_4["success"]:
        # Registriere den erstellten Tensor
        tensor_processor.register_tensor("matrix2", result_4["result"])
        logger.info(f"Matrix erstellt mit Backend: {result_4['backend']}")
        
        # Transponieren mit automatischer Backend-Auswahl
        example_4b = "Transponiere den Tensor matrix2"
        logger.info(f"Beispiel 4b: {example_4b}")
        result_4b = tensor_processor.process_language_request(example_4b)
        
        if result_4b["success"]:
            logger.info(f"Matrix transponiert mit Backend: {result_4b['backend']}")
        else:
            logger.error(f"Fehler: {result_4b['error']}")
    else:
        logger.error(f"Fehler: {result_4['error']}")
    
    logger.info("NEXUS-OS Demonstration abgeschlossen")

def main():
    """Hauptfunktion"""
    try:
        demonstrate_nexus_os()
    except Exception as e:
        logger.error(f"Fehler in der Demonstration: {e}")
    finally:
        # Bereinige NEXUS-OS
        from miso.core.nexus_os.nexus_core import reset_nexus_core
        reset_nexus_core()

if __name__ == "__main__":
    main()
