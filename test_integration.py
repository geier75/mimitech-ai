#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MISO - Test für Q-LOGIK Integration

Dieses Skript testet die Integration zwischen Q-LOGIK, T-Mathematics und MPrime Engine.
"""

import sys
import os
import logging
import importlib.util

# Füge das Projektverzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Importiere erforderliche Bibliotheken
import numpy as np
import torch

# Stelle sicher, dass NumPy und PyTorch verfügbar sind
print(f"NumPy Version: {np.__version__}")
print(f"PyTorch Version: {torch.__version__}")

# Importiere die Integrationsmodule
from miso.logic.qlogik_integration import QLOGIKIntegrationManager, QLOGIKIntegratedDecisionMaker
print("✓ Erfolgreich importiert: QLOGIKIntegrationManager und QLOGIKIntegratedDecisionMaker")

from miso.logic.qlogik_tmathematics import QLOGIKTMathematicsIntegration, MLINGUATMathIntegration
print("✓ Erfolgreich importiert: QLOGIKTMathematicsIntegration und MLINGUATMathIntegration")

from miso.logic.qlogik_mprime import QLOGIKMPrimeIntegration, MLINGUAMPrimeIntegration
print("✓ Erfolgreich importiert: QLOGIKMPrimeIntegration und MLINGUAMPrimeIntegration")

# Teste die Integration
def test_integration():
    """Testet die Q-LOGIK Integration"""
    
    print("\n=== Test der Q-LOGIK Integration ===\n")
    
    # Erstelle Integrationsmanager
    print("Initialisiere QLOGIKIntegrationManager...")
    manager = QLOGIKIntegrationManager()
    print("✓ QLOGIKIntegrationManager initialisiert")
    
    # Erstelle integrierten Entscheidungsträger
    print("\nInitialisiere QLOGIKIntegratedDecisionMaker...")
    decision_maker = QLOGIKIntegratedDecisionMaker(manager)
    print("✓ QLOGIKIntegratedDecisionMaker initialisiert")
    
    # Teste Tensor-Operation
    print("\n--- Test: Tensor-Operation ---")
    tensor_query = "Multipliziere diese Matrizen mit hoher Priorität"
    tensor_context = {
        "memory_available": 0.9,
        "computation_complexity": 0.4,
        "system_load": 0.3,
        "priority": 0.7,
        "risk": 0.2,
        "benefit": 0.8,
        "urgency": 0.6
    }
    
    print(f"Anfrage: '{tensor_query}'")
    print("Kontext:", tensor_context)
    
    try:
        tensor_decision = decision_maker.make_integrated_decision(
            query=tensor_query,
            context=tensor_context
        )
        print("\nTensor-Entscheidung:")
        print(f"- Entscheidung: {tensor_decision.get('decision', 'Unbekannt')}")
        print(f"- Konfidenz: {tensor_decision.get('confidence', 0.0):.2f}")
        print(f"- Priorität: {tensor_decision.get('priority', {}).get('level', 'Unbekannt')}")
        print(f"- Empfehlung: {tensor_decision.get('recommendation', 'Keine')}")
        print("✓ Tensor-Entscheidung erfolgreich")
    except Exception as e:
        print(f"✗ Fehler bei Tensor-Entscheidung: {str(e)}")
    
    # Teste symbolische Operation
    print("\n--- Test: Symbolische Operation ---")
    symbolic_query = "Löse die Gleichung x^2 + 3*x - 5 = 0 mit hoher Priorität"
    symbolic_context = {
        "expression_complexity": 0.4,
        "symbolic_depth": 0.6,
        "system_capacity": 0.8,
        "priority": 0.7,
        "risk": 0.2,
        "benefit": 0.8,
        "urgency": 0.6
    }
    
    print(f"Anfrage: '{symbolic_query}'")
    print("Kontext:", symbolic_context)
    
    try:
        symbolic_decision = decision_maker.make_integrated_decision(
            query=symbolic_query,
            context=symbolic_context
        )
        print("\nSymbolische Entscheidung:")
        print(f"- Entscheidung: {symbolic_decision.get('decision', 'Unbekannt')}")
        print(f"- Konfidenz: {symbolic_decision.get('confidence', 0.0):.2f}")
        print(f"- Priorität: {symbolic_decision.get('priority', {}).get('level', 'Unbekannt')}")
        print(f"- Empfehlung: {symbolic_decision.get('recommendation', 'Keine')}")
        print("✓ Symbolische Entscheidung erfolgreich")
    except Exception as e:
        print(f"✗ Fehler bei symbolischer Entscheidung: {str(e)}")
    
    # Teste natürlichsprachliche Verarbeitung
    print("\n--- Test: Natürlichsprachliche Verarbeitung ---")
    nl_query = "Berechne die Matrix-Multiplikation von A und B mit MLX"
    
    print(f"Anfrage: '{nl_query}'")
    
    try:
        nl_result = manager.process_natural_language(nl_query)
        print("\nNatürlichsprachliches Ergebnis:")
        print(f"- Erfolg: {nl_result.get('success', False)}")
        if nl_result.get('success', False):
            print(f"- Ergebnis: {nl_result.get('result', 'Keins')}")
        else:
            print(f"- Fehler: {nl_result.get('error', 'Unbekannt')}")
        print("✓ Natürlichsprachliche Verarbeitung erfolgreich")
    except Exception as e:
        print(f"✗ Fehler bei natürlichsprachlicher Verarbeitung: {str(e)}")
    
    print("\n=== Test abgeschlossen ===")

if __name__ == "__main__":
    test_integration()
