#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - PRISM ↔ VXOR Basisintegrationstest

Dieses Skript testet die grundlegende Integration zwischen der PRISM-Engine
und den VXOR-Agenten (VX-CHRONOS, VX-GESTALT) ohne komplexe Testumgebung.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.tests.prism_vxor_basic")

print("1. PRISM ↔ VXOR Basisintegrationstest wird gestartet...")

# Definiere Pfade
MISO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__))))
VXOR_AI_PATH = Path(os.path.abspath(os.path.join(MISO_ROOT, "..", "VXOR.AI")))

# Füge Pfade zum Pythonpfad hinzu
sys.path.insert(0, str(MISO_ROOT))
sys.path.insert(0, str(VXOR_AI_PATH))

print(f"2. Python-Pfade konfiguriert:")
print(f"   - MISO_ROOT: {MISO_ROOT}")
print(f"   - VXOR_AI_PATH: {VXOR_AI_PATH}")

# Importiere Module
try:
    print("3. Importiere VXOR-Adapter und PRISM-Engine...")
    from miso.vxor.vx_adapter_core import VXORAdapter
    from miso.simulation.prism_engine import PrismEngine, PrismMatrix

    # Initialisiere VXOR-Adapter
    print("4. Initialisiere VXOR-Adapter...")
    adapter = VXORAdapter()
    
    # Überprüfe VXOR-Module Status
    module_status = adapter.get_module_status()
    print("\n5. Status der VXOR-Module:")
    for module_name, status_data in module_status.items():
        print(f"   - {module_name}: {status_data['status']}")
    
    # Initialisiere PRISM-Engine mit minimaler Konfiguration
    print("\n6. Initialisiere PRISM-Engine...")
    prism_config = {
        "matrix_dimensions": 3,
        "use_hardware_acceleration": False
    }
    prism_engine = PrismEngine(prism_config)
    prism_engine.start()
    print("   - PRISM-Engine erfolgreich gestartet")
    
    # Teste VX-CHRONOS Integration
    print("\n7. Teste PRISM ↔ VX-CHRONOS Integration...")
    try:
        chronos = adapter.get_module("VX-CHRONOS")
        print(f"   - VX-CHRONOS erfolgreich geladen")
        
        # Erstelle TemporalController
        temporal_controller = chronos.TemporalController()
        print("   - TemporalController erstellt")
        
        # Erstelle ein temporales Event
        event_id = temporal_controller.schedule_event(
            event_type="prism_sync",
            timestamp=time.time() + 0.1,
            data={"operation": "sync_matrices"}
        )
        print(f"   - Event erstellt mit ID: {event_id}")
        
        # Registriere Callback
        def event_handler(event_data):
            print(f"   - Event wird verarbeitet: {event_data}")
            
            # Führe eine einfache Matrixoperation mit PRISM durch
            matrix = PrismMatrix.create_random(3, 3)
            result = prism_engine.calculate_matrix_inverse(matrix)
            
            return {
                "status": "success",
                "timestamp": time.time(),
                "matrix_determinant": result.get("determinant", 0)
            }
            
        # Registriere Handler und führe temporale Schleife aus
        temporal_controller.register_handler("prism_sync", event_handler)
        print("   - Event-Handler registriert")
        
        # Starte temporale Schleife (nicht-blockierend)
        temporal_controller.start(blocking=False)
        print("   - Temporale Schleife gestartet")
        
        # Warte kurz
        time.sleep(0.3)
        
        # Stoppe temporale Schleife
        temporal_controller.stop()
        print("   - Temporale Schleife gestoppt")
        
        # Prüfe, ob Event verarbeitet wurde
        events = temporal_controller.get_processed_events()
        print(f"   - Verarbeitete Events: {len(events)}")
        if events and any(e.get("id") == event_id for e in events):
            print("   ✅ PRISM ↔ VX-CHRONOS Integration erfolgreich")
        else:
            print("   ❌ PRISM ↔ VX-CHRONOS Integration fehlgeschlagen")
            
    except Exception as e:
        print(f"   ❌ Fehler bei VX-CHRONOS Integration: {e}")
    
    # Teste VX-GESTALT Integration
    print("\n8. Teste PRISM ↔ VX-GESTALT Integration...")
    try:
        gestalt = adapter.get_module("VX-GESTALT")
        print(f"   - VX-GESTALT erfolgreich geladen")
        
        # Erstelle GestaltIntegrator
        integrator = gestalt.GestaltIntegrator()
        print("   - GestaltIntegrator erstellt")
        
        # Erstelle Testdaten
        test_prism_data = {
            "probability_matrix": [[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.1, 0.1, 0.8]],
            "state_vector": [0.33, 0.33, 0.34]
        }
        
        # Registriere PRISM als Subsystem
        integrator.register_subsystem(
            name="prism_engine",
            system=prism_engine,
            interface={
                "get_data": lambda: test_prism_data,
                "process_feedback": lambda feedback: print(f"   - PRISM erhielt Feedback: {feedback}")
            }
        )
        print("   - PRISM als Subsystem registriert")
        
        # Verarbeite Daten
        result = integrator.process_data({"source": "prism_engine"})
        print(f"   - Ergebnis der Datenverarbeitung: {result}")
        
        if result and result.get("processed") == True:
            print("   ✅ PRISM ↔ VX-GESTALT Integration erfolgreich")
        else:
            print("   ❌ PRISM ↔ VX-GESTALT Integration fehlgeschlagen")
            
    except Exception as e:
        print(f"   ❌ Fehler bei VX-GESTALT Integration: {e}")
    
    # Beende PRISM-Engine
    prism_engine.stop()
    print("\n9. PRISM-Engine gestoppt")
    
    print("\n10. Integrationstest abgeschlossen.")
    
except ImportError as e:
    print(f"❌ Fehler beim Import der Module: {e}")
except Exception as e:
    print(f"❌ Unerwarteter Fehler: {e}")
