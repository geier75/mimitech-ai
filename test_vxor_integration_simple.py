#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Vereinfachter VXOR-Integrationstest

Dieses Skript testet die grundlegende Integration der VXOR-Agenten
ohne Abhängigkeit von der PRISM-Engine oder Hardwarebeschleunigung.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.tests.vxor_integration_simple")

print("\n=== VXOR-Integrationstest (Vereinfacht) ===\n")

# Definiere Pfade
MISO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__))))
VXOR_AI_PATH = Path(os.path.abspath(os.path.join(MISO_ROOT, "..", "VXOR.AI")))

# Füge Pfade zum Pythonpfad hinzu
sys.path.insert(0, str(MISO_ROOT))
sys.path.insert(0, str(VXOR_AI_PATH))

print(f"1. Python-Pfade konfiguriert:")
print(f"   - MISO_ROOT: {MISO_ROOT}")
print(f"   - VXOR_AI_PATH: {VXOR_AI_PATH}")

# Importiere VXOR-Adapter
try:
    print("\n2. Importiere VXOR-Adapter...")
    from miso.vxor.vx_adapter_core import VXORAdapter
    
    # Initialisiere VXOR-Adapter
    print("3. Initialisiere VXOR-Adapter...")
    adapter = VXORAdapter()
    
    # Überprüfe VXOR-Module Status
    module_status = adapter.get_module_status()
    print("\n4. Status der VXOR-Module:")
    for module_name, status_data in module_status.items():
        print(f"   - {module_name}: {status_data['status']}")
    
    # Teste VX-CHRONOS Funktionalität
    print("\n5. Teste VX-CHRONOS Funktionalität...")
    try:
        chronos = adapter.get_module("VX-CHRONOS")
        print(f"   - VX-CHRONOS erfolgreich geladen")
        
        # Erstelle TemporalController
        temporal_controller = chronos.TemporalController()
        print("   - TemporalController erstellt")
        
        # Erstelle ein temporales Event
        current_time = time.time()
        event_id = temporal_controller.schedule_event(
            event_type="test_event",
            timestamp=current_time + 0.1,
            data={"message": "Test-Nachricht"}
        )
        print(f"   - Event erstellt mit ID: {event_id}")
        
        # Registriere Callback
        def event_handler(event_data):
            print(f"   - Event wird verarbeitet: {event_data}")
            return {"status": "processed", "timestamp": time.time()}
            
        # Registriere Handler und führe temporale Schleife aus
        temporal_controller.register_handler("test_event", event_handler)
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
            print("   ✅ VX-CHRONOS Funktionalität erfolgreich")
        else:
            print("   ❌ VX-CHRONOS Funktionalität fehlgeschlagen")
            
    except Exception as e:
        print(f"   ❌ Fehler bei VX-CHRONOS Test: {e}")
    
    # Teste VX-GESTALT Funktionalität
    print("\n6. Teste VX-GESTALT Funktionalität...")
    try:
        # Direkter Import der GestaltIntegrator-Klasse, statt über den Adapter
        from vx_gestalt.gestalt_integrator import GestaltIntegrator
        print(f"   - VX-GESTALT-Komponenten direkt importiert")
        
        # Erstelle GestaltIntegrator
        integrator = GestaltIntegrator()
        print("   - GestaltIntegrator erstellt")
        
        # Erstelle ein Mock-System
        class MockSystem:
            def __init__(self):
                self.name = "MockSystem"
                self.data = {"state": "ready"}
            
            def get_state(self):
                return self.data
        
        mock_system = MockSystem()
        
        # Registriere Mock-System als Subsystem
        integrator.register_subsystem(
            name="mock_system",
            system=mock_system,
            interface={
                "get_data": lambda: mock_system.get_state(),
                "process_feedback": lambda x: None
            }
        )
        print("   - Mock-System als Subsystem registriert")
        
        # Verarbeite Daten
        result = integrator.process_data({"source": "mock_system"})
        print(f"   - Ergebnis der Datenverarbeitung: {result}")
        
        if result and result.get("processed") == True:
            print("   ✅ VX-GESTALT Funktionalität erfolgreich")
        else:
            print("   ❌ VX-GESTALT Funktionalität fehlgeschlagen")
            
    except Exception as e:
        print(f"   ❌ Fehler bei VX-GESTALT Test: {e}")
    
    # Teste Integration zwischen VX-CHRONOS und VX-GESTALT
    print("\n7. Teste VX-CHRONOS ↔ VX-GESTALT Integration...")
    try:
        # Prüfe, ob VX-CHRONOS geladen ist - für GestaltIntegrator verwenden wir den direkten Import
        if "VX-CHRONOS" in adapter.modules:
            print("   - VX-CHRONOS ist geladen, GestaltIntegrator wird direkt importiert")
            
            # Erstelle Controller und Integrator
            temporal_controller = chronos.TemporalController()
            from vx_gestalt.gestalt_integrator import GestaltIntegrator
            integrator = GestaltIntegrator()
            
            # Registriere temporal_controller als Subsystem in GestaltIntegrator
            integrator.register_subsystem(
                name="temporal_controller",
                system=temporal_controller,
                interface={
                    "get_data": lambda: {"status": temporal_controller.is_running(), "events": len(temporal_controller.get_scheduled_events())},
                    "process_feedback": lambda feedback: temporal_controller.schedule_event(
                        event_type="gestalt_feedback",
                        timestamp=time.time() + 0.1,
                        data=feedback
                    )
                }
            )
            print("   - TemporalController als Subsystem im GestaltIntegrator registriert")
            
            # Starte temporale Schleife
            temporal_controller.start(blocking=False)
            print("   - Temporale Schleife gestartet")
            
            # Verarbeite Daten
            result = integrator.process_data({"source": "temporal_controller"})
            print(f"   - Ergebnis der Datenverarbeitung: {result}")
            
            # Warte kurz
            time.sleep(0.3)
            
            # Überprüfe gestalt_feedback Events
            events = temporal_controller.get_processed_events()
            has_feedback_event = any(e.get("type") == "gestalt_feedback" for e in events)
            print(f"   - Feedback-Events gefunden: {has_feedback_event}")
            
            # Stoppe temporale Schleife
            temporal_controller.stop()
            print("   - Temporale Schleife gestoppt")
            
            if result and result.get("processed") == True:
                print("   ✅ VX-CHRONOS ↔ VX-GESTALT Integration erfolgreich")
            else:
                print("   ❌ VX-CHRONOS ↔ VX-GESTALT Integration fehlgeschlagen")
        else:
            print("   ❌ Nicht alle erforderlichen Module sind geladen")
            
    except Exception as e:
        print(f"   ❌ Fehler bei der Integration: {e}")
    
    print("\n8. Test abgeschlossen.")
    
except ImportError as e:
    print(f"❌ Fehler beim Import der Module: {e}")
except Exception as e:
    print(f"❌ Unerwarteter Fehler: {e}")
