#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Integrationstests für PRISM-Engine und VXOR-Agenten

Dieses Modul implementiert Integrationstests zwischen der PRISM-Engine
und den VXOR-Agenten (VX-CHRONOS, VX-GESTALT, VX-CONTROL).

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import pytest
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import MagicMock, patch
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.tests.integration.test_prism_vxor_integration")

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ZTM-Protokoll-Initialisierung (aktiviert für Integrationstests)
os.environ['MISO_ZTM_MODE'] = '1'  # ZTM aktiviert (war '0')
os.environ['MISO_ZTM_LOG_LEVEL'] = 'DEBUG'
# Stelle sicher, dass ZTM-Logs in das richtige Verzeichnis geschrieben werden
ztm_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(ztm_log_dir, exist_ok=True)
logger.info(f"Zero-Trust Monitoring (ZTM) aktiviert mit Log-Verzeichnis: {ztm_log_dir}")

# Importiere zu testende Module
try:
    from miso.simulation.prism_engine import PrismEngine
    from miso.vxor.vx_adapter_core import VXORAdapter
    from miso.timeline.echo_prime_controller import EchoPrimeController
    
    # Definiere Pfade
    MISO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    PROJECT_ROOT = Path(os.path.abspath(os.path.join(MISO_ROOT, "..")))
    VXOR_AI_PATH = PROJECT_ROOT / "VXOR.AI"
    
    # Füge externe VXOR-Pfade hinzu
    if VXOR_AI_PATH.exists():
        sys.path.insert(0, str(VXOR_AI_PATH))
        HAS_VXOR_AI = True
    else:
        HAS_VXOR_AI = False
        logger.warning(f"VXOR.AI-Pfad nicht gefunden: {VXOR_AI_PATH}")
    
    # Flags für verfügbare Module setzen
    HAS_ALL_DEPENDENCIES = True
except ImportError as e:
    logger.error(f"Fehler beim Import von Abhängigkeiten: {e}")
    HAS_ALL_DEPENDENCIES = False
    HAS_VXOR_AI = False

# Benchmark-Dekorator
def benchmark(func):
    """Dekorator für Benchmarking von Funktionsaufrufen"""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.info(f"Funktion {func.__name__} ausgeführt in {elapsed:.6f} Sekunden")
        return result
    return wrapper

# Test-Fixtures
@pytest.fixture
def prism_engine():
    """Fixture für PrismEngine"""
    config = {"matrix_dimensions": 5}
    engine = PrismEngine(config)
    engine.start()
    yield engine
    engine.stop()

@pytest.fixture
def vxor_adapter():
    """Fixture für VXORAdapter"""
    adapter = VXORAdapter()
    yield adapter

@pytest.fixture
def echo_prime_controller():
    """Fixture für EchoPrimeController"""
    controller = EchoPrimeController()
    controller.initialize()
    yield controller

@pytest.fixture
def sample_timeline(echo_prime_controller):
    """Fixture für eine Beispiel-Timeline"""
    timeline_id = echo_prime_controller.create_timeline("Test Timeline")
    # Erstelle einfache Baumstruktur
    root_id = echo_prime_controller.add_node(
        timeline_id, 
        data={"type": "root", "value": 1.0},
        parent_id=None
    )
    yield timeline_id
    echo_prime_controller.delete_timeline(timeline_id)

# =============================================================================
# 1. Integration mit VX-CHRONOS
# =============================================================================
@pytest.mark.integration
@pytest.mark.vxor
@pytest.mark.chronos
class TestPrismChronosIntegration:
    """Tests für die Integration zwischen PRISM-Engine und VX-CHRONOS"""
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES or not HAS_VXOR_AI, 
                         reason="Nicht alle Abhängigkeiten verfügbar")
    def test_chronos_module_loading(self, vxor_adapter):
        """Test für das Laden des VX-CHRONOS-Moduls"""
        # Versuche, VX-CHRONOS zu laden
        chronos = vxor_adapter.load_module("vx_chronos")
        
        # Verifiziere Modul
        assert chronos is not None
        assert hasattr(chronos, "TemporalController")
        
        # Teste Initialisierung
        temporal_controller = chronos.TemporalController()
        assert temporal_controller is not None
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES or not HAS_VXOR_AI, 
                         reason="Nicht alle Abhängigkeiten verfügbar")
    def test_chronos_prism_temporal_sync(self, vxor_adapter, prism_engine):
        """Test für die temporale Synchronisierung zwischen PRISM und VX-CHRONOS"""
        # Lade VX-CHRONOS
        chronos = vxor_adapter.load_module("vx_chronos")
        
        # Initialisiere TemporalController
        temporal_controller = chronos.TemporalController()
        
        # Erstelle temporales Event in CHRONOS
        event_id = temporal_controller.schedule_event(
            event_type="prism_sync",
            timestamp=time.time() + 0.1,
            data={"operation": "sync_matrices"}
        )
        
        # Registriere PRISM-Callback für das Event
        def prism_event_handler(event_data):
            assert event_data["operation"] == "sync_matrices"
            return {"status": "processed", "timestamp": time.time()}
        
        # Registriere Handler
        temporal_controller.register_handler(
            event_type="prism_sync",
            handler=prism_event_handler
        )
        
        # Starte temporale Schleife (kurz)
        temporal_controller.start(blocking=False)
        time.sleep(0.2)  # Warte kurz, damit Event ausgeführt wird
        temporal_controller.stop()
        
        # Prüfe, ob Event ausgeführt wurde
        events = temporal_controller.get_processed_events()
        assert any(e["id"] == event_id for e in events)

# =============================================================================
# 2. Integration mit VX-GESTALT
# =============================================================================
@pytest.mark.integration
@pytest.mark.vxor
@pytest.mark.gestalt
class TestPrismGestaltIntegration:
    """Tests für die Integration zwischen PRISM-Engine und VX-GESTALT"""
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES or not HAS_VXOR_AI, 
                         reason="Nicht alle Abhängigkeiten verfügbar")
    def test_gestalt_module_loading(self, vxor_adapter):
        """Test für das Laden des VX-GESTALT-Moduls"""
        # Versuche, VX-GESTALT zu laden
        gestalt = vxor_adapter.load_module("vx_gestalt")
        
        # Verifiziere Modul
        assert gestalt is not None
        assert hasattr(gestalt, "GestaltIntegrator")
        
        # Teste Initialisierung
        integrator = gestalt.GestaltIntegrator()
        assert integrator is not None
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES or not HAS_VXOR_AI, 
                         reason="Nicht alle Abhängigkeiten verfügbar")
    def test_gestalt_prism_probability_integration(self, vxor_adapter, prism_engine):
        """Test für die Integration der Wahrscheinlichkeitsanalyse zwischen PRISM und VX-GESTALT"""
        # Lade VX-GESTALT
        gestalt = vxor_adapter.load_module("vx_gestalt")
        
        # Initialisiere GestaltIntegrator
        integrator = gestalt.GestaltIntegrator()
        
        # Erstelle Testdaten für die Feedback-Schleife
        test_prism_data = {
            "probability_matrix": [[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.1, 0.1, 0.8]],
            "state_vector": [0.33, 0.33, 0.34]
        }
        
        # Registriere PRISM-Engine als Subsystem
        integrator.register_subsystem(
            name="prism_engine",
            system=prism_engine,
            interface={
                "get_data": lambda: test_prism_data,
                "set_data": lambda x: None  # Mock-Funktion
            }
        )
        
        # Führe Gestalt-Integration durch
        result = integrator.process_data({"source": "prism_engine"})
        
        # Verifiziere Ergebnis
        assert result is not None
        assert "processed" in result
        assert result["processed"] == True

# =============================================================================
# 3. Integration mit VX-CONTROL
# =============================================================================
@pytest.mark.integration
@pytest.mark.vxor
@pytest.mark.control
class TestPrismControlIntegration:
    """Tests für die Integration zwischen PRISM-Engine und VX-CONTROL"""
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                         reason="Nicht alle Abhängigkeiten verfügbar")
    def test_control_module_loading(self):
        """Test für das Laden des VX-CONTROL-Moduls"""
        # Importiere VX-CONTROL direkt
        try:
            sys.path.insert(0, str(MISO_ROOT / ".."))
            import vx_control
            
            # Verifiziere Modul
            assert vx_control is not None
            assert hasattr(vx_control, "ControlAgent")
            
            # Initialisierung nur simulieren (keine tatsächliche Ausführung)
            with patch.object(vx_control.ControlAgent, '__init__', return_value=None):
                agent = vx_control.ControlAgent()
                assert agent is not None
                
        except ImportError as e:
            pytest.skip(f"VX-CONTROL nicht importierbar: {e}")
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                         reason="Nicht alle Abhängigkeiten verfügbar")
    def test_control_prism_resource_monitoring(self, prism_engine):
        """Test für die Ressourcenüberwachung zwischen PRISM und VX-CONTROL"""
        # Importiere VX-CONTROL direkt
        try:
            sys.path.insert(0, str(MISO_ROOT / ".."))
            import vx_control
            
            # Mock-Objekt für ControlAgent erstellen
            with patch.object(vx_control, 'ControlAgent') as MockAgent:
                # Konfiguriere Mock
                instance = MockAgent.return_value
                instance.monitor_resource_usage.return_value = {
                    "cpu": 10.5,
                    "memory": 125.4,
                    "prism_process_id": os.getpid()
                }
                
                # Erstelle Agent
                agent = vx_control.ControlAgent()
                
                # Teste Ressourcenüberwachung
                resources = agent.monitor_resource_usage()
                
                # Verifiziere Ergebnis
                assert resources is not None
                assert "cpu" in resources
                assert "memory" in resources
                assert "prism_process_id" in resources
                
        except ImportError as e:
            pytest.skip(f"VX-CONTROL nicht importierbar: {e}")

# =============================================================================
# 4. PRISM ↔ Mehrere VXOR-Agenten-Integration
# =============================================================================
@pytest.mark.integration
@pytest.mark.vxor
@pytest.mark.multi
class TestPrismMultiVXORIntegration:
    """Tests für die Integration zwischen PRISM-Engine und mehreren VXOR-Agenten"""
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES or not HAS_VXOR_AI, 
                         reason="Nicht alle Abhängigkeiten verfügbar")
    def test_multi_agent_integration(self, vxor_adapter, prism_engine):
        """Test für die Integration mehrerer VXOR-Agenten mit PRISM"""
        # Versuche, VX-CHRONOS und VX-GESTALT zu laden
        chronos = vxor_adapter.load_module("vx_chronos")
        gestalt = vxor_adapter.load_module("vx_gestalt")
        
        if chronos is None or gestalt is None:
            pytest.skip("VX-CHRONOS oder VX-GESTALT nicht verfügbar")
        
        # Initialisiere Agenten
        temporal_controller = chronos.TemporalController()
        integrator = gestalt.GestaltIntegrator()
        
        # Erstelle Event-Handler für die Integration
        def temporal_event_handler(event_data):
            """Handler für temporale Events, integriert mit PRISM"""
            if event_data.get("type") == "prism_probability_update":
                # Simuliere PRISM-Verarbeitung
                result = prism_engine.calculate_matrix_probabilities(
                    data=event_data.get("data", {})
                )
                # Leite Ergebnis an Gestalt-Integrator weiter
                integrator.process_data({
                    "source": "prism_engine",
                    "data": result
                })
                return {"status": "processed"}
            return {"status": "ignored"}
        
        # Registriere Handler
        temporal_controller.register_handler(
            event_type="prism_probability_update",
            handler=temporal_event_handler
        )
        
        # Teste Event-Verarbeitung
        event_id = temporal_controller.schedule_event(
            event_type="prism_probability_update",
            timestamp=time.time() + 0.1,
            data={
                "matrix": [[0.5, 0.5], [0.5, 0.5]],
                "source": "test"
            }
        )
        
        # Starte temporale Schleife (kurz)
        temporal_controller.start(blocking=False)
        time.sleep(0.2)  # Warte kurz, damit Event ausgeführt wird
        temporal_controller.stop()
        
        # Prüfe, ob Event verarbeitet wurde
        events = temporal_controller.get_processed_events()
        assert any(e["id"] == event_id for e in events)
        
        # Mock PRISM-Engine-Methode für den Test
        with patch.object(prism_engine, 'calculate_matrix_probabilities') as mock_calc:
            mock_calc.return_value = {"probability": 0.75}
            
            # Wiederhole Test mit Mock
            event_id = temporal_controller.schedule_event(
                event_type="prism_probability_update",
                timestamp=time.time() + 0.1,
                data={
                    "matrix": [[0.5, 0.5], [0.5, 0.5]],
                    "source": "test"
                }
            )
            
            # Starte temporale Schleife (kurz)
            temporal_controller.start(blocking=False)
            time.sleep(0.2)  # Warte kurz, damit Event ausgeführt wird
            temporal_controller.stop()
            
            # Verifiziere, dass Mock aufgerufen wurde
            mock_calc.assert_called_once()

# Main ausführen, wenn Skript direkt aufgerufen wird
if __name__ == "__main__":
    pytest.main(["-v", __file__])
