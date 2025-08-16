#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Test für zirkuläre Import-Behebung

Dieses Skript testet, ob die Behebung der zirkulären Imports zwischen
PRISM-Engine und ECHO-PRIME erfolgreich war.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import unittest

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.tests.circular_imports_fix")

class TestCircularImportsResolution(unittest.TestCase):
    """Testet die Behebung zirkulärer Imports zwischen PRISM und ECHO-PRIME"""
    
    def test_core_timeline_base_imports(self):
        """Testet, ob die gemeinsamen Basisdatentypen korrekt importiert werden können"""
        try:
            from miso.core.timeline_base import Timeline, TimeNode, TimelineType, TriggerLevel
            
            # Teste grundlegende Funktionalität
            node = TimeNode(
                description="Test Node",
                timestamp=time.time(),
                trigger_level=TriggerLevel.MEDIUM
            )
            
            timeline = Timeline(
                name="Test Timeline",
                description="Test Description",
                timeline_type=TimelineType.PRIMARY
            )
            
            # Füge Node zur Timeline hinzu
            timeline.add_node(node)
            
            # Prüfe, ob Node korrekt hinzugefügt wurde
            self.assertIn(node.id, timeline.nodes)
            
            logger.info("Timeline-Basisdatentypen können erfolgreich importiert und verwendet werden")
        except ImportError as e:
            self.fail(f"Import der Timeline-Basisdatentypen fehlgeschlagen: {e}")
        except Exception as e:
            self.fail(f"Fehler bei der Verwendung der Timeline-Basisdatentypen: {e}")
    
    def test_prism_echo_integration_imports(self):
        """Testet die Imports in der PRISM-ECHO-Integration"""
        try:
            # Import des Integrationsmoduls
            from miso.simulation.prism_echo_prime_integration import PRISMECHOPrimeIntegration, get_echo_prime_controller
            
            # Teste grundlegende Funktionalität
            integration = PRISMECHOPrimeIntegration()
            
            # Lazy-Loading-Methode testen
            echo_controller_class = get_echo_prime_controller()
            
            # Wir prüfen nur, ob keine Exception geworfen wird
            logger.info("PRISM-ECHO-Integration kann erfolgreich importiert werden")
        except ImportError as e:
            self.fail(f"Import der PRISM-ECHO-Integration fehlgeschlagen: {e}")
        except Exception as e:
            self.fail(f"Fehler bei der Verwendung der PRISM-ECHO-Integration: {e}")
    
    def test_echo_prime_controller_imports(self):
        """Testet die Imports im EchoPrimeController"""
        try:
            # Import des Controllers
            from miso.timeline.echo_prime_controller import EchoPrimeController, get_prism_simulator
            
            # Teste grundlegende Funktionalität
            controller = EchoPrimeController()
            
            # Lazy-Loading-Methode testen
            prism_simulator_class = get_prism_simulator()
            
            # Wir prüfen nur, ob keine Exception geworfen wird
            logger.info("EchoPrimeController kann erfolgreich importiert werden")
        except ImportError as e:
            self.fail(f"Import des EchoPrimeControllers fehlgeschlagen: {e}")
        except Exception as e:
            self.fail(f"Fehler bei der Verwendung des EchoPrimeControllers: {e}")
    
    def test_dual_imports(self):
        """Testet, ob beide Module gleichzeitig importiert werden können"""
        try:
            # Beide Module importieren
            from miso.simulation.prism_echo_prime_integration import PRISMECHOPrimeIntegration
            from miso.timeline.echo_prime_controller import EchoPrimeController
            
            # Instanzen erstellen
            integration = PRISMECHOPrimeIntegration()
            controller = EchoPrimeController()
            
            # Integration testen
            if integration.echo_prime is None:
                logger.warning("Integration konnte keinen EchoPrimeController laden - das ist OK in Tests")
            
            # Wir prüfen nur, ob keine Exception geworfen wird
            logger.info("Beide Module können gleichzeitig importiert werden")
        except ImportError as e:
            self.fail(f"Dual-Import fehlgeschlagen: {e}")
        except Exception as e:
            self.fail(f"Fehler beim Dual-Import: {e}")
            
    def test_create_timelines_and_nodes(self):
        """Testet die Erstellung von Zeitlinien und Knoten über beide Systeme"""
        try:
            from miso.timeline.echo_prime_controller import EchoPrimeController
            from miso.simulation.prism_echo_prime_integration import TimelineSimulationContext
            from miso.core.timeline_base import Timeline, TimeNode, TimelineType, TriggerLevel
            
            # Controller erstellen
            controller = EchoPrimeController()
            
            # Zeitlinie über Controller erstellen
            timeline = controller.create_timeline(
                name="Test Timeline via Controller", 
                description="Test Description"
            )
            
            # Knoten über Controller erstellen
            node = controller.add_time_node(
                timeline_id=timeline.id,
                description="Test Node via Controller",
                trigger_level=TriggerLevel.MEDIUM
            )
            
            # Kontext mit dieser Zeitlinie erstellen
            context = TimelineSimulationContext(
                timeline=timeline,
                current_node=node,
                simulation_steps=10
            )
            
            # Kontext in Dict konvertieren
            context_dict = context.to_dict()
            
            # Prüfen, ob die Zeitlinien-ID korrekt übertragen wurde
            self.assertEqual(context_dict['timeline_id'], timeline.id)
            
            logger.info("Zeitlinien und Knoten können über beide Systeme erstellt und verwendet werden")
        except Exception as e:
            self.fail(f"Fehler bei der Erstellung von Zeitlinien und Knoten: {e}")

if __name__ == "__main__":
    logger.info("Starte Tests für zirkuläre Import-Behebung...")
    unittest.main()
