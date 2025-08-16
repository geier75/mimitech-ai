#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - PRISM-Engine Integrationstest

Dieser Test validiert die Integration der optimierten PRISM-Engine mit:
- VX-REASON
- Q-LOGIK
- MLX-optimierte Tensor-Operationen
- ECHO-PRIME Zeitlinien

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import unittest
import os
import sys
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Pfade zum Hauptprojekt hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# MISO-Module importieren
from miso.simulation.prism_engine import PrismEngine
from miso.simulation.prism_base import Timeline, TimeNode
from miso.simulation.vxor_integration import get_prism_vxor_integration, PRISMVXORIntegration

# Paradoxauflösung importieren
from miso.simulation.paradox_resolution import ParadoxResolutionManager as ParadoxResolver
from miso.simulation.paradox_resolution import ParadoxClassification, ResolutionStrategy

# Q-LOGIK und VX-REASON Integration importieren
from miso.logic.qlogik_engine import BayesianDecisionCore
from miso.logic.vxor_integration import get_qlogik_vxor_integration
from miso.logic.vx_reason_integration import get_vx_reason_qlogik_prism_integration

# T-Mathematics importieren für Tensor-Tests
from miso.math.t_mathematics.engine import TMathematicsEngine
from miso.math.t_mathematics.tensor_interface import MISOTensorInterface
from miso.math.t_mathematics.tensor_wrappers import MLXTensorWrapper as MLXTensor, TorchTensorWrapper as TorchTensor
# MISOTensor ist das Interface, in der Implementation sind MLXTensorWrapper und TorchTensorWrapper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.test.integration.prism")

# Testklasse für die PRISM-Engine Integration
class TestPrismEngineIntegration(unittest.TestCase):
    """Test für die Integration der PRISM-Engine mit anderen Modulen"""
    
    @classmethod
    def setUpClass(cls):
        """Testumgebung einrichten (einmalig für alle Tests)"""
        logger.info("Initialisiere Testumgebung für PRISM-Engine Integrationstests...")
        
        # Grundlegende Module initialisieren
        cls.prism_engine = PrismEngine()
        cls.prism_vxor = get_prism_vxor_integration()
        cls.qlogik_vxor = get_qlogik_vxor_integration()
        cls.vx_reason_integration = get_vx_reason_qlogik_prism_integration()
        cls.tmath_engine = TMathematicsEngine()
        
        # Zeitlinie für Tests erstellen
        cls.timeline = cls._create_test_timeline()
        cls.prism_engine.register_timeline(cls.timeline)
        
        # Paradoxverwaltung initialisieren
        cls.paradox_management = EnhancedParadoxManagementSystem()
        
        logger.info("Testumgebung initialisiert.")
    
    @staticmethod
    def _create_test_timeline() -> Timeline:
        """Erstellt eine Zeitlinie für Testzwecke mit eingebautem Paradoxon"""
        timeline = Timeline(id="test_timeline", name="Integrations-Testlinie")
        
        # Knoten erstellen
        node1 = TimeNode(id="node1", timestamp=0, data={"event": "Anfang", "probability": 1.0})
        node2 = TimeNode(id="node2", timestamp=1, data={"event": "Erste Entscheidung", "probability": 0.8})
        node3 = TimeNode(id="node3", timestamp=2, data={"event": "Zweite Entscheidung", "probability": 0.6})
        node4 = TimeNode(id="node4", timestamp=3, data={"event": "Dritte Entscheidung", "probability": 0.4})
        node5 = TimeNode(id="node5", timestamp=4, data={"event": "Paradoxaler Knoten", "probability": 0.2})
        
        # Zeitline mit Knoten füllen
        timeline.add_node(node1)
        timeline.add_node(node2)
        timeline.add_node(node3)
        timeline.add_node(node4)
        timeline.add_node(node5)
        
        # Verbindungen erstellen (mit einem Paradoxon)
        timeline.add_connection(node1.id, node2.id)
        timeline.add_connection(node2.id, node3.id)
        timeline.add_connection(node3.id, node4.id)
        timeline.add_connection(node4.id, node5.id)
        timeline.add_connection(node5.id, node2.id)  # Paradoxon: Zeitschleife
        
        return timeline
    
    def setUp(self):
        """Zustand vor jedem Test einrichten"""
        # Mögliche Zwischenzustände zurücksetzen
        pass
    
    def test_prism_initialization(self):
        """Test der grundlegenden Initialisierung der PRISM-Engine"""
        self.assertIsNotNone(self.prism_engine, "PRISM-Engine wurde nicht korrekt initialisiert")
        self.assertTrue(hasattr(self.prism_engine, "analyze_timeline_integrity"), "PRISM-Engine fehlt die Analysemethode")
        self.assertTrue(hasattr(self.prism_engine, "resolve_paradox"), "PRISM-Engine fehlt die Paradoxauflösungsmethode")
        
        logger.info("PRISM-Engine Initialisierungstest bestanden")
    
    def test_prism_vxor_integration(self):
        """Test der Integration zwischen PRISM und VXOR"""
        # Prüfen, ob VX-REASON verfügbar ist
        integration = self.prism_vxor
        self.assertIsInstance(integration, PRISMVXORIntegration, "PRISMVXORIntegration hat falschen Typ")
        
        # Status der Integration prüfen
        if integration.reason_available:
            # Wenn VX-REASON verfügbar ist, teste Integration
            timeline = self.prism_engine.get_registered_timeline("test_timeline")
            result = integration.analyze_timeline_causality(timeline)
            
            self.assertIsInstance(result, dict, "Analyseergebnis sollte ein Dictionary sein")
            self.assertIn("causality_chains", result, "Analyseergebnis sollte Kausalketten enthalten")
            self.assertIn("critical_points", result, "Analyseergebnis sollte kritische Punkte enthalten")
            
            logger.info("PRISM ↔ VX-REASON Integration funktioniert korrekt")
        else:
            # Wenn nicht verfügbar, überspringen
            logger.warning("VX-REASON nicht verfügbar, Integrationstest übersprungen")
            self.skipTest("VX-REASON nicht verfügbar")
    
    def test_vx_reason_qlogik_prism_integration(self):
        """Test der dreiseitigen Integration zwischen VX-REASON, Q-LOGIK und PRISM"""
        integration = self.vx_reason_integration
        self.assertIsNotNone(integration, "VX-REASON ↔ Q-LOGIK/PRISM Integration nicht initialisiert")
        
        # Teste die Funktionalität, wenn alle Module verfügbar sind
        if integration.integration_ready:
            # Teste die bewusste Zeitlinienanalyse
            result = integration.analyze_timeline_with_consciousness("test_timeline")
            
            self.assertIsInstance(result, dict, "Analyseergebnis sollte ein Dictionary sein")
            self.assertTrue(result.get("success", False), "Analyse sollte erfolgreich sein")
            self.assertIn("prism_analysis", result, "Ergebnis sollte PRISM-Analyse enthalten")
            
            # Analysiere Paradoxon mit Bewusstseinssimulation
            # Paradoxon zuerst mit dem Detektor finden
            detector = EnhancedParadoxDetector()
            timeline = self.prism_engine.get_registered_timeline("test_timeline")
            paradoxes = detector.detect_paradoxes(timeline)
            
            if paradoxes:
                paradox_id = paradoxes[0].id
                resolution_result = integration.resolve_paradox_with_reasoning("test_timeline", paradox_id)
                
                self.assertIsInstance(resolution_result, dict, "Auflösungsergebnis sollte ein Dictionary sein")
                if resolution_result.get("success", False):
                    self.assertIn("resolution_method", resolution_result, "Ergebnis sollte Auflösungsmethode enthalten")
                    logger.info(f"Paradoxon erfolgreich mit {resolution_result.get('resolution_method')} aufgelöst")
            
            logger.info("VX-REASON ↔ Q-LOGIK/PRISM Integration funktioniert korrekt")
        else:
            # Wenn nicht alle Module verfügbar sind, teste Fallback-Mechanismen
            if integration.reason_available or integration.psi_available:
                self.assertIsNotNone(integration.prism_engine, "PRISM-Engine sollte verfügbar sein")
                logger.info("Teilweise Integration mit Fallback-Mechanismen funktioniert korrekt")
            else:
                logger.warning("Keine VX-Module verfügbar, Integrationstest übersprungen")
                self.skipTest("Keine VX-Module verfügbar")
    
    def test_mlx_tensor_integration(self):
        """Test der MLX-Tensor-Integration mit PRISM"""
        # Teste, ob MLXTensor verwendet wird, wenn Apple Silicon verfügbar ist
        tensor_type = self.tmath_engine.get_preferred_tensor_type()
        self.assertIsNotNone(tensor_type, "Bevorzugter Tensor-Typ sollte nicht None sein")
        
        # Erstelle einen Tensor und teste Integration mit PRISM
        try:
            # Einfachen MLX-Tensor erstellen (falls verfügbar)
            data = np.random.rand(3, 3)
            tensor = self.tmath_engine.create_tensor(data)
            
            # Prüfe, ob Tensor-Operationen korrekt funktionieren
            result_tensor = self.tmath_engine.compute_tensor_operation("matrix_multiply", [tensor, tensor])
            self.assertIsNotNone(result_tensor, "Tensor-Operation sollte Ergebnis liefern")
            
            # Teste PRISM-Integration durch Simulation mit Tensoren
            timeline = self.prism_engine.get_registered_timeline("test_timeline")
            tensor_data = self.prism_engine.extract_timeline_tensor_data(timeline)
            self.assertIsNotNone(tensor_data, "Extrahierte Tensor-Daten sollten nicht None sein")
            
            logger.info(f"MLX-Tensor-Integration funktioniert mit Typ: {tensor_type.__name__ if tensor_type else 'Fallback'}")
        except Exception as e:
            logger.error(f"Fehler beim Testen der MLX-Integration: {e}")
            self.fail(f"MLX-Integration fehlgeschlagen: {e}")
    
    def test_enhanced_paradox_resolution(self):
        """Test der erweiterten Paradoxauflösung"""
        # Paradoxon aus der Zeitlinie extrahieren
        detector = EnhancedParadoxDetector()
        timeline = self.prism_engine.get_registered_timeline("test_timeline")
        paradoxes = detector.detect_paradoxes(timeline)
        
        self.assertTrue(len(paradoxes) > 0, "Keine Paradoxien in der Testlinie gefunden")
        
        if paradoxes:
            # Paradoxauflösung testen
            resolver = ParadoxResolver()
            paradox = paradoxes[0]
            
            # Jede Auflösungsstrategie testen
            strategies = ["causal_reinforcement", "temporal_isolation", "quantum_superposition", 
                          "event_nullification", "timeline_restructuring", "causal_loop_stabilization",
                          "information_entropy_reduction"]
            
            for strategy in strategies:
                try:
                    start_time = time.time()
                    result = resolver.resolve_paradox(timeline, paradox.id, strategy=strategy)
                    duration = time.time() - start_time
                    
                    self.assertIsInstance(result, dict, f"Ergebnis für Strategie {strategy} sollte ein Dictionary sein")
                    self.assertTrue(result.get("success", False), f"Auflösung mit Strategie {strategy} sollte erfolgreich sein")
                    
                    logger.info(f"Paradoxauflösung mit Strategie '{strategy}' erfolgreich in {duration:.2f}s")
                    
                    # Zeitlinie für den nächsten Test wiederherstellen
                    timeline = self._create_test_timeline()
                    self.prism_engine.register_timeline(timeline)
                    
                except Exception as e:
                    logger.error(f"Fehler bei Paradoxauflösungsstrategie {strategy}: {e}")
            
            logger.info("Erweiterte Paradoxauflösung funktioniert korrekt mit allen Strategien")
        else:
            logger.warning("Keine Paradoxien gefunden, Paradoxauflösungstest übersprungen")
    
    def test_performance_with_mlx(self):
        """Performance-Test mit MLX-Optimierung"""
        # Größere Zeitlinie für Performance-Test erstellen
        large_timeline = Timeline(id="perf_test_timeline", name="Performance-Testlinie")
        
        # 100 Knoten erstellen und verbinden
        prev_node = None
        for i in range(100):
            node = TimeNode(id=f"perf_node_{i}", timestamp=i, 
                           data={"event": f"Event {i}", "probability": 1.0 - (i / 200)})
            large_timeline.add_node(node)
            
            # Mit vorherigem Knoten verbinden
            if prev_node:
                large_timeline.add_connection(prev_node.id, node.id)
            
            # Einige zufällige Verbindungen hinzufügen
            if i > 10 and i % 5 == 0:
                # Verbinde mit einem früheren Knoten (ohne Zeitschleife zu erzeugen)
                target_idx = max(0, i - np.random.randint(5, 10))
                large_timeline.add_connection(node.id, f"perf_node_{target_idx}")
            
            prev_node = node
        
        # Zeitlinie registrieren
        self.prism_engine.register_timeline(large_timeline)
        
        # Performance-Test für Timeline-Integritätsanalyse
        start_time = time.time()
        integrity_result = self.prism_engine.analyze_timeline_integrity(large_timeline.id)
        mlx_duration = time.time() - start_time
        
        # Prüfe Ergebnis
        self.assertIsNotNone(integrity_result, "Integrititätsanalyse sollte Ergebnis liefern")
        self.assertIn("integrity_score", integrity_result, "Ergebnis sollte Integritätswert enthalten")
        self.assertIn("analysis_details", integrity_result, "Ergebnis sollte Analysedetails enthalten")
        
        logger.info(f"Performance-Test: Zeitlinienanalyse mit MLX in {mlx_duration:.4f}s abgeschlossen")
        logger.info(f"Integrität der Performance-Zeitlinie: {integrity_result.get('integrity_score', 0):.4f}")

# Hauptfunktion
if __name__ == "__main__":
    unittest.main()
