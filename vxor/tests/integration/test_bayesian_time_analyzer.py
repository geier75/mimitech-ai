#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests für die BayesianTimeNodeAnalyzer-Komponente.

Diese Tests überprüfen die Funktionalität der Bayes'schen Analyse von Zeitknoten.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import unittest
import sys
import os
import datetime
import numpy as np
from unittest.mock import patch, MagicMock

# Pfad zum MISO-Paket hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ZTM-Protokoll-Initialisierung (aktiviert für Integrationstests)
os.environ['MISO_ZTM_MODE'] = '1'
os.environ['MISO_ZTM_LOG_LEVEL'] = 'DEBUG'
# Stelle sicher, dass ZTM-Logs in das richtige Verzeichnis geschrieben werden
ztm_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(ztm_log_dir, exist_ok=True)

# Versuche, die erforderlichen Module zu importieren
try:
    from miso.integration.bayesian_time_analyzer import (
        BayesianTimeNodeAnalyzer, 
        BayesianTimeNodeResult,
        get_bayesian_time_analyzer
    )
    from engines.echo_prime.timeline import TimeNode, TemporalEvent
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Warnung: Einige Abhängigkeiten fehlen: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Abhängigkeiten fehlen")
class TestBayesianTimeNodeAnalyzer(unittest.TestCase):
    """Testklasse für die BayesianTimeNodeAnalyzer-Komponente."""

    def setUp(self):
        """Test-Setup mit Testdaten für Zeitknoten."""
        # Analyzer-Instanz erstellen
        self.analyzer = BayesianTimeNodeAnalyzer()
        
        # Testdaten für Zeitknoten vorbereiten
        now = datetime.datetime.now()
        
        # Stabiler Knoten mit hoher Gewissheit
        self.stable_node = TimeNode(timestamp=now)
        self.stable_node.add_event(TemporalEvent(
            name="Stabiles Ereignis",
            description="Ein Ereignis mit hoher Gewissheit",
            timestamp=now,
            data={"type": "observation", "certainty": 0.95, "impact": 0.2}
        ))
        
        # Instabiler Knoten mit niedriger Gewissheit
        self.unstable_node = TimeNode(timestamp=now + datetime.timedelta(minutes=10))
        self.unstable_node.add_event(TemporalEvent(
            name="Instabiles Ereignis",
            description="Ein Ereignis mit niedriger Gewissheit",
            timestamp=now + datetime.timedelta(minutes=10),
            data={"type": "observation", "certainty": 0.3, "impact": 0.8}
        ))
        
        # Entscheidungsknoten mit mehreren Optionen
        self.decision_node = TimeNode(timestamp=now + datetime.timedelta(minutes=20))
        self.decision_node.add_event(TemporalEvent(
            name="Entscheidungsereignis",
            description="Ein Ereignis mit mehreren möglichen Ausgängen",
            timestamp=now + datetime.timedelta(minutes=20),
            data={
                "type": "decision", 
                "options": ["A", "B", "C"], 
                "probabilities": [0.2, 0.5, 0.3]
            }
        ))
        
        # Paradoxer Knoten mit widersprüchlichen Informationen
        self.paradox_node = TimeNode(timestamp=now + datetime.timedelta(minutes=30))
        self.paradox_node.add_event(TemporalEvent(
            name="Ereignis 1",
            description="Ein Ereignis mit Behauptung A",
            timestamp=now + datetime.timedelta(minutes=30),
            data={"type": "observation", "claim": "A", "certainty": 0.9}
        ))
        self.paradox_node.add_event(TemporalEvent(
            name="Ereignis 2",
            description="Ein Ereignis mit Behauptung nicht-A",
            timestamp=now + datetime.timedelta(minutes=30, seconds=1),
            data={"type": "observation", "claim": "not_A", "certainty": 0.8}
        ))

    def test_compute_node_entropy(self):
        """Testet die Berechnung der Entropie eines Zeitknotens."""
        # Entropie für verschiedene Knotentypen berechnen
        stable_entropy = self.analyzer.compute_node_entropy(self.stable_node)
        unstable_entropy = self.analyzer.compute_node_entropy(self.unstable_node)
        decision_entropy = self.analyzer.compute_node_entropy(self.decision_node)
        paradox_entropy = self.analyzer.compute_node_entropy(self.paradox_node)
        
        # Der stabile Knoten sollte eine niedrige Entropie haben
        self.assertLess(stable_entropy, 0.5)
        
        # Der instabile Knoten sollte eine höhere Entropie haben
        self.assertGreater(unstable_entropy, stable_entropy)
        
        # Der Entscheidungsknoten sollte eine hohe Entropie haben
        self.assertGreater(decision_entropy, 0.5)
        
        # Der paradoxe Knoten sollte die höchste Entropie haben
        self.assertGreater(paradox_entropy, decision_entropy)

    def test_compute_node_stability(self):
        """Testet die Berechnung der Stabilität eines Zeitknotens."""
        # Stabilität für verschiedene Knotentypen berechnen
        stable_stability = self.analyzer.compute_node_stability(self.stable_node)
        unstable_stability = self.analyzer.compute_node_stability(self.unstable_node)
        decision_stability = self.analyzer.compute_node_stability(self.decision_node)
        paradox_stability = self.analyzer.compute_node_stability(self.paradox_node)
        
        # Der stabile Knoten sollte eine hohe Stabilität haben
        self.assertGreater(stable_stability, 0.7)
        
        # Der instabile Knoten sollte eine niedrigere Stabilität haben
        self.assertLess(unstable_stability, stable_stability)
        
        # Der Entscheidungsknoten sollte eine mittlere Stabilität haben
        self.assertGreater(decision_stability, 0.3)
        self.assertLess(decision_stability, 0.7)
        
        # Der paradoxe Knoten sollte die niedrigste Stabilität haben
        self.assertLess(paradox_stability, 0.3)

    def test_compute_paradox_probability(self):
        """Testet die Berechnung der Paradoxwahrscheinlichkeit eines Zeitknotens."""
        # Paradoxwahrscheinlichkeit für verschiedene Knotentypen berechnen
        stable_paradox = self.analyzer.compute_paradox_probability(self.stable_node)
        unstable_paradox = self.analyzer.compute_paradox_probability(self.unstable_node)
        decision_paradox = self.analyzer.compute_paradox_probability(self.decision_node)
        paradox_paradox = self.analyzer.compute_paradox_probability(self.paradox_node)
        
        # Der stabile Knoten sollte eine niedrige Paradoxwahrscheinlichkeit haben
        self.assertLess(stable_paradox, 0.1)
        
        # Der instabile Knoten sollte eine höhere Paradoxwahrscheinlichkeit haben
        self.assertGreater(unstable_paradox, stable_paradox)
        
        # Der Entscheidungsknoten sollte eine mittlere Paradoxwahrscheinlichkeit haben
        self.assertLess(decision_paradox, 0.5)
        
        # Der paradoxe Knoten sollte die höchste Paradoxwahrscheinlichkeit haben
        self.assertGreater(paradox_paradox, 0.7)

    def test_analyze_node(self):
        """Testet die vollständige Analyse eines Zeitknotens."""
        # Knoten analysieren
        stable_result = self.analyzer.analyze_node(self.stable_node)
        unstable_result = self.analyzer.analyze_node(self.unstable_node)
        decision_result = self.analyzer.analyze_node(self.decision_node)
        paradox_result = self.analyzer.analyze_node(self.paradox_node)
        
        # Überprüfen, ob die Ergebnisse den erwarteten Typ haben
        self.assertIsInstance(stable_result, BayesianTimeNodeResult)
        self.assertIsInstance(unstable_result, BayesianTimeNodeResult)
        self.assertIsInstance(decision_result, BayesianTimeNodeResult)
        self.assertIsInstance(paradox_result, BayesianTimeNodeResult)
        
        # Überprüfen, ob die Ergebnisse die erwarteten Eigenschaften haben
        
        # Stabiler Knoten
        self.assertGreater(stable_result.stability_score, 0.7)
        self.assertLess(stable_result.entropy, 0.5)
        self.assertLess(stable_result.paradox_probability, 0.1)
        
        # Instabiler Knoten
        self.assertLess(unstable_result.stability_score, stable_result.stability_score)
        self.assertGreater(unstable_result.entropy, stable_result.entropy)
        self.assertGreater(unstable_result.paradox_probability, stable_result.paradox_probability)
        
        # Entscheidungsknoten
        self.assertLess(decision_result.stability_score, stable_result.stability_score)
        self.assertGreater(decision_result.entropy, stable_result.entropy)
        self.assertGreater(decision_result.paradox_probability, stable_result.paradox_probability)
        
        # Paradoxer Knoten
        self.assertLess(paradox_result.stability_score, 0.3)
        self.assertGreater(paradox_result.entropy, 0.7)
        self.assertGreater(paradox_result.paradox_probability, 0.7)

    def test_singleton_access(self):
        """Testet den Singleton-Zugriff auf die BayesianTimeNodeAnalyzer-Instanz."""
        analyzer1 = get_bayesian_time_analyzer()
        analyzer2 = get_bayesian_time_analyzer()
        
        # Überprüfen, ob beide Aufrufe die gleiche Instanz zurückgeben
        self.assertIs(analyzer1, analyzer2)


if __name__ == "__main__":
    unittest.main()
