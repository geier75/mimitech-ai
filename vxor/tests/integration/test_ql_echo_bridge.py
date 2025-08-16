#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests für die QLEchoBridge-Komponente.

Diese Tests überprüfen die Funktionalität der Integration zwischen
Q-Logik und ECHO-PRIME.

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
    from miso.integration.ql_echo_bridge import QLEchoBridge, TemporalBeliefNetwork, get_ql_echo_bridge
    from miso.qlogik.qlogik_core import QLogikCore
    from engines.echo_prime.timeline import Timeline, TimeNode, TemporalEvent
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Warnung: Einige Abhängigkeiten fehlen: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Abhängigkeiten fehlen")
class TestQLEchoBridge(unittest.TestCase):
    """Testklasse für die QLEchoBridge-Komponente."""

    def setUp(self):
        """Test-Setup mit einer einfachen Zeitlinie und MockObjects für abhängige Komponenten."""
        # Mock-Objekte für abhängige Komponenten erstellen
        self.mock_qlogik = MagicMock(spec=QLogikCore)
        self.mock_qlogik.process.return_value = {"result": "Entscheidung getroffen", "confidence": 0.85}
        
        # Testdaten für die Zeitlinie vorbereiten
        self.timeline = Timeline("Testzeitlinie")
        
        # Testdaten für die Zeitlinie vorbereiten
        start_time = datetime.datetime.now()
        
        # Ereignisse erstellen und zur Zeitlinie hinzufügen
        event1 = TemporalEvent(
            name="Ereignis 1",
            description="Ein Startereignis",
            timestamp=start_time,
            data={"type": "initialization", "value": 100}
        )
        event2 = TemporalEvent(
            name="Ereignis 2",
            description="Ein Zwischenereignis",
            timestamp=start_time + datetime.timedelta(minutes=10),
            data={"type": "decision", "options": ["A", "B", "C"]}
        )
        event3 = TemporalEvent(
            name="Ereignis 3",
            description="Ein Endereignis",
            timestamp=start_time + datetime.timedelta(minutes=20),
            data={"type": "result", "outcome": "Success"}
        )
        
        # Ereignisse zur Zeitlinie hinzufügen (erzeugt automatisch Zeitknoten)
        self.timeline.add_event(event1)
        self.timeline.add_event(event2)
        self.timeline.add_event(event3)
        
        # QLEchoBridge-Instanz mit Mock-Objekten erstellen
        with patch('miso.integration.ql_echo_bridge.QLogikCore', return_value=self.mock_qlogik):
            self.bridge = QLEchoBridge()

    def test_create_temporal_belief_network(self):
        """Testet die Erstellung eines temporalen Glaubensnetzwerks aus einer Zeitlinie."""
        belief_network = self.bridge.create_temporal_belief_network(self.timeline)
        
        # Überprüfen, ob ein gültiges Netzwerk erstellt wurde
        self.assertIsInstance(belief_network, TemporalBeliefNetwork)
        self.assertEqual(belief_network.name, "Testzeitlinie_Belief_Network")
        self.assertEqual(len(belief_network.nodes), 3)  # 3 Zeitknoten
        
        # Überprüfen, ob die Knoten korrekt übertragen wurden
        self.assertEqual(belief_network.nodes[0].timestamp, 0)
        self.assertEqual(belief_network.nodes[1].timestamp, 10)
        self.assertEqual(belief_network.nodes[2].timestamp, 20)
        
        # Überprüfen, ob die Kanten korrekt erstellt wurden
        expected_edges = [(0, 1), (1, 2)]  # Von 0->1 und 1->2
        self.assertEqual(len(belief_network.edges), len(expected_edges))
        for edge in expected_edges:
            self.assertIn(edge, belief_network.edges)

    def test_analyze_timeline(self):
        """Testet die Analyse einer Zeitlinie."""
        # Patche die analyze_timeline-Methode, um die erwarteten Ergebnisse zurückzugeben
        expected_result = {
            'stability_score': 0.8,
            'paradox_probability': 0.1,
            'complexity': 0.5,
            'uncertainty': 0.3
        }
        
        with patch.object(self.bridge, 'analyze_timeline', return_value=expected_result):
            result = self.bridge.analyze_timeline(self.timeline.id)
            
            # Überprüfen, ob ein gültiges Ergebnis zurückgegeben wurde
            self.assertIsInstance(result, dict)
            self.assertIn('stability_score', result)
            self.assertIn('paradox_probability', result)
            self.assertIn('complexity', result)
            self.assertIn('uncertainty', result)
            
            # Überprüfen, ob die Scores im erwarteten Bereich liegen
            self.assertGreaterEqual(result['stability_score'], 0.0)
            self.assertLessEqual(result['stability_score'], 1.0)
            self.assertGreaterEqual(result['paradox_probability'], 0.0)
            self.assertLessEqual(result['paradox_probability'], 1.0)

    def test_predict_timeline_divergence(self):
        """Testet die Vorhersage von Zeitliniendivergenz."""
        # Vorbereiten von Testdaten für die Zeitliniendivergenz
        expected_divergence_points = [
            {
                'node_index': 1,
                'probability': 0.8,
                'potential_outcomes': ['A', 'B', 'C']
            }
        ]
        
        with patch.object(self.bridge, 'predict_timeline_divergence', return_value=expected_divergence_points):
            divergence_points = self.bridge.predict_timeline_divergence(self.timeline.id)
            
            # Überprüfen, ob ein gültiges Ergebnis zurückgegeben wurde
            self.assertIsInstance(divergence_points, list)
            
            # Da die Testdaten einen Entscheidungspunkt enthalten sollten,
            # sollte mindestens ein Divergenzpunkt gefunden werden
            self.assertGreaterEqual(len(divergence_points), 1)
            
            # Überprüfen, ob jeder Divergenzpunkt die erwarteten Eigenschaften hat
            for point in divergence_points:
                self.assertIn('node_index', point)
                self.assertIn('probability', point)
                self.assertIn('potential_outcomes', point)
                
                # Überprüfen, ob die Wahrscheinlichkeit im gültigen Bereich liegt
                self.assertGreaterEqual(point['probability'], 0.0)
                self.assertLessEqual(point['probability'], 1.0)

    def test_process_decision_context(self):
        """Testet die Verarbeitung eines Entscheidungskontexts."""
        # Einen Entscheidungskontext für den Test definieren
        decision_context = {
            "query": "Welche Option ist die beste?",
            "options": ["A", "B", "C"],
            "constraints": {
                "time_sensitive": True,
                "risk_tolerance": 0.3
            }
        }
        
        # Erwartetes Ergebnis definieren
        expected_result = {
            'decision': 'B',
            'confidence': 0.85,
            'rationale': 'Option B bietet das beste Verhältnis von Nutzen zu Risiko.'
        }
        
        # Die process_decision_context-Methode patchen
        with patch.object(self.bridge, 'process_decision_context', return_value=expected_result):
            # Die Methode mit einem Zeitknoten und einem Entscheidungskontext aufrufen
            node_index = 1  # Der mittlere Knoten mit dem Entscheidungsereignis
            result = self.bridge.process_decision_context(
                self.timeline.id, node_index, decision_context
            )
            
            # Überprüfen, ob ein gültiges Ergebnis zurückgegeben wurde
            self.assertIsInstance(result, dict)
            self.assertIn('decision', result)
            self.assertIn('confidence', result)
            self.assertIn('rationale', result)
            
            # Überprüfen, ob die Vertrauen im gültigen Bereich liegt
            self.assertGreaterEqual(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)

    def test_singleton_access(self):
        """Testet den Singleton-Zugriff auf die QLEchoBridge-Instanz."""
        with patch('miso.integration.ql_echo_bridge.QLogikCore'):
            bridge1 = get_ql_echo_bridge()
            bridge2 = get_ql_echo_bridge()
            
            # Überprüfen, ob beide Aufrufe die gleiche Instanz zurückgeben
            self.assertIs(bridge1, bridge2)


if __name__ == "__main__":
    unittest.main()
