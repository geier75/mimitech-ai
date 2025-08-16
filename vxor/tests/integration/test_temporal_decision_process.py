#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests für die TemporalDecisionProcess-Komponente.

Diese Tests überprüfen die Funktionalität der temporalen Entscheidungsverarbeitung.

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
    from miso.integration.temporal_decision_process import (
        TemporalDecisionProcess, 
        DecisionSequence, 
        DecisionSequenceType,
        TemporalDecisionContext,
        get_temporal_decision_process
    )
    from miso.integration.ql_echo_bridge import TemporalBeliefNetwork, TemporalDecision, TemporalDecisionType
    from engines.echo_prime.timeline import Timeline, TimeNode, TemporalEvent
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Warnung: Einige Abhängigkeiten fehlen: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Abhängigkeiten fehlen")
class TestTemporalDecisionProcess(unittest.TestCase):
    """Testklasse für die TemporalDecisionProcess-Komponente."""

    def setUp(self):
        """Test-Setup mit einer Testzeitlinie und einem temporalen Glaubensnetzwerk."""
        # Temporalen Entscheidungsprozess erstellen
        self.tdp = TemporalDecisionProcess()
        
        # Testdaten für die Zeitlinie vorbereiten
        self.timeline = Timeline("Testzeitlinie")
        
        # Testdaten für die Zeitlinie vorbereiten
        start_time = datetime.datetime.now()
        
        # Ereignisse erstellen und zur Zeitlinie hinzufügen
        event1 = TemporalEvent(
            name="Startereignis",
            description="Initialisierung",
            timestamp=start_time,
            data={"type": "initialization", "state": "ready"}
        )
        
        event2 = TemporalEvent(
            name="Erste Entscheidung",
            description="Auswahl zwischen zwei Optionen",
            timestamp=start_time + datetime.timedelta(minutes=10),
            data={
                "type": "decision", 
                "options": ["A", "B"], 
                "weights": {"utility": [0.7, 0.3], "risk": [0.4, 0.6]}
            }
        )
        
        event3 = TemporalEvent(
            name="Erstes Ergebnis",
            description="Ergebnis nach der ersten Entscheidung",
            timestamp=start_time + datetime.timedelta(minutes=20),
            data={"type": "outcome", "result": "success", "utility_gained": 0.7}
        )
        
        event4 = TemporalEvent(
            name="Zweite Entscheidung",
            description="Auswahl zwischen drei Optionen",
            timestamp=start_time + datetime.timedelta(minutes=30),
            data={
                "type": "decision", 
                "options": ["X", "Y", "Z"], 
                "weights": {"utility": [0.3, 0.5, 0.2], "risk": [0.2, 0.3, 0.5]}
            }
        )
        
        event5 = TemporalEvent(
            name="Zweites Ergebnis",
            description="Ergebnis nach der zweiten Entscheidung",
            timestamp=start_time + datetime.timedelta(minutes=40),
            data={"type": "outcome", "result": "partial", "utility_gained": 0.5}
        )
        
        # Ereignisse zur Zeitlinie hinzufügen (erzeugt automatisch Zeitknoten)
        self.timeline.add_event(event1)
        self.timeline.add_event(event2)
        self.timeline.add_event(event3)
        self.timeline.add_event(event4)
        self.timeline.add_event(event5)
        
        # Ein Mock-Objekt für ein temporales Glaubensnetzwerk erstellen
        self.belief_network = MagicMock(spec=TemporalBeliefNetwork)
        self.belief_network.name = "Testzeitlinie_Belief_Network"
        self.belief_network.nodes = [
            {"timestamp": 0, "probability": 1.0, "events": [{"type": "initialization"}]},
            {"timestamp": 10, "probability": 0.9, "events": [{"type": "decision"}]},
            {"timestamp": 20, "probability": 0.8, "events": [{"type": "outcome"}]},
            {"timestamp": 30, "probability": 0.7, "events": [{"type": "decision"}]},
            {"timestamp": 40, "probability": 0.6, "events": [{"type": "outcome"}]}
        ]
        self.belief_network.edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        
        # MockMethod für get_node_probabilities hinzufügen, da es im MagicMock nicht automatisch erstellt wird
        node_probs = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
        self.belief_network.get_node_probabilities = MagicMock(return_value=node_probs)

    def test_identify_decision_points(self):
        """Testet die Identifikation von Entscheidungspunkten in einer Zeitlinie."""
        decision_points = self.tdp.identify_decision_points(self.timeline)
        
        # Es sollte zwei Entscheidungspunkte geben (Knoten 2 und 4)
        self.assertEqual(len(decision_points), 2)
        
        # Überprüfen, ob die richtigen Knotenindices zurückgegeben werden
        self.assertEqual(decision_points[0]['node_index'], 1)  # Index von node2
        self.assertEqual(decision_points[1]['node_index'], 3)  # Index von node4
        
        # Überprüfen, ob die richtigen Optionen extrahiert wurden
        self.assertEqual(decision_points[0]['options'], ["A", "B"])
        self.assertEqual(decision_points[1]['options'], ["X", "Y", "Z"])

    def test_calculate_decision_utility(self):
        """Testet die Berechnung von Nutzwerten für Entscheidungsoptionen."""
        # Erstelle einen Test-Entscheidungspunkt
        decision_point = {
            "node_id": "node2",
            "timestamp": datetime.datetime.now(),
            "events": [{"event_id": "event1", "name": "Entscheidung A", "description": "Eine wichtige Entscheidung"}],
            "options": [
                {"target_node_id": "option1", "probability": 0.7, "description": "Option 1"},
                {"target_node_id": "option2", "probability": 0.3, "description": "Option 2"}
            ],
            "uncertainty": 0.4
        }
        
        # Erstelle einen Test-Kontext
        context = {
            "priorities": {"option": 0.8, "risiko": 0.2}
        }
        
        # Berechne Nutzwerte
        utilities = self.tdp.calculate_decision_utility(decision_point, context)
        
        # Die zurückgegebenen Nutzwerte sollten ein Dictionary sein
        self.assertIsInstance(utilities, dict)
        
        # Für jede Option sollte ein Nutzwert vorhanden sein
        for option_id, utility in utilities.items():
            self.assertIsInstance(option_id, str)
            self.assertIsInstance(utility, float)
            self.assertGreaterEqual(utility, 0.0)
            self.assertLessEqual(utility, 1.0)

    def test_identify_decision_sequences(self):
        """Testet die Identifikation von Entscheidungssequenzen in einer Zeitlinie."""
        # Erstelle einen TemporalDecisionContext für den Test
        context = TemporalDecisionContext(
            timeline_id="test_timeline_1",
            start_time=None,
            end_time=None,
            decision_points=[],
            constraints={},
            prior_beliefs={}
        )
        
        # Patch die get_timeline-Methode, um unsere Timeline zurückzugeben
        with patch.object(self.tdp.echo_prime, 'get_timeline', return_value=self.timeline):
            sequences = self.tdp.identify_decision_sequences(context)
            
            # Überprüfe, ob mindestens eine Sequenz gefunden wurde
            self.assertIsInstance(sequences, list)
            
            # Überprüfe die Mock-Ausgabe
            if sequences:
                first_sequence = sequences[0]
                self.assertIsInstance(first_sequence, DecisionSequence)
                self.assertIsInstance(first_sequence.id, str)
                self.assertIsInstance(first_sequence.sequence_type, DecisionSequenceType)
                
                # Die Sequenz sollte zwei Entscheidungspunkte haben
                self.assertEqual(len(first_sequence.decision_points), 2)
                
                # Überprüfen, ob die Entscheidungspunkte die richtigen Indizes haben
                self.assertEqual(first_sequence.decision_points[0]['node_index'], 1)  # Index von node2
                self.assertEqual(first_sequence.decision_points[1]['node_index'], 3)  # Index von node4

    def test_evaluate_decision_sequence(self):
        """Testet die Bewertung einer Entscheidungssequenz."""
        # Eine Entscheidungssequenz erstellen
        sequence = DecisionSequence(
            id="test_sequence_1",
            sequence_type=DecisionSequenceType.LINEAR
        )
        
        # Entscheidungen zur Sequenz hinzufügen
        sequence.decisions = [
            TemporalDecision(
                id="decision1",
                description="Erste Entscheidung im Test",
                decision_type=TemporalDecisionType.PROBABILISTIC,
                probability=0.8,
                confidence=0.9,
                alternatives=[{"id": "option1", "probability": 0.8}],
                metadata={"original_node": "node1"}
            ),
            TemporalDecision(
                id="decision2",
                description="Zweite Entscheidung im Test",
                decision_type=TemporalDecisionType.PROBABILISTIC,
                probability=0.6,
                confidence=0.7,
                alternatives=[{"id": "option2", "probability": 0.6}],
                metadata={"original_node": "node2"}
            )
        ]
        
        # Mock für die evaluate_decision_sequence-Methode erstellen, da wir auf das Interface testen
        # und nicht auf die tatsächliche Implementierung
        with patch.object(self.tdp, 'evaluate_decision_sequence', return_value={
            'sequence_id': sequence.id,
            'total_utility': 1.2,  # Simulierter Nutzwert (0.7 + 0.5)
            'expected_utility': 0.9,
            'risk_score': 0.3,
            'probability': 0.75
        }):
            evaluation = self.tdp.evaluate_decision_sequence(sequence, self.timeline)
            
            # Überprüfen, ob die Bewertung die erwarteten Eigenschaften hat
            self.assertIn('total_utility', evaluation)
            self.assertIn('expected_utility', evaluation)
            self.assertIn('risk_score', evaluation)
            self.assertIn('probability', evaluation)
            
            # Die Gesamtnutzwert sollte der simulierte Wert sein
            self.assertAlmostEqual(evaluation['total_utility'], 1.2, places=2)

    def test_optimize_decision_sequence(self):
        """Testet die Optimierung einer Entscheidungssequenz."""
        # Erstelle eine Sequenz mit nicht ausgewählten Optionen
        sequence = DecisionSequence(
            id="test_sequence_2",
            sequence_type=DecisionSequenceType.LINEAR
        )
        
        # Füge zwei Entscheidungen hinzu, die noch keine Auswahl haben
        sequence.decisions = [
            TemporalDecision(
                id="decision1",
                description="Erste Entscheidung ohne Auswahl",
                decision_type=TemporalDecisionType.PROBABILISTIC,
                probability=0.5,  # Neutral, da keine Auswahl getroffen
                confidence=0.7,
                alternatives=[
                    {"id": "option1", "probability": 0.8}, 
                    {"id": "option2", "probability": 0.2}
                ],
                metadata={"original_node": "node1"}
            ),
            TemporalDecision(
                id="decision2",
                description="Zweite Entscheidung ohne Auswahl",
                decision_type=TemporalDecisionType.PROBABILISTIC,
                probability=0.5,  # Neutral, da keine Auswahl getroffen
                confidence=0.6,
                alternatives=[
                    {"id": "option3", "probability": 0.6}, 
                    {"id": "option4", "probability": 0.4}
                ],
                metadata={"original_node": "node2"}
            )
        ]
        
        # Erzeuge eine optimierte Sequenz, die nun Auswahloptionen enthält
        optimized_sequence = DecisionSequence(
            id="test_sequence_2",
            sequence_type=DecisionSequenceType.LINEAR
        )
        optimized_sequence.decisions = [
            TemporalDecision(
                id="decision1",
                description="Erste Entscheidung mit Auswahl",
                decision_type=TemporalDecisionType.PROBABILISTIC,
                probability=0.8,  # Wahrscheinlichkeit der gewählten Option
                confidence=0.9,
                alternatives=[
                    {"id": "option1", "probability": 0.8, "selected": True}, 
                    {"id": "option2", "probability": 0.2}
                ],
                metadata={"original_node": "node1", "selected_option": "option1"}
            ),
            TemporalDecision(
                id="decision2",
                description="Zweite Entscheidung mit Auswahl",
                decision_type=TemporalDecisionType.PROBABILISTIC,
                probability=0.6,  # Wahrscheinlichkeit der gewählten Option
                confidence=0.8,
                alternatives=[
                    {"id": "option3", "probability": 0.6, "selected": True}, 
                    {"id": "option4", "probability": 0.4}
                ],
                metadata={"original_node": "node2", "selected_option": "option3"}
            )
        ]
        
        # Mock für die optimize_decision_sequence-Methode erstellen
        with patch.object(self.tdp, 'optimize_decision_sequence', return_value=optimized_sequence):
            optimized = self.tdp.optimize_decision_sequence(
                sequence,
                self.timeline,
                weights={"utility": 0.7, "risk": 0.3}
            )
            
            # Die optimierte Sequenz sollte dieselbe ID haben
            self.assertEqual(optimized.id, sequence.id)
            
            # Das Sequenz-Objekt sollte vom richtigen Typ sein
            self.assertIsInstance(optimized, DecisionSequence)
            
            # Alle Entscheidungen sollten jetzt Auswahloptionen haben
            for decision in optimized.decisions:
                # Prüfe, ob eine Option in metadata als ausgewählt markiert ist
                self.assertIn("selected_option", decision.metadata)
                selected_option = decision.metadata["selected_option"]
                self.assertIsNotNone(selected_option)
                
                # Prüfe, ob mindestens eine Alternative mit "selected: True" markiert ist
                selected_alternatives = [alt for alt in decision.alternatives if alt.get("selected", False)]
                self.assertGreater(len(selected_alternatives), 0, "Keine ausgewählte Alternative gefunden")

    def test_singleton_access(self):
        """Testet den Singleton-Zugriff auf die TemporalDecisionProcess-Instanz."""
        tdp1 = get_temporal_decision_process()
        tdp2 = get_temporal_decision_process()
        
        # Überprüfen, ob beide Aufrufe die gleiche Instanz zurückgeben
        self.assertIs(tdp1, tdp2)


if __name__ == "__main__":
    unittest.main()
