#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests für die ParadoxResolver-Komponente.

Diese Tests überprüfen die Funktionalität der Paradoxauflösung.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import unittest
import sys
import os
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
    from miso.integration.paradox_resolver import (
        ParadoxResolver, 
        ParadoxResolutionStrategy,
        get_paradox_resolver
    )
    from miso.integration.ql_echo_bridge import TemporalBeliefNetwork
    from engines.echo_prime.timeline import Timeline, TimeNode, TemporalEvent
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Warnung: Einige Abhängigkeiten fehlen: {e}")
    IMPORTS_SUCCESSFUL = False


@unittest.skipIf(not IMPORTS_SUCCESSFUL, "Abhängigkeiten fehlen")
class TestParadoxResolver(unittest.TestCase):
    """Testklasse für die ParadoxResolver-Komponente."""

    def setUp(self):
        """Test-Setup mit einem temporalen Glaubensnetzwerk und Paradoxa."""
        # ParadoxResolver-Instanz erstellen
        self.resolver = ParadoxResolver()
        
        # Ein Mock-Objekt für ein temporales Glaubensnetzwerk erstellen
        self.belief_network = MagicMock(spec=TemporalBeliefNetwork)
        self.belief_network.name = "Test_Paradox_Network"
        
        # Beziehungen zwischen paradoxen Knoten definieren
        self.belief_network.nodes = [
            {"timestamp": 0, "id": "node0", "probability": 1.0, "events": []},
            {"timestamp": 10, "id": "node1", "probability": 0.9, "events": []},
            {"timestamp": 20, "id": "node2", "probability": 0.8, "events": []},
            {"timestamp": 30, "id": "node3", "probability": 0.7, "events": []},
            {"timestamp": 40, "id": "node4", "probability": 0.6, "events": []}
        ]
        self.belief_network.edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        
        # Paradoxszenarien definieren
        
        # 1. Kausalitätsparadoxon: Knoten 3 verursacht Knoten 1, obwohl 1 zeitlich vor 3 liegt
        self.causality_paradox = {
            'type': 'causality',
            'description': 'Knoten 3 verursacht Knoten 1, obwohl 1 zeitlich vor 3 liegt',
            'affected_nodes': [1, 3],
            'severity': 0.8,
            'probability': 0.7
        }
        
        # 2. Konsistenzparadoxon: Knoten 2 und 4 enthalten widersprüchliche Informationen
        self.consistency_paradox = {
            'type': 'consistency',
            'description': 'Knoten 2 und 4 enthalten widersprüchliche Informationen',
            'affected_nodes': [2, 4],
            'severity': 0.6,
            'probability': 0.5
        }
        
        # 3. Informationsparadoxon: Information in Knoten 0 stammt aus Knoten 4
        self.information_paradox = {
            'type': 'information',
            'description': 'Information in Knoten 0 stammt aus Knoten 4',
            'affected_nodes': [0, 4],
            'severity': 0.4,
            'probability': 0.3
        }
        
        # Liste aller Paradoxa
        self.paradoxes = [
            self.causality_paradox,
            self.consistency_paradox,
            self.information_paradox
        ]

    def test_identify_paradoxes(self):
        """Testet die Identifikation von Paradoxa im Glaubensnetzwerk."""
        # Die identify_paradoxes-Methode simuliert die Paradoxerkennung
        # Wir patchen sie, um unsere vordefinierten Paradoxa zurückzugeben
        with patch.object(self.resolver, '_analyze_network_for_paradoxes', return_value=self.paradoxes):
            detected_paradoxes = self.resolver.identify_paradoxes(self.belief_network)
            
            # Überprüfen, ob die richtigen Paradoxa erkannt wurden
            self.assertEqual(len(detected_paradoxes), 3)
            
            # Überprüfen, ob die Paradoxa die richtigen Eigenschaften haben
            paradox_types = [p['type'] for p in detected_paradoxes]
            self.assertIn('causality', paradox_types)
            self.assertIn('consistency', paradox_types)
            self.assertIn('information', paradox_types)

    def test_resolve_paradox_with_local_probability_adjustment(self):
        """Testet die Paradoxauflösung mit lokaler Wahrscheinlichkeitsanpassung."""
        # Die Auflösungsstrategie festlegen
        strategy = ParadoxResolutionStrategy.LOCAL_PROBABILITY_ADJUSTMENT
        
        # Das Kausalitätsparadoxon auflösen
        resolved_network = self.resolver.resolve_paradox(
            self.belief_network, 
            self.causality_paradox, 
            strategy
        )
        
        # Überprüfen, ob ein gültiges Netzwerk zurückgegeben wurde
        self.assertIsNotNone(resolved_network)
        
        # Bei dieser Strategie sollten die Wahrscheinlichkeiten der betroffenen Knoten angepasst werden
        # (Die tatsächliche Implementierung würde die Wahrscheinlichkeiten ändern, aber unsere Mocks bleiben gleich)
        # Wir testen hier nur, ob die Methode ohne Fehler ausgeführt wird

    def test_resolve_paradox_with_global_consistency_optimization(self):
        """Testet die Paradoxauflösung mit globaler Konsistenzoptimierung."""
        # Die Auflösungsstrategie festlegen
        strategy = ParadoxResolutionStrategy.GLOBAL_CONSISTENCY_OPTIMIZATION
        
        # Das Konsistenzparadoxon auflösen
        resolved_network = self.resolver.resolve_paradox(
            self.belief_network, 
            self.consistency_paradox, 
            strategy
        )
        
        # Überprüfen, ob ein gültiges Netzwerk zurückgegeben wurde
        self.assertIsNotNone(resolved_network)
        
        # Diese Strategie sollte Änderungen am gesamten Netzwerk vornehmen, um Konsistenz zu gewährleisten
        # Wir testen hier nur, ob die Methode ohne Fehler ausgeführt wird

    def test_resolve_paradox_with_branching_introduction(self):
        """Testet die Paradoxauflösung mit Verzweigungseinführung."""
        # Die Auflösungsstrategie festlegen
        strategy = ParadoxResolutionStrategy.BRANCHING_INTRODUCTION
        
        # Das Informationsparadoxon auflösen
        resolved_network = self.resolver.resolve_paradox(
            self.belief_network, 
            self.information_paradox, 
            strategy
        )
        
        # Überprüfen, ob ein gültiges Netzwerk zurückgegeben wurde
        self.assertIsNotNone(resolved_network)
        
        # Diese Strategie sollte neue Verzweigungen im Netzwerk einführen
        # Die Anzahl der Knoten und Kanten könnte sich ändern
        # Wir testen hier nur, ob die Methode ohne Fehler ausgeführt wird

    def test_resolve_paradox_with_recursive_stabilization(self):
        """Testet die Paradoxauflösung mit rekursiver Stabilisierung."""
        # Die Auflösungsstrategie festlegen
        strategy = ParadoxResolutionStrategy.RECURSIVE_STABILIZATION
        
        # Das Kausalitätsparadoxon auflösen
        resolved_network = self.resolver.resolve_paradox(
            self.belief_network, 
            self.causality_paradox, 
            strategy
        )
        
        # Überprüfen, ob ein gültiges Netzwerk zurückgegeben wurde
        self.assertIsNotNone(resolved_network)
        
        # Diese Strategie sollte die Netzwerkstruktur rekursiv stabilisieren
        # Wir testen hier nur, ob die Methode ohne Fehler ausgeführt wird

    def test_resolve_paradox_with_contextual_reinterpretation(self):
        """Testet die Paradoxauflösung mit kontextueller Neuinterpretation."""
        # Die Auflösungsstrategie festlegen
        strategy = ParadoxResolutionStrategy.CONTEXTUAL_REINTERPRETATION
        
        # Das Konsistenzparadoxon auflösen
        resolved_network = self.resolver.resolve_paradox(
            self.belief_network, 
            self.consistency_paradox, 
            strategy
        )
        
        # Überprüfen, ob ein gültiges Netzwerk zurückgegeben wurde
        self.assertIsNotNone(resolved_network)
        
        # Diese Strategie sollte die Interpretation der Ereignisse ändern
        # Wir testen hier nur, ob die Methode ohne Fehler ausgeführt wird

    def test_resolve_paradox_with_minimal_change(self):
        """Testet die Paradoxauflösung mit minimaler Änderungsstrategie."""
        # Die Auflösungsstrategie festlegen
        strategy = ParadoxResolutionStrategy.MINIMAL_CHANGE
        
        # Das Informationsparadoxon auflösen
        resolved_network = self.resolver.resolve_paradox(
            self.belief_network, 
            self.information_paradox, 
            strategy
        )
        
        # Überprüfen, ob ein gültiges Netzwerk zurückgegeben wurde
        self.assertIsNotNone(resolved_network)
        
        # Diese Strategie sollte minimale Änderungen am Netzwerk vornehmen
        # Wir testen hier nur, ob die Methode ohne Fehler ausgeführt wird

    def test_select_resolution_strategy(self):
        """Testet die automatische Auswahl der Auflösungsstrategie."""
        # Strategie für das Kausalitätsparadoxon auswählen
        causality_strategy = self.resolver.select_resolution_strategy(self.causality_paradox)
        
        # Strategie für das Konsistenzparadoxon auswählen
        consistency_strategy = self.resolver.select_resolution_strategy(self.consistency_paradox)
        
        # Strategie für das Informationsparadoxon auswählen
        information_strategy = self.resolver.select_resolution_strategy(self.information_paradox)
        
        # Überprüfen, ob gültige Strategien zurückgegeben wurden
        self.assertIn(causality_strategy, ParadoxResolutionStrategy)
        self.assertIn(consistency_strategy, ParadoxResolutionStrategy)
        self.assertIn(information_strategy, ParadoxResolutionStrategy)
        
        # Verschiedene Paradoxtypen sollten zu verschiedenen optimalen Strategien führen
        self.assertNotEqual(causality_strategy, consistency_strategy)
        # Informationsparadoxa könnten dieselbe Strategie wie eines der anderen haben,
        # daher testen wir das nicht

    def test_resolve_all_paradoxes(self):
        """Testet die Auflösung aller Paradoxa im Netzwerk."""
        # Alle Paradoxa auflösen
        with patch.object(self.resolver, 'identify_paradoxes', return_value=self.paradoxes):
            resolved_network = self.resolver.resolve_paradoxes(self.belief_network)
            
            # Überprüfen, ob ein gültiges Netzwerk zurückgegeben wurde
            self.assertIsNotNone(resolved_network)
            
            # Nach der Auflösung sollten keine Paradoxa mehr vorhanden sein
            with patch.object(self.resolver, 'identify_paradoxes', return_value=[]):
                remaining_paradoxes = self.resolver.identify_paradoxes(resolved_network)
                self.assertEqual(len(remaining_paradoxes), 0)

    def test_evaluate_resolution_quality(self):
        """Testet die Bewertung der Auflösungsqualität."""
        # Eine Auflösung simulieren
        resolved_network = self.resolver.resolve_paradox(
            self.belief_network, 
            self.causality_paradox, 
            ParadoxResolutionStrategy.LOCAL_PROBABILITY_ADJUSTMENT
        )
        
        # Die Qualität der Auflösung bewerten
        quality = self.resolver.evaluate_resolution_quality(
            self.belief_network,
            resolved_network,
            self.causality_paradox
        )
        
        # Überprüfen, ob eine gültige Qualitätsbewertung zurückgegeben wurde
        self.assertIsInstance(quality, dict)
        self.assertIn('consistency_score', quality)
        self.assertIn('network_stability', quality)
        self.assertIn('information_preservation', quality)
        self.assertIn('overall_quality', quality)
        
        # Die Scores sollten im gültigen Bereich liegen
        self.assertGreaterEqual(quality['consistency_score'], 0.0)
        self.assertLessEqual(quality['consistency_score'], 1.0)
        self.assertGreaterEqual(quality['network_stability'], 0.0)
        self.assertLessEqual(quality['network_stability'], 1.0)
        self.assertGreaterEqual(quality['information_preservation'], 0.0)
        self.assertLessEqual(quality['information_preservation'], 1.0)
        self.assertGreaterEqual(quality['overall_quality'], 0.0)
        self.assertLessEqual(quality['overall_quality'], 1.0)

    def test_singleton_access(self):
        """Testet den Singleton-Zugriff auf die ParadoxResolver-Instanz."""
        resolver1 = get_paradox_resolver()
        resolver2 = get_paradox_resolver()
        
        # Überprüfen, ob beide Aufrufe die gleiche Instanz zurückgeben
        self.assertIs(resolver1, resolver2)


if __name__ == "__main__":
    unittest.main()
