#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-Suite für die ECHO-PRIME ↔ VX-CHRONOS Brücke

Diese Tests verifizieren die korrekte Integration zwischen dem ECHO-PRIME 
Zeitlinienmanagement und dem VX-CHRONOS VXOR-Modul.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import json
import uuid
from unittest.mock import MagicMock, patch
from datetime import datetime
from pathlib import Path

# Konfiguriere Umgebungsvariablen für Tests
os.environ['MISO_ZTM_MODE'] = '1'
os.environ['MISO_ZTM_LOG_LEVEL'] = 'INFO'

# Füge Projektpfad zum Pythonpfad hinzu
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, str(PROJECT_ROOT))

# Import der zu testenden Komponenten
from miso.vxor.chronos_echo_prime_bridge import (
    ChronosEchoBridge, sync_timeline, apply_chronos_optimizations, detect_paradoxes
)

class TestChronosEchoBridge(unittest.TestCase):
    """Test-Suite für die ChronosEchoBridge-Klasse"""
    
    def setUp(self):
        """Testumgebung vorbereiten"""
        # ECHO-PRIME-Controller Mock erstellen
        self.echo_prime_mock = MagicMock()
        
        # VXOR-Adapter-Mock erstellen
        self.module_patcher = patch('miso.vxor.chronos_echo_prime_bridge.get_module')
        self.get_module_mock = self.module_patcher.start()
        
        # VX-CHRONOS-Modul-Mock erstellen
        self.chronos_module_mock = MagicMock()
        self.get_module_mock.return_value = self.chronos_module_mock
        
        # Timeline-Mock erstellen
        self.timeline_mock = MagicMock()
        self.timeline_mock.id = str(uuid.uuid4())
        self.timeline_mock.name = "Test Timeline"
        self.timeline_mock.description = "Eine Test-Zeitlinie"
        
        # ECHO-PRIME Timeline.get_all_nodes() Mock
        node1 = MagicMock()
        node1.id = str(uuid.uuid4())
        node1.description = "Wurzelknoten"
        node1.timestamp = datetime.now().timestamp()
        node1.probability = 0.9
        node1.parent_id = None
        node1.trigger_level = MagicMock()
        node1.trigger_level.name = "HIGH"
        node1.metadata = {"key": "value"}
        
        node2 = MagicMock()
        node2.id = str(uuid.uuid4())
        node2.description = "Kindknoten"
        node2.timestamp = datetime.now().timestamp()
        node2.probability = 0.7
        node2.parent_id = node1.id
        node2.trigger_level = MagicMock()
        node2.trigger_level.name = "MEDIUM"
        node2.metadata = {"key2": "value2"}
        
        self.timeline_mock.get_all_nodes.return_value = [node1, node2]
        
        # ECHO-PRIME get_timeline() Mock
        self.echo_prime_mock.get_timeline.return_value = self.timeline_mock
        
        # VX-CHRONOS create_timeline Mock
        self.chronos_module_mock.create_timeline.return_value = {
            "timeline_id": str(uuid.uuid4()),
            "name": "Test Timeline",
            "status": "created"
        }
        
        # VX-CHRONOS create_node Mock
        self.chronos_module_mock.create_node.return_value = {
            "node_id": str(uuid.uuid4()),
            "status": "created"
        }
        
        # Bridge-Instanz mit Mocks erstellen
        with patch('miso.vxor.chronos_echo_prime_bridge.ECHO_PRIME_AVAILABLE', True), \
             patch('miso.vxor.chronos_echo_prime_bridge.VXOR_ADAPTER_AVAILABLE', True):
            self.bridge = ChronosEchoBridge(self.echo_prime_mock)
            
            # Setze Bridge als initialisiert
            self.bridge.initialized = True
            self.bridge.chronos_module = self.chronos_module_mock
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        self.module_patcher.stop()
    
    def test_initialization(self):
        """Test der korrekten Initialisierung der Bridge"""
        # Überprüfe, ob die Bridge korrekt initialisiert wurde
        self.assertTrue(self.bridge.initialized)
        self.assertEqual(self.bridge.echo_prime, self.echo_prime_mock)
        self.assertEqual(self.bridge.chronos_module, self.chronos_module_mock)
        self.assertIsNotNone(self.bridge.bridge_id)
        self.assertEqual(self.bridge.timeline_mappings, {})
        self.assertEqual(self.bridge.node_mappings, {})
    
    def test_sync_timeline(self):
        """Test der Zeitliniensynchronisation"""
        # ECHO-PRIME Timeline-ID
        echo_timeline_id = self.timeline_mock.id
        
        # Führe Synchronisation durch
        result = self.bridge.sync_timeline(echo_timeline_id)
        
        # Überprüfe Ergebnis
        self.assertTrue(result)
        
        # Überprüfe, ob ECHO-PRIME Timeline abgerufen wurde
        self.echo_prime_mock.get_timeline.assert_called_once_with(echo_timeline_id)
        
        # Überprüfe, ob VX-CHRONOS Timeline erstellt wurde
        self.chronos_module_mock.create_timeline.assert_called_once()
        
        # Überprüfe, ob die Timeline-Mapping aktualisiert wurde
        self.assertIn(echo_timeline_id, self.bridge.timeline_mappings)
        
        # Überprüfe, ob _sync_nodes aufgerufen wurde
        self.timeline_mock.get_all_nodes.assert_called_once()
        
        # Überprüfe, ob VX-CHRONOS Knoten erstellt wurden (2 Knoten)
        self.assertEqual(self.chronos_module_mock.create_node.call_count, 2)
    
    def test_apply_chronos_optimizations(self):
        """Test der Anwendung von VX-CHRONOS-Optimierungen"""
        # ECHO-PRIME Timeline-ID
        echo_timeline_id = self.timeline_mock.id
        
        # Erstelle Timeline-Mapping
        chronos_timeline_id = str(uuid.uuid4())
        self.bridge.timeline_mappings[echo_timeline_id] = chronos_timeline_id
        
        # Mock für VX-CHRONOS optimize_timeline
        optimization_results = {
            "summary": {
                "total_changes": 2,
                "improvement_score": 0.85
            },
            "changes": [
                {
                    "type": "probability_update",
                    "node_id": str(uuid.uuid4()),
                    "old_probability": 0.7,
                    "new_probability": 0.8,
                    "reason": "Konsistenzverbesserung"
                },
                {
                    "type": "node_add",
                    "parent_id": str(uuid.uuid4()),
                    "new_node_id": str(uuid.uuid4()),
                    "description": "Optimierter Pfad",
                    "probability": 0.75,
                    "reason": "Alternativpfad mit höherer Erfolgswahrscheinlichkeit"
                }
            ]
        }
        self.chronos_module_mock.optimize_timeline.return_value = optimization_results
        
        # Erstelle eine Knotenzuordnung für die Optimierungen
        node_id_echo = self.timeline_mock.get_all_nodes()[1].id
        node_id_chronos = optimization_results["changes"][0]["node_id"]
        self.bridge.node_mappings[node_id_echo] = node_id_chronos
        
        parent_id_echo = self.timeline_mock.get_all_nodes()[0].id
        parent_id_chronos = optimization_results["changes"][1]["parent_id"]
        self.bridge.node_mappings[parent_id_echo] = parent_id_chronos
        
        # Führe Optimierung durch
        result = self.bridge.apply_chronos_optimizations(echo_timeline_id)
        
        # Überprüfe Ergebnis
        self.assertTrue(result["success"])
        self.assertEqual(result["echo_timeline_id"], echo_timeline_id)
        self.assertEqual(result["chronos_timeline_id"], chronos_timeline_id)
        
        # Überprüfe, ob VX-CHRONOS optimize_timeline aufgerufen wurde
        self.chronos_module_mock.optimize_timeline.assert_called_once_with(
            chronos_timeline_id,
            {
                "depth": 3,
                "methods": ["probability", "consistency", "efficiency"],
                "metadata": {
                    "source": "ECHO-PRIME",
                    "echo_timeline_id": echo_timeline_id,
                    "bridge_id": self.bridge.bridge_id
                }
            }
        )
        
        # Überprüfe, ob ECHO-PRIME update_time_node aufgerufen wurde
        self.echo_prime_mock.update_time_node.assert_called_once_with(
            echo_timeline_id, node_id_echo, 
            {"probability": optimization_results["changes"][0]["new_probability"]}
        )
        
        # Überprüfe, ob ECHO-PRIME add_time_node aufgerufen wurde
        self.echo_prime_mock.add_time_node.assert_called_once()
    
    def test_detect_paradoxes(self):
        """Test der Paradoxieerkennung"""
        # ECHO-PRIME Timeline-ID
        echo_timeline_id = self.timeline_mock.id
        
        # Erstelle Timeline-Mapping
        chronos_timeline_id = str(uuid.uuid4())
        self.bridge.timeline_mappings[echo_timeline_id] = chronos_timeline_id
        
        # Mock für VX-CHRONOS detect_paradoxes
        paradox_results = {
            "paradoxes": [
                {
                    "id": str(uuid.uuid4()),
                    "type": "causal",
                    "description": "Kausale Schleife entdeckt",
                    "severity": "HIGH",
                    "involved_node_ids": [
                        str(uuid.uuid4()),
                        str(uuid.uuid4())
                    ],
                    "recommendations": [
                        "Zeitlinienverzweigung erstellen",
                        "Knoten B löschen"
                    ]
                },
                {
                    "id": str(uuid.uuid4()),
                    "type": "temporal",
                    "description": "Zeitliche Inkonsistenz",
                    "severity": "MEDIUM",
                    "involved_node_ids": [
                        str(uuid.uuid4())
                    ],
                    "recommendations": [
                        "Zeitstempel korrigieren"
                    ]
                }
            ]
        }
        self.chronos_module_mock.detect_paradoxes.return_value = paradox_results
        
        # Erstelle Knotenzuordnungen für die Paradoxien
        for node_index, chronos_node_id in enumerate(paradox_results["paradoxes"][0]["involved_node_ids"]):
            node_id_echo = str(uuid.uuid4())
            self.bridge.node_mappings[node_id_echo] = chronos_node_id
        
        for node_index, chronos_node_id in enumerate(paradox_results["paradoxes"][1]["involved_node_ids"]):
            node_id_echo = str(uuid.uuid4())
            self.bridge.node_mappings[node_id_echo] = chronos_node_id
        
        # Führe Paradoxieerkennung durch
        result = self.bridge.detect_paradoxes(echo_timeline_id)
        
        # Überprüfe Ergebnis
        self.assertTrue(result["success"])
        self.assertEqual(result["echo_timeline_id"], echo_timeline_id)
        self.assertEqual(result["total_count"], 2)
        self.assertEqual(result["severity_summary"]["high"], 1)
        self.assertEqual(result["severity_summary"]["medium"], 1)
        
        # Überprüfe, ob VX-CHRONOS detect_paradoxes aufgerufen wurde
        self.chronos_module_mock.detect_paradoxes.assert_called_once_with(
            chronos_timeline_id,
            {
                "depth": 5,
                "types": ["causal", "temporal", "logical", "probabilistic"],
                "metadata": {
                    "source": "ECHO-PRIME",
                    "echo_timeline_id": echo_timeline_id,
                    "bridge_id": self.bridge.bridge_id
                }
            }
        )
    
    def test_sync_timeline_helper_function(self):
        """Test der Hilfsfunktion sync_timeline"""
        # Patch die get_bridge-Funktion
        with patch('miso.vxor.chronos_echo_prime_bridge.get_bridge') as get_bridge_mock:
            # Mock für die Bridge-Instanz
            bridge_mock = MagicMock()
            bridge_mock.sync_timeline.return_value = True
            get_bridge_mock.return_value = bridge_mock
            
            # Führe sync_timeline aus
            result = sync_timeline("test_timeline_id")
            
            # Überprüfe Ergebnis
            self.assertTrue(result)
            get_bridge_mock.assert_called_once()
            bridge_mock.sync_timeline.assert_called_once_with("test_timeline_id")
    
    def test_apply_chronos_optimizations_helper_function(self):
        """Test der Hilfsfunktion apply_chronos_optimizations"""
        # Patch die get_bridge-Funktion
        with patch('miso.vxor.chronos_echo_prime_bridge.get_bridge') as get_bridge_mock:
            # Mock für die Bridge-Instanz
            bridge_mock = MagicMock()
            bridge_mock.apply_chronos_optimizations.return_value = {"success": True}
            get_bridge_mock.return_value = bridge_mock
            
            # Führe apply_chronos_optimizations aus
            result = apply_chronos_optimizations("test_timeline_id")
            
            # Überprüfe Ergebnis
            self.assertEqual(result, {"success": True})
            get_bridge_mock.assert_called_once()
            bridge_mock.apply_chronos_optimizations.assert_called_once_with("test_timeline_id")
    
    def test_detect_paradoxes_helper_function(self):
        """Test der Hilfsfunktion detect_paradoxes"""
        # Patch die get_bridge-Funktion
        with patch('miso.vxor.chronos_echo_prime_bridge.get_bridge') as get_bridge_mock:
            # Mock für die Bridge-Instanz
            bridge_mock = MagicMock()
            bridge_mock.detect_paradoxes.return_value = {"success": True, "paradoxes": []}
            get_bridge_mock.return_value = bridge_mock
            
            # Führe detect_paradoxes aus
            result = detect_paradoxes("test_timeline_id")
            
            # Überprüfe Ergebnis
            self.assertEqual(result, {"success": True, "paradoxes": []})
            get_bridge_mock.assert_called_once()
            bridge_mock.detect_paradoxes.assert_called_once_with("test_timeline_id")

# Verhindern, dass Tests ausgeführt werden, wenn diese Datei importiert wird
if __name__ == '__main__':
    unittest.main()
