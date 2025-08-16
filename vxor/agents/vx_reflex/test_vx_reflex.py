#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-REFLEX: Test Suite
---------------------
Testsuite für das VX-REFLEX Modul.

Version: 0.1.0
Author: VXOR Build Core / Omega One
"""

import unittest
import json
import time
import os
import sys
import logging
from unittest.mock import MagicMock, patch

# Konfiguriere Logging für Tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/ubuntu/vXor_Modules/VX-REFLEX/logs/test.log"),
        logging.StreamHandler()
    ]
)

# Füge Modul-Pfad zum Suchpfad hinzu
sys.path.append("/home/ubuntu/vXor_Modules/VX-REFLEX/src")

# Importiere Module
from reflex_core import ReflexCore, get_reflex_core
from stimulus_analyzer import StimulusAnalyzer, get_stimulus_analyzer
from reflex_responder import ReflexResponder, get_reflex_responder
from reaction_profile_manager import ReactionProfileManager, get_profile_manager
from vxor_integration import VXORBridge, VXORIntegration, get_vxor_integration


class TestReflexCore(unittest.TestCase):
    """Tests für die ReflexCore-Komponente"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        # Erstelle temporäre Konfigurationsdatei
        self.test_config_path = "/home/ubuntu/vXor_Modules/VX-REFLEX/config/test_config.json"
        self.test_config = {
            "thresholds": {
                "cpu_load": {
                    "high": 90,
                    "medium": 75,
                    "low": 50
                },
                "audio_level": {
                    "high": 75,
                    "medium": 60,
                    "low": 40
                },
                "object_proximity": {
                    "high": 0.8,
                    "medium": 1.5,
                    "low": 3.0
                }
            },
            "reaction_profiles": {
                "default": {
                    "response_delay": 0.05,
                    "priority_weights": {
                        "HIGH": 1.0,
                        "MEDIUM": 0.6,
                        "LOW": 0.3
                    }
                },
                "emergency": {
                    "response_delay": 0.01,
                    "priority_weights": {
                        "HIGH": 1.0,
                        "MEDIUM": 0.8,
                        "LOW": 0.5
                    }
                }
            },
            "active_profile": "default",
            "max_queue_size": 100,
            "processing_interval": 0.01
        }
        
        os.makedirs(os.path.dirname(self.test_config_path), exist_ok=True)
        with open(self.test_config_path, 'w') as config_file:
            json.dump(self.test_config, config_file)
        
        # Erstelle ReflexCore-Instanz mit Testkonfiguration
        self.reflex_core = ReflexCore(config_path=self.test_config_path)
        
        # Mock für Stimulus-Handler
        self.mock_handler = MagicMock()
        self.reflex_core.register_stimulus_handler("test", self.mock_handler)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        # Stoppe ReflexCore
        if self.reflex_core.active:
            self.reflex_core.stop()
        
        # Lösche temporäre Konfigurationsdatei
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
    
    def test_initialization(self):
        """Test der Initialisierung"""
        self.assertIsNotNone(self.reflex_core)
        self.assertFalse(self.reflex_core.active)
        self.assertEqual(self.reflex_core.performance_metrics["total_stimuli"], 0)
    
    def test_start_stop(self):
        """Test des Starts und Stopps"""
        # Starte ReflexCore
        self.reflex_core.start()
        self.assertTrue(self.reflex_core.active)
        
        # Stoppe ReflexCore
        self.reflex_core.stop()
        self.assertFalse(self.reflex_core.active)
    
    def test_process_stimulus_high_priority(self):
        """Test der Reizverarbeitung mit hoher Priorität"""
        # Registriere Handler
        self.reflex_core.register_stimulus_handler("system", self.mock_handler)
        
        # Verarbeite Reiz mit hoher Priorität
        result, _ = self.reflex_core.process_stimulus("system", {"cpu_load": 95})
        
        # Prüfe, ob Handler aufgerufen wurde
        self.assertTrue(result)
        self.mock_handler.assert_called_once()
    
    def test_process_stimulus_low_priority(self):
        """Test der Reizverarbeitung mit niedriger Priorität"""
        # Registriere Handler
        self.reflex_core.register_stimulus_handler("system", self.mock_handler)
        
        # Starte ReflexCore
        self.reflex_core.start()
        
        # Verarbeite Reiz mit niedriger Priorität
        result, data = self.reflex_core.process_stimulus("system", {"cpu_load": 40})
        
        # Prüfe Ergebnis
        self.assertTrue(result)
        self.assertEqual(data["status"], "queued")
        
        # Warte kurz, damit die Verarbeitung stattfinden kann
        time.sleep(0.1)
        
        # Prüfe, ob Handler aufgerufen wurde
        self.mock_handler.assert_called_once()
    
    def test_performance_metrics(self):
        """Test der Performance-Metriken"""
        # Verarbeite einige Reize
        self.reflex_core.process_stimulus("test", {"data": "test1"})
        self.reflex_core.process_stimulus("test", {"data": "test2"})
        
        # Prüfe Metriken
        metrics = self.reflex_core.get_performance_metrics()
        self.assertEqual(metrics["total_stimuli"], 2)
        self.assertGreater(metrics["avg_response_time_ms"], 0)


class TestStimulusAnalyzer(unittest.TestCase):
    """Tests für die StimulusAnalyzer-Komponente"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        # Erstelle temporäre Konfigurationsdatei
        self.test_config_path = "/home/ubuntu/vXor_Modules/VX-REFLEX/config/test_config.json"
        self.test_config = {
            "thresholds": {
                "cpu_load": {
                    "high": 90,
                    "medium": 75,
                    "low": 50
                },
                "audio_level": {
                    "high": 75,
                    "medium": 60,
                    "low": 40
                },
                "object_proximity": {
                    "high": 0.8,
                    "medium": 1.5,
                    "low": 3.0
                }
            },
            "pattern_recognition": {
                "visual_patterns": [
                    "schlag",
                    "sturz",
                    "explosion",
                    "feuer",
                    "waffe"
                ],
                "audio_patterns": [
                    "schrei",
                    "explosion",
                    "alarm",
                    "glasbruch"
                ],
                "danger_objects": [
                    "messer",
                    "feuer",
                    "waffe",
                    "fahrzeug"
                ]
            },
            "analysis": {
                "visual_sensitivity": 0.8,
                "audio_sensitivity": 0.7,
                "system_sensitivity": 0.9,
                "emotional_sensitivity": 0.6
            }
        }
        
        os.makedirs(os.path.dirname(self.test_config_path), exist_ok=True)
        with open(self.test_config_path, 'w') as config_file:
            json.dump(self.test_config, config_file)
        
        # Erstelle StimulusAnalyzer-Instanz mit Testkonfiguration
        self.stimulus_analyzer = StimulusAnalyzer(config_path=self.test_config_path)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        # Lösche temporäre Konfigurationsdatei
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
    
    def test_initialization(self):
        """Test der Initialisierung"""
        self.assertIsNotNone(self.stimulus_analyzer)
        self.assertEqual(self.stimulus_analyzer.performance_metrics["total_analyses"], 0)
    
    def test_determine_stimulus_type(self):
        """Test der Reiztyp-Bestimmung"""
        # Visueller Reiz
        visual_type = self.stimulus_analyzer._determine_stimulus_type({"image": "test"})
        self.assertEqual(visual_type.value, "visual")
        
        # Auditiver Reiz
        audio_type = self.stimulus_analyzer._determine_stimulus_type({"audio": "test"})
        self.assertEqual(audio_type.value, "audio")
        
        # Systemreiz
        system_type = self.stimulus_analyzer._determine_stimulus_type({"cpu_load": 80})
        self.assertEqual(system_type.value, "system")
        
        # Emotionaler Reiz
        emotional_type = self.stimulus_analyzer._determine_stimulus_type({"emotion": "angst"})
        self.assertEqual(emotional_type.value, "emotional")
        
        # Gefahrenreiz
        danger_type = self.stimulus_analyzer._determine_stimulus_type({"danger": "hoch"})
        self.assertEqual(danger_type.value, "danger")
        
        # Unbekannter Reiz
        unknown_type = self.stimulus_analyzer._determine_stimulus_type({"unknown": "test"})
        self.assertEqual(unknown_type.value, "unknown")
    
    def test_analyze_visual_stimulus(self):
        """Test der Analyse visueller Reize"""
        # Visueller Reiz mit Gefahrenobjekt
        visual_data = {
            "objects": [
                {"name": "waffe", "distance": 0.5, "confidence": 0.92}
            ],
            "motion_data": {"velocity": 7.5}
        }
        
        result = self.stimulus_analyzer.analyze_stimulus(visual_data)
        
        # Prüfe Ergebnis
        self.assertEqual(result["type"], "visual")
        self.assertTrue(result["analyzed"])
        self.assertTrue(result["motion_detected"])
        self.assertEqual(result["proximity"], 0.5)
        self.assertIn("waffe", result["danger_objects"])
    
    def test_analyze_audio_stimulus(self):
        """Test der Analyse auditiver Reize"""
        # Auditiver Reiz mit hoher Lautstärke
        audio_data = {
            "level": 80,
            "frequency_data": [100, 500, 2000, 5000],
            "audio_pattern": "schrei"
        }
        
        result = self.stimulus_analyzer.analyze_stimulus(audio_data)
        
        # Prüfe Ergebnis
        self.assertEqual(result["type"], "audio")
        self.assertTrue(result["analyzed"])
        self.assertEqual(result["level"], 80)
        self.assertIn("schrei", result["patterns_detected"])
    
    def test_analyze_system_stimulus(self):
        """Test der Analyse von Systemreizen"""
        # Systemreiz mit hoher CPU-Auslastung
        system_data = {
            "cpu_load": 95,
            "memory": 82,
            "temperature": 78
        }
        
        result = self.stimulus_analyzer.analyze_stimulus(system_data)
        
        # Prüfe Ergebnis
        self.assertEqual(result["type"], "system")
        self.assertTrue(result["analyzed"])
        self.assertEqual(result["system_status"], "critical")
        self.assertIn("cpu_overload", result["warnings"])
        self.assertIn("memory_high", result["warnings"])
        self.assertIn("temperature_high", result["warnings"])
    
    def test_performance_metrics(self):
        """Test der Performance-Metriken"""
        # Analysiere einige Reize
        self.stimulus_analyzer.analyze_stimulus({"image": "test1"})
        self.stimulus_analyzer.analyze_stimulus({"audio": "test2"})
        
        # Prüfe Metriken
        metrics = self.stimulus_analyzer.get_performance_metrics()
        self.assertEqual(metrics["total_analyses"], 2)
        self.assertGreater(metrics["avg_analysis_time_ms"], 0)


class TestReflexResponder(unittest.TestCase):
    """Tests für die ReflexResponder-Komponente"""
    
    def setUp(self):
        """Testumgebung einrichten"""
        # Erstelle temporäre Konfigurationsdatei
        self.test_config_path = "/home/ubuntu/vXor_Modules/VX-REFLEX/config/test_config.json"
        self.test_config = {
            "reaction_profiles": {
                "default": {
                    "response_delay": 0.05,
                    "priority_weights": {
                        "HIGH": 1.0,
                        "MEDIUM": 0.6,
                        "LOW": 0.3
                    }
                },
                "emergency": {
                    "response_delay": 0.01,
                    "priority_weights": {
                        "HIGH": 1.0,
                        "MEDIUM": 0.8,
                        "LOW": 0.5
                    }
                }
            },
            "active_profile": "default",
            "response_rules": {
                "visual": [
                    {"pattern": "schlag", "response": {"type": "soma", "action": "dodge"}},
                    {"pattern": "sturz", "response": {"type": "soma", "action": "brace"}},
                    {"pattern": "feuer", "response": {"type": "combined", "actions": [
                        {"type": "soma", "action": "retreat"},
                        {"type": "psi", "alert": "danger-fire"}
                    ]}}
                ],
                "audio": [
                    {"pattern": "schrei", "response": {"type": "soma", "action": "turn_to_source"}},
                    {"pattern": "explosion", "response": {"type": "soma", "action": "duck"}}
                ],
                "system": [
                    {"warning": "cpu_overload", "response": {"type": "system", "action": "reduce_processes"}},
                    {"warning": "temperature_critical", "response": {"type": "system", "action": "emergency_cooldown"}}
                ]
            }
        }
        
        os.makedirs(os.path.dirname(self.test_config_path), exist_ok=True)
        with open(self.test_config_path, 'w') as config_file:
            json.dump(self.test_config, config_file)
        
        # Erstelle ReflexResponder-Instanz mit Testkonfiguration
        self.reflex_responder = ReflexResponder(config_path=self.test_config_path)
        
        # Mock für VXOR-Bridge
        self.mock_bridge = MagicMock()
        self.mock_bridge.send_signal.return_value = {"status": "success"}
        self.mock_bridge.store_memory.return_value = {"status": "success"}
        
        # Setze Mock-Bridge
        self.reflex_responder.set_vxor_bridge(self.mock_bridge)
    
    def tearDown(self):
        """Testumgebung aufräumen"""
        # Lösche temporäre Konfigurationsdatei
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
    
    def test_initialization(self):
        """Test der Initialisierung"""
        self.assertIsNotNone(self.reflex_responder)
        self.assertEqual(self.reflex_responder.performance_metrics["total_responses"], 0)
    
    def test_determine_response_visual(self):
        """Test der Reaktionsbestimmung für visuelle Reize"""
        # Visueller Reiz mit Schlagmuster
        visual_stimulus = {
            "type": "visual",
            "patterns_detected": ["schlag"],
            "danger_objects": [],
            "motion_detected": True
        }
        
        response = self.reflex_responder._determine_response("visual", visual_stimulus)
        
        # Prüfe Reaktion
        self.assertEqual(response["type"], "soma")
        self.assertEqual(response["action"], "dodge")
    
    def test_determine_response_audio(self):
        """Test der Reaktionsbestimmung für auditive Reize"""
        # Auditiver Reiz mit Schreimuster
        audio_stimulus = {
            "type": "audio",
            "patterns_detected": ["schrei"],
            "level": 80
        }
        
        response = self.reflex_responder._determine_response("audio", audio_stimulus)
        
        # Prüfe Reaktion
        self.assertEqual(response["type"], "soma")
        self.assertEqual(response["action"], "turn_to_s
(Content truncated due to size limit. Use line ranges to read in chunks)