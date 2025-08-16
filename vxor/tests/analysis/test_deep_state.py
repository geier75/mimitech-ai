#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Tests für das Deep-State-Modul

Dieses Modul enthält Tests für das Deep-State-Modul.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importiere Deep-State-Module
from miso.analysis.deep_state import DeepStateAnalyzer, DeepStateConfig, AnalysisResult, ReactionType, get_deep_state_analyzer
from miso.analysis.deep_state_patterns import PatternMatcher, ControlPattern
from miso.analysis.deep_state_network import NetworkAnalyzer, NetworkNode
from miso.analysis.deep_state_security import SecurityManager, EncryptionLevel

class TestDeepStateModule(unittest.TestCase):
    """Tests für das Deep-State-Modul"""
    
    def setUp(self):
        """Initialisiert die Testumgebung"""
        # Konfiguriere Deep-State-Analyzer für Tests
        self.config = DeepStateConfig(
            high_threshold=0.85,
            medium_threshold=0.55,
            bias_weight=0.25,
            pattern_weight=0.25,
            network_weight=0.25,
            paradox_weight=0.25,
            encryption_enabled=False,  # Deaktiviere Verschlüsselung für Tests
            zt_mode_enabled=False,     # Deaktiviere ZT-Modus für Tests
            command_lock="TEST",
            log_mode="TEST"
        )
        
        # Initialisiere Deep-State-Analyzer
        self.analyzer = DeepStateAnalyzer(self.config)
    
    def test_basic_functionality(self):
        """Testet die grundlegende Funktionalität des Deep-State-Analyzers"""
        # Führe Analyse durch
        result = self.analyzer.analyze(
            content_stream="Dies ist ein normaler Text ohne Deep-State-Bezug.",
            source_id="TEST_SOURCE_001",
            source_trust_level=0.8,
            language_code="DE",
            context_cluster="test",
            timeframe=datetime.datetime.now()
        )
        
        # Überprüfe Ergebnis
        self.assertIsInstance(result, AnalysisResult)
        self.assertLessEqual(result.ds_probability, self.config.medium_threshold)
        self.assertEqual(result.reaction, ReactionType.NORMAL)
    
    def test_deep_state_detection(self):
        """Testet die Erkennung von Deep-State-Mustern"""
        # Führe Analyse mit Deep-State-Bezug durch
        result = self.analyzer.analyze(
            content_stream="Die globalen Eliten und die Schattenregierung kontrollieren die Medien und verbreiten Propaganda.",
            source_id="TEST_SOURCE_002",
            source_trust_level=0.5,
            language_code="DE",
            context_cluster="politik,medien",
            timeframe=datetime.datetime.now()
        )
        
        # Überprüfe Ergebnis
        self.assertIsInstance(result, AnalysisResult)
        self.assertGreaterEqual(result.ds_probability, self.config.medium_threshold)
        self.assertIn(result.reaction, [ReactionType.MONITOR, ReactionType.ALERT])
    
    def test_pattern_matcher(self):
        """Testet den PatternMatcher"""
        # Initialisiere PatternMatcher
        pattern_matcher = PatternMatcher()
        
        # Überprüfe Muster
        text = "Die globalen Eliten kontrollieren die Medien."
        score = pattern_matcher.match_patterns(text, "politik,medien")
        
        # Überprüfe Ergebnis
        self.assertGreater(score, 0.0)
        
        # Überprüfe gefundene Muster
        matched_patterns = pattern_matcher.get_matched_patterns(text, "politik,medien")
        self.assertGreater(len(matched_patterns), 0)
    
    def test_network_analyzer(self):
        """Testet den NetworkAnalyzer"""
        # Initialisiere NetworkAnalyzer
        network_analyzer = NetworkAnalyzer()
        
        # Überprüfe Netzwerkanalyse
        score = network_analyzer.analyze_network("MEDIA_001", "medien,politik")
        
        # Überprüfe Ergebnis
        self.assertGreaterEqual(score, 0.0)
        
        # Überprüfe potenzielle Verbindungen
        connections = network_analyzer.get_potential_connections("MEDIA_001", "medien,politik")
        self.assertIsInstance(connections, list)
    
    def test_security_manager(self):
        """Testet den SecurityManager"""
        # Initialisiere SecurityManager
        security_manager = SecurityManager(
            encryption_enabled=True,
            zt_mode_enabled=True,
            command_lock="TEST",
            log_mode="TEST"
        )
        
        # Überprüfe Berechtigungen
        self.assertTrue(security_manager.check_permissions())
        
        # Erstelle Beispielergebnis
        result = AnalysisResult(
            report_text="Testbericht",
            ds_probability=0.5,
            reaction=ReactionType.NORMAL,
            reaction_text="Test",
            paradox_signal=0.1,
            bias_score=0.2,
            pattern_score=0.3,
            network_score=0.4,
            source_id="TEST_SOURCE"
        )
        
        # Verschlüssele Ergebnis
        encrypted_result = security_manager.encrypt_result(result)
        
        # Überprüfe Verschlüsselung
        self.assertTrue(encrypted_result.encrypted)
        self.assertIsNotNone(encrypted_result.encryption_key_id)
        self.assertNotEqual(encrypted_result.report_text, "Testbericht")
        
        # Entschlüssele Ergebnis
        decrypted_result = security_manager.decrypt_result(encrypted_result)
        
        # Überprüfe Entschlüsselung
        self.assertFalse(decrypted_result.encrypted)
        self.assertIsNone(decrypted_result.encryption_key_id)
        self.assertEqual(decrypted_result.report_text, "Testbericht")
    
    def test_singleton_pattern(self):
        """Testet das Singleton-Muster des Deep-State-Analyzers"""
        # Hole globale Instanz
        global_analyzer = get_deep_state_analyzer()
        
        # Überprüfe, ob es sich um dieselbe Instanz handelt
        self.assertIsInstance(global_analyzer, DeepStateAnalyzer)
        
        # Hole erneut globale Instanz
        another_global_analyzer = get_deep_state_analyzer()
        
        # Überprüfe, ob es sich um dieselbe Instanz handelt
        self.assertIs(global_analyzer, another_global_analyzer)

if __name__ == '__main__':
    unittest.main()
