#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: Unit Tests für context_analyzer.py
"""

import os
import sys
import unittest
import time
import threading
from unittest.mock import MagicMock, patch

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from context_core import ContextSource, ContextPriority, ContextData
from context_analyzer import (
    AnalysisDimension, AnalysisResult, AnalysisConfig, ContextAnalyzer
)


class TestAnalysisConfig(unittest.TestCase):
    """Tests für die AnalysisConfig-Klasse"""
    
    def test_init(self):
        """Test der Initialisierung mit Standardwerten"""
        config = AnalysisConfig()
        
        self.assertEqual(config.relevance_threshold, 0.6)
        self.assertEqual(config.importance_threshold, 0.7)
        self.assertEqual(config.urgency_threshold, 0.8)
        self.assertEqual(config.novelty_threshold, 0.5)
        self.assertEqual(config.anomaly_threshold, 0.7)
        
        # Prüfe, ob alle Quellen in source_weights enthalten sind
        for source in ContextSource:
            self.assertIn(source, config.source_weights)
        
        # Prüfe, ob alle Ergebnistypen in priority_mappings enthalten sind
        for result in AnalysisResult:
            self.assertIn(result, config.priority_mappings)
    
    def test_custom_init(self):
        """Test der Initialisierung mit benutzerdefinierten Werten"""
        custom_source_weights = {
            ContextSource.VISUAL: 0.9,
            ContextSource.LANGUAGE: 0.8,
            ContextSource.INTERNAL: 1.0,
            ContextSource.EXTERNAL: 0.7,
            ContextSource.MEMORY: 0.6,
            ContextSource.EMOTION: 0.9,
            ContextSource.INTENT: 1.0,
            ContextSource.REFLEX: 1.0
        }
        
        custom_priority_mappings = {
            AnalysisResult.URGENT: ContextPriority.CRITICAL,
            AnalysisResult.IMPORTANT: ContextPriority.CRITICAL,  # Geändert
            AnalysisResult.RELEVANT: ContextPriority.HIGH,       # Geändert
            AnalysisResult.NOVEL: ContextPriority.HIGH,          # Geändert
            AnalysisResult.ANOMALOUS: ContextPriority.CRITICAL,  # Geändert
            AnalysisResult.EXPECTED: ContextPriority.MEDIUM,     # Geändert
            AnalysisResult.REDUNDANT: ContextPriority.LOW,       # Geändert
            AnalysisResult.IRRELEVANT: ContextPriority.LOW       # Geändert
        }
        
        config = AnalysisConfig(
            relevance_threshold=0.7,
            importance_threshold=0.8,
            urgency_threshold=0.9,
            novelty_threshold=0.6,
            anomaly_threshold=0.8,
            source_weights=custom_source_weights,
            priority_mappings=custom_priority_mappings
        )
        
        self.assertEqual(config.relevance_threshold, 0.7)
        self.assertEqual(config.importance_threshold, 0.8)
        self.assertEqual(config.urgency_threshold, 0.9)
        self.assertEqual(config.novelty_threshold, 0.6)
        self.assertEqual(config.anomaly_threshold, 0.8)
        self.assertEqual(config.source_weights, custom_source_weights)
        self.assertEqual(config.priority_mappings, custom_priority_mappings)


class TestContextAnalyzer(unittest.TestCase):
    """Tests für die ContextAnalyzer-Klasse"""
    
    def setUp(self):
        """Setup für die Tests"""
        # Erstelle Mock für ContextCore
        self.mock_context_core = MagicMock()
        self.mock_context_processor = MagicMock()
        self.mock_context_core.context_processor = self.mock_context_processor
        
        # Erstelle ContextAnalyzer mit Mock
        self.analyzer = ContextAnalyzer(self.mock_context_core)
    
    def test_init(self):
        """Test der Initialisierung"""
        self.assertEqual(self.analyzer.context_core, self.mock_context_core)
        self.assertIsInstance(self.analyzer.config, AnalysisConfig)
        self.assertEqual(len(self.analyzer.historical_data), 0)
        self.assertEqual(self.analyzer.max_history_length, 1000)
        
        # Prüfe, ob alle Analysedimensionen registriert sind
        for dimension in AnalysisDimension:
            self.assertIn(dimension, self.analyzer.analysis_functions)
        
        self.assertEqual(len(self.analyzer.custom_analysis_functions), 0)
        self.assertEqual(self.analyzer.performance_metrics['analyzed_items_count'], 0)
    
    def test_process_context_data(self):
        """Test der Kontextdatenverarbeitung"""
        # Erstelle Testdaten
        context_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.MEDIUM,
            data={"object_detected": "Person", "confidence": 0.95},
            metadata={"camera_id": "main"}
        )
        
        # Verarbeite Kontextdaten
        result = self.analyzer.process_context_data(context_data)
        
        # Prüfe Ergebnis
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.context_data, context_data)
        self.assertIsInstance(result.scores, dict)
        self.assertIsInstance(result.results, list)
        self.assertIsInstance(result.adjusted_priority, ContextPriority)
        self.assertGreater(result.analysis_time_ms, 0)
        
        # Prüfe, ob alle Dimensionen analysiert wurden
        for dimension in AnalysisDimension:
            self.assertIn(dimension, result.scores)
        
        # Prüfe, ob die Performance-Metriken aktualisiert wurden
        self.assertEqual(self.analyzer.performance_metrics['analyzed_items_count'], 1)
        self.assertGreater(self.analyzer.performance_metrics['avg_analysis_time_ms'], 0)
    
    def test_register_custom_analysis(self):
        """Test der Registrierung benutzerdefinierter Analysefunktionen"""
        # Erstelle Mock-Funktion
        mock_func = MagicMock(return_value=0.75)
        
        # Registriere Funktion
        self.analyzer.register_custom_analysis(
            "test_analysis",
            AnalysisDimension.RELEVANCE,
            mock_func
        )
        
        # Prüfe, ob die Funktion registriert wurde
        self.assertIn("test_analysis", self.analyzer.custom_analysis_functions)
        self.assertEqual(
            self.analyzer.custom_analysis_functions["test_analysis"][0],
            AnalysisDimension.RELEVANCE
        )
        self.assertEqual(
            self.analyzer.custom_analysis_functions["test_analysis"][1],
            mock_func
        )
        
        # Verarbeite Kontextdaten mit der benutzerdefinierten Funktion
        context_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.MEDIUM,
            data={"test": "data"}
        )
        
        result = self.analyzer.process_context_data(context_data)
        
        # Prüfe, ob die Funktion aufgerufen wurde
        mock_func.assert_called_once_with(context_data)
    
    def test_analyze_relevance(self):
        """Test der Relevanzanalyse"""
        # Erstelle Testdaten
        context_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.HIGH,
            data={"object_detected": "Person", "confidence": 0.95},
            metadata={"relevance_score": 0.85}  # Vorberechnete Relevanz
        )
        
        # Führe Analyse durch
        relevance = self.analyzer._analyze_relevance(context_data)
        
        # Prüfe Ergebnis
        self.assertEqual(relevance, 0.85)  # Sollte die vorberechnete Relevanz verwenden
        
        # Teste ohne vorberechnete Relevanz
        context_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.HIGH,
            data={"object_detected": "Person", "confidence": 0.95}
        )
        
        relevance = self.analyzer._analyze_relevance(context_data)
        
        # Prüfe Ergebnis
        self.assertGreaterEqual(relevance, 0.0)
        self.assertLessEqual(relevance, 1.0)
    
    def test_analyze_importance(self):
        """Test der Wichtigkeitsanalyse"""
        # Erstelle Testdaten mit vorberechneter Wichtigkeit
        context_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.MEDIUM,
            data={"object_detected": "Person", "confidence": 0.95},
            metadata={"importance_score": 0.75}
        )
        
        # Führe Analyse durch
        importance = self.analyzer._analyze_importance(context_data)
        
        # Prüfe Ergebnis
        self.assertEqual(importance, 0.75)  # Sollte die vorberechnete Wichtigkeit verwenden
        
        # Teste mit Wichtigkeitsindikatoren in den Daten
        context_data = ContextData(
            source=ContextSource.INTERNAL,
            priority=ContextPriority.MEDIUM,
            data={"status": "warning", "message": "Low battery"}
        )
        
        importance = self.analyzer._analyze_importance(context_data)
        
        # Prüfe Ergebnis
        self.assertGreaterEqual(importance, 0.7)  # Sollte mindestens den Wert für "warning" haben
    
    def test_analyze_urgency(self):
        """Test der Dringlichkeitsanalyse"""
        # Erstelle Testdaten mit vorberechneter Dringlichkeit
        context_data = ContextData(
            source=ContextSource.REFLEX,
            priority=ContextPriority.HIGH,
            data={"trigger": "collision", "intensity": 0.8},
            metadata={"urgency_score": 0.9}
        )
        
        # Führe Analyse durch
        urgency = self.analyzer._analyze_urgency(context_data)
        
        # Prüfe Ergebnis
        self.assertEqual(urgency, 0.9)  # Sollte die vorberechnete Dringlichkeit verwenden
        
        # Teste mit Dringlichkeitsindikatoren in den Daten
        context_data = ContextData(
            source=ContextSource.EXTERNAL,
            priority=ContextPriority.MEDIUM,
            data={"message": "Please respond immediately", "type": "urgent"}
        )
        
        urgency = self.analyzer._analyze_urgency(context_data)
        
        # Prüfe Ergebnis
        self.assertGreaterEqual(urgency, 0.8)  # Sollte mindestens den Wert für "urgent" haben
    
    def test_analyze_novelty(self):
        """Test der Neuartigkeitsanalyse"""
        # Erstelle Testdaten
        context_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.MEDIUM,
            data={"object_detected": "Person", "confidence": 0.95}
        )
        
        # Erste Analyse sollte hohe Neuartigkeit ergeben
        novelty1 = self.analyzer._analyze_novelty(context_data)
        self.assertGreaterEqual(novelty1, 0.9)  # Sollte sehr neu sein
        
        # Aktualisiere historische Daten manuell
        self.analyzer._update_historical_data(context_data)
        
        # Zweite Analyse mit identischen Daten sollte niedrigere Neuartigkeit ergeben
        novelty2 = self.analyzer._analyze_novelty(context_data)
        self.assertLess(novelty2, novelty1)  # Sollte weniger neu sein
    
    def test_analyze_anomaly(self):
        """Test der Anomalieanalyse"""
        # Fülle historische Daten mit normalen Werten
        for _ in range(10):
            context_data = ContextData(
                source=ContextSource.INTERNAL,
                priority=ContextPriority.MEDIUM,
                data={"system_load": 0.5, "memory_usage": 0.6, "temperature": 70}
            )
            self.analyzer._update_historical_data(context_data)
        
        # Erstelle anomale Testdaten
        anomaly_data = ContextData(
            source=ContextSource.INTERNAL,
            priority=ContextPriority.MEDIUM,
            data={"system_load": 0.95, "memory_usage": 0.98, "temperature": 90}
        )
        
        # Führe Analyse durch
        anomaly_score = self.analyzer._analyze_anomaly(anomaly_data)
        
        # Prüfe Ergebnis
        self.assertGreaterEqual(anomaly_score, 0.7)  # Sollte als anomal erkannt werden
    
    def test_determine_priority(self):
        """Test der Prioritätsbestimmung"""
        # Teste verschiedene Ergebniskombinationen
        
        # Dringend + Wichtig
        results = [AnalysisResult.URGENT, AnalysisResult.IMPORTANT]
        priority = self.analyzer._determine_priority(results, ContextSource.VISUAL)
        self.assertEqual(priority, ContextPriority.CRITICAL)
        
        # Relevant + Neuartig
        results = [AnalysisResult.RELEVANT, AnalysisResult.NOVEL]
        priority = self.analyzer._determine_priority(results, ContextSource.VISUAL)
        self.assertEqual(priority, ContextPriority.MEDIUM)
        
        # Irrelevant + Redundant
        results = [AnalysisResult.IRRELEVANT, AnalysisResult.REDUNDANT]
        priority = self.analyzer._determine_priority(results, ContextSource.VISUAL)
        self.assertEqual(priority, ContextPriority.BACKGROUND)
        
        # Anomal
        results = [AnalysisResult.ANOMALOUS]
        priority = self.analyzer._determine_priority(results, ContextSource.VISUAL)
        self.assertEqual(priority, ContextPriority.HIGH)
    
    def test_update_historical_data(self):
        """Test der Aktualisierung historischer Daten"""
        # Erstelle Testdaten
        context_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.MEDIUM,
            data={"object_detected": "Person", "confidence": 0.95}
        )
        
        # Aktualisiere historische Daten
        self.analyzer._update_historical_data(context_data)
        
        # Prüfe, ob die Daten hinzugefügt wurden
        self.assertIn(ContextSource.VISUAL, self.analyzer.historical_data)
        self.assertEqual(len(self.analyzer.historical_data[ContextSource.VISUAL]), 1)
        self.assertEqual(self.analyzer.historical_data[ContextSource.VISUAL][0], context_data)
        
        # Füge weitere Daten hinzu
        for _ in range(10):
            self.analyzer._update_historical_data(context_data)
        
        # Prüfe, ob die Daten hinzugefügt wurden
        self.assertEqual(len(self.analyzer.historical_data[ContextSource.VISUAL]), 11)
        
        # Teste Begrenzung der Historiengröße
        self.analyzer.max_history_length = 5
        self.analyzer._update_historical_data(context_data)
        
        # Prüfe, ob die Historie begrenzt wurde
        self.assertEqual(len(self.analyzer.historical_data[ContextSource.VISUAL]), 5)
    
    def test_calculate_similarity(self):
        """Test der Ähnlichkeitsberechnung"""
        # Teste identische Strings
        str1 = "Dies ist ein Test"
        str2 = "Dies ist ein Test"
        similarity = self.analyzer._calculate_similarity(str1, str2)
        self.assertEqual(similarity, 1.0)
        
        # Teste teilweise überlappende Strings
        str1 = "Dies ist ein Test"
        str2 = "Dies ist ein anderer Test"
        similarity = self.analyzer._calculate_similarity(str1, str2)
        self.assertGreater(similarity, 0.5)
        self.assertLess(similarity, 1.0)
        
        # Teste völlig unterschiedliche Strings
        str1 = "Dies ist ein Test"
        str2 = "Völlig anderer Inhalt"
        similarity = self.analyzer._calculate_similarity(str1, str2)
        self.assertLess(similarity, 0.5)
    
    def test_extract_numeric_values(self):
        """Test der Extraktion numerischer Werte"""
        # Teste einfache Daten
        data = {"value1": 10, "value2": 20.5, "text": "test"}
        numeric_values = self.analyzer._extract_numeric_values(data)
        self.assertEqual(numeric_values, {"value1": 10.0, "value2": 20.5})
        
        # Teste verschachtelte Daten
        data = {
            "outer": {
                "inner1": 10,
                "inner2": 20.5
            },
            "list": [1, 2, 3, 4],
            "text": "test"
        }
        numeric_values = self.analyzer._extract_numeric_values(data)
        self.assertEqual
(Content truncated due to size limit. Use line ranges to read in chunks)