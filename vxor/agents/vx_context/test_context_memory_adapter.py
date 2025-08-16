#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: Unit Tests für context_memory_adapter.py
"""

import os
import sys
import unittest
import time
import threading
import json
from unittest.mock import MagicMock, patch

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from context_core import ContextSource, ContextPriority, ContextData
from context_state import ContextStateType, ContextSnapshot
from context_memory_adapter import (
    MemoryType, MemoryQuery, MemoryResult, MemoryConfig, ContextMemoryAdapter
)


class TestMemoryQuery(unittest.TestCase):
    """Tests für die MemoryQuery-Klasse"""
    
    def test_init(self):
        """Test der Initialisierung"""
        query = MemoryQuery(
            query_type=MemoryType.SEMANTIC,
            query_content="What is the capital of France?",
            filters={"time_range": [1000, 2000]},
            max_results=10,
            priority=80,
            context_source=ContextSource.MEMORY,
            metadata={"meta": "data"}
        )
        
        self.assertEqual(query.query_type, MemoryType.SEMANTIC)
        self.assertEqual(query.query_content, "What is the capital of France?")
        self.assertEqual(query.filters, {"time_range": [1000, 2000]})
        self.assertEqual(query.max_results, 10)
        self.assertEqual(query.priority, 80)
        self.assertEqual(query.context_source, ContextSource.MEMORY)
        self.assertEqual(query.metadata, {"meta": "data"})
        self.assertIsNotNone(query.query_id)
        self.assertIsNotNone(query.timestamp)
    
    def test_default_values(self):
        """Test der Standardwerte"""
        query = MemoryQuery(
            query_type=MemoryType.SEMANTIC,
            query_content="What is the capital of France?"
        )
        
        self.assertEqual(query.query_type, MemoryType.SEMANTIC)
        self.assertEqual(query.query_content, "What is the capital of France?")
        self.assertEqual(query.filters, {})
        self.assertEqual(query.max_results, 5)
        self.assertEqual(query.priority, 50)
        self.assertEqual(query.context_source, ContextSource.MEMORY)
        self.assertEqual(query.metadata, {})
        self.assertIsNotNone(query.query_id)
        self.assertIsNotNone(query.timestamp)


class TestMemoryResult(unittest.TestCase):
    """Tests für die MemoryResult-Klasse"""
    
    def test_init(self):
        """Test der Initialisierung"""
        result = MemoryResult(
            query_id="test_query_id",
            memory_type=MemoryType.SEMANTIC,
            content={"answer": "Paris is the capital of France"},
            relevance_score=0.95,
            source_reference="knowledge_base_1",
            metadata={"confidence": 0.9}
        )
        
        self.assertEqual(result.query_id, "test_query_id")
        self.assertEqual(result.memory_type, MemoryType.SEMANTIC)
        self.assertEqual(result.content, {"answer": "Paris is the capital of France"})
        self.assertEqual(result.relevance_score, 0.95)
        self.assertEqual(result.source_reference, "knowledge_base_1")
        self.assertEqual(result.metadata, {"confidence": 0.9})
        self.assertIsNotNone(result.result_id)
        self.assertIsNotNone(result.timestamp)
    
    def test_default_values(self):
        """Test der Standardwerte"""
        result = MemoryResult(
            query_id="test_query_id",
            memory_type=MemoryType.SEMANTIC,
            content={"answer": "Paris is the capital of France"}
        )
        
        self.assertEqual(result.query_id, "test_query_id")
        self.assertEqual(result.memory_type, MemoryType.SEMANTIC)
        self.assertEqual(result.content, {"answer": "Paris is the capital of France"})
        self.assertEqual(result.relevance_score, 1.0)
        self.assertIsNone(result.source_reference)
        self.assertEqual(result.metadata, {})
        self.assertIsNotNone(result.result_id)
        self.assertIsNotNone(result.timestamp)


class TestMemoryConfig(unittest.TestCase):
    """Tests für die MemoryConfig-Klasse"""
    
    def test_init(self):
        """Test der Initialisierung"""
        config = MemoryConfig(
            memex_path="/path/to/memex",
            memex_interface_class="MemexInterface",
            connection_retry_interval_ms=6000,
            max_connection_retries=5,
            query_timeout_ms=5000,
            max_concurrent_queries=8,
            cache_enabled=True,
            cache_size=1000,
            cache_ttl_seconds=3600,
            memory_type_weights={
                MemoryType.EPISODIC: 0.8,
                MemoryType.SEMANTIC: 0.7,
                MemoryType.PROCEDURAL: 0.6,
                MemoryType.EMOTIONAL: 0.5
            }
        )
        
        self.assertEqual(config.memex_path, "/path/to/memex")
        self.assertEqual(config.memex_interface_class, "MemexInterface")
        self.assertEqual(config.connection_retry_interval_ms, 6000)
        self.assertEqual(config.max_connection_retries, 5)
        self.assertEqual(config.query_timeout_ms, 5000)
        self.assertEqual(config.max_concurrent_queries, 8)
        self.assertTrue(config.cache_enabled)
        self.assertEqual(config.cache_size, 1000)
        self.assertEqual(config.cache_ttl_seconds, 3600)
        self.assertEqual(config.memory_type_weights[MemoryType.EPISODIC], 0.8)
        self.assertEqual(config.memory_type_weights[MemoryType.SEMANTIC], 0.7)
        self.assertEqual(config.memory_type_weights[MemoryType.PROCEDURAL], 0.6)
        self.assertEqual(config.memory_type_weights[MemoryType.EMOTIONAL], 0.5)
    
    def test_default_values(self):
        """Test der Standardwerte"""
        config = MemoryConfig()
        
        self.assertEqual(config.memex_path, "/vXor_Modules/VX-MEMEX")
        self.assertEqual(config.memex_interface_class, "MemexInterface")
        self.assertEqual(config.connection_retry_interval_ms, 5000)
        self.assertEqual(config.max_connection_retries, 10)
        self.assertEqual(config.query_timeout_ms, 2000)
        self.assertEqual(config.max_concurrent_queries, 4)
        self.assertTrue(config.cache_enabled)
        self.assertEqual(config.cache_size, 100)
        self.assertEqual(config.cache_ttl_seconds, 300)
        
        # Prüfe, ob alle Speichertypen in memory_type_weights enthalten sind
        for memory_type in MemoryType:
            self.assertIn(memory_type, config.memory_type_weights)


class TestContextMemoryAdapter(unittest.TestCase):
    """Tests für die ContextMemoryAdapter-Klasse"""
    
    def setUp(self):
        """Setup für die Tests"""
        # Erstelle Mock für ContextCore
        self.mock_context_core = MagicMock()
        self.mock_context_processor = MagicMock()
        self.mock_context_state = MagicMock()
        self.mock_context_core.context_processor = self.mock_context_processor
        self.mock_context_core.context_state = self.mock_context_state
        
        # Erstelle ContextMemoryAdapter mit Mock
        self.memory_adapter = ContextMemoryAdapter(self.mock_context_core)
        
        # Deaktiviere Auto-Connect für Tests
        self.memory_adapter.config.auto_connect = False
    
    def tearDown(self):
        """Cleanup nach den Tests"""
        if self.memory_adapter.active:
            self.memory_adapter.stop()
    
    def test_init(self):
        """Test der Initialisierung"""
        self.assertEqual(self.memory_adapter.context_core, self.mock_context_core)
        self.assertIsInstance(self.memory_adapter.config, MemoryConfig)
        self.assertFalse(self.memory_adapter.active)
        self.assertIsNone(self.memory_adapter.memex_connection)
        self.assertIsNone(self.memory_adapter.memex_interface)
        self.assertIsNone(self.memory_adapter.query_thread)
        self.assertEqual(self.memory_adapter.query_callbacks, {})
        self.assertEqual(self.memory_adapter.active_queries, set())
        self.assertEqual(self.memory_adapter.query_results, {})
        self.assertEqual(self.memory_adapter.performance_metrics['queries_processed'], 0)
    
    @patch('importlib.util.find_spec')
    @patch('importlib.import_module')
    def test_connect_to_memex(self, mock_import_module, mock_find_spec):
        """Test der Verbindung zu MEMEX"""
        # Setup Mocks
        mock_find_spec.return_value = MagicMock()
        mock_module = MagicMock()
        mock_interface = MagicMock()
        mock_module.MemexInterface = MagicMock(return_value=mock_interface)
        mock_import_module.return_value = mock_module
        
        # Verbinde zu MEMEX
        result = self.memory_adapter.connect_to_memex()
        
        # Prüfe Ergebnis
        self.assertTrue(result)
        self.assertEqual(self.memory_adapter.memex_connection, mock_module)
        self.assertEqual(self.memory_adapter.memex_interface, mock_interface)
    
    def test_disconnect_from_memex(self):
        """Test der Trennung von MEMEX"""
        # Setup: Füge eine Mock-Verbindung hinzu
        mock_module = MagicMock()
        mock_interface = MagicMock()
        self.memory_adapter.memex_connection = mock_module
        self.memory_adapter.memex_interface = mock_interface
        
        # Trenne Verbindung
        result = self.memory_adapter.disconnect_from_memex()
        
        # Prüfe Ergebnis
        self.assertTrue(result)
        self.assertIsNone(self.memory_adapter.memex_connection)
        self.assertIsNone(self.memory_adapter.memex_interface)
        
        # Prüfe, ob cleanup aufgerufen wurde
        mock_interface.cleanup.assert_called_once()
    
    def test_start_stop(self):
        """Test des Starts und Stopps"""
        # Setup: Füge eine Mock-Verbindung hinzu
        mock_module = MagicMock()
        mock_interface = MagicMock()
        self.memory_adapter.memex_connection = mock_module
        self.memory_adapter.memex_interface = mock_interface
        
        # Start
        self.memory_adapter.start()
        self.assertTrue(self.memory_adapter.active)
        self.assertIsNotNone(self.memory_adapter.query_thread)
        self.assertTrue(self.memory_adapter.query_thread.is_alive())
        
        # Stop
        self.memory_adapter.stop()
        self.assertFalse(self.memory_adapter.active)
        time.sleep(0.1)  # Warte kurz, damit der Thread beendet werden kann
        self.assertFalse(self.memory_adapter.query_thread.is_alive())
    
    def test_query_memory(self):
        """Test der Speicherabfrage"""
        # Setup: Füge eine Mock-Verbindung hinzu und starte den Adapter
        mock_module = MagicMock()
        mock_interface = MagicMock()
        self.memory_adapter.memex_connection = mock_module
        self.memory_adapter.memex_interface = mock_interface
        self.memory_adapter.start()
        
        # Erstelle Mock-Callback
        callback = MagicMock()
        
        # Erstelle Abfrage
        query = MemoryQuery(
            query_type=MemoryType.SEMANTIC,
            query_content="What is the capital of France?"
        )
        
        # Führe Abfrage durch
        query_id = self.memory_adapter.query_memory(query, callback)
        
        # Prüfe Ergebnis
        self.assertIsNotNone(query_id)
        self.assertEqual(query_id, query.query_id)
        self.assertIn(query_id, self.memory_adapter.query_callbacks)
        self.assertEqual(self.memory_adapter.query_callbacks[query_id], callback)
        self.assertIn(query_id, self.memory_adapter.active_queries)
    
    def test_process_query_result(self):
        """Test der Verarbeitung von Abfrageergebnissen"""
        # Setup: Füge eine Mock-Verbindung hinzu und starte den Adapter
        mock_module = MagicMock()
        mock_interface = MagicMock()
        self.memory_adapter.memex_connection = mock_module
        self.memory_adapter.memex_interface = mock_interface
        self.memory_adapter.start()
        
        # Erstelle Mock-Callback
        callback = MagicMock()
        
        # Erstelle Abfrage
        query = MemoryQuery(
            query_type=MemoryType.SEMANTIC,
            query_content="What is the capital of France?"
        )
        
        # Registriere Callback
        self.memory_adapter.query_callbacks[query.query_id] = callback
        self.memory_adapter.active_queries.add(query.query_id)
        
        # Erstelle Ergebnis
        result = MemoryResult(
            query_id=query.query_id,
            memory_type=MemoryType.SEMANTIC,
            content={"answer": "Paris is the capital of France"}
        )
        
        # Verarbeite Ergebnis
        self.memory_adapter._process_query_result(result)
        
        # Prüfe, ob Callback aufgerufen wurde
        callback.assert_called_once_with(result)
        
        # Prüfe, ob Ergebnis gespeichert wurde
        self.assertIn(query.query_id, self.memory_adapter.query_results)
        self.assertEqual(self.memory_adapter.query_results[query.query_id], result)
        
        # Prüfe, ob die Abfrage aus den aktiven Abfragen entfernt wurde
        self.assertNotIn(query.query_id, self.memory_adapter.active_queries)
        
        # Prüfe, ob die Performance-Metriken aktualisiert wurden
        self.assertEqual(self.memory_adapter.performance_metrics['queries_processed'], 1)
    
    def test_get_query_result(self):
        """Test des Abrufs von Abfrageergebnissen"""
        # Erstelle Abfrage und Ergebnis
        query_id = "test_query_id"
        result = MemoryResult(
            query_id=query_id,
            memory_type=MemoryType.SEMANTIC,
            content={"answer": "Paris is the capital of France"}
        )
        
        # Speichere Ergebnis
        self.memory_adapter.query_results[query_id] = result
        
        # Rufe Ergebnis ab
        retrieved_result = self.memory_adapter.get_query_result(query_id)
        
        # Prüfe Ergebnis
        self.assertEqual(retrieved_result, result)
        
        # Teste nicht existierende Abfrage
        nonexistent_result = self.memory_adapter.get_query_result("nonexistent_id")
        self.assertIsNone(nonexistent_result)
    
    def test_query_memory_sync(self):
        """Test der synchronen Speicherabfrage"""
        # Setup: Füge eine Mock-Verbindung hinzu und starte den Adapter
        mock_module = MagicMock()
        mock_interface = MagicMock()
        
        # Konfiguriere Mock-Interface, um ein Ergebnis zurückzugeben
        mock_result = MemoryResult(
            query_id="test_query_id",
            memory_type=MemoryType.SEMANTIC,
            content={"answer": "Paris is the capital of France"}
        )
        mock_interface.query_memory.return_value = mock_result
        
        self.memory_adapter.memex_connection = mock_module
        self.memory_adapter.memex_interface = mock_interface
        
        # Erstelle Abfrage
        query = MemoryQuery(
            query_type=MemoryType.SEMANTIC,
            query_content="What is the capital of France?"
        )
        
        # Führe synchrone Abfrage durch
        result = self.memory_adapter.query_memory_sync(query)
        
        # Prüfe Ergebnis
        self.assertIsNotNone(result)
        self.assertEqual(result.content, {"answer": "Paris is the capital of France"})
        
        # Prüfe, ob die Abfrage an die Schnittstelle weitergeleitet wurde
        mock_interface.query_memory.assert_called_once()
        args, _ = mock_interface.query_memory.call_args
        passed_query = args[0]
        self.assertEqual(passed_query.query_content, "What is the capital of France?")
    
    def test_store_context_in_memory(self):
        """Test der Speicherung von Kontext im Speicher"""
        # Setup: Füge eine Mock-Verbindung hinzu
        mock_module = MagicMock()
        mock_interface = MagicMock()
        mock_interface.store_memory.return_value = True
        self.memory_adapter.memex_connection = mock_module
        se
(Content truncated due to size limit. Use line ranges to read in chunks)