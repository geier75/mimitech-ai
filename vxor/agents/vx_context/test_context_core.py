#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: Unit Tests für context_core.py
"""

import os
import sys
import unittest
import time
import threading
from unittest.mock import MagicMock, patch

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from context_core import (
    ContextSource, ContextPriority, ContextData, ContextProcessor,
    ContextCore, get_context_core
)


class TestContextData(unittest.TestCase):
    """Tests für die ContextData-Klasse"""
    
    def test_init(self):
        """Test der Initialisierung"""
        data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.HIGH,
            data={"test": "value"},
            metadata={"meta": "data"}
        )
        
        self.assertEqual(data.source, ContextSource.VISUAL)
        self.assertEqual(data.priority, ContextPriority.HIGH)
        self.assertEqual(data.data, {"test": "value"})
        self.assertEqual(data.metadata, {"meta": "data"})
        self.assertIsNone(data.processing_time_ms)
    
    def test_post_init_validation(self):
        """Test der Validierung nach Initialisierung"""
        # Ungültige Quelle
        with self.assertRaises(TypeError):
            ContextData(
                source="VISUAL",  # Sollte ein ContextSource-Enum sein
                priority=ContextPriority.HIGH
            )
        
        # Ungültige Priorität
        with self.assertRaises(TypeError):
            ContextData(
                source=ContextSource.VISUAL,
                priority="HIGH"  # Sollte ein ContextPriority-Enum sein
            )


class TestContextProcessor(unittest.TestCase):
    """Tests für die ContextProcessor-Klasse"""
    
    def setUp(self):
        """Setup für die Tests"""
        self.processor = ContextProcessor()
    
    def tearDown(self):
        """Cleanup nach den Tests"""
        if self.processor.active:
            self.processor.stop()
    
    def test_init(self):
        """Test der Initialisierung"""
        self.assertFalse(self.processor.active)
        self.assertIsNone(self.processor.processing_thread)
        self.assertEqual(self.processor.registered_handlers, {})
        self.assertEqual(self.processor.current_context_state, {})
        self.assertEqual(self.processor.performance_metrics['processed_items_count'], 0)
    
    def test_start_stop(self):
        """Test des Starts und Stopps"""
        # Start
        self.processor.start()
        self.assertTrue(self.processor.active)
        self.assertIsNotNone(self.processor.processing_thread)
        self.assertTrue(self.processor.processing_thread.is_alive())
        
        # Stop
        self.processor.stop()
        self.assertFalse(self.processor.active)
        time.sleep(0.1)  # Warte kurz, damit der Thread beendet werden kann
        self.assertFalse(self.processor.processing_thread.is_alive())
    
    def test_submit_context(self):
        """Test der Kontextübermittlung"""
        self.processor.start()
        
        # Erstelle Testdaten
        context_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.HIGH,
            data={"test": "value"}
        )
        
        # Übermittle Kontext
        result = self.processor.submit_context(context_data)
        self.assertTrue(result)
        
        # Prüfe, ob die Queue nicht leer ist
        self.assertFalse(self.processor.context_queue.empty())
    
    def test_register_handler(self):
        """Test der Handler-Registrierung"""
        # Erstelle Mock-Handler
        handler = MagicMock()
        
        # Registriere Handler
        self.processor.register_handler(ContextSource.VISUAL, handler)
        
        # Prüfe, ob Handler registriert wurde
        self.assertIn(ContextSource.VISUAL, self.processor.registered_handlers)
        self.assertEqual(len(self.processor.registered_handlers[ContextSource.VISUAL]), 1)
        self.assertEqual(self.processor.registered_handlers[ContextSource.VISUAL][0], handler)
    
    def test_process_context_queue(self):
        """Test der Kontextverarbeitung"""
        self.processor.start()
        
        # Erstelle Mock-Handler
        handler = MagicMock()
        
        # Registriere Handler
        self.processor.register_handler(ContextSource.VISUAL, handler)
        
        # Erstelle Testdaten
        context_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.HIGH,
            data={"test": "value"}
        )
        
        # Übermittle Kontext
        self.processor.submit_context(context_data)
        
        # Warte kurz, damit der Kontext verarbeitet werden kann
        time.sleep(0.1)
        
        # Prüfe, ob Handler aufgerufen wurde
        handler.assert_called_once()
        
        # Prüfe, ob der Kontextzustand aktualisiert wurde
        self.assertIn(ContextSource.VISUAL.name, self.processor.current_context_state)
        self.assertEqual(
            self.processor.current_context_state[ContextSource.VISUAL.name]['data'],
            {"test": "value"}
        )
        
        # Prüfe, ob die Performance-Metriken aktualisiert wurden
        self.assertEqual(self.processor.performance_metrics['processed_items_count'], 1)
        self.assertGreater(self.processor.performance_metrics['avg_processing_time_ms'], 0)


class TestContextCore(unittest.TestCase):
    """Tests für die ContextCore-Klasse"""
    
    def setUp(self):
        """Setup für die Tests"""
        # Erstelle eine neue Instanz für jeden Test
        self.core = ContextCore()
    
    def tearDown(self):
        """Cleanup nach den Tests"""
        if self.core.context_processor.active:
            self.core.stop()
    
    def test_init(self):
        """Test der Initialisierung"""
        self.assertIsNotNone(self.core.context_processor)
        self.assertIsNone(self.core.focus_router)
        self.assertIsNone(self.core.context_analyzer)
        self.assertIsNone(self.core.context_bridge)
        self.assertIsNone(self.core.context_state)
        self.assertIsNone(self.core.context_memory_adapter)
        self.assertTrue(self.core.config['apple_silicon_optimized'])
    
    def test_start_stop(self):
        """Test des Starts und Stopps"""
        # Start
        result = self.core.start()
        self.assertTrue(result)
        self.assertTrue(self.core.context_processor.active)
        
        # Stop
        result = self.core.stop()
        self.assertTrue(result)
        self.assertFalse(self.core.context_processor.active)
    
    def test_submit_context(self):
        """Test der Kontextübermittlung"""
        self.core.start()
        
        # Übermittle Kontext
        result = self.core.submit_context(
            source=ContextSource.VISUAL,
            data={"test": "value"},
            priority=ContextPriority.HIGH,
            metadata={"meta": "data"}
        )
        
        self.assertTrue(result)
    
    def test_get_performance_metrics(self):
        """Test des Abrufs von Performance-Metriken"""
        metrics = self.core.get_performance_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('avg_processing_time_ms', metrics)
        self.assertIn('max_processing_time_ms', metrics)
        self.assertIn('processed_items_count', metrics)
    
    def test_get_current_context_state(self):
        """Test des Abrufs des aktuellen Kontextzustands"""
        state = self.core.get_current_context_state()
        self.assertIsInstance(state, dict)
    
    def test_singleton(self):
        """Test der Singleton-Implementierung"""
        core1 = get_context_core()
        core2 = get_context_core()
        
        self.assertIs(core1, core2)


if __name__ == '__main__':
    unittest.main()
