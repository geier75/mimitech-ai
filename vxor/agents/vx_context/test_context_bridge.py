#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: Unit Tests für context_bridge.py
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
from focus_router import FocusType, FocusTarget, FocusState, FocusCommand
from context_bridge import (
    ModuleInterface, ModuleConfig, ModuleMessage, ContextBridge
)


class TestModuleConfig(unittest.TestCase):
    """Tests für die ModuleConfig-Klasse"""
    
    def test_init(self):
        """Test der Initialisierung"""
        config = ModuleConfig(
            module_name="TestModule",
            module_path="/path/to/module",
            interface_class="TestInterface",
            enabled=True,
            auto_connect=True,
            polling_interval_ms=200,
            connection_retry_interval_ms=6000,
            max_connection_retries=5,
            context_mapping={"test": ContextSource.VISUAL},
            focus_mapping={"test": FocusTarget.VISION}
        )
        
        self.assertEqual(config.module_name, "TestModule")
        self.assertEqual(config.module_path, "/path/to/module")
        self.assertEqual(config.interface_class, "TestInterface")
        self.assertTrue(config.enabled)
        self.assertTrue(config.auto_connect)
        self.assertEqual(config.polling_interval_ms, 200)
        self.assertEqual(config.connection_retry_interval_ms, 6000)
        self.assertEqual(config.max_connection_retries, 5)
        self.assertEqual(config.context_mapping, {"test": ContextSource.VISUAL})
        self.assertEqual(config.focus_mapping, {"test": FocusTarget.VISION})
    
    def test_default_values(self):
        """Test der Standardwerte"""
        config = ModuleConfig(
            module_name="TestModule",
            module_path="/path/to/module"
        )
        
        self.assertEqual(config.module_name, "TestModule")
        self.assertEqual(config.module_path, "/path/to/module")
        self.assertIsNone(config.interface_class)
        self.assertTrue(config.enabled)
        self.assertTrue(config.auto_connect)
        self.assertEqual(config.polling_interval_ms, 100)
        self.assertEqual(config.connection_retry_interval_ms, 5000)
        self.assertEqual(config.max_connection_retries, 10)
        self.assertEqual(config.context_mapping, {})
        self.assertEqual(config.focus_mapping, {})


class TestModuleMessage(unittest.TestCase):
    """Tests für die ModuleMessage-Klasse"""
    
    def test_init(self):
        """Test der Initialisierung"""
        message = ModuleMessage(
            source_module="SourceModule",
            target_module="TargetModule",
            message_type="test_message",
            data={"test": "data"},
            priority=80,
            metadata={"meta": "data"}
        )
        
        self.assertEqual(message.source_module, "SourceModule")
        self.assertEqual(message.target_module, "TargetModule")
        self.assertEqual(message.message_type, "test_message")
        self.assertEqual(message.data, {"test": "data"})
        self.assertEqual(message.priority, 80)
        self.assertEqual(message.metadata, {"meta": "data"})
        self.assertIsNotNone(message.timestamp)
        self.assertIsNotNone(message.message_id)
    
    def test_default_values(self):
        """Test der Standardwerte"""
        message = ModuleMessage(
            source_module="SourceModule",
            target_module="TargetModule",
            message_type="test_message",
            data={"test": "data"}
        )
        
        self.assertEqual(message.source_module, "SourceModule")
        self.assertEqual(message.target_module, "TargetModule")
        self.assertEqual(message.message_type, "test_message")
        self.assertEqual(message.data, {"test": "data"})
        self.assertEqual(message.priority, 50)  # Standardpriorität
        self.assertEqual(message.metadata, {})
        self.assertIsNotNone(message.timestamp)
        self.assertIsNotNone(message.message_id)


class TestContextBridge(unittest.TestCase):
    """Tests für die ContextBridge-Klasse"""
    
    def setUp(self):
        """Setup für die Tests"""
        # Erstelle Mock für ContextCore
        self.mock_context_core = MagicMock()
        self.mock_context_processor = MagicMock()
        self.mock_focus_router = MagicMock()
        self.mock_context_core.context_processor = self.mock_context_processor
        self.mock_context_core.focus_router = self.mock_focus_router
        
        # Erstelle ContextBridge mit Mock
        self.bridge = ContextBridge(self.mock_context_core)
        
        # Deaktiviere Auto-Connect für Tests
        for interface in ModuleInterface:
            self.bridge.module_configs[interface].auto_connect = False
    
    def tearDown(self):
        """Cleanup nach den Tests"""
        if self.bridge.active:
            self.bridge.stop()
    
    def test_init(self):
        """Test der Initialisierung"""
        self.assertEqual(self.bridge.context_core, self.mock_context_core)
        self.assertEqual(self.bridge.module_connections, {})
        self.assertEqual(self.bridge.module_interfaces, {})
        self.assertIsNotNone(self.bridge.module_configs)
        self.assertTrue(self.bridge.outgoing_messages.empty())
        self.assertTrue(self.bridge.incoming_messages.empty())
        self.assertFalse(self.bridge.active)
        self.assertIsNone(self.bridge.message_processing_thread)
        self.assertEqual(self.bridge.module_polling_threads, {})
        self.assertEqual(self.bridge.message_handlers, {})
        self.assertIsNotNone(self.bridge.thread_pool)
        self.assertEqual(self.bridge.performance_metrics['messages_sent'], 0)
    
    def test_init_default_module_configs(self):
        """Test der Initialisierung von Standardkonfigurationen"""
        # Prüfe, ob für alle Modulschnittstellen Konfigurationen erstellt wurden
        for interface in ModuleInterface:
            self.assertIn(interface, self.bridge.module_configs)
            config = self.bridge.module_configs[interface]
            self.assertIsInstance(config, ModuleConfig)
            self.assertEqual(config.module_name, f"VX-{interface.name}")
            self.assertEqual(config.module_path, f"/vXor_Modules/VX-{interface.name}")
            self.assertIsNotNone(config.interface_class)
            self.assertTrue(config.enabled)
            self.assertIsNotNone(config.context_mapping)
            self.assertIsNotNone(config.focus_mapping)
    
    def test_start_stop(self):
        """Test des Starts und Stopps"""
        # Start
        self.bridge.start()
        self.assertTrue(self.bridge.active)
        self.assertIsNotNone(self.bridge.message_processing_thread)
        self.assertTrue(self.bridge.message_processing_thread.is_alive())
        
        # Stop
        self.bridge.stop()
        self.assertFalse(self.bridge.active)
        time.sleep(0.1)  # Warte kurz, damit der Thread beendet werden kann
        self.assertFalse(self.bridge.message_processing_thread.is_alive())
    
    @patch('importlib.util.find_spec')
    @patch('importlib.import_module')
    def test_connect_to_module(self, mock_import_module, mock_find_spec):
        """Test der Verbindung zu einem Modul"""
        # Setup Mocks
        mock_find_spec.return_value = MagicMock()
        mock_module = MagicMock()
        mock_interface = MagicMock()
        mock_module.TestInterface = MagicMock(return_value=mock_interface)
        mock_import_module.return_value = mock_module
        
        # Konfiguriere Modul
        config = ModuleConfig(
            module_name="TestModule",
            module_path="/path/to/module",
            interface_class="TestInterface"
        )
        self.bridge.module_configs[ModuleInterface.PSI] = config
        
        # Verbinde zu Modul
        result = self.bridge.connect_to_module(ModuleInterface.PSI)
        
        # Prüfe Ergebnis
        self.assertTrue(result)
        self.assertIn(ModuleInterface.PSI, self.bridge.module_connections)
        self.assertEqual(self.bridge.module_connections[ModuleInterface.PSI], mock_module)
        self.assertEqual(self.bridge.module_interfaces[ModuleInterface.PSI], mock_interface)
    
    def test_disconnect_from_module(self):
        """Test der Trennung von einem Modul"""
        # Setup: Füge eine Mock-Verbindung hinzu
        mock_module = MagicMock()
        mock_interface = MagicMock()
        self.bridge.module_connections[ModuleInterface.PSI] = mock_module
        self.bridge.module_interfaces[ModuleInterface.PSI] = mock_interface
        
        # Trenne Verbindung
        result = self.bridge.disconnect_from_module(ModuleInterface.PSI)
        
        # Prüfe Ergebnis
        self.assertTrue(result)
        self.assertNotIn(ModuleInterface.PSI, self.bridge.module_connections)
        self.assertNotIn(ModuleInterface.PSI, self.bridge.module_interfaces)
        
        # Prüfe, ob cleanup aufgerufen wurde
        mock_interface.cleanup.assert_called_once()
    
    def test_send_context_update(self):
        """Test des Sendens eines Kontextupdates"""
        # Setup: Füge eine Mock-Verbindung hinzu
        mock_module = MagicMock()
        mock_interface = MagicMock()
        self.bridge.module_connections[ModuleInterface.PSI] = mock_module
        self.bridge.module_interfaces[ModuleInterface.PSI] = mock_interface
        
        # Erstelle Testdaten
        context_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.HIGH,
            data={"test": "data"},
            metadata={"meta": "data"}
        )
        
        # Sende Kontextupdate
        self.bridge.start()  # Starte Bridge, damit die Nachrichtenverarbeitung läuft
        result = self.bridge.send_context_update(ModuleInterface.PSI, context_data)
        
        # Prüfe Ergebnis
        self.assertTrue(result)
        self.assertFalse(self.bridge.outgoing_messages.empty())
        
        # Hole und prüfe die Nachricht
        _, _, message = self.bridge.outgoing_messages.get()
        self.assertEqual(message.source_module, "VX-CONTEXT")
        self.assertEqual(message.target_module, "PSI")
        self.assertEqual(message.message_type, "context_update")
        self.assertEqual(message.data["source"], "VISUAL")
        self.assertEqual(message.data["priority"], "HIGH")
        self.assertEqual(message.data["data"], {"test": "data"})
        self.assertEqual(message.data["metadata"], {"meta": "data"})
    
    def test_send_focus_update(self):
        """Test des Sendens eines Fokusupdates"""
        # Setup: Füge eine Mock-Verbindung hinzu
        mock_module = MagicMock()
        mock_interface = MagicMock()
        self.bridge.module_connections[ModuleInterface.PSI] = mock_module
        self.bridge.module_interfaces[ModuleInterface.PSI] = mock_interface
        
        # Erstelle Testdaten
        focus_state = FocusState(
            target=FocusTarget.VISION,
            focus_type=FocusType.SENSORY,
            intensity=0.8,
            metadata={"meta": "data"}
        )
        
        # Sende Fokusupdate
        self.bridge.start()  # Starte Bridge, damit die Nachrichtenverarbeitung läuft
        result = self.bridge.send_focus_update(ModuleInterface.PSI, focus_state)
        
        # Prüfe Ergebnis
        self.assertTrue(result)
        self.assertFalse(self.bridge.outgoing_messages.empty())
        
        # Hole und prüfe die Nachricht
        _, _, message = self.bridge.outgoing_messages.get()
        self.assertEqual(message.source_module, "VX-CONTEXT")
        self.assertEqual(message.target_module, "PSI")
        self.assertEqual(message.message_type, "focus_update")
        self.assertEqual(message.data["target"], "VISION")
        self.assertEqual(message.data["focus_type"], "SENSORY")
        self.assertEqual(message.data["intensity"], 0.8)
        self.assertEqual(message.data["metadata"], {"meta": "data"})
        self.assertIsNone(message.data["agent_id"])
    
    def test_register_message_handler(self):
        """Test der Registrierung eines Nachrichtenhandlers"""
        # Erstelle Mock-Handler
        handler = MagicMock()
        
        # Registriere Handler
        self.bridge.register_message_handler("test_message", handler)
        
        # Prüfe, ob Handler registriert wurde
        self.assertIn("test_message", self.bridge.message_handlers)
        self.assertIn(handler, self.bridge.message_handlers["test_message"])
    
    def test_handle_incoming_message(self):
        """Test der Verarbeitung eingehender Nachrichten"""
        # Erstelle Mock-Handler
        handler = MagicMock()
        
        # Registriere Handler
        self.bridge.register_message_handler("test_message", handler)
        
        # Erstelle Testnachricht
        message = ModuleMessage(
            source_module="TestModule",
            target_module="VX-CONTEXT",
            message_type="test_message",
            data={"test": "data"}
        )
        
        # Verarbeite Nachricht
        self.bridge._handle_incoming_message(message)
        
        # Prüfe, ob Handler aufgerufen wurde
        handler.assert_called_once_with(message)
    
    def test_handle_context_update_message(self):
        """Test der Verarbeitung von Kontextupdate-Nachrichten"""
        # Erstelle Testnachricht
        message = ModuleMessage(
            source_module="TestModule",
            target_module="VX-CONTEXT",
            message_type="context_update",
            data={
                "source": "VISUAL",
                "priority": "HIGH",
                "data": {"test": "data"},
                "metadata": {"meta": "data"}
            }
        )
        
        # Verarbeite Nachricht
        self.bridge._handle_context_update_message(message)
        
        # Prüfe, ob Kontext an ContextCore übermittelt wurde
        self.mock_context_processor.submit_context.assert_called_once()
        args, _ = self.mock_context_processor.submit_context.call_args
        context_data = args[0]
        self.assertEqual(context_data.source, ContextSource.VISUAL)
        self.assertEqual(context_data.priority, ContextPriority.HIGH)
        self.assertEqual(context_data.data, {"test": "data"})
        self.assertEqual(context_data.metadata["source_module"], "TestModule")
    
    def test_handle_focus_request_message(self):
        """Test der Verarbeitung von Fokusanfrage-Nachrichten"""
        # Erstelle Testnachricht
        message = ModuleMessage(
            source_module="TestModule",
            target_module="VX-CONTEXT",
            message_type="focus_request",
            data={
                "target": "VISION",
                "focus_type": "SENSORY",
                "intensity_delta": 0.5,
                "priority": 70,
                "metadata": {"meta": "data"}
            }
        )
        
        # Verarbeite Nachricht
        self.bridge._handle_focus_request_message(message)
        
        # Prüfe, ob Fokuskommando an FocusRouter übermittelt wurde
        self.mock_focus_router.submit_focus_command.assert_called_once()
        args, _ = self.mock_focus_router.submit_focus_command.call_args
        focus_command = args[0]
        self.assertEqual(focus_command.target, FocusTarget.VISION)
        self.assertEqual(focus_command.focus_type, FocusType.SENSORY)
        self.assertEqual(focus_command.intensity_delta, 0.5)
        self.assertEqual(focus_command.priority, 70)
        self.assertEqual(focus_command.source, "TestModule")
        self.assertEqual(focus_command.metadata["source_module"], "TestModule")
    
    def test_map_context_priority_to_message_priority(self):
        """Test der Konvertierung von ContextPri
(Content truncated due to size limit. Use line ranges to read in chunks)