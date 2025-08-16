#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: Unit Tests für focus_router.py
"""

import os
import sys
import unittest
import time
import threading
from unittest.mock import MagicMock, patch

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from focus_router import (
    FocusType, FocusTarget, FocusState, FocusCommand, FocusRouter
)


class TestFocusState(unittest.TestCase):
    """Tests für die FocusState-Klasse"""
    
    def test_init(self):
        """Test der Initialisierung"""
        state = FocusState(
            target=FocusTarget.VISION,
            focus_type=FocusType.SENSORY,
            intensity=0.8,
            metadata={"meta": "data"}
        )
        
        self.assertEqual(state.target, FocusTarget.VISION)
        self.assertEqual(state.focus_type, FocusType.SENSORY)
        self.assertEqual(state.intensity, 0.8)
        self.assertEqual(state.metadata, {"meta": "data"})
        self.assertIsNone(state.agent_id)
    
    def test_post_init_validation(self):
        """Test der Validierung nach Initialisierung"""
        # Ungültiger Fokustyp
        with self.assertRaises(TypeError):
            FocusState(
                target=FocusTarget.VISION,
                focus_type="SENSORY",  # Sollte ein FocusType-Enum sein
                intensity=0.8
            )
        
        # Ungültige Intensität (zu hoch)
        with self.assertRaises(ValueError):
            FocusState(
                target=FocusTarget.VISION,
                focus_type=FocusType.SENSORY,
                intensity=1.5  # Sollte zwischen 0.0 und 1.0 liegen
            )
        
        # Ungültige Intensität (zu niedrig)
        with self.assertRaises(ValueError):
            FocusState(
                target=FocusTarget.VISION,
                focus_type=FocusType.SENSORY,
                intensity=-0.5  # Sollte zwischen 0.0 und 1.0 liegen
            )
    
    def test_agent_id_conversion(self):
        """Test der Konvertierung von String-Target zu Agent-ID"""
        state = FocusState(
            target="agent1",  # String-Target wird zu Agent-ID
            focus_type=FocusType.COGNITIVE,
            intensity=0.7
        )
        
        self.assertEqual(state.target, FocusTarget.AGENT)
        self.assertEqual(state.agent_id, "agent1")


class TestFocusCommand(unittest.TestCase):
    """Tests für die FocusCommand-Klasse"""
    
    def test_init(self):
        """Test der Initialisierung"""
        command = FocusCommand(
            target=FocusTarget.VISION,
            focus_type=FocusType.SENSORY,
            intensity_delta=0.5,
            priority=70,
            source="test",
            metadata={"meta": "data"}
        )
        
        self.assertEqual(command.target, FocusTarget.VISION)
        self.assertEqual(command.focus_type, FocusType.SENSORY)
        self.assertEqual(command.intensity_delta, 0.5)
        self.assertEqual(command.priority, 70)
        self.assertEqual(command.source, "test")
        self.assertEqual(command.metadata, {"meta": "data"})
    
    def test_post_init_validation(self):
        """Test der Validierung nach Initialisierung"""
        # Ungültiger Fokustyp
        with self.assertRaises(TypeError):
            FocusCommand(
                target=FocusTarget.VISION,
                focus_type="SENSORY",  # Sollte ein FocusType-Enum sein
                intensity_delta=0.5,
                priority=70,
                source="test"
            )
        
        # Ungültiges Intensitäts-Delta (zu hoch)
        with self.assertRaises(ValueError):
            FocusCommand(
                target=FocusTarget.VISION,
                focus_type=FocusType.SENSORY,
                intensity_delta=1.5,  # Sollte zwischen -1.0 und 1.0 liegen
                priority=70,
                source="test"
            )
        
        # Ungültiges Intensitäts-Delta (zu niedrig)
        with self.assertRaises(ValueError):
            FocusCommand(
                target=FocusTarget.VISION,
                focus_type=FocusType.SENSORY,
                intensity_delta=-1.5,  # Sollte zwischen -1.0 und 1.0 liegen
                priority=70,
                source="test"
            )
        
        # Ungültige Priorität (zu hoch)
        with self.assertRaises(ValueError):
            FocusCommand(
                target=FocusTarget.VISION,
                focus_type=FocusType.SENSORY,
                intensity_delta=0.5,
                priority=110,  # Sollte zwischen 0 und 100 liegen
                source="test"
            )
        
        # Ungültige Priorität (zu niedrig)
        with self.assertRaises(ValueError):
            FocusCommand(
                target=FocusTarget.VISION,
                focus_type=FocusType.SENSORY,
                intensity_delta=0.5,
                priority=-10,  # Sollte zwischen 0 und 100 liegen
                source="test"
            )


class TestFocusRouter(unittest.TestCase):
    """Tests für die FocusRouter-Klasse"""
    
    def setUp(self):
        """Setup für die Tests"""
        self.router = FocusRouter()
    
    def tearDown(self):
        """Cleanup nach den Tests"""
        if self.router.active:
            self.router.stop()
    
    def test_init(self):
        """Test der Initialisierung"""
        self.assertFalse(self.router.active)
        self.assertIsNone(self.router.processing_thread)
        self.assertEqual(self.router.focus_states, {})
        self.assertEqual(self.router.focus_change_callbacks, {})
        self.assertEqual(self.router.global_focus_change_callbacks, [])
        self.assertEqual(self.router.active_agents, set())
        self.assertEqual(self.router.performance_metrics['processed_commands_count'], 0)
    
    def test_start_stop(self):
        """Test des Starts und Stopps"""
        # Start
        self.router.start()
        self.assertTrue(self.router.active)
        self.assertIsNotNone(self.router.processing_thread)
        self.assertTrue(self.router.processing_thread.is_alive())
        
        # Stop
        self.router.stop()
        self.assertFalse(self.router.active)
        time.sleep(0.1)  # Warte kurz, damit der Thread beendet werden kann
        self.assertFalse(self.router.processing_thread.is_alive())
    
    def test_submit_focus_command(self):
        """Test der Fokuskommando-Übermittlung"""
        self.router.start()
        
        # Erstelle Testkommando
        command = FocusCommand(
            target=FocusTarget.VISION,
            focus_type=FocusType.SENSORY,
            intensity_delta=0.5,
            priority=70,
            source="test"
        )
        
        # Übermittle Kommando
        result = self.router.submit_focus_command(command)
        self.assertTrue(result)
        
        # Prüfe, ob die Queue nicht leer ist
        self.assertFalse(self.router.focus_command_queue.empty())
    
    def test_register_agent(self):
        """Test der Agentenregistrierung"""
        self.router.register_agent("test_agent")
        self.assertIn("test_agent", self.router.active_agents)
    
    def test_unregister_agent(self):
        """Test der Agentenabmeldung"""
        self.router.register_agent("test_agent")
        self.router.unregister_agent("test_agent")
        self.assertNotIn("test_agent", self.router.active_agents)
    
    def test_register_focus_change_callback(self):
        """Test der Callback-Registrierung"""
        # Erstelle Mock-Callback
        callback = MagicMock()
        
        # Registriere globalen Callback
        self.router.register_focus_change_callback(callback)
        self.assertIn(callback, self.router.global_focus_change_callbacks)
        
        # Registriere spezifischen Callback
        self.router.register_focus_change_callback(
            callback, target=FocusTarget.VISION, focus_type=FocusType.SENSORY
        )
        key = (FocusTarget.VISION, FocusType.SENSORY)
        self.assertIn(key, self.router.focus_change_callbacks)
        self.assertIn(callback, self.router.focus_change_callbacks[key])
    
    def test_get_current_focus(self):
        """Test des Abrufs des aktuellen Fokus"""
        self.router.start()
        
        # Erstelle Testkommando
        command = FocusCommand(
            target=FocusTarget.VISION,
            focus_type=FocusType.SENSORY,
            intensity_delta=0.5,
            priority=70,
            source="test"
        )
        
        # Übermittle Kommando
        self.router.submit_focus_command(command)
        
        # Warte kurz, damit das Kommando verarbeitet werden kann
        time.sleep(0.1)
        
        # Rufe aktuellen Fokus ab
        focus = self.router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        self.assertIsNotNone(focus)
        self.assertEqual(focus.target, FocusTarget.VISION)
        self.assertEqual(focus.focus_type, FocusType.SENSORY)
        self.assertAlmostEqual(focus.intensity, 0.5, delta=0.01)
    
    def test_get_all_focus_states(self):
        """Test des Abrufs aller Fokuszustände"""
        self.router.start()
        
        # Erstelle Testkommandos
        command1 = FocusCommand(
            target=FocusTarget.VISION,
            focus_type=FocusType.SENSORY,
            intensity_delta=0.5,
            priority=70,
            source="test"
        )
        
        command2 = FocusCommand(
            target=FocusTarget.AUDIO,
            focus_type=FocusType.SENSORY,
            intensity_delta=0.3,
            priority=60,
            source="test"
        )
        
        # Übermittle Kommandos
        self.router.submit_focus_command(command1)
        self.router.submit_focus_command(command2)
        
        # Warte kurz, damit die Kommandos verarbeitet werden können
        time.sleep(0.1)
        
        # Rufe alle Fokuszustände ab
        states = self.router.get_all_focus_states()
        self.assertEqual(len(states), 2)
    
    def test_get_top_focus_targets(self):
        """Test des Abrufs der Top-Fokusziele"""
        self.router.start()
        
        # Erstelle Testkommandos mit unterschiedlichen Intensitäten
        command1 = FocusCommand(
            target=FocusTarget.VISION,
            focus_type=FocusType.SENSORY,
            intensity_delta=0.8,
            priority=70,
            source="test"
        )
        
        command2 = FocusCommand(
            target=FocusTarget.AUDIO,
            focus_type=FocusType.SENSORY,
            intensity_delta=0.3,
            priority=60,
            source="test"
        )
        
        command3 = FocusCommand(
            target=FocusTarget.LANGUAGE,
            focus_type=FocusType.SENSORY,
            intensity_delta=0.5,
            priority=65,
            source="test"
        )
        
        # Übermittle Kommandos
        self.router.submit_focus_command(command1)
        self.router.submit_focus_command(command2)
        self.router.submit_focus_command(command3)
        
        # Warte kurz, damit die Kommandos verarbeitet werden können
        time.sleep(0.1)
        
        # Rufe Top-Fokusziele ab
        top_targets = self.router.get_top_focus_targets(limit=2)
        self.assertEqual(len(top_targets), 2)
        
        # Prüfe, ob die Ziele nach Intensität sortiert sind
        self.assertEqual(top_targets[0][0], FocusTarget.VISION)  # Höchste Intensität
    
    def test_focus_decay(self):
        """Test der Fokusabnahme"""
        self.router.start()
        
        # Erstelle Testkommando
        command = FocusCommand(
            target=FocusTarget.VISION,
            focus_type=FocusType.SENSORY,
            intensity_delta=0.5,
            priority=70,
            source="test"
        )
        
        # Übermittle Kommando
        self.router.submit_focus_command(command)
        
        # Warte kurz, damit das Kommando verarbeitet werden kann
        time.sleep(0.1)
        
        # Rufe initialen Fokus ab
        initial_focus = self.router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        initial_intensity = initial_focus.intensity
        
        # Warte länger, damit die Fokusabnahme wirken kann
        time.sleep(1.0)
        
        # Rufe aktuellen Fokus ab
        current_focus = self.router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        
        # Prüfe, ob die Intensität abgenommen hat
        if current_focus:  # Könnte None sein, wenn die Intensität unter den Schwellenwert gefallen ist
            self.assertLess(current_focus.intensity, initial_intensity)
    
    def test_create_focus_shift(self):
        """Test der Fokusverschiebung"""
        self.router.start()
        
        # Erstelle initialen Fokus
        command = FocusCommand(
            target=FocusTarget.VISION,
            focus_type=FocusType.SENSORY,
            intensity_delta=0.8,
            priority=70,
            source="test"
        )
        
        # Übermittle Kommando
        self.router.submit_focus_command(command)
        
        # Warte kurz, damit das Kommando verarbeitet werden kann
        time.sleep(0.1)
        
        # Verschiebe Fokus
        result = self.router.create_focus_shift(
            from_target=FocusTarget.VISION,
            to_target=FocusTarget.AUDIO,
            focus_type=FocusType.SENSORY,
            shift_amount=0.3
        )
        
        self.assertTrue(result)
        
        # Warte kurz, damit die Verschiebung verarbeitet werden kann
        time.sleep(0.1)
        
        # Rufe Fokuszustände ab
        vision_focus = self.router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        audio_focus = self.router.get_current_focus(FocusTarget.AUDIO, FocusType.SENSORY)
        
        # Prüfe, ob die Verschiebung korrekt durchgeführt wurde
        self.assertIsNotNone(vision_focus)
        self.assertIsNotNone(audio_focus)
        self.assertAlmostEqual(vision_focus.intensity, 0.5, delta=0.01)  # 0.8 - 0.3
        self.assertAlmostEqual(audio_focus.intensity, 0.3, delta=0.01)
    
    def test_distribute_focus(self):
        """Test der Fokusverteilung"""
        self.router.start()
        
        # Verteile Fokus
        result = self.router.distribute_focus(
            targets=[FocusTarget.VISION, FocusTarget.AUDIO, FocusTarget.LANGUAGE],
            focus_type=FocusType.SENSORY,
            total_intensity=0.9,
            weights=[0.5, 0.3, 0.2]
        )
        
        self.assertTrue(result)
        
        # Warte kurz, damit die Verteilung verarbeitet werden kann
        time.sleep(0.1)
        
        # Rufe Fokuszustände ab
        vision_focus = self.router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        audio_focus = self.router.get_current_focus(FocusTarget.AUDIO, FocusType.SENSORY)
        language_focus = self.router.get_current_focus(FocusTarget.LANGUAGE, FocusType.SENSORY)
        
        # Prüfe, ob die Verteilung korrekt durchgeführt wurde
        self.assertIsNotNone(vision_focus)
        self.assertIsNotNone(audio_focus)
        self.assertIsNotNone(language_focus)
        self.assertAlmostEqual(vision_focus.intensity, 0.45, delta=0.01)  # 0.9 * 0.5
        self.assertAlmostEqual(audio_focus.intensity, 0.27, delta=0.01)   # 0.9 * 0.3
        self.assertAlmostEqual(language_focus.intensity, 0.18, delta=0.01)  # 0.9 * 0.2
    
    def test_clear_focus(self):
        """Test der Fokuslöschung"""
        self.router.start()
        
        # Erstelle Testkommandos
        command1 = FocusCommand(
            target=FocusTarget.VISION,
            focus_type=FocusType.SENSORY,
            intensity_delta=0.5,
            priority=70,
            source="test"
        )
        
        command2 = FocusCommand(
            target=FocusTarget.AUDIO,
            focus_type=FocusType.SENSORY,
            intensity_delta=0.3,
            p
(Content truncated due to size limit. Use line ranges to read in chunks)