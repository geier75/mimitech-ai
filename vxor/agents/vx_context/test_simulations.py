#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: Simulationstests für das VX-CONTEXT System
"""

import os
import sys
import unittest
import time
import threading
import random
import json
from unittest.mock import MagicMock, patch

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from context_core import (
    ContextSource, ContextPriority, ContextData, ContextCore, get_context_core
)
from focus_router import (
    FocusType, FocusTarget, FocusState, FocusCommand, FocusRouter
)
from context_analyzer import (
    AnalysisDimension, AnalysisResult, AnalysisConfig, ContextAnalyzer
)
from context_bridge import (
    ModuleInterface, ModuleConfig, ModuleMessage, ContextBridge
)
from context_state import (
    ContextStateType, ContextStateEntry, ContextSnapshot, ContextState
)
from context_memory_adapter import (
    MemoryType, MemoryQuery, MemoryResult, MemoryConfig, ContextMemoryAdapter
)


class TestMultiStimulusSimulation(unittest.TestCase):
    """Tests für die Simulation von Multi-Stimulus-Eingaben mit Fokusverlagerung"""
    
    def setUp(self):
        """Setup für die Tests"""
        # Initialisiere ContextCore
        self.context_core = ContextCore()
        
        # Initialisiere alle Module
        self.focus_router = FocusRouter()
        self.context_analyzer = ContextAnalyzer(self.context_core)
        self.context_bridge = ContextBridge(self.context_core)
        self.context_state = ContextState(self.context_core)
        self.context_memory_adapter = ContextMemoryAdapter(self.context_core)
        
        # Verbinde Module mit ContextCore
        self.context_core.focus_router = self.focus_router
        self.context_core.context_analyzer = self.context_analyzer
        self.context_core.context_bridge = self.context_bridge
        self.context_core.context_state = self.context_state
        self.context_core.context_memory_adapter = self.context_memory_adapter
        
        # Starte alle Module
        self.context_core.start()
        self.focus_router.start()
        self.context_state.start()
        
        # Registriere Handler
        self.context_core.context_processor.register_handler(
            ContextSource.VISUAL, self.context_analyzer.process_context_data
        )
        self.context_core.context_processor.register_handler(
            ContextSource.LANGUAGE, self.context_analyzer.process_context_data
        )
        self.context_core.context_processor.register_handler(
            ContextSource.INTERNAL, self.context_analyzer.process_context_data
        )
        self.context_core.context_processor.register_handler(
            ContextSource.EXTERNAL, self.context_analyzer.process_context_data
        )
        
        # Registriere State-Handler
        self.context_analyzer.register_analysis_callback(
            self.context_state.process_context_data
        )
        
        # Ergebnisse für Tests
        self.focus_changes = []
        self.context_updates = []
        
        # Registriere Fokusänderungs-Callback
        self.focus_router.register_focus_change_callback(self.on_focus_change)
        
        # Registriere Kontextänderungs-Callback
        self.context_state.register_state_change_handler(self.on_context_update)
    
    def tearDown(self):
        """Cleanup nach den Tests"""
        self.context_core.stop()
        self.focus_router.stop()
        self.context_state.stop()
    
    def on_focus_change(self, focus_state):
        """Callback für Fokusänderungen"""
        self.focus_changes.append(focus_state)
    
    def on_context_update(self, state_type, state_data, source):
        """Callback für Kontextänderungen"""
        self.context_updates.append((state_type, state_data, source))
    
    def test_visual_stimulus_priority(self):
        """Test der Priorisierung visueller Stimuli"""
        # Erstelle visuellen Stimulus mit hoher Priorität
        visual_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.HIGH,
            data={
                "object_detected": "Person",
                "confidence": 0.95,
                "position": [0.5, 0.5],
                "size": [0.3, 0.7]
            },
            metadata={"camera_id": "main"}
        )
        
        # Erstelle sprachlichen Stimulus mit mittlerer Priorität
        language_data = ContextData(
            source=ContextSource.LANGUAGE,
            priority=ContextPriority.MEDIUM,
            data={
                "text": "Hallo, wie geht es dir?",
                "confidence": 0.9,
                "sentiment": "positive"
            },
            metadata={"source": "microphone"}
        )
        
        # Übermittle Stimuli
        self.context_core.submit_context(visual_data)
        self.context_core.submit_context(language_data)
        
        # Warte auf Verarbeitung
        time.sleep(0.5)
        
        # Prüfe, ob der visuelle Stimulus priorisiert wurde
        focus_states = self.focus_router.get_all_focus_states()
        
        # Prüfe, ob Fokuswerte korrekt gesetzt wurden
        visual_focus = self.focus_router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        language_focus = self.focus_router.get_current_focus(FocusTarget.LANGUAGE, FocusType.SENSORY)
        
        self.assertIsNotNone(visual_focus)
        self.assertIsNotNone(language_focus)
        self.assertGreater(visual_focus.intensity, language_focus.intensity)
        
        # Prüfe, ob der Kontext korrekt aktualisiert wurde
        visual_state = self.context_state.get_state(ContextStateType.VISUAL)
        language_state = self.context_state.get_state(ContextStateType.LINGUISTIC)
        
        self.assertIn("object_detected", visual_state)
        self.assertEqual(visual_state["object_detected"], "Person")
        
        self.assertIn("text", language_state)
        self.assertEqual(language_state["text"], "Hallo, wie geht es dir?")
    
    def test_focus_shifting(self):
        """Test der Fokusverlagerung bei sich ändernden Stimuli"""
        # Erstelle initialen visuellen Stimulus
        visual_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.MEDIUM,
            data={
                "object_detected": "Person",
                "confidence": 0.8,
                "position": [0.5, 0.5]
            }
        )
        
        # Übermittle Stimulus
        self.context_core.submit_context(visual_data)
        
        # Warte auf Verarbeitung
        time.sleep(0.2)
        
        # Prüfe initialen Fokus
        initial_visual_focus = self.focus_router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        self.assertIsNotNone(initial_visual_focus)
        initial_intensity = initial_visual_focus.intensity
        
        # Erstelle dringenden externen Stimulus
        external_data = ContextData(
            source=ContextSource.EXTERNAL,
            priority=ContextPriority.CRITICAL,
            data={
                "alert": "Kollisionsgefahr",
                "distance": 1.5,
                "time_to_impact": 2.0
            }
        )
        
        # Übermittle Stimulus
        self.context_core.submit_context(external_data)
        
        # Warte auf Verarbeitung
        time.sleep(0.2)
        
        # Prüfe Fokusverlagerung
        new_visual_focus = self.focus_router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        external_focus = self.focus_router.get_current_focus(FocusTarget.EXTERNAL, FocusType.SENSORY)
        
        self.assertIsNotNone(new_visual_focus)
        self.assertIsNotNone(external_focus)
        
        # Prüfe, ob der Fokus zum externen Stimulus verlagert wurde
        self.assertGreater(external_focus.intensity, new_visual_focus.intensity)
        self.assertLess(new_visual_focus.intensity, initial_intensity)
        
        # Prüfe, ob der Kontext korrekt aktualisiert wurde
        external_state = self.context_state.get_state(ContextStateType.EXTERNAL)
        self.assertIn("alert", external_state)
        self.assertEqual(external_state["alert"], "Kollisionsgefahr")
    
    def test_multi_stimulus_focus_distribution(self):
        """Test der Fokusverteilung bei mehreren gleichzeitigen Stimuli"""
        # Erstelle mehrere Stimuli mit unterschiedlichen Prioritäten
        stimuli = [
            ContextData(
                source=ContextSource.VISUAL,
                priority=ContextPriority.MEDIUM,
                data={"object_detected": "Person", "confidence": 0.85}
            ),
            ContextData(
                source=ContextSource.LANGUAGE,
                priority=ContextPriority.HIGH,
                data={"text": "Achtung!", "confidence": 0.9}
            ),
            ContextData(
                source=ContextSource.INTERNAL,
                priority=ContextPriority.LOW,
                data={"status": "Batterie niedrig", "level": 0.15}
            ),
            ContextData(
                source=ContextSource.EXTERNAL,
                priority=ContextPriority.MEDIUM,
                data={"sound": "Klingeln", "volume": 0.7}
            )
        ]
        
        # Übermittle Stimuli
        for stimulus in stimuli:
            self.context_core.submit_context(stimulus)
        
        # Warte auf Verarbeitung
        time.sleep(0.5)
        
        # Prüfe Fokusverteilung
        focus_states = self.focus_router.get_all_focus_states()
        
        # Prüfe, ob alle Stimuli einen Fokus erhalten haben
        visual_focus = self.focus_router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        language_focus = self.focus_router.get_current_focus(FocusTarget.LANGUAGE, FocusType.SENSORY)
        internal_focus = self.focus_router.get_current_focus(FocusTarget.INTERNAL, FocusType.SENSORY)
        external_focus = self.focus_router.get_current_focus(FocusTarget.EXTERNAL, FocusType.SENSORY)
        
        self.assertIsNotNone(visual_focus)
        self.assertIsNotNone(language_focus)
        self.assertIsNotNone(internal_focus)
        self.assertIsNotNone(external_focus)
        
        # Prüfe, ob die Fokusintensitäten der Priorität entsprechen
        self.assertGreater(language_focus.intensity, visual_focus.intensity)  # HIGH > MEDIUM
        self.assertGreater(visual_focus.intensity, internal_focus.intensity)  # MEDIUM > LOW
        
        # Prüfe Top-Fokusziele
        top_targets = self.focus_router.get_top_focus_targets(limit=2)
        self.assertEqual(len(top_targets), 2)
        self.assertEqual(top_targets[0][0], FocusTarget.LANGUAGE)  # Höchste Priorität
    
    def test_focus_decay(self):
        """Test der Fokusabnahme über Zeit"""
        # Erstelle Stimulus
        visual_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.HIGH,
            data={"object_detected": "Person", "confidence": 0.9}
        )
        
        # Übermittle Stimulus
        self.context_core.submit_context(visual_data)
        
        # Warte auf Verarbeitung
        time.sleep(0.2)
        
        # Prüfe initialen Fokus
        initial_focus = self.focus_router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        self.assertIsNotNone(initial_focus)
        initial_intensity = initial_focus.intensity
        
        # Warte auf Fokusabnahme
        time.sleep(1.0)
        
        # Prüfe Fokus nach Wartezeit
        current_focus = self.focus_router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        
        # Prüfe, ob die Intensität abgenommen hat
        self.assertIsNotNone(current_focus)
        self.assertLess(current_focus.intensity, initial_intensity)
    
    def test_context_persistence(self):
        """Test der Kontextpersistenz nach Fokusverlagerung"""
        # Erstelle visuellen Stimulus
        visual_data = ContextData(
            source=ContextSource.VISUAL,
            priority=ContextPriority.HIGH,
            data={"object_detected": "Person", "position": [0.5, 0.5]}
        )
        
        # Übermittle Stimulus
        self.context_core.submit_context(visual_data)
        
        # Warte auf Verarbeitung
        time.sleep(0.2)
        
        # Prüfe Kontext und Fokus
        visual_state_before = self.context_state.get_state(ContextStateType.VISUAL)
        visual_focus_before = self.focus_router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        
        self.assertIn("object_detected", visual_state_before)
        self.assertEqual(visual_state_before["object_detected"], "Person")
        self.assertIsNotNone(visual_focus_before)
        
        # Erstelle neuen, höher priorisierten Stimulus
        language_data = ContextData(
            source=ContextSource.LANGUAGE,
            priority=ContextPriority.CRITICAL,
            data={"text": "Notfall!", "urgency": "high"}
        )
        
        # Übermittle Stimulus
        self.context_core.submit_context(language_data)
        
        # Warte auf Verarbeitung
        time.sleep(0.2)
        
        # Prüfe, ob der Fokus verlagert wurde
        language_focus = self.focus_router.get_current_focus(FocusTarget.LANGUAGE, FocusType.SENSORY)
        visual_focus_after = self.focus_router.get_current_focus(FocusTarget.VISION, FocusType.SENSORY)
        
        self.assertIsNotNone(language_focus)
        self.assertIsNotNone(visual_focus_after)
        self.assertGreater(language_focus.intensity, visual_focus_after.intensity)
        
        # Prüfe, ob der visuelle Kontext trotz Fokusverlagerung erhalten bleibt
        visual_state_after = self.context_state.get_state(ContextStateType.VISUAL)
        self.assertIn("object_detected", visual_state_after)
        self.assertEqual(visual_state_after["object_detected"], "Person")
    
    def test_context_snapshot_creation(self):
        """Test der Erstellung von Kontextschnappschüssen"""
        # Erstelle verschiedene Stimuli
        stimuli = [
            ContextData(
                source=ContextSource.VISUAL,
                priority=ContextPriority.MEDIUM,
                data={"object_detected": "Person", "position": [0.5, 0.5]}
            ),
            ContextData(
                source=ContextSource.LANGUAGE,
                priority=ContextPriority.HIGH,
                data={"text": "Hallo", "sentiment": "positive"}
            ),
            ContextData(
                source=ContextSource.INTERNAL,
                priority=ContextPriority.LOW,
                data={"status": "normal", "load": 0.3}
            )
        ]
        
        # Übermittle Stimuli
        for stimulus in stimuli:
            self.context_core.submit_context(stimulus)
        
        # Warte auf Verarbeitung
        time.sleep(0.5)
        
        # Erstelle Snapshot
        snapshot = self.context_state.create_snapshot()
        
        # Prüfe Snapshot
        self.assertIsNotNone(snapshot)
        self.assertIn(ContextStateType.VISUAL, snapshot.state)
        self.assertIn(ContextStateType.LINGUISTIC, snapshot.state)
        self.assertIn(ContextStateType.SYSTEM, snapshot.state)
        
        # Prüfe Snapshot-Inhalte
        self.assertIn("object_detected", snapshot.state[ContextStateType.VISUAL])
        self.assertEqual(
            snapshot.state[ContextStateType.VISUAL]["object_detected"].value,
            "Person"
        )
        
        self.assertIn("text", snapshot.state[ContextStateType.LINGUISTIC])
        self.assertEqual(
            snapshot.state[ContextStateType.LINGUISTIC]["text"].value,
            "Hallo"
        )
    
    def test_dynamic_focus_shift(self):
        """Test der dynamischen Fokusverlagerung zwischen Agenten"""
        # Registriere Agenten
        self.focus_router.register_agent("agent1")
        self.focus_router.register_agent("agent2")
        self.focus_router.regist
(Content truncated due to size limit. Use line ranges to read in chunks)