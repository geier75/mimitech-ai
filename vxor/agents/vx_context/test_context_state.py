#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: Unit Tests für context_state.py
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
from context_state import (
    ContextStateType, ContextStateEntry, ContextSnapshot, ContextState
)


class TestContextStateEntry(unittest.TestCase):
    """Tests für die ContextStateEntry-Klasse"""
    
    def test_init(self):
        """Test der Initialisierung"""
        entry = ContextStateEntry(
            value="test_value",
            source=ContextSource.VISUAL,
            confidence=0.9,
            expiration=time.time() + 60,
            metadata={"meta": "data"}
        )
        
        self.assertEqual(entry.value, "test_value")
        self.assertEqual(entry.source, ContextSource.VISUAL)
        self.assertAlmostEqual(entry.confidence, 0.9)
        self.assertIsNotNone(entry.expiration)
        self.assertEqual(entry.metadata, {"meta": "data"})
        self.assertIsNotNone(entry.timestamp)
    
    def test_default_values(self):
        """Test der Standardwerte"""
        entry = ContextStateEntry(
            value="test_value",
            source=ContextSource.VISUAL
        )
        
        self.assertEqual(entry.value, "test_value")
        self.assertEqual(entry.source, ContextSource.VISUAL)
        self.assertEqual(entry.confidence, 1.0)  # Standardwert
        self.assertIsNone(entry.expiration)  # Standardwert
        self.assertEqual(entry.metadata, {})  # Standardwert
        self.assertIsNotNone(entry.timestamp)


class TestContextSnapshot(unittest.TestCase):
    """Tests für die ContextSnapshot-Klasse"""
    
    def test_init(self):
        """Test der Initialisierung"""
        # Erstelle Testdaten
        state = {
            ContextStateType.VISUAL: {
                "object": ContextStateEntry(
                    value="Person",
                    source=ContextSource.VISUAL
                )
            },
            ContextStateType.SYSTEM: {
                "status": ContextStateEntry(
                    value="running",
                    source=ContextSource.INTERNAL
                )
            }
        }
        
        focus_state = {
            ("target", "type"): MagicMock()
        }
        
        global_metadata = {
            "system_start_time": time.time(),
            "update_count": 10
        }
        
        # Erstelle Snapshot
        snapshot = ContextSnapshot(
            timestamp=time.time(),
            state=state,
            focus_state=focus_state,
            global_metadata=global_metadata
        )
        
        self.assertIsNotNone(snapshot.timestamp)
        self.assertEqual(snapshot.state, state)
        self.assertEqual(snapshot.focus_state, focus_state)
        self.assertEqual(snapshot.global_metadata, global_metadata)
    
    def test_default_values(self):
        """Test der Standardwerte"""
        # Erstelle minimalen Snapshot
        snapshot = ContextSnapshot(
            timestamp=time.time(),
            state={},
            focus_state={}
        )
        
        self.assertIsNotNone(snapshot.timestamp)
        self.assertEqual(snapshot.state, {})
        self.assertEqual(snapshot.focus_state, {})
        self.assertEqual(snapshot.global_metadata, {})  # Standardwert


class TestContextState(unittest.TestCase):
    """Tests für die ContextState-Klasse"""
    
    def setUp(self):
        """Setup für die Tests"""
        # Erstelle Mock für ContextCore
        self.mock_context_core = MagicMock()
        self.mock_context_processor = MagicMock()
        self.mock_context_core.context_processor = self.mock_context_processor
        
        # Erstelle ContextState mit Mock
        self.state_manager = ContextState(self.mock_context_core)
    
    def tearDown(self):
        """Cleanup nach den Tests"""
        if self.state_manager.active:
            self.state_manager.stop()
    
    def test_init(self):
        """Test der Initialisierung"""
        self.assertEqual(self.state_manager.context_core, self.mock_context_core)
        
        # Prüfe, ob alle Zustandstypen initialisiert wurden
        for state_type in ContextStateType:
            self.assertIn(state_type, self.state_manager.current_state)
            self.assertEqual(self.state_manager.current_state[state_type], {})
        
        self.assertIsNotNone(self.state_manager.global_metadata)
        self.assertIn("system_start_time", self.state_manager.global_metadata)
        self.assertIn("last_update_time", self.state_manager.global_metadata)
        self.assertEqual(self.state_manager.global_metadata["update_count"], 0)
        self.assertEqual(self.state_manager.global_metadata["active_sources"], set())
        
        self.assertEqual(self.state_manager.state_history, [])
        self.assertEqual(self.state_manager.max_history_length, 100)
        
        # Prüfe Mapping von Quellen zu Zustandstypen
        for source in ContextSource:
            self.assertIn(source, self.state_manager.source_to_state_type)
        
        self.assertEqual(self.state_manager.update_handlers, {})
        self.assertEqual(self.state_manager.state_change_handlers, [])
        self.assertIsNotNone(self.state_manager.state_lock)
        self.assertGreater(self.state_manager.snapshot_interval, 0)
        self.assertGreater(self.state_manager.cleanup_interval, 0)
        self.assertIsNotNone(self.state_manager.last_snapshot_time)
        self.assertIsNotNone(self.state_manager.last_cleanup_time)
        self.assertIsNone(self.state_manager.snapshot_thread)
        self.assertFalse(self.state_manager.active)
    
    def test_start_stop(self):
        """Test des Starts und Stopps"""
        # Start
        self.state_manager.start()
        self.assertTrue(self.state_manager.active)
        self.assertIsNotNone(self.state_manager.snapshot_thread)
        self.assertTrue(self.state_manager.snapshot_thread.is_alive())
        
        # Stop
        self.state_manager.stop()
        self.assertFalse(self.state_manager.active)
        time.sleep(0.1)  # Warte kurz, damit der Thread beendet werden kann
        self.assertFalse(self.state_manager.snapshot_thread.is_alive())
    
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
        self.state_manager.process_context_data(context_data)
        
        # Prüfe, ob der Zustand aktualisiert wurde
        visual_state = self.state_manager.get_state(ContextStateType.VISUAL)
        self.assertIn("object_detected", visual_state)
        self.assertEqual(visual_state["object_detected"], "Person")
        self.assertIn("confidence", visual_state)
        self.assertEqual(visual_state["confidence"], 0.95)
        
        # Prüfe, ob die globalen Metadaten aktualisiert wurden
        self.assertEqual(self.state_manager.global_metadata["update_count"], 1)
        self.assertIn("VISUAL", self.state_manager.global_metadata["active_sources"])
    
    def test_update_state(self):
        """Test der Zustandsaktualisierung"""
        # Aktualisiere Zustand
        self.state_manager.update_state(
            ContextStateType.VISUAL,
            {"object": "Person", "position": [0.5, 0.7]},
            ContextSource.VISUAL,
            {"confidence": 0.9, "ttl": 60}
        )
        
        # Prüfe, ob der Zustand aktualisiert wurde
        visual_state = self.state_manager.get_state(ContextStateType.VISUAL)
        self.assertIn("object", visual_state)
        self.assertEqual(visual_state["object"], "Person")
        self.assertIn("position", visual_state)
        self.assertEqual(visual_state["position"], [0.5, 0.7])
        
        # Prüfe, ob die Einträge korrekt erstellt wurden
        entry = self.state_manager.get_state_entry(ContextStateType.VISUAL, "object")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.value, "Person")
        self.assertEqual(entry.source, ContextSource.VISUAL)
        self.assertEqual(entry.confidence, 0.9)
        self.assertIsNotNone(entry.expiration)
        self.assertIn("confidence", entry.metadata)
        self.assertIn("ttl", entry.metadata)
        
        # Prüfe, ob die globalen Metadaten aktualisiert wurden
        self.assertEqual(self.state_manager.global_metadata["update_count"], 1)
        self.assertIn("VISUAL", self.state_manager.global_metadata["active_sources"])
    
    def test_get_state(self):
        """Test des Zustandsabrufs"""
        # Aktualisiere Zustand mit Testdaten
        self.state_manager.update_state(
            ContextStateType.VISUAL,
            {"object": "Person", "position": [0.5, 0.7]},
            ContextSource.VISUAL
        )
        
        self.state_manager.update_state(
            ContextStateType.SYSTEM,
            {"status": "running", "load": 0.3},
            ContextSource.INTERNAL
        )
        
        # Teste Abruf des gesamten Zustands
        full_state = self.state_manager.get_state()
        self.assertIsInstance(full_state, dict)
        self.assertIn("VISUAL", full_state)
        self.assertIn("SYSTEM", full_state)
        self.assertIn("object", full_state["VISUAL"])
        self.assertIn("status", full_state["SYSTEM"])
        
        # Teste Abruf eines bestimmten Zustandstyps
        visual_state = self.state_manager.get_state(ContextStateType.VISUAL)
        self.assertIsInstance(visual_state, dict)
        self.assertIn("object", visual_state)
        self.assertEqual(visual_state["object"], "Person")
        self.assertIn("position", visual_state)
        
        # Teste Abruf eines bestimmten Schlüssels
        object_value = self.state_manager.get_state(ContextStateType.VISUAL, "object")
        self.assertEqual(object_value, "Person")
        
        # Teste Abruf eines nicht existierenden Schlüssels
        nonexistent_value = self.state_manager.get_state(ContextStateType.VISUAL, "nonexistent")
        self.assertIsNone(nonexistent_value)
        
        # Teste Suche nach Schlüssel in allen Zustandstypen
        status_value = self.state_manager.get_state(key="status")
        self.assertEqual(status_value, "running")
    
    def test_get_state_entry(self):
        """Test des Abrufs von Zustandseinträgen"""
        # Aktualisiere Zustand mit Testdaten
        self.state_manager.update_state(
            ContextStateType.VISUAL,
            {"object": "Person"},
            ContextSource.VISUAL,
            {"confidence": 0.9}
        )
        
        # Rufe Eintrag ab
        entry = self.state_manager.get_state_entry(ContextStateType.VISUAL, "object")
        
        # Prüfe Eintrag
        self.assertIsNotNone(entry)
        self.assertEqual(entry.value, "Person")
        self.assertEqual(entry.source, ContextSource.VISUAL)
        self.assertEqual(entry.confidence, 0.9)
    
    def test_get_state_by_source(self):
        """Test des Abrufs von Zuständen nach Quelle"""
        # Aktualisiere Zustand mit Testdaten von verschiedenen Quellen
        self.state_manager.update_state(
            ContextStateType.VISUAL,
            {"object": "Person", "position": [0.5, 0.7]},
            ContextSource.VISUAL
        )
        
        self.state_manager.update_state(
            ContextStateType.LINGUISTIC,
            {"text": "Hello"},
            ContextSource.LANGUAGE
        )
        
        self.state_manager.update_state(
            ContextStateType.COGNITIVE,
            {"thought": "Interesting"},
            ContextSource.INTERNAL
        )
        
        # Rufe Zustände nach Quelle ab
        visual_source_state = self.state_manager.get_state_by_source(ContextSource.VISUAL)
        
        # Prüfe Ergebnis
        self.assertIsInstance(visual_source_state, dict)
        self.assertIn("object", visual_source_state)
        self.assertEqual(visual_source_state["object"], "Person")
        self.assertIn("position", visual_source_state)
    
    def test_create_snapshot(self):
        """Test der Snapshot-Erstellung"""
        # Aktualisiere Zustand mit Testdaten
        self.state_manager.update_state(
            ContextStateType.VISUAL,
            {"object": "Person"},
            ContextSource.VISUAL
        )
        
        # Erstelle Snapshot
        snapshot = self.state_manager.create_snapshot()
        
        # Prüfe Snapshot
        self.assertIsInstance(snapshot, ContextSnapshot)
        self.assertIsNotNone(snapshot.timestamp)
        self.assertIn(ContextStateType.VISUAL, snapshot.state)
        self.assertIn("object", snapshot.state[ContextStateType.VISUAL])
        
        # Prüfe, ob der Snapshot zur Historie hinzugefügt wurde
        self.assertEqual(len(self.state_manager.state_history), 1)
        self.assertEqual(self.state_manager.state_history[0], snapshot)
    
    def test_get_latest_snapshot(self):
        """Test des Abrufs des neuesten Snapshots"""
        # Erstelle einige Snapshots
        self.state_manager.update_state(
            ContextStateType.VISUAL,
            {"object": "Person"},
            ContextSource.VISUAL
        )
        self.state_manager.create_snapshot()
        
        time.sleep(0.01)  # Kurze Pause für unterschiedliche Zeitstempel
        
        self.state_manager.update_state(
            ContextStateType.VISUAL,
            {"object": "Car"},
            ContextSource.VISUAL
        )
        snapshot2 = self.state_manager.create_snapshot()
        
        # Rufe neuesten Snapshot ab
        latest = self.state_manager.get_latest_snapshot()
        
        # Prüfe Ergebnis
        self.assertEqual(latest, snapshot2)
        self.assertEqual(latest.state[ContextStateType.VISUAL]["object"].value, "Car")
    
    def test_get_snapshot_at_time(self):
        """Test des Abrufs eines Snapshots zu einem bestimmten Zeitpunkt"""
        # Erstelle einige Snapshots
        self.state_manager.update_state(
            ContextStateType.VISUAL,
            {"object": "Person"},
            ContextSource.VISUAL
        )
        snapshot1 = self.state_manager.create_snapshot()
        
        time.sleep(0.1)  # Pause für unterschiedliche Zeitstempel
        
        self.state_manager.update_state(
            ContextStateType.VISUAL,
            {"object": "Car"},
            ContextSource.VISUAL
        )
        snapshot2 = self.state_manager.create_snapshot()
        
        # Rufe Snapshot zu einem Zeitpunkt zwischen den beiden Snapshots ab
        mid_time = (snapshot1.timestamp + snapshot2.timestamp) / 2
        snapshot = self.state_manager.get_snapshot_at_time(mid_time)
        
        # Prüfe Ergebnis (sollte der nähere Snapshot sein)
        self.assertIsNotNone(snapshot)
        if abs(mid_time - snapshot1.timestamp) < abs(mid_time - snapshot2.timestamp):
            self.assertEqual(snapshot, snapshot1)
        else:
            self.assertEqual(snapshot, snapshot2)
    
    def test_get_snapshots_in_timerange(self):
        """Test des Abrufs von Snapshots in einem Zeitbereich"""
        # Erstelle einige Snapshots
        timestamps = []
        
        self.state_manager.update_state(
            ContextStateType.VISUAL,
            {"object": "Person"},
            ContextSource.VISUAL
        )
        snapshot1 = self.state_manager.create_snapshot()
        timestamps.append(snapshot1.timestamp)
        
        time.sleep(0.1)  # Pause für unterschiedliche Zeitstempel
        
        self.state_man
(Content truncated due to size limit. Use line ranges to read in chunks)