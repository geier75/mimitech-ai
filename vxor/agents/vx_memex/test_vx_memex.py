#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MEMEX: Unit Tests
--------------------
Testsuite für das VX-MEMEX Gedächtnismodul.
Enthält Tests für alle Komponenten: Memory Core, Semantic Store, Episodic Store,
Working Memory und VXOR Bridge.

Optimiert für Apple Silicon M4 Max.
"""

import unittest
import time
import json
import os
import sys
from typing import Dict, List, Any, Optional, Union, Tuple

# Pfad zum Projektverzeichnis hinzufügen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Module importieren
from memory_core import MemoryCore
from semantic_store import SemanticStore
from episodic_store import EpisodicStore
from working_memory import WorkingMemory
from vxor_bridge import VXORBridge

class TestSemanticStore(unittest.TestCase):
    """Testet die Funktionalität des semantischen Speichers."""
    
    def setUp(self):
        """Initialisiert den semantischen Speicher für jeden Test."""
        self.store = SemanticStore({'vector_dim': 384})
    
    def test_store_and_retrieve(self):
        """Testet das Speichern und Abrufen von Einträgen."""
        # Eintrag speichern
        content = "Dies ist ein Testinhalt für den semantischen Speicher."
        tags = ["test", "semantisch", "speicher"]
        entry_id = self.store.store(content=content, tags=tags, importance=0.8)
        
        # Prüfen, ob die ID zurückgegeben wurde
        self.assertIsNotNone(entry_id)
        self.assertTrue(isinstance(entry_id, str))
        
        # Eintrag abrufen
        entry = self.store.retrieve_by_id(entry_id)
        
        # Prüfen, ob der Eintrag korrekt abgerufen wurde
        self.assertEqual(entry['content'], content)
        self.assertEqual(entry['tags'], tags)
        self.assertEqual(entry['importance'], 0.8)
        self.assertEqual(entry['id'], entry_id)
    
    def test_retrieve_by_tags(self):
        """Testet das Abrufen von Einträgen anhand von Schlagwörtern."""
        # Einträge speichern
        entry1_id = self.store.store(
            content="Eintrag mit Tag A und B",
            tags=["A", "B"],
            importance=0.7
        )
        
        entry2_id = self.store.store(
            content="Eintrag mit Tag B und C",
            tags=["B", "C"],
            importance=0.8
        )
        
        entry3_id = self.store.store(
            content="Eintrag mit Tag A und C",
            tags=["A", "C"],
            importance=0.9
        )
        
        # Einträge mit Tag A abrufen
        entries_a = self.store.retrieve_by_tags(tags=["A"])
        self.assertEqual(len(entries_a), 2)
        
        # Einträge mit Tag A UND B abrufen
        entries_ab = self.store.retrieve_by_tags(tags=["A", "B"], match_all=True)
        self.assertEqual(len(entries_ab), 1)
        self.assertEqual(entries_ab[0]['id'], entry1_id)
        
        # Einträge mit Tag B ODER C abrufen
        entries_bc = self.store.retrieve_by_tags(tags=["B", "C"])
        self.assertEqual(len(entries_bc), 3)
    
    def test_semantic_search(self):
        """Testet die semantische Suche."""
        # Einträge speichern
        self.store.store(
            content="Python ist eine interpretierte Programmiersprache.",
            tags=["python", "programmierung"]
        )
        
        self.store.store(
            content="TensorFlow ist ein Framework für maschinelles Lernen.",
            tags=["tensorflow", "machine learning"]
        )
        
        self.store.store(
            content="PyTorch ist ein Framework für Deep Learning.",
            tags=["pytorch", "deep learning"]
        )
        
        # Semantische Suche durchführen
        results = self.store.retrieve("Framework für KI")
        
        # Prüfen, ob Ergebnisse zurückgegeben wurden
        self.assertGreater(len(results), 0)
        
        # Prüfen, ob die Ähnlichkeit angegeben ist
        self.assertIn('similarity', results[0])
    
    def test_update(self):
        """Testet das Aktualisieren von Einträgen."""
        # Eintrag speichern
        entry_id = self.store.store(
            content="Ursprünglicher Inhalt",
            tags=["original"],
            importance=0.5
        )
        
        # Eintrag aktualisieren
        success = self.store.update(
            entry_id=entry_id,
            content="Aktualisierter Inhalt",
            tags=["aktualisiert"],
            importance=0.8
        )
        
        # Prüfen, ob die Aktualisierung erfolgreich war
        self.assertTrue(success)
        
        # Aktualisierten Eintrag abrufen
        entry = self.store.retrieve_by_id(entry_id)
        
        # Prüfen, ob der Eintrag korrekt aktualisiert wurde
        self.assertEqual(entry['content'], "Aktualisierter Inhalt")
        self.assertEqual(entry['tags'], ["aktualisiert"])
        self.assertEqual(entry['importance'], 0.8)
    
    def test_delete(self):
        """Testet das Löschen von Einträgen."""
        # Eintrag speichern
        entry_id = self.store.store(
            content="Zu löschender Inhalt",
            tags=["löschen"]
        )
        
        # Prüfen, ob der Eintrag existiert
        self.assertTrue(self.store.retrieve_by_id(entry_id) is not None)
        
        # Eintrag löschen
        success = self.store.delete(entry_id)
        
        # Prüfen, ob das Löschen erfolgreich war
        self.assertTrue(success)
        
        # Prüfen, ob der Eintrag nicht mehr existiert
        with self.assertRaises(KeyError):
            self.store.retrieve_by_id(entry_id)
    
    def test_link_entries(self):
        """Testet das Verknüpfen von Einträgen."""
        # Einträge speichern
        entry1_id = self.store.store(content="Eintrag 1")
        entry2_id = self.store.store(content="Eintrag 2")
        
        # Einträge verknüpfen
        success = self.store.link_entries(entry1_id, entry2_id, "related")
        
        # Prüfen, ob die Verknüpfung erfolgreich war
        self.assertTrue(success)
        
        # Verknüpfte Einträge abrufen
        linked_entries = self.store.get_linked_entries(entry1_id)
        
        # Prüfen, ob der verknüpfte Eintrag abgerufen wurde
        self.assertEqual(len(linked_entries), 1)
        self.assertEqual(linked_entries[0]['id'], entry2_id)
    
    def test_cleanup(self):
        """Testet die Bereinigung abgelaufener Einträge."""
        # Eintrag mit kurzer TTL speichern
        entry_id = self.store.store(
            content="Kurzlebiger Eintrag",
            ttl=1  # 1 Sekunde
        )
        
        # Prüfen, ob der Eintrag existiert
        self.assertTrue(self.store.retrieve_by_id(entry_id) is not None)
        
        # Warten, bis der Eintrag abgelaufen ist
        time.sleep(1.5)
        
        # Bereinigung durchführen
        cleaned_count = self.store.cleanup()
        
        # Prüfen, ob ein Eintrag bereinigt wurde
        self.assertEqual(cleaned_count, 1)
        
        # Prüfen, ob der Eintrag nicht mehr existiert
        with self.assertRaises(KeyError):
            self.store.retrieve_by_id(entry_id)
    
    def test_export_import(self):
        """Testet den Export und Import von Daten."""
        # Einträge speichern
        entry1_id = self.store.store(content="Eintrag 1", tags=["export"])
        entry2_id = self.store.store(content="Eintrag 2", tags=["export"])
        
        # Daten exportieren
        export_data = self.store.export_data()
        
        # Neuen Store erstellen
        new_store = SemanticStore({'vector_dim': 384})
        
        # Daten importieren
        imported_count = new_store.import_data(export_data)
        
        # Prüfen, ob die Einträge importiert wurden
        self.assertEqual(imported_count, 2)
        
        # Prüfen, ob die Einträge korrekt importiert wurden
        entry1 = new_store.retrieve_by_id(entry1_id)
        self.assertEqual(entry1['content'], "Eintrag 1")
        self.assertEqual(entry1['tags'], ["export"])
        
        entry2 = new_store.retrieve_by_id(entry2_id)
        self.assertEqual(entry2['content'], "Eintrag 2")
        self.assertEqual(entry2['tags'], ["export"])


class TestEpisodicStore(unittest.TestCase):
    """Testet die Funktionalität des episodischen Speichers."""
    
    def setUp(self):
        """Initialisiert den episodischen Speicher für jeden Test."""
        self.store = EpisodicStore({'vector_dim': 384, 'max_entries': 1000})
    
    def test_store_and_retrieve(self):
        """Testet das Speichern und Abrufen von Einträgen."""
        # Eintrag speichern
        content = "Dies ist ein Testinhalt für den episodischen Speicher."
        tags = ["test", "episodisch", "speicher"]
        timestamp = time.time()
        entry_id = self.store.store(
            content=content,
            tags=tags,
            timestamp=timestamp,
            importance=0.8
        )
        
        # Prüfen, ob die ID zurückgegeben wurde
        self.assertIsNotNone(entry_id)
        self.assertTrue(isinstance(entry_id, str))
        
        # Eintrag abrufen
        entry = self.store.retrieve_by_id(entry_id)
        
        # Prüfen, ob der Eintrag korrekt abgerufen wurde
        self.assertEqual(entry['content'], content)
        self.assertEqual(entry['tags'], tags)
        self.assertEqual(entry['timestamp'], timestamp)
        self.assertEqual(entry['importance'], 0.8)
        self.assertEqual(entry['id'], entry_id)
    
    def test_retrieve_by_time_range(self):
        """Testet das Abrufen von Einträgen anhand eines Zeitbereichs."""
        # Zeitstempel für die Tests
        now = time.time()
        one_hour_ago = now - 3600
        two_hours_ago = now - 7200
        three_hours_ago = now - 10800
        
        # Einträge mit verschiedenen Zeitstempeln speichern
        entry1_id = self.store.store(
            content="Eintrag von vor 3 Stunden",
            timestamp=three_hours_ago
        )
        
        entry2_id = self.store.store(
            content="Eintrag von vor 2 Stunden",
            timestamp=two_hours_ago
        )
        
        entry3_id = self.store.store(
            content="Eintrag von vor 1 Stunde",
            timestamp=one_hour_ago
        )
        
        # Einträge im Zeitbereich der letzten 2 Stunden abrufen
        entries = self.store.retrieve_by_time_range(
            start_time=two_hours_ago,
            end_time=now
        )
        
        # Prüfen, ob die richtigen Einträge abgerufen wurden
        self.assertEqual(len(entries), 2)
        
        # Prüfen, ob die Einträge nach Zeitstempel sortiert sind (neueste zuerst)
        self.assertEqual(entries[0]['timestamp'], one_hour_ago)
        self.assertEqual(entries[1]['timestamp'], two_hours_ago)
    
    def test_retrieve_latest(self):
        """Testet das Abrufen der neuesten Einträge."""
        # Zeitstempel für die Tests
        now = time.time()
        one_min_ago = now - 60
        two_mins_ago = now - 120
        
        # Einträge mit verschiedenen Zeitstempeln speichern
        entry1_id = self.store.store(
            content="Ältester Eintrag",
            timestamp=two_mins_ago
        )
        
        entry2_id = self.store.store(
            content="Mittlerer Eintrag",
            timestamp=one_min_ago
        )
        
        entry3_id = self.store.store(
            content="Neuester Eintrag",
            timestamp=now
        )
        
        # Die neuesten 2 Einträge abrufen
        entries = self.store.retrieve_latest(limit=2)
        
        # Prüfen, ob die richtigen Einträge abgerufen wurden
        self.assertEqual(len(entries), 2)
        
        # Prüfen, ob die Einträge nach Zeitstempel sortiert sind (neueste zuerst)
        self.assertEqual(entries[0]['timestamp'], now)
        self.assertEqual(entries[1]['timestamp'], one_min_ago)
    
    def test_get_time_range(self):
        """Testet das Abrufen des Zeitbereichs aller Einträge."""
        # Zeitstempel für die Tests
        now = time.time()
        one_hour_ago = now - 3600
        
        # Einträge mit verschiedenen Zeitstempeln speichern
        entry1_id = self.store.store(
            content="Älterer Eintrag",
            timestamp=one_hour_ago
        )
        
        entry2_id = self.store.store(
            content="Neuerer Eintrag",
            timestamp=now
        )
        
        # Zeitbereich abrufen
        oldest, newest = self.store.get_time_range()
        
        # Prüfen, ob der Zeitbereich korrekt ist
        self.assertEqual(oldest, one_hour_ago)
        self.assertEqual(newest, now)
    
    def test_storage_management(self):
        """Testet die Speicherverwaltung bei Erreichen des Limits."""
        # Konfiguration mit sehr niedrigem Limit
        store = EpisodicStore({'max_entries': 3})
        
        # Mehr Einträge speichern als das Limit erlaubt
        entry1_id = store.store(content="Eintrag 1", importance=0.1)
        entry2_id = store.store(content="Eintrag 2", importance=0.5)
        entry3_id = store.store(content="Eintrag 3", importance=0.9)
        entry4_id = store.store(content="Eintrag 4", importance=0.8)
        
        # Prüfen, ob die Anzahl der Einträge dem Limit entspricht
        self.assertEqual(store.count(), 3)
        
        # Prüfen, ob der unwichtigste Eintrag entfernt wurde
        with self.assertRaises(KeyError):
            store.retrieve_by_id(entry1_id)


class TestWorkingMemory(unittest.TestCase):
    """Testet die Funktionalität des Arbeitsgedächtnisses."""
    
    def setUp(self):
        """Initialisiert das Arbeitsgedächtnis für jeden Test."""
        self.memory = WorkingMemory({
            'vector_dim': 384,
            'ttl_default': 3600,  # 1 Stunde
            'max_entries': 100
        })
    
    def test_store_and_retrieve(self):
        """Testet das Speichern und Abrufen von Einträgen."""
        # Eintrag speichern
        content = "Dies ist ein Testinhalt für das Arbeitsgedächtnis."
        tags = ["test", "arbeitsgedächtnis"]
        entry_id = self.memory.store(content=content, tags=tags, importance=0.8)
        
        # Prüfen, ob die ID zurückgegeben wurde
        self.assertIsNotNone(entry_id)
        self.assertTrue(isinstance(entry_id, str))
        
        # Eintrag abrufen
        entry = self.memory.retrieve_by_id(entry_id)
        
        # Prüfen, ob der Eintrag korrekt abgerufen wurde
        self.assertEqual(entry['content'], content)
        self.assertEqual(entry['tags'], tags)
        self.assertEqual(entry['importance'], 0.8)
        self.assertEqual(entry['id'], entry_id)
    
    def test_activation_tracking(self):
        """Testet die Aktivierungsverfolgung bei mehrfachem Zugriff."""
        # Einträge speichern
        entry1_id = self.memory.store(content="Eintrag 1", importance=0.5)
        entry2_id = self.memory.store(content="Eintrag 2", importance=0.5)
        
        # Mehrfacher Zugriff auf einen Eintrag
        for _ in range(5):
            self.memory.retrieve_by_id(entry1_id)
        
        # Wichtigste Einträge abrufen
        important_entries = self.memory.get_most_important(limit=2)
        
        # Prüfen, ob der häufiger abgerufene Eintrag zuerst kommt
        self.assertEqual(important_entries[0]['id'], entry1_id)
        self.assertEqual(important_entries[1]['id'], entry2_id)
    
    def test_refresh(self):
        """Testet das Auffrischen von Einträgen."""
        # Eintrag mit kurzer TTL speichern
        entry_id = self.memory.store(
            content="Kurzlebiger Eintrag",
            ttl=2  # 2 Sekunden
        )
        
        # Warten, bis der Eintrag fast abgelaufen ist
        time.sleep(1.5)
        
        # Eintrag auffrischen
        success = self.memory.refresh(entry_id, ttl=5)  # 5 Sekunden
        
        # Prüfen, ob das Auffrischen erfolgreich war
        self.assertTrue(success)
        
        # Warten, bis die ursprüngliche TTL abgelaufen wäre
        time.sleep(1)
        
        # Prüfen, ob der Eintrag noch existiert
        entry = self.memory.retrieve_by_id(entry_id)
        self.assertIsNotNone(entry)
        
        # Warten, bis die neue TTL abgelaufen ist
        time.sleep(3.5)
        
        # Bereinigung durchführen
        cleaned_count = self.memory.cleanup()
        
        # Prüfen, ob ein Eintrag bereinigt wurde
        self.assertEqual(cleaned_count, 1)
        
        # Prüfen, ob der Eintrag nicht mehr existiert
        with self.assertRaises(KeyError):
            self.memory.retrieve_by_id(entry_id)
    
    def test_get_most_important(self):
        """Testet das Abrufen der wichtigsten Einträge."""
        # Einträge mit verschiedenen Wichtigkeiten speichern
        entry1_id = self.memory.store(content="Unwichtiger Eintrag", importance=0.2)
        entry2_id = self.memory.store(content="Wichtiger Eintrag", importance=0.8)
        entry3_id = self.memory.store(content="Mittelwichtiger Eintrag", importance=0.5)
        
        # Wichtigste Einträge abrufen
        important_entries = self.memory.get_most_important(limit=2)
        
        # Prüfen, ob die richtigen Einträge abgerufen wurden
        self.assertEqual(len(important_entries), 2)
        
        # Prüfen, ob die Einträge nach Wichtigkeit sortiert sind
        self.assertEqual(important_entries[0]['id'], entry2_id)
        self.assertEqual(important_entries[1]['id'], entry3_id)
    
    def test_get_least_important(self):
        """Testet das Abrufen der am wenigsten wichtigen Einträge."""
        # Einträge mit verschiedenen Wichtigkeiten speichern
        entry1_id = self.memory.store(content="Unwichtiger Eintrag", importance=0.2)
        entry2_id = self.memory.store(content="Wichtiger Eintrag", importance=0.8)
        entry3_id = self.memory.store(content="Mittelwichtiger Eintrag", importance=0.5)
        
        # Am wenigsten wichtige Einträge abrufen
        least_important_entries = self.memory.get_least_important(limit=2)
        
        # Prüfen, ob die richtigen Einträge abgerufen wurden
        self.assertEqual(len(least_important_entries), 2)
        
        # Prüfen, ob die Einträge nach Wichtigkeit sortiert sind (aufsteigend)
        self.assertEqual(least_important_entries[0]['id'], entry1_id)
        self.assertEqual(least_important_entries[1]['id'], entry3_id)
    
    def test_storage_management(self):
        """Testet die Speicherverwaltung bei Erreichen des Limits."""
        # Konfiguration mit sehr niedrigem Limit
        memory = WorkingMemory({'max_entries': 3})
        
        # Mehr Einträge speichern als das Limit erlaubt
        entry1_id = memory.store(content="Eintrag 1", importance=0.1)
        entry2_id = memory.store(content="Eintrag 2", importance=0.5)
        entry3_id = memory.store(content="Eintrag 3", importance=0.9)
        entry4_id = memory.store(content="Eintrag 4", importance=0.8)
        
        # Prüfen, ob die Anzahl der Einträge dem Limit entspricht
        self.assertEqual(memory.count(), 3)
        
        # Prüfen, ob der unwichtigste Eintrag entfernt wurde
        with self.assertRaises(KeyError):
            memory.retrieve_by_id(entry1_id)


class TestMemoryCore(unittest.TestCase):
    """Testet die Funktionalität des Memory Core."""
    
    def setUp(self):
        """Initialisiert den Memory Core für jeden Test."""
        self.memory = MemoryCore({
            'semantic': {'vector_dim': 384},
            'episodic': {'max_entries': 1000},
            'working': {'ttl_default': 3600}
        })
    
    def test_store_all_memory_types(self):
        """Testet das Speichern in allen Gedächtnissystemen."""
        # Eintrag in allen Gedächtnissystemen speichern
        content = "Dies ist ein Testinhalt für alle Gedächtnissysteme."
        tags = ["test", "alle", "gedächtnissysteme"]
        result = self.memory.store(
            content=content,
            memory_type='all',
            tags=tags,
            importance=0.8
        )
        
        # Prüfen, ob IDs für alle Gedächtnissysteme zurückgegeben wurden
        self.assertIn('semantic', result)
        self.assertIn('episodic', result)
        self.assertIn('working', result)
        
        # Prüfen, ob die IDs gültig sind
        for memory_type, entry_id in result.items():
            self.assertIsNotNone(entry_id)
            self.assertTrue(isinstance(entry_id, str))
    
    def test_store_specific_memory_type(self):
        """Testet das Speichern in einem bestimmten Gedächtnissystem."""
        # Eintrag nur im semantischen Gedächtnis speichern
        content = "Dies ist ein Testinhalt nur für das semantische Gedächtnis."
        tags = ["test", "semantisch"]
        result = self.memory.store(
            content=content,
            memory_type='semantic',
            tags=tags,
            importance=0.8
        )
        
        # Prüfen, ob eine ID für das semantische Gedächtnis zurückgegeben wurde
        self.assertIn('semantic', result)
        
        # ID des Eintrags abrufen
        entry_id = result['semantic']
        
        # Eintrag aus allen Gedächtnissystemen abrufen
        retrieved = self.memory.retrieve_by_id(entry_id)
        
        # Prüfen, ob der Eintrag nur im semantischen Gedächtnis existiert
        self.assertIn('semantic', retrieved)
        self.assertNotIn('episodic', retrieved)
        self.assertNotIn('working', retrieved)
    
    def test_retrieve_by_query(self):
        """Testet das Abrufen von Einträgen anhand einer Suchanfrage."""
        # Einträge in allen Gedächtnissystemen speichern
        self.memory.store(
            content="Python ist eine interpretierte Programmiersprache.",
            memory_type='all',
            tags=["python", "programmierung"]
        )
        
        self.memory.store(
            content="TensorFlow ist ein Framework für maschinelles Lernen.",
            memory_type='all',
            tags=["tensorflow", "machine learning"]
        )
        
        # Semantische Suche durchführen
        results = self.memory.retrieve("Framework für KI")
        
        # Prüfen, ob Ergebnisse für alle Gedächtnissysteme zurückgegeben wurden
        self.assertIn('semantic', results)
        self.assertIn('episodic', results)
        self.assertIn('working', results)
        
        # Prüfen, ob Ergebnisse gefunden wurden
        self.assertGreater(len(results['semantic']), 0)
    
    def test_update(self):
        """Testet das Aktualisieren von Einträgen in allen Gedächtnissystemen."""
        # Eintrag in allen Gedächtnissystemen speichern
        result = self.memory.store(
            content="Ursprünglicher Inhalt",
            memory_type='all',
            tags=["original"],
            importance=0.5
        )
        
        # ID des Eintrags abrufen
        entry_id = result['semantic']  # Alle Systeme verwenden dieselbe ID
        
        # Eintrag in allen Gedächtnissystemen aktualisieren
        update_result = self.memory.update(
            entry_id=entry_id,
            content="Aktualisierter Inhalt",
            tags=["aktualisiert"],
            importance=0.8,
            memory_type='all'
        )
        
        # Prüfen, ob die Aktualisierung in allen Systemen erfolgreich war
        self.assertTrue(update_result['semantic'])
        self.assertTrue(update_result['episodic'])
        self.assertTrue(update_result['working'])
        
        # Aktualisierten Eintrag abrufen
        retrieved = self.memory.retrieve_by_id(entry_id)
        
        # Prüfen, ob der Eintrag in allen Systemen korrekt aktualisiert wurde
        for memory_type, entry in retrieved.items():
            self.assertEqual(entry['content'], "Aktualisierter Inhalt")
            self.assertEqual(entry['tags'], ["aktualisiert"])
            self.assertEqual(entry['importance'], 0.8)
    
    def test_delete(self):
        """Testet das Löschen von Einträgen aus allen Gedächtnissystemen."""
        # Eintrag in allen Gedächtnissystemen speichern
        result = self.memory.store(
            content="Zu löschender Inhalt",
            memory_type='all',
            tags=["löschen"]
        )
        
        # ID des Eintrags abrufen
        entry_id = result['semantic']  # Alle Systeme verwenden dieselbe ID
        
        # Eintrag aus allen Gedächtnissystemen löschen
        delete_result = self.memory.delete(entry_id, memory_type='all')
        
        # Prüfen, ob das Löschen in allen Systemen erfolgreich war
        self.assertTrue(delete_result['semantic'])
        self.assertTrue(delete_result['episodic'])
        self.assertTrue(delete_result['working'])
        
        # Prüfen, ob der Eintrag nicht mehr existiert
        retrieved = self.memory.retrieve_by_id(entry_id)
        self.assertEqual(retrieved, {})
    
    def test_link_entries(self):
        """Testet das Verknüpfen von Einträgen."""
        # Einträge speichern
        result1 = self.memory.store(content="Eintrag 1", memory_type='all')
        result2 = self.memory.store(content="Eintrag 2", memory_type='all')
        
        # IDs der Einträge abrufen
        entry1_id = result1['semantic']
        entry2_id = result2['semantic']
        
        # Einträge verknüpfen
        success = self.memory.link_entries(entry1_id, entry2_id, "related")
        
        # Prüfen, ob die Verknüpfung erfolgreich war
        self.assertTrue(success)
        
        # Verknüpfte Einträge abrufen
        linked_entries = self.memory.get_linked_entries(entry1_id)
        
        # Prüfen, ob der verknüpfte Eintrag in allen Systemen abgerufen wurde
        self.assertIn('semantic', linked_entries)
        self.assertIn('episodic', linked_entries)
        self.assertIn('working', linked_entries)
        
        # Prüfen, ob der verknüpfte Eintrag korrekt abgerufen wurde
        for memory_type, entries in linked_entries.items():
            if entries:  # Einige Systeme könnten keine Verknüpfungen haben
                self.assertEqual(entries[0]['id'], entry2_id)
    
    def test_cleanup(self):
        """Testet die Bereinigung abgelaufener Einträge in allen Gedächtnissystemen."""
        # Eintrag mit kurzer TTL in allen Gedächtnissystemen speichern
        result = self.memory.store(
            content="Kurzlebiger Eintrag",
            memory_type='all',
            ttl=1  # 1 Sekunde
        )
        
        # ID des Eintrags abrufen
        entry_id = result['semantic']
        
        # Warten, bis der Eintrag abgelaufen ist
        time.sleep(1.5)
        
        # Bereinigung in allen Gedächtnissystemen durchführen
        cleaned = self.memory.cleanup(memory_type='all')
        
        # Prüfen, ob in jedem System ein Eintrag bereinigt wurde
        self.assertEqual(cleaned['semantic'], 1)
        self.assertEqual(cleaned['episodic'], 1)
        self.assertEqual(cleaned['working'], 1)
        
        # Prüfen, ob der Eintrag nicht mehr existiert
        retrieved = self.memory.retrieve_by_id(entry_id)
        self.assertEqual(retrieved, {})
    
    def test_get_stats(self):
        """Testet das Abrufen von Statistiken für alle Gedächtnissysteme."""
        # Einige Einträge speichern
        for i in range(3):
            self.memory.store(
                content=f"Eintrag {i}",
                memory_type='all',
                importance=0.5 + i * 0.1
            )
        
        # Statistiken abrufen
        stats = self.memory.get_stats()
        
        # Prüfen, ob Statistiken für alle Gedächtnissysteme zurückgegeben wurden
        self.assertIn('semantic', stats)
        self.assertIn('episodic', stats)
        self.assertIn('working', stats)
        
        # Prüfen, ob die Statistiken korrekt sind
        for memory_type, memory_stats in stats.items():
            self.assertEqual(memory_stats['count'], 3)
            self.assertGreater(memory_stats['size'], 0)
            self.assertGreater(memory_stats['avg_importance'], 0)
            
            # Prüfen, ob der episodische Speicher einen Zeitbereich hat
            if memory_type == 'episodic':
                self.assertIn('time_range', memory_stats)
    
    def test_export_import(self):
        """Testet den Export und Import von Daten für alle Gedächtnissysteme."""
        # Einträge speichern
        for i in range(3):
            self.memory.store(
                content=f"Eintrag {i}",
                memory_type='all',
                tags=["export"],
                importance=0.5 + i * 0.1
            )
        
        # Daten exportieren
        export_json = self.memory.export_to_json()
        
        # Neuen Memory Core erstellen
        new_memory = MemoryCore()
        
        # Daten importieren
        import_result = new_memory.import_from_json(export_json)
        
        # Prüfen, ob die Einträge in allen Systemen importiert wurden
        self.assertEqual(import_result['semantic'], 3)
        self.assertEqual(import_result['episodic'], 3)
        self.assertEqual(import_result['working'], 3)
        
        # Statistiken abrufen
        stats = new_memory.get_stats()
        
        # Prüfen, ob die Statistiken korrekt sind
        for memory_type, memory_stats in stats.items():
            self.assertEqual(memory_stats['count'], 3)


class TestVXORBridge(unittest.TestCase):
    """Testet die Funktionalität der VXOR-Bridge."""
    
    def setUp(self):
        """Initialisiert die VXOR-Bridge für jeden Test."""
        self.bridge = VXORBridge()
    
    def test_manifest_generation(self):
        """Testet die Generierung des Manifests."""
        # Manifest abrufen
        manifest = self.bridge.manifest
        
        # Prüfen, ob das Manifest die erforderlichen Informationen enthält
        self.assertEqual(manifest['module_name'], "VX-MEMEX")
        self.assertIn('version', manifest)
        self.assertIn('description', manifest)
        self.assertIn('components', manifest)
        self.assertIn('dependencies', manifest)
        self.assertIn('integration_points', manifest)
        
        # Prüfen, ob alle Komponenten im Manifest enthalten sind
        component_names = [comp['name'] for comp in manifest['components']]
        self.assertIn('memory_core', component_names)
        self.assertIn('semantic_store', component_names)
        self.assertIn('episodic_store', component_names)
        self.assertIn('working_memory', component_names)
        self.assertIn('vxor_bridge', component_names)
    
    def test_store_and_retrieve(self):
        """Testet das Speichern und Abrufen von Einträgen über die Bridge."""
        # Eintrag speichern
        content = "Dies ist ein Testinhalt für die VXOR-Bridge."
        tags = ["test", "vxor-bridge"]
        result = self.bridge.store(
            content=content,
            memory_type='all',
            tags=tags,
            importance=0.8,
            source_module='test'
        )
        
        # Prüfen, ob IDs für alle Gedächtnissysteme zurückgegeben wurden
        self.assertIn('semantic', result)
        self.assertIn('episodic', result)
        self.assertIn('working', result)
        
        # ID des Eintrags abrufen
        entry_id = result['semantic']
        
        # Eintrag abrufen
        retrieved = self.bridge.retrieve_by_id(entry_id)
        
        # Prüfen, ob der Eintrag korrekt abgerufen wurde
        for memory_type, entry in retrieved.items():
            self.assertEqual(entry['content'], content)
            self.assertEqual(entry['tags'], tags)
            self.assertEqual(entry['importance'], 0.8)
    
    def test_event_handling(self):
        """Testet die Ereignisbehandlung."""
        # Ereigniszähler
        event_count = 0
        
        # Event-Handler-Funktion
        def test_handler(event_data):
            nonlocal event_count
            event_count += 1
        
        # Event-Handler registrieren
        self.bridge.register_event_handler('test_event', test_handler)
        
        # Ereignis auslösen
        handler_count = self.bridge.trigger_event('test_event', {'test': True})
        
        # Prüfen, ob der Handler aufgerufen wurde
        self.assertEqual(handler_count, 1)
        self.assertEqual(event_count, 1)
        
        # Event-Handler entfernen
        success = self.bridge.unregister_event_handler('test_event', test_handler)
        
        # Prüfen, ob der Handler entfernt wurde
        self.assertTrue(success)
        
        # Ereignis erneut auslösen
        handler_count = self.bridge.trigger_event('test_event', {'test': True})
        
        # Prüfen, ob der Handler nicht mehr aufgerufen wurde
        self.assertEqual(handler_count, 0)
        self.assertEqual(event_count, 1)
    
    def test_module_registration(self):
        """Testet die Registrierung von Modulen."""
        # Mock-Modul erstellen
        class MockModule:
            def test_method(self):
                return True
        
        mock_module = MockModule()
        
        # Modul registrieren
        success = self.bridge.register_module('mock', mock_module)
        
        # Prüfen, ob die Registrierung erfolgreich war
        self.assertTrue(success)
        
        # Registrierte Module abrufen
        modules = self.bridge.get_registered_modules()
        
        # Prüfen, ob das Modul registriert wurde
        self.assertIn('mock', modules)
        self.assertEqual(modules['mock'], mock_module)
        
        # Modul abmelden
        success = self.bridge.unregister_module('mock')
        
        # Prüfen, ob die Abmeldung erfolgreich war
        self.assertTrue(success)
        
        # Registrierte Module erneut abrufen
        modules = self.bridge.get_registered_modules()
        
        # Prüfen, ob das Modul abgemeldet wurde
        self.assertNotIn('mock', modules)


if __name__ == '__main__':
    unittest.main()
