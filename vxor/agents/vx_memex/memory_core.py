#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MEMEX: Memory Core Module
----------------------------
Zentrale Verwaltungskomponente für das VXOR-Gedächtnismodul.
Koordiniert die Interaktion zwischen semantischem Speicher, episodischem Speicher
und Arbeitsgedächtnis.

Optimiert für Apple Silicon M4 Max.
"""

import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# Lokale Module importieren
from semantic_store import SemanticStore
from episodic_store import EpisodicStore
from working_memory import WorkingMemory

# Konstanten
DEFAULT_TTL = 60 * 60 * 24 * 30  # 30 Tage in Sekunden
DEFAULT_IMPORTANCE = 0.5  # Standardwert für Wichtigkeit (0-1)
MAX_WORKING_MEMORY_ITEMS = 50  # Maximale Anzahl von Elementen im Arbeitsgedächtnis

class MemoryCore:
    """
    Zentrale Verwaltungsklasse für das VX-MEMEX Gedächtnismodul.
    
    Verantwortlich für:
    - Koordination zwischen den drei Speichersystemen
    - Verwaltung von Speicheroperationen
    - Bereitstellung einer einheitlichen API für VXOR-Komponenten
    - Optimierung der Speichernutzung und -leistung
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert das MemoryCore-Modul mit den drei Speichersystemen.
        
        Args:
            config: Optionale Konfigurationsparameter
        """
        self.config = config or {}
        
        # Logger initialisieren
        self._setup_logging()
        
        # Speichersysteme initialisieren
        self.semantic_store = SemanticStore(self.config.get('semantic', {}))
        self.episodic_store = EpisodicStore(self.config.get('episodic', {}))
        self.working_memory = WorkingMemory(self.config.get('working', {}))
        
        self.logger.info("VX-MEMEX Memory Core initialisiert")
        
    def _setup_logging(self):
        """Konfiguriert das Logging für das Gedächtnismodul."""
        try:
            # Versuche, den VXOR-Logger zu importieren, falls verfügbar
            from vxor_logger import get_logger
            self.logger = get_logger('VX-MEMEX')
        except ImportError:
            # Fallback auf Standard-Logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger('VX-MEMEX')
            self.logger.info("VXOR-Logger nicht gefunden, Standard-Logger wird verwendet")
    
    def store(self, 
              content: Any, 
              memory_type: str = 'all', 
              tags: List[str] = None, 
              ttl: Optional[int] = None,
              importance: float = DEFAULT_IMPORTANCE,
              context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Speichert Inhalte im angegebenen Gedächtnissystem.
        
        Args:
            content: Zu speichernder Inhalt (Text, JSON, etc.)
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            tags: Liste von Schlagwörtern für die Indizierung
            ttl: Time-to-Live in Sekunden (None = Standardwert verwenden)
            importance: Wichtigkeit des Eintrags (0-1)
            context: Zusätzliche Kontextinformationen
            
        Returns:
            Dictionary mit den IDs der erstellten Einträge
        """
        tags = tags or []
        context = context or {}
        ttl = ttl or DEFAULT_TTL
        
        # Zeitstempel für episodische Speicherung
        timestamp = time.time()
        
        # Gemeinsame Eintrags-ID für alle Speichersysteme
        entry_id = str(uuid.uuid4())
        
        # Ergebnisse speichern
        result = {}
        
        # Speichern basierend auf dem angegebenen Typ
        if memory_type in ['semantic', 'all']:
            semantic_id = self.semantic_store.store(
                content=content,
                entry_id=entry_id,
                tags=tags,
                ttl=ttl,
                importance=importance,
                context=context
            )
            result['semantic'] = semantic_id
            self.logger.debug(f"Inhalt im semantischen Speicher gespeichert: {semantic_id}")
        
        if memory_type in ['episodic', 'all']:
            episodic_id = self.episodic_store.store(
                content=content,
                entry_id=entry_id,
                timestamp=timestamp,
                tags=tags,
                ttl=ttl,
                importance=importance,
                context=context
            )
            result['episodic'] = episodic_id
            self.logger.debug(f"Inhalt im episodischen Speicher gespeichert: {episodic_id}")
        
        if memory_type in ['working', 'all']:
            working_id = self.working_memory.store(
                content=content,
                entry_id=entry_id,
                tags=tags,
                ttl=ttl,
                importance=importance,
                context=context
            )
            result['working'] = working_id
            self.logger.debug(f"Inhalt im Arbeitsgedächtnis gespeichert: {working_id}")
            
            # Arbeitsgedächtnis-Management: Älteste/unwichtigste Einträge entfernen, wenn Limit erreicht
            self._manage_working_memory()
        
        return result
    
    def retrieve(self, 
                query: Union[str, Dict[str, Any]], 
                memory_type: str = 'all',
                limit: int = 10,
                min_similarity: float = 0.6) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ruft Inhalte aus dem angegebenen Gedächtnissystem ab.
        
        Args:
            query: Suchanfrage (Text oder strukturierte Abfrage)
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            limit: Maximale Anzahl von Ergebnissen
            min_similarity: Minimale Ähnlichkeit für semantische Suche
            
        Returns:
            Dictionary mit Ergebnissen aus den abgefragten Speichersystemen
        """
        results = {}
        
        if memory_type in ['semantic', 'all']:
            semantic_results = self.semantic_store.retrieve(
                query=query,
                limit=limit,
                min_similarity=min_similarity
            )
            results['semantic'] = semantic_results
            
        if memory_type in ['episodic', 'all']:
            episodic_results = self.episodic_store.retrieve(
                query=query,
                limit=limit,
                min_similarity=min_similarity
            )
            results['episodic'] = episodic_results
            
        if memory_type in ['working', 'all']:
            working_results = self.working_memory.retrieve(
                query=query,
                limit=limit,
                min_similarity=min_similarity
            )
            results['working'] = working_results
        
        return results
    
    def retrieve_by_id(self, entry_id: str, memory_type: str = 'all') -> Dict[str, Any]:
        """
        Ruft einen Eintrag anhand seiner ID aus dem angegebenen Gedächtnissystem ab.
        
        Args:
            entry_id: ID des abzurufenden Eintrags
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            
        Returns:
            Dictionary mit dem Eintrag aus den abgefragten Speichersystemen
        """
        results = {}
        
        if memory_type in ['semantic', 'all']:
            try:
                semantic_result = self.semantic_store.retrieve_by_id(entry_id)
                results['semantic'] = semantic_result
            except KeyError:
                self.logger.debug(f"Eintrag {entry_id} nicht im semantischen Speicher gefunden")
            
        if memory_type in ['episodic', 'all']:
            try:
                episodic_result = self.episodic_store.retrieve_by_id(entry_id)
                results['episodic'] = episodic_result
            except KeyError:
                self.logger.debug(f"Eintrag {entry_id} nicht im episodischen Speicher gefunden")
            
        if memory_type in ['working', 'all']:
            try:
                working_result = self.working_memory.retrieve_by_id(entry_id)
                results['working'] = working_result
            except KeyError:
                self.logger.debug(f"Eintrag {entry_id} nicht im Arbeitsgedächtnis gefunden")
        
        return results
    
    def update(self, 
              entry_id: str, 
              content: Any = None,
              tags: List[str] = None,
              ttl: Optional[int] = None,
              importance: Optional[float] = None,
              context: Dict[str, Any] = None,
              memory_type: str = 'all') -> Dict[str, bool]:
        """
        Aktualisiert einen Eintrag in den angegebenen Gedächtnissystemen.
        
        Args:
            entry_id: ID des zu aktualisierenden Eintrags
            content: Neuer Inhalt (None = unverändert)
            tags: Neue Schlagwörter (None = unverändert)
            ttl: Neue Time-to-Live (None = unverändert)
            importance: Neue Wichtigkeit (None = unverändert)
            context: Neuer Kontext (None = unverändert)
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            
        Returns:
            Dictionary mit Erfolgs-Flags für jedes aktualisierte Speichersystem
        """
        results = {}
        
        if memory_type in ['semantic', 'all']:
            try:
                success = self.semantic_store.update(
                    entry_id=entry_id,
                    content=content,
                    tags=tags,
                    ttl=ttl,
                    importance=importance,
                    context=context
                )
                results['semantic'] = success
            except KeyError:
                results['semantic'] = False
                self.logger.debug(f"Eintrag {entry_id} nicht im semantischen Speicher gefunden")
            
        if memory_type in ['episodic', 'all']:
            try:
                success = self.episodic_store.update(
                    entry_id=entry_id,
                    content=content,
                    tags=tags,
                    ttl=ttl,
                    importance=importance,
                    context=context
                )
                results['episodic'] = success
            except KeyError:
                results['episodic'] = False
                self.logger.debug(f"Eintrag {entry_id} nicht im episodischen Speicher gefunden")
            
        if memory_type in ['working', 'all']:
            try:
                success = self.working_memory.update(
                    entry_id=entry_id,
                    content=content,
                    tags=tags,
                    ttl=ttl,
                    importance=importance,
                    context=context
                )
                results['working'] = success
            except KeyError:
                results['working'] = False
                self.logger.debug(f"Eintrag {entry_id} nicht im Arbeitsgedächtnis gefunden")
        
        return results
    
    def delete(self, entry_id: str, memory_type: str = 'all') -> Dict[str, bool]:
        """
        Löscht einen Eintrag aus den angegebenen Gedächtnissystemen.
        
        Args:
            entry_id: ID des zu löschenden Eintrags
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            
        Returns:
            Dictionary mit Erfolgs-Flags für jedes aktualisierte Speichersystem
        """
        results = {}
        
        if memory_type in ['semantic', 'all']:
            try:
                success = self.semantic_store.delete(entry_id)
                results['semantic'] = success
            except KeyError:
                results['semantic'] = False
                self.logger.debug(f"Eintrag {entry_id} nicht im semantischen Speicher gefunden")
            
        if memory_type in ['episodic', 'all']:
            try:
                success = self.episodic_store.delete(entry_id)
                results['episodic'] = success
            except KeyError:
                results['episodic'] = False
                self.logger.debug(f"Eintrag {entry_id} nicht im episodischen Speicher gefunden")
            
        if memory_type in ['working', 'all']:
            try:
                success = self.working_memory.delete(entry_id)
                results['working'] = success
            except KeyError:
                results['working'] = False
                self.logger.debug(f"Eintrag {entry_id} nicht im Arbeitsgedächtnis gefunden")
        
        return results
    
    def retrieve_by_tags(self, 
                        tags: List[str], 
                        memory_type: str = 'all',
                        match_all: bool = False,
                        limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ruft Einträge anhand von Schlagwörtern aus den angegebenen Gedächtnissystemen ab.
        
        Args:
            tags: Liste von Schlagwörtern
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            match_all: True, wenn alle Schlagwörter übereinstimmen müssen
            limit: Maximale Anzahl von Ergebnissen
            
        Returns:
            Dictionary mit Ergebnissen aus den abgefragten Speichersystemen
        """
        results = {}
        
        if memory_type in ['semantic', 'all']:
            semantic_results = self.semantic_store.retrieve_by_tags(
                tags=tags,
                match_all=match_all,
                limit=limit
            )
            results['semantic'] = semantic_results
            
        if memory_type in ['episodic', 'all']:
            episodic_results = self.episodic_store.retrieve_by_tags(
                tags=tags,
                match_all=match_all,
                limit=limit
            )
            results['episodic'] = episodic_results
            
        if memory_type in ['working', 'all']:
            working_results = self.working_memory.retrieve_by_tags(
                tags=tags,
                match_all=match_all,
                limit=limit
            )
            results['working'] = working_results
        
        return results
    
    def retrieve_by_time_range(self, 
                              start_time: float, 
                              end_time: float,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ruft Einträge aus dem episodischen Speicher anhand eines Zeitbereichs ab.
        
        Args:
            start_time: Startzeit als Unix-Timestamp
            end_time: Endzeit als Unix-Timestamp
            limit: Maximale Anzahl von Ergebnissen
            
        Returns:
            Liste mit Einträgen aus dem episodischen Speicher
        """
        return self.episodic_store.retrieve_by_time_range(
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    def cleanup(self, memory_type: str = 'all') -> Dict[str, int]:
        """
        Bereinigt abgelaufene Einträge in den angegebenen Gedächtnissystemen.
        
        Args:
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            
        Returns:
            Dictionary mit der Anzahl der bereinigten Einträge pro Speichersystem
        """
        results = {}
        
        if memory_type in ['semantic', 'all']:
            count = self.semantic_store.cleanup()
            results['semantic'] = count
            
        if memory_type in ['episodic', 'all']:
            count = self.episodic_store.cleanup()
            results['episodic'] = count
            
        if memory_type in ['working', 'all']:
            count = self.working_memory.cleanup()
            results['working'] = count
        
        return results
    
    def _manage_working_memory(self) -> None:
        """
        Verwaltet das Arbeitsgedächtnis, indem älteste/unwichtigste Einträge entfernt werden,
        wenn das Limit erreicht ist.
        """
        current_count = self.working_memory.count()
        
        if current_count > MAX_WORKING_MEMORY_ITEMS:
            # Berechne, wie viele Einträge entfernt werden müssen
            to_remove = current_count - MAX_WORKING_MEMORY_ITEMS
            
            # Hole die ältesten/unwichtigsten Einträge
            entries = self.working_memory.get_least_important(to_remove)
            
            # Entferne diese Einträge
            for entry in entries:
                self.working_memory.delete(entry['id'])
                self.logger.debug(f"Eintrag {entry['id']} aus dem Arbeitsgedächtnis entfernt (Limit erreicht)")
    
    def link_entries(self, source_id: str, target_id: str, link_type: str = 'related') -> bool:
        """
        Verknüpft zwei Einträge miteinander.
        
        Args:
            source_id: ID des Quelleintrags
            target_id: ID des Zieleintrags
            link_type: Art der Verknüpfung
            
        Returns:
            True, wenn die Verknüpfung erfolgreich war
        """
        # Versuche, die Einträge in allen Speichersystemen zu verknüpfen
        semantic_linked = self.semantic_store.link_entries(source_id, target_id, link_type)
        episodic_linked = self.episodic_store.link_entries(source_id, target_id, link_type)
        working_linked = self.working_memory.link_entries(source_id, target_id, link_type)
        
        # Wenn mindestens eine Verknüpfung erfolgreich war, gilt die Operation als erfolgreich
        return semantic_linked or episodic_linked or working_linked
    
    def get_linked_entries(self, entry_id: str, link_type: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ruft alle mit einem Eintrag verknüpften Einträge ab.
        
        Args:
            entry_id: ID des Eintrags
            link_type: Optional, Art der Verknüpfung
            
        Returns:
            Dictionary mit verknüpften Einträgen aus allen Speichersystemen
        """
        results = {}
        
        # Versuche, verknüpfte Einträge aus allen Speichersystemen abzurufen
        try:
            semantic_links = self.semantic_store.get_linked_entries(entry_id, link_type)
            results['semantic'] = semantic_links
        except KeyError:
            results['semantic'] = []
        
        try:
            episodic_links = self.episodic_store.get_linked_entries(entry_id, link_type)
            results['episodic'] = episodic_links
        except KeyError:
            results['episodic'] = []
        
        try:
            working_links = self.working_memory.get_linked_entries(entry_id, link_type)
            results['working'] = working_links
        except KeyError:
            results['working'] = []
        
        return results
    
    def export_to_json(self, memory_type: str = 'all', file_path: Optional[str] = None) -> Optional[str]:
        """
        Exportiert den Inhalt der angegebenen Gedächtnissysteme als JSON.
        
        Args:
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            file_path: Optional, Pfad für die Ausgabedatei
            
        Returns:
            JSON-String oder None, wenn in eine Datei exportiert wurde
        """
        export_data = {}
        
        if memory_type in ['semantic', 'all']:
            export_data['semantic'] = self.semantic_store.export_data()
            
        if memory_type in ['episodic', 'all']:
            export_data['episodic'] = self.episodic_store.export_data()
            
        if memory_type in ['working', 'all']:
            export_data['working'] = self.working_memory.export_data()
        
        # Exportiere als JSON-String oder in eine Datei
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            return None
        else:
            return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def import_from_json(self, json_data: Union[str, Dict], memory_type: str = 'all') -> Dict[str, int]:
        """
        Importiert Daten aus JSON in die angegebenen Gedächtnissysteme.
        
        Args:
            json_data: JSON-String oder Dictionary
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            
        Returns:
            Dictionary mit der Anzahl der importierten Einträge pro Speichersystem
        """
        # Konvertiere JSON-String zu Dictionary, falls nötig
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        results = {}
        
        if memory_type in ['semantic', 'all'] and 'semantic' in data:
            count = self.semantic_store.import_data(data['semantic'])
            results['semantic'] = count
            
        if memory_type in ['episodic', 'all'] and 'episodic' in data:
            count = self.episodic_store.import_data(data['episodic'])
            results['episodic'] = count
            
        if memory_type in ['working', 'all'] and 'working' in data:
            count = self.working_memory.import_data(data['working'])
            results['working'] = count
        
        return results
    
    def get_stats(self, memory_type: str = 'all') -> Dict[str, Dict[str, Any]]:
        """
        Gibt Statistiken über die angegebenen Gedächtnissysteme zurück.
        
        Args:
            memory_type: Zielgedächtnissystem ('semantic', 'episodic', 'working' oder 'all')
            
        Returns:
            Dictionary mit Statistiken für jedes Speichersystem
        """
        stats = {}
        
        if memory_type in ['semantic', 'all']:
            stats['semantic'] = {
                'count': self.semantic_store.count(),
                'size': self.semantic_store.size(),
                'avg_importance': self.semantic_store.avg_importance()
            }
            
        if memory_type in ['episodic', 'all']:
            stats['episodic'] = {
                'count': self.episodic_store.count(),
                'size': self.episodic_store.size(),
                'avg_importance': self.episodic_store.avg_importance(),
                'time_range': self.episodic_store.get_time_range()
            }
            
        if memory_type in ['working', 'all']:
            stats['working'] = {
                'count': self.working_memory.count(),
                'size': self.working_memory.size(),
                'avg_importance': self.working_memory.avg_importance()
            }
        
        return stats


# Wenn direkt ausgeführt, führe einen einfachen Test durch
if __name__ == "__main__":
    # Konfiguration für den Test
    test_config = {
        'semantic': {'vector_dim': 384},
        'episodic': {'max_entries': 10000},
        'working': {'ttl_default': 3600}  # 1 Stunde
    }
    
    # Memory Core initialisieren
    memory = MemoryCore(test_config)
    
    # Testdaten speichern
    test_content = {
        'text': 'Dies ist ein Testinhalt für das VX-MEMEX Gedächtnismodul.',
        'metadata': {
            'source': 'unit_test',
            'version': '1.0'
        }
    }
    
    # In allen Speichersystemen speichern
    result = memory.store(
        content=test_content,
        memory_type='all',
        tags=['test', 'vx-memex', 'memory'],
        importance=0.8,
        context={'test_run': True}
    )
    
    print(f"Speicherergebnis: {result}")
    
    # Aus allen Speichersystemen abrufen
    retrieved = memory.retrieve_by_id(list(result.values())[0])
    
    print(f"Abgerufene Daten: {retrieved}")
    
    # Statistiken anzeigen
    stats = memory.get_stats()
    
    print(f"Speicherstatistiken: {stats}")
