#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MEMEX: Semantic Store Module
-------------------------------
Semantisches Speichersystem für das VXOR-Gedächtnismodul.
Verantwortlich für die Speicherung und den Abruf von semantischem Wissen
(Konzepte, Begriffe, Fakten) mit Vektorrepräsentationen für semantische Ähnlichkeitssuche.

Optimiert für Apple Silicon M4 Max.
"""

import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

try:
    # Versuche, MLX für optimierte Vektoroperationen auf Apple Silicon zu importieren
    import mlx.core as mx
    USE_MLX = True
except ImportError:
    # Fallback auf NumPy, wenn MLX nicht verfügbar ist
    USE_MLX = False
    logging.warning("MLX nicht verfügbar, verwende NumPy als Fallback")

# Konstanten
DEFAULT_VECTOR_DIM = 384  # Standarddimension für Vektorrepräsentationen
DEFAULT_TTL = 60 * 60 * 24 * 30  # 30 Tage in Sekunden
DEFAULT_IMPORTANCE = 0.5  # Standardwert für Wichtigkeit (0-1)

class SemanticStore:
    """
    Semantisches Speichersystem für das VX-MEMEX Gedächtnismodul.
    
    Verantwortlich für:
    - Speicherung von semantischem Wissen (Konzepte, Begriffe, Fakten)
    - Vektorrepräsentationen für semantische Ähnlichkeitssuche
    - Effiziente Indizierung und Abruf von Wissen
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert das semantische Speichersystem.
        
        Args:
            config: Optionale Konfigurationsparameter
        """
        self.config = config or {}
        
        # Logger initialisieren
        self._setup_logging()
        
        # Vektordimension festlegen
        self.vector_dim = self.config.get('vector_dim', DEFAULT_VECTOR_DIM)
        
        # Speicherstrukturen initialisieren
        self.entries = {}  # Hauptspeicher für Einträge
        self.tag_index = {}  # Index für Schlagwörter
        self.vector_store = {}  # Speicher für Vektorrepräsentationen
        self.links = {}  # Speicher für Verknüpfungen zwischen Einträgen
        
        # Texteinbettungsmodell initialisieren (Mock-Implementierung)
        self._init_embedding_model()
        
        self.logger.info(f"Semantischer Speicher initialisiert (Vektordimension: {self.vector_dim})")
    
    def _setup_logging(self):
        """Konfiguriert das Logging für das semantische Speichersystem."""
        try:
            # Versuche, den VXOR-Logger zu importieren, falls verfügbar
            from vxor_logger import get_logger
            self.logger = get_logger('VX-MEMEX.semantic')
        except ImportError:
            # Fallback auf Standard-Logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger('VX-MEMEX.semantic')
            self.logger.info("VXOR-Logger nicht gefunden, Standard-Logger wird verwendet")
    
    def _init_embedding_model(self):
        """
        Initialisiert das Texteinbettungsmodell für semantische Vektoren.
        
        In einer vollständigen Implementierung würde hier ein echtes Sprachmodell geladen.
        Für diese Implementierung verwenden wir eine einfache Mock-Funktion.
        """
        # Mock-Implementierung für Texteinbettungen
        # In einer echten Implementierung würde hier ein Sprachmodell geladen
        self.logger.info("Mock-Texteinbettungsmodell initialisiert")
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Erzeugt eine Vektorrepräsentation für einen Text.
        
        Args:
            text: Einzubettender Text
            
        Returns:
            Vektorrepräsentation als NumPy-Array
        """
        # Mock-Implementierung für Texteinbettungen
        # In einer echten Implementierung würde hier ein Sprachmodell verwendet
        
        # Einfache deterministische Hash-basierte Einbettung für Konsistenz
        import hashlib
        
        # Text in Bytes umwandeln
        text_bytes = text.encode('utf-8')
        
        # SHA-256-Hash berechnen
        hash_obj = hashlib.sha256(text_bytes)
        hash_bytes = hash_obj.digest()
        
        # Hash in Zahlen umwandeln und auf die gewünschte Dimension skalieren
        hash_values = []
        for i in range(0, min(32, self.vector_dim), 4):
            if i < len(hash_bytes) - 3:
                # 4 Bytes in eine Zahl umwandeln
                value = int.from_bytes(hash_bytes[i:i+4], byteorder='big')
                # Auf den Bereich [-1, 1] normalisieren
                normalized_value = (value / (2**32 - 1)) * 2 - 1
                hash_values.append(normalized_value)
        
        # Auffüllen oder Kürzen auf die gewünschte Dimension
        while len(hash_values) < self.vector_dim:
            # Restliche Werte mit deterministischen, aber unterschiedlichen Werten auffüllen
            seed = len(hash_values)
            value = (hash_values[seed % len(hash_values)] + seed / self.vector_dim) / 2
            hash_values.append(value)
        
        # Auf die gewünschte Dimension kürzen
        hash_values = hash_values[:self.vector_dim]
        
        # In NumPy-Array umwandeln
        embedding = np.array(hash_values, dtype=np.float32)
        
        # Normalisieren
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Berechnet die Kosinus-Ähnlichkeit zwischen zwei Vektoren.
        
        Args:
            vec1: Erster Vektor
            vec2: Zweiter Vektor
            
        Returns:
            Kosinus-Ähnlichkeit als Wert zwischen -1 und 1
        """
        if USE_MLX:
            # MLX-Implementierung für optimierte Leistung auf Apple Silicon
            v1 = mx.array(vec1)
            v2 = mx.array(vec2)
            dot_product = mx.sum(v1 * v2)
            norm1 = mx.sqrt(mx.sum(v1 * v1))
            norm2 = mx.sqrt(mx.sum(v2 * v2))
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        else:
            # NumPy-Implementierung als Fallback
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # Vermeidung von Division durch Null
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
    
    def _extract_text_content(self, content: Any) -> str:
        """
        Extrahiert Textinhalt aus verschiedenen Inhaltstypen für die Vektorisierung.
        
        Args:
            content: Inhalt (Text, Dictionary, Liste, etc.)
            
        Returns:
            Extrahierter Text
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Rekursiv alle Werte extrahieren und zusammenführen
            text_parts = []
            for key, value in content.items():
                text_parts.append(f"{key}: {self._extract_text_content(value)}")
            return " ".join(text_parts)
        elif isinstance(content, list):
            # Rekursiv alle Elemente extrahieren und zusammenführen
            text_parts = []
            for item in content:
                text_parts.append(self._extract_text_content(item))
            return " ".join(text_parts)
        else:
            # Andere Typen in Strings umwandeln
            return str(content)
    
    def store(self, 
             content: Any, 
             entry_id: Optional[str] = None,
             tags: List[str] = None,
             ttl: Optional[int] = None,
             importance: float = DEFAULT_IMPORTANCE,
             context: Dict[str, Any] = None) -> str:
        """
        Speichert einen Eintrag im semantischen Speicher.
        
        Args:
            content: Zu speichernder Inhalt
            entry_id: Optionale ID für den Eintrag (wird generiert, wenn nicht angegeben)
            tags: Liste von Schlagwörtern für die Indizierung
            ttl: Time-to-Live in Sekunden (None = Standardwert verwenden)
            importance: Wichtigkeit des Eintrags (0-1)
            context: Zusätzliche Kontextinformationen
            
        Returns:
            ID des gespeicherten Eintrags
        """
        # Standardwerte setzen
        tags = tags or []
        context = context or {}
        ttl = ttl or DEFAULT_TTL
        
        # ID generieren, falls nicht angegeben
        if entry_id is None:
            entry_id = str(uuid.uuid4())
        
        # Aktuelle Zeit für Zeitstempel
        current_time = time.time()
        
        # Ablaufzeit berechnen
        expiry_time = current_time + ttl if ttl is not None else None
        
        # Text für die Vektorisierung extrahieren
        text_content = self._extract_text_content(content)
        
        # Vektorrepräsentation erzeugen
        vector = self._get_text_embedding(text_content)
        
        # Eintrag erstellen
        entry = {
            'id': entry_id,
            'content': content,
            'tags': tags,
            'created_at': current_time,
            'updated_at': current_time,
            'expiry_time': expiry_time,
            'importance': importance,
            'context': context
        }
        
        # Eintrag speichern
        self.entries[entry_id] = entry
        
        # Vektorrepräsentation speichern
        self.vector_store[entry_id] = vector
        
        # Schlagwörter indizieren
        for tag in tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(entry_id)
        
        # Verknüpfungsstruktur initialisieren
        self.links[entry_id] = {}
        
        self.logger.debug(f"Eintrag im semantischen Speicher gespeichert: {entry_id}")
        
        return entry_id
    
    def retrieve(self, 
                query: Union[str, Dict[str, Any]], 
                limit: int = 10,
                min_similarity: float = 0.6) -> List[Dict[str, Any]]:
        """
        Ruft Einträge anhand einer Suchanfrage ab.
        
        Args:
            query: Suchanfrage (Text oder strukturierte Abfrage)
            limit: Maximale Anzahl von Ergebnissen
            min_similarity: Minimale Ähnlichkeit für Ergebnisse
            
        Returns:
            Liste mit ähnlichen Einträgen
        """
        # Text für die Vektorisierung extrahieren
        if isinstance(query, dict):
            query_text = self._extract_text_content(query)
        else:
            query_text = query
        
        # Vektorrepräsentation für die Anfrage erzeugen
        query_vector = self._get_text_embedding(query_text)
        
        # Ähnlichkeiten berechnen
        similarities = []
        current_time = time.time()
        
        for entry_id, entry in self.entries.items():
            # Abgelaufene Einträge überspringen
            if entry['expiry_time'] is not None and entry['expiry_time'] < current_time:
                continue
            
            # Vektorrepräsentation abrufen
            entry_vector = self.vector_store[entry_id]
            
            # Ähnlichkeit berechnen
            similarity = self._compute_similarity(query_vector, entry_vector)
            
            # Nur Einträge mit ausreichender Ähnlichkeit berücksichtigen
            if similarity >= min_similarity:
                similarities.append((entry_id, similarity))
        
        # Nach Ähnlichkeit sortieren (absteigend)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Auf die gewünschte Anzahl begrenzen
        similarities = similarities[:limit]
        
        # Ergebnisse zusammenstellen
        results = []
        for entry_id, similarity in similarities:
            entry = self.entries[entry_id].copy()
            entry['similarity'] = similarity
            results.append(entry)
        
        return results
    
    def retrieve_by_id(self, entry_id: str) -> Dict[str, Any]:
        """
        Ruft einen Eintrag anhand seiner ID ab.
        
        Args:
            entry_id: ID des abzurufenden Eintrags
            
        Returns:
            Eintrag als Dictionary
            
        Raises:
            KeyError: Wenn der Eintrag nicht gefunden wurde oder abgelaufen ist
        """
        # Prüfen, ob der Eintrag existiert
        if entry_id not in self.entries:
            raise KeyError(f"Eintrag nicht gefunden: {entry_id}")
        
        entry = self.entries[entry_id]
        
        # Prüfen, ob der Eintrag abgelaufen ist
        current_time = time.time()
        if entry['expiry_time'] is not None and entry['expiry_time'] < current_time:
            raise KeyError(f"Eintrag abgelaufen: {entry_id}")
        
        return entry.copy()
    
    def update(self, 
              entry_id: str, 
              content: Any = None,
              tags: List[str] = None,
              ttl: Optional[int] = None,
              importance: Optional[float] = None,
              context: Dict[str, Any] = None) -> bool:
        """
        Aktualisiert einen Eintrag im semantischen Speicher.
        
        Args:
            entry_id: ID des zu aktualisierenden Eintrags
            content: Neuer Inhalt (None = unverändert)
            tags: Neue Schlagwörter (None = unverändert)
            ttl: Neue Time-to-Live (None = unverändert)
            importance: Neue Wichtigkeit (None = unverändert)
            context: Neuer Kontext (None = unverändert)
            
        Returns:
            True, wenn die Aktualisierung erfolgreich war
            
        Raises:
            KeyError: Wenn der Eintrag nicht gefunden wurde
        """
        # Prüfen, ob der Eintrag existiert
        if entry_id not in self.entries:
            raise KeyError(f"Eintrag nicht gefunden: {entry_id}")
        
        entry = self.entries[entry_id]
        
        # Aktuelle Zeit für Zeitstempel
        current_time = time.time()
        
        # Inhalt aktualisieren, falls angegeben
        if content is not None:
            entry['content'] = content
            
            # Text für die Vektorisierung extrahieren
            text_content = self._extract_text_content(content)
            
            # Vektorrepräsentation aktualisieren
            self.vector_store[entry_id] = self._get_text_embedding(text_content)
        
        # Schlagwörter aktualisieren, falls angegeben
        if tags is not None:
            # Alte Schlagwörter aus dem Index entfernen
            for tag in entry['tags']:
                if tag in self.tag_index and entry_id in self.tag_index[tag]:
                    self.tag_index[tag].remove(entry_id)
                    # Leere Sets entfernen
                    if not self.tag_index[tag]:
                        del self.tag_index[tag]
            
            # Neue Schlagwörter setzen
            entry['tags'] = tags
            
            # Neue Schlagwörter indizieren
            for tag in tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(entry_id)
        
        # TTL aktualisieren, falls angegeben
        if ttl is not None:
            entry['expiry_time'] = current_time + ttl
        
        # Wichtigkeit aktualisieren, falls angegeben
        if importance is not None:
            entry['importance'] = importance
        
        # Kontext aktualisieren, falls angegeben
        if context is not None:
            entry['context'] = context
        
        # Aktualisierungszeitstempel setzen
        entry['updated_at'] = current_time
        
        self.logger.debug(f"Eintrag im semantischen Speicher aktualisiert: {entry_id}")
        
        return True
    
    def delete(self, entry_id: str) -> bool:
        """
        Löscht einen Eintrag aus dem semantischen Speicher.
        
        Args:
            entry_id: ID des zu löschenden Eintrags
            
        Returns:
            True, wenn der Eintrag gelöscht wurde, False, wenn der Eintrag nicht existiert
        """
        # Prüfen, ob der Eintrag existiert
        if entry_id not in self.entries:
            return False
        
        entry = self.entries[entry_id]
        
        # Schlagwörter aus dem Index entfernen
        for tag in entry['tags']:
            if tag in self.tag_index and entry_id in self.tag_index[tag]:
                self.tag_index[tag].remove(entry_id)
                # Leere Sets entfernen
                if not self.tag_index[tag]:
                    del self.tag_index[tag]
        
        # Vektorrepräsentation entfernen
        if entry_id in self.vector_store:
            del self.vector_store[entry_id]
        
        # Verknüpfungen entfernen
        if entry_id in self.links:
            del self.links[entry_id]
        
        # Verknüpfungen zu diesem Eintrag in anderen Einträgen entfernen
        for other_id, other_links in self.links.items():
            for link_type, linked_ids in list(other_links.items()):
                if entry_id in linked_ids:
                    linked_ids.remove(entry_id)
                    # Leere Listen entfernen
                    if not linked_ids:
                        del other_links[link_type]
        
        # Eintrag entfernen
        del self.entries[entry_id]
        
        self.logger.debug(f"Eintrag aus dem semantischen Speicher gelöscht: {entry_id}")
        
        return True
    
    def retrieve_by_tags(self, 
                        tags: List[str], 
                        match_all: bool = False,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ruft Einträge anhand von Schlagwörtern ab.
        
        Args:
            tags: Liste von Schlagwörtern
            match_all: True, wenn alle Schlagwörter übereinstimmen müssen
            limit: Maximale Anzahl von Ergebnissen
            
        Returns:
            Liste mit übereinstimmenden Einträgen
        """
        # Einträge anhand von Schlagwörtern finden
        matching_ids = set()
        
        if match_all:
            # Alle Schlagwörter müssen übereinstimmen
            if not tags:
                return []
            
            # Mit dem ersten Schlagwort initialisieren
            if tags[0] in self.tag_index:
                matching_ids = set(self.tag_index[tags[0]])
            else:
                return []
            
            # Mit den restlichen Schlagwörtern schneiden
            for tag in tags[1:]:
                if tag in self.tag_index:
                    matching_ids &= self.tag_index[tag]
                else:
                    return []
                
                # Abbrechen, wenn keine Übereinstimmungen mehr
                if not matching_ids:
                    return []
        else:
            # Mindestens ein Schlagwort muss übereinstimmen
            for tag in tags:
                if tag in self.tag_index:
                    matching_ids |= self.tag_index[tag]
        
        # Abgelaufene Einträge filtern
        current_time = time.time()
        valid_entries = []
        
        for entry_id in matching_ids:
            entry = self.entries[entry_id]
            
            # Abgelaufene Einträge überspringen
            if entry['expiry_time'] is not None and entry['expiry_time'] < current_time:
                continue
            
            valid_entries.append(entry.copy())
        
        # Nach Wichtigkeit sortieren (absteigend)
        valid_entries.sort(key=lambda x: x['importance'], reverse=True)
        
        # Auf die gewünschte Anzahl begrenzen
        return valid_entries[:limit]
    
    def cleanup(self) -> int:
        """
        Bereinigt abgelaufene Einträge im semantischen Speicher.
        
        Returns:
            Anzahl der bereinigten Einträge
        """
        current_time = time.time()
        expired_ids = []
        
        # Abgelaufene Einträge identifizieren
        for entry_id, entry in self.entries.items():
            if entry['expiry_time'] is not None and entry['expiry_time'] < current_time:
                expired_ids.append(entry_id)
        
        # Abgelaufene Einträge löschen
        for entry_id in expired_ids:
            self.delete(entry_id)
        
        self.logger.info(f"{len(expired_ids)} abgelaufene Einträge aus dem semantischen Speicher bereinigt")
        
        return len(expired_ids)
    
    def count(self) -> int:
        """
        Gibt die Anzahl der Einträge im semantischen Speicher zurück.
        
        Returns:
            Anzahl der Einträge
        """
        return len(self.entries)
    
    def size(self) -> int:
        """
        Gibt die Größe des semantischen Speichers in Bytes zurück (Schätzung).
        
        Returns:
            Geschätzte Größe in Bytes
        """
        import sys
        
        # Größe der Einträge schätzen
        entries_size = sys.getsizeof(self.entries)
        for entry_id, entry in self.entries.items():
            entries_size += sys.getsizeof(entry_id)
            entries_size += sys.getsizeof(entry)
            for key, value in entry.items():
                entries_size += sys.getsizeof(key)
                entries_size += sys.getsizeof(value)
        
        # Größe der Vektorrepräsentationen schätzen
        vector_size = sys.getsizeof(self.vector_store)
        for entry_id, vector in self.vector_store.items():
            vector_size += sys.getsizeof(entry_id)
            vector_size += sys.getsizeof(vector)
            vector_size += vector.nbytes
        
        # Größe des Schlagwortindex schätzen
        tag_index_size = sys.getsizeof(self.tag_index)
        for tag, ids in self.tag_index.items():
            tag_index_size += sys.getsizeof(tag)
            tag_index_size += sys.getsizeof(ids)
            for entry_id in ids:
                tag_index_size += sys.getsizeof(entry_id)
        
        # Größe der Verknüpfungen schätzen
        links_size = sys.getsizeof(self.links)
        for entry_id, entry_links in self.links.items():
            links_size += sys.getsizeof(entry_id)
            links_size += sys.getsizeof(entry_links)
            for link_type, linked_ids in entry_links.items():
                links_size += sys.getsizeof(link_type)
                links_size += sys.getsizeof(linked_ids)
                for linked_id in linked_ids:
                    links_size += sys.getsizeof(linked_id)
        
        return entries_size + vector_size + tag_index_size + links_size
    
    def avg_importance(self) -> float:
        """
        Berechnet die durchschnittliche Wichtigkeit aller Einträge.
        
        Returns:
            Durchschnittliche Wichtigkeit oder 0, wenn keine Einträge vorhanden sind
        """
        if not self.entries:
            return 0.0
        
        total_importance = sum(entry['importance'] for entry in self.entries.values())
        return total_importance / len(self.entries)
    
    def link_entries(self, source_id: str, target_id: str, link_type: str = 'related') -> bool:
        """
        Verknüpft zwei Einträge miteinander.
        
        Args:
            source_id: ID des Quelleintrags
            target_id: ID des Zieleintrags
            link_type: Art der Verknüpfung
            
        Returns:
            True, wenn die Verknüpfung erfolgreich war
            
        Raises:
            KeyError: Wenn einer der Einträge nicht gefunden wurde
        """
        # Prüfen, ob die Einträge existieren
        if source_id not in self.entries:
            raise KeyError(f"Quelleintrag nicht gefunden: {source_id}")
        
        if target_id not in self.entries:
            raise KeyError(f"Zieleintrag nicht gefunden: {target_id}")
        
        # Verknüpfungsstruktur initialisieren, falls noch nicht vorhanden
        if source_id not in self.links:
            self.links[source_id] = {}
        
        if link_type not in self.links[source_id]:
            self.links[source_id][link_type] = set()
        
        # Verknüpfung hinzufügen
        self.links[source_id][link_type].add(target_id)
        
        self.logger.debug(f"Einträge verknüpft: {source_id} -> {target_id} ({link_type})")
        
        return True
    
    def get_linked_entries(self, entry_id: str, link_type: str = None) -> List[Dict[str, Any]]:
        """
        Ruft alle mit einem Eintrag verknüpften Einträge ab.
        
        Args:
            entry_id: ID des Eintrags
            link_type: Optional, Art der Verknüpfung
            
        Returns:
            Liste mit verknüpften Einträgen
            
        Raises:
            KeyError: Wenn der Eintrag nicht gefunden wurde
        """
        # Prüfen, ob der Eintrag existiert
        if entry_id not in self.entries:
            raise KeyError(f"Eintrag nicht gefunden: {entry_id}")
        
        # Prüfen, ob Verknüpfungen vorhanden sind
        if entry_id not in self.links:
            return []
        
        linked_ids = set()
        
        if link_type is not None:
            # Nur Verknüpfungen des angegebenen Typs abrufen
            if link_type in self.links[entry_id]:
                linked_ids = self.links[entry_id][link_type]
        else:
            # Alle Verknüpfungen abrufen
            for type_links in self.links[entry_id].values():
                linked_ids.update(type_links)
        
        # Verknüpfte Einträge abrufen
        linked_entries = []
        current_time = time.time()
        
        for linked_id in linked_ids:
            # Prüfen, ob der verknüpfte Eintrag existiert
            if linked_id not in self.entries:
                continue
            
            entry = self.entries[linked_id]
            
            # Abgelaufene Einträge überspringen
            if entry['expiry_time'] is not None and entry['expiry_time'] < current_time:
                continue
            
            linked_entries.append(entry.copy())
        
        return linked_entries
    
    def export_data(self) -> Dict[str, Any]:
        """
        Exportiert alle Daten des semantischen Speichers.
        
        Returns:
            Dictionary mit allen Daten
        """
        # Vektorrepräsentationen in Listen umwandeln
        vector_store_serializable = {}
        for entry_id, vector in self.vector_store.items():
            vector_store_serializable[entry_id] = vector.tolist()
        
        # Schlagwortindex in Listen umwandeln
        tag_index_serializable = {}
        for tag, ids in self.tag_index.items():
            tag_index_serializable[tag] = list(ids)
        
        # Verknüpfungen in Listen umwandeln
        links_serializable = {}
        for entry_id, entry_links in self.links.items():
            links_serializable[entry_id] = {}
            for link_type, linked_ids in entry_links.items():
                links_serializable[entry_id][link_type] = list(linked_ids)
        
        return {
            'entries': self.entries,
            'vector_store': vector_store_serializable,
            'tag_index': tag_index_serializable,
            'links': links_serializable
        }
    
    def import_data(self, data: Dict[str, Any]) -> int:
        """
        Importiert Daten in den semantischen Speicher.
        
        Args:
            data: Dictionary mit zu importierenden Daten
            
        Returns:
            Anzahl der importierten Einträge
        """
        # Einträge importieren
        if 'entries' in data:
            self.entries.update(data['entries'])
        
        # Vektorrepräsentationen importieren
        if 'vector_store' in data:
            for entry_id, vector_list in data['vector_store'].items():
                self.vector_store[entry_id] = np.array(vector_list, dtype=np.float32)
        
        # Schlagwortindex importieren
        if 'tag_index' in data:
            for tag, ids_list in data['tag_index'].items():
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].update(ids_list)
        
        # Verknüpfungen importieren
        if 'links' in data:
            for entry_id, entry_links in data['links'].items():
                if entry_id not in self.links:
                    self.links[entry_id] = {}
                
                for link_type, linked_ids_list in entry_links.items():
                    if link_type not in self.links[entry_id]:
                        self.links[entry_id][link_type] = set()
                    self.links[entry_id][link_type].update(linked_ids_list)
        
        self.logger.info(f"{len(data.get('entries', {}))} Einträge in den semantischen Speicher importiert")
        
        return len(data.get('entries', {}))


# Wenn direkt ausgeführt, führe einen einfachen Test durch
if __name__ == "__main__":
    # Konfiguration für den Test
    test_config = {
        'vector_dim': 384
    }
    
    # Semantischen Speicher initialisieren
    semantic_store = SemanticStore(test_config)
    
    # Testdaten speichern
    test_entries = [
        {
            'content': 'Python ist eine interpretierte Programmiersprache.',
            'tags': ['python', 'programmierung', 'sprache']
        },
        {
            'content': 'TensorFlow ist ein Framework für maschinelles Lernen.',
            'tags': ['tensorflow', 'machine learning', 'framework']
        },
        {
            'content': 'PyTorch ist ein Framework für Deep Learning.',
            'tags': ['pytorch', 'deep learning', 'framework']
        }
    ]
    
    entry_ids = []
    for entry in test_entries:
        entry_id = semantic_store.store(
            content=entry['content'],
            tags=entry['tags'],
            importance=0.8
        )
        entry_ids.append(entry_id)
        print(f"Eintrag gespeichert: {entry_id}")
    
    # Einträge verknüpfen
    semantic_store.link_entries(entry_ids[0], entry_ids[1], 'related')
    semantic_store.link_entries(entry_ids[0], entry_ids[2], 'related')
    semantic_store.link_entries(entry_ids[1], entry_ids[2], 'similar')
    
    # Semantische Suche durchführen
    query = "Framework für künstliche Intelligenz"
    results = semantic_store.retrieve(query, limit=2)
    
    print(f"\nSuchergebnisse für '{query}':")
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Inhalt: {result['content']}")
        print(f"Ähnlichkeit: {result['similarity']:.4f}")
        print(f"Tags: {result['tags']}")
        print()
    
    # Suche nach Schlagwörtern
    tag_results = semantic_store.retrieve_by_tags(['framework'], limit=2)
    
    print(f"\nErgebnisse für Schlagwort 'framework':")
    for result in tag_results:
        print(f"ID: {result['id']}")
        print(f"Inhalt: {result['content']}")
        print(f"Tags: {result['tags']}")
        print()
    
    # Verknüpfte Einträge abrufen
    linked_entries = semantic_store.get_linked_entries(entry_ids[0])
    
    print(f"\nMit '{entry_ids[0]}' verknüpfte Einträge:")
    for entry in linked_entries:
        print(f"ID: {entry['id']}")
        print(f"Inhalt: {entry['content']}")
        print()
    
    # Statistiken anzeigen
    print(f"\nStatistiken:")
    print(f"Anzahl der Einträge: {semantic_store.count()}")
    print(f"Durchschnittliche Wichtigkeit: {semantic_store.avg_importance():.4f}")
    print(f"Geschätzte Größe: {semantic_store.size()} Bytes")
