#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MEMEX: Working Memory Module
-------------------------------
Arbeitsgedächtnissystem für das VXOR-Gedächtnismodul.
Verantwortlich für die temporäre Speicherung von kontextabhängigen Informationen
mit kurzer Lebensdauer und schnellem Zugriff.

Optimiert für Apple Silicon M4 Max.
"""

import json
import time
import uuid
import logging
import heapq
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
DEFAULT_TTL = 60 * 60  # 1 Stunde in Sekunden (kurze Standarddauer für Arbeitsgedächtnis)
DEFAULT_IMPORTANCE = 0.5  # Standardwert für Wichtigkeit (0-1)
MAX_ENTRIES = 100  # Maximale Anzahl von Einträgen im Arbeitsgedächtnis

class WorkingMemory:
    """
    Arbeitsgedächtnissystem für das VX-MEMEX Gedächtnismodul.
    
    Verantwortlich für:
    - Temporäre Speicherung von kontextabhängigen Informationen
    - Schneller Zugriff auf aktuelle Informationen
    - Automatisches Vergessen von nicht mehr benötigten Informationen
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert das Arbeitsgedächtnissystem.
        
        Args:
            config: Optionale Konfigurationsparameter
        """
        self.config = config or {}
        
        # Logger initialisieren
        self._setup_logging()
        
        # Vektordimension festlegen
        self.vector_dim = self.config.get('vector_dim', DEFAULT_VECTOR_DIM)
        
        # TTL-Standardwert festlegen
        self.ttl_default = self.config.get('ttl_default', DEFAULT_TTL)
        
        # Maximale Anzahl von Einträgen festlegen
        self.max_entries = self.config.get('max_entries', MAX_ENTRIES)
        
        # Speicherstrukturen initialisieren
        self.entries = {}  # Hauptspeicher für Einträge
        self.tag_index = {}  # Index für Schlagwörter
        self.vector_store = {}  # Speicher für Vektorrepräsentationen
        self.links = {}  # Speicher für Verknüpfungen zwischen Einträgen
        
        # Prioritätsbasierte Indizes
        self.priority_index = []  # Prioritätswarteschlange für Einträge (Wichtigkeit, Zeitstempel, ID)
        
        # Aktivierungszähler für Einträge
        self.activation_count = {}  # Zähler für Zugriffe auf Einträge
        
        # Texteinbettungsmodell initialisieren (Mock-Implementierung)
        self._init_embedding_model()
        
        self.logger.info(f"Arbeitsgedächtnis initialisiert (Vektordimension: {self.vector_dim}, TTL: {self.ttl_default}s, Max. Einträge: {self.max_entries})")
    
    def _setup_logging(self):
        """Konfiguriert das Logging für das Arbeitsgedächtnissystem."""
        try:
            # Versuche, den VXOR-Logger zu importieren, falls verfügbar
            from vxor_logger import get_logger
            self.logger = get_logger('VX-MEMEX.working')
        except ImportError:
            # Fallback auf Standard-Logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger('VX-MEMEX.working')
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
    
    def _calculate_priority(self, importance: float, timestamp: float, activation: int) -> float:
        """
        Berechnet die Priorität eines Eintrags basierend auf Wichtigkeit, Alter und Aktivierung.
        
        Args:
            importance: Wichtigkeit des Eintrags (0-1)
            timestamp: Zeitstempel des Eintrags
            activation: Aktivierungszähler des Eintrags
            
        Returns:
            Prioritätswert (höher = wichtiger)
        """
        # Alter in Sekunden
        age = time.time() - timestamp
        
        # Alter normalisieren (0-1, wobei 0 = neu, 1 = alt)
        normalized_age = min(1.0, age / self.ttl_default)
        
        # Aktivierung normalisieren (0-1, wobei 0 = nie aktiviert, 1 = häufig aktiviert)
        max_activation = 10  # Annahme: 10 Aktivierungen sind "viel"
        normalized_activation = min(1.0, activation / max_activation)
        
        # Priorität berechnen (gewichtete Summe)
        # - Wichtigkeit: 50%
        # - Aktivierung: 30%
        # - Neuheit: 20%
        priority = (
            importance * 0.5 +
            normalized_activation * 0.3 +
            (1.0 - normalized_age) * 0.2
        )
        
        return priority
    
    def _update_priority_index(self, entry_id: str) -> None:
        """
        Aktualisiert den Prioritätsindex für einen Eintrag.
        
        Args:
            entry_id: ID des Eintrags
        """
        # Prüfen, ob der Eintrag existiert
        if entry_id not in self.entries:
            return
        
        entry = self.entries[entry_id]
        
        # Aktivierungszähler abrufen
        activation = self.activation_count.get(entry_id, 0)
        
        # Priorität berechnen
        priority = self._calculate_priority(
            entry['importance'],
            entry['created_at'],
            activation
        )
        
        # Alten Eintrag aus dem Prioritätsindex entfernen
        self.priority_index = [(p, t, eid) for p, t, eid in self.priority_index if eid != entry_id]
        
        # Neuen Eintrag in den Prioritätsindex einfügen
        # Negatives Prioritätswert für Min-Heap (höchste Priorität zuerst)
        heapq.heappush(self.priority_index, (-priority, entry['created_at'], entry_id))
    
    def _increment_activation(self, entry_id: str) -> None:
        """
        Erhöht den Aktivierungszähler für einen Eintrag.
        
        Args:
            entry_id: ID des Eintrags
        """
        # Aktivierungszähler erhöhen
        self.activation_count[entry_id] = self.activation_count.get(entry_id, 0) + 1
        
        # Prioritätsindex aktualisieren
        self._update_priority_index(entry_id)
    
    def store(self, 
             content: Any, 
             entry_id: Optional[str] = None,
             tags: List[str] = None,
             ttl: Optional[int] = None,
             importance: float = DEFAULT_IMPORTANCE,
             context: Dict[str, Any] = None) -> str:
        """
        Speichert einen Eintrag im Arbeitsgedächtnis.
        
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
        ttl = ttl or self.ttl_default
        
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
        
        # Aktivierungszähler initialisieren
        self.activation_count[entry_id] = 0
        
        # In Prioritätsindex einfügen
        self._update_priority_index(entry_id)
        
        # Verknüpfungsstruktur initialisieren
        self.links[entry_id] = {}
        
        # Speicherlimit prüfen und Einträge mit niedrigster Priorität entfernen, wenn nötig
        self._manage_storage()
        
        self.logger.debug(f"Eintrag im Arbeitsgedächtnis gespeichert: {entry_id}")
        
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
            # Aktivierungszähler erhöhen
            self._increment_activation(entry_id)
            
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
        
        # Aktivierungszähler erhöhen
        self._increment_activation(entry_id)
        
        return entry.copy()
    
    def update(self, 
              entry_id: str, 
              content: Any = None,
              tags: List[str] = None,
              ttl: Optional[int] = None,
              importance: Optional[float] = None,
              context: Dict[str, Any] = None) -> bool:
        """
        Aktualisiert einen Eintrag im Arbeitsgedächtnis.
        
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
            
            # Prioritätsindex aktualisieren
            self._update_priority_index(entry_id)
        
        # Kontext aktualisieren, falls angegeben
        if context is not None:
            entry['context'] = context
        
        # Aktualisierungszeitstempel setzen
        entry['updated_at'] = current_time
        
        # Aktivierungszähler erhöhen
        self._increment_activation(entry_id)
        
        self.logger.debug(f"Eintrag im Arbeitsgedächtnis aktualisiert: {entry_id}")
        
        return True
    
    def delete(self, entry_id: str) -> bool:
        """
        Löscht einen Eintrag aus dem Arbeitsgedächtnis.
        
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
        
        # Aktivierungszähler entfernen
        if entry_id in self.activation_count:
            del self.activation_count[entry_id]
        
        # Aus Prioritätsindex entfernen
        self.priority_index = [(p, t, eid) for p, t, eid in self.priority_index if eid != entry_id]
        heapq.heapify(self.priority_index)
        
        # Eintrag entfernen
        del self.entries[entry_id]
        
        self.logger.debug(f"Eintrag aus dem Arbeitsgedächtnis gelöscht: {entry_id}")
        
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
            
            # Aktivierungszähler erhöhen
            self._increment_activation(entry_id)
            
            valid_entries.append(entry.copy())
        
        # Nach Priorität sortieren
        valid_entries.sort(key=lambda x: self._calculate_priority(
            x['importance'],
            x['created_at'],
            self.activation_count.get(x['id'], 0)
        ), reverse=True)
        
        # Auf die gewünschte Anzahl begrenzen
        return valid_entries[:limit]
    
    def get_most_important(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ruft die wichtigsten Einträge ab.
        
        Args:
            limit: Maximale Anzahl von Ergebnissen
            
        Returns:
            Liste mit den wichtigsten Einträgen
        """
        # Prioritätsindex kopieren
        priority_index_copy = self.priority_index.copy()
        
        # Wichtigste Einträge abrufen
        important_entries = []
        current_time = time.time()
        
        while priority_index_copy and len(important_entries) < limit:
            # Eintrag mit höchster Priorität abrufen
            _, _, entry_id = heapq.heappop(priority_index_copy)
            
            # Prüfen, ob der Eintrag existiert
            if entry_id not in self.entries:
                continue
            
            entry = self.entries[entry_id]
            
            # Abgelaufene Einträge überspringen
            if entry['expiry_time'] is not None and entry['expiry_time'] < current_time:
                continue
            
            # Aktivierungszähler erhöhen
            self._increment_activation(entry_id)
            
            important_entries.append(entry.copy())
        
        return important_entries
    
    def get_least_important(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ruft die am wenigsten wichtigen Einträge ab.
        
        Args:
            limit: Maximale Anzahl von Ergebnissen
            
        Returns:
            Liste mit den am wenigsten wichtigen Einträgen
        """
        # Prioritätsindex kopieren und umkehren
        priority_index_copy = [(-p, t, eid) for p, t, eid in self.priority_index]
        heapq.heapify(priority_index_copy)
        
        # Am wenigsten wichtige Einträge abrufen
        least_important_entries = []
        current_time = time.time()
        
        while priority_index_copy and len(least_important_entries) < limit:
            # Eintrag mit niedrigster Priorität abrufen
            _, _, entry_id = heapq.heappop(priority_index_copy)
            
            # Prüfen, ob der Eintrag existiert
            if entry_id not in self.entries:
                continue
            
            entry = self.entries[entry_id]
            
            # Abgelaufene Einträge überspringen
            if entry['expiry_time'] is not None and entry['expiry_time'] < current_time:
                continue
            
            least_important_entries.append(entry.copy())
        
        return least_important_entries
    
    def cleanup(self) -> int:
        """
        Bereinigt abgelaufene Einträge im Arbeitsgedächtnis.
        
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
        
        self.logger.info(f"{len(expired_ids)} abgelaufene Einträge aus dem Arbeitsgedächtnis bereinigt")
        
        return len(expired_ids)
    
    def _manage_storage(self) -> None:
        """
        Verwaltet den Speicherplatz, indem Einträge mit niedrigster Priorität entfernt werden,
        wenn das Limit erreicht ist.
        """
        # Prüfen, ob das Limit erreicht ist
        if len(self.entries) <= self.max_entries:
            return
        
        # Berechnen, wie viele Einträge entfernt werden müssen
        to_remove = len(self.entries) - self.max_entries
        
        # Einträge mit niedrigster Priorität abrufen
        least_important = self.get_least_important(to_remove)
        
        # Einträge entfernen
        for entry in least_important:
            self.delete(entry['id'])
            self.logger.debug(f"Eintrag {entry['id']} aus dem Arbeitsgedächtnis entfernt (Speicherlimit erreicht)")
    
    def refresh(self, entry_id: str, ttl: Optional[int] = None) -> bool:
        """
        Aktualisiert die Ablaufzeit eines Eintrags.
        
        Args:
            entry_id: ID des zu aktualisierenden Eintrags
            ttl: Neue Time-to-Live in Sekunden (None = Standardwert verwenden)
            
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
        
        # TTL setzen
        ttl = ttl or self.ttl_default
        
        # Ablaufzeit aktualisieren
        entry['expiry_time'] = current_time + ttl
        
        # Aktualisierungszeitstempel setzen
        entry['updated_at'] = current_time
        
        # Aktivierungszähler erhöhen
        self._increment_activation(entry_id)
        
        self.logger.debug(f"Eintrag im Arbeitsgedächtnis aufgefrischt: {entry_id}")
        
        return True
    
    def count(self) -> int:
        """
        Gibt die Anzahl der Einträge im Arbeitsgedächtnis zurück.
        
        Returns:
            Anzahl der Einträge
        """
        return len(self.entries)
    
    def size(self) -> int:
        """
        Gibt die Größe des Arbeitsgedächtnisses in Bytes zurück (Schätzung).
        
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
        
        # Größe des Prioritätsindex schätzen
        priority_index_size = sys.getsizeof(self.priority_index)
        for item in self.priority_index:
            priority_index_size += sys.getsizeof(item)
        
        # Größe der Aktivierungszähler schätzen
        activation_size = sys.getsizeof(self.activation_count)
        for entry_id, count in self.activation_count.items():
            activation_size += sys.getsizeof(entry_id)
            activation_size += sys.getsizeof(count)
        
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
        
        return entries_size + vector_size + tag_index_size + priority_index_size + activation_size + links_size
    
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
        
        # Aktivierungszähler für beide Einträge erhöhen
        self._increment_activation(source_id)
        self._increment_activation(target_id)
        
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
        
        # Aktivierungszähler erhöhen
        self._increment_activation(entry_id)
        
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
            
            # Aktivierungszähler erhöhen
            self._increment_activation(linked_id)
            
            linked_entries.append(entry.copy())
        
        # Nach Priorität sortieren
        linked_entries.sort(key=lambda x: self._calculate_priority(
            x['importance'],
            x['created_at'],
            self.activation_count.get(x['id'], 0)
        ), reverse=True)
        
        return linked_entries
    
    def export_data(self) -> Dict[str, Any]:
        """
        Exportiert alle Daten des Arbeitsgedächtnisses.
        
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
        
        # Prioritätsindex in Liste umwandeln
        priority_index_serializable = [(float(p), t, eid) for p, t, eid in self.priority_index]
        
        return {
            'entries': self.entries,
            'vector_store': vector_store_serializable,
            'tag_index': tag_index_serializable,
            'links': links_serializable,
            'priority_index': priority_index_serializable,
            'activation_count': self.activation_count
        }
    
    def import_data(self, data: Dict[str, Any]) -> int:
        """
        Importiert Daten in das Arbeitsgedächtnis.
        
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
        
        # Prioritätsindex importieren
        if 'priority_index' in data:
            self.priority_index = [(float(p), t, eid) for p, t, eid in data['priority_index']]
            heapq.heapify(self.priority_index)
        
        # Aktivierungszähler importieren
        if 'activation_count' in data:
            self.activation_count.update(data['activation_count'])
        
        self.logger.info(f"{len(data.get('entries', {}))} Einträge in das Arbeitsgedächtnis importiert")
        
        return len(data.get('entries', {}))


# Wenn direkt ausgeführt, führe einen einfachen Test durch
if __name__ == "__main__":
    # Konfiguration für den Test
    test_config = {
        'vector_dim': 384,
        'ttl_default': 3600,  # 1 Stunde
        'max_entries': 100
    }
    
    # Arbeitsgedächtnis initialisieren
    working_memory = WorkingMemory(test_config)
    
    # Testdaten speichern
    test_entries = [
        {
            'content': 'Aktuelle Aufgabe: Implementierung des Arbeitsgedächtnismoduls',
            'tags': ['aufgabe', 'implementierung', 'aktuell'],
            'importance': 0.9
        },
        {
            'content': 'Kontext: Entwicklung des VX-MEMEX Gedächtnismoduls für VXOR',
            'tags': ['kontext', 'vx-memex', 'vxor'],
            'importance': 0.8
        },
        {
            'content': 'Notiz: Optimierung für Apple Silicon M4 Max',
            'tags': ['notiz', 'optimierung', 'apple'],
            'importance': 0.6
        }
    ]
    
    entry_ids = []
    for entry in test_entries:
        entry_id = working_memory.store(
            content=entry['content'],
            tags=entry['tags'],
            importance=entry['importance']
        )
        entry_ids.append(entry_id)
        print(f"Eintrag gespeichert: {entry_id}")
    
    # Einträge verknüpfen
    working_memory.link_entries(entry_ids[0], entry_ids[1], 'related')
    working_memory.link_entries(entry_ids[1], entry_ids[2], 'related')
    
    # Mehrfacher Zugriff auf einen Eintrag, um Aktivierung zu erhöhen
    for _ in range(5):
        working_memory.retrieve_by_id(entry_ids[0])
    
    # Semantische Suche durchführen
    query = "Aktuelle Aufgaben und Kontext"
    results = working_memory.retrieve(query, limit=2)
    
    print(f"\nSuchergebnisse für '{query}':")
    for result in results:
        print(f"ID: {result['id']}")
        print(f"Inhalt: {result['content']}")
        print(f"Ähnlichkeit: {result['similarity']:.4f}")
        print(f"Tags: {result['tags']}")
        print()
    
    # Wichtigste Einträge abrufen
    important_entries = working_memory.get_most_important(limit=2)
    
    print(f"\nWichtigste Einträge:")
    for entry in important_entries:
        print(f"ID: {entry['id']}")
        print(f"Inhalt: {entry['content']}")
        print(f"Wichtigkeit: {entry['importance']:.4f}")
        print()
    
    # Verknüpfte Einträge abrufen
    linked_entries = working_memory.get_linked_entries(entry_ids[0])
    
    print(f"\nMit '{entry_ids[0]}' verknüpfte Einträge:")
    for entry in linked_entries:
        print(f"ID: {entry['id']}")
        print(f"Inhalt: {entry['content']}")
        print()
    
    # Statistiken anzeigen
    print(f"\nStatistiken:")
    print(f"Anzahl der Einträge: {working_memory.count()}")
    print(f"Durchschnittliche Wichtigkeit: {working_memory.avg_importance():.4f}")
    print(f"Geschätzte Größe: {working_memory.size()} Bytes")
