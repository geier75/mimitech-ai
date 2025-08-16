#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Deep-State-Muster-Erkennung

Dieses Modul implementiert die Mustererkennung für das Deep-State-Modul.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import re
import logging
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.analysis.deep_state_patterns")

@dataclass
class ControlPattern:
    """Kontrollmuster für die Deep-State-Erkennung"""
    pattern_id: str
    pattern_name: str
    pattern_description: str
    pattern_regex: str
    pattern_weight: float
    context_tags: List[str] = field(default_factory=list)
    language_codes: List[str] = field(default_factory=lambda: ["DE", "EN"])
    
    def __post_init__(self):
        """Kompiliert den regulären Ausdruck nach der Initialisierung"""
        try:
            self.compiled_regex = re.compile(self.pattern_regex, re.IGNORECASE)
        except re.error as e:
            logger.error(f"Fehler beim Kompilieren des regulären Ausdrucks für {self.pattern_id}: {e}")
            self.compiled_regex = None
    
    def matches(self, text: str) -> bool:
        """
        Überprüft, ob das Muster im Text vorkommt
        
        Args:
            text: Zu überprüfender Text
            
        Returns:
            True, wenn das Muster im Text vorkommt, sonst False
        """
        if not self.compiled_regex:
            return False
        
        return bool(self.compiled_regex.search(text))
    
    def get_matches(self, text: str) -> List[str]:
        """
        Gibt alle Vorkommen des Musters im Text zurück
        
        Args:
            text: Zu überprüfender Text
            
        Returns:
            Liste der gefundenen Vorkommen
        """
        if not self.compiled_regex:
            return []
        
        return [match.group(0) for match in self.compiled_regex.finditer(text)]

class PatternMatcher:
    """
    Mustererkennung für das Deep-State-Modul
    
    Diese Klasse implementiert die Mustererkennung für das Deep-State-Modul.
    Sie erkennt Kontrollmuster in Texten und berechnet einen Score basierend
    auf den gefundenen Mustern.
    """
    
    def __init__(self):
        """Initialisiert den PatternMatcher"""
        self.patterns = self._load_default_patterns()
        logger.info(f"PatternMatcher mit {len(self.patterns)} Mustern initialisiert")
    
    def _load_default_patterns(self) -> List[ControlPattern]:
        """
        Lädt die Standardmuster
        
        Returns:
            Liste der Standardmuster
        """
        # In einer realen Implementierung würden diese Muster aus einer Datenbank oder Konfigurationsdatei geladen
        return [
            # Deutsch
            ControlPattern(
                pattern_id="DS_DE_001",
                pattern_name="Eliten-Narrativ",
                pattern_description="Narrative über globale Eliten, die im Verborgenen agieren",
                pattern_regex=r"(globale|geheime|verborgene)\s+(eliten|elite|mächte|macht)",
                pattern_weight=0.7,
                context_tags=["politik", "wirtschaft", "gesellschaft"],
                language_codes=["DE"]
            ),
            ControlPattern(
                pattern_id="DS_DE_002",
                pattern_name="Medien-Kontrolle",
                pattern_description="Narrative über Kontrolle der Medien durch Machtstrukturen",
                pattern_regex=r"(kontrollierten?|gesteuerten?|manipulierten?)\s+(medien|presse|berichterstattung)",
                pattern_weight=0.6,
                context_tags=["medien", "politik", "gesellschaft"],
                language_codes=["DE"]
            ),
            ControlPattern(
                pattern_id="DS_DE_003",
                pattern_name="Schattenregierung",
                pattern_description="Narrative über eine Schattenregierung oder einen tiefen Staat",
                pattern_regex=r"(schattenregierung|tiefer\s+staat|deep\s+state|strippenzieher|hintermänner)",
                pattern_weight=0.8,
                context_tags=["politik", "regierung", "verschwörung"],
                language_codes=["DE"]
            ),
            
            # Englisch
            ControlPattern(
                pattern_id="DS_EN_001",
                pattern_name="Elite Narrative",
                pattern_description="Narratives about global elites operating in secret",
                pattern_regex=r"(global|secret|hidden)\s+(elites?|powers?)",
                pattern_weight=0.7,
                context_tags=["politics", "economics", "society"],
                language_codes=["EN"]
            ),
            ControlPattern(
                pattern_id="DS_EN_002",
                pattern_name="Media Control",
                pattern_description="Narratives about control of media by power structures",
                pattern_regex=r"(controlled?|manipulated?)\s+(media|press|reporting|news)",
                pattern_weight=0.6,
                context_tags=["media", "politics", "society"],
                language_codes=["EN"]
            ),
            ControlPattern(
                pattern_id="DS_EN_003",
                pattern_name="Shadow Government",
                pattern_description="Narratives about a shadow government or deep state",
                pattern_regex=r"(shadow\s+government|deep\s+state|puppet\s+masters|handlers)",
                pattern_weight=0.8,
                context_tags=["politics", "government", "conspiracy"],
                language_codes=["EN"]
            ),
            
            # Sprachunabhängige Muster
            ControlPattern(
                pattern_id="DS_MULTI_001",
                pattern_name="QAnon-Referenzen",
                pattern_description="Referenzen auf QAnon und verwandte Konzepte",
                pattern_regex=r"(qanon|wwg1wga|the\s+storm|great\s+awakening|trust\s+the\s+plan)",
                pattern_weight=0.9,
                context_tags=["verschwörung", "politik", "gesellschaft", "conspiracy", "politics", "society"],
                language_codes=["DE", "EN"]
            ),
            ControlPattern(
                pattern_id="DS_MULTI_002",
                pattern_name="NWO-Referenzen",
                pattern_description="Referenzen auf die Neue Weltordnung",
                pattern_regex=r"(neue\s+weltordnung|new\s+world\s+order|nwo|one\s+world\s+government|eine\s+welt\s+regierung)",
                pattern_weight=0.8,
                context_tags=["verschwörung", "politik", "gesellschaft", "conspiracy", "politics", "society"],
                language_codes=["DE", "EN"]
            )
        ]
    
    def add_pattern(self, pattern: ControlPattern) -> None:
        """
        Fügt ein neues Muster hinzu
        
        Args:
            pattern: Hinzuzufügendes Muster
        """
        # Überprüfe, ob ein Muster mit derselben ID bereits existiert
        for existing_pattern in self.patterns:
            if existing_pattern.pattern_id == pattern.pattern_id:
                logger.warning(f"Muster mit ID {pattern.pattern_id} existiert bereits und wird überschrieben")
                self.patterns.remove(existing_pattern)
                break
        
        self.patterns.append(pattern)
        logger.info(f"Muster {pattern.pattern_id} hinzugefügt")
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """
        Entfernt ein Muster
        
        Args:
            pattern_id: ID des zu entfernenden Musters
            
        Returns:
            True, wenn das Muster entfernt wurde, sonst False
        """
        for pattern in self.patterns:
            if pattern.pattern_id == pattern_id:
                self.patterns.remove(pattern)
                logger.info(f"Muster {pattern_id} entfernt")
                return True
        
        logger.warning(f"Muster mit ID {pattern_id} nicht gefunden")
        return False
    
    def match_patterns(self, text: str, context_cluster: str = "") -> float:
        """
        Gleicht den Text mit den Kontrollmustern ab
        
        Args:
            text: Zu überprüfender Text
            context_cluster: Kontext-Cluster für die Filterung der Muster
            
        Returns:
            Score zwischen 0 und 1 basierend auf den gefundenen Mustern
        """
        # Filtere Muster basierend auf dem Kontext-Cluster
        relevant_patterns = self._filter_patterns_by_context(context_cluster)
        
        if not relevant_patterns:
            logger.warning(f"Keine relevanten Muster für Kontext-Cluster {context_cluster} gefunden")
            return 0.0
        
        # Überprüfe jedes Muster
        matched_patterns = []
        total_weight = 0.0
        matched_weight = 0.0
        
        for pattern in relevant_patterns:
            total_weight += pattern.pattern_weight
            
            if pattern.matches(text):
                matched_patterns.append(pattern)
                matched_weight += pattern.pattern_weight
        
        # Berechne Score
        if total_weight == 0:
            return 0.0
        
        score = matched_weight / total_weight
        
        # Protokolliere Ergebnis
        if matched_patterns:
            logger.info(f"Gefundene Muster: {[p.pattern_id for p in matched_patterns]}, Score: {score:.2f}")
        else:
            logger.info(f"Keine Muster gefunden, Score: {score:.2f}")
        
        return score
    
    def _filter_patterns_by_context(self, context_cluster: str) -> List[ControlPattern]:
        """
        Filtert Muster basierend auf dem Kontext-Cluster
        
        Args:
            context_cluster: Kontext-Cluster für die Filterung der Muster
            
        Returns:
            Gefilterte Liste von Mustern
        """
        if not context_cluster:
            return self.patterns
        
        # Extrahiere Tags aus dem Kontext-Cluster
        context_tags = [tag.strip().lower() for tag in context_cluster.split(",")]
        
        # Filtere Muster basierend auf den Tags
        filtered_patterns = []
        
        for pattern in self.patterns:
            pattern_tags = [tag.strip().lower() for tag in pattern.context_tags]
            
            # Überprüfe, ob mindestens ein Tag übereinstimmt
            if any(tag in pattern_tags for tag in context_tags):
                filtered_patterns.append(pattern)
        
        # Wenn keine Muster gefunden wurden, verwende alle Muster
        if not filtered_patterns:
            return self.patterns
        
        return filtered_patterns
    
    def get_matched_patterns(self, text: str, context_cluster: str = "") -> List[str]:
        """
        Gibt eine Liste der gefundenen Muster zurück
        
        Args:
            text: Zu überprüfender Text
            context_cluster: Kontext-Cluster für die Filterung der Muster
            
        Returns:
            Liste der gefundenen Muster als Strings
        """
        # Filtere Muster basierend auf dem Kontext-Cluster
        relevant_patterns = self._filter_patterns_by_context(context_cluster)
        
        # Überprüfe jedes Muster
        matched_descriptions = []
        
        for pattern in relevant_patterns:
            if pattern.matches(text):
                matches = pattern.get_matches(text)
                matched_descriptions.append(f"{pattern.pattern_name}: {', '.join(matches)}")
        
        return matched_descriptions
    
    def get_pattern_count(self) -> int:
        """
        Gibt die Anzahl der Muster zurück
        
        Returns:
            Anzahl der Muster
        """
        return len(self.patterns)
    
    def get_patterns_by_language(self, language_code: str) -> List[ControlPattern]:
        """
        Gibt alle Muster für eine bestimmte Sprache zurück
        
        Args:
            language_code: Sprachcode (z.B. "DE", "EN")
            
        Returns:
            Liste der Muster für die angegebene Sprache
        """
        return [p for p in self.patterns if language_code in p.language_codes]
    
    def export_patterns(self) -> Dict[str, Any]:
        """
        Exportiert alle Muster als Wörterbuch
        
        Returns:
            Wörterbuch mit allen Mustern
        """
        return {
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_name": p.pattern_name,
                    "pattern_description": p.pattern_description,
                    "pattern_regex": p.pattern_regex,
                    "pattern_weight": p.pattern_weight,
                    "context_tags": p.context_tags,
                    "language_codes": p.language_codes
                }
                for p in self.patterns
            ]
        }
    
    def import_patterns(self, patterns_data: Dict[str, Any]) -> None:
        """
        Importiert Muster aus einem Wörterbuch
        
        Args:
            patterns_data: Wörterbuch mit Mustern
        """
        if "patterns" not in patterns_data:
            logger.error("Ungültiges Muster-Datenformat: 'patterns' nicht gefunden")
            return
        
        new_patterns = []
        
        for pattern_data in patterns_data["patterns"]:
            try:
                pattern = ControlPattern(
                    pattern_id=pattern_data["pattern_id"],
                    pattern_name=pattern_data["pattern_name"],
                    pattern_description=pattern_data["pattern_description"],
                    pattern_regex=pattern_data["pattern_regex"],
                    pattern_weight=pattern_data["pattern_weight"],
                    context_tags=pattern_data.get("context_tags", []),
                    language_codes=pattern_data.get("language_codes", ["DE", "EN"])
                )
                new_patterns.append(pattern)
            except KeyError as e:
                logger.error(f"Fehler beim Importieren des Musters: Fehlendes Feld {e}")
        
        # Ersetze alle Muster
        self.patterns = new_patterns
        logger.info(f"{len(new_patterns)} Muster importiert")
