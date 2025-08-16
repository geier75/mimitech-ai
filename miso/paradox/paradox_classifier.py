#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Paradox Classifier

Diese Komponente erweitert die bestehende ParadoxType-Enumeration zu einer
vollständigen Klassifizierungsklasse für detaillierte Paradoxklassifizierung.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Optional, Set, Any
from enum import Enum, auto
from dataclasses import dataclass

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [PARADOX-CLASSIFIER] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.Paradox.Classifier")

# Importiere die benötigten Module
try:
    from miso.timeline.temporal_integrity_guard import ParadoxType
    from miso.paradox.enhanced_paradox_detector import (
        EnhancedParadoxType, ParadoxSeverity, ParadoxInstance
    )
    logger.info("Module erfolgreich importiert")
except ImportError as e:
    logger.error(f"Fehler beim Importieren der Module: {e}")
    sys.exit(1)

@dataclass
class ParadoxHierarchy:
    """Hierarchische Klassifizierung eines Paradoxes"""
    primary_type: EnhancedParadoxType
    secondary_type: Optional[EnhancedParadoxType]
    tertiary_type: Optional[EnhancedParadoxType]
    category: str
    subcategory: Optional[str]
    tags: List[str]

class ParadoxClassifier:
    """
    Paradoxklassifizierer
    
    Diese Komponente erweitert die bestehende ParadoxType-Enumeration zu einer
    vollständigen Klassifizierungsklasse für detaillierte Paradoxklassifizierung.
    """
    
    def __init__(self):
        """Initialisiert den Paradoxklassifizierer"""
        # Definiere Kategorien für Paradoxien
        self.categories = {
            EnhancedParadoxType.GRANDFATHER: "Kausal-Eliminierend",
            EnhancedParadoxType.BOOTSTRAP: "Kausal-Zirkulär",
            EnhancedParadoxType.PREDESTINATION: "Kausal-Determinierend",
            EnhancedParadoxType.ONTOLOGICAL: "Kausal-Existentiell",
            EnhancedParadoxType.TEMPORAL_LOOP: "Temporal-Zyklisch",
            EnhancedParadoxType.CAUSAL_VIOLATION: "Kausal-Inkonsistent",
            EnhancedParadoxType.INFORMATION_PARADOX: "Informations-Inkonsistent",
            EnhancedParadoxType.QUANTUM_PARADOX: "Quanten-Inkonsistent",
            EnhancedParadoxType.MULTI_TIMELINE_PARADOX: "Multitemporal-Inkonsistent",
            EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION: "Selbst-Inkonsistent"
        }
        
        # Definiere Subkategorien für Paradoxien
        self.subcategories = {
            EnhancedParadoxType.GRANDFATHER: ["Eliminierend", "Verhindernd", "Modifizierend"],
            EnhancedParadoxType.BOOTSTRAP: ["Informations-Zirkulär", "Objekt-Zirkulär", "Kausal-Zirkulär"],
            EnhancedParadoxType.PREDESTINATION: ["Selbst-Erfüllend", "Unvermeidbar", "Determiniert"],
            EnhancedParadoxType.ONTOLOGICAL: ["Existenz-Widerspruch", "Ursprungs-Widerspruch", "Identitäts-Widerspruch"],
            EnhancedParadoxType.TEMPORAL_LOOP: ["Endlos-Schleife", "Begrenzte Schleife", "Verzweigende Schleife"],
            EnhancedParadoxType.CAUSAL_VIOLATION: ["Ursache-Wirkung-Umkehrung", "Kausale Lücke", "Kausale Überlappung"],
            EnhancedParadoxType.INFORMATION_PARADOX: ["Informationsverlust", "Informationsgewinn", "Informationstransformation"],
            EnhancedParadoxType.QUANTUM_PARADOX: ["Beobachter-Effekt", "Quantenverschränkung", "Superpositions-Kollaps"],
            EnhancedParadoxType.MULTI_TIMELINE_PARADOX: ["Zeitlinien-Konvergenz", "Zeitlinien-Divergenz", "Zeitlinien-Überlappung"],
            EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION: ["Logische Inkonsistenz", "Physikalische Inkonsistenz", "Temporale Inkonsistenz"]
        }
        
        # Definiere Tags für Paradoxien
        self.tags = {
            EnhancedParadoxType.GRANDFATHER: ["kausal", "eliminierend", "vergangenheits-ändernd"],
            EnhancedParadoxType.BOOTSTRAP: ["zirkulär", "selbst-erzeugend", "ursprungslos"],
            EnhancedParadoxType.PREDESTINATION: ["determiniert", "unvermeidbar", "vorbestimmt"],
            EnhancedParadoxType.ONTOLOGICAL: ["existentiell", "ursprungs-paradox", "identitäts-paradox"],
            EnhancedParadoxType.TEMPORAL_LOOP: ["schleife", "zyklisch", "wiederkehrend"],
            EnhancedParadoxType.CAUSAL_VIOLATION: ["kausal-bruch", "ursache-wirkung-umkehrung", "kausal-inkonsistent"],
            EnhancedParadoxType.INFORMATION_PARADOX: ["information", "wissen", "daten-inkonsistenz"],
            EnhancedParadoxType.QUANTUM_PARADOX: ["quanten", "beobachter-effekt", "superposition"],
            EnhancedParadoxType.MULTI_TIMELINE_PARADOX: ["multi-zeitlinie", "parallel", "alternativ"],
            EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION: ["selbst-widerspruch", "inkonsistenz", "logik-bruch"]
        }
        
        # Definiere Beziehungen zwischen Paradoxtypen
        self.related_types = {
            EnhancedParadoxType.GRANDFATHER: [EnhancedParadoxType.CAUSAL_VIOLATION, EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION],
            EnhancedParadoxType.BOOTSTRAP: [EnhancedParadoxType.TEMPORAL_LOOP, EnhancedParadoxType.INFORMATION_PARADOX],
            EnhancedParadoxType.PREDESTINATION: [EnhancedParadoxType.BOOTSTRAP, EnhancedParadoxType.TEMPORAL_LOOP],
            EnhancedParadoxType.ONTOLOGICAL: [EnhancedParadoxType.BOOTSTRAP, EnhancedParadoxType.INFORMATION_PARADOX],
            EnhancedParadoxType.TEMPORAL_LOOP: [EnhancedParadoxType.BOOTSTRAP, EnhancedParadoxType.CAUSAL_VIOLATION],
            EnhancedParadoxType.CAUSAL_VIOLATION: [EnhancedParadoxType.GRANDFATHER, EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION],
            EnhancedParadoxType.INFORMATION_PARADOX: [EnhancedParadoxType.BOOTSTRAP, EnhancedParadoxType.QUANTUM_PARADOX],
            EnhancedParadoxType.QUANTUM_PARADOX: [EnhancedParadoxType.INFORMATION_PARADOX, EnhancedParadoxType.MULTI_TIMELINE_PARADOX],
            EnhancedParadoxType.MULTI_TIMELINE_PARADOX: [EnhancedParadoxType.QUANTUM_PARADOX, EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION],
            EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION: [EnhancedParadoxType.GRANDFATHER, EnhancedParadoxType.CAUSAL_VIOLATION]
        }
        
        # Definiere Schweregrade für Paradoxtypen (Standardwerte)
        self.default_severity = {
            EnhancedParadoxType.GRANDFATHER: ParadoxSeverity.CRITICAL,
            EnhancedParadoxType.BOOTSTRAP: ParadoxSeverity.MAJOR,
            EnhancedParadoxType.PREDESTINATION: ParadoxSeverity.MODERATE,
            EnhancedParadoxType.ONTOLOGICAL: ParadoxSeverity.MAJOR,
            EnhancedParadoxType.TEMPORAL_LOOP: ParadoxSeverity.MODERATE,
            EnhancedParadoxType.CAUSAL_VIOLATION: ParadoxSeverity.MAJOR,
            EnhancedParadoxType.INFORMATION_PARADOX: ParadoxSeverity.MODERATE,
            EnhancedParadoxType.QUANTUM_PARADOX: ParadoxSeverity.MINOR,
            EnhancedParadoxType.MULTI_TIMELINE_PARADOX: ParadoxSeverity.MAJOR,
            EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION: ParadoxSeverity.CRITICAL
        }
        
        logger.info("ParadoxClassifier initialisiert")
    
    def classify_paradox(self, paradox_instance: ParadoxInstance) -> EnhancedParadoxType:
        """
        Klassifiziert ein Paradox detailliert
        
        Args:
            paradox_instance: Instanz des zu klassifizierenden Paradoxes
            
        Returns:
            Detaillierter Paradoxtyp
        """
        # Wenn der Typ bereits gesetzt ist, verwende diesen
        if paradox_instance.type is not None:
            return paradox_instance.type
        
        # Analysiere die Eigenschaften des Paradoxes
        # Implementiere hier einen Klassifizierungsalgorithmus
        
        # Einfache Implementierung als Platzhalter
        if "loop" in paradox_instance.description.lower():
            return EnhancedParadoxType.TEMPORAL_LOOP
        elif "information" in paradox_instance.description.lower():
            return EnhancedParadoxType.INFORMATION_PARADOX
        elif "quantum" in paradox_instance.description.lower():
            return EnhancedParadoxType.QUANTUM_PARADOX
        elif "timeline" in paradox_instance.description.lower():
            return EnhancedParadoxType.MULTI_TIMELINE_PARADOX
        elif "causal" in paradox_instance.description.lower():
            return EnhancedParadoxType.CAUSAL_VIOLATION
        elif "consistency" in paradox_instance.description.lower():
            return EnhancedParadoxType.SELF_CONSISTENCY_VIOLATION
        elif "grandfather" in paradox_instance.description.lower():
            return EnhancedParadoxType.GRANDFATHER
        elif "bootstrap" in paradox_instance.description.lower():
            return EnhancedParadoxType.BOOTSTRAP
        elif "predestination" in paradox_instance.description.lower():
            return EnhancedParadoxType.PREDESTINATION
        elif "ontological" in paradox_instance.description.lower():
            return EnhancedParadoxType.ONTOLOGICAL
        else:
            # Standardmäßig als Kausalitätsverletzung klassifizieren
            return EnhancedParadoxType.CAUSAL_VIOLATION
    
    def evaluate_severity(self, paradox_instance: ParadoxInstance) -> ParadoxSeverity:
        """
        Bewertet den Schweregrad eines Paradoxes
        
        Args:
            paradox_instance: Instanz des zu bewertenden Paradoxes
            
        Returns:
            Schweregrad des Paradoxes
        """
        # Wenn der Schweregrad bereits gesetzt ist, verwende diesen
        if paradox_instance.severity is not None:
            return paradox_instance.severity
        
        # Bestimme den Typ des Paradoxes
        paradox_type = self.classify_paradox(paradox_instance)
        
        # Faktoren, die den Schweregrad beeinflussen
        factors = {
            "affected_nodes": len(paradox_instance.affected_nodes),
            "probability": paradox_instance.probability,
            "causal_chain": len(paradox_instance.causal_chain)
        }
        
        # Standardschweregrad für den Paradoxtyp
        base_severity = self.default_severity[paradox_type]
        
        # Anpassung des Schweregrades basierend auf Faktoren
        if factors["affected_nodes"] > 10:
            # Erhöhe den Schweregrad um eine Stufe
            severity_values = list(ParadoxSeverity)
            current_index = severity_values.index(base_severity)
            if current_index < len(severity_values) - 1:
                return severity_values[current_index + 1]
        
        if factors["probability"] < 0.5:
            # Verringere den Schweregrad um eine Stufe
            severity_values = list(ParadoxSeverity)
            current_index = severity_values.index(base_severity)
            if current_index > 0:
                return severity_values[current_index - 1]
        
        # Standardmäßig den Basisschweregrad zurückgeben
        return base_severity
    
    def get_hierarchical_classification(self, paradox_instance: ParadoxInstance) -> ParadoxHierarchy:
        """
        Gibt eine hierarchische Klassifizierung eines Paradoxes zurück
        
        Args:
            paradox_instance: Instanz des zu klassifizierenden Paradoxes
            
        Returns:
            Hierarchische Klassifizierung des Paradoxes
        """
        # Bestimme den primären Typ des Paradoxes
        primary_type = self.classify_paradox(paradox_instance)
        
        # Bestimme verwandte Typen
        related_types = self.related_types.get(primary_type, [])
        secondary_type = related_types[0] if related_types else None
        tertiary_type = related_types[1] if len(related_types) > 1 else None
        
        # Bestimme Kategorie und Subkategorie
        category = self.categories.get(primary_type, "Unbekannt")
        
        # Wähle eine Subkategorie basierend auf der Beschreibung und den Eigenschaften
        subcategories = self.subcategories.get(primary_type, [])
        subcategory = subcategories[0] if subcategories else None
        
        # Bestimme Tags
        tags = self.tags.get(primary_type, [])
        
        # Erstelle die hierarchische Klassifizierung
        hierarchy = ParadoxHierarchy(
            primary_type=primary_type,
            secondary_type=secondary_type,
            tertiary_type=tertiary_type,
            category=category,
            subcategory=subcategory,
            tags=tags
        )
        
        return hierarchy
    
    def get_related_paradoxes(self, paradox_type: EnhancedParadoxType) -> List[EnhancedParadoxType]:
        """
        Gibt verwandte Paradoxtypen zurück
        
        Args:
            paradox_type: Paradoxtyp, für den verwandte Typen gesucht werden
            
        Returns:
            Liste verwandter Paradoxtypen
        """
        return self.related_types.get(paradox_type, [])
    
    def get_category(self, paradox_type: EnhancedParadoxType) -> str:
        """
        Gibt die Kategorie eines Paradoxtyps zurück
        
        Args:
            paradox_type: Paradoxtyp, für den die Kategorie gesucht wird
            
        Returns:
            Kategorie des Paradoxtyps
        """
        return self.categories.get(paradox_type, "Unbekannt")
    
    def get_subcategories(self, paradox_type: EnhancedParadoxType) -> List[str]:
        """
        Gibt die Subkategorien eines Paradoxtyps zurück
        
        Args:
            paradox_type: Paradoxtyp, für den die Subkategorien gesucht werden
            
        Returns:
            Liste von Subkategorien des Paradoxtyps
        """
        return self.subcategories.get(paradox_type, [])
    
    def get_tags(self, paradox_type: EnhancedParadoxType) -> List[str]:
        """
        Gibt die Tags eines Paradoxtyps zurück
        
        Args:
            paradox_type: Paradoxtyp, für den die Tags gesucht werden
            
        Returns:
            Liste von Tags des Paradoxtyps
        """
        return self.tags.get(paradox_type, [])
    
    def get_default_severity(self, paradox_type: EnhancedParadoxType) -> ParadoxSeverity:
        """
        Gibt den Standardschweregrad eines Paradoxtyps zurück
        
        Args:
            paradox_type: Paradoxtyp, für den der Standardschweregrad gesucht wird
            
        Returns:
            Standardschweregrad des Paradoxtyps
        """
        return self.default_severity.get(paradox_type, ParadoxSeverity.MODERATE)
