#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK Conflict Resolver

Spezialisiertes Modul zur Konfliktlösung für das Q-LOGIK-Entscheidungsmodul.
Behandelt Zielkonflikte zwischen verschiedenen Modulen und Entscheidungspfaden.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import json
import logging
import math
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import os
import sys

# Logger einrichten
logger = logging.getLogger("MISO.Logic.Q-LOGIK.ConflictResolver")

class ConflictDefinition:
    """
    Konfliktdefinition
    
    Definiert einen Konflikt zwischen verschiedenen Zielen oder Entscheidungspfaden.
    """
    
    def __init__(self, name: str, parties: List[Dict[str, Any]], context: Dict[str, Any] = None):
        """
        Initialisiert eine Konfliktdefinition
        
        Args:
            name: Name des Konflikts
            parties: Liste der Konfliktparteien
            context: Kontextinformationen für den Konflikt
        """
        self.name = name
        self.parties = parties
        self.context = context or {}
        self.resolution = None
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert die Konfliktdefinition in ein Dictionary
        
        Returns:
            Dictionary-Repräsentation des Konflikts
        """
        return {
            "name": self.name,
            "parties": self.parties,
            "context": self.context,
            "resolution": self.resolution
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConflictDefinition':
        """
        Erstellt eine Konfliktdefinition aus einem Dictionary
        
        Args:
            data: Dictionary mit Konfliktdaten
            
        Returns:
            Konfliktdefinition
        """
        conflict = cls(
            name=data.get("name", "Unbenannter Konflikt"),
            parties=data.get("parties", []),
            context=data.get("context", {})
        )
        conflict.resolution = data.get("resolution", None)
        return conflict


class ConflictPattern:
    """
    Konfliktmuster
    
    Definiert ein Muster für wiederkehrende Konflikte und deren Lösungsansätze.
    """
    
    def __init__(self, name: str, pattern: Dict[str, Any], resolution_strategy: Dict[str, Any]):
        """
        Initialisiert ein Konfliktmuster
        
        Args:
            name: Name des Musters
            pattern: Muster für die Erkennung des Konflikts
            resolution_strategy: Strategie zur Lösung des Konflikts
        """
        self.name = name
        self.pattern = pattern
        self.resolution_strategy = resolution_strategy
        
    def matches(self, conflict: ConflictDefinition) -> bool:
        """
        Prüft, ob ein Konflikt dem Muster entspricht
        
        Args:
            conflict: Zu prüfender Konflikt
            
        Returns:
            True, wenn der Konflikt dem Muster entspricht, sonst False
        """
        # Einfache Implementierung: Prüfe, ob die Schlüsselwörter im Konflikt vorkommen
        keywords = self.pattern.get("keywords", [])
        if not keywords:
            return False
            
        # Prüfe im Namen des Konflikts
        if any(keyword.lower() in conflict.name.lower() for keyword in keywords):
            return True
            
        # Prüfe in den Namen der Parteien
        for party in conflict.parties:
            if "name" in party and any(keyword.lower() in party["name"].lower() for keyword in keywords):
                return True
                
        # Prüfe im Kontext
        for key, value in conflict.context.items():
            if isinstance(value, str) and any(keyword.lower() in value.lower() for keyword in keywords):
                return True
                
        return False
        
    def apply_resolution(self, conflict: ConflictDefinition) -> Dict[str, Any]:
        """
        Wendet die Lösungsstrategie auf einen Konflikt an
        
        Args:
            conflict: Zu lösender Konflikt
            
        Returns:
            Lösungsvorschlag
        """
        strategy_type = self.resolution_strategy.get("type", "default")
        
        if strategy_type == "prioritize":
            # Priorisiere eine Partei basierend auf einem Kriterium
            criterion = self.resolution_strategy.get("criterion", "benefit")
            reverse = self.resolution_strategy.get("reverse", True)
            
            # Sortiere die Parteien nach dem Kriterium
            sorted_parties = sorted(
                conflict.parties,
                key=lambda p: p.get(criterion, 0),
                reverse=reverse
            )
            
            if sorted_parties:
                return {
                    "resolution": "prioritize",
                    "selected": sorted_parties[0]["name"],
                    "criterion": criterion,
                    "reasoning": f"Priorisierung basierend auf {criterion}"
                }
        elif strategy_type == "compromise":
            # Kompromiss: Kombiniere Aspekte mehrerer Parteien
            return {
                "resolution": "compromise",
                "parties": [p["name"] for p in conflict.parties],
                "reasoning": "Kompromisslösung zwischen allen Parteien"
            }
        elif strategy_type == "defer":
            # Aufschieben: Keine sofortige Entscheidung
            return {
                "resolution": "defer",
                "reasoning": "Entscheidung aufgeschoben"
            }
        
        # Fallback: Standardlösung
        return {
            "resolution": "default",
            "reasoning": "Standardlösungsstrategie angewendet"
        }


class AdvancedConflictResolver:
    """
    Erweiterter Konfliktlöser
    
    Erkennt und löst komplexe Konflikte basierend auf Mustern und Strategien.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialisiert den erweiterten Konfliktlöser
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.patterns = []
        self.config = {}
        
        # Lade Konfiguration, falls vorhanden
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                    
                # Lade Konfliktmuster
                for name, pattern_data in self.config.get("patterns", {}).items():
                    pattern = ConflictPattern(
                        name=name,
                        pattern=pattern_data.get("pattern", {}),
                        resolution_strategy=pattern_data.get("resolution_strategy", {})
                    )
                    self.patterns.append(pattern)
                    
                logger.info(f"Konfiguration geladen aus {config_path}")
            except Exception as e:
                logger.error(f"Fehler beim Laden der Konfiguration: {str(e)}")
                
        # Standardmuster, falls keine geladen wurden
        if not self.patterns:
            self._load_default_patterns()
            
        logger.info("AdvancedConflictResolver initialisiert")
        
    def _load_default_patterns(self):
        """
        Lädt Standardmuster für die Konfliktlösung
        """
        # Sicherheit vs. Nutzen
        self.patterns.append(ConflictPattern(
            name="safety_vs_utility",
            pattern={
                "keywords": ["sicherheit", "risiko", "nutzen", "vorteil"]
            },
            resolution_strategy={
                "type": "prioritize",
                "criterion": "risk",
                "reverse": False,
                "description": "Priorisiere die sicherste Option"
            }
        ))
        
        # Zeit vs. Qualität
        self.patterns.append(ConflictPattern(
            name="time_vs_quality",
            pattern={
                "keywords": ["zeit", "schnell", "qualität", "genauigkeit"]
            },
            resolution_strategy={
                "type": "compromise",
                "description": "Finde einen Kompromiss zwischen Zeit und Qualität"
            }
        ))
        
        # Kosten vs. Leistung
        self.patterns.append(ConflictPattern(
            name="cost_vs_performance",
            pattern={
                "keywords": ["kosten", "preis", "leistung", "performance"]
            },
            resolution_strategy={
                "type": "prioritize",
                "criterion": "benefit",
                "reverse": True,
                "description": "Priorisiere die Option mit dem besten Kosten-Nutzen-Verhältnis"
            }
        ))
        
        logger.info("Standardmuster geladen")
        
    def resolve_conflict(self, conflict: Union[ConflictDefinition, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Löst einen Konflikt
        
        Args:
            conflict: Zu lösender Konflikt
            
        Returns:
            Lösungsvorschlag
        """
        # Konvertiere Dictionary in ConflictDefinition, falls nötig
        if isinstance(conflict, dict):
            conflict = ConflictDefinition.from_dict(conflict)
            
        # Finde passendes Muster
        for pattern in self.patterns:
            if pattern.matches(conflict):
                resolution = pattern.apply_resolution(conflict)
                conflict.resolution = resolution
                return resolution
                
        # Fallback: Standardlösung
        resolution = {
            "resolution": "default",
            "reasoning": "Kein passendes Muster gefunden, Standardlösung angewendet"
        }
        conflict.resolution = resolution
        return resolution
        
    def add_pattern(self, pattern: ConflictPattern):
        """
        Fügt ein neues Konfliktmuster hinzu
        
        Args:
            pattern: Hinzuzufügendes Muster
        """
        self.patterns.append(pattern)
        logger.info(f"Muster '{pattern.name}' hinzugefügt")
        
    def save_config(self, config_path: str):
        """
        Speichert die Konfiguration
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        config = {"patterns": {}}
        
        for pattern in self.patterns:
            config["patterns"][pattern.name] = {
                "pattern": pattern.pattern,
                "resolution_strategy": pattern.resolution_strategy
            }
            
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Konfiguration gespeichert in {config_path}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Konfiguration: {str(e)}")


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Konfiguriere Logging
    logging.basicConfig(level=logging.INFO)
    
    # Erstelle Konfliktlöser
    resolver = AdvancedConflictResolver()
    
    # Beispielkonflikt: Sicherheit vs. Nutzen
    conflict = ConflictDefinition(
        name="Sicherheit vs. Nutzen bei Datenzugriff",
        parties=[
            {
                "name": "Sicherheitsmodul",
                "risk": 0.2,
                "benefit": 0.5,
                "urgency": 0.7
            },
            {
                "name": "Effizienzmodul",
                "risk": 0.6,
                "benefit": 0.8,
                "urgency": 0.4
            }
        ],
        context={
            "description": "Konflikt zwischen Datensicherheit und Zugriffseffizienz"
        }
    )
    
    # Löse Konflikt
    resolution = resolver.resolve_conflict(conflict)
    
    # Ausgabe
    print("Konflikt:", conflict.name)
    print("Parteien:", [p["name"] for p in conflict.parties])
    print("Lösung:", resolution)
