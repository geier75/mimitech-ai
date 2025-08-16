#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - HYPERFILTER

Dieses Modul implementiert das HYPERFILTER-System für MISO Ultimate, das für
die Filterung und Verarbeitung von Eingabe- und Ausgabedaten verwendet wird.
Es bietet fortschrittliche Filtermechanismen für verschiedene Datentypen und
unterstützt adaptive Filterung basierend auf Kontext und Inhalt.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
from dataclasses import dataclass, field

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.filter.hyperfilter")

class FilterMode(Enum):
    """Filtermodi für den HYPERFILTER"""
    STRICT = auto()       # Strenge Filterung
    BALANCED = auto()     # Ausgewogene Filterung
    PERMISSIVE = auto()   # Permissive Filterung
    ADAPTIVE = auto()     # Adaptive Filterung basierend auf Kontext
    CUSTOM = auto()       # Benutzerdefinierte Filterung

@dataclass
class FilterConfig:
    """Konfiguration für den HYPERFILTER"""
    mode: FilterMode = FilterMode.BALANCED
    threshold: float = 0.75
    context_sensitivity: float = 0.5
    adaptive_learning: bool = True
    custom_rules: Dict[str, Any] = field(default_factory=dict)
    allowed_patterns: List[str] = field(default_factory=list)
    blocked_patterns: List[str] = field(default_factory=list)
    max_recursion_depth: int = 3
    use_ml_enhancement: bool = True

class HyperFilter:
    """
    HYPERFILTER-Hauptklasse
    
    Diese Klasse implementiert das HYPERFILTER-System für MISO Ultimate,
    das für die Filterung und Verarbeitung von Eingabe- und Ausgabedaten
    verwendet wird.
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialisiert den HYPERFILTER
        
        Args:
            config: Konfigurationsobjekt für den HYPERFILTER
        """
        self.config = config or FilterConfig()
        self.filter_cache = {}
        self.pattern_database = {}
        self.context_history = []
        self.ml_models = {}
        
        # Initialisiere Filter-Komponenten
        self._initialize_components()
        
        logger.info(f"HYPERFILTER initialisiert mit Modus: {self.config.mode.name}")
        
    def _initialize_components(self):
        """Initialisiert alle Komponenten des HYPERFILTER"""
        self._load_pattern_database()
        
        if self.config.use_ml_enhancement:
            self._initialize_ml_models()
    
    def _load_pattern_database(self):
        """Lädt die Muster-Datenbank"""
        # In einer realen Implementierung würde dies Muster aus einer Datenbank laden
        self.pattern_database = {
            "input_patterns": {
                "malicious": ["exec(", "eval(", "os.system(", "import os;"],
                "sensitive": ["password", "key", "secret", "token"],
                "irrelevant": ["debug", "test", "placeholder"]
            },
            "output_patterns": {
                "sensitive": ["internal error", "stack trace", "exception in"],
                "formatting": ["undefined", "null", "NaN", "[object Object]"]
            }
        }
        
        # Füge benutzerdefinierte Muster hinzu
        for pattern in self.config.allowed_patterns:
            self.pattern_database.setdefault("allowed", []).append(pattern)
            
        for pattern in self.config.blocked_patterns:
            self.pattern_database.setdefault("blocked", []).append(pattern)
    
    def _initialize_ml_models(self):
        """Initialisiert ML-Modelle für die erweiterte Filterung"""
        # In einer realen Implementierung würden hier ML-Modelle geladen
        logger.info("ML-Modelle für HYPERFILTER initialisiert")
        
        # Simuliere ML-Modelle für verschiedene Filtertypen
        self.ml_models = {
            "content_classifier": {"type": "transformer", "loaded": True},
            "pattern_recognizer": {"type": "lstm", "loaded": True},
            "context_analyzer": {"type": "attention", "loaded": True}
        }
    
    def filter_input(self, input_data: Union[str, Dict, List], context: Optional[Dict] = None) -> Tuple[Union[str, Dict, List], Dict]:
        """
        Filtert Eingabedaten
        
        Args:
            input_data: Zu filternde Eingabedaten
            context: Kontext für die Filterung
            
        Returns:
            Tuple aus gefilterten Daten und Metadaten
        """
        logger.debug(f"Filtere Eingabe: {type(input_data)}")
        
        # Aktualisiere Kontext-Historie
        if context:
            self.context_history.append(context)
            if len(self.context_history) > 10:
                self.context_history.pop(0)
        
        # Führe Filterung basierend auf dem Datentyp durch
        if isinstance(input_data, str):
            return self._filter_text_input(input_data, context)
        elif isinstance(input_data, dict):
            return self._filter_dict_input(input_data, context)
        elif isinstance(input_data, list):
            return self._filter_list_input(input_data, context)
        else:
            logger.warning(f"Nicht unterstützter Eingabetyp: {type(input_data)}")
            return input_data, {"status": "unsupported_type", "filtered": False}
    
    def _filter_text_input(self, text: str, context: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        Filtert Texteingabe
        
        Args:
            text: Zu filternder Text
            context: Kontext für die Filterung
            
        Returns:
            Tuple aus gefiltertem Text und Metadaten
        """
        filtered_text = text
        metadata = {
            "original_length": len(text),
            "filtered_length": len(text),
            "patterns_matched": [],
            "filtered": False
        }
        
        # Prüfe auf blockierte Muster
        for pattern in self.pattern_database.get("blocked", []):
            if pattern in filtered_text:
                filtered_text = filtered_text.replace(pattern, "[GEFILTERT]")
                metadata["patterns_matched"].append(pattern)
                metadata["filtered"] = True
        
        # Prüfe auf bösartige Muster
        for pattern in self.pattern_database["input_patterns"]["malicious"]:
            if pattern in filtered_text:
                filtered_text = filtered_text.replace(pattern, "[SICHERHEITSFILTER]")
                metadata["patterns_matched"].append(pattern)
                metadata["filtered"] = True
        
        # Prüfe auf sensible Muster basierend auf dem Modus
        if self.config.mode in [FilterMode.STRICT, FilterMode.BALANCED]:
            for pattern in self.pattern_database["input_patterns"]["sensitive"]:
                if pattern in filtered_text:
                    filtered_text = filtered_text.replace(pattern, "[VERTRAULICH]")
                    metadata["patterns_matched"].append(pattern)
                    metadata["filtered"] = True
        
        # Adaptive Filterung basierend auf Kontext
        if self.config.mode == FilterMode.ADAPTIVE and context:
            filtered_text = self._apply_adaptive_filtering(filtered_text, context)
        
        metadata["filtered_length"] = len(filtered_text)
        return filtered_text, metadata
    
    def _filter_dict_input(self, data: Dict, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Filtert Wörterbucheingabe
        
        Args:
            data: Zu filterndes Wörterbuch
            context: Kontext für die Filterung
            
        Returns:
            Tuple aus gefiltertem Wörterbuch und Metadaten
        """
        filtered_data = {}
        metadata = {
            "fields_total": len(data),
            "fields_filtered": 0,
            "patterns_matched": [],
            "filtered": False
        }
        
        for key, value in data.items():
            # Rekursive Filterung für verschachtelte Strukturen
            if isinstance(value, (dict, list)) and self.config.max_recursion_depth > 0:
                sub_config = FilterConfig(**vars(self.config))
                sub_config.max_recursion_depth = self.config.max_recursion_depth - 1
                sub_filter = HyperFilter(sub_config)
                filtered_value, sub_metadata = sub_filter.filter_input(value, context)
                
                if sub_metadata["filtered"]:
                    metadata["fields_filtered"] += 1
                    metadata["filtered"] = True
                    metadata["patterns_matched"].extend(sub_metadata["patterns_matched"])
                
                filtered_data[key] = filtered_value
            
            # Textfilterung für Zeichenketten
            elif isinstance(value, str):
                filtered_value, sub_metadata = self._filter_text_input(value, context)
                
                if sub_metadata["filtered"]:
                    metadata["fields_filtered"] += 1
                    metadata["filtered"] = True
                    metadata["patterns_matched"].extend(sub_metadata["patterns_matched"])
                
                filtered_data[key] = filtered_value
            
            # Andere Typen unverändert übernehmen
            else:
                filtered_data[key] = value
        
        return filtered_data, metadata
    
    def _filter_list_input(self, data: List, context: Optional[Dict] = None) -> Tuple[List, Dict]:
        """
        Filtert Listeneingabe
        
        Args:
            data: Zu filternde Liste
            context: Kontext für die Filterung
            
        Returns:
            Tuple aus gefilterter Liste und Metadaten
        """
        filtered_data = []
        metadata = {
            "items_total": len(data),
            "items_filtered": 0,
            "patterns_matched": [],
            "filtered": False
        }
        
        for item in data:
            # Rekursive Filterung für verschachtelte Strukturen
            if isinstance(item, (dict, list)) and self.config.max_recursion_depth > 0:
                sub_config = FilterConfig(**vars(self.config))
                sub_config.max_recursion_depth = self.config.max_recursion_depth - 1
                sub_filter = HyperFilter(sub_config)
                filtered_item, sub_metadata = sub_filter.filter_input(item, context)
                
                if sub_metadata["filtered"]:
                    metadata["items_filtered"] += 1
                    metadata["filtered"] = True
                    metadata["patterns_matched"].extend(sub_metadata["patterns_matched"])
                
                filtered_data.append(filtered_item)
            
            # Textfilterung für Zeichenketten
            elif isinstance(item, str):
                filtered_item, sub_metadata = self._filter_text_input(item, context)
                
                if sub_metadata["filtered"]:
                    metadata["items_filtered"] += 1
                    metadata["filtered"] = True
                    metadata["patterns_matched"].extend(sub_metadata["patterns_matched"])
                
                filtered_data.append(filtered_item)
            
            # Andere Typen unverändert übernehmen
            else:
                filtered_data.append(item)
        
        return filtered_data, metadata
    
    def filter_output(self, output_data: Union[str, Dict, List], context: Optional[Dict] = None) -> Tuple[Union[str, Dict, List], Dict]:
        """
        Filtert Ausgabedaten
        
        Args:
            output_data: Zu filternde Ausgabedaten
            context: Kontext für die Filterung
            
        Returns:
            Tuple aus gefilterten Daten und Metadaten
        """
        logger.debug(f"Filtere Ausgabe: {type(output_data)}")
        
        # Aktualisiere Kontext-Historie
        if context:
            self.context_history.append(context)
            if len(self.context_history) > 10:
                self.context_history.pop(0)
        
        # Führe Filterung basierend auf dem Datentyp durch
        if isinstance(output_data, str):
            return self._filter_text_output(output_data, context)
        elif isinstance(output_data, dict):
            return self._filter_dict_output(output_data, context)
        elif isinstance(output_data, list):
            return self._filter_list_output(output_data, context)
        else:
            logger.warning(f"Nicht unterstützter Ausgabetyp: {type(output_data)}")
            return output_data, {"status": "unsupported_type", "filtered": False}
    
    def _filter_text_output(self, text: str, context: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        Filtert Textausgabe
        
        Args:
            text: Zu filternder Text
            context: Kontext für die Filterung
            
        Returns:
            Tuple aus gefiltertem Text und Metadaten
        """
        filtered_text = text
        metadata = {
            "original_length": len(text),
            "filtered_length": len(text),
            "patterns_matched": [],
            "filtered": False
        }
        
        # Prüfe auf sensible Muster
        for pattern in self.pattern_database["output_patterns"]["sensitive"]:
            if pattern in filtered_text:
                filtered_text = filtered_text.replace(pattern, "[INTERNE INFORMATION ENTFERNT]")
                metadata["patterns_matched"].append(pattern)
                metadata["filtered"] = True
        
        # Prüfe auf Formatierungsprobleme
        for pattern in self.pattern_database["output_patterns"]["formatting"]:
            if pattern in filtered_text:
                filtered_text = filtered_text.replace(pattern, "[FORMATIERUNGSPROBLEM]")
                metadata["patterns_matched"].append(pattern)
                metadata["filtered"] = True
        
        # Adaptive Filterung basierend auf Kontext
        if self.config.mode == FilterMode.ADAPTIVE and context:
            filtered_text = self._apply_adaptive_filtering(filtered_text, context, is_output=True)
        
        metadata["filtered_length"] = len(filtered_text)
        return filtered_text, metadata
    
    def _filter_dict_output(self, data: Dict, context: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Filtert Wörterbuchausgabe
        
        Args:
            data: Zu filterndes Wörterbuch
            context: Kontext für die Filterung
            
        Returns:
            Tuple aus gefiltertem Wörterbuch und Metadaten
        """
        filtered_data = {}
        metadata = {
            "fields_total": len(data),
            "fields_filtered": 0,
            "patterns_matched": [],
            "filtered": False
        }
        
        for key, value in data.items():
            # Rekursive Filterung für verschachtelte Strukturen
            if isinstance(value, (dict, list)) and self.config.max_recursion_depth > 0:
                sub_config = FilterConfig(**vars(self.config))
                sub_config.max_recursion_depth = self.config.max_recursion_depth - 1
                sub_filter = HyperFilter(sub_config)
                filtered_value, sub_metadata = sub_filter.filter_output(value, context)
                
                if sub_metadata["filtered"]:
                    metadata["fields_filtered"] += 1
                    metadata["filtered"] = True
                    metadata["patterns_matched"].extend(sub_metadata["patterns_matched"])
                
                filtered_data[key] = filtered_value
            
            # Textfilterung für Zeichenketten
            elif isinstance(value, str):
                filtered_value, sub_metadata = self._filter_text_output(value, context)
                
                if sub_metadata["filtered"]:
                    metadata["fields_filtered"] += 1
                    metadata["filtered"] = True
                    metadata["patterns_matched"].extend(sub_metadata["patterns_matched"])
                
                filtered_data[key] = filtered_value
            
            # Andere Typen unverändert übernehmen
            else:
                filtered_data[key] = value
        
        return filtered_data, metadata
    
    def _filter_list_output(self, data: List, context: Optional[Dict] = None) -> Tuple[List, Dict]:
        """
        Filtert Listenausgabe
        
        Args:
            data: Zu filternde Liste
            context: Kontext für die Filterung
            
        Returns:
            Tuple aus gefilterter Liste und Metadaten
        """
        filtered_data = []
        metadata = {
            "items_total": len(data),
            "items_filtered": 0,
            "patterns_matched": [],
            "filtered": False
        }
        
        for item in data:
            # Rekursive Filterung für verschachtelte Strukturen
            if isinstance(item, (dict, list)) and self.config.max_recursion_depth > 0:
                sub_config = FilterConfig(**vars(self.config))
                sub_config.max_recursion_depth = self.config.max_recursion_depth - 1
                sub_filter = HyperFilter(sub_config)
                filtered_item, sub_metadata = sub_filter.filter_output(item, context)
                
                if sub_metadata["filtered"]:
                    metadata["items_filtered"] += 1
                    metadata["filtered"] = True
                    metadata["patterns_matched"].extend(sub_metadata["patterns_matched"])
                
                filtered_data.append(filtered_item)
            
            # Textfilterung für Zeichenketten
            elif isinstance(item, str):
                filtered_item, sub_metadata = self._filter_text_output(item, context)
                
                if sub_metadata["filtered"]:
                    metadata["items_filtered"] += 1
                    metadata["filtered"] = True
                    metadata["patterns_matched"].extend(sub_metadata["patterns_matched"])
                
                filtered_data.append(filtered_item)
            
            # Andere Typen unverändert übernehmen
            else:
                filtered_data.append(item)
        
        return filtered_data, metadata
    
    def _apply_adaptive_filtering(self, text: str, context: Dict, is_output: bool = False) -> str:
        """
        Wendet adaptive Filterung basierend auf Kontext an
        
        Args:
            text: Zu filternder Text
            context: Kontext für die Filterung
            is_output: Gibt an, ob es sich um Ausgabedaten handelt
            
        Returns:
            Gefilterter Text
        """
        # In einer realen Implementierung würde hier ML verwendet werden
        # Für diese Beispielimplementierung verwenden wir einen einfachen Ansatz
        
        filtered_text = text
        
        # Kontext-basierte Filterung
        if "sensitivity_level" in context:
            sensitivity = context["sensitivity_level"]
            
            if sensitivity == "high":
                # Strenge Filterung für hohe Sensitivität
                for pattern in self.pattern_database["input_patterns"]["sensitive"]:
                    if pattern in filtered_text:
                        filtered_text = filtered_text.replace(pattern, "[VERTRAULICH]")
            
            elif sensitivity == "low" and is_output:
                # Permissive Filterung für niedrige Sensitivität bei Ausgaben
                # Hier könnten wir einige Filterungen überspringen
                pass
        
        # Domänen-spezifische Filterung
        if "domain" in context:
            domain = context["domain"]
            
            if domain == "security" and not is_output:
                # Spezielle Filterung für Sicherheitsdomäne bei Eingaben
                for pattern in self.pattern_database["input_patterns"]["malicious"]:
                    if pattern in filtered_text:
                        filtered_text = filtered_text.replace(pattern, "[SICHERHEITSFILTER]")
            
            elif domain == "development" and is_output:
                # Entwickler benötigen möglicherweise mehr Details in Ausgaben
                pass
        
        return filtered_text
    
    def update_config(self, new_config: FilterConfig) -> None:
        """
        Aktualisiert die Konfiguration des HYPERFILTER
        
        Args:
            new_config: Neue Konfiguration
        """
        self.config = new_config
        logger.info(f"HYPERFILTER-Konfiguration aktualisiert: {self.config.mode.name}")
        
        # Aktualisiere Komponenten nach Konfigurationsänderung
        self._initialize_components()
    
    def add_pattern(self, pattern: str, pattern_type: str, category: str) -> bool:
        """
        Fügt ein Muster zur Datenbank hinzu
        
        Args:
            pattern: Hinzuzufügendes Muster
            pattern_type: Typ des Musters ('input' oder 'output')
            category: Kategorie des Musters
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            if pattern_type == "input":
                self.pattern_database["input_patterns"].setdefault(category, []).append(pattern)
            elif pattern_type == "output":
                self.pattern_database["output_patterns"].setdefault(category, []).append(pattern)
            else:
                logger.warning(f"Ungültiger Mustertyp: {pattern_type}")
                return False
            
            logger.info(f"Muster hinzugefügt: {pattern} ({pattern_type}/{category})")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen des Musters: {e}")
            return False
    
    def remove_pattern(self, pattern: str, pattern_type: str, category: str) -> bool:
        """
        Entfernt ein Muster aus der Datenbank
        
        Args:
            pattern: Zu entfernendes Muster
            pattern_type: Typ des Musters ('input' oder 'output')
            category: Kategorie des Musters
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            if pattern_type == "input" and category in self.pattern_database["input_patterns"]:
                if pattern in self.pattern_database["input_patterns"][category]:
                    self.pattern_database["input_patterns"][category].remove(pattern)
                    logger.info(f"Muster entfernt: {pattern} ({pattern_type}/{category})")
                    return True
            elif pattern_type == "output" and category in self.pattern_database["output_patterns"]:
                if pattern in self.pattern_database["output_patterns"][category]:
                    self.pattern_database["output_patterns"][category].remove(pattern)
                    logger.info(f"Muster entfernt: {pattern} ({pattern_type}/{category})")
                    return True
            
            logger.warning(f"Muster nicht gefunden: {pattern} ({pattern_type}/{category})")
            return False
        except Exception as e:
            logger.error(f"Fehler beim Entfernen des Musters: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über den HYPERFILTER zurück
        
        Returns:
            Statistiken als Wörterbuch
        """
        return {
            "mode": self.config.mode.name,
            "pattern_counts": {
                "input": {k: len(v) for k, v in self.pattern_database["input_patterns"].items()},
                "output": {k: len(v) for k, v in self.pattern_database["output_patterns"].items()}
            },
            "context_history_length": len(self.context_history),
            "cache_size": len(self.filter_cache),
            "ml_models": {k: v["type"] for k, v in self.ml_models.items()} if self.config.use_ml_enhancement else {}
        }
    
    def clear_cache(self) -> None:
        """Leert den Filter-Cache"""
        self.filter_cache.clear()
        logger.info("HYPERFILTER-Cache geleert")
    
    def clear_context_history(self) -> None:
        """Leert die Kontext-Historie"""
        self.context_history.clear()
        logger.info("HYPERFILTER-Kontext-Historie geleert")

# Globale Instanz
_HYPERFILTER = None

def get_hyperfilter(config: Optional[FilterConfig] = None) -> HyperFilter:
    """
    Gibt die globale HYPERFILTER-Instanz zurück
    
    Args:
        config: Konfigurationsobjekt für den HYPERFILTER (optional)
        
    Returns:
        HyperFilter-Instanz
    """
    global _HYPERFILTER
    
    if _HYPERFILTER is None:
        _HYPERFILTER = HyperFilter(config)
    elif config is not None:
        _HYPERFILTER.update_config(config)
    
    return _HYPERFILTER

def reset_hyperfilter() -> None:
    """Setzt die globale HYPERFILTER-Instanz zurück"""
    global _HYPERFILTER
    _HYPERFILTER = None
    logger.info("Globale HYPERFILTER-Instanz zurückgesetzt")
