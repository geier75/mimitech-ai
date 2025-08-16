#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - VXOR-HYPERFILTER Integration

Dieses Modul implementiert die Integration zwischen dem MISO HYPERFILTER-Modul
und dem VX-HYPERFILTER-Agenten aus dem VXOR-System.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import json
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
from dataclasses import dataclass, field

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.filter.vxor_hyperfilter_integration")

# Importiere HYPERFILTER
from miso.filter.hyperfilter import HyperFilter, FilterConfig, FilterMode

# Versuche, VXOR-Module zu importieren
try:
    from miso.vxor_integration import VXORAdapter, get_vxor_adapter
    VXOR_AVAILABLE = True
except ImportError:
    VXOR_AVAILABLE = False
    logger.warning("VXOR-Integration nicht verfügbar, einige Funktionen sind eingeschränkt")

class TrustLevel(Enum):
    """Vertrauensstufen für den VX-HYPERFILTER"""
    HIGH = auto()       # Hohe Vertrauenswürdigkeit
    MEDIUM = auto()     # Mittlere Vertrauenswürdigkeit
    LOW = auto()        # Niedrige Vertrauenswürdigkeit
    UNTRUSTED = auto()  # Nicht vertrauenswürdig

@dataclass
class VXHyperfilterConfig:
    """Konfiguration für den VX-HYPERFILTER"""
    language_analyzer_enabled: bool = True
    trust_validator_enabled: bool = True
    sentiment_engine_enabled: bool = True
    context_normalizer_enabled: bool = True
    default_trust_score: float = 0.5
    default_language_code: str = "de"
    media_source_types: List[str] = field(default_factory=lambda: ["News", "Social", "Forum", "Private"])
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    vxor_integration_enabled: bool = True

class VXHyperfilterAdapter:
    """
    Adapter für den VX-HYPERFILTER
    
    Diese Klasse implementiert die Integration zwischen dem MISO HYPERFILTER-Modul
    und dem VX-HYPERFILTER-Agenten aus dem VXOR-System.
    """
    
    def __init__(self, config: Optional[VXHyperfilterConfig] = None):
        """
        Initialisiert den VX-HYPERFILTER-Adapter
        
        Args:
            config: Konfigurationsobjekt für den VX-HYPERFILTER-Adapter
        """
        self.config = config or VXHyperfilterConfig()
        self.hyperfilter = HyperFilter(self.config.filter_config)
        self.vxor_adapter = None
        
        # Initialisiere VXOR-Integration, falls verfügbar
        if VXOR_AVAILABLE and self.config.vxor_integration_enabled:
            try:
                self.vxor_adapter = get_vxor_adapter()
                logger.info("VXOR-Adapter erfolgreich initialisiert")
            except Exception as e:
                logger.error(f"Fehler bei der Initialisierung des VXOR-Adapters: {e}")
        
        # Initialisiere Komponenten
        self._initialize_components()
        
        logger.info("VX-HYPERFILTER-Adapter initialisiert")
    
    def _initialize_components(self):
        """Initialisiert die Komponenten des VX-HYPERFILTER"""
        # Registriere HYPERFILTER bei VXOR, falls verfügbar
        if self.vxor_adapter:
            try:
                # Registriere den HYPERFILTER als VXOR-Modul
                self.vxor_adapter.register_module(
                    module_id="VX-HYPERFILTER",
                    module_type="INTELLIGENT_FILTER_AGENT",
                    module_version="1.0",
                    module_description="Autonomer Echtzeit-Anti-Propaganda- und Bias-Analyse-Agent",
                    module_capabilities=[
                        "text_filtering",
                        "bias_detection",
                        "propaganda_detection",
                        "deepfake_detection",
                        "trust_validation"
                    ],
                    module_dependencies=[
                        "VX-MEMEX",
                        "VX-REASON",
                        "VX-CONTEXT"
                    ]
                )
                logger.info("HYPERFILTER erfolgreich als VXOR-Modul registriert")
            except Exception as e:
                logger.error(f"Fehler bei der Registrierung des HYPERFILTER als VXOR-Modul: {e}")
    
    def process_content(self, 
                        raw_text: str, 
                        source_trust_score: Optional[float] = None, 
                        language_code: Optional[str] = None, 
                        context_stream: Optional[str] = None, 
                        media_source_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Verarbeitet Inhalte mit dem VX-HYPERFILTER
        
        Args:
            raw_text: Zu verarbeitender Rohtext
            source_trust_score: Vertrauenswert der Quelle
            language_code: Sprachcode
            context_stream: Kontextinformationen
            media_source_type: Typ der Medienquelle
            
        Returns:
            Ergebnis der Verarbeitung
        """
        # Setze Standardwerte für fehlende Parameter
        source_trust_score = source_trust_score if source_trust_score is not None else self.config.default_trust_score
        language_code = language_code or self.config.default_language_code
        media_source_type = media_source_type or "Unknown"
        
        # Erstelle Kontext für die Filterung
        context = {
            "source_trust_score": source_trust_score,
            "language_code": language_code,
            "media_source_type": media_source_type,
            "context_stream": context_stream
        }
        
        # Führe Filterung durch
        filtered_text, metadata = self.hyperfilter.filter_input(raw_text, context)
        
        # Bestimme Vertrauensstufe basierend auf Filterungsergebnissen und Quellvertrauen
        trust_level = self._determine_trust_level(metadata, source_trust_score)
        
        # Erstelle Entscheidung basierend auf Vertrauensstufe
        decision = self._make_decision(trust_level, metadata)
        
        # Erstelle Zusammenfassung
        summary = self._generate_summary(filtered_text, metadata, trust_level, decision)
        
        # Bestimme, ob weitere Aktionen erforderlich sind
        action_trigger = trust_level in [TrustLevel.LOW, TrustLevel.UNTRUSTED]
        
        # Speichere Ergebnis in VX-MEMEX, falls verfügbar
        if self.vxor_adapter and action_trigger:
            self._store_in_memex(raw_text, filtered_text, metadata, trust_level, decision, summary)
        
        # Erstelle Ergebnis
        result = {
            "signal_flag": trust_level.name,
            "report_summary": summary,
            "decision": decision,
            "action_trigger": action_trigger,
            "filtered_text": filtered_text,
            "metadata": metadata
        }
        
        return result
    
    def _determine_trust_level(self, metadata: Dict[str, Any], source_trust_score: float) -> TrustLevel:
        """
        Bestimmt die Vertrauensstufe basierend auf Filterungsergebnissen und Quellvertrauen
        
        Args:
            metadata: Metadaten der Filterung
            source_trust_score: Vertrauenswert der Quelle
            
        Returns:
            Vertrauensstufe
        """
        # Bestimme Vertrauensstufe basierend auf Filterungsergebnissen
        if not metadata["filtered"]:
            # Keine Filterung erforderlich
            if source_trust_score >= 0.8:
                return TrustLevel.HIGH
            elif source_trust_score >= 0.5:
                return TrustLevel.MEDIUM
            else:
                return TrustLevel.LOW
        else:
            # Filterung erforderlich
            patterns_matched = metadata.get("patterns_matched", [])
            
            # Prüfe auf bösartige Muster
            malicious_patterns = [p for p in patterns_matched if "malicious" in p or "exec" in p or "eval" in p]
            if malicious_patterns:
                return TrustLevel.UNTRUSTED
            
            # Prüfe auf sensible Muster
            sensitive_patterns = [p for p in patterns_matched if "sensitive" in p or "vertraulich" in p]
            if sensitive_patterns and source_trust_score < 0.5:
                return TrustLevel.UNTRUSTED
            elif sensitive_patterns:
                return TrustLevel.LOW
            
            # Standardfall
            return TrustLevel.MEDIUM
    
    def _make_decision(self, trust_level: TrustLevel, metadata: Dict[str, Any]) -> str:
        """
        Trifft eine Entscheidung basierend auf der Vertrauensstufe
        
        Args:
            trust_level: Vertrauensstufe
            metadata: Metadaten der Filterung
            
        Returns:
            Entscheidung
        """
        if trust_level == TrustLevel.HIGH:
            return "Vertrauenswürdig"
        elif trust_level == TrustLevel.MEDIUM:
            return "Akzeptieren"
        elif trust_level == TrustLevel.LOW:
            return "Warnung"
        else:  # TrustLevel.UNTRUSTED
            return "Blockieren"
    
    def _generate_summary(self, filtered_text: str, metadata: Dict[str, Any], trust_level: TrustLevel, decision: str) -> str:
        """
        Generiert eine Zusammenfassung der Filterungsergebnisse
        
        Args:
            filtered_text: Gefilterter Text
            metadata: Metadaten der Filterung
            trust_level: Vertrauensstufe
            decision: Entscheidung
            
        Returns:
            Zusammenfassung
        """
        # Erstelle Zusammenfassung
        summary_parts = [
            f"Vertrauensstufe: {trust_level.name}",
            f"Entscheidung: {decision}"
        ]
        
        # Füge Informationen über Filterung hinzu
        if metadata["filtered"]:
            patterns_matched = metadata.get("patterns_matched", [])
            summary_parts.append(f"Gefilterte Muster: {len(patterns_matched)}")
            
            # Füge Details zu gefilterten Mustern hinzu
            if patterns_matched:
                summary_parts.append("Gefundene Muster:")
                for pattern in patterns_matched[:5]:  # Begrenze auf 5 Muster
                    summary_parts.append(f"- {pattern}")
                
                if len(patterns_matched) > 5:
                    summary_parts.append(f"... und {len(patterns_matched) - 5} weitere")
        else:
            summary_parts.append("Keine Filterung erforderlich")
        
        # Erstelle Zusammenfassung
        return "\n".join(summary_parts)
    
    def _store_in_memex(self, raw_text: str, filtered_text: str, metadata: Dict[str, Any], trust_level: TrustLevel, decision: str, summary: str) -> None:
        """
        Speichert Filterungsergebnisse in VX-MEMEX
        
        Args:
            raw_text: Rohtext
            filtered_text: Gefilterter Text
            metadata: Metadaten der Filterung
            trust_level: Vertrauensstufe
            decision: Entscheidung
            summary: Zusammenfassung
        """
        if not self.vxor_adapter:
            return
        
        try:
            # Erstelle Memex-Eintrag
            memex_entry = {
                "module": "VX-HYPERFILTER",
                "entry_type": "filter_result",
                "timestamp": self.vxor_adapter.get_timestamp(),
                "trust_level": trust_level.name,
                "decision": decision,
                "summary": summary,
                "metadata": metadata,
                "raw_text_hash": self.vxor_adapter.compute_hash(raw_text),
                "filtered_text_hash": self.vxor_adapter.compute_hash(filtered_text)
            }
            
            # Speichere in VX-MEMEX
            self.vxor_adapter.store_in_memex("VX-HYPERFILTER", memex_entry)
            logger.info("Filterungsergebnis in VX-MEMEX gespeichert")
        except Exception as e:
            logger.error(f"Fehler beim Speichern in VX-MEMEX: {e}")
    
    def update_config(self, new_config: VXHyperfilterConfig) -> None:
        """
        Aktualisiert die Konfiguration des VX-HYPERFILTER-Adapters
        
        Args:
            new_config: Neue Konfiguration
        """
        self.config = new_config
        
        # Aktualisiere HYPERFILTER-Konfiguration
        self.hyperfilter.update_config(new_config.filter_config)
        
        logger.info("VX-HYPERFILTER-Konfiguration aktualisiert")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Gibt die Fähigkeiten des VX-HYPERFILTER zurück
        
        Returns:
            Fähigkeiten als Wörterbuch
        """
        capabilities = {
            "language_analyzer_enabled": self.config.language_analyzer_enabled,
            "trust_validator_enabled": self.config.trust_validator_enabled,
            "sentiment_engine_enabled": self.config.sentiment_engine_enabled,
            "context_normalizer_enabled": self.config.context_normalizer_enabled,
            "supported_languages": ["de", "en", "fr", "es", "it"],
            "supported_media_sources": self.config.media_source_types,
            "filter_mode": self.config.filter_config.mode.name,
            "vxor_integration": VXOR_AVAILABLE and self.config.vxor_integration_enabled
        }
        
        # Füge HYPERFILTER-Statistiken hinzu
        capabilities["hyperfilter_stats"] = self.hyperfilter.get_statistics()
        
        return capabilities

# Globale Instanz
_VX_HYPERFILTER_ADAPTER = None

def get_vx_hyperfilter_adapter(config: Optional[VXHyperfilterConfig] = None) -> VXHyperfilterAdapter:
    """
    Gibt die globale VX-HYPERFILTER-Adapter-Instanz zurück
    
    Args:
        config: Konfigurationsobjekt für den VX-HYPERFILTER-Adapter (optional)
        
    Returns:
        VXHyperfilterAdapter-Instanz
    """
    global _VX_HYPERFILTER_ADAPTER
    
    if _VX_HYPERFILTER_ADAPTER is None:
        _VX_HYPERFILTER_ADAPTER = VXHyperfilterAdapter(config)
    elif config is not None:
        _VX_HYPERFILTER_ADAPTER.update_config(config)
    
    return _VX_HYPERFILTER_ADAPTER

def reset_vx_hyperfilter_adapter() -> None:
    """Setzt die globale VX-HYPERFILTER-Adapter-Instanz zurück"""
    global _VX_HYPERFILTER_ADAPTER
    _VX_HYPERFILTER_ADAPTER = None
    logger.info("Globale VX-HYPERFILTER-Adapter-Instanz zurückgesetzt")
