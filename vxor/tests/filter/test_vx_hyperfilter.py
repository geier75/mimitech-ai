#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Test für VX-HYPERFILTER

Dieses Skript testet die Funktionalität des VX-HYPERFILTER-Moduls.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_vx_hyperfilter")

# Importiere HYPERFILTER-Module
from miso.filter.hyperfilter import HyperFilter, FilterConfig, FilterMode
from miso.filter.vxor_hyperfilter_integration import (
    VXHyperfilterAdapter, VXHyperfilterConfig, TrustLevel,
    get_vx_hyperfilter_adapter, reset_vx_hyperfilter_adapter
)

def test_hyperfilter_basic():
    """Testet die grundlegende Funktionalität des HYPERFILTER"""
    logger.info("Teste grundlegende HYPERFILTER-Funktionalität...")
    
    # Erstelle HYPERFILTER-Instanz
    filter_config = FilterConfig(mode=FilterMode.STRICT)
    hyperfilter = HyperFilter(filter_config)
    
    # Teste Textfilterung
    test_text = "Dies ist ein Test mit einem Passwort: 12345 und einem exec(code)-Befehl."
    filtered_text, metadata = hyperfilter.filter_input(test_text)
    
    logger.info(f"Original: {test_text}")
    logger.info(f"Gefiltert: {filtered_text}")
    logger.info(f"Metadaten: {json.dumps(metadata, indent=2)}")
    
    # Prüfe, ob die Filterung erfolgreich war
    assert metadata["filtered"], "Filterung sollte aktiviert sein"
    assert "exec(" in metadata["patterns_matched"], "exec(-Muster sollte erkannt werden"
    
    # Zeige die tatsächlich erkannten Muster an
    logger.info(f"Erkannte Muster: {metadata['patterns_matched']}")
    
    logger.info("Grundlegender HYPERFILTER-Test erfolgreich")

def test_vx_hyperfilter_adapter():
    """Testet den VX-HYPERFILTER-Adapter"""
    logger.info("Teste VX-HYPERFILTER-Adapter...")
    
    # Erstelle VX-HYPERFILTER-Adapter-Instanz
    config = VXHyperfilterConfig(
        language_analyzer_enabled=True,
        trust_validator_enabled=True,
        sentiment_engine_enabled=True,
        context_normalizer_enabled=True,
        filter_config=FilterConfig(mode=FilterMode.STRICT)
    )
    adapter = VXHyperfilterAdapter(config)
    
    # Teste Verarbeitung von vertrauenswürdigem Inhalt
    trusted_text = "Dies ist ein vertrauenswürdiger Text ohne problematische Inhalte."
    trusted_result = adapter.process_content(
        trusted_text,
        source_trust_score=0.9,
        language_code="de",
        media_source_type="News"
    )
    
    logger.info(f"Vertrauenswürdiger Text: {trusted_text}")
    logger.info(f"Ergebnis: {json.dumps(trusted_result, indent=2, default=str)}")
    
    # Prüfe, ob die Verarbeitung korrekt war
    assert trusted_result["signal_flag"] in ["HIGH", "MEDIUM"], "Vertrauensstufe sollte HIGH oder MEDIUM sein"
    assert not trusted_result["action_trigger"], "Kein Action-Trigger für vertrauenswürdigen Inhalt"
    
    # Teste Verarbeitung von problematischem Inhalt
    untrusted_text = "Dies ist ein problematischer Text mit exec(code) und einem Passwort: 12345."
    untrusted_result = adapter.process_content(
        untrusted_text,
        source_trust_score=0.3,
        language_code="de",
        media_source_type="Social"
    )
    
    logger.info(f"Problematischer Text: {untrusted_text}")
    logger.info(f"Ergebnis: {json.dumps(untrusted_result, indent=2, default=str)}")
    
    # Prüfe, ob die Verarbeitung korrekt war
    assert untrusted_result["signal_flag"] in ["LOW", "UNTRUSTED"], "Vertrauensstufe sollte LOW oder UNTRUSTED sein"
    assert untrusted_result["action_trigger"], "Action-Trigger für problematischen Inhalt"
    
    logger.info("VX-HYPERFILTER-Adapter-Test erfolgreich")

def test_vx_hyperfilter_singleton():
    """Testet die Singleton-Funktionalität des VX-HYPERFILTER-Adapters"""
    logger.info("Teste VX-HYPERFILTER-Adapter-Singleton...")
    
    # Setze Adapter zurück
    reset_vx_hyperfilter_adapter()
    
    # Erstelle erste Instanz
    adapter1 = get_vx_hyperfilter_adapter()
    
    # Erstelle zweite Instanz
    adapter2 = get_vx_hyperfilter_adapter()
    
    # Prüfe, ob es sich um dieselbe Instanz handelt
    assert adapter1 is adapter2, "Singleton-Funktionalität sollte dieselbe Instanz zurückgeben"
    
    # Aktualisiere Konfiguration
    new_config = VXHyperfilterConfig(
        language_analyzer_enabled=False,
        trust_validator_enabled=False,
        filter_config=FilterConfig(mode=FilterMode.PERMISSIVE)
    )
    adapter3 = get_vx_hyperfilter_adapter(new_config)
    
    # Prüfe, ob die Konfiguration aktualisiert wurde
    assert adapter3 is adapter1, "Singleton-Funktionalität sollte dieselbe Instanz zurückgeben"
    assert adapter3.config.language_analyzer_enabled is False, "Konfiguration sollte aktualisiert sein"
    assert adapter3.config.filter_config.mode == FilterMode.PERMISSIVE, "Filter-Modus sollte aktualisiert sein"
    
    logger.info("VX-HYPERFILTER-Adapter-Singleton-Test erfolgreich")

def test_vx_hyperfilter_capabilities():
    """Testet die Fähigkeiten des VX-HYPERFILTER-Adapters"""
    logger.info("Teste VX-HYPERFILTER-Adapter-Fähigkeiten...")
    
    # Erstelle VX-HYPERFILTER-Adapter-Instanz
    adapter = get_vx_hyperfilter_adapter()
    
    # Hole Fähigkeiten
    capabilities = adapter.get_capabilities()
    
    logger.info(f"Fähigkeiten: {json.dumps(capabilities, indent=2, default=str)}")
    
    # Prüfe, ob die Fähigkeiten korrekt sind
    assert "supported_languages" in capabilities, "Unterstützte Sprachen sollten in den Fähigkeiten enthalten sein"
    assert "supported_media_sources" in capabilities, "Unterstützte Medienquellen sollten in den Fähigkeiten enthalten sein"
    assert "hyperfilter_stats" in capabilities, "HYPERFILTER-Statistiken sollten in den Fähigkeiten enthalten sein"
    
    logger.info("VX-HYPERFILTER-Adapter-Fähigkeiten-Test erfolgreich")

def main():
    """Hauptfunktion"""
    logger.info("Starte Tests für VX-HYPERFILTER...")
    
    # Führe Tests aus
    test_hyperfilter_basic()
    test_vx_hyperfilter_adapter()
    test_vx_hyperfilter_singleton()
    test_vx_hyperfilter_capabilities()
    
    logger.info("Alle Tests für VX-HYPERFILTER erfolgreich abgeschlossen")

if __name__ == "__main__":
    main()
