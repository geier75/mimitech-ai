#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - VXOR-DeepState-Integration

Dieses Modul implementiert die Integration des Deep-State-Moduls mit dem VXOR-System.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import datetime
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.analysis.vxor_deepstate_integration")

# Versuche, VXOR-Module zu importieren
try:
    from miso.vxor_integration import VXORAdapter, get_vxor_adapter
    VXOR_AVAILABLE = True
except ImportError:
    VXOR_AVAILABLE = False
    logger.warning("VXOR-Integration nicht verfügbar, einige Funktionen sind eingeschränkt")

# Importiere Deep-State-Module
from .deep_state import DeepStateAnalyzer, DeepStateConfig, AnalysisResult, ReactionType, get_deep_state_analyzer

class VXDeepStateAdapter:
    """
    VXOR-Adapter für das Deep-State-Modul
    
    Diese Klasse implementiert den VXOR-Adapter für das Deep-State-Modul.
    Sie stellt die Schnittstelle zwischen dem Deep-State-Modul und dem VXOR-System dar.
    """
    
    def __init__(self):
        """Initialisiert den VXDeepStateAdapter"""
        self.vxor_adapter = None
        self.deep_state_analyzer = None
        
        # Initialisiere VXOR-Adapter, falls verfügbar
        if VXOR_AVAILABLE:
            try:
                self.vxor_adapter = get_vxor_adapter()
                logger.info("VXOR-Adapter erfolgreich initialisiert")
            except Exception as e:
                logger.error(f"Fehler bei der Initialisierung des VXOR-Adapters: {e}")
        
        # Initialisiere Deep-State-Analyzer
        try:
            self.deep_state_analyzer = get_deep_state_analyzer()
            logger.info("Deep-State-Analyzer erfolgreich initialisiert")
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des Deep-State-Analyzers: {e}")
        
        # Registriere VX-DEEPSTATE-Modul bei VXOR
        if self.vxor_adapter and self.deep_state_analyzer:
            self._register_with_vxor()
    
    def _register_with_vxor(self):
        """Registriert das VX-DEEPSTATE-Modul bei VXOR"""
        try:
            # Registriere das VX-DEEPSTATE-Modul als VXOR-Modul
            self.vxor_adapter.register_module(
                module_id="VX-DEEPSTATE",
                module_type="STRUKTURINTELLIGENZ_ANALYSE",
                module_version="1.0",
                module_description="Autonomer Deep-State-Analyse-Agent unter Omega-Kontrolle",
                module_capabilities=[
                    "deep_state_analysis",
                    "propaganda_detection",
                    "network_analysis",
                    "influence_pattern_detection"
                ],
                module_dependencies=[
                    "VX-MEMEX",
                    "VX-REASON",
                    "VX-INTENT",
                    "VX-CONTEXT",
                    "QLOGIK_CORE",
                    "VX-HYPERFILTER",
                    "T-MATHEMATICS"
                ]
            )
            
            # Registriere Funktionen
            self.vxor_adapter.register_function(
                module_id="VX-DEEPSTATE",
                function_id="analyze_content",
                function_description="Analysiert Inhalte auf Deep-State-Muster",
                function_handler=self.analyze_content
            )
            
            self.vxor_adapter.register_function(
                module_id="VX-DEEPSTATE",
                function_id="get_network_analysis",
                function_description="Führt eine Netzwerkanalyse für eine Quelle durch",
                function_handler=self.get_network_analysis
            )
            
            self.vxor_adapter.register_function(
                module_id="VX-DEEPSTATE",
                function_id="get_capabilities",
                function_description="Gibt die Fähigkeiten des VX-DEEPSTATE-Moduls zurück",
                function_handler=self.get_capabilities
            )
            
            logger.info("VX-DEEPSTATE-Modul erfolgreich bei VXOR registriert")
        except Exception as e:
            logger.error(f"Fehler bei der Registrierung des VX-DEEPSTATE-Moduls bei VXOR: {e}")
    
    def analyze_content(self, 
                       content_stream: str, 
                       source_id: str, 
                       source_trust_level: float, 
                       language_code: str, 
                       context_cluster: str, 
                       timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Analysiert Inhalte auf Deep-State-Muster
        
        Args:
            content_stream: Eingehender Inhalt
            source_id: Eindeutige Quell-ID
            source_trust_level: Vertrauensbewertung der Quelle
            language_code: Sprachcode (z.B. "DE", "EN")
            context_cluster: Themen- oder Sektor-Kontext
            timeframe: Zeitliche Einordnung als ISO-Zeitstempel
            
        Returns:
            Analyseergebnis als Wörterbuch
        """
        if not self.deep_state_analyzer:
            logger.error("Deep-State-Analyzer nicht initialisiert")
            return {"error": "Deep-State-Analyzer nicht initialisiert"}
        
        # Konvertiere timeframe in datetime
        if timeframe:
            try:
                timeframe_dt = datetime.datetime.fromisoformat(timeframe)
            except ValueError:
                logger.warning(f"Ungültiges Zeitformat: {timeframe}, verwende aktuellen Zeitpunkt")
                timeframe_dt = datetime.datetime.now()
        else:
            timeframe_dt = datetime.datetime.now()
        
        # Führe Analyse durch
        result = self.deep_state_analyzer.analyze(
            content_stream=content_stream,
            source_id=source_id,
            source_trust_level=source_trust_level,
            language_code=language_code,
            context_cluster=context_cluster,
            timeframe=timeframe_dt
        )
        
        # Konvertiere Ergebnis in Wörterbuch
        return self._convert_result_to_dict(result)
    
    def get_network_analysis(self, source_id: str, context_cluster: str = "") -> Dict[str, Any]:
        """
        Führt eine Netzwerkanalyse für eine Quelle durch
        
        Args:
            source_id: ID der Quelle
            context_cluster: Kontext-Cluster für die Filterung der Knoten
            
        Returns:
            Netzwerkanalyse als Wörterbuch
        """
        if not self.deep_state_analyzer:
            logger.error("Deep-State-Analyzer nicht initialisiert")
            return {"error": "Deep-State-Analyzer nicht initialisiert"}
        
        # Führe Netzwerkanalyse durch
        network_score = self.deep_state_analyzer.network_analyzer.analyze_network(source_id, context_cluster)
        potential_connections = self.deep_state_analyzer.network_analyzer.get_potential_connections(source_id, context_cluster)
        
        return {
            "network_score": network_score,
            "potential_connections": potential_connections,
            "source_id": source_id,
            "context_cluster": context_cluster,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Gibt die Fähigkeiten des VX-DEEPSTATE-Moduls zurück
        
        Returns:
            Fähigkeiten als Wörterbuch
        """
        if not self.deep_state_analyzer:
            logger.error("Deep-State-Analyzer nicht initialisiert")
            return {"error": "Deep-State-Analyzer nicht initialisiert"}
        
        return self.deep_state_analyzer.get_capabilities()
    
    def _convert_result_to_dict(self, result: AnalysisResult) -> Dict[str, Any]:
        """
        Konvertiert ein AnalysisResult in ein Wörterbuch
        
        Args:
            result: Zu konvertierendes AnalysisResult
            
        Returns:
            Wörterbuch mit dem Analyseergebnis
        """
        return {
            "report_text": result.report_text,
            "ds_probability": result.ds_probability,
            "reaction": result.reaction.name,
            "reaction_text": result.reaction_text,
            "paradox_signal": result.paradox_signal,
            "bias_score": result.bias_score,
            "pattern_score": result.pattern_score,
            "network_score": result.network_score,
            "source_id": result.source_id,
            "timestamp": result.timestamp.isoformat(),
            "encrypted": result.encrypted,
            "encryption_key_id": result.encryption_key_id
        }

# Globale Instanz
_VX_DEEPSTATE_ADAPTER = None

def get_vx_deepstate_adapter() -> VXDeepStateAdapter:
    """
    Gibt die globale VXDeepStateAdapter-Instanz zurück
    
    Returns:
        VXDeepStateAdapter-Instanz
    """
    global _VX_DEEPSTATE_ADAPTER
    
    if _VX_DEEPSTATE_ADAPTER is None:
        _VX_DEEPSTATE_ADAPTER = VXDeepStateAdapter()
    
    return _VX_DEEPSTATE_ADAPTER

def reset_vx_deepstate_adapter() -> None:
    """Setzt die globale VXDeepStateAdapter-Instanz zurück"""
    global _VX_DEEPSTATE_ADAPTER
    _VX_DEEPSTATE_ADAPTER = None
    logger.info("Globale VXDeepStateAdapter-Instanz zurückgesetzt")
