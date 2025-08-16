#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Deep-State-Modul Kern

Dieses Modul implementiert den Kern des DeepStateAnalyzers für MISO Ultimate.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.analysis.deep_state.core")

# Importiere übergeordnete Module
from ..deep_state import DeepStateAnalyzer, DeepStateConfig, AnalysisResult, ReactionType
from ..deep_state_patterns import PatternMatcher, ControlPattern
from ..deep_state_network import NetworkAnalyzer, NetworkNode
from ..deep_state_security import SecurityManager, EncryptionLevel
from ..vxor_deepstate_integration import VXDeepStateAdapter, get_vx_deepstate_adapter

class DeepStateCore:
    """
    Kernimplementierung des Deep-State-Moduls
    
    Diese Klasse dient als Einstiegspunkt für das Deep-State-Modul und
    stellt die Verbindung zwischen den verschiedenen Komponenten her.
    """
    
    def __init__(self):
        """Initialisiert den DeepStateCore"""
        self.analyzer = DeepStateAnalyzer()
        self.vx_adapter = get_vx_deepstate_adapter()
        logger.info("DeepStateCore initialisiert")
    
    def analyze(self, 
                content_stream: str, 
                source_id: str, 
                source_trust_level: float, 
                language_code: str, 
                context_cluster: str, 
                timeframe: Optional[datetime.datetime] = None) -> AnalysisResult:
        """
        Führt eine Deep-State-Analyse durch
        
        Args:
            content_stream: Eingehender Inhalt
            source_id: Eindeutige Quell-ID
            source_trust_level: Vertrauensbewertung der Quelle
            language_code: Sprachcode (z.B. "DE", "EN")
            context_cluster: Themen- oder Sektor-Kontext
            timeframe: Zeitliche Einordnung
            
        Returns:
            Ergebnis der Analyse
        """
        return self.analyzer.analyze(
            content_stream=content_stream,
            source_id=source_id,
            source_trust_level=source_trust_level,
            language_code=language_code,
            context_cluster=context_cluster,
            timeframe=timeframe
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Gibt die Fähigkeiten des Deep-State-Moduls zurück
        
        Returns:
            Fähigkeiten als Wörterbuch
        """
        return self.analyzer.get_capabilities()
    
    def get_vx_adapter(self) -> VXDeepStateAdapter:
        """
        Gibt den VXOR-Adapter für das Deep-State-Modul zurück
        
        Returns:
            VXOR-Adapter
        """
        return self.vx_adapter

# Globale Instanz
_DEEP_STATE_CORE = None

def get_deep_state_core() -> DeepStateCore:
    """
    Gibt die globale DeepStateCore-Instanz zurück
    
    Returns:
        DeepStateCore-Instanz
    """
    global _DEEP_STATE_CORE
    
    if _DEEP_STATE_CORE is None:
        _DEEP_STATE_CORE = DeepStateCore()
    
    return _DEEP_STATE_CORE
