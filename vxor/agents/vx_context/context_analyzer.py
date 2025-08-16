#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: context_analyzer.py
Klassifikation und Bewertung eingehender Informationen im VXOR-System

Dieses Modul ist verantwortlich für die Analyse und Klassifikation von Kontextinformationen,
die Bewertung ihrer Relevanz und die Priorisierung basierend auf verschiedenen Kriterien
wie Wichtigkeit, Dringlichkeit und Quelle.

Optimiert für: Python 3.10+, Apple Silicon (M4 Max)
"""

import os
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from collections import defaultdict

# Logging-Konfiguration
LOG_DIR = "/home/ubuntu/VXOR_Logs/CONTEXT/"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}context_analyzer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VX-CONTEXT.analyzer")

# Import von ContextCore-Definitionen
try:
    from context_core import ContextSource, ContextPriority, ContextData
except ImportError:
    logger.error("Konnte ContextCore-Definitionen nicht importieren")
    # Fallback-Definitionen
    class ContextSource(Enum):
        VISUAL = auto()
        LANGUAGE = auto()
        INTERNAL = auto()
        EXTERNAL = auto()
        MEMORY = auto()
        EMOTION = auto()
        INTENT = auto()
        REFLEX = auto()

    class ContextPriority(Enum):
        CRITICAL = auto()
        HIGH = auto()
        MEDIUM = auto()
        LOW = auto()
        BACKGROUND = auto()

    @dataclass
    class ContextData:
        source: ContextSource
        priority: ContextPriority
        timestamp: float = field(default_factory=time.time)
        data: Dict[str, Any] = field(default_factory=dict)
        metadata: Dict[str, Any] = field(default_factory=dict)
        processing_time_ms: Optional[float] = None


class AnalysisResult(Enum):
    """Ergebnistypen der Kontextanalyse"""
    RELEVANT = auto()       # Relevant für den aktuellen Kontext
    IRRELEVANT = auto()     # Irrelevant für den aktuellen Kontext
    IMPORTANT = auto()      # Wichtig, unabhängig vom aktuellen Kontext
    URGENT = auto()         # Dringend, erfordert sofortige Aufmerksamkeit
    NOVEL = auto()          # Neuartig, bisher nicht gesehen
    REDUNDANT = auto()      # Redundant, bereits bekannt
    ANOMALOUS = auto()      # Anomal, weicht stark vom erwarteten Muster ab
    EXPECTED = auto()       # Erwartet, entspricht dem erwarteten Muster


class AnalysisDimension(Enum):
    """Dimensionen für die Kontextanalyse"""
    RELEVANCE = auto()      # Relevanz für den aktuellen Kontext
    IMPORTANCE = auto()     # Wichtigkeit der Information
    URGENCY = auto()        # Dringlichkeit der Information
    NOVELTY = auto()        # Neuartigkeit der Information
    ANOMALY = auto()        # Anomalie der Information


@dataclass
class AnalysisConfig:
    """Konfiguration für die Kontextanalyse"""
    relevance_threshold: float = 0.6  # Schwellenwert für Relevanz (0.0 bis 1.0)
    importance_threshold: float = 0.7  # Schwellenwert für Wichtigkeit (0.0 bis 1.0)
    urgency_threshold: float = 0.8     # Schwellenwert für Dringlichkeit (0.0 bis 1.0)
    novelty_threshold: float = 0.5     # Schwellenwert für Neuartigkeit (0.0 bis 1.0)
    anomaly_threshold: float = 0.7     # Schwellenwert für Anomalie (0.0 bis 1.0)
    
    # Gewichtungen für verschiedene Quellen (0.0 bis 1.0)
    source_weights: Dict[ContextSource, float] = field(default_factory=lambda: {
        ContextSource.VISUAL: 0.8,
        ContextSource.LANGUAGE: 0.7,
        ContextSource.INTERNAL: 0.9,
        ContextSource.EXTERNAL: 0.6,
        ContextSource.MEMORY: 0.5,
        ContextSource.EMOTION: 0.8,
        ContextSource.INTENT: 0.9,
        ContextSource.REFLEX: 1.0
    })
    
    # Prioritätszuordnungen basierend auf Analyseergebnissen
    priority_mappings: Dict[AnalysisResult, ContextPriority] = field(default_factory=lambda: {
        AnalysisResult.URGENT: ContextPriority.CRITICAL,
        AnalysisResult.IMPORTANT: ContextPriority.HIGH,
        AnalysisResult.RELEVANT: ContextPriority.MEDIUM,
        AnalysisResult.NOVEL: ContextPriority.MEDIUM,
        AnalysisResult.ANOMALOUS: ContextPriority.HIGH,
        AnalysisResult.EXPECTED: ContextPriority.LOW,
        AnalysisResult.REDUNDANT: ContextPriority.BACKGROUND,
        AnalysisResult.IRRELEVANT: ContextPriority.BACKGROUND
    })


@dataclass
class AnalysisResult:
    """Ergebnis einer Kontextanalyse"""
    context_data: ContextData
    scores: Dict[AnalysisDimension, float]
    results: List[AnalysisResult]
    adjusted_priority: ContextPriority
    analysis_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextAnalyzer:
    """
    Analysiert und bewertet eingehende Kontextinformationen
    
    Diese Klasse ist verantwortlich für:
    1. Klassifikation eingehender Informationen nach verschiedenen Dimensionen
    2. Bewertung der Relevanz, Wichtigkeit und Dringlichkeit
    3. Anpassung der Priorität basierend auf Analyseergebnissen
    4. Erkennung von Anomalien und neuartigen Informationen
    """
    
    def __init__(self, context_core=None):
        """
        Initialisiert den ContextAnalyzer
        
        Args:
            context_core: Referenz zum ContextCore-Modul, falls verfügbar
        """
        self.context_core = context_core
        self.config = AnalysisConfig()
        
        # Historische Daten für Neuartigkeits- und Anomalieanalyse
        self.historical_data = defaultdict(list)
        self.max_history_length = 1000  # Maximale Länge der Historie pro Quelle
        
        # Registrierte Analysefunktionen
        self.analysis_functions = {
            AnalysisDimension.RELEVANCE: self._analyze_relevance,
            AnalysisDimension.IMPORTANCE: self._analyze_importance,
            AnalysisDimension.URGENCY: self._analyze_urgency,
            AnalysisDimension.NOVELTY: self._analyze_novelty,
            AnalysisDimension.ANOMALY: self._analyze_anomaly
        }
        
        # Benutzerdefinierte Analysefunktionen
        self.custom_analysis_functions = {}
        
        # Leistungsmetriken
        self.performance_metrics = {
            'avg_analysis_time_ms': 0,
            'max_analysis_time_ms': 0,
            'analyzed_items_count': 0,
            'priority_upgrades_count': 0,
            'priority_downgrades_count': 0
        }
        
        # Statistiken pro Quelle und Dimension
        self.source_stats = defaultdict(lambda: {
            'count': 0,
            'avg_scores': {dim: 0.0 for dim in AnalysisDimension},
            'priority_distribution': {priority: 0 for priority in ContextPriority}
        })
        
        logger.info("ContextAnalyzer initialisiert")
        
        # Registriere Handler beim ContextCore, falls verfügbar
        if self.context_core:
            try:
                for source in ContextSource:
                    self.context_core.context_processor.register_handler(
                        source, self.process_context_data
                    )
                logger.info("ContextAnalyzer-Handler bei ContextCore registriert")
            except Exception as e:
                logger.error(f"Fehler bei der Registrierung von Handlern: {str(e)}", exc_info=True)
    
    def process_context_data(self, context_data: ContextData) -> AnalysisResult:
        """
        Verarbeitet und analysiert eingehende Kontextdaten
        
        Args:
            context_data: Die zu analysierenden Kontextdaten
            
        Returns:
            AnalysisResult: Das Ergebnis der Analyse
        """
        start_time = time.time()
        
        # Analysiere die Daten in allen Dimensionen
        scores = {}
        results = []
        
        for dimension, analysis_func in self.analysis_functions.items():
            try:
                score = analysis_func(context_data)
                scores[dimension] = score
                
                # Bestimme das Ergebnis basierend auf dem Schwellenwert
                threshold = getattr(self.config, f"{dimension.name.lower()}_threshold")
                if score >= threshold:
                    if dimension == AnalysisDimension.RELEVANCE:
                        results.append(AnalysisResult.RELEVANT)
                    elif dimension == AnalysisDimension.IMPORTANCE:
                        results.append(AnalysisResult.IMPORTANT)
                    elif dimension == AnalysisDimension.URGENCY:
                        results.append(AnalysisResult.URGENT)
                    elif dimension == AnalysisDimension.NOVELTY:
                        results.append(AnalysisResult.NOVEL)
                    elif dimension == AnalysisDimension.ANOMALY:
                        results.append(AnalysisResult.ANOMALOUS)
                else:
                    if dimension == AnalysisDimension.RELEVANCE:
                        results.append(AnalysisResult.IRRELEVANT)
                    elif dimension == AnalysisDimension.NOVELTY:
                        results.append(AnalysisResult.REDUNDANT)
                    elif dimension == AnalysisDimension.ANOMALY:
                        results.append(AnalysisResult.EXPECTED)
            except Exception as e:
                logger.error(f"Fehler bei der Analyse in Dimension {dimension.name}: {str(e)}", exc_info=True)
                scores[dimension] = 0.0
        
        # Führe benutzerdefinierte Analysen durch
        for name, (dimension, func) in self.custom_analysis_functions.items():
            try:
                score = func(context_data)
                scores[dimension] = score
            except Exception as e:
                logger.error(f"Fehler bei der benutzerdefinierten Analyse {name}: {str(e)}", exc_info=True)
        
        # Bestimme die angepasste Priorität
        adjusted_priority = self._determine_priority(results, context_data.source)
        
        # Aktualisiere die historischen Daten
        self._update_historical_data(context_data)
        
        # Aktualisiere die Statistiken
        self._update_statistics(context_data.source, scores, adjusted_priority)
        
        # Erfasse Performance-Metriken
        analysis_time_ms = (time.time() - start_time) * 1000
        self._update_performance_metrics(analysis_time_ms)
        
        # Erstelle das Analyseergebnis
        analysis_result = AnalysisResult(
            context_data=context_data,
            scores=scores,
            results=results,
            adjusted_priority=adjusted_priority,
            analysis_time_ms=analysis_time_ms,
            metadata={
                "original_priority": context_data.priority.name,
                "adjusted_priority": adjusted_priority.name,
                "analysis_timestamp": time.time()
            }
        )
        
        # Passe die Priorität an, falls sie sich geändert hat
        if adjusted_priority != context_data.priority and self.context_core:
            try:
                # Erstelle eine Kopie der Daten mit angepasster Priorität
                adjusted_data = ContextData(
                    source=context_data.source,
                    priority=adjusted_priority,
                    timestamp=context_data.timestamp,
                    data=context_data.data.copy(),
                    metadata={**context_data.metadata, "priority_adjusted_by": "context_analyzer"}
                )
                
                # Übermittle die angepassten Daten an den ContextCore
                self.context_core.context_processor.submit_context(adjusted_data)
                
                # Aktualisiere die Zähler für Prioritätsänderungen
                if adjusted_priority.value > context_data.priority.value:
                    self.performance_metrics['priority_upgrades_count'] += 1
                else:
                    self.performance_metrics['priority_downgrades_count'] += 1
                
                logger.debug(
                    f"Priorität angepasst: {context_data.priority.name} -> {adjusted_priority.name} "
                    f"für Quelle {context_data.source.name}"
                )
            except Exception as e:
                logger.error(f"Fehler bei der Anpassung der Priorität: {str(e)}", exc_info=True)
        
        return analysis_result
    
    def register_custom_analysis(self, name: str, dimension: AnalysisDimension, 
                                func: Callable[[ContextData], float]):
        """
        Registriert eine benutzerdefinierte Analysefunktion
        
        Args:
            name: Name der Analysefunktion
            dimension: Die Analysedimension
            func: Die Analysefunktion, die einen Wert zwischen 0.0 und 1.0 zurückgibt
        """
        self.custom_analysis_functions[name] = (dimension, func)
        logger.debug(f"Benutzerdefinierte Analysefunktion '{name}' für {dimension.name} registriert")
    
    def _analyze_relevance(self, context_data: ContextData) -> float:
        """
        Analysiert die Relevanz der Kontextdaten für den aktuellen Kontext
        
        Args:
            context_data: Die zu analysierenden Kontextdaten
            
        Returns:
            float: Relevanzwert zwischen 0.0 und 1.0
        """
        # Basisrelevanz basierend auf der Quelle
        base_relevance = self.config.source_weights.get(context_data.source, 0.5)
        
        # TODO: Implementiere fortgeschrittene Relevanzanalyse basierend auf dem aktuellen Kontext
        # Hier könnte eine Vektorähnlichkeit zum aktuellen Kontextzustand berechnet werden
        
        # Für jetzt verwenden wir einen einfachen Ansatz
        if "relevance_score" in context_data.metadata:
            # Verwende vorberechnete Relevanz, falls vorhanden
            return min(1.0, max(0.0, float(context_data.metadata["relevance_score"])))
        
        # Berücksichtige die Priorität in der Relevanzberechnung
        priority_factor = {
            ContextPriority.CRITICAL: 1.0,
            ContextPriority.HIGH: 0.8,
            ContextPriority.MEDIUM: 0.6,
            ContextPriority.LOW: 0.4,
            ContextPriority.BACKGROUND: 0.2
        }.get(context_data.priority, 0.5)
        
        # Kombiniere die Faktoren
        relevance = base_relevance * 0.6 + priority_factor * 0.4
        
        return min(1.0, max(0.0, relevance))
    
    def _analyze_importance(self, context_data: ContextData) -> float:
        """
        Analysiert die Wichtigkeit der Kontextdaten
        
        Args:
            context_data: Die zu analysierenden Kontextdaten
            
        Returns:
            float: Wichtigkeitswert zwischen 0.0 und 1.0
        """
        # Basiswichtigkeit basierend auf der Quelle
        base_importance = self.config.source_weights.get(context_data.source, 0.5)
        
        # Verwende vorberechnete Wichtigkeit, falls vorhanden
        if "importance_score" in context_data.metadata:
            return min(1.0, max(0.0, float(context_data.metadata["importance_score"])))
        
        # Berücksichtige spezifische Schlüsselwörter oder Muster in den Daten
        importance_indicators = {
            "error": 0.9,
            "warning": 0.7,
            "critical": 1.0,
            "important": 0.8,
            "alert": 0.9,
            "notification": 0.6
        }
        
        # Suche nach Indikatoren in den Daten
        data_str = str(context_data.data).lower()
        for indicator, value in importance_indicators.items():
            if indicator in data_str:
                base_importance = max(base_importance, value)
        
        # Berücksichtige die Priorität
        priority_factor = {
            ContextPriority.CRITICAL: 1.0,
            ContextPrior
(Content truncated due to size limit. Use line ranges to read in chunks)