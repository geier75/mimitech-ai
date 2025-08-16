#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: context_memory_adapter.py
Zugriff auf relevante Inhalte aus VX-MEMEX im VXOR-System

Dieses Modul dient als Adapter für den Zugriff auf das VX-MEMEX Gedächtnissystem.
Es ermöglicht die kontextbezogene Abfrage und Integration von Gedächtnisinhalten
in den aktuellen Kontext des VXOR-Systems.

Optimiert für: Python 3.10+, Apple Silicon (M4 Max)
"""

import os
import time
import logging
import threading
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from collections import defaultdict
import importlib
import inspect
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Logging-Konfiguration
LOG_DIR = "/home/ubuntu/VXOR_Logs/CONTEXT/"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}context_memory_adapter.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VX-CONTEXT.memory_adapter")

# Import von ContextCore-Definitionen
try:
    from context_core import ContextSource, ContextPriority, ContextData, get_context_core
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
    
    def get_context_core():
        return None

# Import von ContextState-Definitionen
try:
    from context_state import ContextStateType, ContextStateEntry, ContextSnapshot
except ImportError:
    logger.error("Konnte ContextState-Definitionen nicht importieren")
    # Fallback-Definitionen
    class ContextStateType(Enum):
        VISUAL = auto()
        AUDITORY = auto()
        LINGUISTIC = auto()
        EMOTIONAL = auto()
        COGNITIVE = auto()
        PHYSICAL = auto()
        TEMPORAL = auto()
        SOCIAL = auto()
        SYSTEM = auto()
        GLOBAL = auto()

    @dataclass
    class ContextStateEntry:
        value: Any
        source: ContextSource
        timestamp: float = field(default_factory=time.time)
        confidence: float = 1.0
        expiration: Optional[float] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class ContextSnapshot:
        timestamp: float
        state: Dict[ContextStateType, Dict[str, ContextStateEntry]]
        focus_state: Dict[Tuple, Any]
        global_metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryType(Enum):
    """Typen von Gedächtnisinhalten"""
    EPISODIC = auto()      # Episodisches Gedächtnis (Ereignisse, Erfahrungen)
    SEMANTIC = auto()      # Semantisches Gedächtnis (Fakten, Konzepte)
    PROCEDURAL = auto()    # Prozedurales Gedächtnis (Fähigkeiten, Prozeduren)
    WORKING = auto()       # Arbeitsgedächtnis (kurzfristige Informationen)
    ASSOCIATIVE = auto()   # Assoziatives Gedächtnis (Verknüpfungen)
    EMOTIONAL = auto()     # Emotionales Gedächtnis (emotionale Reaktionen)


class MemoryRelevance(Enum):
    """Relevanz von Gedächtnisinhalten für den aktuellen Kontext"""
    HIGH = auto()      # Hohe Relevanz
    MEDIUM = auto()    # Mittlere Relevanz
    LOW = auto()       # Niedrige Relevanz
    NONE = auto()      # Keine Relevanz


@dataclass
class MemoryQuery:
    """Abfrage an das Gedächtnissystem"""
    query_type: str  # Art der Abfrage (z.B. "semantic", "episodic", "associative")
    query_content: Dict[str, Any]  # Inhalt der Abfrage
    context_snapshot: Optional[ContextSnapshot] = None  # Optionaler Kontextsnapshot
    max_results: int = 10  # Maximale Anzahl von Ergebnissen
    relevance_threshold: float = 0.5  # Schwellenwert für Relevanz (0.0 bis 1.0)
    timeout_ms: int = 1000  # Timeout in Millisekunden
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryResult:
    """Ergebnis einer Gedächtnisabfrage"""
    memory_id: str  # Eindeutige ID des Gedächtnisinhalts
    memory_type: MemoryType  # Typ des Gedächtnisinhalts
    content: Dict[str, Any]  # Inhalt des Gedächtnisinhalts
    relevance: float  # Relevanz für den aktuellen Kontext (0.0 bis 1.0)
    confidence: float  # Konfidenz (0.0 bis 1.0)
    timestamp: float  # Zeitstempel der Erstellung des Gedächtnisinhalts
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextMemoryAdapter:
    """
    Adapter für den Zugriff auf das VX-MEMEX Gedächtnissystem
    
    Diese Klasse ist verantwortlich für:
    1. Kontextbezogene Abfrage von Gedächtnisinhalten
    2. Integration von Gedächtnisinhalten in den aktuellen Kontext
    3. Aktualisierung des Gedächtnisses basierend auf dem aktuellen Kontext
    4. Verwaltung der Relevanz von Gedächtnisinhalten für den aktuellen Kontext
    """
    
    def __init__(self, context_core=None):
        """
        Initialisiert den ContextMemoryAdapter
        
        Args:
            context_core: Referenz zum ContextCore-Modul, falls verfügbar
        """
        self.context_core = context_core or get_context_core()
        
        # VX-MEMEX-Schnittstelle
        self.memex_interface = None
        
        # Konfiguration
        self.config = {
            'auto_query_interval_ms': 5000,  # Intervall für automatische Abfragen
            'relevance_threshold': 0.6,      # Schwellenwert für Relevanz
            'max_query_results': 20,         # Maximale Anzahl von Ergebnissen pro Abfrage
            'query_timeout_ms': 2000,        # Timeout für Abfragen
            'memory_cache_size': 100,        # Größe des Gedächtnis-Caches
            'memory_cache_ttl_ms': 60000,    # TTL für Cache-Einträge
            'enable_auto_update': True,      # Automatische Aktualisierung des Gedächtnisses
            'enable_auto_query': True,       # Automatische Abfrage des Gedächtnisses
            'query_batch_size': 5            # Anzahl der gleichzeitigen Abfragen
        }
        
        # Cache für Gedächtnisinhalte
        self.memory_cache = {}
        self.memory_cache_timestamps = {}
        
        # Aktuelle Relevanz-Bewertungen
        self.current_relevance = {}
        
        # Abfrage-Historie
        self.query_history = []
        self.max_query_history_length = 100
        
        # Thread-Pool für asynchrone Verarbeitung
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Automatische Abfrage-Threads
        self.auto_query_thread = None
        self.active = False
        
        # Leistungsmetriken
        self.performance_metrics = {
            'queries_count': 0,
            'successful_queries_count': 0,
            'failed_queries_count': 0,
            'avg_query_time_ms': 0,
            'max_query_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_updates_count': 0
        }
        
        logger.info("ContextMemoryAdapter initialisiert")
        
        # Versuche, VX-MEMEX zu importieren
        self._try_import_memex()
    
    def _try_import_memex(self):
        """Versucht, das VX-MEMEX-Modul zu importieren"""
        try:
            # Versuche, das Modul zu importieren
            import sys
            memex_path = "/vXor_Modules/VX-MEMEX"
            if memex_path not in sys.path:
                sys.path.append(memex_path)
            
            # Versuche verschiedene mögliche Modulnamen
            module_names = ["vx_memex", "VX_MEMEX", "memex", "MEMEX"]
            
            for name in module_names:
                try:
                    module_spec = importlib.util.find_spec(name)
                    if module_spec:
                        memex_module = importlib.import_module(name)
                        
                        # Suche nach der Schnittstellen-Klasse
                        interface_class_names = ["MemexInterface", "MemoryInterface", "VXMemexInterface"]
                        
                        for class_name in interface_class_names:
                            if hasattr(memex_module, class_name):
                                interface_class = getattr(memex_module, class_name)
                                self.memex_interface = interface_class()
                                logger.info(f"VX-MEMEX-Schnittstelle gefunden: {class_name}")
                                return
                except ImportError:
                    continue
            
            logger.warning("Konnte VX-MEMEX-Schnittstelle nicht importieren, verwende Dummy-Implementierung")
            self._create_dummy_interface()
        
        except Exception as e:
            logger.error(f"Fehler beim Importieren von VX-MEMEX: {str(e)}", exc_info=True)
            self._create_dummy_interface()
    
    def _create_dummy_interface(self):
        """Erstellt eine Dummy-Implementierung der VX-MEMEX-Schnittstelle"""
        class DummyMemexInterface:
            def query_memory(self, query_type, query_content, **kwargs):
                logger.debug(f"Dummy-Abfrage: {query_type}, {query_content}")
                return []
            
            def store_memory(self, memory_type, content, **kwargs):
                logger.debug(f"Dummy-Speicherung: {memory_type}, {content}")
                return {"memory_id": f"dummy_{time.time()}"}
            
            def update_memory(self, memory_id, content, **kwargs):
                logger.debug(f"Dummy-Aktualisierung: {memory_id}, {content}")
                return True
            
            def get_memory(self, memory_id):
                logger.debug(f"Dummy-Abruf: {memory_id}")
                return None
        
        self.memex_interface = DummyMemexInterface()
        logger.info("Dummy-VX-MEMEX-Schnittstelle erstellt")
    
    def start(self):
        """Startet den ContextMemoryAdapter"""
        if self.active:
            logger.warning("ContextMemoryAdapter läuft bereits")
            return
        
        self.active = True
        
        # Starte automatische Abfrage-Thread, falls aktiviert
        if self.config['enable_auto_query']:
            self.auto_query_thread = threading.Thread(
                target=self._auto_query_loop,
                name="ContextMemoryAutoQueryThread",
                daemon=True
            )
            self.auto_query_thread.start()
        
        logger.info("ContextMemoryAdapter gestartet")
    
    def stop(self):
        """Stoppt den ContextMemoryAdapter"""
        if not self.active:
            logger.warning("ContextMemoryAdapter läuft nicht")
            return
        
        self.active = False
        
        # Stoppe automatische Abfrage-Thread
        if self.auto_query_thread:
            self.auto_query_thread.join(timeout=1.0)
        
        # Schließe den Thread-Pool
        self.thread_pool.shutdown(wait=False)
        
        logger.info("ContextMemoryAdapter gestoppt")
    
    def query_memory(self, query: MemoryQuery) -> List[MemoryResult]:
        """
        Fragt das Gedächtnissystem ab
        
        Args:
            query: Die Abfrage
            
        Returns:
            List[MemoryResult]: Die Ergebnisse
        """
        if not self.memex_interface:
            logger.warning("Keine VX-MEMEX-Schnittstelle verfügbar")
            return []
        
        start_time = time.time()
        
        # Füge Abfrage zur Historie hinzu
        self.query_history.append({
            "query": query,
            "timestamp": start_time
        })
        
        # Begrenze die Größe der Historie
        if len(self.query_history) > self.max_query_history_length:
            self.query_history = self.query_history[-self.max_query_history_length:]
        
        # Aktualisiere Metriken
        self.performance_metrics['queries_count'] += 1
        
        try:
            # Prüfe Cache
            cache_key = self._get_cache_key(query)
            
            if cache_key in self.memory_cache:
                cache_timestamp = self.memory_cache_timestamps.get(cache_key, 0)
                cache_age_ms = (time.time() - cache_timestamp) * 1000
                
                if cache_age_ms < self.config['memory_cache_ttl_ms']:
                    # Cache-Treffer
                    self.performance_metrics['cache_hits'] += 1
                    return self.memory_cache[cache_key]
            
            # Cache-Fehltreffer
            self.performance_metrics['cache_misses'] += 1
            
            # Führe Abfrage durch
            query_args = {
                'max_results': query.max_results,
                'relevance_threshold': query.relevance_threshold,
                'timeout_ms': query.timeout_ms,
                **query.metadata
            }
            
            if query.context_snapshot:
                # Extrahiere relevante Kontextinformationen aus dem Snapshot
                context_info = self._extract_context_info(query.context_snapshot)
                query_args['context_info'] = context_info
            
            raw_results = self.memex_interface.query_memory(
                query.query_type,
                query.query_content,
                **query_args
            )
            
            # Konvertiere Ergebnisse
            results = self._convert_memory_results(raw_results)
            
            # Speichere im Cache
            self.memory_cache[cache_key] = results
            self.memory_cache_timestamps[cache_key] = time.time()
            
            # Begrenze die Größe des Caches
            if len(self.memory_cache) > self.config['memory_cache_size']:
                # Entferne ältesten Eintrag
                oldest_key = min(
                    self.memory_cache_timestamps.keys(),
                    key=lambda k: self.memory_cache_timestamps[k]
                )
                del self.memory_cache[oldest_key]
                del self.memory_cache_timestamps[oldest_key]
            
            # Aktualisiere Metriken
            query_time_ms = (time.time() - start_time) * 1000
            self._update_query_metrics(query_time_ms, True)
            
            return results
        
        except Exception as e:
            logger.error(f"Fehler bei der Gedächtnisabfrage: {str(e)}", exc_info=True)
            
            # Aktualisiere Metriken
            query_time_ms = (time.time() - start_time) * 1000
            self._update_query_metrics(query_time_ms, False)
            
            return []
    
    def query_memory_async(self, query: MemoryQuery) -> asyncio.Future:
        """
        Fragt das Gedächtnissystem asynchron ab
        
        Args:
            query: Die Abfrage
            
        Returns:
            asyncio.Future: Future mit den Ergebnissen
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        def _query_and_set_result():
            try:
                results = self.query_memory(query)
                loop.call_soon_threadsafe(future.set_result, results)
            except Exception as e:
                loop.call_soon_threadsafe(future.set_exception, e)
        
        self.thread_pool.submit(_query_and_set_result)
        
        return future
    
 
(Content truncated due to size limit. Use line ranges to read in chunks)