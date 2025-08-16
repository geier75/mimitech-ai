#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: context_core.py
Zentrale Steuerungseinheit für Kontextmodelle im VXOR-System

Dieses Modul dient als Hauptsteuerungseinheit für die kontextuelle Analyse,
Situationsverarbeitung und dynamische Echtzeit-Fokusverlagerung im VXOR-System.
Es koordiniert die verschiedenen Komponenten des VX-CONTEXT Moduls und stellt
die Schnittstelle zu anderen VXOR-Modulen dar.

Optimiert für: Python 3.10+, Apple Silicon (M4 Max)
"""

import os
import time
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

# Logging-Konfiguration
import tempfile
LOG_DIR = os.path.join(tempfile.gettempdir(), "VXOR_Logs", "CONTEXT")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}context_core.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VX-CONTEXT.core")

# Konstanten für Performance-Optimierung
MAX_CONTEXT_QUEUE_SIZE = 1000
CONTEXT_PROCESSING_INTERVAL_MS = 10  # 10ms für Hochfrequenz-Verarbeitung
MAX_PROCESSING_TIME_MS = 100  # Maximale Verarbeitungszeit laut Anforderung


class ContextPriority(Enum):
    """Prioritätsstufen für Kontextverarbeitung"""
    CRITICAL = auto()  # Sofortige Verarbeitung erforderlich
    HIGH = auto()      # Hohe Priorität, zeitnah verarbeiten
    MEDIUM = auto()    # Mittlere Priorität
    LOW = auto()       # Niedrige Priorität, kann verzögert werden
    BACKGROUND = auto() # Hintergrundverarbeitung


class ContextSource(Enum):
    """Quellen für Kontextinformationen"""
    VISUAL = auto()     # VX-VISION
    LANGUAGE = auto()   # Sprachverarbeitung
    INTERNAL = auto()   # Systeminterne Zustände
    EXTERNAL = auto()   # Externe Sensordaten
    MEMORY = auto()     # VX-MEMEX
    EMOTION = auto()    # VX-EMO
    INTENT = auto()     # VX-INTENT
    REFLEX = auto()     # VX-REFLEX


@dataclass
class ContextData:
    """Datenstruktur für Kontextinformationen"""
    source: ContextSource
    priority: ContextPriority
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: Optional[float] = None
    
    def __post_init__(self):
        """Validierung nach Initialisierung"""
        if not isinstance(self.source, ContextSource):
            raise TypeError(f"source muss vom Typ ContextSource sein, nicht {type(self.source)}")
        if not isinstance(self.priority, ContextPriority):
            raise TypeError(f"priority muss vom Typ ContextPriority sein, nicht {type(self.priority)}")


class ContextProcessor:
    """Verarbeitet eingehende Kontextdaten und leitet sie an die entsprechenden Module weiter"""
    
    def __init__(self):
        self.context_queue = queue.PriorityQueue(maxsize=MAX_CONTEXT_QUEUE_SIZE)
        self.active = False
        self.processing_thread = None
        self.registered_handlers = {}
        self.current_context_state = {}
        self.performance_metrics = {
            'avg_processing_time_ms': 0,
            'max_processing_time_ms': 0,
            'processed_items_count': 0,
            'dropped_items_count': 0
        }
        logger.info("ContextProcessor initialisiert")
    
    def start(self):
        """Startet den Kontext-Verarbeitungsprozess"""
        if self.active:
            logger.warning("ContextProcessor läuft bereits")
            return
        
        self.active = True
        self.processing_thread = threading.Thread(
            target=self._process_context_queue,
            name="ContextProcessorThread",
            daemon=True
        )
        self.processing_thread.start()
        logger.info("ContextProcessor gestartet")
    
    def stop(self):
        """Stoppt den Kontext-Verarbeitungsprozess"""
        if not self.active:
            logger.warning("ContextProcessor läuft nicht")
            return
        
        self.active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            if self.processing_thread.is_alive():
                logger.warning("ContextProcessor Thread konnte nicht sauber beendet werden")
        
        logger.info("ContextProcessor gestoppt")
    
    def submit_context(self, context_data: ContextData) -> bool:
        """
        Fügt neue Kontextdaten zur Verarbeitung hinzu
        
        Args:
            context_data: Die zu verarbeitenden Kontextdaten
            
        Returns:
            bool: True wenn erfolgreich hinzugefügt, False sonst
        """
        if not self.active:
            logger.warning("Versuch, Kontext zu übermitteln, während ContextProcessor inaktiv ist")
            return False
        
        try:
            # Prioritätsqueue-Eintrag: (Prioritätswert, Zeitstempel für FIFO bei gleicher Priorität, Daten)
            priority_value = context_data.priority.value
            self.context_queue.put_nowait((priority_value, time.time(), context_data))
            return True
        except queue.Full:
            logger.error("Kontext-Queue ist voll, Kontext wird verworfen")
            self.performance_metrics['dropped_items_count'] += 1
            return False
    
    def register_handler(self, source: ContextSource, handler: Callable[[ContextData], None]):
        """
        Registriert einen Handler für eine bestimmte Kontextquelle
        
        Args:
            source: Die Kontextquelle
            handler: Die Handler-Funktion, die aufgerufen werden soll
        """
        if source not in self.registered_handlers:
            self.registered_handlers[source] = []
        
        self.registered_handlers[source].append(handler)
        logger.debug(f"Handler für {source.name} registriert")
    
    def _process_context_queue(self):
        """Interne Methode zur kontinuierlichen Verarbeitung der Kontext-Queue"""
        logger.info("Kontext-Verarbeitungsschleife gestartet")
        
        while self.active:
            try:
                # Warte auf neue Kontextdaten mit Timeout
                try:
                    _, _, context_data = self.context_queue.get(timeout=CONTEXT_PROCESSING_INTERVAL_MS/1000)
                except queue.Empty:
                    # Keine neuen Daten, weiter in der Schleife
                    continue
                
                # Verarbeite Kontextdaten
                start_time = time.time()
                
                # Aktualisiere den aktuellen Kontextzustand
                self._update_context_state(context_data)
                
                # Rufe registrierte Handler auf
                self._dispatch_to_handlers(context_data)
                
                # Markiere als verarbeitet
                self.context_queue.task_done()
                
                # Erfasse Performance-Metriken
                processing_time_ms = (time.time() - start_time) * 1000
                context_data.processing_time_ms = processing_time_ms
                
                self._update_performance_metrics(processing_time_ms)
                
                # Überprüfe, ob die Verarbeitungszeit zu lang war
                if processing_time_ms > MAX_PROCESSING_TIME_MS:
                    logger.warning(
                        f"Kontextverarbeitung überschreitet Zeitlimit: {processing_time_ms:.2f}ms "
                        f"(Limit: {MAX_PROCESSING_TIME_MS}ms)"
                    )
            
            except Exception as e:
                logger.error(f"Fehler bei der Kontextverarbeitung: {str(e)}", exc_info=True)
    
    def _update_context_state(self, context_data: ContextData):
        """Aktualisiert den aktuellen Kontextzustand basierend auf neuen Daten"""
        source_name = context_data.source.name
        
        # Erstelle oder aktualisiere den Kontextzustand für diese Quelle
        if source_name not in self.current_context_state:
            self.current_context_state[source_name] = {}
        
        # Aktualisiere die Daten
        self.current_context_state[source_name].update({
            'last_update': context_data.timestamp,
            'priority': context_data.priority.name,
            'data': context_data.data
        })
    
    def _dispatch_to_handlers(self, context_data: ContextData):
        """Leitet Kontextdaten an registrierte Handler weiter"""
        source = context_data.source
        
        # Rufe alle Handler für diese Quelle auf
        if source in self.registered_handlers:
            for handler in self.registered_handlers[source]:
                try:
                    handler(context_data)
                except Exception as e:
                    logger.error(f"Fehler im Handler für {source.name}: {str(e)}", exc_info=True)
    
    def _update_performance_metrics(self, processing_time_ms: float):
        """Aktualisiert die Performance-Metriken"""
        metrics = self.performance_metrics
        metrics['processed_items_count'] += 1
        
        # Aktualisiere maximale Verarbeitungszeit
        if processing_time_ms > metrics['max_processing_time_ms']:
            metrics['max_processing_time_ms'] = processing_time_ms
        
        # Aktualisiere durchschnittliche Verarbeitungszeit
        n = metrics['processed_items_count']
        metrics['avg_processing_time_ms'] = (
            (metrics['avg_processing_time_ms'] * (n - 1) + processing_time_ms) / n
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Gibt aktuelle Performance-Metriken zurück"""
        return self.performance_metrics.copy()
    
    def get_current_context_state(self) -> Dict[str, Any]:
        """Gibt den aktuellen Kontextzustand zurück"""
        return self.current_context_state.copy()


class ContextCore:
    """
    Hauptklasse für das VX-CONTEXT Modul
    
    Diese Klasse dient als zentrale Steuerungseinheit für das gesamte VX-CONTEXT Modul
    und koordiniert die Interaktion zwischen den verschiedenen Komponenten.
    """
    
    def __init__(self):
        """Initialisiert das ContextCore-Modul"""
        logger.info("Initialisiere VX-CONTEXT Core...")
        
        # Initialisiere den Kontext-Prozessor
        self.context_processor = ContextProcessor()
        
        # Platzhalter für andere Module, die später importiert werden
        self.focus_router = None
        self.context_analyzer = None
        self.context_bridge = None
        self.context_state = None
        self.context_memory_adapter = None
        
        # Konfiguration
        self.config = {
            'apple_silicon_optimized': True,
            'max_processing_time_ms': MAX_PROCESSING_TIME_MS,
            'processing_interval_ms': CONTEXT_PROCESSING_INTERVAL_MS,
        }
        
        logger.info("VX-CONTEXT Core initialisiert")
    
    def initialize_modules(self):
        """Initialisiert alle Untermodule des VX-CONTEXT Systems"""
        logger.info("Initialisiere VX-CONTEXT Untermodule...")
        
        # Hier werden die anderen Module importiert und initialisiert
        # Diese Importe werden hier platziert, um zirkuläre Abhängigkeiten zu vermeiden
        try:
            from focus_router import FocusRouter
            from context_analyzer import ContextAnalyzer
            from context_bridge import ContextBridge
            from context_state import ContextState
            from context_memory_adapter import ContextMemoryAdapter
            
            self.focus_router = FocusRouter(self)
            self.context_analyzer = ContextAnalyzer(self)
            self.context_bridge = ContextBridge(self)
            self.context_state = ContextState(self)
            self.context_memory_adapter = ContextMemoryAdapter(self)
            
            # Registriere Handler für verschiedene Kontextquellen
            self._register_default_handlers()
            
            logger.info("Alle VX-CONTEXT Untermodule erfolgreich initialisiert")
            return True
        
        except ImportError as e:
            logger.error(f"Fehler beim Importieren von VX-CONTEXT Untermodulen: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung von VX-CONTEXT Untermodulen: {str(e)}", exc_info=True)
            return False
    
    def _register_default_handlers(self):
        """Registriert Standard-Handler für verschiedene Kontextquellen"""
        # Diese werden später durch die tatsächlichen Handler ersetzt
        for source in ContextSource:
            self.context_processor.register_handler(
                source, 
                lambda data, src=source: logger.debug(f"Standardhandler für {src.name}: {data}")
            )
    
    def start(self):
        """Startet das VX-CONTEXT System"""
        logger.info("Starte VX-CONTEXT System...")
        
        # Starte den Kontext-Prozessor
        self.context_processor.start()
        
        # Starte andere Module, falls sie initialisiert wurden
        if self.focus_router:
            self.focus_router.start()
        
        logger.info("VX-CONTEXT System gestartet")
        return True
    
    def stop(self):
        """Stoppt das VX-CONTEXT System"""
        logger.info("Stoppe VX-CONTEXT System...")
        
        # Stoppe andere Module, falls sie initialisiert wurden
        if self.focus_router:
            self.focus_router.stop()
        
        # Stoppe den Kontext-Prozessor
        self.context_processor.stop()
        
        logger.info("VX-CONTEXT System gestoppt")
        return True
    
    def submit_context(self, source: ContextSource, data: Dict[str, Any], 
                       priority: ContextPriority = ContextPriority.MEDIUM,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Übermittelt neue Kontextdaten zur Verarbeitung
        
        Args:
            source: Die Quelle der Kontextdaten
            data: Die eigentlichen Kontextdaten
            priority: Die Priorität der Daten (Standard: MEDIUM)
            metadata: Optionale Metadaten
            
        Returns:
            bool: True wenn erfolgreich übermittelt, False sonst
        """
        if metadata is None:
            metadata = {}
        
        context_data = ContextData(
            source=source,
            priority=priority,
            data=data,
            metadata=metadata
        )
        
        return self.context_processor.submit_context(context_data)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Gibt aktuelle Performance-Metriken zurück"""
        return self.context_processor.get_performance_metrics()
    
    def get_current_context_state(self) -> Dict[str, Any]:
        """Gibt den aktuellen Kontextzustand zurück"""
        return self.context_processor.get_current_context_state()


# Singleton-Instanz für globalen Zugriff
_context_core_instance = None

def get_context_core() -> ContextCore:
    """Gibt die Singleton-Instanz von ContextCore zurück"""
    global _context_core_instance
    
    if _context_core_instance is None:
        _context_core_instance = ContextCore()
    
    return _context_core_instance


if __name__ == "__main__":
    # Einfacher Test des Moduls
    logger.info("Starte VX-CONTEXT Core Test...")
    
    core = get_context_core()
    core.start()
    
    # Simuliere einige Kontextdaten
    core.submit_context(
        source=ContextSource.VISUAL,
        data={"object_detected": "Person", "confidence": 0.95},
        priority=ContextPriority.HIGH
    )
    
    core.submit_context(
        source=ContextSource.INTERNAL,
        data={"system_load": 0.75, "memory_usage": 0.65},
        priority=ContextPriority.LOW
    )
    
    # Warte kurz, damit die Daten verarbeitet werden
    time.sleep(0.1)
    
    # Zeige aktive Kontexte
    print(f"Aktive Kontexte: {len(core.get_active_contexts())}")


if __name__ == "__main__":
    test_context_system()