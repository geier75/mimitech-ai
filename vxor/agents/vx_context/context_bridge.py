#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: context_bridge.py
VXOR-Schnittstelle für VX-PSI, VX-EMO, VX-INTENT, VX-VISION

Dieses Modul dient als Brücke zwischen dem VX-CONTEXT Modul und anderen VXOR-Modulen.
Es ermöglicht die bidirektionale Kommunikation und Kontextweitergabe zwischen den Modulen
und sorgt für eine konsistente Verarbeitung von Kontextinformationen im gesamten System.

Optimiert für: Python 3.10+, Apple Silicon (M4 Max)
"""

import os
import time
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import json
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
        logging.FileHandler(f"{LOG_DIR}context_bridge.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VX-CONTEXT.bridge")

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

# Import von FocusRouter-Definitionen
try:
    from focus_router import FocusType, FocusTarget, FocusState, FocusCommand
except ImportError:
    logger.error("Konnte FocusRouter-Definitionen nicht importieren")
    # Fallback-Definitionen
    class FocusType(Enum):
        COGNITIVE = auto()
        SENSORY = auto()
        MOTOR = auto()
        EMOTIONAL = auto()
        MEMORY = auto()

    class FocusTarget(Enum):
        VISION = auto()
        AUDIO = auto()
        LANGUAGE = auto()
        MEMORY = auto()
        EMOTION = auto()
        PLANNING = auto()
        REASONING = auto()
        MOTOR = auto()
        AGENT = auto()

    @dataclass
    class FocusState:
        target: Union[FocusTarget, str]
        focus_type: FocusType
        intensity: float
        timestamp: float = field(default_factory=time.time)
        metadata: Dict[str, Any] = field(default_factory=dict)
        agent_id: Optional[str] = None

    @dataclass
    class FocusCommand:
        target: Union[FocusTarget, str]
        focus_type: FocusType
        intensity_delta: float
        priority: int
        source: str
        timestamp: float = field(default_factory=time.time)
        metadata: Dict[str, Any] = field(default_factory=dict)


class ModuleInterface(Enum):
    """Unterstützte Modulschnittstellen"""
    PSI = auto()      # VX-PSI (Perception, Sensation, Interpretation)
    EMO = auto()      # VX-EMO (Emotion)
    INTENT = auto()   # VX-INTENT (Intention)
    VISION = auto()   # VX-VISION (Vision)
    MEMEX = auto()    # VX-MEMEX (Memory)
    REFLEX = auto()   # VX-REFLEX (Reflex)


@dataclass
class ModuleConfig:
    """Konfiguration für eine Modulschnittstelle"""
    module_name: str
    module_path: str
    interface_class: Optional[str] = None
    enabled: bool = True
    auto_connect: bool = True
    polling_interval_ms: int = 100
    connection_retry_interval_ms: int = 5000
    max_connection_retries: int = 10
    context_mapping: Dict[str, ContextSource] = field(default_factory=dict)
    focus_mapping: Dict[str, FocusTarget] = field(default_factory=dict)


@dataclass
class ModuleMessage:
    """Nachricht zwischen Modulen"""
    source_module: str
    target_module: str
    message_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: f"msg_{time.time()}_{id(threading.current_thread())}")
    priority: int = 50  # 0-100, höher = wichtiger
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextBridge:
    """
    Brücke zwischen VX-CONTEXT und anderen VXOR-Modulen
    
    Diese Klasse ist verantwortlich für:
    1. Verbindung zu anderen VXOR-Modulen herstellen und verwalten
    2. Kontextinformationen zwischen Modulen übersetzen und weiterleiten
    3. Fokusänderungen an relevante Module kommunizieren
    4. Ereignisse von anderen Modulen in Kontextdaten umwandeln
    """
    
    def __init__(self, context_core=None):
        """
        Initialisiert die ContextBridge
        
        Args:
            context_core: Referenz zum ContextCore-Modul, falls verfügbar
        """
        self.context_core = context_core or get_context_core()
        
        # Verbindungen zu anderen Modulen
        self.module_connections = {}
        self.module_interfaces = {}
        self.module_configs = {}
        
        # Nachrichtenwarteschlangen
        self.outgoing_messages = queue.PriorityQueue()
        self.incoming_messages = queue.PriorityQueue()
        
        # Threads
        self.active = False
        self.message_processing_thread = None
        self.module_polling_threads = {}
        
        # Nachrichtenhandler
        self.message_handlers = {}
        
        # Thread-Pool für asynchrone Verarbeitung
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Leistungsmetriken
        self.performance_metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'context_updates_sent': 0,
            'focus_updates_sent': 0,
            'connection_errors': 0,
            'message_processing_errors': 0
        }
        
        # Standardkonfigurationen für Module
        self._init_default_module_configs()
        
        logger.info("ContextBridge initialisiert")
    
    def _init_default_module_configs(self):
        """Initialisiert die Standardkonfigurationen für Module"""
        # VX-PSI
        self.module_configs[ModuleInterface.PSI] = ModuleConfig(
            module_name="VX-PSI",
            module_path="/vXor_Modules/VX-PSI",
            interface_class="PSIInterface",
            context_mapping={
                "perception": ContextSource.VISUAL,
                "sensation": ContextSource.EXTERNAL,
                "interpretation": ContextSource.INTERNAL
            },
            focus_mapping={
                "perception": FocusTarget.VISION,
                "sensation": FocusTarget.AUDIO,
                "interpretation": FocusTarget.REASONING
            }
        )
        
        # VX-EMO
        self.module_configs[ModuleInterface.EMO] = ModuleConfig(
            module_name="VX-EMO",
            module_path="/vXor_Modules/VX-EMO",
            interface_class="EmotionInterface",
            context_mapping={
                "emotion": ContextSource.EMOTION,
                "mood": ContextSource.EMOTION,
                "affect": ContextSource.EMOTION
            },
            focus_mapping={
                "emotion_processor": FocusTarget.EMOTION
            }
        )
        
        # VX-INTENT
        self.module_configs[ModuleInterface.INTENT] = ModuleConfig(
            module_name="VX-INTENT",
            module_path="/vXor_Modules/VX-INTENT",
            interface_class="IntentInterface",
            context_mapping={
                "intent": ContextSource.INTENT,
                "goal": ContextSource.INTENT,
                "plan": ContextSource.INTENT
            },
            focus_mapping={
                "intent_processor": FocusTarget.PLANNING
            }
        )
        
        # VX-VISION
        self.module_configs[ModuleInterface.VISION] = ModuleConfig(
            module_name="VX-VISION",
            module_path="/vXor_Modules/VX-VISION",
            interface_class="VisionInterface",
            context_mapping={
                "visual_input": ContextSource.VISUAL,
                "object_detection": ContextSource.VISUAL,
                "scene_analysis": ContextSource.VISUAL
            },
            focus_mapping={
                "vision_processor": FocusTarget.VISION
            }
        )
        
        # VX-MEMEX
        self.module_configs[ModuleInterface.MEMEX] = ModuleConfig(
            module_name="VX-MEMEX",
            module_path="/vXor_Modules/VX-MEMEX",
            interface_class="MemexInterface",
            context_mapping={
                "memory_retrieval": ContextSource.MEMORY,
                "memory_storage": ContextSource.MEMORY
            },
            focus_mapping={
                "memory_processor": FocusTarget.MEMORY
            }
        )
        
        # VX-REFLEX
        self.module_configs[ModuleInterface.REFLEX] = ModuleConfig(
            module_name="VX-REFLEX",
            module_path="/vXor_Modules/VX-REFLEX",
            interface_class="ReflexInterface",
            context_mapping={
                "reflex_trigger": ContextSource.REFLEX,
                "reflex_action": ContextSource.REFLEX
            },
            focus_mapping={
                "reflex_processor": FocusTarget.MOTOR
            }
        )
    
    def start(self):
        """Startet die ContextBridge"""
        if self.active:
            logger.warning("ContextBridge läuft bereits")
            return
        
        self.active = True
        
        # Starte Nachrichtenverarbeitungs-Thread
        self.message_processing_thread = threading.Thread(
            target=self._process_messages,
            name="ContextBridgeMessageProcessor",
            daemon=True
        )
        self.message_processing_thread.start()
        
        # Verbinde zu Modulen mit auto_connect=True
        for interface, config in self.module_configs.items():
            if config.enabled and config.auto_connect:
                self.connect_to_module(interface)
        
        logger.info("ContextBridge gestartet")
    
    def stop(self):
        """Stoppt die ContextBridge"""
        if not self.active:
            logger.warning("ContextBridge läuft nicht")
            return
        
        self.active = False
        
        # Stoppe alle Polling-Threads
        for thread in self.module_polling_threads.values():
            thread.join(timeout=1.0)
        
        # Stoppe den Nachrichtenverarbeitungs-Thread
        if self.message_processing_thread:
            self.message_processing_thread.join(timeout=1.0)
        
        # Schließe alle Modulverbindungen
        for interface in list(self.module_connections.keys()):
            self.disconnect_from_module(interface)
        
        # Schließe den Thread-Pool
        self.thread_pool.shutdown(wait=False)
        
        logger.info("ContextBridge gestoppt")
    
    def connect_to_module(self, interface: ModuleInterface) -> bool:
        """
        Stellt eine Verbindung zu einem Modul her
        
        Args:
            interface: Die zu verbindende Modulschnittstelle
            
        Returns:
            bool: True wenn erfolgreich verbunden, False sonst
        """
        if interface in self.module_connections:
            logger.warning(f"Bereits mit {interface.name} verbunden")
            return True
        
        config = self.module_configs.get(interface)
        if not config:
            logger.error(f"Keine Konfiguration für {interface.name} gefunden")
            return False
        
        if not config.enabled:
            logger.warning(f"Modul {interface.name} ist deaktiviert")
            return False
        
        logger.info(f"Verbinde zu {interface.name} ({config.module_name})...")
        
        try:
            # Versuche, das Modul zu importieren
            module_spec = importlib.util.find_spec(config.module_name)
            
            if module_spec is None:
                # Modul nicht gefunden, versuche, den Pfad zum Suchpfad hinzuzufügen
                import sys
                if config.module_path not in sys.path:
                    sys.path.append(config.module_path)
                
                # Versuche erneut zu importieren
                module_spec = importlib.util.find_spec(config.module_name)
                
                if module_spec is None:
                    logger.error(f"Modul {config.module_name} konnte nicht gefunden werden")
                    self.performance_metrics['connection_errors'] += 1
                    return False
            
            # Importiere das Modul
            module = importlib.import_module(config.module_name)
            
            # Hole die Schnittstellenklasse, falls angegeben
            if config.interface_class:
                if not hasattr(module, config.interface_class):
                    logger.error(f"Schnittstellenklasse {config.interface_class} nicht in {config.module_name} gefunden")
                    self.performance_metrics['connection_errors'] += 1
                    return False
                
                interface_class = getattr(module, config.interface_class)
                module_interface = interface_class()
            else:
                # Verwende das Modul selbst als Schnittstelle
                module_interface = module
            
            # Speichere die Verbindung
            self.module_connections[interface] = module
            self.module_interfaces[interface] = module_interface
            
            # Starte Polling-Thread, falls erforderlich
            if hasattr(module_interface, "poll") or hasattr(module_interface, "get_events"):
                polling_thread = threading.Thread(
                    target=self._poll_module,
                    args=(interface,),
                    name=f"ContextBridgePolling-{interface.name}",
                    daemon=True
                )
                self.module_polling_threads[interface] = polling_thread
                polling_thread.start()
            
            # Registriere Ereignishandler, falls verfügbar
            if hasattr(module_interface, "register_event_handler"):
                try:
                    module_interface.register_event_handler(self._handle_module_event)
                    logger.debug(f"Ereignishandler bei {interface.name} registriert")
                except Exception as e:
                    logger.warning(f"Konnte Ereignishandler nicht bei {interface.name} registrieren: {str(e)}")
            
            logger.info(f"Erfolgreich mit {interface.name} verbunden")
            return True
        
        except Exception as e:
            logger.error(f"Fehler beim Verbinden zu {interface.name}: {str(e)}", exc_info=True)
            self.performance_metrics['connection_errors'] += 1
            return False
    
    def disconnect_from_module(self, interface: ModuleInterface) -> bool:
        """
        Trennt die Verbindung zu einem Modul
        
        Args:
            interface: Die zu trennende Modulschnittstelle
            
        Returns:
            bool: True wenn erfolgreich getrennt, False sonst
        """
        if interface not in self.module_connections:
            logger.warning(f"Nicht mit {interface.name} verbunden")
            return False
        
        logger.info(f"Trenne Verbindung zu {interface.name}...")
        
        # Stoppe Pollin
(Content truncated due to size limit. Use line ranges to read in chunks)