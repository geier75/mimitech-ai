#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-CONTEXT Module: focus_router.py
Dynamische Zuweisung von Fokusprioritäten an aktive Agenten im VXOR-System

Dieses Modul ist verantwortlich für die dynamische Verlagerung von kognitiver und
sensorischer Aufmerksamkeit in Echtzeit basierend auf dem aktuellen Kontext.
Es priorisiert eingehende Stimuli und leitet Fokus an die relevantesten Prozesse weiter.

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
import numpy as np
import heapq

# Logging-Konfiguration
LOG_DIR = "/home/ubuntu/VXOR_Logs/CONTEXT/"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}focus_router.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VX-CONTEXT.focus_router")

# Konstanten für Performance-Optimierung
FOCUS_UPDATE_INTERVAL_MS = 5  # 5ms für Ultra-Hochfrequenz-Fokus-Updates
MAX_FOCUS_TARGETS = 50  # Maximale Anzahl von Fokuszielen
FOCUS_DECAY_RATE = 0.95  # Exponentieller Abklingfaktor für Fokus
FOCUS_THRESHOLD = 0.1  # Schwellenwert, unter dem Fokus als inaktiv gilt


class FocusType(Enum):
    """Arten von Fokus im System"""
    COGNITIVE = auto()  # Kognitive Aufmerksamkeit
    SENSORY = auto()    # Sensorische Aufmerksamkeit
    MOTOR = auto()      # Motorische Aufmerksamkeit
    EMOTIONAL = auto()  # Emotionale Aufmerksamkeit
    MEMORY = auto()     # Gedächtnisaufmerksamkeit


class FocusTarget(Enum):
    """Ziele für Fokusrouting"""
    VISION = auto()     # Visuelles System
    AUDIO = auto()      # Audiosystem
    LANGUAGE = auto()   # Sprachverarbeitung
    MEMORY = auto()     # Gedächtnissystem
    EMOTION = auto()    # Emotionssystem
    PLANNING = auto()   # Planungssystem
    REASONING = auto()  # Schlussfolgerndes System
    MOTOR = auto()      # Motorisches System
    AGENT = auto()      # Spezifischer Agent


@dataclass
class FocusState:
    """Zustand des Fokus für ein bestimmtes Ziel"""
    target: Union[FocusTarget, str]
    focus_type: FocusType
    intensity: float  # 0.0 bis 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_id: Optional[str] = None  # Falls target=AGENT, spezifiziert welcher Agent
    
    def __post_init__(self):
        """Validierung nach Initialisierung"""
        if not isinstance(self.focus_type, FocusType):
            raise TypeError(f"focus_type muss vom Typ FocusType sein, nicht {type(self.focus_type)}")
        
        if isinstance(self.target, str) and not self.agent_id:
            self.agent_id = self.target
            self.target = FocusTarget.AGENT
        
        if not (0.0 <= self.intensity <= 1.0):
            raise ValueError(f"intensity muss zwischen 0.0 und 1.0 liegen, nicht {self.intensity}")


@dataclass
class FocusCommand:
    """Kommando zur Änderung des Fokus"""
    target: Union[FocusTarget, str]
    focus_type: FocusType
    intensity_delta: float  # Änderung der Intensität (-1.0 bis 1.0)
    priority: int  # 0-100, höher = wichtiger
    source: str  # Quelle des Kommandos
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validierung nach Initialisierung"""
        if not isinstance(self.focus_type, FocusType):
            raise TypeError(f"focus_type muss vom Typ FocusType sein, nicht {type(self.focus_type)}")
        
        if not (-1.0 <= self.intensity_delta <= 1.0):
            raise ValueError(f"intensity_delta muss zwischen -1.0 und 1.0 liegen, nicht {self.intensity_delta}")
        
        if not (0 <= self.priority <= 100):
            raise ValueError(f"priority muss zwischen 0 und 100 liegen, nicht {self.priority}")


class FocusRouter:
    """
    Verwaltet die dynamische Zuweisung von Fokus im VXOR-System
    
    Diese Klasse ist verantwortlich für:
    1. Verwaltung des aktuellen Fokuszustands für verschiedene Ziele
    2. Verarbeitung von Fokusänderungsanfragen
    3. Dynamische Anpassung des Fokus basierend auf Kontext und Priorität
    4. Benachrichtigung von Agenten über Fokusänderungen
    """
    
    def __init__(self, context_core=None):
        """
        Initialisiert den FocusRouter
        
        Args:
            context_core: Referenz zum ContextCore-Modul, falls verfügbar
        """
        self.focus_states = {}  # {(target, focus_type): FocusState}
        self.focus_command_queue = queue.PriorityQueue()
        self.active = False
        self.processing_thread = None
        self.focus_change_callbacks = {}  # {(target, focus_type): [callbacks]}
        self.global_focus_change_callbacks = []  # Callbacks für alle Fokusänderungen
        self.context_core = context_core
        
        # Leistungsmetriken
        self.performance_metrics = {
            'avg_processing_time_ms': 0,
            'max_processing_time_ms': 0,
            'processed_commands_count': 0,
            'dropped_commands_count': 0,
            'focus_changes_count': 0
        }
        
        # Aktive Agenten
        self.active_agents = set()
        
        # Fokushistorie für Analyse
        self.focus_history = {}  # {(target, focus_type): [Zeitreihe von Intensitäten]}
        self.history_max_length = 1000  # Maximale Länge der Fokushistorie
        
        logger.info("FocusRouter initialisiert")
    
    def start(self):
        """Startet den Fokus-Routing-Prozess"""
        if self.active:
            logger.warning("FocusRouter läuft bereits")
            return
        
        self.active = True
        self.processing_thread = threading.Thread(
            target=self._process_focus_commands,
            name="FocusRouterThread",
            daemon=True
        )
        self.processing_thread.start()
        logger.info("FocusRouter gestartet")
    
    def stop(self):
        """Stoppt den Fokus-Routing-Prozess"""
        if not self.active:
            logger.warning("FocusRouter läuft nicht")
            return
        
        self.active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            if self.processing_thread.is_alive():
                logger.warning("FocusRouter Thread konnte nicht sauber beendet werden")
        
        logger.info("FocusRouter gestoppt")
    
    def submit_focus_command(self, command: FocusCommand) -> bool:
        """
        Übermittelt ein Fokusänderungskommando zur Verarbeitung
        
        Args:
            command: Das zu verarbeitende Fokuskommando
            
        Returns:
            bool: True wenn erfolgreich übermittelt, False sonst
        """
        if not self.active:
            logger.warning("Versuch, Fokuskommando zu übermitteln, während FocusRouter inaktiv ist")
            return False
        
        try:
            # Prioritätsqueue-Eintrag: (Prioritätswert, Zeitstempel für FIFO bei gleicher Priorität, Kommando)
            # Höhere Priorität = niedrigerer Wert in der Queue
            priority_value = 100 - command.priority  # Umkehrung, damit höhere Priorität zuerst kommt
            self.focus_command_queue.put_nowait((priority_value, time.time(), command))
            return True
        except queue.Full:
            logger.error("Fokuskommando-Queue ist voll, Kommando wird verworfen")
            self.performance_metrics['dropped_commands_count'] += 1
            return False
    
    def register_agent(self, agent_id: str):
        """
        Registriert einen Agenten für Fokusrouting
        
        Args:
            agent_id: Eindeutige ID des Agenten
        """
        self.active_agents.add(agent_id)
        logger.debug(f"Agent {agent_id} registriert")
    
    def unregister_agent(self, agent_id: str):
        """
        Entfernt einen Agenten aus dem Fokusrouting
        
        Args:
            agent_id: Eindeutige ID des Agenten
        """
        if agent_id in self.active_agents:
            self.active_agents.remove(agent_id)
            logger.debug(f"Agent {agent_id} abgemeldet")
    
    def register_focus_change_callback(self, callback: Callable[[FocusState], None], 
                                      target: Optional[Union[FocusTarget, str]] = None,
                                      focus_type: Optional[FocusType] = None):
        """
        Registriert einen Callback für Fokusänderungen
        
        Args:
            callback: Die aufzurufende Funktion bei Fokusänderungen
            target: Optional, spezifisches Fokusziel
            focus_type: Optional, spezifischer Fokustyp
        """
        if target is None and focus_type is None:
            # Globaler Callback für alle Fokusänderungen
            self.global_focus_change_callbacks.append(callback)
            logger.debug(f"Globaler Fokusänderungs-Callback registriert")
        else:
            # Spezifischer Callback
            key = (target, focus_type)
            if key not in self.focus_change_callbacks:
                self.focus_change_callbacks[key] = []
            
            self.focus_change_callbacks[key].append(callback)
            logger.debug(f"Fokusänderungs-Callback für {key} registriert")
    
    def get_current_focus(self, target: Union[FocusTarget, str], 
                         focus_type: FocusType) -> Optional[FocusState]:
        """
        Gibt den aktuellen Fokuszustand für ein bestimmtes Ziel und Typ zurück
        
        Args:
            target: Das Fokusziel
            focus_type: Der Fokustyp
            
        Returns:
            Optional[FocusState]: Der aktuelle Fokuszustand oder None, wenn nicht vorhanden
        """
        key = self._get_focus_key(target, focus_type)
        return self.focus_states.get(key)
    
    def get_all_focus_states(self) -> Dict[Tuple, FocusState]:
        """
        Gibt alle aktuellen Fokuszustände zurück
        
        Returns:
            Dict[Tuple, FocusState]: Alle aktuellen Fokuszustände
        """
        return self.focus_states.copy()
    
    def get_top_focus_targets(self, focus_type: Optional[FocusType] = None, 
                             limit: int = 5) -> List[Tuple[Union[FocusTarget, str], float]]:
        """
        Gibt die Top-Fokusziele mit der höchsten Intensität zurück
        
        Args:
            focus_type: Optional, filtert nach einem bestimmten Fokustyp
            limit: Maximale Anzahl zurückzugebender Ziele
            
        Returns:
            List[Tuple[Union[FocusTarget, str], float]]: Liste von (Ziel, Intensität) Tupeln
        """
        focus_items = []
        
        for key, state in self.focus_states.items():
            target, state_focus_type = key
            
            # Filtere nach Fokustyp, falls angegeben
            if focus_type is not None and state_focus_type != focus_type:
                continue
            
            # Ignoriere Fokus unter dem Schwellenwert
            if state.intensity < FOCUS_THRESHOLD:
                continue
            
            focus_items.append((target, state.intensity))
        
        # Sortiere nach Intensität (absteigend) und begrenzte die Anzahl
        return sorted(focus_items, key=lambda x: x[1], reverse=True)[:limit]
    
    def _process_focus_commands(self):
        """Interne Methode zur kontinuierlichen Verarbeitung der Fokuskommando-Queue"""
        logger.info("Fokusverarbeitungsschleife gestartet")
        
        last_decay_time = time.time()
        
        while self.active:
            current_time = time.time()
            
            # Periodische Fokusabnahme
            if current_time - last_decay_time > (FOCUS_UPDATE_INTERVAL_MS / 1000):
                self._apply_focus_decay()
                last_decay_time = current_time
            
            # Verarbeite Fokuskommandos
            try:
                # Warte auf neue Fokuskommandos mit Timeout
                try:
                    _, _, command = self.focus_command_queue.get(timeout=FOCUS_UPDATE_INTERVAL_MS/2000)
                except queue.Empty:
                    # Keine neuen Kommandos, weiter in der Schleife
                    continue
                
                # Verarbeite Fokuskommando
                start_time = time.time()
                
                self._apply_focus_command(command)
                
                # Markiere als verarbeitet
                self.focus_command_queue.task_done()
                
                # Erfasse Performance-Metriken
                processing_time_ms = (time.time() - start_time) * 1000
                
                self._update_performance_metrics(processing_time_ms)
            
            except Exception as e:
                logger.error(f"Fehler bei der Fokusverarbeitung: {str(e)}", exc_info=True)
    
    def _apply_focus_command(self, command: FocusCommand):
        """
        Wendet ein Fokuskommando an und aktualisiert den Fokuszustand
        
        Args:
            command: Das anzuwendende Fokuskommando
        """
        key = self._get_focus_key(command.target, command.focus_type)
        
        # Hole den aktuellen Fokuszustand oder erstelle einen neuen
        current_state = self.focus_states.get(key)
        
        if current_state is None:
            # Neuer Fokuszustand
            new_intensity = max(0.0, min(1.0, command.intensity_delta))
            
            if new_intensity > 0:
                new_state = FocusState(
                    target=command.target,
                    focus_type=command.focus_type,
                    intensity=new_intensity,
                    metadata={"source": command.source, **command.metadata}
                )
                
                self.focus_states[key] = new_state
                self._notify_focus_change(new_state)
                self._update_focus_history(key, new_state)
                
                logger.debug(f"Neuer Fokus erstellt: {key} mit Intensität {new_intensity:.2f}")
        else:
            # Aktualisiere bestehenden Fokuszustand
            old_intensity = current_state.intensity
            new_intensity = max(0.0, min(1.0, old_intensity + command.intensity_delta))
            
            # Nur aktualisieren, wenn sich die Intensität signifikant geändert hat
            if abs(new_intensity - old_intensity) > 0.01:
                current_state.intensity = new_intensity
                current_state.timestamp = time.time()
                current_state.metadata.update({
                    "last_source": command.source,
                    "last_update": time.time(),
                    **command.metadata
                })
                
                self._notify_focus_change(current_state)
                self._update_focus_history(key, current_state)
                
                logger.debug(f"Fokus aktualisiert: {key} von {old_intensity:.2f} auf {new_intensity:.2f}")
                
                # Entferne Fokus, wenn Intensität unter Schwellenwert
                if new_intensity < FOCUS_THRESHOLD:
                    del self.focus_states[key]
                    logger.debug(f"Fokus entfernt: {key} (unter Schwellenwert)")
        
        self.performance_metrics['focus_changes_count'] += 1
    
    def _apply_focus_decay(self):
        """Wendet eine exponentielle Abnahme auf alle Fokuswerte an"""
        keys_to_remove = []
        
        for key, state in self.focus_states.items():
            old_intensity = state.intensity
            new_intensity = old_intensity * FOCUS_DECAY_RATE
            
            # Entferne Fokus, wenn Intensität unter Schwellenwert
            if
(Content truncated due to size limit. Use line ranges to read in chunks)