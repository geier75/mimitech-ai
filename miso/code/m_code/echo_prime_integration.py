#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE ECHO-PRIME Integration

Dieses Modul implementiert die Integration zwischen M-CODE und ECHO-PRIME.
Es ermöglicht die Verwendung von temporalen Strategielogiken und Zeitlinienanalysen in M-CODE.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import sys
import time
import importlib.util
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.echo_prime_integration")

# Pfade konfigurieren
ECHO_PRIME_PATH = Path(__file__).parent.parent.parent / "echo_prime"


class EchoPrimeIntegrationError(Exception):
    """Fehler in der ECHO-PRIME-Integration"""
    pass


@dataclass
class TimelineConfig:
    """Konfiguration für Zeitlinien"""
    name: str
    probability: float = 1.0
    max_depth: int = 10
    max_branches: int = 5
    consistency_check: bool = True
    auto_resolve_paradoxes: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeNodeConfig:
    """Konfiguration für Zeitknoten"""
    name: str
    parent_timeline: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)


class EchoPrimeIntegration:
    """Integration zwischen M-CODE und ECHO-PRIME"""
    
    def __init__(self, use_echo_prime: bool = True):
        """
        Initialisiert eine neue ECHO-PRIME-Integration.
        
        Args:
            use_echo_prime: Ob ECHO-PRIME verwendet werden soll
        """
        self.use_echo_prime = use_echo_prime
        self.echo_prime_module = None
        self.timeline_manager = None
        self.integrity_guard = None
        self.active_timelines = {}
        self.active_time_nodes = {}
        
        # Versuche, ECHO-PRIME zu laden
        if self.use_echo_prime:
            self._load_echo_prime()
    
    def _load_echo_prime(self) -> None:
        """Lädt die ECHO-PRIME-Module"""
        try:
            # Füge ECHO-PRIME-Pfad zum Systempfad hinzu
            sys.path.append(str(ECHO_PRIME_PATH))
            
            # Importiere ECHO-PRIME-Komponenten
            from echo_prime.timeline import TimelineManager
            from echo_prime.time_node import TimeNode
            from echo_prime.integrity import TemporalIntegrityGuard
            
            # Speichere Referenzen
            self.echo_prime_module = {
                "TimelineManager": TimelineManager,
                "TimeNode": TimeNode,
                "TemporalIntegrityGuard": TemporalIntegrityGuard
            }
            
            # Initialisiere Kernkomponenten
            self.timeline_manager = TimelineManager()
            self.integrity_guard = TemporalIntegrityGuard(self.timeline_manager)
            
            logger.info("ECHO-PRIME erfolgreich geladen")
        except ImportError as e:
            logger.warning(f"ECHO-PRIME konnte nicht geladen werden: {e}")
            self.use_echo_prime = False
        except Exception as e:
            logger.error(f"Fehler beim Laden von ECHO-PRIME: {e}")
            self.use_echo_prime = False
    
    def is_available(self) -> bool:
        """
        Prüft, ob ECHO-PRIME verfügbar ist.
        
        Returns:
            True, wenn ECHO-PRIME verfügbar ist, sonst False
        """
        return self.use_echo_prime and self.echo_prime_module is not None
    
    def create_timeline(self, config: TimelineConfig) -> str:
        """
        Erstellt eine neue Zeitlinie.
        
        Args:
            config: Konfiguration für die Zeitlinie
            
        Returns:
            ID der Zeitlinie
        
        Raises:
            EchoPrimeIntegrationError: Wenn ECHO-PRIME nicht verfügbar ist
        """
        if not self.is_available():
            raise EchoPrimeIntegrationError("ECHO-PRIME ist nicht verfügbar")
            
        try:
            # Erstelle Zeitlinie
            timeline = self.timeline_manager.create_timeline(
                name=config.name,
                probability=config.probability,
                max_depth=config.max_depth,
                max_branches=config.max_branches,
                metadata=config.metadata
            )
            
            # Speichere Zeitlinie
            timeline_id = timeline.id
            self.active_timelines[timeline_id] = timeline
            
            if config.consistency_check:
                # Prüfe Konsistenz mit Integrity Guard
                self.integrity_guard.check_timeline_consistency(timeline_id)
            
            logger.info(f"Zeitlinie erstellt: {timeline_id} ({config.name})")
            return timeline_id
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Zeitlinie: {e}")
            raise EchoPrimeIntegrationError(f"Fehler beim Erstellen der Zeitlinie: {e}")
    
    def get_timeline(self, timeline_id: str) -> Any:
        """
        Gibt eine Zeitlinie zurück.
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            Zeitlinie
            
        Raises:
            EchoPrimeIntegrationError: Wenn ECHO-PRIME nicht verfügbar ist oder die Zeitlinie nicht existiert
        """
        if not self.is_available():
            raise EchoPrimeIntegrationError("ECHO-PRIME ist nicht verfügbar")
            
        if timeline_id not in self.active_timelines:
            # Versuche, die Zeitlinie zu laden
            try:
                timeline = self.timeline_manager.get_timeline(timeline_id)
                self.active_timelines[timeline_id] = timeline
                return timeline
            except Exception as e:
                logger.error(f"Fehler beim Laden der Zeitlinie {timeline_id}: {e}")
                raise EchoPrimeIntegrationError(f"Zeitlinie {timeline_id} nicht gefunden")
        
        return self.active_timelines[timeline_id]
    
    def create_time_node(self, timeline_id: str, config: TimeNodeConfig) -> str:
        """
        Erstellt einen neuen Zeitknoten in einer Zeitlinie.
        
        Args:
            timeline_id: ID der Zeitlinie
            config: Konfiguration für den Zeitknoten
            
        Returns:
            ID des Zeitknotens
            
        Raises:
            EchoPrimeIntegrationError: Wenn ECHO-PRIME nicht verfügbar ist oder die Zeitlinie nicht existiert
        """
        if not self.is_available():
            raise EchoPrimeIntegrationError("ECHO-PRIME ist nicht verfügbar")
            
        try:
            # Hole Zeitlinie
            timeline = self.get_timeline(timeline_id)
            
            # Erstelle Zeitknoten
            TimeNode = self.echo_prime_module["TimeNode"]
            time_node = TimeNode(
                name=config.name,
                timeline=timeline,
                parent_timeline_id=config.parent_timeline,
                timestamp=config.timestamp or time.time(),
                metadata=config.metadata,
                attributes=config.attributes
            )
            
            # Füge Zeitknoten zur Zeitlinie hinzu
            timeline.add_node(time_node)
            
            # Speichere Zeitknoten
            node_id = time_node.id
            self.active_time_nodes[node_id] = time_node
            
            logger.info(f"Zeitknoten erstellt: {node_id} ({config.name}) in Zeitlinie {timeline_id}")
            return node_id
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Zeitknotens: {e}")
            raise EchoPrimeIntegrationError(f"Fehler beim Erstellen des Zeitknotens: {e}")
    
    def get_time_node(self, node_id: str) -> Any:
        """
        Gibt einen Zeitknoten zurück.
        
        Args:
            node_id: ID des Zeitknotens
            
        Returns:
            Zeitknoten
            
        Raises:
            EchoPrimeIntegrationError: Wenn ECHO-PRIME nicht verfügbar ist oder der Zeitknoten nicht existiert
        """
        if not self.is_available():
            raise EchoPrimeIntegrationError("ECHO-PRIME ist nicht verfügbar")
            
        if node_id not in self.active_time_nodes:
            # Versuche, den Zeitknoten zu laden
            try:
                # Suche in allen aktiven Zeitlinien
                for timeline in self.active_timelines.values():
                    node = timeline.get_node(node_id)
                    if node is not None:
                        self.active_time_nodes[node_id] = node
                        return node
                
                raise EchoPrimeIntegrationError(f"Zeitknoten {node_id} nicht gefunden")
            except Exception as e:
                logger.error(f"Fehler beim Laden des Zeitknotens {node_id}: {e}")
                raise EchoPrimeIntegrationError(f"Zeitknoten {node_id} nicht gefunden")
        
        return self.active_time_nodes[node_id]
    
    def execute_in_timeline(self, timeline_id: str, func: Callable, *args, **kwargs) -> Any:
        """
        Führt eine Funktion im Kontext einer Zeitlinie aus.
        
        Args:
            timeline_id: ID der Zeitlinie
            func: Auszuführende Funktion
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Rückgabewert der Funktion
            
        Raises:
            EchoPrimeIntegrationError: Wenn ECHO-PRIME nicht verfügbar ist oder die Zeitlinie nicht existiert
        """
        if not self.is_available():
            raise EchoPrimeIntegrationError("ECHO-PRIME ist nicht verfügbar")
            
        try:
            # Hole Zeitlinie
            timeline = self.get_timeline(timeline_id)
            
            # Führe Funktion im Kontext der Zeitlinie aus
            with timeline.context():
                return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fehler bei der Ausführung in Zeitlinie {timeline_id}: {e}")
            raise EchoPrimeIntegrationError(f"Fehler bei der Ausführung in Zeitlinie {timeline_id}: {e}")
    
    def check_timeline_consistency(self, timeline_id: str) -> Dict[str, Any]:
        """
        Prüft die Konsistenz einer Zeitlinie.
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            Ergebnis der Konsistenzprüfung
            
        Raises:
            EchoPrimeIntegrationError: Wenn ECHO-PRIME nicht verfügbar ist oder die Zeitlinie nicht existiert
        """
        if not self.is_available():
            raise EchoPrimeIntegrationError("ECHO-PRIME ist nicht verfügbar")
            
        try:
            # Prüfe Konsistenz
            result = self.integrity_guard.check_timeline_consistency(timeline_id)
            
            logger.info(f"Konsistenzprüfung für Zeitlinie {timeline_id}: {result}")
            return result
        except Exception as e:
            logger.error(f"Fehler bei der Konsistenzprüfung für Zeitlinie {timeline_id}: {e}")
            raise EchoPrimeIntegrationError(f"Fehler bei der Konsistenzprüfung für Zeitlinie {timeline_id}: {e}")
    
    def resolve_paradox(self, timeline_id: str, node_id: str) -> bool:
        """
        Löst ein Paradoxon in einer Zeitlinie auf.
        
        Args:
            timeline_id: ID der Zeitlinie
            node_id: ID des problematischen Zeitknotens
            
        Returns:
            True, wenn das Paradoxon erfolgreich aufgelöst wurde, sonst False
            
        Raises:
            EchoPrimeIntegrationError: Wenn ECHO-PRIME nicht verfügbar ist oder die Zeitlinie nicht existiert
        """
        if not self.is_available():
            raise EchoPrimeIntegrationError("ECHO-PRIME ist nicht verfügbar")
            
        try:
            # Löse Paradoxon auf
            result = self.integrity_guard.resolve_paradox(timeline_id, node_id)
            
            if result:
                logger.info(f"Paradoxon in Zeitlinie {timeline_id} bei Knoten {node_id} erfolgreich aufgelöst")
            else:
                logger.warning(f"Paradoxon in Zeitlinie {timeline_id} bei Knoten {node_id} konnte nicht aufgelöst werden")
            
            return result
        except Exception as e:
            logger.error(f"Fehler bei der Auflösung des Paradoxons: {e}")
            raise EchoPrimeIntegrationError(f"Fehler bei der Auflösung des Paradoxons: {e}")
    
    def get_all_timelines(self) -> List[str]:
        """
        Gibt alle verfügbaren Zeitlinien zurück.
        
        Returns:
            Liste von Zeitlinien-IDs
            
        Raises:
            EchoPrimeIntegrationError: Wenn ECHO-PRIME nicht verfügbar ist
        """
        if not self.is_available():
            raise EchoPrimeIntegrationError("ECHO-PRIME ist nicht verfügbar")
            
        try:
            # Hole alle Zeitlinien
            timelines = self.timeline_manager.get_all_timelines()
            
            # Speichere Zeitlinien
            for timeline in timelines:
                self.active_timelines[timeline.id] = timeline
            
            return [timeline.id for timeline in timelines]
        except Exception as e:
            logger.error(f"Fehler beim Laden der Zeitlinien: {e}")
            raise EchoPrimeIntegrationError(f"Fehler beim Laden der Zeitlinien: {e}")
    
    def shutdown(self) -> None:
        """Fährt die ECHO-PRIME-Integration herunter"""
        if not self.is_available():
            return
            
        try:
            # Speichere alle Zeitlinien
            for timeline_id, timeline in self.active_timelines.items():
                self.timeline_manager.save_timeline(timeline_id)
            
            logger.info("ECHO-PRIME-Integration heruntergefahren")
        except Exception as e:
            logger.error(f"Fehler beim Herunterfahren der ECHO-PRIME-Integration: {e}")


# Singleton-Instanz der ECHO-PRIME-Integration
_echo_prime_integration = None

def get_echo_prime_integration(use_echo_prime: bool = True) -> EchoPrimeIntegration:
    """
    Gibt eine Singleton-Instanz der ECHO-PRIME-Integration zurück.
    
    Args:
        use_echo_prime: Ob ECHO-PRIME verwendet werden soll
        
    Returns:
        ECHO-PRIME-Integration
    """
    global _echo_prime_integration
    
    if _echo_prime_integration is None:
        _echo_prime_integration = EchoPrimeIntegration(use_echo_prime=use_echo_prime)
        
    return _echo_prime_integration


class TimelineContext:
    """Kontext-Manager für die Ausführung im Kontext einer Zeitlinie"""
    
    def __init__(self, timeline_id: str):
        """
        Initialisiert einen neuen Zeitlinien-Kontext.
        
        Args:
            timeline_id: ID der Zeitlinie
        """
        self.timeline_id = timeline_id
        self.integration = get_echo_prime_integration()
    
    def __enter__(self) -> Any:
        """
        Betritt den Zeitlinien-Kontext.
        
        Returns:
            Zeitlinie
        """
        if not self.integration.is_available():
            logger.warning("ECHO-PRIME ist nicht verfügbar, Zeitlinien-Kontext wird simuliert")
            return None
            
        try:
            # Hole Zeitlinie
            timeline = self.integration.get_timeline(self.timeline_id)
            
            # Aktiviere Zeitlinien-Kontext
            timeline.context().__enter__()
            
            return timeline
        except Exception as e:
            logger.error(f"Fehler beim Betreten des Zeitlinien-Kontexts: {e}")
            return None
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Verlässt den Zeitlinien-Kontext"""
        if not self.integration.is_available():
            return
            
        try:
            # Hole Zeitlinie
            timeline = self.integration.get_timeline(self.timeline_id)
            
            # Deaktiviere Zeitlinien-Kontext
            timeline.context().__exit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.error(f"Fehler beim Verlassen des Zeitlinien-Kontexts: {e}")


def in_timeline(timeline_id: str) -> Callable:
    """
    Dekorator für die Ausführung im Kontext einer Zeitlinie.
    
    Args:
        timeline_id: ID der Zeitlinie
        
    Returns:
        Dekorierte Funktion
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            integration = get_echo_prime_integration()
            return integration.execute_in_timeline(timeline_id, func, *args, **kwargs)
        return wrapper
    return decorator
