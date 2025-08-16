#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - PRISM-Factory

Zentrales Factory-Modul für die PRISM-Engine und verwandte Komponenten.
Vermeidet zirkuläre Importe durch zentrale Instanzerstellung und Dependency Injection.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
import threading
from typing import Dict, Any, Optional, Type, TypeVar, Generic, Callable

# Konfiguriere Logging
logger = logging.getLogger("MISO.simulation.prism_factory")

# Type Variable für generische Typen
T = TypeVar('T')

class PrismComponentRegistry:
    """
    Zentrales Registry für alle PRISM-Komponenten.
    Ermöglicht Lazy-Loading und verhindert zirkuläre Importe.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PrismComponentRegistry, cls).__new__(cls)
                cls._instance._components = {}
                cls._instance._factories = {}
                cls._instance._initialized = False
            return cls._instance
    
    def register_component(self, component_id: str, component: Any) -> None:
        """
        Registriert eine Komponente in der Registry
        
        Args:
            component_id: Eindeutige ID der Komponente
            component: Die zu registrierende Komponente
        """
        self._components[component_id] = component
        logger.debug(f"Komponente '{component_id}' registriert: {type(component).__name__}")
    
    def register_factory(self, component_id: str, factory_func: Callable[[], Any]) -> None:
        """
        Registriert eine Factory-Funktion für Lazy-Loading
        
        Args:
            component_id: Eindeutige ID der zu erstellenden Komponente
            factory_func: Factory-Funktion zur Erstellung der Komponente
        """
        self._factories[component_id] = factory_func
        logger.debug(f"Factory für '{component_id}' registriert")
    
    def get_component(self, component_id: str) -> Optional[Any]:
        """
        Gibt eine registrierte Komponente zurück oder erstellt sie bei Bedarf
        
        Args:
            component_id: ID der gewünschten Komponente
            
        Returns:
            Die angeforderte Komponente oder None, wenn nicht gefunden
        """
        # Prüfe, ob die Komponente bereits erstellt wurde
        if component_id in self._components:
            return self._components[component_id]
        
        # Prüfe, ob eine Factory für die Komponente existiert
        if component_id in self._factories:
            try:
                # Erstelle die Komponente mit der Factory
                component = self._factories[component_id]()
                # Registriere die Komponente
                self.register_component(component_id, component)
                return component
            except Exception as e:
                logger.error(f"Fehler bei der Erstellung der Komponente '{component_id}': {e}")
                return None
        
        # Komponente nicht gefunden
        logger.warning(f"Komponente '{component_id}' nicht gefunden")
        return None
    
    def has_component(self, component_id: str) -> bool:
        """
        Prüft, ob eine Komponente registriert ist
        
        Args:
            component_id: ID der zu prüfenden Komponente
            
        Returns:
            True, wenn die Komponente registriert ist, sonst False
        """
        return component_id in self._components or component_id in self._factories
    
    def remove_component(self, component_id: str) -> bool:
        """
        Entfernt eine Komponente aus der Registry
        
        Args:
            component_id: ID der zu entfernenden Komponente
            
        Returns:
            True, wenn die Komponente entfernt wurde, sonst False
        """
        if component_id in self._components:
            del self._components[component_id]
            logger.debug(f"Komponente '{component_id}' entfernt")
            return True
        return False
    
    def clear(self) -> None:
        """Entfernt alle registrierten Komponenten"""
        self._components.clear()
        logger.debug("Alle Komponenten entfernt")
    
    def get_all_component_ids(self) -> list:
        """
        Gibt eine Liste aller registrierten Komponenten-IDs zurück
        
        Returns:
            Liste aller Komponenten-IDs
        """
        # Kombiniere registrierte Komponenten und Factories
        all_ids = list(self._components.keys()) + [
            k for k in self._factories.keys() if k not in self._components
        ]
        return all_ids


# Singleton-Funktion für den Zugriff auf die Registry
def get_prism_registry() -> PrismComponentRegistry:
    """
    Gibt die zentrale PRISM-Komponenten-Registry zurück
    
    Returns:
        Die PRISM-Komponenten-Registry
    """
    return PrismComponentRegistry()


# Factory-Funktionen für verschiedene PRISM-Komponenten

def get_prism_engine(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Gibt eine Instanz der PRISM-Engine zurück
    
    Args:
        config: Optionale Konfiguration für die Engine
        
    Returns:
        Eine Instanz der PRISM-Engine
    """
    registry = get_prism_registry()
    
    # Prüfe, ob bereits eine Instanz existiert
    if registry.has_component("prism_engine"):
        return registry.get_component("prism_engine")
    
    # Lazy-Import der PRISM-Engine, um zirkuläre Importe zu vermeiden
    from miso.simulation.prism_engine import PrismEngine
    
    # Erstelle eine neue Instanz
    engine = PrismEngine(config=config)
    
    # Registriere die Instanz
    registry.register_component("prism_engine", engine)
    
    return engine


def get_prism_matrix(dimensions: int = 4, initial_size: int = 10) -> Any:
    """
    Gibt eine Instanz der PrismMatrix zurück
    
    Args:
        dimensions: Anzahl der Dimensionen
        initial_size: Initiale Größe der Matrix
        
    Returns:
        Eine Instanz der PrismMatrix
    """
    registry = get_prism_registry()
    
    # Generiere eine eindeutige ID für diese Matrix-Konfiguration
    matrix_id = f"prism_matrix_{dimensions}_{initial_size}"
    
    # Prüfe, ob bereits eine Instanz mit dieser Konfiguration existiert
    if registry.has_component(matrix_id):
        return registry.get_component(matrix_id)
    
    # Lazy-Import der PrismMatrix-Klasse
    from miso.simulation.prism_matrix import PrismMatrix
    
    # Erstelle eine neue Instanz
    matrix = PrismMatrix(dimensions=dimensions, initial_size=initial_size)
    
    # Registriere die Instanz
    registry.register_component(matrix_id, matrix)
    
    return matrix


def get_time_scope_unit() -> Any:
    """
    Gibt eine Instanz der TimeScopeUnit zurück
    
    Returns:
        Eine Instanz der TimeScopeUnit
    """
    registry = get_prism_registry()
    
    # Prüfe, ob bereits eine Instanz existiert
    if registry.has_component("time_scope_unit"):
        return registry.get_component("time_scope_unit")
    
    # Lazy-Import der TimeScopeUnit-Klasse
    from miso.simulation.time_scope import TimeScopeUnit
    
    # Erstelle eine neue Instanz
    time_scope = TimeScopeUnit()
    
    # Registriere die Instanz
    registry.register_component("time_scope_unit", time_scope)
    
    return time_scope


def get_predictive_stream_analyzer(sequence_length: int = 10, prediction_horizon: int = 5) -> Any:
    """
    Gibt eine Instanz des PredictiveStreamAnalyzer zurück
    
    Args:
        sequence_length: Länge der Sequenz für die Analyse
        prediction_horizon: Horizont für die Vorhersage
        
    Returns:
        Eine Instanz des PredictiveStreamAnalyzer
    """
    registry = get_prism_registry()
    
    # Generiere eine eindeutige ID für diese Analyzer-Konfiguration
    analyzer_id = f"predictive_stream_analyzer_{sequence_length}_{prediction_horizon}"
    
    # Prüfe, ob bereits eine Instanz mit dieser Konfiguration existiert
    if registry.has_component(analyzer_id):
        return registry.get_component(analyzer_id)
    
    # Lazy-Import der PredictiveStreamAnalyzer-Klasse
    from miso.simulation.predictive_stream import PredictiveStreamAnalyzer
    
    # Erstelle eine neue Instanz
    analyzer = PredictiveStreamAnalyzer(
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    # Registriere die Instanz
    registry.register_component(analyzer_id, analyzer)
    
    return analyzer


def get_pattern_dissonance_scanner(dissonance_threshold: float = 0.3) -> Any:
    """
    Gibt eine Instanz des PatternDissonanceScanner zurück
    
    Args:
        dissonance_threshold: Schwellenwert für die Dissonanz
        
    Returns:
        Eine Instanz des PatternDissonanceScanner
    """
    registry = get_prism_registry()
    
    # Generiere eine eindeutige ID für diese Scanner-Konfiguration
    scanner_id = f"pattern_dissonance_scanner_{dissonance_threshold}"
    
    # Prüfe, ob bereits eine Instanz mit dieser Konfiguration existiert
    if registry.has_component(scanner_id):
        return registry.get_component(scanner_id)
    
    # Lazy-Import der PatternDissonanceScanner-Klasse
    from miso.simulation.pattern_dissonance import PatternDissonanceScanner
    
    # Erstelle eine neue Instanz
    scanner = PatternDissonanceScanner(dissonance_threshold=dissonance_threshold)
    
    # Registriere die Instanz
    registry.register_component(scanner_id, scanner)
    
    return scanner
