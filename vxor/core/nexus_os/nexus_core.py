#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - NEXUS-OS Core

Diese Datei implementiert den Kern des NEXUS-OS, das als zentrale Integrationsschicht
zwischen verschiedenen MISO-Komponenten dient.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Konfiguriere Logging
logger = logging.getLogger("MISO.nexus_os")

class NexusCore:
    """
    NEXUS-OS Kernkomponente
    
    Diese Klasse dient als zentrale Steuerungseinheit für die Integration
    verschiedener MISO-Komponenten, insbesondere zwischen der T-Mathematics Engine
    und dem M-LINGUA Interface.
    """
    
    def __init__(self):
        """Initialisiert den NEXUS-OS Kern"""
        self.components = {}
        self.active_bridges = {}
        self.lock = threading.RLock()
        self.initialized = False
        logger.info("NEXUS-OS Kern initialisiert")
    
    def initialize(self):
        """Initialisiert alle NEXUS-OS Komponenten"""
        if self.initialized:
            logger.warning("NEXUS-OS Kern bereits initialisiert")
            return
        
        with self.lock:
            # Initialisiere Komponenten
            from .lingua_math_bridge import LinguaMathBridge
            from .tensor_language_processor import TensorLanguageProcessor
            
            # Erstelle und registriere Hauptkomponenten
            self.register_component("lingua_math_bridge", LinguaMathBridge())
            self.register_component("tensor_processor", TensorLanguageProcessor())
            
            self.initialized = True
            logger.info("NEXUS-OS Komponenten initialisiert")
    
    def register_component(self, name: str, component: Any):
        """
        Registriert eine Komponente im NEXUS-OS.
        
        Args:
            name: Name der Komponente
            component: Komponentenobjekt
        """
        with self.lock:
            if name in self.components:
                logger.warning(f"Komponente '{name}' bereits registriert, wird überschrieben")
            
            self.components[name] = component
            logger.debug(f"Komponente '{name}' registriert")
    
    def get_component(self, name: str) -> Any:
        """
        Gibt eine registrierte Komponente zurück.
        
        Args:
            name: Name der Komponente
            
        Returns:
            Komponentenobjekt oder None, falls nicht gefunden
        """
        return self.components.get(name)
    
    def create_bridge(self, source_component: str, target_component: str, bridge_id: Optional[str] = None) -> str:
        """
        Erstellt eine Brücke zwischen zwei Komponenten.
        
        Args:
            source_component: Quellkomponente
            target_component: Zielkomponente
            bridge_id: Optionale Bridge-ID
            
        Returns:
            Bridge-ID
        """
        with self.lock:
            if not bridge_id:
                bridge_id = f"{source_component}_to_{target_component}_{len(self.active_bridges)}"
            
            if bridge_id in self.active_bridges:
                logger.warning(f"Bridge '{bridge_id}' bereits vorhanden, wird überschrieben")
            
            # Prüfe, ob Komponenten existieren
            if source_component not in self.components:
                raise ValueError(f"Quellkomponente '{source_component}' nicht gefunden")
            
            if target_component not in self.components:
                raise ValueError(f"Zielkomponente '{target_component}' nicht gefunden")
            
            # Erstelle Bridge
            self.active_bridges[bridge_id] = {
                "source": source_component,
                "target": target_component,
                "created": True,
                "status": "active"
            }
            
            logger.info(f"Bridge '{bridge_id}' zwischen '{source_component}' und '{target_component}' erstellt")
            return bridge_id
    
    def remove_bridge(self, bridge_id: str) -> bool:
        """
        Entfernt eine Brücke.
        
        Args:
            bridge_id: Bridge-ID
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        with self.lock:
            if bridge_id not in self.active_bridges:
                logger.warning(f"Bridge '{bridge_id}' nicht gefunden")
                return False
            
            del self.active_bridges[bridge_id]
            logger.info(f"Bridge '{bridge_id}' entfernt")
            return True
    
    def get_bridge_status(self, bridge_id: str) -> Dict[str, Any]:
        """
        Gibt den Status einer Brücke zurück.
        
        Args:
            bridge_id: Bridge-ID
            
        Returns:
            Status-Dictionary oder leeres Dictionary, falls nicht gefunden
        """
        return self.active_bridges.get(bridge_id, {})
    
    def shutdown(self):
        """Fährt den NEXUS-OS Kern herunter"""
        with self.lock:
            # Entferne alle Brücken
            for bridge_id in list(self.active_bridges.keys()):
                self.remove_bridge(bridge_id)
            
            # Bereinige Komponenten
            self.components.clear()
            self.initialized = False
            
            logger.info("NEXUS-OS Kern heruntergefahren")


# Globale Instanz
_NEXUS_CORE = None

def get_nexus_core() -> NexusCore:
    """
    Gibt die globale NEXUS-OS Kern-Instanz zurück.
    
    Returns:
        NexusCore-Instanz
    """
    global _NEXUS_CORE
    
    if _NEXUS_CORE is None:
        _NEXUS_CORE = NexusCore()
        _NEXUS_CORE.initialize()
    
    return _NEXUS_CORE

def reset_nexus_core():
    """Setzt die globale NEXUS-OS Kern-Instanz zurück"""
    global _NEXUS_CORE
    
    if _NEXUS_CORE is not None:
        _NEXUS_CORE.shutdown()
        _NEXUS_CORE = None
