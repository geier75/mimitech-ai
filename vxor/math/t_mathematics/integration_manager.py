#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Integration Manager - Zentrale Verwaltung aller T-MATHEMATICS-Integrationen

Diese Datei implementiert einen zentralen Manager für alle T-MATHEMATICS-Integrationen
mit anderen MISO-Modulen. Sie stellt sicher, dass die T-MATHEMATICS Engine konsistent
in allen Modulen verwendet wird und optimiert die Ressourcennutzung.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import importlib
from functools import lru_cache

# Konfiguriere Logging
logger = logging.getLogger("t_mathematics.integration_manager")

# Importiere T-MATHEMATICS-Komponenten
from miso.math.t_mathematics.engine import TMathEngine
from miso.math.t_mathematics.compat import TMathConfig

# Lazy-Import für Integrationsmodule
def _import_module(module_path):
    """Importiert ein Modul nur bei Bedarf"""
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        logger.warning(f"Modul {module_path} konnte nicht importiert werden: {e}")
        return None

class TMathIntegrationManager:
    """
    Zentraler Manager für alle T-MATHEMATICS-Integrationen.
    
    Diese Klasse stellt sicher, dass die T-MATHEMATICS Engine konsistent
    in allen MISO-Modulen verwendet wird und optimiert die Ressourcennutzung
    durch Wiederverwendung von Instanzen.
    """
    
    def __init__(self):
        """Initialisiert den T-MATHEMATICS Integration Manager."""
        self._engine_instances = {}
        self._integration_modules = {}
        self._is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
        self._default_config = TMathConfig(
            precision="float16",
            device="auto",
            optimize_for_rdna=not self._is_apple_silicon,
            optimize_for_apple_silicon=self._is_apple_silicon
        )
        logger.info(f"T-MATHEMATICS Integration Manager initialisiert (Apple Silicon: {self._is_apple_silicon})")
    
    def register_engine(self, module_name: str, engine_instance) -> None:
        """
        Registriert eine existierende T-MATHEMATICS Engine-Instanz für ein bestimmtes Modul.
        
        Diese Methode ermöglicht es externen Modulen, ihre eigene Engine-Instanz zu registrieren,
        anstatt eine neue über get_engine() anzufordern. Dies ist nützlich für Module, die
        bereits eine Engine-Instanz erstellt haben und diese mit anderen Modulen teilen möchten.
        
        Args:
            module_name: Name des Moduls, unter dem die Engine registriert werden soll
            engine_instance: Die zu registrierende Engine-Instanz
        """
        self._engine_instances[module_name] = engine_instance
        logger.info(f"Externe T-MATHEMATICS Engine-Instanz für Modul '{module_name}' registriert")
    
    def get_engine(self, module_name: str = "default", config: Optional[TMathConfig] = None) -> TMathEngine:
        """
        Gibt eine T-MATHEMATICS Engine-Instanz für ein bestimmtes Modul zurück.
        
        Wenn für das angegebene Modul noch keine Instanz existiert, wird eine neue erstellt.
        Andernfalls wird die bestehende Instanz zurückgegeben, um Ressourcen zu sparen.
        
        Args:
            module_name: Name des Moduls, für das die Engine benötigt wird
            config: Optionale Konfiguration für die Engine
            
        Returns:
            T-MATHEMATICS Engine-Instanz
        """
        if module_name not in self._engine_instances:
            # Verwende die übergebene Konfiguration oder die Standardkonfiguration
            effective_config = config or self._default_config
            
            # Erstelle eine neue Engine-Instanz
            self._engine_instances[module_name] = TMathEngine(
                config=effective_config,
                use_mlx=self._is_apple_silicon
            )
            logger.info(f"Neue T-MATHEMATICS Engine-Instanz für Modul '{module_name}' erstellt")
        
        return self._engine_instances[module_name]
    
    def get_prism_integration(self):
        """
        Gibt die PRISM-Integrationsschnittstelle zurück.
        
        Returns:
            PRISM-Integrationsschnittstelle
        """
        if "prism" not in self._integration_modules:
            # Lazy-Import der PRISM-Integration
            prism_integration = _import_module("miso.math.t_mathematics.prism_integration")
            if prism_integration:
                self._integration_modules["prism"] = prism_integration.PrismSimulationEngine(
                    use_mlx=self._is_apple_silicon,
                    precision="float16",
                    device="auto"
                )
            else:
                logger.error("PRISM-Integration konnte nicht importiert werden")
                return None
        
        return self._integration_modules["prism"]
    
    def get_vxor_integration(self):
        """
        Gibt die VXOR-Integrationsschnittstelle zurück.
        
        Returns:
            VXOR-Integrationsschnittstelle
        """
        if "vxor" not in self._integration_modules:
            # Lazy-Import der VXOR-Integration
            vxor_integration = _import_module("miso.math.t_mathematics.vxor_math_integration")
            if vxor_integration:
                self._integration_modules["vxor"] = vxor_integration.get_vxor_math_integration()
            else:
                logger.error("VXOR-Integration konnte nicht importiert werden")
                return None
        
        return self._integration_modules["vxor"]
    
    def get_echo_prime_integration(self):
        """
        Gibt die ECHO PRIME-Integrationsschnittstelle zurück.
        
        Returns:
            ECHO PRIME-Integrationsschnittstelle
        """
        if "echo_prime" not in self._integration_modules:
            # Lazy-Import der ECHO PRIME-Integration
            echo_prime_integration = _import_module("miso.math.t_mathematics.echo_prime_integration")
            if echo_prime_integration:
                self._integration_modules["echo_prime"] = echo_prime_integration.EchoPrimeIntegration()
            else:
                logger.error("ECHO PRIME-Integration konnte nicht importiert werden")
                return None
        
        return self._integration_modules["echo_prime"]
    
    def get_nexus_os_integration(self):
        """
        Gibt die NEXUS-OS-Integrationsschnittstelle zurück.
        
        Returns:
            NEXUS-OS-Integrationsschnittstelle oder None, falls nicht verfügbar
        """
        # Lazy-Import der NEXUS-OS-Integration, falls sie existiert
        if "nexus_os" not in self._integration_modules:
            try:
                from miso.core.nexus_os.t_math_integration import NexusOSTMathIntegration
                self._integration_modules["nexus_os"] = NexusOSTMathIntegration(
                    t_math_engine=self.get_engine("nexus_os")
                )
                logger.info("NEXUS-OS-Integration erfolgreich initialisiert")
            except ImportError:
                logger.warning("NEXUS-OS-Integration nicht verfügbar")
                return None
        
        return self._integration_modules["nexus_os"]
    
    def clear_cache(self, module_name: Optional[str] = None):
        """
        Leert den Cache für ein bestimmtes Modul oder alle Module.
        
        Args:
            module_name: Name des Moduls, dessen Cache geleert werden soll, oder None für alle Module
        """
        if module_name:
            if module_name in self._engine_instances:
                del self._engine_instances[module_name]
                logger.info(f"Cache für Modul '{module_name}' geleert")
            if module_name in self._integration_modules:
                del self._integration_modules[module_name]
        else:
            self._engine_instances.clear()
            self._integration_modules.clear()
            logger.info("Gesamter Cache geleert")

# Singleton-Instanz des T-MATHEMATICS Integration Managers
_t_math_integration_manager = None

def get_t_math_integration_manager() -> TMathIntegrationManager:
    """
    Gibt die Singleton-Instanz des T-MATHEMATICS Integration Managers zurück.
    
    Returns:
        Singleton-Instanz des T-MATHEMATICS Integration Managers
    """
    global _t_math_integration_manager
    if _t_math_integration_manager is None:
        _t_math_integration_manager = TMathIntegrationManager()
    return _t_math_integration_manager
