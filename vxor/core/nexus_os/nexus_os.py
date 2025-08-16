#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - NEXUS-OS

Diese Datei implementiert die Hauptklasse des NEXUS-OS, das als zentrale
Integrationsschicht zwischen verschiedenen MISO-Komponenten dient.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import json

# Importiere NEXUS-OS Komponenten
from .nexus_core import NexusCore, get_nexus_core
from .task_manager import TaskManager
from .resource_manager import ResourceManager
from .lingua_math_bridge import LinguaMathBridge
from .tensor_language_processor import TensorLanguageProcessor
from miso.core.nexus_os.t_math_integration import NexusOSTMathIntegration

# Importiere M-CODE Sandbox
try:
    from miso.lang.mcode_sandbox import CodeSandbox, execute_code_in_sandbox
    HAS_MCODE_SANDBOX = True
except ImportError:
    HAS_MCODE_SANDBOX = False
    logging.warning("M-CODE Sandbox nicht verfügbar. Einige Funktionen sind eingeschränkt.")

# Konfiguriere Logging
logger = logging.getLogger("MISO.nexus_os")

class NexusOS:
    """
    NEXUS-OS Hauptklasse
    
    Diese Klasse integriert alle Komponenten des NEXUS-OS und dient als zentrale
    Schnittstelle für andere MISO-Module.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert das NEXUS-OS
        
        Args:
            config: Konfigurationsparameter
        """
        self.config = config or {}
        self.initialized = False
        self.running = False
        self.start_time = None
        
        # Initialisiere Komponenten
        self.nexus_core = get_nexus_core()
        self.task_manager = TaskManager()
        self.resource_manager = ResourceManager()
        
        # Initialisiere T-MATHEMATICS-Integration
        try:
            from miso.math.t_mathematics.integration_manager import get_t_math_integration_manager
            t_math_manager = get_t_math_integration_manager()
            self.t_math_integration = t_math_manager.get_nexus_os_integration()
            if not self.t_math_integration:
                self.t_math_integration = NexusOSTMathIntegration()
                logger.info("T-MATHEMATICS-Integration manuell initialisiert")
            else:
                logger.info("T-MATHEMATICS-Integration erfolgreich geladen")
        except ImportError:
            logger.warning("T-MATHEMATICS-Integration konnte nicht geladen werden")
            self.t_math_integration = None
        
        # Initialisiere Sandbox, falls verfügbar
        self.sandbox = None
        if HAS_MCODE_SANDBOX:
            security_level = self.config.get("sandbox_security_level", "high")
            self.sandbox = CodeSandbox(security_level=security_level)
        
        logger.info("NEXUS-OS initialisiert")
    
    def initialize(self):
        """Initialisiert alle NEXUS-OS Komponenten"""
        if self.initialized:
            logger.warning("NEXUS-OS bereits initialisiert")
            return
        
        # Initialisiere Nexus Core
        self.nexus_core.initialize()
        
        # Starte Task Manager
        self.task_manager.start()
        
        # Starte Resource Manager
        self.resource_manager.start_monitoring()
        
        self.initialized = True
        logger.info("NEXUS-OS vollständig initialisiert")
    
    def start(self):
        """Startet das NEXUS-OS"""
        if not self.initialized:
            self.initialize()
        
        if self.running:
            logger.warning("NEXUS-OS läuft bereits")
            return
        
        self.running = True
        self.start_time = time.time()
        logger.info("NEXUS-OS gestartet")
    
    def stop(self):
        """Stoppt das NEXUS-OS"""
        if not self.running:
            logger.warning("NEXUS-OS läuft nicht")
            return
        
        # Stoppe Task Manager
        self.task_manager.stop()
        
        # Stoppe Resource Manager
        self.resource_manager.stop_monitoring()
        
        self.running = False
        logger.info("NEXUS-OS gestoppt")
    
    def execute_code(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Führt Code in der Sandbox aus
        
        Args:
            code: Auszuführender Code
            context: Zusätzlicher Kontext für die Ausführung
            
        Returns:
            Dictionary mit Ausführungsergebnis
        """
        if not HAS_MCODE_SANDBOX:
            return {
                "success": False,
                "error": {
                    "type": "ImportError",
                    "message": "M-CODE Sandbox nicht verfügbar"
                }
            }
        
        # Führe Code in Sandbox aus
        result = self.sandbox.execute_code(code, context)
        
        return result
    
    def process_natural_language(self, text: str) -> Dict[str, Any]:
        """
        Verarbeitet natürliche Sprache und führt entsprechende Aktionen aus
        
        Args:
            text: Natürlichsprachiger Text
            
        Returns:
            Verarbeitungsergebnis
        """
        # Hole TensorLanguageProcessor aus Nexus Core
        tensor_processor = self.nexus_core.get_component("tensor_processor")
        if not tensor_processor:
            return {
                "success": False,
                "error": "TensorLanguageProcessor nicht verfügbar"
            }
        
        # Verarbeite Text
        result = tensor_processor.process_language_request(text)
        
        return {
            "success": True,
            "result": result
        }
    
    def execute_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt eine Aufgabe aus.
        
        Args:
            task_config: Konfiguration der Aufgabe
            
        Returns:
            Ergebnis der Aufgabenausführung
        """
        # Prüfe, ob es sich um eine T-MATHEMATICS-Aufgabe handelt
        if task_config.get("type") == "t_mathematics" and self.t_math_integration:
            operation = task_config.get("operation")
            tensor_data = task_config.get("tensor_data")
            async_execution = task_config.get("async_execution", False)
            
            if async_execution:
                # Asynchrone Ausführung
                task_id = self.t_math_integration.submit_task(operation, tensor_data)
                return {"task_id": task_id, "status": "submitted", "type": "t_mathematics"}
            else:
                # Synchrone Ausführung
                result = self.t_math_integration.execute_tensor_operation(operation, tensor_data)
                return {"status": "completed", "result": result, "type": "t_mathematics"}
        
        # Standard-Aufgabenausführung
        task_id = self.task_manager.submit_task(task_config)
        return {"task_id": task_id, "status": "submitted"}
    
    def create_task(self, task_definition: Dict[str, Any]) -> str:
        """
        Erstellt eine neue Aufgabe
        
        Args:
            task_definition: Aufgabendefinition
            
        Returns:
            ID der erstellten Aufgabe
        """
        return self.task_manager.add_task(task_definition)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Gibt den aktuellen Systemstatus zurück
        
        Returns:
            Dictionary mit Systemstatusinformationen
        """
        # Hole Ressourcennutzung
        resource_usage = self.resource_manager.get_resource_usage()
        
        # Hole Aufgabenstatus
        tasks = {
            "pending": len(self.task_manager.get_task_queue()),
            "running": len(self.task_manager.get_running_tasks()),
            "completed": len(self.task_manager.get_completed_tasks()),
            "failed": len(self.task_manager.get_failed_tasks())
        }
        
        # Erstelle Statusbericht
        status = {
            "system": {
                "initialized": self.initialized,
                "running": self.running,
                "uptime": time.time() - self.nexus_core.creation_time if hasattr(self.nexus_core, "creation_time") else 0
            },
            "resources": resource_usage,
            "tasks": tasks,
            "components": {
                "nexus_core": self.nexus_core.initialized if hasattr(self.nexus_core, "initialized") else False,
                "task_manager": self.task_manager.running,
                "resource_manager": self.resource_manager.running,
                "sandbox": HAS_MCODE_SANDBOX
            }
        }
        
        # Füge T-MATHEMATICS-Status hinzu, falls verfügbar
        if hasattr(self, "t_math_integration") and self.t_math_integration:
            status["components"]["t_mathematics"] = {
                "available": True,
                "backend": "mlx" if hasattr(self.t_math_integration, "use_mlx") and self.t_math_integration.use_mlx else "torch",
                "async_tasks": len(self.t_math_integration.results) if hasattr(self.t_math_integration, "results") else 0
            }
        else:
            status["components"]["t_mathematics"] = {"available": False}
        
        return status
