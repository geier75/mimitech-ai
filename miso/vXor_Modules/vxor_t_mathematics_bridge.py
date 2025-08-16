#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR-T-MATHEMATICS Bridge - Optimierte Integrationsschnittstelle zwischen VXOR-Modulen und T-MATHEMATICS

Diese Datei implementiert eine optimierte Brücke zwischen den VXOR-Modulen und der
T-MATHEMATICS Engine, um eine effiziente Kommunikation und Ressourcennutzung zu ermöglichen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import json
import threading
import queue

# Konfiguriere Logging
logger = logging.getLogger("MISO.vxor.t_mathematics_bridge")

# Importiere VXOR-Adapter
from vXor_Modules.vxor_integration import VXORAdapter, check_module_availability

# Importiere T-MATHEMATICS Integration Manager
from miso.math.t_mathematics.integration_manager import get_t_math_integration_manager

# Importiere MIMIMON: ZTM-Modul für Sicherheitsüberprüfungen
try:
    from miso.security.ztm_module import ZTMSecurityManager, SecurityLevel
    HAS_ZTM = True
except ImportError:
    logger.warning("MIMIMON: ZTM-Modul konnte nicht importiert werden. Sicherheitsüberprüfungen eingeschränkt.")
    HAS_ZTM = False

class VXORTMathBridge:
    """
    Optimierte Brücke zwischen VXOR-Modulen und der T-MATHEMATICS Engine.
    
    Diese Klasse stellt eine effiziente Kommunikation zwischen den VXOR-Modulen
    (insbesondere VX-REASON und VX-MEMEX) und der T-MATHEMATICS Engine sicher.
    Sie implementiert Caching, asynchrone Verarbeitung und Sicherheitsüberprüfungen.
    """
    
    def __init__(self):
        """Initialisiert die VXOR-T-MATHEMATICS-Brücke."""
        # Initialisiere VXOR-Adapter
        self.vxor_adapter = VXORAdapter()
        
        # Hole T-MATHEMATICS Integration Manager
        self.t_math_manager = get_t_math_integration_manager()
        
        # Hole VXOR-Integration
        self.vxor_integration = self.t_math_manager.get_vxor_integration()
        
        # Initialisiere Engine-Cache
        self.engine_cache = {}
        
        # Initialisiere Ergebniscache
        self.result_cache = {}
        
        # Initialisiere Aufgabenwarteschlange für asynchrone Verarbeitung
        self.task_queue = queue.Queue()
        self.results = {}
        self.worker_thread = None
        self.running = False
        
        # Initialisiere ZTM-Sicherheitsmanager, falls verfügbar
        self.ztm_manager = None
        if HAS_ZTM:
            self.ztm_manager = ZTMSecurityManager()
        
        # Starte Worker-Thread
        self.start_worker()
        
        logger.info("VXOR-T-MATHEMATICS-Brücke initialisiert")
    
    def start_worker(self):
        """Startet den Worker-Thread für asynchrone Verarbeitung."""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            logger.info("Worker-Thread für asynchrone Verarbeitung gestartet")
    
    def stop_worker(self):
        """Stoppt den Worker-Thread für asynchrone Verarbeitung."""
        if self.running:
            self.running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=2.0)
            logger.info("Worker-Thread für asynchrone Verarbeitung gestoppt")
    
    def _worker_loop(self):
        """Hauptschleife des Worker-Threads."""
        while self.running:
            try:
                # Hole die nächste Aufgabe aus der Warteschlange
                task_id, vxor_module, operation, args, kwargs = self.task_queue.get(timeout=1.0)
                
                try:
                    # Führe die Operation aus
                    result = self._execute_operation(vxor_module, operation, *args, **kwargs)
                    self.results[task_id] = {"status": "completed", "result": result}
                except Exception as e:
                    # Fehlerbehandlung
                    self.results[task_id] = {"status": "error", "error": str(e)}
                
                # Markiere die Aufgabe als erledigt
                self.task_queue.task_done()
            except queue.Empty:
                # Keine Aufgabe in der Warteschlange
                pass
    
    def _execute_operation(self, vxor_module: str, operation: str, *args, **kwargs):
        """
        Führt eine Operation für ein VXOR-Modul aus.
        
        Args:
            vxor_module: Name des VXOR-Moduls
            operation: Name der Operation
            *args: Positionsargumente für die Operation
            **kwargs: Schlüsselwortargumente für die Operation
            
        Returns:
            Ergebnis der Operation
        """
        # Prüfe, ob das VXOR-Modul verfügbar ist
        if not self.vxor_adapter.is_module_available(vxor_module):
            raise ValueError(f"VXOR-Modul '{vxor_module}' ist nicht verfügbar")
        
        # Prüfe Sicherheitsberechtigungen, falls ZTM verfügbar
        if self.ztm_manager:
            security_level = self.ztm_manager.check_operation_security(
                module=vxor_module,
                operation=operation,
                args=args,
                kwargs=kwargs
            )
            
            if security_level < SecurityLevel.ALLOWED:
                raise PermissionError(f"Operation '{operation}' für Modul '{vxor_module}' ist nicht erlaubt")
        
        # Hole die T-MATHEMATICS Engine für das VXOR-Modul
        if vxor_module not in self.engine_cache:
            self.engine_cache[vxor_module] = self.t_math_manager.get_engine(vxor_module)
        
        engine = self.engine_cache[vxor_module]
        
        # Führe die Operation aus
        if hasattr(engine, operation):
            return getattr(engine, operation)(*args, **kwargs)
        else:
            raise AttributeError(f"Operation '{operation}' ist für Modul '{vxor_module}' nicht verfügbar")
    
    def execute_vxor_operation(self, vxor_module: str, operation: str, *args, **kwargs):
        """
        Führt eine Operation für ein VXOR-Modul synchron aus.
        
        Args:
            vxor_module: Name des VXOR-Moduls
            operation: Name der Operation
            *args: Positionsargumente für die Operation
            **kwargs: Schlüsselwortargumente für die Operation
            
        Returns:
            Ergebnis der Operation
        """
        # Generiere einen Cache-Schlüssel
        cache_key = self._generate_cache_key(vxor_module, operation, args, kwargs)
        
        # Prüfe, ob das Ergebnis im Cache ist
        if cache_key in self.result_cache:
            logger.debug(f"Cache-Treffer für Operation '{operation}' in Modul '{vxor_module}'")
            return self.result_cache[cache_key]
        
        # Führe die Operation aus
        result = self._execute_operation(vxor_module, operation, *args, **kwargs)
        
        # Speichere das Ergebnis im Cache
        self.result_cache[cache_key] = result
        
        return result
    
    def submit_vxor_task(self, vxor_module: str, operation: str, *args, **kwargs):
        """
        Reicht eine Aufgabe für ein VXOR-Modul zur asynchronen Ausführung ein.
        
        Args:
            vxor_module: Name des VXOR-Moduls
            operation: Name der Operation
            *args: Positionsargumente für die Operation
            **kwargs: Schlüsselwortargumente für die Operation
            
        Returns:
            ID der eingereichten Aufgabe
        """
        import uuid
        
        # Generiere eine eindeutige ID für die Aufgabe
        task_id = str(uuid.uuid4())
        
        # Stelle sicher, dass der Worker-Thread läuft
        if not self.running:
            self.start_worker()
        
        # Füge die Aufgabe zur Warteschlange hinzu
        self.task_queue.put((task_id, vxor_module, operation, args, kwargs))
        
        # Initialisiere den Ergebnisstatus
        self.results[task_id] = {"status": "pending"}
        
        return task_id
    
    def get_task_result(self, task_id: str):
        """
        Gibt das Ergebnis einer Aufgabe zurück.
        
        Args:
            task_id: ID der Aufgabe
            
        Returns:
            Ergebnis der Aufgabe oder Status
        """
        return self.results.get(task_id, {"status": "unknown"})
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None):
        """
        Wartet auf den Abschluss einer Aufgabe.
        
        Args:
            task_id: ID der Aufgabe
            timeout: Optionales Timeout in Sekunden
            
        Returns:
            Ergebnis der Aufgabe
        """
        import time
        
        start_time = time.time()
        while True:
            result = self.get_task_result(task_id)
            if result["status"] != "pending":
                return result
            
            # Prüfe, ob das Timeout erreicht wurde
            if timeout is not None and time.time() - start_time > timeout:
                return {"status": "timeout"}
            
            # Kurze Pause, um CPU-Auslastung zu reduzieren
            time.sleep(0.1)
    
    def _generate_cache_key(self, vxor_module: str, operation: str, args: Tuple, kwargs: Dict):
        """
        Generiert einen Cache-Schlüssel für eine Operation.
        
        Args:
            vxor_module: Name des VXOR-Moduls
            operation: Name der Operation
            args: Positionsargumente für die Operation
            kwargs: Schlüsselwortargumente für die Operation
            
        Returns:
            Cache-Schlüssel
        """
        # Konvertiere Argumente in eine serialisierbare Form
        serializable_args = []
        for arg in args:
            if hasattr(arg, "tolist"):
                # Konvertiere NumPy-Arrays und Tensoren in Listen
                serializable_args.append(arg.tolist())
            else:
                serializable_args.append(str(arg))
        
        # Konvertiere Schlüsselwortargumente in eine serialisierbare Form
        serializable_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(value, "tolist"):
                # Konvertiere NumPy-Arrays und Tensoren in Listen
                serializable_kwargs[key] = value.tolist()
            else:
                serializable_kwargs[key] = str(value)
        
        # Erstelle einen JSON-String als Cache-Schlüssel
        cache_key = json.dumps({
            "module": vxor_module,
            "operation": operation,
            "args": serializable_args,
            "kwargs": serializable_kwargs
        })
        
        return cache_key
    
    def clear_cache(self, vxor_module: Optional[str] = None):
        """
        Leert den Cache für ein bestimmtes VXOR-Modul oder alle Module.
        
        Args:
            vxor_module: Name des VXOR-Moduls oder None für alle Module
        """
        if vxor_module:
            # Lösche nur die Einträge für das angegebene Modul
            keys_to_delete = []
            for key in self.result_cache:
                if f'"module": "{vxor_module}"' in key:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.result_cache[key]
            
            # Lösche die Engine aus dem Cache
            if vxor_module in self.engine_cache:
                del self.engine_cache[vxor_module]
            
            logger.info(f"Cache für VXOR-Modul '{vxor_module}' geleert")
        else:
            # Lösche den gesamten Cache
            self.result_cache.clear()
            self.engine_cache.clear()
            logger.info("Gesamter Cache geleert")
    
    def get_available_vxor_modules(self):
        """
        Gibt eine Liste der verfügbaren VXOR-Module zurück.
        
        Returns:
            Dictionary mit Modulnamen und Verfügbarkeitsstatus
        """
        modules = {}
        
        # Prüfe, welche VXOR-Module verfügbar sind
        for module_name in ["VX-REASON", "VX-MEMEX", "VX-PSI", "VX-SOMA"]:
            modules[module_name] = self.vxor_adapter.is_module_available(module_name)
        
        return modules
    
    def get_vxor_module_capabilities(self, vxor_module: str):
        """
        Gibt die Fähigkeiten eines VXOR-Moduls zurück.
        
        Args:
            vxor_module: Name des VXOR-Moduls
            
        Returns:
            Liste von verfügbaren Operationen
        """
        # Prüfe, ob das VXOR-Modul verfügbar ist
        if not self.vxor_adapter.is_module_available(vxor_module):
            raise ValueError(f"VXOR-Modul '{vxor_module}' ist nicht verfügbar")
        
        # Hole die T-MATHEMATICS Engine für das VXOR-Modul
        if vxor_module not in self.engine_cache:
            self.engine_cache[vxor_module] = self.t_math_manager.get_engine(vxor_module)
        
        engine = self.engine_cache[vxor_module]
        
        # Sammle alle verfügbaren Operationen
        operations = []
        for attr_name in dir(engine):
            if not attr_name.startswith("_") and callable(getattr(engine, attr_name)):
                operations.append(attr_name)
        
        return operations

# Singleton-Instanz der VXOR-T-MATHEMATICS-Brücke
_vxor_t_math_bridge = None

def get_vxor_t_math_bridge():
    """
    Gibt die Singleton-Instanz der VXOR-T-MATHEMATICS-Brücke zurück.
    
    Returns:
        Singleton-Instanz der VXOR-T-MATHEMATICS-Brücke
    """
    global _vxor_t_math_bridge
    if _vxor_t_math_bridge is None:
        _vxor_t_math_bridge = VXORTMathBridge()
    return _vxor_t_math_bridge
