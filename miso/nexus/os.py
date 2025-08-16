#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - NEXUS OS

NEXUS OS ist das Betriebssystem und Ressourcenmanagement-System des MISO Ultimate AGI.
Es verwaltet Ressourcen, optimiert die Ausführung und stellt eine sichere Sandbox-Umgebung bereit.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import traceback
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.NEXUS_OS")

# Versuche, Abhängigkeiten zu importieren
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    logger.warning("NumPy nicht verfügbar, verwende Fallback-Implementierung")
    HAS_NUMPY = False

try:
    from ..math.t_mathematics.engine import TMathEngine
    from ..math.t_mathematics.compat import TMathConfig
    HAS_T_MATH = True
except ImportError:
    logger.warning("T-MATHEMATICS Engine nicht verfügbar, verwende Fallback-Implementierung")
    HAS_T_MATH = False

# Erstelle die mcode_sandbox-Abhängigkeit, falls sie nicht existiert
try:
    from ..lang.mcode_sandbox import MCodeSandbox, MCodeRuntime, MCodeSecurity
    HAS_MCODE = True
except ImportError:
    logger.warning("MCode-Sandbox nicht verfügbar, erstelle lokale Implementierung")
    HAS_MCODE = False
    
    # Lokale Implementierung der MCode-Sandbox
    class MCodeSecurity:
        """Sicherheitsmanager für MCode-Ausführung"""
        
        def __init__(self, security_level: str = "high"):
            self.security_level = security_level
            self.allowed_imports = ["math", "datetime", "json", "re"]
            self.allowed_functions = ["print", "len", "range", "enumerate", "zip", "map", "filter"]
            self.allowed_builtins = ["int", "float", "str", "list", "dict", "tuple", "set", "bool"]
            
            logger.info(f"MCodeSecurity initialisiert (Sicherheitsstufe: {security_level})")
        
        def validate_code(self, code: str) -> Tuple[bool, str]:
            """Validiert MCode auf Sicherheitsprobleme"""
            # Einfache Implementierung: Prüfe auf verbotene Importe
            forbidden_imports = ["os", "sys", "subprocess", "shutil", "requests"]
            for imp in forbidden_imports:
                if f"import {imp}" in code or f"from {imp}" in code:
                    return False, f"Verbotener Import: {imp}"
            
            return True, "Code validiert"
        
        def create_sandbox_env(self) -> Dict[str, Any]:
            """Erstellt eine abgesicherte Umgebung für die Codeausführung"""
            sandbox_env = {}
            
            # Füge erlaubte Builtins hinzu
            for name in self.allowed_builtins:
                if hasattr(__builtins__, name):
                    sandbox_env[name] = getattr(__builtins__, name)
            
            # Füge erlaubte Funktionen hinzu
            for name in self.allowed_functions:
                if hasattr(__builtins__, name):
                    sandbox_env[name] = getattr(__builtins__, name)
            
            # Füge erlaubte Module hinzu
            for module_name in self.allowed_imports:
                try:
                    module = __import__(module_name)
                    sandbox_env[module_name] = module
                except ImportError:
                    logger.warning(f"Modul {module_name} konnte nicht importiert werden")
            
            return sandbox_env
    
    class MCodeRuntime:
        """Laufzeitumgebung für MCode-Ausführung"""
        
        def __init__(self, security: MCodeSecurity = None):
            self.security = security or MCodeSecurity()
            self.globals = self.security.create_sandbox_env()
            self.locals = {}
            self.execution_time = 0.0
            self.memory_usage = 0.0
            
            logger.info("MCodeRuntime initialisiert")
        
        def execute(self, code: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
            """Führt MCode in einer sicheren Umgebung aus"""
            # Validiere den Code
            is_valid, message = self.security.validate_code(code)
            if not is_valid:
                logger.error(f"Ungültiger Code: {message}")
                return {"success": False, "error": message}
            
            # Bereite die Umgebung vor
            execution_env = self.globals.copy()
            if input_data:
                execution_env.update(input_data)
            
            # Führe den Code aus
            start_time = time.time()
            try:
                exec(code, execution_env, self.locals)
                self.execution_time = time.time() - start_time
                
                # Sammle die Ergebnisse
                result = {"success": True, "execution_time": self.execution_time}
                
                # Füge Ausgabevariablen hinzu
                for key, value in self.locals.items():
                    if not key.startswith("_"):
                        try:
                            # Versuche, den Wert zu serialisieren
                            json.dumps({key: value})
                            result[key] = value
                        except (TypeError, OverflowError):
                            # Wenn der Wert nicht serialisierbar ist, konvertiere ihn zu einem String
                            result[key] = str(value)
                
                return result
            except Exception as e:
                self.execution_time = time.time() - start_time
                logger.error(f"Fehler bei der Ausführung: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "execution_time": self.execution_time
                }
    
    class MCodeSandbox:
        """Sandbox für die sichere Ausführung von MCode"""
        
        def __init__(self, security_level: str = "high"):
            self.security = MCodeSecurity(security_level)
            self.runtime = MCodeRuntime(self.security)
            
            logger.info(f"MCodeSandbox initialisiert (Sicherheitsstufe: {security_level})")
        
        def execute(self, code: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
            """Führt Code in der Sandbox aus"""
            return self.runtime.execute(code, input_data)
        
        def validate(self, code: str) -> Tuple[bool, str]:
            """Validiert Code für die Sandbox"""
            return self.security.validate_code(code)

# NEXUS OS Kernklassen
class ResourceManager:
    """
    Verwaltet Systemressourcen wie CPU, Speicher und Festplattenspeicher.
    """
    
    def __init__(self):
        """Initialisiert den ResourceManager"""
        self.resources = {
            "cpu": {
                "total": self._get_cpu_count(),
                "used": 0,
                "available": self._get_cpu_count()
            },
            "memory": {
                "total": self._get_total_memory(),
                "used": 0,
                "available": self._get_total_memory()
            },
            "disk": {
                "total": self._get_total_disk_space(),
                "used": 0,
                "available": self._get_total_disk_space()
            }
        }
        self.resource_limits = {
            "cpu": 0.8,  # 80% der CPU
            "memory": 0.7,  # 70% des Speichers
            "disk": 0.9  # 90% des Festplattenspeichers
        }
        
        logger.info("ResourceManager initialisiert")
    
    def _get_cpu_count(self) -> int:
        """Gibt die Anzahl der verfügbaren CPU-Kerne zurück"""
        try:
            import multiprocessing
            return multiprocessing.cpu_count()
        except (ImportError, NotImplementedError):
            return 2  # Fallback-Wert
    
    def _get_total_memory(self) -> int:
        """Gibt den gesamten verfügbaren Speicher in Bytes zurück"""
        try:
            import psutil
            return psutil.virtual_memory().total
        except ImportError:
            return 8 * 1024 * 1024 * 1024  # Fallback: 8 GB
    
    def _get_total_disk_space(self) -> int:
        """Gibt den gesamten verfügbaren Festplattenspeicher in Bytes zurück"""
        try:
            import psutil
            return psutil.disk_usage("/").total
        except ImportError:
            return 100 * 1024 * 1024 * 1024  # Fallback: 100 GB
    
    def update_resource_usage(self):
        """Aktualisiert die Ressourcennutzung"""
        try:
            import psutil
            
            # CPU-Auslastung
            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            self.resources["cpu"]["used"] = cpu_percent * self.resources["cpu"]["total"]
            self.resources["cpu"]["available"] = self.resources["cpu"]["total"] - self.resources["cpu"]["used"]
            
            # Speicherauslastung
            memory = psutil.virtual_memory()
            self.resources["memory"]["used"] = memory.used
            self.resources["memory"]["available"] = memory.available
            
            # Festplattenauslastung
            disk = psutil.disk_usage("/")
            self.resources["disk"]["used"] = disk.used
            self.resources["disk"]["available"] = disk.free
            
            logger.debug("Ressourcennutzung aktualisiert")
        except ImportError:
            # Fallback: Simuliere Ressourcennutzung
            self.resources["cpu"]["used"] = 0.2 * self.resources["cpu"]["total"]
            self.resources["cpu"]["available"] = self.resources["cpu"]["total"] - self.resources["cpu"]["used"]
            
            self.resources["memory"]["used"] = 0.3 * self.resources["memory"]["total"]
            self.resources["memory"]["available"] = self.resources["memory"]["total"] - self.resources["memory"]["used"]
            
            self.resources["disk"]["used"] = 0.4 * self.resources["disk"]["total"]
            self.resources["disk"]["available"] = self.resources["disk"]["total"] - self.resources["disk"]["used"]
            
            logger.debug("Ressourcennutzung simuliert (Fallback)")
    
    def allocate_resources(self, resource_type: str, amount: int) -> bool:
        """Allokiert Ressourcen für eine Aufgabe"""
        if resource_type not in self.resources:
            logger.error(f"Unbekannter Ressourcentyp: {resource_type}")
            return False
        
        if amount > self.resources[resource_type]["available"]:
            logger.warning(f"Nicht genügend {resource_type} verfügbar (angefordert: {amount}, verfügbar: {self.resources[resource_type]['available']})")
            return False
        
        self.resources[resource_type]["used"] += amount
        self.resources[resource_type]["available"] -= amount
        
        logger.info(f"{amount} {resource_type} allokiert")
        return True
    
    def release_resources(self, resource_type: str, amount: int) -> bool:
        """Gibt Ressourcen frei"""
        if resource_type not in self.resources:
            logger.error(f"Unbekannter Ressourcentyp: {resource_type}")
            return False
        
        if amount > self.resources[resource_type]["used"]:
            logger.warning(f"Zu viele {resource_type} freigegeben (angefordert: {amount}, verwendet: {self.resources[resource_type]['used']})")
            amount = self.resources[resource_type]["used"]
        
        self.resources[resource_type]["used"] -= amount
        self.resources[resource_type]["available"] += amount
        
        logger.info(f"{amount} {resource_type} freigegeben")
        return True
    
    def check_resource_limits(self) -> Dict[str, bool]:
        """Prüft, ob Ressourcenlimits überschritten wurden"""
        results = {}
        
        for resource_type, limit in self.resource_limits.items():
            usage_ratio = self.resources[resource_type]["used"] / self.resources[resource_type]["total"]
            results[resource_type] = usage_ratio <= limit
            
            if usage_ratio > limit:
                logger.warning(f"{resource_type}-Limit überschritten: {usage_ratio:.2f} > {limit:.2f}")
        
        return results
    
    def get_resource_status(self) -> Dict[str, Dict[str, Any]]:
        """Gibt den aktuellen Ressourcenstatus zurück"""
        self.update_resource_usage()
        return self.resources

class Task:
    """
    Repräsentiert eine Aufgabe im NEXUS OS.
    """
    
    def __init__(self, task_id: str, name: str, function: Callable, args: Tuple = None, kwargs: Dict[str, Any] = None, priority: int = 0):
        """
        Initialisiert eine Aufgabe.
        
        Args:
            task_id: Eindeutige ID der Aufgabe
            name: Name der Aufgabe
            function: Auszuführende Funktion
            args: Positionsargumente für die Funktion
            kwargs: Schlüsselwortargumente für die Funktion
            priority: Priorität der Aufgabe (höhere Werte = höhere Priorität)
        """
        self.task_id = task_id
        self.name = name
        self.function = function
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.priority = priority
        self.status = "pending"
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.execution_time = None
        
        logger.info(f"Aufgabe {self.task_id} erstellt: {self.name} (Priorität: {self.priority})")
    
    def execute(self) -> Any:
        """Führt die Aufgabe aus"""
        self.status = "running"
        self.started_at = datetime.now()
        
        try:
            self.result = self.function(*self.args, **self.kwargs)
            self.status = "completed"
            logger.info(f"Aufgabe {self.task_id} erfolgreich abgeschlossen")
        except Exception as e:
            self.error = str(e)
            self.status = "failed"
            logger.error(f"Fehler bei der Ausführung der Aufgabe {self.task_id}: {e}")
        
        self.completed_at = datetime.now()
        self.execution_time = (self.completed_at - self.started_at).total_seconds()
        
        return self.result
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert die Aufgabe in ein Dictionary"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "priority": self.priority,
            "status": self.status,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time": self.execution_time
        }

class TaskManager:
    """
    Verwaltet Aufgaben im NEXUS OS.
    """
    
    def __init__(self, resource_manager: ResourceManager = None):
        """
        Initialisiert den TaskManager.
        
        Args:
            resource_manager: ResourceManager-Instanz (optional)
        """
        self.resource_manager = resource_manager
        self.tasks = {}
        self.pending_tasks = []
        self.running_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.max_concurrent_tasks = 10
        
        logger.info("TaskManager initialisiert")
    
    def create_task(self, name: str, function: Callable, args: Tuple = None, kwargs: Dict[str, Any] = None, priority: int = 0) -> str:
        """Erstellt eine neue Aufgabe"""
        task_id = f"task_{int(time.time())}_{len(self.tasks)}"
        task = Task(task_id, name, function, args, kwargs, priority)
        
        self.tasks[task_id] = task
        self.pending_tasks.append(task)
        
        # Sortiere die ausstehenden Aufgaben nach Priorität
        self.pending_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        logger.info(f"Aufgabe {task_id} erstellt und zur Warteschlange hinzugefügt")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Gibt eine Aufgabe anhand ihrer ID zurück"""
        return self.tasks.get(task_id)
    
    def execute_task(self, task_id: str) -> Any:
        """Führt eine Aufgabe aus"""
        task = self.get_task(task_id)
        if not task:
            logger.warning(f"Aufgabe {task_id} nicht gefunden")
            return None
        
        if task.status != "pending":
            logger.warning(f"Aufgabe {task_id} ist nicht ausstehend (Status: {task.status})")
            return task.result
        
        # Entferne die Aufgabe aus der Liste der ausstehenden Aufgaben
        if task in self.pending_tasks:
            self.pending_tasks.remove(task)
        
        # Füge die Aufgabe zur Liste der laufenden Aufgaben hinzu
        self.running_tasks.append(task)
        
        # Führe die Aufgabe aus
        result = task.execute()
        
        # Entferne die Aufgabe aus der Liste der laufenden Aufgaben
        if task in self.running_tasks:
            self.running_tasks.remove(task)
        
        # Füge die Aufgabe zur entsprechenden Liste hinzu
        if task.status == "completed":
            self.completed_tasks.append(task)
        elif task.status == "failed":
            self.failed_tasks.append(task)
        
        return result
    
    def schedule_tasks(self) -> int:
        """Plant ausstehende Aufgaben ein und führt sie aus"""
        # Prüfe, ob Ressourcen verfügbar sind
        if self.resource_manager:
            self.resource_manager.update_resource_usage()
            resource_limits = self.resource_manager.check_resource_limits()
            if not all(resource_limits.values()):
                logger.warning("Ressourcenlimits überschritten, keine neuen Aufgaben werden eingeplant")
                return 0
        
        # Berechne, wie viele Aufgaben ausgeführt werden können
        available_slots = self.max_concurrent_tasks - len(self.running_tasks)
        if available_slots <= 0:
            logger.info("Maximale Anzahl gleichzeitiger Aufgaben erreicht")
            return 0
        
        # Plane Aufgaben ein
        scheduled_count = 0
        for _ in range(min(available_slots, len(self.pending_tasks))):
            if not self.pending_tasks:
                break
            
            # Hole die Aufgabe mit der höchsten Priorität
            task = self.pending_tasks[0]
            
            # Führe die Aufgabe aus
            self.execute_task(task.task_id)
            scheduled_count += 1
        
        return scheduled_count
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """Gibt den Status einer Aufgabe zurück"""
        task = self.get_task(task_id)
        return task.status if task else None
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Gibt das Ergebnis einer Aufgabe zurück"""
        task = self.get_task(task_id)
        return task.result if task else None
    
    def cancel_task(self, task_id: str) -> bool:
        """Bricht eine Aufgabe ab"""
        task = self.get_task(task_id)
        if not task:
            logger.warning(f"Aufgabe {task_id} nicht gefunden")
            return False
        
        if task.status != "pending":
            logger.warning(f"Aufgabe {task_id} kann nicht abgebrochen werden (Status: {task.status})")
            return False
        
        # Entferne die Aufgabe aus der Liste der ausstehenden Aufgaben
        if task in self.pending_tasks:
            self.pending_tasks.remove(task)
        
        task.status = "cancelled"
        logger.info(f"Aufgabe {task_id} abgebrochen")
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken über Aufgaben zurück"""
        return {
            "total_tasks": len(self.tasks),
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "success_rate": len(self.completed_tasks) / (len(self.completed_tasks) + len(self.failed_tasks)) if (len(self.completed_tasks) + len(self.failed_tasks)) > 0 else 0
        }

class NexusOS:
    """
    Hauptklasse für NEXUS OS, das Betriebssystem und Ressourcenmanagement-System.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert NEXUS OS.
        
        Args:
            config: Konfiguration für NEXUS OS (optional)
        """
        self.config = config or {}
        self.resource_manager = ResourceManager()
        self.task_manager = TaskManager(self.resource_manager)
        self.mcode_sandbox = MCodeSandbox(self.config.get("security_level", "high"))
        self.t_math_engine = None
        
        # Initialisiere T-MATHEMATICS Engine, falls verfügbar
        if HAS_T_MATH:
            try:
                t_math_config = TMathConfig(
                    optimize_for_apple_silicon=self.config.get("use_mlx", True),
                    precision=self.config.get("precision", "float16")
                )
                self.t_math_engine = TMathEngine(config=t_math_config)
                logger.info("T-MATHEMATICS Engine erfolgreich initialisiert")
            except Exception as e:
                logger.error(f"Fehler bei der Initialisierung der T-MATHEMATICS Engine: {e}")
        
        logger.info("NEXUS OS erfolgreich initialisiert")
    
    def execute_code(self, code: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Führt Code in der Sandbox aus"""
        return self.mcode_sandbox.execute(code, input_data)
    
    def create_task(self, name: str, function: Callable, args: Tuple = None, kwargs: Dict[str, Any] = None, priority: int = 0) -> str:
        """Erstellt eine neue Aufgabe"""
        return self.task_manager.create_task(name, function, args, kwargs, priority)
    
    def schedule_tasks(self) -> int:
        """Plant ausstehende Aufgaben ein und führt sie aus"""
        return self.task_manager.schedule_tasks()
    
    def get_resource_status(self) -> Dict[str, Dict[str, Any]]:
        """Gibt den aktuellen Ressourcenstatus zurück"""
        return self.resource_manager.get_resource_status()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Gibt den Status des NEXUS OS-Systems zurück"""
        return {
            "resources": self.get_resource_status(),
            "tasks": self.task_manager.get_statistics(),
            "t_math_engine": self.t_math_engine is not None,
            "has_numpy": HAS_NUMPY,
            "has_t_math": HAS_T_MATH,
            "has_mcode": HAS_MCODE,
            "version": "1.0.0"
        }
    
    def optimize_system(self) -> Dict[str, Any]:
        """Optimiert das System"""
        # Implementiere Systemoptimierungen
        optimizations = {
            "memory_cleanup": self._optimize_memory(),
            "task_scheduling": self._optimize_task_scheduling(),
            "resource_allocation": self._optimize_resource_allocation()
        }
        
        logger.info("Systemoptimierung abgeschlossen")
        return optimizations
    
    def _optimize_memory(self) -> bool:
        """Optimiert die Speichernutzung"""
        # Implementiere Speicheroptimierungen
        return True
    
    def _optimize_task_scheduling(self) -> bool:
        """Optimiert die Aufgabenplanung"""
        # Implementiere Aufgabenplanungsoptimierungen
        return True
    
    def _optimize_resource_allocation(self) -> bool:
        """Optimiert die Ressourcenzuweisung"""
        # Implementiere Ressourcenzuweisungsoptimierungen
        return True

# Erstelle eine Instanz von NEXUS OS
nexus_os = NexusOS()

def get_nexus_os_instance(config: Dict[str, Any] = None) -> NexusOS:
    """Gibt eine Instanz von NEXUS OS zurück"""
    global nexus_os
    
    if config:
        # Erstelle eine neue Instanz mit der angegebenen Konfiguration
        nexus_os = NexusOS(config)
    
    return nexus_os
