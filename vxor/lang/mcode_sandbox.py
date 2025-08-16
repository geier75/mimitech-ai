#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Sandbox

Diese Datei implementiert eine sichere Sandbox-Umgebung für die Ausführung von M-CODE.
Die Sandbox isoliert den Code und bietet kontrollierte Zugriffsmöglichkeiten auf
Systemressourcen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import threading
import json
import importlib
import inspect
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Importiere Sicherheitsausnahmen
from miso.lang.security_exceptions import SecurityViolationError, AccessDeniedError, InvalidTokenError, ResourceLimitExceededError

# Konfiguriere Logging
logger = logging.getLogger("MISO.lang.mcode_sandbox")

class SandboxSecurityManager:
    """
    Sicherheitsmanager für die M-CODE Sandbox
    
    Überwacht und kontrolliert den Zugriff auf Systemressourcen und
    stellt sicher, dass der ausgeführte Code innerhalb definierter
    Sicherheitsgrenzen bleibt.
    """
    
    def __init__(self, security_level: str = "high"):
        """
        Initialisiert den Sicherheitsmanager
        
        Args:
            security_level: Sicherheitsstufe (low, medium, high, extreme)
        """
        self.security_level = security_level
        self.allowed_modules = self._get_allowed_modules()
        self.allowed_functions = self._get_allowed_functions()
        self.blocked_modules = self._get_blocked_modules()
        self.access_log = []
        logger.info(f"SandboxSecurityManager initialisiert mit Sicherheitsstufe: {security_level}")
    
    def _get_allowed_modules(self) -> List[str]:
        """Gibt eine Liste der erlaubten Module basierend auf der Sicherheitsstufe zurück"""
        base_allowed = ["math", "random", "time", "datetime", "json", "re"]
        
        if self.security_level == "low":
            return base_allowed + ["os", "sys", "numpy", "torch"]
        elif self.security_level == "medium":
            return base_allowed + ["numpy", "torch"]
        elif self.security_level == "high":
            return base_allowed
        elif self.security_level == "extreme":
            return ["math", "json"]
        
        return base_allowed
    
    def _get_allowed_functions(self) -> Dict[str, List[str]]:
        """Gibt ein Dictionary mit erlaubten Funktionen pro Modul zurück"""
        allowed_functions = {
            "os": ["getcwd", "listdir", "path.join", "path.exists", "path.isfile"],
            "sys": ["version", "platform"],
            "math": ["*"],  # Alle Funktionen erlaubt
            "random": ["random", "randint", "choice", "sample"],
            "time": ["time", "sleep"],
            "datetime": ["*"],
            "json": ["*"],
            "re": ["match", "search", "findall", "sub"]
        }
        
        return allowed_functions
    
    def _get_blocked_modules(self) -> List[str]:
        """Gibt eine Liste der blockierten Module zurück"""
        base_blocked = ["subprocess", "socket", "requests", "urllib", "ftplib", "telnetlib", "smtplib"]
        
        if self.security_level == "extreme":
            return base_blocked + ["os", "sys", "numpy", "torch", "threading", "multiprocessing"]
        elif self.security_level == "high":
            return base_blocked + ["os", "sys", "threading", "multiprocessing"]
        elif self.security_level == "medium":
            return base_blocked + ["threading", "multiprocessing"]
        
        return base_blocked
    
    def check_import(self, module_name: str) -> bool:
        """
        Überprüft, ob ein Modul importiert werden darf
        
        Args:
            module_name: Name des zu importierenden Moduls
            
        Returns:
            True, wenn der Import erlaubt ist, sonst False
        """
        if module_name in self.blocked_modules:
            logger.warning(f"Versuch, blockiertes Modul zu importieren: {module_name}")
            import time
            self.access_log.append({
                "action": "blocked_import",
                "module": module_name,
                "timestamp": time.time()
            })
            return False
        
        if module_name not in self.allowed_modules:
            logger.warning(f"Versuch, nicht erlaubtes Modul zu importieren: {module_name}")
            import time
            self.access_log.append({
                "action": "denied_import",
                "module": module_name,
                "timestamp": time.time()
            })
            return False
        
        logger.debug(f"Import erlaubt: {module_name}")
        import time
        self.access_log.append({
            "action": "allowed_import",
            "module": module_name,
            "timestamp": time.time()
        })
        return True
    
    def check_function_call(self, module_name: str, function_name: str) -> bool:
        """
        Überprüft, ob eine Funktion aufgerufen werden darf
        
        Args:
            module_name: Name des Moduls
            function_name: Name der Funktion
            
        Returns:
            True, wenn der Aufruf erlaubt ist, sonst False
        """
        if module_name not in self.allowed_modules:
            logger.warning(f"Funktionsaufruf in nicht erlaubtem Modul: {module_name}.{function_name}")
            return False
        
        if module_name in self.allowed_functions:
            allowed_funcs = self.allowed_functions[module_name]
            if "*" in allowed_funcs or function_name in allowed_funcs:
                logger.debug(f"Funktionsaufruf erlaubt: {module_name}.{function_name}")
                return True
            else:
                logger.warning(f"Nicht erlaubter Funktionsaufruf: {module_name}.{function_name}")
                return False
        
        # Wenn keine spezifischen Funktionen definiert sind, erlaube alle
        logger.debug(f"Funktionsaufruf erlaubt (keine Einschränkungen): {module_name}.{function_name}")
        return True
    
    def get_access_log(self) -> List[Dict[str, Any]]:
        """Gibt das Zugriffsprotokoll zurück"""
        return self.access_log

class CodeSandbox:
    """
    Sandbox für die sichere Ausführung von M-CODE
    
    Bietet eine isolierte Umgebung für die Ausführung von Code mit
    kontrollierten Zugriffsmöglichkeiten auf Systemressourcen.
    """
    
    def __init__(self, security_level: str = "high"):
        """
        Initialisiert die Sandbox
        
        Args:
            security_level: Sicherheitsstufe (low, medium, high, extreme)
        """
        self.security_manager = SandboxSecurityManager(security_level)
        self.global_context = self._create_safe_globals()
        self.local_context = {}
        self.execution_timeout = self._get_timeout_for_security_level(security_level)
        logger.info(f"CodeSandbox initialisiert mit Sicherheitsstufe: {security_level}")
    
    def _get_timeout_for_security_level(self, security_level: str) -> int:
        """Gibt das Timeout basierend auf der Sicherheitsstufe zurück"""
        if security_level == "low":
            return 30  # 30 Sekunden
        elif security_level == "medium":
            return 15  # 15 Sekunden
        elif security_level == "high":
            return 5   # 5 Sekunden
        elif security_level == "extreme":
            return 2   # 2 Sekunden
        
        return 10  # Standardwert
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Erstellt ein sicheres globals-Dictionary für die Code-Ausführung"""
        safe_globals = {
            "__builtins__": {
                name: __builtins__[name]
                for name in [
                    "abs", "all", "any", "bool", "chr", "complex", "dict",
                    "divmod", "enumerate", "filter", "float", "format", "frozenset",
                    "hash", "hex", "int", "isinstance", "issubclass", "len",
                    "list", "map", "max", "min", "oct", "ord", "pow", "print",
                    "range", "repr", "reversed", "round", "set", "slice",
                    "sorted", "str", "sum", "tuple", "type", "zip"
                ]
            }
        }
        
        # Füge erlaubte Module hinzu
        for module_name in self.security_manager.allowed_modules:
            try:
                module = importlib.import_module(module_name)
                safe_globals[module_name] = module
            except ImportError:
                logger.warning(f"Konnte Modul nicht importieren: {module_name}")
        
        return safe_globals
    
    def execute_code(self, code: str, additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Führt Code in der Sandbox aus
        
        Args:
            code: Auszuführender Code
            additional_context: Zusätzlicher Kontext für die Ausführung
            
        Returns:
            Dictionary mit Ausführungsergebnis
        """
        # Aktualisiere den lokalen Kontext
        self.local_context = {} if additional_context is None else additional_context.copy()
        
        # Bereite Ergebnis vor
        result = {
            "success": False,
            "result": None,
            "error": None,
            "output": [],
            "execution_time": 0
        }
        
        # Erfasse Ausgaben
        original_stdout = sys.stdout
        from io import StringIO
        captured_output = StringIO()
        sys.stdout = captured_output
        
        import time
        start_time = time.time()
        
        try:
            # Führe Code aus
            exec(code, self.global_context, self.local_context)
            result["success"] = True
            
            # Extrahiere Ergebnis, falls vorhanden
            if "result" in self.local_context:
                result["result"] = self.local_context["result"]
            
        except Exception as e:
            result["success"] = False
            result["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        
        finally:
            # Stelle stdout wieder her
            sys.stdout = original_stdout
            result["output"] = captured_output.getvalue().splitlines()
            import time
            result["execution_time"] = time.time() - start_time
        
        return result
    
    def execute_function(self, function_name: str, args: List[Any] = None, kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Führt eine Funktion in der Sandbox aus
        
        Args:
            function_name: Name der Funktion
            args: Positionsargumente
            kwargs: Schlüsselwortargumente
            
        Returns:
            Dictionary mit Ausführungsergebnis
        """
        args = args or []
        kwargs = kwargs or {}
        
        if function_name not in self.local_context and function_name not in self.global_context:
            return {
                "success": False,
                "error": {
                    "type": "NameError",
                    "message": f"Funktion '{function_name}' nicht gefunden"
                }
            }
        
        # Hole die Funktion
        func = self.local_context.get(function_name) or self.global_context.get(function_name)
        
        # Führe die Funktion aus
        return self.execute_code(
            f"result = {function_name}(*args, **kwargs)",
            {"args": args, "kwargs": kwargs}
        )
    
    def get_security_manager(self) -> SandboxSecurityManager:
        """Gibt den Sicherheitsmanager zurück"""
        return self.security_manager
    
    def get_global_context(self) -> Dict[str, Any]:
        """Gibt den globalen Kontext zurück"""
        return self.global_context
    
    def get_local_context(self) -> Dict[str, Any]:
        """Gibt den lokalen Kontext zurück"""
        return self.local_context

# Import für SecuritySandbox (wird von anderen Modulen benötigt)
from miso.lang.security_sandbox import SecuritySandbox

# Hilfsfunktionen für die Verwendung der Sandbox
def create_sandbox(security_level: str = "high") -> CodeSandbox:
    """
    Erstellt eine neue Sandbox
    
    Args:
        security_level: Sicherheitsstufe (low, medium, high, extreme)
        
    Returns:
        CodeSandbox-Instanz
    """
    return CodeSandbox(security_level)

def execute_code_in_sandbox(code: str, security_level: str = "high", context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Führt Code in einer Sandbox aus
    
    Args:
        code: Auszuführender Code
        security_level: Sicherheitsstufe (low, medium, high, extreme)
        context: Zusätzlicher Kontext für die Ausführung
        
    Returns:
        Dictionary mit Ausführungsergebnis
    """
    sandbox = create_sandbox(security_level)
    return sandbox.execute_code(code, context)
