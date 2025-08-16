#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - ZTM Hooks Integration

Implementiert ZTM-Hooks für kritische Module und stellt Decorators
für die automatische Überwachung bereit.

Copyright (c) 2025 MIMI Tech AI. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import functools
import logging
import traceback
from typing import Any, Dict, Callable, Optional, List
import threading

# Importiere ZTM-Monitor
try:
    from .ztm_monitor import get_instance as get_ztm_monitor, track
    ZTM_AVAILABLE = True
except ImportError:
    ZTM_AVAILABLE = False
    def track(*args, **kwargs):
        pass

logger = logging.getLogger("MISO.Security.ZTM.Hooks")

class ZTMHookManager:
    """Verwaltet ZTM-Hooks für kritische Module"""
    
    def __init__(self):
        self.hooks = {}
        self.active = os.environ.get("MISO_ZTM_MODE", "0") == "1"
        self.critical_modules = {
            "vx_gestalt", "vx_reason", "vx_matrix", "vx_memex", 
            "vx_intent", "vx_hyperfilter", "prism_engine", 
            "echo_prime", "t_mathematics", "qlogik_core",
            "void_protocol", "ztm_monitor"
        }
        self._lock = threading.Lock()
        
    def register_hook(self, module: str, operation: str, hook_func: Callable):
        """Registriert einen Hook für ein Modul und eine Operation"""
        with self._lock:
            if module not in self.hooks:
                self.hooks[module] = {}
            if operation not in self.hooks[module]:
                self.hooks[module][operation] = []
            self.hooks[module][operation].append(hook_func)
            logger.debug(f"Hook registriert: {module}.{operation}")
    
    def execute_hooks(self, module: str, operation: str, data: Dict[str, Any]):
        """Führt alle registrierten Hooks für ein Modul und eine Operation aus"""
        if not self.active:
            return
            
        with self._lock:
            if module in self.hooks and operation in self.hooks[module]:
                for hook in self.hooks[module][operation]:
                    try:
                        hook(module, operation, data)
                    except Exception as e:
                        logger.error(f"Fehler beim Ausführen des Hooks {module}.{operation}: {e}")

# Globale Hook-Manager-Instanz
_hook_manager = ZTMHookManager()

def ztm_monitored(component: str, operation: str, severity: str = "INFO"):
    """
    Decorator für ZTM-Überwachung von Funktionen
    
    Args:
        component: Name der Komponente
        operation: Name der Operation
        severity: Schweregrad (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ZTM_AVAILABLE or not _hook_manager.active:
                return func(*args, **kwargs)
            
            start_time = time.time()
            operation_id = f"{component}.{operation}_{int(start_time * 1000000)}"
            
            # Pre-execution tracking
            track("OPERATION_START", component, {
                "operation": operation,
                "operation_id": operation_id,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
                "function_name": func.__name__,
                "module": func.__module__
            }, severity)
            
            try:
                # Führe Hooks vor der Ausführung aus
                _hook_manager.execute_hooks(component, f"{operation}_pre", {
                    "operation_id": operation_id,
                    "args": args,
                    "kwargs": kwargs,
                    "function": func
                })
                
                # Führe die eigentliche Funktion aus
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                
                # Post-execution tracking
                track("OPERATION_SUCCESS", component, {
                    "operation": operation,
                    "operation_id": operation_id,
                    "execution_time": execution_time,
                    "result_type": str(type(result)),
                    "success": True
                }, severity)
                
                # Führe Hooks nach der Ausführung aus
                _hook_manager.execute_hooks(component, f"{operation}_post", {
                    "operation_id": operation_id,
                    "result": result,
                    "execution_time": execution_time,
                    "success": True
                })
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_info = {
                    "operation": operation,
                    "operation_id": operation_id,
                    "execution_time": execution_time,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "success": False
                }
                
                # Error tracking
                track("OPERATION_ERROR", component, error_info, "ERROR")
                
                # Führe Error-Hooks aus
                _hook_manager.execute_hooks(component, f"{operation}_error", {
                    "operation_id": operation_id,
                    "error": e,
                    "error_info": error_info
                })
                
                raise
        
        return wrapper
    return decorator

def ztm_secure_data(component: str, data: Any, operation: str = "data_access"):
    """
    Sichert Daten über ZTM und VOID-Protokoll
    
    Args:
        component: Name der Komponente
        data: Zu sichernde Daten
        operation: Name der Operation
        
    Returns:
        Gesicherte Daten
    """
    if not ZTM_AVAILABLE or not _hook_manager.active:
        return data
    
    try:
        # Importiere VOID-Protokoll
        from .void.void_protocol import get_instance as get_void
        void_protocol = get_void()
        
        # Sichere Daten über VOID
        secured_data = void_protocol.secure(data)
        
        # Protokolliere Sicherungsvorgang
        track("DATA_SECURED", component, {
            "operation": operation,
            "data_type": str(type(data)),
            "secured": True,
            "void_secured": "_void_secured" in secured_data if isinstance(secured_data, dict) else False
        }, "DEBUG")
        
        return secured_data
        
    except Exception as e:
        logger.error(f"Fehler beim Sichern der Daten für {component}: {e}")
        track("DATA_SECURITY_ERROR", component, {
            "operation": operation,
            "error": str(e),
            "data_type": str(type(data))
        }, "ERROR")
        return data

def ztm_verify_data(component: str, data: Any, signature: Optional[str] = None, operation: str = "data_verification"):
    """
    Verifiziert Daten über ZTM und VOID-Protokoll
    
    Args:
        component: Name der Komponente
        data: Zu verifizierende Daten
        signature: Optionale Signatur
        operation: Name der Operation
        
    Returns:
        True, wenn die Daten verifiziert wurden
    """
    if not ZTM_AVAILABLE or not _hook_manager.active:
        return True
    
    try:
        # Importiere VOID-Protokoll
        from .void.void_protocol import get_instance as get_void
        void_protocol = get_void()
        
        # Verifiziere Daten über VOID
        verified = void_protocol.verify(data, signature)
        
        # Protokolliere Verifikationsvorgang
        track("DATA_VERIFIED", component, {
            "operation": operation,
            "data_type": str(type(data)),
            "verified": verified,
            "has_signature": signature is not None
        }, "DEBUG")
        
        return verified
        
    except Exception as e:
        logger.error(f"Fehler beim Verifizieren der Daten für {component}: {e}")
        track("DATA_VERIFICATION_ERROR", component, {
            "operation": operation,
            "error": str(e),
            "data_type": str(type(data))
        }, "ERROR")
        return False

def ztm_critical_operation(component: str, operation: str):
    """
    Decorator für kritische Operationen mit erweiterten Sicherheitsmaßnahmen
    
    Args:
        component: Name der Komponente
        operation: Name der Operation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ZTM_AVAILABLE or not _hook_manager.active:
                return func(*args, **kwargs)
            
            start_time = time.time()
            operation_id = f"CRITICAL_{component}.{operation}_{int(start_time * 1000000)}"
            
            # Kritische Operation starten
            track("CRITICAL_OPERATION_START", component, {
                "operation": operation,
                "operation_id": operation_id,
                "function_name": func.__name__,
                "module": func.__module__,
                "critical": True
            }, "WARNING")
            
            try:
                # Zusätzliche Sicherheitsprüfungen für kritische Operationen
                if component in _hook_manager.critical_modules:
                    # Verifiziere Eingabeparameter
                    for i, arg in enumerate(args):
                        if not ztm_verify_data(component, arg, operation=f"{operation}_arg_{i}"):
                            raise SecurityError(f"Verifikation fehlgeschlagen für Argument {i}")
                    
                    for key, value in kwargs.items():
                        if not ztm_verify_data(component, value, operation=f"{operation}_kwarg_{key}"):
                            raise SecurityError(f"Verifikation fehlgeschlagen für Parameter {key}")
                
                # Führe die kritische Operation aus
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                
                # Sichere das Ergebnis
                secured_result = ztm_secure_data(component, result, operation)
                
                # Erfolgreiche kritische Operation
                track("CRITICAL_OPERATION_SUCCESS", component, {
                    "operation": operation,
                    "operation_id": operation_id,
                    "execution_time": execution_time,
                    "result_secured": True,
                    "critical": True
                }, "WARNING")
                
                return secured_result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Kritischer Fehler
                track("CRITICAL_OPERATION_ERROR", component, {
                    "operation": operation,
                    "operation_id": operation_id,
                    "execution_time": execution_time,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "critical": True
                }, "CRITICAL")
                
                raise
        
        return wrapper
    return decorator

class SecurityError(Exception):
    """Ausnahme für Sicherheitsfehler"""
    pass

# Hook-Registrierungsfunktionen
def register_module_hook(module: str, operation: str, hook_func: Callable):
    """Registriert einen Hook für ein Modul"""
    _hook_manager.register_hook(module, operation, hook_func)

def register_critical_module_hooks():
    """Registriert Standard-Hooks für alle kritischen Module"""
    def default_pre_hook(module: str, operation: str, data: Dict[str, Any]):
        logger.debug(f"Pre-Hook: {module}.{operation}")
    
    def default_post_hook(module: str, operation: str, data: Dict[str, Any]):
        logger.debug(f"Post-Hook: {module}.{operation}")
    
    def default_error_hook(module: str, operation: str, data: Dict[str, Any]):
        logger.warning(f"Error-Hook: {module}.{operation} - {data.get('error', 'Unknown error')}")
    
    for module in _hook_manager.critical_modules:
        register_module_hook(module, "operation_pre", default_pre_hook)
        register_module_hook(module, "operation_post", default_post_hook)
        register_module_hook(module, "operation_error", default_error_hook)

# Automatische Registrierung der Standard-Hooks
if ZTM_AVAILABLE and _hook_manager.active:
    register_critical_module_hooks()
    logger.info("ZTM-Hooks für kritische Module registriert")

# Exportiere wichtige Funktionen
__all__ = [
    'ztm_monitored',
    'ztm_secure_data', 
    'ztm_verify_data',
    'ztm_critical_operation',
    'register_module_hook',
    'SecurityError'
]
