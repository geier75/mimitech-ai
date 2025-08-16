#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Runtime

Dieses Modul implementiert die Runtime für die M-CODE Programmiersprache.
Die Runtime führt kompilierten M-CODE Bytecode aus.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import sys
import time
import threading
import uuid
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.runtime")

# Import von internen Modulen
from .compiler import MCodeByteCode
from .mlx_adapter import get_mlx_adapter, MLXAdapter, MLX_AVAILABLE
from .ai_optimizer import get_ai_optimizer, OptimizationStrategy, optimize


class MCodeRuntimeError(Exception):
    """Fehler während der Ausführung von M-CODE"""
    pass


class MCodeObject:
    """Basisklasse für alle M-CODE Objekte"""
    
    def __init__(self, type_name: str):
        """
        Initialisiert ein neues M-CODE Objekt.
        
        Args:
            type_name: Name des Typs
        """
        self.type_name = type_name
        self.id = str(uuid.uuid4())
        self.attributes = {}
    
    def __repr__(self) -> str:
        return f"<MCodeObject type={self.type_name} id={self.id}>"
    
    def get_attribute(self, name: str) -> Any:
        """
        Gibt ein Attribut zurück.
        
        Args:
            name: Name des Attributs
            
        Returns:
            Wert des Attributs
            
        Raises:
            MCodeRuntimeError: Wenn das Attribut nicht existiert
        """
        if name in self.attributes:
            return self.attributes[name]
        raise MCodeRuntimeError(f"Attribut '{name}' nicht gefunden in {self.type_name}")
    
    def set_attribute(self, name: str, value: Any) -> None:
        """
        Setzt ein Attribut.
        
        Args:
            name: Name des Attributs
            value: Wert des Attributs
        """
        self.attributes[name] = value
    
    def has_attribute(self, name: str) -> bool:
        """
        Prüft, ob ein Attribut existiert.
        
        Args:
            name: Name des Attributs
            
        Returns:
            True, wenn das Attribut existiert, sonst False
        """
        return name in self.attributes
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert das Objekt in ein Dictionary.
        
        Returns:
            Dictionary-Repräsentation des Objekts
        """
        return {
            "type": self.type_name,
            "id": self.id,
            "attributes": self.attributes
        }


class MCodeFunction(MCodeObject):
    """Repräsentation einer M-CODE Funktion"""
    
    def __init__(self, 
                name: str, 
                bytecode: Optional[MCodeByteCode] = None,
                native_func: Optional[Callable] = None,
                is_method: bool = False,
                parent_class: Optional['MCodeClass'] = None):
        """
        Initialisiert eine neue M-CODE Funktion.
        
        Args:
            name: Name der Funktion
            bytecode: Bytecode der Funktion (optional)
            native_func: Native Python-Funktion (optional)
            is_method: Ist die Funktion eine Methode?
            parent_class: Elternklasse (nur für Methoden)
        """
        super().__init__("function")
        self.name = name
        self.bytecode = bytecode
        self.native_func = native_func
        self.is_method = is_method
        self.parent_class = parent_class
        
        # Metadaten
        self.annotations = {}
        self.documentation = ""
        self.source_file = "<unknown>"
        self.line_number = 0
    
    def __repr__(self) -> str:
        if self.is_method:
            return f"<MCodeMethod {self.name}>"
        return f"<MCodeFunction {self.name}>"
    
    def __call__(self, *args, **kwargs) -> Any:
        """
        Ruft die Funktion auf.
        
        Args:
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Rückgabewert der Funktion
            
        Raises:
            MCodeRuntimeError: Bei Fehlern während der Ausführung
        """
        if self.native_func is not None:
            # Rufe native Funktion auf
            return self.native_func(*args, **kwargs)
        elif self.bytecode is not None:
            # Rufe Bytecode-Funktion auf
            # In einer vollständigen Implementierung würde hier die Bytecode-Ausführung erfolgen
            raise NotImplementedError("Bytecode-Ausführung noch nicht implementiert")
        else:
            raise MCodeRuntimeError(f"Funktion '{self.name}' kann nicht ausgeführt werden")
    
    def execute(self, code, optimize_execution=True, **kwargs):
        """
        Führt M-CODE-Code aus.
        
        Args:
            code: M-CODE-Code als String
            optimize_execution: Ob die Ausführung optimiert werden soll
            **kwargs: Zusätzliche Parameter für den Compiler
            
        Returns:
            Ausführungsergebnis
        """
        try:
            # Kompiliere Code
            compiled_code = self.compiler.compile(code, **kwargs)
            
            # Führe kompilierten Code aus, optional mit Optimierung
            if optimize_execution and self.ai_optimizer:
                # Erstelle einen Ausführungskontext für den AI-Optimizer
                from .ai_optimizer import ExecutionContext
                context = ExecutionContext(
                    code_hash=hash(code),
                    input_shapes=[],
                    input_types=[],
                    is_mcode=True
                )
                
                # Wähle eine Optimierungsstrategie basierend auf dem Code
                strategy = self.ai_optimizer.optimize(code, context)
                
                # Führe mit Optimierung aus
                result = self.ai_optimizer.apply_strategy(
                    strategy, 
                    self.interpreter.execute, 
                    compiled_code
                )
            else:
                # Standard-Ausführung ohne Optimierung
                result = self.interpreter.execute(compiled_code)
            
            return result
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            raise MCodeRuntimeError(f"Fehler bei der Ausführung von {self.name}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert die Funktion in ein Dictionary.
        
        Returns:
            Dictionary-Repräsentation der Funktion
        """
        result = super().to_dict()
        result.update({
            "name": self.name,
            "is_method": self.is_method,
            "has_bytecode": self.bytecode is not None,
            "has_native_func": self.native_func is not None,
            "annotations": self.annotations,
            "documentation": self.documentation,
            "source_file": self.source_file,
            "line_number": self.line_number
        })
        return result


class MCodeClass(MCodeObject):
    """Repräsentation einer M-CODE Klasse"""
    
    def __init__(self, name: str, bases: List['MCodeClass'] = None):
        """
        Initialisiert eine neue M-CODE Klasse.
        
        Args:
            name: Name der Klasse
            bases: Basisklassen
        """
        super().__init__("class")
        self.name = name
        self.bases = bases or []
        self.methods = {}
        self.static_attributes = {}
        
        # Metadaten
        self.annotations = {}
        self.documentation = ""
        self.source_file = "<unknown>"
        self.line_number = 0
    
    def __repr__(self) -> str:
        return f"<MCodeClass {self.name}>"
    
    def add_method(self, method: MCodeFunction) -> None:
        """
        Fügt eine Methode hinzu.
        
        Args:
            method: Hinzuzufügende Methode
        """
        method.is_method = True
        method.parent_class = self
        self.methods[method.name] = method
    
    def get_method(self, name: str) -> MCodeFunction:
        """
        Gibt eine Methode zurück.
        
        Args:
            name: Name der Methode
            
        Returns:
            Methode
            
        Raises:
            MCodeRuntimeError: Wenn die Methode nicht existiert
        """
        if name in self.methods:
            return self.methods[name]
        
        # Suche in Basisklassen
        for base in self.bases:
            try:
                return base.get_method(name)
            except MCodeRuntimeError:
                pass
        
        raise MCodeRuntimeError(f"Methode '{name}' nicht gefunden in Klasse '{self.name}'")
    
    def has_method(self, name: str) -> bool:
        """
        Prüft, ob eine Methode existiert.
        
        Args:
            name: Name der Methode
            
        Returns:
            True, wenn die Methode existiert, sonst False
        """
        if name in self.methods:
            return True
        
        # Suche in Basisklassen
        for base in self.bases:
            if base.has_method(name):
                return True
        
        return False
    
    def create_instance(self, *args, **kwargs) -> 'MCodeInstance':
        """
        Erstellt eine neue Instanz der Klasse.
        
        Args:
            *args: Positionsargumente für den Konstruktor
            **kwargs: Schlüsselwortargumente für den Konstruktor
            
        Returns:
            Neue Instanz
        """
        instance = MCodeInstance(self)
        
        # Rufe Konstruktor auf, falls vorhanden
        if self.has_method("__init__"):
            init_method = self.get_method("__init__")
            init_method(instance, *args, **kwargs)
        
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert die Klasse in ein Dictionary.
        
        Returns:
            Dictionary-Repräsentation der Klasse
        """
        result = super().to_dict()
        result.update({
            "name": self.name,
            "bases": [base.name for base in self.bases],
            "methods": list(self.methods.keys()),
            "static_attributes": self.static_attributes,
            "annotations": self.annotations,
            "documentation": self.documentation,
            "source_file": self.source_file,
            "line_number": self.line_number
        })
        return result


class MCodeInstance(MCodeObject):
    """Repräsentation einer M-CODE Klasseninstanz"""
    
    def __init__(self, cls: MCodeClass):
        """
        Initialisiert eine neue M-CODE Klasseninstanz.
        
        Args:
            cls: Klasse der Instanz
        """
        super().__init__("instance")
        self.cls = cls
        self.attributes = {}
    
    def __repr__(self) -> str:
        return f"<MCodeInstance of {self.cls.name}>"
    
    def get_attribute(self, name: str) -> Any:
        """
        Gibt ein Attribut zurück.
        
        Args:
            name: Name des Attributs
            
        Returns:
            Wert des Attributs
            
        Raises:
            MCodeRuntimeError: Wenn das Attribut nicht existiert
        """
        # Prüfe Instanzattribute
        if name in self.attributes:
            return self.attributes[name]
        
        # Prüfe Methoden
        if self.cls.has_method(name):
            return self.cls.get_method(name)
        
        # Prüfe statische Attribute
        if name in self.cls.static_attributes:
            return self.cls.static_attributes[name]
        
        raise MCodeRuntimeError(f"Attribut '{name}' nicht gefunden in Instanz von '{self.cls.name}'")
    
    def set_attribute(self, name: str, value: Any) -> None:
        """
        Setzt ein Attribut.
        
        Args:
            name: Name des Attributs
            value: Wert des Attributs
        """
        self.attributes[name] = value
    
    def has_attribute(self, name: str) -> bool:
        """
        Prüft, ob ein Attribut existiert.
        
        Args:
            name: Name des Attributs
            
        Returns:
            True, wenn das Attribut existiert, sonst False
        """
        return (name in self.attributes or 
                self.cls.has_method(name) or 
                name in self.cls.static_attributes)
    
    def call_method(self, name: str, *args, **kwargs) -> Any:
        """
        Ruft eine Methode auf.
        
        Args:
            name: Name der Methode
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Rückgabewert der Methode
            
        Raises:
            MCodeRuntimeError: Wenn die Methode nicht existiert
        """
        if not self.cls.has_method(name):
            raise MCodeRuntimeError(f"Methode '{name}' nicht gefunden in Instanz von '{self.cls.name}'")
        
        method = self.cls.get_method(name)
        return method(self, *args, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert die Instanz in ein Dictionary.
        
        Returns:
            Dictionary-Repräsentation der Instanz
        """
        result = super().to_dict()
        result.update({
            "class": self.cls.name,
            "attributes": self.attributes
        })
        return result


class MCodeModule(MCodeObject):
    """Repräsentation eines M-CODE Moduls"""
    
    def __init__(self, name: str):
        """
        Initialisiert ein neues M-CODE Modul.
        
        Args:
            name: Name des Moduls
        """
        super().__init__("module")
        self.name = name
        self.functions = {}
        self.classes = {}
        self.variables = {}
        self.submodules = {}
        
        # Metadaten
        self.documentation = ""
        self.source_file = "<unknown>"
    
    def __repr__(self) -> str:
        return f"<MCodeModule {self.name}>"
    
    def add_function(self, function: MCodeFunction) -> None:
        """
        Fügt eine Funktion hinzu.
        
        Args:
            function: Hinzuzufügende Funktion
        """
        self.functions[function.name] = function
    
    def add_class(self, cls: MCodeClass) -> None:
        """
        Fügt eine Klasse hinzu.
        
        Args:
            cls: Hinzuzufügende Klasse
        """
        self.classes[cls.name] = cls
    
    def add_variable(self, name: str, value: Any) -> None:
        """
        Fügt eine Variable hinzu.
        
        Args:
            name: Name der Variable
            value: Wert der Variable
        """
        self.variables[name] = value
    
    def add_submodule(self, module: 'MCodeModule') -> None:
        """
        Fügt ein Submodul hinzu.
        
        Args:
            module: Hinzuzufügendes Submodul
        """
        self.submodules[module.name] = module
    
    def get_attribute(self, name: str) -> Any:
        """
        Gibt ein Attribut zurück.
        
        Args:
            name: Name des Attributs
            
        Returns:
            Wert des Attributs
            
        Raises:
            MCodeRuntimeError: Wenn das Attribut nicht existiert
        """
        if name in self.functions:
            return self.functions[name]
        elif name in self.classes:
            return self.classes[name]
        elif name in self.variables:
            return self.variables[name]
        elif name in self.submodules:
            return self.submodules[name]
        
        raise MCodeRuntimeError(f"Attribut '{name}' nicht gefunden in Modul '{self.name}'")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert das Modul in ein Dictionary.
        
        Returns:
            Dictionary-Repräsentation des Moduls
        """
        result = super().to_dict()
        result.update({
            "name": self.name,
            "functions": list(self.functions.keys()),
            "classes": list(self.classes.keys()),
            "variables": list(self.variables.keys()),
            "submodules": list(self.submodules.keys()),
            "documentation": self.documentation,
            "source_file": self.source_file
        })
        return result


class MCodeRuntime:
    """Runtime für die M-CODE Programmiersprache"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert eine neue M-CODE Runtime.
        
        Args:
            config: Konfiguration der Runtime
        """
        self.config = config or {}
        self.modules = {}
        self.global_functions = {}
        self.global_variables = {}
        self.execution_context = None
        self.thread_local = threading.local()
        
        # Konfiguration
        self.optimization_level = self.config.get("optimization_level", 2)
        self.use_jit = self.config.get("use_jit", True)
        self.memory_limit_mb = self.config.get("memory_limit_mb", 1024)
        self.enable_extensions = self.config.get("enable_extensions", True)
        self.use_ane = self.config.get("use_ane", True)
        self.fallback_on_error = self.config.get("fallback_on_error", True)
        
        # Initialisiere MLX-Adapter
        self._initialize_mlx_adapter()
        
        # Initialisiere den AI-Optimizer
        self.ai_optimizer = get_ai_optimizer()
        
        # Initialisiere Standardbibliothek
        self._initialize_stdlib()
        
        logger.info(f"M-CODE Runtime initialisiert mit Konfiguration: {self.config}")
    
    def _initialize_mlx_adapter(self) -> None:
        """Initialisiert den MLX-Adapter für Tensor-Operationen"""
        try:
            # Initialisiere MLX-Adapter mit den Konfigurationsoptionen
            self.mlx_adapter = get_mlx_adapter(
                use_ane=self.use_ane,
                fallback_on_error=self.fallback_on_error
            )
            
            # Informationen über den Adapter protokollieren
            device_info = self.mlx_adapter.get_device_info()
            logger.info(f"MLX-Adapter initialisiert: {device_info}")
            
            # Status in die Konfiguration schreiben
            self.config["mlx_available"] = device_info["mlx_available"]
            self.config["ane_available"] = device_info["ane_available"]
            
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des MLX-Adapters: {e}")
            self.mlx_adapter = None
            self.config["mlx_available"] = False
            self.config["ane_available"] = False
    
    def _initialize_stdlib(self) -> None:
        """Initialisiert die Standardbibliothek"""
        # Erstelle stdlib-Modul
        stdlib = MCodeModule("stdlib")
        
        # Füge Standardfunktionen hinzu
        for name, func in self._get_stdlib_functions().items():
            stdlib.add_function(MCodeFunction(name, native_func=func))
        
        # Registriere Modul
        self.register_module(stdlib)
        
        logger.info("Standardbibliothek initialisiert")
    
    def _get_stdlib_functions(self) -> Dict[str, Callable]:
        """Gibt die Standardbibliotheksfunktionen zurück"""
        return {
            "print": print,
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "type": type,
            "isinstance": isinstance,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "delattr": delattr,
            "id": id,
            "hash": hash,
            "help": help,
            "dir": dir,
            "vars": vars,
            "globals": globals,
            "locals": locals,
            "input": input,
            "open": open,
            "exit": exit,
            "quit": quit,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
            "next": next,
            "iter": iter,
            "pow": pow,
            "divmod": divmod,
            "bin": bin,
            "hex": hex,
            "oct": oct,
            "chr": chr,
            "ord": ord,
            "format": format,
            "repr": repr,
            "ascii": ascii,
            "bytes": bytes,
            "bytearray": bytearray,
            "memoryview": memoryview,
            "complex": complex,
            "frozenset": frozenset,
            "property": property,
            "slice": slice,
            "super": super,
            "staticmethod": staticmethod,
            "classmethod": classmethod,
            "compile": compile,
            "eval": eval,
            "exec": exec,
            "breakpoint": breakpoint,
            "callable": callable,
            "issubclass": issubclass,
            "object": object,
            "vars": vars,
            "locals": locals,
            "globals": globals,
            "dir": dir,
            "time": time.time,
            "sleep": time.sleep,
            "np": np,
            "torch": torch
        }
    
    def register_module(self, module: MCodeModule) -> None:
        """
        Registriert ein Modul.
        
        Args:
            module: Zu registrierendes Modul
        """
        self.modules[module.name] = module
        logger.info(f"Modul '{module.name}' registriert")
    
    def get_module(self, name: str) -> MCodeModule:
        """
        Gibt ein Modul zurück.
        
        Args:
            name: Name des Moduls
            
        Returns:
            Modul
            
        Raises:
            MCodeRuntimeError: Wenn das Modul nicht existiert
        """
        if name in self.modules:
            return self.modules[name]
        
        raise MCodeRuntimeError(f"Modul '{name}' nicht gefunden")
    
    def register_function(self, function: MCodeFunction) -> None:
        """
        Registriert eine globale Funktion.
        
        Args:
            function: Zu registrierende Funktion
        """
        self.global_functions[function.name] = function
        logger.info(f"Globale Funktion '{function.name}' registriert")
    
    def register_variable(self, name: str, value: Any) -> None:
        """
        Registriert eine globale Variable.
        
        Args:
            name: Name der Variable
            value: Wert der Variable
        """
        self.global_variables[name] = value
        logger.info(f"Globale Variable '{name}' registriert")
    
    def execute_tensor_operation(self, operation: str, *args, **kwargs) -> Any:
        """
        Führt eine Tensor-Operation mit dem MLX-Adapter aus.
        
        Args:
            operation: Name der Operation
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Ergebnis der Operation
            
        Raises:
            MCodeRuntimeError: Bei Fehlern während der Ausführung
        """
        if self.mlx_adapter is None:
            raise RuntimeError("MLX-Adapter ist nicht initialisiert")
        
        # Rufe die optimierte Ausführungsmethode auf
        return self._optimized_execute_operation(operation, *args, **kwargs)
    
    @optimize
    def _optimized_execute_operation(self, operation: str, *args, **kwargs) -> Any:
        """
        Optimierte Ausführung einer Tensor-Operation.
        
        Args:
            operation: Name der Operation
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Ergebnis der Operation
        """
        # Führe die Operation aus
        return self.mlx_adapter.execute_operation(operation, *args, **kwargs)
    
    def execute(self, bytecode: MCodeByteCode, globals: Dict[str, Any] = None, locals: Dict[str, Any] = None) -> Any:
        """
        Führt M-CODE Bytecode aus.
        
        Args:
            bytecode: Auszuführender Bytecode
            globals: Globale Variablen
            locals: Lokale Variablen
            
        Returns:
            Rückgabewert der Ausführung
            
        Raises:
            MCodeRuntimeError: Bei Fehlern während der Ausführung
        """
        # In einer vollständigen Implementierung würde hier die Bytecode-Ausführung erfolgen
        # Für diese Beispielimplementierung geben wir None zurück
        
        logger.info(f"Führe Bytecode aus: {bytecode}")
        
        return None
    
    def execute_function(self, function: MCodeFunction, *args, **kwargs) -> Any:
        """
        Führt eine M-CODE Funktion aus.
        
        Args:
            function: Auszuführende Funktion
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Rückgabewert der Funktion
            
        Raises:
            MCodeRuntimeError: Bei Fehlern während der Ausführung
        """
        return function(*args, **kwargs)
    
    def execute_string(self, source: str, globals: Dict[str, Any] = None, locals: Dict[str, Any] = None) -> Any:
        """
        Führt M-CODE Quellcode aus.
        
        Args:
            source: Auszuführender Quellcode
            globals: Globale Variablen
            locals: Lokale Variablen
            
        Returns:
            Rückgabewert der Ausführung
            
        Raises:
            MCodeRuntimeError: Bei Fehlern während der Ausführung
        """
        # In einer vollständigen Implementierung würde hier die Kompilierung und Ausführung erfolgen
        # Für diese Beispielimplementierung geben wir None zurück
        
        logger.info(f"Führe Quellcode aus: {source[:50]}...")
        
        return None
    
    def shutdown(self) -> None:
        """Fährt die Runtime herunter"""
        logger.info("M-CODE Runtime wird heruntergefahren")
        
        # Führe Cleanup durch
        self.modules.clear()
        self.global_functions.clear()
        self.global_variables.clear()
        
        logger.info("M-CODE Runtime heruntergefahren")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert die Runtime in ein Dictionary.
        
        Returns:
            Dictionary-Repräsentation der Runtime
        """
        return {
            "config": self.config,
            "modules": list(self.modules.keys()),
            "global_functions": list(self.global_functions.keys()),
            "global_variables": list(self.global_variables.keys())
        }
