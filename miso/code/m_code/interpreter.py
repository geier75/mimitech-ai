#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Interpreter

Dieser Modul implementiert den Interpreter für die M-CODE Programmiersprache.
Der Interpreter führt kompilierten M-CODE Bytecode in der MISO Runtime aus.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import time
import traceback
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.interpreter")

# Import von internen Modulen
from .compiler import MCodeByteCode, compile_m_code


class MCodeRuntimeError(Exception):
    """Fehler während der Ausführung von M-CODE"""
    pass


class MCodeExecutionContext:
    """Ausführungskontext für M-CODE"""
    
    def __init__(self, 
                 global_vars: Optional[Dict[str, Any]] = None, 
                 local_vars: Optional[Dict[str, Any]] = None,
                 parent: Optional['MCodeExecutionContext'] = None):
        """
        Initialisiert einen neuen Ausführungskontext.
        
        Args:
            global_vars: Globale Variablen
            local_vars: Lokale Variablen
            parent: Übergeordneter Kontext
        """
        self.globals = global_vars or {}
        self.locals = local_vars or {}
        self.parent = parent
        self.return_value = None
        self.exception = None
        self.call_stack = []
        self.start_time = time.time()
        self.metrics = {
            "instructions_executed": 0,
            "function_calls": 0,
            "memory_allocated": 0
        }
    
    def get_variable(self, name: str) -> Any:
        """
        Gibt den Wert einer Variable zurück.
        
        Args:
            name: Name der Variable
            
        Returns:
            Wert der Variable
        """
        # Zuerst in lokalen Variablen suchen
        if name in self.locals:
            return self.locals[name]
        
        # Dann in globalen Variablen
        if name in self.globals:
            return self.globals[name]
        
        # Dann im übergeordneten Kontext
        if self.parent is not None:
            return self.parent.get_variable(name)
        
        # Variable nicht gefunden
        raise MCodeRuntimeError(f"Variable '{name}' nicht gefunden")
    
    def set_variable(self, name: str, value: Any) -> None:
        """
        Setzt den Wert einer Variable.
        
        Args:
            name: Name der Variable
            value: Neuer Wert
        """
        # Wenn Variable bereits in lokalen Variablen existiert
        if name in self.locals:
            self.locals[name] = value
            return
        
        # Wenn Variable in globalen Variablen existiert
        if name in self.globals:
            self.globals[name] = value
            return
        
        # Wenn Variable im übergeordneten Kontext existiert
        if self.parent is not None and self.parent.has_variable(name):
            self.parent.set_variable(name, value)
            return
        
        # Neue lokale Variable erstellen
        self.locals[name] = value
    
    def has_variable(self, name: str) -> bool:
        """
        Prüft, ob eine Variable existiert.
        
        Args:
            name: Name der Variable
            
        Returns:
            True, wenn die Variable existiert
        """
        return (name in self.locals or 
                name in self.globals or 
                (self.parent is not None and self.parent.has_variable(name)))
    
    def push_call(self, function_name: str, bytecode: MCodeByteCode) -> None:
        """
        Fügt einen Funktionsaufruf zum Call-Stack hinzu.
        
        Args:
            function_name: Name der Funktion
            bytecode: Bytecode der Funktion
        """
        self.call_stack.append({
            "function": function_name,
            "bytecode": bytecode,
            "line": 0,
            "time": time.time() - self.start_time
        })
        self.metrics["function_calls"] += 1
    
    def pop_call(self) -> None:
        """Entfernt den letzten Funktionsaufruf vom Call-Stack"""
        if self.call_stack:
            self.call_stack.pop()
    
    def get_call_stack(self) -> List[Dict[str, Any]]:
        """
        Gibt den aktuellen Call-Stack zurück.
        
        Returns:
            Liste von Funktionsaufrufen
        """
        return self.call_stack.copy()
    
    def create_child_context(self, local_vars: Optional[Dict[str, Any]] = None) -> 'MCodeExecutionContext':
        """
        Erstellt einen neuen untergeordneten Kontext.
        
        Args:
            local_vars: Lokale Variablen für den neuen Kontext
            
        Returns:
            Neuer Ausführungskontext
        """
        return MCodeExecutionContext(
            global_vars=self.globals,
            local_vars=local_vars or {},
            parent=self
        )


class MCodeInterpreter:
    """Interpreter für die M-CODE Programmiersprache"""
    
    def __init__(self, use_jit: bool = True, memory_limit_mb: int = 1024):
        """
        Initialisiert einen neuen M-CODE Interpreter.
        
        Args:
            use_jit: Just-In-Time Compilation aktivieren
            memory_limit_mb: Speicherlimit in MB
        """
        self.use_jit = use_jit
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Umrechnung in Bytes
        self.instruction_handlers = self._build_instruction_handlers()
        self.builtin_functions = self._register_builtin_functions()
        self.jit_cache = {}
        
        logger.info(f"M-CODE Interpreter initialisiert (JIT: {use_jit}, Speicherlimit: {memory_limit_mb} MB)")
    
    def _build_instruction_handlers(self) -> Dict[str, Callable]:
        """
        Erstellt Handler für Bytecode-Instruktionen.
        
        Returns:
            Dictionary mit Handlern für Instruktionen
        """
        handlers = {}
        
        # Lade-Instruktionen
        handlers["LOAD_CONST"] = self._handle_load_const
        handlers["LOAD_NAME"] = self._handle_load_name
        handlers["STORE_NAME"] = self._handle_store_name
        
        # Arithmetische Operationen
        handlers["BINARY_ADD"] = self._handle_binary_add
        handlers["BINARY_SUBTRACT"] = self._handle_binary_subtract
        handlers["BINARY_MULTIPLY"] = self._handle_binary_multiply
        handlers["BINARY_DIVIDE"] = self._handle_binary_divide
        handlers["BINARY_POWER"] = self._handle_binary_power
        
        # Vergleichsoperationen
        handlers["COMPARE_EQ"] = self._handle_compare_eq
        handlers["COMPARE_NE"] = self._handle_compare_ne
        handlers["COMPARE_LT"] = self._handle_compare_lt
        handlers["COMPARE_LE"] = self._handle_compare_le
        handlers["COMPARE_GT"] = self._handle_compare_gt
        handlers["COMPARE_GE"] = self._handle_compare_ge
        
        # Kontrollfluss
        handlers["JUMP"] = self._handle_jump
        handlers["JUMP_IF_TRUE"] = self._handle_jump_if_true
        handlers["JUMP_IF_FALSE"] = self._handle_jump_if_false
        handlers["CALL_FUNCTION"] = self._handle_call_function
        handlers["RETURN"] = self._handle_return
        
        # Weitere Operationen
        handlers["BUILD_LIST"] = self._handle_build_list
        handlers["BUILD_DICT"] = self._handle_build_dict
        handlers["GET_ITEM"] = self._handle_get_item
        handlers["SET_ITEM"] = self._handle_set_item
        
        return handlers
    
    def _register_builtin_functions(self) -> Dict[str, Callable]:
        """
        Registriert eingebaute Funktionen.
        
        Returns:
            Dictionary mit eingebauten Funktionen
        """
        builtins = {}
        
        # Mathematische Funktionen
        builtins["sin"] = np.sin
        builtins["cos"] = np.cos
        builtins["tan"] = np.tan
        builtins["exp"] = np.exp
        builtins["log"] = np.log
        builtins["sqrt"] = np.sqrt
        
        # Ein-/Ausgabe
        builtins["print"] = print
        builtins["input"] = input
        
        # Typkonvertierung
        builtins["int"] = int
        builtins["float"] = float
        builtins["str"] = str
        builtins["bool"] = bool
        builtins["list"] = list
        builtins["dict"] = dict
        
        # Tensor-Operationen
        builtins["tensor"] = torch.tensor
        builtins["zeros"] = torch.zeros
        builtins["ones"] = torch.ones
        builtins["randn"] = torch.randn
        
        return builtins
    
    def _handle_load_const(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für LOAD_CONST Instruktion"""
        const_idx = instruction["arg"]
        stack.append(bytecode.constants[const_idx])
    
    def _handle_load_name(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für LOAD_NAME Instruktion"""
        name_idx = instruction["arg"]
        name = bytecode.names[name_idx]
        
        # Prüfe zuerst auf eingebaute Funktionen
        if name in self.builtin_functions:
            stack.append(self.builtin_functions[name])
            return
        
        # Sonst aus dem Kontext laden
        value = context.get_variable(name)
        stack.append(value)
    
    def _handle_store_name(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für STORE_NAME Instruktion"""
        name_idx = instruction["arg"]
        name = bytecode.names[name_idx]
        value = stack.pop()
        context.set_variable(name, value)
    
    def _handle_binary_add(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für BINARY_ADD Instruktion"""
        right = stack.pop()
        left = stack.pop()
        stack.append(left + right)
    
    def _handle_binary_subtract(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für BINARY_SUBTRACT Instruktion"""
        right = stack.pop()
        left = stack.pop()
        stack.append(left - right)
    
    def _handle_binary_multiply(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für BINARY_MULTIPLY Instruktion"""
        right = stack.pop()
        left = stack.pop()
        stack.append(left * right)
    
    def _handle_binary_divide(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für BINARY_DIVIDE Instruktion"""
        right = stack.pop()
        left = stack.pop()
        stack.append(left / right)
    
    def _handle_binary_power(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für BINARY_POWER Instruktion"""
        right = stack.pop()
        left = stack.pop()
        stack.append(left ** right)
    
    def _handle_compare_eq(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für COMPARE_EQ Instruktion"""
        right = stack.pop()
        left = stack.pop()
        stack.append(left == right)
    
    def _handle_compare_ne(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für COMPARE_NE Instruktion"""
        right = stack.pop()
        left = stack.pop()
        stack.append(left != right)
    
    def _handle_compare_lt(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für COMPARE_LT Instruktion"""
        right = stack.pop()
        left = stack.pop()
        stack.append(left < right)
    
    def _handle_compare_le(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für COMPARE_LE Instruktion"""
        right = stack.pop()
        left = stack.pop()
        stack.append(left <= right)
    
    def _handle_compare_gt(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für COMPARE_GT Instruktion"""
        right = stack.pop()
        left = stack.pop()
        stack.append(left > right)
    
    def _handle_compare_ge(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für COMPARE_GE Instruktion"""
        right = stack.pop()
        left = stack.pop()
        stack.append(left >= right)
    
    def _handle_jump(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> int:
        """
        Handler für JUMP Instruktion
        
        Returns:
            Neue Programmzählerposition
        """
        return instruction["arg"]
    
    def _handle_jump_if_true(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> Optional[int]:
        """
        Handler für JUMP_IF_TRUE Instruktion
        
        Returns:
            Neue Programmzählerposition oder None
        """
        condition = stack.pop()
        if condition:
            return instruction["arg"]
        return None
    
    def _handle_jump_if_false(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> Optional[int]:
        """
        Handler für JUMP_IF_FALSE Instruktion
        
        Returns:
            Neue Programmzählerposition oder None
        """
        condition = stack.pop()
        if not condition:
            return instruction["arg"]
        return None
    
    def _handle_call_function(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für CALL_FUNCTION Instruktion"""
        arg_count = instruction["arg"]
        args = [stack.pop() for _ in range(arg_count)]
        args.reverse()  # Reihenfolge umkehren
        
        func = stack.pop()
        
        # Funktion aufrufen
        try:
            result = func(*args)
            stack.append(result)
        except Exception as e:
            raise MCodeRuntimeError(f"Fehler beim Aufruf von {func.__name__ if hasattr(func, '__name__') else func}: {e}")
    
    def _handle_return(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für RETURN Instruktion"""
        if stack:
            context.return_value = stack.pop()
        else:
            context.return_value = None
    
    def _handle_build_list(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für BUILD_LIST Instruktion"""
        count = instruction["arg"]
        items = [stack.pop() for _ in range(count)]
        items.reverse()  # Reihenfolge umkehren
        stack.append(items)
    
    def _handle_build_dict(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für BUILD_DICT Instruktion"""
        count = instruction["arg"]
        dictionary = {}
        
        for _ in range(count):
            value = stack.pop()
            key = stack.pop()
            dictionary[key] = value
        
        stack.append(dictionary)
    
    def _handle_get_item(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für GET_ITEM Instruktion"""
        key = stack.pop()
        container = stack.pop()
        stack.append(container[key])
    
    def _handle_set_item(self, instruction: Dict[str, Any], stack: List[Any], context: MCodeExecutionContext, bytecode: MCodeByteCode) -> None:
        """Handler für SET_ITEM Instruktion"""
        value = stack.pop()
        key = stack.pop()
        container = stack.pop()
        container[key] = value
        stack.append(container)
    
    def execute(self, bytecode: MCodeByteCode, context: Optional[MCodeExecutionContext] = None) -> Any:
        """
        Führt M-CODE Bytecode aus.
        
        Args:
            bytecode: M-CODE Bytecode
            context: Ausführungskontext
            
        Returns:
            Rückgabewert der Ausführung
        """
        # Erstelle Kontext, falls keiner übergeben wurde
        if context is None:
            context = MCodeExecutionContext(
                global_vars={
                    "__builtins__": self.builtin_functions
                }
            )
        
        # Verwende JIT-Compilation, falls aktiviert
        if self.use_jit and bytecode.filename in self.jit_cache:
            jit_func = self.jit_cache[bytecode.filename]
            return jit_func(context)
        
        # Stack für die Ausführung
        stack = []
        
        # Führe Instruktionen aus
        pc = 0  # Programmzähler
        while pc < len(bytecode.instructions):
            # Prüfe Speichernutzung
            if self._check_memory_usage() > self.memory_limit:
                raise MCodeRuntimeError("Speicherlimit überschritten")
            
            # Hole aktuelle Instruktion
            instruction = bytecode.instructions[pc]
            opcode = instruction["opcode"]
            
            # Führe Instruktion aus
            if opcode in self.instruction_handlers:
                handler = self.instruction_handlers[opcode]
                result = handler(instruction, stack, context, bytecode)
                
                # Prüfe, ob ein Sprung durchgeführt werden soll
                if result is not None and isinstance(result, int):
                    pc = result
                else:
                    pc += 1
                
                # Aktualisiere Metriken
                context.metrics["instructions_executed"] += 1
            else:
                raise MCodeRuntimeError(f"Unbekannter Opcode: {opcode}")
            
            # Prüfe auf Return
            if context.return_value is not None:
                break
        
        # JIT-Compilation für zukünftige Aufrufe
        if self.use_jit and bytecode.filename not in self.jit_cache:
            self._compile_jit(bytecode)
        
        # Gib Rückgabewert zurück
        return context.return_value if context.return_value is not None else (stack[0] if stack else None)
    
    def _check_memory_usage(self) -> int:
        """
        Prüft die aktuelle Speichernutzung.
        
        Returns:
            Speichernutzung in Bytes
        """
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def _compile_jit(self, bytecode: MCodeByteCode) -> None:
        """
        Kompiliert Bytecode für Just-In-Time Ausführung.
        
        Args:
            bytecode: M-CODE Bytecode
        """
        try:
            import numba
            
            # Diese Methode würde den Bytecode in eine optimierte Funktion umwandeln
            # Hier eine vereinfachte Version für das Beispiel
            def jit_function(context):
                return self.execute(bytecode, context)
            
            # Speichere JIT-Funktion im Cache
            self.jit_cache[bytecode.filename] = jit_function
            
            logger.debug(f"JIT-Compilation für {bytecode.filename} erfolgreich")
        except Exception as e:
            logger.warning(f"JIT-Compilation für {bytecode.filename} fehlgeschlagen: {e}")


def execute_m_code(source: str, 
                  filename: str = "<m_code>", 
                  global_vars: Optional[Dict[str, Any]] = None, 
                  optimization_level: Optional[int] = None,
                  use_jit: Optional[bool] = None) -> Any:
    """
    Kompiliert und führt M-CODE Quellcode aus.
    
    Args:
        source: M-CODE Quellcode
        filename: Name der Quelldatei
        global_vars: Globale Variablen
        optimization_level: Optimierungsstufe (0-3)
        use_jit: Just-In-Time Compilation aktivieren
        
    Returns:
        Rückgabewert der Ausführung
    """
    from .compiler import compile_m_code
    
    # Standardwerte aus Umgebungsvariablen
    if optimization_level is None:
        optimization_level = int(os.environ.get("M_CODE_OPTIMIZATION", "2"))
    
    if use_jit is None:
        use_jit = os.environ.get("M_CODE_USE_JIT", "1") == "1"
    
    # Kompiliere Quellcode
    bytecode = compile_m_code(source, filename, optimization_level)
    
    # Erstelle Interpreter
    interpreter = MCodeInterpreter(use_jit=use_jit)
    
    # Erstelle Kontext
    context = MCodeExecutionContext(global_vars=global_vars or {})
    
    # Führe Bytecode aus
    return interpreter.execute(bytecode, context)
