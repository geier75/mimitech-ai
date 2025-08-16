#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Runtime

Dieser Modul implementiert die Laufzeitumgebung für die M-CODE Programmiersprache.
Die Runtime verwaltet die Ausführung von M-CODE und stellt eine sichere Umgebung bereit.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import time
import uuid
import threading
import queue
from enum import Enum, auto
from typing import Dict, Any, List, Tuple, Optional, Union, Set, Callable

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.runtime")

# Importiere interne Module
from .mcode_ast import MCodeBytecode, BytecodeOp
from .mcode_jit import GPUJITEngine, JITCompiledCode, JITTarget, JITOptimizationLevel
from .mcode_security import SecuritySandbox, SecurityLevel, SecurityViolation, CodeVerifier


class MCodeObject:
    """Basisklasse für alle M-CODE-Objekte
    
    Diese Klasse dient als Basisklasse für alle M-CODE-Objekte und
    stellt gemeinsame Funktionalitäten bereit.
    """
    
    def __init__(self, name: str, object_type: str, runtime=None):
        """
        Initialisiert ein neues M-CODE-Objekt.
        
        Args:
            name: Name des Objekts
            object_type: Typ des Objekts
            runtime: Laufzeitumgebung (optional)
        """
        self.name = name
        self.object_type = object_type
        self.runtime = runtime
        self.attributes = {}
        self.created_at = time.time()
        self.last_accessed = self.created_at
        
    def __getattr__(self, name):
        """
        Gibt ein Attribut zurück.
        
        Args:
            name: Name des Attributs
            
        Returns:
            Wert des Attributs
            
        Raises:
            AttributeError: Wenn das Attribut nicht existiert
        """
        if name in self.attributes:
            self.last_accessed = time.time()
            return self.attributes[name]
        raise AttributeError(f"'{self.object_type}' Objekt hat kein Attribut '{name}'")
    
    def __setattr__(self, name, value):
        """
        Setzt ein Attribut.
        
        Args:
            name: Name des Attributs
            value: Wert des Attributs
        """
        if name in ["name", "object_type", "runtime", "attributes", "created_at", "last_accessed"]:
            super().__setattr__(name, value)
        else:
            self.attributes[name] = value
            self.last_accessed = time.time()
    
    def __repr__(self):
        """
        String-Repräsentation des Objekts.
        
        Returns:
            String-Repräsentation
        """
        return f"<MCodeObject {self.object_type}:{self.name}>"
    
    def __str__(self):
        """
        String-Repräsentation des Objekts.
        
        Returns:
            String-Repräsentation
        """
        return f"{self.object_type}:{self.name}"
    
    def get_attributes(self):
        """
        Gibt alle Attribute zurück.
        
        Returns:
            Dictionary mit Attributen
        """
        self.last_accessed = time.time()
        return self.attributes.copy()


class MCodeFunction(MCodeObject):
    """Repräsentiert eine M-CODE-Funktion
    
    Diese Klasse kapselt eine kompilierte M-CODE-Funktion und ermöglicht
    ihre Ausführung in verschiedenen Kontexten.
    """
    
    def __init__(self, 
                 name: str,
                 bytecode: MCodeBytecode,
                 runtime,
                 context_id: Optional[str] = None,
                 compiled_code: Optional[JITCompiledCode] = None,
                 signature: Optional[str] = None,
                 doc: Optional[str] = None):
        """
        Initialisiert eine neue M-CODE-Funktion.
        
        Args:
            name: Name der Funktion
            bytecode: Bytecode der Funktion
            runtime: Laufzeitumgebung
            context_id: ID des Kontexts (optional)
            compiled_code: Kompilierter Code (optional)
            signature: Signatur der Funktion (optional)
            doc: Dokumentation der Funktion (optional)
        """
        self.name = name
        self.bytecode = bytecode
        self.runtime = runtime
        self.context_id = context_id
        self.compiled_code = compiled_code
        self.signature = signature or "(*args, **kwargs)"
        self.doc = doc or ""
        self.call_count = 0
        self.last_call_time = 0
        self.total_execution_time = 0.0
        
    def __call__(self, *args, **kwargs):
        """
        Ruft die Funktion auf.
        
        Args:
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Ergebnis der Funktion
            
        Raises:
            Exception: Bei Ausführungsfehlern
        """
        # Hole Kontext
        context = self.runtime.get_context(self.context_id) if self.context_id else None
        
        # Erstelle neuen Kontext, wenn keiner vorhanden
        if context is None:
            context = self.runtime.create_context()
            self.context_id = context.id
        
        # Füge Argumente zum Kontext hinzu
        context.set_args(args, kwargs)
        
        # Führe Bytecode aus
        start_time = time.time()
        result = self.runtime.execute(self.bytecode, context)
        end_time = time.time()
        
        # Aktualisiere Statistiken
        self.call_count += 1
        self.last_call_time = start_time
        self.total_execution_time += (end_time - start_time)
        
        # Prüfe auf Fehler
        if result.status == ExecutionStatus.FAILED:
            if result.error:
                raise result.error
            else:
                raise RuntimeError(f"Ausführung von {self.name} fehlgeschlagen")
        
        # Gib Ergebnis zurück
        return result.result
    
    def __repr__(self):
        """
        String-Repräsentation der Funktion.
        
        Returns:
            String-Repräsentation
        """
        return f"<MCodeFunction {self.name}{self.signature}>"
    
    def __str__(self):
        """
        String-Repräsentation der Funktion.
        
        Returns:
            String-Repräsentation
        """
        return f"{self.name}{self.signature}"
    
    def get_stats(self):
        """
        Gibt Statistiken zur Funktion zurück.
        
        Returns:
            Dictionary mit Statistiken
        """
        return {
            "name": self.name,
            "call_count": self.call_count,
            "last_call_time": self.last_call_time,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": self.total_execution_time / max(1, self.call_count)
        }


class ExecutionMode(Enum):
    """Ausführungsmodi für M-CODE"""
    
    INTERPRETED = auto()  # Interpretierte Ausführung
    JIT = auto()          # JIT-kompilierte Ausführung
    AOT = auto()          # Ahead-of-Time-kompilierte Ausführung


class ExecutionStatus(Enum):
    """Status der Ausführung"""
    
    PENDING = auto()    # Ausstehend
    RUNNING = auto()    # Läuft
    COMPLETED = auto()  # Abgeschlossen
    FAILED = auto()     # Fehlgeschlagen
    TIMEOUT = auto()    # Zeitüberschreitung


class ExecutionContext:
    """Ausführungskontext für M-CODE"""
    
    def __init__(self, 
                 globals_dict: Optional[Dict[str, Any]] = None,
                 locals_dict: Optional[Dict[str, Any]] = None,
                 module_name: str = "<m_code>"):
        """
        Initialisiert einen neuen Ausführungskontext.
        
        Args:
            globals_dict: Globale Variablen
            locals_dict: Lokale Variablen
            module_name: Name des Moduls
        """
        self.globals_dict = globals_dict or {}
        self.locals_dict = locals_dict or {}
        self.module_name = module_name
        self.id = str(uuid.uuid4())
        self.creation_time = time.time()
        self.last_access_time = self.creation_time
    
    def update_access_time(self) -> None:
        """Aktualisiert die Zugriffszeit"""
        self.last_access_time = time.time()
    
    def get_variable(self, name: str) -> Any:
        """
        Gibt den Wert einer Variable zurück.
        
        Args:
            name: Name der Variable
            
        Returns:
            Wert der Variable
            
        Raises:
            KeyError: Wenn die Variable nicht existiert
        """
        self.update_access_time()
        
        if name in self.locals_dict:
            return self.locals_dict[name]
        elif name in self.globals_dict:
            return self.globals_dict[name]
        else:
            raise KeyError(f"Variable '{name}' nicht gefunden")
    
    def set_variable(self, name: str, value: Any) -> None:
        """
        Setzt den Wert einer Variable.
        
        Args:
            name: Name der Variable
            value: Wert der Variable
        """
        self.update_access_time()
        self.locals_dict[name] = value
    
    def get_all_variables(self) -> Dict[str, Any]:
        """
        Gibt alle Variablen zurück.
        
        Returns:
            Dictionary mit allen Variablen
        """
        self.update_access_time()
        
        # Kombiniere globale und lokale Variablen
        variables = {}
        variables.update(self.globals_dict)
        variables.update(self.locals_dict)
        
        return variables


class ExecutionResult:
    """Ergebnis einer Ausführung"""
    
    def __init__(self, 
                 status: ExecutionStatus,
                 result: Any = None,
                 error: Optional[Exception] = None,
                 execution_time: float = 0.0,
                 context: Optional[ExecutionContext] = None):
        """
        Initialisiert ein neues Ausführungsergebnis.
        
        Args:
            status: Status der Ausführung
            result: Ergebnis der Ausführung
            error: Fehler (falls vorhanden)
            execution_time: Ausführungszeit in Sekunden
            context: Ausführungskontext
        """
        self.status = status
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.context = context
        self.timestamp = time.time()
    
    def __repr__(self) -> str:
        """String-Repräsentation des Ergebnisses"""
        if self.status == ExecutionStatus.COMPLETED:
            return f"<ExecutionResult status={self.status.name} result={self.result} time={self.execution_time:.6f}s>"
        elif self.status == ExecutionStatus.FAILED:
            return f"<ExecutionResult status={self.status.name} error={self.error} time={self.execution_time:.6f}s>"
        else:
            return f"<ExecutionResult status={self.status.name} time={self.execution_time:.6f}s>"


class BytecodeInterpreter:
    """Interpreter für M-CODE Bytecode"""
    
    def __init__(self, sandbox: SecuritySandbox):
        """
        Initialisiert einen neuen Bytecode-Interpreter.
        
        Args:
            sandbox: Sicherheits-Sandbox
        """
        self.sandbox = sandbox
    
    def execute(self, bytecode: MCodeBytecode, context: ExecutionContext) -> Any:
        """
        Führt Bytecode aus.
        
        Args:
            bytecode: M-CODE Bytecode
            context: Ausführungskontext
            
        Returns:
            Ergebnis der Ausführung
        """
        # Initialisiere Stack
        stack = []
        
        # Initialisiere Programmzähler
        pc = 0
        
        # Hole Variablen aus dem Kontext
        globals_dict = context.globals_dict
        locals_dict = context.locals_dict
        
        # Führe Bytecode aus
        while pc < len(bytecode.instructions):
            # Hole aktuelle Instruktion
            instruction = bytecode.instructions[pc]
            opcode = instruction["opcode"]
            arg = instruction["arg"]
            
            # Führe Instruktion aus
            if opcode == BytecodeOp.LOAD_CONST:
                # Lade Konstante
                stack.append(bytecode.constants[arg])
            
            elif opcode == BytecodeOp.LOAD_NAME:
                # Lade Variable
                name = bytecode.names[arg]
                if name in locals_dict:
                    stack.append(locals_dict[name])
                elif name in globals_dict:
                    stack.append(globals_dict[name])
                else:
                    raise NameError(f"Name '{name}' ist nicht definiert")
            
            elif opcode == BytecodeOp.STORE_NAME:
                # Speichere Variable
                name = bytecode.names[arg]
                value = stack.pop()
                locals_dict[name] = value
            
            elif opcode == BytecodeOp.BINARY_ADD:
                # Addition
                right = stack.pop()
                left = stack.pop()
                stack.append(left + right)
            
            elif opcode == BytecodeOp.BINARY_SUBTRACT:
                # Subtraktion
                right = stack.pop()
                left = stack.pop()
                stack.append(left - right)
            
            elif opcode == BytecodeOp.BINARY_MULTIPLY:
                # Multiplikation
                right = stack.pop()
                left = stack.pop()
                stack.append(left * right)
            
            elif opcode == BytecodeOp.BINARY_DIVIDE:
                # Division
                right = stack.pop()
                left = stack.pop()
                stack.append(left / right)
            
            elif opcode == BytecodeOp.BINARY_POWER:
                # Potenzierung
                right = stack.pop()
                left = stack.pop()
                stack.append(left ** right)
            
            elif opcode == BytecodeOp.BINARY_MATMUL:
                # Matrix-Multiplikation
                right = stack.pop()
                left = stack.pop()
                stack.append(left @ right)
            
            elif opcode == BytecodeOp.COMPARE_EQ:
                # Gleichheit
                right = stack.pop()
                left = stack.pop()
                stack.append(left == right)
            
            elif opcode == BytecodeOp.COMPARE_NE:
                # Ungleichheit
                right = stack.pop()
                left = stack.pop()
                stack.append(left != right)
            
            elif opcode == BytecodeOp.COMPARE_LT:
                # Kleiner als
                right = stack.pop()
                left = stack.pop()
                stack.append(left < right)
            
            elif opcode == BytecodeOp.COMPARE_LE:
                # Kleiner oder gleich
                right = stack.pop()
                left = stack.pop()
                stack.append(left <= right)
            
            elif opcode == BytecodeOp.COMPARE_GT:
                # Größer als
                right = stack.pop()
                left = stack.pop()
                stack.append(left > right)
            
            elif opcode == BytecodeOp.COMPARE_GE:
                # Größer oder gleich
                right = stack.pop()
                left = stack.pop()
                stack.append(left >= right)
            
            elif opcode == BytecodeOp.JUMP:
                # Unbedingter Sprung
                pc = arg
                continue
            
            elif opcode == BytecodeOp.JUMP_IF_TRUE:
                # Bedingter Sprung (wenn wahr)
                condition = stack.pop()
                if condition:
                    pc = arg
                    continue
            
            elif opcode == BytecodeOp.JUMP_IF_FALSE:
                # Bedingter Sprung (wenn falsch)
                condition = stack.pop()
                if not condition:
                    pc = arg
                    continue
            
            elif opcode == BytecodeOp.CALL_FUNCTION:
                # Funktionsaufruf
                args = []
                for _ in range(arg):
                    args.insert(0, stack.pop())
                
                func = stack.pop()
                result = func(*args)
                stack.append(result)
            
            elif opcode == BytecodeOp.RETURN:
                # Rückgabe
                if stack:
                    return stack.pop()
                else:
                    return None
            
            elif opcode == BytecodeOp.BUILD_LIST:
                # Liste erstellen
                items = []
                for _ in range(arg):
                    items.insert(0, stack.pop())
                stack.append(items)
            
            elif opcode == BytecodeOp.BUILD_DICT:
                # Dictionary erstellen
                items = {}
                for _ in range(arg):
                    value = stack.pop()
                    key = stack.pop()
                    items[key] = value
                stack.append(items)
            
            elif opcode == BytecodeOp.GET_ITEM:
                # Element holen
                key = stack.pop()
                container = stack.pop()
                stack.append(container[key])
            
            elif opcode == BytecodeOp.SET_ITEM:
                # Element setzen
                value = stack.pop()
                key = stack.pop()
                container = stack.pop()
                container[key] = value
            
            elif opcode == BytecodeOp.REGISTER_EVENT:
                # Ereignis registrieren
                event_name = stack.pop()
                handler = stack.pop()
                
                # Hier würde die eigentliche Ereignisregistrierung stattfinden
                # Für dieses Beispiel ignorieren wir sie
                logger.debug(f"Ereignis registriert: {event_name}")
            
            elif opcode == BytecodeOp.TRIGGER_EVENT:
                # Ereignis auslösen
                event_args = stack.pop()
                event_name = stack.pop()
                
                # Hier würde die eigentliche Ereignisauslösung stattfinden
                # Für dieses Beispiel ignorieren wir sie
                logger.debug(f"Ereignis ausgelöst: {event_name}")
            
            elif opcode == BytecodeOp.TENSOR_TRANSPOSE:
                # Tensor transponieren
                tensor = stack.pop()
                stack.append(tensor.T)
            
            elif opcode == BytecodeOp.TENSOR_DOT:
                # Tensor-Punktprodukt
                right = stack.pop()
                left = stack.pop()
                stack.append(left.dot(right))
            
            elif opcode == BytecodeOp.TENSOR_NORMALIZE:
                # Tensor normalisieren
                tensor = stack.pop()
                norm = tensor.norm()
                if norm > 0:
                    stack.append(tensor / norm)
                else:
                    stack.append(tensor)
            
            else:
                # Unbekannte Instruktion
                raise ValueError(f"Unbekannte Instruktion: {opcode}")
            
            # Erhöhe Programmzähler
            pc += 1
        
        # Rückgabe des obersten Stack-Elements (falls vorhanden)
        if stack:
            return stack.pop()
        else:
            return None


class MCodeRuntime:
    """Laufzeitumgebung für M-CODE"""
    
    def __init__(self, 
                 security_level: SecurityLevel = SecurityLevel.MEDIUM,
                 execution_mode: ExecutionMode = ExecutionMode.JIT,
                 use_gpu: bool = True,
                 use_neural_engine: bool = True,
                 optimization_level: int = 2):
        """
        Initialisiert eine neue M-CODE Laufzeitumgebung.
        
        Args:
            security_level: Sicherheitsstufe
            execution_mode: Ausführungsmodus
            use_gpu: GPU-Beschleunigung aktivieren
            use_neural_engine: Apple Neural Engine verwenden (falls verfügbar)
            optimization_level: Optimierungsstufe (0-3)
        """
        # Initialisiere Sicherheits-Sandbox
        self.sandbox = SecuritySandbox(security_level)
        self.code_verifier = CodeVerifier(self.sandbox)
        
        # Initialisiere JIT-Compiler
        self.jit_engine = GPUJITEngine(
            use_gpu=use_gpu,
            use_neural_engine=use_neural_engine,
            optimization_level=JITOptimizationLevel(optimization_level)
        )
        
        # Initialisiere Bytecode-Interpreter
        self.interpreter = BytecodeInterpreter(self.sandbox)
        
        # Setze Ausführungsmodus
        self.execution_mode = execution_mode
        
        # Initialisiere Kontexte
        self.contexts = {}
        
        # Initialisiere Thread-Pool für asynchrone Ausführung
        self.thread_pool = []
        self.max_threads = 4
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Starte Worker-Threads
        self._start_workers()
        
        # Sicherheitsebene-Anzeige anpassen, je nachdem ob es ein Enum oder ein Dictionary ist
        security_level_name = security_level.name if hasattr(security_level, 'name') else 'MEDIUM'
        execution_mode_name = execution_mode.name if hasattr(execution_mode, 'name') else 'STANDARD'
        logger.info(f"M-CODE Runtime initialisiert: Mode={execution_mode_name}, Security={security_level_name}")
    
    def _start_workers(self) -> None:
        """Startet Worker-Threads für asynchrone Ausführung"""
        for _ in range(self.max_threads):
            thread = threading.Thread(target=self._worker_thread, daemon=True)
            thread.start()
            self.thread_pool.append(thread)
    
    def _worker_thread(self) -> None:
        """Worker-Thread für asynchrone Ausführung"""
        while True:
            try:
                # Hole nächste Aufgabe
                task = self.task_queue.get()
                
                if task is None:
                    # Beende Thread
                    self.task_queue.task_done()
                    break
                
                # Entpacke Aufgabe
                bytecode, context, callback = task
                
                # Führe Bytecode aus
                try:
                    start_time = time.time()
                    
                    if self.execution_mode == ExecutionMode.JIT:
                        # JIT-kompilierte Ausführung
                        compiled_code = self.jit_engine.compile(bytecode)
                        result = self.jit_engine.execute(compiled_code, context.get_all_variables())
                    else:
                        # Interpretierte Ausführung
                        result = self.interpreter.execute(bytecode, context)
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # Erstelle Ergebnis
                    execution_result = ExecutionResult(
                        status=ExecutionStatus.COMPLETED,
                        result=result,
                        execution_time=execution_time,
                        context=context
                    )
                
                except Exception as e:
                    # Fehler bei der Ausführung
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    execution_result = ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        error=e,
                        execution_time=execution_time,
                        context=context
                    )
                
                # Rufe Callback auf
                if callback:
                    callback(execution_result)
                
                # Füge Ergebnis zur Ergebnisqueue hinzu
                self.result_queue.put(execution_result)
                
                # Markiere Aufgabe als erledigt
                self.task_queue.task_done()
            
            except Exception as e:
                # Fehler im Worker-Thread
                logger.error(f"Fehler im Worker-Thread: {e}")
    
    def create_context(self, 
                      globals_dict: Optional[Dict[str, Any]] = None,
                      locals_dict: Optional[Dict[str, Any]] = None,
                      module_name: str = "<m_code>") -> ExecutionContext:
        """
        Erstellt einen neuen Ausführungskontext.
        
        Args:
            globals_dict: Globale Variablen
            locals_dict: Lokale Variablen
            module_name: Name des Moduls
            
        Returns:
            Ausführungskontext
        """
        # Erstelle sicheres globals-Dictionary
        if globals_dict is None:
            globals_dict = self.sandbox.create_secure_globals()
        
        # Erstelle sicheres locals-Dictionary
        if locals_dict is None:
            locals_dict = self.sandbox.create_secure_locals()
        
        # Erstelle Kontext
        context = ExecutionContext(globals_dict, locals_dict, module_name)
        
        # Speichere Kontext
        self.contexts[context.id] = context
        
        return context
    
    def get_context(self, context_id: str) -> Optional[ExecutionContext]:
        """
        Gibt einen Ausführungskontext zurück.
        
        Args:
            context_id: ID des Kontexts
            
        Returns:
            Ausführungskontext oder None
        """
        return self.contexts.get(context_id)
    
    def execute(self, 
               bytecode: MCodeBytecode,
               context: Optional[ExecutionContext] = None,
               async_execution: bool = False,
               callback: Optional[Callable[[ExecutionResult], None]] = None) -> Union[ExecutionResult, str]:
        """
        Führt Bytecode aus.
        
        Args:
            bytecode: M-CODE Bytecode
            context: Ausführungskontext
            async_execution: Asynchrone Ausführung
            callback: Callback-Funktion für asynchrone Ausführung
            
        Returns:
            Ausführungsergebnis oder Task-ID (bei asynchroner Ausführung)
        """
        # Erstelle Kontext, falls keiner angegeben wurde
        if context is None:
            context = self.create_context()
        
        # Asynchrone Ausführung
        if async_execution:
            # Erstelle Task-ID
            task_id = str(uuid.uuid4())
            
            # Füge Aufgabe zur Queue hinzu
            self.task_queue.put((bytecode, context, callback))
            
            return task_id
        
        # Synchrone Ausführung
        try:
            start_time = time.time()
            
            if self.execution_mode == ExecutionMode.JIT:
                # JIT-kompilierte Ausführung
                compiled_code = self.jit_engine.compile(bytecode)
                result = self.jit_engine.execute(compiled_code, context.get_all_variables())
            else:
                # Interpretierte Ausführung
                result = self.interpreter.execute(bytecode, context)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Erstelle Ergebnis
            execution_result = ExecutionResult(
                status=ExecutionStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                context=context
            )
        
        except Exception as e:
            # Fehler bei der Ausführung
            end_time = time.time()
            execution_time = end_time - start_time
            
            execution_result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=e,
                execution_time=execution_time,
                context=context
            )
        
        return execution_result
    
    def execute_code(self, 
                    code: str,
                    context: Optional[ExecutionContext] = None,
                    async_execution: bool = False,
                    callback: Optional[Callable[[ExecutionResult], None]] = None,
                    source: str = "unknown") -> Union[ExecutionResult, str]:
        """
        Führt M-CODE aus.
        
        Args:
            code: M-CODE
            context: Ausführungskontext
            async_execution: Asynchrone Ausführung
            callback: Callback-Funktion für asynchrone Ausführung
            source: Quelle des Codes
            
        Returns:
            Ausführungsergebnis oder Task-ID (bei asynchroner Ausführung)
            
        Raises:
            SecurityViolation: Bei Sicherheitsverletzungen
        """
        # Verifiziere Code
        is_verified, hash_id, violation = self.code_verifier.verify_code(code, source)
        if not is_verified:
            raise violation
        
        # Erstelle Kontext, falls keiner angegeben wurde
        if context is None:
            context = self.create_context()
        
        # Führe Code in Sandbox aus
        try:
            start_time = time.time()
            
            result, locals_dict = self.sandbox.execute_code(code, context.globals_dict, context.locals_dict)
            
            # Aktualisiere lokale Variablen im Kontext
            context.locals_dict.update(locals_dict)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Erstelle Ergebnis
            execution_result = ExecutionResult(
                status=ExecutionStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                context=context
            )
        
        except Exception as e:
            # Fehler bei der Ausführung
            end_time = time.time()
            execution_time = end_time - start_time
            
            execution_result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=e,
                execution_time=execution_time,
                context=context
            )
        
        return execution_result
    
    def shutdown(self) -> None:
        """Fährt die Laufzeitumgebung herunter"""
        # Beende Worker-Threads
        for _ in range(len(self.thread_pool)):
            self.task_queue.put(None)
        
        for thread in self.thread_pool:
            thread.join()
        
        # Leere Queues
        while not self.task_queue.empty():
            self.task_queue.get()
            self.task_queue.task_done()
        
        while not self.result_queue.empty():
            self.result_queue.get()
        
        # Fahre JIT-Compiler herunter
        self.jit_engine.shutdown()
        
        logger.info("M-CODE Runtime heruntergefahren")
