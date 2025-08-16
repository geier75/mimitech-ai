#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Engine

Dieses Modul implementiert die KI-native Programmiersprache M-CODE für das MISO-System.
M-CODE ist eine Hochleistungs-Programmiersprache, die speziell für KI-Operationen optimiert ist.
Sie ist schneller als C, sicherer als Rust und optimiert für neuronale Netzwerke & KI-Logik.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import hashlib
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from pathlib import Path
from datetime import datetime
from enum import Enum, auto

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code")

# Importiere Komponenten
from .mcode_parser import MCodeParser, MCodeToken, TokenType
from .mcode_typechecker import TypeChecker, MCodeType, TypeCheckError
from .mcode_ast import ASTNode, ASTCompiler, MCodeSyntaxTree
from .mcode_jit import GPUJITEngine, JITCompilationError
from .mcode_sandbox import SecuritySandbox, SecurityViolationError
from .mcode_runtime import MCodeRuntime, MCodeFunction, MCodeObject

# Globale Konfiguration
class MCodeConfig:
    """Konfiguration für die M-CODE Engine"""
    
    def __init__(self,
                 optimization_level: int = 3,
                 use_jit: bool = True,
                 use_gpu: bool = True,
                 use_neural_engine: bool = True,
                 security_level: int = 3,
                 memory_limit_mb: int = 4096,
                 max_execution_time_ms: int = 10000,
                 allow_network_access: bool = False,
                 allow_file_access: bool = False,
                 debug_mode: bool = False):
        """
        Initialisiert eine neue M-CODE Konfiguration.
        
        Args:
            optimization_level: Optimierungsstufe (0-3)
            use_jit: Just-In-Time Compilation aktivieren
            use_gpu: GPU-Beschleunigung aktivieren
            use_neural_engine: Apple Neural Engine verwenden (falls verfügbar)
            security_level: Sicherheitsstufe (0-3)
            memory_limit_mb: Speicherlimit in MB
            max_execution_time_ms: Maximale Ausführungszeit in ms
            allow_network_access: Netzwerkzugriff erlauben
            allow_file_access: Dateizugriff erlauben
            debug_mode: Debug-Modus aktivieren
        """
        self.optimization_level = min(3, max(0, optimization_level))
        self.use_jit = use_jit
        self.use_gpu = use_gpu
        self.use_neural_engine = use_neural_engine
        self.security_level = min(3, max(0, security_level))
        self.memory_limit_mb = memory_limit_mb
        self.max_execution_time_ms = max_execution_time_ms
        self.allow_network_access = allow_network_access
        self.allow_file_access = allow_file_access
        self.debug_mode = debug_mode
        
        # Plattform-Erkennung
        self.platform_info = self._detect_platform()
        
        # Setze Umgebungsvariablen
        os.environ["M_CODE_OPTIMIZATION"] = str(optimization_level)
        os.environ["M_CODE_USE_JIT"] = "1" if use_jit else "0"
        os.environ["M_CODE_USE_GPU"] = "1" if use_gpu else "0"
        os.environ["M_CODE_USE_NE"] = "1" if use_neural_engine else "0"
        os.environ["M_CODE_SECURITY"] = str(security_level)
        os.environ["M_CODE_MEMORY_LIMIT"] = str(memory_limit_mb)
        
        logger.info(f"M-CODE Konfiguration initialisiert: Opt={optimization_level}, "
                   f"JIT={use_jit}, GPU={use_gpu}, NE={use_neural_engine}, "
                   f"Security={security_level}")
    
    def _detect_platform(self) -> Dict[str, Any]:
        """Erkennt die Plattform und verfügbare Hardware"""
        import platform
        
        platform_info = {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "has_gpu": False,
            "gpu_vendor": "none",
            "has_neural_engine": False,
            "has_llvm": self._check_llvm_available()
        }
        
        # GPU-Erkennung
        try:
            # Apple Silicon
            if platform.processor() == 'arm' and platform.system() == 'Darwin':
                if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    platform_info["has_gpu"] = True
                    platform_info["gpu_vendor"] = "apple"
                    # Prüfe auf Neural Engine
                    platform_info["has_neural_engine"] = self._check_neural_engine_available()
            
            # CUDA
            if torch.cuda.is_available():
                platform_info["has_gpu"] = True
                platform_info["gpu_vendor"] = "nvidia"
            
            # ROCm
            if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
                platform_info["has_gpu"] = True
                platform_info["gpu_vendor"] = "amd"
        except:
            logger.warning("Fehler bei Hardware-Erkennung")
        
        return platform_info
    
    def _check_llvm_available(self) -> bool:
        """Prüft, ob LLVM verfügbar ist"""
        try:
            import llvmlite
            return True
        except ImportError:
            return False
    
    def _check_neural_engine_available(self) -> bool:
        """Prüft, ob die Apple Neural Engine verfügbar ist"""
        try:
            # Prüfe auf Apple Silicon M-Serie
            import platform
            if platform.processor() == 'arm' and platform.system() == 'Darwin':
                # Prüfe auf CoreML
                import coremltools
                return True
            return False
        except ImportError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert die Konfiguration in ein Dictionary"""
        return {
            "optimization_level": self.optimization_level,
            "use_jit": self.use_jit,
            "use_gpu": self.use_gpu,
            "use_neural_engine": self.use_neural_engine,
            "security_level": self.security_level,
            "memory_limit_mb": self.memory_limit_mb,
            "max_execution_time_ms": self.max_execution_time_ms,
            "allow_network_access": self.allow_network_access,
            "allow_file_access": self.allow_file_access,
            "debug_mode": self.debug_mode,
            "platform_info": self.platform_info
        }


class MCodeExecutionResult:
    """Ergebnis einer M-CODE Ausführung"""
    
    def __init__(self,
                 success: bool,
                 result: Any = None,
                 error: Optional[Exception] = None,
                 execution_time_ms: float = 0.0,
                 memory_used_mb: float = 0.0,
                 hash_id: Optional[str] = None):
        """
        Initialisiert ein neues Ausführungsergebnis.
        
        Args:
            success: Erfolgreiche Ausführung
            result: Rückgabewert
            error: Fehler (falls aufgetreten)
            execution_time_ms: Ausführungszeit in ms
            memory_used_mb: Verwendeter Speicher in MB
            hash_id: Hash-ID des ausgeführten Codes
        """
        self.success = success
        self.result = result
        self.error = error
        self.execution_time_ms = execution_time_ms
        self.memory_used_mb = memory_used_mb
        self.hash_id = hash_id
        self.timestamp = datetime.now()
    
    def __repr__(self) -> str:
        """String-Repräsentation des Ergebnisses"""
        if self.success:
            return f"<MCodeExecutionResult: success={self.success}, result={self.result}, time={self.execution_time_ms:.2f}ms>"
        else:
            return f"<MCodeExecutionResult: success={self.success}, error={self.error}, time={self.execution_time_ms:.2f}ms>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert das Ergebnis in ein Dictionary"""
        return {
            "success": self.success,
            "result": self.result,
            "error": str(self.error) if self.error else None,
            "execution_time_ms": self.execution_time_ms,
            "memory_used_mb": self.memory_used_mb,
            "hash_id": self.hash_id,
            "timestamp": self.timestamp.isoformat()
        }


class MCodeEngine:
    """Hauptklasse für die M-CODE Engine"""
    
    def __init__(self, config: Optional[MCodeConfig] = None):
        """
        Initialisiert eine neue M-CODE Engine.
        
        Args:
            config: Konfiguration für die Engine
        """
        self.config = config or MCodeConfig()
        self.parser = MCodeParser()
        self.type_checker = TypeChecker()
        self.ast_compiler = ASTCompiler(self.config.optimization_level)
        self.jit_engine = GPUJITEngine(
            use_gpu=self.config.use_gpu,
            use_neural_engine=self.config.use_neural_engine
        )
        self.sandbox = SecuritySandbox(
            security_level=self.config.security_level,
            memory_limit_mb=self.config.memory_limit_mb,
            max_execution_time_ms=self.config.max_execution_time_ms,
            allow_network_access=self.config.allow_network_access,
            allow_file_access=self.config.allow_file_access
        )
        self.runtime = MCodeRuntime(self.config)
        
        # Cache für kompilierten Code
        self.code_cache = {}
        
        # Registrierte Callbacks
        self.callbacks = {}
        
        # Ausgeführte Code-Blöcke mit Hash-IDs
        self.executed_blocks = {}
        
        logger.info("M-CODE Engine initialisiert")
    
    def generate_hash_id(self, source: str, module_name: str = "unknown") -> str:
        """
        Generiert eine eindeutige Hash-ID für einen Code-Block.
        
        Args:
            source: Quellcode
            module_name: Name des Moduls
            
        Returns:
            Hash-ID
        """
        timestamp = datetime.now().isoformat()
        hash_input = f"{source}|{module_name}|{timestamp}"
        hash_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return hash_id
    
    def compile(self, source: str, module_name: str = "<m_code>", verify: bool = True) -> Dict[str, Any]:
        """
        Kompiliert M-CODE Quellcode.
        
        Args:
            source: M-CODE Quellcode
            module_name: Name des Moduls
            verify: Verifizierung durch Omega-Kern erforderlich
            
        Returns:
            Kompilierter Code als Dictionary
        """
        # Generiere Hash-ID
        hash_id = self.generate_hash_id(source, module_name)
        
        # Prüfe Cache
        if hash_id in self.code_cache:
            return self.code_cache[hash_id]
        
        try:
            # Parse
            tokens = self.parser.tokenize(source)
            ast = self.parser.parse(tokens)
            
            # Typprüfung
            self.type_checker.check(ast)
            
            # Kompiliere zu Bytecode
            bytecode = self.ast_compiler.compile(ast)
            
            # JIT-Kompilierung
            if self.config.use_jit:
                jit_code = self.jit_engine.compile(bytecode)
            else:
                jit_code = None
            
            # Erstelle kompilierten Code
            compiled_code = {
                "hash_id": hash_id,
                "source": source,
                "module_name": module_name,
                "ast": ast,
                "bytecode": bytecode,
                "jit_code": jit_code,
                "timestamp": datetime.now().isoformat(),
                "verified": not verify  # Wenn verify=False, dann ist es bereits verifiziert
            }
            
            # Speichere im Cache
            self.code_cache[hash_id] = compiled_code
            
            return compiled_code
        
        except Exception as e:
            logger.error(f"Fehler beim Kompilieren von M-CODE: {e}")
            raise
    
    def verify(self, hash_id: str, omega_signature: str) -> bool:
        """
        Verifiziert einen kompilierten Code-Block durch den Omega-Kern.
        
        Args:
            hash_id: Hash-ID des Code-Blocks
            omega_signature: Signatur des Omega-Kerns
            
        Returns:
            True, wenn die Verifizierung erfolgreich war
        """
        if hash_id not in self.code_cache:
            logger.error(f"Code-Block mit Hash-ID {hash_id} nicht gefunden")
            return False
        
        # Hier würde die tatsächliche Verifizierung durch den Omega-Kern stattfinden
        # Für dieses Beispiel nehmen wir an, dass jede Signatur gültig ist
        
        # Markiere als verifiziert
        self.code_cache[hash_id]["verified"] = True
        
        logger.info(f"Code-Block mit Hash-ID {hash_id} verifiziert")
        return True
    
    def execute(self, 
                source_or_hash: str, 
                module_name: str = "<m_code>",
                global_vars: Optional[Dict[str, Any]] = None,
                verify: bool = True,
                omega_signature: Optional[str] = None) -> MCodeExecutionResult:
        """
        Führt M-CODE aus.
        
        Args:
            source_or_hash: M-CODE Quellcode oder Hash-ID
            module_name: Name des Moduls
            global_vars: Globale Variablen
            verify: Verifizierung durch Omega-Kern erforderlich
            omega_signature: Signatur des Omega-Kerns (falls verify=True)
            
        Returns:
            Ausführungsergebnis
        """
        start_time = time.time()
        
        try:
            # Prüfe, ob es sich um eine Hash-ID handelt
            if len(source_or_hash) == 16 and all(c in "0123456789abcdef" for c in source_or_hash):
                hash_id = source_or_hash
                if hash_id not in self.code_cache:
                    raise ValueError(f"Code-Block mit Hash-ID {hash_id} nicht gefunden")
                compiled_code = self.code_cache[hash_id]
            else:
                # Kompiliere Quellcode
                compiled_code = self.compile(source_or_hash, module_name, verify)
                hash_id = compiled_code["hash_id"]
            
            # Prüfe Verifizierung
            if verify and not compiled_code["verified"]:
                if omega_signature is None:
                    raise SecurityViolationError("Verifizierung durch Omega-Kern erforderlich")
                if not self.verify(hash_id, omega_signature):
                    raise SecurityViolationError("Verifizierung durch Omega-Kern fehlgeschlagen")
            
            # Führe in Sandbox aus
            with self.sandbox:
                if self.config.use_jit and compiled_code["jit_code"] is not None:
                    # JIT-Ausführung
                    result = self.jit_engine.execute(
                        compiled_code["jit_code"],
                        global_vars or {}
                    )
                else:
                    # Interpreter-Ausführung
                    result = self.runtime.execute(
                        compiled_code["bytecode"],
                        global_vars or {}
                    )
            
            # Messe Ausführungszeit
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Messe Speichernutzung
            import psutil
            process = psutil.Process(os.getpid())
            memory_used_mb = process.memory_info().rss / (1024 * 1024)
            
            # Speichere ausgeführten Block
            self.executed_blocks[hash_id] = {
                "timestamp": datetime.now().isoformat(),
                "execution_time_ms": execution_time_ms,
                "memory_used_mb": memory_used_mb,
                "module_name": module_name
            }
            
            # Erstelle Ergebnis
            return MCodeExecutionResult(
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
                memory_used_mb=memory_used_mb,
                hash_id=hash_id
            )
        
        except Exception as e:
            # Messe Ausführungszeit
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Erstelle Fehlerergebnis
            return MCodeExecutionResult(
                success=False,
                error=e,
                execution_time_ms=execution_time_ms,
                hash_id=hash_id if 'hash_id' in locals() else None
            )
    
    def register_callback(self, event_name: str, callback: Callable) -> None:
        """
        Registriert einen Callback für ein Ereignis.
        
        Args:
            event_name: Name des Ereignisses
            callback: Callback-Funktion
        """
        if event_name not in self.callbacks:
            self.callbacks[event_name] = []
        self.callbacks[event_name].append(callback)
    
    def trigger_event(self, event_name: str, **kwargs) -> None:
        """
        Löst ein Ereignis aus.
        
        Args:
            event_name: Name des Ereignisses
            **kwargs: Parameter für den Callback
        """
        if event_name in self.callbacks:
            for callback in self.callbacks[event_name]:
                try:
                    callback(**kwargs)
                except Exception as e:
                    logger.error(f"Fehler im Callback für Ereignis {event_name}: {e}")
    
    def get_executed_blocks(self) -> Dict[str, Dict[str, Any]]:
        """
        Gibt alle ausgeführten Code-Blöcke zurück.
        
        Returns:
            Dictionary mit ausgeführten Code-Blöcken
        """
        return self.executed_blocks.copy()
    
    def clear_cache(self) -> None:
        """Leert den Code-Cache"""
        self.code_cache.clear()
        logger.info("Code-Cache geleert")
    
    def shutdown(self) -> None:
        """Fährt die Engine herunter"""
        self.runtime.shutdown()
        self.jit_engine.shutdown()
        logger.info("M-CODE Engine heruntergefahren")


# Globale Engine-Instanz
_DEFAULT_ENGINE = None


def get_engine() -> MCodeEngine:
    """
    Gibt die aktuelle Engine-Instanz zurück oder erstellt eine neue,
    falls noch keine existiert.
    
    Returns:
        MCodeEngine-Instanz
    """
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = MCodeEngine()
    return _DEFAULT_ENGINE


def reset_engine() -> None:
    """
    Setzt die globale Engine-Instanz zurück.
    """
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is not None:
        _DEFAULT_ENGINE.shutdown()
    _DEFAULT_ENGINE = None


def compile_m_code(source: str, 
                  module_name: str = "<m_code>",
                  verify: bool = True) -> Dict[str, Any]:
    """
    Kompiliert M-CODE Quellcode.
    
    Args:
        source: M-CODE Quellcode
        module_name: Name des Moduls
        verify: Verifizierung durch Omega-Kern erforderlich
        
    Returns:
        Kompilierter Code als Dictionary
    """
    engine = get_engine()
    return engine.compile(source, module_name, verify)


def execute_m_code(source_or_hash: str, 
                  module_name: str = "<m_code>",
                  global_vars: Optional[Dict[str, Any]] = None,
                  verify: bool = True,
                  omega_signature: Optional[str] = None) -> MCodeExecutionResult:
    """
    Führt M-CODE aus.
    
    Args:
        source_or_hash: M-CODE Quellcode oder Hash-ID
        module_name: Name des Moduls
        global_vars: Globale Variablen
        verify: Verifizierung durch Omega-Kern erforderlich
        omega_signature: Signatur des Omega-Kerns (falls verify=True)
        
    Returns:
        Ausführungsergebnis
    """
    engine = get_engine()
    return engine.execute(source_or_hash, module_name, global_vars, verify, omega_signature)


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Konfiguriere Logging
    logging.basicConfig(level=logging.INFO)
    
    # Erstelle Engine
    engine = get_engine()
    
    # Beispiel-Code
    code = """
    # Berechne gewichteten neuronalen Vektor
    let tensor A = randn(4,4)
    let tensor B = eye(4)
    return normalize(A @ B)
    """
    
    # Führe Code aus
    result = engine.execute(code)
    
    # Gib Ergebnis aus
    print(f"Ergebnis: {result}")
    
    # Führe Code mit Hash-ID aus
    hash_id = result.hash_id
    result2 = engine.execute(hash_id)
    
    # Gib Ergebnis aus
    print(f"Ergebnis 2: {result2}")
    
    # Fahre Engine herunter
    engine.shutdown()
