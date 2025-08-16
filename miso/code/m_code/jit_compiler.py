#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE JIT-Compiler

Dieses Modul implementiert den Just-In-Time-Compiler für M-CODE mit Fokus auf 
Tensor-Operationen und MLX-Optimierungen für Apple Silicon.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import inspect
import time
import hashlib
import ast
import re
import tempfile
import importlib.util
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.jit_compiler")

# Import von internen Modulen
from .mlx_adapter import get_mlx_adapter, MLX_AVAILABLE
from .syntax import MCodeNode, MCodeTensorOp, MCodeModule, MCodeFunction


@dataclass
class JITCompilationOptions:
    """Optionen für die JIT-Kompilierung"""
    optimization_level: int = 3
    use_ane: bool = True
    trace_enabled: bool = True
    cache_enabled: bool = True
    fusion_enabled: bool = True
    vectorize: bool = True
    unroll_loops: bool = True
    max_unroll_factor: int = 8
    inline_threshold: int = 100
    debug_symbols: bool = False


@dataclass
class JITCompilationStats:
    """Statistiken zur JIT-Kompilierung"""
    compilation_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    optimizations_applied: List[str] = field(default_factory=list)
    tensor_ops_count: int = 0
    jitted_ops_count: int = 0


class JITCompilationError(Exception):
    """Fehler während der JIT-Kompilierung"""
    pass


class JITCache:
    """Cache für JIT-kompilierte Funktionen"""
    
    def __init__(self, max_size: int = 1024):
        """
        Initialisiert einen neuen JIT-Cache.
        
        Args:
            max_size: Maximale Anzahl der Cache-Einträge
        """
        self.max_size = max_size
        self.cache = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        self.last_access = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Gibt einen Cache-Eintrag zurück, wenn er existiert.
        
        Args:
            key: Cache-Schlüssel
            
        Returns:
            Cache-Eintrag oder None
        """
        if key in self.cache:
            self.stats["hits"] += 1
            self.last_access[key] = time.time()
            return self.cache[key]
        
        self.stats["misses"] += 1
        return None
    
    def put(self, key: str, value: Any) -> None:
        """
        Fügt einen Eintrag zum Cache hinzu.
        
        Args:
            key: Cache-Schlüssel
            value: Cache-Wert
        """
        # Prüfe Cache-Größe und eviktiere ggf. Einträge
        if len(self.cache) >= self.max_size:
            # Least Recently Used (LRU) Strategie
            lru_key = min(self.last_access.items(), key=lambda x: x[1])[0]
            self.cache.pop(lru_key)
            self.last_access.pop(lru_key)
            self.stats["evictions"] += 1
        
        self.cache[key] = value
        self.last_access[key] = time.time()
    
    def clear(self) -> None:
        """Leert den Cache"""
        self.cache.clear()
        self.last_access.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gibt Statistiken zum Cache zurück.
        
        Returns:
            Cache-Statistiken
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]) if (self.stats["hits"] + self.stats["misses"]) > 0 else 0,
            **self.stats
        }


class MCodeJITCompiler:
    """Just-In-Time-Compiler für M-CODE"""
    
    def __init__(self, options: Optional[JITCompilationOptions] = None):
        """
        Initialisiert einen neuen JIT-Compiler.
        
        Args:
            options: Kompilierungsoptionen
        """
        self.options = options or JITCompilationOptions()
        self.cache = JITCache()
        self.mlx_adapter = get_mlx_adapter(use_ane=self.options.use_ane)
        self.stats = JITCompilationStats()
        
        # Temporäres Verzeichnis für kompilierte Module
        self.temp_dir = Path(tempfile.gettempdir()) / "mcode_jit"
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"M-CODE JIT-Compiler initialisiert mit Optionen: {self.options}")
    
    def _generate_cache_key(self, func: Callable, args_signature: Tuple) -> str:
        """
        Generiert einen Cache-Schlüssel für eine Funktion.
        
        Args:
            func: Zu kompilierende Funktion
            args_signature: Typ-Signatur der Argumente
            
        Returns:
            Cache-Schlüssel
        """
        # Funktion als String und Hash der Funktion
        func_str = inspect.getsource(func)
        func_hash = hashlib.md5(func_str.encode()).hexdigest()
        
        # Argumente-Signatur als String
        args_str = str(args_signature)
        
        # Kombiniere zu Schlüssel
        return f"{func.__name__}_{func_hash}_{hashlib.md5(args_str.encode()).hexdigest()}"
    
    def _identify_tensor_operations(self, node: MCodeNode) -> List[MCodeTensorOp]:
        """
        Identifiziert Tensor-Operationen in einem AST-Knoten.
        
        Args:
            node: AST-Knoten
            
        Returns:
            Liste von Tensor-Operationen
        """
        tensor_ops = []
        
        # Direkter Check auf MCodeTensorOp
        if isinstance(node, MCodeTensorOp):
            tensor_ops.append(node)
            return tensor_ops
        
        # Rekursive Suche nach Tensor-Operationen in Kindern
        for attr_name in dir(node):
            # Überspringe spezielle Attribute und Methoden
            if attr_name.startswith('_') or callable(getattr(node, attr_name)):
                continue
                
            attr = getattr(node, attr_name)
            
            # Rekursion für MCodeNode-Objekte
            if isinstance(attr, MCodeNode):
                tensor_ops.extend(self._identify_tensor_operations(attr))
            
            # Rekursion für Listen von MCodeNode-Objekten
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, MCodeNode):
                        tensor_ops.extend(self._identify_tensor_operations(item))
            
            # Rekursion für Dictionaries mit MCodeNode-Werten
            elif isinstance(attr, dict):
                for value in attr.values():
                    if isinstance(value, MCodeNode):
                        tensor_ops.extend(self._identify_tensor_operations(value))
                        
        return tensor_ops
    
    def _generate_optimized_code(self, tensor_ops: List[MCodeTensorOp]) -> str:
        """
        Generiert optimierten Code für Tensor-Operationen.
        
        Args:
            tensor_ops: Liste von Tensor-Operationen
            
        Returns:
            Optimierter Code
        """
        # Lokale Importe zur Vermeidung von Zirkelbezügen
        from .tensor import jit, tensor
        
        # Funktion für die JIT-Kompilierung erstellen
        function_lines = []
        function_lines.append("import mlx.core as mx")
        function_lines.append("from miso.code.m_code.tensor import jit, tensor")
        function_lines.append("")
        function_lines.append("@jit")
        function_lines.append("def optimized_tensor_op(tensors):")
        
        # Extrahiere Tensoren als Parameter
        for i, op in enumerate(tensor_ops):
            function_lines.append(f"    # Operation: {op.operation}")
            
            # Transformiere Operation in MLX-Code
            mlx_code = self._transform_to_mlx_code(op)
            for line in mlx_code.split("\n"):
                function_lines.append(f"    {line}")
        
        function_lines.append("    return result")
        
        return "\n".join(function_lines)
    
    def _transform_to_mlx_code(self, tensor_op: MCodeTensorOp) -> str:
        """
        Transformiert eine Tensor-Operation in MLX-Code.
        
        Args:
            tensor_op: Tensor-Operation
            
        Returns:
            MLX-Code
        """
        operation = tensor_op.operation
        operands = tensor_op.operands
        
        # Abbildung von Operationen auf MLX-Funktionen
        operation_mapping = {
            "add": "mx.add",
            "subtract": "mx.subtract",
            "multiply": "mx.multiply",
            "divide": "mx.divide",
            "matmul": "mx.matmul",
            "transpose": "mx.transpose",
            "sum": "mx.sum",
            "mean": "mx.mean",
            "max": "mx.max",
            "min": "mx.min",
            "exp": "mx.exp",
            "log": "mx.log",
            "pow": "mx.power",
            "sqrt": "mx.sqrt",
            "sin": "mx.sin",
            "cos": "mx.cos",
            "tan": "mx.tan",
            "tanh": "mx.tanh",
            "sigmoid": "mx.sigmoid",
            "relu": "mx.relu",
            "softmax": "mx.softmax"
        }
        
        # Prüfe, ob die Operation unterstützt wird
        if operation not in operation_mapping:
            return f"# Operation {operation} nicht unterstützt\n"
            
        # Operanden-Code generieren
        operands_code = []
        for i, operand in enumerate(operands):
            operands_code.append(f"tensors[{i}]")
        
        # MLX-Funktionsaufruf generieren
        mlx_function = operation_mapping[operation]
        return f"result = {mlx_function}({', '.join(operands_code)})"
    
    def compile_function(self, func: Callable, optimize: bool = True) -> Callable:
        """
        Kompiliert eine Funktion mit JIT für optimierte Ausführung.
        
        Args:
            func: Zu kompilierende Funktion
            optimize: Optimierung aktivieren
            
        Returns:
            Kompilierte Funktion
        """
        if not MLX_AVAILABLE:
            logger.warning("MLX nicht verfügbar, JIT-Kompilierung wird übersprungen")
            return func
            
        if not self.options.cache_enabled:
            # Direkte Kompilierung ohne Cache
            return self.mlx_adapter.jit_compile(func)
        
        # Funktionssignatur für den Cache-Schlüssel
        func_signature = inspect.signature(func)
        func_key = self._generate_cache_key(func, func_signature)
        
        # Cache-Lookup
        cached_func = self.cache.get(func_key)
        if cached_func is not None:
            self.stats.cache_hits += 1
            return cached_func
        
        self.stats.cache_misses += 1
        start_time = time.time()
        
        try:
            # JIT-Kompilierung der Funktion
            compiled_func = self.mlx_adapter.jit_compile(func)
            
            # Statistik aktualisieren
            self.stats.compilation_time_ms = (time.time() - start_time) * 1000
            self.stats.jitted_ops_count += 1
            
            # In Cache speichern
            self.cache.put(func_key, compiled_func)
            
            return compiled_func
        except Exception as e:
            logger.error(f"Fehler bei der JIT-Kompilierung: {e}")
            # Fallback zur Originalfunktion
            return func
    
    def compile_module(self, module: MCodeModule) -> MCodeModule:
        """
        Kompiliert ein Modul mit JIT für optimierte Ausführung.
        
        Args:
            module: Zu kompilierendes Modul
            
        Returns:
            Kompiliertes Modul
        """
        for func_name, func in module.functions.items():
            if isinstance(func, MCodeFunction):
                # Identifiziere Tensor-Operationen in der Funktion
                tensor_ops = self._identify_tensor_operations(func)
                self.stats.tensor_ops_count += len(tensor_ops)
                
                if tensor_ops:
                    logger.info(f"Tensor-Operationen in Funktion {func_name} identifiziert: {len(tensor_ops)}")
                    # TODO: Implementiere Funktionskompilierung
        
        return module
    
    def get_stats(self) -> JITCompilationStats:
        """
        Gibt Statistiken zur JIT-Kompilierung zurück.
        
        Returns:
            JIT-Kompilierungsstatistiken
        """
        # Aktualisiere Cache-Statistiken
        cache_stats = self.cache.get_stats()
        self.stats.cache_hits = cache_stats["hits"]
        self.stats.cache_misses = cache_stats["misses"]
        
        return self.stats


# Singleton-Instanz des JIT-Compilers
_jit_compiler_instance = None

def get_jit_compiler(options: Optional[JITCompilationOptions] = None) -> MCodeJITCompiler:
    """
    Gibt eine Singleton-Instanz des JIT-Compilers zurück.
    
    Args:
        options: Kompilierungsoptionen
        
    Returns:
        JIT-Compiler
    """
    global _jit_compiler_instance
    
    if _jit_compiler_instance is None:
        _jit_compiler_instance = MCodeJITCompiler(options)
        
    return _jit_compiler_instance


@dataclass
class JITModuleInfo:
    """Informationen zu einem JIT-kompilierten Modul"""
    module_path: str
    function_count: int
    tensor_ops_count: int
    compilation_time_ms: float
    is_mlx_accelerated: bool
    compilation_date: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class JITModuleRegistry:
    """Registry für JIT-kompilierte Module"""
    
    def __init__(self):
        """Initialisiert eine neue JIT-Modul-Registry"""
        self.modules = {}
    
    def register_module(self, module_info: JITModuleInfo) -> None:
        """
        Registriert ein JIT-kompiliertes Modul.
        
        Args:
            module_info: Modul-Informationen
        """
        self.modules[module_info.module_path] = module_info
    
    def get_module_info(self, module_path: str) -> Optional[JITModuleInfo]:
        """
        Gibt Informationen zu einem JIT-kompilierten Modul zurück.
        
        Args:
            module_path: Modulpfad
            
        Returns:
            Modul-Informationen oder None
        """
        return self.modules.get(module_path)
    
    def list_modules(self) -> List[JITModuleInfo]:
        """
        Gibt eine Liste aller JIT-kompilierten Module zurück.
        
        Returns:
            Liste von Modul-Informationen
        """
        return list(self.modules.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gibt Statistiken zur JIT-Kompilierung zurück.
        
        Returns:
            JIT-Kompilierungsstatistiken
        """
        total_functions = sum(module.function_count for module in self.modules.values())
        total_tensor_ops = sum(module.tensor_ops_count for module in self.modules.values())
        total_compilation_time = sum(module.compilation_time_ms for module in self.modules.values())
        
        return {
            "module_count": len(self.modules),
            "function_count": total_functions,
            "tensor_ops_count": total_tensor_ops,
            "avg_compilation_time_ms": total_compilation_time / len(self.modules) if self.modules else 0,
            "mlx_accelerated_count": sum(1 for module in self.modules.values() if module.is_mlx_accelerated)
        }


# Singleton-Instanz der JIT-Modul-Registry
_jit_module_registry = JITModuleRegistry()

def get_jit_module_registry() -> JITModuleRegistry:
    """
    Gibt eine Singleton-Instanz der JIT-Modul-Registry zurück.
    
    Returns:
        JIT-Modul-Registry
    """
    global _jit_module_registry
    return _jit_module_registry


# Dekorator für JIT-kompilierte Funktionen
def jit(func: Callable) -> Callable:
    """
    Dekoriert eine Funktion für JIT-Kompilierung.
    
    Args:
        func: Zu dekorierende Funktion
        
    Returns:
        Dekorierte Funktion
    """
    compiler = get_jit_compiler()
    return compiler.compile_function(func)
