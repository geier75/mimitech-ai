#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Compiler

Dieser Modul implementiert den Compiler für die M-CODE Programmiersprache.
Der Compiler übersetzt M-CODE in optimierten Bytecode für die MISO Runtime.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import ast
import inspect
import types
import bytecode
import numpy as np
import time
import hashlib
import re
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from pathlib import Path
from dataclasses import dataclass, field

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.compiler")

# Import von internen Modulen
from .syntax import MCodeSyntaxTree, parse_m_code, MCodeNode, MCodeTensorOp
from .optimizer import MCodeOptimizer, optimize_m_code
from .mlx_adapter import get_mlx_adapter, MLX_AVAILABLE


class MCodeCompilationError(Exception):
    """Fehler während der Kompilierung von M-CODE"""
    pass


class MCodeByteCode:
    """Repräsentation von M-CODE Bytecode"""
    
    def __init__(self, 
                 instructions: List[Dict[str, Any]], 
                 constants: List[Any], 
                 names: List[str],
                 source: str = "",
                 filename: str = "<m_code>"):
        """
        Initialisiert einen neuen M-CODE Bytecode.
        
        Args:
            instructions: Liste von Bytecode-Instruktionen
            constants: Liste von Konstanten
            names: Liste von Variablennamen
            source: Quellcode
            filename: Dateiname des Quellcodes
        """
        self.instructions = instructions
        self.constants = constants
        self.names = names
        self.source = source
        self.filename = filename
        self.metadata = {
            "version": "2025.1.0",
            "optimization_level": int(os.environ.get("M_CODE_OPTIMIZATION", "2")),
            "timestamp": np.datetime64('now')
        }
    
    def __repr__(self) -> str:
        """String-Repräsentation des Bytecodes"""
        return f"<MCodeByteCode: {len(self.instructions)} instructions, {len(self.constants)} constants>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert den Bytecode in ein Dictionary"""
        return {
            "instructions": self.instructions,
            "constants": self.constants,
            "names": self.names,
            "source": self.source,
            "filename": self.filename,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCodeByteCode':
        """Erstellt einen Bytecode aus einem Dictionary"""
        bytecode = cls(
            instructions=data["instructions"],
            constants=data["constants"],
            names=data["names"],
            source=data.get("source", ""),
            filename=data.get("filename", "<m_code>")
        )
        bytecode.metadata = data.get("metadata", {})
        return bytecode
    
    def serialize(self) -> bytes:
        """Serialisiert den Bytecode für die Speicherung"""
        import pickle
        return pickle.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'MCodeByteCode':
        """Deserialisiert einen Bytecode aus Binärdaten"""
        import pickle
        return cls.from_dict(pickle.loads(data))


class MCodeCompiler:
    """Compiler für die M-CODE Programmiersprache"""
    
    def __init__(self, optimization_level: int = 2):
        """
        Initialisiert einen neuen M-CODE Compiler.
        
        Args:
            optimization_level: Optimierungsstufe (0-3)
        """
        self.optimization_level = min(3, max(0, optimization_level))
        self.optimizer = MCodeOptimizer(self.optimization_level)
        self.syntax_extensions = {}
        self.target_platform = self._detect_platform()
        
        logger.info(f"M-CODE Compiler initialisiert mit Optimierungsstufe {self.optimization_level}")
        logger.info(f"Zielplattform: {self.target_platform}")
    
    def _detect_platform(self) -> Dict[str, Any]:
        """Erkennt die Zielplattform für Optimierungen"""
        import platform
        import torch
        
        platform_info = {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "has_gpu": False,
            "gpu_vendor": "none",
            "supports_mps": False,
            "supports_cuda": False,
            "supports_rocm": False
        }
        
        # GPU-Erkennung
        try:
            # Apple Silicon
            if platform.processor() == 'arm' and platform.system() == 'Darwin':
                if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    platform_info["has_gpu"] = True
                    platform_info["gpu_vendor"] = "apple"
                    platform_info["supports_mps"] = True
            
            # CUDA
            if torch.cuda.is_available():
                platform_info["has_gpu"] = True
                platform_info["gpu_vendor"] = "nvidia"
                platform_info["supports_cuda"] = True
            
            # ROCm
            if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
                platform_info["has_gpu"] = True
                platform_info["gpu_vendor"] = "amd"
                platform_info["supports_rocm"] = True
        except:
            logger.warning("Fehler bei GPU-Erkennung")
        
        return platform_info
    
    def register_syntax_extension(self, name: str, handler: Callable) -> None:
        """
        Registriert eine Syntax-Erweiterung für M-CODE.
        
        Args:
            name: Name der Erweiterung
            handler: Callback-Funktion für die Verarbeitung
        """
        self.syntax_extensions[name] = handler
        logger.info(f"Syntax-Erweiterung '{name}' registriert")
    
    def parse(self, source: str, filename: str = "<m_code>") -> MCodeSyntaxTree:
        """
        Parst M-CODE Quellcode in einen Syntaxbaum.
        
        Args:
            source: M-CODE Quellcode
            filename: Name der Quelldatei
            
        Returns:
            Syntaxbaum
        """
        return parse_m_code(source, filename, extensions=self.syntax_extensions)
    
    def optimize(self, syntax_tree: MCodeSyntaxTree) -> MCodeSyntaxTree:
        """
        Optimiert einen M-CODE Syntaxbaum.
        
        Args:
            syntax_tree: Zu optimierender Syntaxbaum
            
        Returns:
            Optimierter Syntaxbaum
        """
        return self.optimizer.optimize(syntax_tree)
    
    def generate_bytecode(self, syntax_tree: MCodeSyntaxTree) -> MCodeByteCode:
        """
        Generiert Bytecode aus einem M-CODE Syntaxbaum.
        
        Args:
            syntax_tree: M-CODE Syntaxbaum
            
        Returns:
            M-CODE Bytecode
        """
        # Sammle Konstanten und Namen
        constants = []
        names = []
        
        # Generiere Instruktionen
        instructions = self._generate_instructions(syntax_tree, constants, names)
        
        # Erstelle Bytecode
        return MCodeByteCode(
            instructions=instructions,
            constants=constants,
            names=names,
            source=syntax_tree.source,
            filename=syntax_tree.filename
        )
    
    def _generate_instructions(self, 
                              syntax_tree: MCodeSyntaxTree, 
                              constants: List[Any], 
                              names: List[str]) -> List[Dict[str, Any]]:
        """
        Generiert Bytecode-Instruktionen aus einem Syntaxbaum.
        
        Args:
            syntax_tree: M-CODE Syntaxbaum
            constants: Liste für Konstanten
            names: Liste für Namen
            
        Returns:
            Liste von Bytecode-Instruktionen
        """
        # Diese Methode würde den tatsächlichen Bytecode generieren
        # Hier eine vereinfachte Version für das Beispiel
        instructions = []
        
        # Verarbeite Knoten im Syntaxbaum
        for node in syntax_tree.nodes:
            node_type = node.get("type")
            
            if node_type == "literal":
                # Konstante laden
                value = node.get("value")
                if value not in constants:
                    constants.append(value)
                const_idx = constants.index(value)
                instructions.append({
                    "opcode": "LOAD_CONST",
                    "arg": const_idx
                })
            
            elif node_type == "variable":
                # Variable laden
                var_name = node.get("name")
                if var_name not in names:
                    names.append(var_name)
                name_idx = names.index(var_name)
                instructions.append({
                    "opcode": "LOAD_NAME",
                    "arg": name_idx
                })
            
            elif node_type == "assignment":
                # Variable speichern
                var_name = node.get("target")
                if var_name not in names:
                    names.append(var_name)
                name_idx = names.index(var_name)
                
                # Zuerst den Wert berechnen (rekursiv)
                value_node = node.get("value")
                value_instructions = self._generate_instructions_for_node(value_node, constants, names)
                instructions.extend(value_instructions)
                
                # Dann speichern
                instructions.append({
                    "opcode": "STORE_NAME",
                    "arg": name_idx
                })
            
            elif node_type == "binary_op":
                # Binäre Operation
                left_node = node.get("left")
                right_node = node.get("right")
                op = node.get("operator")
                
                # Linken und rechten Operanden berechnen
                left_instructions = self._generate_instructions_for_node(left_node, constants, names)
                right_instructions = self._generate_instructions_for_node(right_node, constants, names)
                
                instructions.extend(left_instructions)
                instructions.extend(right_instructions)
                
                # Operation ausführen
                instructions.append({
                    "opcode": f"BINARY_{op.upper()}",
                    "arg": 0
                })
            
            # Weitere Knotentypen würden hier verarbeitet
        
        return instructions
    
    def _generate_instructions_for_node(self, 
                                       node: Dict[str, Any], 
                                       constants: List[Any], 
                                       names: List[str]) -> List[Dict[str, Any]]:
        """
        Generiert Instruktionen für einen einzelnen Knoten.
        
        Args:
            node: Syntaxbaumknoten
            constants: Liste für Konstanten
            names: Liste für Namen
            
        Returns:
            Liste von Bytecode-Instruktionen
        """
        # Erstelle einen temporären Syntaxbaum mit nur diesem Knoten
        temp_tree = MCodeSyntaxTree([node], "", "<m_code>")
        return self._generate_instructions(temp_tree, constants, names)
    
    def compile(self, source: str, filename: str = "<m_code>") -> MCodeByteCode:
        """
        Kompiliert M-CODE Quellcode zu Bytecode.
        
        Args:
            source: M-CODE Quellcode
            filename: Name der Quelldatei
            
        Returns:
            M-CODE Bytecode
        """
        try:
            # Parse
            syntax_tree = self.parse(source, filename)
            
            # Optimiere
            if self.optimization_level > 0:
                syntax_tree = self.optimize(syntax_tree)
            
            # Generiere Bytecode
            bytecode = self.generate_bytecode(syntax_tree)
            
            return bytecode
        except Exception as e:
            raise MCodeCompilationError(f"Fehler beim Kompilieren von {filename}: {e}")


def compile_m_code(source: str, 
                  filename: str = "<m_code>", 
                  optimization_level: Optional[int] = None) -> MCodeByteCode:
    """
    Kompiliert M-CODE Quellcode zu Bytecode.
    
    Args:
        source: M-CODE Quellcode
        filename: Name der Quelldatei
        optimization_level: Optimierungsstufe (0-3)
        
    Returns:
        M-CODE Bytecode
    """
    if optimization_level is None:
        optimization_level = int(os.environ.get("M_CODE_OPTIMIZATION", "2"))
    
    compiler = MCodeCompiler(optimization_level)
    return compiler.compile(source, filename)
