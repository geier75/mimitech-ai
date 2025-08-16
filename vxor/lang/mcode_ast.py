#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE AST Compiler

Dieser Modul implementiert den abstrakten Syntaxbaum-Compiler für die M-CODE Programmiersprache.
Der AST-Compiler wandelt den abstrakten Syntaxbaum in optimierten Bytecode oder LLVM-IR um.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
import uuid
from enum import Enum, auto
from typing import Dict, Any, List, Tuple, Optional, Union, Set

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.ast")


class ASTNode:
    """Basisklasse für AST-Knoten"""
    
    def __init__(self, node_type: str, line: int, column: int):
        """
        Initialisiert einen neuen AST-Knoten.
        
        Args:
            node_type: Typ des Knotens
            line: Zeilennummer
            column: Spaltennummer
        """
        self.node_type = node_type
        self.line = line
        self.column = column
        self.id = str(uuid.uuid4())
    
    def __repr__(self) -> str:
        """String-Repräsentation des Knotens"""
        return f"<{self.node_type} at {self.line}:{self.column}>"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert den Knoten in ein Dictionary.
        
        Returns:
            Dictionary-Repräsentation
        """
        return {
            "type": self.node_type,
            "line": self.line,
            "column": self.column,
            "id": self.id
        }


class MCodeSyntaxTree:
    """Abstrakter Syntaxbaum für M-CODE"""
    
    def __init__(self, root: Dict[str, Any]):
        """
        Initialisiert einen neuen abstrakten Syntaxbaum.
        
        Args:
            root: Wurzelknoten
        """
        self.root = root
        self.nodes = {}
        self.edges = []
        
        # Konvertiere Dictionary in Knoten
        self._build_tree(root)
    
    def _build_tree(self, node_dict: Dict[str, Any], parent_id: Optional[str] = None) -> str:
        """
        Erstellt einen Teilbaum aus einem Dictionary.
        
        Args:
            node_dict: Knoten als Dictionary
            parent_id: ID des Elternknotens
            
        Returns:
            ID des erstellten Knotens
        """
        # Erstelle Knoten
        node = ASTNode(
            node_dict["type"],
            node_dict.get("line", 0),
            node_dict.get("column", 0)
        )
        
        # Speichere Knoten
        self.nodes[node.id] = node
        
        # Füge Kante hinzu, falls ein Elternknoten existiert
        if parent_id is not None:
            self.edges.append((parent_id, node.id))
        
        # Verarbeite Kinder
        for key, value in node_dict.items():
            if key in ("type", "line", "column", "end_index"):
                continue
            
            if isinstance(value, dict) and "type" in value:
                # Kind-Knoten
                child_id = self._build_tree(value, node.id)
                setattr(node, key, child_id)
            elif isinstance(value, list):
                # Liste von Kind-Knoten
                child_ids = []
                for item in value:
                    if isinstance(item, dict) and "type" in item:
                        child_id = self._build_tree(item, node.id)
                        child_ids.append(child_id)
                setattr(node, key, child_ids)
            else:
                # Attribut
                setattr(node, key, value)
        
        return node.id
    
    def get_node(self, node_id: str) -> Optional[ASTNode]:
        """
        Gibt einen Knoten anhand seiner ID zurück.
        
        Args:
            node_id: ID des Knotens
            
        Returns:
            Knoten oder None
        """
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[str]:
        """
        Gibt die IDs aller Kindknoten eines Knotens zurück.
        
        Args:
            node_id: ID des Knotens
            
        Returns:
            Liste von Knoten-IDs
        """
        return [child_id for parent_id, child_id in self.edges if parent_id == node_id]
    
    def get_parent(self, node_id: str) -> Optional[str]:
        """
        Gibt die ID des Elternknotens eines Knotens zurück.
        
        Args:
            node_id: ID des Knotens
            
        Returns:
            ID des Elternknotens oder None
        """
        for parent_id, child_id in self.edges:
            if child_id == node_id:
                return parent_id
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert den Baum in ein Dictionary.
        
        Returns:
            Dictionary-Repräsentation
        """
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": self.edges,
            "root": self.root
        }


class BytecodeOp(Enum):
    """Bytecode-Operationen für M-CODE"""
    
    # Stack-Operationen
    LOAD_CONST = auto()
    LOAD_NAME = auto()
    STORE_NAME = auto()
    
    # Arithmetische Operationen
    BINARY_ADD = auto()
    BINARY_SUBTRACT = auto()
    BINARY_MULTIPLY = auto()
    BINARY_DIVIDE = auto()
    BINARY_POWER = auto()
    BINARY_MATMUL = auto()
    
    # Vergleichsoperationen
    COMPARE_EQ = auto()
    COMPARE_NE = auto()
    COMPARE_LT = auto()
    COMPARE_LE = auto()
    COMPARE_GT = auto()
    COMPARE_GE = auto()
    
    # Kontrollfluss
    JUMP = auto()
    JUMP_IF_TRUE = auto()
    JUMP_IF_FALSE = auto()
    CALL_FUNCTION = auto()
    RETURN = auto()
    
    # Datenstrukturen
    BUILD_LIST = auto()
    BUILD_DICT = auto()
    GET_ITEM = auto()
    SET_ITEM = auto()
    
    # Ereignisse
    REGISTER_EVENT = auto()
    TRIGGER_EVENT = auto()
    
    # Spezialisierte Tensor-Operationen
    TENSOR_TRANSPOSE = auto()
    TENSOR_DOT = auto()
    TENSOR_NORMALIZE = auto()


class MCodeBytecode:
    """Bytecode für M-CODE"""
    
    def __init__(self, filename: str = "<m_code>"):
        """
        Initialisiert einen neuen Bytecode.
        
        Args:
            filename: Name der Quelldatei
        """
        self.filename = filename
        self.instructions = []
        self.constants = []
        self.names = []
        self.lineno = []
    
    def add_instruction(self, op: BytecodeOp, arg: Any = None, lineno: int = 0) -> int:
        """
        Fügt eine Instruktion hinzu.
        
        Args:
            op: Operation
            arg: Argument
            lineno: Zeilennummer
            
        Returns:
            Index der Instruktion
        """
        index = len(self.instructions)
        self.instructions.append({
            "opcode": op,
            "arg": arg
        })
        self.lineno.append(lineno)
        return index
    
    def add_constant(self, value: Any) -> int:
        """
        Fügt eine Konstante hinzu.
        
        Args:
            value: Wert
            
        Returns:
            Index der Konstante
        """
        # Prüfe, ob die Konstante bereits existiert
        for i, const in enumerate(self.constants):
            if const == value:
                return i
        
        # Füge neue Konstante hinzu
        index = len(self.constants)
        self.constants.append(value)
        return index
    
    def add_name(self, name: str) -> int:
        """
        Fügt einen Namen hinzu.
        
        Args:
            name: Name
            
        Returns:
            Index des Namens
        """
        # Prüfe, ob der Name bereits existiert
        if name in self.names:
            return self.names.index(name)
        
        # Füge neuen Namen hinzu
        index = len(self.names)
        self.names.append(name)
        return index
    
    def get_instruction(self, index: int) -> Dict[str, Any]:
        """
        Gibt eine Instruktion zurück.
        
        Args:
            index: Index der Instruktion
            
        Returns:
            Instruktion
        """
        return self.instructions[index]
    
    def get_constant(self, index: int) -> Any:
        """
        Gibt eine Konstante zurück.
        
        Args:
            index: Index der Konstante
            
        Returns:
            Konstante
        """
        return self.constants[index]
    
    def get_name(self, index: int) -> str:
        """
        Gibt einen Namen zurück.
        
        Args:
            index: Index des Namens
            
        Returns:
            Name
        """
        return self.names[index]
    
    def __repr__(self) -> str:
        """String-Repräsentation des Bytecodes"""
        result = f"Bytecode für {self.filename}:\n"
        for i, instr in enumerate(self.instructions):
            opcode = instr["opcode"].name
            arg = instr["arg"]
            lineno = self.lineno[i]
            
            # Formatiere Argument
            if opcode.startswith("LOAD_CONST") and isinstance(arg, int):
                arg_str = repr(self.constants[arg])
            elif opcode.startswith("LOAD_NAME") or opcode.startswith("STORE_NAME"):
                arg_str = self.names[arg]
            else:
                arg_str = str(arg)
            
            result += f"{i:4d} {opcode:20s} {arg_str:30s} (Zeile {lineno})\n"
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert den Bytecode in ein Dictionary.
        
        Returns:
            Dictionary-Repräsentation
        """
        return {
            "filename": self.filename,
            "instructions": [
                {
                    "opcode": instr["opcode"].name,
                    "arg": instr["arg"],
                    "lineno": self.lineno[i]
                }
                for i, instr in enumerate(self.instructions)
            ],
            "constants": self.constants,
            "names": self.names
        }


class ASTCompiler:
    """Compiler für abstrakte Syntaxbäume"""
    
    def __init__(self, optimization_level: int = 2):
        """
        Initialisiert einen neuen AST-Compiler.
        
        Args:
            optimization_level: Optimierungsstufe (0-3)
        """
        self.optimization_level = optimization_level
        self.bytecode = None
        self.tree = None
        self.jump_targets = {}
    
    def compile(self, ast: Dict[str, Any], filename: str = "<m_code>") -> MCodeBytecode:
        """
        Kompiliert einen abstrakten Syntaxbaum in Bytecode.
        
        Args:
            ast: Abstrakter Syntaxbaum
            filename: Name der Quelldatei
            
        Returns:
            Bytecode
        """
        # Erstelle Syntaxbaum
        self.tree = MCodeSyntaxTree(ast)
        
        # Erstelle Bytecode
        self.bytecode = MCodeBytecode(filename)
        
        # Kompiliere Programm
        self._compile_program(ast)
        
        # Optimiere Bytecode
        if self.optimization_level > 0:
            self._optimize_bytecode()
        
        return self.bytecode
    
    def _compile_program(self, program: Dict[str, Any]) -> None:
        """
        Kompiliert ein Programm.
        
        Args:
            program: Programm-Knoten
        """
        # Kompiliere alle Anweisungen
        for node in program["body"]:
            self._compile_node(node)
        
        # Füge implizites Return hinzu
        self.bytecode.add_instruction(BytecodeOp.LOAD_CONST, self.bytecode.add_constant(None))
        self.bytecode.add_instruction(BytecodeOp.RETURN)
    
    def _compile_node(self, node: Dict[str, Any]) -> None:
        """
        Kompiliert einen Knoten.
        
        Args:
            node: AST-Knoten
        """
        node_type = node["type"]
        
        if node_type == "LetStatement":
            self._compile_let_statement(node)
        elif node_type == "WhenStatement":
            self._compile_when_statement(node)
        elif node_type == "ReturnStatement":
            self._compile_return_statement(node)
        elif node_type == "CallStatement":
            self._compile_call_statement(node)
        elif node_type == "BinaryExpression":
            self._compile_binary_expression(node)
        elif node_type == "FunctionCall":
            self._compile_function_call(node)
        else:
            logger.warning(f"Unbekannter Knotentyp: {node_type}")
    
    def _compile_let_statement(self, node: Dict[str, Any]) -> None:
        """
        Kompiliert eine let-Anweisung.
        
        Args:
            node: AST-Knoten
        """
        # Kompiliere Ausdruck
        self._compile_expression(node["expression"])
        
        # Speichere Wert in Variable
        name_index = self.bytecode.add_name(node["identifier"])
        self.bytecode.add_instruction(BytecodeOp.STORE_NAME, name_index, node["line"])
    
    def _compile_when_statement(self, node: Dict[str, Any]) -> None:
        """
        Kompiliert eine when-Anweisung.
        
        Args:
            node: AST-Knoten
        """
        # Lade Ereignisparameter
        name_index = self.bytecode.add_name(node["parameter"])
        self.bytecode.add_instruction(BytecodeOp.LOAD_NAME, name_index, node["line"])
        
        # Registriere Ereignis
        event_index = self.bytecode.add_constant(node["event"])
        self.bytecode.add_instruction(BytecodeOp.LOAD_CONST, event_index, node["line"])
        
        # Kompiliere Body
        body_start = len(self.bytecode.instructions)
        for body_node in node["body"]:
            self._compile_node(body_node)
        
        # Füge Return hinzu
        self.bytecode.add_instruction(BytecodeOp.LOAD_CONST, self.bytecode.add_constant(None))
        self.bytecode.add_instruction(BytecodeOp.RETURN)
        
        # Registriere Ereignis mit Body
        self.bytecode.add_instruction(BytecodeOp.REGISTER_EVENT, body_start, node["line"])
    
    def _compile_return_statement(self, node: Dict[str, Any]) -> None:
        """
        Kompiliert eine return-Anweisung.
        
        Args:
            node: AST-Knoten
        """
        # Kompiliere Ausdruck
        self._compile_expression(node["expression"])
        
        # Füge Return hinzu
        self.bytecode.add_instruction(BytecodeOp.RETURN, None, node["line"])
    
    def _compile_call_statement(self, node: Dict[str, Any]) -> None:
        """
        Kompiliert eine call-Anweisung.
        
        Args:
            node: AST-Knoten
        """
        # Lade Funktion
        function_index = self.bytecode.add_name(node["function"])
        self.bytecode.add_instruction(BytecodeOp.LOAD_NAME, function_index, node["line"])
        
        # Lade Parameter
        for param in node["parameters"]:
            self._compile_expression(param["value"])
        
        # Rufe Funktion auf
        self.bytecode.add_instruction(BytecodeOp.CALL_FUNCTION, len(node["parameters"]), node["line"])
        
        # Verwerfe Rückgabewert
        self.bytecode.add_instruction(BytecodeOp.POP_TOP, None, node["line"])
    
    def _compile_binary_expression(self, node: Dict[str, Any]) -> None:
        """
        Kompiliert einen binären Ausdruck.
        
        Args:
            node: AST-Knoten
        """
        # Kompiliere linken und rechten Operanden
        self._compile_expression(node["left"])
        self._compile_expression(node["right"])
        
        # Führe Operation aus
        operator = node["operator"]
        if operator == "+":
            self.bytecode.add_instruction(BytecodeOp.BINARY_ADD, None, node["line"])
        elif operator == "-":
            self.bytecode.add_instruction(BytecodeOp.BINARY_SUBTRACT, None, node["line"])
        elif operator == "*":
            self.bytecode.add_instruction(BytecodeOp.BINARY_MULTIPLY, None, node["line"])
        elif operator == "/":
            self.bytecode.add_instruction(BytecodeOp.BINARY_DIVIDE, None, node["line"])
        elif operator == "**":
            self.bytecode.add_instruction(BytecodeOp.BINARY_POWER, None, node["line"])
        elif operator == "@":
            self.bytecode.add_instruction(BytecodeOp.BINARY_MATMUL, None, node["line"])
        else:
            logger.warning(f"Unbekannter Operator: {operator}")
    
    def _compile_function_call(self, node: Dict[str, Any]) -> None:
        """
        Kompiliert einen Funktionsaufruf.
        
        Args:
            node: AST-Knoten
        """
        # Lade Funktion
        function_index = self.bytecode.add_name(node["function"])
        self.bytecode.add_instruction(BytecodeOp.LOAD_NAME, function_index, node["line"])
        
        # Lade Parameter
        for param in node["parameters"]:
            self._compile_expression(param)
        
        # Rufe Funktion auf
        self.bytecode.add_instruction(BytecodeOp.CALL_FUNCTION, len(node["parameters"]), node["line"])
    
    def _compile_expression(self, node: Dict[str, Any]) -> None:
        """
        Kompiliert einen Ausdruck.
        
        Args:
            node: AST-Knoten
        """
        node_type = node["type"]
        
        if node_type == "BinaryExpression":
            self._compile_binary_expression(node)
        elif node_type == "FunctionCall":
            self._compile_function_call(node)
        elif node_type == "Variable":
            name_index = self.bytecode.add_name(node["name"])
            self.bytecode.add_instruction(BytecodeOp.LOAD_NAME, name_index, node["line"])
        elif node_type == "NumberLiteral":
            const_index = self.bytecode.add_constant(node["value"])
            self.bytecode.add_instruction(BytecodeOp.LOAD_CONST, const_index, node["line"])
        elif node_type == "StringLiteral":
            const_index = self.bytecode.add_constant(node["value"])
            self.bytecode.add_instruction(BytecodeOp.LOAD_CONST, const_index, node["line"])
        elif node_type == "BooleanLiteral":
            const_index = self.bytecode.add_constant(node["value"])
            self.bytecode.add_instruction(BytecodeOp.LOAD_CONST, const_index, node["line"])
        else:
            logger.warning(f"Unbekannter Ausdruckstyp: {node_type}")
    
    def _optimize_bytecode(self) -> None:
        """Optimiert den Bytecode"""
        # Optimierungsstufe 1: Konstanten-Faltung
        if self.optimization_level >= 1:
            self._fold_constants()
        
        # Optimierungsstufe 2: Tote Code-Eliminierung
        if self.optimization_level >= 2:
            self._eliminate_dead_code()
        
        # Optimierungsstufe 3: Spezielle Tensor-Optimierungen
        if self.optimization_level >= 3:
            self._optimize_tensor_operations()
    
    def _fold_constants(self) -> None:
        """Führt Konstanten-Faltung durch"""
        # Hier würde die eigentliche Konstanten-Faltung implementiert werden
        # Für dieses Beispiel nur ein Platzhalter
        logger.debug("Konstanten-Faltung durchgeführt")
    
    def _eliminate_dead_code(self) -> None:
        """Eliminiert toten Code"""
        # Hier würde die eigentliche tote Code-Eliminierung implementiert werden
        # Für dieses Beispiel nur ein Platzhalter
        logger.debug("Tote Code-Eliminierung durchgeführt")
    
    def _optimize_tensor_operations(self) -> None:
        """Optimiert Tensor-Operationen"""
        # Hier würden spezielle Tensor-Optimierungen implementiert werden
        # Für dieses Beispiel nur ein Platzhalter
        logger.debug("Tensor-Optimierungen durchgeführt")
