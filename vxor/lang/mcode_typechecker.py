#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Type Checker

Dieser Modul implementiert den Typprüfer für die M-CODE Programmiersprache.
Der Typprüfer führt statische und dynamische Typanalysen durch und stellt sicher,
dass der Code den Sicherheitsanforderungen entspricht.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
from enum import Enum, auto
from typing import Dict, Any, List, Tuple, Optional, Union, Set

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.typechecker")


class TypeCheckError(Exception):
    """Fehler bei der Typprüfung"""
    pass


class MCodeType(Enum):
    """Datentypen für M-CODE"""
    
    # Primitive Typen
    SCALAR = auto()
    BOOLEAN = auto()
    STRING = auto()
    
    # Tensor-Typen
    TENSOR = auto()
    MATRIX = auto()
    VECTOR = auto()
    
    # Komplexe Typen
    FUNCTION = auto()
    MODULE = auto()
    ANY = auto()
    UNKNOWN = auto()


class TypeChecker:
    """Typprüfer für die M-CODE Programmiersprache"""
    
    def __init__(self, security_level: int = 3):
        """
        Initialisiert einen neuen Typprüfer.
        
        Args:
            security_level: Sicherheitsstufe (0-3)
        """
        self.security_level = security_level
        self.symbol_table = {}
        self.errors = []
        
        # Standardfunktionen und ihre Typen
        self.builtin_types = {
            "randn": {
                "type": MCodeType.FUNCTION,
                "return_type": MCodeType.TENSOR,
                "parameters": [
                    {"name": "rows", "type": MCodeType.SCALAR, "optional": False},
                    {"name": "cols", "type": MCodeType.SCALAR, "optional": True}
                ]
            },
            "eye": {
                "type": MCodeType.FUNCTION,
                "return_type": MCodeType.MATRIX,
                "parameters": [
                    {"name": "size", "type": MCodeType.SCALAR, "optional": False}
                ]
            },
            "normalize": {
                "type": MCodeType.FUNCTION,
                "return_type": MCodeType.TENSOR,
                "parameters": [
                    {"name": "tensor", "type": MCodeType.TENSOR, "optional": False}
                ]
            },
            "prism.predict": {
                "type": MCodeType.FUNCTION,
                "return_type": MCodeType.TENSOR,
                "parameters": [
                    {"name": "mode", "type": MCodeType.STRING, "optional": False},
                    {"name": "input", "type": MCodeType.ANY, "optional": False}
                ]
            }
        }
    
    def check(self, ast: Dict[str, Any]) -> None:
        """
        Führt eine Typprüfung für einen abstrakten Syntaxbaum durch.
        
        Args:
            ast: Abstrakter Syntaxbaum
            
        Raises:
            TypeCheckError: Bei Typfehlern
        """
        # Initialisiere Symbol-Tabelle
        self.symbol_table = {}
        self.errors = []
        
        # Füge Standardfunktionen hinzu
        for name, type_info in self.builtin_types.items():
            self.symbol_table[name] = type_info
        
        # Prüfe alle Anweisungen
        for node in ast["body"]:
            self._check_node(node)
        
        # Prüfe auf Fehler
        if self.errors:
            error_message = "\n".join(self.errors)
            raise TypeCheckError(f"Typfehler in M-CODE:\n{error_message}")
    
    def _check_node(self, node: Dict[str, Any]) -> Optional[MCodeType]:
        """
        Prüft einen AST-Knoten.
        
        Args:
            node: AST-Knoten
            
        Returns:
            Typ des Knotens oder None
        """
        node_type = node.get("type")
        
        if node_type == "LetStatement":
            return self._check_let_statement(node)
        elif node_type == "WhenStatement":
            return self._check_when_statement(node)
        elif node_type == "ReturnStatement":
            return self._check_return_statement(node)
        elif node_type == "CallStatement":
            return self._check_call_statement(node)
        elif node_type == "BinaryExpression":
            return self._check_binary_expression(node)
        elif node_type == "FunctionCall":
            return self._check_function_call(node)
        elif node_type == "Variable":
            return self._check_variable(node)
        elif node_type == "NumberLiteral":
            return MCodeType.SCALAR
        elif node_type == "StringLiteral":
            return MCodeType.STRING
        elif node_type == "BooleanLiteral":
            return MCodeType.BOOLEAN
        else:
            self.errors.append(f"Unbekannter Knotentyp: {node_type} in Zeile {node.get('line', '?')}")
            return MCodeType.UNKNOWN
    
    def _check_let_statement(self, node: Dict[str, Any]) -> None:
        """
        Prüft eine let-Anweisung.
        
        Args:
            node: AST-Knoten
        """
        # Bestimme Variablentyp
        variable_type_str = node["variable_type"]
        variable_type = self._string_to_type(variable_type_str)
        
        # Prüfe Ausdruck
        expression_type = self._check_node(node["expression"])
        
        # Prüfe Typkompatibilität
        if not self._is_compatible(expression_type, variable_type):
            self.errors.append(
                f"Typfehler in Zeile {node['line']}: "
                f"Kann {expression_type.name} nicht {variable_type.name} zuweisen"
            )
        
        # Füge Variable zur Symbol-Tabelle hinzu
        self.symbol_table[node["identifier"]] = {
            "type": variable_type,
            "initialized": True
        }
    
    def _check_when_statement(self, node: Dict[str, Any]) -> None:
        """
        Prüft eine when-Anweisung.
        
        Args:
            node: AST-Knoten
        """
        # Prüfe, ob das Ereignis gültig ist
        event = node["event"]
        if event != "change":
            self.errors.append(
                f"Fehler in Zeile {node['line']}: "
                f"Unbekanntes Ereignis '{event}'"
            )
        
        # Prüfe, ob der Parameter existiert
        parameter = node["parameter"]
        if parameter not in self.symbol_table:
            self.errors.append(
                f"Fehler in Zeile {node['line']}: "
                f"Unbekannte Variable '{parameter}'"
            )
        
        # Prüfe alle Anweisungen im Body
        for body_node in node["body"]:
            self._check_node(body_node)
    
    def _check_return_statement(self, node: Dict[str, Any]) -> MCodeType:
        """
        Prüft eine return-Anweisung.
        
        Args:
            node: AST-Knoten
            
        Returns:
            Typ des Rückgabewerts
        """
        # Prüfe Ausdruck
        return_type = self._check_node(node["expression"])
        return return_type
    
    def _check_call_statement(self, node: Dict[str, Any]) -> MCodeType:
        """
        Prüft eine call-Anweisung.
        
        Args:
            node: AST-Knoten
            
        Returns:
            Typ des Rückgabewerts
        """
        # Prüfe, ob die Funktion existiert
        function_name = node["function"]
        if function_name not in self.symbol_table:
            self.errors.append(
                f"Fehler in Zeile {node['line']}: "
                f"Unbekannte Funktion '{function_name}'"
            )
            return MCodeType.UNKNOWN
        
        # Hole Funktionstyp
        function_type = self.symbol_table[function_name]
        if function_type["type"] != MCodeType.FUNCTION:
            self.errors.append(
                f"Fehler in Zeile {node['line']}: "
                f"'{function_name}' ist keine Funktion"
            )
            return MCodeType.UNKNOWN
        
        # Prüfe Parameter
        parameters = node["parameters"]
        expected_parameters = function_type["parameters"]
        
        # Prüfe Anzahl der Parameter
        required_params = [p for p in expected_parameters if not p["optional"]]
        if len(parameters) < len(required_params):
            self.errors.append(
                f"Fehler in Zeile {node['line']}: "
                f"Funktion '{function_name}' erwartet mindestens {len(required_params)} Parameter, "
                f"aber {len(parameters)} wurden angegeben"
            )
        
        # Prüfe Typen der Parameter
        for i, param in enumerate(parameters):
            # Prüfe, ob der Parameter existiert
            if i >= len(expected_parameters):
                self.errors.append(
                    f"Fehler in Zeile {node['line']}: "
                    f"Zu viele Parameter für Funktion '{function_name}'"
                )
                break
            
            # Hole erwarteten Typ
            expected_type = expected_parameters[i]["type"]
            
            # Prüfe Ausdruck
            param_type = self._check_node(param["value"])
            
            # Prüfe Typkompatibilität
            if not self._is_compatible(param_type, expected_type):
                self.errors.append(
                    f"Fehler in Zeile {node['line']}: "
                    f"Parameter {i+1} von Funktion '{function_name}' erwartet {expected_type.name}, "
                    f"aber {param_type.name} wurde angegeben"
                )
        
        # Gib Rückgabetyp zurück
        return function_type["return_type"]
    
    def _check_binary_expression(self, node: Dict[str, Any]) -> MCodeType:
        """
        Prüft einen binären Ausdruck.
        
        Args:
            node: AST-Knoten
            
        Returns:
            Typ des Ausdrucks
        """
        # Prüfe linken und rechten Operanden
        left_type = self._check_node(node["left"])
        right_type = self._check_node(node["right"])
        
        # Bestimme Operator
        operator = node["operator"]
        
        # Matrixmultiplikation
        if operator == "@":
            # Prüfe, ob beide Operanden Tensoren sind
            if left_type not in (MCodeType.TENSOR, MCodeType.MATRIX, MCodeType.VECTOR):
                self.errors.append(
                    f"Fehler in Zeile {node['line']}: "
                    f"Linker Operand von '@' muss ein Tensor sein, aber ist {left_type.name}"
                )
            
            if right_type not in (MCodeType.TENSOR, MCodeType.MATRIX, MCodeType.VECTOR):
                self.errors.append(
                    f"Fehler in Zeile {node['line']}: "
                    f"Rechter Operand von '@' muss ein Tensor sein, aber ist {right_type.name}"
                )
            
            # Matrixmultiplikation gibt einen Tensor zurück
            return MCodeType.TENSOR
        
        # Arithmetische Operationen
        if operator in ("+", "-", "*", "/"):
            # Prüfe, ob beide Operanden numerisch sind
            if left_type not in (MCodeType.SCALAR, MCodeType.TENSOR, MCodeType.MATRIX, MCodeType.VECTOR):
                self.errors.append(
                    f"Fehler in Zeile {node['line']}: "
                    f"Linker Operand von '{operator}' muss numerisch sein, aber ist {left_type.name}"
                )
            
            if right_type not in (MCodeType.SCALAR, MCodeType.TENSOR, MCodeType.MATRIX, MCodeType.VECTOR):
                self.errors.append(
                    f"Fehler in Zeile {node['line']}: "
                    f"Rechter Operand von '{operator}' muss numerisch sein, aber ist {right_type.name}"
                )
            
            # Wenn einer der Operanden ein Tensor ist, ist das Ergebnis ein Tensor
            if left_type in (MCodeType.TENSOR, MCodeType.MATRIX, MCodeType.VECTOR) or right_type in (MCodeType.TENSOR, MCodeType.MATRIX, MCodeType.VECTOR):
                return MCodeType.TENSOR
            else:
                return MCodeType.SCALAR
        
        # Potenzierung
        if operator == "**":
            # Prüfe, ob beide Operanden numerisch sind
            if left_type not in (MCodeType.SCALAR, MCodeType.TENSOR, MCodeType.MATRIX, MCodeType.VECTOR):
                self.errors.append(
                    f"Fehler in Zeile {node['line']}: "
                    f"Linker Operand von '**' muss numerisch sein, aber ist {left_type.name}"
                )
            
            if right_type != MCodeType.SCALAR:
                self.errors.append(
                    f"Fehler in Zeile {node['line']}: "
                    f"Rechter Operand von '**' muss ein Skalar sein, aber ist {right_type.name}"
                )
            
            # Potenzierung gibt den Typ des linken Operanden zurück
            return left_type
        
        # Unbekannter Operator
        self.errors.append(
            f"Fehler in Zeile {node['line']}: "
            f"Unbekannter Operator '{operator}'"
        )
        return MCodeType.UNKNOWN
    
    def _check_function_call(self, node: Dict[str, Any]) -> MCodeType:
        """
        Prüft einen Funktionsaufruf.
        
        Args:
            node: AST-Knoten
            
        Returns:
            Typ des Rückgabewerts
        """
        # Prüfe, ob die Funktion existiert
        function_name = node["function"]
        if function_name not in self.symbol_table:
            self.errors.append(
                f"Fehler in Zeile {node['line']}: "
                f"Unbekannte Funktion '{function_name}'"
            )
            return MCodeType.UNKNOWN
        
        # Hole Funktionstyp
        function_type = self.symbol_table[function_name]
        if function_type["type"] != MCodeType.FUNCTION:
            self.errors.append(
                f"Fehler in Zeile {node['line']}: "
                f"'{function_name}' ist keine Funktion"
            )
            return MCodeType.UNKNOWN
        
        # Prüfe Parameter
        parameters = node["parameters"]
        expected_parameters = function_type["parameters"]
        
        # Prüfe Anzahl der Parameter
        required_params = [p for p in expected_parameters if not p["optional"]]
        if len(parameters) < len(required_params):
            self.errors.append(
                f"Fehler in Zeile {node['line']}: "
                f"Funktion '{function_name}' erwartet mindestens {len(required_params)} Parameter, "
                f"aber {len(parameters)} wurden angegeben"
            )
        
        # Prüfe Typen der Parameter
        for i, param in enumerate(parameters):
            # Prüfe, ob der Parameter existiert
            if i >= len(expected_parameters):
                self.errors.append(
                    f"Fehler in Zeile {node['line']}: "
                    f"Zu viele Parameter für Funktion '{function_name}'"
                )
                break
            
            # Hole erwarteten Typ
            expected_type = expected_parameters[i]["type"]
            
            # Prüfe Ausdruck
            param_type = self._check_node(param)
            
            # Prüfe Typkompatibilität
            if not self._is_compatible(param_type, expected_type):
                self.errors.append(
                    f"Fehler in Zeile {node['line']}: "
                    f"Parameter {i+1} von Funktion '{function_name}' erwartet {expected_type.name}, "
                    f"aber {param_type.name} wurde angegeben"
                )
        
        # Gib Rückgabetyp zurück
        return function_type["return_type"]
    
    def _check_variable(self, node: Dict[str, Any]) -> MCodeType:
        """
        Prüft eine Variable.
        
        Args:
            node: AST-Knoten
            
        Returns:
            Typ der Variable
        """
        # Prüfe, ob die Variable existiert
        variable_name = node["name"]
        if variable_name not in self.symbol_table:
            self.errors.append(
                f"Fehler in Zeile {node['line']}: "
                f"Unbekannte Variable '{variable_name}'"
            )
            return MCodeType.UNKNOWN
        
        # Gib Variablentyp zurück
        return self.symbol_table[variable_name]["type"]
    
    def _string_to_type(self, type_str: str) -> MCodeType:
        """
        Konvertiert einen Typ-String in einen MCodeType.
        
        Args:
            type_str: Typ-String
            
        Returns:
            MCodeType
        """
        type_map = {
            "tensor": MCodeType.TENSOR,
            "matrix": MCodeType.MATRIX,
            "vector": MCodeType.VECTOR,
            "scalar": MCodeType.SCALAR,
            "string": MCodeType.STRING,
            "boolean": MCodeType.BOOLEAN
        }
        
        return type_map.get(type_str.lower(), MCodeType.UNKNOWN)
    
    def _is_compatible(self, source_type: MCodeType, target_type: MCodeType) -> bool:
        """
        Prüft, ob ein Quelltyp mit einem Zieltyp kompatibel ist.
        
        Args:
            source_type: Quelltyp
            target_type: Zieltyp
            
        Returns:
            True, wenn die Typen kompatibel sind
        """
        # ANY ist mit allem kompatibel
        if target_type == MCodeType.ANY:
            return True
        
        # UNKNOWN ist mit nichts kompatibel
        if source_type == MCodeType.UNKNOWN or target_type == MCodeType.UNKNOWN:
            return False
        
        # Gleiche Typen sind kompatibel
        if source_type == target_type:
            return True
        
        # Tensor-Hierarchie
        if target_type == MCodeType.TENSOR:
            return source_type in (MCodeType.MATRIX, MCodeType.VECTOR)
        
        # Matrix-Hierarchie
        if target_type == MCodeType.MATRIX:
            return source_type == MCodeType.VECTOR
        
        # Alles andere ist nicht kompatibel
        return False
    
    def get_errors(self) -> List[str]:
        """
        Gibt alle Fehler zurück.
        
        Returns:
            Liste von Fehlermeldungen
        """
        return self.errors
