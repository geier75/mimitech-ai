#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Parser

Dieser Modul implementiert den Parser für die M-CODE Programmiersprache.
Der Parser zerlegt M-CODE Quellcode in Token und erstellt einen abstrakten Syntaxbaum.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import re
import logging
from enum import Enum, auto
from typing import Dict, Any, List, Tuple, Optional, Union, Set

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.parser")


class TokenType(Enum):
    """Token-Typen für den M-CODE Parser"""
    
    # Schlüsselwörter
    LET = auto()
    WHEN = auto()
    CHANGE = auto()
    CALL = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    FUNCTION = auto()
    
    # Datentypen
    TENSOR = auto()
    MATRIX = auto()
    VECTOR = auto()
    SCALAR = auto()
    STRING = auto()
    BOOLEAN = auto()
    
    # Literale
    NUMBER = auto()
    STRING_LITERAL = auto()
    BOOLEAN_LITERAL = auto()
    
    # Bezeichner
    IDENTIFIER = auto()
    
    # Operatoren
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    POWER = auto()
    MATMUL = auto()  # @
    ASSIGN = auto()  # =
    
    # Vergleichsoperatoren
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    
    # Satzzeichen
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    COMMA = auto()
    COLON = auto()
    DOT = auto()
    
    # Sonstiges
    COMMENT = auto()
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()
    UNKNOWN = auto()


class MCodeToken:
    """Token für die M-CODE Programmiersprache"""
    
    def __init__(self, type: TokenType, value: str, line: int, column: int):
        """
        Initialisiert ein neues Token.
        
        Args:
            type: Token-Typ
            value: Token-Wert
            line: Zeilennummer
            column: Spaltennummer
        """
        self.type = type
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self) -> str:
        """String-Repräsentation des Tokens"""
        return f"<Token {self.type.name} '{self.value}' at {self.line}:{self.column}>"


class MCodeParser:
    """Parser für die M-CODE Programmiersprache"""
    
    def __init__(self):
        """Initialisiert einen neuen Parser"""
        # Token-Definitionen
        self.keywords = {
            "let": TokenType.LET,
            "when": TokenType.WHEN,
            "change": TokenType.CHANGE,
            "call": TokenType.CALL,
            "return": TokenType.RETURN,
            "if": TokenType.IF,
            "else": TokenType.ELSE,
            "for": TokenType.FOR,
            "while": TokenType.WHILE,
            "function": TokenType.FUNCTION,
            "tensor": TokenType.TENSOR,
            "matrix": TokenType.MATRIX,
            "vector": TokenType.VECTOR,
            "scalar": TokenType.SCALAR,
            "string": TokenType.STRING,
            "boolean": TokenType.BOOLEAN,
            "true": TokenType.BOOLEAN_LITERAL,
            "false": TokenType.BOOLEAN_LITERAL
        }
        
        # Operator-Definitionen
        self.operators = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.MULTIPLY,
            "/": TokenType.DIVIDE,
            "**": TokenType.POWER,
            "@": TokenType.MATMUL,
            "=": TokenType.ASSIGN,
            "==": TokenType.EQUAL,
            "!=": TokenType.NOT_EQUAL,
            "<": TokenType.LESS,
            "<=": TokenType.LESS_EQUAL,
            ">": TokenType.GREATER,
            ">=": TokenType.GREATER_EQUAL,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            ",": TokenType.COMMA,
            ":": TokenType.COLON,
            ".": TokenType.DOT
        }
        
        # Reguläre Ausdrücke für Token
        self.patterns = [
            (r'[ \t]+', None),  # Leerzeichen und Tabs ignorieren
            (r'#.*', TokenType.COMMENT),  # Kommentare
            (r'\n', TokenType.NEWLINE),  # Zeilenumbrüche
            (r'[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?', TokenType.NUMBER),  # Zahlen
            (r'"([^"\\]|\\.)*"', TokenType.STRING_LITERAL),  # Strings in Anführungszeichen
            (r'\'([^\'\\]|\\.)*\'', TokenType.STRING_LITERAL),  # Strings in einfachen Anführungszeichen
            (r'[a-zA-Z_][a-zA-Z0-9_]*', self._identify_keyword_or_identifier),  # Bezeichner oder Schlüsselwörter
        ]
        
        # Füge Operatoren zu den Patterns hinzu
        for op, token_type in sorted(self.operators.items(), key=lambda x: len(x[0]), reverse=True):
            pattern = re.escape(op)
            self.patterns.append((pattern, token_type))
    
    def _identify_keyword_or_identifier(self, value: str) -> TokenType:
        """
        Identifiziert, ob ein Wert ein Schlüsselwort oder ein Bezeichner ist.
        
        Args:
            value: Token-Wert
            
        Returns:
            Token-Typ
        """
        return self.keywords.get(value.lower(), TokenType.IDENTIFIER)
    
    def tokenize(self, source: str) -> List[MCodeToken]:
        """
        Zerlegt M-CODE Quellcode in Token.
        
        Args:
            source: M-CODE Quellcode
            
        Returns:
            Liste von Token
        """
        tokens = []
        line_num = 1
        column_num = 1
        indent_stack = [0]
        current_indent = 0
        
        # Füge Zeilenumbruch am Ende hinzu, falls nicht vorhanden
        if not source.endswith('\n'):
            source += '\n'
        
        i = 0
        while i < len(source):
            # Prüfe auf Einrückung am Anfang der Zeile
            if column_num == 1:
                # Zähle Leerzeichen und Tabs
                j = i
                while j < len(source) and (source[j] == ' ' or source[j] == '\t'):
                    j += 1
                
                # Berechne Einrückungstiefe
                spaces = j - i
                
                # Vergleiche mit aktueller Einrückung
                if spaces > indent_stack[-1]:
                    # Neue Einrückung
                    indent_stack.append(spaces)
                    tokens.append(MCodeToken(TokenType.INDENT, '', line_num, column_num))
                elif spaces < indent_stack[-1]:
                    # Ausrückung
                    while spaces < indent_stack[-1]:
                        indent_stack.pop()
                        tokens.append(MCodeToken(TokenType.DEDENT, '', line_num, column_num))
                    
                    # Prüfe, ob die Einrückung korrekt ist
                    if spaces != indent_stack[-1]:
                        raise SyntaxError(f"Ungültige Einrückung in Zeile {line_num}")
                
                # Aktualisiere Position
                i = j
                column_num = spaces + 1
                continue
            
            # Suche nach passendem Pattern
            match = None
            for pattern, token_type in self.patterns:
                regex = re.compile(pattern)
                match = regex.match(source, i)
                if match:
                    value = match.group(0)
                    
                    # Ignoriere Whitespace
                    if token_type is None:
                        i = match.end()
                        column_num += len(value)
                        continue
                    
                    # Bestimme Token-Typ für Bezeichner
                    if callable(token_type):
                        token_type = token_type(value)
                    
                    # Erstelle Token
                    token = MCodeToken(token_type, value, line_num, column_num)
                    tokens.append(token)
                    
                    # Aktualisiere Position
                    i = match.end()
                    
                    # Aktualisiere Zeilen- und Spaltennummer
                    if token_type == TokenType.NEWLINE:
                        line_num += 1
                        column_num = 1
                    else:
                        column_num += len(value)
                    
                    break
            
            # Kein passendes Pattern gefunden
            if not match:
                # Unbekanntes Zeichen
                token = MCodeToken(TokenType.UNKNOWN, source[i], line_num, column_num)
                tokens.append(token)
                i += 1
                column_num += 1
        
        # Füge DEDENT-Token für alle verbleibenden Einrückungen hinzu
        while len(indent_stack) > 1:
            indent_stack.pop()
            tokens.append(MCodeToken(TokenType.DEDENT, '', line_num, column_num))
        
        # Füge EOF-Token hinzu
        tokens.append(MCodeToken(TokenType.EOF, '', line_num, column_num))
        
        return tokens
    
    def parse(self, tokens: List[MCodeToken]) -> Dict[str, Any]:
        """
        Erstellt einen abstrakten Syntaxbaum aus Token.
        
        Args:
            tokens: Liste von Token
            
        Returns:
            Abstrakter Syntaxbaum
        """
        # Hier würde der eigentliche Parser implementiert werden
        # Für dieses Beispiel erstellen wir einen einfachen AST
        
        ast = {
            "type": "Program",
            "body": []
        }
        
        # Entferne Kommentare und Whitespace
        filtered_tokens = [token for token in tokens if token.type not in (TokenType.COMMENT, TokenType.NEWLINE)]
        
        # Einfacher Parser für Demonstrationszwecke
        i = 0
        while i < len(filtered_tokens) and filtered_tokens[i].type != TokenType.EOF:
            token = filtered_tokens[i]
            
            # Verarbeite Anweisungen
            if token.type == TokenType.LET:
                # let-Anweisung
                node = self._parse_let_statement(filtered_tokens, i)
                ast["body"].append(node)
                i = node["end_index"]
            elif token.type == TokenType.WHEN:
                # when-Anweisung
                node = self._parse_when_statement(filtered_tokens, i)
                ast["body"].append(node)
                i = node["end_index"]
            elif token.type == TokenType.RETURN:
                # return-Anweisung
                node = self._parse_return_statement(filtered_tokens, i)
                ast["body"].append(node)
                i = node["end_index"]
            elif token.type == TokenType.CALL:
                # call-Anweisung
                node = self._parse_call_statement(filtered_tokens, i)
                ast["body"].append(node)
                i = node["end_index"]
            else:
                # Unbekannte Anweisung
                i += 1
        
        return ast
    
    def _parse_let_statement(self, tokens: List[MCodeToken], start_index: int) -> Dict[str, Any]:
        """
        Verarbeitet eine let-Anweisung.
        
        Args:
            tokens: Liste von Token
            start_index: Startindex
            
        Returns:
            AST-Knoten
        """
        # let tensor A = randn(4,4)
        i = start_index
        
        # Überspringe "let"
        i += 1
        
        # Typ
        type_token = tokens[i]
        i += 1
        
        # Bezeichner
        identifier_token = tokens[i]
        i += 1
        
        # Gleichheitszeichen
        if tokens[i].type != TokenType.ASSIGN:
            raise SyntaxError(f"Erwartet '=', gefunden '{tokens[i].value}' in Zeile {tokens[i].line}")
        i += 1
        
        # Ausdruck
        expression, end_index = self._parse_expression(tokens, i)
        
        # Erstelle AST-Knoten
        node = {
            "type": "LetStatement",
            "variable_type": type_token.value,
            "identifier": identifier_token.value,
            "expression": expression,
            "line": tokens[start_index].line,
            "column": tokens[start_index].column,
            "end_index": end_index
        }
        
        return node
    
    def _parse_when_statement(self, tokens: List[MCodeToken], start_index: int) -> Dict[str, Any]:
        """
        Verarbeitet eine when-Anweisung.
        
        Args:
            tokens: Liste von Token
            start_index: Startindex
            
        Returns:
            AST-Knoten
        """
        # when change(X):
        #    call prism.predict(mode="short", input=X)
        i = start_index
        
        # Überspringe "when"
        i += 1
        
        # Ereignis
        event_token = tokens[i]
        i += 1
        
        # Öffnende Klammer
        if tokens[i].type != TokenType.LPAREN:
            raise SyntaxError(f"Erwartet '(', gefunden '{tokens[i].value}' in Zeile {tokens[i].line}")
        i += 1
        
        # Parameter
        parameter_token = tokens[i]
        i += 1
        
        # Schließende Klammer
        if tokens[i].type != TokenType.RPAREN:
            raise SyntaxError(f"Erwartet ')', gefunden '{tokens[i].value}' in Zeile {tokens[i].line}")
        i += 1
        
        # Doppelpunkt
        if tokens[i].type != TokenType.COLON:
            raise SyntaxError(f"Erwartet ':', gefunden '{tokens[i].value}' in Zeile {tokens[i].line}")
        i += 1
        
        # Einrückung
        if tokens[i].type != TokenType.INDENT:
            raise SyntaxError(f"Erwartet Einrückung nach ':' in Zeile {tokens[i].line}")
        i += 1
        
        # Body
        body = []
        while tokens[i].type != TokenType.DEDENT and tokens[i].type != TokenType.EOF:
            if tokens[i].type == TokenType.CALL:
                # call-Anweisung
                node = self._parse_call_statement(tokens, i)
                body.append(node)
                i = node["end_index"]
            else:
                # Unbekannte Anweisung
                i += 1
        
        # Ausrückung
        if tokens[i].type == TokenType.DEDENT:
            i += 1
        
        # Erstelle AST-Knoten
        node = {
            "type": "WhenStatement",
            "event": event_token.value,
            "parameter": parameter_token.value,
            "body": body,
            "line": tokens[start_index].line,
            "column": tokens[start_index].column,
            "end_index": i
        }
        
        return node
    
    def _parse_return_statement(self, tokens: List[MCodeToken], start_index: int) -> Dict[str, Any]:
        """
        Verarbeitet eine return-Anweisung.
        
        Args:
            tokens: Liste von Token
            start_index: Startindex
            
        Returns:
            AST-Knoten
        """
        # return normalize(A @ B)
        i = start_index
        
        # Überspringe "return"
        i += 1
        
        # Ausdruck
        expression, end_index = self._parse_expression(tokens, i)
        
        # Erstelle AST-Knoten
        node = {
            "type": "ReturnStatement",
            "expression": expression,
            "line": tokens[start_index].line,
            "column": tokens[start_index].column,
            "end_index": end_index
        }
        
        return node
    
    def _parse_call_statement(self, tokens: List[MCodeToken], start_index: int) -> Dict[str, Any]:
        """
        Verarbeitet eine call-Anweisung.
        
        Args:
            tokens: Liste von Token
            start_index: Startindex
            
        Returns:
            AST-Knoten
        """
        # call prism.predict(mode="short", input=X)
        i = start_index
        
        # Überspringe "call"
        i += 1
        
        # Funktionsname
        function_name = ""
        while i < len(tokens) and (tokens[i].type == TokenType.IDENTIFIER or tokens[i].type == TokenType.DOT):
            function_name += tokens[i].value
            i += 1
        
        # Öffnende Klammer
        if tokens[i].type != TokenType.LPAREN:
            raise SyntaxError(f"Erwartet '(', gefunden '{tokens[i].value}' in Zeile {tokens[i].line}")
        i += 1
        
        # Parameter
        parameters = []
        while i < len(tokens) and tokens[i].type != TokenType.RPAREN:
            # Parameter-Name
            if tokens[i].type == TokenType.IDENTIFIER:
                param_name = tokens[i].value
                i += 1
                
                # Gleichheitszeichen
                if tokens[i].type == TokenType.ASSIGN:
                    i += 1
                    
                    # Parameter-Wert
                    param_value, end_index = self._parse_expression(tokens, i)
                    i = end_index
                    
                    # Füge Parameter hinzu
                    parameters.append({
                        "name": param_name,
                        "value": param_value
                    })
                else:
                    # Positionaler Parameter
                    param_value, end_index = self._parse_expression(tokens, i - 1)
                    i = end_index
                    
                    # Füge Parameter hinzu
                    parameters.append({
                        "name": None,
                        "value": param_value
                    })
            else:
                # Positionaler Parameter
                param_value, end_index = self._parse_expression(tokens, i)
                i = end_index
                
                # Füge Parameter hinzu
                parameters.append({
                    "name": None,
                    "value": param_value
                })
            
            # Komma
            if i < len(tokens) and tokens[i].type == TokenType.COMMA:
                i += 1
        
        # Schließende Klammer
        if i >= len(tokens) or tokens[i].type != TokenType.RPAREN:
            raise SyntaxError(f"Erwartet ')', gefunden '{tokens[i].value if i < len(tokens) else 'EOF'}' in Zeile {tokens[i].line if i < len(tokens) else tokens[-1].line}")
        i += 1
        
        # Erstelle AST-Knoten
        node = {
            "type": "CallStatement",
            "function": function_name,
            "parameters": parameters,
            "line": tokens[start_index].line,
            "column": tokens[start_index].column,
            "end_index": i
        }
        
        return node
    
    def _parse_expression(self, tokens: List[MCodeToken], start_index: int) -> Tuple[Dict[str, Any], int]:
        """
        Verarbeitet einen Ausdruck.
        
        Args:
            tokens: Liste von Token
            start_index: Startindex
            
        Returns:
            AST-Knoten und Endindex
        """
        # Einfacher Ausdruck-Parser für Demonstrationszwecke
        # In einer vollständigen Implementierung würde hier ein rekursiver Abstieg oder ein Operator-Precedence-Parser verwendet werden
        
        i = start_index
        
        # Literale
        if tokens[i].type == TokenType.NUMBER:
            node = {
                "type": "NumberLiteral",
                "value": float(tokens[i].value),
                "line": tokens[i].line,
                "column": tokens[i].column
            }
            i += 1
        elif tokens[i].type == TokenType.STRING_LITERAL:
            # Entferne Anführungszeichen
            value = tokens[i].value[1:-1]
            node = {
                "type": "StringLiteral",
                "value": value,
                "line": tokens[i].line,
                "column": tokens[i].column
            }
            i += 1
        elif tokens[i].type == TokenType.BOOLEAN_LITERAL:
            node = {
                "type": "BooleanLiteral",
                "value": tokens[i].value.lower() == "true",
                "line": tokens[i].line,
                "column": tokens[i].column
            }
            i += 1
        elif tokens[i].type == TokenType.IDENTIFIER:
            # Funktionsaufruf oder Variable
            if i + 1 < len(tokens) and tokens[i + 1].type == TokenType.LPAREN:
                # Funktionsaufruf
                function_name = tokens[i].value
                i += 2  # Überspringe Bezeichner und öffnende Klammer
                
                # Parameter
                parameters = []
                while i < len(tokens) and tokens[i].type != TokenType.RPAREN:
                    param, end_index = self._parse_expression(tokens, i)
                    parameters.append(param)
                    i = end_index
                    
                    # Komma
                    if i < len(tokens) and tokens[i].type == TokenType.COMMA:
                        i += 1
                
                # Schließende Klammer
                if i >= len(tokens) or tokens[i].type != TokenType.RPAREN:
                    raise SyntaxError(f"Erwartet ')', gefunden '{tokens[i].value if i < len(tokens) else 'EOF'}' in Zeile {tokens[i].line if i < len(tokens) else tokens[-1].line}")
                i += 1
                
                node = {
                    "type": "FunctionCall",
                    "function": function_name,
                    "parameters": parameters,
                    "line": tokens[start_index].line,
                    "column": tokens[start_index].column
                }
            else:
                # Variable
                node = {
                    "type": "Variable",
                    "name": tokens[i].value,
                    "line": tokens[i].line,
                    "column": tokens[i].column
                }
                i += 1
        elif tokens[i].type == TokenType.LPAREN:
            # Geklammerte Ausdrücke
            i += 1
            node, end_index = self._parse_expression(tokens, i)
            i = end_index
            
            # Schließende Klammer
            if i >= len(tokens) or tokens[i].type != TokenType.RPAREN:
                raise SyntaxError(f"Erwartet ')', gefunden '{tokens[i].value if i < len(tokens) else 'EOF'}' in Zeile {tokens[i].line if i < len(tokens) else tokens[-1].line}")
            i += 1
        else:
            # Unbekannter Ausdruck
            node = {
                "type": "Unknown",
                "value": tokens[i].value,
                "line": tokens[i].line,
                "column": tokens[i].column
            }
            i += 1
        
        # Binäre Operatoren
        if i < len(tokens) and tokens[i].type in (TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.POWER, TokenType.MATMUL):
            operator = tokens[i].value
            i += 1
            
            right, end_index = self._parse_expression(tokens, i)
            
            node = {
                "type": "BinaryExpression",
                "operator": operator,
                "left": node,
                "right": right,
                "line": tokens[start_index].line,
                "column": tokens[start_index].column
            }
            
            i = end_index
        
        return node, i
