#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Syntax Parser

Dieses Modul implementiert den Syntax-Parser für die M-CODE Programmiersprache.
Der Parser konvertiert M-CODE Quellcode in einen abstrakten Syntaxbaum.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import ast
import re
import tokenize
import io
from enum import Enum, auto
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.syntax")


class MCodeTokenType(Enum):
    """Token-Typen für M-CODE Lexer"""
    KEYWORD = auto()
    IDENTIFIER = auto()
    LITERAL = auto()
    OPERATOR = auto()
    PUNCTUATION = auto()
    COMMENT = auto()
    WHITESPACE = auto()
    EOF = auto()
    UNKNOWN = auto()
    
    # M-CODE spezifische Token-Typen
    TENSOR_LITERAL = auto()
    QUANTUM_OPERATOR = auto()
    PARALLEL_MARKER = auto()
    PROBABILITY_MARKER = auto()


@dataclass
class MCodeToken:
    """Repräsentation eines M-CODE Tokens"""
    type: MCodeTokenType
    value: str
    line: int
    column: int
    
    def __repr__(self) -> str:
        return f"Token({self.type.name}, '{self.value}', {self.line}:{self.column})"


@dataclass
class MCodeNode:
    """Basisklasse für alle Knoten im M-CODE Syntaxbaum"""
    line: int
    column: int
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.line}:{self.column})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert den Knoten in ein Dictionary"""
        return {
            "type": self.__class__.__name__,
            "line": self.line,
            "column": self.column
        }


@dataclass
class MCodeModule(MCodeNode):
    """Repräsentiert ein M-CODE Modul (oberste Ebene)"""
    body: List[MCodeNode] = field(default_factory=list)
    filename: str = "<m_code>"
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["body"] = [node.to_dict() for node in self.body]
        result["filename"] = self.filename
        return result


@dataclass
class MCodeExpression(MCodeNode):
    """Basisklasse für Ausdrücke"""
    pass


@dataclass
class MCodeStatement(MCodeNode):
    """Basisklasse für Anweisungen"""
    pass


@dataclass
class MCodeIdentifier(MCodeExpression):
    """Repräsentiert einen Bezeichner"""
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["name"] = self.name
        return result


@dataclass
class MCodeLiteral(MCodeExpression):
    """Repräsentiert einen Literalwert"""
    value: Any
    literal_type: str  # int, float, str, bool, tensor, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["value"] = self.value
        result["literal_type"] = self.literal_type
        return result


@dataclass
class MCodeBinaryOp(MCodeExpression):
    """Repräsentiert einen binären Operator"""
    left: MCodeExpression
    operator: str
    right: MCodeExpression
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["left"] = self.left.to_dict()
        result["operator"] = self.operator
        result["right"] = self.right.to_dict()
        return result


@dataclass
class MCodeUnaryOp(MCodeExpression):
    """Repräsentiert einen unären Operator"""
    operator: str
    operand: MCodeExpression
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["operator"] = self.operator
        result["operand"] = self.operand.to_dict()
        return result


@dataclass
class MCodeAssignment(MCodeStatement):
    """Repräsentiert eine Zuweisung"""
    target: MCodeExpression
    value: MCodeExpression
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["target"] = self.target.to_dict()
        result["value"] = self.value.to_dict()
        return result


@dataclass
class MCodeFunctionDef(MCodeStatement):
    """Repräsentiert eine Funktionsdefinition"""
    name: str
    params: List[MCodeIdentifier]
    body: List[MCodeNode]
    return_type: Optional[str] = None
    decorators: List[MCodeExpression] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["name"] = self.name
        result["params"] = [param.to_dict() for param in self.params]
        result["body"] = [node.to_dict() for node in self.body]
        result["return_type"] = self.return_type
        result["decorators"] = [dec.to_dict() for dec in self.decorators]
        return result


@dataclass
class MCodeReturn(MCodeStatement):
    """Repräsentiert eine Return-Anweisung"""
    value: Optional[MCodeExpression] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.value:
            result["value"] = self.value.to_dict()
        return result


@dataclass
class MCodeIf(MCodeStatement):
    """Repräsentiert eine If-Anweisung"""
    test: MCodeExpression
    body: List[MCodeNode]
    orelse: List[MCodeNode] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["test"] = self.test.to_dict()
        result["body"] = [node.to_dict() for node in self.body]
        result["orelse"] = [node.to_dict() for node in self.orelse]
        return result


@dataclass
class MCodeFor(MCodeStatement):
    """Repräsentiert eine For-Schleife"""
    target: MCodeExpression
    iter: MCodeExpression
    body: List[MCodeNode]
    orelse: List[MCodeNode] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["target"] = self.target.to_dict()
        result["iter"] = self.iter.to_dict()
        result["body"] = [node.to_dict() for node in self.body]
        result["orelse"] = [node.to_dict() for node in self.orelse]
        return result


@dataclass
class MCodeWhile(MCodeStatement):
    """Repräsentiert eine While-Schleife"""
    test: MCodeExpression
    body: List[MCodeNode]
    orelse: List[MCodeNode] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["test"] = self.test.to_dict()
        result["body"] = [node.to_dict() for node in self.body]
        result["orelse"] = [node.to_dict() for node in self.orelse]
        return result


@dataclass
class MCodeCall(MCodeExpression):
    """Repräsentiert einen Funktionsaufruf"""
    func: MCodeExpression
    args: List[MCodeExpression] = field(default_factory=list)
    keywords: Dict[str, MCodeExpression] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["func"] = self.func.to_dict()
        result["args"] = [arg.to_dict() for arg in self.args]
        result["keywords"] = {k: v.to_dict() for k, v in self.keywords.items()}
        return result


@dataclass
class MCodeTensorOp(MCodeExpression):
    """Repräsentiert eine Tensor-Operation"""
    operation: str
    operands: List[MCodeExpression]
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["operation"] = self.operation
        result["operands"] = [op.to_dict() for op in self.operands]
        return result


@dataclass
class MCodeParallel(MCodeStatement):
    """Repräsentiert einen parallelen Ausführungsblock"""
    body: List[MCodeNode]
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["body"] = [node.to_dict() for node in self.body]
        return result


@dataclass
class MCodeProbability(MCodeExpression):
    """Repräsentiert einen Wahrscheinlichkeitsausdruck"""
    value: MCodeExpression
    probability: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["value"] = self.value.to_dict()
        result["probability"] = self.probability
        return result


class MCodeSyntaxTree:
    """Repräsentation eines M-CODE Syntaxbaums"""
    
    def __init__(self, root: MCodeModule):
        """
        Initialisiert einen neuen M-CODE Syntaxbaum.
        
        Args:
            root: Wurzelknoten des Syntaxbaums
        """
        self.root = root
    
    def __repr__(self) -> str:
        return f"MCodeSyntaxTree({self.root.filename})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert den Syntaxbaum in ein Dictionary"""
        return {
            "type": "MCodeSyntaxTree",
            "root": self.root.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCodeSyntaxTree':
        """Erstellt einen Syntaxbaum aus einem Dictionary"""
        # Diese Methode würde eine komplexe Rekonstruktion des Baums erfordern
        # Für diese Beispielimplementierung geben wir einen leeren Baum zurück
        return cls(MCodeModule(0, 0))


class MCodeLexer:
    """Lexer für M-CODE"""
    
    def __init__(self, source: str, filename: str = "<m_code>"):
        """
        Initialisiert einen neuen M-CODE Lexer.
        
        Args:
            source: M-CODE Quellcode
            filename: Name der Quelldatei
        """
        self.source = source
        self.filename = filename
        self.tokens = []
        self.current_pos = 0
        
        # M-CODE Schlüsselwörter
        self.keywords = {
            "func", "return", "if", "else", "elif", "for", "while", "break", "continue",
            "tensor", "parallel", "prob", "quantum", "import", "from", "as", "class",
            "true", "false", "null", "self", "super", "async", "await", "yield"
        }
        
        # M-CODE Operatoren
        self.operators = {
            "+", "-", "*", "/", "//", "%", "**", "=", "==", "!=", "<", ">", "<=", ">=",
            "and", "or", "not", "in", "is", "&", "|", "^", "~", "<<", ">>", "+=", "-=",
            "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=", "<<=", ">>=", "@", "@="
        }
        
        # M-CODE Quantum-Operatoren
        self.quantum_operators = {
            "⊗", "⊕", "⊖", "⊙", "⊘", "⊚", "⊛", "⊝", "⊞", "⊟", "⊠", "⊡", "⋉", "⋊", "⋋", "⋌"
        }
    
    def tokenize(self) -> List[MCodeToken]:
        """
        Tokenisiert den Quellcode.
        
        Returns:
            Liste von Tokens
        """
        # Verwende Python's tokenize-Modul als Basis
        tokens = []
        
        try:
            # Konvertiere String in BytesIO für tokenize
            source_bytes = io.BytesIO(self.source.encode('utf-8'))
            
            # Tokenisiere
            for tok in tokenize.tokenize(source_bytes.readline):
                token_type = self._map_token_type(tok)
                token_value = tok.string
                
                # Spezielle Behandlung für M-CODE spezifische Tokens
                if token_type == MCodeTokenType.IDENTIFIER and token_value in self.keywords:
                    token_type = MCodeTokenType.KEYWORD
                elif token_type == MCodeTokenType.OPERATOR and token_value in self.quantum_operators:
                    token_type = MCodeTokenType.QUANTUM_OPERATOR
                elif token_value == "parallel":
                    token_type = MCodeTokenType.PARALLEL_MARKER
                elif token_value == "prob":
                    token_type = MCodeTokenType.PROBABILITY_MARKER
                
                # Erstelle Token
                token = MCodeToken(
                    type=token_type,
                    value=token_value,
                    line=tok.start[0],
                    column=tok.start[1]
                )
                
                tokens.append(token)
        except tokenize.TokenError as e:
            logger.error(f"Fehler beim Tokenisieren: {e}")
            # Füge EOF-Token hinzu, um Parser nicht zu verwirren
            tokens.append(MCodeToken(
                type=MCodeTokenType.EOF,
                value="",
                line=0,
                column=0
            ))
        
        return tokens
    
    def _map_token_type(self, tok: tokenize.TokenInfo) -> MCodeTokenType:
        """Mappt Python-Token-Typen auf M-CODE-Token-Typen"""
        if tok.type == tokenize.NAME:
            return MCodeTokenType.IDENTIFIER
        elif tok.type == tokenize.NUMBER or tok.type == tokenize.STRING:
            return MCodeTokenType.LITERAL
        elif tok.type == tokenize.OP:
            return MCodeTokenType.OPERATOR
        elif tok.type == tokenize.COMMENT:
            return MCodeTokenType.COMMENT
        elif tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            return MCodeTokenType.WHITESPACE
        elif tok.type == tokenize.ENDMARKER:
            return MCodeTokenType.EOF
        else:
            return MCodeTokenType.UNKNOWN


class MCodeParser:
    """Parser für M-CODE"""
    
    def __init__(self, tokens: List[MCodeToken], filename: str = "<m_code>", extensions: Dict[str, Callable] = None):
        """
        Initialisiert einen neuen M-CODE Parser.
        
        Args:
            tokens: Liste von Tokens
            filename: Name der Quelldatei
            extensions: Wörterbuch mit Syntax-Erweiterungen
        """
        self.tokens = tokens
        self.filename = filename
        self.current_pos = 0
        self.extensions = extensions or {}
    
    def parse(self) -> MCodeSyntaxTree:
        """
        Parst die Tokens in einen Syntaxbaum.
        
        Returns:
            M-CODE Syntaxbaum
        """
        # In einer vollständigen Implementierung würde hier ein komplexer Parser stehen
        # Für diese Beispielimplementierung erstellen wir einen einfachen Syntaxbaum
        
        # Erstelle Wurzelknoten
        root = MCodeModule(0, 0, filename=self.filename)
        
        # Erstelle einen einfachen Syntaxbaum
        # In einer vollständigen Implementierung würde hier ein rekursiver Abstieg erfolgen
        
        # Beispiel für eine einfache Funktion im Syntaxbaum
        func_def = MCodeFunctionDef(
            line=1,
            column=0,
            name="example_function",
            params=[
                MCodeIdentifier(1, 17, "x"),
                MCodeIdentifier(1, 20, "y")
            ],
            body=[
                MCodeReturn(
                    line=2,
                    column=4,
                    value=MCodeBinaryOp(
                        line=2,
                        column=11,
                        left=MCodeIdentifier(2, 11, "x"),
                        operator="+",
                        right=MCodeIdentifier(2, 15, "y")
                    )
                )
            ]
        )
        
        root.body.append(func_def)
        
        return MCodeSyntaxTree(root)


def parse_m_code(source: str, filename: str = "<m_code>", extensions: Dict[str, Callable] = None) -> MCodeSyntaxTree:
    """
    Parst M-CODE Quellcode in einen Syntaxbaum.
    
    Args:
        source: M-CODE Quellcode
        filename: Name der Quelldatei
        extensions: Wörterbuch mit Syntax-Erweiterungen
        
    Returns:
        M-CODE Syntaxbaum
    """
    # Tokenisiere Quellcode
    lexer = MCodeLexer(source, filename)
    tokens = lexer.tokenize()
    
    # Parse Tokens
    parser = MCodeParser(tokens, filename, extensions)
    syntax_tree = parser.parse()
    
    return syntax_tree
