#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - MPRIME Mathematikmodul - Symbol Solver

Dieses Modul implementiert den Symbol Solver für das MPRIME Mathematikmodul.
Es ist verantwortlich für die Auflösung und Manipulation von mathematischen Symbolen
und deren Beziehungen in komplexen mathematischen Ausdrücken.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import math
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
from enum import Enum, auto
from dataclasses import dataclass, field

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.math.mprime.symbol_solver")

# Versuche, T-MATHEMATICS zu importieren
try:
    from engines.t_mathematics.engine import TMathematicsEngine
    from engines.t_mathematics.tensor import Tensor
    TMATHEMATICS_AVAILABLE = True
except ImportError:
    TMATHEMATICS_AVAILABLE = False
    logger.warning("T-MATHEMATICS nicht verfügbar, verwende Standard-Implementierung")

class SymbolType(Enum):
    """Typen von mathematischen Symbolen"""
    VARIABLE = auto()
    CONSTANT = auto()
    OPERATOR = auto()
    FUNCTION = auto()
    SPECIAL = auto()
    TENSOR = auto()
    MATRIX = auto()
    VECTOR = auto()
    SCALAR = auto()
    UNKNOWN = auto()


class NodeType(Enum):
    """Typen von Knoten im Symbolbaum"""
    SYMBOL = auto()       # Blattknoten, der ein einzelnes Symbol enthält
    OPERATOR = auto()      # Innerer Knoten, der einen Operator repräsentiert
    FUNCTION = auto()      # Innerer Knoten, der eine Funktion repräsentiert
    GROUP = auto()         # Innerer Knoten, der eine Gruppierung (z.B. Klammern) repräsentiert
    ROOT = auto()          # Wurzelknoten des gesamten Ausdrucksbaums

@dataclass
class MathSymbol:
    """Repräsentation eines mathematischen Symbols"""
    name: str
    symbol_type: SymbolType
    value: Optional[Any] = None
    latex_repr: Optional[str] = None
    unicode_repr: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialisiert Standardwerte nach der Erstellung"""
        if self.latex_repr is None:
            self.latex_repr = f"\\{self.name}" if self.symbol_type == SymbolType.SPECIAL else self.name
        
        if self.unicode_repr is None:
            # Standardmäßig Unicode-Repräsentation für häufig verwendete Symbole
            unicode_map = {
                "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
                "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
                "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο",
                "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
                "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
                "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
                "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
                "inf": "∞", "partial": "∂", "nabla": "∇", "in": "∈", "notin": "∉",
                "subset": "⊂", "supset": "⊃", "cup": "∪", "cap": "∩", "emptyset": "∅",
                "forall": "∀", "exists": "∃", "nexists": "∄", "therefore": "∴", "because": "∵",
                "approx": "≈", "equiv": "≡", "neq": "≠", "leq": "≤", "geq": "≥",
                "times": "×", "div": "÷", "pm": "±", "mp": "∓", "cdot": "·",
                "rightarrow": "→", "leftarrow": "←", "uparrow": "↑", "downarrow": "↓"
            }
            
            if self.name.lower() in unicode_map:
                self.unicode_repr = unicode_map[self.name.lower()]
            else:
                self.unicode_repr = self.name
    
    def __str__(self):
        """String-Repräsentation des Symbols"""
        return self.unicode_repr if self.unicode_repr else self.name
    
    def __repr__(self):
        """Ausführliche Repräsentation des Symbols"""
        return f"MathSymbol(name='{self.name}', type={self.symbol_type.name}, value={self.value})"
    
    def to_latex(self) -> str:
        """
        Gibt die LaTeX-Repräsentation des Symbols zurück
        
        Returns:
            LaTeX-Repräsentation des Symbols
        """
        return self.latex_repr
    
    def to_unicode(self) -> str:
        """
        Gibt die Unicode-Repräsentation des Symbols zurück
        
        Returns:
            Unicode-Repräsentation des Symbols
        """
        return self.unicode_repr
    
    def set_value(self, value: Any) -> None:
        """
        Setzt den Wert des Symbols
        
        Args:
            value: Neuer Wert des Symbols
        """
        self.value = value
    
    def get_value(self) -> Any:
        """
        Gibt den Wert des Symbols zurück
        
        Returns:
            Wert des Symbols
        """
        return self.value
    
    def set_property(self, key: str, value: Any) -> None:
        """
        Setzt eine Eigenschaft des Symbols
        
        Args:
            key: Schlüssel der Eigenschaft
            value: Wert der Eigenschaft
        """
        self.properties[key] = value
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Gibt eine Eigenschaft des Symbols zurück
        
        Args:
            key: Schlüssel der Eigenschaft
            default: Standardwert, falls die Eigenschaft nicht existiert
            
        Returns:
            Wert der Eigenschaft oder Standardwert
        """
        return self.properties.get(key, default)

class SymbolTable:
    """
    Symboltabelle für mathematische Symbole
    
    Diese Klasse verwaltet eine Sammlung von mathematischen Symbolen und
    ermöglicht den Zugriff auf diese Symbole über ihren Namen.
    """
    
    def __init__(self):
        """Initialisiert die Symboltabelle"""
        self.symbols = {}
        self._init_common_symbols()
        logger.info("Symboltabelle initialisiert")
    
    def _init_common_symbols(self):
        """Initialisiert häufig verwendete mathematische Symbole"""
        # Konstanten
        self.add_symbol(MathSymbol("pi", SymbolType.CONSTANT, math.pi, "\\pi", "π"))
        self.add_symbol(MathSymbol("e", SymbolType.CONSTANT, math.e, "e", "e"))
        self.add_symbol(MathSymbol("inf", SymbolType.CONSTANT, float('inf'), "\\infty", "∞"))
        self.add_symbol(MathSymbol("nan", SymbolType.CONSTANT, float('nan'), "\\mathrm{NaN}", "NaN"))
        
        # Spezielle Symbole
        self.add_symbol(MathSymbol("partial", SymbolType.SPECIAL, None, "\\partial", "∂"))
        self.add_symbol(MathSymbol("nabla", SymbolType.SPECIAL, None, "\\nabla", "∇"))
        self.add_symbol(MathSymbol("sum", SymbolType.SPECIAL, None, "\\sum", "∑"))
        self.add_symbol(MathSymbol("prod", SymbolType.SPECIAL, None, "\\prod", "∏"))
        self.add_symbol(MathSymbol("int", SymbolType.SPECIAL, None, "\\int", "∫"))
        
        # Operatoren
        self.add_symbol(MathSymbol("+", SymbolType.OPERATOR, None, "+", "+"))
        self.add_symbol(MathSymbol("-", SymbolType.OPERATOR, None, "-", "-"))
        self.add_symbol(MathSymbol("*", SymbolType.OPERATOR, None, "\\cdot", "·"))
        self.add_symbol(MathSymbol("/", SymbolType.OPERATOR, None, "\\div", "÷"))
        self.add_symbol(MathSymbol("^", SymbolType.OPERATOR, None, "^", "^"))
        
        # Funktionen
        self.add_symbol(MathSymbol("sin", SymbolType.FUNCTION, math.sin, "\\sin", "sin"))
        self.add_symbol(MathSymbol("cos", SymbolType.FUNCTION, math.cos, "\\cos", "cos"))
        self.add_symbol(MathSymbol("tan", SymbolType.FUNCTION, math.tan, "\\tan", "tan"))
        self.add_symbol(MathSymbol("exp", SymbolType.FUNCTION, math.exp, "\\exp", "exp"))
        self.add_symbol(MathSymbol("log", SymbolType.FUNCTION, math.log, "\\log", "log"))
        self.add_symbol(MathSymbol("sqrt", SymbolType.FUNCTION, math.sqrt, "\\sqrt", "√"))
    
    def add_symbol(self, symbol: MathSymbol) -> None:
        """
        Fügt ein Symbol zur Symboltabelle hinzu
        
        Args:
            symbol: Hinzuzufügendes Symbol
        """
        if symbol.name in self.symbols:
            logger.warning(f"Symbol '{symbol.name}' existiert bereits und wird überschrieben")
        
        self.symbols[symbol.name] = symbol
    
    def get_symbol(self, name: str) -> Optional[MathSymbol]:
        """
        Gibt ein Symbol aus der Symboltabelle zurück
        
        Args:
            name: Name des Symbols
            
        Returns:
            Symbol oder None, falls das Symbol nicht existiert
        """
        return self.symbols.get(name)
    
    def remove_symbol(self, name: str) -> bool:
        """
        Entfernt ein Symbol aus der Symboltabelle
        
        Args:
            name: Name des zu entfernenden Symbols
            
        Returns:
            True, wenn das Symbol entfernt wurde, sonst False
        """
        if name in self.symbols:
            del self.symbols[name]
            return True
        
        return False
    
    def get_all_symbols(self) -> Dict[str, MathSymbol]:
        """
        Gibt alle Symbole in der Symboltabelle zurück
        
        Returns:
            Wörterbuch mit allen Symbolen
        """
        return self.symbols.copy()
    
    def get_symbols_by_type(self, symbol_type: SymbolType) -> Dict[str, MathSymbol]:
        """
        Gibt alle Symbole eines bestimmten Typs zurück
        
        Args:
            symbol_type: Typ der Symbole
            
        Returns:
            Wörterbuch mit Symbolen des angegebenen Typs
        """
        return {
            name: symbol
            for name, symbol in self.symbols.items()
            if symbol.symbol_type == symbol_type
        }
    
    def clear(self) -> None:
        """Löscht alle Symbole aus der Symboltabelle"""
        self.symbols.clear()
        self._init_common_symbols()

class SymbolSolver:
    """
    Symbol Solver für mathematische Ausdrücke
    
    Diese Klasse implementiert den Symbol Solver für das MPRIME Mathematikmodul.
    Sie ist verantwortlich für die Auflösung und Manipulation von mathematischen Symbolen
    und deren Beziehungen in komplexen mathematischen Ausdrücken.
    """
    
    def __init__(self):
        """Initialisiert den SymbolSolver"""
        self.symbol_table = SymbolTable()
        self.tmath_engine = None
        
        # Initialisiere T-MATHEMATICS-Engine, falls verfügbar
        if TMATHEMATICS_AVAILABLE:
            try:
                self.tmath_engine = TMathematicsEngine()
                logger.info("T-MATHEMATICS-Engine erfolgreich initialisiert")
            except Exception as e:
                logger.error(f"Fehler bei der Initialisierung der T-MATHEMATICS-Engine: {e}")
        
        logger.info("SymbolSolver initialisiert")
    
    def parse_expression(self, expression: str) -> List[MathSymbol]:
        """
        Parst einen mathematischen Ausdruck in eine Liste von Symbolen
        
        Args:
            expression: Zu parsender mathematischer Ausdruck
            
        Returns:
            Liste von Symbolen
        """
        # In einer realen Implementierung würde hier ein komplexer Parser verwendet
        # Für diese Beispielimplementierung verwenden wir einen einfachen Ansatz
        
        # Tokenisiere den Ausdruck
        tokens = self._tokenize_expression(expression)
        
        # Konvertiere Tokens in Symbole
        symbols = []
        
        for token in tokens:
            symbol = self.symbol_table.get_symbol(token)
            
            if symbol is None:
                # Versuche, den Token als Zahl zu interpretieren
                try:
                    value = float(token)
                    symbol = MathSymbol(token, SymbolType.SCALAR, value)
                except ValueError:
                    # Wenn es keine Zahl ist, interpretiere es als Variable
                    symbol = MathSymbol(token, SymbolType.VARIABLE)
            
            symbols.append(symbol)
        
        return symbols
    
    def _tokenize_expression(self, expression: str) -> List[str]:
        """
        Tokenisiert einen mathematischen Ausdruck
        
        Args:
            expression: Der zu tokenisierende mathematische Ausdruck
            
        Returns:
            Liste von Tokens
        """
        # Verbesserte Implementierung mit besserer Behandlung von Operatoren und Funktionen
        tokens = []
        token = ""
        i = 0
        operator_chars = set(['+', '-', '*', '/', '^', '=', '<', '>', '!'])
        
        while i < len(expression):
            char = expression[i]
            
            # Ignoriere Leerzeichen
            if char.isspace():
                if token:  # Aktuelles Token hinzufügen, wenn vorhanden
                    tokens.append(token)
                    token = ""
                i += 1
                continue
                
            # Zwei-Zeichen-Operatoren wie <=, >=, != erkennen
            if char in ['<', '>', '!'] and i + 1 < len(expression) and expression[i + 1] == '=':
                if token:  # Aktuelles Token hinzufügen, wenn vorhanden
                    tokens.append(token)
                    token = ""
                tokens.append(char + expression[i + 1])  # Zwei-Zeichen-Operator hinzufügen
                i += 2  # Zwei Zeichen überspringen
                continue
                
            # Operatoren
            if char in operator_chars:
                if token:  # Aktuelles Token hinzufügen, wenn vorhanden
                    tokens.append(token)
                    token = ""
                tokens.append(char)  # Operator hinzufügen
                i += 1
                continue
            
            # Klammern
            if char in ['(', ')']:
                if token:  # Aktuelles Token hinzufügen, wenn vorhanden
                    tokens.append(token)
                    token = ""
                tokens.append(char)  # Klammer hinzufügen
                i += 1
                continue
                
            # Zahlen und Variablen
            if char.isalnum() or char == '.':
                # Beginne ein neues Token
                token += char
                i += 1
                continue
            
            # Sonderzeichen ignorieren
            i += 1
            
        # Letztes Token hinzufügen, falls vorhanden
        if token:
            tokens.append(token)
            
        return tokens
    
    def evaluate_expression(self, expression: str, variables: Optional[Dict[str, Any]] = None) -> Any:
        """
        Wertet einen mathematischen Ausdruck aus
        
        Args:
            expression: Auszuwertender mathematischer Ausdruck
            variables: Wörterbuch mit Variablenwerten
            
        Returns:
            Ergebnis der Auswertung
        """
        # In einer realen Implementierung würde hier ein komplexer Evaluator verwendet
        # Für diese Beispielimplementierung verwenden wir einen einfachen Ansatz
        
        # Wenn T-MATHEMATICS verfügbar ist, verwende es für die Auswertung
        if self.tmath_engine:
            try:
                return self.tmath_engine.evaluate_expression(expression, variables or {})
            except Exception as e:
                logger.error(f"Fehler bei der Auswertung mit T-MATHEMATICS: {e}")
        
        # Fallback: Verwende eine einfache Auswertung
        try:
            # Ersetze Variablen durch ihre Werte
            if variables:
                for var_name, var_value in variables.items():
                    expression = expression.replace(var_name, str(var_value))
            
            # Ersetze mathematische Funktionen und Konstanten
            expression = expression.replace("pi", str(math.pi))
            expression = expression.replace("e", str(math.e))
            expression = expression.replace("sin", "math.sin")
            expression = expression.replace("cos", "math.cos")
            expression = expression.replace("tan", "math.tan")
            expression = expression.replace("exp", "math.exp")
            expression = expression.replace("log", "math.log")
            expression = expression.replace("sqrt", "math.sqrt")
            
            # Werte den Ausdruck aus
            result = eval(expression, {"math": math, "np": np})
            
            return result
        except Exception as e:
            logger.error(f"Fehler bei der Auswertung: {e}")
            return None
    
    def simplify_expression(self, expression: str) -> str:
        """
        Vereinfacht einen mathematischen Ausdruck
        
        Args:
            expression: Zu vereinfachender mathematischer Ausdruck
            
        Returns:
            Vereinfachter Ausdruck
        """
        # In einer realen Implementierung würde hier ein komplexer Simplifier verwendet
        # Für diese Beispielimplementierung geben wir den Ausdruck unverändert zurück
        return expression
    
    def solve_equation(self, equation: str, variable: str) -> List[Any]:
        """
        Löst eine Gleichung nach einer Variablen auf
        
        Args:
            equation: Zu lösende Gleichung (z.B. "x^2 + 2*x - 3 = 0")
            variable: Variable, nach der aufgelöst werden soll
            
        Returns:
            Liste der Lösungen
        """
        # In einer realen Implementierung würde hier ein komplexer Solver verwendet
        # Für diese Beispielimplementierung verwenden wir einen einfachen Ansatz für quadratische Gleichungen
        
        # Teile die Gleichung in linke und rechte Seite auf
        sides = equation.split('=')
        if len(sides) != 2:
            logger.error(f"Ungültige Gleichung: {equation}")
            return []
        
        left_side = sides[0].strip()
        right_side = sides[1].strip()
        
        # Bringe alle Terme auf die linke Seite
        if right_side != '0':
            left_side = f"({left_side}) - ({right_side})"
        
        # Versuche, die Gleichung als quadratische Gleichung zu interpretieren
        try:
            # Ersetze die Variable durch 'x'
            expression = left_side.replace(variable, 'x')
            
            # Werte den Ausdruck für x = 0, x = 1 und x = 2 aus, um die Koeffizienten zu bestimmen
            y0 = self.evaluate_expression(expression, {'x': 0})
            y1 = self.evaluate_expression(expression, {'x': 1})
            y2 = self.evaluate_expression(expression, {'x': 2})
            
            # Berechne die Koeffizienten der quadratischen Gleichung a*x^2 + b*x + c = 0
            a = (y2 - 2*y1 + y0) / 2
            b = y1 - y0 - a
            c = y0
            
            # Löse die quadratische Gleichung
            if abs(a) < 1e-10:
                # Lineare Gleichung
                if abs(b) < 1e-10:
                    if abs(c) < 1e-10:
                        # 0 = 0, unendlich viele Lösungen
                        return [float('inf')]
                    else:
                        # c = 0, keine Lösung
                        return []
                else:
                    # b*x + c = 0 => x = -c/b
                    return [-c/b]
            else:
                # Quadratische Gleichung
                discriminant = b**2 - 4*a*c
                
                if discriminant < 0:
                    # Keine reellen Lösungen
                    return []
                elif abs(discriminant) < 1e-10:
                    # Eine Lösung
                    return [-b/(2*a)]
                else:
                    # Zwei Lösungen
                    return [(-b + math.sqrt(discriminant))/(2*a), (-b - math.sqrt(discriminant))/(2*a)]
        except Exception as e:
            logger.error(f"Fehler beim Lösen der Gleichung: {e}")
            return []
    
    def differentiate(self, expression: str, variable: str) -> str:
        """
        Berechnet die Ableitung eines Ausdrucks nach einer Variablen
        
        Args:
            expression: Abzuleitender Ausdruck
            variable: Variable, nach der abgeleitet werden soll
            
        Returns:
            Abgeleiteter Ausdruck
        """
        # In einer realen Implementierung würde hier ein komplexer Differentiator verwendet
        # Für diese Beispielimplementierung geben wir einen Hinweis zurück
        return f"d({expression})/d{variable}"
    
    def integrate(self, expression: str, variable: str) -> str:
        """
        Berechnet das Integral eines Ausdrucks nach einer Variablen
        
        Args:
            expression: Zu integrierender Ausdruck
            variable: Variable, nach der integriert werden soll
            
        Returns:
            Integrierter Ausdruck
        """
        # In einer realen Implementierung würde hier ein komplexer Integrator verwendet
        # Für diese Beispielimplementierung geben wir einen Hinweis zurück
        return f"∫({expression})d{variable}"
    
    def get_symbol_table(self) -> SymbolTable:
        """
        Gibt die Symboltabelle zurück
        
        Returns:
            Symboltabelle
        """
        return self.symbol_table
    
    def add_symbol(self, symbol: MathSymbol) -> None:
        """
        Fügt ein Symbol zur Symboltabelle hinzu
        
        Args:
            symbol: Hinzuzufügendes Symbol
        """
        self.symbol_table.add_symbol(symbol)
    
    def get_symbol(self, name: str) -> Optional[MathSymbol]:
        """
        Gibt ein Symbol aus der Symboltabelle zurück
        
        Args:
            name: Name des Symbols
            
        Returns:
            Symbol oder None, falls das Symbol nicht existiert
        """
        return self.symbol_table.get_symbol(name)

@dataclass
class TreeNode:
    """Repräsentation eines Knotens im Symbolbaum"""
    node_type: NodeType
    value: Any
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, node: 'TreeNode') -> 'TreeNode':
        """Fügt einen Kindknoten hinzu und setzt dessen parent-Attribut"""
        self.children.append(node)
        node.parent = self
        return node
    
    def remove_child(self, node: 'TreeNode') -> bool:
        """Entfernt einen Kindknoten"""
        if node in self.children:
            self.children.remove(node)
            node.parent = None
            return True
        return False
    
    def __str__(self) -> str:
        """String-Repräsentation des Knotens"""
        if self.node_type == NodeType.SYMBOL:
            return f"{self.value.name if hasattr(self.value, 'name') else str(self.value)}"
        elif self.node_type == NodeType.OPERATOR:
            return f"({' '.join([str(child) for child in self.children])})"
        elif self.node_type == NodeType.FUNCTION:
            return f"{self.value}({', '.join([str(child) for child in self.children])})"
        elif self.node_type == NodeType.GROUP:
            return f"({', '.join([str(child) for child in self.children])})"
        else:  # ROOT
            return f"{', '.join([str(child) for child in self.children])}"

class SymbolTree:
    """Baum für die Repräsentation und Manipulation mathematischer Ausdrücke
    
    Diese Klasse implementiert einen Baum für die Repräsentation und Manipulation
    von mathematischen Ausdrücken. Sie ermöglicht die symbolische Verarbeitung
    von komplexen mathematischen Formeln und unterstützt verschiedene Operationen
    wie Substitution, Vereinfachung, Differentiation und Integration.
    """
    
    def __init__(self, config_or_expression: Optional[Union[Dict[str, Any], str]] = None):
        """
        Initialisiert einen neuen Symbolbaum
        
        Args:
            config_or_expression: Optional, entweder:
                - ein Konfigurationsobjekt (Dict) mit Einstellungen für den Symbolbaum
                - ein mathematischer Ausdruck als String
        """
        self.root = TreeNode(NodeType.ROOT, "ROOT")
        self.symbol_solver = get_symbol_solver()
        
        # Wenn ein Ausdruck übergeben wurde, parse ihn
        if config_or_expression is not None:
            if isinstance(config_or_expression, dict):
                # Es wurde ein Konfigurationsobjekt übergeben
                # Initialen Ausdruck aus Konfiguration extrahieren, falls vorhanden
                initial_expr = config_or_expression.get('initial_expression')
                if initial_expr and isinstance(initial_expr, str):
                    self.parse_expression(initial_expr)
            elif isinstance(config_or_expression, str):
                # Es wurde direkt ein Ausdruck übergeben
                self.parse_expression(config_or_expression)
            else:
                logger.warning(f"Unerwarteter Typ für config_or_expression: {type(config_or_expression)}")
            
        logger.info("SymbolTree initialisiert")
    
    def parse_expression(self, expression: str) -> None:
        """Parst einen Ausdruck und baut den Symbolbaum auf
        
        Args:
            expression: Der zu parsende mathematische Ausdruck
        """
        tokens = self.symbol_solver._tokenize_expression(expression)
        self._build_tree(tokens)
    
    def _build_tree(self, tokens: List[str]) -> None:
        """Baut den Symbolbaum aus den gegebenen Tokens auf
        
        Args:
            tokens: Liste von Tokens aus dem geparsten Ausdruck
        """
        # Prüfe, ob Tokens vorhanden sind
        if not tokens:
            return
            
        # Standardimplementierung für die sequentielle Verarbeitung von Tokens
        current_node = self.root  # Starte bei der Wurzel
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # 1. Verarbeite Operatoren
            if token in ['+', '-', '*', '/', '^', '=', '<', '>', '!', '<=', '>=', '!=']:
                # Erstelle einen Operator-Knoten
                op_node = TreeNode(NodeType.OPERATOR, token)
                current_node.add_child(op_node)
                
                # Bei binären Operatoren gehe zum nächsten Token als Kind des Operators
                if i + 1 < len(tokens):
                    current_node = op_node
                
            # 2. Verarbeite Klammern und Gruppen
            elif token == '(':
                # Gruppierung beginnt
                group_node = TreeNode(NodeType.GROUP, "GROUP")
                current_node.add_child(group_node)
                
                # Finde das Ende der Gruppe
                bracket_count = 1
                group_end = i + 1
                
                while group_end < len(tokens) and bracket_count > 0:
                    if tokens[group_end] == '(':
                        bracket_count += 1
                    elif tokens[group_end] == ')':
                        bracket_count -= 1
                    group_end += 1
                
                if bracket_count == 0:
                    # Rekursiv den Inhalt der Klammern verarbeiten
                    inner_tokens = tokens[i+1:group_end-1]
                    inner_tree = SymbolTree()
                    inner_tree._build_tree(inner_tokens)
                    
                    # Füge die Kinder des inneren Baums zum Gruppenknoten hinzu
                    for child in inner_tree.root.children:
                        group_node.add_child(child)
                    
                    i = group_end - 1  # Springe zum schließenden Klammersymbol
                else:
                    # Unausgewogene Klammern behandeln
                    logger.warning(f"Unausgewogene Klammern in: {tokens}")
                    
            elif token == ')':
                # Schließende Klammer: gehe zurück zur Wurzel, falls wir in einem Operator sind
                if current_node.node_type == NodeType.OPERATOR:
                    current_node = self.root
                
            # 3. Verarbeite Funktionen (z.B. sin, cos, etc.)
            elif self._is_function(token):
                # Funktionsaufrufe erkennen: token(arg1, arg2, ...)
                function_name = token.split('(')[0].strip()
                func_node = TreeNode(NodeType.FUNCTION, function_name)
                current_node.add_child(func_node)
                
                # Argumente finden und verarbeiten
                if '(' in token and ')' in token:
                    args_str = token[token.find('(')+1:token.rfind(')')]
                    if args_str.strip():
                        args_tokens = self.symbol_solver._tokenize_expression(args_str)
                        for arg_token in args_tokens:
                            if arg_token != ',':
                                # Rekursiv Argument verarbeiten
                                arg_tree = SymbolTree(arg_token)
                                # Füge die Kinder des Arguments zum Funktionsknoten hinzu
                                for child in arg_tree.root.children:
                                    func_node.add_child(child)
                
            # 4. Verarbeite Symbole (Variablen, Konstanten)
            else:
                # Symbol aus der Tabelle holen oder erstellen
                symbol = self._create_or_get_symbol(token)
                
                # Symbol-Knoten erstellen und hinzufügen
                symbol_node = TreeNode(NodeType.SYMBOL, symbol)
                current_node.add_child(symbol_node)
                
                # Nach einem Symbol zurück zur Wurzel, wenn der aktuelle Knoten ein Operator ist
                if current_node.node_type == NodeType.OPERATOR:
                    current_node = self.root
            
            # Zum nächsten Token gehen
            i += 1
    
    def _structure_tokens(self, tokens: List[str]) -> List[List[str]]:
        """Strukturiert die Tokens für eine bessere Verarbeitung komplexer Ausdrücke
        
        Args:
            tokens: Liste von Tokens aus dem geparsten Ausdruck
            
        Returns:
            Liste von Token-Gruppen, die zusammen verarbeitet werden sollten
        """
        # Gruppiere die Tokens in logische Einheiten (z.B. "a^2" als eine Gruppe)
        result = []
        i = 0
        
        while i < len(tokens):
            # Betrachte den Standardfall: Variable oder Konstante
            if i + 2 < len(tokens) and tokens[i+1] == '^':
                # Potenzierung erkannt (z.B. "a^2")
                result.append([tokens[i], tokens[i+1], tokens[i+2]])
                i += 3
            elif i + 2 < len(tokens) and tokens[i+1] in ['=', '+', '-', '*', '/']:
                # Binärer Operator mit zwei Operanden
                if tokens[i+1] == '=' and i + 3 < len(tokens) and tokens[i+3] == '^':
                    # Spezialfall für "= c^2"
                    result.append([tokens[i+1], tokens[i+2], tokens[i+3], tokens[i+4]])
                    i += 5
                else:
                    result.append([tokens[i], tokens[i+1], tokens[i+2]])
                    i += 3
            elif tokens[i] in ['=', '+', '-', '*', '/', '^']:
                # Einzelne Operatoren
                if i + 1 < len(tokens):
                    result.append([tokens[i], tokens[i+1]])
                    i += 2
                else:
                    result.append([tokens[i]])
                    i += 1
            else:
                # Einzelne Tokens (Variablen, Konstanten, Klammern)
                result.append([tokens[i]])
                i += 1
        
        return result
    
    def _add_token_group_to_tree(self, token_group: List[str]) -> None:
        """Fügt eine Gruppe von Tokens zum Symbolbaum hinzu
        
        Args:
            token_group: Gruppe von zusammenhängenden Tokens
        """
        if not token_group:
            return
            
        # Behandle verschiedene Arten von Token-Gruppen
        if len(token_group) == 1:
            # Einzelnes Token (Variable, Konstante, einzelner Operator)
            token = token_group[0]
            
            if token in ['+', '-', '*', '/', '^', '=', '<', '>', '<=', '>=', '!=']:
                # Operator-Knoten
                operator_node = TreeNode(NodeType.OPERATOR, token)
                self.root.add_child(operator_node)
            elif token.startswith('(') and token.endswith(')'):
                # Gruppierungsknoten
                group_content = token[1:-1]  # Entferne Klammern
                group_node = TreeNode(NodeType.GROUP, "GROUP")
                self.root.add_child(group_node)
                
                # Rekursiv parsen
                group_tokens = self.symbol_solver._tokenize_expression(group_content)
                self._add_subtree(group_node, group_tokens)
            elif self._is_function(token):
                # Funktionsknoten
                function_name = token.split('(')[0]
                function_node = TreeNode(NodeType.FUNCTION, function_name)
                self.root.add_child(function_node)
                
                # Argumente extrahieren und rekursiv parsen
                args_str = token[len(function_name)+1:-1]  # Entferne Funktion und Klammern
                if args_str:
                    args_tokens = self.symbol_solver._tokenize_expression(args_str)
                    self._add_subtree(function_node, args_tokens)
            else:
                # Symbol-Knoten (Variable oder Konstante)
                self._add_symbol_to_tree(token, self.root)
        
        elif len(token_group) >= 2:
            # Mehrere Tokens, die zusammen verarbeitet werden sollten
            if token_group[0] in ['+', '-', '*', '/', '^', '=', '<', '>', '<=', '>=', '!=']:
                # Operator mit nachfolgendem Operanden
                operator_node = TreeNode(NodeType.OPERATOR, token_group[0])
                self.root.add_child(operator_node)
                
                # Füge den Operanden als Kind des Operators hinzu
                if len(token_group) > 1:
                    for token in token_group[1:]:
                        if token in ['^', '=']:
                            # Neuer Operator innerhalb der Gruppe
                            sub_operator = TreeNode(NodeType.OPERATOR, token)
                            operator_node.add_child(sub_operator)
                        else:
                            # Symbol als Kind des Operators
                            self._add_symbol_to_tree(token, operator_node)
            
            elif len(token_group) >= 3 and token_group[1] == '^':
                # Potenzausdruck (z.B. "a^2")
                base_token = token_group[0]
                exponent_token = token_group[2]
                
                # Füge die Basis zum Baum hinzu
                base_symbol = self._create_or_get_symbol(base_token)
                base_node = TreeNode(NodeType.SYMBOL, base_symbol)
                self.root.add_child(base_node)
                
                # Füge den Potenzoperator hinzu
                power_node = TreeNode(NodeType.OPERATOR, '^')
                self.root.add_child(power_node)
                
                # Füge den Exponenten hinzu
                exponent_symbol = self._create_or_get_symbol(exponent_token)
                exponent_node = TreeNode(NodeType.SYMBOL, exponent_symbol)
                power_node.add_child(exponent_node)
    
    def _add_symbol_to_tree(self, token: str, parent_node: TreeNode) -> None:
        """Fügt ein Symbol zum Symbolbaum hinzu
        
        Args:
            token: Symbol-Token
            parent_node: Elternknoten, an den das Symbol angehängt werden soll
        """
        symbol = self._create_or_get_symbol(token)
        symbol_node = TreeNode(NodeType.SYMBOL, symbol)
        parent_node.add_child(symbol_node)
    
    def _create_or_get_symbol(self, token: str) -> MathSymbol:
        """Erstellt ein neues Symbol oder gibt ein bestehendes aus der Symboltabelle zurück
        
        Args:
            token: Symbol-Token
            
        Returns:
            MathSymbol-Instanz
        """
        symbol = self.symbol_solver.get_symbol(token)
        if symbol is None:
            # Wenn das Symbol nicht in der Tabelle ist, erstelle ein neues
            if token.isalpha():
                symbol_type = SymbolType.VARIABLE
            elif self._is_numeric(token):
                symbol_type = SymbolType.CONSTANT
            else:
                symbol_type = SymbolType.UNKNOWN
            symbol = MathSymbol(token, symbol_type=symbol_type)
            self.symbol_solver.add_symbol(symbol)
        
        return symbol
    
    def _add_subtree(self, parent_node: TreeNode, tokens: List[str]) -> None:
        """Fügt einen Teilbaum zum gegebenen Elternknoten hinzu
        
        Args:
            parent_node: Elternknoten, zu dem der Teilbaum hinzugefügt werden soll
            tokens: Liste von Tokens für den Teilbaum
        """
        # Vereinfachte Implementierung für den Anfang
        for token in tokens:
            if token in ['+', '-', '*', '/', '^', '=']:
                operator_node = TreeNode(NodeType.OPERATOR, token)
                parent_node.add_child(operator_node)
            else:
                symbol = self.symbol_solver.get_symbol(token)
                if symbol is None:
                    if token.isalpha():
                        symbol_type = SymbolType.VARIABLE
                    elif self._is_numeric(token):
                        symbol_type = SymbolType.CONSTANT
                    else:
                        symbol_type = SymbolType.UNKNOWN
                    symbol = MathSymbol(token, symbol_type=symbol_type)
                    self.symbol_solver.add_symbol(symbol)
                symbol_node = TreeNode(NodeType.SYMBOL, symbol)
                parent_node.add_child(symbol_node)
    
    def _is_function(self, token: str) -> bool:
        """Prüft, ob ein Token eine Funktion ist
        
        Args:
            token: Zu prüfendes Token
            
        Returns:
            True, wenn das Token eine Funktion ist, sonst False
        """
        return '(' in token and ')' in token and token.index('(') > 0
    
    def _is_numeric(self, token: str) -> bool:
        """Prüft, ob ein Token eine Zahl ist
        
        Args:
            token: Zu prüfendes Token
            
        Returns:
            True, wenn das Token eine Zahl ist, sonst False
        """
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def to_expression(self) -> str:
        """Konvertiert den Symbolbaum zurück in einen Ausdruck
        
        Returns:
            Der Ausdruck als String
        """
        if not self.root or not self.root.children:
            return ""
        
        # Um Ausdrücke wie a^2 + b^2 = c^2 korrekt zu verarbeiten, führen wir eine
        # spezielle Verarbeitung des Baums durch
        return self._structure_expression(self.root)
    
    def _structure_expression(self, root_node: TreeNode) -> str:
        """Strukturiert einen Ausdruck basierend auf dem Symbolbaum
        
        Args:
            root_node: Der Wurzelknoten des Baums
            
        Returns:
            Der strukturierte Ausdruck als String
        """
        # Baue den Ausdruck aus allen Knoten im Baum auf
        result = []
        self._process_node_for_expression(root_node, result)
        
        # Kombiniere die Teile und entferne überflüssige Leerzeichen
        result_str = " ".join(result)
        result_str = re.sub(r'\s+', ' ', result_str).strip()
        return result_str
    
    def _process_node_for_expression(self, node: TreeNode, result: List[str]) -> None:
        """Verarbeitet einen Knoten für den Ausdruck
        
        Args:
            node: Der zu verarbeitende Knoten
            result: Liste der bisher gesammelten Ausdrucksteile
        """
        if node.node_type == NodeType.ROOT:
            # Root-Knoten: Verarbeite alle Kinder
            for child in node.children:
                self._process_node_for_expression(child, result)
        
        elif node.node_type == NodeType.SYMBOL:
            # Symbol-Knoten: Füge den Namen hinzu
            symbol = node.value
            if hasattr(symbol, 'name'):
                result.append(symbol.name)
            else:
                result.append(str(symbol))
        
        elif node.node_type == NodeType.OPERATOR:
            # Operator-Knoten: Verarbeite je nach Operator-Typ
            operator = node.value
            result.append(operator)
            
            # Verarbeite die Kinder des Operators
            for child in node.children:
                self._process_node_for_expression(child, result)
        
        elif node.node_type == NodeType.FUNCTION:
            # Funktions-Knoten: Formatiere als f(args)
            function_name = node.value
            args = []
            
            # Sammle die Argumente
            for child in node.children:
                arg_parts = []
                self._process_node_for_expression(child, arg_parts)
                args.append("".join(arg_parts))
            
            # Füge die formatierte Funktion hinzu
            result.append(f"{function_name}({', '.join(args)})")
        
        elif node.node_type == NodeType.GROUP:
            # Gruppen-Knoten: Umschließe mit Klammern
            group_parts = []
            
            # Sammle den Inhalt der Gruppe
            for child in node.children:
                self._process_node_for_expression(child, group_parts)
            
            # Füge die formatierte Gruppe hinzu
            result.append(f"({' '.join(group_parts)})")

    
    def evaluate(self, variables: Optional[Dict[str, Any]] = None) -> Any:
        """Evaluiert den Ausdruck mit den gegebenen Variablenwerten
        
        Args:
            variables: Dictionary mit Variablenwerten
            
        Returns:
            Ergebnis der Evaluation
        """
        expr = self.to_expression()
        return self.symbol_solver.evaluate_expression(expr, variables)
    
    def simplify(self) -> 'SymbolTree':
        """Vereinfacht den Symbolbaum
        
        Returns:
            Der vereinfachte Symbolbaum
        """
        # Vereinfachte Implementierung - verwendet den SymbolSolver
        expr = self.to_expression()
        simplified_expr = self.symbol_solver.simplify_expression(expr)
        result = SymbolTree(simplified_expr)
        return result
    
    def differentiate(self, variable: str) -> 'SymbolTree':
        """Berechnet die Ableitung nach der angegebenen Variablen
        
        Args:
            variable: Name der Variablen, nach der abgeleitet werden soll
            
        Returns:
            Symbolbaum der Ableitung
        """
        expr = self.to_expression()
        diff_expr = self.symbol_solver.differentiate(expr, variable)
        result = SymbolTree(diff_expr)
        return result
    
    def integrate(self, variable: str) -> 'SymbolTree':
        """Berechnet das Integral nach der angegebenen Variablen
        
        Args:
            variable: Name der Variablen, nach der integriert werden soll
            
        Returns:
            Symbolbaum des Integrals
        """
        expr = self.to_expression()
        int_expr = self.symbol_solver.integrate(expr, variable)
        result = SymbolTree(int_expr)
        return result
    
    def substitute(self, variable: str, replacement: Union[str, 'SymbolTree']) -> 'SymbolTree':
        """
        Ersetzt eine Variable durch einen anderen Ausdruck
        
        Args:
            variable: Name der zu ersetzenden Variablen
            replacement: Ersetzungsausdruck oder -baum
            
        Returns:
            Neuer Symbolbaum mit vorgenommener Substitution
        """
        # Robustere Implementierung
        try:
            expr = self.to_expression()
            
            # Stellen sicher, dass expr ein String ist
            if not isinstance(expr, str):
                logger.warning(f"Expression ist kein String, sondern {type(expr)}")
                expr = str(expr) if expr is not None else ""
            
            # Bestimme den Ersetzungsausdruck
            if isinstance(replacement, SymbolTree):
                replacement_expr = replacement.to_expression()
                if not isinstance(replacement_expr, str):
                    replacement_expr = str(replacement_expr) if replacement_expr is not None else ""
            else:
                replacement_expr = str(replacement) if replacement is not None else ""
                
            # Führe die Ersetzung durch
            if expr and variable:
                modified_expr = expr.replace(variable, f"({replacement_expr})")
            else:
                modified_expr = expr
                
            result = SymbolTree(modified_expr)
            return result
        except Exception as e:
            logger.error(f"Fehler bei der Substitution: {e}")
            # Fallback: Gib den ursprünglichen Baum zurück
            return SymbolTree(self.to_expression())
            
    def parse(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parst einen Eingabetext und liefert das Ergebnis zurück
        
        Diese Methode wird von der MPRIME Engine aufgerufen und ermöglicht die
        Integration mit anderen Komponenten.
        
        Args:
            input_text: Der zu parsende Text (mathematischer Ausdruck)
            context: Optionaler Kontext für die Verarbeitung
            
        Returns:
            Dictionary mit dem Parserergebnis und relevanten Metadaten
        """
        result = {
            "success": False,
            "expression": input_text,
            "parsed_tree": None,
            "symbols": [],
            "error": None
        }
        
        try:
            # Parse den Ausdruck
            self.parse_expression(input_text)
            result["success"] = True
            
            # Extrahiere Symbole und Struktur für das Ergebnis
            result["parsed_tree"] = self._to_dict()
            result["symbols"] = self._extract_symbols()
            
            return result
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Fehler beim Parsen mit Kontext: {e}")
            return result
            
    def _to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert den Symbolbaum in ein Dictionary
        
        Returns:
            Dictionary-Repräsentation des Symbolbaums
        """
        def _node_to_dict(node):
            if node.node_type == NodeType.SYMBOL and isinstance(node.value, MathSymbol):
                value = node.value.name
            else:
                value = str(node.value)
                
            result = {
                "type": node.node_type.name,
                "value": value,
                "children": []
            }
            for child in node.children:
                result["children"].append(_node_to_dict(child))
            return result
            
        return _node_to_dict(self.root)
        
    def _extract_symbols(self) -> List[str]:
        """
        Extrahiert alle Symbole (Variablen) aus dem Baum
        
        Returns:
            Liste aller im Baum vorkommenden Symbole
        """
        symbols = []
        
        def _collect_symbols(node):
            if node.node_type == NodeType.SYMBOL:
                # Wenn der Wert ein MathSymbol ist, extrahiere den Namen
                if isinstance(node.value, MathSymbol):
                    if node.value.symbol_type == SymbolType.VARIABLE:
                        symbols.append(node.value.name)
                # Sonst verwende den Wert direkt
                elif isinstance(node.value, str) and node.value.isalpha():
                    symbols.append(node.value)
            
            # Rekursiv für alle Kindknoten
            for child in node.children:
                _collect_symbols(child)
                
        _collect_symbols(self.root)
        return list(set(symbols))  # Entferne Duplikate


# Globale Instanz
_SYMBOL_SOLVER = None
_SYMBOL_TREE = None

def get_symbol_solver() -> SymbolSolver:
    """
    Gibt die globale SymbolSolver-Instanz zurück
    
    Returns:
        SymbolSolver-Instanz
    """
    global _SYMBOL_SOLVER
    if _SYMBOL_SOLVER is None:
        _SYMBOL_SOLVER = SymbolSolver()
    return _SYMBOL_SOLVER

def get_symbol_tree() -> SymbolTree:
    """
    Gibt die globale SymbolTree-Instanz zurück
    
    Returns:
        SymbolTree-Instanz
    """
    global _SYMBOL_TREE
    if _SYMBOL_TREE is None:
        _SYMBOL_TREE = SymbolTree()
    return _SYMBOL_TREE


# Exportiere wichtige Klassen und Funktionen
__all__ = [
    'SymbolType', 'NodeType', 'MathSymbol', 'SymbolTable', 'SymbolSolver', 'SymbolTree',
    'TreeNode', 'get_symbol_solver', 'get_symbol_tree'
]
