#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - MPRIME Formula Builder

Dynamische Formelkomposition aus semantischen Tokens für die MPRIME Engine.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple, Set

logger = logging.getLogger("MISO.Math.MPRIME.FormulaBuilder")

class FormulaBuilder:
    """
    Dynamische Formelkomposition aus semantischen Tokens
    
    Diese Klasse ermöglicht die Konstruktion mathematischer Formeln
    aus semantischen Tokens und natürlichsprachlichen Beschreibungen.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den FormulaBuilder
        
        Args:
            config: Konfigurationsobjekt für den FormulaBuilder
        """
        self.config = config or {}
        
        # Mathematische Operatoren und Symbole
        self.operators = {
            "addition": "+",
            "subtraction": "-",
            "multiplication": "*",
            "division": "/",
            "power": "^",
            "equals": "=",
            "less_than": "<",
            "greater_than": ">",
            "less_equal": "<=",
            "greater_equal": ">=",
            "not_equal": "!=",
            "integral": "∫",
            "derivative": "d/dx",
            "partial_derivative": "∂/∂x",
            "limit": "lim",
            "sum": "∑",
            "product": "∏",
            "square_root": "√",
            "infinity": "∞",
            "pi": "π",
            "euler": "e"
        }
        
        # Mathematische Funktionen
        self.functions = {
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "arcsin": "arcsin",
            "arccos": "arccos",
            "arctan": "arctan",
            "sinh": "sinh",
            "cosh": "cosh",
            "tanh": "tanh",
            "exp": "exp",
            "log": "log",
            "ln": "ln",
            "abs": "abs",
            "floor": "floor",
            "ceil": "ceil",
            "max": "max",
            "min": "min"
        }
        
        # Semantische Token-Mappings
        self.semantic_tokens = {
            # Operatoren
            "plus": "addition",
            "minus": "subtraction",
            "mal": "multiplication",
            "geteilt": "division",
            "hoch": "power",
            "gleich": "equals",
            "kleiner": "less_than",
            "größer": "greater_than",
            "kleiner gleich": "less_equal",
            "größer gleich": "greater_equal",
            "ungleich": "not_equal",
            
            # Funktionen
            "sinus": "sin",
            "kosinus": "cos",
            "tangens": "tan",
            "arkussinus": "arcsin",
            "arkuskosinus": "arccos",
            "arkustangens": "arctan",
            "exponential": "exp",
            "logarithmus": "log",
            "natürlicher logarithmus": "ln",
            "betrag": "abs",
            "abrunden": "floor",
            "aufrunden": "ceil",
            "maximum": "max",
            "minimum": "min",
            
            # Spezielle Operationen
            "integral": "integral",
            "ableitung": "derivative",
            "partielle ableitung": "partial_derivative",
            "grenzwert": "limit",
            "summe": "sum",
            "produkt": "product",
            "wurzel": "square_root",
            
            # Konstanten
            "unendlich": "infinity",
            "pi": "pi",
            "euler": "euler"
        }
        
        # Formelvorlagen
        self.templates = {
            "quadratic_equation": {
                "pattern": "quadratische gleichung",
                "template": "ax^2 + bx + c = 0",
                "variables": ["a", "b", "c"],
                "solution": "x = (-b ± √(b^2 - 4ac)) / (2a)"
            },
            "linear_equation": {
                "pattern": "lineare gleichung",
                "template": "ax + b = 0",
                "variables": ["a", "b"],
                "solution": "x = -b/a"
            },
            "pythagorean_theorem": {
                "pattern": "pythagoras",
                "template": "a^2 + b^2 = c^2",
                "variables": ["a", "b", "c"],
                "solution": "c = √(a^2 + b^2)"
            },
            "area_circle": {
                "pattern": "kreisfläche",
                "template": "A = πr^2",
                "variables": ["r"],
                "solution": "A = πr^2"
            },
            "volume_sphere": {
                "pattern": "kugelvolumen",
                "template": "V = (4/3)πr^3",
                "variables": ["r"],
                "solution": "V = (4/3)πr^3"
            }
        }
        
        logger.info("FormulaBuilder initialisiert")
    
    def build_from_description(self, description: str) -> Dict[str, Any]:
        """
        Baut eine Formel aus einer natürlichsprachlichen Beschreibung
        
        Args:
            description: Natürlichsprachliche Beschreibung der Formel
            
        Returns:
            Dictionary mit der gebauten Formel und Metadaten
        """
        # Initialisiere Ergebnis
        result = {
            "original_description": description,
            "normalized_description": None,
            "tokens": [],
            "formula": None,
            "variables": set(),
            "operators": set(),
            "functions": set(),
            "template_used": None
        }
        
        try:
            # Normalisiere Beschreibung
            normalized = self._normalize_description(description)
            result["normalized_description"] = normalized
            
            # Prüfe auf Formelvorlagen
            template_match = self._match_template(normalized)
            if template_match:
                result["template_used"] = template_match["name"]
                result["formula"] = template_match["template"]
                result["variables"] = set(template_match["variables"])
                
                logger.info(f"Formelvorlage '{template_match['name']}' erfolgreich angewendet")
                return result
            
            # Tokenisiere Beschreibung
            tokens = self._tokenize_description(normalized)
            result["tokens"] = tokens
            
            # Baue Formel aus Tokens
            formula, variables, operators, functions = self._build_formula(tokens)
            result["formula"] = formula
            result["variables"] = variables
            result["operators"] = operators
            result["functions"] = functions
            
            logger.info(f"Formel erfolgreich aus Beschreibung gebaut: {formula}")
        
        except Exception as e:
            logger.error(f"Fehler beim Bau der Formel aus Beschreibung '{description}': {str(e)}")
            raise
        
        return result
    
    def _normalize_description(self, description: str) -> str:
        """
        Normalisiert eine natürlichsprachliche Beschreibung
        
        Args:
            description: Natürlichsprachliche Beschreibung
            
        Returns:
            Normalisierte Beschreibung
        """
        # Konvertiere zu Kleinbuchstaben
        normalized = description.lower()
        
        # Entferne Satzzeichen
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Entferne mehrfache Leerzeichen
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Entferne führende und abschließende Leerzeichen
        normalized = normalized.strip()
        
        return normalized
    
    def _match_template(self, description: str) -> Optional[Dict[str, Any]]:
        """
        Sucht nach einer passenden Formelvorlage
        
        Args:
            description: Normalisierte Beschreibung
            
        Returns:
            Dictionary mit der passenden Vorlage oder None
        """
        for name, template in self.templates.items():
            if template["pattern"] in description:
                return {
                    "name": name,
                    "template": template["template"],
                    "variables": template["variables"],
                    "solution": template.get("solution")
                }
        
        return None
    
    def _tokenize_description(self, description: str) -> List[Dict[str, Any]]:
        """
        Tokenisiert eine normalisierte Beschreibung
        
        Args:
            description: Normalisierte Beschreibung
            
        Returns:
            Liste von Token-Dictionaries
        """
        tokens = []
        words = description.split()
        
        i = 0
        while i < len(words):
            # Prüfe auf Mehrwort-Tokens
            matched = False
            for j in range(min(3, len(words) - i), 0, -1):
                phrase = " ".join(words[i:i+j])
                if phrase in self.semantic_tokens:
                    token_type = self.semantic_tokens[phrase]
                    
                    # Bestimme Token-Kategorie
                    if token_type in self.operators:
                        category = "operator"
                        symbol = self.operators[token_type]
                    elif token_type in self.functions:
                        category = "function"
                        symbol = self.functions[token_type]
                    else:
                        category = "unknown"
                        symbol = token_type
                    
                    tokens.append({
                        "text": phrase,
                        "type": token_type,
                        "category": category,
                        "symbol": symbol
                    })
                    
                    i += j
                    matched = True
                    break
            
            if not matched:
                # Einzelnes Wort
                word = words[i]
                
                # Prüfe auf Variable
                if len(word) == 1 and word.isalpha():
                    tokens.append({
                        "text": word,
                        "type": "variable",
                        "category": "variable",
                        "symbol": word
                    })
                # Prüfe auf Zahl
                elif word.replace('.', '', 1).isdigit():
                    tokens.append({
                        "text": word,
                        "type": "number",
                        "category": "number",
                        "symbol": word
                    })
                # Unbekanntes Token
                else:
                    tokens.append({
                        "text": word,
                        "type": "unknown",
                        "category": "unknown",
                        "symbol": word
                    })
                
                i += 1
        
        return tokens
    
    def _build_formula(self, tokens: List[Dict[str, Any]]) -> Tuple[str, Set[str], Set[str], Set[str]]:
        """
        Baut eine Formel aus Tokens
        
        Args:
            tokens: Liste von Token-Dictionaries
            
        Returns:
            Tuple aus Formel, Variablen, Operatoren und Funktionen
        """
        formula_parts = []
        variables = set()
        operators = set()
        functions = set()
        
        for i, token in enumerate(tokens):
            category = token["category"]
            symbol = token["symbol"]
            
            if category == "operator":
                # Füge Operator hinzu
                formula_parts.append(symbol)
                operators.add(token["type"])
            elif category == "function":
                # Füge Funktion hinzu
                # Prüfe, ob das nächste Token eine Variable oder Zahl ist
                if i + 1 < len(tokens) and tokens[i + 1]["category"] in ["variable", "number"]:
                    formula_parts.append(f"{symbol}({tokens[i + 1]['symbol']})")
                    functions.add(token["type"])
                    
                    # Überspringe das nächste Token
                    i += 1
                else:
                    formula_parts.append(f"{symbol}(x)")
                    functions.add(token["type"])
                    variables.add("x")
            elif category == "variable":
                # Füge Variable hinzu
                formula_parts.append(symbol)
                variables.add(symbol)
            elif category == "number":
                # Füge Zahl hinzu
                formula_parts.append(symbol)
            else:
                # Ignoriere unbekannte Tokens
                pass
        
        # Verbinde Formelteile
        formula = " ".join(formula_parts)
        
        # Entferne mehrfache Leerzeichen
        formula = re.sub(r'\s+', ' ', formula)
        
        return formula, variables, operators, functions
    
    def build_from_template(self, template_name: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Baut eine Formel aus einer Vorlage
        
        Args:
            template_name: Name der Vorlage
            parameters: Parameter für die Vorlage (optional)
            
        Returns:
            Dictionary mit der gebauten Formel und Metadaten
        """
        # Prüfe, ob Vorlage existiert
        if template_name not in self.templates:
            raise ValueError(f"Unbekannte Formelvorlage: {template_name}")
        
        # Hole Vorlage
        template = self.templates[template_name]
        
        # Initialisiere Parameter
        parameters = parameters or {}
        
        # Initialisiere Ergebnis
        result = {
            "template_name": template_name,
            "template": template["template"],
            "formula": template["template"],
            "variables": set(template["variables"]),
            "parameters": parameters,
            "solution": template.get("solution")
        }
        
        try:
            # Ersetze Parameter in der Formel
            formula = template["template"]
            for var in template["variables"]:
                if var in parameters:
                    # Ersetze Variable durch Parameter
                    formula = formula.replace(var, str(parameters[var]))
            
            result["formula"] = formula
            
            # Aktualisiere Variablen
            remaining_variables = set()
            for var in template["variables"]:
                if var not in parameters:
                    remaining_variables.add(var)
            
            result["variables"] = remaining_variables
            
            # Aktualisiere Lösung, falls vorhanden
            if "solution" in template:
                solution = template["solution"]
                for var in template["variables"]:
                    if var in parameters:
                        # Ersetze Variable durch Parameter
                        solution = solution.replace(var, str(parameters[var]))
                
                result["solution"] = solution
            
            logger.info(f"Formel erfolgreich aus Vorlage '{template_name}' gebaut: {formula}")
        
        except Exception as e:
            logger.error(f"Fehler beim Bau der Formel aus Vorlage '{template_name}': {str(e)}")
            raise
        
        return result
    
    def add_template(self, name: str, pattern: str, template: str, variables: List[str], solution: str = None) -> bool:
        """
        Fügt eine neue Formelvorlage hinzu
        
        Args:
            name: Name der Vorlage
            pattern: Muster für die Erkennung
            template: Formelvorlage
            variables: Liste von Variablen
            solution: Lösungsformel (optional)
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        try:
            # Füge Vorlage hinzu
            self.templates[name] = {
                "pattern": pattern.lower(),
                "template": template,
                "variables": variables,
            }
            
            # Füge Lösung hinzu, falls vorhanden
            if solution:
                self.templates[name]["solution"] = solution
            
            logger.info(f"Formelvorlage '{name}' erfolgreich hinzugefügt")
            return True
        
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen der Formelvorlage '{name}': {str(e)}")
            return False
    
    def compose_formulas(self, formulas: List[Dict[str, Any]], operation: str) -> Dict[str, Any]:
        """
        Komponiert mehrere Formeln zu einer neuen Formel
        
        Args:
            formulas: Liste von Formeln
            operation: Kompositionsoperation ("add", "multiply", "compose", etc.)
            
        Returns:
            Dictionary mit der komponierten Formel und Metadaten
        """
        # Prüfe Eingabe
        if not formulas:
            raise ValueError("Keine Formeln für die Komposition angegeben")
        
        # Initialisiere Ergebnis
        result = {
            "operation": operation,
            "input_formulas": formulas,
            "composed_formula": None,
            "variables": set(),
            "operators": set(),
            "functions": set()
        }
        
        try:
            # Extrahiere Formeln und Variablen
            formula_strings = []
            all_variables = set()
            all_operators = set()
            all_functions = set()
            
            for formula_dict in formulas:
                if "formula" in formula_dict:
                    formula_strings.append(formula_dict["formula"])
                    
                    # Sammle Variablen
                    if "variables" in formula_dict:
                        all_variables.update(formula_dict["variables"])
                    
                    # Sammle Operatoren
                    if "operators" in formula_dict:
                        all_operators.update(formula_dict["operators"])
                    
                    # Sammle Funktionen
                    if "functions" in formula_dict:
                        all_functions.update(formula_dict["functions"])
            
            # Komponiere Formeln basierend auf der Operation
            if operation == "add":
                # Addition
                composed = " + ".join(f"({f})" for f in formula_strings)
            elif operation == "multiply":
                # Multiplikation
                composed = " * ".join(f"({f})" for f in formula_strings)
            elif operation == "compose":
                # Funktionskomposition (f ∘ g)(x) = f(g(x))
                if len(formula_strings) < 2:
                    raise ValueError("Mindestens zwei Formeln für die Komposition erforderlich")
                
                # Beginne mit der innersten Funktion
                composed = formula_strings[-1]
                
                # Komponiere von innen nach außen
                for i in range(len(formula_strings) - 2, -1, -1):
                    # Ersetze Variablen in der äußeren Funktion durch die innere Funktion
                    outer = formula_strings[i]
                    for var in all_variables:
                        outer = re.sub(r'\b' + re.escape(var) + r'\b', f"({composed})", outer)
                    
                    composed = outer
            else:
                # Unbekannte Operation
                raise ValueError(f"Unbekannte Kompositionsoperation: {operation}")
            
            result["composed_formula"] = composed
            result["variables"] = all_variables
            result["operators"] = all_operators
            result["functions"] = all_functions
            
            logger.info(f"Formeln erfolgreich mit Operation '{operation}' komponiert: {composed}")
        
        except Exception as e:
            logger.error(f"Fehler bei der Komposition von Formeln mit Operation '{operation}': {str(e)}")
            raise
        
        return result
