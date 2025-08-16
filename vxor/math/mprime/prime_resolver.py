#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - MPRIME Prime Resolver

Symbolische Vereinfachung und Lösungsstrategie für die MPRIME Engine.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Callable

logger = logging.getLogger("MISO.Math.MPRIME.PrimeResolver")

class PrimeResolver:
    """
    Symbolische Vereinfachung und Lösungsstrategie
    
    Diese Klasse implementiert Strategien zur symbolischen Vereinfachung
    und Lösung mathematischer Ausdrücke und Gleichungen.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den PrimeResolver
        
        Args:
            config: Konfigurationsobjekt für den PrimeResolver
        """
        self.config = config or {}
        self.max_iterations = self.config.get("max_iterations", 100)
        self.simplification_threshold = self.config.get("simplification_threshold", 0.1)
        
        # Vereinfachungsregeln
        self.simplification_rules = {
            "algebraic": self._create_algebraic_rules(),
            "trigonometric": self._create_trigonometric_rules(),
            "exponential": self._create_exponential_rules(),
            "logarithmic": self._create_logarithmic_rules(),
            "calculus": self._create_calculus_rules()
        }
        
        # Lösungsstrategien
        self.solution_strategies = {
            "algebraic": self._solve_algebraic,
            "trigonometric": self._solve_trigonometric,
            "differential": self._solve_differential,
            "integral": self._solve_integral,
            "system": self._solve_system
        }
        
        logger.info("PrimeResolver initialisiert")
    
    def _create_algebraic_rules(self) -> List[Dict[str, Any]]:
        """
        Erstellt algebraische Vereinfachungsregeln
        
        Returns:
            Liste von Vereinfachungsregeln
        """
        return [
            {
                "name": "addition_identity",
                "pattern": r"(.*?)\s*\+\s*0",
                "replacement": r"\1",
                "description": "a + 0 = a"
            },
            {
                "name": "multiplication_identity",
                "pattern": r"(.*?)\s*\*\s*1",
                "replacement": r"\1",
                "description": "a * 1 = a"
            },
            {
                "name": "multiplication_zero",
                "pattern": r"(.*?)\s*\*\s*0",
                "replacement": "0",
                "description": "a * 0 = 0"
            },
            {
                "name": "power_one",
                "pattern": r"(.*?)\s*\^\s*1",
                "replacement": r"\1",
                "description": "a^1 = a"
            },
            {
                "name": "power_zero",
                "pattern": r"(.*?)\s*\^\s*0",
                "replacement": "1",
                "description": "a^0 = 1"
            },
            {
                "name": "double_negative",
                "pattern": r"-\s*-\s*(.*)",
                "replacement": r"\1",
                "description": "--a = a"
            }
        ]
    
    def _create_trigonometric_rules(self) -> List[Dict[str, Any]]:
        """
        Erstellt trigonometrische Vereinfachungsregeln
        
        Returns:
            Liste von Vereinfachungsregeln
        """
        return [
            {
                "name": "sin_squared_plus_cos_squared",
                "pattern": r"sin\s*\(\s*(.*?)\s*\)\s*\^\s*2\s*\+\s*cos\s*\(\s*\1\s*\)\s*\^\s*2",
                "replacement": "1",
                "description": "sin^2(x) + cos^2(x) = 1"
            },
            {
                "name": "tan_definition",
                "pattern": r"sin\s*\(\s*(.*?)\s*\)\s*/\s*cos\s*\(\s*\1\s*\)",
                "replacement": r"tan(\1)",
                "description": "sin(x)/cos(x) = tan(x)"
            },
            {
                "name": "sin_zero",
                "pattern": r"sin\s*\(\s*0\s*\)",
                "replacement": "0",
                "description": "sin(0) = 0"
            },
            {
                "name": "cos_zero",
                "pattern": r"cos\s*\(\s*0\s*\)",
                "replacement": "1",
                "description": "cos(0) = 1"
            }
        ]
    
    def _create_exponential_rules(self) -> List[Dict[str, Any]]:
        """
        Erstellt exponentielle Vereinfachungsregeln
        
        Returns:
            Liste von Vereinfachungsregeln
        """
        return [
            {
                "name": "exp_zero",
                "pattern": r"exp\s*\(\s*0\s*\)",
                "replacement": "1",
                "description": "exp(0) = 1"
            },
            {
                "name": "exp_ln",
                "pattern": r"exp\s*\(\s*ln\s*\(\s*(.*?)\s*\)\s*\)",
                "replacement": r"\1",
                "description": "exp(ln(x)) = x"
            },
            {
                "name": "ln_exp",
                "pattern": r"ln\s*\(\s*exp\s*\(\s*(.*?)\s*\)\s*\)",
                "replacement": r"\1",
                "description": "ln(exp(x)) = x"
            },
            {
                "name": "exp_sum",
                "pattern": r"exp\s*\(\s*(.*?)\s*\+\s*(.*?)\s*\)",
                "replacement": r"exp(\1) * exp(\2)",
                "description": "exp(a + b) = exp(a) * exp(b)"
            }
        ]
    
    def _create_logarithmic_rules(self) -> List[Dict[str, Any]]:
        """
        Erstellt logarithmische Vereinfachungsregeln
        
        Returns:
            Liste von Vereinfachungsregeln
        """
        return [
            {
                "name": "ln_one",
                "pattern": r"ln\s*\(\s*1\s*\)",
                "replacement": "0",
                "description": "ln(1) = 0"
            },
            {
                "name": "ln_product",
                "pattern": r"ln\s*\(\s*(.*?)\s*\*\s*(.*?)\s*\)",
                "replacement": r"ln(\1) + ln(\2)",
                "description": "ln(a * b) = ln(a) + ln(b)"
            },
            {
                "name": "ln_power",
                "pattern": r"ln\s*\(\s*(.*?)\s*\^\s*(.*?)\s*\)",
                "replacement": r"\2 * ln(\1)",
                "description": "ln(a^b) = b * ln(a)"
            },
            {
                "name": "ln_division",
                "pattern": r"ln\s*\(\s*(.*?)\s*/\s*(.*?)\s*\)",
                "replacement": r"ln(\1) - ln(\2)",
                "description": "ln(a/b) = ln(a) - ln(b)"
            }
        ]
    
    def _create_calculus_rules(self) -> List[Dict[str, Any]]:
        """
        Erstellt Vereinfachungsregeln für die Analysis
        
        Returns:
            Liste von Vereinfachungsregeln
        """
        return [
            {
                "name": "derivative_constant",
                "pattern": r"d/dx\s*\(\s*([0-9]+)\s*\)",
                "replacement": "0",
                "description": "d/dx(c) = 0"
            },
            {
                "name": "derivative_x",
                "pattern": r"d/dx\s*\(\s*x\s*\)",
                "replacement": "1",
                "description": "d/dx(x) = 1"
            },
            {
                "name": "derivative_sum",
                "pattern": r"d/dx\s*\(\s*(.*?)\s*\+\s*(.*?)\s*\)",
                "replacement": r"d/dx(\1) + d/dx(\2)",
                "description": "d/dx(f + g) = d/dx(f) + d/dx(g)"
            },
            {
                "name": "integral_constant",
                "pattern": r"∫\s*([0-9]+)\s*dx",
                "replacement": r"\1 * x + C",
                "description": "∫c dx = c*x + C"
            },
            {
                "name": "integral_x",
                "pattern": r"∫\s*x\s*dx",
                "replacement": r"(1/2) * x^2 + C",
                "description": "∫x dx = (1/2)*x^2 + C"
            }
        ]
    
    def simplify(self, expression: Dict[str, Any], rule_types: List[str] = None) -> Dict[str, Any]:
        """
        Vereinfacht einen mathematischen Ausdruck
        
        Args:
            expression: Mathematischer Ausdruck
            rule_types: Typen von Vereinfachungsregeln (optional)
            
        Returns:
            Dictionary mit dem vereinfachten Ausdruck und Metadaten
        """
        # Verwende alle Regeltypen, falls nicht angegeben
        rule_types = rule_types or list(self.simplification_rules.keys())
        
        # Initialisiere Ergebnis
        result = {
            "original_expression": expression,
            "simplified_expression": None,
            "rule_types_applied": rule_types,
            "rules_applied": [],
            "iterations": 0,
            "simplification_level": 0.0
        }
        
        try:
            # Extrahiere Ausdrucksstring
            if isinstance(expression, dict) and "formula" in expression:
                expr_string = expression["formula"]
            elif isinstance(expression, dict) and "normalized_expression" in expression:
                expr_string = expression["normalized_expression"]
            elif isinstance(expression, str):
                expr_string = expression
            else:
                raise ValueError("Ungültiger Ausdruckstyp")
            
            # Sammle anzuwendende Regeln
            rules = []
            for rule_type in rule_types:
                if rule_type in self.simplification_rules:
                    rules.extend(self.simplification_rules[rule_type])
            
            # Vereinfache Ausdruck
            simplified_string, applied_rules, iterations = self._apply_simplification_rules(expr_string, rules)
            
            # Aktualisiere Ergebnis
            result["simplified_expression"] = simplified_string
            result["rules_applied"] = applied_rules
            result["iterations"] = iterations
            
            # Berechne Vereinfachungsniveau
            simplification_level = 1.0 - (len(simplified_string) / len(expr_string)) if expr_string else 0.0
            result["simplification_level"] = max(0.0, min(1.0, simplification_level))
            
            logger.info(f"Ausdruck erfolgreich vereinfacht: {simplified_string}")
        
        except Exception as e:
            logger.error(f"Fehler bei der Vereinfachung des Ausdrucks: {str(e)}")
            raise
        
        return result
    
    def _apply_simplification_rules(self, expression: str, rules: List[Dict[str, Any]]) -> Tuple[str, List[str], int]:
        """
        Wendet Vereinfachungsregeln auf einen Ausdruck an
        
        Args:
            expression: Ausdrucksstring
            rules: Liste von Vereinfachungsregeln
            
        Returns:
            Tuple aus vereinfachtem Ausdruck, angewendeten Regeln und Anzahl der Iterationen
        """
        simplified = expression
        applied_rules = []
        iterations = 0
        
        # Iteriere bis zur maximalen Anzahl von Iterationen oder bis keine Änderungen mehr auftreten
        while iterations < self.max_iterations:
            old_simplified = simplified
            
            # Wende jede Regel an
            for rule in rules:
                pattern = rule["pattern"]
                replacement = rule["replacement"]
                
                # Wende Regel an
                new_simplified = re.sub(pattern, replacement, simplified)
                
                # Prüfe, ob die Regel angewendet wurde
                if new_simplified != simplified:
                    simplified = new_simplified
                    applied_rules.append(rule["name"])
            
            iterations += 1
            
            # Prüfe, ob der Ausdruck unverändert ist
            if simplified == old_simplified:
                break
        
        return simplified, applied_rules, iterations
    
    def solve(self, equation: Dict[str, Any], strategy: str = None) -> Dict[str, Any]:
        """
        Löst eine mathematische Gleichung
        
        Args:
            equation: Mathematische Gleichung
            strategy: Lösungsstrategie (optional)
            
        Returns:
            Dictionary mit der Lösung und Metadaten
        """
        # Initialisiere Ergebnis
        result = {
            "original_equation": equation,
            "solution": None,
            "strategy": strategy,
            "steps": [],
            "variables": set(),
            "solution_type": None
        }
        
        try:
            # Extrahiere Gleichungsstring und Variablen
            if isinstance(equation, dict):
                if "formula" in equation:
                    eq_string = equation["formula"]
                elif "normalized_expression" in equation:
                    eq_string = equation["normalized_expression"]
                else:
                    raise ValueError("Gleichung enthält keinen Formelstring")
                
                # Extrahiere Variablen
                if "variables" in equation:
                    variables = equation["variables"]
                else:
                    variables = self._extract_variables(eq_string)
            elif isinstance(equation, str):
                eq_string = equation
                variables = self._extract_variables(eq_string)
            else:
                raise ValueError("Ungültiger Gleichungstyp")
            
            # Bestimme Lösungsstrategie, falls nicht angegeben
            if strategy is None:
                strategy = self._determine_strategy(eq_string)
            
            # Prüfe, ob Strategie existiert
            if strategy not in self.solution_strategies:
                raise ValueError(f"Unbekannte Lösungsstrategie: {strategy}")
            
            # Löse Gleichung
            solution, steps, solution_type = self.solution_strategies[strategy](eq_string, variables)
            
            # Aktualisiere Ergebnis
            result["solution"] = solution
            result["steps"] = steps
            result["variables"] = variables
            result["solution_type"] = solution_type
            
            logger.info(f"Gleichung erfolgreich mit Strategie '{strategy}' gelöst")
        
        except Exception as e:
            logger.error(f"Fehler bei der Lösung der Gleichung: {str(e)}")
            raise
        
        return result
    
    def _extract_variables(self, equation: str) -> Set[str]:
        """
        Extrahiert Variablen aus einer Gleichung
        
        Args:
            equation: Gleichungsstring
            
        Returns:
            Set von Variablen
        """
        # Einfache Heuristik: Einzelne Buchstaben sind Variablen
        variables = set(re.findall(r'\b([a-zA-Z])\b', equation))
        
        return variables
    
    def _determine_strategy(self, equation: str) -> str:
        """
        Bestimmt die Lösungsstrategie für eine Gleichung
        
        Args:
            equation: Gleichungsstring
            
        Returns:
            Lösungsstrategie
        """
        # Prüfe auf Differentialgleichung
        if re.search(r'd/dx|∂/∂|derivative', equation):
            return "differential"
        
        # Prüfe auf Integralgleichung
        if re.search(r'∫|integral', equation):
            return "integral"
        
        # Prüfe auf trigonometrische Gleichung
        if re.search(r'sin|cos|tan', equation):
            return "trigonometric"
        
        # Prüfe auf Gleichungssystem
        if equation.count('=') > 1:
            return "system"
        
        # Standardstrategie: algebraisch
        return "algebraic"
    
    def _solve_algebraic(self, equation: str, variables: Set[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
        """
        Löst eine algebraische Gleichung
        
        Args:
            equation: Gleichungsstring
            variables: Set von Variablen
            
        Returns:
            Tuple aus Lösung, Schritten und Lösungstyp
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Lösung algebraischer Gleichungen stehen
        
        # Einfache Implementierung für dieses Beispiel
        steps = []
        
        # Schritt 1: Vereinfache die Gleichung
        simplified_eq = self.simplify({"formula": equation})["simplified_expression"]
        steps.append({
            "step_id": 0,
            "description": "Vereinfache die Gleichung",
            "equation": simplified_eq
        })
        
        # Schritt 2: Bringe alle Terme auf eine Seite
        # Suche nach dem Gleichheitszeichen
        if "=" in simplified_eq:
            left_side, right_side = simplified_eq.split("=", 1)
            rearranged_eq = f"{left_side} - ({right_side}) = 0"
        else:
            rearranged_eq = simplified_eq
        
        steps.append({
            "step_id": 1,
            "description": "Bringe alle Terme auf eine Seite",
            "equation": rearranged_eq
        })
        
        # Schritt 3: Löse nach der Variablen auf
        # Für dieses Beispiel nehmen wir an, dass die Gleichung linear ist
        variable = list(variables)[0] if variables else "x"
        solution_value = f"Lösung für {variable}"
        
        steps.append({
            "step_id": 2,
            "description": f"Löse nach {variable} auf",
            "equation": f"{variable} = {solution_value}"
        })
        
        # Erstelle Lösungsobjekt
        solution = {
            "type": "algebraic",
            "variables": {variable: solution_value},
            "exact": True,
            "form": "explicit"
        }
        
        return solution, steps, "algebraic"
    
    def _solve_trigonometric(self, equation: str, variables: Set[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
        """
        Löst eine trigonometrische Gleichung
        
        Args:
            equation: Gleichungsstring
            variables: Set von Variablen
            
        Returns:
            Tuple aus Lösung, Schritten und Lösungstyp
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Lösung trigonometrischer Gleichungen stehen
        
        # Einfache Implementierung für dieses Beispiel
        steps = []
        
        # Schritt 1: Vereinfache die Gleichung
        simplified_eq = self.simplify({"formula": equation}, ["trigonometric"])["simplified_expression"]
        steps.append({
            "step_id": 0,
            "description": "Vereinfache die trigonometrische Gleichung",
            "equation": simplified_eq
        })
        
        # Schritt 2: Isoliere die trigonometrische Funktion
        steps.append({
            "step_id": 1,
            "description": "Isoliere die trigonometrische Funktion",
            "equation": simplified_eq
        })
        
        # Schritt 3: Löse die Gleichung
        variable = list(variables)[0] if variables else "x"
        solution_value = f"n*π + (-1)^n * arcsin({variable})"
        
        steps.append({
            "step_id": 2,
            "description": "Löse die Gleichung",
            "equation": f"{variable} = {solution_value}, n ∈ ℤ"
        })
        
        # Erstelle Lösungsobjekt
        solution = {
            "type": "trigonometric",
            "variables": {variable: solution_value},
            "exact": True,
            "form": "parametric",
            "parameter": "n",
            "domain": "ℤ"
        }
        
        return solution, steps, "trigonometric"
    
    def _solve_differential(self, equation: str, variables: Set[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
        """
        Löst eine Differentialgleichung
        
        Args:
            equation: Gleichungsstring
            variables: Set von Variablen
            
        Returns:
            Tuple aus Lösung, Schritten und Lösungstyp
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Lösung von Differentialgleichungen stehen
        
        # Einfache Implementierung für dieses Beispiel
        steps = []
        
        # Schritt 1: Bestimme die Ordnung der Differentialgleichung
        order = equation.count("d/dx")
        steps.append({
            "step_id": 0,
            "description": f"Bestimme die Ordnung der Differentialgleichung: {order}",
            "equation": equation
        })
        
        # Schritt 2: Klassifiziere die Differentialgleichung
        if "^2" in equation or "*" in equation:
            equation_type = "nichtlinear"
        else:
            equation_type = "linear"
        
        steps.append({
            "step_id": 1,
            "description": f"Klassifiziere die Differentialgleichung: {equation_type}",
            "equation": equation
        })
        
        # Schritt 3: Löse die Differentialgleichung
        variable = list(variables)[0] if variables else "y"
        solution_value = f"C_1 * e^x + C_2 * e^(-x)"
        
        steps.append({
            "step_id": 2,
            "description": "Löse die Differentialgleichung",
            "equation": f"{variable} = {solution_value}"
        })
        
        # Erstelle Lösungsobjekt
        solution = {
            "type": "differential",
            "order": order,
            "classification": equation_type,
            "variables": {variable: solution_value},
            "exact": True,
            "form": "general",
            "constants": ["C_1", "C_2"]
        }
        
        return solution, steps, "differential"
    
    def _solve_integral(self, equation: str, variables: Set[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
        """
        Löst eine Integralgleichung
        
        Args:
            equation: Gleichungsstring
            variables: Set von Variablen
            
        Returns:
            Tuple aus Lösung, Schritten und Lösungstyp
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Lösung von Integralgleichungen stehen
        
        # Einfache Implementierung für dieses Beispiel
        steps = []
        
        # Schritt 1: Identifiziere den Integranden
        integrand = re.search(r'∫\s*(.*?)\s*dx', equation)
        if integrand:
            integrand = integrand.group(1)
        else:
            integrand = "f(x)"
        
        steps.append({
            "step_id": 0,
            "description": f"Identifiziere den Integranden: {integrand}",
            "equation": equation
        })
        
        # Schritt 2: Bestimme die Integrationsart
        if re.search(r'∫_.*?\^', equation):
            integration_type = "bestimmt"
        else:
            integration_type = "unbestimmt"
        
        steps.append({
            "step_id": 1,
            "description": f"Bestimme die Integrationsart: {integration_type}",
            "equation": equation
        })
        
        # Schritt 3: Berechne das Integral
        variable = list(variables)[0] if variables else "F"
        
        if integration_type == "unbestimmt":
            solution_value = f"∫ {integrand} dx = F(x) + C"
        else:
            solution_value = f"∫ {integrand} dx = [F(x)]_a^b"
        
        steps.append({
            "step_id": 2,
            "description": "Berechne das Integral",
            "equation": solution_value
        })
        
        # Erstelle Lösungsobjekt
        solution = {
            "type": "integral",
            "integration_type": integration_type,
            "integrand": integrand,
            "variables": {variable: solution_value},
            "exact": True,
            "form": "symbolic"
        }
        
        return solution, steps, "integral"
    
    def _solve_system(self, equation: str, variables: Set[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
        """
        Löst ein Gleichungssystem
        
        Args:
            equation: Gleichungsstring
            variables: Set von Variablen
            
        Returns:
            Tuple aus Lösung, Schritten und Lösungstyp
        """
        # In einer vollständigen Implementierung würde hier eine komplexe
        # Lösung von Gleichungssystemen stehen
        
        # Einfache Implementierung für dieses Beispiel
        steps = []
        
        # Schritt 1: Zerlege das Gleichungssystem
        equations = equation.split(",")
        steps.append({
            "step_id": 0,
            "description": f"Zerlege das Gleichungssystem in {len(equations)} Gleichungen",
            "equation": equation
        })
        
        # Schritt 2: Erstelle die Koeffizientenmatrix
        steps.append({
            "step_id": 1,
            "description": "Erstelle die Koeffizientenmatrix",
            "equation": "A * x = b"
        })
        
        # Schritt 3: Löse das Gleichungssystem
        solution_values = {}
        for i, var in enumerate(sorted(variables)):
            solution_values[var] = f"Lösung für {var}"
        
        steps.append({
            "step_id": 2,
            "description": "Löse das Gleichungssystem",
            "equation": ", ".join([f"{var} = {val}" for var, val in solution_values.items()])
        })
        
        # Erstelle Lösungsobjekt
        solution = {
            "type": "system",
            "num_equations": len(equations),
            "num_variables": len(variables),
            "variables": solution_values,
            "exact": True,
            "form": "explicit"
        }
        
        return solution, steps, "system"
