#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK MPrime Integration

Integrationsschicht zwischen Q-LOGIK und MPrime Engine.
Ermöglicht die Verwendung von symbolisch-topologischen Operationen in Q-LOGIK und
die Steuerung der MPrime Engine durch Q-LOGIK-Entscheidungen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import os
import sys

# Importiere NumPy (erforderlich)
import numpy as np

# Importiere Q-LOGIK-Komponenten
from miso.logic.qlogik_engine import (
    BayesianDecisionCore,
    FuzzyLogicUnit,
    SymbolMap,
    ConflictResolver,
    simple_emotion_weight,
    simple_priority_mapping,
    advanced_qlogik_decision
)

# Importiere MPrime-Komponenten
from miso.math.mprime_engine import MPrimeEngine
from miso.math.mprime.symbol_solver import SymbolTree

# Logger einrichten
logger = logging.getLogger("MISO.Logic.Q-LOGIK.MPrimeIntegration")


class SymbolicOperationSelector:
    """
    Symbolische Operationsauswahl
    
    Wählt basierend auf Q-LOGIK-Entscheidungen die optimale
    symbolische Operation und Verarbeitungsstrategie aus.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den Symbolischen Operationsauswähler
        
        Args:
            config: Konfigurationsobjekt für den Auswähler
        """
        self.config = config or {}
        
        # Standardkonfiguration
        self.complexity_threshold = self.config.get("complexity_threshold", 0.7)
        self.auto_select_strategy = self.config.get("auto_select_strategy", True)
        self.preferred_strategy = self.config.get("preferred_strategy", "symbolische Verarbeitung")
        
        # Verfügbare Strategien
        self.available_strategies = [
            "symbolische Verarbeitung", 
            "topologische Verformung", 
            "probabilistische Analyse", 
            "babylonische Logik"
        ]
        
        logger.info("SymbolicOperationSelector initialisiert")
        
    def select_strategy(self, operation_context: Dict[str, Any]) -> str:
        """
        Wählt die optimale Strategie für eine symbolische Operation
        
        Args:
            operation_context: Kontext der Operation
            
        Returns:
            Name der ausgewählten Strategie
        """
        if not self.auto_select_strategy:
            return self.preferred_strategy
            
        # Extrahiere relevante Faktoren
        expression_complexity = operation_context.get("expression_complexity", 0.5)
        operation_type = operation_context.get("operation_type", "unknown")
        symbolic_depth = operation_context.get("symbolic_depth", 0.5)
        topological_factors = operation_context.get("topological_factors", 0.3)
        
        # Entscheidungsfaktoren
        strategy_factors = {
            "symbolische Verarbeitung": {
                "complexity_threshold": 0.6,
                "preferred_ops": ["parse", "simplify", "solve"],
                "symbolic_efficiency": 0.9,
                "topological_efficiency": 0.3
            },
            "topologische Verformung": {
                "complexity_threshold": 0.7,
                "preferred_ops": ["transform", "morph", "curve"],
                "symbolic_efficiency": 0.5,
                "topological_efficiency": 0.9
            },
            "probabilistische Analyse": {
                "complexity_threshold": 0.5,
                "preferred_ops": ["probability", "likelihood", "bayes"],
                "symbolic_efficiency": 0.7,
                "topological_efficiency": 0.4
            },
            "babylonische Logik": {
                "complexity_threshold": 0.8,
                "preferred_ops": ["logic", "inference", "deduction"],
                "symbolic_efficiency": 0.8,
                "topological_efficiency": 0.6
            }
        }
        
        # Berechne Scores für jede Strategie
        strategy_scores = {}
        
        for strategy, factors in strategy_factors.items():
            # Basiswert
            score = 0.5
            
            # Komplexitätsanpassung
            if expression_complexity > factors["complexity_threshold"]:
                score += 0.2
            else:
                score -= 0.1
                
            # Operationstyp
            if any(op in operation_type.lower() for op in factors["preferred_ops"]):
                score += 0.2
                
            # Symbolische Effizienz
            score += (symbolic_depth * factors["symbolic_efficiency"])
            
            # Topologische Effizienz
            score += (topological_factors * factors["topological_efficiency"])
                
            strategy_scores[strategy] = score
            
        # Wähle die Strategie mit dem höchsten Score
        selected_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Strategie-Auswahl für {operation_type}: {selected_strategy} (Score: {strategy_scores[selected_strategy]:.2f})")
        
        return selected_strategy


class QLOGIKMPrimeIntegration:
    """
    Q-LOGIK MPrime Integration
    
    Integrationsschicht zwischen Q-LOGIK und MPrime Engine.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialisiert die Integrationsschicht
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        # Lade Konfiguration
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Konfiguration: {str(e)}")
                
        # Initialisiere Q-LOGIK-Komponenten
        self.bayesian = BayesianDecisionCore()
        self.fuzzylogic = FuzzyLogicUnit()
        self.symbolmap = SymbolMap()
        self.conflict_resolver = ConflictResolver()
        self.emotion_weight_function = simple_emotion_weight
        self.priority_mapping_function = simple_priority_mapping
        
        # Initialisiere MPrime-Engine
        self.mprime_engine = MPrimeEngine()
        
        # Initialisiere Symbolischen Operationsauswähler
        symbolic_ops_config = self.config.get("symbolic_operations", {})
        self.symbolic_selector = SymbolicOperationSelector(config=symbolic_ops_config)
        
        logger.info("Q-LOGIK MPrime Integration initialisiert")
        
    def decide_symbolic_operation(self, operation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entscheidet über die Durchführung einer symbolischen Operation
        
        Args:
            operation_context: Kontext der Operation
            
        Returns:
            Entscheidungsergebnis
        """
        # Bereite Q-LOGIK-Kontext vor
        qlogik_context = {
            "data": {
                "hypothesis": "symbolic_operation_feasible",
                "evidence": {
                    "expression_complexity": {
                        "value": operation_context.get("expression_complexity", 0.5),
                        "weight": 0.6
                    },
                    "symbolic_depth": {
                        "value": operation_context.get("symbolic_depth", 0.5),
                        "weight": 0.4
                    }
                }
            },
            "signal": {
                "system_capacity": operation_context.get("system_capacity", 0.7),
                "operation_priority": operation_context.get("priority", 0.5)
            },
            "risk": operation_context.get("risk", 0.3),
            "benefit": operation_context.get("benefit", 0.7),
            "urgency": operation_context.get("urgency", 0.5)
        }
        
        # Führe Q-LOGIK-Entscheidung durch
        decision_result = advanced_qlogik_decision(qlogik_context)
        
        # Wähle Strategie basierend auf der Entscheidung
        if decision_result["decision"] == "JA":
            selected_strategy = self.symbolic_selector.select_strategy(operation_context)
            
            # Füge Strategie-Informationen hinzu
            decision_result["symbolic_operation"] = {
                "execute": True,
                "strategy": selected_strategy,
                "priority": decision_result["priority"]["level"]
            }
        else:
            decision_result["symbolic_operation"] = {
                "execute": False,
                "reason": "Operation nicht empfohlen",
                "alternative": "Verwende vereinfachte Operation oder verschiebe"
            }
            
        return decision_result
        
    def execute_symbolic_operation(self, 
                                 expression: str, 
                                 operation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Führt eine symbolische Operation aus
        
        Args:
            expression: Symbolischer Ausdruck
            operation_context: Kontext der Operation
            
        Returns:
            Ergebnis der Operation
        """
        # Standardkontext, falls nicht angegeben
        if operation_context is None:
            operation_context = {}
            
        # Schätze Komplexität des Ausdrucks
        expression_complexity = self._estimate_expression_complexity(expression)
        operation_context["expression_complexity"] = expression_complexity
        
        # Entscheide über die Operation
        decision = self.decide_symbolic_operation(operation_context)
        
        # Führe Operation aus, wenn empfohlen
        if decision["symbolic_operation"]["execute"]:
            strategy = decision["symbolic_operation"]["strategy"]
            
            try:
                # Bereite MPrime-Kontext vor
                mprime_context = {
                    "process_context": strategy,
                    "theta": operation_context.get("theta", math.pi/4),
                    "depth": operation_context.get("depth", 3),
                    "qlogik_decision": decision
                }
                
                # Führe Operation mit MPrime Engine aus
                result = self.mprime_engine.process(expression, mprime_context)
                
                logger.info(f"Symbolische Operation erfolgreich mit Strategie: {strategy}")
                
                return {
                    "success": True,
                    "result": result,
                    "strategy": strategy,
                    "decision": decision
                }
                
            except Exception as e:
                logger.error(f"Fehler bei symbolischer Operation: {str(e)}")
                
                # Versuche Fallback
                try:
                    # Fallback zur einfachen symbolischen Verarbeitung
                    symbol_tree = SymbolTree()
                    parsed_result = symbol_tree.parse_expression(expression)
                    
                    return {
                        "success": True,
                        "result": {
                            "parsed": parsed_result,
                            "formula": str(parsed_result.get("normalized", expression))
                        },
                        "strategy": "Fallback: Einfache symbolische Verarbeitung",
                        "decision": decision,
                        "warning": f"Ursprünglicher Fehler: {str(e)}"
                    }
                    
                except Exception as e2:
                    return {
                        "success": False,
                        "error": f"Operation fehlgeschlagen: {str(e2)}",
                        "decision": decision
                    }
        else:
            # Operation nicht empfohlen
            return {
                "success": False,
                "error": "Operation nicht empfohlen",
                "reason": decision["symbolic_operation"]["reason"],
                "alternative": decision["symbolic_operation"]["alternative"],
                "decision": decision
            }
            
    def _estimate_expression_complexity(self, expression: str) -> float:
        """
        Schätzt die Komplexität eines symbolischen Ausdrucks
        
        Args:
            expression: Symbolischer Ausdruck
            
        Returns:
            Komplexitätswert zwischen 0 und 1
        """
        # Einfache Heuristiken zur Komplexitätsschätzung
        
        # Länge des Ausdrucks
        length_factor = min(1.0, len(expression) / 200.0)
        
        # Anzahl der Operatoren
        operators = ['+', '-', '*', '/', '^', '√', '∫', '∂', '∑', '∏', '=', '<', '>', '≤', '≥', '≠']
        operator_count = sum(expression.count(op) for op in operators)
        operator_factor = min(1.0, operator_count / 20.0)
        
        # Verschachtelungstiefe
        nesting_level = 0
        max_nesting = 0
        for char in expression:
            if char in '({[':
                nesting_level += 1
                max_nesting = max(max_nesting, nesting_level)
            elif char in ')}]':
                nesting_level = max(0, nesting_level - 1)
                
        nesting_factor = min(1.0, max_nesting / 5.0)
        
        # Spezielle Funktionen
        functions = ['sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'abs']
        function_count = sum(expression.count(func) for func in functions)
        function_factor = min(1.0, function_count / 5.0)
        
        # Gewichtete Kombination der Faktoren
        complexity = (
            0.2 * length_factor +
            0.3 * operator_factor +
            0.3 * nesting_factor +
            0.2 * function_factor
        )
        
        return complexity


class MLINGUAMPrimeIntegration:
    """
    M-LINGUA MPrime Integration
    
    Ermöglicht die Steuerung der MPrime Engine durch
    natürliche Sprache über das M-LINGUA Interface.
    """
    
    def __init__(self, qlogik_mprime: QLOGIKMPrimeIntegration = None):
        """
        Initialisiert die M-LINGUA MPrime Integration
        
        Args:
            qlogik_mprime: Q-LOGIK MPrime Integration
        """
        self.qlogik_mprime = qlogik_mprime or QLOGIKMPrimeIntegration()
        
        # Sprachbefehle und ihre Zuordnungen
        self.command_mappings = {
            "löse": "solve",
            "vereinfache": "simplify",
            "transformiere": "transform",
            "analysiere": "analyze",
            "berechne": "compute",
            "beweise": "prove"
        }
        
        # Kontextparameter und ihre Standardwerte
        self.default_context = {
            "expression_complexity": 0.5,
            "symbolic_depth": 0.6,
            "system_capacity": 0.8,
            "priority": 0.5,
            "risk": 0.3,
            "benefit": 0.7,
            "urgency": 0.5,
            "topological_factors": 0.4
        }
        
        logger.info("M-LINGUA MPrime Integration initialisiert")
        
    def process_natural_language(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet natürlichsprachliche Anfragen für symbolische Operationen
        
        Args:
            text: Natürlichsprachlicher Text
            context: Kontextinformationen (optional)
            
        Returns:
            Verarbeitungsergebnis
        """
        # Standardkontext, falls nicht angegeben
        operation_context = self.default_context.copy()
        if context:
            operation_context.update(context)
            
        # Bestimme Operationstyp aus Text
        operation_type = self._extract_operation_type(text)
        operation_context["operation_type"] = operation_type or "unknown"
        
        # Extrahiere den symbolischen Ausdruck aus dem Text
        expression = self._extract_expression(text)
        
        if not expression:
            return {
                "success": False,
                "error": "Konnte keinen symbolischen Ausdruck im Text erkennen",
                "text": text
            }
            
        # Extrahiere zusätzliche Kontextinformationen aus Text
        context_updates = self._extract_context_from_text(text)
        operation_context.update(context_updates)
        
        # Führe Operation aus
        return self.qlogik_mprime.execute_symbolic_operation(
            expression=expression,
            operation_context=operation_context
        )
            
    def _extract_operation_type(self, text: str) -> Optional[str]:
        """
        Extrahiert den Operationstyp aus natürlichsprachlichem Text
        
        Args:
            text: Natürlichsprachlicher Text
            
        Returns:
            Operationstyp oder None, wenn nicht erkannt
        """
        text = text.lower()
        
        for command, operation in self.command_mappings.items():
            if command in text:
                return operation
                
        return None
        
    def _extract_expression(self, text: str) -> Optional[str]:
        """
        Extrahiert den symbolischen Ausdruck aus natürlichsprachlichem Text
        
        Args:
            text: Natürlichsprachlicher Text
            
        Returns:
            Symbolischer Ausdruck oder None, wenn nicht erkannt
        """
        # Einfache Heuristik: Suche nach mathematischen Ausdrücken in Anführungszeichen oder nach "Ausdruck"
        import re
        
        # Suche nach Ausdrücken in Anführungszeichen
        quote_patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r'\"([^\"]+)\"',
            r'\<([^\>]+)\>'
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Prüfe, ob der gefundene Ausdruck mathematische Symbole enthält
                for match in matches:
                    if any(char in match for char in "+-*/^=()[]{}"):
                        return match
        
        # Suche nach "Ausdruck" oder "Formel" gefolgt von einem Doppelpunkt
        expr_patterns = [
            r"ausdruck:?\s*([^\n.]+)",
            r"formel:?\s*([^\n.]+)",
            r"gleichung:?\s*([^\n.]+)",
            r"expression:?\s*([^\n.]+)"
        ]
        
        for pattern in expr_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return matches[0].strip()
                
        # Fallback: Suche nach einem Teil des Texts, der wie ein mathematischer Ausdruck aussieht
        math_pattern = r"([a-zA-Z0-9+\-*/^=()[\]{}<>≤≥≠√∫∂∑∏\s]+)"
        matches = re.findall(math_pattern, text)
        
        for match in matches:
            # Prüfe, ob der gefundene Ausdruck mathematische Operatoren enthält
            if any(op in match for op in "+-*/^=<>≤≥≠√∫∂∑∏"):
                # Mindestens 3 Zeichen und mindestens ein Operator
                if len(match.strip()) >= 3:
                    return match.strip()
                    
        return None
        
    def _extract_context_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extrahiert Kontextinformationen aus natürlichsprachlichem Text
        
        Args:
            text: Natürlichsprachlicher Text
            
        Returns:
            Extrahierte Kontextinformationen
        """
        context = {}
        text = text.lower()
        
        # Dringlichkeitsanalyse
        urgency_keywords = {
            "sofort": 0.9,
            "dringend": 0.8,
            "schnell": 0.7,
            "bald": 0.6,
            "irgendwann": 0.3,
            "wenn möglich": 0.4
        }
        
        for keyword, value in urgency_keywords.items():
            if keyword in text:
                context["urgency"] = value
                break
                
        # Prioritätsanalyse
        priority_keywords = {
            "höchste priorität": 0.9,
            "hohe priorität": 0.8,
            "wichtig": 0.7,
            "normal": 0.5,
            "niedrige priorität": 0.3,
            "unwichtig": 0.2
        }
        
        for keyword, value in priority_keywords.items():
            if keyword in text:
                context["priority"] = value
                break
                
        # Komplexitätsanalyse
        if "komplex" in text:
            context["expression_complexity"] = 0.8
        elif "einfach" in text:
            context["expression_complexity"] = 0.3
            
        # Topologische Faktoren
        if "topologisch" in text or "verformung" in text:
            context["topological_factors"] = 0.8
            context["process_context"] = "topologische Verformung"
        elif "symbolisch" in text:
            context["symbolic_depth"] = 0.8
            context["process_context"] = "symbolische Verarbeitung"
        elif "probabilistisch" in text or "wahrscheinlichkeit" in text:
            context["process_context"] = "probabilistische Analyse"
        elif "logik" in text or "logisch" in text:
            context["process_context"] = "babylonische Logik"
            
        return context


# Beispiel für die Verwendung
if __name__ == "__main__":
    # Konfiguriere Logging
    logging.basicConfig(level=logging.INFO)
    
    # Erstelle Integration
    integration = QLOGIKMPrimeIntegration()
    
    # Beispiel für symbolische Operation
    expression = "x^2 + 3*x - 5 = 0"
    
    # Operationskontext
    context = {
        "expression_complexity": 0.4,
        "symbolic_depth": 0.6,
        "system_capacity": 0.8,
        "priority": 0.7,
        "risk": 0.2,
        "benefit": 0.8,
        "urgency": 0.6,
        "operation_type": "solve"
    }
    
    # Führe Operation aus
    result = integration.execute_symbolic_operation(
        expression=expression,
        operation_context=context
    )
    
    print("Ergebnis:", result)
    
    # M-LINGUA-Integration
    mlingua = MLINGUAMPrimeIntegration(integration)
    
    # Natürlichsprachliche Anfrage
    nl_result = mlingua.process_natural_language(
        text="Löse die Gleichung x^2 + 3*x - 5 = 0 mit hoher Priorität"
    )
    
    print("M-LINGUA-Ergebnis:", nl_result)
