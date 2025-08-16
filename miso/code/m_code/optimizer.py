#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Optimizer

Dieses Modul implementiert den Optimizer für die M-CODE Programmiersprache.
Der Optimizer führt verschiedene Optimierungen auf dem Syntaxbaum durch.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import ast
import inspect
import types
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.optimizer")

# Import von internen Modulen
from .syntax import MCodeSyntaxTree, MCodeNode, MCodeModule, MCodeBinaryOp, MCodeUnaryOp
from .syntax import MCodeLiteral, MCodeIdentifier, MCodeCall, MCodeAssignment


class OptimizationPass:
    """Basisklasse für Optimierungspässe"""
    
    def __init__(self, name: str):
        """
        Initialisiert einen neuen Optimierungspass.
        
        Args:
            name: Name des Optimierungspasses
        """
        self.name = name
        self.statistics = {
            "applied": 0,
            "skipped": 0
        }
    
    def apply(self, node: MCodeNode) -> MCodeNode:
        """
        Wendet den Optimierungspass auf einen Knoten an.
        
        Args:
            node: Zu optimierender Knoten
            
        Returns:
            Optimierter Knoten
        """
        # Basisimplementierung: Keine Optimierung
        return node
    
    def reset_statistics(self) -> None:
        """Setzt die Statistiken zurück"""
        self.statistics = {
            "applied": 0,
            "skipped": 0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gibt die Statistiken zurück"""
        return self.statistics


class ConstantFoldingPass(OptimizationPass):
    """Optimierungspass für Constant Folding"""
    
    def __init__(self):
        """Initialisiert einen neuen Constant Folding Pass"""
        super().__init__("constant_folding")
    
    def apply(self, node: MCodeNode) -> MCodeNode:
        """
        Wendet Constant Folding auf einen Knoten an.
        
        Args:
            node: Zu optimierender Knoten
            
        Returns:
            Optimierter Knoten
        """
        # Optimiere binäre Operationen mit konstanten Operanden
        if isinstance(node, MCodeBinaryOp):
            if isinstance(node.left, MCodeLiteral) and isinstance(node.right, MCodeLiteral):
                # Berechne konstanten Ausdruck
                try:
                    result = self._evaluate_binary_op(node.left.value, node.operator, node.right.value)
                    
                    # Erstelle neuen Literal-Knoten
                    optimized_node = MCodeLiteral(
                        line=node.line,
                        column=node.column,
                        value=result,
                        literal_type=type(result).__name__
                    )
                    
                    self.statistics["applied"] += 1
                    return optimized_node
                except Exception as e:
                    logger.warning(f"Fehler beim Constant Folding: {e}")
                    self.statistics["skipped"] += 1
        
        # Optimiere unäre Operationen mit konstanten Operanden
        elif isinstance(node, MCodeUnaryOp):
            if isinstance(node.operand, MCodeLiteral):
                # Berechne konstanten Ausdruck
                try:
                    result = self._evaluate_unary_op(node.operator, node.operand.value)
                    
                    # Erstelle neuen Literal-Knoten
                    optimized_node = MCodeLiteral(
                        line=node.line,
                        column=node.column,
                        value=result,
                        literal_type=type(result).__name__
                    )
                    
                    self.statistics["applied"] += 1
                    return optimized_node
                except Exception as e:
                    logger.warning(f"Fehler beim Constant Folding: {e}")
                    self.statistics["skipped"] += 1
        
        # Keine Optimierung möglich
        self.statistics["skipped"] += 1
        return node
    
    def _evaluate_binary_op(self, left: Any, operator: str, right: Any) -> Any:
        """Evaluiert einen binären Operator mit konstanten Operanden"""
        if operator == "+":
            return left + right
        elif operator == "-":
            return left - right
        elif operator == "*":
            return left * right
        elif operator == "/":
            return left / right
        elif operator == "//":
            return left // right
        elif operator == "%":
            return left % right
        elif operator == "**":
            return left ** right
        elif operator == "<<":
            return left << right
        elif operator == ">>":
            return left >> right
        elif operator == "&":
            return left & right
        elif operator == "|":
            return left | right
        elif operator == "^":
            return left ^ right
        elif operator == "and":
            return left and right
        elif operator == "or":
            return left or right
        elif operator == "==":
            return left == right
        elif operator == "!=":
            return left != right
        elif operator == "<":
            return left < right
        elif operator == ">":
            return left > right
        elif operator == "<=":
            return left <= right
        elif operator == ">=":
            return left >= right
        else:
            raise ValueError(f"Unbekannter binärer Operator: {operator}")
    
    def _evaluate_unary_op(self, operator: str, operand: Any) -> Any:
        """Evaluiert einen unären Operator mit konstantem Operanden"""
        if operator == "+":
            return +operand
        elif operator == "-":
            return -operand
        elif operator == "not":
            return not operand
        elif operator == "~":
            return ~operand
        else:
            raise ValueError(f"Unbekannter unärer Operator: {operator}")


class DeadCodeEliminationPass(OptimizationPass):
    """Optimierungspass für Dead Code Elimination"""
    
    def __init__(self):
        """Initialisiert einen neuen Dead Code Elimination Pass"""
        super().__init__("dead_code_elimination")
        self.reachable_nodes = set()
    
    def apply(self, node: MCodeNode) -> MCodeNode:
        """
        Wendet Dead Code Elimination auf einen Knoten an.
        
        Args:
            node: Zu optimierender Knoten
            
        Returns:
            Optimierter Knoten
        """
        # In einer vollständigen Implementierung würde hier eine komplexe Analyse erfolgen
        # Für diese Beispielimplementierung geben wir den Knoten unverändert zurück
        
        self.statistics["skipped"] += 1
        return node


class CommonSubexpressionEliminationPass(OptimizationPass):
    """Optimierungspass für Common Subexpression Elimination"""
    
    def __init__(self):
        """Initialisiert einen neuen Common Subexpression Elimination Pass"""
        super().__init__("common_subexpression_elimination")
        self.expressions = {}
    
    def apply(self, node: MCodeNode) -> MCodeNode:
        """
        Wendet Common Subexpression Elimination auf einen Knoten an.
        
        Args:
            node: Zu optimierender Knoten
            
        Returns:
            Optimierter Knoten
        """
        # In einer vollständigen Implementierung würde hier eine komplexe Analyse erfolgen
        # Für diese Beispielimplementierung geben wir den Knoten unverändert zurück
        
        self.statistics["skipped"] += 1
        return node


class TensorOptimizationPass(OptimizationPass):
    """Optimierungspass für Tensor-Operationen"""
    
    def __init__(self):
        """Initialisiert einen neuen Tensor Optimization Pass"""
        super().__init__("tensor_optimization")
    
    def apply(self, node: MCodeNode) -> MCodeNode:
        """
        Wendet Tensor-Optimierungen auf einen Knoten an.
        
        Args:
            node: Zu optimierender Knoten
            
        Returns:
            Optimierter Knoten
        """
        # In einer vollständigen Implementierung würde hier eine komplexe Analyse erfolgen
        # Für diese Beispielimplementierung geben wir den Knoten unverändert zurück
        
        self.statistics["skipped"] += 1
        return node


class ParallelizationPass(OptimizationPass):
    """Optimierungspass für automatische Parallelisierung"""
    
    def __init__(self):
        """Initialisiert einen neuen Parallelization Pass"""
        super().__init__("parallelization")
    
    def apply(self, node: MCodeNode) -> MCodeNode:
        """
        Wendet automatische Parallelisierung auf einen Knoten an.
        
        Args:
            node: Zu optimierender Knoten
            
        Returns:
            Optimierter Knoten
        """
        # In einer vollständigen Implementierung würde hier eine komplexe Analyse erfolgen
        # Für diese Beispielimplementierung geben wir den Knoten unverändert zurück
        
        self.statistics["skipped"] += 1
        return node


class MCodeOptimizer:
    """Optimizer für die M-CODE Programmiersprache"""
    
    def __init__(self, optimization_level: int = 2):
        """
        Initialisiert einen neuen M-CODE Optimizer.
        
        Args:
            optimization_level: Optimierungsstufe (0-3)
        """
        self.optimization_level = min(3, max(0, optimization_level))
        self.passes = []
        
        # Initialisiere Optimierungspässe basierend auf der Optimierungsstufe
        self._initialize_passes()
        
        logger.info(f"M-CODE Optimizer initialisiert mit Optimierungsstufe {self.optimization_level}")
    
    def _initialize_passes(self) -> None:
        """Initialisiert Optimierungspässe basierend auf der Optimierungsstufe"""
        # Stufe 0: Keine Optimierungen
        if self.optimization_level == 0:
            return
        
        # Stufe 1: Grundlegende Optimierungen
        self.passes.append(ConstantFoldingPass())
        
        # Stufe 2: Erweiterte Optimierungen
        if self.optimization_level >= 2:
            self.passes.append(DeadCodeEliminationPass())
            self.passes.append(CommonSubexpressionEliminationPass())
        
        # Stufe 3: Fortgeschrittene Optimierungen
        if self.optimization_level >= 3:
            self.passes.append(TensorOptimizationPass())
            self.passes.append(ParallelizationPass())
    
    def optimize(self, syntax_tree: MCodeSyntaxTree) -> MCodeSyntaxTree:
        """
        Optimiert einen M-CODE Syntaxbaum.
        
        Args:
            syntax_tree: Zu optimierender Syntaxbaum
            
        Returns:
            Optimierter Syntaxbaum
        """
        if self.optimization_level == 0:
            logger.info("Optimierung übersprungen (Optimierungsstufe 0)")
            return syntax_tree
        
        # Optimiere den Syntaxbaum
        optimized_root = self._optimize_node(syntax_tree.root)
        
        # Erstelle neuen Syntaxbaum
        optimized_tree = MCodeSyntaxTree(optimized_root)
        
        # Protokolliere Statistiken
        self._log_statistics()
        
        return optimized_tree
    
    def _optimize_node(self, node: MCodeNode) -> MCodeNode:
        """
        Optimiert einen einzelnen Knoten rekursiv.
        
        Args:
            node: Zu optimierender Knoten
            
        Returns:
            Optimierter Knoten
        """
        # Optimiere Kindknoten rekursiv
        if isinstance(node, MCodeModule):
            # Optimiere Body
            node.body = [self._optimize_node(child) for child in node.body]
        elif isinstance(node, MCodeBinaryOp):
            # Optimiere linken und rechten Operanden
            node.left = self._optimize_node(node.left)
            node.right = self._optimize_node(node.right)
        elif isinstance(node, MCodeUnaryOp):
            # Optimiere Operanden
            node.operand = self._optimize_node(node.operand)
        elif isinstance(node, MCodeCall):
            # Optimiere Funktion und Argumente
            node.func = self._optimize_node(node.func)
            node.args = [self._optimize_node(arg) for arg in node.args]
            node.keywords = {k: self._optimize_node(v) for k, v in node.keywords.items()}
        
        # Wende Optimierungspässe auf den Knoten an
        for optimization_pass in self.passes:
            node = optimization_pass.apply(node)
        
        return node
    
    def _log_statistics(self) -> None:
        """Protokolliert Statistiken der Optimierungspässe"""
        for optimization_pass in self.passes:
            stats = optimization_pass.get_statistics()
            logger.info(f"Optimierungspass '{optimization_pass.name}': {stats['applied']} angewendet, {stats['skipped']} übersprungen")


def optimize_m_code(syntax_tree: MCodeSyntaxTree, optimization_level: int = 2) -> MCodeSyntaxTree:
    """
    Optimiert einen M-CODE Syntaxbaum.
    
    Args:
        syntax_tree: Zu optimierender Syntaxbaum
        optimization_level: Optimierungsstufe (0-3)
        
    Returns:
        Optimierter Syntaxbaum
    """
    optimizer = MCodeOptimizer(optimization_level)
    return optimizer.optimize(syntax_tree)
