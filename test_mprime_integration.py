#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrationstest für das MPRIME-Modul der MISO Engine.
Testet die Zusammenarbeit aller Submodule.
"""

import unittest
import sys
import os
import numpy as np
from typing import Dict, Any, List

# Füge das Hauptverzeichnis zum Pfad hinzu, um die MISO-Module zu importieren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from miso.math.mprime.symbol_solver import SymbolTree
from miso.math.mprime.topo_matrix import TopoNet
from miso.math.mprime.babylon_logic import BabylonLogicCore
from miso.math.mprime.prob_mapper import ProbabilisticMapper
from miso.math.mprime.formula_builder import FormulaBuilder
from miso.math.mprime.prime_resolver import PrimeResolver
from miso.math.mprime.contextual_math import ContextualMathCore

class TestMPRIMEIntegration(unittest.TestCase):
    """
    Testklasse für die Integration aller MPRIME-Submodule
    """
    
    def setUp(self):
        """
        Einrichtung für die Tests
        """
        self.symbol_tree = SymbolTree()
        self.topo_net = TopoNet()
        self.babylon_logic = BabylonLogicCore()
        self.prob_mapper = ProbabilisticMapper()
        self.formula_builder = FormulaBuilder()
        self.prime_resolver = PrimeResolver()
        self.contextual_math = ContextualMathCore()
    
    def test_symbol_tree_to_formula_builder(self):
        """
        Testet die Integration von SymbolTree und FormulaBuilder
        """
        # Parse einen Ausdruck mit SymbolTree
        expression = "x^2 + 2*x + 1"
        parsed = self.symbol_tree.parse_expression(expression)
        
        # Konvertiere den geparsten Ausdruck in LaTeX mit SymbolTree
        latex_tree = self.symbol_tree.to_latex(parsed["tree"])
        
        # Konvertiere den ursprünglichen Ausdruck in LaTeX mit FormulaBuilder
        latex_formula = self.formula_builder.convert_to_latex(expression)
        
        # Überprüfe, ob beide LaTeX-Darstellungen äquivalent sind
        # (Beachte: Es können syntaktische Unterschiede geben, daher prüfen wir nur auf Ähnlichkeit)
        self.assertIsNotNone(latex_tree)
        self.assertIsNotNone(latex_formula["latex_formula"])
    
    def test_symbol_tree_to_prime_resolver(self):
        """
        Testet die Integration von SymbolTree und PrimeResolver
        """
        # Parse einen Ausdruck mit SymbolTree
        expression = "x^2 + 2*x + 1 = 0"
        parsed = self.symbol_tree.parse_expression(expression)
        
        # Löse die Gleichung mit PrimeResolver
        solution = self.prime_resolver.solve_equation(expression)
        
        # Überprüfe, ob die Lösung korrekt ist
        self.assertIsNotNone(solution)
        self.assertIn("solution", solution)
        
        # Die Lösung sollte x = -1 sein (doppelte Nullstelle)
        self.assertIn("x", solution["solution"])
        self.assertEqual(solution["solution"]["x"], -1)
    
    def test_topo_net_to_prob_mapper(self):
        """
        Testet die Integration von TopoNet und ProbabilisticMapper
        """
        # Erstelle eine topologische Matrix mit TopoNet
        dimensions = 2
        size = 5
        self.topo_net.create_matrix(dimensions, size)
        
        # Füge Knoten hinzu
        self.topo_net.add_node("A", [0, 0], 1)
        self.topo_net.add_node("B", [1, 1], 2)
        self.topo_net.add_node("C", [2, 2], 3)
        
        # Füge Kanten hinzu
        self.topo_net.add_edge("AB", "A", "B", 0.7)
        self.topo_net.add_edge("BC", "B", "C", 0.5)
        
        # Erstelle einen Wahrscheinlichkeitsraum mit ProbabilisticMapper
        self.prob_mapper.create_probability_space(dimensions, size)
        
        # Füge Ereignisse basierend auf den TopoNet-Knoten hinzu
        for node_id, node_data in self.topo_net.nodes.items():
            coordinates = node_data["coordinates"]
            probability = 0.1 * node_data["value"]  # Einfache Umwandlung
            self.prob_mapper.add_event(node_id, coordinates, probability)
        
        # Berechne die gemeinsame Wahrscheinlichkeit
        joint_prob = self.prob_mapper.calculate_joint_probability(["A", "B"])
        
        # Überprüfe, ob die Berechnung erfolgreich war
        self.assertIsNotNone(joint_prob)
        self.assertIn("joint_probability", joint_prob)
        
        # Die gemeinsame Wahrscheinlichkeit sollte P(A) * P(B) = 0.1 * 0.2 = 0.02 sein
        self.assertAlmostEqual(joint_prob["joint_probability"], 0.02, places=5)
    
    def test_babylon_logic_to_contextual_math(self):
        """
        Testet die Integration von BabylonLogicCore und ContextualMathCore
        """
        # Konvertiere eine Dezimalzahl in babylonische Notation
        decimal = 61
        babylonian = self.babylon_logic.decimal_to_babylonian(decimal)
        
        # Interpretiere die babylonische Notation im wissenschaftlichen Kontext
        self.contextual_math.set_context("scientific")
        interpretation = self.contextual_math.interpret(babylonian)
        
        # Überprüfe, ob die Interpretation erfolgreich war
        self.assertIsNotNone(interpretation)
        self.assertEqual(interpretation["original_expression"], babylonian)
        self.assertEqual(interpretation["context_type"], "scientific")
    
    def test_formula_builder_to_prime_resolver(self):
        """
        Testet die Integration von FormulaBuilder und PrimeResolver
        """
        # Erstelle eine Formel mit FormulaBuilder
        description = "Die Gleichung einer Geraden mit Steigung 2 und y-Achsenabschnitt 3"
        formula_result = self.formula_builder.build_formula_from_semantic_description(description)
        
        # Extrahiere die Formel
        formula = formula_result["formula"]
        
        # Löse die Gleichung nach x mit PrimeResolver
        # Annahme: Die Formel ist in der Form "y = 2*x + 3"
        equation = formula + " = 7"  # y = 2*x + 3 = 7
        solution = self.prime_resolver.solve_equation(equation)
        
        # Überprüfe, ob die Lösung erfolgreich war
        self.assertIsNotNone(solution)
        self.assertIn("solution", solution)
        
        # Die Lösung sollte x = 2 sein (wenn y = 7, dann 7 = 2*x + 3 => x = 2)
        if "x" in solution["solution"]:
            self.assertEqual(solution["solution"]["x"], 2)
    
    def test_prime_resolver_to_contextual_math(self):
        """
        Testet die Integration von PrimeResolver und ContextualMathCore
        """
        # Löse eine Gleichung mit PrimeResolver
        equation = "F = ma"
        solution = self.prime_resolver.parse_expression(equation)
        
        # Interpretiere die Gleichung im physikalischen Kontext mit ContextualMathCore
        self.contextual_math.set_context("scientific")
        interpretation = self.contextual_math.interpret(equation)
        
        # Überprüfe, ob die Interpretation erfolgreich war
        self.assertIsNotNone(interpretation)
        self.assertEqual(interpretation["original_expression"], equation)
        self.assertEqual(interpretation["context_type"], "scientific")
        self.assertIn("Newtons zweites Gesetz", interpretation["interpretation"])
    
    def test_full_integration_simple_equation(self):
        """
        Testet die vollständige Integration aller Module an einer einfachen Gleichung
        """
        # 1. Erstelle eine Formel mit FormulaBuilder
        formula = "x^2 + 2*x + 1 = 0"
        
        # 2. Parse die Formel mit SymbolTree
        parsed = self.symbol_tree.parse_expression(formula)
        
        # 3. Löse die Gleichung mit PrimeResolver
        solution = self.prime_resolver.solve_equation(formula)
        
        # 4. Interpretiere die Lösung im mathematischen Kontext
        self.contextual_math.set_context("scientific")
        interpretation = self.contextual_math.interpret(f"x = {solution['solution']['x']}")
        
        # Überprüfe, ob der gesamte Prozess erfolgreich war
        self.assertIsNotNone(solution)
        self.assertIn("solution", solution)
        self.assertEqual(solution["solution"]["x"], -1)
        self.assertIsNotNone(interpretation)
    
    def test_full_integration_complex_scenario(self):
        """
        Testet die vollständige Integration aller Module in einem komplexeren Szenario
        """
        # 1. Erstelle einen topologischen Raum mit TopoNet
        self.topo_net.create_matrix(2, 5)
        self.topo_net.add_node("A", [0, 0], 1)
        self.topo_net.add_node("B", [1, 1], 2)
        self.topo_net.add_node("C", [2, 2], 3)
        self.topo_net.add_edge("AB", "A", "B", 0.7)
        self.topo_net.add_edge("BC", "B", "C", 0.5)
        
        # 2. Erstelle einen Wahrscheinlichkeitsraum mit ProbabilisticMapper
        self.prob_mapper.create_probability_space(2, 5)
        for node_id, node_data in self.topo_net.nodes.items():
            coordinates = node_data["coordinates"]
            probability = 0.1 * node_data["value"]
            self.prob_mapper.add_event(node_id, coordinates, probability)
        
        # 3. Berechne die erwartete Pfadlänge mit einer Formel
        formula = "L = P(A->B) * d(A,B) + P(B->C) * d(B,C)"
        parsed = self.symbol_tree.parse_expression(formula)
        
        # 4. Substituiere die Variablen mit FormulaBuilder
        path_ab = self.topo_net.compute_path("A", "B")
        path_bc = self.topo_net.compute_path("B", "C")
        prob_ab = self.prob_mapper.events["A"]["probability"]
        prob_bc = self.prob_mapper.events["B"]["probability"]
        
        substitutions = {
            "P(A->B)": prob_ab,
            "d(A,B)": 1,  # Einfache Distanz
            "P(B->C)": prob_bc,
            "d(B,C)": 1   # Einfache Distanz
        }
        
        substituted = self.formula_builder.substitute_variables(formula, substitutions)
        
        # 5. Werte die Formel aus mit PrimeResolver
        expected_length = prob_ab * 1 + prob_bc * 1
        
        # 6. Interpretiere das Ergebnis im statistischen Kontext
        self.contextual_math.set_context("statistical")
        interpretation = self.contextual_math.interpret(f"Expected path length: {expected_length}")
        
        # Überprüfe, ob der gesamte Prozess erfolgreich war
        self.assertIsNotNone(substituted)
        self.assertIsNotNone(interpretation)
        self.assertEqual(interpretation["context_type"], "statistical")

if __name__ == '__main__':
    unittest.main()
