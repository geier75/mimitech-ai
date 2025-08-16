#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für das SymbolTree-Modul der MPRIME Engine.
"""

import unittest
import sys
import os
from typing import Dict, Any, List

# Füge das Hauptverzeichnis zum Pfad hinzu, um die MISO-Module zu importieren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from miso.math.mprime.symbol_solver import SymbolTree

class TestSymbolTree(unittest.TestCase):
    """
    Testklasse für das SymbolTree-Modul
    """
    
    def setUp(self):
        """
        Einrichtung für die Tests
        """
        self.symbol_tree = SymbolTree()
    
    def test_initialization(self):
        """
        Testet die korrekte Initialisierung
        """
        self.assertIsNotNone(self.symbol_tree)
        self.assertEqual(self.symbol_tree.max_depth, 100)
        self.assertEqual(self.symbol_tree.operator_precedence['+'], 1)
        self.assertEqual(self.symbol_tree.operator_precedence['-'], 1)
        self.assertEqual(self.symbol_tree.operator_precedence['*'], 2)
        self.assertEqual(self.symbol_tree.operator_precedence['/'], 2)
        self.assertEqual(self.symbol_tree.operator_precedence['^'], 3)
    
    def test_parse_expression(self):
        """
        Testet das Parsen eines einfachen Ausdrucks
        """
        expression = "2 + 3 * 4"
        result = self.symbol_tree.parse_expression(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob der normalisierte Ausdruck korrekt ist
        self.assertEqual(result["normalized_expression"], "2+3*4")
        
        # Überprüfe, ob der Baum korrekt erstellt wurde
        self.assertEqual(result["tree"]["type"], "operator")
        self.assertEqual(result["tree"]["value"], "+")
        self.assertEqual(result["tree"]["left"]["type"], "number")
        self.assertEqual(result["tree"]["left"]["value"], "2")
        self.assertEqual(result["tree"]["right"]["type"], "operator")
        self.assertEqual(result["tree"]["right"]["value"], "*")
        self.assertEqual(result["tree"]["right"]["left"]["type"], "number")
        self.assertEqual(result["tree"]["right"]["left"]["value"], "3")
        self.assertEqual(result["tree"]["right"]["right"]["type"], "number")
        self.assertEqual(result["tree"]["right"]["right"]["value"], "4")
    
    def test_complex_expression(self):
        """
        Testet das Parsen eines komplexeren Ausdrucks mit Klammern
        """
        expression = "(x + y) * (z - 5) / 2^n"
        result = self.symbol_tree.parse_expression(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob der normalisierte Ausdruck korrekt ist
        self.assertEqual(result["normalized_expression"], "(x+y)*(z-5)/2^n")
        
        # Überprüfe, ob die Variablen korrekt erkannt wurden
        self.assertIn("x", result["variables"])
        self.assertIn("y", result["variables"])
        self.assertIn("z", result["variables"])
        self.assertIn("n", result["variables"])
    
    def test_evaluate_expression(self):
        """
        Testet die Auswertung eines Ausdrucks mit Variablenwerten
        """
        expression = "x + y * z"
        variables = {"x": 5, "y": 3, "z": 2}
        
        # Parse den Ausdruck
        parsed = self.symbol_tree.parse_expression(expression)
        
        # Werte den Ausdruck aus
        result = self.symbol_tree.evaluate_expression(parsed["tree"], variables)
        
        # Überprüfe das Ergebnis: 5 + 3 * 2 = 11
        self.assertEqual(result, 11)
    
    def test_derivative(self):
        """
        Testet die Berechnung der Ableitung eines einfachen Ausdrucks
        """
        expression = "x^2 + 2*x + 1"
        variable = "x"
        
        # Parse den Ausdruck
        parsed = self.symbol_tree.parse_expression(expression)
        
        # Berechne die Ableitung
        derivative = self.symbol_tree.compute_derivative(parsed["tree"], variable)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(derivative, dict)
        
        # Überprüfe, ob die Ableitung korrekt berechnet wurde
        # Die Ableitung von x^2 + 2*x + 1 nach x ist 2*x + 2
        evaluated_at_1 = self.symbol_tree.evaluate_expression(derivative, {"x": 1})
        self.assertEqual(evaluated_at_1, 4)  # 2*1 + 2 = 4
        
        evaluated_at_2 = self.symbol_tree.evaluate_expression(derivative, {"x": 2})
        self.assertEqual(evaluated_at_2, 6)  # 2*2 + 2 = 6
    
    def test_simplify_expression(self):
        """
        Testet die Vereinfachung eines Ausdrucks
        """
        expression = "x + 0"
        
        # Parse den Ausdruck
        parsed = self.symbol_tree.parse_expression(expression)
        
        # Vereinfache den Ausdruck
        simplified = self.symbol_tree.simplify_expression(parsed["tree"])
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(simplified, dict)
        
        # Überprüfe, ob der Ausdruck korrekt vereinfacht wurde
        # x + 0 sollte zu x vereinfacht werden
        self.assertEqual(simplified["type"], "variable")
        self.assertEqual(simplified["value"], "x")
    
    def test_to_latex(self):
        """
        Testet die Konvertierung eines Ausdrucks in LaTeX
        """
        expression = "sqrt(x^2 + y^2)"
        
        # Parse den Ausdruck
        parsed = self.symbol_tree.parse_expression(expression)
        
        # Konvertiere in LaTeX
        latex = self.symbol_tree.to_latex(parsed["tree"])
        
        # Überprüfe, ob das Ergebnis ein String ist
        self.assertIsInstance(latex, str)
        
        # Überprüfe, ob die LaTeX-Darstellung korrekt ist
        expected_latex = r"\sqrt{x^{2} + y^{2}}"
        self.assertEqual(latex, expected_latex)

if __name__ == '__main__':
    unittest.main()
