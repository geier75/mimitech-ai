#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für das ContextualMathCore-Modul der MPRIME Engine.
"""

import unittest
import sys
import os
from typing import Dict, Any, List

# Füge das Hauptverzeichnis zum Pfad hinzu, um die MISO-Module zu importieren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from miso.math.mprime.contextual_math import ContextualMathCore

class TestContextualMathCore(unittest.TestCase):
    """
    Testklasse für das ContextualMathCore-Modul
    """
    
    def setUp(self):
        """
        Einrichtung für die Tests
        """
        self.contextual_math = ContextualMathCore()
    
    def test_initialization(self):
        """
        Testet die korrekte Initialisierung
        """
        self.assertIsNotNone(self.contextual_math)
        self.assertEqual(self.contextual_math.active_context, "scientific")
        self.assertIn("scientific", self.contextual_math.context_types)
        self.assertIn("engineering", self.contextual_math.context_types)
        self.assertIn("financial", self.contextual_math.context_types)
        self.assertIn("statistical", self.contextual_math.context_types)
        self.assertIn("geometric", self.contextual_math.context_types)
        self.assertIn("quantum", self.contextual_math.context_types)
        self.assertIn("logical", self.contextual_math.context_types)
        self.assertIn("educational", self.contextual_math.context_types)
    
    def test_set_context(self):
        """
        Testet das Setzen des aktiven Kontexts
        """
        # Teste das Setzen eines gültigen Kontexts
        context_type = "engineering"
        result = self.contextual_math.set_context(context_type)
        
        # Überprüfe, ob das Ergebnis True ist
        self.assertTrue(result)
        
        # Überprüfe, ob der Kontext korrekt gesetzt wurde
        self.assertEqual(self.contextual_math.active_context, context_type)
        
        # Teste das Setzen eines ungültigen Kontexts
        invalid_context = "invalid_context"
        result = self.contextual_math.set_context(invalid_context)
        
        # Überprüfe, ob das Ergebnis False ist
        self.assertFalse(result)
        
        # Überprüfe, ob der Kontext unverändert bleibt
        self.assertEqual(self.contextual_math.active_context, context_type)
    
    def test_interpret_scientific_context(self):
        """
        Testet die Interpretation im wissenschaftlichen Kontext
        """
        # Setze den wissenschaftlichen Kontext
        self.contextual_math.set_context("scientific")
        
        # Teste die Interpretation einer einfachen Formel
        expression = "E=mc^2"
        result = self.contextual_math.interpret(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Interpretation korrekt ist
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "scientific")
        self.assertIsNotNone(result["interpretation"])
        self.assertIn("Einsteins Energieäquivalenz", result["interpretation"])
    
    def test_interpret_engineering_context(self):
        """
        Testet die Interpretation im technischen Kontext
        """
        # Setze den technischen Kontext
        self.contextual_math.set_context("engineering")
        
        # Teste die Interpretation einer einfachen Formel
        expression = "P=VI"
        result = self.contextual_math.interpret(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Interpretation korrekt ist
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "engineering")
        self.assertIsNotNone(result["interpretation"])
        self.assertIn("Elektrische Leistung", result["interpretation"])
    
    def test_interpret_financial_context(self):
        """
        Testet die Interpretation im finanziellen Kontext
        """
        # Setze den finanziellen Kontext
        self.contextual_math.set_context("financial")
        
        # Teste die Interpretation einer einfachen Formel
        expression = "FV=PV(1+r)^t"
        result = self.contextual_math.interpret(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Interpretation korrekt ist
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "financial")
        self.assertIsNotNone(result["interpretation"])
        self.assertIn("Zinseszinsformel", result["interpretation"])
    
    def test_interpret_statistical_context(self):
        """
        Testet die Interpretation im statistischen Kontext
        """
        # Setze den statistischen Kontext
        self.contextual_math.set_context("statistical")
        
        # Teste die Interpretation einer einfachen Formel
        expression = "mu and sigma"
        result = self.contextual_math.interpret(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Interpretation korrekt ist
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "statistical")
        self.assertIsNotNone(result["interpretation"])
        self.assertIn("Normalverteilung", result["interpretation"])
    
    def test_interpret_geometric_context(self):
        """
        Testet die Interpretation im geometrischen Kontext
        """
        # Setze den geometrischen Kontext
        self.contextual_math.set_context("geometric")
        
        # Teste die Interpretation einer einfachen Formel
        expression = "A=pi*r^2"
        result = self.contextual_math.interpret(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Interpretation korrekt ist
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "geometric")
        self.assertIsNotNone(result["interpretation"])
        self.assertIn("Kreisfläche", result["interpretation"])
    
    def test_interpret_quantum_context(self):
        """
        Testet die Interpretation im quantenmechanischen Kontext
        """
        # Setze den quantenmechanischen Kontext
        self.contextual_math.set_context("quantum")
        
        # Teste die Interpretation einer einfachen Formel
        expression = "psi"
        result = self.contextual_math.interpret(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Interpretation korrekt ist
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "quantum")
        self.assertIsNotNone(result["interpretation"])
        self.assertIn("Wellenfunktion", result["interpretation"])
    
    def test_interpret_logical_context(self):
        """
        Testet die Interpretation im logischen Kontext
        """
        # Setze den logischen Kontext
        self.contextual_math.set_context("logical")
        
        # Teste die Interpretation einer einfachen Formel
        expression = "A AND B"
        result = self.contextual_math.interpret(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Interpretation korrekt ist
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "logical")
        self.assertIsNotNone(result["interpretation"])
        self.assertIn("Logisches UND", result["interpretation"])
    
    def test_interpret_educational_context(self):
        """
        Testet die Interpretation im pädagogischen Kontext
        """
        # Setze den pädagogischen Kontext
        self.contextual_math.set_context("educational")
        
        # Teste die Interpretation einer einfachen Formel
        expression = "a + b"
        result = self.contextual_math.interpret(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Interpretation korrekt ist
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "educational")
        self.assertIsNotNone(result["interpretation"])
        self.assertIn("Addition", result["interpretation"])
    
    def test_interpret_with_context_parameter(self):
        """
        Testet die Interpretation mit einem expliziten Kontextparameter
        """
        # Setze den wissenschaftlichen Kontext
        self.contextual_math.set_context("scientific")
        
        # Teste die Interpretation mit einem expliziten Kontext
        expression = "A=pi*r^2"
        context = {"type": "geometric"}
        result = self.contextual_math.interpret(expression, context)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Interpretation dem expliziten Kontext entspricht
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "geometric")
        self.assertIsNotNone(result["interpretation"])
        self.assertIn("Kreisfläche", result["interpretation"])
    
    def test_apply_context_transformation_scientific(self):
        """
        Testet die Anwendung einer Transformation im wissenschaftlichen Kontext
        """
        # Setze den wissenschaftlichen Kontext
        self.contextual_math.set_context("scientific")
        
        # Teste die Transformation einer einfachen Formel
        expression = "c"  # Lichtgeschwindigkeit
        result = self.contextual_math.apply_context_transformation(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Transformation korrekt ist
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "scientific")
        self.assertIsNotNone(result["transformed_expression"])
        self.assertEqual(result["transformation_type"], "scientific_substitution")
        
        # Die Lichtgeschwindigkeit sollte durch ihren Wert ersetzt werden
        self.assertEqual(result["transformed_expression"], "299792458")
    
    def test_apply_context_transformation_engineering(self):
        """
        Testet die Anwendung einer Transformation im technischen Kontext
        """
        # Setze den technischen Kontext
        self.contextual_math.set_context("engineering")
        
        # Teste die Transformation einer einfachen Formel
        expression = "g"  # Erdbeschleunigung
        result = self.contextual_math.apply_context_transformation(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Transformation korrekt ist
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "engineering")
        self.assertIsNotNone(result["transformed_expression"])
        self.assertEqual(result["transformation_type"], "engineering_substitution")
        
        # Die Erdbeschleunigung sollte durch ihren Wert ersetzt werden
        self.assertEqual(result["transformed_expression"], "9.80665")
    
    def test_apply_context_transformation_financial(self):
        """
        Testet die Anwendung einer Transformation im finanziellen Kontext
        """
        # Setze den finanziellen Kontext
        self.contextual_math.set_context("financial")
        
        # Teste die Transformation einer einfachen Formel
        expression = "NPV"  # Nettogegenwartswert
        result = self.contextual_math.apply_context_transformation(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Transformation korrekt ist
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "financial")
        self.assertIsNotNone(result["transformed_expression"])
        self.assertEqual(result["transformation_type"], "financial_substitution")
        
        # NPV sollte durch "Net Present Value" ersetzt werden
        self.assertEqual(result["transformed_expression"], "Net Present Value")
    
    def test_apply_context_transformation_with_context_parameter(self):
        """
        Testet die Anwendung einer Transformation mit einem expliziten Kontextparameter
        """
        # Setze den wissenschaftlichen Kontext
        self.contextual_math.set_context("scientific")
        
        # Teste die Transformation mit einem expliziten Kontext
        expression = "g"  # Erdbeschleunigung
        context_type = "engineering"
        result = self.contextual_math.apply_context_transformation(expression, context_type)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Transformation dem expliziten Kontext entspricht
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["context_type"], "engineering")
        self.assertIsNotNone(result["transformed_expression"])
        self.assertEqual(result["transformation_type"], "engineering_substitution")
        
        # Die Erdbeschleunigung sollte durch ihren Wert ersetzt werden
        self.assertEqual(result["transformed_expression"], "9.80665")

if __name__ == '__main__':
    unittest.main()
