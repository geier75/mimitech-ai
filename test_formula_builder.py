#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für das FormulaBuilder-Modul der MPRIME Engine.
"""

import unittest
import sys
import os
from typing import Dict, Any, List

# Füge das Hauptverzeichnis zum Pfad hinzu, um die MISO-Module zu importieren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from miso.math.mprime.formula_builder import FormulaBuilder

class TestFormulaBuilder(unittest.TestCase):
    """
    Testklasse für das FormulaBuilder-Modul
    """
    
    def setUp(self):
        """
        Einrichtung für die Tests
        """
        self.formula_builder = FormulaBuilder()
    
    def test_initialization(self):
        """
        Testet die korrekte Initialisierung
        """
        self.assertIsNotNone(self.formula_builder)
        self.assertEqual(len(self.formula_builder.token_library), 5)  # Standardbibliotheken
        self.assertIn("basic", self.formula_builder.token_library)
        self.assertIn("calculus", self.formula_builder.token_library)
        self.assertIn("algebra", self.formula_builder.token_library)
        self.assertIn("statistics", self.formula_builder.token_library)
        self.assertIn("physics", self.formula_builder.token_library)
    
    def test_register_token_library(self):
        """
        Testet die Registrierung einer neuen Token-Bibliothek
        """
        # Erstelle eine neue Token-Bibliothek
        new_library = {
            "test_token1": {"symbol": "T1", "latex": "\\text{T1}", "description": "Test Token 1"},
            "test_token2": {"symbol": "T2", "latex": "\\text{T2}", "description": "Test Token 2"}
        }
        
        # Registriere die neue Bibliothek
        result = self.formula_builder.register_token_library("test_library", new_library)
        
        # Überprüfe, ob das Ergebnis True ist
        self.assertTrue(result)
        
        # Überprüfe, ob die Bibliothek korrekt registriert wurde
        self.assertIn("test_library", self.formula_builder.token_library)
        self.assertEqual(self.formula_builder.token_library["test_library"], new_library)
    
    def test_get_token_info(self):
        """
        Testet das Abrufen von Token-Informationen
        """
        # Teste das Abrufen eines vorhandenen Tokens
        token = "integral"
        result = self.formula_builder.get_token_info(token)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Token-Informationen korrekt abgerufen wurden
        self.assertEqual(result["token"], token)
        self.assertEqual(result["library"], "calculus")
        self.assertEqual(result["info"]["symbol"], "∫")
        self.assertEqual(result["info"]["latex"], "\\int")
        
        # Teste das Abrufen eines nicht vorhandenen Tokens
        non_existent_token = "non_existent_token"
        result = self.formula_builder.get_token_info(non_existent_token)
        
        # Überprüfe, ob das Ergebnis None ist
        self.assertIsNone(result)
    
    def test_tokenize_formula(self):
        """
        Testet die Tokenisierung einer Formel
        """
        # Teste die Tokenisierung einer einfachen Formel
        formula = "y = mx + b"
        result = self.formula_builder.tokenize_formula(formula)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Formel korrekt tokenisiert wurde
        self.assertEqual(result["original_formula"], formula)
        self.assertIsInstance(result["tokens"], list)
        
        # Überprüfe die einzelnen Tokens
        expected_tokens = ["y", "=", "m", "x", "+", "b"]
        self.assertEqual(len(result["tokens"]), len(expected_tokens))
        
        for i, token in enumerate(result["tokens"]):
            self.assertEqual(token["value"], expected_tokens[i])
    
    def test_build_formula_from_tokens(self):
        """
        Testet den Aufbau einer Formel aus Tokens
        """
        # Erstelle eine Liste von Tokens
        tokens = [
            {"type": "variable", "value": "y"},
            {"type": "operator", "value": "="},
            {"type": "variable", "value": "m"},
            {"type": "variable", "value": "x"},
            {"type": "operator", "value": "+"},
            {"type": "variable", "value": "b"}
        ]
        
        # Baue die Formel aus den Tokens
        result = self.formula_builder.build_formula_from_tokens(tokens)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Formel korrekt aufgebaut wurde
        self.assertEqual(result["formula"], "y = m x + b")
        self.assertEqual(result["tokens"], tokens)
    
    def test_build_formula_from_semantic_description(self):
        """
        Testet den Aufbau einer Formel aus einer semantischen Beschreibung
        """
        # Erstelle eine semantische Beschreibung
        description = "Die Gleichung einer Geraden mit Steigung m und y-Achsenabschnitt b"
        
        # Baue die Formel aus der semantischen Beschreibung
        result = self.formula_builder.build_formula_from_semantic_description(description)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Formel korrekt aufgebaut wurde
        self.assertEqual(result["description"], description)
        self.assertIsInstance(result["formula"], str)
        self.assertIn("y", result["formula"])
        self.assertIn("m", result["formula"])
        self.assertIn("x", result["formula"])
        self.assertIn("b", result["formula"])
    
    def test_substitute_variables(self):
        """
        Testet die Substitution von Variablen in einer Formel
        """
        # Erstelle eine Formel
        formula = "y = m*x + b"
        
        # Definiere die Variablensubstitutionen
        substitutions = {
            "m": 2,
            "b": 3
        }
        
        # Substituiere die Variablen
        result = self.formula_builder.substitute_variables(formula, substitutions)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Variablen korrekt substituiert wurden
        self.assertEqual(result["original_formula"], formula)
        self.assertEqual(result["substituted_formula"], "y = 2*x + 3")
        self.assertEqual(result["substitutions"], substitutions)
    
    def test_convert_to_latex(self):
        """
        Testet die Konvertierung einer Formel in LaTeX
        """
        # Erstelle eine Formel
        formula = "y = mx + b"
        
        # Konvertiere in LaTeX
        result = self.formula_builder.convert_to_latex(formula)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Formel korrekt in LaTeX konvertiert wurde
        self.assertEqual(result["original_formula"], formula)
        self.assertEqual(result["latex_formula"], "y = m x + b")
    
    def test_parse_natural_language(self):
        """
        Testet das Parsen natürlicher Sprache in eine Formel
        """
        # Erstelle eine natürlichsprachliche Beschreibung
        description = "Die Summe von a und b geteilt durch 2"
        
        # Parse die Beschreibung
        result = self.formula_builder.parse_natural_language(description)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Beschreibung korrekt geparst wurde
        self.assertEqual(result["description"], description)
        self.assertIsInstance(result["formula"], str)
        self.assertIn("a", result["formula"])
        self.assertIn("b", result["formula"])
        self.assertIn("+", result["formula"])
        self.assertIn("/", result["formula"])
        self.assertIn("2", result["formula"])
    
    def test_generate_formula_variations(self):
        """
        Testet die Generierung von Formelvariationen
        """
        # Erstelle eine Formel
        formula = "a + b"
        
        # Generiere Variationen
        result = self.formula_builder.generate_formula_variations(formula)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Variationen korrekt generiert wurden
        self.assertEqual(result["original_formula"], formula)
        self.assertIsInstance(result["variations"], list)
        self.assertGreater(len(result["variations"]), 0)
        
        # Überprüfe, ob die Variationen unterschiedlich sind
        self.assertIn("b + a", result["variations"])
    
    def test_evaluate_formula(self):
        """
        Testet die Auswertung einer Formel
        """
        # Erstelle eine Formel
        formula = "2*x + 3*y"
        
        # Definiere die Variablenwerte
        variables = {
            "x": 5,
            "y": 7
        }
        
        # Werte die Formel aus
        result = self.formula_builder.evaluate_formula(formula, variables)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Formel korrekt ausgewertet wurde
        self.assertEqual(result["formula"], formula)
        self.assertEqual(result["variables"], variables)
        self.assertEqual(result["result"], 2*5 + 3*7)  # 2*5 + 3*7 = 10 + 21 = 31
    
    def test_find_formula_type(self):
        """
        Testet die Erkennung des Formeltyps
        """
        # Teste verschiedene Formeltypen
        test_cases = [
            ("y = mx + b", "linear_equation"),
            ("a^2 + b^2 = c^2", "pythagorean_theorem"),
            ("E = mc^2", "energy_mass_equivalence"),
            ("F = ma", "newton_second_law")
        ]
        
        for formula, expected_type in test_cases:
            result = self.formula_builder.find_formula_type(formula)
            
            # Überprüfe, ob das Ergebnis ein Dictionary ist
            self.assertIsInstance(result, dict)
            
            # Überprüfe, ob der Formeltyp korrekt erkannt wurde
            self.assertEqual(result["formula"], formula)
            self.assertEqual(result["formula_type"], expected_type)

if __name__ == '__main__':
    unittest.main()
