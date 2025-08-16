#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für das PrimeResolver-Modul der MPRIME Engine.
"""

import unittest
import sys
import os
from typing import Dict, Any, List

# Füge das Hauptverzeichnis zum Pfad hinzu, um die MISO-Module zu importieren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from miso.math.mprime.prime_resolver import PrimeResolver

class TestPrimeResolver(unittest.TestCase):
    """
    Testklasse für das PrimeResolver-Modul
    """
    
    def setUp(self):
        """
        Einrichtung für die Tests
        """
        self.prime_resolver = PrimeResolver()
    
    def test_initialization(self):
        """
        Testet die korrekte Initialisierung
        """
        self.assertIsNotNone(self.prime_resolver)
        self.assertEqual(self.prime_resolver.max_iterations, 100)
        self.assertEqual(self.prime_resolver.precision, 1e-10)
        self.assertEqual(len(self.prime_resolver.solution_strategies), 5)
    
    def test_parse_expression(self):
        """
        Testet das Parsen eines Ausdrucks
        """
        # Teste das Parsen eines einfachen Ausdrucks
        expression = "2*x + 3*y = 10"
        result = self.prime_resolver.parse_expression(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob der Ausdruck korrekt geparst wurde
        self.assertEqual(result["original_expression"], expression)
        self.assertIsInstance(result["parsed_expression"], dict)
        self.assertIn("left", result["parsed_expression"])
        self.assertIn("right", result["parsed_expression"])
        self.assertEqual(result["parsed_expression"]["right"]["value"], "10")
        
        # Überprüfe, ob die Variablen korrekt erkannt wurden
        self.assertIn("variables", result)
        self.assertIn("x", result["variables"])
        self.assertIn("y", result["variables"])
    
    def test_simplify_expression(self):
        """
        Testet die Vereinfachung eines Ausdrucks
        """
        # Teste die Vereinfachung eines einfachen Ausdrucks
        expression = "x + 0"
        result = self.prime_resolver.simplify_expression(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob der Ausdruck korrekt vereinfacht wurde
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["simplified_expression"], "x")
        
        # Teste die Vereinfachung eines komplexeren Ausdrucks
        expression = "2*x + 3*x"
        result = self.prime_resolver.simplify_expression(expression)
        
        # Überprüfe, ob der Ausdruck korrekt vereinfacht wurde
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["simplified_expression"], "5*x")
    
    def test_solve_equation(self):
        """
        Testet das Lösen einer Gleichung
        """
        # Teste das Lösen einer einfachen Gleichung
        equation = "2*x = 10"
        result = self.prime_resolver.solve_equation(equation)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Gleichung korrekt gelöst wurde
        self.assertEqual(result["original_equation"], equation)
        self.assertIsInstance(result["solution"], dict)
        self.assertIn("x", result["solution"])
        self.assertEqual(result["solution"]["x"], 5)
        
        # Teste das Lösen einer komplexeren Gleichung
        equation = "2*x + 3 = 7"
        result = self.prime_resolver.solve_equation(equation)
        
        # Überprüfe, ob die Gleichung korrekt gelöst wurde
        self.assertEqual(result["original_equation"], equation)
        self.assertIsInstance(result["solution"], dict)
        self.assertIn("x", result["solution"])
        self.assertEqual(result["solution"]["x"], 2)
    
    def test_solve_equation_system(self):
        """
        Testet das Lösen eines Gleichungssystems
        """
        # Teste das Lösen eines einfachen Gleichungssystems
        equations = [
            "x + y = 10",
            "2*x - y = 5"
        ]
        result = self.prime_resolver.solve_equation_system(equations)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob das Gleichungssystem korrekt gelöst wurde
        self.assertEqual(result["original_equations"], equations)
        self.assertIsInstance(result["solution"], dict)
        self.assertIn("x", result["solution"])
        self.assertIn("y", result["solution"])
        self.assertEqual(result["solution"]["x"], 5)
        self.assertEqual(result["solution"]["y"], 5)
    
    def test_find_roots(self):
        """
        Testet das Finden von Nullstellen
        """
        # Teste das Finden von Nullstellen einer einfachen Funktion
        expression = "x^2 - 4"
        result = self.prime_resolver.find_roots(expression)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Nullstellen korrekt gefunden wurden
        self.assertEqual(result["expression"], expression)
        self.assertIsInstance(result["roots"], list)
        self.assertEqual(len(result["roots"]), 2)
        
        # Die Nullstellen sollten -2 und 2 sein
        roots = sorted(result["roots"])
        self.assertAlmostEqual(roots[0], -2, places=5)
        self.assertAlmostEqual(roots[1], 2, places=5)
    
    def test_compute_derivative(self):
        """
        Testet die Berechnung der Ableitung
        """
        # Teste die Berechnung der Ableitung einer einfachen Funktion
        expression = "x^2"
        variable = "x"
        result = self.prime_resolver.compute_derivative(expression, variable)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Ableitung korrekt berechnet wurde
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["variable"], variable)
        self.assertEqual(result["derivative"], "2*x")
    
    def test_compute_integral(self):
        """
        Testet die Berechnung des Integrals
        """
        # Teste die Berechnung des Integrals einer einfachen Funktion
        expression = "2*x"
        variable = "x"
        result = self.prime_resolver.compute_integral(expression, variable)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob das Integral korrekt berechnet wurde
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["variable"], variable)
        self.assertEqual(result["integral"], "x^2 + C")
    
    def test_compute_limit(self):
        """
        Testet die Berechnung des Grenzwerts
        """
        # Teste die Berechnung des Grenzwerts einer einfachen Funktion
        expression = "sin(x)/x"
        variable = "x"
        point = 0
        result = self.prime_resolver.compute_limit(expression, variable, point)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob der Grenzwert korrekt berechnet wurde
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["variable"], variable)
        self.assertEqual(result["point"], point)
        self.assertAlmostEqual(result["limit"], 1, places=5)
    
    def test_compute_series_expansion(self):
        """
        Testet die Berechnung der Reihenentwicklung
        """
        # Teste die Berechnung der Reihenentwicklung einer einfachen Funktion
        expression = "exp(x)"
        variable = "x"
        point = 0
        order = 3
        result = self.prime_resolver.compute_series_expansion(expression, variable, point, order)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Reihenentwicklung korrekt berechnet wurde
        self.assertEqual(result["original_expression"], expression)
        self.assertEqual(result["variable"], variable)
        self.assertEqual(result["point"], point)
        self.assertEqual(result["order"], order)
        
        # Die Reihenentwicklung von exp(x) um x=0 bis zur Ordnung 3 ist 1 + x + x^2/2 + x^3/6
        expected_terms = ["1", "x", "x^2/2", "x^3/6"]
        self.assertEqual(result["terms"], expected_terms)
    
    def test_find_extrema(self):
        """
        Testet das Finden von Extrema
        """
        # Teste das Finden von Extrema einer einfachen Funktion
        expression = "x^2 - 4*x + 4"
        variable = "x"
        result = self.prime_resolver.find_extrema(expression, variable)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Extrema korrekt gefunden wurden
        self.assertEqual(result["expression"], expression)
        self.assertEqual(result["variable"], variable)
        self.assertIsInstance(result["extrema"], list)
        
        # Die Funktion hat ein Minimum bei x=2
        self.assertEqual(len(result["extrema"]), 1)
        extremum = result["extrema"][0]
        self.assertEqual(extremum["type"], "minimum")
        self.assertEqual(extremum["point"], 2)
        self.assertEqual(extremum["value"], 0)
    
    def test_solve_differential_equation(self):
        """
        Testet das Lösen einer Differentialgleichung
        """
        # Teste das Lösen einer einfachen Differentialgleichung: y' = y
        equation = "diff(y, x) = y"
        variable = "y"
        independent_variable = "x"
        result = self.prime_resolver.solve_differential_equation(equation, variable, independent_variable)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Differentialgleichung korrekt gelöst wurde
        self.assertEqual(result["original_equation"], equation)
        self.assertEqual(result["variable"], variable)
        self.assertEqual(result["independent_variable"], independent_variable)
        
        # Die Lösung sollte y = C*e^x sein
        self.assertEqual(result["solution"], "C*exp(x)")
    
    def test_generate_solution_steps(self):
        """
        Testet die Generierung von Lösungsschritten
        """
        # Teste die Generierung von Lösungsschritten für eine einfache Gleichung
        equation = "2*x + 3 = 7"
        result = self.prime_resolver.generate_solution_steps(equation)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Lösungsschritte korrekt generiert wurden
        self.assertEqual(result["original_equation"], equation)
        self.assertIsInstance(result["steps"], list)
        self.assertGreater(len(result["steps"]), 0)
        
        # Überprüfe, ob die Lösung korrekt ist
        final_step = result["steps"][-1]
        self.assertEqual(final_step["equation"], "x = 2")
    
    def test_check_solution(self):
        """
        Testet die Überprüfung einer Lösung
        """
        # Teste die Überprüfung einer korrekten Lösung
        equation = "2*x + 3 = 7"
        solution = {"x": 2}
        result = self.prime_resolver.check_solution(equation, solution)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Lösung korrekt überprüft wurde
        self.assertEqual(result["equation"], equation)
        self.assertEqual(result["solution"], solution)
        self.assertTrue(result["is_correct"])
        
        # Teste die Überprüfung einer falschen Lösung
        solution = {"x": 3}
        result = self.prime_resolver.check_solution(equation, solution)
        
        # Überprüfe, ob die Lösung korrekt überprüft wurde
        self.assertEqual(result["equation"], equation)
        self.assertEqual(result["solution"], solution)
        self.assertFalse(result["is_correct"])

if __name__ == '__main__':
    unittest.main()
