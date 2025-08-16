#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für das BabylonLogicCore-Modul der MPRIME Engine.
"""

import unittest
import sys
import os
from typing import Dict, Any, List

# Füge das Hauptverzeichnis zum Pfad hinzu, um die MISO-Module zu importieren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from miso.math.mprime.babylon_logic import BabylonLogicCore

class TestBabylonLogicCore(unittest.TestCase):
    """
    Testklasse für das BabylonLogicCore-Modul
    """
    
    def setUp(self):
        """
        Einrichtung für die Tests
        """
        self.babylon_logic = BabylonLogicCore()
    
    def test_initialization(self):
        """
        Testet die korrekte Initialisierung
        """
        self.assertIsNotNone(self.babylon_logic)
        self.assertEqual(self.babylon_logic.base, 60)
        self.assertEqual(self.babylon_logic.max_precision, 10)
        self.assertTrue(self.babylon_logic.use_hybrid_notation)
    
    def test_decimal_to_babylonian(self):
        """
        Testet die Umwandlung von Dezimal- in babylonische Zahlen
        """
        # Teste einige einfache Umwandlungen
        test_cases = [
            (1, "1"),
            (10, "10"),
            (59, "59"),
            (60, "1,0"),
            (61, "1,1"),
            (120, "2,0"),
            (3599, "59,59"),
            (3600, "1,0,0"),
            (3661, "1,1,1")
        ]
        
        for decimal, expected in test_cases:
            result = self.babylon_logic.decimal_to_babylonian(decimal)
            self.assertEqual(result, expected, f"Fehler bei der Umwandlung von {decimal}")
    
    def test_babylonian_to_decimal(self):
        """
        Testet die Umwandlung von babylonischen in Dezimalzahlen
        """
        # Teste einige einfache Umwandlungen
        test_cases = [
            ("1", 1),
            ("10", 10),
            ("59", 59),
            ("1,0", 60),
            ("1,1", 61),
            ("2,0", 120),
            ("59,59", 3599),
            ("1,0,0", 3600),
            ("1,1,1", 3661)
        ]
        
        for babylonian, expected in test_cases:
            result = self.babylon_logic.babylonian_to_decimal(babylonian)
            self.assertEqual(result, expected, f"Fehler bei der Umwandlung von {babylonian}")
    
    def test_add_babylonian(self):
        """
        Testet die Addition babylonischer Zahlen
        """
        # Teste einige einfache Additionen
        test_cases = [
            ("1", "1", "2"),
            ("10", "20", "30"),
            ("59", "1", "1,0"),
            ("59", "2", "1,1"),
            ("1,0", "1,0", "2,0"),
            ("59,59", "1", "1,0,0"),
            ("1,1,1", "1,1,1", "2,2,2")
        ]
        
        for a, b, expected in test_cases:
            result = self.babylon_logic.add_babylonian(a, b)
            self.assertEqual(result, expected, f"Fehler bei der Addition von {a} + {b}")
    
    def test_subtract_babylonian(self):
        """
        Testet die Subtraktion babylonischer Zahlen
        """
        # Teste einige einfache Subtraktionen
        test_cases = [
            ("2", "1", "1"),
            ("30", "20", "10"),
            ("1,0", "1", "59"),
            ("1,1", "2", "59"),
            ("2,0", "1,0", "1,0"),
            ("1,0,0", "1", "59,59"),
            ("2,2,2", "1,1,1", "1,1,1")
        ]
        
        for a, b, expected in test_cases:
            result = self.babylon_logic.subtract_babylonian(a, b)
            self.assertEqual(result, expected, f"Fehler bei der Subtraktion von {a} - {b}")
    
    def test_multiply_babylonian(self):
        """
        Testet die Multiplikation babylonischer Zahlen
        """
        # Teste einige einfache Multiplikationen
        test_cases = [
            ("2", "3", "6"),
            ("10", "6", "1,0,0"),
            ("30", "2", "1,0"),
            ("1,0", "2", "2,0"),
            ("1,0", "1,0", "1,0,0"),
            ("2,0", "30", "1,0,0,0")
        ]
        
        for a, b, expected in test_cases:
            result = self.babylon_logic.multiply_babylonian(a, b)
            self.assertEqual(result, expected, f"Fehler bei der Multiplikation von {a} * {b}")
    
    def test_divide_babylonian(self):
        """
        Testet die Division babylonischer Zahlen
        """
        # Teste einige einfache Divisionen
        test_cases = [
            ("6", "3", "2"),
            ("1,0", "30", "2"),
            ("2,0", "2", "1,0"),
            ("1,0,0", "1,0", "1,0"),
            ("1,0,0,0", "2,0", "30")
        ]
        
        for a, b, expected in test_cases:
            result = self.babylon_logic.divide_babylonian(a, b)
            self.assertEqual(result, expected, f"Fehler bei der Division von {a} / {b}")
    
    def test_babylonian_to_sexagesimal_fraction(self):
        """
        Testet die Umwandlung in Sexagesimalbrüche
        """
        # Teste einige einfache Umwandlungen
        test_cases = [
            (0.5, "0;30"),
            (0.25, "0;15"),
            (0.1, "0;6"),
            (1.5, "1;30"),
            (2.25, "2;15"),
            (3.1, "3;6")
        ]
        
        for decimal, expected in test_cases:
            result = self.babylon_logic.decimal_to_sexagesimal_fraction(decimal)
            self.assertEqual(result, expected, f"Fehler bei der Umwandlung von {decimal}")
    
    def test_sexagesimal_fraction_to_decimal(self):
        """
        Testet die Umwandlung von Sexagesimalbrüchen in Dezimalzahlen
        """
        # Teste einige einfache Umwandlungen
        test_cases = [
            ("0;30", 0.5),
            ("0;15", 0.25),
            ("0;6", 0.1),
            ("1;30", 1.5),
            ("2;15", 2.25),
            ("3;6", 3.1)
        ]
        
        for sexagesimal, expected in test_cases:
            result = self.babylon_logic.sexagesimal_fraction_to_decimal(sexagesimal)
            self.assertAlmostEqual(result, expected, places=2, msg=f"Fehler bei der Umwandlung von {sexagesimal}")
    
    def test_babylonian_square_root(self):
        """
        Testet die Berechnung der Quadratwurzel nach babylonischem Verfahren
        """
        # Teste einige einfache Quadratwurzeln
        test_cases = [
            (4, "2"),
            (9, "3"),
            (16, "4"),
            (25, "5"),
            (36, "6"),
            (3600, "1,0,0")
        ]
        
        for decimal, expected in test_cases:
            result = self.babylon_logic.babylonian_square_root(decimal)
            self.assertEqual(result, expected, f"Fehler bei der Berechnung der Quadratwurzel von {decimal}")
    
    def test_to_cuneiform(self):
        """
        Testet die Umwandlung in Keilschrift-Notation
        """
        # Teste einige einfache Umwandlungen
        test_cases = [
            ("1", "𒁹"),
            ("10", "𒌋"),
            ("30", "𒌍"),
            ("1,0", "𒐙"),
            ("1,1", "𒐙𒁹"),
            ("2,0", "𒌋𒌋"),
            ("10,0", "𒐏")
        ]
        
        for babylonian, expected in test_cases:
            result = self.babylon_logic.to_cuneiform(babylonian)
            self.assertEqual(result, expected, f"Fehler bei der Umwandlung von {babylonian} in Keilschrift")
    
    def test_from_cuneiform(self):
        """
        Testet die Umwandlung von Keilschrift-Notation
        """
        # Teste einige einfache Umwandlungen
        test_cases = [
            ("𒁹", "1"),
            ("𒌋", "10"),
            ("𒌍", "30"),
            ("𒐙", "1,0"),
            ("𒐙𒁹", "1,1"),
            ("𒌋𒌋", "2,0"),
            ("𒐏", "10,0")
        ]
        
        for cuneiform, expected in test_cases:
            result = self.babylon_logic.from_cuneiform(cuneiform)
            self.assertEqual(result, expected, f"Fehler bei der Umwandlung von Keilschrift {cuneiform}")
    
    def test_solve_babylonian_equation(self):
        """
        Testet die Lösung einer babylonischen Gleichung
        """
        # Teste eine einfache Gleichung: x + 10 = 30
        equation = {
            "left": {"type": "variable", "value": "x"},
            "right": {"type": "number", "value": "30"},
            "operator": "+",
            "operand": {"type": "number", "value": "10"}
        }
        
        result = self.babylon_logic.solve_babylonian_equation(equation)
        
        # Überprüfe, ob das Ergebnis ein Dictionary ist
        self.assertIsInstance(result, dict)
        
        # Überprüfe, ob die Lösung korrekt ist
        self.assertEqual(result["solution"], "20")
        self.assertEqual(result["decimal_solution"], 20)

if __name__ == '__main__':
    unittest.main()
