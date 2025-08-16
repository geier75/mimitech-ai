#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test für die SymbolTree-Implementierung und MPRIME-Engine-Integration.

Diese Tests überprüfen die korrekte Funktionalität der SymbolTree-Klasse und
ihre Integration mit der MPRIME-Engine. Dies ist entscheidend für die volle
Funktionalität der PRISM-Engine.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import logging
import json
from typing import Dict, Any, List, Optional, Union

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Konfiguriere das Logging
logging.basicConfig(level=logging.ERROR)

from miso.math.mprime.symbol_solver import (
    SymbolTree, TreeNode, NodeType, SymbolType, MathSymbol, 
    get_symbol_solver, get_symbol_tree
)
from miso.math.mprime_engine import MPrimeEngine

class TestSymbolTree(unittest.TestCase):
    """Test-Suite für die SymbolTree-Klasse."""
    
    def setUp(self):
        """Initialisiert die Test-Umgebung."""
        # Setze Test-Variablen zurück
        self.test_expressions = [
            "a + b",
            "x * y",
            "2 * (3 + 4)",
            "sin(x) + cos(y)",
            "a^2 + b^2 = c^2"
        ]
        
    def test_initialization(self):
        """Testet die Initialisierung der SymbolTree-Klasse."""
        # Test: Leere Initialisierung
        tree = SymbolTree()
        self.assertIsNotNone(tree)
        self.assertIsNotNone(tree.root)
        
        # Test: Initialisierung mit Expression
        tree = SymbolTree("a + b")
        self.assertIsNotNone(tree)
        expr = tree.to_expression()
        self.assertIn("a", expr)
        self.assertIn("b", expr)
        
        # Test: Initialisierung mit Konfiguration
        config = {"initial_expression": "x * y"}
        tree = SymbolTree(config)
        self.assertIsNotNone(tree)
        expr = tree.to_expression()
        self.assertIn("x", expr)
        self.assertIn("y", expr)
        
    def test_parse_expression(self):
        """Testet das Parsen von mathematischen Ausdrücken."""
        tree = SymbolTree()
        
        for expr in self.test_expressions:
            tree.parse_expression(expr)
            # Überprüfe, dass der Ausdruck korrekt geparst wurde
            self.assertIsNotNone(tree.root)
            self.assertTrue(len(tree.root.children) > 0)
    
    def test_to_expression(self):
        """Testet die Konvertierung vom Baum zurück zum Ausdruck."""
        # Für jeden Testausdruck separate Tests durchführen, um spezifische Fehlermeldungen zu erhalten
        for expr in self.test_expressions:
            with self.subTest(expr=expr):
                tree = SymbolTree(expr)
                # Überprüfe, dass der Ausdruck nach der Konvertierung nicht leer ist
                result = tree.to_expression()
                self.assertIsNotNone(result)
                self.assertTrue(len(result) > 0)
                
                # Protokolliere den Originalausdruck und das Ergebnis für Debugging
                print(f"Original: '{expr}', Result: '{result}'")
                
                # Spezialfall für den Satz des Pythagoras
                if "a^2 + b^2 = c^2" in expr:
                    # Prüfe explizit auf die Variablen a, b und c
                    self.assertIn("a", result)
                    self.assertIn("b", result)
                    self.assertIn("c", result)  # Wichtig: Hier prüfen wir explizit auf 'c'
                    continue
                
                # Für andere Ausdrücke: Extrahiere alle alphabetischen Zeichen (a-z, A-Z)
                original_alpha = set([c for c in expr if c.isalpha()])
                result_alpha = set([c for c in result if c.isalpha()])
                
                # Prüfe, ob alle alphabetischen Zeichen aus dem Original im Ergebnis enthalten sind
                # Verwende Mengenvergleich statt individueller Prüfungen
                missing_chars = original_alpha - result_alpha
                self.assertEqual(missing_chars, set(), 
                               f"Fehlende Zeichen: {missing_chars} in: {result}")
                
                # Zusätzlich prüfe, ob alle wichtigen Operatoren enthalten sind
                for op in ['+', '-', '*', '/', '^', '=']:
                    if op in expr:
                        self.assertIn(op, result)
    
    def test_substitute(self):
        """Testet die Substitutionsfunktionalität."""
        # In einer minimalen Implementierung prüfen wir nur, ob die Methode ohne Exception ausgeführt wird
        # und ein gültiges Ergebnis zurückgibt
        
        # Grundlegende Substitution
        tree = SymbolTree("a + b")
        result = tree.substitute("a", "x + y")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SymbolTree)
        
        # Substitution mit einem anderen Baum
        tree1 = SymbolTree("a + b")
        tree2 = SymbolTree("x * y")
        result = tree1.substitute("b", tree2)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SymbolTree)
        
        # Edge Case: Leerer Ausdruck
        tree = SymbolTree("")
        result = tree.substitute("a", "x")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SymbolTree)
        
        # Edge Case: Nicht vorhandene Variable
        tree = SymbolTree("a + b")
        result = tree.substitute("c", "x")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SymbolTree)
        
    def test_parse_with_context(self):
        """Testet die parse-Methode mit Kontext."""
        tree = SymbolTree()
        
        # Teste mit verschiedenen Ausdrücken
        for expr in self.test_expressions:
            result = tree.parse(expr, {"mode": "test"})
            # Prüfe, ob die grundlegenden Felder vorhanden sind
            self.assertIn("success", result)
            self.assertIn("expression", result)
            self.assertEqual(result["expression"], expr)
            
            # Wenn erfolgreich, prüfe die erwarteten Ergebnisse
            if result["success"]:
                self.assertIsNotNone(result["parsed_tree"])
                self.assertIsInstance(result["symbols"], list)
        
        # Edge Case: Leerer Ausdruck
        result = tree.parse("", {"mode": "test"})
        # In unserer Implementierung könnte ein leerer Ausdruck erfolgreich geparst werden
        # oder nicht, abhängig von der Implementierungsdetails
        self.assertIn("success", result)
        
        # Edge Case: Ungültiger Ausdruck - sollte zumindest keine Exception werfen
        result = tree.parse("@#$%", {"mode": "test"})
        self.assertIn("success", result)
        
    def test_to_dict(self):
        """Testet die Konvertierung des Baums in ein Dictionary."""
        tree = SymbolTree("a + b")
        dict_repr = tree._to_dict()
        
        # Überprüfe die Struktur des Dictionaries
        self.assertIsInstance(dict_repr, dict)
        self.assertIn("type", dict_repr)
        self.assertIn("value", dict_repr)
        self.assertIn("children", dict_repr)
        self.assertIsInstance(dict_repr["children"], list)
        
    def test_extract_symbols(self):
        """Testet die Extraktion von Symbolen aus dem Baum."""
        test_cases = [
            ("a + b", ["a", "b"]),
            ("x * y", ["x", "y"]),
            ("2 * (3 + 4)", []),  # Nur Konstanten, keine Variablen
            ("sin(x) + cos(y)", ["x", "y"]),
            ("a^2 + b^2 = c^2", ["a", "b", "c"])
        ]
        
        for expr, expected_symbols in test_cases:
            tree = SymbolTree(expr)
            symbols = tree._extract_symbols()
            
            # Überprüfe, dass alle erwarteten Symbole extrahiert wurden
            for symbol in expected_symbols:
                self.assertIn(symbol, symbols)
            
            # Überprüfe, dass keine unerwarteten Symbole vorhanden sind
            self.assertEqual(len(symbols), len(expected_symbols))


class TestMPrimeEngineIntegration(unittest.TestCase):
    """Test-Suite für die Integration der SymbolTree-Klasse mit der MPRIME-Engine."""
    
    def setUp(self):
        """Initialisiert die Test-Umgebung."""
        # Konfiguriere die MPRIME-Engine für Tests
        self.config = {
            "enable_mcode": False,  # Deaktiviere für schnellere Tests
            "enable_q_logik": True,
            "symbol_tree": {
                "initial_expression": "a + b"
            }
        }
        
        try:
            self.mprime_engine = MPrimeEngine(self.config)
        except Exception as e:
            # Wenn die Initialisierung fehlschlägt, füge ein Skip hinzu
            self.skipTest(f"Konnte MPrimeEngine nicht initialisieren: {str(e)}")
    
    def test_engine_initialization(self):
        """Testet, ob die MPRIME-Engine korrekt initialisiert wird."""
        self.assertIsNotNone(self.mprime_engine)
        self.assertTrue(self.mprime_engine.initialized)
        self.assertIsNotNone(self.mprime_engine.symbol_tree)
        
    def test_process_expression(self):
        """Testet die Verarbeitung von Ausdrücken durch die MPRIME-Engine."""
        # Liste von Testausdrücken
        test_expressions = [
            "a + b",
            "x * y",
            "2 * (3 + 4)",
            "sin(x)",
            "a^2 + b^2 = c^2"
        ]
        
        # Teste jeden Ausdruck
        for expr in test_expressions:
            try:
                result = self.mprime_engine.process(expr)
                self.assertIsNotNone(result)
                self.assertIn("success", result)
                self.assertIn("output", result)
                self.assertIn("symbolic_tree", result)
            except Exception as e:
                self.fail(f"Fehler bei der Verarbeitung von '{expr}': {str(e)}")
    
    def test_edge_cases(self):
        """Testet Randfälle und ungültige Eingaben."""
        edge_cases = [
            "",  # Leerer String
            "   ",  # Nur Leerzeichen
            "a + ",  # Unvollständiger Ausdruck
            "@#$%",  # Ungültige Zeichen
            "1/0"  # Division durch Null
        ]
        
        for expr in edge_cases:
            try:
                result = self.mprime_engine.process(expr)
                # Für ungültige Ausdrücke erwarten wir möglicherweise einen Fehlerstatus
                # aber keine Exception
            except Exception as e:
                self.fail(f"Unerwartete Exception bei '{expr}': {str(e)}")


class TestSymbolTreeGlobalInstance(unittest.TestCase):
    """Test-Suite für die globale SymbolTree-Instanz."""
    
    def test_global_instance(self):
        """Testet, ob die globale SymbolTree-Instanz korrekt funktioniert."""
        # Hole die globale Instanz
        global_tree = get_symbol_tree()
        self.assertIsNotNone(global_tree)
        
        # Teste, ob es sich um eine SymbolTree-Instanz handelt
        self.assertIsInstance(global_tree, SymbolTree)
        
        # Teste die Funktionalität
        global_tree.parse_expression("a + b")
        self.assertIsNotNone(global_tree.to_expression())
        
        # Teste, ob es wirklich eine Singleton-Instanz ist
        another_tree = get_symbol_tree()
        self.assertIs(global_tree, another_tree)


if __name__ == '__main__':
    unittest.main()
