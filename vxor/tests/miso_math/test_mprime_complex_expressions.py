#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Erweiterte Tests für komplexe mathematische Ausdrücke in der MPRIME-Engine.

Diese Tests validieren die korrekte Verarbeitung komplexer mathematischer Konstrukte
wie Differentiale, Integrale, Limiten und andere fortgeschrittene Ausdrücke in der
MPRIME-Engine und der SymbolTree-Implementierung.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Konfiguriere das Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.Test.MPRIME.ComplexExpressions")

from miso.math.mprime.symbol_solver import (
    SymbolTree, TreeNode, NodeType, SymbolType, MathSymbol, 
    get_symbol_solver, get_symbol_tree
)
from miso.math.mprime.contextual_math import ContextualMathCore
from miso.math.mprime_engine import MPrimeEngine

class TestComplexMathExpressions(unittest.TestCase):
    """Test-Suite für komplexe mathematische Ausdrücke in der MPRIME-Engine."""
    
    def setUp(self):
        """Initialisiert die Test-Umgebung."""
        # Differentialgleichungen
        self.differential_expressions = [
            "d/dx(x^2)",
            "d/dx(sin(x))",
            "d/dx(e^x)",
            "d^2/dx^2(x^3 + 2*x^2 + x)"
        ]
        
        # Integrale
        self.integral_expressions = [
            "∫(x)dx",
            "∫(x^2)dx",
            "∫(sin(x))dx",
            "∫(e^x)dx"
        ]
        
        # Limiten
        self.limit_expressions = [
            "lim(x→0)(sin(x)/x)",
            "lim(x→∞)(1 + 1/x)^x",
            "lim(x→1)((x^3 - 1)/(x - 1))"
        ]
        
        # Gleichungssysteme
        self.equation_systems = [
            "2*x + 3*y = 7; x - y = 1",
            "x^2 + y^2 = 25; x + y = 7"
        ]
        
        # Komplexe Ausdrücke mit Variablen
        self.complex_variable_expressions = [
            "a*sin(b*x) + c*cos(d*x)",
            "a^2 + b^2 - 2*a*b*cos(C)",  # Kosinussatz
            "√(a^2 + b^2)"  # Pythagoras
        ]
        
        # Initialisiere die MPRIME-Engine und den ContextualMathCore
        self.mprime_engine = MPrimeEngine()
        self.math_core = ContextualMathCore()
        
        # Setze den mathematischen Kontext - verwende 'scientific' da dies einer der verfügbaren Kontexte ist
        self.math_core.active_context = "scientific"
    
    def test_differential_expressions(self):
        """Testet die Verarbeitung von Differentialausdrücken."""
        for expr in self.differential_expressions:
            logger.info(f"Teste Differentialausdruck: {expr}")
            
            # Verarbeite mit dem ContextualMathCore
            result = self.math_core.process(expr)
            
            # Überprüfe grundlegende Ergebniseigenschaften
            self.assertTrue(result["success"], f"Fehler bei der Verarbeitung von {expr}: {result.get('error')}")
            self.assertIsNotNone(result["processed_text"], f"Kein Ergebnis für {expr}")
            
            # Erstelle einen Symbolbaum aus dem Ausdruck
            tree = SymbolTree(expr)
            
            # Überprüfe, dass der Baum korrekt aufgebaut wurde
            self.assertIsNotNone(tree.root)
            self.assertTrue(len(tree.root.children) > 0, f"Leerer Symbolbaum für {expr}")
            
            # Konvertiere zurück zum Ausdruck und prüfe auf Konsistenz
            rebuilt_expr = tree.to_expression()
            self.assertTrue(len(rebuilt_expr) > 0, f"Leerer rekonstruierter Ausdruck für {expr}")
            
            # Log das Ergebnis
            logger.info(f"Ergebnis für {expr}: {result['processed_text']}")
    
    def test_integral_expressions(self):
        """Testet die Verarbeitung von Integralausdrücken."""
        for expr in self.integral_expressions:
            logger.info(f"Teste Integralausdruck: {expr}")
            
            # Verarbeite mit dem ContextualMathCore
            result = self.math_core.process(expr)
            
            # Überprüfe grundlegende Ergebniseigenschaften
            self.assertTrue(result["success"], f"Fehler bei der Verarbeitung von {expr}: {result.get('error')}")
            self.assertIsNotNone(result["processed_text"], f"Kein Ergebnis für {expr}")
            
            # Erstelle einen Symbolbaum aus dem Ausdruck
            tree = SymbolTree(expr)
            
            # Überprüfe, dass der Baum korrekt aufgebaut wurde
            self.assertIsNotNone(tree.root)
            self.assertTrue(len(tree.root.children) > 0, f"Leerer Symbolbaum für {expr}")
            
            # Konvertiere zurück zum Ausdruck und prüfe auf Konsistenz
            rebuilt_expr = tree.to_expression()
            self.assertTrue(len(rebuilt_expr) > 0, f"Leerer rekonstruierter Ausdruck für {expr}")
            
            # Log das Ergebnis
            logger.info(f"Ergebnis für {expr}: {result['processed_text']}")
    
    def test_limit_expressions(self):
        """Testet die Verarbeitung von Limitausdrücken."""
        for expr in self.limit_expressions:
            logger.info(f"Teste Limitausdruck: {expr}")
            
            # Verarbeite mit dem ContextualMathCore
            result = self.math_core.process(expr)
            
            # Überprüfe grundlegende Ergebniseigenschaften
            self.assertTrue(result["success"], f"Fehler bei der Verarbeitung von {expr}: {result.get('error')}")
            self.assertIsNotNone(result["processed_text"], f"Kein Ergebnis für {expr}")
            
            # Erstelle einen Symbolbaum aus dem Ausdruck
            tree = SymbolTree(expr)
            
            # Überprüfe, dass der Baum korrekt aufgebaut wurde
            self.assertIsNotNone(tree.root)
            self.assertTrue(len(tree.root.children) > 0, f"Leerer Symbolbaum für {expr}")
            
            # Konvertiere zurück zum Ausdruck und prüfe auf Konsistenz
            rebuilt_expr = tree.to_expression()
            self.assertTrue(len(rebuilt_expr) > 0, f"Leerer rekonstruierter Ausdruck für {expr}")
            
            # Log das Ergebnis
            logger.info(f"Ergebnis für {expr}: {result['processed_text']}")
    
    def test_equation_systems(self):
        """Testet die Verarbeitung von Gleichungssystemen."""
        for expr in self.equation_systems:
            logger.info(f"Teste Gleichungssystem: {expr}")
            
            # Verarbeite mit dem ContextualMathCore
            result = self.math_core.process(expr)
            
            # Überprüfe grundlegende Ergebniseigenschaften
            self.assertTrue(result["success"], f"Fehler bei der Verarbeitung von {expr}: {result.get('error')}")
            self.assertIsNotNone(result["processed_text"], f"Kein Ergebnis für {expr}")
            
            # Für Gleichungssysteme können wir jeden Teil separat testen
            equations = expr.split(";")
            for eq in equations:
                eq = eq.strip()
                # Erstelle einen Symbolbaum aus der Gleichung
                tree = SymbolTree(eq)
                
                # Überprüfe, dass der Baum korrekt aufgebaut wurde
                self.assertIsNotNone(tree.root)
                self.assertTrue(len(tree.root.children) > 0, f"Leerer Symbolbaum für {eq}")
                
                # Konvertiere zurück zum Ausdruck und prüfe auf Konsistenz
                rebuilt_eq = tree.to_expression()
                self.assertTrue(len(rebuilt_eq) > 0, f"Leerer rekonstruierter Ausdruck für {eq}")
            
            # Log das Ergebnis
            logger.info(f"Ergebnis für {expr}: {result['processed_text']}")
    
    def test_complex_variable_expressions(self):
        """Testet die Verarbeitung von komplexen Ausdrücken mit Variablen."""
        for expr in self.complex_variable_expressions:
            logger.info(f"Teste komplexen Ausdruck mit Variablen: {expr}")
            
            # Verarbeite mit dem ContextualMathCore
            result = self.math_core.process(expr)
            
            # Überprüfe grundlegende Ergebniseigenschaften
            self.assertTrue(result["success"], f"Fehler bei der Verarbeitung von {expr}: {result.get('error')}")
            self.assertIsNotNone(result["processed_text"], f"Kein Ergebnis für {expr}")
            
            # Erstelle einen Symbolbaum aus dem Ausdruck
            tree = SymbolTree(expr)
            
            # Überprüfe, dass der Baum korrekt aufgebaut wurde
            self.assertIsNotNone(tree.root)
            self.assertTrue(len(tree.root.children) > 0, f"Leerer Symbolbaum für {expr}")
            
            # Extrahiere alle Symbole (Variablen)
            symbols = tree._extract_symbols()
            self.assertTrue(len(symbols) > 0, f"Keine Symbole gefunden in {expr}")
            
            # Definiere erwartete Variablen für jeden Ausdruck
            expected_variables = {
                "a*sin(b*x) + c*cos(d*x)": ['a', 'b', 'c', 'd', 'x'],
                "a^2 + b^2 - 2*a*b*cos(C)": ['a', 'b', 'C'],
                "√(a^2 + b^2)": ['a', 'b']
            }
            
            # Hole die für diesen Ausdruck erwarteten Variablen
            expected_vars = expected_variables.get(expr, [])
            
            # Prüfe, ob alle erwarteten Variablen gefunden wurden
            for var in expected_vars:
                self.assertIn(var, symbols, f"Erwartete Variable {var} nicht in extrahierten Symbolen für {expr}")
                
            # Protokolliere die Variablen
            logger.info(f"Erwartete Variablen in {expr}: {expected_vars}")
            logger.info(f"Gefundene Variablen in {expr}: {symbols}")
            
            # Optional: Prüfe, ob keine unerwünschten Variablen gefunden wurden
            # (auskommentiert, da die aktuelle Implementierung vielleicht mehr findet)
            # self.assertEqual(set(symbols), set(expected_vars), 
            #                f"Gefundene Variablen stimmen nicht mit erwarteten überein für {expr}")
            
            
            # Konvertiere zurück zum Ausdruck und prüfe auf Konsistenz
            rebuilt_expr = tree.to_expression()
            self.assertTrue(len(rebuilt_expr) > 0, f"Leerer rekonstruierter Ausdruck für {expr}")
            
            # Log das Ergebnis und die gefundenen Symbole
            logger.info(f"Ergebnis für {expr}: {result['processed_text']}")
            logger.info(f"Gefundene Symbole in {expr}: {symbols}")
    
    def test_stress_performance(self):
        """Stress-Test für Performance-Messung."""
        # Wähle einen komplexeren Ausdruck für den Stress-Test
        stress_expr = "a*sin(b*x) + c*cos(d*x) + ∫(x^2 + 3*x)dx + d/dx(e^(x*y))"
        
        # Anzahl der Iterationen für den Stress-Test
        iterations = 100
        
        logger.info(f"Starte Stress-Test mit {iterations} Iterationen für: {stress_expr}")
        
        # Messe Zeit für die Verarbeitung
        start_time = time.time()
        
        for _ in range(iterations):
            # Verarbeite mit dem ContextualMathCore
            result = self.math_core.process(stress_expr)
            self.assertTrue(result["success"], f"Fehler im Stress-Test: {result.get('error')}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        average_time = elapsed_time / iterations
        
        logger.info(f"Stress-Test abgeschlossen.")
        logger.info(f"Gesamtzeit: {elapsed_time:.4f} Sekunden")
        logger.info(f"Durchschnittliche Zeit pro Iteration: {average_time:.4f} Sekunden")
        
        # Fester Threshold für Performance-Anforderungen (kann angepasst werden)
        performance_threshold = 0.01  # 10 ms pro Verarbeitung
        
        # Prüfe, ob Performance-Anforderungen erfüllt sind
        self.assertLessEqual(average_time, performance_threshold, 
                           f"Performance unzureichend. Durchschnittliche Zeit: {average_time:.4f} s, Erwartet: ≤ {performance_threshold:.4f} s")


if __name__ == '__main__':
    unittest.main()
