#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Runtime Tests

Dieser Modul enthält Tests für die M-CODE Runtime-Komponenten.
Die Tests überprüfen die Funktionalität des M-CODE Compilers, JIT-Engines,
Security Sandbox und Runtime-Umgebung.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import unittest
import logging
from pathlib import Path

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, str(Path(__file__).parent.parent))

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.tests.m_code")

# Importiere Module
from miso.lang.mcode_engine import MCodeEngine, MCodeConfig
from miso.lang.mcode_parser import MCodeParser
from miso.lang.mcode_typechecker import TypeChecker
from miso.lang.mcode_ast import ASTCompiler
from miso.lang.mcode_jit import GPUJITEngine
from miso.lang.mcode_sandbox import SecuritySandbox
from miso.lang.mcode_runtime import MCodeRuntime


class TestMCodeParser(unittest.TestCase):
    """Tests für den M-CODE Parser"""
    
    def setUp(self):
        """Initialisiert den Test"""
        self.parser = MCodeParser()
    
    def test_basic_parsing(self):
        """Testet grundlegende Parsing-Funktionalität"""
        code = """
        # Einfaches M-CODE Beispiel
        let x = 10
        let y = 20
        return x + y
        """
        
        ast = self.parser.parse(code)
        
        # Überprüfe, ob der AST korrekt erstellt wurde
        self.assertIsNotNone(ast)
        self.assertIn("statements", ast)
        self.assertEqual(len(ast["statements"]), 3)  # 2 Zuweisungen + 1 Return
    
    def test_complex_parsing(self):
        """Testet komplexere Parsing-Funktionalität"""
        code = """
        # Komplexeres M-CODE Beispiel
        let tensor A = randn(4, 4)
        let tensor B = eye(4)
        
        # Matrix-Multiplikation
        let C = A @ B
        
        # Bedingte Anweisung
        if norm(C) > 2.0:
            return normalize(C)
        else:
            return C
        """
        
        ast = self.parser.parse(code)
        
        # Überprüfe, ob der AST korrekt erstellt wurde
        self.assertIsNotNone(ast)
        self.assertIn("statements", ast)
        # 3 Zuweisungen + 1 If-Statement
        self.assertEqual(len(ast["statements"]), 4)


class TestTypeChecker(unittest.TestCase):
    """Tests für den M-CODE Type Checker"""
    
    def setUp(self):
        """Initialisiert den Test"""
        self.parser = MCodeParser()
        self.type_checker = TypeChecker()
    
    def test_basic_type_checking(self):
        """Testet grundlegende Typprüfung"""
        code = """
        let x = 10
        let y = "Hello"
        let z = x + 5  # Korrekte Typkompatibilität
        """
        
        ast = self.parser.parse(code)
        errors = self.type_checker.check(ast)
        
        # Keine Typfehler erwartet
        self.assertEqual(len(errors), 0)
    
    def test_type_error_detection(self):
        """Testet Erkennung von Typfehlern"""
        code = """
        let x = 10
        let y = "Hello"
        let z = x + y  # Typfehler: Integer + String
        """
        
        ast = self.parser.parse(code)
        errors = self.type_checker.check(ast)
        
        # Ein Typfehler erwartet
        self.assertGreater(len(errors), 0)


class TestASTCompiler(unittest.TestCase):
    """Tests für den M-CODE AST Compiler"""
    
    def setUp(self):
        """Initialisiert den Test"""
        self.parser = MCodeParser()
        self.compiler = ASTCompiler(optimization_level=2)
    
    def test_basic_compilation(self):
        """Testet grundlegende Kompilierung"""
        code = """
        let x = 10
        let y = 20
        return x + y
        """
        
        ast = self.parser.parse(code)
        bytecode = self.compiler.compile(ast, "<test>")
        
        # Überprüfe, ob der Bytecode korrekt erstellt wurde
        self.assertIsNotNone(bytecode)
        self.assertIn("instructions", bytecode)
        self.assertGreater(len(bytecode["instructions"]), 0)
    
    def test_optimization(self):
        """Testet Optimierung"""
        code = """
        let x = 10
        let y = 20
        let z = x + y  # Sollte zu einer Konstante optimiert werden
        return z
        """
        
        # Kompiliere mit Optimierung
        ast = self.parser.parse(code)
        bytecode_opt = self.compiler.compile(ast, "<test>")
        
        # Kompiliere ohne Optimierung
        compiler_no_opt = ASTCompiler(optimization_level=0)
        bytecode_no_opt = compiler_no_opt.compile(ast, "<test>")
        
        # Optimierter Bytecode sollte weniger Instruktionen haben
        self.assertLess(len(bytecode_opt["instructions"]), len(bytecode_no_opt["instructions"]))


class TestGPUJITEngine(unittest.TestCase):
    """Tests für die GPU JIT Engine"""
    
    def setUp(self):
        """Initialisiert den Test"""
        self.parser = MCodeParser()
        self.compiler = ASTCompiler(optimization_level=2)
        self.jit_engine = GPUJITEngine(use_gpu=True, use_neural_engine=False)
    
    def test_jit_compilation(self):
        """Testet JIT-Kompilierung"""
        code = """
        let tensor A = randn(4, 4)
        let tensor B = eye(4)
        return A @ B  # Matrix-Multiplikation
        """
        
        ast = self.parser.parse(code)
        bytecode = self.compiler.compile(ast, "<test>")
        
        # JIT-Kompilierung
        jit_module = self.jit_engine.compile(bytecode)
        
        # Überprüfe, ob das JIT-Modul korrekt erstellt wurde
        self.assertIsNotNone(jit_module)
        self.assertIn("module_handle", jit_module)


class TestSecuritySandbox(unittest.TestCase):
    """Tests für die Security Sandbox"""
    
    def setUp(self):
        """Initialisiert den Test"""
        self.sandbox = SecuritySandbox(
            security_level=2,
            memory_limit_mb=1024,
            max_execution_time_ms=5000,
            allow_network_access=False,
            allow_file_access=False
        )
    
    def test_code_verification(self):
        """Testet Code-Verifizierung"""
        # Sicherer Code
        safe_code = """
        let x = 10
        let y = 20
        return x + y
        """
        
        # Unsicherer Code (Dateizugriff)
        unsafe_code = """
        # Versuche, eine Datei zu öffnen
        let file = open("/etc/passwd", "r")
        return file.read()
        """
        
        # Verifiziere sicheren Code
        is_safe, _, _ = self.sandbox.verify_code(safe_code, "<test>")
        self.assertTrue(is_safe)
        
        # Verifiziere unsicheren Code
        is_safe, _, violation = self.sandbox.verify_code(unsafe_code, "<test>")
        self.assertFalse(is_safe)
        self.assertIsNotNone(violation)
    
    def test_resource_limits(self):
        """Testet Ressourcenbeschränkungen"""
        # Code, der viel Speicher verwendet
        memory_intensive_code = """
        # Erstelle einen großen Tensor
        let tensor A = randn(10000, 10000)  # ~800 MB
        return A
        """
        
        # Setze niedrigeres Speicherlimit
        low_memory_sandbox = SecuritySandbox(
            security_level=2,
            memory_limit_mb=100,  # Nur 100 MB
            max_execution_time_ms=5000,
            allow_network_access=False,
            allow_file_access=False
        )
        
        # Verifiziere Code mit hohem Speicherverbrauch
        is_safe, _, violation = low_memory_sandbox.verify_code(memory_intensive_code, "<test>")
        self.assertFalse(is_safe)
        self.assertIsNotNone(violation)


class TestMCodeRuntime(unittest.TestCase):
    """Tests für die M-CODE Runtime"""
    
    def setUp(self):
        """Initialisiert den Test"""
        config = MCodeConfig(
            optimization_level=2,
            use_jit=True,
            use_gpu=True,
            use_neural_engine=False,
            security_level=2,
            memory_limit_mb=1024,
            max_execution_time_ms=5000,
            allow_network_access=False,
            allow_file_access=False
        )
        self.runtime = MCodeRuntime(config)
    
    def test_basic_execution(self):
        """Testet grundlegende Ausführung"""
        code = """
        let x = 10
        let y = 20
        return x + y
        """
        
        # Erstelle Kontext
        context = self.runtime.create_context()
        
        # Führe Code aus
        result = self.runtime.execute_code(code, context)
        
        # Überprüfe Ergebnis
        self.assertTrue(result.success)
        self.assertEqual(result.result, 30)
    
    def test_tensor_operations(self):
        """Testet Tensor-Operationen"""
        code = """
        # Erstelle Tensoren
        let tensor A = ones(2, 2)
        let tensor B = ones(2, 2) * 2
        
        # Führe Operationen aus
        let C = A + B
        let D = A @ B
        
        return [C, D]
        """
        
        # Erstelle Kontext
        context = self.runtime.create_context()
        
        # Führe Code aus
        result = self.runtime.execute_code(code, context)
        
        # Überprüfe Ergebnis
        self.assertTrue(result.success)
        self.assertEqual(len(result.result), 2)  # Liste mit zwei Tensoren
        
        # C sollte ein 2x2 Tensor mit allen Elementen = 3 sein
        c_tensor = result.result[0]
        self.assertEqual(c_tensor.shape, (2, 2))
        self.assertTrue(all(val == 3 for val in c_tensor.flatten()))
        
        # D sollte ein 2x2 Tensor mit allen Elementen = 4 sein
        d_tensor = result.result[1]
        self.assertEqual(d_tensor.shape, (2, 2))
        self.assertTrue(all(val == 4 for val in d_tensor.flatten()))


class TestMCodeEngine(unittest.TestCase):
    """Tests für die M-CODE Engine"""
    
    def setUp(self):
        """Initialisiert den Test"""
        self.engine = MCodeEngine()
    
    def test_compile_and_execute(self):
        """Testet Kompilierung und Ausführung"""
        code = """
        # Berechne Fibonacci-Zahl
        let function fibonacci(n):
            if n <= 1:
                return n
            else:
                return fibonacci(n-1) + fibonacci(n-2)
        
        return fibonacci(10)
        """
        
        # Kompiliere Code
        module = self.engine.compile(code, "<test>")
        
        # Führe Code aus
        result = self.engine.execute(module)
        
        # Überprüfe Ergebnis
        self.assertTrue(result.success)
        self.assertEqual(result.result, 55)  # Fibonacci(10) = 55
    
    def test_event_handling(self):
        """Testet Ereignisbehandlung"""
        # Event-Handler
        events_triggered = []
        
        def on_execution_complete(result):
            events_triggered.append("execution_complete")
        
        # Registriere Event-Handler
        self.engine.register_callback("execution_complete", on_execution_complete)
        
        # Führe Code aus
        code = "return 42"
        result = self.engine.execute_code(code, "<test>")
        
        # Überprüfe, ob Event ausgelöst wurde
        self.assertIn("execution_complete", events_triggered)
    
    def test_error_handling(self):
        """Testet Fehlerbehandlung"""
        # Code mit Fehler
        code_with_error = """
        let x = 10
        let y = 0
        return x / y  # Division durch Null
        """
        
        # Führe Code aus
        result = self.engine.execute_code(code_with_error, "<test>")
        
        # Überprüfe Ergebnis
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
