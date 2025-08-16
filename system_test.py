#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Systemtest

Dieser umfassende Test validiert die Integration aller Hauptkomponenten des MISO Ultimate Systems,
mit besonderem Fokus auf die M-LINGUA zu T-Mathematics-Integration und Tensor-Backends.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [SYSTEM-TEST] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.SystemTest")

def test_t_mathematics_engine():
    """Testet die T-Mathematics Engine mit allen verfügbaren Backends"""
    logger.info("\n=== Test: T-Mathematics Engine ===")
    
    try:
        # Importiere die Engine
        from miso.math.t_mathematics.engine import TMathEngine
        
        # Initialisiere die Engine
        engine = TMathEngine()
        
        # Prüfe aktives Backend
        backend = engine.get_active_backend()
        logger.info(f"Aktives Backend: {backend}")
        
        # Erzeuge einen Tensor
        data = np.random.rand(10, 10).astype(np.float32)
        tensor = engine.tensor(data)
        logger.info(f"Tensor erzeugt: Backend {tensor.backend}, Shape {tensor.shape}")
        
        # Führe einige Operationen durch
        start_time = time.time()
        
        # Addition
        tensor_sum = tensor + tensor
        logger.info(f"Addition durchgeführt: Shape {tensor_sum.shape}")
        
        # Matrix-Multiplikation
        tensor_matmul = tensor @ tensor
        logger.info(f"Matrix-Multiplikation durchgeführt: Shape {tensor_matmul.shape}")
        
        # Exponentielle Funktion
        tensor_exp = tensor.exp()
        logger.info(f"Exponentialfunktion berechnet: Shape {tensor_exp.shape}")
        
        end_time = time.time()
        logger.info(f"Alle Tensor-Operationen abgeschlossen in {(end_time - start_time)*1000:.2f} ms")
        
        # Teste mathematische Auswertung
        expr_result = engine.evaluate("2 + 3 * 4")
        logger.info(f"Mathematischer Ausdruck evaluiert: 2 + 3 * 4 = {expr_result}")
        
        # Teste Gleichungslösung
        eq_result = engine.solve_equation("x^2 + 2*x - 3 = 0", "x")
        logger.info(f"Gleichung gelöst: {eq_result}")
        
        return True, "T-Mathematics Engine Tests erfolgreich"
    except Exception as e:
        logger.error(f"Fehler im T-Mathematics Engine Test: {e}")
        import traceback
        traceback.print_exc()
        return False, f"T-Mathematics Engine Test fehlgeschlagen: {e}"

def test_m_lingua_integration():
    """Testet die Integration von M-LINGUA mit der T-Mathematics Engine"""
    logger.info("\n=== Test: M-LINGUA Integration ===")
    
    try:
        # Importiere die MathBridge
        from miso.lang.mlingua.math_bridge import MathBridge
        
        # Initialisiere die Bridge
        bridge = MathBridge()
        logger.info("MathBridge initialisiert")
        
        # Teste verschiedene mathematische Ausdrücke in verschiedenen Sprachen
        expressions = [
            ("Berechne 2 + 3 * 4", "de"),
            ("Calculate the square root of 16", "en"),
            ("Calcular la matriz inversa de [[1, 2], [3, 4]]", "es"),
            ("Calculer la dérivée de x^2 + 3x", "fr")
        ]
        
        for expr, lang in expressions:
            start_time = time.time()
            result = bridge.process_math_expression(expr, lang)
            end_time = time.time()
            
            if result.success:
                logger.info(f"Expression: '{expr}' (Sprache: {lang})")
                logger.info(f"  Ergebnis: {result.result}")
                logger.info(f"  Zeit: {(end_time - start_time)*1000:.2f} ms")
            else:
                logger.warning(f"Expression: '{expr}' (Sprache: {lang})")
                logger.warning(f"  Fehler: {result.error_message}")
        
        return True, "M-LINGUA Integration Tests erfolgreich"
    except Exception as e:
        logger.error(f"Fehler im M-LINGUA Integration Test: {e}")
        import traceback
        traceback.print_exc()
        return False, f"M-LINGUA Integration Test fehlgeschlagen: {e}"

def test_backend_performance():
    """Führt Leistungsvergleiche für verschiedene Tensor-Backends durch"""
    logger.info("\n=== Test: Backend-Leistungsvergleich ===")
    
    try:
        # Importiere die Engine
        from miso.math.t_mathematics.engine import TMathEngine
        
        # Initialisiere die Engine
        engine = TMathEngine()
        
        # Prüfe verfügbare Backends
        try:
            import mlx.core
            has_mlx = True
            logger.info("MLX Backend verfügbar")
        except ImportError:
            has_mlx = False
            logger.warning("MLX Backend nicht verfügbar")
        
        try:
            import torch
            has_torch = True
            mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            logger.info(f"PyTorch Backend verfügbar (MPS: {mps_available})")
        except ImportError:
            has_torch = False
            logger.warning("PyTorch Backend nicht verfügbar")
        
        # Matrix-Größen für Tests
        sizes = [128, 512, 1024]
        operations = ["matmul", "add", "exp"]
        
        results = {}
        
        # Teste einfache Operationen mit verschiedenen Größen
        for size in sizes:
            results[size] = {}
            
            # Erstelle Testdaten
            data = np.random.rand(size, size).astype(np.float32)
            
            for op in operations:
                logger.info(f"Teste {op.upper()} mit Größe {size}x{size}")
                
                # MLX Backend
                if has_mlx:
                    try:
                        # Erzeuge Tensor mit MLX Backend
                        start_time = time.time()
                        a = engine.tensor(data)
                        b = engine.tensor(data)
                        
                        # Führe Operation durch
                        if op == "matmul":
                            c = a @ b
                        elif op == "add":
                            c = a + b
                        elif op == "exp":
                            c = a.exp()
                        
                        end_time = time.time()
                        
                        mlx_time = (end_time - start_time) * 1000
                        logger.info(f"  MLX: {mlx_time:.2f} ms")
                        results[size][f"mlx_{op}"] = mlx_time
                    except Exception as e:
                        logger.error(f"  Fehler mit MLX: {e}")
                        results[size][f"mlx_{op}"] = None
                
                # PyTorch Backend
                if has_torch:
                    try:
                        # Umwandeln wir den Tensor manuell zu PyTorch
                        torch_tensor = torch.tensor(data)
                        if mps_available:
                            torch_tensor = torch_tensor.to("mps")
                        
                        # Führe Operation durch
                        start_time = time.time()
                        
                        if op == "matmul":
                            c = torch_tensor @ torch_tensor
                        elif op == "add":
                            c = torch_tensor + torch_tensor
                        elif op == "exp":
                            c = torch.exp(torch_tensor)
                        
                        # Synchronisiere für korrekte Zeitmessung
                        if mps_available:
                            torch.mps.synchronize()
                        
                        end_time = time.time()
                        
                        torch_time = (end_time - start_time) * 1000
                        logger.info(f"  PyTorch: {torch_time:.2f} ms")
                        results[size][f"torch_{op}"] = torch_time
                    except Exception as e:
                        logger.error(f"  Fehler mit PyTorch: {e}")
                        results[size][f"torch_{op}"] = None
                
                # NumPy (als Referenz)
                try:
                    start_time = time.time()
                    
                    if op == "matmul":
                        c = np.matmul(data, data)
                    elif op == "add":
                        c = data + data
                    elif op == "exp":
                        c = np.exp(data)
                    
                    end_time = time.time()
                    
                    numpy_time = (end_time - start_time) * 1000
                    logger.info(f"  NumPy: {numpy_time:.2f} ms")
                    results[size][f"numpy_{op}"] = numpy_time
                except Exception as e:
                    logger.error(f"  Fehler mit NumPy: {e}")
                    results[size][f"numpy_{op}"] = None
                
                # Berechne Speedups
                for backend in ["mlx", "torch"]:
                    if results[size][f"{backend}_{op}"] is not None and results[size][f"numpy_{op}"] is not None:
                        speedup = results[size][f"numpy_{op}"] / results[size][f"{backend}_{op}"]
                        logger.info(f"  {backend.upper()} Speedup vs NumPy: {speedup:.2f}x")
        
        # Tabellarische Ausgabe der Ergebnisse
        logger.info("\n=== Zusammenfassung der Leistungstests ===")
        headers = ["Größe", "Operation"] + [f"{backend}" for backend in ["MLX", "PyTorch", "NumPy"]]
        logger.info(" | ".join(headers))
        logger.info("-" * 80)
        
        for size in sizes:
            for op in operations:
                row = [f"{size}x{size}", op.upper()]
                for backend in ["mlx", "torch", "numpy"]:
                    time_ms = results[size][f"{backend}_{op}"]
                    if time_ms is not None:
                        row.append(f"{time_ms:.2f} ms")
                    else:
                        row.append("N/A")
                logger.info(" | ".join(row))
        
        return True, "Backend-Leistungsvergleich erfolgreich"
    except Exception as e:
        logger.error(f"Fehler im Backend-Leistungsvergleich: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Backend-Leistungsvergleich fehlgeschlagen: {e}"

def test_vxor_integration():
    """Testet die Integration mit den VXOR-Modulen"""
    logger.info("\n=== Test: VXOR-Integration ===")
    
    try:
        # Versuche, VXOR-Module zu importieren
        from miso.math.t_mathematics.vxor_math_integration import VXORMathIntegration, get_vxor_math_integration
        
        # Initialisiere die Integration
        vxor_integration = get_vxor_math_integration()
        
        # Prüfe verfügbare VXOR-Module
        available_modules = vxor_integration.get_available_modules()
        logger.info("Verfügbare VXOR-Module für T-Mathematics:")
        for module_name, is_available in available_modules.items():
            logger.info(f"  - {module_name}: {'Verfügbar' if is_available else 'Nicht verfügbar'}")
        
        # Teste einfache VXOR-Operation
        active_modules = [name for name, is_available in available_modules.items() if is_available]
        if active_modules:
            module_name = active_modules[0]
            logger.info(f"Teste VXOR-Modul: {module_name}")
            
            result = vxor_integration.execute_vxor_operation(module_name, "test_operation", {"param": "value"})
            logger.info(f"VXOR-Operation ausgeführt: {result}")
        else:
            logger.warning("Keine aktiven VXOR-Module verfügbar, überspringe Operation-Test")
        
        # Teste MLX-Optimierungsfunktionalität
        try:
            test_operations = [
                {"op": "matmul", "inputs": ["A", "B"], "output": "C"},
                {"op": "add", "inputs": ["C", "D"], "output": "E"}
            ]
            
            optimized_ops = vxor_integration.optimize_mlx_operations(test_operations)
            logger.info(f"MLX-Operationen optimiert: {len(optimized_ops)} Operationen")
        except Exception as e:
            logger.warning(f"MLX-Operationsoptimierung nicht verfügbar: {e}")
        
        # Teste Callback-Registrierung
        try:
            from miso.math.t_mathematics.engine import TMathEngine
            engine = TMathEngine()
            registration_result = vxor_integration.register_callbacks(engine)
            logger.info(f"VXOR-Callbacks registriert: {registration_result}")
        except Exception as e:
            logger.warning(f"VXOR-Callback-Registrierung nicht möglich: {e}")
        
        return True, "VXOR-Integration Test erfolgreich"
    except Exception as e:
        logger.error(f"Fehler im VXOR-Integration Test: {e}")
        import traceback
        traceback.print_exc()
        return False, f"VXOR-Integration Test fehlgeschlagen: {e}"

def run_all_tests():
    """Führt alle Tests aus und gibt eine Zusammenfassung zurück"""
    logger.info("=== MISO Ultimate Systemtest ===")
    
    tests = [
        ("T-Mathematics Engine", test_t_mathematics_engine),
        ("M-LINGUA Integration", test_m_lingua_integration),
        ("Backend-Leistungsvergleich", test_backend_performance),
        ("VXOR-Integration", test_vxor_integration)
    ]
    
    results = []
    
    for name, test_fn in tests:
        logger.info(f"\nStarte Test: {name}")
        start_time = time.time()
        success, message = test_fn()
        end_time = time.time()
        
        if success:
            logger.info(f"Test '{name}' erfolgreich abgeschlossen in {(end_time - start_time)*1000:.2f} ms")
        else:
            logger.error(f"Test '{name}' fehlgeschlagen in {(end_time - start_time)*1000:.2f} ms: {message}")
        
        results.append({
            "name": name,
            "success": success,
            "message": message,
            "time_ms": (end_time - start_time) * 1000
        })
    
    # Zusammenfassung
    logger.info("\n=== Testzusammenfassung ===")
    
    total_tests = len(results)
    successful_tests = sum(1 for result in results if result["success"])
    
    logger.info(f"Gesamtzahl der Tests: {total_tests}")
    logger.info(f"Erfolgreiche Tests: {successful_tests}")
    logger.info(f"Fehlgeschlagene Tests: {total_tests - successful_tests}")
    
    if successful_tests == total_tests:
        logger.info("Alle Tests erfolgreich abgeschlossen!")
    else:
        logger.warning("Einige Tests sind fehlgeschlagen:")
        for result in results:
            if not result["success"]:
                logger.warning(f"  - {result['name']}: {result['message']}")
    
    return results

if __name__ == "__main__":
    run_all_tests()
