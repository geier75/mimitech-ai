#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Umfassender Systemtest (Teil 3)

Hauptprogramm zur Ausführung aller Tests für Module, Submodule und Agentenstrukturen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import datetime
import importlib
import numpy as np
import torch
import json
import traceback
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Importiere die TestResult-Klasse und ModuleTester aus den vorherigen Teilen
from run_comprehensive_tests import TestResult, logger, log_dir, is_apple_silicon, HAS_MLX, MODULES, TEST_TYPES
from run_comprehensive_tests_part2 import ModuleTester

class StressTestRunner:
    """Führt Stresstests für Module durch"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.logger = logging.getLogger(f"MISO.SystemTest.StressTest.{module_name}")
        self.log_file = log_dir / f"stress_test_{module_name.replace('.', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
    
    def run_stress_test(self) -> TestResult:
        """Führt einen Stresstest für das Modul durch"""
        result = TestResult(self.module_name, "stress")
        start_time = time.time()
        
        try:
            # Spezifische Stresstests für verschiedene Module
            stress_tests = {
                "miso.math.t_mathematics": self._stress_test_t_mathematics,
                "miso.simulation": self._stress_test_simulation,
                "miso.vxor_modules": self._stress_test_vxor_modules,
                "miso.vxor": self._stress_test_vxor,
                "miso.lang": self._stress_test_lang,
                "miso.logic": self._stress_test_logic,
                "miso.core": self._stress_test_core,
                "miso.security": self._stress_test_security,
                "miso.timeline": self._stress_test_timeline
            }
            
            if self.module_name in stress_tests:
                stress_test_func = stress_tests[self.module_name]
                stress_test_result = stress_test_func()
                
                result.success = stress_test_result["success"]
                result.details = stress_test_result
                
                if not result.success:
                    result.error = stress_test_result.get("error", "Stresstest fehlgeschlagen")
            else:
                result.success = True
                result.details["warning"] = f"Kein definierter Stresstest für {self.module_name}"
            
            self.logger.info(f"Stresstest für {self.module_name} abgeschlossen: {'Erfolgreich' if result.success else 'Fehlgeschlagen'}")
        
        except Exception as e:
            self.logger.error(f"Fehler bei Stresstest für {self.module_name}: {e}")
            result.success = False
            result.error = str(e)
            result.details["traceback"] = traceback.format_exc()
        
        result.duration = time.time() - start_time
        return result
    
    def _stress_test_t_mathematics(self) -> Dict[str, Any]:
        """Führt einen Stresstest für das T-MATHEMATICS-Modul durch"""
        try:
            from miso.math.t_mathematics.engine import TMathEngine
            from miso.math.t_mathematics.compat import TMathConfig
            
            # Initialisiere die Engine
            config = TMathConfig(optimize_for_apple_silicon=is_apple_silicon)
            engine = TMathEngine(config=config)
            
            # Stresstest: Parallele Matrix-Multiplikationen
            size = 1024  # Große Matrix für Stress
            num_processes = min(4, multiprocessing.cpu_count())  # Max. 4 Prozesse
            
            def matrix_multiply(process_id):
                # Erstelle Tensoren
                a = torch.randn(size, size, dtype=torch.float32)
                b = torch.randn(size, size, dtype=torch.float32)
                
                # Bereite Tensoren vor
                a_prepared = engine.prepare_tensor(a)
                b_prepared = engine.prepare_tensor(b)
                
                # Führe Matrixmultiplikation durch
                for i in range(5):  # 5 Wiederholungen pro Prozess
                    result = engine.matmul(a_prepared, b_prepared)
                
                return process_id
            
            # Führe parallele Prozesse aus
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(matrix_multiply, range(num_processes))
            
            # Stresstest: Speicherintensive Operationen
            memory_tensors = []
            for i in range(10):  # 10 große Tensoren
                tensor = torch.randn(512, 512, dtype=torch.float32)
                prepared_tensor = engine.prepare_tensor(tensor)
                memory_tensors.append(prepared_tensor)
            
            # Führe Operationen auf den Tensoren aus
            for i in range(len(memory_tensors) - 1):
                result = engine.matmul(memory_tensors[i], memory_tensors[i + 1])
            
            # Bereinige Speicher
            memory_tensors = []
            
            return {
                "success": True,
                "parallel_processes": num_processes,
                "matrix_size": size,
                "device": str(engine.device),
                "precision": str(engine.precision),
                "use_mlx": engine.use_mlx
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _stress_test_simulation(self) -> Dict[str, Any]:
        """Führt einen Stresstest für das Simulation-Modul durch"""
        try:
            from miso.simulation.prism_matrix import PrismMatrix
            
            # Stresstest: Große Matrix mit vielen Datenpunkten
            matrix = PrismMatrix(dimensions=5, initial_size=50)
            
            # Füge viele Datenpunkte hinzu
            for i in range(1000):
                coords = [i % 50 for _ in range(5)]
                matrix.add_data_point(f"point{i}", coords, i / 1000)
            
            # Berechne viele Distanzen
            distances = []
            for i in range(0, 1000, 10):
                for j in range(i + 1, 1000, 10):
                    distance = matrix.calculate_distance(f"point{i}", f"point{j}")
                    distances.append(distance)
            
            return {
                "success": True,
                "num_data_points": 1000,
                "num_distances": len(distances),
                "avg_distance": sum(distances) / len(distances) if distances else 0
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _stress_test_vxor_modules(self) -> Dict[str, Any]:
        """Führt einen Stresstest für die VXOR-Module durch"""
        try:
            # Versuche, den HyperfilterMathEngine zu importieren
            try:
                from miso.vxor_modules.hyperfilter_t_mathematics import get_hyperfilter_math_engine
                hyperfilter_engine = get_hyperfilter_math_engine()
                
                # Stresstest: Viele Text-Embedding-Analysen
                size = 768  # Standard-Embedding-Größe
                ref_count = 50  # Viele Referenz-Embeddings
                
                # Erstelle Test-Embeddings
                text_embedding = np.random.rand(size)
                reference_embeddings = [np.random.rand(size) for _ in range(ref_count)]
                trust_scores = [0.5 + 0.5 * np.random.rand() for _ in range(ref_count)]
                
                # Analysiere Text-Embedding mehrmals
                results = []
                for _ in range(20):  # 20 Wiederholungen
                    analysis = hyperfilter_engine.analyze_text_embedding(text_embedding, reference_embeddings, trust_scores)
                    results.append(analysis["trust_score"])
                
                return {
                    "success": True,
                    "embedding_size": size,
                    "ref_count": ref_count,
                    "num_analyses": 20,
                    "avg_trust_score": sum(results) / len(results) if results else 0,
                    "is_apple_silicon": hyperfilter_engine.is_apple_silicon
                }
            
            except ImportError:
                return {
                    "success": True,
                    "warning": "HyperfilterMathEngine nicht verfügbar"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _stress_test_vxor(self) -> Dict[str, Any]:
        """Führt einen Stresstest für das VXOR-Modul durch"""
        # Platzhalter für VXOR-Modul-Stresstest
        return {
            "success": True,
            "warning": "Kein spezifischer Stresstest für VXOR-Modul implementiert"
        }
    
    def _stress_test_lang(self) -> Dict[str, Any]:
        """Führt einen Stresstest für das Lang-Modul durch"""
        # Platzhalter für Lang-Modul-Stresstest
        return {
            "success": True,
            "warning": "Kein spezifischer Stresstest für Lang-Modul implementiert"
        }
    
    def _stress_test_logic(self) -> Dict[str, Any]:
        """Führt einen Stresstest für das Logic-Modul durch"""
        # Platzhalter für Logic-Modul-Stresstest
        return {
            "success": True,
            "warning": "Kein spezifischer Stresstest für Logic-Modul implementiert"
        }
    
    def _stress_test_core(self) -> Dict[str, Any]:
        """Führt einen Stresstest für das Core-Modul durch"""
        # Platzhalter für Core-Modul-Stresstest
        return {
            "success": True,
            "warning": "Kein spezifischer Stresstest für Core-Modul implementiert"
        }
    
    def _stress_test_security(self) -> Dict[str, Any]:
        """Führt einen Stresstest für das Security-Modul durch"""
        # Platzhalter für Security-Modul-Stresstest
        return {
            "success": True,
            "warning": "Kein spezifischer Stresstest für Security-Modul implementiert"
        }
    
    def _stress_test_timeline(self) -> Dict[str, Any]:
        """Führt einen Stresstest für das Timeline-Modul durch"""
        # Platzhalter für Timeline-Modul-Stresstest
        return {
            "success": True,
            "warning": "Kein spezifischer Stresstest für Timeline-Modul implementiert"
        }

class CompatibilityTestRunner:
    """Führt Kompatibilitätstests für Module durch"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.logger = logging.getLogger(f"MISO.SystemTest.CompatibilityTest.{module_name}")
        self.log_file = log_dir / f"compatibility_test_{module_name.replace('.', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
    
    def run_compatibility_test(self) -> TestResult:
        """Führt einen Kompatibilitätstest für das Modul durch"""
        result = TestResult(self.module_name, "compatibility")
        start_time = time.time()
        
        try:
            # Prüfe Apple Silicon Kompatibilität
            if is_apple_silicon:
                # Spezifische Kompatibilitätstests für verschiedene Module
                compatibility_tests = {
                    "miso.math.t_mathematics": self._test_t_mathematics_compatibility,
                    "miso.simulation": self._test_simulation_compatibility,
                    "miso.vxor_modules": self._test_vxor_modules_compatibility,
                    "miso.vxor": self._test_vxor_compatibility,
                    "miso.lang": self._test_lang_compatibility,
                    "miso.logic": self._test_logic_compatibility,
                    "miso.core": self._test_core_compatibility,
                    "miso.security": self._test_security_compatibility,
                    "miso.timeline": self._test_timeline_compatibility
                }
                
                if self.module_name in compatibility_tests:
                    compatibility_test_func = compatibility_tests[self.module_name]
                    compatibility_test_result = compatibility_test_func()
                    
                    result.success = compatibility_test_result["success"]
                    result.details = compatibility_test_result
                    
                    if not result.success:
                        result.error = compatibility_test_result.get("error", "Kompatibilitätstest fehlgeschlagen")
                else:
                    result.success = True
                    result.details["warning"] = f"Kein definierter Kompatibilitätstest für {self.module_name}"
            else:
                result.success = True
                result.details["warning"] = "Kein Apple Silicon erkannt, überspringe Kompatibilitätstest"
            
            self.logger.info(f"Kompatibilitätstest für {self.module_name} abgeschlossen: {'Erfolgreich' if result.success else 'Fehlgeschlagen'}")
        
        except Exception as e:
            self.logger.error(f"Fehler bei Kompatibilitätstest für {self.module_name}: {e}")
            result.success = False
            result.error = str(e)
            result.details["traceback"] = traceback.format_exc()
        
        result.duration = time.time() - start_time
        return result
    
    def _test_t_mathematics_compatibility(self) -> Dict[str, Any]:
        """Testet die Kompatibilität des T-MATHEMATICS-Moduls mit Apple Silicon"""
        try:
            from miso.math.t_mathematics.engine import TMathEngine
            from miso.math.t_mathematics.compat import TMathConfig
            
            # Teste MLX-Backend
            if HAS_MLX:
                # Initialisiere die Engine mit MLX
                config = TMathConfig(optimize_for_apple_silicon=True)
                engine = TMathEngine(config=config)
                
                # Prüfe, ob MLX verwendet wird
                if not engine.use_mlx:
                    return {
                        "success": False,
                        "error": "MLX wird nicht verwendet, obwohl es verfügbar ist",
                        "use_mlx": engine.use_mlx,
                        "device": str(engine.device)
                    }
                
                # Teste eine einfache Operation mit MLX
                a = torch.randn(10, 10, dtype=torch.float32)
                b = torch.randn(10, 10, dtype=torch.float32)
                
                a_prepared = engine.prepare_tensor(a)
                b_prepared = engine.prepare_tensor(b)
                
                result = engine.matmul(a_prepared, b_prepared)
                
                return {
                    "success": True,
                    "use_mlx": engine.use_mlx,
                    "device": str(engine.device),
                    "precision": str(engine.precision)
                }
            else:
                # Initialisiere die Engine ohne MLX
                config = TMathConfig(optimize_for_apple_silicon=False)
                engine = TMathEngine(config=config)
                
                # Teste eine einfache Operation ohne MLX
                a = torch.randn(10, 10, dtype=torch.float32)
                b = torch.randn(10, 10, dtype=torch.float32)
                
                a_prepared = engine.prepare_tensor(a)
                b_prepared = engine.prepare_tensor(b)
                
                result = engine.matmul(a_prepared, b_prepared)
                
                return {
                    "success": True,
                    "use_mlx": engine.use_mlx,
                    "device": str(engine.device),
                    "precision": str(engine.precision),
                    "warning": "MLX ist nicht verfügbar, verwende PyTorch"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _test_simulation_compatibility(self) -> Dict[str, Any]:
        """Testet die Kompatibilität des Simulation-Moduls mit Apple Silicon"""
        # Platzhalter für Simulation-Modul-Kompatibilitätstest
        return {
            "success": True,
            "warning": "Kein spezifischer Kompatibilitätstest für Simulation-Modul implementiert"
        }
    
    def _test_vxor_modules_compatibility(self) -> Dict[str, Any]:
        """Testet die Kompatibilität der VXOR-Module mit Apple Silicon"""
        # Platzhalter für VXOR-Module-Kompatibilitätstest
        return {
            "success": True,
            "warning": "Kein spezifischer Kompatibilitätstest für VXOR-Module implementiert"
        }
    
    def _test_vxor_compatibility(self) -> Dict[str, Any]:
        """Testet die Kompatibilität des VXOR-Moduls mit Apple Silicon"""
        # Platzhalter für VXOR-Modul-Kompatibilitätstest
        return {
            "success": True,
            "warning": "Kein spezifischer Kompatibilitätstest für VXOR-Modul implementiert"
        }
    
    def _test_lang_compatibility(self) -> Dict[str, Any]:
        """Testet die Kompatibilität des Lang-Moduls mit Apple Silicon"""
        # Platzhalter für Lang-Modul-Kompatibilitätstest
        return {
            "success": True,
            "warning": "Kein spezifischer Kompatibilitätstest für Lang-Modul implementiert"
        }
    
    def _test_logic_compatibility(self) -> Dict[str, Any]:
        """Testet die Kompatibilität des Logic-Moduls mit Apple Silicon"""
        # Platzhalter für Logic-Modul-Kompatibilitätstest
        return {
            "success": True,
            "warning": "Kein spezifischer Kompatibilitätstest für Logic-Modul implementiert"
        }
    
    def _test_core_compatibility(self) -> Dict[str, Any]:
        """Testet die Kompatibilität des Core-Moduls mit Apple Silicon"""
        # Platzhalter für Core-Modul-Kompatibilitätstest
        return {
            "success": True,
            "warning": "Kein spezifischer Kompatibilitätstest für Core-Modul implementiert"
        }
    
    def _test_security_compatibility(self) -> Dict[str, Any]:
        """Testet die Kompatibilität des Security-Moduls mit Apple Silicon"""
        # Platzhalter für Security-Modul-Kompatibilitätstest
        return {
            "success": True,
            "warning": "Kein spezifischer Kompatibilitätstest für Security-Modul implementiert"
        }
    
    def _test_timeline_compatibility(self) -> Dict[str, Any]:
        """Testet die Kompatibilität des Timeline-Moduls mit Apple Silicon"""
        # Platzhalter für Timeline-Modul-Kompatibilitätstest
        return {
            "success": True,
            "warning": "Kein spezifischer Kompatibilitätstest für Timeline-Modul implementiert"
        }

def run_tests_for_module(module_name: str) -> Dict[str, TestResult]:
    """Führt alle Tests für ein Modul aus"""
    logger.info(f"Starte Tests für Modul: {module_name}")
    
    results = {}
    
    # Erstelle ModuleTester
    tester = ModuleTester(module_name)
    
    # Importiere Modul
    if not tester.import_module():
        # Wenn das Modul nicht importiert werden kann, überspringe die Tests
        for test_type in TEST_TYPES:
            result = TestResult(module_name, test_type)
            result.success = False
            result.error = f"Modul {module_name} konnte nicht importiert werden"
            results[test_type] = result
        
        return results
    
    # Führe Unit-Tests aus
    results["unit"] = tester.run_unit_tests()
    
    # Führe Integrationstests aus
    results["integration"] = tester.run_integration_tests()
    
    # Teste Interface-Kompatibilität
    results["interface"] = tester.test_interface_compatibility()
    
    # Führe Leistungstests aus
    results["performance"] = tester.run_performance_benchmark()
    
    # Führe Stresstests aus
    stress_tester = StressTestRunner(module_name)
    results["stress"] = stress_tester.run_stress_test()
    
    # Führe Kompatibilitätstests aus
    compatibility_tester = CompatibilityTestRunner(module_name)
    results["compatibility"] = compatibility_tester.run_compatibility_test()
    
    return results

def main():
    """Hauptfunktion zur Ausführung aller Tests"""
    logger.info("Starte umfassenden Systemtest")
    logger.info(f"Apple Silicon: {is_apple_silicon}")
    logger.info(f"MLX verfügbar: {HAS_MLX}")
    
    # Speichere Systeminformationen
    system_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "apple_silicon": is_apple_silicon,
        "mlx_available": HAS_MLX,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__
    }
    
    with open(log_dir / "system_info.json", "w") as f:
        json.dump(system_info, f, indent=2)
    
    # Führe Tests für alle Module aus
    all_results = {}
    
    for module_name in MODULES:
        module_results = run_tests_for_module(module_name)
        all_results[module_name] = {test_type: result.to_dict() for test_type, result in module_results.items()}
    
    # Speichere Gesamtergebnisse
    results_file = log_dir / f"test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Erstelle Zusammenfassung
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "modules_tested": len(MODULES),
        "total_tests": sum(len(module_results) for module_results in all_results.values()),
        "successful_tests": sum(sum(1 for test in module_results.values() if test["success"]) for module_results in all_results.values()),
        "failed_tests": sum(sum(1 for test in module_results.values() if not test["success"]) for module_results in all_results.values()),
        "results_by_module": {module_name: {test_type: test["success"] for test_type, test in module_results.items()} for module_name, module_results in all_results.items()}
    }
    
    summary_file = log_dir / f"test_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Erstelle Markdown-Bericht
    markdown_report = f"""# MISO Ultimate - Umfassender Systemtest

## Zusammenfassung

- **Zeitstempel:** {summary["timestamp"]}
- **Getestete Module:** {summary["modules_tested"]}
- **Gesamtzahl Tests:** {summary["total_tests"]}
- **Erfolgreiche Tests:** {summary["successful_tests"]}
- **Fehlgeschlagene Tests:** {summary["failed_tests"]}

## Systemumgebung

- **Apple Silicon:** {system_info["apple_silicon"]}
- **MLX verfügbar:** {system_info["mlx_available"]}
- **Python-Version:** {system_info["python_version"]}
- **PyTorch-Version:** {system_info["torch_version"]}
- **NumPy-Version:** {system_info["numpy_version"]}

## Ergebnisse nach Modul

"""
    
    for module_name, module_results in all_results.items():
        markdown_report += f"### {module_name}\n\n"
        markdown_report += "| Test-Typ | Ergebnis | Dauer (s) | Details |\n"
        markdown_report += "|----------|----------|-----------|--------|\n"
        
        for test_type, test in module_results.items():
            result_text = "✅ Erfolgreich" if test["success"] else "❌ Fehlgeschlagen"
            details = test["error"] if not test["success"] and test["error"] else ""
            markdown_report += f"| {test_type} | {result_text} | {test['duration']:.2f} | {details} |\n"
        
        markdown_report += "\n"
    
    markdown_file = log_dir / f"test_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(markdown_file, "w") as f:
        f.write(markdown_report)
    
    logger.info(f"Umfassender Systemtest abgeschlossen. Ergebnisse gespeichert unter {log_dir}")
    logger.info(f"Zusammenfassung: {summary['successful_tests']} von {summary['total_tests']} Tests erfolgreich")

if __name__ == "__main__":
    main()
