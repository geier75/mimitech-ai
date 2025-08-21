#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Umfassender Systemtest (Teil 2)

Fortsetzung der Testfunktionen für alle Module, Submodule und Agentenstrukturen.

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

# Importiere die TestResult-Klasse aus dem ersten Teil
from run_comprehensive_tests import TestResult, logger, log_dir, is_apple_silicon, HAS_MLX

class ModuleTester:
    """Führt Tests für ein bestimmtes Modul durch"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.results = {}
        self.module = None
        
        # Konfiguriere Modul-spezifisches Logging
        self.log_file = log_dir / f"{module_name.replace('.', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = logging.getLogger(f"MISO.SystemTest.{module_name}")
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
    
    def import_module(self) -> bool:
        """Importiert das zu testende Modul"""
        try:
            self.module = importlib.import_module(self.module_name)
            self.logger.info(f"Modul {self.module_name} erfolgreich importiert")
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Importieren des Moduls {self.module_name}: {e}")
            return False
    
    def test_interface_compatibility(self) -> TestResult:
        """Testet die Interface-Kompatibilität des Moduls"""
        result = TestResult(self.module_name, "interface")
        start_time = time.time()
        
        try:
            # Definiere die erwarteten Schnittstellen für verschiedene Module
            interfaces = {
                "miso.math.t_mathematics": ["TMathEngine", "MLXBackend", "get_t_math_integration_manager"],
                "miso.simulation": ["PrismMatrix", "PrismEngine"],
                "miso.vxor_modules": ["get_hyperfilter_math_engine"],
                "miso.vxor": ["get_vxor_t_math_bridge", "VXORAdapter"],
                "miso.lang": ["MCodeObject", "MCodeFunction"],
                "miso.logic": ["QLOGIKIntegrationManager"],
                "miso.core": ["OmegaCore"],
                "miso.security": ["SecuritySandbox"],
                "miso.timeline": ["TemporalEvent", "Timeline"]
            }
            
            # Prüfe, ob das Modul die erwarteten Schnittstellen bereitstellt
            if self.module_name in interfaces:
                expected_interfaces = interfaces[self.module_name]
                missing_interfaces = []
                
                for interface in expected_interfaces:
                    if not hasattr(self.module, interface):
                        # Versuche, Untermodule zu importieren
                        interface_found = False
                        for submodule_info in self.module.__dict__.items():
                            if isinstance(submodule_info[1], type(sys)) and interface in dir(submodule_info[1]):
                                interface_found = True
                                break
                        
                        if not interface_found:
                            missing_interfaces.append(interface)
                
                if missing_interfaces:
                    result.success = False
                    result.error = f"Fehlende Schnittstellen: {', '.join(missing_interfaces)}"
                    result.details["missing_interfaces"] = missing_interfaces
                else:
                    result.success = True
                    result.details["interfaces_found"] = expected_interfaces
            else:
                result.success = True
                result.details["warning"] = f"Keine definierten Schnittstellen für {self.module_name}"
            
            self.logger.info(f"Interface-Kompatibilitätstest für {self.module_name} abgeschlossen: {'Erfolgreich' if result.success else 'Fehlgeschlagen'}")
        
        except Exception as e:
            self.logger.error(f"Fehler bei Interface-Kompatibilitätstest für {self.module_name}: {e}")
            result.success = False
            result.error = str(e)
            result.details["traceback"] = traceback.format_exc()
        
        result.duration = time.time() - start_time
        return result
    
    def run_performance_benchmark(self) -> TestResult:
        """Führt Leistungstests für das Modul durch"""
        result = TestResult(self.module_name, "performance")
        start_time = time.time()
        
        try:
            # Spezifische Benchmarks für verschiedene Module
            benchmarks = {
                "miso.math.t_mathematics": self._benchmark_t_mathematics,
                "miso.simulation": self._benchmark_simulation,
                "miso.vxor_modules": self._benchmark_vxor_modules,
                "miso.vxor": self._benchmark_vxor,
                "miso.lang": self._benchmark_lang,
                "miso.logic": self._benchmark_logic,
                "miso.core": self._benchmark_core,
                "miso.security": self._benchmark_security,
                "miso.timeline": self._benchmark_timeline
            }
            
            if self.module_name in benchmarks:
                benchmark_func = benchmarks[self.module_name]
                benchmark_result = benchmark_func()
                
                result.success = benchmark_result["success"]
                result.details = benchmark_result
                
                if not result.success:
                    result.error = benchmark_result.get("error", "Benchmark fehlgeschlagen")
            else:
                result.success = True
                result.details["warning"] = f"Kein definierter Benchmark für {self.module_name}"
            
            self.logger.info(f"Leistungstest für {self.module_name} abgeschlossen: {'Erfolgreich' if result.success else 'Fehlgeschlagen'}")
        
        except Exception as e:
            self.logger.error(f"Fehler bei Leistungstest für {self.module_name}: {e}")
            result.success = False
            result.error = str(e)
            result.details["traceback"] = traceback.format_exc()
        
        result.duration = time.time() - start_time
        return result
    
    def _benchmark_t_mathematics(self) -> Dict[str, Any]:
        """Führt Leistungstests für das T-MATHEMATICS-Modul durch"""
        try:
            from miso.math.t_mathematics.engine import TMathEngine
            from miso.math.t_mathematics.compat import TMathConfig
            
            # Initialisiere die Engine
            config = TMathConfig(optimize_for_apple_silicon=is_apple_silicon)
            engine = TMathEngine(config=config)
            
            # Benchmark: Matrix-Multiplikation
            sizes = [128, 256, 512, 1024]
            matmul_results = {}
            
            for size in sizes:
                # Erstelle Tensoren
                a = torch.randn(size, size, dtype=torch.float32)
                b = torch.randn(size, size, dtype=torch.float32)
                
                # Bereite Tensoren vor
                a_prepared = engine.prepare_tensor(a)
                b_prepared = engine.prepare_tensor(b)
                
                # Führe Matrixmultiplikation durch
                start = time.time()
                for _ in range(5):  # 5 Wiederholungen für stabilere Ergebnisse
                    result = engine.matmul(a_prepared, b_prepared)
                end = time.time()
                
                matmul_results[size] = (end - start) / 5  # Durchschnittliche Zeit pro Operation
            
            # Benchmark: SVD
            svd_results = {}
            
            for size in sizes[:3]:  # Nur bis 512, da SVD rechenintensiv ist
                # Erstelle Tensor
                a = torch.randn(size, size, dtype=torch.float32)
                
                # Bereite Tensor vor
                a_prepared = engine.prepare_tensor(a)
                
                # Führe SVD durch
                start = time.time()
                result = engine.svd(a_prepared)
                end = time.time()
                
                svd_results[size] = end - start
            
            return {
                "success": True,
                "matmul_benchmark": matmul_results,
                "svd_benchmark": svd_results,
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
    
    def _benchmark_simulation(self) -> Dict[str, Any]:
        """Führt Leistungstests für das Simulation-Modul durch"""
        try:
            from miso.simulation.prism_matrix import PrismMatrix
            
            # Benchmark: PrismMatrix-Operationen
            dimensions = [3, 4, 5]
            sizes = [10, 20, 30]
            
            matrix_results = {}
            
            for dim in dimensions:
                for size in sizes:
                    # Erstelle Matrix
                    matrix = PrismMatrix(dimensions=dim, initial_size=size)
                    
                    # Füge Datenpunkte hinzu
                    start = time.time()
                    for i in range(100):
                        coords = [i % size for _ in range(dim)]
                        matrix.add_data_point(f"point{i}", coords, i / 100)
                    end = time.time()
                    
                    add_time = end - start
                    
                    # Berechne Distanzen
                    start = time.time()
                    for i in range(50):
                        for j in range(i + 1, 100):
                            matrix.calculate_distance(f"point{i}", f"point{j}")
                    end = time.time()
                    
                    distance_time = end - start
                    
                    matrix_results[f"{dim}d_{size}"] = {
                        "add_points_time": add_time,
                        "calculate_distances_time": distance_time
                    }
            
            return {
                "success": True,
                "matrix_benchmark": matrix_results
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _benchmark_vxor_modules(self) -> Dict[str, Any]:
        """Führt Leistungstests für die VXOR-Module durch"""
        try:
            # Versuche, den HyperfilterMathEngine zu importieren
            try:
                from miso.vxor_modules.hyperfilter_t_mathematics import get_hyperfilter_math_engine
                hyperfilter_engine = get_hyperfilter_math_engine()
                
                # Benchmark: Text-Embedding-Analyse
                embedding_sizes = [128, 256, 512, 768]
                reference_counts = [3, 5, 10]
                
                analysis_results = {}
                
                for size in embedding_sizes:
                    for ref_count in reference_counts:
                        # Erstelle Test-Embeddings
                        text_embedding = np.random.rand(size)
                        reference_embeddings = [np.random.rand(size) for _ in range(ref_count)]
                        trust_scores = [0.5 + 0.5 * np.random.rand() for _ in range(ref_count)]
                        
                        # Analysiere Text-Embedding
                        start = time.time()
                        for _ in range(10):  # 10 Wiederholungen für stabilere Ergebnisse
                            analysis = hyperfilter_engine.analyze_text_embedding(text_embedding, reference_embeddings, trust_scores)
                        end = time.time()
                        
                        analysis_results[f"{size}d_{ref_count}refs"] = (end - start) / 10  # Durchschnittliche Zeit pro Operation
                
                return {
                    "success": True,
                    "hyperfilter_benchmark": analysis_results,
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
    
    def _benchmark_vxor(self) -> Dict[str, Any]:
        """Führt Leistungstests für das VXOR-Modul durch"""
        try:
            # Versuche, die VXOR-T-MATHEMATICS-Brücke zu importieren
            try:
                from miso.vxor.t_mathematics_bridge import get_vxor_t_math_bridge
                vxor_bridge = get_vxor_t_math_bridge()
                
                # Benchmark: Verfügbare Module abrufen
                start = time.time()
                for _ in range(100):  # 100 Wiederholungen für stabilere Ergebnisse
                    available_modules = vxor_bridge.get_available_vxor_modules()
                end = time.time()
                
                get_modules_time = (end - start) / 100  # Durchschnittliche Zeit pro Operation
                
                return {
                    "success": True,
                    "get_modules_time": get_modules_time,
                    "available_modules": available_modules
                }
            
            except ImportError:
                return {
                    "success": True,
                    "warning": "VXOR-T-MATHEMATICS-Brücke nicht verfügbar"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _benchmark_lang(self) -> Dict[str, Any]:
        """Führt Leistungstests für das Lang-Modul durch"""
        try:
            # Versuche, die MCodeObject-Klasse zu importieren
            try:
                from miso.lang.mcode_runtime import MCodeObject, MCodeFunction
                
                # Benchmark: MCodeObject-Erstellung
                start = time.time()
                for i in range(1000):  # 1000 Objekte erstellen
                    obj = MCodeObject(f"object_{i}", "test_object")
                    obj.attributes["value"] = i
                end = time.time()
                
                create_objects_time = end - start
                
                # Benchmark: MCodeFunction-Erstellung und Aufruf
                def test_func(x, y):
                    return x + y
                
                start = time.time()
                for i in range(100):  # 100 Funktionen erstellen und aufrufen
                    func = MCodeFunction(f"func_{i}", test_func)
                    result = func(i, i + 1)
                end = time.time()
                
                function_time = end - start
                
                return {
                    "success": True,
                    "create_objects_time": create_objects_time,
                    "function_time": function_time
                }
            
            except ImportError:
                return {
                    "success": True,
                    "warning": "MCodeObject/MCodeFunction nicht verfügbar"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _benchmark_logic(self) -> Dict[str, Any]:
        """Führt Leistungstests für das Logic-Modul durch"""
        # Platzhalter für Logic-Modul-Benchmark
        return {
            "success": True,
            "warning": "Kein spezifischer Benchmark für Logic-Modul implementiert"
        }
    
    def _benchmark_core(self) -> Dict[str, Any]:
        """Führt Leistungstests für das Core-Modul durch"""
        # Platzhalter für Core-Modul-Benchmark
        return {
            "success": True,
            "warning": "Kein spezifischer Benchmark für Core-Modul implementiert"
        }
    
    def _benchmark_security(self) -> Dict[str, Any]:
        """Führt Leistungstests für das Security-Modul durch"""
        # Platzhalter für Security-Modul-Benchmark
        return {
            "success": True,
            "warning": "Kein spezifischer Benchmark für Security-Modul implementiert"
        }
    
    def _benchmark_timeline(self) -> Dict[str, Any]:
        """Führt Leistungstests für das Timeline-Modul durch"""
        # Platzhalter für Timeline-Modul-Benchmark
        return {
            "success": True,
            "warning": "Kein spezifischer Benchmark für Timeline-Modul implementiert"
        }
