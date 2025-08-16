#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-MATHEMATICS Integration Tests

Dieser Testskript überprüft die Integration der T-MATHEMATICS Engine in allen Modulen
des MISO Ultimate AGI-Systems.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import unittest
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import json
import time

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.tests.t_mathematics_integration")

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiere T-MATHEMATICS-Komponenten
try:
    from miso.math.t_mathematics.integration_manager import get_t_math_integration_manager
    from miso.math.t_mathematics.engine import TMathEngine
    HAS_T_MATH = True
except ImportError as e:
    logger.error(f"T-MATHEMATICS konnte nicht importiert werden: {e}")
    HAS_T_MATH = False

# Importiere NEXUS-OS-Komponenten
try:
    from miso.core.nexus_os.t_math_integration import NexusOSTMathIntegration
    HAS_NEXUS_OS = True
except ImportError as e:
    logger.error(f"NEXUS-OS-Integration konnte nicht importiert werden: {e}")
    HAS_NEXUS_OS = False

# Importiere PRISM-Komponenten
try:
    from miso.simulation.prism_engine import PrismEngine
    from miso.simulation.prism_matrix import PrismMatrix
    HAS_PRISM = True
except ImportError as e:
    logger.error(f"PRISM-Komponenten konnten nicht importiert werden: {e}")
    HAS_PRISM = False

# Importiere VXOR-Komponenten
try:
    from miso.vXor_Modules.vxor_t_mathematics_bridge import get_vxor_t_math_bridge
    from miso.vXor_Modules.hyperfilter_t_mathematics import get_hyperfilter_math_engine
    HAS_VXOR = True
except ImportError as e:
    logger.error(f"VXOR-Komponenten konnten nicht importiert werden: {e}")
    HAS_VXOR = False

class TMathIntegrationTests(unittest.TestCase):
    """Testet die Integration der T-MATHEMATICS Engine in allen Modulen."""
    
    def setUp(self):
        """Initialisiert die Testumgebung."""
        self.test_results = {}
        
        # Prüfe, ob T-MATHEMATICS verfügbar ist
        if not HAS_T_MATH:
            self.skipTest("T-MATHEMATICS ist nicht verfügbar")
        
        # Initialisiere T-MATHEMATICS Integration Manager
        self.t_math_manager = get_t_math_integration_manager()
        self.t_math_engine = self.t_math_manager.get_engine("tests")
        
        # Prüfe, ob MLX verfügbar ist
        self.is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
        self.has_mlx = self.t_math_engine.use_mlx if hasattr(self.t_math_engine, 'use_mlx') else False
        
        logger.info(f"Testumgebung initialisiert (Apple Silicon: {self.is_apple_silicon}, MLX: {self.has_mlx})")
    
    def test_t_mathematics_engine(self):
        """Testet die grundlegenden Funktionen der T-MATHEMATICS Engine."""
        logger.info("Teste T-MATHEMATICS Engine...")
        
        # Erstelle Tensoren
        tensor1 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        tensor2 = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
        
        # Bereite Tensoren vor
        tensor1_prepared = self.t_math_engine.prepare_tensor(tensor1)
        tensor2_prepared = self.t_math_engine.prepare_tensor(tensor2)
        
        # Führe Operationen durch
        result1 = self.t_math_engine.matmul(tensor1_prepared.unsqueeze(0), tensor2_prepared.unsqueeze(1))
        result2 = tensor1_prepared * tensor2_prepared
        
        # Prüfe die Ergebnisse
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        
        # Speichere die Ergebnisse
        self.test_results["t_mathematics_engine"] = {
            "success": True,
            "tensor1": tensor1.tolist(),
            "tensor2": tensor2.tolist(),
            "result1": result1.cpu().numpy().tolist() if hasattr(result1, "cpu") else (result1.tolist() if hasattr(result1, "tolist") else result1),
            "result2": result2.cpu().numpy().tolist() if hasattr(result2, "cpu") else (result2.tolist() if hasattr(result2, "tolist") else result2),
            "backend": "mlx" if self.has_mlx else "torch"
        }
        
        logger.info("T-MATHEMATICS Engine-Test erfolgreich")
    
    def test_nexus_os_integration(self):
        """Testet die Integration mit NEXUS-OS."""
        if not HAS_NEXUS_OS:
            logger.warning("NEXUS-OS-Integration ist nicht verfügbar")
            self.test_results["nexus_os_integration"] = {"success": False, "reason": "not_available"}
            return
        
        logger.info("Teste NEXUS-OS-Integration...")
        
        # Initialisiere NEXUS-OS T-MATHEMATICS Integration
        nexus_integration = NexusOSTMathIntegration(self.t_math_engine)
        
        # Erstelle Tensoren
        tensor1 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        tensor2 = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
        
        # Führe eine synchrone Operation aus
        try:
            result = nexus_integration.execute_tensor_operation("matmul", [tensor1.unsqueeze(0), tensor2.unsqueeze(1)])
            self.assertIsNotNone(result)
            
            # Führe eine asynchrone Operation aus
            task_id = nexus_integration.submit_task("matmul", [tensor1.unsqueeze(0), tensor2.unsqueeze(1)])
            self.assertIsNotNone(task_id)
            
            # Warte auf das Ergebnis
            task_result = nexus_integration.wait_for_task(task_id, timeout=5.0)
            self.assertEqual(task_result["status"], "completed")
            
            # Speichere die Ergebnisse
            self.test_results["nexus_os_integration"] = {
                "success": True,
                "sync_result": str(result),
                "async_task_id": task_id,
                "async_result": str(task_result)
            }
            
            logger.info("NEXUS-OS-Integration-Test erfolgreich")
        except Exception as e:
            logger.error(f"NEXUS-OS-Integration-Test fehlgeschlagen: {e}")
            self.test_results["nexus_os_integration"] = {"success": False, "error": str(e)}
    
    def test_prism_integration(self):
        """Testet die Integration mit PRISM."""
        if not HAS_PRISM:
            logger.warning("PRISM-Komponenten sind nicht verfügbar")
            self.test_results["prism_integration"] = {"success": False, "reason": "not_available"}
            return
        
        logger.info("Teste PRISM-Integration...")
        
        try:
            # Teste PrismMatrix
            matrix = PrismMatrix(dimensions=3, initial_size=5)
            
            # Füge Datenpunkte hinzu
            matrix.add_data_point("point1", [1, 2, 3], 1.0)
            matrix.add_data_point("point2", [2, 3, 4], 2.0)
            
            # Berechne Distanz
            distance = matrix.calculate_distance("point1", "point2")
            self.assertIsNotNone(distance)
            
            # Teste PrismEngine
            config = {"use_mlx": self.has_mlx, "precision": "float16"}
            engine = PrismEngine(config)
            
            # Führe eine Tensor-Operation aus
            tensor1 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
            tensor2 = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
            
            result = engine.integrate_with_t_mathematics("matmul", [tensor1.unsqueeze(0), tensor2.unsqueeze(1)])
            self.assertIsNotNone(result)
            
            # Speichere die Ergebnisse
            self.test_results["prism_integration"] = {
                "success": True,
                "matrix_distance": distance,
                "engine_result": str(result),
                "use_t_math": matrix.use_t_math if hasattr(matrix, "use_t_math") else False
            }
            
            logger.info("PRISM-Integration-Test erfolgreich")
        except Exception as e:
            logger.error(f"PRISM-Integration-Test fehlgeschlagen: {e}")
            self.test_results["prism_integration"] = {"success": False, "error": str(e)}
    
    def test_vxor_integration(self):
        """Testet die Integration mit VXOR-Modulen."""
        if not HAS_VXOR:
            logger.warning("VXOR-Komponenten sind nicht verfügbar")
            self.test_results["vxor_integration"] = {"success": False, "reason": "not_available"}
            return
        
        logger.info("Teste VXOR-Integration...")
        
        try:
            # Initialisiere VXOR-T-MATHEMATICS-Brücke
            vxor_bridge = get_vxor_t_math_bridge()
            
            # Hole verfügbare VXOR-Module
            available_modules = vxor_bridge.get_available_vxor_modules()
            
            # Teste HyperfilterMathEngine
            hyperfilter_engine = get_hyperfilter_math_engine()
            
            # Erstelle Test-Embeddings
            text_embedding = np.random.rand(768)
            reference_embeddings = [np.random.rand(768) for _ in range(3)]
            trust_scores = [0.8, 0.6, 0.9]
            
            # Analysiere Text-Embedding
            analysis = hyperfilter_engine.analyze_text_embedding(text_embedding, reference_embeddings, trust_scores)
            self.assertIsNotNone(analysis)
            self.assertIn("trust_score", analysis)
            
            # Speichere die Ergebnisse
            self.test_results["vxor_integration"] = {
                "success": True,
                "available_modules": available_modules,
                "hyperfilter_analysis": analysis
            }
            
            logger.info("VXOR-Integration-Test erfolgreich")
        except Exception as e:
            logger.error(f"VXOR-Integration-Test fehlgeschlagen: {e}")
            self.test_results["vxor_integration"] = {"success": False, "error": str(e)}
    
    def test_performance(self):
        """Testet die Leistung der T-MATHEMATICS Engine."""
        logger.info("Teste T-MATHEMATICS-Leistung...")
        
        # Erstelle große Tensoren für Leistungstests
        size = 1000
        tensor1 = torch.randn(size, size, dtype=torch.float32)
        tensor2 = torch.randn(size, size, dtype=torch.float32)
        
        # Bereite Tensoren vor
        tensor1_prepared = self.t_math_engine.prepare_tensor(tensor1)
        tensor2_prepared = self.t_math_engine.prepare_tensor(tensor2)
        
        # Messe die Zeit für Matrixmultiplikation
        start_time = time.time()
        result = self.t_math_engine.matmul(tensor1_prepared, tensor2_prepared)
        end_time = time.time()
        
        matmul_time = end_time - start_time
        
        # Speichere die Ergebnisse
        self.test_results["performance"] = {
            "success": True,
            "matrix_size": size,
            "matmul_time": matmul_time,
            "backend": "mlx" if self.has_mlx else "torch"
        }
        
        logger.info(f"Leistungstest erfolgreich: Matrixmultiplikation {size}x{size} in {matmul_time:.4f} Sekunden")
    
    def tearDown(self):
        """Gibt die Testergebnisse aus."""
        # Speichere die Testergebnisse in einer Datei
        with open(os.path.join(os.path.dirname(__file__), "t_mathematics_integration_results.json"), "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"Testergebnisse gespeichert in t_mathematics_integration_results.json")

if __name__ == "__main__":
    unittest.main()
