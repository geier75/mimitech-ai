#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-Skript für die optimierte Tensor-Operation-Bridge zwischen VXOR-Komponenten.

Dieses Skript testet die folgenden Aspekte:
1. Initialisierung und Korrektheit des MISOTensorInterface
2. Funktionalität der TensorFactory für verschiedene Backends
3. Effizienz des Caching-Systems für wiederholte Operationen
4. Kompatibilität mit der VXOR-Integration

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.tests.tensor_bridge")

# Füge das Hauptverzeichnis zum Pfad hinzu, um Importe zu ermöglichen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importiere die nötigen Module
try:
    from miso.math.t_mathematics.tensor_interface import MISOTensorInterface
    from miso.math.t_mathematics.tensor_wrappers import MLXTensorWrapper, TorchTensorWrapper
    from miso.math.t_mathematics.tensor_factory import TensorFactory, tensor_factory
    from miso.math.t_mathematics.tensor_cache import TensorCacheManager, global_tensor_cache_manager
    from miso.math.t_mathematics.vxor_integration import TMathVXORIntegration
except ImportError as e:
    logger.error(f"Konnte die benötigten Module nicht importieren: {e}")
    sys.exit(1)

# Überprüfe, ob MLX oder PyTorch verfügbar sind
try:
    import mlx
    import mlx.core as mx
    MLX_AVAILABLE = True
    logger.info("MLX wurde erfolgreich importiert.")
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX ist nicht verfügbar.")

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch wurde erfolgreich importiert.")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch ist nicht verfügbar.")

if not MLX_AVAILABLE and not TORCH_AVAILABLE:
    logger.error("Weder MLX noch PyTorch sind verfügbar. Mindestens eines wird benötigt.")
    sys.exit(1)


def test_tensor_wrappers():
    """Testet die Tensor-Wrapper-Klassen für verschiedene Backends."""
    logger.info("=== Teste Tensor-Wrapper-Klassen ===")

    # Erstelle Test-Daten
    test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    
    if MLX_AVAILABLE:
        # Teste MLX Tensor Wrapper
        logger.info("Teste MLXTensorWrapper...")
        mlx_array = mx.array(test_data)
        mlx_wrapper = MLXTensorWrapper(mlx_array)
        
        # Teste Basismethoden
        assert mlx_wrapper.shape == (2, 3), f"Falsche Form: {mlx_wrapper.shape}"
        assert mlx_wrapper.dtype == "float32", f"Falscher Datentyp: {mlx_wrapper.dtype}"
        assert mlx_wrapper.backend == "mlx", f"Falsches Backend: {mlx_wrapper.backend}"
        
        # Teste arithmetische Operationen
        result_add = mlx_wrapper + mlx_wrapper
        assert isinstance(result_add, MLXTensorWrapper), "Addition sollte einen MLXTensorWrapper zurückgeben"
        assert result_add.shape == (2, 3), f"Falsche Form nach Addition: {result_add.shape}"
        
        # Teste Serialisierung und Deserialisierung
        serialized = mlx_wrapper.serialize()
        deserialized = MLXTensorWrapper.deserialize(serialized)
        
        # Überprüfe, ob die Deserialisierung korrekt funktioniert
        assert deserialized.shape == mlx_wrapper.shape, "Shape stimmt nach Deserialisierung nicht überein"
        assert deserialized.dtype == mlx_wrapper.dtype, "Dtype stimmt nach Deserialisierung nicht überein"
        
        logger.info("MLXTensorWrapper-Tests erfolgreich abgeschlossen.")
    
    if TORCH_AVAILABLE:
        # Teste PyTorch Tensor Wrapper
        logger.info("Teste TorchTensorWrapper...")
        torch_tensor = torch.tensor(test_data)
        torch_wrapper = TorchTensorWrapper(torch_tensor)
        
        # Teste Basismethoden
        assert torch_wrapper.shape == (2, 3), f"Falsche Form: {torch_wrapper.shape}"
        assert torch_wrapper.dtype == "float32", f"Falscher Datentyp: {torch_wrapper.dtype}"
        assert torch_wrapper.backend == "torch", f"Falsches Backend: {torch_wrapper.backend}"
        
        # Teste arithmetische Operationen
        result_add = torch_wrapper + torch_wrapper
        assert isinstance(result_add, TorchTensorWrapper), "Addition sollte einen TorchTensorWrapper zurückgeben"
        assert result_add.shape == (2, 3), f"Falsche Form nach Addition: {result_add.shape}"
        
        # Teste Serialisierung und Deserialisierung
        serialized = torch_wrapper.serialize()
        deserialized = TorchTensorWrapper.deserialize(serialized)
        
        # Überprüfe, ob die Deserialisierung korrekt funktioniert
        assert deserialized.shape == torch_wrapper.shape, "Shape stimmt nach Deserialisierung nicht überein"
        assert deserialized.dtype == torch_wrapper.dtype, "Dtype stimmt nach Deserialisierung nicht überein"
        
        logger.info("TorchTensorWrapper-Tests erfolgreich abgeschlossen.")


def test_tensor_factory():
    """Testet die TensorFactory-Klasse."""
    logger.info("=== Teste TensorFactory ===")
    
    # Erstelle Test-Daten
    test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    
    # Teste Erstellung von Tensoren für verschiedene Backends
    if MLX_AVAILABLE:
        logger.info("Teste Tensor-Erstellung für MLX...")
        mlx_tensor = tensor_factory.create_tensor(test_data, "mlx")
        assert isinstance(mlx_tensor, MLXTensorWrapper), "Sollte ein MLXTensorWrapper sein"
        assert mlx_tensor.shape == (2, 3), f"Falsche Form: {mlx_tensor.shape}"
    
    if TORCH_AVAILABLE:
        logger.info("Teste Tensor-Erstellung für PyTorch...")
        torch_tensor = tensor_factory.create_tensor(test_data, "torch")
        assert isinstance(torch_tensor, TorchTensorWrapper), "Sollte ein TorchTensorWrapper sein"
        assert torch_tensor.shape == (2, 3), f"Falsche Form: {torch_tensor.shape}"
    
    # Teste Konvertierung zwischen Backends, falls beide verfügbar sind
    if MLX_AVAILABLE and TORCH_AVAILABLE:
        logger.info("Teste Konvertierung zwischen Backends...")
        
        # MLX -> PyTorch
        mlx_tensor = tensor_factory.create_tensor(test_data, "mlx")
        converted_to_torch = tensor_factory.convert_tensor(mlx_tensor, "torch")
        assert isinstance(converted_to_torch, TorchTensorWrapper), "Sollte ein TorchTensorWrapper sein"
        assert converted_to_torch.shape == (2, 3), f"Falsche Form nach Konvertierung: {converted_to_torch.shape}"
        
        # PyTorch -> MLX
        torch_tensor = tensor_factory.create_tensor(test_data, "torch")
        converted_to_mlx = tensor_factory.convert_tensor(torch_tensor, "mlx")
        assert isinstance(converted_to_mlx, MLXTensorWrapper), "Sollte ein MLXTensorWrapper sein"
        assert converted_to_mlx.shape == (2, 3), f"Falsche Form nach Konvertierung: {converted_to_mlx.shape}"
    
    logger.info("TensorFactory-Tests erfolgreich abgeschlossen.")


def test_tensor_cache():
    """Testet das Tensor-Cache-System."""
    logger.info("=== Teste Tensor-Cache-System ===")
    
    # Erstelle einen neuen Cache-Manager für den Test
    cache_manager = TensorCacheManager(max_cache_size=100, default_ttl=60)
    
    # Erstelle Test-Daten
    test_data1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    test_data2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    
    # Wähle ein verfügbares Backend
    backend = "mlx" if MLX_AVAILABLE else "torch"
    
    # Erstelle Tensoren
    tensor1 = tensor_factory.create_tensor(test_data1, backend)
    tensor2 = tensor_factory.create_tensor(test_data2, backend)
    
    # Operation simulieren
    operation = "matrix_multiply"
    params = {"optimization_level": 3}
    
    # Simuliere ein Ergebnis (hier einfach die Addition der Tensoren)
    result = tensor1 + tensor2
    
    # Speichere das Ergebnis im Cache
    cache_manager.set(operation, [tensor1, tensor2], result, params)
    
    # Überprüfe, ob das Ergebnis im Cache ist
    cached_result = cache_manager.get(operation, [tensor1, tensor2], params)
    assert cached_result is not None, "Ergebnis sollte im Cache sein"
    assert cached_result.shape == result.shape, "Form des gecachten Ergebnisses stimmt nicht überein"
    
    # Teste Cache-Statistiken
    stats = cache_manager.get_stats()
    assert stats["size"] == 1, f"Cache sollte 1 Eintrag haben, hat aber {stats['size']}"
    assert stats["hits"] == 1, f"Cache sollte 1 Hit haben, hat aber {stats['hits']}"
    
    # Teste Cache-Bereinigung
    cache_manager.cleanup()
    
    logger.info("Tensor-Cache-Tests erfolgreich abgeschlossen.")


def test_vxor_integration():
    """Testet die Integration mit der VXOR-Komponente."""
    logger.info("=== Teste VXOR-Integration mit Cache ===")
    
    # Initialisiere die VXOR-Integration
    vxor_integration = TMathVXORIntegration()
    
    # Erstelle Test-Daten
    test_data1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    test_data2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    
    # Wähle ein verfügbares Backend
    backend = "mlx" if MLX_AVAILABLE else "torch"
    
    # Erstelle Tensoren
    tensor1 = tensor_factory.create_tensor(test_data1, backend)
    tensor2 = tensor_factory.create_tensor(test_data2, backend)
    
    # Führe eine Addition aus und messe die Zeit
    logger.info("Führe erste Operation durch (sollte Cache-Miss sein)...")
    start_time = time.time()
    result1 = vxor_integration.compute_tensor_operation("add", [tensor1, tensor2], backend)
    first_op_time = time.time() - start_time
    logger.info(f"Erste Operation dauerte {first_op_time:.6f} Sekunden")
    
    # Führe dieselbe Operation erneut aus (sollte aus dem Cache kommen)
    logger.info("Führe dieselbe Operation erneut durch (sollte Cache-Hit sein)...")
    start_time = time.time()
    result2 = vxor_integration.compute_tensor_operation("add", [tensor1, tensor2], backend)
    second_op_time = time.time() - start_time
    logger.info(f"Zweite Operation dauerte {second_op_time:.6f} Sekunden")
    
    # Überprüfe, ob die zweite Operation schneller war
    assert second_op_time < first_op_time, "Zweite Operation sollte schneller sein (Cache-Hit)"
    
    # Überprüfe die Performance-Metriken
    metrics = vxor_integration.performance_metrics
    assert metrics["operations_count"] >= 2, f"Operation-Count sollte mindestens 2 sein, ist aber {metrics['operations_count']}"
    assert metrics["cache_hits"] >= 1, f"Cache-Hits sollten mindestens 1 sein, sind aber {metrics['cache_hits']}"
    
    logger.info(f"Performance-Metriken: {metrics}")
    logger.info("VXOR-Integration-Tests erfolgreich abgeschlossen.")


def run_tests():
    """Führt alle Tests aus."""
    logger.info("Starte Tests für die Tensor-Operation-Bridge...")
    
    try:
        test_tensor_wrappers()
        test_tensor_factory()
        test_tensor_cache()
        test_vxor_integration()
        
        logger.info("Alle Tests erfolgreich abgeschlossen!")
        return True
    except AssertionError as e:
        logger.error(f"Test fehlgeschlagen: {e}")
        return False
    except Exception as e:
        logger.error(f"Unerwarteter Fehler während des Tests: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
