#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Einfacher Test für die optimierte Tensor-Operation-Bridge.

Dieser Test konzentriert sich auf die Kernfunktionalitäten des neuen Tensor-Systems:
1. MISOTensorInterface
2. MLXTensorWrapper und TorchTensorWrapper 
3. TensorFactory
4. TensorCacheManager

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import numpy as np
import logging

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.tests.simple_tensor_bridge")

# Füge das Hauptverzeichnis zum Pfad hinzu, um Importe zu ermöglichen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importiere die Tensor-Module direkt
try:
    from miso.math.t_mathematics.tensor_interface import MISOTensorInterface
    from miso.math.t_mathematics.tensor_wrappers import MLXTensorWrapper, TorchTensorWrapper
    from miso.math.t_mathematics.tensor_factory import tensor_factory
    from miso.math.t_mathematics.tensor_cache import TensorCacheManager
    
    # Überprüfe, ob die Module erfolgreich importiert wurden
    logger.info("Tensor-Module erfolgreich importiert.")
except ImportError as e:
    logger.error(f"Fehler beim Importieren der Tensor-Module: {e}")
    sys.exit(1)

# Überprüfe, ob MLX oder PyTorch verfügbar sind
MLX_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
    logger.info("MLX ist verfügbar.")
except ImportError:
    logger.warning("MLX ist nicht verfügbar.")

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch ist verfügbar.")
except ImportError:
    logger.warning("PyTorch ist nicht verfügbar.")

def test_tensor_creation():
    """Testet die Grundfunktionalität der Tensor-Erstellung mit der TensorFactory."""
    logger.info("\n=== Test: Tensor-Erstellung ===")
    
    # Erstelle Test-Daten
    test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    logger.info(f"Test-Daten erstellt: Shape={test_data.shape}, Dtype={test_data.dtype}")
    
    # Teste für jedes verfügbare Backend
    if MLX_AVAILABLE:
        logger.info("Teste MLX-Backend...")
        mlx_tensor = tensor_factory.create_tensor(test_data, "mlx")
        logger.info(f"MLX-Tensor erstellt: {type(mlx_tensor).__name__}")
        logger.info(f"  - Shape: {mlx_tensor.shape}")
        logger.info(f"  - Dtype: {mlx_tensor.dtype}")
        logger.info(f"  - Backend: {mlx_tensor.backend}")
        
        # Teste Tensor-Operationen
        mlx_tensor_2x = mlx_tensor + mlx_tensor
        logger.info(f"Addition ausgeführt: {type(mlx_tensor_2x).__name__}")
        logger.info(f"  - Ergebnis (erste Werte): {mlx_tensor_2x.to_numpy().flatten()[:3]}")
    
    if TORCH_AVAILABLE:
        logger.info("Teste PyTorch-Backend...")
        torch_tensor = tensor_factory.create_tensor(test_data, "torch")
        logger.info(f"PyTorch-Tensor erstellt: {type(torch_tensor).__name__}")
        logger.info(f"  - Shape: {torch_tensor.shape}")
        logger.info(f"  - Dtype: {torch_tensor.dtype}")
        logger.info(f"  - Backend: {torch_tensor.backend}")
        
        # Teste Tensor-Operationen
        torch_tensor_2x = torch_tensor + torch_tensor
        logger.info(f"Addition ausgeführt: {type(torch_tensor_2x).__name__}")
        logger.info(f"  - Ergebnis (erste Werte): {torch_tensor_2x.to_numpy().flatten()[:3]}")
    
    logger.info("Tensor-Erstellung erfolgreich getestet.")

def test_tensor_serialization():
    """Testet die Serialisierung und Deserialisierung von Tensoren."""
    logger.info("\n=== Test: Tensor-Serialisierung ===")
    
    # Erstelle Test-Daten
    test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    
    if MLX_AVAILABLE:
        logger.info("Teste MLX-Tensor-Serialisierung...")
        mlx_tensor = tensor_factory.create_tensor(test_data, "mlx")
        
        # Serialisiere den Tensor
        serialized = tensor_factory.serialize_tensor(mlx_tensor)
        logger.info(f"Serialisierter MLX-Tensor: {type(serialized)}")
        logger.info(f"  - Inhalt (Auszug): {str(serialized)[:100]}...")
        
        # Deserialisiere den Tensor
        deserialized = tensor_factory.deserialize_tensor(serialized)
        logger.info(f"Deserialisierter Tensor: {type(deserialized).__name__}")
        logger.info(f"  - Shape: {deserialized.shape}")
        logger.info(f"  - Dtype: {deserialized.dtype}")
        logger.info(f"  - Backend: {deserialized.backend}")
        
        # Überprüfe die Werte
        original_values = mlx_tensor.to_numpy().flatten()[:3]
        deserialized_values = deserialized.to_numpy().flatten()[:3]
        logger.info(f"  - Original-Werte: {original_values}")
        logger.info(f"  - Deserialisierte Werte: {deserialized_values}")
        
        # Überprüfe die Gleichheit
        array_equal = np.array_equal(mlx_tensor.to_numpy(), deserialized.to_numpy())
        logger.info(f"  - Arrays gleich: {array_equal}")
    
    if TORCH_AVAILABLE:
        logger.info("Teste PyTorch-Tensor-Serialisierung...")
        torch_tensor = tensor_factory.create_tensor(test_data, "torch")
        
        # Serialisiere den Tensor
        serialized = tensor_factory.serialize_tensor(torch_tensor)
        logger.info(f"Serialisierter PyTorch-Tensor: {type(serialized)}")
        logger.info(f"  - Inhalt (Auszug): {str(serialized)[:100]}...")
        
        # Deserialisiere den Tensor
        deserialized = tensor_factory.deserialize_tensor(serialized)
        logger.info(f"Deserialisierter Tensor: {type(deserialized).__name__}")
        logger.info(f"  - Shape: {deserialized.shape}")
        logger.info(f"  - Dtype: {deserialized.dtype}")
        logger.info(f"  - Backend: {deserialized.backend}")
        
        # Überprüfe die Werte
        original_values = torch_tensor.to_numpy().flatten()[:3]
        deserialized_values = deserialized.to_numpy().flatten()[:3]
        logger.info(f"  - Original-Werte: {original_values}")
        logger.info(f"  - Deserialisierte Werte: {deserialized_values}")
        
        # Überprüfe die Gleichheit
        array_equal = np.array_equal(torch_tensor.to_numpy(), deserialized.to_numpy())
        logger.info(f"  - Arrays gleich: {array_equal}")
    
    logger.info("Tensor-Serialisierung erfolgreich getestet.")

def test_tensor_cache():
    """Testet den Tensor-Cache-Manager."""
    logger.info("\n=== Test: Tensor-Cache ===")
    
    # Erstelle einen Cache-Manager
    cache_manager = TensorCacheManager(max_cache_size=10, default_ttl=60)
    logger.info(f"Cache-Manager erstellt: max_size={cache_manager.max_cache_size}, ttl={cache_manager.default_ttl}s")
    
    # Erstelle Test-Daten
    test_data1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    test_data2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    
    # Wähle ein verfügbares Backend
    backend = "mlx" if MLX_AVAILABLE else "torch"
    if not MLX_AVAILABLE and not TORCH_AVAILABLE:
        logger.error("Kein Tensor-Backend verfügbar für den Cache-Test.")
        return
    
    # Erstelle Tensoren
    tensor1 = tensor_factory.create_tensor(test_data1, backend)
    tensor2 = tensor_factory.create_tensor(test_data2, backend)
    logger.info(f"Test-Tensoren erstellt mit Backend: {backend}")
    
    # Simuliere eine Operation und deren Ergebnis
    operation = "addition"
    params = {"param1": "value1", "param2": "value2"}
    result = tensor1 + tensor2
    
    # Speichere im Cache
    logger.info("Speichere Ergebnis im Cache...")
    cache_manager.set(operation, [tensor1, tensor2], result, params)
    
    # Überprüfe die Cache-Statistiken
    stats = cache_manager.get_stats()
    logger.info(f"Cache-Statistiken nach dem Setzen: {stats}")
    
    # Hole aus dem Cache
    logger.info("Versuche, das Ergebnis aus dem Cache zu holen...")
    cached_result = cache_manager.get(operation, [tensor1, tensor2], params)
    
    if cached_result is not None:
        logger.info(f"Cache-Hit! Ergebnis aus dem Cache geholt: {type(cached_result).__name__}")
        logger.info(f"  - Shape: {cached_result.shape}")
        
        # Überprüfe die Gleichheit
        array_equal = np.array_equal(result.to_numpy(), cached_result.to_numpy())
        logger.info(f"  - Ergebnis gleich dem Original: {array_equal}")
    else:
        logger.error("Cache-Miss! Konnte das Ergebnis nicht aus dem Cache holen.")
    
    # Überprüfe die aktualisierten Cache-Statistiken
    stats = cache_manager.get_stats()
    logger.info(f"Cache-Statistiken nach dem Holen: {stats}")
    
    # Teste Cache-Bereinigung
    removed = cache_manager.cleanup()
    logger.info(f"Cache-Bereinigung durchgeführt: {removed} Einträge entfernt")
    
    logger.info("Tensor-Cache erfolgreich getestet.")

def test_tensor_conversion():
    """Testet die Konvertierung zwischen verschiedenen Tensor-Backends."""
    logger.info("\n=== Test: Tensor-Konvertierung ===")
    
    if not (MLX_AVAILABLE and TORCH_AVAILABLE):
        logger.warning("Beide Backends (MLX und PyTorch) werden benötigt, um die Tensor-Konvertierung zu testen.")
        return
    
    # Erstelle Test-Daten
    test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    logger.info(f"Test-Daten erstellt: Shape={test_data.shape}, Dtype={test_data.dtype}")
    
    # Erstelle MLX-Tensor
    mlx_tensor = tensor_factory.create_tensor(test_data, "mlx")
    logger.info(f"MLX-Tensor erstellt: {type(mlx_tensor).__name__}, Shape={mlx_tensor.shape}")
    
    # Konvertiere zu PyTorch
    logger.info("Konvertiere MLX -> PyTorch...")
    start_time = time.time()
    torch_tensor = tensor_factory.convert_tensor(mlx_tensor, "torch")
    conversion_time = time.time() - start_time
    logger.info(f"Konvertierung abgeschlossen in {conversion_time:.6f}s")
    logger.info(f"PyTorch-Tensor: {type(torch_tensor).__name__}, Shape={torch_tensor.shape}")
    
    # Überprüfe die Werte
    mlx_values = mlx_tensor.to_numpy().flatten()[:3]
    torch_values = torch_tensor.to_numpy().flatten()[:3]
    logger.info(f"  - MLX-Werte: {mlx_values}")
    logger.info(f"  - PyTorch-Werte: {torch_values}")
    
    # Überprüfe die Gleichheit
    array_equal = np.array_equal(mlx_tensor.to_numpy(), torch_tensor.to_numpy())
    logger.info(f"  - Arrays gleich: {array_equal}")
    
    # Konvertiere zurück zu MLX
    logger.info("Konvertiere PyTorch -> MLX...")
    start_time = time.time()
    mlx_tensor_2 = tensor_factory.convert_tensor(torch_tensor, "mlx")
    conversion_time = time.time() - start_time
    logger.info(f"Konvertierung abgeschlossen in {conversion_time:.6f}s")
    logger.info(f"MLX-Tensor: {type(mlx_tensor_2).__name__}, Shape={mlx_tensor_2.shape}")
    
    # Überprüfe die Gleichheit mit dem Original
    array_equal = np.array_equal(mlx_tensor.to_numpy(), mlx_tensor_2.to_numpy())
    logger.info(f"  - Original-MLX und konvertierter MLX gleich: {array_equal}")
    
    logger.info("Tensor-Konvertierung erfolgreich getestet.")

def run_all_tests():
    """Führt alle Tests aus."""
    logger.info("=== Starte Tests für die Tensor-Operation-Bridge ===")
    
    if not MLX_AVAILABLE and not TORCH_AVAILABLE:
        logger.error("Weder MLX noch PyTorch sind verfügbar. Mindestens eines wird benötigt.")
        sys.exit(1)
    
    try:
        # Teste die Grundfunktionalitäten
        test_tensor_creation()
        test_tensor_serialization()
        test_tensor_cache()
        
        # Teste die Konvertierung, falls beide Backends verfügbar sind
        if MLX_AVAILABLE and TORCH_AVAILABLE:
            test_tensor_conversion()
        
        logger.info("\n=== Alle Tests erfolgreich abgeschlossen! ===")
        return 0
    except Exception as e:
        logger.error(f"Test fehlgeschlagen: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
