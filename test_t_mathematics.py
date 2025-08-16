#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - T-Mathematics Engine Testskript

Dieses Skript testet die Funktionalität der T-Mathematics Engine.
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Füge das Hauptverzeichnis zum Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.test")

# Importiere T-Mathematics Engine
from miso.math.t_mathematics import (
    TMathEngine, 
    TMathConfig,
    get_engine,
    get_default_config,
    tensor_svd,
    amd_optimized_matmul,
    moe_routing,
    mix_experts_outputs,
    OptimizedMultiHeadAttention,
    OptimizedFeedForward,
    MixtureOfExperts,
    TMathTransformerLayer
)


def test_engine_initialization():
    """Test der Engine-Initialisierung"""
    logger.info("Test: Engine-Initialisierung")
    
    # Standardkonfiguration abrufen
    config = get_default_config()
    logger.info(f"Standardkonfiguration: {config}")
    
    # Engine initialisieren
    engine = get_engine()
    logger.info(f"Engine initialisiert: {engine}")
    
    # Hardware-Informationen
    device_info = engine.get_device_info()
    logger.info(f"Hardware-Informationen: {device_info}")
    
    return True


def test_basic_operations():
    """Test der grundlegenden Operationen"""
    logger.info("Test: Grundlegende Operationen")
    
    # Engine abrufen
    engine = get_engine()
    
    # Testmatrizen erstellen
    a = torch.randn(64, 128)
    b = torch.randn(128, 64)
    
    # Matrix-Multiplikation
    result = engine.matmul(a, b)
    logger.info(f"Matrix-Multiplikation: {result.shape}")
    
    # SVD
    u, s, v = engine.svd(a)
    logger.info(f"SVD Shapes: U={u.shape}, S={s.shape}, V={v.shape}")
    
    # Tensor-Operationen
    c = torch.randn(32, 64, 128)
    d = torch.randn(32, 128, 64)
    result = engine.batch_matmul(c, d)
    logger.info(f"Batch Matrix-Multiplikation: {result.shape}")
    
    return True


def test_moe_functionality():
    """Test der Mixture-of-Experts Funktionalität"""
    logger.info("Test: Mixture-of-Experts")
    
    # Parameter
    batch_size = 16
    seq_len = 32
    d_model = 512
    num_experts = 8
    
    # Eingabetensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # MoE-Modell erstellen
    moe = MixtureOfExperts(
        input_dim=d_model,
        output_dim=d_model,
        num_experts=num_experts,
        k=2  # Top-2 Routing
    )
    
    # Forward-Pass
    output = moe(x)
    logger.info(f"MoE Output: {output.shape}")
    
    # Routing-Informationen
    routing_probs = moe.last_routing_probs
    if routing_probs is not None:
        logger.info(f"Routing-Wahrscheinlichkeiten: {routing_probs.shape}")
    
    return True


def test_transformer_layer():
    """Test des Transformer-Layers"""
    logger.info("Test: Transformer-Layer")
    
    # Parameter
    batch_size = 8
    seq_len = 64
    d_model = 768
    num_heads = 12
    
    # Eingabetensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Transformer-Layer erstellen
    layer = TMathTransformerLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=d_model * 4,
        use_moe=True,
        num_experts=8
    )
    
    # Forward-Pass
    output = layer(x)
    logger.info(f"Transformer-Layer Output: {output.shape}")
    
    return True


def run_all_tests():
    """Führt alle Tests aus"""
    tests = [
        test_engine_initialization,
        test_basic_operations,
        test_moe_functionality,
        test_transformer_layer
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            logger.info(f"Test '{test.__name__}' erfolgreich")
        except Exception as e:
            logger.error(f"Test '{test.__name__}' fehlgeschlagen: {e}")
            results.append(False)
    
    # Zusammenfassung
    success = all(results)
    logger.info(f"Testergebnisse: {sum(results)}/{len(results)} Tests erfolgreich")
    
    return success


if __name__ == "__main__":
    logger.info("Starte T-Mathematics Engine Tests")
    
    # Umgebungsvariablen für Tests setzen
    os.environ["T_MATH_PRECISION"] = "mixed"
    os.environ["T_MATH_DEVICE"] = "auto"
    os.environ["T_MATH_OPTIMIZE_AMD"] = "1"
    os.environ["T_MATH_OPTIMIZE_APPLE"] = "1"
    
    # Tests ausführen
    success = run_all_tests()
    
    # Ergebnis
    if success:
        logger.info("Alle Tests erfolgreich abgeschlossen!")
        sys.exit(0)
    else:
        logger.error("Einige Tests sind fehlgeschlagen!")
        sys.exit(1)
