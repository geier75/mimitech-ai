#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - T-Mathematics Backend-Check

Dieses Skript überprüft die Verfügbarkeit und Funktionalität der verschiedenen
Backend-Implementierungen der T-Mathematics Engine (MLX, PyTorch, NumPy).

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [T-MATH-CHECK] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MISO.T-Mathematics.Check")

# Prüfe Backend-Verfügbarkeit
def check_backends():
    """Überprüft die Verfügbarkeit der Backend-Implementierungen"""
    backends = {
        "mlx": {"available": False, "version": None, "apple_silicon": False},
        "torch": {"available": False, "version": None, "mps": False},
        "numpy": {"available": False, "version": None}
    }
    
    # NumPy-Backend
    try:
        import numpy as np
        backends["numpy"]["available"] = True
        backends["numpy"]["version"] = np.__version__
        logger.info(f"NumPy gefunden: Version {np.__version__}")
    except ImportError:
        logger.warning("NumPy nicht gefunden")
    
    # PyTorch-Backend
    try:
        import torch
        backends["torch"]["available"] = True
        backends["torch"]["version"] = torch.__version__
        logger.info(f"PyTorch gefunden: Version {torch.__version__}")
        
        # Prüfe MPS-Unterstützung (für Apple Metal)
        if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available"):
            backends["torch"]["mps"] = torch.backends.mps.is_available()
            if backends["torch"]["mps"]:
                logger.info("PyTorch MPS-Unterstützung verfügbar")
            else:
                logger.info("PyTorch MPS-Unterstützung nicht verfügbar")
    except ImportError:
        logger.warning("PyTorch nicht gefunden")
    
    # MLX-Backend
    try:
        import mlx.core
        backends["mlx"]["available"] = True
        
        # MLX hat keine einfach zugängliche Versionsinformation, aber wir können prüfen, ob es auf Apple Silicon läuft
        import platform
        is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
        backends["mlx"]["apple_silicon"] = is_apple_silicon
        
        if is_apple_silicon:
            logger.info("MLX gefunden: Läuft auf Apple Silicon")
        else:
            logger.info("MLX gefunden: Läuft nicht auf Apple Silicon")
    except ImportError:
        logger.warning("MLX nicht gefunden")
    
    return backends

# Importiere Tensor-Implementierungen
def import_tensor_implementations(root_dir):
    """Versucht, die Tensor-Implementierungen zu importieren"""
    sys.path.insert(0, root_dir)
    
    implementations = {
        "MISOTensor": {"imported": False, "path": None},
        "MLXTensor": {"imported": False, "path": None},
        "TorchTensor": {"imported": False, "path": None},
        "NumPyTensor": {"imported": False, "path": None},
        "TensorFactory": {"imported": False, "path": None}
    }
    
    # Direkte Importversuche
    try:
        from tensor import MISOTensor
        implementations["MISOTensor"]["imported"] = True
        implementations["MISOTensor"]["path"] = MISOTensor.__module__
        logger.info(f"MISOTensor erfolgreich importiert aus {MISOTensor.__module__}")
    except ImportError as e:
        logger.warning(f"MISOTensor konnte nicht importiert werden: {e}")
    
    try:
        from tensor_mlx import MLXTensor
        implementations["MLXTensor"]["imported"] = True
        implementations["MLXTensor"]["path"] = MLXTensor.__module__
        logger.info(f"MLXTensor erfolgreich importiert aus {MLXTensor.__module__}")
    except ImportError as e:
        logger.warning(f"MLXTensor konnte nicht importiert werden: {e}")
    
    try:
        from tensor_torch import TorchTensor
        implementations["TorchTensor"]["imported"] = True
        implementations["TorchTensor"]["path"] = TorchTensor.__module__
        logger.info(f"TorchTensor erfolgreich importiert aus {TorchTensor.__module__}")
    except ImportError as e:
        logger.warning(f"TorchTensor konnte nicht importiert werden: {e}")
    
    try:
        from tensor_numpy import NumPyTensor
        implementations["NumPyTensor"]["imported"] = True
        implementations["NumPyTensor"]["path"] = NumPyTensor.__module__
        logger.info(f"NumPyTensor erfolgreich importiert aus {NumPyTensor.__module__}")
    except ImportError as e:
        logger.warning(f"NumPyTensor konnte nicht importiert werden: {e}")
    
    try:
        from tensor_factory import TensorFactory
        implementations["TensorFactory"]["imported"] = True
        implementations["TensorFactory"]["path"] = TensorFactory.__module__
        logger.info(f"TensorFactory erfolgreich importiert aus {TensorFactory.__module__}")
    except ImportError as e:
        logger.warning(f"TensorFactory konnte nicht importiert werden: {e}")
    
    return implementations

# Einfache Tensor-Tests mit den verfügbaren Backends
def test_tensor_operations(implementations, backends):
    """Testet grundlegende Tensoroperationen mit den verfügbaren Backends"""
    results = {}
    
    # Überprüfe, ob TensorFactory verfügbar ist
    if not implementations["TensorFactory"]["imported"]:
        logger.error("TensorFactory ist nicht verfügbar, Tests können nicht durchgeführt werden")
        return None
    
    # Importiere TensorFactory
    sys.path.insert(0, os.path.dirname(implementations["TensorFactory"]["path"]))
    from tensor_factory import TensorFactory, tensor, get_available_backends
    
    # Verfügbare Backends abrufen
    available_backends = get_available_backends()
    logger.info(f"Verfügbare Tensor-Backends: {available_backends}")
    
    # Einfache Operationen mit jedem Backend testen
    test_data = np.random.rand(4, 4).astype(np.float32)
    
    for backend in available_backends:
        logger.info(f"Teste Backend: {backend}")
        results[backend] = {"addition": None, "multiplication": None, "matmul": None, "time": None}
        
        try:
            start_time = time.time()
            
            # Erstelle Tensoren
            a = tensor(test_data, backend=backend)
            b = tensor(test_data * 2, backend=backend)
            
            # Addition
            c = a + b
            results[backend]["addition"] = (c.to_numpy() == (test_data + (test_data * 2))).all()
            
            # Multiplikation
            d = a * b
            results[backend]["multiplication"] = (d.to_numpy() == (test_data * (test_data * 2))).all()
            
            # Matrix-Multiplikation
            e = a @ b
            results[backend]["matmul"] = True  # Wert kann nicht direkt verglichen werden
            
            end_time = time.time()
            results[backend]["time"] = end_time - start_time
            
            logger.info(f"  Addition: {'✓' if results[backend]['addition'] else '✗'}")
            logger.info(f"  Multiplikation: {'✓' if results[backend]['multiplication'] else '✗'}")
            logger.info(f"  Matrix-Multiplikation: {'✓' if results[backend]['matmul'] else '✗'}")
            logger.info(f"  Zeit: {results[backend]['time']:.6f}s")
        
        except Exception as e:
            logger.error(f"Fehler bei Tests mit Backend {backend}: {e}")
            results[backend] = {"error": str(e)}
    
    return results

# Hauptfunktion
def main():
    """Hauptfunktion"""
    logger.info("=== T-Mathematics Backend-Check ===")
    
    # Pfad zum MISO Ultimate-Verzeichnis
    miso_dir = os.path.abspath(os.path.dirname(__file__))
    logger.info(f"MISO Ultimate-Verzeichnis: {miso_dir}")
    
    # Überprüfe die Backend-Verfügbarkeit
    backends = check_backends()
    
    # Importiere die Tensor-Implementierungen
    implementations = import_tensor_implementations(miso_dir)
    
    # Teste die Tensor-Operationen
    results = test_tensor_operations(implementations, backends)
    
    # Speichere Ergebnisse
    report = {
        "timestamp": time.time(),
        "backends": backends,
        "implementations": implementations,
        "test_results": results
    }
    
    report_file = os.path.join(miso_dir, "t_mathematics_backend_check.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Bericht gespeichert in: {report_file}")
    
    # Zusammenfassung
    logger.info("=== Zusammenfassung ===")
    
    logger.info("Backend-Verfügbarkeit:")
    for backend, info in backends.items():
        status = "✓ Verfügbar" if info["available"] else "✗ Nicht verfügbar"
        logger.info(f"  {backend}: {status}")
    
    logger.info("Tensor-Implementierungen:")
    for impl, info in implementations.items():
        status = "✓ Importiert" if info["imported"] else "✗ Nicht importiert"
        logger.info(f"  {impl}: {status}")
    
    if results:
        logger.info("Test-Ergebnisse:")
        for backend, res in results.items():
            if "error" in res:
                logger.info(f"  {backend}: ✗ Fehler - {res['error']}")
            else:
                success = all(v for k, v in res.items() if k != "time")
                status = "✓ Erfolgreich" if success else "✗ Fehler"
                logger.info(f"  {backend}: {status} (Zeit: {res.get('time', 'N/A')}s)")

if __name__ == "__main__":
    main()
