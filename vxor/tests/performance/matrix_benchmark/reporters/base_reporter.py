#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basis-Reporter für Matrix-Benchmarks.

Dieses Modul definiert die abstrakte Basisklasse für Benchmark-Reporter.
"""

import os
import platform
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

class BenchmarkReporter(ABC):
    """Abstrakte Basisklasse für Benchmark-Reporter.
    
    Diese Klasse definiert die grundlegende Schnittstelle, die alle Reporter
    implementieren müssen, um Benchmark-Ergebnisse darzustellen.
    """
    
    @abstractmethod
    def generate_report(self, results, output_file: str) -> None:
        """Generiert einen Bericht aus den Benchmark-Ergebnissen.
        
        Args:
            results: Liste der Benchmark-Ergebnisse
            output_file: Pfad zur Ausgabedatei
        """
        pass
    
    def _get_system_info(self) -> Dict[str, str]:
        """Sammelt Systeminformationen für den Bericht.
        
        Returns:
            Dictionary mit Systeminformationen
        """
        system_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "timestamp": datetime.now().isoformat(),
            "is_apple_silicon": "Apple" in platform.processor()
        }
        
        # Füge Versionsinformationen für relevante Bibliotheken hinzu
        try:
            import numpy as np
            system_info["numpy_version"] = np.__version__
        except ImportError:
            pass
            
        try:
            import torch
            system_info["pytorch_version"] = torch.__version__
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
                system_info["mps_available"] = str(torch.backends.mps.is_available())
        except ImportError:
            pass
            
        try:
            import mlx.core as mx
            system_info["mlx_version"] = getattr(mx, "__version__", "verfügbar")
            system_info["mlx_optimized"] = "True"  # MLX ist für Apple Silicon optimiert
        except ImportError:
            system_info["mlx_optimized"] = "False"
            
        return system_info
    
    def _ensure_directory_exists(self, file_path: str) -> None:
        """Stellt sicher, dass das Verzeichnis für die Ausgabedatei existiert.
        
        Args:
            file_path: Pfad zur Ausgabedatei
        """
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Verzeichnis erstellt: {directory}")
