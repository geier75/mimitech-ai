#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fortschrittlicher SVD Benchmark für T-Mathematics Engine Optimierungen

Dieses Skript bietet eine robuste, ressourceneffiziente Methode zum Vergleich 
der SVD-Implementierungen mit präziser Leistungsmessung und Ressourcenkontrolle.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import numpy as np
import torch
import logging
import gc
import psutil
import traceback
import argparse
import contextlib
import threading
import tempfile
import json
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from functools import wraps
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.math.t_mathematics.optimizations.advanced_svd_benchmark")

# Prüfe auf MLX
try:
    import mlx.core as mx
    HAS_MLX = True
    logger.info("MLX verfügbar")
except ImportError:
    HAS_MLX = False
    mx = None
    logger.warning("MLX nicht verfügbar - Benchmarks werden mit NumPy durchgeführt")

# T-Mathematics Engine Pfad hinzufügen
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Utility-Funktionen für Ressourcenmanagement und Messung
class ResourceMonitor:
    """Überwacht und kontrolliert Ressourcennutzung während Benchmarks"""
    
    def __init__(self, memory_threshold_mb=1000):
        """
        Args:
            memory_threshold_mb: Speicherschwellwert in MB für Warnungen
        """
        self.process = psutil.Process(os.getpid())
        self.memory_threshold = memory_threshold_mb * 1024 * 1024  # Konvertiere in Bytes
        self.initial_memory = self.get_memory_usage()
        logger.info(f"Initial memory usage: {self.initial_memory / (1024*1024):.2f} MB")
    
    def get_memory_usage(self):
        """Aktuelle Speichernutzung abrufen"""
        return self.process.memory_info().rss
    
    def get_memory_delta(self):
        """Speicheränderung seit Initialisierung"""
        return self.get_memory_usage() - self.initial_memory
    
    def check_resources(self):
        """Prüft aktuelle Ressourcennutzung und warnt bei hohem Verbrauch"""
        memory_usage = self.get_memory_usage()
        memory_delta = self.get_memory_delta()
        
        # Warne bei hohem Speicherverbrauch
        if memory_usage > self.memory_threshold:
            logger.warning(f"Hohe Speichernutzung: {memory_usage / (1024*1024):.2f} MB")
            return False
        
        logger.debug(f"Speichernutzung: {memory_usage / (1024*1024):.2f} MB (Delta: {memory_delta / (1024*1024):.2f} MB)")
        return True
    
    def free_unused_memory(self):
        """Führt explizite Garbage Collection durch und gibt Speicher frei"""
        before = self.get_memory_usage()
        
        # Python-GC
        gc.collect()
        
        # PyTorch-Caches leeren wenn verfügbar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        # MLX-Caches leeren wenn verfügbar
        if HAS_MLX and hasattr(mx, 'clear_cache'):
            mx.clear_cache()
        
        after = self.get_memory_usage()
        freed = before - after
        
        if freed > 0:
            logger.info(f"Speicher freigegeben: {freed / (1024*1024):.2f} MB")
        
        return freed

@contextlib.contextmanager
def resource_context(matrix_name="", operation="", level=0):
    """Context-Manager für sauberes Ressourcenmanagement während Benchmark-Operationen"""
    monitor = ResourceMonitor()
    start_memory = monitor.get_memory_usage()
    
    try:
        logger.debug(f"Starte Operation: {operation} für {matrix_name} (Level {level})")
        yield monitor
    except Exception as e:
        logger.error(f"Fehler in {operation} für {matrix_name} (Level {level}): {e}")
        logger.error(traceback.format_exc())
    finally:
        # Speicher explizit freigeben
        end_memory = monitor.get_memory_usage()
        memory_delta = end_memory - start_memory
        
        if memory_delta > 10 * 1024 * 1024:  # Wenn >10MB Speicherzuwachs
            logger.warning(f"Speicherzuwachs: {memory_delta / (1024*1024):.2f} MB während {operation}")
            monitor.free_unused_memory()
        
        logger.debug(f"Operation beendet: {operation} für {matrix_name} (Level {level})")

def retry_with_cleanup(max_retries=3, cleanup_func=None):
    """Decorator für Funktionen, die mit automatischen Wiederholungen und Bereinigung ausgeführt werden sollen"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    last_exception = e
                    logger.warning(f"Versuch {retries}/{max_retries} fehlgeschlagen: {e}")
                    
                    # Führe Bereinigung durch
                    if cleanup_func:
                        cleanup_func()
                    else:
                        # Standard-Cleanup
                        logger.info("Führe Standard-Bereinigung durch...")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    
                    # Warte kurz vor dem nächsten Versuch
                    time.sleep(1)
            
            # Alle Versuche fehlgeschlagen
            logger.error(f"Alle {max_retries} Versuche fehlgeschlagen.")
            if last_exception:
                logger.error(f"Letzter Fehler: {last_exception}")
                logger.error(traceback.format_exc())
            
            return None
        
        return wrapper
    
    return decorator
