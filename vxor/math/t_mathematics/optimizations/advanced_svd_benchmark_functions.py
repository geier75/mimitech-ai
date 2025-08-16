#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark-Funktionen für fortschrittlichen SVD-Benchmark

Diese Datei enthält die Benchmark-Funktionen für die Leistungsmessung der SVD-Optimierungen
in der T-Mathematics Engine.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import numpy as np
import torch
import logging
import gc
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from functools import wraps
import json
from pathlib import Path

# Konfiguriere Logging
logger = logging.getLogger("MISO.math.t_mathematics.optimizations.advanced_svd_benchmark_functions")

# Prüfe auf MLX
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

# Import der Helper-Funktionen
try:
    from miso.math.t_mathematics.optimizations.advanced_svd_benchmark import (
        resource_context, retry_with_cleanup, ResourceMonitor
    )
except ImportError:
    logger.error("Konnte advanced_svd_benchmark nicht importieren")
    sys.exit(1)

# MLX-Backend importieren
try:
    from miso.math.t_mathematics.mlx_support import MLXBackend
    logger.info("MLXBackend erfolgreich importiert")
except ImportError:
    logger.error("MLXBackend konnte nicht importiert werden")
    sys.exit(1)

# Optimierungen importieren
try:
    from miso.math.t_mathematics.optimizations.integration import optimize_mlx_backend, configure_optimizations
    logger.info("Optimierungsmodul erfolgreich importiert")
except ImportError:
    logger.error("Optimierungsmodul konnte nicht importiert werden")
    sys.exit(1)

class SVDBenchmarker:
    """
    Klasse für fortschrittliche SVD-Benchmark-Tests mit Ressourcenkontrolle
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: Benchmark-Konfiguration
        """
        # Standard-Konfiguration
        self.default_config = {
            "matrix_sizes": [
                ("tiny", (8, 8)),
                ("small", (32, 32)),
                ("medium", (128, 128)),
                ("large", (256, 256))
            ],
            "k_values": [4, 16],
            "runs_per_test": 3,
            "optimization_levels": [0, 2, 3],
            "precision": "float32",
            "timeout_seconds": 60,
            "max_memory_mb": 2000,
            "save_results": True,
            "results_dir": "./benchmark_results"
        }
        
        # Überschreibe Standard-Konfiguration mit bereitgestellter Konfiguration
        self.config = dict(self.default_config)
        if config:
            self.config.update(config)
        
        # Backends initialisieren
        self.backends = {}
        
        # Sicherstellen, dass Ergebnisverzeichnis existiert
        if self.config["save_results"]:
            os.makedirs(self.config["results_dir"], exist_ok=True)
        
        # Ergebnisse initialisieren
        self.results = {}
        
        # Ressourcenmonitor
        self.monitor = ResourceMonitor(memory_threshold_mb=self.config["max_memory_mb"])
    
    def initialize_backends(self):
        """
        Initialisiert Backends für verschiedene Optimierungsstufen
        """
        logger.info("Initialisiere Backends...")
        
        # Alte Backends bereinigen
        for backend in self.backends.values():
            del backend
        self.backends = {}
        
        # Explizite Speicherbereinigung
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        # Neue Backends erstellen
        for level in self.config["optimization_levels"]:
            try:
                # Optimierungslevel konfigurieren
                configure_optimizations(optimization_level=level)
                
                # Backend erstellen
                backend = MLXBackend(precision=self.config["precision"])
                
                # Optimieren, wenn Level > 0
                if level > 0:
                    backend = optimize_mlx_backend(backend, optimization_level=level)
                
                self.backends[level] = backend
                logger.info(f"Backend mit Optimierungsstufe {level} erstellt")
            except Exception as e:
                logger.error(f"Fehler beim Erstellen des Backends mit Level {level}: {e}")
                logger.error(traceback.format_exc())
        
        return len(self.backends) > 0
    
    @retry_with_cleanup(max_retries=2)
    def _run_single_svd_test(self, backend, matrix, k=None):
        """
        Führt einen einzelnen SVD-Test durch
        
        Args:
            backend: MLXBackend-Instanz
            matrix: Matrix-Daten (NumPy-Array)
            k: k-Wert für partielle SVD (None für vollständige SVD)
            
        Returns:
            Tuple (Ergebnis, Ausführungszeit in Sekunden)
        """
        # Konvertiere Matrix zu MLX, wenn verfügbar
        if HAS_MLX:
            matrix_mlx = mx.array(matrix)
        else:
            matrix_mlx = matrix
        
        # Führe SVD durch und messe Zeit
        start_time = time.time()
        result = backend.svd(matrix_mlx, k)
        end_time = time.time()
        
        # Zeit berechnen
        execution_time = end_time - start_time
        
        # Ergebnis validieren
        if result is None or len(result) != 3:
            logger.warning(f"Ungültiges SVD-Ergebnis: {result}")
            raise ValueError("Ungültiges SVD-Ergebnis")
        
        return result, execution_time
    
    def benchmark_matrix(self, matrix_name, matrix, optimization_level):
        """
        Führt Benchmark für eine Matrix und einen Optimierungslevel durch
        
        Args:
            matrix_name: Name der Matrix für Berichte
            matrix: Die zu testende Matrix
            optimization_level: Zu verwendender Optimierungslevel
            
        Returns:
            Dictionary mit Ergebnissen
        """
        result = {
            "full_svd": {"times": [], "status": "not_run"},
            "partial_svd": {}
        }
        
        # Prüfe, ob Backend für diesen Level existiert
        if optimization_level not in self.backends:
            logger.error(f"Kein Backend für Optimierungslevel {optimization_level}")
            return result
        
        backend = self.backends[optimization_level]
        runs = self.config["runs_per_test"]
        
        # Teste vollständige SVD
        with resource_context(matrix_name, "full_svd", optimization_level) as monitor:
            try:
                logger.info(f"Teste vollständige SVD für {matrix_name} (Level {optimization_level})...")
                result["full_svd"]["status"] = "running"
                
                times = []
                for run in range(runs):
                    # Führe SVD durch
                    _, execution_time = self._run_single_svd_test(backend, matrix)
                    times.append(execution_time)
                    
                    # Überprüfe Ressourcen nach jedem Durchlauf
                    if not monitor.check_resources():
                        logger.warning(f"Ressourcennutzung zu hoch, breche weitere Durchläufe ab")
                        break
                
                result["full_svd"]["times"] = times
                result["full_svd"]["status"] = "success"
                
                # Berechne Statistiken
                if times:
                    result["full_svd"]["avg_time"] = sum(times) / len(times)
                    result["full_svd"]["min_time"] = min(times)
                    result["full_svd"]["max_time"] = max(times)
                
                logger.info(f"Vollständige SVD für {matrix_name} (Level {optimization_level}) abgeschlossen")
            except Exception as e:
                logger.error(f"Fehler bei vollständiger SVD für {matrix_name} (Level {optimization_level}): {e}")
                logger.error(traceback.format_exc())
                result["full_svd"]["status"] = "error"
                result["full_svd"]["error"] = str(e)
        
        # Speicher freigeben
        self.monitor.free_unused_memory()
        
        # Teste partielle SVD für verschiedene k-Werte
        for k in self.config["k_values"]:
            # Prüfe, ob k gültig ist
            if k >= min(matrix.shape):
                continue
            
            k_key = f"k={k}"
            result["partial_svd"][k_key] = {"times": [], "status": "not_run"}
            
            with resource_context(matrix_name, f"partial_svd_{k}", optimization_level) as monitor:
                try:
                    logger.info(f"Teste partielle SVD mit k={k} für {matrix_name} (Level {optimization_level})...")
                    result["partial_svd"][k_key]["status"] = "running"
                    
                    times = []
                    for run in range(runs):
                        # Führe SVD durch
                        _, execution_time = self._run_single_svd_test(backend, matrix, k)
                        times.append(execution_time)
                        
                        # Überprüfe Ressourcen nach jedem Durchlauf
                        if not monitor.check_resources():
                            logger.warning(f"Ressourcennutzung zu hoch, breche weitere Durchläufe ab")
                            break
                    
                    result["partial_svd"][k_key]["times"] = times
                    result["partial_svd"][k_key]["status"] = "success"
                    
                    # Berechne Statistiken
                    if times:
                        result["partial_svd"][k_key]["avg_time"] = sum(times) / len(times)
                        result["partial_svd"][k_key]["min_time"] = min(times)
                        result["partial_svd"][k_key]["max_time"] = max(times)
                    
                    logger.info(f"Partielle SVD (k={k}) für {matrix_name} (Level {optimization_level}) abgeschlossen")
                except Exception as e:
                    logger.error(f"Fehler bei partieller SVD (k={k}) für {matrix_name} (Level {optimization_level}): {e}")
                    logger.error(traceback.format_exc())
                    result["partial_svd"][k_key]["status"] = "error"
                    result["partial_svd"][k_key]["error"] = str(e)
            
            # Speicher freigeben
            self.monitor.free_unused_memory()
        
        return result
    
    def generate_test_matrices(self):
        """
        Generiert Testmatrizen basierend auf Konfiguration
        
        Returns:
            Dictionary mit Testmatrizen
        """
        matrices = {}
        
        for name, size in self.config["matrix_sizes"]:
            try:
                # Erstelle Matrix mit zufälligen Werten
                matrix = np.random.rand(*size).astype(np.float32 if self.config["precision"] == "float32" else np.float64)
                matrices[name] = matrix
                logger.info(f"Matrix '{name}' ({size[0]}x{size[1]}) erstellt")
            except Exception as e:
                logger.error(f"Fehler beim Erstellen der Matrix '{name}': {e}")
        
        return matrices
    
    def run_benchmark(self):
        """
        Führt den vollständigen Benchmark durch
        
        Returns:
            Dictionary mit Benchmark-Ergebnissen
        """
        logger.info("Starte SVD-Benchmark...")
        
        # Backends initialisieren
        if not self.initialize_backends():
            logger.error("Konnte keine Backends initialisieren, breche Benchmark ab")
            return {}
        
        # Testmatrizen generieren
        matrices = self.generate_test_matrices()
        if not matrices:
            logger.error("Konnte keine Testmatrizen generieren, breche Benchmark ab")
            return {}
        
        # Ergebnisse für jede Optimierungsstufe und Matrix sammeln
        self.results = {}
        
        for level in self.config["optimization_levels"]:
            self.results[f"level_{level}"] = {}
            
            for matrix_name, matrix in matrices.items():
                # Speicher vor jedem Test freigeben
                self.monitor.free_unused_memory()
                
                # Benchmark durchführen
                logger.info(f"Benchmarke Matrix '{matrix_name}' mit Optimierungslevel {level}...")
                matrix_results = self.benchmark_matrix(matrix_name, matrix, level)
                
                # Ergebnisse speichern
                self.results[f"level_{level}"][matrix_name] = matrix_results
        
        # Speichere Ergebnisse
        if self.config["save_results"]:
            self.save_results()
        
        return self.results
    
    def save_results(self):
        """
        Speichert Benchmark-Ergebnisse in eine JSON-Datei
        """
        if not self.results:
            logger.warning("Keine Ergebnisse zum Speichern")
            return
        
        try:
            # Erstelle Zeitstempel für Dateinamen
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.config["results_dir"], f"svd_benchmark_{timestamp}.json")
            
            # Konvertiere Ergebnisse in JSON-serialisierbares Format
            serializable_results = {}
            
            for level_key, level_results in self.results.items():
                serializable_results[level_key] = {}
                
                for matrix_name, matrix_results in level_results.items():
                    serializable_results[level_key][matrix_name] = {}
                    
                    # Vollständige SVD
                    serializable_results[level_key][matrix_name]["full_svd"] = {
                        k: v for k, v in matrix_results["full_svd"].items()
                    }
                    
                    # Partielle SVD
                    serializable_results[level_key][matrix_name]["partial_svd"] = {}
                    for k_key, k_results in matrix_results["partial_svd"].items():
                        serializable_results[level_key][matrix_name]["partial_svd"][k_key] = {
                            k: v for k, v in k_results.items()
                        }
            
            # Speichere Ergebnisse und Konfiguration
            with open(filename, 'w') as f:
                json.dump({
                    "config": self.config,
                    "results": serializable_results
                }, f, indent=2)
            
            logger.info(f"Ergebnisse in '{filename}' gespeichert")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Ergebnisse: {e}")
            logger.error(traceback.format_exc())
    
    def analyze_results(self):
        """
        Analysiert Benchmark-Ergebnisse und gibt eine Zusammenfassung zurück
        
        Returns:
            Dictionary mit Analysen
        """
        if not self.results:
            logger.warning("Keine Ergebnisse zur Analyse")
            return {}
        
        analysis = {
            "improvements": {},
            "best_levels": {},
            "summary": {}
        }
        
        # Durchschnittliche Verbesserung für jede Matrix
        for matrix_name in next(iter(self.results.values())).keys():
            analysis["improvements"][matrix_name] = {}
            analysis["best_levels"][matrix_name] = {}
            
            # Vollständige SVD
            baseline_time = None
            best_time = float('inf')
            best_level = None
            
            for level_key, level_results in self.results.items():
                level = int(level_key.split('_')[1])
                
                if matrix_name not in level_results:
                    continue
                
                full_svd = level_results[matrix_name]["full_svd"]
                if "avg_time" not in full_svd:
                    continue
                
                # Speichere Basiszeit (Level 0)
                if level == 0:
                    baseline_time = full_svd["avg_time"]
                
                # Finde beste Zeit
                if full_svd["avg_time"] < best_time:
                    best_time = full_svd["avg_time"]
                    best_level = level
            
            # Berechne Verbesserung
            if baseline_time and best_time < float('inf'):
                improvement = (baseline_time - best_time) / baseline_time * 100
                analysis["improvements"][matrix_name]["full_svd"] = improvement
                analysis["best_levels"][matrix_name]["full_svd"] = best_level
            
            # Partielle SVD für jeden k-Wert
            for k in self.config["k_values"]:
                k_key = f"k={k}"
                
                baseline_time = None
                best_time = float('inf')
                best_level = None
                
                for level_key, level_results in self.results.items():
                    level = int(level_key.split('_')[1])
                    
                    if matrix_name not in level_results:
                        continue
                    
                    partial_svd = level_results[matrix_name]["partial_svd"]
                    if k_key not in partial_svd or "avg_time" not in partial_svd[k_key]:
                        continue
                    
                    # Speichere Basiszeit (Level 0)
                    if level == 0:
                        baseline_time = partial_svd[k_key]["avg_time"]
                    
                    # Finde beste Zeit
                    if partial_svd[k_key]["avg_time"] < best_time:
                        best_time = partial_svd[k_key]["avg_time"]
                        best_level = level
                
                # Berechne Verbesserung
                if baseline_time and best_time < float('inf'):
                    improvement = (baseline_time - best_time) / baseline_time * 100
                    analysis["improvements"][matrix_name][k_key] = improvement
                    analysis["best_levels"][matrix_name][k_key] = best_level
        
        # Gesamtzusammenfassung
        total_improvement = 0
        count = 0
        
        for matrix_improvements in analysis["improvements"].values():
            for improvement in matrix_improvements.values():
                total_improvement += improvement
                count += 1
        
        if count > 0:
            analysis["summary"]["avg_improvement"] = total_improvement / count
        
        return analysis
