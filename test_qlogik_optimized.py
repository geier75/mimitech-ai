#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK Optimierungs-Testskript

Dieses Skript testet die Performance und Genauigkeit 
des optimierten Q-LOGIK Frameworks.
"""

import os
import sys
import time
import numpy as np
import json
import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.Tests.QLogikOptimized")

# Importiere Q-LOGIK Komponenten
from miso.logic.qlogik_engine import (
    BayesianDecisionCore,
    FuzzyLogicUnit,
    SymbolMap,
    ConflictResolver,
    simple_emotion_weight,
    qlogik_decision,
    advanced_qlogik_decision
)

# Importiere GPU-Beschleunigung
from miso.logic.qlogik_gpu_acceleration import (
    to_tensor, to_numpy, matmul, attention, parallel_map, batch_process,
    get_backend_info, benchmark
)

# Importiere Memory-Optimierung
from miso.logic.qlogik_memory_optimization import (
    get_from_cache, put_in_cache, clear_cache, get_memory_stats
)

def test_bayesian_performance():
    """Testet die Performance der optimierten bayesianischen Entscheidungsfindung"""
    logger.info("\n=== Bayesianische Entscheidungsfindung Performance-Test ===")
    
    # Erstelle Testdaten
    num_tests = 1000
    hypotheses = ["h1", "h2", "h3"]
    
    # Erstelle BayesianDecisionCore mit Priors
    priors = {
        "h1": 0.3,
        "h2": 0.5,
        "h3": 0.2
    }
    bayesian = BayesianDecisionCore({"priors": priors})
    
    # Generiere Testanfragen
    test_queries = []
    for i in range(num_tests):
        hyp = hypotheses[i % len(hypotheses)]
        evidence = {
            f"evidence_{j}": {
                "value": np.random.random(),
                "weight": 1.0 + np.random.random()
            } for j in range(3)
        }
        test_queries.append({
            "hypothesis": hyp,
            "evidence": evidence
        })
    
    # Erste Ausführung (ohne Cache)
    clear_cache()
    logger.info("Führe Tests ohne Cache aus...")
    start_time = time.time()
    results_nocache = []
    
    for query in test_queries:
        results_nocache.append(bayesian.evaluate(query))
    
    nocache_time = time.time() - start_time
    logger.info(f"Zeit ohne Cache: {nocache_time:.4f}s für {num_tests} Anfragen")
    
    # Zweite Ausführung (mit Cache)
    logger.info("Führe Tests mit Cache aus...")
    start_time = time.time()
    results_cache = []
    
    for query in test_queries:
        results_cache.append(bayesian.evaluate(query))
    
    cache_time = time.time() - start_time
    logger.info(f"Zeit mit Cache: {cache_time:.4f}s für {num_tests} Anfragen")
    logger.info(f"Beschleunigung: {nocache_time / cache_time:.2f}x")
    
    # Cache-Statistiken
    stats = get_memory_stats()
    logger.info(f"Cache-Statistiken: {stats['memory_cache']['size']} Einträge im Memory-Cache")
    
    # Überprüfe Ergebniskonsistenz
    are_equal = all(abs(a - b) < 1e-10 for a, b in zip(results_nocache, results_cache))
    logger.info(f"Ergebnisse sind konsistent: {are_equal}")
    
    return {
        "nocache_time": nocache_time,
        "cache_time": cache_time,
        "speedup": nocache_time / cache_time,
        "consistent": are_equal
    }

def test_fuzzy_membership_functions():
    """Testet die neuen Zugehörigkeitsfunktionen"""
    logger.info("\n=== Fuzzy-Logik Zugehörigkeitsfunktionen-Test ===")
    
    # Erstelle FuzzyLogicUnit mit benutzerdefinierten Membership-Funktionen
    membership_functions = {
        "temperature": {
            "type": "gaussian",
            "params": {"mean": 22.0, "std_dev": 3.0}
        },
        "humidity": {
            "type": "triangular",
            "params": {"a": 30.0, "b": 50.0, "c": 70.0}
        },
        "brightness": {
            "type": "trapezoid",
            "params": {"a": 100, "b": 300, "c": 700, "d": 1000}
        },
        "pressure": {
            "type": "sigmoid",
            "params": {"center": 1000, "slope": 0.01}
        },
        "wind": {
            "type": "bell",
            "params": {"a": 5.0, "b": 2.0, "c": 10.0}
        },
        "noise": {
            "type": "linear",
            "params": {"min": 30, "max": 80}
        }
    }
    
    fuzzy = FuzzyLogicUnit({"membership_functions": membership_functions})
    
    # Teste verschiedene Werte für jede Funktion
    test_values = {
        "temperature": [18, 22, 26],  # Gaussian
        "humidity": [20, 50, 80],     # Triangular
        "brightness": [50, 200, 500, 900], # Trapezoid
        "pressure": [900, 1000, 1100],  # Sigmoid
        "wind": [5, 10, 15],          # Bell
        "noise": [20, 50, 90]         # Linear
    }
    
    # Evaluiere Zugehörigkeitsgrade
    results = {}
    for key, values in test_values.items():
        results[key] = []
        logger.info(f"\nTeste Zugehörigkeitsfunktion für '{key}':")
        
        for value in values:
            # Simuliere Signal-Format
            signal = {key: value}
            membership = fuzzy.score(signal)
            results[key].append((value, membership))
            logger.info(f"  Wert: {value}, Zugehörigkeitsgrad: {membership:.4f}")
    
    # Cache-Statistiken nach der Ausführung
    stats = get_memory_stats()
    logger.info(f"\nCache-Statistiken nach Fuzzy-Tests: {stats['memory_cache']['size']} Einträge im Memory-Cache")
    
    return results

# Definiere eine globale Funktion für prozessbasierte Parallelisierung
def process_complex_data(data):
    """CPU-intensive Berechnungsfunktion für Parallelisierungstests"""
    # Simuliere komplexe Berechnung
    result = 0
    for i in range(10000):
        result += np.sin(data + i * 0.001) * np.cos(data - i * 0.001)
    return result
    
def test_parallel_processing():
    """Testet die optimierte parallele Verarbeitung"""
    logger.info("\n=== Parallel Processing Optimierung Test ===")
    
    # Generiere Testdaten
    num_items = 100
    test_data = [i * 0.1 for i in range(num_items)]
    
    # Sequenzielle Verarbeitung
    logger.info("Führe sequenzielle Verarbeitung durch...")
    start_time = time.time()
    sequential_results = [process_complex_data(x) for x in test_data]
    sequential_time = time.time() - start_time
    logger.info(f"Sequenzielle Verarbeitung Zeit: {sequential_time:.4f}s")
    
    # Parallele Verarbeitung mit Threads
    logger.info("Führe parallele Verarbeitung mit Threads durch...")
    start_time = time.time()
    thread_results = parallel_map(process_complex_data, test_data, use_processes=False)
    thread_time = time.time() - start_time
    logger.info(f"Thread-basierte Verarbeitung Zeit: {thread_time:.4f}s")
    logger.info(f"Beschleunigung mit Threads: {sequential_time / thread_time:.2f}x")
    
    # Wir überspringen die Prozess-basierte Parallelisierung, um das Serialisierungsproblem zu vermeiden
    
    # Batch-Verarbeitung
    batch_size = 20
    logger.info(f"Führe Batch-Verarbeitung mit Batch-Größe {batch_size} durch...")
    
    # Batch-Verarbeitungsfunktion (diese wird nicht serialisiert, da sie nicht in Prozessen ausgeführt wird)
    def batch_processor(items):
        return [process_complex_data(x) for x in items]
    
    start_time = time.time()
    batch_results = batch_process(batch_processor, test_data, batch_size=batch_size)
    batch_time = time.time() - start_time
    logger.info(f"Batch-Verarbeitung Zeit: {batch_time:.4f}s")
    logger.info(f"Beschleunigung mit Batch-Verarbeitung: {sequential_time / batch_time:.2f}x")
    
    # Adaptive Batch-Verarbeitung
    logger.info("Führe adaptive Batch-Verarbeitung durch...")
    start_time = time.time()
    adaptive_results = batch_process(batch_processor, test_data, batch_size=batch_size, adaptive=True)
    adaptive_time = time.time() - start_time
    logger.info(f"Adaptive Batch-Verarbeitung Zeit: {adaptive_time:.4f}s")
    logger.info(f"Beschleunigung mit adaptiver Batch-Verarbeitung: {sequential_time / adaptive_time:.2f}x")
    
    return {
        "sequential_time": sequential_time,
        "thread_time": thread_time,
        "batch_time": batch_time,
        "adaptive_time": adaptive_time,
        "thread_speedup": sequential_time / thread_time,
        "batch_speedup": sequential_time / batch_time,
        "adaptive_speedup": sequential_time / adaptive_time
    }

def test_integrated_decision():
    """Testet die integrierte Entscheidungsfindung mit allen optimierten Komponenten"""
    logger.info("\n=== Integrierte Entscheidungsfindung Test ===")
    
    # Generiere komplexe Entscheidungskontexte
    num_contexts = 50
    test_contexts = []
    
    for i in range(num_contexts):
        risk = 0.3 + 0.5 * np.random.random()
        benefit = 0.2 + 0.6 * np.random.random()
        urgency = 0.4 + 0.4 * np.random.random()
        
        context = {
            "hypothesis": f"decision_{i % 5}",
            "evidence": {
                "risk": {"value": risk, "weight": 2.0},
                "benefit": {"value": benefit, "weight": 1.5},
                "urgency": {"value": urgency, "weight": 1.0},
                "confidence": {"value": 0.7, "weight": 0.8}
            },
            "metadata": {
                "importance": 0.8,
                "time_sensitivity": 0.6
            }
        }
        test_contexts.append(context)
    
    # Teste einfache Entscheidungsfindung
    logger.info("Teste einfache Q-LOGIK-Entscheidungsfindung...")
    start_time = time.time()
    simple_decisions = []
    
    for context in test_contexts:
        simple_decisions.append(qlogik_decision(context))
    
    simple_time = time.time() - start_time
    logger.info(f"Einfache Entscheidungszeit: {simple_time:.4f}s für {num_contexts} Kontexte")
    
    # Teste erweiterte Entscheidungsfindung
    logger.info("Teste erweiterte Q-LOGIK-Entscheidungsfindung...")
    start_time = time.time()
    advanced_decisions = []
    
    for context in test_contexts:
        advanced_decisions.append(advanced_qlogik_decision(context))
    
    advanced_time = time.time() - start_time
    logger.info(f"Erweiterte Entscheidungszeit: {advanced_time:.4f}s für {num_contexts} Kontexte")
    
    # Cache-Statistiken
    stats = get_memory_stats()
    logger.info(f"Cache-Statistiken: {stats['memory_cache']['size']} Einträge im Memory-Cache")
    
    # Entscheidungs-Statistiken
    decision_counts = {}
    for decision in simple_decisions:
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    
    logger.info(f"Entscheidungsverteilung: {decision_counts}")
    
    return {
        "simple_time": simple_time,
        "advanced_time": advanced_time,
        "ratio": advanced_time / simple_time,
        "decision_counts": decision_counts
    }

def run_all_tests():
    """Führt alle Tests aus und generiert Zusammenfassung"""
    logger.info("\n=== Q-LOGIK OPTIMIERUNGS-BENCHMARKS ===")
    logger.info(f"Startzeitpunkt: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Systeminformationen
    backend_info = get_backend_info()
    logger.info(f"\nSysteminformationen:")
    logger.info(f"  Backend: {backend_info.get('backend', 'unknown')}")
    logger.info(f"  CUDA verfügbar: {backend_info.get('cuda_available', False)}")
    logger.info(f"  MLX verfügbar: {backend_info.get('mlx_available', False)}")
    
    # Führe alle Tests aus
    clear_cache()  # Cache leeren vor Beginn
    
    results = {}
    
    # Bayesian Performance
    results["bayesian"] = test_bayesian_performance()
    
    # Fuzzy Membership
    results["fuzzy"] = test_fuzzy_membership_functions()
    
    # Parallel Processing
    results["parallel"] = test_parallel_processing()
    
    # Integrated Decision
    results["decision"] = test_integrated_decision()
    
    # Zusammenfassung der Ergebnisse
    logger.info("\n=== ZUSAMMENFASSUNG ===")
    logger.info(f"Bayesian Speedup: {results['bayesian']['speedup']:.2f}x")
    logger.info(f"Thread-basierte Parallelisierung: {results['parallel']['thread_speedup']:.2f}x")
    logger.info(f"Batch-basierte Parallelisierung: {results['parallel']['batch_speedup']:.2f}x")
    logger.info(f"Adaptive Batch-Verarbeitung: {results['parallel']['adaptive_speedup']:.2f}x")
    
    return results

if __name__ == "__main__":
    run_all_tests()
