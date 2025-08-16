#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Integrierte Optimierungstest

Umfassender Leistungstest für die optimierte Q-LOGIK-Integration,
inkl. Bayesian Caching, adaptive Parallelisierung und PRISM-ECHO-Integration.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys

# Füge Hauptverzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import uuid
import random
import json
import hashlib
import matplotlib.pyplot as plt
from datetime import datetime

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.Tests.IntegratedOptimizations")

# Importiere alle notwendigen Module
from miso.logic.qlogik_engine import BayesianDecisionCore, FuzzyLogicUnit, advanced_qlogik_decision
from miso.logic.qlogik_gpu_acceleration import smart_parallel_map, parallel_map, batch_process
from miso.simulation.prism_echo_prime_integration import qlogik_integration

class IntegratedOptimizationsTest:
    """Klasse zum Testen der integrierten Optimierungen im Q-LOGIK Framework"""
    
    def _generate_hash_key(self, obj: Any, prefix: str = "") -> str:
        """
        Erzeugt einen konsistenten Hash-Schlüssel für komplexe Datenstrukturen
        Diese Methode ist kompatibel mit der Implementierung in QLogikIntegrationLayer
        
        Args:
            obj: Zu hashende Daten
            prefix: Präfix für den Cache-Schlüssel
            
        Returns:
            Cache-Schlüssel als String
        """
        # Behandle None-Werte
        if obj is None:
            return f"{prefix}_none"
            
        try:
            # Versuche ein JSON-serialisierbares Objekt zu erstellen
            if isinstance(obj, dict):
                # Sortiere die Schlüssel für konsistente Hashes
                sorted_items = sorted(obj.items())
                serializable = [(str(k), self._make_serializable(v)) for k, v in sorted_items]
            elif isinstance(obj, (list, tuple)):
                serializable = [self._make_serializable(item) for item in obj]
            else:
                serializable = str(obj)
                
            # Erzeuge einen deterministischen Hash mit hashlib
            serialized = json.dumps(serializable, sort_keys=True).encode('utf-8')
            hash_value = hashlib.md5(serialized).hexdigest()
            return f"{prefix}_{hash_value}"
            
        except Exception as e:
            # Fallback für nicht serialisierbare Objekte
            logger.warning(f"Fehler bei Hash-Generierung: {e}")
            return f"{prefix}_{hash(str(obj))}"
    
    def _make_serializable(self, obj: Any) -> Union[Dict, List, str, int, float, bool, None]:
        """
        Transformiert ein Objekt in ein JSON-serialisierbares Format
        
        Args:
            obj: Zu transformierendes Objekt
            
        Returns:
            JSON-serialisierbares Objekt
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(i) for i in obj]
        else:
            # Für nicht-serialisierbare Objekte
            return str(obj)
    
    def __init__(self):
        """Initialisiert den Testfall mit optimierten Komponenten"""
        self.bayesian_core = BayesianDecisionCore({
            "priors": {
                "hypothesis_a": 0.7,
                "hypothesis_b": 0.3,
                "hypothesis_c": 0.5
            },
            "cache_ttl": 3600,  # 1 Stunde Cache-Gültigkeit
            "use_vector_calc": True,  # Aktiviere vektorisierte Berechnung
            "log_calculations": True   # Numerisch stabilere logarithmische Berechnung
        })
        
        self.fuzzy_unit = FuzzyLogicUnit({
            "membership_functions": {
                "temperature": {
                    "type": "gaussian",
                    "params": {"center": 22, "width": 4}
                },
                "risk": {
                    "type": "sigmoid",
                    "params": {"center": 0.5, "slope": 10}
                },
                "benefit": {
                    "type": "trapezoidal",
                    "params": {"a": 0.2, "b": 0.4, "c": 0.6, "d": 0.8}
                }
            }
        })
        
        # Initialisiere die optimierte QLogikIntegrationLayer
        self.qlogik_layer = qlogik_integration
        
        # Konfiguriere die Testumgebung für optimale Reproduzierbarkeit
        np.random.seed(42)  # Deterministischer Seed für reproduzierbare Tests
        random.seed(42)
        
        # Testdaten
        self.test_sizes = [10, 100, 1000, 10000]
        
        # Messungsresultate
        self.results = {}
        
    def cpu_intensive_task(self, x):
        """Rechenintensive Aufgabe für Parallelisierungstests"""
        # Markiere Funktion als CPU-intensiv für smart_parallel_map
        result = 0
        # Simuliere komplexe Matrixoperationen
        for i in range(1000):
            result += np.sin(x + i * 0.01) * np.cos(x - i * 0.01)
        return result
    cpu_intensive_task._cpu_intensive = True
    
    def io_intensive_task(self, x):
        """I/O-intensive Aufgabe für Parallelisierungstests"""
        # Markiere Funktion als I/O-intensiv für smart_parallel_map
        time.sleep(0.01)  # Simuliere I/O-Verzögerung
        return x * 2
    io_intensive_task._io_intensive = True
    
    def test_bayesian_optimization(self):
        """Testet die optimierte BayesianDecisionCore"""
        logger.info("\n=== Test: Bayesianische Entscheidungsoptimierung ===")
        
        # Generiere feste Testdaten mit deterministischen Werten statt zufälliger
        # um Caching effektiv und reproduzierbar zu testen
        test_hypotheses = ["hypothesis_a", "hypothesis_b", "hypothesis_c"]
        test_data = []
        
        # Wir erzeugen eine kleinere Anzahl eindeutiger Testfälle (50) und wiederholen sie mehrmals
        # um Cache-Hits zu garantieren
        unique_test_cases = []
        
        for i in range(50):
            # Erstelle deterministische Evidenzdaten
            evidence = {}
            for j in range(3):
                # Verwende feste Werte statt zufälliger für bessere Reproduzierbarkeit
                evidence[f"evidence_{j}"] = {
                    "value": (i*j % 10) / 10.0,  # Deterministische Werte zwischen 0 und 0.9
                    "weight": 1.0 + (j % 3) / 2.0  # Gewichte: 1.0, 1.5 oder 2.0
                }
            
            # Deterministische Auswahl der Hypothese
            hypothesis = test_hypotheses[i % len(test_hypotheses)]
            
            unique_test_cases.append({
                "hypothesis": hypothesis,
                "evidence": evidence
            })
        
        # Erweitere auf 1000 Testfälle durch Wiederholung (um Cache zu nutzen)
        for _ in range(20):
            test_data.extend(unique_test_cases)
        
        # Test 1: Geschwindigkeit ohne Cache
        self.bayesian_core.clear_cache_stats()  # Zurücksetzen der Cache-Statistiken
        start_time = time.time()
        no_cache_results = []
        for data in test_data:
            # Diese Aufrufe erzeugen Cache-Misses
            no_cache_results.append(self.bayesian_core.evaluate(data))
        no_cache_time = time.time() - start_time
        
        # Prüfe Cache - verwende einen String-basierten Schlüssel
        cache_key = f"bayes_{hash(str(sorted([(k, str(v)) for k, v in data.items()])))}" if data else "bayes_default"
        cache_stats_after_first_run = self.bayesian_core.get_cache_stats()
        
        # Test 2: Geschwindigkeit mit Cache
        start_time = time.time()
        with_cache_results = []
        for data in test_data:
            # Diese Aufrufe sollten Cache-Hits sein
            with_cache_results.append(self.bayesian_core.evaluate(data))
        with_cache_time = time.time() - start_time
        
        # Cache-Statistiken nach zweitem Durchlauf
        cache_stats_after_second_run = self.bayesian_core.get_cache_stats()
        
        # Speedup berechnen
        speedup = no_cache_time / with_cache_time if with_cache_time > 0 else float('inf')
        
        logger.info(f"Zeit ohne Cache: {no_cache_time:.4f}s")
        logger.info(f"Zeit mit Cache: {with_cache_time:.4f}s")
        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info(f"Cache-Statistik nach erstem Durchlauf: {cache_stats_after_first_run}")
        logger.info(f"Cache-Statistik nach zweitem Durchlauf: {cache_stats_after_second_run}")
        
        # Ergebnisse speichern
        self.results["bayesian"] = {
            "no_cache_time": no_cache_time,
            "with_cache_time": with_cache_time,
            "speedup": speedup,
            "cache_stats": cache_stats_after_second_run
        }
        
        # Validierung
        # Prüfen, ob die Ergebnisse mit und ohne Cache konsistent sind
        consistent = all(abs(a - b) < 1e-10 for a, b in zip(no_cache_results, with_cache_results))
        logger.info(f"Ergebnisse sind konsistent: {consistent}")
        
        return consistent
    
    def test_adaptive_parallelization(self):
        """Testet die adaptive Parallelisierungsstrategie"""
        logger.info("\n=== Test: Adaptive Parallelisierung ===")
        
        test_results = {}
        
        for size in self.test_sizes:
            logger.info(f"\nTestgröße: {size} Elemente")
            test_data = list(range(size))
            
            # Test 1: Sequentielle Verarbeitung
            start_time = time.time()
            seq_results = [self.cpu_intensive_task(x) for x in test_data]
            seq_time = time.time() - start_time
            logger.info(f"Sequentielle Zeit: {seq_time:.4f}s")
            
            # Test 2: Standardparallelisierung
            start_time = time.time()
            parallel_results = parallel_map(self.cpu_intensive_task, test_data)
            parallel_time = time.time() - start_time
            logger.info(f"Standard-Parallelzeit: {parallel_time:.4f}s")
            logger.info(f"Speedup (Standard): {seq_time / parallel_time:.2f}x")
            
            # Test 3: Adaptive Parallelisierung
            start_time = time.time()
            adaptive_results = smart_parallel_map(self.cpu_intensive_task, test_data)
            adaptive_time = time.time() - start_time
            logger.info(f"Adaptive Parallelzeit: {adaptive_time:.4f}s")
            logger.info(f"Speedup (Adaptiv): {seq_time / adaptive_time:.2f}x")
            
            # Testen mit I/O-intensiver Aufgabe
            logger.info("\nTest mit I/O-intensiver Aufgabe:")
            
            # Test 4: Sequentielle I/O
            start_time = time.time()
            seq_io_results = [self.io_intensive_task(x) for x in test_data[:min(size, 100)]]  # Begrenze auf 100 Elemente
            seq_io_time = time.time() - start_time
            logger.info(f"Sequentielle I/O-Zeit: {seq_io_time:.4f}s")
            
            # Test 5: Adaptive I/O-Parallelisierung
            start_time = time.time()
            adaptive_io_results = smart_parallel_map(self.io_intensive_task, test_data[:min(size, 100)])
            adaptive_io_time = time.time() - start_time
            logger.info(f"Adaptive I/O-Parallelzeit: {adaptive_io_time:.4f}s")
            logger.info(f"Speedup (I/O-Adaptiv): {seq_io_time / adaptive_io_time:.2f}x")
            
            # Ergebnisse für diese Größe speichern
            test_results[size] = {
                "seq_time": seq_time,
                "parallel_time": parallel_time,
                "adaptive_time": adaptive_time,
                "parallel_speedup": seq_time / parallel_time,
                "adaptive_speedup": seq_time / adaptive_time,
                "seq_io_time": seq_io_time,
                "adaptive_io_time": adaptive_io_time,
                "io_speedup": seq_io_time / adaptive_io_time
            }
        
        # Speichere Gesamtergebnisse
        self.results["parallelization"] = test_results
        
        return True
    
    def test_qlogik_prism_echo_integration(self):
        """Testet die Integration zwischen Q-LOGIK, PRISM und ECHO-PRIME"""
        logger.info("\n=== Test: Q-LOGIK, PRISM, ECHO-PRIME Integration ===")
        
        # Testdaten für Entscheidungen mit deterministischen Werten
        decision_contexts = []
        simulation_types = ["standard", "quantum", "probabilistic"]
        
        for i in range(50):
            # Erstelle deterministische Simulationskontexte
            decision_contexts.append({
                "timeline_id": f"timeline_{i}",
                "confidence_factor": (i % 10) / 10.0,  # Werte von 0.0 bis 0.9
                "risk_assessment": ((i+5) % 10) / 10.0,  # Versetzt von confidence_factor
                "probability_threshold": 0.7,
                "alternative_paths": (i % 5) + 1,  # 1 bis 5
                "metadata": {
                    "simulation_type": simulation_types[i % len(simulation_types)],
                    "complexity": (i % 10) + 1,  # 1 bis 10
                    "priority": (i % 5) + 1  # 1 bis 5
                }
            })
        
        # Test 1: Bayessche Wahrscheinlichkeitsberechnung über die Integrationsschicht
        start_time = time.time()
        bayes_results = []
        for context in decision_contexts:
            data = {
                "hypothesis": "integration_success",
                "evidence": {
                    "context_quality": {"value": context["confidence_factor"], "weight": 1.0},
                    "risk_level": {"value": context["risk_assessment"], "weight": 0.8},
                    "alternatives": {"value": context["alternative_paths"] / 5.0, "weight": 0.5}
                }
            }
            bayes_results.append(qlogik_integration.evaluate_bayesian_probability(data))
        bayes_integration_time = time.time() - start_time
        
        logger.info(f"Bayessche Integrationszeit: {bayes_integration_time:.4f}s für {len(decision_contexts)} Kontexte")
        logger.info(f"Durchschnittliche Wahrscheinlichkeit: {sum(bayes_results) / len(bayes_results):.4f}")
        
        # Test 2: Fuzzy-Logik-Integration
        start_time = time.time()
        fuzzy_results = []
        for context in decision_contexts:
            signal = {
                "risk": context["risk_assessment"],
                "confidence": context["confidence_factor"],
                "complexity": context["metadata"]["complexity"] / 10.0,
                "priority": context["metadata"]["priority"] / 5.0
            }
            # Prüfe Cache - verwende einen String-basierten Schlüssel
            cache_key = f"fuzzy_{hash(str(sorted([(k, str(v)) for k, v in signal.items()])))}" if signal else "fuzzy_default"
            fuzzy_results.append(qlogik_integration.evaluate_fuzzy_truth(signal))
        fuzzy_integration_time = time.time() - start_time
        
        logger.info(f"Fuzzy-Integrationszeit: {fuzzy_integration_time:.4f}s für {len(decision_contexts)} Kontexte")
        logger.info(f"Durchschnittlicher Wahrheitsgrad: {sum(fuzzy_results) / len(fuzzy_results):.4f}")
        
        # Test 3: Vollständige Entscheidungsintegration
        start_time = time.time()
        decision_results = []
        decision_distribution = {}
        
        for context in decision_contexts:
            result = qlogik_integration.decision_for_context(context)
            decision_results.append(result)
            
            # Erfasse Verteilung der Entscheidungen
            decision = result.get("decision", "UNBESTIMMT")
            decision_distribution[decision] = decision_distribution.get(decision, 0) + 1
            
        decision_integration_time = time.time() - start_time
        
        logger.info(f"Entscheidungsintegrationszeit: {decision_integration_time:.4f}s für {len(decision_contexts)} Kontexte")
        logger.info(f"Entscheidungsverteilung: {decision_distribution}")
        
        # Cache validieren
        qlogik_integration.clear_cache()
        logger.info("Cache der Integrationsschicht zurückgesetzt")
        
        # Ergebnisse speichern
        self.results["integration"] = {
            "bayes_time": bayes_integration_time,
            "fuzzy_time": fuzzy_integration_time,
            "decision_time": decision_integration_time,
            "decision_distribution": decision_distribution,
            "avg_probability": sum(bayes_results) / len(bayes_results),
            "avg_truth_degree": sum(fuzzy_results) / len(fuzzy_results)
        }
        
        return True
    
    def run_all_tests(self):
        """Führt alle Tests in optimierter Reihenfolge aus"""
        logger.info("STARTE INTEGRIERTE OPTIMIERUNGSTESTS")
        logger.info(f"Zeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        success = True
        start_time = time.time()
        
        # Test 1: Bayesianische Optimierung
        try:
            bayesian_success = self.test_bayesian_optimization()
            success = success and bayesian_success
        except Exception as e:
            logger.error(f"Fehler im Bayesianischen Test: {e}")
            success = False
        
        # Test 2: Adaptive Parallelisierung
        try:
            parallel_success = self.test_adaptive_parallelization()
            success = success and parallel_success
        except Exception as e:
            logger.error(f"Fehler im Parallelisierungstest: {e}")
            success = False
        
        # Test 3: Q-LOGIK, PRISM, ECHO-PRIME Integration
        try:
            integration_success = self.test_qlogik_prism_echo_integration()
            success = success and integration_success
        except Exception as e:
            logger.error(f"Fehler im Integrationstest: {e}")
            success = False
        
        total_time = time.time() - start_time
        logger.info(f"\nALLE TESTS ABGESCHLOSSEN. Dauer: {total_time:.2f}s")
        logger.info(f"Gesamtergebnis: {'ERFOLGREICH' if success else 'FEHLER AUFGETRETEN'}")
        
        # Ergebniszusammenfassung
        self.summarize_results()
        
        return success, self.results
    
    def summarize_results(self):
        """Fasst die Testergebnisse zusammen"""
        logger.info("\n====== ZUSAMMENFASSUNG DER OPTIMIERUNGSERGEBNISSE ======")
        
        # Bayesianische Optimierung
        if "bayesian" in self.results:
            bayes = self.results["bayesian"]
            logger.info("\n1. Bayesianische Entscheidungsoptimierung:")
            logger.info(f"   - Speedup durch Caching: {bayes['speedup']:.2f}x")
            logger.info(f"   - Cache-Trefferrate: {bayes['cache_stats']['hit_ratio']:.2%}")
            logger.info(f"   - Cache-Nutzung: {bayes['cache_stats']['hits']} Treffer, {bayes['cache_stats']['misses']} Fehlschläge")
        
        # Parallelisierungsoptimierung
        if "parallelization" in self.results:
            parallel = self.results["parallelization"]
            logger.info("\n2. Adaptive Parallelisierungsoptimierung:")
            
            for size in self.test_sizes:
                if size in parallel:
                    result = parallel[size]
                    logger.info(f"   - Datengröße {size}:")
                    logger.info(f"     * Standard-Parallelisierung: {result['parallel_speedup']:.2f}x Speedup")
                    logger.info(f"     * Adaptive Parallelisierung: {result['adaptive_speedup']:.2f}x Speedup")
                    if "io_speedup" in result:
                        logger.info(f"     * I/O-Parallelisierung: {result['io_speedup']:.2f}x Speedup")
        
        # Integrationsoptimierung
        if "integration" in self.results:
            integ = self.results["integration"]
            logger.info("\n3. Q-LOGIK, PRISM, ECHO-PRIME Integrationsoptimierung:")
            logger.info(f"   - Bayessche Integration: {len(integ.get('decision_distribution', {})) * 50 / integ['bayes_time']:.2f} Entscheidungen/s")
            logger.info(f"   - Fuzzy-Logik Integration: {len(integ.get('decision_distribution', {})) * 50 / integ['fuzzy_time']:.2f} Bewertungen/s")
            logger.info(f"   - Vollständige Entscheidungsintegration: {len(integ.get('decision_distribution', {})) * 50 / integ['decision_time']:.2f} Kontexte/s")
            
            if "decision_distribution" in integ:
                logger.info("   - Entscheidungsverteilung:")
                for decision, count in integ["decision_distribution"].items():
                    logger.info(f"     * {decision}: {count} ({count / 50:.1%})")
        
        logger.info("\n====================================================")

def plot_results(results):
    """Visualisiert die Testergebnisse"""
    try:
        # Erzeuge Verzeichnis für Plots, falls nicht vorhanden
        os.makedirs("test_results", exist_ok=True)
        
        # Plot 1: Bayesianische Optimierung
        if "bayesian" in results:
            plt.figure(figsize=(10, 6))
            bayes = results["bayesian"]
            plt.bar(["Ohne Cache", "Mit Cache"], 
                   [bayes["no_cache_time"], bayes["with_cache_time"]], 
                   color=["#3498db", "#2ecc71"])
            plt.title("Bayesianische Entscheidungsoptimierung")
            plt.ylabel("Zeit (s)")
            plt.text(1, bayes["with_cache_time"] + 0.02, 
                    f"Speedup: {bayes['speedup']:.2f}x", 
                    ha='center', fontweight='bold')
            plt.tight_layout()
            plt.savefig("test_results/bayesian_optimization.png")
            
        # Plot 2: Parallelisierungsoptimierung
        if "parallelization" in results:
            parallel = results["parallelization"]
            plt.figure(figsize=(12, 8))
            
            # Extrahiere Daten für den Plot
            sizes = sorted(parallel.keys())
            seq_times = [parallel[size]["seq_time"] for size in sizes]
            parallel_times = [parallel[size]["parallel_time"] for size in sizes]
            adaptive_times = [parallel[size]["adaptive_time"] for size in sizes]
            
            # Erstelle Balkendiagramm
            x = np.arange(len(sizes))
            width = 0.25
            
            plt.bar(x - width, seq_times, width, label='Sequentiell', color='#3498db')
            plt.bar(x, parallel_times, width, label='Standard-Parallel', color='#2ecc71')
            plt.bar(x + width, adaptive_times, width, label='Adaptiv-Parallel', color='#e74c3c')
            
            plt.title("Vergleich der Parallelisierungsstrategien")
            plt.xlabel("Datensatzgröße")
            plt.ylabel("Zeit (s)")
            plt.xticks(x, sizes)
            plt.legend()
            plt.tight_layout()
            plt.savefig("test_results/parallelization_comparison.png")
            
            # Plot für Speedup
            plt.figure(figsize=(12, 8))
            parallel_speedup = [parallel[size]["parallel_speedup"] for size in sizes]
            adaptive_speedup = [parallel[size]["adaptive_speedup"] for size in sizes]
            
            plt.plot(sizes, parallel_speedup, 'o-', label='Standard-Speedup', linewidth=2, color='#2ecc71')
            plt.plot(sizes, adaptive_speedup, 's-', label='Adaptiv-Speedup', linewidth=2, color='#e74c3c')
            
            plt.title("Speedup der Parallelisierungsstrategien")
            plt.xlabel("Datensatzgröße")
            plt.ylabel("Speedup (x-fach)")
            plt.xscale('log')  # Logarithmische Skala für bessere Visualisierung
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig("test_results/parallelization_speedup.png")
            
        # Plot 3: Integration
        if "integration" in results:
            integ = results["integration"]
            plt.figure(figsize=(10, 6))
            
            # Zeitvergleich der Integrationsmethoden
            plt.bar(["Bayesianische", "Fuzzy-Logik", "Entscheidung"], 
                   [integ["bayes_time"], integ["fuzzy_time"], integ["decision_time"]], 
                   color=["#3498db", "#2ecc71", "#e74c3c"])
            plt.title("Ausführungszeit der Q-LOGIK-Integrationen")
            plt.ylabel("Zeit (s)")
            plt.tight_layout()
            plt.savefig("test_results/integration_time.png")
            
            # Entscheidungsverteilung
            if "decision_distribution" in integ:
                plt.figure(figsize=(10, 6))
                distribution = integ["decision_distribution"]
                plt.pie(distribution.values(), labels=distribution.keys(), autopct='%1.1f%%', 
                       startangle=90, colors=plt.cm.Paired(np.arange(len(distribution))))
                plt.title("Verteilung der Q-LOGIK-Entscheidungen")
                plt.axis('equal')  # Gleiche Verhältnisse für Kreisform
                plt.tight_layout()
                plt.savefig("test_results/decision_distribution.png")
        
        logger.info("Ergebnisvisualisierungen wurden in test_results/ gespeichert.")
    except Exception as e:
        logger.error(f"Fehler bei der Visualisierung: {e}")

if __name__ == "__main__":
    test = IntegratedOptimizationsTest()
    success, results = test.run_all_tests()
    
    # Visualisierung der Ergebnisse
    try:
        plot_results(results)
    except ImportError:
        logger.warning("Matplotlib nicht verfügbar. Keine Visualisierung erstellt.")
    
    sys.exit(0 if success else 1)
