#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Umfassende Integrationstests für PRISM-Engine

Dieses Modul implementiert vollständige Integrationstests für die PRISM-Engine
mit anderen Modulen (T-Mathematics, ECHO-PRIME, VX-CHRONOS) unter Verwendung
optimierter MLX-Tensor-Operationen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import unittest
import pytest
import logging
import time
import numpy as np
import uuid
import platform
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import MagicMock, patch
from contextlib import contextmanager
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MISO.tests.integration.test_prism_comprehensive")

# Füge Projektverzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Prüfe auf Apple Silicon
is_apple_silicon = False
try:
    is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
except Exception as e:
    logger.warning(f"Fehler bei der Erkennung von Apple Silicon: {e}")
    pass

# Prüfe auf MLX
has_mlx = False
try:
    import mlx.core
    has_mlx = True
except ImportError:
    logger.warning("MLX nicht verfügbar, einige Tests werden übersprungen")

# ZTM-Protokoll-Initialisierung (aktiviert für Integrationstests)
os.environ['MISO_ZTM_MODE'] = '1'  # ZTM aktiviert (war '0')
os.environ['MISO_ZTM_LOG_LEVEL'] = 'DEBUG'
# Stelle sicher, dass ZTM-Logs in das richtige Verzeichnis geschrieben werden
ztm_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(ztm_log_dir, exist_ok=True)
logger.info(f"Zero-Trust Monitoring (ZTM) aktiviert mit Log-Verzeichnis: {ztm_log_dir}")

# Importiere zu testende Module
try:
    from miso.simulation.prism_engine import PrismEngine
    from miso.simulation.prism_matrix import PrismMatrix
    from miso.math.t_mathematics.engine import TMathEngine
    from miso.math.t_mathematics.compat import TMathConfig
    from miso.math.t_mathematics.prism_integration import PrismSimulationEngine
    from miso.timeline.echo_prime_controller import EchoPrimeController
    # TimelineManager wird aus AlternativeTimelineBuilder verwendet
    from miso.timeline.echo_prime import AlternativeTimelineBuilder
    from miso.vxor.chronos_echo_prime_bridge import ChronosEchoBridge, get_bridge
    
    # Optimierungsmodule
    from miso.math.t_mathematics.optimizations.optimized_mlx_svd import compute_optimized_svd
    
    # Flag für verfügbare Module setzen
    HAS_ALL_DEPENDENCIES = True
except ImportError as e:
    logger.error(f"Fehler beim Import von Abhängigkeiten: {e}")
    HAS_ALL_DEPENDENCIES = False

# Benchmark-Dekorator
def benchmark(func):
    """Dekorator für Benchmarking von Funktionsaufrufen"""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.info(f"Funktion {func.__name__} ausgeführt in {elapsed:.6f} Sekunden")
        return result
    return wrapper

# Test-Fixtures
@pytest.fixture
def prism_engine():
    """Fixture für PrismEngine mit MLX"""
    config = {
        "use_mlx": True,
        "precision": "float16",
        "matrix_dimensions": 5
    }
    engine = PrismEngine(config)
    engine.start()
    yield engine
    engine.stop()

@pytest.fixture
def t_math_engine():
    """Fixture für TMathEngine mit MLX"""
    config = TMathConfig(
        precision="float16",
        optimize_for_apple_silicon=True
    )
    engine = TMathEngine(config)
    yield engine

@pytest.fixture
def echo_prime_controller():
    """Fixture für EchoPrimeController"""
    controller = EchoPrimeController()
    controller.initialize()
    yield controller

@pytest.fixture
def chronos_bridge(echo_prime_controller):
    """Fixture für ChronosEchoBridge"""
    bridge = get_bridge(echo_prime_controller)
    yield bridge

@pytest.fixture
def sample_timeline(echo_prime_controller):
    """Fixture für eine Beispiel-Timeline"""
    timeline_id = echo_prime_controller.create_timeline("Test Timeline")
    # Erstelle einfache Baumstruktur
    root_id = echo_prime_controller.add_node(
        timeline_id, 
        data={"type": "root", "value": 1.0},
        parent_id=None
    )
    child1_id = echo_prime_controller.add_node(
        timeline_id,
        data={"type": "decision", "value": 0.7},
        parent_id=root_id
    )
    child2_id = echo_prime_controller.add_node(
        timeline_id,
        data={"type": "decision", "value": 0.3},
        parent_id=root_id
    )
    
    # Füge weitere Ebene hinzu
    echo_prime_controller.add_node(
        timeline_id,
        data={"type": "outcome", "value": 0.9},
        parent_id=child1_id
    )
    echo_prime_controller.add_node(
        timeline_id,
        data={"type": "outcome", "value": 0.4},
        parent_id=child2_id
    )
    
    yield timeline_id
    
    # Bereinigung
    echo_prime_controller.delete_timeline(timeline_id)

# Hilfsfunktionen
def create_random_matrix(shape, dtype=np.float32):
    """Erstellt eine zufällige Matrix mit gegebener Form"""
    return np.random.random(shape).astype(dtype)

def create_tensor_batch(batch_size, dimensions, dtype=np.float32):
    """Erstellt einen Batch von Tensoren für Tests"""
    return [create_random_matrix((dimensions, dimensions), dtype) for _ in range(batch_size)]

# Toleranzbasierte Assertion für numerische Stabilität
def assert_close_numeric(expected, actual, rtol=1e-5, atol=1e-8, msg=None):
    """Prüft, ob zwei Zahlenwerte nahe beieinander liegen"""
    if not np.allclose(expected, actual, rtol=rtol, atol=atol):
        standardMsg = f"Werte nicht nahe genug: {expected} != {actual} mit rtol={rtol}, atol={atol}"
        raise AssertionError(standardMsg if msg is None else f"{msg}: {standardMsg}")


# =============================================================================
# 1. PRISM ↔ T-Mathematics Integrationstests
# =============================================================================
@pytest.mark.integration
@pytest.mark.tmath
class TestPrismTMathIntegration:
    """Tests für die Integration zwischen PRISM-Engine und T-Mathematics"""
    
    @pytest.mark.skipif(not is_apple_silicon or not has_mlx, 
                        reason="Benötigt Apple Silicon und MLX")
    def test_matrix_batch_operations(self, prism_engine, t_math_engine):
        """Test für Matrix-Batch-Operationen mit MLX"""
        # Erstelle Batch von Matrizen
        batch_size = 5
        dimensions = 10
        matrix_batch = create_tensor_batch(batch_size, dimensions)
        
        # Führe Batch-Operation mit PRISM durch
        result = prism_engine.integrate_with_t_mathematics(
            tensor_operation="batch_matmul",
            tensor_data={
                "matrices_a": matrix_batch,
                "matrices_b": matrix_batch
            }
        )
        
        # Verifiziere Ergebnisse
        assert result is not None
        assert "status" in result
        assert result["status"] == "success"
        assert "data" in result
        assert len(result["data"]) == batch_size
    
    @pytest.mark.skipif(not is_apple_silicon or not has_mlx, 
                        reason="Benötigt Apple Silicon und MLX")
    @pytest.mark.parametrize("dimensions", [5, 10, 15])
    def test_svd_computation(self, prism_engine, dimensions):
        """Test für SVD-Berechnungen mit verschiedenen Dimensionen"""
        # Erstelle Testmatrix
        test_matrix = create_random_matrix((dimensions, dimensions))
        
        # Führe SVD mit PRISM durch
        result = prism_engine.integrate_with_t_mathematics(
            tensor_operation="svd",
            tensor_data={"matrix": test_matrix}
        )
        
        # Verifiziere Ergebnisse
        assert result is not None
        assert "status" in result
        assert result["status"] == "success"
        assert "data" in result
        assert "u" in result["data"]
        assert "s" in result["data"]
        assert "vh" in result["data"]
        
        # Rekonstruiere Matrix und prüfe Genauigkeit
        u = result["data"]["u"]
        s = result["data"]["s"]
        vh = result["data"]["vh"]
        
        # Konvertiere s zu Diagonalmatrix
        s_matrix = np.zeros((dimensions, dimensions))
        np.fill_diagonal(s_matrix, s)
        
        # Rekonstruiere original
        reconstructed = u @ s_matrix @ vh
        
        # Prüfe Rekonstruktionsgenauigkeit
        assert np.allclose(test_matrix, reconstructed, rtol=1e-4, atol=1e-4)
    
    @benchmark
    def test_caching_performance(self, prism_engine):
        """Test für Caching-Effizienz bei wiederholten Operationen"""
        # Erstelle Testmatrix
        test_matrix = create_random_matrix((20, 20))
        
        # Erste Operation (ohne Cache)
        start_time = time.perf_counter()
        prism_engine.integrate_with_t_mathematics(
            tensor_operation="matrix_power",
            tensor_data={"matrix": test_matrix, "power": 3}
        )
        first_op_time = time.perf_counter() - start_time
        
        # Wiederholte Operation (mit Cache)
        start_time = time.perf_counter()
        prism_engine.integrate_with_t_mathematics(
            tensor_operation="matrix_power",
            tensor_data={"matrix": test_matrix, "power": 3}
        )
        cached_op_time = time.perf_counter() - start_time
        
        # Verifiziere, dass die zweite Operation schneller war
        assert cached_op_time < first_op_time
        logger.info(f"Cache-Beschleunigung: {first_op_time/cached_op_time:.2f}x")
        
        # Erwartet min. 20% Beschleunigung
        assert cached_op_time < 0.8 * first_op_time

    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                        reason="Nicht alle Abhängigkeiten verfügbar")
    def test_fallback_mechanism(self, prism_engine):
        """Test für Fallback-Mechanismen bei Hardware-Inkompatibilität"""
        # Erstelle Testmatrix
        test_matrix = create_random_matrix((10, 10))
        
        # Mock MLX-Fehler
        with patch('miso.math.t_mathematics.engine.TMathEngine.compute_tensor_operation') as mock_compute:
            # Simuliere Fehler in MLX
            mock_compute.side_effect = RuntimeError("MLX error")
            
            # Führe Operation durch, sollte auf Fallback zurückgreifen
            result = prism_engine.integrate_with_t_mathematics(
                tensor_operation="matrix_inverse",
                tensor_data={"matrix": test_matrix}
            )
            
            # Verifiziere Ergebnisse
            assert result is not None
            assert "status" in result
            assert result["status"] == "success"
            assert "data" in result
            
            # Prüfe, ob Ergebnis mathematisch sinnvoll ist
            inverse = result["data"]
            product = np.matmul(test_matrix, inverse)
            identity = np.eye(10)
            assert np.allclose(product, identity, rtol=1e-4, atol=1e-4)


# =============================================================================
# 2. PRISM ↔ ECHO-PRIME Integrationstests
# =============================================================================
@pytest.mark.integration
@pytest.mark.echoprime
class TestPrismEchoPrimeIntegration:
    """Tests für die Integration zwischen PRISM-Engine und ECHO-PRIME"""
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                        reason="Nicht alle Abhängigkeiten verfügbar")
    def test_timeline_synchronization(self, prism_engine, echo_prime_controller, sample_timeline):
        """Test für die Synchronisierung von Zeitlinien zwischen PRISM und ECHO-PRIME"""
        # Registriere Timeline bei PRISM
        prism_engine.register_timeline(echo_prime_controller.get_timeline(sample_timeline))
        
        # Prüfe, ob Timeline registriert wurde
        timeline_ids = prism_engine.get_registered_timeline_ids()
        assert sample_timeline in timeline_ids
        
        # Prüfe, ob Timeline-Daten korrekt abgerufen werden können
        timeline = prism_engine.get_registered_timeline(sample_timeline)
        assert timeline is not None
        assert timeline.id == sample_timeline
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                        reason="Nicht alle Abhängigkeiten verfügbar")
    def test_probability_calculation(self, prism_engine, echo_prime_controller, sample_timeline):
        """Test für die Wahrscheinlichkeitsberechnung von Zeitlinien"""
        # Registriere Timeline bei PRISM
        prism_engine.register_timeline(echo_prime_controller.get_timeline(sample_timeline))
        
        # Berechne Wahrscheinlichkeit
        probability = prism_engine.calculate_timeline_probability(sample_timeline)
        
        # Verifiziere Ergebnis
        assert 0.0 <= probability <= 1.0
        logger.info(f"Berechnete Wahrscheinlichkeit für Timeline {sample_timeline}: {probability}")
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                      reason="Nicht alle Abhängigkeiten verfügbar")
    def test_paradox_detection(self, prism_engine, echo_prime_controller, sample_timeline):
        """Test für die Paradoxerkennung in Zeitlinien"""
        # Registriere Timeline bei PRISM
        prism_engine.register_timeline(echo_prime_controller.get_timeline(sample_timeline))
        
        # Führe Paradoxerkennung durch
        paradoxes = prism_engine.detect_paradoxes(sample_timeline)
        
        # Verifiziere Ergebnis
        assert paradoxes is not None
        assert "status" in paradoxes
        assert "paradoxes" in paradoxes
        
        # Logge gefundene Paradoxe
        logger.info(f"Gefundene Paradoxe: {len(paradoxes['paradoxes'])}")
        for paradox in paradoxes["paradoxes"]:
            logger.info(f"Paradox: {paradox}")
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                        reason="Nicht alle Abhängigkeiten verfügbar")
    def test_timeline_forking(self, prism_engine, echo_prime_controller, sample_timeline):
        """Test für die Verzweigung von Zeitlinien"""
        # Registriere Timeline bei PRISM
        timeline = echo_prime_controller.get_timeline(sample_timeline)
        prism_engine.register_timeline(timeline)
        
        # Wähle einen Knoten für die Verzweigung
        node_id = timeline.root_id
        
        # Verzweige Timeline
        result = prism_engine.fork_timeline(sample_timeline, node_id, variation_factor=0.3)
        
        # Verifiziere Ergebnis
        assert result is not None
        assert "status" in result
        assert result["status"] == "success"
        assert "timeline_id" in result
        
        # Prüfe, ob neue Timeline existiert
        new_timeline_id = result["timeline_id"]
        new_timeline = echo_prime_controller.get_timeline(new_timeline_id)
        assert new_timeline is not None
        
        # Bereinige
        echo_prime_controller.delete_timeline(new_timeline_id)


# =============================================================================
# 3. PRISM ↔ VX-CHRONOS Integrationstests
# =============================================================================
@pytest.mark.integration
@pytest.mark.chronos
class TestPrismChronosIntegration:
    """Tests für die Integration zwischen PRISM-Engine und VX-CHRONOS über die Bridge"""
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                        reason="Nicht alle Abhängigkeiten verfügbar")
    def test_chronos_bridge_initialization(self, chronos_bridge):
        """Test für die Initialisierung der ChronosEchoBridge"""
        assert chronos_bridge is not None
        assert chronos_bridge.initialized
        assert chronos_bridge.echo_prime is not None
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                        reason="Nicht alle Abhängigkeiten verfügbar")
    def test_timeline_synchronization_bridge(self, chronos_bridge, echo_prime_controller, sample_timeline):
        """Test für die Synchronisierung von Zeitlinien über die Bridge"""
        # Synchronisiere Timeline mit VX-CHRONOS
        result = chronos_bridge.sync_timeline(sample_timeline)
        
        # Verifiziere Ergebnis
        assert result is True
        assert sample_timeline in chronos_bridge.timeline_mappings
        
        # Prüfe Timeline-Mapping
        chronos_timeline_id = chronos_bridge.timeline_mappings[sample_timeline]
        assert chronos_timeline_id is not None
        logger.info(f"Timeline {sample_timeline} wurde zu Chronos Timeline {chronos_timeline_id} gemappt")
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                        reason="Nicht alle Abhängigkeiten verfügbar")
    def test_chronos_optimizations(self, chronos_bridge, echo_prime_controller, sample_timeline):
        """Test für die Anwendung von VX-CHRONOS-Optimierungen"""
        # Synchronisiere Timeline mit VX-CHRONOS
        chronos_bridge.sync_timeline(sample_timeline)
        
        # Wende Optimierungen an
        optimizations = chronos_bridge.apply_chronos_optimizations(sample_timeline)
        
        # Verifiziere Ergebnis
        assert optimizations is not None
        logger.info(f"Chronos Optimierungsergebnisse: {optimizations}")
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                        reason="Nicht alle Abhängigkeiten verfügbar")
    def test_chronos_paradox_detection(self, chronos_bridge, echo_prime_controller, sample_timeline):
        """Test für die Paradoxerkennung mit VX-CHRONOS"""
        # Synchronisiere Timeline mit VX-CHRONOS
        chronos_bridge.sync_timeline(sample_timeline)
        
        # Führe Paradoxerkennung durch
        paradoxes = chronos_bridge.detect_paradoxes(sample_timeline)
        
        # Verifiziere Ergebnis
        assert paradoxes is not None
        logger.info(f"Chronos Paradoxerkennungsergebnisse: {paradoxes}")


# =============================================================================
# 4. End-to-End Integrationstests
# =============================================================================
@pytest.mark.integration
@pytest.mark.e2e
class TestEndToEndIntegration:
    """End-to-End-Integrationstests für das Gesamtsystem"""
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                        reason="Nicht alle Abhängigkeiten verfügbar")
    def test_complex_reality_forking(self, prism_engine, echo_prime_controller):
        """Test für komplexe Realitätsverzweigung mit allen Systemen"""
        # Erstelle komplexe Timeline
        timeline_id = echo_prime_controller.create_timeline("Complex Test Timeline")
        
        # Erstelle mehrschichtige Struktur
        root_id = echo_prime_controller.add_node(
            timeline_id, 
            data={"type": "decision_point", "value": 1.0},
            parent_id=None
        )
        
        # Erstelle Verzweigungspunkte
        branches = []
        for i in range(3):
            branch_id = echo_prime_controller.add_node(
                timeline_id,
                data={"type": "branch", "value": 0.33, "variant": i},
                parent_id=root_id
            )
            branches.append(branch_id)
            
            # Füge Blätter hinzu
            for j in range(2):
                echo_prime_controller.add_node(
                    timeline_id,
                    data={"type": "outcome", "value": 0.5, "option": j, "branch": i},
                    parent_id=branch_id
                )
        
        # Registriere bei PRISM
        timeline = echo_prime_controller.get_timeline(timeline_id)
        prism_engine.register_timeline(timeline)
        
        # Führe komplexe Simulation durch
        result = prism_engine.simulate_timeline(timeline_id, steps=10, variation_factor=0.2)
        
        # Verifiziere Ergebnis
        assert result is not None
        assert "status" in result
        assert result["status"] == "success"
        assert "simulation_results" in result
        
        # Erstelle Bridge und synchronisiere
        bridge = get_bridge(echo_prime_controller)
        bridge.sync_timeline(timeline_id)
        
        # Wende Chronos-Optimierungen an
        optimizations = bridge.apply_chronos_optimizations(timeline_id)
        
        # Erkenne Paradoxien mit PRISM
        paradoxes_prism = prism_engine.detect_paradoxes(timeline_id)
        
        # Erkenne Paradoxien mit Chronos
        paradoxes_chronos = bridge.detect_paradoxes(timeline_id)
        
        # Vergleiche Ergebnisse
        prism_count = len(paradoxes_prism.get("paradoxes", []))
        chronos_count = len(paradoxes_chronos.get("paradoxes", []))
        
        logger.info(f"PRISM erkannte {prism_count} Paradoxien")
        logger.info(f"CHRONOS erkannte {chronos_count} Paradoxien")
        
        # Bereinige
        echo_prime_controller.delete_timeline(timeline_id)


# =============================================================================
# 5. Performance-Benchmarks
# =============================================================================
@pytest.mark.benchmark
class TestPrismPerformanceBenchmarks:
    """Performance-Benchmarks für die PRISM-Engine"""
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES or not has_mlx, 
                        reason="Nicht alle Abhängigkeiten verfügbar")
    def test_mlx_vs_numpy_performance(self):
        """Benchmark für MLX vs. NumPy Performance bei Matrix-Operationen"""
        # Testmatrizen
        matrix_a = create_random_matrix((100, 100))
        matrix_b = create_random_matrix((100, 100))
        
        # NumPy-Performance messen
        start_time = time.perf_counter()
        for _ in range(100):
            np.matmul(matrix_a, matrix_b)
        numpy_time = time.perf_counter() - start_time
        
        # MLX-Performance messen (wenn verfügbar)
        if has_mlx:
            import mlx.core as mx
            mx_a = mx.array(matrix_a)
            mx_b = mx.array(matrix_b)
            
            start_time = time.perf_counter()
            for _ in range(100):
                result = mx.matmul(mx_a, mx_b)
                mx.eval(result)  # Ensure computation is executed
            mlx_time = time.perf_counter() - start_time
            
            # Vergleich loggen
            speedup = numpy_time / mlx_time
            logger.info(f"MLX vs. NumPy Speedup: {speedup:.2f}x")
            
            # Speedup sollte signifikant sein auf Apple Silicon
            if is_apple_silicon:
                assert speedup > 1.5
        else:
            logger.warning("MLX nicht verfügbar, Vergleichstest übersprungen")
    
    @pytest.mark.skipif(not HAS_ALL_DEPENDENCIES, 
                        reason="Nicht alle Abhängigkeiten verfügbar")
    def test_scalability_with_matrix_size(self, prism_engine):
        """Benchmark für Skalierbarkeit bei wachsender Matrixgröße"""
        results = {}
        
        # Teste verschiedene Matrixgrößen
        for size in [10, 50, 100, 200]:
            matrix = create_random_matrix((size, size))
            
            # Zeitmessung
            start_time = time.perf_counter()
            prism_engine.integrate_with_t_mathematics(
                tensor_operation="matrix_inverse",
                tensor_data={"matrix": matrix}
            )
            elapsed = time.perf_counter() - start_time
            
            results[size] = elapsed
            logger.info(f"Matrix {size}x{size}: {elapsed:.6f} Sekunden")
        
        # Analysiere Skalierbarkeit
        for i in range(len(results) - 1):
            size1 = list(results.keys())[i]
            size2 = list(results.keys())[i+1]
            time1 = results[size1]
            time2 = results[size2]
            
            # Theoretische kubische Komplexität für Matrix-Inversion
            expected_ratio = (size2/size1)**3
            actual_ratio = time2/time1
            
            logger.info(f"Skalierung von {size1} zu {size2}: Erwartet ~{expected_ratio:.2f}x, Tatsächlich {actual_ratio:.2f}x")


# Main ausführen, wenn Skript direkt aufgerufen wird
if __name__ == "__main__":
    pytest.main(["-v", __file__])
