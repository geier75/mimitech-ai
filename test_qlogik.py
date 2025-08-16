#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK Testskript

Dieses Skript testet die implementierten Q-LOGIK-Komponenten.
"""

import os
import sys
import time
import numpy as np

# Importiere GPU-Beschleunigung
from miso.logic.qlogik_gpu_acceleration import (
    to_tensor, to_numpy, matmul, attention, parallel_map, batch_process,
    get_backend_info, benchmark
)

# Importiere Speicheroptimierung
from miso.logic.qlogik_memory_optimization import (
    get_from_cache, put_in_cache, clear_cache, register_lazy_loader,
    checkpoint, checkpoint_function, get_memory_stats
)

# Importiere adaptive Optimierung
from miso.logic.qlogik_adaptive_optimizer import optimize

# Importiere neuronale Modelle (wenn verfügbar)
try:
    from miso.logic.qlogik_neural_cnn import create_cnn_model
    from miso.logic.qlogik_neural_rnn import create_rnn_model
    NEURAL_MODELS_AVAILABLE = True
except ImportError:
    NEURAL_MODELS_AVAILABLE = False

def test_gpu_acceleration():
    """Testet die GPU-Beschleunigung"""
    print("\n=== GPU-Beschleunigung Test ===")
    
    # Backend-Informationen
    info = get_backend_info()
    print(f"Aktives Backend: {info['backend']}")
    print(f"CUDA verfügbar: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA-Gerät: {info['cuda_device']}")
    print(f"MLX verfügbar: {info.get('mlx_available', False)}")
    
    # Benchmark für Matrix-Multiplikation
    print("\nBenchmark für Matrix-Multiplikation (500x500):")
    results = benchmark("matmul", size=500, iterations=3)
    
    for backend, result in results.items():
        print(f"  {backend}: {result['time']:.6f}s pro Iteration")
    
    # Test für parallele Verarbeitung
    print("\nTest für parallele Verarbeitung:")
    
    def process_item(x):
        # Simuliere eine rechenintensive Operation
        time.sleep(0.01)
        return x * x
    
    items = list(range(100))
    
    start_time = time.time()
    sequential_results = [process_item(x) for x in items]
    sequential_time = time.time() - start_time
    print(f"Sequentielle Verarbeitung: {sequential_time:.4f}s")
    
    start_time = time.time()
    parallel_results = parallel_map(process_item, items)
    parallel_time = time.time() - start_time
    print(f"Parallele Verarbeitung: {parallel_time:.4f}s")
    print(f"Beschleunigung: {sequential_time / parallel_time:.2f}x")

def test_memory_optimization():
    """Testet die Speicheroptimierung"""
    print("\n=== Speicheroptimierung Test ===")
    
    # Cache-Test
    print("\nCache-Test:")
    
    def expensive_computation(x):
        print(f"Führe teure Berechnung für {x} durch...")
        time.sleep(0.5)  # Simuliere lange Berechnung
        return x * x
    
    # Erste Ausführung (nicht im Cache)
    start_time = time.time()
    result1 = get_from_cache("square_10", lambda: expensive_computation(10))
    first_time = time.time() - start_time
    print(f"Erste Ausführung: {first_time:.4f}s, Ergebnis: {result1}")
    
    # Zweite Ausführung (aus dem Cache)
    start_time = time.time()
    result2 = get_from_cache("square_10", lambda: expensive_computation(10))
    second_time = time.time() - start_time
    print(f"Zweite Ausführung (Cache): {second_time:.4f}s, Ergebnis: {result2}")
    print(f"Beschleunigung durch Cache: {first_time / second_time:.2f}x")
    
    # LazyLoader-Test
    print("\nLazyLoader-Test:")
    
    def load_large_model():
        print("Lade großes Modell...")
        time.sleep(1.0)  # Simuliere langes Laden
        return {"name": "LargeModel", "parameters": 10000000}
    
    # Registriere LazyLoader
    model_loader = register_lazy_loader("large_model", load_large_model)
    print("LazyLoader registriert, Modell noch nicht geladen")
    
    # Lade Modell bei Bedarf
    start_time = time.time()
    model = model_loader()
    load_time = time.time() - start_time
    print(f"Modell geladen in {load_time:.4f}s: {model}")
    
    # Zweiter Aufruf (bereits geladen)
    start_time = time.time()
    model = model_loader()
    second_load_time = time.time() - start_time
    print(f"Zweiter Aufruf in {second_load_time:.4f}s")
    
    # Speicherstatistiken
    stats = get_memory_stats()
    print("\nSpeicherstatistiken:")
    print(f"Memory-Cache: {stats['memory_cache']['size']}/{stats['memory_cache']['capacity']} Einträge")
    print(f"Disk-Cache: {stats['disk_cache']['total_entries']} Einträge, "
          f"{stats['disk_cache']['total_size_mb']:.2f}/{stats['disk_cache']['max_size_mb']:.2f} MB "
          f"({stats['disk_cache']['usage_percent']:.2f}%)")
    print(f"LazyLoader: {stats['lazy_loaders']['loaded']}/{stats['lazy_loaders']['total']} geladen")

def test_neural_models():
    """Testet die neuronalen Modelle"""
    if not NEURAL_MODELS_AVAILABLE:
        print("\n=== Neuronale Modelle Test ===")
        print("Neuronale Modelle nicht verfügbar")
        return
    
    print("\n=== Neuronale Modelle Test ===")
    
    try:
        # CNN-Test
        print("\nCNN-Test:")
        cnn_config = {
            "input_channels": 3,
            "num_classes": 10
        }
        cnn_model = create_cnn_model("resnet", cnn_config)
        print(f"CNN-Modell erstellt: {cnn_model.model_name}")
        
        # Testdaten
        batch_size = 4
        input_data = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)
        
        # Vorhersage
        start_time = time.time()
        output = cnn_model.predict(input_data)
        inference_time = time.time() - start_time
        print(f"Inferenzzeit: {inference_time:.4f}s")
        print(f"Ausgabeform: {output.shape}")
        
        # RNN-Test
        print("\nRNN-Test:")
        rnn_config = {
            "input_size": 300,
            "hidden_size": 256,
            "num_classes": 5
        }
        rnn_model = create_rnn_model("lstm", rnn_config)
        print(f"RNN-Modell erstellt: {rnn_model.model_name}")
        
        # Testdaten
        seq_len = 10
        input_data = np.random.randn(batch_size, seq_len, 300).astype(np.float32)
        
        # Vorhersage
        start_time = time.time()
        output = rnn_model.predict(input_data)
        inference_time = time.time() - start_time
        print(f"Inferenzzeit: {inference_time:.4f}s")
        print(f"Ausgabeform: {output.shape}")
        
    except Exception as e:
        print(f"Fehler beim Testen der neuronalen Modelle: {e}")

def test_adaptive_optimization():
    """Testet die adaptive Optimierung"""
    print("\n=== Adaptive Optimierung Test ===")
    
    try:
        # Führe adaptive Optimierung durch
        print("Führe adaptive Optimierung durch...")
        result = optimize({"source": "test"})
        
        print(f"Optimierungsstrategie: {result['strategy']}")
        print(f"Verbesserung: {result['improvement']:.4f}")
        print(f"Ausführungszeit: {result['execution_time']:.4f}s")
        
    except Exception as e:
        print(f"Fehler bei adaptiver Optimierung: {e}")

def main():
    """Hauptfunktion"""
    print("=== Q-LOGIK Testskript ===")
    print(f"Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Führe Tests durch
    test_gpu_acceleration()
    test_memory_optimization()
    test_neural_models()
    test_adaptive_optimization()
    
    print("\n=== Tests abgeschlossen ===")

if __name__ == "__main__":
    main()
