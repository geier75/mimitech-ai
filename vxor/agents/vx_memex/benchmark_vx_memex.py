#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MEMEX: Performance Benchmarks
--------------------------------
Leistungstests für das VX-MEMEX Gedächtnismodul.
Misst die Leistung aller Komponenten und optimiert für Apple Silicon M4 Max.

Führt Benchmarks für folgende Operationen durch:
- Speichern von Einträgen
- Abrufen von Einträgen
- Semantische Suche
- Zeitbasierte Suche
- Schlagwortbasierte Suche
- Speicherverwaltung
- Parallelverarbeitung

Optimiert für Apple Silicon M4 Max.
"""

import time
import random
import string
import json
import os
import sys
import multiprocessing
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple

# Pfad zum Projektverzeichnis hinzufügen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Module importieren
from memory_core import MemoryCore
from semantic_store import SemanticStore
from episodic_store import EpisodicStore
from working_memory import WorkingMemory
from vxor_bridge import VXORBridge

# Konstanten
DEFAULT_VECTOR_DIM = 384
DEFAULT_NUM_ENTRIES = 10000
DEFAULT_QUERY_COUNT = 100
DEFAULT_BATCH_SIZES = [1, 10, 100, 1000]
DEFAULT_OUTPUT_DIR = "/home/ubuntu/vXor_Modules/VX-MEMEX/benchmark_results"

# Globale Variablen
results = {}
use_mlx = False

def generate_random_text(length: int = 100) -> str:
    """
    Generiert zufälligen Text mit der angegebenen Länge.
    
    Args:
        length: Länge des zu generierenden Texts
        
    Returns:
        Zufälliger Text
    """
    words = []
    word_count = length // 5  # Durchschnittliche Wortlänge von 5 Zeichen
    
    for _ in range(word_count):
        word_length = random.randint(2, 10)
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
        words.append(word)
    
    return ' '.join(words)

def generate_random_tags(count: int = 3) -> List[str]:
    """
    Generiert eine Liste mit zufälligen Schlagwörtern.
    
    Args:
        count: Anzahl der zu generierenden Schlagwörter
        
    Returns:
        Liste mit zufälligen Schlagwörtern
    """
    tags = []
    
    for _ in range(count):
        tag_length = random.randint(3, 10)
        tag = ''.join(random.choice(string.ascii_lowercase) for _ in range(tag_length))
        tags.append(tag)
    
    return tags

def benchmark_semantic_store(num_entries: int, batch_sizes: List[int], query_count: int) -> Dict[str, Any]:
    """
    Führt Benchmarks für den semantischen Speicher durch.
    
    Args:
        num_entries: Anzahl der zu speichernden Einträge
        batch_sizes: Liste mit Batch-Größen für das Speichern
        query_count: Anzahl der Suchanfragen
        
    Returns:
        Dictionary mit Benchmark-Ergebnissen
    """
    print(f"Benchmark: Semantischer Speicher (MLX: {use_mlx})")
    
    results = {
        'store': {},
        'retrieve_by_id': {},
        'retrieve_by_query': {},
        'retrieve_by_tags': {}
    }
    
    # Semantischen Speicher initialisieren
    store = SemanticStore({'vector_dim': DEFAULT_VECTOR_DIM})
    
    # Einträge für verschiedene Batch-Größen speichern
    for batch_size in batch_sizes:
        if batch_size > num_entries:
            continue
        
        print(f"  Speichern von {num_entries} Einträgen in Batches von {batch_size}...")
        
        entries = []
        entry_ids = []
        
        # Einträge generieren
        for _ in range(num_entries):
            content = generate_random_text(random.randint(50, 200))
            tags = generate_random_tags(random.randint(1, 5))
            importance = random.random()
            
            entries.append({
                'content': content,
                'tags': tags,
                'importance': importance
            })
        
        # Zeit für das Speichern messen
        start_time = time.time()
        
        for i in range(0, num_entries, batch_size):
            batch = entries[i:i+batch_size]
            
            for entry in batch:
                entry_id = store.store(
                    content=entry['content'],
                    tags=entry['tags'],
                    importance=entry['importance']
                )
                entry_ids.append(entry_id)
        
        end_time = time.time()
        
        # Ergebnisse speichern
        total_time = end_time - start_time
        entries_per_second = num_entries / total_time
        
        results['store'][batch_size] = {
            'total_time': total_time,
            'entries_per_second': entries_per_second
        }
        
        print(f"    Zeit: {total_time:.2f}s, Einträge/s: {entries_per_second:.2f}")
    
    # Einträge nach ID abrufen
    print(f"  Abrufen von {query_count} Einträgen nach ID...")
    
    # Zufällige IDs auswählen
    random_ids = random.sample(entry_ids, min(query_count, len(entry_ids)))
    
    # Zeit für das Abrufen messen
    start_time = time.time()
    
    for entry_id in random_ids:
        store.retrieve_by_id(entry_id)
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    queries_per_second = len(random_ids) / total_time
    
    results['retrieve_by_id'] = {
        'total_time': total_time,
        'queries_per_second': queries_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Abfragen/s: {queries_per_second:.2f}")
    
    # Semantische Suche durchführen
    print(f"  Durchführen von {query_count} semantischen Suchanfragen...")
    
    # Zufällige Suchanfragen generieren
    queries = [generate_random_text(random.randint(10, 50)) for _ in range(query_count)]
    
    # Zeit für die semantische Suche messen
    start_time = time.time()
    
    for query in queries:
        store.retrieve(query, limit=10)
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    queries_per_second = query_count / total_time
    
    results['retrieve_by_query'] = {
        'total_time': total_time,
        'queries_per_second': queries_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Abfragen/s: {queries_per_second:.2f}")
    
    # Suche nach Schlagwörtern durchführen
    print(f"  Durchführen von {query_count} Schlagwort-Suchanfragen...")
    
    # Alle verwendeten Schlagwörter sammeln
    all_tags = set()
    for entry in entries:
        all_tags.update(entry['tags'])
    
    all_tags = list(all_tags)
    
    # Zufällige Schlagwort-Kombinationen generieren
    tag_queries = []
    for _ in range(query_count):
        num_tags = random.randint(1, 3)
        if all_tags:
            tag_query = random.sample(all_tags, min(num_tags, len(all_tags)))
            tag_queries.append(tag_query)
    
    # Zeit für die Schlagwort-Suche messen
    start_time = time.time()
    
    for tag_query in tag_queries:
        store.retrieve_by_tags(tag_query, limit=10)
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    queries_per_second = len(tag_queries) / total_time
    
    results['retrieve_by_tags'] = {
        'total_time': total_time,
        'queries_per_second': queries_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Abfragen/s: {queries_per_second:.2f}")
    
    return results

def benchmark_episodic_store(num_entries: int, batch_sizes: List[int], query_count: int) -> Dict[str, Any]:
    """
    Führt Benchmarks für den episodischen Speicher durch.
    
    Args:
        num_entries: Anzahl der zu speichernden Einträge
        batch_sizes: Liste mit Batch-Größen für das Speichern
        query_count: Anzahl der Suchanfragen
        
    Returns:
        Dictionary mit Benchmark-Ergebnissen
    """
    print(f"Benchmark: Episodischer Speicher (MLX: {use_mlx})")
    
    results = {
        'store': {},
        'retrieve_by_id': {},
        'retrieve_by_time_range': {},
        'retrieve_latest': {}
    }
    
    # Episodischen Speicher initialisieren
    store = EpisodicStore({'vector_dim': DEFAULT_VECTOR_DIM, 'max_entries': num_entries * 2})
    
    # Einträge für verschiedene Batch-Größen speichern
    for batch_size in batch_sizes:
        if batch_size > num_entries:
            continue
        
        print(f"  Speichern von {num_entries} Einträgen in Batches von {batch_size}...")
        
        entries = []
        entry_ids = []
        timestamps = []
        
        # Einträge generieren
        now = time.time()
        for i in range(num_entries):
            content = generate_random_text(random.randint(50, 200))
            tags = generate_random_tags(random.randint(1, 5))
            importance = random.random()
            timestamp = now - random.randint(0, 86400 * 30)  # Zufälliger Zeitpunkt innerhalb der letzten 30 Tage
            
            entries.append({
                'content': content,
                'tags': tags,
                'importance': importance,
                'timestamp': timestamp
            })
            timestamps.append(timestamp)
        
        # Zeit für das Speichern messen
        start_time = time.time()
        
        for i in range(0, num_entries, batch_size):
            batch = entries[i:i+batch_size]
            
            for entry in batch:
                entry_id = store.store(
                    content=entry['content'],
                    tags=entry['tags'],
                    importance=entry['importance'],
                    timestamp=entry['timestamp']
                )
                entry_ids.append(entry_id)
        
        end_time = time.time()
        
        # Ergebnisse speichern
        total_time = end_time - start_time
        entries_per_second = num_entries / total_time
        
        results['store'][batch_size] = {
            'total_time': total_time,
            'entries_per_second': entries_per_second
        }
        
        print(f"    Zeit: {total_time:.2f}s, Einträge/s: {entries_per_second:.2f}")
    
    # Einträge nach ID abrufen
    print(f"  Abrufen von {query_count} Einträgen nach ID...")
    
    # Zufällige IDs auswählen
    random_ids = random.sample(entry_ids, min(query_count, len(entry_ids)))
    
    # Zeit für das Abrufen messen
    start_time = time.time()
    
    for entry_id in random_ids:
        store.retrieve_by_id(entry_id)
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    queries_per_second = len(random_ids) / total_time
    
    results['retrieve_by_id'] = {
        'total_time': total_time,
        'queries_per_second': queries_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Abfragen/s: {queries_per_second:.2f}")
    
    # Suche nach Zeitbereich durchführen
    print(f"  Durchführen von {query_count} Zeitbereich-Suchanfragen...")
    
    # Zeitbereich-Anfragen generieren
    time_range_queries = []
    timestamps.sort()
    
    for _ in range(query_count):
        # Zufälligen Zeitbereich auswählen
        start_idx = random.randint(0, len(timestamps) - 2)
        end_idx = random.randint(start_idx + 1, len(timestamps) - 1)
        
        start_time_query = timestamps[start_idx]
        end_time_query = timestamps[end_idx]
        
        time_range_queries.append((start_time_query, end_time_query))
    
    # Zeit für die Zeitbereich-Suche messen
    start_time = time.time()
    
    for start_time_query, end_time_query in time_range_queries:
        store.retrieve_by_time_range(start_time_query, end_time_query, limit=10)
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    queries_per_second = len(time_range_queries) / total_time
    
    results['retrieve_by_time_range'] = {
        'total_time': total_time,
        'queries_per_second': queries_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Abfragen/s: {queries_per_second:.2f}")
    
    # Neueste Einträge abrufen
    print(f"  Abrufen der neuesten Einträge ({query_count} Anfragen)...")
    
    # Zeit für das Abrufen der neuesten Einträge messen
    start_time = time.time()
    
    for _ in range(query_count):
        limit = random.randint(1, 20)
        store.retrieve_latest(limit=limit)
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    queries_per_second = query_count / total_time
    
    results['retrieve_latest'] = {
        'total_time': total_time,
        'queries_per_second': queries_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Abfragen/s: {queries_per_second:.2f}")
    
    return results

def benchmark_working_memory(num_entries: int, batch_sizes: List[int], query_count: int) -> Dict[str, Any]:
    """
    Führt Benchmarks für das Arbeitsgedächtnis durch.
    
    Args:
        num_entries: Anzahl der zu speichernden Einträge
        batch_sizes: Liste mit Batch-Größen für das Speichern
        query_count: Anzahl der Suchanfragen
        
    Returns:
        Dictionary mit Benchmark-Ergebnissen
    """
    print(f"Benchmark: Arbeitsgedächtnis (MLX: {use_mlx})")
    
    results = {
        'store': {},
        'retrieve_by_id': {},
        'get_most_important': {},
        'refresh': {}
    }
    
    # Arbeitsgedächtnis initialisieren
    memory = WorkingMemory({
        'vector_dim': DEFAULT_VECTOR_DIM,
        'ttl_default': 3600,
        'max_entries': num_entries * 2
    })
    
    # Einträge für verschiedene Batch-Größen speichern
    for batch_size in batch_sizes:
        if batch_size > num_entries:
            continue
        
        print(f"  Speichern von {num_entries} Einträgen in Batches von {batch_size}...")
        
        entries = []
        entry_ids = []
        
        # Einträge generieren
        for _ in range(num_entries):
            content = generate_random_text(random.randint(50, 200))
            tags = generate_random_tags(random.randint(1, 5))
            importance = random.random()
            
            entries.append({
                'content': content,
                'tags': tags,
                'importance': importance
            })
        
        # Zeit für das Speichern messen
        start_time = time.time()
        
        for i in range(0, num_entries, batch_size):
            batch = entries[i:i+batch_size]
            
            for entry in batch:
                entry_id = memory.store(
                    content=entry['content'],
                    tags=entry['tags'],
                    importance=entry['importance']
                )
                entry_ids.append(entry_id)
        
        end_time = time.time()
        
        # Ergebnisse speichern
        total_time = end_time - start_time
        entries_per_second = num_entries / total_time
        
        results['store'][batch_size] = {
            'total_time': total_time,
            'entries_per_second': entries_per_second
        }
        
        print(f"    Zeit: {total_time:.2f}s, Einträge/s: {entries_per_second:.2f}")
    
    # Einträge nach ID abrufen
    print(f"  Abrufen von {query_count} Einträgen nach ID...")
    
    # Zufällige IDs auswählen
    random_ids = random.sample(entry_ids, min(query_count, len(entry_ids)))
    
    # Zeit für das Abrufen messen
    start_time = time.time()
    
    for entry_id in random_ids:
        memory.retrieve_by_id(entry_id)
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    queries_per_second = len(random_ids) / total_time
    
    results['retrieve_by_id'] = {
        'total_time': total_time,
        'queries_per_second': queries_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Abfragen/s: {queries_per_second:.2f}")
    
    # Wichtigste Einträge abrufen
    print(f"  Abrufen der wichtigsten Einträge ({query_count} Anfragen)...")
    
    # Zeit für das Abrufen der wichtigsten Einträge messen
    start_time = time.time()
    
    for _ in range(query_count):
        limit = random.randint(1, 20)
        memory.get_most_important(limit=limit)
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    queries_per_second = query_count / total_time
    
    results['get_most_important'] = {
        'total_time': total_time,
        'queries_per_second': queries_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Abfragen/s: {queries_per_second:.2f}")
    
    # Einträge auffrischen
    print(f"  Auffrischen von {query_count} Einträgen...")
    
    # Zufällige IDs auswählen
    random_ids = random.sample(entry_ids, min(query_count, len(entry_ids)))
    
    # Zeit für das Auffrischen messen
    start_time = time.time()
    
    for entry_id in random_ids:
        memory.refresh(entry_id)
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    operations_per_second = len(random_ids) / total_time
    
    results['refresh'] = {
        'total_time': total_time,
        'operations_per_second': operations_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Operationen/s: {operations_per_second:.2f}")
    
    return results

def benchmark_memory_core(num_entries: int, batch_sizes: List[int], query_count: int) -> Dict[str, Any]:
    """
    Führt Benchmarks für den Memory Core durch.
    
    Args:
        num_entries: Anzahl der zu speichernden Einträge
        batch_sizes: Liste mit Batch-Größen für das Speichern
        query_count: Anzahl der Suchanfragen
        
    Returns:
        Dictionary mit Benchmark-Ergebnissen
    """
    print(f"Benchmark: Memory Core (MLX: {use_mlx})")
    
    results = {
        'store': {},
        'retrieve_by_id': {},
        'retrieve': {},
        'export_import': {}
    }
    
    # Memory Core initialisieren
    memory = MemoryCore({
        'semantic': {'vector_dim': DEFAULT_VECTOR_DIM},
        'episodic': {'max_entries': num_entries * 2},
        'working': {'ttl_default': 3600}
    })
    
    # Einträge für verschiedene Batch-Größen speichern
    for batch_size in batch_sizes:
        if batch_size > num_entries:
            continue
        
        print(f"  Speichern von {num_entries} Einträgen in Batches von {batch_size}...")
        
        entries = []
        entry_ids = []
        
        # Einträge generieren
        for _ in range(num_entries):
            content = generate_random_text(random.randint(50, 200))
            tags = generate_random_tags(random.randint(1, 5))
            importance = random.random()
            
            entries.append({
                'content': content,
                'tags': tags,
                'importance': importance
            })
        
        # Zeit für das Speichern messen
        start_time = time.time()
        
        for i in range(0, num_entries, batch_size):
            batch = entries[i:i+batch_size]
            
            for entry in batch:
                result = memory.store(
                    content=entry['content'],
                    tags=entry['tags'],
                    importance=entry['importance'],
                    memory_type='all'
                )
                entry_ids.append(result['semantic'])  # Alle Systeme verwenden dieselbe ID
        
        end_time = time.time()
        
        # Ergebnisse speichern
        total_time = end_time - start_time
        entries_per_second = num_entries / total_time
        
        results['store'][batch_size] = {
            'total_time': total_time,
            'entries_per_second': entries_per_second
        }
        
        print(f"    Zeit: {total_time:.2f}s, Einträge/s: {entries_per_second:.2f}")
    
    # Einträge nach ID abrufen
    print(f"  Abrufen von {query_count} Einträgen nach ID...")
    
    # Zufällige IDs auswählen
    random_ids = random.sample(entry_ids, min(query_count, len(entry_ids)))
    
    # Zeit für das Abrufen messen
    start_time = time.time()
    
    for entry_id in random_ids:
        memory.retrieve_by_id(entry_id)
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    queries_per_second = len(random_ids) / total_time
    
    results['retrieve_by_id'] = {
        'total_time': total_time,
        'queries_per_second': queries_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Abfragen/s: {queries_per_second:.2f}")
    
    # Semantische Suche durchführen
    print(f"  Durchführen von {query_count} semantischen Suchanfragen...")
    
    # Zufällige Suchanfragen generieren
    queries = [generate_random_text(random.randint(10, 50)) for _ in range(query_count)]
    
    # Zeit für die semantische Suche messen
    start_time = time.time()
    
    for query in queries:
        memory.retrieve(query, limit=10)
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    queries_per_second = query_count / total_time
    
    results['retrieve'] = {
        'total_time': total_time,
        'queries_per_second': queries_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Abfragen/s: {queries_per_second:.2f}")
    
    # Export und Import testen
    print(f"  Testen von Export und Import...")
    
    # Zeit für den Export messen
    start_time = time.time()
    
    export_data = memory.export_to_json()
    
    export_time = time.time() - start_time
    
    # Neuen Memory Core erstellen
    new_memory = MemoryCore()
    
    # Zeit für den Import messen
    start_time = time.time()
    
    import_result = new_memory.import_from_json(export_data)
    
    import_time = time.time() - start_time
    
    # Ergebnisse speichern
    results['export_import'] = {
        'export_time': export_time,
        'import_time': import_time,
        'total_time': export_time + import_time,
        'exported_entries': sum(import_result.values())
    }
    
    print(f"    Export-Zeit: {export_time:.2f}s, Import-Zeit: {import_time:.2f}s")
    print(f"    Exportierte Einträge: {sum(import_result.values())}")
    
    return results

def benchmark_vxor_bridge(num_entries: int, batch_sizes: List[int], query_count: int) -> Dict[str, Any]:
    """
    Führt Benchmarks für die VXOR-Bridge durch.
    
    Args:
        num_entries: Anzahl der zu speichernden Einträge
        batch_sizes: Liste mit Batch-Größen für das Speichern
        query_count: Anzahl der Suchanfragen
        
    Returns:
        Dictionary mit Benchmark-Ergebnissen
    """
    print(f"Benchmark: VXOR-Bridge (MLX: {use_mlx})")
    
    results = {
        'store': {},
        'retrieve_by_id': {},
        'retrieve': {},
        'event_handling': {}
    }
    
    # VXOR-Bridge initialisieren
    bridge = VXORBridge()
    
    # Einträge für verschiedene Batch-Größen speichern
    for batch_size in batch_sizes:
        if batch_size > num_entries:
            continue
        
        print(f"  Speichern von {num_entries} Einträgen in Batches von {batch_size}...")
        
        entries = []
        entry_ids = []
        
        # Einträge generieren
        for _ in range(num_entries):
            content = generate_random_text(random.randint(50, 200))
            tags = generate_random_tags(random.randint(1, 5))
            importance = random.random()
            
            entries.append({
                'content': content,
                'tags': tags,
                'importance': importance
            })
        
        # Zeit für das Speichern messen
        start_time = time.time()
        
        for i in range(0, num_entries, batch_size):
            batch = entries[i:i+batch_size]
            
            for entry in batch:
                result = bridge.store(
                    content=entry['content'],
                    tags=entry['tags'],
                    importance=entry['importance'],
                    memory_type='all',
                    source_module='benchmark'
                )
                entry_ids.append(result['semantic'])  # Alle Systeme verwenden dieselbe ID
        
        end_time = time.time()
        
        # Ergebnisse speichern
        total_time = end_time - start_time
        entries_per_second = num_entries / total_time
        
        results['store'][batch_size] = {
            'total_time': total_time,
            'entries_per_second': entries_per_second
        }
        
        print(f"    Zeit: {total_time:.2f}s, Einträge/s: {entries_per_second:.2f}")
    
    # Einträge nach ID abrufen
    print(f"  Abrufen von {query_count} Einträgen nach ID...")
    
    # Zufällige IDs auswählen
    random_ids = random.sample(entry_ids, min(query_count, len(entry_ids)))
    
    # Zeit für das Abrufen messen
    start_time = time.time()
    
    for entry_id in random_ids:
        bridge.retrieve_by_id(entry_id, source_module='benchmark')
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    queries_per_second = len(random_ids) / total_time
    
    results['retrieve_by_id'] = {
        'total_time': total_time,
        'queries_per_second': queries_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Abfragen/s: {queries_per_second:.2f}")
    
    # Semantische Suche durchführen
    print(f"  Durchführen von {query_count} semantischen Suchanfragen...")
    
    # Zufällige Suchanfragen generieren
    queries = [generate_random_text(random.randint(10, 50)) for _ in range(query_count)]
    
    # Zeit für die semantische Suche messen
    start_time = time.time()
    
    for query in queries:
        bridge.retrieve(query, limit=10, source_module='benchmark')
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    queries_per_second = query_count / total_time
    
    results['retrieve'] = {
        'total_time': total_time,
        'queries_per_second': queries_per_second
    }
    
    print(f"    Zeit: {total_time:.2f}s, Abfragen/s: {queries_per_second:.2f}")
    
    # Event-Handling testen
    print(f"  Testen des Event-Handlings ({query_count} Ereignisse)...")
    
    # Event-Handler-Funktion
    event_count = 0
    
    def test_handler(event_data):
        nonlocal event_count
        event_count += 1
    
    # Event-Handler registrieren
    bridge.register_event_handler('test_event', test_handler)
    
    # Zeit für das Auslösen von Ereignissen messen
    start_time = time.time()
    
    for i in range(query_count):
        bridge.trigger_event('test_event', {'index': i})
    
    end_time = time.time()
    
    # Ergebnisse speichern
    total_time = end_time - start_time
    events_per_second = query_count / total_time
    
    results['event_handling'] = {
        'total_time': total_time,
        'events_per_second': events_per_second,
        'event_count': event_count
    }
    
    print(f"    Zeit: {total_time:.2f}s, Ereignisse/s: {events_per_second:.2f}")
    print(f"    Verarbeitete Ereignisse: {event_count}")
    
    return results

def benchmark_parallel_processing(num_entries: int, num_processes: int) -> Dict[str, Any]:
    """
    Führt Benchmarks für die Parallelverarbeitung durch.
    
    Args:
        num_entries: Anzahl der zu speichernden Einträge
        num_processes: Anzahl der parallelen Prozesse
        
    Returns:
        Dictionary mit Benchmark-Ergebnissen
    """
    print(f"Benchmark: Parallelverarbeitung mit {num_processes} Prozessen (MLX: {use_mlx})")
    
    results = {
        'sequential': {},
        'parallel': {}
    }
    
    # Einträge generieren
    entries = []
    for _ in range(num_entries):
        content = generate_random_text(random.randint(50, 200))
        tags = generate_random_tags(random.randint(1, 5))
        importance = random.random()
        
        entries.append({
            'content': content,
            'tags': tags,
            'importance': importance
        })
    
    # Sequentielle Verarbeitung
    print(f"  Sequentielle Verarbeitung von {num_entries} Einträgen...")
    
    # Memory Core initialisieren
    memory = MemoryCore()
    
    # Zeit für die sequentielle Verarbeitung messen
    start_time = time.time()
    
    for entry in entries:
        memory.store(
            content=entry['content'],
            tags=entry['tags'],
            importance=entry['importance'],
            memory_type='all'
        )
    
    end_time = time.time()
    
    # Ergebnisse speichern
    sequential_time = end_time - start_time
    sequential_entries_per_second = num_entries / sequential_time
    
    results['sequential'] = {
        'total_time': sequential_time,
        'entries_per_second': sequential_entries_per_second
    }
    
    print(f"    Zeit: {sequential_time:.2f}s, Einträge/s: {sequential_entries_per_second:.2f}")
    
    # Parallele Verarbeitung
    print(f"  Parallele Verarbeitung von {num_entries} Einträgen mit {num_processes} Prozessen...")
    
    # Funktion für die parallele Verarbeitung
    def process_entries(entries_chunk):
        # Memory Core initialisieren
        memory = MemoryCore()
        
        # Einträge speichern
        for entry in entries_chunk:
            memory.store(
                content=entry['content'],
                tags=entry['tags'],
                importance=entry['importance'],
                memory_type='all'
            )
    
    # Einträge in Chunks aufteilen
    chunk_size = num_entries // num_processes
    chunks = [entries[i:i+chunk_size] for i in range(0, num_entries, chunk_size)]
    
    # Zeit für die parallele Verarbeitung messen
    start_time = time.time()
    
    # Pool erstellen und Aufgaben zuweisen
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_entries, chunks)
    
    end_time = time.time()
    
    # Ergebnisse speichern
    parallel_time = end_time - start_time
    parallel_entries_per_second = num_entries / parallel_time
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    
    results['parallel'] = {
        'total_time': parallel_time,
        'entries_per_second': parallel_entries_per_second,
        'speedup': speedup,
        'efficiency': speedup / num_processes if num_processes > 0 else 0
    }
    
    print(f"    Zeit: {parallel_time:.2f}s, Einträge/s: {parallel_entries_per_second:.2f}")
    print(f"    Speedup: {speedup:.2f}x, Effizienz: {results['parallel']['efficiency']:.2f}")
    
    return results

def generate_charts(results: Dict[str, Any], output_dir: str) -> None:
    """
    Generiert Diagramme aus den Benchmark-Ergebnissen.
    
    Args:
        results: Dictionary mit Benchmark-Ergebnissen
        output_dir: Verzeichnis für die Ausgabedateien
    """
    print(f"Generiere Diagramme in {output_dir}...")
    
    # Verzeichnis erstellen, falls nicht vorhanden
    os.makedirs(output_dir, exist_ok=True)
    
    # Diagramm für Speicheroperationen
    plt.figure(figsize=(12, 8))
    
    for component, data in results.items():
        if 'store' in data:
            batch_sizes = sorted(data['store'].keys())
            entries_per_second = [data['store'][batch_size]['entries_per_second'] for batch_size in batch_sizes]
            
            plt.plot(batch_sizes, entries_per_second, marker='o', label=component)
    
    plt.title('Speicheroperationen pro Sekunde nach Batch-Größe')
    plt.xlabel('Batch-Größe')
    plt.ylabel('Einträge pro Sekunde')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'store_operations.png'))
    
    # Diagramm für Abrufoperationen
    plt.figure(figsize=(12, 8))
    
    components = []
    retrieve_by_id_values = []
    retrieve_query_values = []
    
    for component, data in results.items():
        if 'retrieve_by_id' in data:
            components.append(component)
            retrieve_by_id_values.append(data['retrieve_by_id']['queries_per_second'])
            
            # Verschiedene Abfragemethoden
            if 'retrieve' in data:
                retrieve_query_values.append(data['retrieve']['queries_per_second'])
            elif 'retrieve_by_query' in data:
                retrieve_query_values.append(data['retrieve_by_query']['queries_per_second'])
            else:
                retrieve_query_values.append(0)
    
    x = np.arange(len(components))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, retrieve_by_id_values, width, label='Abruf nach ID')
    rects2 = ax.bar(x + width/2, retrieve_query_values, width, label='Semantische Suche')
    
    ax.set_title('Abrufoperationen pro Sekunde nach Komponente')
    ax.set_xlabel('Komponente')
    ax.set_ylabel('Abfragen pro Sekunde')
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retrieve_operations.png'))
    
    # Diagramm für Parallelverarbeitung
    if 'parallel_processing' in results:
        data = results['parallel_processing']
        
        plt.figure(figsize=(12, 8))
        
        labels = ['Sequentiell', 'Parallel']
        values = [data['sequential']['entries_per_second'], data['parallel']['entries_per_second']]
        
        plt.bar(labels, values)
        plt.title('Sequentielle vs. Parallele Verarbeitung')
        plt.xlabel('Verarbeitungsmethode')
        plt.ylabel('Einträge pro Sekunde')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'parallel_processing.png'))
    
    print(f"Diagramme wurden in {output_dir} gespeichert.")

def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Speichert die Benchmark-Ergebnisse als JSON-Datei.
    
    Args:
        results: Dictionary mit Benchmark-Ergebnissen
        output_dir: Verzeichnis für die Ausgabedateien
    """
    print(f"Speichere Ergebnisse in {output_dir}...")
    
    # Verzeichnis erstellen, falls nicht vorhanden
    os.makedirs(output_dir, exist_ok=True)
    
    # Ergebnisse als JSON speichern
    output_file = os.path.join(output_dir, 'benchmark_results.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Ergebnisse wurden in {output_file} gespeichert.")

def generate_report(results: Dict[str, Any], output_dir: str) -> None:
    """
    Generiert einen Bericht aus den Benchmark-Ergebnissen.
    
    Args:
        results: Dictionary mit Benchmark-Ergebnissen
        output_dir: Verzeichnis für die Ausgabedateien
    """
    print(f"Generiere Bericht in {output_dir}...")
    
    # Verzeichnis erstellen, falls nicht vorhanden
    os.makedirs(output_dir, exist_ok=True)
    
    # Bericht als Markdown-Datei erstellen
    output_file = os.path.join(output_dir, 'benchmark_report.md')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# VX-MEMEX Performance-Benchmark-Bericht\n\n")
        f.write(f"Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"MLX-Unterstützung: {'Ja' if use_mlx else 'Nein'}\n\n")
        
        f.write("## Zusammenfassung\n\n")
        
        # Tabelle mit Zusammenfassung erstellen
        f.write("| Komponente | Speicheroperationen/s | Abrufoperationen/s | Semantische Suche/s |\n")
        f.write("|------------|----------------------|---------------------|---------------------|\n")
        
        for component, data in results.items():
            if component != 'parallel_processing':
                # Durchschnittliche Speicheroperationen pro Sekunde
                store_ops = 0
                if 'store' in data:
                    store_values = [data['store'][batch_size]['entries_per_second'] for batch_size in data['store']]
                    store_ops = sum(store_values) / len(store_values) if store_values else 0
                
                # Abrufoperationen pro Sekunde
                retrieve_ops = 0
                if 'retrieve_by_id' in data:
                    retrieve_ops = data['retrieve_by_id']['queries_per_second']
                
                # Semantische Suche pro Sekunde
                semantic_ops = 0
                if 'retrieve' in data:
                    semantic_ops = data['retrieve']['queries_per_second']
                elif 'retrieve_by_query' in data:
                    semantic_ops = data['retrieve_by_query']['queries_per_second']
                
                f.write(f"| {component} | {store_ops:.2f} | {retrieve_ops:.2f} | {semantic_ops:.2f} |\n")
        
        f.write("\n## Detaillierte Ergebnisse\n\n")
        
        # Detaillierte Ergebnisse für jede Komponente
        for component, data in results.items():
            f.write(f"### {component}\n\n")
            
            if component == 'parallel_processing':
                # Parallelverarbeitung
                sequential = data['sequential']
                parallel = data['parallel']
                
                f.write("#### Sequentielle vs. Parallele Verarbeitung\n\n")
                f.write("| Methode | Zeit (s) | Einträge/s | Speedup | Effizienz |\n")
                f.write("|---------|----------|------------|---------|----------|\n")
                f.write(f"| Sequentiell | {sequential['total_time']:.2f} | {sequential['entries_per_second']:.2f} | - | - |\n")
                f.write(f"| Parallel | {parallel['total_time']:.2f} | {parallel['entries_per_second']:.2f} | {parallel['speedup']:.2f}x | {parallel['efficiency']:.2f} |\n\n")
            else:
                # Speicheroperationen
                if 'store' in data:
                    f.write("#### Speicheroperationen\n\n")
                    f.write("| Batch-Größe | Zeit (s) | Einträge/s |\n")
                    f.write("|-------------|----------|------------|\n")
                    
                    for batch_size in sorted(data['store'].keys()):
                        batch_data = data['store'][batch_size]
                        f.write(f"| {batch_size} | {batch_data['total_time']:.2f} | {batch_data['entries_per_second']:.2f} |\n")
                    
                    f.write("\n")
                
                # Abrufoperationen
                if 'retrieve_by_id' in data:
                    f.write("#### Abruf nach ID\n\n")
                    f.write(f"Zeit: {data['retrieve_by_id']['total_time']:.2f}s, Abfragen/s: {data['retrieve_by_id']['queries_per_second']:.2f}\n\n")
                
                # Semantische Suche
                if 'retrieve' in data:
                    f.write("#### Semantische Suche\n\n")
                    f.write(f"Zeit: {data['retrieve']['total_time']:.2f}s, Abfragen/s: {data['retrieve']['queries_per_second']:.2f}\n\n")
                elif 'retrieve_by_query' in data:
                    f.write("#### Semantische Suche\n\n")
                    f.write(f"Zeit: {data['retrieve_by_query']['total_time']:.2f}s, Abfragen/s: {data['retrieve_by_query']['queries_per_second']:.2f}\n\n")
                
                # Spezifische Operationen für verschiedene Komponenten
                if component == 'episodic_store' and 'retrieve_by_time_range' in data:
                    f.write("#### Abruf nach Zeitbereich\n\n")
                    f.write(f"Zeit: {data['retrieve_by_time_range']['total_time']:.2f}s, Abfragen/s: {data['retrieve_by_time_range']['queries_per_second']:.2f}\n\n")
                
                if component == 'working_memory' and 'get_most_important' in data:
                    f.write("#### Abruf der wichtigsten Einträge\n\n")
                    f.write(f"Zeit: {data['get_most_important']['total_time']:.2f}s, Abfragen/s: {data['get_most_important']['queries_per_second']:.2f}\n\n")
                
                if component == 'memory_core' and 'export_import' in data:
                    f.write("#### Export und Import\n\n")
                    export_import = data['export_import']
                    f.write(f"Export-Zeit: {export_import['export_time']:.2f}s\n")
                    f.write(f"Import-Zeit: {export_import['import_time']:.2f}s\n")
                    f.write(f"Gesamtzeit: {export_import['total_time']:.2f}s\n")
                    f.write(f"Exportierte Einträge: {export_import['exported_entries']}\n\n")
                
                if component == 'vxor_bridge' and 'event_handling' in data:
                    f.write("#### Event-Handling\n\n")
                    event_handling = data['event_handling']
                    f.write(f"Zeit: {event_handling['total_time']:.2f}s, Ereignisse/s: {event_handling['events_per_second']:.2f}\n")
                    f.write(f"Verarbeitete Ereignisse: {event_handling['event_count']}\n\n")
        
        f.write("\n## Diagramme\n\n")
        f.write("![Speicheroperationen](store_operations.png)\n\n")
        f.write("![Abrufoperationen](retrieve_operations.png)\n\n")
        
        if 'parallel_processing' in results:
            f.write("![Parallelverarbeitung](parallel_processing.png)\n\n")
        
        f.write("\n## Systemumgebung\n\n")
        f.write("- Optimiert für: Apple Silicon M4 Max\n")
        f.write(f"- MLX-Unterstützung: {'Ja' if use_mlx else 'Nein'}\n")
        f.write(f"- Python-Version: {sys.version.split()[0]}\n")
        f.write(f"- NumPy-Version: {np.__version__}\n")
        f.write(f"- Betriebssystem: {os.uname().sysname} {os.uname().release}\n")
        f.write(f"- CPU-Kerne: {multiprocessing.cpu_count()}\n")
    
    print(f"Bericht wurde in {output_file} gespeichert.")

def check_mlx_support() -> bool:
    """
    Prüft, ob MLX unterstützt wird.
    
    Returns:
        True, wenn MLX unterstützt wird, sonst False
    """
    try:
        import mlx.core
        return True
    except ImportError:
        return False

def main():
    """Hauptfunktion für die Benchmarks."""
    # Kommandozeilenargumente parsen
    parser = argparse.ArgumentParser(description='VX-MEMEX Performance-Benchmarks')
    parser.add_argument('--num-entries', type=int, default=DEFAULT_NUM_ENTRIES, help='Anzahl der zu speichernden Einträge')
    parser.add_argument('--query-count', type=int, default=DEFAULT_QUERY_COUNT, help='Anzahl der Suchanfragen')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=DEFAULT_BATCH_SIZES, help='Batch-Größen für das Speichern')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Verzeichnis für die Ausgabedateien')
    parser.add_argument('--components', type=str, nargs='+', default=['semantic_store', 'episodic_store', 'working_memory', 'memory_core', 'vxor_bridge', 'parallel_processing'], help='Zu testende Komponenten')
    parser.add_argument('--processes', type=int, default=multiprocessing.cpu_count(), help='Anzahl der parallelen Prozesse')
    
    args = parser.parse_args()
    
    # MLX-Unterstützung prüfen
    global use_mlx
    use_mlx = check_mlx_support()
    print(f"MLX-Unterstützung: {'Ja' if use_mlx else 'Nein'}")
    
    # Benchmarks durchführen
    global results
    results = {}
    
    if 'semantic_store' in args.components:
        results['semantic_store'] = benchmark_semantic_store(args.num_entries, args.batch_sizes, args.query_count)
    
    if 'episodic_store' in args.components:
        results['episodic_store'] = benchmark_episodic_store(args.num_entries, args.batch_sizes, args.query_count)
    
    if 'working_memory' in args.components:
        results['working_memory'] = benchmark_working_memory(args.num_entries, args.batch_sizes, args.query_count)
    
    if 'memory_core' in args.components:
        results['memory_core'] = benchmark_memory_core(args.num_entries, args.batch_sizes, args.query_count)
    
    if 'vxor_bridge' in args.components:
        results['vxor_bridge'] = benchmark_vxor_bridge(args.num_entries, args.batch_sizes, args.query_count)
    
    if 'parallel_processing' in args.components:
        results['parallel_processing'] = benchmark_parallel_processing(args.num_entries, args.processes)
    
    # Ergebnisse speichern
    save_results(results, args.output_dir)
    
    # Diagramme generieren
    generate_charts(results, args.output_dir)
    
    # Bericht generieren
    generate_report(results, args.output_dir)
    
    print("Benchmarks abgeschlossen.")

if __name__ == '__main__':
    main()
