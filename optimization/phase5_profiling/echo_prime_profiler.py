#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - ECHO-PRIME Profilierungsskript
Datum: 2025-05-03
"""

import os
import sys
import time
import logging
import argparse
import cProfile
import pstats
import io
import json
import gc
import datetime
import numpy as np
import tracemalloc
from memory_profiler import profile as memory_profile

# Füge das Hauptverzeichnis zum Pfad hinzu, um MISO-Module zu importieren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"echo_prime_profiling_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ECHO-PRIME-Profiler")

def import_echo_prime():
    """Importiert die ECHO-PRIME-Module und gibt die wichtigsten Komponenten zurück"""
    try:
        from engines.echo_prime.engine import EchoPrimeEngine
        from engines.echo_prime.timeline import Timeline, TimeNode
        from engines.echo_prime.paradox import ParadoxDetector, ParadoxResolver
        from miso.integration.temporal_belief_network import TemporalBeliefNetwork
        
        return {
            "EchoPrimeEngine": EchoPrimeEngine,
            "Timeline": Timeline,
            "TimeNode": TimeNode,
            "ParadoxDetector": ParadoxDetector,
            "ParadoxResolver": ParadoxResolver,
            "TemporalBeliefNetwork": TemporalBeliefNetwork
        }
    except ImportError as e:
        logger.error(f"Fehler beim Importieren der ECHO-PRIME-Module: {e}")
        raise

def generate_test_data(size="small"):
    """Generiert Testdaten für verschiedene Größen"""
    logger.info(f"Generiere {size} Testdaten für ECHO-PRIME")
    
    now = datetime.datetime.now()
    node_counts = {
        "small": 100,
        "medium": 1000,
        "large": 10000,
        "xlarge": 50000
    }
    
    count = node_counts.get(size, 100)
    
    # Erstelle Zeitstempel im Bereich von ±1 Jahr
    timestamps = [
        now + datetime.timedelta(
            days=np.random.randint(-365, 365),
            hours=np.random.randint(-12, 12),
            minutes=np.random.randint(-30, 30)
        )
        for _ in range(count)
    ]
    
    # Sortiere Zeitstempel chronologisch
    timestamps.sort()
    
    return {
        "timestamps": timestamps,
        "count": count,
        "size_label": size
    }

@memory_profile
def profile_timeline_creation(components, test_data):
    """Profiliert die Erstellung einer Timeline mit vielen Knoten"""
    logger.info(f"Profiliere Timeline-Erstellung mit {test_data['count']} Knoten")
    
    Timeline = components["Timeline"]
    TimeNode = components["TimeNode"]
    
    start_time = time.time()
    
    # Erstelle Timeline
    timeline = Timeline(f"TestTimeline_{test_data['size_label']}")
    
    # Erstelle und füge Knoten hinzu
    for i, timestamp in enumerate(test_data["timestamps"]):
        node_id = f"node_{i:05d}"
        node = TimeNode(timestamp=timestamp, node_id=node_id)
        
        # Hier verwenden wir die add_event-Methode von Timeline, da wir in der vorherigen Analyse
        # festgestellt haben, dass Timeline keine direkte add_node-Methode hat
        timeline.add_event(
            {
                "id": f"event_{i:05d}",
                "name": f"Event {i}",
                "description": f"Test event {i} for profiling",
                "timestamp": timestamp,
                "data": {
                    "type": "test",
                    "index": i,
                    "value": np.random.random()
                }
            }
        )
        
        # Füge einige Verbindungen zwischen Knoten hinzu
        if i > 0 and i % 10 == 0:
            # Verbinde jeden 10. Knoten mit dem vorherigen
            timeline.add_trigger(
                {
                    "source_event_id": f"event_{i-1:05d}",
                    "target_event_id": f"event_{i:05d}",
                    "trigger_type": "causal",
                    "probability": 0.8
                }
            )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    logger.info(f"Timeline-Erstellung: {elapsed:.4f} Sekunden")
    logger.info(f"Knoten pro Sekunde: {test_data['count']/elapsed:.2f}")
    
    return {
        "timeline": timeline,
        "creation_time": elapsed,
        "nodes_per_second": test_data['count']/elapsed
    }

def profile_paradox_detection(components, timeline, iterations=100):
    """Profiliert die Paradoxerkennung in einer Timeline"""
    logger.info(f"Profiliere Paradoxerkennung für {iterations} Iterationen")
    
    ParadoxDetector = components["ParadoxDetector"]
    detector = ParadoxDetector()
    
    start_time = time.time()
    
    paradoxes = []
    for _ in range(iterations):
        result = detector.detect_paradoxes(timeline)
        paradoxes.append(result)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    logger.info(f"Paradoxerkennung: {elapsed:.4f} Sekunden für {iterations} Iterationen")
    logger.info(f"Durchschnittliche Zeit pro Erkennung: {elapsed/iterations:.6f} Sekunden")
    
    return {
        "detection_time": elapsed,
        "average_time": elapsed/iterations,
        "paradox_count": len(paradoxes)
    }

def profile_temporal_belief_network(components, timeline):
    """Profiliert die Erstellung und Verarbeitung eines temporalen Glaubensnetzwerks"""
    logger.info("Profiliere TemporalBeliefNetwork-Erstellung")
    
    TemporalBeliefNetwork = components["TemporalBeliefNetwork"]
    
    start_time = time.time()
    
    # Erstelle und initialisiere das Netzwerk
    belief_network = TemporalBeliefNetwork()
    
    # Füge Knoten aus der Timeline hinzu
    for node_id, node in timeline.nodes.items():
        belief_network.add_time_node(node, {}, 0.0)
    
    # Erstelle Verbindungen
    for i, node_id in enumerate(timeline.nodes.keys()):
        if i > 0 and i % 10 == 0:
            prev_node_id = list(timeline.nodes.keys())[i-1]
            belief_network.connect_nodes(prev_node_id, node_id, 0.8)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    logger.info(f"TemporalBeliefNetwork-Erstellung: {elapsed:.4f} Sekunden")
    logger.info(f"Knoten pro Sekunde: {len(timeline.nodes)/elapsed:.2f}")
    
    # Profiliere Inferenz
    start_time = time.time()
    belief_network.update_beliefs()
    end_time = time.time()
    inference_time = end_time - start_time
    
    logger.info(f"Belief-Propagation: {inference_time:.4f} Sekunden")
    
    return {
        "network_creation_time": elapsed,
        "nodes_per_second": len(timeline.nodes)/elapsed,
        "inference_time": inference_time
    }

def profile_echo_prime_engine(components, timeline):
    """Profiliert die gesamte ECHO-PRIME-Engine"""
    logger.info("Profiliere EchoPrimeEngine Operationen")
    
    EchoPrimeEngine = components["EchoPrimeEngine"]
    engine = EchoPrimeEngine()
    
    # Profiliere Zeitlinien-Import
    start_time = time.time()
    engine.import_timeline(timeline)
    end_time = time.time()
    import_time = end_time - start_time
    
    logger.info(f"Timeline-Import: {import_time:.4f} Sekunden")
    
    # Profiliere Zeitlinien-Analyse
    start_time = time.time()
    analysis = engine.analyze_timeline(timeline.id)
    end_time = time.time()
    analysis_time = end_time - start_time
    
    logger.info(f"Timeline-Analyse: {analysis_time:.4f} Sekunden")
    
    # Profiliere Zeitlinien-Vorhersage
    start_time = time.time()
    prediction = engine.predict_timeline_extension(timeline.id, steps=10)
    end_time = time.time()
    prediction_time = end_time - start_time
    
    logger.info(f"Timeline-Vorhersage: {prediction_time:.4f} Sekunden")
    
    return {
        "import_time": import_time,
        "analysis_time": analysis_time,
        "prediction_time": prediction_time,
        "total_engine_time": import_time + analysis_time + prediction_time
    }

def run_full_profiling(size="small", output_file=None):
    """Führt eine vollständige Profilierung aller ECHO-PRIME-Komponenten durch"""
    logger.info(f"Starte vollständige ECHO-PRIME-Profilierung mit Größe: {size}")
    
    tracemalloc.start()
    
    # Importiere Komponenten
    components = import_echo_prime()
    
    # Generiere Testdaten
    test_data = generate_test_data(size)
    
    # Profiliere Timeline-Erstellung
    pr = cProfile.Profile()
    pr.enable()
    
    timeline_result = profile_timeline_creation(components, test_data)
    timeline = timeline_result["timeline"]
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Drucke die Top 20 Funktionen
    logger.info(f"cProfile für Timeline-Erstellung:\n{s.getvalue()}")
    
    # Profiliere Paradoxerkennung
    pr = cProfile.Profile()
    pr.enable()
    
    paradox_result = profile_paradox_detection(components, timeline)
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    logger.info(f"cProfile für Paradoxerkennung:\n{s.getvalue()}")
    
    # Profiliere TemporalBeliefNetwork
    pr = cProfile.Profile()
    pr.enable()
    
    network_result = profile_temporal_belief_network(components, timeline)
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    logger.info(f"cProfile für TemporalBeliefNetwork:\n{s.getvalue()}")
    
    # Profiliere EchoPrimeEngine
    pr = cProfile.Profile()
    pr.enable()
    
    engine_result = profile_echo_prime_engine(components, timeline)
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    logger.info(f"cProfile für EchoPrimeEngine:\n{s.getvalue()}")
    
    # Speicherüberwachung
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    logger.info(f"Aktueller Speicherverbrauch: {current / 1024 / 1024:.2f} MB")
    logger.info(f"Spitze des Speicherverbrauchs: {peak / 1024 / 1024:.2f} MB")
    
    # Erstelle Ergebnisübersicht
    results = {
        "test_size": size,
        "node_count": test_data["count"],
        "timeline_creation": timeline_result,
        "paradox_detection": paradox_result,
        "temporal_belief_network": network_result,
        "echo_prime_engine": engine_result,
        "memory_usage": {
            "current_mb": current / 1024 / 1024,
            "peak_mb": peak / 1024 / 1024
        },
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Speichere Ergebnisse in Datei, wenn angegeben
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Ergebnisse gespeichert in: {output_file}")
    
    return results

def main():
    """Hauptfunktion zum Starten der Profilierung"""
    parser = argparse.ArgumentParser(description="ECHO-PRIME Profilierungsskript")
    parser.add_argument("--size", choices=["small", "medium", "large", "xlarge"], default="small",
                        help="Größe des Testdatensatzes")
    parser.add_argument("--output", default=None, help="Ausgabedatei für Ergebnisse (JSON)")
    args = parser.parse_args()
    
    output_file = args.output
    if not output_file:
        output_file = f"echo_prime_profiling_{args.size}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    logger.info(f"Starte ECHO-PRIME Profilierung mit Größe {args.size}")
    try:
        results = run_full_profiling(args.size, output_file)
        
        # Zeige Zusammenfassung
        print("\n=== ECHO-PRIME Profilierung Zusammenfassung ===")
        print(f"Testgröße: {args.size} ({results['node_count']} Knoten)")
        print(f"Timeline-Erstellung: {results['timeline_creation']['creation_time']:.4f} s ({results['timeline_creation']['nodes_per_second']:.2f} Knoten/s)")
        print(f"Paradoxerkennung: {results['paradox_detection']['average_time']*1000:.2f} ms pro Durchlauf")
        print(f"Belief-Netzwerk: {results['temporal_belief_network']['network_creation_time']:.4f} s, Inferenz: {results['temporal_belief_network']['inference_time']*1000:.2f} ms")
        print(f"Engine-Gesamtzeit: {results['echo_prime_engine']['total_engine_time']:.4f} s")
        print(f"Spitze des Speicherverbrauchs: {results['memory_usage']['peak_mb']:.2f} MB\n")
        
        logger.info("Profilierung erfolgreich abgeschlossen")
    except Exception as e:
        logger.error(f"Fehler bei der Profilierung: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
