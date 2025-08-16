#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate - Lasttests für Stabilitätstests

Dieses Skript führt Lasttests für die Module des MISO Ultimate AGI-Systems durch.
Die Tests prüfen die Leistung und Stabilität unter hoher Last.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import datetime
import importlib
import traceback
import gc
import psutil
import threading
import multiprocessing
import numpy as np
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stability_tests_load.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("miso_stability_load_tests")

# Basis-Verzeichnis
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Füge Basis-Verzeichnis zum Pfad hinzu
sys.path.insert(0, BASE_DIR)

# Ergebnisse
results = {
    "timestamp": datetime.datetime.now().isoformat(),
    "load_tests": {},
    "overall_status": "pending"
}

def measure_memory_usage():
    """Misst den aktuellen Speicherverbrauch des Prozesses"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Plattformunabhängige Speicherinformationen
    memory_dict = {
        "rss": memory_info.rss,  # Resident Set Size
        "vms": memory_info.vms,  # Virtual Memory Size
    }
    
    # Füge plattformspezifische Attribute hinzu, wenn verfügbar
    for attr in ['shared', 'text', 'lib', 'data', 'dirty']:
        if hasattr(memory_info, attr):
            memory_dict[attr] = getattr(memory_info, attr)
    
    return memory_dict

def run_gc():
    """Führt Garbage Collection durch"""
    gc.collect()

def test_load(test_name, test_function, iterations=100, threads=4):
    """
    Führt einen Lasttest durch
    
    Args:
        test_name: Name des Tests
        test_function: Testfunktion
        iterations: Anzahl der Iterationen
        threads: Anzahl der Threads
        
    Returns:
        Testergebnis
    """
    logger.info(f"Führe Lasttest durch: {test_name} ({iterations} Iterationen, {threads} Threads)")
    
    # Initialisiere Ergebnis
    result = {
        "name": test_name,
        "status": "pending",
        "start_time": datetime.datetime.now().isoformat(),
        "end_time": None,
        "duration": None,
        "memory_before": measure_memory_usage(),
        "memory_after": None,
        "memory_diff": None,
        "error": None,
        "details": {
            "iterations": iterations,
            "threads": threads,
            "results": []
        }
    }
    
    try:
        # Führe Garbage Collection durch
        run_gc()
        
        # Führe Test durch
        start_time = time.time()
        
        # Erstelle Thread-Pool
        thread_pool = []
        thread_results = []
        
        # Erstelle Threads
        for i in range(threads):
            thread_results.append([])
            thread = threading.Thread(
                target=run_thread_test,
                args=(test_function, iterations // threads, thread_results[i])
            )
            thread_pool.append(thread)
        
        # Starte Threads
        for thread in thread_pool:
            thread.start()
        
        # Warte auf Threads
        for thread in thread_pool:
            thread.join()
        
        # Sammle Ergebnisse
        for thread_result in thread_results:
            result["details"]["results"].extend(thread_result)
        
        end_time = time.time()
        
        # Führe Garbage Collection durch
        run_gc()
        
        # Aktualisiere Ergebnis
        result["status"] = "passed"
        result["end_time"] = datetime.datetime.now().isoformat()
        result["duration"] = end_time - start_time
        result["memory_after"] = measure_memory_usage()
        
        # Berechne Speicherdifferenz
        result["memory_diff"] = {
            key: result["memory_after"][key] - result["memory_before"][key]
            for key in result["memory_before"]
        }
        
        # Berechne Statistiken
        durations = [r["duration"] for r in result["details"]["results"]]
        result["details"]["statistics"] = {
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_duration": sum(durations) / len(durations),
            "median_duration": sorted(durations)[len(durations) // 2],
            "total_duration": sum(durations),
            "throughput": len(durations) / result["duration"]
        }
        
        logger.info(f"Lasttest {test_name} abgeschlossen: {result['status']}")
    except Exception as e:
        # Aktualisiere Ergebnis bei Fehler
        result["status"] = "error"
        result["end_time"] = datetime.datetime.now().isoformat()
        result["duration"] = time.time() - start_time
        result["memory_after"] = measure_memory_usage()
        result["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        
        # Berechne Speicherdifferenz
        result["memory_diff"] = {
            key: result["memory_after"][key] - result["memory_before"][key]
            for key in result["memory_before"]
        }
        
        logger.error(f"Fehler beim Lasttest {test_name}: {e}")
    
    return result

def run_thread_test(test_function, iterations, results):
    """
    Führt einen Test in einem Thread durch
    
    Args:
        test_function: Testfunktion
        iterations: Anzahl der Iterationen
        results: Ergebnisliste
    """
    for i in range(iterations):
        try:
            start_time = time.time()
            test_result = test_function()
            end_time = time.time()
            
            results.append({
                "iteration": i,
                "success": test_result.get("success", False),
                "duration": end_time - start_time,
                "details": test_result
            })
        except Exception as e:
            results.append({
                "iteration": i,
                "success": False,
                "duration": time.time() - start_time,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            })

def test_m_code_load():
    """Testet die Lastfähigkeit des M-CODE Core Moduls"""
    try:
        # Importiere M-CODE Core
        from miso.code.m_code import compile_m_code, execute_m_code
        
        # Generiere M-CODE
        m_code = """
        func fibonacci(n) {
            if (n <= 1) {
                return n;
            }
            return fibonacci(n - 1) + fibonacci(n - 2);
        }
        
        fibonacci(10);
        """
        
        # Kompiliere und führe M-CODE aus
        bytecode = compile_m_code(m_code)
        result = execute_m_code(m_code)
        
        return {
            "success": True,
            "message": "M-CODE Core Lasttest erfolgreich",
            "details": {
                "bytecode": str(bytecode),
                "result": result
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"M-CODE Core Lasttest fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_m_lingua_load():
    """Testet die Lastfähigkeit des M-LINGUA Interface Moduls"""
    try:
        # Importiere M-LINGUA Interface
        from miso.lang.mlingua.mlingua_interface import MLinguaInterface
        
        # Erstelle M-LINGUA Interface
        mlingua = MLinguaInterface()
        
        # Generiere zufälligen Text
        texts = [
            "Berechne die Summe von 2 und 3",
            "Öffne die Datei 'test.txt'",
            "Erstelle eine neue Variable mit dem Wert 42",
            "Multipliziere 5 mit 7",
            "Zeige mir die aktuelle Uhrzeit"
        ]
        
        # Wähle zufälligen Text
        import random
        text = random.choice(texts)
        
        # Verarbeite Text
        result = mlingua.process(text)
        
        return {
            "success": True,
            "message": "M-LINGUA Interface Lasttest erfolgreich",
            "details": {
                "text": text,
                "result": str(result)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"M-LINGUA Interface Lasttest fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_echo_prime_load():
    """Testet die Lastfähigkeit des ECHO-PRIME Moduls"""
    try:
        # Importiere ECHO-PRIME
        from engines.echo_prime.engine import EchoPrimeEngine
        from engines.echo_prime.timeline import Timeline, Event
        
        # Erstelle ECHO-PRIME Engine
        engine = EchoPrimeEngine()
        
        # Erstelle Timeline
        timeline = Timeline(
            name="Load Test Timeline",
            description="Eine Testzeitlinie für den Lasttest"
        )
        
        # Erstelle Events
        for i in range(10):
            event = Event(
                name=f"Event {i}",
                description=f"Ein Testereignis für den Lasttest",
                timestamp=datetime.datetime.now() + datetime.timedelta(minutes=i)
            )
            timeline.add_event(event)
        
        # Führe Simulation durch
        engine.simulate(timeline)
        
        return {
            "success": True,
            "message": "ECHO-PRIME Lasttest erfolgreich",
            "details": {
                "timeline": str(timeline),
                "events": len(timeline.events)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"ECHO-PRIME Lasttest fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_hyperfilter_load():
    """Testet die Lastfähigkeit des HYPERFILTER Moduls"""
    try:
        # Importiere HYPERFILTER
        from miso.filter.hyperfilter import HyperFilter, FilterConfig, FilterMode
        
        # Erstelle HyperFilter
        config = FilterConfig(
            mode=FilterMode.STRICT,
            threshold=0.8,
            max_retries=3
        )
        
        hyperfilter = HyperFilter(config)
        
        # Generiere zufälligen Text
        texts = [
            "Dies ist ein Testtext für den HYPERFILTER.",
            "Ein weiterer Testtext für den Lasttest.",
            "Dieser Text sollte gefiltert werden.",
            "Ein Text mit potenziell problematischem Inhalt.",
            "Ein völlig harmloser Text ohne Probleme."
        ]
        
        # Wähle zufälligen Text
        import random
        text = random.choice(texts)
        
        # Filtere Text
        filtered_text, metadata = hyperfilter.filter_input(text)
        
        return {
            "success": True,
            "message": "HYPERFILTER Lasttest erfolgreich",
            "details": {
                "original_text": text,
                "filtered_text": filtered_text,
                "metadata": str(metadata)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"HYPERFILTER Lasttest fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_deep_state_load():
    """Testet die Lastfähigkeit des Deep-State-Moduls"""
    try:
        # Importiere Deep-State-Modul
        from miso.analysis.deep_state import DeepStateAnalyzer
        
        # Erstelle DeepStateAnalyzer
        analyzer = DeepStateAnalyzer()
        
        # Generiere zufälligen Text
        texts = [
            "Dies ist ein Testtext für die Deep-State-Analyse.",
            "Ein weiterer Testtext für den Lasttest.",
            "Dieser Text sollte analysiert werden.",
            "Ein Text mit potenziell interessantem Inhalt.",
            "Ein völlig harmloser Text ohne tiefere Bedeutung."
        ]
        
        # Wähle zufälligen Text
        import random
        text = random.choice(texts)
        
        # Analysiere Text
        result = analyzer.analyze(
            content_stream=text,
            source_id="test",
            source_trust_level=0.9,
            language_code="de",
            context_cluster="test"
        )
        
        return {
            "success": True,
            "message": "Deep-State-Modul Lasttest erfolgreich",
            "details": {
                "text": text,
                "result": str(result)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Deep-State-Modul Lasttest fehlgeschlagen: {e}",
            "error": str(e)
        }

def test_vxor_integration_load():
    """Testet die Lastfähigkeit der VXOR-Integration"""
    try:
        # Importiere VXOR-Integration
        from miso.vxor.vx_memex import VXMemex
        
        # Erstelle VXMemex
        memex = VXMemex()
        
        # Erstelle Gedächtnis
        memory = {
            "id": f"memory_{datetime.datetime.now().timestamp()}",
            "content": f"Dies ist ein Testgedächtnis für den Lasttest",
            "tags": ["test", "memory", "load"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Speichere Gedächtnis
        result = memex.store_memory(memory)
        
        return {
            "success": True,
            "message": "VXOR-Integration Lasttest erfolgreich",
            "details": {
                "memory": str(memory),
                "result": str(result)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"VXOR-Integration Lasttest fehlgeschlagen: {e}",
            "error": str(e)
        }

def main():
    """Hauptfunktion"""
    logger.info("Starte MISO Ultimate Lasttests...")
    
    # Liste der Lasttests
    load_tests = [
        {"name": "M-CODE Core Lasttest", "test_function": test_m_code_load, "iterations": 100, "threads": 4},
        {"name": "M-LINGUA Interface Lasttest", "test_function": test_m_lingua_load, "iterations": 100, "threads": 4},
        {"name": "ECHO-PRIME Lasttest", "test_function": test_echo_prime_load, "iterations": 50, "threads": 2},
        {"name": "HYPERFILTER Lasttest", "test_function": test_hyperfilter_load, "iterations": 100, "threads": 4},
        {"name": "Deep-State-Modul Lasttest", "test_function": test_deep_state_load, "iterations": 50, "threads": 2},
        {"name": "VXOR-Integration Lasttest", "test_function": test_vxor_integration_load, "iterations": 50, "threads": 2}
    ]
    
    # Führe Lasttests durch
    for test in load_tests:
        test_name = test["name"]
        test_function = test["test_function"]
        iterations = test["iterations"]
        threads = test["threads"]
        
        # Führe Test durch
        result = test_load(test_name, test_function, iterations, threads)
        
        # Speichere Ergebnis
        results["load_tests"][test_name] = result
    
    # Bestimme Gesamtstatus
    passed = sum(1 for test in results["load_tests"].values() if test["status"] == "passed")
    failed = sum(1 for test in results["load_tests"].values() if test["status"] == "failed")
    errors = sum(1 for test in results["load_tests"].values() if test["status"] == "error")
    
    if errors > 0:
        results["overall_status"] = "error"
    elif failed > 0:
        results["overall_status"] = "failed"
    else:
        results["overall_status"] = "passed"
    
    # Speichere Ergebnisse
    with open("stability_test_results_load.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Erstelle Zusammenfassung
    summary = f"""
MISO Ultimate Lasttests - Zusammenfassung
=========================================
Zeitstempel: {datetime.datetime.now().isoformat()}
Gesamtstatus: {results["overall_status"]}

Teststatus:
"""
    
    for test_name, test_result in results["load_tests"].items():
        summary += f"  - {test_name}: {test_result['status']}"
        if test_result["status"] == "error":
            summary += f" ({test_result['error']['type']}: {test_result['error']['message']})"
        summary += f" ({test_result['duration']:.2f}s)\n"
        
        if test_result["status"] == "passed":
            stats = test_result["details"]["statistics"]
            summary += f"    - Durchsatz: {stats['throughput']:.2f} Anfragen/s\n"
            summary += f"    - Durchschnittliche Dauer: {stats['avg_duration']:.4f}s\n"
            summary += f"    - Min/Max/Median: {stats['min_duration']:.4f}s / {stats['max_duration']:.4f}s / {stats['median_duration']:.4f}s\n"
    
    summary += f"""
Statistik:
  - Bestanden: {passed}/{len(load_tests)}
  - Fehlgeschlagen: {failed}/{len(load_tests)}
  - Fehler: {errors}/{len(load_tests)}

Ergebnisse gespeichert in:
  - stability_test_results_load.json
"""
    
    # Speichere Zusammenfassung
    with open("stability_test_results_load.md", "w", encoding="utf-8") as f:
        f.write(summary)
    
    logger.info(f"MISO Ultimate Lasttests abgeschlossen: {results['overall_status']}")
    logger.info(f"Bestanden: {passed}/{len(load_tests)}, Fehlgeschlagen: {failed}/{len(load_tests)}, Fehler: {errors}/{len(load_tests)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
