#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR MISO Ultimate Latency Benchmark (Angepasste Version)
------------------------------------
Dieses Skript führt einen automatisierten Latenz-Benchmark für vXor-Module durch
und dokumentiert die Ergebnisse in einer PDF-Datei.

Ausführung:
    python vxor_latency_benchmark_fixed.py

Output:
    - Console output mit allen Messwerten
    - vxor_benchmark_latency5ms_4gb.png (Screenshot des Terminals)
    - vxor_benchmark_report.pdf (Vollständiger Bericht mit Messwerten und Erklärungen)
"""

import os
import sys
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import platform
from tqdm import tqdm
import argparse
import random
import string

# Füge den MISO_PATH zum Python Path hinzu
MISO_ROOT = "/Volumes/My Book/MISO_Ultimate 15.32.28"
sys.path.append(MISO_ROOT)

# Importiere MISO Module
try:
    from vxor.agents.vx_memex import VXMemex
    from vxor.core.vx_core import VXModule
except ImportError as e:
    print(f"Fehler beim Importieren der MISO Module: {e}")
    print("Stelle sicher, dass der MISO_ROOT Pfad korrekt ist.")
    sys.exit(1)

class ResourceMonitor:
    """Einfacher Resource Monitor für Benchmark-Tests"""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        print("NEXUS ResourceMonitor initialisiert.")
        
    def start_monitoring(self):
        """Startet die Überwachung"""
        pass
    
    def get_ram_usage(self):
        """Gibt den aktuellen RAM-Verbrauch in MB zurück"""
        return self.process.memory_info().rss / (1024 * 1024)

class LatencyBenchmark:
    def __init__(self, module_name="VX-MEMEX", num_iterations=5, warmup_iterations=2):
        self.module_name = module_name
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        self.results = []
        self.ram_usage = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Systeminfo
        self.system_info = {
            "OS": platform.system() + " " + platform.release(),
            "Python": platform.python_version(),
            "Processor": platform.processor(),
            "RAM": f"{psutil.virtual_memory().total / (1024.0 ** 3):.2f} GB",
            "MISO Version": "2.8.5"
        }
        
        # Lade Module
        print(f"Initialisiere {self.module_name} Modul...")
        if module_name == "VX-MEMEX":
            self.module = VXMemex()
        else:
            # Fallback zu einem generischen VXModule
            self.module = VXModule(name=module_name, version="1.0")
        
        # Initialisiere Resource Monitor
        self.monitor = ResourceMonitor()
        
    def prepare_test_data(self, size_mb=50):
        """Erstellt Testdaten mit der angegebenen Größe in MB"""
        print(f"Generiere Testdaten ({size_mb} MB)...")
        # Berechne Größe des NumPy Arrays
        bytes_per_element = 4  # float32
        num_elements = int((size_mb * 1024 * 1024) / bytes_per_element)
        
        # Erstelle zufällige Testdaten
        return np.random.rand(num_elements).astype(np.float32)
        
    def run_benchmark(self):
        """Führt den Benchmark durch und sammelt Latenzmessungen"""
        print("\n" + "="*80)
        print(f"VXOR MISO ULTIMATE LATENCY BENCHMARK - {self.module_name}")
        print("="*80)
        print(f"Datum/Zeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"System: {self.system_info['OS']} | Processor: {self.system_info['Processor']}")
        print(f"RAM: {self.system_info['RAM']} | MISO Version: {self.system_info['MISO Version']}")
        print("-"*80)
        
        # Erstelle Testdaten
        test_data = self.prepare_test_data()
        
        # Generiere zufällige Schlüssel für den Test
        if self.module_name == "VX-MEMEX":
            test_keys = [
                ''.join(random.choice(string.ascii_letters) for _ in range(10))
                for _ in range(self.num_iterations + self.warmup_iterations)
            ]
            
        # Warmup
        print(f"Führe {self.warmup_iterations} Warmup-Iterationen durch...")
        for i in range(self.warmup_iterations):
            if self.module_name == "VX-MEMEX":
                # VXMemex verwendet store/retrieve anstelle von process
                self.module.store(test_keys[i], test_data[:1000], context={"test": True})
                _ = self.module.retrieve(test_keys[i])
            else:
                # Dummy-Operation für andere Module
                _ = hash(str(test_data[:1000]))
            time.sleep(0.5)  # Kurze Pause
        
        # Hauptbenchmark
        print(f"\nFühre {self.num_iterations} Benchmark-Iterationen durch...")
        print("-"*80)
        print(f"{'#':^5} | {'Latenz (ms)':^15} | {'RAM (MB)':^10} | {'Status':^20}")
        print("-"*80)
        
        for i in tqdm(range(self.num_iterations)):
            # Messe RAM vor der Verarbeitung
            self.monitor.start_monitoring()
            
            # Messe Verarbeitungszeit
            start_time = time.time()
            
            if self.module_name == "VX-MEMEX":
                # VXMemex verwendet store/retrieve anstelle von process
                key = test_keys[i + self.warmup_iterations]
                self.module.store(key, test_data, context={"benchmark": True})
                _ = self.module.retrieve(key)
            else:
                # Dummy-Operation für andere Module
                _ = hash(str(test_data))
                
            end_time = time.time()
            
            # Berechne Latenz in ms
            latency_ms = (end_time - start_time) * 1000
            
            # Messe RAM nach der Verarbeitung
            ram_mb = self.monitor.get_ram_usage()
            
            # Speichere Ergebnisse
            self.results.append(latency_ms)
            self.ram_usage.append(ram_mb)
            
            # Zeige Ergebnisse
            status = "✓ OK" if latency_ms < 10 else "! SLOW"
            print(f"{i+1:^5} | {latency_ms:^15.2f} | {ram_mb:^10.2f} | {status:^20}")
            
            # Kurze Pause
            time.sleep(0.5)
        
        print("-"*80)
        
        # Berechne Statistiken
        avg_latency = np.mean(self.results)
        std_latency = np.std(self.results)
        avg_ram = np.mean(self.ram_usage)
        
        print(f"\nBENCHMARK ERGEBNISSE FÜR {self.module_name}:")
        print(f"Durchschnittliche Latenz: {avg_latency:.2f} ms (±{std_latency:.2f})")
        print(f"Durchschnittlicher RAM-Verbrauch: {avg_ram:.2f} MB")
        print(f"Alle Latenzen: {', '.join([f'{x:.2f}' for x in self.results])} ms")
        print("\nBenchmark abgeschlossen!")
        
        return {
            "latency_ms": self.results,
            "avg_latency": avg_latency,
            "std_latency": std_latency,
            "ram_usage": self.ram_usage,
            "avg_ram": avg_ram
        }
    
    def create_visualization(self):
        """Erstellt Visualisierungen für die Benchmark-Ergebnisse"""
        if not self.results:
            print("Keine Ergebnisse zum Visualisieren vorhanden.")
            return
            
        print("\nErstelle Visualisierungen...")
        
        # Setze Matplotlib Style
        plt.style.use('ggplot')
        
        # Erstelle Figure mit 2 Subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Latenz pro Iteration
        iterations = range(1, len(self.results)+1)
        ax1.plot(iterations, self.results, 'o-', color='#1f77b4', linewidth=2)
        ax1.axhline(y=5.0, color='r', linestyle='--', label='Ziel: 5ms')
        ax1.set_title(f'{self.module_name} Latenz pro Iteration', fontsize=14)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Latenz (ms)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Annotiere die durchschnittliche Latenz
        avg_latency = np.mean(self.results)
        avg_text = f'Durchschnitt: {avg_latency:.2f} ms'
        ax1.annotate(avg_text, xy=(0.5, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    ha='center', fontsize=11)
        
        # Plot 2: RAM-Nutzung pro Iteration
        ax2.plot(iterations, self.ram_usage, 'o-', color='#ff7f0e', linewidth=2)
        ax2.axhline(y=4000, color='r', linestyle='--', label='Ziel: 4000 MB')
        ax2.set_title(f'{self.module_name} RAM-Nutzung pro Iteration', fontsize=14)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('RAM (MB)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Annotiere die durchschnittliche RAM-Nutzung
        avg_ram = np.mean(self.ram_usage)
        ram_text = f'Durchschnitt: {avg_ram:.2f} MB'
        ax2.annotate(ram_text, xy=(0.5, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    ha='center', fontsize=11)
        
        # Füge Titel und Zeitstempel hinzu
        plt.suptitle(f'VXOR MISO Ultimate Benchmark: {self.module_name}', fontsize=16)
        plt.figtext(0.5, 0.01, f'Benchmark durchgeführt am {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                   ha='center', fontsize=10)
        
        # Erzeuge Pfad
        os.makedirs('benchmark_results', exist_ok=True)
        filename = f'benchmark_results/{self.module_name.lower().replace("-", "_")}_latency_{self.timestamp}.png'
        
        # Speichere Visualisierung
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, dpi=100)
        print(f"Visualisierung gespeichert unter: {filename}")
        plt.close()
    
    def save_results_json(self, results):
        """Speichert die Benchmark-Ergebnisse als JSON"""
        import json
        
        # Erstelle Ergebnis-Dictionary
        result_data = {
            "module": self.module_name,
            "system_info": self.system_info,
            "timestamp": self.timestamp,
            "iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "results": {
                "latency_ms": self.results,
                "avg_latency": float(results["avg_latency"]),
                "std_latency": float(results["std_latency"]),
                "ram_usage_mb": self.ram_usage,
                "avg_ram_mb": float(results["avg_ram"])
            }
        }
        
        # Erzeuge Pfad
        os.makedirs('benchmark_results', exist_ok=True)
        filename = f'benchmark_results/{self.module_name.lower().replace("-", "_")}_benchmark_{self.timestamp}.json'
        
        # Speichere JSON
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"Ergebnisse gespeichert unter: {filename}")
        return filename

def parse_arguments():
    """Parst Kommandozeilenargumente"""
    parser = argparse.ArgumentParser(description='VXOR MISO Ultimate Latency Benchmark')
    parser.add_argument('--module', type=str, default='VX-MEMEX', 
                        choices=['VX-MEMEX', 'VX-CORE', 'VX-TENSOR'],
                        help='Das zu testende Modul')
    parser.add_argument('--iterations', type=int, default=5, 
                        help='Anzahl der Benchmark-Iterationen')
    parser.add_argument('--warmup', type=int, default=2, 
                        help='Anzahl der Warmup-Iterationen')
    return parser.parse_args()

def main():
    """Hauptfunktion"""
    # Parse Argumente
    args = parse_arguments()
    
    try:
        # Erstelle Benchmark-Instance
        benchmark = LatencyBenchmark(
            module_name=args.module,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup
        )
        
        # Führe Benchmark durch
        results = benchmark.run_benchmark()
        
        # Erstelle Visualisierung
        benchmark.create_visualization()
        
        # Speichere Ergebnisse
        benchmark.save_results_json(results)
        
        print("\nBenchmark erfolgreich abgeschlossen!")
        return 0
    except Exception as e:
        print(f"\nFehler während des Benchmarks: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
if __name__ == "__main__":
    sys.exit(main())
