#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hauptskript für SVD-Benchmark

Dieses Skript führt den fortschrittlichen SVD-Benchmark aus und bereitet
die Ergebnisse grafisch und tabellarisch auf.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MISO.math.t_mathematics.optimizations.run_svd_benchmark")

# Benchmark-Funktionen importieren
try:
    from miso.math.t_mathematics.optimizations.advanced_svd_benchmark_functions import SVDBenchmarker
except ImportError:
    logger.error("Konnte SVDBenchmarker nicht importieren. Bitte stellen Sie sicher, dass alle Abhängigkeiten installiert sind.")
    sys.exit(1)

def parse_arguments():
    """
    Kommandozeilenargumente parsen
    
    Returns:
        Argumente-Namespace
    """
    parser = argparse.ArgumentParser(description="SVD-Benchmark für T-Mathematics Engine Optimierungen")
    
    parser.add_argument("--tiny", action="store_true", help="Nur winzige Matrizen (8x8) testen")
    parser.add_argument("--small", action="store_true", help="Nur kleine Matrizen (32x32) testen")
    parser.add_argument("--medium", action="store_true", help="Nur mittlere Matrizen (128x128) testen")
    parser.add_argument("--large", action="store_true", help="Nur große Matrizen (256x256) testen")
    parser.add_argument("--full", action="store_true", help="Nur vollständige SVD testen")
    parser.add_argument("--k", type=int, nargs="+", default=[4, 16], help="k-Werte für partielle SVD")
    parser.add_argument("--runs", type=int, default=3, help="Anzahl der Durchläufe pro Test")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 2, 3], help="Zu testende Optimierungsstufen")
    parser.add_argument("--json", type=str, help="Ergebnisse in JSON-Datei speichern")
    parser.add_argument("--csv", type=str, help="Ergebnisse in CSV-Datei speichern")
    parser.add_argument("--plot", type=str, help="Ergebnisse als Diagramm speichern")
    parser.add_argument("--precision", type=str, choices=["float16", "float32"], default="float32", help="Präzision für Tests")
    
    return parser.parse_args()

def print_results_table(results, analysis):
    """
    Gibt Benchmark-Ergebnisse als Tabelle aus
    
    Args:
        results: Benchmark-Ergebnisse
        analysis: Analyseergebnisse
    """
    print("\n" + "=" * 100)
    print("SVD BENCHMARK ERGEBNISSE".center(100))
    print("=" * 100)
    
    # Hole Optimierungsstufen und Matrixnamen
    levels = sorted([int(level.split('_')[1]) for level in results.keys()])
    matrix_names = list(next(iter(results.values())).keys())
    
    for matrix_name in matrix_names:
        print(f"\nMatrix: {matrix_name}")
        print("-" * 100)
        
        # Header
        header = "SVD-Typ".ljust(15)
        for level in levels:
            level_name = f"Level {level}"
            header += level_name.center(20)
        
        header += "Verbesserung".center(25)
        print(header)
        print("-" * 100)
        
        # Vollständige SVD
        row = "Vollständig".ljust(15)
        
        for level in levels:
            level_key = f"level_{level}"
            if level_key in results and matrix_name in results[level_key]:
                full_svd = results[level_key][matrix_name]["full_svd"]
                if "avg_time" in full_svd:
                    row += f"{full_svd['avg_time']:.6f} s".center(20)
                else:
                    row += "N/A".center(20)
            else:
                row += "N/A".center(20)
        
        # Verbesserung
        if matrix_name in analysis["improvements"] and "full_svd" in analysis["improvements"][matrix_name]:
            improvement = analysis["improvements"][matrix_name]["full_svd"]
            best_level = analysis["best_levels"][matrix_name]["full_svd"]
            row += f"{improvement:.2f}% (Level {best_level})".center(25)
        else:
            row += "N/A".center(25)
        
        print(row)
        
        # Partielle SVD für jeden k-Wert
        for k in sorted(set(sum([list(matrix_results["partial_svd"].keys()) for level_results in results.values() for matrix_results in level_results.values()], []))):
            row = k.ljust(15)
            
            for level in levels:
                level_key = f"level_{level}"
                if (level_key in results and matrix_name in results[level_key] and 
                    k in results[level_key][matrix_name]["partial_svd"] and 
                    "avg_time" in results[level_key][matrix_name]["partial_svd"][k]):
                    avg_time = results[level_key][matrix_name]["partial_svd"][k]["avg_time"]
                    row += f"{avg_time:.6f} s".center(20)
                else:
                    row += "N/A".center(20)
            
            # Verbesserung
            if matrix_name in analysis["improvements"] and k in analysis["improvements"][matrix_name]:
                improvement = analysis["improvements"][matrix_name][k]
                best_level = analysis["best_levels"][matrix_name][k]
                row += f"{improvement:.2f}% (Level {best_level})".center(25)
            else:
                row += "N/A".center(25)
            
            print(row)
    
    print("\n" + "=" * 100)
    print("ZUSAMMENFASSUNG".center(100))
    print("=" * 100)
    
    if "avg_improvement" in analysis["summary"]:
        print(f"\nDurchschnittliche Verbesserung: {analysis['summary']['avg_improvement']:.2f}%")
    
    print("\nOptimierungsstufen:")
    print("  0 = Keine Optimierung (Baseline)")
    print("  2 = Standard-Optimierung (SVD-Optimierung)")
    print("  3 = Aggressive Optimierung (Hybride SVD)")

def create_plots(results, analysis, filename=None):
    """
    Erstellt Diagramme aus den Benchmark-Ergebnissen
    
    Args:
        results: Benchmark-Ergebnisse
        analysis: Analyseergebnisse
        filename: Dateiname zum Speichern des Diagramms
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        logger.error("Matplotlib nicht verfügbar, überspringe Diagrammerstellung")
        return
    
    # Hole Optimierungsstufen und Matrixnamen
    levels = sorted([int(level.split('_')[1]) for level in results.keys()])
    matrix_names = list(next(iter(results.values())).keys())
    
    # Erstelle Farben für Diagramme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Größe des Diagramms
    plt.figure(figsize=(14, 8))
    
    # Balkenbreite
    bar_width = 0.15
    
    # X-Positionen für Balken
    x = np.arange(len(matrix_names))
    
    # Sammle Zeiten für vollständige SVD
    full_svd_times = {}
    for level in levels:
        full_svd_times[level] = []
        
        for matrix_name in matrix_names:
            level_key = f"level_{level}"
            if level_key in results and matrix_name in results[level_key]:
                full_svd = results[level_key][matrix_name]["full_svd"]
                if "avg_time" in full_svd:
                    full_svd_times[level].append(full_svd["avg_time"])
                else:
                    full_svd_times[level].append(0)
            else:
                full_svd_times[level].append(0)
    
    # Erste Grafik: Vollständige SVD
    plt.subplot(2, 1, 1)
    
    for i, level in enumerate(levels):
        plt.bar(x + i * bar_width - (len(levels) - 1) * bar_width / 2, full_svd_times[level], bar_width, 
                label=f'Level {level}', color=colors[i % len(colors)])
    
    plt.xlabel('Matrix')
    plt.ylabel('Zeit (s)')
    plt.title('Vollständige SVD: Ausführungszeit nach Optimierungsstufe')
    plt.xticks(x, matrix_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Zweite Grafik: Partielle SVD (k-Werte)
    plt.subplot(2, 1, 2)
    
    # Sammle alle k-Werte
    all_k_values = sorted(set(sum([list(matrix_results["partial_svd"].keys()) for level_results in results.values() for matrix_results in level_results.values()], [])))
    
    # Nehme den ersten k-Wert für die Grafik
    if all_k_values:
        first_k = all_k_values[0]
        
        # Sammle Zeiten für partielle SVD
        partial_svd_times = {}
        for level in levels:
            partial_svd_times[level] = []
            
            for matrix_name in matrix_names:
                level_key = f"level_{level}"
                if (level_key in results and matrix_name in results[level_key] and 
                    first_k in results[level_key][matrix_name]["partial_svd"] and 
                    "avg_time" in results[level_key][matrix_name]["partial_svd"][first_k]):
                    avg_time = results[level_key][matrix_name]["partial_svd"][first_k]["avg_time"]
                    partial_svd_times[level].append(avg_time)
                else:
                    partial_svd_times[level].append(0)
        
        for i, level in enumerate(levels):
            plt.bar(x + i * bar_width - (len(levels) - 1) * bar_width / 2, partial_svd_times[level], bar_width, 
                    label=f'Level {level}', color=colors[i % len(colors)])
        
        plt.xlabel('Matrix')
        plt.ylabel('Zeit (s)')
        plt.title(f'Partielle SVD ({first_k}): Ausführungszeit nach Optimierungsstufe')
        plt.xticks(x, matrix_names)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=2.0)
    
    # Speichere Diagramm
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Diagramm in '{filename}' gespeichert")
    else:
        plt.show()

def main():
    """
    Hauptfunktion zum Ausführen des Benchmarks
    """
    # Argumente parsen
    args = parse_arguments()
    
    # Benchmark-Konfiguration erstellen
    config = {
        "precision": args.precision,
        "runs_per_test": args.runs,
        "optimization_levels": args.levels,
        "k_values": args.k
    }
    
    # Matrixgrößen basierend auf Argumenten auswählen
    matrix_sizes = []
    
    # Wenn keine spezifische Größe angegeben wurde, alle verwenden
    if not any([args.tiny, args.small, args.medium, args.large]):
        matrix_sizes = [
            ("tiny", (8, 8)),
            ("small", (32, 32)),
            ("medium", (128, 128)),
            ("large", (256, 256))
        ]
    else:
        if args.tiny:
            matrix_sizes.append(("tiny", (8, 8)))
        if args.small:
            matrix_sizes.append(("small", (32, 32)))
        if args.medium:
            matrix_sizes.append(("medium", (128, 128)))
        if args.large:
            matrix_sizes.append(("large", (256, 256)))
    
    config["matrix_sizes"] = matrix_sizes
    
    # SVD-Benchmarker erstellen
    benchmarker = SVDBenchmarker(config)
    
    # Benchmark ausführen
    logger.info("Führe SVD-Benchmark aus...")
    results = benchmarker.run_benchmark()
    
    if not results:
        logger.error("Benchmark hat keine Ergebnisse geliefert")
        return
    
    # Ergebnisse analysieren
    analysis = benchmarker.analyze_results()
    
    # Ergebnisse ausgeben
    print_results_table(results, analysis)
    
    # Diagramme erstellen
    if args.plot:
        create_plots(results, analysis, args.plot)
    
    logger.info("Benchmark abgeschlossen")

if __name__ == "__main__":
    main()
