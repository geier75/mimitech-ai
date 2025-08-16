#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chart-Funktionen für den HTML-Reporter.

Dieses Modul stellt Funktionen für die Datenaufbereitung und Analyse
von Benchmark-Daten für die Visualisierung bereit.
"""

import logging
from typing import List, Dict, Any, Tuple

# Setup logging
logger = logging.getLogger(__name__)

def _prepare_chart_data(self, results) -> Dict[str, Any]:
    """Bereitet Daten für die Charts vor.
    
    Args:
        results: Liste der Benchmark-Ergebnisse
        
    Returns:
        Dictionary mit aufbereiteten Chart-Daten
    """
    chart_data = {
        "operations": {},
        "backends": {},
        "memory": {}
    }
    
    # Gruppiere Daten nach Operation und Backend
    for result in results:
        op_name = result.operation.name
        backend_name = result.backend.name
        dimension = result.dimension
        
        # Initialisiere Operation-Daten, falls nötig
        if op_name not in chart_data["operations"]:
            chart_data["operations"][op_name] = {"backends": {}}
        
        # Initialisiere Backend-Daten für diese Operation, falls nötig
        if backend_name not in chart_data["operations"][op_name]["backends"]:
            chart_data["operations"][op_name]["backends"][backend_name] = {
                "dimensions": [],
                "times": [],
                "std_devs": []
            }
        
        # Füge Daten hinzu
        chart_data["operations"][op_name]["backends"][backend_name]["dimensions"].append(dimension)
        chart_data["operations"][op_name]["backends"][backend_name]["times"].append(result.mean_time)
        chart_data["operations"][op_name]["backends"][backend_name]["std_devs"].append(result.std_dev)
        
        # Speichernutzung, falls verfügbar
        if hasattr(result, 'memory_change') and result.memory_change != 0:
            if op_name not in chart_data["memory"]:
                chart_data["memory"][op_name] = {}
            
            if backend_name not in chart_data["memory"][op_name]:
                chart_data["memory"][op_name][backend_name] = []
            
            chart_data["memory"][op_name][backend_name].append({
                "dimension": dimension,
                "memory_change": result.memory_change / (1024 * 1024)  # In MB
            })
    
    # Sortierfunktion für die Dimensionen (wichtig für korrekte Diagramme)
    for op_name in chart_data["operations"]:
        for backend_name in chart_data["operations"][op_name]["backends"]:
            backend_data = chart_data["operations"][op_name]["backends"][backend_name]
            # Sortiere nach Dimension
            sorted_data = sorted(zip(backend_data["dimensions"], 
                                    backend_data["times"], 
                                    backend_data["std_devs"]))
            
            if sorted_data:
                backend_data["dimensions"], backend_data["times"], backend_data["std_devs"] = zip(*sorted_data)
    
    return chart_data

def _get_fastest_backend(self, results) -> Tuple[str, float]:
    """Ermittelt das schnellste Backend und die Beschleunigung.
    
    Args:
        results: Liste der Benchmark-Ergebnisse
        
    Returns:
        (schnellstes_backend, beschleunigung)
    """
    # Gruppiere nach Backend
    backend_times = {}
    for result in results:
        backend_name = result.backend.name
        if backend_name not in backend_times:
            backend_times[backend_name] = []
        
        backend_times[backend_name].append(result.mean_time)
    
    # Berechne durchschnittliche Zeit pro Backend
    backend_avg_times = {}
    for backend, times in backend_times.items():
        if times:
            backend_avg_times[backend] = sum(times) / len(times)
    
    # Finde schnellstes Backend
    if not backend_avg_times:
        return "Unbekannt", 0.0
        
    fastest_backend = min(backend_avg_times.items(), key=lambda x: x[1])
    
    # Berechne Beschleunigung gegenüber dem Durchschnitt
    avg_time = sum(backend_avg_times.values()) / len(backend_avg_times)
    speedup = avg_time / fastest_backend[1] if fastest_backend[1] > 0 else 1.0
    
    return fastest_backend[0], speedup

def _get_fastest_operation(self, results) -> Tuple[str, float]:
    """Ermittelt die schnellste Operation und ihre Ausführungszeit.
    
    Args:
        results: Liste der Benchmark-Ergebnisse
        
    Returns:
        (schnellste_operation, ausführungszeit)
    """
    # Gruppiere nach Operation
    operation_times = {}
    for result in results:
        op_name = result.operation.name
        if op_name not in operation_times:
            operation_times[op_name] = []
        
        operation_times[op_name].append(result.mean_time)
    
    # Berechne durchschnittliche Zeit pro Operation
    operation_avg_times = {}
    for operation, times in operation_times.items():
        if times:
            operation_avg_times[operation] = sum(times) / len(times)
    
    # Finde schnellste Operation
    if not operation_avg_times:
        return "Unbekannt", 0.0
        
    fastest_operation = min(operation_avg_times.items(), key=lambda x: x[1])
    
    return fastest_operation[0], fastest_operation[1]
