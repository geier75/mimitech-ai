#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empfehlungs- und Analysefunktionen für den HTML-Reporter.

Dieses Modul stellt Funktionen für die Generierung von Empfehlungen und
historischen Vergleichen basierend auf Benchmark-Ergebnissen bereit.
"""

import logging
from typing import List, Dict, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

def _generate_recommendations(self, results) -> List[str]:
    """Generiert automatische Empfehlungen basierend auf den Benchmark-Ergebnissen.
    
    Args:
        results: Liste der Benchmark-Ergebnisse
        
    Returns:
        Liste von Empfehlungen als Strings
    """
    recommendations = []
    
    # Gruppiere Ergebnisse nach Operation und Backend
    operations = {}
    for result in results:
        op_name = result.operation.name
        if op_name not in operations:
            operations[op_name] = {}
        
        backend_name = result.backend.name
        if backend_name not in operations[op_name]:
            operations[op_name][backend_name] = []
            
        operations[op_name][backend_name].append(result)
    
    # Analysiere für jede Operation das beste Backend
    for op_name, backends in operations.items():
        if len(backends) < 2:
            continue
            
        # Finde das beste Backend für diese Operation
        best_backend = None
        best_time = float('inf')
        
        for backend_name, results_list in backends.items():
            # Berechne durchschnittliche Zeit über alle Dimensionen
            avg_time = sum(r.mean_time for r in results_list) / len(results_list)
            
            if avg_time < best_time:
                best_time = avg_time
                best_backend = backend_name
        
        # Generiere Empfehlung
        recommendations.append(
            f"Für die Operation {op_name} ist {best_backend} das schnellste Backend "
            f"mit einer durchschnittlichen Zeit von {best_time:.6f}s."
        )
    
    # MLX und T-Mathematics-spezifische Empfehlungen
    mlx_results = [r for r in results if r.backend.name == 'MLX']
    if mlx_results:
        # Spezifische Empfehlung für Apple Silicon mit ANE
        recommendations.append(
            "Die Apple Neural Engine (ANE) zeigt besondere Leistungsstärke bei Matrix-Operationen. "
            "Für optimale Leistung auf Apple Silicon empfehlen wir die Verwendung von MLX mit BF16-Präzision, "
            "die spezifisch für die T-Mathematics Engine optimiert ist."
        )
            
        bfloat16_results = [r for r in mlx_results if r.precision.name == 'BFLOAT16']
        if bfloat16_results:
            avg_bfloat16_time = sum(r.mean_time for r in bfloat16_results) / len(bfloat16_results)
            float32_results = [r for r in mlx_results if r.precision.name == 'FLOAT32']
            if float32_results:
                avg_float32_time = sum(r.mean_time for r in float32_results) / len(float32_results)
                if avg_bfloat16_time < avg_float32_time:
                    speedup = avg_float32_time / avg_bfloat16_time
                    recommendations.append(
                        f"BFloat16-Präzision ist {speedup:.2f}x schneller als Float32 bei MLX "
                        f"und bietet eine gute Balance zwischen Genauigkeit und Geschwindigkeit für die "
                        f"T-Mathematics Engine."
                    )
    
    # Dimensionsspezifische Empfehlungen
    large_dim_results = [r for r in results if r.dimension >= 512]
    if large_dim_results:
        backend_performance = {}
        for r in large_dim_results:
            if r.backend.name not in backend_performance:
                backend_performance[r.backend.name] = []
            backend_performance[r.backend.name].append(r.mean_time)
        
        # Durchschnittliche Performance pro Backend für große Dimensionen
        backend_avg_performance = {}
        for backend, times in backend_performance.items():
            backend_avg_performance[backend] = sum(times) / len(times)
        
        if backend_avg_performance:
            best_large_dim_backend = min(backend_avg_performance.items(), key=lambda x: x[1])[0]
            recommendations.append(
                f"Für Matrizen mit großen Dimensionen (512+) ist {best_large_dim_backend} "
                f"am effizientesten, was besonders für komplexe MPRIME-Engine Operationen relevant ist."
            )
    
    # T-Mathematics und M-CODE spezifische Empfehlungen
    recommendations.append(
        "Die T-Mathematics Engine bietet optimierte Tensor-Operationen speziell für die "
        "MISO-Kernkomponenten und ist besonders effektiv für:"
    )
    recommendations.append(
        "- Parallele Zeitlinien-Berechnungen in der ECHO-PRIME Engine"
    )
    recommendations.append(
        "- Symboltransformationen in der MPRIME Engine, insbesondere mit SymbolTree und TopoNet"
    )
    recommendations.append(
        "- Beschleunigte Berechnungen für den QTM-Modulator mit MLX-Backend"
    )
    
    # Spezifische Empfehlung für Apple Silicon
    if any(hasattr(r, 'metadata') and r.metadata.get('is_apple_silicon', False) for r in results):
        recommendations.append(
            "Ihre Hardware nutzt Apple Silicon - für MISO wird automatisch die ANE über "
            "die MLXTensor-Implementierung verwendet, was erhebliche Beschleunigungen für "
            "die Paradoxauflösung ermöglicht."
        )
    
    return recommendations

def _prepare_historical_comparison(self, current_results, historical_datasets) -> Dict[str, Any]:
    """Bereitet historische Vergleichsdaten vor.
    
    Args:
        current_results: Aktuelle Benchmark-Ergebnisse
        historical_datasets: Historische Datensätze
        
    Returns:
        Dictionary mit aufbereiteten historischen Vergleichsdaten
    """
    if not historical_datasets:
        return None
        
    historical_comparison = {
        "timestamps": [],
        "datasets": [],
        "performance_changes": {},
        "operation_trends": {},
        "apple_silicon_boost": [],
        "t_math_versions": []
    }
    
    # Extrahiere aktuelle Daten für den Vergleich
    current_data = {}
    for result in current_results:
        op_name = result.operation.name
        backend_name = result.backend.name
        dimension = result.dimension
        
        key = f"{op_name}_{backend_name}_{dimension}"
        current_data[key] = {
            "mean_time": result.mean_time,
            "operation": op_name,
            "backend": backend_name,
            "dimension": dimension
        }
    
    # Vergleiche mit historischen Daten
    for i, dataset in enumerate(historical_datasets):
        timestamp = dataset.get('timestamp', f"Datensatz {i+1}")
        historical_comparison["timestamps"].append(timestamp)
        
        # T-Mathematics Version
        t_math_version = "nicht verfügbar"
        if 'metadata' in dataset and 'system_info' in dataset['metadata']:
            t_math_version = dataset['metadata']['system_info'].get('t_math_version', "nicht verfügbar")
        historical_comparison["t_math_versions"].append(t_math_version)
        
        # Performance-Vergleich
        dataset_comparison = {
            "timestamp": timestamp,
            "improvements": [],
            "regressions": [],
            "unchanged": []
        }
        
        # Historische Daten für Vergleich sammeln
        historical_data = {}
        for result in dataset.get('results', []):
            op_name = result.get('operation', '')
            backend_name = result.get('backend', '')
            dimension = result.get('dimension', 0)
            
            key = f"{op_name}_{backend_name}_{dimension}"
            historical_data[key] = {
                "mean_time": result.get('mean_time', 0.0),
                "operation": op_name,
                "backend": backend_name,
                "dimension": dimension
            }
        
        # Vergleiche aktuelle mit historischen Daten
        for key, current in current_data.items():
            if key in historical_data:
                historical = historical_data[key]
                if current["mean_time"] > 0 and historical["mean_time"] > 0:
                    change_percent = (current["mean_time"] - historical["mean_time"]) / historical["mean_time"] * 100
                    
                    comparison_item = {
                        "operation": current["operation"],
                        "backend": current["backend"],
                        "dimension": current["dimension"],
                        "current_time": current["mean_time"],
                        "historical_time": historical["mean_time"],
                        "change_percent": change_percent
                    }
                    
                    # Kategorisiere als Verbesserung, Regression oder unverändert
                    if change_percent < -5:  # Mehr als 5% schneller
                        dataset_comparison["improvements"].append(comparison_item)
                    elif change_percent > 5:  # Mehr als 5% langsamer
                        dataset_comparison["regressions"].append(comparison_item)
                    else:
                        dataset_comparison["unchanged"].append(comparison_item)
        
        historical_comparison["datasets"].append(dataset_comparison)
        
        # Prüfe, ob Apple Silicon verwendet wurde (besonders wichtig für MISO)
        apple_silicon = False
        mlx_available = False
        
        if 'metadata' in dataset and 'system_info' in dataset['metadata']:
            system_info = dataset['metadata']['system_info']
            apple_silicon = system_info.get('is_apple_silicon', False)
            mlx_available = 'mlx_version' in system_info
            
        historical_comparison["apple_silicon_boost"].append({
            "timestamp": timestamp,
            "boost": apple_silicon and mlx_available,
            "mlx_available": mlx_available,
            "apple_silicon": apple_silicon
        })
    
    return historical_comparison
