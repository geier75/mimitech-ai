"""
VX-MATRIX Profiling Utilities
============================

Dieses Modul enthält Utilities für das Profiling und die Performance-Analyse des VX-MATRIX-Moduls.
ZTM-konform für MISO Ultimate.
"""

import cProfile
import pstats
import io
import time
import functools
import numpy as np
import os
import json
from datetime import datetime

# Profiling-Verzeichnis
PROFILING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'profiling')
os.makedirs(PROFILING_DIR, exist_ok=True)

# Performance-Metriken
class PerformanceMetrics:
    """Sammelt Performance-Metriken für Matrix-Operationen."""
    
    def __init__(self):
        self.metrics = {
            'matrix_inverse': {},
            'matrix_multiply': {},
            'cholesky_inverse': {},
            'tikhonov_regularize': {},
            'strassen_multiply': {},
            'blas_optimized_matmul': {}
        }
        self.current_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def add_metric(self, operation, matrix_size, condition_number=None, execution_time=None, 
                   error=None, algorithm=None, backend=None):
        """Fügt eine Performance-Metrik hinzu."""
        if operation not in self.metrics:
            self.metrics[operation] = {}
            
        # Erstelle einen eindeutigen Schlüssel basierend auf der Matrix-Größe
        key = f"{matrix_size}"
        if condition_number:
            key += f"_cond{condition_number:.1e}"
        
        if key not in self.metrics[operation]:
            self.metrics[operation][key] = []
            
        # Sammeln der Metriken
        metric_data = {
            'timestamp': datetime.now().isoformat(),
            'matrix_size': matrix_size,
            'execution_time': execution_time
        }
        
        # Optionale Metriken
        if condition_number is not None:
            metric_data['condition_number'] = condition_number
        if error is not None:
            metric_data['error'] = float(error)
        if algorithm is not None:
            metric_data['algorithm'] = algorithm
        if backend is not None:
            metric_data['backend'] = backend
            
        self.metrics[operation][key].append(metric_data)
    
    def save_metrics(self):
        """Speichert die gesammelten Metriken in einer JSON-Datei."""
        filename = os.path.join(PROFILING_DIR, f"vx_matrix_metrics_{self.current_run}.json")
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        return filename

# Globale Performance-Metriken-Instanz
performance_metrics = PerformanceMetrics()

def profile_function(func):
    """Decorator zum Profiling einer Funktion mit cProfile."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Profiler initialisieren
        pr = cProfile.Profile()
        pr.enable()
        
        # Startzeit und Ergebnis erfassen
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Profiling beenden
        pr.disable()
        
        # Profiling-Ergebnisse in String-Stream ausgeben
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats(10)  # Limitiere auf Top 10 Funktionen
        
        # Profiling-Ergebnisse in Datei speichern
        filename = f"{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
        filepath = os.path.join(PROFILING_DIR, filename)
        with open(filepath, 'w') as f:
            f.write(s.getvalue())
        
        # Ausgabe für Debugging
        print(f"[PROFILING] {func.__name__}: {execution_time:.6f} Sekunden")
        print(f"[PROFILING] Details in: {filepath}")
        
        return result
    return wrapper

def time_function(func):
    """Decorator zur Zeitmessung einer Funktion ohne volles Profiling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Matrixgröße für Metriken extrahieren (wenn erstes Argument Matrix)
        matrix_size = None
        if args and hasattr(args[0], 'shape'):
            matrix_size = f"{args[0].shape[0]}x{args[0].shape[1]}"
        elif len(args) > 1 and hasattr(args[1], 'shape'):
            # Für Methoden, bei denen self das erste Argument ist
            matrix_size = f"{args[1].shape[0]}x{args[1].shape[1]}"
            
        # Konditionszahl schätzen (wenn möglich)
        condition_number = None
        if args and isinstance(args[0], np.ndarray) and args[0].shape[0] == args[0].shape[1]:
            try:
                condition_number = np.linalg.cond(args[0])
            except:
                pass
        elif len(args) > 1 and isinstance(args[1], np.ndarray) and args[1].shape[0] == args[1].shape[1]:
            try:
                condition_number = np.linalg.cond(args[1])
            except:
                pass
                
        # Performance-Metriken erfassen
        if matrix_size:
            operation = func.__name__
            if operation.startswith('_'):
                operation = operation[1:]  # Private Methodenpräfix entfernen
            performance_metrics.add_metric(
                operation=operation,
                matrix_size=matrix_size,
                condition_number=condition_number,
                execution_time=execution_time
            )
            
        return result
    return wrapper

def compare_performance(original_func, optimized_func, test_matrices, desc=""):
    """Vergleicht die Performance zweier Funktionen mit den gleichen Testmatrizen."""
    results = []
    print(f"\n=== Performance-Vergleich: {desc} ===")
    
    for matrix in test_matrices:
        # Original-Funktion timen
        start = time.time()
        orig_result = original_func(matrix)
        orig_time = time.time() - start
        
        # Optimierte Funktion timen
        start = time.time()
        opt_result = optimized_func(matrix)
        opt_time = time.time() - start
        
        # Fehlerberechnung
        if isinstance(orig_result, np.ndarray) and isinstance(opt_result, np.ndarray):
            error = np.max(np.abs(orig_result - opt_result))
        else:
            error = None
            
        # Speedup berechnen
        speedup = orig_time / opt_time if opt_time > 0 else float('inf')
        
        # Matrix-Größe und Konditionszahl
        matrix_size = f"{matrix.shape[0]}x{matrix.shape[1]}"
        try:
            cond = np.linalg.cond(matrix)
        except:
            cond = None
            
        # Ergebnisse ausgeben
        print(f"Matrix {matrix_size}" + (f", κ={cond:.1e}" if cond else ""))
        print(f"  Original: {orig_time:.6f}s, Optimiert: {opt_time:.6f}s, Speedup: {speedup:.2f}x")
        if error is not None:
            print(f"  Max Fehler: {error:.2e}")
        
        # Ergebnisse sammeln
        results.append({
            'matrix_size': matrix_size,
            'condition_number': cond,
            'original_time': orig_time,
            'optimized_time': opt_time,
            'speedup': speedup,
            'error': error
        })
        
    return results
