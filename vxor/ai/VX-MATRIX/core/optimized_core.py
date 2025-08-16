"""
VX-MATRIX: Optimierte Core-Implementierung
==========================================

Diese Klasse erweitert die Basis-MatrixCore mit den hochoptimierten Implementierungen
aus den optimizers-Modulen. Sie bietet verbesserte Performance für Matrixoperationen
bei Beibehaltung der vollständigen Kompatibilität mit der ursprünglichen API.

ZTM-konform für MISO Ultimate.
"""

import numpy as np
import time
import os
import sys
import warnings
from functools import wraps

# Import der Basis-MatrixCore
from .matrix_core import MatrixCore, ztm_log, TensorType

# Import der optimierten Implementierungen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from optimizers.optimized_matmul import optimized_matrix_multiply
    from optimizers.optimized_inverse import (
        optimized_matrix_inverse, 
        optimized_cholesky_inverse,
        optimized_tikhonov_regularize,
        is_spd
    )
    OPTIMIZERS_AVAILABLE = True
except ImportError:
    OPTIMIZERS_AVAILABLE = False
    ztm_log("Optimierte Matrixoperation-Module nicht verfügbar. Verwende Basis-Implementierung.", level="WARNING")

# Import der Profiling-Utilities, falls verfügbar
try:
    from ..utils.profiling import profile_function, time_function, performance_metrics
    PROFILING_ENABLED = True
except ImportError:
    # Dummy-Dekoratoren, falls Profiling nicht verfügbar
    def profile_function(func): return func
    def time_function(func): return func
    PROFILING_ENABLED = False


class OptimizedMatrixCore(MatrixCore):
    """
    Erweiterte MatrixCore-Klasse mit hochoptimierten Implementierungen.
    
    Diese Klasse überschreibt die kritischen Matrix-Operationen mit den optimierten
    Versionen aus den optimizers-Modulen, während die volle Abwärtskompatibilität
    mit der ursprünglichen MatrixCore-API beibehalten wird.
    """
    
    def __init__(self, preferred_backend="auto", available_backends=None):
        """
        Initialisiert den optimierten MatrixCore.
        
        Args:
            preferred_backend (str, optional): Bevorzugtes Backend ('numpy', 'mlx' oder 'auto')
            available_backends (list, optional): Liste der verfügbaren Backends
        """
        super().__init__(preferred_backend, available_backends)
        
        # Optimierte Schwellenwerte basierend auf Benchmarks
        self.small_matrix_threshold = 100  # Optimierter Schwellwert für kleine Matrizen
        self.medium_matrix_threshold = 500  # Optimierter Schwellwert für mittlere Matrizen
        self.large_matrix_threshold = 2000  # Optimierter Schwellwert für große Matrizen (Strassen)
        self.ill_condition_threshold = 1e8  # Optimierter Schwellwert für Konditionszahl
        
        # Status der Optimierungen
        self.optimization_status = {
            'matmul_optimization': OPTIMIZERS_AVAILABLE,
            'inverse_optimization': OPTIMIZERS_AVAILABLE,
            'profiling_enabled': PROFILING_ENABLED
        }
        
        ztm_log(f"VX-MATRIX Optimierte Core-Implementierung initialisiert. Optimierungen aktiv: {OPTIMIZERS_AVAILABLE}", level="INFO")
    
    @profile_function
    def matrix_multiply(self, a, b):
        """
        Hochoptimierte Matrixmultiplikation mit automatischer Strategiewahl.
        
        Verwendet die optimierte Implementierung, wenn verfügbar, mit verbesserter
        numerischer Stabilität und adaptiver Algorithmuswahl basierend auf Matrixgröße.
        
        Args:
            a: Erste Matrix
            b: Zweite Matrix
            
        Returns:
            Matrix-Produkt a·b
        """
        self.op_counter['matrix_multiply'] += 1
        shape = (getattr(a, 'shape', (None,))[0], getattr(b, 'shape', (None,))[1])
        start = time.time()
        
        # Wenn optimierte Implementierung nicht verfügbar ist, verwende die Basis-Implementierung
        if not OPTIMIZERS_AVAILABLE:
            result = super().matrix_multiply(a, b)
            end = time.time()
            self.timing_stats['matrix_mult'].append((shape, end-start, 'original'))
            return result
        
        # MLX Backend-Prüfung und -Verwendung
        use_mlx = ('mlx' in self.backend_modules and
                  shape[0] > self.small_matrix_threshold and
                  self.preferred_backend == 'mlx')
        
        if use_mlx:
            # MLX-Implementierung verwenden
            mx = self.backend_modules['mlx']
            a_mlx = self.tensor_converter.to_mlx(a)
            b_mlx = self.tensor_converter.to_mlx(b)
            
            # JIT-Kompilierung, wenn aktiviert
            if self.enable_jit:
                try:
                    if 'matmul' not in self._jit_functions:
                        self._jit_functions['matmul'] = mx.compile(lambda x, y: mx.matmul(x, y))
                    result = self._jit_functions['matmul'](a_mlx, b_mlx)
                except Exception as e:
                    ztm_log(f"JIT-MatMul-Fehler: {e}, Fallback auf direkten MLX-Call", level="WARNING")
                    result = mx.matmul(a_mlx, b_mlx)
            else:
                result = mx.matmul(a_mlx, b_mlx)
                
            result = self._ensure_numerical_stability(result, TensorType.MLX)
            selected_backend = 'mlx'
        else:
            # NumPy-Arrays sicherstellen
            if not isinstance(a, np.ndarray):
                a = np.array(a, dtype=np.float64)
            if not isinstance(b, np.ndarray):
                b = np.array(b, dtype=np.float64)
            
            # Numerische Vorbehandlung
            a = self._ensure_numerical_stability(a, TensorType.NUMPY)
            b = self._ensure_numerical_stability(b, TensorType.NUMPY)
            
            # Optimierte Multiplikation ausführen
            result = optimized_matrix_multiply(a, b, strategy='auto')
            selected_backend = 'numpy_optimized'
        
        end = time.time()
        self.timing_stats['matrix_mult'].append((shape, end-start, selected_backend))
        
        # Performance-Metriken sammeln, wenn Profiling aktiviert ist
        if PROFILING_ENABLED:
            # Konditionszahl schätzen, wenn möglich
            condition_number = None
            if isinstance(a, np.ndarray) and a.shape[0] == a.shape[1]:
                try:
                    condition_number = np.linalg.cond(a)
                except:
                    pass
            
            operation = 'matrix_multiply'
            matrix_size = f"{shape[0]}x{shape[1]}" if shape[0] and shape[1] else "unknown"
            
            performance_metrics.add_metric(
                operation=operation,
                matrix_size=matrix_size,
                condition_number=condition_number,
                execution_time=end-start,
                algorithm=selected_backend
            )
        
        return result
    
    @profile_function
    def matrix_inverse(self, matrix):
        """
        Hochoptimierte Matrixinversion mit adaptiver Algorithmuswahl.
        
        Verwendet die optimierte Implementierung mit verbesserter numerischer Stabilität,
        SPD-Erkennung und Tikhonov-Regularisierung für schlecht konditionierte Matrizen.
        
        Args:
            matrix: Zu invertierende Matrix
            
        Returns:
            Inverse der Matrix
        """
        self.op_counter['matrix_inverse'] += 1
        shape = getattr(matrix, 'shape', None)
        start = time.time()
        
        # Wenn optimierte Implementierung nicht verfügbar ist, verwende die Basis-Implementierung
        if not OPTIMIZERS_AVAILABLE:
            result = super().matrix_inverse(matrix)
            end = time.time()
            self.timing_stats['inverse'].append((shape, end-start, 'original'))
            return result
        
        # MLX Backend-Prüfung und -Verwendung
        use_mlx = ('mlx' in self.backend_modules and
                  shape and shape[0] > self.medium_matrix_threshold and
                  self.preferred_backend == 'mlx')
        
        if use_mlx:
            # MLX-Implementierung verwenden
            try:
                result = self._optimized_mlx_inverse(matrix)
                selected_backend = 'mlx'
            except Exception as e:
                ztm_log(f"MLX-Inversion fehlgeschlagen: {e}. Fallback auf NumPy.", level="WARNING")
                # Fallback auf NumPy-Optimierung
                if not isinstance(matrix, np.ndarray):
                    matrix_np = np.array(matrix, dtype=np.float64)
                else:
                    matrix_np = matrix
                
                matrix_np = self._ensure_numerical_stability(matrix_np, TensorType.NUMPY)
                result = optimized_matrix_inverse(matrix_np)
                selected_backend = 'numpy_optimized'
        else:
            # NumPy-Array sicherstellen
            if not isinstance(matrix, np.ndarray):
                matrix_np = np.array(matrix, dtype=np.float64)
            else:
                matrix_np = matrix
            
            # Numerische Vorbehandlung
            matrix_np = self._ensure_numerical_stability(matrix_np, TensorType.NUMPY)
            
            # Optimierte Inversion ausführen
            result = optimized_matrix_inverse(matrix_np)
            selected_backend = 'numpy_optimized'
        
        end = time.time()
        self.timing_stats['inverse'].append((shape, end-start, selected_backend))
        
        # Performance-Metriken sammeln, wenn Profiling aktiviert ist
        if PROFILING_ENABLED:
            # Konditionszahl schätzen, wenn möglich
            condition_number = None
            if isinstance(matrix, np.ndarray):
                try:
                    condition_number = np.linalg.cond(matrix)
                except:
                    pass
            
            operation = 'matrix_inverse'
            matrix_size = f"{shape[0]}x{shape[1]}" if shape else "unknown"
            
            performance_metrics.add_metric(
                operation=operation,
                matrix_size=matrix_size,
                condition_number=condition_number,
                execution_time=end-start,
                algorithm=selected_backend
            )
        
        return result
    
    @time_function
    def _is_spd(self, matrix, tol=1e-8):
        """
        Optimierte Prüfung, ob eine Matrix symmetrisch positiv-definit ist.
        
        Verwendet die effizientere Implementierung aus dem optimizers-Modul.
        
        Args:
            matrix: Zu prüfende Matrix
            tol: Toleranzschwelle für Rundungsfehler
            
        Returns:
            bool: True, wenn die Matrix SPD ist
        """
        if OPTIMIZERS_AVAILABLE:
            return is_spd(matrix, tol)
        else:
            return super()._is_spd(matrix, tol)
    
    @time_function
    def _cholesky_inverse(self, matrix):
        """
        Optimierte Cholesky-basierte Inversion für SPD-Matrizen.
        
        Verwendet die verbesserte Implementierung aus dem optimizers-Modul.
        
        Args:
            matrix: SPD-Matrix
            
        Returns:
            Inverse der Matrix
        """
        if OPTIMIZERS_AVAILABLE:
            try:
                return optimized_cholesky_inverse(matrix)
            except Exception as e:
                ztm_log(f"Optimierte Cholesky-Inversion fehlgeschlagen: {e}. Fallback auf Basis-Implementierung.", level="WARNING")
                return super()._cholesky_inverse(matrix)
        else:
            return super()._cholesky_inverse(matrix)
    
    @time_function
    def _tikhonov_regularize(self, matrix, mu=None):
        """
        Optimierte Tikhonov-Regularisierung für schlecht konditionierte Matrizen.
        
        Verwendet die verbesserte adaptive Implementierung aus dem optimizers-Modul.
        
        Args:
            matrix: Zu regularisierende Matrix
            mu: Regularisierungsparameter (wenn None, wird adaptiv bestimmt)
            
        Returns:
            Regularisierte Matrix
        """
        if OPTIMIZERS_AVAILABLE:
            return optimized_tikhonov_regularize(matrix, mu, adaptive=True)
        else:
            return super()._tikhonov_regularize(matrix, mu)
    
    def export_performance_metrics(self, filepath=None):
        """
        Exportiert erweiterte Performance-Metriken.
        
        Umfasst die Basis-Metriken sowie Profiling-Daten, wenn verfügbar.
        
        Args:
            filepath: Pfad für JSON-Ausgabe (wenn None, wird nur Dictionary zurückgegeben)
            
        Returns:
            dict: Performance-Metriken
        """
        # Basis-Metriken von der Elternklasse
        metrics = super().export_performance_metrics(filepath)
        
        # Erweitere um Optimierungsstatus
        metrics['optimization_status'] = self.optimization_status
        
        # Speichere die erweiterten Metriken
        if filepath:
            import json
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        return metrics
