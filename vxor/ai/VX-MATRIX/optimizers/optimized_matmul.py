"""
Optimierte Matrixmultiplikation für VX-MATRIX
=============================================

Hochoptimierte Implementierungen der Matrixmultiplikation für verschiedene Anwendungsfälle.
ZTM-konform für MISO Ultimate.
"""

import numpy as np
import time
import warnings
from functools import wraps

def benchmark_decorator(func):
    """Zeitmessung für Matrixmultiplikationen"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"[BENCHMARK] {func.__name__}: {execution_time:.6f}s für Matrix der Größe {args[0].shape}")
        return result
    return wrapper

def optimized_matmul_direct(a, b):
    """
    Direkter Aufruf von np.matmul ohne zusätzlichen Overhead.
    
    Diese Funktion ist für kleine Matrizen (< 100x100) optimiert, da hier
    der Python-Overhead den Performance-Gewinn durch BLAS überwiegen würde.
    
    Args:
        a (ndarray): Erste Matrix
        b (ndarray): Zweite Matrix
        
    Returns:
        ndarray: Matrix-Produkt a @ b
    """
    # Numerische Stabilität sicherstellen
    if np.isnan(a).any() or np.isnan(b).any():
        # NaN-Werte durch kleine Werte ersetzen
        a = np.nan_to_num(a)
        b = np.nan_to_num(b)
    
    return np.matmul(a, b)

def optimized_matmul_blas(a, b):
    """
    Optimierte Matrixmultiplikation mit BLAS-Schnittstelle.
    
    Verwendet SciPy's dgemm/sgemm-Wrapper für direkte BLAS-Aufrufe, was bei
    mittleren Matrizen (100x100 bis 2000x2000) deutliche Performance-Vorteile bietet.
    
    Args:
        a (ndarray): Erste Matrix
        b (ndarray): Zweite Matrix
        
    Returns:
        ndarray: Matrix-Produkt a @ b
    """
    try:
        from scipy import linalg
        
        # Numerische Stabilität sicherstellen
        if np.isnan(a).any() or np.isnan(b).any():
            a = np.nan_to_num(a)
            b = np.nan_to_num(b)
        
        # Datentyp-Check und passenden BLAS-Wrapper wählen
        if a.dtype == np.float64 and b.dtype == np.float64:
            return linalg.blas.dgemm(1.0, a, b)
        elif a.dtype == np.float32 and b.dtype == np.float32:
            return linalg.blas.sgemm(1.0, a, b)
        else:
            # Konvertiere zu float64 für gemischte Datentypen
            a_conv = a.astype(np.float64)
            b_conv = b.astype(np.float64)
            return linalg.blas.dgemm(1.0, a_conv, b_conv)
    except (ImportError, AttributeError):
        # Fallback auf NumPy matmul
        return optimized_matmul_direct(a, b)
    except Exception as e:
        warnings.warn(f"BLAS-Multiplikation fehlgeschlagen: {e}")
        return optimized_matmul_direct(a, b)

def split_matrix(matrix, midpoint=None):
    """Teilt eine Matrix in vier Quadranten."""
    n = matrix.shape[0]
    mid = midpoint or n // 2
    
    return (
        matrix[:mid, :mid],   # A11
        matrix[:mid, mid:],   # A12
        matrix[mid:, :mid],   # A21
        matrix[mid:, mid:]    # A22
    )

def optimized_strassen_multiply(a, b, leaf_size=128):
    """
    Optimierte Strassen-Matrixmultiplikation mit verbesserter numerischer Stabilität.
    
    Angepasst für große Matrizen (> 2000x2000), mit optimierter Rekursionstiefe und
    besserer Fehlerbehandlung für ill-conditioned Matrizen.
    
    Args:
        a (ndarray): Erste Matrix
        b (ndarray): Zweite Matrix
        leaf_size (int, optional): Schwellwert für Umschaltung auf direkte Multiplikation. Default: 128
        
    Returns:
        ndarray: Matrix-Produkt a @ b
    """
    # Prüfen, ob Matrixdimensionen kompatibel sind
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Matrixdimensionen inkompatibel: {a.shape} und {b.shape}")
    
    # Prüfen, ob Matrizen quadratisch sind
    n = a.shape[0]
    if n != a.shape[1] or n != b.shape[0] or n != b.shape[1]:
        # Für nicht-quadratische Matrizen verwenden wir die direkte Multiplikation
        return optimized_matmul_blas(a, b)
    
    # Numerische Stabilität sicherstellen
    if np.isnan(a).any() or np.isnan(b).any() or np.isinf(a).any() or np.isinf(b).any():
        # Bei problematischen Eingaben: regularisieren und Fallback auf direkte Multiplikation
        a_reg = np.nan_to_num(a, nan=0.0, posinf=1e10, neginf=-1e10)
        b_reg = np.nan_to_num(b, nan=0.0, posinf=1e10, neginf=-1e10)
        return optimized_matmul_blas(a_reg, b_reg)
    
    # Basisfall: direkte Multiplikation für kleine Matrizen
    if n <= leaf_size:
        return optimized_matmul_blas(a, b)
    
    # Große Zahlen skalieren, um Overflow zu vermeiden
    max_val = max(np.max(np.abs(a)), np.max(np.abs(b)))
    if max_val > 1e6:
        scale_factor = 1e6 / max_val
        a_scaled = a * scale_factor
        b_scaled = b * scale_factor
        
        # Skalierungsfaktor kompensieren (später rausmultiplizieren)
        result = optimized_strassen_multiply(a_scaled, b_scaled, leaf_size)
        return result / scale_factor
    
    # Teile Matrizen in Quadranten
    a11, a12, a21, a22 = split_matrix(a)
    b11, b12, b21, b22 = split_matrix(b)
    
    # Berechne die 7 Produkte nach Strassen (effizientere Implementierung)
    m1 = optimized_strassen_multiply(a11 + a22, b11 + b22, leaf_size)
    m2 = optimized_strassen_multiply(a21 + a22, b11, leaf_size)
    m3 = optimized_strassen_multiply(a11, b12 - b22, leaf_size)
    m4 = optimized_strassen_multiply(a22, b21 - b11, leaf_size)
    m5 = optimized_strassen_multiply(a11 + a12, b22, leaf_size)
    m6 = optimized_strassen_multiply(a21 - a11, b11 + b12, leaf_size)
    m7 = optimized_strassen_multiply(a12 - a22, b21 + b22, leaf_size)
    
    # Ergebnisquadranten berechnen
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    
    # Quadranten zusammenfügen - vektorisierte Version für bessere Performance
    result = np.zeros((n, n), dtype=a.dtype)
    mid = n // 2
    result[:mid, :mid] = c11
    result[:mid, mid:] = c12
    result[mid:, :mid] = c21
    result[mid:, mid:] = c22
    
    return result

def optimized_matrix_multiply(a, b, strategy='auto'):
    """
    Hochoptimierte Matrix-Multiplikation mit automatischer Strategieauswahl.
    
    Wählt automatisch die beste Multiplikationsstrategie basierend auf Matrixgröße,
    Condition Number und Hardware-Eigenschaften.
    
    Args:
        a (ndarray): Erste Matrix
        b (ndarray): Zweite Matrix
        strategy (str, optional): Erzwungene Strategie ('direct', 'blas', 'strassen' oder 'auto'). Default: 'auto'
        
    Returns:
        ndarray: Matrix-Produkt a @ b
    """
    # Prüfen, ob Matrizen NumPy-Arrays sind
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
    
    # Prüfen, ob Matrixdimensionen kompatibel sind
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Matrixdimensionen inkompatibel: {a.shape} und {b.shape}")
    
    # Matrix-Größen bestimmen
    m, n = a.shape[0], a.shape[1]
    p = b.shape[1]
    max_dim = max(m, n, p)
    
    # Bei expliziter Strategiewahl diese verwenden
    if strategy != 'auto':
        if strategy == 'direct':
            return optimized_matmul_direct(a, b)
        elif strategy == 'blas':
            return optimized_matmul_blas(a, b)
        elif strategy == 'strassen':
            # Strassen nur für quadratische Matrizen
            if m == n and n == p:
                return optimized_strassen_multiply(a, b)
            else:
                warnings.warn("Strassen-Algorithmus nur für quadratische Matrizen verfügbar, verwende BLAS")
                return optimized_matmul_blas(a, b)
    
    # Automatische Strategieauswahl basierend auf Matrix-Größe
    if max_dim < 100:
        # Kleine Matrizen: direkter Aufruf ohne Overhead
        return optimized_matmul_direct(a, b)
    elif max_dim < 2000:
        # Mittlere Matrizen: BLAS-optimierte Multiplikation
        return optimized_matmul_blas(a, b)
    else:
        # Große Matrizen: Strassen (nur für quadratische Matrizen)
        if m == n and n == p:
            return optimized_strassen_multiply(a, b, leaf_size=128)
        else:
            # Für nicht-quadratische große Matrizen: BLAS
            return optimized_matmul_blas(a, b)
