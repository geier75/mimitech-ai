"""
Optimierte Matrixinversion für VX-MATRIX
========================================

Hochoptimierte Implementierungen der Matrixinversion für verschiedene Matrixtypen.
Besonders fokussiert auf verbesserte Performance für SPD-Matrizen und stabile Invertierung
schlecht konditionierter Matrizen.

ZTM-konform für MISO Ultimate.
"""

import numpy as np
import time
import warnings
from functools import wraps
from scipy import linalg

def benchmark_decorator(func):
    """Zeitmessung für Matrixoperationen"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"[BENCHMARK] {func.__name__}: {execution_time:.6f}s für Matrix der Größe {args[0].shape}")
        return result
    return wrapper

def is_symmetric(matrix, tol=1e-8):
    """
    Prüft, ob eine Matrix symmetrisch ist.
    
    Args:
        matrix (ndarray): Zu prüfende Matrix
        tol (float, optional): Toleranz für Rundungsfehler
        
    Returns:
        bool: True, wenn die Matrix symmetrisch ist
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    return np.allclose(matrix, matrix.T, rtol=tol, atol=tol)

def is_spd(matrix, tol=1e-8):
    """
    Optimierte Prüfung, ob eine Matrix symmetrisch positiv-definit (SPD) ist.
    
    Verwendet einen zweistufigen Ansatz mit Symmetrieprüfung und Cholesky-Zerlegung
    statt der teuren Eigenwertberechnung.
    
    Args:
        matrix (ndarray): Zu prüfende Matrix
        tol (float, optional): Toleranz für Rundungsfehler
        
    Returns:
        bool: True, wenn die Matrix SPD ist
    """
    # Prüfe zuerst auf Symmetrie
    if not is_symmetric(matrix, tol):
        return False
    
    # Prüfe auf positive Definitheit mit Cholesky-Zerlegung (effizienter als Eigenwertberechnung)
    try:
        linalg.cholesky(matrix)
        return True
    except linalg.LinAlgError:
        return False

def optimized_cholesky_inverse(matrix):
    """
    Hochperformante Matrixinversion für SPD-Matrizen mittels Cholesky-Zerlegung.
    
    Verwendet SciPy's optimierte Implementierung mit direkter LAPACK-Anbindung
    für maximale Performance.
    
    Args:
        matrix (ndarray): Symmetrisch positiv-definite Matrix
        
    Returns:
        ndarray: Inverse der Matrix
        
    Raises:
        LinAlgError: Falls die Cholesky-Zerlegung fehlschlägt
    """
    try:
        # Direkte Verwendung von SciPy's cho_factor und cho_solve
        # Dies vermeidet Python-Overhead und nutzt optimierte LAPACK-Routinen
        c_factor = linalg.cho_factor(matrix)
        n = matrix.shape[0]
        
        # Löse das System für die Einheitsmatrix
        identity = np.eye(n)
        inverse = linalg.cho_solve(c_factor, identity)
        
        return inverse
    except linalg.LinAlgError as e:
        # Bei Fehlern der Cholesky-Zerlegung: Ausnahme weiterleiten
        raise e
    except Exception as e:
        # Bei unerwarteten Fehlern: Fallback auf Standard-Inversion
        warnings.warn(f"Optimierte Cholesky-Inversion fehlgeschlagen: {e}. Fallback auf standard Inversion.")
        return np.linalg.inv(matrix)

def optimized_tikhonov_regularize(matrix, mu=None, adaptive=True):
    """
    Verbesserte adaptive Tikhonov-Regularisierung für Matrixinversion.
    
    Diese Implementierung bestimmt den Regularisierungsparameter μ adaptiv
    basierend auf Konditionszahl, Maschinengenauigkeit und Matrixnorm,
    mit verbesserten Heuristiken für optimale Konditionierung.
    
    Args:
        matrix (ndarray): Zu regularisierende Matrix
        mu (float, optional): Expliziter Regularisierungsparameter
        adaptive (bool, optional): Adaptive Bestimmung von μ aktivieren
        
    Returns:
        ndarray: Regularisierte Matrix
    """
    n = matrix.shape[0]
    
    if mu is None and adaptive:
        # Verbesserte adaptive Bestimmung des Regularisierungsparameters
        try:
            # Konditionszahl mittels Singulärwertzerlegung (genauer als direkte Konditionszahl)
            s = linalg.svdvals(matrix)
            condition = s[0] / s[-1] if s[-1] > 1e-15 else float('inf')
            
            # Matrixnorm (Spektralnorm = größter Singulärwert)
            matrix_norm = s[0]
            
            # Maschinengenauigkeit
            eps_machine = np.finfo(float).eps
            
            # Adaptive Berechnung basierend auf Konditionszahl
            if condition < 1e3:
                # Gut konditioniert: minimaler Regularisierungsterm
                mu = eps_machine * matrix_norm
            elif condition < 1e6:
                # Mäßig konditioniert: moderater Regularisierungsterm
                mu = eps_machine * matrix_norm * np.sqrt(condition) * 0.1
            elif condition < 1e10:
                # Schlecht konditioniert: stärkerer Regularisierungsterm
                mu = eps_machine * matrix_norm * condition * 0.01
            else:
                # Extrem schlecht konditioniert: maximaler Regularisierungsterm
                mu = max(1e-6, eps_machine * matrix_norm)
        except Exception:
            # Fallback bei Fehlern in der Konditionsschätzung
            mu = 1e-6
    elif mu is None:
        # Standard-Regularisierungsparameter, wenn nicht adaptiv
        mu = 1e-6
    
    # Optimierte Anwendung der Regularisierung
    regularized = matrix.copy()
    
    # Effiziente Implementierung für die Addition zur Diagonalen
    diag_indices = np.diag_indices(n)
    regularized[diag_indices] += mu
    
    return regularized

def optimized_matrix_inverse(matrix, method='auto', regularization=True):
    """
    Hochperformante Matrixinversion mit automatischer Methodenauswahl.
    
    Wählt basierend auf Matrixeigenschaften die optimale Inversionsmethode:
    - Cholesky-Zerlegung für SPD-Matrizen
    - Tikhonov-Regularisierung für schlecht konditionierte Matrizen
    - SVD-basierte Inversion für nahezu singuläre Matrizen
    - Standard-Inversion für gut konditionierte Matrizen
    
    Args:
        matrix (ndarray): Zu invertierende Matrix
        method (str, optional): Erzwungene Methode ('cholesky', 'svd', 'standard' oder 'auto')
        regularization (bool, optional): Tikhonov-Regularisierung aktivieren
        
    Returns:
        ndarray: Inverse der Matrix
    """
    # Sicherstellen, dass die Matrix ein NumPy-Array ist
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=np.float64)
    
    # Numerische Stabilität sicherstellen
    if np.isnan(matrix).any() or np.isinf(matrix).any():
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Prüfen, ob die Matrix quadratisch ist
    n = matrix.shape[0]
    if n != matrix.shape[1]:
        raise ValueError(f"Matrix muss quadratisch sein, aber hat Form {matrix.shape}")
    
    # Bei Methodenwahl: Wunschmethode verwenden
    if method != 'auto':
        if method == 'cholesky':
            if is_spd(matrix):
                return optimized_cholesky_inverse(matrix)
            else:
                warnings.warn("Matrix ist nicht SPD, Fallback auf Standard-Inversion")
                return np.linalg.inv(matrix)
        elif method == 'svd':
            # SVD-basierte Pseudoinverse (robust, aber langsamer)
            u, s, vh = linalg.svd(matrix)
            # Numerische Stabilität für kleine Singulärwerte
            rcond = np.finfo(s.dtype).eps * max(n, matrix.shape[1])
            cutoff = np.max(s) * rcond
            s_inv = np.zeros_like(s)
            s_inv[s > cutoff] = 1.0 / s[s > cutoff]
            return vh.T @ np.diag(s_inv) @ u.T
        elif method == 'standard':
            # Standard-Inversion
            return np.linalg.inv(matrix)
    
    # Automatische Methodenauswahl
    # 1. Prüfen, ob SPD - dann Cholesky verwenden
    if is_spd(matrix):
        try:
            return optimized_cholesky_inverse(matrix)
        except Exception:
            # Bei Fehlern: Weiter mit anderen Methoden
            pass
    
    # 2. Konditionszahl prüfen
    try:
        # Effiziente Konditionszahlschätzung über SVD
        s = linalg.svdvals(matrix)
        condition = s[0] / s[-1] if s[-1] > 1e-15 else float('inf')
        
        # Regularisierung für schlecht konditionierte Matrizen
        if condition > 1e6 and regularization:
            # Anwenden der Tikhonov-Regularisierung
            regularized = optimized_tikhonov_regularize(matrix, adaptive=True)
            # Standardinversion der regularisierten Matrix
            return np.linalg.inv(regularized)
        elif condition > 1e12:
            # Extreme Konditionszahl: SVD-basierte Pseudoinverse
            u, s, vh = linalg.svd(matrix)
            # Numerische Stabilität für kleine Singulärwerte
            rcond = np.finfo(s.dtype).eps * max(n, matrix.shape[1])
            cutoff = np.max(s) * rcond
            s_inv = np.zeros_like(s)
            s_inv[s > cutoff] = 1.0 / s[s > cutoff]
            return vh.T @ np.diag(s_inv) @ u.T
    except Exception:
        # Bei Fehlern in der Konditionsschätzung: Weiter mit Standard-Inversion
        pass
    
    # 3. Standard-Inversion als Fallback
    return np.linalg.inv(matrix)
