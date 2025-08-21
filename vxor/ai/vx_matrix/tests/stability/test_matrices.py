#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-MATRIX Stabilitätstest: Robuste Testmatrizen

ZTM-Level: STRICT
Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.

Dieses Modul bietet Funktionen zur Erzeugung verschiedener Arten von Testmatrizen:
- Symmetrisch positiv-definite (SPD) Matrizen: Exzellente numerische Stabilität
- Diagonal-dominante Matrizen: Garantiert invertierbar
- Orthonormale Matrizen: Perfekt konditioniert (cond(A) = 1)
- Hilbert-Matrizen: Extrem schlecht konditioniert für Stabilitätstests
"""

import numpy as np
import scipy.linalg
from sklearn.datasets import make_spd_matrix
import pytest


@pytest.fixture(scope="module")
def matrix_dims():
    """Verschiedene Matrixdimensionen für Tests"""
    return [10, 50, 100, 200, 500]


class MatrixGenerators:
    """Sammlung von Funktionen zur Erzeugung verschiedener Arten von Testmatrizen"""
    
    @staticmethod
    def generate_spd_matrix(n, random_state=42):
        """
        Erzeugt eine symmetrisch positiv-definite (SPD) Matrix
        
        SPD-Matrizen haben ausschließlich positive Eigenwerte und sind garantiert
        invertierbar mit guter Stabilität.
        """
        return make_spd_matrix(n_dim=n, random_state=random_state)
    
    @staticmethod
    def generate_diag_dominant_matrix(n, random_state=42):
        """
        Erzeugt eine diagonal-dominante Matrix
        
        Eine Matrix ist diagonal-dominant, wenn der Betrag jedes Diagonaleintrags
        größer oder gleich der Summe der Beträge aller anderen Einträge in der gleichen
        Zeile ist. Solche Matrizen sind garantiert invertierbar.
        """
        np.random.seed(random_state)
        A = np.random.rand(n, n) * 2 - 1  # Werte zwischen -1 und 1
        
        # Mache die Matrix streng diagonal-dominant
        row_sums = np.sum(np.abs(A), axis=1)
        for i in range(n):
            # Setze Diagonalelement auf Zeilensumme + kleinen Offset
            diagonal_value = row_sums[i] + 0.1
            A[i, i] = diagonal_value
            
        return A
    
    @staticmethod
    def generate_orthonormal_matrix(n, random_state=42):
        """
        Erzeugt eine orthonormale Matrix mittels QR-Zerlegung
        
        Orthonormale Matrizen haben eine Konditionszahl von exakt 1 und sind
        somit numerisch optimal stabil. Die Inverse einer orthonormalen Matrix
        ist einfach ihre Transponierte.
        """
        np.random.seed(random_state)
        A = np.random.randn(n, n)
        Q, _ = np.linalg.qr(A)  # Q ist orthonormal
        return Q
    
    @staticmethod
    def generate_hilbert_matrix(n):
        """
        Erzeugt eine Hilbert-Matrix - berüchtigt für ihre schlechte Konditionierung
        
        Hilbert-Matrizen haben die Einträge a_ij = 1/(i+j-1) und sind extrem
        schlecht konditioniert, was sie zu einem strengen Test für numerische
        Stabilitätsalgorithmen macht.
        """
        return scipy.linalg.hilbert(n)


@pytest.fixture(params=["spd", "diag_dominant", "orthonormal", "hilbert"])
def matrix_type(request):
    """Parametrisierte Fixture für verschiedene Matrixtypen"""
    return request.param


@pytest.fixture
def get_test_matrix():
    """
    Factory-Fixture zur Erzeugung von Testmatrizen
    
    Verwendung in Tests:
    ```
    def test_something(get_test_matrix, matrix_type, matrix_dims):
        dim = 100
        matrix = get_test_matrix(matrix_type, dim)
        # Test mit dieser Matrix...
    ```
    """
    def _get_matrix(matrix_type, n):
        if matrix_type == "spd":
            return MatrixGenerators.generate_spd_matrix(n)
        elif matrix_type == "diag_dominant":
            return MatrixGenerators.generate_diag_dominant_matrix(n)
        elif matrix_type == "orthonormal":
            return MatrixGenerators.generate_orthonormal_matrix(n)
        elif matrix_type == "hilbert":
            return MatrixGenerators.generate_hilbert_matrix(n)
        else:
            raise ValueError(f"Unbekannter Matrix-Typ: {matrix_type}")
    
    return _get_matrix


def get_matrix_stats(matrix):
    """Berechnet wichtige statistische Eigenschaften einer Matrix"""
    cond = np.linalg.cond(matrix)
    det = np.linalg.det(matrix)
    has_nan = np.isnan(matrix).any()
    has_inf = np.isinf(matrix).any()
    min_abs = np.min(np.abs(matrix[matrix != 0])) if np.any(matrix != 0) else 0
    max_abs = np.max(np.abs(matrix))
    
    return {
        "condition": cond,
        "determinant": det,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min_abs": min_abs,
        "max_abs": max_abs
    }


if __name__ == "__main__":
    # Demo zur Erzeugung und Analyse verschiedener Matrixtypen
    dims = [10, 50, 100]
    
    print("Matrix-Eigenschaftsanalyse:")
    for dim in dims:
        print(f"\nDimension: {dim}x{dim}")
        
        for matrix_type in ["spd", "diag_dominant", "orthonormal", "hilbert"]:
            matrix = get_test_matrix()(matrix_type, dim)
            stats = get_matrix_stats(matrix)
            
            print(f"  {matrix_type.upper():-<15} Cond: {stats['condition']:.2e}, "
                  f"Det: {stats['determinant']:.2e}, "
                  f"Min(abs): {stats['min_abs']:.2e}, "
                  f"Max(abs): {stats['max_abs']:.2e}")
