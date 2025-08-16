"""
Tensor-Operations für MISO

Implementiert Tensoroperationen für MISO.
ZTM-Verifiziert: Dieses Modul unterliegt der ZTM-Policy und wird entsprechend überwacht.
"""

import logging
import numpy as np
from typing import List, Tuple, Union, Optional, Any, Dict

# Konfiguration des Loggings
logger = logging.getLogger("miso.math.tensor_operations")

class TensorOperations:
    """
    Implementiert Tensoroperationen für MISO.
    """
    
    def __init__(self, precision: int = 6):
        """
        Initialisiert die TensorOperations.
        
        Args:
            precision: Präzision für Berechnungen
        """
        self.precision = precision
        logger.info(f"[ZTM VERIFIED] TensorOperations initialisiert (Präzision: {precision})")
    
    def add(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """
        Addiert zwei Tensoren.
        
        Args:
            t1: Erster Tensor
            t2: Zweiter Tensor
            
        Returns:
            np.ndarray: Summententor
        """
        if t1.shape != t2.shape:
            logger.error(f"Tensoren haben unterschiedliche Dimensionen: {t1.shape} und {t2.shape}")
            raise ValueError("Tensoren müssen dieselbe Dimension haben")
        
        result = t1 + t2
        return np.round(result, self.precision)
    
    def subtract(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """
        Subtrahiert zwei Tensoren.
        
        Args:
            t1: Erster Tensor
            t2: Zweiter Tensor
            
        Returns:
            np.ndarray: Differenztensor
        """
        if t1.shape != t2.shape:
            logger.error(f"Tensoren haben unterschiedliche Dimensionen: {t1.shape} und {t2.shape}")
            raise ValueError("Tensoren müssen dieselbe Dimension haben")
        
        result = t1 - t2
        return np.round(result, self.precision)
    
    def scalar_multiply(self, scalar: float, tensor: np.ndarray) -> np.ndarray:
        """
        Multipliziert einen Tensor mit einem Skalar.
        
        Args:
            scalar: Skalar
            tensor: Tensor
            
        Returns:
            np.ndarray: Skalierter Tensor
        """
        result = scalar * tensor
        return np.round(result, self.precision)
    
    def tensor_product(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """
        Berechnet das Tensorprodukt zweier Tensoren.
        
        Args:
            t1: Erster Tensor
            t2: Zweiter Tensor
            
        Returns:
            np.ndarray: Tensorprodukt
        """
        result = np.tensordot(t1, t2, axes=0)
        return np.round(result, self.precision)
    
    def contract(self, tensor: np.ndarray, axis1: int, axis2: int) -> np.ndarray:
        """
        Kontrahiert einen Tensor entlang zweier Achsen.
        
        Args:
            tensor: Tensor
            axis1: Erste Achse
            axis2: Zweite Achse
            
        Returns:
            np.ndarray: Kontrahierter Tensor
        """
        if axis1 == axis2:
            logger.error(f"Achsen für Kontraktion müssen unterschiedlich sein: {axis1} und {axis2}")
            raise ValueError("Achsen für Kontraktion müssen unterschiedlich sein")
        
        # Stelle sicher, dass axis1 < axis2
        if axis1 > axis2:
            axis1, axis2 = axis2, axis1
        
        # Anpassen von axis2, da sich die Indizes nach der ersten Kontraktion verschieben
        axis2 -= 1
        
        result = np.trace(tensor, axis1=axis1, axis2=axis2)
        return np.round(result, self.precision)
    
    def transpose(self, tensor: np.ndarray, axes: Optional[List[int]] = None) -> np.ndarray:
        """
        Transponiert einen Tensor.
        
        Args:
            tensor: Tensor
            axes: Permutation der Achsen (optional)
            
        Returns:
            np.ndarray: Transponierter Tensor
        """
        result = np.transpose(tensor, axes)
        return np.round(result, self.precision)
    
    def reshape(self, tensor: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Ändert die Form eines Tensors.
        
        Args:
            tensor: Tensor
            shape: Neue Form
            
        Returns:
            np.ndarray: Umgeformter Tensor
        """
        try:
            result = np.reshape(tensor, shape)
            return np.round(result, self.precision)
        except ValueError as e:
            logger.error(f"Fehler beim Umformen des Tensors: {e}")
            raise ValueError(f"Fehler beim Umformen des Tensors: {e}")
    
    def norm(self, tensor: np.ndarray) -> float:
        """
        Berechnet die Norm eines Tensors.
        
        Args:
            tensor: Tensor
            
        Returns:
            float: Norm
        """
        # Frobenius-Norm
        result = np.sqrt(np.sum(np.abs(tensor) ** 2))
        return round(float(result), self.precision)
    
    def inner_product(self, t1: np.ndarray, t2: np.ndarray) -> float:
        """
        Berechnet das innere Produkt zweier Tensoren.
        
        Args:
            t1: Erster Tensor
            t2: Zweiter Tensor
            
        Returns:
            float: Inneres Produkt
        """
        if t1.shape != t2.shape:
            logger.error(f"Tensoren haben unterschiedliche Dimensionen: {t1.shape} und {t2.shape}")
            raise ValueError("Tensoren müssen dieselbe Dimension haben")
        
        result = np.sum(t1 * t2)
        return round(float(result), self.precision)
    
    def outer_product(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """
        Berechnet das äußere Produkt zweier Tensoren.
        
        Args:
            t1: Erster Tensor
            t2: Zweiter Tensor
            
        Returns:
            np.ndarray: Äußeres Produkt
        """
        result = np.outer(t1.flatten(), t2.flatten()).reshape((*t1.shape, *t2.shape))
        return np.round(result, self.precision)
    
    def eigendecomposition(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Führt eine Eigendekomposition eines Tensors durch.
        
        Args:
            tensor: Tensor (muss eine quadratische Matrix sein)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Eigenwerte und Eigenvektoren
        """
        if len(tensor.shape) != 2 or tensor.shape[0] != tensor.shape[1]:
            logger.error(f"Eigendekomposition nur für quadratische Matrizen definiert, erhalten: {tensor.shape}")
            raise ValueError("Eigendekomposition nur für quadratische Matrizen definiert")
        
        try:
            eigenvalues, eigenvectors = np.linalg.eig(tensor)
            return np.round(eigenvalues, self.precision), np.round(eigenvectors, self.precision)
        except np.linalg.LinAlgError as e:
            logger.error(f"Fehler bei Eigendekomposition: {e}")
            raise ValueError(f"Fehler bei Eigendekomposition: {e}")
    
    def svd(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Führt eine Singulärwertzerlegung eines Tensors durch.
        
        Args:
            tensor: Tensor (muss eine Matrix sein)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: U, S, V
        """
        if len(tensor.shape) != 2:
            logger.error(f"SVD nur für Matrizen definiert, erhalten: {tensor.shape}")
            raise ValueError("SVD nur für Matrizen definiert")
        
        try:
            U, S, V = np.linalg.svd(tensor)
            return np.round(U, self.precision), np.round(S, self.precision), np.round(V, self.precision)
        except np.linalg.LinAlgError as e:
            logger.error(f"Fehler bei SVD: {e}")
            raise ValueError(f"Fehler bei SVD: {e}")
    
    def to_list(self, tensor: np.ndarray) -> List:
        """
        Konvertiert einen Tensor in eine verschachtelte Liste.
        
        Args:
            tensor: Tensor
            
        Returns:
            List: Verschachtelte Liste
        """
        return tensor.tolist()
    
    def from_list(self, data: List) -> np.ndarray:
        """
        Konvertiert eine verschachtelte Liste in einen Tensor.
        
        Args:
            data: Verschachtelte Liste
            
        Returns:
            np.ndarray: Tensor
        """
        return np.array(data, dtype=np.float64)
