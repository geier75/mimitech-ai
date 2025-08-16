#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION NumPy-Kernels

Bildverarbeitungs-Kernels mit NumPy für maximale Kompatibilität.
Diese dienen als Fallback, wenn MLX oder PyTorch nicht verfügbar sind.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import math
from scipy import ndimage, signal

from vxor.vision.kernels.common import (
    KernelOperation, KernelType, register_kernel
)

# Logger konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VX-VISION.kernels.numpy")

# Versuche, SciPy zu importieren für erweiterte Bildverarbeitung
try:
    from scipy import ndimage
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("SciPy nicht verfügbar. Grundlegende NumPy-Implementierungen werden verwendet.")

# Hier beginnen die eigentlichen Kernel-Implementierungen

@register_kernel(KernelOperation.RESIZE, KernelType.NUMPY)
def numpy_resize(images: List[np.ndarray], width: int = None, height: int = None, scale: float = None) -> List[np.ndarray]:
    """
    Ändert die Größe einer Liste von Bildern mit NumPy.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        width: Zielbreite (wenn None, wird über scale berechnet)
        height: Zielhöhe (wenn None, wird über scale berechnet)
        scale: Skalierungsfaktor (wenn width und height None sind)
        
    Returns:
        Liste der größengeänderten Bilder
    """
    result = []
    
    for img in images:
        # Originalgröße bestimmen
        orig_h, orig_w = img.shape[:2]
        
        # Zielgröße berechnen
        if width is None or height is None:
            if scale is not None:
                target_w = int(orig_w * scale)
                target_h = int(orig_h * scale)
            else:
                target_w = width if width is not None else orig_w
                target_h = height if height is not None else orig_h
        else:
            target_w = width
            target_h = height
            
        # Skalierungsfaktoren berechnen
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        
        # Größenänderung mit Scipy oder NumPy
        if HAS_SCIPY:
            # SciPy für bessere Bildqualität verwenden
            if len(img.shape) == 3:
                # Für Farbbilder: Jeder Kanal separat
                resized = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
                for c in range(img.shape[2]):
                    resized[:, :, c] = ndimage.zoom(img[:, :, c], (scale_h, scale_w), order=1)
            else:
                # Für Graustufenbilder
                resized = ndimage.zoom(img, (scale_h, scale_w), order=1)
        else:
            # Einfache Implementierung mit NumPy (bilineare Interpolation)
            resized = _bilinear_resize(img, target_h, target_w)
            
        result.append(resized)
    
    return result

def _bilinear_resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Einfache bilineare Größenänderung mit NumPy.
    
    Args:
        img: Eingabebild
        target_h: Zielhöhe
        target_w: Zielbreite
        
    Returns:
        Größengeändertes Bild
    """
    orig_h, orig_w = img.shape[:2]
    
    # Output-Arrays erstellen
    if len(img.shape) == 3:
        resized = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
    else:
        resized = np.zeros((target_h, target_w), dtype=img.dtype)
    
    # Koordinaten-Mapping
    x_ratio = float(orig_w - 1) / (target_w - 1) if target_w > 1 else 0
    y_ratio = float(orig_h - 1) / (target_h - 1) if target_h > 1 else 0
    
    for y in range(target_h):
        for x in range(target_w):
            # Quellkoordinaten berechnen
            src_x = x * x_ratio
            src_y = y * y_ratio
            
            # Integer und Bruchteil trennen
            x_floor = int(src_x)
            y_floor = int(src_y)
            x_ceil = min(x_floor + 1, orig_w - 1)
            y_ceil = min(y_floor + 1, orig_h - 1)
            
            # Gewichte berechnen
            x_weight = src_x - x_floor
            y_weight = src_y - y_floor
            
            # Bilineare Interpolation
            if len(img.shape) == 3:
                for c in range(img.shape[2]):
                    a = img[y_floor, x_floor, c]
                    b = img[y_floor, x_ceil, c]
                    c_val = img[y_ceil, x_floor, c]
                    d = img[y_ceil, x_ceil, c]
                    
                    pixel = a * (1 - x_weight) * (1 - y_weight) + \
                            b * x_weight * (1 - y_weight) + \
                            c_val * (1 - x_weight) * y_weight + \
                            d * x_weight * y_weight
                            
                    resized[y, x, c] = pixel
            else:
                a = img[y_floor, x_floor]
                b = img[y_floor, x_ceil]
                c_val = img[y_ceil, x_floor]
                d = img[y_ceil, x_ceil]
                
                pixel = a * (1 - x_weight) * (1 - y_weight) + \
                        b * x_weight * (1 - y_weight) + \
                        c_val * (1 - x_weight) * y_weight + \
                        d * x_weight * y_weight
                        
                resized[y, x] = pixel
    
    return resized

@register_kernel(KernelOperation.BLUR, KernelType.NUMPY)
def numpy_blur(images: List[np.ndarray], kernel_size: int = 5, sigma: float = 1.0) -> List[np.ndarray]:
    """
    Wendet einen Gaußschen Weichzeichner auf eine Liste von Bildern an mit NumPy/SciPy.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        kernel_size: Größe des Gaußschen Kernels (ungerade Zahl)
        sigma: Standardabweichung des Gaußschen Kernels
        
    Returns:
        Liste der weichgezeichneten Bilder
    """
    # Sicherstellen, dass kernel_size ungerade ist
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    result = []
    
    for img in images:
        if HAS_SCIPY:
            # SciPy für bessere Leistung verwenden
            if len(img.shape) == 3:
                # Für Farbbilder: Jeder Kanal separat
                blurred = np.zeros_like(img)
                for c in range(img.shape[2]):
                    blurred[:, :, c] = ndimage.gaussian_filter(img[:, :, c], sigma=sigma)
            else:
                # Für Graustufenbilder
                blurred = ndimage.gaussian_filter(img, sigma=sigma)
        else:
            # Eigener Gaußscher Kernel mit NumPy
            kernel = _create_gaussian_kernel(kernel_size, sigma)
            
            if len(img.shape) == 3:
                # Für Farbbilder: Jeder Kanal separat
                blurred = np.zeros_like(img)
                for c in range(img.shape[2]):
                    blurred[:, :, c] = _apply_kernel(img[:, :, c], kernel)
            else:
                # Für Graustufenbilder
                blurred = _apply_kernel(img, kernel)
            
        result.append(blurred)
    
    return result

def _create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Erstellt einen Gaußschen Kernel.
    
    Args:
        size: Größe des Kernels (ungerade Zahl)
        sigma: Standardabweichung
        
    Returns:
        2D-Array mit dem Gaußschen Kernel
    """
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            
    # Kernel normieren
    return kernel / np.sum(kernel)

def _apply_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Wendet einen Kernel auf ein Bild an (Faltung).
    
    Args:
        img: Eingabebild
        kernel: Faltungskernel
        
    Returns:
        Bild nach der Faltung
    """
    if HAS_SCIPY:
        return signal.convolve2d(img, kernel, mode='same', boundary='symm')
    else:
        # Manuelle Faltung mit NumPy
        h, w = img.shape
        k_size = kernel.shape[0]
        pad = k_size // 2
        
        # Bild auspolstern
        padded = np.pad(img, pad, mode='reflect')
        
        # Ausgabebild erstellen
        output = np.zeros_like(img)
        
        # Faltung durchführen
        for i in range(h):
            for j in range(w):
                window = padded[i:i+k_size, j:j+k_size]
                output[i, j] = np.sum(window * kernel)
                
        return output

@register_kernel(KernelOperation.NORMALIZE, KernelType.NUMPY)
def numpy_normalize(images: List[np.ndarray], mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> List[np.ndarray]:
    """
    Normalisiert eine Liste von Bildern mit NumPy.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        mean: Mittelwert für die Normalisierung (pro Kanal)
        std: Standardabweichung für die Normalisierung (pro Kanal)
        
    Returns:
        Liste der normalisierten Bilder
    """
    result = []
    
    for img in images:
        # Standardisierung (Mittelwert 0, Standardabweichung 1)
        if mean is None and std is None:
            if len(img.shape) > 2:
                # Für Farbbilder: Pro Kanal normalisieren
                normalized = np.zeros_like(img, dtype=np.float32)
                for c in range(img.shape[2]):
                    channel = img[:, :, c]
                    ch_mean = np.mean(channel)
                    ch_std = np.std(channel)
                    normalized[:, :, c] = (channel - ch_mean) / (ch_std + 1e-8)
            else:
                # Für Graustufenbilder
                img_mean = np.mean(img)
                img_std = np.std(img)
                normalized = (img - img_mean) / (img_std + 1e-8)
        else:
            # Mit gegebenen Mittelwerten und Standardabweichungen
            if mean is None:
                mean = [0.0] * (img.shape[2] if len(img.shape) > 2 else 1)
            if std is None:
                std = [1.0] * (img.shape[2] if len(img.shape) > 2 else 1)
                
            if len(img.shape) > 2:
                # Für Farbbilder
                normalized = np.zeros_like(img, dtype=np.float32)
                for c in range(img.shape[2]):
                    normalized[:, :, c] = (img[:, :, c] - mean[c]) / (std[c] + 1e-8)
            else:
                # Für Graustufenbilder
                normalized = (img - mean[0]) / (std[0] + 1e-8)
        
        result.append(normalized)
    
    return result

@register_kernel(KernelOperation.EDGE_DETECTION, KernelType.NUMPY)
def numpy_edge_detection(images: List[np.ndarray], method: str = "sobel") -> List[np.ndarray]:
    """
    Wendet Kantenerkennung auf eine Liste von Bildern an mit NumPy/SciPy.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        method: Methode der Kantenerkennung ('sobel', 'prewitt', 'scharr')
        
    Returns:
        Liste der Bilder mit hervorgehobenen Kanten
    """
    # Kernels für verschiedene Kantenerkennungsmethoden
    kernels = {
        "sobel": {
            "x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        },
        "prewitt": {
            "x": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            "y": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        },
        "scharr": {
            "x": np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]),
            "y": np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        }
    }
    
    # Methode validieren
    if method not in kernels:
        logger.warning(f"Unbekannte Kantenerkennungsmethode: {method}. Verwende 'sobel' stattdessen.")
        method = "sobel"
    
    # Kernel auswählen
    kernel_x = kernels[method]["x"]
    kernel_y = kernels[method]["y"]
    
    result = []
    
    for img in images:
        # Zu Graustufen konvertieren, falls Farbbild
        if len(img.shape) > 2:
            # Einfache Graustufen-Umwandlung: Durchschnitt der Kanäle
            gray = np.mean(img, axis=2).astype(np.float32)
        else:
            gray = img.astype(np.float32)
        
        if HAS_SCIPY:
            # SciPy für bessere Leistung verwenden
            gradient_x = signal.convolve2d(gray, kernel_x, mode='same', boundary='symm')
            gradient_y = signal.convolve2d(gray, kernel_y, mode='same', boundary='symm')
        else:
            # Manuelle Faltung mit NumPy
            gradient_x = _apply_kernel(gray, kernel_x)
            gradient_y = _apply_kernel(gray, kernel_y)
        
        # Gradientenbetrag berechnen
        gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
        
        # Normalisieren auf [0, 1]
        max_mag = np.max(gradient_magnitude)
        if max_mag > 0:
            normalized = gradient_magnitude / max_mag
        else:
            normalized = gradient_magnitude
        
        result.append(normalized)
    
    return result

@register_kernel(KernelOperation.ROTATE, KernelType.NUMPY)
def numpy_rotate(images: List[np.ndarray], angle: float = 0.0) -> List[np.ndarray]:
    """
    Rotiert eine Liste von Bildern mit NumPy/SciPy.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        angle: Rotationswinkel in Grad (im Uhrzeigersinn)
        
    Returns:
        Liste der rotierten Bilder
    """
    result = []
    
    for img in images:
        if HAS_SCIPY:
            # SciPy für bessere Leistung verwenden
            # ndimage.rotate erwartet Winkel gegen den Uhrzeigersinn, daher Vorzeichen umkehren
            rotated = ndimage.rotate(img, -angle, reshape=True, order=1, mode='constant', cval=0.0)
        else:
            # Eigene Rotation mit NumPy
            rotated = _manual_rotate(img, angle)
            
        result.append(rotated)
    
    return result

def _manual_rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Manuelle Rotation mit NumPy.
    
    Args:
        img: Eingabebild
        angle: Rotationswinkel in Grad (im Uhrzeigersinn)
        
    Returns:
        Rotiertes Bild
    """
    # Winkel in Bogenmaß umrechnen
    angle_rad = angle * np.pi / 180.0
    
    # Dimensionen des Bildes
    h, w = img.shape[:2]
    
    # Verschiebung zum Zentrum
    cx, cy = w / 2, h / 2
    
    # Rotationsmatrix berechnen
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    
    # Neue Dimensionen berechnen
    new_w = int(abs(h * sin_val) + abs(w * cos_val))
    new_h = int(abs(h * cos_val) + abs(w * sin_val))
    
    # Ziel-Array erstellen
    channels = img.shape[2] if len(img.shape) > 2 else 1
    rotated = np.zeros((new_h, new_w, channels), dtype=img.dtype) if channels > 1 else np.zeros((new_h, new_w), dtype=img.dtype)
    
    # Mittelpunkte
    rcx, rcy = new_w / 2, new_h / 2
    
    # Rotation durch inverse Abbildung
    for y in range(new_h):
        for x in range(new_w):
            # Koordinaten relativ zum Ziel-Mittelpunkt
            xr = x - rcx
            yr = y - rcy
            
            # Inverse Rotation
            src_x = xr * cos_val + yr * sin_val + cx
            src_y = -xr * sin_val + yr * cos_val + cy
            
            # Interpolation, wenn innerhalb der Bildgrenzen
            if 0 <= src_x < w - 1 and 0 <= src_y < h - 1:
                # Bilineare Interpolation
                x0, y0 = int(src_x), int(src_y)
                x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
                
                wx = src_x - x0
                wy = src_y - y0
                
                if channels > 1:
                    for c in range(channels):
                        # Bilineare Interpolation je Kanal
                        top = img[y0, x0, c] * (1 - wx) + img[y0, x1, c] * wx
                        bottom = img[y1, x0, c] * (1 - wx) + img[y1, x1, c] * wx
                        rotated[y, x, c] = top * (1 - wy) + bottom * wy
                else:
                    # Bilineare Interpolation für Graustufenbild
                    top = img[y0, x0] * (1 - wx) + img[y0, x1] * wx
                    bottom = img[y1, x0] * (1 - wx) + img[y1, x1] * wx
                    rotated[y, x] = top * (1 - wy) + bottom * wy
    
    return rotated
