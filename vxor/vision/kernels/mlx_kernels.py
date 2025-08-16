#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION MLX-Kernels

Optimierte Bildverarbeitungs-Kernels mit MLX für Apple Silicon (M-Series).
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import functools
import math
import os
import sys

# Füge das Root-Verzeichnis zum Pythonpfad hinzu, falls nötig
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Importiere den zentralen MLX-Initialisierer
from vxor.core.mlx_core import init_mlx, get_mlx_status

from vxor.vision.kernels.common import (
    KernelOperation, KernelType, register_kernel, kernel_registry
)

# Logger konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VX-VISION.kernels.mlx")

# Initialisiere MLX zentral
init_mlx()

# Versuche, MLX zu importieren nach der Initialisierung
try:
    import mlx.core as mx
    import mlx.nn as nn
    
    # Prüfe, ob mlx.image verfügbar ist (in neueren Versionen ein separates Paket)
    try:
        import mlx.image as mxi
        HAS_MLX_IMAGE = True
    except ImportError:
        HAS_MLX_IMAGE = False
        logger.warning("mlx.image nicht verfügbar - verwende NumPy-basierte Bildoperationen")
    
    # Überprüfe MLX-Initialisierung
    try:
        # Teste grundlegende MLX-Funktionalität
        test_array = mx.array([1, 2, 3])
        mx.eval(test_array)
        
        HAS_MLX = True
        mlx_status = get_mlx_status()
        
        # Verwende mx.default_device() statt mx.device()
        device_info = str(mx.default_device())
        logger.info(f"MLX erfolgreich initialisiert auf {device_info}")
        logger.debug(f"MLX Status: {mlx_status}")
    except Exception as e:
        HAS_MLX = False
        logger.error(f"MLX-Initialisierung fehlgeschlagen: {e}", exc_info=True)
        
except ImportError as e:
    HAS_MLX = False
    logger.warning(f"MLX nicht verfügbar: {e}. Fallback zu CPU-Implementierungen wird verwendet.")

# JIT-Kompilierung für kritische Funktionen
def mlx_jit(func):
    """Decorator für JIT-Kompilierung von MLX-Funktionen."""
    if HAS_MLX:
        # Prüfe JIT-Status vom zentralen MLX-Manager
        mlx_status = get_mlx_status()
        if mlx_status.get("jit_enabled", False):
            logger.debug(f"JIT-Kompilierung aktiviert für {func.__name__}")
            return mx.compile(func)
        else:
            logger.debug(f"JIT-Kompilierung deaktiviert für {func.__name__}")
    return func

def _np_to_mlx(images: Union[np.ndarray, List[np.ndarray]]) -> Union[mx.array, List[mx.array]]:
    """Konvertiert NumPy-Arrays zu MLX-Arrays."""
    if not HAS_MLX:
        return images
    
    try:
        # Für einzelnes Bild
        if isinstance(images, np.ndarray):
            return mx.array(images)
        
        # Für Listen von Bildern
        if isinstance(images, (list, tuple)):
            return [mx.array(img) if isinstance(img, np.ndarray) else img for img in images]
            
        return images  # Fallback für andere Typen
    except Exception as e:
        logger.error(f"Fehler bei der Konvertierung zu MLX: {e}", exc_info=True)
        return images

def _mlx_to_np(arrays: List) -> List[np.ndarray]:
    """Konvertiert MLX-Arrays zurück zu NumPy-Arrays."""
    if not HAS_MLX:
        return arrays
    
    # Für einzelnes Bild
    if hasattr(arrays, 'shape'):
        return arrays.tolist()
    
    # Für Listen von Bildern
    return [arr.tolist() for arr in arrays]

# Hier beginnen die eigentlichen Kernel-Implementierungen

@register_kernel(KernelOperation.RESIZE, KernelType.MLX)
def mlx_resize(images: List[np.ndarray], width: int = None, height: int = None, scale: float = None) -> List[np.ndarray]:
    """
    Ändert die Größe einer Liste von Bildern mit MLX-Optimierungen.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        width: Zielbreite (wenn None, wird über scale berechnet)
        height: Zielhöhe (wenn None, wird über scale berechnet)
        scale: Skalierungsfaktor (wenn width und height None sind)
        
    Returns:
        Liste der größengeänderten Bilder
    """
    if not HAS_MLX:
        logger.warning("MLX nicht verfügbar für Größenänderung. Fallback wird verwendet.")
        fallback = kernel_registry.get(KernelOperation.RESIZE.value, KernelType.NUMPY.value)
        if fallback:
            return fallback(images, width=width, height=height, scale=scale)
        raise RuntimeError("Kein Fallback für Größenänderung verfügbar")
    
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
        
        # Umwandlung zu MLX-Array
        mlx_img = mx.array(img)
        
        # Dimensionen anpassen für mxi.resize
        if len(img.shape) == 3 and img.shape[2] == 1:
            mlx_img = mx.reshape(mlx_img, (orig_h, orig_w))
            resized = mxi.resize(mlx_img, (target_h, target_w))
            resized = mx.expand_dims(resized, axis=-1)
        else:
            resized = mxi.resize(mlx_img, (target_h, target_w))
        
        # Zurück zu NumPy konvertieren
        result.append(np.array(resized))
    
    return result

@register_kernel(KernelOperation.BLUR, KernelType.MLX)
def mlx_blur(images: List[np.ndarray], kernel_size: int = 5, sigma: float = 1.0) -> List[np.ndarray]:
    """
    Wendet einen Gaußschen Weichzeichner auf eine Liste von Bildern an mit MLX-Optimierungen.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        kernel_size: Größe des Gaußschen Kernels (ungerade Zahl)
        sigma: Standardabweichung des Gaußschen Kernels
        
    Returns:
        Liste der weichgezeichneten Bilder
    """
    if not HAS_MLX:
        logger.warning("MLX nicht verfügbar für Weichzeichnung. Fallback wird verwendet.")
        fallback = kernel_registry.get(KernelOperation.BLUR.value, KernelType.NUMPY.value)
        if fallback:
            return fallback(images, kernel_size=kernel_size, sigma=sigma)
        raise RuntimeError("Kein Fallback für Weichzeichnung verfügbar")
    
    # Sicherstellen, dass kernel_size ungerade ist
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Gaußschen Kernel erstellen
    def _create_gaussian_kernel(size, sigma):
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
                
        return kernel / np.sum(kernel)  # Normieren
    
    # Erstelle Gaußschen Kernel als MLX-Array
    gaussian_kernel = mx.array(_create_gaussian_kernel(kernel_size, sigma))
    
    # Erweitere auf 3D für Farbkanäle
    gaussian_kernel_3d = mx.expand_dims(gaussian_kernel, axis=-1)
    
    result = []
    
    for img in images:
        # Umwandlung zu MLX-Array
        mlx_img = mx.array(img)
        
        # Ausführung der Faltung für Gaußsche Weichzeichnung
        channels = []
        for c in range(img.shape[2] if len(img.shape) > 2 else 1):
            if len(img.shape) > 2:
                # Farbkanal extrahieren
                channel = mlx_img[:, :, c]
            else:
                channel = mlx_img
            
            # Auspolsterung hinzufügen
            pad_width = kernel_size // 2
            padded = mx.pad(channel, ((pad_width, pad_width), (pad_width, pad_width)))
            
            # Manuelle Faltung
            h, w = img.shape[:2]
            blurred_channel = mx.zeros((h, w))
            
            # JIT-optimierte innere Faltungsschleife
            @mlx_jit
            def convolve_kernel(padded, kernel, output):
                for i in range(h):
                    for j in range(w):
                        window = padded[i:i+kernel_size, j:j+kernel_size]
                        output[i, j] = mx.sum(window * kernel)
                return output
            
            blurred_channel = convolve_kernel(padded, gaussian_kernel, blurred_channel)
            channels.append(blurred_channel)
        
        # Kanäle zusammenführen
        if len(img.shape) > 2:
            blurred = mx.stack(channels, axis=-1)
        else:
            blurred = channels[0]
        
        # Zurück zu NumPy konvertieren
        result.append(np.array(blurred))
    
    return result

@register_kernel(KernelOperation.NORMALIZE, KernelType.MLX)
def mlx_normalize(images: List[np.ndarray], mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> List[np.ndarray]:
    """
    Normalisiert eine Liste von Bildern mit MLX-Optimierungen.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        mean: Mittelwert für die Normalisierung (pro Kanal)
        std: Standardabweichung für die Normalisierung (pro Kanal)
        
    Returns:
        Liste der normalisierten Bilder
    """
    if not HAS_MLX:
        logger.warning("MLX nicht verfügbar für Normalisierung. Fallback wird verwendet.")
        fallback = kernel_registry.get(KernelOperation.NORMALIZE.value, KernelType.NUMPY.value)
        if fallback:
            return fallback(images, mean=mean, std=std)
        raise RuntimeError("Kein Fallback für Normalisierung verfügbar")
    
    result = []
    
    for img in images:
        # Umwandlung zu MLX-Array
        mlx_img = mx.array(img)
        
        # Standardisierung (Mittelwert 0, Standardabweichung 1)
        if mean is None and std is None:
            if len(img.shape) > 2:
                # Für Farbbilder: Pro Kanal normalisieren
                channels = []
                for c in range(img.shape[2]):
                    channel = mlx_img[:, :, c]
                    ch_mean = mx.mean(channel)
                    ch_std = mx.std(channel)
                    channels.append((channel - ch_mean) / (ch_std + 1e-8))
                
                normalized = mx.stack(channels, axis=-1)
            else:
                # Für Graustufenbilder
                img_mean = mx.mean(mlx_img)
                img_std = mx.std(mlx_img)
                normalized = (mlx_img - img_mean) / (img_std + 1e-8)
        else:
            # Mit gegebenen Mittelwerten und Standardabweichungen
            if mean is None:
                mean = [0.0] * (img.shape[2] if len(img.shape) > 2 else 1)
            if std is None:
                std = [1.0] * (img.shape[2] if len(img.shape) > 2 else 1)
                
            if len(img.shape) > 2:
                # Für Farbbilder
                channels = []
                for c in range(img.shape[2]):
                    channel = mlx_img[:, :, c]
                    channels.append((channel - mean[c]) / (std[c] + 1e-8))
                
                normalized = mx.stack(channels, axis=-1)
            else:
                # Für Graustufenbilder
                normalized = (mlx_img - mean[0]) / (std[0] + 1e-8)
        
        # Zurück zu NumPy konvertieren
        result.append(np.array(normalized))
    
    return result

@register_kernel(KernelOperation.EDGE_DETECTION, KernelType.MLX)
def mlx_edge_detection(images: List[np.ndarray], method: str = "sobel") -> List[np.ndarray]:
    """
    Wendet Kantenerkennung auf eine Liste von Bildern an mit MLX-Optimierungen.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        method: Methode der Kantenerkennung ('sobel', 'prewitt', 'scharr')
        
    Returns:
        Liste der Bilder mit hervorgehobenen Kanten
    """
    if not HAS_MLX:
        logger.warning("MLX nicht verfügbar für Kantenerkennung. Fallback wird verwendet.")
        fallback = kernel_registry.get(KernelOperation.EDGE_DETECTION.value, KernelType.NUMPY.value)
        if fallback:
            return fallback(images, method=method)
        raise RuntimeError("Kein Fallback für Kantenerkennung verfügbar")
    
    # Kernels für verschiedene Kantenerkennungsmethoden
    kernels = {
        "sobel": {
            "x": mx.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            "y": mx.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        },
        "prewitt": {
            "x": mx.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            "y": mx.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        },
        "scharr": {
            "x": mx.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]),
            "y": mx.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
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
        
        # Umwandlung zu MLX-Array
        mlx_img = mx.array(gray)
        
        # Auspolsterung hinzufügen
        pad_width = 1  # Kernelgröße ist 3x3
        padded = mx.pad(mlx_img, ((pad_width, pad_width), (pad_width, pad_width)))
        
        # Dimensionen des Bildes
        h, w = gray.shape
        
        # Ausgabe-Arrays
        gradient_x = mx.zeros((h, w))
        gradient_y = mx.zeros((h, w))
        
        # JIT-optimierte Faltungsfunktion
        @mlx_jit
        def apply_kernel(padded, kernel, output):
            for i in range(h):
                for j in range(w):
                    window = padded[i:i+3, j:j+3]
                    output[i, j] = mx.sum(window * kernel)
            return output
        
        # Anwenden der Kernels
        gradient_x = apply_kernel(padded, kernel_x, gradient_x)
        gradient_y = apply_kernel(padded, kernel_y, gradient_y)
        
        # Gradientenbetrag berechnen
        gradient_magnitude = mx.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalisieren auf [0, 1]
        max_mag = mx.max(gradient_magnitude)
        if max_mag > 0:
            normalized = gradient_magnitude / max_mag
        else:
            normalized = gradient_magnitude
        
        # Zurück zu NumPy konvertieren
        result.append(np.array(normalized))
    
    return result

@register_kernel(KernelOperation.ROTATE, KernelType.MLX)
def mlx_rotate(images: List[np.ndarray], angle: float = 0.0) -> List[np.ndarray]:
    """
    Rotiert eine Liste von Bildern mit MLX-Optimierungen.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        angle: Rotationswinkel in Grad (im Uhrzeigersinn)
        
    Returns:
        Liste der rotierten Bilder
    """
    if not HAS_MLX:
        logger.warning("MLX nicht verfügbar für Rotation. Fallback wird verwendet.")
        fallback = kernel_registry.get(KernelOperation.ROTATE.value, KernelType.NUMPY.value)
        if fallback:
            return fallback(images, angle=angle)
        raise RuntimeError("Kein Fallback für Rotation verfügbar")
    
    # Winkel in Bogenmaß umrechnen
    angle_rad = angle * np.pi / 180.0
    
    result = []
    
    for img in images:
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
        
        result.append(rotated)
    
    return result
