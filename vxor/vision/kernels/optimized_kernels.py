#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION Optimierte Kernels

Dieses Modul enthält optimierte Bildverarbeitungs-Kernels für verschiedene Hardware-Plattformen.
Die Kernels nutzen hardwarespezifische Beschleunigungstechnologien wie SIMD, Apple Neural Engine,
CUDA und OpenCL/Metal.
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
import os
import time
from functools import lru_cache
from threading import Lock

# Versuche, spezielle Beschleunigungsbibliotheken zu importieren
try:
    import mlx
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Konfiguration des Loggers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VX-VISION.kernels")

# Globale Locks für Thread-Sicherheit
mlx_lock = Lock()
torch_lock = Lock()

class OptimizedKernels:
    """
    Sammlung von optimierten Bildverarbeitungs-Kernels mit automatischer
    Hardwareanpassung für maximale Leistung.
    """
    
    def __init__(self, hardware_detector=None):
        """
        Initialisiert die optimierten Kernels mit dem angegebenen Hardware-Detektor.
        
        Args:
            hardware_detector: Eine Instanz des HardwareDetector zur Hardwareerkennung
        """
        from ..hardware.detector import HardwareDetector
        
        self.hardware = hardware_detector or HardwareDetector()
        self.kernel_registry = {}
        self.kernel_performance = {}
        
        # Kernel-Backends registrieren
        self._register_kernels()
        
        # JIT-Warmup für häufig verwendete Operationen
        if self.hardware.has_neural_engine() and HAS_MLX:
            logger.info("MLX JIT-Warmup für häufig verwendete Operationen wird durchgeführt...")
            self._mlx_warmup()
        
    def _register_kernels(self):
        """Registriert alle verfügbaren Kernel-Implementierungen."""
        # Resize-Kernels
        self.kernel_registry["resize"] = {
            "cpu": self._resize_cpu,
            "opencv": self._resize_opencv
        }
        
        # Filter-Kernels
        self.kernel_registry["blur"] = {
            "cpu": self._blur_cpu,
            "opencv": self._blur_opencv
        }
        
        self.kernel_registry["edge_detection"] = {
            "cpu": self._edge_detection_cpu,
            "opencv": self._edge_detection_opencv
        }
        
        # Spezielle Hardware-Beschleuniger registrieren
        if self.hardware.has_neural_engine() and HAS_MLX:
            self.kernel_registry["resize"]["mlx"] = self._resize_mlx
            self.kernel_registry["blur"]["mlx"] = self._blur_mlx
            
        if self.hardware.has_gpu() and HAS_TORCH:
            self.kernel_registry["resize"]["torch"] = self._resize_torch
            self.kernel_registry["blur"]["torch"] = self._blur_torch
            self.kernel_registry["edge_detection"]["torch"] = self._edge_detection_torch
        
        if HAS_NUMBA:
            # Numba-beschleunigte CPU-Kernels
            self.kernel_registry["resize"]["numba"] = self._resize_numba
            self.kernel_registry["blur"]["numba"] = self._blur_numba
            self.kernel_registry["edge_detection"]["numba"] = self._edge_detection_numba
    
    def get_optimal_kernel(self, operation: str, image_shape: Tuple[int, int, int], 
                           params: Dict = None) -> Tuple[Callable, str]:
        """
        Wählt den optimalen Kernel für die angegebene Operation basierend auf 
        Hardwareverfügbarkeit, Bildgröße und Operationsparametern.
        
        Args:
            operation: Name der Operation (z.B. "resize", "blur")
            image_shape: Form des Bildes (Höhe, Breite, Kanäle)
            params: Optionale Parameter für die Operation
            
        Returns:
            Tuple[Callable, str]: (Kernel-Funktion, Name des Backends)
        """
        height, width, channels = image_shape
        image_size = height * width * channels
        
        # Standard-Backend festlegen
        default_backend = "opencv"  # OpenCV ist in der Regel gut optimiert
        
        # Bei sehr großen Bildern und Neural Engine, MLX verwenden
        if image_size > 1_000_000 and self.hardware.has_neural_engine() and HAS_MLX:
            preferred_backend = "mlx"
        # Bei kleineren Bildern, aber immer noch GPU, Torch verwenden
        elif image_size > 500_000 and self.hardware.has_gpu() and HAS_TORCH:
            preferred_backend = "torch"
        # Bei sehr kleinen Bildern ist numba manchmal schneller (weniger Overhead)
        elif image_size < 100_000 and HAS_NUMBA:
            preferred_backend = "numba"
        else:
            preferred_backend = default_backend
        
        # Überprüfen, ob das bevorzugte Backend die Operation unterstützt
        if (operation in self.kernel_registry and 
            preferred_backend in self.kernel_registry[operation]):
            return (self.kernel_registry[operation][preferred_backend], preferred_backend)
        
        # Fallback auf OpenCV
        if operation in self.kernel_registry and default_backend in self.kernel_registry[operation]:
            return (self.kernel_registry[operation][default_backend], default_backend)
        
        # Letzter Fallback auf CPU
        if operation in self.kernel_registry and "cpu" in self.kernel_registry[operation]:
            return (self.kernel_registry[operation]["cpu"], "cpu")
        
        raise ValueError(f"Keine Implementierung für Operation '{operation}' gefunden.")
    
    # ---- CPU-IMPLEMENTIERUNGEN ----
    
    def _resize_cpu(self, image: np.ndarray, width: int = None, height: int = None, 
                     scale: float = None) -> np.ndarray:
        """
        Führt Bildgrößenänderung mit reiner CPU-Implementierung durch.
        
        Args:
            image: Eingabebild als NumPy-Array
            width: Zielbreite (optional)
            height: Zielhöhe (optional)
            scale: Skalierungsfaktor (optional)
            
        Returns:
            np.ndarray: Größengeändertes Bild
        """
        orig_h, orig_w = image.shape[:2]
        
        # Bestimme Zielgröße
        if scale is not None:
            target_w, target_h = int(orig_w * scale), int(orig_h * scale)
        elif width is not None and height is not None:
            target_w, target_h = width, height
        elif width is not None:
            # Seitenverhältnis beibehalten
            target_w = width
            target_h = int(orig_h * (width / orig_w))
        elif height is not None:
            # Seitenverhältnis beibehalten
            target_h = height
            target_w = int(orig_w * (height / orig_h))
        else:
            return image  # Keine Änderung
        
        # Einfache bilineare Interpolation implementieren
        result = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        h_ratio = orig_h / target_h
        w_ratio = orig_w / target_w
        
        for i in range(target_h):
            for j in range(target_w):
                # Quellkoordinaten berechnen
                src_y = i * h_ratio
                src_x = j * w_ratio
                
                # Bilineare Interpolation
                y1, y2 = int(src_y), min(int(src_y) + 1, orig_h - 1)
                x1, x2 = int(src_x), min(int(src_x) + 1, orig_w - 1)
                
                # Gewichtungen berechnen
                wy2 = src_y - y1
                wy1 = 1 - wy2
                wx2 = src_x - x1
                wx1 = 1 - wx2
                
                # Interpolierter Wert
                result[i, j] = (wy1 * wx1 * image[y1, x1] +
                                wy1 * wx2 * image[y1, x2] +
                                wy2 * wx1 * image[y2, x1] +
                                wy2 * wx2 * image[y2, x2])
        
        return result
    
    def _blur_cpu(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Führt Gaußsche Unschärfe mit reiner CPU-Implementierung durch.
        
        Args:
            image: Eingabebild als NumPy-Array
            kernel_size: Größe des Unschärfe-Kernels
            
        Returns:
            np.ndarray: Unscharfes Bild
        """
        # Validiere kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1  # Stelle sicher, dass Kernelgröße ungerade ist
            
        # Erstelle Gauß-Kernel
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        center = kernel_size // 2
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        # Berechne Gauß-Kernel
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalisiere Kernel
        kernel = kernel / kernel.sum()
        
        # Padding für das Bild
        pad_size = kernel_size // 2
        padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
        
        # Output-Bild initialisieren
        h, w, c = image.shape
        result = np.zeros_like(image)
        
        # Faltung anwenden
        for y in range(h):
            for x in range(w):
                for channel in range(c):
                    # Extrahiere Patch
                    patch = padded[y:y+kernel_size, x:x+kernel_size, channel]
                    # Anwenden des Kernels
                    result[y, x, channel] = np.sum(patch * kernel)
        
        return result
    
    def _edge_detection_cpu(self, image: np.ndarray) -> np.ndarray:
        """
        Führt Kantendetektion mit reiner CPU-Implementierung durch (Sobel-Operator).
        
        Args:
            image: Eingabebild als NumPy-Array
            
        Returns:
            np.ndarray: Bild mit hervorgehobenen Kanten
        """
        # Umwandlung in Graustufen, falls erforderlich
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image.copy()
        
        # Sobel-Operatoren
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # Padding
        padded = np.pad(gray, 1, mode='reflect')
        
        # Output initialisieren
        h, w = gray.shape
        gradient_x = np.zeros_like(gray, dtype=np.float32)
        gradient_y = np.zeros_like(gray, dtype=np.float32)
        
        # Sobel-Operatoren anwenden
        for y in range(h):
            for x in range(w):
                # Extrahiere 3x3 Patch
                patch = padded[y:y+3, x:x+3]
                # Anwenden der Sobel-Operatoren
                gradient_x[y, x] = np.sum(patch * sobel_x)
                gradient_y[y, x] = np.sum(patch * sobel_y)
        
        # Kantenstärke berechnen
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalisieren auf 0-255
        gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
        
        # Wenn Originalbild Farbkanäle hat, Ergebnis entsprechend anpassen
        if len(image.shape) == 3:
            result = np.stack([gradient_magnitude] * 3, axis=2).astype(np.uint8)
        else:
            result = gradient_magnitude.astype(np.uint8)
        
        return result

    # ---- MLX-IMPLEMENTIERUNGEN (APPLE SILICON) ----
    
    def _mlx_warmup(self):
        """
        Führt einen Warmup für MLX JIT-Kompilierung durch, um Latenz zu reduzieren.
        """
        try:
            with mlx_lock:
                # Erstelle kleine Test-Arrays für JIT-Warmup
                test_img_small = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                test_img_medium = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
                
                # Warmup für Resize
                _ = self._resize_mlx(test_img_small, width=64, height=64)
                _ = self._resize_mlx(test_img_medium, scale=0.5)
                
                # Warmup für Blur
                _ = self._blur_mlx(test_img_small, kernel_size=3)
                _ = self._blur_mlx(test_img_medium, kernel_size=5)
                
                logger.info("MLX JIT-Warmup abgeschlossen.")
        except Exception as e:
            logger.warning(f"MLX JIT-Warmup fehlgeschlagen: {e}")
    
    def _resize_mlx(self, image: np.ndarray, width: int = None, height: int = None, 
                     scale: float = None) -> np.ndarray:
        """
        Führt Bildgrößenänderung mit MLX für Apple Silicon durch.
        
        Args:
            image: Eingabebild als NumPy-Array
            width: Zielbreite (optional)
            height: Zielhöhe (optional)
            scale: Skalierungsfaktor (optional)
            
        Returns:
            np.ndarray: Größengeändertes Bild
        """
        if not HAS_MLX:
            return self._resize_opencv(image, width, height, scale)
            
        try:
            with mlx_lock:
                orig_h, orig_w = image.shape[:2]
                
                # Bestimme Zielgröße
                if scale is not None:
                    target_w, target_h = int(orig_w * scale), int(orig_h * scale)
                elif width is not None and height is not None:
                    target_w, target_h = width, height
                elif width is not None:
                    # Seitenverhältnis beibehalten
                    target_w = width
                    target_h = int(orig_h * (width / orig_w))
                elif height is not None:
                    # Seitenverhältnis beibehalten
                    target_h = height
                    target_w = int(orig_w * (height / orig_h))
                else:
                    return image  # Keine Änderung
                
                # Definiere eine MLX-Funktion für die Interpolation
                @mx.compile
                def resize_bilinear(img, h_scale, w_scale, out_h, out_w):
                    # Koordinatenraster erstellen
                    y_indices = mx.arange(out_h).reshape(-1, 1) * h_scale
                    x_indices = mx.arange(out_w).reshape(1, -1) * w_scale
                    
                    # Ganze und gebrochene Anteile extrahieren
                    y0 = mx.floor(y_indices).astype(mx.int32)
                    x0 = mx.floor(x_indices).astype(mx.int32)
                    y1 = mx.minimum(y0 + 1, mx.array(img.shape[0] - 1))
                    x1 = mx.minimum(x0 + 1, mx.array(img.shape[1] - 1))
                    
                    # Gewichtungen berechnen
                    wy1 = (y_indices - y0.astype(mx.float32))
                    wx1 = (x_indices - x0.astype(mx.float32))
                    wy0 = 1.0 - wy1
                    wx0 = 1.0 - wx1
                    
                    # Bilineare Interpolation durch gewichtete Summe
                    w00 = (wy0 * wx0).reshape(out_h, out_w, 1)
                    w01 = (wy0 * wx1).reshape(out_h, out_w, 1)
                    w10 = (wy1 * wx0).reshape(out_h, out_w, 1)
                    w11 = (wy1 * wx1).reshape(out_h, out_w, 1)
                    
                    # Effiziente Vektorisierung für Interpolation
                    result = mx.zeros((out_h, out_w, img.shape[2]), dtype=img.dtype)
                    
                    for y in range(out_h):
                        for x in range(out_w):
                            y0_idx, y1_idx = y0[y, 0], y1[y, 0]
                            x0_idx, x1_idx = x0[0, x], x1[0, x]
                            
                            # Pixel abrufen und gewichten
                            p00 = img[y0_idx, x0_idx]
                            p01 = img[y0_idx, x1_idx]
                            p10 = img[y1_idx, x0_idx]
                            p11 = img[y1_idx, x1_idx]
                            
                            # Gewichtete Summe berechnen
                            result = result.at[y, x].set(
                                w00[y, x, 0] * p00 + 
                                w01[y, x, 0] * p01 + 
                                w10[y, x, 0] * p10 + 
                                w11[y, x, 0] * p11
                            )
                    
                    return result
                
                # Konvertiere zu MLX Array und führe Resize durch
                mx_img = mx.array(image)
                h_scale = orig_h / target_h
                w_scale = orig_w / target_w
                
                # Führe die vektorisierte MLX-Funktion aus
                result_mx = resize_bilinear(mx_img, h_scale, w_scale, target_h, target_w)
                
                # Zurück zu NumPy
                return np.array(result_mx).astype(image.dtype)
                
        except Exception as e:
            logger.warning(f"MLX resize fehlgeschlagen, fallback auf OpenCV: {e}")
            return self._resize_opencv(image, width, height, scale)
    
    def _blur_mlx(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Führt Gaußsche Unschärfe mit MLX für Apple Silicon durch.
        
        Args:
            image: Eingabebild als NumPy-Array
            kernel_size: Größe des Unschärfe-Kernels
            
        Returns:
            np.ndarray: Unscharfes Bild
        """
        if not HAS_MLX:
            return self._blur_opencv(image, kernel_size)
            
        try:
            with mlx_lock:
                # Validiere kernel_size
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # Konvertiere zu MLX Array
                mx_img = mx.array(image)
                
                # Erstelle Gauß-Kernel mit MLX
                sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
                center = kernel_size // 2
                
                @mx.compile
                def create_gaussian_kernel(size, sigma):
                    # Erstelle x und y Koordinatenraster
                    ax = mx.arange(size) - center
                    xx, yy = mx.meshgrid(ax, ax)
                    
                    # Berechne Gauß-Kernel
                    kernel = mx.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                    
                    # Normalisiere Kernel
                    return kernel / mx.sum(kernel)
                
                @mx.compile
                def apply_kernel(img, kernel):
                    h, w, c = img.shape
                    k_size = kernel.shape[0]
                    pad = k_size // 2
                    
                    # Padding hinzufügen
                    padded = mx.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
                    result = mx.zeros_like(img)
                    
                    # Faltung anwenden - vektorisiert mit eingeschränkter Schleife
                    for y in range(h):
                        for x in range(w):
                            # Extrahiere Patch für jeden Kanal
                            for ch in range(c):
                                patch = padded[y:y+k_size, x:x+k_size, ch]
                                # Gewichtete Summe berechnen
                                value = mx.sum(patch * kernel)
                                result = result.at[y, x, ch].set(value)
                    
                    return result
                
                # Erstelle und wende Kernel an
                gauss_kernel = create_gaussian_kernel(kernel_size, sigma)
                # Erweitere Kernel für Channel-weise Anwendung
                result_mx = apply_kernel(mx_img, gauss_kernel)
                
                # Zurück zu NumPy
                return np.array(result_mx).astype(image.dtype)
                
        except Exception as e:
            logger.warning(f"MLX blur fehlgeschlagen, fallback auf OpenCV: {e}")
            return self._blur_opencv(image, kernel_size)
            
    # ---- NUMBA-BESCHLEUNIGTE CPU-IMPLEMENTIERUNGEN ----
    
    def _resize_numba(self, image: np.ndarray, width: int = None, height: int = None, 
                       scale: float = None) -> np.ndarray:
        """
        Führt Bildgrößenänderung mit Numba-beschleunigter CPU-Implementierung durch.
        
        Args:
            image: Eingabebild als NumPy-Array
            width: Zielbreite (optional)
            height: Zielhöhe (optional)
            scale: Skalierungsfaktor (optional)
            
        Returns:
            np.ndarray: Größengeändertes Bild
        """
        if not HAS_NUMBA:
            return self._resize_cpu(image, width, height, scale)
            
        orig_h, orig_w = image.shape[:2]
        
        # Bestimme Zielgröße
        if scale is not None:
            target_w, target_h = int(orig_w * scale), int(orig_h * scale)
        elif width is not None and height is not None:
            target_w, target_h = width, height
        elif width is not None:
            # Seitenverhältnis beibehalten
            target_w = width
            target_h = int(orig_h * (width / orig_w))
        elif height is not None:
            # Seitenverhältnis beibehalten
            target_h = height
            target_w = int(orig_w * (height / orig_h))
        else:
            return image  # Keine Änderung
            
        try:
            # Numba-beschleunigte Funktion für bilineare Interpolation
            @numba.njit(parallel=True)
            def resize_bilinear_numba(img, target_h, target_w):
                h, w, c = img.shape
                result = np.zeros((target_h, target_w, c), dtype=img.dtype)
                
                h_ratio = h / target_h
                w_ratio = w / target_w
                
                # Parallele Verarbeitung mit numba.prange
                for i in numba.prange(target_h):
                    for j in range(target_w):
                        # Quellkoordinaten berechnen
                        src_y = i * h_ratio
                        src_x = j * w_ratio
                        
                        # Bilineare Interpolation
                        y1, y2 = int(src_y), min(int(src_y) + 1, h - 1)
                        x1, x2 = int(src_x), min(int(src_x) + 1, w - 1)
                        
                        # Gewichtungen berechnen
                        wy2 = src_y - y1
                        wy1 = 1 - wy2
                        wx2 = src_x - x1
                        wx1 = 1 - wx2
                        
                        # Interpolierter Wert
                        for k in range(c):
                            result[i, j, k] = (wy1 * wx1 * img[y1, x1, k] +
                                            wy1 * wx2 * img[y1, x2, k] +
                                            wy2 * wx1 * img[y2, x1, k] +
                                            wy2 * wx2 * img[y2, x2, k])
                            
                return result
                
            return resize_bilinear_numba(image, target_h, target_w)
        except Exception as e:
            logger.warning(f"Numba resize fehlgeschlagen, fallback auf CPU: {e}")
            return self._resize_cpu(image, width, height, scale)
    
    def _blur_numba(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Führt Gaußsche Unschärfe mit Numba-beschleunigter CPU-Implementierung durch.
        
        Args:
            image: Eingabebild als NumPy-Array
            kernel_size: Größe des Unschärfe-Kernels
            
        Returns:
            np.ndarray: Unscharfes Bild
        """
        if not HAS_NUMBA:
            return self._blur_cpu(image, kernel_size)
            
        # Validiere kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1  # Stelle sicher, dass Kernelgröße ungerade ist
            
        try:
            # Numba-beschleunigte Funktion für Gauß-Unschärfe
            @numba.njit
            def create_gaussian_kernel_numba(size, sigma):
                kernel = np.zeros((size, size), dtype=np.float32)
                center = size // 2
                
                # Berechne Gauß-Kernel
                for i in range(size):
                    for j in range(size):
                        x, y = i - center, j - center
                        kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
                
                # Normalisiere Kernel
                return kernel / np.sum(kernel)
            
            @numba.njit(parallel=True)
            def apply_gaussian_filter_numba(img, kernel):
                h, w, c = img.shape
                k_size = kernel.shape[0]
                pad = k_size // 2
                
                # Output-Bild initialisieren
                result = np.zeros_like(img)
                
                # Erweitere das Bild mit Spiegelrand
                padded = np.zeros((h + 2*pad, w + 2*pad, c), dtype=img.dtype)
                
                # Manuelles Padding (reflect mode)
                # Mittlerer Bereich
                padded[pad:pad+h, pad:pad+w, :] = img
                
                # Obere und untere Ränder
                for i in range(pad):
                    padded[i, pad:pad+w, :] = img[pad-i-1, :, :]
                    padded[pad+h+i, pad:pad+w, :] = img[h-i-1, :, :]
                
                # Linke und rechte Ränder
                for i in range(pad):
                    padded[pad:pad+h, i, :] = img[:, pad-i-1, :]
                    padded[pad:pad+h, pad+w+i, :] = img[:, w-i-1, :]
                
                # Ecken
                for i in range(pad):
                    for j in range(pad):
                        padded[i, j, :] = img[pad-i-1, pad-j-1, :]
                        padded[i, pad+w+j, :] = img[pad-i-1, w-j-1, :]
                        padded[pad+h+i, j, :] = img[h-i-1, pad-j-1, :]
                        padded[pad+h+i, pad+w+j, :] = img[h-i-1, w-j-1, :]
                
                # Parallele Verarbeitung mit numba.prange
                for y in numba.prange(h):
                    for x in range(w):
                        for channel in range(c):
                            # Extrahiere Patch
                            patch = padded[y:y+k_size, x:x+k_size, channel]
                            # Anwenden des Kernels
                            val = 0.0
                            for ky in range(k_size):
                                for kx in range(k_size):
                                    val += patch[ky, kx] * kernel[ky, kx]
                            result[y, x, channel] = val
                
                return result
            
            # Erstelle Gauß-Kernel
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
            kernel = create_gaussian_kernel_numba(kernel_size, sigma)
            
            # Wende Gauß-Filter an
            return apply_gaussian_filter_numba(image, kernel)
            
        except Exception as e:
            logger.warning(f"Numba blur fehlgeschlagen, fallback auf CPU: {e}")
            return self._blur_cpu(image, kernel_size)
    
    def _edge_detection_numba(self, image: np.ndarray) -> np.ndarray:
        """
        Führt Kantendetektion mit Numba-beschleunigter CPU-Implementierung durch.
        
        Args:
            image: Eingabebild als NumPy-Array
            
        Returns:
            np.ndarray: Bild mit hervorgehobenen Kanten
        """
        if not HAS_NUMBA:
            return self._edge_detection_cpu(image)
            
        try:
            # Umwandlung in Graustufen, falls erforderlich
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = image.copy()
            
            @numba.njit(parallel=True)
            def sobel_edge_detection_numba(gray_img):
                h, w = gray_img.shape
                
                # Sobel-Operatoren
                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
                
                # Output initialisieren
                gradient_x = np.zeros_like(gray_img, dtype=np.float32)
                gradient_y = np.zeros_like(gray_img, dtype=np.float32)
                gradient_magnitude = np.zeros_like(gray_img, dtype=np.float32)
                
                # Padding hinzufügen
                padded = np.zeros((h+2, w+2), dtype=gray_img.dtype)
                padded[1:-1, 1:-1] = gray_img
                
                # Spiegelrand-Padding
                padded[0, 1:-1] = gray_img[0, :]
                padded[-1, 1:-1] = gray_img[-1, :]
                padded[1:-1, 0] = gray_img[:, 0]
                padded[1:-1, -1] = gray_img[:, -1]
                padded[0, 0] = gray_img[0, 0]
                padded[0, -1] = gray_img[0, -1]
                padded[-1, 0] = gray_img[-1, 0]
                padded[-1, -1] = gray_img[-1, -1]
                
                # Parallele Verarbeitung mit numba.prange
                for y in numba.prange(h):
                    for x in range(w):
                        # Extrahiere 3x3 Patch
                        patch = padded[y:y+3, x:x+3]
                        
                        # Anwenden der Sobel-Operatoren
                        gx = 0.0
                        gy = 0.0
                        for ky in range(3):
                            for kx in range(3):
                                gx += patch[ky, kx] * sobel_x[ky, kx]
                                gy += patch[ky, kx] * sobel_y[ky, kx]
                                
                        gradient_x[y, x] = gx
                        gradient_y[y, x] = gy
                        
                        # Kantenstärke berechnen
                        gradient_magnitude[y, x] = np.sqrt(gx*gx + gy*gy)
                
                # Normalisieren auf 0-255
                max_mag = np.max(gradient_magnitude)
                if max_mag > 0:
                    gradient_magnitude = 255 * gradient_magnitude / max_mag
                
                return gradient_magnitude.astype(np.uint8)
            
            # Führe Kantendetektion durch
            edges = sobel_edge_detection_numba(gray)
            
            # Wenn Originalbild Farbkanäle hat, Ergebnis entsprechend anpassen
            if len(image.shape) == 3:
                result = np.stack([edges] * 3, axis=2)
            else:
                result = edges
            
            return result
        
        except Exception as e:
            logger.warning(f"Numba edge detection fehlgeschlagen, fallback auf CPU: {e}")
            return self._edge_detection_cpu(image)
            
    # ---- TORCH-IMPLEMENTIERUNGEN (GPU) ----
    
    def _resize_torch(self, image: np.ndarray, width: int = None, height: int = None, 
                      scale: float = None) -> np.ndarray:
        """
        Führt Bildgrößenänderung mit PyTorch GPU-Beschleunigung durch.
        
        Args:
            image: Eingabebild als NumPy-Array
            width: Zielbreite (optional)
            height: Zielhöhe (optional)
            scale: Skalierungsfaktor (optional)
            
        Returns:
            np.ndarray: Größengeändertes Bild
        """
        if not HAS_TORCH:
            return self._resize_opencv(image, width, height, scale)
            
        try:
            with torch_lock:
                orig_h, orig_w = image.shape[:2]
                
                # Bestimme Zielgröße
                if scale is not None:
                    target_w, target_h = int(orig_w * scale), int(orig_h * scale)
                elif width is not None and height is not None:
                    target_w, target_h = width, height
                elif width is not None:
                    # Seitenverhältnis beibehalten
                    target_w = width
                    target_h = int(orig_h * (width / orig_w))
                elif height is not None:
                    # Seitenverhältnis beibehalten
                    target_h = height
                    target_w = int(orig_w * (height / orig_h))
                else:
                    return image  # Keine Änderung
                
                # NumPy zu PyTorch konvertieren
                # PyTorch erwartet NCHW Format (Batch, Channel, Height, Width)
                # PyTorch's F.interpolate erwartet float32
                img_torch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                
                # GPU verwenden, wenn verfügbar
                if torch.cuda.is_available():
                    img_torch = img_torch.cuda()
                
                # Führe Resize mit bilinearer Interpolation durch
                result_torch = torch.nn.functional.interpolate(
                    img_torch, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Zurück zu CPU und NumPy konvertieren
                result_torch = result_torch.squeeze(0).permute(1, 2, 0)
                if torch.cuda.is_available():
                    result_torch = result_torch.cpu()
                    
                # Zurück zum originalen Datentyp konvertieren
                result_np = result_torch.numpy()
                return result_np.astype(image.dtype)
                
        except Exception as e:
            logger.warning(f"PyTorch resize fehlgeschlagen, fallback auf OpenCV: {e}")
            return self._resize_opencv(image, width, height, scale)
    
    def _blur_torch(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Führt Gaußsche Unschärfe mit PyTorch GPU-Beschleunigung durch.
        
        Args:
            image: Eingabebild als NumPy-Array
            kernel_size: Größe des Unschärfe-Kernels
            
        Returns:
            np.ndarray: Unscharfes Bild
        """
        if not HAS_TORCH:
            return self._blur_opencv(image, kernel_size)
            
        try:
            with torch_lock:
                # Validiere kernel_size
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # NumPy zu PyTorch konvertieren
                img_torch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
                
                # GPU verwenden, wenn verfügbar
                if torch.cuda.is_available():
                    img_torch = img_torch.cuda()
                
                # Berechne Sigma basierend auf Kernelgröße
                sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
                
                # Gaussian Blur anwenden mit PyTorch
                channels = image.shape[2]
                padding = kernel_size // 2
                
                # Erstelle 2D Gauß-Kernel
                x_coord = torch.arange(kernel_size)
                x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
                y_grid = x_grid.t()
                xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
                
                mean = (kernel_size - 1) / 2.
                variance = sigma**2
                
                # Berechne 2D Gauß-Kernel
                gaussian_kernel = (1./(2.*math.pi*variance)) * \
                                torch.exp(
                                    -torch.sum((xy_grid - mean)**2., dim=-1) / \
                                    (2*variance)
                                )
                
                # Normalisiere Kernel
                gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
                
                # Erweitere auf die Anzahl der Kanäle
                gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
                gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
                
                # Definiere Faltungsoperation
                conv = torch.nn.Conv2d(
                    in_channels=channels, 
                    out_channels=channels,
                    kernel_size=kernel_size, 
                    groups=channels,
                    bias=False,
                    padding=padding
                )
                
                # Lade Kernel-Gewichte
                conv.weight.data = gaussian_kernel
                conv.weight.requires_grad = False
                
                if torch.cuda.is_available():
                    conv = conv.cuda()
                
                # Wende Faltung an
                result_torch = conv(img_torch)
                
                # Zurück zu CPU und NumPy konvertieren
                result_torch = result_torch.squeeze(0).permute(1, 2, 0)
                if torch.cuda.is_available():
                    result_torch = result_torch.cpu()
                    
                # Zurück zum originalen Datentyp konvertieren
                result_np = result_torch.numpy()
                return result_np.astype(image.dtype)
                
        except Exception as e:
            logger.warning(f"PyTorch blur fehlgeschlagen, fallback auf OpenCV: {e}")
            return self._blur_opencv(image, kernel_size)
    
    def _edge_detection_torch(self, image: np.ndarray) -> np.ndarray:
        """
        Führt Kantendetektion mit PyTorch GPU-Beschleunigung durch.
        
        Args:
            image: Eingabebild als NumPy-Array
            
        Returns:
            np.ndarray: Bild mit hervorgehobenen Kanten
        """
        if not HAS_TORCH:
            return self._edge_detection_opencv(image)
            
        try:
            with torch_lock:
                # Umwandlung in Graustufen, falls erforderlich
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Konvertiere zu Graustufen mit RGB-Gewichtung
                    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
                    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                    gray = gray.astype(np.uint8)
                else:
                    gray = image.copy()
                
                # NumPy zu PyTorch konvertieren
                img_torch = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float()
                
                # GPU verwenden, wenn verfügbar
                if torch.cuda.is_available():
                    img_torch = img_torch.cuda()
                
                # Definiere Sobel-Operatoren
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                       dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                       dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                if torch.cuda.is_available():
                    sobel_x = sobel_x.cuda()
                    sobel_y = sobel_y.cuda()
                
                # Definiere Faltungsoperationen
                conv_x = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
                conv_y = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
                
                # Lade Kernel-Gewichte
                conv_x.weight.data = sobel_x
                conv_y.weight.data = sobel_y
                conv_x.weight.requires_grad = False
                conv_y.weight.requires_grad = False
                
                if torch.cuda.is_available():
                    conv_x = conv_x.cuda()
                    conv_y = conv_y.cuda()
                
                # Wende Sobel-Operatoren an
                grad_x = conv_x(img_torch)
                grad_y = conv_y(img_torch)
                
                # Berechne Kantenstärke
                grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
                
                # Normalisiere auf 0-255
                grad_magnitude = grad_magnitude * (255.0 / grad_magnitude.max())
                
                # Zurück zu CPU und NumPy konvertieren
                grad_magnitude = grad_magnitude.squeeze()
                if torch.cuda.is_available():
                    grad_magnitude = grad_magnitude.cpu()
                
                edges = grad_magnitude.numpy().astype(np.uint8)
                
                # Wenn Originalbild Farbkanäle hat, Ergebnis entsprechend anpassen
                if len(image.shape) == 3:
                    result = np.stack([edges] * 3, axis=2)
                else:
                    result = edges
                
                return result
                
        except Exception as e:
            logger.warning(f"PyTorch edge detection fehlgeschlagen, fallback auf OpenCV: {e}")
            return self._edge_detection_opencv(image)
    
    # Hauptschnittstelle für externe Aufrufe
    def __call__(self, operation: str, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, str]:
        """
        Wendet die angegebene Operation auf das Bild mit dem optimalen Kernel an.
        
        Args:
            operation: Name der Operation (z.B. "resize", "blur", "edge_detection")
            image: Eingabebild als NumPy-Array
            **kwargs: Zusätzliche Parameter für die Operation
            
        Returns:
            Tuple[np.ndarray, str]: (Verarbeitetes Bild, verwendetes Backend)
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Bild muss ein NumPy-Array sein")
            
        # Bestimme optimalen Kernel
        kernel_func, backend = self.get_optimal_kernel(operation, image.shape, kwargs)
        
        # Führe Operation aus
        start_time = time.time()
        result = kernel_func(image, **kwargs)
        elapsed = time.time() - start_time
        
        logger.debug(f"Operation {operation} mit {backend} Backend: {elapsed:.3f}s")
        
        return result, backend
    # ---- OPENCV-IMPLEMENTIERUNGEN ----
    
    def _resize_opencv(self, image: np.ndarray, width: int = None, height: int = None, 
                        scale: float = None) -> np.ndarray:
        """
        Führt Bildgrößenänderung mit OpenCV durch.
        
        Args:
            image: Eingabebild als NumPy-Array
            width: Zielbreite (optional)
            height: Zielhöhe (optional)
            scale: Skalierungsfaktor (optional)
            
        Returns:
            np.ndarray: Größengeändertes Bild
        """
        orig_h, orig_w = image.shape[:2]
        
        # Bestimme Zielgröße
        if scale is not None:
            target_w, target_h = int(orig_w * scale), int(orig_h * scale)
        elif width is not None and height is not None:
            target_w, target_h = width, height
        elif width is not None:
            # Seitenverhältnis beibehalten
            target_w = width
            target_h = int(orig_h * (width / orig_w))
        elif height is not None:
            # Seitenverhältnis beibehalten
            target_h = height
            target_w = int(orig_w * (height / orig_h))
        else:
            return image  # Keine Änderung
        
        # OpenCV resize verwenden
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    def _blur_opencv(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Führt Gaußsche Unschärfe mit OpenCV durch.
        
        Args:
            image: Eingabebild als NumPy-Array
            kernel_size: Größe des Unschärfe-Kernels
            
        Returns:
            np.ndarray: Unscharfes Bild
        """
        # Stelle sicher, dass Kernelgröße ungerade ist
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # OpenCV GaussianBlur verwenden
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _edge_detection_opencv(self, image: np.ndarray) -> np.ndarray:
        """
        Führt Kantendetektion mit OpenCV durch (Canny-Algorithmus).
        
        Args:
            image: Eingabebild als NumPy-Array
            
        Returns:
            np.ndarray: Bild mit hervorgehobenen Kanten
        """
        # Umwandlung in Graustufen, falls erforderlich
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Canny-Kantendetektor anwenden
        edges = cv2.Canny(gray, 100, 200)
        
        # Wenn Originalbild Farbkanäle hat, Ergebnis entsprechend anpassen
        if len(image.shape) == 3:
            result = np.stack([edges] * 3, axis=2)
        else:
            result = edges
        
        return result
