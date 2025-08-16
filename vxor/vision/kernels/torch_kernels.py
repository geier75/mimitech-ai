#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION PyTorch-Kernels

Optimierte Bildverarbeitungs-Kernels mit PyTorch für CUDA, MPS und ROCm.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import math

from vxor.vision.kernels.common import (
    KernelOperation, KernelType, register_kernel, kernel_registry
)

# Logger konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VX-VISION.kernels.torch")

# Versuche, PyTorch zu importieren
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    HAS_TORCH = True
    logger.info("PyTorch erfolgreich importiert für optimierte Bildverarbeitung")
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch nicht verfügbar. Fallback zu NumPy-Implementierungen wird verwendet.")

def _get_torch_device():
    """Ermittelt das beste verfügbare PyTorch-Gerät."""
    if not HAS_TORCH:
        return None
        
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def _np_to_torch(images: List[np.ndarray], device=None) -> List[torch.Tensor]:
    """Konvertiert NumPy-Arrays zu PyTorch-Tensoren."""
    if not HAS_TORCH:
        return images
    
    if device is None:
        device = _get_torch_device()
    
    # Für einzelnes Bild
    if isinstance(images, np.ndarray):
        return torch.from_numpy(images).to(device)
    
    # Für Listen von Bildern
    return [torch.from_numpy(img).to(device) for img in images]

def _torch_to_np(tensors: List[torch.Tensor]) -> List[np.ndarray]:
    """Konvertiert PyTorch-Tensoren zurück zu NumPy-Arrays."""
    if not HAS_TORCH:
        return tensors
    
    # Für einzelnes Bild
    if isinstance(tensors, torch.Tensor):
        return tensors.detach().cpu().numpy()
    
    # Für Listen von Bildern
    return [tensor.detach().cpu().numpy() for tensor in tensors]

# Hier beginnen die eigentlichen Kernel-Implementierungen

@register_kernel(KernelOperation.RESIZE, KernelType.TORCH)
def torch_resize(images: List[np.ndarray], width: int = None, height: int = None, scale: float = None) -> List[np.ndarray]:
    """
    Ändert die Größe einer Liste von Bildern mit PyTorch-Optimierungen.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        width: Zielbreite (wenn None, wird über scale berechnet)
        height: Zielhöhe (wenn None, wird über scale berechnet)
        scale: Skalierungsfaktor (wenn width und height None sind)
        
    Returns:
        Liste der größengeänderten Bilder
    """
    if not HAS_TORCH:
        logger.warning("PyTorch nicht verfügbar für Größenänderung. Fallback wird verwendet.")
        fallback = kernel_registry.get(KernelOperation.RESIZE.value, KernelType.NUMPY.value)
        if fallback:
            return fallback(images, width=width, height=height, scale=scale)
        raise RuntimeError("Kein Fallback für Größenänderung verfügbar")
    
    device = _get_torch_device()
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
        
        # Umwandlung zu PyTorch-Tensor
        # Achsenumordnung für PyTorch: HxWxC -> CxHxW
        if len(img.shape) == 3:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().to(device)
        else:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
        
        # Größenänderung mit PyTorch
        resized_tensor = F.interpolate(
            img_tensor.unsqueeze(0),  # Batch-Dimension hinzufügen
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Batch-Dimension entfernen
        
        # Zurück zu NumPy konvertieren
        if len(img.shape) == 3:
            # CxHxW -> HxWxC
            resized_np = resized_tensor.cpu().numpy().transpose(1, 2, 0)
        else:
            resized_np = resized_tensor.cpu().numpy().squeeze(0)
        
        result.append(resized_np)
    
    return result

@register_kernel(KernelOperation.BLUR, KernelType.TORCH)
def torch_blur(images: List[np.ndarray], kernel_size: int = 5, sigma: float = 1.0) -> List[np.ndarray]:
    """
    Wendet einen Gaußschen Weichzeichner auf eine Liste von Bildern an mit PyTorch-Optimierungen.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        kernel_size: Größe des Gaußschen Kernels (ungerade Zahl)
        sigma: Standardabweichung des Gaußschen Kernels
        
    Returns:
        Liste der weichgezeichneten Bilder
    """
    if not HAS_TORCH:
        logger.warning("PyTorch nicht verfügbar für Weichzeichnung. Fallback wird verwendet.")
        fallback = kernel_registry.get(KernelOperation.BLUR.value, KernelType.NUMPY.value)
        if fallback:
            return fallback(images, kernel_size=kernel_size, sigma=sigma)
        raise RuntimeError("Kein Fallback für Weichzeichnung verfügbar")
    
    # Sicherstellen, dass kernel_size ungerade ist
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    device = _get_torch_device()
    result = []
    
    for img in images:
        # Umwandlung zu PyTorch-Tensor
        if len(img.shape) == 3:
            # HxWxC -> CxHxW
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().to(device)
        else:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
        
        # Batch-Dimension hinzufügen für die Verarbeitung
        img_tensor = img_tensor.unsqueeze(0)
        
        # Gaußsche Weichzeichnung mit PyTorch
        channels = img_tensor.shape[1]
        blurred_tensor = img_tensor
        
        # Bei mehreren Kanälen jeden Kanal separat verarbeiten
        if channels > 1:
            blurred_channels = []
            for c in range(channels):
                channel = img_tensor[:, c:c+1]
                blurred_channel = TF.gaussian_blur(
                    channel,
                    kernel_size=[kernel_size, kernel_size],
                    sigma=[sigma, sigma]
                )
                blurred_channels.append(blurred_channel)
            blurred_tensor = torch.cat(blurred_channels, dim=1)
        else:
            blurred_tensor = TF.gaussian_blur(
                img_tensor,
                kernel_size=[kernel_size, kernel_size],
                sigma=[sigma, sigma]
            )
        
        # Batch-Dimension entfernen
        blurred_tensor = blurred_tensor.squeeze(0)
        
        # Zurück zu NumPy konvertieren
        if len(img.shape) == 3:
            # CxHxW -> HxWxC
            blurred_np = blurred_tensor.cpu().numpy().transpose(1, 2, 0)
        else:
            blurred_np = blurred_tensor.cpu().numpy().squeeze(0)
        
        result.append(blurred_np)
    
    return result

@register_kernel(KernelOperation.NORMALIZE, KernelType.TORCH)
def torch_normalize(images: List[np.ndarray], mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> List[np.ndarray]:
    """
    Normalisiert eine Liste von Bildern mit PyTorch-Optimierungen.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        mean: Mittelwert für die Normalisierung (pro Kanal)
        std: Standardabweichung für die Normalisierung (pro Kanal)
        
    Returns:
        Liste der normalisierten Bilder
    """
    if not HAS_TORCH:
        logger.warning("PyTorch nicht verfügbar für Normalisierung. Fallback wird verwendet.")
        fallback = kernel_registry.get(KernelOperation.NORMALIZE.value, KernelType.NUMPY.value)
        if fallback:
            return fallback(images, mean=mean, std=std)
        raise RuntimeError("Kein Fallback für Normalisierung verfügbar")
    
    device = _get_torch_device()
    result = []
    
    for img in images:
        # Umwandlung zu PyTorch-Tensor
        if len(img.shape) == 3:
            # HxWxC -> CxHxW
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().to(device)
            channels = img.shape[2]
        else:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
            channels = 1
        
        # Standardisierung (Mittelwert 0, Standardabweichung 1)
        if mean is None and std is None:
            # Berechne Mittelwert und Standardabweichung pro Kanal
            img_mean = [img_tensor[c].mean().item() for c in range(channels)]
            img_std = [img_tensor[c].std().item() for c in range(channels)]
            
            # Sicherstellen, dass std > 0
            img_std = [max(s, 1e-8) for s in img_std]
            
            # Normalisieren
            normalized_tensor = TF.normalize(img_tensor, img_mean, img_std)
        else:
            # Mit gegebenen Mittelwerten und Standardabweichungen
            if mean is None:
                mean = [0.0] * channels
            if std is None:
                std = [1.0] * channels
                
            # Sicherstellen, dass std > 0
            std = [max(s, 1e-8) for s in std]
            
            # Normalisieren
            normalized_tensor = TF.normalize(img_tensor, mean[:channels], std[:channels])
        
        # Zurück zu NumPy konvertieren
        if len(img.shape) == 3:
            # CxHxW -> HxWxC
            normalized_np = normalized_tensor.cpu().numpy().transpose(1, 2, 0)
        else:
            normalized_np = normalized_tensor.cpu().numpy().squeeze(0)
        
        result.append(normalized_np)
    
    return result

@register_kernel(KernelOperation.EDGE_DETECTION, KernelType.TORCH)
def torch_edge_detection(images: List[np.ndarray], method: str = "sobel") -> List[np.ndarray]:
    """
    Wendet Kantenerkennung auf eine Liste von Bildern an mit PyTorch-Optimierungen.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        method: Methode der Kantenerkennung ('sobel', 'prewitt', 'scharr')
        
    Returns:
        Liste der Bilder mit hervorgehobenen Kanten
    """
    if not HAS_TORCH:
        logger.warning("PyTorch nicht verfügbar für Kantenerkennung. Fallback wird verwendet.")
        fallback = kernel_registry.get(KernelOperation.EDGE_DETECTION.value, KernelType.NUMPY.value)
        if fallback:
            return fallback(images, method=method)
        raise RuntimeError("Kein Fallback für Kantenerkennung verfügbar")
    
    # Kernels für verschiedene Kantenerkennungsmethoden
    kernels = {
        "sobel": {
            "x": torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32),
            "y": torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        },
        "prewitt": {
            "x": torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32),
            "y": torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        },
        "scharr": {
            "x": torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32),
            "y": torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)
        }
    }
    
    # Methode validieren
    if method not in kernels:
        logger.warning(f"Unbekannte Kantenerkennungsmethode: {method}. Verwende 'sobel' stattdessen.")
        method = "sobel"
    
    device = _get_torch_device()
    result = []
    
    # Kernel vorbereiten
    kernel_x = kernels[method]["x"].to(device).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)
    kernel_y = kernels[method]["y"].to(device).unsqueeze(0).unsqueeze(0)
    
    for img in images:
        # Zu Graustufen konvertieren, falls Farbbild
        if len(img.shape) > 2:
            # Einfache Graustufen-Umwandlung: Durchschnitt der Kanäle
            gray = np.mean(img, axis=2).astype(np.float32)
        else:
            gray = img.astype(np.float32)
        
        # Umwandlung zu PyTorch-Tensor
        img_tensor = torch.from_numpy(gray).float().to(device)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        
        # Anwenden der Faltungsoperationen
        gradient_x = F.conv2d(img_tensor, kernel_x, padding=1)
        gradient_y = F.conv2d(img_tensor, kernel_y, padding=1)
        
        # Gradientenbetrag berechnen
        gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalisieren auf [0, 1]
        max_mag = torch.max(gradient_magnitude)
        if max_mag > 0:
            normalized = gradient_magnitude / max_mag
        else:
            normalized = gradient_magnitude
        
        # Zurück zu NumPy konvertieren
        edges_np = normalized.squeeze().cpu().numpy()
        
        result.append(edges_np)
    
    return result

@register_kernel(KernelOperation.ROTATE, KernelType.TORCH)
def torch_rotate(images: List[np.ndarray], angle: float = 0.0) -> List[np.ndarray]:
    """
    Rotiert eine Liste von Bildern mit PyTorch-Optimierungen.
    
    Args:
        images: Liste von Bildern als NumPy-Arrays
        angle: Rotationswinkel in Grad (im Uhrzeigersinn)
        
    Returns:
        Liste der rotierten Bilder
    """
    if not HAS_TORCH:
        logger.warning("PyTorch nicht verfügbar für Rotation. Fallback wird verwendet.")
        fallback = kernel_registry.get(KernelOperation.ROTATE.value, KernelType.NUMPY.value)
        if fallback:
            return fallback(images, angle=angle)
        raise RuntimeError("Kein Fallback für Rotation verfügbar")
    
    device = _get_torch_device()
    result = []
    
    for img in images:
        # Umwandlung zu PyTorch-Tensor
        if len(img.shape) == 3:
            # HxWxC -> CxHxW
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().to(device)
        else:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
        
        # TorchVision erwartet Bilder im Format (C, H, W)
        # PyTorch erwartet positive Winkel für Drehung gegen den Uhrzeigersinn, daher das Vorzeichen umkehren
        rotated_tensor = TF.rotate(img_tensor, -angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        # Zurück zu NumPy konvertieren
        if len(img.shape) == 3:
            # CxHxW -> HxWxC
            rotated_np = rotated_tensor.cpu().numpy().transpose(1, 2, 0)
        else:
            rotated_np = rotated_tensor.cpu().numpy().squeeze(0)
        
        result.append(rotated_np)
    
    return result
