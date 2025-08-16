#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION Kernel Test

Test der optimierten Bildverarbeitungskernels für verschiedene Backends (MLX, PyTorch, NumPy).
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VX-VISION-KERNEL-TEST")

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importiere die VX-VISION-Komponenten
from vxor.vision.kernels.common import (
    KernelOperation, KernelType, kernel_registry, benchmark_kernel,
    list_available_operations, list_available_backends, get_best_kernel
)
from vxor.vision.hardware.detector import HardwareDetector

# Lade Testbilder oder erstelle synthetische Bilder
def create_test_images(num_images=5, width=512, height=512, channels=3):
    """Erstellt synthetische Bilder für den Test."""
    logger.info(f"Erstelle {num_images} synthetische Bilder ({width}x{height}x{channels})...")
    
    # Verschiedene Testmuster erstellen
    images = []
    
    # Bild 1: Schachbrettmuster
    checkerboard = np.zeros((height, width, channels), dtype=np.float32)
    square_size = 64
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                checkerboard[i:i+square_size, j:j+square_size, :] = 1.0
    images.append(checkerboard)
    
    # Bild 2: Farbverlauf horizontal
    gradient_h = np.zeros((height, width, channels), dtype=np.float32)
    for i in range(width):
        gradient_h[:, i, 0] = i / width  # Rot-Kanal
        gradient_h[:, i, 1] = 0.5  # Grün-Kanal konstant
        gradient_h[:, i, 2] = 1.0 - i / width  # Blau-Kanal
    images.append(gradient_h)
    
    # Bild 3: Farbverlauf vertikal
    gradient_v = np.zeros((height, width, channels), dtype=np.float32)
    for i in range(height):
        gradient_v[i, :, 0] = 0.5  # Rot-Kanal konstant
        gradient_v[i, :, 1] = i / height  # Grün-Kanal
        gradient_v[i, :, 2] = 1.0 - i / height  # Blau-Kanal
    images.append(gradient_v)
    
    # Bild 4: Kreismuster
    circle = np.zeros((height, width, channels), dtype=np.float32)
    center_x, center_y = width // 2, height // 2
    max_radius = min(width, height) // 2
    for y in range(height):
        for x in range(width):
            radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if radius < max_radius:
                # Normalisierte Radius-Werte für Farbe verwenden
                norm_radius = radius / max_radius
                circle[y, x, 0] = norm_radius  # Rot steigt mit dem Radius
                circle[y, x, 1] = 1.0 - norm_radius  # Grün sinkt mit dem Radius
                circle[y, x, 2] = np.sin(norm_radius * np.pi)  # Blau variiert sinusförmig
    images.append(circle)
    
    # Bild 5: Zufälliges Rauschen
    noise = np.random.rand(height, width, channels).astype(np.float32)
    images.append(noise)
    
    # Weitere Bilder mit Zufallsrauschen, falls benötigt
    for _ in range(num_images - len(images)):
        random_image = np.random.rand(height, width, channels).astype(np.float32)
        images.append(random_image)
    
    return images

# Visualisierungsfunktion für die Testergebnisse
def visualize_results(original_images, processed_images, operation_name, backend_name):
    """Visualisiert die Original- und verarbeiteten Bilder."""
    n_images = min(len(original_images), len(processed_images), 5)  # Maximal 5 Bilder anzeigen
    
    fig, axes = plt.subplots(n_images, 2, figsize=(12, 3*n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"{operation_name} mit {backend_name} Backend", fontsize=16)
    
    for i in range(n_images):
        # Originalbild
        if len(original_images[i].shape) == 3 and original_images[i].shape[2] == 1:
            # Graustufen-Bild mit einer Kanalkomponente
            axes[i, 0].imshow(original_images[i].squeeze(), cmap='gray')
        elif len(original_images[i].shape) == 2:
            # Graustufen-Bild
            axes[i, 0].imshow(original_images[i], cmap='gray')
        else:
            # Farbbild: Werte auf 0-1 normalisieren für die Anzeige
            img_to_show = np.clip(original_images[i], 0, 1)
            axes[i, 0].imshow(img_to_show)
        axes[i, 0].set_title(f"Original {i+1}")
        axes[i, 0].axis('off')
        
        # Verarbeitetes Bild
        if len(processed_images[i].shape) == 3 and processed_images[i].shape[2] == 1:
            # Graustufen-Bild mit einer Kanalkomponente
            axes[i, 1].imshow(processed_images[i].squeeze(), cmap='gray')
        elif len(processed_images[i].shape) == 2:
            # Graustufen-Bild
            axes[i, 1].imshow(processed_images[i], cmap='gray')
        else:
            # Farbbild: Werte auf 0-1 normalisieren für die Anzeige
            img_to_show = np.clip(processed_images[i], 0, 1)
            axes[i, 1].imshow(img_to_show)
        axes[i, 1].set_title(f"{operation_name} {i+1}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    # Speichern der Visualisierung
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{operation_name}_{backend_name}.png")
    plt.savefig(save_path)
    logger.info(f"Visualisierung gespeichert unter: {save_path}")
    
    plt.close()

# Benchmark-Funktion für die Kernels
def benchmark_kernels():
    """Führt Benchmarks für alle verfügbaren Kernels durch."""
    # Hardware-Informationen abrufen
    hardware_detector = HardwareDetector()
    hardware_info = hardware_detector.detect_hardware()
    preferred_device = hardware_detector.get_preferred_device()
    
    logger.info(f"Hardware-Info: {hardware_info}")
    logger.info(f"Bevorzugtes Gerät: {preferred_device}")
    
    # Verfügbare Backends bestimmen
    available_backends = []
    if hardware_info['has_apple_silicon']:
        available_backends.append(KernelType.MLX.value)
    if hardware_info['has_cuda'] or hardware_info['has_mps'] or hardware_info['has_rocm']:
        available_backends.append(KernelType.TORCH.value)
    available_backends.append(KernelType.NUMPY.value)  # Immer verfügbar
    
    logger.info(f"Verfügbare Backends: {available_backends}")
    
    # Testbilder erstellen
    test_images = create_test_images(num_images=5, width=512, height=512, channels=3)
    
    # Verfügbare Operationen
    operations = list_available_operations()
    logger.info(f"Verfügbare Operationen: {operations}")
    
    # Ergebnisse sammeln
    results = {}
    
    # Benchmark für jede Operation und jedes Backend
    for op in operations:
        logger.info(f"Teste Operation: {op}")
        op_results = {}
        
        for backend in list_available_backends(op):
            if backend not in available_backends:
                continue
                
            logger.info(f"  Backend: {backend}")
            
            # Kernel-Funktion holen
            kernel_func = kernel_registry.get(op, backend)
            if kernel_func is None:
                logger.warning(f"Kein Kernel gefunden für {op} und {backend}")
                continue
            
            # Parameter für die Operation
            params = {}
            if op == KernelOperation.RESIZE.value:
                params = {"width": 256, "height": 256}
            elif op == KernelOperation.BLUR.value:
                params = {"kernel_size": 5, "sigma": 1.5}
            elif op == KernelOperation.NORMALIZE.value:
                params = {"mean": None, "std": None}  # Standardisierung
            elif op == KernelOperation.EDGE_DETECTION.value:
                params = {"method": "sobel"}
            elif op == KernelOperation.ROTATE.value:
                params = {"angle": 45.0}
            
            try:
                # Operation ausführen und Ergebnisse messen
                start_time = time.time()
                processed_images = kernel_func(test_images, **params)
                execution_time = time.time() - start_time
                
                avg_time_per_image = execution_time / len(test_images)
                logger.info(f"    Ausführungszeit: {execution_time:.4f}s ({avg_time_per_image:.4f}s pro Bild)")
                
                op_results[backend] = {
                    "execution_time": execution_time,
                    "avg_time_per_image": avg_time_per_image,
                    "processed_images": processed_images
                }
                
                # Visualisiere die Ergebnisse (nur für ein paar Operationen zur Demonstration)
                if op in [KernelOperation.RESIZE.value, KernelOperation.BLUR.value, 
                         KernelOperation.EDGE_DETECTION.value, KernelOperation.ROTATE.value]:
                    visualize_results(test_images, processed_images, op, backend)
                
            except Exception as e:
                logger.error(f"    Fehler bei {op} mit {backend}: {str(e)}")
                op_results[backend] = {
                    "error": str(e)
                }
        
        results[op] = op_results
    
    # Ergebnisse zusammenfassen
    logger.info("\nZusammenfassung der Benchmark-Ergebnisse:")
    logger.info("=" * 80)
    
    for op in operations:
        if op not in results:
            continue
            
        logger.info(f"\nOperation: {op}")
        op_results = results[op]
        
        if not op_results:
            logger.info("  Keine Ergebnisse")
            continue
            
        # Schnellstes Backend ermitteln
        fastest_backend = None
        fastest_time = float('inf')
        
        for backend, backend_results in op_results.items():
            if "error" in backend_results:
                logger.info(f"  {backend}: FEHLER - {backend_results['error']}")
                continue
                
            time_per_image = backend_results["avg_time_per_image"]
            logger.info(f"  {backend}: {time_per_image:.6f}s pro Bild")
            
            if time_per_image < fastest_time:
                fastest_time = time_per_image
                fastest_backend = backend
        
        if fastest_backend:
            logger.info(f"  Schnellstes Backend: {fastest_backend} ({fastest_time:.6f}s pro Bild)")
    
    logger.info("\nBenchmark abgeschlossen!")
    return results

if __name__ == "__main__":
    try:
        logger.info("Starte VX-VISION Kernel-Test...")
        benchmark_kernels()
    except Exception as e:
        logger.error(f"Fehler während des Tests: {str(e)}", exc_info=True)
        sys.exit(1)
        
    sys.exit(0)
