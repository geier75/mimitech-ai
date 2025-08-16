#!/usr/bin/env python3
"""
Testskript für den VX-VISION BatchProcessor
"""
import sys
import os
import time
import numpy as np
import logging
from datetime import datetime

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VX-VISION-TEST")

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importiere die VX-VISION-Komponenten
from vxor.vision.batch import BatchProcessor, BatchConfig
from vxor.vision.batch.scheduler import AdaptiveBatchScheduler
from vxor.vision.hardware.detector import HardwareDetector
from vxor.vision.batch.memory_manager import MemoryManager

def create_synthetic_images(num_images=100, width=224, height=224, channels=3):
    """Erstellt synthetische Bilder für den Test."""
    logger.info(f"Erstelle {num_images} synthetische Bilder ({width}x{height}x{channels})...")
    return [np.random.rand(height, width, channels).astype(np.float32) for _ in range(num_images)]

def resize_operation(image, target_size=(112, 112)):
    """Beispieloperation: Bildgröße ändern."""
    h, w = image.shape[:2]
    scale_w, scale_h = target_size[0] / w, target_size[1] / h
    return np.array([[image[int(i/scale_h)][int(j/scale_w)] 
                      for j in range(target_size[0])] 
                      for i in range(target_size[1])])

def normalize_operation(image):
    """Beispieloperation: Bild normalisieren."""
    return (image - image.mean()) / (image.std() + 1e-8)

def edge_detection_operation(image):
    """Beispieloperation: Einfache Kantenerkennung."""
    # Vereinfachter Sobel-Operator
    if len(image.shape) > 2 and image.shape[2] > 1:
        # Wenn es ein Farbbild ist, konvertieren wir es zu Graustufen
        gray = np.mean(image, axis=2)
    else:
        gray = image
        
    h, w = gray.shape
    result = np.zeros((h-2, w-2))
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            gx = gray[i-1, j+1] + 2*gray[i, j+1] + gray[i+1, j+1] - \
                 gray[i-1, j-1] - 2*gray[i, j-1] - gray[i+1, j-1]
            gy = gray[i-1, j-1] + 2*gray[i-1, j] + gray[i-1, j+1] - \
                 gray[i+1, j-1] - 2*gray[i+1, j] - gray[i+1, j+1]
            result[i-1, j-1] = min(255, np.sqrt(gx*gx + gy*gy))
    
    return result

def main():
    try:
        # Hardware-Informationen abrufen
        hardware_detector = HardwareDetector()
        hardware_info = hardware_detector.detect_hardware()
        preferred_device = hardware_detector.get_preferred_device()
        
        logger.info(f"Hardware-Info: {hardware_info}")
        logger.info(f"Bevorzugtes Gerät: {preferred_device}")
        
        # Memory Manager initialisieren
        memory_manager = MemoryManager()
        
        # BatchConfig initialisieren
        config = BatchConfig(
            max_batch_size=32,
            min_batch_size=4,
            memory_fraction=0.7,
            use_mixed_precision=True,
            device_preference=preferred_device
        )
        
        # BatchScheduler initialisieren
        scheduler = AdaptiveBatchScheduler(config=config, memory_manager=memory_manager)
        
        # BatchProcessor initialisieren
        processor = BatchProcessor(
            config=config, 
            scheduler=scheduler,
            memory_manager=memory_manager
        )
        
        # Synthetische Bilder erstellen
        test_sizes = [
            (10, 224, 224, 3),   # Klein
            (50, 512, 512, 3),   # Mittel
            (100, 1024, 1024, 3) # Groß
        ]
        
        operations = [
            resize_operation,
            normalize_operation,
            edge_detection_operation
        ]
        
        # Tests durchführen
        for num_images, width, height, channels in test_sizes:
            logger.info(f"=== Test mit {num_images} Bildern ({width}x{height}x{channels}) ===")
            
            # Bilder erstellen
            images = create_synthetic_images(num_images, width, height, channels)
            
            # Speicherverbrauch vorher
            mem_before = memory_manager.get_memory_usage()
            logger.info(f"Speicherverbrauch vor der Verarbeitung: {mem_before / 1024**2:.2f} MB")
            
            # Zeit messen
            start_time = time.time()
            
            # Batch-Größe bestimmen
            optimal_batch_size = scheduler.determine_optimal_batch_size(images, operations)
            logger.info(f"Optimale Batch-Größe: {optimal_batch_size}")
            
            # Batch verarbeiten
            results = processor.process_batch(images, operations, batch_size=optimal_batch_size)
            
            # Zeit stoppen
            elapsed_time = time.time() - start_time
            
            # Speicherverbrauch nachher
            mem_after = memory_manager.get_memory_usage()
            logger.info(f"Speicherverbrauch nach der Verarbeitung: {mem_after / 1024**2:.2f} MB")
            
            # Ergebnisse ausgeben
            logger.info(f"Verarbeitungszeit: {elapsed_time:.4f} Sekunden")
            logger.info(f"Bilder pro Sekunde: {num_images / elapsed_time:.2f}")
            logger.info(f"Anzahl der Ergebnisbilder: {len(results)}")
            
            # Test für async Mode
            logger.info("=== Async Mode Test ===")
            start_time = time.time()
            async_results = processor.process_batch(images, operations, batch_size=optimal_batch_size, async_mode=True)
            # Warte auf Fertigstellung (in der Praxis könnten wir hier andere Aufgaben ausführen)
            while not processor.is_processing_complete():
                time.sleep(0.01)
            async_results = processor.get_results()
            elapsed_time = time.time() - start_time
            logger.info(f"Async Verarbeitungszeit: {elapsed_time:.4f} Sekunden")
            logger.info(f"Async Bilder pro Sekunde: {num_images / elapsed_time:.2f}")
            logger.info(f"Anzahl der async Ergebnisbilder: {len(async_results)}")
            
        logger.info("Alle Tests abgeschlossen!")
            
    except Exception as e:
        logger.error(f"Fehler während des Tests: {str(e)}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
