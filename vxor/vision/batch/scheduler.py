#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION AdaptiveBatchScheduler

Diese Komponente bestimmt dynamisch die optimale Batch-Größe für Bildverarbeitungsoperationen
basierend auf verfügbarer Hardware, Bildgrößen, Operationskomplexität und historischen Performance-Daten.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import numpy as np
from collections import defaultdict

# Logger konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VX-VISION.scheduler")


class AdaptiveBatchScheduler:
    """
    Intelligenter Scheduler für optimale Batch-Größen in der Bildverarbeitung.
    
    Der Scheduler analysiert Hardware-Ressourcen, Bildparameter und Operationskomplexität, 
    um die effizienteste Batch-Größe für maximalen Durchsatz zu ermitteln.
    """
    
    def __init__(self, hardware_detector, config):
        """
        Initialisiert den AdaptiveBatchScheduler.
        
        Args:
            hardware_detector: Instanz des HardwareDetector
            config: Batch-Verarbeitungskonfiguration
        """
        self.hardware = hardware_detector
        self.config = config
        
        # Speicherung historischer Performance-Daten für Optimierung
        self.performance_history = defaultdict(list)
        self.optimal_batch_sizes = {}
        
        # Komplexitätsbewertungen für verschiedene Operationstypen
        self.operation_complexity = {
            # Grundlegende Operationen
            'resize': 1.0,
            'scale': 0.8,
            'rotate': 1.2,
            'crop': 0.7,
            'flip': 0.5,
            'normalize': 0.5,
            
            # Filteroperationen
            'blur': 1.5,
            'sharpen': 1.5,
            'edge_detection': 2.0,
            'gaussian_blur': 1.8,
            'median_filter': 2.2,
            
            # Farbtransformationen
            'grayscale': 0.7,
            'hsv_convert': 1.0,
            'rgb_to_bgr': 0.3,
            'adjust_contrast': 1.0,
            'adjust_brightness': 0.8,
            
            # Komplexe Operationen
            'convolution': 3.0,
            'deep_model_inference': 5.0,
            'object_detection': 4.5,
            'segmentation': 6.0,
            'feature_extraction': 3.5,
            
            # Standard für unbekannte Operationen
            'default': 2.0
        }
        
    def determine_optimal_batch_size(self, 
                                    images: List[np.ndarray], 
                                    operations: List) -> int:
        """
        Bestimmt die optimale Batch-Größe basierend auf Hardware und Eingaben.
        
        Args:
            images: Liste von Bildern als NumPy-Arrays
            operations: Liste von anzuwendenden Operationen
            
        Returns:
            int: Optimale Batch-Größe
        """
        # Wenn keine dynamische Größenanpassung gewünscht, maximale Größe zurückgeben
        if not self.config.dynamic_sizing:
            return self.config.max_batch_size
            
        # Bildkomplexität analysieren (Größe, Kanäle, etc.)
        image_complexity = self._calculate_image_complexity(images)
        
        # Operationskomplexität analysieren
        operation_complexity = self._calculate_operation_complexity(operations)
        
        # Schlüssel für Caching der berechneten Batch-Größen
        # Füge den Gerätetyp zum Cache-Schlüssel hinzu für genauere Zuordnung
        device = self.hardware.get_preferred_device()
        complexity_key = f"{device}_{image_complexity:.2f}_{operation_complexity:.2f}"
        
        # Prüfen, ob bereits eine optimale Größe für diese Komplexität berechnet wurde
        if complexity_key in self.optimal_batch_sizes:
            batch_size = self.optimal_batch_sizes[complexity_key]
            logger.debug(f"Verwendung zwischengespeicherter Batch-Größe: {batch_size}")
            return batch_size
            
        # Verfügbare Hardware-Ressourcen ermitteln
        available_memory = self.hardware.get_available_memory()
        compute_units = self.hardware.get_compute_units()
        
        # Speicherbasierte Batch-Größe berechnen
        memory_per_image = self._estimate_memory_per_image(images[0], operations)
        memory_based_size = int(available_memory * self.config.memory_fraction / memory_per_image)
        
        # Berechnungseinheiten-basierte Batch-Größe
        compute_factor = self._get_compute_factor(device)
        compute_based_size = int(compute_units * compute_factor)
        
        # Komplexitätsbasierte Anpassung
        complexity_factor = 1.0 / (image_complexity * operation_complexity)
        
        # Basis-Batch-Größe berechnen
        base_batch_size = int(min(memory_based_size, compute_based_size) * complexity_factor)
        
        # Hardware-spezifische Anpassungen
        if self.hardware.has_neural_engine():
            # Größere Batches für Neural Engine
            batch_size = min(max(32, base_batch_size), self.config.max_batch_size)
        elif self.hardware.has_gpu():
            # Mittlere Batches für GPU
            batch_size = min(max(16, base_batch_size), self.config.max_batch_size)
        else:
            # Kleinere Batches für CPU
            batch_size = min(max(8, base_batch_size), self.config.max_batch_size)
        
        # Operationstyp berücksichtigen für zusätzliche Anpassungen
        if operation_complexity > 3.0:  # Sehr komplexe Operationen
            batch_size = max(1, batch_size // 4)
        elif operation_complexity > 2.0:  # Komplexe Operationen
            batch_size = max(1, batch_size // 2)
        elif operation_complexity < 0.5:  # Sehr einfache Operationen
            batch_size = min(batch_size * 2, self.config.max_batch_size)
        
        # Performance-History berücksichtigen, falls verfügbar
        device_key = f"device_{device}"
        if device_key in self.performance_history and len(self.performance_history[device_key]) >= 3:
            # Genügend Daten für eine historische Anpassung
            historical_adjustment = self._calculate_historical_adjustment(device_key)
            batch_size = int(batch_size * historical_adjustment)
            logger.debug(f"Historische Anpassung: {historical_adjustment:.2f}")
        
        # Innerhalb der konfigurierten Grenzen bleiben
        batch_size = max(self.config.min_batch_size, min(batch_size, self.config.max_batch_size))
        
        # Ergebnis zwischenspeichern
        self.optimal_batch_sizes[complexity_key] = batch_size
        
        logger.info(f"Optimale Batch-Größe bestimmt: {batch_size} " +
                   f"(Bild: {image_complexity:.2f}, Op: {operation_complexity:.2f}, Gerät: {device})")
        
        return batch_size
        
    def update_performance_metrics(self, 
                                  num_images: int, 
                                  device: str, 
                                  processing_time: float):
        """
        Aktualisiert die Performance-Metriken für zukünftige Batch-Größenoptimierung.
        
        Args:
            num_images: Anzahl der verarbeiteten Bilder
            device: Verwendetes Gerät
            processing_time: Verarbeitungszeit in Sekunden
        """
        if processing_time <= 0:
            return  # Ungültige Zeitmessung
            
        throughput = num_images / processing_time
        
        # Performance-Daten nach Gerät organisieren
        device_key = f"device_{device}"
        
        # Nur die letzten 20 Messungen aufbewahren
        history = self.performance_history[device_key]
        if len(history) >= 20:
            history.pop(0)
            
        history.append({
            'batch_size': num_images,
            'processing_time': processing_time,
            'throughput': throughput,
            'timestamp': time.time()
        })
        
        logger.debug(f"Performance-Metrik aktualisiert: Gerät={device}, " +
                   f"Batch-Größe={num_images}, Durchsatz={throughput:.2f} Bilder/s")
        
    def _calculate_image_complexity(self, images: List[np.ndarray]) -> float:
        """
        Berechnet die Komplexität der Bilder basierend auf Größe und Kanälen.
        
        Args:
            images: Liste von Bildern
            
        Returns:
            float: Komplexitätswert (höher bedeutet komplexer)
        """
        if not images:
            return 1.0
            
        # Stichprobe von Bildern nehmen (max. 10)
        sample = images[:min(10, len(images))]
        
        # Durchschnittliche Bildgröße in Megapixeln
        total_pixels = 0
        total_channels = 0
        
        for img in sample:
            if img is None or not hasattr(img, 'shape'):
                continue
                
            # Bildgröße berechnen
            if len(img.shape) >= 2:
                pixels = img.shape[0] * img.shape[1]
                total_pixels += pixels
                
                # Kanäle berücksichtigen
                channels = img.shape[2] if len(img.shape) > 2 else 1
                total_channels += channels
        
        if len(sample) == 0:
            return 1.0
            
        avg_megapixels = (total_pixels / len(sample)) / 1e6
        avg_channels = total_channels / len(sample)
        
        # Komplexitätsformel:
        # - Größere Bilder sind komplexer (linear mit Megapixeln)
        # - Mehr Kanäle erhöhen die Komplexität (leicht)
        complexity = avg_megapixels * (0.8 + 0.2 * avg_channels / 3)
        
        # Normalisieren auf einen vernünftigen Bereich
        # Annahme: 1.0 entspricht einem 1-Megapixel-RGB-Bild
        return max(0.1, min(complexity, 10.0))
        
    def _calculate_operation_complexity(self, operations: List) -> float:
        """
        Berechnet die Komplexität der Operationen.
        
        Args:
            operations: Liste von Operationen
            
        Returns:
            float: Komplexitätswert (höher bedeutet komplexer)
        """
        if not operations:
            return 1.0
            
        total_complexity = 1.0
        
        for op in operations:
            op_complexity = self._get_operation_complexity(op)
            # Multiplikative Kombination, da Operationen sequentiell sind
            total_complexity *= op_complexity
            
        # Normalisieren, um extreme Werte zu vermeiden
        # Annahme: 1.0 ist Standardkomplexität, 5.0 ist sehr komplex
        return max(0.2, min(total_complexity, 5.0))
        
    def _get_operation_complexity(self, operation: Any) -> float:
        """
        Gibt die Komplexitätsbewertung für eine bestimmte Operation zurück.
        
        Args:
            operation: Einzelne Operation (Funktion oder Konfigurationsobjekt)
            
        Returns:
            float: Komplexitätsbewertung
        """
        if callable(operation):
            # Funktion: Versuchen, den Namen zu extrahieren
            op_name = operation.__name__.lower()
            
            # Nach Schlüsselwörtern im Funktionsnamen suchen
            for key, value in self.operation_complexity.items():
                if key in op_name:
                    return value
                    
            return self.operation_complexity['default']
            
        elif isinstance(operation, dict):
            # Wörterbuch: Nach 'name' oder 'type' suchen
            op_name = operation.get('name', operation.get('type', '')).lower()
            
            # Direkte Übereinstimmung in der Komplexitätstabelle
            if op_name in self.operation_complexity:
                return self.operation_complexity[op_name]
                
            # Teilweise Übereinstimmung
            for key, value in self.operation_complexity.items():
                if key in op_name:
                    return value
        
        # Standard, wenn keine Übereinstimmung gefunden wird
        return self.operation_complexity['default']
        
    def _estimate_memory_per_image(self, 
                                  image: np.ndarray, 
                                  operations: List) -> int:
        """
        Schätzt den Speicherbedarf pro Bild in Bytes.
        
        Args:
            image: Beispielbild
            operations: Anzuwendende Operationen
            
        Returns:
            int: Geschätzter Speicherbedarf in Bytes
        """
        if image is None or not hasattr(image, 'shape') or not hasattr(image, 'dtype'):
            # Standardwert, wenn kein gültiges Bild
            return 10 * 1024 * 1024  # 10 MB als Standardwert
            
        # Grundlegender Speicherbedarf des Bildes
        element_size = image.itemsize
        num_elements = np.prod(image.shape)
        base_memory = element_size * num_elements
        
        # Operationskomplexität-basierter Overhead
        op_complexity = self._calculate_operation_complexity(operations)
        
        # Overhead für Zwischenergebnisse und temporäre Puffer
        # Komplexere Operationen erfordern mehr Zwischenspeicher
        memory_overhead = 1.0 + 0.5 * op_complexity
        
        # Zusätzlicher Hardware-spezifischer Overhead
        device = self.hardware.get_preferred_device()
        if device in ['cuda', 'rocm']:
            # GPUs haben höhere Overheads für Transfers und Alignment
            memory_overhead += 0.5
        elif device == 'ane':
            # Apple Neural Engine hat spezifischen Overhead
            memory_overhead += 0.3
            
        total_memory = int(base_memory * memory_overhead)
        
        # Mixed Precision reduziert den Speicherbedarf
        if self.config.use_mixed_precision:
            total_memory = int(total_memory * 0.6)  # 40% Reduktion durch FP16 statt FP32
            
        logger.debug(f"Geschätzter Speicherbedarf pro Bild: {total_memory / (1024*1024):.2f} MB")
        
        return total_memory
        
    def _get_compute_factor(self, device: str) -> float:
        """
        Gibt einen Multiplikator basierend auf dem Gerätetyp zurück.
        
        Args:
            device: Gerätetyp ('ane', 'cuda', 'mps', 'rocm', 'cpu')
            
        Returns:
            float: Multiplikator für Compute-Units
        """
        factors = {
            'ane': 8.0,   # Apple Neural Engine ist sehr effizient bei Batches
            'cuda': 6.0,  # NVIDIA GPUs profitieren stark von großen Batches
            'mps': 4.0,   # Metal Performance Shaders (Apple GPUs)
            'rocm': 4.0,  # AMD GPUs
            'cpu': 1.0    # CPUs haben begrenzten Parallelismus
        }
        
        return factors.get(device, 2.0)
        
    def _calculate_historical_adjustment(self, device_key: str) -> float:
        """
        Berechnet einen Anpassungsfaktor basierend auf historischen Performance-Daten.
        
        Args:
            device_key: Schlüssel für das Gerät in der Performance-History
            
        Returns:
            float: Anpassungsfaktor (> 1.0 erhöht Batch-Größe, < 1.0 verringert sie)
        """
        history = self.performance_history[device_key]
        
        if not history:
            return 1.0
            
        # Neuere Messungen stärker gewichten
        weighted_history = []
        total_weight = 0.0
        
        for i, entry in enumerate(history):
            # Gewichtung: Neuere Einträge sind wichtiger
            weight = 0.5 + 0.5 * (i / len(history))
            weighted_history.append((entry, weight))
            total_weight += weight
            
        # Sammeln der Batch-Größen und Durchsätze
        data_points = []
        
        for entry, weight in weighted_history:
            batch_size = entry['batch_size']
            throughput = entry['throughput']
            
            # Normalisieren auf Bilder pro Sekunde pro Bild
            normalized_throughput = throughput / batch_size
            
            # Gewichteten Datenpunkt hinzufügen
            data_points.append((batch_size, normalized_throughput, weight))
            
        # Optimale Batch-Größe finden (höchster gewichteter Durchsatz)
        best_batch_size = None
        best_throughput = 0.0
        
        for batch_size, throughput, weight in data_points:
            weighted_throughput = throughput * weight / total_weight
            
            if weighted_throughput > best_throughput:
                best_throughput = weighted_throughput
                best_batch_size = batch_size
                
        if best_batch_size is None:
            return 1.0
            
        # Aktuelle durchschnittliche Batch-Größe berechnen
        avg_batch_size = sum(entry['batch_size'] for entry in history) / len(history)
        
        # Anpassungsfaktor berechnen
        adjustment = best_batch_size / avg_batch_size if avg_batch_size > 0 else 1.0
        
        # Begrenzen der Anpassung, um extreme Schwankungen zu vermeiden
        return max(0.5, min(adjustment, 2.0))


if __name__ == "__main__":
    # Einfacher Test
    from vxor.vision.hardware.detector import HardwareDetector
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        min_batch_size = 1
        max_batch_size = 64
        dynamic_sizing = True
        memory_fraction = 0.7
        use_mixed_precision = True
    
    hardware = HardwareDetector()
    config = TestConfig()
    scheduler = AdaptiveBatchScheduler(hardware, config)
    
    # Testbilder
    test_images = [np.random.rand(480, 640, 3) for _ in range(5)]
    
    # Testoperationen
    test_operations = [
        {"name": "resize", "width": 224, "height": 224},
        {"name": "normalize"},
        {"name": "convolution"}
    ]
    
    # Optimale Batch-Größe berechnen
    batch_size = scheduler.determine_optimal_batch_size(test_images, test_operations)
    print(f"Optimale Batch-Größe: {batch_size}")
    
    # Performance-Metrik simulieren
    scheduler.update_performance_metrics(batch_size, hardware.get_preferred_device(), 0.5)
    print("Performance-Metrik aktualisiert.")
