#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VX-VISION Enhanced mit Computer-Use Integration für MISO Ultimate

Erweiterte VX-VISION mit vollständiger Computer-Use-Funktionalität,
Hardware-Erkennung und adaptiver Batch-Verarbeitung.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
import torch
import platform
import psutil
import json
import os

# MISO-Module
from miso.math.t_mathematics.engine import TMathEngine
from miso.vision.computer_use_integration import ComputerUseIntegration, InteractionCommand, InteractionType
from miso.monitoring.ztm_monitor import ZTMMonitor

# Logger konfigurieren
logger = logging.getLogger("MISO.Vision.VXVisionEnhanced")

class HardwareType(Enum):
    """Unterstützte Hardware-Typen"""
    APPLE_NEURAL_ENGINE = "ane"
    APPLE_MPS = "mps"
    NVIDIA_CUDA = "cuda"
    AMD_ROCM = "rocm"
    CPU_ONLY = "cpu"
    INTEL_OPENCL = "opencl"

class ProcessingMode(Enum):
    """Verarbeitungsmodi"""
    REALTIME = "realtime"
    BATCH = "batch"
    ADAPTIVE = "adaptive"
    BACKGROUND = "background"

@dataclass
class HardwareConfig:
    """Hardware-Konfiguration"""
    primary_device: HardwareType
    available_devices: List[HardwareType]
    memory_gb: float
    compute_units: int
    supports_fp16: bool
    supports_int8: bool
    max_batch_size: int

@dataclass
class VisionTask:
    """Vision-Aufgabe"""
    task_id: str
    task_type: str
    input_data: Any
    priority: int
    processing_mode: ProcessingMode
    hardware_preference: Optional[HardwareType] = None
    callback: Optional[callable] = None

class AdaptiveBatchScheduler:
    """Adaptiver Batch-Scheduler für optimale Hardware-Nutzung"""
    
    def __init__(self, hardware_config: HardwareConfig):
        self.hardware_config = hardware_config
        self.task_queue = []
        self.processing_queue = []
        self.completed_tasks = []
        self.scheduler_active = False
        self.scheduler_thread = None
        
        # Performance-Metriken
        self.performance_metrics = {
            'tasks_processed': 0,
            'average_processing_time': 0.0,
            'hardware_utilization': 0.0,
            'batch_efficiency': 0.0
        }
        
        logger.info("Adaptive Batch Scheduler initialisiert")
    
    def add_task(self, task: VisionTask):
        """Füge Aufgabe zur Warteschlange hinzu"""
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        logger.debug(f"Task hinzugefügt: {task.task_id} (Priorität: {task.priority})")
    
    def start_scheduler(self):
        """Starte Scheduler"""
        if not self.scheduler_active:
            self.scheduler_active = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            logger.info("Batch Scheduler gestartet")
    
    def stop_scheduler(self):
        """Stoppe Scheduler"""
        self.scheduler_active = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        logger.info("Batch Scheduler gestoppt")
    
    def _scheduler_loop(self):
        """Haupt-Scheduler-Schleife"""
        while self.scheduler_active:
            try:
                if self.task_queue:
                    # Bestimme optimale Batch-Größe
                    batch_size = self._calculate_optimal_batch_size()
                    
                    # Erstelle Batch
                    batch = self._create_batch(batch_size)
                    
                    if batch:
                        # Verarbeite Batch
                        self._process_batch(batch)
                
                time.sleep(0.1)  # Kurze Pause
                
            except Exception as e:
                logger.error(f"Scheduler-Fehler: {e}")
                time.sleep(1.0)
    
    def _calculate_optimal_batch_size(self) -> int:
        """Berechne optimale Batch-Größe basierend auf Hardware"""
        base_batch_size = self.hardware_config.max_batch_size
        
        # Anpassung basierend auf aktueller Systemlast
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # Reduziere Batch-Größe bei hoher Systemlast
        if cpu_usage > 80 or memory_usage > 85:
            base_batch_size = max(1, base_batch_size // 2)
        elif cpu_usage < 50 and memory_usage < 60:
            base_batch_size = min(self.hardware_config.max_batch_size, base_batch_size * 2)
        
        return min(base_batch_size, len(self.task_queue))
    
    def _create_batch(self, batch_size: int) -> List[VisionTask]:
        """Erstelle Batch aus Warteschlange"""
        batch = []
        
        # Gruppiere ähnliche Tasks für bessere Effizienz
        task_groups = {}
        for task in self.task_queue[:batch_size * 2]:  # Betrachte mehr Tasks für Gruppierung
            task_type = task.task_type
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(task)
        
        # Wähle Tasks aus größter Gruppe zuerst
        for task_type in sorted(task_groups.keys(), key=lambda t: len(task_groups[t]), reverse=True):
            remaining_slots = batch_size - len(batch)
            if remaining_slots <= 0:
                break
            
            tasks_to_add = task_groups[task_type][:remaining_slots]
            batch.extend(tasks_to_add)
            
            # Entferne aus Warteschlange
            for task in tasks_to_add:
                self.task_queue.remove(task)
        
        return batch
    
    def _process_batch(self, batch: List[VisionTask]):
        """Verarbeite Batch von Tasks"""
        start_time = time.time()
        
        try:
            # Gruppiere nach Hardware-Präferenz
            hardware_groups = {}
            for task in batch:
                hw_pref = task.hardware_preference or self.hardware_config.primary_device
                if hw_pref not in hardware_groups:
                    hardware_groups[hw_pref] = []
                hardware_groups[hw_pref].append(task)
            
            # Verarbeite jede Hardware-Gruppe
            for hardware, tasks in hardware_groups.items():
                self._process_hardware_group(hardware, tasks)
            
            # Update Performance-Metriken
            processing_time = time.time() - start_time
            self.performance_metrics['tasks_processed'] += len(batch)
            self.performance_metrics['average_processing_time'] = (
                (self.performance_metrics['average_processing_time'] * (self.performance_metrics['tasks_processed'] - len(batch)) + 
                 processing_time) / self.performance_metrics['tasks_processed']
            )
            
            logger.info(f"Batch verarbeitet: {len(batch)} Tasks in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Batch-Verarbeitungsfehler: {e}")
    
    def _process_hardware_group(self, hardware: HardwareType, tasks: List[VisionTask]):
        """Verarbeite Tasks für spezifische Hardware"""
        try:
            # Simuliere Hardware-spezifische Verarbeitung
            for task in tasks:
                # Hier würde die tatsächliche Verarbeitung stattfinden
                result = self._execute_vision_task(task, hardware)
                
                # Callback ausführen falls vorhanden
                if task.callback:
                    task.callback(result)
                
                # Task als abgeschlossen markieren
                self.completed_tasks.append(task)
                
        except Exception as e:
            logger.error(f"Hardware-Gruppen-Verarbeitungsfehler: {e}")
    
    def _execute_vision_task(self, task: VisionTask, hardware: HardwareType) -> Dict[str, Any]:
        """Führe Vision-Task aus"""
        # Simuliere Verarbeitung basierend auf Task-Typ
        if task.task_type == 'screenshot':
            return {'status': 'completed', 'data': 'screenshot_data'}
        elif task.task_type == 'element_detection':
            return {'status': 'completed', 'elements': []}
        elif task.task_type == 'ocr':
            return {'status': 'completed', 'text': 'extracted_text'}
        else:
            return {'status': 'completed', 'result': 'generic_result'}

class VXVisionEnhanced:
    """Erweiterte VX-VISION mit Computer-Use und Hardware-Optimierung"""
    
    def __init__(self, tmath_engine: Optional[TMathEngine] = None):
        self.tmath_engine = tmath_engine or TMathEngine()
        
        # Hardware-Erkennung
        self.hardware_config = self._detect_hardware()
        
        # Computer-Use Integration
        self.computer_use = ComputerUseIntegration(self.tmath_engine)
        
        # Batch-Scheduler
        self.batch_scheduler = AdaptiveBatchScheduler(self.hardware_config)
        
        # ZTM-Monitoring
        self.ztm_monitor = ZTMMonitor()
        
        # Kernel-Registry
        self.kernel_registry = {
            'blur': self._kernel_blur,
            'edge_detection': self._kernel_edge_detection,
            'normalize': self._kernel_normalize,
            'rotate': self._kernel_rotate,
            'resize': self._kernel_resize,
            'screenshot': self._kernel_screenshot,
            'element_detection': self._kernel_element_detection,
            'ocr': self._kernel_ocr,
            'click': self._kernel_click,
            'type': self._kernel_type
        }
        
        # Performance-Tracking
        self.performance_stats = {
            'kernels_executed': 0,
            'total_processing_time': 0.0,
            'hardware_switches': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("VX-VISION Enhanced initialisiert")
        logger.info(f"Hardware: {self.hardware_config.primary_device.value}")
        logger.info(f"Verfügbare Geräte: {[d.value for d in self.hardware_config.available_devices]}")
    
    def _detect_hardware(self) -> HardwareConfig:
        """Erkenne verfügbare Hardware"""
        available_devices = []
        primary_device = HardwareType.CPU_ONLY
        
        # Apple Silicon Detection
        if platform.system() == 'Darwin':
            if 'arm' in platform.machine().lower():
                available_devices.append(HardwareType.APPLE_NEURAL_ENGINE)
                available_devices.append(HardwareType.APPLE_MPS)
                primary_device = HardwareType.APPLE_NEURAL_ENGINE
        
        # CUDA Detection
        if torch.cuda.is_available():
            available_devices.append(HardwareType.NVIDIA_CUDA)
            if primary_device == HardwareType.CPU_ONLY:
                primary_device = HardwareType.NVIDIA_CUDA
        
        # MPS Detection (Apple)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if HardwareType.APPLE_MPS not in available_devices:
                available_devices.append(HardwareType.APPLE_MPS)
        
        # CPU ist immer verfügbar
        available_devices.append(HardwareType.CPU_ONLY)
        
        # Speicher-Info
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # CPU-Info
        cpu_count = psutil.cpu_count()
        
        # Bestimme maximale Batch-Größe basierend auf Hardware
        if primary_device == HardwareType.APPLE_NEURAL_ENGINE:
            max_batch_size = 32
        elif primary_device == HardwareType.NVIDIA_CUDA:
            max_batch_size = 64
        elif primary_device == HardwareType.APPLE_MPS:
            max_batch_size = 16
        else:
            max_batch_size = 8
        
        config = HardwareConfig(
            primary_device=primary_device,
            available_devices=available_devices,
            memory_gb=memory_gb,
            compute_units=cpu_count,
            supports_fp16=primary_device != HardwareType.CPU_ONLY,
            supports_int8=True,
            max_batch_size=max_batch_size
        )
        
        logger.info(f"Hardware erkannt: {config.primary_device.value}")
        return config
    
    def start_system(self):
        """Starte VX-VISION System"""
        try:
            # Starte Batch-Scheduler
            self.batch_scheduler.start_scheduler()
            
            # Starte Computer-Use Monitoring
            self.computer_use.start_monitoring()
            
            # Starte ZTM-Monitoring
            self.ztm_monitor.start_monitoring()
            
            logger.info("VX-VISION Enhanced System gestartet")
            
        except Exception as e:
            logger.error(f"System-Start-Fehler: {e}")
    
    def stop_system(self):
        """Stoppe VX-VISION System"""
        try:
            # Stoppe Batch-Scheduler
            self.batch_scheduler.stop_scheduler()
            
            # Stoppe Computer-Use Monitoring
            self.computer_use.stop_monitoring()
            
            # Stoppe ZTM-Monitoring
            self.ztm_monitor.stop_monitoring()
            
            logger.info("VX-VISION Enhanced System gestoppt")
            
        except Exception as e:
            logger.error(f"System-Stop-Fehler: {e}")
    
    def execute_kernel(self, kernel_name: str, *args, **kwargs) -> Any:
        """Führe Kernel aus"""
        start_time = time.time()
        
        try:
            if kernel_name not in self.kernel_registry:
                raise ValueError(f"Unbekannter Kernel: {kernel_name}")
            
            # ZTM-Logging
            self.ztm_monitor.log_event('kernel_execution', {
                'kernel': kernel_name,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            })
            
            # Führe Kernel aus
            kernel_func = self.kernel_registry[kernel_name]
            result = kernel_func(*args, **kwargs)
            
            # Performance-Tracking
            execution_time = time.time() - start_time
            self.performance_stats['kernels_executed'] += 1
            self.performance_stats['total_processing_time'] += execution_time
            
            logger.debug(f"Kernel {kernel_name} ausgeführt in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Kernel-Ausführungsfehler ({kernel_name}): {e}")
            return None
    
    def see_and_interact(self, description: str, action: str = 'click', text: Optional[str] = None) -> bool:
        """Hauptfunktion: Sehe und interagiere"""
        try:
            logger.info(f"See-and-interact: {description} -> {action}")
            
            # Erstelle Vision-Task
            task = VisionTask(
                task_id=f"interact_{int(time.time())}",
                task_type='interaction',
                input_data={'description': description, 'action': action, 'text': text},
                priority=10,
                processing_mode=ProcessingMode.REALTIME
            )
            
            # Führe sofort aus für Realtime-Tasks
            if action == 'click':
                return self.computer_use.see_and_click(description, 'click')
            elif action == 'double_click':
                return self.computer_use.see_and_click(description, 'double_click')
            elif action == 'right_click':
                return self.computer_use.see_and_click(description, 'right_click')
            elif action == 'type' and text:
                return self.computer_use.see_and_type(description, text)
            else:
                logger.warning(f"Unbekannte Aktion: {action}")
                return False
                
        except Exception as e:
            logger.error(f"See-and-interact-Fehler: {e}")
            return False
    
    def batch_process_images(self, images: List[np.ndarray], operations: List[str]) -> List[Any]:
        """Batch-Verarbeitung von Bildern"""
        try:
            results = []
            
            for i, image in enumerate(images):
                task = VisionTask(
                    task_id=f"batch_{i}",
                    task_type='image_processing',
                    input_data={'image': image, 'operations': operations},
                    priority=5,
                    processing_mode=ProcessingMode.BATCH
                )
                
                # Füge zur Warteschlange hinzu
                self.batch_scheduler.add_task(task)
            
            # Warte auf Verarbeitung (vereinfacht)
            time.sleep(len(images) * 0.1)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch-Verarbeitungsfehler: {e}")
            return []
    
    # Kernel-Implementierungen
    def _kernel_screenshot(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Screenshot-Kernel"""
        return self.computer_use.take_screenshot(region)
    
    def _kernel_element_detection(self, screenshot: Optional[np.ndarray] = None) -> List[Dict]:
        """Element-Erkennungs-Kernel"""
        elements = self.computer_use.find_visual_elements(screenshot)
        return [{'x': e.x, 'y': e.y, 'width': e.width, 'height': e.height, 
                'type': e.element_type, 'text': e.text, 'confidence': e.confidence} 
                for e in elements]
    
    def _kernel_ocr(self, image: np.ndarray) -> str:
        """OCR-Kernel"""
        # Vereinfachte OCR-Implementierung
        elements = self.computer_use._find_text_elements(image)
        return ' '.join([e.text for e in elements if e.text])
    
    def _kernel_click(self, x: int, y: int, button: str = 'left') -> bool:
        """Klick-Kernel"""
        import pyautogui
        if button == 'left':
            pyautogui.click(x, y)
        elif button == 'right':
            pyautogui.rightClick(x, y)
        elif button == 'double':
            pyautogui.doubleClick(x, y)
        return True
    
    def _kernel_type(self, text: str) -> bool:
        """Text-Eingabe-Kernel"""
        return self.computer_use.type_text(text)
    
    def _kernel_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Blur-Kernel"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def _kernel_edge_detection(self, image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
        """Edge-Detection-Kernel"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Canny(gray, low_threshold, high_threshold)
    
    def _kernel_normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalisierungs-Kernel"""
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    def _kernel_rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotations-Kernel"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))
    
    def _kernel_resize(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize-Kernel"""
        return cv2.resize(image, (width, height))
    
    def get_system_status(self) -> Dict[str, Any]:
        """Hole System-Status"""
        return {
            'hardware_config': {
                'primary_device': self.hardware_config.primary_device.value,
                'available_devices': [d.value for d in self.hardware_config.available_devices],
                'memory_gb': self.hardware_config.memory_gb,
                'max_batch_size': self.hardware_config.max_batch_size
            },
            'performance_stats': self.performance_stats,
            'batch_scheduler': {
                'active': self.batch_scheduler.scheduler_active,
                'queue_size': len(self.batch_scheduler.task_queue),
                'completed_tasks': len(self.batch_scheduler.completed_tasks),
                'metrics': self.batch_scheduler.performance_metrics
            },
            'computer_use': self.computer_use.get_screen_info()
        }
