#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-VISION Hardware Detector

Dieses Modul erweitert die automatische Hardwareerkennung für den VX-VISION-Bereich.
Es erkennt automatisch CPU-, GPU-, Speicher- und Betriebssysteminformationen und
integriert diese Daten, um Optimierungen für Visual-Verarbeitung bereitzustellen.
"""

import platform
import psutil
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# Füge Projektverzeichnis zum Pfad hinzu für Importe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '../..')))

# ZTM-Monitor importieren (wenn verfügbar)
try:
    from monitoring.ztm_monitor import track
    _ZTM_AVAILABLE = True
except ImportError:
    _ZTM_AVAILABLE = False
    logger.warning("ZTM-Monitor nicht verfügbar. Sicherheits-Tracking deaktiviert.")

# Konfiguration des Loggers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VX-VISION.hardware")

class HardwareDetector:
    """
    Erkennt und verwaltet Hardware-Informationen für optimale Bildverarbeitungsleistung.
    Unterstützt verschiedene Hardware-Beschleuniger wie Apple Neural Engine, CUDA und ROCm.
    """
    
    def __init__(self):
        """Initialisiert den Hardware-Detektor und führt die initiale Erkennung durch."""
        self.hardware_info = {}
        self.supported_devices = []
        self.preferred_device = None
        
        # Initiale Hardwareerkennung durchführen
        self.detect_hardware()
        
    def detect_hardware(self) -> Dict:
        """
        Erkennt die grundlegenden Hardwarekomponenten des Systems:
          - Betriebssystem und Release
          - Anzahl der CPU-Kerne
          - Gesamter verfügbare Arbeitsspeicher
          - GPU-Informationen und Verfügbarkeit von Beschleunigern
        
        Returns:
            Dict: Hardwareinformationen des Systems.
        """
        # ZTM-Tracking starten
        if _ZTM_AVAILABLE:
            track("vxor.vision.hardware.detector", "hardware_detection_start", 
                  {"timestamp": str(os.times())}, "INFO")
        
        # Basis-Systeminformationen
        self.hardware_info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
        }
        
        # Erkennung von Apple Silicon (Neural Engine)
        self.hardware_info["has_apple_silicon"] = self._detect_apple_silicon()
        
        # ZTM-Tracking für Hardware-Erkennung abschließen
        if _ZTM_AVAILABLE:
            track("vxor.vision.hardware.detector", "hardware_detection_complete", 
                  {"detected_hardware": self.hardware_info}, "INFO")
        
        # GPU-Erkennung für CUDA (NVIDIA)
        self.hardware_info["has_cuda"] = self._detect_cuda()
        
        # GPU-Erkennung für ROCm (AMD)
        self.hardware_info["has_rocm"] = self._detect_rocm()
        
        # MPS (Metal Performance Shaders) für macOS
        self.hardware_info["has_mps"] = self._detect_mps()
        
        # Liste der verfügbaren Geräte erstellen
        self._update_device_list()
        
        logger.info(f"Hardware erkannt: {self.hardware_info}")
        logger.info(f"Verfügbare Geräte: {self.supported_devices}")
        
        return self.hardware_info
    
    def get_available_memory(self) -> int:
        """
        Gibt den aktuell verfügbaren Arbeitsspeicher in Bytes zurück.
        
        Returns:
            int: Verfügbarer Arbeitsspeicher in Bytes
        """
        return psutil.virtual_memory().available
    
    def get_compute_units(self) -> int:
        """
        Schätzt die Anzahl der verfügbaren Compute-Units basierend auf der Hardware.
        
        Returns:
            int: Geschätzte Anzahl der Compute-Units
        """
        if self.hardware_info.get("has_apple_silicon", False):
            # Apple Silicon hat typischerweise 8-16 Performance-Kerne + Neural Engine
            return max(8, self.hardware_info.get("physical_cpu_count", 4)) * 4
        elif self.hardware_info.get("has_cuda", False):
            # Grobe Schätzung für CUDA-Geräte
            return 32  # Typisch für moderne NVIDIA-GPUs
        elif self.hardware_info.get("has_rocm", False):
            # Grobe Schätzung für AMD-GPUs
            return 24
        else:
            # CPU-Fallback
            return max(4, self.hardware_info.get("physical_cpu_count", 2))
    
    def get_preferred_device(self) -> str:
        """
        Gibt das bevorzugte Beschleunigergerät für die Bildverarbeitung zurück.
        
        Returns:
            str: Bezeichner des bevorzugten Geräts ('ane', 'cuda', 'mps', 'rocm', 'cpu')
        """
        if not self.preferred_device:
            self._update_device_list()
            
        return self.preferred_device
        
    def has_neural_engine(self) -> bool:
        """
        Prüft, ob das Gerät eine Apple Neural Engine hat.
        
        Returns:
            bool: True, wenn Apple Neural Engine verfügbar ist
        """
        return self.hardware_info.get("has_apple_silicon", False)
        
    def has_gpu(self) -> bool:
        """
        Prüft, ob das Gerät eine GPU für Beschleunigung hat (CUDA, ROCm oder MPS).
        
        Returns:
            bool: True, wenn eine GPU-Beschleunigung verfügbar ist
        """
        return (self.hardware_info.get("has_cuda", False) or 
                self.hardware_info.get("has_rocm", False) or
                self.hardware_info.get("has_mps", False))
    
    def get_best_device_for(self, operations: List) -> str:
        """
        Bestimmt das beste Gerät für die angegebenen Operationen.
        
        Args:
            operations: Liste von Operationen, die ausgeführt werden sollen
            
        Returns:
            str: Name des optimalen Geräts für die angegebenen Operationen
        """
        # ZTM-Tracking für Geräteauswahl (sicherheitskritisch)
        if _ZTM_AVAILABLE:
            track("vxor.vision.hardware.detector", "device_selection", 
                 {"operations": operations}, "INFO")
        
        # Vereinfachte Logik zur Bestimmung des besten Geräts
        # In einer vollständigen Implementierung würde hier eine komplexere
        # Entscheidungslogik basierend auf den spezifischen Operationen stehen
        
        if not operations:
            return self.get_preferred_device()
    
    def has_apple_silicon(self) -> bool:
        """Prüft, ob Apple Silicon (M-Series) verfügbar ist."""
        return self.hardware_info.get("has_apple_silicon", False)
    
    def has_gpu(self) -> bool:
        """Prüft, ob eine GPU (CUDA, ROCm oder MPS) verfügbar ist."""
        return (self.hardware_info.get("has_cuda", False) or 
                self.hardware_info.get("has_rocm", False) or 
                self.hardware_info.get("has_mps", False))
    
    def _detect_apple_silicon(self) -> bool:
        """Erkennt, ob der Computer Apple Silicon (M1/M2/M3) verwendet."""
        if platform.system() == "Darwin":
            try:
                # Prüfen auf ARM-basierte Mac-Modelle
                machine = platform.machine()
                if machine == "arm64":
                    logger.info("Apple Silicon (ARM64) erkannt")
                    return True
            except Exception as e:
                logger.warning(f"Fehler bei der Apple Silicon-Erkennung: {e}")
        return False
    
    def _detect_cuda(self) -> bool:
        """Erkennt, ob CUDA (NVIDIA) verfügbar ist."""
        try:
            # Versuchen, PyTorch mit CUDA zu importieren
            import torch
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                logger.info(f"CUDA verfügbar: {torch.cuda.get_device_name(0)}")
            return has_cuda
        except ImportError:
            logger.info("PyTorch nicht installiert, CUDA-Erkennung nicht möglich")
            return False
        except Exception as e:
            logger.warning(f"Fehler bei der CUDA-Erkennung: {e}")
            return False
    
    def _detect_rocm(self) -> bool:
        """Erkennt, ob ROCm (AMD) verfügbar ist."""
        try:
            # Prüfen auf ROCm-Umgebungsvariable
            if "ROCM_PATH" in os.environ:
                logger.info("ROCm-Pfad gefunden in Umgebungsvariablen")
                return True
                
            # Alternativ prüfen auf PyTorch mit ROCm
            import torch
            if hasattr(torch, "hip") and torch.hip.is_available():
                logger.info("PyTorch mit ROCm-Unterstützung erkannt")
                return True
                
            return False
        except ImportError:
            logger.info("PyTorch nicht installiert, ROCm-Erkennung eingeschränkt")
            return False
        except Exception as e:
            logger.warning(f"Fehler bei der ROCm-Erkennung: {e}")
            return False
    
    def _detect_mps(self) -> bool:
        """Erkennt, ob Metal Performance Shaders (MPS) auf macOS verfügbar sind."""
        if platform.system() != "Darwin":
            return False
            
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("MPS (Metal Performance Shaders) verfügbar")
                return True
            return False
        except ImportError:
            logger.info("PyTorch nicht installiert, MPS-Erkennung nicht möglich")
            return False
        except Exception as e:
            logger.warning(f"Fehler bei der MPS-Erkennung: {e}")
            return False
    
    def _update_device_list(self):
        """Aktualisiert die Liste der unterstützten Hardware-Beschleuniger."""
        self.supported_devices = ["cpu"]  # CPU ist immer verfügbar
        
        if self.hardware_info.get("has_apple_silicon", False):
            self.supported_devices.append("ane")
            
        if self.hardware_info.get("has_cuda", False):
            self.supported_devices.append("cuda")
            
        if self.hardware_info.get("has_rocm", False):
            self.supported_devices.append("rocm")
            
        if self.hardware_info.get("has_mps", False):
            self.supported_devices.append("mps")


if __name__ == "__main__":
    # Einfacher Test
    detector = HardwareDetector()
    info = detector.detect_hardware()
    print("Erkannte Hardware:", info)
    print("Bevorzugtes Gerät:", detector.get_preferred_device())
    print("Verfügbare Compute-Units:", detector.get_compute_units())
