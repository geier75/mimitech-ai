#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VXOR/MISO System Status Check

Dieses Skript prüft den Status der wichtigsten VXOR/MISO-Komponenten,
insbesondere der VX-VISION-Module und Hardware-Unterstützung.
"""

import os
import sys
import logging
import importlib
from typing import Dict, List, Any

# Logging konfigurieren
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VXOR-STATUS-CHECK")

def check_module_importable(module_path: str) -> bool:
    """Überprüft, ob ein Modul importiert werden kann."""
    try:
        importlib.import_module(module_path)
        return True
    except ImportError as e:
        logger.warning(f"Modul {module_path} konnte nicht importiert werden: {e}")
        return False
    except Exception as e:
        logger.error(f"Unerwarteter Fehler beim Importieren von {module_path}: {e}")
        return False

def check_hardware_detection():
    """Prüft die Hardware-Erkennung"""
    try:
        from vxor.vision.hardware.detector import HardwareDetector
        detector = HardwareDetector()
        info = detector.detect_hardware()
        preferred_device = detector.get_preferred_device()
        
        logger.info(f"Hardware-Info: {info}")
        logger.info(f"Bevorzugtes Gerät: {preferred_device}")
        
        return {
            "success": True,
            "hardware_info": info,
            "preferred_device": preferred_device
        }
    except Exception as e:
        logger.error(f"Fehler bei der Hardware-Erkennung: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def check_kernel_registry():
    """Prüft die Kernel-Registry"""
    try:
        from vxor.vision.kernels.common import kernel_registry, list_available_operations
        
        operations = list_available_operations()
        logger.info(f"Verfügbare Kernel-Operationen: {operations}")
        
        registry_stats = {}
        for op in operations:
            from vxor.vision.kernels.common import list_available_backends
            backends = list_available_backends(op)
            registry_stats[op] = backends
        
        logger.info(f"Kernel-Registry-Statistik: {registry_stats}")
        
        return {
            "success": True,
            "available_operations": operations,
            "registry_stats": registry_stats
        }
    except Exception as e:
        logger.error(f"Fehler bei der Prüfung der Kernel-Registry: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def check_batch_processor():
    """Prüft den BatchProcessor Status"""
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Direkt den Imports-Pfad prüfen ohne zu importieren
        bp_path = "/Volumes/My Book/MISO_Ultimate 15.32.28/vxor/vision/batch/batch_processor.py"
        if not os.path.exists(bp_path):
            return {
                "success": False,
                "error": f"BatchProcessor-Datei nicht gefunden: {bp_path}"
            }
        
        # Versuche cv2 zu importieren, das für den BatchProcessor benötigt wird
        try:
            import cv2
            logger.info(f"OpenCV Version: {cv2.__version__}")
        except ImportError:
            return {
                "success": False,
                "error": "OpenCV (cv2) konnte nicht importiert werden, wird aber für BatchProcessor benötigt"
            }
        
        logger.info("BatchProcessor ist prinzipiell verfügbar (Datei existiert und OpenCV ist installiert)")
        return {
            "success": True,
            "note": "BatchProcessor-Datei existiert und OpenCV ist installiert"
        }
    except Exception as e:
        logger.error(f"Fehler bei der Prüfung des BatchProcessor: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def check_mlx_availability():
    """Prüft die Verfügbarkeit von MLX"""
    try:
        import mlx
        import mlx.core as mx
        
        # Prüfe auf MLX Image-Modul
        has_mlx_image = False
        try:
            import mlx.image
            has_mlx_image = True
        except ImportError:
            pass
            
        # Verwende die neue API mit default_device statt device()
        device_info = str(mx.default_device())
        logger.info(f"MLX ist verfügbar. Gerät: {device_info}")
        
        return {
            "success": True,
            "device": device_info,
            "has_image": has_mlx_image
        }
    except ImportError:
        logger.warning("MLX konnte nicht importiert werden")
        return {
            "success": False,
            "error": "MLX konnte nicht importiert werden"
        }
    except Exception as e:
        logger.error(f"Fehler bei der MLX-Prüfung: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def check_torch_availability():
    """Prüft die Verfügbarkeit von PyTorch"""
    try:
        import torch
        device_info = {
            "cuda": torch.cuda.is_available() if hasattr(torch, 'cuda') else False,
            "mps": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "device": str(torch.device("cuda" if torch.cuda.is_available() else
                                     "mps" if hasattr(torch.backends, 'mps') and
                                     torch.backends.mps.is_available() else "cpu"))
        }
        logger.info(f"PyTorch ist verfügbar. Version: {torch.__version__}")
        logger.info(f"PyTorch Geräteinfo: {device_info}")
        return {
            "success": True,
            "version": torch.__version__,
            "device_info": device_info
        }
    except ImportError:
        logger.warning("PyTorch konnte nicht importiert werden")
        return {
            "success": False,
            "error": "PyTorch konnte nicht importiert werden"
        }
    except Exception as e:
        logger.error(f"Fehler bei der PyTorch-Prüfung: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def check_implementation_status():
    """Überprüft den Implementierungsstatus wichtiger Module"""
    results = {}
    
    # Liste der zu überprüfenden Module
    modules = [
        "vxor.vision.kernels.common",
        "vxor.vision.kernels.mlx_kernels",
        "vxor.vision.kernels.torch_kernels",
        "vxor.vision.kernels.numpy_kernels",
        "vxor.vision.hardware.detector",
        "vxor.core.mlx_core"
    ]
    
    for module in modules:
        results[module] = check_module_importable(module)
    
    return results

def main():
    """Hauptfunktion für die Statusüberprüfung"""
    logger.info("=== VXOR/MISO System Status Check ===")
    
    # Implementierungsstatus überprüfen
    logger.info("\n=== Modulimplementierungsstatus ===")
    implementation_status = check_implementation_status()
    for module, status in implementation_status.items():
        logger.info(f"{module}: {'Verfügbar' if status else 'Nicht verfügbar'}")
    
    # Hardware-Erkennung überprüfen
    logger.info("\n=== Hardware-Erkennung ===")
    hardware_result = check_hardware_detection()
    
    # Kernel-Registry überprüfen
    logger.info("\n=== Kernel-Registry ===")
    registry_result = check_kernel_registry()
    
    # BatchProcessor überprüfen
    logger.info("\n=== BatchProcessor ===")
    batch_result = check_batch_processor()
    
    # MLX überprüfen
    logger.info("\n=== MLX-Verfügbarkeit ===")
    mlx_result = check_mlx_availability()
    
    # PyTorch überprüfen
    logger.info("\n=== PyTorch-Verfügbarkeit ===")
    torch_result = check_torch_availability()
    
    # Zusammenfassung
    logger.info("\n=== Zusammenfassung ===")
    logger.info(f"Modulimplementierung: {sum(implementation_status.values())}/{len(implementation_status)} Module verfügbar")
    logger.info(f"Hardware-Erkennung: {'Erfolgreich' if hardware_result.get('success') else 'Fehlgeschlagen'}")
    logger.info(f"Kernel-Registry: {'Erfolgreich' if registry_result.get('success') else 'Fehlgeschlagen'}")
    logger.info(f"BatchProcessor: {'Verfügbar' if batch_result.get('success') else 'Nicht verfügbar'}")
    logger.info(f"MLX: {'Verfügbar' if mlx_result.get('success') else 'Nicht verfügbar'}")
    logger.info(f"PyTorch: {'Verfügbar' if torch_result.get('success') else 'Nicht verfügbar'}")
    
    return {
        "implementation_status": implementation_status,
        "hardware_detection": hardware_result,
        "kernel_registry": registry_result,
        "batch_processor": batch_result,
        "mlx_availability": mlx_result,
        "torch_availability": torch_result
    }

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unerwarteter Fehler bei der Statusüberprüfung: {e}")
        sys.exit(1)
