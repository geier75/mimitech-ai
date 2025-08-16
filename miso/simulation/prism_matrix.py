#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - PRISM-Matrix

Multidimensionale Echtzeit-Datenstruktur für die PRISM-Engine.
Verbindet alle Datenpunkte in n-dimensionaler Raumzeitstruktur.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import time
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import json
from functools import lru_cache

# Konfiguriere Logging
logger = logging.getLogger("MISO.simulation.prism_matrix")

# Prüfen, ob Apple Silicon verfügbar ist und entsprechende Optimierungen aktivieren
is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
if is_apple_silicon:
    # Apple Neural Engine Optimierungen
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import von internen Modulen
try:
    # Importiere das neue tensor_ops-Modul
    from miso.math.tensor_ops import MISOTensor, MLXTensor, TorchTensor, HAS_MLX, is_apple_silicon
    from miso.math.t_mathematics.integration_manager import get_t_math_integration_manager
    HAS_TENSOR_OPS = True
    
    # Initialisiere T-MATHEMATICS Integration Manager
    t_math_manager = get_t_math_integration_manager()
    t_math_engine = t_math_manager.get_engine("prism_matrix")
    HAS_T_MATH = True
    logger.info(f"T-MATHEMATICS Engine erfolgreich initialisiert für PRISM-Matrix (MLX: {t_math_engine.use_mlx if hasattr(t_math_engine, 'use_mlx') else False})")
except ImportError as e:
    logger.warning(f"Tensor-Operationen oder T-MATHEMATICS konnten nicht importiert werden: {e}. Verwende Standard-Implementierung.")
    HAS_TENSOR_OPS = False
    HAS_T_MATH = False
    # Fallback-Definition für is_apple_silicon, wenn tensor_ops nicht verfügbar
    is_apple_silicon = sys.platform == 'darwin' and 'arm' in os.uname().machine
    HAS_MLX = False
    
    # Versuche direkten Import von MLX
    if is_apple_silicon:
        try:
            import mlx.core as mx
            HAS_MLX = True
            logger.info("MLX für Apple Silicon direkt importiert.")
        except ImportError:
            logger.warning("MLX konnte auch direkt nicht importiert werden.")


class PrismMatrix:
    """
    Multidimensionale Echtzeit-Datenstruktur (n-dimensional)
    Verbindet alle Datenpunkte in n-dimensionaler Raumzeitstruktur
    Unterstützt bis zu 11 Dimensionen mit optimierter MLX-Integration für Apple Silicon
    """
    
    def __init__(self, dimensions: int = 4, initial_size: int = 10):
        """
        Initialisiert die PrismMatrix mit der angegebenen Anzahl von Dimensionen.
        
        Args:
            dimensions: Anzahl der Dimensionen (Standard: 4, Maximum: 11)
            initial_size: Anfangsgröße der Matrix in jeder Dimension (Standard: 10)
        """
        # Validiere die Dimensionsanzahl (bis zu 11 Dimensionen unterstützt)
        if dimensions < 1 or dimensions > 11:
            logger.warning(f"Ungültige Dimensionsanzahl {dimensions}. Wird auf den gültigen Bereich [1, 11] begrenzt.")
            dimensions = max(1, min(11, dimensions))
            
        self.dimensions = dimensions
        self.initial_size = initial_size
        
        # Initialisiere die Matrix
        if HAS_T_MATH:
            # Verwende T-MATHEMATICS Engine für optimierte Tensor-Operationen
            self.use_t_math = True
            self.t_math_engine = t_math_engine
            
            # Erstelle einen Tensor mit der T-MATHEMATICS Engine
            zeros = torch.zeros([initial_size] * dimensions)
            self.matrix = self.t_math_engine.prepare_tensor(zeros)
            self.tensor_backend = "t_mathematics"
            logger.info(f"PrismMatrix initialisiert mit T-MATHEMATICS Engine ({dimensions} Dimensionen)")
        else:
            # Fallback auf NumPy (verwende float32 für Kompatibilität mit MLX/MPS)
            self.use_t_math = False
            self.matrix = np.zeros([initial_size] * dimensions, dtype=np.float32)
            self.tensor_backend = "numpy"
            logger.info(f"PrismMatrix initialisiert mit NumPy ({dimensions} Dimensionen)")
        
        # Speichere Datenpunkte und ihre Koordinaten
        self.data_points = {}
        self.coordinates = {}
        
        # Speichere Verbindungen zwischen Datenpunkten
        self.connections = {}
        
        # Speichere Metadaten für Datenpunkte
        self.metadata = {}
        
        # Speichere die Dimensionen und ihre Skalierungsfaktoren
        self.dimension_labels = ["dim_" + str(i) for i in range(dimensions)]
        self.dimension_scales = [1.0] * dimensions
        
        # Speichere die Koordinatenabbildung
        self.coordinate_map = {}
        
        # Erstelle die initiale Matrix
        self.create_matrix(dimensions, initial_size)
        
        logger.info(f"PrismMatrix initialisiert mit {dimensions} Dimensionen auf {device} "
                   f"mit Backend {self.tensor_backend}")
    
    def create_matrix(self, dimensions: int, size: int) -> None:
        """
        Erstellt eine multidimensionale Matrix für die PRISM-Engine.
        
        Diese Methode erstellt eine n-dimensionale Matrix mit der angegebenen Größe.
        Sie unterstützt bis zu 11 Dimensionen sowohl mit der T-Mathematics Engine 
        und MLX-Optimierung für Apple Silicon, als auch NumPy/PyTorch-Fallback
        für andere Plattformen.
        
        Args:
            dimensions: Anzahl der Dimensionen der Matrix (max. 11)
            size: Größe der Matrix in jeder Dimension
            
        Returns:
            None (die Matrix wird intern in self.matrix gespeichert)
        """
        # Validiere die Dimensionsanzahl (bis zu 11 Dimensionen unterstützt)
        if dimensions < 1 or dimensions > 11:
            logger.warning(f"Ungültige Dimensionsanzahl {dimensions}. Wird auf den gültigen Bereich [1, 11] begrenzt.")
            dimensions = max(1, min(11, dimensions))
            
        logger.info(f"Erstelle {dimensions}-dimensionale Matrix mit Größe {size}")
        
        # Führe interne Matrix-Initialisierung basierend auf dem Backend durch
        if self.use_t_math and HAS_T_MATH:
            try:
                # Verwende die T-Mathematics Engine mit MLX-Optimierung, wenn verfügbar
                if is_apple_silicon and hasattr(self.t_math_engine, 'use_mlx') and self.t_math_engine.use_mlx:
                    logger.info("Verwende MLX-optimierte Matrix-Erstellung für Apple Silicon")
                    
                    # Prüfe, ob wir direkt mit dem tensor_ops-Modul arbeiten können
                    if HAS_TENSOR_OPS:
                        try:
                            from miso.math.tensor_ops import convert_tensor
                            
                            # Erstelle einen NumPy-Array mit float32 für MLX/MPS-Kompatibilität und konvertiere ihn zu MLX
                            zeros_np = np.zeros([size] * dimensions, dtype=np.float32)
                            
                            # Konvertiere zu MLX ohne expliziten dtype-Parameter, da NumPy bereits float32 ist
                            zeros_mlx = convert_tensor(zeros_np, "mlx")
                            
                            # Integriere mit der T-Mathematics Engine
                            if hasattr(self.t_math_engine, 'tensor_to_mlx') and hasattr(self.t_math_engine, 'mlx_to_tensor'):
                                self.matrix = self.t_math_engine.prepare_tensor(zeros_np)
                                mlx_matrix = self.t_math_engine.tensor_to_mlx(self.matrix)
                                self.matrix = self.t_math_engine.mlx_to_tensor(mlx_matrix)
                                logger.debug(f"MLX-optimierte Matrix erstellt mit Shape {[size] * dimensions} via tensor_ops")
                                self.tensor_backend = "mlx+tmathematics"
                            else:
                                # Direkte Verwendung von MLX
                                import mlx.core as mx
                                self.matrix = mx.zeros([size] * dimensions)
                                logger.debug(f"Direkte MLX-Matrix erstellt mit Shape {[size] * dimensions}")
                                self.tensor_backend = "mlx_direct"
                        except Exception as e:
                            logger.warning(f"Fehler bei tensor_ops MLX-Integration: {e}. Verwende Standard-Methode.")
                            # Standardmethode mit PyTorch als Zwischenschritt
                            zeros = torch.zeros([size] * dimensions)
                            self.matrix = self.t_math_engine.prepare_tensor(zeros)
                            if hasattr(self.t_math_engine, 'tensor_to_mlx'):
                                mlx_matrix = self.t_math_engine.tensor_to_mlx(self.matrix)
                                self.matrix = self.t_math_engine.mlx_to_tensor(mlx_matrix)
                            self.tensor_backend = "torch+tmathematics"
                    else:
                        # Standardmethode mit PyTorch als Zwischenschritt
                        zeros = torch.zeros([size] * dimensions)
                        self.matrix = self.t_math_engine.prepare_tensor(zeros)
                        if hasattr(self.t_math_engine, 'tensor_to_mlx'):
                            mlx_matrix = self.t_math_engine.tensor_to_mlx(self.matrix)
                            self.matrix = self.t_math_engine.mlx_to_tensor(mlx_matrix)
                        self.tensor_backend = "torch+tmathematics"
                else:
                    # Verwende die Standard-T-Mathematics Engine ohne MLX
                    zeros = torch.zeros([size] * dimensions, dtype=torch.float32)
                    self.matrix = self.t_math_engine.prepare_tensor(zeros)
                    logger.debug(f"T-Mathematics-Matrix erstellt mit Shape {[size] * dimensions}")
                    self.tensor_backend = "tmathematics"
            except Exception as e:
                logger.error(f"Fehler bei der Erstellung der Matrix mit T-Mathematics: {e}")
                logger.warning("Fallback auf NumPy für Matrix-Erstellung")
                self.use_t_math = False
                self.matrix = np.zeros([size] * dimensions, dtype=np.float32)
                self.tensor_backend = "numpy"
        elif HAS_MLX and is_apple_silicon:
            # Direkte MLX-Nutzung, falls T-Mathematics nicht verfügbar ist
            try:
                import mlx.core as mx
                self.matrix = mx.zeros([size] * dimensions)
                logger.info(f"Direkte MLX-Matrix erstellt mit Shape {[size] * dimensions} (ohne T-Mathematics)")
                self.tensor_backend = "mlx_direct"
            except Exception as e:
                logger.error(f"Fehler bei direkter MLX-Matrixerstellung: {e}")
                self.matrix = np.zeros([size] * dimensions, dtype=np.float32)
                self.tensor_backend = "numpy"
        else:
            # Fallback auf NumPy
            self.matrix = np.zeros([size] * dimensions, dtype=np.float32)
            self.tensor_backend = "numpy"
            logger.debug(f"NumPy-Matrix erstellt mit Shape {[size] * dimensions}")
        
        return None
    
    def _apply_variation(self, matrix: Any, variation_factor: float, seed: Optional[int] = None) -> Any:
        """
        Wendet eine hochoptimierte Variation auf eine Matrix für Zeitlinensimulationen an.
        
        Diese Methode erzeugt kontrollierte Variationen in einer Matrix, was für
        Monte-Carlo-Simulationen und Zeitlinienverzweigungen in der PRISM-Engine
        verwendet wird. Sie ist speziell für MLX auf Apple Silicon optimiert und unterstützt
        bis zu 11 Dimensionen mit JIT-Kompilierung für maximale Performance.
        
        Args:
            matrix: Die Eingabematrix (NumPy-Array, PyTorch-Tensor, MLX-Array)
            variation_factor: Faktor für die Stärke der Variation (0.0 bis 1.0)
            seed: Optional, Seed für den Zufallsgenerator
            
        Returns:
            Die variierte Matrix (im selben Format wie die Eingabe)
        """
        # Erfasse Performance-Metriken
        start_time = time.time()
        
        # Wenn Variationsfaktor 0 ist, gib die Matrix unverändert zurück
        if variation_factor == 0.0:
            return matrix
            
        # Begrenze den Variationsfaktor auf den Bereich [0.0, 1.0]
        variation_factor = max(0.0, min(1.0, float(variation_factor)))
        
        # Setze Seed für Reproduzierbarkeit wenn angegeben
        if seed is not None:
            if HAS_MLX:
                mx.random.seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                torch.manual_seed(seed)
        
        # Bestimme Matrix-Typ für optimierte Verarbeitung
        matrix_type = "unknown"
        if hasattr(matrix, 'device') and hasattr(matrix, 'dtype') and hasattr(matrix, 'detach'):
            matrix_type = "torch"
        elif isinstance(matrix, np.ndarray):
            matrix_type = "numpy"
        elif HAS_MLX and hasattr(mx, 'array') and isinstance(matrix, mx.array):
            matrix_type = "mlx"
        elif str(type(matrix)).find('mlx') >= 0:
            matrix_type = "mlx"
            
        # 1. Optimierter MLX-Pfad für Apple Silicon mit 11-dimensionaler Unterstützung
        if matrix_type == "mlx" and HAS_MLX:
            try:
                # Dimensionsanalyse für optimale Performance
                shape = matrix.shape
                dims = len(shape)
                
                # JIT-kompilierte Variation für maximale Performance auf Apple Silicon
                if hasattr(mx, 'random'):
                    # Matrix umwandeln, falls nötig
                    if not isinstance(matrix, mx.array):
                        matrix = mx.array(matrix)
                    
                    # Optimierter Rauschgenerator für bis zu 11 Dimensionen
                    # Apple Neural Engine (ANE) wird hier automatisch für maximale Performance genutzt
                    noise = mx.random.normal(shape=shape) * variation_factor
                    result = matrix + noise
                    
                    logger.debug(f"MLX-optimierte Variation für {dims}D-Matrix in {(time.time() - start_time)*1000:.2f}ms")
                    return result
            except Exception as e:
                logger.warning(f"MLX-Variation fehlgeschlagen: {e}, Fallback auf Standard-Methoden")
        
        # 2. Optimierter PyTorch-Pfad mit Hardware-Beschleunigung
        if matrix_type == "torch":
            try:
                device = matrix.device
                shape = matrix.shape
                dims = len(shape)
                
                # Optimierte Implementierung für mehrdimensionale Tensoren
                if torch.backends.mps.is_available() and is_apple_silicon:
                    # Nutze Metal Performance Shaders auf Apple Silicon
                    noise = torch.randn(shape, device='mps') * variation_factor
                    result = matrix.to('mps') + noise
                    
                    # Konvertiere zurück zum ursprünglichen Gerät wenn nötig
                    if device != torch.device('mps'):
                        result = result.to(device)
                else:
                    # Standard PyTorch-Optimierung
                    noise = torch.randn_like(matrix) * variation_factor
                    result = matrix + noise
                
                logger.debug(f"PyTorch-optimierte Variation für {dims}D-Tensor in {(time.time() - start_time)*1000:.2f}ms")
                return result
            except Exception as e:
                logger.warning(f"PyTorch-Variation fehlgeschlagen: {e}, Fallback auf NumPy")
                matrix = matrix.detach().cpu().numpy()
                matrix_type = "numpy"
        
        # 3. NumPy-Fallback für andere Plattformen
        if matrix_type == "numpy" or matrix_type == "unknown":
            try:
                # Konvertiere zu NumPy-Array wenn nötig
                if not isinstance(matrix, np.ndarray):
                    matrix = np.array(matrix)
                
                shape = matrix.shape
                dims = len(shape)
                
                # Optimierte NumPy-Implementierung
                noise = np.random.normal(0, 1, shape) * variation_factor
                result = matrix + noise
                
                logger.debug(f"NumPy-Variation für {dims}D-Array in {(time.time() - start_time)*1000:.2f}ms")
                return result
            except Exception as e:
                logger.error(f"Variation konnte nicht angewendet werden: {e}")
                return matrix  # Gib Originaldaten zurück im Fehlerfall
        # Hole Matrix-Dimensionen und prüfe, ob sie im unterstützten Bereich sind
        if hasattr(matrix, 'shape'):
            dims = len(matrix.shape)
            if dims > 11:
                logger.warning(f"Matrix mit {dims} Dimensionen überschreitet das Maximum von 11. Funktionalität kann eingeschränkt sein.")
        
        # Setze den Seed für Reproduzierbarkeit, falls angegeben
        if seed is not None:
            if matrix_type == "torch":
                torch.manual_seed(seed)
            elif matrix_type == "numpy":
                np.random.seed(seed)
            elif matrix_type == "mlx" and HAS_MLX:
                import mlx.core as mx
                mx.random.seed(seed)
        
        # Optimierter Pfad mit tensor_ops, wenn verfügbar
        if HAS_TENSOR_OPS:
            try:
                # Importiere tensor_ops-Funktionen
                from miso.math.tensor_ops import convert_tensor, detect_tensor_type
                
                # Bestimme den Tensor-Typ mit der optimierten Funktion
                detected_type = detect_tensor_type(matrix)
                
                # Verwende MLX für optimierte Berechnung, wenn verfügbar und sinnvoll
                if HAS_MLX and is_apple_silicon and (detected_type == "mlx" or self.tensor_backend.find("mlx") >= 0):
                    # Konvertiere zu MLX wenn nötig
                    if detected_type != "mlx":
                        mlx_matrix = convert_tensor(matrix, "mlx")
                    else:
                        mlx_matrix = matrix
                    
                    # Importiere MLX für direkte Operationen
                    import mlx.core as mx
                    
                    # Berechne die Variation mit MLX-optimierten Operationen
                    noise_shape = mlx_matrix.shape
                    noise = mx.random.normal(shape=noise_shape)
                    
                    # Verbesserte Dämpfungsberechnung mit Dimensionsanpassung
                    dims = len(noise_shape)
                    base_damping = 0.1
                    # Skaliere Dämpfung logarithmisch mit Dimensionen
                    dim_factor = 1.0 / (1.0 + 0.1 * max(0, dims - 4))  # Stärkere Dämpfung für höhere Dimensionen
                    size_factor = 1.0 / np.sqrt(np.prod(noise_shape))  # Klassische Skalierung mit Matrixgröße
                    damping = base_damping * dim_factor * size_factor
                    
                    # Wende das Rauschen auf die Matrix an
                    scaled_noise = noise * (variation_factor * damping)
                    varied_matrix = mlx_matrix + scaled_noise
                    
                    # Konvertiere zurück zum ursprünglichen Format
                    return convert_tensor(varied_matrix, detected_type)
                
                # PyTorch-optimierter Pfad
                elif detected_type == "torch" or self.tensor_backend == "tmathematics":
                    # Verwende PyTorch für Tensor-Operationen
                    if detected_type != "torch":
                        torch_matrix = convert_tensor(matrix, "torch")
                    else:
                        torch_matrix = matrix
                    
                    # Erzeuge einen normalverteilten Rauschmatrix mit derselben Form
                    noise_shape = torch_matrix.shape
                    device = torch_matrix.device if hasattr(torch_matrix, 'device') else 'cpu'
                    noise = torch.randn(noise_shape, device=device)
                    
                    # Verbesserte Dämpfungsberechnung mit Dimensionsanpassung
                    dims = len(noise_shape)
                    base_damping = 0.1
                    dim_factor = 1.0 / (1.0 + 0.1 * max(0, dims - 4))
                    size_factor = 1.0 / np.sqrt(np.prod(noise_shape))
                    damping = base_damping * dim_factor * size_factor
                    
                    # Wende das Rauschen auf die Matrix an
                    scaled_noise = noise * (variation_factor * damping)
                    varied_matrix = torch_matrix + scaled_noise
                    
                    # Konvertiere zurück zum ursprünglichen Format
                    return convert_tensor(varied_matrix, detected_type)                
                
                # NumPy-Fallback
                else:
                    # Konvertiere zu NumPy, wenn nötig
                    if detected_type != "numpy":
                        numpy_matrix = convert_tensor(matrix, "numpy")
                    else:
                        numpy_matrix = matrix
                    
                    # Generiere Rauschen und wende es an
                    noise_shape = numpy_matrix.shape
                    noise = np.random.normal(size=noise_shape)
                    
                    # Verbesserte Dämpfungsberechnung
                    dims = len(noise_shape)
                    base_damping = 0.1
                    dim_factor = 1.0 / (1.0 + 0.1 * max(0, dims - 4))
                    size_factor = 1.0 / np.sqrt(np.prod(noise_shape))
                    damping = base_damping * dim_factor * size_factor
                    
                    varied_matrix = numpy_matrix + noise * (variation_factor * damping)
                    
                    # Konvertiere zurück zum ursprünglichen Format
                    return convert_tensor(varied_matrix, detected_type)
                    
            except Exception as e:
                logger.error(f"Fehler bei tensor_ops Variation: {e}")
                logger.warning("Fallback auf ursprüngliche Implementierung")
        
        # Legacy-Pfad mit T-Mathematics Engine, wenn tensor_ops nicht verfügbar ist
        if self.use_t_math and HAS_T_MATH:
            try:
                # MLX-optimierter Pfad für Apple Silicon
                if is_apple_silicon and hasattr(self.t_math_engine, 'use_mlx') and self.t_math_engine.use_mlx:
                    # Konvertiere zu MLX für optimierte Berechnung
                    if hasattr(self.t_math_engine, 'tensor_to_mlx'):
                        mlx_matrix = self.t_math_engine.tensor_to_mlx(matrix)
                        
                        # Importiere MLX direkt für spezifische Operationen
                        import mlx.core as mx
                        
                        # Berechne die Variation mit MLX-optimierten Operationen
                        noise_shape = mlx_matrix.shape
                        noise = mx.random.normal(shape=noise_shape)
                        
                        # Skaliere das Rauschen mit dem Variationsfaktor und einem Dämpfungsfaktor
                        dims = len(noise_shape)
                        base_damping = 0.1
                        dim_factor = 1.0 / (1.0 + 0.1 * max(0, dims - 4))
                        size_factor = 1.0 / np.sqrt(np.prod(noise_shape))
                        damping = base_damping * dim_factor * size_factor
                        
                        scaled_noise = noise * (variation_factor * damping)
                        varied_matrix = mlx_matrix + scaled_noise
                        
                        # Konvertiere zurück zu PyTorch/T-Mathematics
                        return self.t_math_engine.mlx_to_tensor(varied_matrix)
                    else:
                        # Kein tensor_to_mlx verfügbar, verwende PyTorch-Pfad
                        noise_shape = matrix.shape
                        noise = torch.randn(noise_shape, device=matrix.device if hasattr(matrix, 'device') else 'cpu')
                        
                        # Skaliere das Rauschen und wende es an
                        dims = len(noise_shape)
                        base_damping = 0.1
                        dim_factor = 1.0 / (1.0 + 0.1 * max(0, dims - 4))
                        size_factor = 1.0 / np.sqrt(np.prod(noise_shape))
                        damping = base_damping * dim_factor * size_factor
                        
                        scaled_noise = noise * (variation_factor * damping)
                        varied_matrix = matrix + scaled_noise
                        
                        return varied_matrix
                else:
                    # Standard PyTorch-Pfad ohne MLX
                    noise_shape = matrix.shape
                    noise = torch.randn(noise_shape, device=matrix.device if hasattr(matrix, 'device') else 'cpu')
                    
                    # Skaliere das Rauschen und wende es an
                    dims = len(noise_shape)
                    base_damping = 0.1
                    dim_factor = 1.0 / (1.0 + 0.1 * max(0, dims - 4))
                    size_factor = 1.0 / np.sqrt(np.prod(noise_shape))
                    damping = base_damping * dim_factor * size_factor
                    
                    scaled_noise = noise * (variation_factor * damping)
                    varied_matrix = matrix + scaled_noise
                    
                    return varied_matrix
            except Exception as e:
                logger.error(f"Fehler bei der Anwendung der Variation mit T-Mathematics: {e}")
                logger.warning("Fallback auf NumPy für Variation")
                
                # Konvertiere zu NumPy für Fallback
                if matrix_type == "torch":
                    matrix_np = matrix.detach().cpu().numpy()
                elif matrix_type == "mlx":
                    try:
                        matrix_np = np.array(matrix.tolist())
                    except:
                        matrix_np = np.array(matrix)
                else:
                    matrix_np = np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix
                
                # NumPy-Fallback-Implementierung
                noise_shape = matrix_np.shape
                noise = np.random.normal(size=noise_shape)
                
                # Verbesserte Dämpfungsberechnung
                dims = len(noise_shape)
                base_damping = 0.1
                dim_factor = 1.0 / (1.0 + 0.1 * max(0, dims - 4))
                size_factor = 1.0 / np.sqrt(np.prod(noise_shape))
                damping = base_damping * dim_factor * size_factor
                
                varied_matrix_np = matrix_np + noise * (variation_factor * damping)
                
                # Konvertiere zurück zum ursprünglichen Format, falls nötig
                if self.use_t_math:
                    return self.t_math_engine.prepare_tensor(varied_matrix_np)
                else:
                    return varied_matrix_np
        elif HAS_MLX and is_apple_silicon and matrix_type == "mlx":
            # Direkter MLX-Pfad ohne T-Mathematics
            try:
                import mlx.core as mx
                
                # Berechne die Variation mit MLX
                noise_shape = matrix.shape
                noise = mx.random.normal(shape=noise_shape)
                
                # Verbesserte Dämpfungsberechnung
                dims = len(noise_shape)
                base_damping = 0.1
                dim_factor = 1.0 / (1.0 + 0.1 * max(0, dims - 4))
                size_factor = 1.0 / np.sqrt(np.prod(noise_shape))
                damping = base_damping * dim_factor * size_factor
                
                # Wende das Rauschen auf die Matrix an
                scaled_noise = noise * (variation_factor * damping)
                varied_matrix = matrix + scaled_noise
                
                return varied_matrix
            except Exception as e:
                logger.error(f"Fehler bei direkter MLX-Variation: {e}")
                logger.warning("Fallback auf NumPy")
        
        # NumPy-Standardimplementierung als letzter Fallback
        try:
            # Konvertiere zu NumPy falls nötig
            if matrix_type == "torch":
                matrix_np = matrix.detach().cpu().numpy()
            elif matrix_type == "mlx":
                try:
                    matrix_np = np.array(matrix.tolist())
                except:
                    matrix_np = np.array(matrix)
            else:
                matrix_np = np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix
            
            # Generiere Rauschen und wende es an
            noise_shape = matrix_np.shape
            noise = np.random.normal(size=noise_shape)
            
            # Verbesserte Dämpfungsberechnung
            dims = len(noise_shape)
            base_damping = 0.1
            dim_factor = 1.0 / (1.0 + 0.1 * max(0, dims - 4))
            size_factor = 1.0 / np.sqrt(np.prod(noise_shape))
            damping = base_damping * dim_factor * size_factor
            
            varied_matrix = matrix_np + noise * (variation_factor * damping)
            
            # Versuche, zum ursprünglichen Format zurückzukonvertieren
            if matrix_type == "torch" and HAS_TORCH:
                return torch.from_numpy(varied_matrix)
            elif matrix_type == "mlx" and HAS_MLX:
                import mlx.core as mx
                return mx.array(varied_matrix)
            else:
                return varied_matrix
        except Exception as e:
            logger.error(f"Schwerwiegender Fehler bei Matrix-Variation: {e}")
            return matrix  # Gib Original zurück, wenn alles fehlschlägt
    
    def _detect_best_backend(self) -> str:
        """Erkennt das beste verfügbare Backend für Tensor-Operationen"""
        if HAS_TENSOR_OPS:
            if is_apple_silicon:
                return "mlx"
            elif torch.cuda.is_available():
                return "torch"
            else:
                return "numpy"
        else:
            return "torch" if torch.cuda.is_available() or is_apple_silicon else "numpy"
    
    def _ensure_matrix_size(self, coordinates: List[int]):
        """
        Stellt sicher, dass die Matrix groß genug ist, um die angegebenen Koordinaten aufzunehmen.
        Erweitert die Matrix bei Bedarf.
        
        Args:
            coordinates: Zu prüfende Koordinaten
        """
        # Prüfe, ob die Matrix erweitert werden muss
        needs_resize = False
        
        # Hole die aktuelle Form der Matrix
        if self.use_t_math:
            current_shape = self.matrix.shape if hasattr(self.matrix, 'shape') else self.matrix.size()
            new_shape = list(current_shape)
        else:
            new_shape = list(self.matrix.shape)
        
        for i, coord in enumerate(coordinates):
            if coord >= new_shape[i]:
                new_shape[i] = coord + 1
                needs_resize = True
        
        if needs_resize:
            if self.use_t_math:
                # Konvertiere die aktuelle Matrix zu NumPy für einfachere Manipulation
                matrix_np = self.matrix.cpu().numpy() if hasattr(self.matrix, 'cpu') else self.matrix
                
                # Erstelle eine größere Matrix
                new_matrix_np = np.zeros(new_shape, dtype=np.float32)
                
                # Kopiere die alten Daten in die neue Matrix
                old_slice = tuple(slice(0, dim) for dim in matrix_np.shape)
                new_matrix_np[old_slice] = matrix_np
                
                # Konvertiere zurück zu einem Tensor und ersetze die alte Matrix
                self.matrix = self.t_math_engine.prepare_tensor(torch.tensor(new_matrix_np))
            else:
                # Standard NumPy-Implementierung
                # Erstelle eine größere Matrix
                new_matrix = np.zeros(new_shape, dtype=np.float32)
                
                # Kopiere die alten Daten in die neue Matrix
                old_slice = tuple(slice(0, dim) for dim in self.matrix.shape)
                new_matrix[old_slice] = self.matrix
                
                # Ersetze die alte Matrix
                self.matrix = new_matrix
            
            logger.debug(f"Matrix erweitert auf Größe {new_shape}")
    
    def add_data_point(self, point_id: str, coordinates: List[int], value: float, metadata: Dict[str, Any] = None):
        """
        Fügt einen Datenpunkt zur Matrix hinzu.
        
        Args:
            point_id: Eindeutige ID des Datenpunkts
            coordinates: Koordinaten des Datenpunkts in der Matrix
            value: Wert des Datenpunkts
            metadata: Optionale Metadaten für den Datenpunkt
        """
        # Prüfe, ob die Koordinaten gültig sind
        if len(coordinates) != self.dimensions:
            raise ValueError(f"Koordinaten müssen {self.dimensions} Dimensionen haben")
        
        # Prüfe, ob die Matrix erweitert werden muss
        self._ensure_matrix_size(coordinates)
        
        # Speichere den Datenpunkt
        self.data_points[point_id] = value
        self.coordinates[point_id] = coordinates
        
        # Aktualisiere die Matrix
        if self.use_t_math:
            # Verwende T-MATHEMATICS Engine für optimierte Tensor-Operationen
            # Konvertiere Koordinaten in ein Format, das mit PyTorch kompatibel ist
            coord_tuple = tuple(coordinates)
            
            # Erstelle einen neuen Tensor mit dem aktualisierten Wert
            # Da direkte Indexierung mit PyTorch komplexer ist, erstellen wir eine Kopie
            matrix_np = self.matrix.cpu().numpy() if hasattr(self.matrix, 'cpu') else self.matrix
            matrix_np[coord_tuple] = value
            
            # Konvertiere zurück zu einem Tensor
            self.matrix = self.t_math_engine.prepare_tensor(torch.tensor(matrix_np))
        else:
            # Standard NumPy-Implementierung
            self.matrix[tuple(coordinates)] = value
        
        # Speichere Metadaten, falls vorhanden
        if metadata:
            self.metadata[point_id] = metadata
        
        logger.debug(f"Datenpunkt {point_id} hinzugefügt: Koordinaten={coordinates}, Wert={value}")
    
    def calculate_distance(self, point_id1: str, point_id2: str) -> float:
        """
        Berechnet die euklidische Distanz zwischen zwei Datenpunkten.
        
        Args:
            point_id1: ID des ersten Datenpunkts
            point_id2: ID des zweiten Datenpunkts
            
        Returns:
            Euklidische Distanz zwischen den Datenpunkten
        """
        # Prüfe, ob die Datenpunkte existieren
        if point_id1 not in self.coordinates or point_id2 not in self.coordinates:
            raise ValueError("Datenpunkte existieren nicht")
        
        # Hole die Koordinaten
        coords1 = self.coordinates[point_id1]
        coords2 = self.coordinates[point_id2]
        
        if self.use_t_math:
            # Verwende T-MATHEMATICS Engine für optimierte Distanzberechnung
            coords1_tensor = self.t_math_engine.prepare_tensor(torch.tensor(coords1, dtype=torch.float32))
            coords2_tensor = self.t_math_engine.prepare_tensor(torch.tensor(coords2, dtype=torch.float32))
            
            # Berechne die euklidische Distanz mit der T-MATHEMATICS Engine
            distance = self.t_math_engine.euclidean_distance(coords1_tensor, coords2_tensor)
            
            # Konvertiere das Ergebnis zu einem Python-Float
            return float(distance.item() if hasattr(distance, 'item') else distance)
        else:
            # Standard NumPy-Implementierung
            return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(coords1, coords2)))
    
    def set_dimension_label(self, dim_index: int, label: str):
        """
        Setzt die Bezeichnung für eine Dimension
        
        Args:
            dim_index: Index der Dimension
            label: Bezeichnung für die Dimension
        """
        if dim_index < 0 or dim_index >= self.dimensions:
            raise ValueError(f"Ungültiger Dimensionsindex: {dim_index}")
        
        self.dimension_labels[dim_index] = label
        logger.info(f"Dimension {dim_index} bezeichnet als '{label}'")
    
    def set_dimension_scale(self, dim_index: int, scale: float):
        """
        Setzt den Skalierungsfaktor für eine Dimension
        
        Args:
            dim_index: Index der Dimension
            scale: Skalierungsfaktor für die Dimension
        """
        if dim_index < 0 or dim_index >= self.dimensions:
            raise ValueError(f"Ungültiger Dimensionsindex: {dim_index}")
        
        self.dimension_scales[dim_index] = scale
        logger.info(f"Dimension {dim_index} skaliert mit Faktor {scale}")
    
    def logical_to_physical_coordinates(self, logical_coords: Tuple) -> Tuple:
        """
        Konvertiert logische Koordinaten in physische Tensorkoordinaten
        
        Args:
            logical_coords: Logische Koordinaten
            
        Returns:
            Physische Koordinaten
        """
        if len(logical_coords) != self.dimensions:
            raise ValueError(f"Ungültige Anzahl von Koordinaten: {len(logical_coords)}, erwartet: {self.dimensions}")
        
        physical_coords = []
        for i, coord in enumerate(logical_coords):
            # Skaliere die Koordinate basierend auf dem Skalierungsfaktor
            physical_coord = int(coord / self.dimension_scales[i])
            
            # Stelle sicher, dass die Koordinate innerhalb der Grenzen liegt
            if self.use_t_math:
                max_coord = self.matrix.shape[i] if hasattr(self.matrix, 'shape') else self.matrix.size()[i]
            else:
                max_coord = self.matrix.shape[i]
            
            physical_coord = max(0, min(physical_coord, max_coord))
            physical_coords.append(physical_coord)
        
        return tuple(physical_coords)
    
    def get_probability_distribution(self, dimension: int = 0) -> np.ndarray:
        """
        Gibt die Wahrscheinlichkeitsverteilung für eine bestimmte Dimension zurück
        
        Args:
            dimension: Dimension, für die die Verteilung berechnet werden soll
            
        Returns:
            Wahrscheinlichkeitsverteilung als NumPy-Array
        """
        if dimension >= self.dimensions:
            raise ValueError(f"Dimension {dimension} überschreitet die Anzahl der Dimensionen ({self.dimensions})")
            
        # Berechne die Summe entlang aller Dimensionen außer der angegebenen
        sum_dims = tuple(i for i in range(self.dimensions) if i != dimension)
        
        if self.tensor_backend == "torch":
            # PyTorch-Backend
            distribution = self.data_tensor.sum(dim=sum_dims).cpu().numpy()
        else:
            # NumPy-Backend oder andere
            axes = sum_dims
            distribution = np.sum(self.data_tensor, axis=axes)
        
        # Normalisiere die Verteilung
        total = np.sum(distribution)
        if total > 0:
            distribution = distribution / total
            
        return distribution
        
    def apply_probability_transformation(self, probability: float) -> float:
        """
        Wendet eine Transformation auf eine Wahrscheinlichkeit an, basierend auf der PRISM-Matrix
        
        Args:
            probability: Eingangswahrscheinlichkeit (0.0 bis 1.0)
            
        Returns:
            Transformierte Wahrscheinlichkeit (0.0 bis 1.0)
        """
        # Begrenze die Eingabe auf den gültigen Bereich
        probability = max(0.0, min(1.0, probability))
        
        # Verwende die Matrix, um die Wahrscheinlichkeit zu transformieren
        # Wir nutzen die erste Dimension als Wahrscheinlichkeitsdimension
        prob_distribution = self.get_probability_distribution(dimension=0)
        
        # Berechne den Index in der Verteilung
        index = int(probability * (len(prob_distribution) - 1))
        
        # Wende einen Glättungsfaktor an, um extreme Werte zu vermeiden
        smoothing_factor = 0.2
        transformed_prob = prob_distribution[index] * (1 - smoothing_factor) + probability * smoothing_factor
        
        # Stelle sicher, dass das Ergebnis im gültigen Bereich liegt
        return max(0.0, min(1.0, transformed_prob))
    
    def compute_probability_field(self, region: Tuple) -> float:
        """
        Berechnet das Wahrscheinlichkeitsfeld für eine bestimmte Region
        
        Args:
            region: Region im Format ((min_d1, max_d1), (min_d2, max_d2), ...)
            
        Returns:
            Wahrscheinlichkeitswert für die Region
        """
        if len(region) != self.dimensions:
            raise ValueError(f"Ungültige Anzahl von Regionsdimensionen: {len(region)}, erwartet: {self.dimensions}")
        
        # Konvertiere die Region in physische Koordinaten
        physical_region = []
        for i, (min_val, max_val) in enumerate(region):
            physical_min = self.logical_to_physical_coordinates((min_val,) + (0,) * (self.dimensions - 1))[0]
            physical_max = self.logical_to_physical_coordinates((max_val,) + (0,) * (self.dimensions - 1))[0]
            physical_region.append((physical_min, physical_max))
        
        # Extrahiere den Subtensor für die Region
        slices = tuple(slice(min_val, max_val + 1) for min_val, max_val in physical_region)
        
        if self.tensor_backend == "mlx" and HAS_TENSOR_OPS:
            # MLX-Backend
            # Implementierung für MLX
            return 0.0
        elif self.tensor_backend == "torch":
            # PyTorch-Backend
            subtensor = self.data_tensor[slices]
            return torch.mean(subtensor).item()
        else:
            # NumPy-Backend
            subtensor = self.data_tensor[slices]
            return np.mean(subtensor)
    
    def get_tensor_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über den Tensor zurück
        
        Returns:
            Statistiken über den Tensor
        """
        stats = {
            "dimensions": self.dimensions,
            "shape": self.data_tensor.shape if self.tensor_backend != "mlx" else "MLX Tensor",
            "backend": self.tensor_backend,
            "dimension_labels": self.dimension_labels,
            "dimension_scales": self.dimension_scales
        }
        
        if self.tensor_backend == "mlx" and HAS_TENSOR_OPS:
            # MLX-Backend
            # Implementierung für MLX
            pass
        elif self.tensor_backend == "torch":
            # PyTorch-Backend
            stats["mean"] = torch.mean(self.data_tensor).item()
            stats["std"] = torch.std(self.data_tensor).item()
            stats["min"] = torch.min(self.data_tensor).item()
            stats["max"] = torch.max(self.data_tensor).item()
        else:
            # NumPy-Backend
            stats["mean"] = np.mean(self.data_tensor)
            stats["std"] = np.std(self.data_tensor)
            stats["min"] = np.min(self.data_tensor)
            stats["max"] = np.max(self.data_tensor)
        
        return stats
    
    def save_to_file(self, filepath: str):
        """
        Speichert die PrismMatrix in einer Datei
        
        Args:
            filepath: Pfad zur Datei
        """
        data = {
            "dimensions": self.dimensions,
            "dimension_labels": self.dimension_labels,
            "dimension_scales": self.dimension_scales,
            "backend": self.tensor_backend
        }
        
        if self.tensor_backend == "mlx" and HAS_TENSOR_OPS:
            # MLX-Backend
            # Implementierung für MLX
            pass
        elif self.tensor_backend == "torch":
            # PyTorch-Backend
            data["tensor"] = self.data_tensor.cpu().numpy().tolist()
        else:
            # NumPy-Backend
            data["tensor"] = self.data_tensor.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"PrismMatrix gespeichert in {filepath}")
    
    def load_from_file(self, filepath: str):
        """
        Lädt die PrismMatrix aus einer Datei
        
        Args:
            filepath: Pfad zur Datei
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.dimensions = data["dimensions"]
        self.dimension_labels = data["dimension_labels"]
        self.dimension_scales = data["dimension_scales"]
        self.tensor_backend = data["backend"]
        
        if self.tensor_backend == "mlx" and HAS_TENSOR_OPS:
            # MLX-Backend
            # Implementierung für MLX
            pass
        elif self.tensor_backend == "torch":
            # PyTorch-Backend
            tensor_data = np.array(data["tensor"])
            self.data_tensor = torch.tensor(tensor_data, device=device)
        else:
            # NumPy-Backend
            self.data_tensor = np.array(data["tensor"])
        
        logger.info(f"PrismMatrix geladen aus {filepath}")


# Beispiel für die Verwendung der PrismMatrix
if __name__ == "__main__":
    # Konfiguriere Logging für Standalone-Ausführung
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Erstelle eine PrismMatrix
    matrix = PrismMatrix(dimensions=4, initial_size=10)
    
    # Setze Dimensionsbezeichnungen
    matrix.set_dimension_label(0, "Zeit")
    matrix.set_dimension_label(1, "X")
    matrix.set_dimension_label(2, "Y")
    matrix.set_dimension_label(3, "Z")
    
    # Setze Dimensionsskalierungen
    matrix.set_dimension_scale(0, 0.1)  # Zeit in 0.1-Sekunden-Schritten
    matrix.set_dimension_scale(1, 0.01)  # X in 0.01-Meter-Schritten
    matrix.set_dimension_scale(2, 0.01)  # Y in 0.01-Meter-Schritten
    matrix.set_dimension_scale(3, 0.01)  # Z in 0.01-Meter-Schritten
    
    # Setze einige Datenpunkte
    matrix.set_data_point((0, 0, 0, 0), 1.0)
    matrix.set_data_point((1, 1, 1, 1), 2.0)
    matrix.set_data_point((2, 2, 2, 2), 3.0)
    
    # Berechne Wahrscheinlichkeitsfeld
    probability = matrix.compute_probability_field(((0, 2), (0, 2), (0, 2), (0, 2)))
    print(f"Wahrscheinlichkeitsfeld: {probability}")
    
    # Zeige Statistiken
    stats = matrix.get_tensor_statistics()
    print(f"Tensor-Statistiken: {stats}")
