#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Tiny Model Optimizer

Module zur Optimierung von Modellen für mobile oder energiearme Geräte.
Unterstützt verschiedene Optimierungstechniken wie Quantisierung, Pruning,
und Knowledge Distillation für die Reduktion der Modellgröße und des
Energieverbrauchs.

Teil von Phase 6: Föderiertes Lernen, Mobile-Optimierung und Autonomes Selbsttraining
"""

import os
import sys
import json
import time
import numpy as np
import logging
import copy
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from datetime import datetime

# Konfiguriere Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("miso_model_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MISO.TinyModelOptimizer")

class TinyModelOptimizer:
    """
    Optimiert Modelle für mobile oder energiearme Geräte.
    
    Unterstützt verschiedene Optimierungstechniken:
    - Quantisierung (float32 → float16/int8/int4)
    - Pruning (Entfernen unwichtiger Verbindungen)
    - Knowledge Distillation (Wissen von großen in kleinere Modelle übertragen)
    - Modellkompression für mobilen Einsatz
    - Export in mobile Formate (ONNX, TFLite, CoreML)
    
    Die Optimierung kann einzeln oder in Kombination angewendet werden,
    um maximale Effizienz bei minimaler Leistungseinbuße zu erzielen.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 backend: Optional[str] = None,
                 target_device: str = "mobile",
                 quantization_bits: int = 8,
                 pruning_ratio: float = 0.5,
                 distillation_temperature: float = 2.0,
                 enable_ane: bool = True):
        """
        Initialisiert den TinyModelOptimizer.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            output_dir: Ausgabeverzeichnis für optimierte Modelle
            backend: Zu verwendendes Backend (None = automatisch wählen)
            target_device: Zielgerät für die Optimierung ('mobile', 'edge', 'iot')
            quantization_bits: Bittiefe für die Quantisierung (8, 16, etc.)
            pruning_ratio: Anteil der zu entfernenden Verbindungen beim Pruning
            distillation_temperature: Temperatur für Knowledge Distillation
            enable_ane: Ob Apple Neural Engine-Optimierungen aktiviert werden sollen
        """
        # Lade Konfiguration
        self.config = self._load_config(config_path)
        
        # Setze Basisparameter
        self.output_dir = output_dir or self.config["optimization"]["output_dir"]
        self.output_dir = Path(self.output_dir)
        self.backend = backend or self._determine_optimal_backend()
        self.target_device = target_device.lower()
        
        # Optimierungsparameter
        self.quantization_bits = quantization_bits
        self.pruning_ratio = pruning_ratio
        self.distillation_temperature = distillation_temperature
        self.enable_ane = enable_ane
        
        # Erstelle Verzeichnisse
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "quantized", exist_ok=True)
        os.makedirs(self.output_dir / "pruned", exist_ok=True)
        os.makedirs(self.output_dir / "distilled", exist_ok=True)
        os.makedirs(self.output_dir / "mobile_exports", exist_ok=True)
        
        # Optimierungsstatistiken
        self.optimization_stats = {
            "timestamp": datetime.now().isoformat(),
            "backend": self.backend,
            "target_device": self.target_device,
            "quantization": {
                "original_size": 0,
                "optimized_size": 0,
                "bits": self.quantization_bits,
                "size_reduction": 0,
                "accuracy_loss": 0
            },
            "pruning": {
                "original_parameters": 0,
                "remaining_parameters": 0,
                "pruning_ratio": self.pruning_ratio,
                "accuracy_loss": 0
            },
            "distillation": {
                "teacher_size": 0,
                "student_size": 0,
                "temperature": self.distillation_temperature,
                "accuracy_loss": 0
            },
            "mobile_export": {
                "format": "",
                "size": 0,
                "inference_time": 0,
                "ane_compatible": False
            }
        }
        
        logger.info(f"TinyModelOptimizer initialisiert: Backend={self.backend}, "
                    f"Target={self.target_device}, ANE={self.enable_ane}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Lädt die Konfiguration aus einer JSON-Datei."""
        default_config = {
            "optimization": {
                "output_dir": "/Volumes/My Book/MISO_Ultimate 15.32.28/miso/optimized_models",
                "quantization": {
                    "default_bits": 8,
                    "methods": ["dynamic", "static", "post_training"],
                    "calibration_samples": 100
                },
                "pruning": {
                    "default_ratio": 0.5,
                    "methods": ["magnitude", "structured", "lottery_ticket"],
                    "fine_tuning_epochs": 5
                },
                "distillation": {
                    "default_temperature": 2.0,
                    "teacher_student_ratios": [4, 2, 1.5],
                    "training_epochs": 10
                },
                "mobile_export": {
                    "formats": ["onnx", "tflite", "coreml"],
                    "optimize_for_inference": True,
                    "enable_ane": True
                }
            },
            "device_profiles": {
                "mobile": {
                    "max_model_size_mb": 50,
                    "target_latency_ms": 100,
                    "memory_limit_mb": 200,
                    "battery_sensitive": True
                },
                "edge": {
                    "max_model_size_mb": 200,
                    "target_latency_ms": 200,
                    "memory_limit_mb": 500,
                    "battery_sensitive": True
                },
                "iot": {
                    "max_model_size_mb": 10,
                    "target_latency_ms": 50,
                    "memory_limit_mb": 50,
                    "battery_sensitive": True
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                
                # Rekursives Update des Standard-Configs
                def update_recursive(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                            update_recursive(d[k], v)
                        else:
                            d[k] = v
                
                update_recursive(default_config, custom_config)
                logger.info(f"Konfiguration aus {config_path} geladen")
            except Exception as e:
                logger.error(f"Fehler beim Laden der Konfiguration: {e}")
                logger.warning("Verwende Standard-Konfiguration")
        else:
            logger.info("Keine Konfigurationsdatei angegeben, verwende Standardwerte")
        
        return default_config
    
    def _determine_optimal_backend(self) -> str:
        """Bestimmt das optimale Backend basierend auf der verfügbaren Hardware."""
        import platform
        
        # Prüfe auf Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                # Prüfe, ob MLX verfügbar ist
                import mlx
                logger.info("Apple Silicon mit MLX erkannt")
                return "mlx"
            except ImportError:
                try:
                    # Prüfe, ob PyTorch mit MPS verfügbar ist
                    import torch
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        logger.info("Apple Silicon mit PyTorch MPS erkannt")
                        return "torch"
                except ImportError:
                    pass
        
        # Prüfe auf CUDA
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA-fähige GPU erkannt: {torch.cuda.get_device_name(0)}")
                return "torch"
        except ImportError:
            pass
        
        # Prüfe auf TensorFlow Lite
        try:
            import tensorflow as tf
            logger.info("TensorFlow erkannt, kann für TFLite-Export verwendet werden")
            return "tensorflow"
        except ImportError:
            pass
        
        # Fallback auf PyTorch (CPU)
        logger.info("Keine spezielle Hardware erkannt, verwende PyTorch auf CPU")
        return "torch"
    
    def quantize_model(self, model, calibration_data=None, method="dynamic"):
        """
        Reduziert die Gewichte auf kleinere Datentypen (z.B. float16, int8).
        
        Args:
            model: Das zu quantisierende Modell
            calibration_data: Daten für die Kalibrierung bei statischer Quantisierung
            method: Quantisierungsmethode ('dynamic', 'static', 'post_training')
            
        Returns:
            quantized_model: Das quantisierte Modell
            stats: Statistiken zur Quantisierung
        """
        logger.info(f"Starte Modellquantisierung mit Methode '{method}', Ziel-Bits: {self.quantization_bits}")
        start_time = time.time()
        
        # Bestimme ursprüngliche Modellgröße
        original_size = self._calculate_model_size(model)
        self.optimization_stats["quantization"]["original_size"] = original_size
        
        # Implementiere Quantisierung je nach Backend
        if self.backend == "torch":
            import torch
            
            if method == "dynamic":
                # Dynamische Quantisierung (nur Gewichte)
                quantized_model = self._torch_dynamic_quantization(model)
            elif method == "static" and calibration_data is not None:
                # Statische Quantisierung (Gewichte und Aktivierungen)
                quantized_model = self._torch_static_quantization(model, calibration_data)
            elif method == "post_training":
                # Post-Training Quantisierung
                quantized_model = self._torch_post_training_quantization(model)
            else:
                logger.warning(f"Unbekannte Quantisierungsmethode: {method}, verwende 'dynamic'")
                quantized_model = self._torch_dynamic_quantization(model)
        
        elif self.backend == "mlx":
            # MLX-Quantisierung (unterstützt float16 nativ, int8 über Konvertierung)
            quantized_model = self._mlx_quantization(model)
        
        elif self.backend == "tensorflow":
            # TensorFlow Lite Quantisierung
            quantized_model = self._tflite_quantization(model, calibration_data)
        
        else:
            logger.warning(f"Quantisierung für Backend {self.backend} nicht implementiert")
            quantized_model = model  # Fallback: unverändert zurückgeben
        
        # Bestimme neue Modellgröße und aktualisiere Statistiken
        quantized_size = self._calculate_model_size(quantized_model)
        self.optimization_stats["quantization"]["optimized_size"] = quantized_size
        self.optimization_stats["quantization"]["size_reduction"] = 1.0 - (quantized_size / max(original_size, 1))
        
        # Speichere das quantisierte Modell
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / "quantized" / f"quantized_model_b{self.quantization_bits}_{timestamp}.{self._get_model_extension()}"
        self._save_model(quantized_model, output_path)
        
        quantization_time = time.time() - start_time
        logger.info(f"Modellquantisierung abgeschlossen in {quantization_time:.2f}s. "
                    f"Größenreduktion: {self.optimization_stats['quantization']['size_reduction']*100:.2f}% "
                    f"({original_size/1024/1024:.2f}MB → {quantized_size/1024/1024:.2f}MB)")
        
        return quantized_model, self.optimization_stats["quantization"]
    
    def prune_model(self, model, validation_data=None, fine_tuning_data=None, method="magnitude"):
        """
        Entfernt unwichtige Neuronenverbindungen aus dem Modell.
        
        Args:
            model: Das zu pruning-optimierende Modell
            validation_data: Daten zur Validierung des Prunings
            fine_tuning_data: Daten zum Fine-Tuning nach dem Pruning
            method: Pruning-Methode ('magnitude', 'structured', 'lottery_ticket')
            
        Returns:
            pruned_model: Das optimierte Modell
            stats: Statistiken zum Pruning
        """
        logger.info(f"Starte Modell-Pruning mit Methode '{method}', Ziel-Ratio: {self.pruning_ratio}")
        start_time = time.time()
        
        # Bestimme ursprüngliche Parameteranzahl
        original_params = self._count_model_parameters(model)
        self.optimization_stats["pruning"]["original_parameters"] = original_params
        
        # Implementiere Pruning je nach Backend
        if self.backend == "torch":
            import torch
            
            if method == "magnitude":
                # Magnitude-based Pruning (entferne kleinste Gewichte)
                pruned_model = self._torch_magnitude_pruning(model, self.pruning_ratio)
            elif method == "structured":
                # Strukturiertes Pruning (entferne ganze Neuronen/Filter)
                pruned_model = self._torch_structured_pruning(model, self.pruning_ratio)
            elif method == "lottery_ticket":
                # Lottery Ticket Hypothesis Pruning
                pruned_model = self._torch_lottery_ticket_pruning(model, self.pruning_ratio)
            else:
                logger.warning(f"Unbekannte Pruning-Methode: {method}, verwende 'magnitude'")
                pruned_model = self._torch_magnitude_pruning(model, self.pruning_ratio)
            
            # Fine-Tuning nach dem Pruning
            if fine_tuning_data is not None:
                fine_tuning_epochs = self.config["optimization"]["pruning"]["fine_tuning_epochs"]
                logger.info(f"Führe Fine-Tuning für {fine_tuning_epochs} Epochen durch")
                pruned_model = self._torch_fine_tuning(pruned_model, fine_tuning_data, fine_tuning_epochs)
        
        elif self.backend == "mlx":
            # MLX-Pruning (experimentell)
            pruned_model = self._mlx_pruning(model, self.pruning_ratio)
            
            # Fine-Tuning nach dem Pruning
            if fine_tuning_data is not None:
                fine_tuning_epochs = self.config["optimization"]["pruning"]["fine_tuning_epochs"]
                pruned_model = self._mlx_fine_tuning(pruned_model, fine_tuning_data, fine_tuning_epochs)
        
        else:
            logger.warning(f"Pruning für Backend {self.backend} nicht implementiert")
            pruned_model = model  # Fallback: unverändert zurückgeben
        
        # Bestimme neue Parameteranzahl und aktualisiere Statistiken
        remaining_params = self._count_model_parameters(pruned_model)
        self.optimization_stats["pruning"]["remaining_parameters"] = remaining_params
        actual_pruning_ratio = 1.0 - (remaining_params / max(original_params, 1))
        
        # Validierung des Prunings (wenn Daten vorhanden)
        if validation_data is not None:
            accuracy_loss = self._evaluate_model_accuracy_loss(model, pruned_model, validation_data)
            self.optimization_stats["pruning"]["accuracy_loss"] = accuracy_loss
            logger.info(f"Genauigkeitsverlust durch Pruning: {accuracy_loss:.4f}")
        
        # Speichere das geprunte Modell
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / "pruned" / f"pruned_model_r{actual_pruning_ratio:.2f}_{timestamp}.{self._get_model_extension()}"
        self._save_model(pruned_model, output_path)
        
        pruning_time = time.time() - start_time
        logger.info(f"Modell-Pruning abgeschlossen in {pruning_time:.2f}s. "
                    f"Parameterreduktion: {actual_pruning_ratio*100:.2f}% "
                    f"({original_params:,} → {remaining_params:,} Parameter)")
        
        return pruned_model, self.optimization_stats["pruning"]
    
    def distill_model(self, teacher_model, student_model=None, training_data=None, epochs=None):
        """
        Überträgt Wissen von einem großen Modell (Lehrer) in ein kleineres Modell (Schüler).
        
        Args:
            teacher_model: Das große Quellmodell (Lehrer)
            student_model: Das kleine Zielmodell (Schüler), falls None wird eines erstellt
            training_data: Daten für das Distillation-Training
            epochs: Anzahl der Trainings-Epochen
            
        Returns:
            student_model: Das optimierte Schülermodell
            stats: Statistiken zur Knowledge Distillation
        """
        logger.info(f"Starte Knowledge Distillation mit Temperatur {self.distillation_temperature}")
        start_time = time.time()
        
        # Bestimme Größen der Modelle
        teacher_size = self._calculate_model_size(teacher_model)
        self.optimization_stats["distillation"]["teacher_size"] = teacher_size
        
        # Erstelle ein Schülermodell, falls keines übergeben wurde
        if student_model is None:
            student_model = self._create_student_model(teacher_model)
        
        student_size = self._calculate_model_size(student_model)
        
        # Setze Trainingsepochen
        if epochs is None:
            epochs = self.config["optimization"]["distillation"]["training_epochs"]
        
        # Überprüfe, ob Trainingsdaten vorhanden sind
        if training_data is None:
            logger.warning("Keine Trainingsdaten für Knowledge Distillation bereitgestellt")
            return student_model, self.optimization_stats["distillation"]
        
        # Implementiere Knowledge Distillation je nach Backend
        if self.backend == "torch":
            import torch
            student_model = self._torch_knowledge_distillation(
                teacher_model, student_model, training_data, epochs, self.distillation_temperature)
        
        elif self.backend == "mlx":
            student_model = self._mlx_knowledge_distillation(
                teacher_model, student_model, training_data, epochs, self.distillation_temperature)
        
        else:
            logger.warning(f"Knowledge Distillation für Backend {self.backend} nicht implementiert")
        
        # Aktualisiere Statistiken
        self.optimization_stats["distillation"]["student_size"] = student_size
        self.optimization_stats["distillation"]["size_ratio"] = student_size / max(teacher_size, 1)
        
        # Speichere das Schülermodell
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / "distilled" / f"distilled_model_{timestamp}.{self._get_model_extension()}"
        self._save_model(student_model, output_path)
        
        distillation_time = time.time() - start_time
        logger.info(f"Knowledge Distillation abgeschlossen in {distillation_time:.2f}s. "
                    f"Größenverhältnis: {self.optimization_stats['distillation']['size_ratio']:.2f} "
                    f"({teacher_size/1024/1024:.2f}MB → {student_size/1024/1024:.2f}MB)")
        
        return student_model, self.optimization_stats["distillation"]
    
    def export_for_mobile(self, model, input_shape=None, format="onnx", optimize_for_inference=True, example_input=None):
        """
        Exportiert ein Modell für mobile Plattformen in eines der Standardformate.
        
        Args:
            model: Das zu exportierende Modell
            input_shape: Eingabeform für das Modell (None = automatisch bestimmen)
            format: Exportformat ('onnx', 'tflite', 'coreml')
            optimize_for_inference: Ob das Modell für Inferenz optimiert werden soll
            example_input: Beispieleingabe für den Export
            
        Returns:
            export_path: Pfad zum exportierten Modell
            stats: Exportstatistiken
        """
        logger.info(f"Exportiere Modell für mobile Plattformen im Format: {format}")
        start_time = time.time()
        
        # Überprüfe Format
        format = format.lower()
        valid_formats = self.config["optimization"]["mobile_export"]["formats"]
        if format not in valid_formats:
            logger.warning(f"Ungültiges Exportformat: {format}, verwende 'onnx'")
            format = "onnx"
        
        self.optimization_stats["mobile_export"]["format"] = format
        
        # Bestimme Eingabeform, falls nicht angegeben
        if input_shape is None and example_input is None:
            # Standard-Eingabeform für Bildklassifizierung
            input_shape = (1, 3, 224, 224)  # Batch, Channels, Height, Width
            logger.info(f"Keine Eingabeform angegeben, verwende Standard: {input_shape}")
        
        # Erstelle Beispieleingabe, falls nicht angegeben
        if example_input is None:
            if self.backend == "torch":
                import torch
                example_input = torch.randn(input_shape)
            elif self.backend == "mlx":
                import mlx.core as mx
                example_input = mx.random.normal(input_shape)
            else:
                example_input = np.random.randn(*input_shape)
        
        # Exportiere je nach gewähltem Format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "onnx":
            export_path = self.output_dir / "mobile_exports" / f"model_{timestamp}.onnx"
            self._export_to_onnx(model, export_path, example_input, optimize_for_inference)
        
        elif format == "tflite":
            export_path = self.output_dir / "mobile_exports" / f"model_{timestamp}.tflite"
            self._export_to_tflite(model, export_path, example_input, optimize_for_inference)
        
        elif format == "coreml":
            export_path = self.output_dir / "mobile_exports" / f"model_{timestamp}.mlmodel"
            self._export_to_coreml(model, export_path, example_input, optimize_for_inference)
        
        # Prüfe Apple Neural Engine Kompatibilität
        if format == "coreml" and self.enable_ane:
            ane_compatible = self._check_ane_compatibility(export_path)
            self.optimization_stats["mobile_export"]["ane_compatible"] = ane_compatible
            if ane_compatible:
                logger.info("Modell ist mit Apple Neural Engine kompatibel")
            else:
                logger.info("Modell ist NICHT mit Apple Neural Engine kompatibel")
        
        # Bestimme Größe des exportierten Modells
        exported_size = os.path.getsize(export_path)
        self.optimization_stats["mobile_export"]["size"] = exported_size
        
        # Messe Inferenzzeit (simuliert)
        inference_time = self._measure_inference_time(model, example_input)
        self.optimization_stats["mobile_export"]["inference_time"] = inference_time
        
        export_time = time.time() - start_time
        logger.info(f"Modellexport abgeschlossen in {export_time:.2f}s. "
                    f"Format: {format}, Größe: {exported_size/1024/1024:.2f}MB, "
                    f"Inferenzzeit: {inference_time*1000:.2f}ms")
        
        return str(export_path), self.optimization_stats["mobile_export"]
    
    def _calculate_model_size(self, model):
        """Berechnet die Größe eines Modells in Bytes."""
        if self.backend == "torch":
            import torch
            import io
            buffer = io.BytesIO()
            torch.save(model, buffer)
            size = buffer.tell()
            return size
        elif self.backend == "mlx":
            import mlx.nn as nn
            import tempfile
            with tempfile.NamedTemporaryFile() as f:
                nn.save_weights(f.name, model)
                size = os.path.getsize(f.name)
            return size
        else:
            # Generische Berechnung für andere Modelltypen
            total_size = 0
            
            # Für dict-basierte Modelle
            if isinstance(model, dict):
                for key, value in model.items():
                    if hasattr(value, 'nbytes'):
                        total_size += value.nbytes
                    elif isinstance(value, np.ndarray):
                        total_size += value.nbytes
                    else:
                        try:
                            total_size += sys.getsizeof(value)
                        except:
                            pass
            else:
                # Fallback für andere Modelltypen
                try:
                    total_size = sys.getsizeof(model)
                except:
                    total_size = 1024 * 1024  # Standardgröße 1MB wenn nicht berechenbar
            
            return total_size
    
    def _count_model_parameters(self, model):
        """Zählt die Anzahl der Parameter in einem Modell."""
        if self.backend == "torch":
            import torch.nn as nn
            if isinstance(model, nn.Module):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
        elif self.backend == "mlx":
            import mlx.nn as nn
            if hasattr(model, 'parameters'):
                return sum(p.size for p in model.parameters().values())
        
        # Generische Zählung für dict-basierte Modelle
        if isinstance(model, dict):
            total_params = 0
            for key, value in model.items():
                if hasattr(value, 'size'):
                    total_params += value.size
                elif isinstance(value, np.ndarray):
                    total_params += value.size
            return total_params
        
        # Fallback
        return 0
    
    def _get_model_extension(self):
        """Gibt die passende Dateierweiterung für das aktuelle Backend zurück."""
        if self.backend == "torch":
            return "pt"
        elif self.backend == "mlx":
            return "npz"
        elif self.backend == "tensorflow":
            return "h5"
        else:
            return "bin"
    
    def _save_model(self, model, file_path):
        """Speichert ein Modell in einer Datei."""
        try:
            if self.backend == "torch":
                import torch
                torch.save(model, file_path)
            elif self.backend == "mlx":
                import mlx.nn as nn
                nn.save_weights(file_path, model)
            else:
                # Generische Speichermethode
                np.savez_compressed(file_path, **model if isinstance(model, dict) else {"model": model})
            
            logger.info(f"Modell gespeichert: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Modells: {e}")
            return False
    
    def _measure_inference_time(self, model, example_input, num_runs=10):
        """Misst die durchschnittliche Inferenzzeit eines Modells."""
        if self.backend == "torch":
            import torch
            # Stelle sicher, dass wir im Evaluierungsmodus sind
            if hasattr(model, 'eval'):
                model.eval()
            
            # Wärmlauf
            with torch.no_grad():
                _ = model(example_input)
            
            # Zeitmessung
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(example_input)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            return avg_time
        
        elif self.backend == "mlx":
            import mlx.core as mx
            
            # Wärmlauf
            _ = model(example_input)
            mx.eval(model.parameters())
            
            # Zeitmessung
            start_time = time.time()
            for _ in range(num_runs):
                _ = model(example_input)
                mx.eval(model.parameters())
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            return avg_time
        
        else:
            # Generische Zeitmessung für andere Modelltypen
            start_time = time.time()
            for _ in range(num_runs):
                _ = model(example_input)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            return avg_time
    
    # PyTorch-spezifische Quantisierungsmethoden
    def _torch_dynamic_quantization(self, model):
        """Implementiert dynamische Quantisierung für PyTorch-Modelle."""
        import torch
        import torch.quantization
        
        # Konvertiere zu CPU für Quantisierung
        model = model.cpu()
        
        # Lege fest, welche Operationen quantisiert werden sollen
        qconfig_mapping = torch.quantization.get_default_qconfig_mapping()
        
        # Bestimme Quantisierungstyp basierend auf Bits
        if self.quantization_bits == 8:
            # Int8-Quantisierung
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
        elif self.quantization_bits == 16:
            # Float16-Quantisierung
            quantized_model = model.half()  # Konvertiere zu float16
        else:
            logger.warning(f"Ununterstützte Bittiefe für PyTorch: {self.quantization_bits}, verwende int8")
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
        
        return quantized_model
    
    def _torch_static_quantization(self, model, calibration_data):
        """Implementiert statische Quantisierung für PyTorch-Modelle."""
        import torch
        import torch.quantization
        
        # Konvertiere zu CPU für Quantisierung
        model = model.cpu()
        
        # Füge Quantisierungsstubs hinzu (erfordert Modellmodifikation)
        if hasattr(model, 'qconfig'):
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Kalibriere das Modell mit den Kalibrierungsdaten
            model.eval()
            with torch.no_grad():
                for data in calibration_data:
                    if isinstance(data, tuple) and len(data) >= 2:
                        inputs, _ = data[:2]  # Nimm Eingabe und ignoriere Labels
                    else:
                        inputs = data
                    _ = model(inputs)
            
            # Konvertiere zu einem quantisierten Modell
            torch.quantization.convert(model, inplace=True)
        else:
            logger.warning("Modell unterstützt keine statische Quantisierung, verwende dynamische")
            model = self._torch_dynamic_quantization(model)
        
        return model
    
    def _torch_post_training_quantization(self, model):
        """Implementiert Post-Training-Quantisierung für PyTorch-Modelle."""
        import torch
        
        # Für komplexere Modelle könnte hier ein ONNX-Export mit Quantisierung verwendet werden
        # Für diese einfache Implementierung nutzen wir die dynamische Quantisierung
        return self._torch_dynamic_quantization(model)
    
    # MLX-spezifische Quantisierungsmethoden
    def _mlx_quantization(self, model):
        """Implementiert Quantisierung für MLX-Modelle."""
        import mlx.core as mx
        
        # Prüfe auf MISOTensor oder MLXTensor aus der T-Mathematics Engine
        if hasattr(model, 'to_dtype') and callable(model.to_dtype):
            # Vermutlich ein MISOTensor oder MLXTensor
            if self.quantization_bits == 16:
                return model.to_dtype(mx.float16)
            elif self.quantization_bits == 8 or self.quantization_bits == 4:
                logger.warning(f"MLX unterstützt derzeit keine native {self.quantization_bits}-bit Quantisierung, "  
                              f"verwende float16 als beste Alternative")
                return model.to_dtype(mx.float16)
            else:
                return model  # Originales Modell unverändert lassen
        
        # Implementierung für standard MLX-Modelle (dict mit Parametern)
        if isinstance(model, dict):
            quantized_model = {}
            for k, v in model.items():
                if isinstance(v, mx.array):
                    if self.quantization_bits == 16:
                        quantized_model[k] = v.astype(mx.float16)
                    elif self.quantization_bits == 8 or self.quantization_bits == 4:
                        # MLX unterstützt keine native int8/int4, daher verwenden wir float16
                        logger.warning(f"MLX unterstützt keine {self.quantization_bits}-bit Integer-Quantisierung, "
                                      f"verwende float16")
                        quantized_model[k] = v.astype(mx.float16)
                    else:
                        quantized_model[k] = v
                else:
                    quantized_model[k] = v
            return quantized_model
        
        # Implementierung für mlx.nn Module
        if hasattr(model, 'parameters'):
            params = model.parameters()
            quantized_params = {}
            
            for k, v in params.items():
                if isinstance(v, mx.array):
                    if self.quantization_bits == 16:
                        quantized_params[k] = v.astype(mx.float16)
                    elif self.quantization_bits == 8 or self.quantization_bits == 4:
                        # MLX unterstützt keine native int8/int4, daher verwenden wir float16
                        quantized_params[k] = v.astype(mx.float16)
                    else:
                        quantized_params[k] = v
                else:
                    quantized_params[k] = v
            
            # Aktualisiere Modell mit quantisierten Parametern
            model.update(quantized_params)
            return model
        
        logger.warning("Modelltyp wird nicht unterstützt für MLX-Quantisierung")
        return model
    
    # TFLite-Quantisierung (TensorFlow Lite)
    def _tflite_quantization(self, model, calibration_data):
        """Implementiert Quantisierung für TensorFlow Lite."""
        try:
            import tensorflow as tf
            
            # Konvertiere Modell zu TensorFlow-Format, wenn nötig
            if self.backend != "tensorflow":
                # Wenn model nicht direkt ein TF-Modell ist, müssten wir es konvertieren
                # Dies ist komplex und hängt stark vom Quellformat ab
                logger.warning("Automatische Konvertierung anderer Modelltypen zu TensorFlow nicht implementiert")
                return model
            
            # Erstelle TFLite-Konverter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Konfiguriere Quantisierung basierend auf bits
            if self.quantization_bits == 8:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # Für vollständige Integer-Quantisierung mit Kalibrierungsdaten
                if calibration_data is not None:
                    def representative_dataset():
                        for data in calibration_data:
                            if isinstance(data, tuple):
                                yield [tf.dtypes.cast(tf.expand_dims(data[0], axis=0), tf.float32)]
                            else:
                                yield [tf.dtypes.cast(tf.expand_dims(data, axis=0), tf.float32)]
                    
                    converter.representative_dataset = representative_dataset
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
            elif self.quantization_bits == 16:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            
            # Konvertiere zu TFLite
            tflite_model = converter.convert()
            
            # In dieser Beispielimplementierung geben wir tflite_model zurück
            # In einer echten Implementierung würde man es speichern und Referenz zurückgeben
            return tflite_model
        
        except ImportError:
            logger.error("TensorFlow nicht installiert, TFLite-Quantisierung nicht verfügbar")
            return model
        except Exception as e:
            logger.error(f"Fehler bei TFLite-Quantisierung: {e}")
            return model
    
    # PyTorch-Pruning-Methoden
    def _torch_magnitude_pruning(self, model, pruning_ratio):
        """Implementiert magnitudenbasiertes Pruning für PyTorch-Modelle."""
        try:
            import torch
            import torch.nn.utils.prune as prune
            
            # Kopie des Modells erstellen
            pruned_model = copy.deepcopy(model)
            
            # Anwenden von globalen magnitudenbasiertem Pruning auf alle linearen und Faltungsschichten
            parameters_to_prune = []
            for module in pruned_model.modules():
                if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, 'weight'))
            
            if parameters_to_prune:
                # Globales magnitudenbasiertes Pruning anwenden
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_ratio,
                )
                
                # Pruning permanent machen (optional)
                for module, name in parameters_to_prune:
                    prune.remove(module, name)
            
            return pruned_model
        
        except Exception as e:
            logger.error(f"Fehler bei PyTorch Magnitude Pruning: {e}")
            return model
    
    def _torch_structured_pruning(self, model, pruning_ratio):
        """Implementiert strukturiertes Pruning für PyTorch-Modelle (entfernt ganze Neuronen/Filter)."""
        try:
            import torch
            import torch.nn.utils.prune as prune
            
            # Kopie des Modells erstellen
            pruned_model = copy.deepcopy(model)
            
            # Strukturiertes Pruning auf Convolutional-Schichten anwenden (entfernt ganze Filter)
            for module in pruned_model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    # Strukturiertes L1-Pruning anwenden (entfernt Filter mit geringster L1-Norm)
                    prune.ln_structured(module, name='weight', amount=pruning_ratio, n=1, dim=0)
                    prune.remove(module, 'weight')
                
                # Strukturiertes Pruning auf Linear-Schichten anwenden (entfernt Ausgabeneuronen)
                elif isinstance(module, torch.nn.Linear):
                    prune.ln_structured(module, name='weight', amount=pruning_ratio, n=1, dim=0)
                    prune.remove(module, 'weight')
            
            return pruned_model
        
        except Exception as e:
            logger.error(f"Fehler bei PyTorch strukturiertem Pruning: {e}")
            return model
    
    def _torch_lottery_ticket_pruning(self, model, pruning_ratio, fine_tuning_data=None, max_iterations=3):
        """Implementiert Lottery Ticket Hypothesis Pruning für PyTorch-Modelle."""
        try:
            import torch
            import torch.nn.utils.prune as prune
            
            # Wenn keine Fine-Tuning-Daten verfügbar sind, führe einfaches Magnitude-Pruning durch
            if fine_tuning_data is None:
                logger.warning("Lottery Ticket Pruning benötigt Trainingsdaten, verwende Magnitude Pruning")
                return self._torch_magnitude_pruning(model, pruning_ratio)
            
            # Speichere ursprünglichen Zustand der Gewichte
            orig_state_dict = copy.deepcopy(model.state_dict())
            
            # Beste gefundene Maske und Leistung
            best_model = None
            best_accuracy = -float('inf')
            
            for iteration in range(max_iterations):
                # Pruned-Modell mit ursprünglichen Gewichten initialisieren
                reset_model = copy.deepcopy(model)
                reset_model.load_state_dict(orig_state_dict)
                
                # Schrittweise Pruning anwenden, beginnend mit geringeren Ratios
                current_ratio = pruning_ratio * (iteration + 1) / max_iterations
                pruned_model = self._torch_magnitude_pruning(reset_model, current_ratio)
                
                # Trainiere das beschnittene Modell kurz
                if fine_tuning_data is not None:
                    optimizer = torch.optim.Adam(pruned_model.parameters(), lr=0.001)
                    criterion = torch.nn.CrossEntropyLoss()
                    
                    pruned_model.train()
                    for data, target in fine_tuning_data:
                        optimizer.zero_grad()
                        output = pruned_model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                
                # Evaluiere das Modell
                pruned_model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in fine_tuning_data:
                        output = pruned_model(data)
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                
                accuracy = correct / total
                
                # Aktualisiere beste Maske, wenn dies die beste Leistung ist
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = copy.deepcopy(pruned_model)
                
                logger.info(f"Lottery Ticket Pruning Iteration {iteration + 1}/{max_iterations}: "
                          f"Ratio={current_ratio:.2f}, Accuracy={accuracy:.4f}")
            
            if best_model is not None:
                return best_model
            else:
                return self._torch_magnitude_pruning(model, pruning_ratio)
            
        except Exception as e:
            logger.error(f"Fehler bei PyTorch Lottery Ticket Pruning: {e}")
            return model
    
    def _torch_fine_tuning(self, model, fine_tuning_data, epochs=5):
        """Führt Fine-Tuning eines Modells nach dem Pruning durch."""
        try:
            import torch
            
            # Kopie des Modells erstellen
            fine_tuned_model = copy.deepcopy(model)
            
            # Training Setup
            optimizer = torch.optim.Adam(fine_tuned_model.parameters(), lr=0.0001)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Trainiere das Modell für die angegebene Anzahl von Epochen
            fine_tuned_model.train()
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for i, (data, target) in enumerate(fine_tuning_data):
                    # Forward-Pass
                    optimizer.zero_grad()
                    output = fine_tuned_model(data)
                    loss = criterion(output, target)
                    
                    # Backward-Pass und Optimierung
                    loss.backward()
                    optimizer.step()
                    
                    # Statistiken
                    running_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    
                    if i % 10 == 9:  # Print alle 10 Mini-Batches
                        logger.debug(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}, "
                                   f"Loss: {running_loss / 10:.4f}, Acc: {correct / total:.4f}")
                        running_loss = 0.0
                
                # Epochenergebnis
                accuracy = correct / total
                logger.info(f"Fine-Tuning Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")
            
            return fine_tuned_model
            
        except Exception as e:
            logger.error(f"Fehler bei PyTorch Fine-Tuning: {e}")
            return model
    
    # MLX-Pruning-Methoden
    def _mlx_magnitude_pruning(self, model, pruning_ratio):
        """Implementiert magnitudenbasiertes Pruning für MLX-Modelle."""
        try:
            import mlx.core as mx
            import numpy as np
            
            # Kopie der Modellparameter erstellen
            if hasattr(model, 'parameters'):
                params = model.parameters()
                pruned_params = {}
                
                # Für jede Schicht/Parameter im Modell
                for name, param in params.items():
                    if isinstance(param, mx.array):
                        # Konvertiere parameter zu NumPy für einfachere Verarbeitung
                        weights_np = param.tolist()
                        weights_np = np.array(weights_np)
                        
                        # Bestimme Schwellenwert basierend auf der Magnitude und pruning_ratio
                        abs_weights = np.abs(weights_np)
                        threshold = np.quantile(abs_weights, pruning_ratio)
                        
                        # Erstelle Maske und wende sie an
                        mask = abs_weights > threshold
                        pruned_weights = weights_np * mask
                        
                        # Konvertiere zurück zu MLX array
                        pruned_params[name] = mx.array(pruned_weights)
                    else:
                        pruned_params[name] = param
                
                # Aktualisiere das Modell mit den pruned Parametern
                model.update(pruned_params)
                return model
            
            # Alternative für dict-basierte Modelle
            elif isinstance(model, dict):
                pruned_model = {}
                for name, param in model.items():
                    if isinstance(param, mx.array):
                        # Konvertiere parameter zu NumPy
                        weights_np = param.tolist()
                        weights_np = np.array(weights_np)
                        
                        # Bestimme Schwellenwert basierend auf der Magnitude und pruning_ratio
                        abs_weights = np.abs(weights_np)
                        threshold = np.quantile(abs_weights, pruning_ratio)
                        
                        # Erstelle Maske und wende sie an
                        mask = abs_weights > threshold
                        pruned_weights = weights_np * mask
                        
                        # Konvertiere zurück zu MLX array
                        pruned_model[name] = mx.array(pruned_weights)
                    else:
                        pruned_model[name] = param
                return pruned_model
            
            logger.warning("Modelltyp wird nicht unterstützt für MLX-Pruning")
            return model
            
        except Exception as e:
            logger.error(f"Fehler bei MLX Magnitude Pruning: {e}")
            return model
    
    def _mlx_structured_pruning(self, model, pruning_ratio):
        """Implementiert strukturiertes Pruning für MLX-Modelle."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import numpy as np
            
            # MLX hat keine direkte Unterstützung für strukturiertes Pruning
            # Implementieren wir eine einfache Version für lineare Schichten
            if hasattr(model, 'parameters'):
                params = model.parameters()
                pruned_params = {}
                
                for name, param in params.items():
                    # Pruning für Gewichte in linearen Schichten
                    if isinstance(param, mx.array) and name.endswith('weight'):
                        weights_np = param.tolist()
                        weights_np = np.array(weights_np)
                        
                        # Für strukturiertes Pruning (entfernt ganze Neuronen/Filter)
                        # Berechne L1-Norm für jede Ausgabeneuron/Filter
                        if len(weights_np.shape) >= 2:  # Mindestens 2D für Lineare oder Conv Schichten
                            norms = np.sum(np.abs(weights_np), axis=tuple(range(1, len(weights_np.shape))))
                            num_to_prune = int(pruning_ratio * len(norms))
                            
                            if num_to_prune > 0:
                                # Finde die Indizes der Neuronen mit den niedrigsten Normen
                                indices_to_prune = np.argsort(norms)[:num_to_prune]
                                
                                # Erstelle eine Maske (1 für behalten, 0 für prunen)
                                mask = np.ones(norms.shape, dtype=bool)
                                mask[indices_to_prune] = False
                                
                                # Erstelle die expandierte Maske für die gesamte Gewichtsmatrix
                                expanded_mask = np.expand_dims(mask, axis=tuple(range(1, len(weights_np.shape))))
                                expanded_mask = np.broadcast_to(expanded_mask, weights_np.shape)
                                
                                # Wende die Maske an
                                pruned_weights = weights_np * expanded_mask
                                pruned_params[name] = mx.array(pruned_weights)
                            else:
                                pruned_params[name] = param
                        else:
                            pruned_params[name] = param
                    else:
                        pruned_params[name] = param
                
                # Aktualisiere das Modell mit den pruned Parametern
                model.update(pruned_params)
                return model
            
            # Für dict-basierte Modelle
            elif isinstance(model, dict):
                # Ähnliche Implementierung wie oben, angepasst für dict-basierte Modelle
                pruned_model = {}
                for name, param in model.items():
                    if isinstance(param, mx.array) and ('weight' in name or 'kernel' in name):
                        weights_np = param.tolist()
                        weights_np = np.array(weights_np)
                        
                        if len(weights_np.shape) >= 2:  # Mindestens 2D
                            norms = np.sum(np.abs(weights_np), axis=tuple(range(1, len(weights_np.shape))))
                            num_to_prune = int(pruning_ratio * len(norms))
                            
                            if num_to_prune > 0:
                                indices_to_prune = np.argsort(norms)[:num_to_prune]
                                mask = np.ones(norms.shape, dtype=bool)
                                mask[indices_to_prune] = False
                                
                                expanded_mask = np.expand_dims(mask, axis=tuple(range(1, len(weights_np.shape))))
                                expanded_mask = np.broadcast_to(expanded_mask, weights_np.shape)
                                
                                pruned_weights = weights_np * expanded_mask
                                pruned_model[name] = mx.array(pruned_weights)
                            else:
                                pruned_model[name] = param
                        else:
                            pruned_model[name] = param
                    else:
                        pruned_model[name] = param
                return pruned_model
            
            logger.warning("Modelltyp wird nicht unterstützt für MLX strukturiertes Pruning")
            return model
            
        except Exception as e:
            logger.error(f"Fehler bei MLX strukturiertem Pruning: {e}")
            return model
            
    # Knowledge Distillation Methoden
    def _torch_knowledge_distillation(self, teacher_model, student_model, training_data, 
                                     temperature=3.0, alpha=0.5, epochs=10):
        """Implementiert Knowledge Distillation für PyTorch-Modelle."""
        try:
            import torch
            import torch.nn.functional as F
            
            # Stelle sicher, dass beide Modelle im Evaluation-Modus für den Teacher sind
            teacher_model.eval()
            student_model.train()
            
            # Optimizer für das Student-Modell
            optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
            
            # Training Loop
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for i, (inputs, targets) in enumerate(training_data):
                    optimizer.zero_grad()
                    
                    # Forward pass mit Teacher und Student
                    with torch.no_grad():
                        teacher_outputs = teacher_model(inputs)
                    
                    student_outputs = student_model(inputs)
                    
                    # Berechne zwei Verlustkomponenten
                    # 1. Distillation Loss: KL-Divergenz zwischen den weichen Zielverteilungen
                    # 2. Student Loss: Kreuzentropie zwischen Student-Output und harten Zielen
                    
                    # Soft targets aus Teacher mit Temperatur
                    soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
                    log_soft_student = F.log_softmax(student_outputs / temperature, dim=1)
                    distillation_loss = F.kl_div(log_soft_student, soft_targets, reduction='batchmean') * (temperature ** 2)
                    
                    # Hard targets loss (normale Kreuzentropie)
                    student_loss = F.cross_entropy(student_outputs, targets)
                    
                    # Kombinierter Verlust
                    loss = alpha * distillation_loss + (1 - alpha) * student_loss
                    
                    # Backward und Optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Statistiken
                    running_loss += loss.item()
                    _, predicted = torch.max(student_outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    if i % 10 == 9:  # Print alle 10 Mini-Batches
                        logger.debug(f"Knowledge Distillation Epoch {epoch + 1}/{epochs}, Batch {i + 1}, "
                                   f"Loss: {running_loss / 10:.4f}, Acc: {correct / total:.4f}")
                        running_loss = 0.0
                
                # Epochenergebnis
                accuracy = correct / total
                logger.info(f"Knowledge Distillation Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")
            
            return student_model
            
        except Exception as e:
            logger.error(f"Fehler bei PyTorch Knowledge Distillation: {e}")
            return student_model
    
    def _mlx_knowledge_distillation(self, teacher_model, student_model, training_data, 
                                   temperature=3.0, alpha=0.5, epochs=10):
        """Implementiert Knowledge Distillation für MLX-Modelle."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            
            # Definiere Softmax mit Temperatur
            def softmax_with_temperature(logits, temperature):
                logits = logits / temperature
                max_logits = mx.max(logits, axis=1, keepdims=True)
                exp_logits = mx.exp(logits - max_logits)
                sum_exp = mx.sum(exp_logits, axis=1, keepdims=True)
                return exp_logits / sum_exp
            
            # Definiere KL-Divergenz
            def kl_divergence(y_true, y_pred):
                y_true = mx.clip(y_true, 1e-7, 1.0)
                y_pred = mx.clip(y_pred, 1e-7, 1.0)
                return mx.sum(y_true * mx.log(y_true / y_pred), axis=1)
            
            # Definiere Cross-Entropy-Verlust
            def cross_entropy(logits, targets):
                return nn.losses.cross_entropy(logits, targets)
            
            # Definiere Gesamt-Loss-Funktion
            def loss_fn(teacher_outputs, student_outputs, targets, temperature, alpha):
                # Soft targets mit Temperatur
                soft_targets = softmax_with_temperature(teacher_outputs, temperature)
                soft_student = softmax_with_temperature(student_outputs, temperature)
                
                # KL-Divergenz berechnen
                distillation_loss = mx.mean(kl_divergence(soft_targets, soft_student)) * (temperature ** 2)
                
                # Hard targets loss
                student_loss = cross_entropy(student_outputs, targets)
                
                # Kombinierter Verlust
                return alpha * distillation_loss + (1 - alpha) * student_loss
            
            # Erstelle Optimizer
            optimizer = optim.Adam(learning_rate=0.001)
            
            # Training Loop
            for epoch in range(epochs):
                total_loss = 0.0
                correct = 0
                total = 0
                
                for i, (inputs, targets) in enumerate(training_data):
                    # Konvertiere zu MLX Arrays, wenn nötig
                    if not isinstance(inputs, mx.array):
                        inputs = mx.array(inputs)
                    if not isinstance(targets, mx.array):
                        targets = mx.array(targets)
                    
                    # Forward pass mit Teacher
                    teacher_outputs = teacher_model(inputs)
                    mx.eval(teacher_outputs)  # Evaluiere sofort ohne Ableitungen zu speichern
                    
                    # Definiere den Forward- und Loss-Schritt für den Student
                    def forward_fn(params):
                        student_outputs = student_model.apply(params, inputs)
                        loss = loss_fn(teacher_outputs, student_outputs, targets, temperature, alpha)
                        return loss, student_outputs
                    
                    # Berechne Loss und Gradienten
                    params = student_model.parameters()
                    (loss, student_outputs), grads = mx.value_and_grad(forward_fn, has_aux=True)(params)
                    mx.eval(student_outputs)  # Evaluiere die Ausgaben
                    
                    # Update Parameter
                    updates, opt_state = optimizer.update(grads, optimizer.init(params))
                    params = optim.apply_updates(params, updates)
                    student_model.update(params)
                    
                    # Statistiken
                    total_loss += loss
                    predicted = mx.argmax(student_outputs, axis=1)
                    corrects = mx.sum(predicted == targets)
                    total += len(targets)
                    correct += corrects.item()
                    
                    if i % 10 == 9:  # Print alle 10 Mini-Batches
                        logger.debug(f"MLX Knowledge Distillation Epoch {epoch + 1}/{epochs}, Batch {i + 1}, "
                                   f"Loss: {total_loss / 10:.4f}, Acc: {correct / total:.4f}")
                        total_loss = 0.0
                
                # Epochenergebnis
                accuracy = correct / total
                logger.info(f"MLX Knowledge Distillation Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")
            
            return student_model
            
        except Exception as e:
            logger.error(f"Fehler bei MLX Knowledge Distillation: {e}")
            return student_model
