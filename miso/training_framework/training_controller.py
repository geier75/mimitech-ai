#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Training Controller

Zentraler Controller für das Training der MISO Ultimate AGI Modelle.
Unterstützt verschiedene Modellarchitekturen und Backends und optimiert
das Training basierend auf der verfügbaren Hardware.
"""

import os
import sys
import time
import json
import logging
import tempfile
import numpy as np
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Set, Generator
from datetime import datetime

# Konfiguriere Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("miso_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MISO.TrainingController")

class TrainingController:
    """
    Hauptcontroller für das Training der MISO Ultimate AGI Modelle.
    
    Unterstützt:
    - Verschiedene Modellarchitekturen (Transformer, LSTM, GPT, TopoNet)
    - Multiple Backends (MLX, PyTorch, TensorFlow, JAX)
    - Hardwareoptimierung (Apple Silicon, CUDA, CPU)
    - Verteiltes Training über mehrere Geräte
    - Multi-Sprachen-Training
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 data_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 model_type: str = "transformer",
                 backend: Optional[str] = None,
                 languages: List[str] = None,
                 distributed: bool = False):
        """
        Initialisiert den TrainingController.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            data_dir: Verzeichnis mit den Trainingsdaten
            output_dir: Ausgabeverzeichnis für trainierte Modelle und Logs
            model_type: Typ des zu trainierenden Modells
            backend: Zu verwendendes Backend (None = automatisch wählen)
            languages: Zu unterstützende Sprachen
            distributed: Ob verteiltes Training aktiviert werden soll
        """
        self.config = self._load_config(config_path)
        
        # Setze Basisparameter
        self.data_dir = Path(data_dir) if data_dir else Path(self.config["data_dir"])
        self.output_dir = Path(output_dir) if output_dir else Path(self.config["output_dir"])
        self.model_type = model_type.lower()
        self.languages = languages or self.config["languages"]
        self.distributed = distributed or self.config["training"]["distributed"]
        
        # Stelle sicher, dass die Verzeichnisse existieren
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "models", exist_ok=True)
        os.makedirs(self.output_dir / "logs", exist_ok=True)
        os.makedirs(self.output_dir / "checkpoints", exist_ok=True)
        os.makedirs(self.output_dir / "evaluation", exist_ok=True)
        
        # Bestimme das optimale Backend
        self.backend = backend or self._determine_optimal_backend()
        logger.info(f"Verwende Backend: {self.backend}")
        
        # Initialisiere das Trainingsmodul für das entsprechende Backend und Modell
        self.trainer = self._create_trainer()
        
        # Initialisiere Datenlader
        self.data_loader = self._create_data_loader()
        
        # Training Status und Metriken
        self.training_stats = {
            "start_time": None,
            "end_time": None,
            "epochs": 0,
            "current_epoch": 0,
            "best_loss": float("inf"),
            "best_accuracy": 0.0,
            "learning_rate": self.config["training"]["learning_rate"],
            "batch_size": self.config["training"]["batch_size"],
            "training_samples": 0,
            "validation_samples": 0,
            "time_per_epoch": [],
            "metrics": {
                "train_loss": [],
                "val_loss": [],
                "train_accuracy": [],
                "val_accuracy": []
            }
        }
        
        logger.info(f"TrainingController initialisiert: Modell={self.model_type}, Backend={self.backend}, "
                   f"Sprachen={self.languages}, Distributed={self.distributed}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Lädt die Konfiguration aus einer JSON-Datei."""
        default_config = {
            "data_dir": "/Volumes/My Book/miso/training_data",
            "output_dir": "/Volumes/My Book/miso/models",
            "languages": ["de", "en", "fr", "es"],
            "model_types": {
                "transformer": {
                    "hidden_size": 768,
                    "num_layers": 12,
                    "num_heads": 12,
                    "intermediate_size": 3072,
                    "dropout": 0.1
                },
                "lstm": {
                    "hidden_size": 1024,
                    "num_layers": 4,
                    "dropout": 0.1,
                    "bidirectional": True
                },
                "gpt": {
                    "hidden_size": 1024,
                    "num_layers": 24,
                    "num_heads": 16,
                    "intermediate_size": 4096,
                    "dropout": 0.1
                },
                "toponet": {
                    "hidden_size": 512,
                    "num_layers": 8,
                    "manifold_dim": 32,
                    "dropout": 0.1
                }
            },
            "training": {
                "learning_rate": 2e-5,
                "batch_size": 32,
                "epochs": 20,
                "warmup_steps": 1000,
                "gradient_accumulation_steps": 1,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "early_stopping_patience": 3,
                "early_stopping_min_delta": 0.001,
                "distributed": False,
                "mixed_precision": True,
                "dataset_subset_percentage": 100,  # Voller Datensatz
                "evaluation_steps": 500,
                "checkpoint_steps": 1000
            },
            "backends": {
                "mlx": {
                    "preferred_devices": ["apple"]
                },
                "torch": {
                    "preferred_devices": ["cuda", "mps", "cpu"]
                },
                "tensorflow": {
                    "preferred_devices": ["gpu", "cpu"]
                },
                "jax": {
                    "preferred_devices": ["gpu", "tpu", "cpu"]
                }
            },
            "tokenizer": {
                "vocab_size": 50000,
                "max_length": 512,
                "padding": "max_length",
                "truncation": True
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
        
        # Prüfe auf TensorFlow mit GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"TensorFlow mit GPU erkannt: {len(gpus)} verfügbar")
                return "tensorflow"
        except ImportError:
            pass
        
        # Prüfe auf JAX mit GPU/TPU
        try:
            import jax
            devices = jax.devices()
            if any(d.platform != 'cpu' for d in devices):
                logger.info(f"JAX mit Beschleuniger erkannt: {[d.platform for d in devices]}")
                return "jax"
        except ImportError:
            pass
        
        # Fallback auf PyTorch (CPU)
        logger.info("Keine spezielle Hardware erkannt, verwende PyTorch auf CPU")
        return "torch"
    
    def _create_trainer(self):
        """Erstellt den passenden Trainer für das gewählte Backend und Modelltyp."""
        if self.backend == "mlx":
            from miso.training_framework.backends.mlx_trainer import MLXTrainer
            return MLXTrainer(
                model_type=self.model_type,
                model_config=self.config["model_types"][self.model_type],
                training_config=self.config["training"],
                tokenizer_config=self.config["tokenizer"],
                languages=self.languages,
                output_dir=self.output_dir
            )
        elif self.backend == "torch":
            from miso.training_framework.backends.torch_trainer import TorchTrainer
            return TorchTrainer(
                model_type=self.model_type,
                model_config=self.config["model_types"][self.model_type],
                training_config=self.config["training"],
                tokenizer_config=self.config["tokenizer"],
                languages=self.languages,
                output_dir=self.output_dir,
                distributed=self.distributed
            )
        elif self.backend == "tensorflow":
            from miso.training_framework.backends.tf_trainer import TensorFlowTrainer
            return TensorFlowTrainer(
                model_type=self.model_type,
                model_config=self.config["model_types"][self.model_type],
                training_config=self.config["training"],
                tokenizer_config=self.config["tokenizer"],
                languages=self.languages,
                output_dir=self.output_dir,
                distributed=self.distributed
            )
        elif self.backend == "jax":
            from miso.training_framework.backends.jax_trainer import JAXTrainer
            return JAXTrainer(
                model_type=self.model_type,
                model_config=self.config["model_types"][self.model_type],
                training_config=self.config["training"],
                tokenizer_config=self.config["tokenizer"],
                languages=self.languages,
                output_dir=self.output_dir
            )
        else:
            raise ValueError(f"Nicht unterstütztes Backend: {self.backend}")
    
    def _create_data_loader(self):
        """Erstellt den passenden DataLoader für das Training und Validierung."""
        from miso.training_framework.data.multi_language_data_loader import MultiLanguageDataLoader
        
        return MultiLanguageDataLoader(
            data_dir=self.data_dir,
            languages=self.languages,
            tokenizer_config=self.config["tokenizer"],
            batch_size=self.config["training"]["batch_size"],
            subset_percentage=self.config["training"]["dataset_subset_percentage"]
        )
    
    def train(self, epochs: Optional[int] = None, resume_from: Optional[str] = None):
        """
        Startet das Training des Modells.
        
        Args:
            epochs: Anzahl der zu trainierenden Epochen
            resume_from: Checkpoint, von dem aus das Training fortgesetzt werden soll
        
        Returns:
            Dictionary mit Trainingsstatistiken
        """
        epochs = epochs or self.config["training"]["epochs"]
        
        # Lade Trainingsdaten
        logger.info("Lade Trainingsdaten...")
        train_data, val_data = self.data_loader.load_data()
        
        self.training_stats["training_samples"] = len(train_data)
        self.training_stats["validation_samples"] = len(val_data)
        
        logger.info(f"Geladen: {len(train_data)} Trainingsbeispiele, {len(val_data)} Validierungsbeispiele")
        
        # Lade Checkpoint, falls angegeben
        if resume_from:
            logger.info(f"Setze Training von Checkpoint {resume_from} fort")
            self.trainer.load_checkpoint(resume_from)
        
        # Starte das Training
        self.training_stats["start_time"] = datetime.now().isoformat()
        self.training_stats["epochs"] = epochs
        
        logger.info(f"Starte Training: {epochs} Epochen, Batch-Größe {self.config['training']['batch_size']}, "
                   f"LR {self.config['training']['learning_rate']}")
        
        for epoch in range(1, epochs + 1):
            self.training_stats["current_epoch"] = epoch
            epoch_start_time = time.time()
            
            # Eine Trainingsepoche
            train_metrics = self.trainer.train_epoch(train_data)
            
            # Validierung
            val_metrics = self.trainer.evaluate(val_data)
            
            # Aktualisiere Statistiken
            self.training_stats["metrics"]["train_loss"].append(train_metrics["loss"])
            self.training_stats["metrics"]["train_accuracy"].append(train_metrics["accuracy"])
            self.training_stats["metrics"]["val_loss"].append(val_metrics["loss"])
            self.training_stats["metrics"]["val_accuracy"].append(val_metrics["accuracy"])
            
            epoch_time = time.time() - epoch_start_time
            self.training_stats["time_per_epoch"].append(epoch_time)
            
            # Speichere Checkpoint
            if val_metrics["loss"] < self.training_stats["best_loss"]:
                self.training_stats["best_loss"] = val_metrics["loss"]
                self.training_stats["best_accuracy"] = val_metrics["accuracy"]
                checkpoint_path = self.output_dir / "checkpoints" / f"best_model_{self.model_type}_{self.backend}.pt"
                self.trainer.save_checkpoint(checkpoint_path)
                logger.info(f"Neues bestes Modell gespeichert: {checkpoint_path}")
            
            # Speichere Epochen-Checkpoint
            checkpoint_path = self.output_dir / "checkpoints" / f"epoch_{epoch}_{self.model_type}_{self.backend}.pt"
            self.trainer.save_checkpoint(checkpoint_path)
            
            # Early Stopping prüfen
            patience = self.config["training"]["early_stopping_patience"]
            min_delta = self.config["training"]["early_stopping_min_delta"]
            
            if patience > 0 and len(self.training_stats["metrics"]["val_loss"]) > patience:
                recent_losses = self.training_stats["metrics"]["val_loss"][-patience:]
                if all(recent_losses[i] - recent_losses[i-1] >= -min_delta for i in range(1, len(recent_losses))):
                    logger.info(f"Early Stopping nach {epoch} Epochen - keine Verbesserung seit {patience} Epochen")
                    break
            
            # Log Fortschritt
            logger.info(f"Epoche {epoch}/{epochs} abgeschlossen - "
                       f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} - "
                       f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f} - "
                       f"Zeit: {epoch_time:.2f}s")
        
        # Training abgeschlossen
        self.training_stats["end_time"] = datetime.now().isoformat()
        
        # Finales Modell speichern
        final_model_path = self.output_dir / "models" / f"final_{self.model_type}_{self.backend}.pt"
        self.trainer.save_checkpoint(final_model_path)
        
        # Speichere Trainingsstatistiken
        stats_path = self.output_dir / "logs" / f"training_stats_{self.model_type}_{self.backend}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        total_time = sum(self.training_stats["time_per_epoch"])
        logger.info(f"Training abgeschlossen - "
                   f"Beste Val Loss: {self.training_stats['best_loss']:.4f}, "
                   f"Beste Acc: {self.training_stats['best_accuracy']:.4f}, "
                   f"Gesamtzeit: {total_time:.2f}s")
        
        return self.training_stats
    
    def evaluate(self, 
                test_data=None, 
                checkpoint: Optional[str] = None, 
                output_file: Optional[str] = None):
        """
        Evaluiert das trainierte Modell auf einem Testdatensatz.
        
        Args:
            test_data: Testdaten oder None, um den Testdatensatz zu laden
            checkpoint: Pfad zum zu ladenden Modell-Checkpoint
            output_file: Datei, in der die Evaluationsergebnisse gespeichert werden sollen
            
        Returns:
            Dictionary mit Evaluationsmetriken
        """
        # Lade Checkpoint, falls angegeben
        if checkpoint:
            logger.info(f"Lade Modell-Checkpoint: {checkpoint}")
            self.trainer.load_checkpoint(checkpoint)
        
        # Lade Testdaten, falls nicht angegeben
        if test_data is None:
            logger.info("Lade Testdaten...")
            _, test_data = self.data_loader.load_test_data()
            logger.info(f"Geladen: {len(test_data)} Testbeispiele")
        
        # Führe Evaluation durch
        logger.info("Führe Modell-Evaluation durch...")
        metrics = self.trainer.evaluate(test_data)
        
        # Zusätzliche sprachspezifische Metriken berechnen
        language_metrics = {}
        for lang in self.languages:
            lang_data = self.data_loader.filter_by_language(test_data, lang)
            if lang_data:
                lang_metrics = self.trainer.evaluate(lang_data)
                language_metrics[lang] = lang_metrics
                logger.info(f"Sprache {lang}: Loss {lang_metrics['loss']:.4f}, Acc: {lang_metrics['accuracy']:.4f}")
        
        # Speichere Ergebnisse
        results = {
            "overall": metrics,
            "by_language": language_metrics,
            "model_type": self.model_type,
            "backend": self.backend,
            "timestamp": datetime.now().isoformat()
        }
        
        if output_file:
            output_path = output_file
        else:
            output_path = self.output_dir / "evaluation" / f"eval_{self.model_type}_{self.backend}.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation abgeschlossen - "
                   f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, "
                   f"Ergebnisse gespeichert in {output_path}")
        
        return results
    
    def export_model(self, export_format: str, output_path: Optional[str] = None, 
                   checkpoint: Optional[str] = None, optimize: bool = True):
        """
        Exportiert das trainierte Modell in ein bestimmtes Format.
        
        Args:
            export_format: Format für den Export ('onnx', 'torchscript', 'savedmodel', 'coreml')
            output_path: Pfad zum Speichern des exportierten Modells
            checkpoint: Pfad zum zu ladenden Modell-Checkpoint
            optimize: Ob das Modell für die Inferenz optimiert werden soll
            
        Returns:
            Pfad zum exportierten Modell
        """
        # Lade Checkpoint, falls angegeben
        if checkpoint:
            logger.info(f"Lade Modell-Checkpoint: {checkpoint}")
            self.trainer.load_checkpoint(checkpoint)
        
        # Erstelle Standardpfad, falls nicht angegeben
        if not output_path:
            output_path = self.output_dir / "models" / f"exported_{self.model_type}_{self.backend}.{export_format}"
        
        # Führe Export durch
        logger.info(f"Exportiere Modell im Format {export_format} nach {output_path}...")
        export_path = self.trainer.export_model(export_format, output_path, optimize)
        
        logger.info(f"Modell erfolgreich exportiert: {export_path}")
        
        return export_path


def main():
    """Hauptfunktion für die Kommandozeilenausführung."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MISO Ultimate AGI - Modelltraining")
    parser.add_argument("--config", type=str, default=None,
                        help="Pfad zur Konfigurationsdatei (JSON)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Verzeichnis mit den Trainingsdaten")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Ausgabeverzeichnis für trainierte Modelle und Logs")
    parser.add_argument("--model-type", type=str, default="transformer",
                        choices=["transformer", "lstm", "gpt", "toponet"],
                        help="Typ des zu trainierenden Modells")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["mlx", "torch", "tensorflow", "jax"],
                        help="Zu verwendendes Backend (None = automatisch wählen)")
    parser.add_argument("--languages", type=str, default="de,en,fr,es",
                        help="Komma-getrennte Liste der zu unterstützenden Sprachen")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Anzahl der Trainingsepochen")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint, von dem aus das Training fortgesetzt werden soll")
    parser.add_argument("--distributed", action="store_true",
                        help="Aktiviere verteiltes Training")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluiere das Modell nach dem Training")
    parser.add_argument("--export", type=str, default=None,
                        choices=["onnx", "torchscript", "savedmodel", "coreml"],
                        help="Exportiere das Modell im angegebenen Format")
    
    args = parser.parse_args()
    
    # Erstelle TrainingController
    controller = TrainingController(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        backend=args.backend,
        languages=args.languages.split(","),
        distributed=args.distributed
    )
    
    try:
        # Training durchführen
        train_stats = controller.train(epochs=args.epochs, resume_from=args.resume_from)
        
        print(f"\nTraining abgeschlossen:")
        print(f"  Modelltyp: {args.model_type}")
        print(f"  Backend: {controller.backend}")
        print(f"  Beste Validation Loss: {train_stats['best_loss']:.4f}")
        print(f"  Beste Accuracy: {train_stats['best_accuracy']:.4f}")
        print(f"  Trainingszeit pro Epoche: {np.mean(train_stats['time_per_epoch']):.2f}s")
        
        # Evaluation durchführen, falls angefordert
        if args.evaluate:
            eval_results = controller.evaluate()
            
            print(f"\nEvaluation:")
            print(f"  Gesamtbewertung: Loss={eval_results['overall']['loss']:.4f}, "
                  f"Accuracy={eval_results['overall']['accuracy']:.4f}")
            
            for lang, metrics in eval_results["by_language"].items():
                print(f"  {lang}: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        
        # Modell exportieren, falls angefordert
        if args.export:
            export_path = controller.export_model(args.export)
            print(f"\nModell exportiert nach: {export_path}")
    
    except KeyboardInterrupt:
        print("\nTraining wurde vom Benutzer unterbrochen.")
        logger.warning("Training wurde vom Benutzer unterbrochen.")
    except Exception as e:
        print(f"\nFehler beim Training: {e}")
        logger.error(f"Fehler beim Training: {e}")
        import traceback
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()
