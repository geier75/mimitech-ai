#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - PyTorch Trainer Wrapper

Diese Datei dient als Kompatibilitätsschicht und Brücke zum PyTorchTrainer.
Sie stellt sicher, dass bestehender Code, der den TorchTrainer erwartet,
weiterhin funktioniert, aber unsere neue PyTorchTrainer-Implementierung nutzt.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Konfiguriere Logger
logger = logging.getLogger("MISO.TorchTrainerWrapper")

# Importiere den PyTorchTrainer
from .pytorch_trainer_impl import PyTorchTrainer

# Alias-Klasse für den Trainer, um Abwärtskompatibilität zu gewährleisten
class TorchTrainer(PyTorchTrainer):
    """
    Kompatibilitätsklasse für den PyTorchTrainer.
    
    Diese Klasse passt das Interface des PyTorchTrainers an die im TrainingController
    erwartete Schnittstelle an.
    """
    
    def __init__(self,
                 model_type: str,
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 tokenizer_config: Dict[str, Any],
                 languages: List[str],
                 output_dir: str,
                 distributed: bool = False):
        """
        Initialisiert den TorchTrainer als Wrapper um den PyTorchTrainer.
        
        Args:
            model_type: Modelltyp (transformer, lstm, gpt, toponet)
            model_config: Konfiguration für das Modell
            training_config: Konfiguration für das Training
            tokenizer_config: Konfiguration für den Tokenizer
            languages: Liste der unterstützten Sprachen
            output_dir: Ausgabeverzeichnis für Modelle und Logs
            distributed: Flag für verteiltes Training
        """
        logger.info(f"Initialisiere TorchTrainer mit Modelltyp: {model_type}")
        
        # Bereite die Konfiguration für den PyTorchTrainer vor
        tokenizer_path = os.path.join(output_dir, "tokenizer", "tokenizer.model")
        if not os.path.exists(tokenizer_path):
            tokenizer_path = None
            logger.warning(f"Tokenizer-Modell nicht gefunden unter {tokenizer_path}")
        
        # Kombiniere die Konfigurationen für den PyTorchTrainer
        config = {
            "model_type": model_type,
            "vocab_size": tokenizer_config.get("vocab_size", 30000),
            "hidden_size": model_config.get("hidden_size", 768),
            "num_layers": model_config.get("num_layers", 12),
            "num_heads": model_config.get("num_heads", 12),
            "dropout": model_config.get("dropout", 0.1),
            "manifold_dim": model_config.get("manifold_dim", 32),
            "batch_size": training_config.get("batch_size", 32),
            "learning_rate": training_config.get("learning_rate", 5e-5),
            "weight_decay": training_config.get("weight_decay", 0.01),
            "num_training_steps": training_config.get("epochs", 3) * 1000,  # Wird später angepasst
            "num_warmup_steps": training_config.get("warmup_steps", 1000),
            "max_seq_length": tokenizer_config.get("max_length", 512),
            "num_languages": len(languages),
            "early_stopping_patience": training_config.get("early_stopping_patience", 3),
            "log_steps": 100,
            "eval_steps": training_config.get("evaluation_steps", 500),
            "save_steps": training_config.get("checkpoint_steps", 1000),
            "output_dir": os.path.join(output_dir, "checkpoints"),
            "log_dir": os.path.join(output_dir, "logs"),
            "languages": languages,
            "distributed": distributed,
            "mixed_precision": training_config.get("mixed_precision", True),
            "save_final_model": True,
            "miso_version": "1.0.0"
        }
        
        # Übergib die spezifischen Modellparameter
        if model_type == "transformer":
            config["intermediate_size"] = model_config.get("intermediate_size", 3072)
        elif model_type == "gpt":
            config["intermediate_size"] = model_config.get("intermediate_size", 4096)
        elif model_type == "lstm":
            config["bidirectional"] = model_config.get("bidirectional", True)
        
        # Erstelle den PyTorchTrainer mit der kombinierten Konfiguration
        super().__init__(
            config=config,
            model_path=None,  # Wird später gesetzt, wenn ein Modell geladen werden soll
            tokenizer_path=tokenizer_path,
            device="auto"  # Automatische Geräteauswahl
        )
        
        # Speichere zusätzliche Informationen
        self.model_type = model_type
        self.output_dir = output_dir
        self.languages = languages
        self.distributed = distributed
        
        logger.info(f"TorchTrainer initialisiert als Wrapper für PyTorchTrainer")
    
    def prepare_data(self, train_data, val_data=None):
        """
        Bereitet die Daten für das Training vor und erstellt den DataLoader.
        
        Args:
            train_data: Trainingsdaten
            val_data: Validierungsdaten (optional)
            
        Returns:
            Tuple aus Trainings- und Validierungs-DataLoader
        """
        import torch
        from torch.utils.data import Dataset, DataLoader, TensorDataset
        
        logger.info("Bereite Daten für PyTorch vor")
        
        class MISODataset(Dataset):
            def __init__(self, data):
                self.data = data
                
            def __len__(self):
                return len(self.data["input_ids"])
                
            def __getitem__(self, idx):
                item = {
                    "input_ids": torch.tensor(self.data["input_ids"][idx], dtype=torch.long),
                    "labels": torch.tensor(self.data["labels"][idx], dtype=torch.long)
                }
                
                if "attention_mask" in self.data:
                    item["attention_mask"] = torch.tensor(
                        self.data["attention_mask"][idx], dtype=torch.bool
                    )
                
                if "language_ids" in self.data:
                    item["language_ids"] = torch.tensor(
                        self.data["language_ids"][idx], dtype=torch.long
                    )
                
                return item
        
        # Erstelle Datasets
        train_dataset = MISODataset(train_data)
        val_dataset = MISODataset(val_data) if val_data else None
        
        # Erstelle DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4 if not self.distributed else 0,
            pin_memory=True
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=4 if not self.distributed else 0,
                pin_memory=True
            )
        
        # Aktualisiere die Anzahl der Trainingsschritte im Optimizer
        if hasattr(self, "scheduler") and self.scheduler is not None:
            num_training_steps = len(train_dataloader) * self.config.get("epochs", 3)
            self.config["num_training_steps"] = num_training_steps
            
            # Erstelle einen neuen Scheduler mit der aktualisierten Schrittzahl
            self._setup_optimizer()
            
        logger.info(f"Daten vorbereitet: {len(train_dataset)} Trainingsbeispiele, "
                    f"{len(val_dataset) if val_dataset else 0} Validierungsbeispiele")
        
        return train_dataloader, val_dataloader
    
    def run_training(self, train_dataloader, val_dataloader=None, epochs=None):
        """
        Führt das Training mit den gegebenen Datenloadern durch.
        
        Args:
            train_dataloader: DataLoader für Trainingsdaten
            val_dataloader: DataLoader für Validierungsdaten (optional)
            epochs: Anzahl der Trainingsepochen (optional, überschreibt Konfiguration)
            
        Returns:
            Dict mit Trainingsergebnissen
        """
        if epochs is None:
            epochs = self.config.get("epochs", 3)
        
        logger.info(f"Starte Training für {epochs} Epochen")
        
        # Rufe die train-Methode des PyTorchTrainers auf
        return self.train(train_dataloader, val_dataloader, num_epochs=epochs)
    
    def train_epoch(self, train_data):
        """
        Trainiert das Modell für eine Epoche.
        
        Args:
            train_data: Trainingsdaten
            
        Returns:
            Dict mit Trainingsmetriken für die Epoche
        """
        logger.info("Trainiere eine Epoche")
        
        # Bereite Daten vor
        train_dataloader, _ = self.prepare_data(train_data)
        
        # Setze Modell in Trainingsmodus
        self.model.train()
        
        # Initialisiere Metriken
        total_loss = 0
        total_samples = 0
        total_correct = 0
        total_tokens = 0
        batch_count = 0
        
        # Progress Bar
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(train_dataloader, desc="Training")
        except ImportError:
            iterator = train_dataloader
        
        # Training Loop über alle Batches
        for batch in iterator:
            # Batch auf das richtige Gerät verschieben
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            language_ids = batch.get("language_ids", None)
            if language_ids is not None:
                language_ids = language_ids.to(self.device)
            
            labels = batch["labels"].to(self.device)
            
            # Forward Pass
            self.model.zero_grad()
            outputs = self.model(input_ids, language_ids, attention_mask)
            
            # Berechne Loss
            import torch.nn.functional as F
            loss = F.cross_entropy(outputs.view(-1, self.vocab_size), labels.view(-1), reduction="sum")
            
            # Backward Pass und Optimierung
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Berechne Genauigkeit
            predictions = torch.argmax(outputs, dim=-1)
            mask = (labels != -100) if -100 in labels else None
            if mask is not None:
                correct = (predictions == labels).masked_select(mask).sum().item()
                total_tokens = mask.sum().item()
            else:
                correct = (predictions == labels).sum().item()
                total_tokens = labels.numel()
            
            # Aktualisiere Metriken
            batch_size = input_ids.size(0)
            total_loss += loss.item()
            total_samples += batch_size
            total_correct += correct
            batch_count += 1
            
            # Fortschrittsmeldung in tqdm (wenn verfügbar)
            current_loss = total_loss / total_tokens if total_tokens > 0 else 0
            current_acc = total_correct / total_tokens if total_tokens > 0 else 0
            
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})
            
            # Logging nach bestimmten Schritten
            if batch_count % 50 == 0:
                logger.info(f"Batch {batch_count}: Loss {current_loss:.4f}, Acc {current_acc:.4f}")
        
        # Berechne Durchschnittswerte
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        # Rückgabewerte für den Controller
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "samples": total_samples,
            "steps": batch_count
        }
    
    def evaluate(self, eval_data):
        """
        Evaluiert das Modell auf den gegebenen Daten.
        
        Args:
            eval_data: Evaluierungsdaten
            
        Returns:
            Dict mit Evaluierungsmetriken
        """
        logger.info("Starte Evaluation")
        
        # Bereite Daten vor
        _, eval_dataloader = self.prepare_data(None, eval_data)
        
        # Wenn kein Dataloader erstellt werden konnte
        if eval_dataloader is None:
            logger.error("Konnte keinen Evaluierungs-Dataloader erstellen")
            return {"loss": float("inf"), "accuracy": 0.0, "samples": 0}
        
        # Setze Modell in Evaluierungsmodus
        self.model.eval()
        
        # Initialisiere Metriken
        total_loss = 0
        total_samples = 0
        total_correct = 0
        total_tokens = 0
        
        # Deaktiviere Gradientenberechnung für Evaluation
        import torch
        with torch.no_grad():
            # Progress Bar (wenn tqdm verfügbar)
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(eval_dataloader, desc="Evaluation")
            except ImportError:
                iterator = eval_dataloader
            
            for batch in iterator:
                # Batch auf das richtige Gerät verschieben
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                language_ids = batch.get("language_ids", None)
                if language_ids is not None:
                    language_ids = language_ids.to(self.device)
                
                labels = batch["labels"].to(self.device)
                
                # Forward Pass
                outputs = self.model(input_ids, language_ids, attention_mask)
                
                # Berechne Loss
                import torch.nn.functional as F
                loss = F.cross_entropy(outputs.view(-1, self.vocab_size), labels.view(-1), reduction="sum")
                
                # Berechne Genauigkeit
                predictions = torch.argmax(outputs, dim=-1)
                mask = (labels != -100) if -100 in labels else None
                if mask is not None:
                    correct = (predictions == labels).masked_select(mask).sum().item()
                    num_tokens = mask.sum().item()
                else:
                    correct = (predictions == labels).sum().item()
                    num_tokens = labels.numel()
                
                # Aktualisiere Metriken
                batch_size = input_ids.size(0)
                total_loss += loss.item()
                total_samples += batch_size
                total_correct += correct
                total_tokens += num_tokens
        
        # Berechne Durchschnittswerte
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        
        # Log Ergebnisse
        logger.info(f"Evaluation abgeschlossen: Loss {avg_loss:.4f}, Acc {accuracy:.4f}")
        
        # Füge TensorBoard-Logging hinzu, falls verfügbar
        if hasattr(self, "summary_writer") and self.summary_writer is not None:
            self.summary_writer.add_scalar("eval/loss", avg_loss, self.global_step)
            self.summary_writer.add_scalar("eval/accuracy", accuracy, self.global_step)
        
        # Rückgabewerte für den Controller
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
            "samples": total_samples
        }
    
    def save_checkpoint(self, path=None):
        """
        Speichert einen Checkpoint des Modells.
        
        Args:
            path: Pfad zum Speichern (optional)
            
        Returns:
            Pfad zum gespeicherten Checkpoint
        """
        if path is None:
            path = os.path.join(self.output_dir, "checkpoints", f"checkpoint-{self.global_step}")
        
        logger.info(f"Speichere Checkpoint nach {path}")
        return self.save_model(path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Lädt einen Checkpoint des Modells.
        
        Args:
            checkpoint_path: Pfad zum Checkpoint
            
        Returns:
            True, wenn erfolgreich geladen, sonst False
        """
        logger.info(f"Lade Checkpoint von {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            if os.path.isdir(checkpoint_path):
                # Suche nach der model.pt-Datei im Verzeichnis
                model_path = os.path.join(checkpoint_path, "model.pt")
                config_path = os.path.join(checkpoint_path, "config.json")
            else:
                # Versuche, die Datei direkt zu laden
                model_path = checkpoint_path
                config_path = os.path.splitext(checkpoint_path)[0] + ".json"
        else:
            # Der Pfad existiert, prüfe, ob es eine Datei oder ein Verzeichnis ist
            if os.path.isdir(checkpoint_path):
                model_path = os.path.join(checkpoint_path, "model.pt")
                config_path = os.path.join(checkpoint_path, "config.json")
            else:
                model_path = checkpoint_path
                config_path = os.path.splitext(checkpoint_path)[0] + ".json"
        
        try:
            import torch
            # Prüfe, ob die Modelldatei existiert
            if not os.path.exists(model_path):
                logger.error(f"Modell-Checkpoint nicht gefunden unter {model_path}")
                return False
            
            # Lade Modell-State-Dict
            state_dict = torch.load(model_path, map_location="cpu")
            
            # Wenn der State-Dict ein Dictionary mit zusätzlichen Informationen ist
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                model_state = state_dict["model_state_dict"]
                
                # Falls vorhanden, lade auch den Optimizer- und Scheduler-Zustand
                if "optimizer_state_dict" in state_dict and self.optimizer is not None:
                    self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                
                if "scheduler_state_dict" in state_dict and self.scheduler is not None:
                    self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
                
                if "global_step" in state_dict:
                    self.global_step = state_dict["global_step"]
            else:
                # Angenommen, die Datei enthält nur den Modell-State-Dict
                model_state = state_dict
            
            # Lade den Modell-State
            self.model.load_state_dict(model_state)
            
            # Verschiebe Modell auf das richtige Gerät
            self.model.to(self.device)
            
            # Lade zusätzliche Konfigurationen, wenn verfügbar
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    checkpoint_config = json.load(f)
                
                # Aktualisiere aktuelle Konfiguration mit Werten aus dem Checkpoint
                self.config.update(checkpoint_config)
                
                logger.info(f"Konfiguration aus {config_path} geladen")
            
            logger.info(f"Modell erfolgreich von {model_path} geladen")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Checkpoints: {e}")
            return False
    
    def export_for_inference(self, path=None, format="torchscript"):
        """
        Exportiert das Modell für Inferenz-Zwecke.
        
        Args:
            path: Pfad zum Speichern (optional)
            format: Format zum Exportieren ("torchscript", "onnx")
            
        Returns:
            Pfad zum exportierten Modell
        """
        if path is None:
            path = os.path.join(self.output_dir, "inference", format)
        
        logger.info(f"Exportiere Modell für Inferenz nach {path} im Format {format}")
        return self.export_model(path, export_format=format)
