#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - PyTorch Trainer Implementierung

Hauptimplementierung des PyTorch-Trainers für das MISO Ultimate AGI System.
Diese Datei enthält die Implementierung der PyTorchTrainer-Klasse, die mit dem
TrainingController und den PyTorch-Modellen interagiert.
"""

import os
import sys
import json
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from pathlib import Path
import time
import datetime

# Prüfe, ob PyTorch verfügbar ist
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from torch.nn.utils import clip_grad_norm_
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch nicht verfügbar - Trainer kann nicht verwendet werden")

# Prüfe, ob tqdm für Fortschrittsbalken verfügbar ist
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logging.warning("tqdm nicht verfügbar - keine Fortschrittsbalken möglich")

# Prüfe, ob tensorboard verfügbar ist
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    logging.warning("TensorBoard nicht verfügbar - kein visuelles Logging möglich")

# Prüfe, ob SentencePiece verfügbar ist
try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False
    logging.warning("SentencePiece nicht verfügbar - Tokenisierung eingeschränkt")

# Importiere die PyTorch-Modelle
from .pytorch_trainer import PyTorchTransformer, PyTorchLSTM, PyTorchGPT, PyTorchTopoNet

# Konfiguriere Logger
logger = logging.getLogger("MISO.PyTorchTrainer")

# Implementiere die PyTorchTrainer-Klasse, wenn PyTorch verfügbar ist
if HAS_TORCH:
    class PyTorchTrainer:
        """
        PyTorch-basierter Trainer für mehrsprachige Modelle.
        
        Unterstützt Training, Evaluation, Checkpointing und Export für verschiedene
        Modellarchitekturen (Transformer, LSTM, GPT, TopoNet) mit CUDA/MPS-Optimierung.
        """
        
        def __init__(self, 
                     config: Dict[str, Any],
                     model_path: str = None,
                     tokenizer_path: str = None,
                     device: str = "auto"):
            """
            Initialisiert den PyTorch-Trainer.
            
            Args:
                config: Konfigurationswörterbuch für das Training
                model_path: Pfad zu einem vorhandenen Modell zum Laden
                tokenizer_path: Pfad zum Tokenizer-Modell
                device: Gerät für Training ("cpu", "cuda", "mps" oder "auto")
            """
            self.config = config
            self.model_path = model_path
            self.tokenizer_path = tokenizer_path
            
            # Extrahiere Konfigurationswerte
            self.model_type = config.get("model_type", "transformer")
            self.vocab_size = config.get("vocab_size", 30000)
            self.hidden_size = config.get("hidden_size", 768)
            self.num_layers = config.get("num_layers", 12)
            self.num_heads = config.get("num_heads", 12)
            self.dropout = config.get("dropout", 0.1)
            self.batch_size = config.get("batch_size", 32)
            self.learning_rate = config.get("learning_rate", 5e-5)
            self.weight_decay = config.get("weight_decay", 1e-2)
            self.max_seq_length = config.get("max_seq_length", 512)
            self.num_languages = config.get("num_languages", 4)
            
            # Initialisiere Trainingsvariablen
            self.model = None
            self.tokenizer = None
            self.optimizer = None
            self.scheduler = None
            self.summary_writer = None
            self.global_step = 0
            self.best_eval_loss = float('inf')
            self.early_stopping_patience = config.get("early_stopping_patience", 3)
            self.early_stopping_counter = 0
            
            # Setze Zufallswerte für Reproduzierbarkeit
            seed = config.get("seed", 42)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            # Gerät für Training bestimmen
            self.device = self._setup_device(device)
            
            # Logger-Konfiguration
            self.log_steps = config.get("log_steps", 100)
            self.eval_steps = config.get("eval_steps", 1000)
            self.save_steps = config.get("save_steps", 5000)
            
            # Initialisiere Tokenizer
            self._setup_tokenizer()
            
            # Initialisiere Modell
            self._setup_model()
            
            # Initialisiere Optimizer und Scheduler
            self._setup_optimizer()
            
            # Initialisiere TensorBoard, falls verfügbar
            if HAS_TENSORBOARD:
                log_dir = config.get("log_dir", "logs")
                self.summary_writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard-Logs werden in {log_dir} gespeichert")
            
            logger.info(f"PyTorchTrainer initialisiert: {self.model_type} Modell, "
                       f"Gerät: {self.device}")
        
        def _setup_device(self, device: str) -> torch.device:
            """
            Konfiguriert das Trainingsgerät.
            
            Args:
                device: Geräteangabe ("cpu", "cuda", "mps" oder "auto")
                
            Returns:
                torch.device: Konfiguriertes Gerät
            """
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                    logger.info(f"CUDA verfügbar: {torch.cuda.get_device_name(0)}")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                    logger.info("Apple Metal Performance Shaders (MPS) wird verwendet")
                else:
                    device = "cpu"
                    logger.warning("Keine GPU verfügbar, Training auf CPU")
            
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA nicht verfügbar, falle zurück auf CPU")
                device = "cpu"
            
            if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS nicht verfügbar, falle zurück auf CPU")
                device = "cpu"
            
            return torch.device(device)
        
        def _setup_tokenizer(self):
            """
            Initialisiert oder lädt den Tokenizer.
            """
            if HAS_SENTENCEPIECE and self.tokenizer_path and os.path.exists(self.tokenizer_path):
                try:
                    self.tokenizer = spm.SentencePieceProcessor()
                    self.tokenizer.load(self.tokenizer_path)
                    self.vocab_size = self.tokenizer.get_piece_size()
                    logger.info(f"Tokenizer geladen von {self.tokenizer_path}, Vokabulargröße: {self.vocab_size}")
                except Exception as e:
                    logger.error(f"Fehler beim Laden des Tokenizers: {e}")
                    self.tokenizer = None
            else:
                logger.warning("Kein Tokenizer geladen - es wird angenommen, dass die Eingaben bereits tokenisiert sind")
                self.tokenizer = None
        
        def _setup_model(self):
            """
            Initialisiert das Modell oder lädt ein bestehendes.
            """
            # Erstelle ein neues Modell basierend auf dem Modelltyp
            if self.model_type.lower() == "transformer":
                self.model = PyTorchTransformer(
                    vocab_size=self.vocab_size,
                    hidden_size=self.hidden_size,
                    num_hidden_layers=self.num_layers,
                    num_attention_heads=self.num_heads,
                    intermediate_size=self.hidden_size * 4,
                    dropout=self.dropout,
                    max_position_embeddings=self.max_seq_length,
                    num_languages=self.num_languages
                )
            elif self.model_type.lower() == "lstm":
                self.model = PyTorchLSTM(
                    vocab_size=self.vocab_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                    bidirectional=True,
                    num_languages=self.num_languages
                )
            elif self.model_type.lower() == "gpt":
                self.model = PyTorchGPT(
                    vocab_size=self.vocab_size,
                    hidden_size=self.hidden_size,
                    num_hidden_layers=self.num_layers,
                    num_attention_heads=self.num_heads,
                    intermediate_size=self.hidden_size * 4,
                    dropout=self.dropout,
                    max_position_embeddings=self.max_seq_length,
                    num_languages=self.num_languages
                )
            elif self.model_type.lower() == "toponet":
                self.model = PyTorchTopoNet(
                    vocab_size=self.vocab_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    manifold_dim=self.config.get("manifold_dim", 32),
                    dropout=self.dropout,
                    max_position_embeddings=self.max_seq_length,
                    num_languages=self.num_languages
                )
            else:
                raise ValueError(f"Unbekannter Modelltyp: {self.model_type}")
            
            # Lade vorhandenes Modell, falls vorhanden
            if self.model_path and os.path.exists(self.model_path):
                try:
                    # Lade auf CPU, um Speicherprobleme zu vermeiden
                    state_dict = torch.load(self.model_path, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                    logger.info(f"Modell geladen von {self.model_path}")
                except Exception as e:
                    logger.error(f"Fehler beim Laden des Modells: {e}")
            
            # Verschiebe Modell auf das richtige Gerät
            self.model.to(self.device)
            
            # Zähle die Parameter
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Modellparameter: {trainable_params:,} trainierbar, {total_params:,} gesamt")
        
        def _setup_optimizer(self):
            """
            Initialisiert Optimizer und Learning Rate Scheduler.
            """
            # Optimizer
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            
            self.optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                eps=1e-8
            )
            
            # Learning Rate Scheduler (linear mit Warmup)
            num_training_steps = self.config.get("num_training_steps", 100000)
            num_warmup_steps = self.config.get("num_warmup_steps", int(num_training_steps * 0.1))
            
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0, float(num_training_steps - current_step) / 
                    float(max(1, num_training_steps - num_warmup_steps))
                )
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            
            logger.info(f"Optimizer initialisiert: AdamW, LR={self.learning_rate}, WD={self.weight_decay}")
            logger.info(f"Scheduler: Linear mit {num_warmup_steps} Warmup-Schritten")
        
        def train(self, train_dataloader, eval_dataloader=None, num_epochs=3):
            """
            Trainiert das Modell auf den gegebenen Daten.
            
            Args:
                train_dataloader: DataLoader für die Trainingsdaten
                eval_dataloader: Optional DataLoader für die Evaluierungsdaten
                num_epochs: Anzahl der Trainingsepochen
            
            Returns:
                Dict mit Trainingsergebnissen
            """
            # Vorbereitung
            total_steps = len(train_dataloader) * num_epochs
            start_time = time.time()
            train_loss_values = []
            best_model_state = None
            
            logger.info(f"Starte Training: {num_epochs} Epochen, {total_steps} Schritte")
            
            # Setze Modell in Trainingsmodus
            self.model.train()
            
            # Training Loop über Epochen
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                epoch_loss = 0
                
                # Progress Bar für jeden Batch, falls tqdm verfügbar
                iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") if HAS_TQDM else train_dataloader
                
                # Training über Batches
                for step, batch in enumerate(iterator):
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
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(outputs.view(-1, self.vocab_size), labels.view(-1))
                    
                    # Backward Pass und Optimierung
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    # Aktualisiere Statistiken
                    epoch_loss += loss.item()
                    train_loss_values.append(loss.item())
                    self.global_step += 1
                    
                    # Fortschritt in tqdm aktualisieren
                    if HAS_TQDM:
                        iterator.set_postfix({"loss": f"{loss.item():.4f}"})
                    
                    # Logging nach bestimmten Schritten
                    if self.global_step % self.log_steps == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        logger.info(
                            f"Schritt: {self.global_step}, Verlust: {loss.item():.4f}, "
                            f"Lernrate: {lr:.8f}, Zeit: {elapsed:.2f}s"
                        )
                        
                        # Logging mit TensorBoard
                        if self.summary_writer:
                            self.summary_writer.add_scalar("training/loss", loss.item(), self.global_step)
                            self.summary_writer.add_scalar("training/lr", lr, self.global_step)
                    
                    # Evaluation nach bestimmten Schritten
                    if eval_dataloader and self.global_step % self.eval_steps == 0:
                        eval_results = self.evaluate(eval_dataloader)
                        logger.info(
                            f"Evaluation bei Schritt {self.global_step}: "
                            f"Verlust: {eval_results['loss']:.4f}, "
                            f"Perplexität: {eval_results['perplexity']:.2f}"
                        )
                        
                        # Early Stopping
                        if eval_results['loss'] < self.best_eval_loss:
                            self.best_eval_loss = eval_results['loss']
                            self.early_stopping_counter = 0
                            
                            # Speichere bestes Modell im Speicher
                            best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                            
                            # Speichere Checkpoint
                            output_dir = os.path.join(self.config.get("output_dir", "checkpoints"), f"step-{self.global_step}")
                            self.save_model(output_dir)
                        else:
                            self.early_stopping_counter += 1
                            if self.early_stopping_counter >= self.early_stopping_patience:
                                logger.info(f"Early Stopping nach {self.global_step} Schritten")
                                if best_model_state:
                                    # Lade bestes Modell zurück
                                    self.model.load_state_dict(best_model_state)
                                return {
                                    "epochs_completed": epoch + 1,
                                    "steps_completed": self.global_step,
                                    "best_eval_loss": self.best_eval_loss,
                                    "early_stopped": True
                                }
                        
                        # Setze Modell zurück in den Trainingsmodus
                        self.model.train()
                
                # Logging am Ende jeder Epoche
                epoch_avg_loss = epoch_loss / len(train_dataloader)
                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoche {epoch+1}/{num_epochs} abgeschlossen: "
                    f"Durchschnittsverlust: {epoch_avg_loss:.4f}, Zeit: {epoch_time:.2f}s"
                )
                
                # TensorBoard Logging für die Epoche
                if self.summary_writer:
                    self.summary_writer.add_scalar("training/epoch_loss", epoch_avg_loss, epoch+1)
            
            # Training abgeschlossen
            total_train_time = time.time() - start_time
            logger.info(
                f"Training abgeschlossen: {num_epochs} Epochen, {self.global_step} Schritte, "
                f"Zeit: {total_train_time:.2f}s"
            )
            
            # Finale Evaluation
            if eval_dataloader:
                eval_results = self.evaluate(eval_dataloader)
                logger.info(
                    f"Finale Evaluation: Verlust: {eval_results['loss']:.4f}, "
                    f"Perplexität: {eval_results['perplexity']:.2f}"
                )
            
            # Lade bestes Modell, falls vorhanden
            if best_model_state:
                self.model.load_state_dict(best_model_state)
            
            # Speichere finales Modell
            if self.config.get("save_final_model", True):
                output_dir = os.path.join(self.config.get("output_dir", "checkpoints"), "final")
                self.save_model(output_dir)
            
            return {
                "epochs_completed": num_epochs,
                "steps_completed": self.global_step,
                "best_eval_loss": self.best_eval_loss,
                "early_stopped": False
            }
        
        def evaluate(self, eval_dataloader):
            """
            Evaluiert das Modell auf den gegebenen Daten.
            
            Args:
                eval_dataloader: DataLoader für die Evaluierungsdaten
            
            Returns:
                Dict mit Evaluierungsergebnissen
            """
            # Setze Modell in Evaluierungsmodus
            self.model.eval()
            
            # Vorbereitung
            total_loss = 0
            total_samples = 0
            start_time = time.time()
            
            logger.info("Starte Evaluation")
            
            # Deaktiviere Gradientenberechnung für Evaluation
            with torch.no_grad():
                # Progress Bar, falls tqdm verfügbar
                iterator = tqdm(eval_dataloader, desc="Evaluation") if HAS_TQDM else eval_dataloader
                
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
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(outputs.view(-1, self.vocab_size), labels.view(-1))
                    
                    # Aktualisiere Statistiken
                    batch_size = input_ids.size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
            
            # Berechne Metriken
            avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            eval_time = time.time() - start_time
            
            # Logging
            logger.info(
                f"Evaluation abgeschlossen: Verlust: {avg_loss:.4f}, "
                f"Perplexität: {perplexity:.2f}, Zeit: {eval_time:.2f}s"
            )
            
            # TensorBoard Logging
            if self.summary_writer:
                self.summary_writer.add_scalar("evaluation/loss", avg_loss, self.global_step)
                self.summary_writer.add_scalar("evaluation/perplexity", perplexity, self.global_step)
            
            return {
                "loss": avg_loss,
                "perplexity": perplexity,
                "samples": total_samples
            }
        
        def predict(self, input_text, language_id=None, max_length=100):
            """
            Führt Vorhersagen auf dem gegebenen Eingangstext durch.
            
            Args:
                input_text: Eingangstext als String oder Liste von Tokens
                language_id: ID der Sprache (0=EN, 1=DE, 2=FR, 3=ES) oder None
                max_length: Maximale Länge des generierten Textes
            
            Returns:
                Dict mit Vorhersageergebnissen
            """
            # Setze Modell in Evaluierungsmodus
            self.model.eval()
            
            # Tokenisiere Eingabe, falls nötig
            if isinstance(input_text, str) and self.tokenizer:
                input_ids = self.tokenizer.encode(input_text)
                input_ids = torch.tensor([input_ids]).to(self.device)
            elif isinstance(input_text, list):
                input_ids = torch.tensor([input_text]).to(self.device)
            else:
                raise ValueError("input_text muss ein String oder eine Liste von Token-IDs sein")
            
            # Sprachvektor vorbereiten
            if language_id is not None:
                language_ids = torch.tensor([language_id]).to(self.device)
            else:
                language_ids = None
            
            # Initialisiere Aufmerksamkeitsmaske
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
            # Deaktiviere Gradientenberechnung für Vorhersage
            with torch.no_grad():
                # Generiere Text (autoregressiv)
                generated_ids = input_ids.clone()
                
                for _ in range(max_length):
                    # Erhalte die aktuelle Sequenzlänge (kann während der Generierung wachsen)
                    curr_length = generated_ids.size(1)
                    
                    # Forward Pass für die aktuelle Sequenz
                    outputs = self.model(
                        generated_ids, 
                        language_ids=language_ids if language_ids is not None else None,
                        attention_mask=attention_mask
                    )
                    
                    # Hole das letzte Token für die Vorhersage des nächsten Tokens
                    next_token_logits = outputs[:, -1, :]
                    
                    # Wähle das Token mit der höchsten Wahrscheinlichkeit
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    
                    # Füge das neue Token zur Sequenz hinzu
                    generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                    
                    # Aktualisiere die Aufmerksamkeitsmaske
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), device=self.device)], 
                        dim=1
                    )
                    
                    # Prüfe auf EOS-Token
                    if next_token_id.item() == self.config.get("eos_token_id", -1):
                        break
            
            # Konvertiere die generierten IDs zurück zu Text, falls ein Tokenizer verfügbar ist
            if self.tokenizer:
                generated_text = self.tokenizer.decode(generated_ids[0].cpu().numpy().tolist())
            else:
                generated_text = None
                
            return {
                "input_ids": input_ids.cpu().numpy().tolist(),
                "generated_ids": generated_ids.cpu().numpy().tolist(),
                "generated_text": generated_text
            }
        
        def save_model(self, output_dir):
            """
            Speichert das Modell und die Konfiguration im angegebenen Verzeichnis.
            
            Args:
                output_dir: Zielverzeichnis
            """
            # Erstelle Ausgabeverzeichnis, falls es nicht existiert
            os.makedirs(output_dir, exist_ok=True)
            
            # Speichere Modell
            model_path = os.path.join(output_dir, "model.pt")
            torch.save(self.model.state_dict(), model_path)
            
            # Speichere Konfiguration
            config_path = os.path.join(output_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                # Aktualisiere Konfiguration mit aktuellen Werten
                config_to_save = self.config.copy()
                config_to_save.update({
                    "model_type": self.model_type,
                    "vocab_size": self.vocab_size,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "num_heads": self.num_heads,
                    "dropout": self.dropout,
                    "global_step": self.global_step,
                    "best_eval_loss": self.best_eval_loss,
                    "saved_at": datetime.datetime.now().isoformat()
                })
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Modell und Konfiguration in {output_dir} gespeichert")
            return output_dir
        
        def export_model(self, output_dir, export_format="torchscript"):
            """
            Exportiert das Modell im angegebenen Format für Produktions-Inferenz.
            
            Args:
                output_dir: Zielverzeichnis
                export_format: Format zum Exportieren ("torchscript", "onnx")
                
            Returns:
                Pfad zum exportierten Modell
            """
            # Erstelle Ausgabeverzeichnis, falls es nicht existiert
            os.makedirs(output_dir, exist_ok=True)
            
            # Setze Modell in Evaluierungsmodus
            self.model.eval()
            
            # Exportiere entsprechend dem Format
            if export_format.lower() == "torchscript":
                # Erstelle Beispiel-Input für das Tracing
                dummy_input = {
                    "input_ids": torch.ones((1, 16), dtype=torch.long, device=self.device),
                    "language_ids": torch.zeros((1,), dtype=torch.long, device=self.device) 
                               if self.num_languages > 1 else None,
                    "attention_mask": torch.ones((1, 16), dtype=torch.bool, device=self.device)
                }
                
                # Verpacke das Modell in eine Wrapper-Klasse für einfaches Tracing
                class ModelWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    def forward(self, input_ids, language_ids=None, attention_mask=None):
                        return self.model(input_ids, language_ids, attention_mask)
                
                # Trace das Modell
                wrapped_model = ModelWrapper(self.model)
                traced_model = torch.jit.trace(
                    wrapped_model,
                    (
                        dummy_input["input_ids"], 
                        dummy_input["language_ids"], 
                        dummy_input["attention_mask"]
                    )
                )
                
                # Speichere das exportierte Modell
                export_path = os.path.join(output_dir, "model.pt")
                torch.jit.save(traced_model, export_path)
                
                logger.info(f"Modell als TorchScript nach {export_path} exportiert")
                
            elif export_format.lower() == "onnx":
                # Prüfe, ob ONNX verfügbar ist
                try:
                    import onnx
                    import onnxruntime
                except ImportError:
                    logger.error("ONNX oder ONNX Runtime nicht installiert. Installation mit 'pip install onnx onnxruntime'")
                    return None
                
                # Erstelle Beispiel-Input für das Tracing
                dummy_input = {
                    "input_ids": torch.ones((1, 16), dtype=torch.long, device=self.device),
                    "language_ids": torch.zeros((1,), dtype=torch.long, device=self.device) 
                               if self.num_languages > 1 else None,
                    "attention_mask": torch.ones((1, 16), dtype=torch.bool, device=self.device)
                }
                
                # Pfad für das exportierte Modell
                export_path = os.path.join(output_dir, "model.onnx")
                
                # Exportiere nach ONNX
                torch.onnx.export(
                    self.model,
                    (
                        dummy_input["input_ids"], 
                        dummy_input["language_ids"], 
                        dummy_input["attention_mask"]
                    ),
                    export_path,
                    opset_version=12,
                    input_names=["input_ids", "language_ids", "attention_mask"],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "sequence_length"},
                        "attention_mask": {0: "batch_size", 1: "sequence_length"},
                        "logits": {0: "batch_size", 1: "sequence_length"}
                    }
                )
                
                logger.info(f"Modell als ONNX nach {export_path} exportiert")
                
                # Validiere das exportierte Modell
                onnx_model = onnx.load(export_path)
                onnx.checker.check_model(onnx_model)
                
            else:
                raise ValueError(f"Unbekanntes Export-Format: {export_format}")
            
            # Speichere Modell-Metadaten
            metadata = {
                "model_type": self.model_type,
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "export_format": export_format,
                "exported_at": datetime.datetime.now().isoformat(),
                "miso_version": self.config.get("miso_version", "1.0.0")
            }
            
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return export_path

else:
    # Dummy-Implementierung, wenn PyTorch nicht verfügbar ist
    class PyTorchTrainer:
        """Dummy-Implementierung des PyTorchTrainers."""
        
        def __init__(self, *args, **kwargs):
            logger.error("PyTorch ist nicht installiert. PyTorchTrainer kann nicht verwendet werden.")
            raise ImportError("PyTorch ist nicht installiert")
