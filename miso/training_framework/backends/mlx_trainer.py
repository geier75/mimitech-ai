#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - MLX Trainer Backend

Implementierung des Trainers mit MLX-Backend für optimale Performance auf Apple Silicon.
"""

import os
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from datetime import datetime

try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    logging.warning("MLX nicht verfügbar - Trainingsfunktionalität eingeschränkt")

# Konfiguriere Logger
logger = logging.getLogger("MISO.MLXTrainer")

class MLXTrainer:
    """
    Trainer-Implementierung mit MLX-Backend für Apple Silicon.
    
    Optimiert für:
    - Apple Neural Engine
    - Metal Performance Shaders
    - Multi-Core CPU
    """
    
    def __init__(self, 
                 model_type: str,
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 tokenizer_config: Dict[str, Any],
                 languages: List[str],
                 output_dir: Path):
        """
        Initialisiert den MLX Trainer.
        
        Args:
            model_type: Typ des zu trainierenden Modells
            model_config: Konfiguration des Modells
            training_config: Trainingskonfiguration
            tokenizer_config: Tokenizer-Konfiguration
            languages: Zu unterstützende Sprachen
            output_dir: Ausgabeverzeichnis
        """
        if not HAS_MLX:
            raise ImportError("MLX ist nicht installiert, aber für diesen Trainer erforderlich")
        
        self.model_type = model_type
        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer_config = tokenizer_config
        self.languages = languages
        self.output_dir = Path(output_dir)
        
        # Erstelle Tokenizer
        self.tokenizer = self._create_tokenizer()
        
        # Erstelle Modell
        self.model = self._create_model()
        
        # Erstelle Optimizer
        self.optimizer = self._create_optimizer()
        
        # Erstelle Loss-Funktion
        self.loss_fn = self._create_loss_function()
        
        # Kompiliere Trainingsfunktionen für bessere Performance
        self._compile_functions()
        
        logger.info(f"MLX Trainer initialisiert mit Modelltyp: {model_type}")
    
    def _create_tokenizer(self):
        """Erstellt den Tokenizer basierend auf der Konfiguration."""
        # MLX hat keinen eigenen Tokenizer, daher verwenden wir SentencePiece
        try:
            import sentencepiece as spm
            
            vocab_dir = self.output_dir / "tokenizer"
            os.makedirs(vocab_dir, exist_ok=True)
            
            # Prüfe, ob Tokenizer bereits existiert
            vocab_file = vocab_dir / "miso_tokenizer.model"
            if os.path.exists(vocab_file):
                logger.info(f"Lade existierenden Tokenizer aus {vocab_file}")
                tokenizer = spm.SentencePieceProcessor()
                tokenizer.load(str(vocab_file))
                return tokenizer
            
            # Wenn keine vorhandene Datei, erstelle eine temporäre Dummy-Datei für Initialisierung
            # In einer echten Implementierung würde hier der Tokenizer aus dem Trainingsdaten trainiert werden
            logger.warning("Kein existierender Tokenizer gefunden - erstelle Dummy-Initialisierung")
            
            # Erstelle einen einfachen Tokenizer mit Basisvokabular
            temp_file = self.output_dir / "tokenizer" / "temp_vocab.txt"
            with open(temp_file, 'w') as f:
                for i in range(self.tokenizer_config["vocab_size"]):
                    f.write(f"TOKEN_{i}\n")
            
            # Trainiere SentencePiece-Modell (in einer realen Implementierung würde dies auf echten Daten geschehen)
            spm.SentencePieceTrainer.train(
                input=str(temp_file),
                model_prefix=str(vocab_dir / "miso_tokenizer"),
                vocab_size=self.tokenizer_config["vocab_size"],
                model_type="bpe",
                character_coverage=0.9995,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3
            )
            
            # Lade das trainierte Modell
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.load(str(vocab_file))
            
            # Lösche temporäre Datei
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return tokenizer
            
        except ImportError:
            logger.warning("SentencePiece nicht installiert - verwende vereinfachten Tokenizer")
            
            # Fallback auf einen sehr einfachen Tokenizer (nur für Demo/Test)
            class SimpleTokenizer:
                def __init__(self, vocab_size):
                    self.vocab_size = vocab_size
                
                def encode(self, text, add_special_tokens=True):
                    # Einfache Zeichen-zu-ID-Umwandlung (nicht für Produktion geeignet)
                    result = [ord(c) % (self.vocab_size - 4) + 4 for c in text]
                    if add_special_tokens:
                        result = [2] + result + [3]  # BOS und EOS Token
                    return result
                
                def decode(self, ids):
                    # Einfache ID-zu-Zeichen-Umwandlung
                    text = "".join(chr((id - 4) % 256) if id >= 4 else "" for id in ids)
                    return text
            
            return SimpleTokenizer(self.tokenizer_config["vocab_size"])
    
    def _create_model(self):
        """Erstellt das Modell basierend auf der Konfiguration."""
        if self.model_type == "transformer":
            return self._create_transformer_model()
        elif self.model_type == "lstm":
            return self._create_lstm_model()
        elif self.model_type == "gpt":
            return self._create_gpt_model()
        elif self.model_type == "toponet":
            return self._create_toponet_model()
        else:
            raise ValueError(f"Nicht unterstützter Modelltyp: {self.model_type}")
    
    def _create_transformer_model(self):
        """Erstellt ein Transformer-Modell mit MLX."""
        from miso.training_framework.models.mlx_models import MLXTransformer
        
        model = MLXTransformer(
            vocab_size=self.tokenizer_config["vocab_size"],
            hidden_size=self.model_config["hidden_size"],
            num_hidden_layers=self.model_config["num_layers"],
            num_attention_heads=self.model_config["num_heads"],
            intermediate_size=self.model_config["intermediate_size"],
            dropout=self.model_config["dropout"],
            max_position_embeddings=self.tokenizer_config["max_length"],
            num_languages=len(self.languages)
        )
        
        # Initialisiere Parameter
        batch = mx.zeros((2, self.tokenizer_config["max_length"]))
        language_ids = mx.zeros((2,), dtype=mx.int32)
        model(batch, language_ids)
        
        return model
    
    def _create_lstm_model(self):
        """Erstellt ein LSTM-Modell mit MLX."""
        from miso.training_framework.models.mlx_models import MLXLSTM
        
        model = MLXLSTM(
            vocab_size=self.tokenizer_config["vocab_size"],
            hidden_size=self.model_config["hidden_size"],
            num_layers=self.model_config["num_layers"],
            dropout=self.model_config["dropout"],
            bidirectional=self.model_config.get("bidirectional", True),
            num_languages=len(self.languages)
        )
        
        # Initialisiere Parameter
        batch = mx.zeros((2, self.tokenizer_config["max_length"]))
        language_ids = mx.zeros((2,), dtype=mx.int32)
        model(batch, language_ids)
        
        return model
    
    def _create_gpt_model(self):
        """Erstellt ein GPT-Modell mit MLX."""
        from miso.training_framework.models.mlx_models import MLXGPT
        
        model = MLXGPT(
            vocab_size=self.tokenizer_config["vocab_size"],
            hidden_size=self.model_config["hidden_size"],
            num_hidden_layers=self.model_config["num_layers"],
            num_attention_heads=self.model_config["num_heads"],
            intermediate_size=self.model_config["intermediate_size"],
            dropout=self.model_config["dropout"],
            max_position_embeddings=self.tokenizer_config["max_length"],
            num_languages=len(self.languages)
        )
        
        # Initialisiere Parameter
        batch = mx.zeros((2, self.tokenizer_config["max_length"]))
        language_ids = mx.zeros((2,), dtype=mx.int32)
        model(batch, language_ids)
        
        return model
    
    def _create_toponet_model(self):
        """Erstellt ein TopoNet-Modell mit MLX."""
        from miso.training_framework.models.mlx_models import MLXTopoNet
        
        model = MLXTopoNet(
            vocab_size=self.tokenizer_config["vocab_size"],
            hidden_size=self.model_config["hidden_size"],
            num_layers=self.model_config["num_layers"],
            manifold_dim=self.model_config["manifold_dim"],
            dropout=self.model_config["dropout"],
            max_position_embeddings=self.tokenizer_config["max_length"],
            num_languages=len(self.languages)
        )
        
        # Initialisiere Parameter
        batch = mx.zeros((2, self.tokenizer_config["max_length"]))
        language_ids = mx.zeros((2,), dtype=mx.int32)
        model(batch, language_ids)
        
        return model
    
    def _create_optimizer(self):
        """Erstellt den Optimizer basierend auf der Trainingskonfiguration."""
        learning_rate = self.training_config["learning_rate"]
        weight_decay = self.training_config["weight_decay"]
        
        # AdamW Optimizer
        optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )
        
        # Initialisiere Optimizer
        optimizer = optimizer.init(self.model.parameters())
        
        return optimizer
    
    def _create_loss_function(self):
        """Erstellt die Loss-Funktion."""
        def cross_entropy_loss(logits, targets, ignore_index=-100):
            """
            Cross-Entropy-Verlustfunktion mit Masking für Padding.
            
            Args:
                logits: Modellausgabe (B, S, V) - Batch, Sequence, Vocabulary
                targets: Ground-Truth-Labels (B, S)
                ignore_index: Index, der in der Loss-Berechnung ignoriert werden soll
                
            Returns:
                Loss-Wert (Skalar)
            """
            # Reshape Logits für Effizienz
            logits = logits.reshape(-1, logits.shape[-1])
            targets = targets.reshape(-1)
            
            # Maske für nicht ignorierte Positionen
            mask = targets != ignore_index
            
            # Reduziere auf relevante Positionen
            valid_logits = logits[mask]
            valid_targets = targets[mask]
            
            # Standard-Cross-Entropy
            one_hot_targets = mx.one_hot(valid_targets, valid_logits.shape[-1])
            losses = -mx.sum(one_hot_targets * mx.log_softmax(valid_logits, axis=-1), axis=-1)
            
            # Reduziere zum Mittelwert
            return mx.mean(losses)
        
        return cross_entropy_loss
    
    def _compile_functions(self):
        """Kompiliert die Trainingsfunktionen für verbesserte Performance."""
        # Forward-Pass mit Loss-Berechnung
        def train_step(model, x, y, language_ids, ignore_index=-100):
            """Führt einen Trainingsschritt durch."""
            def loss_fn(model_params):
                # Aktualisiere Modellparameter
                self.model.update(model_params)
                
                # Forward-Pass
                logits = self.model(x, language_ids)
                
                # Berechne Loss
                loss = self.loss_fn(logits, y, ignore_index)
                
                return loss
            
            # Berechne Loss und Gradienten
            loss, grads = mx.value_and_grad(loss_fn)(self.model.parameters())
            
            # Optimizer-Schritt
            self.optimizer = optim.update(self.optimizer, grads)
            
            # Gebe Loss zurück
            return loss
        
        # Evaluation (nur Forward-Pass)
        def eval_step(model, x, y, language_ids, ignore_index=-100):
            """Führt einen Evaluationsschritt durch."""
            # Forward-Pass
            logits = model(x, language_ids)
            
            # Berechne Loss
            loss = self.loss_fn(logits, y, ignore_index)
            
            # Berechne Accuracy
            predictions = mx.argmax(logits, axis=-1)
            mask = y != ignore_index
            correct = mx.sum(mx.logical_and(predictions == y, mask))
            total = mx.sum(mask)
            
            return loss, correct, total
        
        # Kompiliere Funktionen
        self.train_step = train_step
        self.eval_step = eval_step
    
    def train_epoch(self, train_data):
        """
        Trainiert das Modell für eine Epoche.
        
        Args:
            train_data: Trainingsdaten als Liste von Beispielen
            
        Returns:
            Dictionary mit Trainingsmetriken für diese Epoche
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        # Batch-Größe und Gradient Accumulation
        batch_size = self.training_config["batch_size"]
        grad_accum_steps = self.training_config["gradient_accumulation_steps"]
        actual_batch_size = batch_size // grad_accum_steps
        
        # Mixe die Daten
        np.random.shuffle(train_data)
        
        start_time = time.time()
        num_batches = 0
        
        # Iteriere über Batches
        for i in range(0, len(train_data), actual_batch_size):
            batch_data = train_data[i:i+actual_batch_size]
            
            # Bereite Batch vor
            input_ids, attention_mask, labels, language_ids = self._prepare_batch(batch_data)
            
            # Trainiere auf diesem Batch
            loss = self.train_step(
                self.model, 
                input_ids, 
                labels, 
                language_ids, 
                ignore_index=-100
            )
            
            # Aktualisiere Statistiken
            total_loss += loss.item() * len(batch_data)
            total_samples += len(batch_data)
            num_batches += 1
            
            # Zeige Progress alle 100 Batches
            if num_batches % 100 == 0:
                examples_per_second = total_samples / (time.time() - start_time)
                logger.info(f"Training: Batch {num_batches}, Loss: {loss.item():.4f}, "
                           f"Speed: {examples_per_second:.2f} samples/s")
        
        # Berechne Gesamtmetriken
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        # Führe eine schnelle Evaluation durch, um eine grobe Accuracy zu berechnen
        accuracy = self._compute_accuracy_on_subset(train_data, subset_size=min(1000, len(train_data)))
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "samples": total_samples,
            "batches": num_batches,
            "epoch_time": time.time() - start_time
        }
        
        return metrics
    
    def evaluate(self, eval_data):
        """
        Evaluiert das Modell auf einem Datensatz.
        
        Args:
            eval_data: Evaluationsdaten als Liste von Beispielen
            
        Returns:
            Dictionary mit Evaluationsmetriken
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        batch_size = self.training_config["batch_size"] * 2  # Größere Batches für Evaluation
        
        start_time = time.time()
        
        # Iteriere über Batches
        for i in range(0, len(eval_data), batch_size):
            batch_data = eval_data[i:i+batch_size]
            
            # Bereite Batch vor
            input_ids, attention_mask, labels, language_ids = self._prepare_batch(batch_data)
            
            # Evaluiere auf diesem Batch
            loss, correct, total = self.eval_step(
                self.model, 
                input_ids, 
                labels, 
                language_ids, 
                ignore_index=-100
            )
            
            # Aktualisiere Statistiken
            total_loss += loss.item() * len(batch_data)
            total_correct += correct.item()
            total_tokens += total.item()
        
        # Berechne Gesamtmetriken
        avg_loss = total_loss / len(eval_data) if len(eval_data) > 0 else 0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "samples": len(eval_data),
            "tokens": total_tokens,
            "correct": total_correct,
            "eval_time": time.time() - start_time
        }
        
        return metrics
    
    def _prepare_batch(self, batch_data):
        """
        Bereitet einen Batch für das Training/die Evaluation vor.
        
        Args:
            batch_data: Liste von Beispielen
            
        Returns:
            Tupel aus input_ids, attention_mask, labels, language_ids
        """
        # In einer realen Implementierung würden die Daten bereits tokenisiert sein
        # Hier demonstrieren wir einen vereinfachten Prozess
        
        # Extrahiere Texte und Labels aus den Beispielen
        input_texts = [example["text"] for example in batch_data]
        languages = [example["language"] for example in batch_data]
        
        # Konvertiere Sprachen zu IDs
        language_ids = [self.languages.index(lang) if lang in self.languages else 0 for lang in languages]
        language_ids = mx.array(language_ids, dtype=mx.int32)
        
        # Tokenisiere Texte
        tokenized = []
        max_len = self.tokenizer_config["max_length"]
        
        for text in input_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            # Truncate oder Padding
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [0] * (max_len - len(tokens))  # 0 = PAD token
            tokenized.append(tokens)
        
        # Konvertiere zu MLX Arrays
        input_ids = mx.array(tokenized, dtype=mx.int32)
        
        # Erstelle Attention Mask (1 für Tokens, 0 für Padding)
        attention_mask = mx.array([[1 if token != 0 else 0 for token in seq] for seq in tokenized], dtype=mx.int32)
        
        # Erstelle Labels (für kausales LM-Training: Eingabe verschoben um 1 nach rechts)
        labels = mx.array([seq[1:] + [-100] for seq in tokenized], dtype=mx.int32)  # -100 = ignore_index
        
        return input_ids, attention_mask, labels, language_ids
    
    def _compute_accuracy_on_subset(self, data, subset_size=1000):
        """
        Berechnet die Accuracy auf einem Subset der Daten.
        
        Args:
            data: Gesamtdatensatz
            subset_size: Größe des Subsets
            
        Returns:
            Accuracy als Float
        """
        # Wähle zufälliges Subset
        indices = np.random.choice(len(data), min(subset_size, len(data)), replace=False)
        subset = [data[i] for i in indices]
        
        # Evaluiere auf Subset
        self.model.eval()
        
        total_correct = 0
        total_tokens = 0
        
        # Iteriere über Batches
        batch_size = min(64, len(subset))
        for i in range(0, len(subset), batch_size):
            batch_data = subset[i:i+batch_size]
            
            # Bereite Batch vor
            input_ids, attention_mask, labels, language_ids = self._prepare_batch(batch_data)
            
            # Forward-Pass
            logits = self.model(input_ids, language_ids)
            
            # Berechne Accuracy
            predictions = mx.argmax(logits, axis=-1)
            mask = labels != -100
            correct = mx.sum(mx.logical_and(predictions == labels, mask))
            tokens = mx.sum(mask)
            
            total_correct += correct.item()
            total_tokens += tokens.item()
        
        # Berechne Gesamtaccuracy
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return accuracy
    
    def save_checkpoint(self, path):
        """
        Speichert den aktuellen Zustand des Modells und Trainings.
        
        Args:
            path: Pfad, unter dem der Checkpoint gespeichert werden soll
        """
        # Erstelle Verzeichnis, falls nicht vorhanden
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Speichere Modellparameter
        mlx.save_safetensors(self.model.parameters(), f"{path}.safetensors")
        
        # Speichere Optimizer-Zustand
        optimizer_state = {
            "opt_state": self.optimizer
        }
        mlx.save_safetensors(optimizer_state, f"{path}.opt.safetensors")
        
        # Speichere Training-Metadaten
        metadata = {
            "model_type": self.model_type,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "tokenizer_config": self.tokenizer_config,
            "languages": self.languages,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{path}.meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint gespeichert nach: {path}")
    
    def load_checkpoint(self, path):
        """
        Lädt den Zustand des Modells und Trainings aus einem Checkpoint.
        
        Args:
            path: Pfad zum Checkpoint
        """
        # Prüfe, ob Dateien existieren
        if not os.path.exists(f"{path}.safetensors"):
            raise FileNotFoundError(f"Modell-Checkpoint {path}.safetensors nicht gefunden")
        
        # Lade Modellparameter
        loaded_params = mlx.load_safetensors(f"{path}.safetensors")
        self.model.update(loaded_params)
        
        # Lade Optimizer-Zustand, falls vorhanden
        if os.path.exists(f"{path}.opt.safetensors"):
            optimizer_state = mlx.load_safetensors(f"{path}.opt.safetensors")
            self.optimizer = optimizer_state["opt_state"]
        
        # Lade Metadaten, falls vorhanden
        if os.path.exists(f"{path}.meta.json"):
            with open(f"{path}.meta.json", 'r') as f:
                metadata = json.load(f)
                logger.info(f"Checkpoint geladen: {metadata.get('model_type', 'unbekannt')} "
                           f"vom {metadata.get('timestamp', 'unbekanntem Zeitpunkt')}")
        
        logger.info(f"Checkpoint geladen von: {path}")
    
    def export_model(self, export_format, output_path, optimize=True):
        """
        Exportiert das Modell in ein bestimmtes Format.
        
        Args:
            export_format: Format für den Export ('onnx', 'mlmodel')
            output_path: Pfad zum Speichern des exportierten Modells
            optimize: Ob das Modell für die Inferenz optimiert werden soll
            
        Returns:
            Pfad zum exportierten Modell
        """
        # Stelle sicher, dass Verzeichnis existiert
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Core ML Export
        if export_format == "coreml":
            try:
                import coremltools as ct
                
                logger.info("Exportiere Modell nach Core ML...")
                
                # Erstelle eine Wrapper-Funktion für die Modellinferenz
                def inference_fn(x_tensor, language_id_tensor):
                    """Führt Inferenz für Core ML durch."""
                    # Konvertiere nach MLX
                    x = mx.array(x_tensor)
                    language_id = mx.array(language_id_tensor, dtype=mx.int32)
                    
                    # Führe Modell aus
                    logits = self.model(x, language_id)
                    
                    # Ergebnisse
                    return logits.astype(mx.float32)
                
                # Definiere Eingabeformen
                batch_size = 1  # Für Inferenz
                input_shape = (batch_size, self.tokenizer_config["max_length"])
                
                # Beispiel-Eingaben
                example_inputs = {
                    "x_tensor": np.zeros(input_shape, dtype=np.int32),
                    "language_id_tensor": np.zeros((batch_size,), dtype=np.int32)
                }
                
                # Konvertiere das Modell
                mlx_model = mx.compile(inference_fn)
                
                # Core ML fordert PyTorch oder TensorFlow Modelle, daher müssen wir hier einen Umweg gehen
                # In einer realen Implementierung würde man einen direkten Export-Pfad implementieren
                
                # Wir simulieren den Export mit dem Core ML Converter
                model_description = {
                    "input_features": [
                        {"name": "input_ids", "type": "int32", "shape": input_shape},
                        {"name": "language_id", "type": "int32", "shape": (batch_size,)}
                    ],
                    "output_features": [
                        {"name": "logits", "type": "float32", "shape": (batch_size, self.tokenizer_config["max_length"], self.tokenizer_config["vocab_size"])}
                    ]
                }
                
                # In einer realen Implementierung würde hier der tatsächliche Export erfolgen
                # Für diese Demo simulieren wir dies
                
                logger.info(f"Modell würde nach {output_path} exportiert werden (simuliert)")
                
                # Speichere anstelle eines echten Core ML Modells die Safetensors
                mlx.save_safetensors(self.model.parameters(), f"{output_path}.safetensors")
                
                # Speichere Meta-Informationen
                with open(f"{output_path}.meta.json", 'w') as f:
                    json.dump({
                        "model_type": self.model_type,
                        "export_format": export_format,
                        "optimized": optimize,
                        "input_shape": input_shape,
                        "vocab_size": self.tokenizer_config["vocab_size"],
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)
                
                logger.info(f"Modell-Metadaten gespeichert nach: {output_path}.meta.json")
                
                return output_path
                
            except ImportError:
                logger.error("coremltools nicht installiert - kann nicht nach Core ML exportieren")
                raise ImportError("coremltools ist erforderlich für den Core ML Export")
        
        elif export_format == "onnx":
            logger.error("ONNX-Export wird für MLX derzeit nicht direkt unterstützt")
            raise NotImplementedError("ONNX-Export wird für MLX derzeit nicht unterstützt")
        
        else:
            logger.error(f"Nicht unterstütztes Export-Format: {export_format}")
            raise ValueError(f"Nicht unterstütztes Export-Format: {export_format}")
