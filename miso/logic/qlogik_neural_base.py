#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK Neural Base

Basisimplementierung neuronaler Architekturen für Q-LOGIK.
Implementiert grundlegende CNN- und RNN-Strukturen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type
import threading
from dataclasses import dataclass, field

# Importiere PyTorch für neuronale Netze
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Importiere GPU-Beschleunigung
from miso.logic.qlogik_gpu_acceleration import (
    to_tensor, to_numpy, matmul, attention, parallel_map, batch_process,
    get_backend_info
)

# Importiere Speicheroptimierung
from miso.logic.qlogik_memory_optimization import (
    get_from_cache, put_in_cache, clear_cache, register_lazy_loader,
    checkpoint, checkpoint_function, get_memory_stats
)

# Logger einrichten
logger = logging.getLogger("MISO.Logic.Q-LOGIK.NeuralBase")

class BaseModel(nn.Module):
    """Basisklasse für alle neuronalen Modelle in Q-LOGIK"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert das Basismodell
        
        Args:
            config: Konfigurationsobjekt für das Modell
        """
        super().__init__()
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Modellname und Version
        self.model_name = self.config.get("model_name", self.__class__.__name__)
        self.model_version = self.config.get("model_version", "1.0.0")
        
        # Initialisiere Metriken
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "inference_time": []
        }
        
        # Checkpoint-Pfad
        self.checkpoint_dir = self.config.get("checkpoint_dir", os.path.expanduser("~/.miso/models"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das Modell
        
        Args:
            x: Eingabetensor
            
        Returns:
            Ausgabetensor
        """
        raise NotImplementedError("Subklassen müssen forward implementieren")
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Speichert das Modell
        
        Args:
            path: Pfad zum Speichern des Modells (optional)
            
        Returns:
            Pfad, unter dem das Modell gespeichert wurde
        """
        if path is None:
            # Generiere Standardpfad
            timestamp = int(time.time())
            path = os.path.join(
                self.checkpoint_dir,
                f"{self.model_name}_v{self.model_version}_{timestamp}.pt"
            )
        
        # Speichere Modell und Metriken
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "config": self.config,
            "metrics": self.metrics,
            "timestamp": time.time()
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Modell gespeichert unter: {path}")
        
        return path
    
    def load(self, path: str) -> bool:
        """
        Lädt das Modell
        
        Args:
            path: Pfad zum geladenen Modell
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            # Lade Checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # Lade Modellzustand
            self.load_state_dict(checkpoint["model_state_dict"])
            
            # Lade Metadaten
            self.model_name = checkpoint.get("model_name", self.model_name)
            self.model_version = checkpoint.get("model_version", self.model_version)
            self.config.update(checkpoint.get("config", {}))
            self.metrics = checkpoint.get("metrics", self.metrics)
            
            logger.info(f"Modell geladen von: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells von {path}: {e}")
            return False
    
    def get_optimizer(self, optimizer_type: str = "adam", **kwargs) -> optim.Optimizer:
        """
        Erstellt einen Optimizer für das Modell
        
        Args:
            optimizer_type: Typ des Optimizers (adam, sgd, etc.)
            **kwargs: Zusätzliche Parameter für den Optimizer
            
        Returns:
            Optimizer-Instanz
        """
        params = self.parameters()
        
        if optimizer_type.lower() == "adam":
            return optim.Adam(params, **kwargs)
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(params, **kwargs)
        elif optimizer_type.lower() == "rmsprop":
            return optim.RMSprop(params, **kwargs)
        elif optimizer_type.lower() == "adagrad":
            return optim.Adagrad(params, **kwargs)
        else:
            logger.warning(f"Unbekannter Optimizer-Typ: {optimizer_type}, verwende Adam")
            return optim.Adam(params, **kwargs)
    
    def get_scheduler(self, scheduler_type: str, optimizer: optim.Optimizer, **kwargs) -> Any:
        """
        Erstellt einen Lernraten-Scheduler
        
        Args:
            scheduler_type: Typ des Schedulers
            optimizer: Optimizer-Instanz
            **kwargs: Zusätzliche Parameter für den Scheduler
            
        Returns:
            Scheduler-Instanz
        """
        if scheduler_type.lower() == "steplr":
            return optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif scheduler_type.lower() == "multisteplr":
            return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
        elif scheduler_type.lower() == "exponentiallr":
            return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
        elif scheduler_type.lower() == "reducelronplateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        elif scheduler_type.lower() == "cosineannealinglr":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
        else:
            logger.warning(f"Unbekannter Scheduler-Typ: {scheduler_type}, verwende StepLR")
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor], optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Führt einen Trainingsschritt durch
        
        Args:
            batch: Tupel aus (Eingabe, Ziel)
            optimizer: Optimizer-Instanz
            
        Returns:
            Dictionary mit Metriken
        """
        self.train()
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward-Pass
        optimizer.zero_grad()
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward-Pass
        loss.backward()
        optimizer.step()
        
        # Berechne Metriken
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == targets).float().mean().item()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy
        }
    
    def validate(self, val_loader: Any) -> Dict[str, float]:
        """
        Validiert das Modell
        
        Args:
            val_loader: DataLoader für Validierungsdaten
            
        Returns:
            Dictionary mit Validierungsmetriken
        """
        self.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward-Pass
                outputs = self(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                # Berechne Metriken
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == targets).float().mean().item()
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        return {
            "val_loss": total_loss / num_batches,
            "val_accuracy": total_accuracy / num_batches
        }
    
    def fit(self, train_loader: Any, val_loader: Any, epochs: int = 10, 
           optimizer: Optional[optim.Optimizer] = None, 
           scheduler: Any = None,
           callbacks: List[Callable] = None) -> Dict[str, List[float]]:
        """
        Trainiert das Modell
        
        Args:
            train_loader: DataLoader für Trainingsdaten
            val_loader: DataLoader für Validierungsdaten
            epochs: Anzahl der Trainingsepochen
            optimizer: Optimizer-Instanz (optional)
            scheduler: Lernraten-Scheduler (optional)
            callbacks: Liste von Callback-Funktionen (optional)
            
        Returns:
            Dictionary mit Trainingsmetriken
        """
        # Initialisiere Optimizer, falls nicht angegeben
        if optimizer is None:
            optimizer = self.get_optimizer()
        
        # Initialisiere Callbacks
        callbacks = callbacks or []
        
        # Trainingsschleife
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            self.train()
            train_loss = 0.0
            train_accuracy = 0.0
            num_train_batches = 0
            
            for batch in train_loader:
                metrics = self.train_step(batch, optimizer)
                train_loss += metrics["loss"]
                train_accuracy += metrics["accuracy"]
                num_train_batches += 1
            
            # Berechne Durchschnitt
            train_loss /= num_train_batches
            train_accuracy /= num_train_batches
            
            # Validierung
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics["val_loss"]
            val_accuracy = val_metrics["val_accuracy"]
            
            # Aktualisiere Metriken
            self.metrics["train_loss"].append(train_loss)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["train_accuracy"].append(train_accuracy)
            self.metrics["val_accuracy"].append(val_accuracy)
            
            # Aktualisiere Scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Berechne Epochenzeit
            epoch_time = time.time() - epoch_start_time
            
            # Logge Fortschritt
            logger.info(f"Epoche {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                       f"Zeit: {epoch_time:.2f}s")
            
            # Führe Callbacks aus
            for callback in callbacks:
                callback(self, epoch, {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "epoch_time": epoch_time
                })
        
        return self.metrics
    
    def predict(self, inputs: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Führt Vorhersagen durch
        
        Args:
            inputs: Eingabedaten
            
        Returns:
            Vorhersagen als NumPy-Array
        """
        self.eval()
        
        # Konvertiere Eingabe zu Tensor
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        else:
            inputs = inputs.to(self.device)
        
        # Führe Vorhersage durch
        with torch.no_grad():
            start_time = time.time()
            outputs = self(inputs)
            inference_time = time.time() - start_time
        
        # Speichere Inferenzzeit
        self.metrics["inference_time"].append(inference_time)
        
        # Konvertiere Ausgabe zu NumPy
        return outputs.cpu().numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das Modell zurück
        
        Returns:
            Dictionary mit Modellinformationen
        """
        # Zähle Parameter
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Berechne durchschnittliche Inferenzzeit
        avg_inference_time = 0.0
        if self.metrics["inference_time"]:
            avg_inference_time = sum(self.metrics["inference_time"]) / len(self.metrics["inference_time"])
        
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
            "avg_inference_time": avg_inference_time,
            "metrics": {
                "train_loss": self.metrics["train_loss"][-1] if self.metrics["train_loss"] else None,
                "val_loss": self.metrics["val_loss"][-1] if self.metrics["val_loss"] else None,
                "train_accuracy": self.metrics["train_accuracy"][-1] if self.metrics["train_accuracy"] else None,
                "val_accuracy": self.metrics["val_accuracy"][-1] if self.metrics["val_accuracy"] else None
            }
        }


class CNNBase(BaseModel):
    """Basisklasse für CNN-Modelle in Q-LOGIK"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert das CNN-Basismodell
        
        Args:
            config: Konfigurationsobjekt für das Modell
        """
        super().__init__(config)
        
        # Konfigurationsparameter
        self.input_channels = self.config.get("input_channels", 3)
        self.num_classes = self.config.get("num_classes", 10)
        
        # Definiere CNN-Architektur
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Definiere Klassifikationsschicht
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das CNN-Modell
        
        Args:
            x: Eingabetensor
            
        Returns:
            Ausgabetensor
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class RNNBase(BaseModel):
    """Basisklasse für RNN-Modelle in Q-LOGIK"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert das RNN-Basismodell
        
        Args:
            config: Konfigurationsobjekt für das Modell
        """
        super().__init__(config)
        
        # Konfigurationsparameter
        self.input_size = self.config.get("input_size", 300)
        self.hidden_size = self.config.get("hidden_size", 256)
        self.num_layers = self.config.get("num_layers", 2)
        self.bidirectional = self.config.get("bidirectional", True)
        self.dropout = self.config.get("dropout", 0.5)
        self.num_classes = self.config.get("num_classes", 10)
        
        # Definiere RNN-Architektur
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Definiere Ausgabeschicht
        self.fc = nn.Linear(
            self.hidden_size * 2 if self.bidirectional else self.hidden_size,
            self.num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das RNN-Modell
        
        Args:
            x: Eingabetensor [batch_size, seq_len, input_size]
            
        Returns:
            Ausgabetensor [batch_size, num_classes]
        """
        # RNN-Layer
        output, (hidden, cell) = self.rnn(x)
        
        # Verwende letzten Hidden-State
        if self.bidirectional:
            # Konkateniere vorwärts und rückwärts
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        
        # Klassifikation
        output = self.fc(hidden)
        
        return output


# Modell-Factory für einfache Modellinstanziierung
def create_model(model_type: str, config: Dict[str, Any] = None) -> BaseModel:
    """
    Erstellt ein neuronales Modell
    
    Args:
        model_type: Typ des Modells (cnn, rnn)
        config: Konfigurationsobjekt für das Modell
        
    Returns:
        Modellinstanz
    """
    config = config or {}
    
    if model_type.lower() == "cnn":
        return CNNBase(config)
    elif model_type.lower() == "rnn":
        return RNNBase(config)
    else:
        logger.warning(f"Unbekannter Modelltyp: {model_type}, verwende CNN")
        return CNNBase(config)


if __name__ == "__main__":
    # Beispiel für die Verwendung der neuronalen Basismodelle
    logging.basicConfig(level=logging.INFO)
    
    # Erstelle CNN-Modell
    cnn_config = {
        "input_channels": 3,
        "num_classes": 10
    }
    cnn_model = create_model("cnn", cnn_config)
    print(f"CNN-Modell erstellt: {cnn_model.get_model_info()}")
    
    # Erstelle RNN-Modell
    rnn_config = {
        "input_size": 300,
        "hidden_size": 256,
        "num_classes": 5
    }
    rnn_model = create_model("rnn", rnn_config)
    print(f"RNN-Modell erstellt: {rnn_model.get_model_info()}")
    
    # Demonstriere Forward-Pass
    batch_size = 16
    
    # CNN Forward-Pass
    cnn_input = torch.randn(batch_size, 3, 32, 32)
    cnn_output = cnn_model(cnn_input)
    print(f"CNN-Ausgabeform: {cnn_output.shape}")
    
    # RNN Forward-Pass
    seq_len = 20
    rnn_input = torch.randn(batch_size, seq_len, 300)
    rnn_output = rnn_model(rnn_input)
    print(f"RNN-Ausgabeform: {rnn_output.shape}")
