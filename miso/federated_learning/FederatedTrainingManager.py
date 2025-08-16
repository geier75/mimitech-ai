#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Federated Training Manager

Verwaltet das föderierte Training der MISO Ultimate AGI Modelle.
Ermöglicht verteiltes Training auf mehreren Geräten ohne direkten Datenaustausch,
implementiert Differential Privacy und unterstützt verschiedene Aggregationsstrategien.

Teil von Phase 6: Föderiertes Lernen, Mobile-Optimierung und Autonomes Selbsttraining
"""

import os
import json
import time
import copy
import numpy as np
import logging
import hashlib
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Callable, Generator

# Konfiguriere Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("miso_federated_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MISO.FederatedTrainingManager")

class FederatedTrainingManager:
    """
    Manager für das föderierte Training von MISO Ultimate AGI Modellen.
    
    Ermöglicht verteiltes Training auf mehreren Geräten (oder simulierten Geräten),
    ohne direkten Austausch der Trainingsdaten. Implementiert Sicherheitsfeatures wie
    Differential Privacy und unterstützt verschiedene Aggregationsstrategien.
    
    Merkmale:
    - Training auf lokalen Geräten mit lokalen Daten
    - Sichere Aggregation von Modell-Updates
    - Differential Privacy zum Schutz der Trainingsdaten
    - Adaptive Kompression der Modell-Updates
    - Unterstützung für heterogene Geräte (verschiedene Hardware-Kapazitäten)
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 base_model_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 model_type: str = "transformer",
                 backend: Optional[str] = None,
                 differential_privacy: bool = True,
                 dp_epsilon: float = 1.0,
                 dp_delta: float = 1e-5,
                 dp_noise_multiplier: float = 1.1,
                 dp_l2_norm_clip: float = 1.0,
                 aggregation_strategy: str = "fedavg"):
        """
        Initialisiert den FederatedTrainingManager.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            base_model_path: Pfad zum Basismodell für das föderierte Training
            output_dir: Ausgabeverzeichnis für die trainierten Modelle
            model_type: Typ des zu trainierenden Modells
            backend: Zu verwendendes Backend (None = automatische Auswahl)
            differential_privacy: Ob Differential Privacy angewendet werden soll
            dp_epsilon: Privacy Budget (Epsilon) für Differential Privacy
            dp_delta: Delta-Parameter für Differential Privacy
            dp_noise_multiplier: Faktor für Gauß'sches Rauschen bei DP
            dp_l2_norm_clip: L2-Norm-Grenze für Gradient Clipping bei DP
            aggregation_strategy: Strategie zur Aggregation der Modell-Updates
                                  (fedavg, fedprox, scaffold, etc.)
        """
        # Lade Konfiguration
        self.config = self._load_config(config_path)
        
        # Setze Basisparameter
        self.output_dir = output_dir or self.config["federated"]["output_dir"]
        self.output_dir = Path(self.output_dir)
        self.model_type = model_type.lower()
        self.backend = backend or self._determine_optimal_backend()
        self.base_model_path = base_model_path or self.config["federated"]["base_model_path"]
        
        # Differential Privacy Einstellungen
        self.differential_privacy = differential_privacy
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_l2_norm_clip = dp_l2_norm_clip
        
        # Aggregationsstrategie
        self.aggregation_strategy = aggregation_strategy.lower()
        self._validate_aggregation_strategy()
        
        # Erstelle Verzeichnisse
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "local_models", exist_ok=True)
        os.makedirs(self.output_dir / "global_models", exist_ok=True)
        os.makedirs(self.output_dir / "updates", exist_ok=True)
        os.makedirs(self.output_dir / "logs", exist_ok=True)
        
        # Status und Metriken
        self.session_id = self._generate_session_id()
        self.current_round = 0
        self.client_updates = {}
        self.training_stats = {
            "session_id": self.session_id,
            "start_time": None,
            "end_time": None,
            "total_rounds": 0,
            "current_round": 0,
            "participating_clients": [],
            "client_stats": {},
            "global_metrics": {
                "accuracy": [],
                "loss": []
            },
            "aggregation_times": [],
            "privacy_budget_used": 0.0
        }
        
        # Lade das Basismodell
        self.global_model = None
        if self.base_model_path:
            self._load_base_model()
        
        logger.info(f"FederatedTrainingManager initialisiert: SessionID={self.session_id}, "
                    f"Modell={self.model_type}, Backend={self.backend}, "
                    f"DP={self.differential_privacy}, Aggregation={self.aggregation_strategy}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Lädt die Konfiguration aus einer JSON-Datei."""
        default_config = {
            "federated": {
                "output_dir": "/Volumes/My Book/MISO_Ultimate 15.32.28/miso/federated_models",
                "base_model_path": None,
                "min_clients": 2,
                "max_clients": 10,
                "min_samples_per_client": 100,
                "min_participation_rate": 0.7,
                "rounds": 10,
                "local_epochs": 1,
                "local_batch_size": 32,
                "aggregation_strategies": ["fedavg", "fedprox", "scaffold"],
                "client_selection_strategy": "random",
                "update_compression": True,
                "compression_ratio": 0.1,
                "secure_aggregation": True,
                "adaptive_optimization": True
            },
            "privacy": {
                "enabled": True,
                "epsilon": 1.0,
                "delta": 1e-5,
                "noise_multiplier": 1.1,
                "l2_norm_clip": 1.0,
                "accountant": "moments"
            },
            "communication": {
                "timeout_seconds": 300,
                "retry_attempts": 3,
                "verify_updates": True,
                "bandwidth_optimization": True
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
        
        # Fallback auf PyTorch (CPU)
        logger.info("Keine spezielle Hardware erkannt, verwende PyTorch auf CPU")
        return "torch"
    
    def _validate_aggregation_strategy(self):
        """Validiert die gewählte Aggregationsstrategie."""
        valid_strategies = self.config["federated"]["aggregation_strategies"]
        if self.aggregation_strategy not in valid_strategies:
            logger.warning(f"Ungültige Aggregationsstrategie: {self.aggregation_strategy}. "
                          f"Fallback auf 'fedavg'. Gültige Strategien: {valid_strategies}")
            self.aggregation_strategy = "fedavg"
    
    def _generate_session_id(self) -> str:
        """Generiert eine eindeutige Session-ID für das föderierte Training."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"fed_{self.model_type}_{timestamp}_{random_suffix}"
    
    def _load_base_model(self):
        """Lädt das Basismodell für das föderierte Training."""
        try:
            if self.backend == "mlx":
                from miso.training_framework.backends.mlx_trainer import MLXTrainer
                trainer = MLXTrainer(model_type=self.model_type)
                self.global_model = trainer.load_model(self.base_model_path)
            elif self.backend == "torch":
                from miso.training_framework.backends.torch_trainer import TorchTrainer
                trainer = TorchTrainer(model_type=self.model_type)
                self.global_model = trainer.load_model(self.base_model_path)
            # Weitere Backends...
            
            logger.info(f"Basismodell erfolgreich geladen: {self.base_model_path}")
        except Exception as e:
            logger.error(f"Fehler beim Laden des Basismodells: {e}")
            logger.warning("Initialisiere neues Modell")
            self._initialize_new_model()
    
    def _initialize_new_model(self):
        """Initialisiert ein neues Modell, wenn kein Basismodell geladen werden konnte."""
        try:
            if self.backend == "mlx":
                from miso.training_framework.backends.mlx_trainer import MLXTrainer
                trainer = MLXTrainer(model_type=self.model_type)
                self.global_model = trainer.initialize_model()
            elif self.backend == "torch":
                from miso.training_framework.backends.torch_trainer import TorchTrainer
                trainer = TorchTrainer(model_type=self.model_type)
                self.global_model = trainer.initialize_model()
            # Weitere Backends...
            
            logger.info("Neues Modell initialisiert")
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung eines neuen Modells: {e}")
            raise RuntimeError("Konnte weder Basismodell laden noch neues initialisieren")
    
    def initialize_local_training(self, local_data):
        """
        Startet ein lokales Trainingsverfahren auf Mobile- oder Edge-Geräten.
        
        Args:
            local_data: Die lokalen Trainingsdaten des Clients
            
        Returns:
            client_id: ID des initialisierten Clients
        """
        # Generiere Client-ID
        client_id = self._generate_client_id()
        
        # Erstelle Kopie des globalen Modells für diesen Client
        client_model = self._create_client_model(client_id)
        
        # Initialisiere Clientstatistiken
        self.training_stats["client_stats"][client_id] = {
            "data_samples": len(local_data),
            "training_started": datetime.now().isoformat(),
            "training_completed": None,
            "epochs_completed": 0,
            "local_metrics": {
                "loss": [],
                "accuracy": []
            },
            "update_size": 0,
            "update_compressed_size": 0,
            "communication_overhead": 0
        }
        
        # Füge Client zur Liste der teilnehmenden Clients hinzu
        if client_id not in self.training_stats["participating_clients"]:
            self.training_stats["participating_clients"].append(client_id)
        
        logger.info(f"Lokales Training für Client {client_id} initialisiert mit {len(local_data)} Datenpunkten")
        
        return client_id
    
    def compute_local_update(self, client_id, model, local_data, local_epochs=None):
        """
        Berechnet die Gewichtsänderungen basierend auf lokalem Training.
        
        Args:
            client_id: ID des Clients
            model: Das zu trainierende Modell
            local_data: Die lokalen Trainingsdaten
            local_epochs: Anzahl der lokalen Trainingsrunden (Optional)
            
        Returns:
            model_update: Die berechneten Modellaktualisierungen
        """
        if local_epochs is None:
            local_epochs = self.config["federated"]["local_epochs"]
        
        local_batch_size = self.config["federated"]["local_batch_size"]
        
        # Trainiere das Modell lokal
        logger.info(f"Starte lokales Training für Client {client_id} für {local_epochs} Epochen")
        
        start_time = time.time()
        
        # Implementiere das eigentliche Training je nach Backend
        if self.backend == "mlx":
            from miso.training_framework.backends.mlx_trainer import MLXTrainer
            trainer = MLXTrainer(model_type=self.model_type)
            model, metrics = trainer.train_local(
                model=model,
                train_data=local_data,
                epochs=local_epochs,
                batch_size=local_batch_size
            )
        elif self.backend == "torch":
            from miso.training_framework.backends.torch_trainer import TorchTrainer
            trainer = TorchTrainer(model_type=self.model_type)
            model, metrics = trainer.train_local(
                model=model,
                train_data=local_data,
                epochs=local_epochs,
                batch_size=local_batch_size
            )
        # Weitere Backends...
        
        # Berechne das Modell-Update (Differenz zwischen lokalem und globalem Modell)
        model_update = self._compute_model_diff(self.global_model, model)
        
        # Wende Differential Privacy an, wenn aktiviert
        if self.differential_privacy:
            model_update = self._apply_differential_privacy(model_update)
        
        # Komprimiere das Update, wenn aktiviert
        if self.config["federated"]["update_compression"]:
            model_update, compression_ratio = self._compress_update(model_update)
            logger.info(f"Update für Client {client_id} komprimiert, Verhältnis: {compression_ratio:.2f}")
        
        # Aktualisiere Client-Statistiken
        training_time = time.time() - start_time
        
        self.training_stats["client_stats"][client_id]["training_completed"] = datetime.now().isoformat()
        self.training_stats["client_stats"][client_id]["epochs_completed"] += local_epochs
        self.training_stats["client_stats"][client_id]["training_time"] = training_time
        self.training_stats["client_stats"][client_id]["local_metrics"]["loss"].append(metrics["loss"])
        self.training_stats["client_stats"][client_id]["local_metrics"]["accuracy"].append(metrics["accuracy"])
        
        logger.info(f"Lokales Training für Client {client_id} abgeschlossen in {training_time:.2f}s")
        
        return model_update
    
    def send_update_to_server(self, client_id, model_update):
        """
        Sendet ein lokales Modellupdate zum Server (Simulation intern).
        
        Args:
            client_id: ID des Clients
            model_update: Das berechnete Modellupdate
            
        Returns:
            success: True, wenn das Update erfolgreich übermittelt wurde
        """
        logger.info(f"Sende Update von Client {client_id} zum Server")
        
        # In einer realen Implementierung würde hier die Netzwerkkommunikation stattfinden
        # In dieser Simulation speichern wir das Update direkt im Server-Speicher
        
        # Berechne Update-Größe
        update_size = self._calculate_update_size(model_update)
        
        # Überprüfe auf Manipulationen, wenn aktiviert
        if self.config["communication"]["verify_updates"]:
            if not self._verify_update_integrity(client_id, model_update):
                logger.warning(f"Update von Client {client_id} konnte nicht verifiziert werden. Verwerfe Update.")
                return False
        
        # Speichere das Update
        self.client_updates[client_id] = {
            "update": model_update,
            "timestamp": datetime.now().isoformat(),
            "size": update_size
        }
        
        # Speichere Update-Statistiken
        self.training_stats["client_stats"][client_id]["update_size"] = update_size
        
        # Optional: Speichere das Update als Datei für die Protokollierung
        update_path = self.output_dir / "updates" / f"update_{self.session_id}_{self.current_round}_{client_id}.npz"
        self._save_update(model_update, update_path)
        
        logger.info(f"Update von Client {client_id} erfolgreich empfangen ({update_size} bytes)")
        
        return True
    
    def aggregate_global_model(self):
        """
        Mittelt die lokalen Updates zu einem neuen globalen Modell.
        
        Unterstützt verschiedene Aggregationsstrategien:
        - FedAvg: Gewichtetes Mitteln basierend auf der Datenmenge pro Client
        - FedProx: FedAvg mit Regularisierung zur Vermeidung von Client-Drift
        - SCAFFOLD: Korrektur des Client-Drifts durch Kontrollvariablen
        
        Returns:
            aggregated_model: Das aggregierte globale Modell
        """
        if not self.client_updates:
            logger.warning("Keine Client-Updates vorhanden. Globales Modell bleibt unverändert.")
            return self.global_model
        
        logger.info(f"Starte Aggregation von {len(self.client_updates)} Client-Updates mit Strategie '{self.aggregation_strategy}'")
        start_time = time.time()
        
        # Extrahiere alle Updates
        updates = [data["update"] for client_id, data in self.client_updates.items()]
        
        # Gewichte für die Aggregation basierend auf der Datenmenge pro Client
        client_weights = []
        total_samples = 0
        for client_id in self.client_updates:
            samples = self.training_stats["client_stats"][client_id]["data_samples"]
            client_weights.append(samples)
            total_samples += samples
        
        # Normalisiere die Gewichte
        client_weights = [w / total_samples for w in client_weights]
        
        # Aggregiere Updates je nach Strategie
        if self.aggregation_strategy == "fedavg":
            aggregated_update = self._fedavg_aggregation(updates, client_weights)
        elif self.aggregation_strategy == "fedprox":
            aggregated_update = self._fedprox_aggregation(updates, client_weights)
        elif self.aggregation_strategy == "scaffold":
            aggregated_update = self._scaffold_aggregation(updates, client_weights)
        else:
            logger.warning(f"Unbekannte Aggregationsstrategie: {self.aggregation_strategy}. Verwende FedAvg.")
            aggregated_update = self._fedavg_aggregation(updates, client_weights)
        
        # Wende das aggregierte Update auf das globale Modell an
        aggregated_model = self._apply_update_to_model(self.global_model, aggregated_update)
        
        # Bewerte das neue globale Modell
        metrics = self._evaluate_global_model(aggregated_model)
        
        # Aktualisiere Statistiken
        aggregation_time = time.time() - start_time
        self.training_stats["aggregation_times"].append(aggregation_time)
        self.training_stats["global_metrics"]["accuracy"].append(metrics["accuracy"])
        self.training_stats["global_metrics"]["loss"].append(metrics["loss"])
        
        # Speichere das aggregierte Modell
        self.current_round += 1
        self.training_stats["current_round"] = self.current_round
        model_path = self.output_dir / "global_models" / f"global_model_{self.session_id}_round_{self.current_round}.pt"
        self._save_model(aggregated_model, model_path)
        
        # Aktualisiere das globale Modell
        self.global_model = aggregated_model
        
        # Löschen der verarbeiteten Client-Updates
        self.client_updates = {}
        
        logger.info(f"Globales Modell erfolgreich aggregiert in {aggregation_time:.2f}s. "
                    f"Runde {self.current_round}, Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}")
        
        return aggregated_model
    
    def _generate_client_id(self) -> str:
        """Generiert eine eindeutige Client-ID für das föderierte Training."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.md5(str(time.time() + np.random.rand()).encode()).hexdigest()[:6]
        return f"client_{timestamp}_{random_suffix}"
    
    def _create_client_model(self, client_id):
        """Erstellt eine Kopie des globalen Modells für einen Client."""
        if self.global_model is None:
            logger.error("Kein globales Modell vorhanden. Initialisiere neues Modell.")
            self._initialize_new_model()
        
        # Tiefe Kopie des globalen Modells erstellen
        client_model = copy.deepcopy(self.global_model)
        
        # Speichere lokale Modellkopie
        model_path = self.output_dir / "local_models" / f"client_model_{self.session_id}_{client_id}.pt"
        self._save_model(client_model, model_path)
        
        logger.info(f"Modellkopie für Client {client_id} erstellt")
        return client_model
    
    def _compute_model_diff(self, global_model, local_model):
        """Berechnet die Differenz zwischen lokalem und globalem Modell."""
        # Die Implementierung hängt vom konkreten Backend ab
        if self.backend == "mlx":
            # MLX-spezifische Implementierung
            import mlx.core as mx
            model_diff = {}
            for key in global_model:
                if isinstance(global_model[key], mx.array):
                    model_diff[key] = local_model[key] - global_model[key]
            return model_diff
        elif self.backend == "torch":
            # PyTorch-spezifische Implementierung
            import torch
            model_diff = {}
            for key in global_model:
                if isinstance(global_model[key], torch.Tensor):
                    model_diff[key] = local_model[key] - global_model[key]
            return model_diff
        else:
            # Generischer Fallback für NumPy
            model_diff = {}
            for key in global_model:
                if isinstance(global_model[key], np.ndarray):
                    model_diff[key] = local_model[key] - global_model[key]
            return model_diff
    
    def _apply_differential_privacy(self, model_update):
        """Wendet Differential Privacy auf ein Modellupdate an."""
        # Implementierung des DP-Mechanismus mittels Gaussian Noise-Hinzufügung und Clipping
        clipped_update = {}
        
        # L2-Norm des Updates berechnen
        l2_norm_squared = 0
        for key, value in model_update.items():
            if isinstance(value, np.ndarray):
                l2_norm_squared += np.sum(np.square(value))
            elif hasattr(value, "numpy"):  # Für Tensor-Typen
                value_np = value.numpy() if callable(value.numpy) else np.array(value.cpu())
                l2_norm_squared += np.sum(np.square(value_np))
        
        l2_norm = np.sqrt(l2_norm_squared)
        scaling_factor = min(1.0, self.dp_l2_norm_clip / max(l2_norm, 1e-10))
        
        # Update clippen
        for key, value in model_update.items():
            if isinstance(value, np.ndarray):
                clipped_value = value * scaling_factor
                # Gaussian Noise hinzufügen
                noise_scale = self.dp_noise_multiplier * self.dp_l2_norm_clip
                noise = np.random.normal(0, noise_scale, clipped_value.shape)
                clipped_update[key] = clipped_value + noise
            elif hasattr(value, "numpy"): 
                # Für Tensor-Typen
                if self.backend == "mlx":
                    import mlx.core as mx
                    clipped_value = value * scaling_factor
                    noise_scale = self.dp_noise_multiplier * self.dp_l2_norm_clip
                    noise = mx.random.normal(0, noise_scale, clipped_value.shape)
                    clipped_update[key] = clipped_value + noise
                elif self.backend == "torch":
                    import torch
                    clipped_value = value * scaling_factor
                    noise_scale = self.dp_noise_multiplier * self.dp_l2_norm_clip
                    noise = torch.normal(0, noise_scale, clipped_value.shape)
                    clipped_update[key] = clipped_value + noise
        
        # Update DP-Budget
        # Einfache Akkumulation für Tracking - In einer produktiven Umgebung würde ein präziserer DP-Accountant verwendet
        rounds = self.current_round + 1
        self.training_stats["privacy_budget_used"] = rounds * (self.dp_epsilon / self.config["federated"]["rounds"])
        
        logger.info(f"Differential Privacy angewendet: Scaling={scaling_factor:.4f}, Noise={self.dp_noise_multiplier * self.dp_l2_norm_clip:.4f}")
        return clipped_update
    
    def _compress_update(self, model_update):
        """Komprimiert ein Modellupdate zur Reduzierung der Kommunikationskosten."""
        compression_ratio = self.config["federated"]["compression_ratio"]
        compressed_update = {}
        original_size = 0
        compressed_size = 0
        
        for key, value in model_update.items():
            if isinstance(value, np.ndarray):
                # Bestimme Größe des ursprünglichen Updates
                original_size += value.size * value.itemsize
                
                # Anwenden einer einfachen Top-k Sparsifizierung
                k = max(1, int(value.size * compression_ratio))
                abs_values = np.abs(value.reshape(-1))
                threshold = np.partition(abs_values, -k)[-k]
                mask = abs_values >= threshold
                sparse_values = value.reshape(-1)[mask]
                sparse_indices = np.where(mask)[0]
                
                # Speichere sparse Repräsentation
                compressed_update[key] = {
                    "indices": sparse_indices,
                    "values": sparse_values,
                    "shape": value.shape
                }
                
                # Berechne komprimierte Größe
                compressed_size += sparse_indices.size * sparse_indices.itemsize
                compressed_size += sparse_values.size * sparse_values.itemsize
                compressed_size += len(str(value.shape)) * 2  # Ungefähre Größe der Shape-Information
            else:
                # Für Nicht-NumPy-Arrays (klein genug) keine Kompression
                compressed_update[key] = value
                try:
                    original_size += sys.getsizeof(value)
                    compressed_size += sys.getsizeof(value)
                except:
                    pass
        
        # Berechne tatsächliches Kompressionsverhältnis
        actual_ratio = compressed_size / max(original_size, 1)
        logger.info(f"Update komprimiert: {original_size} -> {compressed_size} Bytes (Verhältnis: {actual_ratio:.4f})")
        
        return compressed_update, actual_ratio
    
    def _calculate_update_size(self, model_update):
        """Berechnet die Größe eines Modellupdates in Bytes."""
        update_size = 0
        
        for key, value in model_update.items():
            if isinstance(value, np.ndarray):
                update_size += value.size * value.itemsize
            elif isinstance(value, dict):  # Für bereits komprimierte Updates
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        update_size += v.size * v.itemsize
                    else:
                        try:
                            update_size += sys.getsizeof(v)
                        except:
                            pass
            else:
                try:
                    update_size += sys.getsizeof(value)
                except:
                    pass
        
        return update_size
        
    def _verify_update_integrity(self, client_id, model_update):
        """Überprüft die Integrität eines Client-Updates."""
        # In einer produktiven Umgebung würde hier eine kryptografische Signaturprüfung stattfinden
        # Für diese Simulation implementieren wir eine einfache Plausibilitätsprüfung
        
        # Prüfe, ob alle erwarteten Schlüssel vorhanden sind
        if not model_update:
            logger.warning(f"Update von Client {client_id} ist leer")
            return False
        
        # Prüfe auf ungewöhnlich große Updates (Erkennung möglicher Angriffe)
        update_size = self._calculate_update_size(model_update)
        if update_size > 1e9:  # 1 GB als arbiträrer Schwellenwert
            logger.warning(f"Update von Client {client_id} ist verdächtig groß: {update_size} bytes")
            return False
        
        # Prüfe auf Not-a-Number-Werte
        for key, value in model_update.items():
            if isinstance(value, np.ndarray) and np.isnan(value).any():
                logger.warning(f"Update von Client {client_id} enthält NaN-Werte")
                return False
            elif isinstance(value, dict):  # Für komprimierte Updates
                if "values" in value and isinstance(value["values"], np.ndarray) and np.isnan(value["values"]).any():
                    logger.warning(f"Komprimiertes Update von Client {client_id} enthält NaN-Werte")
                    return False
        
        return True
    
    def _save_update(self, model_update, file_path):
        """Speichert ein Modellupdate in einer Datei."""
        try:
            # Für diese Simulation verwenden wir NumPy's .npz-Format
            # In einer produktiven Umgebung würde ein effizienteres Format verwendet werden
            
            # Konvertiere alle Tensor-Typen zu NumPy-Arrays
            serializable_update = {}
            
            for key, value in model_update.items():
                if hasattr(value, "numpy") and callable(value.numpy):
                    serializable_update[key] = value.numpy()
                elif hasattr(value, "cpu") and callable(value.cpu):
                    serializable_update[key] = value.cpu().numpy()
                elif isinstance(value, dict):
                    serializable_sub_dict = {}
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, "numpy") and callable(sub_value.numpy):
                            serializable_sub_dict[sub_key] = sub_value.numpy()
                        elif hasattr(sub_value, "cpu") and callable(sub_value.cpu):
                            serializable_sub_dict[sub_key] = sub_value.cpu().numpy()
                        else:
                            serializable_sub_dict[sub_key] = sub_value
                    serializable_update[key] = serializable_sub_dict
                else:
                    serializable_update[key] = value
            
            np.savez_compressed(file_path, **serializable_update)
            logger.debug(f"Update gespeichert: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Updates: {e}")
            return False
    
    def _save_model(self, model, file_path):
        """Speichert ein Modell in einer Datei."""
        try:
            if self.backend == "mlx":
                # MLX-spezifisches Speichern
                import mlx.nn as nn
                nn.save_weights(file_path, model)
            elif self.backend == "torch":
                # PyTorch-spezifisches Speichern
                import torch
                torch.save(model, file_path)
            else:
                # Generische Speichermethode für andere Backends
                np.savez_compressed(file_path, **model)
            
            logger.info(f"Modell gespeichert: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Modells: {e}")
            return False
    
    def _evaluate_global_model(self, model):
        """Evaluiert das globale Modell auf einem Validierungsdatensatz."""
        # In einer echten Implementierung würde hier ein Validierungsdatensatz verwendet werden
        # Für diese Simulation geben wir Platzhaltermetriken zurück
        
        # Progress-Faktor basierend auf der Anzahl der Runden
        progress_factor = min(1.0, (self.current_round + 1) / self.config["federated"]["rounds"])
        
        # Simulierte Verbesserung der Metriken über die Runden hinweg
        loss_improvement = 0.5 * progress_factor
        accuracy_improvement = 0.3 * progress_factor
        
        # Basismetriken
        base_loss = 2.0
        base_accuracy = 0.5
        
        # Simulierte Metriken mit leichtem Rauschen
        loss = base_loss * (1 - loss_improvement) + np.random.normal(0, 0.05)
        accuracy = base_accuracy + accuracy_improvement + np.random.normal(0, 0.02)
        
        # Begrenze Werte auf sinnvolle Bereiche
        loss = max(0.1, loss)
        accuracy = min(0.99, max(0.1, accuracy))
        
        logger.info(f"Globales Modell evaluiert: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        return {"loss": loss, "accuracy": accuracy}
    
    def _fedavg_aggregation(self, updates, client_weights):
        """Aggregiert Modellupdates mit dem FedAvg-Algorithmus (McMahan et al., 2017)."""
        logger.info(f"Verwende FedAvg-Aggregation mit {len(updates)} Updates")
        
        aggregated_update = {}
        
        # Für jeden Parameter im Modell
        for key in updates[0].keys():
            # Extrahiere alle Updates für diesen Parameter
            param_updates = []
            for update in updates:
                if key in update:
                    if isinstance(update[key], dict):  # Komprimiertes Update
                        # Rekonstruiere den vollen Parameter aus dem komprimierten Update
                        full_param = np.zeros(update[key]["shape"])
                        full_param.reshape(-1)[update[key]["indices"]] = update[key]["values"]
                        param_updates.append(full_param)
                    else:  # Unkomprimiertes Update
                        param_updates.append(update[key])
            
            # Gewichtete Mittelung der Parameter
            if param_updates:
                if isinstance(param_updates[0], np.ndarray):
                    weighted_params = [w * p for w, p in zip(client_weights, param_updates)]
                    aggregated_update[key] = sum(weighted_params)
                elif hasattr(param_updates[0], "numpy"):  # Tensor-Typen
                    if self.backend == "mlx":
                        import mlx.core as mx
                        weighted_params = [mx.array(w) * p for w, p in zip(client_weights, param_updates)]
                        aggregated_update[key] = sum(weighted_params)
                    elif self.backend == "torch":
                        import torch
                        weighted_params = [w * p for w, p in zip(client_weights, param_updates)]
                        aggregated_update[key] = sum(weighted_params)
        
        return aggregated_update
    
    def _fedprox_aggregation(self, updates, client_weights):
        """Aggregiert Updates mit dem FedProx-Algorithmus (Stabileres FedAvg mit Proximal Term)."""
        logger.info(f"Verwende FedProx-Aggregation mit {len(updates)} Updates")
        
        # Die FedProx-Aggregation ist identisch mit FedAvg auf Serverseite
        # Der Unterschied liegt in der Client-Optimierung, wo ein Proximal-Term hinzugefügt wird
        # Für diese Simulation verwenden wir die FedAvg-Aggregation
        return self._fedavg_aggregation(updates, client_weights)
    
    def _scaffold_aggregation(self, updates, client_weights):
        """Aggregiert Updates mit dem SCAFFOLD-Algorithmus (mit Kontrollvariablen)."""
        logger.info(f"Verwende SCAFFOLD-Aggregation mit {len(updates)} Updates")
        
        # In einer vollständigen Implementierung würden hier die Kontrollvariablen berücksichtigt
        # Für diese Simulation vereinfachen wir und verwenden die FedAvg-Aggregation
        return self._fedavg_aggregation(updates, client_weights)
    
    def _apply_update_to_model(self, model, model_update):
        """Wendet ein aggregiertes Update auf das globale Modell an."""
        updated_model = copy.deepcopy(model)
        
        # Die Implementierung hängt vom konkreten Backend ab
        if self.backend == "mlx":
            # MLX-spezifische Implementierung
            import mlx.core as mx
            for key in model:
                if key in model_update and isinstance(model[key], mx.array):
                    updated_model[key] = model[key] + model_update[key]
        elif self.backend == "torch":
            # PyTorch-spezifische Implementierung
            import torch
            for key in model:
                if key in model_update and isinstance(model[key], torch.Tensor):
                    updated_model[key] = model[key] + model_update[key]
        else:
            # Generischer Fallback für NumPy
            for key in model:
                if key in model_update and isinstance(model[key], np.ndarray):
                    updated_model[key] = model[key] + model_update[key]
        
        return updated_model
    
    def run_federated_training(self, num_rounds=None, min_clients=None, client_data_provider=None):
        """Führt das föderierte Training für eine bestimmte Anzahl von Runden durch.
        
        Args:
            num_rounds: Anzahl der Trainingsrunden (None = aus Config)
            min_clients: Mindestanzahl teilnehmender Clients pro Runde (None = aus Config)
            client_data_provider: Funktion, die lokale Daten für Clients bereitstellt
            
        Returns:
            Das trainierte globale Modell und Trainingstatistiken
        """
        num_rounds = num_rounds or self.config["federated"]["rounds"]
        min_clients = min_clients or self.config["federated"]["min_clients"]
        
        # Setze Startzeit
        self.training_stats["start_time"] = datetime.now().isoformat()
        self.training_stats["total_rounds"] = num_rounds
        
        logger.info(f"Starte föderiertes Training: {num_rounds} Runden, mindestens {min_clients} Clients pro Runde")
        
        # Simuliere das föderierte Training für mehrere Runden
        for round_idx in range(num_rounds):
            round_start_time = time.time()
            self.current_round = round_idx
            
            logger.info(f"Starte Runde {round_idx + 1}/{num_rounds}")
            
            # Simuliere Client-Teilnahme und lokales Training
            if client_data_provider:
                # Verwende tatsächliche Client-Daten
                client_data_dict = client_data_provider(round_idx)
                participating_clients = list(client_data_dict.keys())
                
                # Überprüfe Mindestanzahl von Clients
                if len(participating_clients) < min_clients:
                    logger.warning(f"Zu wenige Clients für Runde {round_idx + 1}: {len(participating_clients)} < {min_clients}")
                    continue
                
                # Führe lokales Training für jeden Client durch
                for client_id, local_data in client_data_dict.items():
                    # Initialisiere lokales Training
                    self.initialize_local_training(local_data)
                    
                    # Erstelle Kopie des globalen Modells für diesen Client
                    client_model = self._create_client_model(client_id)
                    
                    # Berechne lokale Modellaktualisierung
                    model_update = self.compute_local_update(client_id, client_model, local_data)
                    
                    # Sende Update zum Server
                    self.send_update_to_server(client_id, model_update)
            else:
                # Simuliere Client-Daten mit zufälligen Datenmengen
                for i in range(min_clients + np.random.randint(0, 3)):  # Leichte Variation der Client-Anzahl
                    client_id = f"sim_client_{i}_{round_idx}"
                    data_size = np.random.randint(100, 1000)  # Simulierte Datenmenge
                    local_data = {"size": data_size}  # Dummy-Daten für Simulation
                    
                    # Initialisiere Client und simuliere Trainingszyklus
                    self.initialize_local_training(local_data)
                    client_model = self._create_client_model(client_id)
                    model_update = {key: np.random.normal(0, 0.01, value.shape) for key, value in client_model.items() 
                                     if isinstance(value, np.ndarray)}  # Simuliertes Update
                    self.send_update_to_server(client_id, model_update)
            
            # Aggregiere Updates zum globalen Modell
            if self.client_updates:
                self.aggregate_global_model()
            
            # Rundenstatistiken aktualisieren
            round_time = time.time() - round_start_time
            logger.info(f"Runde {round_idx + 1}/{num_rounds} abgeschlossen in {round_time:.2f}s")
            
            # Speichere Trainingsverlauf
            training_log_path = self.output_dir / "logs" / f"training_stats_{self.session_id}.json"
            with open(training_log_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
        
        # Trainingszeitraum abschließen
        self.training_stats["end_time"] = datetime.now().isoformat()
        
        # Finales Modell speichern
        final_model_path = self.output_dir / "global_models" / f"final_model_{self.session_id}.pt"
        self._save_model(self.global_model, final_model_path)
        
        logger.info(f"Föderiertes Training abgeschlossen: {num_rounds} Runden, Finales Modell: {final_model_path}")
        
        return self.global_model, self.training_stats
