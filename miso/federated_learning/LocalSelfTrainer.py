#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LocalSelfTrainer.py
==================

Modul für autonomes On-Device-Training der MISO Ultimate AGI.
Erkennt neue Daten auf dem Gerät, führt lokales Training durch und 
erstellt personalisierte Modell-Updates ohne Datenübertragung.

Teil der MISO Ultimate AGI - Phase 6 (Federated Learning System)
"""

import os
import sys
import time
import json
import uuid
import logging
import datetime
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import copy
import threading
import multiprocessing
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MISO.LocalSelfTrainer')

class LocalSelfTrainer:
    """
    Modul für autonomes On-Device-Training, das neue Daten erkennt und 
    personalisierte Modell-Updates lokal erstellt, ohne dass Daten übertragen werden.
    
    Features:
    - Automatische Erkennung neuer lokaler Daten
    - Kontinuierliches autonomes Training
    - Ressourcen- und energieeffizientes Training
    - Personalisierte Modellanpassungen
    - Kompatibilität mit mehreren ML-Backends (MLX, PyTorch)
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None, 
                 model_path: Optional[str] = None,
                 data_dir: Optional[str] = None,
                 base_output_dir: Optional[str] = None,
                 batch_size: int = 8,
                 learning_rate: float = 0.001,
                 max_epochs: int = 5,
                 device: Optional[str] = None,
                 memory_limit_mb: int = 512,
                 energy_efficient: bool = True):
        """
        Initialisiert den LocalSelfTrainer.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            model_path: Pfad zum vortrainierten Modell, das lokal angepasst werden soll
            data_dir: Verzeichnis, das auf neue Daten überwacht werden soll
            base_output_dir: Basisverzeichnis für Ausgaben (Modelle, Logs, etc.)
            batch_size: Batch-Größe für das Training
            learning_rate: Lernrate für das Training
            max_epochs: Maximale Anzahl an Trainingsepochen
            device: Gerät für das Training (auto, cpu, gpu, mps, etc.)
            memory_limit_mb: Speicherlimit in Megabytes
            energy_efficient: Ob energieeffizientes Training verwendet werden soll
        """
        # Generiere eine eindeutige Session-ID
        self.session_id = str(uuid.uuid4())
        logger.info(f"Initialisiere LocalSelfTrainer mit Session-ID: {self.session_id}")
        
        # Speichere die Konfigurationen
        self.config = self._load_config(config_path) if config_path else {}
        self.model_path = model_path
        self.data_dir = data_dir or self.config.get('data_dir', './data')
        self.base_output_dir = base_output_dir or self.config.get('output_dir', './output')
        self.batch_size = batch_size or self.config.get('batch_size', 8)
        self.learning_rate = learning_rate or self.config.get('learning_rate', 0.001)
        self.max_epochs = max_epochs or self.config.get('max_epochs', 5)
        self.memory_limit_mb = memory_limit_mb or self.config.get('memory_limit_mb', 512)
        self.energy_efficient = energy_efficient or self.config.get('energy_efficient', True)
        
        # Erstelle die erforderlichen Verzeichnisse
        self._setup_directories()
        
        # Detektiere das optimale Backend und Gerät
        self.backend, self.device = self._detect_backend_and_device(device)
        logger.info(f"Verwende Backend: {self.backend} auf Gerät: {self.device}")
        
        # Initialisiere Tracking-Variablen
        self.training_stats = {
            "iterations": 0,
            "trained_samples": 0,
            "detected_new_data": 0,
            "energy_usage": [],
            "memory_usage": [],
            "training_time": 0,
            "last_training": None,
            "personalization_factor": 0.5  # Balance zwischen globalem und personalem Modell
        }
        
        # Initialisiere Modell und Daten-Tracking
        self.model = None
        self.data_registry = {}  # Speichert Informationen über bekannte Daten
        self.is_training = False
        self.training_lock = threading.Lock()
        
        # Energieüberwachung
        self.energy_monitor = None
        if self.energy_efficient:
            self._setup_energy_monitoring()
        
        logger.info("LocalSelfTrainer erfolgreich initialisiert")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Lädt die Konfigurationsdatei."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Konfiguration aus {config_path} geladen")
            return config
        except Exception as e:
            logger.warning(f"Konnte Konfiguration nicht laden: {e}. Verwende Standardeinstellungen.")
            return {}
    
    def _setup_directories(self) -> None:
        """Erstellt die erforderlichen Verzeichnisse."""
        try:
            # Erstelle Basisverzeichnis
            os.makedirs(self.base_output_dir, exist_ok=True)
            
            # Erstelle spezifische Unterverzeichnisse
            self.models_dir = os.path.join(self.base_output_dir, 'models')
            self.logs_dir = os.path.join(self.base_output_dir, 'logs')
            self.cache_dir = os.path.join(self.base_output_dir, 'cache')
            
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.logs_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Erstelle die Datei für das Datenregistry, falls sie nicht existiert
            self.registry_path = os.path.join(self.base_output_dir, 'data_registry.json')
            if not os.path.exists(self.registry_path):
                with open(self.registry_path, 'w') as f:
                    json.dump({}, f)
            
            logger.info(f"Verzeichnisstruktur unter {self.base_output_dir} eingerichtet")
        except Exception as e:
            logger.error(f"Fehler beim Einrichten der Verzeichnisse: {e}")
            raise
    
    def _detect_backend_and_device(self, requested_device: Optional[str] = None) -> Tuple[str, str]:
        """
        Detektiert das optimale Backend und Gerät basierend auf der verfügbaren Hardware.
        Prüft auf MLX (Apple Silicon), PyTorch mit MPS/CUDA, und Fallback auf CPU.
        """
        backend = "numpy"  # Standard-Fallback
        device = "cpu"     # Standard-Fallback
        
        # Prüfe auf verfügbare Backends
        available_backends = []
        
        # Prüfe auf MLX (Apple Silicon)
        try:
            import mlx.core
            available_backends.append("mlx")
            # Prüfe auf Apple Silicon
            if sys.platform == "darwin" and mlx.core.isAvailable():
                backend = "mlx"
                device = "apple_silicon"
                logger.info("MLX auf Apple Silicon verfügbar und wird verwendet")
        except ImportError:
            logger.debug("MLX nicht verfügbar")
        
        # Prüfe auf PyTorch
        try:
            import torch
            available_backends.append("torch")
            
            # Wenn MLX nicht verfügbar ist oder explizit PyTorch angefordert wird
            if backend != "mlx" or (requested_device and 'torch' in requested_device.lower()):
                backend = "torch"
                
                # Prüfe auf MPS (Metal Performance Shaders)
                if torch.backends.mps.is_available():
                    device = "mps"
                    logger.info("PyTorch mit MPS verfügbar und wird verwendet")
                # Prüfe auf CUDA
                elif torch.cuda.is_available():
                    device = "cuda"
                    logger.info("PyTorch mit CUDA verfügbar und wird verwendet")
                else:
                    device = "cpu"
                    logger.info("PyTorch auf CPU wird verwendet")
        except ImportError:
            logger.debug("PyTorch nicht verfügbar")
        
        # Überschreibe mit explizitem Device-Request, wenn möglich
        if requested_device:
            if requested_device.lower() == "mlx" and "mlx" in available_backends:
                backend = "mlx"
                device = "apple_silicon"
            elif requested_device.lower() in ["cuda", "gpu"] and "torch" in available_backends and torch.cuda.is_available():
                backend = "torch"
                device = "cuda"
            elif requested_device.lower() == "mps" and "torch" in available_backends and torch.backends.mps.is_available():
                backend = "torch"
                device = "mps"
            elif requested_device.lower() == "cpu":
                if "torch" in available_backends:
                    backend = "torch"
                    device = "cpu"
                elif "mlx" in available_backends:
                    backend = "mlx"
                    device = "cpu"
                else:
                    backend = "numpy"
                    device = "cpu"
        
        logger.info(f"Detektierte verfügbare Backends: {available_backends}")
        logger.info(f"Ausgewähltes Backend: {backend} auf Gerät: {device}")
        
        return backend, device
    
    def _setup_energy_monitoring(self) -> None:
        """Richtet die Energieüberwachung ein, wenn verfügbar."""
        # Einfaches Interface, könnte später durch komplexere Lösungen ersetzt werden
        self.energy_monitor = {
            'active': self.energy_efficient,
            'start_time': None,
            'cpu_usage': [],
            'memory_usage': [],
            'last_check': time.time()
        }
        
        logger.info(f"Energieüberwachung eingerichtet: {self.energy_efficient}")
    
    # ===================================
    # Kernfunktionen für lokales Training
    # ===================================
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Lädt ein vortrainiertes Modell für lokale Anpassung."""
        model_path = model_path or self.model_path
        if not model_path:
            logger.error("Kein Modellpfad angegeben. Kann Modell nicht laden.")
            return False
        
        try:
            if self.backend == "torch":
                import torch
                self.model = torch.load(model_path, map_location=self.device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()  # Setze Modell in Evaluationsmodus
            
            elif self.backend == "mlx":
                import mlx.nn as nn
                # MLX arbeitet mit Parametern als Dict
                self.model = nn.load_weights(model_path)
            
            else:  # Numpy oder anderes Backend
                # Versuche ein generisches Laden mit NumPy
                self.model = np.load(model_path, allow_pickle=True)
                if isinstance(self.model, np.lib.npyio.NpzFile):
                    # Konvertiere zu Dictionary für einfachere Handhabung
                    self.model = {key: self.model[key] for key in self.model.files}
            
            logger.info(f"Modell erfolgreich geladen aus {model_path}")
            
            # Speichere Modellstatistiken für Vergleiche
            self.model_stats = self._get_model_stats(self.model)
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
            return False
    
    def detect_new_data(self, data_dir: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Erkennt neue Daten im überwachten Verzeichnis."""
        data_dir = data_dir or self.data_dir
        if not data_dir or not os.path.exists(data_dir):
            logger.warning(f"Datenverzeichnis {data_dir} existiert nicht")
            return False, []
        
        # Lade das aktuelle Datenregistry
        try:
            with open(self.registry_path, 'r') as f:
                self.data_registry = json.load(f)
        except Exception as e:
            logger.warning(f"Konnte Datenregistry nicht laden: {e}. Starte mit leerem Registry.")
            self.data_registry = {}
        
        # Scanne das Verzeichnis nach Dateien
        new_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                # Ignoriere versteckte Dateien und Unterverzeichnisse
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                file_info = self._get_file_info(file_path)
                
                # Prüfe, ob die Datei neu oder verändert ist
                if file_path not in self.data_registry or \
                   self.data_registry[file_path]['modified'] != file_info['modified'] or \
                   self.data_registry[file_path]['size'] != file_info['size']:
                    new_files.append(file_path)
                    self.data_registry[file_path] = file_info
        
        # Aktualisiere das Registry, wenn neue Dateien gefunden wurden
        if new_files:
            self.training_stats["detected_new_data"] += len(new_files)
            logger.info(f"{len(new_files)} neue oder veränderte Dateien gefunden")
            
            # Speichere das aktualisierte Registry
            with open(self.registry_path, 'w') as f:
                json.dump(self.data_registry, f)
            
            return True, new_files
        else:
            logger.debug("Keine neuen Daten erkannt")
            return False, []
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Extrahiert Metadaten aus einer Datei."""
        stats = os.stat(file_path)
        return {
            'path': file_path,
            'size': stats.st_size,
            'modified': stats.st_mtime,
            'created': stats.st_ctime,
            'last_accessed': stats.st_atime,
            'extension': os.path.splitext(file_path)[1].lower(),
            'registered': time.time()
        }
    
    def prepare_training_data(self, file_paths: List[str]) -> Any:
        """Bereitet die Trainingsdaten aus den neuen Dateien vor."""
        # Diese Funktion muss an den spezifischen Datentyp angepasst werden
        # Hier ein generisches Beispiel:
        
        training_data = []
        for file_path in file_paths:
            try:
                # Generischer Datenleseversuch basierend auf Dateityp
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext in ['.csv', '.txt']:
                    # Für CSV/Text-Dateien
                    data = np.loadtxt(file_path, delimiter=',', dtype=float)
                    training_data.append(data)
                    
                elif ext in ['.npy', '.npz']:
                    # Für NumPy-Dateien
                    data = np.load(file_path)
                    training_data.append(data)
                    
                elif ext in ['.json']:
                    # Für JSON-Dateien
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    training_data.append(data)
                    
                else:
                    logger.warning(f"Nicht unterstütztes Dateiformat: {ext} für {file_path}")
                    continue
                    
                logger.debug(f"Daten aus {file_path} geladen")
            except Exception as e:
                logger.error(f"Fehler beim Laden der Daten aus {file_path}: {e}")
        
        logger.info(f"{len(training_data)} Datendateien für das Training vorbereitet")
        return training_data
    
    def train_on_local_data(self, training_data: Any, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Führt lokales Training auf den neuen Daten durch."""
        # Prüfe, ob bereits eine Trainingsinstanz läuft
        if self.is_training:
            logger.warning("Training bereits im Gange. Neues Training übersprungen.")
            return {"status": "skipped", "reason": "training_in_progress"}
        
        # Prüfe, ob ein Modell geladen ist
        if self.model is None:
            logger.error("Kein Modell geladen. Trainingsversuch abgebrochen.")
            return {"status": "failed", "reason": "no_model_loaded"}
        
        # Prüfe auf leere Trainingsdaten
        if not training_data or len(training_data) == 0:
            logger.warning("Keine Trainingsdaten vorhanden. Training übersprungen.")
            return {"status": "skipped", "reason": "no_training_data"}
        
        # Setze Trainings-Flag und erwerbe Lock
        with self.training_lock:
            if self.is_training:
                return {"status": "skipped", "reason": "training_in_progress"}
            self.is_training = True
        
        epochs = epochs or self.max_epochs
        
        try:
            # Starte Energieüberwachung, wenn aktiviert
            if self.energy_efficient and self.energy_monitor:
                self.energy_monitor['start_time'] = time.time()
                self.energy_monitor['last_check'] = time.time()
            
            # Speichere ursprüngliches Modell für Vorher-Nachher-Vergleich
            original_model = self._copy_model(self.model)
            
            # Führe training basierend auf Backend durch
            training_start = time.time()
            if self.backend == "torch":
                training_results = self._train_torch_model(training_data, epochs)
            elif self.backend == "mlx":
                training_results = self._train_mlx_model(training_data, epochs)
            else:
                # Generic training (sehr vereinfacht, in der Praxis komplexer)
                logger.warning("Training mit generischem Backend nicht vollständig implementiert")
                training_results = {"status": "not_implemented"}
            
            training_end = time.time()
            training_time = training_end - training_start
            
            # Update Trainingsstatistiken
            self.training_stats["iterations"] += 1
            self.training_stats["training_time"] += training_time
            self.training_stats["last_training"] = datetime.datetime.now().isoformat()
            
            # Vergleiche Modelle und berechne Änderungen
            model_diff = self._calculate_model_diff(original_model, self.model)
            
            # Speichere personalisiertes Modell
            model_save_path = os.path.join(
                self.models_dir, 
                f"personalized_model_{self.session_id}_{self.training_stats['iterations']}.{self._get_model_extension()}"
            )
            self._save_model(self.model, model_save_path)
            
            # Generiere umfassenden Trainingsbericht
            training_report = {
                "status": "completed",
                "training_time": training_time,
                "epochs": epochs,
                "samples_trained": len(training_data),
                "model_diff": model_diff,
                "model_saved_at": model_save_path,
                "backend": self.backend,
                "device": self.device,
                "timestamp": time.time(),
                **training_results  # Fügt backend-spezifische Ergebnisse hinzu
            }
            
            # Speichere Trainingsbericht in JSON-Log
            self._log_training_results(training_report)
            
            logger.info(f"Lokales Training abgeschlossen in {training_time:.2f}s. "
                        f"Modell gespeichert unter {model_save_path}")
            
            return training_report
            
        except Exception as e:
            logger.error(f"Fehler beim lokalen Training: {e}")
            return {"status": "failed", "reason": str(e)}
        
        finally:
            # Setze Trainings-Flag zurück
            with self.training_lock:
                self.is_training = False
    
    def compute_personalized_update(self) -> Dict[str, Any]:
        """Berechnet ein personalisiertes Update basierend auf dem lokalen Training."""
        if self.model is None or not hasattr(self, 'model_stats') or 'original_params' not in self.model_stats:
            logger.error("Kein Modell oder keine Originalparameter vorhanden.")
            return {"status": "failed", "reason": "no_model_or_original_params"}
        
        try:
            # Extrahiere das personalisierte Update (Differenz zwischen lokalem und originalem Modell)
            update = {}
            
            if self.backend == "torch":
                import torch
                # Für PyTorch-Modelle
                if hasattr(self.model, 'state_dict') and hasattr(self.model_stats['original_params'], 'items'):
                    current_state = self.model.state_dict()
                    for key, orig_param in self.model_stats['original_params'].items():
                        if key in current_state:
                            # Berechne die Differenz für jeden Parameter
                            update[key] = current_state[key] - orig_param
            
            elif self.backend == "mlx":
                import mlx.core as mx
                # Für MLX-Modelle (Parameter als Dict)
                if isinstance(self.model, dict) and isinstance(self.model_stats['original_params'], dict):
                    for key, orig_param in self.model_stats['original_params'].items():
                        if key in self.model:
                            # Berechne die Differenz für jeden Parameter
                            update[key] = self.model[key] - orig_param
            
            else:
                # Für NumPy-basierte Modelle
                if isinstance(self.model, dict) and isinstance(self.model_stats['original_params'], dict):
                    for key, orig_param in self.model_stats['original_params'].items():
                        if key in self.model:
                            # Berechne die Differenz für jeden Parameter
                            update[key] = self.model[key] - orig_param
            
            # Erstelle Metadaten für das Update
            update_metadata = {
                "session_id": self.session_id,
                "timestamp": time.time(),
                "backend": self.backend,
                "device": self.device,
                "iterations": self.training_stats["iterations"],
                "trained_samples": self.training_stats["trained_samples"],
                "training_time": self.training_stats["training_time"],
                "personalization_factor": self.training_stats["personalization_factor"]
            }
            
            # Speichere das Update
            update_path = os.path.join(
                self.models_dir, 
                f"personalized_update_{self.session_id}_{self.training_stats['iterations']}.npz"
            )
            
            # Speichere Update und Metadaten
            if self.backend == "torch":
                import torch
                # Speichere als PyTorch-Datei mit Metadaten
                torch.save({"update": update, "metadata": update_metadata}, update_path)
            else:
                # Speichere als NumPy-Datei mit Metadaten
                np.savez_compressed(update_path, **{"update": update, "metadata": update_metadata})
            
            logger.info(f"Personalisiertes Update erstellt und gespeichert unter {update_path}")
            
            return {
                "status": "success",
                "path": update_path,
                "metadata": update_metadata
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des personalisierten Updates: {e}")
            return {"status": "failed", "reason": str(e)}
    
    def apply_global_model_update(self, global_model_path: str, 
                                personalization_factor: Optional[float] = None) -> bool:
        """Wendet ein globales Modellupdate an und behält personalisierte Anpassungen bei."""
        if self.model is None:
            logger.error("Kein lokales Modell geladen. Kann globales Update nicht anwenden.")
            return False
        
        # Setze Personalisierungsfaktor (zwischen 0 und 1)
        # 0: Vollständig globales Modell, 1: Vollständig personalisiertes Modell
        if personalization_factor is not None:
            self.training_stats["personalization_factor"] = max(0.0, min(1.0, personalization_factor))
        
        try:
            # Lade das globale Modell
            if self.backend == "torch":
                import torch
                global_model = torch.load(global_model_path, map_location=self.device)
            elif self.backend == "mlx":
                import mlx.nn as nn
                global_model = nn.load_weights(global_model_path)
            else:
                # NumPy-Fallback
                global_model = np.load(global_model_path, allow_pickle=True)
                if isinstance(global_model, np.lib.npyio.NpzFile):
                    global_model = {key: global_model[key] for key in global_model.files}
            
            # Berechne personalisierte Parameter als gewichtete Kombination von globalem und lokalem Modell
            alpha = self.training_stats["personalization_factor"]
            
            # Interpoliere zwischen globalem und lokalem Modell
            if self.backend == "torch":
                import torch
                # Für PyTorch-Modelle
                if hasattr(self.model, 'state_dict') and hasattr(global_model, 'state_dict'):
                    current_state = self.model.state_dict()
                    global_state = global_model.state_dict()
                    
                    merged_state = {}
                    for key in global_state.keys():
                        if key in current_state:
                            # Lineare Interpolation für jeden Parameter
                            merged_state[key] = (1 - alpha) * global_state[key] + alpha * current_state[key]
                        else:
                            merged_state[key] = global_state[key]
                    
                    # Aktualisiere das lokale Modell mit den gemischten Parametern
                    self.model.load_state_dict(merged_state)
            
            elif self.backend == "mlx":
                import mlx.core as mx
                # Für MLX-Modelle
                if isinstance(self.model, dict) and isinstance(global_model, dict):
                    merged_params = {}
                    for key in global_model.keys():
                        if key in self.model:
                            # Lineare Interpolation für jeden Parameter
                            merged_params[key] = (1 - alpha) * global_model[key] + alpha * self.model[key]
                        else:
                            merged_params[key] = global_model[key]
                    
                    # Aktualisiere das lokale Modell
                    self.model.update(merged_params)
            
            else:
                # Für NumPy-basierte Modelle
                if isinstance(self.model, dict) and isinstance(global_model, dict):
                    for key in global_model.keys():
                        if key in self.model:
                            # Lineare Interpolation für jeden Parameter
                            self.model[key] = (1 - alpha) * global_model[key] + alpha * self.model[key]
                        else:
                            self.model[key] = global_model[key]
            
            # Speichere das aktualisierte Modell
            model_save_path = os.path.join(
                self.models_dir, 
                f"merged_model_{self.session_id}_{time.time()}.{self._get_model_extension()}"
            )
            self._save_model(self.model, model_save_path)
            
            logger.info(f"Globales Modellupdate angewendet mit Personalisierungsfaktor {alpha}. "
                       f"Gespeichert unter {model_save_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Anwenden des globalen Modellupdates: {e}")
            return False
    
    # ========================================
    # Modell-Hilfsfunktionen
    # ========================================
    
    def _get_model_stats(self, model) -> Dict[str, Any]:
        """Extrahiert Statistiken aus einem Modell."""
        stats = {
            "timestamp": time.time(),
            "backend": self.backend,
            "device": self.device
        }
        
        try:
            if self.backend == "torch":
                import torch
                # PyTorch-spezifische Statistiken
                if hasattr(model, 'state_dict'):
                    # Speichere Originalparameter
                    stats["original_params"] = copy.deepcopy(model.state_dict())
                    
                    # Zähle Parameter
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    stats["total_params"] = total_params
                    stats["trainable_params"] = trainable_params
                    
                    # Modellgröße abschätzen (sehr grob)
                    mem_params = sum(p.numel() * p.element_size() for p in model.parameters())
                    mem_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
                    stats["memory_estimate"] = mem_params + mem_buffers
            
            elif self.backend == "mlx":
                import mlx.core as mx
                # MLX-spezifische Statistiken
                if hasattr(model, 'parameters'):
                    # Speichere Originalparameter
                    stats["original_params"] = copy.deepcopy(model.parameters())
                    
                    # Zähle Parameter und schätze Speicherverbrauch
                    total_params = 0
                    memory_estimate = 0
                    for param in model.parameters().values():
                        if isinstance(param, mx.array):
                            total_params += param.size
                            # Schätze Speicherverbrauch (Anzahl Elemente * Größe eines Elements)
                            element_size = 4  # float32 Standard (4 bytes)
                            if param.dtype == mx.float16:
                                element_size = 2
                            memory_estimate += param.size * element_size
                    
                    stats["total_params"] = total_params
                    stats["memory_estimate"] = memory_estimate
                
                elif isinstance(model, dict):
                    # Speichere Originalparameter für dict-basierte Modelle
                    stats["original_params"] = copy.deepcopy(model)
                    
                    # Schätze Größe und Parameter
                    total_params = 0
                    memory_estimate = 0
                    for param in model.values():
                        if isinstance(param, mx.array):
                            total_params += param.size
                            element_size = 4  # float32 Standard
                            if param.dtype == mx.float16:
                                element_size = 2
                            memory_estimate += param.size * element_size
                    
                    stats["total_params"] = total_params
                    stats["memory_estimate"] = memory_estimate
            
            else:
                # Generische Statistiken für andere Modelltypen
                if isinstance(model, dict):
                    # Speichere Original-Parameter
                    stats["original_params"] = copy.deepcopy(model)
                    
                    # Schätze Parameter und Größe
                    total_params = 0
                    memory_estimate = 0
                    for key, param in model.items():
                        if isinstance(param, np.ndarray):
                            total_params += param.size
                            memory_estimate += param.nbytes
                    
                    stats["total_params"] = total_params
                    stats["memory_estimate"] = memory_estimate
                else:
                    # Fallback für unbekannte Modelltypen
                    stats["original_params"] = None
                    stats["memory_estimate"] = sys.getsizeof(model)
            
            return stats
            
        except Exception as e:
            logger.error(f"Fehler bei der Erstellung von Modellstatistiken: {e}")
            # Stelle sicher, dass zumindest die Originalparameter gespeichert werden
            stats["original_params"] = copy.deepcopy(model) if not isinstance(model, (torch.nn.Module, mx.array)) else None
            return stats
    
    def _copy_model(self, model):
        """Erstellt eine tiefe Kopie des Modells."""
        try:
            if self.backend == "torch":
                import torch
                # Für PyTorch-Modelle
                if hasattr(model, 'state_dict'):
                    # Erstelle eine Kopie der Parameter
                    return copy.deepcopy(model)
            
            elif self.backend == "mlx":
                import mlx.core as mx
                # Für MLX-Modelle
                if hasattr(model, 'parameters'):
                    # Deep-Copy der Parameter
                    return copy.deepcopy(model)
                elif isinstance(model, dict):
                    return copy.deepcopy(model)
            
            # Für andere Modelltypen
            return copy.deepcopy(model)
            
        except Exception as e:
            logger.error(f"Fehler beim Kopieren des Modells: {e}")
            return model  # Im Fehlerfall Original zurückgeben
    
    def _calculate_model_diff(self, original_model, current_model) -> Dict[str, Any]:
        """Berechnet die Unterschiede zwischen dem ursprünglichen und dem aktuellen Modell."""
        diff_stats = {
            "total_params_changed": 0,
            "total_params": 0,
            "avg_absolute_change": 0.0,
            "max_absolute_change": 0.0,
            "change_stats": {}
        }
        
        try:
            if self.backend == "torch":
                import torch
                # Für PyTorch-Modelle
                if hasattr(original_model, 'state_dict') and hasattr(current_model, 'state_dict'):
                    orig_state = original_model.state_dict()
                    curr_state = current_model.state_dict()
                    
                    total_abs_diff = 0.0
                    max_abs_diff = 0.0
                    params_changed = 0
                    total_params = 0
                    
                    for key in orig_state.keys():
                        if key in curr_state:
                            orig_param = orig_state[key].cpu().detach().numpy().flatten()
                            curr_param = curr_state[key].cpu().detach().numpy().flatten()
                            
                            # Nur für numerische Parameter
                            if orig_param.dtype.kind in 'iufc':
                                param_diff = np.abs(curr_param - orig_param)
                                has_changed = np.any(param_diff > 0)
                                
                                if has_changed:
                                    params_changed += 1
                                    avg_diff = np.mean(param_diff)
                                    max_diff = np.max(param_diff)
                                    total_abs_diff += avg_diff
                                    max_abs_diff = max(max_abs_diff, max_diff)
                                    
                                    diff_stats["change_stats"][key] = {
                                        "avg_change": float(avg_diff),
                                        "max_change": float(max_diff),
                                        "num_params": len(orig_param)
                                    }
                                
                                total_params += 1
                    
                    diff_stats["total_params_changed"] = params_changed
                    diff_stats["total_params"] = total_params
                    diff_stats["avg_absolute_change"] = total_abs_diff / max(1, params_changed)
                    diff_stats["max_absolute_change"] = max_abs_diff
                    diff_stats["change_percentage"] = (params_changed / max(1, total_params)) * 100
            
            elif self.backend == "mlx" or isinstance(original_model, dict):
                # Für MLX oder dict-basierte Modelle
                
                # Normalisiere auf Dictionary-Form
                orig_params = original_model
                curr_params = current_model
                
                if hasattr(original_model, 'parameters'):
                    orig_params = original_model.parameters()
                if hasattr(current_model, 'parameters'):
                    curr_params = current_model.parameters()
                
                total_abs_diff = 0.0
                max_abs_diff = 0.0
                params_changed = 0
                total_params = 0
                
                for key in orig_params.keys():
                    if key in curr_params:
                        # Konvertiere zu NumPy für Vergleiche
                        orig_param = np.array(orig_params[key].tolist() if hasattr(orig_params[key], 'tolist') else orig_params[key]).flatten()
                        curr_param = np.array(curr_params[key].tolist() if hasattr(curr_params[key], 'tolist') else curr_params[key]).flatten()
                        
                        # Nur für numerische Parameter
                        if orig_param.dtype.kind in 'iufc':
                            param_diff = np.abs(curr_param - orig_param)
                            has_changed = np.any(param_diff > 0)
                            
                            if has_changed:
                                params_changed += 1
                                avg_diff = np.mean(param_diff)
                                max_diff = np.max(param_diff)
                                total_abs_diff += avg_diff
                                max_abs_diff = max(max_abs_diff, max_diff)
                                
                                diff_stats["change_stats"][key] = {
                                    "avg_change": float(avg_diff),
                                    "max_change": float(max_diff),
                                    "num_params": len(orig_param)
                                }
                            
                            total_params += 1
                
                diff_stats["total_params_changed"] = params_changed
                diff_stats["total_params"] = total_params
                diff_stats["avg_absolute_change"] = total_abs_diff / max(1, params_changed)
                diff_stats["max_absolute_change"] = max_abs_diff
                diff_stats["change_percentage"] = (params_changed / max(1, total_params)) * 100
            
            return diff_stats
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der Modelldifferenz: {e}")
            return diff_stats
    
    def _save_model(self, model, file_path: str) -> bool:
        """Speichert ein Modell an den angegebenen Pfad."""
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if self.backend == "torch":
                import torch
                torch.save(model, file_path)
            
            elif self.backend == "mlx":
                import mlx.nn as nn
                if hasattr(model, 'parameters'):
                    nn.save_weights(file_path, model.parameters())
                elif isinstance(model, dict):
                    # Speichere als NPZ-Datei
                    np.savez_compressed(file_path, **model)
            
            else:
                # Für dict oder andere Modelltypen
                if isinstance(model, dict):
                    np.savez_compressed(file_path, **model)
                else:
                    # Versuche Picklen als letzten Ausweg
                    import pickle
                    with open(file_path, 'wb') as f:
                        pickle.dump(model, f)
            
            logger.info(f"Modell gespeichert unter {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Modells: {e}")
            return False
    
    def _get_model_extension(self) -> str:
        """Gibt die passende Dateierweiterung für das aktuelle Backend zurück."""
        if self.backend == "torch":
            return "pt"
        elif self.backend == "mlx":
            return "npz"
        else:
            return "npz"
    
    # ========================================
    # Backend-spezifische Trainingsfunktionen
    # ========================================
    
    def _train_torch_model(self, training_data: List[Any], epochs: int) -> Dict[str, Any]:
        """Führt Training mit PyTorch durch."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            # Starte Energieeffizienzüberwachung
            energy_stats = self._start_energy_monitoring()
            
            # Prüfe, ob das Modell ein PyTorch-Modul ist
            if not isinstance(self.model, nn.Module):
                logger.error("Das Modell ist kein PyTorch-Modul")
                return {"status": "failed", "reason": "model_not_torch_module"}
            
            # Setze Modell in Trainingsmodus
            self.model.train()
            
            # Energieeffizienztipp: Verwende automatische Mixed-Precision-Training, wenn auf CUDA oder MPS
            use_amp = False
            scaler = None
            if self.device in ["cuda", "mps"] and torch.__version__ >= "1.6.0":
                try:
                    from torch.cuda.amp import autocast, GradScaler
                    use_amp = True
                    scaler = GradScaler()
                    logger.info("Automatische Mixed-Precision-Training aktiviert")
                except ImportError:
                    logger.info("Automatische Mixed-Precision-Training nicht verfügbar")
            
            # Verwende den Gerätekontext
            device = torch.device(self.device if self.device in ["cpu", "cuda"] else 
                               ("mps" if self.device == "mps" and torch.backends.mps.is_available() else "cpu"))
            self.model = self.model.to(device)
            
            # Vorbereite Optimizer und Loss-Funktion
            # Erkennen des Modelltyps und Anpassen des Optimizers und der Loss-Funktion
            if hasattr(self.model, "loss_fn") and callable(self.model.loss_fn):
                # Verwende modellspezifischen Loss, wenn definiert
                criterion = self.model.loss_fn
            else:
                # Automatische Erkennung des Loss-Typs
                # Standard: CrossEntropyLoss für Klassifikation
                criterion = nn.CrossEntropyLoss()
            
            # Konfiguriere den Optimizer mit dem Learning Rate aus der Konfiguration
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Bereite Tracking-Variablen vor
            total_samples = 0
            total_loss = 0.0
            start_time = time.time()
            
            # Haupttrainingsschleife
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_samples = 0
                
                # Wir brauchen unterschiedliche Prozesse für verschiedene Datentypen
                # Diese generische Implementierung geht davon aus, dass training_data eine Liste von (Eingabe, Ziel) Tupeln ist
                # In einer realen Implementierung müsste dies weiter angepasst werden
                
                for batch_idx, data_batch in enumerate(training_data):
                    # Überprüfe Energieeffizienz und pausiere bei Bedarf
                    if self.energy_efficient and (batch_idx % 10 == 0):
                        if not self._check_energy_efficiency(optimizer=optimizer):
                            logger.info("Training pausiert für Energieeffizienz")
                            time.sleep(1)  # Warte kurz, bevor fortgefahren wird
                    
                    # Prüfe auf Datentypen und passe entsprechend an
                    if isinstance(data_batch, tuple) and len(data_batch) >= 2:
                        inputs, targets = data_batch[:2]
                    elif isinstance(data_batch, dict) and 'inputs' in data_batch and 'targets' in data_batch:
                        inputs, targets = data_batch['inputs'], data_batch['targets']
                    else:
                        # Hier können weitere Datenformate unterstützt werden
                        logger.warning(f"Unbekanntes Datenformat in Batch {batch_idx}, überspringe")
                        continue
                    
                    # Konvertiere Eingaben und Ziele zu Tensoren und verschiebe auf das entsprechende Gerät
                    if not isinstance(inputs, torch.Tensor):
                        inputs = torch.tensor(inputs, device=device)
                    else:
                        inputs = inputs.to(device)
                    
                    if not isinstance(targets, torch.Tensor):
                        targets = torch.tensor(targets, device=device)
                    else:
                        targets = targets.to(device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward-Pass, ggf. mit Mixed-Precision
                    if use_amp:
                        with autocast():
                            outputs = self.model(inputs)
                            loss = criterion(outputs, targets)
                        
                        # Backward-Pass mit Scaler
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard-Training
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                    
                    # Aktualisiere Statistiken
                    batch_loss = loss.item()
                    batch_size = inputs.size(0) if hasattr(inputs, 'size') else len(inputs)
                    
                    epoch_loss += batch_loss
                    epoch_samples += batch_size
                    
                    # Speicherbereinigung bei jedem n-ten Batch
                    if torch.cuda.is_available() and batch_idx % 10 == 9:
                        torch.cuda.empty_cache()
                    
                    # Fortschrittsausgabe bei jedem n-ten Batch
                    if batch_idx % 10 == 0:
                        logger.debug(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                                   f"Loss: {batch_loss:.4f}, Samples: {batch_size}")
                
                # Epochenstatistiken
                avg_epoch_loss = epoch_loss / max(1, epoch_samples)
                total_loss += epoch_loss
                total_samples += epoch_samples
                
                logger.info(f"Epoch {epoch+1}/{epochs} abgeschlossen. "
                          f"Durchschnittlicher Loss: {avg_epoch_loss:.4f}, "
                          f"Trainierte Samples: {epoch_samples}")
                
                # Führe Validierung aus, falls verfügbar
                if hasattr(self, '_validate_model') and callable(self._validate_model):
                    validation_results = self._validate_model()
                    logger.info(f"Validierungsergebnisse: {validation_results}")
            
            # Beende Training und kehre zum Evaluationsmodus zurück
            self.model.eval()
            training_time = time.time() - start_time
            
            # Beende Energieüberwachung
            energy_stats = self._stop_energy_monitoring(energy_stats)
            
            # Aktualisiere Trainingsstatistiken
            self.training_stats["trained_samples"] += total_samples
            
            # Erstelle Trainingsbericht
            return {
                "status": "completed",
                "average_loss": total_loss / max(1, total_samples),
                "total_samples": total_samples,
                "backend": "torch",
                "device": self.device,
                "use_amp": use_amp,
                "energy_stats": energy_stats
            }
            
        except Exception as e:
            logger.error(f"Fehler beim PyTorch-Training: {e}")
            return {"status": "failed", "reason": str(e)}
    
    def _log_training_results(self, results: Dict[str, Any]) -> None:
        """Protokolliert die Trainingsergebnisse in einer JSON-Datei."""
        try:
            # Generiere Dateinamen für die Protokolldatei
            log_file = os.path.join(
                self.logs_dir, 
                f"training_log_{self.session_id}_{self.training_stats['iterations']}.json"
            )
            
            # Konvertiere nicht serialisierbare Objekte
            def json_serialize(obj):
                if isinstance(obj, (np.ndarray, np.number)):
                    return obj.tolist()
                elif isinstance(obj, (datetime.datetime, datetime.date)):
                    return obj.isoformat()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                else:
                    return str(obj)
            
            # Protokolliere die Ergebnisse in einer JSON-Datei
            with open(log_file, 'w') as f:
                json.dump(results, f, default=json_serialize, indent=2)
            
            logger.info(f"Trainingsergebnisse protokolliert in {log_file}")
            
        except Exception as e:
            logger.error(f"Fehler beim Protokollieren der Trainingsergebnisse: {e}")
    
    def _train_mlx_model(self, training_data: List[Any], epochs: int) -> Dict[str, Any]:
        """Führt Training mit MLX durch."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            
            # Starte Energieüberwachung
            energy_stats = self._start_energy_monitoring()
            
            # Prüfe, ob sich das Modell als MLX-Modell verwenden lässt
            # Unterstütze sowohl MISOTensor/MLXTensor aus der T-Mathematics Engine als auch native MLX-Module
            is_miso_tensor = hasattr(self.model, 'to_dtype') and callable(self.model.to_dtype)
            is_mlx_module = hasattr(self.model, 'parameters') and callable(self.model.parameters)
            is_dict_model = isinstance(self.model, dict)
            
            if not (is_miso_tensor or is_mlx_module or is_dict_model):
                logger.error("Das Modell ist nicht mit MLX kompatibel")
                return {"status": "failed", "reason": "model_not_mlx_compatible"}
            
            # Vorbereite Optimizer und Loss-Funktion
            # Bei MLX muss der Loss und der Update-Schritt als funktionale API definiert werden
            
            # Definiere Standard-Loss-Funktion (kann mit modellspezifischen ersetzt werden)
            def loss_fn(model_params, inputs, targets):
                # Forward-Pass
                if is_miso_tensor:
                    # Für MISOTensor/MLXTensor aus der T-Mathematics Engine
                    outputs = self.model(inputs)
                elif is_mlx_module:
                    # Für mlx.nn Module
                    outputs = self.model.apply(model_params, inputs)
                else:
                    # Fallback für dict-basierte Modelle (erfordert angepasste Vorwärtsfunktion)
                    logger.warning("Dict-basierte Modelle benötigen eine angepasste Vorwärtsfunktion")
                    # Hier müsste eine spezifische Vorwärtsfunktion für das Modell implementiert werden
                    # Vereinfachtes Beispiel:
                    outputs = self._custom_forward_pass(model_params, inputs)
                
                # Berechne Loss (Cross-Entropy als Standard)
                return nn.losses.cross_entropy(outputs, targets)
            
            # Hole Modellparameter
            if is_mlx_module:
                params = self.model.parameters()
            elif is_dict_model:
                params = self.model
            else:
                # Für MISOTensor/MLXTensor müssten wir die Parameter auf andere Weise extrahieren
                params = getattr(self.model, 'parameters', lambda: {})() or self.model
            
            # Erstelle den Optimizer
            optimizer = optim.Adam(learning_rate=self.learning_rate)
            opt_state = optimizer.init(params)
            
            # Gradient-Funktion mit Loss
            @mx.grad
            def compute_loss_and_grad(model_params, inputs, targets):
                return loss_fn(model_params, inputs, targets)
            
            # Bereite Tracking-Variablen vor
            total_samples = 0
            total_loss = 0.0
            start_time = time.time()
            
            # Haupttrainingsschleife
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_samples = 0
                
                for batch_idx, data_batch in enumerate(training_data):
                    # Überprüfe Energieeffizienz und pausiere bei Bedarf
                    if self.energy_efficient and (batch_idx % 10 == 0):
                        if not self._check_energy_efficiency():
                            logger.info("Training pausiert für Energieeffizienz")
                            time.sleep(1)  # Warte kurz, bevor fortgefahren wird
                    
                    # Prüfe auf Datentypen und passe entsprechend an
                    if isinstance(data_batch, tuple) and len(data_batch) >= 2:
                        inputs, targets = data_batch[:2]
                    elif isinstance(data_batch, dict) and 'inputs' in data_batch and 'targets' in data_batch:
                        inputs, targets = data_batch['inputs'], data_batch['targets']
                    else:
                        # Hier können weitere Datenformate unterstützt werden
                        logger.warning(f"Unbekanntes Datenformat in Batch {batch_idx}, überspringe")
                        continue
                    
                    # Konvertiere Eingaben und Ziele zu MLX-Arrays, falls nötig
                    if not isinstance(inputs, mx.array):
                        inputs = mx.array(inputs)
                    if not isinstance(targets, mx.array):
                        targets = mx.array(targets)
                    
                    # Berechne Loss und Gradienten
                    grads = compute_loss_and_grad(params, inputs, targets)
                    
                    # Aktualisiere Parameter mit dem Optimizer
                    updates, opt_state = optimizer.update(grads, opt_state)
                    params = optim.apply_updates(params, updates)
                    
                    # Direkt Loss berechnen nachdem die Parameter aktualisiert wurden
                    loss_value = loss_fn(params, inputs, targets)
                    mx.eval(loss_value)  # Erzwinge sofortige Auswertung
                    batch_loss = loss_value.item()
                    
                    # Aktualisiere Statistiken
                    batch_size = inputs.shape[0] if len(inputs.shape) > 0 else 1
                    
                    epoch_loss += batch_loss
                    epoch_samples += batch_size
                    
                    # Speicherbereinigung bei jedem n-ten Batch in MLX nicht nötig
                    # aber wir können ML Compute-Caches explizit bereinigen, wenn verfügbar
                    
                    # Fortschrittsausgabe bei jedem n-ten Batch
                    if batch_idx % 10 == 0:
                        logger.debug(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, "
                                   f"Loss: {batch_loss:.4f}, Samples: {batch_size}")
                
                # Epochenstatistiken
                avg_epoch_loss = epoch_loss / max(1, epoch_samples)
                total_loss += epoch_loss
                total_samples += epoch_samples
                
                logger.info(f"Epoch {epoch+1}/{epochs} abgeschlossen. "
                          f"Durchschnittlicher Loss: {avg_epoch_loss:.4f}, "
                          f"Trainierte Samples: {epoch_samples}")
                
                # Führe Validierung aus, falls verfügbar
                if hasattr(self, '_validate_model') and callable(self._validate_model):
                    validation_results = self._validate_model()
                    logger.info(f"Validierungsergebnisse: {validation_results}")
            
            # Aktualisiere das Modell mit den finalen Parametern
            if is_mlx_module and hasattr(self.model, 'update'):
                self.model.update(params)
            elif is_dict_model:
                self.model.update(params)
            
            training_time = time.time() - start_time
            
            # Beende Energieüberwachung
            energy_stats = self._stop_energy_monitoring(energy_stats)
            
            # Aktualisiere Trainingsstatistiken
            self.training_stats["trained_samples"] += total_samples
            
            # Erstelle Trainingsbericht
            return {
                "status": "completed",
                "average_loss": total_loss / max(1, total_samples),
                "total_samples": total_samples,
                "backend": "mlx",
                "device": self.device,
                "energy_stats": energy_stats
            }
            
        except Exception as e:
            logger.error(f"Fehler beim MLX-Training: {e}")
            return {"status": "failed", "reason": str(e)}
    
    def _custom_forward_pass(self, model_params, inputs):
        """Führt einen benutzerdefinierten Forward-Pass für dict-basierte Modelle durch."""
        # Diese generische Implementierung muss für konkrete Modelle angepasst werden
        try:
            import mlx.core as mx
            import mlx.nn as nn
            
            # Einfaches MLP als Beispiel
            # In der Praxis sollte dies komplexer und modellspezifisch sein
            x = inputs
            
            # Annahme: model_params enthält Gewichte und Biases für ein MLP
            if 'fc1.weight' in model_params and 'fc1.bias' in model_params:
                x = mx.matmul(x, model_params['fc1.weight'].T) + model_params['fc1.bias']
                x = mx.maximum(x, 0)  # ReLU
            
            if 'fc2.weight' in model_params and 'fc2.bias' in model_params:
                x = mx.matmul(x, model_params['fc2.weight'].T) + model_params['fc2.bias']
            
            return x
            
        except Exception as e:
            logger.error(f"Fehler im benutzerdefinierten Forward-Pass: {e}")
            # Fallback: Gib Zufallswerte zurück, um komplettes Training-Abbrechen zu vermeiden
            return mx.random.normal(shape=(inputs.shape[0], 10))
    
    # ==========================================
    # Energieffizienzüberwachungsfunktionen
    # ==========================================
    
    def _start_energy_monitoring(self) -> Dict[str, Any]:
        """Startet die Energieüberwachung für einen Trainingsprozess."""
        if not self.energy_efficient or not self.energy_monitor:
            return {}
        
        stats = {
            "start_time": time.time(),
            "cpu_usage": [],
            "memory_usage": [],
            "pauses": 0,
            "total_pause_time": 0.0
        }
        
        # Prüfe, ob psutil verfügbar ist für detailliertere Statistiken
        try:
            import psutil
            stats["psutil_available"] = True
            
            # Aktuelle CPU- und Speichernutzung speichern
            stats["cpu_usage"].append(psutil.cpu_percent(interval=0.1))
            stats["memory_usage"].append(psutil.virtual_memory().percent)
            
            # Batteriestatus prüfen, falls verfügbar
            if hasattr(psutil, 'sensors_battery') and callable(psutil.sensors_battery):
                battery = psutil.sensors_battery()
                if battery:
                    stats["battery_start"] = {
                        "percent": battery.percent,
                        "power_plugged": battery.power_plugged
                    }
            
        except ImportError:
            stats["psutil_available"] = False
            logger.info("psutil nicht verfügbar, verwende eingeschränkte Energieüberwachung")
        
        logger.debug("Energieüberwachung gestartet")
        return stats
    
    def _check_energy_efficiency(self, optimizer=None) -> bool:
        """Prüft und passt die Energieeffizienz während des Trainings an."""
        if not self.energy_efficient or not self.energy_monitor:
            return True  # Wenn nicht aktiviert, erlaube immer Training
        
        # Prüfe nur alle paar Sekunden, um Overhead zu reduzieren
        current_time = time.time()
        time_since_last_check = current_time - self.energy_monitor['last_check']
        if time_since_last_check < 2.0:  # Prüfe maximal alle 2 Sekunden
            return True
        
        self.energy_monitor['last_check'] = current_time
        
        try:
            import psutil
            
            # CPU-Auslastung prüfen
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.energy_monitor['cpu_usage'].append(cpu_percent)
            
            # Speicherauslastung prüfen
            memory_percent = psutil.virtual_memory().percent
            self.energy_monitor['memory_usage'].append(memory_percent)
            
            # Batteriezustand prüfen, falls verfügbar
            battery_critical = False
            if hasattr(psutil, 'sensors_battery') and callable(psutil.sensors_battery):
                battery = psutil.sensors_battery()
                if battery and not battery.power_plugged and battery.percent < 20:
                    battery_critical = True
                    logger.warning(f"Kritischer Batteriestand: {battery.percent}%")
            
            # Ressourcenschwellenwerte
            cpu_threshold = 90  # CPU-Auslastung in Prozent
            memory_threshold = self.memory_limit_mb * 100 / psutil.virtual_memory().total  # Speicherlimit in Prozent
            
            # Passe das Training basierend auf der Ressourcennutzung an
            if cpu_percent > cpu_threshold or memory_percent > memory_threshold or battery_critical:
                # Aktiviere Energiesparmodus
                if optimizer and hasattr(optimizer, 'param_groups'):
                    # Reduziere die Lernrate temporär
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.8  # Reduziere Lernrate um 20%
                
                # Erzwinge Speicherbereinigung, wenn verfügbar
                if self.backend == "torch" and 'torch' in sys.modules:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                logger.info(f"Energiesparmodus aktiviert - CPU: {cpu_percent}%, Memory: {memory_percent}%")
                return False  # Empfehle Pause
            
            return True  # Erlaube Training fortzusetzen
            
        except Exception as e:
            logger.error(f"Fehler bei der Energieüberwachung: {e}")
            return True  # Bei Fehlern Training fortsetzen lassen
    
    def _stop_energy_monitoring(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Beendet die Energieüberwachung und fasst die Ergebnisse zusammen."""
        if not self.energy_efficient or not stats:
            return {}
        
        stats["end_time"] = time.time()
        stats["duration"] = stats["end_time"] - stats["start_time"]
        
        # Berechne durchschnittliche Ressourcennutzung
        if stats.get("cpu_usage"):
            stats["avg_cpu_usage"] = sum(stats["cpu_usage"]) / len(stats["cpu_usage"])
            stats["max_cpu_usage"] = max(stats["cpu_usage"])
        
        if stats.get("memory_usage"):
            stats["avg_memory_usage"] = sum(stats["memory_usage"]) / len(stats["memory_usage"])
            stats["max_memory_usage"] = max(stats["memory_usage"])
        
        # Prüfe Batteriestatus am Ende, falls verfügbar
        try:
            import psutil
            if hasattr(psutil, 'sensors_battery') and callable(psutil.sensors_battery):
                battery = psutil.sensors_battery()
                if battery and "battery_start" in stats:
                    stats["battery_end"] = {
                        "percent": battery.percent,
                        "power_plugged": battery.power_plugged
                    }
                    
                    # Berechne Batterieverbrauch
                    if not battery.power_plugged and not stats["battery_start"]["power_plugged"]:
                        stats["battery_consumption"] = stats["battery_start"]["percent"] - battery.percent
                        stats["battery_consumption_per_hour"] = stats["battery_consumption"] * 3600 / stats["duration"]
        except:
            pass
        
        logger.info(f"Energieüberwachung beendet - Dauer: {stats['duration']:.2f}s")
        if 'avg_cpu_usage' in stats:
            logger.info(f"Durchschnittliche CPU-Auslastung: {stats['avg_cpu_usage']:.1f}%")
        if 'avg_memory_usage' in stats:
            logger.info(f"Durchschnittliche Speicherauslastung: {stats['avg_memory_usage']:.1f}%")
        if 'battery_consumption' in stats:
            logger.info(f"Batterieverbrauch: {stats['battery_consumption']:.1f}%")
        
        return stats
