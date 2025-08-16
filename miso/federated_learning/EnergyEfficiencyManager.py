#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EnergyEfficiencyManager.py
========================

Modul zur Überwachung und Optimierung des Energieverbrauchs beim Training und 
der Inferenz von KI-Modellen. Berücksichtigt Batteriestatus, CPU/GPU/NPU-Auslastung
und thermische Bedingungen, um Trainingsaufgaben energieeffizient zu planen.

Teil der MISO Ultimate AGI - Phase 6 (Federated Learning System)
"""

import os
import sys
import time
import json
import uuid
import logging
import datetime
import threading
import subprocess
import platform
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MISO.EnergyEfficiencyManager')

class EnergyEfficiencyManager:
    """
    Modul zur Überwachung und Optimierung des Energieverbrauchs beim Training.
    
    Features:
    - Echtzeit-Überwachung von CPU/GPU/NPU-Auslastung
    - Batteriestandüberwachung auf Mobilgeräten
    - Optimierung des Trainingszeitplans für energieeffiziente Perioden
    - Anpassung der Trainingslast basierend auf verfügbaren Ressourcen
    - Unterstützung für Apple Silicon (M-Serie) und deren Neural Engine
    - Integration mit der T-Mathematics Engine und deren verschiedenen Backends
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 base_output_dir: Optional[str] = None,
                 monitoring_interval: float = 1.0,
                 power_threshold: float = 0.8,
                 thermal_threshold: float = 0.7,
                 battery_threshold: float = 0.2,
                 enable_monitoring: bool = True,
                 enable_optimizations: bool = True,
                 priority_to_performance: float = 0.5):
        """
        Initialisiert den EnergyEfficiencyManager.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            base_output_dir: Basisverzeichnis für Ausgaben
            monitoring_interval: Intervall für Überwachung in Sekunden
            power_threshold: Schwellenwert für Leistungsaufnahme (0-1)
            thermal_threshold: Schwellenwert für thermische Belastung (0-1)
            battery_threshold: Schwellenwert für Batteriestand (0-1)
            enable_monitoring: Ob Überwachung aktiviert werden soll
            enable_optimizations: Ob Optimierungen aktiviert werden sollen
            priority_to_performance: Gewichtung von Performance vs. Energieverbrauch (0-1)
        """
        # Generiere eine eindeutige Session-ID
        self.session_id = str(uuid.uuid4())
        logger.info(f"Initialisiere EnergyEfficiencyManager mit Session-ID: {self.session_id}")
        
        # Lade Konfiguration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Setze Basisparameter
        self.base_output_dir = base_output_dir or self.config.get('output_dir', './output')
        self.monitoring_interval = monitoring_interval
        self.power_threshold = power_threshold
        self.thermal_threshold = thermal_threshold
        self.battery_threshold = battery_threshold
        self.enable_monitoring = enable_monitoring
        self.enable_optimizations = enable_optimizations
        self.priority_to_performance = priority_to_performance
        
        # Erstelle die erforderlichen Verzeichnisse
        self._setup_directories()
        
        # Erkenne System und Hardware
        self.system_info = self._detect_system()
        logger.info(f"System erkannt: {self.system_info['os']} auf {self.system_info['architecture']}")
        
        # Initialisiere Monitoring-Variablen
        self.energy_stats = {
            "cpu_usage": [],
            "gpu_usage": [],
            "memory_usage": [],
            "temperature": [],
            "power_consumption": [],
            "battery_level": [],
            "timestamp": []
        }
        
        # Status-Variablen
        self.is_monitoring = False
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Optimierungsplaner
        self.optimization_schedule = {
            "optimal_times": [],
            "last_updated": time.time(),
            "training_jobs": []
        }
        
        # Starte Monitoring, wenn aktiviert
        if self.enable_monitoring:
            self.start_monitoring()
        
        logger.info("EnergyEfficiencyManager erfolgreich initialisiert")
    
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
            self.logs_dir = os.path.join(self.base_output_dir, 'energy_logs')
            self.stats_dir = os.path.join(self.base_output_dir, 'energy_stats')
            
            os.makedirs(self.logs_dir, exist_ok=True)
            os.makedirs(self.stats_dir, exist_ok=True)
            
            logger.info(f"Verzeichnisstruktur unter {self.base_output_dir} eingerichtet")
        except Exception as e:
            logger.error(f"Fehler beim Einrichten der Verzeichnisse: {e}")
            raise
    
    def _detect_system(self) -> Dict[str, Any]:
        """Erkennt das Betriebssystem und die verfügbare Hardware."""
        system_info = {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "has_gpu": False,
            "has_neural_engine": False,
            "gpu_type": "none",
            "has_battery": False
        }
        
        # Erkennung für macOS
        if system_info["os"] == "Darwin":
            # Erkenne Apple Silicon
            if system_info["architecture"] == "arm64":
                system_info["is_apple_silicon"] = True
                system_info["has_neural_engine"] = True
                system_info["gpu_type"] = "apple_integrated"
                
                # Versuche, das genaue Modell zu bestimmen
                try:
                    model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                    if "M1" in model:
                        system_info["chip"] = "M1"
                    elif "M2" in model:
                        system_info["chip"] = "M2"
                    elif "M3" in model:
                        system_info["chip"] = "M3"
                    elif "M4" in model:
                        system_info["chip"] = "M4"
                    else:
                        system_info["chip"] = "unknown_apple_silicon"
                except:
                    system_info["chip"] = "unknown_apple_silicon"
            else:
                system_info["is_apple_silicon"] = False
                
                # Prüfe auf externe GPU
                try:
                    gpu_info = subprocess.check_output(["system_profiler", "SPDisplaysDataType"]).decode()
                    if "NVIDIA" in gpu_info:
                        system_info["gpu_type"] = "nvidia"
                        system_info["has_gpu"] = True
                    elif "AMD" in gpu_info:
                        system_info["gpu_type"] = "amd"
                        system_info["has_gpu"] = True
                    elif "Intel" in gpu_info:
                        system_info["gpu_type"] = "intel_integrated"
                        system_info["has_gpu"] = True
                except:
                    pass
            
            # Prüfe auf Batterie
            try:
                power_info = subprocess.check_output(["pmset", "-g", "batt"]).decode()
                system_info["has_battery"] = "InternalBattery" in power_info
            except:
                pass
        
        # Erkennung für Linux
        elif system_info["os"] == "Linux":
            # Prüfe auf NVIDIA GPU
            try:
                if subprocess.call(["which", "nvidia-smi"]) == 0:
                    system_info["gpu_type"] = "nvidia"
                    system_info["has_gpu"] = True
            except:
                pass
            
            # Prüfe auf AMD GPU
            try:
                if os.path.exists("/sys/class/drm/card0/device/vendor"):
                    with open("/sys/class/drm/card0/device/vendor", "r") as f:
                        vendor = f.read().strip()
                        if vendor == "0x1002":  # AMD Vendor ID
                            system_info["gpu_type"] = "amd"
                            system_info["has_gpu"] = True
            except:
                pass
            
            # Prüfe auf Batterie
            system_info["has_battery"] = os.path.exists("/sys/class/power_supply/BAT0")
        
        # Erkennung für Windows
        elif system_info["os"] == "Windows":
            # Windows-spezifische Erkennung wäre hier zu implementieren
            pass
        
        # Prüfe auf vorhandene ML-Frameworks
        system_info["ml_frameworks"] = []
        
        # Prüfe auf PyTorch
        try:
            import torch
            system_info["ml_frameworks"].append("pytorch")
            system_info["pytorch_version"] = torch.__version__
            
            system_info["has_cuda"] = torch.cuda.is_available()
            system_info["has_mps"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except ImportError:
            pass
        
        # Prüfe auf TensorFlow
        try:
            import tensorflow as tf
            system_info["ml_frameworks"].append("tensorflow")
            system_info["tensorflow_version"] = tf.__version__
            
            system_info["has_tf_gpu"] = tf.test.is_gpu_available() if hasattr(tf.test, 'is_gpu_available') else tf.config.list_physical_devices('GPU')
        except ImportError:
            pass
        
        # Prüfe auf MLX (Apple)
        try:
            import mlx
            system_info["ml_frameworks"].append("mlx")
            system_info["mlx_version"] = mlx.__version__ if hasattr(mlx, "__version__") else "unknown"
        except ImportError:
            pass
        
        # Prüfe auf T-Mathematics Engine (MISO)
        try:
            # Versuche, die MISOTensor-Klasse zu importieren
            t_math_found = False
            for path in sys.path:
                tensor_path = os.path.join(path, "miso", "math", "tensor.py")
                if os.path.exists(tensor_path):
                    t_math_found = True
                    break
            
            if t_math_found:
                system_info["ml_frameworks"].append("t_mathematics")
                system_info["has_miso_tensor"] = True
        except:
            pass
        
        return system_info
    
    def start_monitoring(self) -> bool:
        """Startet die Überwachung des Energieverbrauchs in einem eigenen Thread."""
        if self.is_monitoring:
            logger.warning("Energieüberwachung läuft bereits")
            return False
        
        try:
            # Setze Stop-Event zurück
            self.stop_monitoring.clear()
            
            # Starte Monitoring-Thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.is_monitoring = True
            logger.info("Energieüberwachung gestartet")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Starten der Energieüberwachung: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stoppt die Überwachung des Energieverbrauchs."""
        if not self.is_monitoring:
            logger.warning("Energieüberwachung läuft nicht")
            return False
        
        try:
            # Signalisiere dem Thread, dass er stoppen soll
            self.stop_monitoring.set()
            
            # Warte auf das Ende des Threads mit Timeout
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=3.0)
            
            self.is_monitoring = False
            logger.info("Energieüberwachung gestoppt")
            
            # Speichere die gesammelten Daten
            self._save_energy_stats()
            
            return True
        except Exception as e:
            logger.error(f"Fehler beim Stoppen der Energieüberwachung: {e}")
            return False
    
    def _monitoring_loop(self) -> None:
        """Hauptschleife für die Überwachung des Energieverbrauchs."""
        try:
            # Initialisiere Überwachungstools basierend auf dem Betriebssystem
            monitoring_tools = self._initialize_monitoring_tools()
            
            # Starte die Überwachungsschleife
            while not self.stop_monitoring.is_set():
                # Erfasse CPU-Nutzung, GPU-Nutzung, Temperatur, Batterie, etc.
                current_stats = self._collect_current_stats(monitoring_tools)
                
                # Speichere Statistiken
                for key, value in current_stats.items():
                    if key in self.energy_stats:
                        self.energy_stats[key].append(value)
                
                # Zeitstempel hinzufügen
                self.energy_stats["timestamp"].append(time.time())
                
                # Überprüfe auf Optimierungsbedarf
                if self.enable_optimizations:
                    self._check_optimization_triggers(current_stats)
                
                # Warte für das nächste Sampling-Intervall
                time.sleep(self.monitoring_interval)
                
                # Begrenze die Größe der Statistiken
                self._trim_stats_if_needed()
        
        except Exception as e:
            logger.error(f"Fehler in der Überwachungsschleife: {e}")
            self.is_monitoring = False
    
    def _initialize_monitoring_tools(self) -> Dict[str, Any]:
        """Initialisiert die benötigten Tools für die Systemüberwachung."""
        tools = {"available": []}
        
        # Prüfe auf psutil für grundlegende System-Überwachung
        try:
            import psutil
            tools["psutil"] = psutil
            tools["available"].append("psutil")
        except ImportError:
            logger.warning("psutil nicht verfügbar. Eingeschränkte Überwachung.")
        
        # Betriebssystemspezifische Tools
        if self.system_info["os"] == "Darwin":  # macOS
            # Prüfe auf powermetrics für Apple-Geräte
            try:
                # Teste, ob powermetrics verfügbar ist
                subprocess.run(["powermetrics", "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                tools["powermetrics_available"] = True
                tools["available"].append("powermetrics")
            except (subprocess.SubprocessError, FileNotFoundError):
                tools["powermetrics_available"] = False
            
            # Für Apple Silicon: MLX und ANE-Überwachung
            if self.system_info.get("is_apple_silicon", False):
                tools["platform"] = "apple_silicon"
                # Prüfen, ob wir ANE-Nutzung überwachen können
                try:
                    import ctypes
                    ane_lib_path = "/System/Library/PrivateFrameworks/ANEServices.framework/ANEServices"
                    if os.path.exists(ane_lib_path):
                        tools["ane_lib_available"] = True
                        tools["available"].append("ane_lib")
                except ImportError:
                    tools["ane_lib_available"] = False
        
        elif self.system_info["os"] == "Linux":  # Linux
            # Prüfe auf NVIDIA GPU
            if self.system_info.get("gpu_type") == "nvidia":
                try:
                    # Prüfe, ob nvidia-smi verfügbar ist
                    subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    tools["nvidia_smi_available"] = True
                    tools["available"].append("nvidia_smi")
                except (subprocess.SubprocessError, FileNotFoundError):
                    tools["nvidia_smi_available"] = False
        
        return tools
    
    def _collect_current_stats(self, monitoring_tools: Dict[str, Any]) -> Dict[str, float]:
        """Sammelt aktuelle Systemstatistiken mit Hilfe der verfügbaren Monitoring-Tools."""
        stats = {
            "cpu_usage": 0.0,
            "gpu_usage": 0.0,
            "memory_usage": 0.0,
            "temperature": 0.0,
            "power_consumption": 0.0,
            "battery_level": 1.0  # Default: 100%
        }
        
        # Verwende psutil für grundlegende Statistiken, falls verfügbar
        if "psutil" in monitoring_tools["available"]:
            psutil = monitoring_tools["psutil"]
            
            # CPU-Nutzung
            stats["cpu_usage"] = psutil.cpu_percent(interval=0.1) / 100.0
            
            # Arbeitsspeicher-Nutzung
            memory = psutil.virtual_memory()
            stats["memory_usage"] = memory.percent / 100.0
            
            # Batterieinformationen, falls verfügbar
            if hasattr(psutil, "sensors_battery") and callable(psutil.sensors_battery):
                battery = psutil.sensors_battery()
                if battery:
                    stats["battery_level"] = battery.percent / 100.0
            
            # Temperaturinformationen, falls verfügbar
            if hasattr(psutil, "sensors_temperatures") and callable(psutil.sensors_temperatures):
                temperatures = psutil.sensors_temperatures()
                if temperatures:
                    # Durchschnitt aller Temperaturwerte berechnen
                    all_temps = []
                    for name, entries in temperatures.items():
                        for entry in entries:
                            if hasattr(entry, "current") and entry.current:
                                all_temps.append(entry.current)
                    
                    if all_temps:
                        avg_temp = sum(all_temps) / len(all_temps)
                        # Normalisieren auf 0-1 Skala (angenommen 100°C ist Maximum)
                        stats["temperature"] = min(1.0, max(0.0, avg_temp / 100.0))
        
        # Spezifische Tools für unterschiedliche Plattformen
        
        # Apple Silicon mit MLX und ANE
        if self.system_info.get("is_apple_silicon", False):
            # GPU/ANE-Nutzung auf Apple Silicon
            if "powermetrics_available" in monitoring_tools and monitoring_tools["powermetrics_available"]:
                try:
                    # Verwende powermetrics für Energie- und GPU-Informationen
                    # Hinweis: Dies erfordert Root-Rechte
                    cmd = ["sudo", "powermetrics", "-n", "1", "-i", "100", "--samplers", "cpu_power,gpu_power", "-f", "json"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1.0, check=False)
                    
                    if result.returncode == 0 and result.stdout:
                        try:
                            data = json.loads(result.stdout)
                            
                            # GPU-Nutzung
                            if "gpu_power" in data:
                                gpu_power = data["gpu_power"]
                                if "gpu_power_mW" in gpu_power:
                                    # Annahme: 20W als Maximum bei voller Leistung
                                    stats["gpu_usage"] = min(1.0, gpu_power["gpu_power_mW"] / 20000.0)
                            
                            # Gesamtleistungsaufnahme
                            if "processor" in data and "CPU_power_mW" in data["processor"]:
                                cpu_power = data["processor"]["CPU_power_mW"]
                                # Annahme: 30W als Maximum bei voller Leistung
                                stats["power_consumption"] = min(1.0, cpu_power / 30000.0)
                        except json.JSONDecodeError:
                            pass
                except:
                    # Powermetrics braucht Sudo-Rechte, kann also fehlschlagen
                    pass
            
            # Prüfen, ob MLX verwendet wird (optimiert für Apple Silicon)
            try:
                import mlx.core as mx
                if mx.is_available():
                    stats["has_mlx"] = True
                    stats["mlx_available"] = True
            except ImportError:
                stats["has_mlx"] = False
                stats["mlx_available"] = False
            
            # T-Mathematics Engine MLXTensor nutzt Apple Neural Engine optimal
            try:
                # Versuche, auf MISOTensor oder MLXTensor zuzugreifen
                # Dies ist ein Check, ob die T-Mathematics Engine verfügbar ist
                miso_tensor_available = False
                for path in sys.path:
                    tensor_path = os.path.join(path, "miso", "math", "tensor.py")
                    if os.path.exists(tensor_path):
                        miso_tensor_available = True
                        break
                
                stats["miso_tensor_available"] = miso_tensor_available
            except:
                stats["miso_tensor_available"] = False
        
        # NVIDIA GPU unter Linux oder Windows
        elif self.system_info.get("gpu_type") == "nvidia" and "nvidia_smi_available" in monitoring_tools and monitoring_tools["nvidia_smi_available"]:
            try:
                # GPU-Nutzung mit nvidia-smi ermitteln
                result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], 
                                        capture_output=True, text=True, timeout=1.0, check=False)
                
                if result.returncode == 0 and result.stdout:
                    # Parse Ausgabe (Format: Utilization, Temperature, Power)
                    values = result.stdout.strip().split(",")
                    if len(values) >= 3:
                        stats["gpu_usage"] = float(values[0].strip()) / 100.0
                        stats["temperature"] = float(values[1].strip()) / 100.0
                        # Annahme: Max Power = 300W
                        stats["power_consumption"] = min(1.0, float(values[2].strip()) / 300.0)
            except:
                pass
        
        # PyTorch mit MPS auf Apple-Geräten
        if "pytorch" in self.system_info.get("ml_frameworks", []) and self.system_info.get("has_mps", False):
            try:
                import torch
                if torch.backends.mps.is_available():
                    stats["has_mps"] = True
                    # Hier können wir zukünftig spezifische MPS-Metriken erfassen
                    # Aktuell bietet PyTorch keine direkte API für MPS-Nutzungsstatistiken
            except ImportError:
                pass
        
        return stats
    
    def _check_optimization_triggers(self, current_stats: Dict[str, float]) -> None:
        """Prüft, ob Optimierungsmaßnahmen basierend auf den aktuellen Statistiken erforderlich sind."""
        # Prüfe alle Trigger, die eine Optimierung erforderlich machen könnten
        triggers = {
            "high_cpu": current_stats["cpu_usage"] > self.power_threshold,
            "high_gpu": current_stats["gpu_usage"] > self.power_threshold,
            "high_temp": current_stats["temperature"] > self.thermal_threshold,
            "low_battery": current_stats["battery_level"] < self.battery_threshold
        }
        
        # Wenn mindestens ein Trigger aktiv ist, plane Optimierungen
        if any(triggers.values()):
            self._plan_optimizations(triggers, current_stats)
    
    def _plan_optimizations(self, triggers: Dict[str, bool], current_stats: Dict[str, float]) -> None:
        """Plant Optimierungsmaßnahmen basierend auf aktivierten Triggern."""
        # Aktualisiere Optimierungsplan nur alle 30 Sekunden
        if time.time() - self.optimization_schedule["last_updated"] < 30:
            return
        
        self.optimization_schedule["last_updated"] = time.time()
        
        # Bestimme Optimierungsstrategien basierend auf Triggern
        strategies = []
        
        
        if triggers["low_battery"]:
            strategies.append({
                "type": "reduce_batch_size",
                "priority": "high",
                "reason": "Niedriger Batteriestand",
                "suggestion": "Reduziere Batch-Größe um 50%"
            })
            strategies.append({
                "type": "use_quantization",
                "priority": "high",
                "reason": "Niedriger Batteriestand",
                "suggestion": "Aktiviere int8-Quantisierung"
            })
        
        if triggers["high_temp"]:
            strategies.append({
                "type": "reduce_frequency",
                "priority": "high",
                "reason": "Hohe Systemtemperatur",
                "suggestion": "Pausiere Training für 5 Minuten"
            })
        
        if triggers["high_cpu"] or triggers["high_gpu"]:
            # Wenn T-Mathematics Engine verfügbar ist und wir auf Apple Silicon sind
            if current_stats.get("miso_tensor_available", False) and self.system_info.get("is_apple_silicon", False):
                strategies.append({
                    "type": "switch_backend",
                    "priority": "medium",
                    "reason": "Hohe CPU/GPU-Auslastung",
                    "suggestion": "Wechsle zu MLXTensor für bessere Energieeffizienz"
                })
            # Wenn PyTorch mit MPS verfügbar ist
            elif current_stats.get("has_mps", False):
                strategies.append({
                    "type": "switch_backend",
                    "priority": "medium",
                    "reason": "Hohe CPU/GPU-Auslastung",
                    "suggestion": "Verwende TorchTensor mit MPS für bessere Energieeffizienz"
                })
        
        # Füge Strategien zum Optimierungsplan hinzu
        self.optimization_schedule["strategies"] = strategies
        
        # Logge die geplanten Optimierungen
        if strategies:
            logger.info(f"Energieoptimierungsstrategien geplant: {len(strategies)}")
            for strategy in strategies:
                logger.info(f"  - {strategy['type']} ({strategy['priority']}): {strategy['suggestion']}")
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Gibt die aktuellen Optimierungsempfehlungen zurück."""
        # Aktualisiere Empfehlungen, falls notwendig
        if not self.is_monitoring:
            # Wenn keine aktive Überwachung läuft, führe eine einmalige Messung durch
            tools = self._initialize_monitoring_tools()
            current_stats = self._collect_current_stats(tools)
            self._check_optimization_triggers(current_stats)
        
        return self.optimization_schedule.get("strategies", [])
    
    def optimize_backend_selection(self, task_type: str = "training") -> str:
        """Wählt das optimale Backend für die gegebene Aufgabe basierend auf Energieeffizienz."""
        # Standardempfehlung basierend auf Systeminfo
        if self.system_info.get("is_apple_silicon", False):
            # Auf Apple Silicon ist MLX optimal für Energieeffizienz
            if "mlx" in self.system_info.get("ml_frameworks", []):
                preferred_backend = "mlx"
                tensor_type = "MLXTensor"  # T-Mathematics Engine optimiert für Apple Silicon
            # Alternativ PyTorch mit MPS
            elif self.system_info.get("has_mps", False):
                preferred_backend = "pytorch_mps"
                tensor_type = "TorchTensor"  # T-Mathematics Engine mit MPS
            else:
                preferred_backend = "numpy"
                tensor_type = "MISOTensor"  # Generische Fallback-Implementation
        
        # NVIDIA GPUs bevorzugen PyTorch mit CUDA
        elif self.system_info.get("has_cuda", False):
            preferred_backend = "pytorch_cuda"
            tensor_type = "TorchTensor"
        
        # Fallback auf CPU
        else:
            preferred_backend = "pytorch_cpu"
            tensor_type = "TorchTensor"
        
        # Anpassung basierend auf Batteriestand und Energieverbrauch
        if self.is_monitoring and len(self.energy_stats["battery_level"]) > 0:
            current_battery = self.energy_stats["battery_level"][-1]
            
            # Bei niedrigem Batteriestand: Energieeffizienz priorisieren
            if current_battery < self.battery_threshold:
                if self.system_info.get("is_apple_silicon", False) and "mlx" in self.system_info.get("ml_frameworks", []):
                    # MLX ist am energieeffizientesten auf Apple Silicon
                    preferred_backend = "mlx"
                    tensor_type = "MLXTensor"
                else:
                    # Fallback auf CPU bei niedrigem Batteriestand
                    preferred_backend = "numpy"
                    tensor_type = "MISOTensor"
        
        logger.info(f"Optimales Backend für {task_type}: {preferred_backend} mit {tensor_type}")
        return preferred_backend
    
    def _trim_stats_if_needed(self, max_entries: int = 1000) -> None:
        """Begrenzt die Größe der gespeicherten Statistiken."""
        for key in self.energy_stats:
            if len(self.energy_stats[key]) > max_entries:
                # Behalte nur die neuesten Einträge
                self.energy_stats[key] = self.energy_stats[key][-max_entries:]
    
    def _save_energy_stats(self) -> None:
        """Speichert die gesammelten Energiestatistiken in einer Datei."""
        if not any(self.energy_stats.values()):
            logger.warning("Keine Energiestatistiken zum Speichern vorhanden")
            return
        
        try:
            # Erstelle eine strukturierte Ausgabe aller Statistiken
            output_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": self.session_id,
                "system_info": self.system_info,
                "stats": {}
            }
            
            # Strukturiere die Zeitreihen-Daten
            num_samples = len(self.energy_stats["timestamp"])
            for key in self.energy_stats:
                if key == "timestamp":
                    # Konvertiere Unix-Timestamps in ISO-Format
                    output_data["stats"][key] = [
                        datetime.datetime.fromtimestamp(ts).isoformat()
                        for ts in self.energy_stats[key][-min(num_samples, 1000):]
                    ]
                else:
                    # Begrenze auf die letzten 1000 Einträge, um JSON nicht zu groß zu machen
                    output_data["stats"][key] = self.energy_stats[key][-min(num_samples, 1000):]
            
            # Erstelle einen sprechenden Dateinamen mit Timestamp
            filename = f"energy_stats_{self.session_id}_{int(time.time())}.json"
            file_path = os.path.join(self.stats_dir, filename)
            
            # Speichere als JSON
            with open(file_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Energiestatistiken gespeichert unter {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Energiestatistiken: {e}")
            return None
    
    def recommend_training_schedule(self, task_size: str = "medium", priority: str = "normal") -> Dict[str, Any]:
        """Empfiehlt einen optimalen Zeitplan für Trainingsaufgaben basierend auf Energieverbrauch und Ressourcenverfügbarkeit.
        
        Args:
            task_size: Größe der Aufgabe (small, medium, large)
            priority: Priorität der Aufgabe (low, normal, high)
            
        Returns:
            Empfehlungen für Zeitplan, Batch-Größe, etc.
        """
        # Wenn möglich, verwende aktuelle Energiestatistiken
        if self.is_monitoring and any(self.energy_stats["cpu_usage"]):
            current_cpu = self.energy_stats["cpu_usage"][-1]
            current_gpu = self.energy_stats["gpu_usage"][-1]
            current_temp = self.energy_stats["temperature"][-1]
            current_battery = self.energy_stats["battery_level"][-1] if self.energy_stats["battery_level"] else 1.0
        else:
            # Einmalige Messung, wenn kein Monitoring läuft
            tools = self._initialize_monitoring_tools()
            stats = self._collect_current_stats(tools)
            current_cpu = stats["cpu_usage"]
            current_gpu = stats["gpu_usage"]
            current_temp = stats["temperature"]
            current_battery = stats["battery_level"]
        
        # Standardwerte für verschiedene Aufgabengrößen
        task_configs = {
            "small": {"batch_size": 8, "max_time": 10, "interval": 60},
            "medium": {"batch_size": 16, "max_time": 20, "interval": 120},
            "large": {"batch_size": 32, "max_time": 45, "interval": 300}
        }
        
        # Basisempfehlung abhängig von Aufgabengröße
        base_config = task_configs.get(task_size, task_configs["medium"])
        
        # Anpassungen basierend auf Systemauslastung
        adjusted_config = base_config.copy()
        
        # Reduziere Batch-Größe bei hoher Auslastung oder hoher Temperatur
        if current_cpu > 0.7 or current_gpu > 0.7 or current_temp > 0.8:
            adjusted_config["batch_size"] = max(1, base_config["batch_size"] // 2)
            adjusted_config["max_time"] = base_config["max_time"] // 2
        
        # Bei niedrigem Batteriestand (unter 30%): erheblich reduzieren
        if current_battery < 0.3:
            adjusted_config["batch_size"] = max(1, base_config["batch_size"] // 4)
            adjusted_config["max_time"] = base_config["max_time"] // 3
            adjusted_config["interval"] = base_config["interval"] * 2
        
        # Bei hoher Priorität: kompromisslos
        if priority == "high":
            adjusted_config = base_config.copy()
            # Nur bei extremer Temperatur einschränken
            if current_temp > 0.9:
                adjusted_config["batch_size"] = max(1, base_config["batch_size"] // 2)
        
        # Optimaler Zeitpunkt
        best_time = "now"
        if current_cpu > 0.8 and priority != "high":
            best_time = "wait_for_idle"
        
        # Füge Empfehlungen hinzu
        recommendation = {
            "best_time": best_time,
            "batch_size": adjusted_config["batch_size"],
            "max_training_minutes": adjusted_config["max_time"],
            "check_interval_seconds": adjusted_config["interval"],
            "backend": self.optimize_backend_selection("training"),
            "reason": self._generate_recommendation_reason(current_cpu, current_gpu, current_temp, current_battery)
        }
        
        logger.info(f"Trainingsplan empfohlen: {recommendation}")
        return recommendation
    
    def _generate_recommendation_reason(self, cpu_usage, gpu_usage, temperature, battery_level) -> str:
        """Erzeugt eine Begründung für die Trainingsempfehlung basierend auf Systemzustand."""
        reasons = []
        
        if battery_level < 0.2:
            reasons.append("Kritisch niedriger Batteriestand")
        elif battery_level < 0.4:
            reasons.append("Niedriger Batteriestand")
        
        if temperature > 0.8:
            reasons.append("Hohe Systemtemperatur")
        
        if cpu_usage > 0.8:
            reasons.append("Hohe CPU-Auslastung")
        
        if gpu_usage > 0.8:
            reasons.append("Hohe GPU-Auslastung")
        
        if not reasons:
            return "Optimale Systembedingungen"
        
        return ", ".join(reasons)
    
    def adjust_learning_rate(self, base_lr: float) -> float:
        """Passt die Lernrate basierend auf Energieverfügbarkeit und Systemzustand an.
        
        Args:
            base_lr: Die Basis-Lernrate
            
        Returns:
            Angepasste Lernrate
        """
        # Default-Anpassungsfaktor
        adjustment = 1.0
        
        # Wenn Monitoring aktiv ist, passe basierend auf aktuellen Werten an
        if self.is_monitoring and any(self.energy_stats["battery_level"]):
            current_battery = self.energy_stats["battery_level"][-1]
            current_temp = self.energy_stats["temperature"][-1]
            
            # Reduziere Lernrate bei niedrigem Batteriestand (mehr Schritte, weniger Aufwand pro Schritt)
            if current_battery < 0.3:
                adjustment *= 0.5
            elif current_battery < 0.5:
                adjustment *= 0.8
            
            # Reduziere bei hoher Temperatur
            if current_temp > 0.8:
                adjustment *= 0.7
        
        adjusted_lr = base_lr * adjustment
        if adjustment != 1.0:
            logger.info(f"Lernrate angepasst: {base_lr:.6f} -> {adjusted_lr:.6f} (Faktor: {adjustment:.2f})")
        
        return adjusted_lr
    
    def should_defer_training(self) -> Tuple[bool, str]:
        """Prüft, ob das Training aufgrund von Energiebedingungen verschoben werden sollte.
        
        Returns:
            Tuple aus (verschieben_ja_nein, grund)
        """
        if not self.is_monitoring or not self.enable_optimizations:
            return False, "Energieoptimierung deaktiviert"
        
        defer = False
        reason = "Optimale Bedingungen"
        
        # Prüfe Batteriestand (falls verfügbar)
        if self.energy_stats["battery_level"] and len(self.energy_stats["battery_level"]) > 0:
            current_battery = self.energy_stats["battery_level"][-1]
            if current_battery < 0.15:  # Kritisch niedriger Batteriestand
                defer = True
                reason = f"Kritisch niedriger Batteriestand ({current_battery*100:.1f}%)"
        
        # Prüfe Temperatur (falls verfügbar)
        if self.energy_stats["temperature"] and len(self.energy_stats["temperature"]) > 0:
            current_temp = self.energy_stats["temperature"][-1]
            # Normalisierte Temperatur (0-1), wobei 1 = 100°C
            if current_temp > 0.9:  # Gefährlich hohe Temperatur
                defer = True
                if reason == "Optimale Bedingungen":
                    reason = f"Gefährlich hohe Systemtemperatur ({current_temp*100:.1f}°C)"
                else:
                    reason += f" und gefährlich hohe Systemtemperatur ({current_temp*100:.1f}°C)"
        
        if defer:
            logger.warning(f"Training wird verschoben: {reason}")
        
        return defer, reason
    
    def calculate_energy_usage(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Berechnet den Energieverbrauch für einen Zeitraum basierend auf gesammelten Statistiken.
        
        Args:
            start_time: Startzeitstempel (Unix-Zeit)
            end_time: Endzeitstempel (Unix-Zeit)
            
        Returns:
            Statistiken zum Energieverbrauch
        """
        if not self.is_monitoring or not self.energy_stats["timestamp"]:
            return {
                "duration_seconds": end_time - start_time,
                "estimated": True,
                "avg_power_consumption": 0.0,
                "avg_cpu_usage": 0.0,
                "avg_gpu_usage": 0.0
            }
        
        # Finde Messwerte im angegebenen Zeitraum
        indices = [i for i, ts in enumerate(self.energy_stats["timestamp"]) if start_time <= ts <= end_time]
        
        if not indices:
            return {
                "duration_seconds": end_time - start_time,
                "estimated": True,
                "avg_power_consumption": 0.0, 
                "avg_cpu_usage": 0.0,
                "avg_gpu_usage": 0.0
            }
        
        # Sammle relevante Statistiken
        avg_power = np.mean([self.energy_stats["power_consumption"][i] for i in indices]) if self.energy_stats["power_consumption"] else 0.0
        avg_cpu = np.mean([self.energy_stats["cpu_usage"][i] for i in indices]) if self.energy_stats["cpu_usage"] else 0.0
        avg_gpu = np.mean([self.energy_stats["gpu_usage"][i] for i in indices]) if self.energy_stats["gpu_usage"] else 0.0
        avg_temp = np.mean([self.energy_stats["temperature"][i] for i in indices]) if self.energy_stats["temperature"] else 0.0
        
        # Batterieänderung berechnen, falls verfügbar
        battery_change = 0.0
        if self.energy_stats["battery_level"]:
            valid_indices = [i for i in indices if i < len(self.energy_stats["battery_level"])]
            if valid_indices and len(valid_indices) >= 2:
                first = self.energy_stats["battery_level"][valid_indices[0]]
                last = self.energy_stats["battery_level"][valid_indices[-1]]
                battery_change = (last - first) * 100  # In Prozent
        
        # Energieverbrauch in Joule abschätzen (grobe Annäherung)
        # Annahme: 0.0 = 0W, 1.0 = max_power (z.B. 100W)
        max_power_estimate = 100.0  # Watt
        duration = end_time - start_time  # Sekunden
        energy_joules = avg_power * max_power_estimate * duration
        
        return {
            "duration_seconds": duration,
            "estimated": True,
            "avg_power_consumption": avg_power,
            "avg_cpu_usage": avg_cpu,
            "avg_gpu_usage": avg_gpu,
            "avg_temperature": avg_temp,
            "battery_change_percent": battery_change,
            "estimated_energy_joules": energy_joules
        }
    
    def check_and_apply_device_constraints(self, model_parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prüft Gerätebeschränkungen und passt Modellparameter entsprechend an.
        
        Args:
            model_parameters: Aktuelle Modellparameter (Größe, Komplexität, etc.)
            
        Returns:
            Angepasste Modellparameter, die Gerätebeschränkungen berücksichtigen
        """
        if not model_parameters:
            model_parameters = {}
        
        adjustments = {}
        constraints = self.get_device_constraints()
        
        # Standard-Basisparameter, falls nicht angegeben
        batch_size = model_parameters.get("batch_size", 16)
        learning_rate = model_parameters.get("learning_rate", 0.001)
        precision = model_parameters.get("precision", "float32")
        
        # Anpassungen basierend auf Speicherbeschränkungen
        if constraints.get("memory_limited", False):
            adjustments["memory_limit_applied"] = True
            # Batch-Größe reduzieren
            adjusted_batch_size = max(1, batch_size // 2)
            adjustments["batch_size"] = adjusted_batch_size
            
            # Präzision reduzieren, falls noch nicht getan
            if precision == "float32":
                adjustments["precision"] = "float16"
        
        # Anpassungen basierend auf Batteriestand
        if constraints.get("battery_limited", False):
            adjustments["battery_limit_applied"] = True
            # Lernrate reduzieren, um Energieverbrauch zu senken
            adjusted_lr = learning_rate * 0.5
            adjustments["learning_rate"] = adjusted_lr
            
            # Wenn noch nicht durch Speicherbeschränkung angepasst, Batch-Größe reduzieren
            if "batch_size" not in adjustments:
                adjusted_batch_size = max(1, batch_size // 4)
                adjustments["batch_size"] = adjusted_batch_size
        
        # Anpassungen basierend auf thermischen Beschränkungen
        if constraints.get("thermal_limited", False):
            adjustments["thermal_limit_applied"] = True
            # Wenn noch nicht angepasst, Batch-Größe reduzieren
            if "batch_size" not in adjustments:
                adjusted_batch_size = max(1, batch_size // 2)
                adjustments["batch_size"] = adjusted_batch_size
        
        # Systemspezifische Optimierungen je nach Backend
        backend_optimization = self.optimize_backend_selection()
        adjustments["recommended_backend"] = backend_optimization
        
        if adjustments:
            logger.info(f"Modellparameter angepasst für Gerätebeschränkungen: {adjustments}")
        
        # Ursprüngliche Parameter mit Anpassungen aktualisieren
        result = model_parameters.copy()
        result.update(adjustments)
        
        return result
    
    def get_device_constraints(self) -> Dict[str, bool]:
        """Ermittelt aktuelle Gerätebeschränkungen basierend auf Systemzustand.
        
        Returns:
            Wörterbuch mit erkannten Beschränkungen
        """
        constraints = {
            "memory_limited": False,
            "battery_limited": False,
            "thermal_limited": False,
            "compute_limited": False
        }
        
        # Prüfe Batteriestand (falls verfügbar)
        if self.is_monitoring and self.energy_stats["battery_level"] and len(self.energy_stats["battery_level"]) > 0:
            current_battery = self.energy_stats["battery_level"][-1]
            if current_battery < self.battery_threshold:
                constraints["battery_limited"] = True
        else:
            # Prüfe über Systemtools direkt
            try:
                import psutil
                if hasattr(psutil, "sensors_battery") and callable(psutil.sensors_battery):
                    battery = psutil.sensors_battery()
                    if battery and battery.percent < self.battery_threshold * 100:
                        constraints["battery_limited"] = True
            except ImportError:
                pass
        
        # Prüfe Temperatur (falls verfügbar)
        if self.is_monitoring and self.energy_stats["temperature"] and len(self.energy_stats["temperature"]) > 0:
            current_temp = self.energy_stats["temperature"][-1]
            if current_temp > self.thermal_threshold:
                constraints["thermal_limited"] = True
        
        # Prüfe Speicherverfügbarkeit
        try:
            import psutil
            memory = psutil.virtual_memory()
            # Wenn weniger als 25% Speicher verfügbar
            if memory.available / memory.total < 0.25:
                constraints["memory_limited"] = True
        except ImportError:
            pass
        
        # Prüfe CPU-Auslastung
        if self.is_monitoring and self.energy_stats["cpu_usage"] and len(self.energy_stats["cpu_usage"]) > 0:
            current_cpu = self.energy_stats["cpu_usage"][-1]
            if current_cpu > 0.8:  # Hohe CPU-Auslastung
                constraints["compute_limited"] = True
        
        return constraints
    
    # Integration mit Federated Learning und T-Mathematics Engine
    
    def integrate_with_local_trainer(self, trainer) -> None:
        """Integriert den EnergyEfficiencyManager mit einem LocalSelfTrainer.
        
        Args:
            trainer: Eine Instanz von LocalSelfTrainer
        """
        if not hasattr(trainer, "energy_manager"):
            trainer.energy_manager = self
            logger.info(f"EnergyEfficiencyManager mit LocalSelfTrainer integriert")
            
            # Methodendefinitionen für Trainer, wenn sie noch nicht existieren
            if not hasattr(trainer, "get_energy_stats"):
                trainer.get_energy_stats = lambda: self.calculate_energy_usage(
                    trainer.training_start_time if hasattr(trainer, "training_start_time") else time.time() - 60,
                    time.time()
                )
            
            if not hasattr(trainer, "check_energy_constraints"):
                trainer.check_energy_constraints = lambda: self.should_defer_training()
            
            if not hasattr(trainer, "optimize_backend"):
                trainer.optimize_backend = lambda: self.optimize_backend_selection("training")
    
    def integrate_with_model_optimizer(self, optimizer) -> None:
        """Integriert den EnergyEfficiencyManager mit einem TinyModelOptimizer.
        
        Args:
            optimizer: Eine Instanz von TinyModelOptimizer
        """
        if not hasattr(optimizer, "energy_manager"):
            optimizer.energy_manager = self
            logger.info(f"EnergyEfficiencyManager mit TinyModelOptimizer integriert")
            
            # Methodendefinitionen für Optimizer, wenn sie noch nicht existieren
            if not hasattr(optimizer, "get_energy_efficient_config"):
                optimizer.get_energy_efficient_config = lambda config=None: self.check_and_apply_device_constraints(config or {})
            
            if not hasattr(optimizer, "should_optimize_for_energy"):
                optimizer.should_optimize_for_energy = lambda: self.get_device_constraints().get("battery_limited", False)
