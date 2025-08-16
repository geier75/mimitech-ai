#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO Ultimate AGI - Konfigurationsmodul
======================================

Dieses Modul verwaltet die Konfiguration für das MISO Ultimate AGI-System.
"""

import os
import json
import logging
from pathlib import Path

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MISO.Config")

class MISOConfig:
    """Verwaltet die Konfiguration für das MISO Ultimate AGI-System."""
    
    def __init__(self, config_dir=None):
        """Initialisiert die Konfigurationsverwaltung."""
        if config_dir is None:
            self.config_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "config"
        else:
            self.config_dir = Path(config_dir)
        
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.configs = {}
        self.current_config = None
        self.load_configs()
    
    def load_configs(self):
        """Lädt alle verfügbaren Konfigurationen."""
        try:
            for config_file in self.config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    self.configs[config_file.stem] = config
                    logger.info(f"Konfiguration geladen: {config_file.stem}")
                except Exception as e:
                    logger.error(f"Fehler beim Laden der Konfiguration {config_file}: {e}")
        except Exception as e:
            logger.error(f"Fehler beim Durchsuchen des Konfigurationsverzeichnisses: {e}")
        
        # Erstelle eine Standardkonfiguration, falls keine existiert
        if not self.configs:
            self.create_default_config()
    
    def create_default_config(self):
        """Erstellt eine Standardkonfiguration."""
        default_config = {
            "name": "Standard-Konfiguration",
            "description": "Standardkonfiguration für MISO Ultimate AGI Training",
            "training": {
                "epochs": 100,
                "batch_size": 64,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "loss_function": "focal_loss",
                "metrics": ["accuracy", "f1_score", "precision", "recall"]
            },
            "components": {
                "MISO_CORE": {
                    "enabled": True,
                    "learning_rate": 0.001,
                    "architecture": "residual"
                },
                "VX_MEMEX": {
                    "enabled": True,
                    "learning_rate": 0.0005,
                    "architecture": "attention"
                },
                "VX_REASON": {
                    "enabled": True,
                    "learning_rate": 0.0008,
                    "architecture": "transformer"
                },
                "VX_INTENT": {
                    "enabled": True,
                    "learning_rate": 0.0012,
                    "architecture": "moe"
                }
            },
            "hardware": {
                "use_mlx": True,
                "mixed_precision": True,
                "float16": True,
                "cpu_threads": 8,
                "memory_limit": 16
            },
            "advanced": {
                "label_smoothing": 0.1,
                "dropout_rate": 0.2,
                "weight_decay": 0.0001,
                "gradient_clipping": 1.0,
                "use_residual_blocks": True,
                "use_attention_layers": True,
                "use_mixture_of_experts": True
            },
            "checkpoints": {
                "save_frequency": 10,
                "keep_best_only": False,
                "max_checkpoints": 5
            },
            "data": {
                "validation_split": 0.2,
                "shuffle": True,
                "augmentation": True
            }
        }
        
        self.save_config("default", default_config)
        self.configs["default"] = default_config
        self.current_config = "default"
        logger.info("Standardkonfiguration erstellt")
    
    def get_config(self, name):
        """Gibt eine Konfiguration zurück."""
        return self.configs.get(name, {})
    
    def save_config(self, name, config):
        """Speichert eine Konfiguration."""
        try:
            config_path = self.config_dir / f"{name}.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self.configs[name] = config
            logger.info(f"Konfiguration {name} gespeichert")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Konfiguration {name}: {e}")
            return False
    
    def delete_config(self, name):
        """Löscht eine Konfiguration."""
        if name == "default":
            logger.warning("Die Standardkonfiguration kann nicht gelöscht werden")
            return False
        
        try:
            config_path = self.config_dir / f"{name}.json"
            if config_path.exists():
                config_path.unlink()
            
            if name in self.configs:
                del self.configs[name]
            
            logger.info(f"Konfiguration {name} gelöscht")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Löschen der Konfiguration {name}: {e}")
            return False
    
    def get_config_names(self):
        """Gibt eine Liste aller Konfigurationsnamen zurück."""
        return list(self.configs.keys())
    
    def set_current_config(self, name):
        """Setzt die aktuelle Konfiguration."""
        if name in self.configs:
            self.current_config = name
            logger.info(f"Aktuelle Konfiguration auf {name} gesetzt")
            return True
        else:
            logger.warning(f"Konfiguration {name} nicht gefunden")
            return False
    
    def get_current_config(self):
        """Gibt die aktuelle Konfiguration zurück."""
        if self.current_config:
            return self.get_config(self.current_config)
        elif self.configs:
            self.current_config = next(iter(self.configs))
            return self.get_config(self.current_config)
        else:
            self.create_default_config()
            return self.get_config("default")

# Exportiere Hauptklasse
__all__ = ['MISOConfig']

if __name__ == "__main__":
    # Einfacher Test
    config_manager = MISOConfig()
    print("Verfügbare Konfigurationen:", config_manager.get_config_names())
    print("Aktuelle Konfiguration:", config_manager.get_current_config()["name"])
