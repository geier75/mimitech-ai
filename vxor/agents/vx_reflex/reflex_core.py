#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-REFLEX: Reflex Core Module
------------------------------
Zentrale Steuerungseinheit für das VX-REFLEX Modul des VXOR-Systems.
Verantwortlich für Reizaufnahme, Entscheidungslogik und Priorisierung.

Version: 0.1.0
Author: VXOR Build Core / Omega One
"""

import time
import json
import logging
import threading
import os
from enum import Enum
from typing import Dict, List, Any, Callable, Optional, Tuple

# Konfiguration des Logging - Fixed path for current environment
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "reflex.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VX-REFLEX.core")

class StimulusPriority(Enum):
    """Prioritätsklassen für eingehende Reize"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class ReflexCore:
    """
    Zentrale Steuerungseinheit des VX-REFLEX Moduls.
    Verarbeitet eingehende Reize und löst entsprechende Reaktionen aus.
    """
    
    def __init__(self, config_path: str = "/home/ubuntu/vXor_Modules/VX-REFLEX/config/reflex_config.json"):
        """
        Initialisiert den ReflexCore mit Konfigurationsparametern.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.logger = logger
        self.logger.info("Initialisiere VX-REFLEX Core...")
        
        # Lade Konfiguration
        try:
            with open(config_path, 'r') as config_file:
                self.config = json.load(config_file)
            self.logger.info("Konfiguration erfolgreich geladen")
        except FileNotFoundError:
            self.logger.warning(f"Konfigurationsdatei {config_path} nicht gefunden. Verwende Standardkonfiguration.")
            self.config = self._get_default_config()
            
            # Erstelle Konfigurationsdatei mit Standardwerten
            import os
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as config_file:
                json.dump(self.config, config_file, indent=4)
        
        # Initialisiere Komponenten
        self.stimulus_handlers = {}  # Typ -> Handler-Funktion
        self.active = False
        self.processing_thread = None
        self.stimulus_queue = []
        self.queue_lock = threading.Lock()
        
        # Performance-Tracking
        self.performance_metrics = {
            "total_stimuli": 0,
            "response_times": [],
            "avg_response_time": 0,
            "max_response_time": 0,
        }
        
        self.logger.info("VX-REFLEX Core initialisiert")
    
    def _get_default_config(self) -> Dict:
        """Liefert die Standardkonfiguration zurück"""
        return {
            "thresholds": {
                "cpu_load": {
                    "high": 90,
                    "medium": 75,
                    "low": 50
                },
                "audio_level": {
                    "high": 75,  # in dB
                    "medium": 60,
                    "low": 40
                },
                "object_proximity": {
                    "high": 0.8,  # in Metern
                    "medium": 1.5,
                    "low": 3.0
                }
            },
            "reaction_profiles": {
                "default": {
                    "response_delay": 0.05,  # in Sekunden
                    "priority_weights": {
                        "HIGH": 1.0,
                        "MEDIUM": 0.6,
                        "LOW": 0.3
                    }
                },
                "emergency": {
                    "response_delay": 0.01,
                    "priority_weights": {
                        "HIGH": 1.0,
                        "MEDIUM": 0.8,
                        "LOW": 0.5
                    }
                }
            },
            "active_profile": "default",
            "max_queue_size": 100,
            "processing_interval": 0.01  # in Sekunden
        }
    
    def register_stimulus_handler(self, stimulus_type: str, handler: Callable):
        """
        Registriert einen Handler für einen bestimmten Reiztyp.
        
        Args:
            stimulus_type: Typ des Reizes (z.B. "visual", "audio", "system")
            handler: Callback-Funktion zur Verarbeitung des Reizes
        """
        self.stimulus_handlers[stimulus_type] = handler
        self.logger.info(f"Handler für Reiztyp '{stimulus_type}' registriert")
    
    def process_stimulus(self, stimulus_type: str, data: Dict[str, Any]) -> Tuple[bool, Optional[Dict]]:
        """
        Verarbeitet einen eingehenden Reiz und bestimmt die Reaktion.
        
        Args:
            stimulus_type: Typ des Reizes
            data: Reizdaten
            
        Returns:
            Tuple aus (Erfolg, Reaktionsdaten)
        """
        start_time = time.perf_counter()
        
        # Protokolliere eingehenden Reiz
        self.logger.debug(f"Eingehender Reiz: Typ={stimulus_type}, Daten={data}")
        
        # Bestimme Priorität des Reizes
        priority = self._determine_priority(stimulus_type, data)
        
        # Verarbeite Reiz entsprechend seiner Priorität
        if priority == StimulusPriority.HIGH:
            # Sofortige Verarbeitung für hochprioritäre Reize
            result = self._handle_stimulus(stimulus_type, data, priority)
        else:
            # Füge Reiz zur Verarbeitungsqueue hinzu
            with self.queue_lock:
                self.stimulus_queue.append((stimulus_type, data, priority, start_time))
                
                # Begrenze Queuegröße
                if len(self.stimulus_queue) > self.config["max_queue_size"]:
                    # Entferne niedrigprioritäre Reize
                    self.stimulus_queue.sort(key=lambda x: x[2].value, reverse=True)
                    self.stimulus_queue = self.stimulus_queue[:self.config["max_queue_size"]]
            
            result = (True, {"status": "queued", "priority": priority.name})
        
        # Aktualisiere Performance-Metriken
        end_time = time.perf_counter()
        response_time = (end_time - start_time) * 1000  # in ms
        
        self.performance_metrics["total_stimuli"] += 1
        self.performance_metrics["response_times"].append(response_time)
        self.performance_metrics["avg_response_time"] = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"])
        self.performance_metrics["max_response_time"] = max(self.performance_metrics["response_times"])
        
        if response_time > 80:  # 80ms Schwellenwert
            self.logger.warning(f"Reaktionszeit überschreitet Schwellenwert: {response_time:.2f}ms für {stimulus_type}")
        
        return result
    
    def _determine_priority(self, stimulus_type: str, data: Dict[str, Any]) -> StimulusPriority:
        """
        Bestimmt die Priorität eines Reizes basierend auf Typ und Daten.
        
        Args:
            stimulus_type: Typ des Reizes
            data: Reizdaten
            
        Returns:
            Prioritätsklasse des Reizes
        """
        # Beispielimplementierung für verschiedene Reiztypen
        if stimulus_type == "system":
            if "cpu_load" in data:
                cpu_load = data["cpu_load"]
                if cpu_load >= self.config["thresholds"]["cpu_load"]["high"]:
                    return StimulusPriority.HIGH
                elif cpu_load >= self.config["thresholds"]["cpu_load"]["medium"]:
                    return StimulusPriority.MEDIUM
                else:
                    return StimulusPriority.LOW
        
        elif stimulus_type == "audio":
            if "level" in data:
                audio_level = data["level"]
                if audio_level >= self.config["thresholds"]["audio_level"]["high"]:
                    return StimulusPriority.HIGH
                elif audio_level >= self.config["thresholds"]["audio_level"]["medium"]:
                    return StimulusPriority.MEDIUM
                else:
                    return StimulusPriority.LOW
        
        elif stimulus_type == "visual":
            # Prüfe auf Gefahrenmuster
            if "danger_object" in data and data["danger_object"]:
                if "proximity" in data:
                    proximity = data["proximity"]
                    if proximity <= self.config["thresholds"]["object_proximity"]["high"]:
                        return StimulusPriority.HIGH
                    elif proximity <= self.config["thresholds"]["object_proximity"]["medium"]:
                        return StimulusPriority.MEDIUM
                    else:
                        return StimulusPriority.LOW
                return StimulusPriority.MEDIUM
            
            # Prüfe auf Bewegungsmuster
            if "motion_pattern" in data:
                if data["motion_pattern"] in ["schlag", "sturz", "explosion"]:
                    return StimulusPriority.HIGH
        
        elif stimulus_type == "emotional":
            if "intensity" in data:
                intensity = data["intensity"]
                if intensity >= 0.8:
                    return StimulusPriority.HIGH
                elif intensity >= 0.5:
                    return StimulusPriority.MEDIUM
                else:
                    return StimulusPriority.LOW
        
        # Standardpriorität für unbekannte Reiztypen
        return StimulusPriority.LOW
    
    def _handle_stimulus(self, stimulus_type: str, data: Dict[str, Any], 
                         priority: StimulusPriority) -> Tuple[bool, Optional[Dict]]:
        """
        Verarbeitet einen Reiz und löst entsprechende Reaktion aus.
        
        Args:
            stimulus_type: Typ des Reizes
            data: Reizdaten
            priority: Priorität des Reizes
            
        Returns:
            Tuple aus (Erfolg, Reaktionsdaten)
        """
        # Prüfe, ob ein Handler für diesen Reiztyp registriert ist
        if stimulus_type in self.stimulus_handlers:
            try:
                # Füge Prioritätsinformation hinzu
                data["priority"] = priority.name
                
                # Rufe Handler auf
                handler = self.stimulus_handlers[stimulus_type]
                result = handler(data)
                
                self.logger.info(f"Reiz verarbeitet: Typ={stimulus_type}, Priorität={priority.name}")
                return True, result
            except Exception as e:
                self.logger.error(f"Fehler bei Reizverarbeitung: {e}")
                return False, {"error": str(e)}
        else:
            self.logger.warning(f"Kein Handler für Reiztyp '{stimulus_type}' registriert")
            return False, {"error": f"Kein Handler für Reiztyp '{stimulus_type}' registriert"}
    
    def start(self):
        """Startet die Reizverarbeitung"""
        if self.active:
            self.logger.warning("VX-REFLEX Core läuft bereits")
            return
        
        self.active = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info("VX-REFLEX Core gestartet")
    
    def stop(self):
        """Stoppt die Reizverarbeitung"""
        if not self.active:
            self.logger.warning("VX-REFLEX Core läuft nicht")
            return
        
        self.active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        self.logger.info("VX-REFLEX Core gestoppt")
    
    def _processing_loop(self):
        """Hauptverarbeitungsschleife für die Reizqueue"""
        self.logger.info("Verarbeitungsschleife gestartet")
        
        while self.active:
            # Verarbeite Reize in der Queue
            with self.queue_lock:
                if self.stimulus_queue:
                    # Sortiere Queue nach Priorität
                    self.stimulus_queue.sort(key=lambda x: x[2].value, reverse=True)
                    
                    # Verarbeite den höchstprioritären Reiz
                    stimulus_type, data, priority, queue_time = self.stimulus_queue.pop(0)
                    
                    # Prüfe auf veraltete Reize
                    current_time = time.perf_counter()
                    if current_time - queue_time > 0.5:  # 500ms Schwellenwert
                        self.logger.warning(f"Veralteter Reiz verworfen: Typ={stimulus_type}, Alter={(current_time-queue_time)*1000:.2f}ms")
                        continue
            
                    # Verarbeite Reiz außerhalb des Lock
                    self._handle_stimulus(stimulus_type, data, priority)
            
            # Kurze Pause zur CPU-Entlastung
            time.sleep(self.config["processing_interval"])
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Liefert Performance-Metriken der Reizverarbeitung.
        
        Returns:
            Dictionary mit Performance-Metriken
        """
        return {
            "total_stimuli": self.performance_metrics["total_stimuli"],
            "avg_response_time_ms": self.performance_metrics["avg_response_time"],
            "max_response_time_ms": self.performance_metrics["max_response_time"],
            "queue_size": len(self.stimulus_queue),
            "active": self.active
        }


# Singleton-Instanz
_reflex_core_instance = None

def get_reflex_core() -> ReflexCore:
    """
    Liefert die Singleton-Instanz des ReflexCore.
    
    Returns:
        ReflexCore-Instanz
    """
    global _reflex_core_instance
    if _reflex_core_instance is None:
        _reflex_core_instance = ReflexCore()
    return _reflex_core_instance


if __name__ == "__main__":
    # Einfacher Test
    core = get_reflex_core()
    core.start()
    
    # Beispiel für einen Reiz
    result = core.process_stimulus("system", {"cpu_load": 95})
    print(f"Ergebnis: {result}")
    
    # Warte kurz, damit die Verarbeitung stattfinden kann
    time.sleep(1)
    
    # Zeige Performance-Metriken
    print(f"Performance: {core.get_performance_metrics()}")
    
    core.stop()
