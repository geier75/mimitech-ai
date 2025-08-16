#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-REFLEX: VXOR Integration Module
----------------------------------
VXORBridge-Adapter zur Anbindung an:
- VX-SOMA (Körperaktionen)
- VX-PSI (Bewusstseinsverknüpfung)
- VX-MEMEX (Reiz-Erinnerungen speichern)
- Q-LOGIK (Regelprüfung für erlaubte Reflexe)

Version: 0.1.0
Author: VXOR Build Core / Omega One
"""

import json
import logging
import time
import os
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable

# Konfiguration des Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/ubuntu/vXor_Modules/VX-REFLEX/logs/reflex.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VX-REFLEX.integration")

class VXORBridge:
    """
    Brücke zur Integration mit anderen VXOR-Modulen.
    """
    
    def __init__(self, config_path: str = "/home/ubuntu/vXor_Modules/VX-REFLEX/config/reflex_config.json"):
        """
        Initialisiert die VXOR-Bridge mit Konfigurationsparametern.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.logger = logger
        self.logger.info("Initialisiere VXOR-Bridge...")
        
        # Lade Konfiguration
        try:
            with open(config_path, 'r') as config_file:
                self.config = json.load(config_file)
            self.logger.info("Konfiguration erfolgreich geladen")
        except FileNotFoundError:
            self.logger.warning(f"Konfigurationsdatei {config_path} nicht gefunden. Verwende Standardkonfiguration.")
            self.config = self._get_default_config()
        
        # Initialisiere Verbindungen zu anderen Modulen
        self.connections = {
            "vx-soma": None,
            "vx-psi": None,
            "vx-memex": None,
            "q-logik": None
        }
        
        # Initialisiere Callback-Handler
        self.callbacks = {
            "vx-soma": {},
            "vx-psi": {},
            "vx-memex": {},
            "q-logik": {}
        }
        
        # Initialisiere Signal-Queue
        self.signal_queue = []
        self.queue_lock = threading.Lock()
        self.processing_thread = None
        self.active = False
        
        # Initialisiere Verbindungen
        self._init_connections()
        
        # Performance-Tracking
        self.performance_metrics = {
            "total_signals": 0,
            "signals_by_target": {
                "vx-soma": 0,
                "vx-psi": 0,
                "vx-memex": 0,
                "q-logik": 0
            },
            "signal_times": [],
            "avg_signal_time": 0,
            "max_signal_time": 0,
        }
        
        self.logger.info("VXOR-Bridge initialisiert")
    
    def _get_default_config(self) -> Dict:
        """Liefert die Standardkonfiguration zurück"""
        return {
            "vxor_bridge": {
                "connection_retry_interval": 5,  # in Sekunden
                "max_connection_attempts": 3,
                "signal_timeout": 2.0,  # in Sekunden
                "queue_processing_interval": 0.05,  # in Sekunden
                "max_queue_size": 100,
                "modules": {
                    "vx-soma": {
                        "enabled": True,
                        "path": "/vXor_Modules/VX-SOMA",
                        "interface": "api"
                    },
                    "vx-psi": {
                        "enabled": True,
                        "path": "/vXor_Modules/VX-PSI",
                        "interface": "api"
                    },
                    "vx-memex": {
                        "enabled": True,
                        "path": "/vXor_Modules/VX-MEMEX",
                        "interface": "api"
                    },
                    "q-logik": {
                        "enabled": True,
                        "path": "/vXor_Modules/Q-LOGIK",
                        "interface": "api"
                    }
                }
            }
        }
    
    def _init_connections(self):
        """Initialisiert Verbindungen zu anderen VXOR-Modulen"""
        bridge_config = self.config.get("vxor_bridge", {})
        modules = bridge_config.get("modules", {})
        
        for module_name, module_config in modules.items():
            if module_config.get("enabled", False):
                self.logger.info(f"Initialisiere Verbindung zu {module_name}...")
                
                # Versuche, Verbindung herzustellen
                connection = self._connect_to_module(module_name, module_config)
                if connection:
                    self.connections[module_name] = connection
                    self.logger.info(f"Verbindung zu {module_name} hergestellt")
                else:
                    self.logger.warning(f"Verbindung zu {module_name} konnte nicht hergestellt werden")
    
    def _connect_to_module(self, module_name: str, module_config: Dict[str, Any]) -> Optional[Any]:
        """
        Stellt eine Verbindung zu einem VXOR-Modul her.
        
        Args:
            module_name: Name des Moduls
            module_config: Modulkonfiguration
            
        Returns:
            Verbindungsobjekt oder None bei Fehler
        """
        # In einer realen Implementierung würde hier die tatsächliche Verbindung hergestellt
        # Für diese Simulation verwenden wir eine Mock-Implementierung
        
        module_path = module_config.get("path", "")
        interface_type = module_config.get("interface", "api")
        
        # Prüfe, ob Modul existiert
        if not os.path.exists(module_path):
            self.logger.warning(f"Modul {module_name} nicht gefunden unter {module_path}")
            return None
        
        # Simuliere Verbindung
        connection = {
            "name": module_name,
            "path": module_path,
            "interface": interface_type,
            "connected": True,
            "last_ping": time.time()
        }
        
        return connection
    
    def start(self):
        """Startet die Signal-Verarbeitung"""
        if self.active:
            self.logger.warning("VXOR-Bridge läuft bereits")
            return
        
        self.active = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info("VXOR-Bridge gestartet")
    
    def stop(self):
        """Stoppt die Signal-Verarbeitung"""
        if not self.active:
            self.logger.warning("VXOR-Bridge läuft nicht")
            return
        
        self.active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        self.logger.info("VXOR-Bridge gestoppt")
    
    def _processing_loop(self):
        """Hauptverarbeitungsschleife für die Signal-Queue"""
        self.logger.info("Verarbeitungsschleife gestartet")
        
        bridge_config = self.config.get("vxor_bridge", {})
        processing_interval = bridge_config.get("queue_processing_interval", 0.05)
        
        while self.active:
            # Verarbeite Signale in der Queue
            with self.queue_lock:
                if self.signal_queue:
                    # Verarbeite das nächste Signal
                    target, data, timestamp = self.signal_queue.pop(0)
                    
                    # Prüfe auf veraltete Signale
                    current_time = time.time()
                    if current_time - timestamp > bridge_config.get("signal_timeout", 2.0):
                        self.logger.warning(f"Veraltetes Signal verworfen: Ziel={target}, Alter={(current_time-timestamp)*1000:.2f}ms")
                        continue
            
                    # Verarbeite Signal außerhalb des Lock
                    self._process_signal(target, data)
            
            # Kurze Pause zur CPU-Entlastung
            time.sleep(processing_interval)
    
    def _process_signal(self, target: str, data: Dict[str, Any]):
        """
        Verarbeitet ein Signal für ein Zielmodul.
        
        Args:
            target: Zielmodul
            data: Signaldaten
        """
        # Prüfe, ob Verbindung zum Zielmodul besteht
        if target not in self.connections or not self.connections[target]:
            self.logger.warning(f"Keine Verbindung zu {target}")
            return
        
        # Prüfe, ob Zielmodul aktiv ist
        connection = self.connections[target]
        if not connection.get("connected", False):
            self.logger.warning(f"Verbindung zu {target} nicht aktiv")
            return
        
        # Sende Signal an Zielmodul
        self.logger.info(f"Sende Signal an {target}: {data}")
        
        # In einer realen Implementierung würde hier das tatsächliche Senden erfolgen
        # Für diese Simulation simulieren wir eine erfolgreiche Übertragung
        
        # Aktualisiere Performance-Metriken
        self.performance_metrics["total_signals"] += 1
        if target in self.performance_metrics["signals_by_target"]:
            self.performance_metrics["signals_by_target"][target] += 1
    
    def send_signal(self, target: str, data: Dict[str, Any], immediate: bool = False) -> Dict[str, Any]:
        """
        Sendet ein Signal an ein Zielmodul.
        
        Args:
            target: Zielmodul
            data: Signaldaten
            immediate: Sofortige Verarbeitung (True) oder Queue (False)
            
        Returns:
            Ergebnis des Signalversands
        """
        start_time = time.perf_counter()
        
        # Protokolliere Signal
        self.logger.debug(f"Sende Signal an {target}: {data}")
        
        # Prüfe, ob Zielmodul unterstützt wird
        if target not in self.connections:
            self.logger.warning(f"Unbekanntes Zielmodul: {target}")
            return {"success": False, "error": f"Unbekanntes Zielmodul: {target}"}
        
        # Prüfe, ob Verbindung zum Zielmodul besteht
        if not self.connections[target]:
            self.logger.warning(f"Keine Verbindung zu {target}")
            return {"success": False, "error": f"Keine Verbindung zu {target}"}
        
        # Verarbeite Signal sofort oder füge es zur Queue hinzu
        if immediate:
            result = self._process_signal(target, data)
        else:
            # Füge Signal zur Queue hinzu
            with self.queue_lock:
                bridge_config = self.config.get("vxor_bridge", {})
                max_queue_size = bridge_config.get("max_queue_size", 100)
                
                if len(self.signal_queue) >= max_queue_size:
                    self.logger.warning(f"Signal-Queue voll, verwerfe ältestes Signal")
                    self.signal_queue.pop(0)
                
                self.signal_queue.append((target, data, time.time()))
            
            result = {"success": True, "queued": True}
        
        # Aktualisiere Performance-Metriken
        end_time = time.perf_counter()
        signal_time = (end_time - start_time) * 1000  # in ms
        
        self.performance_metrics["signal_times"].append(signal_time)
        self.performance_metrics["avg_signal_time"] = sum(self.performance_metrics["signal_times"]) / len(self.performance_metrics["signal_times"])
        self.performance_metrics["max_signal_time"] = max(self.performance_metrics["signal_times"])
        
        return result
    
    def register_callback(self, source: str, event_type: str, callback: Callable):
        """
        Registriert einen Callback für ein bestimmtes Ereignis.
        
        Args:
            source: Quellmodul
            event_type: Ereignistyp
            callback: Callback-Funktion
        """
        if source not in self.callbacks:
            self.callbacks[source] = {}
        
        if event_type not in self.callbacks[source]:
            self.callbacks[source][event_type] = []
        
        self.callbacks[source][event_type].append(callback)
        self.logger.info(f"Callback für {source}/{event_type} registriert")
    
    def unregister_callback(self, source: str, event_type: str, callback: Callable) -> bool:
        """
        Entfernt einen Callback für ein bestimmtes Ereignis.
        
        Args:
            source: Quellmodul
            event_type: Ereignistyp
            callback: Callback-Funktion
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        if source not in self.callbacks:
            return False
        
        if event_type not in self.callbacks[source]:
            return False
        
        if callback in self.callbacks[source][event_type]:
            self.callbacks[source][event_type].remove(callback)
            self.logger.info(f"Callback für {source}/{event_type} entfernt")
            return True
        
        return False
    
    def handle_event(self, source: str, event_type: str, data: Dict[str, Any]):
        """
        Verarbeitet ein eingehendes Ereignis und ruft registrierte Callbacks auf.
        
        Args:
            source: Quellmodul
            event_type: Ereignistyp
            data: Ereignisdaten
        """
        self.logger.debug(f"Ereignis empfangen: {source}/{event_type}: {data}")
        
        # Prüfe, ob Callbacks für dieses Ereignis registriert sind
        if source in self.callbacks and event_type in self.callbacks[source]:
            callbacks = self.callbacks[source][event_type]
            for callback in callbacks:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Fehler in Callback für {source}/{event_type}: {e}")
    
    def store_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Speichert Daten in VX-MEMEX.
        
        Args:
            data: Zu speichernde Daten
            
        Returns:
            Ergebnis des Speichervorgangs
        """
        return self.send_signal("vx-memex", {
            "action": "store",
            "data": data
        })
    
    def retrieve_memory(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ruft Daten aus VX-MEMEX ab.
        
        Args:
            query: Abfrage
            
        Returns:
            Abfrageergebnis
        """
        # In einer realen Implementierung würde hier eine tatsächliche Abfrage erfolgen
        # Für diese Simulation simulieren wir ein Ergebnis
        
        self.logger.info(f"Rufe Daten aus VX-MEMEX ab: {query}")
        
        # Simuliere Abfrageergebnis
        result = {
            "success": True,
            "data": []
        }
        
        return result
    
    def check_rule(self, rule_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prüft eine Regel mit Q-LOGIK.
        
        Args:
            rule_type: Regeltyp
            data: Zu prüfende Daten
            
        Returns:
            Prüfergebnis
        """
        return self.send_signal("q-logik", {
            "action": "check_rule",
            "rule_type": rule_type,
            "data": data
        })
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Liefert Performance-Metriken der VXOR-Bridge.
        
        Returns:
            Dictionary mit Performance-Metriken
        """
        return {
            "total_signals": self.performance_metrics["total_signals"],
            "signals_by_target": self.performance_metrics["signals_by_target"],
            "avg_signal_time_ms": self.performance_metrics["avg_signal_time"],
            "max_signal_time_ms": self.performance_metrics["max_signal_time"],
            "queue_size": len(self.signal_queue),
            "active": self.active
        }
    
    def get_connection_status(self) -> Dict[st
(Content truncated due to size limit. Use line ranges to read in chunks)