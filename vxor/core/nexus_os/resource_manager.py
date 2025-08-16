#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - NEXUS-OS ResourceManager

Diese Datei implementiert den ResourceManager für NEXUS-OS, der die Verwaltung und
Zuweisung von Systemressourcen übernimmt.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import threading
import time
import uuid
import psutil
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import json

# Konfiguriere Logging
logger = logging.getLogger("MISO.nexus_os.resource_manager")

class ResourceManager:
    """
    ResourceManager für NEXUS-OS
    
    Verwaltet die Zuweisung von Systemressourcen wie CPU, Speicher, GPU und
    anderen Hardwarekomponenten an Aufgaben im System.
    """
    
    def __init__(self):
        """Initialisiert den ResourceManager"""
        self.resources = {}
        self.allocations = {}
        self.lock = threading.RLock()
        self.monitoring_thread = None
        self.running = False
        self.monitoring_interval = 5.0  # Sekunden
        
        # Initialisiere verfügbare Ressourcen
        self._initialize_resources()
        
        logger.info("ResourceManager initialisiert")
    
    def _initialize_resources(self):
        """Initialisiert die verfügbaren Ressourcen"""
        with self.lock:
            # CPU
            cpu_count = psutil.cpu_count(logical=True)
            self.resources["cpu"] = {
                "type": "cpu",
                "total": cpu_count,
                "available": cpu_count,
                "allocated": 0,
                "utilization": 0.0
            }
            
            # Speicher (RAM)
            mem = psutil.virtual_memory()
            self.resources["memory"] = {
                "type": "memory",
                "total": mem.total,
                "available": mem.available,
                "allocated": 0,
                "utilization": mem.percent
            }
            
            # GPU (falls verfügbar)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    for i in range(gpu_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        self.resources[f"gpu_{i}"] = {
                            "type": "gpu",
                            "name": gpu_name,
                            "total": 1.0,
                            "available": 1.0,
                            "allocated": 0.0,
                            "utilization": 0.0
                        }
                elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    self.resources["mps"] = {
                        "type": "gpu",
                        "name": "Apple Metal Performance Shaders",
                        "total": 1.0,
                        "available": 1.0,
                        "allocated": 0.0,
                        "utilization": 0.0
                    }
            except (ImportError, AttributeError):
                logger.debug("Keine GPU-Unterstützung gefunden")
            
            # Disk
            disk = psutil.disk_usage('/')
            self.resources["disk"] = {
                "type": "disk",
                "total": disk.total,
                "available": disk.free,
                "allocated": 0,
                "utilization": disk.percent
            }
            
            # Netzwerk (vereinfacht)
            self.resources["network"] = {
                "type": "network",
                "total": 100.0,  # Prozent
                "available": 100.0,
                "allocated": 0.0,
                "utilization": 0.0
            }
            
            logger.info(f"Ressourcen initialisiert: {len(self.resources)} verfügbar")
    
    def start_monitoring(self):
        """Startet die Ressourcenüberwachung"""
        with self.lock:
            if self.running:
                logger.warning("ResourceManager überwacht bereits")
                return
            
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("ResourceManager-Überwachung gestartet")
    
    def stop_monitoring(self):
        """Stoppt die Ressourcenüberwachung"""
        with self.lock:
            if not self.running:
                logger.warning("ResourceManager überwacht nicht")
                return
            
            self.running = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2.0)
            logger.info("ResourceManager-Überwachung gestoppt")
    
    def allocate_resources(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Weist einer Aufgabe Ressourcen zu
        
        Args:
            task: Aufgabendefinition
            
        Returns:
            Zugewiesene Ressourcen
        """
        with self.lock:
            task_id = task["id"]
            requested_resources = task.get("resources", [])
            
            # Erstelle Zuweisung
            allocation = {
                "task_id": task_id,
                "allocated_at": time.time(),
                "resources": {}
            }
            
            # Weise Ressourcen zu
            for resource_name in requested_resources:
                if resource_name not in self.resources:
                    logger.warning(f"Ressource nicht gefunden: {resource_name}")
                    continue
                
                resource = self.resources[resource_name]
                
                # Berechne zuzuweisende Menge (vereinfacht)
                if resource["type"] == "cpu":
                    # Weise 1 CPU-Kern zu
                    allocated_amount = 1
                elif resource["type"] == "memory":
                    # Weise 1 GB Speicher zu
                    allocated_amount = 1 * 1024 * 1024 * 1024
                elif resource["type"] == "gpu":
                    # Weise 25% der GPU zu
                    allocated_amount = 0.25
                elif resource["type"] == "disk":
                    # Weise 100 MB Speicherplatz zu
                    allocated_amount = 100 * 1024 * 1024
                elif resource["type"] == "network":
                    # Weise 10% der Netzwerkbandbreite zu
                    allocated_amount = 10.0
                else:
                    allocated_amount = 0
                
                # Prüfe, ob genügend Ressourcen verfügbar sind
                if resource["available"] < allocated_amount:
                    logger.warning(f"Nicht genügend Ressourcen verfügbar: {resource_name}")
                    continue
                
                # Aktualisiere Ressourcenverfügbarkeit
                resource["available"] -= allocated_amount
                resource["allocated"] += allocated_amount
                
                # Füge zur Zuweisung hinzu
                allocation["resources"][resource_name] = {
                    "type": resource["type"],
                    "allocated": allocated_amount
                }
            
            # Speichere Zuweisung
            self.allocations[task_id] = allocation
            
            logger.info(f"Ressourcen zugewiesen für Aufgabe: {task_id}")
            return allocation
    
    def release_resources(self, task_id: str) -> bool:
        """
        Gibt die einer Aufgabe zugewiesenen Ressourcen frei
        
        Args:
            task_id: ID der Aufgabe
            
        Returns:
            True, wenn die Freigabe erfolgreich war, sonst False
        """
        with self.lock:
            if task_id not in self.allocations:
                logger.warning(f"Keine Ressourcenzuweisung gefunden für Aufgabe: {task_id}")
                return False
            
            allocation = self.allocations[task_id]
            
            # Gebe Ressourcen frei
            for resource_name, resource_allocation in allocation["resources"].items():
                if resource_name not in self.resources:
                    logger.warning(f"Ressource nicht gefunden: {resource_name}")
                    continue
                
                resource = self.resources[resource_name]
                allocated_amount = resource_allocation["allocated"]
                
                # Aktualisiere Ressourcenverfügbarkeit
                resource["available"] += allocated_amount
                resource["allocated"] -= allocated_amount
            
            # Entferne Zuweisung
            del self.allocations[task_id]
            
            logger.info(f"Ressourcen freigegeben für Aufgabe: {task_id}")
            return True
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Gibt die aktuelle Ressourcennutzung zurück
        
        Returns:
            Dictionary mit Ressourcennutzung
        """
        with self.lock:
            return {
                "resources": self.resources,
                "allocations": self.allocations
            }
    
    def get_resource(self, resource_name: str) -> Optional[Dict[str, Any]]:
        """
        Gibt Informationen zu einer bestimmten Ressource zurück
        
        Args:
            resource_name: Name der Ressource
            
        Returns:
            Ressourceninformationen oder None, falls nicht gefunden
        """
        with self.lock:
            return self.resources.get(resource_name)
    
    def _monitoring_loop(self):
        """Hauptschleife für die Ressourcenüberwachung"""
        logger.info("ResourceManager-Überwachung gestartet")
        
        while self.running:
            try:
                # Aktualisiere Ressourcennutzung
                self._update_resource_usage()
                
                # Warte auf nächstes Intervall
                time.sleep(self.monitoring_interval)
            
            except Exception as e:
                logger.error(f"Fehler in ResourceManager-Überwachung: {e}")
                time.sleep(1.0)
    
    def _update_resource_usage(self):
        """Aktualisiert die aktuelle Ressourcennutzung"""
        with self.lock:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if "cpu" in self.resources:
                self.resources["cpu"]["utilization"] = cpu_percent
            
            # Speicher (RAM)
            mem = psutil.virtual_memory()
            if "memory" in self.resources:
                self.resources["memory"]["available"] = mem.available
                self.resources["memory"]["utilization"] = mem.percent
            
            # GPU (falls verfügbar)
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        resource_name = f"gpu_{i}"
                        if resource_name in self.resources:
                            # In einer echten Implementierung würde hier
                            # die tatsächliche GPU-Auslastung ermittelt werden
                            pass
            except (ImportError, AttributeError):
                pass
            
            # Disk
            disk = psutil.disk_usage('/')
            if "disk" in self.resources:
                self.resources["disk"]["available"] = disk.free
                self.resources["disk"]["utilization"] = disk.percent
            
            # Netzwerk (vereinfacht)
            # In einer echten Implementierung würde hier
            # die tatsächliche Netzwerkauslastung ermittelt werden
            
            logger.debug("Ressourcennutzung aktualisiert")
