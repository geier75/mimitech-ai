#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - NEXUS-OS TaskManager

Diese Datei implementiert den TaskManager für NEXUS-OS, der die Verwaltung und
Ausführung von Aufgaben im System übernimmt.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import queue
import json

# Konfiguriere Logging
logger = logging.getLogger("MISO.nexus_os.task_manager")

class TaskManager:
    """
    TaskManager für NEXUS-OS
    
    Verwaltet die Ausführung von Aufgaben im System, einschließlich Priorisierung,
    Scheduling und Ressourcenzuweisung.
    """
    
    def __init__(self):
        """Initialisiert den TaskManager"""
        self.tasks = {}
        self.task_queue = queue.PriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.lock = threading.RLock()
        self.worker_thread = None
        self.running = False
        self.max_concurrent_tasks = 5
        logger.info("TaskManager initialisiert")
    
    def start(self):
        """Startet den TaskManager"""
        with self.lock:
            if self.running:
                logger.warning("TaskManager läuft bereits")
                return
            
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("TaskManager gestartet")
    
    def stop(self):
        """Stoppt den TaskManager"""
        with self.lock:
            if not self.running:
                logger.warning("TaskManager läuft nicht")
                return
            
            self.running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=2.0)
            logger.info("TaskManager gestoppt")
    
    def add_task(self, task: Dict[str, Any]) -> str:
        """
        Fügt eine Aufgabe zur Ausführung hinzu
        
        Args:
            task: Aufgabendefinition
            
        Returns:
            ID der hinzugefügten Aufgabe
        """
        with self.lock:
            # Generiere ID, falls nicht vorhanden
            if "id" not in task:
                task["id"] = str(uuid.uuid4())
            
            # Setze Standardwerte
            task.setdefault("priority", 5)  # Priorität 1-10 (1 höchste)
            task.setdefault("status", "pending")
            task.setdefault("created_at", time.time())
            
            # Speichere Aufgabe
            self.tasks[task["id"]] = task
            
            # Füge zur Warteschlange hinzu
            self.task_queue.put((task["priority"], task["id"]))
            
            logger.info(f"Aufgabe hinzugefügt: {task['id']}")
            return task["id"]
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Gibt eine Aufgabe anhand ihrer ID zurück
        
        Args:
            task_id: ID der Aufgabe
            
        Returns:
            Aufgabendefinition oder None, falls nicht gefunden
        """
        return self.tasks.get(task_id)
    
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """
        Aktualisiert eine Aufgabe
        
        Args:
            task_id: ID der Aufgabe
            updates: Zu aktualisierende Felder
            
        Returns:
            True, wenn die Aktualisierung erfolgreich war, sonst False
        """
        with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Aufgabe nicht gefunden: {task_id}")
                return False
            
            # Aktualisiere Aufgabe
            self.tasks[task_id].update(updates)
            logger.info(f"Aufgabe aktualisiert: {task_id}")
            return True
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Bricht eine Aufgabe ab
        
        Args:
            task_id: ID der Aufgabe
            
        Returns:
            True, wenn der Abbruch erfolgreich war, sonst False
        """
        with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Aufgabe nicht gefunden: {task_id}")
                return False
            
            task = self.tasks[task_id]
            
            if task["status"] == "running":
                # Markiere als abgebrochen
                task["status"] = "cancelled"
                self.failed_tasks[task_id] = task
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
            elif task["status"] == "pending":
                # Entferne aus Warteschlange (nicht direkt möglich bei PriorityQueue)
                task["status"] = "cancelled"
                self.failed_tasks[task_id] = task
            
            logger.info(f"Aufgabe abgebrochen: {task_id}")
            return True
    
    def get_task_queue(self) -> List[Dict[str, Any]]:
        """
        Gibt die aktuelle Aufgabenwarteschlange zurück
        
        Returns:
            Liste von Aufgaben in der Warteschlange
        """
        with self.lock:
            # Erstelle eine Kopie der Warteschlange
            queue_copy = []
            for task_id, task in self.tasks.items():
                if task["status"] == "pending":
                    queue_copy.append(task)
            
            # Sortiere nach Priorität
            queue_copy.sort(key=lambda x: x["priority"])
            return queue_copy
    
    def get_running_tasks(self) -> List[Dict[str, Any]]:
        """
        Gibt die aktuell laufenden Aufgaben zurück
        
        Returns:
            Liste von laufenden Aufgaben
        """
        with self.lock:
            return list(self.running_tasks.values())
    
    def get_completed_tasks(self) -> List[Dict[str, Any]]:
        """
        Gibt die abgeschlossenen Aufgaben zurück
        
        Returns:
            Liste von abgeschlossenen Aufgaben
        """
        with self.lock:
            return list(self.completed_tasks.values())
    
    def get_failed_tasks(self) -> List[Dict[str, Any]]:
        """
        Gibt die fehlgeschlagenen Aufgaben zurück
        
        Returns:
            Liste von fehlgeschlagenen Aufgaben
        """
        with self.lock:
            return list(self.failed_tasks.values())
    
    def _worker_loop(self):
        """Hauptschleife für die Aufgabenverarbeitung"""
        logger.info("TaskManager-Worker gestartet")
        
        while self.running:
            try:
                # Prüfe, ob wir weitere Aufgaben ausführen können
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    time.sleep(0.1)
                    continue
                
                # Hole nächste Aufgabe
                try:
                    priority, task_id = self.task_queue.get(block=True, timeout=0.5)
                except queue.Empty:
                    continue
                
                with self.lock:
                    # Prüfe, ob Aufgabe noch existiert und ausstehend ist
                    if task_id not in self.tasks or self.tasks[task_id]["status"] != "pending":
                        self.task_queue.task_done()
                        continue
                    
                    # Markiere als laufend
                    task = self.tasks[task_id]
                    task["status"] = "running"
                    task["started_at"] = time.time()
                    self.running_tasks[task_id] = task
                
                # Führe Aufgabe in separatem Thread aus
                execution_thread = threading.Thread(
                    target=self._execute_task,
                    args=(task_id,),
                    daemon=True
                )
                execution_thread.start()
                
                # Markiere Aufgabe als erledigt in der Warteschlange
                self.task_queue.task_done()
            
            except Exception as e:
                logger.error(f"Fehler im TaskManager-Worker: {e}")
                time.sleep(1.0)
    
    def _execute_task(self, task_id: str):
        """
        Führt eine Aufgabe aus
        
        Args:
            task_id: ID der Aufgabe
        """
        task = self.tasks[task_id]
        logger.info(f"Führe Aufgabe aus: {task_id}")
        
        try:
            # Simuliere Ausführung
            time.sleep(0.5)
            
            # Markiere als abgeschlossen
            with self.lock:
                task["status"] = "completed"
                task["completed_at"] = time.time()
                self.completed_tasks[task_id] = task
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
            
            logger.info(f"Aufgabe abgeschlossen: {task_id}")
        
        except Exception as e:
            # Markiere als fehlgeschlagen
            with self.lock:
                task["status"] = "failed"
                task["error"] = str(e)
                task["failed_at"] = time.time()
                self.failed_tasks[task_id] = task
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
            
            logger.error(f"Aufgabe fehlgeschlagen: {task_id}, Fehler: {e}")
