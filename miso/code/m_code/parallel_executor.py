#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Parallel Executor

Dieses Modul implementiert einen parallelen Ausführer für M-CODE Operationen.
Es ermöglicht die effiziente Ausführung von rechenintensiven Aufgaben auf Mehrkernsystemen.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import time
import threading
import multiprocessing
import queue
import uuid
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, wait, FIRST_COMPLETED

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.parallel_executor")


@dataclass
class ParallelTaskOptions:
    """Optionen für parallele Aufgaben"""
    priority: int = 0  # Höhere Werte bedeuten höhere Priorität
    timeout_seconds: Optional[float] = None
    use_processes: bool = False  # True für Prozesse, False für Threads
    max_retries: int = 0
    tag: Optional[str] = None
    cancel_on_failure: bool = False
    propagate_exceptions: bool = True


@dataclass
class ParallelTaskResult:
    """Ergebnis einer parallelen Aufgabe"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time_ms: float = 0.0
    retries: int = 0
    task_tag: Optional[str] = None
    creation_timestamp: float = field(default_factory=time.time)
    completion_timestamp: Optional[float] = None


class ParallelExecutionError(Exception):
    """Fehler während der parallelen Ausführung"""
    pass


class TaskCancellationError(Exception):
    """Fehler bei Aufgabenabbruch"""
    pass


class ParallelExecutor:
    """Ausführer für parallele M-CODE Operationen"""
    
    def __init__(self, 
                 max_workers: Optional[int] = None, 
                 use_processes: bool = False,
                 task_queue_size: int = 100):
        """
        Initialisiert einen neuen parallelen Ausführer.
        
        Args:
            max_workers: Maximale Anzahl von Workern (Standard: Anzahl CPUs)
            use_processes: Ob Prozesse statt Threads verwendet werden sollen
            task_queue_size: Größe der Aufgabenwarteschlange
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 4))
        self.use_processes = use_processes
        self.task_queue_size = task_queue_size
        
        # Initialisiere Executor
        self._initialize_executor()
        
        # Aufgabenverfolgung
        self.tasks = {}
        self.task_results = {}
        self.task_lock = threading.RLock()
        
        # Status
        self.is_running = True
        
        logger.info(f"Paralleler Ausführer initialisiert mit {self.max_workers} Workern " 
                   f"({'Prozesse' if self.use_processes else 'Threads'})")
    
    def _initialize_executor(self) -> None:
        """Initialisiert den Thread- oder Process-Pool"""
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
            # Hintergrund-Thread für Prozessüberwachung
            self.monitor_thread = threading.Thread(target=self._monitor_processes, daemon=True)
            self.monitor_thread.start()
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def _monitor_processes(self) -> None:
        """Überwacht Prozesse und sammelt Ergebnisse ein"""
        while self.is_running:
            # Prüfe abgeschlossene Aufgaben
            with self.task_lock:
                for task_id, future in list(self.tasks.items()):
                    if future.done():
                        try:
                            result = future.result()
                            self.task_results[task_id] = result
                        except Exception as e:
                            logger.error(f"Fehler in Aufgabe {task_id}: {e}")
                            # Erfasse Fehler im Ergebnis
                            self.task_results[task_id] = ParallelTaskResult(
                                task_id=task_id,
                                success=False,
                                error=e,
                                completion_timestamp=time.time()
                            )
                        
                        # Entferne Aufgabe aus aktiver Verfolgung
                        self.tasks.pop(task_id, None)
            
            # Kurze Pause, um CPU-Last zu reduzieren
            time.sleep(0.01)
    
    def _wrap_task(self, func: Callable, task_id: str, 
                  options: ParallelTaskOptions, *args, **kwargs) -> ParallelTaskResult:
        """
        Umschließt eine Aufgabe mit Fehlerbehandlung und Zeitmessung.
        
        Args:
            func: Auszuführende Funktion
            task_id: Aufgaben-ID
            options: Aufgabenoptionen
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Aufgabenergebnis
        """
        start_time = time.time()
        retries = 0
        error = None
        result = None
        success = False
        
        while retries <= options.max_retries:
            try:
                # Führe Funktion aus
                result = func(*args, **kwargs)
                success = True
                break
            except Exception as e:
                # Erfasse Fehler
                error = e
                logger.warning(f"Fehler in Aufgabe {task_id} (Versuch {retries + 1}/{options.max_retries + 1}): {e}")
                retries += 1
                
                # Wenn alle Versuche fehlgeschlagen sind oder keine Wiederholungen konfiguriert sind
                if retries > options.max_retries:
                    if options.propagate_exceptions:
                        logger.error(f"Aufgabe {task_id} endgültig fehlgeschlagen: {e}")
                    break
                
                # Kurze Pause vor Wiederholung
                time.sleep(0.1 * retries)  # Exponentielles Backoff
        
        # Berechne Ausführungszeit in Millisekunden
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Erstelle Ergebnisobjekt
        task_result = ParallelTaskResult(
            task_id=task_id,
            success=success,
            result=result if success else None,
            error=error,
            execution_time_ms=execution_time_ms,
            retries=retries,
            task_tag=options.tag,
            completion_timestamp=time.time()
        )
        
        # Speichere Ergebnis im internen Cache
        with self.task_lock:
            self.task_results[task_id] = task_result
        
        return task_result
    
    def submit(self, func: Callable, *args, 
              options: Optional[ParallelTaskOptions] = None, **kwargs) -> str:
        """
        Reicht eine Aufgabe zur parallelen Ausführung ein.
        
        Args:
            func: Auszuführende Funktion
            *args: Positionsargumente
            options: Aufgabenoptionen
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Aufgaben-ID
        """
        if not self.is_running:
            raise ParallelExecutionError("Paralleler Ausführer wurde bereits heruntergefahren")
            
        # Verwende Standardoptionen, wenn keine angegeben sind
        options = options or ParallelTaskOptions()
        
        # Generiere eindeutige Aufgaben-ID
        task_id = str(uuid.uuid4())
        
        # Umschließe Aufgabe mit Wrapper
        wrapped_func = lambda: self._wrap_task(func, task_id, options, *args, **kwargs)
        
        # Reiche Aufgabe ein
        future = self.executor.submit(wrapped_func)
        
        # Speichere Aufgabe
        with self.task_lock:
            self.tasks[task_id] = future
        
        logger.debug(f"Aufgabe {task_id} eingereicht (Tag: {options.tag})")
        
        return task_id
    
    def map(self, func: Callable, items: List[Any], 
           options: Optional[ParallelTaskOptions] = None) -> List[str]:
        """
        Führt eine Funktion parallel für alle Elemente einer Liste aus.
        
        Args:
            func: Auszuführende Funktion
            items: Liste von Elementen
            options: Aufgabenoptionen
            
        Returns:
            Liste von Aufgaben-IDs
        """
        task_ids = []
        for item in items:
            task_id = self.submit(func, item, options=options)
            task_ids.append(task_id)
        
        return task_ids
    
    def wait_for(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, ParallelTaskResult]:
        """
        Wartet auf den Abschluss der angegebenen Aufgaben.
        
        Args:
            task_ids: Liste von Aufgaben-IDs
            timeout: Timeout in Sekunden
            
        Returns:
            Wörterbuch mit Aufgaben-IDs und -Ergebnissen
        """
        start_time = time.time()
        results = {}
        
        # Aufgaben zum Warten
        pending_tasks = set(task_ids)
        
        while pending_tasks and (timeout is None or time.time() - start_time < timeout):
            # Prüfe abgeschlossene Aufgaben
            with self.task_lock:
                for task_id in list(pending_tasks):
                    if task_id in self.task_results:
                        results[task_id] = self.task_results[task_id]
                        pending_tasks.remove(task_id)
            
            # Alle Aufgaben abgeschlossen?
            if not pending_tasks:
                break
                
            # Kurze Pause, um CPU-Last zu reduzieren
            time.sleep(0.01)
        
        # Timeout erreicht?
        if pending_tasks and timeout is not None:
            logger.warning(f"Timeout beim Warten auf Aufgaben: {pending_tasks}")
        
        return results
    
    def wait_all(self, timeout: Optional[float] = None) -> Dict[str, ParallelTaskResult]:
        """
        Wartet auf den Abschluss aller eingereichten Aufgaben.
        
        Args:
            timeout: Timeout in Sekunden
            
        Returns:
            Wörterbuch mit Aufgaben-IDs und -Ergebnissen
        """
        with self.task_lock:
            task_ids = list(self.tasks.keys())
        
        return self.wait_for(task_ids, timeout)
    
    def cancel(self, task_id: str) -> bool:
        """
        Bricht eine Aufgabe ab.
        
        Args:
            task_id: Aufgaben-ID
            
        Returns:
            True, wenn die Aufgabe abgebrochen wurde, sonst False
        """
        with self.task_lock:
            if task_id in self.tasks:
                future = self.tasks[task_id]
                result = future.cancel()
                
                if result:
                    # Erstelle Abbruch-Ergebnis
                    self.task_results[task_id] = ParallelTaskResult(
                        task_id=task_id,
                        success=False,
                        error=TaskCancellationError("Aufgabe abgebrochen"),
                        completion_timestamp=time.time()
                    )
                    
                    # Entferne Aufgabe aus aktiver Verfolgung
                    self.tasks.pop(task_id, None)
                
                return result
            
            return False
    
    def get_result(self, task_id: str) -> Optional[ParallelTaskResult]:
        """
        Gibt das Ergebnis einer Aufgabe zurück.
        
        Args:
            task_id: Aufgaben-ID
            
        Returns:
            Aufgabenergebnis oder None
        """
        with self.task_lock:
            return self.task_results.get(task_id)
    
    def get_pending_count(self) -> int:
        """
        Gibt die Anzahl der ausstehenden Aufgaben zurück.
        
        Returns:
            Anzahl der ausstehenden Aufgaben
        """
        with self.task_lock:
            return len(self.tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gibt Statistiken zum parallelen Ausführer zurück.
        
        Returns:
            Ausführungsstatistiken
        """
        with self.task_lock:
            total_tasks = len(self.task_results)
            successful_tasks = sum(1 for result in self.task_results.values() if result.success)
            failed_tasks = total_tasks - successful_tasks
            pending_tasks = len(self.tasks)
            
            # Berechne durchschnittliche Ausführungszeit
            if total_tasks > 0:
                avg_execution_time = sum(result.execution_time_ms for result in self.task_results.values()) / total_tasks
            else:
                avg_execution_time = 0
            
            return {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "pending_tasks": pending_tasks,
                "avg_execution_time_ms": avg_execution_time,
                "worker_count": self.max_workers,
                "use_processes": self.use_processes
            }
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Fährt den parallelen Ausführer herunter.
        
        Args:
            wait: Ob auf den Abschluss aller Aufgaben gewartet werden soll
        """
        self.is_running = False
        
        # Warte auf ausstehende Aufgaben
        if wait:
            logger.info("Warte auf Abschluss aller ausstehenden Aufgaben...")
            self.wait_all()
        
        # Fahre Executor herunter
        self.executor.shutdown(wait=wait)
        
        logger.info("Paralleler Ausführer heruntergefahren")


class ParallelBlock:
    """Kontext-Manager für parallel auszuführende Codeblöcke"""
    
    def __init__(self, executor: Optional[ParallelExecutor] = None, options: Optional[ParallelTaskOptions] = None):
        """
        Initialisiert einen neuen parallelen Block.
        
        Args:
            executor: Paralleler Ausführer (Standard: globale Instanz)
            options: Aufgabenoptionen
        """
        self.executor = executor or _get_global_executor()
        self.options = options or ParallelTaskOptions()
        self.task_ids = []
    
    def __enter__(self) -> 'ParallelBlock':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Warte auf Abschluss aller Aufgaben
        self.executor.wait_for(self.task_ids)
    
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """
        Reicht eine Aufgabe zur parallelen Ausführung ein.
        
        Args:
            func: Auszuführende Funktion
            *args: Positionsargumente
            **kwargs: Schlüsselwortargumente
            
        Returns:
            Aufgaben-ID
        """
        task_id = self.executor.submit(func, *args, options=self.options, **kwargs)
        self.task_ids.append(task_id)
        return task_id
    
    def map(self, func: Callable, items: List[Any]) -> List[str]:
        """
        Führt eine Funktion parallel für alle Elemente einer Liste aus.
        
        Args:
            func: Auszuführende Funktion
            items: Liste von Elementen
            
        Returns:
            Liste von Aufgaben-IDs
        """
        task_ids = self.executor.map(func, items, options=self.options)
        self.task_ids.extend(task_ids)
        return task_ids
    
    def get_results(self) -> Dict[str, ParallelTaskResult]:
        """
        Gibt die Ergebnisse aller Aufgaben im Block zurück.
        
        Returns:
            Wörterbuch mit Aufgaben-IDs und -Ergebnissen
        """
        return self.executor.wait_for(self.task_ids)


# Globale Instanz des parallelen Ausführers
_global_executor = None

def _get_global_executor() -> ParallelExecutor:
    """
    Gibt die globale Instanz des parallelen Ausführers zurück.
    
    Returns:
        Paralleler Ausführer
    """
    global _global_executor
    
    if _global_executor is None:
        _global_executor = ParallelExecutor()
    
    return _global_executor

def parallel(func: Optional[Callable] = None, *, 
            max_workers: Optional[int] = None, 
            use_processes: bool = False,
            timeout: Optional[float] = None,
            max_retries: int = 0) -> Callable:
    """
    Dekorator für parallel auszuführende Funktionen.
    
    Args:
        func: Zu dekorierende Funktion
        max_workers: Maximale Anzahl von Workern
        use_processes: Ob Prozesse statt Threads verwendet werden sollen
        timeout: Timeout in Sekunden
        max_retries: Maximale Anzahl von Wiederholungen
        
    Returns:
        Dekorierte Funktion
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            # Konfiguriere Optionen
            options = ParallelTaskOptions(
                timeout_seconds=timeout,
                use_processes=use_processes,
                max_retries=max_retries
            )
            
            # Hole globalen Executor oder erstelle spezifischen
            if max_workers is not None:
                executor = ParallelExecutor(max_workers=max_workers, use_processes=use_processes)
            else:
                executor = _get_global_executor()
            
            # Reiche Aufgabe ein und warte auf Ergebnis
            task_id = executor.submit(f, *args, options=options, **kwargs)
            results = executor.wait_for([task_id])
            
            if task_id in results:
                result = results[task_id]
                if result.success:
                    return result.result
                elif result.error:
                    raise result.error
                    
            # Sollte nicht erreicht werden
            raise ParallelExecutionError("Unbekannter Fehler bei paralleler Ausführung")
            
        return wrapper
    
    # Ermöglicht Verwendung mit oder ohne Klammern
    if func is None:
        return decorator
    return decorator(func)

def parallel_map(func: Callable, items: List[Any], 
                max_workers: Optional[int] = None,
                timeout: Optional[float] = None) -> List[Any]:
    """
    Führt eine Funktion parallel für alle Elemente einer Liste aus.
    
    Args:
        func: Auszuführende Funktion
        items: Liste von Elementen
        max_workers: Maximale Anzahl von Workern
        timeout: Timeout in Sekunden
        
    Returns:
        Liste von Ergebnissen
    """
    # Konfiguriere Optionen
    options = ParallelTaskOptions(timeout_seconds=timeout)
    
    # Hole globalen Executor oder erstelle spezifischen
    if max_workers is not None:
        executor = ParallelExecutor(max_workers=max_workers)
    else:
        executor = _get_global_executor()
    
    # Reiche Aufgaben ein
    task_ids = executor.map(func, items, options=options)
    
    # Warte auf Ergebnisse
    results = executor.wait_for(task_ids, timeout=timeout)
    
    # Sammle Ergebnisse in der richtigen Reihenfolge
    return [results[task_id].result if task_id in results and results[task_id].success else None 
            for task_id in task_ids]

def shutdown_parallel_executor(wait: bool = True) -> None:
    """
    Fährt den globalen parallelen Ausführer herunter.
    
    Args:
        wait: Ob auf den Abschluss aller Aufgaben gewartet werden soll
    """
    global _global_executor
    
    if _global_executor is not None:
        _global_executor.shutdown(wait=wait)
        _global_executor = None
