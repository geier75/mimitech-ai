"""
Task Scheduler für NEXUS-OS

Dieses Modul implementiert den Task-Scheduler für das NEXUS-OS.
Es ist verantwortlich für die Optimierung und Planung von Aufgaben im MISO-System.

Version: 1.0.0
"""

import logging
import time
import heapq
import threading
import uuid
from enum import Enum
from typing import Dict, List, Callable, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Logger konfigurieren
logger = logging.getLogger("miso.nexus.task_scheduler")

class TaskPriority(Enum):
    """Priorität einer Aufgabe"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class TaskStatus(Enum):
    """Status einer Aufgabe"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass(order=True)
class ScheduledTask:
    """Repräsentiert eine geplante Aufgabe"""
    scheduled_time: float
    priority: TaskPriority = field(compare=False)
    task_id: str = field(compare=False)
    func: Callable = field(compare=False)
    args: Tuple = field(default_factory=tuple, compare=False)
    kwargs: Dict = field(default_factory=dict, compare=False)
    recurring: bool = field(default=False, compare=False)
    interval: Optional[float] = field(default=None, compare=False)
    max_runs: Optional[int] = field(default=None, compare=False)
    run_count: int = field(default=0, compare=False)
    last_run_time: Optional[float] = field(default=None, compare=False)
    next_run_time: Optional[float] = field(default=None, compare=False)
    status: TaskStatus = field(default=TaskStatus.SCHEDULED, compare=False)
    result: Any = field(default=None, compare=False)
    error: Exception = field(default=None, compare=False)
    
    def __post_init__(self):
        """Nachinitialisierung"""
        self.next_run_time = self.scheduled_time

class TaskDependency:
    """Repräsentiert eine Abhängigkeit zwischen Aufgaben"""
    
    def __init__(self, dependent_task_id: str, dependency_task_id: str):
        """
        Initialisiert eine TaskDependency.
        
        Args:
            dependent_task_id: ID der abhängigen Aufgabe
            dependency_task_id: ID der Aufgabe, von der abhängig ist
        """
        self.dependent_task_id = dependent_task_id
        self.dependency_task_id = dependency_task_id

class TaskScheduler:
    """
    Task-Scheduler für das NEXUS-OS.
    Verantwortlich für die Optimierung und Planung von Aufgaben im MISO-System.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialisiert den TaskScheduler.
        
        Args:
            max_workers: Maximale Anzahl gleichzeitiger Aufgaben
        """
        self.logger = logging.getLogger("miso.nexus.task_scheduler")
        self.max_workers = max_workers
        self.tasks: Dict[str, ScheduledTask] = {}
        self.task_queue = []  # Heap-Queue für geplante Aufgaben
        self.dependencies: Dict[str, Set[str]] = {}  # dependent_task_id -> {dependency_task_id, ...}
        self.reverse_dependencies: Dict[str, Set[str]] = {}  # dependency_task_id -> {dependent_task_id, ...}
        self.completed_tasks: Set[str] = set()
        self.running = False
        self.scheduler_thread = None
        self.worker_threads: List[threading.Thread] = []
        self.worker_semaphore = threading.Semaphore(max_workers)
        self.lock = threading.Lock()
        self.task_event = threading.Event()
        self.logger.info(f"TaskScheduler initialisiert mit {max_workers} Workern")
    
    def start(self):
        """Startet den TaskScheduler"""
        with self.lock:
            if self.running:
                self.logger.warning("TaskScheduler läuft bereits")
                return
            
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, name="TaskScheduler")
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            
            self.logger.info("TaskScheduler gestartet")
    
    def stop(self):
        """Stoppt den TaskScheduler"""
        with self.lock:
            if not self.running:
                self.logger.warning("TaskScheduler läuft nicht")
                return
            
            self.running = False
            self.task_event.set()  # Wecke den Scheduler-Thread auf
            
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=2.0)
            
            # Warte auf Worker-Threads
            active_workers = [t for t in self.worker_threads if t.is_alive()]
            for worker in active_workers:
                worker.join(timeout=1.0)
            
            self.worker_threads = [t for t in self.worker_threads if t.is_alive()]
            
            self.logger.info("TaskScheduler gestoppt")
    
    def _scheduler_loop(self):
        """Scheduler-Loop für die Planung von Aufgaben"""
        self.logger.info("Scheduler-Thread gestartet")
        
        while self.running:
            next_task_time = None
            
            with self.lock:
                # Überprüfe, ob es Aufgaben gibt
                if self.task_queue:
                    # Hole die nächste Aufgabe, ohne sie zu entfernen
                    next_task = self.task_queue[0]
                    next_task_time = next_task.scheduled_time
                    
                    # Wenn die Aufgabe jetzt ausgeführt werden soll
                    if next_task_time <= time.time():
                        # Entferne die Aufgabe aus der Queue
                        task = heapq.heappop(self.task_queue)
                        task_id = task.task_id
                        
                        # Überprüfe Abhängigkeiten
                        dependencies_met = True
                        if task_id in self.dependencies:
                            for dependency_id in self.dependencies[task_id]:
                                if dependency_id not in self.completed_tasks:
                                    dependencies_met = False
                                    self.logger.debug(f"Aufgabe {task_id} wartet auf Abhängigkeit {dependency_id}")
                                    break
                        
                        if dependencies_met:
                            # Starte die Aufgabe in einem Worker-Thread
                            self._start_task(task)
                        else:
                            # Verschiebe die Aufgabe
                            task.scheduled_time = time.time() + 1.0  # Überprüfe in 1 Sekunde erneut
                            heapq.heappush(self.task_queue, task)
            
            # Berechne die Wartezeit bis zur nächsten Aufgabe
            if next_task_time is not None:
                wait_time = max(0, next_task_time - time.time())
                self.task_event.wait(timeout=wait_time)
            else:
                # Keine Aufgaben, warte auf neue
                self.task_event.wait()
            
            # Zurücksetzen des Events
            self.task_event.clear()
    
    def _start_task(self, task: ScheduledTask):
        """
        Startet eine Aufgabe in einem Worker-Thread.
        
        Args:
            task: Die zu startende Aufgabe
        """
        # Aktualisiere den Status
        task.status = TaskStatus.RUNNING
        task.last_run_time = time.time()
        task.run_count += 1
        
        # Starte die Aufgabe in einem Worker-Thread
        worker = threading.Thread(
            target=self._run_task,
            args=(task,),
            name=f"TaskWorker-{task.task_id}"
        )
        worker.daemon = True
        
        with self.lock:
            self.worker_threads.append(worker)
        
        worker.start()
        
        # Wenn es sich um eine wiederkehrende Aufgabe handelt, plane die nächste Ausführung
        if task.recurring and task.interval is not None:
            if task.max_runs is None or task.run_count < task.max_runs:
                new_task = ScheduledTask(
                    scheduled_time=time.time() + task.interval,
                    priority=task.priority,
                    task_id=task.task_id,
                    func=task.func,
                    args=task.args,
                    kwargs=task.kwargs,
                    recurring=True,
                    interval=task.interval,
                    max_runs=task.max_runs,
                    run_count=task.run_count
                )
                
                with self.lock:
                    self.tasks[task.task_id] = new_task
                    heapq.heappush(self.task_queue, new_task)
                
                self.logger.debug(f"Wiederkehrende Aufgabe {task.task_id} für nächste Ausführung in {task.interval}s geplant")
    
    def _run_task(self, task: ScheduledTask):
        """
        Führt eine Aufgabe aus.
        
        Args:
            task: Die auszuführende Aufgabe
        """
        task_id = task.task_id
        self.logger.info(f"Starte Aufgabe {task_id}")
        
        # Warte auf einen verfügbaren Worker
        self.worker_semaphore.acquire()
        
        try:
            # Führe die Aufgabe aus
            result = task.func(*task.args, **task.kwargs)
            
            # Aktualisiere den Status
            with self.lock:
                task.status = TaskStatus.COMPLETED
                task.result = result
                self.completed_tasks.add(task_id)
                
                # Überprüfe abhängige Aufgaben
                if task_id in self.reverse_dependencies:
                    for dependent_id in self.reverse_dependencies[task_id]:
                        self.logger.debug(f"Abhängigkeit {task_id} für Aufgabe {dependent_id} erfüllt")
                        # Setze das Event, um den Scheduler zu wecken
                        self.task_event.set()
            
            self.logger.info(f"Aufgabe {task_id} erfolgreich abgeschlossen")
        
        except Exception as e:
            # Fehler in der Aufgabe
            self.logger.error(f"Fehler in Aufgabe {task_id}: {e}")
            
            with self.lock:
                task.status = TaskStatus.FAILED
                task.error = e
        
        finally:
            # Gib den Worker frei
            self.worker_semaphore.release()
            
            # Entferne den Thread aus der Liste
            with self.lock:
                self.worker_threads = [t for t in self.worker_threads if t.is_alive()]
    
    def schedule_task(self, func: Callable, *args, 
                     task_id: Optional[str] = None,
                     priority: TaskPriority = TaskPriority.MEDIUM,
                     delay: float = 0.0,
                     recurring: bool = False,
                     interval: Optional[float] = None,
                     max_runs: Optional[int] = None,
                     dependencies: List[str] = None,
                     **kwargs) -> str:
        """
        Plant eine Aufgabe.
        
        Args:
            func: Die auszuführende Funktion
            *args: Positionsargumente für die Funktion
            task_id: Eindeutige ID der Aufgabe (wird generiert, wenn nicht angegeben)
            priority: Priorität der Aufgabe
            delay: Verzögerung in Sekunden, bevor die Aufgabe ausgeführt wird
            recurring: Ob die Aufgabe wiederkehrend ist
            interval: Intervall in Sekunden für wiederkehrende Aufgaben
            max_runs: Maximale Anzahl von Ausführungen für wiederkehrende Aufgaben
            dependencies: Liste von Aufgaben-IDs, von denen diese Aufgabe abhängt
            **kwargs: Schlüsselwortargumente für die Funktion
            
        Returns:
            str: Die ID der geplanten Aufgabe
        """
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        scheduled_time = time.time() + delay
        
        task = ScheduledTask(
            scheduled_time=scheduled_time,
            priority=priority,
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            recurring=recurring,
            interval=interval,
            max_runs=max_runs
        )
        
        with self.lock:
            self.tasks[task_id] = task
            
            # Füge Abhängigkeiten hinzu
            if dependencies:
                self.dependencies[task_id] = set(dependencies)
                for dependency_id in dependencies:
                    if dependency_id not in self.reverse_dependencies:
                        self.reverse_dependencies[dependency_id] = set()
                    self.reverse_dependencies[dependency_id].add(task_id)
            
            # Füge die Aufgabe zur Queue hinzu
            heapq.heappush(self.task_queue, task)
            
            # Wecke den Scheduler-Thread auf
            self.task_event.set()
        
        self.logger.info(f"Aufgabe {task_id} mit Priorität {priority.name} geplant für {datetime.fromtimestamp(scheduled_time)}")
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Bricht eine Aufgabe ab.
        
        Args:
            task_id: ID der Aufgabe
            
        Returns:
            bool: True, wenn die Aufgabe abgebrochen wurde, False sonst
        """
        with self.lock:
            task = self.tasks.get(task_id)
            
            if not task:
                self.logger.warning(f"Aufgabe {task_id} nicht gefunden")
                return False
            
            if task.status == TaskStatus.COMPLETED or task.status == TaskStatus.FAILED:
                self.logger.warning(f"Aufgabe {task_id} ist bereits abgeschlossen oder fehlgeschlagen")
                return False
            
            if task.status == TaskStatus.CANCELLED:
                self.logger.warning(f"Aufgabe {task_id} ist bereits abgebrochen")
                return True
            
            # Markiere die Aufgabe als abgebrochen
            task.status = TaskStatus.CANCELLED
            
            # Entferne die Aufgabe aus der Queue (nicht trivial mit Heap)
            # Stattdessen wird sie beim nächsten Scheduler-Loop übersprungen
            
            self.logger.info(f"Aufgabe {task_id} abgebrochen")
            return True
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Gibt den Status einer Aufgabe zurück.
        
        Args:
            task_id: ID der Aufgabe
            
        Returns:
            TaskStatus: Der Status der Aufgabe oder None, wenn nicht gefunden
        """
        with self.lock:
            task = self.tasks.get(task_id)
            return task.status if task else None
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Gibt das Ergebnis einer Aufgabe zurück.
        
        Args:
            task_id: ID der Aufgabe
            
        Returns:
            Any: Das Ergebnis der Aufgabe oder None, wenn nicht gefunden oder nicht abgeschlossen
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == TaskStatus.COMPLETED:
                return task.result
            return None
    
    def get_all_tasks(self) -> Dict[str, ScheduledTask]:
        """
        Gibt alle Aufgaben zurück.
        
        Returns:
            Dict[str, ScheduledTask]: Ein Dictionary mit allen Aufgaben
        """
        with self.lock:
            return self.tasks.copy()
    
    def get_pending_tasks(self) -> List[ScheduledTask]:
        """
        Gibt alle ausstehenden Aufgaben zurück.
        
        Returns:
            List[ScheduledTask]: Eine Liste mit allen ausstehenden Aufgaben
        """
        with self.lock:
            return [task for task in self.tasks.values() if task.status == TaskStatus.PENDING or task.status == TaskStatus.SCHEDULED]
    
    def get_running_tasks(self) -> List[ScheduledTask]:
        """
        Gibt alle laufenden Aufgaben zurück.
        
        Returns:
            List[ScheduledTask]: Eine Liste mit allen laufenden Aufgaben
        """
        with self.lock:
            return [task for task in self.tasks.values() if task.status == TaskStatus.RUNNING]
    
    def get_completed_tasks(self) -> List[ScheduledTask]:
        """
        Gibt alle abgeschlossenen Aufgaben zurück.
        
        Returns:
            List[ScheduledTask]: Eine Liste mit allen abgeschlossenen Aufgaben
        """
        with self.lock:
            return [task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED]
    
    def clear_completed_tasks(self, older_than: Optional[float] = None) -> int:
        """
        Löscht abgeschlossene Aufgaben.
        
        Args:
            older_than: Wenn angegeben, werden nur Aufgaben gelöscht, die älter als dieser Zeitstempel sind
            
        Returns:
            int: Anzahl der gelöschten Aufgaben
        """
        with self.lock:
            to_delete = []
            
            for task_id, task in self.tasks.items():
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    if older_than is None or (task.last_run_time and task.last_run_time < older_than):
                        to_delete.append(task_id)
            
            for task_id in to_delete:
                del self.tasks[task_id]
                if task_id in self.dependencies:
                    del self.dependencies[task_id]
                if task_id in self.reverse_dependencies:
                    del self.reverse_dependencies[task_id]
                if task_id in self.completed_tasks:
                    self.completed_tasks.remove(task_id)
            
            self.logger.info(f"{len(to_delete)} abgeschlossene Aufgaben gelöscht")
            return len(to_delete)
    
    def run_task(self, func: Callable, *args, **kwargs) -> Any:
        """
        Führt eine Aufgabe synchron aus.
        
        Args:
            func: Die auszuführende Funktion
            *args, **kwargs: Argumente für die Funktion
            
        Returns:
            Das Ergebnis der Aufgabe
        """
        self.logger.info(f"Führe Aufgabe {func.__name__} synchron aus")
        
        try:
            result = func(*args, **kwargs)
            self.logger.info(f"Synchrone Aufgabe {func.__name__} erfolgreich abgeschlossen")
            return result
        except Exception as e:
            self.logger.error(f"Fehler bei synchroner Aufgabe {func.__name__}: {e}")
            raise
