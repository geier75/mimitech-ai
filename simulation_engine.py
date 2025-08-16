"""
Simulation Engine für das M-PRIME Framework

Dieses Modul implementiert die Simulationsengine für das M-PRIME Framework.
Es ermöglicht die Durchführung von Simulationen und Analysen für das MISO-System.

Version: 1.0.0

Standardisierte Entry-Points:
- init(): Initialisiert die Engine (keine Parameter benötigt)
- configure(config): Konfiguriert die Engine mit spezifischen Einstellungen
- start(workers): Startet die Engine mit einer bestimmten Anzahl an Workern
"""

import logging
import time
import threading
import queue
from enum import Enum
from typing import Dict, List, Callable, Any, Optional, Tuple

# Logger konfigurieren
logger = logging.getLogger("miso.mprime.simulation_engine")

# Globale Engine-Instanz
_engine = None

# Standardisierte Entry-Points
def init():
    """Standardisierter Entry-Point: Initialisiert die SimulationEngine"""
    global _engine
    if _engine is None:
        _engine = SimulationEngine()
        logger.info("SimulationEngine erfolgreich initialisiert")
    return _engine

def configure(config=None):
    """Standardisierter Entry-Point: Konfiguriert die SimulationEngine
    
    Args:
        config (dict): Konfigurationsobjekt mit Parametern
            - max_workers (int): Anzahl der Worker-Threads
            - log_level (str): Logging-Level (INFO, DEBUG, etc.)
    """
    global _engine
    
    if _engine is None:
        _engine = init()
        
    if config is None:
        config = {}
        
    # Konfiguriere max_workers
    if 'max_workers' in config:
        _engine.max_workers = config['max_workers']
    
    # Konfiguriere Logging
    if 'log_level' in config:
        logger.setLevel(config['log_level'])
    
    logger.info(f"SimulationEngine konfiguriert: {config}")
    return _engine

def start(workers=4):
    """Standardisierter Entry-Point: Startet die SimulationEngine
    
    Args:
        workers (int): Anzahl der Worker-Threads
    """
    global _engine
    
    if _engine is None:
        _engine = SimulationEngine(max_workers=workers)
    elif not _engine.running:
        _engine.max_workers = workers
        _engine.start()
        
    logger.info(f"SimulationEngine gestartet mit {workers} Workern")
    return _engine

def boot():
    """Alias für init() für Kompatibilität mit Boot-Konvention"""
    return init()

def setup():
    """Standardisierter Entry-Point: Richtet die SimulationEngine mit Standardwerten ein"""
    engine = init()
    logger.info("SimulationEngine eingerichtet")
    return engine

def activate():
    """Standardisierter Entry-Point: Aktiviert die SimulationEngine mit Standardeinstellungen"""
    engine = init()
    # Führe start() aus, um die Engine zu starten
    engine = start()
    logger.info("SimulationEngine aktiviert")
    return engine

class SimulationStatus(Enum):
    """Status einer Simulation"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SimulationPriority(Enum):
    """Priorität einer Simulation"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class SimulationResult:
    """Ergebnis einer Simulation"""
    
    def __init__(self, simulation_id: str, status: SimulationStatus, data: Any = None, error: Exception = None):
        """
        Initialisiert ein SimulationResult-Objekt.
        
        Args:
            simulation_id: ID der Simulation
            status: Status der Simulation
            data: Ergebnisdaten der Simulation
            error: Fehler, falls aufgetreten
        """
        self.simulation_id = simulation_id
        self.status = status
        self.data = data
        self.error = error
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        """String-Repräsentation des Simulationsergebnisses"""
        return f"SimulationResult(id={self.simulation_id}, status={self.status.value}, timestamp={self.timestamp})"

class Simulation:
    """Repräsentiert eine Simulation"""
    
    def __init__(self, simulation_id: str, func: Callable, args: Tuple = (), 
                 kwargs: Dict = None, priority: SimulationPriority = SimulationPriority.MEDIUM,
                 timeout: Optional[float] = None):
        """
        Initialisiert ein Simulation-Objekt.
        
        Args:
            simulation_id: Eindeutige ID der Simulation
            func: Die auszuführende Simulationsfunktion
            args: Positionsargumente für die Funktion
            kwargs: Schlüsselwortargumente für die Funktion
            priority: Priorität der Simulation
            timeout: Timeout in Sekunden, nach dem die Simulation abgebrochen wird
        """
        self.simulation_id = simulation_id
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.priority = priority
        self.timeout = timeout
        self.status = SimulationStatus.PENDING
        self.result = None
        self.start_time = None
        self.end_time = None
    
    def __lt__(self, other):
        """Vergleich für Prioritätsqueue"""
        if not isinstance(other, Simulation):
            return NotImplemented
        return self.priority.value > other.priority.value  # Höhere Priorität zuerst

class SimulationEngine:
    """
    Engine zur Durchführung von Simulationen für das M-PRIME Framework.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialisiert die SimulationEngine.
        
        Args:
            max_workers: Maximale Anzahl gleichzeitiger Simulationen
        """
        self.logger = logging.getLogger("miso.mprime.simulation_engine")
        self.max_workers = max_workers
        self.simulations: Dict[str, Simulation] = {}
        self.results: Dict[str, SimulationResult] = {}
        self.queue = queue.PriorityQueue()
        self.workers: List[threading.Thread] = []
        self.running = False
        self.lock = threading.Lock()
        self.logger.info(f"SimulationEngine initialisiert mit {max_workers} Workern")
    
    def start(self):
        """Startet die SimulationEngine"""
        with self.lock:
            if self.running:
                self.logger.warning("SimulationEngine läuft bereits")
                return
            
            self.running = True
            self.workers = []
            
            for i in range(self.max_workers):
                worker = threading.Thread(target=self._worker_loop, name=f"SimulationWorker-{i}")
                worker.daemon = True
                worker.start()
                self.workers.append(worker)
            
            self.logger.info("SimulationEngine gestartet")
    
    def stop(self):
        """Stoppt die SimulationEngine"""
        with self.lock:
            if not self.running:
                self.logger.warning("SimulationEngine läuft nicht")
                return
            
            self.running = False
            
            # Leere die Queue
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break
            
            # Warte auf Worker-Threads
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=1.0)
            
            self.workers = []
            self.logger.info("SimulationEngine gestoppt")
    
    def _worker_loop(self):
        """Worker-Loop für die Ausführung von Simulationen"""
        while self.running:
            try:
                # Hole die nächste Simulation aus der Queue
                simulation = self.queue.get(timeout=1.0)
                
                # Führe die Simulation aus
                self._run_simulation(simulation)
                
                # Markiere die Aufgabe als erledigt
                self.queue.task_done()
            
            except queue.Empty:
                # Keine Simulation in der Queue, warte auf neue
                continue
            
            except Exception as e:
                self.logger.error(f"Fehler im Worker-Thread: {e}")
    
    def _run_simulation(self, simulation: Simulation):
        """
        Führt eine Simulation aus.
        
        Args:
            simulation: Die auszuführende Simulation
        """
        simulation_id = simulation.simulation_id
        self.logger.info(f"Starte Simulation {simulation_id}")
        
        # Aktualisiere den Status
        with self.lock:
            simulation.status = SimulationStatus.RUNNING
            simulation.start_time = time.time()
        
        try:
            # Führe die Simulationsfunktion aus
            if simulation.timeout is not None:
                # Timeout-Handling mit einem separaten Thread
                result_queue = queue.Queue()
                
                def target():
                    try:
                        result = simulation.func(*simulation.args, **simulation.kwargs)
                        result_queue.put((True, result))
                    except Exception as e:
                        result_queue.put((False, e))
                
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout=simulation.timeout)
                
                if thread.is_alive():
                    # Timeout erreicht
                    self.logger.warning(f"Simulation {simulation_id} hat das Timeout von {simulation.timeout}s erreicht")
                    with self.lock:
                        simulation.status = SimulationStatus.CANCELLED
                        simulation.end_time = time.time()
                        self.results[simulation_id] = SimulationResult(
                            simulation_id=simulation_id,
                            status=SimulationStatus.CANCELLED,
                            error=TimeoutError(f"Simulation timeout nach {simulation.timeout}s")
                        )
                    return
                
                # Hole das Ergebnis
                success, result = result_queue.get()
                
                if success:
                    # Erfolgreiche Simulation
                    with self.lock:
                        simulation.status = SimulationStatus.COMPLETED
                        simulation.end_time = time.time()
                        simulation.result = result
                        self.results[simulation_id] = SimulationResult(
                            simulation_id=simulation_id,
                            status=SimulationStatus.COMPLETED,
                            data=result
                        )
                    self.logger.info(f"Simulation {simulation_id} erfolgreich abgeschlossen")
                else:
                    # Fehler in der Simulation
                    error = result
                    with self.lock:
                        simulation.status = SimulationStatus.FAILED
                        simulation.end_time = time.time()
                        self.results[simulation_id] = SimulationResult(
                            simulation_id=simulation_id,
                            status=SimulationStatus.FAILED,
                            error=error
                        )
                    self.logger.error(f"Fehler in Simulation {simulation_id}: {error}")
            
            else:
                # Ohne Timeout
                result = simulation.func(*simulation.args, **simulation.kwargs)
                
                with self.lock:
                    simulation.status = SimulationStatus.COMPLETED
                    simulation.end_time = time.time()
                    simulation.result = result
                    self.results[simulation_id] = SimulationResult(
                        simulation_id=simulation_id,
                        status=SimulationStatus.COMPLETED,
                        data=result
                    )
                self.logger.info(f"Simulation {simulation_id} erfolgreich abgeschlossen")
        
        except Exception as e:
            # Fehler in der Simulation
            with self.lock:
                simulation.status = SimulationStatus.FAILED
                simulation.end_time = time.time()
                self.results[simulation_id] = SimulationResult(
                    simulation_id=simulation_id,
                    status=SimulationStatus.FAILED,
                    error=e
                )
            self.logger.error(f"Fehler in Simulation {simulation_id}: {e}")
    
    def submit_simulation(self, simulation_id: str, func: Callable, *args, 
                         priority: SimulationPriority = SimulationPriority.MEDIUM,
                         timeout: Optional[float] = None, **kwargs) -> str:
        """
        Reicht eine Simulation ein.
        
        Args:
            simulation_id: Eindeutige ID der Simulation
            func: Die auszuführende Simulationsfunktion
            *args: Positionsargumente für die Funktion
            priority: Priorität der Simulation
            timeout: Timeout in Sekunden, nach dem die Simulation abgebrochen wird
            **kwargs: Schlüsselwortargumente für die Funktion
            
        Returns:
            str: Die ID der eingereichten Simulation
        """
        with self.lock:
            if simulation_id in self.simulations:
                self.logger.warning(f"Simulation mit ID {simulation_id} existiert bereits, generiere neue ID")
                simulation_id = f"{simulation_id}_{int(time.time())}"
            
            simulation = Simulation(
                simulation_id=simulation_id,
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                timeout=timeout
            )
            
            self.simulations[simulation_id] = simulation
            self.queue.put(simulation)
            
            self.logger.info(f"Simulation {simulation_id} mit Priorität {priority.name} eingereicht")
            return simulation_id
    
    def get_simulation_status(self, simulation_id: str) -> Optional[SimulationStatus]:
        """
        Gibt den Status einer Simulation zurück.
        
        Args:
            simulation_id: ID der Simulation
            
        Returns:
            SimulationStatus: Der Status der Simulation oder None, wenn nicht gefunden
        """
        with self.lock:
            simulation = self.simulations.get(simulation_id)
            return simulation.status if simulation else None
    
    def get_simulation_result(self, simulation_id: str) -> Optional[SimulationResult]:
        """
        Gibt das Ergebnis einer Simulation zurück.
        
        Args:
            simulation_id: ID der Simulation
            
        Returns:
            SimulationResult: Das Ergebnis der Simulation oder None, wenn nicht gefunden
        """
        with self.lock:
            return self.results.get(simulation_id)
    
    def cancel_simulation(self, simulation_id: str) -> bool:
        """
        Bricht eine Simulation ab.
        
        Args:
            simulation_id: ID der Simulation
            
        Returns:
            bool: True, wenn die Simulation abgebrochen wurde, False sonst
        """
        with self.lock:
            simulation = self.simulations.get(simulation_id)
            
            if not simulation:
                self.logger.warning(f"Simulation {simulation_id} nicht gefunden")
                return False
            
            if simulation.status == SimulationStatus.COMPLETED or simulation.status == SimulationStatus.FAILED:
                self.logger.warning(f"Simulation {simulation_id} ist bereits abgeschlossen oder fehlgeschlagen")
                return False
            
            if simulation.status == SimulationStatus.CANCELLED:
                self.logger.warning(f"Simulation {simulation_id} ist bereits abgebrochen")
                return True
            
            # Wenn die Simulation noch in der Queue ist, entferne sie
            # (Dies ist nicht trivial mit PriorityQueue, daher markieren wir sie nur als abgebrochen)
            simulation.status = SimulationStatus.CANCELLED
            self.results[simulation_id] = SimulationResult(
                simulation_id=simulation_id,
                status=SimulationStatus.CANCELLED
            )
            
            self.logger.info(f"Simulation {simulation_id} abgebrochen")
            return True
    
    def get_all_simulations(self) -> Dict[str, Simulation]:
        """
        Gibt alle Simulationen zurück.
        
        Returns:
            Dict[str, Simulation]: Ein Dictionary mit allen Simulationen
        """
        with self.lock:
            return self.simulations.copy()
    
    def get_all_results(self) -> Dict[str, SimulationResult]:
        """
        Gibt alle Simulationsergebnisse zurück.
        
        Returns:
            Dict[str, SimulationResult]: Ein Dictionary mit allen Simulationsergebnissen
        """
        with self.lock:
            return self.results.copy()
    
    def clear_completed_simulations(self, older_than: Optional[float] = None) -> int:
        """
        Löscht abgeschlossene Simulationen.
        
        Args:
            older_than: Wenn angegeben, werden nur Simulationen gelöscht, die älter als dieser Zeitstempel sind
            
        Returns:
            int: Anzahl der gelöschten Simulationen
        """
        with self.lock:
            to_delete = []
            
            for simulation_id, simulation in self.simulations.items():
                if simulation.status in [SimulationStatus.COMPLETED, SimulationStatus.FAILED, SimulationStatus.CANCELLED]:
                    if older_than is None or (simulation.end_time and simulation.end_time < older_than):
                        to_delete.append(simulation_id)
            
            for simulation_id in to_delete:
                del self.simulations[simulation_id]
                if simulation_id in self.results:
                    del self.results[simulation_id]
            
            self.logger.info(f"{len(to_delete)} abgeschlossene Simulationen gelöscht")
            return len(to_delete)
    
    def run_simulation(self, func: Callable, *args, **kwargs) -> Any:
        """
        Führt eine Simulation synchron aus.
        
        Args:
            func: Die auszuführende Simulationsfunktion
            *args, **kwargs: Argumente für die Funktion
            
        Returns:
            Das Ergebnis der Simulation
        """
        self.logger.info(f"Führe Simulation {func.__name__} synchron aus")
        
        try:
            result = func(*args, **kwargs)
            self.logger.info(f"Synchrone Simulation {func.__name__} erfolgreich abgeschlossen")
            return result
        except Exception as e:
            self.logger.error(f"Fehler bei synchroner Simulation {func.__name__}: {e}")
            raise
