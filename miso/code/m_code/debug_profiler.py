#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - M-CODE Debug & Profiler System

Dieses Modul implementiert ein Debugging- und Profilingsystem für M-CODE.
Es ermöglicht die detaillierte Analyse von Ausführungszeiten, Speichernutzung und Codeausführung.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import logging
import time
import threading
import inspect
import traceback
import functools
import cProfile
import pstats
import io
import sys
import gc
import json
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager

# Konfiguriere Logging
logger = logging.getLogger("MISO.m_code.debug_profiler")


@dataclass
class ProfilerConfig:
    """Konfiguration für den Profiler"""
    enabled: bool = True
    log_level: int = logging.INFO
    time_threshold_ms: float = 1.0  # Schwellenwert für Warnungen (in ms)
    memory_threshold_mb: float = 10.0  # Schwellenwert für Speicherwarnungen (in MB)
    trace_calls: bool = True
    include_tensor_ops: bool = True
    include_io_ops: bool = True
    include_system_functions: bool = False
    output_format: str = "text"  # text, json, flame
    output_dir: Optional[str] = None
    max_depth: int = 10
    use_color: bool = True


@dataclass
class ProfilerResult:
    """Ergebnis eines Profiler-Durchlaufs"""
    function_name: str
    execution_time_ms: float
    memory_usage_mb: float
    call_count: int = 1
    tensor_op_count: int = 0
    io_op_count: int = 0
    peak_memory_mb: float = 0.0
    called_by: Optional[str] = None
    stack_trace: Optional[List[str]] = None
    timestamp: float = field(default_factory=time.time)
    additional_info: Dict[str, Any] = field(default_factory=dict)


class DebuggerBreakpoint:
    """Haltepunkt für den Debugger"""
    
    def __init__(self, function_name: str, condition: Optional[str] = None, 
                line_number: Optional[int] = None, once: bool = False):
        """
        Initialisiert einen neuen Haltepunkt.
        
        Args:
            function_name: Name der Funktion
            condition: Optionale Bedingung (als String)
            line_number: Optionale Zeilennummer
            once: Ob der Haltepunkt nur einmal ausgelöst werden soll
        """
        self.function_name = function_name
        self.condition = condition
        self.line_number = line_number
        self.once = once
        self.hit_count = 0
        self.active = True
        self.id = f"bp_{id(self)}"
    
    def hit(self, frame: Any) -> bool:
        """
        Prüft, ob der Haltepunkt getroffen wurde.
        
        Args:
            frame: Aktueller Frame
            
        Returns:
            True, wenn der Haltepunkt getroffen wurde und aktiv ist
        """
        if not self.active:
            return False
            
        # Prüfe Bedingung
        if self.condition:
            try:
                # Evaluiere Bedingung im Kontext des Frames
                condition_met = eval(self.condition, frame.f_globals, frame.f_locals)
                if not condition_met:
                    return False
            except Exception as e:
                logger.error(f"Fehler bei der Auswertung der Bedingung '{self.condition}': {e}")
                return False
        
        # Prüfe Zeilennummer
        if self.line_number is not None and frame.f_lineno != self.line_number:
            return False
        
        # Erhöhe Trefferanzahl
        self.hit_count += 1
        
        # Deaktiviere, wenn nur einmal auslösen
        if self.once:
            self.active = False
        
        return True


class MCodeProfiler:
    """Profiler für M-CODE"""
    
    def __init__(self, config: Optional[ProfilerConfig] = None):
        """
        Initialisiert einen neuen Profiler.
        
        Args:
            config: Profiler-Konfiguration
        """
        self.config = config or ProfilerConfig()
        self.results = {}
        self.active_profiling = False
        self.start_time = 0.0
        self.initial_memory = 0
        self.total_calls = 0
        self.profiler = None
        self.thread_local = threading.local()
        
        # Logger konfigurieren
        self._configure_logger()
        
        # Ausgabeverzeichnis erstellen
        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)
        
        logger.info(f"M-CODE Profiler initialisiert mit Konfiguration: {self.config}")
    
    def _configure_logger(self) -> None:
        """Konfiguriert den Logger basierend auf der Konfiguration"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.setLevel(self.config.log_level)
        logger.addHandler(handler)
    
    def start(self) -> None:
        """Startet den Profiler"""
        if self.active_profiling:
            logger.warning("Profiler läuft bereits")
            return
            
        self.active_profiling = True
        self.start_time = time.time()
        self.initial_memory = self._get_memory_usage()
        self.total_calls = 0
        self.results = {}
        
        # Starte cProfile, wenn aktiviert
        if self.config.trace_calls:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        
        logger.info("Profiler gestartet")
    
    def stop(self) -> Dict[str, ProfilerResult]:
        """
        Stoppt den Profiler.
        
        Returns:
            Profiler-Ergebnisse
        """
        if not self.active_profiling:
            logger.warning("Profiler ist nicht aktiv")
            return {}
            
        # Stoppe cProfile, wenn aktiviert
        if self.profiler:
            self.profiler.disable()
            
            # Extrahiere Statistiken
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
            ps.print_stats()
            
            # Verarbeite cProfile-Ergebnisse
            profile_text = s.getvalue()
            self._process_cprofile_results(profile_text)
        
        self.active_profiling = False
        
        # Protokolliere zusammenfassende Statistiken
        duration = time.time() - self.start_time
        memory_delta = self._get_memory_usage() - self.initial_memory
        
        logger.info(f"Profiler gestoppt. Laufzeit: {duration:.2f}s, "
                   f"Funktionsaufrufe: {self.total_calls}, "
                   f"Speichernutzung: {memory_delta:.2f} MB")
        
        # Speichere Ergebnisse, wenn Ausgabeverzeichnis angegeben
        if self.config.output_dir:
            self._save_results()
        
        return self.results
    
    def _process_cprofile_results(self, profile_text: str) -> None:
        """
        Verarbeitet die Ergebnisse von cProfile.
        
        Args:
            profile_text: cProfile-Ergebnistext
        """
        # Einfache Verarbeitung des Texts, für fortgeschrittenere Analyse
        # könnte man die pstats-API verwenden
        lines = profile_text.split('\n')
        for line in lines[5:]:  # Überspringe Header
            if not line.strip():
                continue
                
            parts = line.split()
            if len(parts) >= 6:
                try:
                    calls = int(parts[0])
                    time_per_call = float(parts[1])
                    cumtime = float(parts[3])
                    func_name = ' '.join(parts[5:])
                    
                    # Filtere nach Konfiguration
                    if not self.config.include_system_functions and '<' in func_name and '>' in func_name:
                        continue
                        
                    # Aktualisiere oder erstelle Ergebnis
                    if func_name in self.results:
                        self.results[func_name].execution_time_ms += cumtime * 1000
                        self.results[func_name].call_count += calls
                    else:
                        self.results[func_name] = ProfilerResult(
                            function_name=func_name,
                            execution_time_ms=cumtime * 1000,
                            memory_usage_mb=0,  # Kann aus cProfile nicht extrahiert werden
                            call_count=calls
                        )
                    
                    self.total_calls += calls
                except (ValueError, IndexError):
                    continue
    
    def _get_memory_usage(self) -> float:
        """
        Gibt die aktuelle Speichernutzung in MB zurück.
        
        Returns:
            Speichernutzung in MB
        """
        # Erzwinge Garbage Collection, um genaue Werte zu erhalten
        gc.collect()
        
        # Prozessspeichernutzung (plattformunabhängig)
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # In MB
    
    def _save_results(self) -> None:
        """Speichert die Profiler-Ergebnisse"""
        if not self.config.output_dir:
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if self.config.output_format == "json":
            # JSON-Format
            output_file = os.path.join(self.config.output_dir, f"profile_{timestamp}.json")
            with open(output_file, 'w') as f:
                # Konvertiere Ergebnisse in serialisierbares Format
                serializable_results = {}
                for func_name, result in self.results.items():
                    serializable_results[func_name] = {
                        "function_name": result.function_name,
                        "execution_time_ms": result.execution_time_ms,
                        "memory_usage_mb": result.memory_usage_mb,
                        "call_count": result.call_count,
                        "tensor_op_count": result.tensor_op_count,
                        "io_op_count": result.io_op_count,
                        "peak_memory_mb": result.peak_memory_mb,
                        "timestamp": result.timestamp,
                        "additional_info": result.additional_info
                    }
                json.dump(serializable_results, f, indent=2)
        elif self.config.output_format == "flame":
            # Flamegraph-Format (speedscope)
            output_file = os.path.join(self.config.output_dir, f"profile_{timestamp}.speedscope.json")
            # Hier würde die Konvertierung in das speedscope-Format erfolgen
            # Dies erfordert eine komplexere Transformation
            logger.warning("Flamegraph-Ausgabe noch nicht implementiert")
        else:
            # Textformat (Standard)
            output_file = os.path.join(self.config.output_dir, f"profile_{timestamp}.txt")
            with open(output_file, 'w') as f:
                f.write("M-CODE Profiler-Ergebnisse\n")
                f.write("=========================\n\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Gesamtlaufzeit: {time.time() - self.start_time:.2f}s\n")
                f.write(f"Gesamte Funktionsaufrufe: {self.total_calls}\n\n")
                
                # Sortiere nach Ausführungszeit
                sorted_results = sorted(
                    self.results.values(), 
                    key=lambda x: x.execution_time_ms,
                    reverse=True
                )
                
                f.write("Top 20 Funktionen (nach Ausführungszeit):\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Funktion':<40} {'Zeit (ms)':<15} {'Aufrufe':<10} {'Zeit/Aufruf (ms)':<15}\n")
                f.write("-" * 80 + "\n")
                
                for i, result in enumerate(sorted_results[:20]):
                    time_per_call = result.execution_time_ms / max(1, result.call_count)
                    f.write(f"{result.function_name:<40} {result.execution_time_ms:<15.2f} "
                           f"{result.call_count:<10} {time_per_call:<15.2f}\n")
        
        logger.info(f"Profiler-Ergebnisse gespeichert in {output_file}")
    
    @contextmanager
    def profile_function(self, function_name: str) -> None:
        """
        Kontext-Manager zum Profiling einer Funktion.
        
        Args:
            function_name: Name der Funktion
        """
        if not self.config.enabled:
            yield
            return
            
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Erfasse Stack-Trace
            stack = traceback.extract_stack()
            caller = stack[-3] if len(stack) > 2 else None
            caller_name = f"{caller.filename}:{caller.lineno} in {caller.name}" if caller else None
            
            yield
        finally:
            # Berechne Metriken
            execution_time_ms = (time.time() - start_time) * 1000
            memory_delta_mb = self._get_memory_usage() - start_memory
            
            # Erstelle oder aktualisiere Profiler-Ergebnis
            if function_name in self.results:
                self.results[function_name].execution_time_ms += execution_time_ms
                self.results[function_name].memory_usage_mb += memory_delta_mb
                self.results[function_name].call_count += 1
                self.results[function_name].peak_memory_mb = max(
                    self.results[function_name].peak_memory_mb,
                    memory_delta_mb
                )
            else:
                self.results[function_name] = ProfilerResult(
                    function_name=function_name,
                    execution_time_ms=execution_time_ms,
                    memory_usage_mb=memory_delta_mb,
                    called_by=caller_name,
                    stack_trace=[f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in stack[:-1]]
                )
            
            self.total_calls += 1
            
            # Protokolliere Warnung, wenn Schwellenwert überschritten
            if execution_time_ms > self.config.time_threshold_ms:
                logger.warning(f"Langsame Funktion: {function_name} "
                              f"({execution_time_ms:.2f} ms, {memory_delta_mb:.2f} MB)")
            
            # Speicherwarnung
            if memory_delta_mb > self.config.memory_threshold_mb:
                logger.warning(f"Hohe Speichernutzung: {function_name} "
                              f"({memory_delta_mb:.2f} MB)")


class MCodeDebugger:
    """Debugger für M-CODE"""
    
    def __init__(self):
        """Initialisiert einen neuen Debugger"""
        self.breakpoints = {}
        self.watches = {}
        self.is_active = False
        self.step_mode = False
        self.next_line = None
        self.next_frame = None
        self.trace_file = None
        self.trace_enabled = False
        
        logger.info("M-CODE Debugger initialisiert")
    
    def add_breakpoint(self, breakpoint: DebuggerBreakpoint) -> str:
        """
        Fügt einen Haltepunkt hinzu.
        
        Args:
            breakpoint: Haltepunkt
            
        Returns:
            Haltepunkt-ID
        """
        self.breakpoints[breakpoint.id] = breakpoint
        logger.info(f"Haltepunkt hinzugefügt: {breakpoint.function_name} "
                  f"(Zeile: {breakpoint.line_number}, Bedingung: {breakpoint.condition})")
        return breakpoint.id
    
    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """
        Entfernt einen Haltepunkt.
        
        Args:
            breakpoint_id: Haltepunkt-ID
            
        Returns:
            True, wenn der Haltepunkt entfernt wurde, sonst False
        """
        if breakpoint_id in self.breakpoints:
            self.breakpoints.pop(breakpoint_id)
            logger.info(f"Haltepunkt {breakpoint_id} entfernt")
            return True
        
        logger.warning(f"Haltepunkt {breakpoint_id} nicht gefunden")
        return False
    
    def add_watch(self, expression: str, watch_id: Optional[str] = None) -> str:
        """
        Fügt einen Beobachtungsausdruck hinzu.
        
        Args:
            expression: Auszuwertender Ausdruck
            watch_id: Optionale Beobachtungs-ID
            
        Returns:
            Beobachtungs-ID
        """
        watch_id = watch_id or f"watch_{len(self.watches)}"
        self.watches[watch_id] = expression
        logger.info(f"Beobachtung hinzugefügt: {expression} (ID: {watch_id})")
        return watch_id
    
    def remove_watch(self, watch_id: str) -> bool:
        """
        Entfernt einen Beobachtungsausdruck.
        
        Args:
            watch_id: Beobachtungs-ID
            
        Returns:
            True, wenn die Beobachtung entfernt wurde, sonst False
        """
        if watch_id in self.watches:
            self.watches.pop(watch_id)
            logger.info(f"Beobachtung {watch_id} entfernt")
            return True
        
        logger.warning(f"Beobachtung {watch_id} nicht gefunden")
        return False
    
    def activate(self) -> None:
        """Aktiviert den Debugger"""
        if self.is_active:
            logger.warning("Debugger ist bereits aktiv")
            return
            
        self.is_active = True
        
        # Setze Trace-Funktion
        sys.settrace(self._trace_function)
        threading.settrace(self._trace_function)
        
        logger.info("Debugger aktiviert")
    
    def deactivate(self) -> None:
        """Deaktiviert den Debugger"""
        if not self.is_active:
            logger.warning("Debugger ist nicht aktiv")
            return
            
        self.is_active = False
        
        # Deaktiviere Trace-Funktion
        sys.settrace(None)
        threading.settrace(None)
        
        # Schließe Trace-Datei
        if self.trace_file:
            self.trace_file.close()
            self.trace_file = None
        
        logger.info("Debugger deaktiviert")
    
    def enable_trace(self, filename: Optional[str] = None) -> None:
        """
        Aktiviert das Tracing.
        
        Args:
            filename: Dateiname für die Trace-Ausgabe
        """
        self.trace_enabled = True
        
        if filename:
            self.trace_file = open(filename, 'w')
            logger.info(f"Trace-Ausgabe in Datei {filename}")
        
        logger.info("Trace aktiviert")
    
    def disable_trace(self) -> None:
        """Deaktiviert das Tracing"""
        self.trace_enabled = False
        
        # Schließe Trace-Datei
        if self.trace_file:
            self.trace_file.close()
            self.trace_file = None
        
        logger.info("Trace deaktiviert")
    
    def _trace_function(self, frame, event, arg) -> Optional[Callable]:
        """
        Trace-Funktion für den Debugger.
        
        Args:
            frame: Aktueller Frame
            event: Event-Typ
            arg: Zusätzliches Argument
            
        Returns:
            Trace-Funktion oder None
        """
        if not self.is_active:
            return None
            
        # Extrahiere Informationen aus dem Frame
        code = frame.f_code
        function_name = code.co_name
        filename = code.co_filename
        lineno = frame.f_lineno
        
        # Trace-Ausgabe
        if self.trace_enabled:
            trace_line = f"{event}: {filename}:{lineno} in {function_name}"
            
            if event == "call":
                # Funktionsaufruf
                args = inspect.getargvalues(frame)
                args_str = ", ".join(f"{arg}={repr(args.locals[arg])}" for arg in args.args if arg in args.locals)
                trace_line += f"({args_str})"
            elif event == "return":
                # Funktionsrückgabe
                trace_line += f" -> {repr(arg)}"
            
            # Ausgabe in Datei oder Log
            if self.trace_file:
                self.trace_file.write(trace_line + "\n")
                self.trace_file.flush()
            else:
                logger.debug(trace_line)
        
        # Prüfe Haltepunkte
        if event == "line":
            for bp in self.breakpoints.values():
                if bp.function_name == function_name and bp.hit(frame):
                    # Haltepunkt getroffen
                    self._handle_breakpoint(bp, frame)
        
        # Step-Modus
        if self.step_mode and event == "line":
            if (self.next_frame is None or frame == self.next_frame) and \
               (self.next_line is None or lineno == self.next_line):
                # Nächster Schritt erreicht
                self._handle_step(frame)
        
        # Beobachtungen auswerten
        if event == "line" and self.watches:
            for watch_id, expression in self.watches.items():
                try:
                    value = eval(expression, frame.f_globals, frame.f_locals)
                    logger.info(f"Beobachtung {watch_id} ({expression}): {repr(value)}")
                except Exception as e:
                    logger.error(f"Fehler bei der Auswertung der Beobachtung {watch_id} ({expression}): {e}")
        
        return self._trace_function
    
    def _handle_breakpoint(self, bp: DebuggerBreakpoint, frame) -> None:
        """
        Behandelt einen getroffenen Haltepunkt.
        
        Args:
            bp: Getroffener Haltepunkt
            frame: Aktueller Frame
        """
        code = frame.f_code
        function_name = code.co_name
        filename = code.co_filename
        lineno = frame.f_lineno
        
        logger.info(f"Haltepunkt getroffen: {bp.id} in {function_name} ({filename}:{lineno})")
        
        # Hier würde in einer interaktiven Umgebung die Debugger-Steuerung erfolgen
        # Für dieses Beispiel protokollieren wir nur die Variablen
        
        # Lokale Variablen
        locals_str = "\n".join(f"  {name} = {repr(value)}" for name, value in frame.f_locals.items())
        logger.info(f"Lokale Variablen:\n{locals_str}")
    
    def _handle_step(self, frame) -> None:
        """
        Behandelt einen Schritt im Step-Modus.
        
        Args:
            frame: Aktueller Frame
        """
        code = frame.f_code
        function_name = code.co_name
        filename = code.co_filename
        lineno = frame.f_lineno
        
        logger.info(f"Schritt: {function_name} ({filename}:{lineno})")
        
        # Lokale Variablen
        locals_str = "\n".join(f"  {name} = {repr(value)}" for name, value in frame.f_locals.items())
        logger.info(f"Lokale Variablen:\n{locals_str}")
        
        # Zurücksetzen für nächsten Schritt
        self.next_line = None
        self.next_frame = None


def profile(func=None, *, name=None):
    """
    Dekorator für das Profiling von Funktionen.
    
    Args:
        func: Zu dekorierende Funktion
        name: Optionaler Name für das Profiling
        
    Returns:
        Dekorierte Funktion
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            
            # Verwende Funktionsnamen, wenn kein Name angegeben
            profile_name = name or f.__qualname__
            
            with profiler.profile_function(profile_name):
                return f(*args, **kwargs)
        
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


# Singleton-Instanzen
_profiler_instance = None
_debugger_instance = None

def get_profiler(config: Optional[ProfilerConfig] = None) -> MCodeProfiler:
    """
    Gibt eine Singleton-Instanz des Profilers zurück.
    
    Args:
        config: Profiler-Konfiguration
        
    Returns:
        Profiler
    """
    global _profiler_instance
    
    if _profiler_instance is None:
        _profiler_instance = MCodeProfiler(config)
        
    return _profiler_instance

def get_debugger() -> MCodeDebugger:
    """
    Gibt eine Singleton-Instanz des Debuggers zurück.
    
    Returns:
        Debugger
    """
    global _debugger_instance
    
    if _debugger_instance is None:
        _debugger_instance = MCodeDebugger()
        
    return _debugger_instance
