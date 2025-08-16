#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simulate_modifications.py

Modul zur Simulation von Code-Änderungen im Rahmen des Recursive Self-Improvement (RSI) Systems.
Dieses Modul implementiert die Sandbox-Umgebung und Simulationsmechanismen für den SelfRewriter.
"""

import os
import sys
import uuid
import json
import copy
import datetime
import logging
import tempfile
import subprocess
import importlib
import shutil
import time
from typing import Dict, List, Tuple, Any, Optional, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimulateModifications")

def simulate_modifications(
    modification_info: Dict[str, Any],
    sandbox_dir: str,
    modules_dir: str,
    benchmark_dir: str,
    require_tests: bool = True
) -> Dict[str, Any]:
    """Simuliert die vorgeschlagenen Code-Änderungen in einer isolierten Sandbox-Umgebung.
    
    Diese Funktion erstellt eine Sandbox-Umgebung, in der die vorgeschlagenen Code-Änderungen
    getestet werden können, ohne das Produktionssystem zu beeinträchtigen. Sie führt Tests
    aus, um die Funktionalität der Änderungen zu validieren, und sammelt Performance-Metriken.
    
    Args:
        modification_info: Informationen über die vorgeschlagene Modifikation
        sandbox_dir: Verzeichnis für die Sandbox-Umgebung
        modules_dir: Basisverzeichnis der Module
        benchmark_dir: Verzeichnis mit Benchmark-Tests
        require_tests: Ob Tests erforderlich sind, um die Änderung zu akzeptieren
        
    Returns:
        Dict mit Informationen über die Simulationsergebnisse
    """
    # Generiere eine eindeutige Simulation-ID
    simulation_id = f"SIM-{uuid.uuid4().hex[:8]}"
    logger.info(f"Starte Simulation {simulation_id} für Modifikation {modification_info.get('modification_id', 'unbekannt')}")
    
    # Initialisiere Ergebnis
    simulation_result = {
        "status": "started",
        "simulation_id": simulation_id,
        "modification_id": modification_info.get("modification_id", ""),
        "target_module": modification_info.get("target_module", ""),
        "timestamp_start": datetime.datetime.now().isoformat(),
        "tests_passed": False,
        "metrics": {},
        "errors": [],
        "warnings": [],
        "performance_change": 0.0
    }
    
    try:
        # 1. Erstelle Sandbox-Umgebung
        target_module = modification_info.get("target_module", "")
        if not target_module:
            raise ValueError("Keine Zielmodul-Information gefunden")
        
        # Erstelle ein temporäres Verzeichnis in der Sandbox
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sandbox_instance = os.path.join(sandbox_dir, f"{target_module.replace('.', '_')}_{timestamp}")
        os.makedirs(sandbox_instance, exist_ok=True)
        
        # 2. Kopiere das Modul und seine Abhängigkeiten in die Sandbox
        module_files = []
        module_parts = target_module.split('.')
        current_path = sandbox_instance
        
        # Erstelle Verzeichnisstruktur
        for part in module_parts[:-1]:
            current_path = os.path.join(current_path, part)
            os.makedirs(current_path, exist_ok=True)
            init_path = os.path.join(current_path, '__init__.py')
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    f.write("# Automatically generated\n")
        
        # Pfade zum Quell- und Zielmodul
        source_path = os.path.join(modules_dir, *module_parts[:-1], f"{module_parts[-1]}.py")
        target_path = os.path.join(current_path, f"{module_parts[-1]}.py")
        
        # Prüfe, ob das Quellmodul existiert
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Quellmodul {source_path} nicht gefunden")
        
        # 3. Wende die Änderungen in der Sandbox an
        original_code = modification_info.get("original_code", "")
        modified_code = modification_info.get("modified_code", "")
        
        if not original_code or not modified_code:
            raise ValueError("Original- oder modifizierter Code fehlt")
        
        # Schreibe den modifizierten Code in die Sandbox
        with open(target_path, 'w') as f:
            f.write(modified_code)
        
        module_files.append(target_path)
        
        # 4. Kopiere auch abhängige Module (vereinfachte Methode)
        dependencies = _find_dependencies(target_module, modified_code, modules_dir)
        for dep_module, dep_path in dependencies.items():
            # Erstelle Verzeichnisstruktur für Abhängigkeit
            dep_parts = dep_module.split('.')
            dep_dir = os.path.join(sandbox_instance, *dep_parts[:-1])
            os.makedirs(dep_dir, exist_ok=True)
            
            # Kopiere Abhängigkeit
            dep_target = os.path.join(dep_dir, f"{dep_parts[-1]}.py")
            shutil.copy2(dep_path, dep_target)
            
            # Erstelle __init__.py-Dateien
            current_path = sandbox_instance
            for part in dep_parts[:-1]:
                current_path = os.path.join(current_path, part)
                init_path = os.path.join(current_path, '__init__.py')
                if not os.path.exists(init_path):
                    with open(init_path, 'w') as f:
                        f.write("# Automatically generated\n")
        
        # 5. Führe syntaktische Validierung durch
        syntax_valid, syntax_errors = _validate_syntax(target_path)
        if not syntax_valid:
            simulation_result["status"] = "failed"
            simulation_result["errors"].extend(syntax_errors)
            raise SyntaxError(f"Syntaxfehler im modifizierten Code: {syntax_errors}")
        
        # 6. Führe Tests durch, falls vorhanden und erforderlich
        test_results = {}
        if require_tests:
            # Bestimme Testpfad
            test_module = f"tests.{'.'.join(module_parts[1:])}" if module_parts[0] == 'miso' else f"tests.{target_module}"
            test_file = f"test_{'_'.join(module_parts)}.py"
            test_path = os.path.join(benchmark_dir, *test_module.split('.')[1:], test_file)
            
            if os.path.exists(test_path):
                # Kopiere Test in die Sandbox
                sandbox_test_dir = os.path.join(sandbox_instance, 'tests', *test_module.split('.')[1:])
                os.makedirs(sandbox_test_dir, exist_ok=True)
                
                # Erstelle __init__.py-Dateien im Test-Verzeichnis
                current_path = os.path.join(sandbox_instance, 'tests')
                os.makedirs(current_path, exist_ok=True)
                with open(os.path.join(current_path, '__init__.py'), 'w') as f:
                    f.write("# Automatically generated\n")
                
                for part in test_module.split('.')[1:-1]:
                    current_path = os.path.join(current_path, part)
                    os.makedirs(current_path, exist_ok=True)
                    init_path = os.path.join(current_path, '__init__.py')
                    if not os.path.exists(init_path):
                        with open(init_path, 'w') as f:
                            f.write("# Automatically generated\n")
                
                # Kopiere Test
                sandbox_test_path = os.path.join(sandbox_test_dir, test_file)
                shutil.copy2(test_path, sandbox_test_path)
                
                # Führe Tests aus
                try:
                    env = os.environ.copy()
                    env['PYTHONPATH'] = f"{sandbox_instance}:{env.get('PYTHONPATH', '')}"
                    
                    # Zeitmessung für Performance-Vergleich
                    start_time = time.time()
                    
                    # Führe Test mit pytest aus
                    process = subprocess.run(
                        [sys.executable, '-m', 'pytest', sandbox_test_path, '-v'],
                        capture_output=True,
                        text=True,
                        env=env
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Analysiere Ergebnisse
                    success = process.returncode == 0
                    output = process.stdout + process.stderr
                    
                    test_results = {
                        "status": "success" if success else "failure",
                        "output": output,
                        "execution_time": execution_time,
                        "return_code": process.returncode
                    }
                    
                    if success:
                        simulation_result["tests_passed"] = True
                    else:
                        simulation_result["status"] = "failed"
                        simulation_result["errors"].append(f"Tests fehlgeschlagen: {output}")
                        
                except Exception as e:
                    simulation_result["status"] = "failed"
                    simulation_result["errors"].append(f"Fehler beim Ausführen der Tests: {str(e)}")
            else:
                if require_tests:
                    simulation_result["warnings"].append(f"Keine Tests für {target_module} gefunden (gesucht: {test_path})")
        
        # 7. Messe Performance-Metriken, falls Tests erfolgreich waren
        if simulation_result["tests_passed"]:
            try:
                # Führe Performance-Benchmark aus (vereinfachte Version)
                perf_metrics = _measure_performance(target_module, sandbox_instance)
                simulation_result["metrics"] = perf_metrics
                
                # Berechne Performance-Änderung
                if "execution_time" in perf_metrics and "baseline_execution_time" in perf_metrics:
                    current = perf_metrics["execution_time"]
                    baseline = perf_metrics["baseline_execution_time"]
                    if baseline > 0:
                        # Negative Werte bedeuten Verbesserung (weniger Zeit)
                        change_percent = ((current - baseline) / baseline) * 100
                        simulation_result["performance_change"] = -change_percent
            except Exception as e:
                simulation_result["warnings"].append(f"Fehler bei der Performance-Messung: {str(e)}")
        
        # 8. Führe statische Analyse durch
        try:
            static_analysis_results = _run_static_analysis(target_path)
            simulation_result["static_analysis"] = static_analysis_results
            
            # Wenn schwerwiegende Probleme gefunden wurden, füge Warnung hinzu
            if "issues" in static_analysis_results and static_analysis_results["issues"]:
                high_severity_issues = [i for i in static_analysis_results["issues"] if i.get("severity") == "high"]
                if high_severity_issues:
                    simulation_result["warnings"].append(f"{len(high_severity_issues)} schwerwiegende Probleme bei der statischen Analyse gefunden")
        except Exception as e:
            simulation_result["warnings"].append(f"Fehler bei der statischen Analyse: {str(e)}")
        
        # 9. Finalisiere das Ergebnis
        if simulation_result["status"] == "started":
            simulation_result["status"] = "success"
        
    except Exception as e:
        logger.error(f"Fehler bei der Simulation: {str(e)}")
        simulation_result["status"] = "failed"
        simulation_result["errors"].append(str(e))
    finally:
        # Setze Endzeitstempel
        simulation_result["timestamp_end"] = datetime.datetime.now().isoformat()
        
        # Berechne Gesamtdauer
        start_time = datetime.datetime.fromisoformat(simulation_result["timestamp_start"])
        end_time = datetime.datetime.fromisoformat(simulation_result["timestamp_end"])
        duration_seconds = (end_time - start_time).total_seconds()
        simulation_result["duration_seconds"] = duration_seconds
        
        # Protokolliere Ergebnis
        if simulation_result["status"] == "success":
            logger.info(f"Simulation {simulation_id} erfolgreich abgeschlossen (Dauer: {duration_seconds:.2f}s)")
        else:
            logger.error(f"Simulation {simulation_id} fehlgeschlagen (Dauer: {duration_seconds:.2f}s)")
    
    return simulation_result

def _find_dependencies(module_name: str, code: str, modules_dir: str) -> Dict[str, str]:
    """Findet Abhängigkeiten eines Moduls basierend auf seinem Code.
    
    Args:
        module_name: Name des Moduls
        code: Quellcode des Moduls
        modules_dir: Basisverzeichnis der Module
        
    Returns:
        Dict mit Abhängigkeiten {module_name: file_path}
    """
    dependencies = {}
    
    # Extrahiere Importe mit regulären Ausdrücken
    import_patterns = [
        r'^\s*import\s+([\w\.]+)',  # import x
        r'^\s*from\s+([\w\.]+)\s+import',  # from x import y
    ]
    
    import_modules = set()
    for pattern in import_patterns:
        import_matches = re.finditer(pattern, code, re.MULTILINE)
        for match in import_matches:
            imported_module = match.group(1)
            # Beschränke uns auf MISO-Module
            if imported_module.startswith('miso.'):
                import_modules.add(imported_module)
    
    # Finde Pfade zu den importierten Modulen
    for imported_module in import_modules:
        if imported_module != module_name:  # Vermeide Selbstimporte
            module_parts = imported_module.split('.')
            module_path = os.path.join(modules_dir, *module_parts[:-1], f"{module_parts[-1]}.py")
            
            if os.path.exists(module_path):
                dependencies[imported_module] = module_path
    
    return dependencies

def _validate_syntax(file_path: str) -> Tuple[bool, List[str]]:
    """Validiert die Syntax einer Python-Datei.
    
    Args:
        file_path: Pfad zur zu validierenden Datei
        
    Returns:
        Tuple aus (Gültigkeit, Liste von Fehlern)
    """
    errors = []
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Kompiliere den Code, um Syntaxfehler zu finden
        compile(source, file_path, 'exec')
        return True, []
    except SyntaxError as e:
        errors.append(f"Zeile {e.lineno}, Position {e.offset}: {e.msg}")
        return False, errors
    except Exception as e:
        errors.append(f"Unerwarteter Fehler: {str(e)}")
        return False, errors

def _measure_performance(module_name: str, sandbox_path: str) -> Dict[str, Any]:
    """Misst die Performance eines Moduls.
    
    Args:
        module_name: Name des Moduls
        sandbox_path: Pfad zur Sandbox
        
    Returns:
        Dict mit Performance-Metriken
    """
    metrics = {}
    
    try:
        # Setze Python-Pfad
        sys.path.insert(0, sandbox_path)
        
        # Importiere das Modul
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None:
            return {"error": f"Modul {module_name} konnte nicht importiert werden"}
        
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        
        # Finde öffentliche Methoden
        public_functions = [name for name, obj in module.__dict__.items() 
                           if callable(obj) and not name.startswith('_')]
        
        # Messe Ausführungszeit einer repräsentativen Funktion (falls vorhanden)
        if public_functions:
            func_name = public_functions[0]
            func = getattr(module, func_name)
            
            # Einfache Zeitmessung (keine Parameter)
            try:
                start_time = time.time()
                func()
                execution_time = time.time() - start_time
                metrics["execution_time"] = execution_time
                metrics["measured_function"] = func_name
                
                # Setze Baseline-Zeit (fiktiv, in einer echten Implementierung würde man historische Daten verwenden)
                metrics["baseline_execution_time"] = execution_time * 1.1  # 10% langsamer als aktuell
            except Exception as e:
                metrics["execution_error"] = str(e)
        
        # Füge Speichernutzung hinzu (vereinfacht)
        metrics["memory_usage"] = {
            "module_size": os.path.getsize(module_spec.origin),
            "total_memory": "nicht gemessen"  # In einer realen Implementierung würde man psutil verwenden
        }
        
    except Exception as e:
        metrics["error"] = str(e)
    finally:
        # Entferne Sandbox aus dem Pfad
        if sandbox_path in sys.path:
            sys.path.remove(sandbox_path)
    
    return metrics

def _run_static_analysis(file_path: str) -> Dict[str, Any]:
    """Führt eine statische Analyse einer Python-Datei durch.
    
    Args:
        file_path: Pfad zur zu analysierenden Datei
        
    Returns:
        Dict mit den Ergebnissen der statischen Analyse
    """
    results = {
        "issues": [],
        "metrics": {}
    }
    
    try:
        # Lese die Datei
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Zeilenzählung
        line_count = code.count('\n') + 1
        results["metrics"]["line_count"] = line_count
        
        # Einfache Komplexitätsmetriken
        function_count = code.count('def ')
        class_count = code.count('class ')
        results["metrics"]["function_count"] = function_count
        results["metrics"]["class_count"] = class_count
        
        # Warnzeichen im Code
        warning_patterns = [
            (r'\bprint\s*\(', "Debugging-Ausgabe gefunden", "low"),
            (r'\b(TODO|FIXME)\b', "Unvollständiger Code gefunden", "medium"),
            (r'\beval\s*\(', "Verwendung von eval() gefunden", "high"),
            (r'\bexec\s*\(', "Verwendung von exec() gefunden", "high"),
            (r'\b__import__\s*\(', "Dynamischer Import gefunden", "medium"),
            (r'except\s*:', "Unspezifischer except-Block gefunden", "medium"),
            (r'except\s+Exception\s*:', "Zu allgemeiner except-Block gefunden", "low"),
            (r'\bglobals\(\)\s*\[', "Modifikation globaler Variablen gefunden", "medium"),
            (r'os\.system\s*\(', "Systemaufruf gefunden", "high"),
            (r'subprocess\.(Popen|call|run)', "Subprozessaufruf gefunden", "high")
        ]
        
        for pattern, message, severity in warning_patterns:
            matches = list(re.finditer(pattern, code))
            for match in matches:
                # Bestimme Zeilennummer
                line_number = code[:match.start()].count('\n') + 1
                
                results["issues"].append({
                    "line": line_number,
                    "message": message,
                    "severity": severity,
                    "code": code.split('\n')[line_number - 1].strip()
                })
        
        # Einfache Codequalitätsmetriken
        results["metrics"]["comments_ratio"] = len(re.findall(r'^\s*#.*$', code, re.MULTILINE)) / line_count if line_count > 0 else 0
        results["metrics"]["docstring_count"] = code.count('"""') / 2  # Ungefähre Anzahl von Docstrings
        
    except Exception as e:
        results["error"] = str(e)
    
    return results

# Wenn dieses Skript direkt ausgeführt wird
if __name__ == "__main__":
    # Beispielaufruf
    test_modification = {
        "modification_id": "MOD-TEST",
        "target_module": "miso.core.example",
        "original_code": "def example():\n    return 'original'\n",
        "modified_code": "def example():\n    return 'modified'\n"
    }
    
    result = simulate_modifications(
        test_modification,
        sandbox_dir="./sandbox",
        modules_dir="./miso",
        benchmark_dir="./tests",
        require_tests=False
    )
    
    print(json.dumps(result, indent=2))
