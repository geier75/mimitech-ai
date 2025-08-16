#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SelfRewriter.py

Dieses Modul ist Teil des Recursive Self-Improvement (RSI) Systems des VXOR AI
und ermöglicht dem System, seinen eigenen Code zu analysieren, Änderungen vorzuschlagen,
zu simulieren und zu bewerten.

Der SelfRewriter arbeitet eng mit dem RecursiveEvaluator zusammen, um identifizierte
Verbesserungspotenziale in konkrete Code-Änderungen umzusetzen. Dabei wird jede
Änderung in einer sicheren Sandbox-Umgebung getestet, bevor sie für die Deployment-
Phase freigegeben wird.

Funktionen:
1. Analyse bestehender Codestrukturen
2. Generierung von Code-Modifikationsvorschlägen
3. Simulation von Änderungen in einer isolierten Umgebung
4. Automatische Bewertung von Änderungen anhand von Benchmarks und Sicherheitskriterien
5. Dokumentation aller vorgeschlagenen und getesteten Änderungen für Audit-Trails
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
import inspect
import re
import ast
import time
import shutil
import threading
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SelfRewriter")

class SelfRewriter:
    """
    SelfRewriter ermöglicht es dem VXOR-System, Code-Änderungen vorzuschlagen,
    zu simulieren und zu bewerten. Dies ist ein zentraler Bestandteil des
    Recursive Self-Improvement (RSI) Systems.
    
    Features:
    - Schlägt Code-Änderungen basierend auf identifizierten Verbesserungspotenzialen vor
    - Simuliert Änderungen in einer isolierten Umgebung
    - Bewertet Änderungen automatisch anhand von Benchmarks und Sicherheitskriterien
    - Dokumentiert alle vorgeschlagenen und getesteten Änderungen für Audit-Trails
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 base_output_dir: Optional[str] = None,
                 sandbox_dir: Optional[str] = None,
                 modules_dir: Optional[str] = None,
                 benchmark_dir: Optional[str] = None,
                 max_modification_attempts: int = 3,
                 min_improvement_threshold: float = 0.05,
                 safety_checks_required: bool = True,
                 require_tests: bool = True,
                 enable_auto_approval: bool = False,
                 code_generation_backend: str = "ast_analysis",
                 log_level: str = "INFO"):
        """
        Initialisiert den SelfRewriter.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            base_output_dir: Basisverzeichnis für Ausgaben
            sandbox_dir: Verzeichnis für die Sandbox-Umgebung
            modules_dir: Verzeichnis mit den zu modifizierenden Modulen
            benchmark_dir: Verzeichnis mit Benchmark-Tests
            max_modification_attempts: Maximale Anzahl von Änderungsversuchen pro Verbesserungsvorschlag
            min_improvement_threshold: Mindestverbesserung in Prozent, damit eine Änderung als erfolgreich gilt
            safety_checks_required: Ob Sicherheitsprüfungen für alle Änderungen erforderlich sind
            require_tests: Ob Tests für alle geänderten Module erforderlich sind
            enable_auto_approval: Ob Änderungen automatisch genehmigt werden können (nur für niedrigere Risikostufen)
            code_generation_backend: Backend für die Code-Generierung ("ast_analysis", "template_based", "hybrid")
            log_level: Logging-Level ("DEBUG", "INFO", "WARNING", "ERROR")
        """
        # Generiere eine eindeutige Session-ID
        self.session_id = str(uuid.uuid4())
        logger.info(f"Initialisiere SelfRewriter mit Session-ID: {self.session_id}")
        
        # Setze Logging-Level
        logger.setLevel(getattr(logging, log_level))
        
        # Lade Konfiguration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Setze Basisparameter
        self.base_output_dir = base_output_dir or self.config.get('output_dir', './output')
        self.sandbox_dir = sandbox_dir or self.config.get('sandbox_dir', os.path.join(self.base_output_dir, 'sandbox'))
        self.modules_dir = modules_dir or self.config.get('modules_dir', './miso')
        self.benchmark_dir = benchmark_dir or self.config.get('benchmark_dir', './tests')
        
        # Leistungs- und Sicherheitsparameter
        self.max_modification_attempts = max_modification_attempts
        self.min_improvement_threshold = min_improvement_threshold
        self.safety_checks_required = safety_checks_required
        self.require_tests = require_tests
        self.enable_auto_approval = enable_auto_approval
        
        # Code-Generierung
        self.code_generation_backend = code_generation_backend
        
        # Erstelle die erforderlichen Verzeichnisse
        self._setup_directories()
        
        # Speicher für vorgeschlagene Code-Änderungen
        self.proposed_modifications = []
        
        # Speicher für simulierte Code-Änderungen
        self.simulated_modifications = []
        
        # Speicher für genehmigte Code-Änderungen
        self.approved_modifications = []
        
        # Speicher für abgelehnte Code-Änderungen
        self.rejected_modifications = []
        
        # Lookup-Tabelle für Module
        self.module_lookup = {}
        
        # Initialisiere Codebase-Scanner
        self._scan_codebase()
        
        logger.info("SelfRewriter erfolgreich initialisiert")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Lädt die Konfigurationsdatei."""
        if not os.path.exists(config_path):
            logger.warning(f"Konfigurationsdatei {config_path} nicht gefunden. Verwende Standardwerte.")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Konfiguration aus {config_path} geladen")
            return config
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            return {}
    
    def _setup_directories(self) -> None:
        """Erstellt die erforderlichen Verzeichnisse."""
        directories = [
            self.base_output_dir,
            self.sandbox_dir,
            os.path.join(self.base_output_dir, 'proposed_modifications'),
            os.path.join(self.base_output_dir, 'simulated_modifications'),
            os.path.join(self.base_output_dir, 'approved_modifications'),
            os.path.join(self.base_output_dir, 'rejected_modifications'),
            os.path.join(self.base_output_dir, 'audit_logs')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Verzeichnis erstellt/überprüft: {directory}")
    
    def _scan_codebase(self) -> None:
        """Scannt die Codebase, um Modulinformationen zu sammeln."""
        logger.info(f"Scanne Codebase in {self.modules_dir}")
        
        # Zurücksetzen des module_lookup
        self.module_lookup = {}
        
        # Traversiere das Modulverzeichnis
        for root, dirs, files in os.walk(self.modules_dir):
            # Ignoriere __pycache__ und andere nicht-Python-Verzeichnisse
            dirs[:] = [d for d in dirs if not d.startswith('__') and not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.modules_dir)
                    module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
                    
                    # Sammle Metadaten über die Datei
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        # Parse die Datei mit ast, um Klassen und Funktionen zu extrahieren
                        try:
                            tree = ast.parse(content)
                            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                            
                            # Speichere Modulinformationen
                            self.module_lookup[module_name] = {
                                'path': file_path,
                                'last_modified': os.path.getmtime(file_path),
                                'size': os.path.getsize(file_path),
                                'classes': classes,
                                'functions': functions,
                                'content_hash': hash(content)
                            }
                            
                        except SyntaxError as e:
                            logger.warning(f"Syntax-Fehler beim Parsen von {file_path}: {e}")
                    except Exception as e:
                        logger.error(f"Fehler beim Scannen von {file_path}: {e}")
        
        logger.info(f"{len(self.module_lookup)} Module in der Codebase gefunden")
    
    def _prepare_sandbox(self, target_module: str) -> str:
        """Bereitet eine Sandbox-Umgebung für ein Modul vor.
        
        Args:
            target_module: Name des Zielmoduls (z.B. 'miso.core.model')
            
        Returns:
            Pfad zur Sandbox-Umgebung
        """
        # Erstelle ein temporäres Verzeichnis in der Sandbox
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sandbox_instance = os.path.join(self.sandbox_dir, f"{target_module.replace('.', '_')}_{timestamp}")
        os.makedirs(sandbox_instance, exist_ok=True)
        
        # Erstelle Verzeichnisstruktur des Moduls
        module_parts = target_module.split('.')
        current_path = sandbox_instance
        
        for part in module_parts[:-1]:  # Alle außer dem letzten Teil (Dateiname)
            current_path = os.path.join(current_path, part)
            os.makedirs(current_path, exist_ok=True)
            # Erstelle leere __init__.py-Dateien für korrektes Import-Verhalten
            init_path = os.path.join(current_path, '__init__.py')
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    f.write("# Automatically generated by SelfRewriter\n")
        
        # Kopiere das Zielmodul in die Sandbox
        if target_module in self.module_lookup:
            source_path = self.module_lookup[target_module]['path']
            target_filename = module_parts[-1] + '.py'
            target_path = os.path.join(current_path, target_filename)
            
            shutil.copy2(source_path, target_path)
            logger.info(f"Modul {target_module} in Sandbox {sandbox_instance} kopiert")
            
            # Kopiere auch abhängige Module
            self._copy_dependencies(target_module, sandbox_instance)
            
            return sandbox_instance
        else:
            raise ValueError(f"Modul {target_module} wurde nicht in der Codebase gefunden")
    
    def _copy_dependencies(self, target_module: str, sandbox_path: str) -> None:
        """Kopiert abhängige Module in die Sandbox.
        
        Args:
            target_module: Name des Zielmoduls
            sandbox_path: Pfad zur Sandbox-Umgebung
        """
        if target_module not in self.module_lookup:
            return
        
        # Lese Modul-Inhalt
        with open(self.module_lookup[target_module]['path'], 'r') as f:
            content = f.read()
        
        # Extrahiere Import-Statements mit regulären Ausdrücken
        # Beachte sowohl 'import x' als auch 'from x import y'
        import_patterns = [
            r'^\s*import\s+([\w\.]+)',  # 'import x' oder 'import x.y'
            r'^\s*from\s+([\w\.]+)\s+import',  # 'from x import y' oder 'from x.y import z'
        ]
        
        imported_modules = set()
        for pattern in import_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                imported_module = match.group(1)
                # Füge nur MISO-Module hinzu, keine externen Bibliotheken
                if imported_module.startswith('miso.') and imported_module in self.module_lookup:
                    imported_modules.add(imported_module)
        
        # Rekursiv für jedes abhängige Modul
        for imported_module in imported_modules:
            if imported_module != target_module:  # Vermeide Zirkelbezüge
                # Pfad in der Sandbox
                module_parts = imported_module.split('.')
                module_dir = os.path.join(sandbox_path, *module_parts[:-1])
                os.makedirs(module_dir, exist_ok=True)
                
                # Erstelle __init__.py-Dateien
                current_path = sandbox_path
                for part in module_parts[:-1]:
                    current_path = os.path.join(current_path, part)
                    init_path = os.path.join(current_path, '__init__.py')
                    if not os.path.exists(init_path):
                        with open(init_path, 'w') as f:
                            f.write("# Automatically generated by SelfRewriter\n")
                
                # Kopiere Modul
                source_path = self.module_lookup[imported_module]['path']
                target_path = os.path.join(module_dir, module_parts[-1] + '.py')
                shutil.copy2(source_path, target_path)
                
                # Rekursiv für Abhängigkeiten dieses Moduls
                self._copy_dependencies(imported_module, sandbox_path)
    
    def _run_tests(self, module_name: str, sandbox_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Führt Tests für ein Modul in der Sandbox aus.
        
        Args:
            module_name: Name des Moduls
            sandbox_path: Pfad zur Sandbox-Umgebung
            
        Returns:
            Tupel aus (Erfolgsstatus, Testergebnisse)
        """
        logger.info(f"Führe Tests für Modul {module_name} in Sandbox aus")
        
        # Bestimme den entsprechenden Testpfad
        module_parts = module_name.split('.')
        test_module = f"tests.{'.'.join(module_parts[1:])}" if module_parts[0] == 'miso' else f"tests.{module_name}"
        test_file = f"test_{'_'.join(module_parts)}.py"
        
        # Prüfe, ob Test existiert
        test_path = os.path.join(self.benchmark_dir, *test_module.split('.')[1:], test_file)
        if not os.path.exists(test_path):
            logger.warning(f"Keine Tests für Modul {module_name} gefunden (gesucht: {test_path})")
            return False, {"status": "no_tests", "message": f"Keine Tests für {module_name} gefunden"}
        
        # Kopiere Test in die Sandbox
        sandbox_test_dir = os.path.join(sandbox_path, 'tests', *test_module.split('.')[1:])
        os.makedirs(sandbox_test_dir, exist_ok=True)
        
        # Erstelle __init__.py-Dateien im Test-Verzeichnis
        current_path = os.path.join(sandbox_path, 'tests')
        os.makedirs(current_path, exist_ok=True)
        with open(os.path.join(current_path, '__init__.py'), 'w') as f:
            f.write("# Automatically generated by SelfRewriter\n")
        
        for part in test_module.split('.')[1:-1]:
            current_path = os.path.join(current_path, part)
            os.makedirs(current_path, exist_ok=True)
            init_path = os.path.join(current_path, '__init__.py')
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    f.write("# Automatically generated by SelfRewriter\n")
        
        sandbox_test_path = os.path.join(sandbox_test_dir, test_file)
        shutil.copy2(test_path, sandbox_test_path)
        
        # Führe Test in einer Subprocess-Umgebung aus
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{sandbox_path}:{env.get('PYTHONPATH', '')}"  # Sandbox zum Python-Pfad hinzufügen
            
            # Führe Test mit pytest aus
            process = subprocess.run(
                [sys.executable, '-m', 'pytest', sandbox_test_path, '-v'],
                capture_output=True,
                text=True,
                env=env
            )
            
            # Analyse der Ergebnisse
            success = process.returncode == 0
            output = process.stdout + process.stderr
            
            results = {
                "status": "success" if success else "failure",
                "output": output,
                "return_code": process.returncode
            }
            
            # Füge spezifischere Informationen hinzu, wenn verfügbar
            if "collected 0 items" in output:
                results["status"] = "no_tests"
                results["message"] = "Keine Tests in der Datei gefunden"
            
            logger.info(f"Test für {module_name} {'erfolgreich' if success else 'fehlgeschlagen'}")
            return success, results
            
        except Exception as e:
            logger.error(f"Fehler beim Ausführen der Tests für {module_name}: {e}")
            return False, {"status": "error", "message": str(e)}
            
    def _check_safety(self, original_code: str, modified_code: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """Führt Sicherheitsprüfungen für Codeänderungen durch.
        
        Args:
            original_code: Ursprünglicher Code
            modified_code: Geänderter Code
            
        Returns:
            Tupel aus (Sicherheitsstatus, Liste von Sicherheitsproblemen)
        """
        safety_issues = []
        is_safe = True
        
        # Grundlegende statische Analyse mit AST
        try:
            # Parse beide Versionen
            original_ast = ast.parse(original_code)
            modified_ast = ast.parse(modified_code)
            
            # Prüfe auf problematische Änderungen
            original_imports = set()
            modified_imports = set()
            
            # Sammle Imports aus Original-Code
            for node in ast.walk(original_ast):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        original_imports.add(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        original_imports.add(node.module)
            
            # Sammle Imports aus modifiziertem Code
            for node in ast.walk(modified_ast):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        modified_imports.add(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        modified_imports.add(node.module)
            
            # Prüfe auf neue Imports
            new_imports = modified_imports - original_imports
            dangerous_imports = ['os', 'subprocess', 'sys', 'shutil', 'pty', 'socket', 'requests']
            
            for imp in new_imports:
                if any(imp == d_imp or imp.startswith(f"{d_imp}.") for d_imp in dangerous_imports):
                    safety_issues.append({
                        "type": "dangerous_import",
                        "severity": "high",
                        "description": f"Neuer Import von potenziell gefährlichem Modul: {imp}"
                    })
                    is_safe = False
            
            # Prüfe auf eval(), exec() oder andere gefährliche Funktionsaufrufe
            dangerous_functions = ['eval', 'exec', 'compile', 'globals', '__import__']
            
            for node in ast.walk(modified_ast):
                if isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id in dangerous_functions:
                    safety_issues.append({
                        "type": "dangerous_function",
                        "severity": "high",
                        "description": f"Verwendung der gefährlichen Funktion: {node.func.id}()"
                    })
                    is_safe = False
            
        except SyntaxError as e:
            safety_issues.append({
                "type": "syntax_error",
                "severity": "high",
                "description": f"Syntax-Fehler im modifizierten Code: {e}"
            })
            is_safe = False
        except Exception as e:
            safety_issues.append({
                "type": "analysis_error",
                "severity": "medium",
                "description": f"Fehler bei der Sicherheitsanalyse: {e}"
            })
        
        return is_safe, safety_issues
    
    def propose_code_modifications(self, target_module: str, improvement_opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Schlägt Code-Modifikationen für ein Zielmodul vor, basierend auf einem Verbesserungspotenzial.
        
        Diese Funktion analysiert den Code des Zielmoduls und schlägt Änderungen vor, die
        das identifizierte Verbesserungspotenzial adressieren. Sie erstellt einen detaillierten
        Änderungsvorschlag, der später simuliert und bewertet werden kann.
        
        Args:
            target_module: Name des Zielmoduls (z.B. 'miso.core.model')
            improvement_opportunity: Verbesserungspotenzial aus dem RecursiveEvaluator
            
        Returns:
            Dict mit Informationen über die vorgeschlagene Modifikation
        """
        logger.info(f"Schlage Code-Modifikationen für Modul {target_module} vor")
        
        # Überprüfe, ob das Modul in der Codebase existiert
        if target_module not in self.module_lookup:
            logger.error(f"Modul {target_module} wurde nicht in der Codebase gefunden")
            return {
                "status": "error",
                "message": f"Modul {target_module} nicht gefunden",
                "target_module": target_module,
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": self.session_id
            }
        
        # Erzeuge eine eindeutige ID für diese Modifikation
        modification_id = f"MOD-{uuid.uuid4().hex[:8]}"
        
        # Lade den Code des Moduls
        original_code = ""
        try:
            with open(self.module_lookup[target_module]['path'], 'r') as f:
                original_code = f.read()
        except Exception as e:
            logger.error(f"Fehler beim Lesen des Moduls {target_module}: {e}")
            return {
                "status": "error",
                "message": f"Fehler beim Lesen des Moduls: {str(e)}",
                "target_module": target_module,
                "modification_id": modification_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": self.session_id
            }
        
        # Analysiere Code und erstelle AST (Abstract Syntax Tree)
        try:
            tree = ast.parse(original_code)
        except SyntaxError as e:
            logger.error(f"Syntax-Fehler im Modul {target_module}: {e}")
            return {
                "status": "error",
                "message": f"Syntax-Fehler im Modul: {str(e)}",
                "target_module": target_module,
                "modification_id": modification_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": self.session_id
            }
        
        # Extrahiere relevante Informationen aus dem Verbesserungspotenzial
        improvement_type = improvement_opportunity.get("target", "")
        improvement_action = improvement_opportunity.get("action", "")
        improvement_reason = improvement_opportunity.get("reason", "")
        improvement_category = improvement_opportunity.get("category", "")
        
        # Wähle die passende Strategie für die Codegenerierung basierend auf Backend-Einstellung
        if self.code_generation_backend == "ast_analysis":
            modification_result = self._propose_via_ast_analysis(
                target_module, original_code, tree, improvement_opportunity
            )
        elif self.code_generation_backend == "template_based":
            modification_result = self._propose_via_templates(
                target_module, original_code, improvement_opportunity
            )
        else:  # hybrid oder Fallback
            modification_result = self._propose_via_hybrid_approach(
                target_module, original_code, tree, improvement_opportunity
            )
        
        # Überprüfe Ergebnis
        if not modification_result or "modified_code" not in modification_result:
            logger.warning(f"Keine Modifikationen für {target_module} vorgeschlagen")
            return {
                "status": "no_modifications",
                "message": "Keine passenden Modifikationen gefunden",
                "target_module": target_module,
                "modification_id": modification_id,
                "improvement_opportunity": improvement_opportunity,
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": self.session_id
            }
        
        # Bereite Ergebnis vor
        modification_info = {
            "status": "proposed",
            "target_module": target_module,
            "modification_id": modification_id,
            "original_code": original_code,
            "modified_code": modification_result["modified_code"],
            "changes": modification_result.get("changes", []),
            "improvement_opportunity": improvement_opportunity,
            "description": modification_result.get("description", ""),
            "expected_benefits": modification_result.get("expected_benefits", []),
            "risk_assessment": modification_result.get("risk_assessment", {"level": "medium"}),
            "requires_human_review": self._requires_human_review(modification_result, improvement_opportunity),
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": self.session_id,
            "origin": self.code_generation_backend
        }
        
        # Speichere die vorgeschlagene Modifikation
        self._save_proposed_modification(modification_info)
        self.proposed_modifications.append(modification_info)
        
        logger.info(f"Modifikation {modification_id} für {target_module} vorgeschlagen")
        
        return modification_info
    
    def _propose_via_ast_analysis(self, target_module: str, original_code: str, tree: ast.AST, 
                                 improvement_opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Schlägt Codeänderungen durch AST-Analyse vor.
        
        Diese Methode analysiert den AST des ursprünglichen Codes und schlägt Änderungen vor,
        die das Verbesserungspotenzial adressieren. Sie ist besonders nützlich für strukturelle
        Änderungen und Optimierungen.
        """
        # Initialisiere Ergebnis
        result = {
            "modified_code": original_code,  # Standardmäßig unverändert
            "changes": [],
            "description": "",
            "expected_benefits": []
        }
        
        # Extrahiere Informationen aus dem Verbesserungspotenzial
        target = improvement_opportunity.get("target", "")
        action = improvement_opportunity.get("action", "")
        category = improvement_opportunity.get("category", "")
        
        # Dictionary mit Klassen und Funktionen im Modul
        classes_and_functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes_and_functions[node.name] = {
                    "type": "class",
                    "node": node,
                    "lineno": node.lineno,
                    "functions": []
                }
            elif isinstance(node, ast.FunctionDef):
                # Prüfe, ob die Funktion in einer Klasse ist
                parent_class = None
                for class_name, class_info in classes_and_functions.items():
                    if hasattr(node, 'parent') and node.parent == class_info["node"]:
                        parent_class = class_name
                        class_info["functions"].append(node.name)
                        break
                
                if not parent_class:  # Funktion auf Modulebene
                    classes_and_functions[node.name] = {
                        "type": "function",
                        "node": node,
                        "lineno": node.lineno
                    }
        
        # Spezifische Modifikationen basierend auf der Kategorie und dem Ziel
        modified_code = original_code
        changes = []
        expected_benefits = []
        description = ""
        
        # Backend-Optimierungen (MLX, PyTorch, etc.)
        if category.lower() == "backend_optimization" or "backend" in target.lower():
            if "mlx" in target.lower():
                modified_code, mlx_changes, mlx_benefits = self._optimize_for_mlx_backend(original_code, tree)
                changes.extend(mlx_changes)
                expected_benefits.extend(mlx_benefits)
                description = "Optimierung für Apple Neural Engine (MLX-Backend)"
            elif "torch" in target.lower() or "pytorch" in target.lower():
                modified_code, torch_changes, torch_benefits = self._optimize_for_torch_backend(original_code, tree)
                changes.extend(torch_changes)
                expected_benefits.extend(torch_benefits)
                description = "Optimierung für Metal/MPS (PyTorch-Backend)"
            elif "tensor" in target.lower() or "t_mathematics" in target.lower():
                # Generelle Tensor-Operationen optimieren
                modified_code, tensor_changes, tensor_benefits = self._optimize_tensor_operations(original_code, tree)
                changes.extend(tensor_changes)
                expected_benefits.extend(tensor_benefits)
                description = "Optimierung der Tensor-Operationen in der T-Mathematics Engine"
        
        # Integration mit M-LINGUA
        elif "m_lingua" in target.lower() or "nlp" in target.lower() or "natural_language" in target.lower():
            modified_code, lingua_changes, lingua_benefits = self._enhance_m_lingua_integration(original_code, tree)
            changes.extend(lingua_changes)
            expected_benefits.extend(lingua_benefits)
            description = "Verbesserte Integration zwischen M-LINGUA Interface und T-Mathematics Engine"
        
        # Speicheroptimierungen
        elif "memory" in target.lower() or (category.lower() == "resource_usage" and "memory" in target.lower()):
            modified_code, memory_changes, memory_benefits = self._optimize_memory_usage(original_code, tree)
            changes.extend(memory_changes)
            expected_benefits.extend(memory_benefits)
            description = "Optimierung der Speichernutzung"
        
        # CPU-Optimierungen
        elif "cpu" in target.lower() or "processing" in target.lower():
            modified_code, cpu_changes, cpu_benefits = self._optimize_cpu_usage(original_code, tree)
            changes.extend(cpu_changes)
            expected_benefits.extend(cpu_benefits)
            description = "Optimierung der CPU-Auslastung"
        
        # Ethische Ausrichtung
        elif "ethics" in category.lower() or "value" in target.lower() or "alignment" in target.lower():
            modified_code, ethics_changes, ethics_benefits = self._enhance_ethical_alignment(original_code, tree)
            changes.extend(ethics_changes)
            expected_benefits.extend(ethics_benefits)
            description = "Verbesserung der ethischen Ausrichtung und Wertebindung"
        
        # Trainings- oder Inferenzoptimierungen
        elif category.lower() in ["training", "inference"]:
            if category.lower() == "training":
                modified_code, training_changes, training_benefits = self._optimize_training_process(original_code, tree)
                changes.extend(training_changes)
                expected_benefits.extend(training_benefits)
                description = "Optimierung des Trainingsprozesses"
            else:  # inference
                modified_code, inference_changes, inference_benefits = self._optimize_inference_process(original_code, tree)
                changes.extend(inference_changes)
                expected_benefits.extend(inference_benefits)
                description = "Optimierung des Inferenzprozesses"
        
        # Fallback: Generische Code-Verbesserungen basierend auf allgemeinen Best Practices
        if not changes:  # Wenn keine spezifischen Änderungen vorgenommen wurden
            modified_code, generic_changes, generic_benefits = self._apply_generic_improvements(original_code, tree)
            changes.extend(generic_changes)
            expected_benefits.extend(generic_benefits)
            description = description or "Allgemeine Codeoptimierungen basierend auf Best Practices"
        
        # Wenn immer noch keine Änderungen, return None
        if modified_code == original_code:
            return None
        
        result["modified_code"] = modified_code
        result["changes"] = changes
        result["description"] = description
        result["expected_benefits"] = expected_benefits
        result["risk_assessment"] = self._assess_modification_risk(original_code, modified_code, changes)
        
        return result
    
    def _propose_via_templates(self, target_module: str, original_code: str, 
                              improvement_opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Schlägt Codeänderungen durch Template-basierte Methoden vor.
        
        Diese Methode verwendet vordefinierte Templates für häufige Verbesserungsmuster
        und passt sie auf den spezifischen Anwendungsfall an.
        """
        # Template-basierte Ansätze eignen sich gut für bekannte Muster und Optimierungen
        # Diese vereinfachte Implementierung dient als Platzhalter für eine vollständige Template-Engine
        
        # Extrahiere Informationen aus dem Verbesserungspotenzial
        target = improvement_opportunity.get("target", "")
        action = improvement_opportunity.get("action", "")
        category = improvement_opportunity.get("category", "")
        
        # Initialisiere Ergebnis
        result = {
            "modified_code": original_code,  # Standardmäßig unverändert
            "changes": [],
            "description": "",
            "expected_benefits": []
        }
        
        # Wähle das passende Template basierend auf dem Verbesserungspotenzial
        template_key = f"{category}_{target}_{action}".lower()
        
        # Beispiel-Templates (in einer vollständigen Implementierung würden diese aus einer Datenbank oder Konfigurationsdatei geladen)
        templates = {
            "backend_optimization_mlx_backend_optimize": self._get_mlx_optimization_template,
            "backend_optimization_torch_backend_optimize": self._get_torch_optimization_template,
            "resource_usage_memory_optimization_implement": self._get_memory_optimization_template,
            "resource_usage_cpu_usage_optimize": self._get_cpu_optimization_template,
            "integration_m_lingua_integration_enhance": self._get_m_lingua_integration_template,
            "ethics_value_alignment_enhance": self._get_ethical_alignment_template,
            "training_training_process_optimize": self._get_training_optimization_template,
            "inference_inference_process_optimize": self._get_inference_optimization_template
        }
        
        # Versuche, das passende Template zu finden
        template_found = False
        for key, template_func in templates.items():
            if key in template_key or any(k in template_key for k in key.split('_')):
                # Wende das Template an
                try:
                    modified_code, changes, benefits, description = template_func(original_code, improvement_opportunity)
                    result["modified_code"] = modified_code
                    result["changes"] = changes
                    result["expected_benefits"] = benefits
                    result["description"] = description
                    template_found = True
                    break
                except Exception as e:
                    logger.warning(f"Fehler bei der Anwendung des Templates {key}: {e}")
        
        # Wenn kein passendes Template gefunden wurde, gib None zurück
        if not template_found or result["modified_code"] == original_code:
            return None
        
        # Risikobewertung
        result["risk_assessment"] = self._assess_modification_risk(original_code, result["modified_code"], result["changes"])
        
        return result
    
    def _propose_via_hybrid_approach(self, target_module: str, original_code: str, tree: ast.AST, 
                                    improvement_opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Schlägt Codeänderungen durch einen hybriden Ansatz vor.
        
        Diese Methode kombiniert AST-Analyse und Template-basierte Methoden, um die besten
        Änderungsvorschläge zu generieren.
        """
        # Versuche zuerst den AST-Ansatz
        ast_result = self._propose_via_ast_analysis(target_module, original_code, tree, improvement_opportunity)
        
        # Wenn AST-Ansatz erfolgreich war, behalte das Ergebnis
        if ast_result and ast_result.get("modified_code") != original_code:
            ast_result["origin"] = "ast_analysis"
            return ast_result
        
        # Ansonsten versuche den Template-Ansatz
        template_result = self._propose_via_templates(target_module, original_code, improvement_opportunity)
        
        # Wenn auch der Template-Ansatz erfolgreich war, behalte das Ergebnis
        if template_result and template_result.get("modified_code") != original_code:
            template_result["origin"] = "template_based"
            return template_result
        
        # Fallback: Generische Verbesserungen
        generic_result = {
            "modified_code": original_code,
            "changes": [],
            "description": "Generische Codeoptimierungen",
            "expected_benefits": ["Verbesserte Codequalität", "Erhöhte Wartbarkeit"],
            "origin": "generic"
        }
        
        # Wende einfache generische Verbesserungen an
        modified_code, changes, benefits = self._apply_generic_improvements(original_code, tree)
        
        if modified_code != original_code:
            generic_result["modified_code"] = modified_code
            generic_result["changes"] = changes
            generic_result["expected_benefits"] = benefits
            generic_result["risk_assessment"] = self._assess_modification_risk(original_code, modified_code, changes)
            return generic_result
        
        # Wenn keine Änderung möglich ist, gib None zurück
        return None
    
    def _requires_human_review(self, modification_result: Dict[str, Any], improvement_opportunity: Dict[str, Any]) -> bool:
        """Bestimmt, ob die vorgeschlagene Änderung eine menschliche Überprüfung erfordert."""
        # Wenn auto_approval deaktiviert ist, erfordere immer menschliche Überprüfung
        if not self.enable_auto_approval:
            return True
        
        # Risikobewertung aus der Modifikation
        risk_level = modification_result.get("risk_assessment", {}).get("level", "medium")
        
        # Hohe Risiken erfordern immer menschliche Überprüfung
        if risk_level == "high":
            return True
        
        # Prüfe Kategorie und Ziel des Verbesserungspotenzials
        category = improvement_opportunity.get("category", "").lower()
        target = improvement_opportunity.get("target", "").lower()
        
        # Bestimmte sensible Bereiche erfordern immer menschliche Überprüfung
        sensitive_areas = ["ethics", "security", "privacy", "model_architecture", "core"]
        if any(area in category or area in target for area in sensitive_areas):
            return True
        
        # Größere Änderungen erfordern menschliche Überprüfung
        changes = modification_result.get("changes", [])
        if len(changes) > 5:  # Mehr als 5 Änderungen gelten als größere Änderungen
            return True
        
        # Bei niedrigem Risiko und wenigen Änderungen kann die automatische Genehmigung aktiviert werden
        return False
    
    def _save_proposed_modification(self, modification_info: Dict[str, Any]) -> None:
        """Speichert die vorgeschlagenen Modifikationen zur Nachverfolgung und Auditierung."""
        try:
            # Erstelle einen Dateinamen mit Timestamp und Modul-ID
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            target_module_id = modification_info["target_module"].replace(".", "_")
            filename = f"proposed_{target_module_id}_{modification_info['modification_id']}_{timestamp}.json"
            file_path = os.path.join(self.base_output_dir, 'proposed_modifications', filename)
            
            # Speichere als JSON
            with open(file_path, 'w') as f:
                json.dump(modification_info, f, indent=2)
            
            logger.info(f"Vorgeschlagene Modifikation gespeichert unter {file_path}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der vorgeschlagenen Modifikation: {e}")
    
    def _assess_modification_risk(self, original_code: str, modified_code: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bewertet das Risiko einer Codeänderung."""
        # Grundlegende Risikobewertung
        risk_assessment = {
            "level": "medium",  # Standard-Risikostufe
            "factors": []
        }
        
        # Faktoren, die das Risiko erhöhen
        high_risk_patterns = [
            (r"\beval\(", "Verwendung von eval()"),
            (r"\bexec\(", "Verwendung von exec()"),
            (r"\b__import__\(", "Dynamischer Import"),
            (r"\bos\.(system|popen|exec)", "Ausführung von Shell-Befehlen"),
            (r"\bsubprocess\.(Popen|call|run)", "Ausführung von Subprozessen"),
            (r"\bglob[als]{2,3}\(\)\.\w+\s*=\s*", "Modifikation globaler Variablen"),
            (r"\b(open|file)\([^)]*,\s*(['\"]*)([wa+])", "Schreiben in Dateien")
        ]
        
        # Prüfe Risiko-Patterns im modifizierten Code
        risk_factors = []
        for pattern, description in high_risk_patterns:
            if re.search(pattern, modified_code) and not re.search(pattern, original_code):
                risk_factors.append({
                    "pattern": pattern,
                    "description": description,
                    "severity": "high"
                })
        
        # Bewerte das Risiko basierend auf der Anzahl und Art der Änderungen
        if len(changes) > 10:
            risk_factors.append({
                "pattern": "large_changeset",
                "description": "Große Anzahl von Änderungen",
                "severity": "medium"
            })
        
        # Überprüfe Zeilenabdeckung der Änderungen (prozentual zum Gesamtcode)
        original_lines = original_code.count('\n')
        changed_lines = sum(1 for change in changes if "lines" in change)
        if original_lines > 0 and (changed_lines / original_lines) > 0.3:  # Mehr als 30% des Codes geändert
            risk_factors.append({
                "pattern": "high_coverage",
                "description": "Hohe Abdeckung der Änderungen im Code",
                "severity": "medium"
            })
        
        # Bestimme die Gesamtrisikostufe
        high_severity_count = sum(1 for factor in risk_factors if factor["severity"] == "high")
        medium_severity_count = sum(1 for factor in risk_factors if factor["severity"] == "medium")
        
        if high_severity_count > 0:
            risk_level = "high"
        elif medium_severity_count > 2:
            risk_level = "high"
        elif medium_severity_count > 0:
            risk_level = "medium"
        elif len(changes) > 0:
            risk_level = "low"
        else:
            risk_level = "none"
        
        risk_assessment["level"] = risk_level
        risk_assessment["factors"] = risk_factors
        
        return risk_assessment
    
    def _optimize_for_mlx_backend(self, original_code: str, tree: ast.AST) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """Optimiert Code für das MLX-Backend und die Apple Neural Engine.
        
        Args:
            original_code: Ursprünglicher Code
            tree: AST des ursprünglichen Codes
            
        Returns:
            Tuple aus (optimierter Code, Liste von Änderungen, Liste von erwarteten Vorteilen)
        """
        changes = []
        expected_benefits = [
            "Verbesserte Performance auf Apple Neural Engine",
            "Optimierte Tensoroperationen für MLX",
            "Reduzierter Speicherverbrauch bei ML-Operationen"
        ]
        
        # Suche nach Tensor-bezogenen Klassen und Funktionen
        tensor_classes = []
        tensor_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Suche nach Tensor-bezogenen Klassen
                if "tensor" in node.name.lower() or any("tensor" in base.id.lower() for base in node.bases if hasattr(base, 'id')):
                    tensor_classes.append(node)
            elif isinstance(node, ast.FunctionDef):
                # Suche nach Tensor-bezogenen Funktionen
                if "tensor" in node.name.lower() or any("mlx" in arg.lower() for arg in [a.arg for a in node.args.args if hasattr(a, 'arg')]):
                    tensor_functions.append(node)
        
        # Optimiere MLXTensor-Implementierung, falls vorhanden
        mlx_optimizations = [
            {
                "pattern": r"for\s+i\s+in\s+range\(.*\):\s*\n\s*result\s*\+=\s*.*",
                "replacement": "# Vektorisierte Operation mit MLX\n        result = mlx.core.reduce_sum(values, axis=0)",
                "description": "Ersetzung von Schleifen durch vektorisierte MLX-Operationen"
            },
            {
                "pattern": r"np\.array\((.*?)\)",
                "replacement": "mlx.core.array(\\1)",
                "description": "Ersetzung von NumPy-Arrays durch MLX-Arrays"
            },
            {
                "pattern": r"torch\.tensor\((.*?)\)",
                "replacement": "mlx.core.array(\\1)",
                "description": "Ersetzung von PyTorch-Tensoren durch MLX-Arrays"
            },
            {
                "pattern": r"x\.numpy\(\)",
                "replacement": "x.astype(float).tolist()",
                "description": "Optimierte Konvertierung von MLX-Arrays zu Python-Listen"
            }
        ]
        
        # Optimiere MLX-spezifische Codeblöcke
        modified_code = original_code
        
        # Erkenne, ob MLX bereits importiert ist
        mlx_import_exists = re.search(r"import\s+mlx(\.\w+)*|from\s+mlx(\.\w+)*\s+import", original_code)
        
        # Füge MLX-Import hinzu, falls nicht vorhanden
        if not mlx_import_exists and (any("mlx" in original_code.lower() for _ in range(1))):
            import_pos = 0
            # Finde Position nach den anderen Imports
            for match in re.finditer(r"^import\s+|^from\s+\w+\s+import", original_code, re.MULTILINE):
                import_pos = max(import_pos, match.end())
            
            # Finde die Zeile, die den letzten Import enthält
            lines = original_code.split('\n')
            import_line = 0
            for i, line in enumerate(lines):
                if re.match(r"^import\s+|^from\s+\w+\s+import", line):
                    import_line = i
            
            # Füge MLX-Import nach dem letzten Import ein
            if import_line > 0:
                lines.insert(import_line + 1, "import mlx.core\nimport mlx.nn")
                modified_code = '\n'.join(lines)
                changes.append({
                    "type": "addition",
                    "description": "MLX-Module importiert",
                    "lines": [import_line + 1, import_line + 2]
                })
        
        # Wende Optimierungen auf den gesamten Code an
        for opt in mlx_optimizations:
            pattern = opt["pattern"]
            replacement = opt["replacement"]
            description = opt["description"]
            
            # Finde alle Vorkommen des Patterns
            for match in re.finditer(pattern, modified_code):
                # Ersetze das Pattern
                start, end = match.span()
                original_snippet = modified_code[start:end]
                replaced_snippet = re.sub(pattern, replacement, original_snippet)
                
                if original_snippet != replaced_snippet:
                    # Berechne Zeilennummern
                    lines_before = modified_code[:start].count('\n') + 1
                    lines_after = lines_before + original_snippet.count('\n')
                    
                    # Ersetze im Code
                    modified_code = modified_code[:start] + replaced_snippet + modified_code[end:]
                    
                    # Dokumentiere Änderung
                    changes.append({
                        "type": "optimization",
                        "description": description,
                        "original": original_snippet,
                        "replacement": replaced_snippet,
                        "lines": [lines_before, lines_after]
                    })
        
        # Optimiere spezifisch MLXTensor-Klasse, falls vorhanden
        mlx_tensor_class_pattern = r"class\s+MLXTensor\s*\(.*\):\s*[^\}]+"
        mlx_tensor_class_match = re.search(mlx_tensor_class_pattern, modified_code, re.DOTALL)
        
        if mlx_tensor_class_match:
            class_start, class_end = mlx_tensor_class_match.span()
            class_content = modified_code[class_start:class_end]
            
            # Füge Optimierungen für Apple Neural Engine hinzu
            optimized_methods = [
                '\n    def to_ane(self):\n        """Optimiert den Tensor für die Apple Neural Engine."""\n        return self  # MLX nutzt die ANE automatisch\n',
                '\n    def optimize_layout(self):\n        """Optimiert das Speicherlayout für bessere Performance."""\n        # MLX optimiert das Layout automatisch\n        return self\n'
            ]
            
            # Füge die optimierten Methoden am Ende der Klasse hinzu
            last_method_pos = class_content.rfind('\n    def ')
            if last_method_pos > 0:
                # Finde das Ende der letzten Methode
                last_method_end = class_content.find('\n\n', last_method_pos)
                if last_method_end == -1:  # Keine Leerzeile am Ende
                    last_method_end = len(class_content)
                
                # Füge optimierte Methoden hinzu
                updated_class = class_content[:last_method_end] + ''.join(optimized_methods)
                modified_code = modified_code[:class_start] + updated_class + modified_code[class_end:]
                
                # Dokumentiere Änderung
                lines_before = modified_code[:class_start].count('\n') + 1
                lines_after = lines_before + class_content.count('\n')
                
                changes.append({
                    "type": "enhancement",
                    "description": "MLXTensor-Klasse mit Apple Neural Engine Optimierungen erweitert",
                    "lines": [lines_before, lines_after]
                })
                
                # Füge Benefit hinzu
                expected_benefits.append("Automatische Nutzung der Apple Neural Engine (ANE)")
        
        return modified_code, changes, expected_benefits
    
    def _optimize_for_torch_backend(self, original_code: str, tree: ast.AST) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """Optimiert Code für das PyTorch-Backend und Metal Performance Shaders (MPS).
        
        Args:
            original_code: Ursprünglicher Code
            tree: AST des ursprünglichen Codes
            
        Returns:
            Tuple aus (optimierter Code, Liste von Änderungen, Liste von erwarteten Vorteilen)
        """
        changes = []
        expected_benefits = [
            "Verbesserte Performance auf Metal GPU",
            "Optimierte PyTorch-Operationen für MPS",
            "Automatische Tensorauslagerung auf die GPU"
        ]
        
        # Suche nach Tensor-bezogenen Klassen und Funktionen
        tensor_classes = []
        tensor_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Suche nach Tensor-bezogenen Klassen
                if "tensor" in node.name.lower() or any("tensor" in base.id.lower() for base in node.bases if hasattr(base, 'id')):
                    tensor_classes.append(node)
            elif isinstance(node, ast.FunctionDef):
                # Suche nach Tensor-bezogenen Funktionen
                if "tensor" in node.name.lower() or any("torch" in arg.lower() for arg in [a.arg for a in node.args.args if hasattr(a, 'arg')]):
                    tensor_functions.append(node)
        
        # Optimiere TorchTensor-Implementierung, falls vorhanden
        torch_optimizations = [
            {
                "pattern": r"device\s*=\s*torch\.device\(['\"]cpu['\"]\)",
                "replacement": "device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')",
                "description": "Automatische Nutzung von Metal Performance Shaders (MPS) falls verfügbar"
            },
            {
                "pattern": r"torch\.tensor\((.*?)\)",
                "replacement": r"torch.tensor(\1, device=device)",
                "description": "Tensoren direkt auf dem optimalen Device erstellen"
            },
            {
                "pattern": r"x\.to\(['"]cuda['"]\)",
                "replacement": "x.to(device)",
{{ ... }}
            class_start, class_end = torch_tensor_class_match.span()
            class_content = modified_code[class_start:class_end]
            
            # Füge Optimierungen für MPS hinzu
            optimized_methods = [
                '\n    def to_mps(self):\n        """Verlagert den Tensor auf die Metal GPU."""\n        if torch.backends.mps.is_available():\n            return self.data.to("mps")\n        return self\n',
                '\n    def optimize_for_metal(self):\n        """Optimiert den Tensor für Metal Performance Shaders."""\n        # Verwende 16-bit Genauigkeit für bessere Performance\n        if torch.backends.mps.is_available():\n            return self.data.to("mps").to(torch.float16)\n        return self\n'
            ]
            
            # Füge die optimierten Methoden am Ende der Klasse hinzu
            last_method_pos = class_content.rfind('\n    def ')
            if last_method_pos > 0:
{{ ... }}
                last_method_end = class_content.find('\n\n', last_method_pos)
                if last_method_end == -1:  # Keine Leerzeile am Ende
                    last_method_end = len(class_content)
                
                # Füge optimierte Methoden hinzu
                updated_class = class_content[:last_method_end] + ''.join(optimized_methods)
                modified_code = modified_code[:class_start] + updated_class + modified_code[class_end:]
                
                # Dokumentiere Änderung
                lines_before = modified_code[:class_start].count('\n') + 1
                lines_after = lines_before + class_content.count('\n')
                
                changes.append({
                    "type": "enhancement",
                    "description": "TorchTensor-Klasse mit Metal GPU Optimierungen erweitert",
                    "lines": [lines_before, lines_after]
                })
                
                # Füge Benefit hinzu
                expected_benefits.append("Optimierte Nutzung der Metal GPU durch PyTorch MPS")
        
        return modified_code, changes, expected_benefits
