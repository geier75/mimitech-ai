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

Copyright (c) 2025 VXOR AI
Alle Rechte vorbehalten.
"""

import os
import re
import sys
import ast
import json
import uuid
import time
import shutil
import logging
import tempfile
import datetime
import subprocess
import importlib.util
from typing import Dict, List, Tuple, Any, Optional, Union, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/self_rewriter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SelfRewriter")


class SelfRewriter:
    """SelfRewriter ermöglicht es dem VXOR-System, Code-Änderungen vorzuschlagen,
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
        """Lädt die Konfigurationsdatei.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            
        Returns:
            Dict mit Konfigurationsdaten
        """
        default_config = {
            'output_dir': './output',
            'sandbox_dir': './output/sandbox',
            'modules_dir': './miso',
            'benchmark_dir': './tests',
            'max_modification_attempts': 3,
            'min_improvement_threshold': 0.05,
            'safety_checks_required': True,
            'require_tests': True,
            'enable_auto_approval': False,
            'code_generation_backend': 'ast_analysis',
            'high_risk_patterns': [
                r'os\.rmdir',
                r'shutil\.rmtree',
                r'os\.remove',
                r'__del__',
                r'system\(',
                r'subprocess\.call\([\'"]rm ',
                r'urllib\.request\.urlopen',
                r'requests\..*\(',
                r'eval\(',
                r'exec\('
            ]
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Konfiguration: {str(e)}")
                return default_config
        else:
            logger.warning(f"Konfigurationsdatei {config_path} nicht gefunden. Verwende Standardkonfiguration.")
            return default_config
    
    def _setup_directories(self) -> None:
        """Erstellt die erforderlichen Verzeichnisse."""
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.sandbox_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, 'proposals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, 'simulations'), exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, 'approved'), exist_ok=True)
        os.makedirs(os.path.join(self.base_output_dir, 'rejected'), exist_ok=True)
    
    def _scan_codebase(self) -> None:
        """Scannt die Codebase, um Modulinformationen zu sammeln."""
        logger.info(f"Scanne Codebase in {self.modules_dir}")
        
        # Zurücksetzen der Lookup-Tabelle
        self.module_lookup = {}
        
        try:
            # Durchsuche das Modulverzeichnis rekursiv
            for root, dirs, files in os.walk(self.modules_dir):
                # Ignoriere versteckte Verzeichnisse
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for filename in files:
                    if filename.endswith('.py') and not filename.startswith('__'):
                        file_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(file_path, self.modules_dir)
                        
                        # Konvertiere den relativen Pfad in einen Modulnamen
                        module_parts = []
                        path_parts = os.path.dirname(rel_path).split(os.sep)
                        if path_parts[0]:  # Nicht leerer String
                            module_parts.extend(path_parts)
                        
                        module_name = filename[:-3]  # Entferne .py
                        if module_parts:
                            module_name = '.'.join(module_parts + [module_name])
                        
                        # Füge zur Lookup-Tabelle hinzu
                        self.module_lookup[module_name] = file_path
            
            logger.info(f"{len(self.module_lookup)} Module gefunden")
            
        except Exception as e:
            logger.error(f"Fehler beim Scannen der Codebase: {str(e)}")
    
    def _prepare_sandbox(self, target_module: str) -> str:
        """Bereitet eine Sandbox-Umgebung für ein Modul vor.
        
        Args:
            target_module: Name des Zielmoduls (z.B. 'miso.core.model')
            
        Returns:
            Pfad zur Sandbox-Umgebung
        """
        # Erstelle ein eindeutiges Sandbox-Verzeichnis
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sandbox_name = f"sandbox_{target_module.replace('.', '_')}_{timestamp}"
        sandbox_path = os.path.join(self.sandbox_dir, sandbox_name)
        
        os.makedirs(sandbox_path, exist_ok=True)
        logger.info(f"Sandbox erstellt: {sandbox_path}")
        
        return sandbox_path
    
    def _copy_dependencies(self, target_module: str, sandbox_path: str) -> None:
        """Kopiert abhängige Module in die Sandbox.
        
        Args:
            target_module: Name des Zielmoduls
            sandbox_path: Pfad zur Sandbox-Umgebung
        """
        # Finde den Pfad zum Zielmodul
        if target_module not in self.module_lookup:
            logger.error(f"Modul {target_module} nicht gefunden")
            return
        
        target_path = self.module_lookup[target_module]
        
        try:
            # Lese den Code des Zielmoduls
            with open(target_path, 'r') as f:
                target_code = f.read()
            
            # Extrahiere Importe
            import_pattern = r'^import\s+([\w\.]+)|^from\s+([\w\.]+)\s+import'
            potential_dependencies = set()
            
            for match in re.finditer(import_pattern, target_code, re.MULTILINE):
                imported_module = match.group(1) or match.group(2)
                if imported_module.startswith('miso.'):
                    potential_dependencies.add(imported_module)
            
            # Kopiere gefundene Abhängigkeiten
            for dep_module in potential_dependencies:
                if dep_module in self.module_lookup:
                    dep_path = self.module_lookup[dep_module]
                    
                    # Erstelle Zielverzeichnis
                    module_parts = dep_module.split('.')
                    target_dir = os.path.join(sandbox_path, *module_parts[:-1])
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # Kopiere Modul
                    target_file = os.path.join(target_dir, f"{module_parts[-1]}.py")
                    shutil.copy2(dep_path, target_file)
                    
                    # Erstelle __init__.py Dateien im Pfad
                    current_path = sandbox_path
                    for part in module_parts[:-1]:
                        current_path = os.path.join(current_path, part)
                        init_file = os.path.join(current_path, "__init__.py")
                        if not os.path.exists(init_file):
                            with open(init_file, 'w') as f:
                                f.write("# Automatisch generiert\n")
            
            logger.info(f"Abhängigkeiten für {target_module} kopiert")
            
        except Exception as e:
            logger.error(f"Fehler beim Kopieren der Abhängigkeiten: {str(e)}")
    
    def _run_tests(self, module_name: str, sandbox_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Führt Tests für ein Modul in der Sandbox aus.
        
        Args:
            module_name: Name des Moduls
            sandbox_path: Pfad zur Sandbox-Umgebung
            
        Returns:
            Tupel aus (Erfolgsstatus, Testergebnisse)
        """
        logger.info(f"Führe Tests für {module_name} aus")
        
        # Konstruiere Testmodul-Name
        if module_name.startswith('miso.'):
            test_module = f"tests.{module_name[5:]}"  # Entferne 'miso.' Präfix
        else:
            test_module = f"tests.{module_name}"
        
        # Konstruiere Testdateiname
        test_file = f"test_{module_name.replace('.', '_')}.py"
        
        # Konstruiere Testpfad im Benchmark-Verzeichnis
        test_module_parts = test_module.split('.')
        test_dir = os.path.join(self.benchmark_dir, *test_module_parts[:-1])
        test_path = os.path.join(test_dir, test_file)
        
        # Prüfe, ob Testdatei existiert
        if not os.path.exists(test_path):
            logger.warning(f"Keine Tests für {module_name} gefunden (gesucht: {test_path})")
            return False, {"status": "no_tests", "message": f"Keine Tests für {module_name} gefunden"}
        
        try:
            # Kopiere Testdatei in die Sandbox
            sandbox_test_dir = os.path.join(sandbox_path, *test_module_parts[:-1])
            os.makedirs(sandbox_test_dir, exist_ok=True)
            
            # Erstelle __init__.py Dateien im Testpfad
            current_path = os.path.join(sandbox_path, test_module_parts[0])
            os.makedirs(current_path, exist_ok=True)
            
            init_file = os.path.join(current_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("# Automatisch generiert\n")
            
            for part in test_module_parts[1:-1]:
                current_path = os.path.join(current_path, part)
                os.makedirs(current_path, exist_ok=True)
                
                init_file = os.path.join(current_path, "__init__.py")
                if not os.path.exists(init_file):
                    with open(init_file, 'w') as f:
                        f.write("# Automatisch generiert\n")
            
            # Kopiere die Testdatei
            sandbox_test_path = os.path.join(sandbox_test_dir, test_file)
            shutil.copy2(test_path, sandbox_test_path)
            
            # Führe Tests mit pytest aus
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{sandbox_path}:{env.get('PYTHONPATH', '')}"
            
            # Zeitmessung für Performance-Vergleich
            start_time = time.time()
            
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
            
            results = {
                "status": "success" if success else "failure",
                "output": output,
                "execution_time": execution_time,
                "return_code": process.returncode
            }
            
            logger.info(f"Tests für {module_name}: {'Erfolg' if success else 'Fehlschlag'}")
            
            return success, results
            
        except Exception as e:
            logger.error(f"Fehler beim Ausführen der Tests: {str(e)}")
            return False, {"status": "error", "message": str(e)}
    
    def _check_safety(self, original_code: str, modified_code: str) -> Tuple[bool, List[str]]:
        """Führt Sicherheitsprüfungen für Codeänderungen durch.
        
        Args:
            original_code: Ursprünglicher Code
            modified_code: Geänderter Code
            
        Returns:
            Tupel aus (Sicherheitsstatus, Liste von Sicherheitsproblemen)
        """
        safety_issues = []
        
        # Prüfe auf potenziell gefährliche Patterns
        high_risk_patterns = self.config.get('high_risk_patterns', [
            r'os\.rmdir',
            r'shutil\.rmtree',
            r'os\.remove',
            r'__del__',
            r'system\(',
            r'subprocess\.call\([\'"]rm ',
            r'urllib\.request\.urlopen',
            r'requests\..*\(',
            r'eval\(',
            r'exec\('
        ])
        
        for pattern in high_risk_patterns:
            # Prüfe, ob ein gefährliches Pattern im modifizierten Code hinzugefügt wurde
            original_matches = re.findall(pattern, original_code)
            modified_matches = re.findall(pattern, modified_code)
            
            if len(modified_matches) > len(original_matches):
                safety_issues.append(f"Potenziell gefährliches Pattern hinzugefügt: {pattern}")
        
        # Prüfe auf Syntax-Fehler im modifizierten Code
        try:
            ast.parse(modified_code)
        except SyntaxError as e:
            safety_issues.append(f"Syntaxfehler im modifizierten Code: {str(e)}")
        
        # Prüfe auf Imports von potenziell gefährlichen Modulen, die im Original nicht vorhanden waren
        dangerous_imports = ['os', 'subprocess', 'sys', 'shutil', 'urllib', 'requests']
        
        original_import_pattern = r'^import\s+([\w\.]+)|^from\s+([\w\.]+)\s+import'
        modified_import_pattern = r'^import\s+([\w\.]+)|^from\s+([\w\.]+)\s+import'
        
        original_imports = set()
        for match in re.finditer(original_import_pattern, original_code, re.MULTILINE):
            imported_module = match.group(1) or match.group(2)
            original_imports.add(imported_module.split('.')[0])
        
        modified_imports = set()
        for match in re.finditer(modified_import_pattern, modified_code, re.MULTILINE):
            imported_module = match.group(1) or match.group(2)
            modified_imports.add(imported_module.split('.')[0])
        
        new_imports = modified_imports - original_imports
        dangerous_new_imports = [imp for imp in new_imports if imp in dangerous_imports]
        
        if dangerous_new_imports:
            safety_issues.append(f"Potenziell gefährliche Module importiert: {', '.join(dangerous_new_imports)}")
        
        # Gibt zurück, ob der Code sicher ist (keine Sicherheitsprobleme gefunden)
        is_safe = len(safety_issues) == 0
        
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
        logger.info(f"Schlage Code-Modifikationen für {target_module} vor, basierend auf: {improvement_opportunity.get('category', 'unbekannt')}")
        
        try:
            # Überprüfe, ob das Zielmodul existiert
            if target_module not in self.module_lookup:
                logger.error(f"Zielmodul {target_module} nicht gefunden")
                return {
                    "status": "failed",
                    "error": f"Zielmodul {target_module} nicht gefunden"
                }
            
            # Lade den ursprünglichen Code
            original_file_path = self.module_lookup[target_module]
            with open(original_file_path, 'r') as f:
                original_code = f.read()
            
            # Parse AST für die Code-Analyse
            tree = ast.parse(original_code)
            
            # Generiere die Änderungsvorschläge basierend auf dem ausgewählten Backend
            if self.code_generation_backend == "ast_analysis":
                proposal = self._propose_via_ast_analysis(target_module, original_code, tree, improvement_opportunity)
            elif self.code_generation_backend == "template_based":
                proposal = self._propose_via_templates(target_module, original_code, improvement_opportunity)
            elif self.code_generation_backend == "hybrid":
                proposal = self._propose_via_hybrid_approach(target_module, original_code, tree, improvement_opportunity)
            else:
                logger.error(f"Unbekanntes Code-Generierungs-Backend: {self.code_generation_backend}")
                return {
                    "status": "failed",
                    "error": f"Unbekanntes Code-Generierungs-Backend: {self.code_generation_backend}"
                }
            
            # Wenn keine Änderungen vorgeschlagen wurden, Fehler zurückgeben
            if not proposal:
                logger.warning(f"Keine Änderungen für {target_module} vorgeschlagen")
                return {
                    "status": "no_changes",
                    "message": "Keine Änderungen vorgeschlagen",
                    "target_module": target_module
                }
            
            # Extrahiere Ergebnisse aus dem Vorschlag
            modified_code = proposal["modified_code"]
            changes = proposal["changes"]
            expected_benefits = proposal["expected_benefits"]
            
            # Führe Sicherheitsprüfungen durch
            if self.safety_checks_required:
                is_safe, safety_issues = self._check_safety(original_code, modified_code)
                if not is_safe:
                    logger.warning(f"Sicherheitsprüfung fehlgeschlagen: {safety_issues}")
                    return {
                        "status": "unsafe",
                        "target_module": target_module,
                        "safety_issues": safety_issues
                    }
            
            # Bewerte das Risiko der Änderung
            risk_assessment = self._assess_modification_risk(original_code, modified_code, changes)
            
            # Erstelle eine eindeutige ID für diese Modifikation
            modification_id = f"MOD-{uuid.uuid4().hex[:8]}_{target_module.replace('.', '_')}"
            
            # Erstelle den vollständigen Modifikationsvorschlag
            modification_proposal = {
                "status": "success",
                "modification_id": modification_id,
                "target_module": target_module,
                "improvement_opportunity": improvement_opportunity,
                "original_code": original_code,
                "modified_code": modified_code,
                "changes": changes,
                "expected_benefits": expected_benefits,
                "risk_assessment": risk_assessment,
                "requires_human_review": self._requires_human_review(risk_assessment, improvement_opportunity),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Speichere den Vorschlag für Audit-Trails
            self._save_proposed_modification(modification_proposal)
            
            logger.info(f"Code-Modifikation {modification_id} für {target_module} vorgeschlagen")
            
            return modification_proposal
            
        except Exception as e:
            logger.error(f"Fehler beim Vorschlagen von Code-Modifikationen: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "status": "failed",
                "error": str(e),
                "target_module": target_module
            }
    
    def _propose_via_ast_analysis(self, target_module: str, original_code: str, tree: ast.AST, 
                                 improvement_opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Schlägt Codeänderungen durch AST-Analyse vor.
        
        Diese Methode analysiert den AST des ursprünglichen Codes und schlägt Änderungen vor,
        die das Verbesserungspotenzial adressieren. Sie ist besonders nützlich für strukturelle
        Änderungen und Optimierungen.
        
        Args:
            target_module: Name des Zielmoduls
            original_code: Ursprünglicher Code
            tree: AST des ursprünglichen Codes
            improvement_opportunity: Verbesserungspotenzial aus dem RecursiveEvaluator
            
        Returns:
            Dict mit Informationen über die vorgeschlagene Modifikation oder None
        """
        logger.info(f"Verwende AST-Analyse für {target_module}")
        
        category = improvement_opportunity.get('category', '')
        subcategory = improvement_opportunity.get('subcategory', '')
        details = improvement_opportunity.get('details', {})
        
        # Initialisiere das Ergebnis
        result = {
            "modified_code": original_code,
            "changes": [],
            "expected_benefits": []
        }
        
        # Kategoriebasierte Optimierungen
        if category == "performance":
            if subcategory in ["tensor_operations", "math_operations", "numerical_efficiency"]:
                # Optimiere Tensor-Operationen basierend auf der Hardware
                modified_code, changes, benefits = self._optimize_tensor_operations(target_module, original_code, tree)
                
                result["modified_code"] = modified_code
                result["changes"].extend(changes)
                result["expected_benefits"].extend(benefits)
                
            elif subcategory == "memory_usage":
                # Implementiere speichereffizientere Algorithmen
                modified_code, changes, benefits = self._optimize_memory_usage(original_code, tree)
                
                result["modified_code"] = modified_code
                result["changes"].extend(changes)
                result["expected_benefits"].extend(benefits)
                
        elif category == "reliability":
            if subcategory == "error_handling":
                # Verbessere Fehlerbehandlung durch mehr try-except-Blöcke
                modified_code, changes, benefits = self._improve_error_handling(original_code, tree)
                
                result["modified_code"] = modified_code
                result["changes"].extend(changes)
                result["expected_benefits"].extend(benefits)
                
        elif category == "backend_specific" and "backend" in details:
            # Backend-spezifische Optimierungen
            backend = details.get("backend")
            
            if backend == "mlx":
                # Optimiere für Apple Neural Engine / MLX
                modified_code, changes, benefits = self._optimize_for_mlx_backend(original_code, tree)
                
                result["modified_code"] = modified_code
                result["changes"].extend(changes)
                result["expected_benefits"].extend(benefits)
                
            elif backend == "pytorch" or backend == "torch":
                # Optimiere für PyTorch / MPS
                modified_code, changes, benefits = self._optimize_for_torch_backend(original_code, tree)
                
                result["modified_code"] = modified_code
                result["changes"].extend(changes)
                result["expected_benefits"].extend(benefits)
        
        # Falls keine Änderungen vorgeschlagen wurden, gib None zurück
        if result["modified_code"] == original_code or not result["changes"]:
            return None
        
        return result
    
    def _propose_via_templates(self, target_module: str, original_code: str, 
                              improvement_opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Schlägt Codeänderungen durch Template-basierte Methoden vor.
        
        Diese Methode verwendet vordefinierte Templates für häufige Verbesserungsmuster
        und passt sie auf den spezifischen Anwendungsfall an.
        
        Args:
            target_module: Name des Zielmoduls
            original_code: Ursprünglicher Code
            improvement_opportunity: Verbesserungspotenzial aus dem RecursiveEvaluator
            
        Returns:
            Dict mit Informationen über die vorgeschlagene Modifikation oder None
        """
        logger.info(f"Verwende Template-basierte Analyse für {target_module}")
        
        category = improvement_opportunity.get('category', '')
        subcategory = improvement_opportunity.get('subcategory', '')
        
        # Initialisiere das Ergebnis
        result = {
            "modified_code": original_code,
            "changes": [],
            "expected_benefits": []
        }
        
        # Kategoriebasierte Templates
        if category == "performance":
            if subcategory == "loop_optimization":
                # Ersetze for-Schleifen durch List Comprehensions oder vektorisierte Operationen
                modified_code = re.sub(
                    r"for\s+i\s+in\s+range\((\d+)\):\s*\n\s*result\.append\(([^\)]+)\)",
                    r"result = [\2 for i in range(\1)]",
                    original_code
                )
                
                if modified_code != original_code:
                    result["modified_code"] = modified_code
                    result["changes"].append({
                        "type": "optimization",
                        "description": "For-Schleife durch List Comprehension ersetzt"
                    })
                    result["expected_benefits"].append("Verbesserte Schleifenperformance durch List Comprehensions")
            
            elif subcategory == "function_caching":
                # Füge Funktionscaching für rechenintensive Funktionen hinzu
                if "@lru_cache" not in original_code and "def " in original_code:
                    # Importiere functools, falls noch nicht vorhanden
                    if "import functools" not in original_code and "from functools import" not in original_code:
                        import_line = "import functools\n"
                        modified_code = import_line + original_code
                    else:
                        modified_code = original_code
                    
                    # Füge @lru_cache zu geeigneten Funktionen hinzu
                    modified_code = re.sub(
                        r"(def\s+(\w+)\s*\([^\)]*\)\s*:)\s*\n",
                        r"@functools.lru_cache(maxsize=128)\n\1\n",
                        modified_code
                    )
                    
                    if modified_code != original_code:
                        result["modified_code"] = modified_code
                        result["changes"].append({
                            "type": "optimization",
                            "description": "Funktionscaching mit @lru_cache hinzugefügt"
                        })
                        result["expected_benefits"].append("Verbesserte Performance durch Funktionscaching")
        
        elif category == "reliability":
            if subcategory == "input_validation":
                # Füge Eingabevalidierung zu Funktionen hinzu
                modified_code = re.sub(
                    r"(def\s+(\w+)\s*\(([^\)]*)\)\s*:)\s*\n",
                    r"\1\n    # Validiere Eingabeparameter\n    if not all(var is not None for var in [\3]):\n        raise ValueError(\"Parameter dürfen nicht None sein\")\n",
                    original_code
                )
                
                if modified_code != original_code:
                    result["modified_code"] = modified_code
                    result["changes"].append({
                        "type": "enhancement",
                        "description": "Eingabevalidierung zu Funktionen hinzugefügt"
                    })
                    result["expected_benefits"].append("Verbesserte Fehlertoleranz durch Eingabevalidierung")
        
        # Falls keine Änderungen vorgeschlagen wurden, gib None zurück
        if result["modified_code"] == original_code or not result["changes"]:
            return None
        
        return result
    
    def _propose_via_hybrid_approach(self, target_module: str, original_code: str, tree: ast.AST, 
                                    improvement_opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Schlägt Codeänderungen durch einen hybriden Ansatz vor.
        
        Diese Methode kombiniert AST-Analyse und Template-basierte Methoden, um die besten
        Änderungsvorschläge zu generieren.
        
        Args:
            target_module: Name des Zielmoduls
            original_code: Ursprünglicher Code
            tree: AST des ursprünglichen Codes
            improvement_opportunity: Verbesserungspotenzial aus dem RecursiveEvaluator
            
        Returns:
            Dict mit Informationen über die vorgeschlagene Modifikation oder None
        """
        logger.info(f"Verwende hybriden Ansatz für {target_module}")
        
        # Versuche zuerst AST-Analyse
        ast_proposal = self._propose_via_ast_analysis(target_module, original_code, tree, improvement_opportunity)
        
        # Wenn AST-Analyse erfolgreich war, verwende diese
        if ast_proposal:
            modified_code = ast_proposal["modified_code"]
            changes = ast_proposal["changes"]
            expected_benefits = ast_proposal["expected_benefits"]
        else:
            # Sonst versuche Template-basierten Ansatz
            template_proposal = self._propose_via_templates(target_module, original_code, improvement_opportunity)
            
            if template_proposal:
                modified_code = template_proposal["modified_code"]
                changes = template_proposal["changes"]
                expected_benefits = template_proposal["expected_benefits"]
            else:
                # Wenn beide Ansätze keine Änderungen vorschlagen, gib None zurück
                return None
        
        # Ergebnis zusammenbauen
        result = {
            "modified_code": modified_code,
            "changes": changes,
            "expected_benefits": expected_benefits
        }
        
        return result
    
    def _requires_human_review(self, modification_result: Dict[str, Any], improvement_opportunity: Dict[str, Any]) -> bool:
        """Bestimmt, ob die vorgeschlagene Änderung eine menschliche Überprüfung erfordert.
        
        Args:
            modification_result: Ergebnis der Code-Modifikation
            improvement_opportunity: Ursprüngliches Verbesserungspotenzial
            
        Returns:
            True, wenn menschliche Überprüfung erforderlich ist, sonst False
        """
        # Hochrisiko-Kategorien erfordern immer menschliche Überprüfung
        high_risk_categories = ["security", "core_algorithm", "api_change"]
        category = improvement_opportunity.get('category', '')
        
        if category in high_risk_categories:
            return True
        
        # Änderungen mit hohem Risiko erfordern menschliche Überprüfung
        risk_level = modification_result.get('risk_level', 'medium')
        if risk_level == 'high':
            return True
        
        # Überprüfe, ob die Änderungen öffentliche APIs betreffen
        changes = modification_result.get('changes', [])
        for change in changes:
            if change.get('type') == 'api_change':
                return True
        
        # Auto-Approval ist deaktiviert
        if not self.enable_auto_approval:
            return True
        
        return False
    
    def _save_proposed_modification(self, modification_info: Dict[str, Any]) -> None:
        """Speichert die vorgeschlagenen Modifikationen zur Nachverfolgung und Auditierung.
        
        Args:
            modification_info: Informationen über die vorgeschlagene Modifikation
        """
        # Füge zur internen Liste hinzu
        self.proposed_modifications.append(modification_info)
        
        # Speichere als JSON-Datei
        modification_id = modification_info.get('modification_id', f"mod-{uuid.uuid4().hex[:8]}")
        output_file = os.path.join(self.base_output_dir, 'proposals', f"{modification_id}.json")
        
        with open(output_file, 'w') as f:
            json.dump(modification_info, f, indent=2)
        
        logger.info(f"Modifikationsvorschlag in {output_file} gespeichert")
    
    def _assess_modification_risk(self, original_code: str, modified_code: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bewertet das Risiko einer Codeänderung.
        
        Args:
            original_code: Ursprünglicher Code
            modified_code: Geänderter Code
            changes: Liste von Änderungen
            
        Returns:
            Dict mit Risikobewertung
        """
        # Zähle die Anzahl der geänderten Zeilen
        original_lines = original_code.split('\n')
        modified_lines = modified_code.split('\n')
        
        # Einfache Metrik: Prozentsatz der geänderten Zeilen
        changed_lines_count = 0
        for change in changes:
            if 'lines' in change:
                line_range = change['lines']
                changed_lines_count += line_range[1] - line_range[0] + 1
        
        change_percentage = (changed_lines_count / len(original_lines)) * 100 if original_lines else 0
        
        # Kategorisiere das Risiko
        risk_level = "low"
        if change_percentage > 30:
            risk_level = "high"
        elif change_percentage > 10:
            risk_level = "medium"
        
        # Überprüfe auf Änderungen an API-Signaturen
        api_changes = self._detect_api_changes(original_code, modified_code)
        if api_changes:
            risk_level = "high"  # API-Änderungen sind immer hochriskant
        
        # Überprüfe auf Änderungen an Tests
        test_changes = "test" in original_code.lower() and self._detect_test_changes(original_code, modified_code)
        if test_changes:
            risk_level = max(risk_level, "medium")  # Teständerungen sind mindestens mittelriskant
        
        return {
            "risk_level": risk_level,
            "change_percentage": change_percentage,
            "changed_lines_count": changed_lines_count,
            "api_changes": api_changes,
            "test_changes": test_changes
        }
        
    def _detect_api_changes(self, original_code: str, modified_code: str) -> List[Dict[str, Any]]:
        """Erkennt Änderungen an öffentlichen API-Signaturen.
        
        Args:
            original_code: Ursprünglicher Code
            modified_code: Geänderter Code
            
        Returns:
            Liste von erkannten API-Änderungen
        """
        api_changes = []
        
        # Extrahiere öffentliche Funktionen und Methoden
        original_api = self._extract_public_apis(original_code)
        modified_api = self._extract_public_apis(modified_code)
        
        # Vergleiche APIs
        for name, signature in original_api.items():
            if name in modified_api:
                if signature != modified_api[name]:
                    api_changes.append({
                        "type": "modified",
                        "name": name,
                        "original": signature,
                        "modified": modified_api[name]
                    })
            else:
                api_changes.append({
                    "type": "removed",
                    "name": name,
                    "original": signature
                })
        
        for name, signature in modified_api.items():
            if name not in original_api:
                api_changes.append({
                    "type": "added",
                    "name": name,
                    "modified": signature
                })
        
        return api_changes
    
    def _extract_public_apis(self, code: str) -> Dict[str, str]:
        """Extrahiert öffentliche API-Signaturen aus Code.
        
        Args:
            code: Code, aus dem APIs extrahiert werden sollen
            
        Returns:
            Dict mit API-Namen und Signaturen
        """
        apis = {}
        
        # Einfacher Regex für Funktions- und Methodendefinitionen
        api_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^\)]*)\)\s*"
        
        for match in re.finditer(api_pattern, code, re.MULTILINE):
            name = match.group(1)
            params = match.group(2)
            
            # Ignoriere private APIs (mit _ beginnend)
            if not name.startswith('_'):
                apis[name] = params.strip()
        
        return apis
    
    def _detect_test_changes(self, original_code: str, modified_code: str) -> bool:
        """Erkennt Änderungen an Tests.
        
        Args:
            original_code: Ursprünglicher Code
            modified_code: Geänderter Code
            
        Returns:
            True, wenn Tests geändert wurden, sonst False
        """
        # Suche nach Testfunktionen
        test_pattern = r"def\s+test_[a-zA-Z0-9_]*\s*\("
        
        original_tests = re.findall(test_pattern, original_code)
        modified_tests = re.findall(test_pattern, modified_code)
        
        # Überprüfe, ob sich Tests geändert haben
        return original_tests != modified_tests
    
    def _optimize_for_mlx_backend(self, original_code: str, tree: ast.AST) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """Optimiert Code für das MLX-Backend und die Apple Neural Engine.
        
        Args:
            original_code: Ursprünglicher Code
            tree: AST des ursprünglichen Codes
            
        Returns:
            Tuple aus (optimierter Code, Liste von Änderungen, Liste von erwarteten Vorteilen)
        """
        logger.info("Optimiere Code für MLX-Backend und Apple Neural Engine")
        
        changes = []
        expected_benefits = []
        modified_code = original_code
        
        # Importiere MLX, falls noch nicht vorhanden
        if "import mlx" not in modified_code and "from mlx import" not in modified_code:
            # Füge MLX-Import am Anfang der Datei hinzu
            import_line = "import mlx.core\n"
            
            # Finde Position zum Einfügen (nach anderen Imports)
            import_pos = 0
            for match in re.finditer(r"^import\s+|^from\s+\w+\s+import", modified_code, re.MULTILINE):
                import_pos = max(import_pos, match.end())
            
            # Finde die Zeile, die den letzten Import enthält
            lines = modified_code.split('\n')
            import_line_num = 0
            for i, line in enumerate(lines):
                if re.match(r"^import\s+|^from\s+\w+\s+import", line):
                    import_line_num = i
            
            # Füge MLX-Import nach dem letzten Import ein
            if import_line_num > 0:
                lines.insert(import_line_num + 1, import_line)
                modified_code = '\n'.join(lines)
                
                changes.append({
                    "type": "addition",
                    "description": "MLX-Import hinzugefügt",
                    "lines": [import_line_num + 1, import_line_num + 1]
                })
        
        # Sammle Tensor-bezogene Funktionen für Optimierung
        tensor_functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(base.id == "MISOTensor" for base in node.bases if isinstance(base, ast.Name)):
                # Finde MLXTensor-Klasse oder ähnliche
                tensor_functions.append(node)
            elif isinstance(node, ast.FunctionDef):
                # Suche nach Tensor-bezogenen Funktionen
                if "tensor" in node.name.lower() or any("tensor" in arg.arg.lower() for arg in node.args.args if hasattr(arg, 'arg')):
                    tensor_functions.append(node)
        
        # Optimiere NumPy-Arrays durch MLX-Arrays
        if tensor_functions and "numpy" in modified_code:
            # Ersetze numpy.array durch mlx.core.array
            numpy_pattern = r"np\.array\((.*?)\)"
            numpy_replacement = r"mlx.core.array(\1)"
            
            modified_code = re.sub(numpy_pattern, numpy_replacement, modified_code)
            
            if modified_code != original_code:
                changes.append({
                    "type": "optimization",
                    "description": "NumPy-Arrays durch MLX-Arrays ersetzt",
                })
                expected_benefits.append("Optimierte Performance durch MLX und Apple Neural Engine")
        
        # Suche nach der MLXTensor-Klassendefinition
        mlx_tensor_pattern = r"class\s+MLXTensor\s*\(.*\):\s*[^\}]+"
        mlx_tensor_match = re.search(mlx_tensor_pattern, modified_code, re.DOTALL)
        
        if mlx_tensor_match:
            class_start, class_end = mlx_tensor_match.span()
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
                changes.append({
                    "type": "enhancement",
                    "description": "MLXTensor-Klasse mit Apple Neural Engine Optimierungen erweitert",
                })
                
                # Füge Benefit hinzu
                expected_benefits.append("Optimierte Nutzung der Apple Neural Engine")
        
        return modified_code, changes, expected_benefits
    
    def _optimize_for_torch_backend(self, original_code: str, tree: ast.AST) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """Optimiert Code für das PyTorch-Backend und Metal Performance Shaders (MPS).
        
        Args:
            original_code: Ursprünglicher Code
            tree: AST des ursprünglichen Codes
            
        Returns:
            Tuple aus (optimierter Code, Liste von Änderungen, Liste von erwarteten Vorteilen)
        """
        logger.info("Optimiere Code für PyTorch-Backend und Metal Performance Shaders")
        
        changes = []
        expected_benefits = []
        modified_code = original_code
        
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
                "pattern": r"x\.to\(['\"]cuda['\"]\)",
                "replacement": "x.to(device)",
                "description": "Geräteunabhängigen Tensor-Transfer verwenden"
            },
            {
                "pattern": r"for\s+i\s+in\s+range\(.*\):\s*\n\s*result\s*\+=\s*.*",
                "replacement": "# Vektorisierte Operation mit PyTorch\n        result = torch.sum(values, dim=0)",
                "description": "Ersetzung von Schleifen durch vektorisierte PyTorch-Operationen"
            }
        ]
        
        # Optimiere PyTorch-spezifische Codeblöcke
        
        # Erkenne, ob PyTorch bereits importiert ist
        torch_import_exists = re.search(r"import\s+torch(\.[\w]+)*|from\s+torch(\.[\w]+)*\s+import", modified_code)
        
        # Füge PyTorch-Import und MPS-Setup hinzu, falls nicht vorhanden
        if not torch_import_exists and any("torch" in modified_code.lower() for _ in range(1)):
            import_pos = 0
            # Finde Position nach den anderen Imports
            for match in re.finditer(r"^import\s+|^from\s+\w+\s+import", modified_code, re.MULTILINE):
                import_pos = max(import_pos, match.end())
            
            # Finde die Zeile, die den letzten Import enthält
            lines = modified_code.split('\n')
            import_line = 0
            for i, line in enumerate(lines):
                if re.match(r"^import\s+|^from\s+\w+\s+import", line):
                    import_line = i
            
            # Füge PyTorch-Import und MPS-Setup nach dem letzten Import ein
            if import_line > 0:
                torch_import = """import torch\n# Setup für Metal Performance Shaders\ndevice = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')"""
                
                lines.insert(import_line + 1, torch_import)
                modified_code = '\n'.join(lines)
                changes.append({
                    "type": "addition",
                    "description": "PyTorch mit MPS-Support importiert"
                })
        
        # Wende Optimierungen auf den gesamten Code an
        for opt in torch_optimizations:
            pattern = opt["pattern"]
            replacement = opt["replacement"]
            description = opt["description"]
            
            # Finde alle Vorkommen des Patterns
            match_count = len(re.findall(pattern, modified_code))
            if match_count > 0:
                # Ersetze das Pattern
                modified_code = re.sub(pattern, replacement, modified_code)
                
                # Dokumentiere Änderung
                changes.append({
                    "type": "optimization",
                    "description": description
                })
        
        # Optimiere spezifisch TorchTensor-Klasse, falls vorhanden
        torch_tensor_class_pattern = r"class\s+TorchTensor\s*\(.*\):\s*[^\}]+"
        torch_tensor_class_match = re.search(torch_tensor_class_pattern, modified_code, re.DOTALL)
        
        if torch_tensor_class_match:
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
                # Finde das Ende der letzten Methode
                last_method_end = class_content.find('\n\n', last_method_pos)
                if last_method_end == -1:  # Keine Leerzeile am Ende
                    last_method_end = len(class_content)
                
                # Füge optimierte Methoden hinzu
                updated_class = class_content[:last_method_end] + ''.join(optimized_methods)
                modified_code = modified_code[:class_start] + updated_class + modified_code[class_end:]
                
                # Dokumentiere Änderung
                changes.append({
                    "type": "enhancement",
                    "description": "TorchTensor-Klasse mit Metal GPU Optimierungen erweitert"
                })
                
                # Füge Benefit hinzu
                expected_benefits.append("Optimierte Nutzung der Metal GPU durch PyTorch MPS")
        
        return modified_code, changes, expected_benefits
    
    def _optimize_memory_usage(self, original_code: str, tree: ast.AST) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """Optimiert die Speichernutzung des Codes.
        
        Args:
            original_code: Ursprünglicher Code
            tree: AST des ursprünglichen Codes
            
        Returns:
            Tuple aus (optimierter Code, Liste von Änderungen, Liste von erwarteten Vorteilen)
        """
        changes = []
        expected_benefits = []
        modified_code = original_code
        
        # Ersetze große Listen durch Generatoren/Iteratoren
        generator_pattern = r"\[([^\[\]]+)\s+for\s+([^\[\]]+)\]"
        generator_replacement = r"(\1 for \2)"
        
        # Zähle die Vorkommen
        list_comp_count = len(re.findall(generator_pattern, modified_code))
        
        if list_comp_count > 0:
            # Ersetze List Comprehensions durch Generator Expressions,
            # aber nur wenn sie nicht direkt benötigt werden
            modified_code = re.sub(
                r"(\s*)(\w+)\s*=\s*\[([^\[\]]+)\s+for\s+([^\[\]]+)\]\s*\n\s*for\s+.*?\s+in\s+\2",
                r"\1\2 = (\3 for \4)\n\1for ",
                modified_code
            )
            
            if modified_code != original_code:
                changes.append({
                    "type": "optimization",
                    "description": "List Comprehensions durch speichereffiziente Generator Expressions ersetzt"
                })
                expected_benefits.append("Reduzierte Speichernutzung durch Verwendung von Generatoren")
        
        # Füge del-Statements für große Objekte hinzu
        modified_code = re.sub(
            r"(\s*)(\w+)\s*=\s*([^\n]+?)\s*\n\s*(\w+)\s*=\s*\2\.(\w+)\([^\)]*\)\s*\n",
            r"\1\2 = \3\n\1\4 = \2.\5()\n\1del \2  # Speicher freigeben\n",
            modified_code
        )
        
        if modified_code != original_code:
            changes.append({
                "type": "optimization",
                "description": "Speicher-Freigabe für temporäre Objekte hinzugefügt"
            })
            expected_benefits.append("Verbesserte Speichereffizienz durch explizite Freigabe")
        
        return modified_code, changes, expected_benefits
    
    def _improve_error_handling(self, original_code: str, tree: ast.AST) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """Verbessert die Fehlerbehandlung im Code.
        
        Args:
            original_code: Ursprünglicher Code
            tree: AST des ursprünglichen Codes
            
        Returns:
            Tuple aus (optimierter Code, Liste von Änderungen, Liste von erwarteten Vorteilen)
        """
        changes = []
        expected_benefits = []
        modified_code = original_code
        
        # Füge try-except-Blöcke zu Funktionen hinzu, die keine haben
        functions_without_try = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Überspringe private Funktionen
                if node.name.startswith('_') and not node.name.startswith('__'):
                    continue
                
                # Überprüfe, ob die Funktion bereits try-except-Blöcke enthält
                has_try = any(isinstance(child, ast.Try) for child in ast.iter_child_nodes(node))
                
                if not has_try and node.body:
                    functions_without_try.append(node.name)
        
        # Füge try-except-Blöcke zu Funktionen ohne Fehlerbehandlung hinzu
        for func_name in functions_without_try:
            func_pattern = r"(def\s+" + func_name + r"\s*\([^\)]*\)\s*:)(\s*\n(?:(?:\s+)[^\n]*\n)*)"
            func_replacement = r"\1\2    try:\n\2        pass  # Platzhalter, wird ersetzt\n\2    except Exception as e:\n\2        logger.error(f'Fehler in {func_name}: {str(e)}')\n\2        raise\n"
            
            # Finde die Funktion und ihren Inhalt
            func_match = re.search(func_pattern, modified_code)
            if func_match:
                func_def = func_match.group(1)
                func_body_space = func_match.group(2)
                
                # Extrahiere den eigentlichen Funktionskörper
                body_pattern = r"(def\s+" + func_name + r"\s*\([^\)]*\)\s*:)(\s*\n)(((?:\s+)[^\n]*\n)*)"
                body_match = re.search(body_pattern, modified_code)
                
                if body_match:
                    body_indentation = func_body_space
                    body_content = body_match.group(3)
                    
                    # Erstelle neuen Funktionskörper mit try-except-Block
                    new_body = body_indentation + "    try:\n" + body_content
                    new_body += body_indentation + "    except Exception as e:\n"
                    new_body += body_indentation + "        logger.error(f'Fehler in " + func_name + ": {str(e)}')\n"
                    new_body += body_indentation + "        raise\n"
                    
                    # Ersetze den alten Funktionskörper durch den neuen
                    old_func = func_def + body_indentation + body_content
                    new_func = func_def + new_body
                    
                    modified_code = modified_code.replace(old_func, new_func)
                    
                    changes.append({
                        "type": "enhancement",
                        "description": f"Fehlerbehandlung zur Funktion {func_name} hinzugefügt"
                    })
        
        if changes:
            expected_benefits.append("Verbesserte Stabilität durch erweiterte Fehlerbehandlung")
        
        return modified_code, changes, expected_benefits
    
    def simulate_modifications(self, modification_info: Dict[str, Any]) -> Dict[str, Any]:
        """Simuliert die vorgeschlagenen Code-Änderungen in einer isolierten Sandbox-Umgebung.
        
        Diese Methode nutzt die separate simulate_modifications.py-Implementierung, um
        die vorgeschlagenen Änderungen in einer sicheren Umgebung zu testen.
        
        Args:
            modification_info: Informationen über die vorgeschlagene Modifikation
            
        Returns:
            Dict mit Informationen über die Simulationsergebnisse
        """
        # Importiere das dedizierte Simulationsmodul
        from .simulate_modifications import simulate_modifications as sim_mod
        
        # Führe die Simulation durch
        try:
            simulation_result = sim_mod(
                modification_info,
                sandbox_dir=self.sandbox_dir,
                modules_dir=self.modules_dir,
                benchmark_dir=self.benchmark_dir,
                require_tests=self.require_tests
            )
            
            # Speichere Simulationsergebnis für Audit-Trail
            self.simulated_modifications.append(simulation_result)
            
            return simulation_result
        except Exception as e:
            logger.error(f"Fehler bei der Simulation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "status": "failed",
                "modification_id": modification_info.get("modification_id", ""),
                "error": str(e)
            }
    
    def approve_or_reject_changes(self, simulation_result: Dict[str, Any], 
                                  modification_info: Dict[str, Any]) -> Dict[str, Any]:
        """Bewertet Simulationsergebnisse und entscheidet über die Annahme oder Ablehnung von Änderungen.
        
        Diese Methode nutzt die separate approve_or_reject_changes.py-Implementierung für
        die Entscheidungsfindung und Anwendung der Sicherheitsrichtlinien.
        
        Args:
            simulation_result: Ergebnisse der Simulation
            modification_info: Informationen über die vorgeschlagene Änderung
            
        Returns:
            Dict mit Informationen über die Entscheidung
        """
        # Importiere das dedizierte Entscheidungsmodul
        from .approve_or_reject_changes import approve_or_reject_changes as arc
        
        try:
            # Führe die Entscheidungslogik aus
            config_path = os.path.join(self.base_output_dir, "config", "rsi_approval_config.json")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            decision_result = arc(
                simulation_result,
                modification_info,
                config_path=config_path
            )
            
            # Speichere Entscheidung für Audit-Trail
            if decision_result.get("decision") == "approve":
                self.approved_modifications.append(decision_result)
                
                # Speichere als JSON-Datei für Audit-Trail
                output_file = os.path.join(self.base_output_dir, 'approved', 
                                          f"{decision_result.get('decision_id', uuid.uuid4().hex[:8])}.json")
                with open(output_file, 'w') as f:
                    json.dump(decision_result, f, indent=2)
                
            elif decision_result.get("decision") == "reject":
                self.rejected_modifications.append(decision_result)
                
                # Speichere als JSON-Datei für Audit-Trail
                output_file = os.path.join(self.base_output_dir, 'rejected', 
                                          f"{decision_result.get('decision_id', uuid.uuid4().hex[:8])}.json")
                with open(output_file, 'w') as f:
                    json.dump(decision_result, f, indent=2)
            
            return decision_result
        except Exception as e:
            logger.error(f"Fehler bei der Entscheidungsfindung: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "status": "failed",
                "modification_id": modification_info.get("modification_id", ""),
                "error": str(e)
            }
