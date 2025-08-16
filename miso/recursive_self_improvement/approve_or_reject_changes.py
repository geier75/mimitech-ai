#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
approve_or_reject_changes.py

Modul zur Bewertung und Entscheidungsfindung über vorgeschlagene Code-Änderungen im Rahmen des
Recursive Self-Improvement (RSI) Systems. Enthält Funktionen zur Analyse von Simulationsergebnissen, 
Anwendung von Sicherheitsrichtlinien und Entscheidung über Annahme oder Ablehnung von Änderungen.
"""

import os
import re
import sys
import json
import uuid
import logging
import datetime
import importlib.util
from typing import Dict, List, Tuple, Any, Optional, Union, Set

# Setup logging mit audit trail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/rsi_approval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ApprovalSystem")

class ApprovalManager:
    """Manager zur Bewertung und Entscheidung über die Annahme oder Ablehnung
    von vorgeschlagenen Code-Änderungen im RSI-System."""
    
    def __init__(self, config_path: str = "config/rsi_approval_config.json"):
        """Initialisiert den ApprovalManager.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
        """
        self.config = self._load_config(config_path)
        self.audit_log = []
        
        # Sicherheitsparameter
        self.safety_thresholds = self.config.get("safety_thresholds", {
            "minimum_test_coverage": 80.0,
            "maximum_performance_degradation": -5.0,  # Negative Werte = Verbesserung
            "maximum_memory_increase": 10.0,
            "high_risk_patterns_threshold": 2
        })
        
        # Schwellwerte für automatische Entscheidung
        self.auto_approval_thresholds = self.config.get("auto_approval_thresholds", {
            "minimum_performance_improvement": 10.0,
            "minimum_test_success_rate": 100.0,
            "maximum_static_analysis_issues": 0
        })
        
        # Integriere mit T-Mathematics Engine und M-LINGUA
        self.enable_tensor_optimizations = self.config.get("enable_tensor_optimizations", True)
        self.enable_m_lingua_integration = self.config.get("enable_m_lingua_integration", True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Lädt die Konfigurationsdatei.
        
        Args:
            config_path: Pfad zur Konfigurationsdatei
            
        Returns:
            Dict mit Konfigurationsdaten
        """
        default_config = {
            "safety_thresholds": {
                "minimum_test_coverage": 80.0,
                "maximum_performance_degradation": -5.0,
                "maximum_memory_increase": 10.0,
                "high_risk_patterns_threshold": 2
            },
            "auto_approval_thresholds": {
                "minimum_performance_improvement": 10.0,
                "minimum_test_success_rate": 100.0,
                "maximum_static_analysis_issues": 0
            },
            "human_review_required_for": [
                "security_related",
                "core_algorithms",
                "api_changes",
                "new_dependencies"
            ],
            "zero_tolerance_patterns": [
                r"os\.rmdir",
                r"shutil\.rmtree",
                r"os\.remove",
                r"__del__",
                r"system\(",
                r"subprocess\.call\(['\"]rm ",
                r"urllib\.request\.urlopen",
                r"requests\..*\(",
                r"eval\("
            ],
            "enable_tensor_optimizations": True,
            "enable_m_lingua_integration": True,
            "audit_log_path": "logs/approval_audit.json"
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                # Erstelle Standardkonfiguration, wenn die Datei nicht existiert
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(default_config, indent=4, sort_keys=True, f)
                logger.info(f"Standardkonfiguration in {config_path} erstellt")
                return default_config
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfiguration: {str(e)}")
            return default_config
    
    def approve_or_reject_changes(
        self, 
        simulation_result: Dict[str, Any],
        modification_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Bewertet Simulationsergebnisse und entscheidet über die Annahme oder Ablehnung von Änderungen.
        
        Diese Funktion analysiert die Simulationsergebnisse, wendet Sicherheitsrichtlinien an
        und entscheidet basierend auf verschiedenen Metriken und Schwellwerten, ob eine Änderung
        akzeptiert oder abgelehnt werden sollte. Alle Entscheidungen werden protokolliert.
        
        Args:
            simulation_result: Ergebnisse der Simulation
            modification_info: Informationen über die vorgeschlagene Änderung
            
        Returns:
            Dict mit Informationen über die Entscheidung
        """
        # Generiere einzigartige Entscheidungs-ID
        decision_id = f"DECISION-{uuid.uuid4().hex[:8]}"
        logger.info(f"Bewerte Änderung {modification_info.get('modification_id', 'unbekannt')} (Entscheidung {decision_id})")
        
        # Initialisiere Entscheidungsergebnis
        decision_result = {
            "decision_id": decision_id,
            "modification_id": modification_info.get("modification_id", ""),
            "simulation_id": simulation_result.get("simulation_id", ""),
            "timestamp": datetime.datetime.now().isoformat(),
            "decision": "pending",
            "reasons": [],
            "metrics": {},
            "requires_human_review": False
        }
        
        try:
            # 1. Prüfe ob Simulation erfolgreich war
            if simulation_result.get("status") != "success":
                decision_result["decision"] = "reject"
                decision_result["reasons"].append("Simulation fehlgeschlagen")
                logger.info(f"Änderung {decision_id} abgelehnt: Simulation fehlgeschlagen")
                return decision_result
            
            # 2. Prüfe, ob Sicherheitsrichtlinien verletzt wurden
            zero_tolerance_mode = self._zero_tolerance_check(modification_info.get("modified_code", ""))
            if zero_tolerance_mode:
                decision_result["decision"] = "reject"
                decision_result["reasons"].append("Zero-Tolerance-Pattern gefunden")
                logger.warning(f"Änderung {decision_id} abgelehnt: Zero-Tolerance-Pattern gefunden: {zero_tolerance_mode}")
                return decision_result
            
            # 3. Sammle Metriken
            metrics = {}
            
            # Performance-Änderung
            perf_change = simulation_result.get("performance_change", 0.0)
            metrics["performance_change"] = perf_change
            
            # Test-Erfolg
            tests_passed = simulation_result.get("tests_passed", False)
            metrics["tests_passed"] = tests_passed
            
            # Statische Analyse-Probleme
            static_analysis = simulation_result.get("static_analysis", {})
            high_severity_issues = [i for i in static_analysis.get("issues", []) if i.get("severity") == "high"]
            metrics["high_severity_issues"] = len(high_severity_issues)
            metrics["total_issues"] = len(static_analysis.get("issues", []))
            
            # Speichere Metriken im Ergebnis
            decision_result["metrics"] = metrics
            
            # 4. Entscheide, ob menschliche Überprüfung erforderlich ist
            requires_human_review = self._check_if_human_review_required(modification_info)
            decision_result["requires_human_review"] = requires_human_review
            
            if requires_human_review:
                decision_result["decision"] = "pending_review"
                decision_result["reasons"].append("Menschliche Überprüfung erforderlich")
                logger.info(f"Änderung {decision_id} erfordert menschliche Überprüfung")
                return decision_result
            
            # 5. Automatische Entscheidung basierend auf Metriken
            # Bedingungen für automatische Ablehnung
            if not tests_passed:
                decision_result["decision"] = "reject"
                decision_result["reasons"].append("Tests fehlgeschlagen")
                logger.info(f"Änderung {decision_id} abgelehnt: Tests fehlgeschlagen")
                return decision_result
            
            if len(high_severity_issues) > self.safety_thresholds.get("high_risk_patterns_threshold", 2):
                decision_result["decision"] = "reject"
                decision_result["reasons"].append(f"{len(high_severity_issues)} kritische Probleme in statischer Analyse gefunden")
                logger.info(f"Änderung {decision_id} abgelehnt: Zu viele kritische Probleme")
                return decision_result
            
            if perf_change < self.safety_thresholds.get("maximum_performance_degradation", -5.0):
                decision_result["decision"] = "reject"
                decision_result["reasons"].append(f"Performance-Verschlechterung um {-perf_change:.2f}%")
                logger.info(f"Änderung {decision_id} abgelehnt: Performance-Verschlechterung")
                return decision_result
            
            # Bedingungen für automatische Genehmigung
            auto_approve_conditions = [
                perf_change >= self.auto_approval_thresholds.get("minimum_performance_improvement", 10.0),
                tests_passed,
                metrics["total_issues"] <= self.auto_approval_thresholds.get("maximum_static_analysis_issues", 0)
            ]
            
            if all(auto_approve_conditions):
                decision_result["decision"] = "approve"
                decision_result["reasons"].append(f"Automatische Genehmigung: Performance-Verbesserung um {perf_change:.2f}%, Tests erfolgreich, keine kritischen Probleme")
                logger.info(f"Änderung {decision_id} automatisch genehmigt")
            else:
                # Sonst zur menschlichen Überprüfung
                decision_result["decision"] = "pending_review"
                decision_result["reasons"].append("Menschliche Überprüfung empfohlen")
                logger.info(f"Änderung {decision_id} zur menschlichen Überprüfung empfohlen")
            
            # 6. Spezielle Prüfung für T-Mathematics Engine Optimierungen
            if self.enable_tensor_optimizations and self._is_tensor_optimization(modification_info):
                tensor_decision = self._evaluate_tensor_optimization(modification_info, simulation_result)
                decision_result["tensor_optimization"] = tensor_decision
                
                # Überschreibe Entscheidung, wenn Tensor-Optimierung kritisch ist
                if tensor_decision.get("critical_issue", False):
                    decision_result["decision"] = "reject"
                    decision_result["reasons"].append("Kritisches Problem in Tensor-Optimierung")
                    logger.warning(f"Änderung {decision_id} abgelehnt: Kritisches Problem in Tensor-Optimierung")
                    return decision_result
            
            # 7. Spezielle Prüfung für M-LINGUA Integration
            if self.enable_m_lingua_integration and self._is_m_lingua_integration(modification_info):
                m_lingua_decision = self._evaluate_m_lingua_integration(modification_info, simulation_result)
                decision_result["m_lingua_integration"] = m_lingua_decision
                
                # Überschreibe Entscheidung, wenn M-LINGUA-Integration kritisch ist
                if m_lingua_decision.get("critical_issue", False):
                    decision_result["decision"] = "reject"
                    decision_result["reasons"].append("Kritisches Problem in M-LINGUA-Integration")
                    logger.warning(f"Änderung {decision_id} abgelehnt: Kritisches Problem in M-LINGUA-Integration")
                    return decision_result
                
        except Exception as e:
            logger.error(f"Fehler bei der Bewertung: {str(e)}")
            decision_result["decision"] = "error"
            decision_result["reasons"].append(f"Interner Fehler: {str(e)}")
        finally:
            # Protokolliere die Entscheidung
            self._log_decision(decision_result)
        
        return decision_result
    
    def _zero_tolerance_check(self, code: str) -> Optional[str]:
        """Überprüft, ob der Code Zero-Tolerance-Patterns enthält.
        
        Args:
            code: Der zu überprüfende Code
            
        Returns:
            Gefundenes Pattern oder None
        """
        zero_tolerance_patterns = self.config.get("zero_tolerance_patterns", [])
        for pattern in zero_tolerance_patterns:
            if re.search(pattern, code):
                return pattern
        return None
    
    def _check_if_human_review_required(self, modification_info: Dict[str, Any]) -> bool:
        """Überprüft, ob eine menschliche Überprüfung erforderlich ist.
        
        Args:
            modification_info: Informationen über die vorgeschlagene Änderung
            
        Returns:
            True, wenn menschliche Überprüfung erforderlich ist, sonst False
        """
        # Extrahiere relevante Informationen
        target_module = modification_info.get("target_module", "")
        original_code = modification_info.get("original_code", "")
        modified_code = modification_info.get("modified_code", "")
        
        # Regeln für menschliche Überprüfung
        human_review_required_for = self.config.get("human_review_required_for", [])
        
        # 1. Prüfe auf sicherheitsrelevante Module (vereinfacht)
        if "security_related" in human_review_required_for:
            security_modules = ["miso.security", "miso.authentication", "miso.encryption", "miso.deploy"]
            if any(module in target_module for module in security_modules):
                return True
        
        # 2. Prüfe auf Kernalgorithmen
        if "core_algorithms" in human_review_required_for:
            core_modules = ["miso.core", "miso.algorithms", "miso.math", "miso.tensor"]
            if any(module in target_module for module in core_modules):
                return True
        
        # 3. Prüfe auf API-Änderungen
        if "api_changes" in human_review_required_for:
            # Einfache Heuristik: Überprüfe, ob öffentliche Methodensignaturen geändert wurden
            original_methods = self._extract_public_methods(original_code)
            modified_methods = self._extract_public_methods(modified_code)
            
            if original_methods != modified_methods:
                return True
        
        # 4. Prüfe auf neue Abhängigkeiten
        if "new_dependencies" in human_review_required_for:
            original_imports = self._extract_imports(original_code)
            modified_imports = self._extract_imports(modified_code)
            
            if modified_imports - original_imports:
                return True
        
        return False
    
    def _extract_public_methods(self, code: str) -> Set[str]:
        """Extrahiert öffentliche Methoden aus dem Code.
        
        Args:
            code: Der zu analysierende Code
            
        Returns:
            Set von öffentlichen Methodensignaturen
        """
        method_signatures = set()
        # Einfacher Regex für Methodendefinitionen
        pattern = r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*:"
        for match in re.finditer(pattern, code, re.MULTILINE):
            method_name = match.group(1)
            params = match.group(2)
            
            # Ignoriere private Methoden
            if not method_name.startswith("_"):
                method_signatures.add(f"{method_name}({params})")
        
        return method_signatures
    
    def _extract_imports(self, code: str) -> Set[str]:
        """Extrahiert Importe aus dem Code.
        
        Args:
            code: Der zu analysierende Code
            
        Returns:
            Set von importierten Modulen
        """
        imports = set()
        # Patterns für verschiedene Import-Stile
        patterns = [
            r"^\s*import\s+([\w\.]+)",  # import x
            r"^\s*from\s+([\w\.]+)\s+import",  # from x import y
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, code, re.MULTILINE):
                imports.add(match.group(1))
        
        return imports
    
    def _is_tensor_optimization(self, modification_info: Dict[str, Any]) -> bool:
        """Überprüft, ob es sich um eine Tensor-Optimierung handelt.
        
        Args:
            modification_info: Informationen über die vorgeschlagene Änderung
            
        Returns:
            True, wenn es sich um eine Tensor-Optimierung handelt, sonst False
        """
        target_module = modification_info.get("target_module", "")
        modified_code = modification_info.get("modified_code", "")
        
        # Prüfe auf tensor-bezogene Module
        if any(x in target_module for x in ["tensor", "math", "mlx", "torch"]):
            return True
        
        # Prüfe auf tensor-bezogene Klassen oder Methoden
        tensor_patterns = [
            r"class\s+(\w+Tensor)",
            r"def\s+to_(mps|ane|gpu|cpu)",
            r"MLXTensor",
            r"TorchTensor",
            r"MISOTensor"
        ]
        
        for pattern in tensor_patterns:
            if re.search(pattern, modified_code):
                return True
        
        return False
    
    def _evaluate_tensor_optimization(
        self, 
        modification_info: Dict[str, Any],
        simulation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Bewertet eine Tensor-Optimierung.
        
        Args:
            modification_info: Informationen über die vorgeschlagene Änderung
            simulation_result: Ergebnisse der Simulation
            
        Returns:
            Dict mit Bewertungsergebnissen
        """
        evaluation = {
            "is_mlx_optimization": False,
            "is_pytorch_optimization": False,
            "compatibility_issues": [],
            "performance_gain": 0.0,
            "critical_issue": False
        }
        
        modified_code = modification_info.get("modified_code", "")
        
        # Erkenne Backend-spezifische Optimierungen
        if "mlx" in modified_code or "MLXTensor" in modified_code:
            evaluation["is_mlx_optimization"] = True
        if "torch" in modified_code or "TorchTensor" in modified_code:
            evaluation["is_pytorch_optimization"] = True
        
        # Prüfe auf potenzielle Kompatibilitätsprobleme
        compatibility_issues = []
        
        # 1. Prüfe auf fehlende Backend-Fallbacks
        if evaluation["is_mlx_optimization"] and "if not mlx_available" not in modified_code:
            compatibility_issues.append("MLX-Optimierung ohne Fallback-Mechanismus")
        
        if evaluation["is_pytorch_optimization"] and "if not torch.backends.mps.is_available()" not in modified_code:
            compatibility_issues.append("PyTorch-MPS-Optimierung ohne Fallback-Mechanismus")
        
        # 2. Prüfe auf Backend-übergreifende Konsistenz
        if evaluation["is_mlx_optimization"] and evaluation["is_pytorch_optimization"]:
            if "MISOTensor" not in modified_code:
                compatibility_issues.append("Gemischte Backend-Optimierung ohne MISOTensor-Abstraktion")
        
        evaluation["compatibility_issues"] = compatibility_issues
        
        # Extrahiere Performance-Gewinn aus Simulationsergebnissen
        perf_change = simulation_result.get("performance_change", 0.0)
        evaluation["performance_gain"] = perf_change
        
        # Markiere als kritisch, wenn schwerwiegende Kompatibilitätsprobleme vorliegen
        if any("ohne Fallback" in issue for issue in compatibility_issues):
            evaluation["critical_issue"] = True
        
        return evaluation
    
    def _is_m_lingua_integration(self, modification_info: Dict[str, Any]) -> bool:
        """Überprüft, ob es sich um eine M-LINGUA-Integration handelt.
        
        Args:
            modification_info: Informationen über die vorgeschlagene Änderung
            
        Returns:
            True, wenn es sich um eine M-LINGUA-Integration handelt, sonst False
        """
        target_module = modification_info.get("target_module", "")
        modified_code = modification_info.get("modified_code", "")
        
        # Prüfe auf m-lingua-bezogene Module
        if "m_lingua" in target_module or "miso.natural_language" in target_module:
            return True
        
        # Prüfe auf M-LINGUA-bezogene Klassen oder Methoden
        m_lingua_patterns = [
            r"M[-_]LINGUA",
            r"parse_natural_language",
            r"natural_to_tensor",
            r"linguistic_parser"
        ]
        
        for pattern in m_lingua_patterns:
            if re.search(pattern, modified_code, re.IGNORECASE):
                return True
        
        return False
    
    def _evaluate_m_lingua_integration(
        self, 
        modification_info: Dict[str, Any],
        simulation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Bewertet eine M-LINGUA-Integration.
        
        Args:
            modification_info: Informationen über die vorgeschlagene Änderung
            simulation_result: Ergebnisse der Simulation
            
        Returns:
            Dict mit Bewertungsergebnissen
        """
        evaluation = {
            "natural_language_functions": [],
            "tensor_conversion_present": False,
            "issues": [],
            "critical_issue": False
        }
        
        modified_code = modification_info.get("modified_code", "")
        
        # Identifiziere natürliche Sprachfunktionen
        nl_function_pattern = r"def\s+(\w+_natural_language|\w+_nl|\w+_lingua)\s*\("
        nl_functions = re.findall(nl_function_pattern, modified_code)
        evaluation["natural_language_functions"] = nl_functions
        
        # Prüfe auf Tensor-Konvertierungsfunktionalität
        tensor_conversion_patterns = [
            r"natural_to_tensor",
            r"parse_math_expression",
            r"convert_expression_to_tensor"
        ]
        
        for pattern in tensor_conversion_patterns:
            if re.search(pattern, modified_code):
                evaluation["tensor_conversion_present"] = True
                break
        
        # Identifiziere potenzielle Probleme
        issues = []
        
        # 1. Prüfe auf fehlende Fehlerbehandlung
        if not re.search(r"try\s*:", modified_code):
            issues.append("Fehlende Fehlerbehandlung in M-LINGUA-Integration")
        
        # 2. Prüfe auf fehlende Validierung
        if not re.search(r"validate|validation|is_valid", modified_code, re.IGNORECASE):
            issues.append("Fehlende Eingabevalidierung in M-LINGUA-Integration")
        
        # 3. Prüfe auf Backend-Kompatibilität
        if not re.search(r"MISOTensor|MLXTensor|TorchTensor", modified_code):
            issues.append("Keine Integration mit der T-Mathematics Engine")
        
        evaluation["issues"] = issues
        
        # Markiere als kritisch, wenn wichtige Komponenten fehlen
        if len(nl_functions) == 0 or not evaluation["tensor_conversion_present"]:
            evaluation["critical_issue"] = True
            issues.append("Kritisch: M-LINGUA-Integration unvollständig")
        
        return evaluation
    
    def _log_decision(self, decision_result: Dict[str, Any]) -> None:
        """Protokolliert eine Entscheidung im Audit-Log.
        
        Args:
            decision_result: Die zu protokollierende Entscheidung
        """
        self.audit_log.append(decision_result)
        
        # Speichere in Datei
        audit_log_path = self.config.get("audit_log_path", "logs/approval_audit.json")
        os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
        
        try:
            # Lade vorhandene Logs, falls vorhanden
            existing_logs = []
            if os.path.exists(audit_log_path):
                with open(audit_log_path, 'r') as f:
                    existing_logs = json.load(f)
            
            # Füge neue Entscheidung hinzu
            existing_logs.append(decision_result)
            
            # Schreibe in Datei
            with open(audit_log_path, 'w') as f:
                json.dump(existing_logs, indent=4, f)
        except Exception as e:
            logger.error(f"Fehler beim Protokollieren der Entscheidung: {str(e)}")


def approve_or_reject_changes(
    simulation_result: Dict[str, Any],
    modification_info: Dict[str, Any],
    config_path: str = "config/rsi_approval_config.json"
) -> Dict[str, Any]:
    """Wrapper-Funktion für die Bewertung und Entscheidung über die Annahme oder Ablehnung von Änderungen.
    
    Args:
        simulation_result: Ergebnisse der Simulation
        modification_info: Informationen über die vorgeschlagene Änderung
        config_path: Pfad zur Konfigurationsdatei
        
    Returns:
        Dict mit Informationen über die Entscheidung
    """
    manager = ApprovalManager(config_path)
    return manager.approve_or_reject_changes(simulation_result, modification_info)


# Wenn dieses Skript direkt ausgeführt wird
if __name__ == "__main__":
    # Beispielaufruf
    test_simulation = {
        "simulation_id": "SIM-12345678",
        "status": "success",
        "tests_passed": True,
        "performance_change": 15.0,  # 15% Verbesserung
        "static_analysis": {
            "issues": []
        }
    }
    
    test_modification = {
        "modification_id": "MOD-TEST",
        "target_module": "miso.core.example",
        "original_code": "def example():\n    return 'original'\n",
        "modified_code": "def example():\n    return 'modified'\n"
    }
    
    decision = approve_or_reject_changes(test_simulation, test_modification)
    print(json.dumps(decision, indent=2))
