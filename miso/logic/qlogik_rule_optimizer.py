#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK Rule Optimizer

Optimierungsmodul für Q-LOGIK-Regeln.
Implementiert adaptive Regeloptimierung und Regellernen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field

# Importiere Q-LOGIK Komponenten
from miso.logic.qlogik_engine import (
    BayesianDecisionCore,
    FuzzyLogicUnit,
    SymbolMap,
    ConflictResolver,
    simple_emotion_weight,
    simple_priority_mapping,
    qlogik_decision,
    advanced_qlogik_decision
)

# Logger einrichten
logger = logging.getLogger("MISO.Logic.Q-LOGIK.RuleOptimizer")

@dataclass
class RulePerformanceMetrics:
    """Leistungsmetriken für eine Regel"""
    rule_id: str
    success_count: int = 0
    failure_count: int = 0
    application_count: int = 0
    average_confidence: float = 0.0
    average_execution_time: float = 0.0
    last_applied: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """Erfolgsrate der Regel"""
        if self.application_count == 0:
            return 0.0
        return self.success_count / self.application_count
    
    @property
    def failure_rate(self) -> float:
        """Fehlerrate der Regel"""
        if self.application_count == 0:
            return 0.0
        return self.failure_count / self.application_count
    
    @property
    def utility_score(self) -> float:
        """Nutzwert der Regel basierend auf Erfolgsrate und Konfidenz"""
        return self.success_rate * self.average_confidence
    
    def update_metrics(self, success: bool, confidence: float, execution_time: float) -> None:
        """Aktualisiert die Metriken nach einer Regelanwendung"""
        self.application_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            
        # Aktualisiere durchschnittliche Konfidenz
        self.average_confidence = (
            (self.average_confidence * (self.application_count - 1) + confidence) / 
            self.application_count
        )
        
        # Aktualisiere durchschnittliche Ausführungszeit
        self.average_execution_time = (
            (self.average_execution_time * (self.application_count - 1) + execution_time) / 
            self.application_count
        )
        
        self.last_applied = time.time()


class RuleOptimizer:
    """
    Regeloptimierer für Q-LOGIK
    
    Optimiert Regeln basierend auf Leistungsmetriken und Feedback.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den RuleOptimizer
        
        Args:
            config: Konfigurationsobjekt für den RuleOptimizer
        """
        self.config = config or {}
        self.rule_metrics = {}  # rule_id -> RulePerformanceMetrics
        self.rule_cache = {}    # rule_id -> rule
        self.optimization_history = []
        
        # Lade Regeln aus Konfigurationsdatei, falls vorhanden
        rules_file = self.config.get("rules_file", None)
        if rules_file and os.path.exists(rules_file):
            try:
                with open(rules_file, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                    for rule_id, rule in rules_data.items():
                        self.rule_cache[rule_id] = rule
                        self.rule_metrics[rule_id] = RulePerformanceMetrics(rule_id=rule_id)
                logger.info(f"{len(self.rule_cache)} Regeln aus {rules_file} geladen")
            except Exception as e:
                logger.error(f"Fehler beim Laden der Regeln: {e}")
        
        logger.info("RuleOptimizer initialisiert")
    
    def register_rule(self, rule_id: str, rule: Dict[str, Any]) -> bool:
        """
        Registriert eine neue Regel
        
        Args:
            rule_id: ID der Regel
            rule: Regeldefinition
            
        Returns:
            True, wenn erfolgreich registriert, sonst False
        """
        if rule_id in self.rule_cache:
            logger.warning(f"Regel {rule_id} existiert bereits")
            return False
        
        self.rule_cache[rule_id] = rule
        self.rule_metrics[rule_id] = RulePerformanceMetrics(rule_id=rule_id)
        logger.info(f"Regel {rule_id} registriert")
        return True
    
    def update_rule(self, rule_id: str, rule: Dict[str, Any]) -> bool:
        """
        Aktualisiert eine bestehende Regel
        
        Args:
            rule_id: ID der Regel
            rule: Aktualisierte Regeldefinition
            
        Returns:
            True, wenn erfolgreich aktualisiert, sonst False
        """
        if rule_id not in self.rule_cache:
            logger.warning(f"Regel {rule_id} existiert nicht")
            return False
        
        # Speichere alte Version für Optimierungshistorie
        old_rule = self.rule_cache[rule_id]
        
        # Aktualisiere Regel
        self.rule_cache[rule_id] = rule
        
        # Füge Änderung zur Optimierungshistorie hinzu
        self.optimization_history.append({
            "rule_id": rule_id,
            "timestamp": time.time(),
            "action": "update",
            "old_rule": old_rule,
            "new_rule": rule,
            "metrics_before": self._get_metrics_snapshot(rule_id)
        })
        
        logger.info(f"Regel {rule_id} aktualisiert")
        return True
    
    def delete_rule(self, rule_id: str) -> bool:
        """
        Löscht eine Regel
        
        Args:
            rule_id: ID der zu löschenden Regel
            
        Returns:
            True, wenn erfolgreich gelöscht, sonst False
        """
        if rule_id not in self.rule_cache:
            logger.warning(f"Regel {rule_id} existiert nicht")
            return False
        
        # Speichere alte Version für Optimierungshistorie
        old_rule = self.rule_cache[rule_id]
        old_metrics = self._get_metrics_snapshot(rule_id)
        
        # Lösche Regel
        del self.rule_cache[rule_id]
        del self.rule_metrics[rule_id]
        
        # Füge Änderung zur Optimierungshistorie hinzu
        self.optimization_history.append({
            "rule_id": rule_id,
            "timestamp": time.time(),
            "action": "delete",
            "old_rule": old_rule,
            "metrics_before": old_metrics
        })
        
        logger.info(f"Regel {rule_id} gelöscht")
        return True
    
    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """
        Gibt eine Regel zurück
        
        Args:
            rule_id: ID der Regel
            
        Returns:
            Regeldefinition oder None, falls nicht gefunden
        """
        return self.rule_cache.get(rule_id)
    
    def get_all_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Gibt alle Regeln zurück
        
        Returns:
            Dictionary mit allen Regeln
        """
        return self.rule_cache
    
    def get_rule_metrics(self, rule_id: str) -> Optional[RulePerformanceMetrics]:
        """
        Gibt die Metriken einer Regel zurück
        
        Args:
            rule_id: ID der Regel
            
        Returns:
            RulePerformanceMetrics oder None, falls nicht gefunden
        """
        return self.rule_metrics.get(rule_id)
    
    def get_all_rule_metrics(self) -> Dict[str, RulePerformanceMetrics]:
        """
        Gibt die Metriken aller Regeln zurück
        
        Returns:
            Dictionary mit allen Regelmetriken
        """
        return self.rule_metrics
    
    def record_rule_application(self, rule_id: str, success: bool, confidence: float, execution_time: float) -> bool:
        """
        Zeichnet eine Regelanwendung auf
        
        Args:
            rule_id: ID der angewendeten Regel
            success: True, wenn die Anwendung erfolgreich war, sonst False
            confidence: Konfidenz der Regelanwendung (0.0-1.0)
            execution_time: Ausführungszeit in Sekunden
            
        Returns:
            True, wenn erfolgreich aufgezeichnet, sonst False
        """
        if rule_id not in self.rule_metrics:
            logger.warning(f"Regel {rule_id} existiert nicht")
            return False
        
        metrics = self.rule_metrics[rule_id]
        metrics.update_metrics(success, confidence, execution_time)
        logger.debug(f"Regelanwendung für {rule_id} aufgezeichnet: success={success}, confidence={confidence:.2f}")
        return True
    
    def optimize_rules(self, optimization_strategy: str = "performance") -> Dict[str, Any]:
        """
        Optimiert Regeln basierend auf Leistungsmetriken
        
        Args:
            optimization_strategy: Optimierungsstrategie ("performance", "confidence", "balanced")
            
        Returns:
            Optimierungsergebnisse
        """
        if not self.rule_metrics:
            logger.warning("Keine Regeln zum Optimieren vorhanden")
            return {"optimized_rules": 0, "error": "Keine Regeln vorhanden"}
        
        start_time = time.time()
        optimized_rules = []
        
        # Sortiere Regeln nach Optimierungspotenzial
        if optimization_strategy == "performance":
            # Optimiere Regeln mit schlechter Leistung zuerst
            rules_to_optimize = sorted(
                self.rule_metrics.items(),
                key=lambda x: x[1].success_rate
            )
        elif optimization_strategy == "confidence":
            # Optimiere Regeln mit niedriger Konfidenz zuerst
            rules_to_optimize = sorted(
                self.rule_metrics.items(),
                key=lambda x: x[1].average_confidence
            )
        else:  # balanced
            # Optimiere Regeln mit niedrigem Nutzwert zuerst
            rules_to_optimize = sorted(
                self.rule_metrics.items(),
                key=lambda x: x[1].utility_score
            )
        
        # Optimiere die unteren 30% der Regeln
        rules_to_optimize = rules_to_optimize[:max(1, len(rules_to_optimize) // 3)]
        
        for rule_id, metrics in rules_to_optimize:
            rule = self.rule_cache.get(rule_id)
            if not rule:
                continue
                
            # Speichere alte Version für Optimierungshistorie
            old_rule = rule.copy()
            
            # Optimiere Regel basierend auf Metriken
            optimized_rule = self._optimize_single_rule(rule, metrics, optimization_strategy)
            
            if optimized_rule != rule:
                # Aktualisiere Regel
                self.rule_cache[rule_id] = optimized_rule
                
                # Füge Änderung zur Optimierungshistorie hinzu
                self.optimization_history.append({
                    "rule_id": rule_id,
                    "timestamp": time.time(),
                    "action": "optimize",
                    "strategy": optimization_strategy,
                    "old_rule": old_rule,
                    "new_rule": optimized_rule,
                    "metrics_before": self._get_metrics_snapshot(rule_id)
                })
                
                optimized_rules.append(rule_id)
                logger.info(f"Regel {rule_id} optimiert mit Strategie {optimization_strategy}")
        
        end_time = time.time()
        
        result = {
            "optimized_rules": len(optimized_rules),
            "rule_ids": optimized_rules,
            "strategy": optimization_strategy,
            "execution_time": end_time - start_time
        }
        
        logger.info(f"Regeloptimierung abgeschlossen: {len(optimized_rules)} Regeln optimiert")
        return result
    
    def save_rules(self, file_path: str = None) -> bool:
        """
        Speichert alle Regeln in eine Datei
        
        Args:
            file_path: Pfad zur Zieldatei (optional)
            
        Returns:
            True, wenn erfolgreich gespeichert, sonst False
        """
        if not file_path:
            file_path = self.config.get("rules_file", "qlogik_rules.json")
            
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.rule_cache, f, indent=2, ensure_ascii=False)
            logger.info(f"{len(self.rule_cache)} Regeln in {file_path} gespeichert")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Regeln: {e}")
            return False
    
    def _optimize_single_rule(self, 
                             rule: Dict[str, Any], 
                             metrics: RulePerformanceMetrics,
                             strategy: str) -> Dict[str, Any]:
        """
        Optimiert eine einzelne Regel
        
        Args:
            rule: Zu optimierende Regel
            metrics: Leistungsmetriken der Regel
            strategy: Optimierungsstrategie
            
        Returns:
            Optimierte Regel
        """
        optimized_rule = rule.copy()
        
        # Optimiere basierend auf Metriken und Strategie
        if strategy == "performance":
            # Wenn die Erfolgsrate niedrig ist, passe die Bedingungen an
            if metrics.success_rate < 0.5:
                if "conditions" in optimized_rule:
                    # Lockere die Bedingungen etwas
                    for condition_key, condition in optimized_rule["conditions"].items():
                        if "threshold" in condition:
                            # Reduziere den Schwellenwert um 10%
                            threshold = condition["threshold"]
                            if isinstance(threshold, (int, float)):
                                condition["threshold"] = threshold * 0.9
                        elif "min" in condition and "max" in condition:
                            # Erweitere den Bereich um 10%
                            min_val = condition["min"]
                            max_val = condition["max"]
                            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                                range_size = max_val - min_val
                                condition["min"] = max(0, min_val - range_size * 0.05)
                                condition["max"] = max_val + range_size * 0.05
        
        elif strategy == "confidence":
            # Wenn die Konfidenz niedrig ist, erhöhe die Gewichtung
            if metrics.average_confidence < 0.7:
                if "weight" in optimized_rule:
                    # Erhöhe die Gewichtung um 10%
                    weight = optimized_rule["weight"]
                    if isinstance(weight, (int, float)):
                        optimized_rule["weight"] = min(1.0, weight * 1.1)
        
        else:  # balanced
            # Kombiniere beide Strategien
            if metrics.utility_score < 0.5:
                # Passe sowohl Bedingungen als auch Gewichtung an
                if "conditions" in optimized_rule:
                    for condition_key, condition in optimized_rule["conditions"].items():
                        if "threshold" in condition:
                            threshold = condition["threshold"]
                            if isinstance(threshold, (int, float)):
                                condition["threshold"] = threshold * 0.95
                
                if "weight" in optimized_rule:
                    weight = optimized_rule["weight"]
                    if isinstance(weight, (int, float)):
                        optimized_rule["weight"] = min(1.0, weight * 1.05)
        
        return optimized_rule
    
    def _get_metrics_snapshot(self, rule_id: str) -> Dict[str, Any]:
        """
        Erstellt einen Snapshot der Metriken einer Regel
        
        Args:
            rule_id: ID der Regel
            
        Returns:
            Dictionary mit Metrik-Snapshot
        """
        metrics = self.rule_metrics.get(rule_id)
        if not metrics:
            return {}
            
        return {
            "success_rate": metrics.success_rate,
            "failure_rate": metrics.failure_rate,
            "application_count": metrics.application_count,
            "average_confidence": metrics.average_confidence,
            "average_execution_time": metrics.average_execution_time,
            "utility_score": metrics.utility_score,
            "last_applied": metrics.last_applied
        }


# Globale Instanz für einfachen Zugriff
rule_optimizer = RuleOptimizer()

def register_rule(rule_id: str, rule: Dict[str, Any]) -> bool:
    """
    Registriert eine neue Regel
    
    Args:
        rule_id: ID der Regel
        rule: Regeldefinition
        
    Returns:
        True, wenn erfolgreich registriert, sonst False
    """
    return rule_optimizer.register_rule(rule_id, rule)

def update_rule(rule_id: str, rule: Dict[str, Any]) -> bool:
    """
    Aktualisiert eine bestehende Regel
    
    Args:
        rule_id: ID der Regel
        rule: Aktualisierte Regeldefinition
        
    Returns:
        True, wenn erfolgreich aktualisiert, sonst False
    """
    return rule_optimizer.update_rule(rule_id, rule)

def optimize_rules(optimization_strategy: str = "balanced") -> Dict[str, Any]:
    """
    Optimiert Regeln basierend auf Leistungsmetriken
    
    Args:
        optimization_strategy: Optimierungsstrategie ("performance", "confidence", "balanced")
        
    Returns:
        Optimierungsergebnisse
    """
    return rule_optimizer.optimize_rules(optimization_strategy)

def get_rule_metrics(rule_id: str) -> Optional[Dict[str, Any]]:
    """
    Gibt die Metriken einer Regel zurück
    
    Args:
        rule_id: ID der Regel
        
    Returns:
        Dictionary mit Metriken oder None, falls nicht gefunden
    """
    metrics = rule_optimizer.get_rule_metrics(rule_id)
    if not metrics:
        return None
        
    return {
        "rule_id": metrics.rule_id,
        "success_rate": metrics.success_rate,
        "failure_rate": metrics.failure_rate,
        "application_count": metrics.application_count,
        "average_confidence": metrics.average_confidence,
        "average_execution_time": metrics.average_execution_time,
        "utility_score": metrics.utility_score,
        "last_applied": metrics.last_applied,
        "created_at": metrics.created_at
    }


if __name__ == "__main__":
    # Beispiel für die Verwendung des RuleOptimizers
    logging.basicConfig(level=logging.INFO)
    
    # Erstelle Beispielregeln
    example_rules = {
        "rule1": {
            "name": "Hohe Priorität bei hohem Risiko",
            "description": "Weist hohe Priorität zu, wenn das Risiko hoch ist",
            "conditions": {
                "risk": {"threshold": 0.7}
            },
            "weight": 0.8,
            "action": "prioritize_safety"
        },
        "rule2": {
            "name": "Mittlere Priorität bei hohem Nutzen",
            "description": "Weist mittlere Priorität zu, wenn der Nutzen hoch ist",
            "conditions": {
                "benefit": {"threshold": 0.7}
            },
            "weight": 0.7,
            "action": "prioritize_utility"
        },
        "rule3": {
            "name": "Niedrige Priorität bei niedriger Dringlichkeit",
            "description": "Weist niedrige Priorität zu, wenn die Dringlichkeit niedrig ist",
            "conditions": {
                "urgency": {"max": 0.3}
            },
            "weight": 0.5,
            "action": "delay"
        }
    }
    
    # Registriere Beispielregeln
    for rule_id, rule in example_rules.items():
        rule_optimizer.register_rule(rule_id, rule)
    
    # Simuliere Regelanwendungen
    rule_optimizer.record_rule_application("rule1", True, 0.85, 0.005)
    rule_optimizer.record_rule_application("rule1", True, 0.90, 0.004)
    rule_optimizer.record_rule_application("rule1", False, 0.60, 0.006)
    
    rule_optimizer.record_rule_application("rule2", True, 0.75, 0.003)
    rule_optimizer.record_rule_application("rule2", False, 0.65, 0.004)
    
    rule_optimizer.record_rule_application("rule3", True, 0.95, 0.002)
    
    # Optimiere Regeln
    result = rule_optimizer.optimize_rules("balanced")
    print(f"Optimierungsergebnis: {result}")
    
    # Zeige optimierte Regeln
    for rule_id in result.get("rule_ids", []):
        rule = rule_optimizer.get_rule(rule_id)
        metrics = rule_optimizer.get_rule_metrics(rule_id)
        print(f"Optimierte Regel {rule_id}:")
        print(f"  - Erfolgsrate: {metrics.success_rate:.2f}")
        print(f"  - Durchschnittliche Konfidenz: {metrics.average_confidence:.2f}")
        print(f"  - Nutzwert: {metrics.utility_score:.2f}")
        print(f"  - Regel: {rule}")
