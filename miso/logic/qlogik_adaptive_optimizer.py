#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK Adaptive Optimizer

Adaptive Optimierungsschicht für Q-LOGIK.
Implementiert selbstlernende Optimierungsstrategien für Q-LOGIK.

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
import threading
from datetime import datetime

# Importiere Q-LOGIK Komponenten
from miso.logic.qlogik_engine import (
    BayesianDecisionCore,
    FuzzyLogicUnit,
    SymbolMap,
    ConflictResolver,
    simple_emotion_weight,
    simple_priority_mapping,
    advanced_qlogik_decision
)

# Importiere Rule Optimizer
from miso.logic.qlogik_rule_optimizer import (
    RuleOptimizer,
    RulePerformanceMetrics,
    register_rule,
    update_rule,
    optimize_rules,
    get_rule_metrics
)

# Logger einrichten
logger = logging.getLogger("MISO.Logic.Q-LOGIK.AdaptiveOptimizer")

@dataclass
class OptimizationPerformanceMetrics:
    """Leistungsmetriken für eine Optimierungsstrategie"""
    strategy_id: str
    success_count: int = 0
    failure_count: int = 0
    application_count: int = 0
    average_improvement: float = 0.0
    average_execution_time: float = 0.0
    last_applied: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """Erfolgsrate der Optimierungsstrategie"""
        if self.application_count == 0:
            return 0.0
        return self.success_count / self.application_count
    
    @property
    def utility_score(self) -> float:
        """Nutzwert der Optimierungsstrategie basierend auf Erfolgsrate und Verbesserung"""
        return self.success_rate * self.average_improvement
    
    def update_metrics(self, success: bool, improvement: float, execution_time: float) -> None:
        """Aktualisiert die Metriken nach einer Strategieanwendung"""
        self.application_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            
        # Aktualisiere durchschnittliche Verbesserung
        self.average_improvement = (
            (self.average_improvement * (self.application_count - 1) + improvement) / 
            self.application_count
        )
        
        # Aktualisiere durchschnittliche Ausführungszeit
        self.average_execution_time = (
            (self.average_execution_time * (self.application_count - 1) + execution_time) / 
            self.application_count
        )
        
        self.last_applied = time.time()


class AdaptiveOptimizer:
    """
    Adaptive Optimierung für Q-LOGIK
    
    Implementiert selbstlernende Optimierungsstrategien für Q-LOGIK.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den AdaptiveOptimizer
        
        Args:
            config: Konfigurationsobjekt für den AdaptiveOptimizer
        """
        self.config = config or {}
        self.rule_optimizer = RuleOptimizer(self.config.get("rule_optimizer_config"))
        
        # Optimierungsstrategien
        self.strategies = {
            "performance": {
                "description": "Optimiert Regeln basierend auf Leistung",
                "weight": 0.8
            },
            "confidence": {
                "description": "Optimiert Regeln basierend auf Konfidenz",
                "weight": 0.7
            },
            "balanced": {
                "description": "Ausgewogene Optimierung von Leistung und Konfidenz",
                "weight": 0.9
            }
        }
        
        # Metriken für Optimierungsstrategien
        self.strategy_metrics = {
            strategy_id: OptimizationPerformanceMetrics(strategy_id=strategy_id)
            for strategy_id in self.strategies
        }
        
        # Optimierungshistorie
        self.optimization_history = []
        
        # Hintergrund-Optimierung
        self.background_optimization_enabled = self.config.get("background_optimization", False)
        self.background_optimization_interval = self.config.get("background_optimization_interval", 3600)  # 1 Stunde
        self.background_thread = None
        self.stop_background_thread = threading.Event()
        
        if self.background_optimization_enabled:
            self._start_background_optimization()
        
        logger.info("AdaptiveOptimizer initialisiert")
    
    def optimize(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Führt eine adaptive Optimierung durch
        
        Args:
            context: Kontextinformationen für die Optimierung (optional)
            
        Returns:
            Optimierungsergebnisse
        """
        start_time = time.time()
        
        # Wähle die beste Optimierungsstrategie basierend auf Metriken und Kontext
        strategy = self._select_best_strategy(context)
        
        # Speichere Metriken vor der Optimierung
        metrics_before = self._get_system_metrics()
        
        # Führe Optimierung durch
        optimization_result = self.rule_optimizer.optimize_rules(strategy)
        
        # Speichere Metriken nach der Optimierung
        metrics_after = self._get_system_metrics()
        
        # Berechne Verbesserung
        improvement = self._calculate_improvement(metrics_before, metrics_after)
        
        # Aktualisiere Strategiemetriken
        execution_time = time.time() - start_time
        success = optimization_result.get("optimized_rules", 0) > 0
        self.strategy_metrics[strategy].update_metrics(success, improvement, execution_time)
        
        # Füge Optimierung zur Historie hinzu
        self.optimization_history.append({
            "timestamp": time.time(),
            "strategy": strategy,
            "context": context,
            "result": optimization_result,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "improvement": improvement,
            "execution_time": execution_time
        })
        
        # Begrenze die Größe der Historie
        max_history_size = self.config.get("max_history_size", 100)
        if len(self.optimization_history) > max_history_size:
            self.optimization_history = self.optimization_history[-max_history_size:]
        
        result = {
            "strategy": strategy,
            "optimization_result": optimization_result,
            "improvement": improvement,
            "execution_time": execution_time
        }
        
        logger.info(f"Adaptive Optimierung mit Strategie {strategy} abgeschlossen: {improvement:.2f} Verbesserung")
        return result
    
    def get_strategy_metrics(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Gibt die Metriken einer Optimierungsstrategie zurück
        
        Args:
            strategy_id: ID der Optimierungsstrategie
            
        Returns:
            Dictionary mit Metriken oder None, falls nicht gefunden
        """
        metrics = self.strategy_metrics.get(strategy_id)
        if not metrics:
            return None
            
        return {
            "strategy_id": metrics.strategy_id,
            "success_rate": metrics.success_rate,
            "application_count": metrics.application_count,
            "average_improvement": metrics.average_improvement,
            "average_execution_time": metrics.average_execution_time,
            "utility_score": metrics.utility_score,
            "last_applied": metrics.last_applied,
            "created_at": metrics.created_at
        }
    
    def get_all_strategy_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Gibt die Metriken aller Optimierungsstrategien zurück
        
        Returns:
            Dictionary mit allen Strategiemetriken
        """
        return {
            strategy_id: self.get_strategy_metrics(strategy_id)
            for strategy_id in self.strategies
        }
    
    def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Gibt die Optimierungshistorie zurück
        
        Args:
            limit: Maximale Anzahl der zurückzugebenden Einträge
            
        Returns:
            Liste mit Optimierungshistorieneinträgen
        """
        return self.optimization_history[-limit:]
    
    def enable_background_optimization(self, interval: int = 3600) -> bool:
        """
        Aktiviert die Hintergrund-Optimierung
        
        Args:
            interval: Intervall in Sekunden zwischen Optimierungen
            
        Returns:
            True, wenn erfolgreich aktiviert, sonst False
        """
        if self.background_thread and self.background_thread.is_alive():
            logger.warning("Hintergrund-Optimierung bereits aktiv")
            return False
        
        self.background_optimization_enabled = True
        self.background_optimization_interval = interval
        self._start_background_optimization()
        
        logger.info(f"Hintergrund-Optimierung aktiviert mit Intervall {interval}s")
        return True
    
    def disable_background_optimization(self) -> bool:
        """
        Deaktiviert die Hintergrund-Optimierung
        
        Returns:
            True, wenn erfolgreich deaktiviert, sonst False
        """
        if not self.background_thread or not self.background_thread.is_alive():
            logger.warning("Hintergrund-Optimierung nicht aktiv")
            return False
        
        self.background_optimization_enabled = False
        self.stop_background_thread.set()
        self.background_thread.join(timeout=5.0)
        
        logger.info("Hintergrund-Optimierung deaktiviert")
        return True
    
    def _start_background_optimization(self) -> None:
        """Startet die Hintergrund-Optimierung"""
        if self.background_thread and self.background_thread.is_alive():
            return
            
        self.stop_background_thread.clear()
        self.background_thread = threading.Thread(
            target=self._background_optimization_loop,
            daemon=True
        )
        self.background_thread.start()
        
        logger.info("Hintergrund-Optimierung gestartet")
    
    def _background_optimization_loop(self) -> None:
        """Hintergrund-Optimierungsschleife"""
        while not self.stop_background_thread.is_set():
            try:
                # Warte für das konfigurierte Intervall
                if self.stop_background_thread.wait(timeout=self.background_optimization_interval):
                    break
                
                # Führe Optimierung durch
                logger.info("Starte Hintergrund-Optimierung")
                self.optimize({"source": "background"})
                
            except Exception as e:
                logger.error(f"Fehler in Hintergrund-Optimierung: {e}")
                # Warte kurz, um CPU-Überlastung bei wiederholten Fehlern zu vermeiden
                time.sleep(10)
    
    def _select_best_strategy(self, context: Dict[str, Any] = None) -> str:
        """
        Wählt die beste Optimierungsstrategie basierend auf Metriken und Kontext
        
        Args:
            context: Kontextinformationen für die Optimierung
            
        Returns:
            ID der besten Optimierungsstrategie
        """
        context = context or {}
        
        # Wenn eine Strategie im Kontext angegeben ist, verwende diese
        if "strategy" in context:
            strategy = context["strategy"]
            if strategy in self.strategies:
                return strategy
        
        # Wenn keine Metriken vorhanden sind, verwende eine zufällige Strategie
        if all(metrics.application_count == 0 for metrics in self.strategy_metrics.values()):
            strategies = list(self.strategies.keys())
            return np.random.choice(strategies)
        
        # Berechne Nutzwerte für alle Strategien
        utility_scores = {}
        for strategy_id, metrics in self.strategy_metrics.items():
            # Basisnutzwert aus Metriken
            base_score = metrics.utility_score
            
            # Gewichtung aus Strategiekonfiguration
            weight = self.strategies[strategy_id].get("weight", 0.5)
            
            # Exploration-Faktor (bevorzuge weniger häufig verwendete Strategien)
            exploration_factor = 1.0 / (1.0 + metrics.application_count * 0.1)
            
            # Zeitfaktor (bevorzuge Strategien, die länger nicht verwendet wurden)
            time_factor = 1.0
            if metrics.last_applied > 0:
                time_since_last = time.time() - metrics.last_applied
                time_factor = min(2.0, 1.0 + time_since_last / (24 * 3600))  # Max. Bonus nach 24h
            
            # Kombiniere Faktoren
            utility_scores[strategy_id] = base_score * weight * exploration_factor * time_factor
        
        # Wähle Strategie mit höchstem Nutzwert
        best_strategy = max(utility_scores.items(), key=lambda x: x[1])[0]
        
        logger.debug(f"Beste Optimierungsstrategie: {best_strategy} (Nutzwert: {utility_scores[best_strategy]:.2f})")
        return best_strategy
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """
        Sammelt Systemmetriken für die Optimierungsbewertung
        
        Returns:
            Dictionary mit Systemmetriken
        """
        # Sammle Regelmetriken
        rule_metrics = self.rule_optimizer.get_all_rule_metrics()
        
        # Berechne aggregierte Metriken
        avg_success_rate = 0.0
        avg_confidence = 0.0
        avg_utility = 0.0
        total_rules = len(rule_metrics)
        
        if total_rules > 0:
            success_rates = [m.success_rate for m in rule_metrics.values()]
            confidences = [m.average_confidence for m in rule_metrics.values()]
            utilities = [m.utility_score for m in rule_metrics.values()]
            
            avg_success_rate = sum(success_rates) / total_rules
            avg_confidence = sum(confidences) / total_rules
            avg_utility = sum(utilities) / total_rules
        
        return {
            "timestamp": time.time(),
            "total_rules": total_rules,
            "avg_success_rate": avg_success_rate,
            "avg_confidence": avg_confidence,
            "avg_utility": avg_utility
        }
    
    def _calculate_improvement(self, 
                              metrics_before: Dict[str, Any], 
                              metrics_after: Dict[str, Any]) -> float:
        """
        Berechnet die Verbesserung zwischen zwei Metrikzuständen
        
        Args:
            metrics_before: Metriken vor der Optimierung
            metrics_after: Metriken nach der Optimierung
            
        Returns:
            Verbesserungswert (0.0-1.0)
        """
        # Wenn keine Regeln vorhanden sind, keine Verbesserung
        if metrics_before.get("total_rules", 0) == 0:
            return 0.0
        
        # Berechne Verbesserungen für verschiedene Metriken
        success_improvement = metrics_after.get("avg_success_rate", 0.0) - metrics_before.get("avg_success_rate", 0.0)
        confidence_improvement = metrics_after.get("avg_confidence", 0.0) - metrics_before.get("avg_confidence", 0.0)
        utility_improvement = metrics_after.get("avg_utility", 0.0) - metrics_before.get("avg_utility", 0.0)
        
        # Gewichte die Verbesserungen
        weighted_improvement = (
            success_improvement * 0.4 +
            confidence_improvement * 0.3 +
            utility_improvement * 0.3
        )
        
        # Normalisiere auf 0.0-1.0
        normalized_improvement = max(0.0, min(1.0, weighted_improvement + 0.5))
        
        return normalized_improvement


# Globale Instanz für einfachen Zugriff
adaptive_optimizer = AdaptiveOptimizer()

def optimize(context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Führt eine adaptive Optimierung durch
    
    Args:
        context: Kontextinformationen für die Optimierung (optional)
        
    Returns:
        Optimierungsergebnisse
    """
    return adaptive_optimizer.optimize(context)

def get_strategy_metrics(strategy_id: str) -> Optional[Dict[str, Any]]:
    """
    Gibt die Metriken einer Optimierungsstrategie zurück
    
    Args:
        strategy_id: ID der Optimierungsstrategie
        
    Returns:
        Dictionary mit Metriken oder None, falls nicht gefunden
    """
    return adaptive_optimizer.get_strategy_metrics(strategy_id)

def enable_background_optimization(interval: int = 3600) -> bool:
    """
    Aktiviert die Hintergrund-Optimierung
    
    Args:
        interval: Intervall in Sekunden zwischen Optimierungen
        
    Returns:
        True, wenn erfolgreich aktiviert, sonst False
    """
    return adaptive_optimizer.enable_background_optimization(interval)

def disable_background_optimization() -> bool:
    """
    Deaktiviert die Hintergrund-Optimierung
    
    Returns:
        True, wenn erfolgreich deaktiviert, sonst False
    """
    return adaptive_optimizer.disable_background_optimization()


if __name__ == "__main__":
    # Beispiel für die Verwendung des AdaptiveOptimizers
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
        register_rule(rule_id, rule)
    
    # Simuliere Regelanwendungen
    # Hier würden normalerweise die Regeln angewendet und Feedback gesammelt werden
    
    # Führe adaptive Optimierung durch
    result = optimize()
    print(f"Optimierungsergebnis: {result}")
    
    # Zeige Strategiemetriken
    for strategy_id in ["performance", "confidence", "balanced"]:
        metrics = get_strategy_metrics(strategy_id)
        print(f"Strategie {strategy_id}:")
        print(f"  - Erfolgsrate: {metrics['success_rate']:.2f}")
        print(f"  - Durchschnittliche Verbesserung: {metrics['average_improvement']:.2f}")
        print(f"  - Nutzwert: {metrics['utility_score']:.2f}")
    
    # Aktiviere Hintergrund-Optimierung (nur für Beispielzwecke)
    # In einer realen Anwendung würde dies normalerweise länger laufen
    enable_background_optimization(interval=10)  # Alle 10 Sekunden
    print("Hintergrund-Optimierung aktiviert. Drücke Ctrl+C zum Beenden.")
    
    try:
        # Warte kurz, um einige Hintergrund-Optimierungen zu sehen
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        # Deaktiviere Hintergrund-Optimierung
        disable_background_optimization()
        print("Hintergrund-Optimierung deaktiviert.")
