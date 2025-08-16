#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Erweiterte Paradoxauflösung

Modul für fortgeschrittene Paradoxerkennungs- und -auflösungsmechanismen.
Bietet hierarchische Klassifizierung und automatische Strategieauswahl für temporale Paradoxien.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
import time
import math
import enum
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set

# Konfiguriere Logging
logger = logging.getLogger("MISO.Simulation.ParadoxResolution")

class ParadoxSeverity(enum.Enum):
    """Schweregrade für Paradoxien"""
    CRITICAL = "critical"      # Kritisches Paradox, das sofortige Auflösung erfordert
    SIGNIFICANT = "significant"  # Signifikantes Paradox mit mittlerer Auswirkung
    MARGINAL = "marginal"      # Marginales Paradox mit geringer Auswirkung

class ParadoxCategory(enum.Enum):
    """Primäre Paradoxkategorien"""
    TEMPORAL = "temporal"           # Zeitbasierte Paradoxien (Zeitschleifen, kausale Widersprüche)
    QUANTUM = "quantum"             # Quantenmechanische Paradoxien (Superposition, Verschränkung)
    INFORMATIONAL = "informational" # Informationstheoretische Paradoxien (Entropie, Information)

class ParadoxClassification(enum.Enum):
    """Sekundäre Paradoxklassifikation"""
    STRUCTURAL = "structural"     # In der Topologie der Zeitlinie verankert
    PROBABILISTIC = "probabilistic" # In Wahrscheinlichkeitsverteilungen erkennbar
    SEMANTIC = "semantic"         # In der inhaltlichen Bedeutung und Beziehung von Knoten

class ResolutionStrategy:
    """Klasse zur Repräsentation einer Auflösungsstrategie für Paradoxien"""
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 applicability_function: Callable[[Dict[str, Any]], float],
                 resolution_function: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 effectiveness: Dict[ParadoxCategory, float] = None,
                 side_effects: Dict[str, float] = None,
                 computational_cost: float = 0.5):
        """
        Initialisiert eine Auflösungsstrategie
        
        Args:
            name: Name der Strategie
            description: Beschreibung der Strategie
            applicability_function: Funktion zur Berechnung der Anwendbarkeit (0.0-1.0)
            resolution_function: Funktion zur Auflösung des Paradoxes
            effectiveness: Effektivität für verschiedene Paradoxkategorien (0.0-1.0)
            side_effects: Potenzielle Nebenwirkungen der Strategie
            computational_cost: Berechnungskosten der Strategie (0.0-1.0)
        """
        self.name = name
        self.description = description
        self.applicability_function = applicability_function
        self.resolution_function = resolution_function
        self.effectiveness = effectiveness or {cat: 0.5 for cat in ParadoxCategory}
        self.side_effects = side_effects or {}
        self.computational_cost = computational_cost
        
    def calculate_applicability(self, paradox: Dict[str, Any]) -> float:
        """
        Berechnet die Anwendbarkeit der Strategie auf ein spezifisches Paradox
        
        Args:
            paradox: Paradox-Informationen
            
        Returns:
            Anwendbarkeitsgrad (0.0-1.0)
        """
        try:
            return self.applicability_function(paradox)
        except Exception as e:
            logger.error(f"Fehler bei Berechnung der Anwendbarkeit für {self.name}: {e}")
            return 0.0
            
    def apply_resolution(self, paradox: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wendet die Auflösungsstrategie auf ein Paradox an
        
        Args:
            paradox: Paradox-Informationen
            
        Returns:
            Ergebnis der Auflösung
        """
        if not self.resolution_function:
            return {
                "success": False,
                "reason": "Keine Auflösungsfunktion implementiert",
                "paradox": paradox
            }
            
        try:
            return self.resolution_function(paradox)
        except Exception as e:
            logger.error(f"Fehler bei Anwendung der Auflösungsstrategie {self.name}: {e}")
            return {
                "success": False,
                "reason": f"Fehler: {str(e)}",
                "paradox": paradox
            }
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert die Strategie in ein Dictionary
        
        Returns:
            Dictionary-Repräsentation der Strategie
        """
        return {
            "name": self.name,
            "description": self.description,
            "effectiveness": {cat.value: eff for cat, eff in self.effectiveness.items()},
            "computational_cost": self.computational_cost,
            "side_effects": self.side_effects
        }

class ParadoxResolutionManager:
    """Manager für Paradoxauflösungsstrategien"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den Paradoxauflösungs-Manager
        
        Args:
            config: Konfigurationsparameter
        """
        self.config = config or {}
        self.strategies = {}
        self.resolution_history = []
        self.prevention_threshold = self.config.get("prevention_threshold", 0.7)
        
        # Initialisiere Standardstrategien
        self._initialize_strategies()
        
        logger.info(f"ParadoxResolutionManager initialisiert mit {len(self.strategies)} Strategien")
        
    def _initialize_strategies(self):
        """Initialisiert die verfügbaren Auflösungsstrategien"""
        # 1. Zeitschleifenauflösung durch Knotenduplikation
        self.register_strategy(ResolutionStrategy(
            name="temporal_loop_duplication",
            description="Löst Zeitschleifen durch Knotenduplikation mit verschobenen Referenzen",
            applicability_function=lambda p: 0.9 if p.get("type") in ["direct_time_loop", "cyclic_time_loop"] else 0.1,
            effectiveness={
                ParadoxCategory.TEMPORAL: 0.85,
                ParadoxCategory.QUANTUM: 0.2,
                ParadoxCategory.INFORMATIONAL: 0.3
            },
            computational_cost=0.4
        ))
        
        # 2. Kausale Paradoxauflösung durch Wahrscheinlichkeitsmodulation
        self.register_strategy(ResolutionStrategy(
            name="causal_probability_modulation",
            description="Löst kausale Paradoxa durch Anpassung der Knotenwahrscheinlichkeiten",
            applicability_function=lambda p: 0.85 if p.get("type") == "causal_paradox" else 0.2,
            effectiveness={
                ParadoxCategory.TEMPORAL: 0.7,
                ParadoxCategory.QUANTUM: 0.5,
                ParadoxCategory.INFORMATIONAL: 0.6
            },
            computational_cost=0.6,
            side_effects={"timeline_stability": -0.2}
        ))
        
        # 3. Quantensuperposition für inkonsistente Zustände
        self.register_strategy(ResolutionStrategy(
            name="quantum_superposition",
            description="Verwendet Quantensuperposition, um widersprüchliche Zustände zu vereinbaren",
            applicability_function=lambda p: 0.95 if p.get("type") == "quantum_paradox" else 0.4,
            effectiveness={
                ParadoxCategory.TEMPORAL: 0.5,
                ParadoxCategory.QUANTUM: 0.9,
                ParadoxCategory.INFORMATIONAL: 0.4
            },
            computational_cost=0.8,
            side_effects={"timeline_coherence": -0.3}
        ))
        
        # 4. Entropieinjektion für Entropie-Inversionsparadoxa
        self.register_strategy(ResolutionStrategy(
            name="entropy_injection",
            description="Injiziert Entropie in Knoten mit unnatürlicher Entropieabnahme",
            applicability_function=lambda p: 0.9 if p.get("type") == "entropy_inversion" else 0.1,
            effectiveness={
                ParadoxCategory.TEMPORAL: 0.3,
                ParadoxCategory.QUANTUM: 0.4,
                ParadoxCategory.INFORMATIONAL: 0.85
            },
            computational_cost=0.5
        ))
        
        # 5. Zeitlinienisolierung für kritische Paradoxa
        self.register_strategy(ResolutionStrategy(
            name="timeline_isolation",
            description="Isoliert paradoxe Bereiche in einer separaten Zeitlinie",
            applicability_function=lambda p: 0.7 if p.get("severity", 0) > 0.7 else 0.3,
            effectiveness={
                ParadoxCategory.TEMPORAL: 0.8,
                ParadoxCategory.QUANTUM: 0.7,
                ParadoxCategory.INFORMATIONAL: 0.7
            },
            computational_cost=0.9,
            side_effects={"timeline_fragmentation": 0.6}
        ))
        
        # 6. Probabilistische Diffusion
        self.register_strategy(ResolutionStrategy(
            name="probabilistic_diffusion",
            description="Verteilt Paradoxeffekte auf benachbarte Knoten zur Abschwächung",
            applicability_function=lambda p: 0.6 if p.get("type") in ["probability_paradox"] else 0.4,
            effectiveness={
                ParadoxCategory.TEMPORAL: 0.5,
                ParadoxCategory.QUANTUM: 0.6,
                ParadoxCategory.INFORMATIONAL: 0.7
            },
            computational_cost=0.7,
            side_effects={"timeline_stability": -0.1, "node_precision": -0.2}
        ))
        
    def register_strategy(self, strategy: ResolutionStrategy):
        """
        Registriert eine neue Auflösungsstrategie
        
        Args:
            strategy: Zu registrierende Strategie
        """
        self.strategies[strategy.name] = strategy
        logger.debug(f"Strategie registriert: {strategy.name}")
        
    def find_best_strategy(self, paradox: Dict[str, Any]) -> Tuple[str, float]:
        """
        Findet die beste Auflösungsstrategie für ein Paradox
        
        Args:
            paradox: Paradox-Informationen
            
        Returns:
            Tuple aus Strategiename und Bewertungswert
        """
        # Kategorisiere das Paradox
        category = self._categorize_paradox(paradox)
        
        # Berechne Scores für alle Strategien
        strategy_scores = {}
        
        for name, strategy in self.strategies.items():
            # Berechne Anwendbarkeitsgrad
            applicability = strategy.calculate_applicability(paradox)
            
            # Berechne Effektivität basierend auf Paradoxkategorie
            effectiveness = strategy.effectiveness.get(category, 0.3)
            
            # Berechne Kosten-Nutzen-Verhältnis
            cost_benefit = effectiveness / max(0.1, strategy.computational_cost)
            
            # Berechne Gesamtscore
            score = applicability * cost_benefit
            
            # Berücksichtige Nebenwirkungen basierend auf Paradoxschweregrad
            side_effect_penalty = 0.0
            if "severity" in paradox:
                severity = paradox["severity"]
                for effect, magnitude in strategy.side_effects.items():
                    if magnitude < 0:  # Negative Nebenwirkungen
                        side_effect_penalty += abs(magnitude) * severity
            
            # Reduziere Score basierend auf Nebenwirkungen
            final_score = score * (1.0 - min(0.5, side_effect_penalty))
            
            strategy_scores[name] = final_score
            
        # Finde Strategie mit höchstem Score
        if not strategy_scores:
            return ("none", 0.0)
            
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        return best_strategy
        
    def _categorize_paradox(self, paradox: Dict[str, Any]) -> ParadoxCategory:
        """
        Kategorisiert ein Paradox
        
        Args:
            paradox: Paradox-Informationen
            
        Returns:
            Paradoxkategorie
        """
        paradox_type = paradox.get("type", "")
        
        # Temporale Paradoxa
        if paradox_type in ["direct_time_loop", "cyclic_time_loop", "causal_paradox"]:
            return ParadoxCategory.TEMPORAL
            
        # Quantenmechanische Paradoxa
        elif paradox_type in ["quantum_paradox", "decoherence_paradox", "entanglement_paradox"]:
            return ParadoxCategory.QUANTUM
            
        # Informationstheoretische Paradoxa
        elif paradox_type in ["entropy_inversion", "information_loss", "probability_paradox"]:
            return ParadoxCategory.INFORMATIONAL
            
        # Standardfall
        return ParadoxCategory.TEMPORAL
        
    def resolve_paradox(self, paradox: Dict[str, Any]) -> Dict[str, Any]:
        """
        Löst ein Paradox mit der besten verfügbaren Strategie
        
        Args:
            paradox: Paradox-Informationen
            
        Returns:
            Auflösungsergebnis
        """
        # Finde beste Strategie
        best_strategy_name, score = self.find_best_strategy(paradox)
        
        if best_strategy_name == "none" or score < 0.2:
            logger.warning(f"Keine geeignete Auflösungsstrategie für Paradox gefunden: {paradox.get('type')}")
            return {
                "success": False,
                "reason": "Keine geeignete Strategie verfügbar",
                "paradox": paradox,
                "strategy": None,
                "score": score
            }
            
        # Wende Strategie an
        strategy = self.strategies[best_strategy_name]
        resolution_result = strategy.apply_resolution(paradox)
        
        # Speichere in Historie
        self.resolution_history.append({
            "timestamp": time.time(),
            "paradox_type": paradox.get("type"),
            "strategy": best_strategy_name,
            "score": score,
            "success": resolution_result.get("success", False)
        })
        
        # Füge Strategie-Informationen zum Ergebnis hinzu
        resolution_result["strategy"] = best_strategy_name
        resolution_result["strategy_info"] = strategy.to_dict()
        resolution_result["score"] = score
        
        return resolution_result
        
    def resolve_multiple_paradoxes(self, paradoxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Löst mehrere Paradoxa mit optimaler Strategieauswahl
        
        Args:
            paradoxes: Liste von Paradox-Informationen
            
        Returns:
            Auflösungsergebnisse
        """
        if not paradoxes:
            return {"success": True, "resolved_count": 0, "results": []}
            
        # Sortiere Paradoxa nach Schweregrad (absteigend)
        sorted_paradoxes = sorted(paradoxes, key=lambda p: p.get("severity", 0.0), reverse=True)
        
        results = []
        success_count = 0
        failed_count = 0
        critical_failures = 0
        
        for paradox in sorted_paradoxes:
            # Löse Paradox
            result = self.resolve_paradox(paradox)
            results.append(result)
            
            # Aktualisiere Zähler
            if result.get("success", False):
                success_count += 1
            else:
                failed_count += 1
                if paradox.get("severity", 0.0) >= 0.7:
                    critical_failures += 1
                    
        # Bewerte Gesamterfolg
        overall_success = failed_count == 0 or (critical_failures == 0 and failed_count / len(paradoxes) < 0.3)
        
        return {
            "success": overall_success,
            "resolved_count": success_count,
            "failed_count": failed_count,
            "critical_failures": critical_failures,
            "results": results
        }
        
    def get_prevention_recommendations(self, timeline_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generiert Empfehlungen zur Paradoxprävention basierend auf Zeitliniendaten
        
        Args:
            timeline_data: Daten der Zeitlinie
            
        Returns:
            Liste von Präventionsempfehlungen
        """
        recommendations = []
        
        # Prüfe auf Strukturmerkmale, die zu Paradoxa führen könnten
        nodes = timeline_data.get("nodes", {})
        connections = timeline_data.get("connections", [])
        
        # 1. Prüfe auf potenzielle Zeitschleifen
        potential_loops = self._detect_potential_loops(nodes, connections)
        if potential_loops:
            recommendations.append({
                "type": "potential_time_loop",
                "risk_level": min(1.0, 0.3 + (0.1 * len(potential_loops))),
                "affected_nodes": potential_loops,
                "description": "Potenzielle Zeitschleife durch zirkuläre Knotenreferenzen",
                "prevention_action": "Füge zeitliche Pufferknoten ein oder verwende Quantensuperposition"
            })
            
        # 2. Prüfe auf Knoten mit niedriger Wahrscheinlichkeit
        low_prob_nodes = []
        for node_id, node in nodes.items():
            if "probability" in node and node["probability"] < 0.2:
                low_prob_nodes.append(node_id)
                
        if low_prob_nodes and len(low_prob_nodes) > 3:
            recommendations.append({
                "type": "low_probability_cluster",
                "risk_level": min(1.0, 0.4 + (0.05 * len(low_prob_nodes))),
                "affected_nodes": low_prob_nodes,
                "description": "Mehrere Knoten mit niedriger Wahrscheinlichkeit können zu Wahrscheinlichkeitsparadoxa führen",
                "prevention_action": "Verstärke Knotenwahrscheinlichkeiten oder verwende probabilistische Diffusion"
            })
            
        # 3. Prüfe auf kausale Risiken (Knoten mit umgekehrter Zeitrichtung)
        causal_risks = self._detect_causal_risks(nodes, connections)
        if causal_risks:
            recommendations.append({
                "type": "causal_risk",
                "risk_level": min(1.0, 0.6 + (0.1 * len(causal_risks))),
                "affected_connections": causal_risks,
                "description": "Kausale Risiken durch zeitliche Inkonsistenzen",
                "prevention_action": "Normalisiere temporale Beziehungen oder isoliere problematische Pfade"
            })
            
        return recommendations
        
    def _detect_potential_loops(self, nodes: Dict[str, Any], connections: List[Dict[str, Any]]) -> List[str]:
        """
        Erkennt potenzielle Zeitschleifen in einer Zeitlinie
        
        Args:
            nodes: Knoten der Zeitlinie
            connections: Verbindungen zwischen Knoten
            
        Returns:
            Liste von Knoten-IDs, die Teil potenzieller Schleifen sind
        """
        # Erstelle Adjazenzliste
        adjacency = {}
        for node_id in nodes:
            adjacency[node_id] = []
            
        for conn in connections:
            source = conn.get("source_id")
            target = conn.get("target_id")
            if source and target and source in adjacency:
                adjacency[source].append(target)
                
        # Finde potenzielle Schleifen (Knoten, die in einem Pfad der Länge <= 3 zu sich selbst führen)
        potential_loop_nodes = set()
        
        for node_id in nodes:
            # DFS mit maximaler Tiefe 3
            visited = set()
            path = []
            
            def dfs(current, depth=0):
                if depth > 3:
                    return
                    
                if current in path:
                    # Schleife gefunden
                    loop_index = path.index(current)
                    potential_loop_nodes.update(path[loop_index:])
                    return
                    
                path.append(current)
                visited.add(current)
                
                for neighbor in adjacency.get(current, []):
                    if depth < 3:  # Begrenze Tiefe
                        dfs(neighbor, depth + 1)
                        
                path.pop()
                
            dfs(node_id)
            
        return list(potential_loop_nodes)
        
    def _detect_causal_risks(self, nodes: Dict[str, Any], connections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Erkennt kausale Risiken in einer Zeitlinie
        
        Args:
            nodes: Knoten der Zeitlinie
            connections: Verbindungen zwischen Knoten
            
        Returns:
            Liste von risikobehafteten Verbindungen
        """
        risky_connections = []
        
        # Extrahiere Zeitstempel für alle Knoten
        timestamps = {}
        for node_id, node in nodes.items():
            if "timestamp" in node:
                timestamps[node_id] = node["timestamp"]
                
        # Prüfe auf umgekehrte Zeitrichtung in Verbindungen
        for conn in connections:
            source_id = conn.get("source_id")
            target_id = conn.get("target_id")
            
            if source_id in timestamps and target_id in timestamps:
                source_time = timestamps[source_id]
                target_time = timestamps[target_id]
                
                # Wenn Zielknoten älter als Quellknoten ist, liegt ein potenzielles kausales Risiko vor
                if target_time < source_time:
                    time_delta = source_time - target_time
                    # Berechne Risiko basierend auf Zeitunterschied
                    risk = min(1.0, time_delta / (24 * 3600))  # Skaliere auf max. 1 Tag
                    
                    risky_connections.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "time_delta": time_delta,
                        "risk_level": risk
                    })
                    
        return risky_connections

# Initialisiere globalen Manager für einfachen Zugriff
_PARADOX_RESOLUTION_MANAGER = None

def get_paradox_resolution_manager(config: Dict[str, Any] = None) -> ParadoxResolutionManager:
    """
    Gibt den globalen ParadoxResolutionManager zurück oder erstellt ihn bei Bedarf
    
    Args:
        config: Optionale Konfiguration für den Manager
        
    Returns:
        ParadoxResolutionManager-Instanz
    """
    global _PARADOX_RESOLUTION_MANAGER
    if _PARADOX_RESOLUTION_MANAGER is None:
        _PARADOX_RESOLUTION_MANAGER = ParadoxResolutionManager(config)
    return _PARADOX_RESOLUTION_MANAGER

# Exportiere Hauptklassen und Hilfsfunktionen
__all__ = [
    'ParadoxSeverity',
    'ParadoxCategory',
    'ParadoxClassification',
    'ResolutionStrategy',
    'ParadoxResolutionManager',
    'get_paradox_resolution_manager'
]
