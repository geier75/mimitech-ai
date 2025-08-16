"""
DEEP-STATE-MODUL – Strategie-Matrix

Diese Komponente implementiert die Funktionalität zur Verwaltung und Optimierung von Strategien
basierend auf verschiedenen Faktoren wie Risiko, Rendite, Zeithorizont und Ressourcen.
"""

from typing import List, Dict, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass
from enum import Enum, auto
from datetime import datetime
import logging
import json
import random

# Import aus dem Deep-State-Modul
from miso.strategic.deep_state import EscalationLevel
from miso.strategic.strategy_recommender import StrategyRecommendation

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='strategy_matrix.log'
)
logger = logging.getLogger('StrategyMatrix')


class StrategyDimension(Enum):
    """Dimensionen für die Strategiebewertung."""
    RISK = "risk"
    RETURN = "return"
    TIME_HORIZON = "time_horizon"
    RESOURCES = "resources"
    COMPLEXITY = "complexity"
    ADAPTABILITY = "adaptability"
    RESILIENCE = "resilience"


@dataclass
class StrategyProfile:
    """Datenklasse für Strategieprofile."""
    id: str
    name: str
    description: str
    dimensions: Dict[StrategyDimension, float]  # 0.0 bis 1.0
    applicability: Dict[str, float]  # Anwendbarkeit für verschiedene Szenarien
    success_rate: float
    last_updated: datetime
    version: str
    tags: List[str]
    active: bool = True


class StrategyMatrix:
    """
    Verwaltung und Optimierung von Strategien.
    """
    
    def __init__(self):
        # Strategieprofile
        self.strategy_profiles: Dict[str, StrategyProfile] = {}
        
        # Initialisiere Standardprofile
        self._initialize_default_profiles()
        
        # Strategiebewertungen
        self.strategy_evaluations: Dict[str, List[Dict[str, Any]]] = {}
        
        # Aktuelle Strategiekombination
        self.current_strategy_combination: Dict[str, float] = {}
        
        # Optimierungsparameter
        self.optimization_parameters = {
            "risk_tolerance": 0.5,
            "return_expectation": 0.7,
            "time_preference": 0.5,
            "resource_constraint": 0.6
        }
        
        logger.info("StrategyMatrix initialisiert")
    
    def _initialize_default_profiles(self):
        """
        Initialisiert Standardstrategieprofile.
        """
        # Defensive Strategie
        defensive = StrategyProfile(
            id="strategy_defensive",
            name="Defensive Strategie",
            description="Konservative Strategie mit Fokus auf Risikominimierung und Kapitalerhalt.",
            dimensions={
                StrategyDimension.RISK: 0.2,
                StrategyDimension.RETURN: 0.4,
                StrategyDimension.TIME_HORIZON: 0.8,
                StrategyDimension.RESOURCES: 0.3,
                StrategyDimension.COMPLEXITY: 0.3,
                StrategyDimension.ADAPTABILITY: 0.5,
                StrategyDimension.RESILIENCE: 0.9
            },
            applicability={
                "market_volatility": 0.8,
                "geopolitical_tension": 0.7,
                "economic_downturn": 0.9,
                "technological_disruption": 0.4
            },
            success_rate=0.85,
            last_updated=datetime.now(),
            version="1.0",
            tags=["defensive", "low_risk", "capital_preservation"]
        )
        
        # Offensive Strategie
        offensive = StrategyProfile(
            id="strategy_offensive",
            name="Offensive Strategie",
            description="Aggressive Strategie mit Fokus auf Wachstum und hohe Renditen.",
            dimensions={
                StrategyDimension.RISK: 0.8,
                StrategyDimension.RETURN: 0.9,
                StrategyDimension.TIME_HORIZON: 0.4,
                StrategyDimension.RESOURCES: 0.7,
                StrategyDimension.COMPLEXITY: 0.7,
                StrategyDimension.ADAPTABILITY: 0.6,
                StrategyDimension.RESILIENCE: 0.3
            },
            applicability={
                "market_opportunity": 0.9,
                "technological_innovation": 0.8,
                "economic_growth": 0.9,
                "competitive_advantage": 0.8
            },
            success_rate=0.6,
            last_updated=datetime.now(),
            version="1.0",
            tags=["offensive", "high_risk", "growth"]
        )
        
        # Ausgewogene Strategie
        balanced = StrategyProfile(
            id="strategy_balanced",
            name="Ausgewogene Strategie",
            description="Ausgewogene Strategie mit moderatem Risiko-Rendite-Profil.",
            dimensions={
                StrategyDimension.RISK: 0.5,
                StrategyDimension.RETURN: 0.6,
                StrategyDimension.TIME_HORIZON: 0.6,
                StrategyDimension.RESOURCES: 0.5,
                StrategyDimension.COMPLEXITY: 0.5,
                StrategyDimension.ADAPTABILITY: 0.7,
                StrategyDimension.RESILIENCE: 0.6
            },
            applicability={
                "market_stability": 0.8,
                "moderate_growth": 0.9,
                "technological_evolution": 0.7,
                "geopolitical_stability": 0.7
            },
            success_rate=0.75,
            last_updated=datetime.now(),
            version="1.0",
            tags=["balanced", "moderate_risk", "sustainable"]
        )
        
        # Opportunistische Strategie
        opportunistic = StrategyProfile(
            id="strategy_opportunistic",
            name="Opportunistische Strategie",
            description="Flexible Strategie zur Nutzung kurzfristiger Chancen.",
            dimensions={
                StrategyDimension.RISK: 0.7,
                StrategyDimension.RETURN: 0.8,
                StrategyDimension.TIME_HORIZON: 0.3,
                StrategyDimension.RESOURCES: 0.6,
                StrategyDimension.COMPLEXITY: 0.8,
                StrategyDimension.ADAPTABILITY: 0.9,
                StrategyDimension.RESILIENCE: 0.4
            },
            applicability={
                "market_inefficiency": 0.9,
                "rapid_change": 0.8,
                "information_asymmetry": 0.9,
                "regulatory_change": 0.7
            },
            success_rate=0.65,
            last_updated=datetime.now(),
            version="1.0",
            tags=["opportunistic", "tactical", "flexible"]
        )
        
        # Notfallstrategie
        emergency = StrategyProfile(
            id="strategy_emergency",
            name="Notfallstrategie",
            description="Strategie für Krisenzeiten mit Fokus auf Schadensbegrenzung und Überleben.",
            dimensions={
                StrategyDimension.RISK: 0.3,
                StrategyDimension.RETURN: 0.2,
                StrategyDimension.TIME_HORIZON: 0.2,
                StrategyDimension.RESOURCES: 0.9,
                StrategyDimension.COMPLEXITY: 0.6,
                StrategyDimension.ADAPTABILITY: 0.8,
                StrategyDimension.RESILIENCE: 0.9
            },
            applicability={
                "market_crash": 0.9,
                "geopolitical_crisis": 0.9,
                "systemic_failure": 0.9,
                "black_swan_event": 0.8
            },
            success_rate=0.7,
            last_updated=datetime.now(),
            version="1.0",
            tags=["emergency", "crisis", "survival"]
        )
        
        # Füge Profile hinzu
        self.strategy_profiles = {
            "defensive": defensive,
            "offensive": offensive,
            "balanced": balanced,
            "opportunistic": opportunistic,
            "emergency": emergency
        }
    
    def add_strategy_profile(self, profile: StrategyProfile) -> bool:
        """
        Fügt ein neues Strategieprofil hinzu.
        """
        try:
            # Prüfe, ob ID bereits existiert
            profile_id = profile.id.split("strategy_")[-1] if profile.id.startswith("strategy_") else profile.id
            
            if profile_id in self.strategy_profiles:
                logger.warning(f"Strategieprofil mit ID {profile_id} existiert bereits")
                return False
            
            # Füge Profil hinzu
            self.strategy_profiles[profile_id] = profile
            
            logger.info(f"Strategieprofil hinzugefügt: {profile_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen eines Strategieprofils: {str(e)}")
            return False
    
    def update_strategy_profile(self, profile_id: str, updates: Dict[str, Any]) -> bool:
        """
        Aktualisiert ein bestehendes Strategieprofil.
        """
        try:
            # Prüfe, ob Profil existiert
            if profile_id not in self.strategy_profiles:
                logger.warning(f"Strategieprofil mit ID {profile_id} nicht gefunden")
                return False
            
            # Hole Profil
            profile = self.strategy_profiles[profile_id]
            
            # Aktualisiere Felder
            for key, value in updates.items():
                if key == "dimensions" and isinstance(value, dict):
                    # Konvertiere String-Keys zu Enum-Keys
                    dimensions = {}
                    for dim_key, dim_value in value.items():
                        if isinstance(dim_key, str):
                            try:
                                enum_key = StrategyDimension(dim_key)
                                dimensions[enum_key] = dim_value
                            except ValueError:
                                logger.warning(f"Ungültige Dimension: {dim_key}")
                        else:
                            dimensions[dim_key] = dim_value
                    
                    profile.dimensions.update(dimensions)
                elif key == "applicability" and isinstance(value, dict):
                    profile.applicability.update(value)
                elif key == "tags" and isinstance(value, list):
                    profile.tags = value
                elif hasattr(profile, key):
                    setattr(profile, key, value)
            
            # Aktualisiere Zeitstempel
            profile.last_updated = datetime.now()
            
            logger.info(f"Strategieprofil aktualisiert: {profile_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren eines Strategieprofils: {str(e)}")
            return False
    
    def evaluate_strategy(self, profile_id: str, scenario: str, 
                        outcome: Dict[str, Any], effectiveness: float) -> bool:
        """
        Bewertet die Effektivität einer Strategie in einem bestimmten Szenario.
        """
        try:
            # Prüfe, ob Profil existiert
            if profile_id not in self.strategy_profiles:
                logger.warning(f"Strategieprofil mit ID {profile_id} nicht gefunden")
                return False
            
            # Erstelle Bewertung
            evaluation = {
                "timestamp": datetime.now(),
                "scenario": scenario,
                "outcome": outcome,
                "effectiveness": effectiveness,
                "context": {
                    "market_conditions": outcome.get("market_conditions", "unknown"),
                    "risk_level": outcome.get("risk_level", 0.5),
                    "time_frame": outcome.get("time_frame", "medium")
                }
            }
            
            # Füge Bewertung hinzu
            if profile_id not in self.strategy_evaluations:
                self.strategy_evaluations[profile_id] = []
            
            self.strategy_evaluations[profile_id].append(evaluation)
            
            # Aktualisiere Erfolgsrate
            profile = self.strategy_profiles[profile_id]
            evaluations = self.strategy_evaluations[profile_id]
            
            if evaluations:
                success_rates = [eval["effectiveness"] for eval in evaluations]
                profile.success_rate = sum(success_rates) / len(success_rates)
            
            # Aktualisiere Anwendbarkeit
            if scenario in profile.applicability:
                # Gewichteter Durchschnitt aus bisheriger Anwendbarkeit und neuer Effektivität
                profile.applicability[scenario] = (
                    profile.applicability[scenario] * 0.7 + effectiveness * 0.3
                )
            else:
                profile.applicability[scenario] = effectiveness
            
            logger.info(f"Strategie bewertet: {profile_id}, Szenario = {scenario}, Effektivität = {effectiveness:.2f}")
            
            return True
        
        except Exception as e:
            logger.error(f"Fehler bei der Strategiebewertung: {str(e)}")
            return False
    
    def optimize_strategy_combination(self, scenario: str, constraints: Dict[str, float]) -> Dict[str, float]:
        """
        Optimiert die Kombination von Strategien für ein bestimmtes Szenario.
        """
        try:
            # Sammle anwendbare Strategien
            applicable_strategies = {}
            
            for profile_id, profile in self.strategy_profiles.items():
                if profile.active:
                    # Bestimme Anwendbarkeit für das Szenario
                    applicability = profile.applicability.get(scenario, 0.3)
                    
                    # Berücksichtige Erfolgsrate
                    weighted_applicability = applicability * profile.success_rate
                    
                    if weighted_applicability > 0.2:
                        applicable_strategies[profile_id] = weighted_applicability
            
            if not applicable_strategies:
                logger.warning(f"Keine anwendbaren Strategien für Szenario {scenario} gefunden")
                return {}
            
            # Normalisiere Anwendbarkeiten
            total_applicability = sum(applicable_strategies.values())
            
            if total_applicability > 0:
                for profile_id in applicable_strategies:
                    applicable_strategies[profile_id] /= total_applicability
            
            # Berücksichtige Constraints
            for profile_id in applicable_strategies:
                profile = self.strategy_profiles[profile_id]
                
                # Risikotoleranz
                if "risk_tolerance" in constraints:
                    risk_diff = abs(profile.dimensions[StrategyDimension.RISK] - constraints["risk_tolerance"])
                    applicable_strategies[profile_id] *= (1 - risk_diff * 0.5)
                
                # Renditeerwartung
                if "return_expectation" in constraints:
                    if profile.dimensions[StrategyDimension.RETURN] < constraints["return_expectation"]:
                        applicable_strategies[profile_id] *= (
                            profile.dimensions[StrategyDimension.RETURN] / constraints["return_expectation"]
                        )
                
                # Zeithorizont
                if "time_horizon" in constraints:
                    time_diff = abs(profile.dimensions[StrategyDimension.TIME_HORIZON] - constraints["time_horizon"])
                    applicable_strategies[profile_id] *= (1 - time_diff * 0.3)
                
                # Ressourcenbeschränkung
                if "resource_constraint" in constraints:
                    if profile.dimensions[StrategyDimension.RESOURCES] > constraints["resource_constraint"]:
                        applicable_strategies[profile_id] *= (
                            constraints["resource_constraint"] / profile.dimensions[StrategyDimension.RESOURCES]
                        )
            
            # Normalisiere erneut
            total_weight = sum(applicable_strategies.values())
            
            if total_weight > 0:
                for profile_id in applicable_strategies:
                    applicable_strategies[profile_id] /= total_weight
            
            # Speichere aktuelle Kombination
            self.current_strategy_combination = applicable_strategies
            
            logger.info(f"Strategiekombination optimiert für Szenario {scenario}: {len(applicable_strategies)} Strategien")
            
            return applicable_strategies
        
        except Exception as e:
            logger.error(f"Fehler bei der Strategieoptimierung: {str(e)}")
            return {}
    
    def get_strategy_recommendation(self, scenario: str, 
                                  escalation_level: EscalationLevel) -> StrategyRecommendation:
        """
        Generiert eine Strategieempfehlung basierend auf Szenario und Eskalationslevel.
        """
        try:
            # Optimiere Strategiekombination
            constraints = {
                "risk_tolerance": max(0.1, 1.0 - escalation_level.value / 5.0),
                "return_expectation": max(0.3, 0.8 - escalation_level.value / 10.0),
                "time_horizon": max(0.2, 1.0 - escalation_level.value / 5.0),
                "resource_constraint": min(0.9, 0.5 + escalation_level.value / 10.0)
            }
            
            strategy_combination = self.optimize_strategy_combination(scenario, constraints)
            
            if not strategy_combination:
                # Fallback auf Standardstrategie
                if escalation_level.value >= 4:
                    primary_strategy = "emergency"
                elif escalation_level.value >= 3:
                    primary_strategy = "defensive"
                elif escalation_level.value >= 2:
                    primary_strategy = "balanced"
                else:
                    primary_strategy = "offensive"
                
                strategy_combination = {primary_strategy: 1.0}
            
            # Bestimme primäre Strategie
            primary_strategy_id = max(strategy_combination, key=strategy_combination.get)
            primary_strategy = self.strategy_profiles[primary_strategy_id]
            
            # Bestimme sekundäre Strategien
            secondary_strategies = {}
            for profile_id, weight in strategy_combination.items():
                if profile_id != primary_strategy_id and weight > 0.1:
                    secondary_strategies[profile_id] = weight
            
            # Generiere Empfehlungstext
            recommendation_text = f"Empfohlene Strategie: {primary_strategy.name}. "
            recommendation_text += f"{primary_strategy.description} "
            
            if secondary_strategies:
                recommendation_text += "Ergänzt durch: "
                for profile_id, weight in secondary_strategies.items():
                    profile = self.strategy_profiles[profile_id]
                    recommendation_text += f"{profile.name} ({weight:.2f}), "
                
                recommendation_text = recommendation_text.rstrip(", ") + ". "
            
            # Generiere Begründung
            reasoning = f"Basierend auf dem Szenario '{scenario}' und einem Eskalationslevel von {escalation_level.name}. "
            reasoning += f"Die primäre Strategie hat eine Erfolgsrate von {primary_strategy.success_rate:.2f} "
            reasoning += f"und eine Anwendbarkeit von {primary_strategy.applicability.get(scenario, 0.3):.2f} für dieses Szenario. "
            
            if "risk_tolerance" in constraints:
                reasoning += f"Risikotoleranz: {constraints['risk_tolerance']:.2f}. "
            
            # Bestimme Zeitrahmen
            if primary_strategy.dimensions[StrategyDimension.TIME_HORIZON] > 0.7:
                timeframe = "Langfristig (mehrere Monate)"
            elif primary_strategy.dimensions[StrategyDimension.TIME_HORIZON] > 0.4:
                timeframe = "Mittelfristig (mehrere Wochen)"
            else:
                timeframe = "Kurzfristig (Tage bis Wochen)"
            
            # Berechne erwartete Auswirkung
            expected_impact = primary_strategy.dimensions[StrategyDimension.RETURN] * primary_strategy.success_rate
            
            # Berechne Vertrauensniveau
            confidence_level = primary_strategy.success_rate * primary_strategy.applicability.get(scenario, 0.3)
            
            # Generiere alternative Strategien
            alternative_strategies = []
            for profile_id, profile in self.strategy_profiles.items():
                if profile_id != primary_strategy_id and profile.active:
                    alternative_strategies.append({
                        "type": profile_id,
                        "name": profile.name,
                        "description": profile.description,
                        "suitability": profile.applicability.get(scenario, 0.3) * profile.success_rate
                    })
            
            # Sortiere nach Eignung
            alternative_strategies.sort(key=lambda x: x["suitability"], reverse=True)
            
            # Erstelle Empfehlung
            from miso.strategic.strategy_recommender import StrategyRecommendation
            
            recommendation = StrategyRecommendation(
                id=f"strategy_{primary_strategy_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                risk_level=1.0 - constraints["risk_tolerance"],
                escalation_level=escalation_level,
                primary_sources=[scenario],
                recommendation=recommendation_text,
                reasoning=reasoning,
                timeframe=timeframe,
                expected_impact=expected_impact,
                confidence_level=confidence_level,
                alternative_strategies=alternative_strategies[:2]
            )
            
            logger.info(f"Strategieempfehlung generiert: {primary_strategy_id}, Szenario = {scenario}, Eskalation = {escalation_level.name}")
            
            return recommendation
        
        except Exception as e:
            logger.error(f"Fehler bei der Generierung einer Strategieempfehlung: {str(e)}")
            # Rückgabe einer Standardempfehlung im Fehlerfall
            from miso.strategic.strategy_recommender import StrategyRecommendation
            
            default_recommendation = StrategyRecommendation(
                id=f"default_strategy_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                risk_level=0.5,
                escalation_level=escalation_level,
                primary_sources=[scenario, "error_recovery"],
                recommendation="Ausgewogene Strategie mit moderatem Risiko-Rendite-Profil als Fallback.",
                reasoning=f"Fehler bei der Strategieoptimierung: {str(e)}. Fallback auf ausgewogene Strategie.",
                timeframe="Mittelfristig (mehrere Wochen)",
                expected_impact=0.5,
                confidence_level=0.5,
                alternative_strategies=[]
            )
            
            return default_recommendation
    
    def get_strategy_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """
        Gibt detaillierte Informationen zu einem Strategieprofil zurück.
        """
        try:
            if profile_id not in self.strategy_profiles:
                logger.warning(f"Strategieprofil mit ID {profile_id} nicht gefunden")
                return None
            
            profile = self.strategy_profiles[profile_id]
            
            # Konvertiere zu Dictionary
            profile_dict = {
                "id": profile.id,
                "name": profile.name,
                "description": profile.description,
                "dimensions": {dim.value: value for dim, value in profile.dimensions.items()},
                "applicability": profile.applicability,
                "success_rate": profile.success_rate,
                "last_updated": profile.last_updated,
                "version": profile.version,
                "tags": profile.tags,
                "active": profile.active
            }
            
            # Füge Bewertungen hinzu, falls vorhanden
            if profile_id in self.strategy_evaluations:
                profile_dict["evaluations"] = self.strategy_evaluations[profile_id]
            
            return profile_dict
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen eines Strategieprofils: {str(e)}")
            return None
    
    def get_all_strategy_profiles(self) -> List[Dict[str, Any]]:
        """
        Gibt eine Liste aller Strategieprofile zurück.
        """
        try:
            profiles = []
            
            for profile_id, profile in self.strategy_profiles.items():
                # Konvertiere zu Dictionary
                profile_dict = {
                    "id": profile.id,
                    "name": profile.name,
                    "description": profile.description,
                    "dimensions": {dim.value: value for dim, value in profile.dimensions.items()},
                    "success_rate": profile.success_rate,
                    "last_updated": profile.last_updated,
                    "version": profile.version,
                    "tags": profile.tags,
                    "active": profile.active
                }
                
                profiles.append(profile_dict)
            
            return profiles
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen aller Strategieprofile: {str(e)}")
            return []
    
    def generate_strategy_report(self) -> Dict[str, Any]:
        """
        Generiert einen umfassenden Strategiebericht.
        """
        try:
            # Sammle aktive Profile
            active_profiles = [profile for profile_id, profile in self.strategy_profiles.items() if profile.active]
            
            # Sammle Bewertungen
            evaluations = []
            for profile_id, profile_evaluations in self.strategy_evaluations.items():
                for evaluation in profile_evaluations:
                    evaluations.append({
                        "profile_id": profile_id,
                        "timestamp": evaluation["timestamp"],
                        "scenario": evaluation["scenario"],
                        "effectiveness": evaluation["effectiveness"]
                    })
            
            # Sortiere nach Zeitstempel
            evaluations.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Analysiere Erfolgsraten
            success_rates = {}
            for profile_id, profile in self.strategy_profiles.items():
                success_rates[profile_id] = profile.success_rate
            
            # Analysiere Anwendbarkeit
            applicability_analysis = {}
            for scenario in set(eval["scenario"] for eval in evaluations):
                scenario_profiles = {}
                
                for profile_id, profile in self.strategy_profiles.items():
                    if scenario in profile.applicability:
                        scenario_profiles[profile_id] = profile.applicability[scenario]
                
                if scenario_profiles:
                    applicability_analysis[scenario] = scenario_profiles
            
            # Erstelle Bericht
            report = {
                "timestamp": datetime.now(),
                "active_profiles": len(active_profiles),
                "total_profiles": len(self.strategy_profiles),
                "total_evaluations": len(evaluations),
                "recent_evaluations": evaluations[:10],
                "success_rates": success_rates,
                "applicability_analysis": applicability_analysis,
                "current_strategy_combination": self.current_strategy_combination,
                "optimization_parameters": self.optimization_parameters
            }
            
            logger.info(f"Strategiebericht generiert: {len(active_profiles)} aktive Profile, {len(evaluations)} Bewertungen")
            
            return report
        
        except Exception as e:
            logger.error(f"Fehler bei der Berichtsgenerierung: {str(e)}")
            # Rückgabe eines Standardberichts im Fehlerfall
            return {
                "timestamp": datetime.now(),
                "error": str(e),
                "active_profiles": 0,
                "total_profiles": len(self.strategy_profiles),
                "total_evaluations": 0,
                "recent_evaluations": [],
                "success_rates": {},
                "applicability_analysis": {},
                "current_strategy_combination": {},
                "optimization_parameters": self.optimization_parameters
            }
