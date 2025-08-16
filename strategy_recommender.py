"""
DEEP-STATE-MODUL – Strategie-Empfehlungs-Komponente

Diese Komponente implementiert die Funktionalität zur Generierung von Handlungsempfehlungen
basierend auf Q-LOGIK und PRISM, mit Fokus auf Risikobewertung und strategische Entscheidungen.
"""

from typing import List, Dict, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass
from enum import Enum, auto
from datetime import datetime
import random
import json
import logging

# Import aus dem Deep-State-Modul
from miso.strategic.deep_state import EscalationLevel
from miso.quantum.qlogic.qmeasurement import QMeasurement
from miso.timeline.echo_prime import Timeline

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='strategy_recommender.log'
)
logger = logging.getLogger('StrategyRecommender')


@dataclass
class StrategyRecommendation:
    """Datenklasse für Strategieempfehlungen."""
    id: str
    timestamp: datetime
    risk_level: float
    escalation_level: EscalationLevel
    primary_sources: List[str]
    recommendation: str
    reasoning: str
    timeframe: str
    expected_impact: float
    confidence_level: float
    alternative_strategies: List[Dict[str, Any]]
    feedback_score: Optional[float] = None
    implementation_status: str = "pending"


class StrategyRecommender:
    """
    Generiert Handlungsempfehlungen basierend auf Q-LOGIK und PRISM.
    """
    
    def __init__(self):
        # Gewichtungen für verschiedene Datenquellen
        self.source_weights = {
            "market": 0.25,
            "threat": 0.30,
            "geo": 0.25,
            "ki": 0.20
        }
        
        # Strategie-Templates
        self.strategy_templates = {
            "defensive": {
                "description": "Defensive Strategie zur Risikominimierung",
                "actions": [
                    "Reduzierung von Exposure in volatilen Märkten",
                    "Erhöhung von Sicherheitsmaßnahmen",
                    "Diversifikation von Ressourcen",
                    "Aktivierung von Schutzprotokollen"
                ],
                "applicable_scenarios": ["market_crash", "cyber_threat", "geopolitical_tension"]
            },
            "offensive": {
                "description": "Offensive Strategie zur Chancenmaximierung",
                "actions": [
                    "Erhöhung von Exposure in Wachstumsmärkten",
                    "Aggressive Datensammlung",
                    "Proaktive Gegenmaßnahmen",
                    "Ressourcenkonzentration auf Schlüsselbereiche"
                ],
                "applicable_scenarios": ["market_opportunity", "competitive_advantage", "resource_abundance"]
            },
            "balanced": {
                "description": "Ausgewogene Strategie für moderate Risiko-Rendite-Profile",
                "actions": [
                    "Selektive Exposure-Anpassung",
                    "Gezielte Sicherheitsmaßnahmen",
                    "Moderate Ressourcenallokation",
                    "Flexible Reaktionsfähigkeit"
                ],
                "applicable_scenarios": ["mixed_signals", "uncertain_environment", "transitional_phase"]
            },
            "emergency": {
                "description": "Notfallstrategie für kritische Situationen",
                "actions": [
                    "Sofortige Risikominimierung",
                    "Aktivierung aller Schutzprotokolle",
                    "Kommunikation mit allen Stakeholdern",
                    "Vorbereitung auf Worst-Case-Szenarien"
                ],
                "applicable_scenarios": ["imminent_threat", "system_breach", "critical_failure"]
            }
        }
        
        # Risikoschwellenwerte
        self.risk_thresholds = {
            EscalationLevel.NEUTRAL: 0.2,
            EscalationLevel.NIEDRIG: 0.4,
            EscalationLevel.MODERAT: 0.6,
            EscalationLevel.ERHÖHT: 0.8,
            EscalationLevel.HOCH: 0.9,
            EscalationLevel.KRITISCH: 0.95
        }
        
        # Aktuelle Empfehlungen
        self.current_recommendations: List[StrategyRecommendation] = []
        
        # Historische Empfehlungen
        self.historical_recommendations: Dict[str, List[StrategyRecommendation]] = {}
        
        # Feedback-Daten
        self.feedback_data: Dict[str, Dict[str, Any]] = {}
        
        logger.info("StrategyRecommender initialisiert")
    
    def integrate_data(self, market_data: Dict[str, Any], threat_data: Dict[str, Any], 
                      geo_data: Dict[str, Any], ki_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integriert Daten aus verschiedenen Quellen.
        """
        try:
            # Prüfe, ob alle erforderlichen Daten vorhanden sind
            if not all([market_data, threat_data, geo_data, ki_data]):
                missing_data = []
                if not market_data: missing_data.append("market_data")
                if not threat_data: missing_data.append("threat_data")
                if not geo_data: missing_data.append("geo_data")
                if not ki_data: missing_data.append("ki_data")
                
                logger.warning(f"Fehlende Daten: {', '.join(missing_data)}")
                
                # Fülle fehlende Daten mit Standardwerten
                if not market_data: market_data = {"status": "no_data", "risk_level": 0.5}
                if not threat_data: threat_data = {"status": "no_data", "risk_level": 0.5}
                if not geo_data: geo_data = {"status": "no_data", "risk_level": 0.5}
                if not ki_data: ki_data = {"status": "no_data", "risk_level": 0.5}
            
            # Extrahiere Risikobewertungen
            market_risk = market_data.get("risk_level", 0.5) if isinstance(market_data, dict) else 0.5
            threat_risk = threat_data.get("risk_level", 0.5) if isinstance(threat_data, dict) else 0.5
            geo_risk = geo_data.get("risk_level", 0.5) if isinstance(geo_data, dict) else 0.5
            ki_risk = ki_data.get("risk_level", 0.5) if isinstance(ki_data, dict) else 0.5
            
            # Gewichtete Risikobewertung
            weighted_risk = (
                market_risk * self.source_weights["market"] +
                threat_risk * self.source_weights["threat"] +
                geo_risk * self.source_weights["geo"] +
                ki_risk * self.source_weights["ki"]
            )
            
            # Identifiziere Hauptrisikoquellen
            risk_sources = []
            if market_risk > 0.7: risk_sources.append("market_volatility")
            if threat_risk > 0.7: risk_sources.append("external_threats")
            if geo_risk > 0.7: risk_sources.append("geopolitical_tensions")
            if ki_risk > 0.7: risk_sources.append("hostile_ai")
            
            # Identifiziere Korrelationen zwischen Datenquellen
            correlations = {}
            data_pairs = [
                ("market", "threat", market_risk, threat_risk),
                ("market", "geo", market_risk, geo_risk),
                ("market", "ki", market_risk, ki_risk),
                ("threat", "geo", threat_risk, geo_risk),
                ("threat", "ki", threat_risk, ki_risk),
                ("geo", "ki", geo_risk, ki_risk)
            ]
            
            for source1, source2, risk1, risk2 in data_pairs:
                # Simulierte Korrelation
                correlation = abs(risk1 - risk2) < 0.2
                correlations[f"{source1}_{source2}"] = correlation
            
            # Integrierte Daten
            integrated_data = {
                "timestamp": datetime.now(),
                "weighted_risk": weighted_risk,
                "risk_sources": risk_sources,
                "correlations": correlations,
                "market_data": market_data,
                "threat_data": threat_data,
                "geo_data": geo_data,
                "ki_data": ki_data
            }
            
            logger.info(f"Daten integriert: Gewichtetes Risiko = {weighted_risk:.2f}")
            
            return integrated_data
        
        except Exception as e:
            logger.error(f"Fehler bei der Datenintegration: {str(e)}")
            # Rückgabe von Standarddaten im Fehlerfall
            return {
                "timestamp": datetime.now(),
                "weighted_risk": 0.5,
                "risk_sources": ["error_in_data_integration"],
                "correlations": {},
                "error": str(e)
            }
    
    def assess_risk(self, integrated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bewertet Gesamtrisiko basierend auf integrierten Daten.
        """
        try:
            # Extrahiere gewichtetes Risiko
            weighted_risk = integrated_data.get("weighted_risk", 0.5)
            
            # Bestimme Risikobereiche
            risk_areas = []
            
            # Prüfe Marktdaten
            market_data = integrated_data.get("market_data", {})
            if isinstance(market_data, dict):
                if market_data.get("status") != "no_data":
                    # Identifiziere Risikobereiche in Marktdaten
                    if "trends" in market_data:
                        trends = market_data["trends"]
                        for market_type, indicators in trends.get("trends", {}).items():
                            for indicator, data in indicators.items():
                                if data.get("short_term_trend") in ["strong_downtrend", "downtrend"]:
                                    risk_areas.append(f"market_{market_type}_{indicator}")
            
            # Prüfe Bedrohungsdaten
            threat_data = integrated_data.get("threat_data", {})
            if isinstance(threat_data, dict):
                if threat_data.get("status") != "no_data":
                    # Identifiziere Risikobereiche in Bedrohungsdaten
                    for threat in threat_data.get("active_threats", []):
                        if threat.get("severity", 0) > 0.7:
                            risk_areas.append(f"threat_{threat.get('type', 'unknown')}")
            
            # Prüfe geopolitische Daten
            geo_data = integrated_data.get("geo_data", {})
            if isinstance(geo_data, dict):
                if geo_data.get("status") != "no_data":
                    # Identifiziere Risikobereiche in geopolitischen Daten
                    tensions = geo_data.get("global_tensions", {})
                    for region, tension in tensions.get("regional_tensions", {}).items():
                        if tension > 0.7:
                            risk_areas.append(f"geo_tension_{region}")
            
            # Prüfe KI-Daten
            ki_data = integrated_data.get("ki_data", {})
            if isinstance(ki_data, dict):
                if ki_data.get("status") != "no_data":
                    # Identifiziere Risikobereiche in KI-Daten
                    for pattern in ki_data.get("adversarial_patterns", []):
                        if pattern.get("threat_level", 0) > 0.7:
                            risk_areas.append(f"ki_{pattern.get('behavior_pattern', 'unknown')}")
            
            # Bestimme Eskalationslevel
            escalation_level = self._determine_escalation_level(weighted_risk)
            
            # Berechne Risikowerte für verschiedene Szenarien
            scenarios = {
                "base_case": weighted_risk,
                "worst_case": min(1.0, weighted_risk * 1.5),
                "best_case": max(0.0, weighted_risk * 0.5)
            }
            
            # Risikobewertung
            risk_assessment = {
                "timestamp": datetime.now(),
                "overall_risk": weighted_risk,
                "escalation_level": escalation_level,
                "risk_areas": risk_areas,
                "scenarios": scenarios,
                "confidence": 0.8 if len(risk_areas) > 0 else 0.6
            }
            
            logger.info(f"Risikobewertung: Gesamtrisiko = {weighted_risk:.2f}, Eskalationslevel = {escalation_level.name}")
            
            return risk_assessment
        
        except Exception as e:
            logger.error(f"Fehler bei der Risikobewertung: {str(e)}")
            # Rückgabe von Standarddaten im Fehlerfall
            return {
                "timestamp": datetime.now(),
                "overall_risk": 0.5,
                "escalation_level": EscalationLevel.MODERAT,
                "risk_areas": ["error_in_risk_assessment"],
                "scenarios": {"base_case": 0.5, "worst_case": 0.75, "best_case": 0.25},
                "confidence": 0.5,
                "error": str(e)
            }
    
    def _determine_escalation_level(self, risk_value: float) -> EscalationLevel:
        """
        Bestimmt das Eskalationslevel basierend auf dem Risikowert.
        """
        if risk_value < self.risk_thresholds[EscalationLevel.NEUTRAL]:
            return EscalationLevel.NEUTRAL
        elif risk_value < self.risk_thresholds[EscalationLevel.NIEDRIG]:
            return EscalationLevel.NIEDRIG
        elif risk_value < self.risk_thresholds[EscalationLevel.MODERAT]:
            return EscalationLevel.MODERAT
        elif risk_value < self.risk_thresholds[EscalationLevel.ERHÖHT]:
            return EscalationLevel.ERHÖHT
        elif risk_value < self.risk_thresholds[EscalationLevel.HOCH]:
            return EscalationLevel.HOCH
        else:
            return EscalationLevel.KRITISCH
    
    def generate_recommendations(self, risk_assessment: Dict[str, Any]) -> List[StrategyRecommendation]:
        """
        Generiert spezifische Handlungsempfehlungen.
        """
        try:
            # Extrahiere Risikobewertung
            overall_risk = risk_assessment.get("overall_risk", 0.5)
            escalation_level = risk_assessment.get("escalation_level", EscalationLevel.MODERAT)
            risk_areas = risk_assessment.get("risk_areas", [])
            
            # Bestimme Strategietyp basierend auf Risiko
            if escalation_level in [EscalationLevel.HOCH, EscalationLevel.KRITISCH]:
                strategy_type = "emergency"
            elif escalation_level == EscalationLevel.ERHÖHT:
                strategy_type = "defensive"
            elif escalation_level == EscalationLevel.MODERAT:
                strategy_type = "balanced"
            else:
                strategy_type = "offensive" if overall_risk < 0.3 else "balanced"
            
            # Wähle Strategie-Template
            template = self.strategy_templates[strategy_type]
            
            # Generiere Empfehlung
            recommendation_text = f"Empfohlene Strategie: {template['description']}. "
            recommendation_text += "Maßnahmen: " + ", ".join(template["actions"])
            
            # Generiere Begründung
            reasoning = f"Basierend auf einem Gesamtrisiko von {overall_risk:.2f} "
            reasoning += f"und einem Eskalationslevel von {escalation_level.name}. "
            
            if risk_areas:
                reasoning += f"Hauptrisikobereiche: {', '.join(risk_areas[:3])}. "
            
            # Bestimme Zeitrahmen
            if escalation_level in [EscalationLevel.HOCH, EscalationLevel.KRITISCH]:
                timeframe = "Sofort (innerhalb von 24 Stunden)"
            elif escalation_level == EscalationLevel.ERHÖHT:
                timeframe = "Kurzfristig (1-3 Tage)"
            elif escalation_level == EscalationLevel.MODERAT:
                timeframe = "Mittelfristig (1-2 Wochen)"
            else:
                timeframe = "Langfristig (2-4 Wochen)"
            
            # Berechne erwartete Auswirkung
            expected_impact = 0.7 if strategy_type in ["emergency", "defensive"] else 0.5
            
            # Berechne Vertrauensniveau
            confidence_level = risk_assessment.get("confidence", 0.7)
            
            # Generiere alternative Strategien
            alternative_strategies = []
            for alt_type, alt_template in self.strategy_templates.items():
                if alt_type != strategy_type:
                    alternative_strategies.append({
                        "type": alt_type,
                        "description": alt_template["description"],
                        "actions": alt_template["actions"][:2],
                        "suitability": 0.8 if abs(overall_risk - 0.5) < 0.2 else 0.5
                    })
            
            # Erstelle Empfehlung
            recommendation = StrategyRecommendation(
                id=f"strategy_{strategy_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                risk_level=overall_risk,
                escalation_level=escalation_level,
                primary_sources=risk_areas[:3] if risk_areas else ["general_risk_assessment"],
                recommendation=recommendation_text,
                reasoning=reasoning,
                timeframe=timeframe,
                expected_impact=expected_impact,
                confidence_level=confidence_level,
                alternative_strategies=alternative_strategies[:2]
            )
            
            # Speichere Empfehlung
            self.current_recommendations.append(recommendation)
            
            if strategy_type not in self.historical_recommendations:
                self.historical_recommendations[strategy_type] = []
            
            self.historical_recommendations[strategy_type].append(recommendation)
            
            logger.info(f"Empfehlung generiert: {strategy_type}, Risiko = {overall_risk:.2f}, Eskalation = {escalation_level.name}")
            
            return [recommendation]
        
        except Exception as e:
            logger.error(f"Fehler bei der Empfehlungsgenerierung: {str(e)}")
            # Rückgabe einer Standardempfehlung im Fehlerfall
            default_recommendation = StrategyRecommendation(
                id=f"default_strategy_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                risk_level=0.5,
                escalation_level=EscalationLevel.MODERAT,
                primary_sources=["error_in_recommendation_generation"],
                recommendation="Standard-Sicherheitsmaßnahmen beibehalten und Systeme überwachen.",
                reasoning=f"Fehler bei der Empfehlungsgenerierung: {str(e)}",
                timeframe="Mittelfristig (1-2 Wochen)",
                expected_impact=0.5,
                confidence_level=0.5,
                alternative_strategies=[]
            )
            
            return [default_recommendation]
    
    def provide_feedback(self, recommendation_id: str, feedback_score: float, comments: str = "") -> bool:
        """
        Nimmt Feedback zu einer Empfehlung entgegen.
        """
        try:
            # Suche Empfehlung
            recommendation = None
            for rec in self.current_recommendations:
                if rec.id == recommendation_id:
                    recommendation = rec
                    break
            
            if not recommendation:
                for recs in self.historical_recommendations.values():
                    for rec in recs:
                        if rec.id == recommendation_id:
                            recommendation = rec
                            break
                    if recommendation:
                        break
            
            if not recommendation:
                logger.warning(f"Empfehlung mit ID {recommendation_id} nicht gefunden")
                return False
            
            # Aktualisiere Feedback
            recommendation.feedback_score = feedback_score
            
            # Speichere Feedback-Daten
            self.feedback_data[recommendation_id] = {
                "timestamp": datetime.now(),
                "recommendation_id": recommendation_id,
                "feedback_score": feedback_score,
                "comments": comments,
                "strategy_type": recommendation.id.split("_")[1] if "_" in recommendation.id else "unknown"
            }
            
            logger.info(f"Feedback für Empfehlung {recommendation_id}: Score = {feedback_score}")
            
            # Passe Gewichtungen basierend auf Feedback an
            if feedback_score < 0.3:
                # Negative Bewertung - reduziere Gewichtung der Hauptquellen
                for source in recommendation.primary_sources:
                    if "market" in source and "market" in self.source_weights:
                        self.source_weights["market"] = max(0.1, self.source_weights["market"] - 0.05)
                    elif "threat" in source and "threat" in self.source_weights:
                        self.source_weights["threat"] = max(0.1, self.source_weights["threat"] - 0.05)
                    elif "geo" in source and "geo" in self.source_weights:
                        self.source_weights["geo"] = max(0.1, self.source_weights["geo"] - 0.05)
                    elif "ki" in source and "ki" in self.source_weights:
                        self.source_weights["ki"] = max(0.1, self.source_weights["ki"] - 0.05)
            elif feedback_score > 0.7:
                # Positive Bewertung - erhöhe Gewichtung der Hauptquellen
                for source in recommendation.primary_sources:
                    if "market" in source and "market" in self.source_weights:
                        self.source_weights["market"] = min(0.5, self.source_weights["market"] + 0.05)
                    elif "threat" in source and "threat" in self.source_weights:
                        self.source_weights["threat"] = min(0.5, self.source_weights["threat"] + 0.05)
                    elif "geo" in source and "geo" in self.source_weights:
                        self.source_weights["geo"] = min(0.5, self.source_weights["geo"] + 0.05)
                    elif "ki" in source and "ki" in self.source_weights:
                        self.source_weights["ki"] = min(0.5, self.source_weights["ki"] + 0.05)
            
            # Normalisiere Gewichtungen
            total_weight = sum(self.source_weights.values())
            for source in self.source_weights:
                self.source_weights[source] /= total_weight
            
            return True
        
        except Exception as e:
            logger.error(f"Fehler bei der Feedback-Verarbeitung: {str(e)}")
            return False
    
    def generate_strategy_report(self) -> Dict[str, Any]:
        """
        Generiert einen umfassenden Strategiebericht.
        """
        try:
            # Sammle aktuelle Empfehlungen
            current_recommendations = [self._recommendation_to_dict(rec) for rec in self.current_recommendations]
            
            # Analysiere historische Empfehlungen
            historical_analysis = {}
            for strategy_type, recommendations in self.historical_recommendations.items():
                if recommendations:
                    avg_risk = sum(rec.risk_level for rec in recommendations) / len(recommendations)
                    avg_impact = sum(rec.expected_impact for rec in recommendations) / len(recommendations)
                    avg_confidence = sum(rec.confidence_level for rec in recommendations) / len(recommendations)
                    
                    feedback_scores = [rec.feedback_score for rec in recommendations if rec.feedback_score is not None]
                    avg_feedback = sum(feedback_scores) / len(feedback_scores) if feedback_scores else None
                    
                    historical_analysis[strategy_type] = {
                        "count": len(recommendations),
                        "avg_risk": avg_risk,
                        "avg_impact": avg_impact,
                        "avg_confidence": avg_confidence,
                        "avg_feedback": avg_feedback,
                        "last_recommendation": self._recommendation_to_dict(recommendations[-1])
                    }
            
            # Erstelle Bericht
            report = {
                "timestamp": datetime.now(),
                "current_recommendations": current_recommendations,
                "historical_analysis": historical_analysis,
                "source_weights": self.source_weights,
                "recommendation_count": len(self.current_recommendations)
            }
            
            logger.info(f"Strategiebericht generiert: {len(current_recommendations)} aktuelle Empfehlungen")
            
            # Lösche alte Empfehlungen
            self.current_recommendations = []
            
            return report
        
        except Exception as e:
            logger.error(f"Fehler bei der Berichtsgenerierung: {str(e)}")
            # Rückgabe eines Standardberichts im Fehlerfall
            return {
                "timestamp": datetime.now(),
                "error": str(e),
                "current_recommendations": [],
                "historical_analysis": {},
                "source_weights": self.source_weights,
                "recommendation_count": 0
            }
    
    def _recommendation_to_dict(self, recommendation: StrategyRecommendation) -> Dict[str, Any]:
        """
        Konvertiert ein StrategyRecommendation-Objekt in ein Dictionary.
        """
        return {
            "id": recommendation.id,
            "timestamp": recommendation.timestamp,
            "risk_level": recommendation.risk_level,
            "escalation_level": recommendation.escalation_level.name,
            "primary_sources": recommendation.primary_sources,
            "recommendation": recommendation.recommendation,
            "reasoning": recommendation.reasoning,
            "timeframe": recommendation.timeframe,
            "expected_impact": recommendation.expected_impact,
            "confidence_level": recommendation.confidence_level,
            "alternative_strategies": recommendation.alternative_strategies,
            "feedback_score": recommendation.feedback_score,
            "implementation_status": recommendation.implementation_status
        }
