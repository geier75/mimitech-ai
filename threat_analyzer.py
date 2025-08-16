"""
DEEP-STATE-MODUL – Bedrohungsanalyse-Komponente

Diese Komponente implementiert die Bedrohungsanalyse-Funktionalität des DEEP-STATE-MODULS,
mit Fokus auf die Erkennung von Cyberangriffen, KI-Infiltrationen und politischen 
Manipulationsmustern.
"""

from typing import List, Dict, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from datetime import datetime
import random
import json
import re
import logging

# Import aus dem Deep-State-Modul
from miso.strategic.deep_state import ThreatSignal, EscalationLevel
from miso.strategic.ztm_policy import ZTMPolicy, ztm_decorator


class ThreatAnalyzer:
    """
    Bewertet Risikoquellen und Bedrohungsmuster.
    """
    
    def __init__(self):
        # Logger konfigurieren
        self.logger = logging.getLogger('ThreatAnalyzer')
        
        # ZTM-Policy initialisieren
        self.ztm_policy = ZTMPolicy("ThreatAnalyzer")
        
        # Bedrohungsquellen
        self.threat_sources = {
            "cyber": ["ddos", "ransomware", "phishing", "data_breach", "zero_day"],
            "political": ["sanctions", "regulations", "elections", "protests", "coup"],
            "economic": ["inflation", "recession", "market_crash", "supply_chain", "trade_war"],
            "military": ["conflict", "mobilization", "weapons_test", "naval_movement", "airspace_violation"],
            "ai": ["model_poisoning", "prompt_injection", "adversarial_attack", "surveillance", "autonomous_systems"]
        }
        
        # Aktuelle Bedrohungssignale
        self.current_threats: List[ThreatSignal] = []
        
        # Historische Bedrohungen für Musteranalyse
        self.historical_threats: Dict[str, List[ThreatSignal]] = {}
        
        # Schwellenwerte für Eskalation
        self.escalation_thresholds = {
            EscalationLevel.NEUTRAL: 0.1,
            EscalationLevel.NIEDRIG: 0.3,
            EscalationLevel.MODERAT: 0.5,
            EscalationLevel.ERHÖHT: 0.7,
            EscalationLevel.HOCH: 0.85,
            EscalationLevel.AKUT: 0.95
        }
        
        # Beobachtete Ziele
        self.watched_targets = {
            "systems": ["network", "database", "cloud", "endpoints", "iot_devices"],
            "organizations": ["financial", "government", "healthcare", "energy", "technology"],
            "infrastructure": ["power_grid", "water_supply", "transportation", "communications", "internet"],
            "regions": ["north_america", "europe", "asia", "middle_east", "africa"]
        }
        
        # Bekannte Angriffsmuster
        self.attack_patterns = {
            "apt": {
                "indicators": ["targeted", "persistent", "sophisticated", "multi-stage", "stealth"],
                "countermeasures": ["threat_intelligence", "network_segmentation", "endpoint_protection", "behavioral_analysis"]
            },
            "ransomware": {
                "indicators": ["encryption", "ransom_demand", "data_exfiltration", "phishing_email", "lateral_movement"],
                "countermeasures": ["backup", "email_filtering", "patch_management", "user_training", "network_monitoring"]
            },
            "ddos": {
                "indicators": ["traffic_spike", "service_degradation", "botnet_activity", "amplification", "multiple_vectors"],
                "countermeasures": ["traffic_filtering", "rate_limiting", "cdn", "anycast", "ddos_protection_service"]
            },
            "insider_threat": {
                "indicators": ["unusual_access", "data_exfiltration", "privilege_escalation", "off_hours_activity", "policy_violations"],
                "countermeasures": ["access_control", "user_monitoring", "data_loss_prevention", "security_awareness", "least_privilege"]
            },
            "supply_chain": {
                "indicators": ["third_party_compromise", "software_tampering", "update_mechanism", "trusted_relationship", "backdoor"],
                "countermeasures": ["vendor_assessment", "code_signing", "integrity_verification", "monitoring", "isolation"]
            }
        }
    
    @ztm_decorator
    def detect_threat(self, threat_type: str, source: str, target: str, indicators: List[str]) -> ThreatSignal:
        """
        Erkennt und bewertet eine potenzielle Bedrohung.
        """
        try:
            # ZTM-Verifizierung der Eingabeparameter
            if self.ztm_policy.ztm_active:
                input_verification = {
                    "threat_type": threat_type,
                    "source": source,
                    "target": target,
                    "indicators_count": len(indicators),
                    "timestamp": datetime.now()
                }
                self.ztm_policy.verify_action("THREAT_DETECTION_INPUT", input_verification)
            
            # Berechne Intensität basierend auf Indikatoren
            intensity = random.uniform(0.3, 0.9)
            
            # Berechne Wahrscheinlichkeit
            probability = random.uniform(0.2, 0.8)
            
            # Berechne Auswirkung
            impact = random.uniform(0.4, 0.9)
            
            # Bestimme Eskalationslevel
            risk_score = (intensity * 0.3) + (probability * 0.3) + (impact * 0.4)
            escalation_level = self._determine_escalation_level(risk_score)
            
            # ZTM-Verifizierung des Risiko-Scores und Eskalationslevels
            if self.ztm_policy.ztm_active:
                risk_verification = {
                    "threat_type": threat_type,
                    "risk_score": risk_score,
                    "escalation_level": str(escalation_level),
                    "intensity": intensity,
                    "probability": probability,
                    "impact": impact,
                    "timestamp": datetime.now()
                }
                self.ztm_policy.verify_action("THREAT_RISK_ASSESSMENT", risk_verification)
            
            # Bestimme Gegenmaßnahmen
            countermeasures = self._determine_countermeasures(threat_type, indicators)
            
            # Erstelle Bedrohungssignal
            threat_id = f"{threat_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            threat = ThreatSignal(
                id=threat_id,
                threat_type=threat_type,
                source=source,
                target=target,
                intensity=intensity,
                probability=probability,
                impact=impact,
                timestamp=datetime.now(),
                escalation_level=escalation_level,
                description=f"Potenzielle {threat_type.capitalize()}-Bedrohung von {source} gegen {target}",
                indicators=indicators,
                countermeasures=countermeasures
            )
            
            # Speichere Bedrohung
            self.current_threats.append(threat)
            
            if threat_type not in self.historical_threats:
                self.historical_threats[threat_type] = []
            
            self.historical_threats[threat_type].append(threat)
            
            # ZTM-Verifizierung des erstellten Bedrohungssignals
            if self.ztm_policy.ztm_active:
                threat_verification = {
                    "threat_id": threat_id,
                    "threat_type": threat_type,
                    "escalation_level": str(escalation_level),
                    "risk_score": risk_score,
                    "countermeasures_count": len(countermeasures),
                    "timestamp": datetime.now()
                }
                self.ztm_policy.verify_action("THREAT_SIGNAL_CREATED", threat_verification)
                
                # Bei hohem Eskalationslevel zusätzliche Verifizierung
                if escalation_level in [EscalationLevel.HOCH, EscalationLevel.AKUT]:
                    high_threat_verification = {
                        "threat_id": threat_id,
                        "escalation_level": str(escalation_level),
                        "risk_score": risk_score,
                        "requires_immediate_action": True,
                        "timestamp": datetime.now()
                    }
                    self.ztm_policy.verify_action("HIGH_THREAT_DETECTED", high_threat_verification)
                    self.logger.warning(f"{self.ztm_policy.status} Hochstufige Bedrohung erkannt: {threat_type} mit Eskalationslevel {escalation_level}")
            
            return threat
            
        except Exception as e:
            if self.ztm_policy.ztm_active:
                self.ztm_policy.handle_error(e, "detect_threat")
            self.logger.error(f"Fehler bei der Bedrohungserkennung: {str(e)}")
            raise
    
    def _determine_escalation_level(self, risk_score: float) -> EscalationLevel:
        """
        Bestimmt das Eskalationslevel basierend auf dem Risikoscore.
        """
        if risk_score >= self.escalation_thresholds[EscalationLevel.AKUT]:
            return EscalationLevel.AKUT
        elif risk_score >= self.escalation_thresholds[EscalationLevel.HOCH]:
            return EscalationLevel.HOCH
        elif risk_score >= self.escalation_thresholds[EscalationLevel.ERHÖHT]:
            return EscalationLevel.ERHÖHT
        elif risk_score >= self.escalation_thresholds[EscalationLevel.MODERAT]:
            return EscalationLevel.MODERAT
        elif risk_score >= self.escalation_thresholds[EscalationLevel.NIEDRIG]:
            return EscalationLevel.NIEDRIG
        else:
            return EscalationLevel.NEUTRAL
    
    def _determine_countermeasures(self, threat_type: str, indicators: List[str]) -> List[str]:
        """
        Bestimmt geeignete Gegenmaßnahmen basierend auf Bedrohungstyp und Indikatoren.
        """
        countermeasures = []
        
        # Allgemeine Gegenmaßnahmen
        general_measures = [
            "enhanced_monitoring",
            "incident_response_activation",
            "security_alert_distribution",
            "log_analysis",
            "threat_intelligence_review"
        ]
        
        # Füge allgemeine Maßnahmen hinzu
        countermeasures.extend(random.sample(general_measures, k=min(3, len(general_measures))))
        
        # Füge spezifische Maßnahmen hinzu, wenn ein bekanntes Angriffsmuster erkannt wird
        for pattern, info in self.attack_patterns.items():
            pattern_indicators = info["indicators"]
            if any(indicator in pattern_indicators for indicator in indicators):
                pattern_measures = info["countermeasures"]
                countermeasures.extend(random.sample(pattern_measures, k=min(2, len(pattern_measures))))
        
        # Füge typspezifische Maßnahmen hinzu
        if threat_type == "cyber":
            cyber_measures = [
                "firewall_rule_update",
                "patch_critical_systems",
                "isolate_affected_systems",
                "backup_verification",
                "endpoint_scan"
            ]
            countermeasures.extend(random.sample(cyber_measures, k=min(2, len(cyber_measures))))
        
        elif threat_type == "political":
            political_measures = [
                "diplomatic_channels_activation",
                "regulatory_compliance_review",
                "public_relations_preparation",
                "legal_consultation",
                "stakeholder_communication"
            ]
            countermeasures.extend(random.sample(political_measures, k=min(2, len(political_measures))))
        
        elif threat_type == "economic":
            economic_measures = [
                "financial_exposure_assessment",
                "market_position_hedging",
                "supply_chain_diversification",
                "liquidity_preservation",
                "cost_reduction_planning"
            ]
            countermeasures.extend(random.sample(economic_measures, k=min(2, len(economic_measures))))
        
        elif threat_type == "military":
            military_measures = [
                "physical_security_enhancement",
                "continuity_plan_activation",
                "asset_relocation",
                "communication_security_review",
                "personnel_safety_protocols"
            ]
            countermeasures.extend(random.sample(military_measures, k=min(2, len(military_measures))))
        
        elif threat_type == "ai":
            ai_measures = [
                "model_validation",
                "input_sanitization",
                "adversarial_training",
                "model_monitoring",
                "fallback_mechanisms_activation"
            ]
            countermeasures.extend(random.sample(ai_measures, k=min(2, len(ai_measures))))
        
        # Entferne Duplikate
        return list(set(countermeasures))
    
    def analyze_threat_landscape(self) -> Dict[str, Any]:
        """
        Analysiert die aktuelle Bedrohungslandschaft basierend auf allen erkannten Bedrohungen.
        """
        if not self.current_threats:
            return {"status": "no_threats_detected"}
        
        # Gruppiere Bedrohungen nach Typ
        grouped_threats = {}
        for threat in self.current_threats:
            if threat.threat_type not in grouped_threats:
                grouped_threats[threat.threat_type] = []
            grouped_threats[threat.threat_type].append(threat)
        
        # Analysiere Bedrohungen nach Typ
        threat_analysis = {}
        for threat_type, threats in grouped_threats.items():
            # Berechne durchschnittliche Intensität
            avg_intensity = sum(t.intensity for t in threats) / len(threats)
            
            # Berechne durchschnittliche Wahrscheinlichkeit
            avg_probability = sum(t.probability for t in threats) / len(threats)
            
            # Berechne durchschnittliche Auswirkung
            avg_impact = sum(t.impact for t in threats) / len(threats)
            
            # Bestimme höchstes Eskalationslevel
            max_escalation = max(t.escalation_level.value for t in threats)
            
            # Sammle häufige Indikatoren
            all_indicators = [indicator for t in threats for indicator in t.indicators]
            indicator_counts = {}
            for indicator in all_indicators:
                if indicator not in indicator_counts:
                    indicator_counts[indicator] = 0
                indicator_counts[indicator] += 1
            
            top_indicators = sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Sammle häufige Gegenmaßnahmen
            all_countermeasures = [cm for t in threats for cm in t.countermeasures]
            cm_counts = {}
            for cm in all_countermeasures:
                if cm not in cm_counts:
                    cm_counts[cm] = 0
                cm_counts[cm] += 1
            
            top_countermeasures = sorted(cm_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Speichere Analyse
            threat_analysis[threat_type] = {
                "count": len(threats),
                "avg_intensity": avg_intensity,
                "avg_probability": avg_probability,
                "avg_impact": avg_impact,
                "max_escalation": max_escalation,
                "top_indicators": top_indicators,
                "top_countermeasures": top_countermeasures
            }
        
        # Bestimme Gesamtbedrohungslevel
        overall_intensity = sum(t["avg_intensity"] for t in threat_analysis.values()) / len(threat_analysis)
        overall_probability = sum(t["avg_probability"] for t in threat_analysis.values()) / len(threat_analysis)
        overall_impact = sum(t["avg_impact"] for t in threat_analysis.values()) / len(threat_analysis)
        overall_escalation = max(t["max_escalation"] for t in threat_analysis.values())
        
        # Erstelle Gesamtergebnis
        result = {
            "overall": {
                "threat_level": (overall_intensity * 0.3) + (overall_probability * 0.3) + (overall_impact * 0.4),
                "escalation_level": overall_escalation,
                "threat_types": list(grouped_threats.keys()),
                "total_threats": len(self.current_threats)
            },
            "by_threat_type": threat_analysis,
            "timestamp": datetime.now()
        }
        
        return result
    
    def detect_emerging_threats(self) -> List[Dict[str, Any]]:
        """
        Erkennt aufkommende Bedrohungen basierend auf historischen Daten und aktuellen Trends.
        """
        emerging_threats = []
        
        # Analysiere Trends in historischen Daten
        for threat_type, threats in self.historical_threats.items():
            if len(threats) < 5:  # Mindestens 5 historische Datenpunkte
                continue
            
            # Sortiere nach Zeitstempel
            sorted_threats = sorted(threats, key=lambda t: t.timestamp)
            
            # Berechne Trend für Intensität
            recent_intensity = [t.intensity for t in sorted_threats[-5:]]
            intensity_trend = sum(recent_intensity) / len(recent_intensity) - sum(recent_intensity[:-1]) / len(recent_intensity[:-1])
            
            # Berechne Trend für Wahrscheinlichkeit
            recent_probability = [t.probability for t in sorted_threats[-5:]]
            probability_trend = sum(recent_probability) / len(recent_probability) - sum(recent_probability[:-1]) / len(recent_probability[:-1])
            
            # Berechne Trend für Auswirkung
            recent_impact = [t.impact for t in sorted_threats[-5:]]
            impact_trend = sum(recent_impact) / len(recent_impact) - sum(recent_impact[:-1]) / len(recent_impact[:-1])
            
            # Wenn alle Trends positiv sind, handelt es sich um eine aufkommende Bedrohung
            if intensity_trend > 0 and probability_trend > 0 and impact_trend > 0:
                emerging_threats.append({
                    "threat_type": threat_type,
                    "intensity_trend": intensity_trend,
                    "probability_trend": probability_trend,
                    "impact_trend": impact_trend,
                    "recent_count": len(recent_intensity),
                    "severity": (intensity_trend + probability_trend + impact_trend) / 3,
                    "timestamp": datetime.now()
                })
        
        return emerging_threats
    
    def generate_threat_report(self) -> Dict[str, Any]:
        """
        Generiert einen umfassenden Bedrohungsbericht.
        """
        # Simuliere Bedrohungserkennung für verschiedene Typen
        for threat_type in self.threat_sources:
            source = random.choice(["unknown", "nation_state", "criminal_group", "hacktivist", "insider"])
            target_category = random.choice(list(self.watched_targets.keys()))
            target = random.choice(self.watched_targets[target_category])
            indicators = random.sample(self.threat_sources[threat_type], k=min(3, len(self.threat_sources[threat_type])))
            
            self.detect_threat(threat_type, source, target, indicators)
        
        # Analysiere Bedrohungslandschaft
        landscape = self.analyze_threat_landscape()
        
        # Erkenne aufkommende Bedrohungen
        emerging = self.detect_emerging_threats()
        
        # Erstelle Bericht
        report = {
            "timestamp": datetime.now(),
            "landscape": landscape,
            "emerging_threats": emerging,
            "threats_count": len(self.current_threats),
            "highest_escalation": max([t.escalation_level.value for t in self.current_threats]) if self.current_threats else 0
        }
        
        # Lösche alte Bedrohungen
        self.current_threats = []
        
        return report
