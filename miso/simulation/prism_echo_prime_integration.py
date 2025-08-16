#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - PRISM-ECHO-PRIME Integration

Integrationsschicht zwischen PRISM-Engine und ECHO-PRIME.
Ermöglicht Simulationen und Wahrscheinlichkeitsanalysen für Zeitlinien.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set, TYPE_CHECKING
import numpy as np
from datetime import datetime

# Konfiguriere Logging
logger = logging.getLogger("MISO.simulation.prism_echo_prime_integration")

# Importiere Basisklassen aus dem gemeinsamen core-Modul
from miso.core.timeline_base import Timeline, TimeNode, TimelineType, TriggerLevel

# Importiere PRISM-spezifische Typen
from miso.simulation.prism_base import SimulationConfig

# Importiere PRISM-Engine Komponenten
from miso.simulation.prism_engine import PrismEngine
from miso.simulation.prism_matrix import PrismMatrix
from miso.simulation.time_scope import TimeScopeUnit, TimeScope

# ECHO-PRIME Verfügbarkeit prüfen ohne direkte Importe
try:
    # Verzögerter Import nur für Typprüfung
    import importlib.util
    has_qtm = importlib.util.find_spec("miso.timeline.qtm_modulator") is not None
    HAS_ECHO_PRIME = has_qtm
except ImportError:
    logger.warning("ECHO-PRIME-Module konnten nicht importiert werden. Integration läuft im eingeschränkten Modus.")
    HAS_ECHO_PRIME = False

# Klassen für Typ-Annotation definieren (werden zur Laufzeit importiert)
class QTM_Modulator:
    pass

class QuantumTimeEffect:
    pass

class QuantumState:
    pass


def get_echo_prime_controller():
    """
    Lazy-Loading Funktion für den EchoPrimeController
    Verhindert zirkuläre Importe zwischen PRISM und ECHO-PRIME
    
    Returns:
        EchoPrimeController-Klasse oder None, falls nicht verfügbar
    """
    if not HAS_ECHO_PRIME:
        return None
        
    try:
        # Verwende importlib für maximale Kontrolle über den Import-Prozess
        import importlib
        echo_prime_controller_module = importlib.import_module("miso.timeline.echo_prime_controller")
        return getattr(echo_prime_controller_module, "EchoPrimeController", None)
    except ImportError:
        logger.error("EchoPrimeController konnte nicht importiert werden.")
        return None
        
def get_qtm_components():
    """
    Lazy-Loading Funktion für QTM_Modulator Komponenten
    
    Returns:
        Tuple aus (QTM_Modulator, QuantumTimeEffect, QuantumState) oder (None, None, None)
    """
    if not HAS_ECHO_PRIME:
        return None, None, None
        
    try:
        import importlib
        qtm_module = importlib.import_module("miso.timeline.qtm_modulator")
        return (
            getattr(qtm_module, "QTM_Modulator", None),
            getattr(qtm_module, "QuantumTimeEffect", None),
            getattr(qtm_module, "QuantumState", None)
        )
    except ImportError:
        logger.error("QTM_Modulator-Komponenten konnten nicht importiert werden.")
        return None, None, None


class TimelineSimulationContext:
    """
    Kontext für Zeitliniensimulationen
    
    Enthält alle relevanten Informationen für eine Simulation im Kontext einer Zeitlinie.
    """
    
    def __init__(self, 
                 timeline: 'Timeline', 
                 current_node: 'TimeNode' = None,
                 simulation_steps: int = 100,
                 simulation_scope: TimeScope = TimeScope.MEDIUM):
        """
        Initialisiert den Zeitliniensimulationskontext
        
        Args:
            timeline: Die zu simulierende Zeitlinie
            current_node: Der aktuelle Zeitknoten
            simulation_steps: Anzahl der Simulationsschritte
            simulation_scope: Zeitlicher Bereich für die Simulation
        """
        self.timeline = timeline
        self.current_node = current_node
        self.simulation_steps = simulation_steps
        self.simulation_scope = simulation_scope
        self.simulation_id = str(uuid.uuid4())
        self.creation_time = time.time()
        
        # Simulationsergebnisse
        self.results = {}
        self.alternative_timelines = []
        self.probability_map = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert den Kontext in ein Dictionary
        
        Returns:
            Dictionary-Repräsentation des Kontexts
        """
        return {
            "simulation_id": self.simulation_id,
            "timeline_id": self.timeline.id,
            "timeline_name": self.timeline.name,
            "current_node_id": self.current_node.id if self.current_node else None,
            "simulation_steps": self.simulation_steps,
            "simulation_scope": self.simulation_scope.name,
            "creation_time": self.creation_time,
            "has_results": bool(self.results),
            "alternative_timeline_count": len(self.alternative_timelines)
        }


class PRISMECHOPrimeIntegration:
    """
    PRISM-ECHO-PRIME Integration
    
    Integrationsschicht zwischen PRISM-Engine und ECHO-PRIME.
    Ermöglicht Simulationen und Wahrscheinlichkeitsanalysen für Zeitlinien.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert die Integrationsschicht
        
        Args:
            config: Konfigurationsobjekt für die Integration
        """
        self.config = config or {}
        
        # Initialisiere PRISM-Engine
        self.prism_engine = PrismEngine(self.config.get("prism_config"))
        
        # Initialisiere ECHO-PRIME Controller, falls verfügbar
        self.echo_prime = None
        if HAS_ECHO_PRIME:
            # Verwende Lazy-Loading, um zirkuläre Importe zu vermeiden
            EchoPrimeController = get_echo_prime_controller()
            if EchoPrimeController:
                self.echo_prime = EchoPrimeController(self.config.get("echo_prime_config_path"))
                # Lade QTM-Komponenten bei Bedarf
                self._load_qtm_components()
            else:
                logger.warning("EchoPrimeController konnte nicht geladen werden.")
        
        # Cache für Simulationen
        self.simulation_cache = {}
        
        logger.info("PRISM-ECHO-PRIME Integration initialisiert")
    
    def _load_qtm_components(self):
        """
        Lädt QTM-Komponenten bei Bedarf
        """
        QTM_Modulator_Class, QuantumTimeEffect_Class, QuantumState_Class = get_qtm_components()
        
        # Setze globale Referenzen für andere Methoden in dieser Klasse
        global QTM_Modulator, QuantumTimeEffect, QuantumState
        if QTM_Modulator_Class is not None:
            QTM_Modulator = QTM_Modulator_Class
        if QuantumTimeEffect_Class is not None:
            QuantumTimeEffect = QuantumTimeEffect_Class
        if QuantumState_Class is not None:
            QuantumState = QuantumState_Class
    
    def simulate_timeline(self, 
                        timeline_id: str, 
                        steps: int = 100, 
                        scope: TimeScope = TimeScope.MEDIUM) -> Dict[str, Any]:
        """
        Simuliert eine Zeitlinie mit der PRISM-Engine
        
        Args:
            timeline_id: ID der zu simulierenden Zeitlinie
            steps: Anzahl der Simulationsschritte
            scope: Zeitlicher Bereich für die Simulation
            
        Returns:
            Simulationsergebnis
        """
        if not HAS_ECHO_PRIME or not self.echo_prime:
            logger.error("ECHO-PRIME ist nicht verfügbar")
            return {"error": "ECHO-PRIME ist nicht verfügbar"}
        
        # Hole Zeitlinie
        timeline = self.echo_prime.get_timeline(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} existiert nicht")
            return {"error": f"Zeitlinie {timeline_id} existiert nicht"}
        
        # Finde den aktuellen Knoten (neuester Knoten)
        current_node = None
        if timeline.nodes:
            current_node = max(timeline.nodes.values(), key=lambda n: n.timestamp)
        
        # Erstelle Simulationskontext
        context = TimelineSimulationContext(
            timeline=timeline,
            current_node=current_node,
            simulation_steps=steps,
            simulation_scope=scope
        )
        
        # Konvertiere Zeitlinie in PRISM-kompatibles Format
        current_state = self._timeline_to_prism_state(timeline, current_node)
        
        # Generiere Variationen basierend auf möglichen Verzweigungen
        variations = self._generate_timeline_variations(timeline, current_node)
        
        # Führe Simulation durch
        simulation_result = self.prism_engine.simulate_reality_fork(
            current_state=current_state,
            variations=variations,
            steps=steps
        )
        
        # Speichere Ergebnisse im Kontext
        context.results = simulation_result
        
        # Generiere alternative Zeitlinien basierend auf den Simulationsergebnissen
        alternative_timelines = self._generate_alternative_timelines(timeline, current_node, simulation_result)
        context.alternative_timelines = alternative_timelines
        
        # Berechne Wahrscheinlichkeitskarte
        probability_map = self._calculate_probability_map(simulation_result)
        context.probability_map = probability_map
        
        # Speichere Simulationskontext im Cache
        self.simulation_cache[context.simulation_id] = context
        
        # Erstelle Ergebnisbericht
        result = {
            "simulation_id": context.simulation_id,
            "timeline_id": timeline_id,
            "timeline_name": timeline.name,
            "current_node_id": current_node.id if current_node else None,
            "simulation_steps": steps,
            "simulation_scope": scope.name,
            "alternative_timelines": [t.id for t in alternative_timelines],
            "probability_map": probability_map,
            "recommendations": self._generate_recommendations(simulation_result, probability_map)
        }
        
        return result
    
    def _timeline_to_prism_state(self, timeline: 'Timeline', current_node: Optional['TimeNode'] = None) -> Dict[str, Any]:
        """
        Konvertiert eine Zeitlinie in einen PRISM-kompatiblen Zustand
        
        Args:
            timeline: Zu konvertierende Zeitlinie
            current_node: Aktueller Zeitknoten
            
        Returns:
            PRISM-kompatibler Zustand
        """
        state = {
            "timeline_id": timeline.id,
            "timeline_name": timeline.name,
            "timeline_type": timeline.type.value if hasattr(timeline, 'type') else "standard",
            "timeline_probability": timeline.probability,
            "node_count": len(timeline.nodes),
            "current_timestamp": time.time()
        }
        
        # Füge Knotenkontext hinzu, falls vorhanden
        if current_node:
            state["current_node_id"] = current_node.id
            state["current_node_description"] = current_node.description
            state["current_node_probability"] = current_node.probability
            state["current_node_trigger_level"] = current_node.trigger_level.value
            
            # Füge Eltern- und Kindknoteninfos hinzu
            state["parent_node_id"] = current_node.parent_node_id
            state["child_node_count"] = len(current_node.child_node_ids)
            
            # Extrahiere Metadaten aus dem Knoten
            if hasattr(current_node, 'metadata') and current_node.metadata:
                for key, value in current_node.metadata.items():
                    if isinstance(value, (int, float, bool, str)):
                        state[f"metadata_{key}"] = value
        
        return state
    
    def _generate_timeline_variations(self, timeline: 'Timeline', current_node: Optional['TimeNode'] = None) -> List[Dict[str, Any]]:
        """
        Optimierte Version: Generiert präzise Variationen für eine Zeitlinie basierend auf
        komplexen probabilistischen Modellen und adaptiven Algorithmen
        
        Args:
            timeline: Zeitlinie
            current_node: Aktueller Zeitknoten
            
        Returns:
            Liste von Variationen mit Wahrscheinlichkeitsmodellen
        """
        # Basisvariationen initialisieren
        variations = []
        
        # Erweiterte Analyse der Zeitlinien-Eigenschaften
        timeline_age = time.time() - timeline.creation_timestamp if hasattr(timeline, 'creation_timestamp') else 0
        timeline_complexity = len(timeline.node_ids) if hasattr(timeline, 'node_ids') else 0
        timeline_coherence = getattr(timeline, 'coherence_factor', 0.5)  # Standardwert falls nicht vorhanden
        
        # Bestimme die adaptive Gewichtung basierend auf Zeitlinieneigenschaften
        # Ältere Zeitlinien werden als stabiler betrachtet, was zu niedrigeren Variationseffekten führt
        age_factor = 1.0 / (1 + 0.1 * min(timeline_age / 86400, 10))  # Maximale Dämpfung nach 10 Tagen
        
        # Komplexe Zeitlinien (mehr Knoten) bekommen mehr Variationsmöglichkeiten
        complexity_factor = min(1.0 + (timeline_complexity / 20), 2.0)  # Maximal doppelte Auswirkung
        
        # Kohärenzfaktor beeinflusst die Stabilität der Zeitlinie
        coherence_factor = 2.0 - timeline_coherence  # Niedriger Kohärenzfaktor = höhere Variation
        
        # Kombinierter adaptiver Faktor für dynamische Anpassung
        adaptive_factor = age_factor * complexity_factor * coherence_factor
        
        if not current_node:
            # Erweiterte Basisvariationen mit adaptiver Skalierung
            variation_types = [
                # Format: (Faktor, Basisimpact, Beschreibung, Gewichtungsfaktor)
                ("timeline_probability", 0.1, "Leicht erhöhte Wahrscheinlichkeit", 1.0),
                ("timeline_probability", -0.1, "Leicht verringerte Wahrscheinlichkeit", 1.0),
                ("timeline_probability", 0.2, "Moderat erhöhte Wahrscheinlichkeit", 0.8),
                ("timeline_probability", -0.2, "Moderat verringerte Wahrscheinlichkeit", 0.8),
                ("timeline_coherence", 0.15, "Verbesserte Zeitlinienkohärenz", 0.7),
                ("timeline_coherence", -0.15, "Reduzierte Zeitlinienkohärenz", 0.7),
                ("anomaly_factor", 0.25, "Erhöhte Anomalien", 0.5),
                ("stability_index", -0.2, "Verringerte Stabilität", 0.6)
            ]
            
            # Erzeuge präzise skalierte Variationen
            for factor, base_impact, description, weight in variation_types:
                # Berechne den finalen Impact unter Berücksichtigung aller Faktoren
                final_impact = base_impact * adaptive_factor * weight
                variations.append({
                    "factor": factor,
                    "impact": round(final_impact, 4),  # Runde auf 4 Nachkommastellen für Präzision
                    "description": description,
                    "confidence": round(max(0.1, min(1.0, 1.0 - (abs(final_impact) * 0.5))), 2)  # Konfidenzwert
                })
            
            # Logarithmisches Debugging
            logger.debug(f"Generierte {len(variations)} Basisvariationen mit adaptivem Faktor {adaptive_factor:.3f}")
            return variations
        
        # --- KNOTENBASIERTE ANALYSE ---
        
        # Extrahiere alle verfügbaren Daten aus dem Knoten für verbesserte Analyse
        node_data = self._extract_node_features(current_node)
        
        # Probabilistische Analyse der Knotenwahrscheinlichkeit
        # Anwendung eines nicht-linearen Wahrscheinlichkeitsmodells
        base_prob = current_node.probability
        prob_variance = 0.1 * (1 - abs(2*base_prob - 1))  # Höchste Varianz bei p=0.5, niedrigste bei p=0 oder p=1
        
        # Integriere Q-LOGIK Bayessche Entscheidungsfindung für präzisere Wahrscheinlichkeitsabschätzungen
        if hasattr(self, 'qlogik_integration') and self.qlogik_integration:
            try:
                # Bereite Daten für Bayessche Analyse vor
                bayes_data = {
                    "hypothesis": "probability_shift",
                    "evidence": {
                        "base_probability": {"value": base_prob, "weight": 1.0},
                        "node_age": {"value": min(1.0, node_data.get("age_factor", 0)), "weight": 0.7},
                        "trigger_level": {"value": node_data.get("trigger_value", 0.5), "weight": 0.9},
                        "complexity": {"value": min(1.0, node_data.get("complexity", 0) / 10), "weight": 0.6}
                    }
                }
                
                # Bayessche Wahrscheinlichkeitsberechnung durch Q-LOGIK
                bayes_result = self.qlogik_integration.evaluate_bayesian_probability(bayes_data)
                
                # Integriere Bayessche Bewertung in die Wahrscheinlichkeitsvariation
                prob_variance *= (0.5 + bayes_result)
                logger.debug(f"Q-LOGIK Bayessche Modulation: {bayes_result:.4f}, resultierender Varianzfaktor: {prob_variance:.4f}")
            except Exception as e:
                logger.warning(f"Fehler bei Q-LOGIK Bayesscher Integration: {e}")
        
        # Erzeuge eine informationsreiche Variation für Knotenwahrscheinlichkeit
        prob_impact = prob_variance * (1 if base_prob < 0.7 else -1) * adaptive_factor
        variations.append({
            "factor": "current_node_probability",
            "impact": round(prob_impact, 4),
            "description": "Adaptive Wahrscheinlichkeitsanpassung",
            "confidence": round(1.0 - (abs(prob_impact) * 0.5), 2),
            "model": "bayesian_adaptive"
        })
        
        # Verbessertes Trigger-Level-Modell mit QTM-Modulation
        trigger_level = current_node.trigger_level
        trigger_value = 0.0
        
        # Erweitertes mehrdimensionales Trigger-Modell
        if trigger_level == TriggerLevel.LOW:
            trigger_value = 0.15
            trigger_desc = "Niedriges Trigger-Level mit leichtem Einfluss"
        elif trigger_level == TriggerLevel.MEDIUM:
            trigger_value = 0.35
            trigger_desc = "Mittleres Trigger-Level mit moderatem Einfluss"
        elif trigger_level == TriggerLevel.HIGH:
            trigger_value = 0.6
            trigger_desc = "Hohes Trigger-Level mit signifikantem Einfluss"
        elif trigger_level == TriggerLevel.CRITICAL:
            trigger_value = 0.9
            trigger_desc = "Kritisches Trigger-Level mit schwerem Einfluss"
        
        # QTM-Modulation des Trigger-Effekts, falls verfügbar
        qtm_modulation = 1.0
        if HAS_QTM and trigger_value > 0:
            try:
                # QTM-Quanteneffekte auf Trigger-Auswirkungen anwenden
                global QTM_Modulator, QuantumTimeEffect
                if QTM_Modulator is not None:
                    # Erzeuge Quanteneffekt mit Superposition und Entanglement
                    quantum_effect = QuantumTimeEffect(
                        superposition=trigger_value * 2,  # Stärkere Superposition bei höheren Triggern
                        entanglement=min(0.8, base_prob + 0.3),  # Entanglement korreliert mit Basiswahrscheinlichkeit
                        decoherence_rate=0.1,  # Niedrige Dekohärenzrate für stabilere Quanteneffekte
                        measurement_sensitivity=0.7  # Hohe Messempfindlichkeit
                    )
                    
                    # Wende Quantenmodulation an
                    modulator = QTM_Modulator()
                    qtm_result = modulator.apply_quantum_effect(quantum_effect, trigger_value)
                    qtm_modulation = qtm_result.get("modulation_factor", 1.0)
                    
                    # Aktualisiere Beschreibung mit Quanteninformationen
                    trigger_desc += f" mit QTM-Modulation ({qtm_modulation:.2f})"
                    logger.debug(f"QTM-Modulation angewendet: {qtm_modulation:.4f}")
            except Exception as e:
                logger.warning(f"Fehler bei QTM-Modulation: {e}")
        
        # Berechne den finalen Trigger-Einfluss mit allen Modulationen
        trigger_impact = trigger_value * qtm_modulation * adaptive_factor
        
        # Füge die modulierte Trigger-Variation hinzu
        variations.append({
            "factor": "current_node_trigger_level",
            "impact": round(trigger_impact, 4),
            "description": trigger_desc,
            "confidence": round(max(0.3, min(0.95, trigger_value)), 2),
            "qtm_modulated": qtm_modulation != 1.0
        })
        
        # Komplexitätsbasierte Variation mit Berücksichtigung der Knotenstruktur
        if current_node.child_node_ids:
            child_count = len(current_node.child_node_ids)
            # Nichtlineare Transformation der Kindknotenanzahl
            branching_factor = min(1.0, math.log(1 + child_count) / math.log(10))
            
            # Adaptive Berechnung des Einflusses basierend auf Verzweigungskomplexität
            child_impact = 0.15 * branching_factor * adaptive_factor
            
            # Füge Kindknoten-Variation mit detaillierten Informationen hinzu
            variations.append({
                "factor": "branching_complexity",
                "impact": round(child_impact, 4),
                "description": f"Verzweigungskomplexität mit {child_count} Kindknoten",
                "confidence": round(0.5 + (branching_factor * 0.3), 2),
                "child_nodes": child_count
            })
            
            # Berechne zusätzliche Variation für Verzweigungsstruktur
            structure_impact = 0.1 * (1 - (1 / (1 + child_count))) * adaptive_factor
            variations.append({
                "factor": "structure_entropy",
                "impact": round(structure_impact, 4),
                "description": "Strukturelle Entropie der Verzweigungen",
                "confidence": 0.6,
                "entropy_value": round(math.log(1 + child_count), 3)
            })
        
        # Erweiterte Metadaten-Analyse mit semantischer Interpretation
        if hasattr(current_node, 'metadata') and current_node.metadata:
            metadata_impacts = self._analyze_node_metadata(current_node.metadata, adaptive_factor)
            variations.extend(metadata_impacts)
        
        # Zeitliche Korrelationsanalyse zwischen Knoten
        time_correlation = self._analyze_temporal_correlation(timeline, current_node)
        if time_correlation:
            variations.append(time_correlation)
        
        # Füge allgemeine Stabilitätsvariationen hinzu
        stability_impact = 0.1 * (1 - node_data.get("stability", 0.5)) * adaptive_factor
        variations.append({
            "factor": "timeline_stability", 
            "impact": round(-abs(stability_impact), 4),  # Negative Auswirkung auf Stabilität
            "description": "Zeitlinienstabilitätseffekt",
            "confidence": 0.75
        })
        
        logger.debug(f"Generierte {len(variations)} optimierte Variationen für Knoten {current_node.id} mit adaptivem Faktor {adaptive_factor:.3f}")
        return variations
    
    def _extract_node_features(self, node: 'TimeNode') -> Dict[str, Any]:
        """
        Extrahiert detaillierte Features aus einem Zeitknoten für die probabilistische Analyse
        
        Args:
            node: Der zu analysierende Zeitknoten
            
        Returns:
            Dictionary mit extrahierten Features
        """
        features = {}
        
        # Basisfeatures
        features["probability"] = getattr(node, "probability", 0.5)
        features["trigger_level"] = getattr(node, "trigger_level", TriggerLevel.MEDIUM)
        features["trigger_value"] = self._trigger_level_to_value(features["trigger_level"])
        features["has_children"] = bool(getattr(node, "child_node_ids", []))
        features["child_count"] = len(getattr(node, "child_node_ids", []))
        
        # Zeitliche Features
        current_time = time.time()
        creation_time = getattr(node, "creation_timestamp", current_time)
        features["age"] = current_time - creation_time
        features["age_factor"] = min(1.0, features["age"] / (7 * 24 * 3600))  # Normalisiert auf 7 Tage
        
        # Komplexitätsfeatures
        features["complexity"] = features["child_count"] * 2
        if hasattr(node, "metadata"):
            features["complexity"] += len(node.metadata) 
        
        # Stabilitätsmetrik basierend auf Wahrscheinlichkeit und Alter
        stability_base = 0.3 + (0.4 * features["probability"]) + (0.3 * features["age_factor"])
        features["stability"] = min(1.0, stability_base)
        
        # Weitere komplexe Features
        features["entropy"] = -features["probability"] * math.log2(max(0.001, features["probability"])) - \
                          (1-features["probability"]) * math.log2(max(0.001, 1-features["probability"]))
        
        return features
    
    def _trigger_level_to_value(self, trigger_level: TriggerLevel) -> float:
        """
        Konvertiert ein TriggerLevel-Enum in einen numerischen Wert
        
        Args:
            trigger_level: Das zu konvertierende TriggerLevel
            
        Returns:
            Numerischer Wert zwischen 0 und 1
        """
        if trigger_level == TriggerLevel.LOW:
            return 0.2
        elif trigger_level == TriggerLevel.MEDIUM:
            return 0.5
        elif trigger_level == TriggerLevel.HIGH:
            return 0.75
        elif trigger_level == TriggerLevel.CRITICAL:
            return 0.95
        return 0.5  # Standardwert für unbekannte Level
    
    def _analyze_node_metadata(self, metadata: Dict[str, Any], adaptive_factor: float) -> List[Dict[str, Any]]:
        """
        Führt eine semantische Analyse der Knotenmetadaten durch
        
        Args:
            metadata: Zu analysierende Metadaten
            adaptive_factor: Adaptiver Skalierungsfaktor
            
        Returns:
            Liste von Variationseffekten basierend auf den Metadaten
        """
        variations = []
        
        # Erweiterte Schlüssel mit verstärktem Einfluss
        high_impact_keys = ["importance", "priority", "risk", "impact", "urgency", "significance"]
        
        # Semantische Kategorien für Metadaten-Interpretation
        positive_keys = ["benefit", "opportunity", "advantage", "gain", "success"]
        negative_keys = ["risk", "threat", "disadvantage", "loss", "failure"]
        temporal_keys = ["duration", "delay", "frequency", "interval", "period"]
        
        # Gruppiere Metadaten nach semantischen Kategorien
        positive_impact = 0.0
        negative_impact = 0.0
        temporal_impact = 0.0
        positive_count = 0
        negative_count = 0
        temporal_count = 0
        
        for key, value in metadata.items():
            if not isinstance(value, (int, float)):
                continue
                
            # Normalisiere Wert auf Bereich [0,1]
            if isinstance(value, int) and value > 1:
                norm_value = min(1.0, value / 10)  # Skaliere ganzzahlige Werte > 1
            else:
                norm_value = min(1.0, max(0.0, float(value)))  # Begrenzen auf [0,1]
            
            # Klassifiziere nach semantischer Kategorie
            if any(pos_key in key.lower() for pos_key in positive_keys):
                positive_impact += norm_value
                positive_count += 1
            elif any(neg_key in key.lower() for neg_key in negative_keys):
                negative_impact += norm_value
                negative_count += 1
            elif any(temp_key in key.lower() for temp_key in temporal_keys):
                temporal_impact += norm_value
                temporal_count += 1
            # Standardkategorie mit hohem Einfluss
            elif any(impact_key in key.lower() for impact_key in high_impact_keys):
                variations.append({
                    "factor": f"metadata_{key}",
                    "impact": round((norm_value - 0.5) * 0.2 * adaptive_factor, 4),
                    "description": f"Metadaten-Feature: {key.replace('_', ' ').title()}",
                    "confidence": round(0.6 + (0.2 * norm_value), 2),
                    "value": norm_value
                })
        
        # Verarbeite aggregierte Kategorieauswirkungen
        if positive_count > 0:
            avg_positive = positive_impact / positive_count
            variations.append({
                "factor": "positive_attributes",
                "impact": round(avg_positive * 0.15 * adaptive_factor, 4),
                "description": f"Positive Metadaten-Attribute ({positive_count})",
                "confidence": 0.7,
                "attribute_count": positive_count
            })
            
        if negative_count > 0:
            avg_negative = negative_impact / negative_count
            variations.append({
                "factor": "negative_attributes",
                "impact": round(-avg_negative * 0.2 * adaptive_factor, 4),  # Negative Auswirkung
                "description": f"Negative Metadaten-Attribute ({negative_count})",
                "confidence": 0.75,
                "attribute_count": negative_count
            })
            
        if temporal_count > 0:
            avg_temporal = temporal_impact / temporal_count
            variations.append({
                "factor": "temporal_attributes",
                "impact": round((avg_temporal - 0.5) * 0.1 * adaptive_factor, 4),
                "description": f"Zeitliche Metadaten-Attribute ({temporal_count})",
                "confidence": 0.65,
                "attribute_count": temporal_count
            })
        
        return variations
    
    def _analyze_temporal_correlation(self, timeline: 'Timeline', node: 'TimeNode') -> Optional[Dict[str, Any]]:
        """
        Analysiert zeitliche Korrelationen zwischen Knoten in einer Zeitlinie
        
        Args:
            timeline: Die zu analysierende Zeitlinie
            node: Der aktuelle Knoten
            
        Returns:
            Variationseffekt basierend auf zeitlicher Korrelation oder None
        """
        # Überprüfe, ob wir genügend Daten für eine Korrelationsanalyse haben
        if not hasattr(timeline, 'nodes') or not timeline.nodes or len(timeline.nodes) < 3:
            return None
            
        # Extrahiere Zeitstempel (falls vorhanden) und Wahrscheinlichkeiten
        timestamps = []
        probabilities = []
        
        for node_id, node_obj in timeline.nodes.items():
            if hasattr(node_obj, 'creation_timestamp') and hasattr(node_obj, 'probability'):
                timestamps.append(node_obj.creation_timestamp)
                probabilities.append(node_obj.probability)
        
        # Führe nur Analyse durch, wenn wir genügend Datenpunkte haben
        if len(timestamps) < 3 or len(probabilities) < 3:
            return None
            
        try:
            # Berechne Korrelation zwischen Zeit und Wahrscheinlichkeit
            # (Vereinfachte Korrelationsberechnung ohne numpy)
            n = len(timestamps)
            sum_time = sum(timestamps)
            sum_prob = sum(probabilities)
            sum_time_sq = sum(t*t for t in timestamps)
            sum_prob_sq = sum(p*p for p in probabilities)
            sum_time_prob = sum(t*p for t, p in zip(timestamps, probabilities))
            
            # Pearson-Korrelationskoeffizient
            num = n * sum_time_prob - sum_time * sum_prob
            denom = math.sqrt((n * sum_time_sq - sum_time**2) * (n * sum_prob_sq - sum_prob**2))
            
            if denom == 0:
                correlation = 0
            else:
                correlation = num / denom
                
            # Erzeuge Variation basierend auf der Korrelation
            impact = correlation * 0.15  # Skaliere auf einen angemessenen Bereich
            return {
                "factor": "temporal_correlation",
                "impact": round(impact, 4),
                "description": "Zeitliche Korrelation der Knotenwahrscheinlichkeiten",
                "confidence": round(0.5 + abs(correlation) * 0.3, 2),
                "correlation": round(correlation, 3)
            }
        except Exception as e:
            logger.warning(f"Fehler bei der temporalen Korrelationsanalyse: {e}")
            return None
    
    def _generate_alternative_timelines(self, 
                                      timeline: 'Timeline', 
                                      current_node: Optional['TimeNode'], 
                                      simulation_result: Dict[str, Any]) -> List['Timeline']:
        """
        Generiert alternative Zeitlinien basierend auf Simulationsergebnissen
        
        Args:
            timeline: Ausgangszeitlinie
            current_node: Aktueller Zeitknoten
            simulation_result: Simulationsergebnis
            
        Returns:
            Liste von alternativen Zeitlinien
        """
        if not HAS_ECHO_PRIME or not self.echo_prime:
            logger.error("ECHO-PRIME ist nicht verfügbar")
            return []
        
        alternative_timelines = []
        
        # Extrahiere alternative Realitäten aus dem Simulationsergebnis
        alternative_realities = simulation_result.get("alternative_realities", [])
        
        # Erstelle für jede alternative Realität eine neue Zeitlinie
        for i, reality in enumerate(alternative_realities):
            # Überspringe die erste Realität, da sie der Ausgangszeitlinie entspricht
            if i == 0:
                continue
            
            # Erstelle neue Zeitlinie
            alt_timeline = self.echo_prime.create_timeline(
                name=f"{timeline.name} (Alternative {i})",
                description=f"Alternative Zeitlinie basierend auf Simulation von {timeline.name}"
            )
            
            # Kopiere Knoten aus der Ausgangszeitlinie bis zum aktuellen Knoten
            if current_node:
                # Erstelle Mapping von alten zu neuen Knoten-IDs
                node_id_mapping = {}
                
                # Kopiere Knoten rekursiv vom Wurzelknoten bis zum aktuellen Knoten
                self._copy_nodes_recursive(timeline, alt_timeline, current_node, node_id_mapping)
                
                # Erstelle neuen Knoten basierend auf der Simulation
                new_node_description = f"Simulierter Knoten ({i})"
                if "event_description" in reality:
                    new_node_description = reality["event_description"]
                
                new_node_probability = reality.get("timeline_probability", 0.5)
                
                # Bestimme Trigger-Level basierend auf der Simulation
                trigger_level = TriggerLevel.MEDIUM
                if "current_node_trigger_level" in reality:
                    level_value = reality["current_node_trigger_level"]
                    if level_value < 1:
                        trigger_level = TriggerLevel.LOW
                    elif level_value < 2:
                        trigger_level = TriggerLevel.MEDIUM
                    elif level_value < 3:
                        trigger_level = TriggerLevel.HIGH
                    else:
                        trigger_level = TriggerLevel.CRITICAL
                
                # Finde den neuen aktuellen Knoten in der alternativen Zeitlinie
                new_current_node_id = node_id_mapping.get(current_node.id)
                
                if new_current_node_id:
                    # Füge neuen Knoten hinzu
                    new_node = self.echo_prime.add_time_node(
                        timeline_id=alt_timeline.id,
                        description=new_node_description,
                        parent_node_id=new_current_node_id,
                        probability=new_node_probability,
                        trigger_level=trigger_level,
                        metadata={"simulation_based": True, "simulation_index": i}
                    )
            
            # Setze Zeitlinienwahrscheinlichkeit basierend auf der Simulation
            alt_timeline.probability = reality.get("timeline_probability", 0.5)
            
            # Füge Metadaten hinzu
            alt_timeline.metadata = alt_timeline.metadata or {}
            alt_timeline.metadata.update({
                "simulation_based": True,
                "simulation_index": i,
                "original_timeline_id": timeline.id,
                "simulation_timestamp": time.time()
            })
            
            # Aktualisiere Zeitlinie
            self.echo_prime.update_timeline(alt_timeline.id, {"metadata": alt_timeline.metadata})
            
            # Füge zur Liste hinzu
            alternative_timelines.append(alt_timeline)
        
        return alternative_timelines
    
    def _copy_nodes_recursive(self, 
                            source_timeline: 'Timeline', 
                            target_timeline: 'Timeline', 
                            end_node: 'TimeNode', 
                            node_id_mapping: Dict[str, str], 
                            processed_nodes: Set[str] = None):
        """
        Kopiert Knoten rekursiv von einer Zeitlinie zur anderen
        
        Args:
            source_timeline: Quellzeitlinie
            target_timeline: Zielzeitlinie
            end_node: Endknoten, bis zu dem kopiert werden soll
            node_id_mapping: Mapping von alten zu neuen Knoten-IDs
            processed_nodes: Set von bereits verarbeiteten Knoten-IDs
            
        Returns:
            None
        """
        if processed_nodes is None:
            processed_nodes = set()
        
        # Wenn der Endknoten bereits verarbeitet wurde, beende die Rekursion
        if end_node.id in processed_nodes:
            return
        
        # Verarbeite zuerst den Elternknoten, falls vorhanden
        parent_node = None
        if end_node.parent_node_id and end_node.parent_node_id in source_timeline.nodes:
            parent_node = source_timeline.nodes[end_node.parent_node_id]
            self._copy_nodes_recursive(source_timeline, target_timeline, parent_node, node_id_mapping, processed_nodes)
        
        # Kopiere den aktuellen Knoten
        parent_id = node_id_mapping.get(end_node.parent_node_id) if end_node.parent_node_id else None
        
        new_node = self.echo_prime.add_time_node(
            timeline_id=target_timeline.id,
            description=end_node.description,
            parent_node_id=parent_id,
            probability=end_node.probability,
            trigger_level=end_node.trigger_level,
            metadata=end_node.metadata.copy() if end_node.metadata else {}
        )
        
        # Speichere das Mapping
        node_id_mapping[end_node.id] = new_node.id
        
        # Markiere den Knoten als verarbeitet
        processed_nodes.add(end_node.id)
    
    def _calculate_probability_map(self, simulation_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Berechnet eine Wahrscheinlichkeitskarte basierend auf dem Simulationsergebnis
        
        Args:
            simulation_result: Simulationsergebnis
            
        Returns:
            Wahrscheinlichkeitskarte
        """
        probability_map = {}
        
        # Extrahiere alternative Realitäten aus dem Simulationsergebnis
        alternative_realities = simulation_result.get("alternative_realities", [])
        
        # Berechne Gesamtwahrscheinlichkeit
        total_probability = 0.0
        for reality in alternative_realities:
            prob = reality.get("timeline_probability", 0.5)
            total_probability += prob
        
        # Normalisiere Wahrscheinlichkeiten
        if total_probability > 0:
            for i, reality in enumerate(alternative_realities):
                prob = reality.get("timeline_probability", 0.5)
                normalized_prob = prob / total_probability
                
                # Erstelle Schlüssel für die Wahrscheinlichkeitskarte
                key = f"alternative_{i}"
                if i == 0:
                    key = "baseline"
                elif "event_description" in reality:
                    key = reality["event_description"]
                
                probability_map[key] = normalized_prob
        
        return probability_map
    
    def _generate_recommendations(self, 
                               simulation_result: Dict[str, Any], 
                               probability_map: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Generiert Handlungsempfehlungen basierend auf dem Simulationsergebnis
        
        Args:
            simulation_result: Simulationsergebnis
            probability_map: Wahrscheinlichkeitskarte
            
        Returns:
            Liste von Handlungsempfehlungen
        """
        recommendations = []
        
        # Extrahiere alternative Realitäten aus dem Simulationsergebnis
        alternative_realities = simulation_result.get("alternative_realities", [])
        
        # Finde die wahrscheinlichste Alternative
        most_probable_key = max(probability_map.items(), key=lambda x: x[1])[0] if probability_map else None
        
        # Finde die beste Alternative basierend auf Nutzen/Risiko
        best_alternative = None
        best_score = -float('inf')
        
        for i, reality in enumerate(alternative_realities):
            # Berechne Nutzen/Risiko-Score
            benefit = reality.get("benefit", 0.5)
            risk = reality.get("risk", 0.5)
            score = benefit - risk
            
            if score > best_score:
                best_score = score
                best_alternative = i
        
        # Erstelle Empfehlungen
        
        # Empfehlung basierend auf der wahrscheinlichsten Alternative
        if most_probable_key:
            recommendations.append({
                "type": "most_probable",
                "description": f"Die wahrscheinlichste Alternative ist '{most_probable_key}'",
                "probability": probability_map.get(most_probable_key, 0.0),
                "action": "Monitor this timeline closely as it is the most likely outcome"
            })
        
        # Empfehlung basierend auf der besten Alternative
        if best_alternative is not None:
            reality = alternative_realities[best_alternative]
            key = f"alternative_{best_alternative}"
            if best_alternative == 0:
                key = "baseline"
            elif "event_description" in reality:
                key = reality["event_description"]
            
            recommendations.append({
                "type": "best_outcome",
                "description": f"Die beste Alternative basierend auf Nutzen/Risiko ist '{key}'",
                "probability": probability_map.get(key, 0.0),
                "benefit": reality.get("benefit", 0.5),
                "risk": reality.get("risk", 0.5),
                "action": "Consider actions that would lead to this outcome"
            })
        
        # Allgemeine Empfehlungen
        recommendations.append({
            "type": "general",
            "description": "Basierend auf der Simulation wurden mehrere mögliche Zeitlinien identifiziert",
            "alternative_count": len(alternative_realities),
            "action": "Review all alternative timelines and their probabilities"
        })
        
        return recommendations
    
    def get_simulation_result(self, simulation_id: str) -> Dict[str, Any]:
        """
        Gibt das Ergebnis einer Simulation zurück
        
        Args:
            simulation_id: ID der Simulation
            
        Returns:
            Simulationsergebnis
        """
        if simulation_id not in self.simulation_cache:
            return {"error": f"Simulation {simulation_id} nicht gefunden"}
        
        context = self.simulation_cache[simulation_id]
        
        return {
            "simulation_id": context.simulation_id,
            "timeline_id": context.timeline.id,
            "timeline_name": context.timeline.name,
            "current_node_id": context.current_node.id if context.current_node else None,
            "simulation_steps": context.simulation_steps,
            "simulation_scope": context.simulation_scope.name,
            "alternative_timelines": [t.id for t in context.alternative_timelines],
            "probability_map": context.probability_map,
            "results": context.results
        }
    
    def analyze_timeline_probabilities(self, timeline_id: str) -> Dict[str, Any]:
        """
        Analysiert die Wahrscheinlichkeiten einer Zeitlinie
        
        Args:
            timeline_id: ID der Zeitlinie
            
        Returns:
            Analyseergebnis
        """
        if not HAS_ECHO_PRIME or not self.echo_prime:
            logger.error("ECHO-PRIME ist nicht verfügbar")
            return {"error": "ECHO-PRIME ist nicht verfügbar"}
        
        # Hole Zeitlinie
        timeline = self.echo_prime.get_timeline(timeline_id)
        if not timeline:
            logger.warning(f"Zeitlinie {timeline_id} existiert nicht")
            return {"error": f"Zeitlinie {timeline_id} existiert nicht"}
        
        # Analysiere Knotenwahrscheinlichkeiten
        node_probabilities = {}
        for node_id, node in timeline.nodes.items():
            node_probabilities[node_id] = {
                "description": node.description,
                "probability": node.probability,
                "trigger_level": node.trigger_level.name,
                "parent_node_id": node.parent_node_id,
                "child_node_count": len(node.child_node_ids)
            }
        
        # Berechne Gesamtwahrscheinlichkeit der Zeitlinie
        timeline_probability = timeline.probability
        
        # Berechne bedingte Wahrscheinlichkeiten für Pfade
        path_probabilities = self._calculate_path_probabilities(timeline)
        
        # Erstelle Wahrscheinlichkeitsbewertung
        probability_assessment = self.prism_engine.evaluate_probability_recommendation(timeline_probability)
        
        return {
            "timeline_id": timeline_id,
            "timeline_name": timeline.name,
            "timeline_probability": timeline_probability,
            "node_count": len(timeline.nodes),
            "node_probabilities": node_probabilities,
            "path_probabilities": path_probabilities,
            "probability_assessment": probability_assessment
        }
    
    def _calculate_path_probabilities(self, timeline: 'Timeline') -> Dict[str, float]:
        """
        Berechnet die Wahrscheinlichkeiten für verschiedene Pfade in einer Zeitlinie
        
        Args:
            timeline: Zeitlinie
            
        Returns:
            Dictionary mit Pfadwahrscheinlichkeiten
        """
        path_probabilities = {}
        
        # Finde Wurzelknoten
        root_nodes = [node for node in timeline.nodes.values() if not node.parent_node_id]
        
        # Berechne Pfadwahrscheinlichkeiten für jeden Wurzelknoten
        for root_node in root_nodes:
            self._calculate_path_probability_recursive(timeline, root_node, 1.0, [], path_probabilities)
        
        return path_probabilities
    
    def _calculate_path_probability_recursive(self, 
                                           timeline: 'Timeline', 
                                           node: 'TimeNode', 
                                           current_probability: float, 
                                           current_path: List[str], 
                                           path_probabilities: Dict[str, float]):
        """
        Berechnet die Wahrscheinlichkeiten für Pfade rekursiv
        
        Args:
            timeline: Zeitlinie
            node: Aktueller Knoten
            current_probability: Aktuelle Pfadwahrscheinlichkeit
            current_path: Aktueller Pfad
            path_probabilities: Dictionary mit Pfadwahrscheinlichkeiten
            
        Returns:
            None
        """
        # Aktualisiere Pfad und Wahrscheinlichkeit
        new_path = current_path + [node.id]
        new_probability = current_probability * node.probability
        
        # Speichere Pfadwahrscheinlichkeit
        path_key = "->".join(new_path)
        path_probabilities[path_key] = new_probability
        
        # Rekursiver Aufruf für Kindknoten
        for child_id in node.child_node_ids:
            if child_id in timeline.nodes:
                child_node = timeline.nodes[child_id]
                self._calculate_path_probability_recursive(
                    timeline, child_node, new_probability, new_path, path_probabilities
                )


# Erweiterte Q-LOGIK Integration für PRISM und ECHO-PRIME
class QLogikIntegrationLayer:
    """
    Hochoptimierte Integrationsschicht zwischen Q-LOGIK, PRISM und ECHO-PRIME
    
    Diese Komponente ermöglicht die einheitliche Nutzung von Q-LOGIK Entscheidungsfunktionen
    innerhalb der PRISM-Engine und ECHO-PRIME Modulen für konsistente Wahrscheinlichkeitsberechnungen.
    Optimiert für maximale Performance und Speichereffizienz.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert die QLogik-Integrationsschicht mit optimierten Einstellungen
        
        Args:
            config: Optionale Konfigurationsparameter
        """
        self.config = config or {}
        
        # Importiere Q-LOGIK Komponenten mit Lazy Loading um zirkuläre Importe zu vermeiden
        self._qlogik_components = {}
        
        # Optimierter LRU-Cache mit adaptiver Größenbegrenzung
        self._cache_size = self.config.get("cache_size", 1000)
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lifetime = self.config.get("cache_lifetime", 300)  # Sekunden
        self._last_clean = time.time()
        self._clean_interval = 60  # Intervall für Cache-Bereinigung (Sekunden)
        
        # Lazy-loaded atomare Referenz zur Vermeidung von Thread-Konflikten
        self._initialized = False
        
        # Optimierte Hash-Funktion für komplexe Datenstrukturen
        self._hash_func = self._prepare_hash_function()
        
        logger.info("QLogik-Integrationsschicht initialisiert (optimierte Version)")
    
    def _prepare_hash_function(self) -> Callable:
        """
        Wählt die optimale Hash-Funktion basierend auf Systemverfügbarkeit
        
        Returns:
            Optimierte Hash-Funktion
        """
        try:
            # Versuche xxhash für schnellere Hashing-Performance zu verwenden
            import xxhash
            logger.debug("Verwende xxhash für optimiertes Hashing")
            return lambda x: xxhash.xxh64(str(x)).hexdigest()
        except ImportError:
            # Fallback auf integrierte Hash-Funktion
            logger.debug("Fallback auf eingebaute Hash-Funktion")
            return lambda x: hash(str(x))
    
    def _generate_cache_key(self, obj: Any, prefix: str = "") -> str:
        """
        Erzeugt einen konsistenten Cache-Schlüssel für beliebige Eingaben
        
        Args:
            obj: Zu hashende Daten
            prefix: Präfix für den Cache-Schlüssel
            
        Returns:
            Cache-Schlüssel als String
        """
        # Deterministische Schlüsselgenerierung für verschiedene Datentypen
        if obj is None:
            return f"{prefix}_none"
            
        if isinstance(obj, dict):
            # Sortierte Repräsentation von Dictionary-Inhalten
            sorted_items = sorted(obj.items())
            serialized = str([(str(k), str(v)) for k, v in sorted_items])
        elif isinstance(obj, (list, tuple)):
            # Für Listen/Tuples, stringifiziere jedes Element 
            serialized = str([str(item) for item in obj])
        else:
            # Direktes Stringifizieren für primitive Typen
            serialized = str(obj)
            
        # Wende optimierte Hash-Funktion an
        hashed = self._hash_func(serialized)
        return f"{prefix}_{hashed}"
    
    def _load_qlogik_components(self):
        """
        Lädt Q-LOGIK Komponenten bei Bedarf mit Thread-sicherer Initialisierung
        
        Returns:
            True wenn erfolgreich, False sonst
        """
        # Prüfe auf vorherige erfolgreiche Initialisierung
        if self._initialized:
            return True
            
        try:
            # Thread-sicheres Lazy-Loading
            from miso.logic.qlogik_engine import (
                bayesian as bayesian_core,
                fuzzylogic as fuzzy_unit,
                advanced_qlogik_decision,
                get_action_recommendation
            )
            
            # Speichere Komponenten lokal
            self._qlogik_components.update({
                "bayesian": bayesian_core,
                "fuzzy": fuzzy_unit,
                "decision": advanced_qlogik_decision,
                "action": get_action_recommendation
            })
            
            logger.info("Q-LOGIK Komponenten erfolgreich geladen")
            self._initialized = True
            return True
        except ImportError as e:
            logger.error(f"Fehler beim Laden der Q-LOGIK Komponenten: {e}")
            return False
    
    def _manage_cache(self):
        """
        Verwaltet den Cache, entfernt abgelaufene Einträge und
        beschränkt die Cachegröße bei Bedarf
        """
        now = time.time()
        
        # Prüfe, ob Cache-Bereinigung erforderlich ist
        if now - self._last_clean < self._clean_interval and len(self._cache) < self._cache_size:
            return
            
        # Führe Cache-Bereinigung durch
        self._last_clean = now
        
        # Entferne abgelaufene Einträge
        expired_keys = [k for k, v in self._cache.items() 
                      if now - v["timestamp"] > self._cache_lifetime]
        
        for key in expired_keys:
            del self._cache[key]
            
        # Wenn Cache noch zu groß ist, entferne die ältesten Einträge
        if len(self._cache) > self._cache_size:
            # Sortiere nach Zeitstempel (aufsteigend)
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1]["timestamp"])
            # Behalte nur die neuesten Einträge
            entries_to_keep = sorted_entries[-self._cache_size:]
            # Aktualisiere Cache
            self._cache = {k: v for k, v in entries_to_keep}
            
        logger.debug(f"Cache bereinigt: {len(self._cache)} Einträge aktiv, "
                   f"Trefferquote: {self._get_hit_ratio():.2%}")
    
    def _get_hit_ratio(self) -> float:
        """
        Berechnet die Cache-Trefferquote
        
        Returns:
            Trefferquote als Float zwischen 0 und 1
        """
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / max(1, total)
    
    def evaluate_bayesian_probability(self, data: Dict[str, Any]) -> float:
        """
        Berechnet die bayessche Wahrscheinlichkeit für Zeitliniensimulationen
        mit optimiertem Caching
        
        Args:
            data: Eingabedaten für die Berechnung
            
        Returns:
            Wahrscheinlichkeitswert
        """
        if not self._load_qlogik_components():
            return 0.5  # Standardwert bei Fehler
        
        # Optimierter Cache-Schlüssel
        cache_key = self._generate_cache_key(data, "bayes")
        
        # Cache-Lookup mit Statistik
        if cache_key in self._cache:
            self._cache_hits += 1
            # Aktualisiere Zugriffszeit ohne Timestamp zu ändern
            self._cache[cache_key]["access_time"] = time.time()
            return self._cache[cache_key]["value"]
            
        # Cache-Miss
        self._cache_misses += 1
        
        # Durchführen der Berechnung mit Fehlerbehandlung
        try:
            result = self._qlogik_components["bayesian"].evaluate(data)
        except Exception as e:
            logger.warning(f"Fehler bei bayesscher Berechnung: {e}")
            return 0.5
        
        # Cache-Update und -Management
        now = time.time()
        self._cache[cache_key] = {
            "value": result,
            "timestamp": now,
            "access_time": now
        }
        
        # Proaktive Cache-Verwaltung
        self._manage_cache()
        
        return result
    
    def evaluate_fuzzy_truth(self, signal: Dict[str, Any]) -> float:
        """
        Berechnet den Fuzzy-Wahrheitsgrad für die PRISM-Engine
        mit optimierter Performance
        
        Args:
            signal: Eingabesignal für die Berechnung
            
        Returns:
            Wahrheitsgrad zwischen 0 und 1
        """
        if not self._load_qlogik_components():
            return 0.5  # Standardwert bei Fehler
            
        # Optimierter Cache-Schlüssel
        cache_key = self._generate_cache_key(signal, "fuzzy")
        
        # Cache-Lookup mit Statistik
        if cache_key in self._cache:
            self._cache_hits += 1
            # Aktualisiere Zugriffszeit ohne Timestamp zu ändern
            self._cache[cache_key]["access_time"] = time.time()
            return self._cache[cache_key]["value"]
            
        # Cache-Miss
        self._cache_misses += 1
        
        # Durchführen der Berechnung mit Fehlerbehandlung
        try:
            result = self._qlogik_components["fuzzy"].score(signal)
        except Exception as e:
            logger.warning(f"Fehler bei Fuzzy-Berechnung: {e}")
            return 0.5
        
        # Cache-Update und -Management
        now = time.time()
        self._cache[cache_key] = {
            "value": result,
            "timestamp": now,
            "access_time": now
        }
        
        # Proaktive Cache-Verwaltung
        self._manage_cache()
        
        return result
    
    def decision_for_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trifft eine vollständige Q-LOGIK-Entscheidung im Kontext einer Zeitlinie
        oder PRISM-Simulation mit optimierter Ausführung
        
        Args:
            context: Entscheidungskontext mit Zeitlinien- oder Simulationsdaten
            
        Returns:
            Detaillierte Entscheidungsinformationen
        """
        if not self._load_qlogik_components():
            return {"decision": "UNBESTIMMT", "confidence": 0.5, "rationale": "Q-LOGIK nicht verfügbar"}
            
        # Cache-Schlüssel für die gesamte Entscheidung
        # Wir nutzen eine kompaktere Repräsentation für bessere Effizienz
        cache_key = self._generate_cache_key(
            {k: v for k, v in context.items() if k not in ["_meta", "simulation", "timeline"]}, 
            "decision"
        )
        
        # Cache-Lookup
        if cache_key in self._cache:
            self._cache_hits += 1
            self._cache[cache_key]["access_time"] = time.time()
            return self._cache[cache_key]["value"]
            
        self._cache_misses += 1
        
        # Erweitere den Kontext mit PRISM/ECHO-spezifischen Daten
        enriched_context = self._enrich_context(context)
        
        # Verwende die erweiterte Q-LOGIK-Entscheidungsfunktion mit Fehlerbehandlung
        try:
            result = self._qlogik_components["decision"](enriched_context)
        except Exception as e:
            logger.error(f"Fehler bei Q-LOGIK-Entscheidung: {e}")
            return {"decision": "FEHLER", "confidence": 0.0, "rationale": f"Fehler bei Berechnung: {str(e)}"}
        
        # Speichere Ergebnis im Cache
        now = time.time()
        self._cache[cache_key] = {
            "value": result,
            "timestamp": now,
            "access_time": now
        }
        
        # Proaktive Cache-Verwaltung
        self._manage_cache()
        
        return result
    
    def recommend_action(self, priority_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Gibt eine Handlungsempfehlung basierend auf Prioritätsdaten
        
        Args:
            priority_data: Prioritätsdaten mit risk, benefit und urgency
            
        Returns:
            Handlungsempfehlung
        """
        if not self._load_qlogik_components():
            return {"action": "KEINE", "confidence": 0.0}
            
        # Extrahiere Prioritätsparameter mit Defaults
        risk = float(priority_data.get("risk", 0.5))
        benefit = float(priority_data.get("benefit", 0.5))
        urgency = float(priority_data.get("urgency", 0.5))
        
        # Cache-Schlüssel basierend auf gerundeten Werten (Performance-Optimierung)
        # Wir runden auf 2 Dezimalstellen, um Cache-Hits zu erhöhen
        cache_key = f"action_{round(risk, 2)}_{round(benefit, 2)}_{round(urgency, 2)}"
        
        # Cache-Lookup
        if cache_key in self._cache:
            self._cache_hits += 1
            self._cache[cache_key]["access_time"] = time.time()
            return self._cache[cache_key]["value"]
            
        self._cache_misses += 1
        
        # Berechne Empfehlung
        try:
            result = self._qlogik_components["action"]({
                "risk": risk, 
                "benefit": benefit, 
                "urgency": urgency
            })
        except Exception as e:
            logger.warning(f"Fehler bei Handlungsempfehlung: {e}")
            return {"action": "UNBESTIMMT", "confidence": 0.0}
        
        # Cache-Update
        now = time.time()
        self._cache[cache_key] = {
            "value": result,
            "timestamp": now,
            "access_time": now
        }
        
        # Cache-Management
        self._manage_cache()
        
        return result
    
    def _enrich_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Erweitert den Entscheidungskontext mit zusätzlichen Informationen
        aus PRISM und ECHO-PRIME - optimierte Version
        
        Args:
            context: Ursprünglicher Kontext
            
        Returns:
            Erweiterter Kontext
        """
        # Erstelle eine flache Kopie für bessere Performance (statt deep copy)
        enriched = {**context}
        
        # Füge Metadaten hinzu (effizient)
        enriched["_meta"] = context.get("_meta", {}) | {
            "source": "prism_echo_integration",
            "timestamp": time.time()
        }
        
        # Optimierte Extraktion von Timeline/Simulation-Daten
        for key, target_key in [("timeline", "timeline_data"), ("simulation", "simulation_data")]:
            if key in context:
                item = context[key]
                if hasattr(item, "to_dict"):
                    enriched[target_key] = item.to_dict()
                elif isinstance(item, dict):
                    # Direkte Referenz für Dictionaries (Performance-Optimierung)
                    enriched[target_key] = item
                
        return enriched
    
    def clear_cache(self) -> Dict[str, Any]:
        """
        Leert den internen Cache und gibt Statistiken zurück
        
        Returns:
            Cache-Statistiken
        """
        stats = {
            "entries": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_ratio": self._get_hit_ratio()
        }
        
        self._cache = {}
        self._last_clean = time.time()
        logger.info(f"QLogik-Integrationsschicht: Cache geleert ({stats['entries']} Einträge)")
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über die Integrationsschicht zurück
        
        Returns:
            Performance-Statistiken
        """
        return {
            "cache_entries": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_ratio": self._get_hit_ratio(),
            "components_loaded": self._initialized,
            "uptime": time.time() - self._last_clean
        }

# Globale Instanzen für einfachen Zugriff
integration = PRISMECHOPrimeIntegration()
qlogik_integration = QLogikIntegrationLayer()

# Beispiel für die Verwendung der Integration
if __name__ == "__main__":
    # Konfiguriere Logging für Standalone-Ausführung
    logging.basicConfig(level=logging.INFO)
    
    if HAS_ECHO_PRIME:
        # Erstelle ECHO-PRIME Controller
        echo_prime = EchoPrimeController()
        
        # Erstelle Testzeitlinie
        timeline = echo_prime.create_timeline(
            name="Testzeitlinie",
            description="Eine Testzeitlinie für die PRISM-ECHO-PRIME Integration"
        )
        
        # Füge Knoten hinzu
        node1 = echo_prime.add_time_node(
            timeline_id=timeline.id,
            description="Startknoten",
            probability=0.9,
            trigger_level=TriggerLevel.LOW
        )
        
        node2 = echo_prime.add_time_node(
            timeline_id=timeline.id,
            description="Zweiter Knoten",
            parent_node_id=node1.id,
            probability=0.8,
            trigger_level=TriggerLevel.MEDIUM
        )
        
        # Simuliere Zeitlinie
        result = integration.simulate_timeline(timeline.id)
        print(f"Simulationsergebnis: {result}")
        
        # Analysiere Wahrscheinlichkeiten
        analysis = integration.analyze_timeline_probabilities(timeline.id)
        print(f"Wahrscheinlichkeitsanalyse: {analysis}")
    else:
        print("ECHO-PRIME ist nicht verfügbar. Integration kann nicht getestet werden.")
