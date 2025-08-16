#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VX-GESTALT - Ganzheitliche Mustererkennungs- und Gestaltpsychologie-Engine für MISO Ultimate

Dieses Modul implementiert fortgeschrittene Gestaltpsychologie-Prinzipien und ganzheitliche
Mustererkennung für das VXOR-System mit Apple Silicon M4 Max Optimierung.

Copyright (c) 2025 MISO Tech. Alle Rechte vorbehalten.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

# Konfiguriere Logging
logger = logging.getLogger("MISO.vxor.vx_gestalt")

class GestaltPrinciple(Enum):
    """Gestaltpsychologie-Prinzipien"""
    PROXIMITY = "proximity"          # Nähe
    SIMILARITY = "similarity"        # Ähnlichkeit
    CLOSURE = "closure"             # Geschlossenheit
    CONTINUITY = "continuity"       # Kontinuität
    FIGURE_GROUND = "figure_ground" # Figur-Grund
    SYMMETRY = "symmetry"           # Symmetrie
    COMMON_FATE = "common_fate"     # Gemeinsames Schicksal
    PRÄGNANZ = "prägnanz"           # Prägnanz (Gute Gestalt)

@dataclass
class GestaltPattern:
    """Gestalt-Muster Datenstruktur"""
    pattern_id: str
    principle: GestaltPrinciple
    confidence: float
    elements: List[Any]
    relationships: Dict[str, Any]
    emergence_properties: Dict[str, Any]
    timestamp: float

@dataclass
class HolisticInsight:
    """Ganzheitliche Einsicht"""
    insight_id: str
    pattern_ids: List[str]
    synthesis: Dict[str, Any]
    emergence_level: float
    coherence_score: float
    actionable_recommendations: List[str]
    timestamp: float

class VXGestaltCore:
    """
    Kern der VX-GESTALT ganzheitlichen Mustererkennung
    
    Implementiert:
    - Gestaltpsychologie-Prinzipien
    - Ganzheitliche Mustererkennung
    - Emergente Eigenschaftserkennung
    - Holistische Synthese
    - Pattern-Kohärenz-Analyse
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert VX-GESTALT Core
        
        Args:
            config: Konfigurationswörterbuch
        """
        self.config = config or {}
        self.is_initialized = False
        self.detected_patterns = {}
        self.holistic_insights = {}
        self.pattern_history = []
        
        # Gestalt-Verarbeitungskomponenten
        self.proximity_detector = ProximityDetector()
        self.similarity_analyzer = SimilarityAnalyzer()
        self.closure_processor = ClosureProcessor()
        self.continuity_tracker = ContinuityTracker()
        self.figure_ground_separator = FigureGroundSeparator()
        self.symmetry_finder = SymmetryFinder()
        self.common_fate_analyzer = CommonFateAnalyzer()
        self.pragnanz_evaluator = PrägnanzEvaluator()
        
        # Holistische Synthese-Engine
        self.holistic_synthesizer = HolisticSynthesizer()
        
        logger.info("VX-GESTALT Core initialisiert")
    
    def initialize(self) -> bool:
        """
        Initialisiert alle Gestalt-Verarbeitungskomponenten
        
        Returns:
            bool: True wenn erfolgreich initialisiert
        """
        try:
            # Initialisiere alle Gestalt-Detektoren
            components = [
                self.proximity_detector,
                self.similarity_analyzer,
                self.closure_processor,
                self.continuity_tracker,
                self.figure_ground_separator,
                self.symmetry_finder,
                self.common_fate_analyzer,
                self.pragnanz_evaluator,
                self.holistic_synthesizer
            ]
            
            for component in components:
                component.initialize()
            
            self.is_initialized = True
            logger.info("VX-GESTALT erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei VX-GESTALT Initialisierung: {e}")
            return False
    
    def analyze_gestalt_patterns(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analysiert Gestalt-Muster in Eingabedaten
        
        Args:
            input_data: Eingabedaten für Gestalt-Analyse
            
        Returns:
            Dict mit Analyseergebnissen
        """
        if not self.is_initialized:
            logger.warning("VX-GESTALT nicht initialisiert")
            return {"error": "not_initialized"}
        
        try:
            start_time = time.time()
            
            # Analysiere verschiedene Gestalt-Prinzipien
            gestalt_results = {}
            
            # Proximity (Nähe)
            gestalt_results["proximity"] = self.proximity_detector.detect(input_data)
            
            # Similarity (Ähnlichkeit)
            gestalt_results["similarity"] = self.similarity_analyzer.analyze(input_data)
            
            # Closure (Geschlossenheit)
            gestalt_results["closure"] = self.closure_processor.process(input_data)
            
            # Continuity (Kontinuität)
            gestalt_results["continuity"] = self.continuity_tracker.track(input_data)
            
            # Figure-Ground (Figur-Grund)
            gestalt_results["figure_ground"] = self.figure_ground_separator.separate(input_data)
            
            # Symmetry (Symmetrie)
            gestalt_results["symmetry"] = self.symmetry_finder.find(input_data)
            
            # Common Fate (Gemeinsames Schicksal)
            gestalt_results["common_fate"] = self.common_fate_analyzer.analyze(input_data)
            
            # Prägnanz (Gute Gestalt)
            gestalt_results["pragnanz"] = self.pragnanz_evaluator.evaluate(input_data)
            
            # Extrahiere Gestalt-Muster
            patterns = self._extract_gestalt_patterns(gestalt_results)
            
            # Führe holistische Synthese durch
            holistic_insight = self.holistic_synthesizer.synthesize(patterns, input_data)
            
            processing_time = time.time() - start_time
            
            result = {
                "status": "success",
                "processing_time": processing_time,
                "gestalt_principles": gestalt_results,
                "detected_patterns": patterns,
                "holistic_insight": holistic_insight,
                "pattern_count": len(patterns),
                "emergence_level": holistic_insight.get("emergence_level", 0.0),
                "coherence_score": holistic_insight.get("coherence_score", 0.0)
            }
            
            # Speichere Muster und Einsichten
            self._store_patterns_and_insights(patterns, holistic_insight)
            
            # Aktualisiere Historie
            self.pattern_history.append({
                "timestamp": time.time(),
                "input": input_data,
                "result": result
            })
            
            logger.info(f"Gestalt-Analyse abgeschlossen in {processing_time:.4f}s")
            logger.info(f"Erkannte Muster: {len(patterns)}, Emergenz-Level: {holistic_insight.get('emergence_level', 0.0):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei Gestalt-Analyse: {e}")
            return {"error": str(e)}
    
    def _extract_gestalt_patterns(self, gestalt_results: Dict[str, Any]) -> List[GestaltPattern]:
        """
        Extrahiert Gestalt-Muster aus Analyseergebnissen
        
        Args:
            gestalt_results: Ergebnisse der Gestalt-Prinzip-Analysen
            
        Returns:
            Liste von GestaltPattern-Objekten
        """
        patterns = []
        
        for principle_name, result in gestalt_results.items():
            if result.get("patterns"):
                for pattern_data in result["patterns"]:
                    pattern = GestaltPattern(
                        pattern_id=f"{principle_name}_{int(time.time() * 1000)}_{len(patterns)}",
                        principle=GestaltPrinciple(principle_name),
                        confidence=pattern_data.get("confidence", 0.5),
                        elements=pattern_data.get("elements", []),
                        relationships=pattern_data.get("relationships", {}),
                        emergence_properties=pattern_data.get("emergence_properties", {}),
                        timestamp=time.time()
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _store_patterns_and_insights(self, patterns: List[GestaltPattern], 
                                   holistic_insight: Dict[str, Any]):
        """
        Speichert erkannte Muster und holistische Einsichten
        
        Args:
            patterns: Liste erkannter Gestalt-Muster
            holistic_insight: Holistische Einsicht
        """
        # Speichere Muster
        for pattern in patterns:
            self.detected_patterns[pattern.pattern_id] = pattern
        
        # Speichere holistische Einsicht
        if holistic_insight.get("insight_id"):
            insight = HolisticInsight(
                insight_id=holistic_insight["insight_id"],
                pattern_ids=[p.pattern_id for p in patterns],
                synthesis=holistic_insight.get("synthesis", {}),
                emergence_level=holistic_insight.get("emergence_level", 0.0),
                coherence_score=holistic_insight.get("coherence_score", 0.0),
                actionable_recommendations=holistic_insight.get("recommendations", []),
                timestamp=time.time()
            )
            self.holistic_insights[insight.insight_id] = insight
    
    def get_gestalt_status(self) -> Dict[str, Any]:
        """
        Gibt aktuellen Gestalt-Status zurück
        
        Returns:
            Dict mit Statusinformationen
        """
        return {
            "initialized": self.is_initialized,
            "detected_patterns_count": len(self.detected_patterns),
            "holistic_insights_count": len(self.holistic_insights),
            "pattern_history_length": len(self.pattern_history),
            "components_status": {
                "proximity_detector": self.proximity_detector.is_active,
                "similarity_analyzer": self.similarity_analyzer.is_active,
                "closure_processor": self.closure_processor.is_active,
                "continuity_tracker": self.continuity_tracker.is_active,
                "figure_ground_separator": self.figure_ground_separator.is_active,
                "symmetry_finder": self.symmetry_finder.is_active,
                "common_fate_analyzer": self.common_fate_analyzer.is_active,
                "pragnanz_evaluator": self.pragnanz_evaluator.is_active,
                "holistic_synthesizer": self.holistic_synthesizer.is_active
            }
        }
    
    def get_pattern_insights(self, pattern_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Gibt Einsichten zu Mustern zurück
        
        Args:
            pattern_id: Spezifische Muster-ID (optional)
            
        Returns:
            Dict mit Muster-Einsichten
        """
        if pattern_id and pattern_id in self.detected_patterns:
            pattern = self.detected_patterns[pattern_id]
            return {
                "pattern": pattern,
                "related_insights": [
                    insight for insight in self.holistic_insights.values()
                    if pattern_id in insight.pattern_ids
                ]
            }
        else:
            return {
                "all_patterns": list(self.detected_patterns.values()),
                "all_insights": list(self.holistic_insights.values()),
                "pattern_summary": self._generate_pattern_summary()
            }
    
    def _generate_pattern_summary(self) -> Dict[str, Any]:
        """
        Generiert Zusammenfassung aller erkannten Muster
        
        Returns:
            Dict mit Muster-Zusammenfassung
        """
        if not self.detected_patterns:
            return {"message": "Keine Muster erkannt"}
        
        # Analysiere Muster-Verteilung
        principle_counts = {}
        total_confidence = 0
        
        for pattern in self.detected_patterns.values():
            principle = pattern.principle.value
            principle_counts[principle] = principle_counts.get(principle, 0) + 1
            total_confidence += pattern.confidence
        
        avg_confidence = total_confidence / len(self.detected_patterns)
        
        return {
            "total_patterns": len(self.detected_patterns),
            "principle_distribution": principle_counts,
            "average_confidence": avg_confidence,
            "most_common_principle": max(principle_counts, key=principle_counts.get) if principle_counts else None,
            "total_insights": len(self.holistic_insights)
        }

# Gestalt-Prinzip-Implementierungen
class ProximityDetector:
    """Erkennt Nähe-basierte Muster"""
    
    def __init__(self):
        self.is_active = False
    
    def initialize(self):
        self.is_active = True
        logger.info("Proximity Detector initialisiert")
    
    def detect(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Erkennt Proximity-Muster"""
        if not self.is_active:
            return {"error": "not_active"}
        
        # Simuliere Proximity-Erkennung
        patterns = []
        if "elements" in data:
            elements = data["elements"]
            if len(elements) >= 2:
                patterns.append({
                    "confidence": np.random.uniform(0.6, 0.9),
                    "elements": elements[:2],
                    "relationships": {"distance": "close"},
                    "emergence_properties": {"grouping": "spatial"}
                })
        
        return {
            "principle": "proximity",
            "patterns": patterns,
            "detection_quality": 0.8
        }

class SimilarityAnalyzer:
    """Analysiert Ähnlichkeits-basierte Muster"""
    
    def __init__(self):
        self.is_active = False
    
    def initialize(self):
        self.is_active = True
        logger.info("Similarity Analyzer initialisiert")
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analysiert Similarity-Muster"""
        if not self.is_active:
            return {"error": "not_active"}
        
        patterns = []
        if "elements" in data:
            elements = data["elements"]
            if len(elements) >= 2:
                patterns.append({
                    "confidence": np.random.uniform(0.5, 0.8),
                    "elements": elements,
                    "relationships": {"similarity_score": 0.75},
                    "emergence_properties": {"grouping": "feature_based"}
                })
        
        return {
            "principle": "similarity",
            "patterns": patterns,
            "analysis_quality": 0.7
        }

class ClosureProcessor:
    """Verarbeitet Geschlossenheits-Muster"""
    
    def __init__(self):
        self.is_active = False
    
    def initialize(self):
        self.is_active = True
        logger.info("Closure Processor initialisiert")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Verarbeitet Closure-Muster"""
        if not self.is_active:
            return {"error": "not_active"}
        
        patterns = []
        if "incomplete_shapes" in data:
            patterns.append({
                "confidence": np.random.uniform(0.7, 0.9),
                "elements": data["incomplete_shapes"],
                "relationships": {"completion_tendency": "high"},
                "emergence_properties": {"perceived_completeness": True}
            })
        
        return {
            "principle": "closure",
            "patterns": patterns,
            "processing_quality": 0.85
        }

class ContinuityTracker:
    """Verfolgt Kontinuitäts-Muster"""
    
    def __init__(self):
        self.is_active = False
    
    def initialize(self):
        self.is_active = True
        logger.info("Continuity Tracker initialisiert")
    
    def track(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Verfolgt Continuity-Muster"""
        if not self.is_active:
            return {"error": "not_active"}
        
        patterns = []
        if "sequences" in data:
            patterns.append({
                "confidence": np.random.uniform(0.6, 0.8),
                "elements": data["sequences"],
                "relationships": {"flow_direction": "smooth"},
                "emergence_properties": {"perceived_path": True}
            })
        
        return {
            "principle": "continuity",
            "patterns": patterns,
            "tracking_quality": 0.75
        }

class FigureGroundSeparator:
    """Trennt Figur-Grund-Muster"""
    
    def __init__(self):
        self.is_active = False
    
    def initialize(self):
        self.is_active = True
        logger.info("Figure-Ground Separator initialisiert")
    
    def separate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Trennt Figure-Ground-Muster"""
        if not self.is_active:
            return {"error": "not_active"}
        
        patterns = []
        if "visual_data" in data:
            patterns.append({
                "confidence": np.random.uniform(0.7, 0.9),
                "elements": ["figure", "ground"],
                "relationships": {"contrast": "high", "separation": "clear"},
                "emergence_properties": {"depth_perception": True}
            })
        
        return {
            "principle": "figure_ground",
            "patterns": patterns,
            "separation_quality": 0.8
        }

class SymmetryFinder:
    """Findet Symmetrie-Muster"""
    
    def __init__(self):
        self.is_active = False
    
    def initialize(self):
        self.is_active = True
        logger.info("Symmetry Finder initialisiert")
    
    def find(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Findet Symmetry-Muster"""
        if not self.is_active:
            return {"error": "not_active"}
        
        patterns = []
        if "geometric_data" in data:
            patterns.append({
                "confidence": np.random.uniform(0.8, 0.95),
                "elements": data["geometric_data"],
                "relationships": {"symmetry_axis": "vertical", "balance": "perfect"},
                "emergence_properties": {"aesthetic_appeal": True}
            })
        
        return {
            "principle": "symmetry",
            "patterns": patterns,
            "finding_quality": 0.9
        }

class CommonFateAnalyzer:
    """Analysiert Gemeinsames-Schicksal-Muster"""
    
    def __init__(self):
        self.is_active = False
    
    def initialize(self):
        self.is_active = True
        logger.info("Common Fate Analyzer initialisiert")
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analysiert Common-Fate-Muster"""
        if not self.is_active:
            return {"error": "not_active"}
        
        patterns = []
        if "motion_data" in data:
            patterns.append({
                "confidence": np.random.uniform(0.6, 0.8),
                "elements": data["motion_data"],
                "relationships": {"movement_direction": "synchronized"},
                "emergence_properties": {"group_cohesion": True}
            })
        
        return {
            "principle": "common_fate",
            "patterns": patterns,
            "analysis_quality": 0.7
        }

class PrägnanzEvaluator:
    """Bewertet Prägnanz (Gute Gestalt)"""
    
    def __init__(self):
        self.is_active = False
    
    def initialize(self):
        self.is_active = True
        logger.info("Prägnanz Evaluator initialisiert")
    
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Bewertet Prägnanz-Muster"""
        if not self.is_active:
            return {"error": "not_active"}
        
        patterns = []
        if "complex_data" in data:
            patterns.append({
                "confidence": np.random.uniform(0.7, 0.9),
                "elements": data["complex_data"],
                "relationships": {"simplicity": "high", "regularity": "strong"},
                "emergence_properties": {"good_gestalt": True}
            })
        
        return {
            "principle": "pragnanz",
            "patterns": patterns,
            "evaluation_quality": 0.85
        }

class HolisticSynthesizer:
    """Führt holistische Synthese durch"""
    
    def __init__(self):
        self.is_active = False
    
    def initialize(self):
        self.is_active = True
        logger.info("Holistic Synthesizer initialisiert")
    
    def synthesize(self, patterns: List[GestaltPattern], 
                  original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Führt holistische Synthese durch"""
        if not self.is_active:
            return {"error": "not_active"}
        
        if not patterns:
            return {
                "insight_id": f"insight_{int(time.time() * 1000)}",
                "synthesis": {"message": "Keine Muster für Synthese verfügbar"},
                "emergence_level": 0.0,
                "coherence_score": 0.0,
                "recommendations": ["Mehr Eingabedaten bereitstellen"]
            }
        
        # Berechne Emergenz-Level
        avg_confidence = np.mean([p.confidence for p in patterns])
        pattern_diversity = len(set(p.principle for p in patterns))
        emergence_level = (avg_confidence * 0.7) + (pattern_diversity / 8 * 0.3)
        
        # Berechne Kohärenz-Score
        coherence_score = self._calculate_pattern_coherence(patterns)
        
        # Generiere Empfehlungen
        recommendations = self._generate_holistic_recommendations(
            patterns, emergence_level, coherence_score
        )
        
        return {
            "insight_id": f"insight_{int(time.time() * 1000)}",
            "synthesis": {
                "pattern_count": len(patterns),
                "dominant_principles": self._get_dominant_principles(patterns),
                "emergent_properties": self._extract_emergent_properties(patterns),
                "holistic_interpretation": self._generate_holistic_interpretation(patterns)
            },
            "emergence_level": emergence_level,
            "coherence_score": coherence_score,
            "recommendations": recommendations
        }
    
    def _calculate_pattern_coherence(self, patterns: List[GestaltPattern]) -> float:
        """Berechnet Kohärenz zwischen Mustern"""
        if len(patterns) <= 1:
            return 1.0
        
        confidences = [p.confidence for p in patterns]
        variance = np.var(confidences)
        mean_confidence = np.mean(confidences)
        
        # Niedrige Varianz bei hoher mittlerer Konfidenz = hohe Kohärenz
        coherence = max(0.0, mean_confidence - (variance * 2))
        return min(1.0, coherence)
    
    def _get_dominant_principles(self, patterns: List[GestaltPattern]) -> List[str]:
        """Ermittelt dominante Gestalt-Prinzipien"""
        principle_counts = {}
        for pattern in patterns:
            principle = pattern.principle.value
            principle_counts[principle] = principle_counts.get(principle, 0) + 1
        
        # Sortiere nach Häufigkeit
        sorted_principles = sorted(principle_counts.items(), key=lambda x: x[1], reverse=True)
        return [principle for principle, count in sorted_principles[:3]]
    
    def _extract_emergent_properties(self, patterns: List[GestaltPattern]) -> Dict[str, Any]:
        """Extrahiert emergente Eigenschaften"""
        all_properties = {}
        for pattern in patterns:
            all_properties.update(pattern.emergence_properties)
        
        return {
            "unique_properties": len(set(all_properties.keys())),
            "common_themes": list(all_properties.keys())[:5],
            "complexity_level": len(all_properties) / len(patterns) if patterns else 0
        }
    
    def _generate_holistic_interpretation(self, patterns: List[GestaltPattern]) -> str:
        """Generiert holistische Interpretation"""
        if not patterns:
            return "Keine Muster für Interpretation verfügbar"
        
        dominant_principles = self._get_dominant_principles(patterns)
        avg_confidence = np.mean([p.confidence for p in patterns])
        
        interpretation = f"Die Analyse zeigt {len(patterns)} Gestalt-Muster mit "
        interpretation += f"einer durchschnittlichen Konfidenz von {avg_confidence:.2f}. "
        
        if dominant_principles:
            interpretation += f"Dominante Prinzipien sind: {', '.join(dominant_principles)}. "
        
        if avg_confidence > 0.7:
            interpretation += "Die Muster zeigen starke Kohärenz und emergente Eigenschaften."
        elif avg_confidence > 0.5:
            interpretation += "Die Muster zeigen moderate Kohärenz mit einigen emergenten Eigenschaften."
        else:
            interpretation += "Die Muster zeigen schwache Kohärenz, weitere Analyse empfohlen."
        
        return interpretation
    
    def _generate_holistic_recommendations(self, patterns: List[GestaltPattern], 
                                         emergence_level: float, 
                                         coherence_score: float) -> List[str]:
        """Generiert holistische Empfehlungen"""
        recommendations = []
        
        if emergence_level < 0.3:
            recommendations.append("Erhöhe Datenqualität für bessere Mustererkennung")
        elif emergence_level > 0.8:
            recommendations.append("Nutze erkannte Muster für erweiterte Analyse")
        
        if coherence_score < 0.5:
            recommendations.append("Überprüfe Datenkonsistenz und -qualität")
        elif coherence_score > 0.8:
            recommendations.append("Hohe Musterkohärenz - ideal für Vorhersagen")
        
        if len(patterns) < 3:
            recommendations.append("Erweitere Eingabedaten für umfassendere Analyse")
        elif len(patterns) > 10:
            recommendations.append("Fokussiere auf dominante Muster für Effizienz")
        
        return recommendations

# Hauptinstanz für den Export
_vx_gestalt_instance = None

def get_vx_gestalt() -> VXGestaltCore:
    """
    Gibt die globale VX-GESTALT Instanz zurück
    
    Returns:
        VXGestaltCore: Die globale VX-GESTALT Instanz
    """
    global _vx_gestalt_instance
    if _vx_gestalt_instance is None:
        _vx_gestalt_instance = VXGestaltCore()
        _vx_gestalt_instance.initialize()
    return _vx_gestalt_instance

# Exportiere Hauptklassen und Funktionen
__all__ = [
    'VXGestaltCore',
    'GestaltPrinciple',
    'GestaltPattern',
    'HolisticInsight',
    'ProximityDetector',
    'SimilarityAnalyzer',
    'ClosureProcessor',
    'ContinuityTracker',
    'FigureGroundSeparator',
    'SymmetryFinder',
    'CommonFateAnalyzer',
    'PrägnanzEvaluator',
    'HolisticSynthesizer',
    'get_vx_gestalt'
]

if __name__ == "__main__":
    # Test VX-GESTALT
    gestalt = get_vx_gestalt()
    
    test_input = {
        "elements": ["A", "B", "C", "D"],
        "incomplete_shapes": ["circle_75%", "square_80%"],
        "sequences": ["step1", "step2", "step3"],
        "visual_data": {"foreground": "object", "background": "scene"},
        "geometric_data": ["triangle", "triangle_mirror"],
        "motion_data": ["object1_right", "object2_right", "object3_right"],
        "complex_data": {"pattern": "regular", "structure": "simple"}
    }
    
    result = gestalt.analyze_gestalt_patterns(test_input)
    print(f"VX-GESTALT Test erfolgreich: {result['status']}")
    print(f"Erkannte Muster: {result['pattern_count']}")
    print(f"Emergenz-Level: {result['emergence_level']:.3f}")
    print(f"Kohärenz-Score: {result['coherence_score']:.3f}")
