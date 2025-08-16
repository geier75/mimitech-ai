#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enhanced Paradox Detector für MISO Ultimate

Erweiterte Paradoxerkennung mit hierarchischer Klassifizierung
für komplexe, mehrstufige Paradoxien.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import numpy as np

# MISO-Module
from miso.math.t_mathematics.engine import TMathEngine

# Logger konfigurieren
logger = logging.getLogger("MISO.Logic.EnhancedParadoxDetector")

class ParadoxType(Enum):
    """Hierarchische Paradox-Klassifizierung"""
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    INFORMATION = "information"
    QUANTUM = "quantum"
    LOGICAL = "logical"
    ONTOLOGICAL = "ontological"

class ParadoxComplexity(Enum):
    """Paradox-Komplexitätsstufen"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4
    EXTREME = 5

class ResolutionStrategy(Enum):
    """Auflösungsstrategien"""
    TIMELINE_BRANCHING = "timeline_branching"
    CAUSAL_LOOP_BREAKING = "causal_loop_breaking"
    INFORMATION_ISOLATION = "information_isolation"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    LOGICAL_REFRAMING = "logical_reframing"
    TEMPORAL_QUARANTINE = "temporal_quarantine"

@dataclass
class ParadoxSignature:
    """Paradox-Signatur für Erkennung und Klassifizierung"""
    id: str
    type: ParadoxType
    complexity: ParadoxComplexity
    confidence: float
    causal_loops: List[Dict[str, Any]]
    temporal_inconsistencies: List[Dict[str, Any]]
    resolution_candidates: List[ResolutionStrategy]
    metadata: Dict[str, Any]

class EnhancedParadoxDetector:
    """Erweiterte Paradoxerkennung mit hierarchischer Klassifizierung"""
    
    def __init__(self, tmath_engine: Optional[TMathEngine] = None):
        self.tmath_engine = tmath_engine or TMathEngine()
        self.detection_threshold = 0.7
        self.complexity_weights = {
            ParadoxType.TEMPORAL: 1.2,
            ParadoxType.CAUSAL: 1.0,
            ParadoxType.INFORMATION: 1.1,
            ParadoxType.QUANTUM: 1.5,
            ParadoxType.LOGICAL: 0.8,
            ParadoxType.ONTOLOGICAL: 1.3
        }
        
        logger.info("Enhanced Paradox Detector initialisiert")
    
    def detect_paradox(self, timeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Hauptmethode für Paradoxerkennung"""
        try:
            events = timeline_data.get('events', [])
            if not events:
                return {'paradox_detected': False, 'reason': 'no_events'}
            
            # Multi-dimensionale Paradoxanalyse
            analyses = {
                'temporal': self._analyze_temporal_paradoxes(events),
                'causal': self._analyze_causal_paradoxes(events),
                'information': self._analyze_information_paradoxes(events),
                'quantum': self._analyze_quantum_paradoxes(events)
            }
            
            # Erstelle Paradox-Signaturen
            paradox_signatures = []
            
            for analysis_type, analysis in analyses.items():
                if analysis['detected']:
                    paradox_type = getattr(ParadoxType, analysis_type.upper())
                    signature = self._create_paradox_signature(paradox_type, analysis)
                    paradox_signatures.append(signature)
            
            # Bestimme dominantes Paradox
            if paradox_signatures:
                primary_paradox = max(paradox_signatures, key=lambda p: p.confidence)
                
                return {
                    'paradox_detected': True,
                    'paradox_type': primary_paradox.type.value,
                    'complexity': primary_paradox.complexity.value,
                    'confidence': primary_paradox.confidence,
                    'signatures': [self._signature_to_dict(sig) for sig in paradox_signatures],
                    'primary_signature': self._signature_to_dict(primary_paradox)
                }
            else:
                return {'paradox_detected': False, 'reason': 'no_paradoxes_found'}
                
        except Exception as e:
            logger.error(f"Fehler bei Paradoxerkennung: {e}")
            return {'paradox_detected': False, 'error': str(e)}
    
    def _analyze_temporal_paradoxes(self, events: List[Dict]) -> Dict[str, Any]:
        """Analysiere temporale Paradoxien"""
        temporal_issues = []
        sorted_events = sorted(events, key=lambda e: e.get('time', 0))
        
        # Suche nach Zeitschleifen
        for i, event in enumerate(sorted_events):
            for j, other_event in enumerate(sorted_events[i+1:], i+1):
                if self._is_temporal_loop(event, other_event):
                    temporal_issues.append({
                        'type': 'temporal_loop',
                        'events': [event, other_event],
                        'severity': self._calculate_loop_severity(event, other_event)
                    })
        
        # Suche nach Kausalitätsverletzungen
        causality_violations = self._detect_causality_violations(sorted_events)
        temporal_issues.extend(causality_violations)
        
        detected = len(temporal_issues) > 0
        confidence = min(1.0, len(temporal_issues) * 0.3) if detected else 0.0
        
        return {
            'detected': detected,
            'confidence': confidence,
            'issues': temporal_issues,
            'analysis_type': 'temporal'
        }
    
    def _analyze_causal_paradoxes(self, events: List[Dict]) -> Dict[str, Any]:
        """Analysiere kausale Paradoxien"""
        causal_issues = []
        
        # Baue Kausal-Graph
        causal_graph = self._build_causal_graph(events)
        
        # Suche nach kausalen Schleifen
        causal_loops = self._find_causal_loops(causal_graph)
        for loop in causal_loops:
            causal_issues.append({
                'type': 'causal_loop',
                'loop': loop,
                'severity': len(loop)
            })
        
        # Suche nach Großvater-Paradox-Mustern
        grandfather_patterns = self._detect_grandfather_patterns(events)
        causal_issues.extend(grandfather_patterns)
        
        detected = len(causal_issues) > 0
        confidence = min(1.0, len(causal_issues) * 0.4) if detected else 0.0
        
        return {
            'detected': detected,
            'confidence': confidence,
            'issues': causal_issues,
            'analysis_type': 'causal'
        }
    
    def _analyze_information_paradoxes(self, events: List[Dict]) -> Dict[str, Any]:
        """Analysiere Informationsparadoxien"""
        info_issues = []
        
        # Suche nach Bootstrap-Paradoxen
        bootstrap_patterns = self._detect_bootstrap_patterns(events)
        info_issues.extend(bootstrap_patterns)
        
        # Suche nach Informationsschleifen
        info_loops = self._detect_information_loops(events)
        info_issues.extend(info_loops)
        
        detected = len(info_issues) > 0
        confidence = min(1.0, len(info_issues) * 0.35) if detected else 0.0
        
        return {
            'detected': detected,
            'confidence': confidence,
            'issues': info_issues,
            'analysis_type': 'information'
        }
    
    def _analyze_quantum_paradoxes(self, events: List[Dict]) -> Dict[str, Any]:
        """Analysiere Quantenparadoxien"""
        quantum_issues = []
        
        # Suche nach Superpositions-Paradoxen
        superposition_events = [e for e in events if e.get('state') == 'superposition']
        if superposition_events:
            for event in superposition_events:
                if self._is_superposition_paradox(event, events):
                    quantum_issues.append({
                        'type': 'superposition_paradox',
                        'event': event,
                        'severity': 'high'
                    })
        
        # Suche nach Quantenverschränkungs-Paradoxen
        entanglement_issues = self._detect_entanglement_paradoxes(events)
        quantum_issues.extend(entanglement_issues)
        
        detected = len(quantum_issues) > 0
        confidence = min(1.0, len(quantum_issues) * 0.5) if detected else 0.0
        
        return {
            'detected': detected,
            'confidence': confidence,
            'issues': quantum_issues,
            'analysis_type': 'quantum'
        }
    
    def _create_paradox_signature(self, paradox_type: ParadoxType, analysis: Dict) -> ParadoxSignature:
        """Erstelle Paradox-Signatur"""
        complexity = self._determine_complexity(analysis)
        confidence = analysis.get('confidence', 0.0) * self.complexity_weights[paradox_type]
        
        return ParadoxSignature(
            id=str(uuid.uuid4()),
            type=paradox_type,
            complexity=complexity,
            confidence=confidence,
            causal_loops=[],
            temporal_inconsistencies=[],
            resolution_candidates=self._suggest_resolution_strategies(paradox_type, complexity),
            metadata={'analysis': analysis}
        )
    
    def _determine_complexity(self, analysis: Dict) -> ParadoxComplexity:
        """Bestimme Paradox-Komplexität"""
        issues_count = len(analysis.get('issues', []))
        confidence = analysis.get('confidence', 0.0)
        
        complexity_score = issues_count * confidence
        
        if complexity_score >= 2.0:
            return ParadoxComplexity.EXTREME
        elif complexity_score >= 1.5:
            return ParadoxComplexity.VERY_HIGH
        elif complexity_score >= 1.0:
            return ParadoxComplexity.HIGH
        elif complexity_score >= 0.5:
            return ParadoxComplexity.MEDIUM
        else:
            return ParadoxComplexity.LOW
    
    def _suggest_resolution_strategies(self, paradox_type: ParadoxType, complexity: ParadoxComplexity) -> List[ResolutionStrategy]:
        """Schlage Auflösungsstrategien vor"""
        strategies = []
        
        if paradox_type == ParadoxType.TEMPORAL:
            strategies.extend([ResolutionStrategy.TIMELINE_BRANCHING, ResolutionStrategy.TEMPORAL_QUARANTINE])
        elif paradox_type == ParadoxType.CAUSAL:
            strategies.extend([ResolutionStrategy.CAUSAL_LOOP_BREAKING, ResolutionStrategy.TIMELINE_BRANCHING])
        elif paradox_type == ParadoxType.INFORMATION:
            strategies.extend([ResolutionStrategy.INFORMATION_ISOLATION, ResolutionStrategy.LOGICAL_REFRAMING])
        elif paradox_type == ParadoxType.QUANTUM:
            strategies.extend([ResolutionStrategy.QUANTUM_DECOHERENCE])
        
        # Füge komplexitätsbasierte Strategien hinzu
        if complexity.value >= 4:
            strategies.append(ResolutionStrategy.TEMPORAL_QUARANTINE)
        
        return strategies
    
    def _signature_to_dict(self, signature: ParadoxSignature) -> Dict[str, Any]:
        """Konvertiere Paradox-Signatur zu Dictionary"""
        return {
            'id': signature.id,
            'type': signature.type.value,
            'complexity': signature.complexity.value,
            'confidence': signature.confidence,
            'resolution_candidates': [s.value for s in signature.resolution_candidates],
            'metadata': signature.metadata
        }
    
    # Hilfsmethoden für spezifische Paradox-Analysen
    def _is_temporal_loop(self, event1: Dict, event2: Dict) -> bool:
        """Prüfe auf temporale Schleife"""
        return (event1.get('action') == 'send_message' and 
                event2.get('action') == 'receive_message' and
                event1.get('time', 0) > event2.get('time', 0))
    
    def _calculate_loop_severity(self, event1: Dict, event2: Dict) -> str:
        """Berechne Schweregrad einer Schleife"""
        time_diff = abs(event1.get('time', 0) - event2.get('time', 0))
        return 'high' if time_diff > 50 else 'medium' if time_diff > 20 else 'low'
    
    def _detect_causality_violations(self, events: List[Dict]) -> List[Dict]:
        """Erkenne Kausalitätsverletzungen"""
        violations = []
        for i, event in enumerate(events):
            if event.get('action') == 'decide_to_send' and event.get('based_on'):
                for j, other_event in enumerate(events):
                    if other_event.get('action') == 'receive_message' and j < i:
                        violations.append({
                            'type': 'causality_violation',
                            'events': [event, other_event],
                            'severity': 'high'
                        })
        return violations
    
    def _build_causal_graph(self, events: List[Dict]) -> Dict[str, List[str]]:
        """Baue kausalen Graph"""
        graph = {}
        for event in events:
            event_id = event.get('action', str(uuid.uuid4()))
            graph[event_id] = []
            
            if event.get('based_on'):
                graph[event_id].append(event['based_on'])
            if event.get('target'):
                graph[event_id].append(event['target'])
        
        return graph
    
    def _find_causal_loops(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Finde kausale Schleifen"""
        loops = []
        visited = set()
        
        def dfs(node, path):
            if node in path:
                loop_start = path.index(node)
                loops.append(path[loop_start:])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path.copy())
        
        for node in graph:
            dfs(node, [])
        
        return loops
    
    def _detect_grandfather_patterns(self, events: List[Dict]) -> List[Dict]:
        """Erkenne Großvater-Paradox-Muster"""
        patterns = []
        
        time_travel_events = [e for e in events if e.get('action') == 'time_travel']
        prevent_events = [e for e in events if e.get('action') == 'prevent_meeting']
        
        if time_travel_events and prevent_events:
            patterns.append({
                'type': 'grandfather_pattern',
                'events': time_travel_events + prevent_events,
                'severity': 'extreme'
            })
        
        return patterns
    
    def _detect_bootstrap_patterns(self, events: List[Dict]) -> List[Dict]:
        """Erkenne Bootstrap-Paradox-Muster"""
        patterns = []
        
        receive_events = [e for e in events if e.get('action') == 'receive_formula']
        give_events = [e for e in events if e.get('action') == 'give_formula']
        
        if receive_events and give_events:
            patterns.append({
                'type': 'bootstrap_pattern',
                'events': receive_events + give_events,
                'severity': 'high'
            })
        
        return patterns
    
    def _detect_information_loops(self, events: List[Dict]) -> List[Dict]:
        """Erkenne Informationsschleifen"""
        loops = []
        
        info_events = [e for e in events if 'formula' in str(e.get('action', ''))]
        if len(info_events) >= 2:
            loops.append({
                'type': 'information_loop',
                'events': info_events,
                'severity': 'medium'
            })
        
        return loops
    
    def _is_superposition_paradox(self, event: Dict, all_events: List[Dict]) -> bool:
        """Prüfe auf Superpositions-Paradox"""
        same_time_events = [e for e in all_events if e.get('time') == event.get('time')]
        return len(same_time_events) > 2 and event.get('paradox', False)
    
    def _detect_entanglement_paradoxes(self, events: List[Dict]) -> List[Dict]:
        """Erkenne Verschränkungs-Paradoxien"""
        paradoxes = []
        
        entangled_events = [e for e in events if 'entangled' in str(e.get('action', '')).lower()]
        if len(entangled_events) >= 2:
            paradoxes.append({
                'type': 'entanglement_paradox',
                'events': entangled_events,
                'severity': 'high'
            })
        
        return paradoxes
